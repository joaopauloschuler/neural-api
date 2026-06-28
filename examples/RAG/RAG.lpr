program RAG;
(*
RAG: an end-to-end retrieval-augmented generation demo that ties together
the already-landed NLP pieces with NO new core library code.

The pipeline (all helpers live in neuralpretrained.pas / neuralhftokenizer.pas
/ neuralchat.pas / neuraldecode.pas):

  1. CHUNK a small built-in text corpus into retrievable passages.
  2. EMBED each chunk and the user QUESTION with the landed sentence-embedding
     path: BertTokenizeSentence -> BuildBertFromSafeTensors encoder ->
     BertPoolSentenceEmbedding (mean-pool over real tokens + L2 normalize),
     exactly as examples/SemanticSearch. One inference-only encoder is cached
     per distinct token length (the encoder carries no attention pad mask, so
     never padding keeps embeddings at sentence-transformers parity).
  3. RETRIEVE the top-k chunks by cosine similarity (= dot product, the
     embeddings are unit vectors).
  4. SPLICE the retrieved chunks into the prompt template
        Context:
        {chunks}

        Question: {q}
        Answer:
  5. GENERATE a grounded answer with an imported decoder LM through the
     chat-template + streaming-decode infra (BuildFromPretrained ->
     EncodeChat/ApplyChatTemplate -> full-recompute streamed decode, exactly
     as examples/ChatTerminal).

The headline RAG property: a question whose answer is NOT in the model's
weights ("What is Project Halcyon's launch date?" - an invented fact) is
answered correctly ONCE the chunk that states the fact is retrieved and
spliced into the context. Without retrieval the bare model can only guess;
with retrieval the answer is read off the supplied passage.

This is CPU / ulimit bounded. BOTH models are OPTIONAL command-line paths:

  RAG --embed-model <bert-dir> --gen-model <decoder-dir> [options]

  --embed-model DIR  HF sentence encoder dir (model.safetensors + config.json
                     + tokenizer.json), e.g. a downloaded
                     sentence-transformers/all-MiniLM-L6-v2 snapshot. When
                     OMITTED the retrieval half falls back to a deterministic
                     built-in hashing-bag-of-words embedder so the demo still
                     ranks and retrieves with NO download (lower quality, but
                     it shows the wiring end to end).
  --gen-model DIR    HF decoder LM dir for the generation half. When OMITTED
                     the program prints the spliced prompt it WOULD send and
                     stops (retrieval + prompt assembly need no decoder).

  --selftest         run the offline unit checks (chunking / cosine top-k /
                     prompt splicing / chat assembly) and exit. Needs NO model
                     files and runs well within ulimit -v 3000000.

Run with no model arguments to see the retrieval + prompt-splice half end to
end using the built-in fallback embedder; add --embed-model for real sentence
embeddings and --gen-model to actually generate the grounded answer. See
examples/RAG/README.md for the download commands.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork, neuralpretrained, neuralhftokenizer,
  neuralchat, neuraldecode;

const
  // Embedding width of the built-in fallback embedder (no model). Chosen to
  // be small but enough to separate the corpus chunks by topic.
  csFallbackDim = 256;
  csDefaultTopK = 2;
  csDefaultMaxNewTokens = 48;

  // A tiny knowledge base. Chunk #5 carries the INVENTED fact the demo turns
  // on (no pretrained model can know it). The default question targets it so
  // the headline RAG property is visible: retrieval supplies what the weights
  // lack.
  csCorpusSize = 8;
  csCorpus: array[0..csCorpusSize - 1] of string = (
    'The company cafeteria serves lunch between 11:30 and 14:00 on weekdays.',
    'Photosynthesis converts sunlight, water and carbon dioxide into glucose and oxygen.',
    'The Eiffel Tower was completed in 1889 for the World''s Fair in Paris.',
    'Water boils at 100 degrees Celsius at standard atmospheric pressure.',
    'Our internal wiki is hosted on the intranet under the Engineering space.',
    'Project Halcyon is scheduled to launch on the 14th of November, 2027.',
    'The mitochondrion is often called the powerhouse of the cell.',
    'Employees can request vacation through the HR portal up to a year in advance.');

  csDefaultQuestion = 'What is the launch date of Project Halcyon?';

type
  TRagOptions = record
    EmbedModelDir: string;
    GenModelDir: string;
    Question: string;
    TopK: integer;
    MaxNewTokens: integer;
    CtxLen: integer;
    Int8: boolean;
    FormatName: string;
    SelfTest: boolean;
    ShowHelp: boolean;
    ErrorMsg: string;
  end;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------
function DefaultRagOptions(): TRagOptions;
begin
  Result.EmbedModelDir := '';
  Result.GenModelDir := '';
  Result.Question := csDefaultQuestion;
  Result.TopK := csDefaultTopK;
  Result.MaxNewTokens := csDefaultMaxNewTokens;
  Result.CtxLen := 1024;
  Result.Int8 := true;
  Result.FormatName := '';
  Result.SelfTest := false;
  Result.ShowHelp := false;
  Result.ErrorMsg := '';
end;

procedure PrintUsage();
begin
  WriteLn('Usage: RAG [--embed-model DIR] [--gen-model DIR] [options]');
  WriteLn;
  WriteLn('Retrieval-augmented generation over a built-in corpus.');
  WriteLn;
  WriteLn('Options:');
  WriteLn('  --embed-model DIR   HF sentence-encoder dir (model.safetensors +');
  WriteLn('                      config.json + tokenizer.json). Omit to use the');
  WriteLn('                      built-in fallback embedder (no download).');
  WriteLn('  --gen-model DIR     HF decoder LM dir. Omit to print the spliced');
  WriteLn('                      prompt and stop (retrieval needs no decoder).');
  WriteLn('  --question "text"   the question to answer (default: the Project');
  WriteLn('                      Halcyon launch-date probe).');
  WriteLn('  --top-k N           chunks to retrieve and splice (default ', csDefaultTopK, ').');
  WriteLn('  --max-new-tokens N  answer length cap (default ', csDefaultMaxNewTokens, ').');
  WriteLn('  --ctx N             decoder context window (default 1024).');
  WriteLn('  --fp32              full-precision decoder weights (default int8).');
  WriteLn('  --format NAME       chat format override (chatml|llama3|gemma|...).');
  WriteLn('  --selftest          run offline unit checks and exit (no model).');
  WriteLn('  --help              this text.');
end;

function ParseArgs(Args: TStringList; var Opt: TRagOptions): boolean;
var
  ArgPos: integer;

  function NextValue(const FlagName: string; out Value: string): boolean;
  begin
    if ArgPos + 1 >= Args.Count then
    begin
      Opt.ErrorMsg := FlagName + ' needs a value';
      exit(false);
    end;
    Inc(ArgPos);
    Value := Args[ArgPos];
    Result := true;
  end;

  function NextInt(const FlagName: string; out Value: integer): boolean;
  var
    S: string;
  begin
    Result := NextValue(FlagName, S);
    if not Result then exit;
    if not TryStrToInt(S, Value) then
    begin
      Opt.ErrorMsg := FlagName + ': not an integer: ' + S;
      Result := false;
    end;
  end;

var
  Arg, SVal: string;
  IVal: integer;
begin
  Opt := DefaultRagOptions();
  ArgPos := 0;
  while ArgPos < Args.Count do
  begin
    Arg := Args[ArgPos];
    if Arg = '--fp32' then Opt.Int8 := false
    else if Arg = '--int8' then Opt.Int8 := true
    else if Arg = '--selftest' then Opt.SelfTest := true
    else if (Arg = '--help') or (Arg = '-h') then Opt.ShowHelp := true
    else if Arg = '--embed-model' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.EmbedModelDir := SVal;
    end
    else if Arg = '--gen-model' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.GenModelDir := SVal;
    end
    else if Arg = '--question' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.Question := SVal;
    end
    else if Arg = '--format' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.FormatName := SVal;
    end
    else if Arg = '--top-k' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.TopK := IVal;
    end
    else if Arg = '--max-new-tokens' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.MaxNewTokens := IVal;
    end
    else if Arg = '--ctx' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.CtxLen := IVal;
    end
    else if (Length(Arg) >= 2) and (Copy(Arg, 1, 2) = '--') then
    begin
      Opt.ErrorMsg := 'unknown flag: ' + Arg;
      exit(false);
    end
    else
    begin
      Opt.ErrorMsg := 'unexpected argument: ' + Arg;
      exit(false);
    end;
    Inc(ArgPos);
  end;
  if Opt.TopK < 1 then Opt.TopK := 1;
  if Opt.TopK > csCorpusSize then Opt.TopK := csCorpusSize;
  Result := true;
end;

// ---------------------------------------------------------------------------
// Embedding: real BERT path (cached one net per token length), with a
// deterministic hashing-bag-of-words fallback when no encoder is supplied.
// ---------------------------------------------------------------------------
var
  EmbWeightsPath: string;
  EmbTokenizer: TNeuralHFTokenizer;
  UseRealEmbedder: boolean;
  NetLens: array of integer;
  Nets: array of TNNet;

function NetForLength(TokenCount: integer): TNNet;
var
  Cnt: integer;
begin
  for Cnt := 0 to High(NetLens) do
    if NetLens[Cnt] = TokenCount then Exit(Nets[Cnt]);
  WriteLn('  [building encoder for token length ', TokenCount, ' ...]');
  SetLength(NetLens, Length(NetLens) + 1);
  SetLength(Nets, Length(Nets) + 1);
  NetLens[High(NetLens)] := TokenCount;
  Nets[High(Nets)] := BuildBertFromSafeTensors(EmbWeightsPath, TokenCount,
    {pTrainable=}false);
  Result := Nets[High(Nets)];
end;

// Deterministic fallback: lowercase, split on non-letters, hash each word
// into a bucket, accumulate, L2-normalize. Cosine of two such bags is a
// crude lexical-overlap similarity - enough to wire the demo with no model.
procedure FallbackEmbed(const Text: string; Embedding: TNNetVolume);
var
  Cnt, Bucket, WordHash: integer;
  Ch: Char;
  Word: string;
  Norm: TNeuralFloat;

  procedure FlushWord();
  var
    K: integer;
    H: LongWord;
  begin
    if Word = '' then exit;
    H := LongWord(2166136261); // FNV-1a 32-bit offset basis
    for K := 1 to Length(Word) do
      H := (H xor LongWord(Ord(Word[K]))) * LongWord(16777619);
    WordHash := integer(H and $7FFFFFFF);
    Bucket := WordHash mod csFallbackDim;
    Embedding.FData[Bucket] := Embedding.FData[Bucket] + 1;
    Word := '';
  end;

begin
  Embedding.ReSize(1, 1, csFallbackDim);
  Embedding.Fill(0);
  Word := '';
  for Cnt := 1 to Length(Text) do
  begin
    Ch := Text[Cnt];
    if (Ch >= 'A') and (Ch <= 'Z') then Ch := Chr(Ord(Ch) + 32);
    if (Ch >= 'a') and (Ch <= 'z') then Word := Word + Ch
    else FlushWord();
  end;
  FlushWord();
  Norm := 0;
  for Cnt := 0 to csFallbackDim - 1 do
    Norm := Norm + Embedding.FData[Cnt] * Embedding.FData[Cnt];
  Norm := Sqrt(Norm);
  if Norm > 0 then Embedding.Mul(1 / Norm);
end;

procedure EmbedText(const Text: string; Embedding: TNNetVolume);
var
  TokenIds: TNeuralIntegerArray;
begin
  if UseRealEmbedder then
  begin
    TokenIds := BertTokenizeSentence(EmbTokenizer, Text);
    BertEncodeSentence(NetForLength(Length(TokenIds)), EmbTokenizer, Text,
      Embedding);
  end
  else
    FallbackEmbed(Text, Embedding);
end;

function CosineDot(A, B: TNNetVolume): TNeuralFloat;
var
  Cnt: integer;
begin
  Result := 0;
  for Cnt := 0 to A.Size - 1 do Result := Result + A.FData[Cnt] * B.FData[Cnt];
end;

// ---------------------------------------------------------------------------
// Retrieval: rank corpus by cosine to the query, return the descending order.
// ---------------------------------------------------------------------------
procedure RankByCosine(QueryEmb: TNNetVolume; ChunkEmb: array of TNNetVolume;
  var Sims: array of TNeuralFloat; var Order: array of integer);
var
  Cnt, Inner, BestIdx, TmpI: integer;
  BestSim: TNeuralFloat;
  N: integer;
begin
  N := Length(ChunkEmb);
  for Cnt := 0 to N - 1 do
  begin
    Sims[Cnt] := CosineDot(ChunkEmb[Cnt], QueryEmb);
    Order[Cnt] := Cnt;
  end;
  for Cnt := 0 to N - 2 do
  begin
    BestIdx := Cnt;
    BestSim := Sims[Order[Cnt]];
    for Inner := Cnt + 1 to N - 1 do
      if Sims[Order[Inner]] > BestSim then
      begin
        BestIdx := Inner;
        BestSim := Sims[Order[Inner]];
      end;
    TmpI := Order[Cnt];
    Order[Cnt] := Order[BestIdx];
    Order[BestIdx] := TmpI;
  end;
end;

// ---------------------------------------------------------------------------
// Prompt splicing: the canonical RAG template.
// ---------------------------------------------------------------------------
function BuildRagPrompt(const Chunks: array of string;
  const Question: string): string;
var
  Joined: string;
  Cnt: integer;
begin
  Joined := '';
  for Cnt := 0 to High(Chunks) do
  begin
    Joined := Joined + '- ' + Chunks[Cnt];
    if Cnt < High(Chunks) then Joined := Joined + #10;
  end;
  Result := 'Context:' + #10 + Joined + #10 + #10 +
    'Question: ' + Question + #10 + 'Answer:';
end;

// ---------------------------------------------------------------------------
// Generation: one grounded answer streamed to stdout (the ChatTerminal
// full-recompute decode, greedy argmax).
// ---------------------------------------------------------------------------
// Stable in-place softmax of a logits row is now neuralvolume.RowSoftMax.

function ArgMaxRow(Row: TNNetVolume): integer;
var
  Cnt: integer;
begin
  Result := 0;
  for Cnt := 1 to Row.Size - 1 do
    if Row.FData[Cnt] > Row.FData[Result] then Result := Cnt;
end;

function GenerateAnswer(NN: TNNet; Tokenizer: TNeuralHFTokenizer;
  const PromptIds: TNeuralIntegerArray; MaxNewTokens, SeqLen,
  VocabSize: integer): string;
var
  Tokens, Generated: TNeuralIntegerArray;
  Input, Output, Row: TNNetVolume;
  Len, GenLen, StepCnt, Cnt, NewToken: integer;
  Decoded, Printed: string;
begin
  Result := '';
  Len := Length(PromptIds);
  if Len >= SeqLen then
  begin
    WriteLn('[prompt (', Len, ' tokens) does not fit context ', SeqLen,
      ' - raise --ctx or lower --top-k]');
    exit;
  end;
  SetLength(Tokens, SeqLen);
  for Cnt := 0 to Len - 1 do Tokens[Cnt] := PromptIds[Cnt];
  SetLength(Generated, 0);
  Printed := '';
  Input := TNNetVolume.Create(SeqLen, 1, 1);
  Output := TNNetVolume.Create();
  Row := TNNetVolume.Create(VocabSize, 1, 1);
  try
    for StepCnt := 1 to MaxNewTokens do
    begin
      if Len >= SeqLen then break;
      Input.Fill(0);
      for Cnt := 0 to Len - 1 do Input.FData[Cnt] := Tokens[Cnt];
      NN.Compute(Input);
      NN.GetOutput(Output);
      for Cnt := 0 to VocabSize - 1 do
        Row.FData[Cnt] := Output.FData[(Len - 1) * VocabSize + Cnt];
      RowSoftMax(Row);
      NewToken := ArgMaxRow(Row);
      Tokens[Len] := NewToken;
      Inc(Len);
      GenLen := Length(Generated);
      SetLength(Generated, GenLen + 1);
      Generated[GenLen] := NewToken;
      if (Tokenizer.EosId >= 0) and (NewToken = Tokenizer.EosId) then break;
      Decoded := Tokenizer.Decode(Generated, {SkipSpecialTokens=}true);
      if (Length(Decoded) > Length(Printed)) and
        (Copy(Decoded, 1, Length(Printed)) = Printed) then
      begin
        Write(Copy(Decoded, Length(Printed) + 1,
          Length(Decoded) - Length(Printed)));
        Flush(System.Output);
        Printed := Decoded;
      end;
    end;
    Result := Tokenizer.Decode(Generated, {SkipSpecialTokens=}true);
    if (Length(Result) > Length(Printed)) and
      (Copy(Result, 1, Length(Printed)) = Printed) then
      Write(Copy(Result, Length(Printed) + 1, Length(Result) - Length(Printed)));
    WriteLn;
    Flush(System.Output);
  finally
    Row.Free;
    Output.Free;
    Input.Free;
  end;
end;

// ---------------------------------------------------------------------------
// --selftest: offline unit checks (no model needed).
// ---------------------------------------------------------------------------
procedure SelfTest();
var
  Failures: integer;

  procedure Check(Condition: boolean; const What: string);
  begin
    if Condition then WriteLn('PASS: ', What)
    else
    begin
      WriteLn('FAIL: ', What);
      Inc(Failures);
    end;
  end;

var
  Args: TStringList;
  Opt: TRagOptions;
  ChunkEmb: array of TNNetVolume;
  QEmb: TNNetVolume;
  Sims: array of TNeuralFloat;
  Order: array of integer;
  Cnt, HalcyonIdx: integer;
  Prompt: string;
  Msgs: TChatMessages;
  Rendered: string;
begin
  Failures := 0;
  Args := TStringList.Create();
  try
    // Argument parsing.
    Args.Add('--embed-model'); Args.Add('/tmp/bert');
    Args.Add('--gen-model'); Args.Add('/tmp/gen');
    Args.Add('--question'); Args.Add('Q?');
    Args.Add('--top-k'); Args.Add('3');
    Args.Add('--max-new-tokens'); Args.Add('16');
    Args.Add('--fp32');
    Check(ParseArgs(Args, Opt), 'full flag set parses');
    Check(Opt.EmbedModelDir = '/tmp/bert', '--embed-model');
    Check(Opt.GenModelDir = '/tmp/gen', '--gen-model');
    Check(Opt.Question = 'Q?', '--question');
    Check(Opt.TopK = 3, '--top-k');
    Check(Opt.MaxNewTokens = 16, '--max-new-tokens');
    Check(not Opt.Int8, '--fp32 disables int8');

    Args.Clear;
    Args.Add('--bogus');
    Check(not ParseArgs(Args, Opt), 'unknown flag rejected');
    Args.Clear;
    Args.Add('--top-k'); Args.Add('99');
    Check(ParseArgs(Args, Opt) and (Opt.TopK = csCorpusSize),
      'top-k clamped to corpus size');

    // Fallback embedder + cosine retrieval: the Halcyon question must rank
    // the Halcyon chunk (#5) first. This is the headline retrieval property,
    // verified WITHOUT any model.
    UseRealEmbedder := false;
    HalcyonIdx := 5;
    SetLength(ChunkEmb, csCorpusSize);
    SetLength(Sims, csCorpusSize);
    SetLength(Order, csCorpusSize);
    for Cnt := 0 to csCorpusSize - 1 do
    begin
      ChunkEmb[Cnt] := TNNetVolume.Create();
      EmbedText(csCorpus[Cnt], ChunkEmb[Cnt]);
    end;
    QEmb := TNNetVolume.Create();
    EmbedText(csDefaultQuestion, QEmb);
    RankByCosine(QEmb, ChunkEmb, Sims, Order);
    Check(Order[0] = HalcyonIdx,
      'fallback retrieval ranks the Halcyon chunk #1 for the Halcyon question');
    Check(Sims[HalcyonIdx] > 0,
      'Halcyon chunk has positive cosine to the Halcyon question');

    // Self-similarity is 1 for an L2-normalized embedding.
    Check(Abs(CosineDot(ChunkEmb[0], ChunkEmb[0]) - 1.0) < 1e-4,
      'L2-normalized embedding has unit self-cosine');

    // Prompt splicing: the canonical RAG template shape.
    Prompt := BuildRagPrompt(['Chunk A.', 'Chunk B.'], 'What?');
    Check(Prompt =
      'Context:' + #10 + '- Chunk A.' + #10 + '- Chunk B.' + #10 + #10 +
      'Question: What?' + #10 + 'Answer:',
      'RAG prompt template matches Context/Question/Answer shape');
    Check(Pos(csCorpus[HalcyonIdx],
      BuildRagPrompt([csCorpus[HalcyonIdx]], csDefaultQuestion)) > 0,
      'retrieved Halcyon fact is present in the spliced prompt');

    // Chat assembly: the RAG prompt rides as a single user turn through the
    // chat-template engine (ApplyChatTemplate).
    SetLength(Msgs, 1);
    Msgs[0] := ChatMessage('user', Prompt);
    Rendered := ApplyChatTemplate(cfChatML, Msgs, true);
    Check(Pos('Question: What?', Rendered) > 0,
      'spliced prompt survives ChatML rendering');
    Check(Pos('<|im_start|>assistant', Rendered) > 0,
      'generation prompt appended by ApplyChatTemplate');

    for Cnt := 0 to csCorpusSize - 1 do ChunkEmb[Cnt].Free;
    QEmb.Free;
  finally
    Args.Free;
  end;
  if Failures = 0 then WriteLn('SELFTEST OK')
  else
  begin
    WriteLn('SELFTEST FAILED: ', Failures, ' check(s)');
    Halt(1);
  end;
end;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
var
  Opt: TRagOptions;
  Args: TStringList;
  ChunkEmb: array of TNNetVolume;
  QEmb: TNNetVolume;
  Sims: array of TNeuralFloat;
  Order: array of integer;
  Retrieved: array of string;
  Prompt: string;
  Cnt: integer;
  GenNN: TNNet;
  GenTokenizer: TNeuralHFTokenizer;
  ChatFormat: TNeuralChatFormat;
  Msgs: TChatMessages;
  PromptIds: TNeuralIntegerArray;
  SeqLen, VocabSize: integer;
  TokenizerFile, TokenizerConfigFile: string;
begin
  DefaultFormatSettings.DecimalSeparator := '.';
  Args := TStringList.Create();
  for Cnt := 1 to ParamCount do Args.Add(ParamStr(Cnt));
  if not ParseArgs(Args, Opt) then
  begin
    WriteLn('Error: ', Opt.ErrorMsg);
    WriteLn;
    PrintUsage();
    Args.Free;
    Halt(1);
  end;
  Args.Free;

  if Opt.SelfTest then
  begin
    SelfTest();
    Halt(0);
  end;
  if Opt.ShowHelp then
  begin
    PrintUsage();
    Halt(0);
  end;

  // --- Embedder setup ---
  UseRealEmbedder := Opt.EmbedModelDir <> '';
  if UseRealEmbedder then
  begin
    EmbWeightsPath := IncludeTrailingPathDelimiter(Opt.EmbedModelDir) +
      'model.safetensors';
    EmbTokenizer := TNeuralHFTokenizer.Create();
    EmbTokenizer.LoadFromFile(
      IncludeTrailingPathDelimiter(Opt.EmbedModelDir) + 'tokenizer.json');
    WriteLn('Embedder: ', Opt.EmbedModelDir, ' (vocab ',
      EmbTokenizer.GetVocabSize(), ')');
  end
  else
    WriteLn('Embedder: built-in fallback (hashing bag-of-words, ',
      csFallbackDim, ' dims) - pass --embed-model for real sentence ',
      'embeddings.');

  // --- 1+2: chunk + embed the corpus and the question ---
  WriteLn;
  WriteLn('Corpus: ', csCorpusSize, ' chunks. Embedding ...');
  SetLength(ChunkEmb, csCorpusSize);
  for Cnt := 0 to csCorpusSize - 1 do
  begin
    ChunkEmb[Cnt] := TNNetVolume.Create();
    EmbedText(csCorpus[Cnt], ChunkEmb[Cnt]);
  end;
  QEmb := TNNetVolume.Create();
  EmbedText(Opt.Question, QEmb);

  // --- 3: retrieve top-k by cosine ---
  SetLength(Sims, csCorpusSize);
  SetLength(Order, csCorpusSize);
  RankByCosine(QEmb, ChunkEmb, Sims, Order);

  WriteLn;
  WriteLn('Question: ', Opt.Question);
  WriteLn('Ranked chunks (cosine similarity):');
  for Cnt := 0 to csCorpusSize - 1 do
    WriteLn(Format('  %2d. %7.4f  %s',
      [Cnt + 1, Sims[Order[Cnt]], csCorpus[Order[Cnt]]]));

  SetLength(Retrieved, Opt.TopK);
  WriteLn;
  WriteLn('Retrieved top-', Opt.TopK, ':');
  for Cnt := 0 to Opt.TopK - 1 do
  begin
    Retrieved[Cnt] := csCorpus[Order[Cnt]];
    WriteLn('  * ', Retrieved[Cnt]);
  end;

  // --- 4: splice into the RAG prompt template ---
  Prompt := BuildRagPrompt(Retrieved, Opt.Question);
  WriteLn;
  WriteLn('--- spliced prompt ---');
  WriteLn(Prompt);
  WriteLn('----------------------');

  // --- 5: generate the grounded answer (only if a decoder is supplied) ---
  if Opt.GenModelDir = '' then
  begin
    WriteLn;
    WriteLn('[no --gen-model: retrieval + prompt assembly complete. Pass a');
    WriteLn(' decoder dir with --gen-model to generate the grounded answer.]');
    WriteLn('[Headline RAG property: the Project Halcyon launch date is NOT in');
    WriteLn(' any pretrained model''s weights - it is supplied above by the');
    WriteLn(' retrieved chunk, so a grounded decoder can now read it off.]');
  end
  else
  begin
    TokenizerFile := IncludeTrailingPathDelimiter(Opt.GenModelDir) +
      'tokenizer.json';
    if not FileExists(TokenizerFile) then
    begin
      WriteLn('No tokenizer.json in ', Opt.GenModelDir);
      Halt(1);
    end;
    GenTokenizer := TNeuralHFTokenizer.Create();
    GenTokenizer.LoadFromFile(TokenizerFile);

    ChatFormat := cfUnknown;
    if Opt.FormatName <> '' then ChatFormat := ChatFormatFromName(Opt.FormatName)
    else
    begin
      TokenizerConfigFile := IncludeTrailingPathDelimiter(Opt.GenModelDir) +
        'tokenizer_config.json';
      if FileExists(TokenizerConfigFile) then
        ChatFormat := DetectChatFormatFromConfigFile(TokenizerConfigFile);
    end;
    if ChatFormat = cfUnknown then
    begin
      WriteLn('[no chat template detected - defaulting to ChatML]');
      ChatFormat := cfChatML;
    end;

    WriteLn;
    WriteLn('Loading decoder ', Opt.GenModelDir, ' (',
      BoolToStr(Opt.Int8, 'int8', 'fp32'), ') ...');
    GenNN := BuildFromPretrained(Opt.GenModelDir, Opt.CtxLen,
      {pTrainable=}false, '', {pQuantizeInt8=}Opt.Int8);
    SeqLen := GenNN.GetFirstLayer().Output.SizeX;
    VocabSize := GenNN.GetLastLayer().Output.Depth;
    WriteLn('Decoder ready: vocab ', VocabSize, ', context ', SeqLen,
      ', chat format ', ChatFormatName(ChatFormat), '.');

    SetLength(Msgs, 1);
    Msgs[0] := ChatMessage('user', Prompt);
    PromptIds := EncodeChat(GenTokenizer, ChatFormat, Msgs,
      {AddGenerationPrompt=}true);

    WriteLn;
    Write('Grounded answer: ');
    Flush(System.Output);
    GenerateAnswer(GenNN, GenTokenizer, PromptIds, Opt.MaxNewTokens, SeqLen,
      VocabSize);

    GenTokenizer.Free;
    GenNN.Free;
  end;

  for Cnt := 0 to csCorpusSize - 1 do ChunkEmb[Cnt].Free;
  QEmb.Free;
  for Cnt := 0 to High(Nets) do Nets[Cnt].Free;
  if UseRealEmbedder then EmbTokenizer.Free;
end.
