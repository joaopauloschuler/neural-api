program SemanticSearch;
(*
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

// SemanticSearch -- sentence embeddings + semantic search with an imported
// sentence-transformers checkpoint (all-MiniLM-L6-v2 or any HF BertModel).
//
// Pipeline per sentence (see neuralpretrained.pas SENTENCE EMBEDDINGS):
//   tokenizer.json WordPiece encode -> [CLS] ids [SEP]
//   -> BuildBertFromSafeTensors encoder -> (SeqLen,1,hidden) hidden states
//   -> mean pooling over the REAL tokens -> L2 normalize.
// The result is a unit vector whose dot product with another sentence's
// vector is their cosine similarity - the sentence-transformers recipe.
//
// The imported encoder has NO attention padding mask, so the program never
// pads: it builds one inference-only net per DISTINCT token length in the
// corpus (the per-length nets are cached). That keeps every embedding
// exactly equal to sentence-transformers' (cosine > 0.999, verified by
// compare_st_embeddings.py).
//
// Usage:
//   SemanticSearch <modeldir> [-q "query text"] [-dump out.json]
//
//   modeldir  - directory with model.safetensors + config.json +
//               tokenizer.json (e.g. a downloaded
//               sentence-transformers/all-MiniLM-L6-v2 snapshot).
//   -q text   - query to rank the built-in corpus against (default below
//               is a paraphrase of corpus sentence #1).
//   -dump f   - also write corpus+query embeddings to f as JSON, for the
//               HF parity check (compare_st_embeddings.py).

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork, neuralpretrained, neuralhftokenizer;

const
  csCorpusSize = 8;
  csCorpus: array[0..csCorpusSize - 1] of string = (
    'A cat is sitting on the windowsill in the sun.',
    'The weather forecast says it will rain tomorrow.',
    'She is cooking pasta for dinner tonight.',
    'The stock market dropped sharply this morning.',
    'A dog is sleeping on the porch.',
    'Scientists discovered a new species of frog in the rainforest.',
    'He fixed the leaking kitchen faucet himself.',
    'The orchestra performed a beautiful symphony last night.');
  csDefaultQuery = 'A kitty rests on the window ledge enjoying the sunshine.';

var
  ModelDir, WeightsPath, Query, DumpPath: string;
  Tokenizer: TNeuralHFTokenizer;
  // one inference-only net per DISTINCT token length (no padding => exact
  // sentence-transformers parity; the encoder has no attention mask)
  NetLens: array of integer;
  Nets: array of TNNet;
  Embeddings: array of TNNetVolume;
  Sentences: array of string;
  ParamPos, SentCnt, NetCnt, ChanCnt: integer;
  BestIdx: integer;
  Sim, BestSim: TNeuralFloat;
  Sims: array of TNeuralFloat;
  Order: array of integer;
  TmpI: integer;

function NetForLength(TokenCount: integer): TNNet;
var
  Cnt: integer;
begin
  for Cnt := 0 to High(NetLens) do
    if NetLens[Cnt] = TokenCount then Exit(Nets[Cnt]);
  WriteLn('Building encoder for token length ', TokenCount, ' ...');
  SetLength(NetLens, Length(NetLens) + 1);
  SetLength(Nets, Length(Nets) + 1);
  NetLens[High(NetLens)] := TokenCount;
  Nets[High(Nets)] := BuildBertFromSafeTensors(WeightsPath, TokenCount,
    {pInferenceOnly=}true);
  Result := Nets[High(Nets)];
end;

procedure EmbedSentence(const Text: string; Embedding: TNNetVolume);
var
  TokenIds: TNeuralIntegerArray;
begin
  TokenIds := BertTokenizeSentence(Tokenizer, Text);
  BertEncodeSentence(NetForLength(Length(TokenIds)), Tokenizer, Text,
    Embedding);
end;

function JSONEscape(const S: string): string;
var
  Cnt: integer;
begin
  Result := '"';
  for Cnt := 1 to Length(S) do
    case S[Cnt] of
      '"': Result := Result + '\"';
      '\': Result := Result + '\\';
      #0..#31: Result := Result + '\u' + IntToHex(Ord(S[Cnt]), 4);
      else Result := Result + S[Cnt];
    end;
  Result := Result + '"';
end;

procedure DumpEmbeddingsJSON(const FileName: string);
var
  SL: TStringList;
  Line: string;
  SentIdx, ChanIdx: integer;
begin
  SL := TStringList.Create();
  try
    SL.Add('{"sentences": [');
    for SentIdx := 0 to High(Sentences) do
    begin
      Line := '  ' + JSONEscape(Sentences[SentIdx]);
      if SentIdx < High(Sentences) then Line := Line + ',';
      SL.Add(Line);
    end;
    SL.Add('], "embeddings": [');
    for SentIdx := 0 to High(Embeddings) do
    begin
      Line := '  [';
      for ChanIdx := 0 to Embeddings[SentIdx].Size - 1 do
      begin
        Line := Line + FloatToStrF(Embeddings[SentIdx].FData[ChanIdx],
          ffGeneral, 9, 0);
        if ChanIdx < Embeddings[SentIdx].Size - 1 then Line := Line + ',';
      end;
      Line := Line + ']';
      if SentIdx < High(Embeddings) then Line := Line + ',';
      SL.Add(Line);
    end;
    SL.Add(']}');
    SL.SaveToFile(FileName);
    WriteLn('Embeddings written to ', FileName);
  finally
    SL.Free;
  end;
end;

begin
  DefaultFormatSettings.DecimalSeparator := '.'; // JSON dump correctness
  if ParamCount < 1 then
  begin
    WriteLn('Usage: SemanticSearch <modeldir> [-q "query"] [-dump out.json]');
    WriteLn('modeldir needs model.safetensors + config.json + tokenizer.json');
    WriteLn('See examples/SemanticSearch/README.md.');
    Halt(1);
  end;
  ModelDir := ParamStr(1);
  WeightsPath := IncludeTrailingPathDelimiter(ModelDir) + 'model.safetensors';
  Query := csDefaultQuery;
  DumpPath := '';
  ParamPos := 2;
  while ParamPos <= ParamCount do
  begin
    if (ParamStr(ParamPos) = '-q') and (ParamPos < ParamCount) then
    begin
      Inc(ParamPos);
      Query := ParamStr(ParamPos);
    end
    else if (ParamStr(ParamPos) = '-dump') and (ParamPos < ParamCount) then
    begin
      Inc(ParamPos);
      DumpPath := ParamStr(ParamPos);
    end
    else
    begin
      WriteLn('Unknown argument: ', ParamStr(ParamPos));
      Halt(1);
    end;
    Inc(ParamPos);
  end;

  Tokenizer := TNeuralHFTokenizer.Create();
  Tokenizer.LoadFromFile(
    IncludeTrailingPathDelimiter(ModelDir) + 'tokenizer.json');
  WriteLn('Tokenizer loaded: vocab ', Tokenizer.GetVocabSize());

  // corpus first, query last
  SetLength(Sentences, csCorpusSize + 1);
  for SentCnt := 0 to csCorpusSize - 1 do Sentences[SentCnt] := csCorpus[SentCnt];
  Sentences[csCorpusSize] := Query;

  SetLength(Embeddings, Length(Sentences));
  for SentCnt := 0 to High(Sentences) do
  begin
    Embeddings[SentCnt] := TNNetVolume.Create();
    EmbedSentence(Sentences[SentCnt], Embeddings[SentCnt]);
    WriteLn('Embedded (', Embeddings[SentCnt].Size, ' dims): ',
      Sentences[SentCnt]);
  end;

  // cosine = dot product (embeddings are L2-normalized)
  WriteLn;
  WriteLn('Query: ', Query);
  WriteLn('Ranking (cosine similarity):');
  SetLength(Sims, csCorpusSize);
  SetLength(Order, csCorpusSize);
  for SentCnt := 0 to csCorpusSize - 1 do
  begin
    Sim := 0;
    for ChanCnt := 0 to Embeddings[SentCnt].Size - 1 do
      Sim := Sim + Embeddings[SentCnt].FData[ChanCnt] *
        Embeddings[csCorpusSize].FData[ChanCnt];
    Sims[SentCnt] := Sim;
    Order[SentCnt] := SentCnt;
  end;
  // selection sort, descending
  for SentCnt := 0 to csCorpusSize - 2 do
  begin
    BestIdx := SentCnt;
    BestSim := Sims[Order[SentCnt]];
    for NetCnt := SentCnt + 1 to csCorpusSize - 1 do
      if Sims[Order[NetCnt]] > BestSim then
      begin
        BestIdx := NetCnt;
        BestSim := Sims[Order[NetCnt]];
      end;
    TmpI := Order[SentCnt];
    Order[SentCnt] := Order[BestIdx];
    Order[BestIdx] := TmpI;
  end;
  for SentCnt := 0 to csCorpusSize - 1 do
    WriteLn(Format('  %2d. %7.4f  %s',
      [SentCnt + 1, Sims[Order[SentCnt]], csCorpus[Order[SentCnt]]]));

  if DumpPath <> '' then DumpEmbeddingsJSON(DumpPath);

  for SentCnt := 0 to High(Embeddings) do Embeddings[SentCnt].Free;
  for NetCnt := 0 to High(Nets) do Nets[NetCnt].Free;
  Tokenizer.Free;
end.
