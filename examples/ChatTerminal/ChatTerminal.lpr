program ChatTerminal;
(*
ChatTerminal: an interactive terminal chat REPL over any instruct checkpoint
the generic importer dispatch understands (neuralpretrained.pas
BuildFromPretrained: gpt2, llama, mistral, qwen2/3, gemma/2/3, phi/phi3,
gpt_neo(x), gptj, rwkv, mamba, bloom, deepseek_v2, ...).

Point it at a HuggingFace-style model DIRECTORY (config.json +
model.safetensors [+ index] + tokenizer.json [+ tokenizer_config.json]) and
chat:

  ChatTerminal /path/to/model --temperature 0.8 --top-p 0.9

The conversation is maintained as a multi-turn history rendered through the
chat-template engine (neuralchat.pas): the chat format is fingerprinted from
tokenizer_config.json's chat_template (DetectChatFormatFromConfigFile) and
can be overridden with --format chatml|llama2|llama3|zephyr|gemma|phi3|
mistral. Each turn re-renders the WHOLE history (system + user/assistant
turns + generation prompt), encodes it with the HF tokenizer
(EncodeChat) and generates the assistant reply token by token, streaming the
decoded text to stdout as it appears.

Inference parameters map onto the existing decode toolbox (neuraldecode /
neuralvolume): temperature and repetition/frequency/presence penalties run
in the probability domain through a TNNetLogitsProcessorChain
(TNNetTemperatureProcessor / TNNetPenaltyProcessor over
TNNetTokenHistoryPenalty), and --top-k / --top-p / --min-p select the
matching TNNetSampler* (greedy argmax when none is given). NOTE the library
semantics: TNNetSamplerTopK draws UNIFORMLY among the K most probable
tokens; TNNetSamplerTopP / TNNetSamplerMinP draw proportionally.

Generation stops on the tokenizer's EOS id, on the chat format's
end-of-turn marker (e.g. <|im_end|> for ChatML - matched as a token-id stop
sequence in the generated region and trimmed from the reply), or after
--max-new-tokens.

The model is always built with pInferenceOnly=true (the REPL never trains;
frees the per-neuron gradient/momentum buffers). Weight-only int8 storage
(pQuantizeInt8) is the DEFAULT because the FP32 forward path keeps ~3 copies
of every weight matrix (per-neuron + concatenated + interleaved caches), so
a 0.5B model can need >10GB resident; int8 holds ~1/4 of ONE FP32 copy and
dequantizes a layer at a time. Pass --fp32 to opt back into full precision
(much more RAM) when you have the memory and want bit-exact FP32 outputs.

REPL commands: /exit, /reset (clear history), /system <msg> (set the system
prompt; raises on formats without a system role, e.g. gemma/mistral).

--selftest runs the argument-parsing / prompt-assembly / REPL-command unit
checks (no model needed) and exits.

Decoding is a full forward per token (the fixed-width GPT2Import
convention): architecture-agnostic across every imported family, including
the ones whose normalization layers are not KV-cache streamable.

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
  Classes, SysUtils, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained, neuralhftokenizer,
  neuralchat, neuraldecode;

const
  // Default --ctx when the user gives none. Kept modest because build memory
  // is O(ctx^2) (a SeqLen x SeqLen score buffer per head per layer); the full
  // checkpoint context (e.g. 32768) would OOM at load. See the load path.
  DefaultCtxCap = 2048;

type
  TChatOptions = record
    ModelDir: string;
    Int8: boolean;
    CtxLen: integer;             // pSeqLen (0 = the model's full context)
    MaxNewTokens: integer;
    Temperature: TNeuralFloat;   // 1.0 = off
    TopK: integer;               // 0 = off
    WeightedTopK: boolean;       // true = weighted (HF) top-k, false = uniform
    TopP: TNeuralFloat;          // 0 = off
    MinP: TNeuralFloat;          // 0 = off
    RepetitionPenalty: TNeuralFloat; // 1.0 = off
    FrequencyPenalty: TNeuralFloat;  // 0 = off
    PresencePenalty: TNeuralFloat;   // 0 = off
    Seed: integer;               // < 0 = Randomize
    FormatName: string;          // '' = autodetect
    SystemPrompt: string;
    SelfTest: boolean;
    ShowHelp: boolean;
    ErrorMsg: string;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: ChatTerminal <model-dir> [options]');
  WriteLn;
  WriteLn('<model-dir> holds config.json, model.safetensors (or a sharded');
  WriteLn('index / pytorch_model.bin), tokenizer.json and (for chat-format');
  WriteLn('autodetection) tokenizer_config.json.');
  WriteLn;
  WriteLn('Options:');
  WriteLn('  --temperature X       sampling temperature (default 1.0)');
  WriteLn('  --top-k N             top-k sampling (uniform draw among top K)');
  WriteLn('  --weighted-top-k N    top-k sampling (HF: weighted draw among top K)');
  WriteLn('  --top-p X             nucleus sampling (weighted draw)');
  WriteLn('  --min-p X             min-p sampling (weighted draw)');
  WriteLn('  --repetition-penalty X  CTRL repetition penalty (default 1.0)');
  WriteLn('  --frequency-penalty X   frequency penalty (default 0)');
  WriteLn('  --presence-penalty X    presence penalty (default 0)');
  WriteLn('  --max-new-tokens N    reply length cap (default 128)');
  WriteLn('  --seed N              RNG seed (default: randomize)');
  WriteLn('  --ctx N               context window (default min(model max,2048); mem ~O(ctx^2))');
  WriteLn('  --format NAME         chatml|llama2|llama3|zephyr|gemma|phi3|mistral');
  WriteLn('  --system "msg"        initial system prompt');
  WriteLn('  --int8                int8 weight-only quantized inference (DEFAULT)');
  WriteLn('  --fp32                full-precision weights (much more RAM; ~3x the');
  WriteLn('                        weight bytes are held - see --int8)');
  WriteLn('  --selftest            run the offline unit checks and exit');
  WriteLn('  --help                this text');
  WriteLn;
  WriteLn('REPL commands: /exit, /reset, /system <msg>');
end;

function DefaultChatOptions(): TChatOptions;
begin
  Result.ModelDir := '';
  Result.Int8 := true; // int8 weight-only by default (a fraction of FP32 RAM)
  Result.CtxLen := 0;
  Result.MaxNewTokens := 128;
  Result.Temperature := 1.0;
  Result.TopK := 0;
  Result.WeightedTopK := false;
  Result.TopP := 0;
  Result.MinP := 0;
  Result.RepetitionPenalty := 1.0;
  Result.FrequencyPenalty := 0;
  Result.PresencePenalty := 0;
  Result.Seed := -1;
  Result.FormatName := '';
  Result.SystemPrompt := '';
  Result.SelfTest := false;
  Result.ShowHelp := false;
  Result.ErrorMsg := '';
end;

// Parses the command line (already collected into Args). Returns false and
// sets ErrorMsg on a bad flag/value. Kept pure (no ParamStr, no Halt) so
// --selftest can exercise it.
function ParseArgs(Args: TStringList; var Opt: TChatOptions): boolean;
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

  function NextFloat(const FlagName: string; out Value: TNeuralFloat): boolean;
  var
    S: string;
    Code: integer;
    D: double;
  begin
    Result := NextValue(FlagName, S);
    if not Result then exit;
    Val(S, D, Code); // locale-independent, '.' decimal separator
    if Code <> 0 then
    begin
      Opt.ErrorMsg := FlagName + ': not a number: ' + S;
      exit(false);
    end;
    Value := D;
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
  FVal: TNeuralFloat;
  IVal: integer;
begin
  Opt := DefaultChatOptions();
  ArgPos := 0;
  while ArgPos < Args.Count do
  begin
    Arg := Args[ArgPos];
    if Arg = '--int8' then Opt.Int8 := true
    else if Arg = '--fp32' then Opt.Int8 := false
    else if Arg = '--selftest' then Opt.SelfTest := true
    else if (Arg = '--help') or (Arg = '-h') then Opt.ShowHelp := true
    else if Arg = '--temperature' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.Temperature := FVal;
    end
    else if Arg = '--top-k' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.TopK := IVal;
    end
    else if Arg = '--weighted-top-k' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.TopK := IVal;
      Opt.WeightedTopK := true;
    end
    else if Arg = '--top-p' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.TopP := FVal;
    end
    else if Arg = '--min-p' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.MinP := FVal;
    end
    else if Arg = '--repetition-penalty' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.RepetitionPenalty := FVal;
    end
    else if Arg = '--frequency-penalty' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.FrequencyPenalty := FVal;
    end
    else if Arg = '--presence-penalty' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.PresencePenalty := FVal;
    end
    else if Arg = '--max-new-tokens' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.MaxNewTokens := IVal;
    end
    else if Arg = '--seed' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.Seed := IVal;
    end
    else if Arg = '--ctx' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.CtxLen := IVal;
    end
    else if Arg = '--format' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.FormatName := SVal;
    end
    else if Arg = '--system' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.SystemPrompt := SVal;
    end
    else if (Length(Arg) >= 2) and (Copy(Arg, 1, 2) = '--') then
    begin
      Opt.ErrorMsg := 'unknown flag: ' + Arg;
      exit(false);
    end
    else if Opt.ModelDir = '' then Opt.ModelDir := Arg
    else
    begin
      Opt.ErrorMsg := 'unexpected argument: ' + Arg;
      exit(false);
    end;
    Inc(ArgPos);
  end;
  Result := true;
end;

// The end-of-turn marker the assistant reply terminates with in each format
// (the token-id stop sequence; trimmed from the reply when matched).
function EndOfTurnMarker(ChatFormat: TNeuralChatFormat): string;
begin
  case ChatFormat of
    cfChatML:  Result := '<|im_end|>';
    cfLlama2:  Result := '</s>';
    cfLlama3:  Result := '<|eot_id|>';
    cfZephyr:  Result := '</s>';
    cfGemma:   Result := '<end_of_turn>';
    cfPhi3:    Result := '<|end|>';
    cfMistral: Result := '</s>';
  else
    Result := '';
  end;
end;

// Full conversation = optional system message + alternating user/assistant
// History, rendered with the generation prompt so the model continues as
// the assistant.
function AssembleMessages(const SystemPrompt: string;
  const History: TChatMessages): TChatMessages;
var
  Cnt, Ofs: integer;
begin
  Ofs := 0;
  if SystemPrompt <> '' then Ofs := 1;
  SetLength(Result, Length(History) + Ofs);
  if SystemPrompt <> '' then Result[0] := ChatMessage('system', SystemPrompt);
  for Cnt := 0 to High(History) do Result[Cnt + Ofs] := History[Cnt];
end;

// REPL line classification: returns true when Line is a /command and splits
// it into the command word (lowercased, without the slash) and its argument.
function ParseReplCommand(const Line: string; out Cmd, Arg: string): boolean;
var
  SpacePos: integer;
begin
  Cmd := '';
  Arg := '';
  if (Line = '') or (Line[1] <> '/') then exit(false);
  SpacePos := Pos(' ', Line);
  if SpacePos = 0 then Cmd := LowerCase(Copy(Line, 2, Length(Line) - 1))
  else
  begin
    Cmd := LowerCase(Copy(Line, 2, SpacePos - 2));
    Arg := Trim(Copy(Line, SpacePos + 1, Length(Line) - SpacePos));
  end;
  Result := true;
end;

// Stable in-place softmax of a probability row (the imported nets output raw
// logits; the processor chain and the samplers expect POST-SOFTMAX rows).
procedure SoftmaxRow(Row: TNNetVolume);
var
  Cnt: integer;
  MaxV, Sum: TNeuralFloat;
begin
  MaxV := Row.FData[0];
  for Cnt := 1 to Row.Size - 1 do
    if Row.FData[Cnt] > MaxV then MaxV := Row.FData[Cnt];
  Sum := 0;
  for Cnt := 0 to Row.Size - 1 do
  begin
    Row.FData[Cnt] := Exp(Row.FData[Cnt] - MaxV);
    Sum := Sum + Row.FData[Cnt];
  end;
  if Sum > 0 then Row.Mul(1/Sum);
end;

function ArgMaxRow(Row: TNNetVolume): integer;
var
  Cnt: integer;
begin
  Result := 0;
  for Cnt := 1 to Row.Size - 1 do
    if Row.FData[Cnt] > Row.FData[Result] then Result := Cnt;
end;

// True when the tail of Tokens[0..Len-1] equals Marker.
function TailMatches(const Tokens: TNeuralIntegerArray; Len: integer;
  const Marker: TNeuralIntegerArray): boolean;
var
  Cnt, MLen: integer;
begin
  MLen := Length(Marker);
  if (MLen = 0) or (Len < MLen) then exit(false);
  for Cnt := 0 to MLen - 1 do
    if Tokens[Len - MLen + Cnt] <> Marker[Cnt] then exit(false);
  Result := true;
end;

// Reads config.json's model_type for the one-line summary ('' on trouble).
// fpjson gotcha: TJSONParser with options [] (GetJSON mangles non-ASCII).
function ReadModelType(const ConfigFile: string): string;
var
  SL: TStringList;
  Parser: TJSONParser;
  Root: TJSONData;
  Node: TJSONData;
begin
  Result := '';
  if not FileExists(ConfigFile) then exit;
  SL := TStringList.Create();
  try
    SL.LoadFromFile(ConfigFile);
    Parser := TJSONParser.Create(SL.Text, []);
    try
      Root := Parser.Parse();
      try
        Node := Root.FindPath('model_type');
        if Assigned(Node) then Result := Node.AsString;
      finally
        Root.Free;
      end;
    finally
      Parser.Free;
    end;
  except
    Result := '';
  end;
  SL.Free;
end;

// Reads an integer field from config.json (e.g. max_position_embeddings),
// returning Default on any trouble. Same fpjson stance as ReadModelType.
function ReadConfigInt(const ConfigFile, Field: string;
  Default: integer): integer;
var
  SL: TStringList;
  Parser: TJSONParser;
  Root: TJSONData;
  Node: TJSONData;
begin
  Result := Default;
  if not FileExists(ConfigFile) then exit;
  SL := TStringList.Create();
  try
    SL.LoadFromFile(ConfigFile);
    Parser := TJSONParser.Create(SL.Text, []);
    try
      Root := Parser.Parse();
      try
        Node := Root.FindPath(Field);
        if Assigned(Node) and (Node.JSONType = jtNumber) then
          Result := Node.AsInteger;
      finally
        Root.Free;
      end;
    finally
      Parser.Free;
    end;
  except
    Result := Default;
  end;
  SL.Free;
end;

// ---------------------------------------------------------------------------
// Generation: one assistant reply, streamed to stdout as it decodes.
// ---------------------------------------------------------------------------
// Full-recompute decode (one fixed-width forward per token, the GPT2Import
// convention). Probability pipeline per step, matching the
// TGenerationConfig order (penalty -> temperature -> sampler):
//   logits row -> softmax -> Chain.ProcessRow -> Sampler/argmax.
// Stops on EOS (tokenizer's eos id), on the end-of-turn marker token
// sequence, or after Opt.MaxNewTokens. Returns the decoded reply (marker
// trimmed); streamed printing flushes after every token so piped output
// still streams.
function GenerateReply(NN: TNNet; Tokenizer: TNeuralHFTokenizer;
  const PromptIds: TNeuralIntegerArray; const Opt: TChatOptions;
  SeqLen, VocabSize: integer; Chain: TNNetLogitsProcessorChain;
  Sampler: TNNetSamplerBase; const MarkerIds: TNeuralIntegerArray): string;
var
  Tokens: TNeuralIntegerArray;
  Generated: TNeuralIntegerArray;
  Input, Output, Row: TNNetVolume;
  Len, GenLen, StepCnt, Cnt, NewToken: integer;
  Decoded, Printed: string;
begin
  Result := '';
  Len := Length(PromptIds);
  if Len >= SeqLen then
  begin
    WriteLn('[context window full (', Len, ' >= ', SeqLen,
      ' tokens) - /reset the conversation or rebuild with a larger --ctx]');
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
    Chain.Reset(PromptIds);
    for StepCnt := 1 to Opt.MaxNewTokens do
    begin
      if Len >= SeqLen then break;
      Input.Fill(0);
      for Cnt := 0 to Len - 1 do Input.FData[Cnt] := Tokens[Cnt];
      NN.Compute(Input);
      NN.GetOutput(Output);
      // The logits row at the last real position predicts the next token.
      for Cnt := 0 to VocabSize - 1 do
        Row.FData[Cnt] := Output.FData[(Len - 1) * VocabSize + Cnt];
      SoftmaxRow(Row);
      Chain.ProcessRow(Row);
      if Assigned(Sampler) then NewToken := Sampler.GetToken(Row)
      else NewToken := ArgMaxRow(Row);
      Chain.Commit(NewToken);
      Tokens[Len] := NewToken;
      Inc(Len);
      GenLen := Length(Generated);
      SetLength(Generated, GenLen + 1);
      Generated[GenLen] := NewToken;
      // EOS / end-of-turn checks BEFORE printing so markers never echo.
      if (Tokenizer.EosId >= 0) and (NewToken = Tokenizer.EosId) then break;
      if TailMatches(Generated, Length(Generated), MarkerIds) then
      begin
        SetLength(Generated, Length(Generated) - Length(MarkerIds));
        break;
      end;
      // Streamed printing: decode the whole generated region and print the
      // delta (BPE merges/UTF-8 multibyte pieces can rewrite the tail, so
      // only print when the previous text is still a prefix).
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
    // Anything the prefix-guard held back (or trimmed markers shortened).
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
// --selftest: offline unit checks (no model files needed).
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
  Opt: TChatOptions;
  Msgs: TChatMessages;
  History: TChatMessages;
  Rendered, Cmd, Arg: string;
begin
  Failures := 0;
  Args := TStringList.Create();
  try
    // Argument parsing.
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--temperature'); Args.Add('0.7');
    Args.Add('--top-k'); Args.Add('40');
    Args.Add('--top-p'); Args.Add('0.9');
    Args.Add('--min-p'); Args.Add('0.05');
    Args.Add('--repetition-penalty'); Args.Add('1.1');
    Args.Add('--frequency-penalty'); Args.Add('0.2');
    Args.Add('--presence-penalty'); Args.Add('0.3');
    Args.Add('--max-new-tokens'); Args.Add('64');
    Args.Add('--seed'); Args.Add('42');
    Args.Add('--ctx'); Args.Add('128');
    Args.Add('--format'); Args.Add('chatml');
    Args.Add('--system'); Args.Add('Be brief.');
    Args.Add('--int8');
    Check(ParseArgs(Args, Opt), 'full flag set parses');
    Check(Opt.ModelDir = '/tmp/model', 'model dir is the positional arg');
    Check(Abs(Opt.Temperature - 0.7) < 1e-6, '--temperature');
    Check(Opt.TopK = 40, '--top-k');
    Check(Abs(Opt.TopP - 0.9) < 1e-6, '--top-p');
    Check(Abs(Opt.MinP - 0.05) < 1e-6, '--min-p');
    Check(Abs(Opt.RepetitionPenalty - 1.1) < 1e-6, '--repetition-penalty');
    Check(Abs(Opt.FrequencyPenalty - 0.2) < 1e-6, '--frequency-penalty');
    Check(Abs(Opt.PresencePenalty - 0.3) < 1e-6, '--presence-penalty');
    Check(Opt.MaxNewTokens = 64, '--max-new-tokens');
    Check(Opt.Seed = 42, '--seed');
    Check(Opt.CtxLen = 128, '--ctx');
    Check(Opt.FormatName = 'chatml', '--format');
    Check(Opt.SystemPrompt = 'Be brief.', '--system');
    Check(Opt.Int8, '--int8');

    // int8 is the default; --fp32 opts back into full precision.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and Opt.Int8, 'int8 is the default weight mode');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--fp32');
    Check(ParseArgs(Args, Opt) and not Opt.Int8, '--fp32 disables int8');

    Args.Clear;
    Args.Add('--bogus-flag');
    Check(not ParseArgs(Args, Opt), 'unknown flag rejected');
    Args.Clear;
    Args.Add('--temperature');
    Check(not ParseArgs(Args, Opt), 'flag without value rejected');
    Args.Clear;
    Args.Add('--top-k'); Args.Add('abc');
    Check(not ParseArgs(Args, Opt), 'non-integer value rejected');

    // Prompt assembly: system + history through the ChatML template.
    SetLength(History, 2);
    History[0] := ChatMessage('user', 'Hi!');
    History[1] := ChatMessage('assistant', 'Hello!');
    Msgs := AssembleMessages('Be brief.', History);
    Check(Length(Msgs) = 3, 'system prompt prepended to history');
    Check((Msgs[0].Role = 'system') and (Msgs[2].Role = 'assistant'),
      'message order preserved');
    Rendered := ApplyChatTemplate(cfChatML, Msgs, true);
    Check(Rendered =
      '<|im_start|>system' + #10 + 'Be brief.<|im_end|>' + #10 +
      '<|im_start|>user' + #10 + 'Hi!<|im_end|>' + #10 +
      '<|im_start|>assistant' + #10 + 'Hello!<|im_end|>' + #10 +
      '<|im_start|>assistant' + #10,
      'ChatML render matches the HF ground truth shape');
    Msgs := AssembleMessages('', History);
    Check(Length(Msgs) = 2, 'empty system prompt adds no message');

    // End-of-turn markers.
    Check(EndOfTurnMarker(cfChatML) = '<|im_end|>', 'ChatML end marker');
    Check(EndOfTurnMarker(cfLlama3) = '<|eot_id|>', 'Llama-3 end marker');
    Check(EndOfTurnMarker(cfGemma) = '<end_of_turn>', 'Gemma end marker');
    Check(EndOfTurnMarker(cfPhi3) = '<|end|>', 'Phi-3 end marker');

    // REPL command parsing.
    Check(ParseReplCommand('/exit', Cmd, Arg) and (Cmd = 'exit') and
      (Arg = ''), '/exit parses');
    Check(ParseReplCommand('/system  Be terse. ', Cmd, Arg) and
      (Cmd = 'system') and (Arg = 'Be terse.'), '/system parses its argument');
    Check(not ParseReplCommand('plain chat line', Cmd, Arg),
      'plain text is not a command');

    // Format name round trip used by --format.
    Check(ChatFormatFromName('llama3') = cfLlama3, '--format name lookup');
    Check(ChatFormatFromName('nope') = cfUnknown, 'unknown format name');
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
// Main REPL.
// ---------------------------------------------------------------------------
var
  Opt: TChatOptions;
  Args: TStringList;
  NN: TNNet;
  Tokenizer: TNeuralHFTokenizer;
  ChatFormat: TNeuralChatFormat;
  History: TChatMessages;
  Msgs: TChatMessages;
  PromptIds, MarkerIds: TNeuralIntegerArray;
  Chain: TNNetLogitsProcessorChain;
  Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  Cnt, SeqLen, VocabSize: integer;
  Line, Cmd, Arg, Reply, ModelType, Marker: string;
  TokenizerFile, TokenizerConfigFile: string;
begin
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
  if Opt.ShowHelp or (Opt.ModelDir = '') then
  begin
    PrintUsage();
    if Opt.ModelDir = '' then Halt(1);
    Halt(0);
  end;

  // Tokenizer + chat format.
  TokenizerFile := IncludeTrailingPathDelimiter(Opt.ModelDir) +
    'tokenizer.json';
  if not FileExists(TokenizerFile) then
  begin
    WriteLn('No tokenizer.json found in ', Opt.ModelDir);
    Halt(1);
  end;
  Tokenizer := TNeuralHFTokenizer.Create();
  Tokenizer.LoadFromFile(TokenizerFile);

  ChatFormat := cfUnknown;
  if Opt.FormatName <> '' then
  begin
    ChatFormat := ChatFormatFromName(Opt.FormatName);
    if ChatFormat = cfUnknown then
    begin
      WriteLn('Unknown --format name: ', Opt.FormatName);
      Halt(1);
    end;
  end
  else
  begin
    TokenizerConfigFile := IncludeTrailingPathDelimiter(Opt.ModelDir) +
      'tokenizer_config.json';
    if FileExists(TokenizerConfigFile) then
      ChatFormat := DetectChatFormatFromConfigFile(TokenizerConfigFile);
    if ChatFormat = cfUnknown then
    begin
      WriteLn('[no chat template detected - defaulting to ChatML; override',
        ' with --format]');
      ChatFormat := cfChatML;
    end;
  end;

  if Opt.Seed >= 0 then RandSeed := Opt.Seed
  else Randomize;

  // Default context window. The full-recompute decode allocates a persistent
  // SeqLen x SeqLen attention-score buffer PER HEAD PER LAYER, so build memory
  // grows as O(SeqLen^2) (e.g. ~336 such buffers for a 24-layer/14-head 0.5B
  // model). Using the checkpoint's full max_position_embeddings (32768 for
  // Qwen2.5) would request terabytes and OOM at load, so when the user gives
  // no --ctx we cap the default at DefaultCtxCap (clamped to the model's own
  // limit). Raise it with --ctx N (and prefer --int8) if you have the RAM.
  if Opt.CtxLen <= 0 then
  begin
    Cnt := ReadConfigInt(IncludeTrailingPathDelimiter(Opt.ModelDir) +
      'config.json', 'max_position_embeddings', DefaultCtxCap);
    if (Cnt <= 0) or (Cnt > DefaultCtxCap) then Cnt := DefaultCtxCap;
    Opt.CtxLen := Cnt;
    WriteLn('[context not set - defaulting to ', Opt.CtxLen,
      ' tokens; override with --ctx N (memory grows ~O(ctx^2))]');
  end;

  // Model: generic architecture dispatch, inference-only, optional int8.
  // Weight precision. int8 is the default because the FP32 forward path keeps
  // ~3 copies of every weight matrix; FP32 is opt-in via --fp32.
  if Opt.Int8 then
    WriteLn('[int8 weights (default) - pass --fp32 for full-precision',
      ' weights (much more RAM)]')
  else
    WriteLn('[--fp32: full-precision weights - this holds ~3x the weight',
      ' bytes; drop --fp32 to use int8 if you run low on RAM]');

  WriteLn('Loading ', Opt.ModelDir, ' ...');
  NN := BuildFromPretrained(Opt.ModelDir, Opt.CtxLen,
    {pInferenceOnly=}true, '', {pQuantizeInt8=}Opt.Int8);
  SeqLen := NN.GetFirstLayer().Output.SizeX;
  VocabSize := NN.GetLastLayer().Output.Depth;
  ModelType := ReadModelType(IncludeTrailingPathDelimiter(Opt.ModelDir) +
    'config.json');
  if ModelType = '' then ModelType := 'unknown';
  WriteLn('Model: ', ModelType, ', ', NN.CountWeights(), ' params, vocab ',
    VocabSize, ', context ', SeqLen, ', chat format ',
    ChatFormatName(ChatFormat), ', ',
    BoolToStr(Opt.Int8, 'int8', 'fp32'), ' weights.');

  // Distribution pipeline (TGenerationConfig order: penalty -> temperature
  // -> sampler).
  Chain := TNNetLogitsProcessorChain.Create();
  Penalty := nil;
  if (Opt.RepetitionPenalty <> 1.0) or (Opt.FrequencyPenalty <> 0) or
    (Opt.PresencePenalty <> 0) then
  begin
    Penalty := TNNetTokenHistoryPenalty.Create(Opt.RepetitionPenalty,
      Opt.FrequencyPenalty, Opt.PresencePenalty);
    Chain.Add(TNNetPenaltyProcessor.Create(Penalty, {OwnsPenalty=}true),
      {OwnsProcessor=}true);
  end;
  if Opt.Temperature <> 1.0 then
    Chain.Add(TNNetTemperatureProcessor.Create(Opt.Temperature), true);
  Sampler := nil;
  if Opt.TopK > 0 then
  begin
    if Opt.WeightedTopK then Sampler := TNNetSamplerWeightedTopK.Create(Opt.TopK)
    else Sampler := TNNetSamplerTopK.Create(Opt.TopK);
  end
  else if Opt.TopP > 0 then Sampler := TNNetSamplerTopP.Create(Opt.TopP)
  else if Opt.MinP > 0 then Sampler := TNNetSamplerMinP.Create(Opt.MinP);

  // End-of-turn marker as a token-id stop sequence (single id when the
  // tokenizer has it as an added token, a multi-id sequence otherwise).
  Marker := EndOfTurnMarker(ChatFormat);
  if Marker <> '' then MarkerIds := Tokenizer.Encode(Marker)
  else SetLength(MarkerIds, 0);

  SetLength(History, 0);
  WriteLn('Type your message; /exit quits, /reset clears the history,');
  WriteLn('/system <msg> sets the system prompt.');
  while true do
  begin
    Write('> ');
    Flush(System.Output);
    if EOF(System.Input) then break;
    ReadLn(Line);
    Line := Trim(Line);
    if Line = '' then continue;
    if ParseReplCommand(Line, Cmd, Arg) then
    begin
      if Cmd = 'exit' then break
      else if Cmd = 'reset' then
      begin
        SetLength(History, 0);
        WriteLn('[history cleared]');
      end
      else if Cmd = 'system' then
      begin
        Opt.SystemPrompt := Arg;
        WriteLn('[system prompt set]');
      end
      else WriteLn('[unknown command /', Cmd, ' - /exit, /reset, /system]');
      continue;
    end;
    SetLength(History, Length(History) + 1);
    History[High(History)] := ChatMessage('user', Line);
    try
      Msgs := AssembleMessages(Opt.SystemPrompt, History);
      PromptIds := EncodeChat(Tokenizer, ChatFormat, Msgs,
        {AddGenerationPrompt=}true);
      Reply := GenerateReply(NN, Tokenizer, PromptIds, Opt, SeqLen,
        VocabSize, Chain, Sampler, MarkerIds);
      SetLength(History, Length(History) + 1);
      History[High(History)] := ChatMessage('assistant', Reply);
    except
      on E: ENeuralChatError do
      begin
        // e.g. a system prompt on a format without a system role; drop the
        // failed user turn so the history stays consistent.
        WriteLn('[chat template error: ', E.Message, ']');
        SetLength(History, Length(History) - 1);
      end;
    end;
  end;
  WriteLn('Bye.');
  Sampler.Free;
  Chain.Free; // owns the processors, which own the penalty
  Tokenizer.Free;
  NN.Free;
end.
