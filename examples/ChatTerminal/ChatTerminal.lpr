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

SAMPLING DEFAULTS (ApplySamplingDefaults, per parameter): an explicit flag
wins; otherwise the model's generation_config.json (the checkpoint author's
recommended settings - temperature/top_p/top_k/repetition_penalty, with
do_sample=false honored as greedy) supplies the value; otherwise the
built-in fallback top-p 0.2 + repetition-penalty 1.05 (near-greedy
stability, the mild penalty preventing the repetition loops pure greedy
falls into on small models). A config top_k maps to the WEIGHTED top-k and
top_p is preferred over top_k (see the library semantics note above).
--greedy hard-overrides everything - deterministic argmax with no sampler,
temperature or penalties - the mode for CPU/GPU parity checks and debugging.

Generation stops on the tokenizer's EOS id, on the chat format's
end-of-turn marker (e.g. <|im_end|> for ChatML - matched as a token-id stop
sequence in the generated region and trimmed from the reply), or after
--max-new-tokens.

The model is always built inference-only (pTrainable=false): the REPL never
trains, so the per-neuron gradient/momentum (Delta/BackInertia) training
buffers are never allocated.

Independently of trainability, --low-memory / --max-fast-memory toggles the
low-memory forward path (the pLowMemory argument of SetTrainable). When ON
(the DEFAULT) each conv/linear layer DROPS its persistent concatenated
weight cache (FConcatedWeights) and computes per-neuron straight from the
weights: less resident RAM, a somewhat slower forward. --max-fast-memory keeps
the concatenated cache for a faster forward at the cost of more RAM. (The two
settings are orthogonal; low memory only touches the forward weight cache,
trainability only the backprop buffers.) Weight-only int8 storage
(pQuantizeInt8) is the DEFAULT: less RAM AND faster than fp32 on both CPU
(fused int8 kernels, AVX2-accelerated) and GPU. Pass --fp32 for
full-precision FP32 weights (more RAM, slower). Combined with --gpu the
quantized codes and per-row scales are uploaded ONCE as resident device
buffers (cai_dot_product_int8), so int8 runs on the GPU too.

REPL commands: /exit, /reset (clear history), /system <msg> (set the system
prompt; raises on formats without a system role, e.g. gemma/mistral).

--stats prints per-turn timing to stderr (kept off stdout so piped model
output stays clean): time-to-first-token (prefill + the first decode step)
and the steady-state decode rate in tok/s (measured over the tokens after
the first, so prefill is excluded).

--selftest runs the argument-parsing / prompt-assembly / REPL-command unit
checks (no model needed) and exits.

Decoding streams through a TNNetStreamingDecoder KV cache (one width-1
forward per token). Across turns the cache is REUSED: each turn's prompt
shares a long prefix with what is already resident (last turn's prompt +
reply), so the session diffs the new prompt against the cached token ids
(CommonPrefixLen), TruncateTo's the divergent tail and prefills only the new
tokens - time-to-first-token stays roughly flat instead of growing with the
conversation. Correct independent of tokenizer round-tripping (the diff
always finds the true shared prefix; /system and /reset just diverge earlier).
Pure-attention models only: a recurrent (SSM) state cannot be position-
truncated, so those (and --no-cache-reuse) fall back to a full re-prefill.

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
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
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
    LowMemory: boolean;          // true (default) = low-memory forward path
                                 // (drops the concatenated weight cache);
                                 // independent of trainability
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
    Greedy: boolean;             // --greedy: deterministic argmax - no sampler,
                                 // no temperature, no penalties. Hard override:
                                 // beats explicit sampling flags AND the model's
                                 // generation_config.json (CPU/GPU parity and
                                 // debugging mode)
    // "Explicitly set on the command line" trackers. ApplySamplingDefaults
    // fills a parameter from generation_config.json (or the built-in fallback)
    // only when its flag was NOT given: CLI > generation_config > fallback.
    TemperatureSet: boolean;
    TopKSet: boolean;
    TopPSet: boolean;
    MinPSet: boolean;
    RepPenaltySet: boolean;
    Seed: integer;               // < 0 = Randomize
    FormatName: string;          // '' = autodetect
    SystemPrompt: string;
    SelfTest: boolean;
    ShowHelp: boolean;
    Stats: boolean;              // per-turn timing to stderr (TTFT, tok/s)
    Profile: boolean;            // per-layer-class forward timing to stderr after
                                 // each turn (decode steps only); for picking the
                                 // next layer class to optimize (e.g. OpenCL)
    NoCacheReuse: boolean;       // force full re-prefill every turn (A/B + debug)
    KVInt8: boolean;             // int8-quantized KV cache (~1/4 the KV RAM at
                                 // long context; logits not bit-exact). Follows
                                 // the weight mode (on with int8 weights, off
                                 // with --fp32) unless --kv-int8/--kv-fp32
                                 // picks explicitly - identical CPU/GPU.
    KVInt8Set: boolean;          // --kv-int8/--kv-fp32 given: skip the
                                 // follow-the-weights default
    Serial: boolean;             // serial layer loop; default is the parallel
                                 // layer-graph scheduler (ComputeParallel).
                                 // The parallel path also enables intra-layer
                                 // threading (big conv/linear layers split
                                 // across the pool); --serial disables both.
    Gpu: boolean;                // offload conv/linear matmuls via OpenCL
    GpuPlatform: integer;        // OpenCL platform index (default 0)
    GpuDevice: integer;          // OpenCL device index within the platform (0)
    ErrorMsg: string;
  end;

  // Sampling-relevant fields of a HuggingFace generation_config.json, each
  // with a presence flag (an absent field must not override anything).
  // Filled by ReadGenerationConfig, consumed by ApplySamplingDefaults.
  TGenConfigDefaults = record
    Found: boolean;              // file existed and parsed
    HasDoSample: boolean;        DoSample: boolean;
    HasTemperature: boolean;     Temperature: TNeuralFloat;
    HasTopP: boolean;            TopP: TNeuralFloat;
    HasTopK: boolean;            TopK: integer;
    HasRepetitionPenalty: boolean; RepetitionPenalty: TNeuralFloat;
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
  WriteLn('  Sampling defaults come from the model''s generation_config.json when');
  WriteLn('  present; otherwise top-p 0.2 + repetition-penalty 1.05. Explicit');
  WriteLn('  flags override the config; --greedy overrides everything.');
  WriteLn('  --greedy              deterministic argmax: no sampler, no temperature,');
  WriteLn('                        no penalties (CPU/GPU parity + debugging)');
  WriteLn('  --temperature X       sampling temperature (1.0 = off)');
  WriteLn('  --top-k N             top-k sampling (uniform draw among top K)');
  WriteLn('  --weighted-top-k N    top-k sampling (HF: weighted draw among top K)');
  WriteLn('  --top-p X             nucleus sampling (weighted draw)');
  WriteLn('  --min-p X             min-p sampling (weighted draw)');
  WriteLn('  --repetition-penalty X  CTRL repetition penalty (1.0 = off)');
  WriteLn('  --frequency-penalty X   frequency penalty (default 0)');
  WriteLn('  --presence-penalty X    presence penalty (default 0)');
  WriteLn('  --max-new-tokens N    reply length cap (default 8192)');
  WriteLn('  --seed N              RNG seed (default: randomize)');
  WriteLn('  --ctx N               context window (default min(model max,2048); mem ~O(ctx^2))');
  WriteLn('  --format NAME         chatml|llama2|llama3|zephyr|gemma|phi3|mistral');
  WriteLn('  --system "msg"        initial system prompt');
  WriteLn('  --int8                int8 weight-only quantized inference (DEFAULT; less');
  WriteLn('                        RAM and faster on CPU and GPU: resident int8 codes)');
  WriteLn('  --fp32                full-precision weights (more RAM, slower)');
  WriteLn('  --low-memory          drop conv/linear weight cache; per-neuron forward (DEFAULT)');
  WriteLn('  --max-fast-memory     keep the concatenated weight cache (faster forward, more RAM)');
  WriteLn('  --kv-int8             int8-quantized KV cache: ~1/4 the KV RAM at long context');
  WriteLn('                        (per-row scales; slightly lossy logits, argmax stable).');
  WriteLn('                        DEFAULT whenever the weights are int8; --fp32 weights');
  WriteLn('                        default to the FP32 cache');
  WriteLn('  --kv-fp32             keep the bit-exact FP32 KV cache with int8 weights');
  WriteLn('  --gpu                 OpenCL offload of conv/linear matmuls (DEFAULT when');
  WriteLn('                        built with -dOpenCL); --cpu forces CPU');
  WriteLn('  --cpu                 force CPU even when built with -dOpenCL');
  WriteLn('  --gpu-platform N      OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N        OpenCL device index within the platform (default 0)');
  WriteLn('  --stats               per-turn timing to stderr (TTFT, decode tok/s)');
  WriteLn('  --profile             per-layer-class forward timing to stderr after each');
  WriteLn('                        turn (decode steps only); ranks classes to optimize.');
  WriteLn('                        Also prints [sched]: layer-graph parallelism (graph');
  WriteLn('                        width, parallel vs serial passes, peak in-flight)');
  WriteLn('  --no-cache-reuse      re-prefill the whole prompt each turn (default:');
  WriteLn('                        reuse the shared KV-cache prefix from last turn)');
  WriteLn('  --serial              serial layer loop (default: layer-graph parallel');
  WriteLn('                        forward across independent layers; the parallel');
  WriteLn('                        path also threads large conv/linear layers');
  WriteLn('                        internally, --serial runs fully single-threaded)');
  WriteLn('  --selftest            run the offline unit checks and exit');
  WriteLn('  --help                this text');
  WriteLn;
  WriteLn('REPL commands: /exit, /reset, /system <msg>');
end;

function DefaultChatOptions(): TChatOptions;
begin
  Result.ModelDir := '';
  Result.Int8 := true; // int8 weights by default: less RAM, faster (--fp32 opts out)
  Result.LowMemory := true; // low-memory forward path by default (drops weight cache)
  Result.CtxLen := 0;
  Result.MaxNewTokens := 8192;
  Result.Temperature := 1.0;
  Result.TopK := 0;
  Result.WeightedTopK := false;
  Result.TopP := 0;
  Result.MinP := 0;
  Result.RepetitionPenalty := 1.0;
  Result.FrequencyPenalty := 0;
  Result.PresencePenalty := 0;
  Result.Greedy := false;
  Result.TemperatureSet := false;
  Result.TopKSet := false;
  Result.TopPSet := false;
  Result.MinPSet := false;
  Result.RepPenaltySet := false;
  Result.Seed := -1;
  Result.FormatName := '';
  Result.SystemPrompt := '';
  Result.SelfTest := false;
  Result.ShowHelp := false;
  Result.Stats := false;
  Result.Profile := false;
  Result.NoCacheReuse := false;
  Result.KVInt8 := false;    // resolved after parsing: follows the weight mode
  Result.KVInt8Set := false; // unless --kv-int8/--kv-fp32 picked explicitly
  Result.Serial := false; // parallel layer-graph forward by default (--serial)
  // OpenCL offload defaults ON when the binary is built with -dOpenCL (the
  // default compilation), OFF otherwise; --cpu forces CPU either way.
  Result.Gpu := {$IFDEF OpenCL}true{$ELSE}false{$ENDIF};
  Result.GpuPlatform := 0;
  Result.GpuDevice := 0;
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
    else if Arg = '--low-memory' then Opt.LowMemory := true
    else if Arg = '--max-fast-memory' then Opt.LowMemory := false
    else if Arg = '--stats' then Opt.Stats := true
    else if Arg = '--profile' then Opt.Profile := true
    else if Arg = '--no-cache-reuse' then Opt.NoCacheReuse := true
    else if Arg = '--kv-int8' then
    begin
      Opt.KVInt8 := true;
      Opt.KVInt8Set := true;
    end
    else if Arg = '--kv-fp32' then
    begin
      Opt.KVInt8 := false;
      Opt.KVInt8Set := true;
    end
    else if Arg = '--serial' then Opt.Serial := true
    else if Arg = '--gpu' then Opt.Gpu := true
    else if Arg = '--cpu' then Opt.Gpu := false
    else if Arg = '--gpu-platform' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.GpuPlatform := IVal;
    end
    else if Arg = '--gpu-device' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.GpuDevice := IVal;
    end
    else if Arg = '--selftest' then Opt.SelfTest := true
    else if (Arg = '--help') or (Arg = '-h') then Opt.ShowHelp := true
    else if Arg = '--greedy' then Opt.Greedy := true
    else if Arg = '--temperature' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.Temperature := FVal;
      Opt.TemperatureSet := true;
    end
    else if Arg = '--top-k' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.TopK := IVal;
      Opt.TopKSet := true;
    end
    else if Arg = '--weighted-top-k' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      Opt.TopK := IVal;
      Opt.WeightedTopK := true;
      Opt.TopKSet := true;
    end
    else if Arg = '--top-p' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.TopP := FVal;
      Opt.TopPSet := true;
    end
    else if Arg = '--min-p' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.MinP := FVal;
      Opt.MinPSet := true;
    end
    else if Arg = '--repetition-penalty' then
    begin
      if not NextFloat(Arg, FVal) then exit(false);
      Opt.RepetitionPenalty := FVal;
      Opt.RepPenaltySet := true;
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
  // The KV cache follows the weight mode unless picked explicitly: int8
  // weights get the int8 KV cache (same accuracy philosophy, ~1/4 the KV
  // RAM), --fp32 weights keep the bit-exact FP32 cache. Identical on CPU
  // and GPU (the cached decode path is the same code).
  if not Opt.KVInt8Set then Opt.KVInt8 := Opt.Int8;
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
// logits; the processor chain and the samplers expect POST-SOFTMAX rows) is
// now neuralvolume.RowSoftMax.

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

// Length of the longest common prefix of two token-id sequences. Used by the
// incremental KV-cache reuse: A is the sequence currently resident in the
// cache (positions 0..High), B is this turn's freshly rendered prompt; the
// cache can be kept up to this length and only B's tail re-prefilled.
function CommonPrefixLen(const A, B: TNeuralIntegerArray): integer;
var
  N: integer;
begin
  Result := 0;
  N := Length(A);
  if Length(B) < N then N := Length(B);
  while (Result < N) and (A[Result] = B[Result]) do Inc(Result);
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

// Built-in fallback sampling defaults, used only for parameters that neither
// an explicit flag nor the model's generation_config.json supplies. A tight
// nucleus + a mild penalty: near-greedy stability, but the penalty prevents
// the repetition loops pure greedy falls into on small models.
const
  csFallbackTopP = 0.2;
  csFallbackRepetitionPenalty = 1.05;

// Reads the sampling-relevant fields of the model's generation_config.json
// (the checkpoint author's recommended decode settings). Absent file/fields
// leave the presence flags false. Same fpjson stance as ReadModelType.
function ReadGenerationConfig(const FileName: string): TGenConfigDefaults;
var
  SL: TStringList;
  Parser: TJSONParser;
  Root: TJSONData;
  Node: TJSONData;
begin
  FillChar(Result, SizeOf(Result), 0);
  if not FileExists(FileName) then exit;
  SL := TStringList.Create();
  try
    try
      SL.LoadFromFile(FileName);
      Parser := TJSONParser.Create(SL.Text, []);
      try
        Root := Parser.Parse();
        try
          Result.Found := true;
          Node := Root.FindPath('do_sample');
          if Assigned(Node) and (Node.JSONType = jtBoolean) then
          begin
            Result.HasDoSample := true;
            Result.DoSample := Node.AsBoolean;
          end;
          Node := Root.FindPath('temperature');
          if Assigned(Node) and (Node.JSONType = jtNumber) then
          begin
            Result.HasTemperature := true;
            Result.Temperature := Node.AsFloat;
          end;
          Node := Root.FindPath('top_p');
          if Assigned(Node) and (Node.JSONType = jtNumber) then
          begin
            Result.HasTopP := true;
            Result.TopP := Node.AsFloat;
          end;
          Node := Root.FindPath('top_k');
          if Assigned(Node) and (Node.JSONType = jtNumber) then
          begin
            Result.HasTopK := true;
            Result.TopK := Node.AsInteger;
          end;
          Node := Root.FindPath('repetition_penalty');
          if Assigned(Node) and (Node.JSONType = jtNumber) then
          begin
            Result.HasRepetitionPenalty := true;
            Result.RepetitionPenalty := Node.AsFloat;
          end;
        finally
          Root.Free;
        end;
      finally
        Parser.Free;
      end;
    except
      FillChar(Result, SizeOf(Result), 0); // unreadable/bad JSON = no config
    end;
  finally
    SL.Free;
  end;
end;

// Resolves the effective sampling settings in Opt. Per parameter the
// precedence is: explicit CLI flag > generation_config.json > built-in
// fallback (csFallbackTopP / csFallbackRepetitionPenalty). --greedy is a hard
// override of everything, including explicit sampling flags - it is the
// deterministic argmax parity/debug mode. A config with do_sample=false means
// the model author recommends greedy: it contributes greedy defaults (explicit
// flags still override individually, matching the per-parameter rule).
// Sampler choice from a config: top_p is preferred over top_k because this
// library's plain top-k draws UNIFORMLY among the K most probable tokens; a
// config top_k maps to the HF-style WEIGHTED top-k when no top_p is given.
// Kept pure (no file access - the caller passes the parsed config) so
// --selftest can exercise the precedence table.
procedure ApplySamplingDefaults(var Opt: TChatOptions;
  const Cfg: TGenConfigDefaults);
var
  CfgGreedy, UserSampler: boolean;
begin
  if Opt.Greedy then
  begin
    Opt.Temperature := 1.0;
    Opt.TopK := 0;
    Opt.WeightedTopK := false;
    Opt.TopP := 0;
    Opt.MinP := 0;
    Opt.RepetitionPenalty := 1.0;
    Opt.FrequencyPenalty := 0;
    Opt.PresencePenalty := 0;
    exit;
  end;
  CfgGreedy := Cfg.Found and Cfg.HasDoSample and (not Cfg.DoSample);
  UserSampler := Opt.TopKSet or Opt.TopPSet or Opt.MinPSet;
  if not Opt.TemperatureSet then
  begin
    // The fallback deliberately leaves temperature at 1.0 (off): with the
    // tight fallback nucleus it would only reshape 1-3 candidates anyway.
    if (not CfgGreedy) and Cfg.HasTemperature then
      Opt.Temperature := Cfg.Temperature;
  end;
  if not Opt.RepPenaltySet then
  begin
    if CfgGreedy then Opt.RepetitionPenalty := 1.0
    else if Cfg.HasRepetitionPenalty then
      Opt.RepetitionPenalty := Cfg.RepetitionPenalty
    else Opt.RepetitionPenalty := csFallbackRepetitionPenalty;
  end;
  if not UserSampler then
  begin
    if CfgGreedy then
    begin
      Opt.TopK := 0;
      Opt.TopP := 0;
      Opt.MinP := 0;
    end
    else if Cfg.HasTopP then Opt.TopP := Cfg.TopP
    else if Cfg.HasTopK then
    begin
      Opt.TopK := Cfg.TopK;
      Opt.WeightedTopK := true;
    end
    else Opt.TopP := csFallbackTopP;
  end;
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
// CacheReuse: keep the KV cache across turns and only prefill the tail that
// diverges from CachedTokens (the token-id sequence currently resident in the
// cache, updated here in/out). When false, the cache is fully reset and the
// whole prompt re-prefilled (the SSM/recurrent path, where the cache cannot be
// truncated by position, and --no-cache-reuse).
function GenerateReply(NN: TNNet; Session: TNNetStreamingDecoder;
  Tokenizer: TNeuralHFTokenizer;
  const PromptIds: TNeuralIntegerArray; const Opt: TChatOptions;
  SeqLen, VocabSize: integer; Chain: TNNetLogitsProcessorChain;
  Sampler: TNNetSamplerBase; const MarkerIds: TNeuralIntegerArray;
  var CachedTokens: TNeuralIntegerArray; CacheReuse: boolean): string;
var
  Tokens: TNeuralIntegerArray;
  Generated: TNeuralIntegerArray;
  InV, Output, Row: TNNetVolume;
  Len, GenLen, StepCnt, Cnt, NewToken: integer;
  Reused, PromptLen: integer;  // KV-cache reuse bookkeeping (and --stats)
  Decoded, Printed: string;
  // --stats timing (monotonic ms). TStart: before prefill; TFirst: when the
  // first reply token is produced (so TTFT covers prefill + first step);
  // TEnd: after the decode loop. Produced counts emitted tokens.
  TStart, TFirst, TEnd: QWord;
  Produced: integer;
  DecodeSecs: double;
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
  InV := TNNetVolume.Create(1, 1, 1);
  Output := nil; // a reference into the net, returned by Session.Output()
  Row := TNNetVolume.Create(VocabSize, 1, 1);
  TStart := GetTickCount64();
  TFirst := 0;
  TEnd := 0;
  Produced := 0;
  PromptLen := Len;
  try
    Chain.Reset(PromptIds);
    // Prefill the prompt token-at-a-time, reusing the KV-cache prefix shared
    // with last turn when possible. Reused = length of the cached prefix that
    // still matches this prompt; TruncateTo drops the divergent tail (Reused=0
    // is a full reset). The LAST prompt token is fed as the first decode step's
    // input, so the cache must not already hold it - cap reuse at Len-1.
    if CacheReuse then
    begin
      Reused := CommonPrefixLen(CachedTokens, PromptIds);
      if Reused > Len - 1 then Reused := Len - 1;
      Session.TruncateTo(Reused);
    end
    else
    begin
      Reused := 0;
      Session.Reset(); // SSM state cannot be position-truncated; full reset
    end;
    for Cnt := Reused to Len - 2 do
    begin
      InV.FData[0] := Tokens[Cnt];
      Session.StepForward(InV, Cnt);
    end;
    // --profile: discard the one-shot prefill timings (and scheduler stats)
    // so the per-layer-class report below reflects only the repeated
    // single-token decode steps - the steady-state workload whose layer costs
    // we want to rank for optimization.
    if Opt.Profile then
    begin
      NN.ClearTime();
      NN.ResetSchedulerStats();
    end;
    for StepCnt := 1 to Opt.MaxNewTokens do
    begin
      if Len >= SeqLen then break;
      // One width-1 forward of the last committed token over the cached past.
      InV.FData[0] := Tokens[Len - 1];
      Session.StepForward(InV, Len - 1);
      Output := Session.Output(); // (1,1,vocab) -- the single logits row
      for Cnt := 0 to VocabSize - 1 do Row.FData[Cnt] := Output.FData[Cnt];
      RowSoftMax(Row);
      Chain.ProcessRow(Row);
      if Assigned(Sampler) then NewToken := Sampler.GetToken(Row)
      else NewToken := ArgMaxRow(Row);
      Chain.Commit(NewToken);
      Tokens[Len] := NewToken;
      Inc(Len);
      Inc(Produced);
      if Produced = 1 then TFirst := GetTickCount64(); // TTFT boundary
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
    // Record the sequence now resident in the cache for next turn's prefix
    // diff: every token that was FED is cached (positions 0..Len-2); the final
    // produced token (Tokens[Len-1]) was sampled but never fed, so it is not.
    SetLength(CachedTokens, Len - 1);
    for Cnt := 0 to Len - 2 do CachedTokens[Cnt] := Tokens[Cnt];
    // Per-turn timing to stderr (keeps stdout = pure model output). TTFT =
    // prefill + first decode step; tok/s measures the steady-state decode of
    // the tokens AFTER the first, so prefill cost is excluded. prompt N (reused
    // K) shows how much of the prompt the KV-cache reuse skipped re-prefilling.
    if Opt.Stats and (Produced > 0) then
    begin
      TEnd := GetTickCount64();
      Write(StdErr, Format('[stats] %d tokens, TTFT %d ms, prompt %d (reused %d)',
        [Produced, TFirst - TStart, PromptLen, Reused]));
      if Produced > 1 then
      begin
        DecodeSecs := (TEnd - TFirst) / 1000.0;
        if DecodeSecs > 0 then
          Write(StdErr, Format(', decode %.1f tok/s',
            [(Produced - 1) / DecodeSecs]));
      end;
      WriteLn(StdErr);
      Flush(StdErr);
    end;
    // --profile: per-layer-class forward timing accumulated over this turn's
    // decode steps (prefill was cleared above). Printed to stderr so stdout
    // stays pure model output. Ranks layer classes by aggregate forward cost -
    // the actionable signal for picking the next class to optimize (e.g. OpenCL).
    if Opt.Profile and (Produced > 0) then
    begin
      WriteLn(StdErr);
      Write(StdErr, TNNet.LayerClassTimingReport(NN));
      // Layer-graph scheduler parallelism for this turn's decode steps: how
      // wide the graph is, how often the parallel path ran vs the serial
      // fallback, and how much overlap it actually achieved (peak in-flight,
      // share of layers computed off the primary worker).
      WriteLn(StdErr, '[sched] ', NN.SchedulerStatsReport());
      Flush(StdErr);
    end;
  finally
    Row.Free;
    InV.Free;
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
  PA, PB: TNeuralIntegerArray;  // CommonPrefixLen fixtures
  GenCfg: TGenConfigDefaults;   // ApplySamplingDefaults fixtures
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
    Args.Add('--stats');
    Args.Add('--profile');
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
    Check(Opt.Stats, '--stats');
    Check(Opt.Profile, '--profile');

    // int8 is the default; --fp32 opts into full-precision weights.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and Opt.Int8, 'int8 is the default weight mode');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--fp32');
    Check(ParseArgs(Args, Opt) and not Opt.Int8, '--fp32 disables int8');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--fp32'); Args.Add('--int8');
    Check(ParseArgs(Args, Opt) and Opt.Int8, '--int8 re-enables it');

    // --stats is off by default.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and not Opt.Stats, 'stats off by default');

    // --profile is off by default.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and not Opt.Profile, 'profile off by default');

    // Low-memory forward path is on by default; --max-fast-memory keeps the
    // concatenated weight cache; --low-memory re-enables the per-neuron path.
    // Orthogonal to trainability (the build is always inference-only).
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and Opt.LowMemory, 'low-memory on by default');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--max-fast-memory');
    Check(ParseArgs(Args, Opt) and not Opt.LowMemory, '--max-fast-memory disables it');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--max-fast-memory'); Args.Add('--low-memory');
    Check(ParseArgs(Args, Opt) and Opt.LowMemory, '--low-memory re-enables it');

    // int8 KV cache follows the weight mode unless picked explicitly, in
    // any flag order.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and Opt.KVInt8,
      'kv-int8 on by default (int8 weights are the default)');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--fp32');
    Check(ParseArgs(Args, Opt) and not Opt.KVInt8,
      '--fp32 weights default to the FP32 KV cache');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--kv-fp32');
    Check(ParseArgs(Args, Opt) and not Opt.KVInt8,
      '--kv-fp32 opts out with int8 weights');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--kv-int8'); Args.Add('--fp32');
    Check(ParseArgs(Args, Opt) and Opt.KVInt8,
      'explicit --kv-int8 beats the --fp32 default in any order');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--fp32'); Args.Add('--kv-int8');
    Check(ParseArgs(Args, Opt) and Opt.KVInt8,
      'explicit --kv-int8 beats the --fp32 default in any order (2)');

    // OpenCL offload: --gpu/--cpu toggle, platform/device indices parse.
    // (The default depends on the -dOpenCL build define, so only toggles are
    // asserted here.)
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--gpu');
    Check(ParseArgs(Args, Opt) and Opt.Gpu, '--gpu enables OpenCL offload');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--cpu');
    Check(ParseArgs(Args, Opt) and not Opt.Gpu, '--cpu forces CPU');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--gpu-platform'); Args.Add('1');
    Args.Add('--gpu-device'); Args.Add('2');
    Check(ParseArgs(Args, Opt) and (Opt.GpuPlatform = 1) and (Opt.GpuDevice = 2),
      '--gpu-platform/--gpu-device parse');

    // Parallel layer-graph forward is the default; --serial opts out.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and not Opt.Serial,
      'parallel forward on by default');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--serial');
    Check(ParseArgs(Args, Opt) and Opt.Serial, '--serial parses');

    // KV-cache reuse is on by default; --no-cache-reuse disables it.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and not Opt.NoCacheReuse,
      'cache reuse on by default');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--no-cache-reuse');
    Check(ParseArgs(Args, Opt) and Opt.NoCacheReuse, '--no-cache-reuse parses');

    // --greedy and the explicit-flag trackers.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and not Opt.Greedy, 'greedy off by default');
    Check(not (Opt.TemperatureSet or Opt.TopKSet or Opt.TopPSet or
      Opt.MinPSet or Opt.RepPenaltySet), 'no sampling flag marked set by default');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--greedy');
    Check(ParseArgs(Args, Opt) and Opt.Greedy, '--greedy parses');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--temperature'); Args.Add('0.7');
    Args.Add('--top-p'); Args.Add('0.9');
    Args.Add('--repetition-penalty'); Args.Add('1.1');
    Check(ParseArgs(Args, Opt) and Opt.TemperatureSet and Opt.TopPSet and
      Opt.RepPenaltySet and not Opt.TopKSet and not Opt.MinPSet,
      'explicit sampling flags are tracked');

    // ApplySamplingDefaults precedence: flag > generation_config > fallback.
    // No config, no flags -> the built-in fallback (top-p + mild penalty).
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt), 'defaults fixture parses');
    FillChar(GenCfg, SizeOf(GenCfg), 0);
    ApplySamplingDefaults(Opt, GenCfg);
    Check((Abs(Opt.TopP - csFallbackTopP) < 1e-6) and (Opt.TopK = 0) and
      (Abs(Opt.RepetitionPenalty - csFallbackRepetitionPenalty) < 1e-6) and
      (Abs(Opt.Temperature - 1.0) < 1e-6),
      'no config + no flags -> fallback top-p and repetition penalty');
    // Full config (the Qwen2.5 shape): top_p preferred over top_k,
    // temperature and repetition penalty adopted.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt), 'config fixture parses');
    FillChar(GenCfg, SizeOf(GenCfg), 0);
    GenCfg.Found := true;
    GenCfg.HasTemperature := true; GenCfg.Temperature := 0.7;
    GenCfg.HasTopP := true; GenCfg.TopP := 0.8;
    GenCfg.HasTopK := true; GenCfg.TopK := 20;
    GenCfg.HasRepetitionPenalty := true; GenCfg.RepetitionPenalty := 1.05;
    ApplySamplingDefaults(Opt, GenCfg);
    Check((Abs(Opt.TopP - 0.8) < 1e-6) and (Opt.TopK = 0) and
      (Abs(Opt.Temperature - 0.7) < 1e-6) and
      (Abs(Opt.RepetitionPenalty - 1.05) < 1e-6),
      'config adopted; top_p preferred over top_k');
    // Config with top_k only -> HF-style WEIGHTED top-k.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt), 'top_k fixture parses');
    FillChar(GenCfg, SizeOf(GenCfg), 0);
    GenCfg.Found := true;
    GenCfg.HasTopK := true; GenCfg.TopK := 20;
    ApplySamplingDefaults(Opt, GenCfg);
    Check((Opt.TopK = 20) and Opt.WeightedTopK and (Opt.TopP = 0),
      'config top_k maps to weighted top-k');
    // Explicit flag beats the config.
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--top-p'); Args.Add('0.9');
    Check(ParseArgs(Args, Opt), 'override fixture parses');
    FillChar(GenCfg, SizeOf(GenCfg), 0);
    GenCfg.Found := true;
    GenCfg.HasTopP := true; GenCfg.TopP := 0.8;
    ApplySamplingDefaults(Opt, GenCfg);
    Check(Abs(Opt.TopP - 0.9) < 1e-6, 'explicit --top-p beats the config');
    // do_sample=false -> the author recommends greedy; no fallback sampler.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt), 'do_sample fixture parses');
    FillChar(GenCfg, SizeOf(GenCfg), 0);
    GenCfg.Found := true;
    GenCfg.HasDoSample := true; GenCfg.DoSample := false;
    GenCfg.HasTemperature := true; GenCfg.Temperature := 0.7;
    GenCfg.HasTopP := true; GenCfg.TopP := 0.8;
    ApplySamplingDefaults(Opt, GenCfg);
    Check((Opt.TopP = 0) and (Opt.TopK = 0) and (Opt.MinP = 0) and
      (Abs(Opt.Temperature - 1.0) < 1e-6) and
      (Abs(Opt.RepetitionPenalty - 1.0) < 1e-6),
      'do_sample=false yields greedy defaults');
    // --greedy is a hard override of flags AND config.
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--greedy');
    Args.Add('--top-p'); Args.Add('0.9');
    Args.Add('--temperature'); Args.Add('0.7');
    Args.Add('--repetition-penalty'); Args.Add('1.1');
    Check(ParseArgs(Args, Opt), 'greedy override fixture parses');
    FillChar(GenCfg, SizeOf(GenCfg), 0);
    GenCfg.Found := true;
    GenCfg.HasTopP := true; GenCfg.TopP := 0.8;
    ApplySamplingDefaults(Opt, GenCfg);
    Check((Opt.TopP = 0) and (Opt.TopK = 0) and (Opt.MinP = 0) and
      (Abs(Opt.Temperature - 1.0) < 1e-6) and
      (Abs(Opt.RepetitionPenalty - 1.0) < 1e-6),
      '--greedy overrides flags and config');

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

    // CommonPrefixLen: the KV-cache reuse diff. Drives how much of the prompt
    // is re-prefilled each turn (Reused = matching prefix, capped at Len-1).
    SetLength(PA, 0); SetLength(PB, 0);
    Check(CommonPrefixLen(PA, PB) = 0, 'prefix of two empty arrays is 0');
    SetLength(PA, 4); PA[0] := 1; PA[1] := 2; PA[2] := 3; PA[3] := 4;
    SetLength(PB, 0);
    Check(CommonPrefixLen(PA, PB) = 0, 'prefix with an empty array is 0');
    SetLength(PB, 4); PB[0] := 1; PB[1] := 2; PB[2] := 3; PB[3] := 4;
    Check(CommonPrefixLen(PA, PB) = 4, 'identical arrays: full length');
    PB[0] := 9;
    Check(CommonPrefixLen(PA, PB) = 0, 'divergent at position 0');
    PB[0] := 1; PB[2] := 9;
    Check(CommonPrefixLen(PA, PB) = 2, 'divergent mid-sequence');
    SetLength(PB, 2); PB[0] := 1; PB[1] := 2;
    Check(CommonPrefixLen(PA, PB) = 2, 'one array is a prefix of the other');
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
  Session: TNNetStreamingDecoder;
  Tokenizer: TNeuralHFTokenizer;
  ChatFormat: TNeuralChatFormat;
  History: TChatMessages;
  Msgs: TChatMessages;
  PromptIds, MarkerIds, CachedTokens: TNeuralIntegerArray;
  Chain: TNNetLogitsProcessorChain;
  Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  GenCfg: TGenConfigDefaults;   // model's generation_config.json (if any)
  ReuseOK: boolean;             // KV-cache reuse sound for this architecture?
  {$IFDEF OpenCL}
  GpuCL: TEasyOpenCL;           // platform/device handle for OpenCL offload
  {$ENDIF}
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

  // Default context window. KV-cache streamed decode (below) holds K/V for up
  // to CtxLen tokens PER HEAD PER LAYER, so cache memory grows as O(CtxLen)
  // (not the O(CtxLen^2) score buffers a full-recompute decode would allocate).
  // Using the checkpoint's full max_position_embeddings (32768 for Qwen2.5)
  // would still be large, so when the user gives no --ctx we cap the default at
  // DefaultCtxCap (clamped to the model's own limit). Raise it with --ctx N
  // if you have the RAM (the default int8 weights help there too).
  if Opt.CtxLen <= 0 then
  begin
    Cnt := ReadConfigInt(IncludeTrailingPathDelimiter(Opt.ModelDir) +
      'config.json', 'max_position_embeddings', DefaultCtxCap);
    if (Cnt <= 0) or (Cnt > DefaultCtxCap) then Cnt := DefaultCtxCap;
    Opt.CtxLen := Cnt;
    WriteLn('[context not set - defaulting to ', Opt.CtxLen,
      ' tokens; override with --ctx N (memory grows ~O(ctx^2))]');
  end;

  {$IFDEF OpenCL}
  GpuCL := nil;
  if Opt.Gpu and Opt.LowMemory then
  begin
    WriteLn('[--low-memory ignored: incompatible with --gpu]');
    Opt.LowMemory := false;
  end;
  {$ENDIF}

  // Model: generic architecture dispatch, inference-only, int8 by default.
  // Weight precision. Int8 is the default (less RAM and faster on CPU and
  // GPU); --fp32 opts into full-precision weights (more RAM, slower).
  if Opt.Int8 then
    WriteLn('[int8 weights (default) - less RAM, faster on CPU and GPU;',
      ' on --gpu the codes stay resident on the device; --fp32 opts out]')
  else
    WriteLn('[--fp32: full-precision weights - more RAM, slower than int8]');
  if Opt.LowMemory then
    WriteLn('[low-memory forward (default) - concatenated weight cache dropped,',
      ' per-neuron compute, not compatible with GPU,',
      ' pass --max-fast-memory to keep the (faster) cache and/or use GPU.]')
  else
    WriteLn('[--max-fast-memory: concatenated weight cache kept - faster forward,',
      ' more RAM, GPU compatible]');

  WriteLn('Loading ', Opt.ModelDir, ' ...');
  // Built at INPUT WIDTH 1 (pSeqLen=1): streamed decode feeds one token per
  // forward and the KV cache (budget = CtxLen, set on the session below) holds
  // the context. SeqLen is the cache budget, NOT the built input width.
  NN := BuildFromPretrained(Opt.ModelDir, {pSeqLen=}1,
    {pTrainable=}false, '', {pQuantizeInt8=}Opt.Int8);
  // Low-memory forward path, set independently of trainability. The importer
  // built inference-only with low memory ON (SetTrainable's pLowMemory default);
  // honor --max-fast-memory by re-sweeping the layers, then flush each weight
  // cache so the concatenated-weight cache is (re)built or dropped to match.
  NN.SetTrainable({pTrainable=}false, {pLowMemory=}Opt.LowMemory);
  for Cnt := 0 to NN.GetLastLayerIdx() do
    NN.Layers[Cnt].FlushWeightCache();

  {$IFDEF OpenCL}
  // OpenCL offload of the conv/linear matmuls. Enabling it rebuilds each
  // accelerated layer's concatenated weight cache and turns its low-memory
  // forward path off (the GPU kernel needs the cache), so --gpu effectively
  // overrides --low-memory on those layers. With --int8 the layers instead
  // arm the resident int8 device mode (cai_dot_product_int8): the quantized
  // codes + per-row scales are uploaded once and stay on the device.
  if Opt.Gpu then
  begin
    GpuCL := TEasyOpenCL.Create();
    if GpuCL.GetPlatformCount() = 0 then
    begin
      WriteLn('[--gpu: no OpenCL platform found - falling back to CPU]');
      FreeAndNil(GpuCL);
    end
    else
    begin
      if (Opt.GpuPlatform < 0) or
        (Opt.GpuPlatform >= GpuCL.GetPlatformCount()) then Opt.GpuPlatform := 0;
      GpuCL.SetCurrentPlatform(GpuCL.PlatformIds[Opt.GpuPlatform]);
      if GpuCL.GetDeviceCount() = 0 then
      begin
        WriteLn('[--gpu: no OpenCL device on platform ',
          GpuCL.PlatformNames[Opt.GpuPlatform], ' - falling back to CPU]');
        FreeAndNil(GpuCL);
      end
      else
      begin
        if (Opt.GpuDevice < 0) or
          (Opt.GpuDevice >= GpuCL.GetDeviceCount()) then Opt.GpuDevice := 0;
        GpuCL.SetCurrentDevice(GpuCL.Devices[Opt.GpuDevice]);
        WriteLn('[--gpu: OpenCL on ', GpuCL.PlatformNames[Opt.GpuPlatform],
          ' / ', GpuCL.DeviceNames[Opt.GpuDevice], ']');
        NN.EnableOpenCL(GpuCL.PlatformIds[Opt.GpuPlatform],
          GpuCL.Devices[Opt.GpuDevice]);
      end;
    end;
  end;
  {$ENDIF}

  SeqLen := Opt.CtxLen;
  VocabSize := NN.GetLastLayer().Output.Depth;
  // int8 KV cache is armed at construction so the FP32 K/V buffers are never
  // allocated (a post-Create switch frees them, but the allocator arena may
  // keep the pages). /reset and cache-reuse truncation keep the int8 mode
  // (they only rewind the cache length).
  Session := TNNetStreamingDecoder.Create(NN, SeqLen, Opt.KVInt8);
  if Opt.KVInt8 then
    WriteLn('[int8 KV cache (default with int8 weights) - ~1/4 the KV RAM, ',
      'logits not bit-exact; --kv-fp32 opts out]');
  // Layer-graph parallel forward by default: independent layers of one token
  // step (e.g. an MHA block's sibling heads) run across cores. --serial keeps
  // the classic in-order layer loop. The compute path also drives intra-layer
  // threading automatically: ComputeParallel enables it (big WillThread
  // conv/linear layers above the MinWork threshold split across the pool via
  // worker 0), ComputeSerial runs fully single-threaded. No separate flag.
  Session.Parallel := not Opt.Serial;
  // Keep the scheduler's worker pool alive and HOT between decode steps (default
  // policy: ~50% of the pool hot, worker 0 always) so each token's parallel
  // forward reaches the workers without re-warming the pool every step.
  if Session.Parallel then NN.StartThreadWorkers();
  // KV-cache reuse across turns needs position-truncatable attention K/V and no
  // recurrent (SSM) state to rewind. Pure-attention nets qualify; --no-cache-
  // reuse forces the full re-prefill at the call site.
  ReuseOK := (Session.SSMCount = 0) and (Session.SDPACount > 0);
  SetLength(CachedTokens, 0);
  ModelType := ReadModelType(IncludeTrailingPathDelimiter(Opt.ModelDir) +
    'config.json');
  if ModelType = '' then ModelType := 'unknown';
  WriteLn('Model: ', ModelType, ', ', NN.CountWeights(), ' params, vocab ',
    VocabSize, ', context ', SeqLen, ', chat format ',
    ChatFormatName(ChatFormat), ', ',
    BoolToStr(Opt.Int8, 'int8', 'fp32'), ' weights.');
  if Opt.NoCacheReuse then
    WriteLn('[KV-cache reuse OFF (--no-cache-reuse) - full re-prefill each turn]')
  else if ReuseOK then
    WriteLn('[KV-cache reuse ON - only the new prompt tail is prefilled each turn]')
  else
    WriteLn('[KV-cache reuse N/A for this architecture (recurrent/SSM state)',
      ' - full re-prefill each turn]');
  if Opt.Serial then
    WriteLn('[serial layer loop (--serial) - fully single-threaded]')
  else
    WriteLn('[layer-graph parallel forward (default) - independent layers and',
      ' large conv/linear layers threaded; pass --serial for the serial loop]');

  // Sampling defaults: explicit flag > the model's generation_config.json >
  // built-in fallback (top-p 0.2 + repetition-penalty 1.05); --greedy
  // overrides everything (deterministic argmax, the CPU/GPU parity mode).
  GenCfg := ReadGenerationConfig(
    IncludeTrailingPathDelimiter(Opt.ModelDir) + 'generation_config.json');
  ApplySamplingDefaults(Opt, GenCfg);
  if Opt.Greedy then
    WriteLn('[sampling: greedy argmax (--greedy) - deterministic]')
  else
  begin
    Write('[sampling:');
    if Opt.TopK > 0 then
      Write(' top-k ', Opt.TopK,
        BoolToStr(Opt.WeightedTopK, ' weighted', ' uniform'))
    else if Opt.TopP > 0 then Write(' top-p ', Opt.TopP:0:2)
    else if Opt.MinP > 0 then Write(' min-p ', Opt.MinP:0:2)
    else Write(' greedy argmax');
    if Opt.Temperature <> 1.0 then
      Write(', temperature ', Opt.Temperature:0:2);
    if Opt.RepetitionPenalty <> 1.0 then
      Write(', repetition-penalty ', Opt.RepetitionPenalty:0:2);
    if GenCfg.Found then
      WriteLn(' - flags > generation_config.json > fallback]')
    else
      WriteLn(' - flags > built-in fallback (no generation_config.json)]');
  end;

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
      Reply := GenerateReply(NN, Session, Tokenizer, PromptIds, Opt, SeqLen,
        VocabSize, Chain, Sampler, MarkerIds, CachedTokens,
        {CacheReuse=}ReuseOK and not Opt.NoCacheReuse);
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
  Session.Free; // before NN.Free: Destroy ends incremental decode on NN's layers
  Tokenizer.Free;
  NN.Free;
  {$IFDEF OpenCL}
  GpuCL.Free; // after NN.Free; nil-safe when --gpu was off or fell back to CPU
  {$ENDIF}
end.
