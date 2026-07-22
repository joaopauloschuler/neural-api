unit neuralchatengine;

(*
neuralchatengine
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

// neuralchatengine -- the shared chat-inference engine behind the
// ChatTerminal REPL and the ChatServer HTTP frontend (and any other host
// program that wants "point at a HuggingFace model directory, get replies").
//
// The engine wraps the whole pipeline the frontends share:
//
//   - TChatOptions + ParseArgs: the common command-line surface
//     (model dir, sampling flags, --int8/--fp32, --ctx, --gpu, ...).
//   - Config readers: config.json (model_type, max_position_embeddings)
//     and generation_config.json (the checkpoint author's recommended
//     sampling defaults), plus ApplySamplingDefaults, the per-parameter
//     precedence resolver (explicit flag > generation_config > built-in
//     fallback; --greedy hard-overrides everything).
//   - TChatEngine.LoadModel: tokenizer + chat-format autodetection +
//     BuildFromPretrained (inference-only, int8 by default) + optional
//     OpenCL offload + the TNNetStreamingDecoder KV-cache session +
//     sampling-defaults resolution. Progress/informational lines are
//     emitted through OnNotice (the frontends decide where they go).
//   - TChatEngine.ChatReply / GenerateFromIds: one assistant reply,
//     decoded token by token over the KV cache. Streamed text reaches the
//     host through the OnToken event (ChatTerminal prints it live; a
//     server accumulates it). The sampler / logits-processor chain is
//     built PER CALL from the TChatOptions passed in, so a server can
//     overlay per-request sampling parameters on the launch defaults at
//     zero engine-level cost.
//
// KV-cache reuse across calls: each call diffs its prompt against the
// token ids still resident in the cache (CommonPrefixLen), truncates the
// divergent tail and prefills only the new tokens - so consecutive
// requests that share a prefix (a growing conversation) keep a roughly
// flat time-to-first-token. Pure-attention models only: a recurrent (SSM)
// state cannot be position-truncated, so those fall back to a full
// re-prefill (as does NoCacheReuse).
//
// The engine is single-session: one model, one KV cache, one conversation
// position at a time. Callers that serve concurrent clients must
// serialize calls into it.

{$mode objfpc}{$H+}

interface

uses
  {$IFDEF OpenCL}
  neuralopencl,
  {$ENDIF}
  Classes, SysUtils, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained, neuralhftokenizer,
  neuralchat, neuraldecode;

const
  // Default --ctx when the user gives none. Kept modest because build memory
  // is O(ctx^2) (a SeqLen x SeqLen score buffer per head per layer); the full
  // checkpoint context (e.g. 32768) would OOM at load. See the load path.
  DefaultCtxCap = 2048;

  // Built-in fallback sampling defaults, used only for parameters that neither
  // an explicit flag nor the model's generation_config.json supplies. A tight
  // nucleus + a mild penalty: near-greedy stability, but the penalty prevents
  // the repetition loops pure greedy falls into on small models.
  csFallbackTopP = 0.2;
  csFallbackRepetitionPenalty = 1.05;

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
    NoFusedAttn: boolean;        // --no-fused-attn: build per-head attention
                                 // (SplitChannels/SDPA/DeepConcat) instead of
                                 // the fused TNNetFusedSDPA layer. Bit-identical
                                 // output; a performance A/B knob only.
    Gpu: boolean;                // offload conv/linear matmuls via OpenCL
    GpuPlatform: integer;        // OpenCL platform index (default 0)
    GpuDevice: integer;          // OpenCL device index within the platform (0)
    Host: string;                // ChatServer only: HTTP listen address
    Port: integer;               // ChatServer only: HTTP listen port
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

  // Text sinks. OnToken carries streamed reply text (print it live, or
  // accumulate it); OnNotice carries one informational/status LINE at a
  // time, without a trailing newline (load progress, [bracketed] notices).
  TChatTextEvent = procedure(const S: string) of object;
  TChatNotifyEvent = procedure() of object;

  { TChatEngine }

  TChatEngine = class(TObject)
  public
    // Resolved launch options: LoadModel's copy of the options it was given,
    // after ctx defaulting, GPU fallbacks and ApplySamplingDefaults. The
    // baseline a server overlays per-request parameters onto.
    Opt: TChatOptions;
    NN: TNNet;
    Session: TNNetStreamingDecoder;
    Tokenizer: TNeuralHFTokenizer;
    ChatFormat: TNeuralChatFormat;
    RawMode: boolean;            // FormatName 'raw': no chat template at all
    GenCfg: TGenConfigDefaults;  // model's generation_config.json (if any)
    ReuseOK: boolean;            // KV-cache reuse sound for this architecture?
    SeqLen, VocabSize: integer;
    MarkerIds: TNeuralIntegerArray;   // end-of-turn stop sequence (token ids)
    CachedTokens: TNeuralIntegerArray; // token ids resident in the KV cache
    ModelType: string;           // config.json model_type ('unknown' if none)
    ContextFull: boolean;        // last GenerateFromIds hit the context limit
                                 // (empty reply; the host may want to error)
    // Bookkeeping from the last GenerateFromIds call, for hosts that report
    // usage (the OpenAI response shape): prompt/completion token counts and
    // why decoding stopped ('stop' = EOS or end-of-turn marker, 'length' =
    // the MaxNewTokens cap or the context window).
    LastPromptTokens: integer;
    LastCompletionTokens: integer;
    LastFinishReason: string;
    Loaded: boolean;
    {$IFDEF OpenCL}
    GpuCL: TEasyOpenCL;          // platform/device handle for OpenCL offload
    {$ENDIF}
    OnToken: TChatTextEvent;
    OnNotice: TChatTextEvent;
    // Fired once per successful GenerateFromIds, after the last OnToken and
    // BEFORE the --stats/--profile stderr reports: the moment a terminal
    // frontend should terminate the streamed reply line (the pre-engine
    // ChatTerminal printed its newline exactly here). Not fired on the
    // context-full/empty-prompt early exits. A server leaves it nil.
    OnReplyDone: TChatNotifyEvent;
    constructor Create();
    destructor Destroy(); override;
    // Loads tokenizer + model + KV-cache session from AOpt.ModelDir and
    // resolves the sampling defaults into Opt. Emits progress through
    // OnNotice. False + ErrorMsg on a hard error (no tokenizer, unknown
    // --format name). Call once.
    function LoadModel(const AOpt: TChatOptions; out ErrorMsg: string): boolean;
    // One assistant reply from an assembled message list: renders the chat
    // template, encodes, generates. Raises ENeuralChatError on a template
    // error (e.g. a system prompt on a format without a system role).
    function ChatReply(const Msgs: TChatMessages;
      const GenOpt: TChatOptions): string;
    // One reply from raw prompt token ids (the --format raw completion path;
    // also the primitive ChatReply sits on). GenOpt supplies the sampling
    // parameters for THIS call (pass Opt for the launch defaults).
    function GenerateFromIds(const PromptIds: TNeuralIntegerArray;
      const GenOpt: TChatOptions): string;
  private
    procedure Notice(const S: string);
    procedure EmitToken(const S: string);
  end;

function DefaultChatOptions(): TChatOptions;
function ParseArgs(Args: TStringList; var Opt: TChatOptions): boolean;
// The shared block of the --help text (every option ParseArgs understands
// except the server-only --host/--port, which the server's usage adds).
procedure PrintChatOptionsHelp();
function EndOfTurnMarker(ChatFormat: TNeuralChatFormat): string;
function AssembleMessages(const SystemPrompt: string;
  const History: TChatMessages): TChatMessages;
function ArgMaxRow(Row: TNNetVolume): integer;
function TailMatches(const Tokens: TNeuralIntegerArray; Len: integer;
  const Marker: TNeuralIntegerArray): boolean;
function CommonPrefixLen(const A, B: TNeuralIntegerArray): integer;
function ReadModelType(const ConfigFile: string): string;
function ReadConfigInt(const ConfigFile, Field: string;
  Default: integer): integer;
function ReadGenerationConfig(const FileName: string): TGenConfigDefaults;
procedure ApplySamplingDefaults(var Opt: TChatOptions;
  const Cfg: TGenConfigDefaults);

implementation

procedure PrintChatOptionsHelp();
begin
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
  WriteLn('  --format NAME         chatml|llama2|llama3|zephyr|gemma|phi3|mistral|raw');
  WriteLn('                        raw = no chat template: plain text completion for');
  WriteLn('                        BASE models (gpt2, mamba-130m, ...); the model');
  WriteLn('                        continues a running transcript of what you type.');
  WriteLn('                        No end-of-turn marker - stops on EOS or the');
  WriteLn('                        --max-new-tokens cap (use a small cap, e.g. 128)');
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
  WriteLn('  --no-fused-attn       build per-head attention instead of the fused');
  WriteLn('                        multi-head layer (bit-identical; performance A/B)');
  WriteLn('  --selftest            run the offline unit checks and exit');
  WriteLn('  --help                this text');
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
  Result.NoFusedAttn := false; // fused multi-head attention on by default
  // OpenCL offload defaults ON when the binary is built with -dOpenCL (the
  // default compilation), OFF otherwise; --cpu forces CPU either way.
  Result.Gpu := {$IFDEF OpenCL}true{$ELSE}false{$ENDIF};
  Result.GpuPlatform := 0;
  Result.GpuDevice := 0;
  Result.Host := '127.0.0.1'; // loopback-only by default: a local inference
  Result.Port := 8080;        // server, not an internet-facing one
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
    else if Arg = '--no-fused-attn' then Opt.NoFusedAttn := true
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
    else if Arg = '--host' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.Host := SVal;
    end
    else if Arg = '--port' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      if (IVal < 1) or (IVal > 65535) then
      begin
        Opt.ErrorMsg := '--port: out of range (1-65535): ' + IntToStr(IVal);
        exit(false);
      end;
      Opt.Port := IVal;
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
    cfQwen3_5: Result := '<|im_end|>'; // Qwen3.5/3.6 ChatML variant
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
  Cnt, Ofs, HighH: integer;
begin
  Ofs := 0;
  if SystemPrompt <> '' then Ofs := 1;
  SetLength(Result, Length(History) + Ofs);
  if SystemPrompt <> '' then Result[0] := ChatMessage('system', SystemPrompt);
  HighH := High(History);
  for Cnt := 0 to HighH do Result[Cnt + Ofs] := History[Cnt];
end;

// Stable in-place softmax of a probability row (the imported nets output raw
// logits; the processor chain and the samplers expect POST-SOFTMAX rows) is
// neuralvolume.RowSoftMax.

function ArgMaxRow(Row: TNNetVolume): integer;
var
  Cnt, SizeM1: integer;
  Best: TNeuralFloat;
begin
  Result := 0;
  SizeM1 := Row.Size - 1;
  Best := Row.FData[0];
  for Cnt := 1 to SizeM1 do
    if Row.FData[Cnt] > Best then
    begin
      Best := Row.FData[Cnt];
      Result := Cnt;
    end;
end;

// True when the tail of Tokens[0..Len-1] equals Marker.
function TailMatches(const Tokens: TNeuralIntegerArray; Len: integer;
  const Marker: TNeuralIntegerArray): boolean;
var
  Cnt, MLen, MLenM1, Base: integer;
begin
  MLen := Length(Marker);
  if (MLen = 0) or (Len < MLen) then exit(false);
  MLenM1 := MLen - 1;
  Base := Len - MLen;
  for Cnt := 0 to MLenM1 do
    if Tokens[Base + Cnt] <> Marker[Cnt] then exit(false);
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

{ TChatEngine }

constructor TChatEngine.Create();
begin
  inherited Create();
  Opt := DefaultChatOptions();
  NN := nil;
  Session := nil;
  Tokenizer := nil;
  ChatFormat := cfUnknown;
  RawMode := false;
  ReuseOK := false;
  SeqLen := 0;
  VocabSize := 0;
  SetLength(MarkerIds, 0);
  SetLength(CachedTokens, 0);
  ModelType := '';
  ContextFull := false;
  LastPromptTokens := 0;
  LastCompletionTokens := 0;
  LastFinishReason := '';
  Loaded := false;
  {$IFDEF OpenCL}
  GpuCL := nil;
  {$ENDIF}
  OnToken := nil;
  OnNotice := nil;
  OnReplyDone := nil;
end;

destructor TChatEngine.Destroy();
begin
  FreeAndNil(Session); // before NN.Free: Destroy ends incremental decode on
                       // NN's layers
  FreeAndNil(Tokenizer);
  FreeAndNil(NN);
  {$IFDEF OpenCL}
  FreeAndNil(GpuCL); // after NN.Free; nil when GPU was off or fell back to CPU
  {$ENDIF}
  inherited Destroy();
end;

procedure TChatEngine.Notice(const S: string);
begin
  if Assigned(OnNotice) then OnNotice(S);
end;

procedure TChatEngine.EmitToken(const S: string);
begin
  if Assigned(OnToken) then OnToken(S);
end;

function TChatEngine.LoadModel(const AOpt: TChatOptions;
  out ErrorMsg: string): boolean;
var
  TokenizerFile, TokenizerConfigFile, Marker, Line: string;
  LoadStart: QWord;             // per-phase load wall clock (tokenizer,
                                // checkpoint + caches, GPU weight upload)
  Cnt, LastIdx: integer;
begin
  Result := false;
  ErrorMsg := '';
  Opt := AOpt;

  // Tokenizer + chat format.
  TokenizerFile := IncludeTrailingPathDelimiter(Opt.ModelDir) +
    'tokenizer.json';
  if not FileExists(TokenizerFile) then
  begin
    ErrorMsg := 'No tokenizer.json found in ' + Opt.ModelDir;
    exit;
  end;
  Tokenizer := TNeuralHFTokenizer.Create();
  LoadStart := GetTickCount64();
  Tokenizer.LoadFromFile(TokenizerFile);
  Notice(Format('Tokenizer loaded in %.1fs.',
    [(GetTickCount64() - LoadStart) / 1000]));

  // 'raw' is a frontend-level mode, not a chat template, so it is
  // intercepted here and never reaches ChatFormatFromName (which returns
  // cfUnknown for it). ChatFormat stays cfUnknown in raw mode - that also
  // keeps EndOfTurnMarker() = '' (no stop marker, EOS/cap only).
  ChatFormat := cfUnknown;
  RawMode := LowerCase(Opt.FormatName) = 'raw';
  if RawMode then
  begin
    Notice('[raw completion mode (--format raw) - no chat template; the' +
      ' model continues the transcript; stops on EOS or --max-new-tokens]');
    if Opt.SystemPrompt <> '' then
    begin
      Notice('[--system ignored: no system role in raw mode]');
      Opt.SystemPrompt := '';
    end;
  end
  else if Opt.FormatName <> '' then
  begin
    ChatFormat := ChatFormatFromName(Opt.FormatName);
    if ChatFormat = cfUnknown then
    begin
      ErrorMsg := 'Unknown --format name: ' + Opt.FormatName;
      FreeAndNil(Tokenizer);
      exit;
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
      Notice('[no chat template detected - defaulting to ChatML; override' +
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
    Notice(Format('[context not set - defaulting to %d tokens; override' +
      ' with --ctx N (memory grows ~O(ctx^2))]', [Opt.CtxLen]));
  end;

  {$IFDEF OpenCL}
  GpuCL := nil;
  if Opt.Gpu and Opt.LowMemory then
  begin
    Notice('[--low-memory ignored: incompatible with --gpu]');
    Opt.LowMemory := false;
  end;
  {$ENDIF}

  // Model: generic architecture dispatch, inference-only, int8 by default.
  // Weight precision. Int8 is the default (less RAM and faster on CPU and
  // GPU); --fp32 opts into full-precision weights (more RAM, slower).
  if Opt.Int8 then
    Notice('[int8 weights (default) - less RAM, faster on CPU and GPU;' +
      ' on --gpu the codes stay resident on the device; --fp32 opts out]')
  else
    Notice('[--fp32: full-precision weights - more RAM, slower than int8]');
  if Opt.LowMemory then
    Notice('[low-memory forward (default) - concatenated weight cache dropped,' +
      ' per-neuron compute, not compatible with GPU,' +
      ' pass --max-fast-memory to keep the (faster) cache and/or use GPU.]')
  else
    Notice('[--max-fast-memory: concatenated weight cache kept - faster forward,' +
      ' more RAM, GPU compatible]');

  // Fused multi-head attention A/B (bit-identical output, performance only).
  // The global gates the importer's per-block fused-vs-per-head decision, so
  // it must be set BEFORE BuildFromPretrained. Restored after the build so a
  // second load in the same process is unaffected.
  NeuralAllowFusedAttention := not Opt.NoFusedAttn;
  if Opt.NoFusedAttn then
    Notice('[--no-fused-attn: per-head attention wiring (SplitChannels/SDPA/' +
      'DeepConcat) instead of the fused layer - bit-identical, A/B only]');

  Notice('Loading ' + Opt.ModelDir + ' ...');
  LoadStart := GetTickCount64();
  // Built at INPUT WIDTH 1 (pSeqLen=1): streamed decode feeds one token per
  // forward and the KV cache (budget = CtxLen, set on the session below) holds
  // the context. SeqLen is the cache budget, NOT the built input width.
  NN := BuildFromPretrained(Opt.ModelDir, {pSeqLen=}1,
    {pTrainable=}false, '', {pQuantizeInt8=}Opt.Int8);
  NeuralAllowFusedAttention := true; // restore the global default post-build
  // Low-memory forward path, set independently of trainability. The importer
  // built inference-only with low memory ON (SetTrainable's pLowMemory default);
  // honor --max-fast-memory by re-sweeping the layers, then flush each weight
  // cache so the concatenated-weight cache is (re)built or dropped to match.
  NN.SetTrainable({pTrainable=}false, {pLowMemory=}Opt.LowMemory);
  LastIdx := NN.GetLastLayerIdx();
  for Cnt := 0 to LastIdx do
    NN.Layers[Cnt].FlushWeightCache();
  Notice(Format('Model loaded in %.1fs.',
    [(GetTickCount64() - LoadStart) / 1000]));

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
      Notice('[--gpu: no OpenCL platform found - falling back to CPU]');
      FreeAndNil(GpuCL);
    end
    else
    begin
      if (Opt.GpuPlatform < 0) or
        (Opt.GpuPlatform >= GpuCL.GetPlatformCount()) then Opt.GpuPlatform := 0;
      GpuCL.SetCurrentPlatform(GpuCL.PlatformIds[Opt.GpuPlatform]);
      if GpuCL.GetDeviceCount() = 0 then
      begin
        Notice('[--gpu: no OpenCL device on platform ' +
          GpuCL.PlatformNames[Opt.GpuPlatform] + ' - falling back to CPU]');
        FreeAndNil(GpuCL);
      end
      else
      begin
        if (Opt.GpuDevice < 0) or
          (Opt.GpuDevice >= GpuCL.GetDeviceCount()) then Opt.GpuDevice := 0;
        Notice('[--gpu: OpenCL on ' + GpuCL.PlatformNames[Opt.GpuPlatform] +
          ' / ' + GpuCL.DeviceNames[Opt.GpuDevice] + ']');
        LoadStart := GetTickCount64();
        NN.EnableOpenCL(GpuCL.PlatformIds[Opt.GpuPlatform],
          GpuCL.Devices[Opt.GpuDevice]);
        Notice(Format('GPU weights uploaded in %.1fs.',
          [(GetTickCount64() - LoadStart) / 1000]));
      end;
    end;
  end;
  {$ENDIF}

  SeqLen := Opt.CtxLen;
  VocabSize := NN.GetLastLayer().Output.Depth;
  // int8 KV cache is armed at construction so the FP32 K/V buffers are never
  // allocated (a post-Create switch frees them, but the allocator arena may
  // keep the pages). Reset and cache-reuse truncation keep the int8 mode
  // (they only rewind the cache length).
  Session := TNNetStreamingDecoder.Create(NN, SeqLen, Opt.KVInt8);
  if Opt.KVInt8 then
    Notice('[int8 KV cache (default with int8 weights) - ~1/4 the KV RAM, ' +
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
  // recurrent (SSM) state to rewind. Pure-attention nets qualify; NoCacheReuse
  // forces the full re-prefill at the call site.
  ReuseOK := (Session.SSMCount = 0) and (Session.SDPACount > 0);
  SetLength(CachedTokens, 0);
  ModelType := ReadModelType(IncludeTrailingPathDelimiter(Opt.ModelDir) +
    'config.json');
  if ModelType = '' then ModelType := 'unknown';
  Line := Format('Model: %s, %d params, vocab %d, context %d, chat format ',
    [ModelType, NN.CountWeights(), VocabSize, SeqLen]);
  if RawMode then Line := Line + 'raw (completion)'
  else Line := Line + ChatFormatName(ChatFormat);
  Notice(Line + ', ' + BoolToStr(Opt.Int8, 'int8', 'fp32') + ' weights.');
  if Opt.NoCacheReuse then
    Notice('[KV-cache reuse OFF (--no-cache-reuse) - full re-prefill each turn]')
  else if ReuseOK then
    Notice('[KV-cache reuse ON - only the new prompt tail is prefilled each turn]')
  else
    Notice('[KV-cache reuse N/A for this architecture (recurrent/SSM state)' +
      ' - full re-prefill each turn]');
  if Opt.Serial then
    Notice('[serial layer loop (--serial) - fully single-threaded]')
  else
    Notice('[layer-graph parallel forward (default) - independent layers and' +
      ' large conv/linear layers threaded; pass --serial for the serial loop]');

  // Sampling defaults: explicit flag > the model's generation_config.json >
  // built-in fallback (top-p 0.2 + repetition-penalty 1.05); --greedy
  // overrides everything (deterministic argmax, the CPU/GPU parity mode).
  GenCfg := ReadGenerationConfig(
    IncludeTrailingPathDelimiter(Opt.ModelDir) + 'generation_config.json');
  ApplySamplingDefaults(Opt, GenCfg);
  if Opt.Greedy then
    Notice('[sampling: greedy argmax (--greedy) - deterministic]')
  else
  begin
    Line := '[sampling:';
    if Opt.TopK > 0 then
      Line := Line + Format(' top-k %d%s',
        [Opt.TopK, BoolToStr(Opt.WeightedTopK, ' weighted', ' uniform')])
    else if Opt.TopP > 0 then Line := Line + Format(' top-p %.2f', [Opt.TopP])
    else if Opt.MinP > 0 then Line := Line + Format(' min-p %.2f', [Opt.MinP])
    else Line := Line + ' greedy argmax';
    if Opt.Temperature <> 1.0 then
      Line := Line + Format(', temperature %.2f', [Opt.Temperature]);
    if Opt.RepetitionPenalty <> 1.0 then
      Line := Line + Format(', repetition-penalty %.2f',
        [Opt.RepetitionPenalty]);
    if GenCfg.Found then
      Notice(Line + ' - flags > generation_config.json > fallback]')
    else
      Notice(Line + ' - flags > built-in fallback (no generation_config.json)]');
  end;

  // End-of-turn marker as a token-id stop sequence (single id when the
  // tokenizer has it as an added token, a multi-id sequence otherwise).
  Marker := EndOfTurnMarker(ChatFormat);
  if Marker <> '' then MarkerIds := Tokenizer.Encode(Marker)
  else SetLength(MarkerIds, 0);

  Loaded := true;
  Result := true;
end;

function TChatEngine.ChatReply(const Msgs: TChatMessages;
  const GenOpt: TChatOptions): string;
var
  PromptIds: TNeuralIntegerArray;
begin
  PromptIds := EncodeChat(Tokenizer, ChatFormat, Msgs,
    {AddGenerationPrompt=}true);
  Result := GenerateFromIds(PromptIds, GenOpt);
end;

// ---------------------------------------------------------------------------
// Generation: one assistant reply, streamed through OnToken as it decodes.
// ---------------------------------------------------------------------------
// Full-recompute decode (one fixed-width forward per token, the GPT2Import
// convention). The sampler and logits-processor chain are built HERE, per
// call, from GenOpt (cheap - a handful of tiny objects), so every call can
// carry its own sampling parameters (a server overlays per-request fields
// on the launch defaults in Opt). Probability pipeline per step, matching
// the TGenerationConfig order (penalty -> temperature -> sampler):
//   logits row -> softmax -> Chain.ProcessRow -> Sampler/argmax.
// Stops on EOS (tokenizer's eos id), on the end-of-turn marker token
// sequence, or after GenOpt.MaxNewTokens. Returns the decoded reply (marker
// trimmed); streamed emission fires after every token so the host can print
// live. The host prints its own trailing newline - the reply text is exactly
// what the model produced.
// Cache reuse (ReuseOK and not GenOpt.NoCacheReuse): keep the KV cache
// across calls and only prefill the tail that diverges from CachedTokens
// (the token-id sequence currently resident in the cache, updated here).
// Otherwise the cache is fully reset and the whole prompt re-prefilled (the
// SSM/recurrent path, where the cache cannot be truncated by position, and
// NoCacheReuse).
function TChatEngine.GenerateFromIds(const PromptIds: TNeuralIntegerArray;
  const GenOpt: TChatOptions): string;
var
  Chain: TNNetLogitsProcessorChain;
  Penalty: TNNetTokenHistoryPenalty;
  Sampler: TNNetSamplerBase;
  CacheReuse: boolean;
  Tokens: TNeuralIntegerArray;
  Generated: TNeuralIntegerArray;
  InV, Output, Row: TNNetVolume;
  Len, GenLen, StepCnt, Cnt, NewToken: integer;
  Reused, PromptLen: integer;  // KV-cache reuse bookkeeping (and --stats)
  LenM1, LenM2, MarkerLen, EmLen, DecLen: integer;
  LastPos, RowBytes: integer;
  Decoded, Emitted: string;
  // --stats timing (monotonic ms). TStart: before prefill; TFirst: when the
  // first reply token is produced (so TTFT covers prefill + first step);
  // TEnd: after the decode loop. Produced counts emitted tokens.
  TStart, TFirst, TEnd: QWord;
  Produced: integer;
  DecodeSecs: double;
begin
  Result := '';
  if not Loaded then
    raise Exception.Create('TChatEngine.GenerateFromIds before LoadModel');
  ContextFull := false;
  LastPromptTokens := Length(PromptIds);
  LastCompletionTokens := 0;
  LastFinishReason := 'length';
  Len := Length(PromptIds);
  // An empty prompt has no last token to feed as the first decode step's
  // input (a BOS-less tokenizer encodes '' to zero ids): decoding cannot
  // start. Reject here - proceeding would index Tokens[-1] and feed the
  // session a negative position.
  if Len = 0 then
  begin
    Notice('[empty prompt - nothing to decode]');
    exit;
  end;
  if Len >= SeqLen then
  begin
    ContextFull := true;
    Notice(Format('[context window full (%d >= %d tokens) - /reset the' +
      ' conversation or rebuild with a larger --ctx]', [Len, SeqLen]));
    exit;
  end;
  CacheReuse := ReuseOK and not GenOpt.NoCacheReuse;
  // Distribution pipeline (TGenerationConfig order: penalty -> temperature
  // -> sampler).
  Chain := TNNetLogitsProcessorChain.Create();
  Penalty := nil;
  Sampler := nil;
  if (GenOpt.RepetitionPenalty <> 1.0) or (GenOpt.FrequencyPenalty <> 0) or
    (GenOpt.PresencePenalty <> 0) then
  begin
    Penalty := TNNetTokenHistoryPenalty.Create(GenOpt.RepetitionPenalty,
      GenOpt.FrequencyPenalty, GenOpt.PresencePenalty);
    Chain.Add(TNNetPenaltyProcessor.Create(Penalty, {OwnsPenalty=}true),
      {OwnsProcessor=}true);
  end;
  if GenOpt.Temperature <> 1.0 then
    Chain.Add(TNNetTemperatureProcessor.Create(GenOpt.Temperature), true);
  if GenOpt.TopK > 0 then
  begin
    if GenOpt.WeightedTopK then
      Sampler := TNNetSamplerWeightedTopK.Create(GenOpt.TopK)
    else Sampler := TNNetSamplerTopK.Create(GenOpt.TopK);
  end
  else if GenOpt.TopP > 0 then Sampler := TNNetSamplerTopP.Create(GenOpt.TopP)
  else if GenOpt.MinP > 0 then Sampler := TNNetSamplerMinP.Create(GenOpt.MinP);
  SetLength(Tokens, SeqLen);
  LenM1 := Len - 1;
  LenM2 := Len - 2;
  MarkerLen := Length(MarkerIds);
  if Len > 0 then Move(PromptIds[0], Tokens[0], Len * csIntegerSize);
  SetLength(Generated, 0);
  Emitted := '';
  InV := TNNetVolume.Create(1, 1, 1);
  Output := nil; // a reference into the net, returned by Session.Output()
  Row := TNNetVolume.Create(VocabSize, 1, 1);
  TStart := GetTickCount64();
  TFirst := 0;
  TEnd := 0;
  Produced := 0;
  PromptLen := Len;
  // Outer except: past this point the session's KV cache is mutated
  // (TruncateTo + partial prefill) while CachedTokens is only updated on
  // success. A long-lived host that catches the exception and keeps serving
  // (the server's 500 path) would reuse poisoned cache positions on the
  // next call, so invalidate both before re-raising.
  try
  try
    Chain.Reset(PromptIds);
    // Prefill the prompt token-at-a-time, reusing the KV-cache prefix shared
    // with the last call when possible. Reused = length of the cached prefix
    // that still matches this prompt; TruncateTo drops the divergent tail
    // (Reused=0 is a full reset). The LAST prompt token is fed as the first
    // decode step's input, so the cache must not already hold it - cap reuse
    // at Len-1.
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
    for Cnt := Reused to LenM2 do
    begin
      InV.FData[0] := Tokens[Cnt];
      Session.StepForward(InV, Cnt);
    end;
    // --profile: discard the one-shot prefill timings (and scheduler stats)
    // so the per-layer-class report below reflects only the repeated
    // single-token decode steps - the steady-state workload whose layer costs
    // we want to rank for optimization.
    if GenOpt.Profile then
    begin
      NN.ClearTime();
      NN.ResetSchedulerStats();
    end;
    RowBytes := VocabSize * csNeuralFloatSize;
    for StepCnt := 1 to GenOpt.MaxNewTokens do
    begin
      if Len >= SeqLen then break;
      // One width-1 forward of the last committed token over the cached past.
      LastPos := Len - 1;
      InV.FData[0] := Tokens[LastPos];
      Session.StepForward(InV, LastPos);
      Output := Session.Output(); // (1,1,vocab) -- the single logits row
      Move(Output.FData[0], Row.FData[0], RowBytes);
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
      // EOS / end-of-turn checks BEFORE emitting so markers never echo.
      if (Tokenizer.EosId >= 0) and (NewToken = Tokenizer.EosId) then
      begin
        LastFinishReason := 'stop';
        break;
      end;
      if TailMatches(Generated, GenLen + 1, MarkerIds) then
      begin
        SetLength(Generated, GenLen + 1 - MarkerLen);
        LastFinishReason := 'stop';
        break;
      end;
      // Streamed emission: decode the whole generated region and emit the
      // delta (BPE merges/UTF-8 multibyte pieces can rewrite the tail, so
      // only emit when the previous text is still a prefix).
      Decoded := Tokenizer.Decode(Generated, {SkipSpecialTokens=}true);
      DecLen := Length(Decoded);
      EmLen := Length(Emitted);
      if (DecLen > EmLen) and
        (Copy(Decoded, 1, EmLen) = Emitted) then
      begin
        EmitToken(Copy(Decoded, EmLen + 1, DecLen - EmLen));
        Emitted := Decoded;
      end;
    end;
    LastCompletionTokens := Produced;
    Result := Tokenizer.Decode(Generated, {SkipSpecialTokens=}true);
    // Anything the prefix-guard held back (or trimmed markers shortened).
    if (Length(Result) > Length(Emitted)) and
      (Copy(Result, 1, Length(Emitted)) = Emitted) then
      EmitToken(Copy(Result, Length(Emitted) + 1,
        Length(Result) - Length(Emitted)));
    if Assigned(OnReplyDone) then OnReplyDone();
    // Record the sequence now resident in the cache for the next call's
    // prefix diff: every token that was FED is cached (positions 0..Len-2);
    // the final produced token (Tokens[Len-1]) was sampled but never fed, so
    // it is not.
    SetLength(CachedTokens, Len - 1);
    if Len > 1 then Move(Tokens[0], CachedTokens[0], (Len - 1) * csIntegerSize);
    // Per-turn timing to stderr (keeps stdout = pure model output). TTFT =
    // prefill + first decode step; tok/s measures the steady-state decode of
    // the tokens AFTER the first, so prefill cost is excluded. prompt N (reused
    // K) shows how much of the prompt the KV-cache reuse skipped re-prefilling.
    if GenOpt.Stats and (Produced > 0) then
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
    // --profile: per-layer-class forward timing accumulated over this call's
    // decode steps (prefill was cleared above). Printed to stderr so stdout
    // stays pure model output. Ranks layer classes by aggregate forward cost -
    // the actionable signal for picking the next class to optimize (e.g. OpenCL).
    if GenOpt.Profile and (Produced > 0) then
    begin
      WriteLn(StdErr);
      Write(StdErr, TNNet.LayerClassTimingReport(NN));
      // Layer-graph scheduler parallelism for this call's decode steps: how
      // wide the graph is, how often the parallel path ran vs the serial
      // fallback, and how much overlap it actually achieved (peak in-flight,
      // share of layers computed off the primary worker).
      WriteLn(StdErr, '[sched] ', NN.SchedulerStatsReport());
      Flush(StdErr);
    end;
  finally
    Row.Free;
    InV.Free;
    Sampler.Free;
    Chain.Free; // owns the processors, which own the penalty
  end;
  except
    SetLength(CachedTokens, 0);
    Session.Reset();
    raise;
  end;
end;

end.
