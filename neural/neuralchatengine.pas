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
// flat time-to-first-token. A recurrent (SSM) state cannot be
// position-truncated, so a hybrid/recurrent net resumes instead from the
// deepest cache checkpoint (the recurrent half of the state, captured at
// known positions during the prefill and at the turn boundaries) at or
// below that prefix; --cache-checkpoints N sizes that store. NoCacheReuse
// turns both routes off (full re-prefill).
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
  Classes, SysUtils, Math, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained, neuralhftokenizer,
  neuralchat, neuraldecode;

const
  // Default --ctx when the user gives none, clamped to the checkpoint's own
  // limit. Decode is streamed at input width 1, so the cost of context is the
  // KV cache: LINEAR in ctx, but preallocated in full when the session opens
  // (see TNNetScaledDotProductAttention.BeginIncrementalDecode). Per token
  // that is 2 * kv_heads * head_dim * 4 bytes per layer (a quarter of that
  // under --kv-int8) plus an fp32 q_heads-wide score scratch, so a 7B at this
  // cap holds a few GB of cache. Checkpoints declaring more (Llama-3 at
  // 131072) need --ctx to go higher.
  DefaultCtxCap = 32768;

  // Built-in fallback sampling defaults, used only for parameters that neither
  // an explicit flag nor the model's generation_config.json supplies. A tight
  // nucleus + a mild penalty: near-greedy stability, but the penalty prevents
  // the repetition loops pure greedy falls into on small models.
  csFallbackTopP = 0.2;
  csFallbackRepetitionPenalty = 1.05;

  // Default width of the --prefill-window tail twin (--prefill-tail-window 0).
  // After the width-N windows at most T-1 tokens are left for the width-1
  // single steps, the twin costs its activations only, and a 16-wide window
  // is still a full launch of every kernel on the device.
  csDefaultPrefillTailWindow = 16;
  // --cache-checkpoints N: the store's size when the flag is not given, its
  // cap, and the retention band width W when no prefill twin sets it.
  csDefaultCacheCheckpointsOpenCL = 16;
  csDefaultCacheCheckpointsCPU = 8;
  csMaxCacheCheckpoints = 2048;
  csDefaultCheckpointBandWidth = 256;

type
  // Weight storage the chat engine loads the checkpoint into.
  // cwmInt4 is int4 on the convolution/projection layers, int8 elsewhere.
  TChatWeightMode = (cwmFP32, cwmInt8, cwmInt4);

  TChatOptions = record
    ModelDir: string;
    WeightMode: TChatWeightMode; // --fp32 / --int8 (default) / --int4
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
    // How much reasoning the chat template asks for. Only the formats with a
    // reasoning control react (FormatHasReasoningControl): Qwen3.8 to all
    // four values, Qwen3.5/3.6 to reOff. reXHigh is the HF default of both.
    ReasoningEffort: TChatReasoningEffort;
    SystemPrompt: string;
    Prompt: string;              // ChatTerminal only: -p "text" runs this one
                                 // prompt and exits instead of opening the REPL
                                 // (single turn, no interactive input)
    SelfTest: boolean;
    ShowHelp: boolean;
    Stats: boolean;              // per-turn timing to stderr (TTFT, tok/s)
    Profile: boolean;            // per-layer-class forward timing to stderr after
                                 // each turn, prefill and decode reported apart;
                                 // for picking the next layer class to optimize
    NoCacheReuse: boolean;       // force full re-prefill every turn (A/B + debug)
    CacheCheckpoints: integer;   // --cache-checkpoints N: hybrid/recurrent nets
                                 // keep up to N recurrent-state checkpoints to
                                 // resume a prompt from. -1 = not given
                                 // (LoadModel picks 16 under OpenCL, 8 on the
                                 // CPU); 0 = off (full re-prefill); 1 and
                                 // above 2048 are errors
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
    MaxThreads: integer;         // cap on scheduler worker threads
                                 // (TNNet.MaxThreadNum); 0 = all CPU threads.
                                 // The pool is Min(MaxThreads, cpu count), and
                                 // per-layer chunk counts follow it
    NoFusedAttn: boolean;        // --no-fused-attn: build per-head attention
                                 // (SplitChannels/SDPA/DeepConcat) instead of
                                 // the fused TNNetFusedSDPA layer. Bit-identical
                                 // output; a performance A/B knob only.
    Gpu: boolean;                // offload conv/linear matmuls via OpenCL
    GpuPlatform: integer;        // OpenCL platform index (default 0)
    GpuDevice: integer;          // OpenCL device index within the platform (0)
    GpuSharedKernel: boolean;    // one net-wide OpenCL program/kernel cache
                                 // shared by every layer (DEFAULT). False gives
                                 // each layer its own kernel handles and command
                                 // queue - measurably slower here, kept as an
                                 // A/B knob for drivers that dislike sharing
    ExperimentalFP16: boolean;   // --experimental-fp16: half-precision B operand for the
                                 // int8 OpenCL matmuls (TNNet.OpenCLFP16).
                                 // Weights stay int8 and the logits are not
                                 // bit-exact. Needs OpenCL AND int8 weights:
                                 // TNNetConvolutionBase.ShouldOpenCLFP16 tests
                                 // FQuantInt8, so --fp32 and --int4 ignore it
    ExperimentalInt8Input: boolean; // --experimental-int8-input: TNNet.EnableInt8Input
                                 // after the weights are int8. Today only a
                                 // TNNetConvolution has an int8 x int8 kernel;
                                 // the LLM blocks arm an input copy nothing
                                 // reads yet. Needs int8 or int4 weights
    PrefillWindow: integer;      // --prefill-window N: prefill the prompt N
                                 // tokens per forward on a width-N twin of
                                 // the net that borrows its weights (the
                                 // twin costs its activations; a non-Llama
                                 // family falls back to a full second
                                 // build); 0 = one token per forward
    PrefillTailWindow: integer;  // --prefill-tail-window T: a second, width-T
                                 // twin feeds the leftover of the width-N
                                 // windows T tokens per forward. 0 = auto
                                 // (csDefaultPrefillTailWindow), 1 = none;
                                 // LoadModel resolves it (1 when no tail
                                 // twin was built)
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
    // --prefill-window N: a width-N twin of NN with its own session. Whole
    // windows of the prompt go through WindowSession, what they leave over
    // goes through the width-T tail twin (TailSession), and SwitchTo carries
    // the state to Session before the single steps and the decode loop. All
    // nil when the option is 0 (the tail trio also when no tail twin was
    // built). WindowIn / TailIn are the (N,1,1) / (T,1,1) windows of ids.
    WindowNN: TNNet;
    WindowSession: TNNetStreamingDecoder;
    WindowIn: TNNetVolume;
    TailNN: TNNet;
    TailSession: TNNetStreamingDecoder;
    TailIn: TNNetVolume;
    // The one snapshot SwitchTo captures into (SnapshotInto reuses its
    // volumes), allocated with the twins; nil without them.
    TransferSnap: TNNetDecoderSessionSnapshot;
    // True when WindowNN borrows NN's weights (BuildFromPretrained with
    // pWeightOwner: the checkpoint is read once and the twin holds no weight
    // storage); false on the non-Llama fallback, a full second build.
    WindowBorrowsWeights: boolean;
    ActiveSession: TNNetStreamingDecoder; // the session holding the live state
    Tokenizer: TNeuralHFTokenizer;
    ChatFormat: TNeuralChatFormat;
    RawMode: boolean;            // FormatName 'raw': no chat template at all
    GenCfg: TGenConfigDefaults;  // model's generation_config.json (if any)
    ReuseOK: boolean;            // KV-cache reuse sound for this architecture?
    // Cache checkpoints, the hybrid/recurrent counterpart of ReuseOK. A
    // recurrent layer's state has no per-position history to truncate, so a
    // hybrid resumes a prompt from a CHECKPOINT: the recurrent half of the
    // session state captured at a known fed position
    // (TNNetDecoderStateCheckpoint; no attention K/V, Session.TruncateTo
    // rewinds that half in place). Resuming at the checkpoint's position is
    // bit-identical to a full re-prefill. The store holds up to
    // Opt.CacheCheckpoints of them, sized once in LoadModel so that a capture
    // allocates nothing; Position 0 marks a free slot. CaptureCheckpoint runs
    // after every prefill window, at the end of the prompt and at the end of
    // the reply; RetainCheckpointsBefore keeps one checkpoint per geometric
    // band of distance from the fed position (CheckpointBand), so the store
    // is densest near the newest token, where prompts diverge most often.
    // Under OpenCL every slot lives in OpenCL memory (a capture is a copy
    // between resident buffers), else in host RAM: Checkpoints[0].Bytes() /
    // OpenCLBytes() say how much.
    StateReuseOK: boolean;       // checkpoint resume sound for this architecture?
    Checkpoints: array of TNNetDecoderStateCheckpoint; // owned; empty when off
    CheckpointBandWidth: integer; // W: the finest capture spacing (the tail
                                 // window, else the prefill window, else
                                 // csDefaultCheckpointBandWidth capped at
                                 // half the context)
    CheckpointBandRatio: double; // r = (SeqLen / W)^(1 / N)
    CheckpointOnTwins: boolean;  // the twins' captures are legal (they share
                                 // NN's OpenCL context, or OpenCL is off)
    CheckpointBandDeepest: array of integer; // RetainCheckpointsBefore's
                                 // per-band slot index, sized with the store
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
    LastReusedTokens: integer;   // prompt tokens the last call resumed from
                                 // the cache or a checkpoint (the (reused K)
                                 // of --stats)
    LastPrefixTokens: integer;   // token ids the last call's prompt shared
                                 // with CachedTokens before any capping or
                                 // checkpoint choice (the (prefix P of ...))
    LastCachedTokens: integer;   // Length(CachedTokens) when the last call
                                 // started (the (... of C cached))
    LastPrefillTokens: integer;  // prompt tokens the last call fed
    LastPrefillWindows: integer; // whole windows the last call fed through
                                 // WindowSession (0 without --prefill-window)
    LastPrefillTailWindows: integer; // whole windows the last call fed
                                 // through TailSession (0 without a tail twin)
    // Lifetime totals over every GenerateFromIds call of this engine, for a
    // host that reports usage for as long as it runs (the [stats] totals
    // lines). Input = prompt tokens and the prefill time; cached = the part
    // of the prompt the KV-cache/checkpoint reuse skipped; output = reply
    // tokens and the time from the end of prefill to the end of decode.
    TotalInputTokens: Int64;
    TotalCachedInputTokens: Int64;
    TotalInputMs: double;
    TotalOutputTokens: Int64;
    TotalOutputMs: double;
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
    // CheckpointBand maps a distance from the fed position to its retention
    // band 0..N-1 (the rule RetainCheckpointsBefore keeps one checkpoint per).
    function CheckpointBand(Distance: integer): integer;
  private
    // Bytes of an unfinished UTF-8 sequence EmitToken is holding until the
    // token that completes it arrives (a codepoint can straddle two tokens).
    PendingUtf8: string;
    procedure Notice(const S: string);
    procedure EmitToken(const S: string);
    // Emits U+FFFD for anything still held in PendingUtf8 at reply end.
    procedure FlushPendingUtf8();
    // Cache checkpoints (see the Checkpoints field).
    // The live checkpoint with the largest Position at or below Limit; nil
    // when none (a full reset follows).
    function DeepestCheckpointAtOrBelow(Limit: integer):
      TNNetDecoderStateCheckpoint;
    // Frees every checkpoint whose position the fed sequence no longer
    // vouches for (Position above Limit); 0 empties the store.
    procedure DropCheckpointsAbove(Limit: integer);
    // Applies the retention rule against a capture about to land at FedPos
    // and leaves at least one free slot (nothing is allocated).
    procedure RetainCheckpointsBefore(FedPos: integer);
    // Copies ASession's recurrent state into a free slot at FedPos, the
    // number of tokens fed into ASession; skipped when FedPos is held already.
    procedure CaptureCheckpoint(ASession: TNNetStreamingDecoder;
      FedPos: integer);
    procedure FreeCheckpoints();
    // Moves the live state from ActiveSession into Target (snapshot, restore)
    // and makes Target the active session. No-op when Target is active.
    procedure SwitchTo(Target: TNNetStreamingDecoder);
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
// Number of trailing bytes of S that open a UTF-8 sequence S has not
// finished (0 when S ends on a whole codepoint, on ASCII, or on bytes no
// sequence could still complete).
function Utf8IncompleteTailLen(const S: string): integer;
// Streaming helper: Pending + S with any unfinished trailing UTF-8 sequence
// held back into Pending. What is returned is always whole codepoints, so a
// client decoding every chunk on its own never sees half a character.
function TakeCompleteUtf8(var Pending: string; const S: string): string;
function ReadModelType(const ConfigFile: string): string;
function ReadConfigInt(const ConfigFile, Field: string;
  Default: integer): integer;
function ReadGenerationConfig(const FileName: string): TGenConfigDefaults;
procedure ApplySamplingDefaults(var Opt: TChatOptions;
  const Cfg: TGenConfigDefaults);

const
  // U+FFFD, emitted for bytes that never completed a sequence.
  Utf8ReplacementChar = #$EF#$BF#$BD;

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
  WriteLn('  --ctx N               context window (default min(model max,32768); KV RAM ~O(ctx))');
  WriteLn('  --format NAME         chatml|qwen|qwen3_5|qwen3_8|llama2|llama3|zephyr|gemma|');
  WriteLn('                        phi3|mistral|deepseek|phi4mini|llava|raw');
  WriteLn('                        raw = no chat template: plain text completion for');
  WriteLn('                        BASE models (gpt2, mamba-130m, ...); the model');
  WriteLn('                        continues a running transcript of what you type.');
  WriteLn('                        No end-of-turn marker - stops on EOS or the');
  WriteLn('                        --max-new-tokens cap (use a small cap, e.g. 128)');
  WriteLn('  --reasoning-effort E  off|low|medium|xhigh (default xhigh). Qwen3.8 turns');
  WriteLn('                        this into its reasoning_effort system instruction;');
  WriteLn('                        Qwen3.5/3.6 only honour off (thinking disabled);');
  WriteLn('                        every other format ignores it');
  WriteLn('  --system "msg"        initial system prompt');
  WriteLn('  --int8                int8 weight-only quantized inference (DEFAULT; less');
  WriteLn('                        RAM and faster on CPU and GPU: resident int8 codes)');
  WriteLn('  --fp32                full-precision weights (more RAM, slower)');
  WriteLn('  --int4                int4 (Q4_0, blocks of 32) weights on the');
  WriteLn('                        convolution/projection layers, int8 elsewhere;');
  WriteLn('                        half the weight RAM of --int8; CPU path;');
  WriteLn('                        output quality below --int8');
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
  WriteLn('  --experimental-fp16   under construction: half-precision activations for');
  WriteLn('                        the int8 OpenCL matmuls (weights stay int8; logits');
  WriteLn('                        not bit-exact). Needs --gpu and int8 weights;');
  WriteLn('                        ignored otherwise (--fp32, --int4)');
  WriteLn('  --experimental-int8-input  under construction: int8-quantized activations');
  WriteLn('                        (one scale per tensor) feeding the int8 weights on the');
  WriteLn('                        CPU convolution path; other layers arm the copy but');
  WriteLn('                        still run int8 x FP32. Needs int8 or int4');
  WriteLn('                        weights; ignored with --fp32');
  WriteLn('  --no-gpu-shared-kernel  give each layer private OpenCL kernels and command');
  WriteLn('                        queue instead of the net-wide shared ones (default:');
  WriteLn('                        shared, which is faster). Each layer then waits for');
  WriteLn('                        its sources, so --profile charges GPU time per layer');
  WriteLn('                        instead of the queue drain: a profiling mode.');
  WriteLn('  --stats               per-turn timing to stderr: input (prompt, TTFT,');
  WriteLn('                        prefill tok/s), output (tokens, time, decode tok/s),');
  WriteLn('                        per-step split, and lifetime input/output totals');
  WriteLn('  --profile             per-layer-class forward timing to stderr after each');
  WriteLn('                        turn, one report for the prefill and one for the');
  WriteLn('                        decode steps; ranks classes to optimize.');
  WriteLn('                        Also prints [sched]: layer-graph parallelism (graph');
  WriteLn('                        width, parallel vs serial passes, peak in-flight)');
  WriteLn('  --no-cache-reuse      re-prefill the whole prompt each turn (default:');
  WriteLn('                        reuse the shared KV-cache prefix from last turn)');
  WriteLn('  --cache-checkpoints N  hybrid/recurrent nets (qwen3_5, mamba, ...): keep');
  WriteLn('                        up to N checkpoints of the recurrent state, taken');
  WriteLn('                        after every prefill window and at the end of the');
  WriteLn('                        prompt and of the reply, kept geometrically denser');
  WriteLn('                        near the end; a prompt resumes from the deepest one');
  WriteLn('                        at or below its shared token prefix and prefills');
  WriteLn('                        only the tail (default 16 with OpenCL, 8 on the');
  WriteLn('                        CPU; 0 = off, full re-prefill; N is 0 or 2 to');
  WriteLn('                        2048, else the program stops with an error before');
  WriteLn('                        loading). Ignored on pure-attention nets, whose');
  WriteLn('                        KV cache is truncated to the prefix instead');
  WriteLn('  --prefill-window N    prefill the prompt N tokens per forward on a width-N');
  WriteLn('                        twin of the net (default 0 = one token per forward;');
  WriteLn('                        N is 0 or at least 2 and below the context, else');
  WriteLn('                        the program stops with an error before loading).');
  WriteLn('                        The twin shares the loaded weights (in RAM and on');
  WriteLn('                        the device) and the checkpoint is read once: it');
  WriteLn('                        costs its activations only. Model families outside');
  WriteLn('                        the Llama builder (llama, mistral, qwen*, gemma*,');
  WriteLn('                        phi3, ...) fall back to a full second build, which');
  WriteLn('                        the startup notice says. What the windows leave');
  WriteLn('                        over goes through the tail twin (below), then one');
  WriteLn('                        token at a time; nothing is padded');
  WriteLn('  --prefill-tail-window T  width of a second twin that feeds the leftover');
  WriteLn('                        of the width-N windows T tokens per forward, so at');
  WriteLn('                        most T-1 tokens go one at a time (default 0 = auto,');
  WriteLn('                        ' + IntToStr(csDefaultPrefillTailWindow) +
    ' when that is below N; 1 = no tail twin). T must be');
  WriteLn('                        below N and needs --prefill-window, else the program');
  WriteLn('                        stops with an error. Shares the weights like the width-N twin;');
  WriteLn('                        not built on the full-second-build fallback');
  WriteLn('  --serial              serial layer loop (default: layer-graph parallel');
  WriteLn('                        forward across independent layers; the parallel');
  WriteLn('                        path also threads large conv/linear layers');
  WriteLn('                        internally, --serial runs fully single-threaded)');
  WriteLn('  --max-threads N       cap the parallel forward at N worker threads');
  WriteLn('                        (default: every CPU thread). Fewer threads can');
  WriteLn('                        be faster when the GPU/OpenCL device is busy or');
  WriteLn('                        the machine is shared; ignored with --serial');
  WriteLn('  --no-fused-attn       build per-head attention instead of the fused');
  WriteLn('                        multi-head layer (bit-identical; performance A/B)');
  WriteLn('  --selftest            run the offline unit checks and exit');
  WriteLn('  --help                this text');
end;

// 'fp32' | 'int8' | 'int4', the name of the flag that selects the mode.
function ChatWeightModeName(WeightMode: TChatWeightMode): string;
begin
  case WeightMode of
    cwmFP32: Result := 'fp32';
    cwmInt4: Result := 'int4';
    else Result := 'int8';
  end;
end;

function DefaultChatOptions(): TChatOptions;
begin
  Result.ModelDir := '';
  Result.WeightMode := cwmInt8; // less RAM, faster; --fp32 and --int4 opt out
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
  Result.ReasoningEffort := reXHigh;
  Result.SystemPrompt := '';
  Result.Prompt := ''; // '' = interactive REPL; -p "text" runs one turn
  Result.SelfTest := false;
  Result.ShowHelp := false;
  Result.Stats := false;
  Result.Profile := false;
  Result.NoCacheReuse := false;
  Result.CacheCheckpoints := -1; // not given: LoadModel picks 16/8 (OpenCL/CPU)
  Result.PrefillWindow := 0; // one token per prefill forward (--prefill-window N)
  Result.PrefillTailWindow := 0; // auto (--prefill-tail-window T)
  Result.KVInt8 := false;    // resolved after parsing: follows the weight mode
  Result.KVInt8Set := false; // unless --kv-int8/--kv-fp32 picked explicitly
  Result.Serial := false; // parallel layer-graph forward by default (--serial)
  Result.MaxThreads := 0; // 0 = every CPU thread (--max-threads N caps it)
  Result.NoFusedAttn := false; // fused multi-head attention on by default
  // OpenCL offload defaults ON when the binary is built with -dOpenCL (the
  // default compilation), OFF otherwise; --cpu forces CPU either way.
  Result.Gpu := {$IFDEF OpenCL}true{$ELSE}false{$ENDIF};
  Result.GpuPlatform := 0;
  Result.GpuDevice := 0;
  Result.GpuSharedKernel := true; // shared kernels/queue (--no-gpu-shared-kernel)
  Result.ExperimentalFP16 := false; // FP32 activations; --experimental-fp16 opts into the halves
  Result.ExperimentalInt8Input := false; // FP32 activations; --experimental-int8-input opts into int8
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
    if Arg = '--int8' then Opt.WeightMode := cwmInt8
    else if Arg = '--fp32' then Opt.WeightMode := cwmFP32
    else if Arg = '--int4' then Opt.WeightMode := cwmInt4
    else if Arg = '--low-memory' then Opt.LowMemory := true
    else if Arg = '--max-fast-memory' then Opt.LowMemory := false
    else if Arg = '--stats' then Opt.Stats := true
    else if Arg = '--profile' then Opt.Profile := true
    else if Arg = '--no-cache-reuse' then Opt.NoCacheReuse := true
    else if Arg = '--cache-checkpoints' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      if IVal < 0 then
      begin
        Opt.ErrorMsg := '--cache-checkpoints: must be 0 (off) or a count of' +
          ' checkpoints to keep';
        exit(false);
      end;
      Opt.CacheCheckpoints := IVal;
    end
    else if Arg = '--prefill-window' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      if (IVal < 0) or (IVal = 1) then
      begin
        Opt.ErrorMsg := '--prefill-window: must be 0 (off) or at least 2';
        exit(false);
      end;
      Opt.PrefillWindow := IVal;
    end
    else if Arg = '--prefill-tail-window' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      if IVal < 0 then
      begin
        Opt.ErrorMsg := '--prefill-tail-window: must be 0 (auto), 1 (none)' +
          ' or a width below --prefill-window';
        exit(false);
      end;
      Opt.PrefillTailWindow := IVal;
    end
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
    else if Arg = '--max-threads' then
    begin
      if not NextInt(Arg, IVal) then exit(false);
      if IVal < 1 then
      begin
        Opt.ErrorMsg := '--max-threads: must be at least 1';
        exit(false);
      end;
      Opt.MaxThreads := IVal;
    end
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
    else if Arg = '--no-gpu-shared-kernel' then Opt.GpuSharedKernel := false
    else if Arg = '--experimental-fp16' then Opt.ExperimentalFP16 := true
    else if Arg = '--experimental-int8-input' then Opt.ExperimentalInt8Input := true
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
    else if Arg = '--reasoning-effort' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      if not ReasoningEffortFromName(SVal, Opt.ReasoningEffort) then
      begin
        Opt.ErrorMsg := Arg + ': not off|low|medium|xhigh: ' + SVal;
        exit(false);
      end;
    end
    else if Arg = '--system' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.SystemPrompt := SVal;
    end
    else if Arg = '-p' then
    begin
      if not NextValue(Arg, SVal) then exit(false);
      Opt.Prompt := SVal;
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
  // The KV cache follows the weight mode unless picked explicitly: int8 and
  // int4 weights get the int8 KV cache (same accuracy philosophy, ~1/4 the KV
  // RAM), --fp32 weights keep the bit-exact FP32 cache. Identical on CPU
  // and GPU (the cached decode path is the same code).
  if not Opt.KVInt8Set then Opt.KVInt8 := Opt.WeightMode <> cwmFP32;
  Result := true;
end;

// The end-of-turn marker the assistant reply terminates with in each format
// (the token-id stop sequence; trimmed from the reply when matched).
function EndOfTurnMarker(ChatFormat: TNeuralChatFormat): string;
begin
  case ChatFormat of
    cfChatML:  Result := '<|im_end|>';
    cfQwen3_5: Result := '<|im_end|>'; // Qwen3.5/3.6 ChatML variant
    cfQwen3_8: Result := '<|im_end|>'; // Qwen3.8 ChatML variant
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

function Utf8IncompleteTailLen(const S: string): integer;
var
  Len, Pos, Need: integer;
  B: byte;
begin
  Result := 0;
  Len := Length(S);
  Pos := Len;
  // Back over at most three continuation bytes to the candidate lead byte.
  while (Pos > 0) and (Len - Pos < 3) and ((Ord(S[Pos]) and $C0) = $80) do
    Dec(Pos);
  if Pos = 0 then exit;
  B := Ord(S[Pos]);
  if (B and $E0) = $C0 then Need := 2
  else if (B and $F0) = $E0 then Need := 3
  else if (B and $F8) = $F0 then Need := 4
  else exit; // ASCII, a run of stray continuation bytes, or an invalid lead
  if Len - Pos + 1 < Need then Result := Len - Pos + 1;
end;

function TakeCompleteUtf8(var Pending: string; const S: string): string;
var
  HoldLen, WholeLen: integer;
begin
  Result := Pending + S;
  WholeLen := Length(Result);
  HoldLen := Utf8IncompleteTailLen(Result);
  Pending := Copy(Result, WholeLen - HoldLen + 1, HoldLen);
  SetLength(Result, WholeLen - HoldLen);
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
  WindowNN := nil;
  WindowSession := nil;
  WindowIn := nil;
  TailNN := nil;
  TailSession := nil;
  TailIn := nil;
  TransferSnap := nil;
  WindowBorrowsWeights := false;
  ActiveSession := nil;
  Tokenizer := nil;
  ChatFormat := cfUnknown;
  RawMode := false;
  ReuseOK := false;
  StateReuseOK := false;
  SetLength(Checkpoints, 0);
  CheckpointBandWidth := csDefaultCheckpointBandWidth;
  CheckpointBandRatio := 2;
  CheckpointOnTwins := false;
  SeqLen := 0;
  VocabSize := 0;
  SetLength(MarkerIds, 0);
  SetLength(CachedTokens, 0);
  ModelType := '';
  ContextFull := false;
  LastPromptTokens := 0;
  LastCompletionTokens := 0;
  LastFinishReason := '';
  LastReusedTokens := 0;
  LastPrefixTokens := 0;
  LastCachedTokens := 0;
  LastPrefillTokens := 0;
  LastPrefillWindows := 0;
  LastPrefillTailWindows := 0;
  TotalInputTokens := 0;
  TotalCachedInputTokens := 0;
  TotalInputMs := 0;
  TotalOutputTokens := 0;
  TotalOutputMs := 0;
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
  FreeCheckpoints(); // OpenCL slots in NN's context: free before the nets
  FreeAndNil(Session); // before NN.Free: Destroy ends incremental decode on
                       // NN's layers
  FreeAndNil(WindowSession);
  FreeAndNil(TailSession);
  FreeAndNil(WindowNN); // before NN: the twins borrow NN's weights and
  FreeAndNil(TailNN);   // resident device codes
  FreeAndNil(Tokenizer);
  FreeAndNil(NN);
  FreeAndNil(WindowIn);
  FreeAndNil(TailIn);
  FreeAndNil(TransferSnap);
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
var
  Whole: string;
begin
  if not Assigned(OnToken) then exit;
  Whole := TakeCompleteUtf8(PendingUtf8, S);
  if Whole <> '' then OnToken(Whole);
end;

procedure TChatEngine.FlushPendingUtf8();
begin
  if PendingUtf8 = '' then exit;
  PendingUtf8 := '';
  if Assigned(OnToken) then OnToken(Utf8ReplacementChar);
end;

function TChatEngine.CheckpointBand(Distance: integer): integer;
begin
  // Band k covers distances [W * r^k, W * r^(k+1)); a distance below W is
  // band 0 (the newest captures) and one past the context the last band.
  if Distance < CheckpointBandWidth then exit(0);
  Result := Trunc(Ln(Distance / CheckpointBandWidth) / Ln(CheckpointBandRatio));
  if Result > High(Checkpoints) then Result := High(Checkpoints);
  if Result < 0 then Result := 0;
end;

function TChatEngine.DeepestCheckpointAtOrBelow(Limit: integer):
  TNNetDecoderStateCheckpoint;
var
  SlotPos, MaxSlotPos: integer;
begin
  Result := nil;
  MaxSlotPos := High(Checkpoints);
  for SlotPos := 0 to MaxSlotPos do
    if (Checkpoints[SlotPos].Position > 0) and
      (Checkpoints[SlotPos].Position <= Limit) and
      ((Result = nil) or (Checkpoints[SlotPos].Position > Result.Position)) then
      Result := Checkpoints[SlotPos];
end;

procedure TChatEngine.DropCheckpointsAbove(Limit: integer);
var
  SlotPos, MaxSlotPos: integer;
begin
  MaxSlotPos := High(Checkpoints);
  for SlotPos := 0 to MaxSlotPos do
    if Checkpoints[SlotPos].Position > Limit then
      Checkpoints[SlotPos].Position := 0;
end;

procedure TChatEngine.RetainCheckpointsBefore(FedPos: integer);
var
  SlotPos, MaxSlotPos, Band, HolderPos, LiveCount, FarthestPos: integer;
  Chk: TNNetDecoderStateCheckpoint;
begin
  MaxSlotPos := High(Checkpoints);
  DropCheckpointsAbove(FedPos);
  // One checkpoint per band of distance from FedPos, the deepest (largest
  // Position) of the ones held. The capture about to land is not held yet,
  // so the newest held checkpoint keeps its band and moves to a farther one
  // as the fed position grows; the two turn-boundary checkpoints of a
  // request are its two deepest and no capture follows them until the next
  // request resumes, so the rule keeps them on its own.
  for Band := 0 to MaxSlotPos do CheckpointBandDeepest[Band] := -1;
  LiveCount := 0;
  for SlotPos := 0 to MaxSlotPos do
  begin
    Chk := Checkpoints[SlotPos];
    if Chk.Position <= 0 then continue;
    Inc(LiveCount);
    Band := CheckpointBand(FedPos - Chk.Position);
    HolderPos := CheckpointBandDeepest[Band];
    if HolderPos < 0 then CheckpointBandDeepest[Band] := SlotPos
    else if Checkpoints[HolderPos].Position < Chk.Position then
    begin
      Checkpoints[HolderPos].Position := 0;
      CheckpointBandDeepest[Band] := SlotPos;
      Dec(LiveCount);
    end
    else
    begin
      Chk.Position := 0;
      Dec(LiveCount);
    end;
  end;
  // Every band holding one leaves no slot for the capture: evict the
  // farthest checkpoint, where divergence is the least likely.
  if LiveCount >= Length(Checkpoints) then
  begin
    FarthestPos := -1;
    for SlotPos := 0 to MaxSlotPos do
    begin
      Chk := Checkpoints[SlotPos];
      if Chk.Position <= 0 then continue;
      if (FarthestPos < 0) or
        (Chk.Position < Checkpoints[FarthestPos].Position) then
        FarthestPos := SlotPos;
    end;
    if FarthestPos >= 0 then Checkpoints[FarthestPos].Position := 0;
  end;
end;

procedure TChatEngine.CaptureCheckpoint(ASession: TNNetStreamingDecoder;
  FedPos: integer);
var
  SlotPos, MaxSlotPos: integer;
begin
  if (FedPos <= 0) or (Length(Checkpoints) = 0) then exit;
  MaxSlotPos := High(Checkpoints);
  for SlotPos := 0 to MaxSlotPos do
    if Checkpoints[SlotPos].Position = FedPos then exit;
  RetainCheckpointsBefore(FedPos);
  for SlotPos := 0 to MaxSlotPos do
    if Checkpoints[SlotPos].Position <= 0 then
    begin
      ASession.CaptureStateInto(Checkpoints[SlotPos]);
      Checkpoints[SlotPos].Position := FedPos;
      exit;
    end;
  raise Exception.Create('TChatEngine.CaptureCheckpoint: no free slot after' +
    ' the retention pass');
end;

procedure TChatEngine.FreeCheckpoints();
var
  SlotPos, MaxSlotPos: integer;
begin
  MaxSlotPos := High(Checkpoints);
  for SlotPos := 0 to MaxSlotPos do Checkpoints[SlotPos].Free;
  SetLength(Checkpoints, 0);
  SetLength(CheckpointBandDeepest, 0);
end;

procedure TChatEngine.SwitchTo(Target: TNNetStreamingDecoder);
begin
  if Target = ActiveSession then exit;
  // One deep copy of the live state each way, into the snapshot LoadModel
  // allocated once; nothing is allocated per switch.
  ActiveSession.SnapshotInto(TransferSnap);
  Target.RestoreSnapshot(TransferSnap);
  ActiveSession := Target;
end;

function TChatEngine.LoadModel(const AOpt: TChatOptions;
  out ErrorMsg: string): boolean;
var
  TokenizerFile, TokenizerConfigFile, Marker, Line: string;
  LoadStart: QWord;             // per-phase load wall clock (tokenizer,
                                // checkpoint + caches, GPU weight upload)
  Cnt, LastIdx: integer;
  Int4LayerCount: integer;      // layers holding int4 weights after --int4
  Int4DirectLayerCount: integer; // of those, loaded straight from Q4_0 blocks
  TwinBytes: int64;             // the prefill twins' NonWeightBytes
  CheckpointsGiven: boolean;    // --cache-checkpoints N was on the command line
  OpenCLOn: boolean;            // OpenCL offload is live (not fallen back)
  SlotPos: integer;
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
  // That linear cost buys the checkpoint's own context on the common models
  // (32768 for Qwen2.5), so with no --ctx we take max_position_embeddings
  // clamped to DefaultCtxCap. Checkpoints declaring more (Llama-3 at 131072)
  // stop at the cap; --ctx N goes past it if you have the RAM, and --kv-int8
  // quarters what the cache costs at any given size.
  if Opt.CtxLen <= 0 then
  begin
    Cnt := ReadConfigInt(IncludeTrailingPathDelimiter(Opt.ModelDir) +
      'config.json', 'max_position_embeddings', DefaultCtxCap);
    if (Cnt <= 0) or (Cnt > DefaultCtxCap) then Cnt := DefaultCtxCap;
    Opt.CtxLen := Cnt;
    Notice(Format('[context not set - defaulting to %d tokens; override' +
      ' with --ctx N (KV-cache RAM grows ~O(ctx); --kv-int8 quarters it)]',
      [Opt.CtxLen]));
  end;

  // A window must leave room in the cache for at least one more token: the
  // prompt is at most CtxLen-1 tokens and the last one is never prefilled.
  // A value the user typed that cannot be honoured is an error, not a
  // notice: a silently ignored window costs a full model load to discover.
  if Opt.PrefillWindow >= Opt.CtxLen then
  begin
    ErrorMsg := Format('--prefill-window %d: must be below the context of' +
      ' %d tokens (raise it with --ctx N, or pick a smaller window)',
      [Opt.PrefillWindow, Opt.CtxLen]);
    FreeAndNil(Tokenizer);
    exit;
  end;
  // The tail twin's width: below the window, or there is none (1).
  if Opt.PrefillWindow = 0 then
  begin
    if Opt.PrefillTailWindow > 1 then
    begin
      ErrorMsg := Format('--prefill-tail-window %d: needs --prefill-window N' +
        ' with N above it', [Opt.PrefillTailWindow]);
      FreeAndNil(Tokenizer);
      exit;
    end;
    Opt.PrefillTailWindow := 1;
  end
  else if Opt.PrefillTailWindow = 0 then
  begin
    if csDefaultPrefillTailWindow < Opt.PrefillWindow then
      Opt.PrefillTailWindow := csDefaultPrefillTailWindow
    else
    begin
      Notice(Format('[--prefill-window %d: no tail twin, the default tail' +
        ' width %d is not below it (--prefill-tail-window T picks one)]',
        [Opt.PrefillWindow, csDefaultPrefillTailWindow]));
      Opt.PrefillTailWindow := 1;
    end;
  end
  else if Opt.PrefillTailWindow >= Opt.PrefillWindow then
  begin
    ErrorMsg := Format('--prefill-tail-window %d: must be below' +
      ' --prefill-window %d (1 builds no tail twin)',
      [Opt.PrefillTailWindow, Opt.PrefillWindow]);
    FreeAndNil(Tokenizer);
    exit;
  end;
  // One slot cannot hold both turn-boundary checkpoints, and a store past
  // the cap is a typo; both stop the load like the window flags.
  CheckpointsGiven := Opt.CacheCheckpoints >= 0;
  if (Opt.CacheCheckpoints = 1) or
    (Opt.CacheCheckpoints > csMaxCacheCheckpoints) then
  begin
    ErrorMsg := Format('--cache-checkpoints %d: must be 0 (off) or 2 to %d',
      [Opt.CacheCheckpoints, csMaxCacheCheckpoints]);
    FreeAndNil(Tokenizer);
    exit;
  end;

  {$IFDEF OpenCL}
  GpuCL := nil;
  if Opt.Gpu and Opt.LowMemory then
  begin
    Notice('[--low-memory ignored: incompatible with --gpu]');
    Opt.LowMemory := false;
  end;
  {$ENDIF}
  // The half B operand exists only inside the int8 OpenCL matmuls, so both
  // conditions must hold. Reported and cleared here rather than left to do
  // nothing silently at EnableOpenCL.
  if Opt.ExperimentalFP16 and (not Opt.Gpu) then
  begin
    Notice('[--experimental-fp16 ignored: OpenCL offload is off]');
    Opt.ExperimentalFP16 := false;
  end;
  if Opt.ExperimentalFP16 and (Opt.WeightMode <> cwmInt8) then
  begin
    Notice('[--experimental-fp16 ignored: the half activations are wired for int8' +
      ' weights only, and --' + ChatWeightModeName(Opt.WeightMode) +
      ' was requested]');
    Opt.ExperimentalFP16 := false;
  end;
  if Opt.ExperimentalInt8Input and (Opt.WeightMode = cwmFP32) then
  begin
    Notice('[--experimental-int8-input ignored: the int8 input feeds int8' +
      ' weights only, and --fp32 was requested]');
    Opt.ExperimentalInt8Input := false;
  end;

  // Model: generic architecture dispatch, inference-only, int8 by default.
  // Weight precision. Int8 is the default (less RAM and faster on CPU and
  // GPU); --fp32 opts into full-precision weights (more RAM, slower);
  // --int4 quantizes the convolution/projection layers further, on the CPU.
  case Opt.WeightMode of
    cwmInt8: Notice('[int8 weights (default) - less RAM, faster on CPU and GPU;' +
      ' on --gpu the codes stay resident on the device; --fp32 opts out]');
    cwmFP32: Notice('[--fp32: full-precision weights - more RAM, slower than int8]');
    cwmInt4: Notice('[--int4: Q4_0 tensors of a .gguf checkpoint load' +
      ' straight into int4 weight rows; every other tensor streams into int8' +
      ' rows and TNNet.QuantizeWeightsInt4 requantizes it to Q4_0 int4 after' +
      ' the load; on --gpu the packed codes stay resident on the device;' +
      ' output quality below int8]');
  end;
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
  // The direct Q4_0 -> int4 weight route reads a global too, for the same
  // reason: it must reach LoadLlamaLinearWeights without touching the
  // signature of every importer between here and it. Restored after the build.
  NeuralImportInt4FromQ4_0 := Opt.WeightMode = cwmInt4;
  NeuralImportInt4LayerCount := 0;
  NeuralAllowFusedAttention := not Opt.NoFusedAttn;
  if Opt.NoFusedAttn then
    Notice('[--no-fused-attn: per-head attention wiring (SplitChannels/SDPA/' +
      'DeepConcat) instead of the fused layer - bit-identical, A/B only]');

  Notice('Loading ' + Opt.ModelDir + ' ...');
  LoadStart := GetTickCount64();
  ModelType := ReadModelType(IncludeTrailingPathDelimiter(Opt.ModelDir) +
    'config.json');
  if ModelType = '' then ModelType := 'unknown';
  // Built at INPUT WIDTH 1 (pSeqLen=1): streamed decode feeds one token per
  // forward and the KV cache (budget = CtxLen, set on the session below) holds
  // the context. SeqLen is the cache budget, NOT the built input width.
  NN := BuildFromPretrained(Opt.ModelDir, {pSeqLen=}1,
    {pTrainable=}false, '', {pQuantizeInt8=}Opt.WeightMode <> cwmFP32);
  Int4DirectLayerCount := NeuralImportInt4LayerCount;
  // --prefill-window: the width-N twin borrows NN's weights
  // (BuildFromPretrained with pWeightOwner: one checkpoint read, no weight
  // storage of its own) once NN's weight state is final, below. Families the
  // Llama builder does not cover take the full second build here instead;
  // that twin then follows every step NN takes.
  WindowBorrowsWeights := (Opt.PrefillWindow > 0) and
    PretrainedModelTypeCanBorrowWeights(ModelType);
  if (Opt.PrefillWindow > 0) and (not WindowBorrowsWeights) then
  begin
    Notice(Format('[--prefill-window %d: model_type "%s" is outside the' +
      ' Llama builder, so the width-%d twin is a full second build:' +
      ' checkpoint read twice, weights held twice in RAM and on the device]',
      [Opt.PrefillWindow, ModelType, Opt.PrefillWindow]));
    if Opt.PrefillTailWindow > 1 then
      Notice(Format('[--prefill-tail-window %d: no tail twin on the full' +
        ' second build, a third copy of the weights is not worth the' +
        ' leftover single steps]', [Opt.PrefillTailWindow]));
    Opt.PrefillTailWindow := 1;
    NeuralImportInt4LayerCount := 0;
    WindowNN := BuildFromPretrained(Opt.ModelDir, {pSeqLen=}Opt.PrefillWindow,
      {pTrainable=}false, '', {pQuantizeInt8=}Opt.WeightMode <> cwmFP32);
  end;
  NeuralImportInt4FromQ4_0 := false;
  // Low-memory forward path, set independently of trainability. The importer
  // built inference-only with low memory ON (SetTrainable's pLowMemory default);
  // honor --max-fast-memory by re-sweeping the layers, then flush each weight
  // cache so the concatenated-weight cache is (re)built or dropped to match.
  NN.SetTrainable({pTrainable=}false, {pLowMemory=}Opt.LowMemory);
  LastIdx := NN.GetLastLayerIdx();
  for Cnt := 0 to LastIdx do
    NN.Layers[Cnt].FlushWeightCache();
  if Assigned(WindowNN) then
  begin
    WindowNN.SetTrainable({pTrainable=}false, {pLowMemory=}Opt.LowMemory);
    LastIdx := WindowNN.GetLastLayerIdx();
    for Cnt := 0 to LastIdx do
      WindowNN.Layers[Cnt].FlushWeightCache();
  end;
  Notice(Format('Model loaded in %.1fs.',
    [(GetTickCount64() - LoadStart) / 1000]));

  // --int4 runs BEFORE EnableOpenCL: TNNetLayerConcatedWeights.QuantizeWeightsInt4
  // refuses a layer that already has OpenCL enabled.
  if Opt.WeightMode = cwmInt4 then
  begin
    // Counts the layers the loader already direct-loaded too: they exit
    // QuantizeWeightsInt4 immediately and it counts them as int4.
    Int4LayerCount := NN.QuantizeWeightsInt4();
    // The fallback twin quantizes its own copy; a borrowing twin does not
    // exist yet and will link NN's int4 tables as it is built.
    if Assigned(WindowNN) then WindowNN.QuantizeWeightsInt4();
    Notice('[--int4: Q4_0 int4 weights on ' + IntToStr(Int4LayerCount) +
      ' layers (' + IntToStr(Int4DirectLayerCount) +
      ' loaded directly from the checkpoint Q4_0 blocks, ' +
      IntToStr(Int4LayerCount - Int4DirectLayerCount) +
      ' requantized from int8); the other weight layers' +
      ' stay int8; int8 input copy enabled on the int4 layers]');
  end;

  // The borrowing twin: NN's weight state is final (int8, or int4 above) and
  // OpenCL is not enabled yet, the order TNNetLayer.LinkWeightsFrom needs.
  // NeuralAllowFusedAttention is still the build-time value, so the twin
  // gets the same graph. It takes no quantize step of its own (its layers
  // borrow), only the low-memory sweep that decides its own caches.
  if WindowBorrowsWeights then
  begin
    LoadStart := GetTickCount64();
    WindowNN := BuildFromPretrained(Opt.ModelDir, {pSeqLen=}Opt.PrefillWindow,
      {pTrainable=}false, '', {pQuantizeInt8=}Opt.WeightMode <> cwmFP32,
      {pWeightOwner=}NN);
    WindowNN.SetTrainable({pTrainable=}false, {pLowMemory=}Opt.LowMemory);
    LastIdx := WindowNN.GetLastLayerIdx();
    for Cnt := 0 to LastIdx do
      WindowNN.Layers[Cnt].FlushWeightCache();
    if Opt.PrefillTailWindow > 1 then
    begin
      TailNN := BuildFromPretrained(Opt.ModelDir,
        {pSeqLen=}Opt.PrefillTailWindow, {pTrainable=}false, '',
        {pQuantizeInt8=}Opt.WeightMode <> cwmFP32, {pWeightOwner=}NN);
      TailNN.SetTrainable({pTrainable=}false, {pLowMemory=}Opt.LowMemory);
      LastIdx := TailNN.GetLastLayerIdx();
      for Cnt := 0 to LastIdx do
        TailNN.Layers[Cnt].FlushWeightCache();
    end;
    Line := '';
    if Assigned(TailNN) then
      Line := Format(' and the width-%d tail twin', [Opt.PrefillTailWindow]);
    Notice(Format('[--prefill-window %d: width-%d twin%s built in %.1fs' +
      ' sharing the loaded weights (checkpoint read once; the twins hold' +
      ' %d weights of their own and cost their activations only)]',
      [Opt.PrefillWindow, Opt.PrefillWindow, Line,
       (GetTickCount64() - LoadStart) / 1000, WindowNN.CountWeights()]));
  end;
  NeuralAllowFusedAttention := true; // restore the global default post-build

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
        if not Opt.GpuSharedKernel then
          Notice('[--no-gpu-shared-kernel: per-layer kernels and command queues - ' +
            'each layer waits for its sources, so --profile charges GPU time to ' +
            'layers instead of the queue drain; slower than shared]');
        if Opt.WeightMode = cwmInt4 then
          Notice('[--int4 with --gpu: cai_dot_product_int4_splitk reads the' +
            ' packed codes at half the int8 traffic; activations stay FP32]');
        if Opt.ExperimentalFP16 then
          Notice('[--experimental-fp16: under construction -' +
            ' half-precision activations in the int8 matmuls; weights stay' +
            ' int8, logits are not bit-exact. A device that rejects' +
            ' cai_dot_product_int8_h keeps the FP32 activations]');
        LoadStart := GetTickCount64();
        // Read by TNNetLayerConcatedWeights.EnableOpenCL when it acquires the
        // half kernel, so it must be assigned before the call below: that is
        // what sizes the half B buffer, and a later write does nothing.
        NN.OpenCLFP16 := Opt.ExperimentalFP16;
        NN.EnableOpenCL(GpuCL.PlatformIds[Opt.GpuPlatform],
          GpuCL.Devices[Opt.GpuDevice], Opt.GpuSharedKernel);
        if Assigned(WindowNN) then
        begin
          WindowNN.OpenCLFP16 := Opt.ExperimentalFP16;
          // A borrowing twin must live in NN's OpenCL context to retain
          // NN's resident codes (a context of its own cannot share cl_mem).
          if WindowBorrowsWeights then
            WindowNN.EnableOpenCLInContextOf(NN, Opt.GpuSharedKernel)
          else
            WindowNN.EnableOpenCL(GpuCL.PlatformIds[Opt.GpuPlatform],
              GpuCL.Devices[Opt.GpuDevice], Opt.GpuSharedKernel);
        end;
        if Assigned(TailNN) then // only built when it borrows
        begin
          TailNN.OpenCLFP16 := Opt.ExperimentalFP16;
          TailNN.EnableOpenCLInContextOf(NN, Opt.GpuSharedKernel);
        end;
        Notice(Format('GPU weights uploaded in %.1fs.',
          [(GetTickCount64() - LoadStart) / 1000]));
      end;
    end;
  end;
  {$ENDIF}

  // After BuildQuantInt8/QuantizeWeightsInt8 (an FP32 layer is skipped) and
  // after EnableOpenCL (the CPU verdict is what routes to the int8 x int8
  // kernel). The armed count is the user's confirmation that ordering held.
  if Opt.ExperimentalInt8Input then
  begin
    Notice('[--experimental-int8-input: under construction - int8 input copy' +
      ' armed on ' + IntToStr(NN.EnableInt8Input()) + ' layers (the count is' +
      ' every layer holding the copy, so with --int4 it includes the int4' +
      ' layers, which arm it themselves); only TNNetConvolution runs' +
      ' int8 x int8 today, the others still run int8 x FP32]');
    if Assigned(WindowNN) then WindowNN.EnableInt8Input();
    if Assigned(TailNN) then TailNN.EnableInt8Input();
  end;

  SeqLen := Opt.CtxLen;
  VocabSize := NN.GetLastLayer().Output.Depth;
  // int8 KV cache is armed at construction so the FP32 K/V buffers are never
  // allocated (a post-Create switch frees them, but the allocator arena may
  // keep the pages). Reset and cache-reuse truncation keep the int8 mode
  // (they only rewind the cache length).
  Session := TNNetStreamingDecoder.Create(NN, SeqLen, Opt.KVInt8);
  ActiveSession := Session;
  if Assigned(WindowNN) then
  begin
    WindowSession := TNNetStreamingDecoder.Create(WindowNN, SeqLen, Opt.KVInt8);
    WindowIn := TNNetVolume.Create(Opt.PrefillWindow, 1, 1);
    TransferSnap := TNNetDecoderSessionSnapshot.Create();
    if Assigned(TailNN) then
    begin
      TailSession := TNNetStreamingDecoder.Create(TailNN, SeqLen, Opt.KVInt8);
      TailIn := TNNetVolume.Create(Opt.PrefillTailWindow, 1, 1);
      Notice(Format('[--prefill-window %d: the prompt is prefilled %d tokens' +
        ' per forward on the width-%d twin, what those windows leave over %d' +
        ' tokens per forward on the tail twin, and the fewer than %d tokens' +
        ' left go one token at a time]',
        [Opt.PrefillWindow, Opt.PrefillWindow, Opt.PrefillWindow,
         Opt.PrefillTailWindow, Opt.PrefillTailWindow]));
    end
    else
      Notice(Format('[--prefill-window %d: the prompt is prefilled %d tokens' +
        ' per forward on the width-%d twin; the tail that does not fill a' +
        ' window goes one token at a time]',
        [Opt.PrefillWindow, Opt.PrefillWindow, Opt.PrefillWindow]));
    // The twins' whole host footprint besides the borrowed weights: layer
    // outputs, the convolution kernels' raw outputs and bias rows, the KV
    // caches their sessions just sized; inference-only layers keep no
    // per-step backward caches.
    TwinBytes := WindowNN.NonWeightBytes();
    if Assigned(TailNN) then TwinBytes := TwinBytes + TailNN.NonWeightBytes();
    Notice(Format('[--prefill-window %d: the twins hold %.1f MB of' +
      ' activations and caches in host memory besides the borrowed weights]',
      [Opt.PrefillWindow, TwinBytes / (1024 * 1024)]));
  end;
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
  if Assigned(WindowSession) then WindowSession.Parallel := Session.Parallel;
  if Assigned(TailSession) then TailSession.Parallel := Session.Parallel;
  // --max-threads: cap the scheduler pool (and with it the per-layer chunk
  // count). Set BEFORE StartThreadWorkers so the pool is created at the capped
  // size instead of being built wide and resized on the next pass.
  if Opt.MaxThreads > 0 then NN.MaxThreadNum := Opt.MaxThreads;
  if Assigned(WindowNN) and (Opt.MaxThreads > 0) then
    WindowNN.MaxThreadNum := Opt.MaxThreads;
  if Assigned(TailNN) and (Opt.MaxThreads > 0) then
    TailNN.MaxThreadNum := Opt.MaxThreads;
  // Keep the scheduler's worker pool alive and HOT between decode steps (default
  // policy: ~50% of the pool hot, worker 0 always) so each token's parallel
  // forward reaches the workers without re-warming the pool every step.
  if Session.Parallel then NN.StartThreadWorkers();
  if Assigned(WindowNN) and Session.Parallel then WindowNN.StartThreadWorkers();
  if Assigned(TailNN) and Session.Parallel then TailNN.StartThreadWorkers();
  // KV-cache reuse across turns needs position-truncatable attention K/V and no
  // recurrent (SSM) state to rewind. Pure-attention nets qualify; NoCacheReuse
  // forces the full re-prefill at the call site.
  ReuseOK := (Session.SSMCount = 0) and (Session.SDPACount > 0);
  // Hybrid / pure-recurrent nets take the checkpoint route instead: the
  // attention K/V (FP32 or int8) is still truncated in place, and the
  // recurrent state, which has no per-position history, is put back from a
  // checkpoint captured at the resume position.
  StateReuseOK := Session.SSMCount > 0;
  FreeCheckpoints();
  SetLength(CachedTokens, 0);
  // --cache-checkpoints: resolved here because the default follows the
  // OpenCL decision above (a fallen-back --gpu counts as CPU).
  OpenCLOn := {$IFDEF OpenCL}Assigned(GpuCL){$ELSE}false{$ENDIF};
  if Opt.CacheCheckpoints < 0 then
  begin
    if OpenCLOn then Opt.CacheCheckpoints := csDefaultCacheCheckpointsOpenCL
    else Opt.CacheCheckpoints := csDefaultCacheCheckpointsCPU;
  end;
  if StateReuseOK and (Opt.CacheCheckpoints > 0) and not Opt.NoCacheReuse then
  begin
    SetLength(Checkpoints, Opt.CacheCheckpoints);
    for SlotPos := 0 to High(Checkpoints) do
      Checkpoints[SlotPos] := Session.NewStateCheckpoint();
    SetLength(CheckpointBandDeepest, Opt.CacheCheckpoints);
    // The band width is the finest capture spacing, so every checkpoint
    // falls under the one band rule: the tail window, else the prefill
    // window, else (boundary captures only) a fixed width that still leaves
    // at least one band of distance inside the context.
    if Assigned(TailIn) then CheckpointBandWidth := TailIn.SizeX
    else if Assigned(WindowIn) then CheckpointBandWidth := WindowIn.SizeX
    else CheckpointBandWidth := Max(1, Min(csDefaultCheckpointBandWidth,
      SeqLen div 2));
    CheckpointBandRatio := Power(SeqLen / CheckpointBandWidth,
      1 / Opt.CacheCheckpoints);
    if CheckpointBandRatio <= 1 then CheckpointBandRatio := 2;
    // A twin built in its own OpenCL context cannot read a slot owned by
    // NN's context, so the window captures are skipped on that fallback.
    CheckpointOnTwins := (not OpenCLOn) or WindowBorrowsWeights;
  end;
  Line := Format('Model: %s, %d params, vocab %d, context %d, chat format ',
    [ModelType, NN.CountWeights(), VocabSize, SeqLen]);
  if RawMode then Line := Line + 'raw (completion)'
  else Line := Line + ChatFormatName(ChatFormat);
  Notice(Line + ', ' + ChatWeightModeName(Opt.WeightMode) + ' weights.');
  // The reasoning effort is a no-op on a template without a reasoning
  // control, so say which of the two happened rather than dropping it
  // silently.
  if FormatHasReasoningControl(ChatFormat) then
    Notice('[reasoning effort ' + ReasoningEffortName(Opt.ReasoningEffort) +
      ']')
  else if Opt.ReasoningEffort <> reXHigh then
    Notice('[reasoning effort ' + ReasoningEffortName(Opt.ReasoningEffort) +
      ' ignored - ' + BoolToStr(RawMode, 'raw mode',
      ChatFormatName(ChatFormat)) + ' has no reasoning control]');
  if Opt.NoCacheReuse then
    Notice('[KV-cache reuse OFF (--no-cache-reuse) - full re-prefill each turn]')
  else if ReuseOK then
    Notice('[KV-cache reuse ON - only the new prompt tail is prefilled each turn]')
  else if StateReuseOK then
  begin
    if Opt.CacheCheckpoints = 0 then
      Notice('[cache checkpoints OFF (--cache-checkpoints 0) - full' +
        ' re-prefill each turn]')
    else
    begin
      if Checkpoints[0].OpenCLSlotCount > 0 then
        Line := Format('%.1f MB of it in OpenCL memory',
          [Opt.CacheCheckpoints * Checkpoints[0].OpenCLBytes() / (1024 * 1024)])
      else Line := 'in host RAM';
      Notice(Format('[cache checkpoints ON - up to %d checkpoints of the' +
        ' recurrent state (%.1f MB each, %.1f MB in all, %s), captured after' +
        ' every prefill window and at the end of the prompt and of the reply,' +
        ' kept one per band of distance from the newest token (%d bands,' +
        ' width %d, ratio %.2f); a prompt resumes from the deepest' +
        ' checkpoint at or below its shared prefix and only the tail after' +
        ' it is prefilled]',
        [Opt.CacheCheckpoints, Checkpoints[0].Bytes() / (1024 * 1024),
         Opt.CacheCheckpoints * Checkpoints[0].Bytes() / (1024 * 1024), Line,
         Opt.CacheCheckpoints, CheckpointBandWidth, CheckpointBandRatio]));
      if not CheckpointOnTwins then
        Notice('[--cache-checkpoints: the prefill twins run in their own' +
          ' OpenCL context, so only the end-of-prompt and end-of-reply' +
          ' checkpoints are captured]');
    end;
  end
  else
    Notice('[cache reuse N/A for this architecture - full re-prefill each turn]');
  if CheckpointsGiven and not StateReuseOK then
    Notice(Format('[--cache-checkpoints %d ignored: no recurrent state in' +
      ' this net; the KV cache is truncated to the shared prefix instead]',
      [Opt.CacheCheckpoints]));
  if Opt.Serial then
    Notice('[serial layer loop (--serial) - fully single-threaded]')
  else
  begin
    Notice('[layer-graph parallel forward (default) - independent layers and' +
      ' large conv/linear layers threaded; pass --serial for the serial loop]');
    if Opt.MaxThreads > 0 then
      Notice(Format('[--max-threads %d - worker pool capped at %d thread(s)]',
        [Opt.MaxThreads, Opt.MaxThreads]));
  end;

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
    ChatTemplateOptions({AddGenerationPrompt=}true,
      {ContinueFinalMessage=}false, GenOpt.ReasoningEffort));
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
// Prefill reuse, two routes, both keyed on CachedTokens (the token-id
// sequence currently resident in the session, updated here):
//   ReuseOK (pure attention): keep the KV cache across calls, TruncateTo the
//     common prefix and prefill only the diverging tail.
//   StateReuseOK (hybrid/recurrent): recurrent state has no per-position
//     history to truncate, so TruncateTo the deepest cache checkpoint at or
//     below the common prefix, restore the recurrent state it holds, and
//     prefill from there; a prompt below every checkpoint falls back to a
//     full reset.
// NoCacheReuse disables both: full reset, whole prompt re-prefilled.
function TChatEngine.GenerateFromIds(const PromptIds: TNeuralIntegerArray;
  const GenOpt: TChatOptions): string;
var
  Chain: TNNetLogitsProcessorChain;
  Penalty: TNNetTokenHistoryPenalty;
  Sampler: TNNetSamplerBase;
  CacheReuse: boolean;
  StateReuse: boolean;         // the checkpoint route: resume AND capture
  ResumeChk: TNNetDecoderStateCheckpoint; // the checkpoint this call resumes
  GreedyFast: boolean;
  Tokens: TNeuralIntegerArray;
  Generated: TNeuralIntegerArray;
  GenCount, GenCap: integer;   // Generated is grown with spare capacity
  OneId: array[0..0] of integer;
  Incremental: boolean;
  InV, Output, Row: TNNetVolume;
  Len, StepCnt, Cnt, NewToken: integer;
  Reused, PromptLen: integer;  // KV-cache reuse bookkeeping (and --stats)
  WindowLen, WindowCount: integer; // --prefill-window: the width-N twin
  TailLen, TailCount: integer;     // and its width-T tail twin
  SingleSteps: integer;            // prefill tokens fed one at a time
  FirstSession: TNNetStreamingDecoder; // the session the prefill steps first
  PrefillMs: double;           // --stats: TStart..TPrefillEnd
  PrefillTokens: integer;      // tokens fed by this call's prefill
  LenM2, MarkerLen, EmLen, DecLen: integer;
  LastPos, RowBytes: integer;
  Decoded, Emitted, Piece: string;
  // --stats timing (monotonic ms). TStart: before prefill; TPrefillEnd:
  // after the last prefill step; TFirst: when the first reply token is
  // produced (so TTFT covers prefill + first step); TEnd: after the decode
  // loop. Produced counts emitted tokens.
  TStart, TPrefillEnd, TFirst, TEnd: QWord;
  Produced: integer;
  DecodeSecs: double;
  // --stats per-phase split of one decode step, accumulated over the loop in
  // ms: the net forward, the logits row copy + softmax, the processor chain
  // (repetition penalty), the sampler, the detokenize + emit, and the rest.
  PhaseMark, StepMark: TDateTime;
  ForwardMs, SoftMaxMs, ChainMs, SamplerMs, EmitMs, StepMs: double;
  function PhaseElapsedMs(): double;
  var
    NowMark: TDateTime;
  begin
    NowMark := Now();
    Result := (NowMark - PhaseMark) * MSecsPerDay;
    PhaseMark := NowMark;
  end;
  // Up to WindowCount whole windows of Tokens from Cnt on through WSession
  // (the active one), never padded. A window that would overflow the cache is
  // never fed: the fused attention layer only prints on overflow and then
  // scores past its buffers; the caller then feeds what is left in smaller
  // windows or single steps. Fed counts the windows this call fed.
  procedure FeedWindows(WSession: TNNetStreamingDecoder; WIn: TNNetVolume;
    WindowCount: integer; var Fed: integer);
  var
    WLen, WindowPos, WindowRow, MaxRowPos: integer;
  begin
    WLen := WIn.SizeX;
    {$IFDEF Debug}
    Assert(WSession = ActiveSession,
      'TChatEngine: windows fed to a session that does not hold the state');
    Assert(WLen = WSession.Net.GetFirstLayer().Output.SizeX,
      'TChatEngine: window volume width differs from the twin input width');
    {$ENDIF}
    MaxRowPos := WLen - 1;
    for WindowPos := 0 to WindowCount - 1 do
    begin
      if Cnt + WLen > SeqLen then break;
      for WindowRow := 0 to MaxRowPos do
        WIn.FData[WindowRow] := Tokens[Cnt + WindowRow];
      WSession.StepForwardToHidden(WIn, Cnt);
      Inc(Cnt, WLen);
      Inc(Fed);
      if StateReuse and CheckpointOnTwins then CaptureCheckpoint(WSession, Cnt);
    end;
  end;
begin
  Result := '';
  if not Loaded then
    raise Exception.Create('TChatEngine.GenerateFromIds before LoadModel');
  ContextFull := false;
  LastPromptTokens := Length(PromptIds);
  LastCompletionTokens := 0;
  LastFinishReason := 'length';
  LastReusedTokens := 0;
  LastPrefixTokens := 0;
  LastCachedTokens := 0;
  LastPrefillTokens := 0;
  LastPrefillWindows := 0;
  LastPrefillTailWindows := 0;
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
  // The checkpoint route needs a store (LoadModel sizes none at
  // --cache-checkpoints 0); --no-cache-reuse disables both routes.
  StateReuse := StateReuseOK and not GenOpt.NoCacheReuse and
    (Length(Checkpoints) > 0);
  ResumeChk := nil;
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
  LenM2 := Len - 2;
  MarkerLen := Length(MarkerIds);
  if Len > 0 then Move(PromptIds[0], Tokens[0], Len * csIntegerSize);
  SetLength(Generated, 0);
  GenCount := 0;
  GenCap := 0;
  Emitted := '';
  EmLen := 0;
  PendingUtf8 := '';
  // Streamed emission strategy, resolved once (#20/#27). When Decode is an
  // exact left-to-right concatenation of the per-id pieces, each step needs
  // to detokenize ONLY the new id; otherwise (WordPiece space-join, or a
  // decoder that strips leading spaces at position 0) the whole generated
  // region has to be re-decoded and diffed against what was already emitted.
  Incremental := Tokenizer.DecodeIsConcatenative();
  InV := TNNetVolume.Create(1, 1, 1);
  Output := nil; // a reference into the net, returned by Session.Output()
  Row := TNNetVolume.Create(VocabSize, 1, 1);
  TStart := GetTickCount64();
  TPrefillEnd := TStart;
  TFirst := 0;
  TEnd := 0;
  // --profile: the prefill gets its own layer-class report, so both nets
  // start this call with clean timers (the twin runs the windows, NN the
  // tail).
  if GenOpt.Profile then
  begin
    NN.ClearTime();
    NN.ResetSchedulerStats();
    if Assigned(WindowNN) then
    begin
      WindowNN.ClearTime();
      WindowNN.ResetSchedulerStats();
    end;
    if Assigned(TailNN) then
    begin
      TailNN.ClearTime();
      TailNN.ResetSchedulerStats();
    end;
  end;
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
    // Prefill the prompt, reusing the KV-cache prefix shared with the last
    // call when possible. Reused = length of the cached prefix that still
    // matches this prompt. The LAST prompt token is fed as the first decode
    // step's input, so the cache must not already hold it - cap reuse at
    // Len-1. Tokens Reused..LenM2 are fed down a ladder of widths: whole
    // windows of WindowLen through WindowSession, whole windows of TailLen
    // of what is left through TailSession, then the fewer than TailLen
    // tokens left one at a time through Session (never padded). The reused
    // state goes straight into whichever session steps first. No prefill
    // logits are read, so every prefill forward stops at the LM-head input
    // (StepForwardToHidden): the vocab projection runs only in the decode
    // steps.
    LastCachedTokens := Length(CachedTokens);
    LastPrefixTokens := CommonPrefixLen(CachedTokens, PromptIds);
    Reused := LastPrefixTokens;
    if Reused > Len - 1 then Reused := Len - 1;
    if StateReuse then
    begin
      // Checkpoint resume: a checkpoint holds the recurrent state at exactly
      // its position, so the deepest one at or below the shared prefix is
      // resumed and the tokens from there on are prefilled. Every checkpoint
      // past that position describes a sequence the cache will no longer
      // hold.
      ResumeChk := DeepestCheckpointAtOrBelow(Reused);
      if Assigned(ResumeChk) then Reused := ResumeChk.Position
      else Reused := 0;
      DropCheckpointsAbove(Reused);
    end
    else if not CacheReuse then
    begin
      Reused := 0;
      DropCheckpointsAbove(0); // a full reset leaves nothing to resume from
    end;
    PrefillTokens := LenM2 + 1 - Reused;
    LastReusedTokens := Reused;
    LastPrefillTokens := PrefillTokens;
    WindowCount := 0;
    WindowLen := 0;
    TailCount := 0;
    TailLen := 0;
    if Assigned(WindowSession) then
    begin
      WindowLen := WindowIn.SizeX;
      WindowCount := PrefillTokens div WindowLen;
      if Assigned(TailSession) then
      begin
        TailLen := TailIn.SizeX;
        TailCount := (PrefillTokens - WindowCount * WindowLen) div TailLen;
      end;
    end;
    if WindowCount > 0 then FirstSession := WindowSession
    else if TailCount > 0 then FirstSession := TailSession
    else FirstSession := Session;
    ActiveSession := FirstSession;
    if Reused = 0 then ActiveSession.Reset() // SSM state cannot be truncated
    else
    begin
      // The prefix lives in Session's cache (rows 0..Reused-1 of what
      // CachedTokens lists): truncate there, put the checkpoint's recurrent
      // state back when this is the checkpoint route, then carry it.
      Session.TruncateTo(Reused);
      if Assigned(ResumeChk) then Session.RestoreStateFrom(ResumeChk);
      ActiveSession := Session;
      SwitchTo(FirstSession);
    end;
    Cnt := Reused;
    if WindowCount > 0 then
      FeedWindows(WindowSession, WindowIn, WindowCount, LastPrefillWindows);
    // Whole windows of TailLen over what is left, so a window the width-N
    // twin did not feed (the cache-room check) also falls to the tail twin.
    if TailLen > 0 then TailCount := (LenM2 + 1 - Cnt) div TailLen;
    if TailCount > 0 then
    begin
      SwitchTo(TailSession);
      FeedWindows(TailSession, TailIn, TailCount, LastPrefillTailWindows);
    end;
    SwitchTo(Session);
    while Cnt <= LenM2 do
    begin
      InV.FData[0] := Tokens[Cnt];
      Session.StepForwardToHidden(InV, Cnt);
      Inc(Cnt);
    end;
    TPrefillEnd := GetTickCount64();
    // End-of-prompt checkpoint for the next call's resume: Session now holds
    // exactly PromptIds[0..Len-2], a prefix of what CachedTokens will list
    // after the decode loop, so the prefix test above stays truthful for
    // it. The time counts toward TTFT (--stats prefill excludes it).
    if StateReuse then CaptureCheckpoint(Session, Cnt);
    SingleSteps := PrefillTokens - LastPrefillWindows * WindowLen -
      LastPrefillTailWindows * TailLen;
    // --profile: the prefill's own layer-class report, one table per net
    // that ran (the width-N twin, the tail twin, NN for the single steps),
    // then clean timers so the report after the decode loop covers only the
    // repeated single-token decode steps.
    if GenOpt.Profile then
    begin
      if PrefillTokens > 0 then
      begin
        WriteLn(StdErr);
        Write(StdErr, Format('[profile] prefill: %d tokens in %d ms',
          [PrefillTokens, TPrefillEnd - TStart]));
        if WindowLen > 0 then
        begin
          Write(StdErr, Format(' (%d windows of %d', [LastPrefillWindows,
            WindowLen]));
          if TailLen > 0 then
            Write(StdErr, Format(', %d tail windows of %d',
              [LastPrefillTailWindows, TailLen]));
          Write(StdErr, Format(', %d single steps)', [SingleSteps]));
        end;
        WriteLn(StdErr);
        if LastPrefillWindows > 0 then
        begin
          WriteLn(StdErr, Format('[profile] width-%d twin: %d windows',
            [WindowLen, LastPrefillWindows]));
          Write(StdErr, TNNet.LayerClassTimingReport(WindowNN));
          WriteLn(StdErr, '[sched] ', WindowNN.SchedulerStatsReport());
        end;
        if LastPrefillTailWindows > 0 then
        begin
          WriteLn(StdErr, Format('[profile] width-%d tail twin: %d windows',
            [TailLen, LastPrefillTailWindows]));
          Write(StdErr, TNNet.LayerClassTimingReport(TailNN));
          WriteLn(StdErr, '[sched] ', TailNN.SchedulerStatsReport());
        end;
        if SingleSteps > 0 then
        begin
          if WindowLen > 0 then
            WriteLn(StdErr, Format('[profile] width-1 net: %d single steps',
              [SingleSteps]));
          Write(StdErr, TNNet.LayerClassTimingReport(NN));
          WriteLn(StdErr, '[sched] ', NN.SchedulerStatsReport());
        end;
        Flush(StdErr);
      end;
      NN.ClearTime();
      NN.ResetSchedulerStats();
    end;
    RowBytes := VocabSize * csNeuralFloatSize;
    // #14: pure-greedy decode (no sampler, empty processor chain) needs only
    // argmax(logits); softmax and its full-vocab Move are order-preserving, so
    // argmax(softmax(L)) = argmax(L). Skip both on this hot per-token path.
    GreedyFast := (Sampler = nil) and (Chain.Count = 0);
    ForwardMs := 0; SoftMaxMs := 0; ChainMs := 0; SamplerMs := 0;
    EmitMs := 0; StepMs := 0;
    for StepCnt := 1 to GenOpt.MaxNewTokens do
    begin
      if Len >= SeqLen then break;
      StepMark := Now();
      PhaseMark := StepMark;
      // One width-1 forward of the last committed token over the cached past.
      LastPos := Len - 1;
      InV.FData[0] := Tokens[LastPos];
      Session.StepForward(InV, LastPos);
      ForwardMs := ForwardMs + PhaseElapsedMs();
      Output := Session.Output(); // (1,1,vocab) -- the single logits row
      if GreedyFast then
      begin
        NewToken := ArgMaxRow(Output);  // #14: argmax(softmax)=argmax(logits)
        SamplerMs := SamplerMs + PhaseElapsedMs();
      end
      else
      begin
        Move(Output.FData[0], Row.FData[0], RowBytes);
        RowSoftMax(Row);
        SoftMaxMs := SoftMaxMs + PhaseElapsedMs();
        Chain.ProcessRow(Row);
        ChainMs := ChainMs + PhaseElapsedMs();
        if Assigned(Sampler) then NewToken := Sampler.GetToken(Row)
        else NewToken := ArgMaxRow(Row);
        SamplerMs := SamplerMs + PhaseElapsedMs();
      end;
      Chain.Commit(NewToken);
      ChainMs := ChainMs + PhaseElapsedMs();
      Tokens[Len] := NewToken;
      Inc(Len);
      Inc(Produced);
      if Produced = 1 then TFirst := GetTickCount64(); // TTFT boundary
      // Grow by doubling: an exact regrow per token copies the whole reply
      // again on every step.
      if GenCount >= GenCap then
      begin
        if GenCap = 0 then GenCap := 32 else GenCap := GenCap * 2;
        SetLength(Generated, GenCap);
      end;
      Generated[GenCount] := NewToken;
      Inc(GenCount);
      // EOS / end-of-turn checks BEFORE emitting so markers never echo.
      if (Tokenizer.EosId >= 0) and (NewToken = Tokenizer.EosId) then
      begin
        LastFinishReason := 'stop';
        break;
      end;
      if TailMatches(Generated, GenCount, MarkerIds) then
      begin
        Dec(GenCount, MarkerLen);
        LastFinishReason := 'stop';
        break;
      end;
      if Incremental then
      begin
        // The new id's own piece IS the delta, so no re-decode and no prefix
        // test: Decode over this tokenizer concatenates the per-id pieces, so
        // the emitted text stays exactly Decode(Generated[0..GenCount-1]).
        // The piece may be an incomplete UTF-8 sequence when a codepoint
        // straddles two tokens - that is what the old whole-string diff
        // emitted too, byte for byte.
        OneId[0] := NewToken;
        Piece := Tokenizer.Decode(OneId, {SkipSpecialTokens=}true);
        if Piece <> '' then
        begin
          EmitToken(Piece);
          Inc(EmLen, Length(Piece));
        end;
      end
      else
      begin
        // Non-concatenative decoder: re-decode the whole generated region and
        // emit the delta. The join/cleanup can rewrite the tail, so only emit
        // while the previous text is still a prefix.
        Decoded := Tokenizer.DecodeCount(Generated, GenCount,
          {SkipSpecialTokens=}true);
        DecLen := Length(Decoded);
        if (DecLen > EmLen) and
          ((EmLen = 0) or CompareMem(@Decoded[1], @Emitted[1], EmLen)) then
        begin
          EmitToken(Copy(Decoded, EmLen + 1, DecLen - EmLen));
          Emitted := Decoded;
          EmLen := DecLen;
        end;
      end;
      EmitMs := EmitMs + PhaseElapsedMs();
      StepMs := StepMs + (Now() - StepMark) * MSecsPerDay;
    end;
    LastCompletionTokens := Produced;
    Result := Tokenizer.DecodeCount(Generated, GenCount,
      {SkipSpecialTokens=}true);
    // Anything the prefix-guard held back (or trimmed markers shortened). On
    // the incremental path the emitted text is by construction the decode of
    // a PREFIX of Generated, so a longer Result always extends it.
    DecLen := Length(Result);
    if (DecLen > EmLen) and
      (Incremental or (EmLen = 0) or CompareMem(@Result[1], @Emitted[1], EmLen))
      then EmitToken(Copy(Result, EmLen + 1, DecLen - EmLen));
    // A reply can end on a stray lead byte (a byte-level BPE model may stop
    // mid-codepoint). The stream and the returned text both get U+FFFD in
    // its place, so a JSON body built from either stays valid UTF-8.
    FlushPendingUtf8();
    EmLen := Utf8IncompleteTailLen(Result);
    if EmLen > 0 then
      Result := Copy(Result, 1, DecLen - EmLen) + Utf8ReplacementChar;
    if Assigned(OnReplyDone) then OnReplyDone();
    // Record the sequence now resident in the cache for the next call's
    // prefix diff: every token that was FED is cached (positions 0..Len-2);
    // the final produced token (Tokens[Len-1]) was sampled but never fed, so
    // it is not.
    SetLength(CachedTokens, Len - 1);
    if Len > 1 then Move(Tokens[0], CachedTokens[0], (Len - 1) * csIntegerSize);
    // End-of-reply checkpoint for the next call's resume. Taken HERE, after
    // the decode loop, so the session holds exactly the tokens CachedTokens
    // lists (positions 0..Len-2) - the two must agree or the prefix test
    // above would validate a boundary the session is not actually at. Its
    // retention pass is the end-of-request one.
    if StateReuse then CaptureCheckpoint(Session, Len - 1);
    // Lifetime usage totals, kept whether or not --stats prints them.
    // Input time is the prefill; output time runs from the end of prefill
    // to the end of decode (it includes the first decode step).
    TEnd := GetTickCount64();
    PrefillMs := TPrefillEnd - TStart;
    if Produced > 0 then
    begin
      Inc(TotalInputTokens, PromptLen);
      Inc(TotalCachedInputTokens, Reused);
      TotalInputMs := TotalInputMs + PrefillMs;
      Inc(TotalOutputTokens, Produced);
      TotalOutputMs := TotalOutputMs + (TEnd - TPrefillEnd);
    end;
    // Per-turn timing to stderr (keeps stdout = pure model output). TTFT =
    // prefill + first decode step; decode tok/s measures the steady-state
    // decode of the tokens AFTER the first, so prefill cost is excluded.
    // prompt N (reused K, prefix P of C cached): K tokens were resumed from
    // the cache or a checkpoint; P is where the prompt's ids diverged from
    // the C cached ids, so a small K next to a large P names a divergence
    // the checkpoints did not cover.
    if GenOpt.Stats and (Produced > 0) then
    begin
      Write(StdErr, Format('[stats] input: prompt %d tokens (reused %d,' +
        ' prefix %d of %d cached), TTFT %d ms',
        [PromptLen, Reused, LastPrefixTokens, LastCachedTokens,
         TFirst - TStart]));
      if (PrefillTokens > 0) and (PrefillMs > 0) then
        Write(StdErr, Format(', prefill %.1f tok/s',
          [PrefillTokens * 1000.0 / PrefillMs]));
      WriteLn(StdErr);
      Write(StdErr, Format('[stats] output: %d tokens in %.1f s',
        [Produced, (TEnd - TPrefillEnd) / 1000.0]));
      if Produced > 1 then
      begin
        DecodeSecs := (TEnd - TFirst) / 1000.0;
        if DecodeSecs > 0 then
          Write(StdErr, Format(', decode %.1f tok/s',
            [(Produced - 1) / DecodeSecs]));
      end;
      WriteLn(StdErr);
      // The same decode steps split by phase, so the host work outside the
      // net forward is visible next to it. A step that broke on EOS before
      // its emit phase counts in "other".
      WriteLn(StdErr, Format('[stats] per decode step (ms): forward %.2f,' +
        ' softmax %.2f, processors %.2f, sampler %.2f, emit %.2f, other %.2f,' +
        ' step %.2f',
        [ForwardMs / Produced, SoftMaxMs / Produced, ChainMs / Produced,
         SamplerMs / Produced, EmitMs / Produced,
         (StepMs - ForwardMs - SoftMaxMs - ChainMs - SamplerMs - EmitMs) / Produced,
         StepMs / Produced]));
      WriteLn(StdErr, Format('[stats] total input: %d tokens (cached %d), %.1f s',
        [TotalInputTokens, TotalCachedInputTokens, TotalInputMs / 1000.0]));
      WriteLn(StdErr, Format('[stats] total output: %d tokens, %.1f s',
        [TotalOutputTokens, TotalOutputMs / 1000.0]));
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
    DropCheckpointsAbove(0); // captured at positions the sequence
                             // CachedTokens no longer vouches for
    Session.Reset();
    if Assigned(WindowSession) then WindowSession.Reset();
    if Assigned(TailSession) then TailSession.Reset();
    ActiveSession := Session;
    raise;
  end;
end;

end.
