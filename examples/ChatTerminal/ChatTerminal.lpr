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

The heavy lifting (option parsing, model/tokenizer load, sampling-defaults
resolution, KV-cache streamed generation) lives in the shared engine unit
neural/neuralchatengine.pas (TChatEngine), which this REPL and the sibling
ChatServer HTTP frontend both sit on. This file is the terminal-specific
part: the REPL loop, /commands, and stdout streaming.

The conversation is maintained as a multi-turn history rendered through the
chat-template engine (neuralchat.pas): the chat format is fingerprinted from
tokenizer_config.json's chat_template (DetectChatFormatFromConfigFile) and
can be overridden with --format chatml|llama2|llama3|zephyr|gemma|phi3|
mistral. Each turn re-renders the WHOLE history (system + user/assistant
turns + generation prompt), encodes it with the HF tokenizer
(EncodeChat) and generates the assistant reply token by token, streaming the
decoded text to stdout as it appears.

--format raw is the escape hatch for BASE (non-instruct) checkpoints such
as gpt2, mamba-130m or the pythias: no chat template at all. The REPL
becomes a completion notebook over a single running transcript - each typed
line is appended verbatim (no roles, no markup, no BOS) and the model
continues it; the continuation is appended back, so the next turn extends
the same document. Generation stops on the tokenizer's EOS id or
--max-new-tokens only (there is no end-of-turn marker), so base models that
never emit EOS run to the cap - pass a small --max-new-tokens. /system is
ignored (there is no system role); /reset clears the transcript. Raw is
never autodetected - explicit flag only. Coded by Claude (AI).

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
prompt; raises on formats without a system role, e.g. gemma/mistral, and is
ignored with a notice in --format raw).

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
  // cmem is skipped in the Debug build mode: it enables Valgrind (-gv), and
  // FPC then pulls in cmem itself, so naming it here is a duplicate.
  {$IFDEF UNIX}cthreads, {$IFNDEF Debug}cmem,{$ENDIF}{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralchat, neuralchatengine;

procedure PrintUsage();
begin
  WriteLn('Usage: ChatTerminal <model-dir> [options]');
  WriteLn;
  WriteLn('<model-dir> holds config.json, model.safetensors (or a sharded');
  WriteLn('index / pytorch_model.bin), tokenizer.json and (for chat-format');
  WriteLn('autodetection) tokenizer_config.json.');
  WriteLn;
  PrintChatOptionsHelp();
  WriteLn;
  WriteLn('REPL commands: /exit, /reset, /system <msg>');
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

// ---------------------------------------------------------------------------
// Streaming sinks: TChatEngine emits reply text and notices through events;
// the REPL prints both straight to stdout (tokens unbuffered, notices as
// lines) so the terminal behaves exactly like the pre-engine version.
// ---------------------------------------------------------------------------
type
  TStdoutSink = class(TObject)
    procedure TokenOut(const S: string);
    procedure NoticeOut(const S: string);
    procedure ReplyDone();
  end;

procedure TStdoutSink.TokenOut(const S: string);
begin
  Write(S);
  Flush(System.Output);
end;

procedure TStdoutSink.NoticeOut(const S: string);
begin
  WriteLn(S);
end;

// Fired after the last streamed token, before the --stats/--profile stderr
// reports: terminate the reply line exactly where the pre-engine
// ChatTerminal printed its newline.
procedure TStdoutSink.ReplyDone();
begin
  WriteLn;
  Flush(System.Output);
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

    // Server flags (shared parser; ChatServer uses them, the REPL ignores
    // them): defaults + explicit values + the port range check.
    Args.Clear;
    Args.Add('/tmp/model');
    Check(ParseArgs(Args, Opt) and (Opt.Host = '127.0.0.1') and
      (Opt.Port = 8080), 'host/port defaults (loopback:8080)');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--host'); Args.Add('0.0.0.0');
    Args.Add('--port'); Args.Add('8000');
    Check(ParseArgs(Args, Opt) and (Opt.Host = '0.0.0.0') and
      (Opt.Port = 8000), '--host/--port parse');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--port'); Args.Add('70000');
    Check(not ParseArgs(Args, Opt), 'out-of-range --port rejected');

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
    Check(EndOfTurnMarker(cfQwen3_5) = '<|im_end|>', 'Qwen3.5 end marker');
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
    // 'raw' is intercepted by TChatEngine.LoadModel BEFORE ChatFormatFromName
    // (it is a frontend mode, not a chat template); the library must keep NOT
    // knowing it, and cfUnknown must keep meaning "no end-of-turn marker"
    // (raw mode generation stops on EOS/--max-new-tokens only).
    Check(ChatFormatFromName('raw') = cfUnknown,
      'raw is not a library chat format');
    Check(EndOfTurnMarker(cfUnknown) = '', 'no end marker without a format');
    Args.Clear;
    Args.Add('/tmp/model');
    Args.Add('--format'); Args.Add('raw');
    Check(ParseArgs(Args, Opt) and (Opt.FormatName = 'raw'),
      '--format raw parses');

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
  Engine: TChatEngine;
  Sink: TStdoutSink;
  History: TChatMessages;
  Msgs: TChatMessages;
  PromptIds: TNeuralIntegerArray;
  Transcript: string;           // raw mode's running document (turns append)
  Cnt: integer;
  Line, Cmd, Arg, Reply, ErrorMsg: string;
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

  Sink := TStdoutSink.Create();
  Engine := TChatEngine.Create();
  Engine.OnToken := @Sink.TokenOut;
  Engine.OnNotice := @Sink.NoticeOut;
  Engine.OnReplyDone := @Sink.ReplyDone;
  if not Engine.LoadModel(Opt, ErrorMsg) then
  begin
    WriteLn(ErrorMsg);
    Engine.Free;
    Sink.Free;
    Halt(1);
  end;

  SetLength(History, 0);
  Transcript := '';
  if Engine.RawMode then
    WriteLn('Type text to complete; /exit quits, /reset clears the transcript.')
  else
  begin
    WriteLn('Type your message; /exit quits, /reset clears the history,');
    WriteLn('/system <msg> sets the system prompt.');
  end;
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
        Transcript := '';
        if Engine.RawMode then WriteLn('[transcript cleared]')
        else WriteLn('[history cleared]');
      end
      else if Cmd = 'system' then
      begin
        if Engine.RawMode then
          WriteLn('[no system role in raw mode - ignored]')
        else
        begin
          Engine.Opt.SystemPrompt := Arg;
          WriteLn('[system prompt set]');
        end;
      end
      else WriteLn('[unknown command /', Cmd, ' - /exit, /reset, /system]');
      continue;
    end;
    if Engine.RawMode then
    begin
      // Completion notebook: the typed line extends the running transcript
      // verbatim and the reply extends it further, so each turn's token ids
      // are a strict prefix-extension of the previous turn's - the KV-cache
      // reuse diff (CommonPrefixLen) prefills only the new tail for free.
      Transcript := Transcript + Line;
      PromptIds := Engine.Tokenizer.Encode(Transcript);
      Reply := Engine.GenerateFromIds(PromptIds, Engine.Opt);
      Transcript := Transcript + Reply;
      continue;
    end;
    SetLength(History, Length(History) + 1);
    History[High(History)] := ChatMessage('user', Line);
    try
      Msgs := AssembleMessages(Engine.Opt.SystemPrompt, History);
      Reply := Engine.ChatReply(Msgs, Engine.Opt);
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
  Engine.Free; // frees session (before the net), tokenizer, net, GPU handle
  Sink.Free;
end.
