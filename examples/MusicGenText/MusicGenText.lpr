program MusicGenText;
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

// MusicGenText -- the END-TO-END TEXT-CONDITIONED MusicGen demo. A free-text
// prompt is tokenized, run through the REAL T5 text ENCODER, and used to
// condition the MusicGen DECODER, whose EnCodec code stack is synthesized to a
// waveform by the EnCodec decoder:
//
//   prompt text
//     -> T5 SentencePiece tokenizer (real model) OR fixed ids (pico demo)
//     -> T5 text ENCODER (BuildT5FromSafeTensors)
//     -> the encoder's final hidden states (EncSeq x text_d_model)
//     -> MusicGen enc_to_dec_proj -> cross-attention conditioning
//     -> MusicGen DECODER greedy/sampled delay-pattern decode -> [K][frames]
//     -> EnCodec audio DECODER (BuildEnCodecFromSafeTensors) -> mono waveform
//     -> SaveVolumeToWav16 -> a .wav clip
//
// TWO MODES
// ---------
// 1. REAL CHECKPOINT (--download): fetches three STANDARD public HF repos
//    through the native Pascal Hub helper (neuralhfhub, no Python) and imports
//    each directly from its own snapshot -- NO splitting/conversion step:
//        facebook/musicgen-small  -> the LM decoder (the importer reads only
//                                    the decoder.* / enc_to_dec_proj.* keys and
//                                    ignores the bundled text/audio encoders)
//        t5-base                  -> the T5 text encoder + SentencePiece tokenizer
//        facebook/encodec_32khz   -> the 32 kHz audio codec
//    Downloads are cached under ~/.cache/neural-api/hub, so re-runs are offline.
//    The --prompt text is tokenized with the real T5 tokenizer, so the words
//    genuinely steer the music. This produces REAL audio.
//
// 2. PICO DEMO (no --download): the self-contained fallback on the committed
//    RANDOM fixtures (tests/fixtures/tiny_musicgen*, tiny_musicgen_t5enc*,
//    tiny_musicgen_encodec*) with a FIXED pseudo-prompt id sequence -- no
//    network, pure CPU, a fraction of a second. The weights are untrained
//    random, so the clip is NOISE; this only exercises the full wiring.
//
// Usage:
//   MusicGenText                                  # pico demo (random, noise)
//   MusicGenText --download --prompt "lo-fi hip hop beat to relax to"
//   MusicGenText --download --prompt "..." --seconds 8 --guidance 3.0
//   MusicGenText --download --prompt "..." --topk 250 --temperature 1.0
//   MusicGenText --download --prompt "..." --topk 0    # force greedy (drone!)
//   MusicGenText --frames 6 --no-cache            # explicit pico frame count
//   MusicGenText --summary                        # print weights-per-layer tables
//   MusicGenText --download --prompt "..." --no-gpu      # force CPU
//   MusicGenText --download --prompt "..." --gpu-device 1 # pick OpenCL device
// Repo overrides: --musicgen-repo / --t5-repo / --encodec-repo. Gated repos:
// set HF_TOKEN in the environment.
//
// GPU OFFLOAD (OpenCL): when built with -dOpenCL (the default .lpi config) the
// T5 text encoder and the MusicGen decoder offload their conv/linear matmuls to
// the GPU by default; --no-gpu forces CPU and --gpu-platform N / --gpu-device N
// select the OpenCL device. The decode loop's hot path runs on the MusicGen
// width-1 step twins, which are offloaded too (TMusicGenModel.EnableOpenCL).
// The EnCodec codec is a hand-rolled conv path (not a TNNet) and stays on CPU.
//
// DECODE DEFAULTS (--download mode): MusicGen is trained for top-k SAMPLING, so
// --download defaults to MusicGen's top_k=250 / temperature 1.0 and classifier-
// free guidance 3.0 -- the recipe that actually produces music. Greedy argmax
// (--topk 0) collapses into a repetitive drone and is for diagnostics only.
// --topk / --temperature set the per-codebook weighted top-k draw over
// softmax(logits/temperature); --guidance sets the CFG scale (two passes per
// step, KV-cache off). The KV-CACHE incremental-decode fast path
// (TMusicGenModel.GenerateEx with UseCache) is used when guidance = 1.0: each
// step feeds only the newest delayed frame and the self-attention heads reuse
// their cached K/V (bit-identical to the un-cached loop).

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  SysUtils, Math, StrUtils,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained,
  neuralaudio, neuralhftokenizer, neuralhfhub;

const
  PicoFixtureDir = '../../tests/fixtures/';
  OutWavName = 'musicgen_text_demo.wav';
  DefaultSeconds = 5.0;     // real-checkpoint default clip length
  DefaultRealTopK = 250;    // MusicGen's intended top-k sampling (NOT greedy)
  DefaultPrompt = 'lo-fi hip hop beat to relax to';
  DefaultMusicGenRepo = 'facebook/musicgen-small';
  DefaultT5Repo = 't5-base';
  DefaultEnCodecRepo = 'facebook/encodec_32khz';

var
  T5Enc, T5Dec: TNNet;
  T5Cfg: TT5Config;
  Model: TMusicGenModel;
  Config: TMusicGenConfig;
  Codec: TEnCodecModel;
  CodecCfg: TEnCodecConfig;
  Tok: TNeuralHFTokenizer;
  Tokens, EncStates, NullStates, Wave: TNNetVolume;
  Codes: TNNetIntArr2D;
  Waveform: TNeuralFloatDynArr;
  EncSeq, DecSeq, NumFrames, k_i, t, i, TopK: integer;
  cmin, cmax, distinct, modeCount, modeId, cid: integer;
  CodeHist: array of integer;   // per-codebook id frequency (collapse check)
  MgSafe, MgCfg, T5Safe, T5CfgPath, EcSafe, EcCfg, TokPath: string;
  MgDir, T5Dir, EcDir: string;
  Ids: array of integer;
  GuidanceScale, Temperature, Seconds: TNeuralFloat;
  UseCache, RealMode, ShowSummary: boolean;
  Sampler: TNNetSamplerBase;
  TickStart: QWord;   // captured before each timed model load / pipeline step
  Gpu: boolean;                  // OpenCL offload requested (T5 enc + MusicGen dec)
  GpuPlatform, GpuDevice: integer;
  {$IFDEF OpenCL}
  GpuCL: TEasyOpenCL;            // platform/device handle for OpenCL offload (nil = CPU)
  {$ENDIF}

  // Milliseconds elapsed since TickStart, formatted as a human-readable string
  // (e.g. "1234 ms" or "12.34 s"). GetTickCount64 is a monotonic ms clock.
  function Elapsed(const StartTick: QWord): string;
  var ms: QWord;
  begin
    ms := GetTickCount64 - StartTick;
    if ms >= 1000 then Result := FormatFloat('0.00', ms / 1000.0) + ' s'
    else Result := IntToStr(ms) + ' ms';
  end;

  // Returns the value following a "--name" flag, or Def if the flag is absent.
  function ParseStrArg(const Name, Def: string): string;
  var a: integer;
  begin
    Result := Def;
    for a := 1 to ParamCount - 1 do
      if ParamStr(a) = Name then Result := ParamStr(a + 1);
  end;

  // Returns the float following a "--name" flag, or Def if absent/unparsable.
  function ParseFloatArg(const Name: string; Def: TNeuralFloat): TNeuralFloat;
  var a: integer; v: TNeuralFloat;
  begin
    Result := Def;
    for a := 1 to ParamCount - 1 do
      if (ParamStr(a) = Name) and TryStrToFloat(ParamStr(a + 1), v) then Result := v;
  end;

  // Returns the integer following a "--name" flag, or Def if absent/unparsable.
  function ParseIntArg(const Name: string; Def: integer): integer;
  var a, v: integer;
  begin
    Result := Def;
    for a := 1 to ParamCount - 1 do
      if (ParamStr(a) = Name) and TryStrToInt(ParamStr(a + 1), v) then Result := v;
  end;

  // True if a bare "--name" flag is present anywhere on the command line.
  function HasFlag(const Name: string): boolean;
  var a: integer;
  begin
    Result := False;
    for a := 1 to ParamCount do if ParamStr(a) = Name then Result := True;
  end;

  // Resolves the T5 tokenizer file in a snapshot dir: prefer the raw
  // SentencePiece spiece.model, else tokenizer.json. Fetches spiece.model
  // explicitly (HubFetchModel only grabs tokenizer.json by default).
  function ResolveT5Tokenizer(const Repo, Dir: string): string;
  var P: string;
  begin
    if HubTryFetchFile(Repo, 'spiece.model', P) then Result := P
    else if FileExists(Dir + 'tokenizer.json') then Result := Dir + 'tokenizer.json'
    else Result := '';
  end;

begin
  WriteLn('MusicGen TEXT-CONDITIONED generation - prompt -> T5 -> music -> WAV');
  WriteLn('==================================================================');

  // ---- Mode selection ------------------------------------------------------
  RealMode := HasFlag('--download');
  GuidanceScale := ParseFloatArg('--guidance', 1.0);
  TopK := ParseIntArg('--topk', 0);
  Temperature := ParseFloatArg('--temperature', 1.0);
  Seconds := ParseFloatArg('--seconds', DefaultSeconds);
  UseCache := not HasFlag('--no-cache');
  ShowSummary := HasFlag('--summary');
  // OpenCL offload: ON by default when built with -dOpenCL (--no-gpu forces
  // CPU); a non-OpenCL build ignores the --gpu* flags entirely.
  Gpu := {$IFDEF OpenCL}not HasFlag('--no-gpu'){$ELSE}false{$ENDIF};
  GpuPlatform := ParseIntArg('--gpu-platform', 0);
  GpuDevice := ParseIntArg('--gpu-device', 0);
  Tok := nil;

  if RealMode then
  begin
    // Fetch the three standard repos through the native Pascal Hub helper.
    // HubFetchModel returns the local snapshot directory (cached; re-runs are
    // offline) with config.json + model.safetensors (+ tokenizer for t5).
    WriteLn('Downloading checkpoints via HuggingFace Hub (cache: ',
      HubGetCacheDir, ')...');
    TickStart := GetTickCount64;
    MgDir := IncludeTrailingPathDelimiter(
      HubFetchModel(ParseStrArg('--musicgen-repo', DefaultMusicGenRepo)));
    WriteLn('[time] fetch ', ParseStrArg('--musicgen-repo', DefaultMusicGenRepo),
      ': ', Elapsed(TickStart));
    TickStart := GetTickCount64;
    T5Dir := IncludeTrailingPathDelimiter(
      HubFetchModel(ParseStrArg('--t5-repo', DefaultT5Repo)));
    WriteLn('[time] fetch ', ParseStrArg('--t5-repo', DefaultT5Repo),
      ': ', Elapsed(TickStart));
    TickStart := GetTickCount64;
    EcDir := IncludeTrailingPathDelimiter(
      HubFetchModel(ParseStrArg('--encodec-repo', DefaultEnCodecRepo)));
    WriteLn('[time] fetch ', ParseStrArg('--encodec-repo', DefaultEnCodecRepo),
      ': ', Elapsed(TickStart));
    MgSafe := MgDir + 'model.safetensors';  MgCfg := MgDir + 'config.json';
    T5Safe := T5Dir + 'model.safetensors';  T5CfgPath := T5Dir + 'config.json';
    EcSafe := EcDir + 'model.safetensors';  EcCfg := EcDir + 'config.json';
    TokPath := ResolveT5Tokenizer(ParseStrArg('--t5-repo', DefaultT5Repo), T5Dir);
    if TokPath = '' then
    begin
      WriteLn('No T5 tokenizer (spiece.model / tokenizer.json) in ', T5Dir);
      Halt(1);
    end;
    // Default classifier-free guidance to MusicGen's 3.0 unless overridden.
    if not HasFlag('--guidance') then GuidanceScale := 3.0;
    // MusicGen is trained for top-k SAMPLING; greedy argmax (--topk 0) collapses
    // into a repetitive drone. Default to MusicGen's top_k=250 unless overridden
    // (pass --topk 0 to force the greedy path explicitly).
    if not HasFlag('--topk') then TopK := DefaultRealTopK;
    WriteLn('Prompt: "', ParseStrArg('--prompt', DefaultPrompt), '"');
    WriteLn;
  end
  else
  begin
    MgSafe := PicoFixtureDir + 'tiny_musicgen.safetensors';
    MgCfg := PicoFixtureDir + 'tiny_musicgen_config.json';
    T5Safe := PicoFixtureDir + 'tiny_musicgen_t5enc.safetensors';
    T5CfgPath := PicoFixtureDir + 'tiny_musicgen_t5enc_config.json';
    EcSafe := PicoFixtureDir + 'tiny_musicgen_encodec.safetensors';
    EcCfg := PicoFixtureDir + 'tiny_musicgen_encodec_config.json';
    if not FileExists(MgSafe) or not FileExists(T5Safe) or not FileExists(EcSafe)
    then
    begin
      WriteLn('Pico fixtures not found under ', PicoFixtureDir);
      WriteLn('Run from the example directory (examples/MusicGenText), or run');
      WriteLn('  python tools/musicgen_tiny_fixture.py  to (re)generate them.');
      Halt(1);
    end;
    WriteLn('No --download argument: running the self-contained pico demo on');
    WriteLn('the committed RANDOM fixtures (untrained weights, so the clip is');
    WriteLn('NOISE - this exercises the full text->audio wiring end to end).');
    WriteLn('Pass  --download --prompt "..."  to fetch a real checkpoint.');
    WriteLn;
  end;

  T5Enc := nil; T5Dec := nil; Model := nil; Codec := nil; Sampler := nil;
  {$IFDEF OpenCL}GpuCL := nil;{$ENDIF}
  Tokens := TNNetVolume.Create;
  EncStates := TNNetVolume.Create;
  NullStates := TNNetVolume.Create;
  Wave := TNNetVolume.Create;
  try
    // ---- OpenCL device selection -------------------------------------------
    // Resolve the OpenCL platform/device ONCE; the same handle drives both the
    // T5 encoder and the MusicGen decoder (each TNNet gets its own dot-product
    // kernel from it). Any failure falls back to CPU with a message. The pico
    // demo's tiny layers fall below the per-layer ShouldOpenCL size threshold,
    // so GPU mainly matters with --download (real checkpoints). EnCodec is a
    // hand-rolled conv codec (not a TNNet) and always synthesizes on CPU.
    {$IFDEF OpenCL}
    if Gpu then
    begin
      GpuCL := TEasyOpenCL.Create();
      if GpuCL.GetPlatformCount() = 0 then
      begin
        WriteLn('[--gpu: no OpenCL platform found - falling back to CPU]');
        FreeAndNil(GpuCL);
      end
      else
      begin
        if (GpuPlatform < 0) or (GpuPlatform >= GpuCL.GetPlatformCount()) then
          GpuPlatform := 0;
        GpuCL.SetCurrentPlatform(GpuCL.PlatformIds[GpuPlatform]);
        if GpuCL.GetDeviceCount() = 0 then
        begin
          WriteLn('[--gpu: no OpenCL device on platform ',
            GpuCL.PlatformNames[GpuPlatform], ' - falling back to CPU]');
          FreeAndNil(GpuCL);
        end
        else
        begin
          if (GpuDevice < 0) or (GpuDevice >= GpuCL.GetDeviceCount()) then
            GpuDevice := 0;
          GpuCL.SetCurrentDevice(GpuCL.Devices[GpuDevice]);
          WriteLn('[--gpu: OpenCL on ', GpuCL.PlatformNames[GpuPlatform],
            ' / ', GpuCL.DeviceNames[GpuDevice],
            ' - T5 encoder + MusicGen decoder; EnCodec stays on CPU]');
        end;
      end;
    end;
    {$ENDIF}

    // ---- 0. Build the prompt token ids. ------------------------------------
    if RealMode then
    begin
      Tok := TNeuralHFTokenizer.Create;
      TickStart := GetTickCount64;
      Tok.LoadFromFile(TokPath);
      WriteLn('[time] T5 tokenizer load: ', Elapsed(TickStart));
      // T5 encodes content ids then appends </s> (eos), no BOS.
      Ids := Tok.Encode(ParseStrArg('--prompt', DefaultPrompt));
      if (Tok.EosId >= 0) and
         ((Length(Ids) = 0) or (Ids[Length(Ids) - 1] <> Tok.EosId)) then
      begin
        SetLength(Ids, Length(Ids) + 1);
        Ids[Length(Ids) - 1] := Tok.EosId;
      end;
      if Length(Ids) = 0 then
      begin
        WriteLn('Prompt tokenized to zero tokens; nothing to condition on.');
        Halt(1);
      end;
    end
    else
      // A fixed pseudo-prompt: ids standing in for a real tokenizer's output.
      Ids := [3, 8, 1, 5, 2];
    EncSeq := Length(Ids);

    // ---- Pick the number of EnCodec frames to generate. --------------------
    if RealMode then
    begin
      // frame rate = sampling_rate / product(upsampling_ratios). Read the codec
      // config up front so --seconds can be converted before the decoder runs.
      CodecCfg := ReadEnCodecConfigFromJSONFile(EcCfg);
      NumFrames := 1;
      for i := 0 to Length(CodecCfg.UpsamplingRatios) - 1 do
        NumFrames := NumFrames * CodecCfg.UpsamplingRatios[i];   // hop size
      if HasFlag('--frames') then
        NumFrames := ParseIntArg('--frames', 6)
      else
        NumFrames := Max(1, Round(Seconds * CodecCfg.SamplingRate / NumFrames));
      WriteLn('Target length: ', Seconds:0:1, ' s -> ', NumFrames, ' frames.');
    end
    else
      NumFrames := ParseIntArg('--frames', 6);

    // ---- 1. Build the T5 text encoder and run it on the prompt ids. --------
    TickStart := GetTickCount64;
    BuildT5FromSafeTensors(T5Safe, T5Enc, T5Dec, T5Cfg, EncSeq, 1,
      {pTrainable=}false, T5CfgPath);
    WriteLn('[time] T5 model load (', T5Safe, '): ', Elapsed(TickStart));
    {$IFDEF OpenCL}
    if Assigned(GpuCL) then
      T5Enc.EnableOpenCL(GpuCL.PlatformIds[GpuPlatform], GpuCL.Devices[GpuDevice]);
    {$ENDIF}
    WriteLn('T5 text encoder: ', T5ConfigToString(T5Cfg));
    if ShowSummary then
    begin
      WriteLn;
      WriteLn('== T5 text encoder: weights per layer ==');
      T5Enc.PrintSummary();
    end;

    Tokens.ReSize(EncSeq, 1, 1);
    Write('Prompt token ids:');
    for i := 0 to EncSeq - 1 do
    begin
      Tokens.FData[i] := Ids[i];
      Write(' ', Ids[i]);
    end;
    WriteLn;
    TickStart := GetTickCount64;
    T5Enc.Compute(Tokens);
    WriteLn('[time] T5 encoder forward pass: ', Elapsed(TickStart));
    EncStates.Copy(T5Enc.GetLastLayer.Output);
    WriteLn('Encoder hidden states: ', EncStates.SizeX, 'x', EncStates.Depth,
      ' (seq x d_model)');
    // The conditioning is now captured in EncStates; we never run the T5
    // DECODER (BuildT5FromSafeTensors builds both towers). Free it BEFORE
    // loading the much larger MusicGen decoder so the peak RAM never carries
    // both. (FreeAndNil keeps the finally-block T5Dec.Free harmless.)
    FreeAndNil(T5Dec);
    WriteLn;

    // ---- 2. Build the MusicGen decoder and generate the code stack. --------
    Config := ReadMusicGenConfigFromJSONFile(MgCfg);
    if Config.TextDModel <> EncStates.Depth then
    begin
      WriteLn('FATAL: T5 d_model (', EncStates.Depth, ') != MusicGen ',
        'text_d_model (', Config.TextDModel, ')');
      Halt(1);
    end;
    // The decoder needs room for NumFrames + (K - 1) delay steps (+1 headroom).
    DecSeq := NumFrames + Config.NumCodebooks - 1 + 1;
    TickStart := GetTickCount64;
    Model := BuildMusicGenFromSafeTensors(MgSafe, Config, EncSeq, DecSeq,
      {pTrainable=}false, MgCfg);
    WriteLn('[time] MusicGen decoder load (', MgSafe, '): ', Elapsed(TickStart));
    {$IFDEF OpenCL}
    // Offloads the prefill decoder and arms the lazily built width-1 step twins
    // that the KV-cache decode loop actually runs on.
    if Assigned(GpuCL) then
      Model.EnableOpenCL(GpuCL.PlatformIds[GpuPlatform], GpuCL.Devices[GpuDevice]);
    {$ENDIF}
    WriteLn(MusicGenConfigToString(Config));
    if ShowSummary then
    begin
      WriteLn;
      WriteLn('== MusicGen decoder: weights per layer ==');
      Model.Decoder.PrintSummary();
    end;

    WriteLn('Generating ', NumFrames, ' frames over ', Config.NumCodebooks,
      ' codebooks via the delay pattern (conditioned on the T5 prompt)...');
    if TopK > 0 then
    begin
      Sampler := TNNetSamplerWeightedTopK.Create(TopK);
      WriteLn('Sampling: weighted top-k = ', TopK, ', temperature = ',
        Temperature:0:2, '.');
    end;
    TickStart := GetTickCount64;
    if GuidanceScale > 1.0 then
    begin
      // Classifier-free guidance: the unconditional branch is a ZEROED text
      // condition of the same shape. Blend uncond + scale*(cond - uncond).
      // Guidance runs two decoder passes per step; with the cache on, each pass
      // has its own KV-cache (dual-twin path) -- O(frames) instead of the
      // O(frames^2) re-encode loop. --no-cache forces the re-encode loop.
      NullStates.ReSize(EncStates.SizeX, EncStates.SizeY, EncStates.Depth);
      NullStates.Fill(0);
      WriteLn('Classifier-free guidance ON (scale = ', GuidanceScale:0:2,
        ', null = zeroed text condition; KV-cache ',
        IfThen(UseCache, 'on (dual-twin)', 'off'), ').');
      Model.GenerateEx(EncStates, NullStates, NumFrames, GuidanceScale,
        UseCache, Sampler, Temperature, Codes);
    end
    else
    begin
      if UseCache and (Sampler = nil) then
        WriteLn('Decode: greedy with KV-cache incremental decode.')
      else if UseCache then
        WriteLn('Decode: KV-cache incremental decode with sampling.')
      else
        WriteLn('Decode: full re-encode loop (--no-cache).');
      Model.GenerateEx(EncStates, nil, NumFrames, 1.0, UseCache, Sampler,
        Temperature, Codes);
    end;
    WriteLn('[time] MusicGen decode (', NumFrames, ' frames): ',
      Elapsed(TickStart));
    // Always print a compact per-codebook health summary: the range, the
    // number of DISTINCT ids used, and the single most-repeated id with its
    // share of frames. A healthy delay-pattern decode uses many distinct ids;
    // a near-constant codebook (distinct=1, or one id dominating) means the
    // greedy/sampled decode COLLAPSED - the conditioning is not steering it -
    // and the audio will be a drone/noise regardless of the codec.
    WriteLn('Code-stack health (', Config.NumCodebooks, ' codebooks x ',
      NumFrames, ' frames, vocab=', Config.VocabSize, '):');
    SetLength(CodeHist, Config.VocabSize);
    for k_i := 0 to Config.NumCodebooks - 1 do
    begin
      cmin := MaxInt; cmax := -MaxInt; distinct := 0; modeCount := 0; modeId := 0;
      for i := 0 to Config.VocabSize - 1 do CodeHist[i] := 0;
      for t := 0 to NumFrames - 1 do
      begin
        cid := Codes[k_i][t];
        if (cid >= 0) and (cid < Config.VocabSize) then
        begin
          if CodeHist[cid] = 0 then Inc(distinct);
          Inc(CodeHist[cid]);
          if CodeHist[cid] > modeCount then
          begin modeCount := CodeHist[cid]; modeId := cid; end;
        end;
        if cid < cmin then cmin := cid;
        if cid > cmax then cmax := cid;
      end;
      WriteLn('  cb', k_i, ': range [', cmin, '..', cmax, '], distinct=',
        distinct, '/', NumFrames, ', top id ', modeId, ' x', modeCount,
        ' (', (100.0 * modeCount / NumFrames):0:0, '%)',
        IfThen((distinct <= 2) or (modeCount > NumFrames div 2),
          '  <-- COLLAPSED?', ''));
    end;
    if NumFrames <= 16 then
    begin
      WriteLn('Generated code stack (codebook x frame):');
      for k_i := 0 to Config.NumCodebooks - 1 do
      begin
        Write('  cb', k_i, ':');
        for t := 0 to NumFrames - 1 do Write(' ', Codes[k_i][t]:3);
        WriteLn;
      end;
    end;
    WriteLn;

    // ---- 3. Decode the code stack to a waveform with the EnCodec decoder. --
    TickStart := GetTickCount64;
    Codec := BuildEnCodecFromSafeTensors(EcSafe, CodecCfg, EcCfg);
    WriteLn('[time] EnCodec decoder load (', EcSafe, '): ', Elapsed(TickStart));
    WriteLn('EnCodec decoder: ', EnCodecConfigToString(CodecCfg));
    if Codec.NumCodebooks < Config.NumCodebooks then
    begin
      WriteLn('FATAL: EnCodec has ', Codec.NumCodebooks,
        ' quantizers, fewer than MusicGen K=', Config.NumCodebooks);
      Halt(1);
    end;
    TickStart := GetTickCount64;
    Codec.DecodeCodesToAudio(Codes, Waveform, Config.NumCodebooks);
    WriteLn('[time] EnCodec waveform synthesis: ', Elapsed(TickStart));
    WriteLn('Decoded waveform: ', Length(Waveform), ' samples at ',
      CodecCfg.SamplingRate, ' Hz');

    // ---- 4. Write the clip to a 16-bit PCM WAV. ----------------------------
    Wave.ReSize(Length(Waveform), 1, 1);
    for i := 0 to Length(Waveform) - 1 do Wave.FData[i] := Waveform[i];
    SaveVolumeToWav16(Wave, OutWavName, CodecCfg.SamplingRate);
    WriteLn('Wrote ', OutWavName, ' (', Length(Waveform), ' samples, ',
      (Length(Waveform) / CodecCfg.SamplingRate):0:2, ' s).');
    WriteLn;
    if RealMode then
      WriteLn('Done: the prompt drove a real T5 encoder + MusicGen decoder + ',
        'EnCodec synthesis.')
    else
      WriteLn('Done (pico demo): full text->audio wiring exercised on random ',
        'weights.');
  finally
    T5Enc.Free;
    T5Dec.Free;
    Model.Free;
    Codec.Free;
    Sampler.Free;
    Tok.Free;
    Tokens.Free;
    EncStates.Free;
    NullStates.Free;
    Wave.Free;
    // After the nets (their dot-product kernels reference this handle's device).
    {$IFDEF OpenCL}GpuCL.Free;{$ENDIF}
  end;
end.
