program LatentTextToImage;
(*
LatentTextToImage: a complete OFFLINE, CPU latent text-to-image pipeline that
CHAINS the repo's already-landed generative importers into one sampling loop --
the SD3/Sora/PixArt-alpha recipe end to end:

  caller-supplied T5 text states  (Step 3 -- the real T5 tower -- is a follow-up)
    -> PixArt-alpha transformer denoiser (BuildPixArtFromSafeTensors)
    -> multi-step DDIM / DPM-Solver++ reverse loop (TNNetDiffusionScheduler)
       with classifier-free guidance (cond = the prompt T5 states; uncond = the
       null/empty-caption = ZERO T5 states, the PixArt CFG convention)
    -> sampled (sample_size,sample_size,in_channels) latent
    -> VAE decode (BuildVaeDecoderFromSafeTensors; the /0.18215 latent scaling
       lives inside the decoder's first layer)
    -> RGB image -> P6 PPM.

Nothing here is a new leaf layer: it is pure plumbing over landed pieces (the
PixArt importer + PixArtConditioning/PixArtDenoise, the VAE-decoder importer, and
the TNNetDiffusionScheduler DDIM / DPM-Solver++ / CFG driver).

STEP 3 (a tracked follow-up) is the only missing rung: build a REAL T5 encoder
(BuildT5FromSafeTensors) over a tokenized prompt and load a REAL PixArt + VAE
checkpoint. This demo instead supplies DETERMINISTIC synthetic T5 states and uses
the committed config-faithful pico fixtures (random weights), so it is a
WIRING / THROUGHPUT smoke -- it proves the chain runs offline and produces a
FINITE image, not photorealism.

SHAPES (pico fixtures, the default run):
  T5 states (TextSeqLen=5, 1, caption_channels=12)
    -> PixArt latent (6, 6, in_channels=4)
    -> VAE image (12, 12, out_channels=3)

NO NETWORK ACCESS / SELF-CONTAINED: falls back to the committed pico PixArt
(tests/fixtures/tiny_pixart.*, tools/make_pico_pixart_fixture.py, parity-checked
< 1e-4 in TestPixArtParity) and the matched pico VAE decoder
(tests/fixtures/tiny_vae_decoder_ltt.*, tools/vae_decoder_ltt_fixture.py); the
matched pair is sized so the PixArt latent flows straight into the VAE with no
reshaping. Regression-tested by TestLatentTextToImageSmoke (asserts no NaN/Inf in
the latent AND the decoded image).

USAGE
  ./LatentTextToImage                       use the committed pico fixtures
  ./LatentTextToImage pixart.safetensors vae.safetensors
                                            use your own checkpoints (with
                                            sibling config.json files)
Optional trailing flags:
  --steps N      number of reverse steps (default 4)
  --cfg W        classifier-free guidance scale (default 4.0)
  --apg ETA,MOMENTUM,THRESHOLD
                 Adaptive Projected Guidance instead of CFG + guidance-rescale
                 (paper setting e.g. 0,-0.5,15); mixes the x0 predictions
  --apg-eps      apply --apg to the raw eps predictions instead of x0
  --dpm          use DPM-Solver++(2M) instead of DDIM
  --unipc        use UniPC (order-2 bh2 predictor-corrector; better at 5-10 steps)
  --lcm          use the Latent Consistency Model few-step sampler
                 (TNNetLCMScheduler): a SINGLE model pass per step, guidance
                 BAKED IN (no cond/uncond double pass), ~4 steps. The matching
                 training objective is LCM-distillation: a consistency loss that
                 distills a many-step teacher (DDIM/DPM++) into this few-step
                 student so f(x_t,t) maps any noised latent straight to x0. (The
                 pico fixtures here are NOT LCM-distilled, so this is a wiring
                 SMOKE -- it proves the few-step loop runs, not that 4 LCM steps
                 match the teacher.)
  --tiny-vae DIR TAESD folder (diffusion_pytorch_model.safetensors + config.json,
                 e.g. madebyollin/taesd); defaults to the pico TAESD fixture
                 when no checkpoints are passed
  --preview N    every N steps, decode the current x0 estimate with the tiny
                 decoder into latent_text_to_image_preview_NNN.ppm
  --fast-vae     final decode with the tiny decoder instead of the full VAE
  --smoke        run, assert finiteness, print OK/FAIL and exit (no PPM)

OUTPUT
  Writes latent_text_to_image.ppm (the decoded RGB image) and prints the configs,
  the per-step trajectory norm and an ASCII preview of the decoded image. With
  --smoke it prints "SMOKE OK" / "SMOKE FAIL" and sets the exit code (the build
  step exercises this).

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math,
  neuralnetwork, neuralvolume, neuralpretrained, neuraldiffusion;

const
  cFixturePixArtST  = '../../tests/fixtures/tiny_pixart.safetensors';
  cFixturePixArtCfg = '../../tests/fixtures/tiny_pixart_config.json';
  cFixtureVaeST     = '../../tests/fixtures/tiny_vae_decoder_ltt.safetensors';
  cFixtureVaeCfg    = '../../tests/fixtures/tiny_vae_decoder_ltt_config.json';
  cFixtureTinyVaeST  = '../../tests/fixtures/tiny_taesd4_decoder.safetensors';
  cFixtureTinyVaeCfg = '../../tests/fixtures/tiny_taesd4_decoder_config.json';
  cTextSeqLen       = 5;
  cOutFile          = 'latent_text_to_image.ppm';
  cPreviewPrefix    = 'latent_text_to_image_preview_';

// Map a roughly-[-1,1] value to a byte (clamped). The pico fixtures are random
// so the dynamic range is arbitrary; this is just a viewable mapping.
function ToByte(V: TNeuralFloat): byte;
begin
  V := (V + 1) * 0.5;
  if V < 0 then V := 0;
  if V > 1 then V := 1;
  Result := Round(V * 255);
end;

procedure WriteColorPPM(Img: TNNetVolume; const FileName: string);
var
  F: TextFile;
  Bin: TFileStream;
  X, Y, W, H: integer;
  R, G, B: byte;
begin
  W := Img.SizeX; H := Img.SizeY;
  AssignFile(F, FileName); Rewrite(F);
  WriteLn(F, 'P6'); WriteLn(F, W, ' ', H); WriteLn(F, 255);
  CloseFile(F);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for Y := 0 to H - 1 do
      for X := 0 to W - 1 do
      begin
        R := ToByte(Img[X, Y, 0]);
        if Img.Depth > 1 then G := ToByte(Img[X, Y, 1]) else G := R;
        if Img.Depth > 2 then B := ToByte(Img[X, Y, 2]) else B := R;
        Bin.WriteByte(R); Bin.WriteByte(G); Bin.WriteByte(B);
      end;
  finally
    Bin.Free;
  end;
end;

procedure AsciiPreview(Img: TNNetVolume);
const cRamp = ' .:-=+*#%@';
var X, Y, idx: integer; v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
  begin
    for X := 0 to Img.SizeX - 1 do
    begin
      v := (Img[X, Y, 0] + Img[X, Y, 1] + Img[X, Y, 2]) / 3;
      v := (v + 1) * 0.5; if v < 0 then v := 0; if v > 1 then v := 1;
      idx := Round(v * (Length(cRamp) - 1)) + 1;
      Write(cRamp[idx]);
    end;
    WriteLn;
  end;
end;

function AllFinite(V: TNNetVolume): boolean;
var i: integer;
begin
  Result := true;
  for i := 0 to V.Size - 1 do
    // NaN != itself; Inf - Inf is NaN, so this catches both.
    if (V.FData[i] <> V.FData[i]) or (Abs(V.FData[i]) > 1e30) then
    begin
      Result := false;
      Exit;
    end;
end;

type
  TTinyVaePreview = class
  public
    Scheduler: TNNetDiffusionScheduler;
    Net: TNNet;
    Every: integer;
    WriteFiles, AllFiniteSoFar: boolean;
    X0: TNNetVolume;
    constructor Create;
    destructor Destroy; override;
    // When StepNo is a multiple of Every, decodes the x0 estimate of (Xt, RawPred,
    // Tt) with the tiny decoder Net into a preview PPM.
    procedure MaybeEmit(Xt, RawPred: TNNetVolume; Tt, StepNo: integer);
  end;

constructor TTinyVaePreview.Create;
begin
  inherited Create;
  X0 := TNNetVolume.Create;
  AllFiniteSoFar := true;
end;

destructor TTinyVaePreview.Destroy;
begin
  X0.Free;
  inherited Destroy;
end;

procedure TTinyVaePreview.MaybeEmit(Xt, RawPred: TNNetVolume; Tt, StepNo: integer);
var
  PreviewImg: TNNetVolume;
  PreviewFile: string;
  IsFinite: boolean;
begin
  if (Net = nil) or (Every <= 0) or (StepNo mod Every <> 0) then Exit;
  Scheduler.PredictX0(Xt, RawPred, X0, Tt);
  Net.Compute(X0);
  Net.GetLastLayer().ForceOutputOnRAM();
  PreviewImg := Net.GetLastLayer().Output;
  IsFinite := AllFinite(X0) and AllFinite(PreviewImg);
  AllFiniteSoFar := AllFiniteSoFar and IsFinite;
  if WriteFiles then
  begin
    PreviewFile := Format('%s%.3d.ppm', [cPreviewPrefix, StepNo]);
    WriteColorPPM(PreviewImg, PreviewFile);
    WriteLn('    preview step ', StepNo, ' -> ', PreviewFile);
  end
  else
    WriteLn('    preview step ', StepNo, ' ', PreviewImg.SizeX, 'x',
      PreviewImg.SizeY, '  finite=', BoolToStr(IsFinite, true));
end;

// Single-pass denoiser wrapper for the LCM few-step driver. TNNetLCMScheduler.
// LCMSample wants a method-of-object callback (Xt,Output,Tt) that returns the
// raw model output for ONE forward pass -- guidance is baked into a distilled
// consistency model, so the LCM branch uses ONLY the conditional (prompt) pass,
// no cond/uncond CFG. This object just closes over the net/config/prompt states.
type
  TLCMDenoiser = class
  public
    Net: TNNet;
    Cfg: TPixArtConfig;
    Prompt: TNNetVolume;
    Preview: TTinyVaePreview;
    StepNo: integer;
    procedure Denoise(Xt, Output: TNNetVolume; Tt: integer);
  end;

procedure TLCMDenoiser.Denoise(Xt, Output: TNNetVolume; Tt: integer);
begin
  // ONE conditional model pass -- no classifier-free guidance in the LCM path.
  PixArtDenoise(Net, Cfg, Xt, Tt, Prompt, Output);
  Inc(StepNo);
  Preview.MaybeEmit(Xt, Output, Tt, StepNo);
end;

var
  PixArtNet, VaeNet, TinyVaeNet: TNNet;
  PixCfg: TPixArtConfig;
  VaeCfg: TVaeDecoderConfig;
  TinyCfg: TTinyVaeDecoderConfig;
  Preview: TTinyVaePreview;
  Scheduler: TNNetDiffusionScheduler;
  LCMScheduler: TNNetLCMScheduler;
  LCMDenoiser: TLCMDenoiser;
  Latent, TextStates, NullStates, EpsCond, EpsUncond, EpsGuided, Image: TNNetVolume;
  PixST, PixCfgPath, VaeST, VaeCfgPath, TinyVaeDir, TinyST, TinyCfgPath: string;
  Method: TNNetSamplerMethod;
  NumSteps, StepCnt, T, TPrev, i: integer;
  Guidance: TNeuralFloat;
  APG: TNNetAdaptiveProjectedGuidance;
  ApgOnEps: boolean;
  SmokeMode, UseLCM, LatentOk, ImageOk, UseTiling, UseFastVae: boolean;
  UsingFixtures: boolean;
  PreviewEvery: integer;
  TileLatent: integer;
  Arg: string;
begin
  // ---- argument parsing ----
  PixST := cFixturePixArtST; PixCfgPath := cFixturePixArtCfg;
  VaeST := cFixtureVaeST;    VaeCfgPath := cFixtureVaeCfg;
  NumSteps := 4;
  Guidance := 4.0;
  Method := smDDIM;
  SmokeMode := false;
  UseLCM := false;
  UseTiling := false;
  UseFastVae := false;
  PreviewEvery := 0;
  TinyVaeDir := '';
  UsingFixtures := true;
  APG := nil;
  ApgOnEps := false;
  // Positional checkpoint args (the first two non-flag args).
  i := 1;
  if (ParamCount >= 1) and (Copy(ParamStr(1), 1, 2) <> '--') then
  begin
    PixST := ParamStr(1);
    PixCfgPath := ExtractFilePath(PixST) + 'config.json';
    i := 2;
    UsingFixtures := false;
    if (ParamCount >= 2) and (Copy(ParamStr(2), 1, 2) <> '--') then
    begin
      VaeST := ParamStr(2);
      VaeCfgPath := ExtractFilePath(VaeST) + 'config.json';
      i := 3;
    end;
    WriteLn('Loading PixArt + VAE from checkpoints: ', PixST, ' / ', VaeST);
  end
  else
    WriteLn('No checkpoints passed; using the committed pico fixtures ' +
      '(random weights -- wiring SMOKE demo).');
  while i <= ParamCount do
  begin
    Arg := ParamStr(i);
    if Arg = '--steps' then begin Inc(i); NumSteps := StrToIntDef(ParamStr(i), NumSteps); end
    else if Arg = '--cfg' then begin Inc(i); Guidance := StrToFloatDef(ParamStr(i), Guidance); end
    else if Arg = '--apg' then
    begin
      Inc(i);
      APG.Free;
      APG := TNNetAdaptiveProjectedGuidance.CreateFromSpec(ParamStr(i));
    end
    else if Arg = '--apg-eps' then ApgOnEps := true
    else if Arg = '--dpm' then Method := smDPMSolverPP2M
    else if Arg = '--unipc' then Method := smUniPC
    else if Arg = '--lcm' then UseLCM := true
    else if Arg = '--vae-tiling' then UseTiling := true
    else if Arg = '--tiny-vae' then begin Inc(i); TinyVaeDir := ParamStr(i); end
    else if Arg = '--preview' then begin Inc(i); PreviewEvery := StrToIntDef(ParamStr(i), PreviewEvery); end
    else if Arg = '--fast-vae' then UseFastVae := true
    else if Arg = '--smoke' then SmokeMode := true
    else WriteLn('Ignoring unknown argument: ', Arg);
    Inc(i);
  end;

  if TinyVaeDir <> '' then
  begin
    TinyST := IncludeTrailingPathDelimiter(TinyVaeDir) +
      'diffusion_pytorch_model.safetensors';
    TinyCfgPath := IncludeTrailingPathDelimiter(TinyVaeDir) + 'config.json';
  end
  else if UsingFixtures then
  begin
    TinyST := cFixtureTinyVaeST;
    TinyCfgPath := cFixtureTinyVaeCfg;
  end
  else if (PreviewEvery > 0) or UseFastVae then
  begin
    WriteLn('--preview / --fast-vae need --tiny-vae DIR with real checkpoints.');
    Halt(2);
  end;
  if UseFastVae and UseTiling then
  begin
    WriteLn('--fast-vae decodes with the tiny decoder; ignoring --vae-tiling.');
    UseTiling := false;
  end;

  RandSeed := 424242;
  PixArtNet := BuildPixArtFromSafeTensors(PixST, cTextSeqLen, PixCfg,
    {pTrainable=}false, PixCfgPath);
  VaeNet := nil;
  if not UseFastVae then
    VaeNet := BuildVaeDecoderFromSafeTensors(VaeST, VaeCfg,
      {pTrainable=}false, VaeCfgPath);
  // One tiny decoder, sized to the sampled latent grid (TAESD configs carry no size).
  TinyVaeNet := nil;
  if (PreviewEvery > 0) or UseFastVae then
  begin
    TinyCfg := ReadTinyVaeDecoderConfigFromJSONFile(TinyCfgPath);
    if TinyCfg.LatentChannels <> PixCfg.InChannels then
      raise Exception.Create('PixArt in_channels must equal the tiny VAE latent_channels.');
    TinyCfg.LatentWidth := PixCfg.SampleSize;
    TinyCfg.LatentHeight := PixCfg.SampleSize;
    TinyVaeNet := BuildTinyVaeDecoderFromSafeTensorsEx(TinyST, TinyCfg,
      {pTrainable=}false);
  end;
  Preview := TTinyVaePreview.Create;
  Preview.Net := TinyVaeNet;
  Preview.Every := PreviewEvery;
  Preview.WriteFiles := not SmokeMode;
  Scheduler := TNNetDiffusionScheduler.Create(100, dsLinear, dpEps);
  LCMScheduler := TNNetLCMScheduler.Create(100, dsLinear, dpEps);
  LCMDenoiser := TLCMDenoiser.Create;
  Latent := TNNetVolume.Create;
  TextStates := TNNetVolume.Create;
  NullStates := TNNetVolume.Create;
  EpsCond := TNNetVolume.Create;
  EpsUncond := TNNetVolume.Create;
  EpsGuided := TNNetVolume.Create;
  Image := TNNetVolume.Create;
  try
    WriteLn(PixArtConfigToString(PixCfg));
    if Assigned(VaeNet) then
    begin
      WriteLn('VAE decoder: latent grid ', VaeCfg.LatentGrid, ', latent ch ',
        VaeCfg.LatentChannels, ', out ch ', VaeCfg.OutChannels);
      if PixCfg.InChannels <> VaeCfg.LatentChannels then
        raise Exception.Create('PixArt in_channels must equal VAE latent_channels.');
      if PixCfg.SampleSize <> VaeCfg.LatentGrid then
        raise Exception.Create('PixArt sample_size must equal VAE latent grid.');
    end;
    if Assigned(TinyVaeNet) then
      WriteLn('Tiny VAE decoder (', TinyST, '): latent ', TinyCfg.LatentWidth,
        'x', TinyCfg.LatentHeight, 'x', TinyCfg.LatentChannels, ' -> image ',
        TinyVaeNet.GetLastLayer().Output.SizeX, 'x',
        TinyVaeNet.GetLastLayer().Output.SizeY, '; preview every ', PreviewEvery,
        ', fast final decode ', BoolToStr(UseFastVae, true));
    if UseLCM then
      WriteLn('Sampler: LCM (Latent Consistency Model, single pass/step, ',
        'guidance baked in), steps ', NumSteps)
    else
    begin
      case Method of
        smUniPC:         WriteLn('Sampler: UniPC (order-2 bh2 predictor-corrector,',
          ' few-step), steps ', NumSteps, ', CFG ', Guidance:0:2);
        smDPMSolverPP2M: WriteLn('Sampler: DPM-Solver++(2M), steps ', NumSteps,
          ', CFG ', Guidance:0:2);
        else             WriteLn('Sampler: DDIM, steps ', NumSteps,
          ', CFG ', Guidance:0:2);
      end;
      if Assigned(APG) then
        WriteLn('Guidance: APG on ', BoolToStr(ApgOnEps, 'eps', 'x0'), ', eta ',
          APG.Eta:0:3, ', momentum ', APG.Momentum:0:3, ', norm threshold ',
          APG.NormThreshold:0:3);
    end;

    // Caller-supplied T5 states (Step 3 = a real T5 tower over a tokenized
    // prompt). Deterministic synthetic states stand in for the pico run.
    TextStates.ReSize(cTextSeqLen, 1, PixCfg.CaptionChannels);
    for i := 0 to TextStates.Size - 1 do
      TextStates.FData[i] := Sin(i * 0.31) * 0.5;
    // Null/empty-caption uncond branch = zero states (PixArt CFG convention).
    NullStates.ReSize(cTextSeqLen, 1, PixCfg.CaptionChannels);
    NullStates.Fill(0);

    // Start the reverse trajectory from latent noise.
    Latent.ReSize(PixCfg.SampleSize, PixCfg.SampleSize, PixCfg.InChannels);
    Latent.RandomizeGaussian(1.0);
    EpsCond.ReSize(PixCfg.SampleSize, PixCfg.SampleSize, PixCfg.InChannels);
    EpsUncond.ReSize(PixCfg.SampleSize, PixCfg.SampleSize, PixCfg.InChannels);
    EpsGuided.ReSize(PixCfg.SampleSize, PixCfg.SampleSize, PixCfg.InChannels);

    if UseLCM then
    begin
      // LCM few-step path: a SINGLE conditional model pass per step (guidance is
      // baked into a distilled consistency model -- no cond/uncond CFG), driven
      // by TNNetLCMScheduler.LCMSample over its boundary-scaling consistency
      // function. The driver leaves the sampled x0 estimate in Latent.
      LCMDenoiser.Net := PixArtNet;
      LCMDenoiser.Cfg := PixCfg;
      LCMDenoiser.Prompt := TextStates;
      LCMDenoiser.Preview := Preview;
      Preview.Scheduler := LCMScheduler;
      LCMScheduler.LCMSample(Latent, @LCMDenoiser.Denoise, NumSteps, tsUniform);
      WriteLn('  LCM done in ', NumSteps, ' step(s)  |latent|=',
        Latent.GetMagnitude():0:4);
    end
    else
    begin
      Scheduler.ResetMultistep;
      if Assigned(APG) then APG.Reset;
      Preview.Scheduler := Scheduler;
      for StepCnt := 0 to NumSteps - 1 do
      begin
        T := Scheduler.NumTimesteps -
          (StepCnt * Scheduler.NumTimesteps) div NumSteps;
        if StepCnt = NumSteps - 1 then TPrev := 0
        else TPrev := Scheduler.NumTimesteps -
          ((StepCnt + 1) * Scheduler.NumTimesteps) div NumSteps;
        // CFG: cond = prompt T5 states; uncond = null/empty caption (zeros).
        PixArtDenoise(PixArtNet, PixCfg, Latent, T, TextStates, EpsCond);
        PixArtDenoise(PixArtNet, PixCfg, Latent, T, NullStates, EpsUncond);
        if Assigned(APG) then
        begin
          // APG works on the denoised x0 (as in the paper) unless --apg-eps;
          // x0 is affine in eps with the same x_t, so mapping back is exact.
          if not ApgOnEps then
          begin
            Scheduler.PredictX0(Latent, EpsCond, EpsCond, T);
            Scheduler.PredictX0(Latent, EpsUncond, EpsUncond, T);
          end;
          APG.ApplyAPG(EpsCond, EpsUncond, EpsGuided, Guidance);
          if not ApgOnEps then
            Scheduler.RawPredFromX0(Latent, EpsGuided, EpsGuided, T);
        end
        else
          // Guidance-rescale (Lin et al. 2023): counteract the over-exposure of
          // high CFG by pulling the result's std back toward the conditional's.
          TNNetDiffusionScheduler.ApplyCFG(EpsCond, EpsUncond, EpsGuided,
            Guidance, 0.7);
        Preview.MaybeEmit(Latent, EpsGuided, T, StepCnt + 1);
        Scheduler.Step(Latent, EpsGuided, T, TPrev, Method, 0.0);
        WriteLn('  step ', StepCnt + 1, '/', NumSteps, '  t=', T,
          '  |latent|=', Latent.GetMagnitude():0:4);
      end;
    end;

    LatentOk := AllFinite(Latent);
    WriteLn('Sampled latent ', Latent.SizeX, 'x', Latent.SizeY, 'x',
      Latent.Depth, '  finite=', BoolToStr(LatentOk, true));

    // Step 2: decode the latent to RGB (the /0.18215 scaling is inside the net).
    if UseFastVae then
    begin
      TinyVaeNet.Compute(Latent);
      TinyVaeNet.GetLastLayer().ForceOutputOnRAM();
      Image.Copy(TinyVaeNet.GetLastLayer().Output);
      WriteLn('Fast VAE: final decode with the tiny decoder.');
    end
    else if UseTiling then
    begin
      // diffusers enable_vae_tiling: decode the latent in OVERLAPPING tiles and
      // feather-blend the seams, bounding peak memory to a single tile. The
      // tile-sized decoder shares the full decoder's weights (conv weights are
      // grid-independent); we build one at the chosen tile latent grid. Clamp
      // the tile to the latent so the fixture (tiny latent) is a 1-tile no-op.
      TileLatent := 32;
      if TileLatent > VaeCfg.LatentGrid then TileLatent := VaeCfg.LatentGrid;
      // Rebuild the decoder at the TILE latent grid (same weights; conv weights
      // are grid-independent so only the fixed-size Input/Output layers shrink).
      VaeNet.Free; VaeNet := nil;
      VaeCfg.LatentGrid := TileLatent;
      VaeNet := BuildVaeDecoderFromSafeTensorsEx(VaeST, VaeCfg, {pTrainable=}false);
      WriteLn('VAE tiling ON: tile latent ', TileLatent, ' overlap 0.25 -> image ',
        'tile ', VaeNet.GetLastLayer().Output.SizeX);
      Image.Free;
      Image := TiledVaeDecode(VaeNet, Latent, TileLatent, 0.25);
    end
    else
    begin
      VaeNet.Compute(Latent);
      Image.Copy(VaeNet.GetLastLayer().Output);
    end;
    ImageOk := AllFinite(Image);
    WriteLn('Decoded image ', Image.SizeX, 'x', Image.SizeY, 'x', Image.Depth,
      '  finite=', BoolToStr(ImageOk, true));

    if SmokeMode then
    begin
      if Assigned(TinyVaeNet) then
        WriteLn('Previews finite=', BoolToStr(Preview.AllFiniteSoFar, true));
      if LatentOk and ImageOk and Preview.AllFiniteSoFar then
      begin
        WriteLn('SMOKE OK');
        ExitCode := 0;
      end
      else
      begin
        WriteLn('SMOKE FAIL');
        ExitCode := 1;
      end;
    end
    else
    begin
      WriteColorPPM(Image, cOutFile);
      WriteLn('Wrote ', cOutFile, '.');
      WriteLn; WriteLn('decoded image:'); AsciiPreview(Image);
    end;
  finally
    Image.Free;
    APG.Free;
    EpsGuided.Free;
    EpsUncond.Free;
    EpsCond.Free;
    NullStates.Free;
    TextStates.Free;
    Latent.Free;
    LCMDenoiser.Free;
    Preview.Free;
    TinyVaeNet.Free;
    LCMScheduler.Free;
    Scheduler.Free;
    VaeNet.Free;
    PixArtNet.Free;
  end;
end.
