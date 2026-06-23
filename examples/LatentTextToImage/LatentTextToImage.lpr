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
  --dpm          use DPM-Solver++(2M) instead of DDIM
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
  cTextSeqLen       = 5;
  cOutFile          = 'latent_text_to_image.ppm';

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

var
  PixArtNet, VaeNet: TNNet;
  PixCfg: TPixArtConfig;
  VaeCfg: TVaeDecoderConfig;
  Scheduler: TNNetDiffusionScheduler;
  Latent, TextStates, NullStates, EpsCond, EpsUncond, EpsGuided, Image: TNNetVolume;
  PixST, PixCfgPath, VaeST, VaeCfgPath: string;
  Method: TNNetSamplerMethod;
  NumSteps, StepCnt, T, TPrev, i: integer;
  Guidance: TNeuralFloat;
  SmokeMode, LatentOk, ImageOk: boolean;
  Arg: string;
begin
  // ---- argument parsing ----
  PixST := cFixturePixArtST; PixCfgPath := cFixturePixArtCfg;
  VaeST := cFixtureVaeST;    VaeCfgPath := cFixtureVaeCfg;
  NumSteps := 4;
  Guidance := 4.0;
  Method := smDDIM;
  SmokeMode := false;
  // Positional checkpoint args (the first two non-flag args).
  i := 1;
  if (ParamCount >= 1) and (Copy(ParamStr(1), 1, 2) <> '--') then
  begin
    PixST := ParamStr(1);
    PixCfgPath := ExtractFilePath(PixST) + 'config.json';
    i := 2;
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
    else if Arg = '--dpm' then Method := smDPMSolverPP2M
    else if Arg = '--smoke' then SmokeMode := true
    else WriteLn('Ignoring unknown argument: ', Arg);
    Inc(i);
  end;

  RandSeed := 424242;
  PixArtNet := BuildPixArtFromSafeTensors(PixST, cTextSeqLen, PixCfg,
    {pTrainable=}false, PixCfgPath);
  VaeNet := BuildVaeDecoderFromSafeTensors(VaeST, VaeCfg,
    {pTrainable=}false, VaeCfgPath);
  Scheduler := TNNetDiffusionScheduler.Create(100, dsLinear, dpEps);
  Latent := TNNetVolume.Create;
  TextStates := TNNetVolume.Create;
  NullStates := TNNetVolume.Create;
  EpsCond := TNNetVolume.Create;
  EpsUncond := TNNetVolume.Create;
  EpsGuided := TNNetVolume.Create;
  Image := TNNetVolume.Create;
  try
    WriteLn(PixArtConfigToString(PixCfg));
    WriteLn('VAE decoder: latent grid ', VaeCfg.LatentGrid, ', latent ch ',
      VaeCfg.LatentChannels, ', out ch ', VaeCfg.OutChannels);
    if PixCfg.InChannels <> VaeCfg.LatentChannels then
      raise Exception.Create('PixArt in_channels must equal VAE latent_channels.');
    if PixCfg.SampleSize <> VaeCfg.LatentGrid then
      raise Exception.Create('PixArt sample_size must equal VAE latent grid.');
    WriteLn('Sampler: ', BoolToStr(Method = smDPMSolverPP2M, 'DPM-Solver++(2M)',
      'DDIM'), ', steps ', NumSteps, ', CFG ', Guidance:0:2);

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

    Scheduler.ResetMultistep;
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
      TNNetDiffusionScheduler.ApplyCFG(EpsCond, EpsUncond, EpsGuided, Guidance);
      Scheduler.Step(Latent, EpsGuided, T, TPrev, Method, 0.0);
      WriteLn('  step ', StepCnt + 1, '/', NumSteps, '  t=', T,
        '  |latent|=', Latent.GetMagnitude():0:4);
    end;

    LatentOk := AllFinite(Latent);
    WriteLn('Sampled latent ', Latent.SizeX, 'x', Latent.SizeY, 'x',
      Latent.Depth, '  finite=', BoolToStr(LatentOk, true));

    // Step 2: decode the latent to RGB (the /0.18215 scaling is inside the net).
    VaeNet.Compute(Latent);
    Image.Copy(VaeNet.GetLastLayer().Output);
    ImageOk := AllFinite(Image);
    WriteLn('Decoded image ', Image.SizeX, 'x', Image.SizeY, 'x', Image.Depth,
      '  finite=', BoolToStr(ImageOk, true));

    if SmokeMode then
    begin
      if LatentOk and ImageOk then
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
    EpsGuided.Free;
    EpsUncond.Free;
    EpsCond.Free;
    NullStates.Free;
    TextStates.Free;
    Latent.Free;
    Scheduler.Free;
    VaeNet.Free;
    PixArtNet.Free;
  end;
end.
