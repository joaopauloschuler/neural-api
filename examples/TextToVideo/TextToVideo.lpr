program TextToVideo;
(*
TextToVideo: a complete OFFLINE, CPU native text-to-VIDEO pipeline that chains
the landed CogVideoX importer into one sampling loop -- the THUDM/CogVideoX
recipe end to end, distinct from AnimateDiff (which bolts a temporal module onto
a frozen SD UNet):

  caller-supplied T5 text states  (a real T5 tower is a follow-up)
    -> CogVideoX flat-DiT denoiser (BuildCogVideoXFromSafeTensors): a flat
       MMDiT-style transformer over a flattened (frame x height x width) latent
       token sequence with expert adaLN-Zero modulation + 3D RoPE on the video
       tokens
    -> multi-step DDIM / DPM-Solver++ reverse loop (TNNetDiffusionScheduler)
    -> sampled (NumFrames, GridH, GridW, in_channels) video latent
    -> AutoencoderKLCogVideoX decoder (TCogVideoXVaeDecoder, 2 latent frames
       per batch with the diffusers conv cache): causal 3D convs, SpatialNorm3D
       resnets and up blocks that upsample space 2x per block and time 2x on
       the first log2(temporal_compression_ratio) blocks
    -> per-frame RGB -> a sequence of P6 PPM files (frame_00.ppm, frame_01.ppm,
       ...) or PNG files, and optionally one looping animated GIF
       (TNNetAnimatedGIFWriter, neuralgif.pas).

Nothing here is a new leaf layer: it is pure plumbing over landed pieces (the
CogVideoX importer + CogVideoXConditioning, the VAE decoder, and the
TNNetDiffusionScheduler DDIM / DPM-Solver++ driver). The two genuinely-new
CogVideoX primitives -- the 3D causal-conv VAE (TNNetConvolution3D over a
frame-packed volume) and 3D RoPE (TNNetMRotaryEmbedding) -- are reused landed
leaves, not new code.

NO NETWORK ACCESS / SELF-CONTAINED: falls back to the committed pico CogVideoX
fixture (tests/fixtures/tiny_cogvideox.* + tests/fixtures/tiny_cogvideox_vae/,
tools/make_pico_cogvideox_fixture.py, parity-checked < 1e-4 in
TestCogVideoXParity and TestCogVideoXVaeDecoderParity). Random weights -> this is a
WIRING / THROUGHPUT smoke (it proves the chain runs offline and produces FINITE
video frames, not photorealism).

USAGE
  ./TextToVideo                 use the committed pico fixture
  ./TextToVideo cogvideox.safetensors   use your own checkpoint (sibling config.json)
Optional trailing flags:
  --vae PATH     AutoencoderKLCogVideoX: a diffusers vae folder (config.json +
                 diffusion_pytorch_model.safetensors) or a .safetensors with a
                 sibling config.json (default: the pico tiny_cogvideox_vae)
  --steps N      number of reverse steps (default 4)
  --dpm          use DPM-Solver++(2M) instead of DDIM
  --smoke        run, assert finiteness, print SMOKE OK/FAIL and exit (no PPM)
  --gif FILE     also write all frames as one looping animated GIF
  --fps N        GIF frame rate (default 8, the CogVideoX rate)
  --png          write frame_<ii>.png instead of frame_<ii>.ppm
  --fbc-threshold X  First-Block Cache: skip blocks 1..N-1 on steps whose
                 block 0 residual changed by at most X (0 = off, the default)
  --vae-tiling   diffusers enable_tiling(): decode a latent wider or taller
                 than one 240x360-sample tile in overlapping blended tiles, so
                 the VAE RAM follows the tile, not the frame (for 480x720)

OUTPUT
  Writes frame_<ii>.ppm (or .png) for each decoded frame, the --gif file if
  asked, and prints the config, the per-step trajectory norm and an ASCII
  preview of each frame. With --smoke it
  prints "SMOKE OK"/"SMOKE FAIL" and sets the exit code (the build step
  exercises this).

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX} cthreads, {$ENDIF}
  SysUtils, Classes, Math,
  neuralnetwork, neuralvolume, neuralpretrained, neuraldiffusion,
  neuraldatasets, neuralgif;

const
  cFixtureST  = '../../tests/fixtures/tiny_cogvideox.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_cogvideox_config.json';
  cFixtureVae = '../../tests/fixtures/tiny_cogvideox_vae';

function ToByte(V: TNeuralFloat): byte;
begin
  V := (V + 1) * 0.5;
  if V < 0 then V := 0;
  if V > 1 then V := 1;
  Result := Round(V * 255);
end;

// Writes ONE frame (a (W, H, channels) volume) as a P6 PPM.
procedure WriteFramePPM(Frame: TNNetVolume; const FileName: string);
var
  F: TextFile;
  Bin: TFileStream;
  X, Y, W, H: integer;
  R, G, B: byte;
begin
  W := Frame.SizeX; H := Frame.SizeY;
  AssignFile(F, FileName); Rewrite(F);
  WriteLn(F, 'P6'); WriteLn(F, W, ' ', H); WriteLn(F, 255);
  CloseFile(F);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for Y := 0 to H - 1 do
      for X := 0 to W - 1 do
      begin
        R := ToByte(Frame[X, Y, 0]);
        if Frame.Depth > 1 then G := ToByte(Frame[X, Y, 1]) else G := R;
        if Frame.Depth > 2 then B := ToByte(Frame[X, Y, 2]) else B := R;
        Bin.WriteByte(R); Bin.WriteByte(G); Bin.WriteByte(B);
      end;
  finally
    Bin.Free;
  end;
end;

// Writes ONE frame as PNG: [-1,1] maps to 0..255, a 1-channel frame becomes gray.
procedure WriteFramePNG(Frame, Work: TNNetVolume; const FileName: string);
var
  X, Y, C, MaxChannel: integer;
begin
  Work.ReSize(Frame.SizeX, Frame.SizeY, 3);
  MaxChannel := Min(Frame.Depth, 3) - 1;
  for Y := 0 to Frame.SizeY - 1 do
    for X := 0 to Frame.SizeX - 1 do
      for C := 0 to 2 do
        Work[X, Y, C] := ToByte(Frame[X, Y, Min(C, MaxChannel)]);
  SaveImageFromVolumeIntoFile(Work, FileName);
end;

procedure AsciiPreview(Frame: TNNetVolume);
const cRamp = ' .:-=+*#%@';
var X, Y, idx: integer; v: TNeuralFloat;
begin
  for Y := 0 to Frame.SizeY - 1 do
  begin
    for X := 0 to Frame.SizeX - 1 do
    begin
      v := Frame[X, Y, 0];
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
    if (V.FData[i] <> V.FData[i]) or (Abs(V.FData[i]) > 1e30) then
    begin
      Result := false;
      Exit;
    end;
end;

var
  Net: TNNet;
  VaeDecoder: TCogVideoXVaeDecoder;
  Cfg: TCogVideoXConfig;
  Sampler: TCogVideoXSampler;
  FirstBlockCacheThreshold: TNeuralFloat;
  SkippedBefore: integer;
  StepSkipped: boolean;
  Scheduler: TNNetDiffusionScheduler;
  Latent, TextStates, Eps, Decoded, Frame, PngWork: TNNetVolume;
  GifWriter: TNNetAnimatedGIFWriter;
  ST, CfgPath, VaePath, GifFile: string;
  FramesPerSecond: integer;
  Method: TNNetSamplerMethod;
  NumSteps, StepCnt, T, TPrev, i, LV, ft: integer;
  FrameW, FrameH, FrameCount, FrameFloats: integer;
  SmokeMode, LatentOk, DecodedOk, PngFrames, VaeTiling: boolean;
  Arg: string;
begin
  ST := cFixtureST; CfgPath := cFixtureCfg; VaePath := cFixtureVae;
  NumSteps := 4;
  Method := smDDIM;
  SmokeMode := false;
  PngFrames := false;
  VaeTiling := false;
  GifFile := '';
  FramesPerSecond := 8;
  FirstBlockCacheThreshold := 0;
  i := 1;
  if (ParamCount >= 1) and (Copy(ParamStr(1), 1, 2) <> '--') then
  begin
    ST := ParamStr(1);
    CfgPath := ExtractFilePath(ST) + 'config.json';
    i := 2;
    WriteLn('Loading CogVideoX from checkpoint: ', ST);
  end
  else
    WriteLn('No checkpoint passed; using the committed pico fixture ' +
      '(random weights -- wiring SMOKE demo).');
  while i <= ParamCount do
  begin
    Arg := ParamStr(i);
    if Arg = '--steps' then begin Inc(i); NumSteps := StrToIntDef(ParamStr(i), NumSteps); end
    else if Arg = '--dpm' then Method := smDPMSolverPP2M
    else if Arg = '--smoke' then SmokeMode := true
    else if Arg = '--vae' then begin Inc(i); VaePath := ParamStr(i); end
    else if Arg = '--gif' then begin Inc(i); GifFile := ParamStr(i); end
    else if Arg = '--fps' then begin Inc(i); FramesPerSecond := StrToIntDef(ParamStr(i), FramesPerSecond); end
    else if Arg = '--png' then PngFrames := true
    else if Arg = '--vae-tiling' then VaeTiling := true
    else if Arg = '--fbc-threshold' then
    begin
      Inc(i);
      FirstBlockCacheThreshold := StrToFloatDef(ParamStr(i), 0);
    end
    else WriteLn('Ignoring unknown argument: ', Arg);
    Inc(i);
  end;

  RandSeed := 424242;
  Net := BuildCogVideoXFromSafeTensors(ST, Cfg, {pTrainable=}false, CfgPath);
  VaeDecoder := TCogVideoXVaeDecoder.Create(VaePath, Cfg.NumFrames,
    Cfg.GridHeight, Cfg.GridWidth);
  if VaeDecoder.Config.LatentChannels <> Cfg.InChannels then
  begin
    WriteLn('The VAE takes ', VaeDecoder.Config.LatentChannels,
      ' latent channels; the denoiser makes ', Cfg.InChannels, '.');
    Halt(1);
  end;
  if VaeTiling then VaeDecoder.EnableTiling();
  Sampler := TCogVideoXSampler.Create(Net, Cfg);
  Sampler.FirstBlockCache.Threshold := FirstBlockCacheThreshold;
  Scheduler := TNNetDiffusionScheduler.Create(100, dsLinear, dpEps);
  LV := Cfg.NumFrames * Cfg.GridHeight * Cfg.GridWidth;
  Latent := TNNetVolume.Create;
  TextStates := TNNetVolume.Create;
  Eps := TNNetVolume.Create;
  Decoded := TNNetVolume.Create;
  Frame := TNNetVolume.Create;
  PngWork := TNNetVolume.Create;
  GifWriter := nil;
  try
    WriteLn(CogVideoXConfigToString(Cfg));
    WriteLn('Sampler: ', BoolToStr(Method = smDPMSolverPP2M, 'DPM-Solver++(2M)',
      'DDIM'), ', steps ', NumSteps);
    if Sampler.FirstBlockCache.Enabled then
      WriteLn('First-Block Cache threshold ', FirstBlockCacheThreshold:0:4,
        BoolToStr(Cfg.NumLayers < 2, ' (one block: nothing to skip)', ''));

    // Deterministic synthetic T5 states stand in for a real T5 tower.
    TextStates.ReSize(Cfg.TextSeqLen, 1, Cfg.TextDim);
    for i := 0 to TextStates.Size - 1 do
      TextStates.FData[i] := Sin(i * 0.37) * 0.5;

    // Start the reverse trajectory from latent noise (LV,1,in_channels).
    Latent.ReSize(LV, 1, Cfg.InChannels);
    Latent.RandomizeGaussian(1.0);
    Eps.ReSize(LV, 1, Cfg.OutChannels);

    Scheduler.ResetMultistep;
    Sampler.FirstBlockCache.Reset();
    for StepCnt := 0 to NumSteps - 1 do
    begin
      T := Scheduler.NumTimesteps - (StepCnt * Scheduler.NumTimesteps) div NumSteps;
      if StepCnt = NumSteps - 1 then TPrev := 0
      else TPrev := Scheduler.NumTimesteps -
        ((StepCnt + 1) * Scheduler.NumTimesteps) div NumSteps;
      SkippedBefore := Sampler.FirstBlockCache.SkippedStepCount;
      // eps prediction is (LV,1,out_channels); here out=in.
      Sampler.PredictNoise(Latent, T, TextStates, Sampler.FirstBlockCache, Eps);
      StepSkipped := Sampler.FirstBlockCache.SkippedStepCount > SkippedBefore;
      Scheduler.Step(Latent, Eps, T, TPrev, Method, 0.0);
      WriteLn('  step ', StepCnt + 1, '/', NumSteps, '  t=', T,
        '  |latent|=', Latent.GetMagnitude():0:4,
        BoolToStr(StepSkipped, '  (blocks 1..N-1 skipped)', ''));
    end;
    if Sampler.FirstBlockCache.Enabled then
      WriteLn('First-Block Cache skipped ', Sampler.FirstBlockCache.SkippedStepCount,
        ' of ', NumSteps, ' steps.');

    LatentOk := AllFinite(Latent);
    WriteLn('Sampled video latent frames=', Cfg.NumFrames, ' grid=',
      Cfg.GridHeight, 'x', Cfg.GridWidth, ' ch=', Cfg.InChannels,
      '  finite=', BoolToStr(LatentOk, true));

    // VAE decode: the latent is already flat (t, h, w, c); the frames come
    // back stacked along Y.
    VaeDecoder.Decode(Latent, Decoded);
    DecodedOk := AllFinite(Decoded);
    FrameW := Decoded.SizeX;
    FrameCount := VaeDecoder.OutputFrames;
    FrameH := Decoded.SizeY div FrameCount;
    FrameFloats := FrameW * FrameH * Decoded.Depth;
    WriteLn('Decoded video ', FrameCount, ' frames of ', FrameW, 'x', FrameH,
      'x', Decoded.Depth, '  finite=', BoolToStr(DecodedOk, true));
    if VaeTiling then
      WriteLn('VAE tiling: ', VaeDecoder.LastTileCount, ' tile(s) of ',
        VaeDecoder.TileLatentMinHeight, 'x', VaeDecoder.TileLatentMinWidth,
        ' latent cells (1 = the latent fits one tile).');

    if SmokeMode then
    begin
      if LatentOk and DecodedOk then begin WriteLn('SMOKE OK'); ExitCode := 0; end
      else begin WriteLn('SMOKE FAIL'); ExitCode := 1; end;
    end
    else
    begin
      // Frame ft is the contiguous row block [ft*FrameH, (ft+1)*FrameH).
      if GifFile <> '' then
        GifWriter := TNNetAnimatedGIFWriter.Create(GifFile, FrameW, FrameH,
          Max(1, Round(100 / Max(1, FramesPerSecond))));
      Frame.ReSize(FrameW, FrameH, Decoded.Depth);
      for ft := 0 to FrameCount - 1 do
      begin
        Move(Decoded.FData[ft * FrameFloats], Frame.FData[0],
          FrameFloats * csNeuralFloatSize);
        if PngFrames then WriteFramePNG(Frame, PngWork, Format('frame_%.2d.png', [ft]))
        else WriteFramePPM(Frame, Format('frame_%.2d.ppm', [ft]));
        if GifWriter <> nil then GifWriter.AddFrame(Frame, -1, 1);
        WriteLn; WriteLn('frame ', ft, ':'); AsciiPreview(Frame);
      end;
      WriteLn('Wrote ', FrameCount, ' frame_NN.', BoolToStr(PngFrames, 'png', 'ppm'),
        ' files.');
      if GifWriter <> nil then
        WriteLn('Wrote ', GifWriter.FrameCount, '-frame animated GIF ', GifFile, ' at ',
          FramesPerSecond, ' fps.');
    end;
  finally
    GifWriter.Free;
    PngWork.Free;
    Frame.Free;
    Decoded.Free;
    Eps.Free;
    TextStates.Free;
    Latent.Free;
    Scheduler.Free;
    Sampler.Free;
    VaeDecoder.Free;
    Net.Free;
  end;
end.
