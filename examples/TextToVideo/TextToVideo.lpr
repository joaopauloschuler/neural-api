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
    -> 3D-causal-conv VAE decode tail (BuildCogVideoXVaeDecoderFromSafeTensorsEx
       + DecodeCogVideoXVae): a depth-axis CAUSAL temporal convolution over each
       spatial cell (left-pad time, no future-frame leakage) + SiLU + pointwise
       conv
    -> per-frame RGB -> a sequence of P6 PPM files (frame_00.ppm, frame_01.ppm,
       ...), the same writer style as the LatentTextToImage / AnimateDiff demos.

Nothing here is a new leaf layer: it is pure plumbing over landed pieces (the
CogVideoX importer + CogVideoXConditioning, the VAE-decode tail, and the
TNNetDiffusionScheduler DDIM / DPM-Solver++ driver). The two genuinely-new
CogVideoX primitives -- the 3D causal-conv VAE (TNNetCausalConv1D over the
VideoMAE space<->time view) and 3D RoPE (TNNetMRotaryEmbedding) -- are reused
landed leaves, not new code.

NO NETWORK ACCESS / SELF-CONTAINED: falls back to the committed pico CogVideoX
fixture (tests/fixtures/tiny_cogvideox.*, tools/make_pico_cogvideox_fixture.py,
parity-checked < 1e-4 in TestCogVideoXParity). Random weights -> this is a
WIRING / THROUGHPUT smoke (it proves the chain runs offline and produces FINITE
video frames, not photorealism).

USAGE
  ./TextToVideo                 use the committed pico fixture
  ./TextToVideo cogvideox.safetensors   use your own checkpoint (sibling config.json)
Optional trailing flags:
  --steps N      number of reverse steps (default 4)
  --dpm          use DPM-Solver++(2M) instead of DDIM
  --smoke        run, assert finiteness, print SMOKE OK/FAIL and exit (no PPM)

OUTPUT
  Writes frame_<ii>.ppm for each decoded frame and prints the config, the
  per-step trajectory norm and an ASCII preview of each frame. With --smoke it
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
  SysUtils, Classes, Math,
  neuralnetwork, neuralvolume, neuralpretrained, neuraldiffusion;

const
  cFixtureST  = '../../tests/fixtures/tiny_cogvideox.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_cogvideox_config.json';

function ToByte(V: TNeuralFloat): byte;
begin
  V := (V + 1) * 0.5;
  if V < 0 then V := 0;
  if V > 1 then V := 1;
  Result := Round(V * 255);
end;

// Writes ONE frame (a (GridH, GridW, Vout) volume) as a P6 PPM.
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
  Net, VaeNet: TNNet;
  Cfg: TCogVideoXConfig;
  Scheduler: TNNetDiffusionScheduler;
  Latent, TextStates, Eps, VaeLatent, Decoded, Frame: TNNetVolume;
  ST, CfgPath: string;
  Method: TNNetSamplerMethod;
  NumSteps, StepCnt, T, TPrev, i, LV, ft, hh, ww, c, cell: integer;
  SmokeMode, LatentOk, DecodedOk: boolean;
  Arg: string;
begin
  ST := cFixtureST; CfgPath := cFixtureCfg;
  NumSteps := 4;
  Method := smDDIM;
  SmokeMode := false;
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
    else WriteLn('Ignoring unknown argument: ', Arg);
    Inc(i);
  end;

  RandSeed := 424242;
  Net := BuildCogVideoXFromSafeTensors(ST, Cfg, {pTrainable=}false, CfgPath);
  VaeNet := BuildCogVideoXVaeDecoderFromSafeTensorsEx(ST, Cfg, {pTrainable=}false);
  Scheduler := TNNetDiffusionScheduler.Create(100, dsLinear, dpEps);
  LV := Cfg.NumFrames * Cfg.GridHeight * Cfg.GridWidth;
  Latent := TNNetVolume.Create;
  TextStates := TNNetVolume.Create;
  Eps := TNNetVolume.Create;
  VaeLatent := TNNetVolume.Create;
  Decoded := TNNetVolume.Create;
  Frame := TNNetVolume.Create;
  try
    WriteLn(CogVideoXConfigToString(Cfg));
    WriteLn('Sampler: ', BoolToStr(Method = smDPMSolverPP2M, 'DPM-Solver++(2M)',
      'DDIM'), ', steps ', NumSteps);

    // Deterministic synthetic T5 states stand in for a real T5 tower.
    TextStates.ReSize(Cfg.TextSeqLen, 1, Cfg.TextDim);
    for i := 0 to TextStates.Size - 1 do
      TextStates.FData[i] := Sin(i * 0.37) * 0.5;

    // Start the reverse trajectory from latent noise (LV,1,in_channels).
    Latent.ReSize(LV, 1, Cfg.InChannels);
    Latent.RandomizeGaussian(1.0);
    Eps.ReSize(LV, 1, Cfg.OutChannels);

    Scheduler.ResetMultistep;
    for StepCnt := 0 to NumSteps - 1 do
    begin
      T := Scheduler.NumTimesteps - (StepCnt * Scheduler.NumTimesteps) div NumSteps;
      if StepCnt = NumSteps - 1 then TPrev := 0
      else TPrev := Scheduler.NumTimesteps -
        ((StepCnt + 1) * Scheduler.NumTimesteps) div NumSteps;
      CogVideoXConditioning(Net, Cfg, T, TextStates);
      Net.Compute(Latent);
      // eps prediction is (LV,1,out_channels). Keep only the in_channels half
      // (here out=in, so copy whole) into Eps for the scheduler step.
      Eps.CopyNoChecks(Net.GetLastLayer().Output);
      Scheduler.Step(Latent, Eps, T, TPrev, Method, 0.0);
      WriteLn('  step ', StepCnt + 1, '/', NumSteps, '  t=', T,
        '  |latent|=', Latent.GetMagnitude():0:4);
    end;

    LatentOk := AllFinite(Latent);
    WriteLn('Sampled video latent frames=', Cfg.NumFrames, ' grid=',
      Cfg.GridHeight, 'x', Cfg.GridWidth, ' ch=', Cfg.InChannels,
      '  finite=', BoolToStr(LatentOk, true));

    // VAE decode: pack the sampled latent into the (T, HW, Clat) channel-major
    // layout DecodeCogVideoXVae expects, then decode every spatial cell.
    VaeLatent.ReSize(Cfg.NumFrames, Cfg.GridHeight * Cfg.GridWidth,
      Cfg.VaeLatentChannels);
    i := 0;
    for ft := 0 to Cfg.NumFrames - 1 do
      for hh := 0 to Cfg.GridHeight - 1 do
        for ww := 0 to Cfg.GridWidth - 1 do
        begin
          cell := hh * Cfg.GridWidth + ww;
          for c := 0 to Cfg.VaeLatentChannels - 1 do
          begin
            VaeLatent[ft, cell, c] := Latent.FData[i];
            Inc(i);
          end;
        end;
    DecodeCogVideoXVae(VaeNet, Cfg, VaeLatent, Decoded);
    DecodedOk := AllFinite(Decoded);
    WriteLn('Decoded video ', Cfg.NumFrames, ' frames of ',
      Cfg.GridHeight, 'x', Cfg.GridWidth, 'x', Cfg.VaeOutChannels,
      '  finite=', BoolToStr(DecodedOk, true));

    if SmokeMode then
    begin
      if LatentOk and DecodedOk then begin WriteLn('SMOKE OK'); ExitCode := 0; end
      else begin WriteLn('SMOKE FAIL'); ExitCode := 1; end;
    end
    else
    begin
      // Unpack each frame (GridH, GridW, Vout) from the (T, HW, Vout) decode.
      for ft := 0 to Cfg.NumFrames - 1 do
      begin
        Frame.ReSize(Cfg.GridWidth, Cfg.GridHeight, Cfg.VaeOutChannels);
        for hh := 0 to Cfg.GridHeight - 1 do
          for ww := 0 to Cfg.GridWidth - 1 do
          begin
            cell := hh * Cfg.GridWidth + ww;
            for c := 0 to Cfg.VaeOutChannels - 1 do
              Frame[ww, hh, c] := Decoded[ft, cell, c];
          end;
        WriteFramePPM(Frame, Format('frame_%.2d.ppm', [ft]));
        WriteLn; WriteLn('frame ', ft, ':'); AsciiPreview(Frame);
      end;
      WriteLn('Wrote ', Cfg.NumFrames, ' frame_NN.ppm files.');
    end;
  finally
    Frame.Free;
    Decoded.Free;
    VaeLatent.Free;
    Eps.Free;
    TextStates.Free;
    Latent.Free;
    Scheduler.Free;
    VaeNet.Free;
    Net.Free;
  end;
end.
