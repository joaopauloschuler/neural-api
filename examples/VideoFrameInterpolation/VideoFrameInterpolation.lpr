program VideoFrameInterpolation;
(*
VideoFrameInterpolation: synthesises ONE intermediate frame (t=0.5) between two
input frames on the CPU end-to-end with the repo's RIFE frame-interpolation
importer (BuildRIFEFromSafeTensors, neuralpretrained.pas) -- a
VIDEO-generative import. The landed RAFT path estimates optical FLOW
but does NOT synthesise frames; examples/FrameInterpolation is a from-scratch toy
(TNNetFlowWarp), not an importer. RIFE (Huang et al. 2022, "Real-Time
Intermediate Flow Estimation for Video Frame Interpolation", arXiv:2011.06294;
hzwer/Practical-RIFE) estimates a bidirectional intermediate flow with a coarse-
to-fine stack of IFBlocks, BACKWARD-WARPS both frames, and blends them with a
learned soft fusion mask.

The new primitive is the differentiable backward-warp TNNetBackwardWarp
(RIFE convention: pixel-unit flow, bilinear, border-clamp -- the integer-pixel
equivalent of grid_sample(padding_mode='border', align_corners=True)). Everything
else (3x3 conv, per-channel PReLU, sigmoid, channel-broadcast multiply, residual
sum) reuses landed layers.

NO NETWORK ACCESS / SELF-CONTAINED: the official Practical-RIFE checkpoints are
not obtainable offline, so -- exactly like the repo's NAFNet / SwinIR pico
fixtures -- this falls back to the committed CONFIG-FAITHFUL random pico RIFE
(tests/fixtures/tiny_rife.*, built by tools/rife_tiny_fixture.py and parity-
checked < 1e-4 against a float64 numpy oracle in tests/TestNeuralPretrained.pas).
The pico net is random (not trained), so this is a wiring/throughput SMOKE demo:
it builds the net, makes two small synthetic frames (a bright blob translating
across the grid), stacks them on the depth axis as [frame0 | frame1], runs the
interpolation forward pass, and writes the result. Pass a real .safetensors
(+ config.json sibling) to interpolate with your own trained checkpoint.

USAGE
  ./VideoFrameInterpolation                 use the committed pico fixture
  ./VideoFrameInterpolation model.safetensors [config.json]
                                            use a real checkpoint

OUTPUT
  Writes rife_frame0.ppm, rife_frame1.ppm and rife_middle.ppm (P6 color images)
  and prints the config plus a small ASCII preview of frame0 / middle / frame1.

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
  neuralnetwork, neuralvolume, neuralpretrained;

const
  cFixtureST  = '../../tests/fixtures/tiny_rife.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_rife_config.json';
  cFrame0File = 'rife_frame0.ppm';
  cFrame1File = 'rife_frame1.ppm';
  cMiddleFile = 'rife_middle.ppm';

function ToByte(V: TNeuralFloat): byte;
begin
  // Map a roughly-[-1,1] activation into [0,255]; clamp out-of-range.
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

procedure AsciiPreview(Img: TNNetVolume; CBase: integer);
const cRamp = ' .:-=+*#%@';
var X, Y, idx: integer; v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
  begin
    for X := 0 to Img.SizeX - 1 do
    begin
      v := (Img[X, Y, CBase] + Img[X, Y, CBase + 1] + Img[X, Y, CBase + 2]) / 3;
      v := (v + 1) * 0.5; if v < 0 then v := 0; if v > 1 then v := 1;
      idx := Round(v * (Length(cRamp) - 1)) + 1;
      Write(cRamp[idx]);
    end;
    WriteLn;
  end;
end;

// A bright blob centred at (cx,cy) over a dark field, RGB in [-1,1]. Writing into
// channels [CBase..CBase+2] of a (S,S,depth) volume lets us pack two frames.
procedure FillBlob(Img: TNNetVolume; CBase: integer;
  cx, cy, radius: TNeuralFloat);
var X, Y: integer; d, v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
    for X := 0 to Img.SizeX - 1 do
    begin
      d := Sqrt(Sqr(X - cx) + Sqr(Y - cy));
      v := 1.0 - d / radius;            // 1 at centre, fading out
      if v < -1 then v := -1; if v > 1 then v := 1;
      // tint the blob slightly per channel so colour is visible
      Img[X, Y, CBase + 0] := v;
      Img[X, Y, CBase + 1] := v * 0.7;
      Img[X, Y, CBase + 2] := v * 0.4;
    end;
end;

var
  NN: TNNet;
  Config: TRIFEConfig;
  TwoFrames, Frame0, Frame1: TNNetVolume;
  S, InPlanes, X, Y, C: integer;
  STPath, CfgPath: string;
begin
  if ParamCount >= 1 then
  begin
    STPath := ParamStr(1);
    if ParamCount >= 2 then CfgPath := ParamStr(2)
    else CfgPath := ExtractFilePath(STPath) + 'config.json';
    WriteLn('Loading RIFE from checkpoint: ', STPath);
  end
  else
  begin
    STPath := cFixtureST;
    CfgPath := cFixtureCfg;
    WriteLn('No checkpoint passed; using the committed pico fixture ' +
      '(random weights -- wiring SMOKE demo).');
  end;

  NN := BuildRIFEFromSafeTensors(STPath, Config, {pInferenceOnly=}true, CfgPath);
  try
    WriteLn(RIFEConfigToString(Config));
    S := Config.InputSize;
    InPlanes := 2 * Config.InChannel;
    TwoFrames := TNNetVolume.Create(S, S, InPlanes);
    Frame0 := TNNetVolume.Create(S, S, Config.InChannel);
    Frame1 := TNNetVolume.Create(S, S, Config.InChannel);
    try
      // frame0: blob near the left; frame1: same blob shifted right + down.
      // The true middle frame would have the blob halfway between.
      FillBlob(TwoFrames, 0,                S * 0.30, S * 0.40, S * 0.55);
      FillBlob(TwoFrames, Config.InChannel, S * 0.70, S * 0.60, S * 0.55);
      // Mirror into the single-frame volumes for visualisation/preview.
      for Y := 0 to S - 1 do
        for X := 0 to S - 1 do
          for C := 0 to Config.InChannel - 1 do
          begin
            Frame0[X, Y, C] := TwoFrames[X, Y, C];
            Frame1[X, Y, C] := TwoFrames[X, Y, Config.InChannel + C];
          end;

      NN.Compute(TwoFrames);
      WriteLn('Interpolated middle frame: ', NN.GetLastLayer().Output.SizeX,
        'x', NN.GetLastLayer().Output.SizeY, 'x',
        NN.GetLastLayer().Output.Depth, ' (one frame between the two inputs).');

      WriteColorPPM(Frame0, cFrame0File);
      WriteColorPPM(Frame1, cFrame1File);
      WriteColorPPM(NN.GetLastLayer().Output, cMiddleFile);
      WriteLn('Wrote ', cFrame0File, ', ', cMiddleFile, ' and ',
        cFrame1File, '.');

      WriteLn; WriteLn('frame0:'); AsciiPreview(TwoFrames, 0);
      WriteLn; WriteLn('middle (interpolated):');
      AsciiPreview(NN.GetLastLayer().Output, 0);
      WriteLn; WriteLn('frame1:'); AsciiPreview(TwoFrames, Config.InChannel);
    finally
      TwoFrames.Free;
      Frame0.Free;
      Frame1.Free;
    end;
  finally
    NN.Free;
  end;
end.
