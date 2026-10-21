program VideoFrameInterpolation;
(*
VideoFrameInterpolation: synthesises intermediate frames between two input
frames on the CPU end-to-end with the repo's RIFE frame-interpolation importer
(BuildRIFEFromSafeTensors, neuralpretrained.pas) -- a VIDEO-generative import.
The landed RAFT path estimates optical FLOW
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

TIMESTEPS: a checkpoint whose first IFBlock reads 2*in_channel+1 planes takes
Practical-RIFE v4's timestep plane, so it renders any t directly. A checkpoint
without one only predicts t=0.5; RIFEInterpolateFrames then recurses (middle
frame, then the middles of each half), so --factor must be a power of two and
--t must be dyadic (k/2^n). The net is built once for every frame.

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
  ./VideoFrameInterpolation [model.safetensors [config.json]] [options]
  With no checkpoint the committed pico fixture is used.
  --factor N     write N-1 frames at t = k/N between the inputs (default 2)
  --t VALUE      write ONE frame at time VALUE in (0,1) instead of --factor
  --gif FILE     also write all frames (inputs included) as a looping GIF

OUTPUT
  Writes rife_frame_00.ppm (frame0), the intermediate frames in time order, and
  frame1 last (P6 color images), and prints the config plus a small ASCII
  preview of every frame for grids up to 32x32.

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
  neuralnetwork, neuralvolume, neuralpretrained, neuralgif;

const
  cFixtureST  = '../../tests/fixtures/tiny_rife.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_rife_config.json';
  cFramePrefix = 'rife_frame_';
  cMaxPreviewSize = 32;

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

procedure Usage(const Problem: string);
begin
  WriteLn('Error: ', Problem);
  WriteLn('Usage: VideoFrameInterpolation [model.safetensors [config.json]] ' +
    '[--factor N] [--t VALUE] [--gif FILE]');
  Halt(1);
end;

var
  NN: TNNet;
  Config: TRIFEConfig;
  Frame0, Frame1: TNNetVolume;
  Frames, Sequence: TNNetVolumeList;
  GifWriter: TNNetAnimatedGIFWriter;
  S, ArgPos, FramePos, MaxFramePos, Factor: integer;
  FrameT: TNeuralFloat;
  HasFrameT: boolean;
  STPath, CfgPath, GifFile, FrameFile, Arg: string;
begin
  STPath := '';
  CfgPath := '';
  GifFile := '';
  Factor := 2;
  FrameT := 0.5;
  HasFrameT := false;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if (Arg = '--factor') or (Arg = '--t') or (Arg = '--gif') then
    begin
      if ArgPos = ParamCount then Usage(Arg + ' needs a value.');
      Inc(ArgPos);
      if Arg = '--factor' then
      begin
        if not TryStrToInt(ParamStr(ArgPos), Factor) or (Factor < 2) then
          Usage('--factor must be an integer >= 2.');
      end
      else if Arg = '--t' then
      begin
        if not TryStrToFloat(ParamStr(ArgPos), FrameT) or (FrameT <= 0) or
          (FrameT >= 1) then
          Usage('--t must be a number in (0, 1).');
        HasFrameT := true;
      end
      else GifFile := ParamStr(ArgPos);
    end
    else if Copy(Arg, 1, 2) = '--' then Usage('unknown option ' + Arg)
    else if STPath = '' then STPath := Arg
    else if CfgPath = '' then CfgPath := Arg
    else Usage('unexpected argument ' + Arg);
    Inc(ArgPos);
  end;
  if STPath <> '' then
  begin
    if CfgPath = '' then CfgPath := ExtractFilePath(STPath) + 'config.json';
    WriteLn('Loading RIFE from checkpoint: ', STPath);
  end
  else
  begin
    STPath := cFixtureST;
    CfgPath := cFixtureCfg;
    WriteLn('No checkpoint passed; using the committed pico fixture ' +
      '(random weights -- wiring SMOKE demo).');
  end;

  NN := BuildRIFEFromSafeTensors(STPath, Config, {pTrainable=}false, CfgPath);
  Frame0 := nil;
  Frame1 := nil;
  Frames := TNNetVolumeList.Create;
  Sequence := TNNetVolumeList.Create({FreeObjects=}false);
  try
    WriteLn(RIFEConfigToString(Config));
    S := Config.InputSize;
    Frame0 := TNNetVolume.Create(S, S, Config.InChannel);
    Frame1 := TNNetVolume.Create(S, S, Config.InChannel);
    // frame0: blob near the left; frame1: same blob shifted right + down.
    FillBlob(Frame0, 0, S * 0.30, S * 0.40, S * 0.55);
    FillBlob(Frame1, 0, S * 0.70, S * 0.60, S * 0.55);

    try
      if HasFrameT then
      begin
        Frames.AddVolumes(1, S, S, Config.InChannel);
        RIFEInterpolateAt(NN, Config, Frame0, Frame1, FrameT, Frames[0]);
        WriteLn('Interpolated one frame at t=', FrameT:0:4, '.');
      end
      else
      begin
        RIFEInterpolateFrames(NN, Config, Frame0, Frame1, Factor, Frames);
        WriteLn('Interpolated ', Frames.Count, ' frame(s) for a ', Factor,
          'x factor (', BoolToStr(Config.TimestepInput,
          'one call per t', 'midpoint recursion'), ').');
      end;
    except
      // A model without a timestep plane rejects non-dyadic t / factors.
      on E: Exception do Usage(E.Message);
    end;

    Sequence.Add(Frame0);
    for FramePos := 0 to Frames.Count - 1 do Sequence.Add(Frames[FramePos]);
    Sequence.Add(Frame1);
    MaxFramePos := Sequence.Count - 1;
    for FramePos := 0 to MaxFramePos do
    begin
      FrameFile := cFramePrefix + Format('%.2d', [FramePos]) + '.ppm';
      WriteColorPPM(Sequence[FramePos], FrameFile);
      WriteLn('Wrote ', FrameFile);
      if S <= cMaxPreviewSize then
      begin
        AsciiPreview(Sequence[FramePos], 0);
        WriteLn;
      end;
    end;
    if GifFile <> '' then
    begin
      GifWriter := TNNetAnimatedGIFWriter.Create(GifFile, S, S);
      try
        for FramePos := 0 to MaxFramePos do
          GifWriter.AddFrame(Sequence[FramePos], -1, 1);
        GifWriter.Finish;
        WriteLn('Wrote ', GifWriter.FrameCount, '-frame animated GIF ',
          GifFile);
      finally
        GifWriter.Free;
      end;
    end;
  finally
    Sequence.Free;
    Frames.Free;
    Frame1.Free;
    Frame0.Free;
    NN.Free;
  end;
end.
