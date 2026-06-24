program ImageRestoration;
(*
ImageRestoration: denoises a small image on the CPU end-to-end with the repo's
NAFNet image-restoration importer (BuildNAFNetFromSafeTensors,
neuralpretrained.pas) -- a non-diffusion image-to-image RESTORATION
import that is NOT super-resolution (the RRDBNet/ESRGAN path is x4
upscaling only). NAFNet (Chen et al. 2022, "Simple Baselines for Image
Restoration", arXiv:2204.04676) is a symmetric U-Net of NAFBlocks; it restores
at the SAME resolution.

The new primitive the importer needs is the parameter-free SimpleGate
(TNNetSimpleGate): split the channel axis in half and multiply the two halves --
a GLU with no activation. Simplified Channel Attention (SCA) reuses landed
global pooling + a 1x1 conv + a channelwise multiply. Everything else (1x1/3x3
conv, depthwise conv, per-pixel LayerNorm, pixel-shuffle up via
TNNetDepthToSpace, stride-2 down conv, residual adds) reuses landed layers.

NO NETWORK ACCESS / SELF-CONTAINED: the official NAFNet checkpoints are large
and not obtainable offline, so -- exactly like the repo's RRDBNet / VAE-decoder
pico fixtures -- this falls back to the committed CONFIG-FAITHFUL random pico
NAFNet (tests/fixtures/tiny_nafnet.*, built by tools/nafnet_tiny_fixture.py and
parity-checked < 1e-4 against a float64 numpy oracle in
tests/TestNeuralPretrained.pas). The pico net is random (not trained to
denoise), so this is a wiring/throughput SMOKE demo: it builds the net, adds
synthetic Gaussian noise to a deterministic test image, runs the restoration
forward pass, and reports the round-trip. Pass a real .safetensors (+ config.json
sibling) to restore with your own trained checkpoint.

USAGE
  ./ImageRestoration                 use the committed pico fixture
  ./ImageRestoration model.safetensors [config.json]
                                     use a real checkpoint

OUTPUT
  Writes nafnet_noisy.ppm and nafnet_restored.ppm (P6 color images) and prints
  the config plus a small ASCII intensity preview of input / noisy / restored.

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
  cFixtureST  = '../../tests/fixtures/tiny_nafnet.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_nafnet_config.json';
  cNoisyFile  = 'nafnet_noisy.ppm';
  cRestoredFile = 'nafnet_restored.ppm';

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

procedure AsciiPreview(Img: TNNetVolume);
const cRamp = ' .:-=+*#%@';
var X, Y, idx: integer; v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
  begin
    for X := 0 to Img.SizeX - 1 do
    begin
      v := Img[X, Y, 0];
      if Img.Depth > 2 then v := (Img[X, Y, 0] + Img[X, Y, 1] + Img[X, Y, 2]) / 3;
      v := (v + 1) * 0.5; if v < 0 then v := 0; if v > 1 then v := 1;
      idx := Round(v * (Length(cRamp) - 1)) + 1;
      Write(cRamp[idx]);
    end;
    WriteLn;
  end;
end;

// Deterministic "clean" test image: a smooth radial/striped pattern in [-1,1].
procedure FillCleanImage(Img: TNNetVolume);
var X, Y, C: integer; v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
    for X := 0 to Img.SizeX - 1 do
      for C := 0 to Img.Depth - 1 do
      begin
        v := Sin((X + C * 2) * 0.6) * 0.5 + Cos(Y * 0.5) * 0.4;
        if v < -1 then v := -1; if v > 1 then v := 1;
        Img[X, Y, C] := v;
      end;
end;

// Reproducible synthetic Gaussian-ish noise (deterministic, no RNG state).
procedure AddNoise(Src, Dst: TNNetVolume; Sigma: TNeuralFloat);
var i: integer; n: TNeuralFloat;
begin
  for i := 0 to Src.Size - 1 do
  begin
    // cheap deterministic pseudo-noise from a hashed index.
    n := (Sin(i * 12.9898) * 43758.5453);
    n := (n - Floor(n)) * 2 - 1;   // in (-1,1)
    Dst.FData[i] := Src.FData[i] + n * Sigma;
  end;
end;

function RMSE(A, B: TNNetVolume): TNeuralFloat;
var i: integer; s, d: TNeuralFloat;
begin
  s := 0;
  for i := 0 to A.Size - 1 do
  begin
    d := A.FData[i] - B.FData[i];
    s := s + d * d;
  end;
  Result := Sqrt(s / A.Size);
end;

var
  NN: TNNet;
  Config: TNAFNetConfig;
  Clean, Noisy: TNNetVolume;
  STPath, CfgPath: string;
begin
  if ParamCount >= 1 then
  begin
    STPath := ParamStr(1);
    if ParamCount >= 2 then CfgPath := ParamStr(2)
    else CfgPath := ExtractFilePath(STPath) + 'config.json';
    WriteLn('Loading NAFNet from checkpoint: ', STPath);
  end
  else
  begin
    STPath := cFixtureST;
    CfgPath := cFixtureCfg;
    WriteLn('No checkpoint passed; using the committed pico fixture ' +
      '(random weights -- wiring SMOKE demo).');
  end;

  NN := BuildNAFNetFromSafeTensors(STPath, Config,
    {pTrainable=}false, CfgPath);
  try
    WriteLn(NAFNetConfigToString(Config));
    Clean := TNNetVolume.Create(Config.InputSize, Config.InputSize,
      Config.ImgChannel);
    Noisy := TNNetVolume.Create(Config.InputSize, Config.InputSize,
      Config.ImgChannel);
    try
      FillCleanImage(Clean);
      AddNoise(Clean, Noisy, 0.25);
      WriteLn('Clean vs noisy RMSE: ', FormatFloat('0.0000', RMSE(Clean, Noisy)));

      NN.Compute(Noisy);
      WriteLn('Restored image: ', NN.GetLastLayer().Output.SizeX, 'x',
        NN.GetLastLayer().Output.SizeY, 'x',
        NN.GetLastLayer().Output.Depth, ' (same size as input -- restoration).');

      WriteColorPPM(Noisy, cNoisyFile);
      WriteColorPPM(NN.GetLastLayer().Output, cRestoredFile);
      WriteLn('Wrote ', cNoisyFile, ' and ', cRestoredFile, '.');

      WriteLn; WriteLn('clean:');    AsciiPreview(Clean);
      WriteLn; WriteLn('noisy:');    AsciiPreview(Noisy);
      WriteLn; WriteLn('restored:'); AsciiPreview(NN.GetLastLayer().Output);
    finally
      Clean.Free;
      Noisy.Free;
    end;
  finally
    NN.Free;
  end;
end.
