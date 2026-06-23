program SwinIRRestore;
(*
SwinIRRestore: super-resolves (upscales) a small image on the CPU end-to-end
with the repo's SwinIR image-restoration importer (BuildSwinIRFromSafeTensors,
neuralpretrained.pas) -- a TRANSFORMER restoration import.
SwinIR (Liang et al. 2021, "SwinIR: Image Restoration Using Swin Transformer",
arXiv:2108.10257) stacks Residual Swin Transformer Blocks (RSTB = a few Swin
window/shifted-window attention layers + a 3x3 conv + a residual over the block)
on a shallow conv stem, then a pixel-shuffle upsample tail for classical
super-resolution.

Architecturally distinct from the CNN-only RRDBNet/ESRGAN SR path and the
SimpleGate-CNN NAFNet denoiser: this is transformer restoration. The window /
shifted-window attention REUSES the landed Swin building blocks
(TNNetWindowAttention + relative_position_bias + cyclic-shift mask,
TNNetGatherTokens partition/reverse); the new pieces are the conv stem, the RSTB
residual conv, and the TNNetDepthToSpace pixel-shuffle upsample.

NO NETWORK ACCESS / SELF-CONTAINED: the official SwinIR checkpoints are large and
not obtainable offline, so -- exactly like the repo's RRDBNet / NAFNet pico
fixtures -- this falls back to the committed CONFIG-FAITHFUL random pico SwinIR
(tests/fixtures/tiny_swinir.*, built by tools/swinir_tiny_fixture.py and
parity-checked < 1e-4 against a float64 numpy oracle in
tests/TestNeuralPretrained.pas). The pico net is random (not trained), so this
is a wiring/throughput SMOKE demo: it builds the net, runs the SR forward pass
on a deterministic synthetic image, and reports the upscaled shape. Pass a real
.safetensors (+ config.json sibling) to upscale with your own trained checkpoint.

USAGE
  ./SwinIRRestore                 use the committed pico fixture
  ./SwinIRRestore model.safetensors [config.json]
                                  use a real checkpoint

OUTPUT
  Writes swinir_input.ppm and swinir_upscaled.ppm (P6 color images) and prints
  the config plus a small ASCII intensity preview of input / upscaled.

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
  cFixtureST  = '../../tests/fixtures/tiny_swinir.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_swinir_config.json';
  cInputFile  = 'swinir_input.ppm';
  cOutFile    = 'swinir_upscaled.ppm';

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
      v := Img[X, Y, 0];
      if Img.Depth > 2 then v := (Img[X, Y, 0] + Img[X, Y, 1] + Img[X, Y, 2]) / 3;
      v := (v + 1) * 0.5; if v < 0 then v := 0; if v > 1 then v := 1;
      idx := Round(v * (Length(cRamp) - 1)) + 1;
      Write(cRamp[idx]);
    end;
    WriteLn;
  end;
end;

// Deterministic low-res test image: a smooth striped pattern in [-1,1].
procedure FillImage(Img: TNNetVolume);
var X, Y, C: integer; v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
    for X := 0 to Img.SizeX - 1 do
      for C := 0 to Img.Depth - 1 do
      begin
        v := Sin((X + C * 2) * 0.7) * 0.5 + Cos(Y * 0.6) * 0.4;
        if v < -1 then v := -1; if v > 1 then v := 1;
        Img[X, Y, C] := v;
      end;
end;

var
  NN: TNNet;
  Config: TSwinIRConfig;
  LowRes: TNNetVolume;
  STPath, CfgPath: string;
begin
  if ParamCount >= 1 then
  begin
    STPath := ParamStr(1);
    if ParamCount >= 2 then CfgPath := ParamStr(2)
    else CfgPath := ExtractFilePath(STPath) + 'config.json';
    WriteLn('Loading SwinIR from checkpoint: ', STPath);
  end
  else
  begin
    STPath := cFixtureST;
    CfgPath := cFixtureCfg;
    WriteLn('No checkpoint passed; using the committed pico fixture ' +
      '(random weights -- wiring SMOKE demo).');
  end;

  NN := BuildSwinIRFromSafeTensors(STPath, Config,
    {pTrainable=}false, CfgPath);
  try
    WriteLn(SwinIRConfigToString(Config));
    LowRes := TNNetVolume.Create(Config.ImgSize, Config.ImgSize, Config.InChans);
    try
      FillImage(LowRes);
      WriteLn('Input: ', LowRes.SizeX, 'x', LowRes.SizeY, 'x', LowRes.Depth);

      NN.Compute(LowRes);
      WriteLn('Upscaled image: ', NN.GetLastLayer().Output.SizeX, 'x',
        NN.GetLastLayer().Output.SizeY, 'x',
        NN.GetLastLayer().Output.Depth, ' (',
        Config.Upscale, 'x super-resolution).');

      WriteColorPPM(LowRes, cInputFile);
      WriteColorPPM(NN.GetLastLayer().Output, cOutFile);
      WriteLn('Wrote ', cInputFile, ' and ', cOutFile, '.');

      WriteLn; WriteLn('input (low-res):'); AsciiPreview(LowRes);
      WriteLn; WriteLn('upscaled:');        AsciiPreview(NN.GetLastLayer().Output);
    finally
      LowRes.Free;
    end;
  finally
    NN.Free;
  end;
end.
