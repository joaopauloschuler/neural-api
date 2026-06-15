program StyleGAN2Generate;
(*
StyleGAN2Generate: synthesizes ONE image on the CPU from a fixed latent with the
repo's StyleGAN2 generator importer (BuildStyleGAN2Generator,
neuralpretrained.pas) -- a style-based generative import
(Karras et al. 2020, "Analyzing and Improving the Image Quality of StyleGAN",
arXiv:1912.04958).

The synthesis path the importer builds and this example runs:
  - an 8-layer-style mapping MLP turns the latent z into a w latent;
  - the synthesis network grows the image from a LEARNED CONSTANT through a tower
    of resolution blocks, each = nearest-x2 upsample (after the first) ->
    [modulated/demodulated conv (the new TNNetModulatedConv2D primitive) +
    per-pixel noise injection (scaled by a learned strength) + LeakyReLU(0.2)] ->
    a toRGB modulated 1x1 conv (NO demod) summed into an upsampled RGB skip.
  Each modulated conv reads a per-input-channel style vector from a small affine
  layer "A" applied to w. This is INFERENCE-ONLY synthesis (no discriminator /
  path-length reg / training -- v1).

NO NETWORK ACCESS / SELF-CONTAINED: the official StyleGAN2 weights are not
redistributable / not obtainable offline, so -- exactly like the repo's RRDBNet
/ VAE-decoder pico fixtures -- this falls back to the committed CONFIG-FAITHFUL
random pico generator (tests/fixtures/tiny_stylegan2.*, built by
tools/make_pico_stylegan2_fixture.py and parity-checked < 1e-4 against a
float64 numpy oracle in tests/TestNeuralPretrained.pas). Pass a real
.safetensors (+ config.json sibling) to synthesize from your own checkpoint.

USAGE
  ./StyleGAN2Generate                 use the committed pico fixture
  ./StyleGAN2Generate model.safetensors [config.json]
                                      use a real checkpoint

OUTPUT
  Writes stylegan2_sample.ppm (a P6 color image) from a fixed latent and prints
  the config + a small ASCII intensity preview.

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Classes,
  neuralnetwork, neuralvolume, neuralpretrained;

const
  cFixtureST  = '../../tests/fixtures/tiny_stylegan2.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_stylegan2_config.json';
  cOutFile    = 'stylegan2_sample.ppm';

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
      // luminance-ish average across channels in [-1,1] -> ramp.
      v := Img[X, Y, 0];
      if Img.Depth > 2 then v := (Img[X, Y, 0] + Img[X, Y, 1] + Img[X, Y, 2]) / 3;
      v := (v + 1) * 0.5; if v < 0 then v := 0; if v > 1 then v := 1;
      idx := Round(v * (Length(cRamp) - 1)) + 1;
      Write(cRamp[idx]);
    end;
    WriteLn;
  end;
end;

var
  NN: TNNet;
  Config: TStyleGAN2Config;
  Z: TNNetVolume;
  STPath, CfgPath: string;
  i: integer;
begin
  if ParamCount >= 1 then
  begin
    STPath := ParamStr(1);
    if ParamCount >= 2 then CfgPath := ParamStr(2)
    else CfgPath := ExtractFilePath(STPath) + 'config.json';
    WriteLn('Loading StyleGAN2 generator from checkpoint: ', STPath);
  end
  else
  begin
    STPath := cFixtureST;
    CfgPath := cFixtureCfg;
    WriteLn('No checkpoint passed; using the committed pico fixture.');
  end;

  NN := BuildStyleGAN2GeneratorFromSafeTensors(STPath, Config,
    {pInferenceOnly=}true, CfgPath);
  try
    WriteLn(StyleGAN2ConfigToString(Config));
    Z := TNNetVolume.Create(Config.LatentDim, 1, 1);
    try
      // A FIXED, reproducible latent (no RNG): a smooth deterministic ramp.
      for i := 0 to Z.Size - 1 do
        Z.FData[i] := Sin(i * 0.7 + 0.3) * 0.9;
      NN.Compute(Z);
      WriteLn('Generated image: ', NN.GetLastLayer().Output.SizeX, 'x',
        NN.GetLastLayer().Output.SizeY, 'x',
        NN.GetLastLayer().Output.Depth);
      WriteColorPPM(NN.GetLastLayer().Output, cOutFile);
      WriteLn('Wrote ', cOutFile, '.');
      WriteLn('ASCII intensity preview:');
      AsciiPreview(NN.GetLastLayer().Output);
    finally
      Z.Free;
    end;
  finally
    NN.Free;
  end;
end.
