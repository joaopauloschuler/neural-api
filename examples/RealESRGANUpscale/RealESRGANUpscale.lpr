program RealESRGANUpscale;
(*
RealESRGANUpscale: an END-TO-END smoke of the Real-ESRGAN / ESRGAN RRDBNet
super-resolution importer (BuildRRDBNetFromSafeTensors, neuralpretrained.pas).
It is the convolutional, NO-diffusion image upscaler in the repo: load the
committed tiny RRDBNet checkpoint, read a small PNG, run the net forward and
write the upscaled PNG.

Pipeline (pure CPU, no dataset download, well under a minute):
  1. Build the pico RRDBNet from the committed parity fixture
     tests/fixtures/tiny_rrdbnet{.safetensors,_config.json} (scale x4) and the
     x2 sibling tiny_rrdbnet_x2.* (scale x2 = one upsample stage).
  2. Synthesize a tiny 6x6 RGB image (a deterministic colour gradient) and
     SAVE it as a PNG with neuraldatasets.SaveImageFromVolumeIntoFile - so the
     example exercises the repo's real PNG writer.
  3. LOAD that PNG back with LoadImageFromFileIntoVolume, normalise 0..255 ->
     [-1,1] (the value range the RRDBNet weights expect), Compute, then map the
     output back to 0..255 and SAVE the upscaled PNG.
  4. Report the input/output dimensions for both scales.

The committed checkpoint is a random-weight pico net (a faithful smoke of the
import + forward + image-I/O path, not a photoreal upscaler); swap in a real
RealESRGAN_x4plus .pth / .safetensors + its config.json to upscale real images
(the same BuildRRDBNetFromSafeTensors call - .pth params_ema is unwrapped
automatically by TNNetTorchBinReader).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained;

const
  // Fixtures committed by tools/rrdbnet_tiny_fixture.py.
  FixturesDir = '../../tests/fixtures/';

// Builds a tiny synthetic 6x6 RGB image (0..255) as a deterministic colour
// gradient so the example needs no input asset on disk.
procedure MakeTinyImage(V: TNNetVolume; Size: integer);
var
  X, Y: integer;
begin
  V.ReSize(Size, Size, 3);
  for X := 0 to Size - 1 do
    for Y := 0 to Size - 1 do
    begin
      V[X, Y, 0] := (X * 255) div (Size - 1);                 // R ramps in X
      V[X, Y, 1] := (Y * 255) div (Size - 1);                 // G ramps in Y
      V[X, Y, 2] := ((X + Y) * 255) div (2 * (Size - 1));     // B diagonal
    end;
end;

// Loads PngIn, upscales it through the RRDBNet at Config.Scale and writes
// PngOut. Normalises 0..255 -> [-1,1] before the net and maps the raw output
// back to 0..255 (a tanh-free clamp; the pico weights are random so this is a
// smoke, not a tuned tone curve).
procedure UpscalePng(NN: TNNet; const Config: TRRDBNetConfig;
  const PngIn, PngOut: string);
var
  InVol, NormVol, OutVol: TNNetVolume;
  i: integer;
begin
  InVol := TNNetVolume.Create;
  NormVol := TNNetVolume.Create;
  OutVol := TNNetVolume.Create;
  try
    if not LoadImageFromFileIntoVolume(PngIn, InVol) then
      raise Exception.Create('could not load ' + PngIn);
    // 0..255 -> [-1,1].
    NormVol.Copy(InVol);
    NormVol.Divi(127.5);
    NormVol.Add(-1.0);
    NN.Compute(NormVol);
    // [-1,1] -> 0..255 (clamped by the PNG writer).
    OutVol.Copy(NN.GetLastLayer().Output);
    OutVol.Add(1.0);
    OutVol.Mul(127.5);
    for i := 0 to OutVol.Size - 1 do
      OutVol.FData[i] := EnsureRange(OutVol.FData[i], 0, 255);
    if not SaveImageFromVolumeIntoFile(OutVol, PngOut) then
      raise Exception.Create('could not save ' + PngOut);
    WriteLn(Format('  scale x%d: %dx%d -> %dx%d  (%s)',
      [Config.Scale, InVol.SizeX, InVol.SizeY,
       OutVol.SizeX, OutVol.SizeY, PngOut]));
  finally
    OutVol.Free;
    NormVol.Free;
    InVol.Free;
  end;
end;

procedure RunScale(const Stem, InPng, OutPng: string);
var
  NN: TNNet;
  Config: TRRDBNetConfig;
  Img: TNNetVolume;
begin
  NN := BuildRRDBNetFromSafeTensors(
    FixturesDir + Stem + '.safetensors', Config,
    {pTrainable=}false, FixturesDir + Stem + '_config.json');
  Img := TNNetVolume.Create;
  try
    WriteLn(RRDBNetConfigToString(Config));
    MakeTinyImage(Img, Config.InputSize);
    if not SaveImageFromVolumeIntoFile(Img, InPng) then
      raise Exception.Create('could not save ' + InPng);
    UpscalePng(NN, Config, InPng, OutPng);
  finally
    Img.Free;
    NN.Free;
  end;
end;

begin
  WriteLn('Real-ESRGAN / ESRGAN RRDBNet upscale smoke');
  WriteLn('==========================================');
  RunScale('tiny_rrdbnet',    'esrgan_in_x4.png', 'esrgan_out_x4.png');
  RunScale('tiny_rrdbnet_x2', 'esrgan_in_x2.png', 'esrgan_out_x2.png');
  WriteLn('Done.');
end.
