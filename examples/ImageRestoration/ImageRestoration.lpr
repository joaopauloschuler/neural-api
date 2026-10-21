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
sibling) to restore with your own trained checkpoint, and --input FILE to
restore a real image (PNG, JPEG, BMP, PPM, ...) instead of the synthetic one.
Pixels enter the net in [0, 1] RGB, as the official NAFNet.

USAGE
  ./ImageRestoration [model.safetensors [config.json]]
                     [--input FILE] [--output FILE]
  No checkpoint: the committed pico fixture. --input FILE: restore this image
  (no synthetic noise); the net is built at its size, padded on the right and
  bottom to a multiple of 2^levels, and the output is cropped back.

OUTPUT
  The restored image goes to --output (format from the extension; default
  nafnet_restored.ppm). The synthetic run also writes nafnet_noisy.ppm and
  prints the config plus a small ASCII intensity preview of clean / noisy /
  restored.

Coded by Claude (AI).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)
{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  SysUtils, Classes, Math,
  neuralnetwork, neuralvolume, neuralpretrained, neuraldatasets;

const
  cFixtureST  = '../../tests/fixtures/tiny_nafnet.safetensors';
  cFixtureCfg = '../../tests/fixtures/tiny_nafnet_config.json';
  cNoisyFile  = 'nafnet_noisy.ppm';
  cRestoredFile = 'nafnet_restored.ppm';
  cMaxPixelValue = 255;
  // Largest image side previewed as ASCII art.
  cMaxPreviewSide = 64;

// Writes a [0, 1] RGB image to FileName, in the format of its extension.
procedure SaveUnitRangeImage(Img: TNNetVolume; const FileName: string);
var
  ByteImage: TNNetVolume;
begin
  ByteImage := TNNetVolume.Create();
  try
    ByteImage.Copy(Img);
    ByteImage.Mul(cMaxPixelValue);
    if not SaveImageFromVolumeIntoFile(ByteImage, FileName) then
      raise Exception.Create('Could not write ' + FileName + '.');
  finally
    ByteImage.Free;
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
      if v < 0 then v := 0; if v > 1 then v := 1;
      idx := Round(v * (Length(cRamp) - 1)) + 1;
      Write(cRamp[idx]);
    end;
    WriteLn;
  end;
end;

// Deterministic "clean" test image: a smooth radial/striped pattern in [0,1].
procedure FillCleanImage(Img: TNNetVolume);
var X, Y, C: integer; v: TNeuralFloat;
begin
  for Y := 0 to Img.SizeY - 1 do
    for X := 0 to Img.SizeX - 1 do
      for C := 0 to Img.Depth - 1 do
      begin
        v := Sin((X + C * 2) * 0.6) * 0.25 + Cos(Y * 0.5) * 0.2 + 0.5;
        if v < 0 then v := 0; if v > 1 then v := 1;
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
begin
  // SIMD bulk sum of squared differences.
  Result := Sqrt(A.GetDistanceSqr(B) / A.Size);
end;

// Side rounded up to a multiple of Multiple.
function RoundUpToMultiple(Side, Multiple: integer): integer;
begin
  Result := ((Side + Multiple - 1) div Multiple) * Multiple;
end;

procedure PrintUsageAndHalt(const Problem: string);
begin
  WriteLn(Problem);
  WriteLn('Usage: ImageRestoration [model.safetensors [config.json]] ',
    '[--input FILE] [--output FILE]');
  Halt(2);
end;

var
  NN: TNNet;
  Config: TNAFNetConfig;
  Clean, Degraded, Padded, Restored: TNNetVolume;
  STPath, CfgPath, InputFile, OutputFile, Arg: string;
  ArgPos, SizeMultiple: integer;
begin
  STPath := '';
  CfgPath := '';
  InputFile := '';
  OutputFile := cRestoredFile;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if (Arg = '--input') or (Arg = '--output') then
    begin
      if ArgPos = ParamCount then PrintUsageAndHalt(Arg + ' needs a file name.');
      Inc(ArgPos);
      if Arg = '--input' then InputFile := ParamStr(ArgPos)
      else OutputFile := ParamStr(ArgPos);
    end
    else if Copy(Arg, 1, 1) = '-' then
      PrintUsageAndHalt('Unknown option ' + Arg + '.')
    else if STPath = '' then STPath := Arg
    else if CfgPath = '' then CfgPath := Arg
    else PrintUsageAndHalt('Unexpected argument ' + Arg + '.');
    Inc(ArgPos);
  end;
  if STPath <> '' then
  begin
    if CfgPath = '' then CfgPath := ExtractFilePath(STPath) + 'config.json';
    WriteLn('Loading NAFNet from checkpoint: ', STPath);
  end
  else
  begin
    STPath := cFixtureST;
    CfgPath := cFixtureCfg;
    WriteLn('No checkpoint passed; using the committed pico fixture ' +
      '(random weights -- wiring SMOKE demo).');
  end;

  Config := ReadNAFNetConfigFromJSONFile(CfgPath);
  if Config.ImgChannel <> 3 then
    PrintUsageAndHalt('This example reads and writes RGB images; the config ' +
      'has img_channel=' + IntToStr(Config.ImgChannel) + '.');
  Clean := TNNetVolume.Create();
  Degraded := TNNetVolume.Create();
  Padded := TNNetVolume.Create();
  Restored := TNNetVolume.Create();
  NN := nil;
  try
    if InputFile <> '' then
    begin
      try
        if not LoadImageFromFileIntoVolume(InputFile, Degraded) then
          PrintUsageAndHalt('Could not read ' + InputFile + '.');
      except
        on E: Exception do
          PrintUsageAndHalt('Could not read ' + InputFile + ': ' + E.Message);
      end;
      Degraded.Mul(1 / cMaxPixelValue);
      // The stride-2 down convs need sides divisible by 2^levels; the official
      // NAFNet zero-pads on the right and bottom the same way.
      SizeMultiple := 1 shl Length(Config.EncBlkNums);
      Config.InputSize := RoundUpToMultiple(Degraded.SizeX, SizeMultiple);
      Config.InputSizeY := RoundUpToMultiple(Degraded.SizeY, SizeMultiple);
      WriteLn('Input image: ', InputFile, ' (', Degraded.SizeX, 'x',
        Degraded.SizeY, '), net input ', Config.InputSize, 'x',
        Config.InputSizeY);
    end
    else
    begin
      Clean.ReSize(Config.InputSize, Config.InputSize, Config.ImgChannel);
      Degraded.ReSize(Config.InputSize, Config.InputSize, Config.ImgChannel);
      FillCleanImage(Clean);
      AddNoise(Clean, Degraded, 0.125);
      WriteLn('Clean vs noisy RMSE: ', FormatFloat('0.0000', RMSE(Clean, Degraded)));
    end;
    NN := BuildNAFNetFromSafeTensorsEx(STPath, Config, {pTrainable=}false);
    WriteLn(NAFNetConfigToString(Config));

    Padded.ReSize(NN.Layers[0].Output.SizeX, NN.Layers[0].Output.SizeY,
      Config.ImgChannel);
    Padded.Fill(0);
    Padded.AddArea(0, 0, 0, 0, Degraded.SizeX, Degraded.SizeY, Degraded);
    NN.Compute(Padded);
    Restored.CopyCropping(NN.GetLastLayer().Output, 0, 0, Degraded.SizeX,
      Degraded.SizeY);
    WriteLn('Restored image: ', Restored.SizeX, 'x', Restored.SizeY, 'x',
      Restored.Depth, ' (same size as input -- restoration).');
    WriteLn('Input vs restored RMSE: ',
      FormatFloat('0.0000', RMSE(Degraded, Restored)));

    SaveUnitRangeImage(Restored, OutputFile);
    if InputFile = '' then
    begin
      SaveUnitRangeImage(Degraded, cNoisyFile);
      WriteLn('Wrote ', cNoisyFile, ' and ', OutputFile, '.');
    end
    else WriteLn('Wrote ', OutputFile, '.');

    if Max(Restored.SizeX, Restored.SizeY) <= cMaxPreviewSide then
    begin
      if InputFile = '' then
      begin
        WriteLn; WriteLn('clean:'); AsciiPreview(Clean);
      end;
      WriteLn; WriteLn('input:');    AsciiPreview(Degraded);
      WriteLn; WriteLn('restored:'); AsciiPreview(Restored);
    end;
  finally
    NN.Free;
    Restored.Free;
    Padded.Free;
    Degraded.Free;
    Clean.Free;
  end;
end.
