program BackgroundRemoval;
(*
BackgroundRemoval: BiRefNet / RMBG-2.0 foreground segmentation (Zheng et al.,
CAAI AIR 2024) with the importer in neuralpretrained.pas
(BuildBiRefNetFromSafeTensors + BiRefNetRemoveBackground).

Two modes:
  * no arguments: a smoke of the committed random pico fixture
    tests/fixtures/tiny_birefnet: synthesize a small colour PNG, read it back,
    cut it out, write the RGBA and mask PNGs.
  * --model FILE --input IMG --output IMG: remove the background of a real
    image with a real checkpoint (ZhengPeng7/BiRefNet or briaai/RMBG-2.0
    model.safetensors). Every shape comes from the tensors; the net input
    defaults to 1024x1024.

The net predicts a foreground probability at --size x --size; the mask is
resized back to the image's own size and becomes the alpha channel of the
output PNG, so the RGB keeps every detail of the input.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained;

const
  // Fixture committed with the BiRefNet importer tests.
  FixturesDir = '../../tests/fixtures/';

type
  TBackgroundRemovalOptions = record
    ModelFile, ConfigFile, InputFile, OutputFile, MaskFile: string;
    InputSize: integer;
    UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: BackgroundRemoval                       (pico fixture smoke)');
  WriteLn('       BackgroundRemoval --model FILE --input IMG --output IMG [options]');
  WriteLn('  --model FILE         BiRefNet / RMBG-2.0 model.safetensors (or .pth / .bin)');
  WriteLn('  --config FILE        BiRefNet config.json (default: next to --model if');
  WriteLn('                       present; the shapes always come from the tensors)');
  WriteLn('  --input IMG          image (PNG / JPEG / BMP)');
  WriteLn('  --output IMG         cut-out image, PNG with alpha');
  WriteLn('  --mask-output IMG    the alpha mask alone, gray PNG');
  WriteLn('  --size N             net input resolution, a multiple of 32 (default ',
    '1024, the official setting)');
  WriteLn('  --int8               store the conv and linear weights in int8 (less ',
    'RAM, small accuracy loss)');
  WriteLn('  --gpu                run the net on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TBackgroundRemovalOptions): boolean;
var
  ArgPos: integer;
  Arg: string;

  function NextArg(): string;
  begin
    Inc(ArgPos);
    if ArgPos > ParamCount then
      raise Exception.Create(Arg + ' needs a value');
    Result := ParamStr(ArgPos);
  end;

begin
  Opt.ModelFile := '';
  Opt.ConfigFile := '';
  Opt.InputFile := '';
  Opt.OutputFile := '';
  Opt.MaskFile := '';
  Opt.InputSize := 0;
  Opt.UseOpenCL := false;
  Opt.UseInt8 := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then Opt.ModelFile := NextArg()
    else if Arg = '--config' then Opt.ConfigFile := NextArg()
    else if Arg = '--input' then Opt.InputFile := NextArg()
    else if Arg = '--output' then Opt.OutputFile := NextArg()
    else if Arg = '--mask-output' then Opt.MaskFile := NextArg()
    else if Arg = '--size' then Opt.InputSize := StrToInt(NextArg())
    else if Arg = '--int8' then Opt.UseInt8 := true
    else if Arg = '--gpu' then Opt.UseOpenCL := true
    else if Arg = '--gpu-platform' then Opt.OpenCLPlatform := StrToInt(NextArg())
    else if Arg = '--gpu-device' then Opt.OpenCLDevice := StrToInt(NextArg())
    else if (Arg = '--help') or (Arg = '-h') then
    begin
      PrintUsage();
      Result := false;
      exit;
    end
    else
      raise Exception.Create('unknown argument: ' + Arg);
    Inc(ArgPos);
  end;
  if (Opt.ModelFile <> '') and (Opt.InputFile = '') then
    raise Exception.Create('--model needs --input');
  if (Opt.ModelFile <> '') and (Opt.OutputFile = '') and (Opt.MaskFile = '') then
    raise Exception.Create('--model needs --output and/or --mask-output');
  if (Opt.InputSize < 0) or (Opt.InputSize mod 32 <> 0) then
    raise Exception.Create('--size must be a positive multiple of 32');
end;

// Saves the (W, H, 1) 0..255 Mask as a gray RGB PNG.
procedure SaveMaskPNG(Mask: TNNetVolume; const FileName: string);
var
  Gray: TNNetVolume;
  X, Y, Channel, MaxX, MaxY: integer;
begin
  Gray := TNNetVolume.Create(Mask.SizeX, Mask.SizeY, 3);
  try
    MaxX := Mask.SizeX - 1;
    MaxY := Mask.SizeY - 1;
    for Y := 0 to MaxY do
      for X := 0 to MaxX do
        for Channel := 0 to 2 do
          Gray[X, Y, Channel] := Mask[X, Y, 0];
    if not SaveImageFromVolumeIntoFile(Gray, FileName) then
      raise Exception.Create('could not save ' + FileName);
  finally
    Gray.Free;
  end;
end;

// Loads InputFile, removes its background and writes OutputFile / MaskFile.
procedure RemoveBackgroundFromFile(const Opt: TBackgroundRemovalOptions);
var
  NN: TNNet;
  Config: TBiRefNetConfig;
  Image, Mask, RGBA: TNNetVolume;
  ComputeText: string;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
  {$ENDIF}
begin
  Image := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
  RGBA := TNNetVolume.Create;
  NN := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    if not LoadImageFromFileIntoVolume(Opt.InputFile, Image) then
      raise Exception.Create('could not load ' + Opt.InputFile);
    StartTime := Now();
    NN := BuildBiRefNetFromSafeTensors(Opt.ModelFile, Config,
      {pTrainable=}false, Opt.ConfigFile, Opt.InputSize);
    WriteLn(BiRefNetConfigToString(Config));
    WriteLn(Format('  loaded in %.1f s', [(Now() - StartTime) * 86400.0]));
    // Quantize before EnableOpenCL: the OpenCL arming reads the int8 codes.
    if Opt.UseInt8 then NN.QuantizeWeightsInt8();
    ComputeText := 'CPU';
    if Opt.UseOpenCL then
    begin
      {$IFDEF OpenCL}
      OpenCLDevices := TEasyOpenCL.Create();
      OpenCLPlatform := Opt.OpenCLPlatform;
      OpenCLDevice := Opt.OpenCLDevice;
      if OpenCLDevices.SelectPlatformAndDevice(OpenCLPlatform, OpenCLDevice,
        OpenCLProblem) then
      begin
        NN.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
          OpenCLDevices.Devices[OpenCLDevice]);
        ComputeText := 'OpenCL on ' +
          OpenCLDevices.PlatformNames[OpenCLPlatform] + ' / ' +
          OpenCLDevices.DeviceNames[OpenCLDevice];
      end
      else
        ComputeText := 'CPU (' + OpenCLProblem + ')';
      {$ELSE}
      ComputeText := 'CPU (built without -dOpenCL)';
      {$ENDIF}
    end;
    if Opt.UseInt8 then ComputeText := ComputeText + ', int8 weights';
    WriteLn(Format('Segmenting %dx%d on %s, net input %dx%d',
      [Image.SizeX, Image.SizeY, ComputeText, Config.InputWidth,
       Config.InputHeight]));
    StartTime := Now();
    BiRefNetRemoveBackground(NN, Config, Image, Mask);
    WriteLn(Format('  %.1f s', [(Now() - StartTime) * 86400.0]));
    if Opt.OutputFile <> '' then
    begin
      BiRefNetComposeRGBA(Image, Mask, RGBA);
      if not SaveImageFromVolumeIntoFile(RGBA, Opt.OutputFile) then
        raise Exception.Create('could not save ' + Opt.OutputFile);
      WriteLn(Format('  %dx%d RGBA -> %s', [RGBA.SizeX, RGBA.SizeY,
        Opt.OutputFile]));
    end;
    if Opt.MaskFile <> '' then
    begin
      SaveMaskPNG(Mask, Opt.MaskFile);
      WriteLn(Format('  %dx%d mask -> %s', [Mask.SizeX, Mask.SizeY,
        Opt.MaskFile]));
    end;
  finally
    NN.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    RGBA.Free;
    Mask.Free;
    Image.Free;
  end;
end;

// Builds a deterministic RGB colour gradient (0..255) so the smoke needs no
// input asset on disk.
procedure MakeTinyImage(V: TNNetVolume; SizeX, SizeY: integer);
var
  X, Y: integer;
begin
  V.ReSize(SizeX, SizeY, 3);
  for X := 0 to SizeX - 1 do
    for Y := 0 to SizeY - 1 do
    begin
      V[X, Y, 0] := (X * 255) div (SizeX - 1);
      V[X, Y, 1] := (Y * 255) div (SizeY - 1);
      V[X, Y, 2] := ((X + Y) * 255) div (SizeX + SizeY - 2);
    end;
end;

// Pico smoke: writes a 50x40 PNG and cuts it out with the fixture (96x64 net).
procedure RunFixtureSmoke(const InPng, OutPng, MaskPng: string);
var
  Img: TNNetVolume;
  Opt: TBackgroundRemovalOptions;
begin
  Img := TNNetVolume.Create;
  try
    MakeTinyImage(Img, 50, 40);
    if not SaveImageFromVolumeIntoFile(Img, InPng) then
      raise Exception.Create('could not save ' + InPng);
  finally
    Img.Free;
  end;
  ParseOptions(Opt);
  Opt.ModelFile := FixturesDir + 'tiny_birefnet.safetensors';
  Opt.ConfigFile := FixturesDir + 'tiny_birefnet_config.json';
  Opt.InputFile := InPng;
  Opt.OutputFile := OutPng;
  Opt.MaskFile := MaskPng;
  RemoveBackgroundFromFile(Opt);
end;

var
  Options: TBackgroundRemovalOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFile <> '' then
      RemoveBackgroundFromFile(Options)
    else
    begin
      WriteLn('BiRefNet background removal smoke (random pico weights)');
      WriteLn('=======================================================');
      RunFixtureSmoke('birefnet_in.png', 'birefnet_out.png',
        'birefnet_mask.png');
      WriteLn('Done.');
    end;
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
