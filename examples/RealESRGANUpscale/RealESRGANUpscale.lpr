program RealESRGANUpscale;
(*
RealESRGANUpscale: Real-ESRGAN / ESRGAN RRDBNet super-resolution with the
importer in neuralpretrained.pas (BuildRRDBNetFromSafeTensorsEx, RRDBNetUpscale).

Two modes:
  * no arguments: a smoke of the committed random pico fixtures
    tests/fixtures/tiny_rrdbnet and tiny_rrdbnet_x2: synthesize a 6x6 PNG, read it back,
    upscale x4 and x2, write the output PNGs.
  * --model FILE --input IMG --output IMG: upscale a real image with a real
    checkpoint (RealESRGAN_x4plus / x2plus .pth or .safetensors). Without a
    config.json next to FILE the config is inferred from the tensor shapes.
    --tile N runs the net on N x N tiles (plus --tile-pad context on each side,
    seams feather-blended over --tile-overlap pixels) so large images fit in RAM.

Pixel values go in as [0,1] RGB and come out clamped to [0,1], as basicsr's
RealESRGANer does.

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
  {$IFDEF OpenCL}neuralopencl,{$ENDIF}
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained;

const
  // Fixtures committed by tools/rrdbnet_tiny_fixture.py.
  FixturesDir = '../../tests/fixtures/';

type
  TUpscaleOptions = record
    ModelFile, ConfigFile, InputFile, OutputFile: string;
    Tile, TilePad, TileOverlap: integer;
    UseOpenCL: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: RealESRGANUpscale                       (pico fixture smoke)');
  WriteLn('       RealESRGANUpscale --model FILE --input IMG --output IMG [options]');
  WriteLn('  --model FILE         RRDBNet checkpoint (.pth / .safetensors)');
  WriteLn('  --config FILE        config.json (default: next to --model, else');
  WriteLn('                       inferred from the tensor shapes)');
  WriteLn('  --input IMG          image to upscale (PNG / JPEG / BMP)');
  WriteLn('  --output IMG         upscaled image (PNG)');
  WriteLn('  --tile N             tile edge in input pixels; 0 = whole image ',
    '(default 0)');
  WriteLn('  --tile-pad N         context pixels added on each tile side ',
    '(default 10)');
  WriteLn('  --tile-overlap N     pixels feather-blended across seams ',
    '(default 8)');
  WriteLn('  --gpu                run the net on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TUpscaleOptions): boolean;
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
  Opt.Tile := 0;
  Opt.TilePad := 10;
  Opt.TileOverlap := 8;
  Opt.UseOpenCL := false;
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
    else if Arg = '--tile' then Opt.Tile := StrToInt(NextArg())
    else if Arg = '--tile-pad' then Opt.TilePad := StrToInt(NextArg())
    else if Arg = '--tile-overlap' then Opt.TileOverlap := StrToInt(NextArg())
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
  if (Opt.ModelFile <> '') and
     ((Opt.InputFile = '') or (Opt.OutputFile = '')) then
    raise Exception.Create('--model needs --input and --output');
end;

// Loads ImageFile as RGB in [0,1], upscales it and writes OutputFile.
procedure UpscaleFile(const Opt: TUpscaleOptions);
var
  Config, TileConfig: TRRDBNetConfig;
  NN: TNNet;
  Image, Upscaled: TNNetVolume;
  ComputeText: string;
  StartTime: TDateTime;
  PixelPos: integer;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
  {$ENDIF}
begin
  Image := TNNetVolume.Create;
  Upscaled := nil;
  NN := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    if not LoadImageFromFileIntoVolume(Opt.InputFile, Image) then
      raise Exception.Create('could not load ' + Opt.InputFile);
    Image.Divi(255.0);
    Config := LoadRRDBNetConfig(Opt.ModelFile, Opt.ConfigFile);
    if Image.Depth <> Config.NumInCh then
      raise Exception.Create(Format('the image has %d channels, the model ' +
        'takes %d', [Image.Depth, Config.NumInCh]));
    TileConfig := RRDBNetTileConfig(Config, Image.SizeX, Image.SizeY,
      Opt.Tile, Opt.TilePad);
    WriteLn(RRDBNetConfigToString(TileConfig));
    NN := BuildRRDBNetFromSafeTensorsEx(Opt.ModelFile, TileConfig,
      {pTrainable=}false);
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
    WriteLn(Format('Upscaling %dx%d x%d on %s, net input %dx%d',
      [Image.SizeX, Image.SizeY, Config.Scale, ComputeText,
       TileConfig.InputWidth, TileConfig.InputHeight]));
    StartTime := Now();
    Upscaled := RRDBNetUpscale(NN, TileConfig, Image, Opt.TilePad,
      Opt.TileOverlap);
    WriteLn(Format('  %.1f s', [(Now() - StartTime) * 86400.0]));
    for PixelPos := 0 to Upscaled.Size - 1 do
      Upscaled.FData[PixelPos] := EnsureRange(Upscaled.FData[PixelPos], 0, 1) * 255;
    if not SaveImageFromVolumeIntoFile(Upscaled, Opt.OutputFile) then
      raise Exception.Create('could not save ' + Opt.OutputFile);
    WriteLn(Format('  %dx%d -> %dx%d  (%s)', [Image.SizeX, Image.SizeY,
      Upscaled.SizeX, Upscaled.SizeY, Opt.OutputFile]));
  finally
    NN.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    Upscaled.Free;
    Image.Free;
  end;
end;

// Builds a tiny RGB image (0..255) as a deterministic colour gradient so the
// smoke needs no input asset on disk.
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

// Pico smoke: writes a 6x6 PNG and upscales it with the fixture Stem.
procedure RunFixtureSmoke(const Stem, InPng, OutPng: string);
var
  Img: TNNetVolume;
  Opt: TUpscaleOptions;
begin
  Img := TNNetVolume.Create;
  try
    MakeTinyImage(Img, 6);
    if not SaveImageFromVolumeIntoFile(Img, InPng) then
      raise Exception.Create('could not save ' + InPng);
  finally
    Img.Free;
  end;
  ParseOptions(Opt);
  Opt.ModelFile := FixturesDir + Stem + '.safetensors';
  Opt.ConfigFile := FixturesDir + Stem + '_config.json';
  Opt.InputFile := InPng;
  Opt.OutputFile := OutPng;
  UpscaleFile(Opt);
end;

var
  Options: TUpscaleOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFile <> '' then
      UpscaleFile(Options)
    else
    begin
      WriteLn('Real-ESRGAN / ESRGAN RRDBNet upscale smoke (random pico weights)');
      WriteLn('================================================================');
      RunFixtureSmoke('tiny_rrdbnet',    'esrgan_in_x4.png', 'esrgan_out_x4.png');
      RunFixtureSmoke('tiny_rrdbnet_x2', 'esrgan_in_x2.png', 'esrgan_out_x2.png');
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
