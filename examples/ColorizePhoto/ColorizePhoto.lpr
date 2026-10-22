program ColorizePhoto;
(*
ColorizePhoto: DDColor photo colorization (Kang et al., ICCV 2023) with the
importer in neuralpretrained.pas (BuildDDColorFromSafeTensors + DDColorColorize).

Two modes:
  * no arguments: a smoke of the committed random pico fixture
    tests/fixtures/tiny_ddcolor: synthesize a small colour PNG, read it back,
    colorize it, write the output PNG.
  * --model FILE --input IMG --output IMG: colorize a real image with a real
    DDColor checkpoint (official .pth / pytorch_model.bin / .safetensors).
    Without a config.json next to FILE (or --config), the encoder widths come
    from the tensor shapes and the net input defaults to 512x512.

The net sees only the lightness (L) of the photo, at --size x --size. Its a*b*
chroma is resized back to the photo's own size and recombined with the
photo's full-resolution L, so the output keeps every detail of the input.

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
  // Fixture committed with the DDColor importer tests.
  FixturesDir = '../../tests/fixtures/';

type
  TColorizeOptions = record
    ModelFile, ConfigFile, InputFile, OutputFile: string;
    InputSize: integer;
    UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: ColorizePhoto                       (pico fixture smoke)');
  WriteLn('       ColorizePhoto --model FILE --input IMG --output IMG [options]');
  WriteLn('  --model FILE         DDColor checkpoint (.pth / .bin / .safetensors)');
  WriteLn('  --config FILE        config.json (default: next to --model, else');
  WriteLn('                       inferred from the tensor shapes, 512 input)');
  WriteLn('  --input IMG          photo to colorize (PNG / JPEG / BMP, color or gray)');
  WriteLn('  --output IMG         colorized photo (PNG)');
  WriteLn('  --size N             net input resolution, a multiple of 32: 256 is ');
  WriteLn('                       faster, 512 is the official setting (default: ');
  WriteLn('                       config.json input_size, else 512)');
  WriteLn('  --int8               store the weights in int8 (less RAM, small ',
    'accuracy loss)');
  WriteLn('  --gpu                run the net on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TColorizeOptions): boolean;
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
  if (Opt.ModelFile <> '') and
     ((Opt.InputFile = '') or (Opt.OutputFile = '')) then
    raise Exception.Create('--model needs --input and --output');
  if (Opt.InputSize < 0) or (Opt.InputSize mod 32 <> 0) then
    raise Exception.Create('--size must be a positive multiple of 32');
end;

// Loads InputFile, colorizes it with the DDColor checkpoint and writes OutputFile.
procedure ColorizeFile(const Opt: TColorizeOptions);
var
  NN: TNNet;
  Config: TDDColorConfig;
  Image, Colorized: TNNetVolume;
  ComputeText: string;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
  {$ENDIF}
begin
  Image := TNNetVolume.Create;
  Colorized := TNNetVolume.Create;
  NN := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    if not LoadImageFromFileIntoVolume(Opt.InputFile, Image) then
      raise Exception.Create('could not load ' + Opt.InputFile);
    StartTime := Now();
    NN := BuildDDColorFromSafeTensors(Opt.ModelFile, Config,
      {pTrainable=}false, Opt.ConfigFile, Opt.InputSize);
    WriteLn(DDColorConfigToString(Config));
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
    WriteLn(Format('Colorizing %dx%d on %s, net input %dx%d',
      [Image.SizeX, Image.SizeY, ComputeText, Config.InputWidth,
       Config.InputHeight]));
    StartTime := Now();
    DDColorColorize(NN, Config, Image, Colorized);
    WriteLn(Format('  %.1f s', [(Now() - StartTime) * 86400.0]));
    if not SaveImageFromVolumeIntoFile(Colorized, Opt.OutputFile) then
      raise Exception.Create('could not save ' + Opt.OutputFile);
    WriteLn(Format('  %dx%d -> %s', [Colorized.SizeX, Colorized.SizeY,
      Opt.OutputFile]));
  finally
    NN.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    Colorized.Free;
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

// Pico smoke: writes a 40x30 PNG and colorizes it with the fixture.
procedure RunFixtureSmoke(const InPng, OutPng: string);
var
  Img: TNNetVolume;
  Opt: TColorizeOptions;
begin
  Img := TNNetVolume.Create;
  try
    MakeTinyImage(Img, 40, 30);
    if not SaveImageFromVolumeIntoFile(Img, InPng) then
      raise Exception.Create('could not save ' + InPng);
  finally
    Img.Free;
  end;
  ParseOptions(Opt);
  Opt.ModelFile := FixturesDir + 'tiny_ddcolor.safetensors';
  Opt.ConfigFile := FixturesDir + 'tiny_ddcolor_config.json';
  Opt.InputFile := InPng;
  Opt.OutputFile := OutPng;
  ColorizeFile(Opt);
end;

var
  Options: TColorizeOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFile <> '' then
      ColorizeFile(Options)
    else
    begin
      WriteLn('DDColor photo colorization smoke (random pico weights)');
      WriteLn('======================================================');
      RunFixtureSmoke('ddcolor_in.png', 'ddcolor_out.png');
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
