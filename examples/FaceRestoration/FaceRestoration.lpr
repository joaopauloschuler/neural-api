program FaceRestoration;
(*
FaceRestoration: GFPGAN v1.3 / v1.4 blind face restoration (Wang et al.,
CVPR 2021) with the importer in neuralpretrained.pas (BuildGFPGANFromSafeTensors
+ GFPGANRestoreFace).

Two modes:
  * no arguments: a smoke of the committed random pico fixture
    tests/fixtures/tiny_gfpgan: synthesize a small colour PNG, read it back,
    restore it, write the output PNG.
  * --model FILE --input IMG --output IMG: restore a real face with an official
    GFPGAN checkpoint (GFPGANv1.4.pth / GFPGANv1.3.pth, no conversion).

The input must already be an ALIGNED, CROPPED face (eyes and mouth where the
FFHQ alignment puts them, as GFPGAN's own --aligned mode expects). Face
detection and alignment are not part of this program. A face of another size
is resized to the net's resolution (512 for the official checkpoints).

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
  // Fixture committed with the GFPGAN importer tests.
  FixturesDir = '../../tests/fixtures/';

type
  TFaceRestorationOptions = record
    ModelFile, ConfigFile, InputFile, OutputFile: string;
    NoiseMode: TStyleGAN2NoiseMode;
    Seed: integer;
    KeepSize, UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: FaceRestoration                       (pico fixture smoke)');
  WriteLn('       FaceRestoration --model FILE --input IMG --output IMG [options]');
  WriteLn('  --model FILE         GFPGAN checkpoint (GFPGANv1.4.pth / .safetensors)');
  WriteLn('  --config FILE        config.json (default: next to --model, else the');
  WriteLn('                       v1.4 settings; shapes come from the tensors)');
  WriteLn('  --input IMG          ALIGNED, cropped face (PNG / JPEG / BMP)');
  WriteLn('  --output IMG         restored face (PNG)');
  WriteLn('  --keep-size          resize the restored face back to the input size');
  WriteLn('                       (default: the net resolution, 512)');
  WriteLn('  --noise MODE         random (GFPGAN''s default), const (the ',
    'checkpoint''s');
  WriteLn('                       stored maps) or none (default random)');
  WriteLn('  --seed N             seed of the random noise (default: time based)');
  WriteLn('  --int8               store the convolution weights in int8 (less RAM)');
  WriteLn('  --gpu                run the net on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

function ParseNoiseMode(const ModeName: string): TStyleGAN2NoiseMode;
begin
  if ModeName = 'random' then Result := snmRandom
  else if ModeName = 'const' then Result := snmConst
  else if ModeName = 'none' then Result := snmNone
  else
    raise Exception.Create('--noise must be random, const or none, got ' +
      ModeName);
end;

function ParseOptions(out Opt: TFaceRestorationOptions): boolean;
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
  Opt.NoiseMode := snmRandom;
  Opt.Seed := -1;
  Opt.KeepSize := false;
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
    else if Arg = '--keep-size' then Opt.KeepSize := true
    else if Arg = '--noise' then Opt.NoiseMode := ParseNoiseMode(NextArg())
    else if Arg = '--seed' then Opt.Seed := StrToInt(NextArg())
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
end;

// Loads InputFile, restores it with the GFPGAN checkpoint and writes OutputFile.
procedure RestoreFile(const Opt: TFaceRestorationOptions);
var
  NN: TNNet;
  Config: TGFPGANConfig;
  Face, Restored, Resized: TNNetVolume;
  ComputeText: string;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
  {$ENDIF}
begin
  Face := TNNetVolume.Create;
  Restored := TNNetVolume.Create;
  Resized := TNNetVolume.Create;
  NN := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    if not LoadImageFromFileIntoVolume(Opt.InputFile, Face) then
      raise Exception.Create('could not load ' + Opt.InputFile);
    StartTime := Now();
    NN := BuildGFPGANFromSafeTensors(Opt.ModelFile, Config, {pTrainable=}false,
      Opt.ConfigFile, Opt.NoiseMode);
    WriteLn(GFPGANConfigToString(Config));
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
    if Opt.Seed >= 0 then RestartRandSeed(Opt.Seed);
    WriteLn(Format('Restoring a %dx%d face on %s, net resolution %dx%d',
      [Face.SizeX, Face.SizeY, ComputeText, Config.OutSize, Config.OutSize]));
    StartTime := Now();
    GFPGANRestoreFace(NN, Config, Face, Restored);
    WriteLn(Format('  %.1f s', [(Now() - StartTime) * 86400.0]));
    if Opt.KeepSize and
       ((Face.SizeX <> Restored.SizeX) or (Face.SizeY <> Restored.SizeY)) then
    begin
      ResizeImageBilinearNoAntialias(Restored, Resized, Face.SizeX, Face.SizeY);
      Restored.Copy(Resized);
    end;
    if not SaveImageFromVolumeIntoFile(Restored, Opt.OutputFile) then
      raise Exception.Create('could not save ' + Opt.OutputFile);
    WriteLn(Format('  %dx%d -> %s', [Restored.SizeX, Restored.SizeY,
      Opt.OutputFile]));
  finally
    NN.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    Resized.Free;
    Restored.Free;
    Face.Free;
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

// Pico smoke: writes a 40x30 PNG and restores it with the fixture (64x64 net).
procedure RunFixtureSmoke(const InPng, OutPng: string);
var
  Img: TNNetVolume;
  Opt: TFaceRestorationOptions;
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
  Opt.ModelFile := FixturesDir + 'tiny_gfpgan.pth';
  Opt.ConfigFile := FixturesDir + 'tiny_gfpgan_config.json';
  Opt.InputFile := InPng;
  Opt.OutputFile := OutPng;
  Opt.NoiseMode := snmConst;
  RestoreFile(Opt);
end;

var
  Options: TFaceRestorationOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFile <> '' then
      RestoreFile(Options)
    else
    begin
      WriteLn('GFPGAN face restoration smoke (random pico weights)');
      WriteLn('===================================================');
      RunFixtureSmoke('gfpgan_in.png', 'gfpgan_out.png');
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
