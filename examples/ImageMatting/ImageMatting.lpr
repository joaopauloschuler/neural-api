program ImageMatting;
(*
ImageMatting: ViTMatte trimap-based image matting (Yao et al. 2023,
arXiv:2305.15272) with the importer in neuralpretrained.pas
(BuildViTMatteFromSafeTensors + ViTMatteEstimateAlpha).

Two modes:
  * no arguments: a smoke of the committed random pico fixtures
    tests/fixtures/tiny_vitmatte and tiny_birefnet: synthesize a small colour
    PNG, matte it once with a trimap file and once with --auto-trimap.
  * --model DIR_OR_FILE --image IMG (--trimap IMG | --auto-trimap FILE):
    estimate the alpha matte of a real image with a real checkpoint
    (hustvl/vitmatte-small-composition-1k or -base-composition-1k).

The trimap is a gray image: 0 = background, 128 = unknown, 255 = foreground.
--auto-trimap builds it from a BiRefNet / RMBG-2.0 mask with TrimapFromMask.
The ViTMatte net is built for the image size padded to a multiple of 32.

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
  // Fixtures committed with the ViTMatte and BiRefNet importer tests.
  FixturesDir = '../../tests/fixtures/';
  DefaultUnknownBand = 10;

type
  TImageMattingOptions = record
    ModelPath, ConfigFile, ImageFile, TrimapFile: string;
    AutoTrimapModel, AutoTrimapConfig: string;
    OutputFile, CutoutFile, SaveTrimapFile: string;
    UnknownBand, BiRefNetSize: integer;
    UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: ImageMatting                            (pico fixture smoke)');
  WriteLn('       ImageMatting --model DIR_OR_FILE --image IMG');
  WriteLn('                    (--trimap IMG | --auto-trimap FILE) [options]');
  WriteLn('  --model DIR_OR_FILE  ViTMatte model.safetensors, or the folder holding it');
  WriteLn('  --config FILE        ViTMatte config.json (default: next to the model)');
  WriteLn('  --image IMG          image (PNG / JPEG / BMP)');
  WriteLn('  --trimap IMG         gray trimap of the image size: 0 = background,');
  WriteLn('                       128 = unknown, 255 = foreground');
  WriteLn('  --auto-trimap FILE   build the trimap from the mask of this BiRefNet /');
  WriteLn('                       RMBG-2.0 model.safetensors');
  WriteLn('  --auto-trimap-config FILE  BiRefNet config.json (default: next to it)');
  WriteLn('  --birefnet-size N    BiRefNet input resolution, a multiple of 32');
  WriteLn('                       (default: its config, 1024 for the official one)');
  WriteLn('  --unknown-band N     auto-trimap unknown band half-width in pixels ',
    '(default ', DefaultUnknownBand, ')');
  WriteLn('  --output IMG         the alpha matte, gray PNG');
  WriteLn('  --cutout IMG         the image with the alpha matte as alpha, RGBA PNG');
  WriteLn('  --save-trimap IMG    the trimap used, gray PNG');
  WriteLn('  --int8               store the conv and linear weights in int8');
  WriteLn('  --gpu                run the nets on OpenCL (needs a build with -dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TImageMattingOptions): boolean;
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
  Opt.ModelPath := '';
  Opt.ConfigFile := '';
  Opt.ImageFile := '';
  Opt.TrimapFile := '';
  Opt.AutoTrimapModel := '';
  Opt.AutoTrimapConfig := '';
  Opt.OutputFile := '';
  Opt.CutoutFile := '';
  Opt.SaveTrimapFile := '';
  Opt.UnknownBand := DefaultUnknownBand;
  Opt.BiRefNetSize := 0;
  Opt.UseOpenCL := false;
  Opt.UseInt8 := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then Opt.ModelPath := NextArg()
    else if Arg = '--config' then Opt.ConfigFile := NextArg()
    else if Arg = '--image' then Opt.ImageFile := NextArg()
    else if Arg = '--trimap' then Opt.TrimapFile := NextArg()
    else if Arg = '--auto-trimap' then Opt.AutoTrimapModel := NextArg()
    else if Arg = '--auto-trimap-config' then Opt.AutoTrimapConfig := NextArg()
    else if Arg = '--birefnet-size' then Opt.BiRefNetSize := StrToInt(NextArg())
    else if Arg = '--unknown-band' then Opt.UnknownBand := StrToInt(NextArg())
    else if Arg = '--output' then Opt.OutputFile := NextArg()
    else if Arg = '--cutout' then Opt.CutoutFile := NextArg()
    else if Arg = '--save-trimap' then Opt.SaveTrimapFile := NextArg()
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
  if Opt.ModelPath = '' then exit;
  if Opt.ImageFile = '' then
    raise Exception.Create('--model needs --image');
  if (Opt.TrimapFile = '') = (Opt.AutoTrimapModel = '') then
    raise Exception.Create('give exactly one of --trimap and --auto-trimap');
  if (Opt.OutputFile = '') and (Opt.CutoutFile = '') then
    raise Exception.Create('--model needs --output and/or --cutout');
  if Opt.UnknownBand < 0 then
    raise Exception.Create('--unknown-band must be >= 0');
  if (Opt.BiRefNetSize < 0) or (Opt.BiRefNetSize mod 32 <> 0) then
    raise Exception.Create('--birefnet-size must be a positive multiple of 32');
end;

// Saves channel 0 of a (W, H, 1) 0..255 volume as a gray RGB PNG.
procedure SaveGrayPNG(Gray: TNNetVolume; const FileName: string);
var
  RGB: TNNetVolume;
begin
  RGB := TNNetVolume.Create;
  try
    RGB.CopyChannels(Gray, [0, 0, 0]);
    if not SaveImageFromVolumeIntoFile(RGB, FileName) then
      raise Exception.Create('could not save ' + FileName);
  finally
    RGB.Free;
  end;
end;

// Applies --int8 and --gpu to NN; returns where it runs, for the log.
function PrepareCompute(NN: TNNet; const Opt: TImageMattingOptions
  {$IFDEF OpenCL}; var OpenCLDevices: TEasyOpenCL{$ENDIF}): string;
{$IFDEF OpenCL}
var
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
{$ENDIF}
begin
  // Quantize before EnableOpenCL: the OpenCL arming reads the int8 codes.
  if Opt.UseInt8 then NN.QuantizeWeightsInt8();
  Result := 'CPU';
  if Opt.UseOpenCL then
  begin
    {$IFDEF OpenCL}
    if OpenCLDevices = nil then OpenCLDevices := TEasyOpenCL.Create();
    OpenCLPlatform := Opt.OpenCLPlatform;
    OpenCLDevice := Opt.OpenCLDevice;
    if OpenCLDevices.SelectPlatformAndDevice(OpenCLPlatform, OpenCLDevice,
      OpenCLProblem) then
    begin
      NN.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
        OpenCLDevices.Devices[OpenCLDevice]);
      Result := 'OpenCL on ' + OpenCLDevices.PlatformNames[OpenCLPlatform] +
        ' / ' + OpenCLDevices.DeviceNames[OpenCLDevice];
    end
    else
      Result := 'CPU (' + OpenCLProblem + ')';
    {$ELSE}
    Result := 'CPU (built without -dOpenCL)';
    {$ENDIF}
  end;
  if Opt.UseInt8 then Result := Result + ', int8 weights';
end;

// Trimap (W, H, 1) := TrimapFromMask of the BiRefNet mask of Image.
procedure BuildAutoTrimap(Image, Trimap: TNNetVolume;
  const Opt: TImageMattingOptions
  {$IFDEF OpenCL}; var OpenCLDevices: TEasyOpenCL{$ENDIF});
var
  NN: TNNet;
  Config: TBiRefNetConfig;
  Mask: TNNetVolume;
  ComputeText: string;
  StartTime: TDateTime;
begin
  Mask := TNNetVolume.Create;
  NN := nil;
  try
    NN := BuildBiRefNetFromSafeTensors(Opt.AutoTrimapModel, Config,
      {pTrainable=}false, Opt.AutoTrimapConfig, Opt.BiRefNetSize);
    ComputeText := PrepareCompute(NN, Opt
      {$IFDEF OpenCL}, OpenCLDevices{$ENDIF});
    WriteLn(Format('BiRefNet mask on %s, net input %dx%d',
      [ComputeText, Config.InputWidth, Config.InputHeight]));
    StartTime := Now();
    BiRefNetRemoveBackground(NN, Config, Image, Mask);
    TrimapFromMask(Mask, Trimap, Opt.UnknownBand);
    WriteLn(Format('  %.1f s, unknown band %d px',
      [(Now() - StartTime) * 86400.0, Opt.UnknownBand]));
  finally
    NN.Free;
    Mask.Free;
  end;
end;

// Loads a trimap file and keeps its first channel; it must match the image size.
procedure LoadTrimap(const FileName: string; Image, Trimap: TNNetVolume);
var
  TrimapRGB: TNNetVolume;
begin
  TrimapRGB := TNNetVolume.Create;
  try
    if not LoadImageFromFileIntoVolume(FileName, TrimapRGB) then
      raise Exception.Create('could not load ' + FileName);
    if (TrimapRGB.SizeX <> Image.SizeX) or (TrimapRGB.SizeY <> Image.SizeY) then
      raise Exception.Create(Format('trimap %s is %dx%d but the image is %dx%d',
        [FileName, TrimapRGB.SizeX, TrimapRGB.SizeY, Image.SizeX,
         Image.SizeY]));
    Trimap.CopyChannels(TrimapRGB, [0]);
  finally
    TrimapRGB.Free;
  end;
end;

// Loads ImageFile + its trimap, estimates the alpha matte and writes the outputs.
procedure MatteImageFromFile(const Opt: TImageMattingOptions);
var
  NN: TNNet;
  Config: TViTMatteConfig;
  Image, Trimap, Alpha, RGBA: TNNetVolume;
  ModelFile, ComputeText: string;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  {$ENDIF}
begin
  Image := TNNetVolume.Create;
  Trimap := TNNetVolume.Create;
  Alpha := TNNetVolume.Create;
  RGBA := TNNetVolume.Create;
  NN := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    if not LoadImageFromFileIntoVolume(Opt.ImageFile, Image) then
      raise Exception.Create('could not load ' + Opt.ImageFile);
    if Opt.TrimapFile <> '' then
      LoadTrimap(Opt.TrimapFile, Image, Trimap)
    else
      BuildAutoTrimap(Image, Trimap, Opt
        {$IFDEF OpenCL}, OpenCLDevices{$ENDIF});
    if Opt.SaveTrimapFile <> '' then
    begin
      SaveGrayPNG(Trimap, Opt.SaveTrimapFile);
      WriteLn(Format('  %dx%d trimap -> %s', [Trimap.SizeX, Trimap.SizeY,
        Opt.SaveTrimapFile]));
    end;
    ModelFile := Opt.ModelPath;
    if DirectoryExists(ModelFile) then
      ModelFile := IncludeTrailingPathDelimiter(ModelFile) + 'model.safetensors';
    StartTime := Now();
    NN := BuildViTMatteFromSafeTensors(ModelFile, Config, {pTrainable=}false,
      Opt.ConfigFile, Image.SizeX, Image.SizeY);
    WriteLn(ViTMatteConfigToString(Config));
    WriteLn(Format('  loaded in %.1f s', [(Now() - StartTime) * 86400.0]));
    ComputeText := PrepareCompute(NN, Opt
      {$IFDEF OpenCL}, OpenCLDevices{$ENDIF});
    WriteLn(Format('Matting %dx%d on %s, net input %dx%d',
      [Image.SizeX, Image.SizeY, ComputeText, Config.InputWidth,
       Config.InputHeight]));
    StartTime := Now();
    ViTMatteEstimateAlpha(NN, Image, Trimap, Alpha);
    WriteLn(Format('  %.1f s', [(Now() - StartTime) * 86400.0]));
    Alpha.Mul(255);
    if Opt.OutputFile <> '' then
    begin
      SaveGrayPNG(Alpha, Opt.OutputFile);
      WriteLn(Format('  %dx%d alpha -> %s', [Alpha.SizeX, Alpha.SizeY,
        Opt.OutputFile]));
    end;
    if Opt.CutoutFile <> '' then
    begin
      BiRefNetComposeRGBA(Image, Alpha, RGBA);
      if not SaveImageFromVolumeIntoFile(RGBA, Opt.CutoutFile) then
        raise Exception.Create('could not save ' + Opt.CutoutFile);
      WriteLn(Format('  %dx%d RGBA -> %s', [RGBA.SizeX, RGBA.SizeY,
        Opt.CutoutFile]));
    end;
  finally
    NN.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    RGBA.Free;
    Alpha.Free;
    Trimap.Free;
    Image.Free;
  end;
end;

// Builds a deterministic RGB colour gradient (0..255) and a centred
// ellipse mask (0/255) so the smoke needs no input asset on disk.
procedure MakeTinyImageAndMask(Image, Mask: TNNetVolume; SizeX, SizeY: integer);
var
  X, Y, MaxX, MaxY: integer;
begin
  Image.ReSize(SizeX, SizeY, 3);
  Mask.ReSize(SizeX, SizeY, 1);
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      Image[X, Y, 0] := (X * 255) div MaxX;
      Image[X, Y, 1] := (Y * 255) div MaxY;
      Image[X, Y, 2] := ((X + Y) * 255) div (MaxX + MaxY);
      if Sqr((2 * X - MaxX) / SizeX) + Sqr((2 * Y - MaxY) / SizeY) < 0.5 then
        Mask[X, Y, 0] := 255
      else
        Mask[X, Y, 0] := 0;
    end;
end;

// Pico smoke: a 45x30 PNG (64x32 net) matted with a trimap file, then with
// --auto-trimap from the pico BiRefNet.
procedure RunFixtureSmoke();
var
  Image, Mask, Trimap: TNNetVolume;
  Opt: TImageMattingOptions;
begin
  Image := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
  Trimap := TNNetVolume.Create;
  try
    MakeTinyImageAndMask(Image, Mask, 45, 30);
    if not SaveImageFromVolumeIntoFile(Image, 'vitmatte_in.png') then
      raise Exception.Create('could not save vitmatte_in.png');
    TrimapFromMask(Mask, Trimap, 3);
    SaveGrayPNG(Trimap, 'vitmatte_trimap.png');
  finally
    Trimap.Free;
    Mask.Free;
    Image.Free;
  end;
  ParseOptions(Opt);
  Opt.ModelPath := FixturesDir + 'tiny_vitmatte';
  Opt.ImageFile := 'vitmatte_in.png';
  Opt.TrimapFile := 'vitmatte_trimap.png';
  Opt.OutputFile := 'vitmatte_alpha.png';
  Opt.CutoutFile := 'vitmatte_cutout.png';
  MatteImageFromFile(Opt);
  WriteLn('-- auto-trimap from the pico BiRefNet');
  Opt.TrimapFile := '';
  Opt.AutoTrimapModel := FixturesDir + 'tiny_birefnet.safetensors';
  Opt.AutoTrimapConfig := FixturesDir + 'tiny_birefnet_config.json';
  Opt.UnknownBand := 3;
  Opt.SaveTrimapFile := 'vitmatte_auto_trimap.png';
  Opt.OutputFile := 'vitmatte_auto_alpha.png';
  Opt.CutoutFile := 'vitmatte_auto_cutout.png';
  MatteImageFromFile(Opt);
end;

var
  Options: TImageMattingOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelPath <> '' then
      MatteImageFromFile(Options)
    else
    begin
      WriteLn('ViTMatte image matting smoke (random pico weights)');
      WriteLn('==================================================');
      RunFixtureSmoke();
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
