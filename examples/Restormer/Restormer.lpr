program Restormer;
(*
Restormer: image deraining, deblurring and denoising with the Restormer
importer in neuralpretrained.pas (BuildRestormerFromSafeTensorsEx +
PaddedTiledNetCompute). Restormer (Zamir et al. 2022, arXiv:2111.09881,
swz30/Restormer) restores an image at its own resolution with a 4-level U-Net
of transformer blocks; one architecture covers every task, only the weights
and two options (LayerNorm type, input channels) change.

Two modes:
  * no arguments: a smoke run of the committed random pico fixtures
    tests/fixtures/tiny_restormer and tiny_restormer_dp (random weights, so the
    output is not a restoration): a synthetic 37x29 image (and a left/right
    pair for the dual-pixel net) goes through reflect padding and 16x16 tiles;
    the results are written to restormer_smoke.ppm and restormer_dp_smoke.ppm.
  * --task TASK --input IMG --output IMG [--model FILE]: restore a real image
    with an official release checkpoint (.pth, read directly - no conversion).
    --model defaults to the release file name of TASK in the current folder.
    The dual-pixel defocus checkpoint takes --left IMG --right IMG instead of
    --input (--task dual-pixel is implied).

Pixels go in as [0,1] and come out clamped to [0,1], as the official test
scripts do. --tile N runs the net on N x N tiles (plus --tile-pad context on
each side, seams feather-blended over --tile-overlap pixels) so large photos
fit in RAM; all three are rounded up to multiples of 8.

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
  cFixturesDir = '../../tests/fixtures/';
  cSmokeOutput = 'restormer_smoke.ppm';
  cDualPixelSmokeOutput = 'restormer_dp_smoke.ppm';
  cDualPixelTask = 'dual-pixel';
  // Restormer downsamples 3 times: the net input is a multiple of 2^3.
  cSizeMultiple = 8;
  cMaxPixelValue = 255;

type
  TRestoreOptions = record
    Task, ModelFile, ConfigFile, InputFile, OutputFile: string;
    LeftFile, RightFile: string;
    Tile, TilePad, TileOverlap: integer;
    UseOpenCL: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: Restormer                (pico fixture smoke run)');
  WriteLn('       Restormer --task TASK --input IMG --output IMG [options]');
  WriteLn('       Restormer --left IMG --right IMG --output IMG [options]');
  WriteLn('  --task TASK          derain, deblur (motion), defocus (single image),');
  WriteLn('                       dual-pixel (defocus from a left/right pair),');
  WriteLn('                       denoise (real camera noise), denoise-gaussian,');
  WriteLn('                       denoise-gray');
  WriteLn('  --model FILE         checkpoint (.pth or .safetensors); default: the');
  WriteLn('                       official release file name of TASK');
  WriteLn('  --config FILE        config.json overriding the TASK preset');
  WriteLn('  --input IMG          image to restore (PNG / JPEG / BMP / PPM)');
  WriteLn('  --left IMG, --right IMG  dual-pixel views of one shot (same size);');
  WriteLn('                       they imply --task dual-pixel');
  WriteLn('  --output IMG         restored image (format from the extension)');
  WriteLn('  --tile N             tile edge in pixels; 0 = whole image (default 256)');
  WriteLn('  --tile-pad N         context pixels per tile side (default 16)');
  WriteLn('  --tile-overlap N     pixels feather-blended across seams (default 32)');
  WriteLn('  --gpu                run the net on OpenCL (needs a build with -dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

// Maps a --task name to the official option set and its release file name.
procedure ResolveTask(const Task: string; out OfficialTask, ReleaseFile: string);
begin
  if Task = 'derain' then
  begin
    OfficialTask := 'deraining';
    ReleaseFile := 'deraining.pth';
  end
  else if Task = 'deblur' then
  begin
    OfficialTask := 'motion_deblurring';
    ReleaseFile := 'motion_deblurring.pth';
  end
  else if Task = 'defocus' then
  begin
    OfficialTask := 'single_image_defocus_deblurring';
    ReleaseFile := 'single_image_defocus_deblurring.pth';
  end
  else if Task = cDualPixelTask then
  begin
    OfficialTask := 'dual_pixel_defocus_deblurring';
    ReleaseFile := 'dual_pixel_defocus_deblurring.pth';
  end
  else if Task = 'denoise' then
  begin
    OfficialTask := 'real_denoising';
    ReleaseFile := 'real_denoising.pth';
  end
  else if Task = 'denoise-gaussian' then
  begin
    OfficialTask := 'gaussian_color_denoising';
    ReleaseFile := 'gaussian_color_denoising_blind.pth';
  end
  else if Task = 'denoise-gray' then
  begin
    OfficialTask := 'gaussian_gray_denoising';
    ReleaseFile := 'gaussian_gray_denoising_blind.pth';
  end
  else
    raise Exception.Create('unknown --task ' + Task + ' (derain, deblur, ' +
      'defocus, dual-pixel, denoise, denoise-gaussian, denoise-gray)');
end;

function ParseOptions(out Opt: TRestoreOptions): boolean;
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
  Opt.Task := '';
  Opt.ModelFile := '';
  Opt.ConfigFile := '';
  Opt.InputFile := '';
  Opt.OutputFile := '';
  Opt.LeftFile := '';
  Opt.RightFile := '';
  Opt.Tile := 256;
  Opt.TilePad := 16;
  Opt.TileOverlap := 32;
  Opt.UseOpenCL := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--task' then Opt.Task := NextArg()
    else if Arg = '--model' then Opt.ModelFile := NextArg()
    else if Arg = '--config' then Opt.ConfigFile := NextArg()
    else if Arg = '--input' then Opt.InputFile := NextArg()
    else if Arg = '--output' then Opt.OutputFile := NextArg()
    else if Arg = '--left' then Opt.LeftFile := NextArg()
    else if Arg = '--right' then Opt.RightFile := NextArg()
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
  if (Opt.LeftFile = '') <> (Opt.RightFile = '') then
    raise Exception.Create('--left and --right go together (one dual-pixel ' +
      'pair)');
  if (Opt.LeftFile <> '') and (Opt.InputFile <> '') then
    raise Exception.Create('give either --input or a --left/--right pair');
  if (Opt.Task = '') and (Opt.LeftFile <> '') then Opt.Task := cDualPixelTask;
  if (Opt.Task = '') and (Opt.InputFile <> '') then
    raise Exception.Create('--input needs --task');
  if (Opt.Task <> '') and (((Opt.InputFile = '') and (Opt.LeftFile = '')) or
     (Opt.OutputFile = '')) then
    raise Exception.Create('--task needs --input (or --left/--right) and ' +
      '--output');
end;

// Loads FileName as an RGB [0,1] volume.
procedure LoadUnitRangeImage(const FileName: string; Image: TNNetVolume);
begin
  if not LoadImageFromFileIntoVolume(FileName, Image) then
    raise Exception.Create('could not load ' + FileName);
  Image.Divi(cMaxPixelValue);
end;

// ITU-R BT.601 luma of an RGB [0,1] image (cv2 IMREAD_GRAYSCALE weights).
procedure RGBToGray(RGB, Gray: TNNetVolume);
var
  PixelPos, MaxPixelPos: integer;
begin
  Gray.ReSize(RGB.SizeX, RGB.SizeY, 1);
  MaxPixelPos := RGB.SizeX * RGB.SizeY - 1;
  for PixelPos := 0 to MaxPixelPos do
    Gray.FData[PixelPos] := 0.299 * RGB.FData[PixelPos * 3] +
      0.587 * RGB.FData[PixelPos * 3 + 1] + 0.114 * RGB.FData[PixelPos * 3 + 2];
end;

// Writes a [0,1] image (1 or 3 channels) to FileName, clamped to 0..255.
procedure SaveUnitRangeImage(Img: TNNetVolume; const FileName: string);
var
  ByteImage: TNNetVolume;
  PixelPos, MaxPixelPos: integer;
begin
  ByteImage := TNNetVolume.Create();
  try
    if Img.Depth = 1 then
    begin
      ByteImage.ReSize(Img.SizeX, Img.SizeY, 3);
      MaxPixelPos := Img.SizeX * Img.SizeY - 1;
      for PixelPos := 0 to MaxPixelPos do
      begin
        ByteImage.FData[PixelPos * 3] := Img.FData[PixelPos];
        ByteImage.FData[PixelPos * 3 + 1] := Img.FData[PixelPos];
        ByteImage.FData[PixelPos * 3 + 2] := Img.FData[PixelPos];
      end;
    end
    else
      ByteImage.Copy(Img);
    TNNetVolume.ReluL(Addr(ByteImage.FData[0]), Addr(ByteImage.FData[0]), 0, 1,
      0, ByteImage.Size);
    ByteImage.Mul(cMaxPixelValue);
    if not SaveImageFromVolumeIntoFile(ByteImage, FileName) then
      raise Exception.Create('could not save ' + FileName);
  finally
    ByteImage.Free;
  end;
end;

// Builds the net for a Image-sized run (whole image or one tile), restores
// Image and returns the result; ComputeText names where the net ran.
function RestoreImage(const WeightsFile: string; Config: TRestormerConfig;
  Image: TNNetVolume; const Opt: TRestoreOptions;
  out ComputeText: string): TNNetVolume;
var
  NN: TNNet;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
  {$ENDIF}
begin
  Config.InputWidth := TiledNetInputSide(Image.SizeX, Opt.Tile, Opt.TilePad,
    cSizeMultiple);
  Config.InputHeight := TiledNetInputSide(Image.SizeY, Opt.Tile, Opt.TilePad,
    cSizeMultiple);
  WriteLn(RestormerConfigToString(Config));
  NN := BuildRestormerFromSafeTensorsEx(WeightsFile, Config, {pTrainable=}false);
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    NN.ShareHostOutputsByLiveness := true;
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
        NN.ShareOpenCLOutputsByLiveness := true;
        NN.EnableOpenCL(OpenCLDevices.PlatformIds[OpenCLPlatform],
          OpenCLDevices.Devices[OpenCLDevice]);
        if NN.OpenCLEnabled() then
          ComputeText := 'OpenCL on ' +
            OpenCLDevices.PlatformNames[OpenCLPlatform] + ' / ' +
            OpenCLDevices.DeviceNames[OpenCLDevice]
        else
          ComputeText := 'CPU (OpenCL did not arm)';
      end
      else
        ComputeText := 'CPU (' + OpenCLProblem + ')';
      {$ELSE}
      ComputeText := 'CPU (built without -dOpenCL)';
      {$ENDIF}
    end;
    WriteLn(Format('Restoring %dx%dx%d on %s, net input %dx%d',
      [Image.SizeX, Image.SizeY, Image.Depth, ComputeText, Config.InputWidth,
       Config.InputHeight]));
    Result := PaddedTiledNetCompute(NN, Image, cSizeMultiple, Opt.TilePad,
      Opt.TileOverlap);
  finally
    NN.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
  end;
end;

procedure RestoreFile(const Opt: TRestoreOptions);
var
  Config: TRestormerConfig;
  OfficialTask, ReleaseFile, WeightsFile, ComputeText: string;
  Image, RightImage, NetInput, Restored: TNNetVolume;
  StartTime: TDateTime;
begin
  ResolveTask(Opt.Task, OfficialTask, ReleaseFile);
  if Opt.ConfigFile <> '' then
    Config := ReadRestormerConfigFromJSONFile(Opt.ConfigFile)
  else
    Config := RestormerTaskConfig(OfficialTask);
  if Config.DualPixelTask and (Opt.LeftFile = '') then
    raise Exception.Create('the dual-pixel checkpoint (inp_channels=' +
      IntToStr(Config.InpChannels) + ') restores a left/right pair: give ' +
      '--left IMG --right IMG instead of --input');
  if (not Config.DualPixelTask) and (Opt.LeftFile <> '') then
    raise Exception.Create('--left/--right feed the dual-pixel checkpoint ' +
      '(--task dual-pixel, inp_channels=6); this config takes ' +
      IntToStr(Config.InpChannels) + ' channel(s) from one --input image');
  if Opt.ModelFile <> '' then WeightsFile := Opt.ModelFile
  else WeightsFile := ReleaseFile;
  if not FileExists(WeightsFile) then
    raise Exception.Create('checkpoint not found: ' + WeightsFile);
  Image := TNNetVolume.Create;
  RightImage := TNNetVolume.Create;
  NetInput := TNNetVolume.Create;
  Restored := nil;
  try
    if Config.DualPixelTask then
    begin
      LoadUnitRangeImage(Opt.LeftFile, Image);
      LoadUnitRangeImage(Opt.RightFile, RightImage);
      RestormerPackDualPixel(Config, Image, RightImage, NetInput);
    end
    else
    begin
      LoadUnitRangeImage(Opt.InputFile, Image);
      if Config.InpChannels = 1 then RGBToGray(Image, NetInput)
      else if Config.InpChannels = 3 then NetInput.Copy(Image)
      else
        raise Exception.Create('the config takes ' +
          IntToStr(Config.InpChannels) + ' channels; images have 1 or 3');
    end;
    WriteLn('Task ', Opt.Task, ' (', OfficialTask, '), checkpoint ', WeightsFile);
    StartTime := Now();
    Restored := RestoreImage(WeightsFile, Config, NetInput, Opt, ComputeText);
    WriteLn(Format('  %.1f s', [(Now() - StartTime) * 86400.0]));
    SaveUnitRangeImage(Restored, Opt.OutputFile);
    WriteLn(Format('  %dx%d -> %s', [Restored.SizeX, Restored.SizeY,
      Opt.OutputFile]));
  finally
    Restored.Free;
    NetInput.Free;
    RightImage.Free;
    Image.Free;
  end;
end;

// Pico smoke: a synthetic 37x29 RGB image (and a left/right pair shifted by
// one pixel for the dual-pixel net) through the fixture nets in 16x16 tiles
// (37x29 also exercises the reflect padding to multiples of 8).
procedure RunFixtureSmoke(Opt: TRestoreOptions);
var
  Config: TRestormerConfig;
  Image, RightImage, NetInput, Restored: TNNetVolume;
  X, Y, Channel: integer;
  ComputeText: string;
begin
  WriteLn('No --task given: smoke run of the random pico fixtures ',
    '(not trained models).');
  Opt.Tile := 16;
  Opt.TilePad := 8;
  Opt.TileOverlap := 8;
  Image := TNNetVolume.Create(37, 29, 3);
  RightImage := TNNetVolume.Create(37, 29, 3);
  NetInput := TNNetVolume.Create;
  Restored := nil;
  try
    for Y := 0 to Image.SizeY - 1 do
      for X := 0 to Image.SizeX - 1 do
        for Channel := 0 to 2 do
        begin
          Image[X, Y, Channel] := 0.5 + 0.4 * Sin(X * 0.3 + Channel) *
            Cos(Y * 0.2);
          RightImage[X, Y, Channel] := 0.5 + 0.4 *
            Sin((X + 1) * 0.3 + Channel) * Cos(Y * 0.2);
        end;
    Config := ReadRestormerConfigFromJSONFile(cFixturesDir +
      'tiny_restormer_config.json');
    Restored := RestoreImage(cFixturesDir + 'tiny_restormer.safetensors',
      Config, Image, Opt, ComputeText);
    SaveUnitRangeImage(Restored, cSmokeOutput);
    WriteLn(Format('  %dx%dx%d -> %s', [Restored.SizeX, Restored.SizeY,
      Restored.Depth, cSmokeOutput]));
    FreeAndNil(Restored);
    Config := ReadRestormerConfigFromJSONFile(cFixturesDir +
      'tiny_restormer_dp_config.json');
    RestormerPackDualPixel(Config, Image, RightImage, NetInput);
    Restored := RestoreImage(cFixturesDir + 'tiny_restormer_dp.pth',
      Config, NetInput, Opt, ComputeText);
    SaveUnitRangeImage(Restored, cDualPixelSmokeOutput);
    WriteLn(Format('  dual-pixel %dx%dx%d -> %s', [Restored.SizeX,
      Restored.SizeY, Restored.Depth, cDualPixelSmokeOutput]));
  finally
    Restored.Free;
    NetInput.Free;
    RightImage.Free;
    Image.Free;
  end;
end;

var
  Opt: TRestoreOptions;
begin
  try
    if not ParseOptions(Opt) then Halt(0);
    if Opt.Task = '' then RunFixtureSmoke(Opt)
    else RestoreFile(Opt);
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      Halt(1);
    end;
  end;
end.
