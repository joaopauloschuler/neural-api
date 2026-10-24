program FaceRestoration;
(*
FaceRestoration: blind face restoration with GFPGAN v1.3 / v1.4 (Wang et al.,
CVPR 2021) or CodeFormer (Zhou et al., NeurIPS 2022; S-Lab License 1.0,
non-commercial) and the importers in neuralpretrained.pas. --model picks the
restorer from the checkpoint's tensors.

Modes:
  * no arguments: smokes of the committed random pico fixtures
    (tests/fixtures/tiny_gfpgan, tiny_codeformer, tiny_retinaface_mobilenet):
    per restorer an aligned crop, then a whole synthetic photo.
  * --model FILE --detector FILE --input IMG --output IMG: whole-photo
    restoration as inference_gfpgan.py / inference_codeformer.py: RetinaFace
    finds every face, each is aligned to the FFHQ 5-point template, restored and pasted back
    (RestoreFacesInImage, a port of facexlib's FaceRestoreHelper) on the
    background upscaled by --upscale (Lanczos, or an RRDBNet / Real-ESRGAN
    checkpoint with --bg-upsampler); --parse parsing_parsenet.pth pastes
    through facexlib's face-parsing mask, as inference_gfpgan.py does.
  * --aligned --model FILE --input IMG --output IMG: the input is already an
    aligned, cropped face (resized to 512), as GFPGAN's own --aligned.

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
  {$IFDEF OpenCL}neuralopencl, cl,{$ENDIF}
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained,
  neuralimageresize;

const
  // Fixtures committed with the GFPGAN / CodeFormer importer tests.
  FixturesDir = '../../tests/fixtures/';

type
  TFaceRestorationOptions = record
    ModelFile, ConfigFile, InputFile, OutputFile: string;
    DetectorFile, BackgroundFile, SaveFacesPrefix, ParseFile: string;
    NoiseMode: TStyleGAN2NoiseMode;
    Seed, BackgroundTile, MaxFaces: integer;
    Upscale, DetectionThreshold, MinEyeDistance: TNeuralFloat;
    // CodeFormer's w; negative = the config's fidelity_weight (default 0.5).
    Fidelity: TNeuralFloat;
    IsAligned, KeepSize, UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: FaceRestoration                       (pico fixture smokes)');
  WriteLn('       FaceRestoration --model FILE --detector FILE --input IMG --output IMG [options]');
  WriteLn('       FaceRestoration --aligned --model FILE --input IMG --output IMG [options]');
  WriteLn('  --model FILE         GFPGAN (GFPGANv1.4.pth) or CodeFormer (codeformer.pth)');
  WriteLn('                       checkpoint, told apart by its tensors');
  WriteLn('  --config FILE        config.json (default: next to --model, else the');
  WriteLn('                       release settings; shapes come from the tensors)');
  WriteLn('  --fidelity W         CodeFormer fidelity weight, 0 = quality .. 1 = identity');
  WriteLn('                       (default 0.5, as inference_codeformer.py)');
  WriteLn('  --input IMG          photo (or, with --aligned, an aligned face crop)');
  WriteLn('  --output IMG         restored photo / face (PNG)');
  WriteLn('  --aligned            the input is an ALIGNED, cropped face: no detection');
  WriteLn('  --keep-size          (--aligned) resize the restored face back to the');
  WriteLn('                       input size (default: the net resolution, 512)');
  WriteLn('  --detector FILE      RetinaFace checkpoint: facexlib detection_Resnet50_Final.pth');
  WriteLn('                       (GFPGAN''s default) or detection_mobilenet0.25_Final.pth');
  WriteLn('  --parse FILE         facexlib parsing_parsenet.pth: paste through its face-parsing');
  WriteLn('                       mask, as inference_gfpgan.py (default: square mask)');
  WriteLn('  --upscale N          output size / input size (default 2, as inference_gfpgan.py)');
  WriteLn('  --bg-upsampler FILE  RRDBNet / Real-ESRGAN checkpoint for the background');
  WriteLn('                       (e.g. RealESRGAN_x2plus.pth; default: Lanczos resize)');
  WriteLn('  --bg-tile N          background upsampler tile size (default 400, 0 = whole image)');
  WriteLn('  --det-threshold T    RetinaFace score threshold (default 0.97, facexlib)');
  WriteLn('  --min-eye-distance P skip faces whose eyes are closer than P pixels (default 5)');
  WriteLn('  --max-faces N        restore at most the N best-scoring faces (default all)');
  WriteLn('  --save-faces PREFIX  write PREFIX_cropped_NN.png and PREFIX_restored_NN.png');
  WriteLn('  --noise MODE         GFPGAN only: random (its default), const (the ',
    'checkpoint''s');
  WriteLn('                       stored maps) or none (default random)');
  WriteLn('  --seed N             seed of the random noise (default: time based)');
  WriteLn('  --int8               store the convolution weights in int8 (less RAM)');
  WriteLn('  --gpu                run the nets on OpenCL (needs a build with ',
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

  function NextFloat(): TNeuralFloat;
  var
    Settings: TFormatSettings;
  begin
    Settings := DefaultFormatSettings;
    Settings.DecimalSeparator := '.';
    Result := StrToFloat(NextArg(), Settings);
  end;

begin
  Opt.ModelFile := '';
  Opt.ConfigFile := '';
  Opt.InputFile := '';
  Opt.OutputFile := '';
  Opt.DetectorFile := '';
  Opt.BackgroundFile := '';
  Opt.SaveFacesPrefix := '';
  Opt.ParseFile := '';
  Opt.NoiseMode := snmRandom;
  Opt.Seed := -1;
  Opt.BackgroundTile := 400;
  Opt.MaxFaces := 0;
  Opt.Upscale := 2;
  Opt.DetectionThreshold := 0.97;
  Opt.MinEyeDistance := 5;
  Opt.Fidelity := -1;
  Opt.IsAligned := false;
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
    else if Arg = '--aligned' then Opt.IsAligned := true
    else if Arg = '--keep-size' then Opt.KeepSize := true
    else if Arg = '--detector' then Opt.DetectorFile := NextArg()
    else if Arg = '--parse' then Opt.ParseFile := NextArg()
    else if Arg = '--upscale' then Opt.Upscale := NextFloat()
    else if Arg = '--bg-upsampler' then Opt.BackgroundFile := NextArg()
    else if Arg = '--bg-tile' then Opt.BackgroundTile := StrToInt(NextArg())
    else if Arg = '--det-threshold' then Opt.DetectionThreshold := NextFloat()
    else if Arg = '--min-eye-distance' then Opt.MinEyeDistance := NextFloat()
    else if Arg = '--max-faces' then Opt.MaxFaces := StrToInt(NextArg())
    else if Arg = '--save-faces' then Opt.SaveFacesPrefix := NextArg()
    else if Arg = '--fidelity' then Opt.Fidelity := NextFloat()
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
  if Opt.ModelFile = '' then exit;
  if (Opt.InputFile = '') or (Opt.OutputFile = '') then
    raise Exception.Create('--model needs --input and --output');
  if not Opt.IsAligned and (Opt.DetectorFile = '') then
    raise Exception.Create('a whole photo needs --detector FILE ' +
      '(detection_Resnet50_Final.pth); pass --aligned for an aligned face crop');
  if Opt.Upscale <= 0 then
    raise Exception.Create('--upscale must be positive');
  if Opt.Fidelity > 1 then
    raise Exception.Create('--fidelity must be in [0, 1]');
end;

type
  // The OpenCL device the nets run on, or the reason they run on the CPU.
  TComputeDevice = record
    {$IFDEF OpenCL}
    Devices: TEasyOpenCL;
    PlatformId: cl_platform_id;
    DeviceId: cl_device_id;
    {$ENDIF}
    IsOpenCL: boolean;
    Description: string;
  end;

procedure SelectComputeDevice(const Opt: TFaceRestorationOptions;
  out Device: TComputeDevice);
{$IFDEF OpenCL}
var
  Problem: string;
  PlatformPos, DevicePos: integer;
{$ENDIF}
begin
  Device.IsOpenCL := false;
  Device.Description := 'CPU';
  {$IFDEF OpenCL}
  Device.Devices := nil;
  {$ENDIF}
  if not Opt.UseOpenCL then exit;
  {$IFDEF OpenCL}
  Device.Devices := TEasyOpenCL.Create();
  PlatformPos := Opt.OpenCLPlatform;
  DevicePos := Opt.OpenCLDevice;
  if Device.Devices.SelectPlatformAndDevice(PlatformPos, DevicePos, Problem) then
  begin
    Device.PlatformId := Device.Devices.PlatformIds[PlatformPos];
    Device.DeviceId := Device.Devices.Devices[DevicePos];
    Device.IsOpenCL := true;
    Device.Description := 'OpenCL on ' + Device.Devices.PlatformNames[PlatformPos] +
      ' / ' + Device.Devices.DeviceNames[DevicePos];
  end
  else
    Device.Description := 'CPU (' + Problem + ')';
  {$ELSE}
  Device.Description := 'CPU (built without -dOpenCL)';
  {$ENDIF}
end;

procedure EnableDevice(NN: TNNet; const Device: TComputeDevice);
begin
  {$IFDEF OpenCL}
  if Device.IsOpenCL then NN.EnableOpenCL(Device.PlatformId, Device.DeviceId);
  {$ENDIF}
end;

// Builds the GFPGAN or CodeFormer net of Opt.ModelFile; FaceSize := its
// resolution, the only thing RestoreFacesInImage needs to know about it.
function LoadRestorer(const Opt: TFaceRestorationOptions;
  const Device: TComputeDevice; out FaceSize: integer): TNNet;
var
  GFPGANConfig: TGFPGANConfig;
  CodeFormerConfig: TCodeFormerConfig;
  StartTime: TDateTime;
begin
  StartTime := Now();
  if IsCodeFormerCheckpoint(Opt.ModelFile) then
  begin
    Result := BuildCodeFormerFromSafeTensors(Opt.ModelFile, CodeFormerConfig,
      {pTrainable=}false, Opt.ConfigFile);
    if Opt.Fidelity >= 0 then
    begin
      CodeFormerConfig.Fidelity := Opt.Fidelity;
      CodeFormerSetFidelity(Result, Opt.Fidelity);
    end;
    WriteLn(CodeFormerConfigToString(CodeFormerConfig));
    FaceSize := CodeFormerConfig.ImageSize;
  end
  else
  begin
    Result := BuildGFPGANFromSafeTensors(Opt.ModelFile, GFPGANConfig,
      {pTrainable=}false, Opt.ConfigFile, Opt.NoiseMode);
    WriteLn(GFPGANConfigToString(GFPGANConfig));
    FaceSize := GFPGANConfig.OutSize;
  end;
  WriteLn(Format('  loaded in %.1f s', [(Now() - StartTime) * 86400.0]));
  // Quantize before EnableOpenCL: the OpenCL arming reads the int8 codes.
  if Opt.UseInt8 then Result.QuantizeWeightsInt8();
  EnableDevice(Result, Device);
end;

function ComputeText(const Opt: TFaceRestorationOptions;
  const Device: TComputeDevice): string;
begin
  Result := Device.Description;
  if Opt.UseInt8 then Result := Result + ', int8 weights';
end;

// --aligned: restores the face crop in InputFile and writes OutputFile.
procedure RestoreAlignedFile(const Opt: TFaceRestorationOptions;
  const Device: TComputeDevice);
var
  NN: TNNet;
  FaceSize: integer;
  Face, Restored, Resized: TNNetVolume;
  StartTime: TDateTime;
begin
  Face := TNNetVolume.Create;
  Restored := TNNetVolume.Create;
  Resized := TNNetVolume.Create;
  NN := nil;
  try
    if not LoadImageFromFileIntoVolume(Opt.InputFile, Face) then
      raise Exception.Create('could not load ' + Opt.InputFile);
    NN := LoadRestorer(Opt, Device, FaceSize);
    if Opt.Seed >= 0 then RestartRandSeed(Opt.Seed);
    WriteLn(Format('Restoring a %dx%d face on %s, net resolution %dx%d',
      [Face.SizeX, Face.SizeY, ComputeText(Opt, Device), FaceSize, FaceSize]));
    StartTime := Now();
    FaceRestorerRestoreFace(NN, FaceSize, Face, Restored);
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
    Resized.Free;
    Restored.Free;
    Face.Free;
  end;
end;

// Background for the paste: the photo upscaled by the RRDBNet checkpoint, in 0..255.
function UpscaleBackground(const Opt: TFaceRestorationOptions;
  const Device: TComputeDevice; Image: TNNetVolume): TNNetVolume;
var
  Upsampler: TRRDBNetUpscaler;
  StartTime: TDateTime;
begin
  Result := TNNetVolume.Create;
  Upsampler := TRRDBNetUpscaler.Create(Opt.BackgroundFile);
  try
    Upsampler.Tile := Opt.BackgroundTile;
    {$IFDEF OpenCL}
    if Device.IsOpenCL then
      Upsampler.EnableOpenCL(Device.PlatformId, Device.DeviceId);
    {$ENDIF}
    WriteLn(Format('Upscaling the background x%d with %s', [Upsampler.Config.Scale,
      Opt.BackgroundFile]));
    StartTime := Now();
    Result.Copy(Image);
    Result.Mul(1 / 255);
    Upsampler.Upscale(Result, Result);
    Result.Mul(255);
    WriteLn(Format('  %dx%d in %.1f s', [Result.SizeX, Result.SizeY,
      (Now() - StartTime) * 86400.0]));
  finally
    Upsampler.Free;
  end;
end;

procedure SaveFaces(const Prefix, Kind: string; Faces: TNNetVolumeList);
var
  FacePos: integer;
  FileName: string;
begin
  for FacePos := 0 to Faces.Count - 1 do
  begin
    FileName := Format('%s_%s_%.2d.png', [Prefix, Kind, FacePos]);
    if not SaveImageFromVolumeIntoFile(Faces[FacePos], FileName) then
      raise Exception.Create('could not save ' + FileName);
  end;
end;

// Whole photo: detect, align, restore and paste every face (RestoreFacesInImage).
procedure RestorePhotoFile(const Opt: TFaceRestorationOptions;
  const Device: TComputeDevice);
var
  Restorer, Detector, Parser: TNNet;
  RestorerFaceSize: integer;
  ParserConfig: TParseNetConfig;
  DetectorConfig: TRetinaFaceConfig;
  Image, Background, Output: TNNetVolume;
  Crops, RestoredFaces: TNNetVolumeList;
  FaceCount: integer;
  StartTime: TDateTime;
begin
  Image := TNNetVolume.Create;
  Output := TNNetVolume.Create;
  Crops := TNNetVolumeList.Create(true);
  RestoredFaces := TNNetVolumeList.Create(true);
  Background := nil;
  Restorer := nil;
  Detector := nil;
  Parser := nil;
  try
    if not LoadImageFromFileIntoVolume(Opt.InputFile, Image) then
      raise Exception.Create('could not load ' + Opt.InputFile);
    if Image.Depth > 3 then Image.ReSize(Image.SizeX, Image.SizeY, 3);
    StartTime := Now();
    // facexlib runs the detector at the photo's own size.
    Detector := BuildRetinaFaceFromSafeTensors(Opt.DetectorFile, Image.SizeX,
      Image.SizeY, DetectorConfig, {pTrainable=}false);
    WriteLn(RetinaFaceConfigToString(DetectorConfig));
    WriteLn(Format('  loaded in %.1f s', [(Now() - StartTime) * 86400.0]));
    EnableDevice(Detector, Device);
    Restorer := LoadRestorer(Opt, Device, RestorerFaceSize);
    if Opt.ParseFile <> '' then
    begin
      // facexlib builds ParseNet(in_size=512) for every restorer.
      Parser := BuildParseNetFromSafeTensors(Opt.ParseFile, ParserConfig, 512,
        {pTrainable=}false);
      WriteLn(ParseNetConfigToString(ParserConfig));
      EnableDevice(Parser, Device);
    end;
    if Opt.BackgroundFile <> '' then
      Background := UpscaleBackground(Opt, Device, Image);
    if Opt.Seed >= 0 then RestartRandSeed(Opt.Seed);
    WriteLn(Format('Restoring the faces of a %dx%d photo on %s, upscale %g',
      [Image.SizeX, Image.SizeY, ComputeText(Opt, Device), Opt.Upscale]));
    StartTime := Now();
    FaceCount := RestoreFacesInImage(Detector, DetectorConfig, Restorer,
      RestorerFaceSize, Image, Background, Output, Opt.Upscale,
      Opt.DetectionThreshold, Opt.MinEyeDistance, Opt.MaxFaces, Crops,
      RestoredFaces, Parser);
    WriteLn(Format('  %d face(s) in %.1f s', [FaceCount,
      (Now() - StartTime) * 86400.0]));
    if Opt.SaveFacesPrefix <> '' then
    begin
      SaveFaces(Opt.SaveFacesPrefix, 'cropped', Crops);
      SaveFaces(Opt.SaveFacesPrefix, 'restored', RestoredFaces);
    end;
    if not SaveImageFromVolumeIntoFile(Output, Opt.OutputFile) then
      raise Exception.Create('could not save ' + Opt.OutputFile);
    WriteLn(Format('  %dx%d -> %s', [Output.SizeX, Output.SizeY,
      Opt.OutputFile]));
  finally
    Parser.Free;
    Detector.Free;
    Restorer.Free;
    Background.Free;
    RestoredFaces.Free;
    Crops.Free;
    Output.Free;
    Image.Free;
  end;
end;

procedure RunOptions(const Opt: TFaceRestorationOptions);
var
  Device: TComputeDevice;
begin
  SelectComputeDevice(Opt, Device);
  try
    if Opt.IsAligned then RestoreAlignedFile(Opt, Device)
    else RestorePhotoFile(Opt, Device);
  finally
    {$IFDEF OpenCL}
    Device.Devices.Free;
    {$ENDIF}
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

// Pico smokes of one restorer: a 40x30 crop through --aligned, then a 256x192
// photo through the whole-photo path, again with ParseFile unless it is ''.
procedure RunFixtureSmoke(const ModelFile, ConfigFile, OutPrefix,
  ParseFile: string);
const
  InPng = 'face_in.png';
  PhotoPng = 'face_photo_in.png';
var
  Img: TNNetVolume;
  Opt: TFaceRestorationOptions;
  X, Y, Channel: integer;
begin
  Img := TNNetVolume.Create;
  try
    MakeTinyImage(Img, 40, 30);
    if not SaveImageFromVolumeIntoFile(Img, InPng) then
      raise Exception.Create('could not save ' + InPng);
    Img.ReSize(256, 192, 3);
    for Y := 0 to 191 do
      for X := 0 to 255 do
        for Channel := 0 to 2 do
          Img[X, Y, Channel] := Round(128 + 90 * Sin(X * 0.15 + Channel) *
            Cos(Y * 0.125));
    if not SaveImageFromVolumeIntoFile(Img, PhotoPng) then
      raise Exception.Create('could not save ' + PhotoPng);
  finally
    Img.Free;
  end;
  ParseOptions(Opt);
  Opt.ModelFile := ModelFile;
  Opt.ConfigFile := ConfigFile;
  Opt.InputFile := InPng;
  Opt.OutputFile := OutPrefix + '_out.png';
  Opt.NoiseMode := snmConst;
  Opt.IsAligned := true;
  RunOptions(Opt);
  Opt.IsAligned := false;
  Opt.DetectorFile := FixturesDir + 'tiny_retinaface_mobilenet.pth';
  Opt.InputFile := PhotoPng;
  Opt.OutputFile := OutPrefix + '_photo_out.png';
  Opt.DetectionThreshold := 0.5;
  Opt.MaxFaces := 3;
  RunOptions(Opt);
  if ParseFile = '' then exit;
  Opt.ParseFile := ParseFile;
  Opt.OutputFile := OutPrefix + '_parse_photo_out.png';
  RunOptions(Opt);
end;

var
  Options: TFaceRestorationOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFile <> '' then
      RunOptions(Options)
    else
    begin
      WriteLn('GFPGAN face restoration smoke (random pico weights)');
      WriteLn('===================================================');
      RunFixtureSmoke(FixturesDir + 'tiny_gfpgan.pth',
        FixturesDir + 'tiny_gfpgan_config.json', 'gfpgan',
        FixturesDir + 'tiny_parsenet.pth');
      WriteLn('CodeFormer face restoration smoke (random pico weights)');
      WriteLn('=======================================================');
      RunFixtureSmoke(FixturesDir + 'tiny_codeformer.pth',
        FixturesDir + 'tiny_codeformer_config.json', 'codeformer', '');
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
