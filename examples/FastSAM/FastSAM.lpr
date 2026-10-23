program FastSAM;
(*
FastSAM: real-time "segment anything" with a YOLOv8 instance-segmentation net
(Zhao et al. 2023, arXiv:2306.12156; ultralytics FastSAM-s / FastSAM-x), with
the importer in neuralpretrained.pas (BuildYoloFromSafeTensors with
task "segment", DecodeYoloSegmentation, FastSAMSelectByBox / ByPoint).

Two modes:
  * no arguments: a smoke of the committed random pico fixture
    tests/fixtures/tiny_yolo_seg: synthesize a small colour image, segment
    everything, then run one box prompt and one point prompt.
  * --model FILE --image IMG [--box X1,Y1,X2,Y2] [--point X,Y,LABEL ...]:
    segment a real image with a real checkpoint (FastSAM-s.pt / FastSAM-x.pt
    converted to safetensors, see README.md). Without a prompt every mask is
    drawn; with a box or points only the selected mask is drawn.

The image is letterboxed to the net input (ultralytics LetterBox: keep the
aspect ratio, pad with gray 114), and the masks are mapped back to the image.

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
  FixturesDir = '../../tests/fixtures/';
  PaletteR: array[0..5] of byte = (255, 0, 0, 255, 255, 0);
  PaletteG: array[0..5] of byte = (0, 255, 0, 255, 0, 255);
  PaletteB: array[0..5] of byte = (0, 0, 255, 0, 255, 255);

type
  TFastSAMOptions = record
    ModelFile, ConfigFile, ImageFile, OutputFile: string;
    ScoreThreshold, IoUThreshold: TNeuralFloat;
    HasBox: boolean;
    Box: array[0..3] of integer;
    PointX, PointY, PointLabels: array of integer;
    UseOpenCL: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

  // Where the image sits inside the letterboxed net input.
  TLetterbox = record
    Scale: TNeuralFloat;
    Left, Top, ContentX, ContentY: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: FastSAM                                 (pico fixture smoke)');
  WriteLn('       FastSAM --model FILE --image IMG [options]');
  WriteLn('  --model FILE         FastSAM / yolov8-seg model.safetensors');
  WriteLn('  --config FILE        config.json (default: next to the model)');
  WriteLn('  --image IMG          image (PNG / JPEG / BMP)');
  WriteLn('  --output PNG         mask overlay (default fastsam.png)');
  WriteLn('  --conf X             score threshold (default 0.4, FastSAM)');
  WriteLn('  --iou X              NMS IoU threshold (default 0.9, FastSAM)');
  WriteLn('  --box X1,Y1,X2,Y2    box prompt in image pixels');
  WriteLn('  --point X,Y,LABEL    point prompt in image pixels, LABEL 1 = keep,');
  WriteLn('                       0 = remove; repeat for several points');
  WriteLn('  --gpu                run on OpenCL (build with -dOpenCL)');
  WriteLn('  --gpu-platform N / --gpu-device N   select the OpenCL device');
end;

procedure SplitIntegers(const Text: string; var Values: array of integer);
var
  Parts: TStringList;
  PartPos, MaxPartPos: integer;
begin
  Parts := TStringList.Create;
  try
    Parts.Delimiter := ',';
    Parts.StrictDelimiter := true;
    Parts.DelimitedText := Text;
    if Parts.Count <> Length(Values) then
      raise Exception.Create('expected ' + IntToStr(Length(Values)) +
        ' comma-separated integers, got "' + Text + '"');
    MaxPartPos := Parts.Count - 1;
    for PartPos := 0 to MaxPartPos do
      Values[PartPos] := StrToInt(Trim(Parts[PartPos]));
  finally
    Parts.Free;
  end;
end;

function ParseOptions(out Opt: TFastSAMOptions): boolean;
var
  ArgPos, PointCnt: integer;
  Arg: string;
  PointValues: array[0..2] of integer;

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
  Opt.ImageFile := '';
  Opt.OutputFile := 'fastsam.png';
  Opt.ScoreThreshold := 0.4;
  Opt.IoUThreshold := 0.9;
  Opt.HasBox := false;
  SetLength(Opt.PointX, 0);
  SetLength(Opt.PointY, 0);
  SetLength(Opt.PointLabels, 0);
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
    else if Arg = '--image' then Opt.ImageFile := NextArg()
    else if Arg = '--output' then Opt.OutputFile := NextArg()
    else if Arg = '--conf' then Opt.ScoreThreshold := StrToFloat(NextArg())
    else if Arg = '--iou' then Opt.IoUThreshold := StrToFloat(NextArg())
    else if Arg = '--box' then
    begin
      SplitIntegers(NextArg(), Opt.Box);
      Opt.HasBox := true;
    end
    else if Arg = '--point' then
    begin
      SplitIntegers(NextArg(), PointValues);
      PointCnt := Length(Opt.PointX) + 1;
      SetLength(Opt.PointX, PointCnt);
      SetLength(Opt.PointY, PointCnt);
      SetLength(Opt.PointLabels, PointCnt);
      Opt.PointX[PointCnt - 1] := PointValues[0];
      Opt.PointY[PointCnt - 1] := PointValues[1];
      Opt.PointLabels[PointCnt - 1] := PointValues[2];
    end
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
  if (Opt.ModelFile <> '') and (Opt.ImageFile = '') then
    raise Exception.Create('--model needs --image');
end;

// NetInput (Size, Size, 3) in [0,1] := ultralytics LetterBox of Image (0..255).
procedure LetterboxImage(Image, NetInput: TNNetVolume; Size: integer;
  out Box: TLetterbox);
var
  Resized: TNNetVolume;
  X, Y, Channel, MaxX, MaxY: integer;
begin
  Box.Scale := Min(Size / Image.SizeX, Size / Image.SizeY);
  Box.ContentX := Max(1, Round(Image.SizeX * Box.Scale));
  Box.ContentY := Max(1, Round(Image.SizeY * Box.Scale));
  Box.Left := Round((Size - Box.ContentX) / 2 - 0.1);
  Box.Top := Round((Size - Box.ContentY) / 2 - 0.1);
  Resized := TNNetVolume.Create;
  try
    ResizeImageBilinearNoAntialias(Image, Resized, Box.ContentX, Box.ContentY);
    NetInput.ReSize(Size, Size, 3);
    NetInput.Fill(114);
    MaxX := Box.ContentX - 1;
    MaxY := Box.ContentY - 1;
    for Y := 0 to MaxY do
      for X := 0 to MaxX do
        for Channel := 0 to 2 do
          NetInput[Box.Left + X, Box.Top + Y, Channel] := Resized[X, Y, Channel];
    NetInput.Mul(1 / 255);
  finally
    Resized.Free;
  end;
end;

// ImageMasks (image W, H, N) := the letterbox content of Masks, resized, > 0.5.
procedure MasksToImage(Masks, ImageMasks: TNNetVolume; const Box: TLetterbox;
  ImageWidth, ImageHeight: integer);
var
  Content: TNNetVolume;
  ValuePos, MaxValuePos: integer;
begin
  Content := TNNetVolume.Create;
  try
    Content.CopyCropping(Masks, Box.Left, Box.Top, Box.ContentX, Box.ContentY);
    ResizeImageBilinearNoAntialias(Content, ImageMasks, ImageWidth, ImageHeight);
    MaxValuePos := ImageMasks.Size - 1;
    for ValuePos := 0 to MaxValuePos do
      if ImageMasks.FData[ValuePos] > 0.5 then ImageMasks.FData[ValuePos] := 1
      else ImageMasks.FData[ValuePos] := 0;
  finally
    Content.Free;
  end;
end;

// Blends each mask channel of ImageMasks into Image (0..255) in a palette colour.
procedure DrawMasks(Image, ImageMasks: TNNetVolume; const FileName: string);
var
  Overlay: TNNetVolume;
  X, Y, MaskPos, MaxX, MaxY, MaxMaskPos, ColorPos: integer;
begin
  Overlay := TNNetVolume.Create;
  try
    Overlay.Copy(Image);
    MaxX := Image.SizeX - 1;
    MaxY := Image.SizeY - 1;
    MaxMaskPos := ImageMasks.Depth - 1;
    for MaskPos := 0 to MaxMaskPos do
    begin
      ColorPos := MaskPos mod 6;
      for Y := 0 to MaxY do
        for X := 0 to MaxX do
          if ImageMasks[X, Y, MaskPos] > 0.5 then
          begin
            Overlay[X, Y, 0] := 0.5 * Overlay[X, Y, 0] + 0.5 * PaletteR[ColorPos];
            Overlay[X, Y, 1] := 0.5 * Overlay[X, Y, 1] + 0.5 * PaletteG[ColorPos];
            Overlay[X, Y, 2] := 0.5 * Overlay[X, Y, 2] + 0.5 * PaletteB[ColorPos];
          end;
    end;
    if not SaveImageFromVolumeIntoFile(Overlay, FileName) then
      raise Exception.Create('could not save ' + FileName);
    WriteLn(Format('  %d mask(s) drawn -> %s', [ImageMasks.Depth, FileName]));
  finally
    Overlay.Free;
  end;
end;

// Runs the net on Image, applies the prompts of Opt and writes the overlay.
procedure SegmentImage(NN: TNNet; const Config: TYoloConfig; Image: TNNetVolume;
  const Opt: TFastSAMOptions);
var
  NetInput, Masks, Selected, ImageMasks: TNNetVolume;
  Dets: TYoloDetectionArray;
  Box: TLetterbox;
  NetPointX, NetPointY: array of integer;
  PointPos, MaxPointPos, MaskIdx: integer;
  StartTime: TDateTime;
begin
  NetInput := TNNetVolume.Create;
  Masks := TNNetVolume.Create;
  Selected := TNNetVolume.Create;
  ImageMasks := TNNetVolume.Create;
  try
    LetterboxImage(Image, NetInput, Config.ImageSize, Box);
    StartTime := Now();
    NN.Compute(NetInput);
    Dets := DecodeYoloSegmentation(NN, Config, Masks, Opt.ScoreThreshold,
      Opt.IoUThreshold, {IsFastSAM=}Config.ModelType = 'fastsam');
    WriteLn(Format('  %dx%d image -> %d mask(s) in %.2f s', [Image.SizeX,
      Image.SizeY, Length(Dets), (Now() - StartTime) * 86400.0]));
    if Opt.HasBox then
    begin
      MaskIdx := FastSAMSelectByBox(Masks,
        Round(Opt.Box[0] * Box.Scale) + Box.Left,
        Round(Opt.Box[1] * Box.Scale) + Box.Top,
        Round(Opt.Box[2] * Box.Scale) + Box.Left,
        Round(Opt.Box[3] * Box.Scale) + Box.Top);
      WriteLn(Format('  box prompt (%d,%d,%d,%d) selects mask %d',
        [Opt.Box[0], Opt.Box[1], Opt.Box[2], Opt.Box[3], MaskIdx]));
      Selected.ReSize(Masks.SizeX, Masks.SizeY, 1);
      if MaskIdx >= 0 then Selected.CopyChannels(Masks, [MaskIdx])
      else Selected.Fill(0);
      Masks.Copy(Selected);
    end
    else if Length(Opt.PointX) > 0 then
    begin
      MaxPointPos := High(Opt.PointX);
      SetLength(NetPointX, MaxPointPos + 1);
      SetLength(NetPointY, MaxPointPos + 1);
      for PointPos := 0 to MaxPointPos do
      begin
        NetPointX[PointPos] := Round(Opt.PointX[PointPos] * Box.Scale) + Box.Left;
        NetPointY[PointPos] := Round(Opt.PointY[PointPos] * Box.Scale) + Box.Top;
      end;
      FastSAMSelectByPoint(Masks, NetPointX, NetPointY, Opt.PointLabels, Selected);
      WriteLn(Format('  %d point prompt(s) select %d net-input pixels',
        [MaxPointPos + 1, Round(Selected.GetSum())]));
      Masks.Copy(Selected);
    end;
    if Masks.Depth > 0 then
      MasksToImage(Masks, ImageMasks, Box, Image.SizeX, Image.SizeY)
    else
      ImageMasks.ReSize(Image.SizeX, Image.SizeY, 0);
    DrawMasks(Image, ImageMasks, Opt.OutputFile);
  finally
    ImageMasks.Free;
    Selected.Free;
    Masks.Free;
    NetInput.Free;
  end;
end;

// Applies --gpu to NN; returns where it runs, for the log.
function PrepareCompute(NN: TNNet; const Opt: TFastSAMOptions
  {$IFDEF OpenCL}; var OpenCLDevices: TEasyOpenCL{$ENDIF}): string;
{$IFDEF OpenCL}
var
  OpenCLProblem: string;
  OpenCLPlatform, OpenCLDevice: integer;
{$ENDIF}
begin
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
end;

// Deterministic RGB gradient (0..255) so the smoke needs no image on disk.
procedure MakeTinyImage(Image: TNNetVolume; SizeX, SizeY: integer);
var
  X, Y, MaxX, MaxY: integer;
begin
  Image.ReSize(SizeX, SizeY, 3);
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      Image[X, Y, 0] := 255 * X / MaxX;
      Image[X, Y, 1] := 255 * Y / MaxY;
      Image[X, Y, 2] := 127.5 + 127.5 * Sin((X + Y) * 0.15);
    end;
end;

procedure Run(var Opt: TFastSAMOptions);
var
  NN: TNNet;
  Config: TYoloConfig;
  Image: TNNetVolume;
  ComputeText: string;
  IsSmoke: boolean;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  {$ENDIF}
begin
  IsSmoke := Opt.ModelFile = '';
  if IsSmoke then
  begin
    Opt.ModelFile := FixturesDir + 'tiny_yolo_seg.safetensors';
    Opt.ConfigFile := FixturesDir + 'tiny_yolo_seg_config.json';
    // the random pico net scores near 0.5; keep a few masks
    Opt.ScoreThreshold := 0.505;
    Opt.IoUThreshold := 0.5;
  end;
  Image := TNNetVolume.Create;
  NN := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    NN := BuildYoloFromSafeTensors(Opt.ModelFile, Config, {pTrainable=}false,
      Opt.ConfigFile);
    WriteLn(YoloConfigToString(Config));
    if Config.NumMasks <= 0 then
      raise Exception.Create('the config has no "task": "segment" head');
    ComputeText := PrepareCompute(NN, Opt
      {$IFDEF OpenCL}, OpenCLDevices{$ENDIF});
    WriteLn('Running on ', ComputeText);
    if IsSmoke then
    begin
      MakeTinyImage(Image, 120, 80);
      WriteLn('Pico smoke: segment everything');
      Opt.OutputFile := 'fastsam_everything.png';
      SegmentImage(NN, Config, Image, Opt);
      WriteLn('Pico smoke: box prompt');
      Opt.HasBox := true;
      Opt.Box[0] := 10; Opt.Box[1] := 10; Opt.Box[2] := 70; Opt.Box[3] := 60;
      Opt.OutputFile := 'fastsam_box.png';
      SegmentImage(NN, Config, Image, Opt);
      WriteLn('Pico smoke: point prompt');
      Opt.HasBox := false;
      Opt.PointX := [60, 100];
      Opt.PointY := [40, 20];
      Opt.PointLabels := [1, 0];
      Opt.OutputFile := 'fastsam_point.png';
      SegmentImage(NN, Config, Image, Opt);
      WriteLn('(Random pico weights: the masks are not meaningful objects.)');
    end
    else
    begin
      if not LoadImageFromFileIntoVolume(Opt.ImageFile, Image) then
        raise Exception.Create('could not load ' + Opt.ImageFile);
      SegmentImage(NN, Config, Image, Opt);
    end;
  finally
    NN.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    Image.Free;
  end;
end;

var
  Opt: TFastSAMOptions;
begin
  try
    if ParseOptions(Opt) then Run(Opt);
  except
    on E: Exception do
    begin
      WriteLn('ERROR: ', E.Message);
      Halt(1);
    end;
  end;
end.
