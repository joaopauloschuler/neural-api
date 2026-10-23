program ObjectTracking;
(*
ObjectTracking: multi-object tracking over a folder of video frames. A YOLOv8
detector (BuildYoloFromSafeTensors, DecodeYoloDetections) finds the objects of
each frame and TByteTracker (ByteTrack, Zhang et al. ECCV 2022) links them
across frames, so every object keeps one track id for as long as it is seen.

Two modes:
  * no arguments: a smoke of the committed random pico fixture
    tests/fixtures/tiny_yolo.safetensors on a few synthetic frames.
  * --model FILE --input DIR --output DIR: track the objects of every PNG /
    JPEG / PPM frame of a folder (sorted by name, numbers compared by value)
    with a real ultralytics yolov8 checkpoint.

For each frame the program writes <name>_tracks.png: the frame with one box
per track, coloured by track id, labelled with the id. --gif FILE also writes
all the annotated frames as one animated GIF.

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
  neuralnetwork, neuralvolume, neuraldatasets, neuralimageresize,
  neuralpretrained, neuralgif;

const
  // Tracks listed per frame in the log.
  csMaxLoggedTracks = 8;
  // Fixture committed with the YOLO importer tests.
  FixturesDir = '../../tests/fixtures/';
  // ultralytics Colors palette (utils/plotting.py), indexed by track id.
  csTrackPalette: array[0..19] of cardinal = ($FF3838, $FF9D97, $FF701F,
    $FFB21D, $CFD231, $48F90A, $92CC17, $3DDB86, $1A9334, $00D4BB, $2C99A8,
    $00C2FF, $344593, $6473FF, $0018EC, $8438FF, $520085, $CB38FF, $FF95C8,
    $FF37C7);

type
  TObjectTrackingOptions = record
    ModelFile, ConfigFile, InputDir, OutputDir, GifFile: string;
    ScoreThreshold, IoUThreshold: TNeuralFloat;
    KeptClassIds: TNeuralIntegerArray; // empty = every class
    TrackThreshold, MatchThreshold: double;
    TrackBuffer, Fps: integer;
    UseOpenCL, UseInt8: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: ObjectTracking                          (pico fixture smoke)');
  WriteLn('       ObjectTracking --model FILE --input DIR --output DIR [options]');
  WriteLn('  --model FILE         ultralytics yolov8 checkpoint (.safetensors)');
  WriteLn('  --config FILE        its config.json (default: config.json next to ',
    'the model)');
  WriteLn('  --input DIR          folder of consecutive frames ',
    '(PNG / JPEG / PPM), sorted by name');
  WriteLn('  --output DIR         writes <frame>_tracks.png');
  WriteLn('  --gif FILE           also write the annotated frames as an ',
    'animated GIF');
  WriteLn('  --classes LIST       comma-separated class ids to track, e.g. 0,2 ',
    '(default: all)');
  WriteLn('  --score S            detector score threshold (default 0.1, the ',
    'ByteTrack low-score bound)');
  WriteLn('  --iou T              detector NMS IoU threshold (default 0.7)');
  WriteLn('  --track-thresh T     ByteTrack high-score threshold (default 0.5)');
  WriteLn('  --match-thresh T     ByteTrack first-association IoU cost limit ',
    '(default 0.8)');
  WriteLn('  --track-buffer N     frames a lost track is kept at 30 fps ',
    '(default 30)');
  WriteLn('  --fps N              frame rate of the clip, also the GIF rate ',
    '(default 30)');
  WriteLn('  --int8               store the conv weights in int8');
  WriteLn('  --gpu                run the net on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N     OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N       OpenCL device index (default 0)');
end;

// "0,2,7" -> (0, 2, 7).
function ParseClassIds(const ClassList: string): TNeuralIntegerArray;
var
  Items: TStringList;
  ItemPos, MaxItemPos: integer;
begin
  Items := TStringList.Create;
  try
    Items.StrictDelimiter := true;
    Items.Delimiter := ',';
    Items.DelimitedText := ClassList;
    SetLength(Result, Items.Count);
    MaxItemPos := Items.Count - 1;
    for ItemPos := 0 to MaxItemPos do
      Result[ItemPos] := StrToInt(Trim(Items[ItemPos]));
  finally
    Items.Free;
  end;
end;

function ParseOptions(out Opt: TObjectTrackingOptions): boolean;
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
  Opt.InputDir := '';
  Opt.OutputDir := '';
  Opt.GifFile := '';
  Opt.ScoreThreshold := 0.1;
  Opt.IoUThreshold := 0.7;
  Opt.KeptClassIds := nil;
  Opt.TrackThreshold := 0.5;
  Opt.MatchThreshold := 0.8;
  Opt.TrackBuffer := 30;
  Opt.Fps := 30;
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
    else if Arg = '--input' then Opt.InputDir := NextArg()
    else if Arg = '--output' then Opt.OutputDir := NextArg()
    else if Arg = '--gif' then Opt.GifFile := NextArg()
    else if Arg = '--classes' then Opt.KeptClassIds := ParseClassIds(NextArg())
    else if Arg = '--score' then Opt.ScoreThreshold := StrToFloat(NextArg())
    else if Arg = '--iou' then Opt.IoUThreshold := StrToFloat(NextArg())
    else if Arg = '--track-thresh' then Opt.TrackThreshold := StrToFloat(NextArg())
    else if Arg = '--match-thresh' then Opt.MatchThreshold := StrToFloat(NextArg())
    else if Arg = '--track-buffer' then Opt.TrackBuffer := StrToInt(NextArg())
    else if Arg = '--fps' then Opt.Fps := StrToInt(NextArg())
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
  if (Opt.ModelFile <> '') and ((Opt.InputDir = '') or (Opt.OutputDir = '')) then
    raise Exception.Create('--model needs --input and --output');
  if Opt.Fps < 1 then
    raise Exception.Create('--fps must be positive');
  if Opt.TrackBuffer < 0 then
    raise Exception.Create('--track-buffer must not be negative');
end;

// Keeps the detections whose class is in KeptClassIds (all when it is empty).
procedure KeepDetectionsOfClasses(var Dets: TYoloDetectionArray;
  const KeptClassIds: TNeuralIntegerArray);
var
  DetPos, MaxDetPos, KeptCount, ClassPos, MaxClassPos: integer;
  IsKept: boolean;
begin
  if Length(KeptClassIds) = 0 then exit;
  KeptCount := 0;
  MaxDetPos := High(Dets);
  MaxClassPos := High(KeptClassIds);
  for DetPos := 0 to MaxDetPos do
  begin
    IsKept := false;
    for ClassPos := 0 to MaxClassPos do
      if Dets[DetPos].ClassId = KeptClassIds[ClassPos] then IsKept := true;
    if IsKept then
    begin
      Dets[KeptCount] := Dets[DetPos];
      Inc(KeptCount);
    end;
  end;
  SetLength(Dets, KeptCount);
end;

// Draws every track box of Tracks into Image (RGB 0..255): a 2 px outline in
// the colour of its id and "id class" on a filled tag above the top-left corner.
procedure DrawTracks(Image: TNNetVolume; const Tracks: TByteTrackedBoxArray;
  const Config: TYoloConfig);
var
  TrackPos, MaxTrackPos, X1, Y1, X2, Y2, PixelSize, TagWidth, TagHeight,
    TagTop: integer;
  TrackLabel: string;
  RGB: cardinal;
  BoxColor, TextColor: array[0..2] of TNeuralFloat;
begin
  PixelSize := Max(1, Min(Image.SizeX, Image.SizeY) div 240);
  TagHeight := (csBitmapFontGlyphHeight + 2) * PixelSize;
  MaxTrackPos := High(Tracks);
  for TrackPos := 0 to MaxTrackPos do
  begin
    RGB := csTrackPalette[Tracks[TrackPos].TrackId mod Length(csTrackPalette)];
    BoxColor[0] := (RGB shr 16) and $FF;
    BoxColor[1] := (RGB shr 8) and $FF;
    BoxColor[2] := RGB and $FF;
    // ultralytics Annotator: white text on dark colours, black on light ones.
    if 0.299 * BoxColor[0] + 0.587 * BoxColor[1] + 0.114 * BoxColor[2] > 160 then
      TextColor[0] := 0
    else
      TextColor[0] := 255;
    TextColor[1] := TextColor[0];
    TextColor[2] := TextColor[0];
    X1 := Round(EnsureRange(Tracks[TrackPos].X1, -1, Image.SizeX));
    Y1 := Round(EnsureRange(Tracks[TrackPos].Y1, -1, Image.SizeY));
    X2 := Round(EnsureRange(Tracks[TrackPos].X2, -1, Image.SizeX));
    Y2 := Round(EnsureRange(Tracks[TrackPos].Y2, -1, Image.SizeY));
    DrawLineCV2(Image, X1, Y1, X2, Y1, BoxColor);
    DrawLineCV2(Image, X1, Y2, X2, Y2, BoxColor);
    DrawLineCV2(Image, X1, Y1, X1, Y2, BoxColor);
    DrawLineCV2(Image, X2, Y1, X2, Y2, BoxColor);
    DrawLineCV2(Image, X1 + 1, Y1 + 1, X2 - 1, Y1 + 1, BoxColor);
    DrawLineCV2(Image, X1 + 1, Y2 - 1, X2 - 1, Y2 - 1, BoxColor);
    DrawLineCV2(Image, X1 + 1, Y1 + 1, X1 + 1, Y2 - 1, BoxColor);
    DrawLineCV2(Image, X2 - 1, Y1 + 1, X2 - 1, Y2 - 1, BoxColor);
    // The tag sits above the box, or inside it when the box touches the top.
    TagTop := Y1 - TagHeight;
    if TagTop < 0 then TagTop := Y1;
    TrackLabel := IntToStr(Tracks[TrackPos].TrackId) + ' ' +
      YoloClassLabel(Config, Tracks[TrackPos].ClassId);
    TagWidth := BitmapTextWidth(TrackLabel, PixelSize) + 2 * PixelSize;
    FillRectangleCV2(Image, X1, TagTop, X1 + TagWidth - 1, TagTop + TagHeight - 1,
      BoxColor);
    DrawBitmapText(Image, X1 + PixelSize, TagTop + PixelSize, TrackLabel,
      TextColor, PixelSize);
  end;
end;

// Applies --int8 and --gpu to NN; returns where it runs, for the log.
function PrepareCompute(NN: TNNet; const Opt: TObjectTrackingOptions
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

// Tracks the objects of every frame of Opt.InputDir into Opt.OutputDir.
procedure TrackFolder(const Opt: TObjectTrackingOptions);
var
  NN: TNNet;
  Config: TYoloConfig;
  Tracker: TByteTracker;
  Frames: TStringList;
  Image, NetInput: TNNetVolume;
  Letterbox: TYoloLetterbox;
  Dets: TYoloDetectionArray;
  Tracks: TByteTrackedBoxArray;
  GifWriter: TNNetAnimatedGIFWriter;
  FramePos, MaxFramePos, TrackPos, MaxTrackPos, GifWidth, GifHeight,
    MaxTrackId: integer;
  TotalSeconds, FrameSeconds: double;
  OutputFile, ComputeText, TrackText: string;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  {$ENDIF}
begin
  Frames := TStringList.Create;
  Image := TNNetVolume.Create;
  NetInput := TNNetVolume.Create;
  NN := nil;
  Tracker := nil;
  GifWriter := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    ListImageFilesInFolder(Opt.InputDir, Frames);
    if Frames.Count = 0 then
      raise Exception.Create('no PNG / JPEG / PPM frames in ' + Opt.InputDir);
    ForceDirectories(Opt.OutputDir);
    NN := BuildYoloFromSafeTensors(Opt.ModelFile, Config, {pTrainable=}false,
      Opt.ConfigFile);
    WriteLn(YoloConfigToString(Config));
    ComputeText := PrepareCompute(NN, Opt
      {$IFDEF OpenCL}, OpenCLDevices{$ENDIF});
    Tracker := TByteTracker.Create(Opt.TrackThreshold, Opt.MatchThreshold,
      Opt.TrackBuffer, Opt.Fps);
    WriteLn(Format('Tracking %d frames from %s on %s', [Frames.Count,
      Opt.InputDir, ComputeText]));
    TotalSeconds := 0;
    MaxTrackId := 0;
    MaxFramePos := Frames.Count - 1;
    for FramePos := 0 to MaxFramePos do
    begin
      if not LoadImageFromFileIntoVolume(IncludeTrailingPathDelimiter(
        Opt.InputDir) + Frames[FramePos], Image) then
        raise Exception.Create('could not load ' + Frames[FramePos]);
      StartTime := Now();
      LetterboxYoloImage(Image, NetInput, Config.ImageSize, Letterbox);
      NN.Compute(NetInput);
      Dets := DecodeYoloDetections(NN.GetLastLayer().Output, Config,
        Opt.ScoreThreshold, Opt.IoUThreshold);
      ScaleYoloBoxesToImage(Dets, Config.ImageSize, Image.SizeX, Image.SizeY);
      KeepDetectionsOfClasses(Dets, Opt.KeptClassIds);
      Tracks := Tracker.Update(Dets);
      FrameSeconds := (Now() - StartTime) * 86400.0;
      TotalSeconds := TotalSeconds + FrameSeconds;
      DrawTracks(Image, Tracks, Config);
      OutputFile := IncludeTrailingPathDelimiter(Opt.OutputDir) +
        ChangeFileExt(Frames[FramePos], '') + '_tracks.png';
      if not SaveImageFromVolumeIntoFile(Image, OutputFile) then
        raise Exception.Create('could not save ' + OutputFile);
      TrackText := '';
      MaxTrackPos := High(Tracks);
      for TrackPos := 0 to MaxTrackPos do
      begin
        if TrackPos < csMaxLoggedTracks then
          TrackText := TrackText + Format(' %d(class %s, %.2f)',
            [Tracks[TrackPos].TrackId,
            YoloClassLabel(Config, Tracks[TrackPos].ClassId),
            Tracks[TrackPos].Score]);
        MaxTrackId := Max(MaxTrackId, Tracks[TrackPos].TrackId);
      end;
      if Length(Tracks) > csMaxLoggedTracks then
        TrackText := TrackText + Format(' +%d more',
          [Length(Tracks) - csMaxLoggedTracks]);
      WriteLn(Format('  %s %dx%d: %.2f s, %d detection(s), %d track(s):%s',
        [Frames[FramePos], Image.SizeX, Image.SizeY, FrameSeconds,
        Length(Dets), Length(Tracks), TrackText]));
      if Opt.GifFile <> '' then
      begin
        if GifWriter = nil then
        begin
          GifWidth := Image.SizeX;
          GifHeight := Image.SizeY;
          GifWriter := TNNetAnimatedGIFWriter.Create(Opt.GifFile, GifWidth,
            GifHeight, Max(1, Round(100 / Opt.Fps)));
        end;
        // A GIF has one canvas size: frames of another size are left out.
        if (Image.SizeX = GifWidth) and (Image.SizeY = GifHeight) then
          GifWriter.AddFrame(Image, 0, 255)
        else
          WriteLn('  (', Frames[FramePos], ' left out of the GIF: its canvas ',
            'is ', GifWidth, 'x', GifHeight, ')');
      end;
    end;
    WriteLn(Format('Done: %d frames, %.2f s of detection + tracking (%.2f ' +
      's/frame), highest track id %d -> %s', [Frames.Count, TotalSeconds,
      TotalSeconds / Frames.Count, MaxTrackId, Opt.OutputDir]));
    if GifWriter <> nil then
    begin
      GifWriter.Finish;
      WriteLn(Format('  %d-frame GIF -> %s', [GifWriter.FrameCount, Opt.GifFile]));
    end;
  finally
    GifWriter.Free;
    Tracker.Free;
    NN.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
    NetInput.Free;
    Image.Free;
    Frames.Free;
  end;
end;

// Writes Count SizeX x SizeY frames of two bright squares moving in opposite
// directions over a gradient.
procedure MakeSyntheticFrames(const Dir: string; Count, SizeX, SizeY: integer);
var
  Img: TNNetVolume;
  FramePos, X, Y, SquareSize, LeftA, LeftB, Top: integer;
begin
  ForceDirectories(Dir);
  Img := TNNetVolume.Create(SizeX, SizeY, 3);
  try
    SquareSize := Min(SizeX, SizeY) div 4;
    for FramePos := 0 to Count - 1 do
    begin
      LeftA := SizeX div 8 + FramePos * 4;
      LeftB := SizeX - SizeX div 8 - SquareSize - FramePos * 4;
      Top := SizeY div 3;
      for Y := 0 to SizeY - 1 do
        for X := 0 to SizeX - 1 do
        begin
          Img[X, Y, 0] := (X * 255) div (SizeX - 1);
          Img[X, Y, 1] := 60;
          Img[X, Y, 2] := (Y * 255) div (SizeY - 1);
          if (Y >= Top) and (Y < Top + SquareSize) and
             (((X >= LeftA) and (X < LeftA + SquareSize)) or
              ((X >= LeftB) and (X < LeftB + SquareSize))) then
          begin
            Img[X, Y, 0] := 240;
            Img[X, Y, 1] := 230;
            Img[X, Y, 2] := 40;
          end;
        end;
      if not SaveImageFromVolumeIntoFile(Img, IncludeTrailingPathDelimiter(Dir) +
        'frame_' + IntToStr(FramePos + 1) + '.png') then
        raise Exception.Create('could not save a frame in ' + Dir);
    end;
  finally
    Img.Free;
  end;
end;

// Pico smoke: four 128x96 frames through the random tiny_yolo fixture.
procedure RunFixtureSmoke();
var
  Opt: TObjectTrackingOptions;
begin
  MakeSyntheticFrames('tracking_smoke_frames', 4, 128, 96);
  ParseOptions(Opt);
  Opt.ModelFile := FixturesDir + 'tiny_yolo.safetensors';
  Opt.ConfigFile := FixturesDir + 'tiny_yolo_config.json';
  Opt.InputDir := 'tracking_smoke_frames';
  Opt.OutputDir := 'tracking_smoke_out';
  Opt.GifFile := 'tracking_smoke.gif';
  // The random fixture scores about 0.52, below the default birth score of
  // 0.6; the score bound and the class filter keep a handful of boxes.
  Opt.TrackThreshold := 0.4;
  Opt.ScoreThreshold := 0.525;
  Opt.KeptClassIds := ParseClassIds('0');
  TrackFolder(Opt);
end;

var
  Options: TObjectTrackingOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFile <> '' then
      TrackFolder(Options)
    else
    begin
      WriteLn('ObjectTracking smoke (random pico YOLO weights)');
      WriteLn('===============================================');
      RunFixtureSmoke();
    end;
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
