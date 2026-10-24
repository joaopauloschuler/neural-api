program KeypointMatching;
(*
KeypointMatching: matches two images with the importers in
neuralpretrained.pas, either SuperPoint keypoints + LightGlue matching
(TSuperPointKeypoints, TLightGlueMatcher; the default) or EfficientLoFTR
detector-free semi-dense matching (--matcher loftr, TEfficientLoFTRMatcher).

Two modes:
  * no arguments: a smoke of a committed random pico fixture
    (tests/fixtures/tiny_lightglue, or tiny_efficientloftr with --matcher
    loftr): synthesize a texture and the same texture shifted by 8 pixels,
    match them and write a side-by-side PNG with the match lines.
  * --model FILE --image0 IMG --image1 IMG --output PNG: match two real images
    with a LightGlue checkpoint (ETH-CVG/lightglue_superpoint model.safetensors,
    which also holds its SuperPoint) or with --superpoint FILE for a separate
    SuperPoint checkpoint (magic-leap-community/superpoint); with --matcher
    loftr, --model is an EfficientLoFTR checkpoint
    (zju-community/efficientloftr model.safetensors).

Both images are resized to --width x --height (640 x 480 by default, as
LightGlueImageProcessor does), turned gray, and the keypoints are drawn back
on the original images.

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
  neuralnetwork, neuralvolume, neuraldatasets, neuralpretrained;

const
  // Fixtures committed by tools/make_pico_lightglue_fixture.py.
  FixturesDir = '../../tests/fixtures/';

type
  TMatchOptions = record
    MatcherName: string; // 'lightglue' or 'loftr'
    ModelFile, ConfigFile, SuperPointFile, SuperPointConfigFile: string;
    ImageFiles: array[0..1] of string;
    OutputFile: string;
    Width, Height, MaxKeypoints: integer;
    KeypointThreshold, FilterThreshold: TNeuralFloat;
    DepthConfidence, WidthConfidence: TNeuralFloat;
    HasDepthConfidence, HasWidthConfidence: boolean;
    CoarseThreshold: TNeuralFloat; // < 0 keeps the config value
  end;

  // One match in original-image pixels, Score in [0, 1].
  TMatchLine = record
    X0, Y0, X1, Y1: integer;
    Score: TNeuralFloat;
  end;
  TMatchLines = array of TMatchLine;

procedure PrintUsage();
begin
  WriteLn('Usage: KeypointMatching [--matcher loftr]       (pico fixture smoke)');
  WriteLn('       KeypointMatching --model FILE --image0 IMG --image1 IMG ',
    '--output PNG [options]');
  WriteLn('  --matcher NAME           lightglue (SuperPoint + LightGlue, default) ',
    'or loftr (EfficientLoFTR)');
  WriteLn('  --model FILE             LightGlue or EfficientLoFTR model.safetensors');
  WriteLn('  --config FILE            its config.json (default: next to --model)');
  WriteLn('  --superpoint FILE        SuperPoint model.safetensors (default: the');
  WriteLn('                           keypoint_detector inside --model)');
  WriteLn('  --superpoint-config FILE its config.json (default: next to ',
    '--superpoint)');
  WriteLn('  --image0 IMG, --image1 IMG  the two images (PNG / JPEG / BMP)');
  WriteLn('  --output PNG             side-by-side image with the match lines');
  WriteLn('  --width N, --height N    net input size, multiples of 8 ',
    '(32 for loftr; default 640 x 480)');
  WriteLn('  --max-keypoints N        keypoints kept per image, -1 = all ',
    '(default: config)');
  WriteLn('  --keypoint-threshold X   SuperPoint score threshold (default: config)');
  WriteLn('  --filter-threshold X     LightGlue match threshold (default: config)');
  WriteLn('  --depth-confidence X     early exit, <= 0 runs every layer ',
    '(default: config)');
  WriteLn('  --width-confidence X     point pruning, <= 0 keeps every point ',
    '(default: config)');
  WriteLn('  --coarse-threshold X     loftr coarse match threshold ',
    '(default: config, 0.2)');
end;

function ParseOptions(out Opt: TMatchOptions): boolean;
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
  Opt.MatcherName := 'lightglue';
  Opt.CoarseThreshold := -1;
  Opt.ModelFile := '';
  Opt.ConfigFile := '';
  Opt.SuperPointFile := '';
  Opt.SuperPointConfigFile := '';
  Opt.ImageFiles[0] := '';
  Opt.ImageFiles[1] := '';
  Opt.OutputFile := '';
  Opt.Width := 640;
  Opt.Height := 480;
  Opt.MaxKeypoints := -2;  // -2 = keep the config value
  Opt.KeypointThreshold := -1;
  Opt.FilterThreshold := -1;
  Opt.DepthConfidence := 0;
  Opt.WidthConfidence := 0;
  Opt.HasDepthConfidence := false;
  Opt.HasWidthConfidence := false;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--matcher' then Opt.MatcherName := LowerCase(NextArg())
    else if Arg = '--model' then Opt.ModelFile := NextArg()
    else if Arg = '--config' then Opt.ConfigFile := NextArg()
    else if Arg = '--superpoint' then Opt.SuperPointFile := NextArg()
    else if Arg = '--superpoint-config' then Opt.SuperPointConfigFile := NextArg()
    else if Arg = '--image0' then Opt.ImageFiles[0] := NextArg()
    else if Arg = '--image1' then Opt.ImageFiles[1] := NextArg()
    else if Arg = '--output' then Opt.OutputFile := NextArg()
    else if Arg = '--width' then Opt.Width := StrToInt(NextArg())
    else if Arg = '--height' then Opt.Height := StrToInt(NextArg())
    else if Arg = '--max-keypoints' then Opt.MaxKeypoints := StrToInt(NextArg())
    else if Arg = '--keypoint-threshold' then
      Opt.KeypointThreshold := StrToFloat(NextArg())
    else if Arg = '--filter-threshold' then
      Opt.FilterThreshold := StrToFloat(NextArg())
    else if Arg = '--depth-confidence' then
    begin
      Opt.DepthConfidence := StrToFloat(NextArg());
      Opt.HasDepthConfidence := true;
    end
    else if Arg = '--width-confidence' then
    begin
      Opt.WidthConfidence := StrToFloat(NextArg());
      Opt.HasWidthConfidence := true;
    end
    else if Arg = '--coarse-threshold' then
      Opt.CoarseThreshold := StrToFloat(NextArg())
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
  if (Opt.MatcherName <> 'lightglue') and (Opt.MatcherName <> 'loftr') then
    raise Exception.Create('--matcher must be lightglue or loftr');
  if (Opt.ModelFile <> '') and ((Opt.ImageFiles[0] = '') or
     (Opt.ImageFiles[1] = '') or (Opt.OutputFile = '')) then
    raise Exception.Create('--model needs --image0, --image1 and --output');
end;

// Paints pixel (X, Y) of the RGB 0..255 Canvas when it is inside.
procedure PutPixel(Canvas: TNNetVolume; X, Y: integer; R, G, B: TNeuralFloat);
begin
  if (X < 0) or (Y < 0) or (X >= Canvas.SizeX) or (Y >= Canvas.SizeY) then exit;
  Canvas[X, Y, 0] := R;
  Canvas[X, Y, 1] := G;
  Canvas[X, Y, 2] := B;
end;

// Bresenham line from (X0, Y0) to (X1, Y1), Thickness pixels wide.
procedure DrawLine(Canvas: TNNetVolume; X0, Y0, X1, Y1, Thickness: integer;
  R, G, B: TNeuralFloat);
var
  DeltaX, DeltaY, StepX, StepY, Error, DoubleError, Offset, Half: integer;
begin
  DeltaX := Abs(X1 - X0);
  DeltaY := -Abs(Y1 - Y0);
  if X0 < X1 then StepX := 1 else StepX := -1;
  if Y0 < Y1 then StepY := 1 else StepY := -1;
  Error := DeltaX + DeltaY;
  Half := Thickness div 2;
  while true do
  begin
    for Offset := -Half to Thickness - 1 - Half do
      if DeltaX >= -DeltaY then PutPixel(Canvas, X0, Y0 + Offset, R, G, B)
      else PutPixel(Canvas, X0 + Offset, Y0, R, G, B);
    if (X0 = X1) and (Y0 = Y1) then break;
    DoubleError := 2 * Error;
    if DoubleError >= DeltaY then
    begin
      Inc(Error, DeltaY);
      Inc(X0, StepX);
    end;
    if DoubleError <= DeltaX then
    begin
      Inc(Error, DeltaX);
      Inc(Y0, StepY);
    end;
  end;
end;

procedure DrawDot(Canvas: TNNetVolume; X, Y, Radius: integer);
var
  OffsetX, OffsetY: integer;
begin
  for OffsetY := -Radius to Radius do
    for OffsetX := -Radius to Radius do
      if OffsetX * OffsetX + OffsetY * OffsetY <= Radius * Radius then
        PutPixel(Canvas, X + OffsetX, Y + OffsetY, 0, 0, 0);
end;

// Loads FileName as RGB 0..255 (a gray file is repeated over 3 channels).
procedure LoadRGB(const FileName: string; Image: TNNetVolume);
var
  Gray: TNNetVolume;
  PixelX, PixelY, MaxPixelX, MaxPixelY: integer;
begin
  if not LoadImageFromFileIntoVolume(FileName, Image) then
    raise Exception.Create('could not load ' + FileName);
  if Image.Depth = 3 then exit;
  Gray := TNNetVolume.Create;
  try
    Gray.Copy(Image);
    Image.ReSize(Gray.SizeX, Gray.SizeY, 3);
    MaxPixelX := Gray.SizeX - 1;
    MaxPixelY := Gray.SizeY - 1;
    for PixelY := 0 to MaxPixelY do
      for PixelX := 0 to MaxPixelX do
      begin
        Image[PixelX, PixelY, 0] := Gray[PixelX, PixelY, 0];
        Image[PixelX, PixelY, 1] := Gray[PixelX, PixelY, 0];
        Image[PixelX, PixelY, 2] := Gray[PixelX, PixelY, 0];
      end;
  finally
    Gray.Free;
  end;
end;

// Side by side Images[0] | Images[1] with every match drawn as a line colored
// from red (score 0) to green (score 1), as transformers' visualization does.
procedure SaveMatchImage(const FileName: string;
  const Images: array of TNNetVolume; const Lines: TMatchLines);
var
  Canvas: TNNetVolume;
  Side, PixelX, PixelY, MaxPixelX, MaxPixelY, Channel: integer;
  LinePos, MaxLinePos, OffsetX, Thickness: integer;
  Score: TNeuralFloat;
begin
  Canvas := TNNetVolume.Create(Images[0].SizeX + Images[1].SizeX,
    Max(Images[0].SizeY, Images[1].SizeY), 3);
  try
    OffsetX := 0;
    for Side := 0 to 1 do
    begin
      MaxPixelX := Images[Side].SizeX - 1;
      MaxPixelY := Images[Side].SizeY - 1;
      for PixelY := 0 to MaxPixelY do
        for PixelX := 0 to MaxPixelX do
          for Channel := 0 to 2 do
            Canvas[OffsetX + PixelX, PixelY, Channel] :=
              Images[Side][PixelX, PixelY, Channel];
      Inc(OffsetX, Images[Side].SizeX);
    end;
    Thickness := Max(1, Canvas.SizeX div 400);
    OffsetX := Images[0].SizeX;
    MaxLinePos := High(Lines);
    for LinePos := 0 to MaxLinePos do
    begin
      Score := Min(Max(Lines[LinePos].Score, 0), 1);
      DrawLine(Canvas, Lines[LinePos].X0, Lines[LinePos].Y0,
        Lines[LinePos].X1 + OffsetX, Lines[LinePos].Y1, Thickness,
        255 * (1 - Score), 255 * Score, 0);
      DrawDot(Canvas, Lines[LinePos].X0, Lines[LinePos].Y0, Thickness + 1);
      DrawDot(Canvas, Lines[LinePos].X1 + OffsetX, Lines[LinePos].Y1,
        Thickness + 1);
    end;
    if not SaveImageFromVolumeIntoFile(Canvas, FileName) then
      raise Exception.Create('could not save ' + FileName);
  finally
    Canvas.Free;
  end;
end;

// The LightGlue matches mapped to each original image, as transformers'
// post_process_keypoint_matching does.
function LightGlueMatchLines(const Images: array of TNNetVolume;
  const Keypoints: array of TSuperPointKeypoints;
  Matcher: TLightGlueMatcher): TMatchLines;
var
  PointPos, MaxPointPos, Partner, LineCount: integer;
begin
  Result := nil;
  SetLength(Result, Matcher.Count0);
  LineCount := 0;
  MaxPointPos := Matcher.Count0 - 1;
  for PointPos := 0 to MaxPointPos do
  begin
    Partner := Matcher.Matches0[PointPos];
    if Partner < 0 then continue;
    Result[LineCount].X0 := Keypoints[0].ImageX(PointPos, Images[0].SizeX);
    Result[LineCount].Y0 := Keypoints[0].ImageY(PointPos, Images[0].SizeY);
    Result[LineCount].X1 := Keypoints[1].ImageX(Partner, Images[1].SizeX);
    Result[LineCount].Y1 := Keypoints[1].ImageY(Partner, Images[1].SizeY);
    Result[LineCount].Score := Matcher.MatchScores0[PointPos];
    Inc(LineCount);
  end;
  SetLength(Result, LineCount);
end;

procedure MatchFilesLightGlue(const Opt: TMatchOptions);
var
  Config: TLightGlueConfig;
  Detector: TNNet;
  Matcher: TLightGlueMatcher;
  Images: array[0..1] of TNNetVolume;
  Grays: array[0..1] of TNNetVolume;
  Keypoints: array[0..1] of TSuperPointKeypoints;
  Side, PointPos, MaxPointPos, MatchCount: integer;
  StartTime: TDateTime;
  ScoreSum: double;
begin
  Detector := nil;
  Matcher := nil;
  for Side := 0 to 1 do
  begin
    Images[Side] := TNNetVolume.Create;
    Grays[Side] := TNNetVolume.Create;
    Keypoints[Side] := TSuperPointKeypoints.Create;
  end;
  try
    Matcher := BuildLightGlueFromSafeTensors(Opt.ModelFile, Config, true,
      Opt.ConfigFile);
    if Opt.FilterThreshold >= 0 then
      Config.FilterThreshold := Opt.FilterThreshold;
    Matcher.FilterThreshold := Config.FilterThreshold;
    if Opt.HasDepthConfidence then Config.DepthConfidence := Opt.DepthConfidence;
    if Opt.HasWidthConfidence then Config.WidthConfidence := Opt.WidthConfidence;
    Matcher.DepthConfidence := Config.DepthConfidence;
    Matcher.WidthConfidence := Config.WidthConfidence;
    Config.KeypointDetector.InputWidth := Opt.Width;
    Config.KeypointDetector.InputHeight := Opt.Height;
    if Opt.SuperPointFile <> '' then
      Detector := BuildSuperPointFromSafeTensors(Opt.SuperPointFile, Opt.Width,
        Opt.Height, Config.KeypointDetector, false, Opt.SuperPointConfigFile)
    else
      Detector := BuildLightGlueKeypointDetectorFromSafeTensors(Opt.ModelFile,
        Config, false);
    if Opt.MaxKeypoints > -2 then
      Config.KeypointDetector.MaxKeypoints := Opt.MaxKeypoints;
    if Opt.KeypointThreshold >= 0 then
      Config.KeypointDetector.KeypointThreshold := Opt.KeypointThreshold;
    WriteLn(LightGlueConfigToString(Config));
    StartTime := Now();
    for Side := 0 to 1 do
    begin
      LoadRGB(Opt.ImageFiles[Side], Images[Side]);
      SuperPointPreprocessImage(Images[Side], Grays[Side], Opt.Width, Opt.Height);
      Keypoints[Side].Detect(Detector, Grays[Side], Config.KeypointDetector);
      WriteLn(Format('  image%d %s: %dx%d, %d keypoints', [Side,
        Opt.ImageFiles[Side], Images[Side].SizeX, Images[Side].SizeY,
        Keypoints[Side].Count]));
    end;
    WriteLn(Format('  SuperPoint: %.2f s', [(Now() - StartTime) * 86400.0]));
    StartTime := Now();
    Matcher.Match(Keypoints[0], Keypoints[1], Opt.Width, Opt.Height);
    MatchCount := 0;
    ScoreSum := 0;
    MaxPointPos := Matcher.Count0 - 1;
    for PointPos := 0 to MaxPointPos do
      if Matcher.Matches0[PointPos] >= 0 then
      begin
        Inc(MatchCount);
        ScoreSum := ScoreSum + Matcher.MatchScores0[PointPos];
      end;
    WriteLn(Format('  LightGlue: %.2f s, %d matches (mean score %.3f); ' +
      'stopped after layer %d of %d, %d + %d of %d + %d keypoints kept',
      [(Now() - StartTime) * 86400.0, MatchCount,
       ScoreSum / Max(MatchCount, 1), Matcher.StopLayer + 1,
       Config.NumHiddenLayers, Matcher.KeptCount0, Matcher.KeptCount1,
       Matcher.Count0, Matcher.Count1]));
    SaveMatchImage(Opt.OutputFile, Images,
      LightGlueMatchLines(Images, Keypoints, Matcher));
    WriteLn('  wrote ', Opt.OutputFile);
  finally
    for Side := 0 to 1 do
    begin
      Keypoints[Side].Free;
      Grays[Side].Free;
      Images[Side].Free;
    end;
    Matcher.Free;
    Detector.Free;
  end;
end;

procedure MatchFilesLoFTR(const Opt: TMatchOptions);
var
  Config: TEfficientLoFTRConfig;
  NN: TNNet;
  Matcher: TEfficientLoFTRMatcher;
  Images: array[0..1] of TNNetVolume;
  Grays: array[0..1] of TNNetVolume;
  Lines: TMatchLines;
  Side, MatchPos, MaxMatchPos: integer;
  StartTime: TDateTime;
  ScoreSum: double;
begin
  NN := nil;
  Matcher := nil;
  for Side := 0 to 1 do
  begin
    Images[Side] := TNNetVolume.Create;
    Grays[Side] := TNNetVolume.Create;
  end;
  try
    StartTime := Now();
    NN := BuildEfficientLoFTRFromSafeTensors(Opt.ModelFile, Opt.Width,
      Opt.Height, Config, {pTrainable=}false, Opt.ConfigFile);
    if Opt.CoarseThreshold >= 0 then
      Config.CoarseMatchingThreshold := Opt.CoarseThreshold;
    WriteLn(EfficientLoFTRConfigToString(Config));
    WriteLn(Format('  load: %.2f s', [(Now() - StartTime) * 86400.0]));
    Matcher := TEfficientLoFTRMatcher.Create(Config);
    for Side := 0 to 1 do
    begin
      LoadRGB(Opt.ImageFiles[Side], Images[Side]);
      SuperPointPreprocessImage(Images[Side], Grays[Side], Opt.Width, Opt.Height);
      WriteLn(Format('  image%d %s: %dx%d', [Side, Opt.ImageFiles[Side],
        Images[Side].SizeX, Images[Side].SizeY]));
    end;
    StartTime := Now();
    Matcher.Match(NN, Grays[0], Grays[1]);
    ScoreSum := 0;
    Lines := nil;
    SetLength(Lines, Matcher.Count);
    MaxMatchPos := Matcher.Count - 1;
    for MatchPos := 0 to MaxMatchPos do
    begin
      Lines[MatchPos].X0 := Matcher.ImageX(Matcher.KeypointX0[MatchPos],
        Images[0].SizeX);
      Lines[MatchPos].Y0 := Matcher.ImageY(Matcher.KeypointY0[MatchPos],
        Images[0].SizeY);
      Lines[MatchPos].X1 := Matcher.ImageX(Matcher.KeypointX1[MatchPos],
        Images[1].SizeX);
      Lines[MatchPos].Y1 := Matcher.ImageY(Matcher.KeypointY1[MatchPos],
        Images[1].SizeY);
      Lines[MatchPos].Score := Matcher.Scores[MatchPos];
      ScoreSum := ScoreSum + Matcher.Scores[MatchPos];
    end;
    WriteLn(Format('  EfficientLoFTR: %.2f s, %d matches (mean score %.3f) ' +
      'on a %dx%d coarse grid', [(Now() - StartTime) * 86400.0, Matcher.Count,
      ScoreSum / Max(Matcher.Count, 1), Matcher.CoarseWidth,
      Matcher.CoarseHeight]));
    SaveMatchImage(Opt.OutputFile, Images, Lines);
    WriteLn('  wrote ', Opt.OutputFile);
  finally
    for Side := 0 to 1 do
    begin
      Grays[Side].Free;
      Images[Side].Free;
    end;
    Matcher.Free;
    NN.Free;
  end;
end;

procedure MatchFiles(const Opt: TMatchOptions);
begin
  if Opt.MatcherName = 'loftr' then MatchFilesLoFTR(Opt)
  else MatchFilesLightGlue(Opt);
end;

// A blocky gray texture and the same texture shifted right by Shift pixels.
procedure MakeSmokePair(const FileName0, FileName1: string;
  Width, Height, Shift: integer);
var
  Image: TNNetVolume;
  PixelX, PixelY, MaxPixelX, MaxPixelY, Channel: integer;
  Value: TNeuralFloat;

  function BlockValue(X, Y: integer): TNeuralFloat;
  begin
    // A fixed hash per 4x4 block keeps the smoke reproducible.
    Result := Frac(Abs(Sin((X div 4) * 12.9898 + (Y div 4) * 78.233)) *
      43758.5453) * 255;
  end;

begin
  Image := TNNetVolume.Create(Width, Height, 3);
  try
    MaxPixelX := Width - 1;
    MaxPixelY := Height - 1;
    for PixelY := 0 to MaxPixelY do
      for PixelX := 0 to MaxPixelX do
        for Channel := 0 to 2 do
          Image[PixelX, PixelY, Channel] := Round(BlockValue(PixelX, PixelY));
    if not SaveImageFromVolumeIntoFile(Image, FileName0) then
      raise Exception.Create('could not save ' + FileName0);
    for PixelY := 0 to MaxPixelY do
      for PixelX := 0 to MaxPixelX do
      begin
        Value := Round(BlockValue((PixelX - Shift + Width) mod Width, PixelY));
        for Channel := 0 to 2 do Image[PixelX, PixelY, Channel] := Value;
      end;
    if not SaveImageFromVolumeIntoFile(Image, FileName1) then
      raise Exception.Create('could not save ' + FileName1);
  finally
    Image.Free;
  end;
end;

procedure RunFixtureSmoke(Opt: TMatchOptions);
begin
  Opt.ImageFiles[0] := 'keypoint_in0.png';
  Opt.ImageFiles[1] := 'keypoint_in1.png';
  Opt.OutputFile := 'keypoint_matches.png';
  if Opt.MatcherName = 'loftr' then
  begin
    // The pico EfficientLoFTR input is 128 x 96 (a multiple of 32).
    Opt.ModelFile := FixturesDir + 'tiny_efficientloftr.safetensors';
    Opt.ConfigFile := FixturesDir + 'tiny_efficientloftr_config.json';
    Opt.Width := 128;
    Opt.Height := 96;
  end
  else
  begin
    Opt.ModelFile := FixturesDir + 'tiny_lightglue.safetensors';
    Opt.ConfigFile := FixturesDir + 'tiny_lightglue_config.json';
    Opt.Width := 64;
    Opt.Height := 48;
  end;
  MakeSmokePair(Opt.ImageFiles[0], Opt.ImageFiles[1], Opt.Width, Opt.Height, 8);
  MatchFiles(Opt);
end;

var
  Options: TMatchOptions;
begin
  try
    if not ParseOptions(Options) then exit;
    if Options.ModelFile <> '' then
      MatchFiles(Options)
    else
    begin
      if Options.MatcherName = 'loftr' then
        WriteLn('EfficientLoFTR dense matching smoke (random pico weights)')
      else
        WriteLn('SuperPoint + LightGlue keypoint matching smoke (random pico weights)');
      WriteLn('===================================================================');
      RunFixtureSmoke(Options);
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
