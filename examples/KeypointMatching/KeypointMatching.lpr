program KeypointMatching;
(*
KeypointMatching: SuperPoint keypoints + LightGlue matching between two images
with the importers in neuralpretrained.pas (TSuperPointKeypoints,
TLightGlueMatcher).

Two modes:
  * no arguments: a smoke of the committed random pico fixture
    tests/fixtures/tiny_lightglue (its SuperPoint is the pico one): synthesize
    a 64x48 texture and the same texture shifted by 8 pixels, match them and
    write a side-by-side PNG with the match lines.
  * --model FILE --image0 IMG --image1 IMG --output PNG: match two real images
    with a LightGlue checkpoint (ETH-CVG/lightglue_superpoint model.safetensors,
    which also holds its SuperPoint) or with --superpoint FILE for a separate
    SuperPoint checkpoint (magic-leap-community/superpoint).

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
    ModelFile, ConfigFile, SuperPointFile, SuperPointConfigFile: string;
    ImageFiles: array[0..1] of string;
    OutputFile: string;
    Width, Height, MaxKeypoints: integer;
    KeypointThreshold, FilterThreshold: TNeuralFloat;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: KeypointMatching                        (pico fixture smoke)');
  WriteLn('       KeypointMatching --model FILE --image0 IMG --image1 IMG ',
    '--output PNG [options]');
  WriteLn('  --model FILE             LightGlue model.safetensors');
  WriteLn('  --config FILE            its config.json (default: next to --model)');
  WriteLn('  --superpoint FILE        SuperPoint model.safetensors (default: the');
  WriteLn('                           keypoint_detector inside --model)');
  WriteLn('  --superpoint-config FILE its config.json (default: next to ',
    '--superpoint)');
  WriteLn('  --image0 IMG, --image1 IMG  the two images (PNG / JPEG / BMP)');
  WriteLn('  --output PNG             side-by-side image with the match lines');
  WriteLn('  --width N, --height N    net input size, multiples of 8 ',
    '(default 640 x 480)');
  WriteLn('  --max-keypoints N        keypoints kept per image, -1 = all ',
    '(default: config)');
  WriteLn('  --keypoint-threshold X   SuperPoint score threshold (default: config)');
  WriteLn('  --filter-threshold X     LightGlue match threshold (default: config)');
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
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--model' then Opt.ModelFile := NextArg()
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
  const Images: array of TNNetVolume;
  const Keypoints: array of TSuperPointKeypoints; Matcher: TLightGlueMatcher;
  NetWidth, NetHeight: integer);
var
  Canvas: TNNetVolume;
  Side, PixelX, PixelY, MaxPixelX, MaxPixelY, Channel: integer;
  PointPos, MaxPointPos, Partner, OffsetX, Thickness: integer;
  X0, Y0, X1, Y1: integer;
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
    MaxPointPos := Matcher.Count0 - 1;
    for PointPos := 0 to MaxPointPos do
    begin
      Partner := Matcher.Matches0[PointPos];
      if Partner < 0 then continue;
      // Net-input pixels back to each original image, truncated as transformers.
      X0 := Trunc(Keypoints[0].X[PointPos] * Images[0].SizeX / NetWidth);
      Y0 := Trunc(Keypoints[0].Y[PointPos] * Images[0].SizeY / NetHeight);
      X1 := Trunc(Keypoints[1].X[Partner] * Images[1].SizeX / NetWidth) +
        Images[0].SizeX;
      Y1 := Trunc(Keypoints[1].Y[Partner] * Images[1].SizeY / NetHeight);
      Score := Min(Max(Matcher.MatchScores0[PointPos], 0), 1);
      DrawLine(Canvas, X0, Y0, X1, Y1, Thickness, 255 * (1 - Score),
        255 * Score, 0);
      DrawDot(Canvas, X0, Y0, Thickness + 1);
      DrawDot(Canvas, X1, Y1, Thickness + 1);
    end;
    if not SaveImageFromVolumeIntoFile(Canvas, FileName) then
      raise Exception.Create('could not save ' + FileName);
  finally
    Canvas.Free;
  end;
end;

procedure MatchFiles(const Opt: TMatchOptions);
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
    WriteLn(Format('  LightGlue: %.2f s, %d matches (mean score %.3f)',
      [(Now() - StartTime) * 86400.0, MatchCount,
       ScoreSum / Max(MatchCount, 1)]));
    SaveMatchImage(Opt.OutputFile, Images, Keypoints, Matcher, Opt.Width,
      Opt.Height);
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

procedure RunFixtureSmoke();
var
  Opt: TMatchOptions;
begin
  ParseOptions(Opt);
  MakeSmokePair('keypoint_in0.png', 'keypoint_in1.png', 64, 48, 8);
  Opt.ModelFile := FixturesDir + 'tiny_lightglue.safetensors';
  Opt.ConfigFile := FixturesDir + 'tiny_lightglue_config.json';
  Opt.ImageFiles[0] := 'keypoint_in0.png';
  Opt.ImageFiles[1] := 'keypoint_in1.png';
  Opt.OutputFile := 'keypoint_matches.png';
  Opt.Width := 64;
  Opt.Height := 48;
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
      WriteLn('SuperPoint + LightGlue keypoint matching smoke (random pico weights)');
      WriteLn('===================================================================');
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
