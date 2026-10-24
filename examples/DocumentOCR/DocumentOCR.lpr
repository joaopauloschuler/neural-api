program DocumentOCR;
(*
DocumentOCR: reads the text of a whole page or photo. A docTR DBNet detector
(BuildDBNetFromSafeTensors, DetectDBNetTextBoxes) finds the word boxes,
SortDBNetTextBoxesInReadingOrder puts them in reading order, and a TrOCR
recognizer (BuildTrOCRFromSafeTensors, TranscribeTrOCRImageRegion) reads each
box with a KV-cache greedy decode. The text is printed line by line.
--merge-lines reads one crop per text line (MergeDBNetTextBoxesIntoLines);
--rotated decodes rotated boxes and turns each crop upright (CropRotatedRect).

Two modes:
  * --smoke (or no arguments): the committed random pico fixtures
    tests/fixtures/tiny_dbnet_mobilenet_v3_large.* and tiny_trocr.* on a
    synthetic page. Random weights: the boxes and ids are not real text.
  * --detector FILE --recognizer FILE --input IMG: a real docTR DBNet
    checkpoint and a real TrOCR checkpoint (microsoft/trocr-small-printed).

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
  neuralhftokenizer, neuralpretrained;

const
  csFixtureDirs: array[0..1] of string = ('tests/fixtures/',
    '../../tests/fixtures/');
  // Box colours of successive text lines in --boxes.
  csLinePalette: array[0..5] of cardinal = ($FF3838, $1A9334, $0018EC,
    $FF9D97, $CB38FF, $00C2FF);

type
  TDocumentOCROptions = record
    DetectorFile, DetectorConfigFile: string;
    RecognizerFile, RecognizerConfigFile, ProcessorFile, TokenizerFile: string;
    InputFile, BoxesFile: string;
    MaxNewTokens: integer;
    BinThreshold, BoxThreshold: TNeuralFloat; // 0 = the detector config's
    LineGapFactor: TNeuralFloat;              // --merge-lines split gap / height
    IsSmoke, UseOpenCL, UseInt8, MergesLines, IsRotated: boolean;
    OpenCLPlatform, OpenCLDevice: integer;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: DocumentOCR [--smoke]                  (pico fixture smoke)');
  WriteLn('       DocumentOCR --detector FILE --recognizer FILE --input IMG [options]');
  WriteLn('  --detector FILE           docTR DBNet .safetensors (db_mobilenet_v3_large');
  WriteLn('                            or db_resnet50)');
  WriteLn('  --detector-config FILE    docTR config.json (default: next to --detector,');
  WriteLn('                            else docTR defaults, 1024x1024)');
  WriteLn('  --recognizer FILE         TrOCR model.safetensors (microsoft/trocr-small-printed)');
  WriteLn('  --recognizer-config FILE  TrOCR config.json (default: next to --recognizer)');
  WriteLn('  --processor FILE          preprocessor_config.json (default: next to');
  WriteLn('                            --recognizer, else 0.5 mean/std bilinear)');
  WriteLn('  --tokenizer FILE          tokenizer.json (default: next to --recognizer);');
  WriteLn('                            without one the token ids are printed');
  WriteLn('  --input IMG               page or photo (PNG / JPEG / BMP)');
  WriteLn('  --boxes IMG               also write the image with the boxes drawn (PNG)');
  WriteLn('  --max-tokens N            token cap per box (default 32)');
  WriteLn('  --bin-threshold X         text pixel probability (default: config.json)');
  WriteLn('  --box-threshold X         minimum mean box probability (default: config.json)');
  WriteLn('  --merge-lines             read whole text lines: merge the word boxes of a line');
  WriteLn('                            (split at a gap above --line-gap x the box height)');
  WriteLn('  --line-gap FACTOR         --merge-lines split gap in box heights (default 2)');
  WriteLn('  --rotated                 rotated boxes (docTR assume_straight_pages=False),');
  WriteLn('                            each crop turned upright before TrOCR');
  WriteLn('  --int8                    store the weights in int8 (less RAM, small ',
    'accuracy loss)');
  WriteLn('  --gpu                     run the nets on OpenCL (needs a build with ',
    '-dOpenCL)');
  WriteLn('  --gpu-platform N          OpenCL platform index (default 0)');
  WriteLn('  --gpu-device N            OpenCL device index (default 0)');
end;

function ParseOptions(out Opt: TDocumentOCROptions): boolean;
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
  Opt.DetectorFile := '';
  Opt.DetectorConfigFile := '';
  Opt.RecognizerFile := '';
  Opt.RecognizerConfigFile := '';
  Opt.ProcessorFile := '';
  Opt.TokenizerFile := '';
  Opt.InputFile := '';
  Opt.BoxesFile := '';
  Opt.MaxNewTokens := 32;
  Opt.BinThreshold := 0;
  Opt.BoxThreshold := 0;
  Opt.LineGapFactor := 2;
  Opt.MergesLines := false;
  Opt.IsRotated := false;
  Opt.IsSmoke := false;
  Opt.UseOpenCL := false;
  Opt.UseInt8 := false;
  Opt.OpenCLPlatform := 0;
  Opt.OpenCLDevice := 0;
  Result := true;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    Arg := ParamStr(ArgPos);
    if Arg = '--smoke' then Opt.IsSmoke := true
    else if Arg = '--detector' then Opt.DetectorFile := NextArg()
    else if Arg = '--detector-config' then Opt.DetectorConfigFile := NextArg()
    else if Arg = '--recognizer' then Opt.RecognizerFile := NextArg()
    else if Arg = '--recognizer-config' then
      Opt.RecognizerConfigFile := NextArg()
    else if Arg = '--processor' then Opt.ProcessorFile := NextArg()
    else if Arg = '--tokenizer' then Opt.TokenizerFile := NextArg()
    else if Arg = '--input' then Opt.InputFile := NextArg()
    else if Arg = '--boxes' then Opt.BoxesFile := NextArg()
    else if Arg = '--max-tokens' then Opt.MaxNewTokens := StrToInt(NextArg())
    else if Arg = '--bin-threshold' then Opt.BinThreshold := StrToFloat(NextArg())
    else if Arg = '--box-threshold' then Opt.BoxThreshold := StrToFloat(NextArg())
    else if Arg = '--merge-lines' then Opt.MergesLines := true
    else if Arg = '--line-gap' then Opt.LineGapFactor := StrToFloat(NextArg())
    else if Arg = '--rotated' then Opt.IsRotated := true
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
  if (Opt.DetectorFile = '') and (Opt.RecognizerFile = '') then
    Opt.IsSmoke := true;
  if (not Opt.IsSmoke) and ((Opt.DetectorFile = '') or
    (Opt.RecognizerFile = '') or (Opt.InputFile = '')) then
    raise Exception.Create('--detector, --recognizer and --input go together');
  if Opt.MaxNewTokens < 1 then
    raise Exception.Create('--max-tokens must be at least 1');
  if Opt.LineGapFactor <= 0 then
    raise Exception.Create('--line-gap must be positive');
  if Opt.MergesLines and Opt.IsRotated then
    raise Exception.Create('--merge-lines works on straight boxes, not with --rotated');
end;

// FileName in the first fixture folder that has it (repo root or example dir).
function FixturePath(const FileName: string): string;
var
  DirPos: integer;
begin
  for DirPos := Low(csFixtureDirs) to High(csFixtureDirs) do
  begin
    Result := csFixtureDirs[DirPos] + FileName;
    if FileExists(Result) then exit;
  end;
  raise Exception.Create('fixture not found: ' + FileName +
    ' (run from the repo root or the example folder)');
end;

// FileName beside ModelFile when it exists, else ''.
function SidecarFile(const ModelFile, FileName: string): string;
begin
  Result := ExtractFilePath(ModelFile) + FileName;
  if not FileExists(Result) then Result := '';
end;

// Applies --int8 and --gpu to NN; returns where it runs, for the log.
function PrepareCompute(NN: TNNet; const Opt: TDocumentOCROptions
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

// The ids without the trailing EOS, space separated, in brackets.
function TokenIdsText(const Ids: TNeuralIntegerArray; EosTokenId: integer): string;
var
  IdPos, MaxIdPos: integer;
begin
  Result := '';
  MaxIdPos := High(Ids);
  if (MaxIdPos >= 0) and (Ids[MaxIdPos] = EosTokenId) then Dec(MaxIdPos);
  for IdPos := 0 to MaxIdPos do
  begin
    if IdPos > 0 then Result := Result + ' ';
    Result := Result + IntToStr(Ids[IdPos]);
  end;
  Result := '[' + Result + ']';
end;

// The --boxes colour of text line LineId.
procedure GetLineColor(LineId: integer; out Color: array of TNeuralFloat);
var
  RGB: cardinal;
begin
  RGB := csLinePalette[LineId mod Length(csLinePalette)];
  Color[0] := (RGB shr 16) and $FF;
  Color[1] := (RGB shr 8) and $FF;
  Color[2] := RGB and $FF;
end;

// Draws each box in the colour of its line, numbered in reading order.
procedure DrawTextBoxes(Image: TNNetVolume; const Boxes: TDBNetTextBoxArray;
  const LineIds: TNeuralIntegerArray);
var
  BoxPos, MaxBoxPos, PixelSize, Left, Top: integer;
  BoxColor: array[0..2] of TNeuralFloat;
begin
  PixelSize := Max(1, Min(Image.SizeX, Image.SizeY) div 400);
  MaxBoxPos := High(Boxes);
  for BoxPos := 0 to MaxBoxPos do
  begin
    GetLineColor(LineIds[BoxPos], BoxColor);
    Left := Floor(Boxes[BoxPos].X1);
    Top := Floor(Boxes[BoxPos].Y1);
    DrawRectangleOutline(Image, Left, Top, Ceil(Boxes[BoxPos].X2) - 1,
      Ceil(Boxes[BoxPos].Y2) - 1, BoxColor, PixelSize);
    DrawBitmapText(Image, Left + 2 * PixelSize, Top + 2 * PixelSize,
      IntToStr(BoxPos + 1), BoxColor, PixelSize);
  end;
end;

// DrawTextBoxes for rotated boxes: the four sides, the number at the first corner.
procedure DrawRotatedTextBoxes(Image: TNNetVolume;
  const Boxes: TDBNetRotatedTextBoxArray; const LineIds: TNeuralIntegerArray);
var
  BoxPos, MaxBoxPos, PixelSize, CornerPos, NextCornerPos, Shift,
    MaxShift: integer;
  BoxColor: array[0..2] of TNeuralFloat;
begin
  PixelSize := Max(1, Min(Image.SizeX, Image.SizeY) div 400);
  MaxShift := PixelSize - 1;
  MaxBoxPos := High(Boxes);
  for BoxPos := 0 to MaxBoxPos do
  begin
    GetLineColor(LineIds[BoxPos], BoxColor);
    for CornerPos := 0 to 3 do
    begin
      NextCornerPos := (CornerPos + 1) and 3;
      for Shift := 0 to MaxShift do
        DrawLineCV2(Image, Round(Boxes[BoxPos].Rect.CornerX[CornerPos]) + Shift,
          Round(Boxes[BoxPos].Rect.CornerY[CornerPos]) + Shift,
          Round(Boxes[BoxPos].Rect.CornerX[NextCornerPos]) + Shift,
          Round(Boxes[BoxPos].Rect.CornerY[NextCornerPos]) + Shift, BoxColor);
    end;
    DrawBitmapText(Image, Round(Boxes[BoxPos].Rect.CornerX[0]) + 2 * PixelSize,
      Round(Boxes[BoxPos].Rect.CornerY[0]) + 2 * PixelSize,
      IntToStr(BoxPos + 1), BoxColor, PixelSize);
  end;
end;

// Detects, orders and reads the text of Image; prints one output line per text line.
procedure ReadDocument(Image: TNNetVolume; const Opt: TDocumentOCROptions);
const
  // Rotated crops past the image edge read white.
  csCropBorder: array[0..2] of TNeuralFloat = (255, 255, 255);
var
  Detector, Encoder, Decoder: TNNet;
  DetectorConfig: TDBNetConfig;
  RecognizerConfig: TTrOCRConfig;
  Processor: TClipImageProcessorConfig;
  Tokenizer: TNeuralHFTokenizer;
  Boxes: TDBNetTextBoxArray;
  RotatedBoxes: TDBNetRotatedTextBoxArray;
  LineIds, MergedLineIds, Ids: TNeuralIntegerArray;
  BoxesImage, Crop: TNNetVolume;
  BoxPos, MaxBoxPos, Left, Top: integer;
  ComputeText, ProcessorFile, TokenizerFile, WordText, BoxText,
    LineText: string;
  StartTime: TDateTime;
  {$IFDEF OpenCL}
  OpenCLDevices: TEasyOpenCL;
  {$ENDIF}
begin
  Detector := nil;
  Encoder := nil;
  Decoder := nil;
  Tokenizer := nil;
  Crop := nil;
  {$IFDEF OpenCL}
  OpenCLDevices := nil;
  {$ENDIF}
  try
    StartTime := Now();
    Detector := BuildDBNetFromSafeTensors(Opt.DetectorFile, DetectorConfig,
      {pTrainable=}false, Opt.DetectorConfigFile);
    if Opt.BinThreshold > 0 then DetectorConfig.BinThreshold := Opt.BinThreshold;
    if Opt.BoxThreshold > 0 then DetectorConfig.BoxThreshold := Opt.BoxThreshold;
    WriteLn(DBNetConfigToString(DetectorConfig));
    // DecSeqLen=1: TranscribeTrOCRImageRegion then decodes with the KV cache.
    BuildTrOCRFromSafeTensors(Opt.RecognizerFile, Encoder, Decoder,
      RecognizerConfig, {DecSeqLen=}1, {pTrainable=}false,
      Opt.RecognizerConfigFile);
    WriteLn(TrOCRConfigToString(RecognizerConfig));
    ProcessorFile := Opt.ProcessorFile;
    if ProcessorFile = '' then
      ProcessorFile := SidecarFile(Opt.RecognizerFile, 'preprocessor_config.json');
    if ProcessorFile <> '' then
      Processor := ReadClipImageProcessorConfig(ProcessorFile)
    else
      Processor := DefaultTrOCRImageProcessorConfig(RecognizerConfig);
    TokenizerFile := Opt.TokenizerFile;
    if TokenizerFile = '' then
      TokenizerFile := SidecarFile(Opt.RecognizerFile, 'tokenizer.json');
    if TokenizerFile <> '' then
    begin
      Tokenizer := TNeuralHFTokenizer.Create();
      Tokenizer.LoadFromFile(TokenizerFile);
    end;
    WriteLn(Format('  loaded in %.1f s', [(Now() - StartTime) * 86400.0]));
    ComputeText := PrepareCompute(Detector, Opt
      {$IFDEF OpenCL}, OpenCLDevices{$ENDIF});
    PrepareCompute(Encoder, Opt {$IFDEF OpenCL}, OpenCLDevices{$ENDIF});
    PrepareCompute(Decoder, Opt {$IFDEF OpenCL}, OpenCLDevices{$ENDIF});
    WriteLn(Format('Reading %dx%d on %s, detector input %dx%d',
      [Image.SizeX, Image.SizeY, ComputeText, DetectorConfig.InputWidth,
       DetectorConfig.InputHeight]));

    StartTime := Now();
    if Opt.IsRotated then
    begin
      RotatedBoxes := DetectDBNetRotatedTextBoxes(Detector, DetectorConfig, Image);
      SortDBNetRotatedTextBoxesInReadingOrder(RotatedBoxes, LineIds);
      MaxBoxPos := High(RotatedBoxes);
    end
    else
    begin
      Boxes := DetectDBNetTextBoxes(Detector, DetectorConfig, Image);
      SortDBNetTextBoxesInReadingOrder(Boxes, LineIds);
      MaxBoxPos := High(Boxes);
    end;
    WriteLn(Format('  %d text boxes in %.2f s', [MaxBoxPos + 1,
      (Now() - StartTime) * 86400.0]));
    if Opt.MergesLines then
    begin
      Boxes := MergeDBNetTextBoxesIntoLines(Boxes, LineIds, MergedLineIds,
        Opt.LineGapFactor);
      LineIds := MergedLineIds;
      MaxBoxPos := High(Boxes);
      WriteLn(Format('  merged into %d line boxes', [MaxBoxPos + 1]));
    end;
    StartTime := Now();
    LineText := '';
    Crop := TNNetVolume.Create;
    for BoxPos := 0 to MaxBoxPos do
    begin
      if Opt.IsRotated then
      begin
        CropRotatedRect(Image, Crop, RotatedBoxes[BoxPos].Rect, csCropBorder);
        Ids := TranscribeTrOCRImage(Encoder, Decoder, RecognizerConfig,
          Processor, Crop, Opt.MaxNewTokens);
        BoxText := Format('(%.0f,%.0f) %dx%d at %.0f deg score %.3f',
          [RotatedBoxes[BoxPos].Rect.CornerX[0],
           RotatedBoxes[BoxPos].Rect.CornerY[0], Crop.SizeX, Crop.SizeY,
           RadToDeg(ArcTan2(RotatedBoxes[BoxPos].Rect.CornerY[1] -
             RotatedBoxes[BoxPos].Rect.CornerY[0],
             RotatedBoxes[BoxPos].Rect.CornerX[1] -
             RotatedBoxes[BoxPos].Rect.CornerX[0])),
           RotatedBoxes[BoxPos].Score]);
      end
      else
      begin
        Left := Floor(Boxes[BoxPos].X1);
        Top := Floor(Boxes[BoxPos].Y1);
        Ids := TranscribeTrOCRImageRegion(Encoder, Decoder, RecognizerConfig,
          Processor, Image, Left, Top, Ceil(Boxes[BoxPos].X2) - Left,
          Ceil(Boxes[BoxPos].Y2) - Top, Opt.MaxNewTokens);
        BoxText := Format('(%.0f,%.0f)-(%.0f,%.0f) score %.3f',
          [Boxes[BoxPos].X1, Boxes[BoxPos].Y1, Boxes[BoxPos].X2,
           Boxes[BoxPos].Y2, Boxes[BoxPos].Score]);
      end;
      if Tokenizer <> nil then WordText := Trim(Tokenizer.Decode(Ids))
      else WordText := TokenIdsText(Ids, RecognizerConfig.EosTokenId);
      WriteLn(Format('  box %d line %d %s: %s', [BoxPos + 1, LineIds[BoxPos],
        BoxText, WordText]));
      if (BoxPos > 0) and (LineIds[BoxPos] <> LineIds[BoxPos - 1]) then
        LineText := LineText + LineEnding
      else if BoxPos > 0 then
        LineText := LineText + ' ';
      LineText := LineText + WordText;
    end;
    WriteLn(Format('  recognized in %.2f s', [(Now() - StartTime) * 86400.0]));
    WriteLn('Text:');
    WriteLn(LineText);
    if Opt.BoxesFile <> '' then
    begin
      BoxesImage := TNNetVolume.Create;
      try
        BoxesImage.Copy(Image);
        if Opt.IsRotated then DrawRotatedTextBoxes(BoxesImage, RotatedBoxes, LineIds)
        else DrawTextBoxes(BoxesImage, Boxes, LineIds);
        if not SaveImageFromVolumeIntoFile(BoxesImage, Opt.BoxesFile) then
          raise Exception.Create('could not save ' + Opt.BoxesFile);
        WriteLn('Boxes written to ', Opt.BoxesFile);
      finally
        BoxesImage.Free;
      end;
    end;
  finally
    Crop.Free;
    Tokenizer.Free;
    Decoder.Free;
    Encoder.Free;
    Detector.Free;
    {$IFDEF OpenCL}
    OpenCLDevices.Free;
    {$ENDIF}
  end;
end;

// A white page with two lines of black words in the 5x7 bitmap font.
procedure MakeSyntheticPage(Image: TNNetVolume);
const
  csInk: array[0..2] of TNeuralFloat = (0, 0, 0);
begin
  Image.ReSize(240, 128, 3);
  Image.Fill(255);
  DrawBitmapText(Image, 12, 24, 'HELLO WORLD', csInk, 3);
  DrawBitmapText(Image, 12, 72, 'PASCAL OCR', csInk, 3);
end;

var
  Options: TDocumentOCROptions;
  Image: TNNetVolume;
begin
  try
    if not ParseOptions(Options) then exit;
    Image := TNNetVolume.Create;
    try
      if Options.IsSmoke then
      begin
        WriteLn('DocumentOCR smoke (random pico DBNet + TrOCR weights)');
        WriteLn('=====================================================');
        Options.DetectorFile :=
          FixturePath('tiny_dbnet_mobilenet_v3_large.safetensors');
        Options.DetectorConfigFile :=
          FixturePath('tiny_dbnet_mobilenet_v3_large_config.json');
        Options.RecognizerFile := FixturePath('tiny_trocr.safetensors');
        Options.RecognizerConfigFile := FixturePath('tiny_trocr_config.json');
        MakeSyntheticPage(Image);
      end
      else
      begin
        if not LoadImageFromFileIntoVolume(Options.InputFile, Image) then
          raise Exception.Create('could not load ' + Options.InputFile);
      end;
      ReadDocument(Image, Options);
    finally
      Image.Free;
    end;
  except
    on E: Exception do
    begin
      WriteLn('Error: ', E.Message);
      ExitCode := 1;
    end;
  end;
end.
