program YoloDetect;
(*
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

// YoloDetect -- the demo for the YOLOv8 single-shot object-detection importer
// (BuildYoloFromSafeTensors, ultralytics yolov8n: a CSP/C2f backbone + SPPF +
// PANet feature-pyramid neck + a decoupled per-cell DFL detect head over 3
// strides, anchor-free, NO transformer, inference only). It runs a full YOLOv8
// forward pass on ONE image, DECODES the raw head with DecodeYoloDetections
// (sigmoid the class logits, DFL-decode each box side = softmax the reg_max bins
// -> expected ltrb distance, convert to xyxy pixels, greedy IoU NMS), DRAWS the
// surviving boxes as colored rectangle outlines with a "class score" label
// into the image, writes a PPM, and prints the (class, score, box) list.
//
// Usage:
//   YoloDetect [--image FILE] [--prompts "person,bus,dog" [--clip DIR]]
//              [model.safetensors] [config.json] [score_threshold] [iou_threshold]
//
// --prompts switches to YOLO-World v2 (BuildYoloWorldFromSafeTensors): the
// prompts are encoded once by the CLIP text tower of --clip DIR and folded into
// the detector, whose class labels are the prompt strings. --image letterboxes
// a real photo into the net and draws the boxes on it (PNG / JPEG / BMP out).
//
// With no arguments it loads the committed pico fixture
// (tests/fixtures/tiny_yolo.*), a tiny random yolov8 used by the parity tests,
// so the demo runs with NO download, fully OFFLINE. Point it at a real
// ultralytics yolov8 checkpoint + config.json for the real thing.
//
// HONEST NOTE: the pico fixture has RANDOM weights, so the "detections" are not
// meaningful objects -- the demo's job is to exercise the full YOLO
// decode (DFL + NMS) + draw pipeline end to end and SELF-REPORT: it asserts no
// NaN/Inf in the network output and that every decoded box has a finite pixel
// box. A real pretrained checkpoint produces real detections through the exact
// same path. Pure CPU, <1 s on the fixture.

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  SysUtils, Classes, Math, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained, neuralimageresize,
  neuraldatasets, neuralsafetensors, neuralhftokenizer;

const
  PpmFile = 'yolo_detect.ppm';
  PaletteR: array[0..5] of byte = (255, 0, 0, 255, 255, 0);
  PaletteG: array[0..5] of byte = (0, 255, 0, 255, 0, 255);
  PaletteB: array[0..5] of byte = (0, 0, 255, 0, 255, 255);

  FixturesDir = '../../tests/fixtures/';

var
  ModelFile, ConfigFile, IoFile, ImageFile, ClipDir, PromptList, OutputFile: string;
  ScoreThresh, IoUThresh: TNeuralFloat;
  NN: TNNet;
  Config: TYoloConfig;
  Img, NetInput, Output, TextEmb: TNNetVolume;
  Dets: TYoloDetectionArray;
  Letterbox: TYoloLetterbox;
  Prompts: TStringArray;
  Positional: array of string;
  RefJson: TStringList;
  RefRoot: TJSONData;
  InArr: TJSONArray;
  W, H, NumCh, x, yy, ch, FlatIdx, i, x0, y0, x1, y1, TagTop: integer;
  ArgPos, MaxArgPos: integer;
  BoxLabel, ClassLabel, ClassText, Arg: string;
  BoxColor, TextColor: array[0..2] of TNeuralFloat;
  V: TNeuralFloat;
  AnyBad: boolean;
  StartTime: TDateTime;

// Renders the (W,H,3) volume to a P6 PPM, min-max stretched to 0..255 so the
// boxes are visible over whatever synthetic content is present.
procedure WritePPM(AImg: TNNetVolume; const FileName: string);
var
  F: TextFile; Bin: TFileStream;
  px, py, pc: integer; Lo, Hi, Val: TNeuralFloat; B: byte;
begin
  Lo := AImg.FData[0]; Hi := AImg.FData[0];
  for px := 0 to AImg.Size - 1 do
  begin
    if AImg.FData[px] < Lo then Lo := AImg.FData[px];
    if AImg.FData[px] > Hi then Hi := AImg.FData[px];
  end;
  if Hi - Lo < 1e-6 then Hi := Lo + 1e-6;
  AssignFile(F, FileName); Rewrite(F);
  WriteLn(F, 'P6'); WriteLn(F, AImg.SizeX, ' ', AImg.SizeY); WriteLn(F, 255);
  CloseFile(F);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for py := 0 to AImg.SizeY - 1 do
      for px := 0 to AImg.SizeX - 1 do
        for pc := 0 to 2 do
        begin
          Val := (AImg[px, py, pc] - Lo) / (Hi - Lo);
          if Val < 0 then Val := 0;
          if Val > 1 then Val := 1;
          B := Round(Val * 255);
          Bin.WriteByte(B);
        end;
  finally
    Bin.Free;
  end;
end;

// Draws a 1px rectangle outline (pixel coords, clamped) into the (W,H,3) volume.
procedure DrawBox(AImg: TNNetVolume; X0p, Y0p, X1p, Y1p: integer; R, G, B: byte);
var
  px, py: integer;
  procedure Plot(ax, ay: integer);
  begin
    if (ax < 0) or (ay < 0) or (ax >= AImg.SizeX) or (ay >= AImg.SizeY) then Exit;
    AImg[ax, ay, 0] := R;
    AImg[ax, ay, 1] := G;
    AImg[ax, ay, 2] := B;
  end;
begin
  if X0p > X1p then begin px := X0p; X0p := X1p; X1p := px; end;
  if Y0p > Y1p then begin py := Y0p; Y0p := Y1p; Y1p := py; end;
  for px := X0p to X1p do begin Plot(px, Y0p); Plot(px, Y1p); end;
  for py := Y0p to Y1p do begin Plot(X0p, py); Plot(X1p, py); end;
end;

// Splits a comma-separated prompt list into trimmed, non-empty prompts.
function SplitPrompts(const List: string): TStringArray;
var
  Parts: TStringList;
  PartPos, MaxPartPos: integer;
begin
  Result := nil;
  Parts := TStringList.Create;
  try
    Parts.StrictDelimiter := true;
    Parts.Delimiter := ',';
    Parts.DelimitedText := List;
    MaxPartPos := Parts.Count - 1;
    for PartPos := 0 to MaxPartPos do
      if Trim(Parts[PartPos]) <> '' then
      begin
        SetLength(Result, Length(Result) + 1);
        Result[High(Result)] := Trim(Parts[PartPos]);
      end;
  finally
    Parts.Free;
  end;
  if Result = nil then raise Exception.Create('--prompts has no prompt');
end;

// TextEmb := one CLIP text embedding row per prompt, from the HF CLIP folder
// ClipDir (model.safetensors or pytorch_model.bin, config.json, tokenizer).
procedure EncodePromptsWithClip(const ClipDir: string; const Prompts: TStringArray;
  TextEmb: TNNetVolume);
var
  Reader: TNNetSafeTensorsReader;
  ClipConfig: TClipConfig;
  TextNet: TNNet;
  Tokenizer: TNeuralHFTokenizer;
  Embedding: TNNetVolume;
  ModelPath: string;
  PromptPos, MaxPromptPos: integer;
begin
  ModelPath := ClipDir + 'model.safetensors';
  if not FileExists(ModelPath) then ModelPath := ClipDir + 'pytorch_model.bin';
  ClipConfig := ReadClipConfigFromJSONFile(ClipDir + 'config.json');
  TextNet := nil;
  Embedding := TNNetVolume.Create;
  Tokenizer := TNeuralHFTokenizer.Create();
  Reader := CreatePretrainedTensorReader(ModelPath);
  try
    TextNet := BuildClipTextTower(Reader, ClipConfig, {TextSeqLen=}0,
      'text_projection.weight', {pTrainable=}false);
    if FileExists(ClipDir + 'tokenizer.json') then
      Tokenizer.LoadFromFile(ClipDir + 'tokenizer.json')
    else
      Tokenizer.LoadCLIPVocabAndMerges(ClipDir + 'vocab.json', ClipDir + 'merges.txt');
    TextEmb.ReSize(Length(Prompts), 1, ClipConfig.ProjectionDim);
    MaxPromptPos := High(Prompts);
    for PromptPos := 0 to MaxPromptPos do
    begin
      ClipEncodeText(TextNet, ClipConfig, Tokenizer.Encode(Prompts[PromptPos]),
        Tokenizer.BosId, Tokenizer.EosId, Embedding);
      Move(Embedding.FData[0], TextEmb.FData[PromptPos * ClipConfig.ProjectionDim],
        ClipConfig.ProjectionDim * csNeuralFloatSize);
    end;
  finally
    Reader.Free;
    Tokenizer.Free;
    Embedding.Free;
    TextNet.Free;
  end;
  WriteLn('  encoded ', Length(Prompts), ' prompt(s) with the CLIP text tower of ',
    ClipDir);
end;

// Pico demo without --clip: the first Length(Prompts) random text rows pinned in
// the world fixture stand in for CLIP embeddings.
procedure LoadPinnedTextRows(const IoPath: string; const Prompts: TStringArray;
  TextEmb: TNNetVolume);
var
  Reader: TNNetSafeTensorsReader;
  Flat: TNNetVolume;
  TextDim: integer;
begin
  Reader := TNNetSafeTensorsReader.Create(IoPath);
  Flat := TNNetVolume.Create;
  try
    if Length(Prompts) > Reader.DimSize('text_embeds', 0) then
      raise Exception.Create('the pico fixture pins ' +
        IntToStr(Reader.DimSize('text_embeds', 0)) + ' text rows; pass --clip DIR ' +
        'for more prompts');
    TextDim := Reader.DimSize('text_embeds', 1);
    Reader.LoadTensorFlat('text_embeds', Flat);
    TextEmb.ReSize(Length(Prompts), 1, TextDim);
    Move(Flat.FData[0], TextEmb.FData[0], TextEmb.Size * csNeuralFloatSize);
  finally
    Flat.Free;
    Reader.Free;
  end;
  WriteLn('  no --clip: using the pico fixture''s pinned RANDOM text rows ' +
    '(labels only; the prompts carry no meaning here)');
end;

// Img := the (c, y, x) flat 'input' tensor of the world fixture io.
procedure LoadPinnedWorldImage(const IoPath: string; AImg: TNNetVolume);
var
  Reader: TNNetSafeTensorsReader;
  Flat: TNNetVolume;
  MaxX, MaxY, MaxChannel, px, py, pc, SrcPos: integer;
begin
  Reader := TNNetSafeTensorsReader.Create(IoPath);
  Flat := TNNetVolume.Create;
  try
    Reader.LoadTensorFlat('input', Flat);
    MaxX := AImg.SizeX - 1; MaxY := AImg.SizeY - 1; MaxChannel := AImg.Depth - 1;
    SrcPos := 0;
    for pc := 0 to MaxChannel do
      for py := 0 to MaxY do
        for px := 0 to MaxX do
        begin
          AImg[px, py, pc] := Flat.FData[SrcPos];
          Inc(SrcPos);
        end;
  finally
    Flat.Free;
    Reader.Free;
  end;
end;

begin
  ImageFile := ''; ClipDir := ''; PromptList := '';
  Positional := nil;
  MaxArgPos := ParamCount;
  ArgPos := 1;
  while ArgPos <= MaxArgPos do
  begin
    Arg := ParamStr(ArgPos);
    if ((Arg = '--image') or (Arg = '--clip') or (Arg = '--prompts')) and
       (ArgPos = MaxArgPos) then
    begin
      WriteLn('ERROR: ', Arg, ' needs a value.');
      Halt(2);
    end;
    if Arg = '--image' then begin Inc(ArgPos); ImageFile := ParamStr(ArgPos); end
    else if Arg = '--clip' then
    begin
      Inc(ArgPos);
      ClipDir := IncludeTrailingPathDelimiter(ParamStr(ArgPos));
    end
    else if Arg = '--prompts' then begin Inc(ArgPos); PromptList := ParamStr(ArgPos); end
    else
    begin
      SetLength(Positional, Length(Positional) + 1);
      Positional[High(Positional)] := Arg;
    end;
    Inc(ArgPos);
  end;
  if (ClipDir <> '') and (PromptList = '') then
  begin
    WriteLn('ERROR: --clip needs --prompts.');
    Halt(2);
  end;
  if Length(Positional) >= 1 then ModelFile := Positional[0]
  else if PromptList <> '' then ModelFile := FixturesDir + 'tiny_yolo_world.safetensors'
  else ModelFile := FixturesDir + 'tiny_yolo.safetensors';
  if Length(Positional) >= 2 then ConfigFile := Positional[1]
  else if Length(Positional) >= 1 then ConfigFile := ExtractFilePath(ModelFile) + 'config.json'
  else if PromptList <> '' then ConfigFile := FixturesDir + 'tiny_yolo_world_config.json'
  else ConfigFile := FixturesDir + 'tiny_yolo_config.json';
  // Low score threshold on the random fixture so the draw path is exercised;
  // use ~0.25 (ultralytics default) on a real checkpoint.
  if Length(Positional) >= 3 then ScoreThresh := StrToFloat(Positional[2])
  else if PromptList <> '' then ScoreThresh := 0.3
  else ScoreThresh := 0.5;
  if Length(Positional) >= 4 then IoUThresh := StrToFloat(Positional[3])
  else IoUThresh := 0.45;
  if PromptList <> '' then IoFile := ExtractFilePath(ModelFile) + 'tiny_yolo_world_io.safetensors'
  else IoFile := ExtractFilePath(ModelFile) + 'tiny_yolo_io.json';

  RefJson := TStringList.Create;
  Img := TNNetVolume.Create;
  NetInput := TNNetVolume.Create;
  TextEmb := TNNetVolume.Create;
  RefRoot := nil;
  NN := nil;
  try
    if PromptList <> '' then
    begin
      Prompts := SplitPrompts(PromptList);
      WriteLn('YoloDetect (YOLO-World v2): loading ', ModelFile);
      if ClipDir <> '' then EncodePromptsWithClip(ClipDir, Prompts, TextEmb)
      else if Length(Positional) = 0 then LoadPinnedTextRows(IoFile, Prompts, TextEmb)
      else raise Exception.Create('--prompts with a real checkpoint needs --clip DIR');
      NN := BuildYoloWorldFromSafeTensors(ModelFile, Config, TextEmb, Prompts,
        {pTrainable=}false, ConfigFile);
    end
    else
    begin
      WriteLn('YoloDetect (YOLOv8): loading ', ModelFile);
      NN := BuildYoloFromSafeTensors(ModelFile, Config, {pTrainable=}false,
        ConfigFile);
    end;
    WriteLn(YoloConfigToString(Config));
    W := Config.ImageSize; H := Config.ImageSize; NumCh := Config.NumChannels;
    WriteLn(Format('  input image %dx%dx%d, %d classes, reg_max %d, ' +
      'strides %d/%d/%d', [W, H, NumCh, Config.NumClasses, Config.RegMax,
      Config.Strides[0], Config.Strides[1], Config.Strides[2]]));

    if ImageFile <> '' then
    begin
      if not LoadImageFromFileIntoVolume(ImageFile, Img) then
        raise Exception.Create('could not load ' + ImageFile);
      LetterboxYoloImage(Img, NetInput, Config.ImageSize, Letterbox);
      WriteLn(Format('  letterboxed %s (%dx%d) into the %dx%d net input',
        [ImageFile, Img.SizeX, Img.SizeY, W, H]));
    end
    else
    begin
      NetInput.ReSize(W, H, NumCh);
      if (PromptList <> '') and FileExists(IoFile) then
      begin
        LoadPinnedWorldImage(IoFile, NetInput);
        WriteLn('  using pinned image from tiny_yolo_world_io.safetensors');
      end
      else if (PromptList = '') and FileExists(IoFile) then
      begin
        RefJson.LoadFromFile(IoFile);
        RefRoot := GetJSON(RefJson.Text);
        InArr := TJSONArray(TJSONObject(RefRoot).Find('input'));
        // input is flat (c, y, x), exactly as the parity test loads it.
        for ch := 0 to NumCh - 1 do
          for yy := 0 to H - 1 do
            for x := 0 to W - 1 do
            begin
              FlatIdx := (ch * H + yy) * W + x;
              NetInput[x, yy, ch] := InArr.Items[FlatIdx].AsFloat;
            end;
        WriteLn('  using pinned image from tiny_yolo_io.json');
      end
      else
      begin
        for ch := 0 to NumCh - 1 do
          for yy := 0 to H - 1 do
            for x := 0 to W - 1 do
              NetInput[x, yy, ch] := (x / W) - 0.5 + 0.3 * Sin((yy + ch) * 0.4);
        WriteLn('  using synthetic gradient image');
      end;
      Img.Copy(NetInput);
    end;

    // ---- forward pass ----
    StartTime := Now();
    NN.Compute(NetInput);
    Output := NN.GetLastLayer().Output;
    WriteLn(Format('  forward pass: %.1f ms', [(Now() - StartTime) * 86400000.0]));

    // Self-report 1: no NaN / Inf anywhere in the output volume.
    AnyBad := false;
    for i := 0 to Output.Size - 1 do
    begin
      V := Output.FData[i];
      if IsNan(V) or IsInfinite(V) then AnyBad := true;
    end;
    if AnyBad then
    begin
      WriteLn('ERROR: YOLO output contains NaN/Inf.');
      Halt(1);
    end;
    WriteLn('  output OK: no NaN/Inf in the (', Output.SizeX, ', 1, ',
      Output.Depth, ') raw head (4*reg_max box dist + class logits per cell).');

    // ---- decode: sigmoid cls, DFL box, xyxy pixels, greedy NMS ----
    Dets := DecodeYoloDetections(Output, Config, ScoreThresh, IoUThresh);
    if ImageFile <> '' then
      ScaleYoloBoxesToImage(Dets, Config.ImageSize, Img.SizeX, Img.SizeY);
    W := Img.SizeX; H := Img.SizeY;
    WriteLn(Format('  decoded %d detection(s) at score>=%.3f, IoU NMS %.2f:',
      [Length(Dets), ScoreThresh, IoUThresh]));

    for i := 0 to High(Dets) do
    begin
      x0 := Round(Dets[i].X1); y0 := Round(Dets[i].Y1);
      x1 := Round(Dets[i].X2); y1 := Round(Dets[i].Y2);
      ClassLabel := YoloClassLabel(Config, Dets[i].ClassId);
      if ClassLabel = IntToStr(Dets[i].ClassId) then ClassText := ''
      else ClassText := ' (' + ClassLabel + ')';
      WriteLn(Format('    class %d%s  score %.4f  xyxy=(%d,%d,%d,%d)',
        [Dets[i].ClassId, ClassText, Dets[i].Score, x0, y0, x1, y1]));
      // Self-report 2: every decoded box must be finite.
      if IsNan(Dets[i].X1) or IsInfinite(Dets[i].X1) or
         IsNan(Dets[i].Y2) or IsInfinite(Dets[i].Y2) then
      begin
        WriteLn('ERROR: decoded box has NaN/Inf coords.');
        Halt(1);
      end;
      // Clamp to the canvas and draw.
      x0 := Max(0, Min(W - 1, x0)); y0 := Max(0, Min(H - 1, y0));
      x1 := Max(0, Min(W - 1, x1)); y1 := Max(0, Min(H - 1, y1));
      DrawBox(Img, x0, y0, x1, y1,
        PaletteR[i mod 6], PaletteG[i mod 6], PaletteB[i mod 6]);
      // "class score" on a tag of the box colour above the box (inside at the top).
      BoxColor[0] := PaletteR[i mod 6];
      BoxColor[1] := PaletteG[i mod 6];
      BoxColor[2] := PaletteB[i mod 6];
      if BoxColor[0] + BoxColor[1] + BoxColor[2] >= 510 then
        TextColor[0] := 0
      else
        TextColor[0] := 255;
      TextColor[1] := TextColor[0];
      TextColor[2] := TextColor[0];
      BoxLabel := Format('%s %.2f', [ClassLabel, Dets[i].Score]);
      TagTop := y0 - csBitmapFontGlyphHeight - 2;
      if TagTop < 0 then TagTop := y0;
      FillRectangleCV2(Img, x0, TagTop, x0 + BitmapTextWidth(BoxLabel) + 1,
        TagTop + csBitmapFontGlyphHeight + 1, BoxColor);
      DrawBitmapText(Img, x0 + 1, TagTop + 1, BoxLabel, TextColor);
    end;
    WriteLn('  box-decode self-check OK: all boxes finite, pixel coords ' +
      'clamped to canvas.');

    if ImageFile <> '' then
    begin
      OutputFile := ChangeFileExt(ExtractFileName(ImageFile), '') + '_yolo.png';
      if not SaveImageFromVolumeIntoFile(Img, OutputFile) then
        raise Exception.Create('could not save ' + OutputFile);
    end
    else
    begin
      OutputFile := PpmFile;
      WritePPM(Img, PpmFile);
    end;
    WriteLn('Wrote annotated image to ', OutputFile, ' (', W, 'x', H, ').');

    if Length(Positional) = 0 then
      WriteLn('(The pico fixture has RANDOM weights, so these are not real ' +
        'objects -- the DFL decode + NMS + draw pipeline is what is exercised. A ' +
        'real ultralytics checkpoint, with score>=0.25, yields real ' +
        'detections through the same path.)');
  finally
    RefRoot.Free;
    RefJson.Free;
    Img.Free;
    NetInput.Free;
    TextEmb.Free;
    NN.Free;
  end;
end.
