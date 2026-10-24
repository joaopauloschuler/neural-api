program ObjectDetection;
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

// ObjectDetection -- the demo for the DETR-family object-detection importers,
// inference only:
//   * DETR (BuildDetrFromSafeTensors, facebook/detr-resnet-50): a ResNet
//     backbone + transformer encoder-decoder + learned object queries; decoded
//     with DecodeDetrDetections (softmax, drop the "no-object" slot, threshold).
//   * RT-DETR with --rtdetr (BuildRtDetrFromSafeTensors, PekingU/rtdetr_r18vd ..
//     r101vd, and the RT-DETRv2 PekingU/rtdetr_v2_*): a ResNet-vd backbone + the hybrid encoder (AIFI + CSP FPN/PAN) +
//     top-K query selection + a multi-scale deformable-attention decoder with
//     iterative box refinement; decoded with DecodeRtDetrDetections (sigmoid
//     scores, top-K over query x class, threshold, no NMS).
//   * D-FINE with --dfine (BuildDFineFromSafeTensors, ustc-community/dfine-*):
//     an HGNetV2 backbone + the RT-DETR-style hybrid encoder + top-K query
//     selection + a deformable decoder whose boxes come from Fine-grained
//     Distribution Refinement; decoded with DecodeRtDetrDetections as RT-DETR.
// It runs the forward pass on ONE image, draws the surviving boxes (with a
// "class score" tag in RT-DETR and D-FINE mode) and writes the annotated image.
//
// Usage:
//   ObjectDetection [model.safetensors] [config.json] [threshold]
//   ObjectDetection --rtdetr|--dfine [--image FILE] [--size N]
//                   [model.safetensors] [config.json] [threshold]
//
// --image resizes a photo to the N x N net input (default 640, as
// RTDetrImageProcessor: PIL bilinear, / 255), scales the boxes back to the
// photo and writes <name>_rtdetr.png or <name>_dfine.png. With no model
// argument each mode loads its committed pico fixture (tests/fixtures/
// tiny_detr.*, tiny_rtdetr.* or tiny_dfine.*), a tiny random model used by the
// parity tests, so the demo runs offline.
//
// HONEST NOTE: the pico fixtures have RANDOM weights, so their "detections" are
// not meaningful objects -- they exercise the decode + draw pipeline and
// SELF-REPORT: no NaN/Inf in the output and every decoded box within the
// sigmoid [0,1] range. A real checkpoint produces real detections through the
// same path. Pure CPU, < 1 s on the fixtures.

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  SysUtils, Classes, Math, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained, neuralimageresize,
  neuraldatasets, neuralsafetensors;

const
  PpmFile = 'object_detection.ppm';
  FixturesDir = '../../tests/fixtures/';
  // A small fixed palette to color successive boxes.
  PaletteR: array[0..5] of byte = (255, 0, 0, 255, 255, 0);
  PaletteG: array[0..5] of byte = (0, 255, 0, 255, 0, 255);
  PaletteB: array[0..5] of byte = (0, 0, 255, 0, 255, 255);

var
  ModelFile, ConfigFile, IoFile, ImageFile, Arg: string;
  Positional: array of string;
  UseRtDetr, UseDFine: boolean;
  RtDetrSize, ArgPos, MaxArgPos: integer;
  Threshold: TNeuralFloat;
  NN: TNNet;
  Config: TDetrConfig;
  Img, Output: TNNetVolume;
  Dets: TDetrDetectionArray;
  RefJson: TStringList;
  RefRoot: TJSONData;
  CasesArr, InArr: TJSONArray;
  CaseObj: TJSONObject;
  W, H, NumCh, x, yy, ch, FlatIdx, i, q, c: integer;
  V: TNeuralFloat;
  AnyBad: boolean;
  DispThreshold: TNeuralFloat;

// Renders the (W,H,3) volume to a P6 PPM. Channel values are assumed to be the
// raw, possibly normalized, image values; they are min-max stretched to 0..255
// just so the boxes are visible over whatever synthetic content is present.
procedure WritePPM(AImg: TNNetVolume; const FileName: string);
var
  F: TextFile; Bin: TFileStream;
  px, py, pc: integer; Lo, Hi, Val: TNeuralFloat; B: byte;
begin
  // Min-max range over the 3 color channels for display stretch.
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

// Draws a 1px rectangle outline (pixel coords, clamped) of the given color
// directly into the (W,H,3) volume. Sets the color into all 3 channels.
procedure DrawBox(AImg: TNNetVolume; X0, Y0, X1, Y1: integer; R, G, B: byte);
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
  if X0 > X1 then begin px := X0; X0 := X1; X1 := px; end;
  if Y0 > Y1 then begin py := Y0; Y0 := Y1; Y1 := py; end;
  for px := X0 to X1 do begin Plot(px, Y0); Plot(px, Y1); end;
  for py := Y0 to Y1 do begin Plot(X0, py); Plot(X1, py); end;
end;

// Labels a box with "name score" on a tag of the box colour.
procedure DrawBoxTag(AImg: TNNetVolume; X0, Y0: integer; const Text: string;
  ColorIndex: integer);
var
  BoxColor, TextColor: array[0..2] of TNeuralFloat;
  TagTop: integer;
begin
  BoxColor[0] := PaletteR[ColorIndex mod 6];
  BoxColor[1] := PaletteG[ColorIndex mod 6];
  BoxColor[2] := PaletteB[ColorIndex mod 6];
  if BoxColor[0] + BoxColor[1] + BoxColor[2] >= 510 then TextColor[0] := 0
  else TextColor[0] := 255;
  TextColor[1] := TextColor[0];
  TextColor[2] := TextColor[0];
  TagTop := Y0 - csBitmapFontGlyphHeight - 2;
  if TagTop < 0 then TagTop := Y0;
  FillRectangleCV2(AImg, X0, TagTop, X0 + BitmapTextWidth(Text) + 1,
    TagTop + csBitmapFontGlyphHeight + 1, BoxColor);
  DrawBitmapText(AImg, X0 + 1, TagTop + 1, Text, TextColor);
end;

// RT-DETR and D-FINE mode (IsDFine): forward, DecodeRtDetrDetections, draw,
// write.
procedure RunQueryDetector(IsDFine: boolean);
var
  RtConfig: TRtDetrConfig;
  DFineConfig: TDFineConfig;
  Net: TNNet;
  Photo, NetInput, Pinned, NetOutput: TNNetVolume;
  Reader: TNNetSafeTensorsReader;
  Detections: TDetrDetectionArray;
  PhotoW, PhotoH, DetPos, BoxX0, BoxY0, BoxX1, BoxY1, ValuePos, MaxValuePos: integer;
  Name, OutputFile, Fixture, ModeName, ModeSuffix: string;
  ScoreThreshold: TNeuralFloat;
  StartTime: TDateTime;
begin
  if IsDFine then
  begin
    Fixture := 'tiny_dfine';
    ModeName := 'D-FINE';
    ModeSuffix := '_dfine';
  end
  else
  begin
    Fixture := 'tiny_rtdetr';
    ModeName := 'RT-DETR';
    ModeSuffix := '_rtdetr';
  end;
  if Length(Positional) >= 1 then ModelFile := Positional[0]
  else ModelFile := FixturesDir + Fixture + '.safetensors';
  if Length(Positional) >= 2 then ConfigFile := Positional[1]
  else ConfigFile := ExtractFilePath(ModelFile) +
    BoolToStr(Length(Positional) = 0, Fixture + '_config.json', 'config.json');
  // transformers' post-process default is 0.5; the random fixture uses 0.75.
  if Length(Positional) >= 3 then ScoreThreshold := StrToFloat(Positional[2])
  else if Length(Positional) = 0 then ScoreThreshold := 0.75
  else ScoreThreshold := 0.5;
  Photo := TNNetVolume.Create;
  NetInput := TNNetVolume.Create;
  Pinned := TNNetVolume.Create;
  Net := nil;
  try
    if ImageFile <> '' then
    begin
      if not LoadImageFromFileIntoVolume(ImageFile, Photo) then
        raise Exception.Create('could not load ' + ImageFile);
      // RTDetrImageProcessor: PIL bilinear on the 8-bit image, then / 255.
      ResizeImagePIL(Photo, NetInput, RtDetrSize, RtDetrSize, irfBilinear, true);
      NetInput.Mul(1 / 255);
    end
    else
    begin
      // The pinned 64 x 64 fixture image (0..1), drawn on at 0..255.
      Reader := TNNetSafeTensorsReader.Create(FixturesDir + Fixture + '_io.safetensors');
      try
        Reader.LoadTensorFlat('image', Pinned);
        NetInput.ReSize(Reader.DimSize('image', 1), Reader.DimSize('image', 0), 3);
      finally
        Reader.Free;
      end;
      NetInput.CopyNoChecks(Pinned);
      RtDetrSize := NetInput.SizeX;
      Photo.Copy(NetInput);
      Photo.Mul(255);
    end;
    WriteLn('ObjectDetection (', ModeName, '): loading ', ModelFile);
    if IsDFine then
    begin
      Net := BuildDFineFromSafeTensors(ModelFile, DFineConfig, {pTrainable=}false,
        ConfigFile, NetInput.SizeX, NetInput.SizeY);
      WriteLn(DFineConfigToString(DFineConfig));
      RtConfig := DFineConfig.Detector;
    end
    else
    begin
      Net := BuildRtDetrFromSafeTensors(ModelFile, RtConfig, {pTrainable=}false,
        ConfigFile, NetInput.SizeX, NetInput.SizeY);
      WriteLn(RtDetrConfigToString(RtConfig));
    end;
    StartTime := Now();
    Net.Compute(NetInput);
    NetOutput := Net.GetLastLayer().Output;
    WriteLn(Format('  forward pass: %.1f s', [(Now() - StartTime) * 86400.0]));
    MaxValuePos := NetOutput.Size - 1;
    for ValuePos := 0 to MaxValuePos do
      if IsNan(NetOutput.FData[ValuePos]) or IsInfinite(NetOutput.FData[ValuePos]) then
      begin
        WriteLn('ERROR: ', ModeName, ' output contains NaN/Inf.');
        Halt(1);
      end;
    Detections := DecodeRtDetrDetections(NetOutput, RtConfig.NumLabels,
      ScoreThreshold);
    WriteLn(Format('  %d detection(s) with score > %.2f:',
      [Length(Detections), ScoreThreshold]));
    PhotoW := Photo.SizeX;
    PhotoH := Photo.SizeY;
    for DetPos := 0 to High(Detections) do
    begin
      if (Detections[DetPos].Cx < 0) or (Detections[DetPos].Cx > 1) or
         (Detections[DetPos].W < 0) or (Detections[DetPos].W > 1) then
      begin
        WriteLn('ERROR: decoded box outside the sigmoid [0,1] range.');
        Halt(1);
      end;
      // cxcywh in [0, 1] -> photo xyxy (post_process_object_detection).
      BoxX0 := Round((Detections[DetPos].Cx - 0.5 * Detections[DetPos].W) * PhotoW);
      BoxY0 := Round((Detections[DetPos].Cy - 0.5 * Detections[DetPos].H) * PhotoH);
      BoxX1 := Round((Detections[DetPos].Cx + 0.5 * Detections[DetPos].W) * PhotoW);
      BoxY1 := Round((Detections[DetPos].Cy + 0.5 * Detections[DetPos].H) * PhotoH);
      Name := RtConfig.ClassNames[Detections[DetPos].ClassId];
      if Name = '' then Name := IntToStr(Detections[DetPos].ClassId);
      WriteLn(Format('    %-12s score %.4f  xyxy=(%d,%d,%d,%d)',
        [Name, Detections[DetPos].Score, BoxX0, BoxY0, BoxX1, BoxY1]));
      BoxX0 := Max(0, Min(PhotoW - 1, BoxX0));
      BoxY0 := Max(0, Min(PhotoH - 1, BoxY0));
      BoxX1 := Max(0, Min(PhotoW - 1, BoxX1));
      BoxY1 := Max(0, Min(PhotoH - 1, BoxY1));
      DrawBox(Photo, BoxX0, BoxY0, BoxX1, BoxY1, PaletteR[DetPos mod 6],
        PaletteG[DetPos mod 6], PaletteB[DetPos mod 6]);
      DrawBoxTag(Photo, BoxX0, BoxY0, Format('%s %.2f',
        [Name, Detections[DetPos].Score]), DetPos);
    end;
    if ImageFile <> '' then
      OutputFile := ChangeFileExt(ExtractFileName(ImageFile), '') + ModeSuffix + '.png'
    else
      OutputFile := 'object_detection' + ModeSuffix + '.png';
    if not SaveImageFromVolumeIntoFile(Photo, OutputFile) then
      raise Exception.Create('could not save ' + OutputFile);
    WriteLn('Wrote annotated image to ', OutputFile, ' (', PhotoW, 'x', PhotoH, ').');
    if Length(Positional) = 0 then
      WriteLn('(The pico fixture has RANDOM weights, so these are not real ' +
        'objects. A real checkpoint (', BoolToStr(IsDFine,
        'ustc-community/dfine-nano-coco', 'PekingU/rtdetr_r18vd'),
        ') with --image yields real detections through the same path.)');
  finally
    Net.Free;
    Pinned.Free;
    NetInput.Free;
    Photo.Free;
  end;
end;

begin
  ImageFile := '';
  UseRtDetr := false;
  UseDFine := false;
  RtDetrSize := 640;
  Positional := nil;
  MaxArgPos := ParamCount;
  ArgPos := 1;
  while ArgPos <= MaxArgPos do
  begin
    Arg := ParamStr(ArgPos);
    if ((Arg = '--image') or (Arg = '--size')) and (ArgPos = MaxArgPos) then
    begin
      WriteLn('ERROR: ', Arg, ' needs a value.');
      Halt(2);
    end;
    if Arg = '--rtdetr' then UseRtDetr := true
    else if Arg = '--dfine' then UseDFine := true
    else if Arg = '--image' then begin Inc(ArgPos); ImageFile := ParamStr(ArgPos); end
    else if Arg = '--size' then begin Inc(ArgPos); RtDetrSize := StrToInt(ParamStr(ArgPos)); end
    else
    begin
      SetLength(Positional, Length(Positional) + 1);
      Positional[High(Positional)] := Arg;
    end;
    Inc(ArgPos);
  end;
  if UseRtDetr and UseDFine then
  begin
    WriteLn('ERROR: pass one of --rtdetr and --dfine.');
    Halt(2);
  end;
  if (ImageFile <> '') and not (UseRtDetr or UseDFine) then
  begin
    WriteLn('ERROR: --image needs --rtdetr or --dfine.');
    Halt(2);
  end;
  if UseRtDetr or UseDFine then
  begin
    RunQueryDetector(UseDFine);
    Exit;
  end;
  if Length(Positional) >= 1 then ModelFile := Positional[0]
  else ModelFile := FixturesDir + 'tiny_detr.safetensors';
  if Length(Positional) >= 2 then ConfigFile := Positional[1]
  else ConfigFile := FixturesDir + 'tiny_detr_config.json';
  // Threshold defaults to 0 on the random fixture (so EVERY query is decoded and
  // the full draw path is exercised); use ~0.7 on a real checkpoint.
  if Length(Positional) >= 3 then Threshold := StrToFloat(Positional[2])
  else Threshold := 0.0;
  IoFile := ExtractFilePath(ModelFile) + 'tiny_detr_io.json';

  WriteLn('ObjectDetection (DETR): loading ', ModelFile);
  NN := BuildDetrFromSafeTensors(ModelFile, Config, {pTrainable=}true,
    ConfigFile);
  RefJson := TStringList.Create;
  Img := TNNetVolume.Create;
  RefRoot := nil;
  try
    WriteLn(DetrConfigToString(Config));
    W := Config.ImageSize; H := Config.ImageSize; NumCh := Config.NumChannels;
    WriteLn(Format('  input image %dx%dx%d, %d object queries, %d labels (+1 ' +
      'no-object)', [W, H, NumCh, Config.NumQueries, Config.NumLabels]));

    // Load the pinned image from the io fixture if present; else a synthetic
    // gradient with a couple of bright blocks (so a real checkpoint has content).
    Img.ReSize(W, H, NumCh);
    if FileExists(IoFile) then
    begin
      RefJson.LoadFromFile(IoFile);
      RefRoot := GetJSON(RefJson.Text);
      CasesArr := TJSONArray(TJSONObject(RefRoot).Find('cases'));
      CaseObj := TJSONObject(CasesArr.Items[0]);
      InArr := TJSONArray(CaseObj.Find('input'));
      // input is flat (y, x, c), exactly as the parity test loads it.
      for yy := 0 to H - 1 do
        for x := 0 to W - 1 do
          for ch := 0 to NumCh - 1 do
          begin
            FlatIdx := (yy * W + x) * NumCh + ch;
            Img.FData[FlatIdx] := InArr.Items[FlatIdx].AsFloat;
          end;
      WriteLn('  using pinned image from tiny_detr_io.json');
    end
    else
    begin
      for yy := 0 to H - 1 do
        for x := 0 to W - 1 do
          for ch := 0 to NumCh - 1 do
            Img.FData[(yy * W + x) * NumCh + ch] :=
              (x / W) - 0.5 + 0.3 * Sin((yy + ch) * 0.4);
      WriteLn('  using synthetic gradient image');
    end;

    // ---- DETR forward pass ----
    NN.Compute(Img);
    Output := NN.GetLastLayer().Output;

    // Self-report 1: no NaN / Inf anywhere in the output volume.
    AnyBad := false;
    for i := 0 to Output.Size - 1 do
    begin
      V := Output.FData[i];
      if IsNan(V) or IsInfinite(V) then AnyBad := true;
    end;
    if AnyBad then
    begin
      WriteLn('ERROR: DETR output contains NaN/Inf.');
      Halt(1);
    end;
    WriteLn('  output OK: no NaN/Inf in the (', Config.NumQueries, ', 1, ',
      Config.NumLabels + 1 + 4, ') logits+box tensor.');

    // ---- decode: softmax class logits, drop no-object, threshold, cxcywh ----
    Dets := DecodeDetrDetections(Output, Config.NumLabels, Threshold);
    WriteLn(Format('  decoded %d detection(s) at threshold %.3f:',
      [Length(Dets), Threshold]));

    // Draw the surviving boxes and print them. Self-report 2: every decoded box
    // converts to in-range pixel xyxy.
    for i := 0 to High(Dets) do
    begin
      // cxcywh in [0,1] -> pixel xyxy.
      x  := Round((Dets[i].Cx - Dets[i].W * 0.5) * W);   // x0
      yy := Round((Dets[i].Cy - Dets[i].H * 0.5) * H);   // y0
      q  := Round((Dets[i].Cx + Dets[i].W * 0.5) * W);   // x1
      c  := Round((Dets[i].Cy + Dets[i].H * 0.5) * H);   // y1
      WriteLn(Format('    class %d  score %.4f  cxcywh=(%.3f,%.3f,%.3f,%.3f)  ' +
        'xyxy=(%d,%d,%d,%d)', [Dets[i].ClassId, Dets[i].Score,
        Dets[i].Cx, Dets[i].Cy, Dets[i].W, Dets[i].H, x, yy, q, c]));
      // Assert in-range box-decode math (cxcywh is sigmoid -> [0,1], so pixel
      // coords must land within [-W..2W] worst case; clamp + assert sanity).
      if (Dets[i].Cx < 0) or (Dets[i].Cx > 1) or (Dets[i].Cy < 0) or
         (Dets[i].Cy > 1) or (Dets[i].W < 0) or (Dets[i].W > 1) or
         (Dets[i].H < 0) or (Dets[i].H > 1) then
      begin
        WriteLn('ERROR: decoded box outside the sigmoid [0,1] range.');
        Halt(1);
      end;
      // Clamp pixel coords to the canvas and draw.
      x  := Max(0, Min(W - 1, x));
      yy := Max(0, Min(H - 1, yy));
      q  := Max(0, Min(W - 1, q));
      c  := Max(0, Min(H - 1, c));
      DrawBox(Img, x, yy, q, c,
        PaletteR[i mod 6], PaletteG[i mod 6], PaletteB[i mod 6]);
    end;
    WriteLn('  box-decode self-check OK: all boxes within sigmoid [0,1], ' +
      'pixel coords clamped to canvas.');

    // ---- write the annotated image ----
    WritePPM(Img, PpmFile);
    WriteLn('Wrote annotated image to ', PpmFile, ' (', W, 'x', H, ' P6 PPM).');

    DispThreshold := 0.7;
    WriteLn('(The pico fixture has RANDOM weights, so these are not real ' +
      'objects -- the decode+draw pipeline is what is exercised. A real ' +
      'facebook/detr-resnet-50 checkpoint, e.g. with threshold ',
      DispThreshold:0:1, ', yields real detections through the same path.)');
  finally
    RefRoot.Free;
    RefJson.Free;
    Img.Free;
    NN.Free;
  end;
end.
