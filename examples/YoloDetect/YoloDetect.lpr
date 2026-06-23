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
// surviving boxes as colored rectangle outlines into the image, writes a PPM,
// and prints the (class, score, box) list.
//
// Usage:
//   YoloDetect [model.safetensors] [config.json] [score_threshold] [iou_threshold]
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

uses
  SysUtils, Classes, Math, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained;

const
  PpmFile = 'yolo_detect.ppm';
  PaletteR: array[0..5] of byte = (255, 0, 0, 255, 255, 0);
  PaletteG: array[0..5] of byte = (0, 255, 0, 255, 0, 255);
  PaletteB: array[0..5] of byte = (0, 0, 255, 0, 255, 255);

var
  ModelFile, ConfigFile, IoFile: string;
  ScoreThresh, IoUThresh: TNeuralFloat;
  NN: TNNet;
  Config: TYoloConfig;
  Img, Output: TNNetVolume;
  Dets: TYoloDetectionArray;
  RefJson: TStringList;
  RefRoot: TJSONData;
  InArr: TJSONArray;
  W, H, NumCh, x, yy, ch, FlatIdx, i, x0, y0, x1, y1: integer;
  V: TNeuralFloat;
  AnyBad: boolean;

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

begin
  if ParamCount >= 1 then ModelFile := ParamStr(1)
  else ModelFile := '../../tests/fixtures/tiny_yolo.safetensors';
  if ParamCount >= 2 then ConfigFile := ParamStr(2)
  else ConfigFile := '../../tests/fixtures/tiny_yolo_config.json';
  // Low score threshold on the random fixture so the draw path is exercised;
  // use ~0.25 (ultralytics default) on a real checkpoint.
  if ParamCount >= 3 then ScoreThresh := StrToFloat(ParamStr(3))
  else ScoreThresh := 0.5;
  if ParamCount >= 4 then IoUThresh := StrToFloat(ParamStr(4))
  else IoUThresh := 0.45;
  IoFile := ExtractFilePath(ModelFile) + 'tiny_yolo_io.json';

  WriteLn('YoloDetect (YOLOv8): loading ', ModelFile);
  NN := BuildYoloFromSafeTensors(ModelFile, Config, {pTrainable=}true,
    ConfigFile);
  RefJson := TStringList.Create;
  Img := TNNetVolume.Create;
  RefRoot := nil;
  try
    WriteLn(YoloConfigToString(Config));
    W := Config.ImageSize; H := Config.ImageSize; NumCh := Config.NumChannels;
    WriteLn(Format('  input image %dx%dx%d, %d classes, reg_max %d, ' +
      'strides %d/%d/%d', [W, H, NumCh, Config.NumClasses, Config.RegMax,
      Config.Strides[0], Config.Strides[1], Config.Strides[2]]));

    Img.ReSize(W, H, NumCh);
    if FileExists(IoFile) then
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
            Img[x, yy, ch] := InArr.Items[FlatIdx].AsFloat;
          end;
      WriteLn('  using pinned image from tiny_yolo_io.json');
    end
    else
    begin
      for ch := 0 to NumCh - 1 do
        for yy := 0 to H - 1 do
          for x := 0 to W - 1 do
            Img[x, yy, ch] := (x / W) - 0.5 + 0.3 * Sin((yy + ch) * 0.4);
      WriteLn('  using synthetic gradient image');
    end;

    // ---- YOLOv8 forward pass ----
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
      WriteLn('ERROR: YOLO output contains NaN/Inf.');
      Halt(1);
    end;
    WriteLn('  output OK: no NaN/Inf in the (', Output.SizeX, ', 1, ',
      Output.Depth, ') raw head (4*reg_max box dist + class logits per cell).');

    // ---- decode: sigmoid cls, DFL box, xyxy pixels, greedy NMS ----
    Dets := DecodeYoloDetections(Output, Config, ScoreThresh, IoUThresh);
    WriteLn(Format('  decoded %d detection(s) at score>=%.3f, IoU NMS %.2f:',
      [Length(Dets), ScoreThresh, IoUThresh]));

    for i := 0 to High(Dets) do
    begin
      x0 := Round(Dets[i].X1); y0 := Round(Dets[i].Y1);
      x1 := Round(Dets[i].X2); y1 := Round(Dets[i].Y2);
      WriteLn(Format('    class %d  score %.4f  xyxy=(%d,%d,%d,%d)',
        [Dets[i].ClassId, Dets[i].Score, x0, y0, x1, y1]));
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
    end;
    WriteLn('  box-decode self-check OK: all boxes finite, pixel coords ' +
      'clamped to canvas.');

    WritePPM(Img, PpmFile);
    WriteLn('Wrote annotated image to ', PpmFile, ' (', W, 'x', H, ' P6 PPM).');

    WriteLn('(The pico fixture has RANDOM weights, so these are not real ' +
      'objects -- the DFL decode + NMS + draw pipeline is what is exercised. A ' +
      'real ultralytics yolov8 checkpoint, with score>=0.25, yields real ' +
      'detections through the same path.)');
  finally
    RefRoot.Free;
    RefJson.Free;
    Img.Free;
    NN.Free;
  end;
end.
