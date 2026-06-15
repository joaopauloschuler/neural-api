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

// ObjectDetection -- the demo for the landed DETR object-detection importer
// (BuildDetrFromSafeTensors, facebook/detr-resnet-50: a ResNet backbone + the
// DETR transformer encoder-decoder + learned object queries + 2-D sinusoidal
// spatial position embedding + sigmoid-cxcywh box head + class head, inference
// only, NO Hungarian matcher). It runs a full DETR forward pass on ONE image,
// DECODES the per-query predictions (softmax the class logits, drop the
// "no-object" slot, threshold on confidence, convert the normalized cxcywh box
// to pixel xyxy) with DecodeDetrDetections, DRAWS the surviving boxes as colored
// rectangle outlines into the image, and writes the result to a PPM. It also
// prints the (class, score, box) list to the console.
//
// Usage:
//   ObjectDetection [model.safetensors] [config.json] [threshold]
//
// With no arguments it loads the committed pico fixture
// (tests/fixtures/tiny_detr.*), a tiny random DetrForObjectDetection used by the
// parity tests, so the demo runs with NO download, fully OFFLINE. Point it at a
// real facebook/detr-resnet-50 checkpoint + config.json (+ optional confidence
// threshold) for the real thing.
//
// HONEST NOTE: the pico fixture has RANDOM weights, so the "detections" are not
// meaningful objects -- the demo's job is to exercise the full DETR
// decode + draw pipeline end to end and SELF-REPORT: it asserts no NaN/Inf in
// the network output and that every decoded box converts to in-range pixel
// coordinates. A real pretrained checkpoint produces real detections through the
// exact same path. Pure CPU, <1 s on the fixture.

{$mode objfpc}{$H+}

uses
  SysUtils, Classes, Math, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained;

const
  PpmFile = 'object_detection.ppm';
  // A small fixed palette to color successive boxes.
  PaletteR: array[0..5] of byte = (255, 0, 0, 255, 255, 0);
  PaletteG: array[0..5] of byte = (0, 255, 0, 255, 0, 255);
  PaletteB: array[0..5] of byte = (0, 0, 255, 0, 255, 255);

var
  ModelFile, ConfigFile, IoFile: string;
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

begin
  if ParamCount >= 1 then ModelFile := ParamStr(1)
  else ModelFile := '../../tests/fixtures/tiny_detr.safetensors';
  if ParamCount >= 2 then ConfigFile := ParamStr(2)
  else ConfigFile := '../../tests/fixtures/tiny_detr_config.json';
  // Threshold defaults to 0 on the random fixture (so EVERY query is decoded and
  // the full draw path is exercised); use ~0.7 on a real checkpoint.
  if ParamCount >= 3 then Threshold := StrToFloat(ParamStr(3))
  else Threshold := 0.0;
  IoFile := ExtractFilePath(ModelFile) + 'tiny_detr_io.json';

  WriteLn('ObjectDetection (DETR): loading ', ModelFile);
  NN := BuildDetrFromSafeTensors(ModelFile, Config, {pInferenceOnly=}false,
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
