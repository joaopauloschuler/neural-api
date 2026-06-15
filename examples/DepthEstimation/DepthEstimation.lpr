/// Monocular depth estimation with an imported DPT / Depth-Anything model.
///
/// Depth Anything (Yang et al. 2024) and the DPT family (Ranftl et al. 2021,
/// "Vision Transformers for Dense Prediction") pair a plain ViT / DINOv2
/// transformer backbone with a convolutional "reassemble + fusion" neck and a
/// small depth head: four intermediate encoder stages are projected and resized
/// into a feature pyramid, RefineNet-style additive fusion blocks merge them
/// coarse-to-fine, and a 3-conv head emits a SINGLE-channel per-pixel depth map
/// at the full input resolution. It is the repo's first dense per-pixel
/// REGRESSION import (a continuous depth value per pixel, not a class label).
///
/// This example loads the committed PICO Depth-Anything parity fixture
/// (tests/fixtures/tiny_dpt.safetensors, a tiny DINOv2-backbone model) with
/// BuildDPTFromSafeTensors, runs it on a small synthetic CPU image, and renders
/// the resulting depth map as an ASCII grayscale ramp (near = bright glyph, far
/// = dark glyph) at the full input resolution. A real run would load
/// depth-anything/Depth-Anything-V2-Small-hf the same way and process a
/// photograph; the math is identical, only the checkpoint and image differ.
/// Kept tiny and CPU-only so it finishes in well under a second.
///
/// Coded by Joao Paulo Schwarz Schuler with Claude (AI).
/// https://github.com/joaopauloschuler/neural-api
program DepthEstimation;

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralpretrained;

const
  // Grayscale ramp from far (dark) to near (bright): 10 buckets.
  csRamp: array[0..9] of char = (' ', '.', ':', '-', '=', '+', '*', '#', '%', '@');

function FixturePath(const FileName: string): string;
var
  Base: string;
begin
  // Run from the example dir or the repo root; probe both.
  Base := '../../tests/fixtures/' + FileName;
  if FileExists(Base) then Exit(Base);
  Base := 'tests/fixtures/' + FileName;
  if FileExists(Base) then Exit(Base);
  Result := FileName;
end;

var
  NN: TNNet;
  Config: TDPTConfig;
  Img, Output: TNNetVolume;
  x, y, W, H, GW, GH, Bucket: integer;
  v, dMin, dMax, dRange: TNeuralFloat;
  Line: string;
begin
  WriteLn('DPT / Depth-Anything monocular depth-estimation example');
  WriteLn('-------------------------------------------------------');
  NN := BuildDPTFromSafeTensors(
    FixturePath('tiny_dpt.safetensors'), Config,
    {pInferenceOnly=}true, FixturePath('tiny_dpt_config.json'));
  try
    WriteLn(DPTConfigToString(Config));
    W := Config.Backbone.ImageSize; H := Config.Backbone.ImageSize;
    Img := TNNetVolume.Create;
    try
      // Synthetic test image: a smooth radial + diagonal RGB gradient so the
      // depth map has spatial structure to show (any image works the same).
      Img.ReSize(W, H, Config.Backbone.NumChannels);
      for y := 0 to H - 1 do
        for x := 0 to W - 1 do
        begin
          Img[x, y, 0] := Sin(x / 3.0) * 0.6;
          Img[x, y, 1] := Cos(y / 4.0) * 0.6;
          Img[x, y, 2] := (Sqrt(Sqr(x - W / 2) + Sqr(y - H / 2)) /
            (W / 2.0)) - 0.5;
        end;

      NN.Compute(Img);
      Output := NN.GetLastLayer().Output;
      GW := Output.SizeX; GH := Output.SizeY;
      WriteLn(Format('input %dx%dx%d  ->  depth map %dx%dx%d (1 channel)',
        [W, H, Config.Backbone.NumChannels, GW, GH, Output.Depth]));
      WriteLn;

      // Normalize the depth map to [0,1] for the ASCII ramp.
      dMin := Output.FData[0]; dMax := Output.FData[0];
      for y := 0 to GH - 1 do
        for x := 0 to GW - 1 do
        begin
          v := Output[x, y, 0];
          if v < dMin then dMin := v;
          if v > dMax then dMax := v;
        end;
      dRange := dMax - dMin;
      if dRange <= 0 then dRange := 1;

      for y := 0 to GH - 1 do
      begin
        Line := '';
        for x := 0 to GW - 1 do
        begin
          Bucket := Trunc((Output[x, y, 0] - dMin) / dRange * 9 + 0.5);
          if Bucket < 0 then Bucket := 0;
          if Bucket > 9 then Bucket := 9;
          Line := Line + csRamp[Bucket] + csRamp[Bucket]; // 2 chars wide
        end;
        WriteLn(Line);
      end;

      WriteLn;
      WriteLn(Format('depth range: min=%.6f  max=%.6f', [dMin, dMax]));
      WriteLn('(near = bright "@", far = dark " ")');
      WriteLn;
      WriteLn('Done. (A real run loads depth-anything/Depth-Anything-V2-Small-hf '
        + 'the same way and processes a photo.)');
    finally
      Img.Free;
    end;
  finally
    NN.Free;
  end;
end.
