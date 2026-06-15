/// Human-pose keypoint estimation with an imported ViTPose model.
///
/// ViTPose (Xu et al. 2022, "ViTPose: Simple Vision Transformer Baselines for
/// Human Pose Estimation") is a top-down single-person pose estimator: a plain
/// ViT transformer backbone runs over a cropped person image and a small
/// deconvolution head turns the patch-grid features into one 2-D HEATMAP per
/// keypoint (ReLU -> bilinear upsample -> 3x3 conv to num_joints channels). The
/// (x,y) location of each joint is read out by a spatial ARGMAX over its
/// heatmap. It is the repo's first keypoint / pose import - the output modality
/// is a stack of per-joint heatmaps, not a class label, box or dense class map.
///
/// This example loads the committed PICO ViTPose parity fixture
/// (tests/fixtures/tiny_vitpose.safetensors, a tiny ViT-backbone model) with
/// BuildViTPoseFromSafeTensors, runs it on a small synthetic CPU image, decodes
/// the per-joint peaks with DecodeViTPoseKeypoints, and renders the keypoints as
/// an ASCII plot over the heatmap grid (each joint drawn as a digit at its peak).
/// A real run would load usyd-community/vitpose-base-simple the same way and
/// process a person crop from a detector box; the math is identical, only the
/// checkpoint and image differ. Kept tiny and CPU-only so it finishes in well
/// under a second.
///
/// Coded by Joao Paulo Schwarz Schuler with Claude (AI).
/// https://github.com/joaopauloschuler/neural-api
program PoseEstimation;

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralpretrained;

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
  Config: TViTPoseConfig;
  Img, Output: TNNetVolume;
  Keypoints: TViTPoseKeypointArray;
  x, y, W, H, GW, GH, k: integer;
  Line: string;
  Glyph: char;
begin
  WriteLn('ViTPose human-pose keypoint-estimation example');
  WriteLn('----------------------------------------------');
  NN := BuildViTPoseFromSafeTensors(
    FixturePath('tiny_vitpose.safetensors'), Config,
    {pInferenceOnly=}true, FixturePath('tiny_vitpose_config.json'));
  try
    WriteLn(ViTPoseConfigToString(Config));
    W := Config.ImageW; H := Config.ImageH;
    Img := TNNetVolume.Create;
    try
      // Synthetic person-crop stand-in: a smooth radial + diagonal RGB gradient
      // so the heatmaps have spatial structure to show (any image works the
      // same way - a real run feeds a normalized detector crop).
      Img.ReSize(W, H, Config.NumChannels);
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
      WriteLn(Format('input %dx%dx%d  ->  %d heatmaps %dx%d',
        [W, H, Config.NumChannels, Output.Depth, GW, GH]));
      WriteLn;

      Keypoints := DecodeViTPoseKeypoints(Output);
      WriteLn('Decoded keypoints (heatmap-grid argmax):');
      for k := 0 to Length(Keypoints) - 1 do
        WriteLn(Format('  joint %d: (x=%2d, y=%2d)  score=%.6f',
          [k, Keypoints[k].X, Keypoints[k].Y, Keypoints[k].Score]));
      WriteLn;

      // ASCII skeleton plot: '.' background, the joint index digit at its peak.
      WriteLn('Keypoint plot over the heatmap grid:');
      for y := 0 to GH - 1 do
      begin
        Line := '';
        for x := 0 to GW - 1 do
        begin
          Glyph := '.';
          for k := 0 to Length(Keypoints) - 1 do
            if (Keypoints[k].X = x) and (Keypoints[k].Y = y) then
              Glyph := Chr(Ord('0') + (k mod 10));
          Line := Line + Glyph + ' ';
        end;
        WriteLn(Line);
      end;

      WriteLn;
      WriteLn('Done. (A real run loads usyd-community/vitpose-base-simple the '
        + 'same way and processes a detector person crop.)');
    finally
      Img.Free;
    end;
  finally
    NN.Free;
  end;
end.
