/// Depth Anything V2 monocular RELATIVE-depth -> a normalized depth-map IMAGE.
///
/// Depth Anything V2 (Yang et al. 2024) is the DPT dense-prediction stack
/// (Ranftl et al. 2021) on a DINOv2 ViT backbone (S/B/L): four SELECTED
/// intermediate encoder stages (the backbone's `out_indices`) are projected and
/// resized into a feature pyramid, RefineNet-style additive fusion blocks merge
/// them coarse-to-fine, and a 3-conv head emits a SINGLE-channel per-pixel
/// relative-depth map at the full input resolution.
///
/// This example differs from examples/DepthEstimation (which prints an ASCII
/// grayscale ramp of the generic DPT path): it loads the model through the NAMED
/// BuildDepthAnythingV2FromSafeTensors entry point, uses the committed pico
/// fixture that hooks NON-last-4 stages (out_indices=[2,3,5,6]) to exercise the
/// out_indices wiring, and WRITES a real image file - a min/max-normalized 8-bit
/// grayscale depth map as a binary PGM (P5), plus an inverse-depth PPM (P6) using
/// a simple turbo-like color ramp (near = warm, far = cool), the way the Depth
/// Anything demos visualize depth.
///
/// Default = a SMOKE run on the pico fixture (CPU, well under a second). To run a
/// real checkpoint, pass the .safetensors path as argv[1] (config.json must sit
/// beside it) and, optionally, a raw RGB image as argv[2] (see notes below); the
/// math is identical, only the checkpoint and the input image differ.
///
/// Usage:
///   DepthAnythingV2                      # pico smoke run -> depth_pico.{pgm,ppm}
///   DepthAnythingV2 model.safetensors    # real checkpoint, synthetic input
///
/// Coded by Joao Paulo Schwarz Schuler with Claude (AI).
/// https://github.com/joaopauloschuler/neural-api
///
/// Coded by Claude (AI).
program DepthAnythingV2;

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

// Fills Img (W,H,3) with a smooth synthetic scene: a radial gradient plus a
// couple of off-center blobs, so the depth map has visible structure. Values are
// roughly model-normalized (mean 0, modest range), matching the parity fixture's
// input scale.
procedure MakeSyntheticImage(Img: TNNetVolume; W, H: integer);
var
  x, y, c: integer;
  cx, cy, r, v: TNeuralFloat;
begin
  Img.ReSize(W, H, 3);
  cx := W / 2; cy := H / 2;
  for y := 0 to H - 1 do
    for x := 0 to W - 1 do
    begin
      r := Sqrt(Sqr(x - cx) + Sqr(y - cy)) / (cx + 1);
      v := 0.8 - r;                               // bright center, dark edges
      v := v + 0.4 * Exp(-(Sqr(x - W * 0.25) + Sqr(y - H * 0.30)) / (W * 0.6));
      v := v - 0.3 * Exp(-(Sqr(x - W * 0.75) + Sqr(y - H * 0.70)) / (W * 0.6));
      for c := 0 to 2 do
        // slight per-channel tint so all 3 input channels carry signal.
        Img.FData[(y * W + x) * 3 + c] := v + 0.05 * (c - 1);
    end;
end;

// Writes a binary 8-bit grayscale PGM (P5). Gray[i] in 0..255, i = y*W+x.
procedure WritePGM(const FileName: string; const Gray: array of byte;
  W, H: integer);
var
  fs: TFileStream;
  hdr: AnsiString;
  i: integer;
  b: byte;
begin
  fs := TFileStream.Create(FileName, fmCreate);
  try
    hdr := 'P5'#10 + IntToStr(W) + ' ' + IntToStr(H) + #10'255'#10;
    fs.WriteBuffer(hdr[1], Length(hdr));
    for i := 0 to W * H - 1 do
    begin
      b := Gray[i];
      fs.WriteBuffer(b, 1);
    end;
  finally
    fs.Free;
  end;
end;

// Writes a binary RGB PPM (P6) using a simple turbo-like ramp over t in [0,1]
// (t=0 -> cool/blue=far, t=1 -> warm/red=near), the usual depth visualization.
procedure WritePPMColor(const FileName: string; const T: array of TNeuralFloat;
  W, H: integer);
var
  fs: TFileStream;
  hdr: AnsiString;
  i: integer;
  rgb: array[0..2] of byte;
  t8, rr, gg, bb: TNeuralFloat;

  function Clamp255(v: TNeuralFloat): byte;
  begin
    if v < 0 then v := 0;
    if v > 1 then v := 1;
    Result := Round(v * 255);
  end;

begin
  fs := TFileStream.Create(FileName, fmCreate);
  try
    hdr := 'P6'#10 + IntToStr(W) + ' ' + IntToStr(H) + #10'255'#10;
    fs.WriteBuffer(hdr[1], Length(hdr));
    for i := 0 to W * H - 1 do
    begin
      t8 := T[i];
      // cheap blue->green->red ramp.
      rr := 1.5 * t8 - 0.5;
      gg := 1 - Abs(2 * t8 - 1);
      bb := 1.5 * (1 - t8) - 0.5;
      rgb[0] := Clamp255(rr);
      rgb[1] := Clamp255(gg);
      rgb[2] := Clamp255(bb);
      fs.WriteBuffer(rgb[0], 3);
    end;
  finally
    fs.Free;
  end;
end;

var
  NN: TNNet;
  Config: TDPTConfig;
  Img, Output: TNNetVolume;
  ModelPath, ConfigPath, OutBase: string;
  W, H, GW, GH, i, x, y: integer;
  v, dMin, dMax, dRange: TNeuralFloat;
  Gray: array of byte;
  Norm: array of TNeuralFloat;
begin
  WriteLn('Depth Anything V2 monocular relative-depth -> normalized depth image');
  WriteLn('-------------------------------------------------------------------');

  if ParamCount >= 1 then
  begin
    ModelPath := ParamStr(1);
    ConfigPath := ExtractFilePath(ModelPath) + 'config.json';
    OutBase := 'depth_real';
    WriteLn('Real checkpoint: ', ModelPath);
  end
  else
  begin
    ModelPath := FixturePath('tiny_depth_anything_v2.safetensors');
    ConfigPath := FixturePath('tiny_depth_anything_v2_config.json');
    OutBase := 'depth_pico';
    WriteLn('Pico smoke run: ', ModelPath);
  end;

  // pTrainable=false frees per-neuron training buffers during the build.
  NN := BuildDepthAnythingV2FromSafeTensors(ModelPath, Config,
    {pTrainable=}false, ConfigPath);
  Img := TNNetVolume.Create;
  try
    WriteLn('Model: ', DPTConfigToString(Config));
    W := Config.Backbone.ImageSize;
    H := Config.Backbone.ImageSize;
    MakeSyntheticImage(Img, W, H);

    NN.Compute(Img);
    Output := NN.GetLastLayer().Output;
    GW := Output.SizeX; GH := Output.SizeY;
    WriteLn('Depth map: ', GW, 'x', GH, ' (1 channel), input ', W, 'x', H);

    // min/max normalize the relative depth to [0,1] for visualization.
    dMin := Output.FData[0]; dMax := Output.FData[0];
    for i := 1 to GW * GH - 1 do
    begin
      v := Output.FData[i];
      if v < dMin then dMin := v;
      if v > dMax then dMax := v;
    end;
    dRange := dMax - dMin;
    if dRange < 1e-12 then dRange := 1;
    WriteLn('Depth range: min=', dMin:0:5, ' max=', dMax:0:5);

    SetLength(Gray, GW * GH);
    SetLength(Norm, GW * GH);
    for i := 0 to GW * GH - 1 do
    begin
      v := (Output.FData[i] - dMin) / dRange;     // 0=far, 1=near
      Norm[i] := v;
      Gray[i] := Round(v * 255);
    end;

    WritePGM(OutBase + '.pgm', Gray, GW, GH);
    WritePPMColor(OutBase + '.ppm', Norm, GW, GH);
    WriteLn('Wrote ', OutBase, '.pgm (grayscale) and ', OutBase,
      '.ppm (color depth).');

    // a small ASCII preview so the run shows something on a headless console.
    WriteLn('Preview (near = bright):');
    for y := 0 to Min(GH - 1, 15) do
    begin
      Write('  ');
      for x := 0 to Min(GW - 1, 31) do
      begin
        v := Norm[y * GW + x];
        Write(' .:-=+*#%@'[1 + Round(v * 9)]);
      end;
      WriteLn;
    end;
  finally
    Img.Free;
    NN.Free;
  end;
  WriteLn('Done.');
end.
