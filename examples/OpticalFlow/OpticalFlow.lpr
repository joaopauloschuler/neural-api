program OpticalFlow;
(*
OpticalFlow: dense optical-flow estimation with the RAFT importer - the FIRST
two-image-in / dense-(dx,dy)-out vertical in the tree (Teed & Deng 2020, "RAFT:
Recurrent All-Pairs Field Transforms for Optical Flow", arXiv:2003.12039).

WHAT IT SHOWS
-------------
RAFT takes TWO frames and predicts, for every pixel, the (dx, dy) motion that
maps frame-1 onto frame-2. The pipeline:
  shared feature encoder over both frames  ->  an all-pairs CORRELATION volume
  (TNNetCorrelationVolume: dot-products between every pair of feature
  locations)  ->  an iterative ConvGRU update operator (TNNetConvGRUCell)
  that, via a local correlation lookup (TNNetCorrelationLookup) around the
  current flow, refines the flow over N steps.

THE DATA (synthetic, no download)
---------------------------------
A tiny bright square on a dark field; frame-2 is frame-1 translated by a known
(SHIFT_X, SHIFT_Y). The TRUE flow is therefore a constant (SHIFT_X, SHIFT_Y)
everywhere the texture is visible - a sanity target for the visualisation.

THE MODEL
---------
We load the committed pico raft_small fixture (tests/fixtures/tiny_raft) with
BuildRaftFromSafeTensors. The weights are RANDOM (this example is a forward /
plumbing demonstration, not a trained model), so the predicted field is not the
ground-truth flow; the point is to exercise the full RAFT forward and the flow
visualisation + warping primitives end to end. (To get real flow, point
BuildRaftFromSafeTensors at a real torchvision raft_small export.)

OUTPUT
------
Two PPM images written to the working directory:
  opticalflow_field.ppm : the predicted flow color-coded the standard way -
      HUE = flow DIRECTION, SATURATION/VALUE = flow MAGNITUDE (Middlebury-style).
  opticalflow_warp.ppm   : frame-1 | frame-1 warped toward frame-2 by the
      predicted flow (TNNetFlowWarp) | frame-2, side by side at the /4 flow grid.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralpretrained;

const
  SHIFT_X = 2;
  SHIFT_Y = 1;

// Build a synthetic frame: a bright square at (cx, cy) on a dark field.
procedure MakeFrame(V: TNNetVolume; W, H, cx, cy, sz: integer);
var
  x, y, c: integer;
  bright: boolean;
begin
  V.ReSize(W, H, 3);
  V.Fill(-0.8);
  for y := 0 to H - 1 do
    for x := 0 to W - 1 do
    begin
      bright := (Abs(x - cx) <= sz) and (Abs(y - cy) <= sz);
      if bright then
        for c := 0 to 2 do
          // a little per-channel texture so the correlation has signal
          V.Store(x, y, c, 0.9 - 0.15 * c + 0.05 * ((x + y) mod 3));
    end;
end;

// Average-pool a (W,H,3) frame down by Stride into a (W/S,H/S,1) gray map.
procedure DownGray(Src, Dst: TNNetVolume; Stride: integer);
var
  W, H, x, y, dx, dy, c: integer;
  acc: single;
begin
  W := Src.SizeX div Stride;
  H := Src.SizeY div Stride;
  Dst.ReSize(W, H, 1);
  for y := 0 to H - 1 do
    for x := 0 to W - 1 do
    begin
      acc := 0;
      for dy := 0 to Stride - 1 do
        for dx := 0 to Stride - 1 do
          for c := 0 to 2 do
            acc := acc + Src.Get(x * Stride + dx, y * Stride + dy, c);
      Dst.Store(x, y, 0, acc / (Stride * Stride * 3));
    end;
end;

// Middlebury-style flow color: hue = direction, value/sat = magnitude.
procedure FlowColor(dx, dy, maxMag: single; out r, g, b: integer);
var
  ang, mag, h, s, v, f, p, q, t, rr, gg, bb: single;
  i: integer;
begin
  mag := Sqrt(dx * dx + dy * dy);
  if maxMag > 1e-6 then s := Min(1.0, mag / maxMag) else s := 0;
  v := 1.0;
  ang := ArcTan2(dy, dx);                 // -pi..pi
  h := (ang + PI) / (2 * PI) * 6.0;       // 0..6
  i := Floor(h);
  f := h - i;
  p := v * (1 - s);
  q := v * (1 - s * f);
  t := v * (1 - s * (1 - f));
  case (i mod 6) of
    0: begin rr := v; gg := t; bb := p; end;
    1: begin rr := q; gg := v; bb := p; end;
    2: begin rr := p; gg := v; bb := t; end;
    3: begin rr := p; gg := q; bb := v; end;
    4: begin rr := t; gg := p; bb := v; end;
  else begin rr := v; gg := p; bb := q; end;
  end;
  r := Round(rr * 255); g := Round(gg * 255); b := Round(bb * 255);
end;

var
  NN: TNNet;
  Config: TRaftConfig;
  Frame1, Frame2, Flow, G1, G2, Warped: TNNetVolume;
  WarpNet: TNNet;
  inLayer, flowLayer: TNNetLayer;
  FlowW, FlowH, x, y, b: integer;
  r, g, bb: integer;
  maxMag, mg: single;
  f: Text;
  panelW: integer;
  fixturePath: string;
begin
  WriteLn('OpticalFlow (RAFT) example');
  fixturePath := '../../tests/fixtures/tiny_raft.safetensors';
  if not FileExists(fixturePath) then
    fixturePath := 'tests/fixtures/tiny_raft.safetensors';
  if not FileExists(fixturePath) then
  begin
    WriteLn('Could not find tiny_raft.safetensors (run tools/raft_small_tiny_fixture.py).');
    Halt(1);
  end;

  NN := BuildRaftFromSafeTensors(fixturePath, Config, {pInferenceOnly=}True);
  WriteLn(RaftConfigToString(Config));

  Frame1 := TNNetVolume.Create;
  Frame2 := TNNetVolume.Create;
  Flow := TNNetVolume.Create;
  // frame-1: square centred; frame-2: same square translated by (SHIFT_X,SHIFT_Y)
  MakeFrame(Frame1, Config.ImageW, Config.ImageH,
    Config.ImageW div 2, Config.ImageH div 2, 4);
  MakeFrame(Frame2, Config.ImageW, Config.ImageH,
    Config.ImageW div 2 + SHIFT_X, Config.ImageH div 2 + SHIFT_Y, 4);

  RaftPredictFlow(NN, Config, Frame1, Frame2, Flow);
  FlowW := Flow.SizeX; FlowH := Flow.SizeY;
  WriteLn('Predicted flow grid ', FlowW, 'x', FlowH,
    '  (true frame shift = ', SHIFT_X, ',', SHIFT_Y, ' px at full res)');

  // ---- flow color-map PPM ----
  maxMag := 0;
  for y := 0 to FlowH - 1 do
    for x := 0 to FlowW - 1 do
    begin
      mg := Sqrt(Sqr(Flow.Get(x, y, 0)) + Sqr(Flow.Get(x, y, 1)));
      if mg > maxMag then maxMag := mg;
    end;
  if maxMag < 1e-6 then maxMag := 1;
  AssignFile(f, 'opticalflow_field.ppm'); Rewrite(f);
  WriteLn(f, 'P3'); WriteLn(f, FlowW, ' ', FlowH); WriteLn(f, '255');
  for y := 0 to FlowH - 1 do
    for x := 0 to FlowW - 1 do
    begin
      FlowColor(Flow.Get(x, y, 0), Flow.Get(x, y, 1), maxMag, r, g, bb);
      WriteLn(f, r, ' ', g, ' ', bb);
    end;
  CloseFile(f);
  WriteLn('wrote opticalflow_field.ppm (hue=direction, brightness=magnitude)');

  // ---- warp frame-1 toward frame-2 by the predicted flow ----
  // A tiny standalone warp net: gray frame-1 + the predicted flow -> warped.
  G1 := TNNetVolume.Create;
  G2 := TNNetVolume.Create;
  Warped := TNNetVolume.Create;
  DownGray(Frame1, G1, Config.Stride);
  DownGray(Frame2, G2, Config.Stride);
  WarpNet := TNNet.Create;
  inLayer := WarpNet.AddLayer(TNNetInput.Create(FlowW, FlowH, 1));
  flowLayer := WarpNet.AddLayer(TNNetInput.Create(FlowW, FlowH, 2));
  WarpNet.AddLayerAfter(TNNetFlowWarp.Create(flowLayer), inLayer);
  WarpNet.Compute([G1, Flow]);
  Warped.Copy(WarpNet.GetLastLayer().Output);

  panelW := 3 * FlowW;  // frame1 | warped | frame2
  AssignFile(f, 'opticalflow_warp.ppm'); Rewrite(f);
  WriteLn(f, 'P3'); WriteLn(f, panelW, ' ', FlowH); WriteLn(f, '255');
  for y := 0 to FlowH - 1 do
    for x := 0 to panelW - 1 do
    begin
      if x < FlowW then b := Round((G1.Get(x, y, 0) + 1) * 0.5 * 255)
      else if x < 2 * FlowW then
        b := Round((Warped.Get(x - FlowW, y, 0) + 1) * 0.5 * 255)
      else b := Round((G2.Get(x - 2 * FlowW, y, 0) + 1) * 0.5 * 255);
      if b < 0 then b := 0 else if b > 255 then b := 255;
      WriteLn(f, b, ' ', b, ' ', b);
    end;
  CloseFile(f);
  WriteLn('wrote opticalflow_warp.ppm (frame1 | warped-frame1 | frame2)');

  Warped.Free; G2.Free; G1.Free; WarpNet.Free;
  Flow.Free; Frame2.Free; Frame1.Free; NN.Free;
  WriteLn('done.');
end.
