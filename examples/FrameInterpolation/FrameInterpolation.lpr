program FrameInterpolation;
(*
FrameInterpolation: video FRAME INTERPOLATION - predict the unseen MIDDLE frame
that sits BETWEEN two given endpoint frames (the RIFE / FILM task). This is
structurally distinct from examples/VideoPrediction, which EXTRAPOLATES the next
frame; here the network is handed frames t and t+2 and must SYNTHESIZE the
in-between frame t+1.

THE DATA (synthetic, generated in code - no download)
-----------------------------------------------------
Reuses the same self-contained Moving-MNIST-style blob world as VideoPrediction:
a small bright blob translates across a cGrid x cGrid grayscale grid at constant
integer velocity, bouncing off the walls. From each 3-frame clip (t, t+1, t+2)
the model sees the two ENDPOINTS (t and t+2, stacked as two CHANNELS of one
cGrid x cGrid x 2 image) and is supervised on the hidden MIDDLE frame t+1.
Pixels are in [-1,1] (background -1, blob centre +1).

THE LOSS: L1 + (1 - SSIM)
-------------------------
The reconstruction objective is a blend of pixel L1 and the structural-similarity
term from neuralimagemetrics.ComputeSSIMLossAndGradient (a documented loss
HELPER, not a TNNet* layer). We compute the combined per-pixel gradient by hand
and inject it through the standard TNNet.Backpropagate path: the library's
last-layer rule is OutputError = Output - Desired, so handing it the pseudo-
target Desired = Output - GradOut makes the back-propagated error EXACTLY our
custom d(loss)/d(pred). SSIM needs an 11x11 window, so cGrid = 16.

TWO MODELS COMPARED
-------------------
(a) DIRECT conv encoder-decoder: a small CNN that reads the two endpoint
    channels and directly SYNTHESIZES the middle frame's pixels. It has to
    hallucinate the moved blob from scratch.

(b) FLOW-based warping (the textbook reason warping beats direct synthesis for
    motion): the SAME encoder instead predicts, per pixel, a dense optical-flow
    field for each endpoint - F0 = (mid -> t) and F1 = (mid -> t+2) - and the new
    TNNetFlowWarp primitive BACKWARD-WARPS each endpoint frame along its flow so
    its content lands at the middle-frame position. The two warped endpoints are
    averaged (symmetric 0.5/0.5 blend, the standard mid-frame combination). The
    model never has to invent texture; it only has to learn WHERE each pixel
    moved, which is a far smaller, better-posed problem - so on this rigid-motion
    data the flow path reaches a lower held-out error than direct synthesis.

  inp (cGrid,cGrid,2)  [ch0 = frame t, ch1 = frame t+2]
    frame0 = SplitChannels(0,1)         -- the t endpoint as a 1-channel image
    frame1 = SplitChannels(1,1)         -- the t+2 endpoint
    enc    = Conv3x3(C)->ReLU ->Conv3x3(C)->ReLU      (over both channels)
    flow   = ConvLinear(4,3,1,1)        -- 4 maps: F0=(dx0,dy0), F1=(dx1,dy1)
    F0     = SplitChannels(0,2);  F1 = SplitChannels(2,2)
    w0     = FlowWarp(frame0, F0);  w1 = FlowWarp(frame1, F1)
    out    = 0.5*w0 + 0.5*w1            -- symmetric blend (Sum of two *0.5)

OUTPUT
------
Held-out L1 / SSIM for both models, an ASCII panel (before | predicted | after |
truth) and a PPM triplet dump for one held-out clip from each model. The default
SMOKE run finishes well under five minutes on CPU; pass --full for a longer run.

  Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralimagemetrics;

const
  cGrid    = 16;   // frame is cGrid x cGrid grayscale (>= 11 for the SSIM window)
  cBlobR   = 1;    // blob "radius" (half-side of the filled square)
  cSSIMW   = 0.4;  // weight of (1 - SSIM); pixel L1 carries weight (1 - cSSIMW)
  cRange   = 2.0;  // pixel data range ([-1,1] -> span 2)

var
  gC       : integer = 12;   // encoder channels
  gNumTrain: integer = 200;  // training clips
  gNumTest : integer = 40;   // held-out clips
  gEpochs  : integer = 8;    // passes over the training set
  gLR      : single  = 0.002;

// ---------------------------------------------------------------------------
// Synthetic moving-blob clip generator. A clip is 3 frames (t, t+1, t+2),
// each written into Dst at X-row offset f*cGrid (frames stacked on the X axis).
// ---------------------------------------------------------------------------
procedure MakeClip(Frames: TNNetVolume);
var
  px, py, vx, vy, t, dx, dy, x, y, fx: integer;
  v: single;
begin
  Frames.Fill(-1.0);  // background
  px := cBlobR + Random(cGrid - 2 * cBlobR);
  py := cBlobR + Random(cGrid - 2 * cBlobR);
  repeat
    vx := Random(3) - 1;
    vy := Random(3) - 1;
  until (vx <> 0) or (vy <> 0);
  for t := 0 to 2 do
  begin
    for dy := -cBlobR to cBlobR do
      for dx := -cBlobR to cBlobR do
      begin
        x := px + dx; y := py + dy;
        if (x >= 0) and (x < cGrid) and (y >= 0) and (y < cGrid) then
        begin
          if (dx = 0) and (dy = 0) then v := 1.0 else v := 0.4;
          fx := t * cGrid + x;
          if v > Frames[fx, y, 0] then Frames[fx, y, 0] := v;
        end;
      end;
    px := px + vx; py := py + vy;
    if px < cBlobR then begin px := cBlobR; vx := -vx; end;
    if px > cGrid - 1 - cBlobR then begin px := cGrid - 1 - cBlobR; vx := -vx; end;
    if py < cBlobR then begin py := cBlobR; vy := -vy; end;
    if py > cGrid - 1 - cBlobR then begin py := cGrid - 1 - cBlobR; vy := -vy; end;
  end;
end;

// Endpoint channels (t, t+2) -> Inp(cGrid,cGrid,2); middle frame t+1 -> Tgt.
procedure SplitClip(Clip, Inp, Tgt: TNNetVolume);
var x, y: integer;
begin
  for y := 0 to cGrid - 1 do
    for x := 0 to cGrid - 1 do
    begin
      Inp[x, y, 0] := Clip[0 * cGrid + x, y, 0]; // endpoint t      -> channel 0
      Inp[x, y, 1] := Clip[2 * cGrid + x, y, 0]; // endpoint t+2    -> channel 1
      Tgt[x, y, 0] := Clip[1 * cGrid + x, y, 0]; // middle frame t+1 = target
    end;
end;

// ---------------------------------------------------------------------------
// Model (a): direct conv encoder-decoder that synthesizes the middle frame.
// ---------------------------------------------------------------------------
function BuildDirectModel(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 2, 1));
  Result.AddLayer(TNNetConvolutionReLU.Create(gC, 3, 1, 1, 0));
  Result.AddLayer(TNNetConvolutionReLU.Create(gC, 3, 1, 1, 0));
  Result.AddLayer(TNNetConvolutionLinear.Create(1, 3, 1, 1)); // -> 1 channel
  Result.AddLayer(TNNetHyperbolicTangent.Create());           // [-1,1]
end;

// ---------------------------------------------------------------------------
// Model (b): flow-based. The encoder predicts a dense flow field for each
// endpoint; TNNetFlowWarp backward-warps each endpoint frame; the two warps are
// averaged. The warp uses the NEW dense flow-warp primitive.
// ---------------------------------------------------------------------------
function BuildFlowModel(): TNNet;
var
  inp, frame0, frame1, enc1, enc2, flow, f0, f1, w0, w1, b0, b1: TNNetLayer;
begin
  Result := TNNet.Create();
  inp := Result.AddLayer(TNNetInput.Create(cGrid, cGrid, 2, 1));
  // The two endpoint frames as separate 1-channel images.
  frame0 := Result.AddLayerAfter(TNNetSplitChannels.Create(0, 1), inp);
  frame1 := Result.AddLayerAfter(TNNetSplitChannels.Create(1, 1), inp);
  // Shared encoder over BOTH endpoint channels.
  enc1 := Result.AddLayerAfter(TNNetConvolutionReLU.Create(gC, 3, 1, 1, 0), inp);
  enc2 := Result.AddLayerAfter(TNNetConvolutionReLU.Create(gC, 3, 1, 1, 0), enc1);
  // Flow head: 4 maps = F0 (dx0,dy0) for the t endpoint, F1 (dx1,dy1) for t+2.
  flow := Result.AddLayerAfter(TNNetConvolutionLinear.Create(4, 3, 1, 1), enc2);
  f0 := Result.AddLayerAfter(TNNetSplitChannels.Create(0, 2), flow);
  f1 := Result.AddLayerAfter(TNNetSplitChannels.Create(2, 2), flow);
  // Backward-warp each endpoint along its predicted flow (the new primitive).
  w0 := Result.AddLayerAfter(TNNetFlowWarp.Create(f0), frame0);
  w1 := Result.AddLayerAfter(TNNetFlowWarp.Create(f1), frame1);
  // Symmetric 0.5/0.5 blend of the two warped endpoints -> middle frame.
  b0 := Result.AddLayerAfter(TNNetMulByConstant.Create(0.5), w0);
  b1 := Result.AddLayerAfter(TNNetMulByConstant.Create(0.5), w1);
  Result.AddLayer(TNNetSum.Create([b0, b1]));
end;

// ---------------------------------------------------------------------------
// Combined L1 + (1-SSIM) loss and its per-pixel gradient w.r.t. the prediction.
// Returns the scalar loss; fills GradOut (same layout as the frame) with
// d(loss)/d(pred). Single-channel cGrid x cGrid frames.
// ---------------------------------------------------------------------------
function LossAndGrad(Pred, Tgt, GradOut: TNNetVolume): double;
var
  x, y, i: integer;
  pa, pb, ssimGrad: TIMDoubleArray;
  l1, ssimLoss, d: double;
begin
  SetLength(pa, cGrid * cGrid);
  SetLength(pb, cGrid * cGrid);
  for y := 0 to cGrid - 1 do
    for x := 0 to cGrid - 1 do
    begin
      pa[y * cGrid + x] := Pred[x, y, 0];
      pb[y * cGrid + x] := Tgt[x, y, 0];
    end;
  // SSIM term (and its gradient w.r.t. the prediction).
  ssimLoss := ComputeSSIMLossAndGradient(pa, pb, cGrid, cGrid, 1, ssimGrad, cRange);
  // Pixel L1 term + its subgradient.
  l1 := 0;
  GradOut.Fill(0);
  for y := 0 to cGrid - 1 do
    for x := 0 to cGrid - 1 do
    begin
      i := y * cGrid + x;
      d := pa[i] - pb[i];
      l1 := l1 + Abs(d);
      GradOut[x, y, 0] := (1.0 - cSSIMW) * Sign(d) / (cGrid * cGrid)
                        + cSSIMW * ssimGrad[i];
    end;
  l1 := l1 / (cGrid * cGrid);
  Result := (1.0 - cSSIMW) * l1 + cSSIMW * ssimLoss;
end;

// Drive backprop on a CUSTOM per-pixel gradient via the standard, fully-tested
// TNNet.Backpropagate path. The library's last-layer error rule is
//   OutputError = Output - Desired   (ComputeOutputErrorWith),
// so to inject an arbitrary d(loss)/d(pred) = GradOut we hand it the pseudo-
// target  Desired = Output - GradOut, which makes Output - Desired = GradOut
// exactly. This reuses all of TNNet.Backpropagate's branch-counter / departing-
// branch bookkeeping instead of re-implementing it.
procedure BackpropFromGrad(NN: TNNet; GradOut: TNNetVolume; PseudoTgt: TNNetVolume);
begin
  PseudoTgt.Copy(NN.GetLastLayer.Output);
  PseudoTgt.Sub(GradOut);
  NN.Backpropagate(PseudoTgt);
end;

// ---------------------------------------------------------------------------
// Metrics.
// ---------------------------------------------------------------------------
procedure EvalSet(NN: TNNet; const Clips: array of TNNetVolume; Count: integer;
  out AvgL1, AvgSSIM: double);
var
  i, x, y: integer;
  Inp, Tgt, Pred: TNNetVolume;
  pa, pb, dummy: TIMDoubleArray;
  sL1, d: double;
begin
  Inp := TNNetVolume.Create(cGrid, cGrid, 2);
  Tgt := TNNetVolume.Create(cGrid, cGrid, 1);
  SetLength(pa, cGrid * cGrid);
  SetLength(pb, cGrid * cGrid);
  sL1 := 0; AvgSSIM := 0;
  for i := 0 to Count - 1 do
  begin
    SplitClip(Clips[i], Inp, Tgt);
    NN.Compute(Inp);
    Pred := NN.GetLastLayer.Output;
    for y := 0 to cGrid - 1 do
      for x := 0 to cGrid - 1 do
      begin
        d := Pred[x, y, 0] - Tgt[x, y, 0];
        sL1 := sL1 + Abs(d);
        pa[y * cGrid + x] := Pred[x, y, 0];
        pb[y * cGrid + x] := Tgt[x, y, 0];
      end;
    AvgSSIM := AvgSSIM +
      (1.0 - ComputeSSIMLossAndGradient(pa, pb, cGrid, cGrid, 1, dummy, cRange));
  end;
  AvgL1 := sL1 / (Count * cGrid * cGrid);
  AvgSSIM := AvgSSIM / Count;
  Inp.Free; Tgt.Free;
end;

procedure TrainOne(NN: TNNet; const Train: array of TNNetVolume;
  Count, Epochs: integer; const Tag: string);
var
  Inp, Tgt, Grad, PseudoTgt: TNNetVolume;
  Perm: array of integer;
  i, ep, idx, order, tmp: integer;
  StartTime, epLoss: double;
begin
  Inp := TNNetVolume.Create(cGrid, cGrid, 2);
  Tgt := TNNetVolume.Create(cGrid, cGrid, 1);
  Grad := TNNetVolume.Create(cGrid, cGrid, 1);
  PseudoTgt := TNNetVolume.Create(cGrid, cGrid, 1);
  epLoss := 0;
  NN.SetLearningRate(gLR, 0.9);
  SetLength(Perm, Count);
  for i := 0 to Count - 1 do Perm[i] := i;
  StartTime := Now();
  for ep := 1 to Epochs do
  begin
    for i := Count - 1 downto 1 do
    begin
      order := Random(i + 1);
      tmp := Perm[i]; Perm[i] := Perm[order]; Perm[order] := tmp;
    end;
    for i := 0 to Count - 1 do
    begin
      idx := Perm[i];
      SplitClip(Train[idx], Inp, Tgt);
      NN.Compute(Inp);
      epLoss := epLoss + LossAndGrad(NN.GetLastLayer.Output, Tgt, Grad);
      BackpropFromGrad(NN, Grad, PseudoTgt);
    end;
    WriteLn(Format('  [%s] epoch %2d/%2d  train loss=%.5f',
      [Tag, ep, Epochs, epLoss / Count]));
    epLoss := 0;
  end;
  WriteLn(Format('  [%s] trained %d epochs in %.1f s',
    [Tag, Epochs, (Now() - StartTime) * 86400]));
  Inp.Free; Tgt.Free; Grad.Free; PseudoTgt.Free;
end;

function Glyph(v: single): char;
begin
  if v < -0.6 then Result := ' '
  else if v < -0.1 then Result := '.'
  else if v < 0.4 then Result := ':'
  else if v < 0.8 then Result := '+'
  else Result := '#';
end;

procedure RenderPanel(NN: TNNet; Clip: TNNetVolume; const Tag: string);
var
  Inp, Tgt, Pred: TNNetVolume;
  x, y: integer;
  line: string;
begin
  Inp := TNNetVolume.Create(cGrid, cGrid, 2);
  Tgt := TNNetVolume.Create(cGrid, cGrid, 1);
  SplitClip(Clip, Inp, Tgt);
  NN.Compute(Inp);
  Pred := NN.GetLastLayer.Output;
  WriteLn;
  WriteLn('ASCII panel [', Tag, ']: BEFORE(t) | PREDICTED(t+1) | TRUTH(t+1) | AFTER(t+2)');
  for y := 0 to cGrid - 1 do
  begin
    line := '';
    for x := 0 to cGrid - 1 do line := line + Glyph(Inp[x, y, 0]);
    line := line + ' | ';
    for x := 0 to cGrid - 1 do line := line + Glyph(Pred[x, y, 0]);
    line := line + ' | ';
    for x := 0 to cGrid - 1 do line := line + Glyph(Tgt[x, y, 0]);
    line := line + ' | ';
    for x := 0 to cGrid - 1 do line := line + Glyph(Inp[x, y, 1]);
    WriteLn(line);
  end;
  Inp.Free; Tgt.Free;
end;

// Dump a before | predicted | after triplet (predicted green, truth-overlay red
// channel carries the ground-truth middle frame for visual comparison).
procedure DumpPPM(NN: TNNet; Clip: TNNetVolume; const FileName: string);
var
  Inp, Tgt, Pred: TNNetVolume;
  f: TextFile;
  panelW, x, y, gx, b, bt: integer;
  v: single;
begin
  Inp := TNNetVolume.Create(cGrid, cGrid, 2);
  Tgt := TNNetVolume.Create(cGrid, cGrid, 1);
  SplitClip(Clip, Inp, Tgt);
  NN.Compute(Inp);
  Pred := NN.GetLastLayer.Output;
  panelW := 3 * cGrid; // before | predicted | after
  AssignFile(f, FileName); Rewrite(f);
  WriteLn(f, 'P3'); WriteLn(f, panelW, ' ', cGrid); WriteLn(f, '255');
  for y := 0 to cGrid - 1 do
    for gx := 0 to panelW - 1 do
    begin
      x := gx mod cGrid;
      if gx < cGrid then
      begin
        v := Inp[x, y, 0]; b := Round((v + 1) * 0.5 * 255);
        WriteLn(f, b, ' ', b, ' ', b);            // BEFORE frame t (gray)
      end
      else if gx < 2 * cGrid then
      begin
        v := Pred[x, y, 0]; b := Round((v + 1) * 0.5 * 255);
        v := Tgt[x, y, 0];  bt := Round((v + 1) * 0.5 * 255);
        WriteLn(f, bt, ' ', b, ' ', 0);           // pred=green, truth=red overlay
      end
      else
      begin
        v := Inp[x, y, 1]; b := Round((v + 1) * 0.5 * 255);
        WriteLn(f, b, ' ', b, ' ', b);            // AFTER frame t+2 (gray)
      end;
    end;
  CloseFile(f);
  Inp.Free; Tgt.Free;
  WriteLn('Wrote ', FileName, ' (', panelW, 'x', cGrid,
    ' : before gray | middle [green=pred, red=truth] | after gray)');
end;

// ---------------------------------------------------------------------------
var
  Direct, Flow: TNNet;
  Train, Test: array of TNNetVolume;
  i, a: integer;
  dL1, dS, fL1, fS, d0L1, d0S, f0L1, f0S: double;
begin
  Randomize; RandSeed := 20170911;  // arXiv id of the original SSIM-loss paper era

  for a := 1 to ParamCount do
    if ParamStr(a) = '--full' then
    begin
      gC := 20; gNumTrain := 600; gNumTest := 80; gEpochs := 20;
      WriteLn('Running in --full mode (longer, sharper).');
    end;

  WriteLn('=== FrameInterpolation: predict the MIDDLE frame (RIFE/FILM task) ===');
  WriteLn(Format('grid=%dx%d  encChannels=%d  train=%d  test=%d  epochs=%d  loss=L1+%.2f*(1-SSIM)',
    [cGrid, cGrid, gC, gNumTrain, gNumTest, gEpochs, cSSIMW]));

  SetLength(Train, gNumTrain);
  SetLength(Test, gNumTest);
  for i := 0 to gNumTrain - 1 do
  begin
    Train[i] := TNNetVolume.Create(3 * cGrid, cGrid, 1);
    MakeClip(Train[i]);
  end;
  for i := 0 to gNumTest - 1 do
  begin
    Test[i] := TNNetVolume.Create(3 * cGrid, cGrid, 1);
    MakeClip(Test[i]);
  end;

  Direct := BuildDirectModel();
  Flow := BuildFlowModel();
  WriteLn('Direct model layers: ', Direct.CountLayers,
          '   Flow model layers: ', Flow.CountLayers);

  EvalSet(Direct, Test, gNumTest, d0L1, d0S);
  EvalSet(Flow,   Test, gNumTest, f0L1, f0S);
  WriteLn(Format('BEFORE training  DIRECT: L1=%.4f SSIM=%.4f   FLOW: L1=%.4f SSIM=%.4f',
    [d0L1, d0S, f0L1, f0S]));

  WriteLn('Training...');
  TrainOne(Direct, Train, gNumTrain, gEpochs, 'direct');
  TrainOne(Flow,   Train, gNumTrain, gEpochs, 'flow');

  EvalSet(Direct, Test, gNumTest, dL1, dS);
  EvalSet(Flow,   Test, gNumTest, fL1, fS);
  WriteLn;
  WriteLn(Format('FINAL held-out   DIRECT: L1=%.4f SSIM=%.4f   FLOW: L1=%.4f SSIM=%.4f',
    [dL1, dS, fL1, fS]));
  if fL1 < dL1 then
    WriteLn(Format('=> FLOW warping beats direct synthesis (L1 %.1f%% lower).',
      [100.0 * (dL1 - fL1) / Max(dL1, 1e-9)]))
  else
    WriteLn('=> (this smoke run is short; --full sharpens the flow advantage)');

  RenderPanel(Direct, Test[0], 'direct');
  RenderPanel(Flow,   Test[0], 'flow');
  DumpPPM(Direct, Test[0], 'frameinterp_direct.ppm');
  DumpPPM(Flow,   Test[0], 'frameinterp_flow.ppm');

  Direct.Free; Flow.Free;
  for i := 0 to gNumTrain - 1 do Train[i].Free;
  for i := 0 to gNumTest - 1 do Test[i].Free;
  WriteLn('Done.');
end.
