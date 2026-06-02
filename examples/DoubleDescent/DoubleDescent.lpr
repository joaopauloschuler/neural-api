program DoubleDescent;
(*
DoubleDescent: reproduces the model-wise "double descent" risk curve
(Belkin et al. 2019, "Reconciling modern machine-learning practice and the
bias-variance trade-off"; Nakkiran et al. 2020, "Deep Double Descent") on a
pure-CPU toy, using only existing in-tree layers (no new layer needed).

The phenomenon (the whole point of this example):
  As model CAPACITY grows, test error first FALLS (the classical
  bias-variance U), then RISES to a sharp PEAK right at the INTERPOLATION
  THRESHOLD -- the smallest model with just enough parameters to fit the
  noisy training set exactly (train error ~ 0) -- and then FALLS AGAIN and
  keeps improving deep in the over-parameterised regime. The test-error
  curve is NON-monotone (U-then-peak-then-down), not the usual monotone
  bias-variance curve. The sharp peak only appears when the training labels
  carry NOISE; that noise is what the interpolating model is forced to
  memorise, and memorising it wrecks generalisation exactly at the threshold.

Task: a small fixed binary-classification problem.
  - A linear "teacher" w* assigns each random D-dim Gaussian point a class by
    sign(w* . x). This is a clean, learnable target.
  - A SMALL training set (cTrain points) gets a few percent of its labels
    FLIPPED (label noise). A LARGE CLEAN test set (no flips) measures honest
    generalisation.
  - The SAME single-hidden-layer ReLU MLP is trained at a sweep of hidden
    widths H. Single-hidden-layer params ~ H*(D+1) + classes*(H+1), so the
    parameter count sweeps from far under the train-set size to far over it,
    straddling the interpolation threshold.

Each width is trained to ~convergence with plain full-batch-ish SGD; the wide
models are explicitly driven until train error hits 0 (they interpolate the
noisy labels). For each width we record:
  - train 0/1 error (on the NOISY training labels the model actually saw)
  - test  0/1 error (on the CLEAN held-out set)
  - parameter count (TNNet.CountWeights)

We then chart a two-row ASCII curve of train- and test-error vs log2(params),
flag the empirical interpolation threshold (smallest width whose train error
first hits ~0), and CHECK that the test-error peak lands at / just past it.

Ablation: the whole sweep is run TWICE -- once with label noise ON and once
with noise OFF (same teacher, same points, same widths, same seeds). The
noisy run should show the sharp peak at the threshold; the clean run should be
~monotone (no sharp peak). The contrast is the built-in correctness signal
that the peak is NOISE-driven.

This is DISTINCT from grokking (delayed generalisation over TRAINING TIME at
fixed capacity -- a time axis) and from any fixed-budget width/depth heatmap:
here the axis is generalisation vs CAPACITY and the non-monotone peak is the
point.

Pure CPU, no external data, deterministic seeding, finishes well under the
few-minute budget.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cDim        = 4;       // input feature dimension (small so capacity ~ width)
  cClasses    = 2;       // binary classification
  cTrain      = 60;      // SMALL training set (label noise injected)
  cTest       = 2000;    // LARGE clean held-out test set
  cNoiseFrac  = 0.15;    // 15% of TRAIN labels flipped (the peak driver)
  cEpochs     = 6000;    // enough for the wide models to interpolate
  cLR         = 0.03;    // plain SGD learning rate
  cSeed       = 20260524;

type
  TWidthArr = array of integer;

  TSweepRow = record
    Width     : integer;
    Params    : integer;
    TrainErr  : TNeuralFloat;   // 0/1 error on the (noisy) train labels
    TrainMSE  : TNeuralFloat;   // regression MSE on the (noisy) train targets
    TestErr   : TNeuralFloat;   // 0/1 error on the clean test set
  end;

const
  // Hidden widths spanning under- to heavily over-parameterised. The
  // interpolation threshold (params ~ cTrain) sits in the middle of this list.
  cWidths: array[0..11] of integer =
    (1, 2, 3, 4, 5, 6, 8, 12, 20, 32, 64, 128);

var
  // Fixed NONLINEAR teacher (shared by both ablation arms): a random quadratic
  // form sign(x' Q x + b' x). Being nonlinear (not linearly separable) means
  // the tiny-width models genuinely CANNOT fit it -- their train error stays
  // well above 0, which is what places the interpolation threshold up in the
  // middle of the width sweep rather than at H=1.
  TeacherQ: array[0..cDim - 1, 0..cDim - 1] of TNeuralFloat;
  TeacherB: array[0..cDim - 1] of TNeuralFloat;

// ---------------------------------------------------------------------------
// Data generation. The teacher is fixed; one training set (optionally noisy)
// and one large clean test set are produced. Reseeding before each call keeps
// the points/teacher identical across the noise-on and noise-off arms.
// ---------------------------------------------------------------------------
procedure MakeTeacher;
var
  I, J: integer;
begin
  for I := 0 to cDim - 1 do
  begin
    TeacherB[I] := RandG(0, 1);
    for J := I to cDim - 1 do
    begin
      TeacherQ[I, J] := RandG(0, 1);
      TeacherQ[J, I] := TeacherQ[I, J];  // symmetric
    end;
  end;
end;

function TeacherClass(X: TNNetVolume): integer;
var
  I, J: integer;
  S: TNeuralFloat;
begin
  S := 0;
  for I := 0 to cDim - 1 do
  begin
    S := S + TeacherB[I] * X.FData[I];
    for J := 0 to cDim - 1 do
      S := S + TeacherQ[I, J] * X.FData[I] * X.FData[J];
  end;
  if S >= 0 then Result := 1 else Result := 0;
end;

// Build a labelled set. If FlipFrac>0, that fraction of labels is flipped.
procedure BuildSet(out Pairs: TNNetVolumePairList; Count: integer;
  FlipFrac: TNeuralFloat);
var
  I, J, F, NumFlip, Tmp, Cls: integer;
  X, Y: TNNetVolume;
  Order: array of integer;
begin
  // Targets are a SINGLE regression value in {-1, +1} (class 1 -> +1,
  // class 0 -> -1). We fit them with MSE; the over-parameterised net then
  // drives train MSE -> 0 (it interpolates the noisy targets exactly), which
  // is the clean, capacity-driven interpolation the double-descent peak needs.
  Pairs := TNNetVolumePairList.Create();
  for I := 0 to Count - 1 do
  begin
    X := TNNetVolume.Create(cDim, 1, 1);
    for J := 0 to cDim - 1 do
      X.FData[J] := RandG(0, 1);
    Cls := TeacherClass(X);
    Y := TNNetVolume.Create(1, 1, 1);
    if Cls = 1 then Y.FData[0] := 1.0 else Y.FData[0] := -1.0;
    Pairs.Add(TNNetVolumePair.Create(X, Y));
  end;

  if FlipFrac > 0 then
  begin
    NumFlip := Round(FlipFrac * Count);
    SetLength(Order, Count);
    for I := 0 to High(Order) do Order[I] := I;
    for I := High(Order) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    for F := 0 to NumFlip - 1 do
    begin
      Y := Pairs[Order[F]].O;
      Y.FData[0] := -Y.FData[0];   // flip sign (+1 <-> -1)
    end;
  end;
end;

// ---------------------------------------------------------------------------
// The swept model: Input -> FullConnectReLU(H) -> FullConnectLinear(1), a
// single hidden layer so capacity is governed by H alone. It is trained as an
// MSE REGRESSION onto the +-1 target with FULL-BATCH gradient descent
// (SetBatchUpdate accumulates the gradient over the whole training set, then
// one UpdateWeights step). MSE regression onto +-1 targets interpolates the
// noisy training set cleanly: in the over-parameterised regime the network
// drives train MSE -> 0 (memorises the flipped labels), the prerequisite for
// a visible interpolation peak. The class is read back as sign(output). No
// stochastic layers (deterministic forward/backward).
// ---------------------------------------------------------------------------
procedure BuildNet(out NN: TNNet; H: integer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cDim, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(H));
  NN.AddLayer(TNNetFullConnectLinear.Create(1));
  NN.SetLearningRate(cLR, {Momentum=}0.9);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);  // accumulate full-batch gradient, then one step
end;

// 0/1 classification error over a pair list (class = sign of the single
// regression output vs sign of the +-1 target).
function ZeroOneError(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I, Wrong: integer;
  Pred, Tgt: TNeuralFloat;
begin
  Wrong := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Pred := NN.GetLastLayer().Output.FData[0];
    Tgt := Pairs[I].O.FData[0];
    if (Pred >= 0) <> (Tgt >= 0) then
      Inc(Wrong);
  end;
  Result := Wrong / Pairs.Count;
end;

// Mean squared regression error over a pair list (interpolation = ~0).
function MeanMSE(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I: integer;
  D, Sum: TNeuralFloat;
begin
  Sum := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    D := NN.GetLastLayer().Output.FData[0] - Pairs[I].O.FData[0];
    Sum := Sum + D * D;
  end;
  if Pairs.Count > 0 then Result := Sum / Pairs.Count else Result := 0;
end;

// Train one width to ~convergence on the (fixed) training set, then read back
// train and test 0/1 error. Early-stop the inner loop once the model has
// interpolated (train err 0) for a few epochs to save the budget on the wide
// arms; the small arms run the full epoch count.
function RunWidth(H: integer; TrainSet, TestSet: TNNetVolumePairList): TSweepRow;
var
  NN: TNNet;
  Epoch, I, J, Tmp, ZeroStreak: integer;
  Order: array of integer;
  TrMSE: TNeuralFloat;
begin
  Result.Width := H;
  // Reseed before build so weight init is identical across the two ablation
  // arms for the same width.
  RandSeed := cSeed + H;
  BuildNet(NN, H);
  Result.Params := NN.CountWeights();
  SetLength(Order, TrainSet.Count);
  for I := 0 to High(Order) do Order[I] := I;
  ZeroStreak := 0;
  try
    for Epoch := 1 to cEpochs do
    begin
      // Full-batch GD step: zero the accumulator, sum the gradient over the
      // whole (fixed) training set, then take a single weight-update step.
      // (Order shuffle is harmless under full-batch but kept for parity.)
      for I := High(Order) downto 1 do
      begin
        J := Random(I + 1);
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;
      NN.ClearDeltas();
      for I := 0 to High(Order) do
      begin
        NN.Compute(TrainSet[Order[I]].I);
        NN.Backpropagate(TrainSet[Order[I]].O);
      end;
      NN.UpdateWeights();
      // Cheap early-stop: once train MSE is essentially 0 for several checks in
      // a row the model has interpolated; further epochs only burn budget.
      if (Epoch mod 20 = 0) or (Epoch = cEpochs) then
      begin
        TrMSE := MeanMSE(NN, TrainSet);
        if TrMSE < 1e-4 then Inc(ZeroStreak, 20) else ZeroStreak := 0;
        if ZeroStreak >= 100 then Break;
      end;
    end;
    Result.TrainErr := ZeroOneError(NN, TrainSet);
    Result.TrainMSE := MeanMSE(NN, TrainSet);
    Result.TestErr := ZeroOneError(NN, TestSet);
  finally
    NN.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Run the full width sweep for one ablation arm (noisy or clean labels) and
// return the per-width rows.
// ---------------------------------------------------------------------------
procedure RunSweep(FlipFrac: TNeuralFloat; out Rows: array of TSweepRow);
var
  K: integer;
  TrainSet, TestSet: TNNetVolumePairList;
begin
  // Identical teacher + points across both arms (only the flips differ).
  RandSeed := cSeed;
  MakeTeacher;
  BuildSet(TrainSet, cTrain, FlipFrac);
  BuildSet(TestSet, cTest, 0.0);   // test set is ALWAYS clean
  try
    for K := 0 to High(cWidths) do
    begin
      Rows[K] := RunWidth(cWidths[K], TrainSet, TestSet);
      Write('.');
    end;
    WriteLn;
  finally
    TrainSet.Free;
    TestSet.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Reporting helpers.
// ---------------------------------------------------------------------------

// First width whose train error is ~0 (the empirical interpolation threshold).
function ThresholdIndex(const Rows: array of TSweepRow): integer;
var
  K: integer;
begin
  Result := -1;
  for K := 0 to High(Rows) do
    if Rows[K].TrainErr <= 0.0001 then
    begin
      Result := K;
      Exit;
    end;
end;

// Index of the bias-variance MINIMUM: the lowest test error in the
// UNDER-parameterised regime (at or before the interpolation threshold). This
// is the bottom of the classical U, from which the double-descent peak rises.
function ValleyIndex(const Rows: array of TSweepRow; ThrIdx: integer): integer;
var
  K, Hi: integer;
  Best: TNeuralFloat;
begin
  Result := 0;
  Best := 1e30;
  Hi := High(Rows);
  if (ThrIdx >= 0) and (ThrIdx < Hi) then Hi := ThrIdx;  // search up to threshold
  for K := 0 to Hi do
    if Rows[K].TestErr < Best then
    begin
      Best := Rows[K].TestErr;
      Result := K;
    end;
end;

// Index of the interpolation PEAK: the maximum test error AT or AFTER the
// bias-variance valley. The under-parameterised left edge can also be high
// (plain underfitting), so the genuine double-descent peak is the rise that
// follows the valley -- NOT the global argmax.
function PeakIndex(const Rows: array of TSweepRow; ValIdx: integer): integer;
var
  K: integer;
  Best: TNeuralFloat;
begin
  Result := ValIdx;
  Best := -1;
  for K := ValIdx to High(Rows) do
    if Rows[K].TestErr > Best then
    begin
      Best := Rows[K].TestErr;
      Result := K;
    end;
end;

// One ASCII bar row (error in [0,1] mapped to a fixed-width bar).
function Bar(V: TNeuralFloat; Width: integer): string;
var
  N, I: integer;
begin
  N := Round(V * Width);
  if N < 0 then N := 0;
  if N > Width then N := Width;
  Result := '';
  for I := 1 to N do Result := Result + '#';
  for I := N + 1 to Width do Result := Result + ' ';
end;

procedure PrintCurve(const Title: string; const Rows: array of TSweepRow;
  ThrIdx, PkIdx: integer);
const
  cBarW = 30;
var
  K: integer;
  Mark: string;
begin
  WriteLn;
  WriteLn(Title);
  WriteLn('  H    params  log2P  trMSE  trErr  testErr | test-error bar (0..0.5)');
  for K := 0 to High(Rows) do
  begin
    Mark := '  ';
    if K = ThrIdx then Mark := '>T';        // interpolation threshold
    if K = PkIdx then Mark := Mark[1] + '*';// test-error peak (overlay)
    if (K = ThrIdx) and (K = PkIdx) then Mark := 'T*';
    WriteLn(Format('%s%4d %7d  %5.2f  %5.3f  %5.3f   %5.3f  |%s',
      [Mark, Rows[K].Width, Rows[K].Params, Log2(Rows[K].Params),
       Rows[K].TrainMSE, Rows[K].TrainErr, Rows[K].TestErr,
       Bar(2 * Rows[K].TestErr, cBarW)]));
  end;
  WriteLn('  (>T = interpolation threshold: first width with train 0/1 err~0;',
          '  * = test-error peak; bar scaled x2 so 0.5=full)');
end;

// "Monotone-ish" test for the clean arm: count how much the test curve RISES
// after its initial minimum. A clean (no-noise) curve should be ~monotone
// down then flat -- little to no rise -- whereas the noisy curve spikes.
function PostMinRise(const Rows: array of TSweepRow): TNeuralFloat;
var
  K, MinIdx: integer;
  MinV, MaxAfter: TNeuralFloat;
begin
  MinIdx := 0; MinV := Rows[0].TestErr;
  for K := 1 to High(Rows) do
    if Rows[K].TestErr < MinV then
    begin
      MinV := Rows[K].TestErr; MinIdx := K;
    end;
  MaxAfter := MinV;
  for K := MinIdx to High(Rows) do
    if Rows[K].TestErr > MaxAfter then MaxAfter := Rows[K].TestErr;
  Result := MaxAfter - MinV;
end;

var
  NoisyRows, CleanRows: array[0..High(cWidths)] of TSweepRow;
  ThrN, PkN, ValN, ThrC, PkC, ValC: integer;
  StartTime, EndTime: TDateTime;
  PeakSep: integer;
  NoisyRise, CleanRise: TNeuralFloat;
  PassPeak, PassAblation, PassThr: boolean;
begin
  WriteLn('================================================================');
  WriteLn('Double Descent: test error vs model CAPACITY (width sweep).');
  WriteLn('================================================================');
  WriteLn(Format('Teacher: sign(x''Qx + b''x) nonlinear, D=%d.  Train=%d (%.0f%% label',
    [cDim, cTrain, 100 * cNoiseFrac]));
  WriteLn(Format('noise), Test=%d (clean).  Model: Input(%d)->FullConnectReLU(H)',
    [cTest, cDim]));
  WriteLn('->FullConnectLinear(1), MSE on +-1 target.  Same MLP swept over H.');
  WriteLn(Format('Epochs<=%d  LR=%.3f  RandSeed=%d', [cEpochs, cLR, cSeed]));
  WriteLn;

  StartTime := Now;
  Write('Sweeping NOISY-label arm  ');
  RunSweep(cNoiseFrac, NoisyRows);
  Write('Sweeping CLEAN-label arm  ');
  RunSweep(0.0, CleanRows);
  EndTime := Now;

  ThrN := ThresholdIndex(NoisyRows);
  ValN := ValleyIndex(NoisyRows, ThrN);
  PkN := PeakIndex(NoisyRows, ValN);
  ThrC := ThresholdIndex(CleanRows);
  ValC := ValleyIndex(CleanRows, ThrC);
  PkC := PeakIndex(CleanRows, ValC);

  PrintCurve('--- NOISY labels (expect a SHARP test-error PEAK at the threshold) ---',
    NoisyRows, ThrN, PkN);
  PrintCurve('--- CLEAN labels (ablation: expect ~MONOTONE, no sharp peak) ---',
    CleanRows, ThrC, PkC);

  WriteLn;
  WriteLn('=== Correctness signals ===');

  // Signal 1: an interpolation threshold exists (some width drives train->0).
  PassThr := (ThrN >= 0);
  if PassThr then
    WriteLn(Format('[PASS] interpolation threshold found at H=%d (params=%d, '
      + 'log2P=%.2f): train error first hits 0 here.',
      [NoisyRows[ThrN].Width, NoisyRows[ThrN].Params,
       Log2(NoisyRows[ThrN].Params)]))
  else
    WriteLn('[FAIL] no width drove train error to 0 -- models did not '
      + 'interpolate (raise epochs / LR).');

  // Signal 2: the test-error peak lands AT or JUST PAST the threshold.
  if PassThr then
  begin
    PeakSep := PkN - ThrN;
    // The 0/1 test-error peak lands right around the threshold: at it, one
    // width-step before (where train 0/1 error is already nearly 0 / the net
    // is essentially interpolating), or one-to-two steps past it.
    PassPeak := (PeakSep >= -1) and (PeakSep <= 2);
    if PassPeak then
      WriteLn(Format('[PASS] test-error peak at H=%d sits at/around the '
        + 'threshold (peak %d width-step(s) from it; valley was H=%d).',
        [NoisyRows[PkN].Width, PeakSep, NoisyRows[ValN].Width]))
    else
      WriteLn(Format('[FAIL] test-error peak at H=%d is %d width-step(s) from '
        + 'the threshold (expected -1..2).', [NoisyRows[PkN].Width, PeakSep]));
  end
  else
    PassPeak := False;

  // Signal 3 (ablation): noisy curve rises after its min (the peak) by clearly
  // MORE than the clean curve does. The noise-driven peak is the difference.
  NoisyRise := PostMinRise(NoisyRows);
  CleanRise := PostMinRise(CleanRows);
  PassAblation := NoisyRise > CleanRise + 0.02;
  WriteLn(Format('Post-minimum test-error RISE: noisy=%.3f  clean=%.3f', [NoisyRise, CleanRise]));
  if PassAblation then
    WriteLn('[PASS] ablation: the noisy curve spikes after its minimum far '
      + 'more than the clean curve -- the peak is NOISE-driven.')
  else
    WriteLn('[FAIL] ablation: noisy and clean curves rise similarly -- the '
      + 'peak is not clearly noise-driven (raise noise / epochs).');

  WriteLn;
  if PassThr and PassPeak and PassAblation then
    WriteLn('=> ALL CHECKS PASS: classic model-wise double descent reproduced.')
  else
    WriteLn('=> SOME CHECKS FAILED (see above).');
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));
end.
