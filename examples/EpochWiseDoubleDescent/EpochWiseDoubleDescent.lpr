program EpochWiseDoubleDescent;
(*
EpochWiseDoubleDescent: reproduces the THIRD ("epoch-wise") axis of double
descent from Nakkiran et al. 2020 ("Deep Double Descent: Where Bigger Models
and More Data Hurt") on a pure-CPU toy, using only existing in-tree layers (no
new layer is added).

The phenomenon (the whole point of this example):
  A FIXED, mildly over-parameterised network is trained on a SMALL, label-NOISY
  classification set, and held-out TEST error is charted against TRAINING EPOCH.
  Over TIME the test-error curve is NON-monotone:
    1. it FALLS  -- classical learning: the net first captures the clean,
       low-complexity signal that generalises;
    2. it RISES to an interior PEAK around the epoch where TRAIN error hits ~0
       and the net is forced to INTERPOLATE (memorise) the noisy labels --
       fitting the noise wrecks generalisation;
    3. it FALLS AGAIN with continued training, as the interpolating solution
       drifts toward a flatter, better-generalising minimum.
  The only swept axis is EPOCH COUNT. Capacity and weight decay are FIXED.

This is the SIBLING of examples/DoubleDescent/ but a FORK in spirit:
  - DoubleDescent/  sweeps model CAPACITY (width H) at the END of training and
    finds the test-error peak at the interpolation THRESHOLD -- the MODEL-WISE
    axis. Capacity is the variable; time is fixed (train to convergence).
  - HERE capacity AND weight decay are FIXED and the only variable is the
    EPOCH count -- the EPOCH-WISE / TEMPORAL axis. The peak is in time.
  - This is ALSO distinct from grokking: grokking is delayed generalisation at
    fixed capacity driven by WEIGHT DECAY on CLEAN labels (test error stays bad
    then suddenly drops). Here labels carry NOISE, weight decay is fixed (0 or
    tiny), and the signature is the down-UP-down test curve in time, not a late
    sudden jump.

Task: a small fixed binary-classification problem (same idiom as the sibling,
which uses sign(w*.x); here we use well-separated Gaussian BLOBS so the clean
signal is EASY enough that the early test-error valley gets genuinely low --
the prerequisite for a visible "down" before the noise-driven "up").
  - A "teacher" of two well-separated Gaussian blobs (one per class) assigns
    each point a clean, low-complexity, easily-learnable label.
  - A SMALL training set (cTrain points) gets ~18% of its labels FLIPPED
    (label noise). A LARGE CLEAN test set (no flips) measures honest
    generalisation. The clean blob signal is what the net learns first; the
    flipped labels are what it is forced to memorise later (the peak driver).

Net: Input -> FullConnectReLU x2 -> FullConnectLinear -> SoftMax, mildly
over-parameterised relative to cTrain so it can eventually interpolate. Trained
as a SoftMax classifier (one-hot targets) with plain full-batch gradient
descent; NO NeuralFit, so test error is logged deterministically every K
epochs. Class is read back as argmax of the softmax output.

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
  cDim        = 6;        // input feature dimension
  cClasses    = 2;        // binary classification
  cTrain      = 50;       // SMALL training set (label noise injected)
  cTest       = 4000;     // LARGE clean held-out test set
  cNoiseFrac  = 0.18;     // 18% of TRAIN labels flipped (the peak driver)
  cBlobSep    = 1.6;      // class-centre separation (clean signal is EASY to
                          // learn early -> the test curve reaches a low valley
                          // before the noise is memorised)
  cHidden     = 64;       // FIXED hidden width (mildly over-parameterised)
  cBatch      = 5;        // mini-batch size: SGD gradient noise spreads the
                          // noise-memorisation phase over many epochs so the
                          // temporal peak is RESOLVABLE
  cEpochs     = 3000;     // single swept axis: how long we train
  cLogEvery   = 20;       // log train/test error every K epochs
  cLR         = 0.01;     // small SGD learning rate (resolves the phases)
  cMomentum   = 0.0;      // no momentum (it accelerates noise memorisation)
  cWeightDecay= 0.0;      // FIXED weight decay (no grokking-style L2 schedule)
  cSeed       = 424242;

type
  TLogRow = record
    Epoch    : integer;
    TrainErr : TNeuralFloat;   // 0/1 error on the (noisy) train labels
    TestErr  : TNeuralFloat;   // 0/1 error on the clean test set
  end;

var
  // Fixed "teacher": two well-separated Gaussian blobs, one per class. Blob
  // membership is a clean, LOW-COMPLEXITY, easily-learnable target -- the net
  // captures it EARLY (test error drops to a low valley). The flipped labels
  // are the high-complexity NOISE the net is forced to memorise LATER, which is
  // what lifts the interior peak. Reseeding before generation keeps the teacher
  // fixed and the data deterministic.
  Centers: array[0..cClasses - 1, 0..cDim - 1] of TNeuralFloat;

procedure MakeTeacher;
var
  C, J: integer;
begin
  for C := 0 to cClasses - 1 do
    for J := 0 to cDim - 1 do
      Centers[C, J] := RandG(0, 1) * cBlobSep;
end;

// Build a labelled set. Targets are one-hot over cClasses (SoftMax head). If
// FlipFrac>0 that fraction of labels is flipped to the other class.
procedure BuildSet(out Pairs: TNNetVolumePairList; Count: integer;
  FlipFrac: TNeuralFloat);
var
  I, J, F, NumFlip, Tmp, Cls: integer;
  X, Y: TNNetVolume;
  Order: array of integer;
begin
  Pairs := TNNetVolumePairList.Create();
  for I := 0 to Count - 1 do
  begin
    Cls := Random(cClasses);
    X := TNNetVolume.Create(cDim, 1, 1);
    for J := 0 to cDim - 1 do
      X.FData[J] := Centers[Cls, J] + RandG(0, 1);   // blob jitter
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.SetClassForSoftMax(Cls);   // one-hot {0,1}
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
      // flip one-hot to the other class
      if Y.GetClass() = 1 then Y.SetClassForSoftMax(0)
      else Y.SetClassForSoftMax(1);
    end;
  end;
end;

// The FIXED net: Input -> FullConnectReLU(H) -> FullConnectReLU(H) ->
// FullConnectLinear(cClasses) -> SoftMax. Mildly over-parameterised relative to
// cTrain so it can eventually interpolate the noisy labels.
procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cDim, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLR, cMomentum);
  NN.SetL2Decay(cWeightDecay);
  // Mini-batch SGD: accumulate the gradient over a small batch, then one step.
  // The per-batch gradient noise spreads the noise-memorisation phase over many
  // epochs, which is exactly what makes the temporal peak resolvable.
  NN.SetBatchUpdate(True);
end;

// 0/1 classification error (argmax of softmax output vs argmax of one-hot).
function ZeroOneError(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I, Wrong: integer;
begin
  Wrong := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    if NN.GetLastLayer().Output.GetClass() <> Pairs[I].O.GetClass() then
      Inc(Wrong);
  end;
  Result := Wrong / Pairs.Count;
end;

// ---------------------------------------------------------------------------
// Reporting helpers.
// ---------------------------------------------------------------------------

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

// First logged row whose train error is ~0 (the interpolation epoch).
function InterpolationIndex(const Rows: array of TLogRow): integer;
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

// Index of the early VALLEY: the lowest test error in the EARLY (pre-peak)
// phase. We search up to the interpolation epoch (or, if the net never quite
// interpolates, up to the global argmax of test error) -- this is the bottom of
// the first descent, from which the noise-memorisation peak rises.
function ValleyIndex(const Rows: array of TLogRow; UpTo: integer): integer;
var
  K, Hi: integer;
  Best: TNeuralFloat;
begin
  Result := 0;
  Best := 1e30;
  Hi := High(Rows);
  if (UpTo >= 0) and (UpTo < Hi) then Hi := UpTo;
  for K := 0 to Hi do
    if Rows[K].TestErr < Best then
    begin
      Best := Rows[K].TestErr;
      Result := K;
    end;
end;

// Index of the interior PEAK: the maximum test error AT or AFTER the early
// valley. (The very first epochs can be high from plain underfitting, so the
// genuine epoch-wise peak is the rise that FOLLOWS the valley.)
function PeakIndex(const Rows: array of TLogRow; ValIdx: integer): integer;
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

var
  NN: TNNet;
  TrainSet, TestSet: TNNetVolumePairList;
  Rows: array of TLogRow;
  Order: array of integer;
  Epoch, I, J, Tmp, NLog, Idx: integer;
  StartTime, EndTime: TDateTime;
  IntpIdx, ValIdx, PkIdx, SearchHi: integer;
  ValleyV, PeakV, FinalV: TNeuralFloat;
  PassInterp, PassPeak: boolean;
  Mark: string[3];
begin
  // Deterministic, single-threaded: each Compute/Backpropagate runs one sample
  // at a time, so the forward/backward passes are effectively single-threaded.
  RandSeed := cSeed;
  NN := nil;

  WriteLn('================================================================');
  WriteLn('Epoch-wise (TEMPORAL) Double Descent: test error vs EPOCH.');
  WriteLn('================================================================');
  WriteLn(Format('Teacher: %d separated Gaussian blobs, D=%d.  Train=%d (%.0f%% '
    + 'label noise),', [cClasses, cDim, cTrain, 100 * cNoiseFrac]));
  WriteLn(Format('Test=%d (clean).  FIXED net: Input(%d)->FCReLU(%d)->FCReLU(%d)',
    [cTest, cDim, cHidden, cHidden]));
  WriteLn(Format('->FCLinear(%d)->SoftMax.  Only swept axis = EPOCH.', [cClasses]));
  WriteLn(Format('Epochs=%d  log every %d  LR=%.3f  momentum=%.2f  wd=%.3f  seed=%d',
    [cEpochs, cLogEvery, cLR, cMomentum, cWeightDecay, cSeed]));
  WriteLn('Capacity and weight decay are FIXED (contrast: DoubleDescent/ sweeps');
  WriteLn('CAPACITY; grokking is weight-decay-driven on CLEAN labels).');
  WriteLn;

  // Build the fixed teacher, the noisy train set and the large clean test set.
  MakeTeacher;
  BuildSet(TrainSet, cTrain, cNoiseFrac);
  BuildSet(TestSet, cTest, 0.0);   // test set is ALWAYS clean

  // Build the single FIXED net (reseed so init is deterministic w.r.t. seed).
  RandSeed := cSeed;
  BuildNet(NN);
  WriteLn(Format('Net parameters (CountWeights) = %d  vs  train points = %d',
    [NN.CountWeights(), cTrain]));
  WriteLn;

  SetLength(Order, TrainSet.Count);
  for I := 0 to High(Order) do Order[I] := I;

  NLog := (cEpochs div cLogEvery) + 1;
  SetLength(Rows, NLog);
  Idx := 0;

  StartTime := Now;
  Write('Training (epoch-wise): ');
  for Epoch := 0 to cEpochs do
  begin
    // Log BEFORE the step at epoch 0 (initial), then every cLogEvery epochs.
    if (Epoch mod cLogEvery = 0) then
    begin
      Rows[Idx].Epoch    := Epoch;
      Rows[Idx].TrainErr := ZeroOneError(NN, TrainSet);
      Rows[Idx].TestErr  := ZeroOneError(NN, TestSet);
      Inc(Idx);
      Write('.');
    end;
    if Epoch = cEpochs then Break;

    // One epoch of mini-batch SGD: shuffle the index order, then walk it in
    // chunks of cBatch, accumulating each batch's gradient and taking one
    // UpdateWeights step per batch.
    for I := High(Order) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    I := 0;
    while I < TrainSet.Count do
    begin
      NN.ClearDeltas();
      for J := I to Min(I + cBatch - 1, TrainSet.Count - 1) do
      begin
        NN.Compute(TrainSet[Order[J]].I);
        NN.Backpropagate(TrainSet[Order[J]].O);
      end;
      NN.UpdateWeights();
      Inc(I, cBatch);
    end;
  end;
  WriteLn;
  EndTime := Now;

  SetLength(Rows, Idx);  // trim to actually-logged rows

  // ---- Locate the down-up-down signature ----
  IntpIdx := InterpolationIndex(Rows);
  if IntpIdx >= 0 then SearchHi := IntpIdx
  else
  begin
    // Net never quite interpolated: fall back to the global test-error argmax
    // as the upper bound of the early-valley search.
    SearchHi := 0;
    PeakV := Rows[0].TestErr;
    for I := 1 to High(Rows) do
      if Rows[I].TestErr > PeakV then begin PeakV := Rows[I].TestErr; SearchHi := I; end;
  end;
  ValIdx := ValleyIndex(Rows, SearchHi);
  PkIdx  := PeakIndex(Rows, ValIdx);

  // ---- Print the two-row ASCII curve ----
  WriteLn;
  WriteLn('  (V=early valley, ^=interior PEAK, I=interpolation epoch)');
  WriteLn('     epoch  trErr  testErr | test-error curve (bar scaled x2, 0.5=full)');
  for I := 0 to High(Rows) do
  begin
    if (I = ValIdx) or (I = PkIdx) or (I = IntpIdx)
       or (I mod 4 = 0) or (I = High(Rows)) then
    begin
      Mark := '   ';
      if I = IntpIdx then Mark[1] := 'I';
      if I = ValIdx  then Mark[2] := 'V';
      if I = PkIdx   then Mark[3] := '^';
      WriteLn(Format('%s %6d  %5.3f   %5.3f  |%s',
        [Mark, Rows[I].Epoch, Rows[I].TrainErr, Rows[I].TestErr,
         Bar(2 * Rows[I].TestErr, 30)]));
    end;
  end;

  WriteLn;
  WriteLn('=== Self-gate: the GENUINE epoch-wise invariants ===');

  // Gate 1: train error reaches ~0 (interpolation actually happens).
  PassInterp := (IntpIdx >= 0);
  if PassInterp then
    WriteLn(Format('[PASS] interpolation: train 0/1 error first hits ~0 at epoch %d.',
      [Rows[IntpIdx].Epoch]))
  else
    WriteLn('[FAIL] interpolation: train error never reached ~0 -- the net did '
      + 'not interpolate the noisy labels (raise epochs / width / LR).');

  // Gate 2: test trajectory is NON-MONOTONE with an INTERIOR peak strictly
  // above BOTH its earlier valley AND its final value (down-up-down).
  ValleyV := Rows[ValIdx].TestErr;
  PeakV   := Rows[PkIdx].TestErr;
  FinalV  := Rows[High(Rows)].TestErr;
  PassPeak := (PkIdx > ValIdx) and (PkIdx < High(Rows))
              and (PeakV > ValleyV + 0.005)
              and (PeakV > FinalV + 0.005);
  if PassPeak then
    WriteLn(Format('[PASS] interior peak: test err valley=%.3f (epoch %d) -> '
      + 'PEAK=%.3f (epoch %d) -> final=%.3f (epoch %d). Down-up-down confirmed.',
      [ValleyV, Rows[ValIdx].Epoch, PeakV, Rows[PkIdx].Epoch,
       FinalV, Rows[High(Rows)].Epoch]))
  else
    WriteLn(Format('[FAIL] interior peak: valley=%.3f (ep %d) peak=%.3f (ep %d) '
      + 'final=%.3f (ep %d) -- no strict down-up-down. Tune knob that failed: if '
      + 'peak~valley raise NOISE; if peak at the edge raise/lower EPOCHS; if '
      + 'peak~final lower LR so the second descent is resolvable.',
      [ValleyV, Rows[ValIdx].Epoch, PeakV, Rows[PkIdx].Epoch,
       FinalV, Rows[High(Rows)].Epoch]));

  WriteLn;
  if PassInterp and PassPeak then
    WriteLn('=> ALL GATES PASS: epoch-wise (temporal) double descent reproduced.')
  else
    WriteLn('=> GATE FAILURE (see above).');
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  TrainSet.Free;
  TestSet.Free;
  NN.Free;

  if not (PassInterp and PassPeak) then
    Halt(1);
end.
