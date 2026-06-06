program SplineKnotSweep;
(*
SplineKnotSweep: a knot-count / Range capacity sweep for the per-channel
learnable activation TNNetSplineActivation (a Kolmogorov-Arnold Network style
piecewise-linear activation, KAN / Liu et al. 2024).

This is a follow-up to the landed examples/SplineActivationKAN/ demo. There we
showed the spline activation can fit a wiggly toy target at matched parameter
count; here we ask the natural next question: how many KNOTS does it take, and
what does extra capacity buy?

Spline geometry recap. TNNetSplineActivation(K, Range) places K+1 control
points (knots) uniformly on [-Range, +Range] PER CHANNEL and linearly
interpolates between them (extrapolating linearly outside the range). K is the
number of intervals: bigger K = more knots = a more flexible per-channel
activation, costing (K+1)*Depth extra trainable values.

Experiment. We fix one tiny net:
    Input(1) -> FullConnect(W) -> TNNetSplineActivation(K, Range) -> Linear(1)
and sweep K in {2, 4, 8, 16} crossed with Range in {2.0, 4.0}. EVERYTHING else
is held identical across cells: same RandSeed (424242), same training data, same
optimizer, same epoch budget, same width W. Only K and Range change.

Target (wiggly, so a flexible activation has something to earn):
    y = sin(3x) + 0.3*sin(11x),  x in [-2, 2]

The training points are drawn from [-2, 2]; we also evaluate a HELD-OUT MSE on a
dense clean grid that extends slightly BEYOND the training span (into [-2.4,
2.4]) so that extra knots can overfit the wiggles inside the span without
necessarily helping outside it. For each (K, Range) cell we print final TRAIN
MSE and HELD-OUT MSE.

The story this run actually surfaces (and asserts): as K grows at a fixed Range,
HELD-OUT error first IMPROVES substantially (a low-K spline cannot represent the
two-frequency wiggle, so adding knots helps a lot), then STOPS improving and in
fact WORSENS at the largest K — the capacity / overfitting trade, with a clear
sweet spot at an intermediate K. (Note: with fixed-budget SGD, even TRAIN error
is NOT strictly monotone in K — the largest-K spline has more parameters to
optimise and, at a fixed epoch budget, does not always reach a lower train loss.
So we deliberately do NOT assert train-monotonicity; we assert the held-out
generalization story, which is the point of the experiment.)

Self-checking PASS/FAIL gate (Halt(1) on failure):
  (1) At each Range, adding knots HELPS at first: the best HELD-OUT MSE over the
      sweep is strictly better (by more than a slack) than the smallest-K
      held-out MSE — a low-K spline genuinely under-fits.
  (2) The smallest-K and largest-K models both drive TRAIN MSE below a modest
      threshold (both actually train, i.e. the comparison is fair).
  (3) For at least one Range, the held-out error at the LARGEST K is NOT better
      than its best held-out error over the sweep by more than a slack — i.e.
      piling on knots stops helping (or hurts) generalization.

Pure CPU, no dataset download, synthetic data generated in-code, modest threads,
well under a minute.

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

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cXRange   = 2.0;    // training inputs drawn from [-cXRange, +cXRange]
  cEvalPad  = 1.2;    // held-out grid spans [-cXRange*pad, +cXRange*pad]
  cNumTrain = 256;    // synthetic training samples
  cEpochs   = 500;    // same epoch budget for every cell
  cBatch    = 16;
  cLR       = 0.01;
  cMomentum = 0.9;
  cWidth    = 8;      // hidden width, fixed across the whole sweep
  cSeed     = 424242; // identical seed for every cell (repo idiom)

type
  TKArray = array of integer;
  TRArray = array of TNeuralFloat;

  // Target function: a wiggly 1D signal. Two incommensurate frequencies so a
  // single linear segment count cannot trivially nail it.
  function Target(X: TNeuralFloat): TNeuralFloat;
  begin
    Result := Sin(3 * X) + 0.3 * Sin(11 * X);
  end;

  procedure MakeTrainSet(out Xs, Ys: array of TNeuralFloat);
  var
    I: integer;
  begin
    for I := 0 to High(Xs) do
    begin
      Xs[I] := (Random - 0.5) * 2.0 * cXRange;  // x in [-cXRange, +cXRange)
      Ys[I] := Target(Xs[I]);
    end;
  end;

  // MSE on the training inputs (how well we fit what we trained on).
  function TrainMSE(NN: TNNet; const Xs, Ys: array of TNeuralFloat): TNeuralFloat;
  var
    I: integer;
    Diff, Sum: TNeuralFloat;
    Inp, Outp: TNNetVolume;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Outp := TNNetVolume.Create(1, 1, 1);
    Sum := 0;
    for I := 0 to High(Xs) do
    begin
      Inp.Raw[0] := Xs[I];
      NN.Compute(Inp);
      NN.GetOutput(Outp);
      Diff := Outp.Raw[0] - Ys[I];
      Sum := Sum + Diff * Diff;
    end;
    Inp.Free;
    Outp.Free;
    Result := Sum / Length(Xs);
  end;

  // HELD-OUT MSE on a dense clean grid extending beyond the training span.
  function HeldOutMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    X, Lo, Hi, Diff, Sum: TNeuralFloat;
    Inp, Outp: TNNetVolume;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Outp := TNNetVolume.Create(1, 1, 1);
    Lo := -cXRange * cEvalPad;
    Hi :=  cXRange * cEvalPad;
    Sum := 0;
    for I := 0 to 199 do
    begin
      X := Lo + (Hi - Lo) * I / 199.0;
      Inp.Raw[0] := X;
      NN.Compute(Inp);
      NN.GetOutput(Outp);
      Diff := Outp.Raw[0] - Target(X);
      Sum := Sum + Diff * Diff;
    end;
    Inp.Free;
    Outp.Free;
    Result := Sum / 200.0;
  end;

  // One full mini-batch SGD training run on the shared training set.
  procedure TrainCell(NN: TNNet; const Xs, Ys: array of TNeuralFloat);
  var
    Epoch, Step, I, J, Tmp: integer;
    Inp, Tgt: TNNetVolume;
    Order: array of integer;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Tgt := TNNetVolume.Create(1, 1, 1);
    SetLength(Order, Length(Xs));
    for I := 0 to High(Order) do Order[I] := I;
    NN.SetLearningRate(cLR, cMomentum);
    NN.SetL2Decay(0.0);
    try
      for Epoch := 1 to cEpochs do
      begin
        for I := High(Order) downto 1 do  // Fisher-Yates shuffle
        begin
          J := Random(I + 1);
          Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
        end;
        Step := 0;
        NN.ClearDeltas();
        for I := 0 to High(Order) do
        begin
          Inp.Raw[0] := Xs[Order[I]];
          Tgt.Raw[0] := Ys[Order[I]];
          NN.Compute(Inp);
          NN.Backpropagate(Tgt);
          Inc(Step);
          if Step = cBatch then
          begin
            NN.UpdateWeights();
            NN.ClearDeltas();
            Step := 0;
          end;
        end;
        if Step > 0 then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
        end;
      end;
    finally
      Inp.Free;
      Tgt.Free;
    end;
  end;

  // Build the swept net at a given (K, Range) and train it from the SAME seed.
  procedure RunCell(K: integer; Range: TNeuralFloat;
    const Xs, Ys: array of TNeuralFloat;
    out TrMSE, HoMSE: TNeuralFloat);
  var
    NN: TNNet;
  begin
    // Re-seed identically per cell so the ONLY differences are K and Range.
    RandSeed := cSeed;
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    NN.AddLayer(TNNetFullConnect.Create(cWidth));
    NN.AddLayer(TNNetSplineActivation.Create(K, Range));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.InitWeights();
    TrainCell(NN, Xs, Ys);
    TrMSE := TrainMSE(NN, Xs, Ys);
    HoMSE := HeldOutMSE(NN);
    NN.Free;
  end;

var
  Ks: TKArray;
  Rs: TRArray;
  Xs, Ys: array[0..cNumTrain - 1] of TNeuralFloat;
  Tr, Ho: array of array of TNeuralFloat;  // [range][k]
  ri, ki: integer;
  // gate bookkeeping
  cHelpSlack, cFitThreshold, cStallSlack: TNeuralFloat;
  HelpOK, FitOK, StallOK: boolean;
  BestHo, LastHo: TNeuralFloat;
begin
  Ks := TKArray.Create(2, 4, 8, 16);
  Rs := TRArray.Create(2.0, 4.0);
  cHelpSlack    := 1e-2;  // best held-out must beat smallest-K by at least this
  cFitThreshold := 0.20;  // both endpoint K must beat this train MSE
  cStallSlack   := 1e-3;  // "stops helping": largest-K not better than best by this

  WriteLn('SplineKnotSweep: knot-count (K) / Range capacity sweep for ',
    'TNNetSplineActivation.');
  WriteLn('Net: Input(1) -> FullConnect(', cWidth,
    ') -> SplineActivation(K,Range) -> Linear(1).');
  WriteLn('Target y = sin(3x) + 0.3*sin(11x).  Train x in [-', cXRange:0:1,
    ', ', cXRange:0:1, '];  held-out grid x in [-', cXRange*cEvalPad:0:1,
    ', ', cXRange*cEvalPad:0:1, '].');
  WriteLn('Identical seed (', cSeed, '), data, optimizer and ',
    cEpochs, ' epochs in every cell; only K and Range vary.');
  WriteLn;

  // The training set is generated ONCE so every cell sees identical data.
  RandSeed := cSeed;
  MakeTrainSet(Xs, Ys);

  SetLength(Tr, Length(Rs), Length(Ks));
  SetLength(Ho, Length(Rs), Length(Ks));

  for ri := 0 to High(Rs) do
    for ki := 0 to High(Ks) do
    begin
      RunCell(Ks[ki], Rs[ri], Xs, Ys, Tr[ri][ki], Ho[ri][ki]);
      WriteLn(Format('  Range=%.1f  K=%2d  ->  train MSE = %.6f   held-out MSE = %.6f',
        [Rs[ri], Ks[ki], Tr[ri][ki], Ho[ri][ki]]));
    end;
  WriteLn;

  // ---- results table ------------------------------------------------------
  WriteLn('================================================================');
  WriteLn('TRAIN MSE  (rows = Range, cols = K):');
  Write('   Range \ K |');
  for ki := 0 to High(Ks) do Write(Format('%10d', [Ks[ki]]));
  WriteLn;
  for ri := 0 to High(Rs) do
  begin
    Write(Format('   %7.1f   |', [Rs[ri]]));
    for ki := 0 to High(Ks) do Write(Format('%10.6f', [Tr[ri][ki]]));
    WriteLn;
  end;
  WriteLn;
  WriteLn('HELD-OUT MSE  (rows = Range, cols = K):');
  Write('   Range \ K |');
  for ki := 0 to High(Ks) do Write(Format('%10d', [Ks[ki]]));
  WriteLn;
  for ri := 0 to High(Rs) do
  begin
    Write(Format('   %7.1f   |', [Rs[ri]]));
    for ki := 0 to High(Ks) do Write(Format('%10.6f', [Ho[ri][ki]]));
    WriteLn;
  end;
  WriteLn('================================================================');
  WriteLn;

  // ---- self-checking PASS/FAIL gate ---------------------------------------
  // (1) adding knots HELPS at first: best held-out beats smallest-K held-out.
  HelpOK := True;
  for ri := 0 to High(Rs) do
  begin
    BestHo := Ho[ri][0];
    for ki := 1 to High(Ks) do
      if Ho[ri][ki] < BestHo then BestHo := Ho[ri][ki];
    if not (BestHo < Ho[ri][0] - cHelpSlack) then
    begin
      WriteLn(Format('GATE(1) FAIL: at Range=%.1f best held-out %.6f does not ',
        [Rs[ri], BestHo]),
        Format('improve on smallest-K=%d held-out %.6f by %.3f.',
        [Ks[0], Ho[ri][0], cHelpSlack]));
      HelpOK := False;
    end;
  end;

  // (2) smallest- and largest-K both fit train below threshold (each Range).
  FitOK := True;
  for ri := 0 to High(Rs) do
  begin
    if Tr[ri][0] >= cFitThreshold then
    begin
      WriteLn(Format('GATE(2) FAIL: Range=%.1f K=%d train MSE %.6f >= %.3f.',
        [Rs[ri], Ks[0], Tr[ri][0], cFitThreshold]));
      FitOK := False;
    end;
    if Tr[ri][High(Ks)] >= cFitThreshold then
    begin
      WriteLn(Format('GATE(2) FAIL: Range=%.1f K=%d train MSE %.6f >= %.3f.',
        [Rs[ri], Ks[High(Ks)], Tr[ri][High(Ks)], cFitThreshold]));
      FitOK := False;
    end;
  end;

  // (3) for at least one Range, the LARGEST-K held-out error stops helping:
  //     it is NOT better than the best held-out over the sweep (minus slack).
  StallOK := False;
  for ri := 0 to High(Rs) do
  begin
    BestHo := Ho[ri][0];
    for ki := 1 to High(Ks) do
      if Ho[ri][ki] < BestHo then BestHo := Ho[ri][ki];
    LastHo := Ho[ri][High(Ks)];
    if LastHo > BestHo + cStallSlack then
    begin
      WriteLn(Format('GATE(3): at Range=%.1f the largest K=%d held-out MSE %.6f ',
        [Rs[ri], Ks[High(Ks)], LastHo]),
        Format('is WORSE than the sweep best %.6f -> extra knots stopped helping.',
        [BestHo]));
      StallOK := True;
    end;
  end;
  if not StallOK then
    WriteLn('GATE(3): no Range showed the largest K hurting held-out error ',
      'beyond slack (extra knots still helped everywhere).');

  WriteLn;
  WriteLn(Format('GATE SUMMARY: knots-help=%s  endpoints-fit=%s  knots-stall=%s',
    [BoolToStr(HelpOK, True), BoolToStr(FitOK, True), BoolToStr(StallOK, True)]));

  if HelpOK and FitOK and StallOK then
    WriteLn('PASS: more knots first LOWER held-out error (a low-K spline ',
      'under-fits), both endpoints train, and then extra knots STOP helping ',
      '(or hurt) held-out error -- the capacity / overfitting trade is visible.')
  else
  begin
    WriteLn('FAIL: the capacity/overfitting story did not hold on this run.');
    Halt(1);
  end;
end.
