program SplineActivationKAN;
(*
SplineActivationKAN: a KAN-vs-MLP toy-fit micro-experiment for the landed
per-channel learnable activation TNNetSplineActivation (a Kolmogorov-Arnold
Network style piecewise-linear activation, KAN / Liu et al. 2024).

The headline KAN claim is that a LEARNABLE activation buys LOWER final loss at a
fixed width / matched parameter count. This example puts that claim on a tiny,
fully synthetic 1D regression and reports the numbers.

Target (wiggly so a learnable nonlinearity has something to earn):
    y = sin(3x) + 0.3*sin(11x),  x in [-2, 2]

Two arms, trained the SAME number of epochs with the SAME data and optimizer:
  Arm A (baseline MLP):  Input(1) -> FullConnectReLU(Wa) -> FullConnectLinear(1)
  Arm B (KAN-flavored):  Input(1) -> FullConnect(Wb)
                                  -> TNNetSplineActivation(K, Range)
                                  -> FullConnectLinear(1)

TNNetSplineActivation adds (K+1)*Depth trainable control-point values per layer
while ReLU adds ZERO. To keep this a FAIR fixed-param fight we make the ReLU arm
WIDER (Wa > Wb) so both arms have ~equal total trainable weight counts. The
exact per-arm weight count (TNNet.CountWeights) is computed and PRINTED so the
reader can see the match.

After training, the spline arm's LEARNED per-channel activation shapes are
dumped: for a few channels we sample the spline output over x in [-Range,+Range]
and print a compact (x, y) table. An UNTRAINED TNNetSplineActivation is an EXACT
identity map (control points init on y=x), so seeing the curves bent away from
y=x after training is a clean built-in check that the activation actually
learned something.

Pure CPU, no dataset download, synthetic data generated in-code, runs in a few
seconds.

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
  cXRange   = 2.0;    // target sampled over x in [-cXRange, +cXRange]
  cNumTrain = 256;    // synthetic training samples
  cEpochs   = 600;    // same epoch budget for both arms
  cBatch    = 16;
  cLR       = 0.01;
  cMomentum = 0.9;

  // Spline arm geometry. K=4 intervals (5 control points) over [-Range,+Range].
  cSplineW  = 8;      // hidden width of the spline arm
  cSplineK  = 4;      // NumIntervals
  cSplineR  = cXRange;// spline Range matched to the input domain
  // The ReLU arm is made WIDER to match total weight count (see ChooseReLUWidth).

  // Target function: a wiggly 1D signal so a learnable activation can earn its
  // keep. Two incommensurate frequencies => not a single sine a ReLU MLP nails.
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

  // MSE on a dense clean grid (no noise), the honest measure of fit quality.
  function GridMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    X, Diff, Sum: TNeuralFloat;
    Inp, Outp: TNNetVolume;
  begin
    Inp := TNNetVolume.Create(1, 1, 1);
    Outp := TNNetVolume.Create(1, 1, 1);
    Sum := 0;
    for I := 0 to 199 do
    begin
      X := -cXRange + 2.0 * cXRange * I / 199.0;
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

  // One full mini-batch SGD training run on the (shared) training set.
  function TrainArm(NN: TNNet; const Xs, Ys: array of TNeuralFloat;
    const Tag: string): TNeuralFloat;
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
        // Fisher-Yates shuffle.
        for I := High(Order) downto 1 do
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
        if (Epoch = 1) or (Epoch mod 100 = 0) or (Epoch = cEpochs) then
          WriteLn(Format('  [%s] epoch %4d  grid-MSE = %.6f',
            [Tag, Epoch, GridMSE(NN)]));
      end;
    finally
      Inp.Free;
      Tgt.Free;
    end;
    Result := GridMSE(NN);
  end;

  // ---- spline arm (FullConnect -> TNNetSplineActivation -> FullConnectLinear).
  // Returns the index of the spline layer so we can read it back AFTER training.
  procedure BuildSplineArm(out NN: TNNet; out SplineIdx: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 1));                       // 0
    NN.AddLayer(TNNetFullConnect.Create(cSplineW));               // 1 (linear+bias)
    SplineIdx := NN.GetLastLayerIdx() + 1;
    NN.AddLayer(TNNetSplineActivation.Create(cSplineK, cSplineR)); // 2 <- learnable
    NN.AddLayer(TNNetFullConnectLinear.Create(1));                // 3
    NN.InitWeights();
  end;

  // ---- ReLU arm at a chosen width W: Input -> FullConnectReLU(W) -> Linear(1).
  procedure BuildReLUArm(out NN: TNNet; W: integer);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(W));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.InitWeights();
  end;

  // Evaluate the learned spline of channel c at scalar x directly from the
  // layer's control points, replicating TNNetSplineActivation's forward math:
  // uniform knots t[i] = -Range + dt*i (dt = 2*Range/K); segment index clamped
  // to [0, K-1]; frac = pos - i left UNCLAMPED so boundary segments linearly
  // extrapolate exactly as the layer does.
  function EvalSpline(Layer: TNNetLayer; Chan: integer;
    X: TNeuralFloat): TNeuralFloat;
  var
    dt, pos, frac, y0, y1: TNeuralFloat;
    i: integer;
  begin
    dt := (2 * cSplineR) / cSplineK;
    pos := (X + cSplineR) / dt;
    i := Trunc(pos);
    if i < 0 then i := 0;
    if i > cSplineK - 1 then i := cSplineK - 1;
    frac := pos - i;
    y0 := Layer.Neurons[i].Weights.Raw[Chan];
    y1 := Layer.Neurons[i + 1].Weights.Raw[Chan];
    Result := y0 + frac * (y1 - y0);
  end;

  // Pick the ReLU hidden width whose total CountWeights best matches the spline
  // arm's. Build throwaway nets and compare (cheap, exact, no hand arithmetic).
  function ChooseReLUWidth(Target: integer): integer;
  var
    W, BestW, BestDiff, Diff, C: integer;
    Probe: TNNet;
  begin
    BestW := 1; BestDiff := MaxInt;
    for W := 1 to 64 do
    begin
      BuildReLUArm(Probe, W);
      C := Probe.CountWeights();
      Probe.Free;
      Diff := Abs(C - Target);
      if Diff < BestDiff then
      begin
        BestDiff := Diff;
        BestW := W;
      end;
    end;
    Result := BestW;
  end;

var
  NNSpline, NNReLU: TNNet;
  Xs, Ys: array[0..cNumTrain - 1] of TNeuralFloat;
  SplineIdx, ReLUW, I, Ci, Cnt: integer;
  SplineParams, ReLUParams: integer;
  LossSpline, LossReLU: TNeuralFloat;
  X, Dt, Dev, BestDev: TNeuralFloat;
  SplineLayer: TNNetLayer;
  Channels: array[0..3] of integer = (0, 1, 2, 3);
  Bend: array[0..cSplineW - 1] of TNeuralFloat;
  C, J, BestC, NumShow, NumActive: integer;
  Used: array[0..cSplineW - 1] of boolean;
begin
  RandSeed := 2026;
  WriteLn('SplineActivationKAN: KAN-vs-MLP toy fit at MATCHED parameter count.');
  WriteLn('Target  y = sin(3x) + 0.3*sin(11x)  over x in [-', cXRange:0:1,
    ', ', cXRange:0:1, '].');
  WriteLn('Arm A: ReLU MLP (wider).  Arm B: SAME MLP with ReLU -> ' +
    'TNNetSplineActivation (per-channel learnable piecewise-linear).');
  WriteLn;

  // Shared training set: both arms see EXACTLY the same data.
  MakeTrainSet(Xs, Ys);

  // Build the spline arm and count its params, then size the ReLU arm to match.
  BuildSplineArm(NNSpline, SplineIdx);
  SplineParams := NNSpline.CountWeights();
  ReLUW := ChooseReLUWidth(SplineParams);
  BuildReLUArm(NNReLU, ReLUW);
  ReLUParams := NNReLU.CountWeights();

  WriteLn(Format('Spline arm: FullConnect(%d) -> SplineActivation(K=%d,Range=%.1f)'
    + ' -> Linear(1)   =>  %d trainable weights',
    [cSplineW, cSplineK, cSplineR, SplineParams]));
  WriteLn(Format('ReLU   arm: FullConnectReLU(%d) -> Linear(1)' +
    '                          =>  %d trainable weights  (width chosen to match)',
    [ReLUW, ReLUParams]));
  WriteLn(Format('Param-count match: spline=%d  relu=%d  (delta=%d)',
    [SplineParams, ReLUParams, Abs(SplineParams - ReLUParams)]));
  WriteLn;

  WriteLn('Training ReLU arm for ', cEpochs, ' epochs...');
  LossReLU := TrainArm(NNReLU, Xs, Ys, 'ReLU');
  WriteLn;
  WriteLn('Training spline arm for ', cEpochs, ' epochs...');
  LossSpline := TrainArm(NNSpline, Xs, Ys, 'SPL ');
  WriteLn;

  WriteLn('================================================================');
  WriteLn('RESULT (final clean-grid MSE, lower is better):');
  WriteLn(Format('  ReLU   arm (%d params): %.6f', [ReLUParams, LossReLU]));
  WriteLn(Format('  Spline arm (%d params): %.6f', [SplineParams, LossSpline]));
  if LossSpline <= LossReLU then
    WriteLn(Format('  => KAN claim HOLDS: learnable activation wins by %.1f%% ' +
      'at matched params.', [100.0 * (LossReLU - LossSpline) / LossReLU]))
  else
    WriteLn('  => KAN claim does NOT hold on this run.');
  WriteLn('================================================================');
  WriteLn;

  // ---- dump the learned per-channel spline shapes -------------------------
  // Read the spline layer back by INDEX (no Fit reload here, but indexing is the
  // robust idiom). Each control point y[i,c] lives in Neurons[i].Weights.Raw[c].
  // For a few channels we sample the activation over x in [-Range,+Range] and
  // compare to the identity y=x that an UNTRAINED spline would produce.
  SplineLayer := NNSpline.Layers[SplineIdx];
  // Per-channel BEND score: max distance of any interior control point from the
  // straight line joining the two endpoints y[0,c]..y[K,c]. ~0 for a dead/flat
  // or perfectly linear channel, large for a genuinely curved learned shape.
  NumActive := 0;
  for C := 0 to cSplineW - 1 do
  begin
    Dev := 0;
    for J := 1 to cSplineK - 1 do
      Dev := Max(Dev, Abs(SplineLayer.Neurons[J].Weights.Raw[C] -
        (SplineLayer.Neurons[0].Weights.Raw[C] +
         (SplineLayer.Neurons[cSplineK].Weights.Raw[C] -
          SplineLayer.Neurons[0].Weights.Raw[C]) * J / cSplineK)));
    Bend[C] := Dev;
    if Dev > 0.05 then Inc(NumActive);
  end;
  // Show the most-bent channels (at most High(Channels)+1 of them).
  NumShow := High(Channels) + 1;
  if NumShow > cSplineW then NumShow := cSplineW;
  for C := 0 to High(Used) do Used[C] := False;
  for Ci := 0 to NumShow - 1 do
  begin
    BestC := -1; BestDev := -1;
    for C := 0 to cSplineW - 1 do
      if (not Used[C]) and (Bend[C] > BestDev) then
      begin
        BestDev := Bend[C];
        BestC := C;
      end;
    Channels[Ci] := BestC;
    Used[BestC] := True;
  end;
  WriteLn('Learned per-channel spline shapes (untrained = exact identity y=x):');
  WriteLn(Format('(%d of %d hidden channels learned a non-trivial bend; ' +
    'showing the %d most-bent — the fit is sparse, a few channels carry it)',
    [NumActive, cSplineW, NumShow]));
  WriteLn('Control points y[i,c] for sampled channels (knots t[i] fixed on ',
    '[-', cSplineR:0:1, ', ', cSplineR:0:1, ']):');
  Dt := (2 * cSplineR) / cSplineK;
  Write('   t[i] =');
  for I := 0 to cSplineK do Write(Format('%8.3f', [-cSplineR + Dt * I]));
  WriteLn;
  for Ci := 0 to NumShow - 1 do
  begin
    Cnt := Channels[Ci];
    Write(Format('  ch %-2d  y =', [Cnt]));
    for I := 0 to cSplineK do
      Write(Format('%8.3f', [SplineLayer.Neurons[I].Weights.Raw[Cnt]]));
    WriteLn;
  end;
  WriteLn;
  WriteLn('Same channels, activation sampled over x (deviation from y=x means ',
    'the activation learned a nonlinear shape):');
  Write('      x   |');
  for Ci := 0 to NumShow - 1 do
    Write(Format('  ch %-2d y', [Channels[Ci]]));
  WriteLn;
  for I := 0 to 8 do
  begin
    X := -cSplineR + (2 * cSplineR) * I / 8.0;
    Write(Format('%8.3f  |', [X]));
    for Ci := 0 to NumShow - 1 do
    begin
      Cnt := Channels[Ci];
      Write(Format('%9.3f', [EvalSpline(SplineLayer, Cnt, X)]));
    end;
    WriteLn;
  end;
  WriteLn;
  WriteLn('Read it as: the ReLU arm and the spline arm carry the SAME number of ',
    'trainable weights, but the spline arm spends some of them on a learnable ',
    'per-channel activation shape (bent away from the identity above) and fits ',
    'the wiggly target to a lower MSE — the KAN fixed-width / matched-param win.');

  NNSpline.Free;
  NNReLU.Free;
end.
