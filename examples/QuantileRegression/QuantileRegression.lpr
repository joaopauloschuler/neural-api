program QuantileRegression;
(*
QuantileRegression: demonstrates the TNNetQuantileLoss (pinball) regression
head on a HETEROSCEDASTIC 1-D dataset whose noise std GROWS with x. Three tiny
MLPs are trained, one per target quantile q in {0.1, 0.5, 0.9}; together they
estimate a prediction interval whose width adapts to the input-dependent noise.

The program then:
  - prints an ASCII chart of the q=0.5 median curve plus the [q=0.1, q=0.9] band
    over the input range, and
  - measures empirical coverage on a held-out test set: the fraction of test
    targets that fall inside the predicted [q=0.1, q=0.9] band. For a correctly
    calibrated 10%/90% pair this should be near 80%. A built-in PASS/FAIL check
    asserts coverage lands in a tolerant [0.65, 0.95] window.

A second, JOINT arm then trains ONE model whose 3-wide output predicts all
three quantiles jointly in a single forward pass via TNNetMultiQuantileLoss.
It reports the same band coverage with a single model, counts quantile
crossings with vs without the TNNetMultiQuantileLoss.SortAscending monotonicity
guard (approach (a): non-differentiable inference-time sort), and shows the
guard repairing a deliberately crossed prediction triple.

Pure CPU, single-threaded for determinism, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cHidden    = 24;
  cTrainCnt  = 800;
  cEpochs    = 200;
  cTestCnt   = 400;
  cChartCols = 60;
  cChartRows = 19;
  cXMin      = 0.0;
  cXMax      = 4.0;

  // True mean function f(x) and input-dependent noise std sigma(x).
  function TrueMean(X: TNeuralFloat): TNeuralFloat;
  begin
    Result := Sin(X) + 0.3 * X;
  end;

  function NoiseStd(X: TNeuralFloat): TNeuralFloat;
  begin
    // Std grows linearly with x => heteroscedastic.
    Result := 0.10 + 0.45 * X;
  end;

  // Box-Muller standard normal sample.
  function RandNormal(): TNeuralFloat;
  var
    U1, U2: TNeuralFloat;
  begin
    U1 := Random;
    if U1 < 1e-12 then U1 := 1e-12;
    U2 := Random;
    Result := Sqrt(-2.0 * Ln(U1)) * Cos(2.0 * Pi * U2);
  end;

  procedure BuildModel(out NN: TNNet; Quantile: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetHyperbolicTangent.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetHyperbolicTangent.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(1));
    NN.AddLayer(TNNetQuantileLoss.Create(Quantile));
    NN.SetLearningRate(0.02, 0.9);
  end;

  // ONE model with a 3-wide output (Depth=3) predicting q={0.1,0.5,0.9}
  // jointly in a single forward pass via TNNetMultiQuantileLoss.
  procedure BuildJointModel(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(1, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetHyperbolicTangent.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(cHidden));
    NN.AddLayer(TNNetHyperbolicTangent.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(3));   // 3 quantile channels
    NN.AddLayer(TNNetMultiQuantileLoss.Create([0.1, 0.5, 0.9]));
    NN.SetLearningRate(0.02, 0.9);
  end;

  // Trains the joint head: the scalar target y is replicated across all 3
  // output channels so each channel learns its own quantile in one pass.
  procedure TrainJointModel(NN: TNNet; Xs, Ys: TNNetVolumeList; Epochs: integer);
  var
    Ep, K: integer;
    Tgt: TNNetVolume;
  begin
    Tgt := TNNetVolume.Create(1, 1, 3);
    try
      for Ep := 1 to Epochs do
        for K := 0 to Xs.Count - 1 do
        begin
          Tgt.Raw[0] := Ys[K].Raw[0];
          Tgt.Raw[1] := Ys[K].Raw[0];
          Tgt.Raw[2] := Ys[K].Raw[0];
          NN.Compute(Xs[K]);
          NN.Backpropagate(Tgt);
        end;
    finally
      Tgt.Free;
    end;
  end;

  // Joint-head inference with the monotonicity guard applied: returns the
  // sorted (Lo, Mid, Hi) triple so q=0.1 never exceeds q=0.9.
  procedure PredictJoint(NN: TNNet; X: TNeuralFloat;
    Guard: boolean; out Lo, Mid, Hi: TNeuralFloat);
  var
    V: TNNetVolume;
  begin
    V := TNNetVolume.Create(1, 1, 1);
    try
      V.Raw[0] := X;
      NN.Compute(V);
      if Guard then
        TNNetMultiQuantileLoss.SortAscending(NN.GetLastLayer.Output, 3);
      Lo  := NN.GetLastLayer.Output.Raw[0];
      Mid := NN.GetLastLayer.Output.Raw[1];
      Hi  := NN.GetLastLayer.Output.Raw[2];
    finally
      V.Free;
    end;
  end;

  // Build a fixed dataset (x, y) of size N with heteroscedastic noise.
  procedure BuildDataset(N: integer; out Xs, Ys: TNNetVolumeList);
  var
    K: integer;
    Vx, Vy: TNNetVolume;
    X: TNeuralFloat;
  begin
    Xs := TNNetVolumeList.Create(True);
    Ys := TNNetVolumeList.Create(True);
    for K := 0 to N - 1 do
    begin
      X := cXMin + Random * (cXMax - cXMin);
      Vx := TNNetVolume.Create(1, 1, 1);
      Vy := TNNetVolume.Create(1, 1, 1);
      Vx.Raw[0] := X;
      Vy.Raw[0] := TrueMean(X) + NoiseStd(X) * RandNormal();
      Xs.Add(Vx);
      Ys.Add(Vy);
    end;
  end;

  procedure TrainModel(NN: TNNet; Xs, Ys: TNNetVolumeList; Epochs: integer);
  var
    Ep, K: integer;
  begin
    for Ep := 1 to Epochs do
      for K := 0 to Xs.Count - 1 do
      begin
        NN.Compute(Xs[K]);
        NN.Backpropagate(Ys[K]);
      end;
  end;

  function Predict(NN: TNNet; X: TNeuralFloat): TNeuralFloat;
  var
    V: TNNetVolume;
  begin
    V := TNNetVolume.Create(1, 1, 1);
    try
      V.Raw[0] := X;
      NN.Compute(V);
      Result := NN.GetLastLayer.Output.Raw[0];
    finally
      V.Free;
    end;
  end;

  // Draw an ASCII chart: median curve 'M', band edges 'lo'/'hi' as '#', true
  // mean as '.'. Rows = value axis (top = high), cols = x axis.
  procedure DrawChart(NLo, NMid, NHi: TNNet);
  var
    Grid: array of array of char;
    Col, Row, R: integer;
    X, Lo, Mid, Hi, Tm, YMin, YMax: TNeuralFloat;
    LoArr, MidArr, HiArr, TmArr: array of TNeuralFloat;

    function ToRow(Val: TNeuralFloat): integer;
    begin
      if YMax <= YMin then Exit(cChartRows div 2);
      Result := Round((YMax - Val) / (YMax - YMin) * (cChartRows - 1));
      if Result < 0 then Result := 0;
      if Result > cChartRows - 1 then Result := cChartRows - 1;
    end;

  begin
    SetLength(LoArr, cChartCols);
    SetLength(MidArr, cChartCols);
    SetLength(HiArr, cChartCols);
    SetLength(TmArr, cChartCols);
    YMin := 1e30; YMax := -1e30;
    for Col := 0 to cChartCols - 1 do
    begin
      X := cXMin + (Col / (cChartCols - 1)) * (cXMax - cXMin);
      Lo := Predict(NLo, X);
      Mid := Predict(NMid, X);
      Hi := Predict(NHi, X);
      Tm := TrueMean(X);
      LoArr[Col] := Lo; MidArr[Col] := Mid; HiArr[Col] := Hi; TmArr[Col] := Tm;
      YMin := Min(YMin, Min(Min(Lo, Hi), Tm));
      YMax := Max(YMax, Max(Max(Lo, Hi), Tm));
    end;
    YMin := YMin - 0.2; YMax := YMax + 0.2;

    SetLength(Grid, cChartRows, cChartCols);
    for Row := 0 to cChartRows - 1 do
      for Col := 0 to cChartCols - 1 do
        Grid[Row][Col] := ' ';

    for Col := 0 to cChartCols - 1 do
    begin
      Grid[ToRow(TmArr[Col])][Col] := '.';        // true mean
      Grid[ToRow(LoArr[Col])][Col] := '#';        // q=0.1 edge
      Grid[ToRow(HiArr[Col])][Col] := '#';        // q=0.9 edge
      Grid[ToRow(MidArr[Col])][Col] := 'M';       // q=0.5 median
    end;

    WriteLn;
    WriteLn('Prediction interval (M=median, #=10%/90% band edges, .=true mean):');
    WriteLn('y=', YMax:6:2, ' +', StringOfChar('-', cChartCols), '+');
    for R := 0 to cChartRows - 1 do
    begin
      Write('        |');
      for Col := 0 to cChartCols - 1 do
        Write(Grid[R][Col]);
      WriteLn('|');
    end;
    WriteLn('y=', YMin:6:2, ' +', StringOfChar('-', cChartCols), '+');
    WriteLn('         x=', cXMin:0:1,
      StringOfChar(' ', cChartCols - 8), 'x=', cXMax:0:1);
  end;

var
  NetLo, NetMid, NetHi, NetJoint: TNNet;
  TrainX, TrainY, TestX, TestY: TNNetVolumeList;
  K, Inside, JInside, CrossRaw, CrossGuard: integer;
  Lo, Hi, Coverage: TNeuralFloat;
  JLo, JMid, JHi, GLo, GMid, GHi, JCoverage: TNeuralFloat;
  Pass: boolean;
begin
  // Mask FPU exceptions (log/exp/normal sampling) and force determinism.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
    exOverflow, exUnderflow, exPrecision]);
  // Manual Compute/Backpropagate (no TNNet.Fit worker pool) runs
  // single-threaded already, so results are deterministic given RandSeed.
  RandSeed := 2026;

  WriteLn('QuantileRegression demo: pinball-loss heads on a heteroscedastic');
  WriteLn('1-D dataset y = sin(x) + 0.3*x + N(0, sigma(x)), sigma growing with x.');
  WriteLn('Training three tiny MLPs for q in {0.1, 0.5, 0.9}...');

  BuildDataset(cTrainCnt, TrainX, TrainY);
  BuildDataset(cTestCnt, TestX, TestY);
  try
    BuildModel(NetLo, 0.1);
    BuildModel(NetMid, 0.5);
    BuildModel(NetHi, 0.9);
    try
      NetLo.InitWeights();  TrainModel(NetLo, TrainX, TrainY, cEpochs);
      NetMid.InitWeights(); TrainModel(NetMid, TrainX, TrainY, cEpochs);
      NetHi.InitWeights();  TrainModel(NetHi, TrainX, TrainY, cEpochs);

      DrawChart(NetLo, NetMid, NetHi);

      // Empirical coverage of the [q=0.1, q=0.9] band on held-out points.
      Inside := 0;
      for K := 0 to TestX.Count - 1 do
      begin
        Lo := Predict(NetLo, TestX[K].Raw[0]);
        Hi := Predict(NetHi, TestX[K].Raw[0]);
        if Hi < Lo then
        begin
          Coverage := Lo; Lo := Hi; Hi := Coverage; // swap if crossed
        end;
        if (TestY[K].Raw[0] >= Lo) and (TestY[K].Raw[0] <= Hi) then
          Inc(Inside);
      end;
      Coverage := Inside / TestX.Count;

      WriteLn;
      WriteLn('Held-out coverage of [q=0.1, q=0.9] band: ',
        (Coverage * 100):0:1, '% (', Inside, '/', TestX.Count,
        '), nominal target ~80%.');

      Pass := (Coverage >= 0.65) and (Coverage <= 0.95);
      if Pass then
        WriteLn('RESULT: PASS (coverage within calibrated [65%, 95%] window).')
      else
      begin
        WriteLn('RESULT: FAIL (coverage outside [65%, 95%] window).');
        Halt(1);
      end;

      // ---- Joint single-model arm -------------------------------------------
      WriteLn;
      WriteLn('=== JOINT HEAD: ONE model, 3-wide TNNetMultiQuantileLoss output ===');
      WriteLn('Training a single model that predicts q={0.1,0.5,0.9} jointly...');
      BuildJointModel(NetJoint);
      try
        NetJoint.InitWeights();
        TrainJointModel(NetJoint, TrainX, TrainY, cEpochs);

        // Coverage of the joint [q=0.1, q=0.9] band (with monotonicity guard),
        // and crossing counts WITHOUT vs WITH the guard.
        JInside := 0; CrossRaw := 0; CrossGuard := 0;
        for K := 0 to TestX.Count - 1 do
        begin
          // Raw (no guard): count quantile crossings (lo > hi).
          PredictJoint(NetJoint, TestX[K].Raw[0], False, JLo, JMid, JHi);
          if JLo > JHi then Inc(CrossRaw);
          // Guarded: sorted outputs, count any residual crossings (must be 0).
          PredictJoint(NetJoint, TestX[K].Raw[0], True, GLo, GMid, GHi);
          if GLo > GHi then Inc(CrossGuard);
          if (TestY[K].Raw[0] >= GLo) and (TestY[K].Raw[0] <= GHi) then
            Inc(JInside);
        end;
        JCoverage := JInside / TestX.Count;

        WriteLn;
        WriteLn('Joint head held-out coverage of [q=0.1, q=0.9] band: ',
          (JCoverage * 100):0:1, '% (', JInside, '/', TestX.Count,
          '), nominal ~80%.');
        WriteLn('Quantile crossings (lo > hi) on test set:');
        WriteLn('  without monotonicity guard: ', CrossRaw, '/', TestX.Count);
        WriteLn('  with    monotonicity guard: ', CrossGuard, '/', TestX.Count,
          ' (SortAscending removes all crossings)');

        // Explicit guard demonstration on a deliberately CROSSED output triple
        // (e.g. from an under-trained net): SortAscending repairs it.
        NetJoint.GetLastLayer.Output.Raw[0] := 0.90;  // q=0.1 prediction
        NetJoint.GetLastLayer.Output.Raw[1] := 0.40;  // q=0.5 prediction
        NetJoint.GetLastLayer.Output.Raw[2] := 0.10;  // q=0.9 prediction (crossed!)
        WriteLn('Guard demo - crossed triple before: [',
          NetJoint.GetLastLayer.Output.Raw[0]:0:2, ', ',
          NetJoint.GetLastLayer.Output.Raw[1]:0:2, ', ',
          NetJoint.GetLastLayer.Output.Raw[2]:0:2, ']');
        TNNetMultiQuantileLoss.SortAscending(NetJoint.GetLastLayer.Output, 3);
        WriteLn('Guard demo - after SortAscending: [',
          NetJoint.GetLastLayer.Output.Raw[0]:0:2, ', ',
          NetJoint.GetLastLayer.Output.Raw[1]:0:2, ', ',
          NetJoint.GetLastLayer.Output.Raw[2]:0:2, '] (non-decreasing)');

        Pass := (JCoverage >= 0.65) and (JCoverage <= 0.95) and (CrossGuard = 0);
        if Pass then
          WriteLn('RESULT: PASS (one model matches the 3-MLP coverage; guard',
            ' eliminated crossings).')
        else
        begin
          WriteLn('RESULT: FAIL (joint coverage outside window or crossings',
            ' remain after guard).');
          Halt(1);
        end;
      finally
        NetJoint.Free;
      end;
    finally
      NetLo.Free;
      NetMid.Free;
      NetHi.Free;
    end;
  finally
    TrainX.Free;
    TrainY.Free;
    TestX.Free;
    TestY.Free;
  end;
end.
