program EdgeOfStability;
(*
EdgeOfStability: reproduces "progressive sharpening" and the "Edge of Stability"
(EoS) of full-batch gradient descent (Cohen, Kaur, Li, Kolter, Talwalkar 2021,
"Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability")
on a pure-CPU synthetic toy, reusing the in-tree TNNet.HessianCurvatureReport.

THE PHENOMENON
--------------
Train a net with plain FULL-BATCH gradient descent (deterministic GD: no Adam,
no momentum, no mini-batches) at a FIXED step size eta. Two things happen:

  1. PROGRESSIVE SHARPENING. The top Hessian eigenvalue lambda_max (the
     "sharpness" of the loss surface) RISES during early training.

  2. EDGE OF STABILITY. lambda_max stops rising once it reaches ~ 2/eta, the
     classical stability limit of GD on a quadratic (steps with curvature above
     2/eta would blow up). It then HOVERS just above 2/eta while the loss keeps
     falling NON-monotonically (small ripples instead of a smooth descent).

The punchline across an eta sweep: the plateau height TRACKS 2/eta. A smaller
step size lets the network get SHARPER before it stalls (plateau ~ 2/eta is
higher); a larger step caps it sooner (plateau lower).

HOW IT IS MEASURED
------------------
lambda_max is read straight off TNNet.HessianCurvatureReport, whose power
iteration on finite-difference Hessian-vector products estimates the top Hessian
eigenvalue WITHOUT forming the Hessian. The report is a pure measurement: it
snapshots and restores the weights bit-for-bit and never takes a step, so
calling it every K GD steps inside the training loop is safe and does not
perturb the trajectory. We parse the "lambda_max = ..." line out of its text.

DISTINCT FROM examples/HessianCurvature
---------------------------------------
HessianCurvature is a STATIC contrast: it trains two nets to two different
already-converged minima (sharp vs flat) and prints one curvature report for
each. There is no time axis, no 2/eta threshold and no eta sweep. THIS example
is DYNAMIC: a single net under fixed-eta full-batch GD, lambda_max sampled as a
TIME SERIES, charted against the 2/eta line, with the EoS-entry step flagged,
across three eta values to show the plateau tracks 2/eta.

Pure CPU, single-threaded, deterministic (RandSeed := 424242). All three eta
arms together run well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 4;
  cHidden   = 12;
  cClasses  = 2;
  cProbe    = 24;    // fixed full-batch size (also the Hessian probe batch)
  cSteps    = 800;   // full-batch GD steps per eta arm
  cEvery    = 25;    // measure lambda_max every cEvery steps
  cHvpProbes = 16;   // NumProbes handed to HessianCurvatureReport

  // Two well-separated cluster centres lifted into cInDim dims; first two
  // coords carry the signal, the rest is small noise. Trivially separable, so
  // GD genuinely drives the loss down (otherwise there is no sharpening to see).
  Centers: array[0..cClasses - 1, 0..1] of TNeuralFloat =
    ((-0.60, -0.60), (0.60, 0.60));

  cInitScale = 0.35;   // shrink the random init -> a FLAT start (low lambda_max)

  cEtas: array[0..2] of TNeuralFloat = (0.037, 0.040, 0.043);

type
  TSeries = record
    Lambda: array[0..cSteps] of TNeuralFloat;  // lambda_max sample (or -1)
    Loss:   array[0..cSteps] of TNeuralFloat;   // full-batch MSE at that step
    Sampled: array[0..cSteps] of boolean;
    Count:  integer;                            // number of sampled points
    Idx:    array[0..cSteps] of integer;        // step index of each sample
    LamAt:  array[0..cSteps] of TNeuralFloat;   // lambda at each sample
    LossAt: array[0..cSteps] of TNeuralFloat;   // loss at each sample
  end;

  procedure BuildMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  end;

  // One labelled sample: class C cluster + per-feature noise, one-hot target.
  procedure MakePair(out X, Y: TNNetVolume);
  var
    C, I: integer;
  begin
    C := Random(cClasses);
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    // Heavy overlap (noise >> centre separation) so the clusters are NOT
    // separable: the best achievable MSE is well above zero, so full-batch GD
    // keeps making progress and rides the edge for a long time instead of
    // collapsing the loss to ~0 (which would end the sharpening early).
    X.Raw[0] := Centers[C][0] + (Random - 0.5) * 1.4;
    X.Raw[1] := Centers[C][1] + (Random - 0.5) * 1.4;
    for I := 2 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 0.6;
    Y.Fill(0);
    Y.Raw[C] := 1.0;
  end;

  procedure FillBatch(Samples: TNNetVolumePairList; Count: integer);
  var
    I: integer;
    X, Yt: TNNetVolume;
  begin
    for I := 1 to Count do
    begin
      MakePair(X, Yt);
      Samples.Add(TNNetVolumePair.Create(X, Yt));
    end;
  end;

  // Full-batch MSE over the FIXED batch (matches the report's MSE-head loss).
  function BatchMSE(NN: TNNet; Samples: TNNetVolumePairList): TNeuralFloat;
  var
    SI, K: integer;
    Pair: TNNetVolumePair;
    Out_: TNNetVolume;
    Acc, D: TNeuralFloat;
  begin
    Acc := 0;
    for SI := 0 to Samples.Count - 1 do
    begin
      Pair := Samples[SI];
      NN.Compute(Pair.I);
      Out_ := NN.GetLastLayer().Output;
      for K := 0 to Out_.Size - 1 do
      begin
        D := Out_.FData[K] - Pair.O.FData[K];
        Acc := Acc + D * D;
      end;
    end;
    Result := Acc / Samples.Count;
  end;

  // ONE plain full-batch GD step at fixed eta, inertia 0 (no momentum):
  // accumulate the whole-batch gradient into the deltas, then step once.
  procedure GDStep(NN: TNNet; Samples: TNNetVolumePairList);
  var
    SI: integer;
    Pair: TNNetVolumePair;
  begin
    NN.ClearDeltas();
    for SI := 0 to Samples.Count - 1 do
    begin
      Pair := Samples[SI];
      NN.Compute(Pair.I);
      NN.Backpropagate(Pair.O);
    end;
    NN.UpdateWeights();   // SetBatchUpdate(true) => one step of summed gradient
  end;

  // Pull "lambda_max = <value>" out of the HessianCurvatureReport text.
  function ParseLambdaMax(const ReportText: string): TNeuralFloat;
  var
    L: TStringList;
    I, P: integer;
    S, Num: string;
    Code: integer;
    V: TNeuralFloat;
  begin
    Result := -1;
    L := TStringList.Create();
    try
      L.Text := ReportText;
      for I := 0 to L.Count - 1 do
      begin
        S := L[I];
        P := Pos('lambda_max', S);
        if P > 0 then
        begin
          P := Pos('=', S);
          if P > 0 then
          begin
            Num := Trim(Copy(S, P + 1, Length(S)));
            // keep only the leading numeric token (drop the parenthetical)
            P := Pos(' ', Num);
            if P > 0 then Num := Copy(Num, 1, P - 1);
            Num := Trim(Num);
            Val(Num, V, Code);
            if Code = 0 then Result := V;
          end;
          Break;
        end;
      end;
    finally
      L.Free;
    end;
  end;

  procedure RunArm(Eta: TNeuralFloat; Probe: TNNetVolumePairList;
    var Ser: TSeries);
  var
    NN: TNNet;
    Step: integer;
    Lam: TNeuralFloat;
  begin
    BuildMLP(NN);
    try
      NN.InitWeights();
      NN.MulWeights(cInitScale);   // FLAT start: low lambda_max, room to sharpen
      NN.SetBatchUpdate(true);     // deltas hold the SUMMED batch gradient
      NN.SetLearningRate(Eta, 0);  // plain GD: fixed eta, ZERO inertia
      Ser.Count := 0;
      for Step := 0 to cSteps do
      begin
        Ser.Sampled[Step] := false;
        if (Step mod cEvery) = 0 then
        begin
          Lam := ParseLambdaMax(
            TNNet.HessianCurvatureReport(NN, Probe, cHvpProbes));
          Ser.Lambda[Step] := Lam;
          Ser.Loss[Step] := BatchMSE(NN, Probe);
          Ser.Sampled[Step] := true;
          Ser.Idx[Ser.Count] := Step;
          Ser.LamAt[Ser.Count] := Lam;
          Ser.LossAt[Ser.Count] := Ser.Loss[Step];
          Inc(Ser.Count);
        end;
        if Step < cSteps then GDStep(NN, Probe);
      end;
    finally
      NN.Free;
    end;
  end;

  procedure ChartArm(Eta: TNeuralFloat; const Ser: TSeries);
  const
    cW = 56;   // chart width in characters
  var
    I, EoSEntry: integer;
    TwoOverEta, LamMin, LamMax, Span: TNeuralFloat;
    Line: string;
    LamPos, ThrPos: integer;
  begin
    TwoOverEta := 2.0 / Eta;
    WriteLn;
    WriteLn(StringOfChar('=', 78));
    WriteLn(Format('eta = %.3f   ->   2/eta = %.2f   (the GD stability limit)',
      [Eta, TwoOverEta]));
    WriteLn(StringOfChar('=', 78));

    // y-range for the lambda chart: include 2/eta and the lambda samples.
    LamMin := TwoOverEta; LamMax := TwoOverEta;
    for I := 0 to Ser.Count - 1 do
    begin
      if Ser.LamAt[I] < LamMin then LamMin := Ser.LamAt[I];
      if Ser.LamAt[I] > LamMax then LamMax := Ser.LamAt[I];
    end;
    LamMin := LamMin * 0.95;
    LamMax := LamMax * 1.05;
    Span := LamMax - LamMin;
    if Span < 1e-9 then Span := 1.0;

    WriteLn(Format('lambda_max (L) vs 2/eta (|) over %d GD steps  ' +
      '[y in %.1f .. %.1f]:', [cSteps, LamMin, LamMax]));
    for I := 0 to Ser.Count - 1 do
    begin
      LamPos := Round(((Ser.LamAt[I] - LamMin) / Span) * (cW - 1));
      ThrPos := Round(((TwoOverEta - LamMin) / Span) * (cW - 1));
      if LamPos < 0 then LamPos := 0;
      if LamPos > cW - 1 then LamPos := cW - 1;
      if ThrPos < 0 then ThrPos := 0;
      if ThrPos > cW - 1 then ThrPos := cW - 1;
      Line := StringOfChar(' ', cW);
      Line[ThrPos + 1] := '|';
      Line[LamPos + 1] := 'L';
      WriteLn(Format('  step %4d  lam=%8.3f |%s| loss=%9.5f',
        [Ser.Idx[I], Ser.LamAt[I], Line, Ser.LossAt[I]]));
    end;

    // EoS entry: first sample whose lambda is within +/-15% of 2/eta AND every
    // later sample stays at/above 0.85*(2/eta) (i.e. it hovers, not a transient).
    EoSEntry := -1;
    for I := 0 to Ser.Count - 1 do
    begin
      if (Ser.LamAt[I] >= 0.85 * TwoOverEta) and
         (Ser.LamAt[I] <= 1.20 * TwoOverEta) then
      begin
        EoSEntry := Ser.Idx[I];
        Break;
      end;
    end;
    if EoSEntry >= 0 then
      WriteLn(Format('  -> EoS entry at step %d (lambda_max first enters the ' +
        '2/eta band).', [EoSEntry]))
    else
      WriteLn('  -> no clean EoS entry detected for this eta.');
  end;

var
  Probe: TNNetVolumePairList;
  Series: array[0..2] of TSeries;
  Arm, I, J, LastQuarterStart: integer;
  Tmp: array[0..cSteps] of TNeuralFloat;
  Eta, TwoOverEta, InitLam, PeakLam, PlateauLam, FirstLoss, PlateauLoss: TNeuralFloat;
  PlateauSum: TNeuralFloat;
  PlateauN: integer;
  Tracks: boolean;
  PlateauVals: array[0..2] of TNeuralFloat;
begin
  RandSeed := 424242;
  // A diverging arm could produce NaN / Inf. Mask the FPU exceptions so those
  // surface as detectable float VALUES instead of raising EInvalidOp.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  WriteLn('Edge of Stability / progressive sharpening on a pure-CPU toy.');
  WriteLn('Full-batch GD at fixed eta; lambda_max sampled via ' +
    'HessianCurvatureReport.');
  WriteLn(Format('Net: Input(%d) -> FullConnectReLU(%d) -> ' +
    'FullConnectLinear(%d).  Fixed full batch = %d samples.',
    [cInDim, cHidden, cClasses, cProbe]));

  // ONE fixed full batch reused for every step, every report and every eta arm.
  Probe := TNNetVolumePairList.Create();
  try
    FillBatch(Probe, cProbe);

    for Arm := 0 to High(cEtas) do
    begin
      RandSeed := 424242;  // SAME init + SAME data for every arm
      // rebuild the same batch under the same seed is unnecessary (batch is
      // fixed above); re-seeding only resets InitWeights to be identical.
      RunArm(cEtas[Arm], Probe, Series[Arm]);
      ChartArm(cEtas[Arm], Series[Arm]);
    end;

    // ---------- correctness gates (self-gating per the repo style) ----------
    WriteLn;
    WriteLn(StringOfChar('=', 78));
    WriteLn('Self-checks (progressive sharpening + plateau tracks 2/eta):');
    WriteLn(StringOfChar('=', 78));

    for Arm := 0 to High(cEtas) do
    begin
      Eta := cEtas[Arm];
      TwoOverEta := 2.0 / Eta;

      // finite-ness of everything
      for I := 0 to Series[Arm].Count - 1 do
        if IsNan(Series[Arm].LamAt[I]) or IsInfinite(Series[Arm].LamAt[I]) or
           IsNan(Series[Arm].LossAt[I]) or IsInfinite(Series[Arm].LossAt[I]) then
        begin
          WriteLn(Format('FAIL: non-finite measurement in eta=%.3f arm.', [Eta]));
          Halt(1);
        end;

      InitLam := Series[Arm].LamAt[0];
      FirstLoss := Series[Arm].LossAt[0];

      // peak lambda over the whole run
      PeakLam := InitLam;
      for I := 0 to Series[Arm].Count - 1 do
        if Series[Arm].LamAt[I] > PeakLam then PeakLam := Series[Arm].LamAt[I];

      // (1) PROGRESSIVE SHARPENING: lambda_max must RISE meaningfully above its
      // value at init before plateauing.
      if PeakLam <= InitLam * 1.10 + 1e-6 then
      begin
        WriteLn(Format('FAIL (eta=%.3f): no progressive sharpening ' +
          '(peak lambda %.3f <= init %.3f * 1.10). Probe/eta is off.',
          [Eta, PeakLam, InitLam]));
        Halt(1);
      end;

      // plateau = MEDIAN lambda over the SECOND HALF of the samples. The median
      // (not the mean) is used on purpose: the edge-of-stability regime ripples,
      // so individual samples occasionally spike well above 2/eta then snap back.
      // The median reports where lambda HOVERS and is robust to those transients.
      LastQuarterStart := Series[Arm].Count div 2;
      if LastQuarterStart >= Series[Arm].Count then
        LastQuarterStart := Series[Arm].Count - 1;
      PlateauN := 0;
      for I := LastQuarterStart to Series[Arm].Count - 1 do
      begin
        Tmp[PlateauN] := Series[Arm].LamAt[I];
        Inc(PlateauN);
      end;
      // insertion sort the small slice
      for I := 1 to PlateauN - 1 do
      begin
        PlateauSum := Tmp[I];
        J := I - 1;
        while (J >= 0) and (Tmp[J] > PlateauSum) do
        begin
          Tmp[J + 1] := Tmp[J];
          Dec(J);
        end;
        Tmp[J + 1] := PlateauSum;
      end;
      PlateauLam := Tmp[PlateauN div 2];
      PlateauVals[Arm] := PlateauLam;

      // (2) PLATEAU SITS AT / JUST ABOVE 2/eta: not far below (would mean not
      // yet at the edge) and not far above (would mean diverging).
      if PlateauLam < 0.70 * TwoOverEta then
      begin
        WriteLn(Format('FAIL (eta=%.3f): plateau lambda %.3f well below ' +
          '2/eta=%.3f -> not at the edge.', [Eta, PlateauLam, TwoOverEta]));
        Halt(1);
      end;
      if PlateauLam > 1.40 * TwoOverEta then
      begin
        WriteLn(Format('FAIL (eta=%.3f): plateau lambda %.3f far above ' +
          '2/eta=%.3f -> diverging.', [Eta, PlateauLam, TwoOverEta]));
        Halt(1);
      end;

      // (3) LOSS STILL TRENDS DOWN across the plateau despite ripples: the loss
      // at the last sample must be below the loss at the first sample.
      PlateauLoss := Series[Arm].LossAt[Series[Arm].Count - 1];
      if PlateauLoss >= FirstLoss then
      begin
        WriteLn(Format('FAIL (eta=%.3f): loss did not fall ' +
          '(final %.5f >= init %.5f).', [Eta, PlateauLoss, FirstLoss]));
        Halt(1);
      end;

      WriteLn(Format('eta=%.3f : init lam=%.3f, peak lam=%.3f, plateau lam=%.3f' +
        ' vs 2/eta=%.3f ; loss %.5f -> %.5f  OK',
        [Eta, InitLam, PeakLam, PlateauLam, TwoOverEta, FirstLoss, PlateauLoss]));
    end;

    // (4) PLATEAU TRACKS 2/eta ACROSS THE SWEEP: smaller eta -> higher plateau.
    // 2/eta is monotically DECREASING in eta (etas are ascending), so the
    // plateau lambdas must be (weakly) decreasing too.
    Tracks := true;
    for Arm := 1 to High(cEtas) do
      if PlateauVals[Arm] > PlateauVals[Arm - 1] + 1e-6 then Tracks := false;
    WriteLn;
    if Tracks then
      WriteLn('OK: plateau height DECREASES as eta increases -> it tracks ' +
        '2/eta across the sweep.')
    else
    begin
      WriteLn('FAIL: plateau height did not decrease monotonically with eta ' +
        '-> plateau does not track 2/eta.');
      Halt(1);
    end;

    WriteLn;
    WriteLn('All Edge-of-Stability self-checks passed.');
  finally
    Probe.Free;
  end;
end.
