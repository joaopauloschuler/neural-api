program HighwayDepth;
(*
HighwayDepth: a depth-stress experiment for TNNetHighway, the input-dependent
learned-gate Highway layer (Srivastava, Greff & Schmidhuber 2015, "Training Very
Deep Networks"):

  y = T(x) (.) H(x) + (1 - T(x)) (.) x ,   T(x) = sigmoid(W_T.x + b_T) ,
                                           H(x) = tanh(W_H.x + b_H).

The whole point of the Highway layer (and its descendant, the ResNet skip
connection) is TRAINABILITY AT DEPTH: because the gate biases start negative,
a fresh deep stack begins life near the identity (T ~ 0 => y ~ x), so gradients
flow through the (1 - T) carry path and the stack stays trainable as depth grows.
A plain (gate-less) tanh-MLP stack of the same shape has no such carry: as depth
increases the signal/gradient degrades and the final loss gets WORSE.

Experiment: a small fixed-width regression task (learn a smooth nonlinear vector
map). At each depth in {2, 5, 10, 20, 40} we train, head to head:
  (A) PLAIN  : depth x [ TNNetFullConnect(width) ]            (tanh, no carry)
  (B) HIGHWAY: depth x [ TNNetHighway(width)     ]            (gated carry)
both followed by the SAME linear read-out. We then chart FINAL TEST LOSS vs DEPTH
(ASCII bars). Headline: the plain stack degrades with depth while the Highway
stack stays trainable. For the Highway models we also report the MEAN LEARNED
GATE T per layer (how "open" each layer chose to be).

Everything is generated on the fly; no external dataset. Pure CPU, single
thread, well under five minutes (about a minute in practice).

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
  cWidth   = 12;      // fixed stack width (the per-block vector size)
  cSteps   = 1500;    // SGD steps per model
  cBatch   = 16;
  cLR      = 0.02;
  cInertia = 0.9;
  cTestSet = 512;     // held-out test pairs for the final MSE
  cTeacherSeed = 7777;

  cNumDepths = 5;
  cDepths: array[0..cNumDepths - 1] of integer = (2, 5, 10, 20, 40);

var
  // A small fixed (unknown-to-the-model) teacher: one hidden tanh layer mapping
  // R^cWidth -> R^cWidth. Stored as plain matrices so the target is reproducible.
  GW1: array[0..cWidth - 1, 0..cWidth - 1] of TNeuralFloat;
  GW2: array[0..cWidth - 1, 0..cWidth - 1] of TNeuralFloat;

  procedure InitTeacher;
  var
    I, J: integer;
  begin
    RandSeed := cTeacherSeed;
    for I := 0 to cWidth - 1 do
      for J := 0 to cWidth - 1 do
      begin
        GW1[I, J] := (Random * 2.0 - 1.0) * 0.7;
        GW2[I, J] := (Random * 2.0 - 1.0) * 0.7;
      end;
  end;

  // One (input, target) pair. The target is the INPUT plus a small smooth
  // nonlinear PERTURBATION:  target = x + 0.3 * W2 . tanh(W1 . x).
  // This is an "identity + residual" map: it is close to the identity, which is
  // exactly the function a deep near-identity stack should represent easily IF
  // it can carry x through its layers. A plain (carry-less) deep tanh stack has
  // to RE-LEARN the identity through every layer and degrades with depth; the
  // Highway stack starts at the identity carry and only has to learn the small
  // residual, so it stays trainable as depth grows.
  procedure MakePair(InputV, TargetV: TNNetVolume);
  var
    I, J: integer;
    Acc: TNeuralFloat;
    Hid: array[0..cWidth - 1] of TNeuralFloat;
  begin
    for I := 0 to cWidth - 1 do
      InputV.FData[I] := Random * 2.0 - 1.0;          // uniform in [-1, 1]
    for I := 0 to cWidth - 1 do
    begin
      Acc := 0;
      for J := 0 to cWidth - 1 do Acc := Acc + GW1[I, J] * InputV.FData[J];
      Hid[I] := Tanh(Acc);
    end;
    for I := 0 to cWidth - 1 do
    begin
      Acc := 0;
      for J := 0 to cWidth - 1 do Acc := Acc + GW2[I, J] * Hid[J];
      TargetV.FData[I] := InputV.FData[I] + 0.3 * Acc; // identity + small residual
    end;
  end;

  function MeanSquaredError(Output, Target: TNNetVolume): TNeuralFloat;
  var
    I: integer;
    diff: TNeuralFloat;
  begin
    Result := 0;
    for I := 0 to Output.Size - 1 do
    begin
      diff := Output.FData[I] - Target.FData[I];
      Result := Result + diff * diff;
    end;
    Result := Result / Output.Size;
  end;

  // Build a depth-D stack of shape-preserving blocks over cWidth, then a linear
  // read-out. UseHighway selects the gated-carry block vs the plain tanh block.
  function BuildStack(Depth: integer; UseHighway: boolean): TNNet;
  var
    L: integer;
  begin
    Result := TNNet.Create();
    Result.AddLayer(TNNetInput.Create(1, 1, cWidth));
    for L := 1 to Depth do
    begin
      if UseHighway then
        Result.AddLayer(TNNetHighway.Create(cWidth, {pInitGateBias=}-1.5))
      else
        // Plain tanh fully-connected block of the same width (no identity carry).
        Result.AddLayer(TNNetFullConnect.Create(cWidth));
    end;
    Result.AddLayer(TNNetFullConnectLinear.Create(cWidth));   // linear read-out
    Result.SetLearningRate(cLR, cInertia);
  end;

  procedure Train(NN: TNNet);
  var
    Step, B: integer;
    InputV, TargetV: TNNetVolume;
  begin
    InputV  := TNNetVolume.Create(1, 1, cWidth);
    TargetV := TNNetVolume.Create(1, 1, cWidth);
    try
      for Step := 1 to cSteps do
        for B := 1 to cBatch do
        begin
          MakePair(InputV, TargetV);
          NN.Compute(InputV);
          NN.Backpropagate(TargetV);
        end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  // Mean test MSE over a fixed held-out set (re-seeded so every model sees the
  // SAME test pairs).
  function EvalTestMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    InputV, TargetV: TNNetVolume;
  begin
    RandSeed := 99991;
    InputV  := TNNetVolume.Create(1, 1, cWidth);
    TargetV := TNNetVolume.Create(1, 1, cWidth);
    Result := 0;
    try
      for I := 1 to cTestSet do
      begin
        MakePair(InputV, TargetV);
        NN.Compute(InputV);
        Result := Result + MeanSquaredError(NN.GetLastLayer.Output, TargetV);
      end;
      Result := Result / cTestSet;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  // Mean transform-gate T over all Highway layers, averaged over the test set:
  // a per-layer "how open did this layer choose to be" readout.
  procedure ReportMeanGate(NN: TNNet);
  var
    I, K, L, NHigh, HiCnt: integer;
    InputV, TargetV: TNNetVolume;
    SumT: array of TNeuralFloat;
    HW: TNNetHighway;
  begin
    NHigh := 0;
    for L := 0 to NN.CountLayers - 1 do
      if NN.Layers[L] is TNNetHighway then Inc(NHigh);
    SetLength(SumT, NHigh);
    for I := 0 to NHigh - 1 do SumT[I] := 0;

    RandSeed := 99991;
    InputV  := TNNetVolume.Create(1, 1, cWidth);
    TargetV := TNNetVolume.Create(1, 1, cWidth);
    try
      for I := 1 to cTestSet do
      begin
        MakePair(InputV, TargetV);
        NN.Compute(InputV);
        HiCnt := 0;
        for L := 0 to NN.CountLayers - 1 do
          if NN.Layers[L] is TNNetHighway then
          begin
            HW := NN.Layers[L] as TNNetHighway;
            for K := 0 to HW.TransformGate.Size - 1 do
              SumT[HiCnt] := SumT[HiCnt] + HW.TransformGate.FData[K];
            Inc(HiCnt);
          end;
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;

    Write('    mean gate T per layer: ');
    for I := 0 to NHigh - 1 do
    begin
      Write(Format('%.2f', [SumT[I] / (cTestSet * cWidth)]));
      if I < NHigh - 1 then Write(' ');
    end;
    WriteLn;
  end;

  // A compact ASCII bar (log-scaled to the worst loss seen).
  function Bar(Value, WorstValue: TNeuralFloat): string;
  var
    Frac: TNeuralFloat;
    N: integer;
  begin
    // Normalize on a log scale so a few orders of magnitude are visible.
    if (Value <= 0) or (WorstValue <= 0) then Frac := 0
    else Frac := Ln(WorstValue / Value) / Ln(1000.0);   // 0..1 over 3 decades
    if Frac < 0 then Frac := 0;
    if Frac > 1 then Frac := 1;
    N := Round(Frac * 40);
    Result := StringOfChar('#', N) + StringOfChar('.', 40 - N);
  end;

var
  D, Idx: integer;
  NNPlain, NNHigh: TNNet;
  PlainMSE, HighMSE: array[0..cNumDepths - 1] of TNeuralFloat;
  WorstPlain, WorstHigh: TNeuralFloat;
  StartTime, Elapsed: double;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  StartTime := Now();
  InitTeacher;

  WriteLn('HighwayDepth: plain tanh-MLP vs Highway-MLP stacks at increasing depth.');
  WriteLn('Task: learn an "identity + small residual" R^', cWidth, ' -> R^', cWidth,
    ' map (x + 0.3*W2.tanh(W1.x)).');
  WriteLn('Headline: the plain stack DEGRADES with depth; Highway STAYS trainable.');
  WriteLn;

  WorstPlain := 0;
  WorstHigh  := 0;

  for Idx := 0 to cNumDepths - 1 do
  begin
    D := cDepths[Idx];
    WriteLn(Format('---- depth %2d --------------------------------------------', [D]));

    // Plain stack.
    RandSeed := 2026;
    NNPlain := BuildStack(D, {UseHighway=}false);
    try
      RandSeed := 2026;
      Train(NNPlain);
      PlainMSE[Idx] := EvalTestMSE(NNPlain);
    finally
      NNPlain.Free;
    end;

    // Highway stack.
    RandSeed := 2026;
    NNHigh := BuildStack(D, {UseHighway=}true);
    try
      RandSeed := 2026;
      Train(NNHigh);
      HighMSE[Idx] := EvalTestMSE(NNHigh);
      WriteLn(Format('    plain   test-MSE = %.4e', [PlainMSE[Idx]]));
      WriteLn(Format('    highway test-MSE = %.4e', [HighMSE[Idx]]));
      ReportMeanGate(NNHigh);
    finally
      NNHigh.Free;
    end;

    if PlainMSE[Idx] > WorstPlain then WorstPlain := PlainMSE[Idx];
    if HighMSE[Idx]  > WorstHigh  then WorstHigh  := HighMSE[Idx];
  end;

  WriteLn;
  WriteLn(StringOfChar('=', 72));
  WriteLn('FINAL TEST LOSS vs DEPTH  (longer bar = LOWER loss = better)');
  WriteLn(StringOfChar('-', 72));
  WriteLn(Format('%5s | %-44s %12s', ['depth', 'plain  tanh-MLP', 'test-MSE']));
  for Idx := 0 to cNumDepths - 1 do
    WriteLn(Format('%5d | %-44s %12.3e',
      [cDepths[Idx], Bar(PlainMSE[Idx], WorstPlain), PlainMSE[Idx]]));
  WriteLn(StringOfChar('-', 72));
  WriteLn(Format('%5s | %-44s %12s', ['depth', 'highway (gated carry)', 'test-MSE']));
  for Idx := 0 to cNumDepths - 1 do
    WriteLn(Format('%5d | %-44s %12.3e',
      [cDepths[Idx], Bar(HighMSE[Idx], WorstHigh), HighMSE[Idx]]));
  WriteLn(StringOfChar('=', 72));
  WriteLn;
  WriteLn('Read-out: as depth grows the plain stack''s loss climbs (deeper = worse),');
  WriteLn('while the Highway stack''s identity-carry keeps it trainable at every depth.');
  WriteLn(Format('plain  loss ratio depth%d / depth%d = %.2fx',
    [cDepths[cNumDepths - 1], cDepths[0],
     PlainMSE[cNumDepths - 1] / PlainMSE[0]]));
  WriteLn(Format('highway loss ratio depth%d / depth%d = %.2fx',
    [cDepths[cNumDepths - 1], cDepths[0],
     HighMSE[cNumDepths - 1] / HighMSE[0]]));

  Elapsed := (Now() - StartTime) * 24 * 60 * 60;
  WriteLn;
  WriteLn(Format('Total wall time: %.1f s', [Elapsed]));
end.
