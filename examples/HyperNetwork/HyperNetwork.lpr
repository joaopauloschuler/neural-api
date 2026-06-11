program HyperNetwork;
(*
HyperNetwork: a tiny, pure-CPU reproduction of the core idea of Ha, Dai & Le
2016, "HyperNetworks". A small GENERATOR network consumes a per-task CONTEXT
vector (here a one-hot task id, mapped through a learned task embedding) and
EMITS the weight matrix (+ bias) of a "main" layer. That generated weight
matrix is then applied to the actual 2D input by a TNNetHyperLinear, whose
forward reads its weights from the generated-weights TENSOR rather than from
its own Neurons[]. One shared main layer thus implements a whole FAMILY of
input->output maps; the per-task behaviour is carried ENTIRELY by the
context-conditioned generated weights, and the whole stack (embedding +
generator + hyper layer) trains end-to-end by ordinary backprop.

Multi-task target: cTasks distinct 2D->2D linear maps. Task t rotates the input
by angle theta_t and scales it by s_t:
    y = s_t * R(theta_t) * x        (a different 2x2 matrix per task)
These are genuinely DIFFERENT functions of the same input space, selected only
by the task id. A single ordinary shared linear layer cannot fit them all (it
has one fixed matrix); the hyper layer can, because its matrix is regenerated
from the task context on every forward pass.

Network wiring (two input branches, joined at the hyper layer):
    CONTEXT branch:  Input(one-hot task id, cTasks)
                       -> FullConnect(cHidden)            (task embedding)
                       -> FullConnectLinear(Din*Dout+Dout) = GENERATED WEIGHTS
    FEATURE branch:  Input(2D point)
                       -> TNNetHyperLinear(Dout, WeightsSource = generated)

Baseline for contrast: ONE shared TNNetFullConnectLinear(Dout) trained on the
same mixed multi-task stream (no task conditioning) -- it is forced to average
the tasks and cannot solve the family.

Everything is generated on the fly; no external dataset. Pure CPU, single
thread, well under five minutes (a couple of seconds in practice).

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
  cTasks   = 4;       // number of distinct 2D->2D tasks
  cDin     = 2;       // input dimension (a 2D point)
  cDout    = 2;       // output dimension (rotated/scaled 2D point)
  cHidden  = 8;       // generator hidden width (task embedding)
  cWeights = cDin * cDout + cDout;   // generated weight-matrix + bias size
  cSteps   = 2500;
  cBatch   = 16;
  cLR      = 0.02;
  cInertia = 0.9;
  cTestSet = 1024;    // held-out test points per task

var
  GAngle: array[0..cTasks - 1] of TNeuralFloat;  // per-task rotation angle
  GScale: array[0..cTasks - 1] of TNeuralFloat;  // per-task scale

  // Define the cTasks distinct teacher maps (unknown to the models).
  procedure InitTasks;
  var T: integer;
  begin
    for T := 0 to cTasks - 1 do
    begin
      GAngle[T] := (Pi / 2.0) * T / cTasks + 0.15; // spread the rotations out
      GScale[T] := 0.6 + 0.25 * T;                 // a different scale each task
    end;
  end;

  // Teacher map for task t: y = s_t * R(theta_t) * x.
  procedure ApplyTask(T: integer; const x0, x1: TNeuralFloat;
    out y0, y1: TNeuralFloat);
  var c, s: TNeuralFloat;
  begin
    c := GScale[T] * Cos(GAngle[T]);
    s := GScale[T] * Sin(GAngle[T]);
    y0 := c * x0 - s * x1;
    y1 := s * x0 + c * x1;
  end;

  // One (input, target) pair for task T, plus the one-hot task context.
  procedure MakePair(T: integer; FeatV, CtxV, TargetV: TNNetVolume);
  var x0, x1, y0, y1: TNeuralFloat; I: integer;
  begin
    x0 := Random * 2.0 - 1.0;
    x1 := Random * 2.0 - 1.0;
    FeatV.FData[0] := x0;
    FeatV.FData[1] := x1;
    for I := 0 to cTasks - 1 do CtxV.FData[I] := 0;
    CtxV.FData[T] := 1;                 // one-hot task id
    ApplyTask(T, x0, x1, y0, y1);
    TargetV.FData[0] := y0;
    TargetV.FData[1] := y1;
  end;

  function MeanSquaredError(Output, Target: TNNetVolume): TNeuralFloat;
  var I: integer; diff: TNeuralFloat;
  begin
    Result := 0;
    for I := 0 to Output.Size - 1 do
    begin
      diff := Output.FData[I] - Target.FData[I];
      Result := Result + diff * diff;
    end;
    Result := Result / Output.Size;
  end;

var
  // ---- HyperNetwork model ----
  NNHyper: TNNet;
  FeatIn, CtxIn, GenW, HyperLayer: TNNetLayer;
  // ---- Shared-linear baseline ----
  NNBase: TNNet;

  FeatV, CtxV, TargetV: TNNetVolume;
  Step, B, T, I: integer;
  SumLoss: TNeuralFloat;
  HyperMSE, BaseMSE: TNeuralFloat;
  PerTaskHyper, PerTaskBase: array[0..cTasks - 1] of TNeuralFloat;
  Cnt: integer;
  StartTime, Elapsed: double;
  Saved: string;
  NNReload: TNNet;
  ReloadMSE: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  StartTime := Now();
  InitTasks;

  WriteLn('HyperNetwork demo: ', cTasks,
    ' distinct 2D->2D tasks solved by ONE shared main layer');
  WriteLn('whose weights are GENERATED per-task from a learned context ',
    '(Ha et al. 2016).');
  WriteLn;

  // ================= Build the HyperNetwork =================
  RandSeed := 2026;
  NNHyper := TNNet.Create();
  // Feature branch (layer 0): the 2D point.
  FeatIn := NNHyper.AddLayer(TNNetInput.Create(cDin, 1, 1));
  // Context branch (layer 1+): one-hot task id -> embedding -> generated weights.
  CtxIn  := NNHyper.AddLayerAfter(TNNetInput.Create(cTasks, 1, 1), 0);
  NNHyper.AddLayer(TNNetFullConnect.Create(cHidden));          // task embedding
  GenW   := NNHyper.AddLayer(TNNetFullConnectLinear.Create(cWeights)); // weights
  // The hyper layer: main input = FeatIn, weight source = GenW.
  HyperLayer := NNHyper.AddLayerAfter(
    TNNetHyperLinear.Create(cDout, {UseBias=}true, GenW), FeatIn);
  NNHyper.SetLearningRate(cLR, cInertia);

  // ================= Build the shared-linear baseline =================
  RandSeed := 2026;
  NNBase := TNNet.Create();
  NNBase.AddLayer(TNNetInput.Create(cDin, 1, 1));
  NNBase.AddLayer(TNNetFullConnectLinear.Create(cDout));
  NNBase.SetLearningRate(cLR, cInertia);

  FeatV   := TNNetVolume.Create(cDin, 1, 1);
  CtxV    := TNNetVolume.Create(cTasks, 1, 1);
  TargetV := TNNetVolume.Create(cDout, 1, 1);
  try
    WriteLn('HyperNetwork structure:');
    WriteLn('  context Input(', cTasks, ') -> FullConnect(', cHidden,
      ') -> FullConnectLinear(', cWeights, ') = generated weights');
    WriteLn('  feature Input(', cDin, ') -> TNNetHyperLinear(', cDout,
      ', WeightsSource=generated)');
    WriteLn('  (the hyper layer itself owns ZERO trainable weights: ',
      HyperLayer.CountWeights, ')');
    WriteLn;

    // ---- Train the HyperNetwork ----
    WriteLn('Training HyperNetwork...');
    RandSeed := 2026;
    for Step := 1 to cSteps do
    begin
      SumLoss := 0;
      for B := 1 to cBatch do
      begin
        T := Random(cTasks);
        MakePair(T, FeatV, CtxV, TargetV);
        FeatIn.Output.Copy(FeatV);
        CtxIn.Output.Copy(CtxV);
        NNHyper.Compute(FeatIn.Output);
        SumLoss := SumLoss +
          MeanSquaredError(NNHyper.GetLastLayer.Output, TargetV);
        NNHyper.Backpropagate(TargetV);
      end;
      if (Step = 1) or (Step mod 500 = 0) or (Step = cSteps) then
        WriteLn(Format('  [hyper] step %4d / %4d   train-MSE=%.6e',
          [Step, cSteps, SumLoss / cBatch]));
    end;

    // ---- Train the shared-linear baseline on the same mixed stream ----
    WriteLn('Training shared-linear baseline (no task conditioning)...');
    RandSeed := 2026;
    for Step := 1 to cSteps do
    begin
      for B := 1 to cBatch do
      begin
        T := Random(cTasks);
        MakePair(T, FeatV, CtxV, TargetV);
        NNBase.Compute(FeatV);
        NNBase.Backpropagate(TargetV);
      end;
    end;
    WriteLn;

    // ================= Evaluate per-task on a held-out set =================
    for T := 0 to cTasks - 1 do begin PerTaskHyper[T] := 0; PerTaskBase[T] := 0; end;
    HyperMSE := 0; BaseMSE := 0;
    RandSeed := 99991;
    for I := 1 to cTestSet do
      for T := 0 to cTasks - 1 do
      begin
        MakePair(T, FeatV, CtxV, TargetV);
        // HyperNetwork
        FeatIn.Output.Copy(FeatV);
        CtxIn.Output.Copy(CtxV);
        NNHyper.Compute(FeatIn.Output);
        PerTaskHyper[T] := PerTaskHyper[T] +
          MeanSquaredError(NNHyper.GetLastLayer.Output, TargetV);
        // Baseline
        NNBase.Compute(FeatV);
        PerTaskBase[T] := PerTaskBase[T] +
          MeanSquaredError(NNBase.GetLastLayer.Output, TargetV);
      end;
    Cnt := cTestSet;
    for T := 0 to cTasks - 1 do
    begin
      PerTaskHyper[T] := PerTaskHyper[T] / Cnt;
      PerTaskBase[T]  := PerTaskBase[T] / Cnt;
      HyperMSE := HyperMSE + PerTaskHyper[T];
      BaseMSE  := BaseMSE  + PerTaskBase[T];
    end;
    HyperMSE := HyperMSE / cTasks;
    BaseMSE  := BaseMSE  / cTasks;

    WriteLn(StringOfChar('=', 64));
    WriteLn('RESULTS (per-task held-out test MSE)');
    WriteLn(StringOfChar('-', 64));
    WriteLn(Format('  %-6s %14s %16s', ['task', 'hyper', 'shared-linear']));
    for T := 0 to cTasks - 1 do
      WriteLn(Format('  %-6d %14.3e %16.3e',
        [T, PerTaskHyper[T], PerTaskBase[T]]));
    WriteLn(StringOfChar('-', 64));
    WriteLn(Format('  %-6s %14.3e %16.3e', ['mean', HyperMSE, BaseMSE]));
    WriteLn(StringOfChar('=', 64));
    WriteLn(Format('  hyper mean-MSE %.2e  vs  shared-linear mean-MSE %.2e',
      [HyperMSE, BaseMSE]));
    WriteLn('  Per-task generated weights SOLVE the family (MSE -> ~0),');
    WriteLn('  while a single fixed shared matrix can only average them.');
    WriteLn(StringOfChar('=', 64));
    WriteLn;

    // ================= Save / load round-trip =================
    Saved := NNHyper.SaveToString();
    NNReload := TNNet.Create();
    try
      NNReload.LoadFromString(Saved);
      // Re-evaluate the reloaded net on task 0 to confirm it round-trips.
      RandSeed := 99991;
      ReloadMSE := 0;
      for I := 1 to cTestSet do
      begin
        MakePair(0, FeatV, CtxV, TargetV);
        NNReload.Layers[0].Output.Copy(FeatV);
        NNReload.Layers[1].Output.Copy(CtxV);
        NNReload.Compute(NNReload.Layers[0].Output);
        ReloadMSE := ReloadMSE +
          MeanSquaredError(NNReload.GetLastLayer.Output, TargetV);
      end;
      ReloadMSE := ReloadMSE / cTestSet;
      WriteLn(Format('Save/load round-trip: reloaded net task-0 MSE=%.3e ' +
        '(original %.3e).', [ReloadMSE, PerTaskHyper[0]]));
    finally
      NNReload.Free;
    end;

    Elapsed := (Now() - StartTime) * 86400.0;
    WriteLn;
    WriteLn(Format('Total wall-clock: %.1f s (pure CPU, single thread).',
      [Elapsed]));
  finally
    NNHyper.Free;
    NNBase.Free;
    FeatV.Free;
    CtxV.Free;
    TargetV.Free;
  end;
end.
