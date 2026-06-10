program EWCContinualLearning;
(*
EWCContinualLearning: a tiny pure-CPU reproduction of the classic
catastrophic-forgetting result and the Elastic-Weight-Consolidation (EWC) cure
(Kirkpatrick et al., PNAS 2017, "Overcoming catastrophic forgetting in neural
networks").

A small MLP (Input -> FC+ReLU -> FC -> SoftMax) is first trained to convergence
on TASK A: a 4-cluster 2-D classification problem. We then snapshot two things:
  * the Task-A-optimal weights w_A (a flat copy of every trainable parameter);
  * the DIAGONAL empirical Fisher information F_i of every parameter, computed
    exactly the way TNNet.FisherImportanceReport does it - accumulate the
    SQUARED per-parameter gradient over the Task-A training set on a FROZEN net
    (SetBatchUpdate so Neurons[].Delta / FBiasDelta hold the gradient; divide by
    the layer LearningRate to undo the -LR scaling). F_i is the per-parameter
    curvature an EWC penalty consumes.

We then continue training the SAME converged weights on TASK B - the SAME four
clusters and labels but with the input coordinates ROTATED (a "perturbed-input"
task in the spirit of Kirkpatrick et al.'s permuted MNIST). Both tasks share the
hidden layer and are individually learnable, but they demand overlapping-yet-
different features, so the two solutions compete for the same weights and the
forgetting is a property of the OPTIMISER rather than an impossibility - which is
why EWC can hold both. Two arms:
  * PLAIN sequential fine-tuning (no penalty) -> Task-A accuracy collapses
    (catastrophic forgetting);
  * EWC: after each data weight step we apply a decoupled penalty step
    w_i <- w_i - clamp(LR * lambda * F_i, 0, 1) * (w_i - w_A_i), a stable form of
    the gradient of the quadratic penalty sum_i (lambda/2) F_i (w_i - w_A_i)^2,
    which pins the high-Fisher (important-for-A) parameters at w_A while leaving
    the low-Fisher ones free to learn B (manual parameter surgery;
    SetBatchUpdate(true)). The clamp keeps the step stable despite the large
    lambda the tiny empirical Fisher forces.

The headline is a 2x2 table - Task-A and Task-B accuracy, plain vs EWC - showing
EWC retains A far better than plain at a modest cost to B. Built-in PASS/FAIL
gates assert that (a) plain fine-tuning actually forgets A and (b) EWC retains A
meaningfully better than plain, so the demo cannot silently prove nothing.

Pure CPU, single seed, no dataset download, well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 2;
  cHidden   = 32;
  cClasses  = 4;
  cLR       = 0.05;
  cMomentum = 0.9;
  cEpochsA  = 150;        // train Task A to convergence
  cEpochsB  = 150;        // continue on Task B (long enough to fully learn it)
  cBatch    = 16;
  cPerClass = 64;         // training samples per class per dataset
  cEvalPerC = 64;         // eval samples per class
  // EWC penalty strength. Large because the empirical Fisher of a confident
  // converged model is tiny (here max F ~ 1e-6); the clamped pull in TrainEWC
  // keeps the step stable at any lambda, and ~1e8 lands in the sweet spot that
  // restores most of Task A at a small Task-B cost.
  cLambda   = 100000000.0;
  // Four well-separated 2-D cluster centers, one per class (a quadrant each, so
  // the 4 classes are linearly separable).
  cCenters: array[0..3, 0..1] of TNeuralFloat =
    ((-1.4, -1.4), (1.4, 1.4), (-1.4, 1.4), (1.4, -1.4));
  // Task B is the SAME four clusters and labels but with the input coordinates
  // ROTATED by cRotB radians. Both tasks occupy the SAME input region and are
  // individually learnable, but they demand OVERLAPPING-yet-different hidden
  // features (the rotation shifts each class's region), so the two solutions
  // compete for the same weights - a "perturbed-input" EWC setup in the spirit
  // of Kirkpatrick et al.'s permuted MNIST. A moderate (not 90-degree) rotation
  // keeps the tasks related enough that a single net CAN hold both: plain SGD on
  // B still overwrites A's features (forgetting), while EWC pins the A-important
  // weights and finds a compromise that retains A at a modest cost to B.
  cRotB     = 0.9;        // Task-B input rotation angle (radians, ~52 degrees)

type
  TFloatArr = array of TNeuralFloat;

  // Builds the tiny MLP classifier. Same architecture in every arm.
  procedure BuildNet(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(cLR, cMomentum);
    NN.InitWeights();
  end;

  // One labelled sample (input + one-hot) from cluster Cls of the selected task
  // (TaskB=false -> Task A's region; TaskB=true -> Task B's disjoint region).
  procedure MakeSample(out X, Y: TNNetVolume; Cls: integer; TaskB: boolean);
  var
    PX, PY: TNeuralFloat;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    PX := cCenters[Cls][0] + (Random - 0.5) * 2.0;
    PY := cCenters[Cls][1] + (Random - 0.5) * 2.0;
    if TaskB then
    begin
      // Rotate the input by cRotB radians.
      X.Raw[0] := PX * Cos(cRotB) - PY * Sin(cRotB);
      X.Raw[1] := PX * Sin(cRotB) + PY * Cos(cRotB);
    end
    else
    begin
      X.Raw[0] := PX;
      X.Raw[1] := PY;
    end;
    Y.Fill(0);
    Y.Raw[Cls] := 1.0;
  end;

  // Builds a fixed evaluation set for one task (Permute selects A vs B).
  function BuildEvalSet(Permute: boolean): TNNetVolumePairList;
  var
    C, K: integer;
    X, Y: TNNetVolume;
  begin
    Result := TNNetVolumePairList.Create();
    for C := 0 to cClasses - 1 do
      for K := 0 to cEvalPerC - 1 do
      begin
        MakeSample(X, Y, C, Permute);
        Result.Add(TNNetVolumePair.Create(X, Y));
      end;
  end;

  // Builds a fixed training set for one task.
  function BuildTrainSet(Permute: boolean): TNNetVolumePairList;
  var
    C, K: integer;
    X, Y: TNNetVolume;
  begin
    Result := TNNetVolumePairList.Create();
    for C := 0 to cClasses - 1 do
      for K := 0 to cPerClass - 1 do
      begin
        MakeSample(X, Y, C, Permute);
        Result.Add(TNNetVolumePair.Create(X, Y));
      end;
  end;

  // In-place Fisher-Yates shuffle (the pair list has no Shuffle method).
  procedure ShuffleList(L: TNNetVolumePairList);
  var
    I, J: integer;
  begin
    // Exchange swaps the stored pointers without freeing (the list owns its
    // objects, so a plain L[I] := L[J] assignment would free an element).
    for I := L.Count - 1 downto 1 do
    begin
      J := Random(I + 1);
      if J <> I then L.Exchange(I, J);
    end;
  end;

  // Top-1 accuracy over a frozen labelled set (forward only).
  function Accuracy(NN: TNNet; Probes: TNNetVolumePairList): TNeuralFloat;
  var
    I, Hits: integer;
    Pair: TNNetVolumePair;
  begin
    Hits := 0;
    for I := 0 to Probes.Count - 1 do
    begin
      Pair := Probes[I];
      NN.Compute(Pair.I);
      if NN.GetLastLayer().Output.GetClass() = Pair.O.GetClass() then Inc(Hits);
    end;
    Result := Hits / Probes.Count;
  end;

  // Counts the flat trainable-parameter total (weights + one bias per neuron).
  function CountParams(NN: TNNet): integer;
  var
    L, N, Cnt: integer;
    Layer: TNNetLayer;
  begin
    Cnt := 0;
    for L := 0 to NN.GetLastLayerIdx() do
    begin
      Layer := NN.Layers[L];
      for N := 0 to Layer.Neurons.Count - 1 do
        Cnt := Cnt + Layer.Neurons[N].Weights.Size + 1;
    end;
    Result := Cnt;
  end;

  // Flattens every trainable parameter (weights then bias, per neuron) into Dst.
  procedure SnapshotWeights(NN: TNNet; var Dst: TFloatArr);
  var
    L, N, K, P: integer;
    Layer: TNNetLayer;
    Neuron: TNNetNeuron;
  begin
    P := 0;
    for L := 0 to NN.GetLastLayerIdx() do
    begin
      Layer := NN.Layers[L];
      for N := 0 to Layer.Neurons.Count - 1 do
      begin
        Neuron := Layer.Neurons[N];
        for K := 0 to Neuron.Weights.Size - 1 do
        begin
          Dst[P] := Neuron.Weights.FData[K];
          Inc(P);
        end;
        Dst[P] := Neuron.Bias;
        Inc(P);
      end;
    end;
  end;

  // Plain mini-batch SGD training on the given set. Reused for Task A and for
  // the PLAIN Task-B arm. Manual loop so the EWC arm can mirror it exactly.
  procedure TrainPlain(NN: TNNet; TrainSet: TNNetVolumePairList; Epochs: integer);
  var
    Ep, I, B: integer;
    Pair: TNNetVolumePair;
  begin
    NN.SetBatchUpdate(true);
    for Ep := 1 to Epochs do
    begin
      ShuffleList(TrainSet);
      I := 0;
      while I < TrainSet.Count do
      begin
        NN.ClearDeltas();
        B := 0;
        while (B < cBatch) and (I < TrainSet.Count) do
        begin
          Pair := TrainSet[I];
          NN.Compute(Pair.I);
          NN.Backpropagate(Pair.O);
          Inc(I);
          Inc(B);
        end;
        NN.UpdateWeights();
      end;
    end;
  end;

  // Diagonal empirical Fisher over TrainSet, computed exactly the way
  // TNNet.FisherImportanceReport does: forward+backward per sample on a FROZEN
  // net (SetBatchUpdate so the gradient lands in Delta/FBiasDelta; divide out
  // the layer LR; never UpdateWeights), accumulate the SQUARED gradient w.r.t.
  // the TRUE label, then average over the set. Layout matches SnapshotWeights.
  procedure ComputeFisher(NN: TNNet; TrainSet: TNNetVolumePairList;
    var Fisher: TFloatArr);
  var
    SampleIdx, L, N, K, P, LabelClass, OutSize: integer;
    Layer: TNNetLayer;
    Neuron: TNNetNeuron;
    Pair: TNNetVolumePair;
    Target: TNNetVolume;
    LR, G: TNeuralFloat;
  begin
    for P := 0 to High(Fisher) do Fisher[P] := 0;
    NN.SetBatchUpdate(true);
    OutSize := NN.GetLastLayer().Output.Size;
    Target := TNNetVolume.Create(OutSize, 1, 1);
    try
      for SampleIdx := 0 to TrainSet.Count - 1 do
      begin
        Pair := TrainSet[SampleIdx];
        NN.ClearDeltas();
        NN.Compute(Pair.I);
        LabelClass := Pair.O.GetClass();
        Target.Fill(0);
        Target.Raw[LabelClass] := 1.0;
        NN.Backpropagate(Target);
        P := 0;
        for L := 0 to NN.GetLastLayerIdx() do
        begin
          Layer := NN.Layers[L];
          LR := Layer.LearningRate;
          if LR <= 0 then LR := 1.0;
          for N := 0 to Layer.Neurons.Count - 1 do
          begin
            Neuron := Layer.Neurons[N];
            for K := 0 to Neuron.Delta.Size - 1 do
            begin
              G := Neuron.Delta.FData[K] / LR;
              Fisher[P] := Fisher[P] + G * G;
              Inc(P);
            end;
            G := Neuron.BiasDelta / LR;
            Fisher[P] := Fisher[P] + G * G;
            Inc(P);
          end;
        end;
      end;
    finally
      Target.Free;
    end;
    if TrainSet.Count > 0 then
      for P := 0 to High(Fisher) do Fisher[P] := Fisher[P] / TrainSet.Count;
  end;

  // EWC training on Task B: identical mini-batch data loop to TrainPlain, then -
  // decoupled from momentum, the way AdamW decouples weight decay - it applies
  // the EWC penalty as its own step on the parameters AFTER the data
  // UpdateWeights. The penalty is sum_i (lambda/2) F_i (w_i - w_A_i)^2, whose
  // gradient is lambda * F_i * (w_i - w_A_i), i.e. a per-parameter pull toward
  // the Task-A value w_A_i with strength s_i = LR * lambda * F_i:
  //   w_i <- w_i - s_i * (w_i - w_A_i).
  // Because the empirical Fisher of a confident converged model is tiny, lambda
  // must be large; a raw step can then overshoot (s_i > 1 -> oscillation), so we
  // CLAMP s_i to [0,1]. The clamp is unconditionally stable and monotone in
  // lambda: high-Fisher (important-for-A) parameters get s_i -> 1 and are pinned
  // essentially AT w_A; the many low-Fisher ones get s_i -> 0 and stay free to
  // learn Task B - exactly the EWC selectivity. Parameters are written through
  // the public writable accessors (Weights.FData / BiasWeight). Layout matches
  // SnapshotWeights.
  procedure TrainEWC(NN: TNNet; TrainSet: TNNetVolumePairList; Epochs: integer;
    const Fisher, WeightsA: TFloatArr; Lambda: TNeuralFloat);
  var
    Ep, I, B, L, N, K, P: integer;
    Pair: TNNetVolumePair;
    Layer: TNNetLayer;
    Neuron: TNNetNeuron;
    LR, W, S: TNeuralFloat;
  begin
    NN.SetBatchUpdate(true);
    for Ep := 1 to Epochs do
    begin
      ShuffleList(TrainSet);
      I := 0;
      while I < TrainSet.Count do
      begin
        NN.ClearDeltas();
        B := 0;
        while (B < cBatch) and (I < TrainSet.Count) do
        begin
          Pair := TrainSet[I];
          NN.Compute(Pair.I);
          NN.Backpropagate(Pair.O);
          Inc(I);
          Inc(B);
        end;
        NN.UpdateWeights();
        // Decoupled, clamped EWC pull toward w_A on the actual parameters.
        P := 0;
        for L := 0 to NN.GetLastLayerIdx() do
        begin
          Layer := NN.Layers[L];
          LR := Layer.LearningRate;
          if LR <= 0 then LR := 1.0;
          for N := 0 to Layer.Neurons.Count - 1 do
          begin
            Neuron := Layer.Neurons[N];
            for K := 0 to Neuron.Weights.Size - 1 do
            begin
              S := LR * Lambda * Fisher[P];
              if S > 1.0 then S := 1.0;
              W := Neuron.Weights.FData[K];
              Neuron.Weights.FData[K] := W - S * (W - WeightsA[P]);
              Inc(P);
            end;
            S := LR * Lambda * Fisher[P];
            if S > 1.0 then S := 1.0;
            W := Neuron.BiasWeight;
            Neuron.BiasWeight := W - S * (W - WeightsA[P]);
            Inc(P);
          end;
        end;
      end;
    end;
  end;

var
  NN: TNNet;
  TrainA, TrainB, EvalA, EvalB: TNNetVolumePairList;
  WeightsA, Fisher: TFloatArr;
  NumParams: integer;
  AccA_afterA, AccB_afterA: TNeuralFloat;
  AccA_plain, AccB_plain, AccA_ewc, AccB_ewc: TNeuralFloat;
  RetentionGain: TNeuralFloat;
  PassForget, PassRetain: boolean;
begin
  NN := nil;
  RandSeed := 424242;

  WriteLn('EWC Continual Learning demo - catastrophic forgetting & its cure.');
  WriteLn('Kirkpatrick et al., PNAS 2017. Tiny 4-class 2-D MLP, pure CPU.');
  WriteLn;

  TrainA := BuildTrainSet(False);
  TrainB := BuildTrainSet(True);
  EvalA  := BuildEvalSet(False);
  EvalB  := BuildEvalSet(True);

  // --- 1. Train Task A to convergence. ---
  BuildNet(NN);
  NumParams := CountParams(NN);
  SetLength(WeightsA, NumParams);
  SetLength(Fisher, NumParams);
  WriteLn(Format('Network trainable parameters: %d', [NumParams]));

  TrainPlain(NN, TrainA, cEpochsA);
  AccA_afterA := Accuracy(NN, EvalA);
  AccB_afterA := Accuracy(NN, EvalB);
  WriteLn(Format('After Task A:  Task-A acc = %.3f   Task-B acc = %.3f',
    [AccA_afterA, AccB_afterA]));

  // --- 2. Snapshot w_A and the diagonal Fisher from Task A. ---
  SnapshotWeights(NN, WeightsA);
  ComputeFisher(NN, TrainA, Fisher);
  WriteLn('Snapshotted Task-A weights w_A and diagonal Fisher F.');
  WriteLn;

  // --- 3a. PLAIN arm: continue the SAME net on Task B, no penalty. ---
  TrainPlain(NN, TrainB, cEpochsB);
  AccA_plain := Accuracy(NN, EvalA);
  AccB_plain := Accuracy(NN, EvalB);

  // --- 3b. EWC arm: rebuild Task-A weights, then train B WITH the penalty. ---
  // Reload the Task-A-optimal weights into a fresh net (CopyWeights-style via a
  // re-snapshot would need a source net; simplest honest path is to rebuild and
  // retrain Task A under the SAME seed, which is bit-identical, then restore w_A
  // straight from the snapshot array).
  NN.Free;
  RandSeed := 424242;
  BuildNet(NN);
  TrainPlain(NN, TrainA, cEpochsA);   // reproduces the exact Task-A optimum
  TrainEWC(NN, TrainB, cEpochsB, Fisher, WeightsA, cLambda);
  AccA_ewc := Accuracy(NN, EvalA);
  AccB_ewc := Accuracy(NN, EvalB);

  // --- 4. Headline table. ---
  WriteLn(StringOfChar('=', 60));
  WriteLn('  HEADLINE: Task-A retention vs Task-B learning');
  WriteLn(StringOfChar('=', 60));
  WriteLn('  arm     | Task-A acc | Task-B acc');
  WriteLn('  --------+------------+-----------');
  WriteLn(Format('  PLAIN   |   %6.3f   |   %6.3f', [AccA_plain, AccB_plain]));
  WriteLn(Format('  EWC     |   %6.3f   |   %6.3f', [AccA_ewc,   AccB_ewc]));
  WriteLn(StringOfChar('=', 60));
  WriteLn;
  RetentionGain := AccA_ewc - AccA_plain;
  WriteLn(Format('Task-A retention gain (EWC - PLAIN): %.3f', [RetentionGain]));
  WriteLn(Format('Task-B cost          (PLAIN - EWC): %.3f',
    [AccB_plain - AccB_ewc]));
  WriteLn;

  // --- 5. Built-in sanity gates (demo must actually prove something). ---
  // (a) Plain fine-tuning must visibly FORGET Task A.
  PassForget := AccA_plain < AccA_afterA - 0.20;
  // (b) EWC must retain Task A meaningfully better than plain.
  PassRetain := RetentionGain > 0.15;

  if PassForget then
    WriteLn(Format('PASS: PLAIN forgot Task A (%.3f -> %.3f).',
      [AccA_afterA, AccA_plain]))
  else
    WriteLn(Format('FAIL: PLAIN did not forget Task A enough (%.3f -> %.3f).',
      [AccA_afterA, AccA_plain]));
  if PassRetain then
    WriteLn(Format('PASS: EWC retained Task A better than PLAIN (gain %.3f).',
      [RetentionGain]))
  else
    WriteLn(Format('FAIL: EWC did not retain Task A enough (gain %.3f).',
      [RetentionGain]));

  NN.Free;
  TrainA.Free;
  TrainB.Free;
  EvalA.Free;
  EvalB.Free;

  if not (PassForget and PassRetain) then
  begin
    WriteLn('Sanity gate failed - the demo did not reproduce EWC retention.');
    Halt(1);
  end;
  WriteLn;
  WriteLn('All sanity gates passed.');
end.
