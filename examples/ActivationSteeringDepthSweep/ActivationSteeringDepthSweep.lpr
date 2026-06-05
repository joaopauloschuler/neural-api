program ActivationSteeringDepthSweep;
(*
ActivationSteeringDepthSweep: the DEPTH-SWEEP follow-up to the landed
examples/ActivationSteering example. ActivationSteering injects a concept
direction into ONE fixed hidden layer (k=2) and shows the injection CAUSALLY
controls the softmax output (ActAdd / activation addition / representation
engineering, Turner et al. 2023). This example asks the natural next question:

    WHERE does a concept vector bite hardest?

It sweeps the steering layer k across ALL steerable hidden layers (not just one)
and, per layer, charts how cleanly the diff-of-class-means direction controls
P(target) and how small an alpha is needed to flip the prediction.

Recipe (pure CPU, no dataset, forward-only — weights are NEVER stepped after the
classifier is trained):
  1. Train a small softmax classifier on a synthetic 2-class toy (a two-cluster /
     sign-of-x0 task, the same synthetic style as examples/ActivationSteering),
     but with FOUR steerable FC+ReLU hidden layers so "sweep across all hidden
     layers" is meaningful.
  2. For EACH steerable hidden layer k: compute a STEERING VECTOR
        v_k = mean(act_k | class 1) - mean(act_k | class 0)
     a diff-of-class-means direction over the TRAINING activations at layer k.
     No extra training is run.
  3. Run forward layer-by-layer up to k, do Output_k.MulAdd(alpha, v_k), then
     recompute layers k+1..last (the SAME recompute machinery the landed
     ActivationPatchingReport / ActivationSteering drive: overwrite a cached
     activation and call FLayers[i].Compute() for i = k+1..last).
  4. Sweep alpha in {-3,-2,-1,0,1,2,3}, recording P(target class) vs alpha.

Per-layer report:
  - the P(target)-vs-alpha curve (ASCII bar);
  - a MONOTONICITY measure: the fraction of adjacent-alpha steps that increase
    (1.0 = perfectly monotone up in alpha);
  - the ALPHA-TO-FLIP: the smallest |alpha| at which the predicted argmax flips
    from the plain argmax (reported as "none" if it never flips in range).
Then a SUMMARY line naming which k gave the cleanest monotone control and which
gave the smallest alpha-to-flip.

Built-in correctness check (carried over from ActivationSteering, applied at
EVERY layer k): alpha = 0 reproduces the unsteered forward pass BIT-FOR-BIT.

Pure CPU, no dataset download, well under a minute.

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
  Classes, SysUtils,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 6;
  cHidden   = 12;
  cClasses  = 2;
  cEpochs   = 200;
  cBlob     = 1.3;    // distance of the signal coord from the origin
  cNoise    = 0.30;   // per-coordinate Gaussian-ish spread
  cMeanN    = 400;    // training samples per class for the mean estimate

  // The four steerable hidden layer indices (FC+ReLU layers). See BuildNet.
  cFirstK   = 1;
  cLastK    = 4;

  // A plain feed-forward softmax classifier with FOUR steerable FC+ReLU hidden
  // layers so a depth sweep is meaningful. The deciding feature is the SIGN of
  // x0 (a two-cluster task); the remaining coordinates are pure noise
  // distractors. We keep it a simple chain (no skip) so each hidden layer
  // carries a clean linear "class direction" the diff-of-means picks up.
  //
  //   Input(6) -> FC12+ReLU(k=1) -> FC12+ReLU(k=2) -> FC12+ReLU(k=3)
  //            -> FC12+ReLU(k=4) -> FC2 -> SoftMax
  //
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));            // 0
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));       // 1  <- steerable
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));       // 2  <- steerable
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));       // 3  <- steerable
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));       // 4  <- steerable
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));    // 5
    NN.AddLayer(TNNetSoftMax.Create());                      // 6
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // Two-cluster task: the class is decided ONLY by sign(x0).
  //   class 0: x0 < 0    class 1: x0 > 0
  // All other coordinates are pure noise distractors.
  procedure MakeSample(X: TNNetVolume; Cls: integer);
  var
    I: integer;
    Sign: integer;
  begin
    for I := 0 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 2.0 * cNoise;
    if Cls = 0 then Sign := -1 else Sign := 1;
    X.Raw[0] := Sign * cBlob + (Random - 0.5) * 2.0 * cNoise;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B, Cls, Correct: integer;
    X, Y: TNNetVolume;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    try
      for Ep := 1 to Epochs do
      begin
        Correct := 0;
        for B := 1 to 96 do
        begin
          Cls := Random(2);
          MakeSample(X, Cls);
          Y.Fill(0);
          Y.Raw[Cls] := 1.0;
          NN.Compute(X);
          if NN.GetLastLayer.Output.GetClass() = Cls then Inc(Correct);
          NN.Backpropagate(Y);
        end;
        if (Ep = 1) or (Ep mod 40 = 0) or (Ep = Epochs) then
          WriteLn(Format('  epoch %3d  train-acc=%.3f', [Ep, Correct / 96.0]));
      end;
    finally
      X.Free;
      Y.Free;
    end;
  end;

  // v = mean(act_k | class 1) - mean(act_k | class 0) over fresh training draws.
  procedure ComputeSteeringVector(NN: TNNet; K: integer; V: TNNetVolume);
  var
    X, Mean0, Mean1: TNNetVolume;
    I, Cls: integer;
  begin
    X := TNNetVolume.Create(cInDim, 1, 1);
    Mean0 := TNNetVolume.Create();
    Mean1 := TNNetVolume.Create();
    try
      Mean0.Copy(NN.Layers[K].Output);  Mean0.Fill(0);
      Mean1.Copy(NN.Layers[K].Output);  Mean1.Fill(0);
      for Cls := 0 to 1 do
        for I := 1 to cMeanN do
        begin
          MakeSample(X, Cls);
          NN.Compute(X);
          if Cls = 0 then
            Mean0.Add(NN.Layers[K].Output)
          else
            Mean1.Add(NN.Layers[K].Output);
        end;
      Mean0.Mul(1.0 / cMeanN);
      Mean1.Mul(1.0 / cMeanN);
      V.Copy(Mean1);
      V.Sub(Mean0);
    finally
      X.Free;
      Mean0.Free;
      Mean1.Free;
    end;
  end;

  // Forward pass with layer K's activation shifted by Direction*alpha.
  // Mirrors the ActivationPatchingReport / ActivationSteering recompute: a full
  // forward pass, then overwrite layer K's Output and recompute layers
  // K+1..last via Compute(). CleanK is the unsteered activation at K,
  // snapshotted once by the caller. Returns the SoftMax probability of class
  // TargetCls, and (via OutCls) the predicted argmax after steering.
  function SteeredProb(NN: TNNet; Input: TNNetVolume; K: integer;
    CleanK, Direction: TNNetVolume; Alpha: TNeuralFloat;
    TargetCls: integer; out OutCls: integer): TNeuralFloat;
  var
    I, LastLayer: integer;
  begin
    LastLayer := NN.GetLastLayerIdx();
    // Full clean forward pass (fills every layer's Output, K included).
    NN.Compute(Input);
    // Restore K to the snapshot, then inject the direction. CopyNoChecks: same
    // shape by construction (CleanK was copied from this very layer).
    NN.Layers[K].Output.CopyNoChecks(CleanK);
    NN.Layers[K].Output.MulAdd(Alpha, Direction);
    // Recompute every downstream layer from the steered activation.
    for I := K + 1 to LastLayer do
      NN.Layers[I].Compute();
    OutCls := NN.GetLastLayer.Output.GetClass();
    Result := NN.GetLastLayer.Output.Raw[TargetCls];
  end;

var
  NN: TNNet;
  Input, V, CleanK: TNNetVolume;
  TargetCls, I, K, BarLen, MaxBar, PlainCls, StepCls: integer;
  Plain, P0, VNorm: TNeuralFloat;
  Alphas: array[0..6] of TNeuralFloat = (-3, -2, -1, 0, 1, 2, 3);
  ProbV: array[0..6] of TNeuralFloat;
  PassZero, AllPassZero: boolean;
  UpSteps: integer;
  MonoFrac, FlipAlpha: TNeuralFloat;
  BestMonoK, BestFlipK: integer;
  BestMono, BestFlip: TNeuralFloat;
  HasFlip: boolean;
begin
  RandSeed := 2026;
  Input  := TNNetVolume.Create(cInDim, 1, 1);
  V      := TNNetVolume.Create();
  CleanK := TNNetVolume.Create();
  try
    WriteLn('ActivationSteeringDepthSweep: sweep the steering layer k across ' +
      'ALL hidden layers and ask WHERE a concept vector bites hardest.');
    WriteLn('Depth-sweep follow-up to examples/ActivationSteering (ActAdd / ' +
      'activation addition, Turner et al. 2023). Diff-of-class-means direction ' +
      'v_k per layer; no extra training.');
    WriteLn;

    BuildNet(NN, 0.01);
    WriteLn('Training for ', cEpochs, ' epochs on a sign(x0) two-cluster task...');
    TrainOnce(NN, cEpochs);
    WriteLn;

    // A FIXED probe input from the boundary-ish region so steering can swing the
    // decision either way. We steer TOWARD class 1 from a class-0 sample.
    MakeSample(Input, 0);             // a class-0 sample (x0 < 0)
    NN.Compute(Input);
    TargetCls := 1;
    Plain := NN.GetLastLayer.Output.Raw[TargetCls];
    PlainCls := NN.GetLastLayer.Output.GetClass();
    WriteLn(Format('Probe input: plain forward P(class %d)=%.6f (argmax=%d). ' +
      'Steering TOWARD class %d.', [TargetCls, Plain, PlainCls, TargetCls]));
    WriteLn;

    MaxBar := 48;
    AllPassZero := True;
    BestMonoK := -1;  BestMono := -1;
    BestFlipK := -1;  BestFlip := 1e30;

    // ---- sweep the steering layer k across all hidden layers ----
    for K := cFirstK to cLastK do
    begin
      // Steering vector v_k = mean(act_k|1) - mean(act_k|0) at layer k.
      V.Copy(NN.Layers[K].Output);   // shape v to match layer k
      ComputeSteeringVector(NN, K, V);
      VNorm := V.GetMagnitude();

      // Snapshot the unsteered activation at layer k for the fixed probe input.
      NN.Compute(Input);
      CleanK.Copy(NN.Layers[K].Output);

      // alpha sweep at this layer.
      for I := 0 to High(Alphas) do
        ProbV[I] := SteeredProb(NN, Input, K, CleanK, V, Alphas[I],
          TargetCls, StepCls);
      P0 := ProbV[3]; // alpha = 0 entry

      WriteLn(StringOfChar('=', 78));
      WriteLn(Format('LAYER k=%d (%s, size=%d, ||v_k||=%.4f): ' +
        'P(target) vs alpha', [K, NN.Layers[K].ClassName,
        NN.Layers[K].Output.Size, VNorm]));
      WriteLn(StringOfChar('=', 78));
      WriteLn('  alpha   P(v)      argmax | P(v) bar (0..1)');

      // Recompute per-row argmax for the table (cheap; same machinery).
      for I := 0 to High(Alphas) do
      begin
        ProbV[I] := SteeredProb(NN, Input, K, CleanK, V, Alphas[I],
          TargetCls, StepCls);
        BarLen := Round(ProbV[I] * MaxBar);
        if BarLen < 0 then BarLen := 0;
        if BarLen > MaxBar then BarLen := MaxBar;
        WriteLn(Format('%8.4f  %8.6f   %4d  | %s',
          [Alphas[I], ProbV[I], StepCls, StringOfChar('#', BarLen)]));
      end;

      // ---- correctness check: alpha=0 reproduces plain forward bit-for-bit ----
      PassZero := (P0 = Plain);
      if not PassZero then AllPassZero := False;

      // ---- monotonicity: fraction of adjacent-alpha steps that increase ----
      UpSteps := 0;
      for I := 1 to High(Alphas) do
        if ProbV[I] >= ProbV[I - 1] - 1e-9 then Inc(UpSteps);
      MonoFrac := UpSteps / High(Alphas);

      // ---- alpha-to-flip: smallest |alpha| at which argmax leaves PlainCls ----
      // Scan alphas in order of increasing magnitude.
      HasFlip := False;
      FlipAlpha := 0;
      for I := 0 to High(Alphas) do
      begin
        SteeredProb(NN, Input, K, CleanK, V, Alphas[I], TargetCls, StepCls);
        if (StepCls <> PlainCls) and
           ((not HasFlip) or (Abs(Alphas[I]) < Abs(FlipAlpha))) then
        begin
          HasFlip := True;
          FlipAlpha := Alphas[I];
        end;
      end;

      if HasFlip then
        WriteLn(Format('  monotonicity(up-frac)=%.3f   alpha-to-flip=%8.4f   ' +
          'alpha=0 bit-for-bit: %s',
          [MonoFrac, FlipAlpha, BoolToStr(PassZero, 'PASS', 'FAIL')]))
      else
        WriteLn(Format('  monotonicity(up-frac)=%.3f   alpha-to-flip=none in ' +
          'range   alpha=0 bit-for-bit: %s',
          [MonoFrac, BoolToStr(PassZero, 'PASS', 'FAIL')]));
      WriteLn;

      // Track winners. Cleanest monotone control = highest up-fraction (ties
      // broken by larger absolute swing toward 1.0). Smallest alpha-to-flip =
      // smallest |flip alpha| among layers that DO flip.
      if MonoFrac > BestMono + 1e-9 then
      begin
        BestMono := MonoFrac;
        BestMonoK := K;
      end;
      if HasFlip and (Abs(FlipAlpha) < BestFlip - 1e-9) then
      begin
        BestFlip := Abs(FlipAlpha);
        BestFlipK := K;
      end;
    end;

    // ---- summary ----
    WriteLn(StringOfChar('=', 78));
    WriteLn('SUMMARY');
    WriteLn(StringOfChar('=', 78));
    if BestMonoK >= 0 then
      WriteLn(Format('Cleanest monotone P(target)-vs-alpha control: layer k=%d ' +
        '(up-fraction=%.3f).', [BestMonoK, BestMono]))
    else
      WriteLn('No layer produced a monotone curve.');
    if BestFlipK >= 0 then
      WriteLn(Format('Smallest alpha-to-flip (concept bites hardest): layer ' +
        'k=%d (|alpha|=%.4f).', [BestFlipK, BestFlip]))
    else
      WriteLn('No layer flipped the prediction anywhere in the alpha range.');
    WriteLn(Format('CHECK (alpha=0 reproduces plain forward BIT-FOR-BIT at ' +
      'EVERY layer k): %s', [BoolToStr(AllPassZero, 'PASS', 'FAIL')]));
    WriteLn;
    if AllPassZero then
      WriteLn('Read it as: the same diff-of-means concept direction is a CAUSAL ' +
        'knob at every depth, but it does not bite equally hard everywhere — ' +
        'sweeping k reveals which layer gives the smoothest, lowest-alpha ' +
        'control. alpha=0 is a bit-for-bit no-op at every k; weights are never ' +
        'stepped (the only mutation is the transient activation shift, reverted ' +
        'by the next clean forward pass).')
    else
      WriteLn('CHECK FAILED: alpha=0 was not a bit-for-bit no-op at some layer.');
  finally
    NN.Free;
    Input.Free;
    V.Free;
    CleanK.Free;
  end;
end.
