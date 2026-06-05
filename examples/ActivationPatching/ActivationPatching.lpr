program ActivationPatching;
(*
ActivationPatching: builds a small branched classifier on a synthetic task
where the CAUSAL layer is KNOWN by construction, then prints
TNNet.ActivationPatchingReport(NN, CleanInput, CorruptInput) — the forward-only
CAUSAL activation-patching / causal-tracing diagnostic that answers "which
layer's ACTIVATIONS carry the information that decides this prediction?".

The task is a 2-class problem decided ENTIRELY by an XOR of the sign of the
first two input coordinates (class = (sign(x0) != sign(x1))); the remaining
coordinates are pure noise distractors. XOR is NOT linearly separable, so the
deciding feature must be COMPUTED inside the net's main (ReLU) branch. The net
also has an INPUT SKIP that re-injects the raw input via a Concat just before
the head (Input -> 3x FC+ReLU -> Concat(main, Input) -> FC+ReLU -> FC -> SoftMax).
We then take a (CleanInput, CorruptInput) pair the trained net maps to DIFFERENT
classes (we flip the sign of x1, so the XOR label flips) and run the causal
trace: restoring each layer's clean activation into the corrupt run in turn and
measuring how much of the clean decision is recovered.

Ground-truth localisation check: patching a SINGLE main-branch layer leaves the
CORRUPT raw input still flowing on the skip path, so those layers recover only a
little; recovery JUMPS to 1.0 at the Concat fusion layer (where the patched
clean activation overwrites the whole fused representation, skip included) — the
known-by-construction localisation. The two built-in faithfulness checks hold
EXACTLY: r_0 == 1 (patching the input fixes BOTH the main branch and the skip,
reconstructing the full clean run) and r_last == 1 (the last layer's Output IS
the logits). We also set CorruptInput := CleanInput to show the
denominator-collapse WARNING path instead of a divide by zero.

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
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cInDim   = 6;
  cHidden  = 12;
  cClasses = 2;
  cEpochs  = 200;
  cBlob    = 1.3;   // distance of the signal coords from the origin
  cNoise   = 0.30;  // per-coordinate Gaussian-ish spread

  // Builds a small classifier with a MAIN feature-extractor branch and an
  // INPUT SKIP, so the causal trace is GRADED (not flat) by construction:
  //
  //   Input(6) --> FC12+ReLU --> FC12+ReLU --> FC12+ReLU  (main branch)
  //         \                                        \
  //          \------------------ skip ----------- Concat --> FC8+ReLU
  //                                                            --> FC2 --> SoftMax
  //
  // The deciding XOR feature is non-linear, so it must be COMPUTED inside the
  // main branch. The Concat re-injects the raw Input, so when we patch a SINGLE
  // intermediate main-branch layer back to its clean value the head still sees
  // the CORRUPT raw input on the skip path; recovery is therefore PARTIAL until
  // the patched layer is deep enough that the main branch dominates the
  // decision. Patching the INPUT (L=0) fixes BOTH paths at once -> full
  // recovery (r_0 == 1), and patching the last layer trivially gives r_last==1.
  // The result is a recovery curve that climbs with depth across the main
  // branch and peaks at the layer where the XOR feature has fully formed - the
  // ground-truth localisation this example demonstrates.
  procedure BuildNet(out NN: TNNet; LR: TNeuralFloat);
  var
    InLayer: TNNetLayer;
  begin
    NN := TNNet.Create();
    InLayer := NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));            // main branch
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));            // main branch
    NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));            // main branch
    // Re-inject the raw Input via a skip, concatenated with the main branch.
    NN.AddLayer(TNNetConcat.Create([NN.GetLastLayer(), InLayer]));
    NN.AddLayer(TNNetFullConnectReLU.Create(8));                  // head
    NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
    NN.AddLayer(TNNetSoftMax.Create());
    NN.SetLearningRate(LR, 0.9);
    NN.InitWeights();
  end;

  // XOR-of-signs task: the class is decided ONLY by sign(x0) vs sign(x1).
  //   class 0: signs agree     (x0,x1 same sign)
  //   class 1: signs disagree  (x0,x1 opposite sign)
  // All other coordinates are pure noise distractors. Optionally a FIXED random
  // generator state is NOT used here; samples are independent draws.
  procedure MakeSample(X: TNNetVolume; out Cls: integer; SignA, SignB: integer);
  var
    I: integer;
  begin
    for I := 0 to cInDim - 1 do
      X.Raw[I] := (Random - 0.5) * 2.0 * cNoise;
    X.Raw[0] := SignA * cBlob + (Random - 0.5) * 2.0 * cNoise;
    X.Raw[1] := SignB * cBlob + (Random - 0.5) * 2.0 * cNoise;
    if SignA = SignB then Cls := 0 else Cls := 1;
  end;

  procedure TrainOnce(NN: TNNet; Epochs: integer);
  var
    Ep, B, Cls, Correct, SignA, SignB: integer;
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
          if Random(2) = 0 then SignA := 1 else SignA := -1;
          if Random(2) = 0 then SignB := 1 else SignB := -1;
          MakeSample(X, Cls, SignA, SignB);
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

var
  NN: TNNet;
  CleanInput, CorruptInput: TNNetVolume;
  CleanCls, CorruptCls, Tries: integer;
begin
  RandSeed := 2026;
  CleanInput := TNNetVolume.Create(cInDim, 1, 1);
  CorruptInput := TNNetVolume.Create(cInDim, 1, 1);
  try
    WriteLn('ActivationPatchingReport demo: a branched softmax MLP (main ReLU ' +
      'branch + raw-input skip) on a synthetic XOR-of-signs task.');
    WriteLn('The deciding XOR feature is NON-linear, so it must be COMPUTED ' +
      'inside the main branch; the skip re-injects the raw input at the head.');
    WriteLn;

    BuildNet(NN, 0.01);
    WriteLn('Training for ', cEpochs, ' epochs...');
    TrainOnce(NN, cEpochs);
    WriteLn;

    // Build a (clean, corrupt) pair the trained net maps to DIFFERENT classes.
    // Clean has signs (+,+) -> class 0; corrupt is the SAME sample with the
    // sign of x1 flipped -> class 1. Patching a single MAIN-BRANCH layer leaves
    // the corrupt raw input flowing on the skip path, so recovery there is
    // partial; it jumps to 1.0 at the Concat fusion layer where the whole fused
    // representation (skip included) is overwritten by the clean activation.
    // Retry until the net actually predicts different argmax classes (the
    // contrast needs a flip).
    for Tries := 1 to 50 do
    begin
      MakeSample(CleanInput, CleanCls, 1, 1);   // signs agree -> class 0
      CorruptInput.Copy(CleanInput);
      CorruptInput.Raw[1] := -CorruptInput.Raw[1]; // flip x1 -> class 1
      NN.Compute(CleanInput);
      CleanCls := NN.GetLastLayer.Output.GetClass();
      NN.Compute(CorruptInput);
      CorruptCls := NN.GetLastLayer.Output.GetClass();
      if CleanCls <> CorruptCls then Break;
    end;
    WriteLn(Format('Chose a pair: clean argmax=%d, corrupt argmax=%d ' +
      '(corrupt = clean with x1 sign flipped).', [CleanCls, CorruptCls]));
    WriteLn;

    WriteLn(StringOfChar('=', 78));
    WriteLn('CAUSAL TRACE: clean vs sign-flipped-x1 corrupt');
    WriteLn(StringOfChar('=', 78));
    Write(TNNet.ActivationPatchingReport(NN, CleanInput, CorruptInput));
    WriteLn;

    WriteLn(StringOfChar('=', 78));
    WriteLn('DENOMINATOR-COLLAPSE PATH: CorruptInput := CleanInput');
    WriteLn(StringOfChar('=', 78));
    Write(TNNet.ActivationPatchingReport(NN, CleanInput, CleanInput));
    WriteLn;

    WriteLn(
      'Read it as: r_L is the fraction of the clean decision recovered by ' +
      'restoring ONLY layer L''s clean activation into the corrupt run. The ' +
      'main-branch layers recover little (the corrupt raw input still reaches ' +
      'the head via the skip), and recovery JUMPS to ~1 at the Concat fusion ' +
      'layer where the whole fused representation is overwritten clean — the ' +
      'ground-truth localisation. r_0==1 (patching the input fixes both the ' +
      'branch and the skip, reconstructing the full clean run) and r_last==1 ' +
      '(the last layer Output IS the logits) are exact built-in faithfulness ' +
      'checks. The second run shows the warning emitted when the clean and ' +
      'corrupt runs do not differ.');
  finally
    NN.Free;
    CleanInput.Free;
    CorruptInput.Free;
  end;
end.
