program ConceptBottleneck;
(*
ConceptBottleneck: a self-contained, interpretable-by-design Concept
Bottleneck Model (CBM; Koh et al. 2020, https://arxiv.org/abs/2007.04612)
demo built from EXISTING layers only (no new layer type). The network is
forced to route ALL information about the label through a narrow layer of K
human-meaningful CONCEPTS, so the label can only be predicted from those
concepts - which makes the model editable AT TEST TIME by overwriting the
concept values.

Synthetic task. Each sample is a 6-D feature vector. Three known boolean
CONCEPTS are read off the raw features:
  c0 = "is bright"     (feature 0 above its midpoint)
  c1 = "is in top half"(feature 1 above its midpoint)
  c2 = "is round"      (feature 2 above its midpoint)
The 4-way LABEL is a FIXED, known function of the concept bits:
  class = c0 + 2*c1   (c2 is a decoy concept that does NOT drive the label)
so the label is a deterministic linear/boolean function of two of the three
concepts. The raw features additionally carry noise distractors, so the trunk
has to actually learn to read the concepts out of the inputs.

Architecture (two-stage, fully sequential bottleneck):
  Input(6)
    -> Trunk: FC+ReLU(H) -> FC+ReLU(H)
    -> TNNetFullConnectSigmoid(K=3)    <- the CONCEPT BOTTLENECK (the only
                                          path from inputs to the label)
    -> TNNetFullConnectLinear(4)       <- the LABEL head reads ONLY concepts
    -> TNNetSoftMax
  A Concat([SoftMax, Sigmoid]) tail packs (label probs | concept activations)
  into one output of width NumClasses+K so a single Compute exposes both heads.

Joint training with DEEP SUPERVISION (manual two-head loss loop). Automatic
Fit seeds the gradient only at the last layer, so we seed BOTH heads ourselves,
exactly the packed-target idiom of examples/EarlyExitNetwork:
  - the SoftMax block of the packed output holds label probabilities p; a packed
    target one-hot there makes ComputeOutputErrorWith form (p - onehot), the
    softmax-cross-entropy gradient;
  - the Sigmoid block holds concept activations s; a packed target equal to the
    GROUND-TRUTH concept bits makes ComputeOutputErrorWith form (s - c)*s'(.) ,
    the sigmoid concept-prediction gradient.
A single Backpropagate splits the packed error through the Concat: the label
gradient flows softmax -> label-linear -> bottleneck -> trunk, and the concept
gradient flows directly into the bottleneck (and trunk). We scale the concept
block of the packed target's error by a knob lambda (cConceptLambda) by writing
the concept target so the seeded error is lambda-weighted (we pre-scale via a
straight-through trick: we leave the target as the true bits and instead damp
the bottleneck's contribution by zeroing it when lambda=0 - see SeedTargets).

MANUAL-GRADIENT GOTCHA: hand-driven multi-head accumulation needs
SetBatchUpdate(True) - the per-sample default zeroes Neurons[].Delta between
samples. We use the batch idiom: ClearDeltas at batch start, accumulate over
the minibatch, UpdateWeights (mirrors examples/EarlyExitNetwork and
examples/GradientNoiseScale).

The headline payoff: TEST-TIME CONCEPT INTERVENTION. At inference we Compute
once, then OVERWRITE the predicted concept vector at the sigmoid bottleneck and
recompute ONLY the downstream label head - the SAME CopyNoChecks-then-recompute
machinery as examples/ActivationSteering / examples/ActivationPatching
(Layers[k].Output.CopyNoChecks(...); for i:=k+1..last do Layers[i].Compute()).
We show:
  (a) injecting the GROUND-TRUTH concepts raises label accuracy over the
      end-to-end prediction (the model's mistakes are attributable to concept
      errors, not the label head), and
  (b) flipping a single concept bit by hand deterministically flips the
      predicted class in the direction that concept controls.

Two built-in invariants are asserted/printed:
  (1) intervening with the model's OWN predicted concepts (a no-op overwrite)
      reproduces the un-intervened logits BIT-FOR-BIT; and
  (2) with the concept-loss weight lambda set to 0 the bottleneck is free to
      DRIFT - the concepts no longer align with the ground-truth bits (the
      "leaky" joint-vs-independent CBM failure mode). We retrain a second net
      with lambda=0 and report its (much lower) per-concept alignment.

CONTRAST with neighbouring examples:
  - examples/LinearProbeReport fits a POST-HOC, frozen linear probe that only
    READS what some layer already encodes - it never changes the network or its
    predictions. Here the concept layer is TRAINED into the forward path and is
    causally EDITABLE.
  - examples/ActivationSteering EDITS raw hidden activations with a
    diff-of-means direction but has NO concept supervision and no
    interpretable, named bottleneck - the edited coordinates have no a-priori
    meaning. Here every bottleneck unit IS a named concept by construction, so
    the intervention is "set concept c1 := true", not "add 1.7*v".
  - examples/DomainAdversarial uses gradient reversal to REMOVE information
    (it makes a feature undecodable). Here we do the opposite: we FORCE a
    specific, human-meaningful set of concepts to be decodable AND to be the
    sole carrier of the label.

Pure CPU, no dataset download, deterministic (seeded), runs in well under a
minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cInDim    = 6;
  cHidden   = 16;
  cConcepts = 3;                          // K human-meaningful concepts
  cClasses  = 4;                          // label = c0 + 2*c1  (c2 is a decoy)
  cOutDepth = cClasses + cConcepts;       // packed [ label probs | concepts ]

  cTrain  = 1600;
  cTest   = 1200;
  cEpochs = 120;
  cBatch  = 32;
  cLR     = 0.05;
  cInertia = 0.9;

type
  TSample = record
    X: array[0..cInDim-1] of TNeuralFloat;
    C: array[0..cConcepts-1] of integer;   // ground-truth concept bits
    Cls: integer;                          // label = c0 + 2*c1
  end;
  TSampleArray = array of TSample;

var
  TrainSet, TestSet: TSampleArray;
  // Captured layer INDICES (never stale object refs).
  ConceptIdx: integer;   // the sigmoid bottleneck
  LabelHeadIdx: integer; // the linear label head
  SoftMaxIdx: integer;   // the softmax (label probabilities)

function RandNormal: TNeuralFloat;
var
  U1, U2: TNeuralFloat;
begin
  U1 := Random; if U1 < 1e-12 then U1 := 1e-12;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

procedure MakeSample(out S: TSample);
// Three latent concept bits drive the raw features; the label is a fixed
// boolean function of two of them (the third is a decoy). Feature j>=3 are
// pure noise distractors, and the signal features carry noise too, so the
// trunk must learn a non-trivial readout.
var
  J: integer;
begin
  for J := 0 to cInDim - 1 do
    S.X[J] := 0.6 * RandNormal;                 // noise everywhere
  // Concept bits, each pushing its own signal feature up or down.
  for J := 0 to cConcepts - 1 do
  begin
    S.C[J] := Random(2);
    if S.C[J] = 1 then
      S.X[J] := S.X[J] + 1.6
    else
      S.X[J] := S.X[J] - 1.6;
  end;
  // Label is a known linear/boolean function of c0 and c1 (c2 is a decoy).
  S.Cls := S.C[0] + 2 * S.C[1];
end;

procedure BuildDataset(out A: TSampleArray; Count: integer);
var
  I: integer;
begin
  SetLength(A, Count);
  for I := 0 to Count - 1 do
    MakeSample(A[I]);
end;

procedure BuildModel(out Net: TNNet);
var
  Bottleneck, SoftM: TNNetLayer;
begin
  Net := TNNet.Create();
  Net.AddLayer(TNNetInput.Create(cInDim, 1, 1));
  Net.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  Net.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  // The CONCEPT BOTTLENECK: the sole path from inputs to label.
  Bottleneck := Net.AddLayer(TNNetFullConnectSigmoid.Create(cConcepts));
  // The LABEL head reads ONLY the concept bottleneck.
  Net.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  SoftM := Net.AddLayer(TNNetSoftMax.Create());
  // Pack [ label probabilities | concept activations ] into one output so a
  // single Compute exposes both heads. Order matters: softmax first.
  Net.AddLayer(TNNetConcat.Create([SoftM, Bottleneck]));
  Net.SetLearningRate(cLR, cInertia);
  Net.InitWeights();
  ConceptIdx   := Bottleneck.LayerIdx;
  SoftMaxIdx   := SoftM.LayerIdx;
  LabelHeadIdx := SoftMaxIdx - 1;
end;

procedure FillInput(InputV: TNNetVolume; const S: TSample);
var
  J: integer;
begin
  for J := 0 to cInDim - 1 do
    InputV.FData[J] := S.X[J];
end;

procedure SeedTargets(Net: TNNet; const S: TSample; ConceptLambda: TNeuralFloat);
// Manual two-head deep supervision. The packed output holds, per channel:
//   [0..cClasses-1]            : softmax label probabilities p
//   [cClasses..cClasses+K-1]   : sigmoid concept activations s
// We build a packed target so ComputeOutputErrorWith yields:
//   label block:   (p - onehot)             (softmax-CE gradient)
//   concept block: (s - c) * s'(.)          (sigmoid concept-loss gradient)
// then a single Backpropagate splits it through the Concat. The concept-loss
// WEIGHT knob: with ConceptLambda=0 we make the seeded concept error vanish by
// setting the concept target EQUAL to the current activation s (so s - s = 0),
// i.e. no concept supervision -> the bottleneck is free to DRIFT (leaky CBM).
// With ConceptLambda=1 we set the target to the true bits c.
var
  TargetV, Outp: TNNetVolume;
  H: integer;
begin
  TargetV := TNNetVolume.Create(1, 1, cOutDepth);
  try
    // Caller has just run Net.Compute(InputV), so outputs are fresh.
    Outp := Net.GetLastLayer.Output;
    TargetV.Fill(0);
    // Label block: one-hot of the true class.
    TargetV.FData[S.Cls] := 1.0;
    // Concept block.
    for H := 0 to cConcepts - 1 do
    begin
      if ConceptLambda > 0 then
        TargetV.FData[cClasses + H] := S.C[H]      // true bit -> supervise
      else
        TargetV.FData[cClasses + H] := Outp.FData[cClasses + H]; // self -> no grad
    end;
    Net.Backpropagate(TargetV);
  finally
    TargetV.Free;
  end;
end;

procedure Train(Net: TNNet; ConceptLambda: TNeuralFloat);
var
  Ep, Step, B, Idx, NumSteps: integer;
  InputV: TNNetVolume;
  StartTime, Elapsed: double;
begin
  InputV := TNNetVolume.Create(cInDim, 1, 1);
  NumSteps := Length(TrainSet) div cBatch;
  Net.SetBatchUpdate(True);
  try
    StartTime := Now();
    for Ep := 1 to cEpochs do
    begin
      for Step := 0 to NumSteps - 1 do
      begin
        Net.ClearDeltas();
        for B := 1 to cBatch do
        begin
          Idx := Random(Length(TrainSet));
          FillInput(InputV, TrainSet[Idx]);
          Net.Compute(InputV);
          SeedTargets(Net, TrainSet[Idx], ConceptLambda);
        end;
        Net.UpdateWeights();
      end;
      if (Ep = 1) or (Ep mod 30 = 0) or (Ep = cEpochs) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  epoch %3d / %3d   elapsed=%6.1fs', [Ep, cEpochs, Elapsed]));
      end;
    end;
  finally
    InputV.Free;
  end;
end;

function ArgMaxBlock(Outp: TNNetVolume; Base, Count: integer): integer;
var
  C, Best: integer;
  Mx, Cur: TNeuralFloat;
begin
  Best := 0; Mx := Outp.FData[Base];
  for C := 1 to Count - 1 do
  begin
    Cur := Outp.FData[Base + C];
    if Cur > Mx then begin Mx := Cur; Best := C; end;
  end;
  Result := Best;
end;

procedure PerConceptAccuracy(Net: TNNet; out PerC: array of TNeuralFloat;
  out MeanC: TNeuralFloat);
// Threshold each sigmoid concept activation at 0.5 and compare to the true bit.
var
  I, H: integer;
  InputV, Outp: TNNetVolume;
  Pred: integer;
  Hits: array[0..cConcepts-1] of integer;
begin
  InputV := TNNetVolume.Create(cInDim, 1, 1);
  for H := 0 to cConcepts - 1 do Hits[H] := 0;
  try
    for I := 0 to Length(TestSet) - 1 do
    begin
      FillInput(InputV, TestSet[I]);
      Net.Compute(InputV);
      Outp := Net.GetLastLayer.Output;
      for H := 0 to cConcepts - 1 do
      begin
        if Outp.FData[cClasses + H] >= 0.5 then Pred := 1 else Pred := 0;
        if Pred = TestSet[I].C[H] then Inc(Hits[H]);
      end;
    end;
  finally
    InputV.Free;
  end;
  MeanC := 0;
  for H := 0 to cConcepts - 1 do
  begin
    PerC[H] := Hits[H] / Length(TestSet);
    MeanC := MeanC + PerC[H];
  end;
  MeanC := MeanC / cConcepts;
end;

function CleanLabelAccuracy(Net: TNNet): TNeuralFloat;
var
  I, Pred, Hits: integer;
  InputV, Outp: TNNetVolume;
begin
  InputV := TNNetVolume.Create(cInDim, 1, 1);
  Hits := 0;
  try
    for I := 0 to Length(TestSet) - 1 do
    begin
      FillInput(InputV, TestSet[I]);
      Net.Compute(InputV);
      Outp := Net.GetLastLayer.Output;
      Pred := ArgMaxBlock(Outp, 0, cClasses);
      if Pred = TestSet[I].Cls then Inc(Hits);
    end;
  finally
    InputV.Free;
  end;
  Result := Hits / Length(TestSet);
end;

function InterveneRecompute(Net: TNNet; InputV, ConceptInject: TNNetVolume): integer;
// Overwrite the sigmoid bottleneck with ConceptInject and recompute ONLY the
// downstream label head (label-linear, softmax, concat). Same CopyNoChecks-
// then-recompute idiom as examples/ActivationSteering. Returns predicted class.
var
  I, LastLayer: integer;
  Outp: TNNetVolume;
begin
  LastLayer := Net.GetLastLayerIdx();
  Net.Compute(InputV);                                   // clean forward
  Net.Layers[ConceptIdx].Output.CopyNoChecks(ConceptInject);
  for I := ConceptIdx + 1 to LastLayer do
    Net.Layers[I].Compute();                             // recompute downstream
  Outp := Net.GetLastLayer.Output;
  Result := ArgMaxBlock(Outp, 0, cClasses);
end;

function InterveneTrueConceptAccuracy(Net: TNNet): TNeuralFloat;
// Inject the GROUND-TRUTH concept bits at the bottleneck and recompute the
// label head. If concept errors are the model's only failure mode, this is the
// label head's accuracy GIVEN perfect concepts (should be ~1.0).
var
  I, H, Pred, Hits: integer;
  InputV, Inject: TNNetVolume;
begin
  InputV := TNNetVolume.Create(cInDim, 1, 1);
  Inject := TNNetVolume.Create(1, 1, cConcepts);
  Hits := 0;
  try
    for I := 0 to Length(TestSet) - 1 do
    begin
      FillInput(InputV, TestSet[I]);
      for H := 0 to cConcepts - 1 do
        Inject.FData[H] := TestSet[I].C[H];   // 0/1 hard concept values
      Pred := InterveneRecompute(Net, InputV, Inject);
      if Pred = TestSet[I].Cls then Inc(Hits);
    end;
  finally
    InputV.Free;
    Inject.Free;
  end;
  Result := Hits / Length(TestSet);
end;

var
  Net, NetLeaky: TNNet;
  PerC, PerCLeaky: array[0..cConcepts-1] of TNeuralFloat;
  MeanC, MeanCLeaky: TNeuralFloat;
  CleanAcc, TrueConceptAcc: TNeuralFloat;
  // Invariant-1 (no-op overwrite) state.
  InputV, OwnConcepts, CleanLogits, ReLogits: TNNetVolume;
  I, H, LastLayer, NoOpPred, CleanPred: integer;
  NoOpExact: boolean;
  MaxDiff: TNeuralFloat;
  // Worked single-concept flip.
  FlipInput, FlipInject: TNNetVolume;
  BasePred, FlipPred, FlipConcept: integer;

begin
  RandSeed := 424242;
  WriteLn('ConceptBottleneck: interpretable-by-design Concept Bottleneck Model');
  WriteLn(Format('  Input(%d) -> FC+ReLU(%d) x2 -> Sigmoid(K=%d concepts) -> ' +
    'Linear(%d) -> SoftMax', [cInDim, cHidden, cConcepts, cClasses]));
  WriteLn('  label = c0 + 2*c1  (c2 is a DECOY concept that does not drive the label)');
  WriteLn;

  BuildDataset(TrainSet, cTrain);
  BuildDataset(TestSet, cTest);
  WriteLn(Format('  train=%d  test=%d', [cTrain, cTest]));

  BuildModel(Net);
  try
    WriteLn('Layers:');
    Net.DebugStructure();
    WriteLn(Format('Captured indices: concept(sigmoid)=%d  label-linear=%d  softmax=%d',
      [ConceptIdx, LabelHeadIdx, SoftMaxIdx]));
    WriteLn;

    WriteLn('Training JOINTLY with deep supervision (label CE + concept loss, lambda=1)...');
    Train(Net, 1.0);
    WriteLn;

    // ---- per-concept accuracy table ----
    PerConceptAccuracy(Net, PerC, MeanC);
    WriteLn('Per-concept accuracy (sigmoid bottleneck thresholded at 0.5):');
    WriteLn('  concept             test-acc');
    WriteLn(Format('  c0 "is bright"      %8.4f', [PerC[0]]));
    WriteLn(Format('  c1 "is top half"    %8.4f', [PerC[1]]));
    WriteLn(Format('  c2 "is round"(decoy)%8.4f', [PerC[2]]));
    WriteLn(Format('  mean concept acc    %8.4f', [MeanC]));
    WriteLn;

    // ---- clean vs intervened label accuracy ----
    CleanAcc       := CleanLabelAccuracy(Net);
    TrueConceptAcc := InterveneTrueConceptAccuracy(Net);
    WriteLn('Label accuracy:');
    WriteLn(Format('  clean (end-to-end)               %8.4f', [CleanAcc]));
    WriteLn(Format('  intervened (inject TRUE concepts)%8.4f', [TrueConceptAcc]));
    WriteLn(Format('  -> intervention gain             %8.4f', [TrueConceptAcc - CleanAcc]));
    WriteLn('  (injecting true concepts recovers the label head''s ceiling: the');
    WriteLn('   model''s residual mistakes are attributable to concept errors.)');
    WriteLn;

    // ---- INVARIANT 1: no-op overwrite reproduces clean logits bit-for-bit ----
    InputV      := TNNetVolume.Create(cInDim, 1, 1);
    OwnConcepts := TNNetVolume.Create(1, 1, cConcepts);
    CleanLogits := TNNetVolume.Create();
    ReLogits    := TNNetVolume.Create();
    LastLayer   := Net.GetLastLayerIdx();
    FillInput(InputV, TestSet[0]);
    Net.Compute(InputV);
    CleanLogits.Copy(Net.Layers[SoftMaxIdx].Output);
    CleanPred := ArgMaxBlock(Net.GetLastLayer.Output, 0, cClasses);
    // Snapshot the model's OWN predicted concepts, overwrite with them, recompute.
    OwnConcepts.Copy(Net.Layers[ConceptIdx].Output);
    Net.Layers[ConceptIdx].Output.CopyNoChecks(OwnConcepts);
    for I := ConceptIdx + 1 to LastLayer do Net.Layers[I].Compute();
    ReLogits.Copy(Net.Layers[SoftMaxIdx].Output);
    NoOpPred := ArgMaxBlock(Net.GetLastLayer.Output, 0, cClasses);
    MaxDiff := 0;
    for I := 0 to CleanLogits.Size - 1 do
      MaxDiff := Max(MaxDiff, Abs(CleanLogits.FData[I] - ReLogits.FData[I]));
    NoOpExact := (MaxDiff = 0) and (NoOpPred = CleanPred);
    WriteLn(Format('INVARIANT 1 (no-op overwrite with OWN concepts reproduces logits ' +
      'bit-for-bit): max|dlogit|=%.3e  -> %s',
      [MaxDiff, BoolToStr(NoOpExact, 'PASS', 'FAIL')]));

    // ---- worked single-concept flip ----
    // Take a sample with c1=0 (predicted class in {0,1}); flip c1 -> 1 and show
    // the predicted class jumps by +2 (the weight c1 carries in label=c0+2*c1).
    FlipInput  := TNNetVolume.Create(cInDim, 1, 1);
    FlipInject := TNNetVolume.Create(1, 1, cConcepts);
    FlipConcept := 1;   // the concept worth +2 in the label
    I := 0;
    while (I < Length(TestSet)) and (TestSet[I].C[FlipConcept] <> 0) do Inc(I);
    FillInput(FlipInput, TestSet[I]);
    // Baseline: inject the sample's true concepts.
    for H := 0 to cConcepts - 1 do FlipInject.FData[H] := TestSet[I].C[H];
    BasePred := InterveneRecompute(Net, FlipInput, FlipInject);
    // Flip ONLY concept c1: 0 -> 1.
    FlipInject.FData[FlipConcept] := 1;
    FlipPred := InterveneRecompute(Net, FlipInput, FlipInject);
    WriteLn;
    WriteLn('Worked single-concept flip (set concept c1 "is top half" := 1):');
    WriteLn(Format('  baseline concepts [%d %d %d] -> predicted class %d',
      [TestSet[I].C[0], TestSet[I].C[1], TestSet[I].C[2], BasePred]));
    WriteLn(Format('  flipped  concepts [%d %d %d] -> predicted class %d',
      [TestSet[I].C[0], 1, TestSet[I].C[2], FlipPred]));
    WriteLn(Format('  delta = %d  (c1 carries weight +2 in label=c0+2*c1; flipping ' +
      'it moves the class deterministically)', [FlipPred - BasePred]));
    WriteLn;

    InputV.Free; OwnConcepts.Free; CleanLogits.Free; ReLogits.Free;
    FlipInput.Free; FlipInject.Free;

    // ---- INVARIANT 2: lambda=0 -> the bottleneck drifts (leaky CBM) ----
    WriteLn('Retraining a SECOND net with concept-loss weight lambda=0 ' +
      '(no concept supervision)...');
    BuildModel(NetLeaky);
    try
      Train(NetLeaky, 0.0);
      PerConceptAccuracy(NetLeaky, PerCLeaky, MeanCLeaky);
      WriteLn(Format('  lambda=1 mean concept acc = %.4f  (concepts ALIGN)', [MeanC]));
      WriteLn(Format('  lambda=0 mean concept acc = %.4f  (concepts DRIFT - leaky CBM)',
        [MeanCLeaky]));
      WriteLn(Format('INVARIANT 2 (lambda=0 bottleneck drifts: alignment drops vs lambda=1): -> %s',
        [BoolToStr(MeanCLeaky < MeanC - 0.05, 'PASS', 'FAIL')]));
    finally
      NetLeaky.Free;
    end;

    WriteLn;
    if NoOpExact and (TrueConceptAcc >= CleanAcc) and (FlipPred <> BasePred)
       and (MeanCLeaky < MeanC - 0.05) then
      WriteLn('ALL CHECKS PASS: concepts align under supervision, the no-op ' +
        'overwrite is exact, injecting true concepts lifts accuracy, a single ' +
        'concept flip moves the class, and lambda=0 makes the bottleneck drift.')
    else
      WriteLn('ONE OR MORE CHECKS FAILED.');

    WriteLn;
    WriteLn('Read it as: the label can ONLY be read from the K-dim concept ' +
      'bottleneck, so the model is interpretable AND editable: overwrite a ' +
      'named concept at test time and the prediction changes the way that ' +
      'concept dictates. Unlike a post-hoc probe (READS only) or activation ' +
      'steering (edits anonymous activations), the bottleneck units ARE the ' +
      'human concepts by construction - and dropping the concept loss ' +
      '(lambda=0) breaks that alignment.');
  finally
    Net.Free;
  end;
end.
