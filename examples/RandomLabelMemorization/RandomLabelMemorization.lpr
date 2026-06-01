program RandomLabelMemorization;
(*
RandomLabelMemorization: reproduces the headline result of Zhang, Bengio,
Hardt, Recht & Vinyals, ICLR 2017, "Understanding deep learning requires
rethinking generalization", on a pure-CPU toy, using only existing in-tree
layers (no new layer is added).

The phenomenon (the whole point of this example):
  A sufficiently over-parameterised network can fit ANYTHING -- including a
  training set whose labels have been replaced by pure noise. Zhang et al.
  showed that the SAME network that learns a real task to ~100% train accuracy
  ALSO drives train accuracy to ~100% on the very same images with their labels
  RANDOMLY SHUFFLED. The fit to random labels is genuine memorisation: there is
  no signal left to learn, so the network simply stores the answers. The
  shocker is what this does to GENERALISATION: with true labels the network
  generalises (held-out test accuracy far above chance); with random labels it
  cannot possibly generalise (the held-out labels are independent of the inputs)
  so test accuracy sits at chance (~1/K). The training loss/accuracy is IDENTICAL
  in both regimes (both ~100%), yet one generalises and one does not.

  => Train error alone says NOTHING about generalisation. Capacity to fit the
     training data is not, by itself, evidence of having learned anything.

Task: a small fixed K-class classification problem.
  - K Gaussian blobs in a low-dimensional space form a clean, learnable target
    (true labels = which blob a point was drawn from).
  - The SAME fixed over-parameterised MLP
        Input -> FullConnectReLU(64) -> FullConnectReLU(64)
              -> FullConnectLinear(K) -> SoftMax
    is trained TWICE on the SAME inputs:
      (a) with the TRUE labels, and
      (b) with the labels RANDOMLY SHUFFLED across the training set (the inputs
          are untouched; only the label column is permuted, destroying all
          input->label signal).
  - A held-out TEST set (always TRUE labels) measures honest generalisation.

Both runs are trained with mini-batch SGD (SetBatchUpdate, accumulate the
gradient over a shuffled mini-batch, one UpdateWeights step per batch). The
random-label run is given more epochs because memorising pure noise is harder
than learning real structure, but it still reaches ~100% train accuracy -- that
is the capacity-to-memorise claim.

Built-in correctness gate (Halt(1) on failure), DoubleDescent/BitLinearBakeoff
house style:
  - random-label TRAIN accuracy >= 0.99           (it memorises pure noise)
  - random-label TEST  accuracy <= chance + margin (it cannot generalise)
  - true-label   TEST  accuracy >> chance          (it does generalise)
  - both TRAIN accuracies ~100%                     (same train error, different generalisation)

PART 2 -- label-corruption-fraction sweep: the binary true-vs-fully-shuffled
contrast above is just the two endpoints. Part 2 sweeps the label-corruption
fraction p across {0.0, 0.25, 0.5, 1.0} on the SAME inputs/net/weight-init seed
(corrupting a fraction p of the training labels by reassigning each chosen
sample to a uniformly random class; inputs untouched) and charts, per p:
  - epochs-to-fit-train (epochs until train accuracy >= 98%) -- RISES with p,
  - the test gap (train acc minus test acc)                  -- WIDENS with p.
This is the smooth interpolation between "real structure" (p=0, generalises)
and "pure memorisation" (p=1, no generalisation). Its gate (Halt(1) on failure)
asserts the qualitative trend: epochs-to-fit larger at p=1 than p=0, the test
gap at p=1 wider than at p=0 by a comfortable margin, and every level still
fits within budget.

This is DISTINCT from examples/DoubleDescent (which sweeps CAPACITY under a
fixed amount of label noise and looks at the non-monotone test-error curve).
Here capacity is FIXED and the contrast is TRUE labels vs RANDOM labels.

Pure CPU, no external data, deterministic seeding (RandSeed=424242,
MaxThreadNum via single-thread Compute/Backpropagate), finishes well under the
few-minute budget.

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
  cSeed     = 424242;
  cDim      = 20;     // input feature dimension (high-dim -> points separable)
  cClasses  = 5;      // K classes => chance accuracy = 1/K = 0.20
  cTrain    = 200;    // SMALL training set (so memorising random labels is cheap)
  cTest     = 1000;   // LARGE held-out test set (always TRUE labels)
  cHidden   = 64;     // hidden width -- heavily over-parameterised for cTrain
  cBatch    = 25;     // mini-batch size for SGD (cTrain/cBatch steps per epoch)
  cLR       = 0.02;   // SGD learning rate
  cMomentum = 0.9;
  cEpochsTrue   = 200;   // true labels learn fast
  cEpochsRandom = 4000;  // random labels need many more epochs to memorise
  cChance   = 1.0 / cClasses;

type
  TRunResult = record
    Name      : string;
    TrainAcc  : TNeuralFloat;
    TestAcc   : TNeuralFloat;
    Epochs    : integer;
    Params    : integer;
  end;

// ---------------------------------------------------------------------------
// Data generation. K Gaussian blobs with fixed random centres. The TRUE label
// is which blob a point was drawn from -- a clean, learnable target. Reseeding
// before the call keeps points/centres identical across runs.
// ---------------------------------------------------------------------------
var
  Centers: array[0..cClasses - 1, 0..cDim - 1] of TNeuralFloat;

procedure MakeCenters;
var
  C, J: integer;
begin
  for C := 0 to cClasses - 1 do
    for J := 0 to cDim - 1 do
      Centers[C, J] := RandG(0, 1) * 2.5;  // well-separated blob centres
end;

// Build a labelled set with TRUE labels (blob membership).
procedure BuildSet(out Pairs: TNNetVolumePairList; Count: integer);
var
  I, J, Cls: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for I := 0 to Count - 1 do
  begin
    Cls := Random(cClasses);
    X := TNNetVolume.Create(cDim, 1, 1);
    for J := 0 to cDim - 1 do
      X.FData[J] := Centers[Cls, J] + RandG(0, 1);   // blob jitter
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.SetClassForSoftMax(Cls);
    Pairs.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// Deep-copy a pair list (inputs and targets), so the random-label run gets
// the EXACT same inputs as the true-label run.
function CopyPairs(Src: TNNetVolumePairList): TNNetVolumePairList;
var
  I: integer;
  X, Y: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for I := 0 to Src.Count - 1 do
  begin
    X := TNNetVolume.Create(cDim, 1, 1);
    X.Copy(Src[I].I);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Copy(Src[I].O);
    Result.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// Replace every training target with a RANDOMLY SHUFFLED label. The inputs are
// untouched; only the label column is permuted, so all input->label signal is
// destroyed (the marginal label distribution is preserved by permuting the
// existing labels rather than redrawing them).
procedure ShuffleLabels(Pairs: TNNetVolumePairList);
var
  I, J: integer;
  Labels: array of integer;
  Tmp: integer;
begin
  SetLength(Labels, Pairs.Count);
  for I := 0 to Pairs.Count - 1 do
    Labels[I] := Pairs[I].O.GetClass();
  // Fisher-Yates permutation of the label vector.
  for I := High(Labels) downto 1 do
  begin
    J := Random(I + 1);
    Tmp := Labels[I]; Labels[I] := Labels[J]; Labels[J] := Tmp;
  end;
  for I := 0 to Pairs.Count - 1 do
  begin
    Pairs[I].O.Fill(0);
    Pairs[I].O.SetClassForSoftMax(Labels[I]);
  end;
end;

// Corrupt a FRACTION p of the training labels by reassigning each chosen
// sample to a uniformly random class (drawn fresh, may coincide with the true
// class). The inputs are untouched. p=0 leaves the set clean; p=1 corrupts
// every label. This is the smooth interpolation between "real structure" (p=0)
// and "pure memorisation" (p=1) -- distinct from ShuffleLabels (which permutes,
// preserving the marginal). Reseed before the call for reproducibility.
procedure CorruptLabels(Pairs: TNNetVolumePairList; p: TNeuralFloat);
var
  I, NewCls: integer;
begin
  for I := 0 to Pairs.Count - 1 do
    if Random < p then
    begin
      NewCls := Random(cClasses);
      Pairs[I].O.Fill(0);
      Pairs[I].O.SetClassForSoftMax(NewCls);
    end;
end;

// ---------------------------------------------------------------------------
// The FIXED over-parameterised classifier (identical for both runs):
//   Input(cDim) -> FullConnectReLU(cHidden) -> FullConnectReLU(cHidden)
//              -> FullConnectLinear(cClasses) -> SoftMax
// Trained with mini-batch SGD (SetBatchUpdate(True): accumulate the
// cross-entropy gradient over a shuffled mini-batch, then one UpdateWeights
// step per batch). The mini-batch stochasticity is what lets the net escape
// the full-batch plateau and drive RANDOM-label train accuracy all the way to
// ~100% -- this is the SGD regime Zhang et al. (2017) used. No stochastic
// layers => deterministic forward/backward.
// ---------------------------------------------------------------------------
procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cDim, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLR, cMomentum);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);  // accumulate the mini-batch gradient, then one step
end;

// Argmax classification accuracy over a pair list.
function Accuracy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I, Hits: integer;
begin
  Hits := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    if NN.GetLastLayer().Output.GetClass() = Pairs[I].O.GetClass() then
      Inc(Hits);
  end;
  if Pairs.Count > 0 then Result := Hits / Pairs.Count else Result := 0;
end;

// Train the fixed net on TrainSet for up to MaxEpochs of mini-batch SGD; report
// train (on the labels the net actually saw) and test (always TRUE labels)
// accuracy. Each epoch shuffles the sample order and steps once per mini-batch.
// Early-stop once train accuracy is essentially perfect to save budget.
function RunOne(const Name: string; MaxEpochs: integer;
  TrainSet, TestSet: TNNetVolumePairList): TRunResult;
var
  NN: TNNet;
  Epoch, I, J, B, Tmp, PerfectStreak, InBatch: integer;
  Order: array of integer;
  TrAcc: TNeuralFloat;
begin
  Result.Name := Name;
  Result.Epochs := 0;
  // Reseed before build so weight init is IDENTICAL across both runs.
  RandSeed := cSeed;
  BuildNet(NN);
  Result.Params := NN.CountWeights();
  PerfectStreak := 0;
  SetLength(Order, TrainSet.Count);
  for I := 0 to High(Order) do Order[I] := I;
  try
    for Epoch := 1 to MaxEpochs do
    begin
      Result.Epochs := Epoch;
      // Shuffle the sample order, then run mini-batch SGD: accumulate the
      // gradient over cBatch samples and take one UpdateWeights step per batch.
      for I := High(Order) downto 1 do
      begin
        J := Random(I + 1);
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;
      InBatch := 0;
      NN.ClearDeltas();
      for B := 0 to High(Order) do
      begin
        NN.Compute(TrainSet[Order[B]].I);
        NN.Backpropagate(TrainSet[Order[B]].O);
        Inc(InBatch);
        if (InBatch >= cBatch) or (B = High(Order)) then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
          InBatch := 0;
        end;
      end;
      // Cheap early-stop: once train accuracy is perfect for several checks in
      // a row the net has fully memorised; further epochs only burn budget.
      if (Epoch mod 25 = 0) or (Epoch = MaxEpochs) then
      begin
        TrAcc := Accuracy(NN, TrainSet);
        if TrAcc >= 1.0 then Inc(PerfectStreak) else PerfectStreak := 0;
        if PerfectStreak >= 2 then Break;
      end;
    end;
    Result.TrainAcc := Accuracy(NN, TrainSet);
    Result.TestAcc  := Accuracy(NN, TestSet);
  finally
    NN.Free;
  end;
end;

// ---------------------------------------------------------------------------
// Corruption-fraction SWEEP support.
// RunSweepOne forks RunOne's net/data/training loop but records the FIRST epoch
// at which train accuracy reaches >= cFitThreshold (the "epochs-to-fit"
// counter). Train accuracy is probed every cProbeEvery epochs; EpochsToFit is
// the probe epoch at which the threshold was first crossed (cMaxSweepEpochs if
// never reached within budget). The net keeps training to cMaxSweepEpochs (or a
// short tail past the fit) so the reported final train/test accuracies reflect
// a fully-converged fit.
// ---------------------------------------------------------------------------
type
  TSweepResult = record
    P           : TNeuralFloat;
    TrainAcc    : TNeuralFloat;
    TestAcc     : TNeuralFloat;
    EpochsToFit : integer;
    Fitted      : boolean;
  end;

const
  cFitThreshold   = 0.98;   // "fit the train set" = train acc >= this (well
                            // above the 1/K=20% chance rate: unambiguous
                            // memorisation, yet reachable for full corruption
                            // within budget under plain mini-batch SGD)
  cProbeEvery     = 10;     // probe train accuracy every N epochs
  cMaxSweepEpochs = 4000;   // budget cap per corruption level (ample headroom;
                            // full corruption fits within ~100 epochs)

function RunSweepOne(p: TNeuralFloat;
  TrainSet, TestSet: TNNetVolumePairList): TSweepResult;
var
  NN: TNNet;
  Epoch, I, J, B, Tmp, InBatch: integer;
  Order: array of integer;
  TrAcc: TNeuralFloat;
begin
  Result.P := p;
  Result.Fitted := False;
  Result.EpochsToFit := cMaxSweepEpochs;
  // Reseed before build so weight init is IDENTICAL across all corruption levels.
  RandSeed := cSeed;
  BuildNet(NN);
  SetLength(Order, TrainSet.Count);
  for I := 0 to High(Order) do Order[I] := I;
  try
    for Epoch := 1 to cMaxSweepEpochs do
    begin
      for I := High(Order) downto 1 do
      begin
        J := Random(I + 1);
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;
      InBatch := 0;
      NN.ClearDeltas();
      for B := 0 to High(Order) do
      begin
        NN.Compute(TrainSet[Order[B]].I);
        NN.Backpropagate(TrainSet[Order[B]].O);
        Inc(InBatch);
        if (InBatch >= cBatch) or (B = High(Order)) then
        begin
          NN.UpdateWeights();
          NN.ClearDeltas();
          InBatch := 0;
        end;
      end;
      if (Epoch mod cProbeEvery = 0) or (Epoch = cMaxSweepEpochs) then
      begin
        TrAcc := Accuracy(NN, TrainSet);
        if (not Result.Fitted) and (TrAcc >= cFitThreshold) then
        begin
          Result.Fitted := True;
          Result.EpochsToFit := Epoch;
          Break;  // fitted -> stop; final accuracies measured right after fit
        end;
      end;
    end;
    Result.TrainAcc := Accuracy(NN, TrainSet);
    Result.TestAcc  := Accuracy(NN, TestSet);
  finally
    NN.Free;
  end;
end;

var
  TrainTrue, TrainRandom, TestSet: TNNetVolumePairList;
  TrainCleanForSweep, TestSetForSweep, SweepTrain: TNNetVolumePairList;
  TrueRun, RandRun: TRunResult;
  SweepRes: array[0..3] of TSweepResult;
  SweepI: integer;
  GapLo, GapHi: TNeuralFloat;
  StartTime, EndTime: TDateTime;
  PassRandTrain, PassRandTest, PassTrueTest, PassBothTrain: boolean;
  PassEpochsRise, PassGapWiden, PassAllFit, PassSweep: boolean;
const
  cTestMargin = 0.05;  // random-label test acc must stay within chance+margin
  cPs: array[0..3] of TNeuralFloat = (0.0, 0.25, 0.5, 1.0);  // corruption sweep
begin
  // Manual Compute/Backpropagate are single-threaded, so this whole run is
  // deterministic on one CPU core without any thread-pool setup.
  WriteLn('================================================================');
  WriteLn('Random-Label Memorization (Zhang et al. 2017): train error says');
  WriteLn('NOTHING about generalization.');
  WriteLn('================================================================');
  WriteLn(Format('Task: %d-class Gaussian blobs, D=%d.  Train=%d, Test=%d (clean).',
    [cClasses, cDim, cTrain, cTest]));
  WriteLn(Format('Chance test accuracy = 1/K = %.3f.', [cChance]));
  WriteLn(Format('FIXED net: Input(%d)->FullConnectReLU(%d)->FullConnectReLU(%d)'
    + '->FullConnectLinear(%d)->SoftMax.', [cDim, cHidden, cHidden, cClasses]));
  WriteLn(Format('Mini-batch SGD  batch=%d  LR=%.3f  momentum=%.2f  RandSeed=%d',
    [cBatch, cLR, cMomentum, cSeed]));
  WriteLn('Same net + same inputs, trained on TRUE labels vs RANDOMLY SHUFFLED labels.');
  WriteLn;

  StartTime := Now;

  // Build ONE training set and ONE test set under the fixed seed; the
  // random-label training set is a deep copy of the true one with its labels
  // permuted (inputs identical).
  RandSeed := cSeed;
  MakeCenters;
  BuildSet(TrainTrue, cTrain);
  BuildSet(TestSet, cTest);

  // Random-label training set: an EXACT copy of the true set (same inputs and
  // same label distribution), with its label column permuted to destroy signal.
  TrainRandom := CopyPairs(TrainTrue);
  ShuffleLabels(TrainRandom);

  // Keep clean copies of the train inputs/labels and the test set for PART 2's
  // corruption sweep (Part 1's try/finally below frees the originals).
  TrainCleanForSweep := CopyPairs(TrainTrue);
  TestSetForSweep := CopyPairs(TestSet);

  try
    Write('Training on TRUE labels   ');
    TrueRun := RunOne('TRUE labels  ', cEpochsTrue, TrainTrue, TestSet);
    WriteLn(Format('done (%d epochs).', [TrueRun.Epochs]));

    Write('Training on RANDOM labels ');
    RandRun := RunOne('RANDOM labels', cEpochsRandom, TrainRandom, TestSet);
    WriteLn(Format('done (%d epochs).', [RandRun.Epochs]));
  finally
    TrainTrue.Free;
    TrainRandom.Free;
    TestSet.Free;
  end;

  WriteLn;
  WriteLn('=== Results ===');
  WriteLn('run            params  epochs   TRAIN acc   TEST acc');
  WriteLn(Format('%-13s  %6d  %6d    %6.2f%%     %6.2f%%',
    [TrueRun.Name, TrueRun.Params, TrueRun.Epochs,
     TrueRun.TrainAcc * 100, TrueRun.TestAcc * 100]));
  WriteLn(Format('%-13s  %6d  %6d    %6.2f%%     %6.2f%%',
    [RandRun.Name, RandRun.Params, RandRun.Epochs,
     RandRun.TrainAcc * 100, RandRun.TestAcc * 100]));
  WriteLn(Format('chance (1/K) test accuracy = %.2f%%', [cChance * 100]));
  WriteLn;

  WriteLn('=== Correctness gate ===');

  // 1. The net MEMORISES pure noise: random-label train accuracy ~100%.
  PassRandTrain := RandRun.TrainAcc >= 0.99;
  WriteLn(Format('[%s] random-label TRAIN acc = %.2f%% (must be >= 99%%): '
    + 'the net memorised pure noise.',
    [BoolToStr(PassRandTrain, 'PASS', 'FAIL'), RandRun.TrainAcc * 100]));

  // 2. Memorising noise does NOT generalise: random-label test acc ~chance.
  PassRandTest := RandRun.TestAcc <= cChance + cTestMargin;
  WriteLn(Format('[%s] random-label TEST  acc = %.2f%% (must be <= chance+%.0f%% '
    + '= %.2f%%): no generalisation.',
    [BoolToStr(PassRandTest, 'PASS', 'FAIL'), RandRun.TestAcc * 100,
     cTestMargin * 100, (cChance + cTestMargin) * 100]));

  // 3. The SAME net on TRUE labels DOES generalise: test acc >> chance.
  PassTrueTest := TrueRun.TestAcc >= cChance + 0.40;
  WriteLn(Format('[%s] true-label   TEST  acc = %.2f%% (must be >> chance, '
    + '>= %.2f%%): real generalisation.',
    [BoolToStr(PassTrueTest, 'PASS', 'FAIL'), TrueRun.TestAcc * 100,
     (cChance + 0.40) * 100]));

  // 4. BOTH runs drive train accuracy to ~100% -- identical train error,
  //    opposite generalisation.
  PassBothTrain := (TrueRun.TrainAcc >= 0.99) and (RandRun.TrainAcc >= 0.99);
  WriteLn(Format('[%s] BOTH runs reach ~100%% TRAIN acc (true=%.2f%%, '
    + 'random=%.2f%%): same train error, opposite generalisation.',
    [BoolToStr(PassBothTrain, 'PASS', 'FAIL'),
     TrueRun.TrainAcc * 100, RandRun.TrainAcc * 100]));

  WriteLn;
  WriteLn('TAKEAWAY: both runs fit the training set perfectly (train error ~0),');
  WriteLn('yet only the true-label run generalises. TRAIN ERROR ALONE SAYS');
  WriteLn('NOTHING ABOUT GENERALIZATION -- capacity to fit the data is not');
  WriteLn('evidence of having learned anything (Zhang et al. 2017).');
  WriteLn;

  if PassRandTrain and PassRandTest and PassTrueTest and PassBothTrain then
    WriteLn('=> ALL CHECKS PASS: random-label memorization reproduced.')
  else
    WriteLn('=> SOME CHECKS FAILED (see above).');

  // -------------------------------------------------------------------------
  // PART 2: label-corruption-fraction SWEEP.
  // The binary contrast above is the two endpoints. Now sweep the corruption
  // fraction p across {0.0, 0.25, 0.5, 1.0} on the SAME inputs/net/seed and
  // chart the smooth interpolation:
  //   - epochs-to-fit-train (epochs until train acc >= 99%) should RISE with p
  //     (more noise to memorise => more steps), and
  //   - the test gap (train acc - test acc) should WIDEN with p (the fit comes
  //     more and more from memorisation, less and less from real structure).
  // -------------------------------------------------------------------------
  WriteLn;
  WriteLn('================================================================');
  WriteLn('PART 2: label-corruption-fraction sweep (smooth interpolation).');
  WriteLn('================================================================');
  WriteLn(Format('Sweep p in {0.00, 0.25, 0.50, 1.00}; fit = train acc >= %.0f%%; '
    + 'budget %d epochs/level.', [cFitThreshold * 100, cMaxSweepEpochs]));
  WriteLn('Same inputs, same net, same weight-init seed; only the training-label');
  WriteLn('corruption fraction p changes.  gap = TRAIN acc - TEST acc.');
  WriteLn;

  for SweepI := 0 to High(cPs) do
  begin
    // Fresh corrupted copy of the clean true-label train set, reseeded so the
    // corruption draw (and thus the result) is deterministic per level.
    SweepTrain := CopyPairs(TrainCleanForSweep);
    RandSeed := cSeed + 1 + SweepI;  // distinct, deterministic corruption draw
    CorruptLabels(SweepTrain, cPs[SweepI]);
    Write(Format('  p=%.2f training ... ', [cPs[SweepI]]));
    SweepRes[SweepI] := RunSweepOne(cPs[SweepI], SweepTrain, TestSetForSweep);
    SweepTrain.Free;
    WriteLn(Format('epochs-to-fit=%d%s', [SweepRes[SweepI].EpochsToFit,
      BoolToStr(SweepRes[SweepI].Fitted, '', ' (NOT fitted within budget)')]));
  end;

  WriteLn;
  WriteLn('=== Sweep results ===');
  WriteLn('   p     epochs-to-fit   TRAIN acc   TEST acc      gap');
  for SweepI := 0 to High(cPs) do
    WriteLn(Format(' %.2f   %10d      %6.2f%%    %6.2f%%   %6.2f%%',
      [SweepRes[SweepI].P, SweepRes[SweepI].EpochsToFit,
       SweepRes[SweepI].TrainAcc * 100, SweepRes[SweepI].TestAcc * 100,
       (SweepRes[SweepI].TrainAcc - SweepRes[SweepI].TestAcc) * 100]));
  WriteLn;

  WriteLn('=== Sweep correctness gate ===');

  // 5. epochs-to-fit RISES with p (lenient: the p=1.0 endpoint needs strictly
  //    more epochs than the clean p=0.0 endpoint -- memorising noise is harder).
  PassEpochsRise := SweepRes[High(cPs)].EpochsToFit > SweepRes[0].EpochsToFit;
  WriteLn(Format('[%s] epochs-to-fit rises with p: p=1.00 took %d epochs vs '
    + 'p=0.00 took %d (more noise => more steps).',
    [BoolToStr(PassEpochsRise, 'PASS', 'FAIL'),
     SweepRes[High(cPs)].EpochsToFit, SweepRes[0].EpochsToFit]));

  // 6. The test gap WIDENS with p: gap at p=1.0 is much larger than at p=0.0.
  GapLo := SweepRes[0].TrainAcc - SweepRes[0].TestAcc;
  GapHi := SweepRes[High(cPs)].TrainAcc - SweepRes[High(cPs)].TestAcc;
  PassGapWiden := GapHi >= GapLo + 0.40;
  WriteLn(Format('[%s] test gap widens with p: gap(p=1.00)=%.2f%% vs '
    + 'gap(p=0.00)=%.2f%% (must widen by >= 40%%).',
    [BoolToStr(PassGapWiden, 'PASS', 'FAIL'), GapHi * 100, GapLo * 100]));

  // 7. Every corruption level still FITS the train set within budget (capacity
  //    to memorise holds across the whole sweep).
  PassAllFit := True;
  for SweepI := 0 to High(cPs) do
    if not SweepRes[SweepI].Fitted then PassAllFit := False;
  WriteLn(Format('[%s] all %d corruption levels reached train acc >= %.0f%% '
    + 'within budget.',
    [BoolToStr(PassAllFit, 'PASS', 'FAIL'), Length(cPs), cFitThreshold * 100]));

  PassSweep := PassEpochsRise and PassGapWiden and PassAllFit;

  WriteLn;
  WriteLn('SWEEP TAKEAWAY: as the label-corruption fraction p goes 0 -> 1, the');
  WriteLn('network needs MORE epochs to fit the train set and its test gap WIDENS');
  WriteLn('-- a smooth interpolation from learning real structure (p=0) to pure');
  WriteLn('memorization (p=1).');
  WriteLn;

  TestSetForSweep.Free;
  TrainCleanForSweep.Free;

  EndTime := Now;

  if PassRandTrain and PassRandTest and PassTrueTest and PassBothTrain
     and PassSweep then
    WriteLn('=> ALL CHECKS PASS: binary contrast + corruption sweep reproduced.')
  else
    WriteLn('=> SOME CHECKS FAILED (see above).');
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  if not (PassRandTrain and PassRandTest and PassTrueTest and PassBothTrain
          and PassSweep) then
    Halt(1);
end.
