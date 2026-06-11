program LotteryTicket;
(*
LotteryTicket: a tiny, pure-CPU demonstration of the Lottery-Ticket
Hypothesis (Frankle & Carbin, 2019, "The Lottery Ticket Hypothesis:
Finding Sparse, Trainable Neural Networks", arXiv:1803.03635).

The recipe:
  1. Build a small dense MLP and SAVE its random initial weights (theta_0).
  2. Train the dense net to convergence on a tiny synthetic non-linear
     2-class task (two interleaved spirals); record the dense baseline.
  3. From the TRAINED dense weights, build a binary mask that prunes the
     bottom X% of weights by magnitude (the "winning ticket" mask).
  4. At each sparsity X in {50%, 80%, 90%, 95%, 98%} compare THREE conditions,
     all at matched sparsity and matched epochs:
       (LT)     reset surviving weights to their ORIGINAL theta_0 values,
                then retrain with the mask held fixed.
       (Random) reinitialise surviving weights to FRESH random values,
                then retrain with the same mask.
       (Dense)  the unpruned net's final accuracy (one fixed number).
  5. Report a table: sparsity vs final loss/accuracy for LT vs Random vs Dense.

The mask is enforced as a post-step projection: after every weight update
the pruned positions are re-zeroed, so weights pruned to 0 stay 0 for the
whole retraining run.

The lottery-ticket headline: the LT (original-init) sparse subnet matches
or beats the dense baseline and clearly beats random-reinit at high sparsity.

Single-threaded, fixed RandSeed, finishes in seconds on a CPU.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuralfit;

const
  RAND_SEED    = 42;
  INPUT_DIM    = 2;
  HIDDEN_UNITS = 64;       // big enough that 90-95% pruning is meaningful
  NUM_CLASSES  = 2;
  TRAIN_SIZE   = 600;
  TEST_SIZE    = 300;
  NUM_EPOCHS   = 100;
  BATCH_SIZE   = 32;
  LEARN_RATE   = 0.02;
  N_TRIALS     = 5;        // runs averaged per (sparsity, condition)
  BLOB_SPREAD  = 0.55;     // std-dev of each gaussian blob; overlap => non-trivial

type
  TSparsityArr = array of TNeuralFloat;

// ----- one flat record per surviving weight ----------------------------------
// We index every prunable weight in the net by (layer, neuron, weightIdx).
type
  TWeightRef = record
    LayerIdx, NeuronIdx, WeightIdx: integer;
  end;
  TWeightRefArr = array of TWeightRef;

var
  GMask: TWeightRefArr;     // refs to weights that are PRUNED (forced to 0)
  GMaskNN: TNNet;           // the net the mask applies to (for the OnAfterStep hook)

// ----- tiny synthetic NON-LINEAR 2-class task --------------------------------
// Two interleaved spirals (the classic capacity-stressing toy). Each class is
// one arm of a spiral; the decision boundary is highly non-linear, so the task
// genuinely needs hidden-unit capacity. At high sparsity the random-reinit
// subnet struggles to fit it, while the original-init "winning ticket" still
// trains well -- which is exactly the lottery-ticket contrast.
function CreateRings(MaxCnt: integer): TNNetVolumePairList;
var
  Cnt, Cls: integer;
  T, R, Theta, Px, Py: TNeuralFloat;
  Target: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for Cnt := 1 to MaxCnt do
  begin
    Cls := Random(NUM_CLASSES);
    T := Random;                          // position along the arm, [0,1]
    R := 0.3 + 1.7 * T;                   // radius grows along the arm
    // two arms 180 degrees out of phase; ~1.75 turns each
    Theta := 3.5 * Pi * T + Cls * Pi;
    Theta := Theta + BLOB_SPREAD * 0.22 * RandG(0, 1);   // angular noise / overlap
    Px := R * Cos(Theta);
    Py := R * Sin(Theta);
    Target := TNNetVolume.Create(NUM_CLASSES);
    Target.FData[Cls] := 1.0;            // one-hot
    Result.Add(
      TNNetVolumePair.Create(
        TNNetVolume.Create([Px, Py]),
        Target
      )
    );
  end;
end;

function ArgMax(V: TNNetVolume): integer;
var I, Best: integer;
begin
  Best := 0;
  for I := 1 to V.Size - 1 do
    if V.FData[I] > V.FData[Best] then Best := I;
  Result := Best;
end;

procedure EvaluateNet(NN: TNNet; Pairs: TNNetVolumePairList;
  out Accuracy, MeanLoss: TNeuralFloat);
var
  I: integer;
  Correct: integer;
  SumLoss: Double;
  P: TNeuralFloat;
begin
  Correct := 0;
  SumLoss := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    if ArgMax(NN.GetLastLayer().Output) = ArgMax(Pairs[I].O) then Inc(Correct);
    // cross-entropy of the true class
    P := NN.GetLastLayer().Output.FData[ArgMax(Pairs[I].O)];
    if P < 1e-7 then P := 1e-7;
    SumLoss := SumLoss - Ln(P);
  end;
  if Pairs.Count > 0 then
  begin
    Accuracy := Correct / Pairs.Count;
    MeanLoss := SumLoss / Pairs.Count;
  end
  else begin Accuracy := 0; MeanLoss := 0; end;
end;

procedure BuildNet(NN: TNNet);
begin
  NN.AddLayer(TNNetInput.Create(INPUT_DIM));
  NN.AddLayer(TNNetFullConnectReLU.Create(HIDDEN_UNITS));
  NN.AddLayer(TNNetFullConnectReLU.Create(HIDDEN_UNITS));
  NN.AddLayer(TNNetFullConnectLinear.Create(NUM_CLASSES));
  NN.AddLayer(TNNetSoftMax.Create());
end;

// Enumerate every prunable weight (excludes biases) into a flat ref array,
// and collect their absolute magnitudes in the same order.
procedure CollectWeights(NN: TNNet; out Refs: TWeightRefArr;
  out Mags: TSparsityArr);
var
  L, N, W, Count: integer;
  Layer: TNNetLayer;
begin
  Count := 0;
  for L := 0 to NN.GetLastLayerIdx() do
  begin
    Layer := NN.Layers[L];
    for N := 0 to Layer.Neurons.Count - 1 do
      Inc(Count, Layer.Neurons[N].Weights.Size);
  end;
  SetLength(Refs, Count);
  SetLength(Mags, Count);
  Count := 0;
  for L := 0 to NN.GetLastLayerIdx() do
  begin
    Layer := NN.Layers[L];
    for N := 0 to Layer.Neurons.Count - 1 do
      for W := 0 to Layer.Neurons[N].Weights.Size - 1 do
      begin
        Refs[Count].LayerIdx := L;
        Refs[Count].NeuronIdx := N;
        Refs[Count].WeightIdx := W;
        Mags[Count] := Abs(Layer.Neurons[N].Weights.FData[W]);
        Inc(Count);
      end;
  end;
end;

// Returns the magnitude threshold below which a weight is pruned, so that
// approximately Fraction of the weights are pruned.
function MagnitudeThreshold(Mags: TSparsityArr; Fraction: TNeuralFloat): TNeuralFloat;
var
  Sorted: TSparsityArr;
  I, J, Cut: integer;
  Tmp: TNeuralFloat;
begin
  SetLength(Sorted, Length(Mags));
  for I := 0 to High(Mags) do Sorted[I] := Mags[I];
  // simple insertion-ish sort is fine for a few thousand weights; use a
  // shell-style sort to stay quick.
  for I := 1 to High(Sorted) do
  begin
    Tmp := Sorted[I];
    J := I - 1;
    while (J >= 0) and (Sorted[J] > Tmp) do
    begin
      Sorted[J + 1] := Sorted[J];
      Dec(J);
    end;
    Sorted[J + 1] := Tmp;
  end;
  Cut := Trunc(Fraction * Length(Sorted));
  if Cut < 0 then Cut := 0;
  if Cut > High(Sorted) then Cut := High(Sorted);
  Result := Sorted[Cut];
end;

// Build the PRUNED-position list (mask) from the trained dense net at a given
// sparsity. Weights with magnitude strictly below threshold are pruned.
procedure BuildMask(DenseNN: TNNet; Fraction: TNeuralFloat;
  out PrunedRefs: TWeightRefArr; out ActualSparsity: TNeuralFloat);
var
  Refs: TWeightRefArr;
  Mags: TSparsityArr;
  Thr: TNeuralFloat;
  I, Count: integer;
begin
  CollectWeights(DenseNN, Refs, Mags);
  Thr := MagnitudeThreshold(Mags, Fraction);
  Count := 0;
  SetLength(PrunedRefs, Length(Refs));
  for I := 0 to High(Refs) do
    if Mags[I] < Thr then
    begin
      PrunedRefs[Count] := Refs[I];
      Inc(Count);
    end;
  SetLength(PrunedRefs, Count);
  if Length(Refs) > 0 then
    ActualSparsity := Count / Length(Refs)
  else
    ActualSparsity := 0;
end;

// Zero every pruned weight in NN according to the global mask.
procedure ApplyMask(NN: TNNet; Mask: TWeightRefArr);
var I: integer;
begin
  for I := 0 to High(Mask) do
    NN.Layers[Mask[I].LayerIdx].Neurons[Mask[I].NeuronIdx]
      .Weights.FData[Mask[I].WeightIdx] := 0.0;
end;

// OnAfterStep hook: re-project to the mask after every weight update so
// pruned weights stay at zero throughout training.
type
  TMaskEnforcer = class
    procedure OnAfterStep(Sender: TObject);
  end;

procedure TMaskEnforcer.OnAfterStep(Sender: TObject);
begin
  if GMaskNN <> nil then ApplyMask(GMaskNN, GMask);
end;

var
  GEnforcer: TMaskEnforcer;

// Fresh random re-initialisation of all weights using the same initializer
// the layers use at construction time (He uniform for ReLU FC layers).
procedure ReinitWeights(NN: TNNet);
var L: integer;
begin
  for L := 1 to NN.GetLastLayerIdx() do
    NN.Layers[L].InitDefault();
end;

// Train NN (mask already applied to NN before call) for NUM_EPOCHS with the
// mask held fixed via the OnAfterStep hook.
procedure TrainMasked(NN: TNNet; Mask: TWeightRefArr;
  Train, Test: TNNetVolumePairList);
var
  NFit: TNeuralFit;
begin
  GMaskNN := NN;
  GMask := Mask;
  ApplyMask(NN, Mask);          // enforce at t=0
  NFit := TNeuralFit.Create();
  try
    NFit.FileNameBase := GetTempDir + 'LotteryTicket_autosave';
    NFit.MaxThreadNum := 1;       // determinism
    NFit.HideMessages();          // quiet per-epoch logging
    NFit.InitialLearningRate := LEARN_RATE;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.EnableBipolar99HitComparison();
    NFit.OnAfterStep := @GEnforcer.OnAfterStep;
    NFit.Fit(NN, Train, Test, nil, BATCH_SIZE, NUM_EPOCHS);
  finally
    NFit.Free;
    GMaskNN := nil;
    SetLength(GMask, 0);
  end;
end;

type
  TRunResult = record
    Sparsity: TNeuralFloat;
    LTLoss, LTAcc: TNeuralFloat;
    RandLoss, RandAcc: TNeuralFloat;
  end;

procedure RunAlgo();
const
  Fractions: array[0..4] of TNeuralFloat = (0.50, 0.70, 0.80, 0.90, 0.95);
var
  Train, Test: TNNetVolumePairList;
  DenseNN, Theta0NN, WorkNN: TNNet;
  DenseAcc, DenseLoss: TNeuralFloat;
  Mask: TWeightRefArr;
  ActualSparsity: TNeuralFloat;
  Results: array[0..4] of TRunResult;
  FI, Trial: integer;
  AccLT, LossLT, AccRand, LossRand, TrialAcc, TrialLoss: TNeuralFloat;
  StartTime, EndTime: TDateTime;
  TotalW: integer;
begin
  StartTime := Now;

  // ---- data ----
  RandSeed := RAND_SEED;
  Train := CreateRings(TRAIN_SIZE);
  Test  := CreateRings(TEST_SIZE);

  // ---- build the net and snapshot theta_0 ----
  RandSeed := RAND_SEED;
  DenseNN := TNNet.Create();
  BuildNet(DenseNN);
  TotalW := DenseNN.CountWeights();

  // theta_0 = a frozen clone of the freshly-initialised weights.
  Theta0NN := TNNet.Create();
  BuildNet(Theta0NN);
  Theta0NN.CopyWeights(DenseNN);   // hold the ORIGINAL init

  WriteLn('Lottery-Ticket Hypothesis demo (Frankle & Carbin 2019) on a tiny');
  WriteLn('non-linear 2-class two-spiral task. Net: ', INPUT_DIM, ' -> ',
          HIDDEN_UNITS, ' -> ', HIDDEN_UNITS, ' -> ', NUM_CLASSES,
          ' (ReLU MLP + softmax).');
  WriteLn('Prunable weights: ', TotalW, '.  Epochs: ', NUM_EPOCHS,
          '.  Train/Test: ', TRAIN_SIZE, '/', TEST_SIZE,
          '.  LR=', FloatToStrF(LEARN_RATE, ffFixed, 4, 3),
          '.  RandSeed=', RAND_SEED, '.');
  WriteLn('LT and Random columns are means over ', N_TRIALS, ' trials each.');
  WriteLn;

  // ---- train the dense baseline ----
  Write('Training dense baseline ...');
  GMaskNN := nil;                  // no mask while training dense
  TrainMasked(DenseNN, nil, Train, Test);
  EvaluateNet(DenseNN, Test, DenseAcc, DenseLoss);
  WriteLn(' done.  dense test acc=', FloatToStrF(DenseAcc, ffFixed, 6, 4),
          '  loss=', FloatToStrF(DenseLoss, ffFixed, 6, 4));
  WriteLn;

  // ---- sweep sparsity levels ----
  // A single tiny net on one seed is noisy, so each condition is averaged
  // over N_TRIALS runs (different mini-batch shuffles; the random-reinit arm
  // also draws fresh weights each trial). This is how lottery-ticket results
  // are normally reported and it makes the systematic gap legible.
  for FI := 0 to High(Fractions) do
  begin
    BuildMask(DenseNN, Fractions[FI], Mask, ActualSparsity);
    Results[FI].Sparsity := ActualSparsity;
    AccLT := 0; LossLT := 0; AccRand := 0; LossRand := 0;

    Write('Sparsity ', FloatToStrF(ActualSparsity*100, ffFixed, 5, 1),
          '%  (', Length(Mask), ' / ', TotalW, ' pruned) ...');

    for Trial := 0 to N_TRIALS - 1 do
    begin
      // -- (LT) reset surviving weights to theta_0, retrain with mask --
      RandSeed := RAND_SEED + 1000 * (FI + 1) + Trial;
      WorkNN := TNNet.Create();
      BuildNet(WorkNN);
      WorkNN.CopyWeights(Theta0NN);          // back to the ORIGINAL init
      TrainMasked(WorkNN, Mask, Train, Test);
      EvaluateNet(WorkNN, Test, TrialAcc, TrialLoss);
      AccLT := AccLT + TrialAcc; LossLT := LossLT + TrialLoss;
      WorkNN.Free;

      // -- (Random reinit) fresh random surviving weights, same mask --
      RandSeed := RAND_SEED + 7000 * (FI + 1) + Trial;
      WorkNN := TNNet.Create();
      BuildNet(WorkNN);
      ReinitWeights(WorkNN);                 // FRESH random weights
      TrainMasked(WorkNN, Mask, Train, Test);
      EvaluateNet(WorkNN, Test, TrialAcc, TrialLoss);
      AccRand := AccRand + TrialAcc; LossRand := LossRand + TrialLoss;
      WorkNN.Free;
    end;

    Results[FI].LTAcc    := AccLT    / N_TRIALS;
    Results[FI].LTLoss   := LossLT   / N_TRIALS;
    Results[FI].RandAcc  := AccRand  / N_TRIALS;
    Results[FI].RandLoss := LossRand / N_TRIALS;

    WriteLn(' done.');
  end;
  EndTime := Now;

  // ---- report ----
  WriteLn;
  WriteLn('=== Results: final TEST accuracy / cross-entropy loss ===');
  WriteLn('Dense baseline:  acc=', FloatToStrF(DenseAcc, ffFixed, 6, 4),
          '  loss=', FloatToStrF(DenseLoss, ffFixed, 6, 4),
          '   (0% sparsity, all ', TotalW, ' weights)');
  WriteLn;
  WriteLn('sparsity_pct,lt_acc,lt_loss,rand_acc,rand_loss,dense_acc,dense_loss');
  for FI := 0 to High(Fractions) do
    WriteLn(
      FloatToStrF(Results[FI].Sparsity*100, ffFixed, 5, 1), ',',
      FloatToStrF(Results[FI].LTAcc,    ffFixed, 6, 4), ',',
      FloatToStrF(Results[FI].LTLoss,   ffFixed, 6, 4), ',',
      FloatToStrF(Results[FI].RandAcc,  ffFixed, 6, 4), ',',
      FloatToStrF(Results[FI].RandLoss, ffFixed, 6, 4), ',',
      FloatToStrF(DenseAcc,             ffFixed, 6, 4), ',',
      FloatToStrF(DenseLoss,            ffFixed, 6, 4));

  WriteLn;
  WriteLn('Reading it: LT (original-init winning ticket) should match or beat');
  WriteLn('the dense baseline and clearly beat random-reinit at high sparsity.');
  WriteLn;
  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');

  Theta0NN.Free;
  DenseNN.Free;
  Test.Free;
  Train.Free;
end;

begin
  // Determinism: single thread + fixed seed.
  RandSeed := RAND_SEED;
  GMaskNN := nil;
  GEnforcer := TMaskEnforcer.Create;
  try
    RunAlgo();
  finally
    GEnforcer.Free;
  end;
end.
