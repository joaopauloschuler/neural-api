program LotteryTicketIMP;
(*
LotteryTicketIMP: ITERATIVE magnitude pruning (IMP) -- the actual method from
Frankle & Carbin (2019, "The Lottery Ticket Hypothesis: Finding Sparse,
Trainable Neural Networks", arXiv:1803.03635) -- contrasted against the ONE-SHOT
prune-to-target baseline implemented in the sibling examples/LotteryTicket.

The sibling example prunes the trained dense net to the final sparsity in a
SINGLE step, then retrains from theta_0. The paper instead recommends ITERATIVE
pruning: prune a small fraction, rewind to theta_0, retrain, and repeat. Each
round the mask is grown by pruning the bottom p% of the weights that are STILL
SURVIVING (not of all weights). After N rounds the surviving fraction is
(1-p)^N, so we choose p so that (1-p)^N matches the one-shot target (~5%
surviving = 95% sparsity).

  IMP loop (N rounds):
    survivors := all prunable weights
    for round 1..N:
       reset survivors to theta_0  (rewinding)
       train with the current mask held fixed (post-step re-zeroing)
       rank the SURVIVING weights by |magnitude|
       prune the bottom p% of survivors -> grow the mask
    final retrain from theta_0 at the final mask; report accuracy

  One-shot baseline (matched final sparsity, matched TOTAL train budget):
       prune the trained-dense net to the final sparsity in one step,
       reset survivors to theta_0, retrain for (N+1)*epochs_per_round epochs.

The headline question: at the 95% sparsity where the one-shot run COLLAPSED in
examples/LotteryTicket (~67% acc, no better than random-reinit), does iterative
pruning find a winning ticket that recovers accuracy?

HONEST CAVEAT: this is a tiny net on an easy-ish toy. Whether the IMP-vs-one-shot
gap actually opens depends on capacity. The program GRADES the outcome and prints
an explicit verdict either way -- including "no clear gap on this toy" if IMP does
not measurably beat one-shot. Mirroring the sibling example's honesty.

Pure CPU, single-threaded, fixed RandSeed, finishes in ~1-2 minutes.

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
  RAND_SEED        = 42;
  INPUT_DIM        = 2;
  HIDDEN_UNITS     = 64;
  NUM_CLASSES      = 2;
  TRAIN_SIZE       = 600;
  TEST_SIZE        = 300;
  BATCH_SIZE       = 32;
  LEARN_RATE       = 0.02;
  BLOB_SPREAD      = 0.55;

  // --- IMP schedule -----------------------------------------------------------
  // N_ROUNDS prune steps; per-round survivors keep (1-PRUNE_PER_ROUND).
  // (1-0.45)^5 = 0.55^5 = 0.0503  ->  ~94.97% final sparsity, matching one-shot.
  N_ROUNDS         = 5;
  PRUNE_PER_ROUND  = 0.45;
  EPOCHS_PER_ROUND = 90;
  // A single tiny net on one seed is noisy, so the FINAL head-to-head numbers
  // (IMP ticket vs one-shot at ~95% sparsity) are each averaged over N_TRIALS
  // retrains with different mini-batch shuffles. The mask discovery itself is
  // deterministic given the seed; only the final-retrain shuffle varies.
  N_TRIALS         = 2;
  // One-shot is trained for the SAME total budget as IMP's whole schedule
  // (IMP does N_ROUNDS training passes plus one final pass = N_ROUNDS+1).
  ONESHOT_EPOCHS   = (N_ROUNDS + 1) * EPOCHS_PER_ROUND;

type
  TFloatArr = array of TNeuralFloat;

  // One flat ref per prunable weight: (layer, neuron, weightIdx).
  TWeightRef = record
    LayerIdx, NeuronIdx, WeightIdx: integer;
  end;
  TWeightRefArr = array of TWeightRef;

var
  GMask: TWeightRefArr;     // refs to weights that are PRUNED (forced to 0)
  GMaskNN: TNNet;           // the net the mask applies to (OnAfterStep hook)

// ----- tiny synthetic NON-LINEAR 2-class task (two interleaved spirals) ------
// Identical generator to examples/LotteryTicket so the tasks are comparable.
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
    T := Random;
    R := 0.3 + 1.7 * T;
    Theta := 3.5 * Pi * T + Cls * Pi;
    Theta := Theta + BLOB_SPREAD * 0.22 * RandG(0, 1);
    Px := R * Cos(Theta);
    Py := R * Sin(Theta);
    Target := TNNetVolume.Create(NUM_CLASSES);
    Target.FData[Cls] := 1.0;
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
  I, Correct: integer;
  SumLoss: Double;
  P: TNeuralFloat;
begin
  Correct := 0;
  SumLoss := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    if ArgMax(NN.GetLastLayer().Output) = ArgMax(Pairs[I].O) then Inc(Correct);
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

// Enumerate every prunable weight (excludes biases) into a flat ref array.
procedure CollectRefs(NN: TNNet; out Refs: TWeightRefArr);
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
        Inc(Count);
      end;
  end;
end;

function WeightOf(NN: TNNet; const R: TWeightRef): TNeuralFloat;
begin
  Result := NN.Layers[R.LayerIdx].Neurons[R.NeuronIdx]
              .Weights.FData[R.WeightIdx];
end;

// Ascending insertion-into-sorted; fine for a few thousand magnitudes.
function MagnitudeThreshold(Mags: TFloatArr; Fraction: TNeuralFloat): TNeuralFloat;
var
  Sorted: TFloatArr;
  I, J, Cut: integer;
  Tmp: TNeuralFloat;
begin
  SetLength(Sorted, Length(Mags));
  for I := 0 to High(Mags) do Sorted[I] := Mags[I];
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

// Zero every pruned weight in NN according to the global mask.
procedure ApplyMask(NN: TNNet; const Mask: TWeightRefArr);
var I: integer;
begin
  for I := 0 to High(Mask) do
    NN.Layers[Mask[I].LayerIdx].Neurons[Mask[I].NeuronIdx]
      .Weights.FData[Mask[I].WeightIdx] := 0.0;
end;

// OnAfterStep hook: re-project to the mask after every weight update.
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

// Train NN for Epochs with the mask held fixed via the OnAfterStep hook.
procedure TrainMasked(NN: TNNet; const Mask: TWeightRefArr;
  Train, Test: TNNetVolumePairList; Epochs: integer);
var
  NFit: TNeuralFit;
begin
  GMaskNN := NN;
  GMask := Mask;
  ApplyMask(NN, Mask);          // enforce at t=0
  NFit := TNeuralFit.Create();
  try
    NFit.FileNameBase := GetTempDir + 'LotteryTicketIMP_autosave';
    NFit.MaxThreadNum := 1;       // determinism
    NFit.HideMessages();
    NFit.InitialLearningRate := LEARN_RATE;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.EnableBipolar99HitComparison();
    NFit.OnAfterStep := @GEnforcer.OnAfterStep;
    NFit.Fit(NN, Train, Test, nil, BATCH_SIZE, Epochs);
  finally
    NFit.Free;
    GMaskNN := nil;
    SetLength(GMask, 0);
  end;
end;

// Grow PRUNED from the bottom `Fraction` of weights still SURVIVING in NN.
// Survivors = refs not already in PRUNED. Re-derives a survivor list, ranks it
// by |magnitude|, and appends the bottom `Fraction` of survivors to PRUNED.
procedure PruneSurvivors(NN: TNNet; AllRefs: TWeightRefArr;
  var Pruned: TWeightRefArr; Fraction: TNeuralFloat);
var
  IsPruned: array of boolean;
  Survivors: TWeightRefArr;
  Mags: TFloatArr;
  I, SCount, PCount: integer;
  Thr: TNeuralFloat;
begin
  // Mark currently-pruned positions. Each pruned ref equals some AllRefs[k];
  // a direct scan is O(n^2) but n is only a few thousand here, so it's fine.
  SetLength(IsPruned, Length(AllRefs));
  for I := 0 to High(IsPruned) do IsPruned[I] := False;
  for I := 0 to High(Pruned) do
  begin
    // find matching slot in AllRefs
    PCount := 0;
    while PCount <= High(AllRefs) do
    begin
      if (AllRefs[PCount].LayerIdx = Pruned[I].LayerIdx) and
         (AllRefs[PCount].NeuronIdx = Pruned[I].NeuronIdx) and
         (AllRefs[PCount].WeightIdx = Pruned[I].WeightIdx) then
      begin
        IsPruned[PCount] := True;
        Break;
      end;
      Inc(PCount);
    end;
  end;

  // Collect survivors + their magnitudes.
  SetLength(Survivors, Length(AllRefs));
  SetLength(Mags, Length(AllRefs));
  SCount := 0;
  for I := 0 to High(AllRefs) do
    if not IsPruned[I] then
    begin
      Survivors[SCount] := AllRefs[I];
      Mags[SCount] := Abs(WeightOf(NN, AllRefs[I]));
      Inc(SCount);
    end;
  SetLength(Survivors, SCount);
  SetLength(Mags, SCount);

  if SCount = 0 then Exit;
  Thr := MagnitudeThreshold(Mags, Fraction);

  // Append survivors below threshold to the pruned set.
  PCount := Length(Pruned);
  SetLength(Pruned, PCount + SCount);
  for I := 0 to SCount - 1 do
    if Mags[I] < Thr then
    begin
      Pruned[PCount] := Survivors[I];
      Inc(PCount);
    end;
  SetLength(Pruned, PCount);
end;

// One-shot mask: prune the bottom `Fraction` of ALL weights of NN in one step.
procedure BuildOneShotMask(NN: TNNet; AllRefs: TWeightRefArr;
  Fraction: TNeuralFloat; out Pruned: TWeightRefArr);
var
  Mags: TFloatArr;
  Thr: TNeuralFloat;
  I, Count: integer;
begin
  SetLength(Mags, Length(AllRefs));
  for I := 0 to High(AllRefs) do
    Mags[I] := Abs(WeightOf(NN, AllRefs[I]));
  Thr := MagnitudeThreshold(Mags, Fraction);
  Count := 0;
  SetLength(Pruned, Length(AllRefs));
  for I := 0 to High(AllRefs) do
    if Mags[I] < Thr then
    begin
      Pruned[Count] := AllRefs[I];
      Inc(Count);
    end;
  SetLength(Pruned, Count);
end;

type
  TRoundRec = record
    Sparsity, Acc, Loss: TNeuralFloat;
  end;

procedure RunAlgo();
var
  Train, Test: TNNetVolumePairList;
  DenseNN, Theta0NN, WorkNN: TNNet;
  AllRefs: TWeightRefArr;
  Pruned: TWeightRefArr;
  DenseAcc, DenseLoss: TNeuralFloat;
  TotalW, Round, FI: integer;
  Rounds: array[1..N_ROUNDS] of TRoundRec;
  IMPAcc, IMPLoss, IMPSparsity: TNeuralFloat;
  OneShotAcc, OneShotLoss, OneShotSparsity: TNeuralFloat;
  OneShotMask: TWeightRefArr;
  StartTime, EndTime: TDateTime;
  Verdict: string;
  Margin, TrialAcc, TrialLoss: TNeuralFloat;
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
  CollectRefs(DenseNN, AllRefs);

  Theta0NN := TNNet.Create();
  BuildNet(Theta0NN);
  Theta0NN.CopyWeights(DenseNN);   // hold the ORIGINAL init theta_0

  WriteLn('Iterative Magnitude Pruning (IMP) vs one-shot prune -- Lottery Ticket');
  WriteLn('Hypothesis (Frankle & Carbin 2019). Two interleaved spirals, net: ',
          INPUT_DIM, ' -> ', HIDDEN_UNITS, ' -> ', HIDDEN_UNITS, ' -> ',
          NUM_CLASSES, ' (ReLU MLP + softmax).');
  WriteLn('Prunable weights: ', TotalW, '.  Train/Test: ', TRAIN_SIZE, '/',
          TEST_SIZE, '.  LR=', FloatToStrF(LEARN_RATE, ffFixed, 4, 3),
          '.  RandSeed=', RAND_SEED, '.');
  WriteLn('IMP: ', N_ROUNDS, ' rounds, prune ',
          FloatToStrF(PRUNE_PER_ROUND*100, ffFixed, 4, 0),
          '% of SURVIVORS/round, ', EPOCHS_PER_ROUND, ' epochs/round, rewind to',
          ' theta_0 each round.');
  WriteLn('Target final survivors = (1-', FloatToStrF(PRUNE_PER_ROUND, ffFixed, 3, 2),
          ')^', N_ROUNDS, ' = ',
          FloatToStrF(Power(1-PRUNE_PER_ROUND, N_ROUNDS)*100, ffFixed, 5, 2),
          '% (~', FloatToStrF((1-Power(1-PRUNE_PER_ROUND, N_ROUNDS))*100, ffFixed, 4, 1),
          '% sparsity).');
  WriteLn('One-shot baseline: prune to the same final sparsity in ONE step,',
          ' retrain ', ONESHOT_EPOCHS, ' epochs (matched total budget).');
  WriteLn;

  // ---- dense baseline (one short train, just for a reference number) -------
  Write('Training dense baseline (', EPOCHS_PER_ROUND, ' epochs) ...');
  TrainMasked(DenseNN, nil, Train, Test, EPOCHS_PER_ROUND);
  EvaluateNet(DenseNN, Test, DenseAcc, DenseLoss);
  WriteLn(' done.  dense test acc=', FloatToStrF(DenseAcc, ffFixed, 6, 4),
          '  loss=', FloatToStrF(DenseLoss, ffFixed, 6, 4));
  WriteLn;

  // ====================================================================
  // IMP: iterative prune -> rewind -> retrain.
  // ====================================================================
  WriteLn('--- IMP rounds ---');
  SetLength(Pruned, 0);                 // start fully dense
  WorkNN := TNNet.Create();
  BuildNet(WorkNN);
  for Round := 1 to N_ROUNDS do
  begin
    // 1. rewind survivors to theta_0 (pruned positions re-zeroed by the mask).
    WorkNN.CopyWeights(Theta0NN);
    ApplyMask(WorkNN, Pruned);
    // 2. retrain with current mask held fixed.
    TrainMasked(WorkNN, Pruned, Train, Test, EPOCHS_PER_ROUND);
    // 3. grow the mask: prune the bottom p% of SURVIVORS by trained magnitude.
    PruneSurvivors(WorkNN, AllRefs, Pruned, PRUNE_PER_ROUND);
    // 4. evaluate THIS round's ticket (rewind+retrain at the NEW sparsity so
    //    the reported number is the lottery-ticket accuracy at that sparsity).
    WorkNN.CopyWeights(Theta0NN);
    ApplyMask(WorkNN, Pruned);
    TrainMasked(WorkNN, Pruned, Train, Test, EPOCHS_PER_ROUND);
    EvaluateNet(WorkNN, Test, Rounds[Round].Acc, Rounds[Round].Loss);
    Rounds[Round].Sparsity := Length(Pruned) / TotalW;
    WriteLn('  round ', Round, ': sparsity ',
            FloatToStrF(Rounds[Round].Sparsity*100, ffFixed, 5, 1),
            '%  acc=', FloatToStrF(Rounds[Round].Acc, ffFixed, 6, 4),
            '  loss=', FloatToStrF(Rounds[Round].Loss, ffFixed, 6, 4));
  end;
  IMPSparsity := Rounds[N_ROUNDS].Sparsity;
  // Pruned now holds the final IMP mask. Build the one-shot mask at the SAME
  // final sparsity from the trained dense net for an apples-to-apples contrast.
  BuildOneShotMask(DenseNN, AllRefs, IMPSparsity, OneShotMask);
  OneShotSparsity := Length(OneShotMask) / TotalW;
  WriteLn;

  // ====================================================================
  // Averaged final head-to-head: retrain BOTH final masks (IMP & one-shot)
  // from theta_0 over N_TRIALS shuffles and average. The IMP mask retrains for
  // EPOCHS_PER_ROUND (its per-round budget); the one-shot mask retrains for the
  // matched TOTAL budget (N_ROUNDS+1 rounds' worth), giving one-shot the same
  // total epochs IMP spent across all its rounds.
  // ====================================================================
  Write('--- Final head-to-head (', N_TRIALS, ' trials, ~',
        FloatToStrF(IMPSparsity*100, ffFixed, 4, 1), '% sparsity) ...');
  IMPAcc := 0; IMPLoss := 0; OneShotAcc := 0; OneShotLoss := 0;
  for Round := 0 to N_TRIALS - 1 do
  begin
    // IMP winning ticket: rewind to theta_0, retrain at the final IMP mask.
    RandSeed := RAND_SEED + 1000 * (Round + 1);
    WorkNN.CopyWeights(Theta0NN);
    ApplyMask(WorkNN, Pruned);
    TrainMasked(WorkNN, Pruned, Train, Test, EPOCHS_PER_ROUND);
    EvaluateNet(WorkNN, Test, TrialAcc, TrialLoss);
    IMPAcc := IMPAcc + TrialAcc; IMPLoss := IMPLoss + TrialLoss;

    // One-shot ticket: rewind to theta_0, retrain at the one-shot mask for the
    // matched total budget.
    RandSeed := RAND_SEED + 7000 * (Round + 1);
    WorkNN.CopyWeights(Theta0NN);
    ApplyMask(WorkNN, OneShotMask);
    TrainMasked(WorkNN, OneShotMask, Train, Test, ONESHOT_EPOCHS);
    EvaluateNet(WorkNN, Test, TrialAcc, TrialLoss);
    OneShotAcc := OneShotAcc + TrialAcc;
    OneShotLoss := OneShotLoss + TrialLoss;
  end;
  IMPAcc := IMPAcc / N_TRIALS;     IMPLoss := IMPLoss / N_TRIALS;
  OneShotAcc := OneShotAcc / N_TRIALS; OneShotLoss := OneShotLoss / N_TRIALS;
  WorkNN.Free;
  WriteLn(' done.');
  WriteLn;

  EndTime := Now;

  // ====================================================================
  // Report.
  // ====================================================================
  WriteLn('=== IMP per-round sparsity / accuracy ===');
  WriteLn('round,sparsity_pct,imp_acc,imp_loss');
  for FI := 1 to N_ROUNDS do
    WriteLn(FI, ',',
      FloatToStrF(Rounds[FI].Sparsity*100, ffFixed, 5, 1), ',',
      FloatToStrF(Rounds[FI].Acc,  ffFixed, 6, 4), ',',
      FloatToStrF(Rounds[FI].Loss, ffFixed, 6, 4));
  WriteLn;
  WriteLn('=== Head-to-head at the final (~95%) sparsity ===');
  WriteLn('method,sparsity_pct,acc,loss');
  WriteLn('dense_ref,0.0,',
      FloatToStrF(DenseAcc, ffFixed, 6, 4), ',',
      FloatToStrF(DenseLoss, ffFixed, 6, 4));
  WriteLn('one_shot,',
      FloatToStrF(OneShotSparsity*100, ffFixed, 5, 1), ',',
      FloatToStrF(OneShotAcc, ffFixed, 6, 4), ',',
      FloatToStrF(OneShotLoss, ffFixed, 6, 4));
  WriteLn('imp,',
      FloatToStrF(IMPSparsity*100, ffFixed, 5, 1), ',',
      FloatToStrF(IMPAcc, ffFixed, 6, 4), ',',
      FloatToStrF(IMPLoss, ffFixed, 6, 4));
  WriteLn;

  // ---- grade it (honest) ----
  Margin := IMPAcc - OneShotAcc;
  WriteLn('=== Verdict ===');
  WriteLn('IMP acc - one-shot acc = ',
          FloatToStrF(Margin*100, ffFixed, 6, 2), ' accuracy points',
          ' (at ', FloatToStrF(IMPSparsity*100, ffFixed, 4, 1), '% sparsity).');
  if Margin > 0.03 then
    Verdict := 'PASS: IMP recovers a clearly better winning ticket than one-shot'
             + ' at this extreme sparsity (>3 acc points).'
  else if Margin > 0.005 then
    Verdict := 'WEAK PASS: IMP edges out one-shot, but only by a small margin'
             + ' on this toy.'
  else if Margin >= -0.005 then
    Verdict := 'NO CLEAR GAP: IMP and one-shot tie within noise on this toy.'
             + ' The task is likely too easy / the net too small for the'
             + ' iterative-vs-one-shot gap to open here -- reported honestly.'
  else
    Verdict := 'INCONCLUSIVE: one-shot beat IMP this seed (toy noise);'
             + ' no IMP advantage demonstrated.';
  WriteLn(Verdict);
  WriteLn;
  WriteLn('Honest note: the sibling one-shot example collapses to ~67% at 95%');
  WriteLn('sparsity. IMP is the paper''s remedy, but on a net this small and a');
  WriteLn('task this easy the remedy may or may not visibly help -- the printed');
  WriteLn('margin above is the actual measured outcome, not an assumed result.');
  WriteLn;
  WriteLn('Total wall time: ',
          FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');

  Theta0NN.Free;
  DenseNN.Free;
  Test.Free;
  Train.Free;
end;

begin
  RandSeed := RAND_SEED;
  GMaskNN := nil;
  GEnforcer := TMaskEnforcer.Create;
  try
    RunAlgo();
  finally
    GEnforcer.Free;
  end;
end.
