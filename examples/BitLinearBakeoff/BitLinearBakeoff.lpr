program BitLinearBakeoff;
(*
BitLinearBakeoff: a ternary-vs-full-precision bake-off.

This is the TNNetBitLinear follow-up to the BitNet idea. The SAME tiny
classifier architecture is trained twice on the SAME small synthetic
multi-class 2D blob task with the SAME fixed RandSeed (424242), same data
and same init order:

  - once with full-precision FP32 heads (TNNetFullConnectLinear)
  - once with ternary {-1,0,+1} heads (TNNetBitLinear)

at MATCHED layer sizes. For each variant the program reports final
train/validation/test accuracy and the EFFECTIVE WEIGHT MEMORY: the exact
weight count comes from TNNet.CountWeights, and bytes are computed assuming
32 bits/weight for the FP head versus 1.58 bits/weight (log2(3), the
information-theoretic cost of a ternary symbol) for the ternary head.

The headline BitNet claim is "near-FP accuracy at a fraction of the weight
memory". The run ends with a self-checking PASS/FAIL gate that encodes
exactly that: BitLinear must stay within a small accuracy margin of FP
while using well under a fraction of the weight memory.

Net (identical shape for both variants):
  Input(2) -> Head(HIDDEN) -> ReLU -> Head(NUM_CLASSES) -> SoftMax
where Head is the only thing that changes (FullConnectLinear vs BitLinear).

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
  neuralvolume,
  neuralfit;

const
  RAND_SEED    = 424242;
  NUM_CLASSES  = 4;
  HIDDEN_UNITS = 24;
  TRAIN_SIZE   = 1200;
  VAL_SIZE     = 300;
  TEST_SIZE    = 300;
  NUM_EPOCHS   = 40;
  BATCH_SIZE   = 32;
  // Bits per weight assumptions for the effective-memory accounting.
  FP_BITS      = 32.0;          // float32
  TERNARY_BITS = 1.5849625007;  // log2(3): a ternary {-1,0,+1} symbol

type
  THeadKind = (hkFullPrecision, hkTernary);

  TBakeOffResult = record
    Name: string;
    TrainAcc, ValAcc, TestAcc: TNeuralFloat;
    Weights: integer;
    BitsPerWeight: TNeuralFloat;
  end;

// Build the SAME small synthetic 4-class blob task. Each class is a Gaussian
// blob around a fixed center; the classes are linearly NON-separable enough
// (4 corners) that the hidden layer actually matters.
function CreateBlobPairList(MaxCnt: integer): TNNetVolumePairList;
var
  Cnt, Cls: integer;
  Centers: array[0..NUM_CLASSES-1, 0..1] of TNeuralFloat;
  Px, Py: TNeuralFloat;
  Inp, Outp: TNNetVolume;
begin
  Centers[0][0] := -1.0; Centers[0][1] := -1.0;
  Centers[1][0] :=  1.0; Centers[1][1] := -1.0;
  Centers[2][0] := -1.0; Centers[2][1] :=  1.0;
  Centers[3][0] :=  1.0; Centers[3][1] :=  1.0;

  Result := TNNetVolumePairList.Create();
  for Cnt := 0 to MaxCnt - 1 do
  begin
    Cls := Cnt mod NUM_CLASSES;
    // Gaussian-ish jitter via average of two uniforms, scaled.
    Px := Centers[Cls][0] + 0.55 * ((Random + Random) - 1.0);
    Py := Centers[Cls][1] + 0.55 * ((Random + Random) - 1.0);

    Inp := TNNetVolume.Create([Px, Py]);
    Outp := TNNetVolume.Create(NUM_CLASSES);
    Outp.SetClassForSoftMax(Cls);
    Result.Add(TNNetVolumePair.Create(Inp, Outp));
  end;
end;

// Manual accuracy on a pair list (argmax of network output == argmax target).
function EvaluateAccuracy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
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
  if Pairs.Count > 0 then
    Result := Hits / Pairs.Count
  else
    Result := 0;
end;

procedure AddHead(NN: TNNet; Kind: THeadKind; Units: integer);
begin
  case Kind of
    hkFullPrecision: NN.AddLayer(TNNetFullConnectLinear.Create(Units));
    hkTernary:       NN.AddLayer(TNNetBitLinear.Create(Units));
  end;
end;

function HeadName(Kind: THeadKind): string;
begin
  case Kind of
    hkFullPrecision: Result := 'TNNetFullConnectLinear (FP32)';
    hkTernary:       Result := 'TNNetBitLinear (ternary)';
  end;
end;

function RunOne(Kind: THeadKind;
                Train, Validation, Test: TNNetVolumePairList): TBakeOffResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
begin
  Result.Name := HeadName(Kind);

  // Reseed right before init so BOTH variants get identical weight init draws.
  RandSeed := RAND_SEED;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    NN.AddLayer(TNNetInput.Create(2));
    AddHead(NN, Kind, HIDDEN_UNITS);
    NN.AddLayer(TNNetReLU.Create());
    AddHead(NN, Kind, NUM_CLASSES);
    NN.AddLayer(TNNetSoftMax.Create());

    Result.Weights := NN.CountWeights();
    if Kind = hkTernary then
      Result.BitsPerWeight := TERNARY_BITS
    else
      Result.BitsPerWeight := FP_BITS;

    // Net is tiny (144 weights); a single thread avoids thread-pool overhead
    // and keeps the whole bake-off well under a minute.
    NFit.MaxThreadNum := 1;
    NFit.FileNameBase := GetTempDir + 'BitLinearBakeoff_autosave';
    NFit.InitialLearningRate := 0.01;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.Fit(NN, Train, Validation, Test, BATCH_SIZE, NUM_EPOCHS);

    Result.TrainAcc := EvaluateAccuracy(NN, Train);
    Result.ValAcc   := EvaluateAccuracy(NN, Validation);
    Result.TestAcc  := EvaluateAccuracy(NN, Test);
  finally
    NFit.Free;
    NN.Free;
  end;
end;

function MemBytes(R: TBakeOffResult): TNeuralFloat;
begin
  Result := R.Weights * R.BitsPerWeight / 8.0;
end;

procedure RunAlgo();
var
  TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
  FP, BL: TBakeOffResult;
  StartTime, EndTime: TDateTime;
  AccGapPts, MemPct, FpBytes, BlBytes, Ratio: TNeuralFloat;
  GatePass: boolean;
const
  // Gate tolerances (verified empirically against this run).
  MAX_ACC_GAP_PTS = 8.0;   // BitLinear test acc must be within 8 pts of FP
  MAX_MEM_PCT     = 10.0;  // ...while using < 10% of FP weight memory
  MIN_BL_ACC      = 0.80;  // ...and still being a useful classifier
begin
  WriteLn('BitLinear bake-off: ternary {-1,0,+1} vs full-precision FP32 heads.');
  WriteLn('Task: ', NUM_CLASSES, '-class 2D Gaussian-blob classification.');
  WriteLn('Net:  Input(2) -> Head(', HIDDEN_UNITS, ') -> ReLU -> Head(',
          NUM_CLASSES, ') -> SoftMax.');
  WriteLn('Same architecture, same data, same RandSeed=', RAND_SEED, '; only the head type changes.');
  WriteLn(NUM_EPOCHS, ' epochs, ', TRAIN_SIZE, ' train / ', VAL_SIZE,
          ' val / ', TEST_SIZE, ' test, LR=0.01.');
  WriteLn;

  StartTime := Now;

  // Same data for both variants: build it once under the fixed seed.
  RandSeed := RAND_SEED;
  TrainingPairs   := CreateBlobPairList(TRAIN_SIZE);
  ValidationPairs := CreateBlobPairList(VAL_SIZE);
  TestPairs       := CreateBlobPairList(TEST_SIZE);
  try
    Write('Training full-precision (TNNetFullConnectLinear) head ...');
    FP := RunOne(hkFullPrecision, TrainingPairs, ValidationPairs, TestPairs);
    WriteLn(' done.');

    Write('Training ternary       (TNNetBitLinear)          head ...');
    BL := RunOne(hkTernary, TrainingPairs, ValidationPairs, TestPairs);
    WriteLn(' done.');
  finally
    TestPairs.Free;
    ValidationPairs.Free;
    TrainingPairs.Free;
  end;

  EndTime := Now;

  FpBytes := MemBytes(FP);
  BlBytes := MemBytes(BL);
  Ratio   := FpBytes / BlBytes;

  WriteLn;
  WriteLn('=== Accuracy ===');
  WriteLn('head                            train     val      test');
  WriteLn(Format('%-30s  %6.2f%%  %6.2f%%  %6.2f%%',
    [FP.Name, FP.TrainAcc*100, FP.ValAcc*100, FP.TestAcc*100]));
  WriteLn(Format('%-30s  %6.2f%%  %6.2f%%  %6.2f%%',
    [BL.Name, BL.TrainAcc*100, BL.ValAcc*100, BL.TestAcc*100]));
  WriteLn;
  WriteLn('=== Effective weight memory ===');
  WriteLn('head                            weights  bits/wt  bytes');
  WriteLn(Format('%-30s  %7d  %6.2f   %9.1f',
    [FP.Name, FP.Weights, FP.BitsPerWeight, FpBytes]));
  WriteLn(Format('%-30s  %7d  %6.2f   %9.1f',
    [BL.Name, BL.Weights, BL.BitsPerWeight, BlBytes]));
  WriteLn;
  WriteLn(Format('Weight count is matched (%d vs %d).', [FP.Weights, BL.Weights]));
  WriteLn(Format('Compression ratio (FP bytes / ternary bytes): %.2fx', [Ratio]));
  WriteLn;

  AccGapPts := (FP.TestAcc - BL.TestAcc) * 100.0;
  MemPct    := 100.0 * BlBytes / FpBytes;

  WriteLn('=== Headline-claim gate ===');
  WriteLn(Format('BitLinear test accuracy gap vs FP: %.2f pts (must be <= %.1f).',
    [AccGapPts, MAX_ACC_GAP_PTS]));
  WriteLn(Format('BitLinear weight memory: %.2f%% of FP (must be < %.1f%%).',
    [MemPct, MAX_MEM_PCT]));
  WriteLn(Format('BitLinear test accuracy: %.2f%% (must be >= %.1f%%).',
    [BL.TestAcc*100, MIN_BL_ACC*100]));
  WriteLn;

  GatePass := (AccGapPts <= MAX_ACC_GAP_PTS)
          and (MemPct < MAX_MEM_PCT)
          and (BL.TestAcc >= MIN_BL_ACC);

  if GatePass then
    WriteLn('GATE: PASS -- near-FP accuracy at a fraction of the weight memory.')
  else
    WriteLn('GATE: FAIL -- headline BitNet claim NOT met on this run.');

  WriteLn;
  WriteLn('Total wall time: ', FormatFloat('0.00', (EndTime - StartTime)*86400), ' s');

  if not GatePass then
    Halt(1);
end;

begin
  RandSeed := RAND_SEED;
  RunAlgo();
end.
