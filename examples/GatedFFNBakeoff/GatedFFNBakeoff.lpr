program GatedFFNBakeoff;
(*
GatedFFNBakeoff: a head-to-head bake-off of the five in-tree gated
feed-forward activation layers. The SAME transformer-style feed-forward
block is built five times on the SAME tiny synthetic sequence task; the
ONLY thing that differs between arms is which gating layer sits in the
middle of the "Dense -> GATE -> Dense" sandwich:

  - TNNetGLU      value * sigmoid(gate)        (https://arxiv.org/abs/1612.08083)
  - TNNetReGLU    ReLU(value) * gate           (https://arxiv.org/abs/2002.05202)
  - TNNetGEGLU    value * GELU(gate)           (https://arxiv.org/abs/2002.05202)
  - TNNetSwiGLU   value * Swish(gate)          (https://arxiv.org/abs/2002.05202)
  - TNNetTanhGLU  value * tanh(gate)

All five gates share the depth-doubling convention used by the
single-layer GLU/GEGLU/SwiGLU demos: the FullConnectLinear in front of
the gate emits 2*d_ff channels (gate || value packed on depth), the gate
halves that back to d_ff, and the read-out projects d_ff -> d_model. The
five arms therefore have IDENTICAL parameter counts (the gates are all
parameter-free) and identical wiring; only the gate non-linearity moves.

Task: a tiny synthetic sequence-to-sequence regression. Each sample is a
length-cSeqLen sequence of cDModel-dim vectors drawn from [-1, 1]; the
target at each position is a fixed, deterministic, mildly-nonlinear
function of that position's features (a trig term plus a feature product
plus a linear term). It is applied independently per position, so the
FFN block is exactly the right tool and every arm can learn it.

Every arm reseeds RandSeed to the same value before generating its data
and before building/initialising its net, so all arms see identical
inputs and identical weight init; only the gate differs.

The printed ranking is SEED-DEPENDENT. The point of this program is the
comparison harness ("all five gates work, here is how they stack up on
one task at one seed"), NOT to crown a universal winner.

Pure CPU, no external dataset, finishes in a few seconds.

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
  cSeqLen     = 6;      // positions per sequence
  cDModel     = 4;      // features per position (input depth)
  cDFF        = 16;     // hidden width of the FFN block
  cTrain      = 600;    // training sequences
  cVal        = 200;    // validation sequences
  cEpochs     = 25;
  cLearnRate  = 0.01;
  cInertia    = 0.9;
  cSeed       = 42;
  cConvThresh = 0.02;   // MSE below this counts as "converged"

type
  TGateKind = (gkGLU, gkReGLU, gkGEGLU, gkSwiGLU, gkTanhGLU);

  TArm = record
    Name: string;
    Kind: TGateKind;
  end;

  TArmResult = record
    Name        : string;
    InitLoss    : TNeuralFloat;   // MSE before any training
    FinalLoss   : TNeuralFloat;   // MSE after training
    Seconds     : TNeuralFloat;   // wall-clock for the arm
    EpochsToConv: integer;        // first epoch with val MSE < cConvThresh, else -1
  end;

const
  cArms: array[0..4] of TArm =
  (
    (Name: 'TNNetGLU';     Kind: gkGLU),
    (Name: 'TNNetReGLU';   Kind: gkReGLU),
    (Name: 'TNNetGEGLU';   Kind: gkGEGLU),
    (Name: 'TNNetSwiGLU';  Kind: gkSwiGLU),
    (Name: 'TNNetTanhGLU'; Kind: gkTanhGLU)
  );

function RandUniform: TNeuralFloat;
begin
  Result := (Random * 2.0) - 1.0;
end;

// Deterministic, learnable, mildly nonlinear per-position target.
function TargetFn(x0, x1, x2, x3: TNeuralFloat): TNeuralFloat;
begin
  Result := Sin(x0) + 0.5 * x1 * x2 - 0.3 * x3;
end;

// One sample: a (cSeqLen x 1 x cDModel) input sequence and a
// (cSeqLen x 1 x 1) target sequence (one regression value per position).
procedure MakeSample(out X, Y: TNNetVolume);
var
  Pos, C: integer;
  f: array[0..cDModel - 1] of TNeuralFloat;
begin
  X := TNNetVolume.Create(cSeqLen, 1, cDModel);
  Y := TNNetVolume.Create(cSeqLen, 1, 1);
  for Pos := 0 to cSeqLen - 1 do
  begin
    for C := 0 to cDModel - 1 do
    begin
      f[C] := RandUniform;
      X.Data[Pos, 0, C] := f[C];
    end;
    Y.Data[Pos, 0, 0] := TargetFn(f[0], f[1], f[2], f[3]);
  end;
end;

procedure BuildSet(out Pairs: TNNetVolumePairList; Count: integer);
var
  I: integer;
  X, Y: TNNetVolume;
begin
  Pairs := TNNetVolumePairList.Create();
  for I := 1 to Count do
  begin
    MakeSample(X, Y);
    Pairs.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// Shared FFN block; the ONLY thing that changes is the gate layer.
// FullConnectLinear emits 2*d_ff channels per position (gate || value),
// the gate halves it to d_ff, the read-out projects d_ff -> 1.
procedure BuildNet(out NN: TNNet; Arm: TArm);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cSeqLen, 1, cDModel));
  NN.AddLayer(TNNetPointwiseConvLinear.Create(2 * cDFF));

  case Arm.Kind of
    gkGLU    : NN.AddLayer(TNNetGLU.Create());
    gkReGLU  : NN.AddLayer(TNNetReGLU.Create());
    gkGEGLU  : NN.AddLayer(TNNetGEGLU.Create());
    gkSwiGLU : NN.AddLayer(TNNetSwiGLU.Create());
    gkTanhGLU: NN.AddLayer(TNNetTanhGLU.Create());
  end;

  NN.AddLayer(TNNetPointwiseConvLinear.Create(1));
end;

// Mean squared error over a pair list (mean over all positions).
function MeanMSE(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I, Pos: integer;
  P: TNNetVolume;
  Sum: Double;
  Diff: TNeuralFloat;
begin
  Sum := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    P := NN.GetLastLayer().Output;
    for Pos := 0 to cSeqLen - 1 do
    begin
      Diff := P.Data[Pos, 0, 0] - Pairs[I].O.Data[Pos, 0, 0];
      Sum := Sum + Diff * Diff;
    end;
  end;
  if Pairs.Count > 0 then
    Result := Sum / (Pairs.Count * cSeqLen)
  else
    Result := 0;
end;

function RunArm(Arm: TArm): TArmResult;
var
  NN: TNNet;
  TrainSet, ValSet: TNNetVolumePairList;
  Epoch, I: integer;
  ValMSE: TNeuralFloat;
  StartTime, EndTime: TDateTime;
begin
  Result.Name := Arm.Name;
  Result.EpochsToConv := -1;

  // Reseed BEFORE data gen and BEFORE net build so every arm sees the
  // same inputs and the same weight initialisation.
  RandSeed := cSeed;
  BuildSet(TrainSet, cTrain);
  BuildSet(ValSet, cVal);

  RandSeed := cSeed;
  BuildNet(NN, Arm);
  try
    NN.SetLearningRate(cLearnRate, cInertia);
    Result.InitLoss := MeanMSE(NN, ValSet);

    StartTime := Now;
    for Epoch := 1 to cEpochs do
    begin
      for I := 0 to TrainSet.Count - 1 do
      begin
        NN.Compute(TrainSet[I].I);
        NN.Backpropagate(TrainSet[I].O);
      end;
      ValMSE := MeanMSE(NN, ValSet);
      if (Result.EpochsToConv < 0) and (not IsNan(ValMSE)) and
         (not IsInfinite(ValMSE)) and (ValMSE < cConvThresh) then
        Result.EpochsToConv := Epoch;
    end;
    EndTime := Now;

    Result.Seconds := (EndTime - StartTime) * 86400.0;
    Result.FinalLoss := MeanMSE(NN, ValSet);
  finally
    NN.Free;
    ValSet.Free;
    TrainSet.Free;
  end;
end;

function SafeF(V: TNeuralFloat; Dec: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then
    Result := 'NaN'
  else
    Result := FloatToStrF(V, ffFixed, 8, Dec);
end;

function ConvStr(R: TArmResult): string;
begin
  if R.EpochsToConv > 0 then
    Result := IntToStr(R.EpochsToConv)
  else
    Result := '>' + IntToStr(cEpochs);  // did not cross the threshold
end;

procedure RunBakeoff();
var
  K: integer;
  Results: array[0..High(cArms)] of TArmResult;
  AllFinite, AllLearned: boolean;
  R: TArmResult;
begin
  WriteLn('Gated feed-forward bake-off: five gates, one synthetic sequence task.');
  WriteLn('Block: Input(', cSeqLen, 'x1x', cDModel, ')',
          ' -> PointwiseConvLinear(', 2 * cDFF, ')',
          ' -> [GATE -> ', cDFF, '] -> PointwiseConvLinear(1)');
  WriteLn('Per-position target: y = sin(x0) + 0.5*x1*x2 - 0.3*x3');
  WriteLn('Train=', cTrain, ' Val=', cVal, ' seqs  SeqLen=', cSeqLen,
          '  Epochs=', cEpochs, '  LR=', SafeF(cLearnRate, 3),
          '  RandSeed=', cSeed);
  WriteLn('Same data, same init, same FFN wiring; only the gate layer changes.');
  WriteLn('Identical parameter counts across arms (all gates are parameter-free).');
  WriteLn;

  for K := 0 to High(cArms) do
  begin
    Write('Training ', cArms[K].Name, ' ...');
    Results[K] := RunArm(cArms[K]);
    WriteLn(' done.  final_mse=', SafeF(Results[K].FinalLoss, 4),
            '  ', SafeF(Results[K].Seconds, 2), 's');
  end;

  WriteLn;
  WriteLn('=== Comparison (val MSE, wall-clock, epochs-to-converge < ',
          SafeF(cConvThresh, 2), ') ===');
  WriteLn(Format('%-14s %10s %10s %9s %12s',
          ['gate', 'init_mse', 'final_mse', 'seconds', 'epochs_conv']));
  for K := 0 to High(cArms) do
  begin
    R := Results[K];
    WriteLn(Format('%-14s %10s %10s %9s %12s',
            [R.Name, SafeF(R.InitLoss, 4), SafeF(R.FinalLoss, 4),
             SafeF(R.Seconds, 2), ConvStr(R)]));
  end;

  // --- Sanity checks ---
  WriteLn;
  WriteLn('=== Sanity checks ===');
  AllFinite := True;
  AllLearned := True;
  for K := 0 to High(cArms) do
  begin
    R := Results[K];
    if IsNan(R.FinalLoss) or IsInfinite(R.FinalLoss) then
      AllFinite := False;
    if not (R.FinalLoss < R.InitLoss) then
      AllLearned := False;
  end;

  if AllFinite then
    WriteLn('[PASS] all 5 arms produced a finite (no NaN/Inf) final loss.')
  else
    WriteLn('[FAIL] at least one arm produced NaN/Inf final loss.');

  if AllLearned then
    WriteLn('[PASS] all 5 arms reduced loss below their pre-training baseline.')
  else
    WriteLn('[FAIL] at least one arm did not improve over its baseline.');

  WriteLn;
  WriteLn('Note: the per-arm ranking above is seed-dependent. This harness shows');
  WriteLn('that all five gates train on this task; it does not crown a winner.');
end;

begin
  RandSeed := cSeed;
  RunBakeoff();
end.
