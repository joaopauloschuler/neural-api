program HyenaOperator;
(*
HyenaOperator: a long-range recall bake-off of the attention-free Hyena
operator (TNNet.AddHyenaOperator, built on TNNetImplicitLongConv) against a
single-head self-attention block, on a synthetic task where the implicit long
convolution's GLOBAL receptive field should shine.

Task (long-range recall):
  Each example is a length-SEQ_LEN sequence of D_MODEL-channel tokens. Channel 0
  carries a random "payload" value, but it is non-zero at ONLY ONE early random
  position p (0 .. SEQ_LEN div 3). Channel 1 is a constant "query" marker that
  is 1 at the LAST position and 0 elsewhere. Every other channel is small noise.
  The network must copy the early payload to the LAST position's channel-0
  output: target[t,0,0] = payload if t = SEQ_LEN-1 else 0. Success therefore
  requires moving information across the WHOLE sequence - exactly the regime
  where a full-length filter / global attention beats a short local kernel.

Two arms, trained with the SAME data, seed, learning rate and epochs:
  - HYENA     : Input -> AddHyenaOperator(D_MODEL, HIDDEN) -> pointwise readout
  - ATTENTION : Input -> [Q|K|V pointwise proj] -> single-head SDPA -> readout

Both arms are sequence mixers over a (SEQ_LEN,1,D_MODEL) tensor and use only
token-wise (1x1 conv) projections. The implicit-MLP filter parametrization makes
Hyena the LIGHTER model (its trainable-weight count does not grow with SeqLen);
both counts are reported in the final table so the param/compute trade-off is
explicit. We print a per-epoch training
MSE trace for each arm plus a final comparison table (final train MSE, held-out
recall MSE, trainable-weight count). Printing is guarded against NaN/Inf.

This is a small CPU toy (well under a minute); it demonstrates the layer/builder
rather than chasing SOTA. Pairs naturally with the downstream
../gpt-3-for-pascal decoder as an attention-free block.

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
  SEQ_LEN    = 16;
  D_MODEL    = 4;
  HIDDEN     = 8;     // implicit-MLP width of the long conv
  TRAIN_SIZE = 256;
  TEST_SIZE  = 64;
  NUM_EPOCHS = 60;
  LR         = 0.02;
  SEED       = 42;

type
  TArmKind = (akHyena, akAttention);

  TSample = record
    X: TNNetVolume;       // (SEQ_LEN,1,D_MODEL) input
    Y: TNNetVolume;       // (SEQ_LEN,1,D_MODEL) target (only [last,0,0] matters)
  end;

var
  TrainSet, TestSet: array of TSample;

function SafeF(v: TNeuralFloat): string;
begin
  if IsNan(v) or IsInfinite(v)
    then Result := '   nan/inf'
    else Result := Format('%10.6f', [v]);
end;

procedure MakeSample(var S: TSample);
var
  t, c, p: integer;
  payload: TNeuralFloat;
begin
  S.X := TNNetVolume.Create(SEQ_LEN, 1, D_MODEL);
  S.Y := TNNetVolume.Create(SEQ_LEN, 1, D_MODEL);
  S.X.Fill(0);
  S.Y.Fill(0);
  // Small noise on the "spare" channels so the readout cannot cheat with a bias.
  for t := 0 to SEQ_LEN - 1 do
    for c := 2 to D_MODEL - 1 do
      S.X[t, 0, c] := (Random - 0.5) * 0.1;
  // Payload at an early random position p, recalled at the last position.
  p := Random(SEQ_LEN div 3 + 1);
  payload := (Random - 0.5) * 2.0;     // in [-1,1]
  S.X[p, 0, 0] := payload;
  S.X[SEQ_LEN - 1, 0, 1] := 1.0;       // query marker at the end
  S.Y[SEQ_LEN - 1, 0, 0] := payload;
end;

procedure BuildData;
var i: integer;
begin
  SetLength(TrainSet, TRAIN_SIZE);
  SetLength(TestSet, TEST_SIZE);
  for i := 0 to TRAIN_SIZE - 1 do MakeSample(TrainSet[i]);
  for i := 0 to TEST_SIZE - 1 do MakeSample(TestSet[i]);
end;

procedure FreeData;
var i: integer;
begin
  for i := 0 to High(TrainSet) do begin TrainSet[i].X.Free; TrainSet[i].Y.Free; end;
  for i := 0 to High(TestSet) do begin TestSet[i].X.Free; TestSet[i].Y.Free; end;
end;

function BuildNet(Kind: TArmKind): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(SEQ_LEN, 1, D_MODEL, 1));
  case Kind of
    akHyena:
      // Attention-free long-conv sequence mixer.
      Result.AddHyenaOperator(D_MODEL, HIDDEN);
    akAttention:
      begin
        // Single-head self-attention baseline: token-wise Q|K|V projection
        // into a 3*D_MODEL slab, one SDPA head (causal), then out-projection.
        Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * D_MODEL));
        Result.AddSelfAttention(1);
        Result.AddLayer(TNNetPointwiseConvLinear.Create(D_MODEL));
      end;
  end;
  // Shared token-wise readout to the D_MODEL target channels.
  Result.AddLayer(TNNetPointwiseConvLinear.Create(D_MODEL));
end;

// Recall MSE measured ONLY on the (last position, channel 0) cell - the cell
// the task actually scores.
function RecallMSE(NN: TNNet; const Data: array of TSample): TNeuralFloat;
var
  i: integer;
  diff, acc: TNeuralFloat;
begin
  acc := 0;
  for i := 0 to High(Data) do
  begin
    NN.Compute(Data[i].X);
    diff := NN.GetLastLayer.Output[SEQ_LEN - 1, 0, 0] - Data[i].Y[SEQ_LEN - 1, 0, 0];
    acc := acc + diff * diff;
  end;
  Result := acc / Length(Data);
end;

function TrainArm(Kind: TArmKind; const ArmName: string;
  out FinalTrain, FinalTest: TNeuralFloat; out WeightCnt: integer): TNNet;
var
  NN: TNNet;
  epoch, i: integer;
  trainMSE: TNeuralFloat;
begin
  RandSeed := SEED;
  NN := BuildNet(Kind);
  NN.SetLearningRate(LR, 0.0);
  NN.SetBatchUpdate(false);
  WeightCnt := NN.CountWeights();
  WriteLn;
  WriteLn('=== ', ArmName, ' === (', WeightCnt, ' trainable weights)');
  for epoch := 0 to NUM_EPOCHS - 1 do
  begin
    NN.ClearDeltas();
    for i := 0 to High(TrainSet) do
    begin
      NN.Compute(TrainSet[i].X);
      NN.Backpropagate(TrainSet[i].Y);
    end;
    if (epoch mod 10 = 0) or (epoch = NUM_EPOCHS - 1) then
    begin
      trainMSE := RecallMSE(NN, TrainSet);
      WriteLn('  epoch ', epoch:3, '  train recall MSE = ', SafeF(trainMSE));
    end;
  end;
  FinalTrain := RecallMSE(NN, TrainSet);
  FinalTest := RecallMSE(NN, TestSet);
  Result := NN;
end;

var
  HyenaNet, AttnNet: TNNet;
  hTrain, hTest, aTrain, aTest: TNeuralFloat;
  hW, aW: integer;
begin
  // Mask FPU exceptions so a diverging arm reports a NaN/Inf instead of crashing.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
    exUnderflow, exPrecision]);
  RandSeed := SEED;
  WriteLn('HyenaOperator long-range recall bake-off');
  WriteLn('  SeqLen=', SEQ_LEN, ' d_model=', D_MODEL, ' hidden=', HIDDEN,
    ' train=', TRAIN_SIZE, ' test=', TEST_SIZE, ' epochs=', NUM_EPOCHS);
  BuildData;
  try
    HyenaNet := TrainArm(akHyena, 'HYENA (implicit long conv)',
      hTrain, hTest, hW);
    AttnNet := TrainArm(akAttention, 'ATTENTION (single-head SDPA)',
      aTrain, aTest, aW);
    try
      WriteLn;
      WriteLn('==================== FINAL COMPARISON ====================');
      WriteLn('arm          weights   train recall MSE   test recall MSE');
      WriteLn('HYENA        ', hW:7, '   ', SafeF(hTrain), '       ', SafeF(hTest));
      WriteLn('ATTENTION    ', aW:7, '   ', SafeF(aTrain), '       ', SafeF(aTest));
      WriteLn('==========================================================');
      if hTest < aTest then
        WriteLn('Verdict: the attention-free Hyena operator generalised better ',
          'on this long-range recall toy.')
      else
        WriteLn('Verdict: single-head attention matched or beat Hyena here ',
          '(small toy; tune SeqLen/hidden/epochs to explore the regime).');
    finally
      AttnNet.Free;
      HyenaNet.Free;
    end;
  finally
    FreeData;
  end;
end.
