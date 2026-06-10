program MLPMixer;
(*
MLPMixer: a tiny all-MLP token-classification demo of TNNet.AddMLPMixerBlock,
the attention-free MLP-Mixer block (Tolstikhin et al. 2021,
"MLP-Mixer: An all-MLP Architecture for Vision", https://arxiv.org/abs/2105.01601).

A Mixer block replaces self-attention with TWO pre-LayerNorm residual MLPs over
a (Tokens,1,Channels) sequence:
  - a TOKEN-mixing MLP that mixes information ACROSS token positions (shared
    over channels, implemented by transposing the token/channel axes and running
    a pointwise MLP over the new Depth = Tokens axis), and
  - a CHANNEL-mixing MLP, a standard per-token pointwise FFN over the channels.

Task (which-half classification, needs token mixing):
  Each example is a length-SEQ_LEN sequence of CHANNELS-channel tokens filled
  with small noise. A single "spike" of magnitude SPIKE is planted on channel 0
  at one random position. The label is 0 if that spike sits in the FIRST half of
  the sequence and 1 if it sits in the SECOND half. Deciding the class therefore
  requires comparing information ACROSS token positions - exactly what the
  token-mixing MLP provides; a purely channel-wise (per-token) network cannot
  solve it. The all-MLP Mixer stack learns it in a few seconds on CPU.

Network:
  Input(SEQ_LEN,1,CHANNELS)
    -> PointwiseConvLinear(CHANNELS)          (token-wise stem / embedding)
    -> AddMLPMixerBlock(TOKENS_HIDDEN, CHANNELS_HIDDEN, ReLU)  x NUM_BLOCKS
    -> LayerNorm
    -> AvgChannel  (mean-pool over tokens -> a single CHANNELS vector)
    -> FullConnectLinear(NUM_CLASSES) -> SoftMax

We print a per-epoch train loss + train/test accuracy trace; it converges to
near-perfect accuracy. This is a small CPU toy (well under a minute) that
demonstrates the builder rather than chasing SOTA. Printing is NaN/Inf-guarded.

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
  SEQ_LEN         = 12;
  CHANNELS        = 8;
  TOKENS_HIDDEN   = 16;    // token-mixing MLP hidden width
  CHANNELS_HIDDEN = 16;    // channel-mixing MLP hidden width
  NUM_BLOCKS      = 2;
  NUM_CLASSES     = 2;
  TRAIN_SIZE      = 400;
  TEST_SIZE       = 100;
  NUM_EPOCHS      = 60;
  LR              = 0.01;
  SPIKE           = 2.0;
  SEED            = 42;

type
  TSample = record
    X: TNNetVolume;       // (SEQ_LEN,1,CHANNELS) input
    Y: TNNetVolume;       // (NUM_CLASSES) one-hot target
    Cls: integer;
  end;

var
  TrainSet, TestSet: array of TSample;

function SafeF(v: TNeuralFloat): string;
begin
  if IsNan(v) or IsInfinite(v)
    then Result := '   nan/inf'
    else Result := Format('%8.5f', [v]);
end;

procedure MakeSample(var S: TSample);
var
  t, c, p: integer;
begin
  S.X := TNNetVolume.Create(SEQ_LEN, 1, CHANNELS);
  S.Y := TNNetVolume.Create(NUM_CLASSES);
  S.X.Fill(0);
  S.Y.Fill(0);
  // Small background noise on every token/channel so the readout cannot cheat.
  for t := 0 to SEQ_LEN - 1 do
    for c := 0 to CHANNELS - 1 do
      S.X[t, 0, c] := (Random - 0.5) * 0.2;
  // Plant a spike at one random position; label = which half it lands in.
  p := Random(SEQ_LEN);
  S.X[p, 0, 0] := SPIKE;
  if p < SEQ_LEN div 2 then S.Cls := 0 else S.Cls := 1;
  S.Y.Raw[S.Cls] := 1.0;
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

function BuildNet: TNNet;
var b: integer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(SEQ_LEN, 1, CHANNELS, 1));
  // Token-wise embedding stem (keeps the (SEQ_LEN,1,CHANNELS) shape).
  Result.AddLayer(TNNetPointwiseConvLinear.Create(CHANNELS));
  // The all-MLP Mixer stack: each block mixes tokens then channels.
  for b := 0 to NUM_BLOCKS - 1 do
    Result.AddMLPMixerBlock(TOKENS_HIDDEN, CHANNELS_HIDDEN, TNNetReLU);
  Result.AddLayer(TNNetLayerNorm.Create());
  // Mean-pool over tokens -> one CHANNELS-vector, then classify.
  Result.AddLayer(TNNetAvgChannel.Create());
  Result.AddLayer(TNNetFullConnectLinear.Create(NUM_CLASSES));
  Result.AddLayer(TNNetSoftMax.Create());
  Result.InitWeights();
end;

function Accuracy(NN: TNNet; const Data: array of TSample): TNeuralFloat;
var
  i, hit: integer;
begin
  hit := 0;
  for i := 0 to High(Data) do
  begin
    NN.Compute(Data[i].X);
    if NN.GetLastLayer.Output.GetClass() = Data[i].Cls then Inc(hit);
  end;
  Result := hit / Length(Data);
end;

var
  NN: TNNet;
  epoch, i: integer;
  loss, trainAcc, testAcc: TNeuralFloat;
begin
  // Mask FPU exceptions so a diverging run reports NaN/Inf instead of crashing.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide, exOverflow,
    exUnderflow, exPrecision]);
  RandSeed := SEED;
  WriteLn('MLP-Mixer which-half token-classification demo');
  WriteLn('  SeqLen=', SEQ_LEN, ' Channels=', CHANNELS,
    ' blocks=', NUM_BLOCKS, ' tokensHidden=', TOKENS_HIDDEN,
    ' channelsHidden=', CHANNELS_HIDDEN);
  WriteLn('  train=', TRAIN_SIZE, ' test=', TEST_SIZE, ' epochs=', NUM_EPOCHS);
  BuildData;
  try
    RandSeed := SEED;
    NN := BuildNet;
    NN.SetLearningRate(LR, 0.9);
    NN.SetBatchUpdate(false);
    WriteLn('  trainable weights = ', NN.CountWeights());
    WriteLn;
    WriteLn(' epoch    train loss   train acc    test acc');
    for epoch := 0 to NUM_EPOCHS - 1 do
    begin
      loss := 0;
      for i := 0 to High(TrainSet) do
      begin
        NN.Compute(TrainSet[i].X);
        // Cross-entropy on the SoftMax output for the true class.
        loss := loss - Ln( Max(NN.GetLastLayer.Output.Raw[TrainSet[i].Cls], 1e-7) );
        NN.Backpropagate(TrainSet[i].Y);
      end;
      loss := loss / TRAIN_SIZE;
      if (epoch mod 5 = 0) or (epoch = NUM_EPOCHS - 1) then
      begin
        trainAcc := Accuracy(NN, TrainSet);
        testAcc := Accuracy(NN, TestSet);
        WriteLn('  ', epoch:4, '    ', SafeF(loss), '   ', SafeF(trainAcc),
          '   ', SafeF(testAcc));
      end;
    end;
    WriteLn;
    trainAcc := Accuracy(NN, TrainSet);
    testAcc := Accuracy(NN, TestSet);
    WriteLn('FINAL  train acc = ', SafeF(trainAcc),
      '   test acc = ', SafeF(testAcc));
    if testAcc > 0.9 then
      WriteLn('The all-MLP Mixer stack solved which-half classification ',
        '(requires token mixing).')
    else
      WriteLn('Mixer did not fully converge on this run; ',
        'tune epochs / LR / hidden widths.');
    NN.Free;
  finally
    FreeData;
  end;
end.
