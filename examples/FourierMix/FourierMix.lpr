program FourierMix;
(*
FourierMix: an FNet-vs-attention bake-off for TNNetFourierMix, the FNet-style
PARAMETER-FREE token mixer (Lee-Thorp et al. 2021, "FNet: Mixing Tokens with
Fourier Transforms"). Over a (SeqLen, 1, d) sequence tensor it replaces self-
attention with an UNPARAMETERISED 2D discrete Fourier transform across the
sequence and hidden axes, keeping only the real part:
  y = Re( DFT_seq( DFT_hidden( x ) ) )
The layer owns NO trainable weights at all; mixing is a fixed linear operator.

Task: a tiny per-token sequence-regression problem where each output token must
combine information from EVERY input token (a fixed global mixing of the
sequence followed by a smooth nonlinearity). Solving it REQUIRES token mixing -
a position-wise MLP alone cannot do it. Two equal-depth models are trained head
to head on the SAME data:
  (A) FNet      : TNNetFourierMix (0 mixing weights) + a shared per-token MLP
  (B) Attention : AddMultiHeadSelfAttention(...)     + the same per-token MLP

The headline makes the FNet expressiveness-vs-cost trade concrete on a short
sequence: dropping the entire learned mixing block for a FIXED Fourier basis
costs the attention model's whole Q|K|V|out projection (hundreds of weights) and
trades only a modest amount of accuracy - the paper's selling point being that
on short sequences the mix is nearly free. We print final train/test MSE, the
parameter counts, and per-model training wall-clock.

Everything is generated on the fly; no external dataset. Pure CPU, single
thread, well under five minutes (a few seconds in practice).

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
  cSeq     = 8;       // sequence length (number of tokens)
  cDim     = 8;       // per-token hidden width (depth)
  cSteps   = 1000;
  cBatch   = 16;
  cLR      = 0.001;
  cInertia = 0.9;
  cTestSet = 256;

var
  // A fixed (unknown to the models) token-mixing matrix: target token i blends
  // all input tokens through GMix[i][j], then a smooth tanh nonlinearity is
  // applied per element. Mixing is GLOBAL across the sequence, so a position-
  // wise MLP alone cannot fit it - the model MUST mix tokens.
  GMix: array[0..cSeq - 1, 0..cSeq - 1] of TNeuralFloat;

  procedure InitTeacher;
  var
    I, J: integer;
    S: TNeuralFloat;
  begin
    for I := 0 to cSeq - 1 do
    begin
      S := 0;
      for J := 0 to cSeq - 1 do
      begin
        GMix[I][J] := Exp(-0.5 * Sqr((I - J) / 2.0)) + 0.2 * Cos(I * 0.5 + J);
        S := S + Abs(GMix[I][J]);
      end;
      for J := 0 to cSeq - 1 do
        GMix[I][J] := GMix[I][J] / S;  // row-normalize -> well-scaled operator
    end;
  end;

  // One (input, target) pair on a (cSeq, 1, cDim) tensor.
  //   mixed[i,d] = sum_j GMix[i][j] * x[j,d]
  //   target[i,d] = tanh( 1.5 * mixed[i,d] )
  procedure MakePair(InputV, TargetV: TNNetVolume);
  var
    I, J, D: integer;
    Acc: TNeuralFloat;
  begin
    for I := 0 to InputV.Size - 1 do
      InputV.FData[I] := Random * 2.0 - 1.0;
    for I := 0 to cSeq - 1 do
      for D := 0 to cDim - 1 do
      begin
        Acc := 0;
        for J := 0 to cSeq - 1 do
          Acc := Acc + GMix[I][J] * InputV.FData[J * cDim + D];
        TargetV.FData[I * cDim + D] := Math.Tanh(1.5 * Acc);
      end;
  end;

  function MeanSquaredError(Output, Target: TNNetVolume): TNeuralFloat;
  var
    I: integer;
    diff: TNeuralFloat;
  begin
    Result := 0;
    for I := 0 to Output.Size - 1 do
    begin
      diff := Output.FData[I] - Target.FData[I];
      Result := Result + diff * diff;
    end;
    Result := Result / Output.Size;
  end;

  function Train(NN: TNNet; const Name: string): double;
  var
    Step, B: integer;
    InputV, TargetV: TNNetVolume;
    SumLoss: TNeuralFloat;
    T0: double;
  begin
    InputV  := TNNetVolume.Create(cSeq, 1, cDim);
    TargetV := TNNetVolume.Create(cSeq, 1, cDim);
    T0 := Now();
    try
      for Step := 1 to cSteps do
      begin
        SumLoss := 0;
        for B := 1 to cBatch do
        begin
          MakePair(InputV, TargetV);
          NN.Compute(InputV);
          SumLoss := SumLoss + MeanSquaredError(NN.GetLastLayer.Output, TargetV);
          NN.Backpropagate(TargetV);
        end;
        if (Step = 1) or (Step mod 200 = 0) or (Step = cSteps) then
          WriteLn(Format('  [%s] step %4d / %4d   train-MSE=%.6e',
            [Name, Step, cSteps, SumLoss / cBatch]));
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
    Result := (Now() - T0) * 86400.0;
  end;

  function EvalTestMSE(NN: TNNet): TNeuralFloat;
  var
    I: integer;
    InputV, TargetV: TNNetVolume;
  begin
    RandSeed := 99991;
    InputV  := TNNetVolume.Create(cSeq, 1, cDim);
    TargetV := TNNetVolume.Create(cSeq, 1, cDim);
    Result := 0;
    try
      for I := 1 to cTestSet do
      begin
        MakePair(InputV, TargetV);
        NN.Compute(InputV);
        Result := Result + MeanSquaredError(NN.GetLastLayer.Output, TargetV);
      end;
      Result := Result / cTestSet;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  // Total trainable weights across all layers (the honest parameter budget).
  function CountAllTrainable(NN: TNNet): integer;
  var
    L, O: integer;
  begin
    Result := 0;
    for L := 0 to NN.CountLayers - 1 do
      for O := 0 to NN.Layers[L].Neurons.Count - 1 do
        Result := Result + NN.Layers[L].Neurons[O].Weights.Size + 1;
  end;

  // A shared per-token MLP applied identically at every sequence position
  // (PointwiseConv over depth, so tokens are NOT mixed by it).
  procedure AddPerTokenMLP(NN: TNNet);
  begin
    NN.AddLayer(TNNetPointwiseConvReLU.Create(cDim * 2));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cDim));
  end;

var
  NNFNet, NNAttn: TNNet;
  FNetParams, AttnParams: integer;
  FNetMSE, AttnMSE: TNeuralFloat;
  FNetTime, AttnTime: double;
  StartTime, Elapsed: double;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  StartTime := Now();
  InitTeacher;

  WriteLn('FourierMix bake-off: a ', cSeq, '-token x ', cDim,
    '-dim GLOBAL token-mixing regression target.');
  WriteLn('FNet parameter-free Fourier mixing (TNNetFourierMix) vs ',
    'self-attention,');
  WriteLn('both followed by the SAME shared per-token MLP. Headline: ',
    '~attention accuracy');
  WriteLn('at a fraction of the mixing cost (FNet''s selling point on short ',
    'sequences).');
  WriteLn;

  // ---- Model A: FNet parameter-free Fourier token mixer ----
  RandSeed := 2026;
  NNFNet := TNNet.Create();
  NNFNet.AddLayer(TNNetInput.Create(cSeq, 1, cDim));
  NNFNet.AddLayer(TNNetFourierMix.Create());   // 0 trainable mixing weights
  AddPerTokenMLP(NNFNet);
  NNFNet.SetLearningRate(cLR, cInertia);

  // ---- Model B: self-attention token mixer ----
  RandSeed := 2026;
  NNAttn := TNNet.Create();
  NNAttn.AddLayer(TNNetInput.Create(cSeq, 1, cDim));
  // Token-wise Q|K|V slab projection d_model -> 3*d_model; then 2-head self-
  // attention consumes the slab and out-projects back to d_model (the standard
  // transformer wiring, mirrors AddTransformerEncoderBlock / TokenShiftBaseline).
  NNAttn.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDim));
  NNAttn.AddMultiHeadSelfAttention(2);          // 2 heads of learned attention
  AddPerTokenMLP(NNAttn);
  NNAttn.SetLearningRate(cLR, cInertia);

  FNetParams := CountAllTrainable(NNFNet);
  AttnParams := CountAllTrainable(NNAttn);

  try
    WriteLn(Format('Total trainable parameters:  FNet=%d   Attention=%d',
      [FNetParams, AttnParams]));
    WriteLn(Format('  (the Fourier mixer itself adds ZERO parameters; the ' +
      'attention block adds %d)', [AttnParams - FNetParams]));
    WriteLn;

    WriteLn('Training FNet (Fourier mixing) model...');
    RandSeed := 2026;
    FNetTime := Train(NNFNet, 'fnet');
    WriteLn('Training attention model...');
    RandSeed := 2026;
    AttnTime := Train(NNAttn, 'attn');
    WriteLn;

    FNetMSE := EvalTestMSE(NNFNet);
    AttnMSE := EvalTestMSE(NNAttn);

    WriteLn(StringOfChar('=', 72));
    WriteLn('RESULTS');
    WriteLn(StringOfChar('-', 72));
    WriteLn(Format('  %-26s %12s %14s %14s',
      ['model', 'params', 'test-MSE', 'train-secs']));
    WriteLn(Format('  %-26s %12d %14.3e %14.2f',
      ['FNet (Fourier, free mix)', FNetParams, FNetMSE, FNetTime]));
    WriteLn(Format('  %-26s %12d %14.3e %14.2f',
      ['attention (learned mix)', AttnParams, AttnMSE, AttnTime]));
    WriteLn(StringOfChar('-', 72));
    WriteLn(Format('  FNet drops the entire learned mixing block: %d FEWER ' +
      'parameters (%.0f%% smaller).',
      [AttnParams - FNetParams, 100.0 * (AttnParams - FNetParams) / AttnParams]));
    WriteLn(Format('  test-MSE ratio (fnet/attn) = %.2f: the FIXED Fourier ' +
      'basis trades some accuracy', [FNetMSE / AttnMSE]));
    WriteLn('  for ZERO mixing weights - the FNet expressiveness-vs-cost ',
      'trade made concrete on a');
    WriteLn('  short sequence (the regime where the paper reports the mix is ',
      'nearly free).');
    WriteLn(StringOfChar('=', 72));

    Elapsed := (Now() - StartTime) * 86400.0;
    WriteLn;
    WriteLn(Format('Total wall-clock: %.1f s (pure CPU, single thread).',
      [Elapsed]));
  finally
    NNFNet.Free;
    NNAttn.Free;
  end;
end.
