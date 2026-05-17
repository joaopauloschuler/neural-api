program AttentionCopyTask;
(*
AttentionCopyTask: smallest possible end-to-end attention training demo.
A single TNNetScaledDotProductAttention head learns to COPY a 16-token
input sequence to its output. The model uses a small vocabulary (8) and
a small embedding dim, so the whole run completes in well under a
minute on a single CPU thread.

Pipeline (per position):
  token-id -> TNNetEmbedding(d_model)
           -> TNNetSinusoidalPositionalEmbedding (parameter-free)
           -> TNNetPointwiseConvLinear(3*d_k)         { pack Q | K | V }
           -> TNNetScaledDotProductAttention(d_k)     { single head, non-causal }
           -> TNNetPointwiseConvLinear(Vocab)         { per-position readout }
           -> TNNetPointwiseSoftMax(1)                { softmax across depth }

Training data is generated on the fly: each pair is a random sequence
of length 16 over a vocabulary of size 8; the target is identical to
the input. No external dataset.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

{ The model uses fast-math kernels that can transiently produce
  denormals / underflows during early training; mask FPU exceptions
  so the host doesn't abort. Matches what other examples do (e.g.
  LearningRateFinder). }

const
  cVocab   = 8;
  cSeqLen  = 16;
  cDModel  = 16;
  cDk      = 16;
  cSteps   = 400;
  cBatch   = 32;
  cLR      = 0.005;
  cInertia = 0.9;

  procedure BuildModel(out NN: TNNet);
  begin
    NN := TNNet.Create();
    // Input: cSeqLen token IDs along X axis.
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    // Look up a learned embedding per token (EncodeZero=1 so token 0 trains too).
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
    // Inject parameter-free sinusoidal position information.
    NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
    // Pack Q|K|V (depth = 3 * d_k) for a single attention head.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));
    // Single-head non-causal SDPA: every query may see every key.
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, False));
    // Per-position linear projection to vocab logits.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    // Softmax along depth axis -> per-position probability over vocab.
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
  end;

  // Build a (input, target) pair. The input lives on X (token IDs).
  // The target is one-hot over the depth axis, shape (cSeqLen, 1, cVocab),
  // matching the per-position softmax head.
  procedure MakePair(const Tokens: array of integer;
    InputV, TargetV: TNNetVolume);
  var
    I: integer;
  begin
    TargetV.Fill(0);
    for I := 0 to cSeqLen - 1 do
    begin
      InputV.FData[I] := Tokens[I];
      TargetV[I, 0, Tokens[I]] := 1.0;
    end;
  end;

  procedure RandomTokens(var Tokens: array of integer);
  var
    I: integer;
  begin
    for I := 0 to High(Tokens) do
      Tokens[I] := Random(cVocab);
  end;

  function ArgMaxDepth(V: TNNetVolume; Pos: integer): integer;
  var
    D, Best: integer;
    BestVal, Cur: TNeuralFloat;
  begin
    Best := 0;
    BestVal := V[Pos, 0, 0];
    for D := 1 to cVocab - 1 do
    begin
      Cur := V[Pos, 0, D];
      if Cur > BestVal then
      begin
        BestVal := Cur;
        Best := D;
      end;
    end;
    Result := Best;
  end;

  function CrossEntropy(Output, Target: TNNetVolume): TNeuralFloat;
  var
    I: integer;
    P: TNeuralFloat;
  begin
    Result := 0;
    for I := 0 to Output.Size - 1 do
      if Target.FData[I] > 0 then
      begin
        P := Output.FData[I];
        if P < 1e-12 then P := 1e-12;
        Result := Result - Target.FData[I] * Ln(P);
      end;
    Result := Result / cSeqLen;
  end;

  procedure Train(NN: TNNet);
  var
    Step, B: integer;
    InputV, TargetV: TNNetVolume;
    Tokens: array[0..cSeqLen - 1] of integer;
    SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    try
      StartTime := Now();
      for Step := 1 to cSteps do
      begin
        SumLoss := 0;
        for B := 1 to cBatch do
        begin
          RandomTokens(Tokens);
          MakePair(Tokens, InputV, TargetV);
          NN.Compute(InputV);
          SumLoss := SumLoss + CrossEntropy(NN.GetLastLayer.Output, TargetV);
          NN.Backpropagate(TargetV);
        end;
        if (Step = 1) or (Step mod 25 = 0) or (Step = cSteps) then
        begin
          Elapsed := (Now() - StartTime) * 86400.0;
          WriteLn(Format('  step %4d / %4d   mean-CE=%.5f   elapsed=%.1fs',
            [Step, cSteps, SumLoss / cBatch, Elapsed]));
        end;
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  procedure Evaluate(NN: TNNet);
  const
    cNumProbes = 5;
  var
    K, I, Pred, Correct, Total: integer;
    InputV, TargetV: TNNetVolume;
    Tokens: array[0..cSeqLen - 1] of integer;
    InS, OutS: string;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    Correct := 0;
    Total := 0;
    try
      for K := 1 to cNumProbes do
      begin
        RandomTokens(Tokens);
        MakePair(Tokens, InputV, TargetV);
        NN.Compute(InputV);
        InS := '';
        OutS := '';
        for I := 0 to cSeqLen - 1 do
        begin
          Pred := ArgMaxDepth(NN.GetLastLayer.Output, I);
          InS  := InS + IntToStr(Tokens[I]) + ' ';
          OutS := OutS + IntToStr(Pred) + ' ';
          if Pred = Tokens[I] then Inc(Correct);
          Inc(Total);
        end;
        WriteLn('  INPUT     : ', InS);
        WriteLn('  PREDICTED : ', OutS);
        WriteLn;
      end;
      WriteLn(Format('Per-token accuracy on %d held-out probes: %d / %d = %.2f%%',
        [cNumProbes, Correct, Total, 100.0 * Correct / Total]));
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

var
  NN: TNNet;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 2026;
  WriteLn('AttentionCopyTask demo: a single SDPA head learns to copy a ',
    cSeqLen, '-token input over a vocab of ', cVocab, '.');
  BuildModel(NN);
  try
    WriteLn;
    WriteLn('Architecture:');
    NN.PrintSummary();
    WriteLn;
    WriteLn('Training for ', cSteps, ' steps of batch size ', cBatch, '...');
    Train(NN);
    WriteLn;
    WriteLn(StringOfChar('=', 72));
    WriteLn('Evaluation on fresh random sequences:');
    WriteLn(StringOfChar('=', 72));
    Evaluate(NN);
    WriteLn;
    WriteLn('Expect: per-token accuracy approaches 100% after training as ',
      'the single attention head learns to route each query position to ',
      'its matching key position (identity attention).');
  finally
    NN.Free;
  end;
end.
