program SpatialGatingUnit;
(*
SpatialGatingUnit: a gMLP (Spatial Gating Unit) block versus a same-parameter-
budget single-head self-attention baseline on a long-range sequence task.

The toy task is LONG-RANGE FIRST-TOKEN BROADCAST: the input is a length-cSeqLen
sequence of small tokens and the target at EVERY position n is the value of
token 0. Solving it requires each output position to pull information from a
single, distant source position - a clean long-range dependency that local
mixing cannot capture and that needs a genuine cross-token operator. This is
exactly the regime gMLP was designed for, and a clean way to show the Spatial
Gating Unit's static SeqLen x SeqLen mixing matrix matching (or beating) an
attention head of the same size.

Two models are trained on identical data:

  (A) gMLP   : Embedding -> SinusoidalPosEmb -> AddgMLPBlock(SeqLen,d,d_ffn)
               -> PointwiseConvLinear(2) -> PointwiseSoftMax
               The block is up-proj -> Spatial Gating Unit -> down-proj with a
               pre-LayerNorm residual, plus the two gMLP-paper LayerNorms that
               bound the multiplicative gate (one before the SGU, one after).
               The SGU's W is a fixed (after training) SeqLen x SeqLen matrix
               shared across channels - attention-free.

  (B) attn   : Embedding -> SinusoidalPosEmb -> PointwiseConvLinear(3*d_k)
               -> single-head SDPA -> PointwiseConvLinear(2) -> PointwiseSoftMax
               The d_ffn / d_k are sized so the two models have comparable
               trainable-weight counts (printed at startup).

Pure CPU, no external dataset, no files written. Runs in well under a minute on
one thread.

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
  cVocab   = 4;     // small token alphabet
  cSeqLen  = 24;    // long-range: position 0 must reach every output position
  cDModel  = 16;    // residual-stream / embedding width (even for the SGU)
  cDFFN    = 32;    // gMLP channel-MLP hidden width (even; SGU halves it)
  cDk      = 24;    // attention head width, tuned for a matched param budget
  cSteps   = 800;
  cBatch   = 32;
  cLR      = 0.001;
  cInertia = 0.9;

  procedure BuildGMLP(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
    NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
    // The attention-free gMLP sequence mixer (static SeqLen x SeqLen matrix).
    NN.AddgMLPBlock(cSeqLen, cDModel, cDFFN);
    // Per-position readout to 2 logits + softmax across depth.
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
  end;

  procedure BuildAttn(out NN: TNNet);
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
    NN.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
    // Pack Q|K|V and run one non-causal SDPA head (the matched baseline).
    NN.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDk));
    NN.AddLayer(TNNetScaledDotProductAttention.Create(cDk, False));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
  end;

  // Build a (input, target) pair. Input lives on X (token IDs); target is
  // one-hot over depth, shape (cSeqLen,1,cVocab), for the per-position softmax.
  procedure MakePair(const Tokens: array of integer;
    InputV, TargetV: TNNetVolume);
  var
    I: integer;
  begin
    TargetV.Fill(0);
    for I := 0 to cSeqLen - 1 do
    begin
      InputV.FData[I] := Tokens[I];
      // Long-range broadcast: every output position must reproduce token 0.
      TargetV[I, 0, Tokens[0]] := 1.0;
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
      if Cur > BestVal then begin BestVal := Cur; Best := D; end;
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

  // Train one model on freshly-sampled data; return final held-out per-token
  // accuracy (%). A fixed seed is set BEFORE each call so both models see the
  // exact same data stream and the comparison is fair.
  function TrainAndEval(NN: TNNet; const ATag: string): TNeuralFloat;
  var
    Step, B, K, I, Pred, Correct, Total: integer;
    InputV, TargetV: TNNetVolume;
    Tokens: array[0..cSeqLen - 1] of integer;
    SumLoss: TNeuralFloat;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    try
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
        if (Step = 1) or (Step mod 50 = 0) or (Step = cSteps) then
          WriteLn(Format('  [%s] step %4d / %4d   mean-CE=%.5f',
            [ATag, Step, cSteps, SumLoss / cBatch]));
      end;
      // Held-out accuracy over fresh sequences.
      Correct := 0; Total := 0;
      for K := 1 to 200 do
      begin
        RandomTokens(Tokens);
        MakePair(Tokens, InputV, TargetV);
        NN.Compute(InputV);
        for I := 0 to cSeqLen - 1 do
        begin
          Pred := ArgMaxDepth(NN.GetLastLayer.Output, I);
          if Pred = ArgMaxDepth(TargetV, I) then Inc(Correct);
          Inc(Total);
        end;
      end;
      Result := 100.0 * Correct / Total;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

var
  GMLP, Attn: TNNet;
  AccGMLP, AccAttn: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('SpatialGatingUnit demo: gMLP (attention-free SGU mixer) vs a ',
    'same-budget single-head attention baseline on long-range first-token ',
    'broadcast over ', cSeqLen, ' tokens.');
  WriteLn;

  // Seed before EACH build so both models get a deterministic, well-scaled
  // weight init (the same seed that the data stream uses below).
  RandSeed := 2026;
  BuildGMLP(GMLP);
  RandSeed := 2026;
  BuildAttn(Attn);
  WriteLn('Trainable weights: gMLP=', GMLP.CountWeights(),
    '   attention=', Attn.CountWeights());
  WriteLn;

  try
    WriteLn('Training gMLP (Spatial Gating Unit) ...');
    RandSeed := 2026;
    AccGMLP := TrainAndEval(GMLP, 'gMLP');
    WriteLn;

    WriteLn('Training attention baseline ...');
    RandSeed := 2026;                 // identical data stream
    AccAttn := TrainAndEval(Attn, 'attn');
    WriteLn;

    WriteLn(StringOfChar('=', 64));
    WriteLn(Format('Held-out per-token accuracy:  gMLP=%.2f%%   attention=%.2f%%',
      [AccGMLP, AccAttn]));
    if AccGMLP >= AccAttn - 0.01 then
      WriteLn('RESULT: the attention-free Spatial Gating Unit MATCHES or BEATS ',
        'the same-budget attention head.')
    else
      WriteLn('RESULT: attention edged ahead this run (try more steps).');
    WriteLn(StringOfChar('=', 64));
  finally
    GMLP.Free;
    Attn.Free;
  end;
end.
