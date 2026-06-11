program RoPEBaseFrequencySweep;
(*
RoPEBaseFrequencySweep: empirically interrogate the "magic 10000" in RoPE.

RoPE (Rotary Position Embedding) injects position by rotating consecutive
channel pairs of the embedding by an angle that grows linearly with the
token's absolute position. For channel-pair index i (0..d/2-1) the rotation
frequency is

    theta_i = base^(-2*i / d)

so the first pair rotates almost once per step (high frequency) and the last
pair barely turns over the whole sequence (low frequency). The single scalar
"base" sets how quickly the frequencies decay across channel pairs:

  - a SMALL base (e.g. 1e2) packs all pairs into high frequencies: even the
    slow channels turn quickly, so positions that are far apart alias onto
    similar rotations and long-range distance is hard to read off.
  - a LARGE base (e.g. 1e5) stretches the low-frequency tail very flat: the
    slow channels barely move across the whole sequence, giving smooth
    long-range position signal but coarse short-range resolution.

The constant base = 10000 comes straight from the RoPE / original
Transformer sinusoidal paper and is cargo-culted into nearly every
implementation without re-derivation. This program varies base over
{1e2, 1e3, 1e4, 1e5} and trains the SAME tiny attention model once per value
on a POSITION-sensitive next-token task, then prints final train/val loss so
a reader can see whether 10000 is actually special at this scale.

Task (LONG-RANGE POSITION MATTERS): a causal "copy-from-k-steps-back"
stream. The target at position i is the INPUT token at position i-cOffset
(a fixed begin token where i < cOffset). Self-attention is
permutation-invariant over the key positions, so the model can only solve
this if the position encoding lets a query at position i pick out the key
exactly cOffset steps back. RoPE makes the relative rotation between query i
and key i-cOffset a fixed offset, which is exactly the signal needed -- but
how cleanly that fixed long-range offset reads out of the dot product
depends on the frequency spread set by base. A larger offset over a longer
sequence leans on the slow, low-frequency channel pairs, which is precisely
where the base constant decides whether distant positions are resolvable or
alias together, so the task is genuinely discriminating across base values.

Shared single-head CAUSAL attention stack (only the RoPE base changes):
  TNNetInput(SeqLen, 1, 1)                       { token IDs along X }
  -> TNNetEmbedding(Vocab, d_model)              { learned token vectors }
  -> TNNetRotaryEmbedding.Create(Base)           { THE SWEPT KNOB; applied
                                                   to the embedded sequence }
  -> packed Q|K|V via TNNetPointwiseConvLinear + three TNNetSplitChannels
  -> ValueT = TNNetTransposeXD(V)
  -> scores = TNNetDotProducts(Q, K) / sqrt(d_k)     { (key,1,query) }
  -> reshape -> (key, query, 1)
  -> TNNetMaskedFill                             { causal upper triangle }
  -> reshape -> (key,1,query) -> ReLUL -> PointwiseSoftMax(depth)
  -> TNNetDotProducts(ValueT, W)                 { weighted sum of V }
  -> TNNetPointwiseConvLinear(Vocab) -> PointwiseSoftMax(1)

This reuses the exact attention wiring validated in
examples/PositionEncodingBakeoff. The ONLY difference between arms is the
scalar passed to TNNetRotaryEmbedding.Create. Every arm shares the RNG seed,
steps, learning rate and data, so the comparison is apples-to-apples. Pure
CPU, no dataset download, runs in well under four minutes.

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
  // Alphabet: token 0 = begin; 1..cVocab-1 are content tokens.
  cVocab        = 8;
  cSeqLen       = 24;       // long enough that low-freq channels matter
  cOffset       = 3;        // copy the token cOffset positions back
  cDModel       = 32;       // even (required by RoPE)
  cDk           = 32;
  cSteps        = 300;      // training steps per arm
  cBatch        = 32;       // sequences per step
  cLR           = 0.005;
  cInertia      = 0.9;
  cSeed         = 2026;     // shared train seed
  cValSeed      = 7777;     // shared validation seed (disjoint stream)
  cValBatches   = 64;       // validation sequences

  cNumBases = 4;
  cBases: array[0..cNumBases - 1] of TNeuralFloat =
    (1.0e2, 1.0e3, 1.0e4, 1.0e5);

  // Build the identical single-head causal attention model; the only knob
  // is the RoPE base frequency.
  function BuildNet(Base: TNeuralFloat): TNNet;
  var
    NN: TNNet;
    EmbeddedLayer, QKV, Query, Key, ValueT: TNNetLayer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));

    // *** THE SWEPT KNOB: RoPE base frequency. ***
    NN.AddLayer(TNNetRotaryEmbedding.Create(Base));
    EmbeddedLayer := NN.GetLastLayer;

    // Packed Q|K|V projection sliced into three views of the same layer.
    QKV := NN.AddLayerAfter(TNNetPointwiseConvLinear.Create(3 * cDk),
             EmbeddedLayer);
    Query := NN.AddLayerAfter(TNNetSplitChannels.Create(0, cDk), QKV);
    Key   := NN.AddLayerAfter(TNNetSplitChannels.Create(cDk, cDk), QKV);
    NN.AddLayerAfter(TNNetSplitChannels.Create(2 * cDk, cDk), QKV);
    ValueT := NN.AddLayer(TNNetTransposeXD.Create());

    // scores: (key,1,query) = Q . K^T scaled by 1/sqrt(d_k).
    NN.AddLayer(TNNetDotProducts.Create(Query, Key, False));
    NN.AddLayer(TNNetMulByConstant.Create(1.0 / Sqrt(cDk)));

    // Reshape to (key, query, 1): X=key, Y=query (layout MaskedFill wants).
    NN.AddLayer(TNNetReshape.Create(cSeqLen, cSeqLen, 1));

    // Causal mask (identical across every arm).
    NN.AddLayer(TNNetMaskedFill.Create());

    NN.AddLayer(TNNetReshape.Create(cSeqLen, 1, cSeqLen));
    NN.AddLayer(TNNetReLUL.Create(-500, +500, 0));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(0));

    NN.AddLayer(TNNetDotProducts.Create(ValueT, NN.GetLastLayer, False));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    NN.SetLearningRate(cLR, cInertia);
    Result := NN;
  end;

  procedure RandomTokens(var Tokens: array of integer);
  var
    I: integer;
  begin
    // Real tokens are 1..cVocab-1; token 0 is the begin symbol only.
    for I := 0 to High(Tokens) do
      Tokens[I] := 1 + Random(cVocab - 1);
  end;

  // target[i] = input token at position i-cOffset; begin (0) where i<cOffset.
  procedure MakePair(const Tokens: array of integer;
    InputV, TargetV: TNNetVolume);
  var
    I, Gold: integer;
  begin
    TargetV.Fill(0);
    for I := 0 to cSeqLen - 1 do
    begin
      InputV.FData[I] := Tokens[I];
      if I >= cOffset then Gold := Tokens[I - cOffset] else Gold := 0;
      TargetV[I, 0, Gold] := 1.0;
    end;
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

  // Mean cross-entropy + token accuracy over a fixed validation stream.
  procedure Evaluate(NN: TNNet; out ValLoss, ValAcc: TNeuralFloat);
  var
    B, I, Pred, Gold: integer;
    InputV, TargetV: TNNetVolume;
    Tokens: array[0..cSeqLen - 1] of integer;
    SumLoss: TNeuralFloat;
    Correct, Total: integer;
  begin
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    SumLoss := 0; Correct := 0; Total := 0;
    RandSeed := cValSeed;
    try
      for B := 1 to cValBatches do
      begin
        RandomTokens(Tokens);
        MakePair(Tokens, InputV, TargetV);
        NN.Compute(InputV);
        SumLoss := SumLoss + CrossEntropy(NN.GetLastLayer.Output, TargetV);
        for I := 0 to cSeqLen - 1 do
        begin
          Pred := ArgMaxDepth(NN.GetLastLayer.Output, I);
          // Recover the gold token from the one-hot target row.
          Gold := 0;
          while (Gold < cVocab - 1) and (TargetV[I, 0, Gold] = 0) do Inc(Gold);
          if Pred = Gold then Inc(Correct);
          Inc(Total);
        end;
      end;
      ValLoss := SumLoss / cValBatches;
      ValAcc  := Correct / Total;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

  // Train one arm; return final-step mean train cross-entropy.
  function Train(NN: TNNet; Base: TNeuralFloat): TNeuralFloat;
  var
    Step, B: integer;
    InputV, TargetV: TNNetVolume;
    Tokens: array[0..cSeqLen - 1] of integer;
    SumLoss: TNeuralFloat;
    StartTime, Elapsed: double;
  begin
    Result := 0;
    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    RandSeed := cSeed;
    try
      WriteLn(Format('  --- training RoPE base=%.0f ---', [Base]));
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
        Result := SumLoss / cBatch;
        if (Step = 1) or (Step mod 40 = 0) or (Step = cSteps) then
        begin
          Elapsed := (Now() - StartTime) * 86400.0;
          WriteLn(Format('    step %4d / %4d   mean-CE=%.5f   elapsed=%.1fs',
            [Step, cSteps, Result, Elapsed]));
        end;
      end;
    finally
      InputV.Free;
      TargetV.Free;
    end;
  end;

var
  Idx, BestIdx: integer;
  NN: TNNet;
  TrainLoss, ValLoss, ValAcc: array[0..cNumBases - 1] of TNeuralFloat;
  GStart, GElapsed: double;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  WriteLn('RoPEBaseFrequencySweep: same tiny single-head CAUSAL attention ',
    'model, four RoPE base frequencies.');
  WriteLn(Format('Task: copy the token %d positions back (target[i] = ' +
    'input[i-%d]).', [cOffset, cOffset]));
  WriteLn(Format('Vocab=%d, SeqLen=%d, d_model=%d. theta_i = base^(-2i/d).',
    [cVocab, cSeqLen, cDModel]));
  WriteLn('Sweeping the "magic 10000": base in {1e2, 1e3, 1e4, 1e5}.');
  WriteLn;

  GStart := Now();
  for Idx := 0 to cNumBases - 1 do
  begin
    NN := BuildNet(cBases[Idx]);
    try
      if Idx = 0 then
      begin
        WriteLn('Architecture (base=100 arm shown; other arms differ only in ',
          'the TNNetRotaryEmbedding base):');
        NN.PrintSummary();
        WriteLn;
      end;
      TrainLoss[Idx] := Train(NN, cBases[Idx]);
      Evaluate(NN, ValLoss[Idx], ValAcc[Idx]);
    finally
      NN.Free;
    end;
    WriteLn;
  end;
  GElapsed := (Now() - GStart) * 86400.0;

  WriteLn(StringOfChar('=', 72));
  WriteLn('RESULTS (lower cross-entropy / higher accuracy is better):');
  WriteLn(StringOfChar('=', 72));
  WriteLn('     base      train-CE      val-CE      val-acc');
  WriteLn('  --------     --------     --------     -------');
  BestIdx := 0;
  for Idx := 0 to cNumBases - 1 do
  begin
    WriteLn(Format('  %8.0f     %.5f      %.5f      %5.1f%%',
      [cBases[Idx], TrainLoss[Idx], ValLoss[Idx], 100.0 * ValAcc[Idx]]));
    if ValLoss[Idx] < ValLoss[BestIdx] then BestIdx := Idx;
  end;
  WriteLn;
  WriteLn(StringOfChar('=', 72));
  WriteLn(Format('Best base by validation CE: %.0f  (val-CE=%.5f).',
    [cBases[BestIdx], ValLoss[BestIdx]]));
  if cBases[BestIdx] = 10000.0 then
    WriteLn('The cargo-culted 10000 wins here.')
  else
    WriteLn('The cargo-culted 10000 is NOT the winner on this task -- ',
      'the base is empirical, not sacred.');
  WriteLn(Format('Total runtime for all %d arms: %.1fs.',
    [cNumBases, GElapsed]));
end.
