program ALiBiSlopeSweep;
(*
ALiBiSlopeSweep: empirically interrogate the "magic 8" in ALiBi.

ALiBi (Attention with Linear Biases) adds a per-head, distance-dependent
bias to the raw attention scores:

    score(query Y, key X) += slope[h] * (X - Y)

with the per-head slope

    slope[h] = 2^(-Base * (h + 1) / H)

The constant Base is almost universally set to 8 -- a value lifted from
the original ALiBi paper and then cargo-culted into countless reimplemen-
tations without re-derivation. This program varies Base over
{4, 6, 8, 12} and trains the SAME tiny attention model once per value on a
LOCALITY-sensitive next-token task, then prints final train/val loss so a
reader can see whether 8 is actually special.

Task (RECENCY MATTERS): a char-level "copy-the-most-recent-vowel" stream.
The alphabet is a few consonants plus a few vowels. The target at every
position is the MOST RECENTLY SEEN vowel at or before that position (the
begin token if none yet). Because the answer is almost always a nearby
token, a strong recency / locality prior -- exactly what ALiBi injects --
helps; a slope that is too flat (small Base) spreads attention too far
back, and a slope that is too steep (large Base) collapses onto the
query's own position and can miss a vowel that sits one or two steps back.
So the task is genuinely discriminating across Base values rather than
flat.

Shared single-head CAUSAL attention stack (only the ALiBi Base changes):
  TNNetInput(SeqLen, 1, 1)                       { token IDs along X }
  -> TNNetEmbedding(Vocab, d_model)              { learned token vectors }
  -> packed Q|K|V via TNNetPointwiseConvLinear + three TNNetSplitChannels
  -> ValueT = TNNetTransposeXD(V)
  -> scores = TNNetDotProducts(Q, K) / sqrt(d_k)     { (key,1,query) }
  -> reshape -> (key, query, 1)
  -> TNNetALiBi.Create(Base)                      { THE SWEPT KNOB }
  -> TNNetMaskedFill                             { causal upper triangle }
  -> reshape -> (key,1,query) -> ReLUL -> PointwiseSoftMax(depth)
  -> TNNetDotProducts(ValueT, W)                 { weighted sum of V }
  -> TNNetPointwiseConvLinear(Vocab) -> PointwiseSoftMax(1)

This reuses the exact attention wiring validated in
examples/PositionEncodingBakeoff. The ONLY difference between arms is the
scalar passed to TNNetALiBi.Create. Every arm shares the RNG seed, epochs,
learning rate and data, so the comparison is apples-to-apples. Pure CPU,
no dataset download, runs in well under four minutes.

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
  // Alphabet: token 0 = begin; 1..3 = consonants; 4..6 = vowels.
  cVocab        = 7;
  cFirstVowel   = 4;        // tokens 4,5,6 are "vowels"
  cSeqLen       = 16;
  cDModel       = 32;
  cDk           = 32;
  cSteps        = 220;      // training steps per arm
  cBatch        = 24;       // sequences per step
  cLR           = 0.02;
  cInertia      = 0.9;
  cSeed         = 2026;     // shared train seed
  cValSeed      = 7777;     // shared validation seed (disjoint stream)
  cValBatches   = 64;       // validation sequences

  cNumBases = 4;
  cBases: array[0..cNumBases - 1] of TNeuralFloat = (4.0, 6.0, 8.0, 12.0);

  // Build the identical single-head causal attention model; the only knob
  // is the ALiBi slope Base.
  function BuildNet(Base: TNeuralFloat): TNNet;
  var
    NN: TNNet;
    EmbeddedLayer, QKV, Query, Key, ValueT: TNNetLayer;
  begin
    NN := TNNet.Create();
    NN.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
    NN.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
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

    // Reshape to (key, query, 1): X=key, Y=query (layout ALiBi/Mask want).
    NN.AddLayer(TNNetReshape.Create(cSeqLen, cSeqLen, 1));

    // *** THE SWEPT KNOB: ALiBi slope base. ***
    NN.AddLayer(TNNetALiBi.Create(Base));

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

  // target[i] = most recent vowel token at position <= i, else 0 (begin).
  procedure MakePair(const Tokens: array of integer;
    InputV, TargetV: TNNetVolume);
  var
    I, LastVowel: integer;
  begin
    TargetV.Fill(0);
    LastVowel := 0;
    for I := 0 to cSeqLen - 1 do
    begin
      InputV.FData[I] := Tokens[I];
      if Tokens[I] >= cFirstVowel then LastVowel := Tokens[I];
      TargetV[I, 0, LastVowel] := 1.0;
    end;
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
          Pred := NN.GetLastLayer.Output.GetClassOnPixel(I, 0);
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
      WriteLn(Format('  --- training ALiBi Base=%.0f  (slope = 2^-%.0f) ---',
        [Base, Base]));
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
  WriteLn('ALiBiSlopeSweep: same tiny single-head CAUSAL attention model, ',
    'four ALiBi slope bases.');
  WriteLn('Task: copy the most-recent vowel (target[i] = last vowel seen at ',
    'pos <= i).');
  WriteLn(Format('Vocab=%d (vowels=%d..%d), SeqLen=%d, d_model=%d. Slope = ' +
    '2^(-Base*(h+1)/H); here H=1.',
    [cVocab, cFirstVowel, cVocab - 1, cSeqLen, cDModel]));
  WriteLn('Sweeping the "magic 8": Base in {4, 6, 8, 12}.');
  WriteLn;

  GStart := Now();
  for Idx := 0 to cNumBases - 1 do
  begin
    NN := BuildNet(cBases[Idx]);
    try
      if Idx = 0 then
      begin
        WriteLn('Architecture (Base=4 arm shown; other arms differ only in ',
          'the TNNetALiBi base):');
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
  WriteLn('  Base k    slope=2^-k     train-CE     val-CE      val-acc');
  WriteLn('  ------    ----------    ---------    ---------    -------');
  BestIdx := 0;
  for Idx := 0 to cNumBases - 1 do
  begin
    WriteLn(Format('  %5.0f     %.6f      %.5f      %.5f      %5.1f%%',
      [cBases[Idx], Power(2.0, -cBases[Idx]),
       TrainLoss[Idx], ValLoss[Idx], 100.0 * ValAcc[Idx]]));
    if ValLoss[Idx] < ValLoss[BestIdx] then BestIdx := Idx;
  end;
  WriteLn;
  WriteLn(StringOfChar('=', 72));
  WriteLn(Format('Best base by validation CE: k = %.0f  (val-CE=%.5f).',
    [cBases[BestIdx], ValLoss[BestIdx]]));
  if cBases[BestIdx] = 8.0 then
    WriteLn('The cargo-culted 8 wins here.')
  else
    WriteLn('The cargo-culted 8 is NOT the winner on this task -- ',
      'the constant is empirical, not sacred.');
  WriteLn(Format('Total runtime for all %d arms: %.1fs.',
    [cNumBases, GElapsed]));
end.
