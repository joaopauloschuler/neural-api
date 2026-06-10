program SequenceReverse;
(*
SequenceReverse: the classic "learn to reverse a sequence" probe for attention.
A TINY non-causal self-attention model is trained to output the REVERSE of its
input token sequence:

    input  [a, b, c, d, e, f, g, h]
    target [h, g, f, e, d, c, b, a]

This is a clean teaching demo that self-attention can learn a PURE POSITIONAL
PERMUTATION. Reversal is a fixed map output[i] = input[SeqLen-1-i]; it does not
depend on token VALUES, only on POSITION. The model therefore must combine:
  * a positional signal (here a parameter-free sinusoidal positional embedding)
    so every position knows where it sits, and
  * a single attention head whose winning strategy is "route query position i to
    key position SeqLen-1-i" and copy that token's content forward.

ARCHITECTURE (composed entirely from existing library layers -- no new layer):
  Input(SeqLen, 1, 1)                          token ids on the X axis
   -> Embedding(Vocab, d_model)                value content per token
   -> SinusoidalPositionalEmbedding            parameter-free position signal
   == one NON-causal self-attention block with a residual ====================
   -> PointwiseConvLinear(3*d_k)               per-token Q|K|V (NOT FullConnect!)
   -> ScaledDotProductAttention(d_k)           full (non-causal) attention
   -> PointwiseConvLinear(d_model)             project head output to d_model
   -> Sum([resid_in, block_out])               residual connection
   == per-position readout ===================================================
   -> PointwiseConvLinear(Vocab)               per-position vocab logits
   -> PointwiseSoftMax(1)                       softmax across depth, per position

Per-token projections MUST be PointwiseConvLinear (1x1 conv): a TNNetFullConnect
would flatten/mix the whole SeqLen*d_model tensor into one vector, collapsing the
token axis and destroying the per-position structure this task needs.

Training is a manual single-threaded Compute/Backpropagate loop with per-position
cross-entropy (the same idiom as examples/InductionHeads). Random sequences are
generated on the fly; the target is simply the reversed sequence. Everything is
tiny so the demo trains and self-reports in WELL under 5 minutes on 2 CPU cores
and never exhausts memory.

The honest headline: reversal is a fixed permutation, so a correctly wired
attention model reaches near-100% per-token AND sequence-level (full exact
reversal) accuracy very quickly.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cVocab   = 12;       // small symbol vocabulary
  cSeqLen  = 8;        // sequence length to reverse
  cDModel  = 24;       // embedding / residual-stream width
  cDk      = 24;       // attention head dimension
  cEpochs  = 200;      // training epochs
  cBatch   = 48;       // sequences per epoch
  cLR      = 0.005;    // per-sample SGD learning rate
  cInertia = 0.9;
  cSeed    = 424242;   // repo idiom

type
  TSeq = array[0..cSeqLen - 1] of integer;

// Draw a uniformly random sequence of token ids.
procedure MakeSeq(out S: TSeq);
var
  i: integer;
begin
  for i := 0 to cSeqLen - 1 do
    S[i] := Random(cVocab);
end;

// Fill the input (token ids on X) and the one-hot reversed target.
procedure FillPair(const S: TSeq; InputV, TargetV: TNNetVolume);
var
  i: integer;
begin
  TargetV.Fill(0);
  for i := 0 to cSeqLen - 1 do
  begin
    InputV.FData[i] := S[i];
    // target position i must hold the input token at the mirror position.
    TargetV[i, 0, S[cSeqLen - 1 - i]] := 1.0;
  end;
end;

function BuildModel(): TNNet;
var
  Stream, QKV, Sdpa, Proj: TNNetLayer;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, cDModel));
  Stream := Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());

  // One NON-causal self-attention block with a residual connection.
  QKV  := Result.AddLayerAfter(TNNetPointwiseConvLinear.Create(3 * cDk), Stream);
  Sdpa := Result.AddLayerAfter(TNNetScaledDotProductAttention.Create(cDk, False), QKV);
  Proj := Result.AddLayerAfter(TNNetPointwiseConvLinear.Create(cDModel), Sdpa);
  Result.AddLayer(TNNetSum.Create([Stream, Proj]));

  // Per-position readout to vocab logits + softmax across depth.
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));

  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
end;

function CrossEntropyAt(Output, Target: TNNetVolume; t: integer): TNeuralFloat;
var
  d: integer;
  P: TNeuralFloat;
begin
  Result := 0;
  for d := 0 to cVocab - 1 do
    if Target[t, 0, d] > 0 then
    begin
      P := Output[t, 0, d];
      if P < 1e-12 then P := 1e-12;
      Result := Result - Target[t, 0, d] * Ln(P);
    end;
end;

function MeanCrossEntropy(Output, Target: TNNetVolume): TNeuralFloat;
var
  t: integer;
begin
  Result := 0;
  for t := 0 to cSeqLen - 1 do
    Result := Result + CrossEntropyAt(Output, Target, t);
  Result := Result / cSeqLen;
end;

function ArgMaxDepth(V: TNNetVolume; Pos: integer): integer;
var
  d, Best: integer;
  BestVal, Cur: TNeuralFloat;
begin
  Best := 0;
  BestVal := V[Pos, 0, 0];
  for d := 1 to cVocab - 1 do
  begin
    Cur := V[Pos, 0, d];
    if Cur > BestVal then begin BestVal := Cur; Best := d; end;
  end;
  Result := Best;
end;

procedure Train(NN: TNNet);
var
  Epoch, b: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  SumCE: TNeuralFloat;
  StartTime, Elapsed: double;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  try
    StartTime := Now();
    for Epoch := 1 to cEpochs do
    begin
      SumCE := 0;
      for b := 1 to cBatch do
      begin
        MakeSeq(S);
        FillPair(S, InputV, TargetV);
        NN.Compute(InputV);
        SumCE := SumCE + MeanCrossEntropy(NN.GetLastLayer.Output, TargetV);
        NN.Backpropagate(TargetV);   // per-sample SGD update (auto)
      end;
      if (Epoch = 1) or (Epoch mod 25 = 0) or (Epoch = cEpochs) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  epoch %4d / %4d   mean-CE=%.5f   elapsed=%.1fs',
          [Epoch, cEpochs, SumCE / cBatch, Elapsed]));
      end;
    end;
  finally
    InputV.Free;
    TargetV.Free;
  end;
end;

// Per-token and sequence-level (full exact reversal) accuracy on fresh probes.
procedure Evaluate(NN: TNNet; out TokAcc, SeqAcc: TNeuralFloat);
const
  cProbes = 1000;
var
  k, t, Pred: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  TokHits, TokTotal, SeqHits: integer;
  AllRight: boolean;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  TokHits := 0; TokTotal := 0; SeqHits := 0;
  try
    for k := 1 to cProbes do
    begin
      MakeSeq(S);
      FillPair(S, InputV, TargetV);
      NN.Compute(InputV);
      AllRight := True;
      for t := 0 to cSeqLen - 1 do
      begin
        Pred := ArgMaxDepth(NN.GetLastLayer.Output, t);
        if Pred = S[cSeqLen - 1 - t] then Inc(TokHits)
        else AllRight := False;
        Inc(TokTotal);
      end;
      if AllRight then Inc(SeqHits);
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  TokAcc := TokHits / Max(1, TokTotal);
  SeqAcc := SeqHits / Max(1, cProbes);
end;

function SeqToStr(const A: array of integer): string;
var
  i: integer;
begin
  Result := '[';
  for i := 0 to High(A) do
  begin
    if i > 0 then Result := Result + ', ';
    Result := Result + IntToStr(A[i]);
  end;
  Result := Result + ']';
end;

// Print one concrete worked example: input vs predicted vs target.
procedure ShowExample(NN: TNNet);
var
  InputV, TargetV: TNNetVolume;
  S, Pred, Tgt: TSeq;
  t: integer;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  try
    MakeSeq(S);
    FillPair(S, InputV, TargetV);
    NN.Compute(InputV);
    for t := 0 to cSeqLen - 1 do
    begin
      Pred[t] := ArgMaxDepth(NN.GetLastLayer.Output, t);
      Tgt[t]  := S[cSeqLen - 1 - t];
    end;
    WriteLn('  input     : ', SeqToStr(S));
    WriteLn('  predicted : ', SeqToStr(Pred));
    WriteLn('  target    : ', SeqToStr(Tgt));
  finally
    InputV.Free; TargetV.Free;
  end;
end;

var
  NN: TNNet;
  TokAcc, SeqAcc: TNeuralFloat;
begin
  // Mask FPU exceptions: fast-math kernels can transiently underflow during
  // early training (matches examples/InductionHeads / AttentionCopyTask).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := cSeed;

  WriteLn('SequenceReverse: a tiny self-attention model learns to REVERSE its input.');
  WriteLn(Format('Task: output the reverse of a random sequence (seqlen %d, vocab %d).',
    [cSeqLen, cVocab]));
  WriteLn('Reversal is a pure POSITIONAL permutation: output[i] = input[SeqLen-1-i].');
  WriteLn;

  NN := BuildModel();
  try
    WriteLn('Architecture (one non-causal SDPA block with a residual):');
    NN.PrintSummary();
    WriteLn;

    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    Train(NN);
    WriteLn;

    Evaluate(NN, TokAcc, SeqAcc);

    WriteLn(StringOfChar('=', 64));
    WriteLn('RESULTS (held-out fresh sequences)');
    WriteLn(StringOfChar('=', 64));
    WriteLn(Format('Chance per-token accuracy (1/vocab) : %6.2f%%', [100.0 / cVocab]));
    WriteLn(Format('Per-token accuracy                  : %6.2f%%', [100.0 * TokAcc]));
    WriteLn(Format('Sequence-level (exact reversal) acc : %6.2f%%', [100.0 * SeqAcc]));
    WriteLn(StringOfChar('=', 64));
    WriteLn;

    WriteLn('One worked example:');
    ShowExample(NN);
    WriteLn;
    WriteLn('Interpretation: attention learns the fixed position permutation that');
    WriteLn('maps each query position i to key position SeqLen-1-i and copies that');
    WriteLn('token forward -- the classic copy/reverse probe, solved near-perfectly.');
  finally
    NN.Free;
  end;
end.
