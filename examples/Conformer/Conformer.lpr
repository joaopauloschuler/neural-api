program Conformer;
(*
Conformer: a tiny pure-CPU demo of TNNet.AddConformerBlock, the convolution-
augmented transformer block of Gulati et al. 2020 ("Conformer: Convolution-
augmented Transformer for Speech Recognition"). The Conformer block is a
"macaron" sandwich: two HALF-step feed-forward modules wrap a multi-head self-
attention module and a (depthwise-separable-style) convolution module, every
sub-module a pre-norm residual, with a final LayerNorm:
    x := x + 0.5 * FFN(x)        (first half-step FFN)
    x := x + MHSA(x)             (multi-head self-attention -- GLOBAL mixing)
    x := x + ConvModule(x)       (1-D conv over time   -- LOCAL mixing)
    x := x + 0.5 * FFN(x)        (second half-step FFN)
    x := LayerNorm(x)
The whole block is composed from existing serializable primitives (LayerNorm,
PointwiseConvLinear, multi-head self-attention, GLU, TNNetCausalConv1D, Swish,
Sum), so it needs no new class and round-trips through SaveToString/LoadFromString.

WHY A TASK THAT NEEDS BOTH CONV AND ATTENTION. The whole point of a Conformer is
that the convolution module captures LOCAL structure while self-attention
captures GLOBAL/long-range structure. To show the block actually exercises both,
this toy is a PER-TOKEN binary tagging task whose label at position t is the XOR
of:
  (A) a LOCAL feature at t: is the adjacent bigram (S[t], S[t+1]) == (3, 4)? A
      1-D conv over the time axis (kernel >= 2) detects this two-token pattern
      directly; pure attention (a permutation-invariant bag of tokens) struggles
      to localise an ADJACENT pair to position t.
  (B) a GLOBAL feature: is the token at the FAR-AWAY first position == 1? This is
      a single long-range bit that every position must condition on; attention
      routes S[0] to every token in one hop, whereas a small-kernel conv cannot
      see that far.
target[t] = LocalBigram(t) XOR (S[0] == 1). Because it is an XOR, a model that
recovers only the LOCAL feature (conv-only) flips its answer on half the inputs
(every sequence with S[0]==1), and a model that recovers only the GLOBAL feature
(attention-only) can never spot the local bigram -- so reaching high accuracy
REQUIRES both the conv and the attention pathway, exactly what a Conformer block
provides. No pooling: a per-token PointwiseSoftMax readout keeps every position's
gradient alive.

ARCHITECTURE.
  Input(SeqLen)                              token ids on X
   -> Embedding(Vocab, d_model)
   -> SinusoidalPositionalEmbedding          parameter-free positions
   -> AddConformerBlock(Heads, d_ff, K)      macaron FFN | MHSA | conv | FFN
   -> PointwiseConvLinear(2)                  per-token 2-way logits (keeps SeqLen)
   -> PointwiseSoftMax(1)                     per-token softmax across depth

RESULT GATE (prints PASS/FAIL; Halt(1) on failure): per-token test accuracy must
clear a margin (90%) that neither single pathway can reach (the global XOR caps a
local-only model near the rate of S[0]<>1, and a global-only model can never beat
the local bigram, both well under 90%).

Pure CPU, single-threaded, deterministic (fixed RandSeed). Trains and self-checks
in well under 5 minutes on two cores (about a minute in practice).

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
  cVocab   = 6;        // token ids 0..5
  cSeqLen  = 16;       // sequence length
  cDModel  = 24;       // embedding / residual-stream width
  cHeads   = 3;        // attention heads per Conformer block
  cDFF     = 48;       // feed-forward inner width
  cKernel  = 5;        // conv-module kernel size (>= 2 to span the bigram)
  cEpochs  = 160;      // training epochs
  cBatch   = 32;       // sequences per epoch
  cLR      = 0.01;     // per-sample SGD
  cInertia = 0.9;
  cSeed    = 424242;   // repo idiom

  // The LOCAL adjacent-bigram pattern the conv module must detect.
  cPatA = 3; cPatB = 4;
  // The GLOBAL long-range sentinel: target flips when token[0] == this.
  cGlobalTok = 1;

type
  TSeq = array[0..cSeqLen - 1] of integer;

procedure MakeSeq(out S: TSeq);
var i: integer;
begin
  for i := 0 to cSeqLen - 1 do S[i] := Random(cVocab);
  // Seed the LOCAL bigram (cPatA, cPatB) at MANY interior positions (~1/3 of
  // them) so local-firing positions are common: otherwise a random (3,4) is so
  // rare the model can ignore the conv pathway and still score high. Keep clear
  // of position 0 (the global bit) so the two features stay independent.
  for i := 1 to cSeqLen - 2 do
    if Random(3) = 0 then
    begin
      S[i]     := cPatA;
      S[i + 1] := cPatB;
    end;
end;

// Per-token label = LocalBigram(t) XOR GlobalBit.
//   LocalBigram(t) = (S[t]=cPatA) and (S[t+1]=cPatB)   (false at the last pos)
//   GlobalBit      = (S[0] = cGlobalTok)
function TokenLabel(const S: TSeq; t: integer): integer;
var loc, glob: boolean;
begin
  loc  := (t < cSeqLen - 1) and (S[t] = cPatA) and (S[t + 1] = cPatB);
  glob := S[0] = cGlobalTok;
  if loc <> glob then Result := 1 else Result := 0;
end;

procedure FillSample(const S: TSeq; InputV, TargetV: TNNetVolume);
var t: integer;
begin
  TargetV.Fill(0);
  for t := 0 to cSeqLen - 1 do
  begin
    InputV.FData[t] := S[t];
    TargetV.OneHotEncodingOnPixel(t, 0, TokenLabel(S, t));
  end;
end;

function BuildModel(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, cDModel));
  Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());
  // One Conformer block (macaron FFN | MHSA | conv | FFN | LayerNorm).
  Result.AddConformerBlock(cHeads, cDFF, cKernel);
  // Per-token 2-way readout (PointwiseConvLinear keeps the sequence axis; a
  // FullConnect would flatten/mix tokens). PointwiseSoftMax over depth gives a
  // per-position class distribution.
  Result.AddLayer(TNNetPointwiseConvLinear.Create(2));
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));
  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
end;

procedure Train(NN: TNNet);
var
  Epoch, b: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  StartTime, Elapsed: double;
  SumErr: TNeuralFloat;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, 2);
  try
    StartTime := Now();
    for Epoch := 1 to cEpochs do
    begin
      SumErr := 0;
      // Mini-batch update: accumulate gradients over the batch, then step once.
      // This is markedly more stable than per-sample SGD at momentum 0.9 on this
      // deep stack (per-sample noise stalls the XOR; batched updates converge).
      NN.ClearDeltas();
      for b := 1 to cBatch do
      begin
        MakeSeq(S);
        FillSample(S, InputV, TargetV);
        NN.Compute(InputV);
        SumErr := SumErr + NN.GetLastLayer.Output.SumDiff(TargetV);
        NN.Backpropagate(TargetV);
      end;
      NN.UpdateWeights();
      if (Epoch = 1) or (Epoch mod 20 = 0) or (Epoch = cEpochs) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  epoch %4d / %4d   mean-|err|=%.5f   elapsed=%.1fs',
          [Epoch, cEpochs, SumErr / cBatch, Elapsed]));
      end;
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
end;

// Returns overall per-token accuracy and the accuracy split by the two feature
// axes so the contribution of each pathway is visible:
//   AccLocalPos = accuracy at positions where the LOCAL bigram fires
//                 (these are exactly the positions the conv module must localise);
//   AccGlobalOn = accuracy on sequences where the GLOBAL bit is set
//                 (the long-range attention case).
procedure Evaluate(NN: TNNet; out Acc, AccLocalPos, AccGlobalOn: TNeuralFloat);
const cProbes = 800;
var
  k, t, pred, lbl: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  Corr, Tot, CorrLoc, TotLoc, CorrGlob, TotGlob: integer;
  glob, loc: boolean;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, 2);
  Corr := 0; Tot := 0; CorrLoc := 0; TotLoc := 0; CorrGlob := 0; TotGlob := 0;
  try
    for k := 1 to cProbes do
    begin
      MakeSeq(S);
      FillSample(S, InputV, TargetV);
      NN.Compute(InputV);
      glob := S[0] = cGlobalTok;
      for t := 0 to cSeqLen - 1 do
      begin
        lbl := TokenLabel(S, t);
        if NN.GetLastLayer.Output[t, 0, 1] > NN.GetLastLayer.Output[t, 0, 0]
          then pred := 1 else pred := 0;
        if pred = lbl then Inc(Corr);
        Inc(Tot);
        loc := (t < cSeqLen - 1) and (S[t] = cPatA) and (S[t + 1] = cPatB);
        if loc then
        begin
          if pred = lbl then Inc(CorrLoc);
          Inc(TotLoc);
        end;
        if glob then
        begin
          if pred = lbl then Inc(CorrGlob);
          Inc(TotGlob);
        end;
      end;
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  Acc         := Corr / Max(1, Tot);
  AccLocalPos := CorrLoc / Max(1, TotLoc);
  AccGlobalOn := CorrGlob / Max(1, TotGlob);
end;

var
  NN: TNNet;
  Acc, AccLocalPos, AccGlobalOn: TNeuralFloat;
  Gate: boolean;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := cSeed;

  WriteLn('Conformer: a convolution-augmented transformer (Gulati et al. 2020).');
  WriteLn(Format('Toy (per-token tag): target[t] = ((S[t],S[t+1])==(%d,%d)) XOR ' +
    '(S[0]==%d).', [cPatA, cPatB, cGlobalTok]));
  WriteLn('The adjacent bigram is a LOCAL pattern (needs the conv module); S[0] is');
  WriteLn('a LONG-RANGE bit (needs self-attention). The XOR forces the model to use');
  WriteLn(Format('BOTH pathways. Seqlen %d, vocab %d.', [cSeqLen, cVocab]));
  WriteLn;

  NN := BuildModel();
  try
    WriteLn('Architecture (one AddConformerBlock builder):');
    NN.PrintSummary();
    WriteLn;

    WriteLn('Training for ', cEpochs, ' epochs of batch size ', cBatch, '...');
    Train(NN);
    WriteLn;

    Evaluate(NN, Acc, AccLocalPos, AccGlobalOn);
    WriteLn(StringOfChar('=', 68));
    WriteLn(Format('Per-token test accuracy (overall)        : %6.2f%%', [100.0 * Acc]));
    WriteLn(Format('  ... at LOCAL bigram-firing positions   : %6.2f%%   (conv pathway)',
      [100.0 * AccLocalPos]));
    WriteLn(Format('  ... on GLOBAL-bit-set sequences        : %6.2f%%   (attention pathway)',
      [100.0 * AccGlobalOn]));
    WriteLn(StringOfChar('=', 68));

    Gate := Acc > 0.90;
    if Gate then
      WriteLn('GATE (acc > 90%) : PASS  -- the Conformer used BOTH conv and attention.')
    else
      WriteLn('GATE (acc > 90%) : FAIL');
  finally
    NN.Free;
  end;

  if not Gate then Halt(1);
end.
