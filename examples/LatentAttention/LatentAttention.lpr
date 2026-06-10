program LatentAttention;
(*
LatentAttention: a tiny demonstration of Multi-head Latent Attention (MLA, the
DeepSeek-V2 attention of Liu et al. 2024) built with
TNNet.AddMultiHeadLatentAttention, plus a head-to-head next-token bake-off
against a parameter-matched plain multi-head self-attention (MHA).

WHAT MLA IS (and how it differs from MHA / GQA).
Plain MHA projects each token to a full-width Key and Value (total 2*d_model per
token) and -- in an incremental decoder -- must CACHE both, so the KV cache grows
with d_model and head count. GQA shrinks that cache by SHARING K/V across query-
head groups. MLA takes a different, ORTHOGONAL route: it LOW-RANK-FACTORS the K/V
projection. Each token is first DOWN-projected to one tiny shared latent

    c_KV := x . W_DKV        (width d_c = LatentDim, with d_c << d_model)

and K and V for every head are then reconstructed by UP-projections
K := c_KV . W_UK, V := c_KV . W_UV. In a decoder, c_KV (width d_c) is the ONLY
per-token state that needs caching, so the cacheable state shrinks from
2*d_model (full K and V) to just d_c -- a saving of

    d_c / (2 * d_model)

that is RANK-based and independent of the number of heads. Query is projected
from d_model directly (per head) exactly as in MHA.

THE DECOUPLED-RoPE SLICE (RopeDim > 0).
RoPE cannot be applied to the compressed latent c_KV: the per-head up-projection
would smear positions across channels. The paper's fix (carried by the builder's
RopeDim parameter) keeps the content path position-free and adds a SMALL extra
rope dimension: rope-Q is a per-head projection of x with RoPE applied; rope-K
is a SINGLE projection of x (shared by ALL heads) with RoPE applied. Each head
attends with concat(Q_h, ropeQ_h) . concat(K_h, ropeK), so the cacheable decode
state grows by only RopeDim (the one shared rotated rope-K slice).

WHAT THIS PROGRAM SHOWS.
  1. The cacheable-state saving d_c/(2*d_model) reported numerically.
  2. A tiny next-token COPY task (fixed long-range offset; the same family of
     task used in SequenceMixerBakeoff) trained with three attention arms:
       - Arm MLA:      Embedding -> SinPos -> AddMultiHeadLatentAttention(causal)
                       -> MLP -> softmax  (NoPE scores; absolute SinPos input).
       - Arm MLA+RoPE: Embedding -> AddMultiHeadLatentAttention(causal,
                       RopeDim>0) -> MLP -> softmax. NO SinPos: position enters
                       ONLY through the decoupled rope slice, proving it works.
       - Arm MHA:      Embedding -> SinPos -> [QKV slab] ->
                       AddMultiHeadSelfAttention(causal) -> MLP -> softmax.
     All arms share the embedding front-end, MLP read-out, data and weight
     init; only the attention block (and positional pathway) differs.
  3. KV-CACHE INCREMENTAL DECODE (the MLA headline win), two demonstrations:
     a) Cache-machinery decode: the SAME random-weight MLA stack (NoPE and
        decoupled-RoPE) is run once as a full causal forward and once token-at-
        a-time through a weight-copied step net whose per-head SDPA layers use
        BeginIncrementalDecode; for the RoPE arm each TNNetRotaryEmbedding gets
        PositionOffset := t so a streamed length-1 token is rotated with its
        ABSOLUTE position. Faithfulness asserted to < 1e-5.
     b) TRUE LATENT-ONLY CACHING: a decode loop whose ONLY per-token state is
        the latent c_KV stream -- each step re-runs the K/V up-projections over
        the cached latents and performs one query row of attention manually.
        Output again matches the full forward to < 1e-5, demonstrating that
        d_c floats/token really is sufficient decode state.
     The printed table compares cache bytes/token: MHA full K+V (2*d_model) vs
     the SDPA-level K/V caches of (a) vs the latent-only cache of (b).

HONEST SCOPE.
  - Arm (a) caches post-up-projection per-head K/V (the existing SDPA cache
    machinery), so its MEMORY matches MHA; it demonstrates the O(1)-per-step
    compute path. Arm (b) is the memory win: latent-only state at d_c
    floats/token, paying an O(t) up-projection recompute per step.
  - Per-arm bake-off numbers are SEED-DEPENDENT; the point is the mechanism +
    the cache arithmetic, not crowning a winner.

Pure CPU, single-threaded (manual Compute/Backpropagate), no external dataset,
finishes well under a minute.

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

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cVocab   = 6;        // small char vocabulary
  cSeqLen  = 16;       // sequence length
  cLag     = 4;        // fixed long-range source offset
  cDModel  = 24;       // embedding / attention-stream width
  cHeads   = 4;        // attention heads (d_model must be divisible by heads)
  cLatent  = 8;        // MLA latent dim d_c (the cacheable c_KV width, d_c<<d_model)
  cRope    = 4;        // decoupled-RoPE slice width (even; per-head rope-Q + shared rope-K)
  cDecLen  = 24;       // sequence length for the incremental-decode demos
  cDFF     = 32;       // hidden width of the pointwise MLP read-out
  cEpochs  = 300;      // training epochs
  cBatch   = 48;       // sequences per epoch
  cLR      = 0.005;    // per-sample SGD
  cInertia = 0.9;
  cSeed    = 424242;   // repo idiom
  cProbes  = 600;      // evaluation probes

type
  TArmKind = (akMLA, akMLARope, akMHA);

  TSeq = array[0..cSeqLen - 1] of integer;

  TArmResult = record
    Name      : string;
    Params    : integer;
    InitLoss  : TNeuralFloat;
    FinalLoss : TNeuralFloat;
    Acc       : TNeuralFloat;
    AccLong   : TNeuralFloat;   // positions t >= cLag (long-range region)
    Seconds   : TNeuralFloat;
  end;

procedure MakeSeq(out S: TSeq);
var i: integer;
begin
  for i := 0 to cSeqLen - 1 do S[i] := Random(cVocab);
end;

// Two-regime fixed-offset copy target (reachable by attention via positions).
function TargetTok(const S: TSeq; t: integer): integer;
begin
  if t < cLag then
  begin
    if t - 1 >= 0 then Result := S[t - 1] else Result := 0;
  end
  else
    Result := S[t - cLag];
end;

procedure FillPair(const S: TSeq; InputV, TargetV: TNNetVolume);
var t: integer;
begin
  TargetV.Fill(0);
  for t := 0 to cSeqLen - 1 do
  begin
    InputV.FData[t] := S[t];
    TargetV[t, 0, TargetTok(S, t)] := 1.0;
  end;
end;

// Shared front-end + read-out; only the attention block in the middle moves.
//   Input -> Embedding(V,d_model) -> SinusoidalPositionalEmbedding
//         -> [ MLA(causal) | (QKV slab -> MHA causal) ]
//         -> PointwiseConvLinear(d_ff) -> ReLU
//         -> PointwiseConvLinear(V) -> PointwiseSoftMax(depth)
function BuildNet(Kind: TArmKind): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
  // Parameter-free absolute positions -- EXCEPT for the decoupled-RoPE arm,
  // where position must enter ONLY through the rotated rope slice (mixing
  // absolute and rotary positions would be a hybrid, not RoPE).
  if Kind <> akMLARope then
    Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());

  case Kind of
    akMLA:
      // The builder does its OWN Q projection + the low-rank c_KV down/up
      // projections from the (SeqLen,1,d_model) token stream.
      Result.AddMultiHeadLatentAttention(cDModel, cHeads, cLatent, True);
    akMLARope:
      // DeepSeek-V2 decoupled-RoPE slice: per-head rotated rope-Q + ONE
      // shared rotated rope-K of width cRope concatenated to the content
      // slices before the dot product. No other positional pathway.
      Result.AddMultiHeadLatentAttention(cDModel, cHeads, cLatent, True, cRope);
    akMHA:
      begin
        // MHA consumes a pre-projected Q|K|V slab of depth 3*d_model.
        Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDModel));
        Result.AddMultiHeadSelfAttention(cHeads, True);
      end;
  end;

  // Per-token MLP read-out (PointwiseConvLinear keeps the sequence axis).
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cDFF));
  Result.AddLayer(TNNetReLU.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));

  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
end;

function CrossEntropyAt(Output, Target: TNNetVolume; t: integer): TNeuralFloat;
var d: integer; P: TNeuralFloat;
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
var t: integer;
begin
  Result := 0;
  for t := 0 to cSeqLen - 1 do
    Result := Result + CrossEntropyAt(Output, Target, t);
  Result := Result / cSeqLen;
end;

function ArgMaxDepth(V: TNNetVolume; Pos: integer): integer;
var d, Best: integer; BestVal, Cur: TNeuralFloat;
begin
  Best := 0; BestVal := V[Pos, 0, 0];
  for d := 1 to cVocab - 1 do
  begin
    Cur := V[Pos, 0, d];
    if Cur > BestVal then begin BestVal := Cur; Best := d; end;
  end;
  Result := Best;
end;

function EvalMeanCE(NN: TNNet): TNeuralFloat;
var k: integer; InputV, TargetV: TNNetVolume; S: TSeq; Sum: TNeuralFloat;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  Sum := 0;
  try
    for k := 1 to cProbes do
    begin
      MakeSeq(S);
      FillPair(S, InputV, TargetV);
      NN.Compute(InputV);
      Sum := Sum + MeanCrossEntropy(NN.GetLastLayer.Output, TargetV);
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  Result := Sum / cProbes;
end;

procedure Evaluate(NN: TNNet; out Acc, AccLong: TNeuralFloat);
var
  k, t, Pred, Tgt: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  C, Ctot, Cl, Nl: integer;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  C := 0; Ctot := 0; Cl := 0; Nl := 0;
  try
    for k := 1 to cProbes do
    begin
      MakeSeq(S);
      FillPair(S, InputV, TargetV);
      NN.Compute(InputV);
      for t := 0 to cSeqLen - 1 do
      begin
        Pred := ArgMaxDepth(NN.GetLastLayer.Output, t);
        Tgt  := TargetTok(S, t);
        if Pred = Tgt then Inc(C);
        Inc(Ctot);
        if t >= cLag then
        begin
          if Pred = Tgt then Inc(Cl);
          Inc(Nl);
        end;
      end;
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  Acc     := C  / Max(1, Ctot);
  AccLong := Cl / Max(1, Nl);
end;

function RunArm(Kind: TArmKind; const AName: string): TArmResult;
var
  NN: TNNet;
  Epoch, b: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  StartTime: double;
begin
  Result.Name := AName;
  RandSeed := cSeed;             // identical embedding init per arm
  NN := BuildNet(Kind);
  try
    Result.Params := NN.CountWeights();

    RandSeed := cSeed + 1;       // shared probe stream for the init loss
    Result.InitLoss := EvalMeanCE(NN);

    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    RandSeed := cSeed;           // identical training stream per arm
    StartTime := Now();
    try
      for Epoch := 1 to cEpochs do
        for b := 1 to cBatch do
        begin
          MakeSeq(S);
          FillPair(S, InputV, TargetV);
          NN.Compute(InputV);
          NN.Backpropagate(TargetV);
        end;
    finally
      InputV.Free; TargetV.Free;
    end;
    Result.Seconds := (Now() - StartTime) * 86400.0;

    RandSeed := cSeed + 1;
    Result.FinalLoss := EvalMeanCE(NN);
    RandSeed := cSeed + 2;
    Evaluate(NN, Result.Acc, Result.AccLong);
  finally
    NN.Free;
  end;
end;

// ===========================================================================
// KV-cache incremental decode demos (random weights; no training needed).
// ===========================================================================

// Bare causal MLA stack on a raw (SeqLen,1,d_model) stream (no embedding).
// Builder layer order: 0=Input, 1=QProj, 2=CKV down-proj, 3=K up-proj,
// 4=V up-proj, ... per-head wiring ..., last=out-projection.
function BuildBareMLA(SeqLen, RopeDim: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(SeqLen, 1, cDModel));
  Result.AddMultiHeadLatentAttention(cDModel, cHeads, cLatent, True, RopeDim);
end;

// (a) Cache-machinery decode: one full causal forward vs token-at-a-time
// through a weight-copied step net whose per-head SDPA layers run in
// BeginIncrementalDecode mode. For RopeDim>0 every TNNetRotaryEmbedding gets
// PositionOffset := t before each step (a streamed length-1 token must be
// rotated with its ABSOLUTE position, not position 0). Returns max |diff|.
function RunCachedDecode(RopeDim: integer): TNeuralFloat;
var
  NNFull, NNStep: TNNet;
  FullIn, StepIn: TNNetVolume;
  T, D, LayerCnt: integer;
  Diff: TNeuralFloat;
begin
  Result := 0;
  NNFull := BuildBareMLA(cDecLen, RopeDim);
  NNStep := BuildBareMLA(1, RopeDim);
  FullIn := TNNetVolume.Create(cDecLen, 1, cDModel);
  StepIn := TNNetVolume.Create(1, 1, cDModel);
  try
    NNStep.CopyWeights(NNFull);
    for T := 0 to cDecLen - 1 do
      for D := 0 to cDModel - 1 do
        FullIn[T, 0, D] := (Random(2000) - 1000) / 1000;
    NNFull.Compute(FullIn);

    for LayerCnt := 0 to NNStep.Layers.Count - 1 do
      if NNStep.Layers[LayerCnt] is TNNetScaledDotProductAttention then
        TNNetScaledDotProductAttention(NNStep.Layers[LayerCnt]).
          BeginIncrementalDecode(cDecLen);

    for T := 0 to cDecLen - 1 do
    begin
      for D := 0 to cDModel - 1 do
        StepIn[0, 0, D] := FullIn[T, 0, D];
      for LayerCnt := 0 to NNStep.Layers.Count - 1 do
        if NNStep.Layers[LayerCnt] is TNNetRotaryEmbedding then
          TNNetRotaryEmbedding(NNStep.Layers[LayerCnt]).PositionOffset := T;
      NNStep.Compute(StepIn);
      for D := 0 to cDModel - 1 do
      begin
        Diff := Abs(NNFull.GetLastLayer.Output[T, 0, D] -
          NNStep.GetLastLayer.Output[0, 0, D]);
        if Diff > Result then Result := Diff;
      end;
    end;
  finally
    StepIn.Free;
    FullIn.Free;
    NNStep.Free;
    NNFull.Free;
  end;
end;

// (b) TRUE LATENT-ONLY CACHING (NoPE arm). The ONLY per-token decode state is
// the latent c_KV stream (cLatent floats/token). Each step:
//   1. projects the new token to its query Q_t and latent c_t (cache c_t);
//   2. RE-RUNS the K/V up-projections over ALL cached latents (O(t) compute
//      traded for the d_c-only memory footprint);
//   3. performs the single query row of per-head attention + out-projection.
// The result must match the full causal forward at every position.
function RunLatentOnlyDecode(): TNeuralFloat;
var
  NNFull, ProjNet, ReconNet, OutNet: TNNet;
  QP, CP, KP, VP: TNNetLayer;
  FullIn, StepIn, LatIn, HeadIn: TNNetVolume;
  LatentCache: array of TNeuralFloat;          // THE decode state: d_c/token
  KRow, VRow: array of TNeuralFloat;           // step-local recompute scratch
  Scores: array of TNeuralFloat;
  T, J, D, H, d_k: integer;
  MaxScore, SumExp, Acc, Diff, InvSqrtDk: TNeuralFloat;
begin
  Result := 0;
  d_k := cDModel div cHeads;
  InvSqrtDk := 1.0 / Sqrt(d_k);
  NNFull := BuildBareMLA(cDecLen, 0);
  // Step nets re-using the FULL net's weights (per-layer CopyWeights):
  // token -> (Q_t, c_t); latent -> (K_j, V_j); head concat -> output.
  ProjNet := TNNet.Create();
  ProjNet.AddLayer(TNNetInput.Create(1, 1, cDModel));
  QP := ProjNet.AddLayerAfter(TNNetPointwiseConvLinear.Create(cDModel),
    ProjNet.Layers[0]);
  CP := ProjNet.AddLayerAfter(TNNetPointwiseConvLinear.Create(cLatent),
    ProjNet.Layers[0]);
  ReconNet := TNNet.Create();
  ReconNet.AddLayer(TNNetInput.Create(1, 1, cLatent));
  KP := ReconNet.AddLayerAfter(TNNetPointwiseConvLinear.Create(cDModel),
    ReconNet.Layers[0]);
  VP := ReconNet.AddLayerAfter(TNNetPointwiseConvLinear.Create(cDModel),
    ReconNet.Layers[0]);
  OutNet := TNNet.Create();
  OutNet.AddLayer(TNNetInput.Create(1, 1, cDModel));
  OutNet.AddLayer(TNNetPointwiseConvLinear.Create(cDModel));
  FullIn := TNNetVolume.Create(cDecLen, 1, cDModel);
  StepIn := TNNetVolume.Create(1, 1, cDModel);
  LatIn  := TNNetVolume.Create(1, 1, cLatent);
  HeadIn := TNNetVolume.Create(1, 1, cDModel);
  SetLength(LatentCache, cDecLen * cLatent);
  SetLength(KRow, cDecLen * cDModel);
  SetLength(VRow, cDecLen * cDModel);
  SetLength(Scores, cDecLen);
  try
    QP.CopyWeights(NNFull.Layers[1]);                          // W_Q
    CP.CopyWeights(NNFull.Layers[2]);                          // W_DKV
    KP.CopyWeights(NNFull.Layers[3]);                          // W_UK
    VP.CopyWeights(NNFull.Layers[4]);                          // W_UV
    OutNet.Layers[1].CopyWeights(
      NNFull.Layers[NNFull.Layers.Count - 1]);                 // W_O

    for T := 0 to cDecLen - 1 do
      for D := 0 to cDModel - 1 do
        FullIn[T, 0, D] := (Random(2000) - 1000) / 1000;
    NNFull.Compute(FullIn);

    for T := 0 to cDecLen - 1 do
    begin
      // 1. New token -> query + latent; cache ONLY the latent.
      for D := 0 to cDModel - 1 do
        StepIn[0, 0, D] := FullIn[T, 0, D];
      ProjNet.Compute(StepIn);
      for D := 0 to cLatent - 1 do
        LatentCache[T * cLatent + D] := CP.Output.Raw[D];
      // 2. Recompute K/V for every cached position from the latents alone.
      for J := 0 to T do
      begin
        for D := 0 to cLatent - 1 do
          LatIn.Raw[D] := LatentCache[J * cLatent + D];
        ReconNet.Compute(LatIn);
        for D := 0 to cDModel - 1 do
        begin
          KRow[J * cDModel + D] := KP.Output.Raw[D];
          VRow[J * cDModel + D] := VP.Output.Raw[D];
        end;
      end;
      // 3. One query row of attention per head (causal by construction).
      for H := 0 to cHeads - 1 do
      begin
        MaxScore := -1e30;
        for J := 0 to T do
        begin
          Acc := 0;
          for D := 0 to d_k - 1 do
            Acc := Acc + QP.Output.Raw[H * d_k + D] *
              KRow[J * cDModel + H * d_k + D];
          Scores[J] := Acc * InvSqrtDk;
          if Scores[J] > MaxScore then MaxScore := Scores[J];
        end;
        SumExp := 0;
        for J := 0 to T do
        begin
          Scores[J] := Exp(Scores[J] - MaxScore);
          SumExp := SumExp + Scores[J];
        end;
        for D := 0 to d_k - 1 do
        begin
          Acc := 0;
          for J := 0 to T do
            Acc := Acc + Scores[J] * VRow[J * cDModel + H * d_k + D];
          HeadIn.Raw[H * d_k + D] := Acc / SumExp;
        end;
      end;
      OutNet.Compute(HeadIn);
      for D := 0 to cDModel - 1 do
      begin
        Diff := Abs(NNFull.GetLastLayer.Output[T, 0, D] -
          OutNet.GetLastLayer.Output.Raw[D]);
        if Diff > Result then Result := Diff;
      end;
    end;
  finally
    SetLength(Scores, 0);
    SetLength(VRow, 0);
    SetLength(KRow, 0);
    SetLength(LatentCache, 0);
    HeadIn.Free;
    LatIn.Free;
    StepIn.Free;
    FullIn.Free;
    OutNet.Free;
    ReconNet.Free;
    ProjNet.Free;
    NNFull.Free;
  end;
end;

function SafeF(V: TNeuralFloat; Dec: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then Result := 'NaN'
  else Result := FloatToStrF(V, ffFixed, 8, Dec);
end;

procedure PrintRow(const R: TArmResult);
begin
  WriteLn(Format('%-8s %8d %9s %9s %8s %9s %7s',
    [R.Name, R.Params, SafeF(R.InitLoss, 3), SafeF(R.FinalLoss, 3),
     SafeF(R.Acc, 3), SafeF(R.AccLong, 3), SafeF(R.Seconds, 2)]));
end;

var
  Arm: array[0..2] of TArmResult;
  Uniform, Saving: TNeuralFloat;
  DiffNoPE, DiffRope, DiffLatent: TNeuralFloat;
  i: integer;
  GateFinite, GateBeatBaseline, GateDecode: boolean;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  Uniform := Ln(cVocab);
  Saving  := cLatent / (2.0 * cDModel);

  WriteLn('LatentAttention: Multi-head Latent Attention (MLA, DeepSeek-V2) demo + ',
    'bake-off vs MHA.');
  WriteLn(Format('d_model=%d, heads=%d, MLA latent d_c=%d, vocab=%d, seqlen=%d.',
    [cDModel, cHeads, cLatent, cVocab, cSeqLen]));
  WriteLn;
  WriteLn('=== MLA cacheable-state saving ===');
  WriteLn(Format('Plain MHA caches full-width K AND V: 2*d_model = %d floats per token.',
    [2 * cDModel]));
  WriteLn(Format('MLA caches only the shared latent c_KV: d_c = %d floats per token.',
    [cLatent]));
  WriteLn(Format('Cacheable-state saving d_c/(2*d_model) = %d/%d = %.4f '
    + '(%.1f%% of the MHA cache).',
    [cLatent, 2 * cDModel, Saving, 100.0 * Saving]));
  WriteLn(Format('With the decoupled-RoPE slice the decode state is d_c+RopeDim = %d+%d = %d',
    [cLatent, cRope, cLatent + cRope]));
  WriteLn('floats/token (the rope-K slice is ONE shared projection, not per-head).');
  WriteLn;

  Write('Training Arm MLA (AddMultiHeadLatentAttention, NoPE + SinPos) ...');
  Arm[0] := RunArm(akMLA, 'MLA');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[0].FinalLoss, 4), SafeF(Arm[0].Seconds, 2)]));

  Write('Training Arm MLA+RoPE (RopeDim=' + IntToStr(cRope) + ', NO absolute positions) ...');
  Arm[1] := RunArm(akMLARope, 'MLA+RoPE');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[1].FinalLoss, 4), SafeF(Arm[1].Seconds, 2)]));

  Write('Training Arm MHA (AddMultiHeadSelfAttention) ...');
  Arm[2] := RunArm(akMHA, 'MHA');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[2].FinalLoss, 4), SafeF(Arm[2].Seconds, 2)]));
  WriteLn;

  WriteLn('=== Next-token bake-off (lower CE is better; acc is argmax over vocab) ===');
  WriteLn(Format('Task: fixed-offset copy target[t]=S[t-%d] for t>=%d (else S[t-1]).',
    [cLag, cLag]));
  WriteLn(Format('%-8s %8s %9s %9s %8s %9s %7s',
    ['arm', 'params', 'init_CE', 'final_CE', 'acc', 'acc_long', 'sec']));
  for i := 0 to 2 do PrintRow(Arm[i]);
  WriteLn;
  WriteLn(Format('acc_long = positions t>=%d (the long-range copy region). '
    + 'Uniform baseline CE=ln(vocab)=%.4f.', [cLag, Uniform]));
  WriteLn('MLA+RoPE has NO absolute positional embedding: long-range accuracy there');
  WriteLn('is only reachable through the decoupled rope slice.');
  WriteLn;

  WriteLn('=== KV-cache incremental decode (the MLA headline win) ===');
  WriteLn(Format('Random-weight causal MLA stack, %d tokens decoded one at a time.',
    [cDecLen]));
  RandSeed := cSeed + 3;
  DiffNoPE := RunCachedDecode(0);
  DiffRope := RunCachedDecode(cRope);
  DiffLatent := RunLatentOnlyDecode();
  WriteLn(Format('(a) SDPA cache machinery, NoPE:           max |full - cached| = %.9f',
    [DiffNoPE]));
  WriteLn(Format('(a) SDPA cache machinery, RopeDim=%d:      max |full - cached| = %.9f',
    [cRope, DiffRope]));
  WriteLn('    (each TNNetRotaryEmbedding gets PositionOffset := t per step so the');
  WriteLn('     streamed length-1 token is rotated with its ABSOLUTE position)');
  WriteLn(Format('(b) TRUE latent-only cache (d_c floats/token): max |full - latent| = %.9f',
    [DiffLatent]));
  GateDecode := (DiffNoPE < 1e-5) and (DiffRope < 1e-5) and (DiffLatent < 1e-5);
  WriteLn;
  WriteLn('KV-cache memory per token (4-byte floats):');
  WriteLn(Format('  %-46s %4d floats = %4d bytes', ['MHA full K+V (2*d_model):',
    2 * cDModel, 8 * cDModel]));
  WriteLn(Format('  %-46s %4d floats = %4d bytes',
    ['MLA via SDPA caches, arm (a) NoPE (2*d_model):',
    2 * cHeads * (cDModel div cHeads), 8 * cHeads * (cDModel div cHeads)]));
  WriteLn(Format('  %-46s %4d floats = %4d bytes',
    [Format('MLA via SDPA caches, arm (a) RopeDim=%d:', [cRope]),
    2 * cHeads * (cDModel div cHeads + cRope),
    8 * cHeads * (cDModel div cHeads + cRope)]));
  WriteLn(Format('  %-46s %4d floats = %4d bytes',
    ['MLA latent-only, arm (b) NoPE (d_c):', cLatent, 4 * cLatent]));
  WriteLn(Format('  %-46s %4d floats = %4d bytes',
    [Format('MLA latent-only + shared rope-K (d_c+%d):', [cRope]),
    cLatent + cRope, 4 * (cLatent + cRope)]));
  WriteLn(Format('Arm (b) stores %d vs MHA''s %d floats/token: %.1f%% of the MHA cache',
    [cLatent, 2 * cDModel, 100.0 * Saving]));
  WriteLn('(arm (a) demonstrates the O(1)-per-step compute path with MHA-sized');
  WriteLn(' caches; arm (b) pays an O(t) up-projection recompute per step for the');
  WriteLn(' d_c-only memory footprint).');
  WriteLn;

  WriteLn('=== Sanity checks ===');
  GateFinite := True;
  GateBeatBaseline := True;
  for i := 0 to 2 do
  begin
    if IsNan(Arm[i].FinalLoss) or IsInfinite(Arm[i].FinalLoss) then GateFinite := False;
    if not (Arm[i].FinalLoss < Uniform) then GateBeatBaseline := False;
  end;

  if GateFinite then
    WriteLn('[PASS] all arms produced a finite (no NaN/Inf) final loss.')
  else
    WriteLn('[FAIL] an arm produced NaN/Inf final loss.');

  if GateBeatBaseline then
    WriteLn(Format('[PASS] all arms beat the uniform baseline (%.4f).', [Uniform]))
  else
    WriteLn(Format('[FAIL] an arm did not beat the uniform baseline (%.4f).', [Uniform]));

  if GateDecode then
    WriteLn('[PASS] incremental decode matches the full forward (< 1e-5) on all'
      + ' three paths.')
  else
    WriteLn('[FAIL] an incremental-decode path diverged from the full forward'
      + ' (>= 1e-5).');

  if Saving < 1.0 then
    WriteLn(Format('[PASS] MLA cacheable state (%d) is smaller than MHA''s (%d).',
      [cLatent, 2 * cDModel]))
  else
    WriteLn('[WARN] choose d_c < 2*d_model for a cache saving.');

  WriteLn;
  WriteLn('Interpretation: MLA factors the K/V projection through a tiny shared latent');
  WriteLn('c_KV, so the cacheable per-token state shrinks to d_c regardless of head');
  WriteLn('count, while the model still trains competitively against full MHA at a');
  WriteLn('comparable parameter budget. The decoupled-RoPE slice restores positional');
  WriteLn('awareness without touching the latent, and the latent-only decode loop');
  WriteLn('proves d_c floats/token is genuinely sufficient decode state. Per-arm');
  WriteLn('bake-off numbers are seed-dependent.');

  if not (GateFinite and GateBeatBaseline and GateDecode) then Halt(1);
end.
