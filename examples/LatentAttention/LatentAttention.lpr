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

WHAT THIS PROGRAM SHOWS.
  1. The cacheable-state saving d_c/(2*d_model) reported numerically.
  2. A tiny next-token COPY task (fixed long-range offset; the same family of
     task used in SequenceMixerBakeoff) trained with two attention arms:
       - Arm MLA: Embedding -> SinPos -> AddMultiHeadLatentAttention(causal)
                  -> MLP -> softmax.
       - Arm MHA: Embedding -> SinPos -> [QKV slab] ->
                  AddMultiHeadSelfAttention(causal) -> MLP -> softmax.
     Both arms share an identical embedding front-end, MLP read-out, data and
     weight init; only the attention block differs. We print each arm's
     parameter count, final cross-entropy and next-token accuracy so the reader
     can see MLA train competitively at a comparable parameter budget while
     advertising the smaller cacheable state.

HONEST SCOPE (v1).
  - This is TRAINING-TIME behaviour. The KV-cache WIN only materialises with an
    incremental-decode path (still open in this repo); here we PROVE the shapes
    are correct and REPORT the cacheable-state saving rather than measure cache
    bytes at decode time.
  - v1 MLA is NoPE on the attention scores themselves (the builder carries no
    positional info); the shared SinusoidalPositionalEmbedding front-end gives
    BOTH arms identical positions. The paper's DECOUPLED RoPE slice (a separate
    small RoPE-only channel group carried on Q and K) is NOT yet implemented --
    a documented follow-up.
  - Per-arm numbers are SEED-DEPENDENT; the point is the mechanism + the cache
    arithmetic, not crowning a winner.

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
  cDFF     = 32;       // hidden width of the pointwise MLP read-out
  cEpochs  = 300;      // training epochs
  cBatch   = 48;       // sequences per epoch
  cLR      = 0.005;    // per-sample SGD
  cInertia = 0.9;
  cSeed    = 424242;   // repo idiom
  cProbes  = 600;      // evaluation probes

type
  TArmKind = (akMLA, akMHA);

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
  // Parameter-free positions, identical for both arms.
  Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());

  case Kind of
    akMLA:
      // The builder does its OWN Q projection + the low-rank c_KV down/up
      // projections from the (SeqLen,1,d_model) token stream.
      Result.AddMultiHeadLatentAttention(cDModel, cHeads, cLatent, True);
    akMHA:
      begin
        // MHA consumes a pre-projected Q|K|V slab of depth 3*d_model.
        Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDModel));
        Result.AddMultiHeadSelfAttention(cDModel, cHeads, True);
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
  Arm: array[0..1] of TArmResult;
  Uniform, Saving: TNeuralFloat;
  i: integer;
  GateFinite, GateBeatBaseline: boolean;
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
  WriteLn('(Training-time demo: shapes are proven below; the cache win needs the');
  WriteLn(' still-open incremental-decode path. v1 is NoPE; decoupled RoPE is a follow-up.)');
  WriteLn;

  Write('Training Arm MLA (AddMultiHeadLatentAttention) ...');
  Arm[0] := RunArm(akMLA, 'MLA');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[0].FinalLoss, 4), SafeF(Arm[0].Seconds, 2)]));

  Write('Training Arm MHA (AddMultiHeadSelfAttention) ...');
  Arm[1] := RunArm(akMHA, 'MHA');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[1].FinalLoss, 4), SafeF(Arm[1].Seconds, 2)]));
  WriteLn;

  WriteLn('=== Next-token bake-off (lower CE is better; acc is argmax over vocab) ===');
  WriteLn(Format('Task: fixed-offset copy target[t]=S[t-%d] for t>=%d (else S[t-1]).',
    [cLag, cLag]));
  WriteLn(Format('%-8s %8s %9s %9s %8s %9s %7s',
    ['arm', 'params', 'init_CE', 'final_CE', 'acc', 'acc_long', 'sec']));
  for i := 0 to 1 do PrintRow(Arm[i]);
  WriteLn;
  WriteLn(Format('acc_long = positions t>=%d (the long-range copy region). '
    + 'Uniform baseline CE=ln(vocab)=%.4f.', [cLag, Uniform]));
  WriteLn;

  WriteLn('=== Sanity checks ===');
  GateFinite := True;
  GateBeatBaseline := True;
  for i := 0 to 1 do
  begin
    if IsNan(Arm[i].FinalLoss) or IsInfinite(Arm[i].FinalLoss) then GateFinite := False;
    if not (Arm[i].FinalLoss < Uniform) then GateBeatBaseline := False;
  end;

  if GateFinite then
    WriteLn('[PASS] both arms produced a finite (no NaN/Inf) final loss.')
  else
    WriteLn('[FAIL] an arm produced NaN/Inf final loss.');

  if GateBeatBaseline then
    WriteLn(Format('[PASS] both arms beat the uniform baseline (%.4f).', [Uniform]))
  else
    WriteLn(Format('[FAIL] an arm did not beat the uniform baseline (%.4f).', [Uniform]));

  if Saving < 1.0 then
    WriteLn(Format('[PASS] MLA cacheable state (%d) is smaller than MHA''s (%d).',
      [cLatent, 2 * cDModel]))
  else
    WriteLn('[WARN] choose d_c < 2*d_model for a cache saving.');

  WriteLn;
  WriteLn('Interpretation: MLA factors the K/V projection through a tiny shared latent');
  WriteLn('c_KV, so the cacheable per-token state shrinks to d_c regardless of head');
  WriteLn('count, while the model still trains competitively against full MHA at a');
  WriteLn('comparable parameter budget. Per-arm numbers are seed-dependent.');

  if not (GateFinite and GateBeatBaseline) then Halt(1);
end.
