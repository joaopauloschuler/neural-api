program TokenShiftBaseline;
(*
TokenShiftBaseline: a head-to-head bake-off of two TOKEN-MIXING primitives on
the SAME tiny char-level sequence task. The two arms share an identical
embedding front-end, an identical pointwise MLP read-out, the same data and the
same weight initialisation; the ONLY thing that differs is HOW information moves
between sequence positions:

  - Arm 1 (TokenShift): Embedding -> TNNetTokenShift -> MLP -> softmax.
      TNNetTokenShift is the RWKV-style, attention-FREE time-mixing primitive
      y[t,c] = mix[c]*x[t,c] + (1-mix[c])*x[t-1,c]. It is O(n), parameter-cheap
      (one Depth-long mix vector), and mixes information from EXACTLY one step
      back (t-1). It can therefore capture first-order / lag-1 context but
      structurally CANNOT reach an arbitrary earlier position.

  - Arm 2 (Attention): Embedding -> [QKV slab] ->
      AddMultiHeadSelfAttention(causal) -> MLP -> softmax. Self-attention is
      O(n^2) and far heavier in parameters, but with the parameter-free
      sinusoidal positions every query can route to ANY earlier key, so it can
      recover a long-range, fixed-offset dependency that TokenShift cannot.
      (AddMultiHeadSelfAttention consumes a Q|K|V slab of depth 3*d_model and
      out-projects back to d_model, so the arm adds a PointwiseConvLinear(3*d)
      in front of it, exactly as AddTransformerEncoderBlock does.)

THE TASK (chosen to make the contrast informative AND learnable by BOTH arms).
A small vocabulary of cVocab chars. Each sample is a random length-cSeqLen
string. The target is a fixed-offset COPY of an earlier source token, with two
regimes:

    target[t] = S[t-1]      for t <  cLag   (lag-1 copy; S[-1] := 0)
    target[t] = S[t-cLag]   for t >= cLag   (long-range fixed-offset copy)

Both regimes are deterministic copies of a SINGLE source token; the only
difference is HOW FAR BACK that source sits. The lag-1 region is exactly what
TokenShift's t-1 mixing exposes, so TokenShift solves it and clears the uniform
baseline ln(vocab). The lag-cLag region sits too far back for a one-step shift
to reach, but causal self-attention can route a query to a fixed earlier offset
(via the parameter-free sinusoidal positions), so it copies that token too. The
EXPECTED outcome: both arms beat the uniform baseline, but attention reaches
markedly higher accuracy because it can ALSO solve the long-range region where
TokenShift is stuck near chance.

Metrics reported per-arm:
  - overall next-token accuracy,
  - "lag-1" accuracy on the early positions t < cLag (TokenShift's home turf), and
  - "long-range" accuracy on positions t >= cLag (attention's home turf).

Every arm reseeds RandSeed to the same value before generating its data and
before building/initialising its net, so both arms see identical inputs and
identical embedding init; only the mixing layer differs.

Pure CPU, single-threaded (manual Compute/Backpropagate), no external dataset,
finishes in well under a minute. The per-arm numbers are SEED-DEPENDENT; the
point of this program is the comparison harness and the mechanistic contrast,
not crowning a universal winner.

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
  cLag     = 4;        // long-range source offset (well beyond TokenShift's t-1)
  cDModel  = 24;       // embedding / mixing-stream width
  cHeads   = 4;        // attention heads (d_model must be divisible by heads)
  cDFF     = 32;       // hidden width of the pointwise MLP read-out
  cEpochs  = 350;      // training epochs
  cBatch   = 48;       // sequences per epoch
  cLR      = 0.005;    // per-sample SGD (same idiom as InductionHeads)
  cInertia = 0.9;
  cSeed    = 424242;   // repo idiom
  cProbes  = 600;      // evaluation probes

type
  TArmKind = (akTokenShift, akAttention);

  TSeq = array[0..cSeqLen - 1] of integer;

  TArmResult = record
    Name      : string;
    Params    : integer;
    InitLoss  : TNeuralFloat;   // mean CE before training
    FinalLoss : TNeuralFloat;   // mean CE after training (all positions)
    Acc       : TNeuralFloat;   // overall next-token accuracy
    AccShort  : TNeuralFloat;   // lag-1 positions t < cLag (TokenShift turf)
    AccLong   : TNeuralFloat;   // long-range positions t >= cLag (attention turf)
    Seconds   : TNeuralFloat;
  end;

procedure MakeSeq(out S: TSeq);
var
  i: integer;
begin
  for i := 0 to cSeqLen - 1 do
    S[i] := Random(cVocab);
end;

// Target at position t. A two-regime task:
//   - t <  cLag : target = S[t-1]   (lag-1 copy; S[-1] := 0). TokenShift turf.
//   - t >= cLag : target = S[t-cLag] (fixed long-range offset copy). Only
//                 attention (positional routing) can reach this source token.
// Both regimes are deterministic copies of a single source token, so each is
// individually easy for the arm that can SEE the source -- the only question is
// reachability, which is exactly the TokenShift-vs-attention distinction.
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
var
  t: integer;
begin
  TargetV.Fill(0);
  for t := 0 to cSeqLen - 1 do
  begin
    InputV.FData[t] := S[t];
    TargetV[t, 0, TargetTok(S, t)] := 1.0;
  end;
end;

// Shared front-end + read-out; only the token-mixing layer in the middle moves.
//   Input -> Embedding(V, d_model) -> SinusoidalPositionalEmbedding
//         -> [ TokenShift | (QKV slab -> MultiHeadSelfAttention causal) ]
//         -> PointwiseConvLinear(d_ff) -> ReLU
//         -> PointwiseConvLinear(V) -> PointwiseSoftMax(depth)
function BuildNet(Kind: TArmKind): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
  // Parameter-free positions: attention needs them to route to a fixed offset;
  // harmless for TokenShift. Added to BOTH arms so the front-end is identical.
  Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());

  case Kind of
    akTokenShift: Result.AddLayer(TNNetTokenShift.Create());
    akAttention:
      begin
        // Token-wise QKV slab projection d_model -> 3*d_model (1x1 conv per
        // token); AddMultiHeadSelfAttention consumes the Q|K|V slab and
        // out-projects back to d_model (see AddTransformerEncoderBlock).
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

// Mean CE over many fresh probes (used for init/final overall loss).
function EvalMeanCE(NN: TNNet): TNeuralFloat;
var
  k: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  Sum: TNeuralFloat;
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

// Overall + short-range (t < cLag) + long-range (t >= cLag) accuracy.
procedure Evaluate(NN: TNNet; out Acc, AccShort, AccLong: TNeuralFloat);
var
  k, t, Pred, Tgt: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  C, Ctot, Cs, Ns, Cl, Nl: integer;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  C := 0; Ctot := 0; Cs := 0; Ns := 0; Cl := 0; Nl := 0;
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
        if t < cLag then
        begin
          if Pred = Tgt then Inc(Cs);
          Inc(Ns);
        end
        else
        begin
          if Pred = Tgt then Inc(Cl);
          Inc(Nl);
        end;
      end;
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  Acc      := C  / Max(1, Ctot);
  AccShort := Cs / Max(1, Ns);
  AccLong  := Cl / Max(1, Nl);
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

  // Reseed BEFORE the build so both arms get the same embedding init.
  RandSeed := cSeed;
  NN := BuildNet(Kind);
  try
    Result.Params := NN.CountWeights();

    RandSeed := cSeed + 1;        // shared probe stream for the init loss
    Result.InitLoss := EvalMeanCE(NN);

    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    RandSeed := cSeed;            // identical training stream per arm
    StartTime := Now();
    try
      for Epoch := 1 to cEpochs do
        for b := 1 to cBatch do
        begin
          MakeSeq(S);
          FillPair(S, InputV, TargetV);
          NN.Compute(InputV);
          NN.Backpropagate(TargetV);   // per-sample SGD (single-threaded)
        end;
    finally
      InputV.Free; TargetV.Free;
    end;
    Result.Seconds := (Now() - StartTime) * 86400.0;

    RandSeed := cSeed + 1;        // same eval stream as the init loss
    Result.FinalLoss := EvalMeanCE(NN);
    RandSeed := cSeed + 2;
    Evaluate(NN, Result.Acc, Result.AccShort, Result.AccLong);
  finally
    NN.Free;
  end;
end;

function SafeF(V: TNeuralFloat; Dec: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then
    Result := 'NaN'
  else
    Result := FloatToStrF(V, ffFixed, 8, Dec);
end;

var
  RS, RA: TArmResult;
  Uniform: TNeuralFloat;
  GateFinite, GateBeatBaseline, GateAttnLong: boolean;
begin
  // Mask FPU exceptions so transient early-training underflows don't abort.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  Uniform := Ln(cVocab);

  WriteLn('TokenShiftBaseline: attention-free TokenShift vs causal self-attention.');
  WriteLn(Format('Task: fixed-offset copy. target[t]=S[t-1] for t<%d, else '
    + 'S[t-%d]   (seqlen %d, vocab %d).', [cLag, cLag, cSeqLen, cVocab]));
  WriteLn('A lag-1 region (TokenShift can reach the source) and a lag-', cLag,
    ' region');
  WriteLn('(only attention can reach the source). Both arms can beat the uniform');
  WriteLn('baseline; attention should win by also solving the long-range region.');
  WriteLn(Format('Shared front-end: Embedding(%d) -> SinPos -> [MIXER] -> PWConv(%d)'
    + ' -> ReLU -> PWConv(%d) -> softmax.', [cDModel, cDFF, cVocab]));
  WriteLn(Format('Uniform-guess baseline loss = ln(vocab) = %.4f, chance acc = %.3f.',
    [Uniform, 1.0 / cVocab]));
  WriteLn('Same data, same init, same read-out; only the token-mixing layer changes.');
  WriteLn;

  Write('Training Arm 1 (TNNetTokenShift) ...');
  RS := RunArm(akTokenShift, 'TokenShift');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(RS.FinalLoss, 4), SafeF(RS.Seconds, 2)]));

  Write('Training Arm 2 (AddMultiHeadSelfAttention) ...');
  RA := RunArm(akAttention, 'Attention');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(RA.FinalLoss, 4), SafeF(RA.Seconds, 2)]));
  WriteLn;

  WriteLn('=== Comparison (lower CE is better; accuracy is argmax over vocab) ===');
  WriteLn(Format('%-12s %8s %8s %9s %8s %9s %8s %7s',
    ['arm', 'params', 'init_CE', 'final_CE', 'acc', 'acc_lag1', 'acc_long', 'sec']));
  WriteLn(Format('%-12s %8d %8s %9s %8s %9s %8s %7s',
    [RS.Name, RS.Params, SafeF(RS.InitLoss, 3), SafeF(RS.FinalLoss, 3),
     SafeF(RS.Acc, 3), SafeF(RS.AccShort, 3), SafeF(RS.AccLong, 3), SafeF(RS.Seconds, 2)]));
  WriteLn(Format('%-12s %8d %8s %9s %8s %9s %8s %7s',
    [RA.Name, RA.Params, SafeF(RA.InitLoss, 3), SafeF(RA.FinalLoss, 3),
     SafeF(RA.Acc, 3), SafeF(RA.AccShort, 3), SafeF(RA.AccLong, 3), SafeF(RA.Seconds, 2)]));
  WriteLn;
  WriteLn(Format('acc_lag1 = early positions t<%d (source is S[t-1], TokenShift turf).',
    [cLag]));
  WriteLn(Format('acc_long = positions t>=%d (source is S[t-%d], attention turf).',
    [cLag, cLag]));
  WriteLn;

  // ---- Sanity checks ----
  WriteLn('=== Sanity checks ===');
  GateFinite := not (IsNan(RS.FinalLoss) or IsInfinite(RS.FinalLoss) or
                     IsNan(RA.FinalLoss) or IsInfinite(RA.FinalLoss));
  GateBeatBaseline := (RS.FinalLoss < Uniform) and (RA.FinalLoss < Uniform);
  // Attention can reach the lag-cLag source token that a one-step shift cannot,
  // so its long-range accuracy should clearly lead.
  GateAttnLong := (RA.AccLong > RS.AccLong + 0.05);

  if GateFinite then
    WriteLn('[PASS] both arms produced a finite (no NaN/Inf) final loss.')
  else
    WriteLn('[FAIL] at least one arm produced NaN/Inf final loss.');

  if GateBeatBaseline then
    WriteLn(Format('[PASS] both arms beat the uniform baseline (%.3f).', [Uniform]))
  else
    WriteLn(Format('[FAIL] an arm did not beat the uniform baseline (%.3f).', [Uniform]));

  if GateAttnLong then
    WriteLn(Format('[PASS] attention wins the long-range signal: acc_long %.3f > %.3f.',
      [RA.AccLong, RS.AccLong]))
  else
    WriteLn(Format('[WARN] attention did not clearly lead on acc_long (%.3f vs %.3f);'
      + ' seed-dependent.', [RA.AccLong, RS.AccLong]));

  WriteLn;
  WriteLn('Interpretation: TokenShift is a cheap O(n) attention-FREE mixer that only');
  WriteLn('reaches one step back (t-1), so it learns the lag-1 half of the target but');
  WriteLn('is blind to the long-range source token; causal self-attention captures');
  WriteLn('BOTH, at far higher parameter/compute cost. Both clear the uniform');
  WriteLn('baseline; the gap lives in acc_long. Per-arm numbers are seed-dependent.');

  // Finite + beats-baseline are mandatory; the long-range lead is informative.
  if not (GateFinite and GateBeatBaseline) then Halt(1);
end.
