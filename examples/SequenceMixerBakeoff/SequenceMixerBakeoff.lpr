program SequenceMixerBakeoff;
(*
SequenceMixerBakeoff: a head-to-head bake-off of FOUR token-mixing primitives on
the SAME tiny char-level next-token task. Every arm shares an identical
embedding front-end, an identical pointwise MLP read-out, the same data and the
same weight initialisation; the ONLY thing that differs is HOW information moves
between sequence positions:

  - Arm 1 (TokenShift): Embedding -> TNNetTokenShift -> MLP -> softmax.
      TNNetTokenShift is the RWKV-style, attention-FREE time-mixing primitive
      y[t,c] = mix[c]*x[t,c] + (1-mix[c])*x[t-1,c]. It is O(n), parameter-cheap
      (one Depth-long mix vector), and mixes information from EXACTLY one step
      back (t-1). It can capture first-order / lag-1 context but structurally
      CANNOT reach an arbitrary earlier position.

  - Arm 2 (CausalConv1D): Embedding -> TNNetCausalConv1D(d_model, K) -> MLP ->
      softmax. A learnable causal 1D convolution along the time axis with
      left-only zero padding, so output position t depends only on input
      positions in [t-(K-1), t]. It is O(n*K) and reaches EXACTLY K-1 steps
      back: with a kernel wide enough to cover the long-range offset it can copy
      that token, but a token older than K-1 is out of its (fixed) window.

  - Arm 3 (DiagonalSSM): Embedding -> TNNetDiagonalSSM -> MLP -> softmax. A
      diagonal-state linear recurrence (S4D/S5-style SSM-lite): per channel a
      scalar state is swept left-to-right, h_t = a*h_{t-1} + b*x_t,
      y_t = c*h_t + e*x_t. It is O(n) and recurrent, so in principle its state
      carries information arbitrarily far; whether it pins a SHARP fixed offset
      (as opposed to a smeared decaying average) is the empirical question.

  - Arm 4 (Attention): Embedding -> [QKV slab] ->
      AddMultiHeadSelfAttention(causal) -> MLP -> softmax. Self-attention is
      O(n^2) and far heavier in parameters, but with the parameter-free
      sinusoidal positions every query can route to ANY earlier key, so it can
      recover a long-range, fixed-offset dependency exactly.
      (AddMultiHeadSelfAttention consumes a Q|K|V slab of depth 3*d_model and
      out-projects back to d_model, so the arm adds a PointwiseConvLinear(3*d)
      in front of it, exactly as AddTransformerEncoderBlock does.)

THE TASK (chosen to separate the four primitives by REACH).
A small vocabulary of cVocab chars. Each sample is a random length-cSeqLen
string. The target is a fixed-offset COPY of an earlier source token, with two
regimes:

    target[t] = S[t-1]      for t <  cLag   (lag-1 copy; S[-1] := 0)
    target[t] = S[t-cLag]   for t >= cLag   (long-range fixed-offset copy)

Both regimes are deterministic copies of a SINGLE source token; the only
difference is HOW FAR BACK that source sits. The lag-1 region is reachable by
all four primitives. The lag-cLag region sits cLag steps back: a one-step shift
cannot see it, a causal conv can iff its kernel K > cLag, an SSM can iff its
recurrence learns to pin that exact offset, and attention can route to it via
the positions. The harness simply reports who reaches what.

Metrics reported per-arm:
  - overall next-token accuracy,
  - "lag-1" accuracy on the early positions t < cLag, and
  - "long-range" accuracy on positions t >= cLag (the discriminating region).

Every arm reseeds RandSeed to the same value before generating its data and
before building/initialising its net, so all arms see identical inputs and
identical embedding init; only the mixing layer differs.

Pure CPU, single-threaded (manual Compute/Backpropagate), no external dataset,
finishes in a few seconds. The per-arm numbers are SEED-DEPENDENT; the point of
this program is the comparison harness and the mechanistic contrast, not
crowning a universal winner.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cVocab   = 6;        // small char vocabulary
  cSeqLen  = 16;       // sequence length
  cLag     = 4;        // long-range source offset (beyond TokenShift's t-1)
  cDModel  = 24;       // embedding / mixing-stream width
  cHeads   = 4;        // attention heads (d_model must be divisible by heads)
  cKernel  = 5;        // causal-conv kernel K (K-1 = 4 reaches the lag-4 source)
  cDFF     = 32;       // hidden width of the pointwise MLP read-out
  cEpochs  = 350;      // training epochs
  cBatch   = 48;       // sequences per epoch
  cLR      = 0.005;    // per-sample SGD (same idiom as TokenShiftBaseline)
  cInertia = 0.9;
  cSeed    = 424242;   // repo idiom
  cProbes  = 600;      // evaluation probes

type
  TArmKind = (akTokenShift, akCausalConv, akDiagonalSSM, akAttention);

  TSeq = array[0..cSeqLen - 1] of integer;

  TArmResult = record
    Name      : string;
    Params    : integer;
    InitLoss  : TNeuralFloat;   // mean CE before training
    FinalLoss : TNeuralFloat;   // mean CE after training (all positions)
    Acc       : TNeuralFloat;   // overall next-token accuracy
    AccShort  : TNeuralFloat;   // lag-1 positions t < cLag
    AccLong   : TNeuralFloat;   // long-range positions t >= cLag
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
//   - t <  cLag : target = S[t-1]    (lag-1 copy; S[-1] := 0). Reachable by all.
//   - t >= cLag : target = S[t-cLag] (fixed long-range offset copy). The
//                 discriminating region: reach depends on the mixer.
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
//         -> [ TokenShift | CausalConv1D | DiagonalSSM
//              | (QKV slab -> MultiHeadSelfAttention causal) ]
//         -> PointwiseConvLinear(d_ff) -> ReLU
//         -> PointwiseConvLinear(V) -> PointwiseSoftMax(depth)
function BuildNet(Kind: TArmKind): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(cVocab, cDModel, 1));
  // Parameter-free positions: attention needs them to route to a fixed offset;
  // harmless for the other arms. Added to ALL arms so the front-end is identical.
  Result.AddLayer(TNNetSinusoidalPositionalEmbedding.Create());

  case Kind of
    akTokenShift: Result.AddLayer(TNNetTokenShift.Create());
    // Depthwise-equivalent causal conv that PRESERVES the d_model stream:
    // NumFeatures = cDModel, kernel K = cKernel, dense (Dilation = 1).
    akCausalConv: Result.AddLayer(TNNetCausalConv1D.Create(cDModel, cKernel));
    akDiagonalSSM: Result.AddLayer(TNNetDiagonalSSM.Create());
    akAttention:
      begin
        // Token-wise QKV slab projection d_model -> 3*d_model (1x1 conv per
        // token); AddMultiHeadSelfAttention consumes the Q|K|V slab and
        // out-projects back to d_model (see AddTransformerEncoderBlock).
        Result.AddLayer(TNNetPointwiseConvLinear.Create(3 * cDModel));
        Result.AddMultiHeadSelfAttention(cHeads, {CausalMask=}True);
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
      Sum := Sum + NN.GetLastLayer.Output.MeanCrossEntropy(TargetV);
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
        Pred := NN.GetLastLayer.Output.GetClassOnPixel(t, 0);
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

  // Reseed BEFORE the build so all arms get the same embedding init.
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

procedure PrintRow(const R: TArmResult);
begin
  WriteLn(Format('%-12s %8d %8s %9s %8s %9s %8s %7s',
    [R.Name, R.Params, SafeF(R.InitLoss, 3), SafeF(R.FinalLoss, 3),
     SafeF(R.Acc, 3), SafeF(R.AccShort, 3), SafeF(R.AccLong, 3),
     SafeF(R.Seconds, 2)]));
end;

var
  Arm: array[0..3] of TArmResult;
  Uniform, BestLong: TNeuralFloat;
  BestLongName: string;
  i: integer;
  GateFinite, GateBeatBaseline, GateLongReach: boolean;
begin
  // Mask FPU exceptions so transient early-training underflows don't abort.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  Uniform := Ln(cVocab);

  WriteLn('SequenceMixerBakeoff: four token-mixing primitives on ONE next-token task.');
  WriteLn('Arms: TokenShift (RWKV lag-1, O(n)), CausalConv1D (depthwise causal conv,');
  WriteLn('O(n*K)), DiagonalSSM (linear recurrence / SSM, O(n)), Attention (causal');
  WriteLn('SDPA, O(n^2)). Identical embedding front-end, MLP read-out, data and init;');
  WriteLn('only the sequence-mixing layer differs.');
  WriteLn(Format('Task: fixed-offset copy. target[t]=S[t-1] for t<%d, else '
    + 'S[t-%d]   (seqlen %d, vocab %d).', [cLag, cLag, cSeqLen, cVocab]));
  WriteLn(Format('Lag-1 region is reachable by all; the lag-%d region is the '
    + 'discriminator.', [cLag]));
  WriteLn(Format('Shared front-end: Embedding(%d) -> SinPos -> [MIXER] -> PWConv(%d)'
    + ' -> ReLU -> PWConv(%d) -> softmax.', [cDModel, cDFF, cVocab]));
  WriteLn(Format('CausalConv kernel K=%d (reaches K-1=%d back). Uniform-guess '
    + 'baseline = ln(vocab) = %.4f, chance acc = %.3f.',
    [cKernel, cKernel - 1, Uniform, 1.0 / cVocab]));
  WriteLn;

  Write('Training Arm 1 (TNNetTokenShift) ...');
  Arm[0] := RunArm(akTokenShift, 'TokenShift');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[0].FinalLoss, 4), SafeF(Arm[0].Seconds, 2)]));

  Write('Training Arm 2 (TNNetCausalConv1D) ...');
  Arm[1] := RunArm(akCausalConv, 'CausalConv1D');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[1].FinalLoss, 4), SafeF(Arm[1].Seconds, 2)]));

  Write('Training Arm 3 (TNNetDiagonalSSM) ...');
  Arm[2] := RunArm(akDiagonalSSM, 'DiagonalSSM');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[2].FinalLoss, 4), SafeF(Arm[2].Seconds, 2)]));

  Write('Training Arm 4 (AddMultiHeadSelfAttention) ...');
  Arm[3] := RunArm(akAttention, 'Attention');
  WriteLn(Format(' done.  final_CE=%s  %ss', [SafeF(Arm[3].FinalLoss, 4), SafeF(Arm[3].Seconds, 2)]));
  WriteLn;

  WriteLn('=== Comparison (lower CE is better; accuracy is argmax over vocab) ===');
  WriteLn(Format('%-12s %8s %8s %9s %8s %9s %8s %7s',
    ['arm', 'params', 'init_CE', 'final_CE', 'acc', 'acc_lag1', 'acc_long', 'sec']));
  for i := 0 to 3 do PrintRow(Arm[i]);
  WriteLn;
  WriteLn(Format('acc_lag1 = early positions t<%d (source is S[t-1], reachable by all).',
    [cLag]));
  WriteLn(Format('acc_long = positions t>=%d (source is S[t-%d], the discriminating region).',
    [cLag, cLag]));
  WriteLn;

  // ---- Sanity checks ----
  WriteLn('=== Sanity checks ===');
  GateFinite := True;
  GateBeatBaseline := True;
  for i := 0 to 3 do
  begin
    if IsNan(Arm[i].FinalLoss) or IsInfinite(Arm[i].FinalLoss) then GateFinite := False;
    if not (Arm[i].FinalLoss < Uniform) then GateBeatBaseline := False;
  end;

  // Which arm reaches the long-range source best, and does TokenShift (the
  // structurally short-sighted arm) clearly fall behind it?
  BestLong := Arm[0].AccLong; BestLongName := Arm[0].Name;
  for i := 1 to 3 do
    if Arm[i].AccLong > BestLong then
    begin BestLong := Arm[i].AccLong; BestLongName := Arm[i].Name; end;
  GateLongReach := (BestLong > Arm[0].AccLong + 0.05);

  if GateFinite then
    WriteLn('[PASS] all arms produced a finite (no NaN/Inf) final loss.')
  else
    WriteLn('[FAIL] at least one arm produced NaN/Inf final loss.');

  if GateBeatBaseline then
    WriteLn(Format('[PASS] all arms beat the uniform baseline (%.3f).', [Uniform]))
  else
    WriteLn(Format('[FAIL] an arm did not beat the uniform baseline (%.3f).', [Uniform]));

  if GateLongReach then
    WriteLn(Format('[PASS] a longer-reach mixer beats TokenShift on acc_long: %s %.3f > %.3f.',
      [BestLongName, BestLong, Arm[0].AccLong]))
  else
    WriteLn(Format('[WARN] no arm clearly beat TokenShift on acc_long (best %s %.3f);'
      + ' seed-dependent.', [BestLongName, BestLong]));

  WriteLn;
  WriteLn('Interpretation: TokenShift only reaches t-1, so it solves the lag-1 region');
  WriteLn('but is near chance on the long-range copy. CausalConv1D reaches K-1 steps');
  WriteLn('back and so can cover the lag-', cLag, ' source with a wide-enough kernel.');
  WriteLn('DiagonalSSM carries a recurrent state (unbounded reach in principle) and');
  WriteLn('Attention can route to any earlier position. Per-arm numbers are');
  WriteLn('seed-dependent; the reach-vs-cost trade-off is the structural point.');

  // Finite + beats-baseline are mandatory; the long-range reach is informative.
  if not (GateFinite and GateBeatBaseline) then Halt(1);
end.
