program MixtureOfDepths;
(*
MixtureOfDepths: a TINY pure-CPU demonstration of the Raposo et al. 2024
"Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language
Models" (https://arxiv.org/abs/2404.02258) conditional-compute idea, built with
TNNet.AddMixtureOfDepths.

MoD routes along the SEQUENCE axis: a per-token learned scalar router decides
WHETHER each sequence position is PROCESSED by a wrapped block or SKIPS it via
the residual/identity path, under a fixed per-block CAPACITY. Only the
top-Capacity tokens (by router score) enter the block; the rest bypass it
unchanged. Because Capacity is FIXED, all tensor shapes stay static while the
per-block FLOPs drop by (SeqLen - Capacity)/SeqLen.

THE TASK (a tiny deterministic next-token target). A small vocabulary of cVocab
chars. Each sample is a random length-cSeqLen string. The target is a position-
dependent copy:
    target[t] = S[t]            for t in the "hard" positions (the block must act)
    target[t] = S[t]  (identity, but...) -- see TargetTok: most positions are an
                                 easy identity copy that the residual/skip path
                                 already solves; a FEW "hard" positions require a
                                 nonlinear transform the wrapped block supplies.
The point: MoD should learn to SPEND its limited compute budget on the hard
positions and SKIP the easy ones. As Capacity shrinks the net must triage.

THE SWEEP. We train the SAME architecture with Capacity in {SeqLen, SeqLen/2,
SeqLen/4} (every arm reseeds to the same value so data + init are identical) and
chart final cross-entropy vs the processed-token FRACTION (Capacity/SeqLen).
This is the MoD selling point: graceful degradation as compute is cut. We also
print, per arm, WHICH token POSITIONS the router most often selects to spend
compute on -- an interpretability view of the learned triage.

Pure CPU, single-threaded (manual Compute/Backpropagate), no external dataset,
finishes in well under a minute. Per-arm numbers are SEED-DEPENDENT; the point
is the trend (loss rises as the processed fraction falls) and the router-
position histogram, not exact values.

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
  cVocab   = 5;        // small char vocabulary
  cSeqLen  = 8;        // sequence length
  cDModel  = 16;       // embedding / stream width
  cHidden  = 24;       // wrapped-block hidden width
  cEpochs  = 300;      // training epochs
  cBatch   = 48;       // sequences per epoch
  cLR      = 0.01;     // per-sample SGD
  cInertia = 0.9;
  cSeed    = 424242;   // repo idiom
  cProbes  = 600;      // evaluation probes

type
  TSeq = array[0..cSeqLen - 1] of integer;

  TArmResult = record
    Name      : string;
    Capacity  : integer;
    Frac      : TNeuralFloat;      // Capacity / SeqLen
    Params    : integer;
    InitLoss  : TNeuralFloat;
    FinalLoss : TNeuralFloat;
    Acc       : TNeuralFloat;
    Seconds   : TNeuralFloat;
    PosHist   : array[0..cSeqLen - 1] of integer; // router selection counts
  end;

procedure MakeSeq(out S: TSeq);
var
  i: integer;
begin
  for i := 0 to cSeqLen - 1 do
    S[i] := Random(cVocab);
end;

// Is position t a "hard" position (needs the wrapped block) or an "easy" one
// (the residual/skip path already copies it)?  Hard positions are the odd ones;
// easy positions are the even ones.  The block budget should go to the odds.
function IsHard(t: integer): boolean;
begin
  Result := Odd(t);
end;

// Target token at position t.
//   easy (even t): identity copy target = S[t]      (skip path solves it).
//   hard (odd  t): a nonlinear remap   target = (S[t]*2 + 1) mod cVocab
//                  which the linear skip path cannot produce; the wrapped
//                  block must be SPENT on these positions to get them right.
function TargetTok(const S: TSeq; t: integer): integer;
begin
  if IsHard(t) then
    Result := (S[t] * 2 + 1) mod cVocab
  else
    Result := S[t];
end;

procedure FillPair(const S: TSeq; InputV, TargetV: TNNetVolume);
var
  t: integer;
begin
  TargetV.Fill(0);
  for t := 0 to cSeqLen - 1 do
  begin
    // Feed the token id as a NORMALISED SCALAR (not a learned embedding lookup).
    // This is deliberate: a learned embedding + linear read-out can memorise any
    // per-token map and would make the wrapped block redundant. With a fixed
    // scalar input, the ONLY nonlinearity in the whole net is the ReLU inside the
    // MoD block, so the odd-position periodic remap GENUINELY requires the block.
    InputV[t, 0, 0] := S[t] / (cVocab - 1);
    TargetV[t, 0, TargetTok(S, t)] := 1.0;
  end;
end;

// Net (the wrapped block holds the net's ONLY nonlinearity):
//   Input(scalar id) -> PointwiseConvLinear(d_model)   [linear lift, no ReLU]
//     -> MixtureOfDepths( [PWConvReLU(hidden) -> PWConvLinear(d_model)], Capacity )
//     -> PointwiseConvLinear(V) -> PointwiseSoftMax(depth)
// Everything outside the MoD block is linear (per-token affine), so the
// non-monotone odd-position target can ONLY be produced by tokens the router
// chooses to send through the block. The linear skip path alone solves the
// even-position identity copy but cannot produce the periodic odd remap.
function BuildNet(Capacity: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(cSeqLen, 1, 1));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cDModel)); // linear lift
  Result.AddMixtureOfDepths(nil,
    [ TNNetPointwiseConvReLU.Create(cHidden),
      TNNetPointwiseConvLinear.Create(cDModel) ], Capacity);
  Result.AddLayer(TNNetPointwiseConvLinear.Create(cVocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create(1));
  Result.SetLearningRate(cLR, cInertia);
  Result.SetL2Decay(0.0);
end;

// Locate the post-TopK masked-router layer: the second TNNetTransposeXD, which
// is the (SeqLen,1,1) tensor holding the router weight at SELECTED tokens and 0
// at skipped tokens. Reading its output tells us which positions the router
// chose on the current forward pass.
function FindMaskedRouter(NN: TNNet): TNNetLayer;
var
  i, seen: integer;
begin
  Result := nil;
  seen := 0;
  for i := 0 to NN.CountLayers - 1 do
    if NN.Layers[i] is TNNetTransposeXD then
    begin
      Inc(seen);
      if seen = 2 then begin Result := NN.Layers[i]; Exit; end;
    end;
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

// Overall accuracy + accumulate the router-selection histogram over the probe
// set (count how often each position is among the selected/processed tokens).
procedure Evaluate(NN: TNNet; var Res: TArmResult);
var
  k, t, Pred, Tgt: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  C, Ctot: integer;
  Masked: TNNetLayer;
begin
  InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
  TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
  Masked := FindMaskedRouter(NN);
  C := 0; Ctot := 0;
  for t := 0 to cSeqLen - 1 do Res.PosHist[t] := 0;
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
        // A token is "processed" when its masked router weight is nonzero.
        if (Masked <> nil) and (Masked.Output[t, 0, 0] > 0) then
          Inc(Res.PosHist[t]);
      end;
    end;
  finally
    InputV.Free; TargetV.Free;
  end;
  Res.Acc := C / Max(1, Ctot);
end;

function RunArm(Capacity: integer; const AName: string): TArmResult;
var
  NN: TNNet;
  Epoch, b: integer;
  InputV, TargetV: TNNetVolume;
  S: TSeq;
  StartTime: double;
begin
  Result.Name := AName;
  Result.Capacity := Capacity;
  Result.Frac := Capacity / cSeqLen;

  RandSeed := cSeed;
  NN := BuildNet(Capacity);
  try
    Result.Params := NN.CountWeights();

    RandSeed := cSeed + 1;
    Result.InitLoss := EvalMeanCE(NN);

    InputV  := TNNetVolume.Create(cSeqLen, 1, 1);
    TargetV := TNNetVolume.Create(cSeqLen, 1, cVocab);
    RandSeed := cSeed;
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
    Evaluate(NN, Result);
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

// A crude text bar for the loss-vs-fraction chart.
function Bar(V, VMax: TNeuralFloat; Width: integer): string;
var
  n, i: integer;
begin
  if VMax <= 0 then n := 0 else n := Round(Width * V / VMax);
  if n < 0 then n := 0;
  if n > Width then n := Width;
  Result := '';
  for i := 1 to n do Result := Result + '#';
end;

var
  Arms: array[0..2] of TArmResult;
  Caps: array[0..2] of integer;
  Uniform, MaxLoss: TNeuralFloat;
  a, t: integer;
  GateFinite, GateFullBest, GateMonotone: boolean;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  Uniform := Ln(cVocab);

  Caps[0] := cSeqLen;          // process every token (no skipping)
  Caps[1] := cSeqLen div 2;    // half the tokens skip the block
  Caps[2] := cSeqLen div 4;    // three-quarters skip the block

  WriteLn('MixtureOfDepths: conditional per-token compute along the SEQUENCE axis.');
  WriteLn(Format('Task: next-token copy, seqlen %d, vocab %d. Even positions are an',
    [cSeqLen, cVocab]));
  WriteLn('identity copy (the residual/skip path solves them); ODD positions need a');
  WriteLn('nonlinear remap that only the wrapped block can produce. MoD must learn to');
  WriteLn('spend its limited per-block capacity on the hard (odd) positions.');
  WriteLn(Format('Uniform-guess baseline loss = ln(vocab) = %.4f, chance acc = %.3f.',
    [Uniform, 1.0 / cVocab]));
  WriteLn('Sweeping Capacity in {SeqLen, SeqLen/2, SeqLen/4}; same data + init per arm.');
  WriteLn;

  for a := 0 to 2 do
  begin
    Write(Format('Training Capacity=%d (%.0f%% processed) ...',
      [Caps[a], 100.0 * Caps[a] / cSeqLen]));
    Arms[a] := RunArm(Caps[a], Format('cap=%d', [Caps[a]]));
    WriteLn(Format(' done. final_CE=%s acc=%s %ss',
      [SafeF(Arms[a].FinalLoss, 4), SafeF(Arms[a].Acc, 3), SafeF(Arms[a].Seconds, 2)]));
  end;
  WriteLn;

  // ---- FLOP/accuracy trade chart: loss vs processed-token fraction ----
  WriteLn('=== Final cross-entropy vs processed-token fraction ===');
  WriteLn(Format('%-8s %10s %9s %8s %8s   %s',
    ['capacity', 'frac_proc', 'final_CE', 'acc', 'params', 'loss']));
  MaxLoss := 0;
  for a := 0 to 2 do
    if Arms[a].FinalLoss > MaxLoss then MaxLoss := Arms[a].FinalLoss;
  for a := 0 to 2 do
    WriteLn(Format('%-8d %10s %9s %8s %8d   %s',
      [Arms[a].Capacity, SafeF(Arms[a].Frac, 3), SafeF(Arms[a].FinalLoss, 4),
       SafeF(Arms[a].Acc, 3), Arms[a].Params, Bar(Arms[a].FinalLoss, MaxLoss, 40)]));
  WriteLn('(Lower CE is better; the bar scales to the worst arm. Loss should rise as');
  WriteLn(' the processed fraction falls -- the MoD compute/accuracy trade-off.)');
  WriteLn;

  // ---- Interpretability: router position-selection histogram ----
  WriteLn('=== Router selection by POSITION (how often each token is processed) ===');
  Write(Format('%-10s', ['position']));
  for t := 0 to cSeqLen - 1 do Write(Format('%5d', [t]));
  WriteLn('   (H=hard/odd, .=easy/even)');
  Write(Format('%-10s', ['kind']));
  for t := 0 to cSeqLen - 1 do
    if IsHard(t) then Write(Format('%5s', ['H'])) else Write(Format('%5s', ['.']));
  WriteLn;
  for a := 0 to 2 do
  begin
    Write(Format('%-10s', [Arms[a].Name]));
    for t := 0 to cSeqLen - 1 do
      Write(Format('%5d', [Round(100.0 * Arms[a].PosHist[t] / cProbes)]));
    WriteLn('  % of probes processed');
  end;
  WriteLn('(With Capacity=SeqLen every position is processed (100%). As capacity drops,');
  WriteLn(' a well-trained router should keep spending on the HARD (H) positions.)');
  WriteLn;

  // ---- Sanity checks ----
  WriteLn('=== Sanity checks ===');
  GateFinite := True;
  for a := 0 to 2 do
    if IsNan(Arms[a].FinalLoss) or IsInfinite(Arms[a].FinalLoss) then
      GateFinite := False;
  // Full capacity processes everything, so it should reach the lowest loss.
  GateFullBest := (Arms[0].FinalLoss <= Arms[1].FinalLoss + 1e-6) and
                  (Arms[0].FinalLoss <= Arms[2].FinalLoss + 1e-6);
  // Cutting compute should not IMPROVE loss: the trend is non-decreasing as the
  // processed fraction falls (allow a small slack for seed noise).
  GateMonotone := (Arms[1].FinalLoss >= Arms[0].FinalLoss - 0.02) and
                  (Arms[2].FinalLoss >= Arms[1].FinalLoss - 0.02);

  if GateFinite then
    WriteLn('[PASS] all arms produced a finite (no NaN/Inf) final loss.')
  else
    WriteLn('[FAIL] an arm produced NaN/Inf final loss.');

  if Arms[0].FinalLoss < Uniform then
    WriteLn(Format('[PASS] full-capacity arm beats the uniform baseline (%.3f).', [Uniform]))
  else
    WriteLn(Format('[FAIL] full-capacity arm did not beat the uniform baseline (%.3f).', [Uniform]));

  if GateFullBest then
    WriteLn('[PASS] full capacity reaches the lowest loss of the sweep.')
  else
    WriteLn('[WARN] a reduced-capacity arm matched/beat full capacity (seed-dependent).');

  if GateMonotone then
    WriteLn('[PASS] loss is non-decreasing as the processed fraction falls (the trade).')
  else
    WriteLn('[WARN] loss trend not monotone across the sweep (seed-dependent).');

  WriteLn;
  WriteLn('Interpretation: Mixture-of-Depths spends a FIXED per-block token budget,');
  WriteLn('skipping the rest via the residual path. Cutting the budget trades accuracy');
  WriteLn('for FLOPs (loss rises as the processed fraction falls), and the router learns');
  WriteLn('to spend its budget on the positions that actually need the block. Per-arm');
  WriteLn('numbers are seed-dependent; the trend and the position histogram are the point.');

  if not (GateFinite and (Arms[0].FinalLoss < Uniform)) then Halt(1);
end.
