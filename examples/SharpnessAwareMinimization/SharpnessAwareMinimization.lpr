program SharpnessAwareMinimization;
(*
SharpnessAwareMinimization: a self-contained demo of Sharpness-Aware
Minimization (Foret et al. 2021, https://arxiv.org/abs/2010.01412) on a tiny
noisy-label 2D classification toy, with a hand-rolled mini-batch training loop
(no TNeuralFit surgery).

SAM, in one step (the two-pass "sharpness-aware" update):
  Given current weights w,
   (1) forward+backward on a batch to get the gradient g = dL/dw;
   (2) climb to the worst-case neighbour  w_adv = w + rho * g/||g||
       (the ascent step: snapshot + perturb the whole net);
   (3) a SECOND forward+backward AT w_adv to get the perturbed gradient g_adv;
   (4) restore w and apply g_adv with the normal (plain-SGD) optimizer.
The step minimises the loss of the WORST point in a rho-ball around w, which
biases training toward FLAT minima -- minima that generalise better and are
more robust to label noise.

This library accumulates, per neuron, FDelta = -learningRate * grad (the
descent step) in Neurons[].Delta. We therefore read the ascent direction
straight out of Neurons[].Delta: g/||g|| == -Delta/||Delta||, so the perturb
applied in step (2) is  w_i += -rho * Delta_i / ||Delta||. The global L2 norm
||g|| (equivalently ||Delta||) is taken over ALL weight-gradient tensors.
The whole-net snapshot/restore for step (4) reuses the deep-copy of each
neuron's Weights volume (the same restore trick used by LossLandscapeProbe /
LayerSensitivityReport).

We train the SAME tiny MLP twice at matched LR / epochs / seed: plain SGD vs
SAM. Then, the headline FLATNESS contrast: TNNet.LossLandscapeProbe is called
on BOTH trained nets, and its sharpness scalar + loss-doubling radius are
parsed and printed -- SAM should land FLATTER / WIDER. Finally we sweep
rho in {0.0, 0.01, 0.05, 0.1, 0.2} and chart (ASCII) sharpness-vs-rho and
val-accuracy-vs-rho.

TWO BUILT-IN INVARIANTS (acceptance tests):
  (1) rho = 0 reproduces plain SGD BIT-FOR-BIT. The SAM routine is structured
      so that at rho=0 the perturb is zero (w_adv == w), the second gradient
      equals the first, the restore is a no-op, and the applied update equals
      a single plain-SGD step. The net is deterministic (no Dropout), data is
      pre-shuffled per epoch (forward/backward consume no RNG), and updates are
      plain SGD (no momentum/Adam state for the extra pass to desync). We
      assert the rho=0 SAM-arm final weights equal the plain-SGD-arm final
      weights bit-for-bit (max abs weight diff == 0).
  (2) Higher rho trades a little train fit for a flatter minimum: the sharpness
      scalar should generally DECREASE as rho grows; we document the observed
      trend.

Pure CPU, no external data, well under ~4.5 minutes.

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

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeed        = 42;
  cNumClasses  = 3;
  cTrainPerCls = 80;                       // 240 training samples
  cValPerCls   = 60;                       // 180 clean validation samples
  cNumTrain    = cNumClasses * cTrainPerCls;
  cNumVal      = cNumClasses * cValPerCls;
  cNoiseFrac   = 0.12;                      // 12% of TRAIN labels flipped
  cBlobRadius  = 1.7;                       // class-centre radius (overlapping)
  cBlobStd     = 0.55;                      // cluster spread (classes overlap)
  cHiddenW     = 32;
  cBatchSize   = 16;
  cEpochs      = 90;
  cLearnRate   = 0.02;
  // LossLandscapeProbe parameters (cross-entropy on softmax output).
  cProbeK      = 21;
  cProbeR      = 1.0;
  cProbeSeed   = 7;

type
  TFloatArr = array of TNeuralFloat;

  // Snapshot of all weight tensors of all trainable neurons (deep copies).
  TWeightSnapshot = record
    LayerIdx: array of integer;            // trainable layer indices
    W: array of array of TNNetVolume;      // [trainable][neuron] weights copy
  end;

// ---------------------------------------------------------------------------
// Synthetic 3-blob 2D dataset. Blob centres on a circle; each class is a
// Gaussian cloud. Training labels are flipped on cNoiseFrac of the samples;
// the validation set is clean.
// ---------------------------------------------------------------------------
procedure MakeBlobs(out Samples: TNNetVolumePairList; PerClass: integer;
  FlipFrac: TNeuralFloat);
var
  Cls, I, NewCls, NumFlip, F: integer;
  cx, cy, ang: TNeuralFloat;
  X, Y: TNNetVolume;
  Order: array of integer;
  J, Tmp: integer;
begin
  Samples := TNNetVolumePairList.Create();
  for Cls := 0 to cNumClasses - 1 do
  begin
    ang := 2 * Pi * Cls / cNumClasses;
    cx := cBlobRadius * Cos(ang);
    cy := cBlobRadius * Sin(ang);
    for I := 0 to PerClass - 1 do
    begin
      X := TNNetVolume.Create(2, 1, 1);
      X.FData[0] := cx + cBlobStd * RandG(0, 1);
      X.FData[1] := cy + cBlobStd * RandG(0, 1);
      Y := TNNetVolume.Create(cNumClasses, 1, 1);
      Y.Fill(0);
      Y.FData[Cls] := 1.0;
      Samples.Add(TNNetVolumePair.Create(X, Y));
    end;
  end;

  // Deliberately flip a fraction of labels (label noise).
  if FlipFrac > 0 then
  begin
    NumFlip := Round(FlipFrac * Samples.Count);
    SetLength(Order, Samples.Count);
    for I := 0 to High(Order) do Order[I] := I;
    for I := High(Order) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    for F := 0 to NumFlip - 1 do
    begin
      Y := Samples[Order[F]].O;
      // current class
      Cls := Y.GetClass();
      NewCls := (Cls + 1 + Random(cNumClasses - 1)) mod cNumClasses;
      Y.Fill(0);
      Y.FData[NewCls] := 1.0;
    end;
  end;
end;

// ---------------------------------------------------------------------------
// Deterministic 3-2-class MLP classifier with softmax head (no Dropout / no
// stochastic layers, so forward/backward consume no RNG).
// ---------------------------------------------------------------------------
procedure BuildClassifier(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(2, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenW));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenW));
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLearnRate, {Momentum=}0.0);  // plain SGD, no momentum
  NN.SetL2Decay(0.0);
  // CRITICAL: batch mode makes Backpropagate ACCUMULATE the gradient into
  // Neurons[].Delta (FDelta = -lr*grad) instead of applying it per-sample.
  // SAM reads that accumulated tensor for ||g|| and the ascent perturb, and
  // UpdateWeights then applies the batch delta as a single plain-SGD step.
  NN.SetBatchUpdate(True);
end;

// Indices of trainable layers (neurons own a non-empty weight tensor).
procedure CollectTrainable(NN: TNNet; out Idx: array of integer;
  out Cnt: integer);
var
  L: integer;
begin
  Cnt := 0;
  for L := 0 to NN.GetLastLayerIdx() do
  begin
    if NN.Layers[L].Neurons.Count = 0 then Continue;
    if NN.Layers[L].Neurons[0].Weights = nil then Continue;
    if NN.Layers[L].Neurons[0].Weights.Size = 0 then Continue;
    Idx[Cnt] := L;
    Inc(Cnt);
  end;
end;

// ---------------------------------------------------------------------------
// Whole-net weight snapshot / restore (deep copies of every trainable
// neuron's weight tensor). Biases are NOT perturbed by SAM, so they need no
// snapshot here; the normal optimizer step updates them as usual.
// ---------------------------------------------------------------------------
procedure SnapshotWeights(NN: TNNet; var Snap: TWeightSnapshot);
var
  T, Tc, NIdx: integer;
  Tmp: array of integer;
begin
  SetLength(Tmp, NN.GetLastLayerIdx() + 1);
  CollectTrainable(NN, Tmp, Tc);
  SetLength(Snap.LayerIdx, Tc);
  SetLength(Snap.W, Tc);
  for T := 0 to Tc - 1 do
  begin
    Snap.LayerIdx[T] := Tmp[T];
    SetLength(Snap.W[T], NN.Layers[Tmp[T]].Neurons.Count);
    for NIdx := 0 to NN.Layers[Tmp[T]].Neurons.Count - 1 do
    begin
      Snap.W[T][NIdx] := TNNetVolume.Create();
      Snap.W[T][NIdx].Copy(NN.Layers[Tmp[T]].Neurons[NIdx].Weights);
    end;
  end;
end;

procedure RestoreWeights(NN: TNNet; const Snap: TWeightSnapshot);
var
  T, NIdx: integer;
begin
  for T := 0 to High(Snap.LayerIdx) do
  begin
    for NIdx := 0 to High(Snap.W[T]) do
      NN.Layers[Snap.LayerIdx[T]].Neurons[NIdx].Weights.Copy(Snap.W[T][NIdx]);
    // The TNNetFullConnect{Linear,ReLU}/SoftMax layers used here read neuron
    // weights directly in ComputeCPU (no concatenated-weight cache to refresh).
  end;
end;

procedure FreeSnapshot(var Snap: TWeightSnapshot);
var
  T, NIdx: integer;
begin
  for T := 0 to High(Snap.W) do
    for NIdx := 0 to High(Snap.W[T]) do
      Snap.W[T][NIdx].Free;
  SetLength(Snap.W, 0);
  SetLength(Snap.LayerIdx, 0);
end;

// Global L2 norm of the per-parameter weight gradient (read from the descent
// step in Neurons[].Delta: ||g|| == ||Delta|| / learningRate, but the constant
// cancels in g/||g||, so we just use ||Delta||).
function GlobalDeltaNorm(NN: TNNet; const Idx: array of integer;
  Cnt: integer): TNeuralFloat;
var
  T, NIdx, W: integer;
  Sum: TNeuralFloat;
  D: TNNetVolume;
begin
  Sum := 0;
  for T := 0 to Cnt - 1 do
    for NIdx := 0 to NN.Layers[Idx[T]].Neurons.Count - 1 do
    begin
      D := NN.Layers[Idx[T]].Neurons[NIdx].Delta;
      for W := 0 to D.Size - 1 do
        Sum := Sum + D.FData[W] * D.FData[W];
    end;
  Result := Sqrt(Sum);
end;

// Ascent perturb: w_i += rho * g_i/||g|| == w_i + rho * (-Delta_i)/||Delta||.
procedure PerturbToAdversarial(NN: TNNet; const Idx: array of integer;
  Cnt: integer; Rho, DeltaNorm: TNeuralFloat);
var
  T, NIdx, W: integer;
  Scale: TNeuralFloat;
  Wv, Dv: TNNetVolume;
begin
  if (Rho = 0) or (DeltaNorm = 0) then Exit;  // rho=0 -> exact no-op
  Scale := -Rho / DeltaNorm;                  // climb toward higher loss
  for T := 0 to Cnt - 1 do
  begin
    for NIdx := 0 to NN.Layers[Idx[T]].Neurons.Count - 1 do
    begin
      Wv := NN.Layers[Idx[T]].Neurons[NIdx].Weights;
      Dv := NN.Layers[Idx[T]].Neurons[NIdx].Delta;
      for W := 0 to Wv.Size - 1 do
        Wv.FData[W] := Wv.FData[W] + Scale * Dv.FData[W];
    end;
  end;
end;

// ---------------------------------------------------------------------------
// One SAM mini-batch step over the index window [Lo, Hi). At Rho=0 this is
// IDENTICAL to a plain-SGD step (single pass): the perturb is a no-op, the
// second pass reproduces the first gradient, and the restore is a no-op.
// ---------------------------------------------------------------------------
procedure SAMBatchStep(NN: TNNet; Pairs: TNNetVolumePairList;
  const Order: array of integer; Lo, Hi: integer; Rho: TNeuralFloat;
  const TrainIdx: array of integer; TrainCnt: integer);
var
  I: integer;
  DeltaNorm: TNeuralFloat;
  Snap: TWeightSnapshot;
begin
  if Rho = 0 then
  begin
    // Plain-SGD path: single pass, plain update. (Bit-for-bit baseline.)
    NN.ClearDeltas();
    for I := Lo to Hi - 1 do
    begin
      NN.Compute(Pairs[Order[I]].I);
      NN.Backpropagate(Pairs[Order[I]].O);
    end;
    NN.UpdateWeights();  // momentum=0 -> per-neuron plain SGD (no inertia)
    Exit;
  end;

  // ---- SAM two-pass step ----
  // Pass 1: gradient g at w.
  NN.ClearDeltas();
  for I := Lo to Hi - 1 do
  begin
    NN.Compute(Pairs[Order[I]].I);
    NN.Backpropagate(Pairs[Order[I]].O);
  end;
  DeltaNorm := GlobalDeltaNorm(NN, TrainIdx, TrainCnt);

  // Snapshot w, then climb to w_adv = w + rho * g/||g||.
  SnapshotWeights(NN, Snap);
  PerturbToAdversarial(NN, TrainIdx, TrainCnt, Rho, DeltaNorm);

  // Pass 2: gradient g_adv at w_adv (same batch).
  NN.ClearDeltas();
  for I := Lo to Hi - 1 do
  begin
    NN.Compute(Pairs[Order[I]].I);
    NN.Backpropagate(Pairs[Order[I]].O);
  end;

  // Restore w, then apply g_adv with the plain optimizer.
  RestoreWeights(NN, Snap);
  FreeSnapshot(Snap);
  NN.UpdateWeights();  // momentum=0 -> per-neuron plain SGD (no inertia)
end;

procedure ShuffleIndices(var Idx: array of integer);
var
  I, J, Tmp: integer;
begin
  for I := High(Idx) downto 1 do
  begin
    J := Random(I + 1);
    Tmp := Idx[I]; Idx[I] := Idx[J]; Idx[J] := Tmp;
  end;
end;

// Mean cross-entropy on a softmax-output classifier vs one-hot targets.
function MeanCrossEntropy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
const
  cEps = 1e-12;
var
  I, K: integer;
  Sum, P, Tgt, SampleLoss: TNeuralFloat;
  Output: TNNetVolume;
begin
  Sum := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Output := NN.GetLastLayer().Output;
    SampleLoss := 0;
    for K := 0 to Output.Size - 1 do
    begin
      Tgt := 0;
      if K < Pairs[I].O.Size then Tgt := Pairs[I].O.FData[K];
      if Tgt > 0 then
      begin
        P := Output.FData[K];
        if P < cEps then P := cEps;
        if P > 1 then P := 1;
        SampleLoss := SampleLoss - Tgt * Ln(P);
      end;
    end;
    Sum := Sum + SampleLoss;
  end;
  if Pairs.Count > 0 then Result := Sum / Pairs.Count else Result := 0;
end;

function Accuracy(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I, Hit: integer;
begin
  Hit := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    if NN.GetLastLayer().Output.GetClass() = Pairs[I].O.GetClass() then
      Inc(Hit);
  end;
  if Pairs.Count > 0 then Result := Hit / Pairs.Count else Result := 0;
end;

// Train a fresh classifier for cEpochs at the given rho, returning the net.
function TrainArm(Train: TNNetVolumePairList; Rho: TNeuralFloat): TNNet;
var
  NN: TNNet;
  Order: array of integer;
  TrainIdx: array of integer;
  TrainCnt, Epoch, Lo, I: integer;
begin
  RandSeed := cSeed;            // identical seed before each arm
  BuildClassifier(NN);
  SetLength(TrainIdx, NN.GetLastLayerIdx() + 1);
  CollectTrainable(NN, TrainIdx, TrainCnt);

  SetLength(Order, Train.Count);
  for I := 0 to High(Order) do Order[I] := I;

  for Epoch := 1 to cEpochs do
  begin
    ShuffleIndices(Order);      // RNG used only between forward/backward passes
    Lo := 0;
    while Lo < Train.Count do
    begin
      I := Lo + cBatchSize;
      if I > Train.Count then I := Train.Count;
      SAMBatchStep(NN, Train, Order, Lo, I, Rho, TrainIdx, TrainCnt);
      Lo := I;
    end;
  end;
  Result := NN;
end;

// Max absolute difference between all weight tensors of two same-shape nets.
function MaxWeightDiff(A, B: TNNet): TNeuralFloat;
var
  L, NIdx, W: integer;
  Wa, Wb: TNNetVolume;
  D, M: TNeuralFloat;
begin
  M := 0;
  for L := 0 to A.GetLastLayerIdx() do
  begin
    if A.Layers[L].Neurons.Count = 0 then Continue;
    for NIdx := 0 to A.Layers[L].Neurons.Count - 1 do
    begin
      Wa := A.Layers[L].Neurons[NIdx].Weights;
      Wb := B.Layers[L].Neurons[NIdx].Weights;
      if (Wa = nil) or (Wb = nil) then Continue;
      for W := 0 to Wa.Size - 1 do
      begin
        D := Abs(Wa.FData[W] - Wb.FData[W]);
        if D > M then M := D;
      end;
      // bias too
      D := Abs(A.Layers[L].Neurons[NIdx].Bias - B.Layers[L].Neurons[NIdx].Bias);
      if D > M then M := D;
    end;
  end;
  Result := M;
end;

// Pull the two scalars we care about out of the LossLandscapeProbe report.
procedure ParseProbe(const Report: string; out Sharpness: TNeuralFloat;
  out DoublingStr: string);
var
  Lines: TStringList;
  I, P: integer;
  S: string;

  // Returns everything after the LAST '=' in Str. The probe's sharpness and
  // doubling lines also contain a '=' inside "h=..."; we want the value.
  function AfterLastEq(const Str: string): string;
  var
    K, Last: integer;
  begin
    Last := 0;
    for K := 1 to Length(Str) do
      if Str[K] = '=' then Last := K;
    if Last > 0 then Result := Trim(Copy(Str, Last + 1, Length(Str)))
    else Result := '';
  end;

begin
  Sharpness := NaN;
  DoublingStr := '(not found)';
  Lines := TStringList.Create();
  try
    Lines.Text := Report;
    for I := 0 to Lines.Count - 1 do
    begin
      S := Lines[I];
      P := Pos('=', S);
      if (Pos('sharpness', S) > 0) and (P > 0) then
        Sharpness := StrToFloatDef(AfterLastEq(S), NaN);
      if (Pos('loss-doubling radius', S) > 0) and (P > 0) then
        DoublingStr := AfterLastEq(S);
    end;
  finally
    Lines.Free;
  end;
end;

function FmtF(V: TNeuralFloat; W, D: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then
    Result := Format('%*s', [W, 'NaN/Inf'])
  else
    Result := Format('%*.*f', [W, D, V]);
end;

function FmtE(V: TNeuralFloat): string;
begin
  if IsNan(V) or IsInfinite(V) then
    Result := 'NaN/Inf'
  else
    Result := Format('%.6e', [V]);
end;

// Simple horizontal ASCII bar scaled to [Lo,Hi] over Width chars.
function Bar(V, Lo, Hi: TNeuralFloat; Width: integer): string;
var
  N: integer;
begin
  if IsNan(V) or IsInfinite(V) then begin Result := '(NaN/Inf)'; Exit; end;
  if Hi - Lo < 1e-12 then Hi := Lo + 1e-12;
  N := Round((V - Lo) / (Hi - Lo) * Width);
  if N < 0 then N := 0;
  if N > Width then N := Width;
  Result := StringOfChar('#', N);
end;

// ===========================================================================
var
  Train, Val: TNNetVolumePairList;
  NNplain, NNsam, NNrho0: TNNet;
  Rhos: array[0..4] of TNeuralFloat = (0.0, 0.01, 0.05, 0.1, 0.2);
  SweepSharp, SweepValAcc: TFloatArr;
  SweepTrLoss: TNeuralFloat;
  SweepDoubling: array of string;
  ReportPlain, ReportSam, ReportR: string;
  SharpPlain, SharpSam, SharpR: TNeuralFloat;
  DblPlain, DblSam, DblR: string;
  TrLossPlain, TrLossSam, VaLossPlain, VaLossSam: TNeuralFloat;
  TrAccPlain, TrAccSam, VaAccPlain, VaAccSam: TNeuralFloat;
  MaxDiff: TNeuralFloat;
  StartT: TDateTime;
  I: integer;
  SamRho: TNeuralFloat;
  MinS, MaxS, MinA, MaxA: TNeuralFloat;
  TrendOk: boolean;
  PrevS: TNeuralFloat;
  Decreases, Increases: integer;

begin
  // Mask FPU exceptions so any stray NaN/Inf propagates as a float VALUE we can
  // detect and print cleanly (guarded by IsNan/IsInfinite) instead of raising
  // EInvalidOp mid-report.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  StartT := Now;
  DefaultFormatSettings.DecimalSeparator := '.';
  SamRho := 0.05;   // headline SAM rho for the SGD-vs-SAM contrast

  WriteLn('========================================================================');
  WriteLn('Sharpness-Aware Minimization (Foret et al. 2021) on a noisy-label toy');
  WriteLn('========================================================================');
  WriteLn(Format('Dataset: %d-blob 2D classification, %d train (%.0f%% labels flipped) / %d clean val.',
    [cNumClasses, cNumTrain, cNoiseFrac * 100, cNumVal]));
  WriteLn(Format('Model:   2 -> %d(ReLU) -> %d(ReLU) -> %d(linear) -> softmax  (plain SGD, lr=%.3f, no momentum).',
    [cHiddenW, cHiddenW, cNumClasses, cLearnRate]));
  WriteLn(Format('Train:   %d epochs, batch=%d, seed=%d. Headline SAM rho=%.2f.',
    [cEpochs, cBatchSize, cSeed, SamRho]));
  WriteLn;

  // Build datasets once (after seeding) so both arms see identical data.
  RandSeed := cSeed;
  MakeBlobs(Train, cTrainPerCls, cNoiseFrac);
  MakeBlobs(Val,   cValPerCls,   0.0);

  // ----------------------- Arm A: plain SGD (rho=0) -----------------------
  NNplain := TrainArm(Train, 0.0);
  TrLossPlain := MeanCrossEntropy(NNplain, Train);
  VaLossPlain := MeanCrossEntropy(NNplain, Val);
  TrAccPlain  := Accuracy(NNplain, Train);
  VaAccPlain  := Accuracy(NNplain, Val);

  // ----------------------- Arm B: SAM (rho=SamRho) ------------------------
  NNsam := TrainArm(Train, SamRho);
  TrLossSam := MeanCrossEntropy(NNsam, Train);
  VaLossSam := MeanCrossEntropy(NNsam, Val);
  TrAccSam  := Accuracy(NNsam, Train);
  VaAccSam  := Accuracy(NNsam, Val);

  WriteLn('------------------------------------------------------------------------');
  WriteLn('SGD vs SAM (loss = mean cross-entropy, acc = top-1):');
  WriteLn('------------------------------------------------------------------------');
  WriteLn('  arm        train-loss   train-acc    val-loss    val-acc');
  WriteLn(Format('  plain SGD  %s   %s%%   %s  %s%%',
    [FmtF(TrLossPlain, 9, 4), FmtF(TrAccPlain * 100, 7, 2),
     FmtF(VaLossPlain, 9, 4), FmtF(VaAccPlain * 100, 6, 2)]));
  WriteLn(Format('  SAM rho=%.2f%s   %s%%   %s  %s%%',
    [SamRho, FmtF(TrLossSam, 7, 4), FmtF(TrAccSam * 100, 7, 2),
     FmtF(VaLossSam, 9, 4), FmtF(VaAccSam * 100, 6, 2)]));
  WriteLn;

  // ----------------------- Flatness contrast ------------------------------
  ReportPlain := TNNet.LossLandscapeProbe(NNplain, Val, cProbeK, cProbeR, 1, cProbeSeed);
  ReportSam   := TNNet.LossLandscapeProbe(NNsam,   Val, cProbeK, cProbeR, 1, cProbeSeed);
  ParseProbe(ReportPlain, SharpPlain, DblPlain);
  ParseProbe(ReportSam,   SharpSam,   DblSam);

  WriteLn('------------------------------------------------------------------------');
  WriteLn('Flatness contrast via TNNet.LossLandscapeProbe (cross-entropy loss):');
  WriteLn(Format('  (K=%d, R=%.2f, %d val samples, seed=%d -- same random direction for both)',
    [cProbeK, cProbeR, Val.Count, cProbeSeed]));
  WriteLn('------------------------------------------------------------------------');
  WriteLn('  arm        sharpness (2nd central diff)   loss-doubling radius');
  WriteLn(Format('  plain SGD  %s            %s', [FmtE(SharpPlain), DblPlain]));
  WriteLn(Format('  SAM        %s            %s', [FmtE(SharpSam),   DblSam]));
  if (not IsNan(SharpPlain)) and (not IsNan(SharpSam)) then
  begin
    if SharpSam < SharpPlain then
      WriteLn('  => SAM landed FLATTER (smaller sharpness scalar).')
    else
      WriteLn('  => SAM did NOT land flatter on this run.');
  end;
  WriteLn;

  // ----------------------- rho sweep --------------------------------------
  WriteLn('------------------------------------------------------------------------');
  WriteLn('rho sweep (each arm retrained from the SAME seed):');
  WriteLn('------------------------------------------------------------------------');
  SetLength(SweepSharp, Length(Rhos));
  SetLength(SweepValAcc, Length(Rhos));
  SetLength(SweepDoubling, Length(Rhos));
  WriteLn('     rho   train-loss   val-acc       sharpness        loss-doubling');
  for I := 0 to High(Rhos) do
  begin
    if Rhos[I] = 0.0 then
    begin
      // Reuse the plain-SGD net for rho=0 (also used for the invariant check).
      NNrho0 := NNplain;
      ReportR := ReportPlain;
      SharpR := SharpPlain; DblR := DblPlain;
      SweepValAcc[I] := VaAccPlain;
      SweepTrLoss := TrLossPlain;
    end
    else
    begin
      NNrho0 := TrainArm(Train, Rhos[I]);
      SweepTrLoss := MeanCrossEntropy(NNrho0, Train);
      SweepValAcc[I] := Accuracy(NNrho0, Val);
      ReportR := TNNet.LossLandscapeProbe(NNrho0, Val, cProbeK, cProbeR, 1, cProbeSeed);
      ParseProbe(ReportR, SharpR, DblR);
    end;
    SweepSharp[I] := SharpR;
    SweepDoubling[I] := DblR;
    WriteLn(Format('  %6.2f   %s   %s%%   %s     %s',
      [Rhos[I], FmtF(SweepTrLoss, 8, 4), FmtF(SweepValAcc[I] * 100, 6, 2),
       FmtE(SharpR), DblR]));
    if Rhos[I] <> 0.0 then NNrho0.Free;
  end;
  WriteLn;

  // ASCII chart: sharpness vs rho.
  MinS := 1e30; MaxS := -1e30;
  for I := 0 to High(Rhos) do
    if not IsNan(SweepSharp[I]) then
    begin
      if SweepSharp[I] < MinS then MinS := SweepSharp[I];
      if SweepSharp[I] > MaxS then MaxS := SweepSharp[I];
    end;
  WriteLn(Format('sharpness vs rho  [%.3e .. %.3e]:', [MinS, MaxS]));
  for I := 0 to High(Rhos) do
    WriteLn(Format('  rho=%4.2f | %s', [Rhos[I],
      Bar(SweepSharp[I], MinS, MaxS, 50)]));
  WriteLn;

  // ASCII chart: val-accuracy vs rho.
  MinA := 1e30; MaxA := -1e30;
  for I := 0 to High(Rhos) do
  begin
    if SweepValAcc[I] < MinA then MinA := SweepValAcc[I];
    if SweepValAcc[I] > MaxA then MaxA := SweepValAcc[I];
  end;
  WriteLn(Format('val-accuracy vs rho  [%.2f%% .. %.2f%%]:', [MinA * 100, MaxA * 100]));
  for I := 0 to High(Rhos) do
    WriteLn(Format('  rho=%4.2f | %s  %s%%', [Rhos[I],
      Bar(SweepValAcc[I], MinA, MaxA, 40), FmtF(SweepValAcc[I] * 100, 6, 2)]));
  WriteLn;

  // Trend verdict for invariant (2): does sharpness generally fall as rho rises?
  Decreases := 0; Increases := 0;
  PrevS := SweepSharp[0];
  for I := 1 to High(Rhos) do
  begin
    if (not IsNan(SweepSharp[I])) and (not IsNan(PrevS)) then
    begin
      if SweepSharp[I] < PrevS then Inc(Decreases)
      else if SweepSharp[I] > PrevS then Inc(Increases);
    end;
    PrevS := SweepSharp[I];
  end;
  TrendOk := (not IsNan(SweepSharp[0])) and (not IsNan(SweepSharp[High(Rhos)]))
             and (SweepSharp[High(Rhos)] < SweepSharp[0]);
  WriteLn('Invariant (2) -- sharpness trend over rising rho:');
  WriteLn(Format('  step-to-step decreases=%d, increases=%d; ' +
    'endpoint sharpness %s from rho=%.2f to rho=%.2f.',
    [Decreases, Increases,
     BoolToStr(TrendOk, 'FELL', 'did NOT fall'),
     Rhos[0], Rhos[High(Rhos)]]));
  WriteLn;

  // ----------------------- Invariant (1): rho=0 == plain SGD --------------
  // Retrain a SAM arm at rho=0 and compare to the plain-SGD arm bit-for-bit.
  NNrho0 := TrainArm(Train, 0.0);
  MaxDiff := MaxWeightDiff(NNplain, NNrho0);
  WriteLn('------------------------------------------------------------------------');
  WriteLn('Invariant (1) -- rho=0 SAM step == plain SGD, bit-for-bit:');
  WriteLn(Format('  max abs weight+bias diff (SAM rho=0 vs plain SGD) = %s',
    [FmtE(MaxDiff)]));
  if MaxDiff = 0.0 then
    WriteLn('  rho=0 == plain SGD: PASS')
  else
    WriteLn('  rho=0 == plain SGD: FAIL');
  NNrho0.Free;
  WriteLn;

  WriteLn(Format('Total wall time: %.2f s', [(Now - StartT) * 24 * 60 * 60]));

  NNplain.Free;
  NNsam.Free;
  Train.Free;
  Val.Free;
end.
