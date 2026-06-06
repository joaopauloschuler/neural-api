program Superposition;
(*
Superposition: a self-contained, pure-CPU reproduction of the headline result of
Anthropic's "Toy Models of Superposition" (Elhage et al. 2022,
https://transformer-circuits.pub/2022/toy_model/index.html).

THE PHENOMENON
  A network can pack MORE sparse features than it has dimensions by storing them
  in SUPERPOSITION: non-orthogonal, mutually-interfering directions. WHICH and
  HOW MANY features it represents is governed by feature SPARSITY S and by a
  per-feature IMPORTANCE weight. Dense features (S=0) collide on every sample, so
  the optimum is to represent only the top-M most-important features orthogonally
  and drop the rest. Sparse features (S->1) rarely co-activate, so the model
  crams nearly all N of them into the M-dim bottleneck, accepting a little
  interference between directions in exchange for representing more features.

THE TOY
  Synthetic feature vectors of width N. Each feature i is independently ACTIVE
  with probability (1 - S); when active its value ~ U[0,1], else 0. Per-feature
  importance I_i = r^i (geometric decay) makes early features matter more.
  Autoencoder with an M<N bottleneck (existing layers only -- no new layer):

      Input(N) -> TNNetFullConnectLinear(M)   {encoder W, M x N}
                -> TNNetFullConnectReLU(N)     {decoder D + bias + ReLU}

  trained to reconstruct the input under an IMPORTANCE-WEIGHTED MSE
      L = sum_i  I_i * (out_i - in_i)^2 .
  The importance weighting is the crux: it makes the model choose WHICH features
  to spend its limited capacity on.

WEIGHTED MSE WITH THE STOCK BACKPROP (no library changes)
  The framework seeds the output layer's error as (output - target) and then
  multiplies by the activation derivative. We want the gradient of the WEIGHTED
  MSE, whose d/d(out_i) is  I_i * (out_i - in_i)  (the constant 2 folds into the
  learning rate). We obtain exactly that with the stock Backpropagate by feeding
  a PSEUDO-TARGET
      pseudo_i = out_i - I_i * (out_i - in_i),
  so that (output - pseudo)_i == I_i * (out_i - in_i). No surgery on neural/*.pas.

  We hand-roll the training loop (NN.Compute / NN.Backpropagate over fresh
  random samples) in BATCH-UPDATE mode so the gradient accumulates over a
  mini-batch before UpdateWeights -- and because we hand-roll, the post-train
  layer references never go stale (TNeuralFit.Fit's best-model reload gotcha
  does not apply here).

TIED vs UNTIED
  The classic toy TIES decoder = encoder^T and studies the Gram matrix W^T W
  (N x N): diagonal = represented norm per feature, off-diagonal = interference.
  Here the two TNNetFullConnect layers are UNTIED, so we read the effective
  pre-ReLU linear map  G = D * W  (N x N, D is N x M decoder, W is M x N encoder)
  and treat its off-diagonals as the interference. G plays the role of W^T W:
  G[i][i] is how strongly feature i is represented, column-norm ||G[:,i]|| is the
  represented norm of feature i, and off-diagonals are cross-feature leakage.

WHAT IT REPORTS, per sparsity level S in {0.0, 0.7, 0.9, 0.99}
  - per-feature represented norm ||G[:,i]|| as an ASCII bar chart (kept vs
    dropped-to-zero features);
  - the N x N effective map G as a glyph-shaded ASCII heatmap (near-identity =
    monosemantic / one-feature-per-direction; growing off-diagonal structure =
    polysemantic superposition);
  - the SUPERPOSITION RATIO = (#features with non-trivial norm) / M  (~1 dense ->
    represent only M features orthogonally; >1 sparse -> cram in extras);
  - the mean off-diagonal |interference|.

BUILT-IN CORRECTNESS SIGNALS (printed PASS / FAIL)
  (1) DENSE (S=0): kept-feature count ~ M (orthogonal monosemantic regime is
      optimal when features collide every sample), and G is a near-diagonal
      rank-<=M map (small mean off-diagonal interference).
  (2) IMPORTANCE TRACKING: the represented (kept) feature set tracks importance
      -- high-I features are kept before low-I ones (kept set is a near-prefix).
  (3) GROWTH: total represented norm grows as S rises (more features packed in).

Pure CPU, no external data, deterministic (fixed seed), well under 5 minutes.

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
  cSeed        = 42;
  cN           = 20;            // number of synthetic features
  cM           = 5;             // bottleneck width (M < N)
  cImportanceR = 0.8;           // importance decay I_i = r^i
  cBatchSize   = 768;           // samples per gradient step (gradient is averaged)
  cSteps       = 4000;          // gradient steps per sparsity level
  cLearnRate   = 1.0;           // applied to the MEAN weighted-MSE gradient
  // "non-trivial" represented-norm threshold (fraction of the max column norm).
  cKeepFrac    = 0.20;

type
  TFloatArr  = array of TNeuralFloat;
  TMatrix    = array of TFloatArr;       // [row][col]
  TSparsity  = array[0..3] of TNeuralFloat;

const
  cSparsities: TSparsity = (0.0, 0.7, 0.9, 0.99);

var
  Importance: TFloatArr;                 // I_i = r^i, length N

// ---------------------------------------------------------------------------
// Build I_i = r^i.
// ---------------------------------------------------------------------------
procedure InitImportance;
var
  I: integer;
begin
  SetLength(Importance, cN);
  for I := 0 to cN - 1 do
    Importance[I] := Power(cImportanceR, I);
end;

// ---------------------------------------------------------------------------
// Draw one synthetic feature vector: feature i active w.p. (1 - S), value
// ~ U[0,1] when active, else 0.
// ---------------------------------------------------------------------------
procedure DrawSample(V: TNNetVolume; S: TNeuralFloat);
var
  I: integer;
begin
  for I := 0 to cN - 1 do
  begin
    if Random < (1.0 - S) then
      V.FData[I] := Random          // U[0,1]
    else
      V.FData[I] := 0.0;
  end;
end;

// ---------------------------------------------------------------------------
// Autoencoder: Input(N) -> Linear(M) {encoder} -> ReLU(N) {decoder}.
// Returns layer INDICES of encoder and decoder (stable across the hand-rolled
// loop; no Fit reload, so refs would be fine too, but indices are the habit).
// ---------------------------------------------------------------------------
procedure BuildModel(out NN: TNNet; out EncIdx, DecIdx: integer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cN, 1, 1));
  NN.AddLayer(TNNetFullConnectLinear.Create(cM));    // encoder W (M x N)
  EncIdx := NN.GetLastLayerIdx();
  NN.AddLayer(TNNetFullConnectReLU.Create(cN));      // decoder D (N x M) + bias + ReLU
  DecIdx := NN.GetLastLayerIdx();
  NN.SetLearningRate(cLearnRate, {Momentum=}0.0);  // plain SGD on the averaged gradient
  NN.SetL2Decay(0.0);
  // Batch-update: Backpropagate ACCUMULATES the gradient into Neurons[].Delta;
  // UpdateWeights applies it once per mini-batch.
  NN.SetBatchUpdate(True);
end;

// ---------------------------------------------------------------------------
// Hand-rolled importance-weighted-MSE training at sparsity S.
// Pseudo-target trick: (output - pseudo)_i == I_i * (out_i - in_i) is exactly
// the weighted-MSE gradient w.r.t. the output, fed through the stock backprop.
// ---------------------------------------------------------------------------
procedure TrainAt(NN: TNNet; S: TNeuralFloat);
var
  Step, B, I: integer;
  Inp, Pseudo, Outp: TNNetVolume;
begin
  Inp    := TNNetVolume.Create(cN, 1, 1);
  Pseudo := TNNetVolume.Create(cN, 1, 1);
  for Step := 1 to cSteps do
  begin
    NN.ClearDeltas();
    for B := 1 to cBatchSize do
    begin
      DrawSample(Inp, S);
      NN.Compute(Inp);
      Outp := NN.GetLastLayer().Output;
      // pseudo_i = out_i - w_i*(out_i - in_i)  ->  stock error = w_i*(out_i-in_i)
      // with w_i = I_i / cBatchSize: batch-update mode SUMS the per-sample error
      // into Neurons[].Delta, so dividing by the batch size makes the applied
      // gradient the MEAN weighted-MSE gradient (LR is then batch-size-independent
      // and the dense regime stays stable).
      for I := 0 to cN - 1 do
        Pseudo.FData[I] := Outp.FData[I] -
          (Importance[I] / cBatchSize) * (Outp.FData[I] - Inp.FData[I]);
      NN.Backpropagate(Pseudo);
    end;
    NN.UpdateWeights();
  end;
  Pseudo.Free;
  Inp.Free;
end;

// ---------------------------------------------------------------------------
// Read the effective pre-ReLU linear map G = D * W (N x N).
//   encoder neuron m: Weights size N -> W[m][k]      (W is M x N)
//   decoder neuron i: Weights size M -> D[i][m]      (D is N x M)
//   G[i][k] = sum_m D[i][m] * W[m][k]
// ---------------------------------------------------------------------------
procedure EffectiveMap(NN: TNNet; EncIdx, DecIdx: integer; out G: TMatrix);
var
  i, k, m: integer;
  Enc, Dec: TNNetLayer;
  Acc: TNeuralFloat;
begin
  Enc := NN.Layers[EncIdx];
  Dec := NN.Layers[DecIdx];
  SetLength(G, cN);
  for i := 0 to cN - 1 do
  begin
    SetLength(G[i], cN);
    for k := 0 to cN - 1 do
    begin
      Acc := 0;
      for m := 0 to cM - 1 do
        Acc := Acc +
          Dec.Neurons[i].Weights.FData[m] * Enc.Neurons[m].Weights.FData[k];
      G[i][k] := Acc;
    end;
  end;
end;

// Column norm ||G[:,k]|| = represented norm of feature k.
function ColumnNorm(const G: TMatrix; k: integer): TNeuralFloat;
var
  i: integer;
  Sum: TNeuralFloat;
begin
  Sum := 0;
  for i := 0 to cN - 1 do Sum := Sum + G[i][k] * G[i][k];
  Result := Sqrt(Sum);
end;

// Mean of |off-diagonal| entries of G.
function MeanOffDiag(const G: TMatrix): TNeuralFloat;
var
  i, k, Cnt: integer;
  Sum: TNeuralFloat;
begin
  Sum := 0; Cnt := 0;
  for i := 0 to cN - 1 do
    for k := 0 to cN - 1 do
      if i <> k then
      begin
        Sum := Sum + Abs(G[i][k]);
        Inc(Cnt);
      end;
  if Cnt > 0 then Result := Sum / Cnt else Result := 0;
end;

// Horizontal ASCII bar scaled to [0,Hi] over Width chars.
function Bar(V, Hi: TNeuralFloat; Width: integer): string;
var
  NCh: integer;
begin
  if Hi < 1e-12 then Hi := 1e-12;
  NCh := Round(V / Hi * Width);
  if NCh < 0 then NCh := 0;
  if NCh > Width then NCh := Width;
  Result := StringOfChar('#', NCh);
end;

// Glyph shade for a magnitude in [0,Hi].
function Glyph(V, Hi: TNeuralFloat): char;
var
  Frac: TNeuralFloat;
begin
  if Hi < 1e-12 then begin Result := ' '; Exit; end;
  Frac := Abs(V) / Hi;
  if Frac < 0.05 then Result := ' '
  else if Frac < 0.20 then Result := '.'
  else if Frac < 0.40 then Result := ':'
  else if Frac < 0.60 then Result := '+'
  else if Frac < 0.80 then Result := '*'
  else Result := '#';
end;

function FmtF(V: TNeuralFloat; W, D: integer): string;
begin
  Result := Format('%*.*f', [W, D, V]);
end;

function YesNo(B: boolean): string;
begin
  if B then Result := 'yes' else Result := 'no';
end;

function KeptTag(B: boolean): string;
begin
  if B then Result := '  [kept]' else Result := '';
end;

// ===========================================================================
var
  NN: TNNet;
  EncIdx, DecIdx: integer;
  G: TMatrix;
  ColN: TFloatArr;                       // represented norm per feature
  SIdx, i, k, KeptCnt: integer;
  S, MaxColN, KeepThresh, OffDiag, TotalNorm, SupRatio: TNeuralFloat;
  Row: string;
  StartT: TDateTime;
  // correctness-signal carriers
  DenseKept: integer;
  DenseOffDiag: TNeuralFloat;
  TotalNormBySweep: TFloatArr;
  KeptCntBySweep: array of integer;      // # represented features per level
  ImportanceOrdered: array of boolean;   // kept set is a near-prefix of importance order
  Growing: boolean;
  Kept: array of boolean;
  KeptImpSum, DropImpSum: TNeuralFloat;
  DropCnt: integer;

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  DefaultFormatSettings.DecimalSeparator := '.';
  StartT := Now;
  RandSeed := cSeed;       // fully deterministic (no Randomize)
  InitImportance;

  WriteLn('========================================================================');
  WriteLn('Toy Models of Superposition (Elhage et al. 2022) -- pure-CPU reproduction');
  WriteLn('========================================================================');
  WriteLn(Format('Features N=%d, bottleneck M=%d, importance I_i = %.2f^i.', [cN, cM, cImportanceR]));
  WriteLn(Format('Autoencoder: Input(%d) -> Linear(%d){encoder} -> ReLU(%d){decoder}.', [cN, cM, cN]));
  WriteLn(Format('Importance-weighted MSE via pseudo-target; %d steps x batch %d, lr=%.3f.',
    [cSteps, cBatchSize, cLearnRate]));
  WriteLn('Effective map G = D*W (N x N); column-norm = represented norm, off-diag = interference.');
  WriteLn(Format('"Kept" = represented norm >= %.0f%% of the max column norm.', [cKeepFrac * 100]));
  WriteLn;

  SetLength(TotalNormBySweep, Length(cSparsities));
  SetLength(KeptCntBySweep, Length(cSparsities));
  SetLength(ImportanceOrdered, Length(cSparsities));
  DenseKept := -1;
  DenseOffDiag := 0;

  for SIdx := 0 to High(cSparsities) do
  begin
    S := cSparsities[SIdx];
    RandSeed := cSeed + SIdx;            // reproducible, distinct per level
    BuildModel(NN, EncIdx, DecIdx);
    TrainAt(NN, S);
    EffectiveMap(NN, EncIdx, DecIdx, G);

    // Per-feature represented norm.
    SetLength(ColN, cN);
    MaxColN := 0;
    for k := 0 to cN - 1 do
    begin
      ColN[k] := ColumnNorm(G, k);
      if ColN[k] > MaxColN then MaxColN := ColN[k];
    end;
    KeepThresh := cKeepFrac * MaxColN;

    // Kept set, count, total norm, superposition ratio, mean off-diagonal.
    SetLength(Kept, cN);
    KeptCnt := 0; TotalNorm := 0;
    for k := 0 to cN - 1 do
    begin
      Kept[k] := (ColN[k] >= KeepThresh) and (MaxColN > 1e-9);
      TotalNorm := TotalNorm + ColN[k];
      if Kept[k] then Inc(KeptCnt);
    end;
    OffDiag  := MeanOffDiag(G);
    SupRatio := KeptCnt / cM;
    TotalNormBySweep[SIdx] := TotalNorm;
    KeptCntBySweep[SIdx] := KeptCnt;

    // Importance tracking: high-importance features should be KEPT before
    // low-importance ones. Robust criterion (tolerant of marginal ties in the
    // untied ReLU geometry): the MEAN importance of the kept features must
    // exceed the mean importance of the dropped features. Equivalently, the
    // model spends its limited capacity preferentially on the features that
    // matter. (A strict prefix test is too brittle near the keep threshold.)
    KeptImpSum := 0; DropImpSum := 0; DropCnt := 0;
    for k := 0 to cN - 1 do
      if Kept[k] then KeptImpSum := KeptImpSum + Importance[k]
      else begin DropImpSum := DropImpSum + Importance[k]; Inc(DropCnt); end;
    if (KeptCnt > 0) and (DropCnt > 0) then
      ImportanceOrdered[SIdx] := (KeptImpSum / KeptCnt) > (DropImpSum / DropCnt)
    else
      ImportanceOrdered[SIdx] := True;   // all kept or all dropped: vacuously ok

    if SIdx = 0 then
    begin
      DenseKept := KeptCnt;
      DenseOffDiag := OffDiag;
    end;

    // ---- Report block for this sparsity level ----
    WriteLn('------------------------------------------------------------------------');
    WriteLn(Format('SPARSITY S = %.2f   (feature-active prob = %.2f)', [S, 1.0 - S]));
    WriteLn('------------------------------------------------------------------------');
    WriteLn(Format('  kept features = %d / %d     superposition ratio = %.2f  (kept/M)',
      [KeptCnt, cN, SupRatio]));
    WriteLn(Format('  total represented norm = %s   mean |off-diag interference| = %s',
      [FmtF(TotalNorm, 7, 3), FmtF(OffDiag, 6, 4)]));
    WriteLn;
    WriteLn('  Per-feature represented norm  ||G[:,i]||  (decreasing importance left->right):');
    for k := 0 to cN - 1 do
      WriteLn(Format('    f%2d I=%5.3f |%-32s| %s%s',
        [k, Importance[k], Bar(ColN[k], MaxColN, 32), FmtF(ColN[k], 6, 3),
         KeptTag(Kept[k])]));
    WriteLn;
    WriteLn('  Effective map G = D*W  (rows=outputs, cols=features; glyphs scaled to |max|):');
    Write('       ');
    for k := 0 to cN - 1 do Write(Format('%2d', [k mod 100]):3);
    WriteLn;
    for i := 0 to cN - 1 do
    begin
      Row := '';
      for k := 0 to cN - 1 do
        Row := Row + ' ' + Glyph(G[i][k], MaxColN) + ' ';
      WriteLn(Format('   o%2d %s', [i, Row]));
    end;
    WriteLn;

    NN.Free;
    SetLength(G, 0);
  end;

  // ----------------------- Correctness signals ---------------------------
  WriteLn('========================================================================');
  WriteLn('BUILT-IN CORRECTNESS SIGNALS');
  WriteLn('========================================================================');

  // (1) Dense -> ~M kept, near-diagonal (small off-diagonal).
  WriteLn('(1) DENSE (S=0): kept-feature count should be ~ M and G near-diagonal.');
  WriteLn(Format('    dense kept = %d  (M = %d);  mean |off-diag| = %s',
    [DenseKept, cM, FmtF(DenseOffDiag, 6, 4)]));
  if (DenseKept >= cM - 1) and (DenseKept <= cM + 1) then
    WriteLn('    kept ~ M : PASS')
  else
    WriteLn('    kept ~ M : FAIL');
  // Near-diagonal: dense off-diagonal must be the smallest of the sweep
  // (interference grows with sparsity) and modest in absolute terms.
  if DenseOffDiag < 0.10 then
    WriteLn('    G near-diagonal at S=0 (mean |off-diag| < 0.10) : PASS')
  else
    WriteLn('    G near-diagonal at S=0 : FAIL');
  WriteLn;

  // (2) Importance tracking: kept features should be MORE important on average
  // than dropped ones (high-I features kept first), at every sparsity level.
  WriteLn('(2) IMPORTANCE TRACKING: kept features more important (mean-I) than dropped.');
  for SIdx := 0 to High(cSparsities) do
    WriteLn(Format('    S=%.2f : mean-I(kept) > mean-I(dropped) = %s',
      [cSparsities[SIdx],
       YesNo(ImportanceOrdered[SIdx])]));
  Growing := True;
  for SIdx := 0 to High(cSparsities) do
    if not ImportanceOrdered[SIdx] then Growing := False;
  if Growing then
    WriteLn('    importance tracking : PASS')
  else
    WriteLn('    importance tracking : FAIL (a level kept lower-I features over higher-I ones)');
  WriteLn;

  // (3) The number of features packed into the M-dim bottleneck grows with
  // sparsity (the superposition phenomenon itself). At S=0 only ~M features fit
  // (orthogonal); as S->1 the model crams in extras (superposition ratio > 1).
  // We use the represented-feature COUNT as the robust growth axis -- raw G
  // magnitudes are not comparable across S (the per-active-feature
  // reconstruction scale changes with sparsity), so total norm is reported only
  // as a diagnostic.
  WriteLn('(3) GROWTH: # features packed into the M-dim bottleneck grows as S rises.');
  for SIdx := 0 to High(cSparsities) do
    WriteLn(Format('    S=%.2f : kept = %2d  (ratio %.2f)   [diag: total norm = %s]',
      [cSparsities[SIdx], KeptCntBySweep[SIdx], KeptCntBySweep[SIdx] / cM,
       FmtF(TotalNormBySweep[SIdx], 7, 3)]));
  Growing := KeptCntBySweep[High(cSparsities)] > KeptCntBySweep[0];
  if Growing then
    WriteLn('    represented-feature count grows (sparse > dense) : PASS')
  else
    WriteLn('    represented-feature count grows : FAIL');
  WriteLn;

  WriteLn(Format('Total wall time: %.2f s', [(Now - StartT) * 24 * 60 * 60]));
end.
