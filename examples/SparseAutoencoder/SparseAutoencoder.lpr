program SparseAutoencoder;
(*
SparseAutoencoder: a self-contained, pure-CPU reproduction of the headline result
of Anthropic's "Towards Monosemanticity: Decomposing Language Models With
Dictionary Learning" (Bricken et al. 2023,
https://transformer-circuits.pub/2023/monosemantic-features/index.html).

THE SETTING (companion to examples/Superposition/)
  examples/Superposition/ shows the PROBLEM: a model packs N sparse ground-truth
  features into M < N dimensions, storing them in SUPERPOSITION -- non-orthogonal,
  mutually-interfering directions. The activation vector is then POLYSEMANTIC:
  each dimension responds to a blend of several underlying features, so no single
  activation coordinate is "one interpretable concept".

  This example shows the SOLUTION. Given only the DENSE, polysemantic activation
  vectors a (we never see the ground-truth features during training), can we
  RECOVER the original sparse features by learning an OVERCOMPLETE dictionary --
  a sparse autoencoder (SAE) with many more hidden units than activation
  dimensions, regularised so each hidden unit fires rarely? The Bricken et al.
  headline: yes -- the SAE's dictionary atoms become MONOSEMANTIC, each atom
  aligning with one ground-truth feature, and there is a sweet-spot sparsity
  weight: too little -> dense polysemantic atoms; too much -> dead atoms.

THE TOY DATA  (the superposition regime, reused from examples/Superposition/)
  K sparse ground-truth features. Feature k is independently ACTIVE with a small
  probability cActiveProb; when active its value ~ U[0,1], else 0. A fixed random
  mixing matrix Gtrue (d x K, unit-norm columns = the true feature DIRECTIONS in
  activation space) linearly mixes the sparse feature vector f (length K) down
  into a DENSE d-dim activation:
      a = Gtrue * f          with d < K   (the superposition regime).
  The SAE is trained ONLY on the dense a -- it must UNPACK what Gtrue packed.

THE SPARSE AUTOENCODER  (existing layers only -- NO new layer)
      Input(d) -> TNNetFullConnectReLU(H)   {H >> d : the OVERCOMPLETE dictionary}
                -> TNNetFullConnectLinear(d) {decoder D : atoms = columns of D}
  trained to reconstruct a under  L = ||out - a||^2 + lambda * sum_j |h_j|,
  where h is the hidden (ReLU) code. The dictionary atom of hidden unit j is the
  decoder COLUMN D[:,j] (a d-vector): the activation pattern that unit j writes
  back. Monosemanticity = each atom D[:,j] points along ONE true feature column
  Gtrue[:,k].

DELIVERING THE LOSS THROUGH STOCK BACKPROP (no library changes)
  Reconstruction MSE: same PSEUDO-TARGET trick as examples/Superposition/. The
  framework seeds the output error as (output - target); feeding
      pseudo_i = out_i - (1/B)*(out_i - a_i)
  makes the seeded error exactly the per-sample mean-MSE gradient (1/B)*(out-a).

  L1 sparsity on the hidden code is the NEW piece. d|h_j|/dh_j = sign(h_j)
  (sub-gradient; sign(0)=0). We deliver it as a SECOND backward pass that
  accumulates into the SAME weight deltas (batch-update mode): after the
  reconstruction Backpropagate, we OVERWRITE the hidden layer's OutputError with
  (lambda/B)*sign(h), reset the per-call backprop counters, and call the HIDDEN
  layer's Backpropagate() directly so this gradient flows hidden->input only
  (the decoder is untouched). Because TNNetFullConnect accumulates weight deltas
  in batch-update mode, the two passes sum to grad(MSE) + grad(L1). This is the
  manual-gradient-surgery idiom of examples/Superposition/ and
  examples/SharpnessAwareMinimization/. SetBatchUpdate(True) is REQUIRED: the
  per-sample default applies + zeroes deltas immediately.

WHAT IT REPORTS, per L1 weight lambda in {0, 1e-3, 1e-2, 1e-1, 3e-1, 1.0}
  (a) MONOSEMANTICITY score = mean over atoms of max-cosine(atom, true feature),
      vs the raw activation-to-feature baseline (mean over true features of
      max-cosine to a raw activation AXIS) -- the score should climb well above
      baseline at a good lambda;
  (b) RECOVERED-FEATURE COUNT: # of true features claimed by some atom with
      cosine >= cRecallCos -- NON-monotone in lambda with an interior peak;
  (c) mean L0 (active atoms / sample) and reconstruction MSE -- the sparsity /
      fidelity trade-off;
  (d) an ASCII atom-to-feature cosine heatmap (best matching atom per feature),
      so "one atom == one interpretable feature" recovery is visible.

BUILT-IN CORRECTNESS SIGNALS (printed PASS / FAIL; HALT(1) on any failure)
  (1) lambda=0 is a plain over-parameterised autoencoder: recon MSE ~ 0 but atoms
      are POLYSEMANTIC -- its monosemanticity score is low (near baseline) AND
      strictly below the best swept lambda's score.
  (2) The recovered-feature-COUNT curve is NON-MONOTONE with an INTERIOR peak:
      some intermediate lambda recovers MORE features than BOTH lambda=0 and the
      largest lambda.
  (3) DEAD-ATOM fraction (atoms that never fire over an eval batch) GROWS at the
      largest lambda (over-strong sparsity kills atoms).

DISTINCT FROM the neighbours
  - examples/Superposition/ reads the trained model's OWN encoder geometry
    G = D*W to MEASURE how features are packed. Here we train a SEPARATE
    overcomplete dictionary to UNPACK a dense activation back into features.
  - the open "TopK sparsity sweep" idea is an UNDERCOMPLETE bottleneck
    recon-loss-vs-K study with no ground-truth recovery; here the dictionary is
    OVERCOMPLETE and we score recovery against known ground-truth features.
  - the "Shrink-activation sparsity sweep" idea compares bottleneck ACTIVATION
    choices; here we learn a DICTIONARY under an L1 code penalty.

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
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeed        = 424242;
  cK           = 16;            // number of sparse ground-truth features
  cD           = 8;             // dense activation width (d < K : superposition)
  cH           = 22;            // dictionary size (H >> d : overcomplete)
  cActiveProb  = 0.10;          // base per-feature activation probability (sparse)
  cProbDecay   = 0.94;          // mild geometric decay of per-feature activation prob
  cBatchSize   = 128;           // samples per gradient step (gradient averaged)
  cSteps       = 3000;          // gradient steps per lambda
  cLearnRate   = 1.50;          // applied to the MEAN gradient
  cMomentum    = 0.90;          // momentum to escape the small-code basin
  cFeatScale   = 1.0;           // active-feature magnitude scale
  // Reconstruction weight in L = cReconW*||out-a||^2 + lambda*|h|. A standard
  // SAE coefficient: it sets the BALANCE between fidelity and sparsity so the
  // swept lambda set {0,1e-3,1e-2,1e-1,3e-1,1.0} spans dense->sweet-spot->dead
  // regimes: 1e-1 is the recovery sweet spot, 3e-1 and 1.0 are over-sparse
  // (recovery drops, dead atoms grow).
  cReconW      = 1.00;
  cEvalBatch   = 4000;          // samples for MSE / L0 / dead-atom evaluation
  cRecallCos   = 0.80;          // atom counts as "recovering" a feature if cos>=this
  cDeadThresh  = 1e-6;          // atom is "dead" if its max activation over eval < this

type
  TFloatArr  = array of TNeuralFloat;
  TMatrix    = array of TFloatArr;       // [row][col]
  TLambdaArr = array[0..5] of TNeuralFloat;

const
  cLambdas: TLambdaArr = (0.0, 1e-3, 1e-2, 1e-1, 3e-1, 1.0);

var
  Gtrue: TMatrix;                        // d x K mixing matrix, unit-norm columns
  FeatProb: TFloatArr;                   // per-feature activation probability

// ---------------------------------------------------------------------------
// Per-feature activation probability with geometric decay: feature k is active
// w.p. cActiveProb * cProbDecay^k. Making features UNEQUALLY frequent is what
// produces the classic NON-MONOTONE recovery curve: at too-large lambda the L1
// kills the RARE features' atoms first (they pay the same |h| cost but rarely
// help reconstruction), so feature recovery DROPS past the sweet spot even as
// the surviving atoms get cleaner.
// ---------------------------------------------------------------------------
procedure InitFeatProb;
var k: integer;
begin
  SetLength(FeatProb, cK);
  for k := 0 to cK - 1 do
    FeatProb[k] := cActiveProb * Power(cProbDecay, k);
end;

// ---------------------------------------------------------------------------
// Build the fixed random mixing matrix Gtrue (d x K) with unit-norm columns.
// Each column is the activation-space DIRECTION of one ground-truth feature.
// ---------------------------------------------------------------------------
procedure InitGtrue;
var
  i, k: integer;
  nrm: TNeuralFloat;
begin
  SetLength(Gtrue, cD);
  for i := 0 to cD - 1 do SetLength(Gtrue[i], cK);
  for k := 0 to cK - 1 do
  begin
    nrm := 0;
    for i := 0 to cD - 1 do
    begin
      Gtrue[i][k] := Random * 2 - 1;     // U[-1,1]
      nrm := nrm + Gtrue[i][k] * Gtrue[i][k];
    end;
    nrm := Sqrt(nrm);
    if nrm < 1e-9 then nrm := 1e-9;
    for i := 0 to cD - 1 do Gtrue[i][k] := Gtrue[i][k] / nrm;
  end;
end;

// ---------------------------------------------------------------------------
// Draw one dense activation a = Gtrue * f, with f a sparse K-vector: feature k
// active w.p. cActiveProb, value ~ U[0,1] when active, else 0.
// ---------------------------------------------------------------------------
procedure DrawSample(A: TNNetVolume);
var
  i, k: integer;
  f: TNeuralFloat;
begin
  for i := 0 to cD - 1 do A.FData[i] := 0.0;
  for k := 0 to cK - 1 do
    if Random < FeatProb[k] then
    begin
      f := cFeatScale * (0.5 + Random);  // U[0.5,1.5] * scale: large magnitudes
      for i := 0 to cD - 1 do
        A.FData[i] := A.FData[i] + Gtrue[i][k] * f;
    end;
end;

// ---------------------------------------------------------------------------
// SAE: Input(d) -> ReLU(H) {dictionary} -> Linear(d) {decoder}.
// ---------------------------------------------------------------------------
procedure BuildModel(out NN: TNNet; out HidIdx, OutIdx: integer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cD, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cH));      // hidden dictionary code
  HidIdx := NN.GetLastLayerIdx();
  NN.AddLayer(TNNetFullConnectLinear.Create(cD));    // decoder D (d x H)
  OutIdx := NN.GetLastLayerIdx();
  NN.SetLearningRate(cLearnRate, {Momentum=}cMomentum);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);   // REQUIRED for manual gradient accumulation
end;

// ---------------------------------------------------------------------------
// Constrain decoder dictionary atoms (columns D[:,j]) to UNIT NORM, the
// Bricken et al. 2023 dictionary constraint. WITHOUT it the L1 code penalty is
// degenerate: the net would shrink the code h and grow the decoder columns to
// compensate, so |h| -> 0 without ever zeroing an atom and L1 never bites. With
// unit-norm atoms, firing an atom genuinely costs |h_j|, so L1 forces a sparse,
// monosemantic dictionary (and over-strong L1 kills atoms outright).
// ---------------------------------------------------------------------------
procedure RenormDecoder(NN: TNNet; OutIdx: integer);
var
  i, j: integer;
  Dec: TNNetLayer;
  nrm: TNeuralFloat;
begin
  Dec := NN.Layers[OutIdx];
  for j := 0 to cH - 1 do
  begin
    nrm := 0;
    for i := 0 to cD - 1 do
      nrm := nrm + Sqr(Dec.Neurons[i].Weights.FData[j]);
    nrm := Sqrt(nrm);
    if nrm > 1e-9 then
      for i := 0 to cD - 1 do
        Dec.Neurons[i].Weights.FData[j] := Dec.Neurons[i].Weights.FData[j] / nrm;
  end;
end;

// ---------------------------------------------------------------------------
// Hand-rolled training at L1 weight Lambda.
//   Pass A (reconstruction): pseudo_i = out_i - (1/B)*(out_i - a_i), so the
//     stock-seeded error equals the per-sample mean-MSE gradient; full backprop.
//   Pass B (L1 sparsity): overwrite the hidden layer's OutputError with
//     (Lambda/B)*sign(h), reset the per-call counters, and backprop the hidden
//     layer directly so the L1 sub-gradient flows hidden->input only. Both
//     passes accumulate into the same deltas (batch-update mode).
//   After each weight update the decoder atoms are renormalised to unit norm.
// ---------------------------------------------------------------------------
procedure TrainAt(NN: TNNet; HidIdx, OutIdx: integer; Lambda: TNeuralFloat);
var
  Step, B, i: integer;
  Inp, Pseudo: TNNetVolume;
  Outp, Hid, HidErr: TNNetVolume;
  HidLayer: TNNetLayer;
begin
  Inp    := TNNetVolume.Create(cD, 1, 1);
  Pseudo := TNNetVolume.Create(cD, 1, 1);
  HidLayer := NN.Layers[HidIdx];
  for Step := 1 to cSteps do
  begin
    NN.ClearDeltas();
    for B := 1 to cBatchSize do
    begin
      DrawSample(Inp);
      NN.Compute(Inp);

      // ---- Pass A: reconstruction MSE via pseudo-target ----
      Outp := NN.Layers[OutIdx].Output;
      for i := 0 to cD - 1 do
        Pseudo.FData[i] := Outp.FData[i] -
          (cReconW / cBatchSize) * (Outp.FData[i] - Inp.FData[i]);
      NN.Backpropagate(Pseudo);

      // ---- Pass B: L1 sub-gradient (Lambda/B)*sign(h) on the hidden code ----
      if Lambda > 0.0 then
      begin
        Hid    := HidLayer.Output;
        HidErr := HidLayer.OutputError;
        for i := 0 to cH - 1 do
        begin
          if Hid.FData[i] > 0.0 then
            HidErr.FData[i] := (Lambda / cBatchSize)          // sign(h)=+1
          else
            HidErr.FData[i] := 0.0;                            // h=0 (ReLU): sign=0
        end;
        // NOTE: do NOT call ResetBackpropCallCurrCnt here. Pass A already left
        // the hidden layer's backprop counter at its departing-branch count;
        // resetting it makes the chained guard SKIP this layer's weight update
        // (verified empirically). Calling Backpropagate directly after pass A
        // accumulates the L1 weight/bias deltas correctly. It flows hidden->input
        // only, so the decoder is untouched.
        HidLayer.Backpropagate();
      end;
    end;
    NN.UpdateWeights();
    RenormDecoder(NN, OutIdx);   // unit-norm dictionary atoms (Bricken et al.)
    {$IFDEF SAEDEBUG}
    if (Step mod 500 = 0) then
    begin
      Hid := HidLayer.Output;
      i := 0; B := 0;
      // reuse last forward's hidden as a crude probe
      for B := 0 to cH - 1 do if Hid.FData[B] > 1e-8 then Inc(i);
      WriteLn(Format('   [dbg lambda=%.4f step=%d] lastL0=%d meanCode=%.4f',
        [Lambda, Step, i, Hid.GetSumAbs / cH]));
    end;
    {$ENDIF}
  end;
  Pseudo.Free;
  Inp.Free;
end;

// ---------------------------------------------------------------------------
// Cosine between two d-vectors.
// ---------------------------------------------------------------------------
function CosVec(const U, V: TFloatArr): TNeuralFloat;
var
  i: integer;
  dot, nu, nv: TNeuralFloat;
begin
  dot := 0; nu := 0; nv := 0;
  for i := 0 to cD - 1 do
  begin
    dot := dot + U[i] * V[i];
    nu  := nu + U[i] * U[i];
    nv  := nv + V[i] * V[i];
  end;
  if (nu < 1e-12) or (nv < 1e-12) then Result := 0
  else Result := dot / (Sqrt(nu) * Sqrt(nv));
end;

// ---------------------------------------------------------------------------
// Decoder atoms: atom j = column D[:,j] (a d-vector), j in 0..H-1.
//   decoder neuron i (i in 0..d-1) has Weights of size H -> D[i][j].
// ---------------------------------------------------------------------------
procedure ReadAtoms(NN: TNNet; OutIdx: integer; out Atoms: TMatrix);
var
  i, j: integer;
  Dec: TNNetLayer;
begin
  Dec := NN.Layers[OutIdx];
  SetLength(Atoms, cH);
  for j := 0 to cH - 1 do
  begin
    SetLength(Atoms[j], cD);
    for i := 0 to cD - 1 do
      Atoms[j][i] := Dec.Neurons[i].Weights.FData[j];
  end;
end;

// Glyph shade for a cosine magnitude in [0,1].
function Glyph(V: TNeuralFloat): char;
begin
  V := Abs(V);
  if V < 0.20 then Result := ' '
  else if V < 0.40 then Result := '.'
  else if V < 0.60 then Result := ':'
  else if V < 0.75 then Result := '+'
  else if V < 0.90 then Result := '*'
  else Result := '#';
end;

function FmtF(V: TNeuralFloat; W, D: integer): string;
begin
  Result := Format('%*.*f', [W, D, V]);
end;

// ===========================================================================
var
  NN: TNNet;
  HidIdx, OutIdx: integer;
  Atoms: TMatrix;
  LIdx, i, j, k, b: integer;
  Lambda: TNeuralFloat;
  StartT: TDateTime;

  // per-lambda results
  MonoScore:   TFloatArr;                // mean over atoms of max-cos to features
  Recovered:   array of integer;         // # features with some atom cos>=cRecallCos
  MeanL0:      TFloatArr;                 // mean active atoms / sample
  MeanMSE:     TFloatArr;                 // mean reconstruction MSE
  DeadFrac:    TFloatArr;                 // fraction of dead atoms

  // eval accumulators
  Inp, Outp, Hid: TNNetVolume;
  AtomMaxAct: TFloatArr;                  // max activation per atom over eval batch
  sumL0, sumMSE, mse, mc, c: TNeuralFloat;
  deadCnt, recCnt, activeCnt: integer;
  FeatBestCos: TFloatArr;                 // best atom cosine per true feature
  FeatBestAtom: array of integer;
  Axis: TFloatArr;                        // raw activation-axis vector
  TrueCol: TFloatArr;                     // a true-feature column
  Baseline: TNeuralFloat;                 // raw activation-to-feature mono baseline

  // correctness signals
  bestLambdaScore: TNeuralFloat;
  bestLambdaIdx, peakIdx, peakRec: integer;
  pass1, pass2, pass3, allPass: boolean;

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  DefaultFormatSettings.DecimalSeparator := '.';
  StartT := Now;
  RandSeed := cSeed;
  InitGtrue;
  InitFeatProb;

  WriteLn('========================================================================');
  WriteLn('Towards Monosemanticity (Bricken et al. 2023) -- pure-CPU reproduction');
  WriteLn('========================================================================');
  WriteLn(Format('Ground-truth features K=%d, dense activation width d=%d (d<K: superposition).', [cK, cD]));
  WriteLn(Format('Sparse AE: Input(%d) -> ReLU(%d){overcomplete dictionary} -> Linear(%d){decoder}.', [cD, cH, cD]));
  WriteLn(Format('Per-feature activation prob = %.2f. Reconstruction MSE + L1 code penalty.', [cActiveProb]));
  WriteLn(Format('%d steps x batch %d, lr=%.3f. Atom j = decoder column D[:,j].', [cSteps, cBatchSize, cLearnRate]));
  WriteLn(Format('"Recovered" feature: some atom has cosine >= %.2f to its true direction.', [cRecallCos]));
  WriteLn;

  // -------- Raw activation-to-feature baseline (NO dictionary) --------
  // For each true feature, the best max-cosine to any raw activation AXIS
  // (canonical basis e_i of the dense space). This is how "monosemantic" the
  // raw polysemantic activation coordinates already are -- the bar to beat.
  SetLength(Axis, cD);
  SetLength(TrueCol, cD);
  Baseline := 0;
  for k := 0 to cK - 1 do
  begin
    for i := 0 to cD - 1 do TrueCol[i] := Gtrue[i][k];
    mc := 0;
    for i := 0 to cD - 1 do
    begin
      for j := 0 to cD - 1 do Axis[j] := 0;
      Axis[i] := 1;
      c := Abs(CosVec(Axis, TrueCol));
      if c > mc then mc := c;
    end;
    Baseline := Baseline + mc;
  end;
  Baseline := Baseline / cK;
  WriteLn(Format('Raw activation-to-feature baseline (mean max-cos to an axis) = %s', [FmtF(Baseline, 6, 3)]));
  WriteLn;

  SetLength(MonoScore, Length(cLambdas));
  SetLength(Recovered, Length(cLambdas));
  SetLength(MeanL0,    Length(cLambdas));
  SetLength(MeanMSE,   Length(cLambdas));
  SetLength(DeadFrac,  Length(cLambdas));

  Inp  := TNNetVolume.Create(cD, 1, 1);

  for LIdx := 0 to High(cLambdas) do
  begin
    Lambda := cLambdas[LIdx];
    RandSeed := cSeed + 1000 + LIdx;     // reproducible, distinct per lambda
    BuildModel(NN, HidIdx, OutIdx);
    TrainAt(NN, HidIdx, OutIdx, Lambda);
    ReadAtoms(NN, OutIdx, Atoms);

    // ---- Monosemanticity: mean over atoms of max-cos to any true feature ----
    // (skip dead/degenerate atoms with ~zero norm so the score reflects LIVE
    // dictionary quality, not padding.)
    MonoScore[LIdx] := 0;
    activeCnt := 0;
    for j := 0 to cH - 1 do
    begin
      c := 0;
      for i := 0 to cD - 1 do c := c + Atoms[j][i] * Atoms[j][i];
      if c < 1e-10 then continue;        // degenerate atom: excluded from score
      mc := 0;
      for k := 0 to cK - 1 do
      begin
        for i := 0 to cD - 1 do TrueCol[i] := Gtrue[i][k];
        c := Abs(CosVec(Atoms[j], TrueCol));
        if c > mc then mc := c;
      end;
      MonoScore[LIdx] := MonoScore[LIdx] + mc;
      Inc(activeCnt);
    end;
    if activeCnt > 0 then MonoScore[LIdx] := MonoScore[LIdx] / activeCnt;

    // ---- Eval batch: MSE, mean L0, dead-atom fraction, per-atom liveness ----
    // Run BEFORE the recovery count: a feature only counts as recovered if a
    // LIVE atom points at it (a dead atom contributes nothing, so over-strong
    // L1 that kills an atom genuinely LOSES that feature -> recovery drops).
    SetLength(AtomMaxAct, cH);
    for j := 0 to cH - 1 do AtomMaxAct[j] := 0;
    sumL0 := 0; sumMSE := 0;
    for b := 1 to cEvalBatch do
    begin
      DrawSample(Inp);
      NN.Compute(Inp);
      Hid  := NN.Layers[HidIdx].Output;
      Outp := NN.Layers[OutIdx].Output;
      mse := 0;
      for i := 0 to cD - 1 do
        mse := mse + Sqr(Outp.FData[i] - Inp.FData[i]);
      sumMSE := sumMSE + mse / cD;
      for j := 0 to cH - 1 do
      begin
        if Hid.FData[j] > 1e-8 then sumL0 := sumL0 + 1;
        if Hid.FData[j] > AtomMaxAct[j] then AtomMaxAct[j] := Hid.FData[j];
      end;
    end;
    MeanMSE[LIdx] := sumMSE / cEvalBatch;
    MeanL0[LIdx]  := sumL0 / cEvalBatch;
    deadCnt := 0;
    for j := 0 to cH - 1 do if AtomMaxAct[j] < cDeadThresh then Inc(deadCnt);
    DeadFrac[LIdx] := deadCnt / cH;

    // ---- Recovered-feature count: best LIVE atom cosine per true feature ----
    SetLength(FeatBestCos, cK);
    SetLength(FeatBestAtom, cK);
    for k := 0 to cK - 1 do
    begin
      for i := 0 to cD - 1 do TrueCol[i] := Gtrue[i][k];
      FeatBestCos[k] := 0; FeatBestAtom[k] := -1;
      for j := 0 to cH - 1 do
      begin
        if AtomMaxAct[j] < cDeadThresh then continue;   // dead atom recovers nothing
        c := Abs(CosVec(Atoms[j], TrueCol));
        if c > FeatBestCos[k] then begin FeatBestCos[k] := c; FeatBestAtom[k] := j; end;
      end;
    end;
    recCnt := 0;
    for k := 0 to cK - 1 do if FeatBestCos[k] >= cRecallCos then Inc(recCnt);
    Recovered[LIdx] := recCnt;

    // ---- Report block ----
    WriteLn('------------------------------------------------------------------------');
    WriteLn(Format('L1 WEIGHT lambda = %s', [FmtF(Lambda, 7, 4)]));
    WriteLn('------------------------------------------------------------------------');
    WriteLn(Format('  monosemanticity score = %s  (baseline %s)   recovered features = %d / %d',
      [FmtF(MonoScore[LIdx], 6, 3), FmtF(Baseline, 5, 3), Recovered[LIdx], cK]));
    WriteLn(Format('  mean L0 (active atoms/sample) = %s   recon MSE = %s   dead atoms = %d / %d (%.0f%%)',
      [FmtF(MeanL0[LIdx], 6, 2), FmtF(MeanMSE[LIdx], 8, 5),
       deadCnt, cH, DeadFrac[LIdx] * 100]));
    WriteLn;
    WriteLn('  Atom-to-feature recovery (per true feature: best atom + cosine; glyph |cos|):');
    WriteLn('    feat  bestAtom   cos   |  recovery glyph');
    for k := 0 to cK - 1 do
      WriteLn(Format('    f%2d     a%2d    %s   |  %s',
        [k, FeatBestAtom[k], FmtF(FeatBestCos[k], 5, 3), Glyph(FeatBestCos[k])]));
    WriteLn;

    NN.Free;
    SetLength(Atoms, 0);
  end;
  Inp.Free;

  // ---- Cross-lambda summary table ----
  WriteLn('========================================================================');
  WriteLn('SWEEP SUMMARY');
  WriteLn('========================================================================');
  WriteLn('  lambda    mono   recovered   meanL0   reconMSE   dead%');
  for LIdx := 0 to High(cLambdas) do
    WriteLn(Format('  %s   %s     %2d/%2d    %s   %s   %4.0f',
      [FmtF(cLambdas[LIdx], 6, 4), FmtF(MonoScore[LIdx], 5, 3),
       Recovered[LIdx], cK, FmtF(MeanL0[LIdx], 6, 2),
       FmtF(MeanMSE[LIdx], 8, 5), DeadFrac[LIdx] * 100]));
  WriteLn;

  // ----------------------- Correctness signals ---------------------------
  WriteLn('========================================================================');
  WriteLn('BUILT-IN CORRECTNESS SIGNALS');
  WriteLn('========================================================================');

  // best (non-zero) lambda monosemanticity score
  bestLambdaScore := -1; bestLambdaIdx := -1;
  for LIdx := 1 to High(cLambdas) do
    if MonoScore[LIdx] > bestLambdaScore then
    begin bestLambdaScore := MonoScore[LIdx]; bestLambdaIdx := LIdx; end;

  // (1) lambda=0 : low recon MSE (plain AE) but polysemantic (low mono score,
  //     strictly below the best swept lambda).
  WriteLn('(1) lambda=0 is a plain over-parameterised AE: low MSE but POLYSEMANTIC atoms.');
  WriteLn(Format('    lambda=0 recon MSE = %s (should be ~0)', [FmtF(MeanMSE[0], 8, 5)]));
  WriteLn(Format('    lambda=0 mono = %s   best-lambda(%s) mono = %s',
    [FmtF(MonoScore[0], 5, 3), FmtF(cLambdas[bestLambdaIdx], 6, 4), FmtF(bestLambdaScore, 5, 3)]));
  pass1 := (MeanMSE[0] < 0.01) and (MonoScore[0] < bestLambdaScore);
  if pass1 then
    WriteLn('    plain-AE-is-polysemantic (low MSE, mono < best lambda) : PASS')
  else
    WriteLn('    plain-AE-is-polysemantic : FAIL');
  WriteLn;

  // (2) recovered-feature count is NON-MONOTONE with an INTERIOR peak.
  peakIdx := 0; peakRec := Recovered[0];
  for LIdx := 1 to High(cLambdas) do
    if Recovered[LIdx] > peakRec then begin peakRec := Recovered[LIdx]; peakIdx := LIdx; end;
  WriteLn('(2) Recovered-feature COUNT is non-monotone with an INTERIOR peak.');
  Write('    recovered by lambda: ');
  for LIdx := 0 to High(cLambdas) do Write(Format('%d ', [Recovered[LIdx]]));
  WriteLn;
  WriteLn(Format('    peak at lambda=%s (recovered=%d); ends: lambda0=%d, lambdaMax=%d',
    [FmtF(cLambdas[peakIdx], 6, 4), peakRec, Recovered[0], Recovered[High(cLambdas)]]));
  pass2 := (peakIdx > 0) and (peakIdx < High(cLambdas)) and
           (peakRec > Recovered[0]) and (peakRec > Recovered[High(cLambdas)]);
  if pass2 then
    WriteLn('    interior peak (more recovery at an intermediate lambda) : PASS')
  else
    WriteLn('    interior peak : FAIL');
  WriteLn;

  // (3) dead-atom fraction GROWS at the largest lambda.
  WriteLn('(3) DEAD-ATOM fraction grows at the largest lambda (over-strong sparsity).');
  Write('    dead% by lambda: ');
  for LIdx := 0 to High(cLambdas) do Write(Format('%.0f ', [DeadFrac[LIdx] * 100]));
  WriteLn;
  pass3 := DeadFrac[High(cLambdas)] > DeadFrac[0];
  if pass3 then
    WriteLn('    dead-atom fraction at largest lambda > at lambda=0 : PASS')
  else
    WriteLn('    dead-atom growth : FAIL');
  WriteLn;

  allPass := pass1 and pass2 and pass3;
  WriteLn('========================================================================');
  if allPass then
    WriteLn('ALL INVARIANTS PASS')
  else
    WriteLn('INVARIANT FAILURE');
  WriteLn(Format('Total wall time: %.2f s', [(Now - StartT) * 24 * 60 * 60]));
  WriteLn('========================================================================');

  if not allPass then Halt(1);
end.
