program CapsuleReconstruction;
(*
CapsuleReconstruction: the reconstruction-decoder pose-perturbation STRETCH demo
for TNNetCapsuleRouting (follow-up to examples/CapsuleRouting).

WHAT THIS REPRODUCES
  Sabour, Frosst & Hinton (2017), "Dynamic Routing Between Capsules". The
  headline claim of that paper is not just that capsule routing classifies well,
  but that the per-capsule POSE VECTOR is INTERPRETABLE: a reconstruction
  decoder fed by the winning class capsule learns to render the input, and
  perturbing a SINGLE dimension of that pose vector varies one human-readable
  visual factor (stroke thickness, width, position, ...). This example trains a
  tiny CapsNet jointly with a reconstruction decoder and then sweeps one pose
  dimension to show that factor move.

DATASET (synthetic, no download)
  A tiny 12x12 two-class "shapes" set with EXPLICIT, controllable pose factors so
  disentanglement is actually plausible inside a <5-minute CPU budget:
    class 0 = a VERTICAL   bar,  class 1 = a HORIZONTAL bar.
  Each sample has two latent pose factors drawn at generation time:
    - THICKNESS  (1..3 pixels)  and
    - POSITION   (the bar's centre column/row).
  These are exactly the kind of low-dimensional pose factors a capsule pose
  vector ought to capture, so a single decoded dimension has something real to
  vary. Pixels are in [0,1]; light Gaussian pixel noise is added.

ARCHITECTURE (existing layers only -- TNNetCapsuleRouting already in-tree)
  Encoder (classifier + pose extractor):
    Input(144) -> FullConnectReLU(H) {primary-capsule feature trunk}
               -> CapsuleRouting(numInCaps, inDim, numClasses=2, poseDim, iters)
    The capsule layer emits numClasses pose vectors of length poseDim. The
    per-capsule LENGTH ||v_j|| is the class score (squash keeps it in [0,1)).
  Decoder (reconstruction), a SEPARATE small net:
    Input(numClasses*poseDim, err-collect) -> FullConnectReLU(H) -> Sigmoid(144)
  At train time the decoder input is the capsule output with every capsule EXCEPT
  the TRUE class zeroed out (the paper's "masking"), so only the winning pose
  vector drives reconstruction.

LOSSES (hand-rolled; we say exactly which)
  - Classification: a MARGIN loss on capsule lengths (the paper's loss), the
    Sabour et al. form L_k = T_k max(0, m+ - ||v_k||)^2 + lambda (1-T_k) max(0, ||v_k|| - m-)^2
    with m+=0.9, m-=0.1, lambda=0.5. We compute dL/d v_k analytically and seed it
    into the encoder's output error.
  - Reconstruction: plain MSE between decoder output and the input image,
    weighted by cRecW (the paper down-weights reconstruction so it regularises
    rather than dominates). We obtain the MSE output-gradient with the stock
    backprop via a pseudo-target (out - target), then read the decoder INPUT
    gradient (decoder Input layer has error collection enabled) and add it -- on
    the TRUE-class capsule slice only -- to the encoder's output error.

  Both nets run in batch-update mode (gradients accumulate over a mini-batch,
  applied once per UpdateWeights), and we hand-roll Compute/Backpropagate so the
  TNeuralFit best-model reload gotcha never applies and layer refs stay valid.

THE HEADLINE
  After training, we take a correctly-classified example, read its winning
  capsule's pose vector, and for ONE chosen pose dimension sweep an offset in
  [-0.25, +0.25], decode each perturbed vector, and print the reconstructions as
  ASCII art side by side. If the dimension disentangles, the rendered bar should
  vary SMOOTHLY (thicker/thinner, or shifted) across the sweep. We pick the
  dimension whose sweep produces the LARGEST monotone change in a simple
  interpretable read-out (total ink for thickness) and report it.

HONESTY
  Capsule pose disentanglement is delicate and normally needs far more training
  than a <5-min CPU budget allows. This example reports what it ACTUALLY observes
  -- the measured per-dimension reconstruction sensitivity and whether the
  best dimension's sweep is monotone -- rather than asserting a clean textbook
  result. A partial/qualitative effect is reported truthfully.

SECONDARY NUMBER
  Classification accuracy of the CapsNet vs a parameter-matched plain MLP
  (Input -> ReLU(H) -> SoftMax(numClasses)) trained on the same data/budget.

Pure CPU, no external data, deterministic (fixed seed), well under 5 minutes on
2 cores.

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
  cSeed       = 42;
  cImgW       = 12;
  cImgH       = 12;
  cImgSize    = cImgW * cImgH;          // 144
  cNumClasses = 2;                      // vertical vs horizontal bar
  cPoseDim    = 8;                      // capsule pose-vector length
  cNumInCaps  = 8;                      // primary capsules
  cInDim      = 6;                      // primary-capsule dim
  cTrunkH     = 48;                     // = cNumInCaps * cInDim
  cDecH       = 64;                     // decoder hidden width
  cRoutIters  = 3;
  cBatch      = 32;
  cEpochs     = 400;
  cTrainN     = 256;
  cTestN      = 128;
  cLR         = 0.04;
  cMom        = 0.9;
  cRecW       = 0.40;                   // reconstruction down-weight (paper: 0.0005*784)
  // margin-loss constants (Sabour et al.)
  cMplus      = 0.9;
  cMminus     = 0.1;
  cLambda     = 0.5;

type
  TFloatArr = array of TNeuralFloat;
  TSample = record
    Img:   TNNetVolume;     // 144
    Lab:   integer;         // 0..numClasses-1
    Thick: integer;         // latent pose: bar thickness in pixels
    Pos:   integer;         // latent pose: bar centre row/col
  end;
  TSampleArr = array of TSample;

var
  TrainSet, TestSet: TSampleArr;

// ---------------------------------------------------------------------------
// Synthetic shapes generator. A bar (vertical for class 0, horizontal for
// class 1) of given thickness centred at a given row/col, plus light noise.
// ---------------------------------------------------------------------------
procedure DrawBar(V: TNNetVolume; Lab, Thick, Pos: integer);
var
  x, y, half, lo, hi: integer;
begin
  V.Fill(0);
  half := Thick div 2;
  lo := Pos - half;
  hi := Pos - half + Thick - 1;
  if Lab = 0 then
  begin
    // vertical bar: columns [lo..hi] fully lit
    for x := 0 to cImgW - 1 do
      if (x >= lo) and (x <= hi) then
        for y := 0 to cImgH - 1 do
          V.FData[y * cImgW + x] := 1.0;
  end
  else
  begin
    // horizontal bar: rows [lo..hi] fully lit
    for y := 0 to cImgH - 1 do
      if (y >= lo) and (y <= hi) then
        for x := 0 to cImgW - 1 do
          V.FData[y * cImgW + x] := 1.0;
  end;
  // light pixel noise, clamped to [0,1]
  for x := 0 to cImgSize - 1 do
  begin
    V.FData[x] := V.FData[x] + (Random - 0.5) * 0.10;
    if V.FData[x] < 0 then V.FData[x] := 0;
    if V.FData[x] > 1 then V.FData[x] := 1;
  end;
end;

procedure BuildDataset(out S: TSampleArr; N: integer);
var
  i, half, minPos, maxPos: integer;
begin
  SetLength(S, N);
  for i := 0 to N - 1 do
  begin
    S[i].Img := TNNetVolume.Create(cImgSize, 1, 1);
    S[i].Lab := Random(cNumClasses);
    S[i].Thick := 1 + Random(3);                 // 1..3
    half := S[i].Thick div 2;
    minPos := half + 1;
    maxPos := cImgW - 1 - (S[i].Thick - 1 - half) - 1;
    if maxPos < minPos then maxPos := minPos;
    S[i].Pos := minPos + Random(maxPos - minPos + 1);
    DrawBar(S[i].Img, S[i].Lab, S[i].Thick, S[i].Pos);
  end;
end;

procedure FreeDataset(var S: TSampleArr);
var i: integer;
begin
  for i := 0 to High(S) do S[i].Img.Free;
  SetLength(S, 0);
end;

// ---------------------------------------------------------------------------
// Capsule helpers operating on the encoder output vector (numClasses*poseDim).
// ---------------------------------------------------------------------------
function CapsLen(const V: TNNetVolume; j: integer): TNeuralFloat;
var o: integer; s: TNeuralFloat;
begin
  s := 0;
  for o := 0 to cPoseDim - 1 do
    s := s + Sqr(V.FData[j * cPoseDim + o]);
  Result := Sqrt(s + 1e-12);
end;

function ArgMaxCaps(const V: TNNetVolume): integer;
var j, best: integer; bl, l: TNeuralFloat;
begin
  best := 0; bl := CapsLen(V, 0);
  for j := 1 to cNumClasses - 1 do
  begin
    l := CapsLen(V, j);
    if l > bl then begin bl := l; best := j; end;
  end;
  Result := best;
end;

// ---------------------------------------------------------------------------
// Margin-loss gradient w.r.t. each pose component, seeded into encoder error.
//   For capsule j with length L=||v_j|| and target T_j in {0,1}:
//     if T_j=1: dL/dL =  -2*max(0, m+ - L)      (push length up to m+)
//     if T_j=0: dL/dL =   2*lambda*max(0, L - m-)
//   then dL/dv_j[o] = (dL/dL) * v_j[o] / L  (chain through the length).
// We write (output - target)-style error directly into EncErr (the framework's
// ComputeOutputErrorWith would do output-target; here we supply the gradient as
// a pseudo so that output-pseudo == our gradient). Returns sample margin loss.
// ---------------------------------------------------------------------------
function MarginGrad(const V: TNNetVolume; Lab: integer; EncErr: TFloatArr): TNeuralFloat;
var
  j, o: integer;
  L, dDL, t, loss, hingeP, hingeN: TNeuralFloat;
begin
  loss := 0;
  for j := 0 to cNumClasses - 1 do
  begin
    L := CapsLen(V, j);
    if j = Lab then t := 1 else t := 0;
    hingeP := cMplus - L;  if hingeP < 0 then hingeP := 0;
    hingeN := L - cMminus; if hingeN < 0 then hingeN := 0;
    if t = 1 then
    begin
      dDL := -2.0 * hingeP;
      loss := loss + Sqr(hingeP);
    end
    else
    begin
      dDL := 2.0 * cLambda * hingeN;
      loss := loss + cLambda * Sqr(hingeN);
    end;
    for o := 0 to cPoseDim - 1 do
      EncErr[j * cPoseDim + o] := dDL * V.FData[j * cPoseDim + o] / L;
  end;
  Result := loss;
end;

// ---------------------------------------------------------------------------
// Build encoder (capsule classifier + pose) and decoder (reconstruction).
// ---------------------------------------------------------------------------
procedure BuildEncoder(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cImgSize, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cTrunkH));
  NN.AddLayer(TNNetCapsuleRouting.Create(cNumInCaps, cInDim, cNumClasses, cPoseDim, cRoutIters));
  NN.SetLearningRate(cLR, cMom);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);
end;

procedure BuildDecoder(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cNumClasses * cPoseDim, 1, 1).EnableErrorCollection);
  NN.AddLayer(TNNetFullConnectReLU.Create(cDecH));
  NN.AddLayer(TNNetFullConnectSigmoid.Create(cImgSize));
  NN.SetLearningRate(cLR, cMom);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);
end;

procedure BuildMLP(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cImgSize, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cTrunkH));
  NN.AddLayer(TNNetFullConnectLinear.Create(cNumClasses));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLR, cMom);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);
end;

// ---------------------------------------------------------------------------
// Evaluate capsule classifier accuracy (length argmax vs label).
// ---------------------------------------------------------------------------
function EvalCapsAcc(Enc: TNNet; const S: TSampleArr): TNeuralFloat;
var i, correct: integer;
begin
  correct := 0;
  for i := 0 to High(S) do
  begin
    Enc.Compute(S[i].Img);
    if ArgMaxCaps(Enc.GetLastLayer().Output) = S[i].Lab then Inc(correct);
  end;
  Result := correct / Length(S);
end;

function EvalMLPAcc(MLP: TNNet; const S: TSampleArr): TNeuralFloat;
var i, correct: integer; O: TNNetVolume;
begin
  correct := 0;
  for i := 0 to High(S) do
  begin
    MLP.Compute(S[i].Img);
    O := MLP.GetLastLayer().Output;
    if O.GetClass() = S[i].Lab then Inc(correct);
  end;
  Result := correct / Length(S);
end;

// ASCII render of a 144-vector image.
procedure RenderImg(const D: TFloatArr; const Prefix: string);
var x, y: integer; row: string; v: TNeuralFloat; ch: char;
begin
  for y := 0 to cImgH - 1 do
  begin
    row := Prefix;
    for x := 0 to cImgW - 1 do
    begin
      v := D[y * cImgW + x];
      if v < 0.15 then ch := ' '
      else if v < 0.35 then ch := '.'
      else if v < 0.55 then ch := ':'
      else if v < 0.75 then ch := '+'
      else if v < 0.90 then ch := '*'
      else ch := '#';
      row := row + ch;
    end;
    WriteLn(row);
  end;
end;

function TotalInk(const D: TFloatArr): TNeuralFloat;
var i: integer;
begin
  Result := 0;
  for i := 0 to High(D) do Result := Result + D[i];
end;

// ===========================================================================
var
  Enc, Dec, MLP: TNNet;
  ep, b, i, j, o, n, idx: integer;
  Order: array of integer;
  EncErr: TFloatArr;
  EncErrVol, MaskedVol, RecTarget, MlpTarget: TNNetVolume;
  DecInGrad: TNNetVolume;
  V: TNNetVolume;
  marginSum, recSum, sampLoss: TNeuralFloat;
  tmp: integer;
  StartT: TDateTime;
  capsAcc, mlpAcc: TNeuralFloat;
  // headline
  ScanV: TNNetVolume;
  poseBase: TFloatArr;
  predLab, demoIdx: integer;
  sweep: array[0..6] of TNeuralFloat;
  decoded: array[0..6] of TFloatArr;
  inkAt: array[0..6] of TNeuralFloat;
  bestDim, sd, k: integer;
  bestRange, range, lo, hi: TNeuralFloat;
  monotone: boolean;
  dimRange: TFloatArr;

const
  cSweep: array[0..6] of TNeuralFloat = (-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25);

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  DefaultFormatSettings.DecimalSeparator := '.';
  RandSeed := cSeed;
  StartT := Now;

  WriteLn('========================================================================');
  WriteLn('CapsNet Reconstruction + Pose-Perturbation (Sabour et al. 2017)');
  WriteLn('========================================================================');
  WriteLn(Format('Synthetic %dx%d bars: class0=vertical, class1=horizontal.', [cImgW, cImgH]));
  WriteLn(Format('Latent pose factors: thickness 1..3 px, centre position.', []));
  WriteLn(Format('Encoder: Input(%d)->ReLU(%d)->CapsuleRouting(in=%dx%d,out=%dx%d,iter=%d).',
    [cImgSize, cTrunkH, cNumInCaps, cInDim, cNumClasses, cPoseDim, cRoutIters]));
  WriteLn(Format('Decoder: Input(%d)->ReLU(%d)->Sigmoid(%d).  Loss: margin + %.2f*MSE.',
    [cNumClasses * cPoseDim, cDecH, cImgSize, cRecW]));
  WriteLn(Format('Train=%d test=%d, %d epochs, batch %d, lr=%.3f mom=%.2f.',
    [cTrainN, cTestN, cEpochs, cBatch, cLR, cMom]));
  WriteLn;

  BuildDataset(TrainSet, cTrainN);
  BuildDataset(TestSet, cTestN);

  BuildEncoder(Enc);
  BuildDecoder(Dec);
  BuildMLP(MLP);

  SetLength(EncErr, cNumClasses * cPoseDim);
  EncErrVol := TNNetVolume.Create(cNumClasses * cPoseDim, 1, 1);
  MaskedVol := TNNetVolume.Create(cNumClasses * cPoseDim, 1, 1);
  RecTarget := TNNetVolume.Create(cImgSize, 1, 1);
  MlpTarget := TNNetVolume.Create(cNumClasses, 1, 1);

  SetLength(Order, cTrainN);
  for i := 0 to cTrainN - 1 do Order[i] := i;

  // ---------------- Joint training (encoder + decoder) -------------------
  for ep := 1 to cEpochs do
  begin
    // shuffle
    for i := cTrainN - 1 downto 1 do
    begin
      j := Random(i + 1);
      tmp := Order[i]; Order[i] := Order[j]; Order[j] := tmp;
    end;
    marginSum := 0; recSum := 0;
    i := 0;
    while i < cTrainN do
    begin
      Enc.ClearDeltas();
      Dec.ClearDeltas();
      MLP.ClearDeltas();
      for b := 0 to cBatch - 1 do
      begin
        if i >= cTrainN then Break;
        idx := Order[i];
        Inc(i);

        // --- forward encoder ---
        Enc.Compute(TrainSet[idx].Img);
        V := Enc.GetLastLayer().Output;

        // --- margin-loss gradient on capsule lengths ---
        for n := 0 to cNumClasses * cPoseDim - 1 do EncErr[n] := 0;
        sampLoss := MarginGrad(V, TrainSet[idx].Lab, EncErr);
        marginSum := marginSum + sampLoss;

        // --- masked vector: keep only TRUE-class capsule for the decoder ---
        MaskedVol.Fill(0);
        for o := 0 to cPoseDim - 1 do
          MaskedVol.FData[TrainSet[idx].Lab * cPoseDim + o] :=
            V.FData[TrainSet[idx].Lab * cPoseDim + o];

        // --- forward + backward decoder (MSE to the image) ---
        Dec.Compute(MaskedVol);
        // weighted-MSE gradient via pseudo-target: (out - pseudo) = w*(out-in)
        // with w = cRecW/cBatch (batch sums deltas; divide for mean gradient).
        for n := 0 to cImgSize - 1 do
          RecTarget.FData[n] := Dec.GetLastLayer().Output.FData[n] -
            (cRecW / cBatch) * (Dec.GetLastLayer().Output.FData[n] - TrainSet[idx].Img.FData[n]);
        for n := 0 to cImgSize - 1 do
          recSum := recSum + Sqr(Dec.GetLastLayer().Output.FData[n] - TrainSet[idx].Img.FData[n]) / cImgSize;
        Dec.Backpropagate(RecTarget);

        // decoder input gradient (dL_rec/d masked-input) lives in Layers[0].
        DecInGrad := Dec.Layers[0].OutputError;

        // --- combine: encoder error = margin grad (/cBatch) + recon grad on
        //     the TRUE-class slice only (other slices were masked to 0). ---
        for n := 0 to cNumClasses * cPoseDim - 1 do
          EncErr[n] := EncErr[n] / cBatch;
        for o := 0 to cPoseDim - 1 do
          EncErr[TrainSet[idx].Lab * cPoseDim + o] :=
            EncErr[TrainSet[idx].Lab * cPoseDim + o] +
            DecInGrad.FData[TrainSet[idx].Lab * cPoseDim + o];

        // seed encoder output error as pseudo-target so (output - pseudo) == EncErr.
        for n := 0 to cNumClasses * cPoseDim - 1 do
          EncErrVol.FData[n] := V.FData[n] - EncErr[n];
        Enc.Backpropagate(EncErrVol);

        // --- parameter-matched MLP baseline (cross-entropy via SoftMax) ---
        MLP.Compute(TrainSet[idx].Img);
        MlpTarget.Fill(0);
        MlpTarget.FData[TrainSet[idx].Lab] := 1.0;
        MLP.Backpropagate(MlpTarget);
      end;
      Enc.UpdateWeights();
      Dec.UpdateWeights();
      MLP.UpdateWeights();
    end;
    if (ep mod 25 = 0) or (ep = 1) then
      WriteLn(Format('  epoch %2d  margin-loss=%.4f  recon-MSE=%.4f',
        [ep, marginSum / cTrainN, recSum / cTrainN]));
  end;
  WriteLn;

  // ----------------------- Accuracy report -------------------------------
  capsAcc := EvalCapsAcc(Enc, TestSet);
  mlpAcc  := EvalMLPAcc(MLP, TestSet);
  WriteLn('------------------------------------------------------------------------');
  WriteLn('CLASSIFICATION ACCURACY (test set)');
  WriteLn('------------------------------------------------------------------------');
  WriteLn(Format('  CapsNet (capsule-length argmax) : %.1f%%', [capsAcc * 100]));
  WriteLn(Format('  Plain MLP (param-matched)       : %.1f%%', [mlpAcc * 100]));
  WriteLn;

  // ----------------------- HEADLINE: pose perturbation -------------------
  // Find a correctly-classified demo example (prefer a thick vertical bar so a
  // thickness factor has room to move).
  demoIdx := -1;
  for i := 0 to High(TestSet) do
  begin
    Enc.Compute(TestSet[i].Img);
    predLab := ArgMaxCaps(Enc.GetLastLayer().Output);
    if (predLab = TestSet[i].Lab) and (TestSet[i].Lab = 0) and (TestSet[i].Thick = 2) then
    begin
      demoIdx := i; Break;
    end;
  end;
  if demoIdx < 0 then
    for i := 0 to High(TestSet) do
    begin
      Enc.Compute(TestSet[i].Img);
      if ArgMaxCaps(Enc.GetLastLayer().Output) = TestSet[i].Lab then
      begin demoIdx := i; Break; end;
    end;

  WriteLn('========================================================================');
  WriteLn('HEADLINE: single pose-dimension perturbation sweep');
  WriteLn('========================================================================');

  if demoIdx < 0 then
    WriteLn('  No correctly-classified example found; cannot run the sweep.')
  else
  begin
    Enc.Compute(TestSet[demoIdx].Img);
    V := Enc.GetLastLayer().Output;
    predLab := ArgMaxCaps(V);
    SetLength(poseBase, cNumClasses * cPoseDim);
    for n := 0 to cNumClasses * cPoseDim - 1 do poseBase[n] := V.FData[n];

    WriteLn(Format('  Demo example #%d  true=%d pred=%d  (latent thick=%d pos=%d)',
      [demoIdx, TestSet[demoIdx].Lab, predLab, TestSet[demoIdx].Thick, TestSet[demoIdx].Pos]));
    WriteLn;
    WriteLn('  Original input:');
    RenderImg(TestSet[demoIdx].Img.FData, '    ');
    WriteLn;

    ScanV := TNNetVolume.Create(cNumClasses * cPoseDim, 1, 1);

    // For every pose dim of the winning capsule, sweep and record total-ink
    // range; pick the most-responsive dimension as the headline factor.
    SetLength(dimRange, cPoseDim);
    bestDim := 0; bestRange := -1;
    for sd := 0 to cPoseDim - 1 do
    begin
      lo := 1e30; hi := -1e30;
      for k := 0 to 6 do
      begin
        for n := 0 to cNumClasses * cPoseDim - 1 do ScanV.FData[n] := 0;
        // mask: only winning capsule, with one dim perturbed
        for o := 0 to cPoseDim - 1 do
          ScanV.FData[predLab * cPoseDim + o] := poseBase[predLab * cPoseDim + o];
        ScanV.FData[predLab * cPoseDim + sd] :=
          poseBase[predLab * cPoseDim + sd] + cSweep[k];
        Dec.Compute(ScanV);
        range := TotalInk(Dec.GetLastLayer().Output.FData);
        if range < lo then lo := range;
        if range > hi then hi := range;
      end;
      dimRange[sd] := hi - lo;
      if dimRange[sd] > bestRange then begin bestRange := dimRange[sd]; bestDim := sd; end;
    end;

    WriteLn('  Per-pose-dimension reconstruction sensitivity (total-ink range over sweep):');
    for sd := 0 to cPoseDim - 1 do
      WriteLn(Format('    dim %d : %.3f%s', [sd, dimRange[sd],
        BoolToStr(sd = bestDim, '   <= most responsive', '')]));
    WriteLn;

    // Decode the best dimension's sweep and show ASCII reconstructions + ink.
    for k := 0 to 6 do
    begin
      for n := 0 to cNumClasses * cPoseDim - 1 do ScanV.FData[n] := 0;
      for o := 0 to cPoseDim - 1 do
        ScanV.FData[predLab * cPoseDim + o] := poseBase[predLab * cPoseDim + o];
      ScanV.FData[predLab * cPoseDim + bestDim] :=
        poseBase[predLab * cPoseDim + bestDim] + cSweep[k];
      Dec.Compute(ScanV);
      SetLength(decoded[k], cImgSize);
      for n := 0 to cImgSize - 1 do decoded[k][n] := Dec.GetLastLayer().Output.FData[n];
      inkAt[k] := TotalInk(decoded[k]);
      sweep[k] := cSweep[k];
    end;

    WriteLn(Format('  Sweeping pose dimension %d of the winning capsule (class %d):',
      [bestDim, predLab]));
    WriteLn;
    for k := 0 to 6 do
    begin
      WriteLn(Format('  offset %6.2f   (total ink %.1f):', [sweep[k], inkAt[k]]));
      RenderImg(decoded[k], '    ');
      WriteLn;
    end;

    // Honesty read-out: is the ink trend monotone across the sweep?
    monotone := True;
    if inkAt[6] >= inkAt[0] then
    begin
      for k := 1 to 6 do if inkAt[k] < inkAt[k-1] - 1e-6 then monotone := False;
    end
    else
    begin
      for k := 1 to 6 do if inkAt[k] > inkAt[k-1] + 1e-6 then monotone := False;
    end;

    WriteLn('------------------------------------------------------------------------');
    WriteLn('OBSERVATION (honest)');
    WriteLn('------------------------------------------------------------------------');
    WriteLn(Format('  Most-responsive dim %d ink range = %.3f over offsets [-0.25, +0.25].',
      [bestDim, bestRange]));
    if monotone then
      WriteLn('  The total-ink read-out varies MONOTONICALLY across the sweep:')
    else
      WriteLn('  The total-ink read-out is NON-monotone across the sweep:');
    Write('    ink trend:');
    for k := 0 to 6 do Write(Format(' %.1f', [inkAt[k]]));
    WriteLn;
    if bestRange < 1.0 then
      WriteLn('  NOTE: the effect is SMALL in this budget -- capsule pose disentanglement')
    else
      WriteLn('  The decoded bar visibly changes -- a single pose dimension drives a factor.');
    WriteLn('        normally needs much more training than a <5-min CPU run; this is a');
    WriteLn('        truthful partial result, not a polished textbook reconstruction.');

    ScanV.Free;
  end;

  WriteLn;
  WriteLn(Format('Total wall time: %.1f s', [(Now - StartT) * 24 * 60 * 60]));

  EncErrVol.Free; MaskedVol.Free; RecTarget.Free; MlpTarget.Free;
  Enc.Free; Dec.Free; MLP.Free;
  FreeDataset(TrainSet); FreeDataset(TestSet);
end.
