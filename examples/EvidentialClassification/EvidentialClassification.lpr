program EvidentialClassification;
(*
EvidentialClassification: a tiny demo of Evidential (Dirichlet) Deep Learning for
CLASSIFICATION (Sensoy, Kaplan & Kandemir, NeurIPS 2018, "Evidential Deep
Learning to Quantify Classification Uncertainty",
https://arxiv.org/abs/1806.01768) — the classification sibling of the
EvidentialRegression example.

THE PROBLEM. Two well-separated 2-D Gaussian blobs are the in-distribution
classes, placed SIDE BY SIDE in the upper half-plane: class 0 centred at
(-2.5, +2.5) and class 1 at (+2.5, +2.5), both with small spread. They share the
"up" direction, so the discriminative axis is x but the entire LOWER half-plane
is off BOTH manifolds (this geometry — rather than antipodal blobs, which would
tile the plane into two confident halves — is what makes the OOD region provable).
A plain softmax classifier always returns a confident probability vector, even
for inputs that look nothing like the training data. Evidential Deep Learning
instead treats the network output as EVIDENCE for a Dirichlet over the
class-probability simplex. From the K=2 concentrations alpha_k = softplus(out_k)+1
it reads off, in ONE deterministic forward pass (no sampling, no ensemble):
    strength    S   = sum_k alpha_k
    probability p_k = alpha_k / S
    UNCERTAINTY u   = K / S   in [0,1].
u -> 0 means strong evidence (a confident, in-distribution prediction); u -> 1
means NO evidence (the network abstains). The interesting quantity is u: it stays
LOW on the two training blobs and RISES toward 1 in the lower half-plane and far
shell, because the evidence collapses where the network has accumulated none.

MODEL. a SATURATING (tanh) trunk (x,y) -> 32 -tanh-> 32 -tanh-> so that far
       out-of-distribution inputs drive the hidden units flat and the head falls
       back to its zero-evidence (maximal-uncertainty) prior instead of
       extrapolating evidence -> TNNetFullConnectLinear(K=2) ->
       TNNetEvidentialClassification(K=2, lambda). The head turns the trunk
       output into the Dirichlet concentrations via softplus links and OWNS the
       EDL Bayes-risk-MSE + KL loss; its Backpropagate emits the exact
       dL/d(raw evidence).

TRAINING. A manual mini-batch loop (a few thousand updates, well under a minute
on two CPU cores). The framework seeds the head's error with (output - target);
we set the target volume to the ONE-HOT label (the head reads the label from
there). The evidence starts LOW (a high-uncertainty prior) and only GROWS where
accumulated evidence (training data) supports it, so out-of-distribution probes
keep their low-evidence (high-uncertainty) prior.

REPORT. After training we probe (1) points near the two upper blob centres
(in-distribution) and (2) points in the lower half-plane / far shell
(out-of-distribution) and print the predicted class, probabilities and
uncertainty u (which reaches exactly 1 far below the blobs). Expected headline:
the mean uncertainty on the OOD probes is several times larger and near 1, while
the in-distribution uncertainty is near 0.

Small CPU toy; printing is NaN/Inf guarded. Not added to the main README (see
examples/README.md).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  HIDDEN   = 32;        // trunk hidden width
  NCLASSES = 2;         // two blobs
  SPREAD   = 0.40;      // blob standard deviation
  LAMBDA   = 0.02;      // KL-regularizer weight (annealing coefficient)
  EPOCHS   = 8000;      // mini-batch updates
  BATCH    = 64;        // samples per mini-batch
  LR       = 0.01;
  MOMENTUM = 0.9;

// Class centres.
// The two blobs sit SIDE BY SIDE in the upper half-plane (not antipodal): the
// discriminative direction is x, but they SHARE the "up" direction, so the whole
// lower half-plane and the far shell are off BOTH manifolds and provably
// uncertain. (Antipodal blobs would tile the plane into two confident halves.)
function CentreX(c: integer): TNeuralFloat;
begin
  if c = 0 then Result := -2.5 else Result := 2.5;
end;

function CentreY(c: integer): TNeuralFloat;
begin
  Result := 2.5;
end;

// Draw one training sample from blob c (Box-Muller Gaussian noise).
procedure SampleBlob(c: integer; out px, py: TNeuralFloat);
var
  u1, u2, g1, g2: TNeuralFloat;
begin
  u1 := Random; if u1 < 1e-7 then u1 := 1e-7;
  u2 := Random;
  g1 := Sqrt(-2 * Ln(u1)) * Cos(2 * Pi * u2);
  g2 := Sqrt(-2 * Ln(u1)) * Sin(2 * Pi * u2);
  px := CentreX(c) + SPREAD * g1;
  py := CentreY(c) + SPREAD * g2;
end;

// --------------------------------------------------------------------------
// Build the trunk + evidential classification head.
// --------------------------------------------------------------------------
function BuildNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, 2));
  // A SATURATING (tanh) trunk: far out-of-distribution inputs drive the hidden
  // units into their flat regions, so the head sees near-constant features and
  // falls back to its zero-evidence (maximal-uncertainty) prior rather than
  // extrapolating evidence. This makes the uncertainty signal honest on OOD
  // probes.
  Result.AddLayer(TNNetFullConnectLinear.Create(HIDDEN));
  Result.AddLayer(TNNetHyperbolicTangent.Create());
  Result.AddLayer(TNNetFullConnectLinear.Create(HIDDEN));
  Result.AddLayer(TNNetHyperbolicTangent.Create());
  // Emit K raw evidence channels along the DEPTH axis: shape (1,1,K).
  Result.AddLayer(TNNetFullConnectLinear.Create(1, 1, NCLASSES));
  Result.AddLayer(TNNetEvidentialClassification.Create(NCLASSES, LAMBDA));
  Result.SetL2Decay(0.0);
  Result.SetBatchUpdate(True);
end;

// --------------------------------------------------------------------------
// Initialise the head biases so evidence starts LOW (high-uncertainty prior):
// softplus(-3) ~ 0.049 => alpha ~ 1.05 => S ~ 2.1 => u ~ 0.95. Accumulated
// gradient at training points then GROWS evidence locally, leaving OOD probes at
// their low-evidence (high-uncertainty) prior.
// --------------------------------------------------------------------------
procedure InitHead(NN: TNNet);
var
  Head: TNNetLayer;
  c: integer;
begin
  Head := NN.Layers[NN.GetLastLayerIdx - 1];  // the FullConnectLinear head
  for c := 0 to NCLASSES - 1 do
    Head.Neurons[c].BiasWeight := -3.0;
end;

// --------------------------------------------------------------------------
// Train by minimising the EDL Bayes-risk MSE + lambda*KL loss.
// --------------------------------------------------------------------------
procedure Train(NN: TNNet);
var
  ep, b, i, c: integer;
  Inp, Tgt: TNNetVolume;
  px, py, loss: TNeuralFloat;
  EC: TNNetEvidentialClassification;
begin
  Inp := TNNetVolume.Create(1, 1, 2);
  Tgt := TNNetVolume.Create(1, 1, NCLASSES);
  EC := NN.GetLastLayer as TNNetEvidentialClassification;
  NN.SetLearningRate(LR, MOMENTUM);
  for ep := 1 to EPOCHS do
  begin
    NN.ClearDeltas();
    for b := 1 to BATCH do
    begin
      c := Random(NCLASSES);
      SampleBlob(c, px, py);
      Inp.FData[0] := px;
      Inp.FData[1] := py;
      NN.Compute(Inp);
      // Target = one-hot label; the head recovers it from the alpha channels.
      Tgt.Fill(0);
      Tgt.FData[c] := 1;
      NN.Backpropagate(Tgt);   // accumulates the exact EDL gradient (batch mode)
    end;
    // Mean gradient over the mini-batch (batch-update path sums per-sample deltas).
    NN.MulDeltas(1.0 / BATCH);
    NN.UpdateWeights();
    if (ep mod 1000 = 0) or (ep = 1) then
    begin
      loss := 0;
      for i := 1 to 256 do
      begin
        c := Random(NCLASSES);
        SampleBlob(c, px, py);
        Inp.FData[0] := px;
        Inp.FData[1] := py;
        NN.Compute(Inp);
        Tgt.Fill(0);
        Tgt.FData[c] := 1;
        loss := loss + EC.EvidentialLoss([Tgt.FData[0], Tgt.FData[1]]);
      end;
      WriteLn(Format('  epoch %4d   avg EDL loss=%9.4f', [ep, loss / 256]));
    end;
  end;
  Inp.Free; Tgt.Free;
end;

// --------------------------------------------------------------------------
// Probe a single 2-D point and print prediction + uncertainty.
// --------------------------------------------------------------------------
procedure Probe(NN: TNNet; Inp: TNNetVolume; px, py: TNeuralFloat;
  const Tag: string; var sumU: TNeuralFloat; var n: integer);
var
  EC: TNNetEvidentialClassification;
  predC: integer;
  u, p0, p1: TNeuralFloat;
begin
  EC := NN.GetLastLayer as TNNetEvidentialClassification;
  Inp.FData[0] := px;
  Inp.FData[1] := py;
  NN.Compute(Inp);
  p0 := EC.Prediction(0);
  p1 := EC.Prediction(1);
  u := EC.Uncertainty();
  if p1 > p0 then predC := 1 else predC := 0;
  WriteLn(Format('  (%6.2f,%6.2f)   class %d   p=[%.3f %.3f]   u=%.3f   %s',
    [px, py, predC, p0, p1, u, Tag]));
  sumU := sumU + u; Inc(n);
end;

// --------------------------------------------------------------------------
// Report: in-distribution probes (near the blobs) vs OOD probes (far away).
// --------------------------------------------------------------------------
procedure Report(NN: TNNet);
var
  Inp: TNNetVolume;
  sumIn, sumOut: TNeuralFloat;
  nIn, nOut, k: integer;
  ang, rad: TNeuralFloat;
begin
  Inp := TNNetVolume.Create(1, 1, 2);
  sumIn := 0; sumOut := 0; nIn := 0; nOut := 0;
  WriteLn('');
  WriteLn('==== UNCERTAINTY PROBES ====');
  WriteLn('     (x, y)        pred    probabilities      u       region');
  // In-distribution: at / near the two (upper) blob centres.
  Probe(NN, Inp, -2.5,  2.5, 'in-dist (blob 0)', sumIn, nIn);
  Probe(NN, Inp, -2.2,  2.2, 'in-dist (blob 0)', sumIn, nIn);
  Probe(NN, Inp,  2.5,  2.5, 'in-dist (blob 1)', sumIn, nIn);
  Probe(NN, Inp,  2.2,  2.8, 'in-dist (blob 1)', sumIn, nIn);
  // Out-of-distribution: the two blobs share the "up" direction, so the whole
  // LOWER half-plane and the far shell are off both manifolds — the head has no
  // evidence there and provably abstains. Probe directly below each blob, the
  // central gap, and a far lower semicircle.
  Probe(NN, Inp, -2.5, -4.0, 'OOD (below blob 0)', sumOut, nOut);
  Probe(NN, Inp,  2.5, -4.0, 'OOD (below blob 1)', sumOut, nOut);
  Probe(NN, Inp,  0.0, -6.0, 'OOD (far below)',    sumOut, nOut);
  Probe(NN, Inp,  0.0, -9.0, 'OOD (far below)',    sumOut, nOut);
  rad := 9.0;
  for k := 0 to 3 do
  begin
    // a fan across the lower semicircle (-135 deg .. -45 deg).
    ang := -3*Pi/4 + k * (Pi / 6);
    Probe(NN, Inp, rad * Cos(ang), rad * Sin(ang), 'OOD (lower shell)', sumOut, nOut);
  end;

  if (nIn > 0) and (nOut > 0) then
  begin
    sumIn := sumIn / nIn;
    sumOut := sumOut / nOut;
    WriteLn('');
    WriteLn('==== VERDICT ====');
    WriteLn(Format('  mean uncertainty   in-distribution=%.3f   OOD=%.3f',
      [sumIn, sumOut]));
    if (sumIn > 1e-6) and (sumOut > sumIn) then
      WriteLn(Format('  Uncertainty is %.1fx larger on the OOD probes (single ' +
        'deterministic pass, no ensemble).', [sumOut / sumIn]))
    else if sumOut > sumIn then
      WriteLn('  Uncertainty rises out of distribution (single deterministic pass).')
    else
      WriteLn('  (Unexpected: uncertainty did not rise out of distribution in ' +
        'this budget.)');
  end;
  Inp.Free;
end;

var
  Net: TNNet;
begin
  // softplus/log/lgamma in the EDL loss stay finite by construction, but mask
  // FP exceptions defensively like the sibling probabilistic examples.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 424242;

  WriteLn('EvidentialClassification: two 2-D blobs, Dirichlet evidential head');
  WriteLn(Format('  trunk 2->%d->%d->%d raw evidence channels, lambda=%.3f',
    [HIDDEN, HIDDEN, NCLASSES, LAMBDA]));
  WriteLn('');

  Net := BuildNet();
  InitHead(Net);

  WriteLn('Training evidential classification (Bayes-risk MSE + KL)...');
  Train(Net);

  Report(Net);

  Net.Free;
end.
