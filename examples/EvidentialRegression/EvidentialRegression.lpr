program EvidentialRegression;
(*
EvidentialRegression: a tiny demo of Deep Evidential Regression (Amini et al.,
NeurIPS 2020, "Deep Evidential Regression", https://arxiv.org/abs/1910.02600)
on a 1-D function with a HELD-OUT region the model never sees during training.

THE PROBLEM. We learn the scalar function
    f(x) = sin(1.5*x) + 0.2*x        (plus small observation noise on y)
from samples whose inputs x are drawn ONLY from the central band [-3, +3]. The
outer tails x < -3 and x > +3 are HELD OUT: the model never sees data there.
A plain regression head returns a single point estimate everywhere and gives no
hint that the held-out tails are unsupported. Deep Evidential Regression instead
places a Normal-Inverse-Gamma (NIG) prior over the Gaussian likelihood and, in
ONE deterministic forward pass (no sampling, no ensemble), emits its 4
parameters per target:
    gamma (mean),  nu (>0),  alpha (>1),  beta (>0).
From these it reads off, in closed form,
    prediction    = gamma,
    aleatoric var = beta/(alpha-1),                 (irreducible data noise)
    epistemic var = beta/(nu*(alpha-1)).            (model/knowledge uncertainty)
The EPISTEMIC variance is the interesting one: it SPIKES in the held-out tails,
because the evidence parameter nu collapses where the network has accumulated no
evidence, and stays LOW across the densely observed central band. This mirrors
Figure 3 of the paper (cubic regression: epistemic uncertainty explodes outside
the training support).

MODEL. a SATURATING (tanh) trunk x -> 64 -tanh-> 64 -tanh-> so that far
       out-of-distribution inputs drive the hidden units flat and the head falls
       back to its low-evidence bias prior instead of linearly extrapolating the
       NIG params -> TNNetFullConnectLinear(1,1,4) ->
       TNNetEvidentialRegression(D=1, lambda). The head turns the trunk output
       into (gamma, nu, alpha, beta) via softplus links and OWNS the NIG
       negative-log-likelihood + evidence-regularizer loss; its Backpropagate
       emits the exact dL/d(raw param).

TRAINING. A manual mini-batch loop (a few thousand updates, well under three
minutes on two CPU cores). The framework seeds the head's error with
(output - target); we build the target volume so its gamma channel holds the
true y (the head reads y from there) and the rest match the output (zero
residual). The head's nu output starts LOW (a high-uncertainty prior) and only
GROWS where accumulated evidence (training data) supports it, so the held-out
tails keep their low-evidence (high-epistemic) prior.

REPORT. After training we sweep x across the trained band and the held-out tails
and print the epistemic variance, then summarise the mean epistemic variance
IN-DISTRIBUTION (the band) vs HELD-OUT (the tails). Expected headline: the
held-out epistemic uncertainty is several times larger.

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
  HIDDEN   = 64;        // trunk hidden width
  NOISE    = 0.05;      // observation noise on y
  LAMBDA   = 0.01;      // evidence-regularizer weight
  EPOCHS   = 6000;      // mini-batch updates
  BATCH    = 64;        // samples per mini-batch
  LR       = 0.02;
  MOMENTUM = 0.9;
  TRAIN_LO = -3.0;      // training inputs are drawn from [TRAIN_LO, TRAIN_HI]
  TRAIN_HI =  3.0;
  SWEEP_LO = -6.0;      // the report sweeps the wider [SWEEP_LO, SWEEP_HI]
  SWEEP_HI =  6.0;

// --------------------------------------------------------------------------
// The 1-D function we regress.
// --------------------------------------------------------------------------
function TargetFun(x: TNeuralFloat): TNeuralFloat;
begin
  Result := Sin(1.5 * x) + 0.2 * x;
end;

// Draw one training input from the central band [TRAIN_LO, TRAIN_HI].
function SampleTrainX(): TNeuralFloat;
begin
  Result := TRAIN_LO + Random * (TRAIN_HI - TRAIN_LO);
end;

// --------------------------------------------------------------------------
// Build the trunk + evidential head.
// --------------------------------------------------------------------------
function BuildNet(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, 1));
  // A SATURATING (tanh) trunk: far out-of-distribution inputs drive the hidden
  // units into their flat regions, so the head sees near-constant features and
  // falls back to its low-evidence (high-epistemic) bias prior rather than
  // linearly extrapolating the NIG params. This makes the epistemic signal
  // honest in the held-out tails.
  Result.AddLayer(TNNetFullConnectLinear.Create(HIDDEN));
  Result.AddLayer(TNNetHyperbolicTangent.Create());
  Result.AddLayer(TNNetFullConnectLinear.Create(HIDDEN));
  Result.AddLayer(TNNetHyperbolicTangent.Create());
  // Emit the 4 raw NIG parameters along the DEPTH axis: shape (1,1,4).
  Result.AddLayer(TNNetFullConnectLinear.Create(1, 1, 4));
  Result.AddLayer(TNNetEvidentialRegression.Create(1, LAMBDA));
  Result.SetL2Decay(0.0);
  Result.SetBatchUpdate(True);
end;

// --------------------------------------------------------------------------
// Initialise the head biases. Channel layout per target d:
// [4d+0]=gamma, [4d+1]=raw nu, [4d+2]=raw alpha, [4d+3]=raw beta.
// nu starts LOW (high-uncertainty prior); accumulated gradient at training
// points then GROWS nu locally, leaving the held-out tails at their low prior.
// --------------------------------------------------------------------------
procedure InitHead(NN: TNNet);
var
  Head: TNNetLayer;
begin
  Head := NN.Layers[NN.GetLastLayerIdx - 1];  // the FullConnectLinear head
  Head.Neurons[0].BiasWeight := 0.0;     // gamma
  // softplus(-3)~0.049 => nu ~ 0.05 (low-evidence prior).
  Head.Neurons[1].BiasWeight := -3.0;    // nu
  // softplus(s)=1 => s=ln(e-1)~0.5413.
  Head.Neurons[2].BiasWeight := 0.5413;  // alpha -> 1+1 = 2
  Head.Neurons[3].BiasWeight := 0.5413;  // beta  -> ~1
end;

// --------------------------------------------------------------------------
// Train by minimising the NIG NLL + evidence regularizer.
// --------------------------------------------------------------------------
procedure Train(NN: TNNet);
var
  ep, b, i: integer;
  Inp, Tgt: TNNetVolume;
  x, y, loss: TNeuralFloat;
  EV: TNNetEvidentialRegression;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Tgt := TNNetVolume.Create(1, 1, 4);
  EV := NN.GetLastLayer as TNNetEvidentialRegression;
  NN.SetLearningRate(LR, MOMENTUM);
  for ep := 1 to EPOCHS do
  begin
    NN.ClearDeltas();
    for b := 1 to BATCH do
    begin
      x := SampleTrainX();
      y := TargetFun(x) + (Random - 0.5) * 2 * NOISE;
      Inp.FData[0] := x;
      NN.Compute(Inp);
      // Target: copy the (transformed) output so every channel has zero
      // residual, then overwrite the gamma channel with the true y. The head
      // recovers y as output - (output - target) = target on that channel.
      Tgt.Copy(EV.Output);
      Tgt.FData[0] := y;
      NN.Backpropagate(Tgt);   // accumulates the exact NIG gradient (batch mode)
    end;
    // Mean gradient over the mini-batch (batch-update path sums per-sample
    // deltas).
    NN.MulDeltas(1.0 / BATCH);
    NN.UpdateWeights();
    if (ep mod 1000 = 0) or (ep = 1) then
    begin
      loss := 0;
      for i := 1 to 256 do
      begin
        x := SampleTrainX();
        y := TargetFun(x);
        Inp.FData[0] := x;
        NN.Compute(Inp);
        loss := loss + EV.EvidentialLoss([y]);
      end;
      WriteLn(Format('  epoch %4d   avg NIG loss=%9.4f', [ep, loss / 256]));
    end;
  end;
  Inp.Free; Tgt.Free;
end;

// --------------------------------------------------------------------------
// Sweep x and report the epistemic variance, then summarise in-distribution
// (the training band) vs held-out (the tails).
// --------------------------------------------------------------------------
procedure Report(NN: TNNet);
var
  EV: TNNetEvidentialRegression;
  Inp: TNNetVolume;
  i, nIn, nOut: integer;
  x, epi, sumIn, sumOut: TNeuralFloat;
  HeldOut: boolean;
const
  STEPS = 60;
begin
  EV := NN.GetLastLayer as TNNetEvidentialRegression;
  Inp := TNNetVolume.Create(1, 1, 1);
  WriteLn('');
  WriteLn(Format('==== EPISTEMIC UNCERTAINTY SWEEP (trained on [%.1f, %.1f]) ====',
    [TRAIN_LO, TRAIN_HI]));
  WriteLn('     x        pred      true      aleatoric   epistemic   region');
  sumIn := 0; sumOut := 0; nIn := 0; nOut := 0;
  for i := 0 to STEPS do
  begin
    x := SWEEP_LO + (SWEEP_HI - SWEEP_LO) * i / STEPS;
    Inp.FData[0] := x;
    NN.Compute(Inp);
    epi := EV.EpistemicVar(0);
    HeldOut := (x < TRAIN_LO) or (x > TRAIN_HI);
    if HeldOut then
    begin
      sumOut := sumOut + epi; Inc(nOut);
      WriteLn(Format('  %7.3f  %8.3f  %8.3f  %10.4f  %10.4f   HELD-OUT',
        [x, EV.Prediction(0), TargetFun(x), EV.AleatoricVar(0), epi]));
    end
    else
    begin
      sumIn := sumIn + epi; Inc(nIn);
      WriteLn(Format('  %7.3f  %8.3f  %8.3f  %10.4f  %10.4f   train',
        [x, EV.Prediction(0), TargetFun(x), EV.AleatoricVar(0), epi]));
    end;
  end;
  if (nIn > 0) and (nOut > 0) then
  begin
    sumIn := sumIn / nIn;
    sumOut := sumOut / nOut;
    WriteLn('');
    WriteLn('==== VERDICT ====');
    WriteLn(Format('  mean epistemic var   in-distribution=%.4f   held-out=%.4f',
      [sumIn, sumOut]));
    if sumOut > sumIn then
      WriteLn(Format('  Epistemic uncertainty is %.1fx larger in the held-out ' +
        'tails (single deterministic pass, no ensemble).', [sumOut / sumIn]))
    else
      WriteLn('  (Unexpected: epistemic uncertainty did not rise out of ' +
        'distribution in this budget.)');
  end;
  Inp.Free;
end;

var
  Net: TNNet;
begin
  // softplus/log/lgamma in the NIG loss stay finite by construction, but mask
  // FP exceptions defensively like the sibling probabilistic examples.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 424242;

  WriteLn('EvidentialRegression: f(x) = sin(1.5x) + 0.2x, trained on a central band');
  WriteLn(Format('  trunk %d->%d->4 raw NIG params, lambda=%.3f', [HIDDEN, HIDDEN, LAMBDA]));
  WriteLn('');

  Net := BuildNet();
  InitHead(Net);

  WriteLn('Training deep evidential regression (NIG NLL + evidence reg)...');
  Train(Net);

  Report(Net);

  Net.Free;
end.
