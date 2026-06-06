program MixtureDensity;
(*
MixtureDensity: a tiny demo of a Mixture Density Network (Bishop 1994,
"Mixture Density Networks", https://publications.aston.ac.uk/id/eprint/373/)
on the CLASSIC one-to-many inverse-mapping problem.

THE PROBLEM. We generate data from the forward map
    x = y + 0.3*sin(2*pi*y) + small noise,   y ~ Uniform(0,1)
and then try to learn the INVERSE: predict y given x. The forward map is
non-monotonic, so for many x values there are SEVERAL valid y branches. A plain
regression head trained with mean-squared error (MSE) is forced to predict a
SINGLE y per x, and the least-squares optimum is the conditional MEAN of those
branches - which falls in the GAP between them, a value the true process never
produces. A Mixture Density Network instead predicts a full conditional
distribution p(y|x) as a mixture of Gaussians, so it can place one component on
each branch and recover the multi-valued structure.

MODELS (both share the same small MLP trunk x -> 32 -> 32):
  MDN : trunk -> TNNetFullConnectLinear(K*(1+2*D)) -> TNNetMixtureDensity(K,D)
        with K mixture components over a D=1 target. The head turns the trunk
        output into (pi, mu, sigma) and OWNS the negative-log-likelihood loss;
        its Backpropagate emits the exact responsibility-weighted dNLL/dparam.
  MSE : trunk -> TNNetFullConnectLinear(1)  (a single-value regression head)
        trained to the same targets with plain squared error.

TRAINING. Manual mini-batch loops (a few thousand updates, well under five
minutes on two CPU cores). For the MDN, the framework seeds the head's error
with (output - target); we build the target volume so its FIRST D channels hold
the true y (the head reads y from there) and the rest match the output (zero
residual). For the MSE head we seed the standard (pred - y) gradient.

REPORT. After training we probe a handful of x values that lie on the
multi-valued part of the curve. For each we print:
  - the MDN's mixture (the dominant component means = the recovered branches),
  - several SAMPLES drawn from the MDN (they land on the separate branches),
  - the single MSE prediction (which sits in the gap between branches),
and we score both heads by how well their predictions/samples match a TRUE y
branch. Expected headline: the MDN recovers the multiple branches; the MSE head
averages them into the gap.

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
  neuralnetwork, neuralvolume, neuralfit;

const
  K_COMP     = 5;        // mixture components
  D_TARGET   = 1;        // target dimensionality (scalar y)
  HEAD_DEPTH = K_COMP * (1 + 2 * D_TARGET);  // 5*(1+2) = 15 raw params
  HIDDEN     = 32;       // trunk hidden width
  NOISE      = 0.03;     // observation noise on x
  EPOCHS     = 8000;     // mini-batch updates
  BATCH      = 64;       // samples per mini-batch
  MDN_LR     = 0.30;     // step on the MEAN mini-batch gradient (see TrainMDN)
  MSE_LR     = 0.05;
  MOMENTUM   = 0.9;

// --------------------------------------------------------------------------
// Forward generative map y -> x (the thing whose inverse is multi-valued).
// --------------------------------------------------------------------------
function ForwardMap(y: TNeuralFloat): TNeuralFloat;
begin
  Result := y + 0.3 * Sin(2 * Pi * y);
end;

// Draw one (x,y) training pair: pick y uniformly, map to x, add noise to x.
procedure SamplePair(out x, y: TNeuralFloat);
begin
  y := Random;                                  // y in [0,1)
  x := ForwardMap(y) + (Random - 0.5) * 2 * NOISE;
end;

// --------------------------------------------------------------------------
// Build the shared trunk + the requested head onto a fresh network.
// HeadDepth = HEAD_DEPTH for the MDN, 1 for the MSE baseline.
// --------------------------------------------------------------------------
function BuildNet(IsMDN: boolean): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, 1));
  Result.AddLayer(TNNetFullConnectReLU.Create(HIDDEN));
  Result.AddLayer(TNNetFullConnectReLU.Create(HIDDEN));
  if IsMDN then
  begin
    // Emit the K*(1+2*D) raw parameters along the DEPTH axis (the axis the
    // mixture head packs over): shape (1,1,HEAD_DEPTH).
    Result.AddLayer(TNNetFullConnectLinear.Create(1, 1, HEAD_DEPTH));
    Result.AddLayer(TNNetMixtureDensity.Create(K_COMP, D_TARGET));
  end
  else
    Result.AddLayer(TNNetFullConnectLinear.Create(1));
  Result.SetL2Decay(0.0);
  Result.SetBatchUpdate(True);
end;

// --------------------------------------------------------------------------
// Break the symmetry of the MDN head. The K mean components must START spread
// across the target range [0,1] or training collapses every component onto the
// conditional mean (the very failure mode an MDN exists to avoid). We set the
// mean-neuron biases to evenly spaced values and the raw-scale biases so the
// initial sigma = softplus(bias) ~ 0.15 (narrow enough to separate branches).
// Neuron layout in the K*(1+2*D) head: [0..K-1] mixing logits,
// [K..K+K*D-1] means, [K+K*D..end] raw scales.
// --------------------------------------------------------------------------
procedure InitMDNHead(NN: TNNet);
var
  Head: TNNetLayer;
  kk, dd, idx, BaseMu, BaseS: integer;
begin
  Head := NN.Layers[NN.GetLastLayerIdx - 1];  // the FullConnectLinear head
  BaseMu := K_COMP;
  BaseS := K_COMP + K_COMP * D_TARGET;
  for kk := 0 to K_COMP - 1 do
  begin
    // Mixing logits start at 0 (uniform pi).
    Head.Neurons[kk].BiasWeight := 0;
    for dd := 0 to D_TARGET - 1 do
    begin
      idx := BaseMu + kk * D_TARGET + dd;
      // Spread means across [0.1, 0.9].
      Head.Neurons[idx].BiasWeight := 0.1 + 0.8 * kk / (K_COMP - 1);
      // softplus(s) = 0.15  =>  s = ln(exp(0.15)-1) ~ -1.84.
      Head.Neurons[BaseS + kk * D_TARGET + dd].BiasWeight := -1.84;
    end;
  end;
end;

// --------------------------------------------------------------------------
// Train the MDN by maximum likelihood. Per mini-batch: forward each sample,
// build a target volume whose first D channels carry the true y (rest = output
// so they contribute zero residual), accumulate gradients, then one update.
// --------------------------------------------------------------------------
procedure TrainMDN(NN: TNNet);
var
  ep, b, i: integer;
  Inp, Tgt: TNNetVolume;
  x, y: TNeuralFloat;
  MD: TNNetMixtureDensity;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Tgt := TNNetVolume.Create(1, 1, HEAD_DEPTH);
  MD := NN.GetLastLayer as TNNetMixtureDensity;
  NN.SetLearningRate(MDN_LR, MOMENTUM);
  for ep := 1 to EPOCHS do
  begin
    NN.ClearDeltas();
    for b := 1 to BATCH do
    begin
      SamplePair(x, y);
      Inp.FData[0] := x;
      NN.Compute(Inp);
      // Target: copy the (transformed) output so every channel has zero
      // residual, then overwrite the first D channels with the true y. The
      // head recovers y as output - (output - target) = target on those.
      Tgt.Copy(MD.Output);
      Tgt.FData[0] := y;
      NN.Backpropagate(Tgt);   // accumulates the exact NLL gradient (batch mode)
    end;
    // Scale accumulated deltas to the MEAN gradient over the mini-batch (the
    // batch-update path sums raw per-sample deltas). A clean mean gradient is
    // what stops one component from collapsing and lets the trunk learn the
    // x-dependence.
    NN.MulDeltas(1.0 / BATCH);
    NN.UpdateWeights();
    if (ep mod 1000 = 0) or (ep = 1) then
    begin
      // Report the average NLL over a fresh small probe batch.
      x := 0;
      for i := 1 to 256 do
      begin
        SamplePair(Inp.FData[0], y);
        NN.Compute(Inp);
        x := x + MD.MixtureNLL([y]);
      end;
      WriteLn(Format('  [MDN] epoch %4d   avg NLL=%8.4f', [ep, x / 256]));
    end;
  end;
  Inp.Free; Tgt.Free;
end;

// --------------------------------------------------------------------------
// Train the MSE baseline with the standard (pred - y) gradient.
// --------------------------------------------------------------------------
procedure TrainMSE(NN: TNNet);
var
  ep, b, i: integer;
  Inp, Tgt: TNNetVolume;
  x, y, mse: TNeuralFloat;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Tgt := TNNetVolume.Create(1, 1, 1);
  NN.SetLearningRate(MSE_LR, MOMENTUM);
  for ep := 1 to EPOCHS do
  begin
    NN.ClearDeltas();
    for b := 1 to BATCH do
    begin
      SamplePair(x, y);
      Inp.FData[0] := x;
      NN.Compute(Inp);
      Tgt.FData[0] := y;     // framework seeds (pred - y) for the linear head
      NN.Backpropagate(Tgt);
    end;
    NN.UpdateWeights();
    if (ep mod 1000 = 0) or (ep = 1) then
    begin
      mse := 0;
      for i := 1 to 256 do
      begin
        SamplePair(Inp.FData[0], y);
        NN.Compute(Inp);
        mse := mse + Sqr(NN.GetLastLayer.Output.FData[0] - y);
      end;
      WriteLn(Format('  [MSE] epoch %4d   avg MSE=%8.6f', [ep, mse / 256]));
    end;
  end;
  Inp.Free; Tgt.Free;
end;

// --------------------------------------------------------------------------
// For a given x, return how many DISTINCT true y branches exist (the y in
// [0,1] with ForwardMap(y) ~ x) and fill Branches with them. Found by a fine
// scan + sign-change bracketing of g(y)=ForwardMap(y)-x.
// --------------------------------------------------------------------------
function TrueBranches(x: TNeuralFloat; var Branches: array of TNeuralFloat): integer;
const
  STEPS = 2000;
var
  i, n: integer;
  y0, y1, g0, g1, ym: TNeuralFloat;
begin
  n := 0;
  y0 := 0; g0 := ForwardMap(y0) - x;
  for i := 1 to STEPS do
  begin
    y1 := i / STEPS;
    g1 := ForwardMap(y1) - x;
    if (g0 = 0) or (g0 * g1 < 0) then
    begin
      // Bisect for the root in [y0,y1].
      while (y1 - y0) > 1e-5 do
      begin
        ym := 0.5 * (y0 + y1);
        if (ForwardMap(y0) - x) * (ForwardMap(ym) - x) <= 0 then y1 := ym
        else y0 := ym;
      end;
      if (n <= High(Branches)) then
      begin
        Branches[n] := 0.5 * (y0 + y1);
        Inc(n);
      end;
    end;
    y0 := y1; g0 := g1;
  end;
  Result := n;
end;

// Nearest distance from value v to any of the n true branches.
function DistToBranch(v: TNeuralFloat; const Branches: array of TNeuralFloat;
  n: integer): TNeuralFloat;
var
  i: integer;
  d: TNeuralFloat;
begin
  Result := 1e30;
  for i := 0 to n - 1 do
  begin
    d := Abs(v - Branches[i]);
    if d < Result then Result := d;
  end;
end;

// --------------------------------------------------------------------------
// Final qualitative + quantitative report on the multi-valued region.
// --------------------------------------------------------------------------
procedure Report(MDNNet, MSENet: TNNet);
var
  MD: TNNetMixtureDensity;
  Inp: TNNetVolume;
  Branches: array[0..7] of TNeuralFloat;
  Probe: array[0..4] of TNeuralFloat;
  Sample: array[0..D_TARGET-1] of TNeuralFloat;
  nb, pi_idx, s, kk, jj, BaseMu, BaseS: integer;
  x, msePred, mseDist, mdnSampDist: TNeuralFloat;
  mdnCover, mseCover, dMu, best: TNeuralFloat;
  sumMSEcov, sumMDNcov: TNeuralFloat;
  nProbe: integer;
const
  N_SAMPLES = 8;
  PI_MIN = 0.05;   // a component counts as "active" above this mixing weight
begin
  MD := MDNNet.GetLastLayer as TNNetMixtureDensity;
  Inp := TNNetVolume.Create(1, 1, 1);
  BaseMu := K_COMP;
  BaseS := K_COMP + K_COMP * D_TARGET;
  // x values that lie squarely on the multi-valued (folded) region.
  Probe[0] := 0.45; Probe[1] := 0.50; Probe[2] := 0.55;
  Probe[3] := 0.40; Probe[4] := 0.60;
  nProbe := 0; sumMSEcov := 0; sumMDNcov := 0;

  WriteLn('');
  WriteLn('==== INVERSE-MAP RECOVERY ON THE MULTI-VALUED REGION ====');
  for pi_idx := 0 to High(Probe) do
  begin
    x := Probe[pi_idx];
    nb := TrueBranches(x, Branches);
    if nb < 2 then Continue;  // only interesting where >1 valid y exists
    Inc(nProbe);

    Inp.FData[0] := x;
    MDNNet.Compute(Inp);
    Write(Format('x=%.3f  true y branches: ', [x]));
    for kk := 0 to nb - 1 do Write(Format('%.3f ', [Branches[kk]]));
    WriteLn('');

    // MDN component means weighted by their mixing weight (show top ones).
    Write('   MDN components (pi: mu):');
    for kk := 0 to K_COMP - 1 do
      Write(Format('  %.2f:%.3f',
        [MD.Output.FData[kk], MD.Output.FData[BaseMu + kk * D_TARGET]]));
    WriteLn('');

    // Draw samples; measure how close they fall to a true branch.
    Write('   MDN samples:');
    mdnSampDist := 0;
    for s := 1 to N_SAMPLES do
    begin
      MD.SampleMixture(Sample, 0, 0);
      Write(Format(' %.3f', [Sample[0]]));
      mdnSampDist := mdnSampDist + DistToBranch(Sample[0], Branches, nb);
    end;
    mdnSampDist := mdnSampDist / N_SAMPLES;
    WriteLn('');

    // MSE single prediction.
    MSENet.Compute(Inp);
    msePred := MSENet.GetLastLayer.Output.FData[0];
    mseDist := DistToBranch(msePred, Branches, nb);

    // BRANCH-COVERAGE error: for every TRUE branch, how far is the nearest
    // model prediction that could explain it?
    //  - MDN: nearest ACTIVE (pi>PI_MIN) component mean (it can cover many).
    //  - MSE: the single prediction (one point cannot cover several branches).
    mdnCover := 0; mseCover := 0;
    for jj := 0 to nb - 1 do
    begin
      best := 1e30;
      for kk := 0 to K_COMP - 1 do
        if MD.Output.FData[kk] > PI_MIN then
        begin
          dMu := Abs(MD.Output.FData[BaseMu + kk * D_TARGET] - Branches[jj]);
          if dMu < best then best := dMu;
        end;
      mdnCover := mdnCover + best;
      mseCover := mseCover + Abs(msePred - Branches[jj]);
    end;
    mdnCover := mdnCover / nb;
    mseCover := mseCover / nb;

    WriteLn(Format('   MSE prediction: %.3f   (dist to nearest branch=%.3f)',
      [msePred, mseDist]));
    WriteLn(Format('   branch-coverage error:  MDN means=%.3f   MSE point=%.3f' +
      '   (avg MDN-sample dist=%.3f)', [mdnCover, mseCover, mdnSampDist]));
    WriteLn('');
    sumMSEcov := sumMSEcov + mseCover;
    sumMDNcov := sumMDNcov + mdnCover;
  end;

  if nProbe > 0 then
  begin
    sumMSEcov := sumMSEcov / nProbe;
    sumMDNcov := sumMDNcov / nProbe;
    WriteLn('==== VERDICT ====');
    WriteLn(Format('  mean branch-coverage error   MDN component means=%.3f   ' +
      'MSE single point=%.3f', [sumMDNcov, sumMSEcov]));
    if sumMDNcov < sumMSEcov then
      WriteLn('  The MDN places a component on EACH branch (low coverage error); ' +
        'the single MSE point cannot, so it collapses to the gap-filling mean.')
    else
      WriteLn('  (Unexpected: MDN did not cover the branches better in this budget.)');
  end;
  Inp.Free;
end;

var
  MDNNet, MSENet: TNNet;
begin
  // softplus/log in the mixture NLL stay finite by construction, but mask FP
  // exceptions defensively like the sibling probabilistic examples.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 424242;

  WriteLn('MixtureDensity: one-to-many inverse map  x = y + 0.3*sin(2*pi*y) + noise');
  WriteLn(Format('  K=%d Gaussian components, D=%d target, trunk %d->%d->%d',
    [K_COMP, D_TARGET, HIDDEN, HIDDEN, HEAD_DEPTH]));
  WriteLn('');

  MDNNet := BuildNet(True);
  InitMDNHead(MDNNet);
  MSENet := BuildNet(False);

  WriteLn('Training mixture density network (maximum likelihood)...');
  TrainMDN(MDNNet);
  WriteLn('Training MSE regression baseline...');
  TrainMSE(MSENet);

  Report(MDNNet, MSENet);

  MDNNet.Free;
  MSENet.Free;
end.
