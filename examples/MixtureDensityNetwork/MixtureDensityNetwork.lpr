program MixtureDensityNetwork;
(*
MixtureDensityNetwork: a self-contained, pure-CPU reproduction of Bishop's
Mixture Density Network (Bishop 1994, "Mixture Density Networks", NCRG/4288) on
the classic INVERSE-problem toy.

THE PHENOMENON
  Generate FORWARD data  x = y + 0.3*sin(2*pi*y) + noise  with y ~ U[0,1], then
  ask a network to predict y FROM x (the inverse map). Over the fold region a
  single x has up to THREE valid y values, so the conditional p(y|x) is
  genuinely MULTIMODAL. A plain MSE regressor minimises E[(y-f(x))^2] and is
  therefore driven to the conditional MEAN E[y|x]; in the fold region the mean
  sits in the LOW-density GAP between the modes -- a value the data almost never
  takes. That is the textbook failure MDNs exist to fix.

THE FIX (Bishop 1994)
  Model the conditional as a Gaussian MIXTURE whose parameters are functions of
  x emitted by the net:
      p(y|x) = sum_k pi_k(x) * Normal(y ; mu_k(x), sigma_k(x)^2)
  The net emits K*(1+2*D) RAW outputs along the DEPTH axis, which the LIBRARY
  layer TNNetMixtureDensity(K, D) turns into the mixture parameters in place:
      pi_k    = softmax(logit)_k      (mixing coefficients, sum to 1)
      mu_k    = m_k                   (component means, linear)
      sigma_k = softplus(s_k)         (positive component widths)
  trained on the mixture NEGATIVE LOG-LIKELIHOOD
      NLL = -log( sum_k pi_k * Normal(y ; mu_k, sigma_k) ).

LIBRARY HEAD (TNNetMixtureDensity) + ITS OWN log-sum-exp NLL BACKWARD
  The MDN head is the real library layer, NOT a hand-coded reimplementation.
  TNNetMixtureDensity packs the K*(1+2*D) channels over the DEPTH axis as
      [ K mixing logits | K*D means | K*D raw scales ]
  (for D=1: [a_0..a_{K-1}, m_0..m_{K-1}, s_0..s_{K-1}]), applies softmax to the
  logits and softplus to the scales in Compute(), and owns the EXACT
  responsibility-weighted dNLL/dparam in Backpropagate() (stable log-sum-exp).

  We drive that backward the way the layer expects: the framework seeds the
  head's error as (output - target), so we build a target volume whose FIRST D
  channels carry the true y (the head recovers y from output - error there) and
  whose remaining channels equal the output (zero residual). We run the
  mini-batch loop in BATCH-UPDATE mode (NN.SetBatchUpdate(True) -> Backpropagate
  ACCUMULATES into Neurons[].Delta; UpdateWeights applies once), scaling the
  accumulated deltas by 1/batch so the applied step is the MEAN gradient. We
  never call TNeuralFit.Fit, so layer references never go stale. The MSE arm uses
  the stock (out - y) gradient on its single linear output.

BUILT-IN CORRECTNESS INVARIANTS (HALT(1) on violation)
  (1) K=1 REDUCTION: a K=1 MDN's NLL is a homoscedastic Gaussian NLL whose only
      mean is mu_0; its mu must match the independently-trained MSE arm's
      prediction (both recover E[y|x]). We assert the mean abs difference between
      the K=1 mu and the MSE prediction over a probe grid is small.
  (2) SIMPLEX: the mixture weights pi_k(x) must sum to 1 (within 1e-5) at every
      probe x (softmax sanity).

DISTINCT FROM other in-tree uncertainty work -- see README:
  - examples/MCDropoutUncertainty/ models EPISTEMIC uncertainty by sampling
    dropout masks; it does NOT model multimodal targets.
  - the pointwise regression loss heads (Huber/Charbonnier/LogCosh/Wing) are all
    UNIMODAL -- they reshape the residual penalty but still collapse to one
    prediction per x. MDN is the only in-tree thing that models ALEATORIC
    MULTIMODALity (several valid y per x).

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

// D = target dimensionality (scalar y). The library head packs K*(1+2*D)
// channels over the depth axis; with D=1 that is the familiar 3*K.
const
  cD = 1;

const
  cSeed       = 424242;
  cNumPts     = 600;          // training points (tiny)
  cNoise      = 0.05;         // additive noise on the forward map
  cHiddenW    = 32;           // MLP hidden width
  cK          = 3;            // mixture components (headline arm)
  cBatchSize  = 60;
  cEpochs     = 600;
  cLearnRate  = 0.02;
  cPi2        = 6.2831853071795864769;  // 2*pi
  cNLLEps     = 1e-12;

type
  TFloatArr = array of TNeuralFloat;
  TMixParams = record
    Pi:    array of TNeuralFloat;   // length K
    Mu:    array of TNeuralFloat;
    Sigma: array of TNeuralFloat;
  end;

var
  DataX, DataY: TFloatArr;          // training set (x -> predict y)

// ---------------------------------------------------------------------------
// Forward map x = y + 0.3*sin(2*pi*y) + noise, with y ~ U[0,1]. We then learn
// the INVERSE y|x. The middle band of x is reached by three different y, so the
// inverse is one-to-many there.
// ---------------------------------------------------------------------------
procedure MakeData;
var
  I: integer;
  y: TNeuralFloat;
begin
  SetLength(DataX, cNumPts);
  SetLength(DataY, cNumPts);
  for I := 0 to cNumPts - 1 do
  begin
    y := Random;                                  // U[0,1]
    DataX[I] := y + 0.3 * Sin(cPi2 * y) + cNoise * RandG(0, 1);
    DataY[I] := y;
  end;
end;

// ---------------------------------------------------------------------------
// Decode the TNNetMixtureDensity head OUTPUT (the layer already applied softmax
// to the logits and softplus to the scales in Compute()) into (pi, mu, sigma).
// Depth-axis packing the layer uses: [ pi_0..pi_{K-1} | mu_0.. | sigma_0.. ],
// i.e. for D=1: [pi_0..pi_{K-1}, mu_0..mu_{K-1}, sigma_0..sigma_{K-1}].
// ---------------------------------------------------------------------------
procedure DecodeMix(const Outp: TNNetVolume; K: integer; out P: TMixParams);
var
  k1, BaseMu, BaseS: integer;
begin
  SetLength(P.Pi, K); SetLength(P.Mu, K); SetLength(P.Sigma, K);
  BaseMu := K;                 // K + k*D with D=1
  BaseS  := K + K * cD;        // start of the K*D scale block
  for k1 := 0 to K - 1 do
  begin
    P.Pi[k1]    := Outp.FData[k1];
    P.Mu[k1]    := Outp.FData[BaseMu + k1 * cD];
    P.Sigma[k1] := Outp.FData[BaseS + k1 * cD];
  end;
end;

// Single-component Gaussian density Normal(y; mu, sigma).
function GaussPdf(y, mu, sigma: TNeuralFloat): TNeuralFloat;
var
  d: TNeuralFloat;
begin
  d := y - mu;
  Result := Exp(-(d * d) / (2 * sigma * sigma)) / (sigma * Sqrt(cPi2));
end;

// Mixture NLL for a decoded triple at target y.
function MixtureNLL(const P: TMixParams; K: integer; y: TNeuralFloat): TNeuralFloat;
var
  k1: integer;
  S: TNeuralFloat;
begin
  S := 0;
  for k1 := 0 to K - 1 do
    S := S + P.Pi[k1] * GaussPdf(y, P.Mu[k1], P.Sigma[k1]);
  if S < cNLLEps then S := cNLLEps;
  Result := -Ln(S);
end;

// ---------------------------------------------------------------------------
// MLP trunk + head.
//   MSE arm (K=0): Input(1) -> Tanh(H) -> Tanh(H) -> Linear(1)
//   MDN arm (K>0): trunk -> Linear(K*(1+2*D)) -> TNNetMixtureDensity(K, D)
// The Linear layer emits the K*(1+2*D) RAW parameters along the DEPTH axis (the
// axis the mixture head packs over); the library head transforms them in place.
// Tanh hidden layers keep forward/backward RNG-free and give smooth fits.
// ---------------------------------------------------------------------------
procedure BuildNet(out NN: TNNet; K: integer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, 1));
  NN.AddLayer(TNNetFullConnect.Create(cHiddenW));        // Tanh
  NN.AddLayer(TNNetFullConnect.Create(cHiddenW));        // Tanh
  if K > 0 then
  begin
    // Raw mixture parameters on the depth axis: shape (1,1,K*(1+2*D)).
    NN.AddLayer(TNNetFullConnectLinear.Create(1, 1, K * (1 + 2 * cD)));
    NN.AddLayer(TNNetMixtureDensity.Create(K, cD));      // library head + NLL
  end
  else
    NN.AddLayer(TNNetFullConnectLinear.Create(1));       // MSE: single output
  NN.SetLearningRate(cLearnRate, {Momentum=}0.0);
  NN.SetL2Decay(0.0);
  // Batch-update: Backpropagate accumulates into Neurons[].Delta; UpdateWeights
  // applies once per mini-batch. We scale the accumulated deltas to the MEAN.
  NN.SetBatchUpdate(True);
end;

// ---------------------------------------------------------------------------
// Break the symmetry of the MDN head. The K mean components must START spread
// across the target range [0,1] or training collapses every component onto the
// conditional mean (the very failure mode an MDN exists to avoid). We set the
// mean-neuron biases to evenly spaced values and the raw-scale biases so the
// initial sigma = softplus(bias) ~ 0.15 (narrow enough to separate branches).
// The Linear head neuron layout matches the layer's K*(1+2*D) depth packing:
// [0..K-1] mixing logits, [K..K+K*D-1] means, [K+K*D..end] raw scales.
// ---------------------------------------------------------------------------
procedure InitMDNHead(NN: TNNet; K: integer);
var
  Head: TNNetLayer;
  kk, dd, idx, BaseMu, BaseS: integer;
  spread: TNeuralFloat;
begin
  Head := NN.Layers[NN.GetLastLayerIdx - 1];  // the FullConnectLinear head
  BaseMu := K;
  BaseS := K + K * cD;
  for kk := 0 to K - 1 do
  begin
    // Mixing logits start at 0 (uniform pi).
    Head.Neurons[kk].BiasWeight := 0;
    if K > 1 then spread := 0.1 + 0.8 * kk / (K - 1) else spread := 0.5;
    for dd := 0 to cD - 1 do
    begin
      idx := BaseMu + kk * cD + dd;
      Head.Neurons[idx].BiasWeight := spread;          // means spread over [0.1,0.9]
      // softplus(s) = 0.15  =>  s = ln(exp(0.15)-1) ~ -1.84.
      Head.Neurons[BaseS + kk * cD + dd].BiasWeight := -1.84;
    end;
  end;
end;

// ---------------------------------------------------------------------------
// Train the MSE regressor (single linear output). Pseudo-target trick:
// g = (out - y)/batch  ->  pseudo = out - g.
// ---------------------------------------------------------------------------
procedure TrainMSE(NN: TNNet);
var
  Order: array of integer;
  Epoch, Lo, Hi, I, J, Tmp: integer;
  Inp, Pseudo, Outp: TNNetVolume;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Pseudo := TNNetVolume.Create(1, 1, 1);
  SetLength(Order, cNumPts);
  for I := 0 to cNumPts - 1 do Order[I] := I;
  for Epoch := 1 to cEpochs do
  begin
    for I := High(Order) downto 1 do
    begin
      J := Random(I + 1); Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    Lo := 0;
    while Lo < cNumPts do
    begin
      Hi := Lo + cBatchSize; if Hi > cNumPts then Hi := cNumPts;
      NN.ClearDeltas();
      for I := Lo to Hi - 1 do
      begin
        Inp.FData[0] := DataX[Order[I]];
        NN.Compute(Inp);
        Outp := NN.GetLastLayer().Output;
        Pseudo.FData[0] := Outp.FData[0]
          - (Outp.FData[0] - DataY[Order[I]]) / (Hi - Lo);
        NN.Backpropagate(Pseudo);
      end;
      NN.UpdateWeights();
      Lo := Hi;
    end;
  end;
  Pseudo.Free; Inp.Free;
end;

// ---------------------------------------------------------------------------
// Train an MDN with K components on the mixture NLL using the LIBRARY layer's
// own log-sum-exp NLL backward (TNNetMixtureDensity.Backpropagate).
//
// The framework seeds the head error as (output - target). The library head
// recovers the regression target y from the FIRST D channels as
// output - (output - target). So we build a target volume that copies the head
// output (every channel zero residual) and overwrites the first D channels with
// the true y. The head then emits the exact responsibility-weighted dNLL/dparam.
// ---------------------------------------------------------------------------
procedure TrainMDN(NN: TNNet; K: integer);
var
  Order: array of integer;
  Epoch, Lo, Hi, I, J, Tmp, HeadDepth: integer;
  Inp, Tgt, Outp: TNNetVolume;
begin
  HeadDepth := K * (1 + 2 * cD);
  Inp := TNNetVolume.Create(1, 1, 1);
  Tgt := TNNetVolume.Create(1, 1, HeadDepth);
  SetLength(Order, cNumPts);
  for I := 0 to cNumPts - 1 do Order[I] := I;
  for Epoch := 1 to cEpochs do
  begin
    for I := High(Order) downto 1 do
    begin
      J := Random(I + 1); Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    Lo := 0;
    while Lo < cNumPts do
    begin
      Hi := Lo + cBatchSize; if Hi > cNumPts then Hi := cNumPts;
      NN.ClearDeltas();
      for I := Lo to Hi - 1 do
      begin
        Inp.FData[0] := DataX[Order[I]];
        NN.Compute(Inp);
        Outp := NN.GetLastLayer().Output;
        // Copy the (transformed) output so every channel has zero residual,
        // then overwrite the first D channels with the true y; the head reads
        // y from there and produces the exact NLL gradient (batch mode).
        Tgt.Copy(Outp);
        Tgt.FData[0] := DataY[Order[I]];
        NN.Backpropagate(Tgt);
      end;
      // Scale accumulated deltas to the MEAN gradient over the mini-batch.
      NN.MulDeltas(1.0 / (Hi - Lo));
      NN.UpdateWeights();
      Lo := Hi;
    end;
  end;
  Tgt.Free; Inp.Free;
end;

// Forward a single x through an MDN, returning decoded params.
procedure MDNForward(NN: TNNet; K: integer; x: TNeuralFloat; out P: TMixParams);
var
  Inp: TNNetVolume;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Inp.FData[0] := x;
  NN.Compute(Inp);
  DecodeMix(NN.GetLastLayer().Output, K, P);
  Inp.Free;
end;

// Forward a single x through the MSE regressor.
function MSEForward(NN: TNNet; x: TNeuralFloat): TNeuralFloat;
var
  Inp: TNNetVolume;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Inp.FData[0] := x;
  NN.Compute(Inp);
  Result := NN.GetLastLayer().Output.FData[0];
  Inp.Free;
end;

// Sample one y from an MDN's mixture at x (pick a component by pi, then Gaussian).
function MDNSample(const P: TMixParams; K: integer): TNeuralFloat;
var
  k1: integer;
  r, acc: TNeuralFloat;
begin
  r := Random; acc := 0; k1 := 0;
  while (k1 < K - 1) do
  begin
    acc := acc + P.Pi[k1];
    if r <= acc then Break;
    Inc(k1);
  end;
  Result := P.Mu[k1] + P.Sigma[k1] * RandG(0, 1);
end;

// ---------------------------------------------------------------------------
// Dataset mean NLL / MSE for each arm (over the training set).
// ---------------------------------------------------------------------------
function MDNMeanNLL(NN: TNNet; K: integer): TNeuralFloat;
var
  I: integer;
  S: TNeuralFloat;
  P: TMixParams;
begin
  S := 0;
  for I := 0 to cNumPts - 1 do
  begin
    MDNForward(NN, K, DataX[I], P);
    S := S + MixtureNLL(P, K, DataY[I]);
  end;
  Result := S / cNumPts;
end;

// Mean point-prediction MSE of an MDN, using the most-likely component's mean
// (argmax pi) as the point estimate.
function MDNMeanMSE(NN: TNNet; K: integer): TNeuralFloat;
var
  I, k1, kbest: integer;
  S, best, pred: TNeuralFloat;
  P: TMixParams;
begin
  S := 0;
  for I := 0 to cNumPts - 1 do
  begin
    MDNForward(NN, K, DataX[I], P);
    kbest := 0; best := P.Pi[0];
    for k1 := 1 to K - 1 do if P.Pi[k1] > best then begin best := P.Pi[k1]; kbest := k1; end;
    pred := P.Mu[kbest];
    S := S + Sqr(pred - DataY[I]);
  end;
  Result := S / cNumPts;
end;

function MSEMeanMSE(NN: TNNet): TNeuralFloat;
var
  I: integer;
  S: TNeuralFloat;
begin
  S := 0;
  for I := 0 to cNumPts - 1 do S := S + Sqr(MSEForward(NN, DataX[I]) - DataY[I]);
  Result := S / cNumPts;
end;

// Mean Gaussian NLL of the MSE arm, treating its single prediction as the mean
// of a homoscedastic Gaussian with the dataset-fit sigma (for an apples-to-
// apples NLL comparison against the MDN). sigma = sqrt(MSE).
function MSEMeanNLL(NN: TNNet; sigma: TNeuralFloat): TNeuralFloat;
var
  I: integer;
  S: TNeuralFloat;
begin
  S := 0;
  for I := 0 to cNumPts - 1 do
    S := S - Ln(GaussPdf(DataY[I], MSEForward(NN, DataX[I]), sigma) + cNLLEps + 0);
  Result := S / cNumPts;
end;

function FmtF(V: TNeuralFloat; W, D: integer): string;
begin
  if IsNan(V) or IsInfinite(V) then Result := Format('%*s', [W, 'NaN/Inf'])
  else Result := Format('%*.*f', [W, D, V]);
end;

// ASCII scatter: x on the horizontal axis, y on the vertical. Plots the TRUE
// data as '.', the MSE prediction as 'M', and MDN samples as 'o'. Shows the MDN
// recovering all three branches where the MSE arm cannot.
procedure DrawScatter(NNmse, NNmdn: TNNet; K: integer);
const
  W = 64; Hh = 24; NSamp = 6;
var
  grid: array of array of char;
  ix, iy, I, s: integer;
  xMin, xMax, x, ys, pred: TNeuralFloat;
  P: TMixParams;

  procedure Plot(xx, yy: TNeuralFloat; ch: char);
  var gx, gy: integer;
  begin
    if (yy < 0) or (yy > 1) then Exit;
    gx := Round((xx - xMin) / (xMax - xMin) * (W - 1));
    gy := Round((1 - yy) * (Hh - 1));      // y up
    if (gx < 0) or (gx >= W) or (gy < 0) or (gy >= Hh) then Exit;
    if grid[gy][gx] = ' ' then grid[gy][gx] := ch;
  end;

begin
  xMin := 1e30; xMax := -1e30;
  for I := 0 to cNumPts - 1 do
  begin
    if DataX[I] < xMin then xMin := DataX[I];
    if DataX[I] > xMax then xMax := DataX[I];
  end;
  SetLength(grid, Hh);
  for iy := 0 to Hh - 1 do
  begin
    SetLength(grid[iy], W);
    for ix := 0 to W - 1 do grid[iy][ix] := ' ';
  end;
  // true data points
  for I := 0 to cNumPts - 1 do Plot(DataX[I], DataY[I], '.');
  // MDN samples + MSE prediction along a sweep of x
  I := 0;
  while I < W do
  begin
    x := xMin + (xMax - xMin) * I / (W - 1);
    MDNForward(NNmdn, K, x, P);
    for s := 1 to NSamp do
    begin
      ys := MDNSample(P, K);
      Plot(x, ys, 'o');
    end;
    pred := MSEForward(NNmse, x);
    Plot(x, pred, 'M');
    Inc(I);
  end;
  WriteLn('  Legend: . = true data,  o = MDN samples,  M = MSE prediction.');
  WriteLn('  (y axis 0..1 top-to-bottom; x axis = observed x left-to-right)');
  for iy := 0 to Hh - 1 do
  begin
    Write('  |');
    for ix := 0 to W - 1 do Write(grid[iy][ix]);
    WriteLn('|');
  end;
end;

// ===========================================================================
var
  NNmse, NNmdn, NNmdn1: TNNet;
  StartT: TDateTime;
  ProbeX: array[0..4] of TNeuralFloat = (0.30, 0.45, 0.50, 0.55, 0.70);
  I, k1: integer;
  P, P1: TMixParams;
  PiSum, MaxSimplexErr, MeanK1Diff, msePred, mseMSE, mseSigma: TNeuralFloat;
  NLLmdn, MSEmdn, NLLmse, NLLmdn1: TNeuralFloat;
  foldPred, foldLowX, foldHighX, foldGapWidth: TNeuralFloat;
  inv1ok, inv2ok: boolean;

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  DefaultFormatSettings.DecimalSeparator := '.';
  StartT := Now;

  WriteLn('========================================================================');
  WriteLn('Mixture Density Network (Bishop 1994) on the inverse-problem toy');
  WriteLn('========================================================================');
  WriteLn(Format('Forward map: x = y + 0.3*sin(2*pi*y) + N(0,%.2f^2), y~U[0,1]; learn y|x.',
    [cNoise]));
  WriteLn(Format('Net: 1 -> %d(Tanh) -> %d(Tanh) -> Linear(K*(1+2D)) -> TNNetMixtureDensity. seed=%d.',
    [cHiddenW, cHiddenW, cSeed]));
  WriteLn(Format('MDN head: K=%d components, D=%d (%d raw params); train=%d, batch=%d, epochs=%d, lr=%.3f.',
    [cK, cD, cK * (1 + 2 * cD), cNumPts, cBatchSize, cEpochs, cLearnRate]));
  WriteLn;

  // Build the shared dataset once (after seeding) so all arms see identical x,y.
  RandSeed := cSeed;
  MakeData;

  // ----------------------- Arm A: plain MSE regressor ---------------------
  RandSeed := cSeed;       // identical seed/arch before each arm
  BuildNet(NNmse, 0);      // K=0 -> single linear output (MSE arm)
  TrainMSE(NNmse);
  mseMSE := MSEMeanMSE(NNmse);
  mseSigma := Sqrt(mseMSE);
  NLLmse := MSEMeanNLL(NNmse, mseSigma);

  // ----------------------- Arm B: K=3 MDN ---------------------------------
  RandSeed := cSeed;
  BuildNet(NNmdn, cK);
  InitMDNHead(NNmdn, cK);
  TrainMDN(NNmdn, cK);
  NLLmdn := MDNMeanNLL(NNmdn, cK);
  MSEmdn := MDNMeanMSE(NNmdn, cK);

  // ----------------------- Arm C: K=1 MDN (for invariant 1) ---------------
  RandSeed := cSeed;
  BuildNet(NNmdn1, 1);
  InitMDNHead(NNmdn1, 1);
  TrainMDN(NNmdn1, 1);
  NLLmdn1 := MDNMeanNLL(NNmdn1, 1);

  // ----------------------- Per-arm headline numbers -----------------------
  WriteLn('------------------------------------------------------------------------');
  WriteLn('Per-arm scores (over the training set):');
  WriteLn('------------------------------------------------------------------------');
  WriteLn('  arm                mean NLL     point MSE');
  WriteLn(Format('  MSE regressor    %s    %s   (NLL uses homoscedastic sigma=%.3f)',
    [FmtF(NLLmse, 9, 4), FmtF(mseMSE, 9, 5), mseSigma]));
  WriteLn(Format('  MDN  K=%d         %s    %s   (point MSE = argmax-pi mean)',
    [cK, FmtF(NLLmdn, 9, 4), FmtF(MSEmdn, 9, 5)]));
  WriteLn(Format('  MDN  K=1         %s        -        (homoscedastic; see invariant 1)',
    [FmtF(NLLmdn1, 9, 4)]));
  if NLLmdn < NLLmse then
    WriteLn('  => MDN K=3 achieves LOWER NLL than the MSE arm (it models the multimodality).')
  else
    WriteLn('  => NOTE: MDN K=3 NLL did not beat the MSE arm on this run.');
  WriteLn;

  // ----------------------- Fold-region collapse evidence ------------------
  // At x ~ 0.5 (the fold), the true y has three valid branches near ~0.13, 0.5,
  // 0.87. The MSE arm must sit near the conditional mean ~0.5 (the LOW-density
  // GAP), while the MDN keeps three modes. We probe x=0.5.
  MDNForward(NNmdn, cK, 0.5, P);
  msePred := MSEForward(NNmse, 0.5);
  // Find the MDN's lowest and highest component means at x=0.5 (the outer
  // branches) to show the MSE prediction lands BETWEEN them.
  foldLowX := P.Mu[0]; foldHighX := P.Mu[0];
  for k1 := 1 to cK - 1 do
  begin
    if P.Mu[k1] < foldLowX then foldLowX := P.Mu[k1];
    if P.Mu[k1] > foldHighX then foldHighX := P.Mu[k1];
  end;
  foldGapWidth := foldHighX - foldLowX;
  foldPred := msePred;
  WriteLn('------------------------------------------------------------------------');
  WriteLn('Fold-region collapse (probe x = 0.50, where y is genuinely 3-valued):');
  WriteLn('------------------------------------------------------------------------');
  WriteLn(Format('  MSE prediction at x=0.50            = %s', [FmtF(msePred, 7, 4)]));
  WriteLn(Format('  MDN outer component means at x=0.50 = [%s .. %s]  (spread %s)',
    [FmtF(foldLowX, 6, 4), FmtF(foldHighX, 6, 4), FmtF(foldGapWidth, 6, 4)]));
  if (foldPred > foldLowX + 0.10) and (foldPred < foldHighX - 0.10) then
    WriteLn('  => MSE prediction sits in the LOW-DENSITY GAP between the outer MDN modes')
  else
    WriteLn('  => (MSE prediction not strictly between the outer modes on this run)');
  WriteLn('     -- the textbook MSE-collapses-to-the-mean failure that MDN fixes.');
  WriteLn;

  // ----------------------- Learned (pi, mu, sigma) at probes --------------
  WriteLn('------------------------------------------------------------------------');
  WriteLn(Format('Learned MDN mixture at probe x (K=%d):', [cK]));
  WriteLn('------------------------------------------------------------------------');
  for I := 0 to High(ProbeX) do
  begin
    MDNForward(NNmdn, cK, ProbeX[I], P);
    Write(Format('  x=%.2f : ', [ProbeX[I]]));
    for k1 := 0 to cK - 1 do
      Write(Format('(pi=%.3f mu=%.3f sg=%.3f) ', [P.Pi[k1], P.Mu[k1], P.Sigma[k1]]));
    WriteLn;
  end;
  WriteLn;

  // ----------------------- ASCII scatter ----------------------------------
  WriteLn('------------------------------------------------------------------------');
  WriteLn('Sampled MDN predictions vs ground truth (all three branches should appear):');
  WriteLn('------------------------------------------------------------------------');
  RandSeed := cSeed;        // deterministic sampling for the chart
  DrawScatter(NNmse, NNmdn, cK);
  WriteLn;

  // ======================= INVARIANTS (HALT on fail) ======================
  WriteLn('========================================================================');
  WriteLn('BUILT-IN CORRECTNESS INVARIANTS');
  WriteLn('========================================================================');

  // (1) K=1 reduction: K=1 MDN's mu must match the MSE arm's prediction over a
  // probe grid (both recover the conditional mean E[y|x]).
  MeanK1Diff := 0;
  for I := 0 to 40 do
  begin
    MDNForward(NNmdn1, 1, 0.0 + 1.3 * I / 40.0, P1);   // sweep x over data range
    msePred := MSEForward(NNmse, 0.0 + 1.3 * I / 40.0);
    MeanK1Diff := MeanK1Diff + Abs(P1.Mu[0] - msePred);
  end;
  MeanK1Diff := MeanK1Diff / 41.0;
  inv1ok := MeanK1Diff < 0.05;
  WriteLn('(1) K=1 MDN reduces to homoscedastic Gaussian; argmax-mu == MSE arm:');
  WriteLn(Format('    mean |mu_{K=1}(x) - f_MSE(x)| over a 41-pt grid = %s',
    [FmtF(MeanK1Diff, 8, 5)]));
  if inv1ok then WriteLn('    K=1 == MSE arm (mean abs diff < 0.05) : PASS')
  else WriteLn('    K=1 == MSE arm : FAIL');

  // (2) Simplex: mixture weights sum to 1 at every probe x (within 1e-5).
  MaxSimplexErr := 0;
  for I := 0 to High(ProbeX) do
  begin
    MDNForward(NNmdn, cK, ProbeX[I], P);
    PiSum := 0;
    for k1 := 0 to cK - 1 do PiSum := PiSum + P.Pi[k1];
    if Abs(PiSum - 1.0) > MaxSimplexErr then MaxSimplexErr := Abs(PiSum - 1.0);
  end;
  inv2ok := MaxSimplexErr < 1e-5;
  WriteLn('(2) Mixture weights sum to 1 at every probe x:');
  WriteLn(Format('    max |sum_k pi_k - 1| over %d probes = %.3e',
    [Length(ProbeX), MaxSimplexErr]));
  if inv2ok then WriteLn('    simplex (|sum pi - 1| < 1e-5) : PASS')
  else WriteLn('    simplex : FAIL');
  WriteLn;

  if not (inv1ok and inv2ok) then
  begin
    WriteLn('ONE OR MORE INVARIANTS FAILED -- HALT.');
    NNmse.Free; NNmdn.Free; NNmdn1.Free;
    Halt(1);
  end;
  WriteLn('All invariants PASS.');
  WriteLn;
  WriteLn(Format('Total wall time: %.2f s', [(Now - StartT) * 24 * 60 * 60]));

  NNmse.Free; NNmdn.Free; NNmdn1.Free;
end.
