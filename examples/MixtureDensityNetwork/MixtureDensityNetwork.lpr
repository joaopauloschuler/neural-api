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
  The net emits 3*K RAW outputs, reshaped into K triples (a_k, m_k, s_k):
      pi_k    = softmax(a)_k          (mixing coefficients, sum to 1)
      mu_k    = m_k                   (component means, linear)
      sigma_k = softplus(s_k) + eps   (positive component widths)
  trained on the mixture NEGATIVE LOG-LIKELIHOOD
      NLL = -log( sum_k pi_k * Normal(y ; mu_k, sigma_k) ).

MANUAL GRADIENT SURGERY (no library changes; pseudo-target trick)
  Both arms emit RAW linear outputs from a TNNetFullConnectLinear head. The
  framework's stock Backpropagate seeds the output layer's error as
  (output - target) and, for a Linear head (Identity activation, derivative 1),
  delivers exactly that as the gradient w.r.t. the raw outputs. So to inject an
  arbitrary analytic gradient g_i w.r.t. raw output i we feed a PSEUDO-TARGET
      pseudo_i = output_i - g_i      ->  (output - pseudo)_i == g_i .
  The closed-form mixture-NLL gradients (responsibilities gamma_k):
      gamma_k    = pi_k*N_k / sum_j pi_j*N_j
      dNLL/da_k  = pi_k - gamma_k                          (softmax / logits)
      dNLL/dm_k  = gamma_k * (mu_k - y) / sigma_k^2        (mu, linear)
      dNLL/ds_k  = gamma_k * (1/sigma_k - (y-mu_k)^2/sigma_k^3) * sigmoid(s_k)
                   (sigma, chained through softplus' = sigmoid).
  The MSE arm uses the same trick with g_i = (out_i - y) (single linear output).
  We hand-roll the mini-batch loop in BATCH-UPDATE mode (NN.SetBatchUpdate(True)
  -> Backpropagate ACCUMULATES into Neurons[].Delta; UpdateWeights applies once),
  scaling each per-sample gradient by 1/batch so the applied step is the MEAN.
  We never call TNeuralFit.Fit, so layer references never go stale.

BUILT-IN CORRECTNESS INVARIANTS (HALT(1) on violation)
  (1) K=1 REDUCTION: a K=1 MDN's NLL is a homoscedastic Gaussian NLL whose only
      mean is mu_0; its mu must match the independently-trained MSE arm's
      prediction (both recover E[y|x]). We assert the mean abs difference between
      the K=1 mu and the MSE prediction over a probe grid is small.
  (2) SIMPLEX: the mixture weights pi_k(x) must sum to 1 (within 1e-5) at every
      probe x (softmax sanity).
  Also a finite-difference GRADIENT CHECK of the analytic NLL gradient at startup
  (HALT(1) if the closed form disagrees with central differences).

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
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cSeed       = 424242;
  cNumPts     = 600;          // training points (tiny)
  cNoise      = 0.05;         // additive noise on the forward map
  cHiddenW    = 32;           // MLP hidden width
  cK          = 3;            // mixture components (headline arm)
  cBatchSize  = 60;
  cEpochs     = 600;
  cLearnRate  = 0.02;
  cSigmaEps   = 1e-3;         // sigma floor (softplus + eps)
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
// Numerically-stable softplus and its derivative (= logistic sigmoid).
// ---------------------------------------------------------------------------
function SoftPlus(z: TNeuralFloat): TNeuralFloat;
begin
  if z > 30 then Result := z
  else if z < -30 then Result := Exp(z)
  else Result := Ln(1 + Exp(z));
end;

function Sigmoid(z: TNeuralFloat): TNeuralFloat;
begin
  if z >= 0 then Result := 1.0 / (1.0 + Exp(-z))
  else begin Result := Exp(z); Result := Result / (1.0 + Result); end;
end;

// ---------------------------------------------------------------------------
// Decode the 3*K raw network outputs of an MDN with K components into
// (pi, mu, sigma). Layout: [a_0..a_{K-1}, m_0..m_{K-1}, s_0..s_{K-1}].
// ---------------------------------------------------------------------------
procedure DecodeMix(const Raw: TNNetVolume; K: integer; out P: TMixParams);
var
  k1: integer;
  MaxA, SumExp: TNeuralFloat;
  Ex: array of TNeuralFloat;
begin
  SetLength(P.Pi, K); SetLength(P.Mu, K); SetLength(P.Sigma, K);
  SetLength(Ex, K);
  // softmax over the pi logits (subtract max for stability)
  MaxA := Raw.FData[0];
  for k1 := 1 to K - 1 do if Raw.FData[k1] > MaxA then MaxA := Raw.FData[k1];
  SumExp := 0;
  for k1 := 0 to K - 1 do
  begin
    Ex[k1] := Exp(Raw.FData[k1] - MaxA);
    SumExp := SumExp + Ex[k1];
  end;
  for k1 := 0 to K - 1 do
  begin
    P.Pi[k1]    := Ex[k1] / SumExp;
    P.Mu[k1]    := Raw.FData[K + k1];
    P.Sigma[k1] := SoftPlus(Raw.FData[2 * K + k1]) + cSigmaEps;
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
// Analytic gradient of the mixture NLL w.r.t. the 3*K RAW outputs, written into
// Grad (same layout as the raw outputs). RawSigma holds the pre-softplus sigma
// logits (needed for the softplus' = sigmoid chain). Returns the NLL value.
// ---------------------------------------------------------------------------
function MixtureNLLGrad(const Raw: TNNetVolume; const P: TMixParams; K: integer;
  y: TNeuralFloat; Grad: TNNetVolume): TNeuralFloat;
var
  k1: integer;
  Nk, Denom, gamma, dmu, dsigma, sig: TNeuralFloat;
  Resp: array of TNeuralFloat;
begin
  SetLength(Resp, K);
  Denom := 0;
  for k1 := 0 to K - 1 do
  begin
    Nk := GaussPdf(y, P.Mu[k1], P.Sigma[k1]);
    Resp[k1] := P.Pi[k1] * Nk;       // pi_k * N_k (un-normalised responsibility)
    Denom := Denom + Resp[k1];
  end;
  if Denom < cNLLEps then Denom := cNLLEps;

  for k1 := 0 to K - 1 do
  begin
    gamma := Resp[k1] / Denom;       // posterior responsibility gamma_k
    // d/d a_k (pi logits): pi_k - gamma_k
    Grad.FData[k1] := P.Pi[k1] - gamma;
    // d/d mu_k: gamma_k * (mu_k - y)/sigma_k^2 ; mu = m_k so chain is 1
    dmu := gamma * (P.Mu[k1] - y) / (P.Sigma[k1] * P.Sigma[k1]);
    Grad.FData[K + k1] := dmu;
    // d/d sigma_k: gamma_k * (1/sigma - (y-mu)^2/sigma^3) ; chain softplus'=sigmoid
    dsigma := gamma * (1.0 / P.Sigma[k1]
              - Sqr(y - P.Mu[k1]) / (P.Sigma[k1] * P.Sigma[k1] * P.Sigma[k1]));
    sig := Sigmoid(Raw.FData[2 * K + k1]);
    Grad.FData[2 * K + k1] := dsigma * sig;
  end;
  Result := MixtureNLL(P, K, y);
end;

// ---------------------------------------------------------------------------
// MLP with a RAW linear head of OutDim outputs.
//   Input(1) -> Tanh(H) -> Tanh(H) -> Linear(OutDim)
// Tanh hidden layers keep forward/backward RNG-free and give smooth fits.
// ---------------------------------------------------------------------------
procedure BuildNet(out NN: TNNet; OutDim: integer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, 1));
  NN.AddLayer(TNNetFullConnect.Create(cHiddenW));        // Tanh
  NN.AddLayer(TNNetFullConnect.Create(cHiddenW));        // Tanh
  NN.AddLayer(TNNetFullConnectLinear.Create(OutDim));    // RAW linear head
  NN.SetLearningRate(cLearnRate, {Momentum=}0.0);
  NN.SetL2Decay(0.0);
  // Batch-update: Backpropagate accumulates into Neurons[].Delta; UpdateWeights
  // applies once per mini-batch. REQUIRED for manual gradient surgery.
  NN.SetBatchUpdate(True);
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
// Train an MDN with K components on the mixture NLL via the analytic gradient.
// ---------------------------------------------------------------------------
procedure TrainMDN(NN: TNNet; K: integer);
var
  Order: array of integer;
  Epoch, Lo, Hi, I, J, R, Tmp: integer;
  Inp, Pseudo, Grad, Outp: TNNetVolume;
  P: TMixParams;
begin
  Inp := TNNetVolume.Create(1, 1, 1);
  Pseudo := TNNetVolume.Create(3 * K, 1, 1);
  Grad := TNNetVolume.Create(3 * K, 1, 1);
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
        DecodeMix(Outp, K, P);
        MixtureNLLGrad(Outp, P, K, DataY[Order[I]], Grad);
        // pseudo = out - g/batch  -> stock error == g/batch == MEAN gradient
        for R := 0 to 3 * K - 1 do
          Pseudo.FData[R] := Outp.FData[R] - Grad.FData[R] / (Hi - Lo);
        NN.Backpropagate(Pseudo);
      end;
      NN.UpdateWeights();
      Lo := Hi;
    end;
  end;
  Grad.Free; Pseudo.Free; Inp.Free;
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

// ---------------------------------------------------------------------------
// Finite-difference check of the analytic mixture-NLL gradient w.r.t. the raw
// outputs, at a random raw vector / random target. HALT(1) on disagreement.
// ---------------------------------------------------------------------------
// This check operates entirely in DOUBLE precision (not the single-precision
// TNNetVolume) so it validates the analytic FORMULAS, not float-rounding: a
// single-precision round-trip would inject ~1e-2 noise into the central diff.
procedure GradientCheck;
const
  K = 3; H = 1e-5;
var
  raw, grad: array[0 .. 3 * K - 1] of Double;
  y, fdg, ana, maxrel, rel, save, lp, lm: Double;
  R: integer;

  // Double-precision softplus / sigmoid (the unit-level ones take single args,
  // which would truncate the Double probe and pollute the central difference).
  function SoftPlusD(z: Double): Double;
  begin
    if z > 30 then Result := z
    else if z < -30 then Result := Exp(z)
    else Result := Ln(1 + Exp(z));
  end;

  function SigmoidD(z: Double): Double;
  begin
    if z >= 0 then Result := 1.0 / (1.0 + Exp(-z))
    else begin Result := Exp(z); Result := Result / (1.0 + Result); end;
  end;

  // Self-contained Double-precision mixture NLL of the raw vector at target ty.
  function NLLat(const rr: array of Double; ty: Double): Double;
  var
    k1: integer;
    maxa, sumexp, pk, muk, sgk, d, s: Double;
    pp: array[0 .. K - 1] of Double;
  begin
    maxa := rr[0];
    for k1 := 1 to K - 1 do if rr[k1] > maxa then maxa := rr[k1];
    sumexp := 0;
    for k1 := 0 to K - 1 do begin pp[k1] := Exp(rr[k1] - maxa); sumexp := sumexp + pp[k1]; end;
    s := 0;
    for k1 := 0 to K - 1 do
    begin
      pk  := pp[k1] / sumexp;
      muk := rr[K + k1];
      sgk := SoftPlusD(rr[2 * K + k1]) + cSigmaEps;
      d := ty - muk;
      s := s + pk * Exp(-(d * d) / (2 * sgk * sgk)) / (sgk * Sqrt(cPi2));
    end;
    if s < cNLLEps then s := cNLLEps;
    Result := -Ln(s);
  end;

  // Double-precision analytic gradient (mirrors MixtureNLLGrad exactly).
  procedure AnaGrad(const rr: array of Double; ty: Double; var g: array of Double);
  var
    k1: integer;
    maxa, sumexp, denom, gamma, muk, sgk, nk: Double;
    pp, mu, sg, resp: array[0 .. K - 1] of Double;
  begin
    maxa := rr[0];
    for k1 := 1 to K - 1 do if rr[k1] > maxa then maxa := rr[k1];
    sumexp := 0;
    for k1 := 0 to K - 1 do begin pp[k1] := Exp(rr[k1] - maxa); sumexp := sumexp + pp[k1]; end;
    denom := 0;
    for k1 := 0 to K - 1 do
    begin
      pp[k1] := pp[k1] / sumexp;
      mu[k1] := rr[K + k1];
      sg[k1] := SoftPlusD(rr[2 * K + k1]) + cSigmaEps;
      nk := Exp(-Sqr(ty - mu[k1]) / (2 * sg[k1] * sg[k1])) / (sg[k1] * Sqrt(cPi2));
      resp[k1] := pp[k1] * nk;
      denom := denom + resp[k1];
    end;
    if denom < cNLLEps then denom := cNLLEps;
    for k1 := 0 to K - 1 do
    begin
      gamma := resp[k1] / denom;
      muk := mu[k1]; sgk := sg[k1];
      g[k1]         := pp[k1] - gamma;
      g[K + k1]     := gamma * (muk - ty) / (sgk * sgk);
      g[2 * K + k1] := gamma * (1.0 / sgk - Sqr(ty - muk) / (sgk * sgk * sgk))
                       * SigmoidD(rr[2 * K + k1]);
    end;
  end;

begin
  for R := 0 to 3 * K - 1 do raw[R] := RandG(0, 1);
  y := 0.4;
  AnaGrad(raw, y, grad);
  maxrel := 0;
  for R := 0 to 3 * K - 1 do
  begin
    save := raw[R];
    raw[R] := save + H; lp := NLLat(raw, y);
    raw[R] := save - H; lm := NLLat(raw, y);
    raw[R] := save;
    fdg := (lp - lm) / (2 * H);
    ana := grad[R];
    rel := Abs(fdg - ana) / (Abs(fdg) + Abs(ana) + 1e-8);
    if rel > maxrel then maxrel := rel;
  end;
  WriteLn(Format('Startup gradient check (analytic vs central-diff): max rel err = %.3e',
    [maxrel]));
  if maxrel > 1e-3 then
  begin
    WriteLn('  GRADIENT CHECK FAILED -- analytic NLL gradient is wrong. HALT.');
    Halt(1);
  end
  else
    WriteLn('  gradient check PASS (< 1e-3).');
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
  WriteLn(Format('Net: 1 -> %d(Tanh) -> %d(Tanh) -> Linear head (raw). seed=%d.',
    [cHiddenW, cHiddenW, cSeed]));
  WriteLn(Format('MDN head: K=%d components (3K=%d raw outputs); train=%d, batch=%d, epochs=%d, lr=%.3f.',
    [cK, 3 * cK, cNumPts, cBatchSize, cEpochs, cLearnRate]));
  WriteLn;

  // Startup analytic-gradient finite-difference check.
  RandSeed := cSeed;
  GradientCheck;
  WriteLn;

  // Build the shared dataset once (after seeding) so all arms see identical x,y.
  RandSeed := cSeed;
  MakeData;

  // ----------------------- Arm A: plain MSE regressor ---------------------
  RandSeed := cSeed;       // identical seed/arch before each arm
  BuildNet(NNmse, 1);
  TrainMSE(NNmse);
  mseMSE := MSEMeanMSE(NNmse);
  mseSigma := Sqrt(mseMSE);
  NLLmse := MSEMeanNLL(NNmse, mseSigma);

  // ----------------------- Arm B: K=3 MDN ---------------------------------
  RandSeed := cSeed;
  BuildNet(NNmdn, 3 * cK);
  TrainMDN(NNmdn, cK);
  NLLmdn := MDNMeanNLL(NNmdn, cK);
  MSEmdn := MDNMeanMSE(NNmdn, cK);

  // ----------------------- Arm C: K=1 MDN (for invariant 1) ---------------
  RandSeed := cSeed;
  BuildNet(NNmdn1, 3 * 1);
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
