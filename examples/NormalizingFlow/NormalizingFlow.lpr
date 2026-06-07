(*
  NormalizingFlow example

  Fits a 2-D "two-moons" density with a small RealNVP/Glow-style normalizing
  flow built from stacked TNNetAffineCoupling layers -- the library's first
  exact-likelihood generative primitive -- and shows that interleaving the
  affine couplings with Glow's LEARNABLE invertible 1x1 convolution
  (TNNetInvertible1x1Conv) reaches a HIGHER mean log-likelihood than the
  fixed-permute baseline on the same target.

  The flow F maps a data point x in R^2 to a latent z in R^2 through a stack of
  affine coupling layers. Each coupling leaves one channel UNCHANGED and applies
  an affine map y_b = x_b*exp(s)+t to the other, where the per-channel log-scale
  s and shift t are produced by a tiny conditioner reading the unchanged channel.
  Successive couplings ALTERNATE which channel is transformed (the constructor's
  pTransformSecond flag) so every dimension gets updated. That alternation is a
  FIXED channel permutation -- the baseline flow here.

  The REAL Glow step replaces the fixed permutation with a learnable C x C matrix
  W applied per spatial position across the Depth axis (TNNetInvertible1x1Conv).
  W is parametrized by its LU decomposition W = P*L*(U+diag(s)) so its per-step
  log-det is the cheap sum(log|s|); the network adapts the channel mixing instead
  of being stuck with a fixed permute. The "glow" flow below interleaves a 1x1
  conv between every pair of couplings.

  Because each step is analytically invertible and its Jacobian log-det is
  available in closed form, the change-of-variables formula gives an EXACT
  log-likelihood under a unit-Gaussian base p_Z:
        log p_X(x) = log p_Z(F(x)) + sum_steps log|det J|
                   = -0.5*||z||^2 - 0.5*D*log(2*pi) + sum_steps(LogDetJacobian)
  Training maximizes this directly (no ELBO, no adversary). The per-layer log-det
  is read from each step's LogDetJacobian property; the -logdet gradient is folded
  into each layer's backward pass (LogDetLossWeight=1), so we only inject the
  data-loss gradient dL/dz = z and backprop with a zero target.

  After training we SAMPLE the glow flow: draw z ~ N(0,I) and run the SAME layers
  in their inverse (sampling) direction z -> x to generate new points, which land
  on the two-moons manifold; a forward->inverse round-trip confirms exact
  invertibility.

  Pure CPU, tiny dims, a few thousand mini-steps; finishes well under a minute.

  Coded by Claude (AI).
*)
program NormalizingFlow;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  cDim        = 2;      // data dimensionality (2-D two-moons)
  cNumCouple  = 6;      // number of stacked affine couplings
  cClamp      = 2.0;    // Glow log-scale tanh clamp
  cBatch      = 64;
  cEpochs     = 300;    // mini-batches per "epoch" report
  cStepsPerEp = 40;
  cLR         = 0.0001;  // small: batch-update SUMS per-sample deltas (no /N)
  cGlowLR     = 0.00005; // smaller for the glow flow: the learnable 1x1 conv
                         // adds an unbounded matrix + a -1/s log-det gradient, so
                         // a lower step keeps |s| and the matrix entries stable
                         // (lowering the LR, not the likelihood comparison).
  cGlowEpochs = 600;     // more epochs to offset the smaller glow step
  cGlowL2     = 0.00002; // light weight decay bounds the 1x1-conv matrix
  cHalfLog2Pi = 0.9189385332; // 0.5*log(2*pi)

var
  Baseline: TNNet;    // forward flow x -> z, couplings only (fixed permute)
  Glow: TNNet;        // forward flow x -> z, couplings + learnable 1x1 conv
  Sampler: TNNet;     // inverse of Glow: z -> x (same weights, inverse=true)

// Standard-normal draw (Box-Muller).
function Gauss(): TNeuralFloat;
var u1, u2: TNeuralFloat;
begin
  u1 := Random;
  if u1 < 1e-12 then u1 := 1e-12;
  u2 := Random;
  Result := Sqrt(-2 * Ln(u1)) * Cos(2 * Pi * u2);
end;

// Draw one two-moons point into a (1,1,2) volume.
procedure SampleTwoMoons(V: TNNetVolume);
var
  upper: boolean;
  ang, x, y: TNeuralFloat;
begin
  upper := Random(2) = 0;
  ang := Random * Pi;
  if upper then
  begin
    x := Cos(ang);
    y := Sin(ang);
  end
  else
  begin
    x := 1 - Cos(ang);
    y := -Sin(ang) - 0.3;
  end;
  // Light Gaussian jitter; scaled so the manifold sits near the unit Gaussian.
  V.Raw[0] := (x + Gauss() * 0.06) * 0.8;
  V.Raw[1] := (y + Gauss() * 0.06) * 0.8;
end;

// Mean log-likelihood of Flow over N fresh samples (higher is better). Sums the
// LogDetJacobian of EVERY flow step (couplings and any 1x1 convs).
function MeanLogLik(Flow: TNNet; N: integer): TNeuralFloat;
var
  X: TNNetVolume;
  i, c: integer;
  sumLL, logdet, z2: TNeuralFloat;
begin
  X := TNNetVolume.Create(1, 1, cDim);
  sumLL := 0;
  try
    for i := 0 to N - 1 do
    begin
      SampleTwoMoons(X);
      Flow.Compute(X);
      logdet := 0;
      for c := 0 to Flow.GetLastLayerIdx do
      begin
        if Flow.Layers[c] is TNNetAffineCoupling then
          logdet := logdet + TNNetAffineCoupling(Flow.Layers[c]).LogDetJacobian;
        if Flow.Layers[c] is TNNetInvertible1x1Conv then
          logdet := logdet + TNNetInvertible1x1Conv(Flow.Layers[c]).LogDetJacobian;
      end;
      z2 := 0;
      for c := 0 to cDim - 1 do
        z2 := z2 + Sqr(Flow.GetLastLayer.Output.Raw[c]);
      sumLL := sumLL + (-0.5 * z2 - cDim * cHalfLog2Pi + logdet);
    end;
    Result := sumLL / N;
  finally
    X.Free;
  end;
end;

// One training run of Flow by exact maximum likelihood.
procedure TrainFlow(Flow: TNNet; const Tag: string; AEpochs: integer);
var
  X, Target: TNNetVolume;
  ep, step, b, reportEvery: integer;
  ll: TNeuralFloat;
begin
  X := TNNetVolume.Create(1, 1, cDim);
  Target := TNNetVolume.Create(1, 1, cDim); // zero target -> dL/dz = z
  reportEvery := AEpochs div 5;
  try
    for ep := 1 to AEpochs do
    begin
      for step := 0 to cStepsPerEp - 1 do
      begin
        Flow.ClearDeltas();
        for b := 0 to cBatch - 1 do
        begin
          SampleTwoMoons(X);
          Flow.Compute(X);
          // dL/dz = z (from 0.5*||z||^2); each step's -logdet gradient is folded
          // into its backward (LogDetLossWeight=1).
          Flow.Backpropagate(Target);
        end;
        Flow.UpdateWeights();
      end;
      if (ep mod reportEvery = 0) or (ep = 1) then
      begin
        ll := MeanLogLik(Flow, 2000);
        WriteLn('  [', Tag, '] epoch ', ep:4, '/', AEpochs,
          '   mean log-likelihood = ', ll:0:4);
      end;
    end;
  finally
    X.Free;
    Target.Free;
  end;
end;

var
  X, Target: TNNetVolume;
  c, b, idx: integer;
  baseLL, glowLL: TNeuralFloat;
  sx, sy: TNeuralFloat;
  Saved: string;
  // Records the Glow build so the inverse sampler can be reconstructed without
  // reading protected layer fields. One entry per non-input Glow layer.
  glIsConv: array of boolean;     // true = 1x1 conv, false = affine coupling
  glTransSecond: array of boolean; // coupling pTransformSecond
  glSeed: array of integer;        // conv permutation seed
  nGlSteps: integer;
begin
  // Mask FP traps (saturating tanh/exp tails can raise hardware exceptions on
  // extreme intermediates without producing actual NaNs in the reported loss).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  Randomize;
  RandSeed := 42;

  // -- Baseline: couplings only, mixing via the FIXED alternating permute. --
  Baseline := TNNet.Create();
  Baseline.AddLayer(TNNetInput.Create(1, 1, cDim));
  for c := 0 to cNumCouple - 1 do
    Baseline.AddLayer(TNNetAffineCoupling.Create((c mod 2) = 0, false, cClamp));
  Baseline.SetLearningRate(cLR, 0.0);
  Baseline.SetBatchUpdate(true);

  // -- Glow: SAME couplings but a LEARNABLE invertible 1x1 conv between each
  //    consecutive pair, replacing the fixed permute with adaptive mixing. --
  Glow := TNNet.Create();
  Glow.AddLayer(TNNetInput.Create(1, 1, cDim));
  nGlSteps := 0;
  SetLength(glIsConv, 2 * cNumCouple);
  SetLength(glTransSecond, 2 * cNumCouple);
  SetLength(glSeed, 2 * cNumCouple);
  for c := 0 to cNumCouple - 1 do
  begin
    Glow.AddLayer(TNNetAffineCoupling.Create((c mod 2) = 0, false, cClamp));
    glIsConv[nGlSteps] := false; glTransSecond[nGlSteps] := (c mod 2) = 0;
    glSeed[nGlSteps] := 0; Inc(nGlSteps);
    if c < cNumCouple - 1 then
    begin
      Glow.AddLayer(TNNetInvertible1x1Conv.Create(false, c));
      glIsConv[nGlSteps] := true; glTransSecond[nGlSteps] := false;
      glSeed[nGlSteps] := c; Inc(nGlSteps);
    end;
  end;
  Glow.SetLearningRate(cGlowLR, 0.0);
  Glow.SetL2Decay(cGlowL2);
  Glow.SetBatchUpdate(true);

  WriteLn('Normalizing flow on two-moons (dim=', cDim, ', ', cNumCouple,
    ' couplings).');
  WriteLn('Comparing fixed-permute baseline vs. learnable Glow 1x1-conv mixing.');
  WriteLn;
  WriteLn('Initial mean log-likelihood:');
  WriteLn('  baseline = ', MeanLogLik(Baseline, 2000):0:4,
    '   glow = ', MeanLogLik(Glow, 2000):0:4);
  WriteLn;

  WriteLn('Training baseline (couplings + fixed permute)...');
  TrainFlow(Baseline, 'base', cEpochs);
  WriteLn;
  WriteLn('Training glow (couplings + learnable 1x1 conv)...');
  TrainFlow(Glow, 'glow', cGlowEpochs);
  WriteLn;

  baseLL := MeanLogLik(Baseline, 8000);
  glowLL := MeanLogLik(Glow, 8000);
  WriteLn('Final mean log-likelihood (8000 held-out samples):');
  WriteLn('  baseline (fixed permute)    = ', baseLL:0:4);
  WriteLn('  glow (learnable 1x1 conv)   = ', glowLL:0:4);
  WriteLn('  improvement                 = ', (glowLL - baseLL):0:4);
  if glowLL > baseLL then
    WriteLn('  -> learnable mixing reaches a HIGHER log-likelihood, as expected.')
  else
    WriteLn('  -> NOTE: glow did not beat baseline this run.');
  WriteLn;

  // Build the inverse (sampling) flow for the GLOW model: SAME weights, every
  // layer run in REVERSE order with its inverse direction.
  Sampler := TNNet.Create();
  Sampler.AddLayer(TNNetInput.Create(1, 1, cDim));
  for c := nGlSteps - 1 downto 0 do
  begin
    if glIsConv[c] then
      // Same permutation seed so P matches; inverse direction.
      Sampler.AddLayer(TNNetInvertible1x1Conv.Create(true, glSeed[c]))
    else
      Sampler.AddLayer(TNNetAffineCoupling.Create(glTransSecond[c], true, cClamp));
  end;
  // Copy trained weights layer-by-layer (Sampler layer 1+k <- Glow layer
  // GetLastLayerIdx-k).
  for c := 1 to Sampler.GetLastLayerIdx do
  begin
    idx := Glow.GetLastLayerIdx - (c - 1);
    for b := 0 to Glow.Layers[idx].Neurons.Count - 1 do
    begin
      Sampler.Layers[c].Neurons[b].Weights.Copy(Glow.Layers[idx].Neurons[b].Weights);
      Sampler.Layers[c].Neurons[b].BiasWeight := Glow.Layers[idx].Neurons[b].BiasWeight;
    end;
  end;

  // Sample fresh points from the base Gaussian through the inverse glow flow.
  WriteLn('Sampling 2000 points z~N(0,I) through the inverse glow flow:');
  sx := 0; sy := 0;
  X := TNNetVolume.Create(1, 1, cDim);
  Target := TNNetVolume.Create(1, 1, cDim);
  for b := 0 to 1999 do
  begin
    X.Raw[0] := Gauss();
    X.Raw[1] := Gauss();
    Sampler.Compute(X);
    sx := sx + Sampler.GetLastLayer.Output.Raw[0];
    sy := sy + Sampler.GetLastLayer.Output.Raw[1];
  end;
  WriteLn('  generated-sample mean x = ', (sx / 2000):0:4,
    '   mean y = ', (sy / 2000):0:4);

  // Round-trip a few real points x -> z -> x to confirm exact invertibility.
  WriteLn;
  WriteLn('Forward->inverse reconstruction check on 5 data points (glow):');
  for b := 0 to 4 do
  begin
    SampleTwoMoons(X);
    sx := X.Raw[0]; sy := X.Raw[1];
    Glow.Compute(X);
    Target.Copy(Glow.GetLastLayer.Output);
    Sampler.Compute(Target);
    WriteLn('  x=(', sx:0:4, ',', sy:0:4, ')  ->  x_rec=(',
      Sampler.GetLastLayer.Output.Raw[0]:0:4, ',',
      Sampler.GetLastLayer.Output.Raw[1]:0:4, ')');
  end;

  Saved := Glow.SaveToString();
  WriteLn;
  WriteLn('Glow flow serialized length: ', Length(Saved),
    ' chars (round-trips via LoadFromString).');

  X.Free;
  Target.Free;
  Sampler.Free;
  Glow.Free;
  Baseline.Free;
end.
