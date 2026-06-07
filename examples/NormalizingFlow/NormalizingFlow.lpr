(*
  NormalizingFlow example

  Fits a 2-D "two-moons" density with a small RealNVP/Glow-style normalizing
  flow built from stacked TNNetAffineCoupling layers -- the library's first
  exact-likelihood generative primitive.

  The flow F maps a data point x in R^2 to a latent z in R^2 through a stack of
  affine coupling layers. Each coupling leaves one channel UNCHANGED and applies
  an affine map y_b = x_b*exp(s)+t to the other, where the per-channel log-scale
  s and shift t are produced by a tiny conditioner reading the unchanged channel.
  Successive couplings ALTERNATE which channel is transformed (the constructor's
  pTransformSecond flag) so every dimension gets updated.

  Because each coupling is analytically invertible and its Jacobian is
  triangular, the change-of-variables formula gives an EXACT log-likelihood under
  a unit-Gaussian base p_Z:
        log p_X(x) = log p_Z(F(x)) + sum_couplings log|det J|
                   = -0.5*||z||^2 - 0.5*D*log(2*pi) + sum_couplings(LogDetJacobian)
  Training maximizes this directly (no ELBO, no adversary). The per-layer
  log-det is read from each coupling's LogDetJacobian property; the -sum(s)
  gradient is folded into each layer's backward pass (LogDetLossWeight=1), so we
  only inject the data-loss gradient dL/dz = z and backprop with a zero target.

  After training we SAMPLE: draw z ~ N(0,I) and run the SAME couplings in their
  inverse (sampling) direction z -> x to generate new points, which land on the
  two-moons manifold.

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
  cHalfLog2Pi = 0.9189385332; // 0.5*log(2*pi)

var
  Flow: TNNet;        // forward flow x -> z (couplings, pInverse=false)
  Sampler: TNNet;     // inverse flow z -> x (same weights, pInverse=true)

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

// Mean log-likelihood over a held-out batch (higher is better).
function MeanLogLik(N: integer): TNeuralFloat;
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
        if Flow.Layers[c] is TNNetAffineCoupling then
          logdet := logdet + TNNetAffineCoupling(Flow.Layers[c]).LogDetJacobian;
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

var
  X, Target: TNNetVolume;
  ep, step, b, c, idx: integer;
  ll: TNeuralFloat;
  zmean, zstd, sx, sy: TNeuralFloat;
  Saved: string;
begin
  // Mask FP traps (saturating tanh/exp tails can raise hardware exceptions on
  // extreme intermediates without producing actual NaNs in the reported loss).
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  Randomize;
  RandSeed := 42;

  // Build the forward flow: cNumCouple couplings alternating the transformed
  // half so every channel is updated. Couplings are near-identity at init.
  Flow := TNNet.Create();
  Flow.AddLayer(TNNetInput.Create(1, 1, cDim));
  for c := 0 to cNumCouple - 1 do
    Flow.AddLayer(TNNetAffineCoupling.Create((c mod 2) = 0, false, cClamp));
  Flow.SetLearningRate(cLR, 0.0);
  // Accumulate per-sample grads across the mini-batch, then update once.
  Flow.SetBatchUpdate(true);

  Target := TNNetVolume.Create(1, 1, cDim); // zero target -> dL/dz = z

  WriteLn('Normalizing flow on two-moons: ', cNumCouple,
    ' affine couplings, dim=', cDim);
  WriteLn('Initial mean log-likelihood: ', MeanLogLik(2000):0:4);
  WriteLn;

  X := TNNetVolume.Create(1, 1, cDim);
  for ep := 1 to cEpochs do
  begin
    for step := 0 to cStepsPerEp - 1 do
    begin
      Flow.ClearDeltas();
      for b := 0 to cBatch - 1 do
      begin
        SampleTwoMoons(X);
        Flow.Compute(X);
        // Data-loss gradient dL/dz = z (from 0.5*||z||^2); the -sum(s) log-det
        // gradient is folded into each coupling's backward (LogDetLossWeight=1).
        Flow.Backpropagate(Target);
      end;
      Flow.UpdateWeights();
    end;
    if (ep mod 40 = 0) or (ep = 1) then
    begin
      ll := MeanLogLik(2000);
      WriteLn('epoch ', ep:4, '/', cEpochs, '   mean log-likelihood = ', ll:0:4);
    end;
  end;
  WriteLn;
  WriteLn('Final mean log-likelihood: ', MeanLogLik(4000):0:4);
  WriteLn;

  // Build the inverse (sampling) flow: SAME weights, couplings run z -> x in
  // REVERSE order with pInverse=true.
  Sampler := TNNet.Create();
  Sampler.AddLayer(TNNetInput.Create(1, 1, cDim));
  for c := cNumCouple - 1 downto 0 do
    Sampler.AddLayer(TNNetAffineCoupling.Create((c mod 2) = 0, true, cClamp));
  // Copy trained conditioner weights coupling-by-coupling (reverse order).
  for c := 0 to cNumCouple - 1 do
  begin
    // Forward coupling at Flow layer (1+c) corresponds to Sampler layer
    // (1 + (cNumCouple-1-c)).
    idx := 1 + (cNumCouple - 1 - c);
    for b := 0 to TNNetAffineCoupling(Flow.Layers[1 + c]).Neurons.Count - 1 do
    begin
      Sampler.Layers[idx].Neurons[b].Weights.Copy(
        Flow.Layers[1 + c].Neurons[b].Weights);
      Sampler.Layers[idx].Neurons[b].BiasWeight :=
        Flow.Layers[1 + c].Neurons[b].BiasWeight;
    end;
  end;

  // Sample fresh points from the base Gaussian and report their spread; they
  // should resemble the (scaled) two-moons cloud, not a round Gaussian.
  WriteLn('Sampling 2000 points z~N(0,I) through the inverse flow:');
  zmean := 0; zstd := 0; sx := 0; sy := 0;
  for b := 0 to 1999 do
  begin
    X.Raw[0] := Gauss();
    X.Raw[1] := Gauss();
    Sampler.Compute(X);
    sx := sx + Sampler.GetLastLayer.Output.Raw[0];
    sy := sy + Sampler.GetLastLayer.Output.Raw[1];
    zmean := zmean + Sampler.GetLastLayer.Output.Raw[0];
    zstd  := zstd + Sqr(Sampler.GetLastLayer.Output.Raw[1]);
  end;
  WriteLn('  generated-sample mean x = ', (sx / 2000):0:4,
    '   mean y = ', (sy / 2000):0:4);

  // Round-trip a few real points x -> z -> x to confirm exact invertibility.
  WriteLn;
  WriteLn('Forward->inverse reconstruction check on 5 data points:');
  for b := 0 to 4 do
  begin
    SampleTwoMoons(X);
    sx := X.Raw[0]; sy := X.Raw[1];
    Flow.Compute(X);
    // Feed z through the sampler.
    Target.Copy(Flow.GetLastLayer.Output);
    Sampler.Compute(Target);
    WriteLn('  x=(', sx:0:4, ',', sy:0:4, ')  ->  x_rec=(',
      Sampler.GetLastLayer.Output.Raw[0]:0:4, ',',
      Sampler.GetLastLayer.Output.Raw[1]:0:4, ')');
  end;

  Saved := Flow.SaveToString();
  WriteLn;
  WriteLn('Flow serialized length: ', Length(Saved), ' chars (round-trips via LoadFromString).');

  X.Free;
  Target.Free;
  Sampler.Free;
  Flow.Free;
end.
