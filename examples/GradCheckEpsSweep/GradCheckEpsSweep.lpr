program GradCheckEpsSweep;
(*
GradCheckEpsSweep: a didactic numerical-gradient epsilon sweep.

Takes ONE well-tested, simple layer (TNNetFullConnectLinear) with
hand-set deterministic weights and a fixed input, attaches a trivial
0.5*sum((y-d)^2) MSE loss, and runs the SAME central-difference
gradient check used in tests/TestNeuralNumerical.pas
(CellLayerGradientCheck) for a sweep of step sizes:

    eps in {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7}

For each eps it central-differences the loss,

    g_num(eps) = ( L(w+eps) - L(w-eps) ) / (2*eps)

and compares it against the analytic gradient from Backpropagate
(the weight Delta and the input OutputError). It prints, per eps, the
MAX absolute and MAX relative error across every swept parameter.

The point is the classic U-shaped finite-difference error curve:
  * large eps -> truncation (discretisation) error dominates  ~ O(eps^2)
  * tiny  eps -> floating-point round-off / cancellation dominates ~ O(machine_eps / eps)
with a sweet spot in between. Because TNeuralFloat is single-precision
(FP32) in this repo, the round-off arm kicks in early and the curve
bottoms out around eps ~ 1e-3..1e-4, NOT the 1e-8 you would see in
float64. That is exactly why the gradient tests use eps = 1e-4 with a
~0.01 tolerance.

No training loop: forward + backward only. Fully deterministic.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  // Step sizes to sweep. 1e-1 exposes the truncation (large-eps) arm,
  // 1e-7 exposes the round-off (tiny-eps) arm; the sweet spot is between.
  Epsilons: array[0..6] of TNeuralFloat =
    (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7);

var
  NN: TNNet;
  Layer: TNNetLayer;
  Input, Desired: TNNetVolume;
  i, e: integer;
  eps, lossPlus, lossMinus, gNum, gAna: TNeuralFloat;
  absErr, relErr, maxAbs, maxRel: TNeuralFloat;
  results: array[0..6] of record AbsErr, RelErr: TNeuralFloat; end;
  bestIdx: integer;

  // 0.5 * sum( (y - d)^2 ) MSE loss, the same loss the test idiom uses.
  function ComputeLoss(AInput: TNNetVolume): TNeuralFloat;
  var
    k: integer;
    diff: TNeuralFloat;
  begin
    NN.Compute(AInput);
    Result := 0;
    for k := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Desired.Raw[k];
      Result := Result + 0.5 * diff * diff;
    end;
  end;

begin
  // Forward + backward on a single fixed sample; no Fit / threading involved,
  // so the run is fully deterministic by construction.
  WriteLn('Numerical-gradient epsilon sweep (central differences).');
  WriteLn('Net: FullConnectLinear(3) -> HyperbolicTangent, fixed 4-element input,');
  WriteLn('     0.5*sum((y-d)^2) MSE loss. Tanh makes the loss genuinely nonlinear,');
  WriteLn('     so BOTH arms of the U-shape (truncation + round-off) are visible.');
  WriteLn('TNeuralFloat is FP32 in this build, so the round-off arm appears early.');
  WriteLn('Formula: g_num(eps) = ( L(w+eps) - L(w-eps) ) / (2*eps), compared to Backprop.');
  WriteLn;

  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Desired := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1, 1)); // pError=1 sizes the error volumes
    // pSuppressBias=1: no bias term, so the layer is purely y = W*x and the
    // whole experiment is deterministic from the hand-set weights alone.
    Layer := NN.AddLayer(TNNetFullConnectLinear.Create(3, {pSuppressBias=}1));
    // Tanh head: a smooth, well-tested nonlinearity (TestHyperbolicTangent-
    // GradientCheck covers it). It gives the loss a nonzero third derivative,
    // which is what makes the large-eps O(eps^2) truncation arm appear.
    NN.AddLayer(TNNetHyperbolicTangent.Create());
    NN.SetLearningRate(1.0, 0.0);
    NN.SetBatchUpdate(true); // keep Delta intact between forward/backward

    // Fully deterministic, hand-set input / target / weights.
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.7) * 2.0 + 0.3;
    for i := 0 to Desired.Size - 1 do
      Desired.Raw[i] := Cos(i * 0.5);
    for i := 0 to Layer.Neurons.Count - 1 do
    begin
      Layer.Neurons[i].Weights.Fill(0);
      // Linear, non-trivial weights so the analytic gradient is rock-solid.
      Layer.Neurons[i].Weights.Raw[0] := 0.5 + i * 0.13;
      Layer.Neurons[i].Weights.Raw[1] := -0.4 + i * 0.07;
      Layer.Neurons[i].Weights.Raw[2] := 0.9 - i * 0.11;
      Layer.Neurons[i].Weights.Raw[3] := 0.2 + i * 0.05;
    end;

    for e := 0 to High(Epsilons) do
    begin
      eps := Epsilons[e];
      maxAbs := 0;
      maxRel := 0;

      // ---- Gradient w.r.t. the first neuron's learnable weights ----
      for i := 0 to Layer.Neurons[0].Weights.Size - 1 do
      begin
        // Probe the first neuron's weights (representative; identical pattern).
        Layer.Neurons[0].Weights.Raw[i] := Layer.Neurons[0].Weights.Raw[i] + eps;
        lossPlus := ComputeLoss(Input);
        Layer.Neurons[0].Weights.Raw[i] := Layer.Neurons[0].Weights.Raw[i] - 2 * eps;
        lossMinus := ComputeLoss(Input);
        Layer.Neurons[0].Weights.Raw[i] := Layer.Neurons[0].Weights.Raw[i] + eps;
        gNum := (lossPlus - lossMinus) / (2 * eps);

        NN.Compute(Input);
        Layer.Neurons[0].ClearDelta;
        NN.Backpropagate(Desired);
        // Backprop accumulates Delta := Delta - LearningRate*gradient.
        gAna := -Layer.Neurons[0].Delta.Raw[i];

        absErr := Abs(gNum - gAna);
        if absErr > maxAbs then maxAbs := absErr;
        if Abs(gAna) > 1e-12 then
        begin
          relErr := absErr / Abs(gAna);
          if relErr > maxRel then maxRel := relErr;
        end;
      end;

      // ---- Gradient w.r.t. every input element ----
      for i := 0 to Input.Size - 1 do
      begin
        Input.Raw[i] := Input.Raw[i] + eps;
        lossPlus := ComputeLoss(Input);
        Input.Raw[i] := Input.Raw[i] - 2 * eps;
        lossMinus := ComputeLoss(Input);
        Input.Raw[i] := Input.Raw[i] + eps;
        gNum := (lossPlus - lossMinus) / (2 * eps);

        NN.Compute(Input);
        NN.Layers[0].OutputError.Fill(0);
        NN.Backpropagate(Desired);
        gAna := NN.Layers[0].OutputError.Raw[i];

        absErr := Abs(gNum - gAna);
        if absErr > maxAbs then maxAbs := absErr;
        if Abs(gAna) > 1e-12 then
        begin
          relErr := absErr / Abs(gAna);
          if relErr > maxRel then maxRel := relErr;
        end;
      end;

      results[e].AbsErr := maxAbs;
      results[e].RelErr := maxRel;
    end;

    // Find the eps with the smallest max-abs error (the sweet spot).
    bestIdx := 0;
    for e := 1 to High(Epsilons) do
      if results[e].AbsErr < results[bestIdx].AbsErr then
        bestIdx := e;

    WriteLn('      eps        max_abs_err     max_rel_err   ');
    WriteLn('  -----------   -------------   -------------  ');
    for e := 0 to High(Epsilons) do
    begin
      Write(Format('   %10.1e   %13.6e   %13.6e',
        [Epsilons[e], results[e].AbsErr, results[e].RelErr]));
      if e = bestIdx then
        WriteLn('   <-- min (sweet spot)')
      else
        WriteLn;
    end;
    WriteLn;
    WriteLn(Format('Minimum max-abs error at eps = %.1e (abs=%.3e, rel=%.3e).',
      [Epsilons[bestIdx], results[bestIdx].AbsErr, results[bestIdx].RelErr]));
    WriteLn;
    WriteLn('Large eps -> O(eps^2) truncation error;  tiny eps -> round-off (cancellation).');
    WriteLn('The FP32 sweet spot is why TestNeuralNumerical.pas uses eps = 1e-4.');
  finally
    NN.Free;
    Input.Free;
    Desired.Free;
  end;
end.
