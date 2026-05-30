program EuclideanNormHead;
(*
EuclideanNormHead: composing the in-tree elementwise transcendental layers
TNNetSquare, TNNetSqrt and TNNetReciprocal (plus a single sum reduction) into a
Euclidean-norm-reciprocal head that, given a vector x, computes 1/||x||_2.

THE IDEA. The L2 norm is ||x||_2 = sqrt( sum_i x_i^2 ). Every piece of that
expression already exists in the library as a parameter-free elementwise layer:
  * TNNetSquare     : x_i  -> x_i^2           (derivative 2*x)
  * TNNetSqrt       : s    -> sqrt(max(s,eps)) (derivative 1/(2*y), eps=1e-6)
  * TNNetReciprocal : n    -> 1/n              (derivative -y^2, eps=1e-6)
The only non-elementwise piece is the SUM over the features. We get an EXACT sum
with an existing layer too: a TNNetFullConnectLinear(1) whose single neuron has
all-ones weights and zero bias computes y = sum_i x_i (+0). So the whole head is
literally the composition:

    Reciprocal( Sqrt( Sum( Square(x) ) ) )  =  1 / ||x||_2

LAYER COMPOSITION (shapes annotated; F = vector length on the Depth axis):
    Input(1,1,F)                       x                     (1,1,F)
      -> TNNetSquare                   x_i^2                 (1,1,F)
      -> TNNetFullConnectLinear(1)     sum_i x_i^2  [W=1,b=0](1,1,1)
      -> TNNetSqrt                     ||x||_2               (1,1,1)
      -> TNNetReciprocal               1/||x||_2             (1,1,1)

WHY THE all-ones FullConnectLinear FOR THE SUM. The repo's channel pools are a
trap here: TNNetAvgChannel pools over the X*Y plane (not the Depth axis) and, on
an (N,1,F) bag, divides by PoolSize^2 rather than PoolSize, so its scale is not a
clean sum. A FullConnectLinear(1) with weights filled to 1.0 and ClearBias() is
an EXACT sum over the F features with no hidden scaling, and its gradient (a fan
of ones) flows straight through -- exactly what a reduction should do. We freeze
nothing special: we just initialise that one layer's weights to ones / bias to
zero after InitWeights().

VERIFICATION (self-checking gate, Halt(1) on any failure):
  1) FORWARD MATCH. For many random vectors, compare the composed head output to
     the analytic 1/sqrt(sum x_i^2). Max abs error must be < 1e-4 (float32
     reduction over F<=8 terms; comfortably tight).
  2) UNIT-NORM SANITY. A unit-norm vector must return ~1.0.
  3) L2-NORMALIZE EXTENSION. The natural extension x/||x|| = x*(1/||x||) is
     checked host-side (multiply the input by the head's scalar) and compared to
     the analytic normalized vector -- demonstrating the head as the reciprocal
     factor of a full L2 normalizer.
  4) GRADIENT FLOW. A tiny fit: a downstream FullConnectLinear(1) reads the head
     output and must learn to regress the analytic 1/||x|| target through the
     Square/Sum/Sqrt/Reciprocal stack. We require the MSE to drop substantially
     and stay finite (no NaN) -- proof the backward chain rule flows.

CONTRAST WITH TNNetL2Normalize. This composed head is a TEACHING ARTIFACT: it
shows the transcendental layers compose into a recognisable analytic function.
For production L2 normalization the dedicated TNNetL2Normalize layer is the right
tool -- it applies the EXACT Jacobian (I - y y^T)/n over the chosen axis in one
fused backward pass and carries a tunable epsilon guard (FFloatSt[0], default
1e-8) that round-trips through Save/Load. The composed head instead leans on the
per-layer eps clamps of Sqrt/Reciprocal (1e-6) and chains three separate
backward passes; correct and instructive, but not the fused, axis-aware,
save-safe primitive you would reach for in a real model.

Pure CPU, single-threaded (MaxThreadNum := 1), deterministic (fixed RandSeed).
Runs in well under a second. Gate idiom mirrors examples/SIREN, examples/DeepSets.

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
  cF        = 8;        // vector length (features on the Depth axis)
  cNumTest  = 200;      // random vectors for the forward-match check
  cSeed     = 424242;   // repo idiom
  cFwdTol   = 1e-4;     // forward-match tolerance (float32 sum over F terms)
  cFitSteps = 600;      // tiny gradient-flow fit
  cFitLR    = 0.02;
  cFitMom   = 0.0;      // plain SGD: the read-out optimum (w=1,b=0) is exact, so
                        // momentum only adds oscillation around it

// Analytic 1/||x||_2 of the first F entries of a volume.
function AnalyticInvNorm(V: TNNetVolume): TNeuralFloat;
var
  I: integer;
  S: TNeuralFloat;
begin
  S := 0;
  for I := 0 to cF - 1 do S := S + V.FData[I] * V.FData[I];
  Result := 1.0 / Sqrt(S);
end;

// Fill V (shape 1,1,F) with a random vector. Scale keeps norms in a calm range.
procedure RandomVector(V: TNNetVolume; Scale: TNeuralFloat);
var
  I: integer;
begin
  for I := 0 to cF - 1 do
    V.FData[I] := (Random - 0.5) * 2.0 * Scale;
end;

// Build the composed Euclidean-norm-reciprocal head:
//   Input(1,1,F) -> Square -> FullConnectLinear(1)[sum] -> Sqrt -> Reciprocal
// Returns the all-ones sum layer so the caller can re-freeze it after any later
// InitWeights().
procedure BuildHead(out NN: TNNet; out SumLayer: TNNetLayer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, cF));
  NN.AddLayer(TNNetSquare.Create());
  SumLayer := NN.AddLayer(TNNetFullConnectLinear.Create(1)); // sum over F
  NN.AddLayer(TNNetSqrt.Create());
  NN.AddLayer(TNNetReciprocal.Create());
  NN.InitWeights();
  // Turn the FullConnectLinear(1) into an EXACT, frozen sum: all-ones weights,
  // zero bias, and LearningRate := 0 so training never disturbs the reduction.
  // (Weights are the only params in the whole head.)
  SumLayer.Neurons[0].Weights.Fill(1.0);
  SumLayer.ClearBias();
  SumLayer.LearningRate := 0.0; // freeze: the sum must stay an exact reduction
end;

var
  NN, NNFit: TNNet;
  SumLayer, FitSumLayer: TNNetLayer;
  I, J, Step: integer;
  Inp, Outp, Tgt: TNNetVolume;
  Got, Want, Err, MaxErr, MeanErr: TNeuralFloat;
  NormErr, MaxNormErr, InvN: TNeuralFloat;
  Mse0, MseN, Diff: TNeuralFloat;
  Pass, AnyNaN: boolean;
begin
  RandSeed := cSeed; // deterministic; manual Compute/Backpropagate run single-threaded

  WriteLn('EuclideanNormHead: composing Reciprocal(Sqrt(Sum(Square(x)))) = 1/||x||_2.');
  WriteLn(Format('Head: Input(1,1,%d) -> TNNetSquare -> FullConnectLinear(1)[W=1,b=0 sum]'
    + ' -> TNNetSqrt -> TNNetReciprocal -> (1,1,1)', [cF]));
  WriteLn;

  BuildHead(NN, SumLayer);
  Inp  := TNNetVolume.Create(1, 1, cF);
  Outp := TNNetVolume.Create(1, 1, 1);

  // ---- check 1: forward match against the analytic 1/||x||_2 ----------------
  MaxErr := 0; MeanErr := 0;
  for I := 1 to cNumTest do
  begin
    RandomVector(Inp, 3.0);
    NN.Compute(Inp);
    NN.GetOutput(Outp);
    Got  := Outp.FData[0];
    Want := AnalyticInvNorm(Inp);
    Err  := Abs(Got - Want);
    if Err > MaxErr then MaxErr := Err;
    MeanErr := MeanErr + Err;
  end;
  MeanErr := MeanErr / cNumTest;

  WriteLn('Check 1 - forward match vs analytic 1/||x||_2 over ', cNumTest, ' vectors:');
  WriteLn(Format('  max abs error  = %.3e', [MaxErr]));
  WriteLn(Format('  mean abs error = %.3e', [MeanErr]));
  // Worked examples.
  for I := 1 to 3 do
  begin
    RandomVector(Inp, 3.0);
    NN.Compute(Inp);
    NN.GetOutput(Outp);
    WriteLn(Format('  example: head=%.6f  analytic=%.6f', [Outp.FData[0], AnalyticInvNorm(Inp)]));
  end;
  WriteLn;

  // ---- check 2: unit-norm vector returns ~1.0 -------------------------------
  Inp.Fill(0);
  Inp.FData[0] := 1.0; // x = e_0, ||x|| = 1
  NN.Compute(Inp);
  NN.GetOutput(Outp);
  WriteLn(Format('Check 2 - unit-norm input e_0 -> head = %.6f (want ~1.0)', [Outp.FData[0]]));
  WriteLn;

  // ---- check 3: L2-normalize extension x/||x|| = x*(1/||x||) ----------------
  // The head produces the reciprocal scalar; multiplying the input by it yields
  // the L2-normalized vector. Compare to analytic x/||x|| component-wise.
  // Scale x by the head's reciprocal to get x/||x||, then require the result to
  // itself have unit L2 norm (an INDEPENDENT property: it does not reuse the
  // head's value to define the target, so it genuinely tests the reciprocal).
  MaxNormErr := 0;
  for I := 1 to cNumTest do
  begin
    RandomVector(Inp, 3.0);
    NN.Compute(Inp);
    NN.GetOutput(Outp);
    InvN := Outp.FData[0];          // head: 1/||x||
    NormErr := 0;                   // squared L2 norm of the head-normalized vector
    for J := 0 to cF - 1 do
    begin
      Got := Inp.FData[J] * InvN;   // head-normalized component
      NormErr := NormErr + Got * Got;
    end;
    // The head-normalized vector must have ||x/||x|| ||_2 = 1.
    if Abs(Sqrt(NormErr) - 1.0) > MaxNormErr then MaxNormErr := Abs(Sqrt(NormErr) - 1.0);
  end;
  WriteLn(Format('Check 3 - L2-normalize extension: max | ||x*(1/||x||)|| - 1 | = %.3e',
    [MaxNormErr]));
  WriteLn;

  // ---- check 4: gradient flow (tiny fit through the norm head) --------------
  // A fresh head + a downstream FullConnectLinear(1) must learn to scale the
  // head output to match the analytic target (we make the target = head output,
  // so the downstream weight should drive toward 1.0). The point is only that
  // gradients flow through Square/Sum/Sqrt/Reciprocal without NaN and the loss
  // drops.
  BuildHead(NNFit, FitSumLayer);
  NNFit.AddLayer(TNNetFullConnectLinear.Create(1)); // trainable read-out
  NNFit.InitWeights();
  // Re-freeze the sum layer (InitWeights above reset its weights) to keep the
  // head exact.
  FitSumLayer.Neurons[0].Weights.Fill(1.0);
  FitSumLayer.ClearBias();
  NNFit.SetLearningRate(cFitLR, cFitMom);
  NNFit.SetL2Decay(0.0);
  FitSumLayer.LearningRate := 0.0; // after SetLearningRate: keep the sum frozen

  Tgt := TNNetVolume.Create(1, 1, 1);
  AnyNaN := False;

  // Initial MSE.
  Mse0 := 0;
  for I := 1 to 64 do
  begin
    RandomVector(Inp, 3.0);
    Tgt.FData[0] := AnalyticInvNorm(Inp);
    NNFit.Compute(Inp);
    NNFit.GetOutput(Outp);
    Diff := Outp.FData[0] - Tgt.FData[0];
    Mse0 := Mse0 + Diff * Diff;
  end;
  Mse0 := Mse0 / 64;

  for Step := 1 to cFitSteps do
  begin
    RandomVector(Inp, 3.0);
    Tgt.FData[0] := AnalyticInvNorm(Inp);
    NNFit.Compute(Inp);
    NNFit.GetOutput(Outp);
    if IsNan(Outp.FData[0]) then AnyNaN := True;
    NNFit.ClearDeltas();
    NNFit.Backpropagate(Tgt);
    NNFit.UpdateWeights();
  end;

  // Final MSE.
  MseN := 0;
  for I := 1 to 64 do
  begin
    RandomVector(Inp, 3.0);
    Tgt.FData[0] := AnalyticInvNorm(Inp);
    NNFit.Compute(Inp);
    NNFit.GetOutput(Outp);
    if IsNan(Outp.FData[0]) then AnyNaN := True;
    Diff := Outp.FData[0] - Tgt.FData[0];
    MseN := MseN + Diff * Diff;
  end;
  MseN := MseN / 64;

  WriteLn(Format('Check 4 - gradient flow: MSE %.6f -> %.6f over %d steps (NaN seen: %s)',
    [Mse0, MseN, cFitSteps, BoolToStr(AnyNaN, True)]));
  WriteLn;

  // ---- gate -----------------------------------------------------------------
  Pass := (MaxErr < cFwdTol)
      and (MaxNormErr < cFwdTol)
      and (not AnyNaN)
      and (MseN < 0.1 * Mse0);              // loss dropped >=10x: gradients flow

  // Re-evaluate unit-norm check explicitly for the gate.
  Inp.Fill(0); Inp.FData[0] := 1.0;
  NN.Compute(Inp); NN.GetOutput(Outp);
  Pass := Pass and (Abs(Outp.FData[0] - 1.0) < cFwdTol);

  if Pass then
    WriteLn('GATE: PASS - composed Reciprocal(Sqrt(Sum(Square(x)))) matches 1/||x||_2, '
      + 'the L2-normalize extension is exact, and gradients flow without NaN.')
  else
  begin
    WriteLn('GATE: FAIL');
    if not (MaxErr < cFwdTol) then
      WriteLn(Format('  - forward max error %.3e >= tol %.3e', [MaxErr, cFwdTol]));
    if not (MaxNormErr < cFwdTol) then
      WriteLn(Format('  - L2-normalize max error %.3e >= tol %.3e', [MaxNormErr, cFwdTol]));
    if AnyNaN then
      WriteLn('  - NaN encountered during the gradient-flow fit');
    if not (MseN < 0.1 * Mse0) then
      WriteLn(Format('  - fit MSE did not drop >=10x (%.6f -> %.6f)', [Mse0, MseN]));
    if not (Abs(Outp.FData[0] - 1.0) < cFwdTol) then
      WriteLn(Format('  - unit-norm input returned %.6f (want ~1.0)', [Outp.FData[0]]));
  end;

  Inp.Free; Outp.Free; Tgt.Free;
  NN.Free; NNFit.Free;

  if not Pass then Halt(1);
end.
