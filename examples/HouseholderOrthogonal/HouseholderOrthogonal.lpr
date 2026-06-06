program HouseholderOrthogonal;
(*
HouseholderOrthogonal: exact orthogonality kills exploding/vanishing
gradients in DEEP plain stacks.

TNNetHouseholderLinear parameterizes its n x n weight as a product of K
Householder reflections

    Q = H_1 * H_2 * ... * H_K,   H_i = I - 2 * (v_i v_i^T) / (v_i^T v_i),

so Q is EXACTLY orthogonal for ANY reflection vectors v_i -- no constrained
optimization, no re-projection. An orthogonal Jacobian is an isometry:
||Q*x|| = ||x|| (bias off) and ||Q^T*g|| = ||g||. Stack D of these with no
nonlinearity and the end-to-end Jacobian is still orthogonal, so a gradient
pushed back from the top arrives at the input with its norm UNCHANGED,
regardless of depth.

An unconstrained TNNetFullConnectLinear stack has no such guarantee: each
layer's singular values multiply, so the backward signal grows or shrinks
geometrically with depth and a deep plain stack explodes or collapses.

WHAT THIS PROGRAM SHOWS
-----------------------
For depths D = 1, 2, 4, 8, 16, 32 it builds TWO deep plain linear stacks of
width N (no activation, no normalization, no residuals):
  (A) D x TNNetHouseholderLinear  (exactly orthogonal blocks)
  (B) D x TNNetFullConnectLinear  (unconstrained dense blocks)
pushes one fixed input forward, plants a unit gradient at the top, backprops,
and prints the L2 norm of the gradient that reaches the INPUT. The orthogonal
stack holds its gradient norm ~constant across all depths; the unconstrained
stack drifts away from 1 geometrically (explodes or vanishes depending on the
random init).

It then sweeps the number of reflections K in {1, N/2, N} at a fixed depth to
illustrate the K vs cost/expressivity trade-off: K reflections cost O(K*n)
per layer and span a K-reflection sub-group of O(n); K = n gives the full
orthogonal group, K < n a cheaper, lower-rank-of-motion sub-group (still
exactly orthogonal). The Jacobian stays an isometry for every K, so the
gradient norm is preserved regardless of K -- K trades representational
reach for compute, NOT gradient stability.

CPU-only, no data files, runs in well under a minute.
*)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils,
  neuralnetwork, neuralvolume;

const
  Width = 16;                         // n
  Depths: array[0..5] of integer = (1, 2, 4, 8, 16, 32);

// Build a deep plain stack and return the L2 norm of the input-side gradient
// produced by a unit gradient planted at the top output. UseHouseholder
// selects orthogonal vs unconstrained blocks. NumRefl is K (ignored for the
// unconstrained stack).
function InputGradNorm(Depth: integer; UseHouseholder: boolean;
  NumRefl: integer): TNeuralFloat;
var
  NN: TNNet;
  X, Desired: TNNetVolume;
  d, i: integer;
begin
  NN := TNNet.Create();
  X := TNNetVolume.Create(1, 1, Width);
  Desired := TNNetVolume.Create(1, 1, Width);
  try
    NN.AddLayer(TNNetInput.Create(1, 1, Width, 1));
    for d := 0 to Depth - 1 do
    begin
      if UseHouseholder then
        NN.AddHouseholderLinear(Width, NumRefl, {UseBias=}False)
      else
        NN.AddLayer(TNNetFullConnectLinear.Create(Width, 1, 1, {SuppressBias=}1));
    end;
    // For the unconstrained stack, mildly scale every dense weight so its
    // spectral radius sits a touch above 1 -- a realistic init -- to expose the
    // geometric blow-up across depth that exact orthogonality avoids.
    if not UseHouseholder then
      for d := 1 to Depth do
        for i := 0 to NN.Layers[d].Neurons.Count - 1 do
          NN.Layers[d].Neurons[i].Weights.Mul(1.25);
    // Freeze weights: we measure the Jacobian at init, not after any update.
    NN.SetLearningRate(0.0, 0.0);

    // A fixed, reproducible input.
    for i := 0 to Width - 1 do
      X.Raw[i] := Sin(i * 0.7) * 1.3 + 0.2;
    NN.Compute(X);

    // Plant a unit gradient at the top: with half-MSE loss, output error =
    // (output - desired); set desired so the error is the all-ones vector
    // scaled to unit L2 norm.
    for i := 0 to Width - 1 do
      Desired.Raw[i] := NN.GetLastLayer.Output.Raw[i] - 1.0 / Sqrt(Width);
    NN.Layers[0].OutputError.Fill(0);
    NN.Backpropagate(Desired);

    Result := NN.Layers[0].OutputError.GetMagnitude();
  finally
    NN.Free;
    X.Free;
    Desired.Free;
  end;
end;

procedure DepthSweep();
var
  idx: integer;
  hNorm, fNorm: TNeuralFloat;
begin
  WriteLn('Top gradient norm planted = 1.0 (unit vector).');
  WriteLn('Input-side gradient norm vs depth (no activation, no norm):');
  WriteLn('');
  WriteLn('  depth | Householder (K=n, orthogonal) | FullConnectLinear (unconstrained)');
  WriteLn('  ------+-------------------------------+----------------------------------');
  for idx := 0 to High(Depths) do
  begin
    RandSeed := 770077; // same init family across depths for a fair comparison
    hNorm := InputGradNorm(Depths[idx], True, Width);
    RandSeed := 770077;
    fNorm := InputGradNorm(Depths[idx], False, 0);
    WriteLn(Format('  %5d |  %26.6f   |  %26.6f',
      [Depths[idx], hNorm, fNorm]));
  end;
  WriteLn('');
  WriteLn('The Householder column stays ~1.0 at every depth (orthogonal Jacobian =');
  WriteLn('isometry); the unconstrained column drifts away from 1 geometrically as');
  WriteLn('depth grows -> exploding/vanishing gradients.');
end;

procedure ReflectionSweep();
var
  Ks: array[0..2] of integer;
  i: integer;
  norm: TNeuralFloat;
begin
  Ks[0] := 1;
  Ks[1] := Width div 2;
  Ks[2] := Width;
  WriteLn('');
  WriteLn('K (#reflections) vs gradient norm at depth 32 (orthogonal for every K):');
  WriteLn('');
  WriteLn('     K | cost/layer | input gradient norm');
  WriteLn('  -----+------------+--------------------');
  for i := 0 to High(Ks) do
  begin
    RandSeed := 770077;
    norm := InputGradNorm(32, True, Ks[i]);
    WriteLn(Format('  %4d |  O(%2d * n)  |  %18.6f', [Ks[i], Ks[i], norm]));
  end;
  WriteLn('');
  WriteLn('Cost scales linearly in K; the gradient norm is preserved for ALL K');
  WriteLn('(every product of reflections is exactly orthogonal). K = n spans the');
  WriteLn('full orthogonal group O(n); K < n a cheaper sub-group with less');
  WriteLn('expressivity but the SAME exact gradient stability.');
end;

begin
  WriteLn('=== TNNetHouseholderLinear: exact orthogonality vs deep-stack gradients ===');
  WriteLn('width n = ', Width);
  WriteLn('');
  DepthSweep();
  ReflectionSweep();
  WriteLn('');
  WriteLn('Done.');
end.
