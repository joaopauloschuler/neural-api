program MuonOptimizer;
(*
MuonOptimizer: a self-contained demo of the Muon optimizer (Jordan et al. 2024,
https://kellerjordan.github.io/posts/muon/) on a tiny synthetic regression toy.
The Muon arm drives the LIBRARY optimizer path (TNNet.CalcMuonDelta ->
UpdateWeightsAdam, the same sequence TNeuralOptimizerMuon.Optimize runs inside
TNeuralFit) from a hand-rolled per-step training loop (no TNeuralFit), in the
same idiom as examples/SharpnessAwareMinimization/. The standalone NewtonSchulz5
below is retained ONLY for the headline orthogonality probe (a self-contained
demonstration of the orthogonalizer); the weight update itself is the real
facility, not hand-rolled surgery.

Muon, in one step, for EACH 2D weight matrix W of a dense layer:
  (1) momentum buffer  M <- mu*M + G        (G = accumulated gradient of W);
  (2) ORTHOGONALIZE the update: O <- NewtonSchulz5(M), i.e. replace M by the
      nearest orthogonal matrix (all singular values pushed to ~1) using ~5
      fixed quintic Newton-Schulz iterations on the Frobenius-normalized
      X = M / ||M||_F :
            X <- a*X + b*(X X^T X) + c*(X X^T)^2 X
      with the paper's coefficients (a,b,c) ~ (3.4445, -4.7750, 2.0315);
  (3) apply  W <- W - lr * sqrt(max(rows,cols)) * O .
The sqrt(max(rows,cols)) factor makes the per-element update RMS match Adam's,
so the SAME learning rate transfers across the three optimizers compared here.

WHY THIS IS GRADIENT SURGERY, NOT A REPARAMETRIZATION. Muon normalizes the
*update direction* each step; it never changes the forward pass. This is
distinct from the differentiable forward-weight normalizers in this library --
TNNetWeightNormLinear / TNNetWeightStandardization -- which reparametrize the
WEIGHTS the forward pass reads (and are trained through). Here W is an ordinary
dense weight matrix; we only intercept how its gradient is turned into a step.

THE REPO GRADIENT-SURGERY IDIOM (mirrors SharpnessAwareMinimization):
  * NN.SetBatchUpdate(True) is MANDATORY. In per-sample mode Backpropagate
    applies and zeroes the gradient immediately; only batch mode ACCUMULATES it
    into Neurons[].Delta (FDelta = -lr*grad), which is the tensor we read.
  * We read the gradient G straight out of Neurons[].Delta. The library stores
    the DESCENT step there (Delta = -lr*grad), so G = -Delta/lr; the lr cancels
    into our own step scale, so we work with Gloc = -Delta directly.
  * For a TNNetFullConnectLinear layer, neuron n owns row n of the weight matrix
    (Weights = FanIn values), exactly the (FanOut x FanIn) layout that
    TNNet.EstimateSpectralNorm assumes. We pack rows contiguously into a
    TNNetVolume and drive all matmuls through TNNetVolume.DotProducts
    (result[a,b] = dot(row a of A, row b of B) == (A B^T)[a,b]) -- no
    hand-rolled triple loops.

BAKE-OFF: the SAME tiny MLP is trained three times at matched lr / epochs /
seed -- plain SGD-momentum, Adam, and Muon -- on a noise-free 3-input -> scalar
regression. We chart loss-vs-step (printed columns) and wall-clock per arm.

HEADLINE CORRECTNESS SIGNAL (PASS/FAIL, Halt(1) on failure): after the
Newton-Schulz pass the orthogonalized update O is SEMI-orthogonal -- the
published 5-step quintic is an approximate orthogonalizer whose stable fixed
points are sigma ~ 0.868 and ~1.264 (f(1)=0.701, so sigma=1 is NOT a fixed
point). It squeezes every singular value into roughly [0.7, 1.3], which is all
Muon needs to make the update near-isotropic. We assert that band on a random
probe matrix: ||O^T O - I||_F is bounded (NOT ~0) and the top singular value,
cross-checked with TNNet.EstimateSpectralNorm, lands in [0.65, 1.35].

Pure CPU, no external data, tiny dims, well under a minute.

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

const
  cSeed       = 42;
  cInDim      = 3;     // 3 inputs
  cHiddenW    = 16;    // tiny hidden width
  cNumTrain   = 256;   // synthetic regression samples
  cBatchSize  = 32;
  cEpochs     = 40;
  cLearnRate  = 0.02;
  // Muon's update is already unit-RMS after orthogonalization and then blown up
  // by sqrt(max(rows,cols)) (~4 for the 16-wide hidden layers here), so the same
  // raw lr would take far larger steps than SGD/Adam. We give the Muon arm its
  // own (smaller) lr so the three arms take comparably sized steps -- this is
  // the lr that "transfers" once the sqrt scaling is accounted for.
  cMuonLR     = 0.005;
  cMomentum   = 0.9;   // shared by SGD-momentum and Muon's momentum buffer
  cAdamBeta1  = 0.9;
  cAdamBeta2  = 0.999;
  cAdamEps    = 1e-8;
  cNSIters    = 5;     // Newton-Schulz iterations
  // Paper's quintic Newton-Schulz coefficients.
  cNS_a       = 3.4445;
  cNS_b       = -4.7750;
  cNS_c       = 2.0315;

type
  TOptKind = (okSGDm, okAdam, okMuon);

  // Per-trainable-layer optimizer state (momentum / Adam moments), one volume
  // per neuron mirroring that neuron's Weights tensor.
  TLayerState = record
    LayerIdx: integer;
    M: array of TNNetVolume;   // momentum (SGDm/Muon) or 1st moment (Adam)
    V: array of TNNetVolume;   // Adam 2nd moment (unused otherwise)
  end;

// ===========================================================================
// Newton-Schulz orthogonalization of a packed (Rows x Cols) matrix.
// Matrices are stored row-major in a TNNetVolume (element [r,c] at r*Cols + c).
//
// All matmuls go through TNNetVolume.DotProducts. IMPORTANT layout note: that
// primitive stores its output as out[b*NumAs + a] = dot(VAs row a, VBs row b),
// i.e. as a (NumBs x NumAs) matrix. So to get an ordinary product P*Q with
// P (R x I) and Q (I x C) we feed VAs = Q^T (C x I), VBs = P (R x I),
// NumAs=C, NumBs=R, VectorSize=I; then out[r*C + c] = dot(Q^T row c, P row r)
// = dot(P row r, Q col c) = (P Q)[r,c]. MatMul wraps exactly that.
// On return Dst holds O = NewtonSchulz5(M), the nearest orthogonal matrix.
// ===========================================================================

// Transpose a packed (Rows x Cols) matrix into Dst (Cols x Rows).
procedure MatTranspose(Dst, Src: TNNetVolume; Rows, Cols: integer);
var
  R, C: integer;
begin
  Dst.ReSize(Cols * Rows, 1, 1);
  for R := 0 to Rows - 1 do
    for C := 0 to Cols - 1 do
      Dst.FData[C * Rows + R] := Src.FData[R * Cols + C];
end;

// Dst (R x C) := P (R x I) * Q (I x C). Qt scratch holds Q^T (C x I).
procedure MatMul(Dst, P, Q, Qt: TNNetVolume; R, I, C: integer);
begin
  MatTranspose(Qt, Q, I, C);            // Qt: C x I
  Dst.ReSize(R * C, 1, 1);
  Dst.Fill(0);                          // DotProducts accumulates; clear first.
  // out[r*C + c] = dot(Qt row c [len I], P row r [len I]) = (P Q)[r,c].
  Dst.DotProducts({NumAs=}C, {NumBs=}R, {VectorSize=}I, {VAs=}Qt, {VBs=}P);
end;

// O := NewtonSchulz5(M), both packed (Rows x Cols).
procedure NewtonSchulz5(O, M: TNNetVolume; Rows, Cols: integer);
var
  X, Xt, A, AX, A2X, B, Qt: TNNetVolume;
  FroNorm: TNeuralFloat;
  Iter: integer;
begin
  X   := TNNetVolume.Create();  // current iterate (Rows x Cols)
  Xt  := TNNetVolume.Create();  // X^T (Cols x Rows)
  A   := TNNetVolume.Create();  // A = X X^T (Rows x Rows, symmetric)
  AX  := TNNetVolume.Create();  // A X   (Rows x Cols)
  A2X := TNNetVolume.Create();  // A^2 X (Rows x Cols)
  B   := TNNetVolume.Create();  // accumulator a*X + b*AX + c*A2X
  Qt  := TNNetVolume.Create();  // transpose scratch for MatMul
  try
    // X <- M / (||M||_F + eps).
    X.Copy(M);
    FroNorm := Sqrt(X.GetSumSqr());
    if FroNorm < 1e-12 then FroNorm := 1e-12;
    X.Mul(1.0 / FroNorm);

    for Iter := 1 to cNSIters do
    begin
      // A = X X^T  (Rows x Rows) = X (Rows x Cols) * X^T (Cols x Rows).
      MatTranspose(Xt, X, Rows, Cols);          // Xt = X^T (Cols x Rows)
      MatMul(A, X, Xt, Qt, Rows, Cols, Rows);   // A (Rows x Rows)

      // AX = A X  (Rows x Cols).
      MatMul(AX, A, X, Qt, Rows, Rows, Cols);

      // A2X = A (A X)  (Rows x Cols).  (No aliasing: Dst, P=A, Q=AX all distinct.)
      MatMul(A2X, A, AX, Qt, Rows, Rows, Cols);

      // X_next = a*X + b*(A X) + c*(A^2 X).
      B.Copy(X);
      B.Mul(cNS_a);
      B.MulAdd(cNS_b, AX);
      B.MulAdd(cNS_c, A2X);
      X.Copy(B);
    end;

    O.Copy(X);
  finally
    X.Free; Xt.Free; A.Free; AX.Free; A2X.Free; B.Free; Qt.Free;
  end;
end;

// ===========================================================================
// Synthetic regression: y = sin(x0) + 0.5*x1*x2 + 0.3*x0  on inputs in [-1,1].
// Deterministic, noise-free, identical across arms (built once after seeding).
// ===========================================================================
function MakeRegression(N: integer): TNNetVolumePairList;
var
  I: integer;
  x0, x1, x2, yt: TNeuralFloat;
  X, Y: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for I := 0 to N - 1 do
  begin
    x0 := 2 * Random - 1;
    x1 := 2 * Random - 1;
    x2 := 2 * Random - 1;
    yt := Sin(x0) + 0.5 * x1 * x2 + 0.3 * x0;
    X := TNNetVolume.Create(cInDim, 1, 1);
    X.FData[0] := x0; X.FData[1] := x1; X.FData[2] := x2;
    Y := TNNetVolume.Create(1, 1, 1);
    Y.FData[0] := yt;
    Result.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// Tiny MLP regressor: 3 -> hidden(ReLU) -> hidden(ReLU) -> 1(linear).
procedure BuildRegressor(out NN: TNNet);
begin
  // pSuppressBias=1: no bias on any layer. Muon's paper applies the
  // orthogonalization only to 2D weight matrices and routes vector params (like
  // biases) to a scalar optimizer; the library's only public bias accessor is
  // read-only (Neuron.Bias), so rather than reach into the private bias-gradient
  // field we simply drop biases on ALL arms -- keeping the three-way comparison
  // apples-to-apples and letting Muon fully own every trainable tensor.
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cInDim, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenW, {pSuppressBias=}1));
  NN.AddLayer(TNNetFullConnectReLU.Create(cHiddenW, {pSuppressBias=}1));
  NN.AddLayer(TNNetFullConnectLinear.Create(1, {pSuppressBias=}1));
  // lr/momentum here only matter for arms that call NN.UpdateWeights (SGDm).
  // Muon and Adam compute their own step and write Weights directly, but we
  // still need BatchUpdate so Backpropagate ACCUMULATES the gradient.
  NN.SetLearningRate(cLearnRate, cMomentum);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);   // CRITICAL: populate Neurons[].Delta (the gotcha)
end;

// Trainable dense-layer indices (neurons own a non-empty weight tensor).
procedure CollectTrainable(NN: TNNet; out Idx: array of integer; out Cnt: integer);
var
  L: integer;
begin
  Cnt := 0;
  for L := 0 to NN.GetLastLayerIdx() do
  begin
    if NN.Layers[L].Neurons.Count = 0 then Continue;
    if NN.Layers[L].Neurons[0].Weights = nil then Continue;
    if NN.Layers[L].Neurons[0].Weights.Size = 0 then Continue;
    Idx[Cnt] := L;
    Inc(Cnt);
  end;
end;

// Allocate optimizer state (moment buffers) mirroring each trainable neuron.
procedure InitStates(NN: TNNet; const Idx: array of integer; Cnt: integer;
  out States: array of TLayerState);
var
  T, NIdx: integer;
begin
  for T := 0 to Cnt - 1 do
  begin
    States[T].LayerIdx := Idx[T];
    SetLength(States[T].M, NN.Layers[Idx[T]].Neurons.Count);
    SetLength(States[T].V, NN.Layers[Idx[T]].Neurons.Count);
    for NIdx := 0 to NN.Layers[Idx[T]].Neurons.Count - 1 do
    begin
      States[T].M[NIdx] := TNNetVolume.Create();
      States[T].M[NIdx].Copy(NN.Layers[Idx[T]].Neurons[NIdx].Weights);
      States[T].M[NIdx].Fill(0);
      States[T].V[NIdx] := TNNetVolume.Create();
      States[T].V[NIdx].Copy(NN.Layers[Idx[T]].Neurons[NIdx].Weights);
      States[T].V[NIdx].Fill(0);
    end;
  end;
end;

procedure FreeStates(var States: array of TLayerState; Cnt: integer);
var
  T, NIdx: integer;
begin
  for T := 0 to Cnt - 1 do
    for NIdx := 0 to High(States[T].M) do
    begin
      States[T].M[NIdx].Free;
      States[T].V[NIdx].Free;
    end;
end;

// Apply one Adam step to every trainable neuron (weights + bias use the same
// per-element rule). G = -Delta (library stores Delta = -lr*grad, but for Adam
// the magnitude is rescaled by the moments, so we use grad = G/lr; the constant
// lr cancels in m/sqrt(v) up to the bias-corrected step, so we read grad
// directly as -Delta/lr).
procedure AdamStep(NN: TNNet; var States: array of TLayerState; Cnt: integer;
  StepNo: integer);
var
  T, NIdx, W: integer;
  Neuron: TNNetNeuron;
  Mv, Vv, Dv: TNNetVolume;
  g, mhat, vhat, b1c, b2c: TNeuralFloat;
begin
  b1c := 1 - Power(cAdamBeta1, StepNo);
  b2c := 1 - Power(cAdamBeta2, StepNo);
  for T := 0 to Cnt - 1 do
    for NIdx := 0 to High(States[T].M) do
    begin
      Neuron := NN.Layers[States[T].LayerIdx].Neurons[NIdx];
      Mv := States[T].M[NIdx];
      Vv := States[T].V[NIdx];
      Dv := Neuron.Delta;
      for W := 0 to Dv.Size - 1 do
      begin
        g := -Dv.FData[W] / cLearnRate;           // gradient of this weight
        Mv.FData[W] := cAdamBeta1 * Mv.FData[W] + (1 - cAdamBeta1) * g;
        Vv.FData[W] := cAdamBeta2 * Vv.FData[W] + (1 - cAdamBeta2) * g * g;
        mhat := Mv.FData[W] / b1c;
        vhat := Vv.FData[W] / b2c;
        Neuron.Weights.FData[W] := Neuron.Weights.FData[W]
          - cLearnRate * mhat / (Sqrt(vhat) + cAdamEps);
      end;
    end;
end;

// Apply one Muon step using the LIBRARY optimizer path
// (TNNet.CalcMuonDelta -> UpdateWeightsAdam), exactly what
// TNeuralOptimizerMuon.Optimize drives inside TNeuralFit. CalcMuonDelta builds
// each layer's momentum buffer (the neurons' FBackInertia, zeroed by
// ClearInertia at setup), Newton-Schulz-orthogonalizes the FanOut x FanIn
// matrix, and rewrites every neuron's Delta to -lr*sqrt(max(rows,cols))*O; the
// layer learning rate (set to cMuonLR for this arm) supplies lr. This replaces
// the former hand-rolled per-neuron weight surgery with the real facility.
procedure MuonStep(NN: TNNet);
begin
  NN.CalcMuonDelta(cMomentum, cNSIters);
  NN.UpdateWeightsAdam();
end;

// Mean-squared-error over the dataset.
function MeanMSE(NN: TNNet; Pairs: TNNetVolumePairList): TNeuralFloat;
var
  I: integer;
  Sum, d: TNeuralFloat;
begin
  Sum := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    d := NN.GetLastLayer().Output.FData[0] - Pairs[I].O.FData[0];
    Sum := Sum + d * d;
  end;
  if Pairs.Count > 0 then Result := Sum / Pairs.Count else Result := 0;
end;

procedure ShuffleIndices(var Idx: array of integer);
var
  I, J, Tmp: integer;
begin
  for I := High(Idx) downto 1 do
  begin
    J := Random(I + 1);
    Tmp := Idx[I]; Idx[I] := Idx[J]; Idx[J] := Tmp;
  end;
end;

// Train one arm for cEpochs, recording end-of-epoch train MSE. Returns the
// per-epoch loss curve and the wall-clock seconds in WallSec.
function TrainArm(Train: TNNetVolumePairList; Opt: TOptKind;
  out Curve: array of TNeuralFloat; out WallSec: TNeuralFloat): TNNet;
var
  NN: TNNet;
  Idx: array of integer;
  States: array of TLayerState;
  Order: array of integer;
  Cnt, Epoch, Lo, Hi, I, StepNo: integer;
  StartT: TDateTime;
begin
  RandSeed := cSeed;                 // identical init + shuffle order per arm
  BuildRegressor(NN);
  SetLength(Idx, NN.GetLastLayerIdx() + 1);
  CollectTrainable(NN, Idx, Cnt);
  SetLength(States, Cnt);
  InitStates(NN, Idx, Cnt, States);

  if Opt = okMuon then
  begin
    // Muon reads its learning rate from the layers and its momentum buffer from
    // each neuron's FBackInertia; set the (smaller) Muon lr and zero the buffer.
    NN.SetLearningRate(cMuonLR, cMomentum);
    NN.ClearInertia();
  end;

  SetLength(Order, Train.Count);
  for I := 0 to High(Order) do Order[I] := I;

  StepNo := 0;
  StartT := Now;
  for Epoch := 1 to cEpochs do
  begin
    ShuffleIndices(Order);
    Lo := 0;
    while Lo < Train.Count do
    begin
      Hi := Lo + cBatchSize;
      if Hi > Train.Count then Hi := Train.Count;

      // Accumulate the batch gradient into Neurons[].Delta.
      NN.ClearDeltas();
      for I := Lo to Hi - 1 do
      begin
        NN.Compute(Train[Order[I]].I);
        NN.Backpropagate(Train[Order[I]].O);
      end;
      Inc(StepNo);

      case Opt of
        okSGDm: NN.UpdateWeights();          // library SGD+momentum step
        okAdam: AdamStep(NN, States, Cnt, StepNo);
        okMuon: MuonStep(NN);
      end;

      Lo := Hi;
    end;
    Curve[Epoch - 1] := MeanMSE(NN, Train);
  end;
  WallSec := (Now - StartT) * 24 * 60 * 60;

  FreeStates(States, Cnt);
  Result := NN;
end;

// ---------------------------------------------------------------------------
// Headline orthogonality check: build a random (Rows x Cols) probe matrix,
// orthogonalize it, and assert ||O^T O - I||_F is tiny (singular values ~1).
// ---------------------------------------------------------------------------
function OrthoErrorFrobenius(Rows, Cols: integer; out TopSigma: TNeuralFloat): TNeuralFloat;
var
  M, O, Ot, Gram, Qt: TNNetVolume;
  ProbeNN: TNNet;
  ProbeLayer: TNNetFullConnectLinear;
  R, C, K, Inner: integer;
  Sum, diff, expected: TNeuralFloat;
begin
  M := TNNetVolume.Create(Rows * Cols, 1, 1);
  O := TNNetVolume.Create();
  Ot := TNNetVolume.Create();
  Gram := TNNetVolume.Create();
  Qt := TNNetVolume.Create();
  try
    for K := 0 to Rows * Cols - 1 do M.FData[K] := RandG(0, 1);

    NewtonSchulz5(O, M, Rows, Cols);

    // Gram = O^T O if Rows>=Cols else O O^T; either way the nonzero singular
    // values are 1, so we test the smaller Gram (Inner = min dim) against I.
    MatTranspose(Ot, O, Rows, Cols);            // Ot = O^T (Cols x Rows)
    if Rows >= Cols then
    begin
      // O^T O  (Cols x Cols) = O^T (Cols x Rows) * O (Rows x Cols).
      MatMul(Gram, Ot, O, Qt, Cols, Rows, Cols);
      Inner := Cols;
    end
    else
    begin
      // O O^T  (Rows x Rows) = O (Rows x Cols) * O^T (Cols x Rows).
      MatMul(Gram, O, Ot, Qt, Rows, Cols, Rows);
      Inner := Rows;
    end;

    Sum := 0;
    for R := 0 to Inner - 1 do
      for C := 0 to Inner - 1 do
      begin
        if R = C then expected := 1.0 else expected := 0.0;
        diff := Gram.FData[R * Inner + C] - expected;
        Sum := Sum + diff * diff;
      end;
    Result := Sqrt(Sum);

    // Cross-check top singular value with the power-iteration helper: stuff O's
    // rows into a dense layer and call EstimateSpectralNorm (~1 when orthogonal).
    ProbeNN := TNNet.Create();
    try
      ProbeNN.AddLayer(TNNetInput.Create(Cols, 1, 1));
      ProbeLayer := TNNetFullConnectLinear.Create(Rows);
      ProbeNN.AddLayer(ProbeLayer);
      for R := 0 to Rows - 1 do
        for C := 0 to Cols - 1 do
          ProbeLayer.Neurons[R].Weights.FData[C] := O.FData[R * Cols + C];
      TopSigma := TNNet.EstimateSpectralNorm(ProbeLayer, 50);
    finally
      ProbeNN.Free;
    end;
  finally
    M.Free; O.Free; Ot.Free; Gram.Free; Qt.Free;
  end;
end;

function Bar(V, Lo, Hi: TNeuralFloat; Width: integer): string;
var N: integer;
begin
  if Hi - Lo < 1e-12 then Hi := Lo + 1e-12;
  N := Round((V - Lo) / (Hi - Lo) * Width);
  if N < 0 then N := 0;
  if N > Width then N := Width;
  Result := StringOfChar('#', N);
end;

// ===========================================================================
var
  Train: TNNetVolumePairList;
  NNsgd, NNadam, NNmuon: TNNet;
  CurveSGD, CurveAdam, CurveMuon: array of TNeuralFloat;
  WallSGD, WallAdam, WallMuon: TNeuralFloat;
  OrthoErr, TopSigma: TNeuralFloat;
  E: integer;
  MaxC, MinC: TNeuralFloat;
  StartT: TDateTime;

begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  StartT := Now;
  DefaultFormatSettings.DecimalSeparator := '.';

  WriteLn('========================================================================');
  WriteLn('Muon optimizer (Newton-Schulz orthogonalized momentum, Jordan et al. 2024)');
  WriteLn('========================================================================');
  WriteLn(Format('Model:   %d -> %d(ReLU) -> %d(ReLU) -> 1(linear) regressor.',
    [cInDim, cHiddenW, cHiddenW]));
  WriteLn(Format('Data:    %d synthetic samples, batch=%d, %d epochs, lr=%.3f, mu=%.2f, seed=%d.',
    [cNumTrain, cBatchSize, cEpochs, cLearnRate, cMomentum, cSeed]));
  WriteLn(Format('Muon:    lr=%.3f, %d Newton-Schulz iters, coeffs (a,b,c)=(%.4f, %.4f, %.4f).',
    [cMuonLR, cNSIters, cNS_a, cNS_b, cNS_c]));
  WriteLn;

  // ----------------- headline orthogonality check FIRST -------------------
  // (cheap, and a hard gate before we bother with the bake-off).
  WriteLn('------------------------------------------------------------------------');
  WriteLn('Headline check: Newton-Schulz drives the singular values into ~[0.7,1.3].');
  WriteLn('------------------------------------------------------------------------');
  // The published 5-step quintic is a DELIBERATELY approximate orthogonalizer:
  // its stable fixed points are sigma ~ 0.868 and ~1.264 (f(1)=0.701 != 1), so
  // it produces a SEMI-orthogonal update with every singular value squeezed into
  // roughly [0.7, 1.3] rather than exactly 1 -- which is all Muon needs (the
  // update directions are made near-isotropic). We therefore assert that band
  // (top sigma in [0.65, 1.35]) and that ||O^T O - I||_F is bounded, NOT ~0.
  RandSeed := cSeed;
  OrthoErr := OrthoErrorFrobenius(cHiddenW, cInDim, TopSigma);
  WriteLn(Format('  probe matrix %dx%d:  ||O^T O - I||_F = %.6e   top sigma = %.6f',
    [cHiddenW, cInDim, OrthoErr, TopSigma]));
  WriteLn('  (exact orthogonality => 0 and 1; the 5-step quintic lands in a band.)');
  if (OrthoErr < 1.2) and (TopSigma > 0.65) and (TopSigma < 1.35) then
    WriteLn('  orthogonality (semi, Muon band): PASS')
  else
  begin
    WriteLn('  orthogonality: FAIL');
    Halt(1);
  end;
  WriteLn;

  // Build data once after seeding (identical across all three arms).
  RandSeed := cSeed;
  Train := MakeRegression(cNumTrain);

  SetLength(CurveSGD,  cEpochs);
  SetLength(CurveAdam, cEpochs);
  SetLength(CurveMuon, cEpochs);

  NNsgd  := TrainArm(Train, okSGDm, CurveSGD,  WallSGD);
  NNadam := TrainArm(Train, okAdam, CurveAdam, WallAdam);
  NNmuon := TrainArm(Train, okMuon, CurveMuon, WallMuon);

  // ----------------------- bake-off table ---------------------------------
  WriteLn('------------------------------------------------------------------------');
  WriteLn('Bake-off: train MSE vs epoch (lower is better). Same seed / data / epochs;');
  WriteLn('Muon uses its own lr (see header) to match step size after sqrt-scaling.');
  WriteLn('------------------------------------------------------------------------');
  WriteLn('  epoch    SGD-momentum         Adam            Muon');
  for E := 0 to cEpochs - 1 do
    if (E < 5) or (E mod 5 = 4) or (E = cEpochs - 1) then
      WriteLn(Format('  %5d   %12.6f   %12.6f   %12.6f',
        [E + 1, CurveSGD[E], CurveAdam[E], CurveMuon[E]]));
  WriteLn;

  WriteLn(Format('Final train MSE:  SGD-momentum=%.6f   Adam=%.6f   Muon=%.6f',
    [CurveSGD[cEpochs-1], CurveAdam[cEpochs-1], CurveMuon[cEpochs-1]]));
  WriteLn(Format('Wall-clock (s):   SGD-momentum=%.3f      Adam=%.3f      Muon=%.3f',
    [WallSGD, WallAdam, WallMuon]));
  WriteLn(Format('Per-step (ms):    SGD-momentum=%.4f      Adam=%.4f      Muon=%.4f',
    [WallSGD / (cEpochs * Ceil(cNumTrain / cBatchSize)) * 1000,
     WallAdam / (cEpochs * Ceil(cNumTrain / cBatchSize)) * 1000,
     WallMuon / (cEpochs * Ceil(cNumTrain / cBatchSize)) * 1000]));
  WriteLn;

  // ASCII loss curves (log-ish via shared min/max across all arms).
  MinC := 1e30; MaxC := -1e30;
  for E := 0 to cEpochs - 1 do
  begin
    MinC := Min(MinC, Min(CurveSGD[E], Min(CurveAdam[E], CurveMuon[E])));
    MaxC := Max(MaxC, Max(CurveSGD[E], Max(CurveAdam[E], CurveMuon[E])));
  end;
  WriteLn(Format('train-MSE vs epoch  [%.4f .. %.4f]  (S=SGDm A=Adam M=Muon):', [MinC, MaxC]));
  for E := 0 to cEpochs - 1 do
    if (E mod 5 = 4) or (E = cEpochs - 1) then
    begin
      WriteLn(Format('  e%3d S |%s', [E + 1, Bar(CurveSGD[E],  MinC, MaxC, 50)]));
      WriteLn(Format('       A |%s', [Bar(CurveAdam[E], MinC, MaxC, 50)]));
      WriteLn(Format('       M |%s', [Bar(CurveMuon[E], MinC, MaxC, 50)]));
    end;
  WriteLn;

  WriteLn(Format('Total wall time: %.2f s', [(Now - StartT) * 24 * 60 * 60]));
  WriteLn('Done. Orthogonality check PASSED.');

  NNsgd.Free;
  NNadam.Free;
  NNmuon.Free;
  Train.Free;
end.
