program PredictiveCoding;
(*
PredictiveCoding: a backprop-free Predictive Coding Network (PCN) trained by
local inference relaxation + a purely local Hebbian weight rule, on a pure-CPU
toy. The value/error math is implemented DIRECTLY with TNNetVolume arithmetic
(no new core layer is added; this is an example, like ForwardForward /
LotteryTicket).

What makes this example UNIQUE in the tree:
  This is a SECOND biologically-plausible, backprop-free learning paradigm,
  DISTINCT from the Forward-Forward example. Where Forward-Forward replaces
  forward+backward with TWO forward passes and a per-layer goodness contrast,
  Predictive Coding keeps a single feedforward generative model but learns by
  ITERATIVELY SETTLING explicit per-layer VALUE nodes so that local
  PREDICTION ERRORS shrink, then takes ONE local Hebbian weight step. No global
  loss is ever backpropagated through the stack; every gradient is computed from
  the two ADJACENT layers only.

The model (Rao & Ballard 1999; Whittington & Bogacz 2017 link to backprop):
  A small MLP with layers 0..L (0 = input, L = output). Each layer l holds a
  VALUE vector x_l. Layer l "predicts" the value of layer l-1 (top-down
  generative direction) through  mu_{l-1} = W_l * act(x_l). The local
  PREDICTION ERROR at layer l-1 is
        e_{l-1} = x_{l-1} - mu_{l-1}.
  The total energy is  E = sum_l ||e_l||^2  (sum of squared prediction errors).

Training alternates TWO phases per example:
  1. INFERENCE RELAXATION (clamp both ends): clamp x_0 = input and x_L = label
     (one-hot). Iterate a few fixed-point gradient steps on the FREE hidden
     value nodes x_l to MINIMISE E. The gradient of E w.r.t. a free hidden
     value x_l combines its OWN error (it is the target of the error e_l) and
     the error it generates below (it predicts x_{l-1}):
        dE/dx_l = e_l  -  act'(x_l) .* (W_l^T * e_{l-1}).
     A handful of small steps  x_l -= beta * dE/dx_l  settle the values so the
     errors flow locally between adjacent layers only. We PRINT E per sweep for
     one example to prove the relaxation actually descends.
  2. LOCAL WEIGHT UPDATE (Hebbian): once values have settled, each W_l is
     updated by the OUTER PRODUCT of the error it explains and the activation
     that produced the prediction -- a purely LOCAL rule using only adjacent
     nodes:
        dW_l = e_{l-1} (x) act(x_l)^T,   W_l += lr * dW_l.
     There is NO chained backward pass; W_2's update never sees W_1's Jacobian.

Inference (prediction) on a NEW input: clamp x_0 = input, leave the top free,
run the SAME relaxation (no label clamp) and read the argmax of the settled
top value x_L. This is the standard PCN read-out.

Headline (side-by-side, in ONE program):
  - A plain BACKPROP MLP of the SAME shape is trained with the normal TNNet API,
    and the PCN's held-out accuracy is reported next to it -> COMPARABLE.
  - The per-sweep energy trace for a sample is printed -> E DECREASES, proving
    the inference relaxation settles.

Built-in correctness gates (Halt(1) on failure, DoubleDescent house style):
  GATE 1: the relaxation energy E must DECREASE over the sweeps (settled).
  GATE 2: PCN held-out accuracy must BEAT CHANCE by a clear margin AND be in the
          same ballpark as the backprop baseline.

Pure CPU, no external data, single-threaded + fixed RandSeed (deterministic).
Tiny task/net -> finishes in a few seconds, well under the few-minute budget.

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
  cClasses  = 3;        // number of blob classes (chance = 1/3)
  cFeat     = 2;        // raw feature dimension (2D blobs)
  cHidden   = 16;       // hidden layer width
  cTrain    = 300;      // training points
  cTest     = 300;      // held-out test points
  cEpochs   = 80;       // passes over the training set
  cSweeps   = 30;       // inference-relaxation fixed-point steps per example
  cBeta     = 0.30;     // relaxation (value-node) step size
  cLR       = 0.05;     // local Hebbian weight learning rate
  cBlobSpread = 0.55;   // blob std-dev
  cSeed     = 424242;

  // Network value-node sizes (3 layers: input -> hidden -> output).
  // Layer 0 = input (cFeat), Layer 1 = hidden (cHidden), Layer 2 = output (cClasses).
  cL        = 2;        // top layer index (L); layers are 0..cL

type
  TSample = record
    Feat : array[0..cFeat - 1] of TNeuralFloat;
    Cls  : integer;
  end;
  TSampleArr = array of TSample;

var
  Centres: array[0..cClasses - 1, 0..cFeat - 1] of TNeuralFloat;
  Sizes  : array[0..cL] of integer;          // value-node sizes per layer
  // Value, error and activation buffers per layer.
  Xv     : array[0..cL] of TNNetVolume;      // value nodes x_l
  Av     : array[0..cL] of TNNetVolume;      // act(x_l)
  Ev     : array[0..cL] of TNNetVolume;      // prediction errors e_l (e_0..e_{cL-1} used)
  // Weights W_l predict layer l-1 from act(x_l). W[l] has Sizes[l-1] rows x Sizes[l] cols.
  // Stored flat row-major in a TNNetVolume of size Sizes[l-1]*Sizes[l].
  Wm     : array[1..cL] of TNNetVolume;
  Bv     : array[1..cL] of TNNetVolume;      // bias for prediction of layer l-1

// ---------------------------------------------------------------------------
// Data: cClasses Gaussian blobs in cFeat-D space.
// ---------------------------------------------------------------------------
procedure MakeCentres;
var C, F: integer;
begin
  for C := 0 to cClasses - 1 do
    for F := 0 to cFeat - 1 do
      Centres[C, F] := RandG(0, 2.0);
end;

procedure BuildSet(out S: TSampleArr; Count: integer);
var I, F, C: integer;
begin
  SetLength(S, Count);
  for I := 0 to Count - 1 do
  begin
    C := Random(cClasses);
    S[I].Cls := C;
    for F := 0 to cFeat - 1 do
      S[I].Feat[F] := Centres[C, F] + RandG(0, cBlobSpread);
  end;
end;

// ---------------------------------------------------------------------------
// Activation: tanh (smooth, has a clean derivative act'(x)=1-tanh(x)^2).
// Input layer is treated linearly (its value is the clamped data).
// ---------------------------------------------------------------------------
function ActFwd(x: TNeuralFloat): TNeuralFloat; inline;
begin
  Result := Tanh(x);
end;

function ActDeriv(x: TNeuralFloat): TNeuralFloat; inline;
var t: TNeuralFloat;
begin
  t := Tanh(x);
  Result := 1.0 - t * t;
end;

// Fill Av[l] = act(Xv[l]). Layer 0 (input) is linear: Av[0] = Xv[0].
procedure ComputeAct(l: integer);
var j: integer;
begin
  if l = 0 then
    Av[0].Copy(Xv[0])
  else
    for j := 0 to Sizes[l] - 1 do
      Av[l].FData[j] := ActFwd(Xv[l].FData[j]);
end;

// mu_{l-1} = W_l * Av[l] + b_l ; e_{l-1} = X_{l-1} - mu_{l-1}.
// Stores the error into Ev[l-1].
procedure ComputeError(l: integer);
var r, c, rows, cols: integer; mu: TNeuralFloat;
begin
  rows := Sizes[l - 1];
  cols := Sizes[l];
  for r := 0 to rows - 1 do
  begin
    mu := Bv[l].FData[r];
    for c := 0 to cols - 1 do
      mu := mu + Wm[l].FData[r * cols + c] * Av[l].FData[c];
    Ev[l - 1].FData[r] := Xv[l - 1].FData[r] - mu;
  end;
end;

// Total energy E = sum over predicted layers of ||e||^2.
function TotalEnergy: TNeuralFloat;
var l: integer;
begin
  Result := 0;
  for l := 1 to cL do
    Result := Result + Ev[l - 1].GetSumSqr();
end;

// ---------------------------------------------------------------------------
// One inference-relaxation sweep over the FREE hidden value nodes (1..cL-1).
// Clamped: layer 0 always; layer cL only when ClampTop. Updates each free x_l:
//   dE/dx_l = e_l - act'(x_l) .* (W_l^T * e_{l-1})
//   x_l    -= beta * dE/dx_l
// Errors are recomputed first so they reflect the current values.
// ---------------------------------------------------------------------------
procedure RelaxSweep(ClampTop: boolean);
var l, r, c, rows, cols, topFree: integer; back, grad: TNeuralFloat;
begin
  // Refresh activations + errors for the whole stack.
  for l := 0 to cL do ComputeAct(l);
  for l := 1 to cL do ComputeError(l);

  if ClampTop then topFree := cL - 1 else topFree := cL;

  for l := 1 to topFree do
  begin
    rows := Sizes[l - 1];   // size of e_{l-1}
    cols := Sizes[l];       // size of x_l
    for c := 0 to cols - 1 do
    begin
      // W_l^T * e_{l-1} at component c = sum_r W[r,c] * e_{l-1}[r]
      back := 0;
      for r := 0 to rows - 1 do
        back := back + Wm[l].FData[r * cols + c] * Ev[l - 1].FData[r];
      // e_l[c] is this layer's own error (its prediction from above); for the
      // top free layer when top is NOT clamped, e_cL is undefined (no layer
      // above), so treat its own-error term as zero.
      if l < cL then
        grad := Ev[l].FData[c] - ActDeriv(Xv[l].FData[c]) * back
      else
        grad := - ActDeriv(Xv[l].FData[c]) * back;
      Xv[l].FData[c] := Xv[l].FData[c] - cBeta * grad;
    end;
  end;
end;

// ---------------------------------------------------------------------------
// Local Hebbian weight update after relaxation has settled:
//   dW_l = e_{l-1} (x) act(x_l)^T ;  W_l += lr * dW_l ;  b_l += lr * e_{l-1}
// Purely local: only adjacent nodes (the error it explains, the activation
// that produced the prediction). No chained backward pass.
// ---------------------------------------------------------------------------
procedure LocalWeightUpdate;
var l, r, c, rows, cols: integer; er: TNeuralFloat;
begin
  for l := 0 to cL do ComputeAct(l);
  for l := 1 to cL do ComputeError(l);
  for l := 1 to cL do
  begin
    rows := Sizes[l - 1];
    cols := Sizes[l];
    for r := 0 to rows - 1 do
    begin
      er := Ev[l - 1].FData[r];
      for c := 0 to cols - 1 do
        Wm[l].FData[r * cols + c] := Wm[l].FData[r * cols + c]
          + cLR * er * Av[l].FData[c];
      Bv[l].FData[r] := Bv[l].FData[r] + cLR * er;
    end;
  end;
end;

// Set the input (clamp x_0) and initialise free value nodes by a forward pass
// guess so the relaxation starts near the generative manifold.
procedure SetInput(const S: TSample);
var f, l, c, r, rows, cols: integer; v: TNeuralFloat;
begin
  for f := 0 to cFeat - 1 do Xv[0].FData[f] := S.Feat[f];
  // Bottom-up init: x_l = W_l^T-ish feedforward of previous activation. We use
  // a simple feedforward guess x_l = act-free linear map of Av[l-1] via W_l^T.
  // (Any reasonable init works; relaxation corrects it.)
  for l := 1 to cL do
  begin
    ComputeAct(l - 1);
    rows := Sizes[l - 1];
    cols := Sizes[l];
    for c := 0 to cols - 1 do
    begin
      v := 0;
      for r := 0 to rows - 1 do
        v := v + Wm[l].FData[r * cols + c] * Av[l - 1].FData[r];
      Xv[l].FData[c] := v;
    end;
  end;
end;

procedure ClampTopOneHot(Cls: integer);
var j: integer;
begin
  for j := 0 to Sizes[cL] - 1 do Xv[cL].FData[j] := 0;
  Xv[cL].FData[Cls] := 1.0;
end;

// PCN inference on a new input (energy-based read-out). For each candidate
// class we clamp BOTH ends (input + that one-hot label), relax the hidden
// values, and measure the settled total energy E. The class whose label best
// explains the input -- i.e. yields the LOWEST settled prediction-error energy
// -- is chosen. This is the standard PCN supervised read-out and is far more
// robust than reading a freely-relaxing top node (the hidden value also moves,
// so several labels can explain it similarly). The energy is still composed of
// purely LOCAL prediction errors; no backprop is involved.
function PCNClassify(const S: TSample): integer;
var sweep, cls, best: integer; e, bestE: TNeuralFloat;
begin
  best := 0; bestE := 1e30;
  for cls := 0 to cClasses - 1 do
  begin
    SetInput(S);
    ClampTopOneHot(cls);
    for sweep := 1 to cSweeps do RelaxSweep(True);
    ComputeAct(0); ComputeAct(1); ComputeAct(2);
    ComputeError(1); ComputeError(2);
    e := TotalEnergy;
    if e < bestE then begin bestE := e; best := cls; end;
  end;
  Result := best;
end;

function PCNAccuracy(const SS: TSampleArr): TNeuralFloat;
var I, ok: integer;
begin
  ok := 0;
  for I := 0 to High(SS) do
    if PCNClassify(SS[I]) = SS[I].Cls then Inc(ok);
  Result := ok / Length(SS);
end;

// ---------------------------------------------------------------------------
// PCN training: per example, clamp BOTH ends, relax, then one local update.
// ---------------------------------------------------------------------------
procedure TrainPCN(const TrainSet: TSampleArr);
var Epoch, I, sweep, J, Tmp: integer; Order: array of integer;
begin
  SetLength(Order, Length(TrainSet));
  for I := 0 to High(Order) do Order[I] := I;
  for Epoch := 1 to cEpochs do
  begin
    for I := High(Order) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    for I := 0 to High(Order) do
    begin
      SetInput(TrainSet[Order[I]]);
      ClampTopOneHot(TrainSet[Order[I]].Cls);
      for sweep := 1 to cSweeps do RelaxSweep(True);
      LocalWeightUpdate;
    end;
    if (Epoch mod 10 = 0) or (Epoch = 1) then Write('.');
  end;
  WriteLn;
end;

// ---------------------------------------------------------------------------
// PCN network allocation + init.
// ---------------------------------------------------------------------------
procedure BuildPCN;
var l, i: integer;
begin
  Sizes[0] := cFeat; Sizes[1] := cHidden; Sizes[2] := cClasses;
  for l := 0 to cL do
  begin
    Xv[l] := TNNetVolume.Create(Sizes[l], 1, 1);
    Av[l] := TNNetVolume.Create(Sizes[l], 1, 1);
    Ev[l] := TNNetVolume.Create(Sizes[l], 1, 1);
  end;
  for l := 1 to cL do
  begin
    Wm[l] := TNNetVolume.Create(Sizes[l - 1] * Sizes[l], 1, 1);
    Bv[l] := TNNetVolume.Create(Sizes[l - 1], 1, 1);
    for i := 0 to Wm[l].Size - 1 do
      Wm[l].FData[i] := RandG(0, 1.0 / Sqrt(Sizes[l]));
    Bv[l].Fill(0);
  end;
end;

procedure FreePCN;
var l: integer;
begin
  for l := 0 to cL do begin Xv[l].Free; Av[l].Free; Ev[l].Free; end;
  for l := 1 to cL do begin Wm[l].Free; Bv[l].Free; end;
end;

// ---------------------------------------------------------------------------
// Plain BACKPROP baseline of the SAME shape (input -> tanh(cHidden) -> softmax)
// using the normal TNNet API. Side-by-side accuracy number.
// ---------------------------------------------------------------------------
function TrainBackpropBaseline(const TrainSet, TestSet: TSampleArr): TNeuralFloat;
var
  NN: TNNet;
  Inp, Tgt, Pred: TNNetVolume;
  Epoch, I, J, Tmp, ok: integer;
  Order: array of integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cFeat, 1, 1));
  NN.AddLayer(TNNetFullConnect.Create(cHidden, 1, 1));  // tanh full-connect
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses, 1, 1));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(0.05, 0.9);
  NN.SetL2Decay(0.0);

  Inp := TNNetVolume.Create(cFeat, 1, 1);
  Tgt := TNNetVolume.Create(cClasses, 1, 1);
  Pred := TNNetVolume.Create(cClasses, 1, 1);
  SetLength(Order, Length(TrainSet));
  for I := 0 to High(Order) do Order[I] := I;

  for Epoch := 1 to cEpochs do
  begin
    for I := High(Order) downto 1 do
    begin
      J := Random(I + 1);
      Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    for I := 0 to High(Order) do
    begin
      Inp.FData[0] := TrainSet[Order[I]].Feat[0];
      Inp.FData[1] := TrainSet[Order[I]].Feat[1];
      Tgt.Fill(0); Tgt.FData[TrainSet[Order[I]].Cls] := 1.0;
      NN.Compute(Inp);
      NN.GetOutput(Pred);
      NN.Backpropagate(Tgt);
    end;
  end;

  ok := 0;
  for I := 0 to High(TestSet) do
  begin
    Inp.FData[0] := TestSet[I].Feat[0];
    Inp.FData[1] := TestSet[I].Feat[1];
    NN.Compute(Inp);
    NN.GetOutput(Pred);
    if Pred.GetClass() = TestSet[I].Cls then Inc(ok);
  end;
  Result := ok / Length(TestSet);

  Inp.Free; Tgt.Free; Pred.Free; NN.Free;
end;

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
var
  TrainSet, TestSet: TSampleArr;
  EnergyTrace: array[0..cSweeps] of TNeuralFloat;
  sweep: integer;
  PCNAcc, BPAcc, Chance: TNeuralFloat;
  StartTime, EndTime: TDateTime;
  Gate1, Gate2: boolean;
begin
  RandSeed := cSeed;

  MakeCentres;
  BuildSet(TrainSet, cTrain);
  BuildSet(TestSet, cTest);
  BuildPCN;

  WriteLn('================================================================');
  WriteLn('Predictive Coding Network: local relaxation + Hebbian (NO backprop).');
  WriteLn('================================================================');
  WriteLn(Format('Task: %d Gaussian blobs in %dD.  Net (value nodes): '
    + '%d -> %d -> %d', [cClasses, cFeat, Sizes[0], Sizes[1], Sizes[2]]));
  WriteLn(Format('sweeps=%d  beta=%.2f  lr=%.3f  epochs=%d  train=%d  test=%d  '
    + 'seed=%d', [cSweeps, cBeta, cLR, cEpochs, cTrain, cTest, cSeed]));
  WriteLn('Each W_l predicts layer l-1 top-down; errors e_l = x_l - W*act(x).');
  WriteLn('Phase 1 settles value nodes to shrink E=sum||e||^2; Phase 2 takes one');
  WriteLn('LOCAL Hebbian step dW = e (x) act. No error is chained across layers.');
  WriteLn;

  // ---- Demonstrate the relaxation settles (energy per sweep, one example) ----
  // Use a settled-weight init is not needed; even with random weights the
  // clamped relaxation must DESCEND E. We measure on a fresh (untrained) net so
  // the descent is purely the inference dynamics, not training.
  SetInput(TrainSet[0]);
  ClampTopOneHot(TrainSet[0].Cls);
  // energy before any sweep
  ComputeAct(0); ComputeAct(1); ComputeAct(2);
  ComputeError(1); ComputeError(2);
  EnergyTrace[0] := TotalEnergy;
  for sweep := 1 to cSweeps do
  begin
    RelaxSweep(True);
    // recompute errors at the new values for an honest energy reading
    ComputeAct(0); ComputeAct(1); ComputeAct(2);
    ComputeError(1); ComputeError(2);
    EnergyTrace[sweep] := TotalEnergy;
  end;

  WriteLn('=== Phase-1 inference relaxation (one clamped example, untrained) ===');
  WriteLn('Energy E = sum_l ||e_l||^2 should DECREASE as values settle:');
  for sweep := 0 to cSweeps do
    if (sweep mod 2 = 0) or (sweep = cSweeps) then
      WriteLn(Format('  sweep %2d:  E = %.5f', [sweep, EnergyTrace[sweep]]));
  WriteLn;

  Gate1 := EnergyTrace[cSweeps] < EnergyTrace[0];

  // ---- Train PCN ----
  StartTime := Now;
  Write('Training PCN (clamp both ends, relax, local Hebbian step) ');
  TrainPCN(TrainSet);
  EndTime := Now;

  PCNAcc := PCNAccuracy(TestSet);

  // ---- Backprop baseline of the SAME shape ----
  Write('Training backprop baseline (same shape, normal TNNet API) ... ');
  BPAcc := TrainBackpropBaseline(TrainSet, TestSet);
  WriteLn('done.');

  Chance := 1.0 / cClasses;

  WriteLn;
  WriteLn('=== Held-out accuracy: PCN vs plain backprop (same net shape) ===');
  WriteLn(Format('  PCN (relaxation + local Hebbian, NO backprop) : %.3f', [PCNAcc]));
  WriteLn(Format('  Backprop MLP baseline (global gradient)       : %.3f', [BPAcc]));
  WriteLn(Format('  Chance (%d classes)                            : %.3f',
    [cClasses, Chance]));

  WriteLn;
  WriteLn('=== Correctness gates ===');
  if Gate1 then
    WriteLn(Format('[PASS] GATE 1: relaxation energy fell %.5f -> %.5f (settled).',
      [EnergyTrace[0], EnergyTrace[cSweeps]]))
  else
    WriteLn(Format('[FAIL] GATE 1: energy did NOT fall (%.5f -> %.5f).',
      [EnergyTrace[0], EnergyTrace[cSweeps]]));

  // PCN must clearly beat chance and be in the backprop ballpark (>= BP - 0.15).
  Gate2 := (PCNAcc > Chance + 0.15) and (PCNAcc >= BPAcc - 0.15);
  if Gate2 then
    WriteLn(Format('[PASS] GATE 2: PCN %.3f beats chance %.3f and is comparable '
      + 'to backprop %.3f.', [PCNAcc, Chance, BPAcc]))
  else
    WriteLn(Format('[FAIL] GATE 2: PCN %.3f not comparable (chance %.3f, '
      + 'backprop %.3f).', [PCNAcc, Chance, BPAcc]));

  WriteLn;
  WriteLn(Format('Total PCN wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  FreePCN;

  if Gate1 and Gate2 then
    WriteLn('=> ALL GATES PASS: a backprop-free Predictive Coding Network matched '
      + 'backprop.')
  else
  begin
    WriteLn('=> SOME GATES FAILED (see above).');
    Halt(1);
  end;
end.
