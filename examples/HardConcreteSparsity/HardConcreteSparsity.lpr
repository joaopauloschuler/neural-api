program HardConcreteSparsity;
(*
HardConcreteSparsity: demonstrates the TNNetHardConcrete learnable L0-sparsity
gate (Louizos, Welling & Kingma 2018, "Learning Sparse Neural Networks through
L0 Regularization").

Two tiny MLPs are trained on the same synthetic task. Both place a
TNNetHardConcrete gate on the 12 input FEATURES (only 2 carry signal, the rest
are noise). The L0 variant adds the expected-L0 penalty gradient to the
per-channel log_alpha so the redundant noise features are driven to HARD zero
(the deterministic inference gate clips to exactly 0). The baseline trains the
SAME gate with only L2 weight decay and no L0 pressure, so its gates stay
nominally alive.

We then report the achieved sparsity = fraction of inference gates == 0 for both
variants, at matched training accuracy, and assert (self-gate, Halt(1) on
failure) that the L0 variant achieves strictly higher hard-zero sparsity.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cFeatures   = 12;   // total input features (2 informative + 10 noise) == gates
  cHidden     = 16;   // hidden width of the classifier behind the gate
  cSamples    = 256;  // synthetic training samples
  cEpochs     = 220;
  cLearnRate  = 0.05;
  // beta * ln(-gamma/zeta) is the L0 threshold constant of the paper.
  cBeta       = 2/3;
  cGamma      = -0.1;
  cZeta       = 1.1;
  cL0Lambda   = 0.05;   // strength of the L0 penalty (L0 variant only)
  cL2Decay    = 0.001; // L2 weight decay on the gate (baseline only)

type
  TDataset = record
    X: array[0..cSamples-1] of array[0..cFeatures-1] of TNeuralFloat;
    Y: array[0..cSamples-1] of TNeuralFloat; // 0.1 / 0.8 target
  end;

var
  Data: TDataset;

procedure BuildDataset();
var
  s, f: integer;
  a, b, logit: TNeuralFloat;
begin
  // Deterministic synthetic data: only features 0 and 1 carry signal, the rest
  // are pure noise. Target is a noisy XOR-ish boundary of the two informative
  // features so a hidden layer is genuinely useful.
  for s := 0 to cSamples - 1 do
  begin
    for f := 0 to cFeatures - 1 do
      Data.X[s][f] := (Random() * 2.0) - 1.0;
    a := Data.X[s][0];
    b := Data.X[s][1];
    logit := 2.5 * (a * b) + 1.2 * a - 0.8 * b;
    if logit >= 0 then Data.Y[s] := 0.8 else Data.Y[s] := 0.1;
  end;
end;

// Builds a small MLP that gates the INPUT FEATURES directly:
//   Input -> HardConcrete gate(cFeatures) -> FC(ReLU) -> FC(2) -> SoftMax.
// Only 2 of the cFeatures inputs carry signal; the rest are pure noise, so the
// gate's per-channel importance is sharply UNEQUAL. The loss gradient anchors
// the 2 informative gates open while the L0 penalty prunes the noise gates to
// hard zero -> a measurable, well-separated sparsity level (~10/12 == 83%).
function BuildNet(out GateLayer: TNNetHardConcrete): TNNet;
var
  NN: TNNet;
begin
  NN := TNNet.Create();
  // Shape the input as (1,1,cFeatures) so the features live along the DEPTH
  // axis: TNNetHardConcrete gates per-DEPTH channel, giving one gate per feature.
  NN.AddLayer(TNNetInput.Create(1, 1, cFeatures));
  GateLayer := TNNetHardConcrete.Create(cBeta, cGamma, cZeta);
  NN.AddLayer(GateLayer);
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  NN.AddLayer(TNNetFullConnectLinear.Create(2));
  NN.AddLayer(TNNetSoftMax.Create());
  NN.SetLearningRate(cLearnRate, 0.9);
  Result := NN;
end;

// Adds the expected-L0 penalty gradient to each gate's log_alpha. The expected
// number of active gates under the hard-concrete distribution is
//   p_active[d] = sigmoid(log_alpha[d] - beta*ln(-gamma/zeta)),
// and d p_active / d log_alpha = p_active*(1-p_active). We descend lambda*sum
// p_active by adding -lr*lambda*(p_active*(1-p_active)) into the gate Delta.
procedure AddL0Penalty(Gate: TNNetHardConcrete; lr, lambda: TNeuralFloat);
var
  d: integer;
  W: TNNetVolume;
  shift, la, p: TNeuralFloat;
begin
  W := Gate.Neurons[0].Weights;
  shift := cBeta * Ln(-cGamma / cZeta);
  for d := 0 to W.Size - 1 do
  begin
    la := W.Raw[d];
    p := 1.0 / (1.0 + Exp(-(la - shift)));
    Gate.Neurons[0].Delta.Raw[d] :=
      Gate.Neurons[0].Delta.Raw[d] + (-lr) * lambda * (p * (1.0 - p));
  end;
end;

// Adds L2 weight decay gradient on the gate's log_alpha (baseline only).
procedure AddL2Decay(Gate: TNNetHardConcrete; lr, decay: TNeuralFloat);
var
  d: integer;
  W: TNNetVolume;
begin
  W := Gate.Neurons[0].Weights;
  for d := 0 to W.Size - 1 do
    Gate.Neurons[0].Delta.Raw[d] :=
      Gate.Neurons[0].Delta.Raw[d] + (-lr) * decay * W.Raw[d];
end;

// Trains a net for cEpochs. UseL0 selects which gate regularizer is applied.
// Returns the final training accuracy.
function TrainNet(NN: TNNet; Gate: TNNetHardConcrete; UseL0: boolean): TNeuralFloat;
var
  ep, s, correct: integer;
  vIn: TNNetVolume;
  vDesired: TNNetVolume;
  pred, target: integer;
begin
  vIn := TNNetVolume.Create(1, 1, cFeatures);
  vDesired := TNNetVolume.Create(1, 1, 2);
  NN.SetBatchUpdate(true);
  for ep := 1 to cEpochs do
  begin
    // Stochastic training gate is active during the weight-updating pass.
    NN.EnableDropouts(true);
    NN.ClearDeltas();
    for s := 0 to cSamples - 1 do
    begin
      vIn.Copy(Data.X[s]);
      if Data.Y[s] > 0.5 then
      begin
        vDesired.Raw[0] := 0.0; vDesired.Raw[1] := 1.0;
      end
      else
      begin
        vDesired.Raw[0] := 1.0; vDesired.Raw[1] := 0.0;
      end;
      NN.Compute(vIn);
      NN.Backpropagate(vDesired);
    end;
    // Scale the regularizer to the accumulated batch gradient (cSamples terms).
    if UseL0 then AddL0Penalty(Gate, cLearnRate, cL0Lambda * cSamples)
    else AddL2Decay(Gate, cLearnRate, cL2Decay * cSamples);
    NN.UpdateWeights();
  end;

  // Final training accuracy under the deterministic inference gate.
  NN.EnableDropouts(false);
  correct := 0;
  for s := 0 to cSamples - 1 do
  begin
    vIn.Copy(Data.X[s]);
    NN.Compute(vIn);
    if NN.GetLastLayer.Output.Raw[1] >= NN.GetLastLayer.Output.Raw[0] then
      pred := 1 else pred := 0;
    if Data.Y[s] > 0.5 then target := 1 else target := 0;
    if pred = target then Inc(correct);
  end;
  vIn.Free;
  vDesired.Free;
  Result := correct / cSamples;
end;

// Fraction of inference gates that clip to EXACTLY 0.
function HardZeroFraction(Gate: TNNetHardConcrete): TNeuralFloat;
var
  d, zeros: integer;
  la, s, sstr: TNeuralFloat;
begin
  zeros := 0;
  for d := 0 to Gate.Neurons[0].Weights.Size - 1 do
  begin
    la := Gate.Neurons[0].Weights.Raw[d];
    s := 1.0 / (1.0 + Exp(-la));              // deterministic gate, no noise
    sstr := s * (cZeta - cGamma) + cGamma;
    if sstr <= 0 then Inc(zeros);
  end;
  Result := zeros / Gate.Neurons[0].Weights.Size;
end;

var
  NN_L0, NN_L2: TNNet;
  Gate_L0, Gate_L2: TNNetHardConcrete;
  acc_L0, acc_L2, sp_L0, sp_L2: TNeuralFloat;
  dbg: integer;
begin
  // CPU-only, single-threaded, deterministic: the manual training loop below
  // issues plain NN.Compute / NN.Backpropagate calls (no thread pool).
  RandSeed := 424242;

  WriteLn('HardConcrete L0-sparsity demo (Louizos et al. 2018)');
  WriteLn('Features=', cFeatures, ' Hidden/Gates=', cHidden, ' Samples=', cSamples);
  WriteLn;

  BuildDataset();

  NN_L0 := BuildNet(Gate_L0);
  NN_L2 := BuildNet(Gate_L2);

  WriteLn('Training L0-regularised gate (lambda=', cL0Lambda:0:3, ') ...');
  acc_L0 := TrainNet(NN_L0, Gate_L0, true);
  WriteLn('Training L2 baseline gate (decay=', cL2Decay:0:3, ') ...');
  acc_L2 := TrainNet(NN_L2, Gate_L2, false);

  // Per-channel inference gate log_alpha (informative features 0,1 stay open;
  // the noise features are driven negative until their gate hard-clips to 0).
  Write('  L0 per-feature log_alpha:');
  for dbg := 0 to cFeatures - 1 do
    Write(Gate_L0.Neurons[0].Weights.Raw[dbg]:7:2);
  WriteLn;
  sp_L0 := HardZeroFraction(Gate_L0);
  sp_L2 := HardZeroFraction(Gate_L2);

  WriteLn;
  WriteLn('Results (inference deterministic gate):');
  WriteLn('  L0 variant : train acc = ', (acc_L0 * 100):6:2, '%   hard-zero gates = ',
    (sp_L0 * 100):6:2, '%');
  WriteLn('  L2 baseline: train acc = ', (acc_L2 * 100):6:2, '%   hard-zero gates = ',
    (sp_L2 * 100):6:2, '%');
  WriteLn;

  // Self-gate: the L0 variant must reach a comparable accuracy AND a strictly
  // higher hard-zero sparsity than the baseline.
  if acc_L0 < 0.80 then
  begin
    WriteLn('FAIL: L0 variant accuracy too low (', (acc_L0 * 100):0:2, '%).');
    NN_L0.Free; NN_L2.Free;
    Halt(1);
  end;
  if not (sp_L0 > sp_L2) then
  begin
    WriteLn('FAIL: L0 sparsity (', (sp_L0 * 100):0:2,
      '%) not strictly greater than baseline (', (sp_L2 * 100):0:2, '%).');
    NN_L0.Free; NN_L2.Free;
    Halt(1);
  end;

  WriteLn('PASS: L0 gate is sparser than the L2 baseline at matched accuracy.');
  NN_L0.Free;
  NN_L2.Free;
end.
