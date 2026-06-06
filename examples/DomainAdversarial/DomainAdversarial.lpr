program DomainAdversarial;
(*
DomainAdversarial: smallest possible Domain-Adversarial Neural Network
(DANN, Ganin et al. 2015, https://arxiv.org/abs/1505.07818) demo built
around the new TNNetGradientReversal layer.

Toy task. Two 2D-Gaussian-blob domains share the same binary class
label, but the blobs are rotated 90 degrees between domain A and
domain B:
  Domain A:  class 0 ~ N((-1,-1), 0.35I)    class 1 ~ N((+1,+1), 0.35I)
  Domain B:  class 0 ~ N((-1,+1), 0.35I)    class 1 ~ N((+1,-1), 0.35I)

A linear classifier on (x, y) trained only on domain A will fail on
domain B, because the class-conditional means swapped along one axis.

Architecture:
  Input(x, y, domain_indicator_unused)
    -> shared trunk (Dense -> ReLU -> Dense -> ReLU)
       -> Label head (Dense -> Softmax over 2 classes)
       -> [TNNetGradientReversal(lambda)?]
          -> Domain head (Dense -> Softmax over 2 domains)
  Both heads are concatenated on the depth axis so a single target
  vector (label_one_hot | domain_one_hot) drives the joint training.

Toggle cUseGRL = False to disable the gradient reversal: the domain
head becomes accurate (the trunk learns domain-discriminative
features) and the label head transfers poorly to the unseen domain.
With cUseGRL = True the trunk is pushed toward domain-invariant
features and label accuracy on the unseen domain improves while the
domain head drops toward chance (~50%).

Runs in well under a minute on CPU.

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
  // Set False to disable the Gradient Reversal Layer (vanilla multi-task
  // learning: the domain head will become accurate and the label head
  // will overfit to the source domain).
  cUseGRL: boolean = True;
  cLambda: TNeuralFloat = 1.0;

  cHidden    = 16;
  cClasses   = 2;
  cDomains   = 2;
  cOutDepth  = cClasses + cDomains;  // packed (label_logits | domain_logits)

  cSteps     = 800;
  cBatch     = 32;
  cLR        = 0.05;
  cInertia   = 0.9;

  cNumEval   = 1000;

function RandNormal: TNeuralFloat;
// Box-Muller.
var
  U1, U2: TNeuralFloat;
begin
  U1 := Random;
  if U1 < 1e-12 then U1 := 1e-12;
  U2 := Random;
  Result := Sqrt(-2 * Ln(U1)) * Cos(2 * Pi * U2);
end;

procedure SampleExample(out X, Y: TNeuralFloat; out Cls, Dom: integer);
const
  cSigma: TNeuralFloat = 0.35;
var
  Mx, My: TNeuralFloat;
begin
  Cls := Random(cClasses);
  Dom := Random(cDomains);
  if Dom = 0 then
  begin
    // Domain A: class 0 -> (-1,-1), class 1 -> (+1,+1).
    if Cls = 0 then begin Mx := -1; My := -1; end
                  else begin Mx := +1; My := +1; end;
  end
  else
  begin
    // Domain B: class 0 -> (-1,+1), class 1 -> (+1,-1).
    if Cls = 0 then begin Mx := -1; My := +1; end
                  else begin Mx := +1; My := -1; end;
  end;
  X := Mx + cSigma * RandNormal;
  Y := My + cSigma * RandNormal;
end;

procedure BuildModel(out NN: TNNet);
var
  TrunkOut: TNNetLayer;
  LabelHead, DomainHead: TNNetLayer;
begin
  NN := TNNet.Create();
  // Inputs: (x, y). Depth carries the two features.
  NN.AddLayer(TNNetInput.Create(1, 1, 2));
  // Shared feature trunk.
  NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));
  TrunkOut := NN.AddLayer(TNNetFullConnectReLU.Create(cHidden));

  // Label head: dense -> softmax (depth = cClasses).
  NN.AddLayer(TNNetFullConnectLinear.Create(cClasses));
  LabelHead := NN.AddLayer(TNNetSoftMax.Create());

  // Domain head: optionally preceded by Gradient Reversal so the trunk
  // is pushed AWAY from features that help the domain classifier.
  if cUseGRL then
    NN.AddLayerAfter(TNNetGradientReversal.Create(cLambda), TrunkOut)
  else
    NN.AddLayerAfter(TNNetIdentity.Create(), TrunkOut);
  NN.AddLayer(TNNetFullConnectLinear.Create(cDomains));
  DomainHead := NN.AddLayer(TNNetSoftMax.Create());

  // Concatenate the two heads so a single target vector drives both.
  NN.AddLayer(TNNetConcat.Create([LabelHead, DomainHead]));

  NN.SetLearningRate(cLR, cInertia);
end;

procedure MakePair(InputV, TargetV: TNNetVolume;
  X, Y: TNeuralFloat; Cls, Dom: integer; TrainDomainHead: boolean);
var
  I: integer;
begin
  InputV.FData[0] := X;
  InputV.FData[1] := Y;
  TargetV.Fill(0);
  TargetV.FData[Cls] := 1.0;
  // The domain-head section of the target. When TrainDomainHead = False
  // we zero out those entries so the domain head sees no error signal
  // (forward still computes; the gradient contribution is zero).
  if TrainDomainHead then
    TargetV.FData[cClasses + Dom] := 1.0
  else
    for I := 0 to cDomains - 1 do
      TargetV.FData[cClasses + I] := 0.0;
end;

function ArgMax(V: TNNetVolume; StartIdx, Count: integer): integer;
var
  I, Best: integer;
  BestVal, Cur: TNeuralFloat;
begin
  Best := 0;
  BestVal := V.FData[StartIdx];
  for I := 1 to Count - 1 do
  begin
    Cur := V.FData[StartIdx + I];
    if Cur > BestVal then
    begin
      BestVal := Cur;
      Best := I;
    end;
  end;
  Result := Best;
end;

procedure Train(NN: TNNet);
var
  Step, B: integer;
  InputV, TargetV: TNNetVolume;
  X, Y: TNeuralFloat;
  Cls, Dom: integer;
  Elapsed, StartTime: double;
begin
  InputV  := TNNetVolume.Create(1, 1, 2);
  TargetV := TNNetVolume.Create(1, 1, cOutDepth);
  try
    StartTime := Now();
    for Step := 1 to cSteps do
    begin
      for B := 1 to cBatch do
      begin
        SampleExample(X, Y, Cls, Dom);
        // Train the domain head on every example so it has a chance to
        // adapt; the GRL (when enabled) flips its trunk-bound gradient.
        MakePair(InputV, TargetV, X, Y, Cls, Dom, True);
        NN.Compute(InputV);
        NN.Backpropagate(TargetV);
      end;
      if (Step = 1) or (Step mod 200 = 0) or (Step = cSteps) then
      begin
        Elapsed := (Now() - StartTime) * 86400.0;
        WriteLn(Format('  step %4d / %4d   elapsed=%.1fs',
          [Step, cSteps, Elapsed]));
      end;
    end;
  finally
    InputV.Free;
    TargetV.Free;
  end;
end;

procedure Evaluate(NN: TNNet;
  out LabelAccA, LabelAccB, DomainAcc: TNeuralFloat);
var
  I: integer;
  InputV: TNNetVolume;
  X, Y: TNeuralFloat;
  Cls, Dom, PredCls, PredDom: integer;
  HitA, TotA, HitB, TotB, HitD: integer;
begin
  InputV := TNNetVolume.Create(1, 1, 2);
  HitA := 0; TotA := 0; HitB := 0; TotB := 0; HitD := 0;
  try
    for I := 1 to cNumEval do
    begin
      SampleExample(X, Y, Cls, Dom);
      InputV.FData[0] := X;
      InputV.FData[1] := Y;
      NN.Compute(InputV);
      PredCls := ArgMax(NN.GetLastLayer.Output, 0, cClasses);
      PredDom := ArgMax(NN.GetLastLayer.Output, cClasses, cDomains);
      if Dom = 0 then
      begin
        Inc(TotA);
        if PredCls = Cls then Inc(HitA);
      end
      else
      begin
        Inc(TotB);
        if PredCls = Cls then Inc(HitB);
      end;
      if PredDom = Dom then Inc(HitD);
    end;
  finally
    InputV.Free;
  end;
  if TotA = 0 then LabelAccA := 0 else LabelAccA := HitA / TotA;
  if TotB = 0 then LabelAccB := 0 else LabelAccB := HitB / TotB;
  DomainAcc := HitD / cNumEval;
end;

procedure RunAlgo();
var
  NN: TNNet;
  LabelAccA, LabelAccB, DomainAcc: TNeuralFloat;
begin
  RandSeed := 42;
  WriteLn('Domain-Adversarial NN (DANN) demo');
  WriteLn(Format('  cUseGRL = %s,  lambda = %.2f', [BoolToStr(cUseGRL, True), cLambda]));
  BuildModel(NN);
  try
    WriteLn('Layers:');
    NN.DebugStructure();
    WriteLn('Training...');
    Train(NN);
    Evaluate(NN, LabelAccA, LabelAccB, DomainAcc);
    WriteLn(Format('Label-head accuracy on domain A: %.3f', [LabelAccA]));
    WriteLn(Format('Label-head accuracy on domain B: %.3f', [LabelAccB]));
    WriteLn(Format('Domain-head accuracy           : %.3f   (chance = 0.500)', [DomainAcc]));
    if cUseGRL then
      WriteLn('Expectation: label-head ~ accurate on both domains; domain-head ~ 0.5.')
    else
      WriteLn('Expectation: domain-head ~ accurate; label-head may diverge between domains.');
  finally
    NN.Free;
  end;
end;

var
  Application: record Title: string; end;

begin
  Application.Title := 'Domain Adversarial Example';
  RunAlgo();
end.
