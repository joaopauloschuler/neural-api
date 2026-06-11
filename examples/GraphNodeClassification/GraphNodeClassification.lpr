program GraphNodeClassification;
(*
GraphNodeClassification: semi-supervised (TRANSDUCTIVE) node classification on a
tiny synthetic two-community Stochastic Block Model (SBM) graph, contrasting a
2-layer spectral Graph Convolutional Network (TNNetGraphConvolution, Kipf &
Welling 2017) against a param-matched, label-agnostic per-node MLP baseline.

Setup
-----
We build an SBM with two communities (classes 0 and 1). Edges are dense WITHIN
a community and sparse BETWEEN communities, so the graph structure carries most
of the class signal. Each node gets a SHORT, DELIBERATELY WEAK feature vector
(a noisy 1-hot-ish community hint): on its own a per-node classifier can barely
separate the two classes, but a message-passing model that mixes a node's
features with its neighbours' recovers the community cleanly.

Only a HANDFUL of nodes are labelled (a few per class). We train on those and
report accuracy on ALL the held-out nodes (transductive: the unlabelled nodes'
FEATURES are seen during the forward pass via message passing, only their LABELS
are hidden).

Two models, same input, same shapes, same parameter budget:
  (A) GCN   : Input -> GraphConvolution(H) -> ReLU -> GraphConvolution(2) -> SoftMax
              SetAdjacency(SBM adjacency)  => neighbour aggregation is ON.
  (B) MLP   : identical layer stack, but SetAdjacency(IDENTITY) => Ahat = I, so
              the aggregation step is a no-op and every node is classified PURELY
              from its own (weak) features. This is the label-agnostic baseline:
              same weights, same training, the ONLY difference is whether the
              message passing carries the signal.

The headline is the transductive accuracy gap: the GCN should substantially beat
the feature-only MLP, demonstrating that the graph structure (message passing) is
what carries the class signal.

Everything is generated on the fly; no external dataset. Pure CPU, single thread,
well under five minutes (a couple of seconds in practice).

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cNodesPerClass = 30;                       // 30 + 30 = 60 nodes total
  cNumNodes      = 2 * cNodesPerClass;
  cFeat          = 4;                         // weak per-node feature dimension
  cHidden        = 8;                         // GCN hidden width
  cPIn           = 0.35;                      // within-community edge probability
  cPOut          = 0.03;                      // between-community edge probability
  cLabelsPerClass = 4;                        // labelled nodes per class (rest held out)
  cEpochs        = 400;
  cLR            = 0.05;
  cInertia       = 0.9;

var
  GAdj:      TNNetVolume;                     // raw 0/1 adjacency (no self loops)
  GIdentity: TNNetVolume;                     // identity adjacency (MLP baseline)
  GFeat:     TNNetVolume;                     // (cNumNodes,1,cFeat) node features
  GLabel:    array[0..cNumNodes - 1] of integer;
  GIsTrain:  array[0..cNumNodes - 1] of boolean;

// Class of a node by index: first half class 0, second half class 1.
function NodeClass(n: integer): integer;
begin
  if n < cNodesPerClass then Result := 0 else Result := 1;
end;

procedure BuildGraph;
var
  i, j, c: integer;
  p: TNeuralFloat;
begin
  GAdj := TNNetVolume.Create(cNumNodes, cNumNodes, 1);
  GIdentity := TNNetVolume.Create(cNumNodes, cNumNodes, 1);
  GFeat := TNNetVolume.Create(cNumNodes, 1, cFeat);
  GAdj.Fill(0);
  GIdentity.Fill(0); // identity has NO off-diagonal edges; SetAdjacency adds +I.

  // SBM edges (undirected, symmetric).
  for i := 0 to cNumNodes - 1 do
    for j := i + 1 to cNumNodes - 1 do
    begin
      if NodeClass(i) = NodeClass(j) then p := cPIn else p := cPOut;
      if Random < p then
      begin
        GAdj.Raw[GAdj.GetRawPos(i, j, 0)] := 1;
        GAdj.Raw[GAdj.GetRawPos(j, i, 0)] := 1;
      end;
    end;

  // Weak node features: a noisy community hint. The signal-to-noise ratio is
  // deliberately low so a per-node classifier alone struggles.
  for i := 0 to cNumNodes - 1 do
  begin
    c := NodeClass(i);
    GLabel[i] := c;
    GIsTrain[i] := false;
    for j := 0 to cFeat - 1 do
      GFeat.Raw[i * cFeat + j] := 0.7 * (Random - 0.5); // mostly noise
    // A faint class bias on the first two channels (|bias| << noise spread).
    GFeat.Raw[i * cFeat + c] := GFeat.Raw[i * cFeat + c] + 0.13;
  end;

  // Pick the first cLabelsPerClass nodes of each class as the labelled set.
  for c := 0 to 1 do
    for i := 0 to cLabelsPerClass - 1 do
      GIsTrain[c * cNodesPerClass + i] := true;
end;

// Build the 2-layer GCN. With AAdj = SBM adjacency this is a true GCN; with
// AAdj = identity it degenerates to a per-node MLP (no message passing).
function BuildNet(AAdj: TNNetVolume): TNNet;
var
  NN: TNNet;
  GC1, GC2: TNNetGraphConvolution;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cNumNodes, 1, cFeat));
  GC1 := TNNetGraphConvolution.Create(cHidden);
  NN.AddLayer(GC1);
  NN.AddLayer(TNNetReLU.Create());
  GC2 := TNNetGraphConvolution.Create(2);
  NN.AddLayer(GC2);
  // PER-NODE softmax over the 2 class channels (NOT the whole-graph TNNetSoftMax,
  // which would normalize across nodes). SkipBackpropDerivative=1 lets us push the
  // cross-entropy gradient (p - y) straight through.
  NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
  GC1.SetAdjacency(AAdj);
  GC2.SetAdjacency(AAdj);
  Result := NN;
end;

// Full-batch transductive training: one forward over the WHOLE graph per epoch,
// backprop the cross-entropy error only at the labelled nodes.
procedure TrainNet(NN: TNNet);
var
  epoch, n, k, OutFeat: integer;
  Pred, ErrV: TNNetVolume;
  Target: array[0..1] of TNeuralFloat;
begin
  NN.SetLearningRate(cLR, cInertia);
  NN.SetBatchUpdate(true);
  OutFeat := 2;
  ErrV := TNNetVolume.Create(cNumNodes, 1, OutFeat);
  // The last (SoftMax) layer has no branch loading from it, so its departing
  // count is 0; bump it once so the direct Backpropagate() calls are accepted.
  NN.GetLastLayer.IncDepartingBranchesCnt();
  for epoch := 1 to cEpochs do
  begin
    NN.Compute(GFeat);
    Pred := NN.GetLastLayer.Output;
    // Cross-entropy gradient at the SoftMax output is (p - y); zero it out for
    // unlabelled nodes so only the labelled nodes drive the update.
    ErrV.Fill(0);
    for n := 0 to cNumNodes - 1 do
      if GIsTrain[n] then
      begin
        Target[0] := 0; Target[1] := 0;
        Target[GLabel[n]] := 1;
        for k := 0 to OutFeat - 1 do
          ErrV.Raw[n * OutFeat + k] := Pred.Raw[n * OutFeat + k] - Target[k];
      end;
    // Drive backprop from a custom (masked) output error, mirroring the wiring
    // inside TNNet.Backpropagate: reset the per-call counters, then push the
    // masked error in at the last layer.
    NN.ResetBackpropCallCurrCnt();
    NN.GetLastLayer.OutputError.Copy(ErrV);
    NN.GetLastLayer.Backpropagate();
    NN.UpdateWeights();
    NN.ClearDeltas();
  end;
  ErrV.Free;
end;

// Transductive accuracy over the HELD-OUT (unlabelled) nodes.
function EvalHeldOut(NN: TNNet): TNeuralFloat;
var
  n, OutFeat, correct, total: integer;
  Pred: TNNetVolume;
  predClass: integer;
begin
  OutFeat := 2;
  NN.Compute(GFeat);
  Pred := NN.GetLastLayer.Output;
  correct := 0; total := 0;
  for n := 0 to cNumNodes - 1 do
    if not GIsTrain[n] then
    begin
      if Pred.Raw[n * OutFeat + 1] > Pred.Raw[n * OutFeat + 0]
        then predClass := 1 else predClass := 0;
      if predClass = GLabel[n] then Inc(correct);
      Inc(total);
    end;
  if total > 0 then Result := correct / total else Result := 0;
end;

var
  GcnNet, MlpNet: TNNet;
  GcnAcc, MlpAcc: TNeuralFloat;
  EdgeCount, i, j: integer;
begin
  RandSeed := 424242;
  WriteLn('Graph node classification: GCN vs feature-only MLP baseline');
  WriteLn('-----------------------------------------------------------');

  BuildGraph;

  EdgeCount := 0;
  for i := 0 to cNumNodes - 1 do
    for j := i + 1 to cNumNodes - 1 do
      if GAdj.Raw[GAdj.GetRawPos(i, j, 0)] > 0 then Inc(EdgeCount);

  WriteLn('Nodes              : ', cNumNodes, ' (', cNodesPerClass, ' per class)');
  WriteLn('Edges              : ', EdgeCount);
  WriteLn('Features/node      : ', cFeat, ' (weak, noisy community hint)');
  WriteLn('Labelled nodes     : ', 2 * cLabelsPerClass, ' (', cLabelsPerClass, ' per class)');
  WriteLn('Held-out nodes     : ', cNumNodes - 2 * cLabelsPerClass);
  WriteLn;

  // (A) GCN with the real adjacency: message passing ON.
  GcnNet := BuildNet(GAdj);
  TrainNet(GcnNet);
  GcnAcc := EvalHeldOut(GcnNet);

  // (B) Same stack, identity adjacency: per-node MLP, message passing OFF.
  MlpNet := BuildNet(GIdentity);
  TrainNet(MlpNet);
  MlpAcc := EvalHeldOut(MlpNet);

  WriteLn('Transductive accuracy on held-out nodes');
  WriteLn('  GCN (message passing ON) : ', (GcnAcc * 100):6:2, ' %');
  WriteLn('  MLP (features only)      : ', (MlpAcc * 100):6:2, ' %');
  WriteLn('  gap (GCN - MLP)          : ', ((GcnAcc - MlpAcc) * 100):6:2, ' pp');
  WriteLn;
  if GcnAcc > MlpAcc + 0.05 then
    WriteLn('=> The graph structure (neighbour aggregation) carries the signal: the',
            sLineBreak, '   GCN beats the feature-only MLP by a clear margin.')
  else
    WriteLn('=> NOTE: GCN did not clearly beat the MLP this run.');

  GcnNet.Free;
  MlpNet.Free;
  GAdj.Free;
  GIdentity.Free;
  GFeat.Free;
end.
