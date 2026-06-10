program GraphAttention;
(*
GraphAttention: semi-supervised (TRANSDUCTIVE) node classification on a tiny
synthetic two-community Stochastic Block Model (SBM) graph, contrasting a
single-head Graph ATTENTION network (TNNetGraphAttention, Velickovic et al. 2018)
against the param-matched spectral Graph CONVOLUTION network
(TNNetGraphConvolution, Kipf & Welling 2017) on the SAME graph.

The question
-----------
A plain GCN aggregates every neighbour with FIXED, degree-normalized weights: it
cannot tell a "good" (same-community) edge from a "bad" (cross-community) one. A
GAT instead LEARNS a per-edge attention coefficient from the node features, so it
can DOWN-WEIGHT misleading edges. Does that help when the graph is noisy /
partially heterophilous?

The graph
---------
Two communities (classes 0 and 1). Clean SBM edges are dense WITHIN a community
and sparse between. We then INJECT extra NOISY cross-community edges (the
"heterophilous" corruption): these connect a node to neighbours of the WRONG
class. A fixed-weight averager (GCN) is forced to mix in this wrong-class signal;
an attention model can learn to ignore it.

Each node carries a short CLASS-INDICATIVE feature vector (a class signal plus
noise). The features alone are informative, so attention can learn to attend to
neighbours that "look like me" and ignore the wrong-class ones — but the GCN's
fixed edge weights cannot exploit that, it must average every neighbour in. Only
a handful of nodes are labelled (a few per class); we train on those and report
accuracy on ALL held-out nodes (transductive).

Two models, same input, same shapes, same training loop, same adjacency:
  (A) GAT : Input -> GraphAttention(H) -> ReLU -> GraphAttention(2) -> per-node SoftMax
  (B) GCN : Input -> GraphConvolution(H) -> ReLU -> GraphConvolution(2) -> per-node SoftMax

The headline is the held-out accuracy gap on the corrupted graph: the GAT's
learned edge weighting should be MORE ROBUST to the injected noisy edges than the
GCN's fixed averaging.

This example also includes two GAT follow-ups:
  (C) MULTI-HEAD GAT (TNNet.AddMultiHeadGraphAttention): K independent attention
      heads CONCATENATED in the hidden layer and AVERAGED at the output layer
      (the paper's eq. 5 / eq. 6 split), single head vs 4 heads.
  (D) ATTENTION-DROPOUT ablation: the per-edge dropout regulariser (paper Sec
      2.2) applied to the normalized coefficients at training time only, off vs
      on (p=0.3) on the noisy-edge SBM regime.

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

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume;

const
  cNodesPerClass  = 25;                       // 25 + 25 = 50 nodes total
  cNumNodes       = 2 * cNodesPerClass;
  cFeat           = 4;                         // weak per-node feature dimension
  cHidden         = 8;                         // hidden width
  cPIn            = 0.22;                       // within-community edge probability
  cPOut           = 0.01;                       // clean between-community edge prob
  cNoisyEdges     = 200;                        // INJECTED noisy cross-community edges
  cLabelsPerClass = 5;                          // labelled nodes per class (rest held out)
  cEpochs         = 400;
  cLR             = 0.05;
  cInertia        = 0.9;

var
  GAdj:   TNNetVolume;                          // raw 0/1 adjacency (no self loops)
  GFeat:  TNNetVolume;                          // (cNumNodes,1,cFeat) node features
  GLabel: array[0..cNumNodes - 1] of integer;
  GIsTrain: array[0..cNumNodes - 1] of boolean;

function NodeClass(n: integer): integer;
begin
  if n < cNodesPerClass then Result := 0 else Result := 1;
end;

procedure AddEdge(i, j: integer);
begin
  if i = j then exit;
  GAdj.Raw[GAdj.GetRawPos(i, j, 0)] := 1;
  GAdj.Raw[GAdj.GetRawPos(j, i, 0)] := 1;
end;

procedure BuildGraph;
var
  i, j, c, added: integer;
  p: TNeuralFloat;
begin
  GAdj := TNNetVolume.Create(cNumNodes, cNumNodes, 1);
  GFeat := TNNetVolume.Create(cNumNodes, 1, cFeat);
  GAdj.Fill(0);

  // Clean SBM edges (undirected, symmetric).
  for i := 0 to cNumNodes - 1 do
    for j := i + 1 to cNumNodes - 1 do
    begin
      if NodeClass(i) = NodeClass(j) then p := cPIn else p := cPOut;
      if Random < p then AddEdge(i, j);
    end;

  // Inject NOISY cross-community edges (the heterophilous corruption): connect
  // random nodes of opposite classes. A fixed-weight averager must mix these in.
  added := 0;
  while added < cNoisyEdges do
  begin
    i := Random(cNodesPerClass);                       // class 0
    j := cNodesPerClass + Random(cNodesPerClass);      // class 1
    if GAdj.Raw[GAdj.GetRawPos(i, j, 0)] = 0 then
    begin
      AddEdge(i, j);
      Inc(added);
    end;
  end;

  // Weak node features: a noisy community hint, low signal-to-noise.
  for i := 0 to cNumNodes - 1 do
  begin
    c := NodeClass(i);
    GLabel[i] := c;
    GIsTrain[i] := false;
    for j := 0 to cFeat - 1 do
      GFeat.Raw[i * cFeat + j] := 0.35 * (Random - 0.5);
    // A clear class signal so attention can key on it (attend to neighbours that
    // look like me); the fixed-weight GCN still has to average wrong-class
    // neighbours in regardless of how class-indicative their features are.
    GFeat.Raw[i * cFeat + c] := GFeat.Raw[i * cFeat + c] + 1.0;
  end;

  for c := 0 to 1 do
    for i := 0 to cLabelsPerClass - 1 do
      GIsTrain[c * cNodesPerClass + i] := true;
end;

// kind = 0 -> GCN (TNNetGraphConvolution), kind = 1 -> single-head GAT.
function BuildNet(kind: integer): TNNet;
var
  NN: TNNet;
  GcnA, GcnB: TNNetGraphConvolution;
  GatA, GatB: TNNetGraphAttention;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cNumNodes, 1, cFeat));
  if kind = 0 then
  begin
    GcnA := TNNetGraphConvolution.Create(cHidden);
    NN.AddLayer(GcnA);
    NN.AddLayer(TNNetReLU.Create());
    GcnB := TNNetGraphConvolution.Create(2);
    NN.AddLayer(GcnB);
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    GcnA.SetAdjacency(GAdj);
    GcnB.SetAdjacency(GAdj);
  end
  else
  begin
    GatA := TNNetGraphAttention.Create(cHidden);
    NN.AddLayer(GatA);
    NN.AddLayer(TNNetReLU.Create());
    GatB := TNNetGraphAttention.Create(2);
    NN.AddLayer(GatB);
    NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
    GatA.SetAdjacency(GAdj);
    GatB.SetAdjacency(GAdj);
  end;
  Result := NN;
end;

// Multi-head GAT (TNNet.AddMultiHeadGraphAttention): Heads CONCAT heads in the
// hidden layer (each cHidden wide -> Heads*cHidden), then Heads AVERAGED heads at
// the 2-class output (the paper's eq. 5 / eq. 6 split). pAttDrop applies the
// per-edge attention-dropout regulariser to every head (training time only).
function BuildMultiHeadNet(Heads: integer; pAttDrop: TNeuralFloat): TNNet;
var
  NN: TNNet;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cNumNodes, 1, cFeat));
  // Hidden: Heads concatenated single-head GAT layers.
  NN.AddMultiHeadGraphAttention(Heads, cHidden, GAdj, {Concat=}true, pAttDrop);
  NN.AddLayer(TNNetReLU.Create());
  // Output: Heads averaged single-head GAT layers (2 classes).
  NN.AddMultiHeadGraphAttention(Heads, 2, GAdj, {Concat=}false, pAttDrop);
  NN.AddLayer(TNNetPointwiseSoftMax.Create(1));
  Result := NN;
end;

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
  NN.GetLastLayer.IncDepartingBranchesCnt();
  NN.EnableDropouts(true); // activate GAT attention-dropout for training (no-op if rate=0)
  for epoch := 1 to cEpochs do
  begin
    NN.Compute(GFeat);
    Pred := NN.GetLastLayer.Output;
    ErrV.Fill(0);
    for n := 0 to cNumNodes - 1 do
      if GIsTrain[n] then
      begin
        Target[0] := 0; Target[1] := 0;
        Target[GLabel[n]] := 1;
        for k := 0 to OutFeat - 1 do
          ErrV.Raw[n * OutFeat + k] := Pred.Raw[n * OutFeat + k] - Target[k];
      end;
    NN.ResetBackpropCallCurrCnt();
    NN.GetLastLayer.OutputError.Copy(ErrV);
    NN.GetLastLayer.Backpropagate();
    NN.UpdateWeights();
    NN.ClearDeltas();
  end;
  ErrV.Free;
end;

function EvalHeldOut(NN: TNNet): TNeuralFloat;
var
  n, OutFeat, correct, total, predClass: integer;
  Pred: TNNetVolume;
begin
  OutFeat := 2;
  NN.EnableDropouts(false); // deterministic inference (disable attention-dropout)
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

// Train a freshly-built multi-head GAT (rebuilds the net so RNG / init are
// deterministic per call) and return its held-out accuracy.
function RunMultiHead(Heads: integer; pAttDrop: TNeuralFloat): TNeuralFloat;
var
  NN: TNNet;
begin
  NN := BuildMultiHeadNet(Heads, pAttDrop);
  TrainNet(NN);
  Result := EvalHeldOut(NN);
  NN.Free;
end;

var
  GatNet, GcnNet: TNNet;
  GatAcc, GcnAcc, Mh1, Mh4, AblOff, AblOn: TNeuralFloat;
  CleanEdges, NoisyEdges, i, j: integer;
begin
  RandSeed := 424242;
  WriteLn('Graph node classification: GAT vs GCN on a NOISY (heterophilous) graph');
  WriteLn('----------------------------------------------------------------------');

  BuildGraph;

  CleanEdges := 0; NoisyEdges := 0;
  for i := 0 to cNumNodes - 1 do
    for j := i + 1 to cNumNodes - 1 do
      if GAdj.Raw[GAdj.GetRawPos(i, j, 0)] > 0 then
      begin
        if NodeClass(i) = NodeClass(j) then Inc(CleanEdges) else Inc(NoisyEdges);
      end;

  WriteLn('Nodes                : ', cNumNodes, ' (', cNodesPerClass, ' per class)');
  WriteLn('Same-class edges     : ', CleanEdges);
  WriteLn('Cross-class edges    : ', NoisyEdges, '  (noisy / heterophilous)');
  WriteLn('Features/node        : ', cFeat, ' (class-indicative + noise)');
  WriteLn('Labelled nodes       : ', 2 * cLabelsPerClass, ' (', cLabelsPerClass, ' per class)');
  WriteLn('Held-out nodes       : ', cNumNodes - 2 * cLabelsPerClass);
  WriteLn;

  // (A) GAT: learned per-edge attention.
  GatNet := BuildNet(1);
  TrainNet(GatNet);
  GatAcc := EvalHeldOut(GatNet);

  // (B) GCN: fixed degree-normalized edge weights.
  GcnNet := BuildNet(0);
  TrainNet(GcnNet);
  GcnAcc := EvalHeldOut(GcnNet);

  WriteLn('Transductive accuracy on held-out nodes');
  WriteLn('  GAT (learned edge weights) : ', (GatAcc * 100):6:2, ' %');
  WriteLn('  GCN (fixed edge weights)   : ', (GcnAcc * 100):6:2, ' %');
  WriteLn('  gap (GAT - GCN)            : ', ((GatAcc - GcnAcc) * 100):6:2, ' pp');
  WriteLn;
  if GatAcc > GcnAcc + 0.02 then
    WriteLn('=> Learned edge weighting helps: the GAT down-weights the injected',
            sLineBreak, '   cross-community edges that the fixed-weight GCN is forced to average in.')
  else if GatAcc > GcnAcc - 0.02 then
    WriteLn('=> GAT roughly matches the GCN on this run (both cope with the noise).')
  else
    WriteLn('=> NOTE: GAT did not beat the GCN this run.');

  GatNet.Free;
  GcnNet.Free;

  // -------------------------------------------------------------------------
  // (C) MULTI-HEAD GAT: K independent attention heads (concat in the hidden
  // layer, averaged at the output) via TNNet.AddMultiHeadGraphAttention. More
  // heads = more independent edge-weighting views averaged together, which is
  // more robust on the noisy graph than a single head.
  // -------------------------------------------------------------------------
  Mh1 := RunMultiHead(1, 0.0);   // single-head (built via the multi-head builder)
  Mh4 := RunMultiHead(4, 0.0);   // 4 heads
  WriteLn('Multi-head GAT (concat hidden, averaged output)');
  WriteLn('  1 head                     : ', (Mh1 * 100):6:2, ' %');
  WriteLn('  4 heads                    : ', (Mh4 * 100):6:2, ' %');
  WriteLn('  gain (4 heads - 1 head)    : ', ((Mh4 - Mh1) * 100):6:2, ' pp');
  WriteLn;

  // -------------------------------------------------------------------------
  // (D) ATTENTION-DROPOUT ABLATION on the noisy-edge SBM regime: the per-edge
  // dropout regulariser (paper Sec 2.2) randomly drops normalized attention
  // coefficients during training only. On this noisy graph it discourages the
  // model from over-committing to any single (possibly cross-community) edge.
  // -------------------------------------------------------------------------
  AblOff := RunMultiHead(4, 0.0);  // 4 heads, no attention-dropout
  AblOn  := RunMultiHead(4, 0.3);  // 4 heads, attention-dropout p=0.3
  WriteLn('Attention-dropout ablation (4-head GAT on the noisy graph)');
  WriteLn('  dropout OFF (p=0.0)        : ', (AblOff * 100):6:2, ' %');
  WriteLn('  dropout ON  (p=0.3)        : ', (AblOn * 100):6:2, ' %');
  WriteLn('  effect (on - off)          : ', ((AblOn - AblOff) * 100):6:2, ' pp');
  WriteLn;
  if AblOn > AblOff + 0.01 then
    WriteLn('=> Attention-dropout helps on the noisy graph: dropping per-edge',
            sLineBreak, '   coefficients during training regularises the edge weighting.')
  else if AblOn > AblOff - 0.01 then
    WriteLn('=> Attention-dropout roughly neutral on this run.')
  else
    WriteLn('=> NOTE: attention-dropout did not help this run.');

  GAdj.Free;
  GFeat.Free;
end.
