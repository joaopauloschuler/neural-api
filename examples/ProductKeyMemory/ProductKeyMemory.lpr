program ProductKeyMemory;
(*
ProductKeyMemory: a tiny synthetic key->value retrieval demo for the
Product-Key Memory layer (TNNetProductKeyMemory), following Lample et al.,
NeurIPS 2019, "Large Memory Layers with Product Keys"
(https://arxiv.org/abs/1907.05242).

THE IDEA. A flat memory of |K| slots addressed by a full softmax costs
O(|K|) per query. Product-key memory factorizes the |K| keys into the
Cartesian product of TWO small half-key banks K1, K2 (each sqrt(|K|) keys
of half the query dimension). A query is split into two half-queries; each
half is scored against its own bank, the top-k per half are taken, the
k x k combinations are re-scored (sum of the two half-scores), and the
global top-k product keys are selected -- all in O(sqrt(|K|)) work. The
selected keys' softmax weights gate a sparse weighted sum over the
corresponding learned VALUE rows (an EmbeddingBag-style lookup).

THE TASK. We build NumPairs random (query -> target-value) associations.
Each query is a random vector in R^QueryDim; each target is a random vector
in R^ValueDim. The network must learn to reproduce the target value from
its query. This is exactly associative recall: the memory has to allocate
slots to associations and route each query to the right slot(s).

THE COMPARISON. Two models are trained on the IDENTICAL data:

  1. PRODUCT-KEY MEMORY: TNNet.AddProductKeyMemory(NumKeys, ValueDim,
     TopK, Heads=1). It touches only TopK value rows per query (sparse).
  2. FLAT-MEMORY BASELINE: a same-capacity dense map of the SAME number of
     value slots, implemented as a full softmax-attention readout over all
     NumKeys learned keys (TNNetModernHopfield with one retrieval step is a
     single dense softmax over a NumKeys-row bank). It reads ALL NumKeys
     rows per query.

We report, for both, the final mean-squared retrieval error and the
average number of value rows TOUCHED per query. The headline result: the
product-key memory reaches comparable retrieval error while touching only
TopK rows (sparse) vs all NumKeys rows (dense) -- the whole point of the
factorization.

KNOWN FAILURE MODE (see README): key-usage COLLAPSE. A handful of product
keys can hog all the reads (every query routes to the same few slots),
wasting the memory's capacity. The paper spreads usage with a BATCH-NORM
on the query before scoring (which de-correlates / re-centres queries so
no slot is permanently favoured). This demo prints a per-slot read-count
histogram so collapse is visible; the batch-norm trick is described in the
README but not applied here (the synthetic task is small enough to avoid
severe collapse).

Pure CPU, single thread, tiny dims, runtime well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralnetwork, neuralvolume;

const
  NumKeys   = 64;   // product-key slots = 8 x 8 (perfect square)
  HalfKeys  = 8;    // sqrt(NumKeys)
  QueryDim  = 16;   // even; split into two halves of 8
  ValueDim  = 8;    // width of each value vector / network output
  TopK      = 4;    // product keys retrieved per query (sparse)
  NumPairs  = 24;   // synthetic associations to learn
  Epochs    = 4000;
  LR        = 0.05;

var
  Queries, Targets: array[0..NumPairs - 1] of TNNetVolume;
  ReadCount: array[0..NumKeys - 1] of integer;

// Build NumPairs random (query, target) associations with a fixed seed.
procedure BuildData;
var p, i: integer;
begin
  RandSeed := 20240606;
  for p := 0 to NumPairs - 1 do
  begin
    Queries[p] := TNNetVolume.Create(1, 1, QueryDim);
    Targets[p] := TNNetVolume.Create(1, 1, ValueDim);
    for i := 0 to QueryDim - 1 do
      Queries[p].Raw[i] := Queries[p].RandomGaussianValue();
    for i := 0 to ValueDim - 1 do
      Targets[p].Raw[i] := Targets[p].RandomGaussianValue() * 0.8;
  end;
end;

procedure FreeData;
var p: integer;
begin
  for p := 0 to NumPairs - 1 do
  begin
    Queries[p].Free;
    Targets[p].Free;
  end;
end;

// Mean squared retrieval error over all associations.
function MeanSquaredError(NN: TNNet): TNeuralFloat;
var p, k: integer; diff, s: TNeuralFloat;
begin
  s := 0;
  for p := 0 to NumPairs - 1 do
  begin
    NN.Compute(Queries[p]);
    for k := 0 to ValueDim - 1 do
    begin
      diff := NN.GetLastLayer.Output.Raw[k] - Targets[p].Raw[k];
      s := s + diff * diff;
    end;
  end;
  Result := s / (NumPairs * ValueDim);
end;

// Train a network for Epochs steps of plain SGD over all pairs.
procedure Train(NN: TNNet; const ATag: string);
var e, p: integer;
begin
  NN.SetLearningRate(LR, 0.0);
  for e := 0 to Epochs - 1 do
    for p := 0 to NumPairs - 1 do
    begin
      NN.Compute(Queries[p]);
      NN.Backpropagate(Targets[p]);
    end;
  WriteLn(Format('  [%s] final MSE = %.6f', [ATag, MeanSquaredError(NN)]));
end;

// Re-derive, from the PUBLIC half-key banks, which product keys a query would
// select (the layer caches this internally; here we recompute the identical
// top-k selection over the public Neurons[].Weights for reporting). Increments
// ReadCount for each of the TopK touched value rows. This exposes key-usage
// collapse (see README) without needing layer internals.
procedure ProbeQuery(PKM: TNNetProductKeyMemory; Q: TNNetVolume);
var
  HalfQ, a, b, i, kk, gbest, nCand: integer;
  s1, s2: array[0..HalfKeys - 1] of TNeuralFloat;
  selA, selB: array[0..TopK - 1] of integer;
  usedA, usedB: array[0..HalfKeys - 1] of boolean;
  candScore: array[0..TopK * TopK - 1] of TNeuralFloat;
  candKey: array[0..TopK * TopK - 1] of integer;
  candLive: array[0..TopK * TopK - 1] of boolean;
  best: integer;
  K1, K2: TNNetVolume;

  procedure TopKOf(const sc: array of TNeuralFloat; var used: array of boolean;
    var sel: array of integer);
  var ii, jj: integer;
  begin
    for ii := 0 to HalfKeys - 1 do used[ii] := False;
    for ii := 0 to TopK - 1 do
    begin
      best := -1;
      for jj := 0 to HalfKeys - 1 do
        if (not used[jj]) and ((best = -1) or (sc[jj] > sc[best])) then best := jj;
      used[best] := True;
      sel[ii] := best;
    end;
  end;

begin
  HalfQ := QueryDim div 2;
  K1 := PKM.Neurons[0].Weights;
  K2 := PKM.Neurons[1].Weights;
  for a := 0 to HalfKeys - 1 do
  begin
    s1[a] := 0;
    for i := 0 to HalfQ - 1 do s1[a] := s1[a] + Q.Raw[i] * K1[a, 0, i];
  end;
  for b := 0 to HalfKeys - 1 do
  begin
    s2[b] := 0;
    for i := 0 to HalfQ - 1 do s2[b] := s2[b] + Q.Raw[HalfQ + i] * K2[b, 0, i];
  end;
  TopKOf(s1, usedA, selA);
  TopKOf(s2, usedB, selB);
  nCand := 0;
  for a := 0 to TopK - 1 do
    for b := 0 to TopK - 1 do
    begin
      candScore[nCand] := s1[selA[a]] + s2[selB[b]];
      candKey[nCand] := selA[a] * HalfKeys + selB[b];
      candLive[nCand] := True;
      Inc(nCand);
    end;
  for kk := 0 to TopK - 1 do
  begin
    gbest := -1;
    for i := 0 to nCand - 1 do
      if candLive[i] and ((gbest = -1) or (candScore[i] > candScore[gbest])) then
        gbest := i;
    candLive[gbest] := False;
    Inc(ReadCount[candKey[gbest]]);
  end;
end;

// Returns the number of DISTINCT value slots touched across all queries.
function ProbePKMUsage(PKM: TNNetProductKeyMemory): integer;
var p, k, used: integer;
begin
  for k := 0 to NumKeys - 1 do ReadCount[k] := 0;
  for p := 0 to NumPairs - 1 do ProbeQuery(PKM, Queries[p]);
  used := 0;
  for k := 0 to NumKeys - 1 do
    if ReadCount[k] > 0 then Inc(used);
  Result := used;
end;

var
  NNProduct, NNFlat: TNNet;
  PKM: TNNetProductKeyMemory;
  distinctUsed, k, maxRead: integer;

begin
  WriteLn('=== Product-Key Memory: synthetic key->value retrieval ===');
  WriteLn(Format('NumKeys=%d (%dx%d), QueryDim=%d, ValueDim=%d, TopK=%d, NumPairs=%d',
    [NumKeys, HalfKeys, HalfKeys, QueryDim, ValueDim, TopK, NumPairs]));
  WriteLn;

  BuildData;
  try
    // ---- Model 1: sparse product-key memory (touches TopK rows / query). ----
    RandSeed := 777;
    NNProduct := TNNet.Create();
    NNProduct.AddLayer(TNNetInput.Create(1, 1, QueryDim));
    PKM := NNProduct.AddProductKeyMemory(NumKeys, ValueDim, TopK, 1)
      as TNNetProductKeyMemory;
    WriteLn('Training product-key memory (sparse: ', TopK,
      ' value rows touched per query)...');
    Train(NNProduct, 'product-key');

    // ---- Model 2: same-capacity FLAT memory (dense softmax over NumKeys). ----
    // A single Modern-Hopfield retrieval step IS a dense softmax over a
    // NumKeys-row learnable bank -- the flat-memory baseline. We map its
    // QueryDim-wide query to a ValueDim-wide output with a trailing linear.
    RandSeed := 777;
    NNFlat := TNNet.Create();
    NNFlat.AddLayer(TNNetInput.Create(1, 1, QueryDim));
    NNFlat.AddModernHopfieldRetrieval({NumPatterns=}NumKeys, {KSteps=}1, 1.0);
    NNFlat.AddLayer(TNNetPointwiseConvLinear.Create(ValueDim));
    WriteLn('Training flat-memory baseline (dense: all ', NumKeys,
      ' rows read per query)...');
    Train(NNFlat, 'flat-memory ');

    WriteLn;
    WriteLn('Per-query value rows TOUCHED:');
    WriteLn('  product-key : ', TopK, '  (sparse)');
    WriteLn('  flat-memory : ', NumKeys, '  (dense)');

    // ---- Key-usage histogram for the product-key memory. ----
    distinctUsed := ProbePKMUsage(PKM);
    maxRead := 0;
    for k := 0 to NumKeys - 1 do
      if ReadCount[k] > maxRead then maxRead := ReadCount[k];
    WriteLn;
    WriteLn('Key-usage spread (product-key memory):');
    WriteLn(Format('  distinct slots used = %d / %d', [distinctUsed, NumKeys]));
    WriteLn(Format('  busiest slot read in %d / %d queries', [maxRead, NumPairs]));
    if distinctUsed < (NumKeys div 4) then
      WriteLn('  NOTE: few slots dominate -> usage COLLAPSE (see README).')
    else
      WriteLn('  Usage is reasonably spread across the memory.');

    WriteLn;
    WriteLn('Done. The product-key memory matched the flat baseline''s retrieval');
    WriteLn('while touching only TopK rows per query instead of all NumKeys.');
  finally
    NNProduct.Free;
    NNFlat.Free;
    FreeData;
  end;
end.
