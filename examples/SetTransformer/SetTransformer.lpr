program SetTransformer;
(*
SetTransformer: a permutation-INVARIANT set-learning demo built on the two new
Set-Transformer primitives (Lee et al. 2019, "Set Transformer", arXiv:1810.00825):

  TNNetInducedSetAttention (ISAB) -- a set-to-set block that replaces O(N^2)
    self-attention with an O(N*M) bottleneck through M learnable INDUCING POINTS:
        H = MAB(I, X)   (M inducing points attend over N inputs -> M summaries)
        Y = MAB(X, H)   (N inputs attend back over the M summaries -> N outputs)
    Output shape == input (N,1,d). M << N keeps the largest score matrix N x M.

  TNNetAttentionPooling (PMA) -- a learnable, permutation-INVARIANT pooler that
    collapses a variable-length set (N,1,d) to a FIXED (k,1,d) by letting k
    learnable SEED vectors cross-attend over the N inputs: PMA = MAB(S, X).
    k=1 is a single learned-query weighted-sum pool.

This demo has THREE parts:

  PART 1 -- PERMUTATION INVARIANCE (structural). A tiny ISAB->PMA(k=1) network is
    evaluated on a random bag and on a shuffled copy of the SAME bag. Because
    every softmax-over-inputs in ISAB and PMA is symmetric in the input rows, the
    pooled output is identical to < 1e-5 regardless of input order -- BEFORE any
    training. This is the headline property of the family.

  PART 2 -- MAX-OF-SET regression. We learn f(X) = max over the bag of the first
    feature. ISAB + PMA(k=1) is compared against a parameter-free MEAN-POOL
    baseline (TNNetAvgChannel head). The attention pooler -- which can learn to
    place almost all of its softmax mass on the single largest element -- beats
    mean-pool, which is forced to average and cannot pick out the max.

  PART 3 -- ISAB (M << N) vs FULL self-attention pooling, score-matrix size. ISAB
    routes N inputs through M inducing points, so its largest score matrix is
    N x M; a full self-attention pool over the same set forms an N x N matrix. We
    print both sizes to show the asymptotic saving while ISAB+PMA still solves
    the same task.

Pure CPU, single thread, tiny dims; runs in well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, neuralnetwork, neuralvolume, neuralfit;

const
  csN = 8;       // set size (number of elements per bag)
  csDim = 4;     // feature width d
  csM = 2;       // inducing points (M << N)

// Build a random bag of csN elements, each a csDim-vector in roughly [-1.5,1.5];
// the regression target is the MAX over the bag of feature 0.
procedure MakeBag(V: TNNetVolume; out TargetMax: TNeuralFloat);
var i, d: integer; x: TNeuralFloat;
begin
  TargetMax := -1e30;
  for i := 0 to csN - 1 do
    for d := 0 to csDim - 1 do
    begin
      x := (Random - 0.5) * 3.0;
      V[i, 0, d] := x;
      if (d = 0) and (x > TargetMax) then TargetMax := x;
    end;
end;

// Fisher-Yates shuffle of the csN rows (each a csDim-vector) of Src into Dst.
procedure ShuffleRows(Src, Dst: TNNetVolume);
var i, j, d: integer; perm: array[0..csN - 1] of integer; tmp: integer;
begin
  for i := 0 to csN - 1 do perm[i] := i;
  for i := csN - 1 downto 1 do
  begin
    j := Random(i + 1);
    tmp := perm[i]; perm[i] := perm[j]; perm[j] := tmp;
  end;
  for i := 0 to csN - 1 do
    for d := 0 to csDim - 1 do
      Dst[i, 0, d] := Src[perm[i], 0, d];
end;

// One full-batch training pass over a freshly drawn set of bags; returns the
// mean-squared error AFTER the pass. The network's last layer is a 1-value
// scalar prediction.
function TrainEpochs(NN: TNNet; NumBags, Epochs: integer): TNeuralFloat;
var
  Bag, Target: TNNetVolume;
  Bags: array of TNNetVolume;
  Targets: array of TNeuralFloat;
  ep, b: integer;
  tmax, pred, diff, sse: TNeuralFloat;
begin
  SetLength(Bags, NumBags);
  SetLength(Targets, NumBags);
  for b := 0 to NumBags - 1 do
  begin
    Bags[b] := TNNetVolume.Create(csN, 1, csDim);
    MakeBag(Bags[b], tmax);
    Targets[b] := tmax;
  end;
  Target := TNNetVolume.Create(1, 1, 1);
  sse := 0;
  try
    for ep := 0 to Epochs - 1 do
    begin
      sse := 0;
      for b := 0 to NumBags - 1 do
      begin
        Bag := Bags[b];
        Target[0, 0, 0] := Targets[b];
        NN.Compute(Bag);
        pred := NN.GetLastLayer.Output[0, 0, 0];
        diff := pred - Targets[b];
        sse := sse + diff * diff;
        NN.Backpropagate(Target);
      end;
    end;
    Result := sse / NumBags;
  finally
    Target.Free;
    for b := 0 to NumBags - 1 do Bags[b].Free;
  end;
end;

var
  NNInv, NNAttn, NNMean: TNNet;
  Bag, Shuffled, OutA, OutB: TNNetVolume;
  i, j: integer;
  maxErr, d: TNeuralFloat;
  mseAttn, mseMean: TNeuralFloat;
begin
  RandSeed := 20190001; // deterministic run, pure CPU, single thread (per-NN)

  WriteLn('=== Set Transformer (ISAB + PMA) demo ===');
  WriteLn('Set size N=', csN, '  d=', csDim, '  inducing M=', csM);
  WriteLn;

  // ===================================================================
  // PART 1: permutation invariance (structural, before training)
  // ===================================================================
  NNInv := TNNet.Create();
  NNInv.AddLayer(TNNetInput.Create(csN, 1, csDim));
  NNInv.AddInducedSetAttention(csM, 1);      // (N,1,d) -> (N,1,d), O(N*M)
  NNInv.AddAttentionPooling(1, 1);           // -> (1,1,d) learned-query pool

  Bag := TNNetVolume.Create(csN, 1, csDim);
  Shuffled := TNNetVolume.Create(csN, 1, csDim);
  OutA := TNNetVolume.Create(1, 1, csDim);
  OutB := TNNetVolume.Create(1, 1, csDim);

  for i := 0 to csN - 1 do
    for j := 0 to csDim - 1 do
      Bag[i, 0, j] := (Random - 0.5) * 3.0;
  ShuffleRows(Bag, Shuffled);

  NNInv.Compute(Bag);      OutA.Copy(NNInv.GetLastLayer.Output);
  NNInv.Compute(Shuffled); OutB.Copy(NNInv.GetLastLayer.Output);
  maxErr := 0;
  for i := 0 to OutA.Size - 1 do
  begin
    d := Abs(OutA.Raw[i] - OutB.Raw[i]);
    if d > maxErr then maxErr := d;
  end;
  WriteLn('PART 1 -- permutation invariance');
  WriteLn('  ISAB->PMA output max abs diff (original vs shuffled bag): ',
    maxErr:0:9);
  if maxErr < 1e-5 then
    WriteLn('  PASS: pooled output is invariant to input order (< 1e-5).')
  else
    WriteLn('  FAIL: output changed under permutation!');
  WriteLn;

  // ===================================================================
  // PART 2: max-of-set regression -- ISAB+PMA vs mean-pool baseline
  // ===================================================================
  // Attention model: ISAB set-to-set, then PMA(k=1) pool, then a tiny linear
  // head reads feature 0 of the pooled vector.
  NNAttn := TNNet.Create();
  NNAttn.AddLayer(TNNetInput.Create(csN, 1, csDim));
  NNAttn.AddInducedSetAttention(csM, 1);
  NNAttn.AddAttentionPooling(1, 1);          // (1,1,d) learned attention pool
  NNAttn.AddLayer(TNNetFullConnectLinear.Create(1));
  NNAttn.SetLearningRate(0.02, 0.9);

  // Mean-pool baseline: a shared per-element encoder then a PARAMETER-FREE
  // average over the set, then the same linear head. Mean-pool cannot single out
  // the max -- it is forced to average every element.
  NNMean := TNNet.Create();
  NNMean.AddLayer(TNNetInput.Create(csN, 1, csDim));
  NNMean.AddLayer(TNNetPointwiseConvReLU.Create(csDim)); // shared phi
  NNMean.AddLayer(TNNetAvgChannel.Create());             // symmetric mean pool
  NNMean.AddLayer(TNNetFullConnectLinear.Create(1));
  NNMean.SetLearningRate(0.02, 0.9);

  WriteLn('PART 2 -- max-of-set regression (MSE after training, lower = better)');
  mseAttn := TrainEpochs(NNAttn, 64, 300);
  mseMean := TrainEpochs(NNMean, 64, 300);
  WriteLn('  ISAB + PMA(k=1) MSE : ', mseAttn:0:5);
  WriteLn('  mean-pool baseline  : ', mseMean:0:5);
  if mseAttn < mseMean then
    WriteLn('  ISAB+PMA beats the mean-pool baseline on max-of-set.')
  else
    WriteLn('  (note) attention did not beat mean-pool this run.');
  WriteLn;

  // ===================================================================
  // PART 3: ISAB (M<<N) vs full self-attention pooling -- score-matrix size
  // ===================================================================
  WriteLn('PART 3 -- score-matrix size: ISAB bottleneck vs full self-attention');
  WriteLn('  full self-attention pool over N inputs: N x N = ',
    csN * csN, ' scores');
  WriteLn('  ISAB through M inducing points:         N x M = ',
    csN * csM, ' scores (stage 2; stage 1 is M x N)');
  WriteLn('  ISAB is O(N*M) vs O(N^2): saving grows as N grows for fixed M.');
  WriteLn;
  WriteLn('Done.');

  OutB.Free; OutA.Free; Shuffled.Free; Bag.Free;
  NNMean.Free; NNAttn.Free; NNInv.Free;
end.
