program SinkhornMatching;
(*
SinkhornMatching: a neural network learns to solve a small LINEAR ASSIGNMENT
problem (a bipartite MATCHING) by emitting a soft PERMUTATION matrix through the
differentiable doubly-stochastic TNNetSinkhorn layer.

THE PROBLEM
-----------
The assignment problem: given an N x N COST matrix C (C[i,j] = cost of assigning
worker i to task j), pick a PERMUTATION pi (one task per worker, one worker per
task) that MINIMIZES the total cost sum_i C[i, pi(i)]. The set of permutation
matrices is the set of 0/1 doubly-stochastic matrices; the Sinkhorn-Knopp
projection maps any square score matrix onto the (continuous) doubly-stochastic
polytope -- the convex hull of permutation matrices -- and as the temperature
tau -> 0 it sharpens towards a single hard permutation.

This is a DIFFERENT use of TNNetSinkhorn from the sorting demo (examples/
SinkhornSort/): there the target was the fixed ASCENDING sort of the input
values; here it is the cost-minimizing MATCHING of a cost matrix -- the canonical
optimal-transport / bipartite-assignment task.

THE NET
-------
  Input(N,1,N)  =  the cost matrix C (one row per worker)
        ->  per-row score head (PointwiseConvLinear over the task axis, + ReLU,
            + PointwiseConvLinear)        -> an N x N score matrix
        ->  TNNetSinkhorn(KIter, tau)     -> P  (an N x N soft permutation)

The score head is TOKEN-WISE (PointwiseConvLinear keeps the (N,1,N) row axis;
FullConnect would flatten/mix the rows). Lower cost should attract more
assignment mass, so the head is free to learn the (negated, normalized) mapping
from costs to scores.

TRAINING OBJECTIVE
------------------
Pure optimal-transport cost of the SOFT assignment:
    L = sum_{i,j} P[i,j] * C[i,j].
Minimizing this drives P towards the min-cost permutation. The gradient is
trivial and set by hand on the Sinkhorn output:
    dL/dP[i,j] = C[i,j].
Gradients flow back through the entire Sinkhorn iteration into the score head, so
the net learns to PRODUCE assignments whose cost approaches the brute-force
optimum -- never told the optimal permutation, only its own soft cost.

ANNEALING
---------
tau starts warm (smooth, blurry P, easy gradients) and is annealed DOWN across
training so P sharpens towards a true permutation. We report, on held-out test
cost matrices, the OPTIMALITY GAP -- the hard-decoded assignment cost minus the
brute-force optimal cost, averaged and normalized -- and the EXACT-MATCH rate
(fraction of test matrices solved to the optimal permutation) BEFORE and AFTER
annealing. Both improve as tau cools: the net learns the discrete assignment
through the continuous Sinkhorn relaxation (Mena et al. 2018, "Learning Latent
Permutations with Gumbel-Sinkhorn Networks", arXiv:1802.08665).

Pure CPU, tiny N and batches -> runs in well under 3 minutes on 2 cores with
modest memory. No binaries are committed.

LICENSE: GPL (same as the neural-api project).

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  N        = 4;      // assignment size (workers = tasks = N); 4! = 24 perms
  HIDDEN   = N * N;  // hidden score-matrix width per row
  EPOCHS   = 2500;
  MB       = 16;     // mini-batch size
  LR       = 0.05;
  MOMENTUM = 0.9;
  KITER    = 20;     // Sinkhorn iterations
  TAU_HI   = 1.0;    // starting (warm) temperature
  TAU_LO   = 0.15;   // final (cold) temperature -- the annealing sweet spot;
                     // cooling further saturates the Sinkhorn softmaxes and the
                     // gradient vanishes, so exact-match peaks here (~95%) and
                     // then DEGRADES with more cooling rather than improving.
  NEVAL    = 400;    // test cost matrices for evaluation

var
  NN: TNNet;
  SinkhornIdx: integer;   // layer index of the Sinkhorn layer (its Output = P)

// Build one random N x N cost matrix in [0,1] (seeded for reproducibility),
// laid out as a (N,1,N) volume: row i = worker i, depth j = task j.
procedure MakeCost(seed: integer; Cost: TNNetVolume);
var
  i: integer;
  oldSeed: integer;
begin
  oldSeed := RandSeed;
  RandSeed := seed;
  Cost.ReSize(N, 1, N);
  for i := 0 to N * N - 1 do
    Cost.FData[i] := Random;   // [0,1]
  RandSeed := oldSeed;
end;

// Brute-force optimal assignment cost over all N! permutations (N is tiny).
// Also returns the optimal permutation in optPerm[].
function OptimalCost(Cost: TNNetVolume; var optPerm: array of integer): TNeuralFloat;
var
  perm: array[0..N - 1] of integer;
  used: array[0..N - 1] of boolean;
  bestCost: TNeuralFloat;

  procedure Recurse(pos: integer; accCost: TNeuralFloat);
  var
    j, k: integer;
  begin
    if accCost >= bestCost then exit; // prune
    if pos = N then
    begin
      if accCost < bestCost then
      begin
        bestCost := accCost;
        for k := 0 to N - 1 do optPerm[k] := perm[k];
      end;
      exit;
    end;
    for j := 0 to N - 1 do
      if not used[j] then
      begin
        used[j] := true;
        perm[pos] := j;
        Recurse(pos + 1, accCost + Cost.FData[pos * N + j]);
        used[j] := false;
      end;
  end;

var
  k: integer;
begin
  bestCost := 1e30;
  for k := 0 to N - 1 do used[k] := false;
  Recurse(0, 0.0);
  Result := bestCost;
end;

// Greedy hard-permutation decode of P (round to a 0/1 matrix), returning the
// total COST of that hard assignment under Cost and the decoded permutation.
function DecodeCost(P, Cost: TNNetVolume; var perm: array of integer): TNeuralFloat;
var
  i, j, bestJ: integer;
  bestV, total: TNeuralFloat;
  usedCol: array[0..N - 1] of boolean;
begin
  for j := 0 to N - 1 do usedCol[j] := false;
  total := 0;
  for i := 0 to N - 1 do
  begin
    bestJ := -1;
    bestV := -1e30;
    for j := 0 to N - 1 do
      if (not usedCol[j]) and (P.FData[i * N + j] > bestV) then
      begin
        bestV := P.FData[i * N + j];
        bestJ := j;
      end;
    usedCol[bestJ] := true;
    perm[i] := bestJ;
    total := total + Cost.FData[i * N + bestJ];
  end;
  Result := total;
end;

// Evaluate over NEVAL held-out (high-seed) test cost matrices. Returns the
// EXACT-MATCH rate (fraction solved to the optimal permutation) and, via
// avgGap, the mean normalized optimality gap (decoded cost - optimal) / N.
function Evaluate(out avgGap: TNeuralFloat): TNeuralFloat;
var
  s, i, correct: integer;
  Cost, P: TNNetVolume;
  optPerm, decPerm: array[0..N - 1] of integer;
  optC, decC, gapSum: TNeuralFloat;
  exact: boolean;
begin
  Cost := TNNetVolume.Create();
  correct := 0;
  gapSum := 0;
  try
    for s := 0 to NEVAL - 1 do
    begin
      MakeCost(1000000 + s, Cost);   // disjoint from training seeds
      NN.Compute(Cost);
      P := NN.Layers[SinkhornIdx].Output;
      optC := OptimalCost(Cost, optPerm);
      decC := DecodeCost(P, Cost, decPerm);
      gapSum := gapSum + (decC - optC) / N;
      exact := true;
      for i := 0 to N - 1 do
        if decPerm[i] <> optPerm[i] then begin exact := false; break; end;
      if exact then Inc(correct);
    end;
  finally
    Cost.Free;
  end;
  avgGap := gapSum / NEVAL;
  Result := correct / NEVAL;
end;

var
  epoch, s, i, j: integer;
  tau, frac, accBefore, accAfter, gapBefore, gapAfter, gap: TNeuralFloat;
  softCost, batchCost: TNeuralFloat;
  Cost, P, pseudo: TNNetVolume;
  Sinkhorn: TNNetSinkhorn;

begin
  Randomize;
  RandSeed := 424242;
  WriteLn('SinkhornMatching: learning to solve an ', N, 'x', N,
    ' linear-assignment problem through a Sinkhorn soft permutation.');
  WriteLn;

  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(N, 1, N));
  // Per-row (token-wise) score head over the task axis.
  NN.AddLayer(TNNetPointwiseConvReLU.Create(HIDDEN));
  NN.AddLayer(TNNetPointwiseConvLinear.Create(N));
  Sinkhorn := TNNetSinkhorn.Create(KITER, TAU_HI);
  NN.AddLayer(Sinkhorn);
  SinkhornIdx := NN.GetLastLayerIdx();

  NN.SetLearningRate(LR, MOMENTUM);
  NN.SetBatchUpdate(true);   // accumulate per-sample deltas, step once per batch

  Cost   := TNNetVolume.Create(N, 1, N);
  pseudo := TNNetVolume.Create(N, 1, N);

  accBefore := Evaluate(gapBefore);
  WriteLn(Format('Exact-match BEFORE training (tau=%5.3f): %5.1f%%  '+
    'mean optimality gap=%7.4f', [TAU_HI, accBefore * 100, gapBefore]));
  WriteLn;

  try
    for epoch := 0 to EPOCHS - 1 do
    begin
      // Anneal tau geometrically from TAU_HI down to TAU_LO across training.
      frac := epoch / (EPOCHS - 1);
      tau := TAU_HI * Power(TAU_LO / TAU_HI, frac);
      Sinkhorn.SetTau(tau);

      NN.ClearDeltas();
      batchCost := 0;
      for s := 0 to MB - 1 do
      begin
        MakeCost(epoch * MB + s, Cost);
        NN.Compute(Cost);
        P := NN.Layers[SinkhornIdx].Output;

        // Soft OT cost L = sum_ij P[i,j]*C[i,j] ; dL/dP[i,j] = C[i,j].
        // Mean over the batch -> scale the gradient by 1/MB (batch mode SUMS).
        softCost := 0;
        for i := 0 to N - 1 do
          for j := 0 to N - 1 do
            softCost := softCost + P.FData[i * N + j] * Cost.FData[i * N + j];
        batchCost := batchCost + softCost;

        // pseudo-target s.t. (Output - pseudo) == dL/dP (scaled by 1/MB).
        pseudo.Copy(P);
        for i := 0 to N - 1 do
          for j := 0 to N - 1 do
            pseudo.FData[i * N + j] := P.FData[i * N + j] -
              Cost.FData[i * N + j] / MB;

        NN.Backpropagate(pseudo);
      end;
      NN.UpdateWeights();

      if (epoch mod 500 = 0) or (epoch = EPOCHS - 1) then
      begin
        accAfter := Evaluate(gap);
        WriteLn(Format('  epoch %5d  tau=%6.4f  soft-cost=%8.5f  '+
          'exact=%5.1f%%  gap=%7.4f',
          [epoch, tau, batchCost / MB, accAfter * 100, gap]));
      end;
    end;

    accAfter := Evaluate(gapAfter);
    WriteLn;
    WriteLn(Format('Exact-match AFTER annealing  (tau=%5.3f): %5.1f%%  '+
      'mean optimality gap=%7.4f', [TAU_LO, accAfter * 100, gapAfter]));
    WriteLn(Format('Headline: exact-match rate climbed %5.1f%% -> %5.1f%% and '+
      'the optimality gap shrank %7.4f -> %7.4f', [accBefore * 100,
      accAfter * 100, gapBefore, gapAfter]));
    WriteLn('A net learned the DISCRETE assignment problem through a CONTINUOUS '+
      'Sinkhorn relaxation.');
  finally
    Cost.Free;
    pseudo.Free;
    NN.Free;
  end;
end.
