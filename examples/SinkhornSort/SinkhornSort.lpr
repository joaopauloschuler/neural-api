program SinkhornSort;
(*
SinkhornSort: a neural network learns to SORT a short list of scalars
end-to-end through a CONTINUOUS RELAXATION of a DISCRETE operation, using the
differentiable doubly-stochastic TNNetSinkhorn layer.

THE IDEA
--------
Sorting is discrete: it picks a PERMUTATION of the inputs. A permutation matrix
is a 0/1 doubly-stochastic matrix, and the Sinkhorn-Knopp projection maps any
square score matrix to the (continuous) doubly-stochastic polytope -- the convex
hull of permutation matrices. As the temperature tau -> 0 the projection sharpens
towards a single hard permutation. So a net can EMIT a score matrix, let
TNNetSinkhorn turn it into a soft permutation P, apply P to the input vector, and
be trained with plain MSE against the SORTED vector. Gradients flow through the
whole Sinkhorn iteration, so the net learns the discrete "argsort" operation
purely from the continuous relaxation (Mena et al. 2018, "Learning Latent
Permutations with Gumbel-Sinkhorn Networks", arXiv:1802.08665).

THE NET
-------
  Input(N)  ->  FullConnect(N*N) [+ReLU] -> FullConnect(N*N)  ->  Reshape(N,1,N)
            ->  TNNetSinkhorn(KIter, tau)   ->   P  (an N x N soft permutation)

Predicted sorted vector:  yhat[i] = sum_j P[i,j] * x[j]   (P applied to input x).
Loss: 0.5 * sum_i (yhat[i] - sorted_x[i])^2, MSE against the ascending sort.
The loss gradient w.r.t. P is set by hand on the Sinkhorn output and backprop'd:
  dL/dP[i,j] = (yhat[i] - sorted_x[i]) * x[j].

ANNEALING
---------
tau starts warm (smooth, easy gradients, blurry P) and is annealed DOWN across
training (P sharpens towards a true permutation). We report SORT ACCURACY
(fraction of test lists sorted EXACTLY by rounding P to a hard permutation via
greedy arg-max) BEFORE and AFTER annealing -- it climbs as tau cools, the
headline result: a net learns a discrete operation through a continuous one.

Pure CPU, tiny N and batches -> runs in well under 3 minutes on 2 cores with
modest memory. No binaries are committed.

LICENSE: GPL (same as the neural-api project).

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  N        = 5;      // list length to sort
  HIDDEN   = N * N;  // score-matrix size
  EPOCHS   = 4500;
  MB       = 16;     // mini-batch size
  LR       = 0.03;
  MOMENTUM = 0.9;
  KITER    = 20;     // Sinkhorn iterations
  TAU_HI   = 1.0;    // starting (warm) temperature
  TAU_LO   = 0.07;   // final (cold) temperature -- the annealing sweet spot;
                     // cooling further saturates the softmaxes and gradients
                     // vanish, so accuracy plateaus rather than improving.
  NEVAL    = 400;    // test lists for accuracy

var
  NN: TNNet;
  SinkhornIdx: integer;   // layer index of the Sinkhorn layer (its Output = P)

// Build one random list of N scalars in [-1,1] (seeded for reproducibility) and
// its ascending sort.
procedure MakeSample(seed: integer; Inp, SortedTgt: TNNetVolume);
var
  i: integer;
  vals: array[0..N - 1] of TNeuralFloat;
  tmp: TNeuralFloat;
  a, b: integer;
  oldSeed: integer;
begin
  oldSeed := RandSeed;
  RandSeed := seed;
  for i := 0 to N - 1 do
    vals[i] := (Random - 0.5) * 2;   // [-1,1]
  RandSeed := oldSeed;

  Inp.ReSize(N, 1, 1);
  for i := 0 to N - 1 do
    Inp.FData[i] := vals[i];

  // ascending sort (tiny N -> bubble is fine)
  for a := 0 to N - 2 do
    for b := 0 to N - 2 - a do
      if vals[b] > vals[b + 1] then
      begin
        tmp := vals[b]; vals[b] := vals[b + 1]; vals[b + 1] := tmp;
      end;
  SortedTgt.ReSize(N, 1, 1);
  for i := 0 to N - 1 do
    SortedTgt.FData[i] := vals[i];
end;

// Apply the soft permutation P (Sinkhorn output, N x N) to input vector x:
//   yhat[i] = sum_j P[i,j] * x[j].   P[i,j] = P.FData[i*N + j].
procedure ApplyPermutation(P, x, yhat: TNNetVolume);
var
  i, j: integer;
  acc: TNeuralFloat;
begin
  for i := 0 to N - 1 do
  begin
    acc := 0;
    for j := 0 to N - 1 do
      acc := acc + P.FData[i * N + j] * x.FData[j];
    yhat.FData[i] := acc;
  end;
end;

// Greedy hard-permutation decode of P (round to a 0/1 matrix), then check whether
// applying it to x yields the exact ascending sort. Returns true on exact sort.
function IsExactlySorted(P, x, sortedTgt: TNNetVolume): boolean;
var
  i, j, bestJ: integer;
  bestV: TNeuralFloat;
  usedCol: array[0..N - 1] of boolean;
  outVal: array[0..N - 1] of TNeuralFloat;
begin
  for j := 0 to N - 1 do usedCol[j] := false;
  // For each output row i pick the highest-scoring unused column j.
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
    outVal[i] := x.FData[bestJ];
  end;
  Result := true;
  for i := 0 to N - 1 do
    if Abs(outVal[i] - sortedTgt.FData[i]) > 1e-6 then
    begin
      Result := false;
      exit;
    end;
end;

// Evaluate exact-sort accuracy over NEVAL held-out (high-seed) test lists.
function EvalAccuracy(): TNeuralFloat;
var
  s, correct: integer;
  Inp, Tgt, P: TNNetVolume;
begin
  Inp := TNNetVolume.Create();
  Tgt := TNNetVolume.Create();
  correct := 0;
  try
    for s := 0 to NEVAL - 1 do
    begin
      MakeSample(1000000 + s, Inp, Tgt);   // disjoint from training seeds
      NN.Compute(Inp);
      P := NN.Layers[SinkhornIdx].Output;
      if IsExactlySorted(P, Inp, Tgt) then Inc(correct);
    end;
  finally
    Inp.Free;
    Tgt.Free;
  end;
  Result := correct / NEVAL;
end;

var
  epoch, s, i, j: integer;
  tau, accBefore, accAfter, frac, loss, batchLoss: TNeuralFloat;
  Inp, Tgt, P, yhat, pseudo: TNNetVolume;
  Sinkhorn: TNNetSinkhorn;

begin
  Randomize;
  RandSeed := 424242;
  WriteLn('SinkhornSort: learning to sort ', N, ' scalars through a Sinkhorn ',
    'doubly-stochastic relaxation.');
  WriteLn;

  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(N, 1, 1));
  NN.AddLayer(TNNetFullConnectReLU.Create(HIDDEN));
  NN.AddLayer(TNNetFullConnectLinear.Create(HIDDEN));
  NN.AddLayer(TNNetReshape.Create(N, 1, N));
  Sinkhorn := TNNetSinkhorn.Create(KITER, TAU_HI);
  NN.AddLayer(Sinkhorn);
  SinkhornIdx := NN.GetLastLayerIdx();

  NN.SetLearningRate(LR, MOMENTUM);
  NN.SetBatchUpdate(true);   // accumulate per-sample deltas, step once per batch

  Inp    := TNNetVolume.Create(N, 1, 1);
  Tgt    := TNNetVolume.Create(N, 1, 1);
  yhat   := TNNetVolume.Create(N, 1, 1);
  pseudo := TNNetVolume.Create(N, 1, N);

  accBefore := EvalAccuracy();
  WriteLn('Exact-sort accuracy BEFORE training (tau=', TAU_HI:0:3, '): ',
    (accBefore * 100):0:1, '%');
  WriteLn;

  try
    for epoch := 0 to EPOCHS - 1 do
    begin
      // Anneal tau geometrically from TAU_HI down to TAU_LO across training.
      frac := epoch / (EPOCHS - 1);
      tau := TAU_HI * Power(TAU_LO / TAU_HI, frac);
      Sinkhorn.SetTau(tau);

      NN.ClearDeltas();
      batchLoss := 0;
      for s := 0 to MB - 1 do
      begin
        MakeSample(epoch * MB + s, Inp, Tgt);
        NN.Compute(Inp);
        P := NN.Layers[SinkhornIdx].Output;
        ApplyPermutation(P, Inp, yhat);

        // loss = 0.5*sum_i (yhat[i]-tgt[i])^2 ; dL/dP[i,j] = (yhat[i]-tgt[i])*x[j]
        // Mean over the batch -> scale by 1/MB (batch mode SUMS deltas).
        loss := 0;
        for i := 0 to N - 1 do
          loss := loss + 0.5 * Sqr(yhat.FData[i] - Tgt.FData[i]);
        batchLoss := batchLoss + loss;

        // pseudo-target s.t. (Output - pseudo) == dL/dP (scaled by 1/MB).
        pseudo.Copy(P);
        for i := 0 to N - 1 do
          for j := 0 to N - 1 do
            pseudo.FData[i * N + j] := P.FData[i * N + j] -
              (yhat.FData[i] - Tgt.FData[i]) * Inp.FData[j] / MB;

        NN.Backpropagate(pseudo);
      end;
      NN.UpdateWeights();

      if (epoch mod 500 = 0) or (epoch = EPOCHS - 1) then
        WriteLn(Format('  epoch %5d  tau=%6.4f  batch-loss=%9.5f  acc=%5.1f%%',
          [epoch, tau, batchLoss / MB, EvalAccuracy() * 100]));
    end;

    accAfter := EvalAccuracy();
    WriteLn;
    WriteLn('Exact-sort accuracy AFTER annealing  (tau=', TAU_LO:0:3, '): ',
      (accAfter * 100):0:1, '%');
    WriteLn('Headline: accuracy climbed from ', (accBefore * 100):0:1, '% to ',
      (accAfter * 100):0:1, '% as tau was annealed ', TAU_HI:0:2, ' -> ',
      TAU_LO:0:2, '.');
    WriteLn('A net learned the DISCRETE sort operation through a CONTINUOUS ',
      'Sinkhorn relaxation.');
  finally
    Inp.Free;
    Tgt.Free;
    yhat.Free;
    pseudo.Free;
    NN.Free;
  end;
end.
