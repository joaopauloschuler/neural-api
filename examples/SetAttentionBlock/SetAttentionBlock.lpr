program SetAttentionBlock;
(*
SetAttentionBlock: a permutation-EQUIVARIANT set-to-set demo built on the new
TNNet.AddSAB builder -- the Set Attention Block (SAB) of the Set Transformer
(Lee et al. 2019, "Set Transformer", arXiv:1810.00825).

The SAB wraps a Multihead Attention Block (MAB) in two post-norm residual
sublayers:

    H   = LayerNorm(X + MAB(X, X))      (self-attention over the set)
    out = LayerNorm(H + FFN(H))         (token-wise 2-layer MLP)

In THIS fork the MAB is the multi-head ISAB block (AddInducedSetAttention):
H independent heads, each a per-token input projection feeding a single-head
TNNetInducedSetAttention with its own inducing bank, concatenated and run
through a per-token out-projection. Everything is a 1x1 (pointwise) conv over
the Depth axis except the symmetric softmax-over-inputs, so the block is
permutation-EQUIVARIANT: shuffle the input rows and the output rows follow the
same permutation.

TASK (chosen to REQUIRE cross-element interaction): for each element of a bag
of N scalars-in-a-d-vector, predict +1 if that element's feature 0 is ABOVE the
bag's mean of feature 0, else -1. A per-element MLP CANNOT solve this -- the
answer for one element depends on every OTHER element (the mean). The SAB stack
mixes across the set through attention, so it can.

This demo has TWO parts:

  PART 1 -- PERMUTATION EQUIVARIANCE (structural, before training). The SAB
    stack is evaluated on a random bag and on a shuffled copy. The output rows
    of the shuffled run match the correspondingly-permuted output rows of the
    original to < 1e-5 -- the headline structural property, for free.

  PART 2 -- ABOVE-THE-MEAN classification. A 2-layer SAB stack is trained and
    its per-element accuracy is compared against a per-element MLP baseline
    (TNNetPointwiseConvReLU + linear head) that has NO way to see other
    elements. The SAB stack learns the task; the per-element baseline is stuck
    near chance because it cannot compute the set mean.

Pure CPU, single thread, tiny dims; runs in well under a minute.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

const
  csN = 6;        // set size (elements per bag)
  csDim = 8;      // feature width d_model (divisible by Heads)
  csHeads = 2;    // attention heads (csDim mod csHeads = 0)
  csM = 3;        // ISAB inducing points
  csDFF = 16;     // FFN hidden width
  csTrain = 96;   // training bags
  csEpochs = 250;

// Build a random bag of csN elements (each a csDim-vector). The per-element
// target (channel 0) is +1 if feature 0 is above the bag mean of feature 0,
// else -1. The remaining feature channels are random noise the model must learn
// to ignore.
procedure MakeBag(Inp, Tgt: TNNetVolume);
var i, d: integer; mean, f0: TNeuralFloat;
begin
  mean := 0;
  for i := 0 to csN - 1 do
  begin
    for d := 0 to csDim - 1 do
      Inp[i, 0, d] := (Random - 0.5) * 3.0;
    mean := mean + Inp[i, 0, 0];
  end;
  mean := mean / csN;
  for i := 0 to csN - 1 do
  begin
    f0 := Inp[i, 0, 0];
    if f0 > mean then Tgt[i, 0, 0] := 1.0 else Tgt[i, 0, 0] := -1.0;
  end;
end;

// Fisher-Yates shuffle of the csN rows of Src into Dst, returning the perm
// used so the caller can compare permuted outputs.
procedure ShuffleRows(Src, Dst: TNNetVolume; var perm: array of integer);
var i, j, d, tmp: integer;
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

// Train one set-to-set net for csEpochs full-batch passes and return the final
// per-element classification accuracy on a fresh set of bags. The net's last
// layer must be a per-element (N,1,1) prediction; sign decides the class.
function TrainAndScore(NN: TNNet): TNeuralFloat;
var
  Ins, Tgs: array of TNNetVolume;
  ep, b, i, correct, total: integer;
  pred: TNeuralFloat;
begin
  SetLength(Ins, csTrain);
  SetLength(Tgs, csTrain);
  for b := 0 to csTrain - 1 do
  begin
    Ins[b] := TNNetVolume.Create(csN, 1, csDim);
    Tgs[b] := TNNetVolume.Create(csN, 1, 1);
    MakeBag(Ins[b], Tgs[b]);
  end;
  NN.SetLearningRate(0.01, 0.9);
  try
    for ep := 0 to csEpochs - 1 do
      for b := 0 to csTrain - 1 do
      begin
        NN.Compute(Ins[b]);
        NN.Backpropagate(Tgs[b]);
      end;
    // Score on freshly drawn bags.
    correct := 0; total := 0;
    for b := 0 to csTrain - 1 do
    begin
      MakeBag(Ins[b], Tgs[b]);
      NN.Compute(Ins[b]);
      for i := 0 to csN - 1 do
      begin
        pred := NN.GetLastLayer.Output[i, 0, 0];
        if (pred >= 0) = (Tgs[b][i, 0, 0] >= 0) then Inc(correct);
        Inc(total);
      end;
    end;
    Result := correct / total;
  finally
    for b := 0 to csTrain - 1 do begin Ins[b].Free; Tgs[b].Free; end;
  end;
end;

var
  NNEq, NNSab, NNMlp: TNNet;
  Bag, Shuffled, OutA, OutB: TNNetVolume;
  perm: array[0..csN - 1] of integer;
  i, j: integer;
  maxErr, dd: TNeuralFloat;
  accSab, accMlp: TNeuralFloat;
begin
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);
  RandSeed := 20190825;

  WriteLn('=== Set Attention Block (SAB) demo ===');
  WriteLn('Set N=', csN, '  d_model=', csDim, '  Heads=', csHeads,
    '  inducing M=', csM, '  d_ff=', csDFF);
  WriteLn;

  // ===================================================================
  // PART 1: permutation equivariance (structural, before training)
  // ===================================================================
  NNEq := TNNet.Create();
  NNEq.AddLayer(TNNetInput.Create(csN, 1, csDim));
  NNEq.AddSAB(csM, csHeads, csDFF);          // (N,1,d) -> (N,1,d)

  Bag := TNNetVolume.Create(csN, 1, csDim);
  Shuffled := TNNetVolume.Create(csN, 1, csDim);
  OutA := TNNetVolume.Create(csN, 1, csDim);
  OutB := TNNetVolume.Create(csN, 1, csDim);

  for i := 0 to csN - 1 do
    for j := 0 to csDim - 1 do
      Bag[i, 0, j] := (Random - 0.5) * 3.0;
  ShuffleRows(Bag, Shuffled, perm);

  NNEq.Compute(Bag);      OutA.Copy(NNEq.GetLastLayer.Output);
  NNEq.Compute(Shuffled); OutB.Copy(NNEq.GetLastLayer.Output);
  // Output of shuffled input, row i, must equal output of original at perm[i].
  maxErr := 0;
  for i := 0 to csN - 1 do
    for j := 0 to csDim - 1 do
    begin
      dd := Abs(OutB[i, 0, j] - OutA[perm[i], 0, j]);
      if dd > maxErr then maxErr := dd;
    end;
  WriteLn('PART 1 -- permutation equivariance');
  WriteLn('  SAB output max abs diff (shuffled[i] vs original[perm[i]]): ',
    maxErr:0:9);
  if maxErr < 1e-5 then
    WriteLn('  PASS: output rows follow the input permutation (< 1e-5).')
  else
    WriteLn('  FAIL: SAB output is not permutation-equivariant!');
  WriteLn;

  // ===================================================================
  // PART 2: above-the-mean classification -- SAB vs per-element MLP
  // ===================================================================
  // SAB stack: two SAB blocks (cross-element mixing) then a per-element linear
  // head down to a single +/-1 prediction.
  NNSab := TNNet.Create();
  NNSab.AddLayer(TNNetInput.Create(csN, 1, csDim));
  NNSab.AddSAB(csM, csHeads, csDFF);
  NNSab.AddSAB(csM, csHeads, csDFF);
  NNSab.AddLayer(TNNetPointwiseConvLinear.Create(1));   // per-element head

  // Per-element MLP baseline: the SAME per-token capacity but NO cross-element
  // path, so it cannot see the set mean. Expected to stay near chance.
  NNMlp := TNNet.Create();
  NNMlp.AddLayer(TNNetInput.Create(csN, 1, csDim));
  NNMlp.AddLayer(TNNetPointwiseConvReLU.Create(csDFF));
  NNMlp.AddLayer(TNNetPointwiseConvReLU.Create(csDFF));
  NNMlp.AddLayer(TNNetPointwiseConvLinear.Create(1));

  WriteLn('PART 2 -- above-the-mean per-element classification (accuracy, higher = better)');
  accSab := TrainAndScore(NNSab);
  accMlp := TrainAndScore(NNMlp);
  WriteLn('  SAB stack accuracy        : ', (accSab * 100):0:1, '%');
  WriteLn('  per-element MLP baseline  : ', (accMlp * 100):0:1, '%');
  if accSab > accMlp then
    WriteLn('  SAB beats the per-element baseline: attention sees the set mean.')
  else
    WriteLn('  (note) SAB did not beat the baseline this run.');
  WriteLn;
  WriteLn('Done.');

  OutB.Free; OutA.Free; Shuffled.Free; Bag.Free;
  NNMlp.Free; NNSab.Free; NNEq.Free;
end.
