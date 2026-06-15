program ForwardForward;
(*
ForwardForward: reproduces Geoffrey Hinton's 2022 "The Forward-Forward
Algorithm: Some Preliminary Investigations" on a pure-CPU toy, using only
existing in-tree layers (NO new layer class is added).

What makes this example distinctive:
  It does NOT learn by end-to-end backpropagation. Forward-Forward (FF)
  NEVER forms a global loss and NEVER chains a backward pass between layers. It replaces the
  forward+backward pair with TWO forward passes (one on POSITIVE/real data,
  one on NEGATIVE/fake data) and trains EACH layer GREEDILY by its OWN LOCAL
  objective. No gradient ever flows from one layer into the layer below.

The algorithm (Hinton 2022), reusing only existing layers:
  - Stack a few TNNetFullConnectReLU layers. After EACH, insert a
    TNNetL2Normalize. The length-normalisation is essential: it forces a layer
    to feed only the DIRECTION of its activity to the next layer, never its
    MAGNITUDE. Without it, every layer could trivially satisfy its own goodness
    objective just by reading (and re-scaling) the previous layer's length, and
    no real features would be learned.
  - A layer's "goodness" is G = sum(activation^2) over its units
    (TNNetVolume.GetSumSqr on that layer's Output).
  - LOCAL objective per layer: push G ABOVE a threshold theta on POSITIVE
    (real) samples and BELOW theta on NEGATIVE (fake) samples, via the logistic
    loss  log(1 + exp(-(G - theta)))  on positives and
          log(1 + exp(+(G - theta)))  on negatives.
  - The per-unit gradient of that loss w.r.t. the layer's (pre-normalisation)
    activation a_j is  dL/dG * 2 * a_j, with
       dL/dG = -sigmoid(-(G - theta))   for positives  (push G up)
       dL/dG = +sigmoid(+(G - theta))   for negatives  (push G down).
    We write that vector into the layer's OutputError and call the layer's OWN
    BackpropagateCPU (which accumulates the weight gradient into Neurons[].Delta
    using the ReLU mask and the normalised input feeding the layer), passing NO
    error to the layer below. That "no downward error" is the defining FF
    property.
  - This is driven through the gradient-surgery idiom from the repo memory:
    NN.SetBatchUpdate(True) so Neurons[].Delta ACCUMULATES (the per-sample
    default would apply and zero it immediately), accumulate over a mini-batch,
    then NN.UpdateWeights() once.

Classification (Hinton's label-in-input trick):
  Embed the one-hot class label in the FIRST cClasses input slots. A POSITIVE
  sample carries the CORRECT label; a NEGATIVE sample carries a WRONG label
  (same features, different label). At INFERENCE we run the net once per
  candidate label and pick the label whose ACCUMULATED goodness (summed over
  all FF layers) is highest. The net is never asked to output a class directly;
  classification falls out of "which label makes the features look real".

Task: a tiny synthetic cClasses-way Gaussian-blob problem in cFeat dimensions
(few classes, few hidden units), sized so the greedy per-layer loop finishes
well under the few-minute CPU budget.

Built-in correctness gates (Halt(1) on failure, in the DoubleDescent house
style):
  GATE 1: after training, mean POSITIVE goodness must exceed mean NEGATIVE
          goodness at EVERY FF layer (the local contrast actually separated the
          two streams).
  GATE 2: the goodness-argmax classifier must BEAT CHANCE on a held-out set by
          a clear margin (accuracy well above 1/cClasses).

Pure CPU, no external data, single-threaded + fixed RandSeed (deterministic).

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
  cClasses  = 4;        // number of blob classes (chance = 1/4 = 25%)
  cFeat     = 2;        // raw feature dimension (2D blobs)
  cInput    = cClasses + cFeat;  // label one-hot overlaid in first cClasses slots
  cHidden1  = 30;       // first FF layer width
  cHidden2  = 30;       // second FF layer width
  cTrain    = 400;      // training points
  cTest     = 400;      // held-out test points
  cEpochs   = 60;       // greedy passes over the training set
  cBatch    = 20;       // mini-batch for accumulated FF updates
  cLR       = 0.02;     // per-layer learning rate
  cMomentum = 0.9;
  cTheta    = 2.0;      // goodness threshold
  cBlobSpread = 0.6;    // blob std-dev (overlap kept moderate)
  cSeed     = 424242;

type
  TSample = record
    Feat : array[0..cFeat - 1] of TNeuralFloat;
    Cls  : integer;
  end;
  TSampleArr = array of TSample;

var
  // Fixed blob centres (one per class), drawn once.
  Centres: array[0..cClasses - 1, 0..cFeat - 1] of TNeuralFloat;
  // The two trainable FF layers (their layer indices in NN).
  FFIdx: array[0..1] of integer;

// ---------------------------------------------------------------------------
// Data generation: cClasses Gaussian blobs in cFeat-D space.
// ---------------------------------------------------------------------------
procedure MakeCentres;
var C, F: integer;
begin
  for C := 0 to cClasses - 1 do
    for F := 0 to cFeat - 1 do
      Centres[C, F] := RandG(0, 2.0);  // spread the centres apart
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

// Build the input volume: one-hot label in the first cClasses slots, raw
// features after it. Label may differ from the true class (negative sample).
procedure FillInput(V: TNNetVolume; const S: TSample; Lbl: integer);
var I, F: integer;
begin
  for I := 0 to cClasses - 1 do V.FData[I] := 0;
  V.FData[Lbl] := 1.0;
  for F := 0 to cFeat - 1 do
    V.FData[cClasses + F] := S.Feat[F];
end;

// ---------------------------------------------------------------------------
// Network: Input -> [FullConnectReLU -> L2Normalize] x2.
// The L2Normalize layers use full-volume mode (axis=1) so each (H,1,1) hidden
// vector is normalised to unit length -> only DIRECTION feeds forward.
// ---------------------------------------------------------------------------
procedure BuildNet(out NN: TNNet);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(cInput, 1, 1));
  FFIdx[0] := NN.AddLayer(TNNetFullConnectReLU.Create(cHidden1)).LayerIdx;
  NN.AddLayer(TNNetL2Normalize.Create(1));   // axis 1 = whole-vector unit norm
  FFIdx[1] := NN.AddLayer(TNNetFullConnectReLU.Create(cHidden2)).LayerIdx;
  NN.AddLayer(TNNetL2Normalize.Create(1));
  NN.SetLearningRate(cLR, cMomentum);
  NN.SetL2Decay(0.0);
  NN.SetBatchUpdate(True);  // CRITICAL: accumulate Neurons[].Delta across batch
end;

// Goodness of FF layer #k for the input currently held in the net (must Compute
// first): G = sum of squared activations of that layer's units.
function LayerGoodness(NN: TNNet; k: integer): TNeuralFloat;
begin
  Result := NN.Layers[FFIdx[k]].Output.GetSumSqr();
end;

function Sigmoid(x: TNeuralFloat): TNeuralFloat;
begin
  Result := 1.0 / (1.0 + Exp(-x));
end;

// Apply the FF LOCAL update for ONE sample to ONE FF layer. The net must have
// been Computed on this sample already. Positive=true pushes goodness up,
// Positive=false pushes it down. Writes the local gradient into the layer's
// OutputError and calls the layer's OWN BackpropagateCPU (accumulates the
// weight delta; does NOT touch the layer below). Returns this layer's goodness.
function FFLocalStep(NN: TNNet; k: integer; Positive: boolean): TNeuralFloat;
var
  Lay: TNNetLayer;
  G, dLdG, Scale: TNeuralFloat;
  j: integer;
begin
  Lay := NN.Layers[FFIdx[k]];
  G := Lay.Output.GetSumSqr();
  Result := G;
  if Positive then
    dLdG := -Sigmoid(-(G - cTheta))   // push G above theta
  else
    dLdG := +Sigmoid(+(G - cTheta));  // push G below theta
  // Mean over the mini-batch: scale the per-sample gradient by 1/cBatch so the
  // accumulated Delta is the batch MEAN (batch mode SUMS deltas without
  // averaging -- see repo memory note).
  Scale := (dLdG * 2.0) / cBatch;
  // dL/da_j = dL/dG * 2 * a_j  (a_j is this layer's pre-normalisation output).
  for j := 0 to Lay.Output.Size - 1 do
    Lay.OutputError.FData[j] := Scale * Lay.Output.FData[j];
  // Cast to TNNetFullConnect to reach BackpropagateCPU (accumulates THIS
  // layer's weight gradient via the ReLU mask + normalised input; it does NOT
  // call FPrevLayer.Backpropagate, so no error crosses into the layer below).
  TNNetFullConnect(Lay).BackpropagateCPU();
end;

// ---------------------------------------------------------------------------
// Inference: run the net once per candidate label, sum goodness over both FF
// layers, pick the argmax label.
// ---------------------------------------------------------------------------
function Classify(NN: TNNet; Inp: TNNetVolume; const S: TSample): integer;
var
  Lbl, Best: integer;
  G, BestG: TNeuralFloat;
begin
  Best := 0; BestG := -1e30;
  for Lbl := 0 to cClasses - 1 do
  begin
    FillInput(Inp, S, Lbl);
    NN.Compute(Inp);
    G := LayerGoodness(NN, 0) + LayerGoodness(NN, 1);
    if G > BestG then begin BestG := G; Best := Lbl; end;
  end;
  Result := Best;
end;

function TestAccuracy(NN: TNNet; const TestSet: TSampleArr): TNeuralFloat;
var I, Correct: integer; Inp: TNNetVolume;
begin
  Inp := TNNetVolume.Create(cInput, 1, 1);
  Correct := 0;
  try
    for I := 0 to High(TestSet) do
      if Classify(NN, Inp, TestSet[I]) = TestSet[I].Cls then Inc(Correct);
  finally
    Inp.Free;
  end;
  Result := Correct / Length(TestSet);
end;

// Mean positive / negative goodness per FF layer over a sample set (diagnostic
// + GATE 1). For each sample: positive = correct label, negative = a fixed
// wrong label (cls+1 mod cClasses).
procedure MeasureGoodness(NN: TNNet; const SS: TSampleArr;
  out PosG, NegG: array of TNeuralFloat);
var
  I, k, NegLbl: integer;
  Inp: TNNetVolume;
begin
  Inp := TNNetVolume.Create(cInput, 1, 1);
  for k := 0 to 1 do begin PosG[k] := 0; NegG[k] := 0; end;
  try
    for I := 0 to High(SS) do
    begin
      // positive
      FillInput(Inp, SS[I], SS[I].Cls);
      NN.Compute(Inp);
      for k := 0 to 1 do PosG[k] := PosG[k] + LayerGoodness(NN, k);
      // negative (a wrong label)
      NegLbl := (SS[I].Cls + 1) mod cClasses;
      FillInput(Inp, SS[I], NegLbl);
      NN.Compute(Inp);
      for k := 0 to 1 do NegG[k] := NegG[k] + LayerGoodness(NN, k);
    end;
  finally
    Inp.Free;
  end;
  for k := 0 to 1 do
  begin
    PosG[k] := PosG[k] / Length(SS);
    NegG[k] := NegG[k] / Length(SS);
  end;
end;

// One ASCII bar (value mapped to fixed width, scaled by VMax).
function Bar(V, VMax: TNeuralFloat; Width: integer): string;
var N, I: integer;
begin
  if VMax <= 0 then VMax := 1;
  N := Round((V / VMax) * Width);
  if N < 0 then N := 0;
  if N > Width then N := Width;
  Result := '';
  for I := 1 to N do Result := Result + '#';
  for I := N + 1 to Width do Result := Result + ' ';
end;

// ---------------------------------------------------------------------------
// Training: for each epoch, walk the training set in mini-batches. Within a
// batch, for every sample feed BOTH a positive (correct label) and a negative
// (a randomly chosen wrong label) pass, accumulating each FF layer's local
// gradient. After the batch, one UpdateWeights step.
// ---------------------------------------------------------------------------
procedure Train(NN: TNNet; const TrainSet: TSampleArr);
var
  Epoch, I, k, b, NegLbl: integer;
  Inp: TNNetVolume;
  Order: array of integer;
  Tmp, J: integer;
  BatchCount: integer;
begin
  Inp := TNNetVolume.Create(cInput, 1, 1);
  SetLength(Order, Length(TrainSet));
  for I := 0 to High(Order) do Order[I] := I;
  try
    for Epoch := 1 to cEpochs do
    begin
      // shuffle
      for I := High(Order) downto 1 do
      begin
        J := Random(I + 1);
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;
      I := 0;
      while I <= High(Order) do
      begin
        NN.ClearDeltas();
        BatchCount := 0;
        b := 0;
        while (b < cBatch) and (I <= High(Order)) do
        begin
          // POSITIVE pass: correct label.
          FillInput(Inp, TrainSet[Order[I]], TrainSet[Order[I]].Cls);
          NN.Compute(Inp);
          for k := 0 to 1 do FFLocalStep(NN, k, True);
          // NEGATIVE pass: a wrong label (uniformly chosen among the others).
          NegLbl := Random(cClasses - 1);
          if NegLbl >= TrainSet[Order[I]].Cls then Inc(NegLbl);
          FillInput(Inp, TrainSet[Order[I]], NegLbl);
          NN.Compute(Inp);
          for k := 0 to 1 do FFLocalStep(NN, k, False);
          Inc(BatchCount);
          Inc(I); Inc(b);
        end;
        if BatchCount > 0 then NN.UpdateWeights();
      end;
      if (Epoch mod 10 = 0) or (Epoch = 1) then
        Write('.');
    end;
    WriteLn;
  finally
    Inp.Free;
  end;
end;

var
  NN: TNNet;
  TrainSet, TestSet: TSampleArr;
  PosG, NegG: array[0..1] of TNeuralFloat;
  Acc, Chance, Gmax: TNeuralFloat;
  k: integer;
  StartTime, EndTime: TDateTime;
  Gate1, Gate2: boolean;
begin
  RandSeed := cSeed;

  WriteLn('================================================================');
  WriteLn('Forward-Forward: per-layer LOCAL goodness training (NO backprop).');
  WriteLn('================================================================');
  WriteLn(Format('Task: %d Gaussian blobs in %dD; label one-hot overlaid in the',
    [cClasses, cFeat]));
  WriteLn(Format('first %d input slots (input dim=%d).  Net: Input -> '
    + '[FCReLU(%d)->L2Norm]', [cClasses, cInput, cHidden1]));
  WriteLn(Format('-> [FCReLU(%d)->L2Norm].  theta=%.2f  LR=%.3f  mom=%.2f  '
    + 'epochs=%d', [cHidden2, cTheta, cLR, cMomentum, cEpochs]));
  WriteLn(Format('batch=%d  train=%d  test=%d  RandSeed=%d',
    [cBatch, cTrain, cTest, cSeed]));
  WriteLn('POSITIVE = correct label; NEGATIVE = a wrong label. Each FF layer is');
  WriteLn('trained ONLY by its own goodness contrast; no error crosses layers.');
  WriteLn;

  // Build data + net (reseed kept fixed; single-threaded -> deterministic).
  MakeCentres;
  BuildSet(TrainSet, cTrain);
  BuildSet(TestSet, cTest);
  BuildNet(NN);

  // Goodness BEFORE training (sanity baseline).
  MeasureGoodness(NN, TestSet, PosG, NegG);
  WriteLn('Pre-training goodness (untrained net):');
  for k := 0 to 1 do
    WriteLn(Format('  FF layer %d: pos=%.3f  neg=%.3f', [k, PosG[k], NegG[k]]));
  WriteLn;

  StartTime := Now;
  Write('Training (FF, two forward passes per sample) ');
  Train(NN, TrainSet);
  EndTime := Now;

  // ---- Reporting ----
  MeasureGoodness(NN, TestSet, PosG, NegG);
  Gmax := 0;
  for k := 0 to 1 do
  begin
    if PosG[k] > Gmax then Gmax := PosG[k];
    if NegG[k] > Gmax then Gmax := NegG[k];
  end;

  WriteLn;
  WriteLn('=== Per-layer goodness on held-out set (after training) ===');
  WriteLn(Format('theta = %.2f.  POSITIVE should sit clearly ABOVE NEGATIVE.',
    [cTheta]));
  WriteLn('  layer    posG    negG  margin | pos bar / neg bar (scaled)');
  Gate1 := True;
  for k := 0 to 1 do
  begin
    if PosG[k] <= NegG[k] then Gate1 := False;
    WriteLn(Format('  FF %d   %6.3f  %6.3f  %6.3f | %s',
      [k, PosG[k], NegG[k], PosG[k] - NegG[k], Bar(PosG[k], Gmax, 24)]));
    WriteLn(Format('                                  | %s',
      [Bar(NegG[k], Gmax, 24)]));
  end;

  Acc := TestAccuracy(NN, TestSet);
  Chance := 1.0 / cClasses;

  WriteLn;
  WriteLn('=== Goodness-argmax classifier (held-out) ===');
  WriteLn(Format('  accuracy = %.3f   (chance = %.3f for %d classes)',
    [Acc, Chance, cClasses]));

  WriteLn;
  WriteLn('=== Correctness gates ===');
  if Gate1 then
    WriteLn('[PASS] GATE 1: positive goodness exceeds negative goodness at '
      + 'EVERY FF layer.')
  else
    WriteLn('[FAIL] GATE 1: some FF layer did NOT separate pos/neg goodness '
      + '(tune theta / LR / negatives).');

  // Beat chance by a clear margin (half-way to perfect, at least +15pp).
  Gate2 := Acc > Chance + 0.15;
  if Gate2 then
    WriteLn(Format('[PASS] GATE 2: goodness-argmax accuracy %.3f beats chance '
      + '%.3f by a clear margin.', [Acc, Chance]))
  else
    WriteLn(Format('[FAIL] GATE 2: accuracy %.3f did NOT clear chance %.3f by '
      + 'the required margin.', [Acc, Chance]));

  WriteLn;
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  NN.Free;

  if Gate1 and Gate2 then
    WriteLn('=> ALL GATES PASS: Forward-Forward learned features by a purely '
      + 'local objective.')
  else
  begin
    WriteLn('=> SOME GATES FAILED (see above).');
    Halt(1);
  end;
end.
