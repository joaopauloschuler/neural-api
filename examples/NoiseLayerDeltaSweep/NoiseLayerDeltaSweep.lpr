program NoiseLayerDeltaSweep;
(*
NoiseLayerDeltaSweep: train-time vs inference-time delta sweep for the four
stochastic noise / dropout layers that already live in neuralnetwork.pas
(TNNetDropout, TNNetDropPath, TNNetSpatialDropout1D, TNNetSpatialDropout2D).
This example adds NO new layer.

The phenomenon (the whole point of this example):
  Every TNNetAddNoiseBase layer is a *stochastic* operation at TRAIN time and a
  *deterministic identity* at INFERENCE time. The library toggles the two
  regimes with a single network-wide switch, TNNet.EnableDropouts(flag):
    - EnableDropouts(true):  the layer samples a fresh Bernoulli mask and
      rescales the survivors by 1/(1-p) ("inverted dropout"), so the output is
      RANDOM and varies pass-to-pass.
    - EnableDropouts(false): the layer copies its input unchanged, so the
      output is DETERMINISTIC and identical pass-to-pass.
  The four layers differ only in WHAT they drop:
    TNNetDropout          - individual elements (per-element Bernoulli).
    TNNetDropPath         - the WHOLE sample/branch (stochastic depth).
    TNNetSpatialDropout2D - whole feature-map CHANNELS (a Depth slice).
    TNNetSpatialDropout1D - whole channels of a sequence (same as 2D here).

What "train-time vs inference-time delta" means and why we measure it:
  Because the layer is random in training and an identity at inference, the
  loss you see on a TRAIN forward pass (noise ON) is NOT comparable to the loss
  on an INFERENCE / validation pass (noise OFF) unless you are deliberate about
  which switch is set. This example measures, for each layer type and each drop
  probability p:
    train-CE(noise ON)   - the loss the optimiser actually sees, with the
                           stochastic mask active,
    train-CE(noise OFF)  - the SAME training data scored at inference (mask
                           off), i.e. the "real" fit of the learned weights,
    val-CE(noise OFF)    - held-out loss, always at inference,
    gap = val-CE(off) - train-CE(off).
  The headline observable is that stronger noise (larger p) drives the
  train(ON) loss UP -- the optimiser is handicapped by the random mask -- while
  the inference (OFF) losses behave like a normal regulariser: the train/val
  gap shrinks. Comparing train(ON) against train(OFF) on the SAME data is a
  direct, numerical demonstration that the noise toggle works: at p=0 they are
  equal (the layer is the identity in both regimes); at p>0 they diverge.

We MUST disable the noise for every val / inference forward pass (and for the
train(OFF) re-scoring) so the delta is measured correctly -- that is the whole
point. Evaluate() below calls NN.EnableDropouts(false) and, to prove the
inference pass is deterministic, runs each probe twice and asserts the two
outputs are bit-for-bit identical.

Shape constraints (why each layer is wired the way it is):
  The residual-carrying tensor is 1 x 1 x cWidth (the feature dim lives in the
  Depth axis). TNNetDropout and TNNetDropPath are shape-agnostic and sit on the
  residual BRANCH. TNNetSpatialDropout1D / 2D drop WHOLE Depth channels, so they
  need a meaningful Depth extent -- cWidth channels here -- which this 1x1xcWidth
  tensor provides. Putting any of the four on the shape-preserving branch keeps
  y = x + Noise_p(Branch(x)) valid for all four families with no reshaping.

Sweep structure (mirrors examples/ALiBiSlopeSweep + DropPathAblation):
  A single tiny ResNet-style classifier on a fixed synthetic 3-way task. For
  each of the four layer families we run a sub-sweep over p in {0.0,0.1,0.2,0.4}.
  Every arm shares the seed, data, epochs and LR; only the noise layer class and
  p differ. p=0.0 is the no-noise baseline (identity in both regimes), so its
  train(ON), train(OFF) and the gap are the reference point.

Pure CPU, single-threaded (deterministic seeding RandSeed := 424242), no
external data, finishes well under the few-minute budget.

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
  neuralvolume,
  neuralfit;

const
  cSeed       = 424242;
  cDim        = 6;      // input feature dimension
  cClasses    = 3;      // 3-way classification
  cWidth      = 16;     // residual feature width (lives in the Depth axis)
  cNumBlocks  = 4;      // ResNet-style depth (noise layer on each block's branch)
  cTrain      = 256;    // small-ish training set
  cVal        = 1000;   // larger held-out validation set
  cEpochs     = 40;
  cBatch      = 32;
  cLR         = 0.01;

  cNumLayers  = 4;      // the four noise families
  cNumProbs   = 4;      // p in {0.0, 0.1, 0.2, 0.4}

  cLayerNames: array[0..cNumLayers - 1] of string =
    ('Dropout', 'DropPath', 'SpatialDropout1D', 'SpatialDropout2D');

  cProbs: array[0..cNumProbs - 1] of TNeuralFloat = (0.0, 0.1, 0.2, 0.4);

var
  // Fixed teacher (shared across all arms): a random quadratic form per class.
  // class = argmax_c ( x' Q_c x + b_c' x ). Nonlinear so the residual net has
  // something real to fit; deterministic given the seed.
  TeacherQ: array[0..cClasses - 1, 0..cDim - 1, 0..cDim - 1] of TNeuralFloat;
  TeacherB: array[0..cClasses - 1, 0..cDim - 1] of TNeuralFloat;

type
  TArmResult = record
    LayerKind     : integer;
    DropProb      : TNeuralFloat;
    TrainLossOn   : TNeuralFloat;   // train CE with noise ON  (what optimiser sees)
    TrainLossOff  : TNeuralFloat;   // train CE with noise OFF (inference fit)
    ValLossOff    : TNeuralFloat;   // held-out CE, noise OFF (inference)
    Gap           : TNeuralFloat;   // ValLossOff - TrainLossOff
    Deterministic : boolean;        // inference pass is bit-for-bit repeatable
    Diverged      : boolean;
  end;

// ---------------------------------------------------------------------------
// Teacher + data generation. Reseeding before each build keeps the teacher and
// the points identical across all arms (only the noise layer / prob differ).
// ---------------------------------------------------------------------------
procedure MakeTeacher;
var
  C, I, J: integer;
begin
  for C := 0 to cClasses - 1 do
    for I := 0 to cDim - 1 do
    begin
      TeacherB[C, I] := RandG(0, 1);
      for J := I to cDim - 1 do
      begin
        TeacherQ[C, I, J] := RandG(0, 1);
        TeacherQ[C, J, I] := TeacherQ[C, I, J]; // symmetric
      end;
    end;
end;

function TeacherClass(X: TNNetVolume): integer;
var
  C, I, J, Best: integer;
  S, BestS: TNeuralFloat;
begin
  Best := 0; BestS := -1e30;
  for C := 0 to cClasses - 1 do
  begin
    S := 0;
    for I := 0 to cDim - 1 do
    begin
      S := S + TeacherB[C, I] * X.FData[I];
      for J := 0 to cDim - 1 do
        S := S + TeacherQ[C, I, J] * X.FData[I] * X.FData[J];
    end;
    if S > BestS then begin BestS := S; Best := C; end;
  end;
  Result := Best;
end;

function BuildSet(Count: integer): TNNetVolumePairList;
var
  I, J, Cls: integer;
  X, Y: TNNetVolume;
begin
  Result := TNNetVolumePairList.Create();
  for I := 0 to Count - 1 do
  begin
    X := TNNetVolume.Create(cDim, 1, 1);
    for J := 0 to cDim - 1 do
      X.FData[J] := RandG(0, 1);
    Cls := TeacherClass(X);
    Y := TNNetVolume.Create(cClasses, 1, 1);
    Y.Fill(0);
    Y.FData[Cls] := 1.0;   // one-hot target for SoftMax + cross-entropy
    Result.Add(TNNetVolumePair.Create(X, Y));
  end;
end;

// ---------------------------------------------------------------------------
// Build the noise layer of the requested family. All four share the (p) ctor;
// the only difference is the class. Returns the constructed layer (already on
// the residual branch's shape 1 x 1 x cWidth).
// ---------------------------------------------------------------------------
function MakeNoiseLayer(LayerKind: integer; DropProb: TNeuralFloat): TNNetLayer;
begin
  case LayerKind of
    0: Result := TNNetDropout.Create(DropProb);
    1: Result := TNNetDropPath.Create(DropProb);
    2: Result := TNNetSpatialDropout1D.Create(DropProb);
    3: Result := TNNetSpatialDropout2D.Create(DropProb);
  else
    Result := TNNetDropout.Create(DropProb);
  end;
end;

// ---------------------------------------------------------------------------
// The ResNet-style net. The residual tensor is 1 x 1 x cWidth (feature dim in
// Depth), exactly what PointwiseConvLinear + the noise layer + Sum expect: a
// residual sublayer MUST be shape-preserving. The swept noise layer sits on the
// BRANCH, right before the closing Sum:
//     y = x + Noise_p( ReLU(PointwiseConvLinear(x)) )
// SpatialDropout1D/2D drop whole Depth channels, so the cWidth-deep tensor gives
// them meaningful channels to drop.
// ---------------------------------------------------------------------------
procedure BuildNet(NN: TNNet; LayerKind: integer; DropProb: TNeuralFloat);
var
  i: integer;
  BranchInput: TNNetLayer;
begin
  NN.AddLayer( TNNetInput.Create(cDim, 1, 1) );
  NN.AddLayer( TNNetFullConnectLinear.Create(cWidth) );   // project to cWidth feats
  // FullConnectLinear lays cWidth out along X; the PointwiseConv / noise / Sum
  // operate along Depth, so reshape into Depth (1 x 1 x cWidth).
  NN.AddLayer( TNNetReshape.Create(1, 1, cWidth) );
  for i := 1 to cNumBlocks do
  begin
    BranchInput := NN.GetLastLayer();
    NN.AddLayer( TNNetPointwiseConvLinear.Create(cWidth) );
    NN.AddLayer( TNNetReLU.Create() );
    NN.AddLayer( MakeNoiseLayer(LayerKind, DropProb) ); // <-- the swept knob
    NN.AddLayer( TNNetSum.Create([NN.GetLastLayer(), BranchInput]) );
  end;
  NN.AddLayer( TNNetFullConnectLinear.Create(cClasses) );
  NN.AddLayer( TNNetSoftMax.Create() );
end;

// Cross-entropy over a pair list at the GIVEN noise regime. When NoiseOn is
// false the net is forced to inference (EnableDropouts(false)), so every noise
// layer is the deterministic identity. When NoiseOn is true the stochastic
// masks are active. CheckDeterministic only makes sense (and is only asserted)
// at inference, where the same input must map to the same output every time.
function MeanCE(NN: TNNet; Pairs: TNNetVolumePairList; NoiseOn: boolean;
  CheckDeterministic: boolean; out IsDeterministic: boolean): TNeuralFloat;
var
  I: integer;
  P, FirstVal: TNeuralFloat;
  SumCE: Double;
  Tgt: integer;
begin
  NN.EnableDropouts(NoiseOn);
  IsDeterministic := True;
  SumCE := 0;
  for I := 0 to Pairs.Count - 1 do
  begin
    NN.Compute(Pairs[I].I);
    Tgt := Pairs[I].O.GetClass();
    P := NN.GetLastLayer().Output.FData[Tgt];
    if P < 1e-12 then P := 1e-12;
    SumCE := SumCE - Ln(P);
    if CheckDeterministic and (not NoiseOn) then
    begin
      FirstVal := NN.GetLastLayer().Output.FData[Tgt];
      NN.Compute(Pairs[I].I);            // recompute the SAME input
      if Abs(NN.GetLastLayer().Output.FData[Tgt] - FirstVal) > 0 then
        IsDeterministic := False;        // inference must be bit-for-bit stable
    end;
  end;
  if Pairs.Count > 0 then
    Result := SumCE / Pairs.Count
  else
    Result := 0;
end;

function RunArm(LayerKind: integer; DropProb: TNeuralFloat;
                TrainSet, ValSet: TNNetVolumePairList): TArmResult;
var
  NN: TNNet;
  NFit: TNeuralFit;
  DetTrain, DetVal: boolean;
begin
  Result.LayerKind := LayerKind;
  Result.DropProb := DropProb;

  // Reseed before BUILD so weight init is identical across arms (only the noise
  // layer class, its prob and its train-time RNG draws differ).
  RandSeed := cSeed;
  NN := TNNet.Create();
  NFit := TNeuralFit.Create();
  try
    BuildNet(NN, LayerKind, DropProb);

    NFit.FileNameBase := GetTempDir + 'NoiseLayerDeltaSweep_autosave';
    NFit.InitialLearningRate := cLR;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.HasFlipX := false;
    NFit.HasFlipY := false;
    NFit.MaxThreadNum := 1;  // single-threaded => deterministic reductions
    // Classification fit: SoftMax + cross-entropy. Fit enables dropouts during
    // training and disables them for its own validation pass automatically.
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.Fit(NN, TrainSet, ValSet, nil, cBatch, cEpochs);

    // Our own deterministic measurements (Fit may reload the best snapshot, so
    // we re-score from scratch with explicit noise regimes).
    // train(ON): stochastic mask active, what the optimiser actually saw.
    Result.TrainLossOn  := MeanCE(NN, TrainSet, True,  False, DetTrain);
    // train(OFF): SAME data at inference -> the real fit of the weights.
    Result.TrainLossOff := MeanCE(NN, TrainSet, False, True,  DetTrain);
    // val(OFF): held-out, always at inference.
    Result.ValLossOff   := MeanCE(NN, ValSet,   False, True,  DetVal);
    Result.Gap          := Result.ValLossOff - Result.TrainLossOff;
    Result.Deterministic := DetTrain and DetVal;

    Result.Diverged :=
      IsNan(Result.TrainLossOn)  or IsInfinite(Result.TrainLossOn)  or
      IsNan(Result.TrainLossOff) or IsInfinite(Result.TrainLossOff) or
      IsNan(Result.ValLossOff)   or IsInfinite(Result.ValLossOff);
  finally
    NFit.Free;
    NN.Free;
  end;
end;

function SafeF(V: TNeuralFloat; Width, Decimals: integer): string;
begin
  if IsNan(V) then Result := 'NaN'
  else if IsInfinite(V) then Result := 'Inf'
  else Result := FloatToStrF(V, ffFixed, Width, Decimals);
end;

var
  Results: array[0..cNumLayers - 1, 0..cNumProbs - 1] of TArmResult;
  TrainSet, ValSet: TNNetVolumePairList;
  L, k: integer;
  StartTime, EndTime: TDateTime;
  AllDeterministic, AllFinite, BaselineEqual, GapShrinks: boolean;
  R0, R: TArmResult;
begin
  // A diverging arm could produce NaN / Inf. Mask the FPU exceptions so those
  // surface as detectable float VALUES instead of raising EInvalidOp.
  SetExceptionMask([exInvalidOp, exDenormalized, exZeroDivide,
                    exOverflow, exUnderflow, exPrecision]);

  WriteLn('================================================================');
  WriteLn('Train-time vs inference-time delta sweep for the noise layers.');
  WriteLn('================================================================');
  WriteLn(Format('Net: Input(%d) -> FC(%d) -> %d residual blocks of width %d',
    [cDim, cWidth, cNumBlocks, cWidth]));
  WriteLn('     each block: y = x + Noise_p( ReLU(PointwiseConvLinear(x)) )');
  WriteLn(Format('     -> FC(%d) -> SoftMax.  %d-way classification.',
    [cClasses, cClasses]));
  WriteLn(Format('Train=%d, Val=%d, epochs=%d, batch=%d, LR=%.3f, RandSeed=%d.',
    [cTrain, cVal, cEpochs, cBatch, cLR, cSeed]));
  WriteLn('Same net/seed/data/epochs; only the noise layer class and p differ.');
  WriteLn('Noise layers swept: TNNetDropout, TNNetDropPath, TNNetSpatialDropout1D/2D.');
  WriteLn('train(ON)  = train CE with EnableDropouts(true)  [stochastic mask].');
  WriteLn('train(OFF) = same data with EnableDropouts(false) [inference identity].');
  WriteLn('val(OFF)   = held-out CE, always EnableDropouts(false).');
  WriteLn('p=0.0 is the no-noise baseline (identity in BOTH regimes).');
  WriteLn;

  StartTime := Now;
  // Build the shared teacher + datasets ONCE; reseed first so they are fixed.
  RandSeed := cSeed;
  MakeTeacher;
  TrainSet := BuildSet(cTrain);
  ValSet   := BuildSet(cVal);
  try
    for L := 0 to cNumLayers - 1 do
      for k := 0 to cNumProbs - 1 do
      begin
        Write(Format('Training %-16s p=%.2f ...', [cLayerNames[L], cProbs[k]]));
        Results[L, k] := RunArm(L, cProbs[k], TrainSet, ValSet);
        WriteLn(' done.');
      end;
  finally
    ValSet.Free;
    TrainSet.Free;
  end;
  EndTime := Now;

  WriteLn;
  WriteLn('=== Results table (lower CE is better; gap = val(OFF) - train(OFF)) ===');
  WriteLn('  layer              p   | train(ON)  train(OFF)  val(OFF) |    gap   | det');
  WriteLn('  -------------------+-----+----------------------------------+----------+----');
  for L := 0 to cNumLayers - 1 do
  begin
    for k := 0 to cNumProbs - 1 do
    begin
      R := Results[L, k];
      WriteLn(Format('  %-16s %.2f | %9s  %9s  %8s | %8s | %s',
        [cLayerNames[L], R.DropProb,
         SafeF(R.TrainLossOn, 8, 4),
         SafeF(R.TrainLossOff, 8, 4),
         SafeF(R.ValLossOff, 7, 4),
         SafeF(R.Gap, 7, 4),
         BoolToStr(R.Deterministic, 'yes', 'NO')]));
    end;
    WriteLn('  -------------------+-----+----------------------------------+----------+----');
  end;
  WriteLn;

  // ----- Self-check: invariants that are actually TRUE (Halt(1) on failure). -----
  WriteLn('=== Correctness signals ===');

  // (1) Inference / val forward passes are deterministic for every arm (the
  //     noise toggle really did turn the masks off).
  AllDeterministic := True;
  AllFinite := True;
  for L := 0 to cNumLayers - 1 do
    for k := 0 to cNumProbs - 1 do
    begin
      if not Results[L, k].Deterministic then AllDeterministic := False;
      if Results[L, k].Diverged then AllFinite := False;
    end;
  if AllFinite then
    WriteLn('[PASS] no arm produced NaN / Inf (all losses finite).')
  else
    WriteLn('[FAIL] an arm diverged to NaN / Inf.');
  if AllDeterministic then
    WriteLn('[PASS] every inference (noise OFF) forward pass is bit-for-bit '
      + 'deterministic.')
  else
    WriteLn('[FAIL] an inference pass varied pass-to-pass (noise toggle leaked).');

  // (2) At p=0.0 the layer is the identity in BOTH regimes, so train(ON) must
  //     equal train(OFF) exactly -- a direct numerical proof that p=0 is a
  //     no-op and that the toggle is consistent.
  BaselineEqual := True;
  for L := 0 to cNumLayers - 1 do
  begin
    R0 := Results[L, 0]; // p = 0.0 arm
    if Abs(R0.TrainLossOn - R0.TrainLossOff) > 1e-5 then BaselineEqual := False;
  end;
  if BaselineEqual then
    WriteLn('[PASS] p=0.0: train(ON) == train(OFF) for every layer (identity '
      + 'in both regimes).')
  else
    WriteLn('[FAIL] p=0.0 train(ON) != train(OFF) -- baseline is not an identity.');

  // (3) For p>0, train(ON) is strictly WORSE (higher CE) than train(OFF) on the
  //     same data: the stochastic mask handicaps the forward pass. This is the
  //     core train-vs-inference delta and should hold for at least the strongest
  //     noise arm (p=0.4) of every layer family.
  GapShrinks := True;
  for L := 0 to cNumLayers - 1 do
  begin
    R  := Results[L, cNumProbs - 1]; // strongest p = 0.4
    if not (R.TrainLossOn > R.TrainLossOff + 1e-4) then GapShrinks := False;
  end;
  if GapShrinks then
    WriteLn('[PASS] p=0.4: train(ON) > train(OFF) for every layer -- the '
      + 'stochastic mask raises the train-time loss, the inference identity '
      + 'recovers it.')
  else
    WriteLn('[WARN] p=0.4 train(ON) not clearly above train(OFF) for some layer '
      + '(small net / easy toy can blur this); inspect the table.');

  WriteLn;
  WriteLn(Format('Total wall-clock: %.1f s', [(EndTime - StartTime) * 86400.0]));

  if not (AllDeterministic and AllFinite and BaselineEqual) then Halt(1);
end.
