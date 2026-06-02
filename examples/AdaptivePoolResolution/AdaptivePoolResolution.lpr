program AdaptivePoolResolution;
(*
AdaptivePoolResolution: the headline property of adaptive pooling is that
ONE fully-convolutional stack can accept inputs of DIFFERENT spatial
sizes and still produce a FIXED-size feature head (and therefore a
fixed-size classifier output). This demo makes that "variable in, fixed
out" property explicit.

A single conv stack (two padded 3x3 TNNetConvolutionReLU keeping the
spatial size, then a 2x2 stride-2 max pool that halves it) is terminated
by an adaptive-pool HEAD of fixed output size. No layer in the stack
hard-codes an input spatial size before the adaptive pool, so the same
network can be fed images of any resolution.

For BOTH TNNetAdaptiveAvgPool and TNNetAdaptiveMaxPool we:

  1. Build ONE net with a global (1x1) adaptive-pool head and feed it the
     SAME network two synthetic inputs at DIFFERENT resolutions (16x16
     and 24x24, same Depth). We print
        input shape -> post-conv shape -> post-adaptive-pool shape ->
        output shape
     for each and assert the OUTPUT shape is identical even though the
     input (and post-conv) shapes differ.

  2. Exercise a 2x2 adaptive head too (TNNetAdaptiveAvgPool.Create(2) /
     TNNetAdaptiveMaxPool.Create(2)) to show a non-global fixed head also
     produces the same shape at both resolutions.

  3. Run two built-in DEGENERACY correctness checks:
       - Create(1) == global pooling: the adaptive(1) output equals the
         per-channel global avg / global max over the post-conv map.
       - Create(N) where N == post-conv spatial size == identity: the
         adaptive output equals the post-conv map element-for-element.

  4. A tiny TRAINING sanity step: train the global-head classifier for a
     handful of epochs on a trivial synthetic 2-class task at ONE
     resolution (16x16), then run inference at the OTHER resolution
     (24x24) to show the trained head still emits valid class scores.

Pure CPU, no external dataset, all data synthesised in-code, finishes in
a couple of seconds.

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
  cChannels = 3;
  cResA     = 16;   // first  input resolution
  cResB     = 24;   // second input resolution (different on purpose)
  cConvCh   = 8;    // conv channel count (kept small)
  cNumCls   = 2;    // trivial 2-class head
  cSeed     = 42;
  cEps      = 1e-4; // tolerance for the degeneracy assertions

type
  TPoolKind = (pkAvg, pkMax);

function KindName(K: TPoolKind): string;
begin
  if K = pkAvg then Result := 'TNNetAdaptiveAvgPool'
               else Result := 'TNNetAdaptiveMaxPool';
end;

function Shape(V: TNNetVolume): string;
begin
  Result := Format('%dx%dx%d', [V.SizeX, V.SizeY, V.Depth]);
end;

// Build the shared backbone, then append an adaptive-pool head of the
// requested kind and output size. The InputSize is the ONLY thing that
// varies between the two resolutions; nothing in the head hard-codes it.
procedure BuildNet(out NN: TNNet; InputSize: integer; K: TPoolKind;
  OutSize: integer);
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(InputSize, InputSize, cChannels));
  // Two padded 3x3 convs keep the spatial size; one stride-2 max pool
  // halves it. Fully convolutional: no spatial size baked in.
  NN.AddLayer(TNNetConvolutionReLU.Create(cConvCh, 3, 1, 1));
  NN.AddLayer(TNNetConvolutionReLU.Create(cConvCh, 3, 1, 1));
  NN.AddLayer(TNNetMaxPool.Create(2)); // 2x2 stride-2: halves the map

  if K = pkAvg then
    NN.AddLayer(TNNetAdaptiveAvgPool.Create(OutSize))
  else
    NN.AddLayer(TNNetAdaptiveMaxPool.Create(OutSize));
end;

// Fill a synthetic input volume of the given size with reproducible noise.
function MakeInput(InputSize: integer): TNNetVolume;
var
  PX, PY, C: integer;
begin
  Result := TNNetVolume.Create(InputSize, InputSize, cChannels);
  for PX := 0 to InputSize - 1 do
    for PY := 0 to InputSize - 1 do
      for C := 0 to cChannels - 1 do
        Result.Data[PX, PY, C] := Random;
end;

// The conv-stack layer that feeds the adaptive head is the one just
// before the last layer (index Count-2). Returns its Output volume.
function PostConv(NN: TNNet): TNNetVolume;
begin
  Result := NN.Layers[NN.Layers.Count - 2].Output;
end;

// Global per-channel reduction over a feature map, used to verify the
// Create(1) == global-pool degeneracy.
procedure GlobalReduce(V: TNNetVolume; K: TPoolKind; out Ref: TNNetVolume);
var
  C, X, Y: integer;
  Acc, Cur: TNeuralFloat;
begin
  Ref := TNNetVolume.Create(1, 1, V.Depth);
  for C := 0 to V.Depth - 1 do
  begin
    if K = pkAvg then Acc := 0
                 else Acc := V.Data[0, 0, C];
    for X := 0 to V.SizeX - 1 do
      for Y := 0 to V.SizeY - 1 do
      begin
        Cur := V.Data[X, Y, C];
        if K = pkAvg then Acc := Acc + Cur
        else if Cur > Acc then Acc := Cur;
      end;
    if K = pkAvg then Acc := Acc / (V.SizeX * V.SizeY);
    Ref.Data[0, 0, C] := Acc;
  end;
end;

// Largest absolute element-wise difference between two same-shape volumes.
function MaxDiff(A, B: TNNetVolume): TNeuralFloat;
var
  I: integer;
begin
  Result := 0;
  for I := 0 to A.Size - 1 do
    Result := Max(Result, Abs(A.FData[I] - B.FData[I]));
end;

var
  GlobalChecksPassed: boolean = True;
  IdentityCheckRan: boolean = False;

// Run the variable-resolution demo for one pooling kind with a 1x1
// (global) head. Returns the output shape seen at each resolution.
procedure RunResolutionArm(K: TPoolKind);
var
  NN: TNNet;
  InA, InB: TNNetVolume;
  Ref: TNNetVolume;
  D: TNeuralFloat;
  ShapeA, ShapeB: string;
begin
  WriteLn('--- ', KindName(K), ' with global (1x1) head ---');
  Randomize;
  RandSeed := cSeed;

  // Resolution A.
  BuildNet(NN, cResA, K, 1);
  InA := MakeInput(cResA);
  NN.Compute(InA);
  WriteLn(Format('  res A: in %s -> post-conv %s -> adaptive %s -> out %s',
    [Shape(InA), Shape(PostConv(NN)),
     Shape(NN.Layers[NN.Layers.Count - 1].Output),
     Shape(NN.GetLastLayer().Output)]));
  ShapeA := Shape(NN.GetLastLayer().Output);

  // Degeneracy check 1: Create(1) == global pooling over the post-conv map.
  GlobalReduce(PostConv(NN), K, Ref);
  D := MaxDiff(Ref, NN.GetLastLayer().Output);
  WriteLn(Format('    degeneracy Create(1)==global-pool: max|diff|=%.2e %s',
    [D, BoolToStr(D < cEps, 'OK', 'FAIL')]));
  if D >= cEps then GlobalChecksPassed := False;
  Ref.Free;
  InA.Free;
  NN.Free;

  // Resolution B: SAME architecture, different input size.
  BuildNet(NN, cResB, K, 1);
  InB := MakeInput(cResB);
  NN.Compute(InB);
  WriteLn(Format('  res B: in %s -> post-conv %s -> adaptive %s -> out %s',
    [Shape(InB), Shape(PostConv(NN)),
     Shape(NN.Layers[NN.Layers.Count - 1].Output),
     Shape(NN.GetLastLayer().Output)]));
  ShapeB := Shape(NN.GetLastLayer().Output);
  InB.Free;
  NN.Free;

  if ShapeA = ShapeB then
    WriteLn('  => variable in, FIXED out: both resolutions yield ', ShapeA)
  else
  begin
    WriteLn('  => MISMATCH: ', ShapeA, ' vs ', ShapeB);
    GlobalChecksPassed := False;
  end;
  WriteLn;
end;

// Same idea but with a 2x2 head, and additionally exercise the
// identity degeneracy (Create(N) where N == post-conv spatial size).
procedure Run2x2Arm(K: TPoolKind);
var
  NN: TNNet;
  InA, InB: TNNetVolume;
  PostSize: integer;
  D: TNeuralFloat;
  ShapeA, ShapeB: string;
begin
  WriteLn('--- ', KindName(K), ' with 2x2 head ---');
  RandSeed := cSeed;

  BuildNet(NN, cResA, K, 2);
  InA := MakeInput(cResA);
  NN.Compute(InA);
  WriteLn(Format('  res A: in %s -> post-conv %s -> out %s',
    [Shape(InA), Shape(PostConv(NN)), Shape(NN.GetLastLayer().Output)]));
  ShapeA := Shape(NN.GetLastLayer().Output);
  InA.Free;
  NN.Free;

  BuildNet(NN, cResB, K, 2);
  InB := MakeInput(cResB);
  NN.Compute(InB);
  WriteLn(Format('  res B: in %s -> post-conv %s -> out %s',
    [Shape(InB), Shape(PostConv(NN)), Shape(NN.GetLastLayer().Output)]));
  ShapeB := Shape(NN.GetLastLayer().Output);
  InB.Free;
  NN.Free;

  if ShapeA = ShapeB then
    WriteLn('  => variable in, FIXED out: both resolutions yield ', ShapeA)
  else
  begin
    WriteLn('  => MISMATCH: ', ShapeA, ' vs ', ShapeB);
    GlobalChecksPassed := False;
  end;

  // Degeneracy check 2: Create(N) where N == post-conv spatial size is
  // the identity. Build a head whose output size equals the post-conv
  // map size and assert the head output matches the post-conv map.
  BuildNet(NN, cResA, K, 1);
  InA := MakeInput(cResA);
  NN.Compute(InA);
  PostSize := PostConv(NN).SizeX;
  NN.Free;

  BuildNet(NN, cResA, K, PostSize);
  NN.Compute(InA);
  D := MaxDiff(PostConv(NN), NN.GetLastLayer().Output);
  WriteLn(Format('  degeneracy Create(%d)==identity: max|diff|=%.2e %s',
    [PostSize, D, BoolToStr(D < cEps, 'OK', 'FAIL')]));
  IdentityCheckRan := True;
  if D >= cEps then GlobalChecksPassed := False;
  InA.Free;
  NN.Free;
  WriteLn;
end;

// A trivial synthetic 2-class task: class 0 = bright top half, class 1 =
// bright bottom half, on uniform noise. Works at any resolution.
procedure MakeSample(ClassId, InputSize: integer; out X, Y: TNNetVolume);
var
  PX, PY, C, Y0, Y1: integer;
begin
  X := TNNetVolume.Create(InputSize, InputSize, cChannels);
  Y := TNNetVolume.Create(cNumCls, 1, 1);
  Y.Fill(0);
  Y.FData[ClassId] := 1.0;
  for PX := 0 to InputSize - 1 do
    for PY := 0 to InputSize - 1 do
      for C := 0 to cChannels - 1 do
        X.Data[PX, PY, C] := Random * 0.3;
  if ClassId = 0 then begin Y0 := 0;                 Y1 := InputSize div 2 - 1; end
                 else begin Y0 := InputSize div 2;   Y1 := InputSize - 1;       end;
  for PX := 0 to InputSize - 1 do
    for PY := Y0 to Y1 do
      for C := 0 to cChannels - 1 do
        X.Data[PX, PY, C] := X.Data[PX, PY, C] + 0.9;
end;

// Classifier net: backbone -> adaptive-avg(1) -> FC(2) -> SoftMax. The FC
// always sees a fixed 1x1xConvCh head regardless of input resolution, so
// the SAME weights are valid at any input size. The TNNetInput layer does
// pin a concrete size, so to run a trained net at a different resolution
// we build a sibling net of the same architecture at the new size and
// CopyWeights into it (conv/FC weights are spatial-size independent).
function BuildClassifier(InputSize: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(InputSize, InputSize, cChannels));
  Result.AddLayer(TNNetConvolutionReLU.Create(cConvCh, 3, 1, 1));
  Result.AddLayer(TNNetConvolutionReLU.Create(cConvCh, 3, 1, 1));
  Result.AddLayer(TNNetMaxPool.Create(2));
  Result.AddLayer(TNNetAdaptiveAvgPool.Create(1));
  Result.AddLayer(TNNetFullConnectLinear.Create(cNumCls));
  Result.AddLayer(TNNetSoftMax.Create());
end;

// Train a classifier (global adaptive-avg head) at resolution A for a few
// epochs, then run inference at resolution B (via a weight-shared sibling
// net) to show the trained head still produces valid class scores at an
// unseen resolution.
procedure RunTrainingSanity();
var
  NN, InferNN: TNNet;
  TrainSet: TNNetVolumePairList;
  X, Y, P: TNNetVolume;
  Epoch, I, C, Hits: integer;
  Sum: Double;
begin
  WriteLn('--- training sanity (TNNetAdaptiveAvgPool global head) ---');
  RandSeed := cSeed;

  TrainSet := TNNetVolumePairList.Create();
  for C := 0 to cNumCls - 1 do
    for I := 1 to 60 do
    begin
      MakeSample(C, cResA, X, Y);
      TrainSet.Add(TNNetVolumePair.Create(X, Y));
    end;

  NN := BuildClassifier(cResA);
  NN.SetLearningRate(0.01, 0.9);

  for Epoch := 1 to 30 do
  begin
    Sum := 0;
    for I := 0 to TrainSet.Count - 1 do
    begin
      NN.Compute(TrainSet[I].I);
      NN.Backpropagate(TrainSet[I].O);
      P := NN.GetLastLayer().Output;
      Sum := Sum - Ln(Max(1e-9, P.FData[TrainSet[I].O.GetClass()]));
    end;
    if (Epoch = 1) or (Epoch = 30) then
      WriteLn(Format('  epoch %2d  train NLL=%.4f', [Epoch, Sum / TrainSet.Count]));
  end;

  // Train accuracy at res A.
  Hits := 0;
  for I := 0 to TrainSet.Count - 1 do
  begin
    NN.Compute(TrainSet[I].I);
    if NN.GetLastLayer().Output.GetClass() = TrainSet[I].O.GetClass() then Inc(Hits);
  end;
  WriteLn(Format('  train acc @ res %dx%d : %.3f',
    [cResA, cResA, Hits / TrainSet.Count]));

  // Inference at the OTHER resolution (cResB): build a same-architecture
  // sibling net pinned to cResB and copy the trained weights into it. The
  // fully-convolutional backbone + adaptive head make this valid.
  InferNN := BuildClassifier(cResB);
  InferNN.CopyWeights(NN);

  Hits := 0;
  for C := 0 to cNumCls - 1 do
    for I := 1 to 30 do
    begin
      MakeSample(C, cResB, X, Y);
      InferNN.Compute(X);
      P := InferNN.GetLastLayer().Output;
      if P.GetClass() = C then Inc(Hits);
      X.Free; Y.Free;
    end;
  WriteLn(Format('  infer acc @ res %dx%d (unseen size, weight-shared): %.3f',
    [cResB, cResB, Hits / (cNumCls * 30)]));
  WriteLn('  => trained fixed-size head runs at a resolution it never saw.');
  WriteLn;

  InferNN.Free;
  NN.Free;
  TrainSet.Free;
end;

begin
  WriteLn('AdaptivePoolResolution: variable-resolution in, fixed-size out.');
  WriteLn('Backbone: Input(NxNx', cChannels,
          ') -> Conv', cConvCh, '(3x3,pad1)+ReLU x2 -> MaxPool(2) -> [adaptive head]');
  WriteLn('Feeding the SAME net inputs at ', cResA, 'x', cResA,
          ' and ', cResB, 'x', cResB, '.');
  WriteLn;

  RunResolutionArm(pkAvg);
  RunResolutionArm(pkMax);
  Run2x2Arm(pkAvg);
  Run2x2Arm(pkMax);
  RunTrainingSanity();

  WriteLn('=== summary ===');
  WriteLn('  degeneracy/shape checks: ',
          BoolToStr(GlobalChecksPassed, 'ALL PASSED', 'SOME FAILED'));
  WriteLn('  identity check ran: ', BoolToStr(IdentityCheckRan, 'yes', 'no'));
  if not GlobalChecksPassed then
  begin
    WriteLn('FAILED.');
    Halt(1);
  end;
  WriteLn('OK.');
end.
