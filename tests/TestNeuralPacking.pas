unit TestNeuralPacking;
(*
Tests for TNNetSequencePacker (neuraldatasets.pas): sequence packing for
autoregressive LM pretraining. Layout expectations are hand-computed on tiny
corpora; the loss-mask integration tests verify on a small fixed network that
masked (pad-target) positions carry exactly zero output error / zero weight
deltas with the framework's e = Output - Desired error convention.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuraldatasets;

type
  TTestNeuralPacking = class(TTestCase)
  private
    // Tiny per-position causal LM: Input(Ctx,1,1) token ids -> Embedding ->
    // PointwiseConvLinear(Vocab) -> PointwiseSoftMax (one distribution per
    // input position; output row P predicts the token at position P+1).
    function BuildPerPositionLM(ContextLen, Vocab: integer): TNNet;
    function SumAbsDeltas(NN: TNNet): TNeuralFloat;
  published
    // docs [2,3,4],[5,6] with ctx 4 -> exact windows [2,3,4,1],[5,6,1,0]
    // (every doc followed by one separator=1; final partial window padded=0).
    procedure TestSplitModeExactLayout;
    // Packing the same corpus twice produces identical windows.
    procedure TestSplitModeDeterministic;
    // Every real token appears exactly once across all windows; separator
    // count = document count; everything else is padding.
    procedure TestAllRealTokensOnceAndSeparatorCount;
    // IsTargetPredictable is exactly "next token within the window exists
    // and is not the pad token".
    procedure TestMaskMatchesPadTargets;
    // pmNoSplitGreedy: a document (+ separator) never crosses a window
    // boundary; hand-computed bin layout.
    procedure TestNoSplitDocsNeverCrossBoundary;
    // pmOneDocPerWindow: one window per document; long docs truncated to
    // ContextLen-1 tokens + separator.
    procedure TestOneDocPerWindowCountAndTruncation;
    // Utilization = predictable target slots / (WindowCount * (Ctx-1)),
    // hand-computed on the layout from TestSplitModeExactLayout.
    procedure TestUtilizationHandComputed;
    // GetTrainingPair: ids encoding (Depth=1) and one-hot encoding
    // (Depth=Vocab); per-position one-hot targets, all-zero masked rows.
    procedure TestGetTrainingPairEncodings;
    // Token ids < 2 are reserved and rejected.
    procedure TestRejectsSpecialTokenIds;
    // After Compute + ApplyLossMask + Backpropagate, the last layer's
    // OutputError is exactly zero at every masked position and non-zero at
    // some predictable position.
    procedure TestLossMaskZeroErrorAtMaskedPositions;
    // A window whose every position is masked (final window [sep,0,0,0])
    // produces exactly zero weight deltas.
    procedure TestLossMaskAllMaskedZeroDeltas;
  end;

implementation

function TTestNeuralPacking.BuildPerPositionLM(ContextLen, Vocab: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(ContextLen, 1, 1),
    TNNetEmbedding.Create(Vocab, 8),
    TNNetPointwiseConvLinear.Create(Vocab),
    TNNetPointwiseSoftMax.Create(1)
  ]);
end;

function TTestNeuralPacking.SumAbsDeltas(NN: TNNet): TNeuralFloat;
var
  L, N: integer;
begin
  Result := 0;
  for L := 0 to NN.CountLayers() - 1 do
    for N := 0 to NN.Layers[L].Neurons.Count - 1 do
      Result := Result + NN.Layers[L].Neurons[N].Delta.GetSumAbs() +
        Abs(NN.Layers[L].Neurons[N].BiasDelta);
end;

procedure TTestNeuralPacking.TestSplitModeExactLayout;
const
  ExpectedW0: array[0..3] of integer = (2, 3, 4, 1);
  ExpectedW1: array[0..3] of integer = (5, 6, 1, 0);
var
  Packer: TNNetSequencePacker;
  P: integer;
begin
  Packer := TNNetSequencePacker.Create(4);
  try
    Packer.AddDocument([2, 3, 4]);
    Packer.AddDocument([5, 6]);
    Packer.Pack();
    AssertEquals('window count', 2, Packer.WindowCount());
    for P := 0 to 3 do
    begin
      AssertEquals('w0 pos ' + IntToStr(P), ExpectedW0[P], Packer.GetToken(0, P));
      AssertEquals('w1 pos ' + IntToStr(P), ExpectedW1[P], Packer.GetToken(1, P));
    end;
    // Window 0 mask: targets 3,4,1 all real -> predictable; last pos never.
    AssertTrue('w0 p0', Packer.IsTargetPredictable(0, 0));
    AssertTrue('w0 p1', Packer.IsTargetPredictable(0, 1));
    AssertTrue('w0 p2 (separator IS a target)', Packer.IsTargetPredictable(0, 2));
    AssertFalse('w0 p3 (last pos)', Packer.IsTargetPredictable(0, 3));
    // Window 1 mask: targets 6,1 real; target at p2 is pad -> masked.
    AssertTrue('w1 p0', Packer.IsTargetPredictable(1, 0));
    AssertTrue('w1 p1', Packer.IsTargetPredictable(1, 1));
    AssertFalse('w1 p2 (pad target)', Packer.IsTargetPredictable(1, 2));
    AssertFalse('w1 p3 (last pos)', Packer.IsTargetPredictable(1, 3));
  finally
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestSplitModeDeterministic;
var
  PackerA, PackerB: TNNetSequencePacker;
  W, P: integer;
begin
  PackerA := TNNetSequencePacker.Create(5);
  PackerB := TNNetSequencePacker.Create(5);
  try
    PackerA.AddDocument([7, 8, 9, 10]);
    PackerA.AddDocument([11, 12]);
    PackerA.AddDocument([13, 14, 15]);
    PackerB.AddDocument([7, 8, 9, 10]);
    PackerB.AddDocument([11, 12]);
    PackerB.AddDocument([13, 14, 15]);
    PackerA.Pack();
    PackerB.Pack();
    // Re-pack A a second time: must be idempotent.
    PackerA.Pack();
    AssertEquals('window counts', PackerA.WindowCount(), PackerB.WindowCount());
    for W := 0 to PackerA.WindowCount() - 1 do
      for P := 0 to 4 do
        AssertEquals('w' + IntToStr(W) + ' p' + IntToStr(P),
          PackerA.GetToken(W, P), PackerB.GetToken(W, P));
  finally
    PackerB.Free;
    PackerA.Free;
  end;
end;

procedure TTestNeuralPacking.TestAllRealTokensOnceAndSeparatorCount;
var
  Packer: TNNetSequencePacker;
  Counts: array[0..63] of integer;
  W, P, T, RealTokens, DocCount: integer;
begin
  // Distinct ids 2..21 over 4 docs -> each must appear exactly once.
  Packer := TNNetSequencePacker.Create(6);
  try
    Packer.AddDocument([2, 3, 4, 5, 6]);
    Packer.AddDocument([7, 8]);
    Packer.AddDocument([9, 10, 11, 12, 13, 14, 15]);
    Packer.AddDocument([16, 17, 18, 19, 20, 21]);
    DocCount := 4;
    RealTokens := 20;
    Packer.Pack();
    for T := 0 to High(Counts) do Counts[T] := 0;
    for W := 0 to Packer.WindowCount() - 1 do
      for P := 0 to 5 do
        Inc(Counts[Packer.GetToken(W, P)]);
    for T := 2 to RealTokens + 1 do
      AssertEquals('token ' + IntToStr(T) + ' appears once', 1, Counts[T]);
    AssertEquals('separator count = doc count', DocCount, Counts[1]);
    AssertEquals('rest is padding',
      Packer.WindowCount() * 6 - RealTokens - DocCount, Counts[0]);
  finally
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestMaskMatchesPadTargets;
var
  Packer: TNNetSequencePacker;
  W, P: integer;
  Expected: boolean;
begin
  Packer := TNNetSequencePacker.Create(4);
  try
    Packer.AddDocument([2, 3, 4, 5, 6]);
    Packer.AddDocument([7, 8, 9]);
    Packer.Pack();
    for W := 0 to Packer.WindowCount() - 1 do
      for P := 0 to 3 do
      begin
        Expected := (P < 3) and (Packer.GetToken(W, P + 1) <> 0);
        AssertEquals('w' + IntToStr(W) + ' p' + IntToStr(P),
          Expected, Packer.IsTargetPredictable(W, P));
      end;
  finally
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestNoSplitDocsNeverCrossBoundary;
const
  // ctx 6; docs [2,3,4],[5,6,7],[8,9]: doc+sep = 4 tokens each (3 for the
  // last); greedy fill: each new doc needs 4 (3) slots, only 2 remain after
  // the previous doc, so every doc opens a fresh window.
  Expected: array[0..2, 0..5] of integer = (
    (2, 3, 4, 1, 0, 0),
    (5, 6, 7, 1, 0, 0),
    (8, 9, 1, 0, 0, 0));
var
  Packer: TNNetSequencePacker;
  W, P: integer;
begin
  Packer := TNNetSequencePacker.Create(6, pmNoSplitGreedy);
  try
    Packer.AddDocument([2, 3, 4]);
    Packer.AddDocument([5, 6, 7]);
    Packer.AddDocument([8, 9]);
    Packer.Pack();
    AssertEquals('window count', 3, Packer.WindowCount());
    for W := 0 to 2 do
      for P := 0 to 5 do
        AssertEquals('w' + IntToStr(W) + ' p' + IntToStr(P),
          Expected[W, P], Packer.GetToken(W, P));
  finally
    Packer.Free;
  end;
  // Two short docs that DO fit together share one window: [2,3,1,4,5,1].
  Packer := TNNetSequencePacker.Create(6, pmNoSplitGreedy);
  try
    Packer.AddDocument([2, 3]);
    Packer.AddDocument([4, 5]);
    Packer.Pack();
    AssertEquals('shared window count', 1, Packer.WindowCount());
    AssertEquals('p2 is separator', 1, Packer.GetToken(0, 2));
    AssertEquals('p3 starts doc 2', 4, Packer.GetToken(0, 3));
    AssertEquals('p5 is separator', 1, Packer.GetToken(0, 5));
  finally
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestOneDocPerWindowCountAndTruncation;
var
  Packer: TNNetSequencePacker;
  P: integer;
begin
  Packer := TNNetSequencePacker.Create(4, pmOneDocPerWindow);
  try
    Packer.AddDocument([2, 3]);
    Packer.AddDocument([4, 5]); // would fit with doc 1, but mode forbids it
    Packer.AddDocument([6, 7, 8, 9, 10, 11]); // truncated to 3 tokens + sep
    Packer.Pack();
    AssertEquals('one window per doc', 3, Packer.WindowCount());
    AssertEquals('w0 p2 separator', 1, Packer.GetToken(0, 2));
    AssertEquals('w0 p3 pad', 0, Packer.GetToken(0, 3));
    AssertEquals('w1 p0', 4, Packer.GetToken(1, 0));
    // Truncated long doc: [6,7,8,sep] - exactly fills the window.
    for P := 0 to 2 do
      AssertEquals('w2 p' + IntToStr(P), 6 + P, Packer.GetToken(2, P));
    AssertEquals('w2 p3 separator', 1, Packer.GetToken(2, 3));
  finally
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestUtilizationHandComputed;
var
  Packer: TNNetSequencePacker;
begin
  // Layout from TestSplitModeExactLayout: predictable slots 3 + 2 = 5 of
  // 2 windows * (4-1) = 6 -> 5/6.
  Packer := TNNetSequencePacker.Create(4);
  try
    Packer.AddDocument([2, 3, 4]);
    Packer.AddDocument([5, 6]);
    Packer.Pack();
    AssertEquals('predictable w0', 3, Packer.PredictableTargetCount(0));
    AssertEquals('predictable w1', 2, Packer.PredictableTargetCount(1));
    AssertEquals('utilization', 5 / 6, Packer.Utilization(), 1e-6);
  finally
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestGetTrainingPairEncodings;
var
  Packer: TNNetSequencePacker;
  IdsIn, OneHotIn, Target: TNNetVolume;
  P, D: integer;
begin
  Packer := TNNetSequencePacker.Create(4);
  IdsIn := TNNetVolume.Create(4, 1, 1);
  OneHotIn := TNNetVolume.Create(4, 1, 8);
  Target := TNNetVolume.Create(4, 1, 8);
  try
    Packer.AddDocument([2, 3, 4]);
    Packer.AddDocument([5, 6]);
    Packer.Pack();
    // Window 1 = [5,6,1,0]: ids encoding.
    Packer.GetTrainingPair(1, IdsIn, Target);
    AssertEquals('ids p0', 5, IdsIn[0, 0, 0], 1e-6);
    AssertEquals('ids p1', 6, IdsIn[1, 0, 0], 1e-6);
    AssertEquals('ids p2', 1, IdsIn[2, 0, 0], 1e-6);
    AssertEquals('ids p3', 0, IdsIn[3, 0, 0], 1e-6);
    // Targets: p0 -> one-hot(6), p1 -> one-hot(1), p2/p3 masked all-zero.
    AssertEquals('t p0 hot', 1, Target[0, 0, 6], 1e-6);
    AssertEquals('t p1 hot', 1, Target[1, 0, 1], 1e-6);
    for D := 0 to 7 do
    begin
      AssertEquals('t p2 zero d' + IntToStr(D), 0, Target[2, 0, D], 1e-6);
      AssertEquals('t p3 zero d' + IntToStr(D), 0, Target[3, 0, D], 1e-6);
    end;
    // One-hot input encoding for window 0 = [2,3,4,1].
    Packer.GetTrainingPair(0, OneHotIn, Target);
    for P := 0 to 3 do
      for D := 0 to 7 do
        if ((P < 3) and (D = P + 2)) or ((P = 3) and (D = 1))
        then AssertEquals('in hot p' + IntToStr(P), 1, OneHotIn[P, 0, D], 1e-6)
        else AssertEquals('in cold p' + IntToStr(P) + ' d' + IntToStr(D),
          0, OneHotIn[P, 0, D], 1e-6);
  finally
    Target.Free;
    OneHotIn.Free;
    IdsIn.Free;
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestRejectsSpecialTokenIds;
var
  Packer: TNNetSequencePacker;
  Raised: boolean;
begin
  Packer := TNNetSequencePacker.Create(4);
  try
    Raised := false;
    try
      Packer.AddDocument([2, 1, 3]);
    except
      on E: Exception do Raised := true;
    end;
    AssertTrue('token id 1 rejected', Raised);
    Raised := false;
    try
      Packer.AddDocument([0]);
    except
      on E: Exception do Raised := true;
    end;
    AssertTrue('token id 0 rejected', Raised);
  finally
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestLossMaskZeroErrorAtMaskedPositions;
var
  Packer: TNNetSequencePacker;
  NN: TNNet;
  InputV, TargetV: TNNetVolume;
  ErrV: TNNetVolume;
  P, D: integer;
  RowAbs, UnmaskedAbs: TNeuralFloat;
begin
  RandSeed := 424242;
  Packer := TNNetSequencePacker.Create(4);
  NN := BuildPerPositionLM(4, 8);
  InputV := TNNetVolume.Create(4, 1, 1);
  TargetV := TNNetVolume.Create(4, 1, 8);
  try
    Packer.AddDocument([2, 3, 4]);
    Packer.AddDocument([5, 6]);
    Packer.Pack();
    NN.SetBatchUpdate(true);
    NN.ClearDeltas();
    // Window 1 = [5,6,1,0]: positions 2 and 3 are masked.
    Packer.GetTrainingPair(1, InputV, TargetV);
    NN.Compute(InputV);
    Packer.ApplyLossMask(1, TargetV, NN.GetLastLayer().Output);
    NN.Backpropagate(TargetV);
    ErrV := NN.GetLastLayer().OutputError;
    UnmaskedAbs := 0;
    for P := 0 to 3 do
    begin
      RowAbs := 0;
      for D := 0 to 7 do RowAbs := RowAbs + Abs(ErrV[P, 0, D]);
      if Packer.IsTargetPredictable(1, P)
      then UnmaskedAbs := UnmaskedAbs + RowAbs
      else AssertEquals('masked row ' + IntToStr(P) + ' error', 0, RowAbs, 0);
    end;
    AssertTrue('some unmasked error flows', UnmaskedAbs > 1e-6);
  finally
    TargetV.Free;
    InputV.Free;
    NN.Free;
    Packer.Free;
  end;
end;

procedure TTestNeuralPacking.TestLossMaskAllMaskedZeroDeltas;
var
  Packer: TNNetSequencePacker;
  NN: TNNet;
  InputV, TargetV: TNNetVolume;
begin
  RandSeed := 424242;
  // docs [2,3,4,5] with ctx 4 -> stream [2,3,4,5,1]; window 1 = [1,0,0,0]
  // whose every position is masked (targets pad or last position).
  Packer := TNNetSequencePacker.Create(4);
  NN := BuildPerPositionLM(4, 8);
  InputV := TNNetVolume.Create(4, 1, 1);
  TargetV := TNNetVolume.Create(4, 1, 8);
  try
    Packer.AddDocument([2, 3, 4, 5]);
    Packer.Pack();
    AssertEquals('two windows', 2, Packer.WindowCount());
    AssertEquals('all-masked window', 0, Packer.PredictableTargetCount(1));
    NN.SetBatchUpdate(true);
    NN.ClearDeltas();
    Packer.GetTrainingPair(1, InputV, TargetV);
    NN.Compute(InputV);
    Packer.ApplyLossMask(1, TargetV, NN.GetLastLayer().Output);
    NN.Backpropagate(TargetV);
    AssertEquals('zero gradient everywhere', 0, SumAbsDeltas(NN), 0);
  finally
    TargetV.Free;
    InputV.Free;
    NN.Free;
    Packer.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralPacking);
end.
