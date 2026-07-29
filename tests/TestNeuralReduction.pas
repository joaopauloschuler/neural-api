unit TestNeuralReduction;
(*
Tests for TNeuralFitBase.ReduceThreadDeltas, the cross-thread delta reduction
used by both training loops. The reduction is a binary tree over the worker
threads, so the property under test is that after every worker has run,
FThreadNN[0] holds the sum of all workers' deltas - for ANY thread count, not
just powers of two, which is where a tree's completion protocol usually breaks.

Each worker's deltas are pre-loaded with a distinct small integer, so the
expected total is exact in single precision and independent of the order the
tree happens to add them in.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuralfit, neuralthread;

type
  // Exposes the protected reduction machinery so a test can drive it directly
  // with real threads, without running a whole Fit.
  TReductionProbe = class(TNeuralFit)
  private
    FExpected: TNeuralFloat;
    procedure WorkerProc(index, threadnum: integer);
  public
    // Builds ThreadCount clones of Proto, fills worker I's deltas with I+1,
    // runs the reduction on ThreadCount real threads and returns the delta sum
    // left in worker 0.
    function ReduceAndGetTotal(ThreadCount: integer; Proto: TNNet): TNeuralFloat;
    property Expected: TNeuralFloat read FExpected;
  end;

  TTestNeuralReduction = class(TTestCase)
  private
    function BuildProto: TNNet;
    procedure CheckThreadCount(ThreadCount: integer);
  published
    // Powers of two: the shape the old two-level tree was written for.
    procedure TestReductionPowersOfTwo;
    // The cases a stride tree gets wrong: 3, 5, 6 and 7 workers.
    procedure TestReductionNonPowersOfTwo;
    // A single worker must reduce to itself without waiting on anything.
    procedure TestReductionSingleThread;
  end;

implementation

// Fills every neuron's Delta in Net with Value.
procedure FillDeltas(Net: TNNet; Value: TNeuralFloat);
var
  L, N: integer;
begin
  for L := 0 to Net.GetLastLayerIdx() do
    for N := 0 to Net.Layers[L].Neurons.Count - 1 do
      Net.Layers[L].Neurons[N].Delta.Fill(Value);
end;

procedure TReductionProbe.WorkerProc(index, threadnum: integer);
begin
  ReduceThreadDeltas(index, FThreadNN[index]);
end;

function TReductionProbe.ReduceAndGetTotal(ThreadCount: integer;
  Proto: TNNet): TNeuralFloat;
var
  TL: TNeuralThreadList;
  I: integer;
  UnitSum: TNeuralFloat;
begin
  FThreadNum := ThreadCount;
  FThreadNN := TNNetDataParallelism.Create(Proto, ThreadCount);
  SetLength(FFinishedThread, ThreadCount);
  ClearFinishedThread();
  FShouldQuit := false;

  // Worker I contributes I+1 in every delta slot, so the expected total is
  // (1 + 2 + ... + ThreadCount) times the per-net delta count. Dropping or
  // double-counting any single worker changes the total.
  FillDeltas(FThreadNN[0], 1);
  UnitSum := FThreadNN[0].GetDeltaSum();
  for I := 0 to ThreadCount - 1 do
    FillDeltas(FThreadNN[I], I + 1);
  FExpected := UnitSum * (ThreadCount * (ThreadCount + 1) div 2);

  TL := TNeuralThreadList.Create(ThreadCount);
  try
    TL.StartEngine();
    TL.StartProc({$IFDEF FPC}@{$ENDIF}WorkerProc, true);
    TL.StopEngine();
  finally
    TL.Free;
  end;

  Result := FThreadNN[0].GetDeltaSum();
  FThreadNN.Free;
  FThreadNN := nil;
  SetLength(FFinishedThread, 0);
end;

function TTestNeuralReduction.BuildProto: TNNet;
begin
  // Deliberately small: the delta total must stay well inside the range where
  // single-precision integer sums are exact.
  RandSeed := 424242;
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(4, 1, 1));
  Result.AddLayer(TNNetFullConnectReLU.Create(3));
  Result.AddLayer(TNNetFullConnectLinear.Create(2));
end;

procedure TTestNeuralReduction.CheckThreadCount(ThreadCount: integer);
var
  Proto: TNNet;
  Probe: TReductionProbe;
  Total: TNeuralFloat;
begin
  Proto := BuildProto();
  Probe := TReductionProbe.Create;
  try
    Total := Probe.ReduceAndGetTotal(ThreadCount, Proto);
    AssertTrue('Delta fixture must be non-trivial for ' +
      IntToStr(ThreadCount) + ' threads', Probe.Expected > 0);
    AssertEquals('Reduced delta total with ' + IntToStr(ThreadCount) +
      ' threads', Probe.Expected, Total, 0.0);
  finally
    Probe.Free;
    Proto.Free;
  end;
end;

procedure TTestNeuralReduction.TestReductionSingleThread;
begin
  CheckThreadCount(1);
end;

procedure TTestNeuralReduction.TestReductionPowersOfTwo;
begin
  CheckThreadCount(2);
  CheckThreadCount(4);
  CheckThreadCount(8);
end;

procedure TTestNeuralReduction.TestReductionNonPowersOfTwo;
begin
  CheckThreadCount(3);
  CheckThreadCount(5);
  CheckThreadCount(6);
  CheckThreadCount(7);
end;

initialization
  RegisterTest(TTestNeuralReduction);
end.
