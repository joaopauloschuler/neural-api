unit TestNeuralThread;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralthread;

type
  TTestNeuralThread = class(TTestCase)
  private
    FCounter: integer;
    procedure IncrementCounter(index, threadnum: integer);
  published
    procedure TestThreadListCreation;
    procedure TestCalculateWorkingRange;
    procedure TestParallelExecution;
    procedure TestDefaultThreadCount;
  end;

implementation

procedure TTestNeuralThread.IncrementCounter(index, threadnum: integer);
begin
  InterlockedIncrement(FCounter);
end;

procedure TTestNeuralThread.TestThreadListCreation;
var
  ThreadList: TNeuralThreadList;
begin
  ThreadList := TNeuralThreadList.Create(4);
  try
    AssertEquals('Thread list should have 4 threads', 4, ThreadList.Count);
  finally
    ThreadList.Free;
  end;
end;

procedure TTestNeuralThread.TestCalculateWorkingRange;
var
  StartPos, FinishPos: integer;
begin
  // Test dividing 100 items among 4 threads
  
  // Thread 0 of 4 with 100 items
  TNeuralThreadList.CalculateWorkingRange(0, 4, 100, StartPos, FinishPos);
  AssertEquals('Thread 0 start should be 0', 0, StartPos);
  AssertEquals('Thread 0 finish should be 24', 24, FinishPos);
  
  // Thread 1 of 4 with 100 items
  TNeuralThreadList.CalculateWorkingRange(1, 4, 100, StartPos, FinishPos);
  AssertEquals('Thread 1 start should be 25', 25, StartPos);
  AssertEquals('Thread 1 finish should be 49', 49, FinishPos);
  
  // Thread 2 of 4 with 100 items
  TNeuralThreadList.CalculateWorkingRange(2, 4, 100, StartPos, FinishPos);
  AssertEquals('Thread 2 start should be 50', 50, StartPos);
  AssertEquals('Thread 2 finish should be 74', 74, FinishPos);
  
  // Thread 3 of 4 with 100 items
  TNeuralThreadList.CalculateWorkingRange(3, 4, 100, StartPos, FinishPos);
  AssertEquals('Thread 3 start should be 75', 75, StartPos);
  AssertEquals('Thread 3 finish should be 99', 99, FinishPos);
end;

procedure TTestNeuralThread.TestParallelExecution;
var
  ThreadList: TNeuralThreadList;
begin
  FCounter := 0;
  ThreadList := TNeuralThreadList.Create(4);
  try
    ThreadList.StartProc(@IncrementCounter, true);
    // Each of 4 threads should increment counter once
    AssertEquals('Counter should be 4 after parallel execution', 4, FCounter);
  finally
    ThreadList.Free;
  end;
end;

procedure TTestNeuralThread.TestDefaultThreadCount;
var
  DefaultCount: integer;
begin
  DefaultCount := NeuralDefaultThreadCount();
  AssertTrue('Default thread count should be at least 1', DefaultCount >= 1);
end;

initialization
  RegisterTest(TTestNeuralThread);

end.
