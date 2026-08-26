unit TestNeuralABFun;
(*
Tests for neuralabfun.pas operation semantics: TRunOperation.OperateAndTestOperation
must produce the byte documented next to each opcode constant. The state
operations are exercised over a fixed Actions/CurrentStates/NextStates triple so
every result is pinned by hand.

csNot has its own test because it is the one arm whose integer form is negative
(not 245 = -246) and therefore the one that a range check on the byte assignment
would reject; the operation is defined as a byte complement and must return
255 - x.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralabfun;

type
  TTestNeuralABFun = class(TTestCase)
  private
    FRun: TRunOperation;
    // Run a single opcode against the fixture and return the produced next
    // state (not the test result, which only binary tests use).
    function NextStateOf(pOpCode: byte; pOp1, pOp2, pBase: integer): byte;
  protected
    procedure SetUp; override;
  published
    // A byte complement, not a sign-extended integer complement.
    procedure TestNotIsByteComplement;
    // The wrapping arithmetic opcodes truncate to a byte rather than trapping.
    procedure TestWrappingArithmetic;
    // The saturating and shift opcodes clamp/mask as documented.
    procedure TestSaturatingAndShiftOps;
    // Division and modulo by zero leave the next state at its initial value.
    procedure TestDivModByZeroIsInert;
    // The bidimensional position of (X,Y) strides by the image width.
    procedure TestMake2DStridesByImageWidth;
  end;

implementation

procedure TTestNeuralABFun.SetUp;
var
  Actions, CurrentStates, NextStates: array of byte;
  I: integer;
begin
  SetLength(Actions, 256);
  SetLength(CurrentStates, 256);
  SetLength(NextStates, 256);
  for I := 0 to 255 do
  begin
    Actions[I] := I;
    CurrentStates[I] := 255 - I;
    NextStates[I] := 0;
  end;
  FRun.Load(csCreateOpDefault, Actions, CurrentStates, NextStates);
end;

function TTestNeuralABFun.NextStateOf(pOpCode: byte;
  pOp1, pOp2, pBase: integer): byte;
var
  Oper: TOperation;
begin
  Oper := CreateOperation(pOpCode, pOp1, pOp2);
  Result := 0;
  FRun.OperateAndTestOperation(Oper, pBase, Result);
end;

procedure TTestNeuralABFun.TestNotIsByteComplement;
begin
  // CurrentStates[10] = 245; not 245 = -246, which as a byte is 10.
  AssertEquals('not 245', 10, NextStateOf(csNot, 0, 0, 10));
  // CurrentStates[255] = 0; the complement of 0 is the all-ones byte.
  AssertEquals('not 0', 255, NextStateOf(csNot, 0, 0, 255));
  AssertEquals('not 255', 0, NextStateOf(csNot, 0, 0, 0));
end;

procedure TTestNeuralABFun.TestWrappingArithmetic;
begin
  // CurrentStates[0] = 255, so inc wraps to 0 and dec of 0 wraps to 255.
  AssertEquals('inc wraps', 0, NextStateOf(csInc, 0, 0, 0));
  AssertEquals('dec wraps', 255, NextStateOf(csDec, 0, 0, 255));
  // CurrentStates[10] = 245, CurrentStates[20] = 235; 245 + 235 = 480 -> 224.
  AssertEquals('add wraps', 224, NextStateOf(csAdd, 10, 20, 0));
  // 245 - 235 = 10.
  AssertEquals('sub in range', 10, NextStateOf(csSub, 10, 20, 0));
  // 235 - 245 = -10 -> 246.
  AssertEquals('sub wraps', 246, NextStateOf(csSub, 20, 10, 0));
  // 245 * 235 = 57575 -> 57575 and 255 = 231.
  AssertEquals('mul wraps', 231, NextStateOf(csMul, 10, 20, 0));
end;

procedure TTestNeuralABFun.TestSaturatingAndShiftOps;
begin
  // 245 + 235 saturates at 255 instead of wrapping.
  AssertEquals('adds saturates', 255, NextStateOf(csAddS, 10, 20, 0));
  // 235 - 245 saturates at 0.
  AssertEquals('subs saturates', 0, NextStateOf(csSubS, 20, 10, 0));
  // CurrentStates[250] = 5, CurrentStates[253] = 2; 5 shl 2 = 20.
  AssertEquals('shl in range', 20, NextStateOf(csShl, 250, 253, 0));
  // 245 shl (2) = 980 -> 980 and 255 = 212.
  AssertEquals('shl masks', 212, NextStateOf(csShl, 10, 253, 0));
  AssertEquals('shr', 61, NextStateOf(csShr, 10, 253, 0));
  // abs(245 - 235); both operands widen to integer before subtracting.
  AssertEquals('absdiff', 10, NextStateOf(csAbsDiff, 10, 20, 0));
  AssertEquals('avg', 240, NextStateOf(csAvg, 10, 20, 0));
end;

procedure TTestNeuralABFun.TestDivModByZeroIsInert;
begin
  // CurrentStates[255] = 0, so this is a division by zero: the arm is skipped
  // and the caller-initialised next state survives.
  AssertEquals('div by zero', 0, NextStateOf(csDiv, 10, 255, 0));
  AssertEquals('mod by zero', 0, NextStateOf(csMod, 10, 255, 0));
  // CurrentStates[20] = 235; 245 div 235 = 1, 245 mod 235 = 10.
  AssertEquals('div', 1, NextStateOf(csDiv, 10, 20, 0));
  AssertEquals('mod', 10, NextStateOf(csMod, 10, 20, 0));
end;

procedure TTestNeuralABFun.TestMake2DStridesByImageWidth;
var
  Engine: TRunOperation;
  Settings: TCreateOperationSettings;
  Actions, CurrentStates, NextStates: array of byte;
  X, Y: integer;
begin
  Settings := csCreateOpImageProcessing;
  Settings.ImageSizeX := 8;
  Settings.ImageSizeY := 4;
  SetLength(Actions, Settings.ImageSizeX * Settings.ImageSizeY);
  SetLength(CurrentStates, Settings.ImageSizeX * Settings.ImageSizeY);
  SetLength(NextStates, Settings.ImageSizeX * Settings.ImageSizeY);
  Engine.Load(Settings, Actions, CurrentStates, NextStates);

  // A step along X is one element; a step along Y is one full row of 8.
  AssertEquals('origin', 0, Engine.Make2D(0, 0));
  AssertEquals('x step', 3, Engine.Make2D(3, 0));
  AssertEquals('y step', 8, Engine.Make2D(0, 1));
  AssertEquals('x and y', 19, Engine.Make2D(3, 2));
  // Every pixel of the non-square image must map to a distinct position inside
  // the array; the last one is exactly the last element.
  AssertEquals('last pixel',
    Settings.ImageSizeX * Settings.ImageSizeY - 1,
    Engine.Make2D(Settings.ImageSizeX - 1, Settings.ImageSizeY - 1));
  for Y := 0 to Settings.ImageSizeY - 1 do
    for X := 0 to Settings.ImageSizeX - 1 do
      AssertEquals('pixel '+IntToStr(X)+','+IntToStr(Y),
        Y * Settings.ImageSizeX + X, Engine.Make2D(X, Y));
end;

initialization
  RegisterTest(TTestNeuralABFun);
end.
