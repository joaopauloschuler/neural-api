{
A->B FUNCTIONS pascal Unit
Copyright (C) 2001 Joao Paulo Schwarz Schuler

This unit contains/is able to:
* Contains all neurons (tests and operations) to be used by the prediction system.
* Can create neurons of the type "operations".
* Can create tests (conditions) and test lists.
* Can run neurons/operations and produce the next state.
* Verify tests.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
}

unit neuralabfun;

interface

uses neuralab;

// available operations. Some operations are logic/test operations such as <,> and <>.
// Other operations are math operations such as +,- and *.
const
  csNop = 0;     // no operation
  csEqual = 1;   // NextState[Base] := (Op1 = State[Op2]);
  csEqualM = 2;  // NextState[Base] := (State[Op1] = State[Op2]);
  csDifer = 3;   // NextState[Base] := (State[Op1] <> State[Op2]);
  csGreater = 4; // NextState[Base] := (State[Op1] > State[Op2]);
  csLesser = 5;  // NextState[Base] := (State[Op1] < State[Op2]);
  csTrue = 6;    // NextState[Base] := TRUE;
  csSet = 7;     // NextState[Base] := Op1;
  csInc = 8;     // NextState[Base] := State[Base] + 1;
  csDec = 9;     // NextState[Base] := State[Base] - 1;
  csAdd = 10;    // NextState[Base] := State[Op1] +   State[Op2];
  csSub = 11;    // NextState[Base] := State[Op1] -   State[Op2];
  csMul = 12;    // NextState[Base] := State[Op1] *   State[Op2];
  csDiv = 13;    // NextState[Base] := State[Op1] div State[Op2];
  csMod = 14;    // NextState[Base] := State[Op1] mod State[Op2];
  csAnd = 15;    // NextState[Base] := State[Op1] and State[Op2];
  csOr = 16;    // NextState[Base] := State[Op1] or  State[Op2];
  csXor = 17;    // NextState[Base] := State[Op1] xor State[Op2];
  csInj = 18;    // NextState[Base] := State[Op1];
  csNot = 19;    // NextState[BASE] := not(PreviousState[BASE])

// An Operation type contains: an operation, 2 operands and boolean operand modifiers.
type

  TPositionArray = array of integer;

  { TOperation }
  TOperation = object
    OpCode: byte; //Operand Code
    Op1: integer; //Operand 1
    Op2: integer; //Operand 2
    RelativeOperandPosition1: boolean;//Operand position is relative
    RelativeOperandPosition2: boolean;
    RunOnAction: boolean;

    // creates a string from the operation
    function ToString(): string;

    // creates a human readable string
    function ToHumanString(): string;

    // loads the object from string
    procedure LoadFromString(str: string);

    function Define(FN: byte; pOp1: integer; pOp2: integer; RelOp1: boolean = False;
      RelOp2: boolean = False; PRunOnAction: boolean = False): TOperation;
  end;

// "RelativeOperandPosition" Modifier Examples
// As an example, if RelativeOperandPosition1 is false, then we have
// NextState[Base] := State[Op1] + State[Op2];

// If RelativeOperandPosition1 is TRUE, then we have
// NextState[Base] := State[BASE + Op1] + State[Op2];

// If RunOnAction is TRUE and RelativeOperandPosition1 is FALSE, then we have:
// NextState[Base] := State[Op1] + Action[Op2];

// "RunOnAction" modifies first operator in Unary operations and
// modifies second operator in binary operations.

function CreateOperation(FN: byte; pOp1: integer; pOp2: integer; RelOp1: boolean = False;
  RelOp2: boolean = False; PRunOnAction: boolean = False): TOperation;

function CreateValidBinaryTest(Val1, Val2: byte; pOp1, pOp2: integer; RelOp1, RelOp2,
  PRunOnAction: boolean ): TOperation;

// number of available operations
const
  csMaxOperations = 20;

// this array maps OpCode into its string representation
const
  csStrOp: array[0..csMaxOperations - 1] of string[15] =
    //0    1   2    3    4   5   6   7      8     9     10  11  12  13    14
    ('nop', '=', '=', '<>', '>', '<', 'V', ' := ', 'inc', 'dec', '+', '-', '*', 'div', 'mod',
    //15    16   17    18    19
    'and', 'or', 'xor', 'inj', 'not');

// this type represents sets of operations
type
  TOperationSet = set of 0 .. csMaxOperations - 1;

const
  TestOperationSet: TOperationSet =
    [csEqual, csEqualM, csDifer, csGreater, csLesser, csTrue];

  ImediatSet: TOperationSet = [csEqual, csSet];

  FirstImediatSet: TOperationSet = [csEqual];

  StateOperationSet: TOperationSet =
    [csInc, csDec, csAdd, csSub, csMul, csDiv, csMod, csAnd, csXor, csOr, csNot, csInj];

  BinaryOperationSet: TOperationSet =
    [csAdd, csSub, csMul, csDiv, csMod, csAnd, csXor, csOr, csEqual, csEqualM,
    csDifer, csGreater, csLesser];

  NoArgSet: TOperationSet = [csInc, csDec, csNot, csInj];

// maximum number of tests on a given rule (tests implies into operation).
const
  csMaxTests = 30;

// TTestsClass represents a list of tests such as (A>B) and (C=D) and (E>F).
type
  TTestsClass = object
    TestBasePosition: integer;

    // Neuronal TestThreshold - minimum number of valid test to fire the operation
    TestThreshold: byte;

    // when N is Zero, there is no operation/neuron
    N: byte;
    T: array [0..csMaxTests - 1] of TOperation;

    // creates a human readable string
    function ToHumanString(): string;

    // Removes an operation from T array at operationIndex
    procedure DeleteOperation(operationIndex: integer);

    // creates a string for storage
    function ToString(): string;

    // loads the object data from string
    procedure LoadFromString(str: string);

    // Adds a test to the end of test list
    procedure AddTest(Adding: TOperation);
  end;

type
  TCreateOperationSettings = record
    AddSetOp: boolean;
    AddBinaryOp: boolean;
    AddBinaryTest: boolean;
    AddEqualTest: boolean;
    AddTrueTest: boolean;
    AddIncOp: boolean;
    AddDecOp: boolean;
    AddInjOp: boolean;
    AddNotOp: boolean;
    TestOnActions: boolean;
    TestOnStates: boolean;
    Bidimensional: boolean;
    PartialTestEval: boolean;
    MinTests: byte;
    MaxTests: byte;
    FeatureSize: byte;
    ImageSizeX: integer;
    ImageSizeY: integer;
    MinSampleForPrediction: integer;
  end;

const
  csCreateOpDefault: TCreateOperationSettings =
  (
    AddSetOp: true;
    AddBinaryOp: true;
    AddBinaryTest: true;
    AddEqualTest: true;
    AddTrueTest: true;
    AddIncOp: true;
    AddDecOp: true;
    AddInjOp: true;
    AddNotOp: true;
    TestOnActions: true;
    TestOnStates: true;
    Bidimensional: false;
    PartialTestEval: false;
    MinTests: 1;
    MaxTests: csMaxTests;
    FeatureSize: 0;
    ImageSizeX: 0;
    ImageSizeY: 0;
    MinSampleForPrediction: 0;
  );

  csCreateOpImageProcessing: TCreateOperationSettings =
  (
    AddSetOp: true;
    AddBinaryOp: false;
    AddBinaryTest: true;
    AddEqualTest: false;
    AddTrueTest: false;
    AddIncOp: false;
    AddDecOp: false;
    AddInjOp: false;
    AddNotOp: false;
    TestOnActions: true;
    TestOnStates: false;
    Bidimensional: True;
    PartialTestEval: true;
    MinTests: 1;
    MaxTests: csMaxTests;
    FeatureSize: 2;
    ImageSizeX: 32;
    ImageSizeY: 32;
    MinSampleForPrediction: 10;
  );

  csCreateOpSimplest: TCreateOperationSettings =
  (
    AddSetOp: true;
    AddBinaryOp: false;
    AddBinaryTest: false;
    AddEqualTest: true;
    AddTrueTest: false;
    AddIncOp: false;
    AddDecOp: false;
    AddInjOp: false;
    AddNotOp: false;
    TestOnActions: true;
    TestOnStates: false;
    Bidimensional: false;
    PartialTestEval: false;
    MinTests: 1;
    MaxTests: csMaxTests;
    FeatureSize: 0;
    ImageSizeX: 0;
    ImageSizeY: 0;
    MinSampleForPrediction: 0;
  );

(*
  This is one of the most interesting classes in the entire project.
  This class is responsible for processing operations. Operations can run tests on
  Actions array or on Current States array and create a new state on Next States array.
*)
type

  { TRunOperation }
  TRunOperation = object
  private
    Actions, CurrentStates, NextStates: array of byte;
    NumberOfActions, NumberOfCurrentStates, NumberOfNextStates: integer;
    FCS: TCreateOperationSettings;

    function LocalTestTests(var Tests: TTestsClass): integer;
    function ConvoluteTests(var Tests: TTestsClass): integer;

  public
    // This method loads arrays to be later operated.
    procedure Load(PCS: TCreateOperationSettings; var PActions, PCurrentStates,
      PNextStates: array of byte);

    // 2D Image Specific Functions
    function Make2D(X,Y: integer): integer;
    function CreateFeatureCenter(): integer;
    function GetRandom2DPos(Center:integer): integer;

    // creates a random binary test from action array
    function CreateActionRandomBinaryTest(): TOperation;

    function OperateAndTestOperation(Oper: TOperation; BasePosition: integer;
      var NextState: byte): byte;
    function TestTests(var Tests: TTestsClass): integer;
  end;

const
  csMaxOperationsArray = 1500;

type
  TOperationsArray = array[0..csMaxOperationsArray] of TOperation;

type

  { TCreateValidOperations }

  TCreateValidOperations = object(TRunOperation)
  private
    FOperations: TOperationsArray;
    FOperationsCnt: integer;
    FBasePosition: integer;
    FPredictedBytePos: integer;
    FFeatureCenter2D: integer;

    procedure Include(O: TOperation);
    function CanInclude: boolean;

  public
    procedure LoadCreationData(PredictedBytePos: integer);

    // Returns the number of current valid operations in FOperations.
    function GetCount(): integer;

    // This function returns an already created random valid operation.
    function GetRandomOp: TOperation;

    // returns the Image Feature Center
    function GetFeatureCenter2D: integer;

    procedure Clear;
    (*
    This function creates valid operations and includes them into FOperations.
    Tests: field indicates if tests/conditions should be included.
    FullEqual: indicates if all input data should be used. False means only non zero values will be used.
    ERRORS: prediction error are used to create better operations.
    *)
    procedure Create(Tests, FullEqual: boolean; var ERRORS: array of byte);
  end;


implementation

uses Classes, SysUtils, Math;

procedure TTestsClass.DeleteOperation(operationIndex: integer);
var
  I: integer;
begin
  if operationIndex = Self.N - 1 then
    Dec(Self.N)
  else
  begin
    if (operationIndex >= 0) and (operationIndex < Self.N - 1) and
      (operationIndex < Self.N - 1) then
    begin
      for I := operationIndex to Self.N - 2 do
        Self.T[I] := Self.T[I + 1];
      Dec(Self.N);
    end
    else
    begin
      Writeln('ERROR: DeleteOperation N:', Self.N);
    end;
  end;
end; // of procedure DeleteOperation

function TTestsClass.ToString: string;
var
  S: TStringList;
  NCount: integer;
begin
  Result := '';
  if (Self.N > 0) then
  begin
    S := TStringList.Create;
    S.Sorted := false;
    S.Delimiter := '=';
    S.StrictDelimiter := true;
    S.Add( IntToStr(Self.TestBasePosition) );
    S.Add( IntToStr(Self.TestThreshold) );
    for NCount := 0 to Self.N - 1 do
    begin
      S.Add( Self.T[NCount].ToString() );
    end;
    Result := S.DelimitedText;
    S.Free;
  end;
end;

procedure TTestsClass.LoadFromString(str: string);
var
  S: TStringList;
  NCount: integer;
begin
  if (Length(str) > 0) then
  begin
    S := TStringList.Create;
    S.Sorted := false;
    S.Delimiter := '=';
    S.StrictDelimiter := true;
    S.DelimitedText := str;
    Self.TestBasePosition := StrToInt(S[0]);
    Self.TestThreshold := StrToInt(S[1]);
    Self.N := S.Count - 2;
    for NCount := 2 to S.Count - 1 do
    begin
      Self.T[NCount-1].LoadFromString(S[NCount]);
    end;
    S.Free;
  end;
end;

procedure TTestsClass.AddTest(Adding: TOperation);
begin
  if N < csMaxTests then
  begin
    T[N] := Adding;
    Inc(N);
  end;
end;

function SignToStr(X: extended): char;
begin
  if X >= 0 then
    SignToStr := '+'
  else
    SignToStr := '-';
end;

function TOperation.ToHumanString: string;
var
  bOpCode: byte;
  sOp1, sOp2: string;
begin
  bOpCode := self.OpCode and 63;
  sOp1 := IntToStr(self.Op1);
  sOp2 := IntToStr(self.Op2);

  if self.RelativeOperandPosition1 then
  begin
    if self.Op1 > 0 then
      sOp1 := '+' + sOp1;
    if self.Op1 <> 0 then
      sOp1 := 'B' + sOp1
    else
      sOp1 := 'B';
  end;

  if self.RelativeOperandPosition2 then
  begin
    if self.Op2 > 0 then
      sOp2 := '+' + sOp2;
    if self.Op2 <> 0 then
      sOp2 := 'B' + sOp2
    else
      sOp2 := 'B';
  end;

  if not (bOpCode in ImediatSet) then
  begin
    sOp1 := '[' + sOp1 + ']';
    sOp2 := '[' + sOp2 + ']';
    if self.RunOnAction then
    begin
      sOp1 := 'A' + sOp1;
      sOp2 := 'A' + sOp2;
    end
    else
    begin
      sOp1 := 'S' + sOp1;
      sOp2 := 'S' + sOp2;
    end;
  end
  else
  begin
    if (bOpCode in FirstImediatSet) then
    begin
      sOp2 := '[' + sOp2 + ']';
      if self.RunOnAction then
        sOp2 := 'A' + sOp2
      else
        sOp2 := 'S' + sOp2;
    end;
  end;

  if OpCode in BinaryOperationSet then
  begin
    result := '(' + sOp1 + ' ' + csStrOp[OpCode] + ' ' + sOp2 + ')';
    exit;
  end;

  if OpCode = csTrue then
  begin
    result := 'V';
    exit;
  end;

  if OpCode = csSet then
  begin
    result := IntToStr(self.Op1);
    exit;
  end;

  if OpCode in NoArgSet then
  begin
    result := csStrOp[OpCode] + ' S[B]';
    exit;
  end;

  result := '(' + csStrOp[OpCode] + ',' + sOp1 + ',' + sOp2 + ')';
end;

function TTestsClass.ToHumanString: string;
var
  I: integer;
  R: string;
begin
  R := '';
  if self.N > 0 then
  begin
    R := R + 'B=' + IntToStr(self.TestBasePosition) + ' ';
    for I := 0 to self.N - 1 do
      R := R + ( self.T[I].ToHumanString() );
  end;
  result := R;
end;

{ TOperation }

function TOperation.ToString: string;
begin
  Result :=
    IntToStr(self.OpCode) + ',' +
    IntToStr(self.Op1) + ',' +
    IntToStr(self.Op2) + ',' +
    BoolToChar[self.RelativeOperandPosition1] + ',' +
    BoolToChar[self.RelativeOperandPosition2] + ',' +
    BoolToChar[self.RunOnAction];
end;

procedure TOperation.LoadFromString(str: string);
var
  S: TStringList;
begin
  S := TStringList.Create;
  S.Sorted := false;
  S.Delimiter := ',';
  S.StrictDelimiter := true;
  S.DelimitedText := str;
  self.OpCode := StrToInt(S[0]);
  self.Op1 := StrToInt(S[1]);
  self.Op2 := StrToInt(S[2]);
  self.RelativeOperandPosition1 := CharToBool[S[3][1]];
  self.RelativeOperandPosition2 := CharToBool[S[4][1]];
  self.RunOnAction := CharToBool[S[5][1]];
  S.Free;
end;

function TOperation.Define(FN: byte; pOp1: integer; pOp2: integer;
  RelOp1: boolean; RelOp2: boolean; PRunOnAction: boolean): TOperation;
begin
  Self := CreateOperation(FN, pOp1, pOp2, RelOp1, RelOp2, PRunOnAction);
  Result := Self;
end;

function TCreateValidOperations.GetCount: integer;
begin
  Result := FOperationsCnt;
end;

function TCreateValidOperations.CanInclude: boolean;
var
  R: boolean;
begin
  R := FOperationsCnt < csMaxOperationsArray - 1;
  //if not (R) then
    //writeln('Warning: can not include operation.');
  CanInclude := R;
end;

procedure TCreateValidOperations.LoadCreationData(PredictedBytePos: integer);
begin
  if FCS.MaxTests>csMaxTests then FCS.MaxTests := csMaxTests;
  Self.FPredictedBytePos := PredictedBytePos;
  FFeatureCenter2D := 0;

  if FCS.Bidimensional then
    Self.FFeatureCenter2D := Self.CreateFeatureCenter();
end;

procedure TCreateValidOperations.Include(O: TOperation);
var
  predictState: byte;
begin

  if (O.OpCode in ImediatSet) then
  begin
    O.RelativeOperandPosition1 := False;
    O.RelativeOperandPosition2 := False;
  end;

  if O.OpCode in FirstImediatSet then
    O.RelativeOperandPosition1 := False;

  if O.RelativeOperandPosition1 then
  begin
    O.OP1 := O.OP1 - FBasePosition;
  end;

  if O.RelativeOperandPosition2 then
  begin
    O.OP2 := O.OP2 - FBasePosition;
  end;

  // if the operation has the predicted state (returns true), include it.
  if ByteToBool[OperateAndTestOperation(O, FBasePosition, predictState)] then
  begin
    // include the new operation in FOperations
    FOperations[FOperationsCnt] := O;
    Inc(FOperationsCnt);
  end;
end;

procedure TCreateValidOperations.Clear;
begin
  FOperationsCnt := 0;
end;

// This function returns an array with all non zero elements
function getNonZeroElementsPos(InputLen: integer; var InputData: array of byte; var OutputData: TPositionArray):integer;
var
  InputPos, OutputCount: integer;
begin
  OutputCount := 0;
  for InputPos := 0 to InputLen - 1 do
  begin
    if InputData[InputPos] <> 0 then
    begin
      OutputData[OutputCount] := InputPos;
      OutputCount := OutputCount + 1;
    end;
  end;
  Result := OutputCount;
end;

function CreateOperation(FN: byte; pOp1: integer; pOp2: integer; RelOp1: boolean = False;
  RelOp2: boolean = False; PRunOnAction: boolean = False): TOperation;
begin
  Result.OpCode := FN;
  Result.Op1 := pOp1;
  Result.Op2 := pOp2;
  Result.RelativeOperandPosition1 := RelOp1;
  Result.RelativeOperandPosition2 := RelOp2;
  Result.RunOnAction := PRunOnAction;
end;

function CreateValidBinaryTest(Val1, Val2: byte; pOp1, pOp2: integer; RelOp1, RelOp2,
  PRunOnAction: boolean): TOperation;
var
  FN: byte;
begin
  if Val1>Val2 then FN := csGreater
  else if Val1<Val2 then FN := csLesser
  else FN := csEqual;

  Result := CreateOperation(FN,pOp1,pOp2,RelOp1,RelOp2,PRunOnAction);
end;

const
  csUnitaryTests: array [0..0] of byte = (csEqual);
  csBinaryTests: array [0..3] of byte = (csEqualM, csDifer, csGreater, csLesser);
  csBinaryOperations: array[0..7] of byte =
    (csAdd, csSub, csMul, csDiv, csMod, csAnd, csXor, csOr);

function GetRandom2DDist(FeatureSize: byte): integer;
begin
  Result := random(FeatureSize*2+1)-FeatureSize;
end;

function TRunOperation.Make2D(X,Y: integer): integer;
begin
  Result := X + Y * FCS.ImageSizeY;
end;

function TRunOperation.CreateFeatureCenter(): integer;
var
  X, Y: integer;
begin
  X := FCS.FeatureSize + random(FCS.ImageSizeX-FCS.FeatureSize*2);
  Y := FCS.FeatureSize + random(FCS.ImageSizeY-FCS.FeatureSize*2);
  Result := Make2D(X,Y);
end;

function TRunOperation.GetRandom2DPos(Center:integer): integer;
begin
  Result := Center + Make2D( GetRandom2DDist(FCS.FeatureSize), GetRandom2DDist(FCS.FeatureSize) );
end;

procedure TCreateValidOperations.Create(Tests, FullEqual: boolean;
  var ERRORS: array of byte);
var
  LocalNonZeroPrevStates, NonZeroErrors: TPositionArray;
  LocalNonZeroPrevStatesCount, NonZeroErrorsCount: integer;
  LocalNumberOfPreviousStates: integer;
  LocalPreviousStates: array of byte;
  OnAction: boolean;

  procedure IncludeEqual();
  var
    ElementPosition, J, MJ: integer;
  begin
    if LocalNumberOfPreviousStates > csMaxTests then
      MJ := FCS.MaxTests
    else
      MJ := LocalNumberOfPreviousStates;
    for J := 1 to MJ do
    begin
      ElementPosition := random(LocalNumberOfPreviousStates);
      if CanInclude then
        Include(CreateOperation(csEqual, LocalPreviousStates[ElementPosition],
          ElementPosition, False, False, OnAction))
      else
        exit;
    end;
  end;

  procedure IncludeEqualForNonZero;
  var
    ElementPosition, J, MJ: integer;
    Value: byte;
  begin
    if (LocalNonZeroPrevStatesCount > 0) then
    begin
      if LocalNonZeroPrevStatesCount > csMaxTests then
        MJ := FCS.MaxTests
      else
        MJ := LocalNonZeroPrevStatesCount+1;
      for J := 1 to MJ do
      begin
        ElementPosition := LocalNonZeroPrevStates[random(LocalNonZeroPrevStatesCount)];
        Value := LocalPreviousStates[ElementPosition];

        if (Value=0) then
          Writeln('ERROR: IncludeEqualForNonZero created zero equal: ',ElementPosition);

        if (CanInclude and (Value <>0) )then
          Include(CreateOperation(csEqual, Value,
            ElementPosition, False, False, OnAction))
        else
          exit;
      end;
    end;
  end;

  procedure IncludeBinaryTestsForNonZero;
  var
    I, MI: integer;
    Pos1, Pos2: integer;
    NZPos1, NZPos2: integer;
    Val1, Val2: byte;
  begin
    if (LocalNonZeroPrevStatesCount >= 2) then
    begin
      MI := FCS.MaxTests;
      for I := 0 to MI - 1 do
      begin
        NZPos1 := random(LocalNonZeroPrevStatesCount);
        Pos1 := LocalNonZeroPrevStates[NZPos1];
        if (FCS.Bidimensional) then
        begin
          Pos2 := GetRandom2DPos(Pos1);
        end else
        begin
          NZPos2 := random(LocalNonZeroPrevStatesCount);
          Pos2 := LocalNonZeroPrevStates[NZPos2];
        end;
        Val1 := LocalPreviousStates[Pos1];
        Val2 := LocalPreviousStates[Pos2];

        if (Val1=0) or (Val2=0) then
          Writeln('ERROR: IncludeBinaryTestsForNonZero created zero equal: ',Val1,' ',Val2);

        if CanInclude and (Pos1 <> Pos2) then
        begin
          Include( CreateValidBinaryTest(Val1, Val2, Pos1, Pos2, True, True, OnAction) );
        end;
      end;
    end;
  end;

  procedure IncludeBinaryTests;
  var
    I, MI: integer;
    Pos1, Pos2: integer;
    Val1, Val2: byte;

  begin
    if (LocalNumberOfPreviousStates >= 2) then
    begin
      MI := FCS.MaxTests;

      for I := 0 to MI - 1 do
      begin
        if (FCS.Bidimensional) then
        begin
          Pos1 := GetRandom2DPos(FBasePosition);
          Pos2 := GetRandom2DPos(FBasePosition);
        end else
        begin
          Pos1 := random(LocalNumberOfPreviousStates);
          Pos2 := random(LocalNumberOfPreviousStates);
        end;
        Val1 := LocalPreviousStates[Pos1];
        Val2 := LocalPreviousStates[Pos2];

        if CanInclude and (Pos1 <> Pos2) then
          Include( CreateValidBinaryTest(Val1, Val2, Pos1, Pos2, True, True, OnAction) );
      end;
    end;
  end;

  procedure IncludeBinaryOperationsForNonZero;
  var
    I, J, K: integer;
  begin
    if (LocalNonZeroPrevStatesCount >= 2) then
    begin
      begin
        I := random(LocalNonZeroPrevStatesCount);
        J := random(LocalNonZeroPrevStatesCount);
        begin
          K := random(length(csBinaryOperations));
          if CanInclude and (I <> J) then
            Include(CreateOperation(csBinaryOperations[K], LocalNonZeroPrevStates[I],
              LocalNonZeroPrevStates[J], False, False, OnAction));
        end;
      end;
    end;
  end;

  procedure IncludeBinaryOperations;
  var
    I, J, K: integer;
  begin
    if (LocalNumberOfPreviousStates >= 2)
    then
    begin
      I := random(LocalNumberOfPreviousStates);
      J := random(LocalNumberOfPreviousStates);
      K := random(length(csBinaryOperations));
      if CanInclude and (I <> J) then
        Include(CreateOperation(csBinaryOperations[K], I, J, False, False, OnAction));
    end;
  end;

  procedure PSet;
  begin
    if CanInclude then
      Include(CreateOperation(csSet, NextStates[FPredictedBytePos], 0, False, False, False))
  end;


var
  NumberOfErrors: integer;
  RunOnActionFlag: integer;
begin
  Clear;
  SetLength(NonZeroErrors,Self.NumberOfCurrentStates);
  NumberOfErrors := NumberOfNextStates;
  NonZeroErrorsCount := getNonZeroElementsPos(NumberOfErrors, ERRORS, NonZeroErrors);

  if not(FCS.TestOnStates) then RunOnActionFlag := 1
  else if not(FCS.TestOnActions) then RunOnActionFlag := 0
  // 50% of change that operations will be based on actions or on states.
  else RunOnActionFlag := random(2);

  if ( RunOnActionFlag > 0) then // should we use action or previous states for inference?
  begin
    // using LocalPreviousStates
    LocalNumberOfPreviousStates := NumberOfActions;
    SetLength(LocalPreviousStates, NumberOfActions);
    SetLength(LocalNonZeroPrevStates, NumberOfActions);
    ABCopy(LocalPreviousStates, Actions);
    LocalNonZeroPrevStatesCount := getNonZeroElementsPos(LocalNumberOfPreviousStates, LocalPreviousStates,
      LocalNonZeroPrevStates);
    OnAction := True;
  end
  else
  begin
    // using previous states
    SetLength(LocalNonZeroPrevStates,Self.NumberOfCurrentStates);
    LocalNonZeroPrevStatesCount := getNonZeroElementsPos(NumberOfCurrentStates, CurrentStates, LocalNonZeroPrevStates);
    OnAction := False;
    LocalNumberOfPreviousStates := NumberOfCurrentStates;
    SetLength(LocalPreviousStates, NumberOfCurrentStates);
    ABCopy(LocalPreviousStates, CurrentStates);
  end;

  FBasePosition := FPredictedBytePos;

  if Tests then
  begin
    if FCS.Bidimensional then FBasePosition := FFeatureCenter2D;
    if FullEqual or FCS.Bidimensional then
    begin
      if FCS.AddEqualTest then IncludeEqual;
      if FCS.AddBinaryTest then IncludeBinaryTests;
    end
    else
    begin
      if FCS.AddEqualTest then IncludeEqualForNonZero;
      if FCS.AddBinaryTest then IncludeBinaryTestsForNonZero;
    end;
    if FCS.AddTrueTest then Include(CreateOperation(csTrue, 0, 0, False, False, False));
  end
  else
  begin

    if FCS.AddSetOp then PSet;

    if FCS.AddBinaryOp then
    begin
      if FullEqual then
        IncludeBinaryOperations
      else
        IncludeBinaryOperationsForNonZero;
    end;

    if FCS.AddIncOp then Include(CreateOperation(csInc, 0, 0, False, False, False));
    if FCS.AddDecOp then Include(CreateOperation(csDec, 0, 0, False, False, False));
    if FCS.AddInjOp then Include(CreateOperation(csInj, 0, 0, False, False, False));
    if FCS.AddNotOp then Include(CreateOperation(csNot, 0, 0, False, False, False));
  end;

  SetLength(LocalNonZeroPrevStates, 0);
  SetLength(NonZeroErrors, 0);
  SetLength(LocalPreviousStates, 0);

end;

function TCreateValidOperations.GetRandomOp: TOperation;
var
  ResultingPosition: longint;
  MAX: longint;
  R: TOperation;
begin
  MAX := 0;

  if (FOperationsCnt = 1) then
  begin
    ResultingPosition := 0;
  end
  else
  begin
    ResultingPosition := random(FOperationsCnt);
  end;

  R := FOperations[ResultingPosition];

  if (FOperationsCnt > 1) then
  begin
    while (MAX < 100) and (R.OpCode = 0) do
    begin
      Inc(MAX);
      ResultingPosition := random(FOperationsCnt);
      R := FOperations[ResultingPosition];
    end;
  end
{
// this code has been commented as it creates too many warnings.
else
begin
  if (R.OpCode=0) then
    writeln('WARNING: Operand Code is zero.',FOperationsCnt);
end
};

  FOperations[ResultingPosition].OpCode := 0;
  GetRandomOp := R;
  //if MAX = 100 then
    //writeln('ERROR: max on getrandom max.', FOperationsCnt);
end;

function TCreateValidOperations.GetFeatureCenter2D: integer;
begin
  Result := Self.FFeatureCenter2D;
end;

function TRunOperation.CreateActionRandomBinaryTest: TOperation;
var
  Pos1, Pos2: integer;
  Val1, Val2: byte;
begin
  if (FCS.Bidimensional) then
  begin
    Pos1 := GetRandom2DPos(0);
    Pos2 := GetRandom2DPos(0);
  end else
  begin
    Pos1 := random(Self.NumberOfActions);
    Pos2 := random(Self.NumberOfActions);
  end;

  Val1 := Random(255);
  Val2 := Random(255);

  Result := CreateValidBinaryTest(Val1, Val2, Pos1, Pos2, True, True, True);
end;

function TRunOperation.LocalTestTests(var Tests: TTestsClass): integer;
var
  POS: integer;
  efeito: byte;
  PermissibleErrors: ShortInt;
begin
  if Tests.N > 0 then
  begin
    PermissibleErrors := Tests.N - Tests.TestThreshold;
    POS := 0;
    while (PermissibleErrors >= 0) and (POS < Tests.N) do
    begin
      if (0 = OperateAndTestOperation(Tests.T[POS], Tests.TestBasePosition, efeito))
        then Dec(PermissibleErrors);
      Inc(POS);
    end;
  end;

  if ( (PermissibleErrors >= 0) and (Tests.N > 0) )
    then Result := Tests.N - PermissibleErrors
    else Result := 0;
end;

function TRunOperation.ConvoluteTests(var Tests: TTestsClass): integer;
var
  X,Y: integer;
  TopX, TopY: integer;
  Pos: integer;
begin
  Result := 0;
  TopX := FCS.ImageSizeX - FCS.FeatureSize - 1;
  TopY := FCS.ImageSizeY - FCS.FeatureSize - 1;
  for X := FCS.FeatureSize to TopX do
  begin
    for Y := FCS.FeatureSize to TopY do
    begin
      Pos := Self.Make2D(X,Y);
      Tests.TestBasePosition := Pos;
      if (Self.LocalTestTests(Tests)>0) then
      begin
        Result := Pos;
        Break;
      end;
    end;
    if (Result>0) then Break;
  end;
end;

procedure TRunOperation.Load(PCS: TCreateOperationSettings; var PActions,
  PCurrentStates, PNextStates: array of byte);
begin
  FCS := PCS;
  NumberOfActions := length(PActions);
  NumberOfNextStates := length(PNextStates);
  NumberOfCurrentStates := length(PCurrentStates);

  SetLength(Self.Actions, NumberOfActions);
  SetLength(Self.NextStates, NumberOfNextStates);
  SetLength(Self.CurrentStates, NumberOfCurrentStates);

  ABCopy(Self.Actions, PActions);
  ABCopy(Self.NextStates, PNextStates);
  ABCopy(Self.CurrentStates, PCurrentStates);
end;

function TRunOperation.OperateAndTestOperation(
  {input} Oper: TOperation; BasePosition: integer;
  {output}var NextState: byte): byte;
var
  RelativeOperandPosition1,         //Operand position is relative
  RelativeOperandPosition2: boolean;
  OpCode: byte;
  Operand1Value, Operand2Value: integer;
  StatePosition1, StatePosition2: integer;
begin
  RelativeOperandPosition1 := Oper.RelativeOperandPosition1;
  RelativeOperandPosition2 := Oper.RelativeOperandPosition2;
  OpCode := Oper.OpCode and 63;

  if RelativeOperandPosition1 then
    StatePosition1 := BasePosition + Oper.Op1
  else
    StatePosition1 := Oper.Op1;

  if RelativeOperandPosition2 then
    StatePosition2 := BasePosition + Oper.Op2
  else
    StatePosition2 := Oper.Op2;

  if (not (OpCode in ImediatSet) and ((StatePosition1 >= NumberOfActions) or
    (StatePosition1 < 0))) or ((StatePosition2 >= NumberOfActions) or
    (StatePosition2 < 0)) then
  begin
    OperateAndTestOperation := 0;
    exit;
  end;

  if Oper.RunOnAction then
  begin
    //TODO: this is a hack. We need to find why the range is wrong.
    StatePosition1 := Min(Self.NumberOfActions - 1, StatePosition1);
    StatePosition2 := Min(Self.NumberOfActions - 1, StatePosition2);
    Operand1Value := Actions[StatePosition1];
    Operand2Value := Actions[StatePosition2];
  end
  else
  begin
    //TODO: this is a hack. We need to find why the range is wrong.
    StatePosition1 := Min(Self.NumberOfCurrentStates - 1, StatePosition1);
    StatePosition2 := Min(Self.NumberOfCurrentStates - 1, StatePosition2);
    Operand1Value := CurrentStates[StatePosition1];
    Operand2Value := CurrentStates[StatePosition2];
  end;

  try
    case OpCode of
      csNop:      begin { nop } end;
      csEqual:    NextState:=BoolToByte[Oper.Op1=Operand2Value];
      csEqualM:   NextState:=BoolToByte[Operand1Value=Operand2Value];
      csDifer:    NextState:=BoolToByte[Operand1Value<>Operand2Value];
      csGreater:  NextState:=BoolToByte[Operand1Value>Operand2Value];
      csLesser:   NextState:=BoolToByte[Operand1Value<Operand2Value];
      csTrue:     NextState:=1;
      csSet:      NextState:=Oper.Op1;
      csInc:      NextState:=(CurrentStates[BasePosition]+1) and 255{$R+};
      csDec:      NextState:=(CurrentStates[BasePosition]-1) and 255{$R+};
      csAdd:      NextState:=(Operand1Value+Operand2Value) and 255{$R+};
      csSub:      NextState:=(Operand1Value-Operand2Value) and 255{$R+};
      csMul:      NextState:=(Operand1Value*Operand2Value) and 255{$R+};
      csDiv:      if Operand2Value<>0
                     then{$R-}NextState:=(Operand1Value div Operand2Value) and 255{$R+};
      csMod:      if Operand2Value<>0
                     then {$R-}NextState:=(Operand1Value mod Operand2Value) and 255{$R+};
      csAnd:      NextState:=(Operand1Value and Operand2Value) and 255{$R+};
      csOr:       NextState:=(Operand1Value or Operand2Value) and 255{$R+};
      csXor:      NextState:=(Operand1Value xor Operand2Value) and 255{$R+};
      csInj:      NextState:=CurrentStates[BasePosition];
      csNot:      NextState:=not(CurrentStates[BasePosition]);
    else
      writeln('ERROR: invalid operation code:',Oper.OpCode);
    end;
  except
    Writeln('Error: OpCode =', OpCode, '  VOp1=', Operand1Value,
      '  VOp2=', Operand2Value, '  BASE=', BasePosition);
  end;
  if (OpCode in TestOperationSet) then
    OperateAndTestOperation := NextState
  else
    OperateAndTestOperation := BoolToByte[NextStates[BasePosition] = NextState];
end;

function TRunOperation.TestTests(var Tests: TTestsClass): integer;
begin
  if FCS.Bidimensional
    then Result := Self.ConvoluteTests(Tests)
    else Result := Self.LocalTestTests(Tests);
end;

end.
