{
Universal Byte Prediction Unit
Copyright (C) 2001 Joao Paulo Schwarz Schuler

  Classes in this unit implement combinatorial NEURAL NETWORKS:
  * They are problem independent (universal).
  * After learning, these classes can predict/classify future states.
  * In this code, "neurons" are sometimes named as "relations".
  * Some neurons are a "test" that relates a condition (current state) to an
  OPERATION.
  * Some neurons are "operations" that transform the current state into the
  next state (SECOND NEURAL NETWORK LAYER).
  * Does a NEURON fire at a given state? Yes. Each neuron "tests" for the
  existence of a given state.
  * There are 2 neural network layers: tests and operations.
  * The current state and the predicted state are Arrays of Bytes. This is why
  you can find "AB" along the source code.
  * NEURONS in this unit resemble microprocessor OPERATIONS and TESTS.
  * Neurons in this unit ARE NOT floating point operations.
  * Neurons that correctly predict future states get stronger.
  * Stronger neurons win the prediction.

  If you aren't sure what/where to look at, please have a look at
  TEasyLearnAndPredictClass

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

unit neuralbyteprediction; // this unit used to be called UBup3

{$IFDEF FPC}
{$MODE Delphi}
{$ENDIF}

interface

uses neuralabfun, neuralcache, neuralvolume;

type
  TCountings = array of longint;

  { TNeuronGroupBase }
  TNeuronGroupBase = object
    //Where in the output array this neuron group predicts a state
    PredictionPos: integer;

    // Number of Neuron Group Victories
    Vitories: integer;
    LastVictory: integer;

    // neuron prediction countings
    WrongNeuronPredictionCnt: integer;
    CorrectNeuronPredictionCnt: integer;

    // neuron prediction countings at win (at victory)
    WrongPredictionAtWin: integer;
    CorrectPredictionAtWin: integer;

    // returns Correct Prediction / Total = Correct Prediction Frequency
    function GetF: single;

    // returns (m+1)/(n+2)
    function GetD: single;

    // creates a string for storage
    function ToString(): string;

    procedure Clear;

  end;

  { TNeuronGroup }
  TNeuronGroup = object(TNeuronGroupBase)

    // FIRST NEURAL NETWORK LAYER: testing layer - test are <, >, =, ...
    TestNeuronLayer: TTestsClass;

    // SECOND NEURAL NETWORK LAYER: A+B, A-B, A AND B, ...
    OperationNeuronLayer: TOperation;

    procedure Clear;

    // this function is useful if you need to index neuron groups
    function GetUniqueString: string;

    // this function returns a string that a human understand.
    function GetHumanReadableString: string;

    // creates a string for storage
    function ToString(): string;

    // returns true if there is a valid group
    function Filled(): boolean;

    // loads the object data from string
    procedure LoadFromString(str: string);
  end;

  // These are arrays of NEURONS.
  TNeuralNetwork = array of TNeuronGroup;

  // TStatePredictionClass
  // This class is one of the most important classes in all project.
  // This class is capable of predicting the next state of an array of bytes.
  TStatePredictionClass = object
  public
    {
     This function creates the neural network that will later be used for prediction.
     * The "number of searches" means the number of random attempts will be made in a given point in time.
     * "number of searches" increases CPU time while "relation table size" increases memory.
     * "zeros included" makes the system to use "zero" as an imput parameter. It makes the prediction
       slower but more capable.
    }
    procedure Init(pZerosIncluded: boolean; operationLayerSize: longint;
      pNumberOfSearches: longint);

    // creates a string for storage
    function NeuronsToString(evaluation: integer): string;

    // Cleans the neural network and add neurons from beginning.
    procedure CleanAndLoadNeuronsFromString(var str: string);

    // Defrags the existing neural network and add neurons at the end.
    procedure DefragAndAddNeuronsFromString(var str: string);

    // This function clears all memory from past learning and all stored relations.
    // This function causes a "reset" in the learning.
    procedure ClearAll;

    // This function clears data used to calculate probability of a give relation of index I.
    procedure RemoveNeuronsAtPos(neuronPos: longint);

    // Same as ClearCountingAtPos but for all relations.
    procedure RemoveAllNeurons;

    // This function updates statistics based on the current state / actions and found states.
    // This function can be seen as a function responsible for learning.
    procedure UpdatePredictionStats
      (
    {input} var PActions: array of byte;
    {input} var pCurrentStates, pFoundStates: array of byte);

    // This function creates new relations given a prediction error.
    procedure CreateNewNeuronsOnError(PActions, pred, PCurrentStates,
      PNextStates: array of byte);

    // This function might be one of the most important in all code.
    // Based on Actions and currents states, next states are predicted. For each
    // predicted state in the output array (PNextStates), the probability is given
    // by pRelationProbability. The relation index used when predicting is stored in
    // pVictoryIndex.
    procedure Prediction
      (
    {input}  PActions, PCurrentStates: array of byte;
    {output} var PNextStates: array of byte;
      var pRelationProbability: array of single;
      var pVictoryIndex: array of longint  { victory position }
      );

    // This function is similar to Prediction but doesn't touch "next states" array.
    // This function calculates the probability of each next state but doesn't calculate the state.
    procedure PredictionProbability
      (
    {input} PActions, PCurrentStates: array of byte;
      var PNextStates: array of byte; {states to be predicted}
    {output}var pRelationProbability: array of single; {probabilities}
      var pVictoryIndex: array of longint  {victory position}
      );

    // define action length and state length
    procedure DefineLens(pActionByteLen, pStateByteLen: word);

    // clear victory countings
    procedure ClearNeuronVictories;

    // update victory countings based on the found state and the predicted state.
    // found state is compared to the predicted state.
    procedure UpdateNeuronVictories
      (var pFoundStates, pPredictedStates: array of byte;
      var pVictoryIndex: array of longint{ winner's position}
      );

    procedure Dump;

    // returns Correct Prediction / Total = Correct Prediction Frequency
    function GetF(neuronPos: longint): single;

    // returns (m+1)/(n+2)
    function GetD(neuronPos: longint): single;

    // this function returns a string from the index passed as parameter.
    function NeuronToHumanReadableString(pos: longint): string;

    function NeuronToUniqueString(pos: longint): string;

    function getVictoriousNeuronsCnt(): integer;

    procedure deleteRarelyUsedNeuros(minPredictions: integer);

    procedure deleteNeverWinningNeurons(minVictories: integer);

    procedure deleteDuplicateNeurons();

    // Defrag all neurons from start to end. Returns the first empty space.
    function Defrag(): integer;
  public
    FGeneralize: boolean;
    FUseBelief: boolean;
    FCS: TCreateOperationSettings;

  private
    FNN: TNeuralNetwork;
    // action and state array byte len
    FActionByteLen, FStateByteLen: word;
    // MAX NEURONS ON SECOND (OPERATION) NEURAL NETWORK LAYER
    FMaxOperationNeuronCount: longint;
    // true = creates operation/neurons for all entries - 0 or non 0.
    FZerosIncluded: boolean;
    FNumSearch: longint;
    FCycle: longint;
    FFastPrediction: boolean;
    SelectedIndexes: TCountings;      //best prediction indexes
    NumberOfSelectedIndexes: longint; //number of best prediction indexes
    NumMinSelectedIndexes: longint;   //minimun usage number of best selected indexes

    // loads the object data from string
    procedure AddNeuronsFromStringFromPos(var str: string; pos: integer);

    // This function returns the worst neuron index.
    function GetWorstNeuronIndex(var posWorst: longint;
      searchedNeuronsCnt: longint): extended;

    // Returns the neuron group score. Better neurons survive.
    function GetNeuronGroupScore(neuronPos: longint): extended;

    function EvalNeuronGroup(NG: TNeuronGroup; var ABF: TRunOperation): byte;

    function GetBestNeuronIndex
      (var posBest: longint; Num: longint;
      var ABF: TCreateValidOperations): boolean;

    // This function copies neurons from position Source to position Dest.
    procedure CopyNeurons(SourceRelationPos, DestRelationPos: longint);

    // This function returns the probability to win of a given neuron from position pos.
    function ProbToWin(neuronPos: longint): extended;

    // This function returns all relation indexes with a minimun number of victories (selections)
    // and a minimum probability MinF.
    procedure SelectBestIndexes(MinimumNumberOfVictories: longint; MinF: extended);
    procedure ResumeSlowPrediction;

  end;

  TLabeledState = object
  public
    FLabel: byte;
    FTester: TRunOperation;
  end;

  { TClassifier }

  TClassifier = object (TStatePredictionClass)
  private
    FStates: array of TLabeledState;
    FNextFreePos: integer;
    FNumClasses: integer;
    FRandomProbability: single;
  public
    procedure AddClassifier(NumClasses, NumStates: integer);
    function AddState(pLabel: integer; pState: array of byte): integer;
    function EvolveNeuronGroup(var NG: TNeuronGroup): integer;
    function EvolveNeuronGroupAtPos(neuronpos: integer): integer;
    function NeuronGroupFitness(var NG: TNeuronGroup): single;
    function MutateNeuronGroup(NG: TNeuronGroup): TNeuronGroup;
    procedure CreateRandomNeuronGroup(neuronpos, pClass: integer);
    function PredictClass(PActions: array of byte): byte;
  end;

type
  // This class is an AMAZING class. This class is capable of learning and predicting
  // next states with easy to use methods. Use this class on your projects
  TEasyLearnAndPredictClass = object
  private
    aActions, aCurrentState, aPredictedState: array of byte;
    FRelationProbability: array of single;
    FVictoryIndex: array of longint;
    // array with indexes of neurons that predicted each output state
    FCached: boolean;
    FUseCache: boolean;

  public
    BytePred: TClassifier;
    FCache: TCacheMem;

    // Defines size of neural network, size of input, size of output and flags.
    procedure Initiate
      (pActionByteLen {action array size in bytes}, pStateByteLen
    {state array size in bytes}: word; pIncludeZeros: boolean;
    // false = creates operation/neurons for non zero entries only.
      NumNeuronGroups: integer;
    // number of combinatorial NEURONS. If you don't know how many to crete, give 200.
      pNumberOfSearches: integer;
    // the higher the number, more computations are used on each step. If you don't know what number to use, give 40.
      pUseCache: boolean;
    // replies the same prediction for the same given state. Use false if you aren't sure.
      CacheSize: integer = 1000
    );

    /// Frees memory
    procedure DeInitiate();

    /// THIS METHOD WILL PREDICT THE NEXT SATE GIVEN AN ARRAY OF ACTIONS AND STATES.
    // You can understand ACTIONS as a kind of "current state".
    // Returned value "predicted states" contains the neural network prediction.
    procedure Predict(var pActions, pCurrentState: array of byte;
      var pPredictedState: array of byte);

    // Call this method to train the neural network so it can learn from the "found state".
    // Call this method and when the state of your environment changes so the neural
    // network can learn how the state changes from time to time.
    function newStateFound(stateFound: array of byte): extended;

    // This function updates predition values without creating or destroying neurons
    procedure UpdatePredictionStats(stateFound: array of byte);

    // Prints the neural network as "relations" that map cause into effect.
    // It could also be said that it maps current state into predicted state.
    procedure printRelationTable;

    // returns m/n = frequency
    function GetF(posPredictedState: longint): single;

    // returns (m+1)/(n+2) = confidence.
    function GetD(posPredictedState: longint): single;
  end;


implementation

uses neuralab, SysUtils, Classes;


{ TClassifier }

procedure TClassifier.AddClassifier(NumClasses, NumStates: integer);
begin
  FNextFreePos := 0;
  FNumClasses := NumClasses;
  FRandomProbability := 1/FNumClasses;
  SetLength(FStates, NumStates);
end;

function TClassifier.AddState(pLabel: integer; pState: array of byte): integer;
var
  CurrentState, NextState: array[0..0] of byte;
begin
  CurrentState[0] := 0;
  NextState[0] := pLabel;
  FStates[FNextFreePos].FTester.Load(FCS, pState, CurrentState, NextState);

  Inc(FNextFreePos);
end;

function TClassifier.EvolveNeuronGroup(var NG: TNeuronGroup): integer;
var
  Mutaded: TNeuronGroup;
  BaseScore, NewScore: single;
  MutationCount: integer;
begin
  Result := -1;
  BaseScore := NG.GetF();
  if NG.CorrectNeuronPredictionCnt < FCS.MinSampleForPrediction then BaseScore := 0;
  //Write(' Start:', NG.CorrectNeuronPredictionCnt,'x',BaseScore:6:4, ' Size:', NG.TestNeuronLayer.N);
  for MutationCount := 1 to 10 do
  begin
    Mutaded := Self.MutateNeuronGroup(NG);
    NewScore := Self.NeuronGroupFitness(Mutaded);

    if (NewScore > BaseScore) and (Mutaded.CorrectNeuronPredictionCnt > FCS.MinSampleForPrediction) then
    begin
      NG := Mutaded;
      Result := MutationCount;
      //WriteLn(' Better: ', Mutaded.CorrectNeuronPredictionCnt,'x',
        //NewScore:6:4, ' Exit@: ', MutationCount);
      BaseScore := NewScore;
      if (MutationCount>0) then exit;
    end;
  end;
  //WriteLn(' Not Found!!!');
end;

function TClassifier.EvolveNeuronGroupAtPos(neuronpos: integer): integer;
begin
  Result := EvolveNeuronGroup(FNN[neuronpos]);
end;

function TClassifier.NeuronGroupFitness(var NG: TNeuronGroup): single;
var
  StateCount: integer;
  SuccessCnt, TotalCnt: integer;
  Effect: byte;
  StatePos: integer;
begin
  Result := 0;
  SuccessCnt := 0;
  TotalCnt := 0;
  if (FNextFreePos>0) then
  begin
    for StateCount := 0 to 1000 do
    begin
      StatePos := Random(FNextFreePos);
      if FStates[StatePos].FTester.TestTests(NG.TestNeuronLayer) > 0 then
      begin
        Inc(TotalCnt);
        if FStates[StatePos].FTester.OperateAndTestOperation(NG.OperationNeuronLayer, NG.PredictionPos, Effect) > 0
          then Inc(SuccessCnt);
      end;

      if (SuccessCnt>100) then break;
    end;
  end;

  if (TotalCnt > {(FNextFreePos/FMaxOperationNeuronCount)}FCS.MinSampleForPrediction ) then
  begin
    NG.CorrectNeuronPredictionCnt := SuccessCnt;
    NG.WrongNeuronPredictionCnt := TotalCnt - SuccessCnt;
    NG.Vitories := 0;
    NG.CorrectPredictionAtWin := 0;
    NG.WrongPredictionAtWin := 0;
    Result := SuccessCnt/TotalCnt;
  end;
end;

function TClassifier.MutateNeuronGroup(NG: TNeuronGroup): TNeuronGroup;
var
  MutationType: byte;
begin
  MutationType := random(3);
  //0: delete
  //1: add
  //2: modify (both)

  if (NG.TestNeuronLayer.N = 0) then
  begin
    MutationType := 1;
  end;

  if ((NG.TestNeuronLayer.N < FCS.MinTests) or (NG.TestNeuronLayer.N = 1)) and (MutationType = 0) then
  begin
    MutationType := 2;
  end;

  if (NG.TestNeuronLayer.N = csMaxTests ) and (MutationType = 1) then
  begin
    MutationType := 0;
  end;

  if MutationType = 0 then
  begin
    NG.TestNeuronLayer.DeleteOperation(Random(NG.TestNeuronLayer.N));
  end
  else if MutationType = 1 then
  begin
    NG.TestNeuronLayer.AddTest( FStates[0].FTester.CreateActionRandomBinaryTest() );
  end else
  begin
    NG.TestNeuronLayer.DeleteOperation(Random(NG.TestNeuronLayer.N));
    NG.TestNeuronLayer.AddTest( FStates[0].FTester.CreateActionRandomBinaryTest() );
  end;

  if (NG.TestNeuronLayer.N > 10) then
  begin
    NG.TestNeuronLayer.TestThreshold :=
    NG.TestNeuronLayer.N - Random(NG.TestNeuronLayer.N div 10);
  end
  else
  begin
    NG.TestNeuronLayer.TestThreshold := NG.TestNeuronLayer.N;
  end;

  Result := NG;
end;

procedure TClassifier.CreateRandomNeuronGroup(neuronpos, pClass: integer
  );
begin
  Self.RemoveNeuronsAtPos(neuronpos);
  FNN[neuronpos].TestNeuronLayer.AddTest( FStates[0].FTester.CreateActionRandomBinaryTest() );
  FNN[neuronpos].TestNeuronLayer.TestThreshold := FNN[neuronpos].TestNeuronLayer.N;
  FNN[neuronpos].OperationNeuronLayer :=
    CreateOperation(csSet, pClass, 0, False, False, False);
end;

function TClassifier.PredictClass(PActions: array of byte): byte;
var
  I: word;
  ABF: TRunOperation;
  Probability, Best: single;
  NextState: byte;
  PredictionPosition: integer;
  PCurrentStates, PNextStates: array[0..0] of byte;
  PossibleStates: array of single;
begin
  PCurrentStates[0] := 0;
  PNextStates[0] := 0;
  SetLength(PossibleStates, FNumClasses);

  ABF.Load(FCS, PActions, PCurrentStates, PNextStates);

  for i := Low(PossibleStates) to High(PossibleStates) do
  begin
    PossibleStates[I] := 0;
  end;

  for I := 0 to FMaxOperationNeuronCount - 1 do
  begin
    Probability := FNN[I].GetF();
    PredictionPosition := 0;
    if (FNN[I].Filled) and
      (Probability > 0.1) and
      (FNN[I].CorrectNeuronPredictionCnt > 10) then
    begin
      if ( ABF.TestTests(FNN[I].TestNeuronLayer) > 0) then
      begin
        ABF.OperateAndTestOperation(
          FNN[I].OperationNeuronLayer, PredictionPosition, NextState);
        PossibleStates[NextState] := (Probability - FRandomProbability) + PossibleStates[NextState];
      end;// of if
    end;
  end;// of for

  Best := 0;
  Result := 0;
  for i := 0 to 9 do
  begin
    //Write(' ',PossibleStates[I]:6:4 );
    if ( PossibleStates[I] > Best ) then
    begin
      Best := PossibleStates[I];
      Result := I;
    end;
  end;
  SetLength(PossibleStates,0);
  //WriteLn(' Best:', Result);
end;// of procedure PredProb

{ TNeuronGroup }

function TNeuronGroupBase.GetF: single;
var
  T: longint;
begin
  T := WrongNeuronPredictionCnt + CorrectNeuronPredictionCnt;
  if T > 0 then
    GetF := CorrectNeuronPredictionCnt / T
  else
    GetF := 0;
end;

function TNeuronGroupBase.GetD: single;
var
  T: longint;
begin
  T := WrongNeuronPredictionCnt + CorrectNeuronPredictionCnt;
  GetD := (CorrectNeuronPredictionCnt / (T + 2));
end;

function TNeuronGroupBase.ToString: string;
begin
  Result :=
    IntToStr(PredictionPos) + '>' +
    IntToStr(Vitories) + '>' +
    IntToStr(LastVictory) + '>' +
    IntToStr(WrongNeuronPredictionCnt) + '>' +
    IntToStr(CorrectNeuronPredictionCnt) + '>' +
    IntToStr(WrongPredictionAtWin) + '>' +
    IntToStr(CorrectPredictionAtWin);
end;

procedure TNeuronGroupBase.Clear;
begin
  WrongNeuronPredictionCnt := 0;
  CorrectNeuronPredictionCnt := 0;
  WrongPredictionAtWin := 0;
  CorrectPredictionAtWin := 0;
  Vitories := 0;
  LastVictory := 0;
end;

procedure TNeuronGroup.Clear;
begin
  TestNeuronLayer.N := 0;
  OperationNeuronLayer.OpCode := 0;
  inherited Clear;
end;

function TNeuronGroup.GetUniqueString: string;
begin
  if (TestNeuronLayer.N > 0) then
  begin
    GetUniqueString :=
      TestNeuronLayer.ToHumanString() + OperationNeuronLayer.ToHumanString();
  end
  else
    GetUniqueString := 'nil';
end;

function TNeuronGroup.GetHumanReadableString: string;
var
  R: string;
  f, m, n: extended;
begin
  if (TestNeuronLayer.N > 0) then
  begin
    n := WrongNeuronPredictionCnt;
    n := n + CorrectNeuronPredictionCnt;
    m := CorrectNeuronPredictionCnt;
    if n > 0 then
      f := m / n
    else
      f := 0;
    R := TestNeuronLayer.ToHumanString() + ' => fE[B] := ' +
      OperationNeuronLayer.ToHumanString() + ' [ f=' + FloatToStr(f) +
      ' Vit=' + IntToStr(Vitories) + ' UltVit=' + IntToStr(LastVictory) +
      ' n=' + floatToStr(n) + ' ]';
  end
  else
    R := 'nil';
  GetHumanReadableString := R;
end;

function TNeuronGroup.ToString: string;
begin
  Result :=
    inherited ToString() + '>' +
    TestNeuronLayer.ToString() + '>' +
    OperationNeuronLayer.ToString();
end;

function TNeuronGroup.Filled: boolean;
begin
  Result := (Self.TestNeuronLayer.N > 0) ;
end;

procedure TNeuronGroup.LoadFromString(str: string);
var
  S: TStringList;
begin
  if (Length(str) > 0) then
  begin
    S := TStringList.Create;
    S.Sorted := false;
    S.Delimiter := '>';
    S.StrictDelimiter := true;
    S.DelimitedText := str;

    Self.PredictionPos := StrToInt(S[0]);
    Self.Vitories := StrToInt(S[1]);
    Self.LastVictory := StrToInt(S[2]);

    Self.WrongNeuronPredictionCnt := StrToInt(S[3]);
    Self.CorrectNeuronPredictionCnt := StrToInt(S[4]);

    Self.WrongPredictionAtWin := StrToInt(S[5]);
    Self.CorrectPredictionAtWin := StrToInt(S[6]);

    Self.TestNeuronLayer.LoadFromString(S[7]);
    Self.OperationNeuronLayer.LoadFromString(S[8]);
  end;
end;

procedure TEasyLearnAndPredictClass.printRelationTable;
begin
  BytePred.Dump;
end;

procedure TEasyLearnAndPredictClass.Initiate(pActionByteLen, pStateByteLen: word;
  pIncludeZeros: boolean; NumNeuronGroups: integer; pNumberOfSearches: integer;
  pUseCache: boolean; CacheSize: integer);
begin
  BytePred.Init(pIncludeZeros, NumNeuronGroups, pNumberOfSearches);
  BytePred.DefineLens(pActionByteLen, pStateByteLen);
  SetLength(aActions, pActionByteLen);
  SetLength(aCurrentState, pStateByteLen);
  SetLength(aPredictedState, pStateByteLen);
  SetLength(FRelationProbability, pStateByteLen);
  SetLength(FVictoryIndex, pStateByteLen);
  if (pUseCache)
  then FCache.Init(pActionByteLen, pStateByteLen, CacheSize)
  else FCache.Init(1, 1, 1);
  FUseCache := pUseCache;
end;

procedure TEasyLearnAndPredictClass.DeInitiate();
begin
  FCache.DeInit;
end;

procedure TEasyLearnAndPredictClass.Predict(var pActions, pCurrentState: array of byte;
  var pPredictedState: array of byte);
var
  idxCache: longint;
  Equal: boolean;
begin
  ABCopy(aActions, pActions);
  ABCopy(aCurrentState, pCurrentState);
  if FUseCache then
    idxCache := FCache.Read(pActions, pPredictedState);
  Equal := ABCmp(pActions, pCurrentState);
  if FUseCache and (idxCache <> -1) and Equal then
  begin
    FCached := True;
  end
  else
  begin
    BytePred.Prediction(aActions, aCurrentState, pPredictedState,
      FRelationProbability, FVictoryIndex);
    FCached := False;
  end;
  ABCopy(aPredictedState, pPredictedState);
end;

function TEasyLearnAndPredictClass.newStateFound(stateFound: array of byte): extended;

  procedure PNormalProcedure;
  var
    J: longint;
  begin
    BytePred.UpdatePredictionStats(aActions, aCurrentState, stateFound);
    if not (FCached) then
      BytePred.UpdateNeuronVictories(stateFound{current}, aPredictedState{predicted},
        FVictoryIndex);
    // compares predicted state with found state.
    if ABCountDif(aPredictedState, stateFound) <> 0
    then
    begin
      for J := 1 to 1 do
        BytePred.CreateNewNeuronsOnError(aActions, aPredictedState,
          aCurrentState, stateFound);
    end;
  end;

var
  predictionError: extended;
begin
  predictionError := ABCountDif(stateFound, aPredictedState);
  // Do we have a cached prediction and was the prediction correct?
  if FCached then
  begin
    if predictionError <> 0 // was the prediction wrong?
    then
    begin
      PNormalProcedure; // forgets the cache and recalculate
    end;
  end
  else
    PNormalProcedure;
  newStateFound := predictionError;
  if FUseCache and (predictionError=0) then
    FCache.Include(aActions, stateFound);
end;

procedure TEasyLearnAndPredictClass.UpdatePredictionStats(
  stateFound: array of byte);
begin
  BytePred.UpdatePredictionStats(aActions, aCurrentState, stateFound);
end;

function TEasyLearnAndPredictClass.GetF(posPredictedState: longint): single;
begin
  if FVictoryIndex[posPredictedState] <> -1 then
    GetF := BytePred.GetF(FVictoryIndex[posPredictedState])
  else
    GetF := 0;
end;

function TEasyLearnAndPredictClass.GetD(posPredictedState: longint): single;
begin
  if FVictoryIndex[posPredictedState] <> -1 then
    GetD := BytePred.GetD(FVictoryIndex[posPredictedState])
  else
    GetD := 0;
end;

{ **************** TStatePredictionClass ***************** }
procedure TStatePredictionClass.Init(pZerosIncluded: boolean;
  operationLayerSize: longint; pNumberOfSearches: longint);
begin
  FZerosIncluded := pZerosIncluded;
  SetLength(SelectedIndexes, operationLayerSize);
  SetLength(FNN, operationLayerSize);
  NumberOfSelectedIndexes := 0;
  FNumSearch := pNumberOfSearches;
  ClearAll;
  FGeneralize := False;
  FUseBelief := False;
  NumMinSelectedIndexes := 10;
  FFastPrediction := False;
  FCS := csCreateOpDefault;
end;

function TStatePredictionClass.NeuronsToString(evaluation: integer): string;
var
  S: TStringList;
  neuronPos: longint;
  version: integer;
begin
  version := 1;
  S := TStringList.Create;
  S.Sorted := false;
  S.Delimiter := chr(10);
  S.StrictDelimiter := true;

  S.Add( IntToStr(version) );
  S.Add( IntToStr(evaluation) );
  S.Add( IntToStr(FActionByteLen) );
  S.Add( IntToStr(FStateByteLen)  );

  for neuronPos := Low(FNN) to High(FNN) do
  begin
    if ( FNN[neuronPos].Filled() ) then
    begin
      S.Add( FNN[neuronPos].ToString() );
    end;
  end;

  Result := S.DelimitedText;
  S.Free;
end;

procedure TStatePredictionClass.CleanAndLoadNeuronsFromString(var str: string);
begin
  Self.ClearAll;
  Self.AddNeuronsFromStringFromPos(str, 0);
end;

procedure TStatePredictionClass.DefragAndAddNeuronsFromString(var str: string);
var
  neuronPos: integer;
begin
  neuronPos := Self.Defrag();
  Self.AddNeuronsFromStringFromPos(str, neuronPos);
end;

procedure TStatePredictionClass.AddNeuronsFromStringFromPos(var str: string; pos: integer);
var
  S: TStringList;
  neuronPos: longint;
  version: integer;
  evaluation: integer;
  pActionLen: integer;
  pStatelen: integer;
  inputNeuronCnt: integer;
begin
  version := 1;
  S := TStringList.Create;
  S.Sorted := false;
  S.Delimiter := chr(10);
  S.StrictDelimiter := true;
  S.DelimitedText := str;

  version := StrToInt(S[0]);
  evaluation := StrToInt(S[1]);
  pActionLen := StrToInt(S[2]);
  pStatelen := StrToInt(S[3]);

  //TODO: treat above info here.

  if (S.Count>4) then
  begin
    neuronPos := pos;
    for inputNeuronCnt := 4 to S.Count-1 do
    begin
      FNN[neuronPos].LoadFromString(S[inputNeuronCnt]);
      inc(neuronPos);
    end;
  end;
end;

procedure TStatePredictionClass.Dump;
var
  I: longint;
begin
  for I := Low(FNN) to High(FNN) do
  begin
    if (FNN[I].Filled()) then
    begin
      writeln(NeuronToHumanReadableString(I));
    end;
  end;
end;

function TStatePredictionClass.NeuronToHumanReadableString(pos: longint): string;
var
  R: string;
begin
  if (pos >= 0) then
    R := FNN[pos].GetHumanReadableString
  else
    R := 'nil';
  NeuronToHumanReadableString := R;
end;

function TStatePredictionClass.NeuronToUniqueString(pos: longint): string;
begin
  if (pos >= 0) then
    NeuronToUniqueString := FNN[pos].GetUniqueString
  else
    NeuronToUniqueString := 'nil';
end;

procedure TStatePredictionClass.ClearAll;
var
  I: longint;
begin
  FCycle := 0;
  for I := Low(FNN) to High(FNN) do
  begin
    FNN[I].Clear;
  end;
  FMaxOperationNeuronCount := Length(FNN);
end;

procedure TStatePredictionClass.RemoveNeuronsAtPos(neuronPos: longint);
begin
  FNN[neuronPos].Clear;
end;

procedure TStatePredictionClass.RemoveAllNeurons;
var
  neuronPos: longint;
begin
  for neuronPos := 0 to FMaxOperationNeuronCount do
    RemoveNeuronsAtPos(neuronPos);
end;

procedure TStatePredictionClass.UpdatePredictionStats(var PActions: array of byte;
  var pCurrentStates, pFoundStates: array of byte);
var
  neuronPos: longint;
  ABF: TRunOperation;
  PredictedState: byte;
begin
  ABF.Load(FCS, PActions, pCurrentStates, pFoundStates);
  for neuronPos := 0 to FMaxOperationNeuronCount - 1 do
  begin
    if ABF.TestTests(FNN[neuronPos].TestNeuronLayer) > 0 then
    begin
      if ByteToBool[ABF.OperateAndTestOperation(
        FNN[neuronPos].OperationNeuronLayer, FNN[neuronPos].PredictionPos,
        PredictedState)] then
        Inc(FNN[neuronPos].CorrectNeuronPredictionCnt)
      else
        Inc(FNN[neuronPos].WrongNeuronPredictionCnt);
    end;
  end;
end;

function TStatePredictionClass.EvalNeuronGroup(NG: TNeuronGroup;
  var ABF: TRunOperation): byte;
var
  Effect: byte;
begin
  EvalNeuronGroup := ABF.TestTests(NG.TestNeuronLayer) and
    ABF.OperateAndTestOperation(NG.OperationNeuronLayer,
    NG.PredictionPos, Effect);
end;

function TStatePredictionClass.GetNeuronGroupScore(neuronPos: longint): extended;
var
  R: extended;
begin
  if (FNN[neuronPos].TestNeuronLayer.N = 0) then
    R := -100
  else
    R := FNN[neuronPos].Vitories;
    //R := FNN[neuronPos].GetF();
  Result := R;
end;

// Searches for the worst neuron;  1 - empty neuronal position,
//                                 2 - smallest number of victories
function TStatePredictionClass.GetWorstNeuronIndex(var posWorst: longint;
  searchedNeuronsCnt: longint): extended;
var
  I, neuronPos: longint;
  Actual, worst: extended;
begin
  posWorst := random(FMaxOperationNeuronCount);
  worst := GetNeuronGroupScore(posWorst);
  for I := 1 to searchedNeuronsCnt do
  begin
    neuronPos := random(FMaxOperationNeuronCount);
    Actual := GetNeuronGroupScore(neuronPos);
    if Actual < worst then
    begin
      worst := actual;
      posWorst := neuronPos;
    end;
  end;
  GetWorstNeuronIndex := Worst;
end;

function TStatePredictionClass.ProbToWin(neuronPos: longint): extended;
var
  TotalCount, ResultingProbability: extended;
begin
  ResultingProbability := 0;
  TotalCount := FNN[neuronPos].WrongPredictionAtWin +
    FNN[neuronPos].CorrectPredictionAtWin;
  if (TotalCount > 0) then
    ResultingProbability := FNN[neuronPos].CorrectPredictionAtWin / TotalCount
  else
  begin
    TotalCount := FNN[neuronPos].WrongNeuronPredictionCnt +
      FNN[neuronPos].CorrectNeuronPredictionCnt;
    if TotalCount > 0 then
      ResultingProbability := (FNN[neuronPos].CorrectNeuronPredictionCnt /
        TotalCount) / 100;
  end;
  ProbToWin := ResultingProbability;
end;

function TStatePredictionClass.GetBestNeuronIndex(var posBest: longint;
  Num: longint; var ABF: TCreateValidOperations): boolean;
var
  I, neuronPos: longint;
  Actual, Best: extended;
  R: boolean;
begin
  R := False;
  Result := R;
  posBest := 0;
  Best := 0;
  for I := 1 to Num do
  begin
    neuronPos := random(FMaxOperationNeuronCount);
    if ByteToBool[EvalNeuronGroup(FNN[neuronPos], ABF)] and
      FNN[neuronPos].Filled() and (GetD(neuronPos) > 0.8) then
      Actual := 1
    else
      Actual := 0;

    if (Actual > Best) then
    begin
      Best := actual;
      posBest := neuronPos;
      R := True;
      Result := R;
      exit;
    end;
  end;
end;

procedure TStatePredictionClass.CopyNeurons(SourceRelationPos, DestRelationPos: longint);
begin
  RemoveNeuronsAtPos(DestRelationPos);
  FNN[DestRelationPos].TestNeuronLayer := FNN[SourceRelationPos].TestNeuronLayer;
  FNN[DestRelationPos].OperationNeuronLayer :=
    FNN[SourceRelationPos].OperationNeuronLayer;
  FNN[DestRelationPos].PredictionPos := FNN[SourceRelationPos].PredictionPos;
end;

procedure TStatePredictionClass.CreateNewNeuronsOnError(
  PActions, pred, PCurrentStates, PNextStates: array of byte);

  procedure CreateNeuronGroup(neuronPos, PredictedBytePos: longint;
  var NewOp: TCreateValidOperations);
  var
    IV, NumV: word;
    OP: neuralabfun.TOperation;
  begin
    NewOp.Create(False{no tests}, FZerosIncluded, pred{prediction errors});

    if NewOp.GetCount = 0 then
      writeln(' ERROR: relation creation has failed.')
    else
    begin
      RemoveNeuronsAtPos(neuronPos);
      FNN[neuronPos].OperationNeuronLayer := NewOp.GetRandomOp;

      NumV := FCS.MinTests+random(FCS.MaxTests-FCS.MinTests)+1;

      FNN[neuronPos].TestNeuronLayer.N := NumV;
      NewOp.Create(True{with tests}, FZerosIncluded, pred{prediction errorss});
      for IV := 0 to NumV - 1 do
      begin
        OP := NewOp.GetRandomOp;
        if (OP.OpCode = 0) then
        begin
          FNN[neuronPos].TestNeuronLayer.N := IV;
          break;
        end;
        FNN[neuronPos].TestNeuronLayer.T[IV] := OP;
      end;

      if (FCS.PartialTestEval)
        then FNN[neuronPos].TestNeuronLayer.TestThreshold := random(FNN[neuronPos].TestNeuronLayer.N)+1
        else FNN[neuronPos].TestNeuronLayer.TestThreshold := FNN[neuronPos].TestNeuronLayer.N;
    end;
  end;

var
  SourceI, WorstNeuronGrPos: longint;
  PredictedBytePos: byte;
  NewOp: TCreateValidOperations;
begin
  WorstNeuronGrPos := 0;
  SourceI := 0;
  GetWorstNeuronIndex(WorstNeuronGrPos, FNumSearch);
  RemoveNeuronsAtPos(WorstNeuronGrPos);
  ABGetDif(Pred, Pred, PNextStates);
  PredictedBytePos := ABGetNext1(pred, random(Length(pred)));

  NewOp.Load(FCS, PActions, PCurrentStates, PNextStates);
  NewOp.LoadCreationData(PredictedBytePos);

  // select a predicted byte that was wrongly predicted.
  FNN[WorstNeuronGrPos].PredictionPos := PredictedBytePos;

  if FCS.Bidimensional
    then FNN[WorstNeuronGrPos].TestNeuronLayer.TestBasePosition :=
      NewOp.GetFeatureCenter2D()
    else FNN[WorstNeuronGrPos].TestNeuronLayer.TestBasePosition :=
      PredictedBytePos;

  if (Random(100) < 5) and FGeneralize and         // should copy the neuron ?
    GetBestNeuronIndex(SourceI, FNumSearch, NewOp) and
    (SourceI <> WorstNeuronGrPos) then
  begin
    CopyNeurons(SourceI, WorstNeuronGrPos);
    FNN[WorstNeuronGrPos].TestNeuronLayer.DeleteOperation(Random(
      FNN[WorstNeuronGrPos].TestNeuronLayer.N));
  end
  else
  begin
    CreateNeuronGroup(WorstNeuronGrPos, PredictedBytePos, NewOp);
  end;
end; // of CreateNewRelationOnError

procedure TStatePredictionClass.ClearNeuronVictories;
var
  I: longint;
begin
  for I := 0 to FMaxOperationNeuronCount - 1 do
    FNN[I].Vitories := 0;
end;

function TStatePredictionClass.GetF(neuronPos: longint): single;
begin
  GetF := FNN[neuronPos].GetF;
end;

function TStatePredictionClass.GetD(neuronPos: longint): single;
begin
  GetD := FNN[neuronPos].GetD;
end;

procedure TStatePredictionClass.deleteRarelyUsedNeuros(minPredictions: integer);
var
  neuronPos: integer;
begin
  for neuronPos := Low(FNN) to High(FNN) do
  begin
    if (FNN[neuronPos].CorrectNeuronPredictionCnt) < minPredictions then
      RemoveNeuronsAtPos(neuronPos);
  end;
end;

procedure TStatePredictionClass.deleteNeverWinningNeurons(minVictories: integer);
var
  neuronPos: integer;
begin
  for neuronPos := Low(FNN) to High(FNN) do
  begin
    if (FNN[neuronPos].Vitories) < minVictories then
      RemoveNeuronsAtPos(neuronPos);
  end;
end;

procedure TStatePredictionClass.deleteDuplicateNeurons;
var
  neuronPos: integer;
  NeuronList: TStringList;
  NeuronStr: string;
begin
  NeuronList := TStringList.Create();
  NeuronList.Sorted := True;
  {$IFDEF FPC}
  NeuronList.SortStyle := sslAuto;
  {$ENDIF}
  for neuronPos := Low(FNN) to High(FNN) do
  begin

    if FNN[neuronPos].Filled then
    begin
      NeuronStr := NeuronToUniqueString(neuronPos);
      if NeuronList.IndexOf(NeuronStr) <> -1 then
        RemoveNeuronsAtPos(neuronPos)
      else
        NeuronList.Add(NeuronStr);
    end;
  end;
  NeuronList.Free;
end;

function TStatePredictionClass.getVictoriousNeuronsCnt(): integer;
var
  neuronPos: integer;
  victoriousNeuronsCnt: integer;
begin
  victoriousNeuronsCnt := 0;
  for neuronPos := Low(FNN) to High(FNN) do
  begin
    if (FNN[neuronPos].Vitories) > 0 then
      Inc(victoriousNeuronsCnt);
  end;
  getVictoriousNeuronsCnt := victoriousNeuronsCnt;
end;

procedure TStatePredictionClass.Prediction(
  {input} PActions, PCurrentStates: array of byte;
  {output}var PNextStates: array of byte; var pRelationProbability: array of single;
  var pVictoryIndex: array of longint  { index of victorious neuron });

var
  I, J, MaxIndex: word;
  ABF: TRunOperation;
  Probability, TotalCount: single;
  NextState: byte;
  PredictionPos: integer;
begin
  ABCopy(PNextStates, PCurrentStates); // LOOK
  ABF.Load(FCS, PActions, PCurrentStates, PNextStates);

  for i := Low(pRelationProbability) to High(pRelationProbability) do
  begin
    pRelationProbability[I] := 0;
    pVictoryIndex[I] := -1;
  end;

  if FFastPrediction then
    MaxIndex := NumberOfSelectedIndexes - 1
  else
    MaxIndex := FMaxOperationNeuronCount - 1;

  for J := 0 to MaxIndex do
  begin

    if FFastPrediction then
      I := SelectedIndexes[J]
    else
      I := J;

    PredictionPos := FNN[I].PredictionPos;
    TotalCount := FNN[I].WrongNeuronPredictionCnt +
      FNN[I].CorrectNeuronPredictionCnt + 1;

    if (TotalCount = 0) or not(FNN[I].Filled()) then
    begin
      Probability := 0;
    end
    else if FUseBelief then
    begin
      Probability :=
        (FNN[I].CorrectNeuronPredictionCnt + 1) / (TotalCount + 2);
    end
    else
      Probability := FNN[I].CorrectNeuronPredictionCnt / TotalCount;  // best method

    if (Probability > pRelationProbability[PredictionPos]) and
      (FNN[I].CorrectNeuronPredictionCnt > FCS.MinSampleForPrediction)
      then
    begin
      if (ABF.TestTests(FNN[I].TestNeuronLayer) > 0) then
      begin
        ABF.OperateAndTestOperation(FNN[I].OperationNeuronLayer,
          PredictionPos, NextState);
        if (Probability > pRelationProbability[PredictionPos]) then
        begin
          PNextStates[PredictionPos] := NextState;
          pRelationProbability[PredictionPos] := Probability;
          pVictoryIndex[PredictionPos] := I;
        end;
      end;// of if
    end; // of probability if
  end;// of for
end;// of procedure

procedure TStatePredictionClass.PredictionProbability(
  {input} PActions, PCurrentStates: array of byte; var PNextStates: array of byte;
  {efeitos a serem medidos}
  {output}var pRelationProbability: array of single; {probabilidades}
  var pVictoryIndex: array of longint  { posicao do vitorioso});
var
  I: word;
  ABF: TRunOperation;
  Probability, TotalCount: single;
  NextState: byte;
  PredictionPosition: integer;
begin
  ABF.Load(FCS, PActions, PCurrentStates, PNextStates);

  for i := Low(pRelationProbability) to High(pRelationProbability) do
  begin
    pRelationProbability[I] := 0;
    pVictoryIndex[I] := -1;
  end;

  for I := 0 to FMaxOperationNeuronCount - 1 do
  begin
    PredictionPosition := FNN[I].PredictionPos;
    if (FNN[I].Filled) and
      ByteToBool[ABF.OperateAndTestOperation(
      FNN[I].OperationNeuronLayer, PredictionPosition, NextState)] and
      ( ABF.TestTests(FNN[I].TestNeuronLayer) > 0) then
    begin
      TotalCount := FNN[I].WrongNeuronPredictionCnt +
        FNN[I].CorrectNeuronPredictionCnt;
      if TotalCount > 0 then
      begin
        if FUseBelief then
          Probability := (FNN[I].CorrectNeuronPredictionCnt + 1) / (TotalCount + 2)
        else
          Probability := FNN[I].CorrectNeuronPredictionCnt / TotalCount; // best option
      end
      else
        Probability := 0;
      if (Probability > pRelationProbability[PredictionPosition]) then
      begin
        pRelationProbability[PredictionPosition] := Probability;
        pVictoryIndex[PredictionPosition] := I;
      end;
    end;// of if
  end;// of for
end;// of procedure PredProb

procedure TStatePredictionClass.SelectBestIndexes(MinimumNumberOfVictories: longint;
  MinF: extended);
var
  I: longint;
  TotalCount, Probability: extended;
begin
  FFastPrediction := True;
  NumberOfSelectedIndexes := 0;
  for I := Low(FNN) to High(FNN) do
  begin
    TotalCount := FNN[I].WrongNeuronPredictionCnt + FNN[I].CorrectNeuronPredictionCnt;
    if TotalCount > 0 then
    begin
      if FUseBelief then
        Probability := (FNN[I].CorrectNeuronPredictionCnt + 1) / (TotalCount + 2)
      else
        Probability := FNN[I].CorrectNeuronPredictionCnt / TotalCount;  // best option
    end
    else
      Probability := 0;

    if (MinimumNumberOfVictories <= FNN[I].Vitories) and (MinF <= Probability) then
    begin
      SelectedIndexes[NumberOfSelectedIndexes] := I;
      Inc(NumberOfSelectedIndexes);
    end;
  end;
  FFastPrediction := FFastPrediction and (NumberOfSelectedIndexes >=
    NumMinSelectedIndexes);
end;

procedure TStatePredictionClass.ResumeSlowPrediction;
begin
  FFastPrediction := False;
end;

function TStatePredictionClass.Defrag: integer;
var
  NeuroCount: integer;
begin
  Result := 0;
  for NeuroCount := Low(FNN) to High(FNN) do
  begin
    if ( FNN[NeuroCount].Filled() ) then
    begin
      if (NeuroCount>Result) then
      begin
        FNN[Result] := FNN[NeuroCount];
        FNN[NeuroCount].TestNeuronLayer.N := 0;
      end;
      Inc(Result);
    end;
  end;
end;

procedure TStatePredictionClass.UpdateNeuronVictories(
  var pFoundStates, pPredictedStates: array of byte;
  var pVictoryIndex: array of longint { positions of victorious neurons });
var
  I: word;
  IVict: integer;
begin
  Inc(FCycle);
  for I := Low(pFoundStates) to High(pFoundStates) do
  begin
    IVict := pVictoryIndex[I];
    if (IVict <> -1) then
    begin
      if (pFoundStates[I] = pPredictedStates[I]) then
      begin
        Inc(FNN[IVict].Vitories);
        Inc(FNN[IVict].CorrectPredictionAtWin);
        FNN[IVict].LastVictory := FCycle;
      end
      else
      begin
        Dec(FNN[IVict].Vitories);
        Inc(FNN[IVict].WrongPredictionAtWin);
      end;
    end;
  end;
end;

procedure TStatePredictionClass.DefineLens(pActionByteLen, pStateByteLen: word);
begin
  FActionByteLen := pActionByteLen;
  FStateByteLen := pStateByteLen;
end;


end.
