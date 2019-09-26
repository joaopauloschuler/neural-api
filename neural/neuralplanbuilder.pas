(*
neuralplanbuilder: This unit "builds and optimizea plans" towards success with
AI techniques. A plan (TPlan) is a sequence of actions and predicted states.
A composite plan (TCompositePlan) is an array of plans that can work together.
Copyright (C) 2001 Joao Paulo Schwarz Schuler

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
*)

unit neuralplanbuilder; // this unit used to be called UFway44.

{$IFDEF FPC}
{$MODE Delphi}
{$ENDIF}

{ Lazarus/Free Pascal Source }
// made by Joao Paulo Schwarz Schuler
// jpss@schulers.com

interface

uses neuralab;

const
  MaxStates = 400;
  MaxPlans = 100;

type
  TState = array of byte;

type
  TProcPred = function(var ST: array of byte; Action: byte): boolean of object;

type
  {
    This class contains a list of states. FKeyCache is used to speed up by keeping a
    kind of hash of all included states. An ordered list of actions and states is
    actually a PLAN.
  }
  TActionStateList = object
  protected
    FKeyCache: TABHash; // 1 in a given position might mean that the entry is there.

  public
    ListStates: array[0..MaxStates - 1] of TState; // list of stored states.
    ListActions: array[0..MaxStates - 1] of byte;   // list of action on state.

    NumStates: longint; {number of stored states }

    procedure Init(StateLength: longint);

    // Recreates the FKeyCache list.
    procedure ReDoHash;

    // Clear all entries
    procedure Clear;

    // Removes the first state and action.
    procedure RemoveFirst;

    // includes a state and action in the end of the plan.
    procedure Include(ST: array of byte; Action: byte);

    // This function returns -1 when the parameter doesn't exist; else returns position ...
    function Exists(ST: array of byte): longint;

    // This function returns -1 when the parameter doesn't exist; else returns position ...
    // The same as Exists but faster. Depends on valid FKeyCache.
    function FastExists(ST: array of byte): longint;

    // Removes some duplicate states.
    function RemoveCicles: boolean;

    // Removes all duplicate states.
    procedure RemoveAllCicles;

    // Removes all entries from InitPost to FinishPos
    procedure RemoveSubList(InitPos, FinishPos: longint);
  end;

procedure TVisitedStatesCopy(var A, B: TActionStateList);

type
  {
   This class has 2 main goals:
   * This class can be used as a "plan builder" done by methods "Run" and "Multiple Run".
   * This class can be used as a "plan optimizer" done by methods "Optimize From" and "MultipleOptimizeFrom".
   The plan is stored in the FPlan property.
  }
  TPlan = object
  private
    //Auxiliar Plan used in optimization.
    V2: TActionStateList;
    Pred, PredOpt: TProcPred;

    // State number of bytes.
    FStateLength: longint;
    FNumberActions: longint;

  public
    // true means that the object is a plan that leads to success.
    Found: boolean;

    // the plan.
    FPlan: TActionStateList;

    procedure Init(PPred, PPredOpt: TProcPred; PNumberActions, StateLength: longint);

    // This function builds a plan. It returns true if a solution (plan) is found.
    // Deep is the maximum number of steps that a plan can have.
    function Run(InitState: array of byte; deep: longint): boolean;

    // This function runs the "Run" method the "Number" of tries.
    function MultipleRun(InitState: array of byte; deep, Number: longint): boolean;

    // Plan optimization methods.
    function OptimizeFrom(ST, deep: longint): longint;
    procedure MultipleOptimizeFrom(ST, deep, Number: longint);

    // Removes First Element of a Plan.
    procedure RemoveFirst;

    // if state ST is a state in the plan, it returns the index of the next step in the plan.
    function GetNextStep(CurrentState: array of byte): longint;

    // Returns the plan action of position I.
    function Action(I: longint): byte;

    // Invalidate the plan (Means the plan doesn't lead to success).
    procedure Invalidate;

    // Returns the number of states of the plan.
    function NumStates: longint;
  end;

type
  {
   An agent has a composition of plans. These plans can Mix together so the composite
   plan is better than the individual plans. As an example, plan A can be better in its
   first half while plan B is better in its second half. But the composition A+B makes
   a better plan than each individual plan.
  }
  TCompositePlan = object
  private
    // Saved state of "LastUsedPlan".
    SavedLastUsedPlan: longint;

    // Saved state of "LastPlanedPlan".
    SavedLastPlanedPlan: longint;

    // returns the index of the worst plan.
    function ChooseWorst: longint;

    // evaluates a  plan.
    function EvalPlan(I: longint): extended;

    // choose the best plan to be used using the evaluation based on the next step of each plan.
    function ChooseBestPlanBasedOnNextStep(var CurrentState: array of byte): longint;

    // evaluates how good a plan is based on the next step of the plan.
    function EvalPlanBasedOnNextStep(var CurrentState: array of byte;
      PlanIndex: longint): extended;

  public
    // List of plans.
    Plans: array[0..MaxPlans - 1] of TPlan;

    // Most recently used plan (plan from where the last action has been choosen).
    LastUsedPlan: longint;

    // Most recently built plan
    LastPlanedPlan: longint;

    LastReceivedPlan: longint;

    procedure Init(PPred, PPredOpt: TProcPred; PNumberActions, StateLength: longint);

    // chooses the worst plan and replaces by a newly built plan.
    function Run(InitState: array of byte; deep: longint): boolean;

    // chooses the worst plan and replaces by a newly built plan.
    function MultipleRun(InitState: array of byte; deep, Number: longint): boolean;

    function Optimize(ActState: array of byte; deep: longint): longint;
    procedure MultipleOptimize(ActState: array of byte; deep, Number: longint);
    procedure MultipleOptimizeLastUsedPlan(ActState: array of byte; deep, Number: longint);
    procedure ReceivePlan(var P: TActionStateList);

    function ToAct(ST: array of byte; var Action: byte; var LastAct: boolean;
      var FutureS: array of byte): boolean;

    // saves LastUsedPlan and LastPlanedPlan
    procedure SaveState;

    // restores LastUsedPlan and LastPlanedPlan
    procedure RestoreState;

    // forgets the last used plan
    procedure ForgetLastUsedPlan;

    // marks the last used plan as invalid plan.
    procedure InvalidateLastUsedPlan;

    //plan X is added plan Y - unites plans
    procedure CollapsePlans(X, Y: longint); // unite plans.
  end;

implementation

procedure TVisitedStatesCopy(var A, B: TActionStateList);
var
  I: longint;
begin
  A.NumStates := B.NumStates;
  A.ListActions := B.ListActions;
  for I := 0 to B.NumStates - 1 do
    ABCopy(A.ListStates[I], B.ListStates[I]);
  // copy hash table.
  ABCopy(A.FKeyCache.A, B.FKeyCache.A);
end;

procedure TCompositePlan.Init(PPred, PPredOpt: TProcPred;
  PNumberActions, StateLength: longint);
var
  I: longint;
begin
  for I := Low(Plans) to High(Plans) do
    Plans[I].Init(PPred, PPredOpt, PNumberActions, StateLength);
  LastUsedPlan := -1;
  LastPlanedPlan := -1;
  LastReceivedPlan := -1;
end;

procedure TCompositePlan.ForgetLastUsedPlan;
begin
  LastUsedPlan := -1;
end;

procedure TCompositePlan.SaveState;
begin
  SavedLastUsedPlan := LastUsedPlan;
  SavedLastPlanedPlan := LastPlanedPlan;
end;

procedure TCompositePlan.RestoreState;
begin
  LastUsedPlan := SavedLastUsedPlan;
  LastPlanedPlan := SavedLastPlanedPlan;
end;

function TCompositePlan.Run(InitState: array of byte; deep: longint): boolean;
var
  I: longint;
begin
  I := ChooseWorst;
  Run := Plans[I].Run(InitState, deep);
  LastPlanedPlan := I;
end;

procedure TCompositePlan.ReceivePlan(var P: TActionStateList);
var
  I: longint;
begin
  I := ChooseWorst;
  TVisitedStatesCopy(Plans[I].FPlan, P);
  Plans[I].Found := True;
  LastReceivedPlan := I;
end;

function TCompositePlan.Optimize(ActState: array of byte; deep: longint): longint;
begin
  MultipleOptimize(ActState, deep, 1);
  Optimize := 0;
end;

//plan X is added plan Y - unites plans
procedure TCompositePlan.CollapsePlans(X, Y: longint);
var
  I: longint;
begin
  if (Plans[Y].Found) and (Plans[Y].NumStates + Plans[X].NumStates < MaxStates) then
  begin
    for I := 0 to Plans[Y].NumStates - 1 do
    begin
      Plans[X].FPlan.Include(Plans[Y].FPlan.ListStates[I], Plans[Y].Action(I));
    end;
  end;
  Plans[X].FPlan.RemoveAllCicles;
end;

procedure TCompositePlan.MultipleOptimizeLastUsedPlan(ActState: array of byte;
  deep, Number: longint);
var
  ACI: longint;
begin
  if (LastUsedPlan <> -1) then
  begin
    AcI := Plans[LastUsedPlan].GetNextStep(ActState);
    if Aci <> -1 then
      Plans[LastUsedPlan].MultipleOptimizeFrom(AcI + 1, deep, Number)
    else
      Plans[LastUsedPlan].MultipleOptimizeFrom(0, deep, Number);
  end;
end;

procedure TCompositePlan.MultipleOptimize(ActState: array of byte; deep, Number: longint);
var
  I: longint;
var
  ACI: longint;
begin
  for I := Low(Plans) to High(Plans) do
    if Plans[I].Found then
    begin
      if (I <> LastUsedPlan) then
        Plans[I].MultipleOptimizeFrom(0, deep, Number)
      else
      begin
        AcI := Plans[I].GetNextStep(ActState);
        if Aci <> -1 then
          Plans[I].MultipleOptimizeFrom(AcI + 1, deep, Number);
      end;
    end;
end;

function TCompositePlan.MultipleRun(InitState: array of byte;
  deep, Number: longint): boolean;
var
  I: longint;
begin
  I := ChooseWorst;
  MultipleRun := Plans[I].MultipleRun(InitState, deep, Number);
  LastPlanedPlan := I;
end;

procedure TCompositePlan.InvalidateLastUsedPlan;
begin
  if LastUsedPlan <> -1 then
    Plans[LastUsedPlan].Invalidate;
end;

function TCompositePlan.ToAct(ST: array of byte;
  var Action: byte; var LastAct: boolean;
  var FutureS: array of byte): boolean;
var
  I, AcI, C: longint;
begin
  ToAct := False;
  LastAct := False;

  C := ChooseBestPlanBasedOnNextStep(ST);

  for I := Low(Plans) to High(Plans) do
  begin
    AcI := Plans[C].GetNextStep(ST);
    if AcI <> -1 then
    begin
      Action := Plans[C].Action(AcI);
      LastUsedPlan := C;
      ToAct := True;
      LastAct := (Plans[C].FPlan.NumStates - 1 = AcI);
      ABCopy(FutureS, Plans[C].FPlan.ListStates[AcI]);
      exit;
    end;
    C := (C + 1) mod MaxPlans;
  end;
end;

function TCompositePlan.ChooseWorst: longint;
var
  I: longint;
  WorstPlan: longint;
  WorstPlanEvaluation: extended;
  PlanEvaluation: extended;
begin
  WorstPlan := 0;
  WorstPlanEvaluation := -1000000;
  for I := Low(Plans) to High(Plans) do
  begin
    PlanEvaluation := EvalPlan(I);
    if (PlanEvaluation > WorstPlanEvaluation) or
      ((PlanEvaluation = WorstPlanEvaluation) and (Random(2) = 0)) then
    begin
      WorstPlanEvaluation := PlanEvaluation;
      WorstPlan := I;
    end;
  end;
  ChooseWorst := WorstPlan;
end;

function TCompositePlan.ChooseBestPlanBasedOnNextStep(
  var CurrentState: array of byte): longint;
var
  I: longint;
  BestPlan: longint;
  BestPlanEvaluation: extended;
  PlanEvaluation: extended;
begin
  BestPlan := 0;
  BestPlanEvaluation := 1000000;
  for I := Low(Plans) to High(Plans) do
  begin
    PlanEvaluation := EvalPlanBasedOnNextStep(CurrentState, I);
    if (PlanEvaluation < BestPlanEvaluation) or
      ((PlanEvaluation = BestPlanEvaluation) and (Random(2) = 0)) then
    begin
      BestPlanEvaluation := PlanEvaluation;
      BestPlan := I;
    end;
  end;
  ChooseBestPlanBasedOnNextStep := BestPlan;
end;

// the bigger the number, the worse is.
function TCompositePlan.EvalPlan(I: longint): extended;
begin
  if not (Plans[I].Found) then
    EvalPlan := 100000
  else
    EvalPlan := 1 - Plans[I].NumStates;
end;

function TCompositePlan.EvalPlanBasedOnNextStep(
  var CurrentState: array of byte; PlanIndex: longint): extended; // quanto maior, pior
var
  ACI: longint;
begin
  AcI := Plans[PlanIndex].GetNextStep(CurrentState);
  if (AcI = -1) then
    EvalPlanBasedOnNextStep := 100000
  else
    EvalPlanBasedOnNextStep := Plans[PlanIndex].NumStates - AcI;
end;

procedure TActionStateList.Init(StateLength: longint);
var
  I: integer;
begin
  Clear;
  for I := 0 to MaxStates - 1 do
    SetLength(ListStates[I], StateLength);
  FKeyCache.Init(8000);
end;

procedure TActionStateList.Clear;
begin
  NumStates := 0;
  FKeyCache.Clear;
end;

procedure TActionStateList.RemoveFirst;
var
  I: integer;
begin
  FKeyCache.Clear;
  if NumStates > 0 then
  begin
    for I := 1 to NumStates - 1 do
    begin
      ABCopy(ListStates[I - 1], ListStates[I]);
      ListActions[I - 1] := ListActions[I];
      FKeyCache.Include(ListStates[I]);
    end;
    NumStates := NumStates - 1;
  end;
end;

procedure TActionStateList.Include(ST: array of byte; Action: byte);
begin
  FKeyCache.Include(ST);
  if (NumStates < MaxStates - 1) and (NumStates >= 0) then
  begin
    ABCopy(ListStates[NumStates], ST);
    ListActions[NumStates] := Action;
    Inc(NumStates);
  end
  else
  begin
    Writeln('TVisitedStates: Out of Memory ', NumStates, ' ');
    Readln;
    halt;
  end;
  if NumStates >= MaxStates - 5 then
    RemoveCicles;
end;

function TActionStateList.FastExists(ST: array of byte): longint;
begin
  if FKeyCache.Test(ST) then
    FastExists := Exists(ST)
  else
    FastExists := -1;
end;

function TActionStateList.Exists(ST: array of byte): longint;
var
  I: longint;
begin
  Exists := -1;
  for I := NumStates - 1 downto 0 do
  begin
    if ABCmp(ST, ListStates[I]) then
    begin
      Exists := I;
      exit;
    end;
  end;
end;

procedure TActionStateList.ReDoHash;
var
  I: longint;
begin
  FKeyCache.Clear;
  for I := 0 to NumStates - 1 do
    FKeyCache.Include(ListStates[I]);
end;

function TPlan.GetNextStep(CurrentState: array of byte): longint;
var
  R: longint;
begin
  if Found then
  begin
    R := FPlan.FastExists(CurrentState);
    if (R <> FPlan.NumStates - 1) and (R <> -1) then
      GetNextStep := R + 1
    else
      GetNextStep := -1;
  end
  else
    GetNextStep := -1;
end;

function TPlan.Action(I: longint): byte;
begin
  Action := FPlan.ListActions[I];
end;

procedure TPlan.Invalidate;
begin
  Found := False;
end;

function TPlan.NumStates: longint;
begin
  NumStates := FPlan.NumStates;
end;

procedure TActionStateList.RemoveSubList(InitPos, FinishPos: longint);
var
  I, difer: longint;
begin
  difer := FinishPos - InitPos + 1;
  if (FinishPos + 1 <= NumStates - 1) then
    for I := FinishPos + 1 to NumStates - 1 do
    begin
      ABCopy(ListStates[I - Difer], ListStates[I]);
      ListActions[I - Difer] := ListActions[I];
    end;
  NumStates := NumStates - Difer;
  ReDoHash;
end;

function TActionStateList.RemoveCicles: boolean;
var
  I, J: integer;
begin
  RemoveCicles := False;
  (* Write('Removing'); *)
  for I := NumStates - 1 downto 1 do
  begin
    for J := 0 to I - 1 do
    begin
      if ABCmp(ListStates[J], ListStates[I]) then
      begin
        RemoveCicles := True;
        RemoveSubList(J + 1, I);
        (* Writeln('removed:',NumStates,'  '); *)
        exit;
      end;
    end;
  end;
end;

procedure TActionStateList.RemoveAllCicles;
begin
  while RemoveCicles do ;
end;

function TryToBuildSubPath
  (var TVS: TActionStateList;
  var TargetState: array of byte; currentState: array of byte;
  NCicles: longint; PPred: TProcPred; NumberActions: longint): extended;
var
  Cicles: longint;
  Action: byte;
  Prefered: boolean;
  PreferedAct: byte;
begin
  Prefered := (random(2) > 0);
  PreferedAct := random(NumberActions);
  TryToBuildSubPath := 0;
  for Cicles := 1 to NCicles do
  begin
    if Prefered and (random(2) > 0) then
      Action := PreferedAct            // 50% is this action.
    else
      Action := random(NumberActions); // 50% is a random action.
    // produces a new state based on a random action.
    PPred(currentState, Action);
    TVS.Include(currentState, Action);
    if ABCmp(currentState, TargetState) then
    begin
      TryToBuildSubPath := Cicles;
      exit;
    end;
  end;
end; { of procedure TryToBuildSubPath }


// Gera Plano
// TVS: onde vai ser montado o plano
// Estado: Estado Inicial do Planejamento
// NCicles: Numero de ciclos do planejamento (profundidade máxima)
// PPred: Funcao de predicao. Retorna true quando acha satisfacao
// NumberActions: numero de acoes possiveis. O intervalo de acoes
// eh 0.. NumberActions-1
function BuildPlanFn(
  {output} var aNewPlan: TActionStateList;
  {input}  pState: array of byte;
  {input}  planSize: longint;
  {input}  PPred: TProcPred;
  {input}  NumberActions: longint): extended;

  function ChooseRandomAction(var State: array of byte): integer;
  var
    I: integer;
    NewState: TState;
    Action: byte;
  begin
    SetLength(NewState, Length(State));
    for I := 0 to NumberActions - 1 do
    begin
      Action := random(NumberActions);
      ABCopy(NewState, State);
      PPred(NewState, Action);
      if (aNewPlan.FastExists(NewState) = -1) then
      begin
        ChooseRandomAction := Action;
        exit;
      end;
    end;
    ChooseRandomAction := random(NumberActions);
  end;

  {
    This function tries to find a solution in just 1 step.
    Returns -1 if can't find solution in 1 step.
    In the case a solution is found, returns the action id.
  }
  function ChooseActionIn1Step(var State: array of byte): integer;
  var
    I: integer;
    NewState: TState;
  begin
    ChooseActionIn1Step := -1;
    SetLength(NewState, Length(State));
    for I := 0 to NumberActions - 1 do
    begin
      ABCopy(NewState, State);
      if PPred(NewState, I) then
      begin
        ChooseActionIn1Step := I;
        ABCopy(State, NewState);
        exit;
      end;
    end;
  end;

var
  Cicles: longint;
  Action: integer;
  PlanFound: boolean;
  Prefered: boolean;
  PreferedAct: byte;
begin
  Prefered := (random(2) > 0);
  PreferedAct := random(NumberActions);
  BuildPlanFn := 0;
  for Cicles := 1 to planSize do
  begin
    Action := ChooseActionIn1Step(pState);
    // Has found satisfaction in 1 step?
    if Action <> -1 then
    begin
      aNewPlan.Include(pState, Action);
      BuildPlanFn := Cicles;
      exit;
    end
    else
    begin
      // has not found satisfaction in 1 step.
      if Prefered and (random(2) > 0) then
        Action := PreferedAct
      else
        Action := ChooseRandomAction(pState);
      PlanFound := PPred(pState, Action);
      aNewPlan.Include(pState, Action);
      if PlanFound then
      begin
        BuildPlanFn := Cicles;
        exit;
      end;
    end;
  end;
end; { of procedure BuildPlanFn }

procedure TPlan.Init(PPred, PPredOpt: TProcPred; PNumberActions, StateLength: longint);
begin
  Pred := PPred;
  PredOpt := PPredOpt;
  FNumberActions := PNumberActions;
  FStateLength := StateLength;
  FPlan.Init(FStateLength);
  V2.Init(FStateLength);
end;

function TPlan.Run(InitState: array of byte; deep: longint): boolean;
var
  R: extended;
begin
  Found := False;
  FPlan.Clear;
  FPlan.Include(InitState, 0);
  R := BuildPlanFn(FPlan, InitState, deep, Pred, FNumberActions);
  Found := (R <> 0);
  if Found then
    FPlan.RemoveAllCicles;
  Run := Found;
end;

function TPlan.MultipleRun(InitState: array of byte;
  deep, Number: longint): boolean;
var
  I: longint;
begin
  I := 0;
  while (I < Number) and not (Run(InitState, deep)) do
    Inc(I);
  MultipleRun := Found;
end;

procedure TPlan.RemoveFirst;
begin
  FPlan.RemoveFirst;
end;

procedure LExchange(var X, Y: longint);
var
  AUX: longint;
begin
  AUX := X;
  X := Y;
  Y := AUX;
end;

procedure TPlan.MultipleOptimizeFrom(ST, deep, Number: longint);
var
  I: longint;
begin
  I := 0;
  while (OptimizeFrom(ST, deep) = 0) and (I < Number) do
    Inc(I);
end;

function TPlan.OptimizeFrom(ST, deep: longint): longint;
var
  StartPos, FinishPos: longint;
  IniState, FinalState: TState;
  v2pos, FPlanPos, K: longint;
begin
  OptimizeFrom := 0;

  if ST > FPlan.NumStates - 3 then
    exit;

  SetLength(IniState, FStateLength);
  SetLength(FinalState, FStateLength);
  StartPos := random(FPlan.NumStates - ST) + ST;
  FinishPos := random(FPlan.NumStates - ST) + ST;
  if FinishPos < StartPos then
    LExchange(StartPos, FinishPos);

  if abs(StartPos - FinishPos) < 2 then
    exit;

  ABCopy(IniState, FPlan.ListStates[StartPos]);
  V2.Clear;
  V2.Include(IniState, FPlan.ListActions[StartPos]);
  ABCopy(FinalState, FPlan.ListStates[FinishPos]);
  // creates a new subway/subpath
  TryToBuildSubPath(V2, FinalState, IniState, deep, PredOpt, FNumberActions);
  V2.RemoveAllCicles;
  // for each state in the new sub path
  for v2pos := 1 to V2.NumStates - 1 do
  begin
    if ((StartPos + v2pos) < (FPlan.NumStates - 1))
      // is it a state in the existing plan?
      and (FPlan.FastExists(V2.ListStates[v2pos]) <> -1) then
    begin
      // for each of the following states in the current plan FPlan.
      for FPlanPos := StartPos + v2pos to FPlan.NumStates - 1 do
      begin
        // Does the state in V2 exist in FPlan? Has shorter path been found?
        if ABCmp(V2.ListStates[v2pos], FPlan.ListStates[FPlanPos])
        then
        begin  {Copy Shortes Segment from V2 to FPlan}
          for K := 1 to v2pos do
          begin
            ABCopy(FPlan.ListStates[StartPos + K],
              V2.ListStates[K]);
            FPlan.ListActions[StartPos + K] := V2.ListActions[K];
          end;
          FPlan.RemoveSubList(StartPos + v2pos + 1, FPlanPos);
          OptimizeFrom := FPlan.NumStates;
          exit;
        end;
      end;
    end;
  end;
end;
//       if (StartPos+I) <= (V1.NumStates-1)
//          then for J:=StartPos+I to V1.NumStates-1

(* procedure PrintPlan(var P:TPlan);
var I:longint;
var X,Y: TState;
begin
for I:=0 to P.V1.NumStates-1
    do begin
       gotoxy(P.V1.ListStates[I,0],P.V1.ListStates[I,1]);
       if I>0
          then begin
               Y:=P.V1.ListStates[I-1];
               Pred(Y,P.V1.ListActions[I]);
               if not(ABCmp(P.V1.ListStates[I],Y))
                  then writeln('ERRO');
               end;
       Write(P.V1.ListStates[I,2]);
       end;
end; *)

(*var P:TPlan;
var X,Y: TState;
    R:extended;
    I:integer;
begin
randomize;
clrscr;
X[0]:=10; X[1]:=10; X[2]:=0;
Y[0]:=0 ; Y[1]:=0 ; Y[2]:=2;
P.Run(X,Y,100000);
Writeln('Resultado:',P.V1.NumStates);
readkey;

clrscr;
PrintPlan(P);
readkey;
for I:=1 to 20000
    do begin
       clrscr;
       P.Optimize(random(50));
       clrscr;
       PrintPlan(P);
       end;
readkey;
clrscr;
Writeln('Estado inicial:');
Write('  ',P.V1.ListStates[0,0],'  ');
Write('  ',P.V1.ListStates[0,1],'  ');
Writeln('  ',P.V1.ListStates[0,2],'  ');
Writeln('Estado Final:');
Write('  ',P.V1.ListStates[P.V1.NumStates-1,0],'  ');
Write('  ',P.V1.ListStates[P.V1.NumStates-1,1],'  ');
Writeln('  ',P.V1.ListStates[P.V1.NumStates-1,2],'  ');*)

end.
