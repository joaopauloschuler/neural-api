unit neuralrl;
(*
neuralrl: Classical Reinforcement Learning for the CAI NEURAL API.
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by AI.

--------------------------------------------------------------------------
neuralrl provides classical (model-free) reinforcement learning trainers
that operate on TNNet networks. It sits alongside neuraldpo.pas and extends
the same "trainer-level helper" pattern to general MDPs:

  * TNeuralEnv               - abstract environment (state, action, reward, done)
  * TNeuralReplayBuffer      - circular experience replay buffer
  * TNeuralDQNTrainer        - DQN (discrete actions, target net, epsilon-greedy)
  * TNeuralPolicyGradTrainer - REINFORCE with optional value baseline (A2C)
  * TNeuralCartPoleEnv       - built-in CartPole environment (Gym-classic)
  * TNeuralGridWorldEnv      - built-in 2-D stochastic grid world

NETWORK REQUIREMENTS
--------------------------------------------------------------------------
  Q-NET (DQN):    Input(1,1,D) -> ... -> TNNetFullConnectLinear(NumActions)
  POLICY (PG):    Input(1,1,D) -> ... -> TNNetSoftMax(NumActions)
  CRITIC (opt):   Input(1,1,D) -> ... -> TNNetFullConnectLinear(1)

  The caller builds and owns the networks. The trainer clones the Q-net
  for the target network. All updates use the same three TNNet calls:
    NN.Compute(FInput) / NN.Backpropagate(P) / NN.UpdateWeights()

API EXAMPLE (DQN on CartPole)
--------------------------------------------------------------------------
  Env := TNeuralCartPoleEnv.Create();
  QNet := BuildQNetwork(Env.StateDim, Env.NumActions);  // caller builds
  QNet.SetLearningRate(0.001, 0);
  Trainer := TNeuralDQNTrainer.Create(QNet, 0.99, 1.0, 0.05, 0.999, 100, 50000, 32);
  for Epoch := 1 to 200 do
    Trainer.TrainEpisode(Env, 500);
  R := Trainer.TotalReward(Env, 200, 0.0);
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, neuralnetwork, neuralvolume, pascoremath32;

type
  TNeuralRLState = array of TNeuralFloat;

  TNeuralRLTransition = record
    State: TNeuralRLState;
    Action: integer;
    Reward: TNeuralFloat;
    NextState: TNeuralRLState;
    Done: boolean;
  end;

  // Coded by AI.
  TNeuralEnv = class
  private
    FStateDim, FNumActions: integer;
    FState: TNeuralRLState;
    FDone: boolean;
    FRngState: cardinal;
  protected
    function Rand(): TNeuralFloat;
    function RandGauss(ASigma: TNeuralFloat): TNeuralFloat;
  public
    constructor Create;
    destructor Destroy; override;
    procedure Reset; virtual; abstract;
    function Step(ACode: integer): TNeuralFloat; virtual; abstract;
    property Done: boolean read FDone;
    property StateDim: integer read FStateDim;
    property NumActions: integer read FNumActions;
    property State: TNeuralRLState read FState;
    property RngState: cardinal read FRngState write FRngState;
    procedure CopyState(out AOut: TNeuralRLState);
  end;

  // Coded by AI.
  TNeuralRLTransitions = array of TNeuralRLTransition;

  TNeuralReplayBuffer = class
  private
    FCapacity, FHead, FSize: integer;
    FBuffer: TNeuralRLTransitions;
    FBatch: TNeuralRLTransitions;
    FRngState: cardinal;
  protected
    function RandInt(AMax: integer): integer;
  public
    constructor Create(pCapacity: integer);
    destructor Destroy; override;
    procedure Add(const T: TNeuralRLTransition);
    procedure Sample(MiniBatch: integer);
    property Capacity: integer read FCapacity;
    property Size: integer read FSize;
    property Batch: TNeuralRLTransitions read FBatch;
  end;

  // Coded by AI.
  TNeuralDQNTrainer = class(TObject)
  private
    FQ, FTarget: TNNet;
    FOwnsTarget: boolean;
    FReplay: TNeuralReplayBuffer;
    FGamma, FEpsilon, FEpsilonMin, FEpsilonDecay: TNeuralFloat;
    FTargetUpdateFreq, FStepCount, FMiniBatch: integer;
    FDoubleDQN: boolean;
    FInput, FTargetInput, FPseudoTarget: TNNetVolume;
    FRngState: cardinal;
    FLastLoss: TNeuralFloat;
  private
    function Rand(): TNeuralFloat;
    function RandInt(AMax: integer): integer;
    function ForwardQ(const State: TNeuralRLState;
      ANet: TNNet; AInput: TNNetVolume): TNeuralRLState;
    procedure HardUpdateTarget;
  public
    constructor Create(pQNet: TNNet;
      pGamma: TNeuralFloat = 0.99; pEpsilon: TNeuralFloat = 1.0;
      pEpsilonMin: TNeuralFloat = 0.1; pEpsilonDecay: TNeuralFloat = 0.995;
      pTargetUpdateFreq: integer = 100; pReplayCapacity: integer = 10000;
      pMiniBatch: integer = 32);
    destructor Destroy; override;
    function Act(const State: TNeuralRLState): integer;
    function ActGreedy(const State: TNeuralRLState): integer;
    function QValues(const State: TNeuralRLState): TNeuralRLState;
    procedure Observe(const T: TNeuralRLTransition);
    function TrainStep: TNeuralFloat;
    function TrainEpisode(Env: TNeuralEnv; MaxSteps: integer): TNeuralFloat;
    function TotalReward(Env: TNeuralEnv; MaxSteps: integer;
      pEpsilonOverride: TNeuralFloat = 0.0): TNeuralFloat;
    procedure UpdateTarget;
    property QNet: TNNet read FQ;
    property TargetNet: TNNet read FTarget;
    property Replay: TNeuralReplayBuffer read FReplay;
    property Gamma: TNeuralFloat read FGamma write FGamma;
    property Epsilon: TNeuralFloat read FEpsilon write FEpsilon;
    property MiniBatch: integer read FMiniBatch write FMiniBatch;
    property DoubleDQN: boolean read FDoubleDQN write FDoubleDQN;
    property StepCount: integer read FStepCount;
    property LastLoss: TNeuralFloat read FLastLoss;
  end;

  // Coded by AI.
  TNeuralPolicyGradTrainer = class(TObject)
  private
    FPolicy, FCritic: TNNet;
    FOwnsCritic: boolean;
    FGamma: TNeuralFloat;
    FSkipDerivSoftmax: boolean;
    FInput, FPseudoTarget: TNNetVolume;
    FLastPolicyLoss, FLastValueLoss: TNeuralFloat;
    FRngState: cardinal;
    FEpsilon, FEpsilonMin, FEpsilonDecay: TNeuralFloat;
  private
    function Rand(): TNeuralFloat;
    function RandInt(AMax: integer): integer;
    procedure DetectSoftmaxVariant;
    function Forward(NN: TNNet; AInput: TNNetVolume;
      const State: TNeuralRLState): TNeuralRLState;
  public
    constructor Create(pPolicy: TNNet; pGamma: TNeuralFloat = 0.99;
      pEpsilon: TNeuralFloat = 1.0; pEpsilonMin: TNeuralFloat = 0.05;
      pEpsilonDecay: TNeuralFloat = 0.995);
    constructor CreateWithCritic(pPolicy, pCritic: TNNet;
      pGamma: TNeuralFloat = 0.99; pOwnsCritic: boolean = false;
      pEpsilon: TNeuralFloat = 1.0; pEpsilonMin: TNeuralFloat = 0.05;
      pEpsilonDecay: TNeuralFloat = 0.995);
    destructor Destroy; override;
    function Act(const State: TNeuralRLState): integer;
    function ActGreedy(const State: TNeuralRLState): integer;
    function PolicyDist(const State: TNeuralRLState): TNeuralRLState;
    function Value(const State: TNeuralRLState): TNeuralFloat;
    function TrainOnEpisode(Env: TNeuralEnv; MaxSteps: integer): TNeuralFloat;
    function TotalReward(Env: TNeuralEnv; MaxSteps: integer): TNeuralFloat;
    property Policy: TNNet read FPolicy;
    property Critic: TNNet read FCritic;
    property Gamma: TNeuralFloat read FGamma write FGamma;
    property Epsilon: TNeuralFloat read FEpsilon write FEpsilon;
    property LastPolicyLoss: TNeuralFloat read FLastPolicyLoss;
    property LastValueLoss: TNeuralFloat read FLastValueLoss;
  end;

  // Coded by AI.
  TNeuralCartPoleEnv = class(TNeuralEnv)
  private
    FSteps, FMaxSteps: integer;
    FGravity, FMassCart, FMassPole, FPoleLen, FForce, FDT: TNeuralFloat;
  public
    constructor Create;
    procedure Reset; override;
    function Step(ACode: integer): TNeuralFloat; override;
  end;

  // Coded by AI.
  TNeuralGridWorldEnv = class(TNeuralEnv)
  private
    FW, FH, FGoalX, FGoalY, FPitX, FPitY, FMaxSteps, FSteps: integer;
    FSlipProb: TNeuralFloat;
  public
    constructor Create(pW, pH, pMaxSteps: integer;
      pSlipProb: TNeuralFloat = 0.2);
    procedure Reset; override;
    function Step(ACode: integer): TNeuralFloat; override;
    property Width: integer read FW;
    property Height: integer read FH;
    property GoalX: integer read FGoalX;
    property GoalY: integer read FGoalY;
    procedure SetGoal(pX, pY: integer);
    procedure SetPit(pX, pY: integer);
  end;

implementation

// ======================================================================
// TNeuralEnv
// ======================================================================

constructor TNeuralEnv.Create;
begin
  inherited Create;
  FStateDim := 0; FNumActions := 0; FDone := false; FRngState := 12345;
end;

destructor TNeuralEnv.Destroy;
begin
  FState := nil;
  inherited Destroy;
end;

function TNeuralEnv.Rand(): TNeuralFloat;
begin
  FRngState := (FRngState * 1103515245 + 12345) and $7FFFFFFF;
  Result := FRngState / 2147483647.0;
end;

function TNeuralEnv.RandGauss(ASigma: TNeuralFloat): TNeuralFloat;
var U1, U2: TNeuralFloat;
begin
  repeat U1 := Rand until U1 > 1e-12;
  U2 := Rand;
  Result := Sqrt(-2.0 * Ln(U1)) * Cos(2.0 * Pi * U2) * ASigma;
end;

procedure TNeuralEnv.CopyState(out AOut: TNeuralRLState);
begin
  SetLength(AOut, FStateDim);
  if FStateDim > 0 then
    Move(FState[0], AOut[0], FStateDim * SizeOf(TNeuralFloat));
end;

// ======================================================================
// TNeuralReplayBuffer
// ======================================================================

function TNeuralReplayBuffer.RandInt(AMax: integer): integer;
var U: TNeuralFloat;
begin
  if AMax <= 0 then begin Result := 0; Exit; end;
  FRngState := (FRngState * 1103515245 + 12345) and $7FFFFFFF;
  U := FRngState / 2147483647.0;
  Result := Trunc(U * AMax);
  if Result >= AMax then Result := AMax - 1;
end;

constructor TNeuralReplayBuffer.Create(pCapacity: integer);
begin
  inherited Create;
  FCapacity := pCapacity; if FCapacity < 1 then FCapacity := 1;
  FHead := 0; FSize := 0; FRngState := 54321;
  SetLength(FBuffer, FCapacity);
end;

destructor TNeuralReplayBuffer.Destroy;
var I: integer;
begin
  for I := 0 to High(FBuffer) do
  begin FBuffer[I].State := nil; FBuffer[I].NextState := nil; end;
  for I := 0 to High(FBatch) do
  begin FBatch[I].State := nil; FBatch[I].NextState := nil; end;
  FBuffer := nil; FBatch := nil;
  inherited Destroy;
end;

procedure TNeuralReplayBuffer.Add(const T: TNeuralRLTransition);
begin
  if FSize >= FCapacity then
  begin FBuffer[FHead].State := nil; FBuffer[FHead].NextState := nil; end;
  SetLength(FBuffer[FHead].State, Length(T.State));
  if Length(T.State) > 0 then
    Move(T.State[0], FBuffer[FHead].State[0], Length(T.State) * SizeOf(TNeuralFloat));
  FBuffer[FHead].Action := T.Action;
  FBuffer[FHead].Reward := T.Reward;
  FBuffer[FHead].Done := T.Done;
  SetLength(FBuffer[FHead].NextState, Length(T.NextState));
  if Length(T.NextState) > 0 then
    Move(T.NextState[0], FBuffer[FHead].NextState[0], Length(T.NextState) * SizeOf(TNeuralFloat));
  FHead := (FHead + 1) mod FCapacity;
  if FSize < FCapacity then Inc(FSize);
end;

procedure TNeuralReplayBuffer.Sample(MiniBatch: integer);
var I, SrcIdx: integer;
begin
  if MiniBatch > FSize then MiniBatch := FSize;
  if MiniBatch <= 0 then MiniBatch := 1;
  if Length(FBatch) <> MiniBatch then
  begin
    for I := 0 to High(FBatch) do
    begin FBatch[I].State := nil; FBatch[I].NextState := nil; end;
    SetLength(FBatch, MiniBatch);
  end;
  for I := 0 to MiniBatch - 1 do
  begin
    SrcIdx := RandInt(FSize);
    FBatch[I].Action := FBuffer[SrcIdx].Action;
    FBatch[I].Reward := FBuffer[SrcIdx].Reward;
    FBatch[I].Done := FBuffer[SrcIdx].Done;
    SetLength(FBatch[I].State, Length(FBuffer[SrcIdx].State));
    if Length(FBuffer[SrcIdx].State) > 0 then
      Move(FBuffer[SrcIdx].State[0], FBatch[I].State[0],
        Length(FBuffer[SrcIdx].State) * SizeOf(TNeuralFloat));
    SetLength(FBatch[I].NextState, Length(FBuffer[SrcIdx].NextState));
    if Length(FBuffer[SrcIdx].NextState) > 0 then
      Move(FBuffer[SrcIdx].NextState[0], FBatch[I].NextState[0],
        Length(FBuffer[SrcIdx].NextState) * SizeOf(TNeuralFloat));
  end;
end;

// ======================================================================
// TNeuralDQNTrainer
// ======================================================================

function TNeuralDQNTrainer.Rand(): TNeuralFloat;
begin
  FRngState := (FRngState * 1103515245 + 12345) and $7FFFFFFF;
  Result := FRngState / 2147483647.0;
end;

function TNeuralDQNTrainer.RandInt(AMax: integer): integer;
var U: TNeuralFloat;
begin
  if AMax <= 0 then begin Result := 0; Exit; end;
  U := Rand;
  Result := Trunc(U * AMax);
  if Result >= AMax then Result := AMax - 1;
end;

constructor TNeuralDQNTrainer.Create(pQNet: TNNet;
  pGamma, pEpsilon, pEpsilonMin, pEpsilonDecay: TNeuralFloat;
  pTargetUpdateFreq, pReplayCapacity, pMiniBatch: integer);
begin
  inherited Create;
  FQ := pQNet;
  FTarget := pQNet.Clone;
  FOwnsTarget := true;
  FGamma := pGamma; FEpsilon := pEpsilon; FEpsilonMin := pEpsilonMin;
  FEpsilonDecay := pEpsilonDecay; FTargetUpdateFreq := pTargetUpdateFreq;
  FStepCount := 0; FMiniBatch := pMiniBatch; FDoubleDQN := false;
  FLastLoss := 0; FRngState := 42;
  FReplay := TNeuralReplayBuffer.Create(pReplayCapacity);
  FInput := TNNetVolume.Create;
  FTargetInput := TNNetVolume.Create;
  FPseudoTarget := TNNetVolume.Create;
end;

destructor TNeuralDQNTrainer.Destroy;
begin
  FInput.Free; FTargetInput.Free; FPseudoTarget.Free; FReplay.Free;
  if FOwnsTarget then FTarget.Free;
  inherited Destroy;
end;

function TNeuralDQNTrainer.ForwardQ(const State: TNeuralRLState;
  ANet: TNNet; AInput: TNNetVolume): TNeuralRLState;
var FirstOut, LastOut: TNNetVolume; D, I: integer;
begin
  FirstOut := ANet.GetFirstLayer().Output;
  if AInput.Size <> FirstOut.Size then
    AInput.ReSize(FirstOut.SizeX, FirstOut.SizeY, FirstOut.Depth);
  D := FirstOut.Size;
  for I := 0 to D - 1 do AInput.FData[I] := State[I];
  ANet.Compute(AInput);
  LastOut := ANet.GetLastLayer().Output;
  SetLength(Result, LastOut.Size);
  for I := 0 to LastOut.Size - 1 do Result[I] := LastOut.FData[I];
end;

function TNeuralDQNTrainer.QValues(const State: TNeuralRLState): TNeuralRLState;
begin
  Result := ForwardQ(State, FQ, FInput);
end;

function TNeuralDQNTrainer.ActGreedy(const State: TNeuralRLState): integer;
var Qs: TNeuralRLState; I, BestI: integer; BestV: TNeuralFloat;
begin
  Qs := QValues(State);
  BestI := 0; BestV := Qs[0];
  for I := 1 to High(Qs) do
    if Qs[I] > BestV then begin BestV := Qs[I]; BestI := I; end;
  Result := BestI;
end;

function TNeuralDQNTrainer.Act(const State: TNeuralRLState): integer;
var Qs: TNeuralRLState;
begin
  if (FEpsilon > 0) and (Rand < FEpsilon) then
  begin
    Qs := QValues(State);
    Result := RandInt(Length(Qs));
  end
  else
    Result := ActGreedy(State);
end;

procedure TNeuralDQNTrainer.Observe(const T: TNeuralRLTransition);
begin
  FReplay.Add(T);
end;

function TNeuralDQNTrainer.TrainStep: TNeuralFloat;
var
  I, N, A, D: integer;
  QVals, TVals, OnlineQs: TNeuralRLState;
  Target, MeanLoss, BestV: TNeuralFloat;
  BestA: integer;
begin
  if FReplay.Size < FMiniBatch then
  begin FLastLoss := 0; Result := 0; Exit; end;

  FReplay.Sample(FMiniBatch);
  N := Length(FReplay.Batch);
  FQ.SetBatchUpdate(true);
  FQ.ClearDeltas;
  MeanLoss := 0;

  for I := 0 to N - 1 do
  begin
    QVals := ForwardQ(FReplay.Batch[I].State, FQ, FInput);
    D := Length(QVals);

    if FReplay.Batch[I].Done then
      Target := FReplay.Batch[I].Reward
    else
    begin
      if FDoubleDQN then
      begin
        OnlineQs := ForwardQ(FReplay.Batch[I].NextState, FQ, FTargetInput);
        BestA := 0; BestV := OnlineQs[0];
        for A := 1 to High(OnlineQs) do
          if OnlineQs[A] > BestV then begin BestV := OnlineQs[A]; BestA := A; end;
        TVals := ForwardQ(FReplay.Batch[I].NextState, FTarget, FInput);
        Target := FReplay.Batch[I].Reward + FGamma * TVals[BestA];
      end
      else
      begin
        TVals := ForwardQ(FReplay.Batch[I].NextState, FTarget, FInput);
        BestV := TVals[0];
        for A := 1 to High(TVals) do
          if TVals[A] > BestV then BestV := TVals[A];
        Target := FReplay.Batch[I].Reward + FGamma * BestV;
      end;
    end;

    // Pseudo-target: error[action] = Q(s,a) - target, error[a'] = 0.
    if FPseudoTarget.Size <> D then FPseudoTarget.ReSize(1, 1, D);
    for A := 0 to D - 1 do FPseudoTarget.FData[A] := QVals[A];
    FPseudoTarget.FData[FReplay.Batch[I].Action] := Target;

    FQ.Backpropagate(FPseudoTarget);
    MeanLoss := MeanLoss + Sqr(QVals[FReplay.Batch[I].Action] - Target);
  end;

  FQ.UpdateWeights;
  FQ.SetBatchUpdate(false);
  FLastLoss := MeanLoss / N;
  Result := FLastLoss;

  Inc(FStepCount);
  if (FStepCount mod FTargetUpdateFreq = 0) then HardUpdateTarget;
end;

procedure TNeuralDQNTrainer.HardUpdateTarget;
begin
  if FOwnsTarget then FTarget.Free;
  FTarget := FQ.Clone;
  FOwnsTarget := true;
end;

procedure TNeuralDQNTrainer.UpdateTarget;
begin
  HardUpdateTarget;
end;

function TNeuralDQNTrainer.TrainEpisode(Env: TNeuralEnv; MaxSteps: integer): TNeuralFloat;
var
  I, A: integer; R, TotalR: TNeuralFloat;
  S, SN: TNeuralRLState; T: TNeuralRLTransition;
begin
  Env.Reset;
  Env.CopyState(S);
  TotalR := 0;
  for I := 0 to MaxSteps - 1 do
  begin
    A := Act(S);
    R := Env.Step(A);
    TotalR := TotalR + R;
    Env.CopyState(SN);
    T.Action := A; T.Reward := R; T.Done := Env.Done;
    T.State := S; T.NextState := SN;
    Observe(T);
    // Standard DQN: one gradient step per environment step.
    if FReplay.Size >= FMiniBatch then
      TrainStep;
    S := SN;
    if Env.Done then Break;
  end;
  FEpsilon := Max(FEpsilonMin, FEpsilon * FEpsilonDecay);
  Result := TotalR;
end;

function TNeuralDQNTrainer.TotalReward(Env: TNeuralEnv; MaxSteps: integer;
  pEpsilonOverride: TNeuralFloat): TNeuralFloat;
var
  I, A: integer; R, OldEps: TNeuralFloat;
  S: TNeuralRLState; Qs: TNeuralRLState;
begin
  OldEps := FEpsilon;
  FEpsilon := pEpsilonOverride;
  Env.Reset;
  Env.CopyState(S);
  Result := 0;
  for I := 0 to MaxSteps - 1 do
  begin
    if (FEpsilon > 0) and (Rand < FEpsilon) then
    begin Qs := QValues(S); A := RandInt(Length(Qs)); end
    else
      A := ActGreedy(S);
    R := Env.Step(A);
    Result := Result + R;
    Env.CopyState(S);
    if Env.Done then Break;
  end;
  FEpsilon := OldEps;
end;

// ======================================================================
// TNeuralPolicyGradTrainer
// ======================================================================

function TNeuralPolicyGradTrainer.Rand(): TNeuralFloat;
begin
  FRngState := (FRngState * 1103515245 + 12345) and $7FFFFFFF;
  Result := FRngState / 2147483647.0;
end;

function TNeuralPolicyGradTrainer.RandInt(AMax: integer): integer;
var U: TNeuralFloat;
begin
  if AMax <= 0 then begin Result := 0; Exit; end;
  U := Rand;
  Result := Trunc(U * AMax);
  if Result >= AMax then Result := AMax - 1;
end;

procedure TNeuralPolicyGradTrainer.DetectSoftmaxVariant;
var
  LastLayer: TNNetLayer;
  StructStr: string;
  ColonPos, SemiPos: integer;
begin
  FSkipDerivSoftmax := false;
  LastLayer := FPolicy.GetLastLayer;
  if not (LastLayer is TNNetPointwiseSoftMax) then
    raise Exception.Create(
      'TNeuralPolicyGradTrainer requires the policy to end in a softmax layer ' +
      '(TNNetSoftMax or TNNetPointwiseSoftMax). Found: ' + LastLayer.ClassName + '.');
  StructStr := LastLayer.SaveStructureToString;
  ColonPos := Pos(':', StructStr);
  SemiPos := Pos(';', StructStr);
  if (ColonPos > 0) and (SemiPos > ColonPos) then
    FSkipDerivSoftmax :=
      StrToIntDef(Copy(StructStr, ColonPos + 1, SemiPos - ColonPos - 1), 0) > 0;
end;

function TNeuralPolicyGradTrainer.Forward(NN: TNNet; AInput: TNNetVolume;
  const State: TNeuralRLState): TNeuralRLState;
var FirstOut, LastOut: TNNetVolume; D, I: integer;
begin
  FirstOut := NN.GetFirstLayer().Output;
  if AInput.Size <> FirstOut.Size then
    AInput.ReSize(FirstOut.SizeX, FirstOut.SizeY, FirstOut.Depth);
  D := FirstOut.Size;
  for I := 0 to D - 1 do AInput.FData[I] := State[I];
  NN.Compute(AInput);
  LastOut := NN.GetLastLayer().Output;
  SetLength(Result, LastOut.Size);
  for I := 0 to LastOut.Size - 1 do Result[I] := LastOut.FData[I];
end;

constructor TNeuralPolicyGradTrainer.Create(pPolicy: TNNet; pGamma: TNeuralFloat;
  pEpsilon, pEpsilonMin, pEpsilonDecay: TNeuralFloat);
begin
  inherited Create;
  FPolicy := pPolicy; FCritic := nil; FOwnsCritic := false;
  FGamma := pGamma; FLastPolicyLoss := 0; FLastValueLoss := 0;
  FEpsilon := pEpsilon; FEpsilonMin := pEpsilonMin; FEpsilonDecay := pEpsilonDecay;
  FRngState := 777;
  FInput := TNNetVolume.Create;
  FPseudoTarget := TNNetVolume.Create;
  DetectSoftmaxVariant;
end;

constructor TNeuralPolicyGradTrainer.CreateWithCritic(pPolicy, pCritic: TNNet;
  pGamma: TNeuralFloat; pOwnsCritic: boolean;
  pEpsilon, pEpsilonMin, pEpsilonDecay: TNeuralFloat);
begin
  Create(pPolicy, pGamma, pEpsilon, pEpsilonMin, pEpsilonDecay);
  FCritic := pCritic;
  FOwnsCritic := pOwnsCritic;
end;

destructor TNeuralPolicyGradTrainer.Destroy;
begin
  FInput.Free; FPseudoTarget.Free;
  if FOwnsCritic then FCritic.Free;
  inherited Destroy;
end;

function TNeuralPolicyGradTrainer.PolicyDist(const State: TNeuralRLState): TNeuralRLState;
begin
  Result := Forward(FPolicy, FInput, State);
end;

function TNeuralPolicyGradTrainer.Act(const State: TNeuralRLState): integer;
var Dist: TNeuralRLState; U, Cum: TNeuralFloat; I: integer;
begin
  // Epsilon-greedy exploration.
  if (FEpsilon > 0) and (Rand < FEpsilon) then
  begin
    Dist := PolicyDist(State);
    Result := RandInt(Length(Dist));
    Exit;
  end;
  Dist := PolicyDist(State);
  U := Rand; Cum := 0;
  for I := 0 to High(Dist) do
  begin
    Cum := Cum + Dist[I];
    if (U < Cum) or (I = High(Dist)) then begin Result := I; Exit; end;
  end;
  Result := High(Dist);
end;

function TNeuralPolicyGradTrainer.ActGreedy(const State: TNeuralRLState): integer;
var Dist: TNeuralRLState; I, BestI: integer; BestV: TNeuralFloat;
begin
  Dist := PolicyDist(State);
  BestI := 0; BestV := Dist[0];
  for I := 1 to High(Dist) do
    if Dist[I] > BestV then begin BestV := Dist[I]; BestI := I; end;
  Result := BestI;
end;

function TNeuralPolicyGradTrainer.Value(const State: TNeuralRLState): TNeuralFloat;
var V: TNeuralRLState;
begin
  if FCritic = nil then begin Result := 0; Exit; end;
  V := Forward(FCritic, FInput, State);
  Result := V[0];
end;

function TNeuralPolicyGradTrainer.TrainOnEpisode(Env: TNeuralEnv;
  MaxSteps: integer): TNeuralFloat;
var
  States: array of TNeuralRLState;
  Actions: array of integer;
  Rewards: array of TNeuralFloat;
  Dones: array of boolean;
  Returns, Advantages: array of TNeuralFloat;
  I, N, D, A, J: integer;
  S: TNeuralRLState;
  R, TotalR, G, MeanRet, StdRet: TNeuralFloat;
  Pi, V: TNeuralRLState;
  VLoss: TNeuralFloat;
begin
  Env.Reset;
  Env.CopyState(S);
  N := 0; TotalR := 0;

  // Phase 1: Collect episode.
  for I := 0 to MaxSteps - 1 do
  begin
    if Length(States) < N + 1 then
    begin
      SetLength(States, (N + 1) * 2);
      SetLength(Actions, (N + 1) * 2);
      SetLength(Rewards, (N + 1) * 2);
      SetLength(Dones, (N + 1) * 2);
    end;
    States[N] := S;
    A := Act(S); Actions[N] := A;
    R := Env.Step(A); Rewards[N] := R; Dones[N] := Env.Done;
    TotalR := TotalR + R;
    Inc(N);
    Env.CopyState(S);
    if Env.Done then Break;
  end;
  SetLength(States, N); SetLength(Actions, N);
  SetLength(Rewards, N); SetLength(Dones, N);
  SetLength(Returns, N); SetLength(Advantages, N);

  if N = 0 then begin Result := 0; Exit; end;

  // Phase 2: Discounted returns.
  G := 0;
  for I := N - 1 downto 0 do
  begin
    if Dones[I] then G := Rewards[I]
    else G := Rewards[I] + FGamma * G;
    Returns[I] := G;
  end;

  // Phase 3: Advantages.
  // Use raw returns with fixed scaling (no within-episode normalization).
  // Within-episode mean/std destroys the signal: in a bad episode all returns
  // are ~-15 (tiny variance), so normalized advantages are pure noise.
  // Scaling by 0.05 maps typical range [-15,+10] to [-0.75,+0.5], giving
  // a clear contrast: good episodes (goal reached) reinforce, bad ones suppress.
  if FCritic <> nil then
    for I := 0 to N - 1 do
    begin
      Advantages[I] := (Returns[I] - Value(States[I])) * 0.05;
      if Advantages[I] > 0.5 then Advantages[I] := 0.5;
      if Advantages[I] < -0.5 then Advantages[I] := -0.5;
    end
  else
    for I := 0 to N - 1 do
    begin
      Advantages[I] := Returns[I] * 0.05;
      if Advantages[I] > 0.5 then Advantages[I] := 0.5;
      if Advantages[I] < -0.5 then Advantages[I] := -0.5;
    end;

  // Phase 4: Policy gradient (REINFORCE).
  // Minimize L = -sum A_t * log pi(a_t|s_t).
  // dL/dlogit = -A*(onehot - pi) = A*(pi - onehot).
  // Pseudo-target (same trick as neuraldpo):
  //   SkipDeriv=0: pDesired[a] = pi[a] - A/pi[a], pDesired[a'] = pi[a']
  //   SkipDeriv=1: pDesired[a] = pi[a] + A*(pi[a]-1), pDesired[a'] = pi[a']*(1+A)
  FPolicy.SetBatchUpdate(true);
  FPolicy.ClearDeltas;
  for I := 0 to N - 1 do
  begin
    Pi := Forward(FPolicy, FInput, States[I]);
    D := Length(Pi); A := Actions[I];
    if FPseudoTarget.Size <> D then FPseudoTarget.ReSize(1, 1, D);

    if FSkipDerivSoftmax then
      for J := 0 to D - 1 do
        if J = A then
          FPseudoTarget.FData[J] := Pi[J] * (1.0 - Advantages[I]) + Advantages[I]
        else
          FPseudoTarget.FData[J] := Pi[J] * (1.0 - Advantages[I])
    else
      for J := 0 to D - 1 do
        if J = A then
          if Pi[J] > 0.1 then
            FPseudoTarget.FData[J] := Pi[J] + Advantages[I] / Pi[J]
          else
            FPseudoTarget.FData[J] := Pi[J]
        else
          FPseudoTarget.FData[J] := Pi[J];

    FPolicy.Backpropagate(FPseudoTarget);
  end;
  FPolicy.UpdateWeights;
  FPolicy.SetBatchUpdate(false);

  // Phase 5: Critic update.
  if FCritic <> nil then
  begin
    FCritic.SetBatchUpdate(true);
    FCritic.ClearDeltas;
    VLoss := 0;
    for I := 0 to N - 1 do
    begin
      V := Forward(FCritic, FInput, States[I]);
      if FPseudoTarget.Size <> 1 then FPseudoTarget.ReSize(1, 1, 1);
      FPseudoTarget.FData[0] := Returns[I];
      FCritic.Backpropagate(FPseudoTarget);
      VLoss := VLoss + Sqr(V[0] - Returns[I]);
    end;
    FCritic.UpdateWeights;
    FCritic.SetBatchUpdate(false);
    FLastValueLoss := VLoss / N;
  end;

  FEpsilon := Max(FEpsilonMin, FEpsilon * FEpsilonDecay);
  Result := TotalR;
end;

function TNeuralPolicyGradTrainer.TotalReward(Env: TNeuralEnv;
  MaxSteps: integer): TNeuralFloat;
var I, A: integer; R: TNeuralFloat; S: TNeuralRLState;
begin
  Env.Reset;
  Env.CopyState(S);
  Result := 0;
  for I := 0 to MaxSteps - 1 do
  begin
    A := ActGreedy(S);
    R := Env.Step(A);
    Result := Result + R;
    Env.CopyState(S);
    if Env.Done then Break;
  end;
end;

// ======================================================================
// TNeuralCartPoleEnv
// ======================================================================

constructor TNeuralCartPoleEnv.Create;
begin
  inherited Create;
  FStateDim := 4; FNumActions := 2; FMaxSteps := 500;
  FGravity := 9.8; FMassCart := 1.0; FMassPole := 0.1;
  FPoleLen := 0.5; FForce := 10.0; FDT := 0.02;
  SetLength(FState, 4);
end;

procedure TNeuralCartPoleEnv.Reset;
begin
  FState[0] := (Rand * 2 - 1) * 0.05;
  FState[1] := (Rand * 2 - 1) * 0.05;
  FState[2] := (Rand * 2 - 1) * (2.0 * Pi / 45.0);
  FState[3] := (Rand * 2 - 1) * (2.0 * Pi / 45.0);
  FSteps := 0; FDone := false;
end;

function TNeuralCartPoleEnv.Step(ACode: integer): TNeuralFloat;
var
  U, SinTh, CosTh, TotalMass, ThAcc, CartAcc: TNeuralFloat;
begin
  if ACode = 1 then U := FForce else U := -FForce;
  SinTh := Sin(FState[2]);
  CosTh := Cos(FState[2]);
  TotalMass := FMassCart + FMassPole;

  // OpenAI Gym CartPole-v1 equations:
  //   xacc  = (u + m*L*w^2*sin(th) - m*g*sin(th)*cos(th)) / (M+m)
  //   thacc = (g*sin(th) - cos(th) * (u/(M+m) + w^2*L*sin(th)*m/(M+m))) / L
  CartAcc := (U + FMassPole * FPoleLen * FState[3] * FState[3] * SinTh
    - FMassPole * FGravity * SinTh * CosTh) / TotalMass;
  ThAcc := (FGravity * SinTh
    - CosTh * (U / TotalMass
    + FState[3] * FState[3] * FPoleLen * SinTh * FMassPole / TotalMass)) / FPoleLen;

  FState[1] := FState[1] + FDT * CartAcc;
  FState[3] := FState[3] + FDT * ThAcc;
  FState[0] := FState[0] + FDT * FState[1];
  FState[2] := FState[2] + FDT * FState[3];

  Inc(FSteps);
  if (Abs(FState[0]) > 2.4) or (Abs(FState[2]) > 0.2095) or (FSteps >= FMaxSteps)
    then FDone := true;
  Result := 1.0;
end;

// ======================================================================
// TNeuralGridWorldEnv
// ======================================================================

constructor TNeuralGridWorldEnv.Create(pW, pH, pMaxSteps: integer;
  pSlipProb: TNeuralFloat);
begin
  inherited Create;
  FW := pW; FH := pH; FMaxSteps := pMaxSteps; FSlipProb := pSlipProb;
  FStateDim := 4; FNumActions := 4;
  SetLength(FState, 4);
  FGoalX := pW - 1; FGoalY := 0;
  FPitX := pW div 2; FPitY := pH div 2;
end;

procedure TNeuralGridWorldEnv.SetGoal(pX, pY: integer);
begin
  FGoalX := pX; FGoalY := pY;
end;

procedure TNeuralGridWorldEnv.SetPit(pX, pY: integer);
begin
  FPitX := pX; FPitY := pY;
end;

procedure TNeuralGridWorldEnv.Reset;
begin
  FState[0] := 0; FState[1] := 0; FState[2] := 0; FState[3] := 0;
  FSteps := 0; FDone := false;
end;

function TNeuralGridWorldEnv.Step(ACode: integer): TNeuralFloat;
var
  DX, DY, NewX, NewY, ActualAction: integer;
begin
  Result := -1.0;

  // Stochastic slip.
  ActualAction := ACode;
  if (Rand < FSlipProb) then
    ActualAction := Trunc(Rand * 4);
  if ActualAction >= 4 then ActualAction := 3;

  case ActualAction of
    0: begin DX := 0; DY := -1; end;
    1: begin DX := 1; DY := 0;  end;
    2: begin DX := 0; DY := 1;  end;
    3: begin DX := -1; DY := 0; end;
  else begin DX := 0; DY := 0; end;
  end;

  NewX := Trunc(FState[0]) + DX;
  NewY := Trunc(FState[1]) + DY;
  if NewX < 0 then NewX := 0;
  if NewX >= FW then NewX := FW - 1;
  if NewY < 0 then NewY := 0;
  if NewY >= FH then NewY := FH - 1;

  FState[0] := NewX;
  FState[1] := NewY;
  Inc(FSteps);

  if (NewX = FGoalX) and (NewY = FGoalY) then
  begin Result := 10.0; FDone := true; end
  else if (NewX = FPitX) and (NewY = FPitY) then
  begin Result := -10.0; FDone := true; end
  else if FSteps >= FMaxSteps then
    FDone := true;
end;

end.
