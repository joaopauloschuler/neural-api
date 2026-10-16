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
  * TNeuralPPOTrainer        - PPO (on-policy rollouts, GAE with optional critic,
                               clipped surrogate objective, multiple epochs of
                               minibatch updates)
  * TNeuralCartPoleEnv       - built-in CartPole environment (Gym-classic)
  * TNeuralGridWorldEnv      - built-in 2-D stochastic grid world

NETWORK REQUIREMENTS
--------------------------------------------------------------------------
  Q-NET (DQN):    Input(1,1,D) -> ... -> TNNetFullConnectLinear(NumActions)
  POLICY (PG):    Input(1,1,D) -> ... -> TNNetFullConnectLinear(NumActions)
                  -> TNNetSoftMax
  CRITIC (opt):   Input(1,1,D) -> ... -> TNNetFullConnectLinear(1)

  IMPORTANT: TNNetSoftMax is a PASS-THROUGH layer (output size = input size,
  no width parameter), so the policy MUST project to NumActions with a
  FullConnect layer BEFORE the softmax. A softmax placed directly on a hidden
  layer defines a distribution over the hidden width instead of over the
  actions. All trainers validate the shapes against the environment at the
  first TrainEpisode/TrainOnEpisode/TotalReward call and raise on mismatch.

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
    FActionCount: integer;
  private
    function Rand(): TNeuralFloat;
    function RandInt(AMax: integer): integer;
    function ForwardQ(const State: TNeuralRLState;
      ANet: TNNet; AInput: TNNetVolume): TNeuralRLState;
    procedure HardUpdateTarget;
    procedure ValidateAgainstEnv(Env: TNeuralEnv);
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
    FProbFloor: TNeuralFloat;
    FUpdateFloor: TNeuralFloat;
    FActionCount: integer;
  private
    function Rand(): TNeuralFloat;
    function RandInt(AMax: integer): integer;
    procedure DetectSoftmaxVariant;
    procedure ValidatePolicyAgainstEnv(Env: TNeuralEnv);
    procedure ValidateCriticHead;
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
    // Numerical floor for the log(pi) term of the LastPolicyLoss diagnostic
    // (same convention as TNeuralDPOTrainer.ProbFloor in neuraldpo.pas).
    // Default 1e-9; keeps log() finite when a taken action has probability 0.
    property ProbFloor: TNeuralFloat read FProbFloor write FProbFloor;
    // Policy-gradient update floor: steps whose taken-action probability is at
    // or below this value are skipped by the REINFORCE update (see Phase 4 of
    // TrainOnEpisode). Default Min(0.1, 0.5/NumActions): being relative to the
    // action count keeps a uniform policy (pi = 1/NumActions) updating even
    // with many actions, while still suppressing the off-policy bias of
    // epsilon-random rare actions.
    property UpdateFloor: TNeuralFloat read FUpdateFloor write FUpdateFloor;
    property LastPolicyLoss: TNeuralFloat read FLastPolicyLoss;
    property LastValueLoss: TNeuralFloat read FLastValueLoss;
  end;

  // Coded by AI.
  // TNeuralPPOTrainer: Proximal Policy Optimization for discrete-action
  // environments (Schulman et al. 2017). One call to TrainOnEpisode runs the
  // full PPO cycle:
  //   1) on-policy rollout: actions sampled from pi with a small optional
  //      exploration epsilon (ExploreEpsilon, default 0.1, in the same
  //      epsilon-greedy style as every other trainer in this unit; the
  //      recorded log-prob is always the POLICY's, so the PPO ratio stays
  //      anchored to pi_old) while recording log pi_old(a|s);
  //   2) advantages: GAE(lambda) against a scalar critic, or Monte-Carlo
  //      discounted returns when no critic is provided; then normalized to
  //      mean 0 / unit variance over the whole batch (NormalizeAdvantages,
  //      default on - textbook PPO), which also removes the environment's
  //      raw reward scale so one LR works across envs;
  //   3) E epochs of shuffled minibatch updates on the SAME rollout, each
  //      token's gradient taken from the PPO clipped surrogate
  //      min(r*A, clip(r, 1-e, 1+e)*A) with r = pi_new/pi_old: the clipped
  //      branch contributes ZERO gradient (exact, same convention as the
  //      optional clip in TNeuralGRPOTrainer), the unclipped branch
  //      contributes r*A through the shared ErrScale*(onehot-pi) plumbing.
  //   4) critic regression against the GAE value targets A + V_old.
  // The ratio mechanism is numerically friendlier than REINFORCE: the policy
  // gradient is r*A with NO 1/pi(a) division, so no update floor is needed.
  // Not implemented (documented on purpose): the entropy bonus (its gradient
  // touches ALL logits and cannot be expressed as a single pseudo-target
  // scale, unlike every other term handled by this unit's plumbing) and
  // value clipping. Truncation at MaxSteps is treated as end-of-return
  // (no bootstrap at the horizon), matching TNeuralPolicyGradTrainer.
  TNeuralPPOTrainer = class(TObject)
  private
    FPolicy, FCritic: TNNet;
    FOwnsCritic: boolean;
    FGamma, FLambda, FClipEpsilon: TNeuralFloat;
    FEpochs, FMiniBatchSize: integer;
    FNormalizeAdvantages: boolean;
    FSkipDerivSoftmax: boolean;
    FInput, FPseudoTarget: TNNetVolume;
    FRngState: cardinal;
    FProbFloor: TNeuralFloat;
    FExploreEpsilon: TNeuralFloat;
    FLastPolicyLoss, FLastValueLoss, FLastClipFraction: TNeuralFloat;
    FActionCount: integer;
  private
    function Rand(): TNeuralFloat;
    function RandInt(AMax: integer): integer;
    procedure DetectSoftmaxVariant;
    procedure ValidatePolicyAgainstEnv(Env: TNeuralEnv);
    procedure ValidateCriticHead;
    function Forward(NN: TNNet; AInput: TNNetVolume;
      const State: TNeuralRLState): TNeuralRLState;
    // Same plumbing as TNeuralGRPOTrainer.BackpropagateFromLastForward
    // (neuraldpo.pas): turns dL/dlogpi(target) = ErrScale into the exact
    // logit gradient ErrScale*(onehot - pi) via a pseudo-target.
    procedure BackpropagateFromLastForward(Target: integer;
      ErrScale: TNeuralFloat);
  public
    // Critic-less PPO: advantages are Monte-Carlo discounted returns.
    constructor Create(pPolicy: TNNet; pGamma: TNeuralFloat = 0.99;
      pLambda: TNeuralFloat = 0.95; pClipEpsilon: TNeuralFloat = 0.2;
      pEpochs: integer = 4; pMiniBatchSize: integer = 64);
    // Standard PPO: GAE(lambda) advantages against the critic's V(s).
    constructor CreateWithCritic(pPolicy, pCritic: TNNet;
      pGamma: TNeuralFloat = 0.99; pLambda: TNeuralFloat = 0.95;
      pClipEpsilon: TNeuralFloat = 0.2; pEpochs: integer = 4;
      pMiniBatchSize: integer = 64; pOwnsCritic: boolean = false);
    destructor Destroy; override;
    // Samples from the current policy and reports log pi(a|s) of the SAMPLED
    // action. Use this (not Act) during rollouts: the recorded log-prob is
    // the denominator of the PPO ratio.
    function ActWithLogProb(const State: TNeuralRLState;
      out ALogProb: TNeuralFloat): integer;
    function PolicyDist(const State: TNeuralRLState): TNeuralRLState;
    function ActGreedy(const State: TNeuralRLState): integer;
    function Value(const State: TNeuralRLState): TNeuralFloat;
    // Standard PPO usage: collect NumEpisodes episodes into ONE batch, then
    // run the multi-epoch clipped-surrogate update over it (the GAE recursion
    // restarts at every episode boundary, so the mixed buffer is correct).
    // Single-episode updates give a very noisy advantage distribution and can
    // let a confidently-wrong policy trap itself; prefer NumEpisodes >= 8.
    function TrainOnEpisodes(Env: TNeuralEnv; NumEpisodes,
      MaxSteps: integer): TNeuralFloat;
    // Single-episode convenience wrapper around TrainOnEpisodes.
    function TrainOnEpisode(Env: TNeuralEnv; MaxSteps: integer): TNeuralFloat;
    function TotalReward(Env: TNeuralEnv; MaxSteps: integer): TNeuralFloat;
    property Policy: TNNet read FPolicy;
    property Critic: TNNet read FCritic;
    property Gamma: TNeuralFloat read FGamma write FGamma;
    // GAE discount factor for the advantage recursion (0 = MC returns).
    property Lambda: TNeuralFloat read FLambda write FLambda;
    property ClipEpsilon: TNeuralFloat read FClipEpsilon write FClipEpsilon;
    property Epochs: integer read FEpochs write FEpochs;
    // Minibatch size for the per-epoch updates; values >= N degrade to a
    // single full-batch update per epoch.
    property MiniBatchSize: integer read FMiniBatchSize write FMiniBatchSize;
    // Numerical floor for the log(pi) terms (same convention as
    // TNeuralPolicyGradTrainer.ProbFloor). Default 1e-9.
    property ProbFloor: TNeuralFloat read FProbFloor write FProbFloor;
    // Exploration: with probability ExploreEpsilon the rollout samples a
    // UNIFORM-random action instead of the policy's sample (epsilon-greedy,
    // same style as the other trainers in this unit). The recorded log-prob
    // is ALWAYS the policy's log-prob of the taken action, so the ratio
    // r = pi_new/pi_old stays anchored to pi_old; the epsilon-mixed behavior
    // introduces a small, second-order bias that buys the exploration a
    // no-entropy-bonus PPO needs on sparse-reward envs. Default 0.1.
    property ExploreEpsilon: TNeuralFloat read FExploreEpsilon write FExploreEpsilon;
    // Standard PPO batch normalization of the advantages (mean 0, unit
    // variance over the WHOLE multi-episode batch, guard std < 1e-6). This
    // is the batch-wide version - it does NOT hit the within-episode
    // normalization trap of TNeuralPolicyGradTrainer Phase 3, because a
    // multi-episode batch mixes good and bad episodes and keeps real
    // variance. It also removes the environment's raw reward scale
    // (GridWorld +/-10 vs CartPole +1) so one LR works across envs.
    // Default true (textbook PPO behavior).
    property NormalizeAdvantages: boolean read FNormalizeAdvantages write FNormalizeAdvantages;
    // Diagnostics from the last TrainOnEpisode: -mean(surrogate), mean
    // squared critic error, and the fraction of token gradients cut by the
    // clip window (a healthy PPO run sits well below 1; climbing towards it
    // means the policy is drifting faster than the clip allows).
    property LastPolicyLoss: TNeuralFloat read FLastPolicyLoss;
    property LastValueLoss: TNeuralFloat read FLastValueLoss;
    property LastClipFraction: TNeuralFloat read FLastClipFraction;
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

// Validates the Q-net against the environment's dimensions at the API
// boundary (same fail-loudly style as neuraldpo.pas): a Q-net whose output
// size differs from NumActions would silently emit out-of-range action ids,
// and an input size that differs from StateDim would read past the state
// vector.
procedure TNeuralDQNTrainer.ValidateAgainstEnv(Env: TNeuralEnv);
var
  FirstLayer, LastLayer: TNNetLayer;
begin
  LastLayer := FQ.GetLastLayer();
  if LastLayer.Output.Size <> Env.NumActions then
    raise Exception.Create('TNeuralDQNTrainer: the Q-net''s last layer (' +
      LastLayer.ClassName + ') outputs ' + IntToStr(LastLayer.Output.Size) +
      ' values but the environment has ' + IntToStr(Env.NumActions) +
      ' actions. Build the Q-net as Input(1,1,StateDim) -> ... -> ' +
      'TNNetFullConnectLinear(NumActions).');
  FirstLayer := FQ.GetFirstLayer();
  if FirstLayer.Output.Size <> Env.StateDim then
    raise Exception.Create('TNeuralDQNTrainer: the Q-net''s input layer (' +
      FirstLayer.ClassName + ') has size ' +
      IntToStr(FirstLayer.Output.Size) + ' but the environment''s states ' +
      'have ' + IntToStr(Env.StateDim) + ' dimensions. Build the input as ' +
      'TNNetInput.Create(1, 1, StateDim).');
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
  // The action count is the Q-net's output size, known before any Compute.
  FActionCount := pQNet.GetLastLayer().Output.Size;
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
  ValidateAgainstEnv(Env);
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
  ValidateAgainstEnv(Env);
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

// Validates the policy (and critic, when present) against the environment's
// dimensions at the API boundary (same fail-loudly style as neuraldpo.pas).
// The classic failure this catches: TNNetSoftMax is a pass-through layer, so
// a policy built as "... -> FullConnectReLU(Hidden) -> TNNetSoftMax" defines
// a distribution over the HIDDEN width, not over the actions; with far more
// outputs than actions, every update degenerates and learning silently never
// happens.
procedure TNeuralPolicyGradTrainer.ValidatePolicyAgainstEnv(Env: TNeuralEnv);
var
  FirstLayer, LastLayer: TNNetLayer;
begin
  LastLayer := FPolicy.GetLastLayer();
  if LastLayer.Output.Size <> Env.NumActions then
    raise Exception.Create('TNeuralPolicyGradTrainer: the policy''s last ' +
      'layer (' + LastLayer.ClassName + ') outputs ' +
      IntToStr(LastLayer.Output.Size) + ' values but the environment has ' +
      IntToStr(Env.NumActions) + ' actions. TNNetSoftMax is a pass-through ' +
      'layer: project to the action space with ' +
      'TNNetFullConnectLinear.Create(NumActions) BEFORE the softmax.');
  FirstLayer := FPolicy.GetFirstLayer();
  if FirstLayer.Output.Size <> Env.StateDim then
    raise Exception.Create('TNeuralPolicyGradTrainer: the policy''s input ' +
      'layer (' + FirstLayer.ClassName + ') has size ' +
      IntToStr(FirstLayer.Output.Size) + ' but the environment''s states ' +
      'have ' + IntToStr(Env.StateDim) + ' dimensions. Build the input as ' +
      'TNNetInput.Create(1, 1, StateDim).');
  if FCritic <> nil then
  begin
    if FCritic.GetLastLayer().Output.Size <> 1 then
      raise Exception.Create('TNeuralPolicyGradTrainer: the critic must end ' +
        'in a scalar head (last layer Output.Size = 1, e.g. ' +
        'TNNetFullConnectLinear(1)). Found Output.Size = ' +
        IntToStr(FCritic.GetLastLayer().Output.Size) + ' (' +
        FCritic.GetLastLayer().ClassName + ').');
    if FCritic.GetFirstLayer().Output.Size <> Env.StateDim then
      raise Exception.Create('TNeuralPolicyGradTrainer: the critic''s input ' +
        'layer (' + FCritic.GetFirstLayer().ClassName + ') has size ' +
        IntToStr(FCritic.GetFirstLayer().Output.Size) + ' but the ' +
        'environment''s states have ' + IntToStr(Env.StateDim) + ' dimensions.');
  end;
end;

// Constructor-time check (no environment needed): the critic must end in a
// single-scalar head, exactly like the reward model in neuraldpo.pas.
procedure TNeuralPolicyGradTrainer.ValidateCriticHead;
begin
  if FCritic.GetLastLayer().Output.Size <> 1 then
    raise Exception.Create('TNeuralPolicyGradTrainer: the critic must end in ' +
      'a scalar head (last layer Output.Size = 1, e.g. ' +
      'TNNetFullConnectLinear(1)). Found Output.Size = ' +
      IntToStr(FCritic.GetLastLayer().Output.Size) + ' (' +
      FCritic.GetLastLayer().ClassName + ').');
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
  if Length(State) < D then
    raise Exception.Create('TNeuralPolicyGradTrainer.Forward: state vector ' +
      'length (' + IntToStr(Length(State)) + ') is smaller than the network ' +
      'input size (' + IntToStr(D) + ').');
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
  FProbFloor := 1e-9;
  FRngState := 777;
  // The action count is the policy's output size, known before any Compute.
  FActionCount := pPolicy.GetLastLayer().Output.Size;
  // K-relative update floor, computed after FActionCount (see Phase 4).
  if FActionCount > 0 then
    FUpdateFloor := Min(0.1, 0.5 / FActionCount)
  else
    FUpdateFloor := 0.1;
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
  if FCritic <> nil then ValidateCriticHead;
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
    // Random action: no forward pass needed, the action count is cached.
    Result := RandInt(FActionCount);
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
  ValidatePolicyAgainstEnv(Env);
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
  //   SkipDeriv=0: pDesired[a] = pi[a] + A/pi[a], pDesired[a'] = pi[a']
  //   SkipDeriv=1: pDesired[a] = pi[a]*(1-A) + A, pDesired[a'] = pi[a']*(1-A)
  // The 1/pi[a] term is numerically harmless even for tiny pi[a]: the softmax
  // Jacobian multiplies the large e_target back down by y_target, so the
  // gradient stays exact and bounded for ANY pi[a] > 0. Steps are nevertheless
  // SKIPPED when pi[a] is at or below UpdateFloor, for a behavioral reason:
  // Phase 1 explores epsilon-greedily, so rare actions are disproportionately
  // epsilon-RANDOM picks whose disadvantage (A < 0) would push their
  // probability UP, systematically corrupting a sharpened policy (verified: an
  // unconditional ProbFloor-style update collapsed Test 5). UpdateFloor
  // defaults to Min(0.1, 0.5/NumActions) instead of the old hardcoded 0.1: the
  // old value silently FROZE any policy whose action count makes the uniform
  // probability 1/NumActions <= 0.1 (e.g. K >= 10 never learns at all), while
  // K-relative thresholds keep uniform initialization above the floor.
  // UpdateFloor must stay > 0 (the division below divides by pi[a] directly).
  FPolicy.SetBatchUpdate(true);
  FPolicy.ClearDeltas;
  FLastPolicyLoss := 0;
  for I := 0 to N - 1 do
  begin
    Pi := Forward(FPolicy, FInput, States[I]);
    D := Length(Pi); A := Actions[I];
    if FPseudoTarget.Size <> D then FPseudoTarget.ReSize(1, 1, D);
    // Diagnostic surrogate loss L = -sum A_t * log pi(a_t|s_t).
    FLastPolicyLoss := FLastPolicyLoss
      - Advantages[I] * pcr_logf(Max(Pi[A], FProbFloor));

    if FSkipDerivSoftmax then
      for J := 0 to D - 1 do
        if J = A then
          FPseudoTarget.FData[J] := Pi[J] * (1.0 - Advantages[I]) + Advantages[I]
        else
          FPseudoTarget.FData[J] := Pi[J] * (1.0 - Advantages[I])
    else
      // NOTE: divide by Pi[J] DIRECTLY (no Max): Pi[J] > FUpdateFloor (>= 0)
      // guarantees a positive divisor, and keeping the expression
      // Single/Single preserves the exact same rounding as the original
      // formula (a Max() here promotes the division to a wider type and the
      // resulting 1-ulp differences cascade into different training
      // trajectories over hundreds of episodes).
      for J := 0 to D - 1 do
        if (J = A) and (Pi[J] > FUpdateFloor) then
          FPseudoTarget.FData[J] := Pi[J] + Advantages[I] / Pi[J]
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
  ValidatePolicyAgainstEnv(Env);
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
// TNeuralPPOTrainer
// ======================================================================

function TNeuralPPOTrainer.Rand(): TNeuralFloat;
begin
  FRngState := (FRngState * 1103515245 + 12345) and $7FFFFFFF;
  Result := FRngState / 2147483647.0;
end;

function TNeuralPPOTrainer.RandInt(AMax: integer): integer;
var U: TNeuralFloat;
begin
  if AMax <= 0 then begin Result := 0; Exit; end;
  U := Rand;
  Result := Trunc(U * AMax);
  if Result >= AMax then Result := AMax - 1;
end;

// Same fail-loudly validation as TNeuralPolicyGradTrainer (softmax head over
// NumActions, input width = StateDim, scalar critic head). See that method
// for the pass-through softmax trap this catches.
procedure TNeuralPPOTrainer.ValidatePolicyAgainstEnv(Env: TNeuralEnv);
var
  FirstLayer, LastLayer: TNNetLayer;
begin
  LastLayer := FPolicy.GetLastLayer();
  if LastLayer.Output.Size <> Env.NumActions then
    raise Exception.Create('TNeuralPPOTrainer: the policy''s last ' +
      'layer (' + LastLayer.ClassName + ') outputs ' +
      IntToStr(LastLayer.Output.Size) + ' values but the environment has ' +
      IntToStr(Env.NumActions) + ' actions. TNNetSoftMax is a pass-through ' +
      'layer: project to the action space with ' +
      'TNNetFullConnectLinear.Create(NumActions) BEFORE the softmax.');
  FirstLayer := FPolicy.GetFirstLayer();
  if FirstLayer.Output.Size <> Env.StateDim then
    raise Exception.Create('TNeuralPPOTrainer: the policy''s input ' +
      'layer (' + FirstLayer.ClassName + ') has size ' +
      IntToStr(FirstLayer.Output.Size) + ' but the environment''s states ' +
      'have ' + IntToStr(Env.StateDim) + ' dimensions. Build the input as ' +
      'TNNetInput.Create(1, 1, StateDim).');
  if FCritic <> nil then
  begin
    if FCritic.GetLastLayer().Output.Size <> 1 then
      raise Exception.Create('TNeuralPPOTrainer: the critic must end ' +
        'in a scalar head (last layer Output.Size = 1, e.g. ' +
        'TNNetFullConnectLinear(1)). Found Output.Size = ' +
        IntToStr(FCritic.GetLastLayer().Output.Size) + ' (' +
        FCritic.GetLastLayer().ClassName + ').');
    if FCritic.GetFirstLayer().Output.Size <> Env.StateDim then
      raise Exception.Create('TNeuralPPOTrainer: the critic''s input ' +
        'layer (' + FCritic.GetFirstLayer().ClassName + ') has size ' +
        IntToStr(FCritic.GetFirstLayer().Output.Size) + ' but the ' +
        'environment''s states have ' + IntToStr(Env.StateDim) + ' dimensions.');
  end;
end;

// Constructor-time check (no environment needed): the critic must end in a
// single-scalar head, exactly like the reward model in neuraldpo.pas.
procedure TNeuralPPOTrainer.ValidateCriticHead;
begin
  if FCritic.GetLastLayer().Output.Size <> 1 then
    raise Exception.Create('TNeuralPPOTrainer: the critic must end in ' +
      'a scalar head (last layer Output.Size = 1, e.g. ' +
      'TNNetFullConnectLinear(1)). Found Output.Size = ' +
      IntToStr(FCritic.GetLastLayer().Output.Size) + ' (' +
      FCritic.GetLastLayer().ClassName + ').');
end;

procedure TNeuralPPOTrainer.DetectSoftmaxVariant;
var
  LastLayer: TNNetLayer;
  StructStr: string;
  ColonPos, SemiPos: integer;
begin
  FSkipDerivSoftmax := false;
  LastLayer := FPolicy.GetLastLayer;
  if not (LastLayer is TNNetPointwiseSoftMax) then
    raise Exception.Create(
      'TNeuralPPOTrainer requires the policy to end in a softmax layer ' +
      '(TNNetSoftMax or TNNetPointwiseSoftMax). Found: ' + LastLayer.ClassName + '.');
  StructStr := LastLayer.SaveStructureToString;
  ColonPos := Pos(':', StructStr);
  SemiPos := Pos(';', StructStr);
  if (ColonPos > 0) and (SemiPos > ColonPos) then
    FSkipDerivSoftmax :=
      StrToIntDef(Copy(StructStr, ColonPos + 1, SemiPos - ColonPos - 1), 0) > 0;
end;

function TNeuralPPOTrainer.Forward(NN: TNNet; AInput: TNNetVolume;
  const State: TNeuralRLState): TNeuralRLState;
var FirstOut, LastOut: TNNetVolume; D, I: integer;
begin
  FirstOut := NN.GetFirstLayer().Output;
  if AInput.Size <> FirstOut.Size then
    AInput.ReSize(FirstOut.SizeX, FirstOut.SizeY, FirstOut.Depth);
  D := FirstOut.Size;
  if Length(State) < D then
    raise Exception.Create('TNeuralPPOTrainer.Forward: state vector ' +
      'length (' + IntToStr(Length(State)) + ') is smaller than the network ' +
      'input size (' + IntToStr(D) + ').');
  for I := 0 to D - 1 do AInput.FData[I] := State[I];
  NN.Compute(AInput);
  LastOut := NN.GetLastLayer().Output;
  SetLength(Result, LastOut.Size);
  for I := 0 to LastOut.Size - 1 do Result[I] := LastOut.FData[I];
end;

procedure TNeuralPPOTrainer.BackpropagateFromLastForward(Target: integer;
  ErrScale: TNeuralFloat);
var
  Y: TNNetVolume;
  PTarget: TNeuralFloat;
begin
  Y := FPolicy.GetLastLayer().Output;
  if FPseudoTarget.Size <> Y.Size then FPseudoTarget.ReSize(Y);
  // ErrScale = dL/d logpi(target). Since d logpi/d logit = (onehot - y),
  //   dL/d logit = ErrScale*(onehot - y) = -ErrScale*(y - onehot),
  // identical to the GRPO plumbing in neuraldpo.pas: pass pDesired so the
  // logit gradient is -ErrScale*(y - onehot).
  if FSkipDerivSoftmax then
  begin
    FPseudoTarget.Copy(Y);
    FPseudoTarget.Mul(1 + ErrScale);
    FPseudoTarget.FData[Target] := FPseudoTarget.FData[Target] - ErrScale;
  end
  else
  begin
    FPseudoTarget.Copy(Y);
    PTarget := Max(Y.FData[Target], FProbFloor);
    FPseudoTarget.FData[Target] := Y.FData[Target] - (ErrScale / PTarget);
  end;
  FPolicy.Backpropagate(FPseudoTarget);
end;

constructor TNeuralPPOTrainer.Create(pPolicy: TNNet; pGamma, pLambda,
  pClipEpsilon: TNeuralFloat; pEpochs, pMiniBatchSize: integer);
begin
  inherited Create;
  FPolicy := pPolicy; FCritic := nil; FOwnsCritic := false;
  FGamma := pGamma; FLambda := pLambda; FClipEpsilon := pClipEpsilon;
  FEpochs := pEpochs; if FEpochs < 1 then FEpochs := 1;
  FMiniBatchSize := pMiniBatchSize; if FMiniBatchSize < 1 then FMiniBatchSize := 1;
  FProbFloor := 1e-9;
  FExploreEpsilon := 0.1;
  FNormalizeAdvantages := true;
  FLastPolicyLoss := 0; FLastValueLoss := 0; FLastClipFraction := 0;
  FRngState := 777;
  // The action count is the policy's output size, known before any Compute.
  FActionCount := pPolicy.GetLastLayer().Output.Size;
  FInput := TNNetVolume.Create;
  FPseudoTarget := TNNetVolume.Create;
  DetectSoftmaxVariant;
end;

constructor TNeuralPPOTrainer.CreateWithCritic(pPolicy, pCritic: TNNet;
  pGamma, pLambda, pClipEpsilon: TNeuralFloat; pEpochs,
  pMiniBatchSize: integer; pOwnsCritic: boolean);
begin
  Create(pPolicy, pGamma, pLambda, pClipEpsilon, pEpochs, pMiniBatchSize);
  FCritic := pCritic;
  FOwnsCritic := pOwnsCritic;
  if FCritic <> nil then ValidateCriticHead;
end;

destructor TNeuralPPOTrainer.Destroy;
begin
  FInput.Free; FPseudoTarget.Free;
  if FOwnsCritic then FCritic.Free;
  inherited Destroy;
end;

function TNeuralPPOTrainer.PolicyDist(const State: TNeuralRLState): TNeuralRLState;
begin
  Result := Forward(FPolicy, FInput, State);
end;

// Samples an action for the rollout and reports the POLICY's log-probability
// of the taken action. Sampling is epsilon-greedy over the current policy
// (see ExploreEpsilon); PPO's ratio + clip window is what keeps the update
// anchored to pi_old and suppresses the influence of off-policy noise - the
// role UpdateFloor plays in TNeuralPolicyGradTrainer.
function TNeuralPPOTrainer.ActWithLogProb(const State: TNeuralRLState;
  out ALogProb: TNeuralFloat): integer;
var
  Dist: TNeuralRLState;
  U, Cum: TNeuralFloat;
  I: integer;
begin
  Dist := PolicyDist(State);
  if (FExploreEpsilon > 0) and (Rand < FExploreEpsilon) then
    Result := RandInt(Length(Dist))
  else
  begin
    U := Rand; Cum := 0;
    Result := High(Dist);
    for I := 0 to High(Dist) do
    begin
      Cum := Cum + Dist[I];
      if U < Cum then begin Result := I; Break; end;
    end;
  end;
  ALogProb := pcr_logf(Max(Dist[Result], FProbFloor));
end;

function TNeuralPPOTrainer.ActGreedy(const State: TNeuralRLState): integer;
var Dist: TNeuralRLState; I, BestI: integer; BestV: TNeuralFloat;
begin
  Dist := PolicyDist(State);
  BestI := 0; BestV := Dist[0];
  for I := 1 to High(Dist) do
    if Dist[I] > BestV then begin BestV := Dist[I]; BestI := I; end;
  Result := BestI;
end;

function TNeuralPPOTrainer.Value(const State: TNeuralRLState): TNeuralFloat;
var V: TNeuralRLState;
begin
  if FCritic = nil then begin Result := 0; Exit; end;
  V := Forward(FCritic, FInput, State);
  Result := V[0];
end;

function TNeuralPPOTrainer.TrainOnEpisodes(Env: TNeuralEnv;
  NumEpisodes, MaxSteps: integer): TNeuralFloat;
var
  States: array of TNeuralRLState;
  Actions: array of integer;
  Rewards: array of TNeuralFloat;
  Ends: array of boolean;          // last transition of each episode
  OldLogPi: array of TNeuralFloat; // log pi_old(a|s) captured at rollout time
  Values: array of TNeuralFloat;   // V(s_i) BEFORE the update
  Returns, Advantages: array of TNeuralFloat;
  Order: array of integer;
  I, N, D, A, J, Epoch, BatchStart, BatchSize, Stop, Idx, Tmp: integer;
  E, Len: integer;
  S: TNeuralRLState;
  R, TotalR, G, GAE, Delta: TNeuralFloat;
  AMean, AStd, AVar, Sum: TNeuralFloat;
  Ratio, RatioClip, Surr1, Surr2, Surr, PGGrad, ErrScale: TNeuralFloat;
  Pi: TNeuralRLState;
  LogPi, V: TNeuralFloat;
  ObjSum, ValueLossSum: TNeuralFloat;
  Clipped, StepCount: integer;
begin
  ValidatePolicyAgainstEnv(Env);
  if NumEpisodes < 1 then NumEpisodes := 1;
  N := 0; TotalR := 0;

  // ---- Phase 1: on-policy rollout over NumEpisodes episodes. --------------
  // All episodes share ONE contiguous batch (standard PPO practice): the GAE
  // recursion in Phase 2 restarts at every episode boundary (Ends), so the
  // mixed buffer stays correct, and the multi-epoch update sees a broader
  // advantage distribution than any single episode provides.
  for E := 1 to NumEpisodes do
  begin
    Env.Reset;
    Env.CopyState(S);
    Len := 0;
    for I := 0 to MaxSteps - 1 do
    begin
      if Length(States) < N + 1 then
      begin
        SetLength(States, (N + 1) * 2);
        SetLength(Actions, (N + 1) * 2);
        SetLength(Rewards, (N + 1) * 2);
        SetLength(Ends, (N + 1) * 2);
        SetLength(OldLogPi, (N + 1) * 2);
      end;
      States[N] := S;
      A := ActWithLogProb(S, LogPi);
      Actions[N] := A; OldLogPi[N] := LogPi;
      R := Env.Step(A); Rewards[N] := R; Ends[N] := Env.Done;
      TotalR := TotalR + R;
      Inc(N); Inc(Len);
      Env.CopyState(S);
      if Env.Done then Break;
    end;
    if (Len > 0) and (N > 0) then Ends[N - 1] := true; // terminal OR horizon cut
  end;
  SetLength(States, N); SetLength(Actions, N); SetLength(Rewards, N);
  SetLength(Ends, N); SetLength(OldLogPi, N);
  SetLength(Values, N); SetLength(Returns, N); SetLength(Advantages, N);
  SetLength(Order, N);
  if N = 0 then begin Result := 0; Exit; end;

  // ---- Phase 2: advantages. ----------------------------------------------
  // With a critic: GAE(lambda). delta_i = r_i + gamma*V(s_{i+1}) - V(s_i),
  // A_i = delta_i + gamma*lambda*A_{i+1}, with the recursion zeroed at every
  // episode boundary and no bootstrap at the horizon truncation (same
  // no-bootstrap convention as TNeuralPolicyGradTrainer). The critic's
  // regression target is the standard PPO one: T_i = A_i + V_old(s_i).
  // Without a critic: Monte-Carlo discounted returns as advantages.
  if FCritic <> nil then
  begin
    for I := 0 to N - 1 do Values[I] := Value(States[I]);
    GAE := 0;
    for I := N - 1 downto 0 do
    begin
      if Ends[I] or (I = N - 1) then
        Delta := Rewards[I] - Values[I]
      else
        Delta := Rewards[I] + FGamma * Values[I + 1] - Values[I];
      GAE := Delta + FGamma * FLambda * GAE;
      Advantages[I] := GAE;
      if Ends[I] then GAE := 0; // advantage chain restarts after a boundary
    end;
    for I := 0 to N - 1 do Returns[I] := Advantages[I] + Values[I];
  end
  else
  begin
    G := 0;
    for I := N - 1 downto 0 do
    begin
      if Ends[I] then G := Rewards[I]
      else G := Rewards[I] + FGamma * G;
      Advantages[I] := G;
    end;
  end;

  // ---- Advantage normalization over the whole batch (standard PPO). -------
  if FNormalizeAdvantages and (N > 1) then
  begin
    Sum := 0;
    for I := 0 to N - 1 do Sum := Sum + Advantages[I];
    AMean := Sum / N;
    AVar := 0;
    for I := 0 to N - 1 do AVar := AVar + Sqr(Advantages[I] - AMean);
    AStd := Sqrt(AVar / N);
    if AStd > 1e-6 then
      for I := 0 to N - 1 do
        Advantages[I] := (Advantages[I] - AMean) / (AStd + 1e-6);
  end;

  // ---- Phase 3: E epochs of clipped-surrogate minibatch updates. ----------
  // For each sampled (s,a): r = pi_new(a|s)/pi_old(a|s) = exp(logpi - logold).
  //   unclipped branch: d min-objective / d logpi = -r*A (ErrScale = -r*A).
  //   clipped branch:   the ratio is saturated and PPO zeroes the gradient.
  // Unlike REINFORCE there is NO 1/pi(a) division anywhere: the pseudo-target
  // plumbing's internal 1/pi[target] is cancelled by the softmax Jacobian
  // (see TNeuralPolicyGradTrainer Phase 4), so no UpdateFloor is needed.
  // Weights update PER MINIBATCH (ClearDeltas -> accumulate -> UpdateWeights):
  // sequential updates within one rollout are what create the policy drift
  // that the clip window guards against; a single end-of-episode update would
  // degenerate PPO into one big REINFORCE step.
  ObjSum := 0; ValueLossSum := 0; Clipped := 0; StepCount := 0;
  FPolicy.SetBatchUpdate(true);
  if FCritic <> nil then FCritic.SetBatchUpdate(true);
  for Epoch := 1 to FEpochs do
  begin
    // Fisher-Yates shuffle of the rollout order for this epoch.
    for I := 0 to N - 1 do Order[I] := I;
    for I := N - 1 downto 1 do
    begin
      J := RandInt(I + 1);
      Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
    end;
    BatchSize := FMiniBatchSize;
    if (BatchSize > N) or (BatchSize < 1) then BatchSize := N;
    BatchStart := 0;
    while BatchStart < N do
    begin
      Stop := BatchStart + BatchSize; if Stop > N then Stop := N;
      // Policy: clipped surrogate gradient per sampled transition.
      FPolicy.ClearDeltas;
      for Idx := BatchStart to Stop - 1 do
      begin
        I := Order[Idx];
        Pi := Forward(FPolicy, FInput, States[I]);
        D := Length(Pi); A := Actions[I];
        if FPseudoTarget.Size <> D then FPseudoTarget.ReSize(1, 1, D);
        LogPi := pcr_logf(Max(Pi[A], FProbFloor));
        Ratio := NeuralExp(LogPi - OldLogPi[I]); // fast, trap-free exp
        Surr1 := Ratio * Advantages[I];
        RatioClip := Ratio;
        if RatioClip > 1 + FClipEpsilon then RatioClip := 1 + FClipEpsilon
        else if RatioClip < 1 - FClipEpsilon then RatioClip := 1 - FClipEpsilon;
        Surr2 := RatioClip * Advantages[I];
        // min(surr1, surr2): when surr2 is the (smaller) clipped branch the
        // ratio is saturated and PPO zeroes the gradient (same convention as
        // TNeuralGRPOTrainer's optional clip in neuraldpo.pas).
        if Surr2 < Surr1 then
        begin
          Surr := Surr2;
          PGGrad := 0;
          Inc(Clipped);
        end
        else
        begin
          Surr := Surr1;
          PGGrad := Ratio * Advantages[I]; // d(r*A)/dlogpi = r*A
        end;
        ObjSum := ObjSum + Surr;
        Inc(StepCount);
        // L = -surrogate  ->  dL/dlogpi = -PGGrad.
        ErrScale := -PGGrad;
        BackpropagateFromLastForward(A, ErrScale);
      end;
      FPolicy.UpdateWeights;
      // Critic: MSE against the GAE value targets, same minibatch schedule.
      if FCritic <> nil then
      begin
        FCritic.ClearDeltas;
        for Idx := BatchStart to Stop - 1 do
        begin
          I := Order[Idx];
          V := Value(States[I]);
          if FPseudoTarget.Size <> 1 then FPseudoTarget.ReSize(1, 1, 1);
          FPseudoTarget.FData[0] := Returns[I];
          FCritic.Backpropagate(FPseudoTarget);
          ValueLossSum := ValueLossSum + Sqr(V - Returns[I]);
        end;
        FCritic.UpdateWeights;
      end;
      BatchStart := Stop;
    end;
  end;
  FPolicy.SetBatchUpdate(false);
  if FCritic <> nil then FCritic.SetBatchUpdate(false);

  if StepCount > 0 then
  begin
    FLastPolicyLoss := -ObjSum / StepCount;
    FLastClipFraction := Clipped / StepCount;
    FLastValueLoss := ValueLossSum / StepCount;
  end
  else
  begin
    FLastPolicyLoss := 0; FLastClipFraction := 0; FLastValueLoss := 0;
  end;
  Result := TotalR;
end;

function TNeuralPPOTrainer.TrainOnEpisode(Env: TNeuralEnv;
  MaxSteps: integer): TNeuralFloat;
begin
  Result := TrainOnEpisodes(Env, 1, MaxSteps);
end;

function TNeuralPPOTrainer.TotalReward(Env: TNeuralEnv;
  MaxSteps: integer): TNeuralFloat;
var I, A: integer; R: TNeuralFloat; S: TNeuralRLState;
begin
  ValidatePolicyAgainstEnv(Env);
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
