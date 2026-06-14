unit neuraldpo;
(*
neuraldpo: Direct Preference Optimization (DPO) for autoregressive LMs.
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Coded by Claude (AI).

--------------------------------------------------------------------------
WHAT THIS UNIT IS
--------------------------------------------------------------------------
Direct Preference Optimization (Rafailov et al. 2023,
https://arxiv.org/abs/2305.18290) fine-tunes a language model directly on
human/synthetic preference pairs (prompt, chosen completion, rejected
completion) WITHOUT training a separate reward model and WITHOUT RL:

  margin = (logpi(chosen|prompt)  - logref(chosen|prompt))
         - (logpi(rejected|prompt) - logref(rejected|prompt))

  loss   = -ln sigmoid(beta * margin)

where logpi(.) is the summed per-token log-probability of the completion
tokens under the trainable POLICY net and logref(.) is the same quantity
under a FROZEN REFERENCE net (usually a clone of the policy taken before
fine-tuning; the KL anchor). beta scales how strongly the policy is pushed
away from the reference.

GRADIENT / ERROR-SIGNAL DERIVATION (what Step() backpropagates)
--------------------------------------------------------------------------
Let s = sigmoid(-beta*margin) (the "how wrong are we" scale; 0.5 when the
margin is 0, ->0 when the pair is already well separated). Then

  dLoss/d logpi(chosen)   = -s*beta
  dLoss/d logpi(rejected) = +s*beta

For each completion token t with target id "target", the model performs one
forward pass on the prefix (prompt + completion[0..t-1]) and yields a
softmax distribution y. Since d log y_target / d logit_i = (onehot_i - y_i),

  dLoss/d logit = +s*beta * (y - onehot)   for CHOSEN tokens    (push up)
  dLoss/d logit = -s*beta * (y - onehot)   for REJECTED tokens  (push down)

i.e. the standard scaled cross-entropy (softmax - onehot) gradient with a
positive sign on chosen and a negative sign on rejected tokens.

This codebase's networks end in a softmax LAYER (probabilities, not
logits), and TNNet.Backpropagate(pDesired) forms the output error as
(Output - pDesired) ON THE SOFTMAX OUTPUT, which the softmax layer then
maps to the logit gradient. Two softmax backward conventions exist here:

 * TNNetSoftMax / TNNetPointwiseSoftMax with SkipBackpropDerivative=0
   (the default, e.g. examples/TinyGPT) applies the FULL softmax Jacobian:
   dL/dlogit_i = y_i*(e_i - sum_j y_j e_j) for output error e. Setting the
   single component e_target = ErrScale / y_target (zero elsewhere) yields
   EXACTLY dL/dlogit = -ErrScale*(y - onehot). So this unit passes
   pDesired = y except pDesired[target] = y_target - ErrScale/y_target,
   with ErrScale = -s*beta for chosen and +s*beta for rejected tokens.

 * TNNetPointwiseSoftMax with SkipBackpropDerivative=1 adds the output
   error UNTOUCHED to the previous layer (the error already IS dL/dlogit).
   There this unit passes pDesired = y - e with
   e = -ErrScale*(y - onehot) directly.

The variant is auto-detected from the last layer's class and serialized
structure. Either way the gradient that reaches the logits is the exact
DPO gradient above.

EXPECTED MODEL SHAPE
--------------------------------------------------------------------------
The policy/reference nets must be "next-token" LMs in the style of
examples/TinyGPT and examples/SimpleNLP:
  input : TNNetInput(ContextLen, 1, VocabSize), one-hot tokens encoded
          with OneHotEncodingReversed (most recent token at x=0);
  output: a SINGLE next-token distribution of size VocabSize produced by
          a final softmax layer.
Each completion token therefore costs one forward pass on its prefix
(prefixes longer than ContextLen are truncated to the most recent
ContextLen tokens - a sliding context window).

API
--------------------------------------------------------------------------
  Trainer := TNeuralDPOTrainer.Create(Policy, Reference, {beta=}0.5);
    (or TNeuralDPOTrainer.CreateWithClonedReference(Policy, 0.5) to clone
    the policy into an owned frozen reference.)
  Loss := Trainer.Step(Prompt, Chosen, Rejected);  // one SGD DPO update
After any Step/ComputeLoss the scalar diagnostics are available:
  Trainer.LastLoss    -ln sigmoid(beta*margin)
  Trainer.LastMargin  the (policy - reference) log-prob margin
  Trainer.LastScale   sigmoid(-beta*margin), the gradient scale
  Trainer.LastPolicyChosenLogProb / LastPolicyRejectedLogProb (and Ref...)
ComputeLoss() is forward-only (no update); AccumulateGradients() performs
forward+backward WITHOUT ClearDeltas/UpdateWeights, for gradient
inspection (the policy must already be in batch-update mode). Weight
updates use the policy's per-layer learning rate/inertia (plain SGD via
TNNet.UpdateWeights); set them with Policy.SetLearningRate beforehand.
DPOTokens('abc') converts a string to char-level token ids.
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, neuralnetwork, neuralvolume;

type
  TNeuralDPOTokenArray = array of integer;

  /// Trainer-level helper implementing the DPO update for next-token LMs.
  // Coded by Claude (AI).
  TNeuralDPOTrainer = class(TObject)
    private
      FPolicy: TNNet;
      FReference: TNNet;
      FOwnsReference: boolean;
      FBeta: TNeuralFloat;
      FProbFloor: TNeuralFloat;
      FSkipDerivSoftmax: boolean;
      FLastLoss: TNeuralFloat;
      FLastMargin: TNeuralFloat;
      FLastScale: TNeuralFloat;
      FLastPolicyChosen, FLastPolicyRejected: TNeuralFloat;
      FLastRefChosen, FLastRefRejected: TNeuralFloat;
      FInput: TNNetVolume;
      FPseudoTarget: TNNetVolume;
      // Concatenates prompt + completion[0..UpToTokenCount-1], keeps the most
      // recent ContextLen tokens and one-hot encodes them into FInput.
      procedure EncodePrefix(NN: TNNet;
        const Prompt, Completion: array of integer; UpToTokenCount: integer);
      // Detects the softmax backward convention of the policy's last layer.
      procedure DetectSoftmaxVariant();
      // Backpropagates ErrScale-scaled (softmax - onehot) logit gradients for
      // every completion token (one forward+backward per token). The policy
      // must be in batch-update mode; weights are NOT updated here.
      procedure BackpropagateCompletion(
        const Prompt, Completion: array of integer; ErrScale: TNeuralFloat);
    public
      constructor Create(pPolicy, pReference: TNNet;
        pBeta: TNeuralFloat = 0.5; pOwnsReference: boolean = false);
      // Clones the policy into an owned, frozen reference net.
      constructor CreateWithClonedReference(pPolicy: TNNet;
        pBeta: TNeuralFloat = 0.5);
      destructor Destroy(); override;

      // Summed log-probability of Completion given Prompt under NN
      // (forward passes only; per-token probabilities floored at ProbFloor).
      function SequenceLogProb(NN: TNNet;
        const Prompt, Completion: array of integer): TNeuralFloat;
      // Forward-only DPO loss for one preference pair; refreshes all Last*
      // diagnostics. Returns -ln sigmoid(beta*margin).
      function ComputeLoss(
        const Prompt, Chosen, Rejected: array of integer): TNeuralFloat;
      // Forward + backward (accumulates deltas; requires the policy to be in
      // batch-update mode; does NOT ClearDeltas nor UpdateWeights). Returns
      // the loss.
      function AccumulateGradients(
        const Prompt, Chosen, Rejected: array of integer): TNeuralFloat;
      // One full DPO SGD step on the pair (batch mode + ClearDeltas +
      // backward on chosen and rejected + UpdateWeights). Returns the loss.
      function Step(
        const Prompt, Chosen, Rejected: array of integer): TNeuralFloat;
      // Policy-only preference margin logpi(chosen) - logpi(rejected); the
      // pair is "correctly ranked" when this is positive.
      function PolicyMargin(
        const Prompt, Chosen, Rejected: array of integer): TNeuralFloat;

      property Beta: TNeuralFloat read FBeta write FBeta;
      property ProbFloor: TNeuralFloat read FProbFloor write FProbFloor;
      property Policy: TNNet read FPolicy;
      property Reference: TNNet read FReference;
      property LastLoss: TNeuralFloat read FLastLoss;
      property LastMargin: TNeuralFloat read FLastMargin;
      property LastScale: TNeuralFloat read FLastScale;
      property LastPolicyChosenLogProb: TNeuralFloat read FLastPolicyChosen;
      property LastPolicyRejectedLogProb: TNeuralFloat read FLastPolicyRejected;
      property LastRefChosenLogProb: TNeuralFloat read FLastRefChosen;
      property LastRefRejectedLogProb: TNeuralFloat read FLastRefRejected;
  end;

  // Reward callback for GRPO: maps a generated completion (token ids, WITHOUT
  // the prompt) to a scalar reward. Higher is better. May close over the
  // prompt if needed. Implemented as a method-of-object reference so it can
  // carry state (e.g. a reward model, a target token, a verifier).
  TNeuralGRPORewardEvent =
    function(const Completion: array of integer): TNeuralFloat of object;

  /// Trainer-level helper implementing the GRPO (Group Relative Policy
  /// Optimization, DeepSeekMath/R1) update for next-token LMs. GRPO is the
  /// RL-from-feedback method that needs NO value/critic network: for each
  /// prompt it samples a GROUP of N completions from the current policy,
  /// scores them with a reward callback, and uses the group-relative
  /// advantage A_i = (r_i - mean(group))/(std(group)+eps) as the
  /// policy-gradient weight, minus a KL penalty against a frozen reference.
  ///
  /// OBJECTIVE (per completion i, summed over its tokens t):
  ///   maximize  A_i * logpi(token_t)  -  beta * KL(pi || ref)
  /// so the loss minimized is
  ///   L = -A_i * logpi(token_t)  +  beta * KL_t
  /// with the DeepSeek k3 UNBIASED per-token KL estimator
  ///   KL_t = exp(logref - logpi) - (logref - logpi) - 1   (>= 0),
  /// whose gradient w.r.t. logpi is  d KL_t/d logpi = 1 - exp(logref - logpi)
  /// = 1 - ref/pi (evaluated at the SAMPLED token only - a per-token, not a
  /// full-distribution, KL surrogate, exactly as in the GRPO paper).
  ///
  /// Therefore the total error signal on the SAMPLED token's log-prob is
  ///   dL/d logpi(token_t) = -A_i + beta*(1 - exp(logref - logpi))
  /// and this maps to the logit gradient  dL/d logit = ErrScale*(y - onehot)
  /// with ErrScale = dL/d logpi(token_t), reusing the DPO softmax-backward
  /// plumbing (BackpropagateCompletion-style).
  ///
  /// PPO CLIP: when ClipEpsilon > 0 the per-token policy-gradient term uses
  /// the clipped surrogate ratio = exp(logpi - logpi_old) (logpi_old captured
  /// when the group was sampled): the token contributes
  ///   min(ratio*A, clip(ratio,1-eps,1+eps)*A)
  /// and its gradient is zeroed when the clip branch is active (the standard
  /// PPO "no gradient outside the trust region" behaviour). With a fresh
  /// sample logpi==logpi_old so ratio=1 on the first inner epoch (set
  /// ClipEpsilon=0 to disable clipping entirely, the default).
  // Coded by Claude (AI).
  TNeuralGRPOTrainer = class(TObject)
    private
      FPolicy: TNNet;
      FReference: TNNet;
      FOwnsReference: boolean;
      FBeta: TNeuralFloat;          // KL penalty weight
      FClipEpsilon: TNeuralFloat;   // PPO clip; 0 disables clipping
      FGroupSize: integer;          // N completions per prompt
      FMaxNewTokens: integer;       // generated length per completion
      FTemperature: TNeuralFloat;   // sampling temperature
      FAdvantageEps: TNeuralFloat;  // std floor in advantage normalization
      FProbFloor: TNeuralFloat;
      FSkipDerivSoftmax: boolean;
      FReward: TNeuralGRPORewardEvent;
      FInput: TNNetVolume;
      FPseudoTarget: TNNetVolume;
      // Diagnostics of the last TrainOnPrompt call.
      FLastMeanReward: TNeuralFloat;
      FLastStdReward: TNeuralFloat;
      FLastMeanKL: TNeuralFloat;
      FLastObjective: TNeuralFloat;
      procedure EncodePrefix(NN: TNNet;
        const Prompt, Completion: array of integer; UpToTokenCount: integer);
      procedure DetectSoftmaxVariant();
      // Pushes one ErrScale-scaled (softmax - onehot) logit gradient for the
      // single token `Target` given the prefix (Prompt + Completion[0..T-1]).
      procedure BackpropagateToken(
        const Prompt, Completion: array of integer; UpToTokenCount,
        Target: integer; ErrScale: TNeuralFloat);
      // Samples one completion from the policy (multinomial w/ temperature).
      // Returns the per-token logpi of the sampled tokens in OldLogProbs.
      procedure SampleCompletion(const Prompt: array of integer;
        out Completion: TNeuralDPOTokenArray;
        out OldLogProbs: array of TNeuralFloat; var SampledLen: integer);
    public
      constructor Create(pPolicy, pReference: TNNet;
        pGroupSize: integer = 8; pBeta: TNeuralFloat = 0.04;
        pOwnsReference: boolean = false);
      constructor CreateWithClonedReference(pPolicy: TNNet;
        pGroupSize: integer = 8; pBeta: TNeuralFloat = 0.04);
      destructor Destroy(); override;

      // Group-normalizes raw rewards into advantages in place-style: fills
      // Advantages[0..N-1] with (r-mean)/(std+eps). A zero-variance group
      // yields all-zero advantages. Also returns the group mean and std.
      class procedure ComputeAdvantages(const Rewards: array of TNeuralFloat;
        var Advantages: array of TNeuralFloat; Eps: TNeuralFloat;
        out Mean, Std: TNeuralFloat);

      // Summed log-prob of Completion given Prompt under NN (forward only).
      function SequenceLogProb(NN: TNNet;
        const Prompt, Completion: array of integer): TNeuralFloat;

      // One full GRPO step on a single prompt: samples FGroupSize completions,
      // scores them with the reward callback, computes group-relative
      // advantages, and does the policy-gradient + KL backward + weight
      // update. Returns the mean group reward (also in LastMeanReward).
      function TrainOnPrompt(const Prompt: array of integer): TNeuralFloat;

      property Beta: TNeuralFloat read FBeta write FBeta;
      property ClipEpsilon: TNeuralFloat read FClipEpsilon write FClipEpsilon;
      property GroupSize: integer read FGroupSize write FGroupSize;
      property MaxNewTokens: integer read FMaxNewTokens write FMaxNewTokens;
      property Temperature: TNeuralFloat read FTemperature write FTemperature;
      property AdvantageEps: TNeuralFloat read FAdvantageEps write FAdvantageEps;
      property ProbFloor: TNeuralFloat read FProbFloor write FProbFloor;
      property Reward: TNeuralGRPORewardEvent read FReward write FReward;
      property Policy: TNNet read FPolicy;
      property Reference: TNNet read FReference;
      property LastMeanReward: TNeuralFloat read FLastMeanReward;
      property LastStdReward: TNeuralFloat read FLastStdReward;
      property LastMeanKL: TNeuralFloat read FLastMeanKL;
      property LastObjective: TNeuralFloat read FLastObjective;
  end;

// Char-level tokenization helper: token i is Ord(S[i+1]).
function DPOTokens(const S: string): TNeuralDPOTokenArray;

implementation

function DPOTokens(const S: string): TNeuralDPOTokenArray;
var
  I: integer;
begin
  SetLength(Result, Length(S));
  for I := 1 to Length(S) do Result[I-1] := Ord(S[I]);
end;

// Numerically stable softplus: ln(1+exp(X)).
function StableSoftplus(X: TNeuralFloat): TNeuralFloat;
begin
  if X > 0
  then Result := X + Ln(1 + Exp(-X))
  else Result := Ln(1 + Exp(X));
end;

// Numerically stable sigmoid.
function StableSigmoid(X: TNeuralFloat): TNeuralFloat;
begin
  if X >= 0
  then Result := 1 / (1 + Exp(-X))
  else Result := Exp(X) / (1 + Exp(X));
end;

{ TNeuralDPOTrainer }

constructor TNeuralDPOTrainer.Create(pPolicy, pReference: TNNet;
  pBeta: TNeuralFloat; pOwnsReference: boolean);
begin
  inherited Create();
  FPolicy := pPolicy;
  FReference := pReference;
  FOwnsReference := pOwnsReference;
  FBeta := pBeta;
  FProbFloor := 1e-9;
  FLastLoss := 0; FLastMargin := 0; FLastScale := 0.5;
  FLastPolicyChosen := 0; FLastPolicyRejected := 0;
  FLastRefChosen := 0; FLastRefRejected := 0;
  FInput := TNNetVolume.Create();
  FPseudoTarget := TNNetVolume.Create();
  DetectSoftmaxVariant();
end;

constructor TNeuralDPOTrainer.CreateWithClonedReference(pPolicy: TNNet;
  pBeta: TNeuralFloat);
begin
  Create(pPolicy, pPolicy.Clone(), pBeta, {pOwnsReference=}true);
end;

destructor TNeuralDPOTrainer.Destroy();
begin
  FPseudoTarget.Free;
  FInput.Free;
  if FOwnsReference then FReference.Free;
  inherited Destroy();
end;

procedure TNeuralDPOTrainer.DetectSoftmaxVariant();
var
  LastLayer: TNNetLayer;
  StructStr: string;
  ColonPos, SemiPos: integer;
begin
  FSkipDerivSoftmax := false;
  LastLayer := FPolicy.GetLastLayer();
  if not (LastLayer is TNNetPointwiseSoftMax) then
  begin
    raise Exception.Create(
      'TNeuralDPOTrainer requires the policy to end in a softmax layer ' +
      '(TNNetSoftMax or TNNetPointwiseSoftMax). Found: ' +
      LastLayer.ClassName + '.');
  end;
  // FStruct[0] of TNNetPointwiseSoftMax holds SkipBackpropDerivative; it is
  // the first integer in the serialized structure 'ClassName:S0;S1;...'.
  StructStr := LastLayer.SaveStructureToString();
  ColonPos := Pos(':', StructStr);
  SemiPos := Pos(';', StructStr);
  if (ColonPos > 0) and (SemiPos > ColonPos) then
  begin
    FSkipDerivSoftmax :=
      StrToIntDef(Copy(StructStr, ColonPos+1, SemiPos-ColonPos-1), 0) > 0;
  end;
end;

procedure TNeuralDPOTrainer.EncodePrefix(NN: TNNet;
  const Prompt, Completion: array of integer; UpToTokenCount: integer);
var
  Prefix: TNeuralDPOTokenArray;
  PrefixLen, ContextLen, StartPos, I: integer;
  FirstLayerOutput: TNNetVolume;
begin
  FirstLayerOutput := NN.GetFirstLayer().Output;
  if FInput.Size <> FirstLayerOutput.Size then FInput.ReSize(FirstLayerOutput);
  ContextLen := FirstLayerOutput.SizeX;
  PrefixLen := Length(Prompt) + UpToTokenCount;
  // Sliding window: keep only the most recent ContextLen tokens.
  StartPos := Max(0, PrefixLen - ContextLen);
  SetLength(Prefix, PrefixLen - StartPos);
  for I := StartPos to PrefixLen - 1 do
  begin
    if I < Length(Prompt)
    then Prefix[I - StartPos] := Prompt[I]
    else Prefix[I - StartPos] := Completion[I - Length(Prompt)];
  end;
  FInput.OneHotEncodingReversed(Prefix);
end;

function TNeuralDPOTrainer.SequenceLogProb(NN: TNNet;
  const Prompt, Completion: array of integer): TNeuralFloat;
var
  T: integer;
  P: TNeuralFloat;
begin
  Result := 0;
  for T := 0 to High(Completion) do
  begin
    EncodePrefix(NN, Prompt, Completion, T);
    NN.Compute(FInput);
    P := NN.GetLastLayer().Output.FData[Completion[T]];
    Result := Result + Ln(Max(P, FProbFloor));
  end;
end;

function TNeuralDPOTrainer.ComputeLoss(
  const Prompt, Chosen, Rejected: array of integer): TNeuralFloat;
var
  Z: TNeuralFloat;
begin
  FLastPolicyChosen   := SequenceLogProb(FPolicy,    Prompt, Chosen);
  FLastPolicyRejected := SequenceLogProb(FPolicy,    Prompt, Rejected);
  FLastRefChosen      := SequenceLogProb(FReference, Prompt, Chosen);
  FLastRefRejected    := SequenceLogProb(FReference, Prompt, Rejected);
  FLastMargin := (FLastPolicyChosen - FLastRefChosen)
               - (FLastPolicyRejected - FLastRefRejected);
  Z := FBeta * FLastMargin;
  FLastLoss := StableSoftplus(-Z);   // -ln sigmoid(z) = softplus(-z)
  FLastScale := StableSigmoid(-Z);   // gradient scale s
  Result := FLastLoss;
end;

procedure TNeuralDPOTrainer.BackpropagateCompletion(
  const Prompt, Completion: array of integer; ErrScale: TNeuralFloat);
var
  T, I, Target: integer;
  Y: TNNetVolume;
  PTarget: TNeuralFloat;
begin
  for T := 0 to High(Completion) do
  begin
    Target := Completion[T];
    EncodePrefix(FPolicy, Prompt, Completion, T);
    FPolicy.Compute(FInput);
    Y := FPolicy.GetLastLayer().Output;
    if FPseudoTarget.Size <> Y.Size then FPseudoTarget.ReSize(Y);
    // TNNet.Backpropagate(pDesired) sets the output error e = Y - pDesired,
    // so pDesired = Y - e for the error signal e we want (see unit header).
    if FSkipDerivSoftmax then
    begin
      // e must equal dL/dlogit = -ErrScale*(y - onehot) directly.
      for I := 0 to Y.Size - 1 do
        FPseudoTarget.FData[I] := Y.FData[I] * (1 + ErrScale);
      FPseudoTarget.FData[Target] := FPseudoTarget.FData[Target] - ErrScale;
    end
    else
    begin
      // Full-Jacobian softmax: e_target = ErrScale / y_target (0 elsewhere)
      // maps through the Jacobian to dL/dlogit = -ErrScale*(y - onehot).
      FPseudoTarget.Copy(Y);
      PTarget := Max(Y.FData[Target], FProbFloor);
      FPseudoTarget.FData[Target] := Y.FData[Target] - (ErrScale / PTarget);
    end;
    FPolicy.Backpropagate(FPseudoTarget);
  end;
end;

function TNeuralDPOTrainer.AccumulateGradients(
  const Prompt, Chosen, Rejected: array of integer): TNeuralFloat;
var
  ErrScale: TNeuralFloat;
begin
  Result := ComputeLoss(Prompt, Chosen, Rejected);
  ErrScale := FLastScale * FBeta;
  // dL/dlogit = +s*beta*(y - onehot) on chosen tokens (BackpropagateCompletion
  // produces -ErrScale*(y - onehot)) and the opposite sign on rejected ones.
  BackpropagateCompletion(Prompt, Chosen,   -ErrScale);
  BackpropagateCompletion(Prompt, Rejected, +ErrScale);
end;

function TNeuralDPOTrainer.Step(
  const Prompt, Chosen, Rejected: array of integer): TNeuralFloat;
begin
  // Batch mode: the two backward passes accumulate deltas against the SAME
  // weights; per-sample mode would update weights inside Backpropagate.
  FPolicy.SetBatchUpdate(true);
  FPolicy.ClearDeltas();
  Result := AccumulateGradients(Prompt, Chosen, Rejected);
  FPolicy.UpdateWeights();
  FPolicy.SetBatchUpdate(false);
end;

function TNeuralDPOTrainer.PolicyMargin(
  const Prompt, Chosen, Rejected: array of integer): TNeuralFloat;
begin
  Result := SequenceLogProb(FPolicy, Prompt, Chosen)
          - SequenceLogProb(FPolicy, Prompt, Rejected);
end;

{ TNeuralGRPOTrainer }

constructor TNeuralGRPOTrainer.Create(pPolicy, pReference: TNNet;
  pGroupSize: integer; pBeta: TNeuralFloat; pOwnsReference: boolean);
begin
  inherited Create();
  FPolicy := pPolicy;
  FReference := pReference;
  FOwnsReference := pOwnsReference;
  FGroupSize := pGroupSize;
  FBeta := pBeta;
  FClipEpsilon := 0.0;          // clipping off by default (fresh on-policy)
  FMaxNewTokens := 4;
  FTemperature := 1.0;
  FAdvantageEps := 1e-4;
  FProbFloor := 1e-9;
  FReward := nil;
  FInput := TNNetVolume.Create();
  FPseudoTarget := TNNetVolume.Create();
  FLastMeanReward := 0; FLastStdReward := 0;
  FLastMeanKL := 0; FLastObjective := 0;
  DetectSoftmaxVariant();
end;

constructor TNeuralGRPOTrainer.CreateWithClonedReference(pPolicy: TNNet;
  pGroupSize: integer; pBeta: TNeuralFloat);
begin
  Create(pPolicy, pPolicy.Clone(), pGroupSize, pBeta, {pOwnsReference=}true);
end;

destructor TNeuralGRPOTrainer.Destroy();
begin
  FPseudoTarget.Free;
  FInput.Free;
  if FOwnsReference then FReference.Free;
  inherited Destroy();
end;

procedure TNeuralGRPOTrainer.DetectSoftmaxVariant();
var
  LastLayer: TNNetLayer;
  StructStr: string;
  ColonPos, SemiPos: integer;
begin
  FSkipDerivSoftmax := false;
  LastLayer := FPolicy.GetLastLayer();
  if not (LastLayer is TNNetPointwiseSoftMax) then
    raise Exception.Create(
      'TNeuralGRPOTrainer requires the policy to end in a softmax layer ' +
      '(TNNetSoftMax or TNNetPointwiseSoftMax). Found: ' +
      LastLayer.ClassName + '.');
  StructStr := LastLayer.SaveStructureToString();
  ColonPos := Pos(':', StructStr);
  SemiPos := Pos(';', StructStr);
  if (ColonPos > 0) and (SemiPos > ColonPos) then
    FSkipDerivSoftmax :=
      StrToIntDef(Copy(StructStr, ColonPos+1, SemiPos-ColonPos-1), 0) > 0;
end;

procedure TNeuralGRPOTrainer.EncodePrefix(NN: TNNet;
  const Prompt, Completion: array of integer; UpToTokenCount: integer);
var
  Prefix: TNeuralDPOTokenArray;
  PrefixLen, ContextLen, StartPos, I: integer;
  FirstLayerOutput: TNNetVolume;
begin
  FirstLayerOutput := NN.GetFirstLayer().Output;
  if FInput.Size <> FirstLayerOutput.Size then FInput.ReSize(FirstLayerOutput);
  ContextLen := FirstLayerOutput.SizeX;
  PrefixLen := Length(Prompt) + UpToTokenCount;
  StartPos := Max(0, PrefixLen - ContextLen);
  SetLength(Prefix, PrefixLen - StartPos);
  for I := StartPos to PrefixLen - 1 do
  begin
    if I < Length(Prompt)
    then Prefix[I - StartPos] := Prompt[I]
    else Prefix[I - StartPos] := Completion[I - Length(Prompt)];
  end;
  FInput.OneHotEncodingReversed(Prefix);
end;

function TNeuralGRPOTrainer.SequenceLogProb(NN: TNNet;
  const Prompt, Completion: array of integer): TNeuralFloat;
var
  T: integer;
  P: TNeuralFloat;
begin
  Result := 0;
  for T := 0 to High(Completion) do
  begin
    EncodePrefix(NN, Prompt, Completion, T);
    NN.Compute(FInput);
    P := NN.GetLastLayer().Output.FData[Completion[T]];
    Result := Result + Ln(Max(P, FProbFloor));
  end;
end;

procedure TNeuralGRPOTrainer.SampleCompletion(const Prompt: array of integer;
  out Completion: TNeuralDPOTokenArray;
  out OldLogProbs: array of TNeuralFloat; var SampledLen: integer);
var
  T, I, Vocab, Picked: integer;
  Y: TNNetVolume;
  R, Acc, Sum, MaxLogit, P, InvTemp: TNeuralFloat;
  Probs: array of TNeuralFloat;
begin
  SetLength(Completion, FMaxNewTokens);
  SampledLen := FMaxNewTokens;
  Vocab := FPolicy.GetLastLayer().Output.Size;
  SetLength(Probs, Vocab);
  if FTemperature > 0 then InvTemp := 1.0 / FTemperature else InvTemp := 1.0;
  for T := 0 to FMaxNewTokens - 1 do
  begin
    EncodePrefix(FPolicy, Prompt, Completion, T);
    FPolicy.Compute(FInput);
    Y := FPolicy.GetLastLayer().Output;
    // Re-shape the softmax probabilities by temperature: p_i^(1/Tau), renorm.
    // (The last layer already outputs probabilities, so apply temperature in
    // log space: logit' = ln(p)/Tau.)
    MaxLogit := -1e30;
    for I := 0 to Vocab - 1 do
    begin
      P := Ln(Max(Y.FData[I], FProbFloor)) * InvTemp;
      Probs[I] := P;
      if P > MaxLogit then MaxLogit := P;
    end;
    Sum := 0;
    for I := 0 to Vocab - 1 do
    begin
      Probs[I] := Exp(Probs[I] - MaxLogit);
      Sum := Sum + Probs[I];
    end;
    R := Random * Sum;
    Acc := 0; Picked := Vocab - 1;
    for I := 0 to Vocab - 1 do
    begin
      Acc := Acc + Probs[I];
      if R <= Acc then begin Picked := I; Break; end;
    end;
    Completion[T] := Picked;
    // OldLogProb is the UNTEMPERED policy log-prob of the picked token (the
    // ratio in PPO is against the actual sampling policy log-prob; we store
    // the model's own log p(token) so logpi==logpi_old on the first epoch).
    OldLogProbs[T] := Ln(Max(Y.FData[Picked], FProbFloor));
  end;
end;

class procedure TNeuralGRPOTrainer.ComputeAdvantages(
  const Rewards: array of TNeuralFloat; var Advantages: array of TNeuralFloat;
  Eps: TNeuralFloat; out Mean, Std: TNeuralFloat);
var
  I, N: integer;
  S, Var_: TNeuralFloat;
begin
  N := Length(Rewards);
  Mean := 0; Std := 0;
  if N = 0 then Exit;
  S := 0;
  for I := 0 to N - 1 do S := S + Rewards[I];
  Mean := S / N;
  Var_ := 0;
  for I := 0 to N - 1 do Var_ := Var_ + Sqr(Rewards[I] - Mean);
  Var_ := Var_ / N;            // population variance (GRPO uses /N)
  Std := Sqrt(Var_);
  for I := 0 to N - 1 do
    Advantages[I] := (Rewards[I] - Mean) / (Std + Eps);
end;

procedure TNeuralGRPOTrainer.BackpropagateToken(
  const Prompt, Completion: array of integer; UpToTokenCount,
  Target: integer; ErrScale: TNeuralFloat);
var
  I: integer;
  Y: TNNetVolume;
  PTarget: TNeuralFloat;
begin
  EncodePrefix(FPolicy, Prompt, Completion, UpToTokenCount);
  FPolicy.Compute(FInput);
  Y := FPolicy.GetLastLayer().Output;
  if FPseudoTarget.Size <> Y.Size then FPseudoTarget.ReSize(Y);
  // ErrScale = dL/d logpi(target). Since d logpi/d logit = (onehot - y),
  //   dL/d logit = ErrScale*(onehot - y) = -ErrScale*(y - onehot),
  // identical to the DPO plumbing: pass pDesired so the logit gradient is
  // -ErrScale*(y - onehot). (See the DPO unit header for both softmax
  // backward conventions.)
  if FSkipDerivSoftmax then
  begin
    for I := 0 to Y.Size - 1 do
      FPseudoTarget.FData[I] := Y.FData[I] * (1 + ErrScale);
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

function TNeuralGRPOTrainer.TrainOnPrompt(
  const Prompt: array of integer): TNeuralFloat;
var
  G, T, Len, MaxLen: integer;
  Completions: array of TNeuralDPOTokenArray;
  OldLP: array of array of TNeuralFloat;
  Lengths: array of integer;
  Rewards, Advantages: array of TNeuralFloat;
  Mean, Std, A, LogPi, LogRef, KLt, PGGrad, KLGrad, ErrScale: TNeuralFloat;
  Ratio, RatioClip, Surr1, Surr2: TNeuralFloat;
  P, PRef: TNeuralFloat;
  Y: TNNetVolume;
  TotalKL, TotalObj: TNeuralFloat;
  TokenCount: integer;
begin
  if not Assigned(FReward) then
    raise Exception.Create('TNeuralGRPOTrainer: Reward callback not assigned.');
  if FGroupSize < 1 then FGroupSize := 1;

  SetLength(Completions, FGroupSize);
  SetLength(OldLP, FGroupSize);
  SetLength(Lengths, FGroupSize);
  SetLength(Rewards, FGroupSize);
  SetLength(Advantages, FGroupSize);

  // ---- 1) Sample the group + score each completion. -----------------------
  for G := 0 to FGroupSize - 1 do
  begin
    SetLength(OldLP[G], FMaxNewTokens);
    MaxLen := FMaxNewTokens;
    SampleCompletion(Prompt, Completions[G], OldLP[G], MaxLen);
    Lengths[G] := MaxLen;
    Rewards[G] := FReward(Completions[G]);
  end;

  // ---- 2) Group-relative advantages. --------------------------------------
  ComputeAdvantages(Rewards, Advantages, FAdvantageEps, Mean, Std);
  FLastMeanReward := Mean;
  FLastStdReward := Std;

  // ---- 3) Policy-gradient + KL backward over every sampled token. ---------
  FPolicy.SetBatchUpdate(true);
  FPolicy.ClearDeltas();
  TotalKL := 0; TotalObj := 0; TokenCount := 0;
  for G := 0 to FGroupSize - 1 do
  begin
    A := Advantages[G];
    Len := Lengths[G];
    for T := 0 to Len - 1 do
    begin
      // Current policy log-prob of the sampled token (forward pass).
      EncodePrefix(FPolicy, Prompt, Completions[G], T);
      FPolicy.Compute(FInput);
      Y := FPolicy.GetLastLayer().Output;
      P := Max(Y.FData[Completions[G][T]], FProbFloor);
      LogPi := Ln(P);
      // Reference log-prob (frozen) for the KL estimator.
      EncodePrefix(FReference, Prompt, Completions[G], T);
      FReference.Compute(FInput);
      PRef := Max(FReference.GetLastLayer().Output.FData[Completions[G][T]],
                  FProbFloor);
      LogRef := Ln(PRef);

      // DeepSeek k3 unbiased per-token KL estimator (>= 0):
      //   KL_t = exp(logref-logpi) - (logref-logpi) - 1
      // d KL_t / d logpi = 1 - exp(logref - logpi).
      KLt := Exp(LogRef - LogPi) - (LogRef - LogPi) - 1;
      KLGrad := 1 - Exp(LogRef - LogPi);
      TotalKL := TotalKL + KLt;

      // Policy-gradient term with optional PPO clip.
      if FClipEpsilon > 0 then
      begin
        Ratio := Exp(LogPi - OldLP[G][T]);
        Surr1 := Ratio * A;
        RatioClip := Ratio;
        if RatioClip > 1 + FClipEpsilon then RatioClip := 1 + FClipEpsilon
        else if RatioClip < 1 - FClipEpsilon then RatioClip := 1 - FClipEpsilon;
        Surr2 := RatioClip * A;
        // min(surr1, surr2): when surr2 is the (smaller) clipped branch the
        // ratio is saturated and PPO zeroes the gradient.
        if Surr2 < Surr1 then PGGrad := 0     // clipped: no PG gradient
        else PGGrad := Ratio * A;             // d(ratio*A)/dlogpi = ratio*A
        if Surr1 < Surr2 then TotalObj := TotalObj + Surr1
        else TotalObj := TotalObj + Surr2;
      end
      else
      begin
        PGGrad := A;                          // d(A*logpi)/dlogpi = A
        TotalObj := TotalObj + A * LogPi;
      end;

      // L = -PG_objective + beta*KL  ->  dL/dlogpi = -PGGrad + beta*KLGrad.
      // Logit gradient = ErrScale*(y - onehot) with ErrScale = dL/dlogpi.
      ErrScale := -PGGrad + FBeta * KLGrad;
      BackpropagateToken(Prompt, Completions[G], T, Completions[G][T], ErrScale);
      Inc(TokenCount);
    end;
  end;
  FPolicy.UpdateWeights();
  FPolicy.SetBatchUpdate(false);

  if TokenCount > 0 then FLastMeanKL := TotalKL / TokenCount else FLastMeanKL := 0;
  FLastObjective := TotalObj;
  Result := FLastMeanReward;
end;

end.
