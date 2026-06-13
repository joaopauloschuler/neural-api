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

end.
