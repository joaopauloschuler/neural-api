unit TestNeuralGRPO;
(*
Tests for the GRPO (Group Relative Policy Optimization, DeepSeekMath/R1)
trainer in neuraldpo.pas.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuraldpo;

type
  TTestNeuralGRPO = class(TTestCase)
  private
    FTargetToken: integer;
    function BuildTinyLM(ContextLen, Vocab, Hidden: integer): TNNet;
    // Reward = number of occurrences of FTargetToken in the completion.
    function CountTargetReward(const Completion: array of integer): TNeuralFloat;
    // Probability the policy emits FTargetToken given the prompt (single step).
    function ProbOfTarget(NN: TNNet; const Prompt: array of integer): TNeuralFloat;
  published
    // Advantages are group-normalized: mean ~0, population std ~1.
    procedure TestAdvantagesAreGroupNormalized;
    // A zero-variance group (all rewards equal) yields all-zero advantages.
    procedure TestZeroVarianceGroupZeroAdvantage;
    // Learning-signal test: a few GRPO steps with a "emit the target token"
    // reward must INCREASE the policy probability of the target token.
    procedure TestGRPOIncreasesRewardedTokenProb;
  end;

implementation

const
  csContext = 4;
  csVocab   = 6;
  csHidden  = 12;

function TTestNeuralGRPO.BuildTinyLM(ContextLen, Vocab, Hidden: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(Vocab));
  Result.AddLayer(TNNetSoftMax.Create());
end;

function TTestNeuralGRPO.CountTargetReward(
  const Completion: array of integer): TNeuralFloat;
var
  I: integer;
begin
  Result := 0;
  for I := 0 to High(Completion) do
    if Completion[I] = FTargetToken then Result := Result + 1;
end;

function TTestNeuralGRPO.ProbOfTarget(NN: TNNet;
  const Prompt: array of integer): TNeuralFloat;
var
  Input: TNNetVolume;
  Toks: TNeuralDPOTokenArray;
  I: integer;
begin
  Input := TNNetVolume.Create(NN.GetFirstLayer().Output.SizeX, 1,
    NN.GetFirstLayer().Output.Depth);
  SetLength(Toks, Length(Prompt));
  for I := 0 to High(Prompt) do Toks[I] := Prompt[I];
  Input.OneHotEncodingReversed(Toks);
  NN.Compute(Input);
  Result := NN.GetLastLayer().Output.FData[FTargetToken];
  Input.Free;
end;

procedure TTestNeuralGRPO.TestAdvantagesAreGroupNormalized;
var
  Rewards, Adv: array of TNeuralFloat;
  Mean, Std, S, S2: TNeuralFloat;
  I: integer;
begin
  SetLength(Rewards, 5);
  Rewards[0] := 1; Rewards[1] := 3; Rewards[2] := 2;
  Rewards[3] := 8; Rewards[4] := 6;
  SetLength(Adv, 5);
  TNeuralGRPOTrainer.ComputeAdvantages(Rewards, Adv, {eps=}1e-9, Mean, Std);
  // Mean of advantages ~ 0, population std ~ 1.
  S := 0; S2 := 0;
  for I := 0 to 4 do begin S := S + Adv[I]; S2 := S2 + Sqr(Adv[I]); end;
  AssertTrue('advantage mean must be ~0, got ' + FloatToStr(S / 5),
    Abs(S / 5) < 1e-5);
  AssertTrue('advantage population std must be ~1, got ' +
    FloatToStr(Sqrt(S2 / 5)), Abs(Sqrt(S2 / 5) - 1) < 1e-4);
end;

procedure TTestNeuralGRPO.TestZeroVarianceGroupZeroAdvantage;
var
  Rewards, Adv: array of TNeuralFloat;
  Mean, Std: TNeuralFloat;
  I: integer;
begin
  SetLength(Rewards, 4);
  for I := 0 to 3 do Rewards[I] := 2.5;   // all equal -> zero variance
  SetLength(Adv, 4);
  TNeuralGRPOTrainer.ComputeAdvantages(Rewards, Adv, {eps=}1e-4, Mean, Std);
  AssertTrue('zero-variance group std must be 0', Abs(Std) < 1e-9);
  for I := 0 to 3 do
    AssertTrue('zero-variance group advantage must be 0', Abs(Adv[I]) < 1e-9);
end;

procedure TTestNeuralGRPO.TestGRPOIncreasesRewardedTokenProb;
var
  Policy: TNNet;
  Trainer: TNeuralGRPOTrainer;
  Prompt: TNeuralDPOTokenArray;
  ProbBefore, ProbAfter: TNeuralFloat;
  I: integer;
begin
  RandSeed := 424242;
  FTargetToken := 4;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.2, 0);
  Trainer := TNeuralGRPOTrainer.CreateWithClonedReference(
    Policy, {groupSize=}8, {beta=}0.02);
  Trainer.MaxNewTokens := 3;
  Trainer.Temperature := 1.0;
  Trainer.Reward := @CountTargetReward;

  Prompt := DPOTokens(#1#2);
  ProbBefore := ProbOfTarget(Policy, Prompt);
  for I := 1 to 30 do Trainer.TrainOnPrompt(Prompt);
  ProbAfter := ProbOfTarget(Policy, Prompt);

  AssertTrue('GRPO must increase P(target token): before ' +
    FloatToStr(ProbBefore) + ' after ' + FloatToStr(ProbAfter),
    ProbAfter > ProbBefore + 0.02);
  // Diagnostics must be populated and the mean KL non-negative.
  AssertTrue('mean KL must be >= 0', Trainer.LastMeanKL >= -1e-6);

  Trainer.Free;
  Policy.Free;
end;

initialization
  RegisterTest(TTestNeuralGRPO);
end.
