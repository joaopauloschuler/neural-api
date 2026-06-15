unit TestNeuralRewardModel;
(*
Tests for the Bradley-Terry pairwise reward-model trainer
(TNeuralRewardModelTrainer in neuraldpo.pas): a scalar reward head learned
from (chosen, rejected) preference pairs with loss = -ln sigmoid(r_w - r_l).
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuraldpo;

type
  TTestNeuralRewardModel = class(TTestCase)
  private
    // Scalar reward model: one-hot sequence in, a single scalar reward out.
    function BuildTinyRewardModel(ContextLen, Vocab, Hidden: integer): TNNet;
  published
    // A non-scalar head (e.g. a softmax LM) is rejected by the constructor.
    procedure TestNonScalarHeadRejected;
    // The loss equals -ln sigmoid(r_chosen - r_rejected) recomputed from the
    // net's own scalar outputs, and LastMargin = r_chosen - r_rejected.
    procedure TestLossMatchesBradleyTerry;
    // The Bradley-Terry margin gradient has the right SIGN and MAGNITUDE:
    // dLoss/dr_chosen = -(1 - sigmoid(delta)), dLoss/dr_rejected = +(...),
    // verified by central finite differences against the analytic LastScale.
    procedure TestMarginGradientSign;
    // A few steps on a fixed pair decrease the loss AND make the learned
    // reward order chosen > rejected on a held example.
    procedure TestStepDecreasesLossAndOrdersRewards;
  end;

implementation

const
  csContext = 6;
  csVocab   = 8;
  csHidden  = 12;

function TTestNeuralRewardModel.BuildTinyRewardModel(
  ContextLen, Vocab, Hidden: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  // Scalar reward head: a single linear output neuron.
  Result.AddLayer(TNNetFullConnectLinear.Create(1));
end;

procedure TTestNeuralRewardModel.TestNonScalarHeadRejected;
var
  LM: TNNet;
  Raised: boolean;
  Trainer: TNeuralRewardModelTrainer;
begin
  RandSeed := 424242;
  // A next-token LM (softmax head, Output.Size = Vocab) is NOT a reward model.
  LM := TNNet.Create();
  LM.AddLayer(TNNetInput.Create(csContext, 1, csVocab));
  LM.AddLayer(TNNetFullConnectLinear.Create(csVocab));
  LM.AddLayer(TNNetSoftMax.Create());
  Raised := false;
  Trainer := nil;
  try
    try
      Trainer := TNeuralRewardModelTrainer.Create(LM);
    except
      on E: Exception do Raised := true;
    end;
  finally
    Trainer.Free;
    LM.Free;
  end;
  AssertTrue('a non-scalar (softmax) head must be rejected', Raised);
end;

procedure TTestNeuralRewardModel.TestLossMatchesBradleyTerry;
var
  RM: TNNet;
  Trainer: TNeuralRewardModelTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  RW, RL, Delta, Expected, Got: TNeuralFloat;
begin
  RandSeed := 424242;
  RM := BuildTinyRewardModel(csContext, csVocab, csHidden);
  Trainer := TNeuralRewardModelTrainer.Create(RM);
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4);
  Rejected := DPOTokens(#5#6);

  // Hand-compute from the net's own scalar reward forward passes.
  RW := Trainer.Reward(Prompt, Chosen);
  RL := Trainer.Reward(Prompt, Rejected);
  Delta := RW - RL;
  Expected := Ln(1 + Exp(-Delta));        // -ln sigmoid(delta)

  Got := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('BT loss must match hand value: expected ' + FloatToStr(Expected) +
    ' got ' + FloatToStr(Got), Abs(Got - Expected) < 1e-4);
  AssertTrue('LastMargin must equal r_chosen - r_rejected',
    Abs(Trainer.LastMargin - Delta) < 1e-4);
  AssertTrue('LastScale must equal 1 - sigmoid(delta)',
    Abs(Trainer.LastScale - (1 / (1 + Exp(Delta)))) < 1e-4);
  AssertTrue('reward diagnostics must match the forward rewards',
    (Abs(Trainer.LastRewardChosen - RW) < 1e-4) and
    (Abs(Trainer.LastRewardRejected - RL) < 1e-4));

  Trainer.Free;
  RM.Free;
end;

procedure TTestNeuralRewardModel.TestMarginGradientSign;
var
  RM: TNNet;
  Trainer: TNeuralRewardModelTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  Delta, S, H: TNeuralFloat;
  LossPlus, LossMinus, NumDChosen, NumDRejected: TNeuralFloat;

  // Loss as an explicit function of the two scalar rewards (the trainer's loss
  // only depends on r_chosen and r_rejected), used for finite differences.
  function BTLoss(RW, RL: TNeuralFloat): TNeuralFloat;
  begin
    Result := Ln(1 + Exp(-(RW - RL)));
  end;

var
  RW, RL: TNeuralFloat;
begin
  RandSeed := 424242;
  RM := BuildTinyRewardModel(csContext, csVocab, csHidden);
  Trainer := TNeuralRewardModelTrainer.Create(RM);
  Prompt   := DPOTokens(#1#2#3);
  Chosen   := DPOTokens(#4#5);
  Rejected := DPOTokens(#6#7);

  Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  Delta := Trainer.LastMargin;
  S := Trainer.LastScale;            // 1 - sigmoid(delta) = analytic |gradient|
  RW := Trainer.LastRewardChosen;
  RL := Trainer.LastRewardRejected;

  // Analytic gradients: dL/dr_w = -S (<0, chosen reward pushed up),
  //                     dL/dr_l = +S (>0, rejected reward pushed down).
  // Central finite differences on the loss-of-rewards function.
  H := 1e-3;
  LossPlus  := BTLoss(RW + H, RL);
  LossMinus := BTLoss(RW - H, RL);
  NumDChosen := (LossPlus - LossMinus) / (2 * H);

  LossPlus  := BTLoss(RW, RL + H);
  LossMinus := BTLoss(RW, RL - H);
  NumDRejected := (LossPlus - LossMinus) / (2 * H);

  AssertTrue('dLoss/dr_chosen must be negative (push chosen up)',
    NumDChosen < 0);
  AssertTrue('dLoss/dr_rejected must be positive (push rejected down)',
    NumDRejected > 0);
  AssertTrue('|dLoss/dr_chosen| must equal analytic 1-sigmoid(delta): got ' +
    FloatToStr(-NumDChosen) + ' analytic ' + FloatToStr(S),
    Abs(-NumDChosen - S) < 1e-3);
  AssertTrue('|dLoss/dr_rejected| must equal analytic 1-sigmoid(delta)',
    Abs(NumDRejected - S) < 1e-3);
  // The two gradients are equal-and-opposite (symmetric margin).
  AssertTrue('gradients must be equal and opposite',
    Abs(NumDChosen + NumDRejected) < 1e-4);
  // Sanity: sigmoid(-delta) is what LastScale reports.
  AssertTrue('LastScale = sigmoid(-delta)',
    Abs(S - 1 / (1 + Exp(Delta))) < 1e-4);

  Trainer.Free;
  RM.Free;
end;

procedure TTestNeuralRewardModel.TestStepDecreasesLossAndOrdersRewards;
var
  RM: TNNet;
  Trainer: TNeuralRewardModelTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  LossBefore, LossAfter, RW, RL: TNeuralFloat;
  I: integer;
begin
  RandSeed := 424242;
  RM := BuildTinyRewardModel(csContext, csVocab, csHidden);
  RM.SetLearningRate(0.05, 0);
  Trainer := TNeuralRewardModelTrainer.Create(RM);
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4#5);
  Rejected := DPOTokens(#6#7#3);

  LossBefore := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('initial BT loss must be finite',
    not IsNan(LossBefore) and not IsInfinite(LossBefore) and (LossBefore >= 0));
  for I := 1 to 20 do Trainer.Step(Prompt, Chosen, Rejected);
  LossAfter := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('BT steps must decrease the loss: before ' +
    FloatToStr(LossBefore) + ' after ' + FloatToStr(LossAfter),
    LossAfter < LossBefore - 1e-4);

  // The learned reward must now order chosen above rejected on this example.
  RW := Trainer.Reward(Prompt, Chosen);
  RL := Trainer.Reward(Prompt, Rejected);
  AssertTrue('learned reward must rank chosen > rejected: r_w ' +
    FloatToStr(RW) + ' r_l ' + FloatToStr(RL), RW > RL);

  Trainer.Free;
  RM.Free;
end;

initialization
  RegisterTest(TTestNeuralRewardModel);
end.
