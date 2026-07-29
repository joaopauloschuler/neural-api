unit TestNeuralDPO;
(*
Tests for neuraldpo.pas: Direct Preference Optimization (DPO) trainer.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuraldpo;

type
  TTestNeuralDPO = class(TTestCase)
  private
    // Tiny next-token LM: Input(Context,1,Vocab) -> FC ReLU -> FC -> SoftMax.
    function BuildTinyLM(ContextLen, Vocab, Hidden: integer): TNNet;
    // Sets every weight AND bias to zero -> logits all zero -> uniform softmax.
    procedure ZeroAllWeights(NN: TNNet);
    // Runs ComputeLoss (Mode=0) or Step (Mode=1) on the pair and returns the
    // raised exception message, or '' when the call succeeded.
    function CaptureError(Trainer: TNeuralDPOTrainer; Mode: integer;
      const Prompt, Chosen, Rejected: TNeuralDPOTokenArray): string;
  published
    // Fixed, hand-knowable weights (all-zero except the LM-head biases) make
    // the next-token distribution input-independent and analytic:
    // p_i = exp(b_i)/sum_j exp(b_j). Both completions then have hand-computed
    // log-probs, the policy equals the cloned reference, so margin=0 and the
    // DPO loss is EXACTLY ln(2) with gradient scale EXACTLY 0.5.
    procedure TestUniformNetLossIsLn2;
    // Loss for a random fixed-weight net matches the value recomputed by hand
    // from independently-read per-token softmax probabilities.
    procedure TestLossMatchesHandComputed;
    // A few DPO steps must increase the policy log-prob margin
    // logpi(chosen) - logpi(rejected) on a toy pair.
    procedure TestStepIncreasesMargin;
    // beta=0 -> scale sigmoid(0)=0.5, loss=ln 2, zero gradient (weights do
    // not move on Step).
    procedure TestBetaZeroScaleHalfAndNoUpdate;
    // The gradient sign flips between chosen and rejected: one step pushes
    // P(chosen) up and P(rejected) down; swapping roles flips the movement.
    procedure TestGradientSignFlipsBetweenChosenAndRejected;
    // Finite-difference check of dLoss/dWeight against the implemented
    // backward (LR=1, inertia=0, batch mode: gradient = -Delta).
    procedure TestFiniteDifferenceGradient;
    // A completion token id >= VocabSize indexes past the end of the
    // vocabulary-sized output distribution: an out-of-bounds READ in
    // SequenceLogProb and an out-of-bounds WRITE into the pseudo-target volume
    // in BackpropagateCompletion. Both must be rejected at the boundary.
    procedure TestOutOfVocabTargetRejectedOnComputeLoss;
    procedure TestOutOfVocabTargetRejectedOnStep;
    // The skip-backprop-derivative softmax convention takes the other write
    // branch of BackpropagateCompletion; it must be guarded too.
    procedure TestOutOfVocabTargetRejectedOnSkipDerivativeStep;
    // Prompt tokens are one-hot input channels; an out-of-range id there is
    // silently dropped by the encoder, corrupting the training signal.
    procedure TestOutOfVocabPromptRejected;
    // Negative ids index before the start of the output volume.
    procedure TestNegativeTokenRejected;
    // Every preference-loss mode routes through the same read/write sites.
    procedure TestAllLossModesRejectOutOfVocabTarget;
    // The guard must not reject the largest valid id (VocabSize-1).
    procedure TestLastValidTokenAccepted;
  end;

implementation

const
  csContext = 4;
  csVocab   = 8;
  csHidden  = 10;

function TTestNeuralDPO.BuildTinyLM(ContextLen, Vocab, Hidden: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(Vocab));
  Result.AddLayer(TNNetSoftMax.Create());
end;

procedure TTestNeuralDPO.ZeroAllWeights(NN: TNNet);
var
  L, N: integer;
begin
  for L := 0 to NN.Layers.Count - 1 do
    for N := 0 to NN.Layers[L].Neurons.Count - 1 do
    begin
      NN.Layers[L].Neurons[N].Weights.Fill(0);
      NN.Layers[L].Neurons[N].BiasWeight := 0;
    end;
end;

procedure TTestNeuralDPO.TestUniformNetLossIsLn2;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Loss, LogSumExp, ExpectedChosenLogProb: TNeuralFloat;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  I: integer;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  // Zero weights everywhere; give the LM head fixed analytic biases so the
  // logits are b_i = 0.1*i regardless of the input (the hidden ReLU layer
  // outputs all zeros) and the softmax is hand-computable.
  ZeroAllWeights(Policy);
  for I := 0 to csVocab - 1 do
    Policy.Layers[2].Neurons[I].BiasWeight := 0.1 * I;
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.7);
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4);
  Rejected := DPOTokens(#5#6);
  Loss := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  // Policy = reference -> margin = 0 -> loss = ln 2 for ANY beta.
  AssertTrue('fixed-weight DPO loss must be ln(2), got ' +
    FloatToStr(Loss), Abs(Loss - Ln(2)) < 1e-5);
  AssertTrue('fixed-weight margin must be 0',
    Abs(Trainer.LastMargin) < 1e-5);
  AssertTrue('fixed-weight scale must be 0.5',
    Abs(Trainer.LastScale - 0.5) < 1e-5);
  // Hand-computed: ln p_3 + ln p_4 = 0.1*(3+4) - 2*ln(sum_i exp(0.1*i)).
  LogSumExp := 0;
  for I := 0 to csVocab - 1 do LogSumExp := LogSumExp + Exp(0.1 * I);
  LogSumExp := Ln(LogSumExp);
  ExpectedChosenLogProb := 0.1 * (3 + 4) - 2 * LogSumExp;
  AssertTrue('chosen log-prob must match the analytic softmax value: ' +
    'expected ' + FloatToStr(ExpectedChosenLogProb) + ' got ' +
    FloatToStr(Trainer.LastPolicyChosenLogProb),
    Abs(Trainer.LastPolicyChosenLogProb - ExpectedChosenLogProb) < 1e-4);
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralDPO.TestLossMatchesHandComputed;
var
  Policy, Reference: TNNet;
  Trainer: TNeuralDPOTrainer;
  Input: TNNetVolume;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  Beta, Margin, ExpectedLoss, GotLoss: TNeuralFloat;

  // Independent re-implementation: per-token forward + ln(prob) sum.
  function HandLogProb(NN: TNNet; const Completion: TNeuralDPOTokenArray): TNeuralFloat;
  var
    T, I, PrefLen: integer;
    Prefix: TNeuralDPOTokenArray;
  begin
    Result := 0;
    for T := 0 to High(Completion) do
    begin
      PrefLen := Length(Prompt) + T;
      SetLength(Prefix, PrefLen);
      for I := 0 to PrefLen - 1 do
        if I < Length(Prompt)
        then Prefix[I] := Prompt[I]
        else Prefix[I] := Completion[I - Length(Prompt)];
      Input.OneHotEncodingReversed(Prefix);
      NN.Compute(Input);
      Result := Result + Ln(NN.GetLastLayer().Output.FData[Completion[T]]);
    end;
  end;

begin
  RandSeed := 424242;
  Beta := 0.35;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Reference := BuildTinyLM(csContext, csVocab, csHidden); // different weights
  Trainer := TNeuralDPOTrainer.Create(Policy, Reference, Beta);
  Input := TNNetVolume.Create(csContext, 1, csVocab);
  Prompt   := DPOTokens(#1#2#3);
  Chosen   := DPOTokens(#4#5);
  Rejected := DPOTokens(#6#7);

  Margin := (HandLogProb(Policy, Chosen) - HandLogProb(Reference, Chosen))
          - (HandLogProb(Policy, Rejected) - HandLogProb(Reference, Rejected));
  ExpectedLoss := Ln(1 + Exp(-Beta * Margin));
  GotLoss := Trainer.ComputeLoss(Prompt, Chosen, Rejected);

  AssertTrue('DPO loss must match hand-computed -ln sigmoid(beta*margin): ' +
    'expected ' + FloatToStr(ExpectedLoss) + ' got ' + FloatToStr(GotLoss),
    Abs(GotLoss - ExpectedLoss) < 1e-5);
  AssertTrue('DPO margin must match hand-computed margin',
    Abs(Trainer.LastMargin - Margin) < 1e-5);
  AssertTrue('DPO scale must be sigmoid(-beta*margin)',
    Abs(Trainer.LastScale - 1/(1 + Exp(Beta*Margin))) < 1e-5);

  Input.Free;
  Trainer.Free;
  Reference.Free;
  Policy.Free;
end;

procedure TTestNeuralDPO.TestStepIncreasesMargin;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  MarginBefore, MarginAfter: TNeuralFloat;
  I: integer;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.05, 0);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4#5);
  Rejected := DPOTokens(#6#7#3);
  MarginBefore := Trainer.PolicyMargin(Prompt, Chosen, Rejected);
  for I := 1 to 10 do Trainer.Step(Prompt, Chosen, Rejected);
  MarginAfter := Trainer.PolicyMargin(Prompt, Chosen, Rejected);
  AssertTrue('DPO steps must increase the policy log-prob margin: before ' +
    FloatToStr(MarginBefore) + ' after ' + FloatToStr(MarginAfter),
    MarginAfter > MarginBefore + 0.01);
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralDPO.TestBetaZeroScaleHalfAndNoUpdate;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  ProbeBefore, ProbeAfter, Loss: TNeuralFloat;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.1, 0);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.0);
  Prompt   := DPOTokens(#1);
  Chosen   := DPOTokens(#2#3);
  Rejected := DPOTokens(#4#5);
  ProbeBefore := Policy.Layers[1].Neurons[0].Weights.FData[0];
  Loss := Trainer.Step(Prompt, Chosen, Rejected);
  ProbeAfter := Policy.Layers[1].Neurons[0].Weights.FData[0];
  // beta=0: sigmoid(0)=0.5 scale, loss=ln 2 and the error signal
  // s*beta*(y-onehot) is identically zero -> no weight movement.
  AssertTrue('beta=0 loss must be ln 2', Abs(Loss - Ln(2)) < 1e-6);
  AssertTrue('beta=0 scale must be 0.5', Abs(Trainer.LastScale - 0.5) < 1e-6);
  AssertTrue('beta=0 step must not move weights', ProbeBefore = ProbeAfter);
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralDPO.TestGradientSignFlipsBetweenChosenAndRejected;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Input: TNNetVolume;
  Prompt, SeqA, SeqB: TNeuralDPOTokenArray;
  PA0, PB0, PA1, PB1: TNeuralFloat;

  procedure ReadProbs(out PA, PB: TNeuralFloat);
  begin
    Input.OneHotEncodingReversed(Prompt);
    Policy.Compute(Input);
    PA := Policy.GetLastLayer().Output.FData[SeqA[0]];
    PB := Policy.GetLastLayer().Output.FData[SeqB[0]];
  end;

begin
  RandSeed := 424242;
  Input := TNNetVolume.Create(csContext, 1, csVocab);
  Prompt := DPOTokens(#1#2);
  SeqA   := DPOTokens(#3);
  SeqB   := DPOTokens(#5);

  // Direction 1: chosen=A, rejected=B -> P(A) up, P(B) down.
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.05, 0);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}1.0);
  ReadProbs(PA0, PB0);
  Trainer.Step(Prompt, SeqA, SeqB);
  ReadProbs(PA1, PB1);
  AssertTrue('chosen token prob must increase', PA1 > PA0);
  AssertTrue('rejected token prob must decrease', PB1 < PB0);
  Trainer.Free;
  Policy.Free;

  // Direction 2 (sign flip): from the SAME init, chosen=B, rejected=A.
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.05, 0);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}1.0);
  ReadProbs(PA0, PB0);
  Trainer.Step(Prompt, SeqB, SeqA);
  ReadProbs(PA1, PB1);
  AssertTrue('after role swap the rejected token prob must decrease', PA1 < PA0);
  AssertTrue('after role swap the chosen token prob must increase', PB1 > PB0);
  Trainer.Free;
  Policy.Free;
  Input.Free;
end;

procedure TTestNeuralDPO.TestFiniteDifferenceGradient;
const
  csEps = 1e-3;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  LayerIdx, NeuronIdx, WeightIdx, N, I, Checked: integer;
  W: TNNetVolume;
  Saved, LossPlus, LossMinus, FDGrad, ImplGrad, Best: TNeuralFloat;
begin
  // FPC's mtwist Random only RE-initializes when RandSeed changes to a value
  // different from the one it last seeded with; once the generator has been
  // advanced by earlier tests, a bare "RandSeed := 424242" can be silently
  // skipped (it reads back as 424242 yet keeps the stale stream), so this
  // net's weights would depend on prior tests' RNG draws. Force a genuine
  // re-init by seeding a DIFFERENT value, consuming one draw, then seeding the
  // target - this makes BuildTinyLM deterministic regardless of test order.
  RandSeed := 1; Random; RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  // Keep the net well-conditioned: shrink random weights so no softmax
  // probability is near 0/1 (FD of Ln(p) stays accurate).
  Policy.MulWeights(0.5);
  // LR=1 + inertia 0 + batch mode: each neuron accumulates Delta = -gradient.
  Policy.SetLearningRate(1.0, 0);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.8);
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4);
  Rejected := DPOTokens(#6#7);

  // One backward pass fills every neuron's Delta with -gradient.
  Policy.SetBatchUpdate(true);
  Policy.ClearDeltas();
  Trainer.AccumulateGradients(Prompt, Chosen, Rejected);
  Policy.SetBatchUpdate(false);

  // FD-check the LARGEST-gradient weight in each trainable layer (a dead-ReLU
  // zero-gradient weight would only measure single-precision FD noise).
  Checked := 0;
  for LayerIdx := 1 to 2 do
  begin
    Best := -1; NeuronIdx := 0; WeightIdx := 0;
    for N := 0 to Policy.Layers[LayerIdx].Neurons.Count - 1 do
      for I := 0 to Policy.Layers[LayerIdx].Neurons[N].Weights.Size - 1 do
        if Abs(Policy.Layers[LayerIdx].Neurons[N].Delta.FData[I]) > Best then
        begin
          Best := Abs(Policy.Layers[LayerIdx].Neurons[N].Delta.FData[I]);
          NeuronIdx := N; WeightIdx := I;
        end;
    AssertTrue('layer ' + IntToStr(LayerIdx) + ' must have a non-zero gradient',
      Best > 1e-4);
    ImplGrad := -Policy.Layers[LayerIdx].Neurons[NeuronIdx].Delta.FData[WeightIdx];

    // Central finite difference of the scalar DPO loss.
    W := Policy.Layers[LayerIdx].Neurons[NeuronIdx].Weights;
    Saved := W.FData[WeightIdx];
    W.FData[WeightIdx] := Saved + csEps;
    LossPlus := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
    W.FData[WeightIdx] := Saved - csEps;
    LossMinus := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
    W.FData[WeightIdx] := Saved;
    FDGrad := (LossPlus - LossMinus) / (2 * csEps);

    AssertTrue(Format(
      'DPO finite-difference mismatch at layer %d neuron %d weight %d: ' +
      'fd=%g implemented=%g', [LayerIdx, NeuronIdx, WeightIdx, FDGrad, ImplGrad]),
      Abs(FDGrad - ImplGrad) / Max(Abs(ImplGrad), 1e-4) < 0.02);
    Inc(Checked);
  end;
  AssertTrue('must have FD-checked two layers', Checked = 2);
  Policy.ClearDeltas();

  Trainer.Free;
  Policy.Free;
end;

function TTestNeuralDPO.CaptureError(Trainer: TNeuralDPOTrainer; Mode: integer;
  const Prompt, Chosen, Rejected: TNeuralDPOTokenArray): string;
begin
  Result := '';
  try
    if Mode = 0
    then Trainer.ComputeLoss(Prompt, Chosen, Rejected)
    else Trainer.Step(Prompt, Chosen, Rejected);
  except
    on E: Exception do Result := E.Message;
  end;
end;

procedure TTestNeuralDPO.TestOutOfVocabTargetRejectedOnComputeLoss;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Msg: string;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
  // Ord('a') = 97, far past csVocab = 8: an out-of-bounds read of the output.
  Msg := CaptureError(Trainer, 0, DPOTokens(#1#2), DPOTokens('a'),
    DPOTokens(#5));
  Trainer.Free;
  Policy.Free;
  AssertTrue('an out-of-vocabulary chosen token must be rejected', Msg <> '');
  AssertTrue('the message must name the offending id and character, got: ' +
    Msg, (Pos('97', Msg) > 0) and (Pos('"a"', Msg) > 0));
  AssertTrue('the message must name the vocabulary size, got: ' + Msg,
    Pos('vocabulary size 8', Msg) > 0);
end;

procedure TTestNeuralDPO.TestOutOfVocabTargetRejectedOnStep;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Msg: string;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.01, 0);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
  // Step reaches BackpropagateCompletion, which WRITES at FData[Target].
  Msg := CaptureError(Trainer, 1, DPOTokens(#1#2), DPOTokens(#3),
    DPOTokens('z'));
  Trainer.Free;
  Policy.Free;
  AssertTrue('an out-of-vocabulary rejected token must be rejected on Step',
    Msg <> '');
  AssertTrue('the message must name the offending id, got: ' + Msg,
    Pos('122', Msg) > 0);
end;

procedure TTestNeuralDPO.TestOutOfVocabTargetRejectedOnSkipDerivativeStep;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Msg: string;
begin
  RandSeed := 424242;
  Policy := TNNet.Create();
  Policy.AddLayer(TNNetInput.Create(csContext, 1, csVocab));
  Policy.AddLayer(TNNetFullConnectReLU.Create(csHidden));
  Policy.AddLayer(TNNetFullConnectLinear.Create(csVocab));
  Policy.AddLayer(TNNetPointwiseSoftMax.Create({SkipBackpropDerivative=}1));
  Policy.SetLearningRate(0.01, 0);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
  Msg := CaptureError(Trainer, 1, DPOTokens(#1#2), DPOTokens('a'),
    DPOTokens(#5));
  Trainer.Free;
  Policy.Free;
  AssertTrue('the skip-derivative write branch must be guarded too', Msg <> '');
  AssertTrue('the message must name the offending id, got: ' + Msg,
    Pos('97', Msg) > 0);
end;

procedure TTestNeuralDPO.TestOutOfVocabPromptRejected;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Msg: string;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
  Msg := CaptureError(Trainer, 0, DPOTokens('Q'), DPOTokens(#3),
    DPOTokens(#5));
  Trainer.Free;
  Policy.Free;
  AssertTrue('an out-of-vocabulary prompt token must be rejected', Msg <> '');
  AssertTrue('the message must say it is the prompt, got: ' + Msg,
    Pos('prompt', Msg) > 0);
end;

procedure TTestNeuralDPO.TestNegativeTokenRejected;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Chosen: TNeuralDPOTokenArray;
  Msg: string;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
  SetLength(Chosen, 2);
  Chosen[0] := 3; Chosen[1] := -1;
  Msg := CaptureError(Trainer, 0, DPOTokens(#1#2), Chosen, DPOTokens(#5));
  Trainer.Free;
  Policy.Free;
  AssertTrue('a negative token id must be rejected', Msg <> '');
  AssertTrue('the message must name the offending position, got: ' + Msg,
    Pos('#1', Msg) > 0);
end;

procedure TTestNeuralDPO.TestAllLossModesRejectOutOfVocabTarget;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Mode: TNeuralPreferenceLossMode;
  Msg: string;
begin
  for Mode := Low(TNeuralPreferenceLossMode) to High(TNeuralPreferenceLossMode) do
  begin
    RandSeed := 424242;
    Policy := BuildTinyLM(csContext, csVocab, csHidden);
    Policy.SetLearningRate(0.01, 0);
    // A cloned reference serves every mode (ignored by the reference-free ones).
    Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
    Trainer.LossMode := Mode;
    Msg := CaptureError(Trainer, 1, DPOTokens(#1#2), DPOTokens('a'),
      DPOTokens(#5));
    Trainer.Free;
    Policy.Free;
    AssertTrue('loss mode ' + IntToStr(Ord(Mode)) +
      ' must reject an out-of-vocabulary target', Msg <> '');
    AssertTrue('loss mode ' + IntToStr(Ord(Mode)) +
      ' must name the offending id, got: ' + Msg, Pos('97', Msg) > 0);
  end;
end;

procedure TTestNeuralDPO.TestLastValidTokenAccepted;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Chosen: TNeuralDPOTokenArray;
  Msg: string;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.01, 0);
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
  SetLength(Chosen, 1);
  Chosen[0] := csVocab - 1;                       // the largest valid id
  Msg := CaptureError(Trainer, 1, DPOTokens(#0#1), Chosen, DPOTokens(#5));
  Trainer.Free;
  Policy.Free;
  AssertTrue('id 0 and id VocabSize-1 must be accepted, got: ' + Msg, Msg = '');
end;

initialization
  RegisterTest(TTestNeuralDPO);
end.
