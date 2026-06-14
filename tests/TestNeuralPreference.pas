unit TestNeuralPreference;
(*
Tests for the SimPO / ORPO / KTO preference loss-formula siblings on
TNeuralDPOTrainer (neuraldpo.pas).
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuraldpo;

type
  TTestNeuralPreference = class(TTestCase)
  private
    function BuildTinyLM(ContextLen, Vocab, Hidden: integer): TNNet;
    // Independent per-token forward + ln(prob) sum, mirroring SequenceLogProb.
    function HandLogProb(NN: TNNet; Input: TNNetVolume;
      const Prompt, Completion: TNeuralDPOTokenArray): TNeuralFloat;
  published
    // SimPO is reference-free: CreateReferenceFree(plmSimPO) builds a trainer
    // with no reference net and ComputeLoss runs without one.
    procedure TestSimPONeedsNoReference;
    // SimPO loss matches -ln sigmoid((beta/|y_w|)logp_w - (beta/|y_l|)logp_l - gamma)
    // recomputed by hand from independent forward passes.
    procedure TestSimPOLossMatchesHandComputed;
    // A few SimPO steps increase the length-normalized reward margin.
    procedure TestSimPOStepDecreasesLoss;
    // ORPO is reference-free too.
    procedure TestORPONeedsNoReference;
    // ORPO loss matches L_SFT + lambda*L_OR recomputed by hand.
    procedure TestORPOLossMatchesHandComputed;
    // A few ORPO steps decrease the ORPO loss on a toy pair.
    procedure TestORPOStepDecreasesLoss;
    // KTO uses a reference; loss is finite and a few steps decrease it.
    procedure TestKTOFiniteAndStepDecreasesLoss;
    // All three losses are finite on a random net.
    procedure TestAllLossesFinite;
  end;

implementation

const
  csContext = 4;
  csVocab   = 8;
  csHidden  = 10;

function TTestNeuralPreference.BuildTinyLM(
  ContextLen, Vocab, Hidden: integer): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(ContextLen, 1, Vocab));
  Result.AddLayer(TNNetFullConnectReLU.Create(Hidden));
  Result.AddLayer(TNNetFullConnectLinear.Create(Vocab));
  Result.AddLayer(TNNetSoftMax.Create());
end;

function TTestNeuralPreference.HandLogProb(NN: TNNet; Input: TNNetVolume;
  const Prompt, Completion: TNeuralDPOTokenArray): TNeuralFloat;
var
  T, I, PrefLen, Ctx, StartPos: integer;
  Prefix: TNeuralDPOTokenArray;
begin
  Result := 0;
  Ctx := NN.GetFirstLayer().Output.SizeX;
  for T := 0 to High(Completion) do
  begin
    PrefLen := Length(Prompt) + T;
    StartPos := Max(0, PrefLen - Ctx);
    SetLength(Prefix, PrefLen - StartPos);
    for I := StartPos to PrefLen - 1 do
      if I < Length(Prompt)
      then Prefix[I - StartPos] := Prompt[I]
      else Prefix[I - StartPos] := Completion[I - Length(Prompt)];
    Input.OneHotEncodingReversed(Prefix);
    NN.Compute(Input);
    Result := Result + Ln(NN.GetLastLayer().Output.FData[Completion[T]]);
  end;
end;

procedure TTestNeuralPreference.TestSimPONeedsNoReference;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Loss: TNeuralFloat;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Trainer := TNeuralDPOTrainer.CreateReferenceFree(Policy, plmSimPO, {beta=}2.0);
  AssertTrue('SimPO trainer must have no reference net',
    Trainer.Reference = nil);
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4);
  Rejected := DPOTokens(#5#6);
  Loss := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('SimPO loss must be finite without a reference',
    not IsNan(Loss) and not IsInfinite(Loss) and (Loss >= 0));
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralPreference.TestSimPOLossMatchesHandComputed;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Input: TNNetVolume;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  Beta, Gamma, LpW, LpL, RewW, RewL, Margin, Expected, Got: TNeuralFloat;
begin
  RandSeed := 424242;
  Beta := 1.5; Gamma := 0.3;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Trainer := TNeuralDPOTrainer.CreateReferenceFree(Policy, plmSimPO, Beta);
  Trainer.SimPOGamma := Gamma;
  Input := TNNetVolume.Create(csContext, 1, csVocab);
  Prompt   := DPOTokens(#1#2#3);
  Chosen   := DPOTokens(#4#5);      // |y_w| = 2
  Rejected := DPOTokens(#6#7#1);    // |y_l| = 3

  LpW := HandLogProb(Policy, Input, Prompt, Chosen);
  LpL := HandLogProb(Policy, Input, Prompt, Rejected);
  RewW := (Beta / Length(Chosen)) * LpW;
  RewL := (Beta / Length(Rejected)) * LpL;
  Margin := RewW - RewL - Gamma;
  Expected := Ln(1 + Exp(-Margin));

  Got := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('SimPO loss must match hand value: expected ' +
    FloatToStr(Expected) + ' got ' + FloatToStr(Got),
    Abs(Got - Expected) < 1e-4);
  AssertTrue('SimPO margin must match hand value',
    Abs(Trainer.LastMargin - Margin) < 1e-4);

  Input.Free;
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralPreference.TestSimPOStepDecreasesLoss;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  LossBefore, LossAfter: TNeuralFloat;
  I: integer;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.05, 0);
  Trainer := TNeuralDPOTrainer.CreateReferenceFree(Policy, plmSimPO, {beta=}2.0);
  Trainer.SimPOGamma := 0.2;
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4#5);
  Rejected := DPOTokens(#6#7#3);
  LossBefore := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  for I := 1 to 12 do Trainer.Step(Prompt, Chosen, Rejected);
  LossAfter := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('SimPO steps must decrease the loss: before ' +
    FloatToStr(LossBefore) + ' after ' + FloatToStr(LossAfter),
    LossAfter < LossBefore - 1e-4);
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralPreference.TestORPONeedsNoReference;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Loss: TNeuralFloat;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Trainer := TNeuralDPOTrainer.CreateReferenceFree(Policy, plmORPO, {beta=}1.0);
  AssertTrue('ORPO trainer must have no reference net',
    Trainer.Reference = nil);
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4);
  Rejected := DPOTokens(#5#6);
  Loss := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('ORPO loss must be finite without a reference',
    not IsNan(Loss) and not IsInfinite(Loss));
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralPreference.TestORPOLossMatchesHandComputed;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Input: TNNetVolume;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  Lambda, LpW, LpL, NormW, NormL, OddsW, OddsL, DOdds: TNeuralFloat;
  LSft, LOr, Expected, Got: TNeuralFloat;
begin
  RandSeed := 424242;
  Lambda := 0.25;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Trainer := TNeuralDPOTrainer.CreateReferenceFree(Policy, plmORPO, {beta=}1.0);
  Trainer.ORPOLambda := Lambda;
  Input := TNNetVolume.Create(csContext, 1, csVocab);
  Prompt   := DPOTokens(#1#2#3);
  Chosen   := DPOTokens(#4#5);
  Rejected := DPOTokens(#6#7#1);

  LpW := HandLogProb(Policy, Input, Prompt, Chosen);
  LpL := HandLogProb(Policy, Input, Prompt, Rejected);
  NormW := LpW / Length(Chosen);
  NormL := LpL / Length(Rejected);
  // log_odds(y) = L_n - ln(1 - exp(L_n)).
  OddsW := NormW - Ln(1 - Exp(NormW));
  OddsL := NormL - Ln(1 - Exp(NormL));
  DOdds := OddsW - OddsL;
  LSft := -(LpW / Length(Chosen));
  LOr := Ln(1 + Exp(-DOdds));
  Expected := LSft + Lambda * LOr;

  Got := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('ORPO loss must match hand value: expected ' +
    FloatToStr(Expected) + ' got ' + FloatToStr(Got),
    Abs(Got - Expected) < 1e-4);
  AssertTrue('ORPO margin (log-odds difference) must match',
    Abs(Trainer.LastMargin - DOdds) < 1e-4);

  Input.Free;
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralPreference.TestORPOStepDecreasesLoss;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  LossBefore, LossAfter: TNeuralFloat;
  I: integer;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.05, 0);
  Trainer := TNeuralDPOTrainer.CreateReferenceFree(Policy, plmORPO, {beta=}1.0);
  Trainer.ORPOLambda := 0.2;
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4#5);
  Rejected := DPOTokens(#6#7#3);
  LossBefore := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  for I := 1 to 12 do Trainer.Step(Prompt, Chosen, Rejected);
  LossAfter := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('ORPO steps must decrease the loss: before ' +
    FloatToStr(LossBefore) + ' after ' + FloatToStr(LossAfter),
    LossAfter < LossBefore - 1e-4);
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralPreference.TestKTOFiniteAndStepDecreasesLoss;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  LossBefore, LossAfter: TNeuralFloat;
  I: integer;
begin
  RandSeed := 424242;
  Policy := BuildTinyLM(csContext, csVocab, csHidden);
  Policy.SetLearningRate(0.1, 0);
  // KTO needs a (frozen) reference: clone the policy.
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.5);
  Trainer.LossMode := plmKTO;
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4#5);    // desirable example
  Rejected := DPOTokens(#6#7#3);    // undesirable example
  LossBefore := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('KTO loss must be finite',
    not IsNan(LossBefore) and not IsInfinite(LossBefore));
  for I := 1 to 15 do Trainer.Step(Prompt, Chosen, Rejected);
  LossAfter := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
  AssertTrue('KTO steps must decrease the loss: before ' +
    FloatToStr(LossBefore) + ' after ' + FloatToStr(LossAfter),
    LossAfter < LossBefore - 1e-4);
  Trainer.Free;
  Policy.Free;
end;

procedure TTestNeuralPreference.TestAllLossesFinite;
var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Prompt, Chosen, Rejected: TNeuralDPOTokenArray;
  Mode: TNeuralPreferenceLossMode;
  Loss: TNeuralFloat;
begin
  Prompt   := DPOTokens(#1#2);
  Chosen   := DPOTokens(#3#4);
  Rejected := DPOTokens(#5#6);
  for Mode := Low(TNeuralPreferenceLossMode) to High(TNeuralPreferenceLossMode) do
  begin
    RandSeed := 424242;
    Policy := BuildTinyLM(csContext, csVocab, csHidden);
    // DPO and KTO need a reference; SimPO/ORPO do not (but a reference is
    // harmless), so always clone for a uniform setup.
    Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, {beta=}0.7);
    Trainer.LossMode := Mode;
    Loss := Trainer.ComputeLoss(Prompt, Chosen, Rejected);
    AssertTrue('loss mode ' + IntToStr(Ord(Mode)) + ' must be finite',
      not IsNan(Loss) and not IsInfinite(Loss));
    Trainer.Free;
    Policy.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralPreference);
end.
