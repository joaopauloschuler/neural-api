program DPOFineTune;
(*
DPOFineTune: aligning a tiny char-level LM with Direct Preference
Optimization (Rafailov et al. 2023, https://arxiv.org/abs/2305.18290) using
TNeuralDPOTrainer from neural/neuraldpo.pas.

THE EXPERIMENT
  1. PRETRAIN: a TinyGPT-style causal transformer (one-hot char input ->
     pointwise token projection -> positional embedding -> 1 causal
     transformer block -> next-char softmax head) is briefly trained with
     plain next-char SGD on a corpus where the prompt "say: " continues
     EQUALLY OFTEN into a "good" patterned completion ("ababab..") and a
     "bad" noise completion -- so the pretrained model has NO preference
     between the two styles (DPO accuracy starts around chance).
  2. REFERENCE: the pretrained policy is cloned into a frozen reference net
     (the KL anchor of the DPO objective).
  3. DPO: a handful of preference pairs (prompt, chosen=patterned
     completion, rejected=noise completion) are optimized with
     loss = -ln sigmoid(beta*((logpi_c-logref_c)-(logpi_r-logref_r))).
     Each Step backpropagates the exact scaled (softmax - onehot) gradient:
     positive sign on chosen tokens, negative on rejected ones, both scaled
     by sigmoid(-beta*margin).
  4. REPORT: average margin (policy log-prob of chosen minus rejected,
     relative to the reference) and preference accuracy (fraction of pairs
     with margin > 0) are printed per epoch -- both should climb.

Built-in self-checks (PASS/FAIL, Halt(1) on hard failure):
  * preference accuracy reaches 100% after DPO;
  * the average margin increases substantially over its initial value;
  * the average DPO loss drops below its ln(2)~0.693 starting point;
  * a probed reference-net weight is bit-identical after training (frozen).

Pure CPU, single-threaded manual loops; runs in well under a minute.

Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork,
  neuralvolume,
  neuraldpo;

const
  csContextLen = 16;   // context window in characters
  csVocabSize  = 128;  // char-level ASCII vocabulary, one-hot encoded
  csDModel     = 32;   // residual-stream width
  csHeads      = 2;    // attention heads
  csDFF        = 32;   // feed-forward inner width

  csPretrainSteps = 1500;
  csPretrainLR    = 0.01;

  csBeta       = 0.5;  // DPO beta
  csDPOLR      = 0.01; // DPO learning rate
  csDPOEpochs  = 12;   // epochs over the preference pairs

  csPrompt = 'say: ';
  // Chosen completions share the "patterned" style; rejected are noise.
  csNumPairs = 6;
  csChosen: array[0..csNumPairs-1] of string = (
    'ababab', 'abab', 'ababa', 'abababa', 'aba', 'abababab');
  csRejected: array[0..csNumPairs-1] of string = (
    'qzkxwv', 'kwqz', 'zxqwk', 'wvkxqzx', 'xqw', 'kzwxqvkz');

// TinyGPT-style causal next-char LM (see examples/TinyGPT).
function BuildPolicy(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer([
    TNNetInput.Create(csContextLen, 1, csVocabSize),
    TNNetPointwiseConvLinear.Create(csDModel),
    TNNetAddPositionalEmbedding.Create(10000)
  ]);
  Result.AddTransformerEncoderBlock(
    {Heads=}csHeads, {d_ff=}csDFF,
    {PreNorm=}true, {CausalMask=}true,
    {UseRoPE=}false, {NormClass=}nil);
  Result.AddLayer([
    TNNetPointwiseConvReLU.Create(csDModel),
    TNNetFullConnectReLU.Create(csDModel),
    TNNetFullConnectLinear.Create(csVocabSize),
    TNNetSoftMax.Create()
  ]);
end;

// Brief next-char pretraining on BOTH continuation styles equally, so the
// pretrained model is style-agnostic before DPO.
procedure Pretrain(NN: TNNet);
var
  Corpus: array of string;
  Input, Target: TNNetVolume;
  StepIdx, CutPos, TokenId, I: integer;
  Line: string;
begin
  SetLength(Corpus, 2 * csNumPairs);
  for I := 0 to csNumPairs - 1 do
  begin
    Corpus[2*I]     := csPrompt + csChosen[I];
    Corpus[2*I + 1] := csPrompt + csRejected[I];
  end;
  Input := TNNetVolume.Create(csContextLen, 1, csVocabSize);
  Target := TNNetVolume.Create(csVocabSize, 1, 1);
  NN.SetLearningRate(csPretrainLR, 0.9);
  for StepIdx := 1 to csPretrainSteps do
  begin
    Line := Corpus[Random(Length(Corpus))];
    // Random (prefix -> next char) sample; keep at least the prompt prefix.
    CutPos := Length(csPrompt) + Random(Length(Line) - Length(csPrompt));
    TokenId := Min(Ord(Line[CutPos+1]), csVocabSize - 1);
    Input.OneHotEncodingReversed(copy(Line, 1, CutPos));
    Target.SetClassForSoftMax(TokenId);
    NN.Compute(Input);
    NN.Backpropagate(Target);
  end;
  Target.Free;
  Input.Free;
end;

var
  Policy: TNNet;
  Trainer: TNeuralDPOTrainer;
  Prompt: TNeuralDPOTokenArray;
  ChosenTok, RejectedTok: array[0..csNumPairs-1] of TNeuralDPOTokenArray;
  Epoch, PairIdx, Hits: integer;
  AvgLoss, AvgMargin, InitialMargin, FinalMargin, InitialLoss, FinalLoss: TNeuralFloat;
  InitialAcc, FinalAcc: TNeuralFloat;
  RefProbeBefore, RefProbeAfter: TNeuralFloat;
  AllOk: boolean;

// Evaluates the current policy on all pairs WITHOUT updating weights.
procedure Evaluate(out pLoss, pMargin, pAcc: TNeuralFloat);
var
  P: integer;
begin
  pLoss := 0; pMargin := 0; pAcc := 0;
  for P := 0 to csNumPairs - 1 do
  begin
    pLoss := pLoss + Trainer.ComputeLoss(Prompt, ChosenTok[P], RejectedTok[P]);
    pMargin := pMargin + Trainer.LastMargin;
    // Ties (margin exactly 0, e.g. policy == reference) count as chance 0.5.
    if Trainer.LastMargin > 0 then pAcc := pAcc + 1
    else if Trainer.LastMargin = 0 then pAcc := pAcc + 0.5;
  end;
  pLoss := pLoss / csNumPairs;
  pMargin := pMargin / csNumPairs;
  pAcc := pAcc / csNumPairs;
end;

begin
  RandSeed := 424242;

  WriteLn('=== 1. Pretraining tiny causal LM on both styles equally ===');
  Policy := BuildPolicy();
  Pretrain(Policy);
  WriteLn('Pretrained for ', csPretrainSteps, ' next-char SGD steps.');
  WriteLn;

  WriteLn('=== 2. Cloning frozen reference + building preference pairs ===');
  Trainer := TNeuralDPOTrainer.CreateWithClonedReference(Policy, csBeta);
  Prompt := DPOTokens(csPrompt);
  for PairIdx := 0 to csNumPairs - 1 do
  begin
    ChosenTok[PairIdx]   := DPOTokens(csChosen[PairIdx]);
    RejectedTok[PairIdx] := DPOTokens(csRejected[PairIdx]);
  end;
  RefProbeBefore := Trainer.Reference.Layers[1].Neurons[0].Weights.FData[0];

  Evaluate(InitialLoss, InitialMargin, InitialAcc);
  WriteLn('Before DPO: avg loss ', InitialLoss:0:4,
    '  avg margin ', InitialMargin:0:4,
    '  preference accuracy ', (InitialAcc*100):0:1, ' %');
  WriteLn;

  WriteLn('=== 3. DPO fine-tuning (beta=', csBeta:0:2, ', LR=', csDPOLR:0:3, ') ===');
  Policy.SetLearningRate(csDPOLR, 0);
  WriteLn(' epoch |  avg loss | avg margin | accuracy');
  WriteLn('-------+-----------+------------+---------');
  for Epoch := 1 to csDPOEpochs do
  begin
    AvgLoss := 0; AvgMargin := 0; Hits := 0;
    for PairIdx := 0 to csNumPairs - 1 do
    begin
      AvgLoss := AvgLoss +
        Trainer.Step(Prompt, ChosenTok[PairIdx], RejectedTok[PairIdx]);
      AvgMargin := AvgMargin + Trainer.LastMargin;
      if Trainer.LastMargin > 0 then Inc(Hits);
    end;
    AvgLoss := AvgLoss / csNumPairs;
    AvgMargin := AvgMargin / csNumPairs;
    WriteLn(Epoch:6, ' | ', AvgLoss:9:4, ' | ', AvgMargin:10:4, ' | ',
      (100*Hits/csNumPairs):7:1, ' %');
  end;
  WriteLn;

  WriteLn('=== 4. Report and checks ===');
  Evaluate(FinalLoss, FinalMargin, FinalAcc);
  RefProbeAfter := Trainer.Reference.Layers[1].Neurons[0].Weights.FData[0];
  WriteLn('After DPO:  avg loss ', FinalLoss:0:4,
    '  avg margin ', FinalMargin:0:4,
    '  preference accuracy ', (FinalAcc*100):0:1, ' %');
  WriteLn;

  AllOk := true;

  if FinalAcc >= 0.999 then
    WriteLn('CHECK 1 PASS: preference accuracy reached 100%.')
  else
  begin
    WriteLn('CHECK 1 FAIL: preference accuracy is ', (FinalAcc*100):0:1, ' %.');
    AllOk := false;
  end;

  if FinalMargin > InitialMargin + 1.0 then
    WriteLn('CHECK 2 PASS: avg margin rose from ', InitialMargin:0:4,
      ' to ', FinalMargin:0:4, '.')
  else
  begin
    WriteLn('CHECK 2 FAIL: avg margin did not rise enough (',
      InitialMargin:0:4, ' -> ', FinalMargin:0:4, ').');
    AllOk := false;
  end;

  if FinalLoss < Min(InitialLoss, Ln(2)) then
    WriteLn('CHECK 3 PASS: avg DPO loss dropped from ', InitialLoss:0:4,
      ' to ', FinalLoss:0:4, ' (< ln 2).')
  else
  begin
    WriteLn('CHECK 3 FAIL: avg DPO loss did not drop (', InitialLoss:0:4,
      ' -> ', FinalLoss:0:4, ').');
    AllOk := false;
  end;

  if RefProbeAfter = RefProbeBefore then
    WriteLn('CHECK 4 PASS: frozen reference weight unchanged.')
  else
  begin
    WriteLn('CHECK 4 FAIL: the reference net moved.');
    AllOk := false;
  end;

  WriteLn;
  if AllOk then
    WriteLn('ALL CHECKS PASSED.')
  else
  begin
    WriteLn('ONE OR MORE CHECKS FAILED.');
    Halt(1);
  end;

  Trainer.Free;
  Policy.Free;
end.
