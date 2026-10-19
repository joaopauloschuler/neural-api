unit neuralanyjev;

(*
neuralanyjev
Copyright (C) 2026 Joao Paulo Schwarz Schuler

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

Coded by Claude (AI).
*)

// neuralanyjev -- AnyJev's L0 readout (anyjev 0.3.0: readout.py,
// calibrate/permute.py, calibrate/contextual.py, decider.py) as a
// /v1/systemone scorer on any chat model a TChatEngine has loaded. Nothing
// is generated: each question becomes one prompt per rotation of its
// options, the model's next-token logits are read at the answer position
// and restricted to the label tokens (A, B, ... / 1, 2, ... / Yes, No).
//
//   - raw: one prompt, options in the order given, no prior.
//   - L0: every rotation (choice: cyclic shifts; noul: both phrasings;
//     score: none), each rotation divided by the content-free prior (the
//     same prompts with the states "N/A", "" and "[MASK]"), mapped back to
//     the options and combined by the log-mean.
//
// The math is in pure routines over probability arrays, so it is testable
// without a model. TAnyJevL0Scorer feeds every prompt of one state (all
// questions, all rotations) after one shared prefix: the token prefix those
// prompts have in common, so the model reads exactly each full prompt.
//
// This unit is separate from neuraldecision, which stays model-free.

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralhftokenizer, neuralchat,
  neuralchatengine, neuraldecision;

const
  csAnyJevSystemPrompt =
    'You are a decision function. You will be given a state and one question. ' +
    'Reply with the answer label only: no words, no punctuation, no explanation.';
  // Letter readout: A..Z.
  csAnyJevMaxChoiceOptions = 26;
  csAnyJevContentFreeProbeCount = 3;
  csAnyJevContentFreeProbes: array[0..csAnyJevContentFreeProbeCount - 1] of string =
    ('N/A', '', '[MASK]');
  // The exponent of the content-free prior (AnyJev's prior_strength default).
  csAnyJevContentFreePriorStrength: double = 1.0;
  // contextual.py EPS (prior and corrected probabilities) and permute.py EPS (log-mean).
  csAnyJevPriorFloor: double = 1e-8;
  csAnyJevLogMeanFloor: double = 1e-12;

type
  // A tokenizer that cannot spell a label as one token, or spells two alike.
  EAnyJevLabelTokenError = class(Exception);

  TAnyJevLevel = (ajlRaw, ajlL0);
  TAnyJevPrior = (ajpContentFree, ajpNone);
  TAnyJevCombine = (ajcLogMean, ajcMean);
  // Rotation[j] = the option shown at position j.
  TAnyJevRotation = TNeuralIntegerArray;
  TAnyJevRotations = array of TAnyJevRotation;
  // Row r = a distribution of rotation r (or of probe r), indexed by position.
  TAnyJevMatrix = array of TDecisionProbabilities;

  // What one question of the last request read, for parity checks. AnyJev's
  // option order: choice/score as given, noul (Yes, No).
  TAnyJevQuestionDiagnostics = record
    Labels: TStringArray;
    LabelIds: TNeuralIntegerArray;          // per label, in label order
    Rotations: TAnyJevRotations;
    RawByRotation: TAnyJevMatrix;           // [rotation][position]
    PriorByRotation: TAnyJevMatrix;         // [rotation][position]; empty: no prior
    Probabilities: TDecisionProbabilities;  // by option
  end;
  TAnyJevDiagnostics = array of TAnyJevQuestionDiagnostics;

  TAnyJevL0Scorer = class(TDecisionScorer)
  private
    FEngine: TChatEngine;
    FLevel: TAnyJevLevel;
    FPrior: TAnyJevPrior;
    FMaxRotations: integer;
    FSharePrefix: boolean;
    FPlainPrompt: boolean;
    FAppendClosedThinking: boolean;
    FLastDiagnostics: TAnyJevDiagnostics;
  public
    // AEngine is loaded and not owned. AMaxRotations caps the choice shifts (0 = K).
    constructor Create(AEngine: TChatEngine; const AModelName: string;
      ALevel: TAnyJevLevel = ajlL0; APrior: TAnyJevPrior = ajpContentFree;
      AMaxRotations: integer = 0);
    procedure Score(const Request: TDecisionRequest; out Answers: TDecisionAnswers;
      out InputTokens, OutputTokens: integer); override;
    // The full prompt text for one user message: the chat template with the
    // generation prompt and thinking off, or AnyJevPlainPrompt without a template.
    function RenderPrompt(const UserMessage: string): string;
    // False: every prompt is fed from an empty session (same probabilities).
    property SharePrefix: boolean read FSharePrefix write FSharePrefix;
    property LastDiagnostics: TAnyJevDiagnostics read FLastDiagnostics;
  end;

// readout.answer_labels: noul (Yes, No), score '1'..'K', choice 'A'...
function AnyJevAnswerLabels(Kind: TDecisionQuestionKind; OptionCount: integer): TStringArray;
function AnyJevLetterLabels(OptionCount: integer): TStringArray;
// readout.build_prompt's user text. Options are unused for noul.
function BuildAnyJevUserMessage(const StateText, QuestionText: string;
  Kind: TDecisionQuestionKind; const Options: array of string;
  const Rotation: TAnyJevRotation; const Labels: array of string): string;
// "{system}\n\n{user}\nAnswer:", the prompt of a model without a chat template.
function AnyJevPlainPrompt(const SystemPrompt, UserMessage: string): string;
// readout.map_label_tokens: per label the first of Label, ' ' + Label that is one token.
function MapAnyJevLabelTokens(Tokenizer: TNeuralHFTokenizer;
  const Labels: array of string): TNeuralIntegerArray;
// readout.resolve_labels: score falls back to letters when a digit is not one token.
procedure ResolveAnyJevLabels(Tokenizer: TNeuralHFTokenizer;
  Kind: TDecisionQuestionKind; OptionCount: integer;
  out Labels: TStringArray; out LabelIds: TNeuralIntegerArray);
// permute.cyclic_shifts: Shift s shows option (j + s) mod K at position j.
function AnyJevCyclicShifts(OptionCount, MaxRotations: integer): TAnyJevRotations;
// The Jev question as AnyJev reads it, special-token strings rewritten: noul
// criteria appended ("Yes means:" first), choice "name: desc", score levels.
procedure AnyJevQuestionParts(const Question: TDecisionQuestion;
  out QuestionText: string; out Options: TStringArray);
// Decider._perms: raw and score the identity, noul both orders, choice the shifts.
function AnyJevRotations(Kind: TDecisionQuestionKind; OptionCount: integer;
  Level: TAnyJevLevel; MaxRotations: integer): TAnyJevRotations;
// readout.label_ids_for_perm: the label id read at each position of Rotation.
function AnyJevPositionLabelIds(Kind: TDecisionQuestionKind;
  const LabelIds: TNeuralIntegerArray; const Rotation: TAnyJevRotation): TNeuralIntegerArray;
// contextual.content_free_prior: mean of the probe rows, floored, normalised.
function AnyJevContentFreePrior(const ProbeProbabilities: TAnyJevMatrix): TDecisionProbabilities;
// contextual.apply_contextual(p, prior ^ PriorStrength): divide, floor, normalise.
function AnyJevApplyPrior(const RawProbabilities, Prior: TDecisionProbabilities;
  PriorStrength: double): TDecisionProbabilities;
// permute.marginalize: positions back to options, combined over the rotations.
function AnyJevMarginalize(const PositionProbabilities: TAnyJevMatrix;
  const Rotations: TAnyJevRotations; Combine: TAnyJevCombine = ajcLogMean): TDecisionProbabilities;
// One question's answer by option: each rotation corrected by its prior row
// (PriorByRotation empty = no prior), then the log-mean.
function AnyJevOptionProbabilities(const RawByRotation, PriorByRotation: TAnyJevMatrix;
  const Rotations: TAnyJevRotations; PriorStrength: double): TDecisionProbabilities;

implementation

uses
  Math;

function NormalizedFloored(const P: TDecisionProbabilities; Floor: double): TDecisionProbabilities;
var
  ItemPos, MaxItemPos: integer;
  Total: double;
begin
  Result := nil;
  SetLength(Result, Length(P));
  Total := 0;
  MaxItemPos := High(P);
  for ItemPos := 0 to MaxItemPos do
  begin
    Result[ItemPos] := Max(P[ItemPos], Floor);
    Total := Total + Result[ItemPos];
  end;
  for ItemPos := 0 to MaxItemPos do
    Result[ItemPos] := Result[ItemPos] / Total;
end;

function AnyJevLetterLabels(OptionCount: integer): TStringArray;
var
  LabelPos: integer;
begin
  if OptionCount > csAnyJevMaxChoiceOptions then
    raise EDecisionUnsupported.CreateFmt('AnyJev letter labels cover at most %d' +
      ' options, not %d', [csAnyJevMaxChoiceOptions, OptionCount]);
  Result := nil;
  SetLength(Result, OptionCount);
  for LabelPos := 0 to OptionCount - 1 do
    Result[LabelPos] := Chr(Ord('A') + LabelPos);
end;

function AnyJevAnswerLabels(Kind: TDecisionQuestionKind; OptionCount: integer): TStringArray;
var
  LabelPos: integer;
begin
  Result := nil;
  case Kind of
    dqkNoul:
      begin
        SetLength(Result, 2);
        Result[0] := 'Yes';
        Result[1] := 'No';
      end;
    dqkScore:
      begin
        SetLength(Result, OptionCount);
        for LabelPos := 0 to OptionCount - 1 do
          Result[LabelPos] := IntToStr(LabelPos + 1);
      end;
    else
      Result := AnyJevLetterLabels(OptionCount);
  end;
end;

function BuildAnyJevUserMessage(const StateText, QuestionText: string;
  Kind: TDecisionQuestionKind; const Options: array of string;
  const Rotation: TAnyJevRotation; const Labels: array of string): string;
var
  ShownState: string;
  PositionPos, MaxPositionPos: integer;
begin
  if StateText = '' then ShownState := '(empty)'
  else ShownState := StateText;
  Result := 'State:' + #10 + ShownState + #10#10 + 'Question: ' + QuestionText + #10;
  if Kind = dqkNoul then
  begin
    Result := Result + 'Answer ' + Labels[Rotation[0]] + ' or ' +
      Labels[Rotation[1]] + '.';
    exit;
  end;
  if Kind = dqkScore then
    Result := Result + 'Pick the level that applies (the levels are ordered):' + #10
  else
    Result := Result + 'Options:' + #10;
  MaxPositionPos := High(Rotation);
  for PositionPos := 0 to MaxPositionPos do
    Result := Result + Labels[PositionPos] + '. ' + Options[Rotation[PositionPos]] + #10;
  // Python str.isdigit on the first label decides the closing line.
  if (Kind = dqkScore) and (Labels[0] <> '') and (Labels[0][1] in ['0'..'9']) then
    Result := Result + 'Answer with the number only.'
  else
    Result := Result + 'Answer with the letter only.';
end;

function AnyJevPlainPrompt(const SystemPrompt, UserMessage: string): string;
begin
  Result := SystemPrompt + #10#10 + UserMessage + #10 + 'Answer:';
end;

function MapAnyJevLabelTokens(Tokenizer: TNeuralHFTokenizer;
  const Labels: array of string): TNeuralIntegerArray;
var
  LabelPos, OtherPos, MaxLabelPos: integer;
  Ids: TNeuralIntegerArray;
begin
  Result := nil;
  SetLength(Result, Length(Labels));
  MaxLabelPos := High(Labels);
  for LabelPos := 0 to MaxLabelPos do
  begin
    Ids := Tokenizer.Encode(Labels[LabelPos]);
    if Length(Ids) <> 1 then Ids := Tokenizer.Encode(' ' + Labels[LabelPos]);
    if Length(Ids) <> 1 then
      raise EAnyJevLabelTokenError.CreateFmt('label "%s" is not a single token' +
        ' for this tokenizer', [Labels[LabelPos]]);
    Result[LabelPos] := Ids[0];
    for OtherPos := 0 to LabelPos - 1 do
      if Result[OtherPos] = Ids[0] then
        raise EAnyJevLabelTokenError.CreateFmt('labels "%s" and "%s" map to the' +
          ' same token %d', [Labels[OtherPos], Labels[LabelPos], Ids[0]]);
  end;
end;

procedure ResolveAnyJevLabels(Tokenizer: TNeuralHFTokenizer;
  Kind: TDecisionQuestionKind; OptionCount: integer;
  out Labels: TStringArray; out LabelIds: TNeuralIntegerArray);
begin
  Labels := AnyJevAnswerLabels(Kind, OptionCount);
  try
    LabelIds := MapAnyJevLabelTokens(Tokenizer, Labels);
  except
    on EAnyJevLabelTokenError do
    begin
      if Kind <> dqkScore then raise;
      Labels := AnyJevLetterLabels(OptionCount);
      LabelIds := MapAnyJevLabelTokens(Tokenizer, Labels);
    end;
  end;
end;

function IdentityRotation(OptionCount: integer): TAnyJevRotation;
var
  OptionPos: integer;
begin
  Result := nil;
  SetLength(Result, OptionCount);
  for OptionPos := 0 to OptionCount - 1 do
    Result[OptionPos] := OptionPos;
end;

function AnyJevCyclicShifts(OptionCount, MaxRotations: integer): TAnyJevRotations;
var
  ShiftCount, Shift, PositionPos: integer;
begin
  ShiftCount := OptionCount;
  if MaxRotations > 0 then ShiftCount := Max(1, Min(OptionCount, MaxRotations));
  Result := nil;
  SetLength(Result, ShiftCount);
  for Shift := 0 to ShiftCount - 1 do
  begin
    SetLength(Result[Shift], OptionCount);
    for PositionPos := 0 to OptionCount - 1 do
      Result[Shift][PositionPos] := (PositionPos + Shift) mod OptionCount;
  end;
end;

function AnyJevRotations(Kind: TDecisionQuestionKind; OptionCount: integer;
  Level: TAnyJevLevel; MaxRotations: integer): TAnyJevRotations;
begin
  Result := nil;
  if (Level = ajlRaw) or (Kind = dqkScore) then
  begin
    SetLength(Result, 1);
    Result[0] := IdentityRotation(OptionCount);
  end
  else if Kind = dqkNoul then
  begin
    SetLength(Result, 2);
    Result[0] := IdentityRotation(2);
    SetLength(Result[1], 2);
    Result[1][0] := 1;
    Result[1][1] := 0;
  end
  else
    Result := AnyJevCyclicShifts(OptionCount, MaxRotations);
end;

function AnyJevPositionLabelIds(Kind: TDecisionQuestionKind;
  const LabelIds: TNeuralIntegerArray; const Rotation: TAnyJevRotation): TNeuralIntegerArray;
var
  PositionPos, MaxPositionPos: integer;
begin
  // Choice and score labels name a position; noul labels name the option.
  if Kind <> dqkNoul then Exit(Copy(LabelIds));
  Result := nil;
  SetLength(Result, Length(Rotation));
  MaxPositionPos := High(Rotation);
  for PositionPos := 0 to MaxPositionPos do
    Result[PositionPos] := LabelIds[Rotation[PositionPos]];
end;

function AnyJevContentFreePrior(const ProbeProbabilities: TAnyJevMatrix): TDecisionProbabilities;
var
  ProbePos, MaxProbePos, ItemPos, MaxItemPos: integer;
  Mean: TDecisionProbabilities;
begin
  Mean := nil;
  SetLength(Mean, Length(ProbeProbabilities[0]));
  MaxProbePos := High(ProbeProbabilities);
  MaxItemPos := High(Mean);
  for ProbePos := 0 to MaxProbePos do
    for ItemPos := 0 to MaxItemPos do
      Mean[ItemPos] := Mean[ItemPos] + ProbeProbabilities[ProbePos][ItemPos];
  for ItemPos := 0 to MaxItemPos do
    Mean[ItemPos] := Mean[ItemPos] / Length(ProbeProbabilities);
  Result := NormalizedFloored(Mean, csAnyJevPriorFloor);
end;

function AnyJevApplyPrior(const RawProbabilities, Prior: TDecisionProbabilities;
  PriorStrength: double): TDecisionProbabilities;
var
  ItemPos, MaxItemPos: integer;
  Corrected: TDecisionProbabilities;
  Divisor: double;
begin
  Corrected := nil;
  SetLength(Corrected, Length(RawProbabilities));
  MaxItemPos := High(RawProbabilities);
  for ItemPos := 0 to MaxItemPos do
  begin
    if PriorStrength = 1.0 then Divisor := Prior[ItemPos]
    else Divisor := Power(Prior[ItemPos], PriorStrength);
    Corrected[ItemPos] := RawProbabilities[ItemPos] / Max(Divisor, csAnyJevPriorFloor);
  end;
  Result := NormalizedFloored(Corrected, csAnyJevPriorFloor);
end;

function AnyJevMarginalize(const PositionProbabilities: TAnyJevMatrix;
  const Rotations: TAnyJevRotations; Combine: TAnyJevCombine): TDecisionProbabilities;
var
  RotationPos, MaxRotationPos, PositionPos, MaxPositionPos, OptionPos: integer;
  Sums: TDecisionProbabilities;
  Value, Largest: double;
begin
  Sums := nil;
  SetLength(Sums, Length(Rotations[0]));
  MaxRotationPos := High(Rotations);
  MaxPositionPos := High(Sums);
  for RotationPos := 0 to MaxRotationPos do
    for PositionPos := 0 to MaxPositionPos do
    begin
      OptionPos := Rotations[RotationPos][PositionPos];
      Value := PositionProbabilities[RotationPos][PositionPos];
      if Combine = ajcLogMean then Value := Ln(Max(Value, csAnyJevLogMeanFloor));
      Sums[OptionPos] := Sums[OptionPos] + Value;
    end;
  for OptionPos := 0 to MaxPositionPos do
    Sums[OptionPos] := Sums[OptionPos] / Length(Rotations);
  if Combine = ajcLogMean then
  begin
    Largest := MaxValue(Sums);
    for OptionPos := 0 to MaxPositionPos do
      Sums[OptionPos] := Exp(Sums[OptionPos] - Largest);
  end;
  Result := NormalizedFloored(Sums, 0);
end;

function AnyJevOptionProbabilities(const RawByRotation, PriorByRotation: TAnyJevMatrix;
  const Rotations: TAnyJevRotations; PriorStrength: double): TDecisionProbabilities;
var
  Corrected: TAnyJevMatrix;
  RotationPos, MaxRotationPos: integer;
begin
  if Length(PriorByRotation) = 0 then
    Exit(AnyJevMarginalize(RawByRotation, Rotations));
  Corrected := nil;
  SetLength(Corrected, Length(RawByRotation));
  MaxRotationPos := High(RawByRotation);
  for RotationPos := 0 to MaxRotationPos do
    Corrected[RotationPos] := AnyJevApplyPrior(RawByRotation[RotationPos],
      PriorByRotation[RotationPos], PriorStrength);
  Result := AnyJevMarginalize(Corrected, Rotations);
end;

procedure AnyJevQuestionParts(const Question: TDecisionQuestion;
  out QuestionText: string; out Options: TStringArray);
var
  OptionCount, OptionPos, OtherPos: integer;
begin
  OptionCount := Length(Question.OptionNames);
  QuestionText := RewriteSpecialTokenStrings(Question.Instructions);
  Options := nil;
  if Question.Kind = dqkNoul then
  begin
    // OptionDescriptions = (false, true): AnyJev's fixed Yes/No, true first.
    if Question.OptionDescriptions[1] <> '' then
      QuestionText := QuestionText + #10 + 'Yes means: ' +
        RewriteSpecialTokenStrings(Question.OptionDescriptions[1]);
    if Question.OptionDescriptions[0] <> '' then
      QuestionText := QuestionText + #10 + 'No means: ' +
        RewriteSpecialTokenStrings(Question.OptionDescriptions[0]);
    exit;
  end;
  SetLength(Options, OptionCount);
  for OptionPos := 0 to OptionCount - 1 do
  begin
    if Question.Kind = dqkChoice then
      Options[OptionPos] := DecisionOptionText(Question.OptionNames[OptionPos],
        Question.OptionDescriptions[OptionPos])
    else
      Options[OptionPos] := Question.OptionDescriptions[OptionPos];
    Options[OptionPos] := RewriteSpecialTokenStrings(Options[OptionPos]);
    if Question.Kind = dqkChoice then
      for OtherPos := 0 to OptionPos - 1 do
        if Options[OtherPos] = Options[OptionPos] then
          raise EDecisionUnsupported.CreateFmt('question "%s": two options' +
            ' read "%s"; AnyJev needs distinct option texts',
            [Question.Id, Options[OptionPos]]);
  end;
end;

{ TAnyJevL0Scorer }

type
  // One prompt to score: the question and rotation it belongs to, the probe
  // it carries (-1: the request's state), its ids and the ids read per position.
  TAnyJevPrompt = record
    QuestionPos, RotationPos, ProbePos: integer;
    Ids, PositionLabelIds: TNeuralIntegerArray;
  end;
  TAnyJevPrompts = array of TAnyJevPrompt;

constructor TAnyJevL0Scorer.Create(AEngine: TChatEngine; const AModelName: string;
  ALevel: TAnyJevLevel; APrior: TAnyJevPrior; AMaxRotations: integer);
var
  ConfigFile, TemplateText: string;
begin
  inherited Create(AModelName);
  FEngine := AEngine;
  FLevel := ALevel;
  FPrior := APrior;
  FMaxRotations := AMaxRotations;
  FSharePrefix := true;
  TemplateText := '';
  ConfigFile := IncludeTrailingPathDelimiter(AEngine.Opt.ModelDir) + 'tokenizer_config.json';
  if FileExists(ConfigFile) then TemplateText := LoadChatTemplateString(ConfigFile);
  // AnyJev renders without a template only when the tokenizer has none; the
  // engine would fall back to ChatML there.
  FPlainPrompt := AEngine.RawMode or
    ((AEngine.Opt.FormatName = '') and (Trim(TemplateText) = ''));
  // The Qwen3 template closes the thinking block itself under
  // enable_thinking=false; the plain ChatML renderers have no such switch.
  FAppendClosedThinking := (AEngine.ChatFormat in [cfChatML, cfQwen]) and
    (Pos('enable_thinking', TemplateText) > 0) and
    (Pos('<think>\n\n</think>\n\n', TemplateText) > 0);
end;

function TAnyJevL0Scorer.RenderPrompt(const UserMessage: string): string;
var
  Msgs: array[0..1] of TChatMessage;
begin
  if FPlainPrompt then
    Exit(AnyJevPlainPrompt(csAnyJevSystemPrompt, UserMessage));
  Msgs[0] := ChatMessage('system', csAnyJevSystemPrompt);
  Msgs[1] := ChatMessage('user', UserMessage);
  Result := ApplyChatTemplate(FEngine.ChatFormat, Msgs,
    ChatTemplateOptions({AddGenerationPrompt=}true, {ContinueFinalMessage=}false, reOff));
  if FAppendClosedThinking then Result := Result + csQwenThinkClosed;
end;

procedure TAnyJevL0Scorer.Score(const Request: TDecisionRequest;
  out Answers: TDecisionAnswers; out InputTokens, OutputTokens: integer);
var
  QuestionCount, QuestionPos, MaxQuestionPos, OptionCount: integer;
  RotationPos, MaxRotationPos, ProbePos, GroupPos: integer;
  StateText, QuestionText: string;
  Options: TStringArray;
  // [question][rotation][probe] -> distribution by position
  ProbeRows: array of array of TAnyJevMatrix;
  // Group 0: the request's state; group 1 + c: content-free probe c.
  Groups: array[0..csAnyJevContentFreeProbeCount] of TAnyJevPrompts;
  GroupCount: integer;
  Logits, LabelProbs: TNNetVolume;
  P: TDecisionProbabilities;
  Question: TDecisionQuestion;

  procedure AddPrompt(AGroupPos, AQuestionPos, ARotationPos, AProbePos: integer;
    const AStateText: string);
  var
    Diagnostics: TAnyJevQuestionDiagnostics;
    Prompt: TAnyJevPrompt;
    NewPos: integer;
  begin
    Diagnostics := FLastDiagnostics[AQuestionPos];
    Prompt.QuestionPos := AQuestionPos;
    Prompt.RotationPos := ARotationPos;
    Prompt.ProbePos := AProbePos;
    Prompt.Ids := FEngine.Tokenizer.Encode(RenderPrompt(BuildAnyJevUserMessage(
      RewriteSpecialTokenStrings(AStateText), QuestionText, Question.Kind,
      Options, Diagnostics.Rotations[ARotationPos], Diagnostics.Labels)));
    Prompt.PositionLabelIds := AnyJevPositionLabelIds(Question.Kind,
      Diagnostics.LabelIds, Diagnostics.Rotations[ARotationPos]);
    if Length(Prompt.Ids) > FEngine.SeqLen then
      raise EDecisionUnsupported.CreateFmt('question "%s": the AnyJev prompt' +
        ' of %d tokens exceeds the %d-token context',
        [Question.Id, Length(Prompt.Ids), FEngine.SeqLen]);
    NewPos := Length(Groups[AGroupPos]);
    SetLength(Groups[AGroupPos], NewPos + 1);
    Groups[AGroupPos][NewPos] := Prompt;
  end;

  // Feeds the group's common token prefix once, then each prompt's rest.
  procedure ScoreGroup(var Prompts: TAnyJevPrompts);
  var
    PrefixLen, MemberPos, MaxMemberPos: integer;
  begin
    MaxMemberPos := High(Prompts);
    PrefixLen := 0;
    if FSharePrefix then
    begin
      PrefixLen := Length(Prompts[0].Ids) - 1;
      for MemberPos := 1 to MaxMemberPos do
        PrefixLen := Min(PrefixLen, Min(Length(Prompts[MemberPos].Ids) - 1,
          CommonPrefixLen(Prompts[0].Ids, Prompts[MemberPos].Ids)));
    end;
    FEngine.PrefillDecisionPrefix(Copy(Prompts[0].Ids, 0, PrefixLen));
    InputTokens := InputTokens + PrefixLen;
    for MemberPos := 0 to MaxMemberPos do
    begin
      FEngine.ScoreDecisionBranch(Copy(Prompts[MemberPos].Ids, PrefixLen, MaxInt),
        nil, Logits, nil);
      InputTokens := InputTokens + Length(Prompts[MemberPos].Ids) - PrefixLen;
      RestrictedSoftMax(Logits, Prompts[MemberPos].PositionLabelIds, LabelProbs);
      if Prompts[MemberPos].ProbePos < 0 then
        FLastDiagnostics[Prompts[MemberPos].QuestionPos].RawByRotation[
          Prompts[MemberPos].RotationPos] := DecisionProbabilitiesFromVolume(LabelProbs)
      else
        ProbeRows[Prompts[MemberPos].QuestionPos][Prompts[MemberPos].RotationPos][
          Prompts[MemberPos].ProbePos] := DecisionProbabilitiesFromVolume(LabelProbs);
    end;
  end;

begin
  InputTokens := 0;
  OutputTokens := 0;
  Answers := nil;
  QuestionCount := Length(Request.Questions);
  SetLength(Answers, QuestionCount);
  FLastDiagnostics := nil;
  SetLength(FLastDiagnostics, QuestionCount);
  ProbeRows := nil;
  SetLength(ProbeRows, QuestionCount);
  for GroupPos := 0 to High(Groups) do Groups[GroupPos] := nil;
  if (FLevel = ajlL0) and (FPrior = ajpContentFree) then
    GroupCount := 1 + csAnyJevContentFreeProbeCount
  else
    GroupCount := 1;
  StateText := Request.State;
  MaxQuestionPos := QuestionCount - 1;
  for QuestionPos := 0 to MaxQuestionPos do
  begin
    Question := Request.Questions[QuestionPos];
    OptionCount := Length(Question.OptionNames);
    // One option (Jev allows it, AnyJev does not) needs no model call.
    if (Question.Kind <> dqkNoul) and (OptionCount = 1) then continue;
    if (Question.Kind = dqkChoice) and (OptionCount > csAnyJevMaxChoiceOptions) then
      raise EDecisionUnsupported.CreateFmt('question "%s": the AnyJev readout' +
        ' labels at most %d choice options (A-Z), not %d',
        [Question.Id, csAnyJevMaxChoiceOptions, OptionCount]);
    with FLastDiagnostics[QuestionPos] do
    begin
      ResolveAnyJevLabels(FEngine.Tokenizer, Question.Kind, OptionCount,
        Labels, LabelIds);
      Rotations := AnyJevRotations(Question.Kind, Length(Labels), FLevel, FMaxRotations);
      SetLength(RawByRotation, Length(Rotations));
      if GroupCount > 1 then
      begin
        SetLength(ProbeRows[QuestionPos], Length(Rotations));
        for RotationPos := 0 to High(Rotations) do
          SetLength(ProbeRows[QuestionPos][RotationPos], csAnyJevContentFreeProbeCount);
      end;
    end;
  end;
  // Prompts are built after every label check, so an unsupported request
  // fails before the model reads anything.
  for QuestionPos := 0 to MaxQuestionPos do
  begin
    if Length(FLastDiagnostics[QuestionPos].Rotations) = 0 then continue;
    Question := Request.Questions[QuestionPos];
    AnyJevQuestionParts(Question, QuestionText, Options);
    MaxRotationPos := High(FLastDiagnostics[QuestionPos].Rotations);
    for RotationPos := 0 to MaxRotationPos do
    begin
      AddPrompt(0, QuestionPos, RotationPos, -1, StateText);
      for ProbePos := 0 to GroupCount - 2 do
        AddPrompt(1 + ProbePos, QuestionPos, RotationPos, ProbePos,
          csAnyJevContentFreeProbes[ProbePos]);
    end;
  end;
  Logits := TNNetVolume.Create();
  LabelProbs := TNNetVolume.Create();
  try
    for GroupPos := 0 to GroupCount - 1 do
      if Length(Groups[GroupPos]) > 0 then ScoreGroup(Groups[GroupPos]);
  finally
    LabelProbs.Free;
    Logits.Free;
  end;
  for QuestionPos := 0 to MaxQuestionPos do
  begin
    Question := Request.Questions[QuestionPos];
    with FLastDiagnostics[QuestionPos] do
    begin
      if Length(Rotations) = 0 then
      begin
        P := nil;
        SetLength(P, 1);
        P[0] := 1;
      end
      else
      begin
        if GroupCount > 1 then
        begin
          SetLength(PriorByRotation, Length(Rotations));
          for RotationPos := 0 to High(Rotations) do
            PriorByRotation[RotationPos] :=
              AnyJevContentFreePrior(ProbeRows[QuestionPos][RotationPos]);
        end;
        Probabilities := AnyJevOptionProbabilities(RawByRotation, PriorByRotation,
          Rotations, csAnyJevContentFreePriorStrength);
        P := Copy(Probabilities);
        // Jev's noul order is (false, true).
        if Question.Kind = dqkNoul then
        begin
          P[0] := Probabilities[1];
          P[1] := Probabilities[0];
        end;
      end;
    end;
    Answers[QuestionPos] := DecisionAnswerFromProbabilities(Question.Kind, P);
  end;
end;

end.
