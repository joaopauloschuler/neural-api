unit TestNeuralAnyJev;
(*
Tests for neuralanyjev.pas, the AnyJev L0 readout. The model-free half ports
anyjev 0.3.0's tests/test_calibrate.py and the synthetic-bias checks of
tests/test_decider_fake.py; expected prompt strings and probabilities come
from the real anyjev package (readout.build_prompt and the calibrate
routines). The engine half runs the scorer on a random pico Qwen2 chat model.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, fpjson, jsonparser,
  neuralvolume, neuralhftokenizer, neuralchat, neuralchatengine,
  neuralsafetensors, neuraldecision, neuralanyjev;

type
  TTestNeuralAnyJev = class(TTestCase)
  private
    function FixturePath(const FileName: string): string;
    // A random pico Qwen2 whose vocabulary covers the fixture tokenizer plus
    // the added tokens <|im_start|>, Yes and No; ChatML template.
    function MakeAnyJevModelDir(): string;
    procedure RemoveAnyJevModelDir(const Dir: string);
    function NewEngine(const Dir: string): TChatEngine;
    procedure AssertProbabilities(const Msg: string;
      const Expected: array of double; const Actual: TDecisionProbabilities;
      Tolerance: double);
  published
    procedure TestContextualRemovesAdditivePriorExactly;
    // README: raw P(Yes) 0.62 with prior 0.70 -> 0.4115.
    procedure TestReadmeWorkedExample;
    procedure TestCyclicShiftsCoverEveryPosition;
    procedure TestMarginalizeMapsPositionsBackToOptions;
    procedure TestLogMeanRemovesAdditivePositionBiasExactly;
    procedure TestMeanIsInvariantToRotationOnly;
    procedure TestRawIsFooledByPositionBiasL0IsNot;
    procedure TestContentFreePriorRemovesLabelPriorExactly;
    procedure TestNoulPhrasingSwapKeepsYesAttachedToYes;
    // Prior + correction + log-mean on fixed matrices, against numpy.
    procedure TestOptionProbabilitiesMatchAnyJev;
    procedure TestRotationsAndLabels;
    procedure TestUserMessagesMatchAnyJev;
    procedure TestQuestionPartsFromJev;
    procedure TestLabelTokenMapping;
    // Choice, score and noul end to end: sums, sharing on = off, raw level,
    // 422 cases, and the engine chats afterwards like a fresh one.
    procedure TestScorerOnPicoChatModel;
  end;

implementation

const
  csPicoVocab = 403;
  csPicoHidden = 8;
  csPicoKVDim = 4;
  csPicoIntermediate = 12;

var
  TestFormat: TFormatSettings;

// Double literal without FPC's constant folding to single.
function D(const S: string): double;
begin
  Result := StrToFloat(S, TestFormat);
end;

function SoftMaxOf(const Logits: array of double): TDecisionProbabilities;
var
  ItemPos: integer;
  Largest, Total: double;
begin
  Result := nil;
  SetLength(Result, Length(Logits));
  Largest := MaxValue(Logits);
  Total := 0;
  for ItemPos := 0 to High(Logits) do
  begin
    Result[ItemPos] := Exp(Logits[ItemPos] - Largest);
    Total := Total + Result[ItemPos];
  end;
  for ItemPos := 0 to High(Logits) do
    Result[ItemPos] := Result[ItemPos] / Total;
end;

function Rotation(const Items: array of integer): TAnyJevRotation;
var
  ItemPos: integer;
begin
  Result := nil;
  SetLength(Result, Length(Items));
  for ItemPos := 0 to High(Items) do Result[ItemPos] := Items[ItemPos];
end;

function Row(const Items: array of double): TDecisionProbabilities;
var
  ItemPos: integer;
begin
  Result := nil;
  SetLength(Result, Length(Items));
  for ItemPos := 0 to High(Items) do Result[ItemPos] := Items[ItemPos];
end;

// The test_calibrate.py fake: softmax(content[order][perm] + bias) per cyclic
// shift, marginalised, then mapped back to the original listing.
function RunMarginalize(const Content, PositionBias: array of double;
  const Order: array of integer; Combine: TAnyJevCombine): TDecisionProbabilities;
var
  Rotations: TAnyJevRotations;
  ByRotation: TAnyJevMatrix;
  Logits: array of double;
  Marginal: TDecisionProbabilities;
  RotationPos, PositionPos, OptionCount: integer;
begin
  OptionCount := Length(Content);
  Rotations := AnyJevCyclicShifts(OptionCount, 0);
  ByRotation := nil;
  SetLength(ByRotation, OptionCount);
  Logits := nil;
  SetLength(Logits, OptionCount);
  for RotationPos := 0 to OptionCount - 1 do
  begin
    for PositionPos := 0 to OptionCount - 1 do
      Logits[PositionPos] := Content[Order[Rotations[RotationPos][PositionPos]]] +
        PositionBias[PositionPos];
    ByRotation[RotationPos] := SoftMaxOf(Logits);
  end;
  Marginal := AnyJevMarginalize(ByRotation, Rotations, Combine);
  Result := nil;
  SetLength(Result, OptionCount);
  for PositionPos := 0 to OptionCount - 1 do
    Result[Order[PositionPos]] := Marginal[PositionPos];
end;

function TTestNeuralAnyJev.FixturePath(const FileName: string): string;
begin
  Result := 'fixtures' + DirectorySeparator + FileName;
  if not FileExists(Result) then
    Result := 'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      FileName;
  if not FileExists(Result) then Fail('Fixture not found: ' + FileName);
end;

procedure TTestNeuralAnyJev.AssertProbabilities(const Msg: string;
  const Expected: array of double; const Actual: TDecisionProbabilities;
  Tolerance: double);
var
  ItemPos: integer;
begin
  AssertEquals(Msg + ': length', Length(Expected), Length(Actual));
  for ItemPos := 0 to High(Expected) do
    AssertEquals(Msg + ' [' + IntToStr(ItemPos) + ']', Expected[ItemPos],
      Actual[ItemPos], Tolerance);
end;

procedure TTestNeuralAnyJev.TestContextualRemovesAdditivePriorExactly;
var
  Content, Bias, Sum: array of double;
  Probes: TAnyJevMatrix;
  Prior, Corrected, Expected: TDecisionProbabilities;
  ItemPos: integer;
begin
  Content := nil;
  SetLength(Content, 3);
  Content[0] := 1.0; Content[1] := D('0.2'); Content[2] := -0.5;
  SetLength(Bias, 3);
  Bias[0] := 2.0; Bias[1] := -1.0; Bias[2] := 0.5;
  SetLength(Sum, 3);
  for ItemPos := 0 to 2 do Sum[ItemPos] := Content[ItemPos] + Bias[ItemPos];
  Probes := nil;
  SetLength(Probes, 3);
  for ItemPos := 0 to 2 do Probes[ItemPos] := SoftMaxOf(Bias);
  Prior := AnyJevContentFreePrior(Probes);
  Corrected := AnyJevApplyPrior(SoftMaxOf(Sum), Prior, 1.0);
  Expected := SoftMaxOf(Content);
  AssertProbabilities('additive prior removed', Expected, Corrected, 1e-9);
end;

procedure TTestNeuralAnyJev.TestReadmeWorkedExample;
var
  Corrected: TDecisionProbabilities;
begin
  Corrected := AnyJevApplyPrior(Row([D('0.62'), D('0.38')]),
    Row([D('0.70'), D('0.30')]), 1.0);
  AssertEquals('0.62 raw, 0.70 prior', D('0.4115'), Corrected[0], 1e-3);
end;

procedure TTestNeuralAnyJev.TestCyclicShiftsCoverEveryPosition;
var
  Shifts: TAnyJevRotations;
  PositionPos, ShiftPos: integer;
  Seen: array[0..4] of boolean;
begin
  Shifts := AnyJevCyclicShifts(5, 0);
  AssertEquals('one shift per option', 5, Length(Shifts));
  for PositionPos := 0 to 4 do
  begin
    FillChar(Seen, SizeOf(Seen), 0);
    for ShiftPos := 0 to 4 do Seen[Shifts[ShiftPos][PositionPos]] := true;
    for ShiftPos := 0 to 4 do
      AssertTrue('every option at position ' + IntToStr(PositionPos), Seen[ShiftPos]);
  end;
  AssertEquals('shift 1, position 4', 0, Shifts[1][4]);
  AssertEquals('max_permutations=2', 2, Length(AnyJevCyclicShifts(5, 2)));
  AssertEquals('max_permutations above K', 5, Length(AnyJevCyclicShifts(5, 9)));
end;

procedure TTestNeuralAnyJev.TestMarginalizeMapsPositionsBackToOptions;
var
  ByRotation: TAnyJevMatrix;
  RotationPos: integer;
  Third: double;
begin
  // A model that always puts all mass on position 0.
  ByRotation := nil;
  SetLength(ByRotation, 3);
  for RotationPos := 0 to 2 do ByRotation[RotationPos] := Row([1, 0, 0]);
  Third := 1 / 3;
  AssertProbabilities('position 0 bias marginalises to uniform',
    [Third, Third, Third], AnyJevMarginalize(ByRotation, AnyJevCyclicShifts(3, 0)), 1e-12);
end;

procedure TTestNeuralAnyJev.TestLogMeanRemovesAdditivePositionBiasExactly;
const
  Content: array[0..3] of double = (0.3, 1.2, -0.4, 0.9);
  PositionBias: array[0..3] of double = (1.5, 0.0, 0.0, -0.5);
var
  Listed, Shuffled, Expected: TDecisionProbabilities;
begin
  Listed := RunMarginalize(Content, PositionBias, [0, 1, 2, 3], ajcLogMean);
  Expected := SoftMaxOf(Content);
  AssertProbabilities('log-mean = softmax(content)', Expected, Listed, 1e-12);
  Shuffled := RunMarginalize(Content, PositionBias, [2, 0, 3, 1], ajcLogMean);
  AssertProbabilities('invariant to the listing order', Listed, Shuffled, 1e-12);
end;

procedure TTestNeuralAnyJev.TestMeanIsInvariantToRotationOnly;
const
  Content: array[0..3] of double = (0.3, 1.2, -0.4, 0.9);
  PositionBias: array[0..3] of double = (1.5, 0.0, 0.0, -0.5);
var
  Listed, Rotated, Other: TDecisionProbabilities;
  ItemPos: integer;
  Differs: boolean;
begin
  Listed := RunMarginalize(Content, PositionBias, [0, 1, 2, 3], ajcMean);
  Rotated := RunMarginalize(Content, PositionBias, [1, 2, 3, 0], ajcMean);
  Other := RunMarginalize(Content, PositionBias, [2, 0, 3, 1], ajcMean);
  AssertProbabilities('mean: invariant to a rotation', Listed, Rotated, 1e-12);
  Differs := false;
  for ItemPos := 0 to 3 do
    if Abs(Listed[ItemPos] - Other[ItemPos]) > 1e-6 then Differs := true;
  AssertTrue('mean: not invariant to another order', Differs);
  AssertTrue('mean: argmax is option 1', Listed[1] = MaxValue(Listed));
end;

// test_decider_fake: logit(position j) = content(option at j) + bias[j], with
// position A loved by 4 logits; the right option is "technical".
procedure TTestNeuralAnyJev.TestRawIsFooledByPositionBiasL0IsNot;
const
  Content: array[0..3] of double = (0.0, 3.0, 0.0, 0.0);
  PositionBias: array[0..3] of double = (4.0, 0.0, 0.0, 0.0);
var
  Raw, L0: TDecisionProbabilities;
begin
  Raw := SoftMaxOf([Content[0] + PositionBias[0], Content[1], Content[2], Content[3]]);
  AssertTrue('raw picks position A', Raw[0] = MaxValue(Raw));
  L0 := RunMarginalize(Content, PositionBias, [0, 1, 2, 3], ajcLogMean);
  AssertTrue('L0 picks technical', L0[1] = MaxValue(L0));
end;

// Noul with a +2 label prior on Yes and no content: the content-free probes
// see exactly that prior, so L0 answers 0.5.
procedure TTestNeuralAnyJev.TestContentFreePriorRemovesLabelPriorExactly;
var
  Rotations: TAnyJevRotations;
  RawByRotation, PriorByRotation, Probes: TAnyJevMatrix;
  RotationPos, PositionPos, ProbePos: integer;
  Logits: array[0..1] of double;
  P: TDecisionProbabilities;
begin
  Rotations := AnyJevRotations(dqkNoul, 2, ajlL0, 0);
  RawByRotation := nil;
  SetLength(RawByRotation, 2);
  PriorByRotation := nil;
  SetLength(PriorByRotation, 2);
  for RotationPos := 0 to 1 do
  begin
    for PositionPos := 0 to 1 do
      if Rotations[RotationPos][PositionPos] = 0 then Logits[PositionPos] := 2.0
      else Logits[PositionPos] := 0.0;
    RawByRotation[RotationPos] := SoftMaxOf(Logits);
    Probes := nil;
    SetLength(Probes, csAnyJevContentFreeProbeCount);
    for ProbePos := 0 to csAnyJevContentFreeProbeCount - 1 do
      Probes[ProbePos] := SoftMaxOf(Logits);
    PriorByRotation[RotationPos] := AnyJevContentFreePrior(Probes);
  end;
  P := AnyJevOptionProbabilities(RawByRotation, nil, Rotations, 1.0);
  AssertTrue('raw p(Yes) > 0.85', P[0] > 0.85);
  P := AnyJevOptionProbabilities(RawByRotation, PriorByRotation, Rotations, 1.0);
  AssertEquals('L0 p(Yes)', 0.5, P[0], 1e-6);
end;

procedure TTestNeuralAnyJev.TestNoulPhrasingSwapKeepsYesAttachedToYes;
var
  Rotations: TAnyJevRotations;
  RawByRotation: TAnyJevMatrix;
  LabelIds, PositionIds: TNeuralIntegerArray;
  P: TDecisionProbabilities;
begin
  // Yes = id 7, No = id 9; the second phrasing ("No or Yes") reads No first.
  LabelIds := nil;
  SetLength(LabelIds, 2);
  LabelIds[0] := 7;
  LabelIds[1] := 9;
  Rotations := AnyJevRotations(dqkNoul, 2, ajlL0, 0);
  PositionIds := AnyJevPositionLabelIds(dqkNoul, LabelIds, Rotations[1]);
  AssertEquals('No or Yes: position 0 reads No', 9, PositionIds[0]);
  AssertEquals('No or Yes: position 1 reads Yes', 7, PositionIds[1]);
  AssertEquals('Yes or No: position 0 reads Yes', 7,
    AnyJevPositionLabelIds(dqkNoul, LabelIds, Rotations[0])[0]);
  AssertEquals('choice ids are positional', 9,
    AnyJevPositionLabelIds(dqkChoice, LabelIds, Rotations[1])[1]);
  // A model reading the label: 2 logits on Yes wherever it sits.
  RawByRotation := nil;
  SetLength(RawByRotation, 2);
  RawByRotation[0] := SoftMaxOf([2.0, 0.0]);
  RawByRotation[1] := SoftMaxOf([0.0, 2.0]);
  P := AnyJevOptionProbabilities(RawByRotation, nil, Rotations, 1.0);
  AssertTrue('p(Yes) > 0.8', P[0] > 0.8);
  AssertEquals('p(Yes) = the per-phrasing value', RawByRotation[0][0], P[0], 1e-12);
end;

procedure TTestNeuralAnyJev.TestOptionProbabilitiesMatchAnyJev;
var
  Raw, Probes, PriorByRotation: TAnyJevMatrix;
  ProbeSets: array[0..2] of TAnyJevMatrix;
  RotationPos: integer;
begin
  Raw := nil;
  SetLength(Raw, 3);
  Raw[0] := Row([D('0.5'), D('0.3'), D('0.2')]);
  Raw[1] := Row([D('0.6'), D('0.1'), D('0.3')]);
  Raw[2] := Row([D('0.25'), D('0.25'), D('0.5')]);
  for RotationPos := 0 to 2 do
  begin
    ProbeSets[RotationPos] := nil;
    SetLength(ProbeSets[RotationPos], 3);
  end;
  ProbeSets[0][0] := Row([D('0.7'), D('0.2'), D('0.1')]);
  ProbeSets[0][1] := Row([D('0.6'), D('0.3'), D('0.1')]);
  ProbeSets[0][2] := Row([D('0.5'), D('0.25'), D('0.25')]);
  ProbeSets[1][0] := Row([D('0.4'), D('0.4'), D('0.2')]);
  ProbeSets[1][1] := Row([D('0.3'), D('0.3'), D('0.4')]);
  ProbeSets[1][2] := Row([D('0.5'), D('0.0'), D('0.5')]);
  ProbeSets[2][0] := Row([D('0.2'), D('0.3'), D('0.5')]);
  ProbeSets[2][1] := Row([D('0.1'), D('0.1'), D('0.8')]);
  ProbeSets[2][2] := Row([D('1e-12'), D('0.5'), D('0.5')]);
  PriorByRotation := nil;
  SetLength(PriorByRotation, 3);
  for RotationPos := 0 to 2 do
  begin
    Probes := ProbeSets[RotationPos];
    PriorByRotation[RotationPos] := AnyJevContentFreePrior(Probes);
  end;
  AssertProbabilities('prior, rotation 2 (floored at 1e-8)',
    [D('0.1000000000003'), D('0.2999999999999'), D('0.5999999999998')],
    PriorByRotation[2], 1e-15);
  AssertProbabilities('choice L0',
    [D('0.2672457914757906'), D('0.3693563051074056'), D('0.3633979034168037')],
    AnyJevOptionProbabilities(Raw, PriorByRotation, AnyJevCyclicShifts(3, 0), 1.0),
    1e-14);
  Raw := nil;
  SetLength(Raw, 2);
  Raw[0] := Row([D('0.62'), D('0.38')]);
  Raw[1] := Row([D('0.45'), D('0.55')]);
  PriorByRotation := nil;
  SetLength(PriorByRotation, 2);
  PriorByRotation[0] := Row([D('0.70'), D('0.30')]);
  PriorByRotation[1] := Row([D('0.4'), D('0.6')]);
  AssertProbabilities('noul L0', [D('0.4301419500928873'), D('0.5698580499071126')],
    AnyJevOptionProbabilities(Raw, PriorByRotation,
      AnyJevRotations(dqkNoul, 2, ajlL0, 0), 1.0), 1e-14);
end;

procedure TTestNeuralAnyJev.TestRotationsAndLabels;
var
  Labels: TStringArray;
  Raised: boolean;
begin
  AssertEquals('choice L0: K shifts', 4, Length(AnyJevRotations(dqkChoice, 4, ajlL0, 0)));
  AssertEquals('choice L0 capped', 2, Length(AnyJevRotations(dqkChoice, 4, ajlL0, 2)));
  AssertEquals('choice raw: one', 1, Length(AnyJevRotations(dqkChoice, 4, ajlRaw, 0)));
  AssertEquals('score L0: identity only', 1, Length(AnyJevRotations(dqkScore, 5, ajlL0, 0)));
  AssertEquals('noul L0: both orders', 2, Length(AnyJevRotations(dqkNoul, 2, ajlL0, 1)));
  AssertEquals('noul raw: Yes or No', 0, AnyJevRotations(dqkNoul, 2, ajlRaw, 0)[0][0]);
  Labels := AnyJevAnswerLabels(dqkScore, 10);
  AssertEquals('score label 10', '10', Labels[9]);
  Labels := AnyJevAnswerLabels(dqkChoice, 26);
  AssertEquals('choice label 26', 'Z', Labels[25]);
  Labels := AnyJevAnswerLabels(dqkNoul, 2);
  AssertEquals('noul labels', 'Yes,No', Labels[0] + ',' + Labels[1]);
  Raised := false;
  try
    AnyJevAnswerLabels(dqkChoice, 27);
  except
    on EDecisionUnsupported do Raised := true;
  end;
  AssertTrue('27 letters are unsupported', Raised);
end;

procedure TTestNeuralAnyJev.TestUserMessagesMatchAnyJev;
begin
  AssertEquals('choice, rotation (1, 2, 0)',
    'State:'#10'The card was declined twice.'#10#10'Question: Which team?'#10 +
    'Options:'#10'A. technical: app crashes'#10'B. sales'#10'C. billing'#10 +
    'Answer with the letter only.',
    BuildAnyJevUserMessage('The card was declined twice.', 'Which team?', dqkChoice,
      ['billing', 'technical: app crashes', 'sales'], Rotation([1, 2, 0]),
      ['A', 'B', 'C']));
  AssertEquals('score, digits, empty state',
    'State:'#10'(empty)'#10#10'Question: How urgent?'#10 +
    'Pick the level that applies (the levels are ordered):'#10 +
    '1. low'#10'2. medium'#10'3. high'#10'Answer with the number only.',
    BuildAnyJevUserMessage('', 'How urgent?', dqkScore, ['low', 'medium', 'high'],
      Rotation([0, 1, 2]), ['1', '2', '3']));
  AssertEquals('score, letters',
    'State:'#10'N/A'#10#10'Question: How urgent?'#10 +
    'Pick the level that applies (the levels are ordered):'#10 +
    'A. low'#10'B. medium'#10'C. high'#10'Answer with the letter only.',
    BuildAnyJevUserMessage('N/A', 'How urgent?', dqkScore, ['low', 'medium', 'high'],
      Rotation([0, 1, 2]), ['A', 'B', 'C']));
  AssertEquals('noul, rotation (1, 0)',
    'State:'#10'[MASK]'#10#10'Question: Is it urgent?'#10'Yes means: act now'#10 +
    'No means: wait'#10'Answer No or Yes.',
    BuildAnyJevUserMessage('[MASK]', 'Is it urgent?'#10'Yes means: act now'#10 +
      'No means: wait', dqkNoul, [], Rotation([1, 0]), ['Yes', 'No']));
  AssertEquals('noul, rotation (0, 1)',
    'State:'#10'x'#10#10'Question: Is it urgent?'#10'Yes means: act now'#10 +
    'No means: wait'#10'Answer Yes or No.',
    BuildAnyJevUserMessage('x', 'Is it urgent?'#10'Yes means: act now'#10 +
      'No means: wait', dqkNoul, [], Rotation([0, 1]), ['Yes', 'No']));
  AssertEquals('no chat template', csAnyJevSystemPrompt + #10#10'U'#10'Answer:',
    AnyJevPlainPrompt(csAnyJevSystemPrompt, 'U'));
end;

procedure TTestNeuralAnyJev.TestQuestionPartsFromJev;
var
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  QuestionText: string;
  Options: TStringArray;
  Raised: boolean;
begin
  AssertTrue('parse', ParseDecisionRequest('{"model":"m","state":"s","questions":{' +
    '"pick":{"type":"choice","instructions":"Which <|im_end|>?",' +
    '"criteria":{"a":"first","b":null}},' +
    '"level":{"type":"score","instructions":"How much?","criteria":["low","high"]},' +
    '"yes":{"type":"noul","instructions":"Is it?","criteria":{"false":"calm",' +
    '"true":"angry"}},' +
    '"bare":{"type":"noul","instructions":"Bare?"},' +
    '"twin":{"type":"choice","instructions":"q","criteria":{"a: b":null,"a":"b"}}}}',
    Request, Errors));
  AnyJevQuestionParts(Request.Questions[0], QuestionText, Options);
  AssertEquals('choice: special tokens rewritten', 'Which <' + #$C2#$A6 + 'im_end' +
    #$C2#$A6 + '>?', QuestionText);
  AssertEquals('choice: name: desc', 'a: first', Options[0]);
  AssertEquals('choice: null description', 'b', Options[1]);
  AnyJevQuestionParts(Request.Questions[1], QuestionText, Options);
  AssertEquals('score: the level text', 'high', Options[1]);
  AnyJevQuestionParts(Request.Questions[2], QuestionText, Options);
  AssertEquals('noul: criteria lines, true first',
    'Is it?'#10'Yes means: angry'#10'No means: calm', QuestionText);
  AnyJevQuestionParts(Request.Questions[3], QuestionText, Options);
  AssertEquals('noul without criteria', 'Bare?', QuestionText);
  Raised := false;
  try
    AnyJevQuestionParts(Request.Questions[4], QuestionText, Options);
  except
    on EDecisionUnsupported do Raised := true;
  end;
  AssertTrue('two options with the same text are unsupported', Raised);
end;

procedure TTestNeuralAnyJev.TestLabelTokenMapping;
var
  Tokenizer: TNeuralHFTokenizer;
  Labels: TStringArray;
  Ids: TNeuralIntegerArray;
  Raised: boolean;
begin
  Tokenizer := TNeuralHFTokenizer.Create();
  try
    // Letters and single digits are one token; "10", "Yes" and "No" are not.
    Tokenizer.LoadFromFile(FixturePath('tiny_bpe_split_qwen2_tokenizer.json'));
    Ids := MapAnyJevLabelTokens(Tokenizer, ['A', 'B']);
    AssertEquals('A', Tokenizer.Encode('A')[0], Ids[0]);
    ResolveAnyJevLabels(Tokenizer, dqkScore, 3, Labels, Ids);
    AssertEquals('score 3: digits', '1', Labels[0]);
    ResolveAnyJevLabels(Tokenizer, dqkScore, 10, Labels, Ids);
    AssertEquals('score 10: letters', 'J', Labels[9]);
    AssertEquals('score 10: ten ids', 10, Length(Ids));
    Raised := false;
    try
      ResolveAnyJevLabels(Tokenizer, dqkNoul, 2, Labels, Ids);
    except
      on EAnyJevLabelTokenError do Raised := true;
    end;
    AssertTrue('Yes is two tokens: error', Raised);
    Raised := false;
    try
      MapAnyJevLabelTokens(Tokenizer, ['A', 'A']);
    except
      on EAnyJevLabelTokenError do Raised := true;
    end;
    AssertTrue('colliding labels: error', Raised);
  finally
    Tokenizer.Free;
  end;
end;

function TTestNeuralAnyJev.MakeAnyJevModelDir(): string;
const
  ChatMLTemplate = '{% for message in messages %}{{''<|im_start|>'' + message[''role''] +' +
    ' ''\n'' + message[''content''] + ''<|im_end|>'' + ''\n''}}{% endfor %}' +
    '{% if add_generation_prompt %}{{ ''<|im_start|>assistant\n'' }}{% endif %}';
var
  Writer: TNNetSafeTensorsWriter;
  Config, TokenizerJSON, TokenizerConfig: TJSONObject;
  AddedTokens: TJSONArray;
  Lines: TStringList;
  LayerPos: integer;
  Prefix: string;

  procedure AddRandom(const Name: string; const Shape: array of Int64; IsNorm: boolean);
  var
    V: TNNetVolume;
    Count, ItemPos: integer;
  begin
    Count := 1;
    for ItemPos := 0 to High(Shape) do Count := Count * Shape[ItemPos];
    V := TNNetVolume.Create(Count, 1, 1);
    try
      for ItemPos := 0 to Count - 1 do
        if IsNorm then V.FData[ItemPos] := 1
        else V.FData[ItemPos] := Random - 0.5;
      Writer.AddTensorFlat(Name, Shape, V);
    finally
      V.Free;
    end;
  end;

  procedure AddToken(Id: integer; const Content: string; IsSpecial: boolean);
  begin
    AddedTokens.Add(TJSONObject.Create(['id', Id, 'content', Content,
      'single_word', false, 'lstrip', false, 'rstrip', false,
      'normalized', not IsSpecial, 'special', IsSpecial]));
  end;

  procedure SaveJSON(Data: TJSONData; const FileName: string);
  begin
    Lines.Text := Data.AsJSON;
    Lines.SaveToFile(FileName);
  end;

begin
  RandSeed := 20261018;
  Result := IncludeTrailingPathDelimiter(GetTempDir(false)) +
    'cai_anyjev_' + IntToStr(Random(1000000)) + DirectorySeparator;
  ForceDirectories(Result);
  Lines := TStringList.Create();
  Config := TJSONObject.Create(['architectures', TJSONArray.Create(['Qwen2ForCausalLM']),
    'model_type', 'qwen2', 'hidden_size', csPicoHidden,
    'intermediate_size', csPicoIntermediate, 'num_hidden_layers', 2,
    'num_attention_heads', 2, 'num_key_value_heads', 1, 'vocab_size', csPicoVocab,
    'max_position_embeddings', 2048, 'rms_norm_eps', 1e-05, 'rope_theta', 10000.0,
    'use_sliding_window', false, 'tie_word_embeddings', false, 'hidden_act', 'silu']);
  TokenizerJSON := nil;
  TokenizerConfig := TJSONObject.Create(['chat_template', ChatMLTemplate]);
  Writer := TNNetSafeTensorsWriter.Create(Result + 'model.safetensors');
  try
    SaveJSON(Config, Result + 'config.json');
    SaveJSON(TokenizerConfig, Result + 'tokenizer_config.json');
    Lines.LoadFromFile(FixturePath('tiny_bpe_split_qwen2_tokenizer.json'));
    TokenizerJSON := GetJSON(Lines.Text) as TJSONObject;
    AddedTokens := TokenizerJSON.Arrays['added_tokens'];
    AddToken(400, '<|im_start|>', true);
    AddToken(401, 'Yes', false);
    AddToken(402, 'No', false);
    SaveJSON(TokenizerJSON, Result + 'tokenizer.json');
    AddRandom('model.embed_tokens.weight', [csPicoVocab, csPicoHidden], false);
    AddRandom('lm_head.weight', [csPicoVocab, csPicoHidden], false);
    AddRandom('model.norm.weight', [csPicoHidden], true);
    for LayerPos := 0 to 1 do
    begin
      Prefix := 'model.layers.' + IntToStr(LayerPos) + '.';
      AddRandom(Prefix + 'input_layernorm.weight', [csPicoHidden], true);
      AddRandom(Prefix + 'post_attention_layernorm.weight', [csPicoHidden], true);
      AddRandom(Prefix + 'mlp.down_proj.weight', [csPicoHidden, csPicoIntermediate], false);
      AddRandom(Prefix + 'mlp.gate_proj.weight', [csPicoIntermediate, csPicoHidden], false);
      AddRandom(Prefix + 'mlp.up_proj.weight', [csPicoIntermediate, csPicoHidden], false);
      AddRandom(Prefix + 'self_attn.q_proj.weight', [csPicoHidden, csPicoHidden], false);
      AddRandom(Prefix + 'self_attn.q_proj.bias', [csPicoHidden], false);
      AddRandom(Prefix + 'self_attn.k_proj.weight', [csPicoKVDim, csPicoHidden], false);
      AddRandom(Prefix + 'self_attn.k_proj.bias', [csPicoKVDim], false);
      AddRandom(Prefix + 'self_attn.v_proj.weight', [csPicoKVDim, csPicoHidden], false);
      AddRandom(Prefix + 'self_attn.v_proj.bias', [csPicoKVDim], false);
      AddRandom(Prefix + 'self_attn.o_proj.weight', [csPicoHidden, csPicoHidden], false);
    end;
    Writer.SaveToFile;
  finally
    Writer.Free;
    TokenizerJSON.Free;
    TokenizerConfig.Free;
    Config.Free;
    Lines.Free;
  end;
end;

procedure TTestNeuralAnyJev.RemoveAnyJevModelDir(const Dir: string);
begin
  DeleteFile(Dir + 'model.safetensors');
  DeleteFile(Dir + 'config.json');
  DeleteFile(Dir + 'tokenizer.json');
  DeleteFile(Dir + 'tokenizer_config.json');
  RemoveDir(Dir);
end;

function TTestNeuralAnyJev.NewEngine(const Dir: string): TChatEngine;
var
  Args: TStringList;
  Opt: TChatOptions;
  ErrorMsg: string;
begin
  Result := TChatEngine.Create();
  Args := TStringList.Create();
  try
    Args.AddStrings(['--greedy', '--fp32', '--cpu', '--serial', '--ctx', '1024',
      '--max-new-tokens', '3']);
    Args.Insert(0, Dir);
    AssertTrue('chat options parse', ParseArgs(Args, Opt));
    AssertTrue('LoadModel', Result.LoadModel(Opt, ErrorMsg));
  except
    Result.Free;
    Args.Free;
    raise;
  end;
  Args.Free;
end;

procedure TTestNeuralAnyJev.TestScorerOnPicoChatModel;
const
  RequestBody = '{"model":"anyjev-test","state":"The card was declined twice.",' +
    '"questions":{' +
    '"team":{"type":"choice","instructions":"Which team?",' +
    '"criteria":{"billing":"payments","technical":null,"sales":"buying more"}},' +
    '"urgency":{"type":"score","instructions":"How urgent?",' +
    '"criteria":["low","medium","high"]},' +
    '"angry":{"type":"noul","instructions":"Is the customer angry?",' +
    '"criteria":{"true":"raised voice","false":"calm"}},' +
    '"single":{"type":"choice","instructions":"Only one","criteria":{"only":null}}}}';
var
  Dir, Reply, FreshReply: string;
  Engine, Fresh: TChatEngine;
  Scorer, RawScorer: TAnyJevL0Scorer;
  Request, BigRequest: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  Answers, UnsharedAnswers: TDecisionAnswers;
  Shared: TAnyJevDiagnostics;
  InputTokens, OutputTokens, UnsharedInputTokens, QuestionPos: integer;
  RotationPos, ItemPos: integer;
  Total: double;
  Criteria: string;
  Msgs: TChatMessages;
  Raised: boolean;
begin
  Dir := MakeAnyJevModelDir();
  Engine := nil;
  Fresh := nil;
  Scorer := nil;
  RawScorer := nil;
  try
    Engine := NewEngine(Dir);
    Fresh := NewEngine(Dir);
    AssertTrue('ChatML detected', Engine.ChatFormat = cfChatML);
    Scorer := TAnyJevL0Scorer.Create(Engine, 'anyjev-test');
    AssertEquals('ChatML render',
      '<|im_start|>system'#10 + csAnyJevSystemPrompt + '<|im_end|>'#10 +
      '<|im_start|>user'#10'U<|im_end|>'#10'<|im_start|>assistant'#10,
      Scorer.RenderPrompt('U'));
    AssertTrue('parse', ParseDecisionRequest(RequestBody, Request, Errors));
    Scorer.Score(Request, Answers, InputTokens, OutputTokens);
    Shared := Copy(Scorer.LastDiagnostics);
    AssertEquals('nothing generated', 0, OutputTokens);
    AssertEquals('choice: 3 rotations', 3, Length(Shared[0].Rotations));
    AssertEquals('score: 1 rotation', 1, Length(Shared[1].Rotations));
    AssertEquals('score: digit labels', '1', Shared[1].Labels[0]);
    AssertEquals('noul: 2 rotations', 2, Length(Shared[2].Rotations));
    AssertEquals('noul: Yes is the added token', 401, Shared[2].LabelIds[0]);
    AssertEquals('one option: no model call', 0, Length(Shared[3].Rotations));
    AssertEquals('one option: certain', 1.0, Answers[3].Probabilities[0], 0);
    AssertEquals('noul: a prior row per rotation', 2, Length(Shared[2].PriorByRotation));
    for QuestionPos := 0 to 2 do
    begin
      Total := 0;
      for ItemPos := 0 to High(Answers[QuestionPos].Probabilities) do
        Total := Total + Answers[QuestionPos].Probabilities[ItemPos];
      AssertEquals('question ' + IntToStr(QuestionPos) + ' sums to 1', 1.0, Total, 1e-9);
    end;
    AssertEquals('noul answer = p(Yes)', Shared[2].Probabilities[0], Answers[2].Noul, 1e-12);

    Scorer.SharePrefix := false;
    Scorer.Score(Request, UnsharedAnswers, UnsharedInputTokens, OutputTokens);
    AssertTrue('sharing feeds fewer tokens', InputTokens < UnsharedInputTokens);
    for QuestionPos := 0 to 2 do
    begin
      for RotationPos := 0 to High(Shared[QuestionPos].Rotations) do
        AssertProbabilities('raw rows, sharing on = off',
          Shared[QuestionPos].RawByRotation[RotationPos],
          Scorer.LastDiagnostics[QuestionPos].RawByRotation[RotationPos], 1e-5);
      AssertProbabilities('answers, sharing on = off',
        Answers[QuestionPos].Probabilities, UnsharedAnswers[QuestionPos].Probabilities, 1e-5);
    end;

    RawScorer := TAnyJevL0Scorer.Create(Engine, 'anyjev-test', ajlRaw);
    RawScorer.Score(Request, Answers, InputTokens, OutputTokens);
    AssertEquals('raw: one choice prompt', 1, Length(RawScorer.LastDiagnostics[0].Rotations));
    AssertEquals('raw: no prior', 0, Length(RawScorer.LastDiagnostics[0].PriorByRotation));
    AssertProbabilities('raw: rotation 0 of L0', Shared[0].RawByRotation[0],
      RawScorer.LastDiagnostics[0].RawByRotation[0], 1e-5);

    // 27 choice options and an over-long state are a 422, before any model call.
    Criteria := '';
    for ItemPos := 0 to 26 do
    begin
      if ItemPos > 0 then Criteria := Criteria + ',';
      Criteria := Criteria + '"o' + IntToStr(ItemPos) + '":null';
    end;
    AssertTrue('parse 27', ParseDecisionRequest('{"model":"m","state":"s","questions":' +
      '{"q":{"type":"choice","criteria":{' + Criteria + '}}}}', BigRequest, Errors));
    Raised := false;
    try
      Scorer.Score(BigRequest, Answers, InputTokens, OutputTokens);
    except
      on EDecisionUnsupported do Raised := true;
    end;
    AssertTrue('27 options: EDecisionUnsupported', Raised);
    BigRequest := Request;
    BigRequest.State := StringOfChar('x', 20000);
    Raised := false;
    try
      Scorer.Score(BigRequest, Answers, InputTokens, OutputTokens);
    except
      on EDecisionUnsupported do Raised := true;
    end;
    AssertTrue('prompt over the context: EDecisionUnsupported', Raised);

    // The engine chats afterwards exactly as a fresh one.
    SetLength(Msgs, 1);
    Msgs[0] := ChatMessage('user', 'Hello there');
    Reply := Engine.ChatReply(Msgs, Engine.Opt);
    FreshReply := Fresh.ChatReply(Msgs, Fresh.Opt);
    AssertEquals('chat after scoring', FreshReply, Reply);

    // Without a chat template AnyJev's plain prompt is used.
    DeleteFile(Dir + 'tokenizer_config.json');
    FreeAndNil(RawScorer);
    RawScorer := TAnyJevL0Scorer.Create(Engine, 'anyjev-test');
    AssertEquals('plain prompt', AnyJevPlainPrompt(csAnyJevSystemPrompt, 'U'),
      RawScorer.RenderPrompt('U'));
  finally
    RawScorer.Free;
    Scorer.Free;
    Fresh.Free;
    Engine.Free;
    RemoveAnyJevModelDir(Dir);
  end;
end;

initialization
  TestFormat := DefaultFormatSettings;
  TestFormat.DecimalSeparator := '.';
  RegisterTest(TTestNeuralAnyJev);
end.
