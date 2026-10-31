unit TestNeuralDecision;
(*
Model-free tests for neuraldecision.pas: the /v1/systemone request parser and
its 422 details, Kev's value renderer, the special-token rewrite, the
confidence formulas, the response JSON and the restricted softmax.
Expected renderer and float strings come from Python 3 (Kev's api.render,
repr(float), json.dumps).
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, fpjson, jsonparser,
  neuralvolume, neuraldecision;

type
  TTestNeuralDecision = class(TTestCase)
  private
    function ParseOK(const Body: string): TDecisionRequest;
    // Parses Body expecting failure; returns the 422 JSON.
    function ParseErrors(const Body: string): string;
    procedure AssertHasError(const ErrorsJSON, LocJSON, ErrorType: string);
  published
    procedure TestParseValidRequest;
    procedure TestParseMissingTopLevelFields;
    procedure TestParseIgnoresModel;
    procedure TestParseQuestionsShape;
    procedure TestParseChoiceCriteriaRules;
    procedure TestParseScoreCriteriaRules;
    procedure TestParseNoulCriteria;
    procedure TestParseInstructionsKinds;
    procedure TestParseRejectsBoxAndAssets;
    // Option names and question ids over 255 bytes get a 422, never an alias.
    procedure TestParseLongKeys;
    procedure TestParseInvalidJSON;
    // " and \, which fpjson 3.2.2 mis-reads, plus non-ASCII escapes.
    procedure TestParseUnicodeEscapes;
    procedure TestErrorsToJSON;
    procedure TestRenderValue;
    procedure TestFloatToStr;
    procedure TestJSONString;
    procedure TestRewriteSpecialTokens;
    procedure TestConfidence;
    procedure TestAnswerFromProbabilities;
    procedure TestResponseJSON;
    procedure TestRestrictedSoftMax;
  end;

implementation

var
  TestFormat: TFormatSettings;

// Double literal without FPC's constant folding to single.
function ParseDouble(const S: string): double;
begin
  Result := StrToFloat(S, TestFormat);
end;

function TTestNeuralDecision.ParseOK(const Body: string): TDecisionRequest;
var
  Errors: TDecisionErrorDetails;
begin
  if not ParseDecisionRequest(Body, Result, Errors) then
    Fail('unexpected 422: ' + DecisionErrorsToJSON(Errors));
end;

function TTestNeuralDecision.ParseErrors(const Body: string): string;
var
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
begin
  AssertFalse('expected a 422 for ' + Body,
    ParseDecisionRequest(Body, Request, Errors));
  AssertTrue('error list not empty', Length(Errors) > 0);
  Result := DecisionErrorsToJSON(Errors);
end;

procedure TTestNeuralDecision.AssertHasError(const ErrorsJSON, LocJSON,
  ErrorType: string);
begin
  AssertTrue('expected loc ' + LocJSON + ' type ' + ErrorType + ' in ' + ErrorsJSON,
    Pos('{"loc":' + LocJSON + ',"msg":', ErrorsJSON) > 0);
  AssertTrue('expected type ' + ErrorType + ' in ' + ErrorsJSON,
    Pos('"type":"' + ErrorType + '"', ErrorsJSON) > 0);
end;

procedure TTestNeuralDecision.TestParseValidRequest;
var
  Request: TDecisionRequest;
begin
  Request := ParseOK('{"model":"jev-latest","state":{"user":"Ann","n":2},' +
    '"questions":{' +
    '"q_choice":{"type":"choice","instructions":"Pick one",' +
      '"criteria":{"zeta":"last letter","alpha":null,"mid":{"why":"x"}}},' +
    '"q_score":{"type":"score","criteria":["low",{"label":"high","w":1.5}]},' +
    '"q_noul":{"type":"noul","instructions":"Is it?"}}}');
  AssertEquals('state', 'user: Ann'#10'n: 2', Request.State);
  AssertEquals('question count', 3, Length(Request.Questions));
  AssertEquals('id 0', 'q_choice', Request.Questions[0].Id);
  AssertTrue('kind 0', Request.Questions[0].Kind = dqkChoice);
  AssertEquals('instructions', 'Pick one', Request.Questions[0].Instructions);
  AssertEquals('choice keeps key order', 'zeta', Request.Questions[0].OptionNames[0]);
  AssertEquals('alpha', 'alpha', Request.Questions[0].OptionNames[1]);
  AssertEquals('description', 'last letter',
    Request.Questions[0].OptionDescriptions[0]);
  AssertEquals('null description', '', Request.Questions[0].OptionDescriptions[1]);
  AssertEquals('object description', 'why: x',
    Request.Questions[0].OptionDescriptions[2]);
  AssertTrue('kind 1', Request.Questions[1].Kind = dqkScore);
  AssertEquals('score names', '1', Request.Questions[1].OptionNames[1]);
  AssertEquals('score description', 'label: high'#10'w: 1.5',
    Request.Questions[1].OptionDescriptions[1]);
  AssertEquals('legend keeps the original JSON', '{"label":"high","w":1.5}',
    Request.Questions[1].LegendJSON[1]);
  AssertEquals('legend string', '"low"', Request.Questions[1].LegendJSON[0]);
  AssertTrue('kind 2', Request.Questions[2].Kind = dqkNoul);
  AssertEquals('noul names', 'false', Request.Questions[2].OptionNames[0]);
  AssertEquals('noul names', 'true', Request.Questions[2].OptionNames[1]);
  AssertEquals('noul no description', '', Request.Questions[2].OptionDescriptions[1]);
end;

procedure TTestNeuralDecision.TestParseMissingTopLevelFields;
var
  ErrorsJSON: string;
begin
  ErrorsJSON := ParseErrors('{}');
  AssertHasError(ErrorsJSON, '["body","state"]', 'missing');
  AssertHasError(ErrorsJSON, '["body","questions"]', 'missing');
  AssertFalse('model is optional', Pos('"model"', ErrorsJSON) > 0);
  ParseOK('{"state":null,"questions":{"q":{"type":"noul"}}}');
end;

procedure TTestNeuralDecision.TestParseIgnoresModel;
const
  csStateAndQuestionsJSON = '"state":"s","questions":{"q":{"type":"noul"}}}';
begin
  ParseOK('{' + csStateAndQuestionsJSON);
  ParseOK('{"model":"anything",' + csStateAndQuestionsJSON);
  ParseOK('{"model":42,' + csStateAndQuestionsJSON);
end;

procedure TTestNeuralDecision.TestParseQuestionsShape;
var
  ErrorsJSON: string;
begin
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":{}}');
  AssertHasError(ErrorsJSON, '["body","questions"]', 'too_short');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":[]}');
  AssertHasError(ErrorsJSON, '["body","questions"]', 'dict_type');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":{"q":"x"}}');
  AssertHasError(ErrorsJSON, '["body","questions","q"]', 'model_attributes_type');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":{"q":{}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q"]', 'union_tag_not_found');
end;

procedure TTestNeuralDecision.TestParseChoiceCriteriaRules;
var
  ErrorsJSON, Options: string;
  OptionPos: integer;
  Request: TDecisionRequest;
begin
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"choice"}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria"]', 'missing');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"choice","criteria":["a"]}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria"]', 'dict_type');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"choice","criteria":{}}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria"]', 'value_error');
  Options := '';
  for OptionPos := 0 to 255 do
  begin
    if OptionPos > 0 then Options := Options + ',';
    Options := Options + '"o' + IntToStr(OptionPos) + '":null';
  end;
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"choice","criteria":{' + Options + '}}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria"]', 'value_error');
  // 255 options is the maximum
  Options := Copy(Options, 1, LastDelimiter(',', Options) - 1);
  Request := ParseOK('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"choice","criteria":{' + Options + '}}}}');
  AssertEquals('255 options', 255, Length(Request.Questions[0].OptionNames));
end;

procedure TTestNeuralDecision.TestParseScoreCriteriaRules;
var
  ErrorsJSON: string;
  Request: TDecisionRequest;
begin
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"score","criteria":{"a":1}}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria"]', 'list_type');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"score","criteria":[]}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria"]', 'too_short');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"score","criteria":[0,1,2,3,4,5,6,7,8,9,10]}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria"]', 'too_long');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"score","criteria":["a",null]}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria",1]', 'value_error');
  Request := ParseOK('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"score","criteria":[0,1,2,3,4,5,6,7,8,9]}}}');
  AssertEquals('10 levels', 10, Length(Request.Questions[0].OptionNames));
  AssertEquals('number level', '9', Request.Questions[0].OptionDescriptions[9]);
end;

procedure TTestNeuralDecision.TestParseNoulCriteria;
var
  Request: TDecisionRequest;
  ErrorsJSON: string;
begin
  Request := ParseOK('{"model":"m","state":"s","questions":{' +
    '"a":{"type":"noul","criteria":null},' +
    '"b":{"type":"noul","criteria":{"true":"it is"}},' +
    '"c":{"type":"noul","criteria":{"false":["x","y"],"true":"yes","other":1}}}}');
  AssertEquals('null criteria', '', Request.Questions[0].OptionDescriptions[1]);
  AssertEquals('false absent', '', Request.Questions[1].OptionDescriptions[0]);
  AssertEquals('true', 'it is', Request.Questions[1].OptionDescriptions[1]);
  AssertEquals('false list', '- x'#10'- y', Request.Questions[2].OptionDescriptions[0]);
  AssertEquals('true', 'yes', Request.Questions[2].OptionDescriptions[1]);
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"noul","criteria":"x"}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria"]', 'dict_type');
end;

procedure TTestNeuralDecision.TestParseInstructionsKinds;
var
  Request: TDecisionRequest;
begin
  Request := ParseOK('{"model":"m","state":["a","b"],"questions":{' +
    '"s":{"type":"noul","instructions":"text"},' +
    '"o":{"type":"noul","instructions":{"goal":"g"}},' +
    '"a":{"type":"noul","instructions":["one","two"]},' +
    '"n":{"type":"noul","instructions":null},' +
    '"x":{"type":"noul"},' +
    '"i":{"type":"noul","instructions":7}}}');
  AssertEquals('state list', '- a'#10'- b', Request.State);
  AssertEquals('string', 'text', Request.Questions[0].Instructions);
  AssertEquals('object', 'goal: g', Request.Questions[1].Instructions);
  AssertEquals('array', '- one'#10'- two', Request.Questions[2].Instructions);
  AssertEquals('null', '', Request.Questions[3].Instructions);
  AssertEquals('absent', '', Request.Questions[4].Instructions);
  AssertEquals('number', '7', Request.Questions[5].Instructions);
end;

procedure TTestNeuralDecision.TestParseRejectsBoxAndAssets;
var
  ErrorsJSON: string;
begin
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"box","criteria":{"a":"b"}}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q"]', 'union_tag_invalid');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","assets":[{"url":"x"}],' +
    '"questions":{"q":{"type":"noul"}}}');
  AssertHasError(ErrorsJSON, '["body","assets"]', 'value_error');
  ParseOK('{"model":"m","state":"s","assets":null,"questions":{"q":{"type":"noul"}}}');
  ParseOK('{"model":"m","state":"s","assets":[],"questions":{"q":{"type":"noul"}}}');
end;

procedure TTestNeuralDecision.TestParseLongKeys;
var
  Long255, Long300A, Long300B, ErrorsJSON: string;
  Request: TDecisionRequest;
  Errors: TDecisionErrorDetails;
  Root: TJSONData;
begin
  Long255 := StringOfChar('k', 255);
  Long300A := StringOfChar('x', 299) + 'A';
  Long300B := StringOfChar('x', 299) + 'B';
  // two names sharing a 255-byte prefix: a 422 per name, no EJSON duplicate
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":{"q":' +
    '{"type":"choice","criteria":{"' + Long300A + '":"a","' + Long300B + '":"b"}}}}');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria","' + Long300A + '"]',
    'value_error');
  AssertHasError(ErrorsJSON, '["body","questions","q","criteria","' + Long300B + '"]',
    'value_error');
  ErrorsJSON := ParseErrors('{"model":"m","state":"s","questions":{"' + Long300A +
    '":{"type":"noul"}}}');
  AssertHasError(ErrorsJSON, '["body","questions","' + Long300A + '"]', 'value_error');
  // exactly 255 bytes is safe through the string entry point
  Request := ParseOK('{"model":"m","state":{"' + Long300A + '":1},"questions":{"q":' +
    '{"type":"choice","criteria":{"' + Long255 + '":"d"}}}}');
  AssertEquals('255-byte name', Long255, Request.Questions[0].OptionNames[0]);
  AssertEquals('a long state key renders whole', Long300A + ': 1', Request.State);
  // a caller-parsed object cannot tell a 255-byte name from a truncated one
  Root := GetJSON('{"model":"m","state":"s","questions":{"q":' +
    '{"type":"choice","criteria":{"' + Long255 + '":"d"}}}}');
  try
    AssertFalse('nil shield list rejects 255 bytes',
      ParseDecisionRequest(Root, nil, Request, Errors));
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralDecision.TestParseInvalidJSON;
var
  ErrorsJSON: string;
begin
  ErrorsJSON := ParseErrors('{"model": ');
  AssertHasError(ErrorsJSON, '["body"]', 'json_invalid');
  ErrorsJSON := ParseErrors('');
  AssertHasError(ErrorsJSON, '["body"]', 'json_invalid');
  ErrorsJSON := ParseErrors('[1,2]');
  AssertHasError(ErrorsJSON, '["body"]', 'model_attributes_type');
  // fpjson rejects a repeated key (Python keeps the last one)
  ErrorsJSON := ParseErrors('{"model":"m","model":"n","state":"s",' +
    '"questions":{"q":{"type":"noul"}}}');
  AssertHasError(ErrorsJSON, '["body"]', 'json_invalid');
end;

procedure TTestNeuralDecision.TestParseUnicodeEscapes;
const
  csEscape = #92'u';
var
  Request: TDecisionRequest;
begin
  Request := ParseOK('{"model":"m","state":"say ' + csEscape + '0022hi' + csEscape +
    '0022 ' + csEscape + '005c ' + csEscape + '00e9 ' + csEscape + 'd83d' + csEscape +
    'de00 '#92#92'u0041 '#92'"q'#92'"","questions":{"q":{"type":"choice",' +
    '"criteria":{"caf' + csEscape + '00e9":"d' + csEscape + '00e9j' + csEscape +
    '00e0","' + csEscape + '0022quoted' + csEscape + '0022":null}}}}');
  AssertEquals('state', 'say "hi" '#92' '#$C3#$A9' '#$F0#$9F#$98#$80' '#92'u0041 "q"',
    Request.State);
  AssertEquals('escaped name', 'caf'#$C3#$A9, Request.Questions[0].OptionNames[0]);
  AssertEquals('escaped description', 'd'#$C3#$A9'j'#$C3#$A0,
    Request.Questions[0].OptionDescriptions[0]);
  AssertEquals('quote in a name', '"quoted"', Request.Questions[0].OptionNames[1]);
  // raw UTF-8 passes through unchanged
  Request := ParseOK('{"model":"m","state":"'#$C3#$A9'","questions":{"'#$C3#$A9 +
    '":{"type":"noul"}}}');
  AssertEquals('raw state', #$C3#$A9, Request.State);
  AssertEquals('raw id', #$C3#$A9, Request.Questions[0].Id);
end;

procedure TTestNeuralDecision.TestErrorsToJSON;
var
  Errors: TDecisionErrorDetails;
  Request: TDecisionRequest;
begin
  AssertFalse(ParseDecisionRequest('{"model":"m","state":"s","questions":' +
    '{"q":{"type":"score","criteria":[null]}}}', Request, Errors));
  AssertEquals('{"detail":[{"loc":["body","questions","q","criteria",0],' +
    '"msg":"Value error, a score level must not be null","type":"value_error"}]}',
    DecisionErrorsToJSON(Errors));
  AssertEquals('{"detail":"no model loaded"}',
    DecisionErrorMessageToJSON('no model loaded'));
end;

procedure TTestNeuralDecision.TestRenderValue;
var
  Value: TJSONData;
begin
  Value := GetJSON('{"name":"Ann","age":42,"ok":true,"bad":false,"none":null,' +
    '"ratio":1.5,"three":3.0,"tags":["a"," b",{"k":"v","n":[1,2]},[7,8]],' +
    '"addr":{"city":"X","geo":{"lat":-0.25}},"empty":{},"el":[]}');
  try
    // Python: render(json.loads(...)) with Kev's api.render
    AssertEquals('name: Ann'#10'age: 42'#10'ok: True'#10'bad: False'#10'none: '#10 +
      'ratio: 1.5'#10'three: 3.0'#10'tags:'#10'  - a'#10'  - b'#10'  - k: v'#10 +
      '    n:'#10'      - 1'#10'      - 2'#10'  - - 7'#10'    - 8'#10'addr:'#10 +
      '  city: X'#10'  geo:'#10'    lat: -0.25'#10'empty:'#10#10'el:'#10,
      RenderDecisionValue(Value));
  finally
    Value.Free;
  end;
  AssertEquals('nil', '', RenderDecisionValue(nil));
  Value := GetJSON('true');
  try
    AssertEquals('bool', 'True', RenderDecisionValue(Value));
  finally
    Value.Free;
  end;
end;

procedure TTestNeuralDecision.TestFloatToStr;
const
  // Python repr() of the same doubles (subnormals excluded: FloatToStrF keeps 2 digits)
  csInputs: array[0..16] of string = ('1.5', '3.0', '0.1', '0.30000000000000004',
    '0.3333333333333333', '1e16', '1e15', '123456789012345678', '1e-4', '1e-5',
    '0.00012345', '1.7976931348623157e308', '-2.5', '100', '1e22',
    '2.5e-7', '0.4115');
  csExpected: array[0..16] of string = ('1.5', '3.0', '0.1', '0.30000000000000004',
    '0.3333333333333333', '1e+16', '1000000000000000.0', '1.2345678901234568e+17',
    '0.0001', '1e-05', '0.00012345', '1.7976931348623157e+308', '-2.5',
    '100.0', '1e+22', '2.5e-07', '0.4115');
var
  InputPos: integer;
begin
  for InputPos := 0 to High(csInputs) do
    AssertEquals(csInputs[InputPos], csExpected[InputPos],
      DecisionFloatToStr(ParseDouble(csInputs[InputPos])));
  AssertEquals('zero', '0.0', DecisionFloatToStr(0));
  AssertEquals('negative zero', '-0.0', DecisionFloatToStr(-ParseDouble('0')));
end;

procedure TTestNeuralDecision.TestJSONString;
var
  Value: TJSONData;
begin
  // Python: json.dumps(..., ensure_ascii=False)
  AssertEquals('"x\"y\\z\n\t'#92'u0001'#$C3#$A9'/"',
    DecisionJSONString('x"y\z'#10#9#1#$C3#$A9'/'));
  Value := GetJSON('{"a":[1,2.5,true,null,{"b":"c"}],"d":-3}');
  try
    AssertEquals('{"a":[1,2.5,true,null,{"b":"c"}],"d":-3}',
      DecisionValueToJSON(Value));
  finally
    Value.Free;
  end;
end;

procedure TTestNeuralDecision.TestRewriteSpecialTokens;
const
  csBar = #$C2#$A6;
begin
  AssertEquals('plain', 'no specials', RewriteSpecialTokenStrings('no specials'));
  AssertEquals('one', 'a <' + csBar + 'im_end' + csBar + '> b',
    RewriteSpecialTokenStrings('a <|im_end|> b'));
  AssertEquals('two', '<' + csBar + 'fim_prefix' + csBar + '><' + csBar + 'x9' +
    csBar + '>', RewriteSpecialTokenStrings('<|fim_prefix|><|x9|>'));
  AssertEquals('empty name', '<||>', RewriteSpecialTokenStrings('<||>'));
  AssertEquals('other chars', '<|a-b|>', RewriteSpecialTokenStrings('<|a-b|>'));
  AssertEquals('unterminated', '<|abc|', RewriteSpecialTokenStrings('<|abc|'));
  AssertEquals('nested start', '<|<' + csBar + 'a' + csBar + '>',
    RewriteSpecialTokenStrings('<|<|a|>'));
end;

procedure TTestNeuralDecision.TestConfidence;
var
  P: TDecisionProbabilities;
begin
  P := [0, ParseDouble('0.57'), ParseDouble('0.43')];
  AssertEquals('score (0,0.57,0.43)', 0.355, ScoreConfidence(P), 1e-3);
  P := [0, ParseDouble('0.95'), ParseDouble('0.05')];
  AssertEquals('score (0,0.95,0.05)', 0.925, ScoreConfidence(P), 1e-3);
  P := [ParseDouble('0.25'), ParseDouble('0.25'), ParseDouble('0.25'), ParseDouble('0.25')];
  AssertEquals('score uniform', 0, ScoreConfidence(P), 1e-12);
  P := [ParseDouble('0.5'), 0, ParseDouble('0.5')];
  AssertEquals('score clamps at 0', 0, ScoreConfidence(P), 1e-12);
  P := [ParseDouble('0.7')];
  AssertEquals('score n=1', 1, ScoreConfidence(P), 1e-12);
  AssertEquals('choice n=1', 1, ChoiceConfidence(P), 1e-12);
  P := [0, 0, 0];
  AssertEquals('all zero -> uniform', 0, ChoiceConfidence(P), 1e-12);
  AssertEquals('all zero -> uniform', 0, ScoreConfidence(P), 1e-12);
  P := [ParseDouble('0.8'), ParseDouble('0.2')];
  AssertEquals('choice', 0.6, ChoiceConfidence(P), 1e-12);
  // unnormalised input is normalised first
  P := [ParseDouble('1.6'), ParseDouble('0.4')];
  AssertEquals('choice normalises', 0.6, ChoiceConfidence(P), 1e-12);
  P := [1, 0, 0, 0];
  AssertEquals('choice certain', 1, ChoiceConfidence(P), 1e-12);
  AssertEquals('round', 0.4115, RoundDecisionProbability(ParseDouble('0.41149')), 1e-12);
end;

procedure TTestNeuralDecision.TestAnswerFromProbabilities;
var
  Answer: TDecisionAnswer;
begin
  Answer := DecisionAnswerFromProbabilities(dqkChoice, [ParseDouble('0.2'), ParseDouble('0.5'), ParseDouble('0.5')]);
  AssertEquals('first argmax', 1, Answer.ChoiceIndex);
  Answer := DecisionAnswerFromProbabilities(dqkScore, [ParseDouble('0.1'), ParseDouble('0.2'), ParseDouble('0.7')]);
  AssertEquals('expected level', 1.6, Answer.Score, 1e-12);
  AssertEquals('score confidence', ScoreConfidence(Answer.Probabilities),
    Answer.Confidence, 1e-12);
  Answer := DecisionAnswerFromProbabilities(dqkNoul, [ParseDouble('0.3'), ParseDouble('0.7')]);
  AssertEquals('noul = p(true)', 0.7, Answer.Noul, 1e-12);
end;

procedure TTestNeuralDecision.TestResponseJSON;
var
  Request: TDecisionRequest;
  Answers: TDecisionAnswers;
  Response: string;
  Root: TJSONData;
  Answer: TJSONObject;
begin
  Request := ParseOK('{"model":"jev-latest","state":"s","questions":{' +
    '"c":{"type":"choice","criteria":{"yes":null,"no":"x"}},' +
    '"s":{"type":"score","criteria":["low",{"label":"high"}]},' +
    '"n":{"type":"noul"}}}');
  SetLength(Answers, 3);
  Answers[0] := DecisionAnswerFromProbabilities(dqkChoice, [ParseDouble('0.25'), ParseDouble('0.75')]);
  Answers[1] := DecisionAnswerFromProbabilities(dqkScore, [ParseDouble('0.4'), ParseDouble('0.6')]);
  Answers[2] := DecisionAnswerFromProbabilities(dqkNoul, [ParseDouble('0.123456'), ParseDouble('0.876544')]);
  Response := DecisionResponseToJSON(Request, 'q25', Answers, 17, 9);
  AssertEquals('{"model":"q25","answers":{' +
    '"c":{"type":"choice","choice":"no","probabilities":{"yes":0.25,"no":0.75},' +
      '"confidence":0.5},' +
    '"s":{"type":"score","score":0.6,"legend":{"0":"low","1":{"label":"high"}},' +
      '"probabilities":{"0":0.4,"1":0.6},"confidence":0.2},' +
    '"n":{"type":"noul","noul":0.8765}},' +
    '"usage":{"input_tokens":17,"output_tokens":9}}', Response);
  Root := GetJSON(Response);
  try
    Answer := TJSONObject(Root).Objects['answers'].Objects['s'];
    AssertEquals('round trip legend', 'high',
      Answer.Objects['legend'].Objects['1'].Strings['label']);
    AssertEquals('round trip probability', 0.6,
      Answer.Objects['probabilities'].Floats['1'], 1e-12);
    AssertEquals('round trip usage', 17,
      TJSONObject(Root).Objects['usage'].Integers['input_tokens']);
  finally
    Root.Free;
  end;
end;

procedure TTestNeuralDecision.TestRestrictedSoftMax;
var
  Logits, Probs: TNNetVolume;
  Raised: boolean;
  ExpectedSum: double;
begin
  Logits := TNNetVolume.Create(8, 1, 1);
  Probs := TNNetVolume.Create();
  try
    Logits.FData[1] := 1000;
    Logits.FData[3] := 999;
    Logits.FData[6] := 998;
    RestrictedSoftMax(Logits, [3, 1, 6], Probs);
    AssertEquals('size', 3, Probs.Size);
    ExpectedSum := 1 + Exp(1) + Exp(-1);
    AssertEquals('id 3', 1 / ExpectedSum, Probs.FData[0], 1e-5);
    AssertEquals('id 1', Exp(1) / ExpectedSum, Probs.FData[1], 1e-5);
    AssertEquals('id 6', Exp(-1) / ExpectedSum, Probs.FData[2], 1e-5);
    RestrictedSoftMax(Logits, [0], Probs);
    AssertEquals('one id', 1, Probs.FData[0], 1e-7);
    AssertEquals('to probabilities', 1, Length(DecisionProbabilitiesFromVolume(Probs)));
    Raised := false;
    try
      RestrictedSoftMax(Logits, [8], Probs);
    except
      on E: Exception do Raised := true;
    end;
    AssertTrue('out-of-range id raises', Raised);
  finally
    Probs.Free;
    Logits.Free;
  end;
end;

initialization
  TestFormat := DefaultFormatSettings;
  TestFormat.DecimalSeparator := '.';
  RegisterTest(TTestNeuralDecision);
end.
