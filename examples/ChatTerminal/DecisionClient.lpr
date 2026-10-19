program DecisionClient;
(*
DecisionClient: a command-line client for ChatServer's POST /v1/systemone
(TypeSafe/Jev typed decisions: a state plus typed questions in, option
probabilities out). It uses only FCL units (fphttpclient, fpjson), none of
the neural-api units, to show that any HTTP client can call the endpoint.

  DecisionClient                                  the built-in demo request
  DecisionClient --request decision_examples/ticket_routing.json
  DecisionClient --server http://127.0.0.1:8094 --retries 30 --raw

Without --request the client sends a built-in support-ticket request (an
object state, a choice, a score and a noul question). --model sets the
request's "model" field (default "jev-latest", the alias every ChatServer
decision scorer accepts); without --model a request file keeps its own.

Exit status: 0 on HTTP 200, 1 on any other HTTP status (the server's
"detail" is printed), 2 when the server cannot be reached, 3 on a bad
command line or request file.

--selftest runs the offline request-building and response-printing checks
(no server needed) and exits.

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

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, fpjson, jsonparser, fphttpclient, opensslsockets, ssockets;

const
  csDefaultServerURL = 'http://127.0.0.1:8080';
  csDefaultDecisionModel = 'jev-latest';
  csSystemOnePath = '/v1/systemone';
  csRetryDelayMs = 1000;

  ceExitOK = 0;
  ceExitHTTPError = 1;
  ceExitUnreachable = 2;
  ceExitUsage = 3;

type
  TClientOptions = record
    ServerURL: string;
    ModelName: string;        // '' = keep the request file's model (demo: jev-latest)
    RequestFile: string;      // '' = the built-in demo request
    APIKey: string;           // '' = no Authorization header
    RetryCount: integer;      // extra connection attempts, csRetryDelayMs apart
    PrintRaw: boolean;
    RunSelfTest: boolean;
    ShowHelp: boolean;
  end;

procedure PrintUsage();
begin
  WriteLn('Usage: DecisionClient [options]');
  WriteLn;
  WriteLn('Sends one TypeSafe/Jev typed decision request to POST /v1/systemone and');
  WriteLn('prints each answer, its probabilities in the server''s order, its');
  WriteLn('confidence and the token usage.');
  WriteLn;
  WriteLn('Options:');
  WriteLn('  --server URL     server root (default ', csDefaultServerURL, ')');
  WriteLn('  --model NAME     the request''s "model" (default ', csDefaultDecisionModel,
    '; a request');
  WriteLn('                   file keeps its own model unless --model is given)');
  WriteLn('  --request FILE   send the JSON request in FILE instead of the built-in');
  WriteLn('                   support-ticket demo');
  WriteLn('  --api-key KEY    send "Authorization: Bearer KEY" (ChatServer ignores it;');
  WriteLn('                   other TypeSafe-compatible servers may need it)');
  WriteLn('  --retries N      retry a refused connection N times, 1 s apart (default 0)');
  WriteLn('  --raw            print the raw JSON response body');
  WriteLn('  --selftest       run the offline checks and exit');
  WriteLn('  --help           this text');
  WriteLn;
  WriteLn('Exit status: 0 = HTTP 200, 1 = other HTTP status, 2 = server unreachable,');
  WriteLn('3 = bad command line or request file.');
end;

function ParseClientArgs(Args: TStrings; out Opt: TClientOptions;
  out ErrorMsg: string): boolean;
var
  ArgPos, MaxArgPos: integer;
  Arg: string;

  function NextValue(out Value: string): boolean;
  begin
    Result := ArgPos < MaxArgPos;
    if Result then
    begin
      Inc(ArgPos);
      Value := Args[ArgPos];
    end
    else
      ErrorMsg := Arg + ' needs a value';
  end;

var
  Value: string;
begin
  Opt := Default(TClientOptions);
  Opt.ServerURL := csDefaultServerURL;
  ErrorMsg := '';
  MaxArgPos := Args.Count - 1;
  ArgPos := 0;
  while ArgPos <= MaxArgPos do
  begin
    Arg := Args[ArgPos];
    if Arg = '--server' then
    begin
      if not NextValue(Opt.ServerURL) then Exit(false);
    end
    else if Arg = '--model' then
    begin
      if not NextValue(Opt.ModelName) then Exit(false);
      if Opt.ModelName = '' then
      begin
        ErrorMsg := '--model must not be empty';
        Exit(false);
      end;
    end
    else if Arg = '--request' then
    begin
      if not NextValue(Opt.RequestFile) then Exit(false);
    end
    else if Arg = '--api-key' then
    begin
      if not NextValue(Opt.APIKey) then Exit(false);
    end
    else if Arg = '--retries' then
    begin
      if not NextValue(Value) then Exit(false);
      if not TryStrToInt(Value, Opt.RetryCount) or (Opt.RetryCount < 0) then
      begin
        ErrorMsg := '--retries needs a whole number >= 0, not "' + Value + '"';
        Exit(false);
      end;
    end
    else if Arg = '--raw' then Opt.PrintRaw := true
    else if Arg = '--selftest' then Opt.RunSelfTest := true
    else if (Arg = '--help') or (Arg = '-h') then Opt.ShowHelp := true
    else
    begin
      ErrorMsg := 'unknown option "' + Arg + '"';
      Exit(false);
    end;
    Inc(ArgPos);
  end;
  Result := true;
end;

// The endpoint URL for a server root; a root that already ends in /v1 is accepted too.
function SystemOneURL(const ServerURL: string): string;
var
  RootURL: string;
begin
  RootURL := ServerURL;
  while (RootURL <> '') and (RootURL[Length(RootURL)] = '/') do
    Delete(RootURL, Length(RootURL), 1);
  if LowerCase(Copy(RootURL, Length(RootURL) - 2, 3)) = '/v1' then
    Delete(RootURL, Length(RootURL) - 2, 3);
  Result := RootURL + csSystemOnePath;
end;

// Parses JSON text; nil when it is not valid JSON.
function ParseJSONOrNil(const Text: string): TJSONData;
begin
  try
    Result := GetJSON(Text);
  except
    on E: EJSON do Result := nil;
    on E: EParserError do Result := nil;
  end;
end;

function BuildDemoRequest(const ModelName: string): string;
var
  Root, State, Questions, Team, TeamCriteria, Frustration, Urgent,
    UrgentCriteria: TJSONObject;
begin
  State := TJSONObject.Create([
    'ticket', 'I was charged twice for order 4471 and the refund button in the' +
      ' app just spins. This is the third time I am writing!',
    'customer_tier', 'premium',
    'open_tickets', 2]);
  TeamCriteria := TJSONObject.Create([
    'billing', 'charges, refunds, invoices and payment methods',
    'technical', 'app crashes, bugs, login and account problems',
    'shipping', 'delivery status, lost or damaged parcels']);
  TeamCriteria.Add('other', TJSONNull.Create);
  Team := TJSONObject.Create([
    'type', 'choice',
    'instructions', 'Which team should handle this ticket?',
    'criteria', TeamCriteria]);
  Frustration := TJSONObject.Create([
    'type', 'score',
    'instructions', 'How frustrated is the customer?',
    'criteria', TJSONArray.Create(['calm', 'annoyed', 'furious'])]);
  UrgentCriteria := TJSONObject.Create([
    'true', 'money is at stake or the customer is blocked',
    'false', 'the request can wait for the normal queue']);
  Urgent := TJSONObject.Create([
    'type', 'noul',
    'instructions', 'Is this ticket urgent?',
    'criteria', UrgentCriteria]);
  Questions := TJSONObject.Create([
    'team', Team,
    'frustration', Frustration,
    'urgent', Urgent]);
  Root := TJSONObject.Create([
    'model', ModelName,
    'state', State,
    'questions', Questions]);
  try
    Result := Root.AsJSON;
  finally
    Root.Free;
  end;
end;

// The body to send for request file text: unchanged unless the "model" field
// must be set (ModelName <> '') or added (the file has none).
function PrepareRequestBody(const RequestText, ModelName: string;
  out Body, ErrorMsg: string): boolean;
var
  Root: TJSONData;
  Request: TJSONObject;
  ModelField: TJSONData;
begin
  Body := '';
  ErrorMsg := '';
  Root := ParseJSONOrNil(RequestText);
  try
    if not (Root is TJSONObject) then
    begin
      ErrorMsg := 'the request must be a JSON object';
      Exit(false);
    end;
    Request := TJSONObject(Root);
    ModelField := Request.Find('model');
    if (ModelName = '') and (ModelField <> nil) then
      Body := RequestText
    else
    begin
      if ModelField <> nil then Request.Delete('model');
      if ModelName = '' then Request.Add('model', csDefaultDecisionModel)
      else Request.Add('model', ModelName);
      Body := Request.AsJSON;
    end;
    Result := true;
  finally
    Root.Free;
  end;
end;

function JSONValueText(Value: TJSONData): string;
begin
  if Value = nil then Result := '(missing)'
  else if Value.JSONType = jtString then Result := Value.AsString
  else if Value.JSONType = jtNumber then Result := FormatFloat('0.####', Value.AsFloat)
  else Result := Value.AsJSON;
end;

function ProbabilityText(Value: TJSONData): string;
begin
  if (Value <> nil) and (Value.JSONType = jtNumber) then
    Result := FormatFloat('0.0000', Value.AsFloat)
  else
    Result := JSONValueText(Value);
end;

function MaxNameLength(Probabilities: TJSONObject): integer;
var
  OptionPos, MaxOptionPos: integer;
begin
  Result := 0;
  MaxOptionPos := Probabilities.Count - 1;
  for OptionPos := 0 to MaxOptionPos do
    if Length(Probabilities.Names[OptionPos]) > Result then
      Result := Length(Probabilities.Names[OptionPos]);
end;

// One line per option: name, legend text (score) and probability, in the server's order.
procedure AddProbabilityLines(Answer: TJSONObject; const ChosenName: string;
  Lines: TStrings);
var
  Probabilities, Legend: TJSONObject;
  OptionPos, MaxOptionPos, NameWidth: integer;
  OptionName, LegendText, Marker: string;
begin
  if not (Answer.Find('probabilities') is TJSONObject) then Exit;
  Probabilities := TJSONObject(Answer.Find('probabilities'));
  if Answer.Find('legend') is TJSONObject then
    Legend := TJSONObject(Answer.Find('legend'))
  else
    Legend := nil;
  NameWidth := MaxNameLength(Probabilities);
  MaxOptionPos := Probabilities.Count - 1;
  for OptionPos := 0 to MaxOptionPos do
  begin
    OptionName := Probabilities.Names[OptionPos];
    if OptionName = ChosenName then Marker := '* ' else Marker := '  ';
    if Legend <> nil then LegendText := '  ' + JSONValueText(Legend.Find(OptionName))
    else LegendText := '';
    Lines.Add('    ' + Marker + Format('%-*s  %s%s', [NameWidth, OptionName,
      ProbabilityText(Probabilities.Items[OptionPos]), LegendText]));
  end;
end;

// The number of levels of a score answer: its legend's size.
function LevelCount(Answer: TJSONObject): integer;
begin
  if Answer.Find('legend') is TJSONObject then
    Result := Answer.Find('legend').Count
  else
    Result := 0;
end;

// Human-readable lines for a 200 /v1/systemone body; false when the body has no answers object.
function FormatDecisionAnswers(const Body: string; Lines: TStrings): boolean;
var
  Root: TJSONData;
  Response, Answers, Answer, Usage: TJSONObject;
  QuestionPos, MaxQuestionPos: integer;
  QuestionId, AnswerKind, ChosenName: string;
begin
  Root := ParseJSONOrNil(Body);
  try
    Result := (Root is TJSONObject) and
      (TJSONObject(Root).Find('answers') is TJSONObject);
    if not Result then Exit;
    Response := TJSONObject(Root);
    Lines.Add('model: ' + JSONValueText(Response.Find('model')));
    Answers := TJSONObject(Response.Find('answers'));
    MaxQuestionPos := Answers.Count - 1;
    for QuestionPos := 0 to MaxQuestionPos do
    begin
      QuestionId := Answers.Names[QuestionPos];
      if not (Answers.Items[QuestionPos] is TJSONObject) then
      begin
        Lines.Add(QuestionId + ': ' + Answers.Items[QuestionPos].AsJSON);
        continue;
      end;
      Answer := TJSONObject(Answers.Items[QuestionPos]);
      AnswerKind := JSONValueText(Answer.Find('type'));
      ChosenName := '';
      if AnswerKind = 'choice' then
      begin
        ChosenName := JSONValueText(Answer.Find('choice'));
        Lines.Add(Format('%s (choice): %s   confidence %s', [QuestionId,
          ChosenName, ProbabilityText(Answer.Find('confidence'))]));
      end
      else if AnswerKind = 'score' then
        Lines.Add(Format('%s (score): %s on levels 0..%d   confidence %s',
          [QuestionId, ProbabilityText(Answer.Find('score')),
          LevelCount(Answer) - 1,
          ProbabilityText(Answer.Find('confidence'))]))
      else if AnswerKind = 'noul' then
        Lines.Add(Format('%s (noul): P(true) = %s', [QuestionId,
          ProbabilityText(Answer.Find('noul'))]))
      else
        Lines.Add(QuestionId + ': ' + Answer.AsJSON);
      AddProbabilityLines(Answer, ChosenName, Lines);
    end;
    if Response.Find('usage') is TJSONObject then
    begin
      Usage := TJSONObject(Response.Find('usage'));
      Lines.Add(Format('usage: %s input tokens, %s output tokens',
        [JSONValueText(Usage.Find('input_tokens')),
        JSONValueText(Usage.Find('output_tokens'))]));
    end;
  finally
    Root.Free;
  end;
end;

// "body.questions.team.criteria" from a FastAPI loc array.
function LocPathText(Loc: TJSONData): string;
var
  PartPos, MaxPartPos: integer;
begin
  if not (Loc is TJSONArray) then Exit(JSONValueText(Loc));
  Result := '';
  MaxPartPos := Loc.Count - 1;
  for PartPos := 0 to MaxPartPos do
  begin
    if PartPos > 0 then Result := Result + '.';
    Result := Result + JSONValueText(Loc.Items[PartPos]);
  end;
end;

// Lines for a non-200 body: the {"detail": string | list} shape, the OpenAI
// {"error": {"message"}} shape of the chat routes, else the raw body.
procedure FormatErrorBody(StatusCode: integer; const Body: string; Lines: TStrings);
var
  Root, Detail, ErrorField: TJSONData;
  Item: TJSONData;
  ItemPos, MaxItemPos: integer;
begin
  Lines.Add(Format('HTTP %d', [StatusCode]));
  Root := ParseJSONOrNil(Body);
  try
    if Root is TJSONObject then
    begin
      Detail := TJSONObject(Root).Find('detail');
      ErrorField := TJSONObject(Root).Find('error');
    end
    else
    begin
      Detail := nil;
      ErrorField := nil;
    end;
    if Detail is TJSONArray then
    begin
      MaxItemPos := Detail.Count - 1;
      for ItemPos := 0 to MaxItemPos do
      begin
        Item := Detail.Items[ItemPos];
        if Item is TJSONObject then
          Lines.Add(Format('  %s: %s (%s)', [
            LocPathText(TJSONObject(Item).Find('loc')),
            JSONValueText(TJSONObject(Item).Find('msg')),
            JSONValueText(TJSONObject(Item).Find('type'))]))
        else
          Lines.Add('  ' + JSONValueText(Item));
      end;
    end
    else if Detail <> nil then
      Lines.Add('  ' + JSONValueText(Detail))
    else if (ErrorField is TJSONObject) and
      (TJSONObject(ErrorField).Find('message') <> nil) then
      Lines.Add('  ' + JSONValueText(TJSONObject(ErrorField).Find('message')))
    else
      Lines.Add('  ' + Body);
  finally
    Root.Free;
  end;
end;

// The lines to print for one HTTP reply and the exit status it maps to.
function ReplyToLines(StatusCode: integer; const Body: string; PrintRaw: boolean;
  Lines: TStrings): integer;
begin
  if StatusCode = 200 then
  begin
    if PrintRaw then
    begin
      Lines.Add(Body);
      Exit(ceExitOK);
    end;
    if FormatDecisionAnswers(Body, Lines) then Exit(ceExitOK);
    Lines.Add('HTTP 200 without an "answers" object:');
    Lines.Add('  ' + Body);
    Exit(ceExitHTTPError);
  end;
  if PrintRaw then
  begin
    Lines.Add(Format('HTTP %d', [StatusCode]));
    Lines.Add(Body);
  end
  else
    FormatErrorBody(StatusCode, Body, Lines);
  Result := ceExitHTTPError;
end;

// POSTs Body; false (with ErrorMsg) when no connection could be made in 1 + RetryCount attempts.
function PostDecisionRequest(const URL, Body, APIKey: string; RetryCount: integer;
  out StatusCode: integer; out ResponseBody, ErrorMsg: string): boolean;
var
  Client: TFPHTTPClient;
  RequestStream, ResponseStream: TStringStream;
  AttemptPos: integer;
begin
  StatusCode := 0;
  ResponseBody := '';
  ErrorMsg := '';
  for AttemptPos := 0 to RetryCount do
  begin
    if AttemptPos > 0 then Sleep(csRetryDelayMs);
    Client := TFPHTTPClient.Create(nil);
    RequestStream := TStringStream.Create(Body);
    ResponseStream := TStringStream.Create('');
    try
      Client.AddHeader('Content-Type', 'application/json');
      Client.AddHeader('Accept', 'application/json');
      if APIKey <> '' then Client.AddHeader('Authorization', 'Bearer ' + APIKey);
      Client.RequestBody := RequestStream;
      try
        Client.HTTPMethod('POST', URL, ResponseStream, []);
        StatusCode := Client.ResponseStatusCode;
        ResponseBody := ResponseStream.DataString;
        Exit(true);
      except
        on E: ESocketError do ErrorMsg := E.Message;
        on E: Exception do
        begin
          ErrorMsg := E.ClassName + ': ' + E.Message;
          Exit(false);
        end;
      end;
    finally
      Client.RequestBody := nil;
      ResponseStream.Free;
      RequestStream.Free;
      Client.Free;
    end;
  end;
  Result := false;
end;

// ---------------------------------------------------------------------------
// --selftest: offline checks (no server needed).
// ---------------------------------------------------------------------------
procedure SelfTest();
const
  csChoiceScoreNoulReply =
    '{"model":"jev-latest","answers":{' +
    '"team":{"type":"choice","choice":"billing","probabilities":' +
      '{"technical":0.15,"billing":0.8,"other":0.05},"confidence":0.7},' +
    '"frustration":{"type":"score","score":1.3,"legend":' +
      '{"0":"calm","1":"annoyed","2":"furious"},' +
      '"probabilities":{"0":0.1,"1":0.5,"2":0.4},"confidence":0.25},' +
    '"urgent":{"type":"noul","noul":0.73}},' +
    '"usage":{"input_tokens":2140,"output_tokens":0}}';
  csValidationReply =
    '{"detail":[{"loc":["body","questions","team","criteria"],' +
    '"msg":"Field required","type":"missing"},' +
    '{"loc":["body","questions","level","criteria",1],' +
    '"msg":"Value error, a score level must not be null","type":"value_error"}]}';
  csDecisionOnlyReply =
    '{"detail":"no decision scorer is active on this server, so POST' +
    ' /v1/systemone cannot answer"}';
var
  Failures: integer;
  Lines, Args: TStringList;
  Root: TJSONData;
  Request, Questions: TJSONObject;
  Body, ErrorMsg: string;
  ExitStatus: integer;
  Opt: TClientOptions;

  procedure Check(Condition: boolean; const What: string);
  begin
    if Condition then WriteLn('PASS: ', What)
    else
    begin
      WriteLn('FAIL: ', What);
      Inc(Failures);
    end;
  end;

  function HasLine(const Expected: string): boolean;
  begin
    Result := Lines.IndexOf(Expected) >= 0;
  end;

  function QuestionType(const Id: string): string;
  begin
    Result := '';
    if Questions.Find(Id) is TJSONObject then
      Result := TJSONObject(Questions.Find(Id)).Get('type', '');
  end;

begin
  Failures := 0;
  Lines := TStringList.Create;
  Args := TStringList.Create;
  try
    Root := ParseJSONOrNil(BuildDemoRequest(csDefaultDecisionModel));
    try
      Check(Root is TJSONObject, 'the built-in demo request is a JSON object');
      if Root is TJSONObject then
      begin
        Request := TJSONObject(Root);
        Check(Request.Get('model', '') = csDefaultDecisionModel,
          'the demo request names the jev-latest model');
        Check(Request.Find('state') is TJSONObject,
          'the demo state is an object');
        Questions := TJSONObject(Request.Find('questions'));
        Check((Questions is TJSONObject) and (Questions.Count = 3) and
          (QuestionType('team') = 'choice') and
          (QuestionType('frustration') = 'score') and
          (QuestionType('urgent') = 'noul'),
          'the demo asks one choice, one score and one noul question');
        Check((Questions.FindPath('team.criteria') is TJSONObject) and
          (Questions.FindPath('team.criteria').Count = 4) and
          (Questions.FindPath('team.criteria.other').JSONType = jtNull),
          'the demo choice has 4 options, one with a null description');
        Check((Questions.FindPath('frustration.criteria') is TJSONArray) and
          (Questions.FindPath('frustration.criteria').Count = 3),
          'the demo score has 3 levels');
        Check(Questions.FindPath('urgent.criteria') is TJSONObject,
          'the demo noul has true/false criteria');
      end;
    finally
      Root.Free;
    end;

    Check(PrepareRequestBody('{"model":"kev-0.8b","state":"s","questions":{}}',
      '', Body, ErrorMsg) and (Body = '{"model":"kev-0.8b","state":"s","questions":{}}'),
      'a request file with a model is sent unchanged without --model');
    Root := nil;
    if PrepareRequestBody('{"model":"kev-0.8b","state":"s"}', 'other', Body,
      ErrorMsg) then Root := ParseJSONOrNil(Body);
    Check((Root is TJSONObject) and (TJSONObject(Root).Get('model', '') = 'other'),
      '--model replaces the request file''s model');
    Root.Free;
    Root := nil;
    if PrepareRequestBody('{"state":"s"}', '', Body, ErrorMsg) then
      Root := ParseJSONOrNil(Body);
    Check((Root is TJSONObject) and
      (TJSONObject(Root).Get('model', '') = csDefaultDecisionModel),
      'a request file without a model gets jev-latest');
    Root.Free;
    Check(not PrepareRequestBody('[1,2]', '', Body, ErrorMsg) and
      not PrepareRequestBody('{not json', '', Body, ErrorMsg),
      'a request file that is not a JSON object is refused');

    Check((SystemOneURL('http://127.0.0.1:8080') = 'http://127.0.0.1:8080/v1/systemone') and
      (SystemOneURL('http://h:1/') = 'http://h:1/v1/systemone') and
      (SystemOneURL('http://h:1/v1/') = 'http://h:1/v1/systemone'),
      'the endpoint URL from a server root, with or without /v1');

    Lines.Clear;
    ExitStatus := ReplyToLines(200, csChoiceScoreNoulReply, false, Lines);
    Check(ExitStatus = ceExitOK, 'a 200 reply exits with 0');
    Check(HasLine('team (choice): billing   confidence 0.7000') and
      (Lines.IndexOf('      technical  0.1500') >= 0) and
      (Lines.IndexOf('    * billing    0.8000') >= 0) and
      (Lines.IndexOf('      technical  0.1500') < Lines.IndexOf('    * billing    0.8000')),
      'a choice answer prints the choice, its confidence and the probabilities' +
      ' in the server''s order');
    Check(HasLine('frustration (score): 1.3000 on levels 0..2   confidence 0.2500') and
      HasLine('      1  0.5000  annoyed'),
      'a score answer prints the expected level and the legend per level');
    Check(HasLine('urgent (noul): P(true) = 0.7300'),
      'a noul answer prints P(true)');
    Check(HasLine('usage: 2140 input tokens, 0 output tokens'),
      'the usage line follows the answers');

    Lines.Clear;
    ExitStatus := ReplyToLines(200, csChoiceScoreNoulReply, true, Lines);
    Check((ExitStatus = ceExitOK) and (Lines.Count = 1) and
      (Lines[0] = csChoiceScoreNoulReply), '--raw prints the body unchanged');

    Lines.Clear;
    ExitStatus := ReplyToLines(422, csValidationReply, false, Lines);
    Check((ExitStatus = ceExitHTTPError) and HasLine('HTTP 422') and
      HasLine('  body.questions.team.criteria: Field required (missing)') and
      HasLine('  body.questions.level.criteria.1: Value error, a score level' +
        ' must not be null (value_error)'),
      'a 422 detail list prints one loc: msg (type) line per error, exit 1');

    Lines.Clear;
    ExitStatus := ReplyToLines(501, csDecisionOnlyReply, false, Lines);
    Check((ExitStatus = ceExitHTTPError) and HasLine('HTTP 501') and
      HasLine('  no decision scorer is active on this server, so POST' +
        ' /v1/systemone cannot answer'),
      'a 501 string detail prints the detail, exit 1');

    Lines.Clear;
    ExitStatus := ReplyToLines(404,
      '{"error":{"message":"no such endpoint","type":"invalid_request_error"}}',
      false, Lines);
    Check((ExitStatus = ceExitHTTPError) and HasLine('  no such endpoint'),
      'an OpenAI-style error body prints its message, exit 1');

    Lines.Clear;
    ExitStatus := ReplyToLines(200, '{"model":"x"}', false, Lines);
    Check(ExitStatus = ceExitHTTPError, 'a 200 reply without answers exits with 1');

    Args.CommaText := '--server,http://h:1,--model,m,--retries,5,--raw';
    Check(ParseClientArgs(Args, Opt, ErrorMsg) and (Opt.ServerURL = 'http://h:1') and
      (Opt.ModelName = 'm') and (Opt.RetryCount = 5) and Opt.PrintRaw,
      'the command line sets server, model, retries and --raw');
    Args.CommaText := '--retries,-1';
    Check(not ParseClientArgs(Args, Opt, ErrorMsg), 'a negative --retries is refused');
    Args.CommaText := '--bogus';
    Check(not ParseClientArgs(Args, Opt, ErrorMsg), 'an unknown option is refused');
    Args.CommaText := '--request';
    Check(not ParseClientArgs(Args, Opt, ErrorMsg), 'a flag without its value is refused');
  finally
    Args.Free;
    Lines.Free;
  end;

  if Failures = 0 then WriteLn('SELFTEST OK')
  else
  begin
    WriteLn('SELFTEST FAILED: ', Failures, ' check(s)');
    Halt(1);
  end;
end;

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
var
  Opt: TClientOptions;
  Args, RequestLines, Lines: TStringList;
  ArgPos: integer;
  URL, Body, ResponseBody, ErrorMsg: string;
  StatusCode, ExitStatus: integer;
  IsCommandLineValid, IsRequestValid: boolean;
begin
  Args := TStringList.Create;
  for ArgPos := 1 to ParamCount do Args.Add(ParamStr(ArgPos));
  IsCommandLineValid := ParseClientArgs(Args, Opt, ErrorMsg);
  Args.Free;
  if not IsCommandLineValid then
  begin
    WriteLn('Error: ', ErrorMsg);
    WriteLn;
    PrintUsage();
    Halt(ceExitUsage);
  end;
  if Opt.RunSelfTest then
  begin
    SelfTest();
    Halt(ceExitOK);
  end;
  if Opt.ShowHelp then
  begin
    PrintUsage();
    Halt(ceExitOK);
  end;

  if Opt.RequestFile = '' then
  begin
    if Opt.ModelName = '' then Body := BuildDemoRequest(csDefaultDecisionModel)
    else Body := BuildDemoRequest(Opt.ModelName);
  end
  else
  begin
    if not FileExists(Opt.RequestFile) then
    begin
      WriteLn('Error: request file not found: ', Opt.RequestFile);
      Halt(ceExitUsage);
    end;
    RequestLines := TStringList.Create;
    RequestLines.LoadFromFile(Opt.RequestFile);
    IsRequestValid := PrepareRequestBody(RequestLines.Text, Opt.ModelName, Body,
      ErrorMsg);
    RequestLines.Free;
    if not IsRequestValid then
    begin
      WriteLn('Error: ', Opt.RequestFile, ': ', ErrorMsg);
      Halt(ceExitUsage);
    end;
  end;

  URL := SystemOneURL(Opt.ServerURL);
  if not PostDecisionRequest(URL, Body, Opt.APIKey, Opt.RetryCount, StatusCode,
    ResponseBody, ErrorMsg) then
  begin
    WriteLn('Error: cannot reach ', URL, ': ', ErrorMsg);
    WriteLn('Start ChatServer first (it answers once the model has loaded), e.g.');
    WriteLn('  ChatServer /path/to/model --port 8080');
    WriteLn('or wait for a loading server with --retries N.');
    Halt(ceExitUnreachable);
  end;

  Lines := TStringList.Create;
  ExitStatus := ReplyToLines(StatusCode, ResponseBody, Opt.PrintRaw, Lines);
  Write(Lines.Text);
  Lines.Free;
  Halt(ExitStatus);
end.
