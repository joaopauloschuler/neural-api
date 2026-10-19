unit neuraldecision;

(*
neuraldecision
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

// neuraldecision -- the model-free half of typed decision models (TypeSafe's
// POST /v1/systemone wire format, as served by Jev, Kev and AnyJev): a state
// plus typed questions in, probabilities over each question's options out.
//
//   - ParseDecisionRequest validates a request body and reports failures as
//     FastAPI-style 422 details ({"detail":[{loc, msg, type}]}).
//   - RenderDecisionValue flattens string/object/array values to the text a
//     scorer feeds the model (Kev's api.render).
//   - RewriteSpecialTokenStrings turns "<|x|>" into "<¦x¦>" so caller text
//     cannot spell a special token.
//   - DecisionAnswerFromProbabilities + DecisionResponseToJSON derive the
//     answer fields (argmax, expected level, confidence) and write the response.
//   - RestrictedSoftMax turns a logits row into probabilities over a few ids.
//
// fpjson 3.2.2 keeps object keys in 255-byte shortstrings, so the string entry
// point shields longer keys (HFShieldLongJSONKeys) and rejects question ids and
// option names over 255 bytes with a 422 instead of letting them alias.

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, neuralvolume;

const
  csDecisionMaxChoiceOptions = 255;
  csDecisionMaxScoreLevels = 10;
  // fpjson's TJSONObject member names are shortstrings of at most this length.
  csDecisionMaxJSONKeyBytes = 255;

type
  TDecisionQuestionKind = (dqkChoice, dqkScore, dqkNoul);
  TDecisionProbabilities = array of double;

  TDecisionQuestion = record
    Id: string;                        // the caller's key in "questions"
    Kind: TDecisionQuestionKind;
    Instructions: string;              // rendered
    // choice: criteria names in key order; score: '0'..'n-1'; noul: 'false', 'true'
    OptionNames: TStringArray;
    OptionDescriptions: TStringArray;  // rendered; '' for null or absent
    LegendJSON: TStringArray;          // score only: each level's original JSON value
  end;
  TDecisionQuestions = array of TDecisionQuestion;

  TDecisionRequest = record
    Model: string;
    State: string;                     // rendered
    Questions: TDecisionQuestions;     // in request key order
  end;

  TDecisionAnswer = record
    Kind: TDecisionQuestionKind;
    Probabilities: TDecisionProbabilities; // in option order (noul: false, true)
    ChoiceIndex: integer;              // choice: first argmax
    Score: double;                     // score: sum i * p_i
    Noul: double;                      // noul: p(true)
    Confidence: double;                // choice and score
  end;
  TDecisionAnswers = array of TDecisionAnswer;

  // One element of a 422 "loc" path: a field name, or an array position when Index >= 0.
  TDecisionLocPart = record
    Name: string;
    Index: integer;
  end;
  TDecisionLoc = array of TDecisionLocPart;

  TDecisionErrorDetail = record
    Loc: TDecisionLoc;
    Msg: string;
    ErrorType: string;
  end;
  TDecisionErrorDetails = array of TDecisionErrorDetail;

// Parses and validates a /v1/systemone body. False: Errors holds the 422 details.
function ParseDecisionRequest(const Body: string; out Request: TDecisionRequest;
  out Errors: TDecisionErrorDetails): boolean; overload;
// Root parsed by the caller; ShieldedKeys from HFShieldLongJSONKeys, or nil, which
// rejects 255-byte ids and option names as possibly truncated by fpjson.
function ParseDecisionRequest(Root: TJSONData; ShieldedKeys: TStrings;
  out Request: TDecisionRequest; out Errors: TDecisionErrorDetails): boolean; overload;

// {"detail":[{"loc":[...],"msg":...,"type":...}, ...]}
function DecisionErrorsToJSON(const Errors: TDecisionErrorDetails): string;
// {"detail":"Msg"}, the body of every non-validation error.
function DecisionErrorMessageToJSON(const Msg: string): string;

// Kev's api.render: null -> '', scalars as Python str(), list -> "{pad}- item"
// lines, object -> "{pad}key: value" lines, nested values one indent (2 spaces) deeper.
function RenderDecisionValue(Value: TJSONData; Indent: integer = 0;
  ShieldedKeys: TStrings = nil): string;
// Compact JSON (Python json.dumps separators=(',', ':'), ensure_ascii=False).
function DecisionValueToJSON(Value: TJSONData; ShieldedKeys: TStrings = nil): string;
// S as a quoted JSON string, non-ASCII kept as raw UTF-8.
function DecisionJSONString(const S: string): string;
// Python repr(float): shortest round-trip digits, exponent form below 1e-4 and from
// 1e16. Subnormals may keep one digit too many (5e-324 -> 4.9e-324).
function DecisionFloatToStr(V: double): string;

// "<|name|>" -> "<¦name¦>" (U+00A6), name = [A-Za-z0-9_]+, as Kev's _SPECIAL_RE.
function RewriteSpecialTokenStrings(const Text: string): string;
// Kev's option_text: Name, or "Name: Description" when Description is not ''.
function DecisionOptionText(const Name, Description: string): string;

// (p_max - 1/n) / (1 - 1/n) on the normalised p; 1 when n = 1.
function ChoiceConfidence(const P: TDecisionProbabilities): double;
// max(0, 1 - sum p_i |i - mode| / D), D = mean |i - (n-1)/2|; 1 when n = 1.
function ScoreConfidence(const P: TDecisionProbabilities): double;
// Round to 4 decimals, the precision of every number in the response.
function RoundDecisionProbability(X: double): double;
// Fills the derived fields of an answer from its option probabilities.
function DecisionAnswerFromProbabilities(Kind: TDecisionQuestionKind;
  const P: TDecisionProbabilities): TDecisionAnswer;
// {model, answers:{id:{...}}, usage:{input_tokens, output_tokens}}; Answers in question order.
function DecisionResponseToJSON(const Request: TDecisionRequest;
  const Answers: TDecisionAnswers; InputTokens, OutputTokens: integer): string;

// Probs (resized to the id count) := softmax over LogitsRow[TokenIds[k]] only.
procedure RestrictedSoftMax(LogitsRow: TNeuralFloatArrPtr; LogitCount: integer;
  const TokenIds: array of integer; Probs: TNNetVolume); overload;
procedure RestrictedSoftMax(LogitsRow: TNNetVolume;
  const TokenIds: array of integer; Probs: TNNetVolume); overload;
function DecisionProbabilitiesFromVolume(V: TNNetVolume): TDecisionProbabilities;

implementation

uses
  Math, neuralhftokenizer;

var
  DecisionFormat: TFormatSettings;

function LocName(const Name: string): TDecisionLocPart;
begin
  Result.Name := Name;
  Result.Index := -1;
end;

function LocIndex(Index: integer): TDecisionLocPart;
begin
  Result.Name := '';
  Result.Index := Index;
end;

procedure AddDecisionError(var Errors: TDecisionErrorDetails;
  const Loc: array of TDecisionLocPart; const Msg, ErrorType: string);
var
  ErrorPos, LocPos, MaxLocPos: integer;
begin
  ErrorPos := Length(Errors);
  SetLength(Errors, ErrorPos + 1);
  SetLength(Errors[ErrorPos].Loc, Length(Loc));
  MaxLocPos := High(Loc);
  for LocPos := 0 to MaxLocPos do
    Errors[ErrorPos].Loc[LocPos] := Loc[LocPos];
  Errors[ErrorPos].Msg := Msg;
  Errors[ErrorPos].ErrorType := ErrorType;
end;

// fpjson decodes a non-ASCII escape through the widestring manager ('?' without a
// UTF-8 one), so they become raw UTF-8 here; quote, backslash and controls stay escaped.
function DecodeJSONEscapesForFPJSON(const S: string): string;
var
  Position, Total, OutPos, EscapeLen: integer;
  CP: cardinal;
  PBuf: PAnsiChar;
begin
  Total := Length(S);
  SetLength(Result, Total);
  PBuf := PAnsiChar(Result);
  OutPos := 0;
  Position := 1;
  while Position <= Total do
  begin
    if (S[Position] = '\') and (Position < Total) then
    begin
      if (S[Position + 1] = 'u') and
        HFReadUnicodeEscape(S, Position + 2, CP, EscapeLen) and
        (CP >= $20) then
      begin
        Inc(Position, 2 + EscapeLen);
        if (CP = $22) or (CP = $5C) then
        begin
          PBuf[OutPos] := '\';
          PBuf[OutPos + 1] := Chr(CP);
          Inc(OutPos, 2);
        end
        else
          HFWriteCodePointUTF8(PBuf, OutPos, CP);
      end
      else
      begin
        PBuf[OutPos] := S[Position];
        PBuf[OutPos + 1] := S[Position + 1];
        Inc(OutPos, 2);
        Inc(Position, 2);
      end;
    end
    else
    begin
      PBuf[OutPos] := S[Position];
      Inc(OutPos);
      Inc(Position);
    end;
  end;
  SetLength(Result, OutPos);
end;

function ObjectKey(Obj: TJSONObject; Index: integer;
  ShieldedKeys: TStrings): string;
begin
  Result := HFUnshieldJSONKey(Obj.Names[Index], ShieldedKeys);
end;

// True when Key may not be the caller's key: over 255 bytes, or (no shield
// list) exactly 255 bytes, the length fpjson truncates longer keys to.
function IsUnsafeJSONKey(const Key: string; ShieldedKeys: TStrings): boolean;
begin
  if ShieldedKeys = nil then
    Result := Length(Key) >= csDecisionMaxJSONKeyBytes
  else
    Result := Length(Key) > csDecisionMaxJSONKeyBytes;
end;

const
  csUnsafeKeyMsg = 'Value error, object keys longer than 255 bytes are not supported';

function DecisionFloatToStr(V: double): string;
var
  Precision, ExpPos, Exp10, DigitCount, PointPos: integer;
  Mantissa, Digits: string;
  Parsed: double;
  ParsedWide: extended;
  Negative: boolean;
begin
  if IsNan(V) then Exit('NaN');
  if IsInfinite(V) then
  begin
    if V > 0 then Exit('Infinity') else Exit('-Infinity');
  end;
  if V = 0 then
  begin
    if (PQWord(@V)^ shr 63) <> 0 then Exit('-0.0') else Exit('0.0');
  end;
  Negative := V < 0;
  if Negative then V := -V;
  Mantissa := '';
  for Precision := 1 to 17 do
  begin
    Mantissa := FloatToStrF(V, ffExponent, Precision, 0, DecisionFormat);
    // StrToFloat returns Extended: round to double (when in range) before comparing.
    ParsedWide := StrToFloat(Mantissa, DecisionFormat);
    if ParsedWide > MaxDouble then continue;
    Parsed := ParsedWide;
    if Parsed = V then break;
  end;
  // Mantissa = "d.dddE+X", or "d.ddd" when the exponent is 0
  ExpPos := Pos('E', Mantissa);
  if ExpPos = 0 then
  begin
    Exp10 := 0;
    ExpPos := Length(Mantissa) + 1;
  end
  else
    Exp10 := StrToInt(Copy(Mantissa, ExpPos + 1, Length(Mantissa)));
  Digits := StringReplace(Copy(Mantissa, 1, ExpPos - 1), '.', '', []);
  DigitCount := Length(Digits);
  while (DigitCount > 1) and (Digits[DigitCount] = '0') do Dec(DigitCount);
  SetLength(Digits, DigitCount);
  if (Exp10 < -4) or (Exp10 >= 16) then
  begin
    Result := Digits[1];
    if DigitCount > 1 then Result := Result + '.' + Copy(Digits, 2, DigitCount);
    if Exp10 < 0 then
      Result := Result + 'e-' + Format('%.2d', [-Exp10])
    else
      Result := Result + 'e+' + Format('%.2d', [Exp10]);
  end
  else if Exp10 < 0 then
    Result := '0.' + StringOfChar('0', -Exp10 - 1) + Digits
  else
  begin
    PointPos := Exp10 + 1;
    if DigitCount <= PointPos then
      Result := Digits + StringOfChar('0', PointPos - DigitCount) + '.0'
    else
      Result := Copy(Digits, 1, PointPos) + '.' +
        Copy(Digits, PointPos + 1, DigitCount);
  end;
  if Negative then Result := '-' + Result;
end;

function DecisionJSONString(const S: string): string;
var
  Position, Total: integer;
  C: char;
  Builder: TStringBuilder;
begin
  Builder := TStringBuilder.Create(Length(S) + 2);
  try
    Builder.Append('"');
    Total := Length(S);
    for Position := 1 to Total do
    begin
      C := S[Position];
      case C of
        '"': Builder.Append('\"');
        '\': Builder.Append('\\');
        #8: Builder.Append('\b');
        #9: Builder.Append('\t');
        #10: Builder.Append('\n');
        #12: Builder.Append('\f');
        #13: Builder.Append('\r');
        #0..#7, #11, #14..#31:
          Builder.Append('\u00' + LowerCase(IntToHex(Ord(C), 2)));
        else Builder.Append(C);
      end;
    end;
    Builder.Append('"');
    Result := Builder.ToString();
  finally
    Builder.Free;
  end;
end;

function ScalarToText(Value: TJSONData): string;
begin
  case Value.JSONType of
    jtNull: Result := '';
    jtBoolean:
      if Value.AsBoolean then Result := 'True' else Result := 'False';
    jtNumber:
      if Value is TJSONFloatNumber then
        Result := DecisionFloatToStr(Value.AsFloat)
      else
        Result := Value.AsString;
    else Result := Value.AsString;
  end;
end;

// Python str.lstrip() over the ASCII whitespace it strips.
function LeftStripPython(const S: string): string;
var
  First, Total: integer;
begin
  Total := Length(S);
  First := 1;
  while (First <= Total) and (S[First] in [#9..#13, #28..#32]) do Inc(First);
  Result := Copy(S, First, Total);
end;

function RenderDecisionValue(Value: TJSONData; Indent: integer;
  ShieldedKeys: TStrings): string;
var
  Pad: string;
  ItemPos, MaxItemPos: integer;
  Item: TJSONData;
  Lines: TStringList;
begin
  if Value = nil then Exit('');
  if not (Value.JSONType in [jtArray, jtObject]) then Exit(ScalarToText(Value));
  Pad := StringOfChar(' ', 2 * Indent);
  Lines := TStringList.Create;
  try
    MaxItemPos := Value.Count - 1;
    for ItemPos := 0 to MaxItemPos do
    begin
      Item := Value.Items[ItemPos];
      if Value.JSONType = jtArray then
        Lines.Add(Pad + '- ' + LeftStripPython(
          RenderDecisionValue(Item, Indent + 1, ShieldedKeys)))
      else if Item.JSONType in [jtArray, jtObject] then
        Lines.Add(Pad + ObjectKey(TJSONObject(Value), ItemPos, ShieldedKeys) +
          ':' + #10 + RenderDecisionValue(Item, Indent + 1, ShieldedKeys))
      else
        Lines.Add(Pad + ObjectKey(TJSONObject(Value), ItemPos, ShieldedKeys) +
          ': ' + ScalarToText(Item));
    end;
    Lines.LineBreak := #10;
    Lines.SkipLastLineBreak := true;
    Result := Lines.Text;
  finally
    Lines.Free;
  end;
end;

function DecisionValueToJSON(Value: TJSONData; ShieldedKeys: TStrings): string;
var
  ItemPos, MaxItemPos: integer;
  Builder: TStringBuilder;
begin
  if Value = nil then Exit('null');
  case Value.JSONType of
    jtNull: Result := 'null';
    jtBoolean:
      if Value.AsBoolean then Result := 'true' else Result := 'false';
    jtNumber:
      if Value is TJSONFloatNumber then
        Result := DecisionFloatToStr(Value.AsFloat)
      else
        Result := Value.AsString;
    jtString: Result := DecisionJSONString(Value.AsString);
    jtArray, jtObject:
    begin
      Builder := TStringBuilder.Create;
      try
        if Value.JSONType = jtArray then Builder.Append('[') else Builder.Append('{');
        MaxItemPos := Value.Count - 1;
        for ItemPos := 0 to MaxItemPos do
        begin
          if ItemPos > 0 then Builder.Append(',');
          if Value.JSONType = jtObject then
            Builder.Append(DecisionJSONString(
              ObjectKey(TJSONObject(Value), ItemPos, ShieldedKeys))).Append(':');
          Builder.Append(DecisionValueToJSON(Value.Items[ItemPos], ShieldedKeys));
        end;
        if Value.JSONType = jtArray then Builder.Append(']') else Builder.Append('}');
        Result := Builder.ToString();
      finally
        Builder.Free;
      end;
    end;
    else Result := 'null';
  end;
end;

function RewriteSpecialTokenStrings(const Text: string): string;
var
  Position, NameEnd, Total: integer;
  Builder: TStringBuilder;
begin
  if Pos('<|', Text) = 0 then Exit(Text);
  Total := Length(Text);
  Builder := TStringBuilder.Create(Total + 8);
  try
    Position := 1;
    while Position <= Total do
    begin
      if (Text[Position] = '<') and (Position < Total) and
        (Text[Position + 1] = '|') then
      begin
        NameEnd := Position + 2;
        while (NameEnd <= Total) and
          (Text[NameEnd] in ['A'..'Z', 'a'..'z', '0'..'9', '_']) do
          Inc(NameEnd);
        if (NameEnd > Position + 2) and (NameEnd < Total) and
          (Text[NameEnd] = '|') and (Text[NameEnd + 1] = '>') then
        begin
          Builder.Append('<'#$C2#$A6);
          Builder.Append(Copy(Text, Position + 2, NameEnd - Position - 2));
          Builder.Append(#$C2#$A6'>');
          Position := NameEnd + 2;
          continue;
        end;
      end;
      Builder.Append(Text[Position]);
      Inc(Position);
    end;
    Result := Builder.ToString();
  finally
    Builder.Free;
  end;
end;

function DecisionOptionText(const Name, Description: string): string;
begin
  if Description = '' then
    Result := Name
  else
    Result := Name + ': ' + Description;
end;

// p / sum(p), or uniform when the sum is 0 (TypeSafe's reference adapter).
function NormalizedProbabilities(const P: TDecisionProbabilities): TDecisionProbabilities;
var
  OptionPos, MaxOptionPos: integer;
  Total: double;
begin
  Result := nil;
  SetLength(Result, Length(P));
  MaxOptionPos := High(P);
  Total := 0;
  for OptionPos := 0 to MaxOptionPos do Total := Total + P[OptionPos];
  for OptionPos := 0 to MaxOptionPos do
    if Total = 0 then
      Result[OptionPos] := 1 / Length(P)
    else
      Result[OptionPos] := P[OptionPos] / Total;
end;

function FirstArgMax(const P: TDecisionProbabilities): integer;
var
  OptionPos, MaxOptionPos: integer;
begin
  Result := 0;
  MaxOptionPos := High(P);
  for OptionPos := 1 to MaxOptionPos do
    if P[OptionPos] > P[Result] then Result := OptionPos;
end;

function ChoiceConfidence(const P: TDecisionProbabilities): double;
var
  OptionCount: integer;
  Normalized: TDecisionProbabilities;
begin
  OptionCount := Length(P);
  if OptionCount <= 1 then Exit(1.0);
  Normalized := NormalizedProbabilities(P);
  Result := (Normalized[FirstArgMax(Normalized)] - 1 / OptionCount) /
    (1 - 1 / OptionCount);
end;

function ScoreConfidence(const P: TDecisionProbabilities): double;
var
  LevelCount, Level, MaxLevel, Mode: integer;
  Normalized: TDecisionProbabilities;
  UniformSpread, Spread, Center: double;
begin
  LevelCount := Length(P);
  if LevelCount <= 1 then Exit(1.0);
  Normalized := NormalizedProbabilities(P);
  Mode := FirstArgMax(Normalized);
  Center := (LevelCount - 1) / 2;
  MaxLevel := LevelCount - 1;
  UniformSpread := 0;
  Spread := 0;
  for Level := 0 to MaxLevel do
  begin
    UniformSpread := UniformSpread + Abs(Level - Center);
    Spread := Spread + Normalized[Level] * Abs(Level - Mode);
  end;
  UniformSpread := UniformSpread / LevelCount;
  Result := 1 - Spread / UniformSpread;
  if Result < 0 then Result := 0;
end;

function RoundDecisionProbability(X: double): double;
begin
  Result := Round(X * 10000) / 10000;
end;

function DecisionAnswerFromProbabilities(Kind: TDecisionQuestionKind;
  const P: TDecisionProbabilities): TDecisionAnswer;
var
  Level, MaxLevel: integer;
begin
  Result := Default(TDecisionAnswer);
  Result.Kind := Kind;
  Result.Probabilities := Copy(P);
  if Length(P) = 0 then exit;
  case Kind of
    dqkChoice:
    begin
      Result.ChoiceIndex := FirstArgMax(P);
      Result.Confidence := ChoiceConfidence(P);
    end;
    dqkScore:
    begin
      MaxLevel := High(P);
      for Level := 0 to MaxLevel do
        Result.Score := Result.Score + Level * P[Level];
      Result.Confidence := ScoreConfidence(P);
    end;
    dqkNoul:
      Result.Noul := P[High(P)];
  end;
end;

function RoundedJSON(X: double): string;
begin
  Result := DecisionFloatToStr(RoundDecisionProbability(X));
end;

function DecisionResponseToJSON(const Request: TDecisionRequest;
  const Answers: TDecisionAnswers; InputTokens, OutputTokens: integer): string;
var
  QuestionPos, MaxQuestionPos, OptionPos, MaxOptionPos: integer;
  Builder: TStringBuilder;
  Question: TDecisionQuestion;
  Answer: TDecisionAnswer;

  procedure AppendProbabilities;
  var
    OptionPos, MaxOptionPos: integer;
  begin
    MaxOptionPos := High(Question.OptionNames);
    Builder.Append('"probabilities":{');
    for OptionPos := 0 to MaxOptionPos do
    begin
      if OptionPos > 0 then Builder.Append(',');
      Builder.Append(DecisionJSONString(Question.OptionNames[OptionPos]))
        .Append(':').Append(RoundedJSON(Answer.Probabilities[OptionPos]));
    end;
    Builder.Append('}');
  end;

begin
  if Length(Answers) <> Length(Request.Questions) then
    raise Exception.Create('DecisionResponseToJSON: ' + IntToStr(Length(Answers)) +
      ' answers for ' + IntToStr(Length(Request.Questions)) + ' questions');
  Builder := TStringBuilder.Create;
  try
    Builder.Append('{"model":').Append(DecisionJSONString(Request.Model))
      .Append(',"answers":{');
    MaxQuestionPos := High(Request.Questions);
    for QuestionPos := 0 to MaxQuestionPos do
    begin
      Question := Request.Questions[QuestionPos];
      Answer := Answers[QuestionPos];
      if Length(Answer.Probabilities) <> Length(Question.OptionNames) then
        raise Exception.Create('DecisionResponseToJSON: question "' +
          Question.Id + '" has ' + IntToStr(Length(Question.OptionNames)) +
          ' options but ' + IntToStr(Length(Answer.Probabilities)) + ' probabilities');
      MaxOptionPos := High(Question.OptionNames);
      if QuestionPos > 0 then Builder.Append(',');
      Builder.Append(DecisionJSONString(Question.Id)).Append(':{"type":');
      case Question.Kind of
        dqkChoice:
        begin
          Builder.Append('"choice","choice":')
            .Append(DecisionJSONString(Question.OptionNames[Answer.ChoiceIndex]))
            .Append(',');
          AppendProbabilities;
          Builder.Append(',"confidence":').Append(RoundedJSON(Answer.Confidence));
        end;
        dqkScore:
        begin
          Builder.Append('"score","score":').Append(RoundedJSON(Answer.Score))
            .Append(',"legend":{');
          for OptionPos := 0 to MaxOptionPos do
          begin
            if OptionPos > 0 then Builder.Append(',');
            Builder.Append(DecisionJSONString(Question.OptionNames[OptionPos]))
              .Append(':').Append(Question.LegendJSON[OptionPos]);
          end;
          Builder.Append('},');
          AppendProbabilities;
          Builder.Append(',"confidence":').Append(RoundedJSON(Answer.Confidence));
        end;
        dqkNoul:
          Builder.Append('"noul","noul":').Append(RoundedJSON(Answer.Noul));
      end;
      Builder.Append('}');
    end;
    Builder.Append('},"usage":{"input_tokens":').Append(InputTokens)
      .Append(',"output_tokens":').Append(OutputTokens).Append('}}');
    Result := Builder.ToString();
  finally
    Builder.Free;
  end;
end;

function DecisionErrorsToJSON(const Errors: TDecisionErrorDetails): string;
var
  ErrorPos, MaxErrorPos, LocPos, MaxLocPos: integer;
  Builder: TStringBuilder;
begin
  Builder := TStringBuilder.Create;
  try
    Builder.Append('{"detail":[');
    MaxErrorPos := High(Errors);
    for ErrorPos := 0 to MaxErrorPos do
    begin
      if ErrorPos > 0 then Builder.Append(',');
      Builder.Append('{"loc":[');
      MaxLocPos := High(Errors[ErrorPos].Loc);
      for LocPos := 0 to MaxLocPos do
      begin
        if LocPos > 0 then Builder.Append(',');
        if Errors[ErrorPos].Loc[LocPos].Index >= 0 then
          Builder.Append(Errors[ErrorPos].Loc[LocPos].Index)
        else
          Builder.Append(DecisionJSONString(Errors[ErrorPos].Loc[LocPos].Name));
      end;
      Builder.Append('],"msg":').Append(DecisionJSONString(Errors[ErrorPos].Msg))
        .Append(',"type":').Append(DecisionJSONString(Errors[ErrorPos].ErrorType))
        .Append('}');
    end;
    Builder.Append(']}');
    Result := Builder.ToString();
  finally
    Builder.Free;
  end;
end;

function DecisionErrorMessageToJSON(const Msg: string): string;
begin
  Result := '{"detail":' + DecisionJSONString(Msg) + '}';
end;

function ParseDecisionQuestion(const Id: string; QuestionData: TJSONData;
  ShieldedKeys: TStrings; var Question: TDecisionQuestion;
  var Errors: TDecisionErrorDetails): boolean;
var
  QuestionObj, CriteriaObj: TJSONObject;
  Criteria, TypeData: TJSONData;
  TypeName, OptionName: string;
  OptionPos, MaxOptionPos, OptionCount, ErrorCount: integer;
  CriteriaLoc: array[0..3] of TDecisionLocPart;
begin
  ErrorCount := Length(Errors);
  Question := Default(TDecisionQuestion);
  Question.Id := Id;
  if not (QuestionData is TJSONObject) then
  begin
    AddDecisionError(Errors, [LocName('body'), LocName('questions'), LocName(Id)],
      'Input should be a valid dictionary or object to extract fields from',
      'model_attributes_type');
    Exit(false);
  end;
  QuestionObj := TJSONObject(QuestionData);
  TypeData := QuestionObj.Find('type');
  if TypeData = nil then
  begin
    AddDecisionError(Errors, [LocName('body'), LocName('questions'), LocName(Id)],
      'Unable to extract tag using discriminator ''type''', 'union_tag_not_found');
    Exit(false);
  end;
  if TypeData.JSONType = jtString then TypeName := TypeData.AsString
  else TypeName := TypeData.AsJSON;
  if TypeName = 'choice' then Question.Kind := dqkChoice
  else if TypeName = 'score' then Question.Kind := dqkScore
  else if TypeName = 'noul' then Question.Kind := dqkNoul
  else
  begin
    AddDecisionError(Errors, [LocName('body'), LocName('questions'), LocName(Id)],
      'Input tag ''' + TypeName + ''' found using ''type'' does not match any ' +
      'of the expected tags: ''choice'', ''score'', ''noul''', 'union_tag_invalid');
    Exit(false);
  end;
  Question.Instructions := RenderDecisionValue(QuestionObj.Find('instructions'),
    0, ShieldedKeys);
  CriteriaLoc[0] := LocName('body');
  CriteriaLoc[1] := LocName('questions');
  CriteriaLoc[2] := LocName(Id);
  CriteriaLoc[3] := LocName('criteria');
  Criteria := QuestionObj.Find('criteria');
  case Question.Kind of
    dqkChoice:
    begin
      if Criteria = nil then
        AddDecisionError(Errors, CriteriaLoc, 'Field required', 'missing')
      else if not (Criteria is TJSONObject) then
        AddDecisionError(Errors, CriteriaLoc, 'Input should be a valid dictionary',
          'dict_type')
      else if (Criteria.Count < 1) or
        (Criteria.Count > csDecisionMaxChoiceOptions) then
        AddDecisionError(Errors, CriteriaLoc, 'Value error, criteria must have 1..' +
          IntToStr(csDecisionMaxChoiceOptions) + ' options', 'value_error')
      else
      begin
        CriteriaObj := TJSONObject(Criteria);
        OptionCount := CriteriaObj.Count;
        SetLength(Question.OptionNames, OptionCount);
        SetLength(Question.OptionDescriptions, OptionCount);
        MaxOptionPos := OptionCount - 1;
        for OptionPos := 0 to MaxOptionPos do
        begin
          OptionName := ObjectKey(CriteriaObj, OptionPos, ShieldedKeys);
          if IsUnsafeJSONKey(OptionName, ShieldedKeys) then
            AddDecisionError(Errors, [CriteriaLoc[0], CriteriaLoc[1],
              CriteriaLoc[2], CriteriaLoc[3], LocName(OptionName)],
              csUnsafeKeyMsg, 'value_error');
          Question.OptionNames[OptionPos] := OptionName;
          Question.OptionDescriptions[OptionPos] := RenderDecisionValue(
            CriteriaObj.Items[OptionPos], 0, ShieldedKeys);
        end;
      end;
    end;
    dqkScore:
    begin
      if Criteria = nil then
        AddDecisionError(Errors, CriteriaLoc, 'Field required', 'missing')
      else if not (Criteria is TJSONArray) then
        AddDecisionError(Errors, CriteriaLoc, 'Input should be a valid list',
          'list_type')
      else if Criteria.Count < 1 then
        AddDecisionError(Errors, CriteriaLoc,
          'List should have at least 1 item after validation, not 0', 'too_short')
      else if Criteria.Count > csDecisionMaxScoreLevels then
        AddDecisionError(Errors, CriteriaLoc, 'List should have at most ' +
          IntToStr(csDecisionMaxScoreLevels) + ' items after validation, not ' +
          IntToStr(Criteria.Count), 'too_long')
      else
      begin
        OptionCount := Criteria.Count;
        SetLength(Question.OptionNames, OptionCount);
        SetLength(Question.OptionDescriptions, OptionCount);
        SetLength(Question.LegendJSON, OptionCount);
        MaxOptionPos := OptionCount - 1;
        for OptionPos := 0 to MaxOptionPos do
        begin
          if Criteria.Items[OptionPos].JSONType = jtNull then
            AddDecisionError(Errors, [CriteriaLoc[0], CriteriaLoc[1],
              CriteriaLoc[2], CriteriaLoc[3], LocIndex(OptionPos)],
              'Value error, a score level must not be null', 'value_error');
          Question.OptionNames[OptionPos] := IntToStr(OptionPos);
          Question.OptionDescriptions[OptionPos] := RenderDecisionValue(
            Criteria.Items[OptionPos], 0, ShieldedKeys);
          Question.LegendJSON[OptionPos] := DecisionValueToJSON(
            Criteria.Items[OptionPos], ShieldedKeys);
        end;
      end;
    end;
    dqkNoul:
    begin
      SetLength(Question.OptionNames, 2);
      SetLength(Question.OptionDescriptions, 2);
      Question.OptionNames[0] := 'false';
      Question.OptionNames[1] := 'true';
      if (Criteria <> nil) and (Criteria.JSONType <> jtNull) then
      begin
        if not (Criteria is TJSONObject) then
          AddDecisionError(Errors, CriteriaLoc, 'Input should be a valid dictionary',
            'dict_type')
        else
        begin
          CriteriaObj := TJSONObject(Criteria);
          Question.OptionDescriptions[0] := RenderDecisionValue(
            CriteriaObj.Find('false'), 0, ShieldedKeys);
          Question.OptionDescriptions[1] := RenderDecisionValue(
            CriteriaObj.Find('true'), 0, ShieldedKeys);
        end;
      end;
    end;
  end;
  Result := Length(Errors) = ErrorCount;
end;

function ParseDecisionRequest(Root: TJSONData; ShieldedKeys: TStrings;
  out Request: TDecisionRequest; out Errors: TDecisionErrorDetails): boolean;
var
  Body, QuestionsObj: TJSONObject;
  Field: TJSONData;
  QuestionPos, MaxQuestionPos, QuestionCount: integer;
  QuestionId: string;
begin
  Request := Default(TDecisionRequest);
  Errors := nil;
  if not (Root is TJSONObject) then
  begin
    AddDecisionError(Errors, [LocName('body')],
      'Input should be a valid dictionary or object to extract fields from',
      'model_attributes_type');
    Exit(false);
  end;
  Body := TJSONObject(Root);
  Field := Body.Find('model');
  if Field = nil then
    AddDecisionError(Errors, [LocName('body'), LocName('model')], 'Field required',
      'missing')
  else if Field.JSONType <> jtString then
    AddDecisionError(Errors, [LocName('body'), LocName('model')],
      'Input should be a valid string', 'string_type')
  else
    Request.Model := Field.AsString;
  Field := Body.Find('state');
  if Field = nil then
    AddDecisionError(Errors, [LocName('body'), LocName('state')], 'Field required',
      'missing')
  else
    Request.State := RenderDecisionValue(Field, 0, ShieldedKeys);
  Field := Body.Find('assets');
  if (Field <> nil) and (Field.JSONType <> jtNull) and
    not ((Field.JSONType in [jtArray, jtObject]) and (Field.Count = 0)) then
    AddDecisionError(Errors, [LocName('body'), LocName('assets')],
      'Value error, image assets are not supported', 'value_error');
  Field := Body.Find('questions');
  if Field = nil then
    AddDecisionError(Errors, [LocName('body'), LocName('questions')],
      'Field required', 'missing')
  else if not (Field is TJSONObject) then
    AddDecisionError(Errors, [LocName('body'), LocName('questions')],
      'Input should be a valid dictionary', 'dict_type')
  else if Field.Count = 0 then
    AddDecisionError(Errors, [LocName('body'), LocName('questions')],
      'Dictionary should have at least 1 item after validation, not 0', 'too_short')
  else
  begin
    QuestionsObj := TJSONObject(Field);
    QuestionCount := QuestionsObj.Count;
    SetLength(Request.Questions, QuestionCount);
    MaxQuestionPos := QuestionCount - 1;
    for QuestionPos := 0 to MaxQuestionPos do
    begin
      QuestionId := ObjectKey(QuestionsObj, QuestionPos, ShieldedKeys);
      if IsUnsafeJSONKey(QuestionId, ShieldedKeys) then
        AddDecisionError(Errors, [LocName('body'), LocName('questions'),
          LocName(QuestionId)], csUnsafeKeyMsg, 'value_error');
      ParseDecisionQuestion(QuestionId, QuestionsObj.Items[QuestionPos],
        ShieldedKeys, Request.Questions[QuestionPos], Errors);
    end;
  end;
  Result := Length(Errors) = 0;
end;

function ParseDecisionRequest(const Body: string; out Request: TDecisionRequest;
  out Errors: TDecisionErrorDetails): boolean;
var
  ShieldedKeys: TStringList;
  Root: TJSONData;
begin
  Request := Default(TDecisionRequest);
  Errors := nil;
  ShieldedKeys := TStringList.Create;
  Root := nil;
  try
    try
      Root := HFParseJSONRaw(HFShieldLongJSONKeys(
        DecodeJSONEscapesForFPJSON(Body), ShieldedKeys));
    except
      on E: Exception do
      begin
        AddDecisionError(Errors, [LocName('body')], 'JSON decode error: ' +
          E.Message, 'json_invalid');
        Exit(false);
      end;
    end;
    if Root = nil then
    begin
      AddDecisionError(Errors, [LocName('body')], 'JSON decode error: empty body',
        'json_invalid');
      Exit(false);
    end;
    Result := ParseDecisionRequest(Root, ShieldedKeys, Request, Errors);
  finally
    Root.Free;
    ShieldedKeys.Free;
  end;
end;

procedure RestrictedSoftMax(LogitsRow: TNeuralFloatArrPtr; LogitCount: integer;
  const TokenIds: array of integer; Probs: TNNetVolume);
var
  IdPos, MaxIdPos, TokenId: integer;
begin
  Probs.ReSize(Length(TokenIds), 1, 1);
  MaxIdPos := High(TokenIds);
  for IdPos := 0 to MaxIdPos do
  begin
    TokenId := TokenIds[IdPos];
    if (TokenId < 0) or (TokenId >= LogitCount) then
      raise Exception.Create('RestrictedSoftMax: token id ' + IntToStr(TokenId) +
        ' outside a row of ' + IntToStr(LogitCount) + ' logits');
    Probs.FData[IdPos] := LogitsRow^[TokenId];
  end;
  RowSoftMax(Probs);
end;

procedure RestrictedSoftMax(LogitsRow: TNNetVolume;
  const TokenIds: array of integer; Probs: TNNetVolume);
begin
  RestrictedSoftMax(LogitsRow.DataPtr, LogitsRow.Size, TokenIds, Probs);
end;

function DecisionProbabilitiesFromVolume(V: TNNetVolume): TDecisionProbabilities;
var
  ElementPos, MaxElementPos: integer;
begin
  Result := nil;
  SetLength(Result, V.Size);
  MaxElementPos := V.Size - 1;
  for ElementPos := 0 to MaxElementPos do
    Result[ElementPos] := V.FData[ElementPos];
end;

initialization
  DecisionFormat := DefaultFormatSettings;
  DecisionFormat.DecimalSeparator := '.';
  DecisionFormat.ThousandSeparator := #0;
end.
