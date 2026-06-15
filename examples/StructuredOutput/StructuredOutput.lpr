// StructuredOutput example
//
// Schema-constrained (function-calling) decoding: extends the ConstrainedDecoding
// example from a hardcoded free-form JSON grammar (TNNetJSONConstraint) to a
// USER-SUPPLIED JSON Schema. CompileJSONSchemaToGBNF (neuraldecode) turns a JSON
// Schema into a GBNF grammar the existing TNNetGrammar consumes, and
// CreateJSONSchemaConstraint wraps that grammar in a TNNetGrammarConstraint - the
// same "allowed next tokens" hook the streamed generation loop applies to the
// post-softmax probability row before the sampler.
//
// The schema below is a typical tool-call arguments object (think
// get_weather(location, days, unit)): two required fields and one optional
// enum, additionalProperties:false. Because the constraint only ever exposes
// grammar-legal tokens, even this tiny UNTRAINED char-level network can ONLY
// emit JSON that VALIDATES against the schema - the right keys, in order, with
// the right value types, and nothing extra. EOS only becomes legal once a
// complete schema-valid object stands.
//
// As in ConstrainedDecoding, a small hand-set bias toward structural JSON
// characters keeps the sampled values short; it is a stand-in for a trained
// model's preferences and plays NO part in correctness - validity comes from
// the compiled grammar alone. Every sample is checked back through a fresh
// TNNetGrammar machine (and parsed as JSON) to prove it. Runs in seconds on CPU.
//
// Coded by Claude (AI).
program StructuredOutput;

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, fpjson, jsonparser,
  neuralnetwork, neuralvolume, neuraldecode;

const
  csVocab   = 128; // char-level: token id = character code, ids < 2 special
  csDim     = 16;
  csMaxNew  = 160;
  csCacheLen= 200;

  // A get_weather(...) tool-call arguments schema. Strict object: declared
  // properties only, "location" and "days" required, "unit" an optional enum.
  csSchema =
    '{' +
    '  "type": "object",' +
    '  "properties": {' +
    '    "location": { "type": "string" },' +
    '    "days":     { "type": "integer" },' +
    '    "unit":     { "enum": ["celsius", "fahrenheit"] }' +
    '  },' +
    '  "required": ["location", "days"],' +
    '  "additionalProperties": false' +
    '}';

// Builds the width-1 char-level next-token net used for streamed decoding.
function BuildCharLM(): TNNet;
begin
  Result := TNNet.Create();
  Result.AddLayer(TNNetInput.Create(1, 1, 1));
  Result.AddLayer(TNNetEmbedding.Create(csVocab, csDim, 0, 0.02));
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csDim));
  Result.AddLayer(TNNetDiagonalSSM.Create());
  Result.AddLayer(TNNetPointwiseConvLinear.Create(csVocab));
  Result.AddLayer(TNNetPointwiseSoftMax.Create());
end;

// Mild structural-character preference on the logit head (see the header note:
// a stand-in for a trained model so samples close quickly; validity is
// guaranteed by the constraint either way). The closing quote is biased above
// the other in-string characters so generated strings terminate promptly.
procedure BiasStructuralChars(Net: TNNet);
const
  Structural = '{}[]:,0123456789';
var
  Head: TNNetLayer;
  I: integer;
begin
  Head := Net.Layers[Net.Layers.Count - 2]; // the logit conv under softmax
  for I := 1 to Length(Structural) do
    Head.Neurons[Ord(Structural[I])].BiasWeight :=
      Head.Neurons[Ord(Structural[I])].BiasWeight + 1.5;
  // The closing quote is strongly favored so sampled string values terminate
  // quickly (an untrained model would otherwise wander inside the string until
  // the length cap, leaving the object incomplete).
  Head.Neurons[Ord('"')].BiasWeight := Head.Neurons[Ord('"')].BiasWeight + 7.0;
  // EOS mildly attractive so complete objects actually stop.
  Head.Neurons[1].BiasWeight := Head.Neurons[1].BiasWeight + 1.5;
  // No-op multiply so the conv layer's packed weights see the hand edits.
  Head.MulWeights(1.0);
end;

// One streamed generation pass from the prompt 'a'; returns the emitted
// characters (stops at the first special id).
function Generate(Net: TNNet; Sampler: TNNetSamplerBase;
  Constraint: TNNetTokenConstraint): string;
var
  Session: TNNetStreamingDecoder;
  Toks: TNeuralIntegerArray;
  OutLen, T: integer;
begin
  Result := '';
  Session := TNNetStreamingDecoder.Create(Net, csCacheLen);
  try
    SetLength(Toks, 1);
    Toks[0] := Ord('a');
    OutLen := GenerateTokensStreamed(Session, Toks, 1, csMaxNew, csCacheLen,
      Sampler, nil, nil, Constraint);
    for T := 1 to OutLen - 1 do
    begin
      if Toks[T] < 2 then break;
      Result := Result + Chr(Toks[T]);
    end;
  finally
    Session.Free;
  end;
end;

var
  Net: TNNet;
  Sampler: TNNetSamplerBase;
  Dict: TStringListInt;
  Schema: TNNetGrammarConstraint;
  CheckG: TNNetGrammar;
  Checker: TNNetGrammarMachine;
  Parsed: TJSONData;
  S, GBNF: string;
  Round, I: integer;
begin
  RandSeed := 20260614;
  Net := BuildCharLM();
  BiasStructuralChars(Net);
  Sampler := TNNetSamplerTopP.Create(0.9);

  // Show the compiled grammar once (this is exactly what the constraint runs).
  GBNF := CompileJSONSchemaToGBNF(csSchema);
  WriteLn('=== JSON Schema -> GBNF ===');
  WriteLn(GBNF);

  // Char-level Dict: token id = character code (ids 0/1 special). This is the
  // convention DecodeGreedy / the streamed loop use, and the one the constraint
  // snapshots via Dict.DeTokenize(id).
  Dict := TStringListInt.Create();
  Dict.Sorted := false;
  Dict.Add('<eos>'); // id 0
  Dict.Add('<pad>'); // id 1
  for I := 2 to csVocab - 1 do Dict.Add(Chr(I));
  Dict.SaveCurrentPosition();

  Schema := CreateJSONSchemaConstraint(csSchema, Dict);
  CheckG := TNNetGrammar.Create(GBNF);
  Checker := TNNetGrammarMachine.Create(CheckG);
  try
    WriteLn('=== FREE decoding (untrained net, top-p 0.9) ===');
    for Round := 1 to 2 do
      WriteLn('  sample ', Round, ': "', Generate(Net, Sampler, nil), '"');
    WriteLn;
    WriteLn('=== Schema-constrained decoding (same net, same sampler) ===');
    for Round := 1 to 4 do
    begin
      S := Generate(Net, Sampler, Schema);
      // Every generated byte is, BY CONSTRUCTION, a legal grammar continuation:
      // an independent fresh machine must accept the whole string as a prefix.
      // (It may stop INCOMPLETE only if the length cap cut it off mid-object.)
      Checker.Reset();
      if not Checker.FeedString(S) then
        raise Exception.Create('BUG: emitted text is not a legal prefix: ' + S);
      WriteLn('  sample ', Round, ': ', S);
      if Checker.IsComplete() then
      begin
        // A complete object also parses as JSON.
        Parsed := nil;
        try
          Parsed := GetJSON(S);
        finally
          Parsed.Free;
        end;
        WriteLn('     -> schema-valid: yes; parses as JSON: yes');
      end
      else
        WriteLn('     -> legal prefix; hit the length cap before closing');
    end;
  finally
    Checker.Free;
    CheckG.Free;
    Schema.Free;
    Dict.Free;
    Sampler.Free;
    Net.Free;
  end;
end.
