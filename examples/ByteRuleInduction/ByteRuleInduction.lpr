program ByteRuleInduction;
(*
ByteRuleInduction: does the symbolic byte engine INDUCE a generalizable rule
from partial data, or does it just MEMORIZE the examples it saw?

This is a discovery experiment, not a demo. TNNetByteProcessing is powered by
TEasyLearnAndPredictClass, a non-gradient symbolic predictor that induces
discrete bit-pattern relations (cause -> effect). Unlike a dense net, its
learned state is human-readable (printRelationTable). That lets us ask a
question most frameworks can't answer cleanly: when we train it on only a
SUBSET of all 256 single-byte inputs, does it get the held-out inputs right?

Design
------
Independent variable 1 -- the target function's bit structure:

  * XOR  : out := in xor $5A     (bit-local: out bit i depends ONLY on in bit i)
  * NOT  : out := not in         (bit-local: pure per-bit complement)
  * INC  : out := (in + 1) and $FF   (carry-coupled: bit i depends on all lower
                                      bits via the carry chain)

XOR and NOT are fully bit-decomposable, so a learner that discovers the eight
independent per-bit rules from a partial sample should extrapolate to unseen
bytes. INC is the control: its carry chain couples the bits, so per-bit rules
cannot capture it and generalization should collapse. The contrast is the
point -- it isolates *why* induction can work (bit-locality) from *whether* it
happens.

Independent variable 2 -- the INPUT ENCODING, which turned out to be the
decisive factor:

  * BYTE : the input is one action byte (value 0..255). The engine then forms
           whole-byte equality relations ("A[0] = 82 => 8") -- i.e. a lookup
           table. It cannot represent a per-bit rule, so it memorizes the
           training bytes perfectly and scores 0% on every held-out byte,
           regardless of the function or FGeneralize. (This is the surprising
           result of the first cut of this experiment.)
  * BITS : the input byte is unpacked into 8 action positions, one bit each.
           Now a per-position relation ("A[3] = 0 => ...") IS representable, so
           the bit-local functions become inducible from a partial sample.

Independent variable 3 -- the engine's FGeneralize flag (False = only memorize
the patterns seen; True = occasionally clone neurons onto broader patterns).

Protocol
--------
For each (encoding, function, FGeneralize) cell: deterministic 50/50 train/test
split of the 256 inputs (fixed RandSeed for reproducibility), train ONLY on the
train half for a fixed number of shuffled epochs, then report accuracy
separately on the train half (did it fit?) and the held-out half (did it
generalize?). The gap between the two is the memorization-vs-induction signal.

Findings (this is a real experiment -- the data refined the hypothesis)
-----------------------------------------------------------------------
1. ENCODING is decisive. Under BYTE the engine forms whole-byte equality
   relations -> a lookup table: 100% train, 0% held-out for every function.
   Pure memorization. Under BITS it generalizes well above the 1/256 ~ 0.4%
   chance of guessing a held-out byte (typically 20-45% exact-byte accuracy).
   So exposing the bits in the input is what makes the rule inducible at all.

2. The original hypothesis -- "bit-local XOR/NOT generalize, carry-coupled INC
   collapses" -- is NOT supported. INC generalizes about as well as XOR/NOT.
   The relation table shows why: the engine's rule grammar is far richer than
   per-bit equality. It combines per-position bit tests (A[3]=0) with
   INTER-position relations (A[i] < A[j]) and arithmetic effects
   (fE[B] := inc S[B], dec S[B], S[i] xor S[j]). It even has a native `inc`
   operator, which is exactly why the carry-coupled increment is learnable.
   The flip side: those extra degrees of freedom let it bolt spurious
   conditions onto otherwise-correct rules, so it overfits and held-out
   accuracy plateaus well below 100% rather than snapping to a clean rule.

The honest lesson for a discovery library: the symbolic engine can induce
genuinely generalizing rules, but (a) only when the input encoding exposes the
relevant structure, and (b) its expressive rule grammar means the rules it
finds are richer and messier than the minimal human description of the target
function. We print the BITS+XOR relation table so this can be seen directly.

Reproducible (fixed RandSeed), single CPU, no external data.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}
uses SysUtils, neuralbyteprediction;

const
  NumInputs   = 256;
  Epochs      = 80;
  NeuronGroups= 256;   // generous budget: enough to memorize all train patterns
  Searches    = 40;

type
  TFuncId = (fnXOR, fnNOT, fnINC);
  TEncoding = (encByte, encBits);

function ApplyFunc(fn: TFuncId; inByte: byte): byte;
begin
  case fn of
    fnXOR: Result := inByte xor $5A;
    fnNOT: Result := byte(not inByte);
    fnINC: Result := byte((inByte + 1) and $FF);
  else
    Result := inByte;
  end;
end;

function FuncName(fn: TFuncId): string;
begin
  case fn of
    fnXOR: Result := 'XOR $5A (bit-local)';
    fnNOT: Result := 'NOT     (bit-local)';
    fnINC: Result := 'INC +1  (carry-coupled)';
  else
    Result := '?';
  end;
end;

function BinStr(value: byte): string;
var i: integer;
begin
  Result := '';
  for i := 7 downto 0 do
    if (value shr i) and 1 = 1 then Result := Result + '1' else Result := Result + '0';
end;

// Deterministic 50/50 split: IsTrain[i] flags the training half. Seeded so the
// split is identical across every cell and across runs.
procedure BuildSplit(out IsTrain: array of boolean);
var i, j, t: integer; order: array[0..NumInputs-1] of integer;
begin
  RandSeed := 424242;
  for i := 0 to NumInputs-1 do order[i] := i;
  for i := NumInputs-1 downto 1 do          // Fisher-Yates shuffle
  begin
    j := Random(i+1);
    t := order[i]; order[i] := order[j]; order[j] := t;
  end;
  for i := 0 to NumInputs-1 do IsTrain[i] := False;
  for i := 0 to (NumInputs div 2)-1 do IsTrain[order[i]] := True;
end;

function EncName(enc: TEncoding): string;
begin
  if enc = encByte then Result := 'BYTE' else Result := 'BITS';
end;

// Encodes input byte into the engine's action/state arrays for the chosen
// encoding (1 byte vs 8 bit-positions). Same vector is used as action & state.
procedure Encode(enc: TEncoding; value: byte; var arr: array of byte);
var j: integer;
begin
  if enc = encByte then
    arr[0] := value
  else
    for j := 0 to 7 do arr[j] := (value shr j) and 1;
end;

// Reconstructs the predicted byte from the engine's predicted-state vector.
function Decode(enc: TEncoding; const arr: array of byte): byte;
var j: integer;
begin
  if enc = encByte then
    Result := arr[0]
  else
  begin
    Result := 0;
    for j := 0 to 7 do
      if (arr[j] and 1) = 1 then Result := Result or (1 shl j);
  end;
end;

// Trains a fresh engine on the train half only, then returns train/test
// accuracy. printTable dumps the learned relations for the trained engine.
procedure RunCell(enc: TEncoding; fn: TFuncId; generalize: boolean;
  const IsTrain: array of boolean;
  out trainAcc, testAcc: double; printTable: boolean);
var
  Engine: TEasyLearnAndPredictClass;
  ep, i, k, t, width: integer;
  trainOrder: array[0..NumInputs-1] of integer;
  nTrain, trainOK, testOK, nTest: integer;
  aAct, aState, aPred, aTgt: array of byte;
begin
  // TEasyLearnAndPredictClass is an old-style value `object`; FPC does not
  // auto-zero its managed fields on routine entry, so on the 2nd+ call this
  // stack slot holds garbage and Initiate's SetLength faults. Zero it first.
  FillChar(Engine, SizeOf(Engine), 0);

  if enc = encByte then width := 1 else width := 8;
  SetLength(aAct, width); SetLength(aState, width);
  SetLength(aPred, width); SetLength(aTgt, width);

  // BYTE: includeZeros=False (whole-byte equality conditions). BITS: a 0-bit
  // is informative, so includeZeros=True lets "A[j] = 0" conditions form.
  Engine.Initiate(width, width, enc = encBits, NeuronGroups, Searches, False, 2000);
  Engine.BytePred.FUseBelief := True;
  Engine.BytePred.FGeneralize := generalize;

  // collect the train indices into a list we can shuffle each epoch
  nTrain := 0;
  for i := 0 to NumInputs-1 do
    if IsTrain[i] then begin trainOrder[nTrain] := i; Inc(nTrain); end;

  RandSeed := 1234;  // training-shuffle stream, separate from the split stream
  for ep := 0 to Epochs-1 do
  begin
    for i := nTrain-1 downto 1 do
    begin
      k := Random(i+1);
      t := trainOrder[i]; trainOrder[i] := trainOrder[k]; trainOrder[k] := t;
    end;
    for i := 0 to nTrain-1 do
    begin
      Encode(enc, byte(trainOrder[i]), aAct);
      Encode(enc, byte(trainOrder[i]), aState);
      Engine.Predict(aAct, aState, aPred);
      Encode(enc, ApplyFunc(fn, byte(trainOrder[i])), aTgt);
      Engine.newStateFound(aTgt);
    end;
  end;

  trainOK := 0; testOK := 0; nTest := 0;
  for i := 0 to NumInputs-1 do
  begin
    Encode(enc, byte(i), aAct);
    Encode(enc, byte(i), aState);
    Engine.Predict(aAct, aState, aPred);
    if Decode(enc, aPred) = ApplyFunc(fn, byte(i)) then
    begin
      if IsTrain[i] then Inc(trainOK) else Inc(testOK);
    end;
    if not IsTrain[i] then Inc(nTest);
  end;
  trainAcc := trainOK / nTrain;
  testAcc  := testOK  / nTest;

  if printTable then
  begin
    WriteLn;
    WriteLn('  --- relation table: ', EncName(enc), ' + ', FuncName(fn),
            '  FGeneralize=', BoolToStr(generalize, True), ' ---');
    WriteLn('  (A[j]/S[j] = bit j of the input; target bit j = in_bit_j xor mask_j,');
    WriteLn('   mask $5A = 01011010, so bits 1,3,4,6 flip and bits 0,2,5,7 pass)');
    Engine.printRelationTable;
  end;

  Engine.DeInitiate;
end;

var
  IsTrain: array[0..NumInputs-1] of boolean;
  enc: TEncoding;
  fn: TFuncId;
  g: boolean;
  trA, teA: double;
begin
  WriteLn('ByteRuleInduction: rule induction vs memorization in the symbolic byte engine');
  WriteLn('=============================================================================');
  WriteLn('256 single-byte inputs, deterministic 50/50 train/test split, ', Epochs, ' epochs.');
  WriteLn('train acc = fit on seen inputs; test acc = generalization to held-out inputs.');
  WriteLn('Chance (guessing a held-out byte) = 1/256 = 0.4%, so any sizable test acc');
  WriteLn('is real generalization, not luck. A memorizer scores ~0% on held-out.');
  WriteLn;

  BuildSplit(IsTrain);

  WriteLn(Format('%-6s %-26s %-12s %10s %10s',
    ['enc', 'function', 'FGeneralize', 'train acc', 'test acc']));
  WriteLn('---------------------------------------------------------------------------');
  for enc := encByte to encBits do
  begin
    for fn := fnXOR to fnINC do
      for g := False to True do
      begin
        RunCell(enc, fn, g, IsTrain, trA, teA, False);
        WriteLn(Format('%-6s %-26s %-12s %9.1f%% %9.1f%%',
          [EncName(enc), FuncName(fn), BoolToStr(g, True), trA*100, teA*100]));
      end;
    WriteLn;
  end;

  // Read the learned rules for the cell where induction is possible AND the
  // table is compact enough to check by eye: BITS + XOR + generalization on.
  RunCell(encBits, fnXOR, True, IsTrain, trA, teA, True);

  WriteLn;
  WriteLn('Interpretation:');
  WriteLn('  BYTE rows: whole-byte equality relations => a lookup table. Perfect');
  WriteLn('    train fit, 0% held-out: pure memorization, no matter the function.');
  WriteLn('  BITS rows: exposing the bits unlocks generalization far above the');
  WriteLn('    0.4% chance level (~20-45% exact-byte accuracy on unseen inputs).');
  WriteLn('  But note INC generalizes about as well as XOR/NOT -- the bit-locality');
  WriteLn('    hypothesis fails. The relation table above shows why: the engine''s');
  WriteLn('    grammar mixes per-bit tests with inter-position relations and');
  WriteLn('    arithmetic (inc/dec/xor), so it can express increment directly but');
  WriteLn('    also overfits with spurious conditions (test acc plateaus < 100%).');
  WriteLn('  Lesson: the engine DOES induce generalizing rules, but only when the');
  WriteLn('  encoding exposes structure, and the rules it finds are richer and');
  WriteLn('  messier than the minimal human description of the target function.');
  WriteLn;
  WriteLn('Done.');
end.
