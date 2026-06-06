program ByteProcessingRelationTable;
(*
ByteProcessingRelationTable: a tiny, readable look inside the symbolic byte
engine (TEasyLearnAndPredictClass) that powers TNNetByteProcessing.

The engine is NOT a gradient network -- it induces discrete cause->effect
rules ("relations") that map an input byte pattern to an output byte. This
demo drives it exactly the way TNNetByteProcessing does internally
(Predict() followed by newStateFound()) on a small, hand-crafted mapping of
four 1-byte "codewords" to four 1-byte targets, then calls
printRelationTable to print the rules it learned.

Mapping (codeword -> target):

  00000001 -> 11110000      00000010 -> 00001111
  00000100 -> 10101010      00001000 -> 01010101

Each printed rule reads as:

  B=0 <condition on input bits> => fE[B] := <output byte> [ f=freq Vit=wins n=samples ]

  - B=0                  output byte position (here a single byte)
  - (1 = A[0])           condition: input action byte A[0] matches this pattern
                         (S[0] is the state byte; the layer feeds the same
                         bits as both action and state, hence A[0]/S[0] twins)
  - => fE[B] := 240      THEN set the output byte to 240 (= 11110000)
  - f=1                  confidence (correct / total) for this rule
  - Vit                  number of times this rule won the prediction
  - n                    samples seen

A small neuron-group budget and FGeneralize := False keep the table tiny so
the four winning rules (one per class, f=1, Vit~298) are easy to read. With a
distinct codeword per class the engine reaches f=1.0 on every rule; if two
classes were forced to share a codeword with different targets, the contended
rule would instead get stuck near f=0.5 -- the symbolic signature of an
unresolvable collision.

Designed to finish instantly on a single CPU.

Coded by Claude (AI).
*)
{$mode objfpc}{$H+}
uses SysUtils, neuralbyteprediction;

const
  NumClasses = 4;
  Epochs     = 300;

var
  Engine: TEasyLearnAndPredictClass;
  Codeword: array[0..NumClasses-1] of byte = ($01, $02, $04, $08); // 0001 0010 0100 1000
  Target:   array[0..NumClasses-1] of byte = ($F0, $0F, $AA, $55); // 11110000 00001111 10101010 01010101

function BinStr(value: byte): string;
var i: integer;
begin
  Result := '';
  for i := 7 downto 0 do
    if (value shr i) and 1 = 1 then Result := Result + '1' else Result := Result + '0';
end;

var
  ep, c: integer;
  aAct, aState, aPred: array[0..0] of byte;
begin
  WriteLn('Symbolic byte engine: codeword -> target, then the relation table');
  WriteLn('-------------------------------------------------------------------');

  // A small neuron-group budget and FGeneralize := False keep the relation
  // table tiny and easy to read (no ~5% random rule cloning).
  Engine.Initiate(1 {action bytes}, 1 {state bytes}, False {includeZeros},
                  16 {neuron groups}, 40 {searches}, False {cache}, 1000);
  Engine.BytePred.FUseBelief := True;
  Engine.BytePred.FGeneralize := False;

  for ep := 0 to Epochs-1 do
    for c := 0 to NumClasses-1 do
    begin
      aAct[0]   := Codeword[c];
      aState[0] := Codeword[c];
      Engine.Predict(aAct, aState, aPred);
      Engine.newStateFound(Target[c]);
    end;

  WriteLn;
  WriteLn('Final predictions:');
  for c := 0 to NumClasses-1 do
  begin
    aAct[0]   := Codeword[c];
    aState[0] := Codeword[c];
    Engine.Predict(aAct, aState, aPred);
    WriteLn(Format('  codeword %s  -> predicted %s   (target %s)  %s',
      [BinStr(Codeword[c]), BinStr(aPred[0]), BinStr(Target[c]),
       BoolToStr(aPred[0] = Target[c], 'OK', 'x')]));
  end;

  WriteLn;
  WriteLn('Relation table (printRelationTable):');
  WriteLn('  format: B=<out byte> <condition> => fE[B] := <output byte> [ f=freq Vit=wins n=samples ]');
  WriteLn;
  Engine.printRelationTable;

  Engine.DeInitiate;
  WriteLn;
  WriteLn('Done.');
end.
