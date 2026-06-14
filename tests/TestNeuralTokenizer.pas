unit TestNeuralTokenizer;
(*
Tests for neuraltokenizer.pas BPE-dropout subword regularization
(Provilkov et al. 2020). A tiny in-memory vocabulary of subwords of the word
"lowest" is built so the greedy merge loop in TNeuralTokenizer.TokenizeWord has
real merge choices to drop. With DropoutProb = 0 the tokenizer is bit-identical
to the deterministic greedy-longest-match path (pinned ids). With DropoutProb >
0 at a fixed reseeded RandSeed an alternative segmentation is produced (pinned
ids), and that alternative still decodes back to the original string.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralvolume, neuraltokenizer;

type
  TTestNeuralTokenizer = class(TTestCase)
  private
    // Build a deterministic in-memory tokenizer over subwords of "lowest".
    // Vocab entries are stored sorted; SaveCurrentPosition() makes each entry's
    // integer value equal its sorted index, and TokenizeWord emits index+128.
    function BuildTokenizer(): TNeuralTokenizer;
    // Collect token ids for pText into a comma-separated string for easy
    // pinning in assertions.
    function TokenizeToCsv(T: TNeuralTokenizer; const pText: string): string;
  published
    // DropoutProb = 0 must be the deterministic greedy-longest-match path.
    procedure TestDropoutZeroIsDeterministic;
    // DropoutProb > 0 at a fixed seed yields a pinned alternative segmentation.
    procedure TestDropoutAtFixedSeedAlternativeSegmentation;
    // The dropped-merge segmentation still decodes back to the original string.
    procedure TestDropoutRoundTrips;
  end;

implementation

function TTestNeuralTokenizer.BuildTokenizer(): TNeuralTokenizer;
begin
  Result := TNeuralTokenizer.Create;
  Result.Sorted := true;
  Result.Duplicates := dupIgnore;
  // Subwords of "lowest". Deliberately include both the full word and shorter
  // prefixes/suffixes so the greedy loop can pick the longest match (no
  // dropout) or shorter ones (dropout).
  Result.Add('lo');
  Result.Add('low');
  Result.Add('lowe');
  Result.Add('lowes');
  Result.Add('lowest');
  Result.Add('owe');
  Result.Add('we');
  Result.Add('wes');
  Result.Add('west');
  Result.Add('es');
  Result.Add('est');
  Result.Add('st');
  Result.SaveCurrentPosition();
end;

function TTestNeuralTokenizer.TokenizeToCsv(T: TNeuralTokenizer;
  const pText: string): string;
var
  IL: TIntegerList;
  i: integer;
begin
  IL := TIntegerList.Create;
  try
    T.Tokenize(pText, IL);
    Result := '';
    for i := 0 to IL.Count - 1 do
    begin
      if i > 0 then Result := Result + ',';
      Result := Result + IntToStr(IL[i]);
    end;
  finally
    IL.Free;
  end;
end;

procedure TTestNeuralTokenizer.TestDropoutZeroIsDeterministic;
var
  T: TNeuralTokenizer;
  Ids: string;
begin
  T := BuildTokenizer();
  try
    AssertEquals('DropoutProb defaults to OFF', 0.0, T.DropoutProb, 0.0);
    // Greedy longest match: "lowest" is a single vocab token (id 134).
    Ids := TokenizeToCsv(T, 'lowest');
    AssertEquals('deterministic segmentation', '134', Ids);
  finally
    T.Free;
  end;
end;

procedure TTestNeuralTokenizer.TestDropoutAtFixedSeedAlternativeSegmentation;
var
  T: TNeuralTokenizer;
  Ids: string;
begin
  T := BuildTokenizer();
  try
    T.DropoutProb := 0.5;
    RandSeed := 424242;
    // Alternative segmentation: 'l','o','w' (raw chars 108,111,119), then the
    // merge 'es' (id 128), then 't' (116) - instead of the single 'lowest'
    // token (134) produced deterministically. Pinned from this implementation.
    Ids := TokenizeToCsv(T, 'lowest');
    AssertEquals('dropout alternative segmentation', '108,111,119,128,116', Ids);
  finally
    T.Free;
  end;
end;

procedure TTestNeuralTokenizer.TestDropoutRoundTrips;
var
  T: TNeuralTokenizer;
  IL: TIntegerList;
  Decoded: string;
begin
  T := BuildTokenizer();
  IL := TIntegerList.Create;
  try
    T.DropoutProb := 0.5;
    RandSeed := 424242;
    T.Tokenize('lowest', IL);
    Decoded := T.DeTokenize(IL);
    AssertEquals('dropout segmentation round-trips', 'lowest', Decoded);
  finally
    IL.Free;
    T.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralTokenizer);
end.
