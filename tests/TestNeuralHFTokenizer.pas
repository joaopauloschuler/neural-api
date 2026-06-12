unit TestNeuralHFTokenizer;
// Tests for the HuggingFace tokenizer.json loader (neuralhftokenizer.pas).
//
// The flagship tests are the parity batteries: tests/fixtures contains two
// tiny-but-realistic tokenizer.json files (a GPT-2-style byte-level BPE and
// a Llama-style metaspace BPE with byte fallback) plus
// hf_tokenizer_cases.json, all generated against the real HuggingFace
// `tokenizers` library by tools/hf_tokenizer_fixture.py. For every pinned
// case the Pascal Encode must reproduce HF's ids EXACTLY and Decode must
// reproduce HF's decoded string EXACTLY (ASCII, punctuation, contractions,
// multi-space, tabs/newlines, UTF-8 accents, emoji and special tokens are
// all covered).
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, fpjson, jsonparser,
  neuralvolume, neuralhftokenizer;

type
  TTestNeuralHFTokenizer = class(TTestCase)
  private
    function FixturePath(const FileName: string): string;
    procedure RunParityBattery(const GroupName: string);
  published
    procedure TestByteLevelParityWithHF;
    procedure TestMetaspaceParityWithHF;
    procedure TestByteLevelSpecialTokenIds;
    procedure TestMetaspaceSpecialTokenIds;
    procedure TestDecodeKeepsSpecialsWhenAsked;
    procedure TestRejectsNonBPEModel;
    procedure TestEncodeIntegerArrayHelper;
  end;

implementation

function TTestNeuralHFTokenizer.FixturePath(const FileName: string): string;
begin
  Result := 'fixtures' + DirectorySeparator + FileName;
  if not FileExists(Result) then
    Result := 'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      FileName;
  if not FileExists(Result) then
    Fail('Fixture not found: ' + FileName +
      ' (run python3 tools/hf_tokenizer_fixture.py from the repo root).');
end;

// Loads the tokenizer named by hf_tokenizer_cases.json[GroupName] and
// asserts exact Encode-id and Decode-string parity for every pinned case.
procedure TTestNeuralHFTokenizer.RunParityBattery(const GroupName: string);
var
  Tok: TNeuralHFTokenizer;
  Root: TJSONData;
  Group, CaseObj: TJSONObject;
  Cases, ExpectedIds: TJSONArray;
  Ids: TIntegerList;
  CaseCnt, IdCnt: integer;
  Text, Context: string;
  FS: TFileStream;
begin
  FS := TFileStream.Create(FixturePath('hf_tokenizer_cases.json'),
    fmOpenRead or fmShareDenyWrite);
  try
    Root := GetJSON(FS);
  finally
    FS.Free;
  end;
  Tok := TNeuralHFTokenizer.Create();
  Ids := TIntegerList.Create();
  try
    Group := TJSONObject(Root).Objects[GroupName];
    Tok.LoadFromFile(FixturePath(Group.Get('tokenizer', '')));
    AssertTrue(GroupName + ': vocab not loaded', Tok.GetVocabSize() > 100);
    Cases := Group.Arrays['cases'];
    AssertTrue(GroupName + ': no cases pinned', Cases.Count >= 10);
    for CaseCnt := 0 to Cases.Count - 1 do
    begin
      CaseObj := Cases.Objects[CaseCnt];
      Text := CaseObj.Get('text', '');
      ExpectedIds := CaseObj.Arrays['ids'];
      Context := GroupName + ' case ' + IntToStr(CaseCnt) + ' "' + Text + '"';
      Ids.Clear;
      Tok.Encode(Text, Ids);
      AssertEquals(Context + ': id count', ExpectedIds.Count, Ids.Count);
      for IdCnt := 0 to ExpectedIds.Count - 1 do
        AssertEquals(Context + ': id[' + IntToStr(IdCnt) + ']',
          ExpectedIds.Integers[IdCnt], Ids[IdCnt]);
      AssertEquals(Context + ': decode',
        CaseObj.Get('decoded', ''), Tok.Decode(Ids, true));
    end;
  finally
    Ids.Free;
    Tok.Free;
    Root.Free;
  end;
end;

procedure TTestNeuralHFTokenizer.TestByteLevelParityWithHF;
begin
  RunParityBattery('byte_level');
end;

procedure TTestNeuralHFTokenizer.TestMetaspaceParityWithHF;
begin
  RunParityBattery('metaspace');
end;

procedure TTestNeuralHFTokenizer.TestByteLevelSpecialTokenIds;
var
  Tok: TNeuralHFTokenizer;
begin
  Tok := TNeuralHFTokenizer.Create();
  try
    Tok.LoadFromFile(FixturePath('tiny_bpe_bytelevel_tokenizer.json'));
    AssertTrue('byte-level flag', Tok.IsByteLevel);
    AssertEquals('eos = <|endoftext|>', 0, Tok.EosId);
    AssertEquals('bos doubles as eos in the GPT-2 family', 0, Tok.BosId);
    AssertEquals('no unk in byte-level BPE', -1, Tok.UnkId);
    AssertEquals('added token lookup', 0, Tok.TokenToId('<|endoftext|>'));
  finally
    Tok.Free;
  end;
end;

procedure TTestNeuralHFTokenizer.TestMetaspaceSpecialTokenIds;
var
  Tok: TNeuralHFTokenizer;
begin
  Tok := TNeuralHFTokenizer.Create();
  try
    Tok.LoadFromFile(FixturePath('tiny_bpe_metaspace_tokenizer.json'));
    AssertTrue('not byte-level', not Tok.IsByteLevel);
    AssertEquals('unk', 0, Tok.UnkId);
    AssertEquals('bos = <s>', 1, Tok.BosId);
    AssertEquals('eos = </s>', 2, Tok.EosId);
    AssertEquals('byte token in model vocab', 3, Tok.TokenToId('<0x00>'));
    AssertEquals('id->token round trip', '<0x00>', Tok.IdToToken(3));
  finally
    Tok.Free;
  end;
end;

procedure TTestNeuralHFTokenizer.TestDecodeKeepsSpecialsWhenAsked;
var
  Tok: TNeuralHFTokenizer;
  Ids: TIntegerList;
begin
  Tok := TNeuralHFTokenizer.Create();
  Ids := TIntegerList.Create();
  try
    Tok.LoadFromFile(FixturePath('tiny_bpe_metaspace_tokenizer.json'));
    Tok.Encode('<s>hello</s>', Ids);
    AssertTrue('specials matched in text', Ids.Count >= 3);
    AssertEquals('first id is BOS', Tok.BosId, Ids[0]);
    AssertEquals('last id is EOS', Tok.EosId, Ids[Ids.Count - 1]);
    AssertEquals('skip specials', 'hello', Tok.Decode(Ids, true));
    AssertEquals('keep specials', '<s> hello</s>', Tok.Decode(Ids, false));
  finally
    Ids.Free;
    Tok.Free;
  end;
end;

procedure TTestNeuralHFTokenizer.TestRejectsNonBPEModel;
var
  Tok: TNeuralHFTokenizer;
  TempFile: string;
  SL: TStringList;
  Raised: boolean;
begin
  TempFile := GetTempDir(true) + 'not_bpe_tokenizer.json';
  SL := TStringList.Create();
  Tok := TNeuralHFTokenizer.Create();
  try
    SL.Text := '{"model": {"type": "Unigram", "vocab": []}}';
    SL.SaveToFile(TempFile);
    Raised := false;
    try
      Tok.LoadFromFile(TempFile);
    except
      on E: EHFTokenizerError do Raised := true;
    end;
    AssertTrue('Unigram must raise EHFTokenizerError', Raised);
  finally
    Tok.Free;
    SL.Free;
    DeleteFile(TempFile);
  end;
end;

procedure TTestNeuralHFTokenizer.TestEncodeIntegerArrayHelper;
var
  Tok: TNeuralHFTokenizer;
  Arr: TNeuralIntegerArray;
begin
  Tok := TNeuralHFTokenizer.Create();
  try
    Tok.LoadFromFile(FixturePath('tiny_bpe_bytelevel_tokenizer.json'));
    Arr := Tok.Encode('the cat');
    AssertTrue('array helper returns ids', Length(Arr) > 0);
    AssertEquals('decode round-trips the array helper',
      'the cat', Tok.Decode(Arr, true));
  finally
    Tok.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralHFTokenizer);
end.
