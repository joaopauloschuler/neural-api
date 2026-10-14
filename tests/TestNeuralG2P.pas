unit TestNeuralG2P;
(*
Tests for neuralg2p.pas, the pure-Pascal port of misaki's English G2P.

The fixture tests/fixtures/g2p_english/ (tools/make_g2p_english_fixture.py,
misaki 0.9.4 + spaCy en_core_web_sm) holds 297 sentences with misaki's
phoneme strings and tokens, for US and GB, and dictionary subsets that
reproduce misaki exactly on those sentences. TestEnglishFixtureAgreement*
runs the port on every sentence and reports token- and sentence-level
agreement per group. Excluded, because later tasks own them: the numbers,
currency and percent groups and every token with a digit (task 5.3), and
tokens misaki itself left unknown (task 5.5). Tokens whose letters are a
POS-keyed gold entry are counted separately and compared with the baseline
of reading every such entry as DEFAULT (notes.json counts).
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, fpjson,
  neuralhftokenizer, neuralg2p;

type
  TG2PAgreementCounts = record
    Tokens, TokensEqual: integer;
    POSTokens, POSTokensEqual: integer;
    Sentences, SentencesEqual: integer;
  end;

  TTestNeuralG2P = class(TTestCase)
  private
    function FixtureDir(): string;
    function CreateFixtureG2P(Accent: TNeuralG2PAccent): TNeuralEnglishG2P;
    procedure AssertPhonemes(G2P: TNeuralEnglishG2P; const Text,
      Expected: string);
    procedure RunFixtureAgreement(Accent: TNeuralG2PAccent);
  published
    // Hand-written JSON: plain, null, POS-keyed and escaped values, a
    // repeated key (the last one wins) and misaki's grown-key lookup.
    procedure TestDictionaryParseAndLookup;
    // A missing dictionary raises ENeuralG2PError naming both files.
    procedure TestMissingDictionaryError;
    procedure TestAccentFromVoice;
    procedure TestApplyStress;
    // misaki's sub-token regex: camelCase, hyphens, dots, apostrophes, digits.
    procedure TestSubTokenize;
    // -s / -ed / -ing over known stems, US vs GB endings, US flap.
    procedure TestStemRulesUS;
    procedure TestStemRulesGB;
    // Multi-piece words: the lighter primary stresses become secondary.
    procedure TestStressDemotion;
    // Acronyms and dotted abbreviations spelled from the letter entries;
    // a shouting sentence is read as words.
    procedure TestAcronymSpelling;
    // Right-to-left context: the -> ði / ðə, to -> tə / tʊ, "None" readings.
    procedure TestNextWordContext;
    // The rule tagger: tags and the readings they select.
    procedure TestPOSRules;
    // Per-token resolution report (dictionary / stem / spelled / unknown ...).
    procedure TestTokenResolutions;
    procedure TestEnglishFixtureAgreementUS;
    procedure TestEnglishFixtureAgreementGB;
    // The full misaki dictionaries (ignored when not installed): entry
    // counts against notes.json, load time and a few readings.
    procedure TestRealDictionaryLoad;
  end;

implementation

function TTestNeuralG2P.FixtureDir(): string;
begin
  Result := 'fixtures' + DirectorySeparator + 'g2p_english';
  if not DirectoryExists(Result) then
    Result := 'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'g2p_english';
  if not DirectoryExists(Result) then
    Fail('Fixture not found: g2p_english (tools/make_g2p_english_fixture.py).');
end;

function TTestNeuralG2P.CreateFixtureG2P(
  Accent: TNeuralG2PAccent): TNeuralEnglishG2P;
begin
  Result := TNeuralEnglishG2P.Create(FixtureDir(), Accent);
end;

procedure TTestNeuralG2P.AssertPhonemes(G2P: TNeuralEnglishG2P; const Text,
  Expected: string);
begin
  AssertEquals(Text, Expected, G2P.Phonemize(Text));
end;

procedure TTestNeuralG2P.TestDictionaryParseAndLookup;
var
  Dictionary: TNeuralG2PDictionary;
  Phonemes: string;
  EntryIdx: integer;
begin
  Dictionary := TNeuralG2PDictionary.Create;
  try
    Dictionary.LoadFromString('{' + LineEnding +
      '  "torch": "tˈɔɹʧ",' + LineEnding +
      '  "Polish": "pˈOlɪʃ",' + LineEnding +
      '  "polish": "pˈɑlɪʃ",' + LineEnding +
      '  "Smith": "smˈɪθ",' + LineEnding +
      '  "gone": null,' + LineEnding +
      '  "esc\"apedé": "a\\bæ",' + LineEnding +
      '  "read": {"ADJ": "ɹˈɛd", "DEFAULT": "ɹˈid", "VBD": "ɹˈɛd"},' +
      LineEnding +
      '  "IT": {"DEFAULT": "ɪt", "NOUN": null},' + LineEnding +
      '  "torch": "tˈɔɹʧ2"' + LineEnding + '}');
    AssertEquals('unique keys', 8, Dictionary.Count);
    EntryIdx := Dictionary.IndexOf('torch');
    AssertTrue('torch', EntryIdx >= 0);
    AssertTrue(Dictionary.DefaultReading(EntryIdx, Phonemes));
    AssertEquals('repeated key keeps the last value', 'tˈɔɹʧ2', Phonemes);
    AssertEquals('exact lookup is case-sensitive', -1, Dictionary.IndexOf('Torch'));
    AssertEquals('lowercase key grows its Capitalized form',
      EntryIdx, Dictionary.IndexOfGrown('Torch'));
    AssertEquals('Capitalized key grows its lowercase form',
      Dictionary.IndexOf('Smith'), Dictionary.IndexOfGrown('smith'));
    AssertEquals('the exact key wins over a grown one',
      Dictionary.IndexOf('polish'), Dictionary.IndexOfGrown('polish'));
    AssertEquals('ALL-CAPS does not grow', -1, Dictionary.IndexOfGrown('TORCH'));
    EntryIdx := Dictionary.IndexOf('gone');
    AssertTrue('null entry exists', EntryIdx >= 0);
    AssertFalse('null value', Dictionary.DefaultReading(EntryIdx, Phonemes));
    EntryIdx := Dictionary.IndexOf('esc"apedé');
    AssertTrue('escaped key', EntryIdx >= 0);
    AssertTrue(Dictionary.DefaultReading(EntryIdx, Phonemes));
    AssertEquals('escaped value', 'a\bæ', Phonemes);
    EntryIdx := Dictionary.IndexOf('read');
    AssertTrue(Dictionary.IsPOSKeyed(EntryIdx));
    AssertTrue(Dictionary.TagReading(EntryIdx, 'VBD', Phonemes));
    AssertEquals('VBD', 'ɹˈɛd', Phonemes);
    AssertTrue(Dictionary.TagReading(EntryIdx, 'VERB', Phonemes));
    AssertEquals('absent tag -> DEFAULT', 'ɹˈid', Phonemes);
    EntryIdx := Dictionary.IndexOf('IT');
    AssertFalse('NOUN: null', Dictionary.TagReading(EntryIdx, 'NOUN', Phonemes));
    AssertTrue(Dictionary.DefaultReading(EntryIdx, Phonemes));
    AssertEquals('ɪt', Phonemes);
  finally
    Dictionary.Free;
  end;
end;

procedure TTestNeuralG2P.TestMissingDictionaryError;
var
  Raised: boolean;
  Message: string;
begin
  Raised := false;
  try
    TNeuralEnglishG2P.Create('/nonexistent-g2p-dir', g2pBritish).Free;
  except
    on E: ENeuralG2PError do
    begin
      Raised := true;
      Message := E.Message;
    end;
  end;
  AssertTrue('raises', Raised);
  AssertTrue('names gold: ' + Message, Pos('gb_gold.json', Message) > 0);
  AssertTrue('names silver: ' + Message, Pos('gb_silver.json', Message) > 0);
end;

procedure TTestNeuralG2P.TestAccentFromVoice;
var
  Raised: boolean;
begin
  AssertTrue(EnglishG2PAccentFromVoice('af_heart') = g2pAmerican);
  AssertTrue(EnglishG2PAccentFromVoice('am_adam') = g2pAmerican);
  AssertTrue(EnglishG2PAccentFromVoice('bf_emma') = g2pBritish);
  Raised := false;
  try
    EnglishG2PAccentFromVoice('jf_alpha');
  except
    on E: ENeuralG2PError do Raised := true;
  end;
  AssertTrue('Japanese voice raises', Raised);
end;

procedure TTestNeuralG2P.TestApplyStress;
begin
  AssertEquals('None', 'kˈæt', ApplyG2PStress('kˈæt', csG2PNoStress));
  AssertEquals('-2 strips', 'kæt', ApplyG2PStress('kˈæt', -2));
  AssertEquals('-1 drops secondaries, demotes primaries', 'OkˌA',
    ApplyG2PStress('ˌOkˈA', -1));
  AssertEquals('-0.5 demotes a primary', 'kˌæt', ApplyG2PStress('kˈæt', -0.5));
  AssertEquals('0.5 adds a secondary before the first vowel', 'ʃˌi',
    ApplyG2PStress('ʃi', 0.5));
  AssertEquals('0.5 keeps an existing mark', 'kˈæt', ApplyG2PStress('kˈæt', 0.5));
  AssertEquals('2 promotes a secondary', 'nˈɪɹ', ApplyG2PStress('nˌɪɹ', 2));
  AssertEquals('2 adds a primary', 'ˈɪz', ApplyG2PStress('ɪz', 2));
  AssertEquals('no vowel: unchanged', 'ft', ApplyG2PStress('ft', 2));
end;

procedure TTestNeuralG2P.TestSubTokenize;
var
  Pieces: TStringList;

  procedure Check(const Word, Expected: string);
  begin
    G2PSubTokenize(Word, Pieces);
    Pieces.Delimiter := '|';
    Pieces.StrictDelimiter := true;
    AssertEquals(Word, Expected, Pieces.DelimitedText);
  end;

begin
  Pieces := TStringList.Create;
  try
    Check('PyTorch', 'Py|Torch');
    Check('TensorFlow', 'Tensor|Flow');
    Check('iPhone', 'i|Phone');
    Check('JSON', 'JSON');
    Check('mother-in-law', 'mother|-|in|-|law');
    Check('U.S.', 'U|.|S|.');
    Check('don''t', 'don''t');
    Check('''s', '''|s');
    Check('y''all', 'y''a|ll');
    Check('1,234.5', '1,234.5');
    Check('B2B', 'B|2|B');
    Check('someone@example.com', 'someone|@|example|.|com');
  finally
    Pieces.Free;
  end;
end;

procedure TTestNeuralG2P.TestStemRulesUS;
var
  G2P: TNeuralEnglishG2P;
begin
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    AssertPhonemes(G2P, 'googled', 'ɡˈuɡᵊld');
    AssertPhonemes(G2P, 'googling', 'ɡˈuɡᵊlɪŋ');
    AssertPhonemes(G2P, 'texted', 'tˈɛkstᵻd');
    AssertPhonemes(G2P, 'upvoted', 'ˈʌpvˌOTᵻd');     // US flap -> T
    AssertPhonemes(G2P, 'downvoting', 'dˈWnvˌOTɪŋ');
    AssertPhonemes(G2P, 'skyped', 'skˈIpt');
    AssertPhonemes(G2P, 'foxes', 'fˈɑksᵻz');
    AssertPhonemes(G2P, 'ponies', 'pˈOniz');
    AssertPhonemes(G2P, 'overclocking', 'ˌOvəɹklˈɑkɪŋ');
    AssertPhonemes(G2P, 'refactoring', 'ɹifˈæktəɹɪŋ');
    AssertPhonemes(G2P, 'boss''s', 'bˈɔsᵻz');
    AssertPhonemes(G2P, 'dogs''', 'dˈɔɡz');
    // one level only, no doubled-consonant -ed: misaki leaves it unknown
    AssertPhonemes(G2P, 'blogged', '');
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.TestStemRulesGB;
var
  G2P: TNeuralEnglishG2P;
begin
  G2P := CreateFixtureG2P(g2pBritish);
  try
    AssertPhonemes(G2P, 'texted', 'tˈɛkstɪd');
    AssertPhonemes(G2P, 'upvoted', 'ˈʌpvQtɪd');       // no flap in GB
    AssertPhonemes(G2P, 'foxes', 'fˈɒksɪz');
    AssertPhonemes(G2P, 'boss''s', 'bˈɒsɪz');
    AssertPhonemes(G2P, 'googling', 'ɡˈuːɡᵊlɪŋ');
    // GB -ing refuses a stem ending in ə
    AssertPhonemes(G2P, 'refactoring', '');
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.TestStressDemotion;
var
  G2P: TNeuralEnglishG2P;
begin
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    AssertPhonemes(G2P, 'well-known', 'wˌɛlnˈOn');
    AssertPhonemes(G2P, 'twenty-first', 'twˈɛntifˌɜɹst');
    AssertPhonemes(G2P, 'self-driving', 'sˌɛlfdɹˈIvɪŋ');
    AssertPhonemes(G2P, 'state-of-the-art', 'stˈAtʌvðiˈɑɹt');
    AssertPhonemes(G2P, 'Re-enter', 'ɹiˈɛntəɹ');
    // camelCase: an unknown piece vanishes, the known one stays
    AssertPhonemes(G2P, 'PyTorch', 'tˈɔɹʧ');
    AssertPhonemes(G2P, 'WiFi', '');
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.TestAcronymSpelling;
var
  G2P: TNeuralEnglishG2P;
begin
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    AssertPhonemes(G2P, 'The FBI and the CIA are agencies.',
      'ði ˌɛfbˌiˈI ænd ðə sˌiˌIˈA ɑɹ ˈAʤənsiz.');
    AssertPhonemes(G2P, 'The U.S. and the U.K. signed a treaty.',
      'ðə jˌuˈɛs ænd ðə jˌukˈA sˈInd ɐ tɹˈiTi.');
    AssertPhonemes(G2P, 'He has a Ph.D. in physics.',
      'hˌi hæz ɐ pˌiˌAʧdˈi ɪn fˈɪzɪks.');
    AssertPhonemes(G2P, 'NASA launched a new rocket.',
      'nˈæsə lˈɔnʧt ɐ nˈu ɹˈɑkət.');
    AssertPhonemes(G2P, 'The IT department fixed the PC.',
      'ði ˌItˈi dəpˈɑɹtmənt fˈɪkst ðə pˌisˈi.');
    // every word upper case: read as words with primary stress
    AssertPhonemes(G2P, 'THIS IS ALL CAPS TEXT.',
      'ðˈɪs ˈɪz ˈɔl kˈæps tˈɛkst.');
    AssertPhonemes(G2P, 'Dr. Smith met Mr. Jones and Mrs. Brown.',
      'dˈɑktəɹ smˈɪθ mˈɛt mˈɪstəɹ ʤˈOnz ænd mˈɪsɪz bɹˈWn.');
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.TestNextWordContext;
var
  G2P: TNeuralEnglishG2P;
begin
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    AssertPhonemes(G2P, 'The end of the year is near; the evening is calm.',
      'ði ˈɛnd ʌv ðə jˈɪɹ ɪz nˌɪɹ; ði ˈivnɪŋ ɪz kˈɑm.');
    AssertPhonemes(G2P, 'Go to the park, then to an office.',
      'ɡˌO tə ðə pˈɑɹk, ðˈɛn tə ɐn ˈɔfəs.');
    // "by" and "in" at a clause end take their stressed readings
    AssertPhonemes(G2P, 'Stand by me, and do not walk by.',
      'stˈænd bI mˌi, ænd dˈu nˌɑt wˈɔk bˈI.');
    AssertPhonemes(G2P, 'Please open the window and let the fresh air in.',
      'plˈiz ˈOpᵊn ðə wˈɪndO ænd lˈɛt ðə fɹˈɛʃ ˈɛɹ ˈɪn.');
    AssertPhonemes(G2P, 'I used to use this; it is used.',
      'ˌI jˈust tə jˈuz ðˈɪs; ɪt ɪz jˈuzd.');
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.TestPOSRules;
var
  G2P: TNeuralEnglishG2P;
  Tokens: TNeuralG2PTokenArray;

  procedure CheckTag(const Text: string; TokenIdx: integer;
    const ExpectedText, ExpectedTag: string);
  begin
    G2P.Phonemize(Text, Tokens);
    AssertEquals(Text + ' token', ExpectedText, Tokens[TokenIdx].Text);
    AssertEquals(Text + ' tag of ' + ExpectedText, ExpectedTag,
      Tokens[TokenIdx].Tag);
  end;

begin
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    CheckTag('I object to that object.', 1, 'object', 'VBP');
    CheckTag('I object to that object.', 3, 'that', 'DT');
    CheckTag('I object to that object.', 4, 'object', 'NN');
    CheckTag('Close the door.', 0, 'Close', 'VB');
    CheckTag('She read; they read.', 1, 'read', 'VBD');
    CheckTag('I couldn''t believe that the lead singer left.', 3, 'that', 'IN');
    CheckTag('A is the first letter.', 0, 'A', 'DT');
    CheckTag('Plan A.', 1, 'A', 'NNP');
    AssertPhonemes(G2P, 'I object to that object.',
      'ˌI əbʤˈɛkt tə ðˈæt ˈɑbʤəkt.');
    AssertPhonemes(G2P, 'She will present the present to him.',
      'ʃˌi wɪl pɹizˈɛnt ðə pɹˈɛzᵊnt tə hˌɪm.');
    AssertPhonemes(G2P, 'Please read the record before you record a new one.',
      'plˈiz ɹˈid ðə ɹˈɛkəɹd bəfˈɔɹ ju ɹəkˈɔɹd ɐ nˈu wˈʌn.');
    AssertPhonemes(G2P, 'He reads; she read; they are reading.',
      'hˌi ɹˈidz; ʃi ɹˈɛd; ðA ɑɹ ɹˈidɪŋ.');
    AssertPhonemes(G2P, 'They house people in a house.',
      'ðˌA hˈWz pˈipᵊl ɪn ɐ hˈWs.');
    AssertPhonemes(G2P, 'Excuse me, that is no excuse.',
      'ɪkskjˈuz mˌi, ðˈæt ɪz nˈO ɪkskjˈus.');
    AssertPhonemes(G2P, 'The farm will produce fresh produce.',
      'ðə fˈɑɹm wɪl pɹədˈus fɹˈɛʃ pɹˈOdˌus.');
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.TestTokenResolutions;
var
  G2P: TNeuralEnglishG2P;
  Tokens: TNeuralG2PTokenArray;
begin
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    G2P.Phonemize('They blogged, googled FBI and I read it.', Tokens);
    AssertEquals('token count', 10, Length(Tokens));
    AssertEquals('They', Tokens[0].Text);
    AssertTrue('dictionary', g2prDictionary in Tokens[0].Resolutions);
    AssertEquals('blogged', Tokens[1].Text);
    AssertFalse('blogged unknown', Tokens[1].IsKnown);
    AssertTrue('unknown flag', g2prUnknown in Tokens[1].Resolutions);
    AssertTrue('comma', g2prPunctuation in Tokens[2].Resolutions);
    AssertEquals(' ', Tokens[2].Whitespace);
    AssertTrue('googled: stem rule', g2prStemRule in Tokens[3].Resolutions);
    AssertTrue('FBI: spelled', g2prSpelled in Tokens[4].Resolutions);
    AssertTrue('and: dictionary', g2prDictionary in Tokens[5].Resolutions);
    AssertEquals('read', Tokens[7].Text);
    AssertEquals('VBP', Tokens[7].Tag);
    AssertTrue('read: POS rule', g2prPOSRule in Tokens[7].Resolutions);
    G2P.Phonemize('It costs 5 dollars in the 1990s.', Tokens);
    AssertTrue('digits wait for task 5.3',
      g2prNumberPending in Tokens[2].Resolutions);
    AssertEquals('1990s', Tokens[6].Text);
    AssertEquals('a pending number reads nothing, not its letters', '',
      Tokens[6].Phonemes);
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.RunFixtureAgreement(Accent: TNeuralG2PAccent);
const
  cGroups: array[0..13] of string = ('plain', 'punctuation',
    'capitals_acronyms', 'possessives', 'contractions', 'hyphenated',
    'suffix_stems', 'numbers', 'currency', 'percent', 'heteronyms',
    'unknown_and_names', 'long_mixed', 'numbers_more');
  // Lowest token agreement (%) each group must keep; numbers/currency/
  // percent/numbers_more are task 5.3 (-1: not counted).
  cMinTokenAgreement: array[0..13] of double = (100, 100, 99, 100, 100, 100,
    100, -1, -1, -1, 100, 100, 99, -1);
var
  Root, Notes: TJSONData;
  Sentences: TJSONArray;
  Sentence, AccentObj, RefToken, GroupCounts: TJSONObject;
  RefTokens: TJSONArray;
  G2P: TNeuralEnglishG2P;
  AccentKey, Group, Mine, RefText, RefPhonemes, MyText, Joined, RefJoined,
    Letters: string;
  Tokens: TNeuralG2PTokenArray;
  Counts: array[0..13] of TG2PAgreementCounts;
  Total: TG2PAgreementCounts;
  SentenceCnt, GroupIdx, GroupCnt, MyIdx, RefIdx, SpanEnd, MySpanEnd,
    BaselineTokens, BaselineDiffering, CharPos, MaxCharPos, MaxSentencePos,
    MaxLinePos: integer;
  IsCounted, IsPOSToken, IsEqual, IsRefUnknown: boolean;
  Lines: TStringList;

  function ReadFile(const FileName: string): string;
  var
    Stream: TFileStream;
  begin
    Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
    try
      SetLength(Result, Stream.Size);
      if Length(Result) > 0 then Stream.ReadBuffer(Result[1], Length(Result));
    finally
      Stream.Free;
    end;
  end;

  function HasDigitOrMoney(const S: string): boolean;
  var
    BytePos, MaxBytePos: integer;
  begin
    MaxBytePos := Length(S);
    for BytePos := 1 to MaxBytePos do
      if S[BytePos] in ['0'..'9', '$', '%'] then Exit(true);
    Result := (Pos('£', S) > 0) or (Pos('€', S) > 0) or (Pos('¥', S) > 0);
  end;

  function IsPOSKeyedGold(const Key: string): boolean;
  var
    GoldIndex: integer;
  begin
    GoldIndex := G2P.Gold.IndexOfGrown(Key);
    Result := (GoldIndex >= 0) and G2P.Gold.IsPOSKeyed(GoldIndex);
  end;

  function Percent(Part, Whole: integer): string;
  begin
    if Whole = 0 then Exit('  n/a');
    Result := FormatFloat('0.0', 100.0 * Part / Whole) + '%';
  end;

  procedure CountToken(var C: TG2PAgreementCounts; IsPOS, Equal: boolean);
  begin
    if IsPOS then
    begin
      Inc(C.POSTokens);
      if Equal then Inc(C.POSTokensEqual);
    end
    else
    begin
      Inc(C.Tokens);
      if Equal then Inc(C.TokensEqual);
    end;
  end;

begin
  if Accent = g2pBritish then AccentKey := 'gb' else AccentKey := 'us';
  Root := HFParseJSONRaw(ReadFile(FixtureDir() + DirectorySeparator +
    'sentences.json'));
  Notes := HFParseJSONRaw(ReadFile(FixtureDir() + DirectorySeparator +
    'notes.json'));
  G2P := CreateFixtureG2P(Accent);
  Lines := TStringList.Create;
  try
    FillChar(Counts, SizeOf(Counts), 0);
    FillChar(Total, SizeOf(Total), 0);
    Sentences := TJSONObject(Root).Arrays['sentences'];
    MaxSentencePos := Sentences.Count - 1;
    for SentenceCnt := 0 to MaxSentencePos do
    begin
      Sentence := Sentences.Objects[SentenceCnt];
      Group := Sentence.Strings['group'];
      GroupIdx := -1;
      for GroupCnt := 0 to High(cGroups) do
        if cGroups[GroupCnt] = Group then GroupIdx := GroupCnt;
      AssertTrue('known group ' + Group, GroupIdx >= 0);
      AccentObj := Sentence.Objects[AccentKey];
      RefTokens := AccentObj.Arrays['tokens'];
      Mine := G2P.Phonemize(Sentence.Strings['text'], Tokens);
      // POS-keyed tokens are counted in every group (their words have no
      // digits); everything else skips the number groups.
      Inc(Counts[GroupIdx].Sentences);
      if Mine = AccentObj.Strings['phonemes'] then
        Inc(Counts[GroupIdx].SentencesEqual);
      MyIdx := 0;
      RefIdx := 0;
      while RefIdx < RefTokens.Count do
      begin
        RefToken := RefTokens.Objects[RefIdx];
        RefText := RefToken.Strings['text'];
        IsRefUnknown := RefToken.Items[RefToken.IndexOfName('phonemes')].IsNull;
        if IsRefUnknown then RefPhonemes := ''
        else RefPhonemes := RefToken.Strings['phonemes'];
        // notes.json's rule: the token's letters, as written or lowercased,
        // name a POS-keyed gold entry
        Letters := '';
        MaxCharPos := Length(RefText);
        for CharPos := 1 to MaxCharPos do
          if RefText[CharPos] in ['A'..'Z', 'a'..'z'] then
            Letters := Letters + RefText[CharPos];
        IsPOSToken := (Letters <> '') and
          (IsPOSKeyedGold(Letters) or IsPOSKeyedGold(LowerCase(Letters)));
        IsCounted := not IsRefUnknown and (IsPOSToken or
          (not HasDigitOrMoney(RefText) and (cMinTokenAgreement[GroupIdx] >= 0)));
        if (MyIdx < Length(Tokens)) and (Tokens[MyIdx].Text = RefText) then
        begin
          IsEqual := Tokens[MyIdx].Phonemes = RefPhonemes;
          if IsCounted then
          begin
            CountToken(Counts[GroupIdx], IsPOSToken, IsEqual);
            if not IsEqual then
              Lines.Add(Format('  %s #%d %s: "%s" misaki %s, port %s',
                [AccentKey, Sentence.Integers['id'], Group, RefText,
                RefPhonemes, Tokens[MyIdx].Phonemes]));
          end;
          Inc(MyIdx);
          Inc(RefIdx);
          Continue;
        end;
        // token boundaries differ: resynchronize on the joined text
        Joined := '';
        if MyIdx < Length(Tokens) then Joined := Tokens[MyIdx].Text;
        RefJoined := RefText;
        MySpanEnd := MyIdx + 1;
        SpanEnd := RefIdx + 1;
        while (Joined <> RefJoined) and
          ((MySpanEnd < Length(Tokens)) or (SpanEnd < RefTokens.Count)) do
          if ((Length(Joined) < Length(RefJoined)) and
            (MySpanEnd < Length(Tokens))) or (SpanEnd >= RefTokens.Count) then
          begin
            Joined := Joined + Tokens[MySpanEnd].Text;
            Inc(MySpanEnd);
          end
          else
          begin
            RefJoined := RefJoined + RefTokens.Objects[SpanEnd].Strings['text'];
            Inc(SpanEnd);
          end;
        MyText := Joined;
        Lines.Add(Format('  %s #%d %s: token boundaries differ at "%s" vs "%s"',
          [AccentKey, Sentence.Integers['id'], Group, RefJoined, MyText]));
        while RefIdx < SpanEnd do
        begin
          RefToken := RefTokens.Objects[RefIdx];
          RefText := RefToken.Strings['text'];
          if not RefToken.Items[RefToken.IndexOfName('phonemes')].IsNull and
            not HasDigitOrMoney(RefText) and
            (cMinTokenAgreement[GroupIdx] >= 0) then
            CountToken(Counts[GroupIdx], false, false);
          Inc(RefIdx);
        end;
        MyIdx := MySpanEnd;
      end;
    end;
    WriteLn('G2P fixture agreement (', AccentKey, '), misaki 0.9.4 as reference:');
    WriteLn('  group               tokens equal/counted   sentences   ',
      'POS-keyed equal/counted');
    for GroupCnt := 0 to High(cGroups) do
    begin
      with Counts[GroupCnt] do
      begin
        if cMinTokenAgreement[GroupCnt] >= 0 then
          WriteLn(Format('  %-18s %4d/%-4d %6s   %3d/%-3d     %3d/%-3d',
            [cGroups[GroupCnt], TokensEqual, Tokens,
            Percent(TokensEqual, Tokens), SentencesEqual, Sentences,
            POSTokensEqual, POSTokens]))
        else
          WriteLn(Format('  %-18s (task 5.3, not counted)               %3d/%-3d',
            [cGroups[GroupCnt], POSTokensEqual, POSTokens]));
        Inc(Total.Tokens, Tokens);
        Inc(Total.TokensEqual, TokensEqual);
        Inc(Total.POSTokens, POSTokens);
        Inc(Total.POSTokensEqual, POSTokensEqual);
        if cMinTokenAgreement[GroupCnt] >= 0 then
        begin
          Inc(Total.Sentences, Sentences);
          Inc(Total.SentencesEqual, SentencesEqual);
        end;
      end;
    end;
    GroupCounts := TJSONObject(TJSONObject(TJSONObject(Notes).Objects['counts'])
      .Objects[AccentKey]).Objects['total'];
    BaselineTokens := GroupCounts.Integers['heteronym_tokens'];
    BaselineDiffering := GroupCounts.Integers['heteronym_tokens_differing'];
    WriteLn(Format('  total: tokens %d/%d (%s), sentences %d/%d (%s), ' +
      'POS-keyed %d/%d (%s; DEFAULT baseline %d/%d = %s)',
      [Total.TokensEqual, Total.Tokens, Percent(Total.TokensEqual, Total.Tokens),
      Total.SentencesEqual, Total.Sentences,
      Percent(Total.SentencesEqual, Total.Sentences), Total.POSTokensEqual,
      Total.POSTokens, Percent(Total.POSTokensEqual, Total.POSTokens),
      BaselineTokens - BaselineDiffering, BaselineTokens,
      Percent(BaselineTokens - BaselineDiffering, BaselineTokens)]));
    MaxLinePos := Lines.Count - 1;
    for GroupCnt := 0 to MaxLinePos do WriteLn(Lines[GroupCnt]);
    AssertEquals('POS-keyed token count matches notes.json', BaselineTokens,
      Total.POSTokens);
    // DEFAULT alone agrees on 70.3%; the rule tagger reaches 96.8% (153/158)
    AssertTrue('POS-keyed agreement ' +
      Percent(Total.POSTokensEqual, Total.POSTokens) + ' >= 95%',
      Total.POSTokensEqual * 100 >= Total.POSTokens * 95);
    for GroupCnt := 0 to High(cGroups) do
      if cMinTokenAgreement[GroupCnt] >= 0 then
        with Counts[GroupCnt] do
          AssertTrue(cGroups[GroupCnt] + ' token agreement ' +
            Percent(TokensEqual, Tokens),
            100.0 * TokensEqual >= cMinTokenAgreement[GroupCnt] * Tokens);
  finally
    Lines.Free;
    G2P.Free;
    Notes.Free;
    Root.Free;
  end;
end;

procedure TTestNeuralG2P.TestEnglishFixtureAgreementUS;
begin
  RunFixtureAgreement(g2pAmerican);
end;

procedure TTestNeuralG2P.TestEnglishFixtureAgreementGB;
begin
  RunFixtureAgreement(g2pBritish);
end;

procedure TTestNeuralG2P.TestRealDictionaryLoad;
const
  cExpectedCounts: array[TNeuralG2PAccent, boolean] of integer =
    ((93361, 90201), (109766, 87352)); // [accent, IsGold]
var
  DataDir: string;
  Accent: TNeuralG2PAccent;
  G2P: TNeuralEnglishG2P;
  StartTime: QWord;
begin
  DataDir := GetEnvironmentVariable('NEURAL_MISAKI_DATA_DIR');
  if DataDir = '' then
    DataDir := IncludeTrailingPathDelimiter(GetUserDir) +
      'x/lib/python3.12/site-packages/misaki/data';
  if not FileExists(IncludeTrailingPathDelimiter(DataDir) + 'us_gold.json') then
    Ignore('misaki dictionaries not found in ' + DataDir +
      ' (set NEURAL_MISAKI_DATA_DIR to misaki''s data/ folder)');
  for Accent := g2pAmerican to g2pBritish do
  begin
    StartTime := GetTickCount64;
    G2P := TNeuralEnglishG2P.Create(DataDir, Accent);
    try
      WriteLn(Format('misaki %s dictionaries: %d gold + %d silver entries, ' +
        'loaded in %d ms', [EnglishG2PDictionaryFileName(Accent, true),
        G2P.Gold.Count, G2P.Silver.Count, GetTickCount64 - StartTime]));
      AssertEquals('gold entries', cExpectedCounts[Accent, true], G2P.Gold.Count);
      AssertEquals('silver entries', cExpectedCounts[Accent, false],
        G2P.Silver.Count);
      AssertEquals('ˌI ɹˈɛd ðə bˈʊk.', G2P.Phonemize('I read the book.'));
    finally
      G2P.Free;
    end;
  end;
end;

initialization
  RegisterTest(TTestNeuralG2P);
end.
