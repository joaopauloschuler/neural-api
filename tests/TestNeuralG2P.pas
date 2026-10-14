unit TestNeuralG2P;
(*
Tests for neuralg2p.pas, the pure-Pascal port of misaki's English G2P.

The fixture tests/fixtures/g2p_english/ (tools/make_g2p_english_fixture.py,
misaki 0.9.4 + spaCy en_core_web_sm) holds 320 sentences with misaki's
phoneme strings and tokens, for US and GB, and dictionary subsets that
reproduce misaki exactly on those sentences. TestEnglishFixtureAgreement*
runs the port on every sentence and reports token- and sentence-level
agreement per group. Excluded, because task 5.5 owns them: tokens misaki
itself left unknown. Tokens whose letters are a POS-keyed gold entry are
counted separately and compared with the baseline of reading every such
entry as DEFAULT (notes.json counts).
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
    function ReadFixtureJSON(const FileName: string): TJSONData;
    // misaki's data/ folder ('' when missing; NEURAL_MISAKI_DATA_DIR overrides)
    function RealDictionaryDir(): string;
    // One row per fixture token left unknown or read by an unknown-word
    // route: "text pieces phonemes", first occurrence only.
    procedure CollectUnknownWordRows(G2P: TNeuralEnglishG2P; Rows: TStrings);
    // The long_mixed sentences against KPipeline.en_tokenize's chunks.
    procedure RunChunkFixture(Accent: TNeuralG2PAccent);
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
    // num2words 0.5.14 English: cardinal, ordinal, year, decimal, negatives,
    // the illion names and the input errors.
    procedure TestEnglishNumberWords;
    // Number readings the fixture does not pin: currency on the CD word
    // after the amount, a number too large for words, the yen sign.
    procedure TestNumberReading;
    // Tasklist 5.5: words misaki leaves unknown are split into dictionary
    // words or spelled; pieces, stress pattern and resolution kinds.
    procedure TestUnknownWords;
    // Every fixture word misaki left unknown, word -> pieces -> phonemes with
    // the dictionary subset (US, GB); printed for the tasklist.
    procedure TestUnknownWordReport;
    // The same rows with misaki's full dictionaries (ignored when missing):
    // equal to the subset rows except for words with accents folded.
    procedure TestUnknownWordRealDictionaries;
    // KPipeline.en_tokenize on hand-made tokens: sentence cut, the comma
    // fallback, the ')' bump, a word over the limit, truncation, paragraphs.
    procedure TestChunkSplitRules;
    // misaki's own tokens give the fixture's chunks exactly; the port's
    // tokens give the same chunk texts.
    procedure TestChunksMatchKPipelineUS;
    procedure TestChunksMatchKPipelineGB;
    procedure TestEnglishFixtureAgreementUS;
    procedure TestEnglishFixtureAgreementGB;
    // The full misaki dictionaries (ignored when not installed): entry
    // counts against notes.json, load time and a few readings.
    procedure TestRealDictionaryLoad;
  end;

implementation

const
  cUnknownRoutes: TNeuralG2PResolutions = [g2prUnknownSplit, g2prUnknownSpelled];

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

function TTestNeuralG2P.ReadFixtureJSON(const FileName: string): TJSONData;
var
  Stream: TFileStream;
  Text: string;
begin
  Stream := TFileStream.Create(FixtureDir() + DirectorySeparator + FileName,
    fmOpenRead or fmShareDenyWrite);
  try
    SetLength(Text, Stream.Size);
    if Length(Text) > 0 then Stream.ReadBuffer(Text[1], Length(Text));
  finally
    Stream.Free;
  end;
  Result := HFParseJSONRaw(Text);
end;

function TTestNeuralG2P.RealDictionaryDir(): string;
begin
  Result := GetEnvironmentVariable('NEURAL_MISAKI_DATA_DIR');
  if Result = '' then
    Result := IncludeTrailingPathDelimiter(GetUserDir) +
      'x/lib/python3.12/site-packages/misaki/data';
  if not FileExists(IncludeTrailingPathDelimiter(Result) + 'us_gold.json') then
    Result := '';
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
    // doubled consonant before -ed (not in misaki): blog + ed, not B + logged
    AssertPhonemes(G2P, 'blogged', 'blˈɑɡd');
    AssertPhonemes(G2P, 'vlogged', 'vlˈɑɡd');
    AssertPhonemes(G2P, 'hashtagged', 'hˈæʃtˌæɡd');
    AssertPhonemes(G2P, 'geotagged', 'ʤˈiOtˌæɡd');
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
    AssertPhonemes(G2P, 'blogged', 'blˈɒɡd');
    AssertPhonemes(G2P, 'geotagged', 'ʤˈiːQtaɡd');
    // GB -ing refuses a stem ending in ə; the unknown-word split reads
    // refactor + ing
    AssertPhonemes(G2P, 'refactoring', 'ɹiːfˈaktəˌɪŋ');
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
    G2P.Phonemize('They Wrocław, googled FBI and I read it.', Tokens);
    AssertEquals('token count', 10, Length(Tokens));
    AssertEquals('They', Tokens[0].Text);
    AssertTrue('dictionary', g2prDictionary in Tokens[0].Resolutions);
    AssertEquals('Wrocław', Tokens[1].Text);
    AssertFalse('Wrocław unknown', Tokens[1].IsKnown);
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
    AssertTrue('5: number', g2prNumber in Tokens[2].Resolutions);
    AssertEquals('5', 'fˈIv', Tokens[2].Phonemes);
    AssertEquals('5: tagged CD', 'CD', Tokens[2].Tag);
    AssertEquals('1990s', Tokens[6].Text);
    AssertTrue('1990s: number', g2prNumber in Tokens[6].Resolutions);
    AssertEquals('1990s: a year with the -s ending', 'nˌIntˈin nˈIndiz',
      Tokens[6].Phonemes);
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.TestEnglishNumberWords;

  procedure CheckRaises(const Kind, Input: string);
  var
    Raised: boolean;
  begin
    Raised := false;
    try
      if Kind = 'cardinal' then EnglishCardinalWords(Input)
      else if Kind = 'ordinal' then EnglishOrdinalWords(Input)
      else EnglishDecimalWords(Input);
    except
      on ENeuralG2PError do Raised := true;
    end;
    AssertTrue(Kind + ' "' + Input + '" raises', Raised);
  end;

const
  cCardinals: array[0..36, 0..1] of string = (('0', 'zero'), ('7', 'seven'),
    ('11', 'eleven'), ('12', 'twelve'), ('13', 'thirteen'),
    ('15', 'fifteen'), ('19', 'nineteen'), ('20', 'twenty'),
    ('21', 'twenty-one'), ('42', 'forty-two'), ('99', 'ninety-nine'),
    ('100', 'one hundred'), ('101', 'one hundred and one'),
    ('110', 'one hundred and ten'), ('123', 'one hundred and twenty-three'),
    ('999', 'nine hundred and ninety-nine'), ('1000', 'one thousand'),
    ('1001', 'one thousand and one'), ('1100', 'one thousand, one hundred'),
    ('1234', 'one thousand, two hundred and thirty-four'),
    ('2001', 'two thousand and one'),
    ('12345', 'twelve thousand, three hundred and forty-five'),
    ('123456', 'one hundred and twenty-three thousand, four hundred and ' +
    'fifty-six'), ('1000000', 'one million'), ('1000001', 'one million and one'),
    ('1001000', 'one million, one thousand'),
    ('1234567', 'one million, two hundred and thirty-four thousand, five ' +
    'hundred and sixty-seven'),
    ('2500000000', 'two billion, five hundred million'),
    ('1000000000000', 'one trillion'),
    ('1000000000000000000', 'one quintillion'),
    ('999999999999', 'nine hundred and ninety-nine billion, nine hundred and ' +
    'ninety-nine million, nine hundred and ninety-nine thousand, nine ' +
    'hundred and ninety-nine'),
    ('007', 'seven'), ('-5', 'minus five'),
    ('-1234', 'minus one thousand, two hundred and thirty-four'),
    ('-0', 'zero'), ('1000000000000000000000000000000', 'one nonillion'),
    ('1000000000000000000000000000000000', 'one decillion'));
  cOrdinals: array[0..22, 0..1] of string = (('0', 'zeroth'), ('1', 'first'),
    ('2', 'second'), ('3', 'third'), ('4', 'fourth'), ('5', 'fifth'),
    ('8', 'eighth'), ('9', 'ninth'), ('11', 'eleventh'), ('12', 'twelfth'),
    ('13', 'thirteenth'), ('20', 'twentieth'), ('21', 'twenty-first'),
    ('22', 'twenty-second'), ('23', 'twenty-third'), ('40', 'fortieth'),
    ('99', 'ninety-ninth'), ('100', 'one hundredth'),
    ('101', 'one hundred and first'), ('111', 'one hundred and eleventh'),
    ('1000', 'one thousandth'), ('1000000', 'one millionth'),
    ('1234', 'one thousand, two hundred and thirty-fourth'));
  cYears: array[0..20, 0..1] of string = (('0', 'zero'), ('5', 'five'),
    ('99', 'ninety-nine'), ('100', 'one hundred'), ('105', 'one oh-five'),
    ('1000', 'one thousand'), ('1066', 'ten sixty-six'),
    ('1100', 'eleven hundred'), ('1900', 'nineteen hundred'),
    ('1905', 'nineteen oh-five'), ('1999', 'nineteen ninety-nine'),
    ('2000', 'two thousand'), ('2001', 'two thousand and one'),
    ('2009', 'two thousand and nine'), ('2010', 'twenty ten'),
    ('2024', 'twenty twenty-four'), ('10000', 'ten thousand'),
    ('12345', 'twelve thousand, three hundred and forty-five'),
    ('0123', 'one twenty-three'), ('-44', 'forty-four BC'),
    ('-1066', 'ten sixty-six BC'));
  cDecimals: array[0..15, 0..1] of string = (('3.14', 'three point one four'),
    ('0.5', 'zero point five'), ('0.75', 'zero point seven five'),
    ('26.2', 'twenty-six point two'), ('99.9', 'ninety-nine point nine'),
    ('123.456', 'one hundred and twenty-three point four five six'),
    ('2.0', 'two'), ('0.05', 'zero point zero five'),
    ('1.50', 'one point five'), ('10.01', 'ten point zero one'),
    ('0.001', 'zero point zero zero one'), ('-3.5', 'minus three point five'),
    ('-0.5', 'zero point five'), ('.5', 'zero point five'), ('5.', 'five'),
    ('42', 'forty-two'));
var
  CaseCnt: integer;
  Centillion: string;
begin
  for CaseCnt := 0 to High(cCardinals) do
    AssertEquals('cardinal ' + cCardinals[CaseCnt, 0], cCardinals[CaseCnt, 1],
      EnglishCardinalWords(cCardinals[CaseCnt, 0]));
  for CaseCnt := 0 to High(cOrdinals) do
    AssertEquals('ordinal ' + cOrdinals[CaseCnt, 0], cOrdinals[CaseCnt, 1],
      EnglishOrdinalWords(cOrdinals[CaseCnt, 0]));
  for CaseCnt := 0 to High(cYears) do
    AssertEquals('year ' + cYears[CaseCnt, 0], cYears[CaseCnt, 1],
      EnglishYearWords(cYears[CaseCnt, 0]));
  for CaseCnt := 0 to High(cDecimals) do
    AssertEquals('decimal ' + cDecimals[CaseCnt, 0], cDecimals[CaseCnt, 1],
      EnglishDecimalWords(cDecimals[CaseCnt, 0]));
  // num2words' last name (10^303) and its limit (10^306)
  Centillion := '1' + StringOfChar('0', 303);
  AssertEquals('10^303', 'one centillion', EnglishCardinalWords(Centillion));
  AssertEquals('10^36', 'one undecillion',
    EnglishCardinalWords('1' + StringOfChar('0', 36)));
  AssertEquals('10^300', 'one novemnonagintillion',
    EnglishCardinalWords('1' + StringOfChar('0', 300)));
  AssertEquals('10^306 - 1', 'nine hundred and ninety-nine centillion,',
    Copy(EnglishCardinalWords(StringOfChar('9', 306)), 1, 40));
  CheckRaises('cardinal', '1' + StringOfChar('0', 306));
  CheckRaises('cardinal', '');
  CheckRaises('cardinal', '12a');
  CheckRaises('cardinal', '1.5');
  CheckRaises('ordinal', '-1');
  CheckRaises('decimal', '1.2.3');
  CheckRaises('decimal', '.');
end;

procedure TTestNeuralG2P.TestNumberReading;
var
  G2P: TNeuralEnglishG2P;
  Tokens: TNeuralG2PTokenArray;
  Phonemes: string;
begin
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    G2P.Phonemize('They owe us $12 million.', Tokens);
    AssertEquals('token count', 7, Length(Tokens));
    AssertEquals('$ reads nothing', '', Tokens[3].Phonemes);
    AssertEquals('12: the next word is CD too, no unit', 'twˈɛlv',
      Tokens[4].Phonemes);
    AssertTrue('12: number', g2prNumber in Tokens[4].Resolutions);
    AssertEquals('million takes the unit', 'mˈɪljᵊn dˈɑləɹz', Tokens[5].Phonemes);
    AssertEquals('CD', Tokens[5].Tag);
    // ¥ is not one of misaki's currencies
    AssertPhonemes(G2P, 'That is ¥500 in yen.',
      'ðˈæt ɪz fˈIv hˈʌndɹəd ɪn jˈɛn.');
    // 10^306 and above: num2words raises; the port reads the digits
    Phonemes := G2P.Phonemize('1' + StringOfChar('0', 400));
    AssertEquals('digit by digit', 'wˈʌn zˈɪɹO zˈɪɹO', Copy(Phonemes, 1,
      Length('wˈʌn zˈɪɹO zˈɪɹO')));
    AssertEquals('one word per digit', 401,
      Length(Phonemes) - Length(StringReplace(Phonemes, ' ', '', [rfReplaceAll])) + 1);
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.TestUnknownWords;
var
  G2P: TNeuralEnglishG2P;
  Tokens: TNeuralG2PTokenArray;
  Phonemes, Pieces, LongWord: string;
  Resolutions: TNeuralG2PResolutions;

  // one-token text: its pieces, phonemes and resolution kinds
  procedure Check(const Text, ExpectedPieces, ExpectedPhonemes: string;
    ExpectedResolutions: TNeuralG2PResolutions);
  begin
    G2P.Phonemize(Text, Tokens);
    AssertEquals(Text + ' token count', 1, Length(Tokens));
    AssertEquals(Text + ' pieces', ExpectedPieces, Tokens[0].Pieces);
    AssertEquals(Text + ' phonemes', ExpectedPhonemes, Tokens[0].Phonemes);
    AssertTrue(Text + ' resolutions',
      Tokens[0].Resolutions = ExpectedResolutions);
    AssertTrue(Text + ' known', Tokens[0].IsKnown);
  end;

  function CountOf(const Mark, S: string): integer;
  begin
    Result := (Length(S) - Length(StringReplace(S, Mark, '', [rfReplaceAll]))) div
      Length(Mark);
  end;

begin
  AssertEquals('unknown word split', G2PResolutionName(g2prUnknownSplit));
  AssertEquals('unknown word spelled', G2PResolutionName(g2prUnknownSpelled));
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    // fewest pieces: sunflower + field, not sun + flower + field; only the
    // first piece keeps its primary stress
    Check('sunflowerfield', 'sunflower+field', 'sˈʌnflˌWəɹfˌild',
      [g2prUnknownSplit]);
    AssertEquals('one primary', 1, CountOf('ˈ', Tokens[0].Phonemes));
    Check('firetruckladder', 'fire+truck+ladder', 'fˈIəɹtɹˌʌklˌædəɹ',
      [g2prUnknownSplit]);
    // Capitalized: split on the lowercase form
    Check('Greenhousegas', 'greenhouse+gas', 'ɡɹˈinhˌWsɡˌæs', [g2prUnknownSplit]);
    // a letter no word covers is spelled with its letter entry
    Check('Paulo', 'pa+U+lo', 'pˈɑjˌulˌO', [g2prUnknownSplit, g2prUnknownSpelled]);
    // no dictionary word at all: every letter spelled, first one primary
    Check('Qwrtz', 'Q+W+R+T+Z', 'kjˈudˌʌbᵊljuˌɑɹtˌizˌi', [g2prUnknownSpelled]);
    // an ALL-CAPS string misaki cannot read (É) is spelled as misaki spells
    // acronyms: all secondary, the last one primary
    Check('ÉCOLE', 'E+C+O+L+E', G2P.Phonemize('ECOLE'), [g2prUnknownSpelled]);
    // accents of Latin-1 / Latin Extended-A are folded before the split ...
    Check('Sünflowerfield', 'sunflower+field', 'sˈʌnflˌWəɹfˌild',
      [g2prUnknownSplit]);
    // ... a letter without an ASCII base (ł) leaves the word unknown
    G2P.Phonemize('Wrocław', Tokens);
    AssertFalse('Wrocław unknown', Tokens[0].IsKnown);
    AssertEquals('Wrocław no pieces', '', Tokens[0].Pieces);
    AssertTrue('Wrocław', Tokens[0].Resolutions * cUnknownRoutes = []);
    // camelCase: the pieces misaki dropped are read with their neighbours
    Check('PyTorch', 'P+Y+torch', 'pˈiwˌItˌɔɹʧ',
      [g2prUnknownSplit, g2prUnknownSpelled]);
    Check('WiFi', 'W+if+I', 'dˈʌbᵊljuɪfˌI', [g2prUnknownSplit, g2prUnknownSpelled]);
    // possessive: the name is split, misaki's 's stays
    G2P.Phonemize('Nguyen''s', Tokens);
    AssertEquals('Nguyen''s pieces', 'N+guy+en', Tokens[0].Pieces);
    AssertEquals('Nguyen''s', 'ˈɛnɡˌIˌɛns', Tokens[0].Phonemes);
    // a clitic misaki drops is not a word: I'd've stays as misaki reads it
    G2P.Phonemize('I''d''ve', Tokens);
    AssertEquals('I''d''ve', 'ˌId', Tokens[0].Phonemes);
    AssertTrue('I''d''ve', Tokens[0].Resolutions * cUnknownRoutes = []);
    // context: the word before now sees the split word's first sound
    AssertPhonemes(G2P, 'the applegarden', 'ði ˈæpᵊlɡˌɑɹdən');
    // the length cap: 64 letters are read, 65 stay unknown
    LongWord := StringOfChar('b', csG2PMaxSplitLetters);
    Resolutions := [];
    AssertTrue('64 letters', G2P.ReadUnknownWord(LongWord, Phonemes, Pieces,
      Resolutions));
    AssertEquals('64 spelled letters', csG2PMaxSplitLetters,
      CountOf('B', Pieces));
    Resolutions := [];
    AssertFalse('65 letters', G2P.ReadUnknownWord(LongWord + 'b', Phonemes,
      Pieces, Resolutions));
    AssertEquals('65 letters: no phonemes', '', Phonemes);
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.CollectUnknownWordRows(G2P: TNeuralEnglishG2P;
  Rows: TStrings);
var
  Root: TJSONData;
  Sentences: TJSONArray;
  Tokens: TNeuralG2PTokenArray;
  Token: TNeuralG2PToken;
  Seen: TStringList;
  SentenceCnt, MaxSentencePos, TokenCnt, MaxTokenPos: integer;
  Pieces: string;
begin
  Root := ReadFixtureJSON('sentences.json');
  Seen := TStringList.Create;
  try
    Seen.Sorted := true;
    Sentences := TJSONObject(Root).Arrays['sentences'];
    MaxSentencePos := Sentences.Count - 1;
    for SentenceCnt := 0 to MaxSentencePos do
    begin
      G2P.Phonemize(Sentences.Objects[SentenceCnt].Strings['text'], Tokens);
      MaxTokenPos := Length(Tokens) - 1;
      for TokenCnt := 0 to MaxTokenPos do
      begin
        Token := Tokens[TokenCnt];
        if Token.IsKnown and (cUnknownRoutes * Token.Resolutions = []) then
          Continue;
        if Seen.IndexOf(Token.Text) >= 0 then Continue;
        Seen.Add(Token.Text);
        if Token.Pieces = '' then Pieces := '(unknown)' else Pieces := Token.Pieces;
        Rows.Add(Format('%-22s %-34s %s', [Token.Text, Pieces, Token.Phonemes]));
      end;
    end;
  finally
    Seen.Free;
    Root.Free;
  end;
end;

procedure TTestNeuralG2P.TestUnknownWordReport;
var
  G2P: TNeuralEnglishG2P;
  Rows: TStringList;
  Accent: TNeuralG2PAccent;
  RowCnt, MaxRowPos: integer;
begin
  Rows := TStringList.Create;
  try
    for Accent := g2pAmerican to g2pBritish do
    begin
      G2P := CreateFixtureG2P(Accent);
      try
        Rows.Clear;
        CollectUnknownWordRows(G2P, Rows);
      finally
        G2P.Free;
      end;
      WriteLn('Unknown fixture words (', EnglishG2PDictionaryFileName(Accent,
        true), ' subset): word, pieces, phonemes');
      MaxRowPos := Rows.Count - 1;
      for RowCnt := 0 to MaxRowPos do WriteLn('  ', Rows[RowCnt]);
      AssertTrue('rows', Rows.Count > 40);
    end;
  finally
    Rows.Free;
  end;
end;

procedure TTestNeuralG2P.TestUnknownWordRealDictionaries;
var
  DataDir, Word: string;
  G2P: TNeuralEnglishG2P;
  SubsetRows, RealRows: TStringList;
  Accent: TNeuralG2PAccent;
  RowCnt, MaxRowPos: integer;
begin
  DataDir := RealDictionaryDir();
  if DataDir = '' then
    Ignore('misaki dictionaries not found (set NEURAL_MISAKI_DATA_DIR to ' +
      'misaki''s data/ folder)');
  SubsetRows := TStringList.Create;
  RealRows := TStringList.Create;
  try
    for Accent := g2pAmerican to g2pBritish do
    begin
      SubsetRows.Clear;
      RealRows.Clear;
      G2P := CreateFixtureG2P(Accent);
      try
        CollectUnknownWordRows(G2P, SubsetRows);
      finally
        G2P.Free;
      end;
      G2P := TNeuralEnglishG2P.Create(DataDir, Accent);
      try
        CollectUnknownWordRows(G2P, RealRows);
      finally
        G2P.Free;
      end;
      AssertEquals('row count', SubsetRows.Count, RealRows.Count);
      MaxRowPos := RealRows.Count - 1;
      for RowCnt := 0 to MaxRowPos do
      begin
        if RealRows[RowCnt] = SubsetRows[RowCnt] then Continue;
        WriteLn(EnglishG2PDictionaryFileName(Accent, true), ' full vs subset:');
        WriteLn('  full   ', RealRows[RowCnt]);
        WriteLn('  subset ', SubsetRows[RowCnt]);
        // the subset holds the keys inside each fixture word; a folded
        // accent (café -> cafe) makes a key the subset does not have
        Word := Copy(RealRows[RowCnt], 1, Pos(' ', RealRows[RowCnt]) - 1);
        AssertTrue('only folded words differ: ' + RealRows[RowCnt],
          Length(Word) > CodePointCount(Word));
      end;
    end;
  finally
    RealRows.Free;
    SubsetRows.Free;
  end;
end;

procedure TTestNeuralG2P.RunFixtureAgreement(Accent: TNeuralG2PAccent);
const
  cGroups: array[0..13] of string = ('plain', 'punctuation',
    'capitals_acronyms', 'possessives', 'contractions', 'hyphenated',
    'suffix_stems', 'numbers', 'currency', 'percent', 'heteronyms',
    'unknown_and_names', 'long_mixed', 'numbers_more');
  // Lowest token agreement (%) each group must keep.
  cMinTokenAgreement: array[0..13] of double = (100, 100, 99, 100, 100, 100,
    100, 100, 100, 100, 100, 100, 100, 100);
  // Tokens misaki read only in part (a camelCase, possessive, hyphen or URL
  // piece dropped) that the unknown-word routes now read whole.
  cRoutedKnownUS = 'blorp-zing,Mia''s,Nguyen''s,PyTorch,Schuler''s,' +
    'usr/local/bin,WiFi,www.example.org/page';
  cRoutedKnownGB = cRoutedKnownUS;
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
    BaselineTokens, BaselineDiffering, UncountedPOSTokens, CharPos, MaxCharPos, MaxSentencePos,
    MaxLinePos: integer;
  IsCounted, IsPOSToken, IsEqual, IsRefUnknown, IsRouted,
    IsBeforeRouted: boolean;
  Lines, ContextLines, RoutedKnown: TStringList;
  ExpectedRoutedKnown: string;

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
  Root := ReadFixtureJSON('sentences.json');
  Notes := ReadFixtureJSON('notes.json');
  G2P := CreateFixtureG2P(Accent);
  Lines := TStringList.Create;
  ContextLines := TStringList.Create;
  RoutedKnown := TStringList.Create;
  RoutedKnown.Sorted := true;
  RoutedKnown.Duplicates := dupIgnore;
  try
    FillChar(Counts, SizeOf(Counts), 0);
    FillChar(Total, SizeOf(Total), 0);
    UncountedPOSTokens := 0;
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
        IsCounted := not IsRefUnknown;
        if (MyIdx < Length(Tokens)) and (Tokens[MyIdx].Text = RefText) then
        begin
          IsEqual := Tokens[MyIdx].Phonemes = RefPhonemes;
          // task 5.5 owns a token an unknown-word route read (misaki dropped
          // all or part of it) and the context it gives the word before it;
          // so does a word misaki left unknown that the -ed stem rule reads
          IsRouted := cUnknownRoutes * Tokens[MyIdx].Resolutions <> [];
          IsBeforeRouted := (MyIdx + 1 < Length(Tokens)) and
            ((cUnknownRoutes * Tokens[MyIdx + 1].Resolutions <> []) or
            ((RefIdx + 1 < RefTokens.Count) and (Tokens[MyIdx + 1].Phonemes <> '')
            and RefTokens.Objects[RefIdx + 1].Items[RefTokens.Objects[RefIdx + 1]
            .IndexOfName('phonemes')].IsNull));
          if IsCounted and IsPOSToken and (IsRouted or (IsBeforeRouted and
            not IsEqual)) then
            Inc(UncountedPOSTokens);
          if IsCounted and IsRouted then
            RoutedKnown.Add(RefText)
          else if IsCounted and IsBeforeRouted and not IsEqual then
            ContextLines.Add(Format('  %s #%d %s: "%s" misaki %s, port %s ' +
              '(next word now read)', [AccentKey, Sentence.Integers['id'], Group,
              RefText, RefPhonemes, Tokens[MyIdx].Phonemes]))
          else if IsCounted then
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
          if not RefToken.Items[RefToken.IndexOfName('phonemes')].IsNull then
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
        WriteLn(Format('  %-18s %4d/%-4d %6s   %3d/%-3d     %3d/%-3d',
          [cGroups[GroupCnt], TokensEqual, Tokens,
          Percent(TokensEqual, Tokens), SentencesEqual, Sentences,
          POSTokensEqual, POSTokens]));
        Inc(Total.Tokens, Tokens);
        Inc(Total.TokensEqual, TokensEqual);
        Inc(Total.POSTokens, POSTokens);
        Inc(Total.POSTokensEqual, POSTokensEqual);
        Inc(Total.Sentences, Sentences);
        Inc(Total.SentencesEqual, SentencesEqual);
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
    WriteLn('  not counted, read by an unknown-word route although misaki read ',
      'part of it: ', RoutedKnown.CommaText);
    MaxLinePos := ContextLines.Count - 1;
    for GroupCnt := 0 to MaxLinePos do WriteLn(ContextLines[GroupCnt]);
    if Accent = g2pBritish then ExpectedRoutedKnown := cRoutedKnownGB
    else ExpectedRoutedKnown := cRoutedKnownUS;
    AssertEquals('tokens misaki read in part, now read whole',
      ExpectedRoutedKnown, RoutedKnown.CommaText);
    // a POS-keyed token task 5.5 changed (minus its context) is not compared
    AssertEquals('POS-keyed token count matches notes.json', BaselineTokens,
      Total.POSTokens + UncountedPOSTokens);
    // DEFAULT alone agrees on 70.3%; the rule tagger reaches 96.8% (153/158)
    AssertTrue('POS-keyed agreement ' +
      Percent(Total.POSTokensEqual, Total.POSTokens) + ' >= 95%',
      Total.POSTokensEqual * 100 >= Total.POSTokens * 95);
    for GroupCnt := 0 to High(cGroups) do
      with Counts[GroupCnt] do
        AssertTrue(cGroups[GroupCnt] + ' token agreement ' +
          Percent(TokensEqual, Tokens),
          100.0 * TokensEqual >= cMinTokenAgreement[GroupCnt] * Tokens);
  finally
    RoutedKnown.Free;
    ContextLines.Free;
    Lines.Free;
    G2P.Free;
    Notes.Free;
    Root.Free;
  end;
end;

procedure TTestNeuralG2P.TestChunkSplitRules;
var
  Tokens: TNeuralG2PTokenArray;
  Chunks: TNeuralG2PChunkArray;
  ChunkCount: integer;
  G2P: TNeuralEnglishG2P;

  // Each token as 'phonemes' + an optional trailing space (its whitespace);
  // the text equals the phonemes.
  procedure SetTokens(const Spelled: array of string);
  var
    TokenPos: integer;
    Phonemes: string;
  begin
    SetLength(Tokens, Length(Spelled));
    for TokenPos := 0 to High(Spelled) do
    begin
      Tokens[TokenPos] := Default(TNeuralG2PToken);
      Phonemes := Spelled[TokenPos];
      if (Phonemes <> '') and (Phonemes[Length(Phonemes)] = ' ') then
      begin
        SetLength(Phonemes, Length(Phonemes) - 1);
        Tokens[TokenPos].Whitespace := ' ';
      end;
      Tokens[TokenPos].Text := Phonemes;
      Tokens[TokenPos].Phonemes := Phonemes;
      Tokens[TokenPos].IsKnown := true;
    end;
  end;

  procedure CheckChunks(MaxPhonemes: integer; const Expected: array of string);
  var
    ChunkPos: integer;
  begin
    Chunks := nil;
    ChunkCount := 0;
    SplitG2PTokensIntoChunks(Tokens, MaxPhonemes, 7, Chunks, ChunkCount);
    AssertEquals('chunk count', Length(Expected), ChunkCount);
    for ChunkPos := 0 to High(Expected) do
    begin
      AssertEquals('chunk ' + IntToStr(ChunkPos), Expected[ChunkPos],
        Chunks[ChunkPos].Phonemes);
      AssertEquals('chunk text ' + IntToStr(ChunkPos), Expected[ChunkPos],
        Chunks[ChunkPos].Text);
      AssertEquals('paragraph index', 7, Chunks[ChunkPos].ParagraphIndex);
    end;
  end;

begin
  // the cut follows the last '.' that leaves the rest within the limit
  SetTokens(['aa', ', ', 'bb', '. ', 'cc ', 'dd ', 'ee']);
  CheckChunks(10, ['aa, bb.', 'cc dd ee']);
  AssertEquals('tokens of the first chunk', 4, Length(Chunks[0].Tokens));
  CheckChunks(16, ['aa, bb. cc dd ee']);
  // cutting at '.' leaves 11 > 10 for the rest: the comma takes it
  SetTokens(['a', '. ', 'bbbbbb', ', ', 'cc ', 'dd']);
  CheckChunks(10, ['a. bbbbbb,', 'cc dd']);
  // a ')' right after the '.' goes with the first chunk
  SetTokens(['aa ', '(', 'bb', '.', ') ', 'cc ', 'dd ', 'ee']);
  CheckChunks(10, ['aa (bb.)', 'cc dd ee']);
  // no punctuation: every token that would overflow starts a chunk; a token
  // longer than the limit is a chunk of its own
  SetTokens(['ab ', 'cdefghij ', 'kl']);
  CheckChunks(5, ['ab', 'cdefghij', 'kl']);
  // ... and when it comes first, en_tokenize yields an empty chunk before it
  SetTokens(['abcdefg ', 'hi']);
  CheckChunks(5, ['', 'abcdefg', 'hi']);
  // PhonemizeInChunks (KPipeline.__call__): empty chunks dropped, a chunk
  // over the limit cut to MaxPhonemes code points, '\n' runs split paragraphs
  G2P := CreateFixtureG2P(g2pAmerican);
  try
    Chunks := G2P.PhonemizeInChunks('The quick brown fox.', 4);
    // a lone '.' is a chunk too, as in en_tokenize
    AssertEquals('chunks', 5, Length(Chunks));
    AssertEquals('ðə', Chunks[0].Phonemes);
    AssertEquals('.', Chunks[4].Phonemes);
    AssertEquals('kwˈɪ', Chunks[1].Phonemes);
    AssertTrue('cut', Chunks[1].IsTruncated);
    AssertFalse('not cut', Chunks[0].IsTruncated);
    AssertEquals('quick', Chunks[1].Text);
    Chunks := G2P.PhonemizeInChunks(LineEnding + ' The fox.' + #10#10 +
      '  ' + #10 + 'The dog. ' + #10, 510);
    AssertEquals('paragraph chunks', 2, Length(Chunks));
    AssertEquals('ðə fˈɑks.', Chunks[0].Phonemes);
    AssertEquals(0, Chunks[0].ParagraphIndex);
    AssertEquals('ðə dˈɔɡ.', Chunks[1].Phonemes);
    AssertEquals('The dog.', Chunks[1].Text);
    // the whitespace-only part between them keeps its index
    AssertEquals(2, Chunks[1].ParagraphIndex);
    AssertEquals('empty text', 0, Length(G2P.PhonemizeInChunks(' ' + #10, 510)));
  finally
    G2P.Free;
  end;
end;

procedure TTestNeuralG2P.RunChunkFixture(Accent: TNeuralG2PAccent);
var
  Root: TJSONData;
  Sentences, RefTokens, RefChunks: TJSONArray;
  Sentence, AccentObj, RefToken: TJSONObject;
  Tokens: TNeuralG2PTokenArray;
  Chunks: TNeuralG2PChunkArray;
  G2P: TNeuralEnglishG2P;
  AccentKey, Context, CutDifferences, ExpectedCutDifferences: string;
  SentencePos, TokenPos, ChunkPos, ChunkCount, LongSentences,
    PhonemeDifferences: integer;
  IsCutEqual: boolean;
begin
  if Accent = g2pBritish then AccentKey := 'gb' else AccentKey := 'us';
  ExpectedCutDifferences := AccentKey + ' #294';
  CutDifferences := '';
  Root := ReadFixtureJSON('sentences.json');
  G2P := CreateFixtureG2P(Accent);
  try
    Sentences := TJSONObject(Root).Arrays['sentences'];
    LongSentences := 0;
    PhonemeDifferences := 0;
    for SentencePos := 0 to Sentences.Count - 1 do
    begin
      Sentence := Sentences.Objects[SentencePos];
      AccentObj := Sentence.Objects[AccentKey];
      if AccentObj.IndexOfName('chunks') < 0 then Continue;
      Inc(LongSentences);
      Context := AccentKey + ' #' + IntToStr(Sentence.Integers['id']);
      RefChunks := AccentObj.Arrays['chunks'];
      // (1) misaki's tokens through the port's chunker: exact
      RefTokens := AccentObj.Arrays['tokens'];
      SetLength(Tokens, RefTokens.Count);
      for TokenPos := 0 to RefTokens.Count - 1 do
      begin
        RefToken := RefTokens.Objects[TokenPos];
        Tokens[TokenPos] := Default(TNeuralG2PToken);
        Tokens[TokenPos].Text := RefToken.Strings['text'];
        Tokens[TokenPos].Whitespace := RefToken.Strings['whitespace'];
        if not RefToken.Items[RefToken.IndexOfName('phonemes')].IsNull then
          Tokens[TokenPos].Phonemes := RefToken.Strings['phonemes'];
      end;
      Chunks := nil;
      ChunkCount := 0;
      SplitG2PTokensIntoChunks(Tokens, 510, 0, Chunks, ChunkCount);
      AssertEquals(Context + ' chunk count', RefChunks.Count, ChunkCount);
      for ChunkPos := 0 to ChunkCount - 1 do
      begin
        AssertEquals(Context + ' chunk text',
          RefChunks.Objects[ChunkPos].Strings['text'], Chunks[ChunkPos].Text);
        AssertEquals(Context + ' chunk phonemes',
          RefChunks.Objects[ChunkPos].Strings['phonemes'],
          Chunks[ChunkPos].Phonemes);
      end;
      // (2) the port's own tokens: the cuts differ only where the port reads
      // words misaki left empty (more phonemes before the cut)
      Chunks := G2P.PhonemizeInChunks(Sentence.Strings['text'], 510);
      IsCutEqual := Length(Chunks) = RefChunks.Count;
      for ChunkPos := 0 to High(Chunks) do
      begin
        AssertTrue(Context + ' port chunk within 510',
          CodePointCount(Chunks[ChunkPos].Phonemes) <= 510);
        if IsCutEqual and (Chunks[ChunkPos].Text <>
           RefChunks.Objects[ChunkPos].Strings['text']) then
          IsCutEqual := false;
      end;
      if not IsCutEqual then
      begin
        if CutDifferences <> '' then CutDifferences := CutDifferences + ',';
        CutDifferences := CutDifferences + Context;
        for ChunkPos := 0 to High(Chunks) do
          WriteLn('  ', Context, ' port chunk ', ChunkPos, ': ',
            CodePointCount(Chunks[ChunkPos].Phonemes), ' code points, ends "',
            Copy(Chunks[ChunkPos].Text, Length(Chunks[ChunkPos].Text) - 30, 31),
            '"');
        Continue;
      end;
      for ChunkPos := 0 to High(Chunks) do
        if Chunks[ChunkPos].Phonemes <>
           RefChunks.Objects[ChunkPos].Strings['phonemes'] then
        begin
          Inc(PhonemeDifferences);
          WriteLn('  ', Context, ' chunk ', ChunkPos, ' phonemes differ: misaki ',
            CodePointCount(RefChunks.Objects[ChunkPos].Strings['phonemes']),
            ' code points, port ', CodePointCount(Chunks[ChunkPos].Phonemes));
        end;
    end;
    WriteLn('KPipeline chunks (', AccentKey, '): misaki tokens -> chunks equal ',
      'in all ', LongSentences, ' sentences; port tokens: cuts differ in [',
      CutDifferences, '], chunk phonemes differ in ', PhonemeDifferences);
    AssertEquals('sentences whose port cuts differ', ExpectedCutDifferences,
      CutDifferences);
    AssertEquals('long_mixed sentences with chunks', 6, LongSentences);
  finally
    G2P.Free;
    Root.Free;
  end;
end;

procedure TTestNeuralG2P.TestChunksMatchKPipelineUS;
begin
  RunChunkFixture(g2pAmerican);
end;

procedure TTestNeuralG2P.TestChunksMatchKPipelineGB;
begin
  RunChunkFixture(g2pBritish);
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
  DataDir := RealDictionaryDir();
  if DataDir = '' then
    Ignore('misaki dictionaries not found (set NEURAL_MISAKI_DATA_DIR to ' +
      'misaki''s data/ folder)');
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
