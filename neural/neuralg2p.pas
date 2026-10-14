unit neuralg2p;

(*
neuralg2p
English grapheme-to-phoneme conversion (text -> the IPA-like phoneme string
Kokoro-82M reads). A pure-Pascal port of misaki 0.9.4 `en.py`
(github.com/hexgrad/misaki, Apache-2.0) WITHOUT spaCy:

  * Dictionaries: misaki's us_gold.json + us_silver.json (American) or
    gb_gold.json + gb_silver.json (British), supplied by the user in a folder.
    TNeuralG2PDictionary parses each file once with its own scanner (no
    TJSONObject: no 255-byte key limit, no per-member objects) and indexes it
    in an open-addressing FNV-1a hash table. Keys and readings stay inside the
    file text; entries hold byte spans.
  * Tokenization: whitespace chunks with punctuation split off both ends (an
    approximation of spaCy's tokenizer), clitics ('s n't 're 've 'll 'd 'm),
    then misaki's sub-token regex (hyphens, camelCase, dots) ported by hand.
  * misaki's lexicon: capital-letter stress, acronyms and dotted
    abbreviations spelled with the letter entries, possessives, contractions,
    multi-piece words with stress demotion, the -s/-ed/-ing stem rules,
    right-to-left context (the -> ði before a vowel, to -> tə/tʊ, the "None"
    readings at a clause end) and the final ɾ -> T / ʔ -> t mapping.
  * Part of speech: a small rule tagger (TNeuralEnglishG2P.TagRawTokens)
    replaces spaCy. It reads the neighbouring words against fixed tables
    (cPreviousWordCues and friends) to choose the determiner / verb / noun /
    adjective reading of POS-keyed entries (read, present, that, ...).
  * Numbers (ReadNumber): misaki get_number on top of a port of num2words'
    English converter (EnglishCardinalWords, EnglishOrdinalWords,
    EnglishYearWords, EnglishDecimalWords): years, ordinals, decimals,
    version numbers, digits inside words, $ £ € amounts and % & + @.

Not here (later tasks), with their hook points: splitting unknown words into
dictionary words (5.5: a token or piece with g2prUnknown has empty phonemes,
as misaki does with fallback=None); long-text chunking (5.6: Phonemize's
per-token output). Not ported: misaki's markdown overrides ([word](/IPA/),
[word](+1), [n](#flags#), so no number flags) and its Unicode digit folding
(numeric_if_needed).
Deviations: a word of more than csMaxMergedPieces sub-tokens only tries merges
of up to that many (misaki's merge loop is cubic in the sub-token count); a
decimal is read from its digits as written (num2words reads the float64
value, which differs past 15 significant digits); a number misaki would crash
on (10^306 and above, a number word missing from the dictionary) is read digit
by digit or without that word.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

const
  // misaki's stress argument None (no change); real values are -2..2.
  csG2PNoStress = 1000.0;

type
  ENeuralG2PError = class(Exception);

  TNeuralG2PAccent = (g2pAmerican, g2pBritish);

  // How a token got its phonemes; a multi-piece word collects one per piece.
  TNeuralG2PResolution = (
    g2prPunctuation,   // punctuation, fixed by the tokenizer
    g2prDictionary,    // a gold or silver dictionary reading
    g2prContextRule,   // misaki special case (a/an/am/I/by/to/in/the/used/vs,
                       // symbols, a "None" reading at a clause end)
    g2prPOSRule,       // a POS-keyed entry read with a non-DEFAULT tag
    g2prStemRule,      // -s / -ed / -ing over a known stem
    g2prSpelled,       // spelled with the letter entries
    g2prUnknown,       // no reading: empty phonemes (misaki None)
    g2prNumber         // misaki get_number: digits read as number words
  );
  TNeuralG2PResolutions = set of TNeuralG2PResolution;

  // One output token (misaki's merged MToken).
  TNeuralG2PToken = record
    Text: string;
    Whitespace: string;
    Tag: string;          // Penn tag from the rule tagger, '' when none
    Phonemes: string;
    IsKnown: boolean;     // false: misaki's None, written as ''
    Resolutions: TNeuralG2PResolutions;
  end;
  TNeuralG2PTokenArray = array of TNeuralG2PToken;

  TNeuralG2PDictionaryEntry = record
    KeyStart, KeyLength: integer;      // byte span in the file text
    ValueStart, ValueLength: integer;  // plain reading, or the DEFAULT one
    ValueIsNull: boolean;
    FirstVariant, VariantCount: integer; // POS-keyed readings; 0 = plain
  end;

  TNeuralG2PDictionaryVariant = record
    TagStart, TagLength: integer;
    ValueStart, ValueLength: integer;
    ValueIsNull: boolean;
  end;

  { TNeuralG2PDictionary }
  // One misaki dictionary file: {"word": "phonemes" | null |
  // {"DEFAULT": ..., "NOUN": ..., "None": ...}, ...}.
  TNeuralG2PDictionary = class(TObject)
  private
    FText: string;
    FSourceName: string;
    FEntries: array of TNeuralG2PDictionaryEntry;
    FEntryCount: integer;
    FVariants: array of TNeuralG2PDictionaryVariant;
    FVariantCount: integer;
    FSlots: array of integer;  // entry index + 1, 0 = empty
    FSlotMask: integer;
    procedure ParseText();
    procedure BuildIndex();
    procedure RaiseParseError(Position: integer; const Expected: string);
    procedure ScanString(var Position: integer; out Start, Len: integer);
    function SpanToString(Start, Len: integer): string;
    function SpanEquals(Start, Len: integer; const S: string): boolean;
    function FindSlot(P: PAnsiChar; Len: integer): integer;
    function VariantIndex(Index: integer; const Tag: string): integer;
  public
    procedure LoadFromFile(const FileName: string);
    procedure LoadFromString(const JSONText: string;
      const SourceName: string = '');
    // Exact key lookup; -1 when absent.
    function IndexOf(const Key: string): integer;
    // misaki grow_dictionary: a lowercase key also answers its Capitalized
    // form and vice versa (keys of 2+ characters, the exact key wins).
    function IndexOfGrown(const Key: string): integer;
    function IsPOSKeyed(Index: integer): boolean;
    function HasTag(Index: integer; const Tag: string): boolean;
    // Plain reading or DEFAULT; false when the value is null.
    function DefaultReading(Index: integer; out Phonemes: string): boolean;
    // misaki ps.get(Tag, ps['DEFAULT']); false when that value is null.
    function TagReading(Index: integer; const Tag: string;
      out Phonemes: string): boolean;
    function KeyAt(Index: integer): string;
    property Count: integer read FEntryCount;
  end;

  // misaki TokenContext.future_vowel: None / False / True.
  TNeuralG2PNextSound = (g2pnsUnknown, g2pnsConsonant, g2pnsVowel);

  TNeuralG2PContext = record
    NextSound: TNeuralG2PNextSound;
    NextIsTo: boolean;
  end;

  // Tokenizer output before sub-token splitting (one spaCy-like token).
  TNeuralG2PRawToken = record
    Text: string;
    Whitespace: string;
    Tag: string;
    IsPunctuation: boolean; // phonemes fixed by the tokenizer
    Phonemes: string;
    EndsSentence: boolean;
  end;

  // A sub-token (misaki subtokenize piece) and its reading.
  TNeuralG2PPiece = record
    Text: string;
    Whitespace: string;
    Tag: string;
    Alias: string;
    HasAlias: boolean;
    Phonemes: string;
    HasPhonemes: boolean;
    Prespace: boolean;
    Currency: string;  // misaki _.currency: '$', '£' or '€' on an amount
    Resolutions: TNeuralG2PResolutions;
  end;

  // Consecutive pieces read as one word (misaki's list tokens).
  TNeuralG2PGroup = record
    FirstPiece, LastPiece: integer;
    IsOpenList: boolean;
  end;

  { TNeuralEnglishG2P }
  // English text -> phonemes with misaki's dictionaries and rules.
  TNeuralEnglishG2P = class(TObject)
  private
    FAccent: TNeuralG2PAccent;
    FIsBritish: boolean;
    FGold, FSilver: TNeuralG2PDictionary;
    FRaw: array of TNeuralG2PRawToken;
    FRawCount: integer;
    FPieces: array of TNeuralG2PPiece;
    FPieceCount: integer;
    FGroups: array of TNeuralG2PGroup;
    FGroupCount: integer;
    FDoubleQuoteOpen, FSingleQuoteOpen: boolean;
    // tokenizer
    procedure AddRawToken(const Text: string; IsPunctuation: boolean;
      const Phonemes, Tag: string);
    procedure AddPunctuationToken(const Text: string);
    procedure SplitIntoRawTokens(const Text: string);
    procedure ProcessChunk(const Chunk, Whitespace: string);
    procedure ProcessPunctuationChunk(const Chunk: string);
    procedure ProcessCore(const Core: string);
    function IsAbbreviation(const Word: string): boolean;
    function PunctuationPhonemes(const Text: string; IsPrefix,
      IsSuffix: boolean): string;
    // tagger
    procedure TagRawTokens();
    function TagWord(Index: integer; IsShouting, IsSentenceStart: boolean): string;
    function PreviousWord(Index: integer; out Word: string): boolean;
    function NextWord(Index: integer; out Word: string): boolean;
    // pieces and groups
    procedure AddPiece(const Piece: TNeuralG2PPiece);
    procedure BuildPiecesAndGroups();
    procedure AppendWordPieces(const RawToken: TNeuralG2PRawToken);
    procedure ResolveGroups();
    procedure ResolveMultiPieceGroup(var Ctx: TNeuralG2PContext;
      FirstPiece, LastPiece: integer);
    procedure ResolveTokens(FirstPiece, LastPiece: integer);
    function HeaviestPieceTag(FirstPiece, LastPiece: integer): string;
    function PiecesText(FirstPiece, LastPiece: integer): string;
    function PiecesPhonemes(FirstPiece, LastPiece: integer;
      out AnyKnown: boolean): string;
    function BuildOutput(out Tokens: TNeuralG2PTokenArray): string;
    // misaki Lexicon
    function GoldHas(const Word: string): boolean;
    function SilverHas(const Word: string): boolean;
    function IsKnown(const Word: string): boolean;
    function SpellLetters(const Word: string; out Phonemes: string): boolean;
    function Lookup(const AWord, Tag: string; Stress: single;
      const Ctx: TNeuralG2PContext; HasContext: boolean; out Phonemes: string;
      var Resolutions: TNeuralG2PResolutions): boolean;
    function SpecialCase(const Word, Tag: string; Stress: single;
      const Ctx: TNeuralG2PContext; out Phonemes: string;
      var Resolutions: TNeuralG2PResolutions): boolean;
    function SuffixS(const Stem: string): string;
    function SuffixEd(const Stem: string): string;
    function SuffixIng(const Stem: string; out Phonemes: string): boolean;
    function StemS(const Word, Tag: string; Stress: single;
      const Ctx: TNeuralG2PContext; out Phonemes: string;
      var Resolutions: TNeuralG2PResolutions): boolean;
    function StemEd(const Word, Tag: string; Stress: single;
      const Ctx: TNeuralG2PContext; out Phonemes: string;
      var Resolutions: TNeuralG2PResolutions): boolean;
    function StemIng(const Word, Tag: string; Stress: single;
      const Ctx: TNeuralG2PContext; out Phonemes: string;
      var Resolutions: TNeuralG2PResolutions): boolean;
    function GetWord(const AWord, Tag: string; Stress: single;
      const Ctx: TNeuralG2PContext; out Phonemes: string;
      var Resolutions: TNeuralG2PResolutions): boolean;
    function ReadNumber(const AWord, Currency: string; IsHead: boolean;
      out Phonemes: string): boolean;
    function AppendCurrencyUnit(const Phonemes, Currency: string): string;
    function ReadWord(const Text, Tag, Currency: string; IsHead: boolean;
      const Ctx: TNeuralG2PContext; out Phonemes: string;
      var Resolutions: TNeuralG2PResolutions): boolean;
  public
    // Loads <DictionaryDir>/{us|gb}_gold.json and _silver.json; raises
    // ENeuralG2PError naming every missing file.
    constructor Create(const DictionaryDir: string; Accent: TNeuralG2PAccent);
    destructor Destroy; override;
    function Phonemize(const Text: string): string; overload;
    function Phonemize(const Text: string;
      out Tokens: TNeuralG2PTokenArray): string; overload;
    property Accent: TNeuralG2PAccent read FAccent;
    property Gold: TNeuralG2PDictionary read FGold;
    property Silver: TNeuralG2PDictionary read FSilver;
  end;

// File name of one misaki dictionary: us_gold.json, gb_silver.json, ...
function EnglishG2PDictionaryFileName(Accent: TNeuralG2PAccent;
  IsGold: boolean): string;
// Kokoro voice prefix: 'a...' = American, 'b...' = British; others raise.
function EnglishG2PAccentFromVoice(const VoiceName: string): TNeuralG2PAccent;
// misaki apply_stress (Stress = csG2PNoStress leaves Phonemes unchanged).
function ApplyG2PStress(const Phonemes: string; Stress: single): string;
// misaki subtokenize: the sub-token regex of en.py, matched by hand.
procedure G2PSubTokenize(const Word: string; Pieces: TStrings);
function G2PResolutionName(Resolution: TNeuralG2PResolution): string;
// num2words(int(Digits), lang='en') 0.5.14: "one hundred and one",
// "twenty-one". Digits: [-]0-9..., below 10^306; others raise ENeuralG2PError.
function EnglishCardinalWords(const Digits: string): string;
// num2words to='ordinal': "twenty-first", "one hundredth"; no '-' allowed.
function EnglishOrdinalWords(const Digits: string): string;
// num2words to='year': "ten sixty-six", "nineteen oh-five", "two thousand".
function EnglishYearWords(const Digits: string): string;
// num2words(float(Number)): [-]digits[.digits] -> "three point one four";
// a zero fraction reads as the integer ("2.0" -> "two").
function EnglishDecimalWords(const Number: string): string;

implementation

uses
  Math, neuralhftokenizer;

const
  csPrimaryStress = #$CB#$88;   // ˈ
  csSecondaryStress = #$CB#$8C; // ˌ
  csSchwaA = #$C9#$90;          // ɐ
  csSmallCapI = #$C9#$AA;       // ɪ
  csBarredI = #$E1#$B5#$BB;     // ᵻ
  csFlapT = #$C9#$BE;           // ɾ
  csGlottalStop = #$CA#$94;     // ʔ
  csEng = #$C5#$8B;             // ŋ
  csEth = #$C3#$B0;             // ð
  csSchwa = #$C9#$99;           // ə
  csUpsilon = #$CA#$8A;         // ʊ
  csEmDash = #$E2#$80#$94;      // —
  // Longest merge of sub-tokens tried as one word (ResolveMultiPieceGroup);
  // the longest key in misaki's four files has 13 (what-you-see-is-what-you-get).
  csMaxMergedPieces = 32;
  csLeftDoubleQuote = #$E2#$80#$9C;  // “
  csRightDoubleQuote = #$E2#$80#$9D; // ”
  cpLeftSingleQuote = $2018;
  cpRightSingleQuote = $2019;
  cpLeftDoubleQuote = $201C;
  cpRightDoubleQuote = $201D;
  cpEmDash = $2014;
  cpEnDash = $2013;
  cpEllipsis = $2026;

type
  TG2PTagCue = record
    Word: string;
    Tag: string;
  end;

const
  // Part-of-speech cues read from the PREVIOUS word (lowercase). 'VBD' on
  // he/she/it becomes 'VBZ' when the word itself ends in -s.
  cPreviousWordCues: array[0..63] of TG2PTagCue = (
    (Word: 'to'; Tag: 'VB'), (Word: 'will'; Tag: 'VB'),
    (Word: 'would'; Tag: 'VB'), (Word: 'shall'; Tag: 'VB'),
    (Word: 'should'; Tag: 'VB'), (Word: 'can'; Tag: 'VB'),
    (Word: 'could'; Tag: 'VB'), (Word: 'may'; Tag: 'VB'),
    (Word: 'might'; Tag: 'VB'), (Word: 'must'; Tag: 'VB'),
    (Word: 'do'; Tag: 'VB'), (Word: 'does'; Tag: 'VB'),
    (Word: 'did'; Tag: 'VB'), (Word: 'not'; Tag: 'VB'),
    (Word: 'cannot'; Tag: 'VB'), (Word: 'please'; Tag: 'VB'),
    (Word: 'let''s'; Tag: 'VB'), (Word: 'i'; Tag: 'VBP'),
    (Word: 'you'; Tag: 'VBP'), (Word: 'we'; Tag: 'VBP'),
    (Word: 'they'; Tag: 'VBP'), (Word: 'he'; Tag: 'VBD'),
    (Word: 'she'; Tag: 'VBD'), (Word: 'it'; Tag: 'VBD'),
    (Word: 'had'; Tag: 'VBN'), (Word: 'has'; Tag: 'VBN'),
    (Word: 'have'; Tag: 'VBN'), (Word: 'was'; Tag: 'VBN'),
    (Word: 'were'; Tag: 'VBN'), (Word: 'been'; Tag: 'VBN'),
    (Word: 'is'; Tag: 'JJ'), (Word: 'are'; Tag: 'JJ'),
    (Word: 'am'; Tag: 'JJ'), (Word: 'be'; Tag: 'JJ'),
    (Word: 'being'; Tag: 'JJ'), (Word: 'very'; Tag: 'JJ'),
    (Word: 'so'; Tag: 'JJ'), (Word: 'too'; Tag: 'JJ'),
    (Word: 'quite'; Tag: 'JJ'), (Word: 'rather'; Tag: 'JJ'),
    (Word: 'more'; Tag: 'JJ'), (Word: 'most'; Tag: 'JJ'),
    (Word: 'stay'; Tag: 'JJ'), (Word: 'the'; Tag: 'NN'),
    (Word: 'a'; Tag: 'NN'), (Word: 'an'; Tag: 'NN'),
    (Word: 'this'; Tag: 'NN'), (Word: 'these'; Tag: 'NN'),
    (Word: 'those'; Tag: 'NN'), (Word: 'my'; Tag: 'NN'),
    (Word: 'your'; Tag: 'NN'), (Word: 'his'; Tag: 'NN'),
    (Word: 'her'; Tag: 'NN'), (Word: 'its'; Tag: 'NN'),
    (Word: 'our'; Tag: 'NN'), (Word: 'their'; Tag: 'NN'),
    (Word: 'no'; Tag: 'NN'), (Word: 'every'; Tag: 'NN'),
    (Word: 'each'; Tag: 'NN'), (Word: 'of'; Tag: 'NN'),
    (Word: 'with'; Tag: 'NN'), (Word: 'for'; Tag: 'NN'),
    (Word: 'from'; Tag: 'NN'), (Word: 'about'; Tag: 'NN')
  );
  // Contraction endings of the PREVIOUS word: we'll / don't / I've / I'd.
  cPreviousSuffixCues: array[0..3] of TG2PTagCue = (
    (Word: '''ll'; Tag: 'VB'), (Word: 'n''t'; Tag: 'VB'),
    (Word: '''ve'; Tag: 'VBN'), (Word: '''d'; Tag: 'VB')
  );
  // A following object pronoun marks a verb: "excuse me", "subject them".
  cObjectPronouns: array[0..3] of string = ('me', 'him', 'them', 'us');
  // At a clause start or after a conjunction, a following determiner marks
  // an imperative verb: "Close the door", "and present the history".
  cDeterminers: array[0..20] of string = ('the', 'a', 'an', 'it', 'this',
    'that', 'these', 'those', 'my', 'your', 'his', 'her', 'our', 'their',
    'its', 'some', 'all', 'every', 'each', 'any', 'no');
  cConjunctions: array[0..2] of string = ('and', 'or', 'but');
  // Words whose tag does not depend on context (lowercase, clitics removed).
  cClosedClassTags: array[0..21] of TG2PTagCue = (
    (Word: 'an'; Tag: 'DT'), (Word: 'the'; Tag: 'DT'),
    (Word: 'to'; Tag: 'TO'), (Word: 'in'; Tag: 'IN'),
    (Word: 'by'; Tag: 'IN'), (Word: 'vs'; Tag: 'IN'),
    (Word: 'vs.'; Tag: 'IN'), (Word: 'is'; Tag: 'VBZ'),
    (Word: 'are'; Tag: 'VBP'), (Word: 'was'; Tag: 'VBD'),
    (Word: 'were'; Tag: 'VBD'), (Word: 'be'; Tag: 'VB'),
    (Word: 'been'; Tag: 'VBN'), (Word: 'am'; Tag: 'VBP'),
    (Word: 'has'; Tag: 'VBZ'), (Word: 'have'; Tag: 'VBP'),
    (Word: 'had'; Tag: 'VBD'), (Word: 'do'; Tag: 'VBP'),
    (Word: 'does'; Tag: 'VBZ'), (Word: 'did'; Tag: 'VBD'),
    (Word: 'a'; Tag: 'DT'), (Word: 'i'; Tag: 'PRP')
  );
  // 'that' followed by one of these is a complementizer (DEFAULT reading).
  cThatComplementNext: array[0..17] of string = ('the', 'a', 'an', 'i',
    'you', 'he', 'she', 'we', 'they', 'it', 'this', 'there', 'my', 'your',
    'his', 'her', 'our', 'their');

{ ---------------------------------------------------------------- }
{ Code point helpers                                                }
{ ---------------------------------------------------------------- }

function IsVowelCP(CP: cardinal): boolean;
begin
  case CP of
    Ord('A'), Ord('I'), Ord('O'), Ord('Q'), Ord('W'), Ord('Y'), Ord('a'),
    Ord('i'), Ord('u'), $E6, $251, $252, $254, $259, $25B, $25C, $26A,
    $28A, $28C, $1D7B: Result := true;
  else
    Result := false;
  end;
end;

function IsConsonantCP(CP: cardinal): boolean;
begin
  case CP of
    Ord('b'), Ord('d'), Ord('f'), Ord('h'), Ord('j'), Ord('k'), Ord('l'),
    Ord('m'), Ord('n'), Ord('p'), Ord('s'), Ord('t'), Ord('v'), Ord('w'),
    Ord('z'), $F0, $14B, $261, $279, $27E, $283, $292, $2A4, $2A7, $3B8:
      Result := true;
  else
    Result := false;
  end;
end;

function IsNonQuotePunctCP(CP: cardinal): boolean;
begin
  case CP of
    Ord(';'), Ord(':'), Ord(','), Ord('.'), Ord('!'), Ord('?'), cpEmDash,
    cpEllipsis: Result := true;
  else
    Result := false;
  end;
end;

function IsPunctCP(CP: cardinal): boolean;
begin
  Result := IsNonQuotePunctCP(CP) or (CP = Ord('"')) or
    (CP = cpLeftDoubleQuote) or (CP = cpRightDoubleQuote);
end;

function IsDiphthongCP(CP: cardinal): boolean;
begin
  case CP of
    Ord('A'), Ord('I'), Ord('O'), Ord('Q'), Ord('W'), Ord('Y'), $2A4, $2A7:
      Result := true;
  else
    Result := false;
  end;
end;

// misaki US_TAUS: sounds before a t that make the US flap (ɾ).
function IsUSTauCP(CP: cardinal): boolean;
begin
  case CP of
    Ord('A'), Ord('I'), Ord('O'), Ord('W'), Ord('Y'), Ord('i'), Ord('u'),
    $E6, $251, $259, $25B, $26A, $279, $28A, $28C: Result := true;
  else
    Result := false;
  end;
end;

// misaki SUBTOKEN_JUNKS: ' , - . _ ‘ ’ /
function IsJunkCP(CP: cardinal): boolean;
begin
  case CP of
    Ord(''''), Ord(','), Ord('-'), Ord('.'), Ord('_'), Ord('/'),
    cpLeftSingleQuote, cpRightSingleQuote: Result := true;
  else
    Result := false;
  end;
end;

function IsApostropheCP(CP: cardinal): boolean;
begin
  Result := (CP = Ord('''')) or (CP = cpLeftSingleQuote) or
    (CP = cpRightSingleQuote);
end;

function IsDigitCP(CP: cardinal): boolean;
begin
  Result := (CP >= Ord('0')) and (CP <= Ord('9'));
end;

// \p{Lu} / \p{Ll}, exact for ASCII and the blocks UppercaseCP covers;
// other letters count as lowercase.
function IsUpperLetterCP(CP: cardinal): boolean;
begin
  Result := IsLetterCP(CP) and (LowercaseCP(CP) <> CP);
end;

function IsLowerLetterCP(CP: cardinal): boolean;
begin
  Result := IsLetterCP(CP) and (LowercaseCP(CP) = CP);
end;

function HasNonASCII(const S: string): boolean;
var
  BytePos, MaxBytePos: integer;
begin
  MaxBytePos := Length(S);
  for BytePos := 1 to MaxBytePos do
    if Ord(S[BytePos]) >= $80 then Exit(true);
  Result := false;
end;

function MapCase(const S: string; ToUpper: boolean): string;
var
  Position, Start, OutPos, ByteCnt, TextLen, EncodedLen: integer;
  CP, Mapped: cardinal;
  Encoded: string;
begin
  if not HasNonASCII(S) then
  begin
    if ToUpper then Result := UpperCase(S) else Result := LowerCase(S);
    Exit;
  end;
  TextLen := Length(S);
  SetLength(Result, TextLen * 4);
  OutPos := 0;
  Position := 1;
  while Position <= TextLen do
  begin
    Start := Position;
    CP := NextCodePoint(S, Position);
    if ToUpper then Mapped := UppercaseCP(CP) else Mapped := LowercaseCP(CP);
    if Mapped = CP then
      Encoded := Copy(S, Start, Position - Start)
    else
      Encoded := CodePointToUTF8(Mapped);
    EncodedLen := Length(Encoded);
    for ByteCnt := 1 to EncodedLen do
    begin
      Inc(OutPos);
      Result[OutPos] := Encoded[ByteCnt];
    end;
  end;
  SetLength(Result, OutPos);
end;

function G2PLower(const S: string): string;
begin
  Result := MapCase(S, false);
end;

function G2PUpper(const S: string): string;
begin
  Result := MapCase(S, true);
end;

// Byte index just past the first code point (S non-empty).
function AfterFirstCP(const S: string): integer;
begin
  Result := 1;
  NextCodePoint(S, Result);
end;

// Python str.capitalize: first code point upper, the rest lower.
function G2PCapitalize(const S: string): string;
var
  Rest: integer;
begin
  if S = '' then Exit('');
  Rest := AfterFirstCP(S);
  Result := G2PUpper(Copy(S, 1, Rest - 1)) + G2PLower(Copy(S, Rest, MaxInt));
end;

// Everything after the first code point (Python word[1:]).
function TailAfterFirstCP(const S: string): string;
begin
  if S = '' then Exit('');
  Result := Copy(S, AfterFirstCP(S), MaxInt);
end;

// Python str.isalpha: non-empty and every code point a letter.
function IsAlphaText(const S: string): boolean;
var
  Position, TextLen: integer;
begin
  TextLen := Length(S);
  if TextLen = 0 then Exit(false);
  Position := 1;
  while Position <= TextLen do
    if not IsLetterCP(NextCodePoint(S, Position)) then Exit(false);
  Result := true;
end;

// misaki LEXICON_ORDS: ' - A-Z a-z only.
function HasOnlyLexiconChars(const S: string): boolean;
var
  BytePos, MaxBytePos: integer;
begin
  MaxBytePos := Length(S);
  for BytePos := 1 to MaxBytePos do
    if not (S[BytePos] in ['''', '-', 'A'..'Z', 'a'..'z']) then Exit(false);
  Result := true;
end;

function LastCPStart(const S: string): integer;
begin
  Result := Length(S);
  while (Result > 1) and ((Ord(S[Result]) and $C0) = $80) do Dec(Result);
end;

function LastCP(const S: string): cardinal;
var
  Position: integer;
begin
  if S = '' then Exit(0);
  Position := LastCPStart(S);
  Result := NextCodePoint(S, Position);
end;

function DropLastCP(const S: string): string;
begin
  if S = '' then Exit('');
  Result := Copy(S, 1, LastCPStart(S) - 1);
end;

function SecondLastCP(const S: string): cardinal;
begin
  Result := LastCP(DropLastCP(S));
end;

function HasVowel(const S: string): boolean;
var
  Position, TextLen: integer;
begin
  TextLen := Length(S);
  Position := 1;
  while Position <= TextLen do
    if IsVowelCP(NextCodePoint(S, Position)) then Exit(true);
  Result := false;
end;

function InsertBeforeFirstVowel(const S, Mark: string): string;
var
  Position, Start, TextLen: integer;
begin
  TextLen := Length(S);
  Position := 1;
  while Position <= TextLen do
  begin
    Start := Position;
    if IsVowelCP(NextCodePoint(S, Position)) then
      Exit(Copy(S, 1, Start - 1) + Mark + Copy(S, Start, MaxInt));
  end;
  Result := S;
end;

// misaki stress_weight: diphthong letters count double.
function StressWeight(const S: string): integer;
var
  Position, TextLen: integer;
begin
  Result := 0;
  TextLen := Length(S);
  Position := 1;
  while Position <= TextLen do
    if IsDiphthongCP(NextCodePoint(S, Position)) then Inc(Result, 2)
    else Inc(Result);
end;

function EndsWith(const S, Suffix: string): boolean;
var
  SuffixLen: integer;
begin
  SuffixLen := Length(Suffix);
  Result := (Length(S) >= SuffixLen) and
    (CompareByte(S[Length(S) - SuffixLen + 1], Suffix[1], SuffixLen) = 0);
end;

function DropBytes(const S: string; Count: integer): string;
begin
  Result := Copy(S, 1, Length(S) - Count);
end;

function InStringArray(const S: string; const Items: array of string): boolean;
var
  ItemCnt, MaxItemPos: integer;
begin
  MaxItemPos := High(Items);
  for ItemCnt := 0 to MaxItemPos do
    if Items[ItemCnt] = S then Exit(true);
  Result := false;
end;

function ReplaceCurlyApostrophes(const S: string): string;
begin
  Result := StringReplace(S, #$E2#$80#$98, '''', [rfReplaceAll]);
  Result := StringReplace(Result, #$E2#$80#$99, '''', [rfReplaceAll]);
end;

// misaki get_parent_tag.
function ParentTag(const Tag: string): string;
begin
  if Copy(Tag, 1, 2) = 'VB' then Result := 'VERB'
  else if Copy(Tag, 1, 2) = 'NN' then Result := 'NOUN'
  else if (Copy(Tag, 1, 3) = 'ADV') or (Copy(Tag, 1, 2) = 'RB') then
    Result := 'ADV'
  else if (Copy(Tag, 1, 3) = 'ADJ') or (Copy(Tag, 1, 2) = 'JJ') then
    Result := 'ADJ'
  else Result := Tag;
end;

const
  // misaki CURRENCIES: unit and sub-unit per sign (¥ is not one).
  cCurrencySigns: array[0..2] of string = ('$', #$C2#$A3, #$E2#$82#$AC);
  cCurrencyUnits: array[0..2, 0..1] of string = (('dollar', 'cent'),
    ('pound', 'pence'), ('euro', 'cent'));

function CurrencyIndex(const Sign: string): integer;
var
  SignCnt: integer;
begin
  for SignCnt := 0 to High(cCurrencySigns) do
    if cCurrencySigns[SignCnt] = Sign then Exit(SignCnt);
  Result := -1;
end;

// misaki SYMBOLS: % & + @.
function SymbolWord(const Word: string): string;
begin
  if Word = '%' then Result := 'percent'
  else if Word = '&' then Result := 'and'
  else if Word = '+' then Result := 'plus'
  else if Word = '@' then Result := 'at'
  else Result := '';
end;

// misaki is_number: digits , . and the suffixes ing 'd ed 's st nd rd th s;
// a leading '-' only at the head of a word.
function IsNumberWord(const Word: string; IsHead: boolean): boolean;
const
  cSuffixes: array[0..9] of string = ('ing', '''d', 'ed', '''s', 'st', 'nd',
    'rd', 'th', 's', '');
var
  Stripped: string;
  SuffixCnt, BytePos, MaxBytePos: integer;
  HasDigit: boolean;
begin
  HasDigit := false;
  MaxBytePos := Length(Word);
  for BytePos := 1 to MaxBytePos do
    if Word[BytePos] in ['0'..'9'] then HasDigit := true;
  if not HasDigit then Exit(false);
  Stripped := Word;
  for SuffixCnt := 0 to 8 do
    if EndsWith(Word, cSuffixes[SuffixCnt]) then
    begin
      Stripped := DropBytes(Word, Length(cSuffixes[SuffixCnt]));
      Break;
    end;
  MaxBytePos := Length(Stripped);
  for BytePos := 1 to MaxBytePos do
    if not ((Stripped[BytePos] in ['0'..'9', ',', '.']) or
      (IsHead and (BytePos = 1) and (Stripped[BytePos] = '-'))) then Exit(false);
  Result := true;
end;

{ ---------------------------------------------------------------- }
{ num2words 0.5.14, English (lang_EN.py over base.py)               }
{ ---------------------------------------------------------------- }

const
  cLowNumberWords: array[0..20] of string = ('zero', 'one', 'two', 'three',
    'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen',
    'nineteen', 'twenty');
  cTensNumberWords: array[2..9] of string = ('twenty', 'thirty', 'forty',
    'fifty', 'sixty', 'seventy', 'eighty', 'ninety');
  // num2words' "illion" prefixes: 10^6 .. 10^30, then unit + ten prefixes
  // up to 10^330 (only 10^33 .. 10^300 used), then cent (10^303).
  cIllionLowPrefixes: array[0..8] of string = ('m', 'b', 'tr', 'quadr',
    'quint', 'sext', 'sept', 'oct', 'non');
  cIllionUnitPrefixes: array[0..9] of string = ('', 'un', 'duo', 'tre',
    'quattuor', 'quin', 'sex', 'sept', 'octo', 'novem');
  cIllionTenPrefixes: array[0..8] of string = ('dec', 'vigint', 'trigint',
    'quadragint', 'quinquagint', 'sexagint', 'septuagint', 'octogint',
    'nonagint');
  // num2words' MAXVAL is 1000 * 10^303.
  csMaxCardinalDigits = 306;

// 0..99: "seven", "forty-two".
function SmallCardinalWords(Value: integer): string;
begin
  if Value <= 20 then Exit(cLowNumberWords[Value]);
  Result := cTensNumberWords[Value div 10];
  if Value mod 10 > 0 then
    Result := Result + '-' + cLowNumberWords[Value mod 10];
end;

// Name of 10^Exponent for Exponent = 6, 9, ..., 303.
function IllionWord(Exponent: integer): string;
var
  PrefixIdx: integer;
begin
  PrefixIdx := (Exponent - 6) div 3;
  if PrefixIdx <= High(cIllionLowPrefixes) then
    Result := cIllionLowPrefixes[PrefixIdx]
  else if PrefixIdx < 99 then
  begin
    Dec(PrefixIdx, Length(cIllionLowPrefixes));
    Result := cIllionUnitPrefixes[PrefixIdx mod 10] +
      cIllionTenPrefixes[PrefixIdx div 10];
  end
  else Result := 'cent';
  Result := Result + 'illion';
end;

function StripLeadingZeros(const Digits: string): string;
var
  FirstPos, MaxFirstPos: integer;
begin
  FirstPos := 1;
  MaxFirstPos := Length(Digits) - 1;
  while (FirstPos <= MaxFirstPos) and (Digits[FirstPos] = '0') do Inc(FirstPos);
  Result := Copy(Digits, FirstPos, MaxInt);
end;

function IsDigitText(const S: string): boolean;
var
  BytePos, MaxBytePos: integer;
begin
  MaxBytePos := Length(S);
  if MaxBytePos = 0 then Exit(false);
  for BytePos := 1 to MaxBytePos do
    if not (S[BytePos] in ['0'..'9']) then Exit(false);
  Result := true;
end;

// Splits an optional leading '-' off and checks the digits.
procedure ParseSignedDigits(const Digits: string; out IsNegative: boolean;
  out Magnitude: string);
begin
  IsNegative := (Digits <> '') and (Digits[1] = '-');
  if IsNegative then Magnitude := Copy(Digits, 2, MaxInt)
  else Magnitude := Digits;
  if not IsDigitText(Magnitude) then
    raise ENeuralG2PError.Create('Not an integer: "' + Digits + '"');
  Magnitude := StripLeadingZeros(Magnitude);
  if Length(Magnitude) > csMaxCardinalDigits then
    raise ENeuralG2PError.Create('Number too large for words (10^306 and ' +
      'above): ' + Digits);
end;

// num2words splitnum + clean + merge on a digit string without leading zeros:
// largest card (100, 1000, 10^6, ...) times its multiplier, then the rest
// after "and" (below 100) or ", ".
function CardinalWordsOfDigits(const Digits: string): string;
var
  DigitCount, CardExponent: integer;
  Rest: string;
begin
  DigitCount := Length(Digits);
  if DigitCount <= 2 then Exit(SmallCardinalWords(StrToInt(Digits)));
  if DigitCount = 3 then CardExponent := 2
  else CardExponent := 3 * ((DigitCount - 1) div 3);
  Result := CardinalWordsOfDigits(Copy(Digits, 1, DigitCount - CardExponent));
  case CardExponent of
    2: Result := Result + ' hundred';
    3: Result := Result + ' thousand';
  else
    Result := Result + ' ' + IllionWord(CardExponent);
  end;
  Rest := StripLeadingZeros(Copy(Digits, DigitCount - CardExponent + 1,
    CardExponent));
  if Rest = '0' then Exit;
  if Length(Rest) <= 2 then Result := Result + ' and '
  else Result := Result + ', ';
  Result := Result + CardinalWordsOfDigits(Rest);
end;

function EnglishCardinalWords(const Digits: string): string;
var
  IsNegative: boolean;
  Magnitude: string;
begin
  ParseSignedDigits(Digits, IsNegative, Magnitude);
  Result := CardinalWordsOfDigits(Magnitude);
  if IsNegative and (Magnitude <> '0') then Result := 'minus ' + Result;
end;

function EnglishOrdinalWords(const Digits: string): string;
const
  cOrdinalExceptions: array[0..11, 0..1] of string = (('one', 'first'),
    ('two', 'second'), ('three', 'third'), ('four', 'fourth'),
    ('five', 'fifth'), ('six', 'sixth'), ('seven', 'seventh'),
    ('eight', 'eighth'), ('nine', 'ninth'), ('ten', 'tenth'),
    ('eleven', 'eleventh'), ('twelve', 'twelfth'));
var
  IsNegative: boolean;
  Magnitude, LastWord, Ordinal: string;
  LastWordStart, ExceptionCnt: integer;
begin
  ParseSignedDigits(Digits, IsNegative, Magnitude);
  if IsNegative then
    raise ENeuralG2PError.Create('Cannot treat negative num ' + Digits +
      ' as ordinal.');
  Result := CardinalWordsOfDigits(Magnitude);
  LastWordStart := Length(Result);
  while (LastWordStart > 1) and not (Result[LastWordStart - 1] in [' ', '-']) do
    Dec(LastWordStart);
  LastWord := Copy(Result, LastWordStart, MaxInt);
  Ordinal := '';
  for ExceptionCnt := 0 to High(cOrdinalExceptions) do
    if cOrdinalExceptions[ExceptionCnt, 0] = LastWord then
      Ordinal := cOrdinalExceptions[ExceptionCnt, 1];
  if Ordinal = '' then
  begin
    if LastWord[Length(LastWord)] = 'y' then
      Ordinal := DropBytes(LastWord, 1) + 'ieth'
    else Ordinal := LastWord + 'th';
  end;
  Result := Copy(Result, 1, LastWordStart - 1) + Ordinal;
end;

function EnglishYearWords(const Digits: string): string;
var
  IsNegative: boolean;
  Magnitude, Century: string;
  YearInCentury, CenturyLength: integer;
begin
  ParseSignedDigits(Digits, IsNegative, Magnitude);
  if Length(Magnitude) <= 2 then Century := '0'
  else Century := Copy(Magnitude, 1, Length(Magnitude) - 2);
  YearInCentury := StrToInt(Copy(Magnitude, Length(Magnitude) - 1, 2));
  CenturyLength := Length(Century);
  // 00XX, X00X and five or more digits read as a cardinal
  if (Century = '0') or ((Century[CenturyLength] = '0') and
    (YearInCentury < 10)) or (CenturyLength >= 3) then
    Result := CardinalWordsOfDigits(Magnitude)
  else
  begin
    Result := CardinalWordsOfDigits(Century);
    if YearInCentury = 0 then Result := Result + ' hundred'
    else if YearInCentury < 10 then
      Result := Result + ' oh-' + SmallCardinalWords(YearInCentury)
    else Result := Result + ' ' + SmallCardinalWords(YearInCentury);
  end;
  if IsNegative then Result := Result + ' BC';
end;

function EnglishDecimalWords(const Number: string): string;
var
  DotPos, FractionLength, DigitCnt: integer;
  IntegerPart, Fraction: string;
  IsNegative: boolean;
begin
  DotPos := Pos('.', Number);
  if DotPos = 0 then Exit(EnglishCardinalWords(Number));
  IntegerPart := Copy(Number, 1, DotPos - 1);
  Fraction := Copy(Number, DotPos + 1, MaxInt);
  IsNegative := (IntegerPart <> '') and (IntegerPart[1] = '-');
  if IsNegative then Delete(IntegerPart, 1, 1);
  if ((IntegerPart = '') and (Fraction = '')) or
    ((IntegerPart <> '') and not IsDigitText(IntegerPart)) or
    ((Fraction <> '') and not IsDigitText(Fraction)) then
    raise ENeuralG2PError.Create('Not a decimal number: "' + Number + '"');
  if IntegerPart = '' then IntegerPart := '0';
  FractionLength := Length(Fraction);
  while (FractionLength > 0) and (Fraction[FractionLength] = '0') do
    Dec(FractionLength);
  IntegerPart := StripLeadingZeros(IntegerPart);
  // Python's int() keeps the sign of -3.5 but not of -0.5
  if IsNegative and ((FractionLength = 0) or (IntegerPart <> '0')) then
    IntegerPart := '-' + IntegerPart;
  Result := EnglishCardinalWords(IntegerPart);
  if FractionLength = 0 then Exit;
  Result := Result + ' point';
  for DigitCnt := 1 to FractionLength do
    Result := Result + ' ' + cLowNumberWords[Ord(Fraction[DigitCnt]) - Ord('0')];
end;

function ApplyG2PStress(const Phonemes: string; Stress: single): string;
var
  HasPrimary, HasSecondary: boolean;
begin
  if Stress = csG2PNoStress then Exit(Phonemes);
  HasPrimary := Pos(csPrimaryStress, Phonemes) > 0;
  HasSecondary := Pos(csSecondaryStress, Phonemes) > 0;
  if Stress < -1 then
    Result := StringReplace(StringReplace(Phonemes, csPrimaryStress, '',
      [rfReplaceAll]), csSecondaryStress, '', [rfReplaceAll])
  else if (Stress = -1) or (((Stress = 0) or (Stress = -0.5)) and HasPrimary) then
    Result := StringReplace(StringReplace(Phonemes, csSecondaryStress, '',
      [rfReplaceAll]), csPrimaryStress, csSecondaryStress, [rfReplaceAll])
  else if ((Stress = 0) or (Stress = 0.5) or (Stress = 1)) and
    not HasPrimary and not HasSecondary then
  begin
    if HasVowel(Phonemes) then
      Result := InsertBeforeFirstVowel(Phonemes, csSecondaryStress)
    else Result := Phonemes;
  end
  else if (Stress >= 1) and not HasPrimary and HasSecondary then
    Result := StringReplace(Phonemes, csSecondaryStress, csPrimaryStress,
      [rfReplaceAll])
  else if (Stress > 1) and not HasPrimary and not HasSecondary then
  begin
    if HasVowel(Phonemes) then
      Result := InsertBeforeFirstVowel(Phonemes, csPrimaryStress)
    else Result := Phonemes;
  end
  else Result := Phonemes;
end;

function EnglishG2PDictionaryFileName(Accent: TNeuralG2PAccent;
  IsGold: boolean): string;
begin
  if Accent = g2pBritish then Result := 'gb_' else Result := 'us_';
  if IsGold then Result := Result + 'gold.json'
  else Result := Result + 'silver.json';
end;

function EnglishG2PAccentFromVoice(const VoiceName: string): TNeuralG2PAccent;
begin
  if (VoiceName <> '') and (VoiceName[1] = 'a') then Result := g2pAmerican
  else if (VoiceName <> '') and (VoiceName[1] = 'b') then Result := g2pBritish
  else
    raise ENeuralG2PError.Create('No English G2P for voice "' + VoiceName +
      '": only American (a*) and British (b*) voices have a text front end; ' +
      'pass phonemes instead.');
end;

function G2PResolutionName(Resolution: TNeuralG2PResolution): string;
begin
  case Resolution of
    g2prPunctuation: Result := 'punctuation';
    g2prDictionary: Result := 'dictionary';
    g2prContextRule: Result := 'context rule';
    g2prPOSRule: Result := 'POS rule';
    g2prStemRule: Result := 'stem rule';
    g2prSpelled: Result := 'spelled';
    g2prUnknown: Result := 'unknown';
    g2prNumber: Result := 'number';
  else
    Result := '?';
  end;
end;

function ClosedClassTag(const LowerWord: string): string;
var
  CueCnt, MaxCuePos: integer;
begin
  MaxCuePos := High(cClosedClassTags);
  for CueCnt := 0 to MaxCuePos do
    if cClosedClassTags[CueCnt].Word = LowerWord then
      Exit(cClosedClassTags[CueCnt].Tag);
  Result := '';
end;

// The word before a contraction ending: that'll -> that, it's -> it.
function StripClitic(const LowerWord: string): string;
const
  cEndings: array[0..6] of string = ('n''t', '''s', '''re', '''ve', '''ll',
    '''d', '''m');
var
  EndingCnt: integer;
begin
  for EndingCnt := 0 to High(cEndings) do
    if (Length(LowerWord) > Length(cEndings[EndingCnt])) and
      EndsWith(LowerWord, cEndings[EndingCnt]) then
      Exit(DropBytes(LowerWord, Length(cEndings[EndingCnt])));
  Result := LowerWord;
end;

{ ---------------------------------------------------------------- }
{ misaki subtokenize                                                }
{ ---------------------------------------------------------------- }

// Regex: ^['‘’]+ | \p{Lu}(?=\p{Lu}\p{Ll}) | (?:^-)?(?:\d?[,.]?\d)+ | [-_]+ |
// ['‘’]{2,} | \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu}) | \p{L}+(?:['‘’]\p{L})*
// | [^-_\p{L}'‘’\d] | ['‘’]+$ ; tried in this order at each position.
procedure G2PSubTokenize(const Word: string; Pieces: TStrings);
var
  CPs: array of cardinal;
  Starts: array of integer;  // byte start of each code point, plus the end
  CPCount, Position, CPPos, MatchLen: integer;

  function CPAt(Index: integer): cardinal;
  begin
    if (Index >= 0) and (Index < CPCount) then Result := CPs[Index]
    else Result := 0;
  end;

  function IsLetterAt(Index: integer): boolean;
  begin
    Result := (Index < CPCount) and IsLetterCP(CPs[Index]);
  end;

  function IsApostropheAt(Index: integer): boolean;
  begin
    Result := (Index < CPCount) and IsApostropheCP(CPs[Index]);
  end;

  function IsDigitAt(Index: integer): boolean;
  begin
    Result := (Index < CPCount) and IsDigitCP(CPs[Index]);
  end;

  // One (?:\d?[,.]?\d) repetition at Index; returns its length or 0.
  function NumberRep(Index: integer): integer;
  var
    IsSepNext: boolean;
  begin
    if IsDigitAt(Index) then
    begin
      IsSepNext := (CPAt(Index + 1) = Ord(',')) or (CPAt(Index + 1) = Ord('.'));
      if IsSepNext and IsDigitAt(Index + 2) then Exit(3);
      if IsDigitAt(Index + 1) then Exit(2);
    end;
    if ((CPAt(Index) = Ord(',')) or (CPAt(Index) = Ord('.'))) and
      IsDigitAt(Index + 1) then Exit(2);
    if IsDigitAt(Index) then Exit(1);
    Result := 0;
  end;

  function MatchAt(P: integer): integer;
  var
    Q, RepLen, LetterRun, PairPos: integer;
  begin
    // ^['‘’]+
    if (P = 0) and IsApostropheAt(0) then
    begin
      Q := 0;
      while IsApostropheAt(Q) do Inc(Q);
      Exit(Q);
    end;
    // \p{Lu}(?=\p{Lu}\p{Ll})
    if IsUpperLetterCP(CPAt(P)) and IsUpperLetterCP(CPAt(P + 1)) and
      (P + 2 < CPCount) and IsLowerLetterCP(CPs[P + 2]) then Exit(1);
    // (?:^-)?(?:\d?[,.]?\d)+
    Q := P;
    if (P = 0) and (CPAt(0) = Ord('-')) and (NumberRep(1) > 0) then Q := 1;
    RepLen := NumberRep(Q);
    if RepLen > 0 then
    begin
      while RepLen > 0 do
      begin
        Inc(Q, RepLen);
        RepLen := NumberRep(Q);
      end;
      Exit(Q - P);
    end;
    // [-_]+
    if (CPAt(P) = Ord('-')) or (CPAt(P) = Ord('_')) then
    begin
      Q := P;
      while (Q < CPCount) and ((CPs[Q] = Ord('-')) or (CPs[Q] = Ord('_'))) do
        Inc(Q);
      Exit(Q - P);
    end;
    // ['‘’]{2,}
    if IsApostropheAt(P) and IsApostropheAt(P + 1) then
    begin
      Q := P;
      while IsApostropheAt(Q) do Inc(Q);
      Exit(Q - P);
    end;
    // \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu}): fewest letters, then fewest
    // apostrophe-letter pairs.
    LetterRun := 0;
    while IsLetterAt(P + LetterRun) do Inc(LetterRun);
    for Q := P to P + LetterRun do
    begin
      PairPos := Q;
      while true do
      begin
        if (PairPos < CPCount) and IsLowerLetterCP(CPs[PairPos]) and
          IsUpperLetterCP(CPAt(PairPos + 1)) and (PairPos + 1 < CPCount) then
          Exit(PairPos + 1 - P);
        if IsApostropheAt(PairPos) and IsLetterAt(PairPos + 1) then
          Inc(PairPos, 2)
        else Break;
      end;
    end;
    // \p{L}+(?:['‘’]\p{L})*
    if IsLetterAt(P) then
    begin
      Q := P;
      while IsLetterAt(Q) do Inc(Q);
      while IsApostropheAt(Q) and IsLetterAt(Q + 1) do Inc(Q, 2);
      Exit(Q - P);
    end;
    // [^-_\p{L}'‘’\d]
    if not ((CPAt(P) = Ord('-')) or (CPAt(P) = Ord('_')) or IsLetterAt(P) or
      IsApostropheAt(P) or IsDigitAt(P)) then Exit(1);
    // ['‘’]+$
    if IsApostropheAt(P) then
    begin
      Q := P;
      while IsApostropheAt(Q) do Inc(Q);
      if Q = CPCount then Exit(Q - P);
    end;
    Result := 0;
  end;

begin
  Pieces.Clear;
  SetLength(CPs, Length(Word));
  SetLength(Starts, Length(Word) + 1);
  CPCount := 0;
  Position := 1;
  while Position <= Length(Word) do
  begin
    Starts[CPCount] := Position;
    CPs[CPCount] := NextCodePoint(Word, Position);
    Inc(CPCount);
  end;
  Starts[CPCount] := Position;
  CPPos := 0;
  while CPPos < CPCount do
  begin
    MatchLen := MatchAt(CPPos);
    if MatchLen > 0 then
    begin
      Pieces.Add(Copy(Word, Starts[CPPos], Starts[CPPos + MatchLen] -
        Starts[CPPos]));
      Inc(CPPos, MatchLen);
    end
    else Inc(CPPos);
  end;
end;

{ ---------------------------------------------------------------- }
{ TNeuralG2PDictionary                                              }
{ ---------------------------------------------------------------- }

function G2PKeyHash(P: PAnsiChar; Len: integer): cardinal;
var
  ByteCnt, MaxBytePos: integer;
begin
  {$PUSH}{$Q-}{$R-}
  Result := 2166136261;
  MaxBytePos := Len - 1;
  for ByteCnt := 0 to MaxBytePos do
    Result := (Result xor cardinal(byte(P[ByteCnt]))) * 16777619;
  {$POP}
end;

procedure TNeuralG2PDictionary.RaiseParseError(Position: integer;
  const Expected: string);
begin
  raise ENeuralG2PError.Create('G2P dictionary ' + FSourceName +
    ': expected ' + Expected + ' at byte ' + IntToStr(Position) +
    ' (misaki dictionary JSON: {"word": "phonemes" | null | {"TAG": ...}})');
end;

procedure TNeuralG2PDictionary.LoadFromFile(const FileName: string);
var
  Stream: TFileStream;
  Text: string;
begin
  if not FileExists(FileName) then
    raise ENeuralG2PError.Create('G2P dictionary not found: ' + FileName);
  Stream := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    SetLength(Text, Stream.Size);
    if Length(Text) > 0 then Stream.ReadBuffer(Text[1], Length(Text));
  finally
    Stream.Free;
  end;
  LoadFromString(Text, FileName);
end;

procedure TNeuralG2PDictionary.LoadFromString(const JSONText: string;
  const SourceName: string);
begin
  FText := JSONText;
  UniqueString(FText); // ScanString decodes escapes in place
  if SourceName <> '' then FSourceName := SourceName
  else FSourceName := '(string)';
  FEntryCount := 0;
  FVariantCount := 0;
  ParseText();
  BuildIndex();
end;

// Reads the JSON string at Position; escapes are decoded in place (a decoded
// string is never longer than its escaped form). Position ends past the quote.
procedure TNeuralG2PDictionary.ScanString(var Position: integer;
  out Start, Len: integer);
var
  ReadPos, WritePos, TextLen, ByteCnt, EncodedLen: integer;
  C: char;
  CP, LowSurrogate: cardinal;
  Encoded: string;

  function HexAt(At: integer): cardinal;
  var
    DigitCnt: integer;
    Digit: char;
  begin
    Result := 0;
    if At + 3 > TextLen then RaiseParseError(At, 'four hex digits');
    for DigitCnt := 0 to 3 do
    begin
      Digit := FText[At + DigitCnt];
      case Digit of
        '0'..'9': Result := Result * 16 + cardinal(Ord(Digit) - Ord('0'));
        'a'..'f': Result := Result * 16 + cardinal(Ord(Digit) - Ord('a') + 10);
        'A'..'F': Result := Result * 16 + cardinal(Ord(Digit) - Ord('A') + 10);
      else
        RaiseParseError(At + DigitCnt, 'a hex digit');
      end;
    end;
  end;

begin
  TextLen := Length(FText);
  if (Position > TextLen) or (FText[Position] <> '"') then
    RaiseParseError(Position, 'a string');
  Start := Position + 1;
  ReadPos := Start;
  WritePos := Start;
  while true do
  begin
    if ReadPos > TextLen then RaiseParseError(ReadPos, 'a closing quote');
    C := FText[ReadPos];
    if C = '"' then Break;
    if C <> '\' then
    begin
      if WritePos <> ReadPos then FText[WritePos] := C;
      Inc(WritePos);
      Inc(ReadPos);
      Continue;
    end;
    if ReadPos + 1 > TextLen then RaiseParseError(ReadPos, 'an escape');
    C := FText[ReadPos + 1];
    Inc(ReadPos, 2);
    case C of
      '"', '\', '/': Encoded := C;
      'b': Encoded := #8;
      'f': Encoded := #12;
      'n': Encoded := #10;
      'r': Encoded := #13;
      't': Encoded := #9;
      'u':
        begin
          CP := HexAt(ReadPos);
          Inc(ReadPos, 4);
          if (CP >= $D800) and (CP <= $DBFF) and (ReadPos + 5 <= TextLen) and
            (FText[ReadPos] = '\') and (FText[ReadPos + 1] = 'u') then
          begin
            LowSurrogate := HexAt(ReadPos + 2);
            if (LowSurrogate >= $DC00) and (LowSurrogate <= $DFFF) then
            begin
              CP := $10000 + ((CP - $D800) shl 10) + (LowSurrogate - $DC00);
              Inc(ReadPos, 6);
            end;
          end;
          Encoded := CodePointToUTF8(CP);
        end;
    else
      RaiseParseError(ReadPos - 1, 'a JSON escape');
    end;
    EncodedLen := Length(Encoded);
    for ByteCnt := 1 to EncodedLen do
    begin
      FText[WritePos] := Encoded[ByteCnt];
      Inc(WritePos);
    end;
  end;
  Len := WritePos - Start;
  Position := ReadPos + 1;
end;

procedure TNeuralG2PDictionary.ParseText();
var
  Position, TextLen, TagStart, TagLen: integer;
  Entry: TNeuralG2PDictionaryEntry;
  Variant: TNeuralG2PDictionaryVariant;

  procedure SkipSpace;
  begin
    while (Position <= TextLen) and (FText[Position] in [' ', #9, #10, #13]) do
      Inc(Position);
  end;

  procedure Expect(C: char);
  begin
    SkipSpace;
    if (Position > TextLen) or (FText[Position] <> C) then
      RaiseParseError(Position, '''' + C + '''');
    Inc(Position);
    SkipSpace;
  end;

  // A string or null value; returns false for null.
  function ScanValue(out Start, Len: integer): boolean;
  begin
    if (Position <= TextLen) and (FText[Position] = '"') then
    begin
      ScanString(Position, Start, Len);
      Exit(true);
    end;
    if (Position + 3 <= TextLen) and (Copy(FText, Position, 4) = 'null') then
    begin
      Inc(Position, 4);
      Start := 0;
      Len := 0;
      Exit(false);
    end;
    RaiseParseError(Position, 'a string or null');
    Result := false;
  end;

begin
  TextLen := Length(FText);
  Position := 1;
  Expect('{');
  if (Position <= TextLen) and (FText[Position] = '}') then Exit;
  while true do
  begin
    FillChar(Entry, SizeOf(Entry), 0);
    ScanString(Position, Entry.KeyStart, Entry.KeyLength);
    Expect(':');
    if (Position <= TextLen) and (FText[Position] = '{') then
    begin
      Inc(Position);
      SkipSpace;
      Entry.FirstVariant := FVariantCount;
      Entry.ValueIsNull := true;
      while (Position <= TextLen) and (FText[Position] <> '}') do
      begin
        ScanString(Position, TagStart, TagLen);
        Variant.TagStart := TagStart;
        Variant.TagLength := TagLen;
        Expect(':');
        Variant.ValueIsNull := not ScanValue(Variant.ValueStart,
          Variant.ValueLength);
        if FVariantCount = Length(FVariants) then
          SetLength(FVariants, FVariantCount * 2 + 64);
        FVariants[FVariantCount] := Variant;
        Inc(FVariantCount);
        if SpanEquals(TagStart, TagLen, 'DEFAULT') then
        begin
          Entry.ValueStart := Variant.ValueStart;
          Entry.ValueLength := Variant.ValueLength;
          Entry.ValueIsNull := Variant.ValueIsNull;
        end;
        SkipSpace;
        if (Position <= TextLen) and (FText[Position] = ',') then
        begin
          Inc(Position);
          SkipSpace;
        end;
      end;
      Expect('}');
      Entry.VariantCount := FVariantCount - Entry.FirstVariant;
    end
    else
      Entry.ValueIsNull := not ScanValue(Entry.ValueStart, Entry.ValueLength);
    if FEntryCount = Length(FEntries) then
      SetLength(FEntries, FEntryCount * 2 + 1024);
    FEntries[FEntryCount] := Entry;
    Inc(FEntryCount);
    SkipSpace;
    if (Position <= TextLen) and (FText[Position] = ',') then
    begin
      Inc(Position);
      SkipSpace;
      Continue;
    end;
    if (Position <= TextLen) and (FText[Position] = '}') then Break;
    RaiseParseError(Position, ''','' or ''}''');
  end;
end;

// Hashes every entry; a repeated key keeps the LAST value (Python dict).
procedure TNeuralG2PDictionary.BuildIndex();
var
  Capacity, EntryCnt, MaxEntryPos, WriteCnt, Slot, Existing: integer;
begin
  Capacity := 16;
  while Capacity < FEntryCount * 2 do Capacity := Capacity shl 1;
  SetLength(FSlots, Capacity);
  FillChar(FSlots[0], Capacity * SizeOf(integer), 0);
  FSlotMask := Capacity - 1;
  WriteCnt := 0;
  MaxEntryPos := FEntryCount - 1;
  for EntryCnt := 0 to MaxEntryPos do
  begin
    Slot := FindSlot(@FText[FEntries[EntryCnt].KeyStart],
      FEntries[EntryCnt].KeyLength);
    Existing := FSlots[Slot] - 1;
    if Existing >= 0 then
      FEntries[Existing] := FEntries[EntryCnt]
    else
    begin
      FEntries[WriteCnt] := FEntries[EntryCnt];
      FSlots[Slot] := WriteCnt + 1;
      Inc(WriteCnt);
    end;
  end;
  FEntryCount := WriteCnt;
end;

// The slot holding Key, or the empty slot where it would go.
function TNeuralG2PDictionary.FindSlot(P: PAnsiChar; Len: integer): integer;
var
  EntryIdx: integer;
begin
  Result := integer(G2PKeyHash(P, Len)) and FSlotMask;
  while true do
  begin
    EntryIdx := FSlots[Result] - 1;
    if EntryIdx < 0 then Exit;
    if (FEntries[EntryIdx].KeyLength = Len) and ((Len = 0) or
      (CompareByte(FText[FEntries[EntryIdx].KeyStart], P^, Len) = 0)) then
      Exit;
    Result := (Result + 1) and FSlotMask;
  end;
end;

function TNeuralG2PDictionary.SpanToString(Start, Len: integer): string;
begin
  Result := Copy(FText, Start, Len);
end;

function TNeuralG2PDictionary.SpanEquals(Start, Len: integer;
  const S: string): boolean;
begin
  Result := (Len = Length(S)) and ((Len = 0) or
    (CompareByte(FText[Start], S[1], Len) = 0));
end;

function TNeuralG2PDictionary.IndexOf(const Key: string): integer;
begin
  if FSlots = nil then Exit(-1);
  if Key = '' then
    Result := FSlots[FindSlot(nil, 0)] - 1
  else
    Result := FSlots[FindSlot(@Key[1], Length(Key))] - 1;
end;

function TNeuralG2PDictionary.IndexOfGrown(const Key: string): integer;
var
  LowerKey, CapitalizedKey: string;
begin
  Result := IndexOf(Key);
  if (Result >= 0) or (CodePointCount(Key) < 2) then Exit;
  LowerKey := G2PLower(Key);
  if Key = LowerKey then
  begin
    CapitalizedKey := G2PCapitalize(Key);
    if CapitalizedKey <> Key then Result := IndexOf(CapitalizedKey);
  end
  else if Key = G2PCapitalize(LowerKey) then
    Result := IndexOf(LowerKey);
end;

function TNeuralG2PDictionary.IsPOSKeyed(Index: integer): boolean;
begin
  Result := FEntries[Index].VariantCount > 0;
end;

function TNeuralG2PDictionary.VariantIndex(Index: integer;
  const Tag: string): integer;
var
  MaxVariantPos: integer;
begin
  MaxVariantPos := FEntries[Index].FirstVariant +
    FEntries[Index].VariantCount - 1;
  for Result := FEntries[Index].FirstVariant to MaxVariantPos do
    if SpanEquals(FVariants[Result].TagStart, FVariants[Result].TagLength,
      Tag) then Exit;
  Result := -1;
end;

function TNeuralG2PDictionary.HasTag(Index: integer; const Tag: string): boolean;
begin
  Result := VariantIndex(Index, Tag) >= 0;
end;

function TNeuralG2PDictionary.DefaultReading(Index: integer;
  out Phonemes: string): boolean;
begin
  Result := not FEntries[Index].ValueIsNull;
  if Result then
    Phonemes := SpanToString(FEntries[Index].ValueStart,
      FEntries[Index].ValueLength)
  else Phonemes := '';
end;

function TNeuralG2PDictionary.TagReading(Index: integer; const Tag: string;
  out Phonemes: string): boolean;
var
  VariantIdx: integer;
begin
  VariantIdx := VariantIndex(Index, Tag);
  if VariantIdx < 0 then Exit(DefaultReading(Index, Phonemes));
  Result := not FVariants[VariantIdx].ValueIsNull;
  if Result then
    Phonemes := SpanToString(FVariants[VariantIdx].ValueStart,
      FVariants[VariantIdx].ValueLength)
  else Phonemes := '';
end;

function TNeuralG2PDictionary.KeyAt(Index: integer): string;
begin
  Result := SpanToString(FEntries[Index].KeyStart, FEntries[Index].KeyLength);
end;

{ ---------------------------------------------------------------- }
{ TNeuralEnglishG2P: construction                                   }
{ ---------------------------------------------------------------- }

constructor TNeuralEnglishG2P.Create(const DictionaryDir: string;
  Accent: TNeuralG2PAccent);
var
  GoldFile, SilverFile, Missing: string;
begin
  inherited Create;
  FAccent := Accent;
  FIsBritish := Accent = g2pBritish;
  GoldFile := IncludeTrailingPathDelimiter(DictionaryDir) +
    EnglishG2PDictionaryFileName(Accent, true);
  SilverFile := IncludeTrailingPathDelimiter(DictionaryDir) +
    EnglishG2PDictionaryFileName(Accent, false);
  Missing := '';
  if not FileExists(GoldFile) then Missing := GoldFile;
  if not FileExists(SilverFile) then
  begin
    if Missing <> '' then Missing := Missing + ', ';
    Missing := Missing + SilverFile;
  end;
  if Missing <> '' then
    raise ENeuralG2PError.Create('English G2P dictionary not found: ' +
      Missing + '. Copy misaki''s data/' +
      EnglishG2PDictionaryFileName(Accent, true) + ' and ' +
      EnglishG2PDictionaryFileName(Accent, false) +
      ' (github.com/hexgrad/misaki, Apache-2.0) into that folder.');
  FGold := TNeuralG2PDictionary.Create;
  FSilver := TNeuralG2PDictionary.Create;
  FGold.LoadFromFile(GoldFile);
  FSilver.LoadFromFile(SilverFile);
end;

destructor TNeuralEnglishG2P.Destroy;
begin
  FGold.Free;
  FSilver.Free;
  inherited Destroy;
end;

{ ---------------------------------------------------------------- }
{ Tokenizer: spaCy-like tokens                                      }
{ ---------------------------------------------------------------- }

procedure TNeuralEnglishG2P.AddRawToken(const Text: string;
  IsPunctuation: boolean; const Phonemes, Tag: string);
begin
  if FRawCount = Length(FRaw) then SetLength(FRaw, FRawCount * 2 + 32);
  FRaw[FRawCount].Text := Text;
  FRaw[FRawCount].Whitespace := '';
  FRaw[FRawCount].Tag := Tag;
  FRaw[FRawCount].IsPunctuation := IsPunctuation;
  FRaw[FRawCount].Phonemes := Phonemes;
  FRaw[FRawCount].EndsSentence := IsPunctuation and
    ((Pos('.', Phonemes) > 0) or (Pos('!', Phonemes) > 0) or
    (Pos('?', Phonemes) > 0) or (Pos(#$E2#$80#$A6, Phonemes) > 0));
  Inc(FRawCount);
end;

// Phonemes misaki gives a punctuation token: brackets become ( ), quotes
// become “ ”, characters outside ;:,.!?—…"“” vanish.
function TNeuralEnglishG2P.PunctuationPhonemes(const Text: string; IsPrefix,
  IsSuffix: boolean): string;
var
  Position, TextLen: integer;
  CP: cardinal;
begin
  Result := '';
  TextLen := Length(Text);
  Position := 1;
  while Position <= TextLen do
  begin
    CP := NextCodePoint(Text, Position);
    case CP of
      Ord('('), Ord('{'): Result := Result + '(';
      Ord(')'), Ord('}'): Result := Result + ')';
      Ord('`'), cpLeftSingleQuote, cpLeftDoubleQuote:
        Result := Result + csLeftDoubleQuote;
      cpRightSingleQuote, cpRightDoubleQuote:
        Result := Result + csRightDoubleQuote;
      Ord('"'):
        begin
          if IsPrefix or (not IsSuffix and not FDoubleQuoteOpen) then
          begin
            Result := Result + csLeftDoubleQuote;
            FDoubleQuoteOpen := true;
          end
          else
          begin
            Result := Result + csRightDoubleQuote;
            FDoubleQuoteOpen := false;
          end;
        end;
      cpEnDash: Result := Result + csEmDash;
    else
      if IsNonQuotePunctCP(CP) then Result := Result + CodePointToUTF8(CP);
    end;
  end;
end;

procedure TNeuralEnglishG2P.AddPunctuationToken(const Text: string);
begin
  AddRawToken(Text, true, PunctuationPhonemes(Text, false, false), '.');
end;

// A trailing '.' stays on a dictionary abbreviation (Dr., Mr.) and on a
// dotted one (U.S., Ph.D.); spaCy keeps those whole.
function TNeuralEnglishG2P.IsAbbreviation(const Word: string): boolean;
var
  BytePos, MaxBytePos: integer;
  HasInnerDot: boolean;
begin
  if (Length(Word) < 2) or (Word[Length(Word)] <> '.') or
    (Word[Length(Word) - 1] = '.') then Exit(false);
  if (FGold.IndexOfGrown(Word) >= 0) or (FSilver.IndexOfGrown(Word) >= 0) then
    Exit(true);
  HasInnerDot := false;
  MaxBytePos := Length(Word) - 1;
  for BytePos := 1 to MaxBytePos do
  begin
    if Word[BytePos] = '.' then
    begin
      if (BytePos = 1) or (Word[BytePos - 1] = '.') then Exit(false);
      HasInnerDot := true;
    end
    else if not (Word[BytePos] in ['A'..'Z', 'a'..'z']) then Exit(false);
  end;
  Result := HasInnerDot;
end;

// A chunk without letters or digits: dashes, ellipses, quotes, symbols;
// one token per character, as spaCy splits "..." and "?!".
procedure TNeuralEnglishG2P.ProcessPunctuationChunk(const Chunk: string);
var
  Position, Start, TextLen: integer;
begin
  // misaki/spaCy: a spaced '-' and '--' give no phonemes, a spaced en dash
  // gives an em dash; % & + @ are words (percent, and, plus, at).
  if (Chunk = '-') or (Chunk = '--') then
  begin
    AddRawToken(Chunk, true, '', ',');
    Exit;
  end;
  if SymbolWord(Chunk) <> '' then
  begin
    AddRawToken(Chunk, false, '', 'SYM');
    Exit;
  end;
  TextLen := Length(Chunk);
  Position := 1;
  while Position <= TextLen do
  begin
    Start := Position;
    NextCodePoint(Chunk, Position);
    AddPunctuationToken(Copy(Chunk, Start, Position - Start));
  end;
end;

// Splits a word core at '--', '—' or '–' between letters or digits
// (spaCy infixes); every other character stays in the word.
procedure TNeuralEnglishG2P.ProcessCore(const Core: string);
var
  Position, Start, TextLen, WordStart, DashLen: integer;
  CP, PrevCP, NextCP: cardinal;
  Probe: integer;
begin
  TextLen := Length(Core);
  Position := 1;
  WordStart := 1;
  PrevCP := 0;
  while Position <= TextLen do
  begin
    Start := Position;
    CP := NextCodePoint(Core, Position);
    DashLen := 0;
    if (CP = cpEmDash) or (CP = cpEnDash) then DashLen := Position - Start
    else if (CP = Ord('-')) and (Position <= TextLen) and
      (Core[Position] = '-') then DashLen := 2;
    if (DashLen > 0) and (Start > WordStart) and
      (IsLetterCP(PrevCP) or IsDigitCP(PrevCP)) and
      (Start + DashLen <= TextLen) then
    begin
      Probe := Start + DashLen;
      NextCP := NextCodePoint(Core, Probe);
      if IsLetterCP(NextCP) or IsDigitCP(NextCP) then
      begin
        AddRawToken(Copy(Core, WordStart, Start - WordStart), false, '', '');
        if DashLen = 2 then AddRawToken('--', true, '', ':')
        else AddRawToken(Copy(Core, Start, DashLen), true, csEmDash, ':');
        Position := Start + DashLen;
        WordStart := Position;
        PrevCP := 0;
        Continue;
      end;
    end;
    PrevCP := CP;
  end;
  if WordStart <= TextLen then
    AddRawToken(Copy(Core, WordStart, MaxInt), false, '', '');
end;

procedure TNeuralEnglishG2P.ProcessChunk(const Chunk, Whitespace: string);
var
  CPs: array of cardinal;
  Starts: array of integer;
  CPCount, Position, FirstCPPos, LastCPPos, SuffixCount, SuffixCnt: integer;
  HasWordChar: boolean;
  SuffixTexts: array of string;
  SuffixIsSymbol: array of boolean;
  CP: cardinal;
  Core: string;

  function SpanText(FromCP, ToCP: integer): string;
  begin
    Result := Copy(Chunk, Starts[FromCP], Starts[ToCP + 1] - Starts[FromCP]);
  end;

  procedure PushSuffix(const Text: string; IsSymbol: boolean);
  begin
    if SuffixCount = Length(SuffixTexts) then
    begin
      SetLength(SuffixTexts, SuffixCount * 2 + 4);
      SetLength(SuffixIsSymbol, SuffixCount * 2 + 4);
    end;
    SuffixTexts[SuffixCount] := Text;
    SuffixIsSymbol[SuffixCount] := IsSymbol;
    Inc(SuffixCount);
  end;

begin
  SetLength(CPs, Length(Chunk));
  SetLength(Starts, Length(Chunk) + 1);
  CPCount := 0;
  Position := 1;
  HasWordChar := false;
  while Position <= Length(Chunk) do
  begin
    Starts[CPCount] := Position;
    CPs[CPCount] := NextCodePoint(Chunk, Position);
    if IsLetterCP(CPs[CPCount]) or IsDigitCP(CPs[CPCount]) then
      HasWordChar := true;
    Inc(CPCount);
  end;
  Starts[CPCount] := Position;
  if not HasWordChar then
  begin
    ProcessPunctuationChunk(Chunk);
    if FRawCount > 0 then FRaw[FRawCount - 1].Whitespace := Whitespace;
    Exit;
  end;
  FirstCPPos := 0;
  LastCPPos := CPCount - 1;
  // prefixes: ( { " “ ‘ ` / and currency signs
  while FirstCPPos < LastCPPos do
  begin
    CP := CPs[FirstCPPos];
    case CP of
      Ord('('), Ord('{'), Ord('"'), Ord('`'), cpLeftDoubleQuote:
        AddRawToken(SpanText(FirstCPPos, FirstCPPos), true,
          PunctuationPhonemes(SpanText(FirstCPPos, FirstCPPos), true, false), '``');
      cpLeftSingleQuote:
        begin
          AddRawToken(SpanText(FirstCPPos, FirstCPPos), true, csLeftDoubleQuote, '``');
          FSingleQuoteOpen := true;
        end;
      Ord('$'), $A3, $20AC, $A5:
        AddRawToken(SpanText(FirstCPPos, FirstCPPos), true, '', '$');
      Ord('/'):
        AddRawToken('/', true, '', 'NFP');
      cpEmDash, cpEnDash:
        AddRawToken(SpanText(FirstCPPos, FirstCPPos), true, csEmDash, ':');
    else
      Break;
    end;
    Inc(FirstCPPos);
  end;
  // suffixes, collected right to left
  SuffixCount := 0;
  while LastCPPos > FirstCPPos do
  begin
    CP := CPs[LastCPPos];
    if CP = Ord('.') then
    begin
      // a dot run (ellipsis) splits dot by dot; a single dot may end an
      // abbreviation
      if (CPs[LastCPPos - 1] <> Ord('.')) and
        IsAbbreviation(SpanText(FirstCPPos, LastCPPos)) then Break;
      PushSuffix('.', false);
    end
    else if (CP = cpRightSingleQuote) then
    begin
      if not FSingleQuoteOpen then Break;
      PushSuffix(SpanText(LastCPPos, LastCPPos), false);
      FSingleQuoteOpen := false;
    end
    else if CP = Ord('%') then
      PushSuffix('%', true)
    else if IsNonQuotePunctCP(CP) or (CP = Ord(')')) or (CP = Ord('}')) or
      (CP = Ord('"')) or (CP = cpRightDoubleQuote) or (CP = cpEnDash) then
      PushSuffix(SpanText(LastCPPos, LastCPPos), false)
    else Break;
    Dec(LastCPPos);
  end;
  Core := SpanText(FirstCPPos, LastCPPos);
  ProcessCore(Core);
  for SuffixCnt := SuffixCount - 1 downto 0 do
    if SuffixIsSymbol[SuffixCnt] then
      AddRawToken(SuffixTexts[SuffixCnt], false, '', 'NN')
    else
      AddRawToken(SuffixTexts[SuffixCnt], true,
        PunctuationPhonemes(SuffixTexts[SuffixCnt], false, true), '.');
  FRaw[FRawCount - 1].Whitespace := Whitespace;
end;

procedure TNeuralEnglishG2P.SplitIntoRawTokens(const Text: string);
var
  Position, Start, ChunkStart, TextLen, ChunkEnd: integer;
  Whitespace: string;
begin
  FRawCount := 0;
  FDoubleQuoteOpen := false;
  FSingleQuoteOpen := false;
  TextLen := Length(Text);
  Position := 1;
  // leading whitespace (misaki lstrip)
  while Position <= TextLen do
  begin
    Start := Position;
    if not IsWhitespaceCP(NextCodePoint(Text, Position)) then
    begin
      Position := Start;
      Break;
    end;
  end;
  while Position <= TextLen do
  begin
    ChunkStart := Position;
    ChunkEnd := Position;
    while ChunkEnd <= TextLen do
    begin
      Position := ChunkEnd;
      if IsWhitespaceCP(NextCodePoint(Text, Position)) then Break;
      ChunkEnd := Position;
    end;
    // ChunkEnd = first whitespace byte (or past the end)
    Whitespace := '';
    Position := ChunkEnd;
    if Position <= TextLen then
    begin
      NextCodePoint(Text, Position);
      // spaCy keeps one whitespace character per token; a longer run is a
      // whitespace token that misaki reads as ''.
      Whitespace := Copy(Text, ChunkEnd, Position - ChunkEnd);
      while Position <= TextLen do
      begin
        Start := Position;
        if not IsWhitespaceCP(NextCodePoint(Text, Position)) then
        begin
          Position := Start;
          Break;
        end;
      end;
    end;
    ProcessChunk(Copy(Text, ChunkStart, ChunkEnd - ChunkStart), Whitespace);
  end;
end;

{ ---------------------------------------------------------------- }
{ Rule tagger (replaces spaCy)                                      }
{ ---------------------------------------------------------------- }

function TNeuralEnglishG2P.PreviousWord(Index: integer; out Word: string): boolean;
begin
  Word := '';
  Result := (Index > 0) and not FRaw[Index - 1].IsPunctuation;
  if Result then Word := G2PLower(ReplaceCurlyApostrophes(FRaw[Index - 1].Text));
end;

function TNeuralEnglishG2P.NextWord(Index: integer; out Word: string): boolean;
begin
  Word := '';
  Result := (Index + 1 < FRawCount) and not FRaw[Index + 1].IsPunctuation;
  if Result then Word := G2PLower(ReplaceCurlyApostrophes(FRaw[Index + 1].Text));
end;

function TNeuralEnglishG2P.TagWord(Index: integer; IsShouting,
  IsSentenceStart: boolean): string;
var
  Word, Lower, Base, Previous, Next: string;
  HasPrevious, HasNext: boolean;
  LetterCount, Position, TextLen, CueCnt, MaxCuePos: integer;
  CP: cardinal;
begin
  Word := ReplaceCurlyApostrophes(FRaw[Index].Text);
  Lower := G2PLower(Word);
  HasPrevious := PreviousWord(Index, Previous);
  HasNext := NextWord(Index, Next);
  if Pos('@', Word) > 0 then Exit('ADD');
  if IsNumberWord(Word, true) and (Word[Length(Word)] in ['0'..'9', ',', '.']) then
    Exit('CD');
  LetterCount := 0;
  TextLen := Length(Word);
  Position := 1;
  while Position <= TextLen do
  begin
    CP := NextCodePoint(Word, Position);
    if IsLetterCP(CP) then Inc(LetterCount);
  end;
  // acronyms: ALL-CAPS words are proper nouns unless the sentence shouts
  if (LetterCount >= 2) and not IsShouting and (G2PUpper(Word) = Word) then
    Exit('NNP');
  // closed-class words misaki reads by tag (a contraction tags as its
  // first word: that'll -> that)
  Base := StripClitic(Lower);
  if Base = 'a' then
  begin
    // "A." and "A + B" name the letter
    if (Word = 'A') and ((HasPrevious and not HasNext) or
      (SymbolWord(Next) <> '')) then Exit('NNP');
    Exit('DT');
  end;
  if Word = 'I' then Exit('PRP');
  if Base = 'used' then
  begin
    if Next = 'to' then Exit('VBD');
    Exit('VBN');
  end;
  if Base = 'that' then
  begin
    if HasNext and HasPrevious and InStringArray(Next, cThatComplementNext) then
      Exit('IN');
    Exit('DT');
  end;
  Result := ClosedClassTag(Base);
  if Result <> '' then Exit;
  if InStringArray(Lower, cLowNumberWords) or
    InStringArray(Lower, cTensNumberWords) or InStringArray(Lower,
    ['hundred', 'thousand', 'million', 'billion', 'trillion']) then Exit('CD');
  // Capitalized words inside a sentence are proper nouns
  if not IsSentenceStart and (LetterCount >= 2) and
    (G2PCapitalize(Word) = Word) and (G2PLower(Word) <> Word) then
    Exit('NNP');
  // POS-keyed readings: previous-word cues, then next-word cues
  if HasPrevious then
  begin
    MaxCuePos := High(cPreviousWordCues);
    for CueCnt := 0 to MaxCuePos do
      if cPreviousWordCues[CueCnt].Word = Previous then
      begin
        Result := cPreviousWordCues[CueCnt].Tag;
        if (Result = 'VBD') and EndsWith(Lower, 's') then Result := 'VBZ';
        Exit;
      end;
    MaxCuePos := High(cPreviousSuffixCues);
    for CueCnt := 0 to MaxCuePos do
      if EndsWith(Previous, cPreviousSuffixCues[CueCnt].Word) then
        Exit(cPreviousSuffixCues[CueCnt].Tag);
    if (Previous = 'that') and (FRaw[Index - 1].Tag = 'DT') then Exit('NN');
  end;
  if HasNext and InStringArray(Next, cObjectPronouns) then Exit('VB');
  if HasNext and InStringArray(Next, cDeterminers) and
    (not HasPrevious or InStringArray(Previous, cConjunctions)) then
    Exit('VB');
  if HasPrevious and not HasNext then Exit('NN');
  if (Length(Lower) > 3) and EndsWith(Lower, 'ed') then Exit('VBD');
  if (Length(Lower) > 4) and EndsWith(Lower, 'ing') then Exit('VBG');
  Result := '';
end;

// Tags every word token; a sentence whose multi-letter words are all
// upper case "shouts", and its words are read as ordinary words.
procedure TNeuralEnglishG2P.TagRawTokens();
var
  SentenceStart, SentenceEnd, TokenCnt, MultiLetterWords: integer;
  IsShouting, AllUpper, SeenWord: boolean;
  Position, TextLen, LetterCount: integer;
begin
  SentenceStart := 0;
  while SentenceStart < FRawCount do
  begin
    SentenceEnd := SentenceStart;
    while (SentenceEnd < FRawCount - 1) and not FRaw[SentenceEnd].EndsSentence do
      Inc(SentenceEnd);
    MultiLetterWords := 0;
    AllUpper := true;
    for TokenCnt := SentenceStart to SentenceEnd do
      if not FRaw[TokenCnt].IsPunctuation then
      begin
        LetterCount := 0;
        TextLen := Length(FRaw[TokenCnt].Text);
        Position := 1;
        while Position <= TextLen do
          if IsLetterCP(NextCodePoint(FRaw[TokenCnt].Text, Position)) then
            Inc(LetterCount);
        if LetterCount >= 2 then
        begin
          Inc(MultiLetterWords);
          if G2PUpper(FRaw[TokenCnt].Text) <> FRaw[TokenCnt].Text then
            AllUpper := false;
        end;
      end;
    IsShouting := (MultiLetterWords >= 2) and AllUpper;
    SeenWord := false;
    for TokenCnt := SentenceStart to SentenceEnd do
      if not FRaw[TokenCnt].IsPunctuation then
      begin
        if FRaw[TokenCnt].Tag = '' then
          FRaw[TokenCnt].Tag := TagWord(TokenCnt, IsShouting, not SeenWord);
        SeenWord := true;
      end;
    SentenceStart := SentenceEnd + 1;
  end;
end;

{ ---------------------------------------------------------------- }
{ Pieces and groups (misaki retokenize)                             }
{ ---------------------------------------------------------------- }

procedure TNeuralEnglishG2P.AddPiece(const Piece: TNeuralG2PPiece);
var
  GroupIdx: integer;
begin
  if FPieceCount = Length(FPieces) then SetLength(FPieces, FPieceCount * 2 + 64);
  FPieces[FPieceCount] := Piece;
  GroupIdx := FGroupCount - 1;
  if not Piece.HasAlias and not Piece.HasPhonemes and (GroupIdx >= 0) and
    FGroups[GroupIdx].IsOpenList and
    (FPieces[FGroups[GroupIdx].LastPiece].Whitespace = '') then
    FGroups[GroupIdx].LastPiece := FPieceCount
  else
  begin
    if FGroupCount = Length(FGroups) then
      SetLength(FGroups, FGroupCount * 2 + 32);
    FGroups[FGroupCount].FirstPiece := FPieceCount;
    FGroups[FGroupCount].LastPiece := FPieceCount;
    FGroups[FGroupCount].IsOpenList := not Piece.HasAlias and
      not Piece.HasPhonemes and (Piece.Whitespace = '');
    Inc(FGroupCount);
  end;
  Inc(FPieceCount);
end;

// Clitic split (spaCy: dog|'s, do|n't, I|'d|'ve, dogs|'), then misaki's
// sub-token regex on each part; a '2' between letters reads "to" (P2P).
procedure TNeuralEnglishG2P.AppendWordPieces(const RawToken: TNeuralG2PRawToken);
const
  cClitics: array[0..6] of string = ('n''t', '''s', '''re', '''ve', '''ll',
    '''d', '''m');
var
  Parts, SubPieces: TStringList;
  Rest, Lower, Clitic, PieceTag: string;
  CliticCnt, PartCnt, MaxPartPos, PieceCnt, MaxPieceCnt, FirstNew: integer;
  Found, IsCurly: boolean;
  Piece: TNeuralG2PPiece;
begin
  Parts := TStringList.Create;
  SubPieces := TStringList.Create;
  try
    Rest := RawToken.Text;
    repeat
      Found := false;
      Lower := LowerCase(Rest);
      for CliticCnt := 0 to High(cClitics) do
      begin
        for IsCurly := false to true do
        begin
          if IsCurly then
            Clitic := StringReplace(cClitics[CliticCnt], '''', #$E2#$80#$99, [])
          else Clitic := cClitics[CliticCnt];
          if (Length(Lower) > Length(Clitic)) and EndsWith(Lower, Clitic) then
          begin
            Parts.Insert(0, Copy(Rest, Length(Rest) - Length(Clitic) + 1, MaxInt));
            Rest := DropBytes(Rest, Length(Clitic));
            Found := true;
            Break;
          end;
        end;
        if Found then Break;
      end;
      // plural possessive: dogs' -> dogs | '
      if not Found then
        for IsCurly := false to true do
        begin
          if IsCurly then Clitic := 's' + #$E2#$80#$99 else Clitic := 's''';
          if (Length(Lower) > Length(Clitic)) and EndsWith(Lower, Clitic) then
          begin
            Parts.Insert(0, Copy(Rest, Length(Rest) - Length(Clitic) + 2, MaxInt));
            Rest := DropBytes(Rest, Length(Clitic) - 1);
            Found := true;
            Break;
          end;
        end;
    until not Found;
    Parts.Insert(0, Rest);
    FirstNew := FPieceCount;
    MaxPartPos := Parts.Count - 1;
    for PartCnt := 0 to MaxPartPos do
    begin
      if Parts[PartCnt] = '' then Continue;
      G2PSubTokenize(Parts[PartCnt], SubPieces);
      MaxPieceCnt := SubPieces.Count - 1;
      for PieceCnt := 0 to MaxPieceCnt do
      begin
        Piece := Default(TNeuralG2PPiece);
        Piece.Text := SubPieces[PieceCnt];
        Piece.Tag := RawToken.Tag;
        // spaCy tags the parts of a hyphenated word one by one (not A380)
        if (Parts.Count > 1) or ((MaxPieceCnt > 0) and
          (Pos('-', Parts[PartCnt]) > 0)) then
        begin
          PieceTag := ClosedClassTag(LowerCase(Piece.Text));
          if (PieceTag = 'PRP') and (Piece.Text <> 'I') then PieceTag := '';
          if PieceTag <> '' then Piece.Tag := PieceTag;
        end;
        if (PieceCnt > 0) and (PieceCnt < MaxPieceCnt) and (Piece.Text = '2') and
          IsLetterCP(LastCP(SubPieces[PieceCnt - 1])) and
          (SubPieces[PieceCnt + 1] <> '') and
          (SubPieces[PieceCnt + 1][1] in ['A'..'Z', 'a'..'z']) then
        begin
          Piece.Alias := 'to';
          Piece.HasAlias := true;
        end;
        if (PartCnt = MaxPartPos) and (PieceCnt = MaxPieceCnt) then
          Piece.Whitespace := RawToken.Whitespace;
        AddPiece(Piece);
      end;
    end;
    if FPieceCount = FirstNew then
    begin
      Piece := Default(TNeuralG2PPiece);
      Piece.Text := RawToken.Text;
      Piece.Tag := RawToken.Tag;
      Piece.Whitespace := RawToken.Whitespace;
      AddPiece(Piece);
    end;
  finally
    SubPieces.Free;
    Parts.Free;
  end;
end;

// misaki retokenize, including its currency rule: after $ £ €, the last
// piece of a CD token takes the sign unless the next token is CD too
// ("$12 million" -> million dollars); a token that is not CD ends it.
procedure TNeuralEnglishG2P.BuildPiecesAndGroups();
var
  RawCnt, MaxRawPos: integer;
  Piece: TNeuralG2PPiece;
  Currency: string;
begin
  FPieceCount := 0;
  FGroupCount := 0;
  Currency := '';
  MaxRawPos := FRawCount - 1;
  for RawCnt := 0 to MaxRawPos do
    if FRaw[RawCnt].IsPunctuation then
    begin
      Piece := Default(TNeuralG2PPiece);
      Piece.Text := FRaw[RawCnt].Text;
      Piece.Tag := FRaw[RawCnt].Tag;
      Piece.Whitespace := FRaw[RawCnt].Whitespace;
      Piece.Phonemes := FRaw[RawCnt].Phonemes;
      Piece.HasPhonemes := true;
      Piece.Resolutions := [g2prPunctuation];
      AddPiece(Piece);
      if (FRaw[RawCnt].Tag = '$') and (CurrencyIndex(FRaw[RawCnt].Text) >= 0) then
        Currency := FRaw[RawCnt].Text;
    end
    else
    begin
      AppendWordPieces(FRaw[RawCnt]);
      if Currency <> '' then
      begin
        if FRaw[RawCnt].Tag <> 'CD' then Currency := ''
        else if (RawCnt = MaxRawPos) or (FRaw[RawCnt + 1].Tag <> 'CD') then
          FPieces[FPieceCount - 1].Currency := Currency;
      end;
    end;
end;

{ ---------------------------------------------------------------- }
{ misaki Lexicon                                                    }
{ ---------------------------------------------------------------- }

function TNeuralEnglishG2P.GoldHas(const Word: string): boolean;
begin
  Result := FGold.IndexOfGrown(Word) >= 0;
end;

function TNeuralEnglishG2P.SilverHas(const Word: string): boolean;
begin
  Result := FSilver.IndexOfGrown(Word) >= 0;
end;

// misaki is_known.
function TNeuralEnglishG2P.IsKnown(const Word: string): boolean;
begin
  if GoldHas(Word) or (SymbolWord(Word) <> '') or SilverHas(Word) then
    Exit(true);
  if not IsAlphaText(Word) or not HasOnlyLexiconChars(Word) then Exit(false);
  if Length(Word) = 1 then Exit(true);
  if (Word = UpperCase(Word)) and GoldHas(LowerCase(Word)) then Exit(true);
  Result := TailAfterFirstCP(Word) = UpperCase(TailAfterFirstCP(Word));
end;

// misaki get_NNP: the letter entries, all stresses secondary, the last
// one primary.
function TNeuralEnglishG2P.SpellLetters(const Word: string;
  out Phonemes: string): boolean;
var
  Position, TextLen, LetterIdx, LastSecondary: integer;
  CP: cardinal;
  Letter: string;
begin
  Phonemes := '';
  TextLen := Length(Word);
  Position := 1;
  while Position <= TextLen do
  begin
    CP := NextCodePoint(Word, Position);
    if not IsLetterCP(CP) then Continue;
    LetterIdx := FGold.IndexOfGrown(CodePointToUTF8(UppercaseCP(CP)));
    if (LetterIdx < 0) or not FGold.DefaultReading(LetterIdx, Letter) then
    begin
      Phonemes := '';
      Exit(false);
    end;
    Phonemes := Phonemes + Letter;
  end;
  Phonemes := ApplyG2PStress(Phonemes, 0);
  LastSecondary := Length(Phonemes) - Length(csSecondaryStress) + 1;
  while (LastSecondary >= 1) and
    (Copy(Phonemes, LastSecondary, Length(csSecondaryStress)) <> csSecondaryStress) do
    Dec(LastSecondary);
  if LastSecondary >= 1 then
    Phonemes := Copy(Phonemes, 1, LastSecondary - 1) + csPrimaryStress +
      Copy(Phonemes, LastSecondary + Length(csSecondaryStress), MaxInt);
  Result := true;
end;

// misaki lookup.
function TNeuralEnglishG2P.Lookup(const AWord, Tag: string; Stress: single;
  const Ctx: TNeuralG2PContext; HasContext: boolean; out Phonemes: string;
  var Resolutions: TNeuralG2PResolutions): boolean;
var
  Word, ChosenTag: string;
  IsNNP, HasReading: boolean;
  EntryIdx: integer;
  Dictionary: TNeuralG2PDictionary;
  ReadingResolution: TNeuralG2PResolution;
begin
  Word := AWord;
  IsNNP := false;
  if (Word = G2PUpper(Word)) and not GoldHas(Word) then
  begin
    Word := G2PLower(Word);
    IsNNP := Tag = 'NNP';
  end;
  Dictionary := FGold;
  EntryIdx := FGold.IndexOfGrown(Word);
  HasReading := (EntryIdx >= 0) and (FGold.IsPOSKeyed(EntryIdx) or
    not FGold.FEntries[EntryIdx].ValueIsNull);
  if not HasReading and not IsNNP then
  begin
    Dictionary := FSilver;
    EntryIdx := FSilver.IndexOfGrown(Word);
  end;
  HasReading := false;
  ReadingResolution := g2prDictionary;
  Phonemes := '';
  if EntryIdx >= 0 then
  begin
    if Dictionary.IsPOSKeyed(EntryIdx) then
    begin
      if HasContext and (Ctx.NextSound = g2pnsUnknown) and
        Dictionary.HasTag(EntryIdx, 'None') then ChosenTag := 'None'
      else if Dictionary.HasTag(EntryIdx, Tag) then ChosenTag := Tag
      else ChosenTag := ParentTag(Tag);
      HasReading := Dictionary.TagReading(EntryIdx, ChosenTag, Phonemes);
      if ChosenTag = 'None' then ReadingResolution := g2prContextRule
      else if Dictionary.HasTag(EntryIdx, ChosenTag) and
        (ChosenTag <> 'DEFAULT') then ReadingResolution := g2prPOSRule;
    end
    else HasReading := Dictionary.DefaultReading(EntryIdx, Phonemes);
  end;
  if not HasReading or (IsNNP and (Pos(csPrimaryStress, Phonemes) = 0)) then
    if SpellLetters(Word, Phonemes) then
    begin
      Include(Resolutions, g2prSpelled);
      Exit(true);
    end;
  if not HasReading then
  begin
    Phonemes := '';
    Exit(false);
  end;
  Include(Resolutions, ReadingResolution);
  Phonemes := ApplyG2PStress(Phonemes, Stress);
  Result := true;
end;

// misaki get_special_case.
function TNeuralEnglishG2P.SpecialCase(const Word, Tag: string; Stress: single;
  const Ctx: TNeuralG2PContext; out Phonemes: string;
  var Resolutions: TNeuralG2PResolutions): boolean;
var
  Stripped, Part: string;
  Parts: TStringList;
  PartCnt, MaxPartPos, LongestPart, EntryIdx: integer;
  IsDotted: boolean;
begin
  Phonemes := '';
  Result := false;
  if (Tag = 'ADD') and ((Word = '.') or (Word = '/')) then
  begin
    if Word = '.' then Part := 'dot' else Part := 'slash';
    Result := Lookup(Part, '', -0.5, Ctx, true, Phonemes, Resolutions);
    Exit;
  end;
  if SymbolWord(Word) <> '' then
  begin
    Result := Lookup(SymbolWord(Word), '', csG2PNoStress, Ctx, true, Phonemes,
      Resolutions);
    if Result then Include(Resolutions, g2prContextRule);
    Exit;
  end;
  // dotted abbreviation with parts of at most two letters: spelled
  Stripped := Word;
  while (Stripped <> '') and (Stripped[1] = '.') do Delete(Stripped, 1, 1);
  while (Stripped <> '') and (Stripped[Length(Stripped)] = '.') do
    Delete(Stripped, Length(Stripped), 1);
  if (Pos('.', Stripped) > 0) and
    IsAlphaText(StringReplace(Word, '.', '', [rfReplaceAll])) then
  begin
    Parts := TStringList.Create;
    try
      Parts.StrictDelimiter := true;
      Parts.Delimiter := '.';
      Parts.DelimitedText := Word;
      LongestPart := 0;
      MaxPartPos := Parts.Count - 1;
      for PartCnt := 0 to MaxPartPos do
        if CodePointCount(Parts[PartCnt]) > LongestPart then
          LongestPart := CodePointCount(Parts[PartCnt]);
    finally
      Parts.Free;
    end;
    IsDotted := LongestPart < 3;
    if IsDotted then
    begin
      Result := SpellLetters(Word, Phonemes);
      if Result then Include(Resolutions, g2prSpelled);
      Exit;
    end;
  end;
  if (Word = 'a') or (Word = 'A') then
  begin
    if Tag = 'DT' then Phonemes := csSchwaA
    else Phonemes := csPrimaryStress + 'A';
    Include(Resolutions, g2prContextRule);
    Exit(true);
  end;
  if (Word = 'am') or (Word = 'Am') or (Word = 'AM') then
  begin
    if Copy(Tag, 1, 2) = 'NN' then
    begin
      Result := SpellLetters(Word, Phonemes);
      if Result then Include(Resolutions, g2prSpelled);
      Exit;
    end;
    if (Ctx.NextSound = g2pnsUnknown) or (Word <> 'am') or
      ((Stress <> csG2PNoStress) and (Stress > 0)) then
    begin
      EntryIdx := FGold.IndexOfGrown('am');
      Result := (EntryIdx >= 0) and FGold.DefaultReading(EntryIdx, Phonemes);
    end
    else
    begin
      Phonemes := csSchwaA + 'm';
      Result := true;
    end;
    if Result then Include(Resolutions, g2prContextRule);
    Exit;
  end;
  if (Word = 'an') or (Word = 'An') or (Word = 'AN') then
  begin
    if (Word = 'AN') and (Copy(Tag, 1, 2) = 'NN') then
    begin
      Result := SpellLetters(Word, Phonemes);
      if Result then Include(Resolutions, g2prSpelled);
      Exit;
    end;
    Phonemes := csSchwaA + 'n';
    Include(Resolutions, g2prContextRule);
    Exit(true);
  end;
  if (Word = 'I') and (Tag = 'PRP') then
  begin
    Phonemes := csSecondaryStress + 'I';
    Include(Resolutions, g2prContextRule);
    Exit(true);
  end;
  if ((Word = 'by') or (Word = 'By') or (Word = 'BY')) and
    (ParentTag(Tag) = 'ADV') then
  begin
    Phonemes := 'b' + csPrimaryStress + 'I';
    Include(Resolutions, g2prContextRule);
    Exit(true);
  end;
  if (Word = 'to') or (Word = 'To') or
    ((Word = 'TO') and ((Tag = 'TO') or (Tag = 'IN'))) then
  begin
    case Ctx.NextSound of
      g2pnsUnknown:
        begin
          EntryIdx := FGold.IndexOfGrown('to');
          Result := (EntryIdx >= 0) and FGold.DefaultReading(EntryIdx, Phonemes);
        end;
      g2pnsConsonant:
        begin
          Phonemes := 't' + csSchwa;
          Result := true;
        end;
      g2pnsVowel:
        begin
          Phonemes := 't' + csUpsilon;
          Result := true;
        end;
    end;
    if Result then Include(Resolutions, g2prContextRule);
    Exit;
  end;
  if (Word = 'in') or (Word = 'In') or ((Word = 'IN') and (Tag <> 'NNP')) then
  begin
    if (Ctx.NextSound = g2pnsUnknown) or (Tag <> 'IN') then
      Phonemes := csPrimaryStress + csSmallCapI + 'n'
    else Phonemes := csSmallCapI + 'n';
    Include(Resolutions, g2prContextRule);
    Exit(true);
  end;
  if (Word = 'the') or (Word = 'The') or ((Word = 'THE') and (Tag = 'DT')) then
  begin
    if Ctx.NextSound = g2pnsVowel then Phonemes := csEth + 'i'
    else Phonemes := csEth + csSchwa;
    Include(Resolutions, g2prContextRule);
    Exit(true);
  end;
  if (Tag = 'IN') and ((LowerCase(Word) = 'vs') or (LowerCase(Word) = 'vs.')) then
  begin
    Result := Lookup('versus', '', csG2PNoStress, Ctx, true, Phonemes,
      Resolutions);
    if Result then Include(Resolutions, g2prContextRule);
    Exit;
  end;
  if (Word = 'used') or (Word = 'Used') or (Word = 'USED') then
  begin
    EntryIdx := FGold.IndexOfGrown('used');
    if EntryIdx < 0 then Exit(false);
    if ((Tag = 'VBD') or (Tag = 'JJ')) and Ctx.NextIsTo then
    begin
      Result := FGold.TagReading(EntryIdx, 'VBD', Phonemes);
      if Result then Include(Resolutions, g2prPOSRule);
    end
    else
    begin
      Result := FGold.DefaultReading(EntryIdx, Phonemes);
      if Result then Include(Resolutions, g2prDictionary);
    end;
    Exit;
  end;
end;

// misaki _s: -s after p t k f θ, -ᵻz/-ɪz after sibilants, else -z.
function TNeuralEnglishG2P.SuffixS(const Stem: string): string;
begin
  case LastCP(Stem) of
    Ord('p'), Ord('t'), Ord('k'), Ord('f'), $3B8: Result := Stem + 's';
    Ord('s'), Ord('z'), $283, $292, $2A7, $2A4:
      if FIsBritish then Result := Stem + csSmallCapI + 'z'
      else Result := Stem + csBarredI + 'z';
  else
    Result := Stem + 'z';
  end;
end;

// misaki _ed: -t / -ᵻd / -d, and the US flap before -ᵻd after a vowel + t.
function TNeuralEnglishG2P.SuffixEd(const Stem: string): string;
var
  Last: cardinal;
begin
  Last := LastCP(Stem);
  case Last of
    Ord('p'), Ord('k'), Ord('f'), $3B8, $283, Ord('s'), $2A7:
      Exit(Stem + 't');
    Ord('d'):
      begin
        if FIsBritish then Exit(Stem + csSmallCapI + 'd');
        Exit(Stem + csBarredI + 'd');
      end;
  end;
  if Last <> Ord('t') then Exit(Stem + 'd');
  if FIsBritish or (CodePointCount(Stem) < 2) then
    Exit(Stem + csSmallCapI + 'd');
  if IsUSTauCP(SecondLastCP(Stem)) then
    Exit(DropLastCP(Stem) + csFlapT + csBarredI + 'd');
  Result := Stem + csBarredI + 'd';
end;

// misaki _ing: GB refuses a stem ending in ə or ː; US flaps a vowel + t.
function TNeuralEnglishG2P.SuffixIng(const Stem: string;
  out Phonemes: string): boolean;
var
  Last: cardinal;
begin
  Phonemes := '';
  if Stem = '' then Exit(false);
  Last := LastCP(Stem);
  if FIsBritish then
  begin
    if (Last = $259) or (Last = $2D0) then Exit(false);
  end
  else if (CodePointCount(Stem) > 1) and (Last = Ord('t')) and
    IsUSTauCP(SecondLastCP(Stem)) then
  begin
    Phonemes := DropLastCP(Stem) + csFlapT + csSmallCapI + csEng;
    Exit(true);
  end;
  Phonemes := Stem + csSmallCapI + csEng;
  Result := true;
end;

function TNeuralEnglishG2P.StemS(const Word, Tag: string; Stress: single;
  const Ctx: TNeuralG2PContext; out Phonemes: string;
  var Resolutions: TNeuralG2PResolutions): boolean;
var
  WordLen: integer;
  Stem, StemPhonemes: string;
begin
  Phonemes := '';
  WordLen := CodePointCount(Word);
  if (WordLen < 3) or not EndsWith(Word, 's') then Exit(false);
  if not EndsWith(Word, 'ss') and IsKnown(DropBytes(Word, 1)) then
    Stem := DropBytes(Word, 1)
  else if (EndsWith(Word, '''s') or ((WordLen > 4) and EndsWith(Word, 'es') and
    not EndsWith(Word, 'ies'))) and IsKnown(DropBytes(Word, 2)) then
    Stem := DropBytes(Word, 2)
  else if (WordLen > 4) and EndsWith(Word, 'ies') and
    IsKnown(DropBytes(Word, 3) + 'y') then
    Stem := DropBytes(Word, 3) + 'y'
  else Exit(false);
  if not Lookup(Stem, Tag, Stress, Ctx, true, StemPhonemes, Resolutions) or
    (StemPhonemes = '') then Exit(false);
  Phonemes := SuffixS(StemPhonemes);
  Include(Resolutions, g2prStemRule);
  Result := true;
end;

function TNeuralEnglishG2P.StemEd(const Word, Tag: string; Stress: single;
  const Ctx: TNeuralG2PContext; out Phonemes: string;
  var Resolutions: TNeuralG2PResolutions): boolean;
var
  WordLen: integer;
  Stem, StemPhonemes: string;
begin
  Phonemes := '';
  WordLen := CodePointCount(Word);
  if (WordLen < 4) or not EndsWith(Word, 'd') then Exit(false);
  if not EndsWith(Word, 'dd') and IsKnown(DropBytes(Word, 1)) then
    Stem := DropBytes(Word, 1)
  else if (WordLen > 4) and EndsWith(Word, 'ed') and not EndsWith(Word, 'eed')
    and IsKnown(DropBytes(Word, 2)) then
    Stem := DropBytes(Word, 2)
  else Exit(false);
  if not Lookup(Stem, Tag, Stress, Ctx, true, StemPhonemes, Resolutions) or
    (StemPhonemes = '') then Exit(false);
  Phonemes := SuffixEd(StemPhonemes);
  Include(Resolutions, g2prStemRule);
  Result := true;
end;

function TNeuralEnglishG2P.StemIng(const Word, Tag: string; Stress: single;
  const Ctx: TNeuralG2PContext; out Phonemes: string;
  var Resolutions: TNeuralG2PResolutions): boolean;
var
  WordLen, ByteLen: integer;
  Stem, StemPhonemes: string;
  IsDoubled: boolean;
begin
  Phonemes := '';
  WordLen := CodePointCount(Word);
  if (WordLen < 5) or not EndsWith(Word, 'ing') then Exit(false);
  ByteLen := Length(Word);
  // ([bcdgklmnprstvxz])\1ing$|cking$
  IsDoubled := EndsWith(Word, 'cking') or ((ByteLen >= 5) and
    (Word[ByteLen - 3] = Word[ByteLen - 4]) and
    (Word[ByteLen - 3] in ['b', 'c', 'd', 'g', 'k', 'l', 'm', 'n', 'p', 'r',
    's', 't', 'v', 'x', 'z']));
  if (WordLen > 5) and IsKnown(DropBytes(Word, 3)) then
    Stem := DropBytes(Word, 3)
  else if IsKnown(DropBytes(Word, 3) + 'e') then
    Stem := DropBytes(Word, 3) + 'e'
  else if (WordLen > 5) and IsDoubled and IsKnown(DropBytes(Word, 4)) then
    Stem := DropBytes(Word, 4)
  else Exit(false);
  if not Lookup(Stem, Tag, Stress, Ctx, true, StemPhonemes, Resolutions) or
    (StemPhonemes = '') then Exit(false);
  if not SuffixIng(StemPhonemes, Phonemes) then Exit(false);
  Include(Resolutions, g2prStemRule);
  Result := true;
end;

// misaki get_word.
function TNeuralEnglishG2P.GetWord(const AWord, Tag: string; Stress: single;
  const Ctx: TNeuralG2PContext; out Phonemes: string;
  var Resolutions: TNeuralG2PResolutions): boolean;
var
  Word, Lower, Tail: string;
  WordLen: integer;
  Probe: string;
  Unused: TNeuralG2PResolutions;
  ProbeStress: single;
begin
  if SpecialCase(AWord, Tag, Stress, Ctx, Phonemes, Resolutions) then Exit(true);
  Word := AWord;
  Lower := G2PLower(Word);
  WordLen := CodePointCount(Word);
  Tail := TailAfterFirstCP(Word);
  Unused := [];
  if (WordLen > 1) and IsAlphaText(StringReplace(Word, '''', '', [rfReplaceAll]))
    and (Word <> Lower) and ((Tag <> 'NNP') or (WordLen > 7)) and
    not GoldHas(Word) and not SilverHas(Word) and
    ((Word = G2PUpper(Word)) or (Tail = G2PLower(Tail))) then
  begin
    ProbeStress := Stress;
    if GoldHas(Lower) or SilverHas(Lower) or
      StemS(Lower, Tag, ProbeStress, Ctx, Probe, Unused) or
      StemEd(Lower, Tag, ProbeStress, Ctx, Probe, Unused) or
      StemIng(Lower, Tag, ProbeStress, Ctx, Probe, Unused) then
      Word := Lower;
  end;
  if IsKnown(Word) then
    Exit(Lookup(Word, Tag, Stress, Ctx, true, Phonemes, Resolutions));
  if EndsWith(Word, 's''') and IsKnown(DropBytes(Word, 2) + '''s') then
    Exit(Lookup(DropBytes(Word, 2) + '''s', Tag, Stress, Ctx, true, Phonemes,
      Resolutions));
  if EndsWith(Word, '''') and IsKnown(DropBytes(Word, 1)) then
    Exit(Lookup(DropBytes(Word, 1), Tag, Stress, Ctx, true, Phonemes,
      Resolutions));
  if StemS(Word, Tag, Stress, Ctx, Phonemes, Resolutions) then Exit(true);
  if StemEd(Word, Tag, Stress, Ctx, Phonemes, Resolutions) then Exit(true);
  if Stress = csG2PNoStress then ProbeStress := 0.5 else ProbeStress := Stress;
  if StemIng(Word, Tag, ProbeStress, Ctx, Phonemes, Resolutions) then Exit(true);
  Phonemes := '';
  Result := false;
end;

// misaki append_currency: "million" after "$12" reads "million dollars".
function TNeuralEnglishG2P.AppendCurrencyUnit(const Phonemes,
  Currency: string): string;
var
  CurrencyIdx: integer;
  UnitPhonemes: string;
  Unused: TNeuralG2PResolutions;
begin
  Result := Phonemes;
  CurrencyIdx := CurrencyIndex(Currency);
  if CurrencyIdx < 0 then Exit;
  Unused := [];
  if StemS(cCurrencyUnits[CurrencyIdx, 0] + 's', '', csG2PNoStress,
    Default(TNeuralG2PContext), UnitPhonemes, Unused) then
    Result := Phonemes + ' ' + UnitPhonemes;
end;

// misaki get_number: num2words words read from the dictionary ("and"
// dropped, "point" unstressed), then the -s/-ed/-ing ending ("1990s").
function TNeuralEnglishG2P.ReadNumber(const AWord, Currency: string;
  IsHead: boolean; out Phonemes: string): boolean;
var
  Word, Suffix, Digits, UnitPhonemes: string;
  PartCount, SuffixStart, CurrencyIdx, DotCount, BytePos,
    MaxBytePos: integer;
  HadMinus, IsOrdinal: boolean;
  Unused: TNeuralG2PResolutions;

  procedure AddPhonemes(const Part: string);
  begin
    if PartCount > 0 then Phonemes := Phonemes + ' ';
    Phonemes := Phonemes + Part;
    Inc(PartCount);
  end;

  // misaki lookup(word, None, stress, None); a word missing from the
  // dictionary is skipped (misaki would fail joining None).
  procedure AddWord(const NumberWord: string; Stress: single);
  var
    WordPhonemes: string;
  begin
    if Lookup(NumberWord, '', Stress, Default(TNeuralG2PContext), false,
      WordPhonemes, Unused) then AddPhonemes(WordPhonemes);
  end;

  // misaki extend_num(escape=True): split on non-letters, drop "and".
  procedure AddNumberWords(const NumberText: string);
  var
    WordStart, Position, TextLen: integer;
    NumberWord: string;
  begin
    TextLen := Length(NumberText);
    Position := 1;
    while Position <= TextLen do
    begin
      while (Position <= TextLen) and not (NumberText[Position] in ['a'..'z']) do
        Inc(Position);
      WordStart := Position;
      while (Position <= TextLen) and (NumberText[Position] in ['a'..'z']) do
        Inc(Position);
      if Position = WordStart then Break;
      NumberWord := Copy(NumberText, WordStart, Position - WordStart);
      if NumberWord = 'point' then AddWord(NumberWord, -2)
      else if NumberWord <> 'and' then AddWord(NumberWord, csG2PNoStress);
    end;
  end;

  procedure AddDigitByDigit(const DigitText: string);
  var
    DigitPos, MaxDigitPos: integer;
  begin
    MaxDigitPos := Length(DigitText);
    for DigitPos := 1 to MaxDigitPos do
      if DigitText[DigitPos] in ['0'..'9'] then
        AddNumberWords(cLowNumberWords[Ord(DigitText[DigitPos]) - Ord('0')]);
  end;

  function HasNonZeroAfterFirst(const DigitText: string): boolean;
  var
    DigitPos, MaxDigitPos: integer;
  begin
    MaxDigitPos := Length(DigitText);
    for DigitPos := 2 to MaxDigitPos do
      if DigitText[DigitPos] <> '0' then Exit(true);
    Result := false;
  end;

  // Digits glued to letters (A380, 50p): short groups or digit by digit.
  procedure AddGluedDigits;
  begin
    if (Digits[1] = '0') or (Length(Digits) > 3) then AddDigitByDigit(Digits)
    else if (Length(Digits) = 3) and not EndsWith(Digits, '00') then
    begin
      AddNumberWords(EnglishCardinalWords(Digits[1]));
      if Digits[2] = '0' then
      begin
        AddWord('O', -2);
        AddNumberWords(EnglishCardinalWords(Digits[3]));
      end
      else AddNumberWords(EnglishCardinalWords(Copy(Digits, 2, 2)));
    end
    else AddNumberWords(EnglishCardinalWords(Digits));
  end;

  // Version numbers (2.0.1): each group digit by digit, unless it has two
  // digits or is round (10, 100).
  procedure AddVersionGroups;
  var
    Groups: TStringList;
    GroupCnt, MaxGroupPos: integer;
  begin
    Groups := TStringList.Create;
    try
      Groups.StrictDelimiter := true;
      Groups.Delimiter := '.';
      Groups.DelimitedText := Digits;
      MaxGroupPos := Groups.Count - 1;
      for GroupCnt := 0 to MaxGroupPos do
      begin
        if Groups[GroupCnt] = '' then Continue;
        if (Groups[GroupCnt][1] = '0') or ((Length(Groups[GroupCnt]) <> 2) and
          HasNonZeroAfterFirst(Groups[GroupCnt])) then
          AddDigitByDigit(Groups[GroupCnt])
        else AddNumberWords(EnglishCardinalWords(Groups[GroupCnt]));
      end;
    finally
      Groups.Free;
    end;
  end;

  procedure AddUnit(const Amount, UnitWord: string);
  var
    UnitPlural: string;
  begin
    AddNumberWords(EnglishCardinalWords(Amount));
    if (Amount <> '1') and (UnitWord <> 'pence') then
    begin
      if StemS(UnitWord + 's', '', csG2PNoStress, Default(TNeuralG2PContext),
        UnitPlural, Unused) then AddPhonemes(UnitPlural);
    end
    else AddWord(UnitWord, csG2PNoStress);
  end;

  // $5.50 -> five dollars and fifty cents; a zero part is dropped.
  procedure AddAmount;
  var
    WholeDigits, CentDigits: string;
    HasWhole: boolean;
  begin
    WholeDigits := Digits;
    CentDigits := '0';
    if DotCount > 0 then
    begin
      WholeDigits := Copy(Digits, 1, Pos('.', Digits) - 1);
      CentDigits := Copy(Digits, Pos('.', Digits) + 1, MaxInt);
      if WholeDigits = '' then WholeDigits := '0';
      if CentDigits = '' then CentDigits := '0';
    end;
    WholeDigits := StripLeadingZeros(WholeDigits);
    CentDigits := StripLeadingZeros(CentDigits);
    HasWhole := (CentDigits = '0') or (WholeDigits <> '0');
    if HasWhole then AddUnit(WholeDigits, cCurrencyUnits[CurrencyIdx, 0]);
    if CentDigits <> '0' then
    begin
      if HasWhole then AddWord('and', csG2PNoStress);
      AddUnit(CentDigits, cCurrencyUnits[CurrencyIdx, 1]);
    end;
  end;

begin
  Phonemes := '';
  PartCount := 0;
  Unused := [];
  Word := AWord;
  SuffixStart := Length(Word) + 1;
  while (SuffixStart > 1) and (Word[SuffixStart - 1] in ['a'..'z', '''']) do
    Dec(SuffixStart);
  Suffix := Copy(Word, SuffixStart, MaxInt);
  Word := Copy(Word, 1, SuffixStart - 1);
  IsOrdinal := (Suffix = 'st') or (Suffix = 'nd') or (Suffix = 'rd') or
    (Suffix = 'th');
  HadMinus := (Word <> '') and (Word[1] = '-');
  if HadMinus then
  begin
    AddWord('minus', csG2PNoStress);
    Delete(Word, 1, 1);
  end;
  Digits := StringReplace(Word, ',', '', [rfReplaceAll]);
  DotCount := 0;
  MaxBytePos := Length(Digits);
  for BytePos := 1 to MaxBytePos do
    if Digits[BytePos] = '.' then Inc(DotCount);
  CurrencyIdx := CurrencyIndex(Currency);
  try
    if IsDigitText(Word) and IsOrdinal then
      AddNumberWords(EnglishOrdinalWords(Word))
    else if not HadMinus and (Length(Word) = 4) and (CurrencyIdx < 0) and
      IsDigitText(Word) then
      AddNumberWords(EnglishYearWords(Word))
    else if not IsHead and (DotCount = 0) then AddGluedDigits
    else if (DotCount > 1) or not IsHead then AddVersionGroups
    else if (CurrencyIdx >= 0) and ((DotCount = 0) or
      (Length(Word) - Pos('.', Word) < 3)) then AddAmount
    else if IsDigitText(Word) then
      AddNumberWords(EnglishCardinalWords(Word))
    else if DotCount = 0 then
    begin
      if IsOrdinal then AddNumberWords(EnglishOrdinalWords(Digits))
      else AddNumberWords(EnglishCardinalWords(Digits));
    end
    else if Digits[1] = '.' then
    begin
      AddWord('point', -2);
      AddDigitByDigit(Copy(Digits, 2, MaxInt));
    end
    else AddNumberWords(EnglishDecimalWords(Digits));
  except
    // 10^306 and above: num2words raises, the port reads the digits
    on ENeuralG2PError do
    begin
      Phonemes := '';
      PartCount := 0;
      if HadMinus then AddWord('minus', csG2PNoStress);
      AddDigitByDigit(Digits);
    end;
  end;
  if PartCount = 0 then Exit(false);
  if (Suffix = 's') or (Suffix = '''s') then Phonemes := SuffixS(Phonemes)
  else if (Suffix = 'ed') or (Suffix = '''d') then Phonemes := SuffixEd(Phonemes)
  else if Suffix = 'ing' then
  begin
    UnitPhonemes := Phonemes;
    if not SuffixIng(UnitPhonemes, Phonemes) then Exit(false);
  end;
  Result := true;
end;

// misaki Lexicon.__call__ for one (merged) token text; IsHead: the text
// starts its word (misaki _.is_head), Currency: the amount's sign or ''.
function TNeuralEnglishG2P.ReadWord(const Text, Tag, Currency: string;
  IsHead: boolean; const Ctx: TNeuralG2PContext; out Phonemes: string;
  var Resolutions: TNeuralG2PResolutions): boolean;
var
  Word: string;
  Stress: single;
begin
  Word := ReplaceCurlyApostrophes(Text);
  if HasNonASCII(Word) then Word := UnicodeNormalize(Word, true, true);
  if Word = G2PLower(Word) then Stress := csG2PNoStress
  else if Word = G2PUpper(Word) then Stress := 2
  else Stress := 0.5;
  if GetWord(Word, Tag, Stress, Ctx, Phonemes, Resolutions) then
  begin
    if Currency <> '' then Phonemes := AppendCurrencyUnit(Phonemes, Currency);
    Exit(true);
  end;
  if IsNumberWord(Word, IsHead) and
    ReadNumber(Word, Currency, IsHead, Phonemes) then
  begin
    Include(Resolutions, g2prNumber);
    Exit(true);
  end;
  Phonemes := '';
  Result := false;
end;

{ ---------------------------------------------------------------- }
{ Right-to-left resolution (misaki G2P.__call__)                    }
{ ---------------------------------------------------------------- }

// misaki token_context.
procedure UpdateContext(var Ctx: TNeuralG2PContext; HasPhonemes: boolean;
  const Phonemes, Text, Tag: string);
var
  Position, TextLen: integer;
  CP: cardinal;
begin
  if HasPhonemes and (Phonemes <> '') then
  begin
    TextLen := Length(Phonemes);
    Position := 1;
    while Position <= TextLen do
    begin
      CP := NextCodePoint(Phonemes, Position);
      if IsNonQuotePunctCP(CP) then
      begin
        Ctx.NextSound := g2pnsUnknown;
        Break;
      end;
      if IsVowelCP(CP) then
      begin
        Ctx.NextSound := g2pnsVowel;
        Break;
      end;
      if IsConsonantCP(CP) then
      begin
        Ctx.NextSound := g2pnsConsonant;
        Break;
      end;
    end;
  end;
  Ctx.NextIsTo := (Text = 'to') or (Text = 'To') or
    ((Text = 'TO') and ((Tag = 'TO') or (Tag = 'IN')));
end;

// misaki resolve_tokens: junk pieces read '', prespace across mixed pieces,
// and the lighter half of several primary stresses demoted.
procedure TNeuralEnglishG2P.ResolveTokens(FirstPiece, LastPiece: integer);
var
  Text: string;
  PieceCnt, Position, TextLen, KindMask, KindCount, IndexCount, IndexCnt,
    MaxIndexPos, MaxDemotedPos, InnerCnt, PrimaryCount: integer;
  CP: cardinal;
  Prespace, AllJunk: boolean;
  HasPrimary: array of boolean;
  Weights, Indices: array of integer;
  SwapBool: boolean;
  SwapInt: integer;

  function IndexBefore(A, B: integer): boolean;
  begin
    if HasPrimary[A] <> HasPrimary[B] then Exit(not HasPrimary[A]);
    if Weights[A] <> Weights[B] then Exit(Weights[A] < Weights[B]);
    Result := Indices[A] < Indices[B];
  end;

begin
  Text := PiecesText(FirstPiece, LastPiece);
  // character kinds outside the junk set: letter 1, digit 2, other 4
  KindMask := 0;
  TextLen := Length(Text);
  Position := 1;
  while Position <= TextLen do
  begin
    CP := NextCodePoint(Text, Position);
    if IsJunkCP(CP) then Continue;
    if IsLetterCP(CP) then KindMask := KindMask or 1
    else if IsDigitCP(CP) then KindMask := KindMask or 2
    else KindMask := KindMask or 4;
  end;
  KindCount := (KindMask and 1) + ((KindMask shr 1) and 1) + ((KindMask shr 2) and 1);
  Prespace := (Pos(' ', Text) > 0) or (Pos('/', Text) > 0) or (KindCount > 1);
  for PieceCnt := FirstPiece to LastPiece do
  begin
    if not FPieces[PieceCnt].HasPhonemes then
    begin
      if (PieceCnt = LastPiece) and (CodePointCount(FPieces[PieceCnt].Text) = 1) and
        IsNonQuotePunctCP(LastCP(FPieces[PieceCnt].Text)) then
      begin
        FPieces[PieceCnt].Phonemes := FPieces[PieceCnt].Text;
        FPieces[PieceCnt].HasPhonemes := true;
        Include(FPieces[PieceCnt].Resolutions, g2prPunctuation);
      end
      else
      begin
        AllJunk := true;
        TextLen := Length(FPieces[PieceCnt].Text);
        Position := 1;
        while Position <= TextLen do
          if not IsJunkCP(NextCodePoint(FPieces[PieceCnt].Text, Position)) then
          begin
            AllJunk := false;
            Break;
          end;
        if AllJunk then
        begin
          FPieces[PieceCnt].Phonemes := '';
          FPieces[PieceCnt].HasPhonemes := true;
        end;
      end;
    end
    else if PieceCnt > FirstPiece then
      FPieces[PieceCnt].Prespace := Prespace;
  end;
  if Prespace then Exit;
  SetLength(HasPrimary, LastPiece - FirstPiece + 1);
  SetLength(Weights, LastPiece - FirstPiece + 1);
  SetLength(Indices, LastPiece - FirstPiece + 1);
  IndexCount := 0;
  PrimaryCount := 0;
  for PieceCnt := FirstPiece to LastPiece do
    if FPieces[PieceCnt].HasPhonemes and (FPieces[PieceCnt].Phonemes <> '') then
    begin
      HasPrimary[IndexCount] := Pos(csPrimaryStress, FPieces[PieceCnt].Phonemes) > 0;
      Weights[IndexCount] := StressWeight(FPieces[PieceCnt].Phonemes);
      Indices[IndexCount] := PieceCnt;
      if HasPrimary[IndexCount] then Inc(PrimaryCount);
      Inc(IndexCount);
    end;
  if (IndexCount = 2) and (CodePointCount(FPieces[Indices[0]].Text) = 1) then
  begin
    FPieces[Indices[1]].Phonemes :=
      ApplyG2PStress(FPieces[Indices[1]].Phonemes, -0.5);
    Exit;
  end;
  if (IndexCount < 2) or (PrimaryCount <= (IndexCount + 1) div 2) then Exit;
  // sort (has primary, weight, index) ascending; demote the first half
  MaxIndexPos := IndexCount - 1;
  for IndexCnt := 1 to MaxIndexPos do
  begin
    InnerCnt := IndexCnt;
    while (InnerCnt > 0) and IndexBefore(InnerCnt, InnerCnt - 1) do
    begin
      SwapBool := HasPrimary[InnerCnt];
      HasPrimary[InnerCnt] := HasPrimary[InnerCnt - 1];
      HasPrimary[InnerCnt - 1] := SwapBool;
      SwapInt := Weights[InnerCnt];
      Weights[InnerCnt] := Weights[InnerCnt - 1];
      Weights[InnerCnt - 1] := SwapInt;
      SwapInt := Indices[InnerCnt];
      Indices[InnerCnt] := Indices[InnerCnt - 1];
      Indices[InnerCnt - 1] := SwapInt;
      Dec(InnerCnt);
    end;
  end;
  MaxDemotedPos := IndexCount div 2 - 1;
  for IndexCnt := 0 to MaxDemotedPos do
    FPieces[Indices[IndexCnt]].Phonemes :=
      ApplyG2PStress(FPieces[Indices[IndexCnt]].Phonemes, -0.5);
end;

// misaki merge_tokens text: the pieces with the whitespace between them
// (not after the last one), built in one allocation.
function TNeuralEnglishG2P.PiecesText(FirstPiece, LastPiece: integer): string;
var
  PieceCnt, TotalLen, OutPos, PartLen: integer;

  procedure Append(const Part: string);
  begin
    PartLen := Length(Part);
    if PartLen > 0 then Move(Part[1], Result[OutPos], PartLen);
    Inc(OutPos, PartLen);
  end;

begin
  TotalLen := 0;
  for PieceCnt := FirstPiece to LastPiece do
  begin
    Inc(TotalLen, Length(FPieces[PieceCnt].Text));
    if PieceCnt < LastPiece then
      Inc(TotalLen, Length(FPieces[PieceCnt].Whitespace));
  end;
  SetLength(Result, TotalLen);
  OutPos := 1;
  for PieceCnt := FirstPiece to LastPiece do
  begin
    Append(FPieces[PieceCnt].Text);
    if PieceCnt < LastPiece then Append(FPieces[PieceCnt].Whitespace);
  end;
end;

// misaki merge_tokens(unk='') phonemes: unread pieces add nothing, a piece
// marked Prespace gets a space before it. Two passes, one allocation.
function TNeuralEnglishG2P.PiecesPhonemes(FirstPiece, LastPiece: integer;
  out AnyKnown: boolean): string;
var
  OutLen, PartLen: integer;
  EndsWithSpace, IsWriting: boolean;

  procedure Emit(const Part: string);
  begin
    PartLen := Length(Part);
    if PartLen = 0 then Exit;
    if IsWriting then Move(Part[1], Result[OutLen + 1], PartLen);
    Inc(OutLen, PartLen);
    EndsWithSpace := Part[PartLen] = ' ';
  end;

  procedure EmitAll;
  var
    PieceCnt: integer;
  begin
    OutLen := 0;
    EndsWithSpace := false;
    for PieceCnt := FirstPiece to LastPiece do
      if FPieces[PieceCnt].HasPhonemes then
      begin
        AnyKnown := true;
        if FPieces[PieceCnt].Prespace and (OutLen > 0) and not EndsWithSpace and
          (FPieces[PieceCnt].Phonemes <> '') then Emit(' ');
        Emit(FPieces[PieceCnt].Phonemes);
      end;
  end;

begin
  AnyKnown := false;
  Result := '';
  IsWriting := false;
  EmitAll;
  SetLength(Result, OutLen);
  IsWriting := true;
  EmitAll;
end;

// misaki merge_tokens tag: the piece whose text weighs most (an upper-case
// character counts 2, any other 1); the first such piece on a tie.
function TNeuralEnglishG2P.HeaviestPieceTag(FirstPiece, LastPiece: integer): string;
var
  PieceCnt, Weight, BestWeight, Position, TextLen: integer;
  CP: cardinal;
begin
  Result := FPieces[FirstPiece].Tag;
  BestWeight := -1;
  for PieceCnt := FirstPiece to LastPiece do
  begin
    Weight := 0;
    TextLen := Length(FPieces[PieceCnt].Text);
    Position := 1;
    while Position <= TextLen do
    begin
      CP := NextCodePoint(FPieces[PieceCnt].Text, Position);
      if LowercaseCP(CP) <> CP then Inc(Weight, 2) else Inc(Weight);
    end;
    if Weight > BestWeight then
    begin
      BestWeight := Weight;
      Result := FPieces[PieceCnt].Tag;
    end;
  end;
end;

// misaki's loop over a multi-piece word: the longest merge ending at the
// right edge first, then the pieces to its left. Merges span at most
// csMaxMergedPieces pieces (misaki has no cap; its loop is cubic).
procedure TNeuralEnglishG2P.ResolveMultiPieceGroup(var Ctx: TNeuralG2PContext;
  FirstPiece, LastPiece: integer);
var
  Left, Right, MaxMergedPos, PieceCnt, Position, TextLen: integer;
  MergedText, MergedPhonemes, MergedTag, MergedCurrency: string;
  Resolutions: TNeuralG2PResolutions;
  AllJunk: boolean;
begin
  Right := LastPiece + 1;
  Left := Max(FirstPiece, Right - csMaxMergedPieces);
  while Left < Right do
  begin
    MaxMergedPos := Right - 1;
    MergedText := PiecesText(Left, MaxMergedPos);
    Resolutions := [];
    MergedTag := HeaviestPieceTag(Left, MaxMergedPos);
    // misaki merge_tokens: the largest currency sign of the pieces
    MergedCurrency := '';
    for PieceCnt := Left to MaxMergedPos do
      if FPieces[PieceCnt].Currency > MergedCurrency then
        MergedCurrency := FPieces[PieceCnt].Currency;
    if ReadWord(MergedText, MergedTag, MergedCurrency, Left = FirstPiece, Ctx,
      MergedPhonemes, Resolutions) then
    begin
      FPieces[Left].Phonemes := MergedPhonemes;
      FPieces[Left].HasPhonemes := true;
      FPieces[Left].Resolutions := Resolutions;
      for PieceCnt := Left + 1 to MaxMergedPos do
      begin
        FPieces[PieceCnt].Phonemes := '';
        FPieces[PieceCnt].HasPhonemes := true;
        FPieces[PieceCnt].Resolutions := Resolutions;
      end;
      UpdateContext(Ctx, true, MergedPhonemes, MergedText, MergedTag);
      Right := Left;
      Left := Max(FirstPiece, Right - csMaxMergedPieces);
    end
    else if Left + 1 < Right then
      Inc(Left)
    else
    begin
      Dec(Right);
      if not FPieces[Right].HasPhonemes then
      begin
        AllJunk := true;
        TextLen := Length(FPieces[Right].Text);
        Position := 1;
        while Position <= TextLen do
          if not IsJunkCP(NextCodePoint(FPieces[Right].Text, Position)) then
          begin
            AllJunk := false;
            Break;
          end;
        if AllJunk then
        begin
          FPieces[Right].Phonemes := '';
          FPieces[Right].HasPhonemes := true;
        end
        else
          FPieces[Right].Resolutions := FPieces[Right].Resolutions +
            Resolutions + [g2prUnknown];
      end;
      Left := Max(FirstPiece, Right - csMaxMergedPieces);
    end;
  end;
  ResolveTokens(FirstPiece, LastPiece);
end;

procedure TNeuralEnglishG2P.ResolveGroups();
var
  GroupCnt, PieceIdx: integer;
  Ctx: TNeuralG2PContext;
  Resolutions: TNeuralG2PResolutions;
  Phonemes, Text: string;
begin
  Ctx.NextSound := g2pnsUnknown;
  Ctx.NextIsTo := false;
  for GroupCnt := FGroupCount - 1 downto 0 do
  begin
    if FGroups[GroupCnt].FirstPiece = FGroups[GroupCnt].LastPiece then
    begin
      PieceIdx := FGroups[GroupCnt].FirstPiece;
      if not FPieces[PieceIdx].HasPhonemes then
      begin
        Resolutions := [];
        if FPieces[PieceIdx].HasAlias then Text := FPieces[PieceIdx].Alias
        else Text := FPieces[PieceIdx].Text;
        FPieces[PieceIdx].HasPhonemes := ReadWord(Text, FPieces[PieceIdx].Tag,
          FPieces[PieceIdx].Currency, true, Ctx, Phonemes, Resolutions);
        FPieces[PieceIdx].Phonemes := Phonemes;
        if not FPieces[PieceIdx].HasPhonemes then Include(Resolutions, g2prUnknown);
        FPieces[PieceIdx].Resolutions := Resolutions;
      end;
      UpdateContext(Ctx, FPieces[PieceIdx].HasPhonemes,
        FPieces[PieceIdx].Phonemes, FPieces[PieceIdx].Text,
        FPieces[PieceIdx].Tag);
    end
    else
      ResolveMultiPieceGroup(Ctx, FGroups[GroupCnt].FirstPiece,
        FGroups[GroupCnt].LastPiece);
  end;
end;

// misaki merge_tokens(unk='') per group, the ɾ -> T / ʔ -> t mapping, and
// the joined phoneme string.
function TNeuralEnglishG2P.BuildOutput(out Tokens: TNeuralG2PTokenArray): string;
var
  GroupCnt, PieceCnt, MaxGroupPos, TotalLen, OutPos, PartLen: integer;
  Token: TNeuralG2PToken;
  AnyKnown: boolean;
begin
  SetLength(Tokens, FGroupCount);
  MaxGroupPos := FGroupCount - 1;
  TotalLen := 0;
  for GroupCnt := 0 to MaxGroupPos do
  begin
    Token := Default(TNeuralG2PToken);
    AnyKnown := false;
    with FGroups[GroupCnt] do
    begin
      Token.Tag := FPieces[FirstPiece].Tag;
      Token.Text := PiecesText(FirstPiece, LastPiece);
      Token.Phonemes := PiecesPhonemes(FirstPiece, LastPiece, AnyKnown);
      for PieceCnt := FirstPiece to LastPiece do
        Token.Resolutions := Token.Resolutions + FPieces[PieceCnt].Resolutions;
      Token.Whitespace := FPieces[LastPiece].Whitespace;
      // a single unread piece is misaki's None; a merged list reads ''
      Token.IsKnown := AnyKnown or (FirstPiece <> LastPiece);
    end;
    Token.Phonemes := StringReplace(Token.Phonemes, csFlapT, 'T', [rfReplaceAll]);
    Token.Phonemes := StringReplace(Token.Phonemes, csGlottalStop, 't',
      [rfReplaceAll]);
    Tokens[GroupCnt] := Token;
    Inc(TotalLen, Length(Token.Phonemes) + Length(Token.Whitespace));
  end;
  SetLength(Result, TotalLen);
  OutPos := 1;
  for GroupCnt := 0 to MaxGroupPos do
  begin
    PartLen := Length(Tokens[GroupCnt].Phonemes);
    if PartLen > 0 then Move(Tokens[GroupCnt].Phonemes[1], Result[OutPos], PartLen);
    Inc(OutPos, PartLen);
    PartLen := Length(Tokens[GroupCnt].Whitespace);
    if PartLen > 0 then Move(Tokens[GroupCnt].Whitespace[1], Result[OutPos], PartLen);
    Inc(OutPos, PartLen);
  end;
end;

function TNeuralEnglishG2P.Phonemize(const Text: string;
  out Tokens: TNeuralG2PTokenArray): string;
begin
  SplitIntoRawTokens(Text);
  TagRawTokens();
  BuildPiecesAndGroups();
  ResolveGroups();
  Result := BuildOutput(Tokens);
end;

function TNeuralEnglishG2P.Phonemize(const Text: string): string;
var
  Tokens: TNeuralG2PTokenArray;
begin
  Result := Phonemize(Text, Tokens);
end;

end.
