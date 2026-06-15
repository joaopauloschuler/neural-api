(*
neuralhftokenizer
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

// neuralhftokenizer -- loads the HuggingFace single-file fast-tokenizer
// format (tokenizer.json) so the pretrained-checkpoint importers in
// neuralpretrained.pas (examples/GPT2Import, examples/LlamaImport) can take
// text prompts end to end instead of raw token ids.
//
// Supported (the three families the importers need):
//   * Byte-level BPE (GPT-2 / distilgpt2 / SmolLM2): the GPT-2
//     bytes-to-unicode alphabet, the GPT-2 pre-tokenization regex
//     ('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|
//      \s+(?!\S)|\s+) and ranked merges.
//   * Metaspace BPE with byte fallback (Llama / TinyLlama / Mistral): the
//     Prepend("▁") + Replace(" "->"▁") normalizer chain, whole-segment BPE
//     (no pre-tokenizer), <0xNN> byte-fallback tokens and the
//     Replace/ByteFallback/Fuse/Strip decoder chain. The Metaspace
//     PRE_TOKENIZER variant (Mistral-v0.x / legacy=false Llama exports:
//     replacement, prepend_scheme always/first/never, split) and the
//     Metaspace decoder are supported too.
//   * Split-regex byte-level BPE (Qwen2/Qwen3, Llama-3 style):
//     Sequence[Split(pattern, behavior=Isolated), ByteLevel(use_regex=
//     false)]. No regex engine is vendored -- the shipped pattern string
//     is matched verbatim against the known Qwen2 (\p{N}) and
//     Llama-3/cl100k (\p{N}{1,3}) pattern literals and dispatched to a
//     hand-written splitter; unknown patterns raise EHFTokenizerError.
//   * WordPiece (BERT / MiniLM / all-MiniLM-L6-v2): BertNormalizer
//     (clean_text, handle_chinese_chars, strip_accents, lowercase),
//     BertPreTokenizer (whitespace split + isolated punctuation), greedy
//     longest-match-first WordPiece with the "##" continuation prefix and
//     the WordPiece decoder (space join, "##" strip, cleanup). Strip-accents
//     covers Latin-1 Supplement and Latin Extended-A (cafe/naive/resume
//     class); marks on exotic scripts pass through -- same approximation
//     stance as the \p{L} tables below. Like the BPE families, special
//     tokens ([CLS]/[SEP]/...) are NEVER auto-injected: BERT-style callers
//     add them via TokenToId('[CLS]') / TokenToId('[SEP]') (see
//     BertTokenizeSentence in neuralpretrained.pas).
//
//   * Unigram (SentencePiece-Unigram: ALBERT / T5 / XLNet / DeBERTa-v3):
//     the `vocab` array of [piece, log_prob] pairs plus unk_id, Viterbi
//     maximum-log-probability segmentation over the vocab pieces, with a
//     single fused <unk> covering each run of characters no piece can
//     cover (HF behavior). Composes with the Metaspace pre_tokenizer /
//     decoder that these tokenizers ship with.
//
//   * Raw SentencePiece `.model` protobuf (LoadSentencePieceModel, also
//     auto-dispatched from LoadFromFile by the '.model' extension or a
//     non-'{' first byte): the ModelProto wire format is hand-decoded (no
//     vendored proto library) into the SAME Unigram structures as the
//     tokenizer.json-Unigram path -- pieces (text, score, type) + the
//     trainer_spec unk/bos/eos ids -- and the SentencePiece Metaspace
//     convention is wired so Encode/Decode behave identically. This is what
//     T5 / ALBERT / XLNet / DeBERTa-v3 / mBART(-Unigram) checkpoints ship
//     when they carry NO tokenizer.json. Both UNIGRAM and BPE model_types are
//     read; a WORD/CHAR ModelProto raises EHFTokenizerError. A BPE .model
//     stores only scored pieces (no merge list), so the merges are
//     reconstructed from the pieces (ReconstructBPEMerges, matching HF's
//     generate_merges) and routed through the metaspace byte-level-BPE path --
//     this covers the NLLB / mBART-BPE / DeBERTa-v3 sentencepiece.bpe.model
//     exports. The SentencePiece precompiled_charsmap (an opaque per-model
//     normalization trie, usually NFKC-derived) is NOT parsed/applied here; a
//     tokenizer.json that declares a standard NFKC/NFKD normalizer IS applied
//     in full (see UnicodeNormalize).
//
// NOT supported (raises EHFTokenizerError with a clear message):
//   * model.type <> "BPE"/"WordPiece"/"Unigram".
//   * A raw SentencePiece .model whose trainer_spec.model_type is WORD / CHAR
//     (only UNIGRAM and BPE are read from the protobuf -- use tokenizer.json
//     for WORD/CHAR).
//
// Unicode \p{L} / \p{N} classification for the GPT-2 regex is implemented
// over explicit ranges covering Latin (incl. Latin-1/Extended), Greek,
// Cyrillic, Armenian, Hebrew, Arabic, Devanagari, Hangul, Kana and CJK --
// an approximation of the full Unicode tables that is exact for these
// scripts; exotic codepoints outside them fall into the punctuation class.
//
// Special/added tokens are matched verbatim in the input text (HF fast
// tokenizer behavior) and their ids are exposed (BosId/EosId/UnkId), but
// they are never auto-injected: Encode('hello') returns only content ids.
//
// BPE-dropout (Provilkov et al. 2020) is available on the byte-level / BPE
// merge loop via the DropoutProb property (default 0.0 = OFF = bit-identical
// eval; set during TRAINING only to subword-regularize).
//
// Parity with HF `tokenizers` is pinned by tests/TestNeuralHFTokenizer.pas
// against fixtures generated by tools/hf_tokenizer_fixture.py and
// tools/hf_unigram_fixture.py.

unit neuralhftokenizer;
{$include neuralnetwork.inc}
{$H+}

interface

uses
  Classes, SysUtils, fpjson, neuralvolume, neuralgguf;

type
  EHFTokenizerError = class(Exception);

  TNeuralAddedToken = record
    Content: string;
    Id: integer;
    Special: boolean;
  end;

  // One emitted token's char span in the ORIGINAL input text. Start is the
  // 1-based byte index of the first char of the token's surface text; Length
  // is the byte length. Copy(Text, Start, Length) slices the surface text
  // back. WordId is the 0-based index of the whitespace-split word the token
  // belongs to (subword -> word alignment, HF word_ids() equivalent), or -1
  // for a token with no surface text in the input (e.g. an added/special
  // token like [CLS]). The HF (start,end) convention is (Start-1, Start-1+Length).
  TNeuralTokenOffset = record
    Id: integer;
    Start: integer;   // 1-based byte offset into the input text
    Length: integer;  // byte length of the surface text (0 if none)
    WordId: integer;  // 0-based whitespace word index, or -1
  end;
  TNeuralTokenOffsetArray = array of TNeuralTokenOffset;

  { TNeuralHFTokenizer }
  // Encoder/decoder for the HuggingFace tokenizer.json fast-tokenizer
  // format (byte-level BPE, metaspace/byte-fallback BPE and BERT
  // WordPiece).
  // Coded by Claude (AI).
  TNeuralHFTokenizer = class(TObject)
    private
      FVocab: TStringList;        // token -> id (Objects hold the id)
      FMerges: TStringList;       // left#1right -> rank (Objects hold rank)
      FIdToToken: array of string;
      FAddedTokens: array of TNeuralAddedToken;
      // model flags
      FByteFallback, FFuseUnk, FIgnoreMerges: boolean;
      FUnkId, FBosId, FEosId: integer;
      // Unigram (SentencePiece-Unigram: ALBERT/T5/XLNet/DeBERTa-v3) family
      FUnigram: boolean;          // model.type = Unigram
      FUniScore: array of double; // per-id piece log-prob (score), id-indexed
      FUniMinScore: double;       // min vocab score (unk penalty base)
      // WordPiece (BERT) family
      FWordPiece: boolean;        // model.type = WordPiece
      FWPPrefix: string;          // continuing_subword_prefix ('##')
      FWPMaxChars: integer;       // max_input_chars_per_word
      FBertLowercase: boolean;    // BertNormalizer lowercase
      FBertStripAccents: boolean; // BertNormalizer strip_accents (resolved)
      FBertCleanText: boolean;    // BertNormalizer clean_text
      FBertHandleChinese: boolean;// BertNormalizer handle_chinese_chars
      FDecWordPieceCleanup: boolean; // WordPiece decoder cleanup flag
      // pre-tokenizer / normalizer
      FByteLevel: boolean;        // ByteLevel pre-tokenizer present
      FAddPrefixSpace: boolean;
      // ByteLevel use_regex: when false a STANDALONE ByteLevel pre_tokenizer
      // does NOT apply the GPT-2 regex split -- the whole segment is one
      // chunk fed straight to the byte-level alphabet. (Inside a
      // Sequence[Split, ByteLevel] the upstream Split already did the
      // splitting, so ByteLevel always has use_regex=false there.) Default
      // true = legacy GPT-2 behavior.
      FByteLevelUseRegex: boolean;
      // Split pre-tokenizer (Qwen2 / Llama-3 cl100k-style pattern)
      FSplitPreTok: boolean;
      FSplitDigitsMax: integer;   // 1 (Qwen2 \p{N}) or 3 (cl100k \p{N}{1,3})
      // DeepSeek-V2/V3 multi-Split+Digits pre-tokenizer (a distinct splitter
      // from the cl100k family -- see SplitDeepSeekPieces).
      FDeepSeekPreTok: boolean;
      // o200k_base / GPT-4o-family Split pattern (case-aware letter runs --
      // see SplitO200kPieces).
      FO200kPreTok: boolean;
      // Metaspace pre-tokenizer (Llama-2 / Mistral SentencePiece-BPE)
      FMetaspacePreTok: boolean;
      FMSReplacement: string;     // usually U+2581
      FMSPrependScheme: string;   // 'always' | 'first' | 'never'
      FMSSplit: boolean;          // split before each replacement char
      // Unicode normalization form to apply to input before everything else.
      // '' = none; 'NFC'/'NFD' = real canonical normalization; 'NFKC'/'NFKD' =
      // real compatibility normalization (compat decompositions folded -- see
      // UnicodeNormalize).
      FNormUnicodeForm: string;
      FNormPrepend: string;       // Prepend normalizer content ('' = none)
      FNormReplaceFrom: array of string; // Replace normalizers, in order
      FNormReplaceTo: array of string;
      // decoder
      FDecReplaceFrom: array of string;  // decoder Replace rules, in order
      FDecReplaceTo: array of string;
      FDecStripLeft: integer;     // decoder Strip(' ', start, 0)
      // GPT-2 bytes<->unicode table
      FByteToCP: array[0..255] of cardinal;
      FCPToByte: array[0..$143] of integer;
      FKeysMangled: boolean;
      // BPE-dropout (Provilkov et al. 2020): probability of skipping each
      // otherwise-applicable merge during TRAINING tokenization, so the model
      // sees alternative subword segmentations of the same text. Default 0.0
      // = OFF = bit-identical to deterministic eval/inference tokenization.
      FDropoutProb: TNeuralFloat;
      procedure DetectKeyMangling();
      function FixJSONKey(const Key: string): string;
      procedure BuildByteTable();
      procedure ClearState();
      function VocabFind(const Token: string): integer; // id or -1
      function MergeRank(const A, B: string): integer;  // rank or MaxInt
      procedure BPEWord(const Symbols: TStringList; Ids: TIntegerList);
      // Reconstructs FMerges from BPE-.model pieces (HF generate_merges).
      procedure ReconstructBPEMerges(const PieceTextV: array of string;
        APieceCount: integer);
      procedure UnigramWord(const PieceText: string; Ids: TIntegerList);
      procedure EmitTokenOrFallback(const Symbol: string; Ids: TIntegerList);
      function BertNormalize(const Segment: string): string;
      procedure BertPieces(const Segment: string; Pieces: TStringList);
      procedure WordPieceWord(const WordText: string; Ids: TIntegerList);
      procedure EncodeSegment(const Segment: string; Ids: TIntegerList;
        IsFirstSegment: boolean);
      procedure ByteLevelPieces(const Segment: string; Pieces: TStringList);
      procedure SplitCl100kPieces(const Segment: string;
        Pieces: TStringList);
      procedure SplitDeepSeekPieces(const Segment: string;
        Pieces: TStringList);
      procedure SplitO200kPieces(const Segment: string;
        Pieces: TStringList);
      function MapPieceToByteLevel(const Piece: string): TStringList;
      function FindAddedToken(const Text: string; Position: integer;
        out TokenIndex: integer): boolean;
      function IsAddedTokenId(Id: integer; out TokenIndex: integer): boolean;
    public
      constructor Create();
      destructor Destroy(); override;

      // Loads a HuggingFace tokenizer.json file. A path ending in '.model'
      // (or whose first byte is not '{') is dispatched to the raw
      // SentencePiece ModelProto reader (LoadSentencePieceModel).
      procedure LoadFromFile(const FileName: string);

      // Loads a raw SentencePiece `.model` (sentencepiece.bpe.model /
      // spm.model / tokenizer.model) ModelProto protobuf, as shipped by
      // T5 / ALBERT / XLNet / DeBERTa-v3 / mBART(-Unigram) checkpoints that
      // carry no tokenizer.json. Only the UNIGRAM model_type is supported
      // (the Viterbi path); a BPE/WORD/CHAR ModelProto raises
      // EHFTokenizerError ("use tokenizer.json"). The pieces (text, score,
      // type) populate the SAME internal Unigram structures the
      // tokenizer.json-Unigram path uses, and the SentencePiece Metaspace
      // convention (add_dummy_prefix -> prepend U+2581, space -> U+2581) is
      // wired so Encode/Decode behave identically. The SentencePiece
      // precompiled_charsmap (opaque per-model normalization trie) is NOT
      // parsed/applied here; a standard NFKC/NFKD normalizer declared in a
      // tokenizer.json IS applied in full (see UnicodeNormalize).
      procedure LoadSentencePieceModel(const FileName: string);

      // Loads the tokenizer straight from the tokenizer.ggml.* metadata of a
      // .gguf checkpoint, so a single .gguf file is self-contained for
      // generation (no sidecar tokenizer.json / .model needed). Opens the
      // file through TNNetGGUFReader, reads tokenizer.ggml.model and the
      // tokens / scores / token_type (and, for gpt2, merges) arrays, and
      // populates the SAME internal structures the tokenizer.json / .model
      // loaders use, so Encode/Decode behave identically.
      //   * tokenizer.ggml.model = "llama"  -> SentencePiece-with-scores fed
      //     into the Unigram (Viterbi) path: metaspace U+2581, <0xNN> byte
      //     fallback, special pieces (token_type CONTROL/UNKNOWN/USER_DEFINED)
      //     exposed as added tokens.
      //   * tokenizer.ggml.model = "gpt2"   -> byte-level BPE: tokens + the
      //     tokenizer.ggml.merges array fed into the BPE path with the GPT-2
      //     bytes<->unicode table and the cl100k Split pre-tokenizer.
      //   * "bert" / "no_vocab" (and any other value) raise EHFTokenizerError
      //     (deferred). Special-token ids come from
      //     tokenizer.ggml.{bos,eos,unknown,padding}_token_id when present.
      procedure LoadFromGGUF(const FileName: string);

      // Emits this tokenizer's vocabulary as a GGUF tokenizer.ggml.* metadata
      // block on an open TNNetGGUFWriter, the exact inverse of LoadFromGGUF.
      // Writes tokenizer.ggml.model ("gpt2" for byte-level BPE, "llama" for
      // Unigram-with-scores), the id-indexed .tokens / .scores / .token_type
      // arrays, the .merges array (byte-level BPE only, ordered by rank), and
      // the scalar bos/eos/unknown token ids when present. The caller adds the
      // rest of the file (general.* metadata + tensors) and calls SaveToFile.
      // Raises EHFTokenizerError for the WordPiece path (no GGUF "bert" model
      // support yet) - mirrors LoadFromGGUF's supported-model set.
      procedure SaveTokenizerToGGUF(Writer: TNNetGGUFWriter);

      // Text -> token ids. Special tokens are matched if present verbatim
      // in the text but never auto-injected.
      procedure Encode(const Text: string; Ids: TIntegerList); overload;
      function Encode(const Text: string): TNeuralIntegerArray; overload;

      // Encode + offset mapping (HF return_offsets_mapping + word_ids
      // equivalent). Returns one TNeuralTokenOffset per emitted token id with
      // the (Start,Length) byte span in Text and the whitespace word index.
      // Alignment is byte-exact: each non-special token's decoded surface bytes
      // are consumed, byte for byte, against the original text from a running
      // cursor (NOT a post-hoc whole-surface search), so byte-fallback
      // fragments and metaspace/leading-space tokens that the old heuristic
      // left unmapped now all get a span. A multi-byte UTF-8 char split across
      // several <0xNN> byte tokens yields a per-byte span on each fragment
      // (each points at its byte sub-range of that same source char). Synthetic
      // prepended spaces (add_dummy_prefix / AddPrefixSpace) are dropped.
      // Added/special tokens and a true unk with no source bytes keep
      // Start=0/Length=0/WordId=-1. Works for the byte-level BPE, metaspace
      // BPE, byte-fallback Unigram and WordPiece paths.
      function EncodeWithOffsets(const Text: string): TNeuralTokenOffsetArray;

      // Token ids -> text. SkipSpecialTokens drops added special tokens
      // (BOS/EOS/...) from the output, like HF decode(skip_special_tokens).
      function Decode(const Ids: array of integer;
        SkipSpecialTokens: boolean = true): string; overload;
      function Decode(Ids: TIntegerList;
        SkipSpecialTokens: boolean = true): string; overload;
      function DecodeToken(Id: integer): string;

      // Unicode normalization of UTF-8 text. Compat=False: canonical (NFC if
      // Compose else NFD); Compat=True: compatibility (NFKC if Compose else
      // NFKD). Exposed for testing and reuse; the encode path applies the
      // normalizer declared in the tokenizer.json automatically.
      class function NormalizeUnicodeText(const Text: string;
        Compose: boolean; Compat: boolean = false): string;

      // TOKEN-HEALING vocab prefix scan (byte-level-BPE sibling of
      // neuraldecode's TStringListInt PrepareTokenHealing). Returns every
      // vocabulary id whose STORED SURFACE has Fragment as a prefix, after
      // mapping Fragment into the SAME surface space the vocab is stored in
      // (byte-level: each raw byte -> its GPT-2 byte-alphabet codepoint;
      // metaspace: spaces -> U+2581; WordPiece/plain: raw). So healing can
      // re-roll the boundary token to any vocab entry that EXTENDS the
      // trailing prompt fragment. Matching happens in surface space, so a
      // multi-byte UTF-8 fragment whose bytes map to several alphabet
      // codepoints still scans correctly. Added (special) tokens are not
      // included. Empty Fragment -> empty result.
      function PrefixScanVocab(const Fragment: string): TNeuralIntegerArray;
      // Maps a raw text Fragment into the storage surface space used by
      // PrefixScanVocab / FIdToToken (byte alphabet / metaspace / raw).
      // Exposed for callers that want to build the healing constraint
      // themselves; mirrors what DecodeToken inverts.
      function FragmentToSurface(const Fragment: string): string;

      function GetVocabSize(): integer;
      function TokenToId(const Token: string): integer; // -1 if absent
      function IdToToken(Id: integer): string;
      function AddedTokenCount(): integer;
      function GetAddedToken(Index: integer): TNeuralAddedToken;

      property UnkId: integer read FUnkId; // -1 if none
      property BosId: integer read FBosId; // -1 if none
      property EosId: integer read FEosId; // -1 if none
      property IsByteLevel: boolean read FByteLevel;
      property IsWordPiece: boolean read FWordPiece;
      property IsUnigram: boolean read FUnigram;
      // BPE-dropout merge-skip probability for the rank-based BPE merge loop.
      // 0.0 (default) = deterministic / OFF (bit-identical eval); set to e.g.
      // 0.1 during TRAINING only to subword-regularize. Uses the global RNG,
      // so reseed RandSeed for reproducible segmentations. No effect on the
      // WordPiece or Unigram paths (only the BPE merge loop drops merges).
      property DropoutProb: TNeuralFloat read FDropoutProb write FDropoutProb;
  end;

// fpjson portability helpers, exported for the other HF-JSON readers
// (neuralchat.pas reads tokenizer_config.json with them):
//   * HFDecodeUnicodeEscapes decodes \uXXXX escapes (incl. surrogate
//     pairs) to raw UTF-8 up front, because fpjson pipes them through the
//     widestring manager and mangles non-ASCII to '?' in programs without
//     LazUTF8/cwstring.
//   * HFParseJSONRaw parses WITHOUT joUTF8 (TJSONParser.Create(S, [])),
//     so string values keep their raw UTF-8 bytes.
function HFDecodeUnicodeEscapes(const S: string): string;
function HFParseJSONRaw(const S: string): TJSONData;

implementation

uses
  jsonparser, Math, StrUtils;

const
  csMergeSep = #1;
  csMetaspace = #$E2#$96#$81; // '▁' U+2581 in UTF-8
  // The two cl100k-style Split pre_tokenizer patterns shipped with the
  // already-importable checkpoint families (matched VERBATIM; anything
  // else raises EHFTokenizerError -- no regex engine is vendored).
  // Qwen2/Qwen2.5/Qwen3: single-digit \p{N}.
  csQwen2SplitPattern =
    '(?i:''s|''t|''re|''ve|''m|''ll|''d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|' +
    ' ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';
  // Llama-3 / cl100k / GPT-4 style: digit runs capped at 3.
  csCl100kSplitPattern =
    '(?i:''s|''t|''re|''ve|''m|''ll|''d)|[^\r\n\p{L}\p{N}]?\p{L}+|' +
    '\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';
  // o200k_base / GPT-4o family (gpt-4o, gpt-4o-mini, o1/o3). Differs from
  // cl100k: the single letter-run alternation is split into two case-aware
  // alternations ([\p{Lu}...]*[\p{Ll}...]+ and [\p{Lu}...]+[\p{Ll}...]*),
  // the contraction suffix rides each letter run, \p{N}{1,3} digit groups,
  // and the punct run trails [\r\n/]* (note the extra '/'). Matched VERBATIM
  // -- dispatched to SplitO200kPieces.
  csO200kSplitPattern =
    '[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*' +
    '[\p{Ll}\p{Lo}\p{M}]+(?i:''s|''t|''re|''ve|''m|''ll|''d)?|' +
    '[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+' +
    '[\p{Ll}\p{Lo}\p{M}]*(?i:''s|''t|''re|''ve|''m|''ll|''d)?|' +
    '\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+';

// Compact CANONICAL Unicode tables (decomposition, composition, combining
// class) backing the NFD/NFC normalizers below. AUTO-GENERATED from Python
// unicodedata by tools/gen_nfc_tables.py; regenerate with
//   /home/bpsa/x/bin/python tools/gen_nfc_tables.py neural/neuralnfctables.inc
// Hangul is handled algorithmically (NOT tabled); NFKC/NFKD compatibility
// decompositions ARE tabled (cNFKCDecompIdx/cNFKCDecompPool, see
// UnicodeNormalize).
{$include neuralnfctables.inc}

{ ---------------------------------------------------------------- }
{ UTF-8 helpers                                                     }
{ ---------------------------------------------------------------- }

// Decodes the codepoint starting at byte Position (1-based); advances
// Position past it. Invalid bytes are returned as-is (latin-1 style).
function NextCodePoint(const S: string; var Position: integer): cardinal;
var
  B: byte;
  Extra, Cnt: integer;
begin
  B := Ord(S[Position]);
  Inc(Position);
  if B < $80 then Exit(B);
  if (B and $E0) = $C0 then begin Result := B and $1F; Extra := 1; end
  else if (B and $F0) = $E0 then begin Result := B and $0F; Extra := 2; end
  else if (B and $F8) = $F0 then begin Result := B and $07; Extra := 3; end
  else Exit(B); // invalid lead byte
  for Cnt := 1 to Extra do
  begin
    if (Position > Length(S)) or ((Ord(S[Position]) and $C0) <> $80) then
      Exit(B); // truncated/invalid sequence
    Result := (Result shl 6) or (cardinal(Ord(S[Position])) and $3F);
    Inc(Position);
  end;
end;

function CodePointToUTF8(CP: cardinal): string;
begin
  if CP < $80 then
    Result := Chr(CP)
  else if CP < $800 then
    Result := Chr($C0 or (CP shr 6)) + Chr($80 or (CP and $3F))
  else if CP < $10000 then
    Result := Chr($E0 or (CP shr 12)) + Chr($80 or ((CP shr 6) and $3F)) +
      Chr($80 or (CP and $3F))
  else
    Result := Chr($F0 or (CP shr 18)) + Chr($80 or ((CP shr 12) and $3F)) +
      Chr($80 or ((CP shr 6) and $3F)) + Chr($80 or (CP and $3F));
end;

{ ---------------------------------------------------------------- }
{ Unicode canonical normalization (NFD / NFC)                       }
{                                                                   }
{ Real canonical decomposition + canonical ordering (NFD) and       }
{ canonical composition (NFC), driven by the compact tables in      }
{ neuralnfctables.inc plus the algorithmic Hangul L/V/T arithmetic. }
{ Coded by Claude (AI).                                             }
{ ---------------------------------------------------------------- }

const
  cHangulSBase = $AC00;
  cHangulLBase = $1100;
  cHangulVBase = $1161;
  cHangulTBase = $11A7;
  cHangulLCount = 19;
  cHangulVCount = 21;
  cHangulTCount = 28;
  cHangulNCount = cHangulVCount * cHangulTCount; // 588
  cHangulSCount = cHangulLCount * cHangulNCount; // 11172

// Canonical_Combining_Class for CP (0 when unlisted). Binary search.
function NFCCombiningClass(CP: cardinal): integer;
var
  Lo, Hi, Mid: integer;
begin
  Lo := 0;
  Hi := cNFCCombClassCount - 1;
  while Lo <= Hi do
  begin
    Mid := (Lo + Hi) div 2;
    if cNFCCombClass[Mid][0] = CP then Exit(integer(cNFCCombClass[Mid][1]))
    else if cNFCCombClass[Mid][0] < CP then Lo := Mid + 1
    else Hi := Mid - 1;
  end;
  Result := 0;
end;

// One-step canonical decomposition of CP. Returns True and fills A (and B,
// or 0 for a singleton) when CP has a tabled canonical decomposition.
function NFCDecomposeStep(CP: cardinal; out A, B: cardinal): boolean;
var
  Lo, Hi, Mid: integer;
begin
  Lo := 0;
  Hi := cNFCDecompCount - 1;
  while Lo <= Hi do
  begin
    Mid := (Lo + Hi) div 2;
    if cNFCDecomp[Mid][0] = CP then
    begin
      A := cNFCDecomp[Mid][1];
      B := cNFCDecomp[Mid][2];
      Exit(True);
    end
    else if cNFCDecomp[Mid][0] < CP then Lo := Mid + 1
    else Hi := Mid - 1;
  end;
  Result := False;
end;

// Canonical composition of a starter A with following char B. Returns the
// composite CP or 0 when (A,B) does not canonically compose. Hangul L+V and
// LV+T compositions are handled algorithmically.
function NFCComposePair(A, B: cardinal): cardinal;
var
  Lo, Hi, Mid: integer;
  LIndex, VIndex, TIndex, SIndex: integer;
begin
  // Hangul: L + V
  if (A >= cHangulLBase) and (A < cHangulLBase + cHangulLCount) and
     (B >= cHangulVBase) and (B < cHangulVBase + cHangulVCount) then
  begin
    LIndex := integer(A) - cHangulLBase;
    VIndex := integer(B) - cHangulVBase;
    Exit(cardinal(cHangulSBase + (LIndex * cHangulVCount + VIndex) * cHangulTCount));
  end;
  // Hangul: LV + T
  if (A >= cHangulSBase) and (A < cHangulSBase + cHangulSCount) and
     (((integer(A) - cHangulSBase) mod cHangulTCount) = 0) and
     (B > cHangulTBase) and (B < cHangulTBase + cHangulTCount) then
  begin
    SIndex := integer(A) - cHangulSBase;
    TIndex := integer(B) - cHangulTBase;
    Exit(cardinal(integer(A) + TIndex));
  end;
  Lo := 0;
  Hi := cNFCComposeCount - 1;
  while Lo <= Hi do
  begin
    Mid := (Lo + Hi) div 2;
    if cNFCCompose[Mid][0] = A then
    begin
      if cNFCCompose[Mid][1] = B then Exit(cNFCCompose[Mid][2])
      else if cNFCCompose[Mid][1] < B then Lo := Mid + 1
      else Hi := Mid - 1;
    end
    else if cNFCCompose[Mid][0] < A then Lo := Mid + 1
    else Hi := Mid - 1;
  end;
  Result := 0;
end;

// Compatibility (NFKD) decomposition of CP. Returns True and the (Offset,
// Length) span into cNFKCDecompPool of CP's full NFKD expansion when CP has a
// tabled compatibility decomposition. Binary search by CP.
function NFKCDecomposeStep(CP: cardinal; out Offset, Count: integer): boolean;
var
  Lo, Hi, Mid: integer;
begin
  Lo := 0;
  Hi := cNFKCDecompCount - 1;
  while Lo <= Hi do
  begin
    Mid := (Lo + Hi) div 2;
    if cNFKCDecompIdx[Mid][0] = CP then
    begin
      Offset := integer(cNFKCDecompIdx[Mid][1]);
      Count := integer(cNFKCDecompIdx[Mid][2]);
      Exit(True);
    end
    else if cNFKCDecompIdx[Mid][0] < CP then Lo := Mid + 1
    else Hi := Mid - 1;
  end;
  Result := False;
end;

// Recursively decompose CP, appending resulting codepoints to Buf (1-based
// dynamic array; Len is the live length). Hangul is decomposed
// algorithmically. When Compat is True the COMPATIBILITY (NFKD) decomposition
// is applied (compat-tagged mappings are pre-expanded fully in the table, so
// no recursion through them is needed); otherwise pure canonical (NFD).
procedure NFCDecomposeRec(CP: cardinal; var Buf: array of cardinal;
  var Len: integer; Compat: boolean);
var
  A, B: cardinal;
  SIndex, LIndex, VIndex, TIndex: integer;
  Offset, Count, K: integer;
begin
  if Compat and NFKCDecomposeStep(CP, Offset, Count) then
  begin
    // Pool holds the FULL recursive NFKD expansion; any canonical parts in it
    // are already canonical (NFKD result), so copy verbatim.
    for K := 0 to Count - 1 do
    begin
      Buf[Len] := cNFKCDecompPool[Offset + K]; Inc(Len);
    end;
    Exit;
  end;
  if (CP >= cHangulSBase) and (CP < cHangulSBase + cHangulSCount) then
  begin
    SIndex := integer(CP) - cHangulSBase;
    LIndex := SIndex div cHangulNCount;
    VIndex := (SIndex mod cHangulNCount) div cHangulTCount;
    TIndex := SIndex mod cHangulTCount;
    Buf[Len] := cardinal(cHangulLBase + LIndex); Inc(Len);
    Buf[Len] := cardinal(cHangulVBase + VIndex); Inc(Len);
    if TIndex > 0 then
    begin
      Buf[Len] := cardinal(cHangulTBase + TIndex); Inc(Len);
    end;
    Exit;
  end;
  if NFCDecomposeStep(CP, A, B) then
  begin
    NFCDecomposeRec(A, Buf, Len, Compat);
    if B <> 0 then NFCDecomposeRec(B, Buf, Len, Compat);
  end
  else
  begin
    Buf[Len] := CP; Inc(Len);
  end;
end;

// Canonical ordering: stable sort runs of combining marks (ccc<>0) by
// combining class (insertion sort on the ccc key is stable and exact).
procedure NFCCanonicalOrder(var Buf: array of cardinal; Len: integer);
var
  I, CCI, CCPrev: integer;
  Tmp: cardinal;
begin
  I := 1;
  while I < Len do
  begin
    CCI := NFCCombiningClass(Buf[I]);
    if CCI <> 0 then
    begin
      CCPrev := NFCCombiningClass(Buf[I - 1]);
      if (CCPrev <> 0) and (CCPrev > CCI) then
      begin
        Tmp := Buf[I];
        Buf[I] := Buf[I - 1];
        Buf[I - 1] := Tmp;
        if I > 1 then begin Dec(I); continue; end;
      end;
    end;
    Inc(I);
  end;
end;

// Returns S normalized to one of the four forms. Input and output are UTF-8.
//   Compat=False: canonical -- NFD (Compose=False) / NFC (Compose=True).
//   Compat=True : compatibility -- NFKD (Compose=False) / NFKC (Compose=True);
//                 compatibility decompositions (ligatures, full/half-width,
//                 super/subscripts, fractions, no-break space, ...) are folded
//                 BEFORE canonical ordering, then canonical compose for NFKC.
function UnicodeNormalize(const S: string; Compose: boolean;
  Compat: boolean = false): string;
var
  Buf: array of cardinal;
  Len, Position, CP, OutLen: integer;
  StarterPos: integer;
  StarterCC, CC, Composite: integer;
  I: integer;
begin
  if S = '' then Exit('');
  // Worst-case compatibility expansion is bounded but larger than canonical
  // (some compat decompositions emit up to ~18 codepoints); 18x the input
  // codepoint count is a safe over-allocation for both modes.
  SetLength(Buf, Length(S) * 18 + 8);
  Len := 0;
  Position := 1;
  while Position <= Length(S) do
  begin
    CP := integer(NextCodePoint(S, Position));
    NFCDecomposeRec(cardinal(CP), Buf, Len, Compat);
  end;
  NFCCanonicalOrder(Buf, Len);

  if Compose then
  begin
    // Canonical composition (Unicode 3.11 algorithm). StarterPos tracks the
    // last starter we may still compose onto; StarterCC is the ccc of the
    // most recent character between the starter and the candidate.
    OutLen := 0;
    StarterPos := -1;
    StarterCC := 0;
    for I := 0 to Len - 1 do
    begin
      CC := NFCCombiningClass(Buf[I]);
      // Candidate Buf[I] may compose onto the starter unless it is BLOCKED:
      // it is blocked when some character sits between the starter and the
      // candidate whose ccc is >= the candidate's ccc (StarterCC tracks the
      // max ccc seen since the starter). A candidate immediately after the
      // starter (StarterCC = 0) is never blocked.
      if (StarterPos >= 0) and ((StarterCC < CC) or
         ((StarterCC = 0) and (CC = 0))) then
      begin
        Composite := integer(NFCComposePair(Buf[StarterPos], Buf[I]));
        if (Composite <> 0) and (NFCCombiningClass(cardinal(Composite)) = 0)
        then
        begin
          Buf[StarterPos] := cardinal(Composite);
          continue; // candidate consumed; keep same starter
        end;
      end;
      if CC = 0 then
      begin
        StarterPos := OutLen;
        StarterCC := 0;
      end
      else
        StarterCC := CC;
      Buf[OutLen] := Buf[I];
      Inc(OutLen);
    end;
    Len := OutLen;
  end;

  Result := '';
  for I := 0 to Len - 1 do
    Result := Result + CodePointToUTF8(Buf[I]);
end;

{ ---------------------------------------------------------------- }
{ Unicode character classes for the GPT-2 pre-tokenization regex    }
{ ---------------------------------------------------------------- }

function IsWhitespaceCP(CP: cardinal): boolean;
begin
  Result := ((CP >= 9) and (CP <= 13)) or (CP = 32) or (CP = $85) or
    (CP = $A0) or (CP = $1680) or ((CP >= $2000) and (CP <= $200A)) or
    (CP = $2028) or (CP = $2029) or (CP = $202F) or (CP = $205F) or
    (CP = $3000);
end;

// \p{L} over the major scripts (see unit header for the approximation).
function IsLetterCP(CP: cardinal): boolean;
begin
  Result :=
    ((CP >= Ord('A')) and (CP <= Ord('Z'))) or
    ((CP >= Ord('a')) and (CP <= Ord('z'))) or
    (CP = $AA) or (CP = $B5) or (CP = $BA) or
    ((CP >= $C0) and (CP <= $D6)) or
    ((CP >= $D8) and (CP <= $F6)) or
    ((CP >= $F8) and (CP <= $2C1)) or            // Latin Extended-A/B, IPA
    ((CP >= $370) and (CP <= $3FF) and (CP <> $375) and (CP <> $378) and
      (CP <> $379) and (CP <> $37E) and (CP <> $384) and (CP <> $385) and
      (CP <> $387) and (CP <> $38B) and (CP <> $38D) and (CP <> $3A2)) or
    ((CP >= $400) and (CP <= $481)) or           // Cyrillic
    ((CP >= $48A) and (CP <= $52F)) or
    ((CP >= $531) and (CP <= $556)) or           // Armenian
    ((CP >= $561) and (CP <= $587)) or
    ((CP >= $5D0) and (CP <= $5EA)) or           // Hebrew
    ((CP >= $620) and (CP <= $64A)) or           // Arabic
    ((CP >= $671) and (CP <= $6D3)) or
    ((CP >= $904) and (CP <= $939)) or           // Devanagari
    ((CP >= $1E00) and (CP <= $1FBC)) or         // Latin Ext Additional/Greek Ext
    ((CP >= $3041) and (CP <= $3096)) or         // Hiragana
    ((CP >= $30A1) and (CP <= $30FA)) or         // Katakana
    ((CP >= $4E00) and (CP <= $9FFF)) or         // CJK Unified
    ((CP >= $AC00) and (CP <= $D7A3));           // Hangul
end;

// \p{N} over the common ranges (Nd digits + frequent No like ², ½).
function IsNumberCP(CP: cardinal): boolean;
begin
  Result :=
    ((CP >= Ord('0')) and (CP <= Ord('9'))) or
    (CP = $B2) or (CP = $B3) or (CP = $B9) or
    ((CP >= $BC) and (CP <= $BE)) or
    ((CP >= $660) and (CP <= $669)) or           // Arabic-Indic
    ((CP >= $6F0) and (CP <= $6F9)) or
    ((CP >= $966) and (CP <= $96F)) or           // Devanagari
    ((CP >= $FF10) and (CP <= $FF19));           // Fullwidth
end;

// [^\s\p{L}\p{N}] -- the "other"/punctuation class of the byte-level
// pre-tokenization regexes.
function IsOtherCP(CP: cardinal): boolean;
begin
  Result := (not IsWhitespaceCP(CP)) and (not IsLetterCP(CP)) and
    (not IsNumberCP(CP));
end;

// Punctuation for the BERT pre-tokenizer: HF treats every ASCII
// non-alphanumeric printable as punctuation (33-47, 58-64, 91-96, 123-126)
// plus Unicode category P. The category-P part is approximated over the
// common blocks (Latin-1 punctuation, General Punctuation, CJK symbols,
// fullwidth forms) -- same stance as IsLetterCP above.
function IsBertPunctuationCP(CP: cardinal): boolean;
begin
  Result :=
    ((CP >= 33) and (CP <= 47)) or
    ((CP >= 58) and (CP <= 64)) or
    ((CP >= 91) and (CP <= 96)) or
    ((CP >= 123) and (CP <= 126)) or
    (CP = $A1) or (CP = $A7) or (CP = $AB) or (CP = $B6) or (CP = $B7) or
    (CP = $BB) or (CP = $BF) or
    ((CP >= $2010) and (CP <= $2027)) or       // General Punctuation
    ((CP >= $2030) and (CP <= $205E)) or
    ((CP >= $3001) and (CP <= $3003)) or       // CJK punctuation
    ((CP >= $3008) and (CP <= $3011)) or
    ((CP >= $FF01) and (CP <= $FF0F)) or       // fullwidth ASCII punct
    ((CP >= $FF1A) and (CP <= $FF20)) or
    ((CP >= $FF3B) and (CP <= $FF40)) or
    ((CP >= $FF5B) and (CP <= $FF65));
end;

// CJK ideographs for BertNormalizer handle_chinese_chars (HF's
// _is_chinese_char ranges).
function IsCJKIdeographCP(CP: cardinal): boolean;
begin
  Result :=
    ((CP >= $4E00) and (CP <= $9FFF)) or
    ((CP >= $3400) and (CP <= $4DBF)) or
    ((CP >= $F900) and (CP <= $FAFF)) or
    ((CP >= $20000) and (CP <= $2A6DF)) or
    ((CP >= $2A700) and (CP <= $2B73F)) or
    ((CP >= $2B740) and (CP <= $2B81F)) or
    ((CP >= $2B820) and (CP <= $2CEAF)) or
    ((CP >= $2F800) and (CP <= $2FA1F));
end;

// NFD base letter for strip_accents over Latin-1 Supplement and Latin
// Extended-A: returns the unaccented base codepoint, or 0 when the
// character has no canonical decomposition (ae/oe ligatures, stroked
// letters, thorn, eth, dotless i, ...) or is outside the covered blocks.
// Combining marks themselves (U+0300..U+036F) return 1 meaning "drop".
function StripAccentBaseCP(CP: cardinal): cardinal;
begin
  Result := 0;
  if (CP >= $300) and (CP <= $36F) then Exit(1); // bare combining mark
  case CP of
    // Latin-1 Supplement
    $C0..$C5: Result := Ord('A');
    $C7: Result := Ord('C');
    $C8..$CB: Result := Ord('E');
    $CC..$CF: Result := Ord('I');
    $D1: Result := Ord('N');
    $D2..$D6: Result := Ord('O');
    $D9..$DC: Result := Ord('U');
    $DD: Result := Ord('Y');
    $E0..$E5: Result := Ord('a');
    $E7: Result := Ord('c');
    $E8..$EB: Result := Ord('e');
    $EC..$EF: Result := Ord('i');
    $F1: Result := Ord('n');
    $F2..$F6: Result := Ord('o');
    $F9..$FC: Result := Ord('u');
    $FD, $FF: Result := Ord('y');
    // Latin Extended-A (pairs upper/lower; gaps = no decomposition)
    $100, $102, $104: Result := Ord('A');
    $101, $103, $105: Result := Ord('a');
    $106, $108, $10A, $10C: Result := Ord('C');
    $107, $109, $10B, $10D: Result := Ord('c');
    $10E: Result := Ord('D');
    $10F: Result := Ord('d');
    $112, $114, $116, $118, $11A: Result := Ord('E');
    $113, $115, $117, $119, $11B: Result := Ord('e');
    $11C, $11E, $120, $122: Result := Ord('G');
    $11D, $11F, $121, $123: Result := Ord('g');
    $124: Result := Ord('H');
    $125: Result := Ord('h');
    $128, $12A, $12C, $12E, $130: Result := Ord('I');
    $129, $12B, $12D, $12F: Result := Ord('i');
    $134: Result := Ord('J');
    $135: Result := Ord('j');
    $136: Result := Ord('K');
    $137: Result := Ord('k');
    $139, $13B, $13D: Result := Ord('L');
    $13A, $13C, $13E: Result := Ord('l');
    $143, $145, $147: Result := Ord('N');
    $144, $146, $148: Result := Ord('n');
    $14C, $14E, $150: Result := Ord('O');
    $14D, $14F, $151: Result := Ord('o');
    $154, $156, $158: Result := Ord('R');
    $155, $157, $159: Result := Ord('r');
    $15A, $15C, $15E, $160: Result := Ord('S');
    $15B, $15D, $15F, $161: Result := Ord('s');
    $162, $164: Result := Ord('T');
    $163, $165: Result := Ord('t');
    $168, $16A, $16C, $16E, $170, $172: Result := Ord('U');
    $169, $16B, $16D, $16F, $171, $173: Result := Ord('u');
    $174: Result := Ord('W');
    $175: Result := Ord('w');
    $176, $178: Result := Ord('Y');
    $177: Result := Ord('y');
    $179, $17B, $17D: Result := Ord('Z');
    $17A, $17C, $17E: Result := Ord('z');
  end;
end;

// Lowercase for the BertNormalizer over ASCII, Latin-1 and Latin
// Extended-A (after strip_accents most input is ASCII anyway).
function LowercaseCP(CP: cardinal): cardinal;
begin
  Result := CP;
  if (CP >= Ord('A')) and (CP <= Ord('Z')) then Exit(CP + 32);
  if ((CP >= $C0) and (CP <= $DE)) and (CP <> $D7) then Exit(CP + 32);
  if CP = $178 then Exit($FF); // Y diaeresis
  // Latin Extended-A: upper/lower alternate in pairs.
  if (CP >= $100) and (CP <= $177) then
  begin
    if ((CP <= $137) or ((CP >= $14A) and (CP <= $177))) and
      ((CP and 1) = 0) then Exit(CP + 1);
    if (CP >= $139) and (CP <= $148) and ((CP and 1) = 1) then Exit(CP + 1);
  end;
  if (CP >= $179) and (CP <= $17E) and ((CP and 1) = 1) then Exit(CP + 1);
end;

{ ---------------------------------------------------------------- }
{ fpjson portability helpers                                        }
{ ---------------------------------------------------------------- }

// fpjson converts \uXXXX escapes through the widestring manager; without
// one installed (plain FPC programs that don't use LazUTF8/cwstring) every
// non-ASCII escape decodes to '?'. Decode them to raw UTF-8 up front so
// parsing is environment-independent. Backslash handling is exact: the
// scan steps over every two-character escape, so "\\u0041" (literal
// backslash) is never decoded.
function DecodeUnicodeEscapes(const S: string): string;
var
  Position, Total: integer;
  CP, LowCP: cardinal;

  function HexAt(StartPos: integer; out Value: cardinal): boolean;
  var
    Cnt: integer;
    Digit: char;
  begin
    Value := 0;
    Result := StartPos + 3 <= Length(S);
    if not Result then exit;
    for Cnt := StartPos to StartPos + 3 do
    begin
      Digit := S[Cnt];
      case Digit of
        '0'..'9': Value := (Value shl 4) or cardinal(Ord(Digit) - Ord('0'));
        'a'..'f': Value := (Value shl 4) or cardinal(Ord(Digit) - Ord('a') + 10);
        'A'..'F': Value := (Value shl 4) or cardinal(Ord(Digit) - Ord('A') + 10);
        else Exit(false);
      end;
    end;
  end;

var
  OutPos, ByteCnt: integer;
  Encoded: string;

  // Writes into the preallocated Result buffer. Plain Result := Result + c
  // is QUADRATIC in this unit: {$CODEPAGE UTF8} (via neuralnetwork.inc)
  // defeats FPC's in-place append optimization, so every concat copies the
  // whole accumulated string -- a 466 KB BERT tokenizer.json then takes
  // minutes instead of milliseconds. Escapes never grow the text (6 or 12
  // escape bytes become at most 4 UTF-8 bytes), so Length(S) is a safe
  // upper bound for the output size.
  procedure AppendChar(C: char); inline;
  begin
    Inc(OutPos);
    Result[OutPos] := C;
  end;

begin
  Total := Length(S);
  SetLength(Result, Total);
  OutPos := 0;
  Position := 1;
  while Position <= Total do
  begin
    if (S[Position] = '\') and (Position < Total) then
    begin
      if (S[Position + 1] = 'u') and HexAt(Position + 2, CP) then
      begin
        Inc(Position, 6);
        if (CP >= $D800) and (CP <= $DBFF) and (Position + 1 <= Total) and
          (S[Position] = '\') and (S[Position + 1] = 'u') and
          HexAt(Position + 2, LowCP) and (LowCP >= $DC00) and
          (LowCP <= $DFFF) then
        begin // surrogate pair
          CP := $10000 + ((CP - $D800) shl 10) + (LowCP - $DC00);
          Inc(Position, 6);
        end;
        Encoded := CodePointToUTF8(CP);
        for ByteCnt := 1 to Length(Encoded) do AppendChar(Encoded[ByteCnt]);
      end
      else
      begin // any other escape: copy verbatim, never reinterpret char 2
        AppendChar(S[Position]);
        AppendChar(S[Position + 1]);
        Inc(Position, 2);
      end;
    end
    else
    begin
      AppendChar(S[Position]);
      Inc(Position);
    end;
  end;
  SetLength(Result, OutPos);
end;

function HFDecodeUnicodeEscapes(const S: string): string;
begin
  Result := DecodeUnicodeEscapes(S);
end;

{ ---------------------------------------------------------------- }
{ TNeuralHFTokenizer                                                }
{ ---------------------------------------------------------------- }

constructor TNeuralHFTokenizer.Create();
begin
  inherited Create();
  FVocab := TStringList.Create();
  FVocab.Sorted := true;
  FVocab.CaseSensitive := true;
  FVocab.UseLocale := false;
  FVocab.Duplicates := dupIgnore;
  FMerges := TStringList.Create();
  FMerges.Sorted := true;
  FMerges.CaseSensitive := true;
  FMerges.UseLocale := false;
  FMerges.Duplicates := dupIgnore;
  FDropoutProb := 0.0;
  BuildByteTable();
  ClearState();
end;

destructor TNeuralHFTokenizer.Destroy();
begin
  FMerges.Free;
  FVocab.Free;
  inherited Destroy();
end;

procedure TNeuralHFTokenizer.ClearState();
begin
  FVocab.Clear;
  FMerges.Clear;
  SetLength(FIdToToken, 0);
  SetLength(FAddedTokens, 0);
  SetLength(FNormReplaceFrom, 0);
  SetLength(FNormReplaceTo, 0);
  SetLength(FDecReplaceFrom, 0);
  SetLength(FDecReplaceTo, 0);
  FByteFallback := false;
  FFuseUnk := false;
  FIgnoreMerges := false;
  FUnigram := false;
  SetLength(FUniScore, 0);
  FUniMinScore := 0;
  FByteLevel := false;
  FAddPrefixSpace := false;
  FByteLevelUseRegex := true;
  FSplitPreTok := false;
  FSplitDigitsMax := 1;
  FDeepSeekPreTok := false;
  FO200kPreTok := false;
  FMetaspacePreTok := false;
  FMSReplacement := csMetaspace;
  FMSPrependScheme := 'always';
  FMSSplit := true;
  FWordPiece := false;
  FWPPrefix := '##';
  FWPMaxChars := 100;
  FBertLowercase := false;
  FBertStripAccents := false;
  FBertCleanText := false;
  FBertHandleChinese := false;
  FDecWordPieceCleanup := false;
  FNormPrepend := '';
  FNormUnicodeForm := '';
  FDecStripLeft := 0;
  FUnkId := -1;
  FBosId := -1;
  FEosId := -1;
end;

// The GPT-2 bytes-to-unicode table: printable latin-1 bytes map to
// themselves; the rest map to 256, 257, ... in byte order.
procedure TNeuralHFTokenizer.BuildByteTable();
var
  B, Shifted: integer;
  CP: cardinal;
begin
  for B := 0 to High(FCPToByte) do FCPToByte[B] := -1;
  Shifted := 0;
  for B := 0 to 255 do
  begin
    if ((B >= 33) and (B <= 126)) or ((B >= 161) and (B <= 172)) or
      (B >= 174) then
      CP := cardinal(B)
    else
    begin
      CP := cardinal(256 + Shifted);
      Inc(Shifted);
    end;
    FByteToCP[B] := CP;
    FCPToByte[CP] := B;
  end;
end;

// fpjson stores object keys as raw bytes but converts them through the
// widestring manager on retrieval; in programs without a UTF-8 manager
// (no LazUTF8/cwstring) each raw UTF-8 byte of a key comes back latin-1
// re-encoded. The conversion is injective, so probe once with a known
// non-ASCII key and invert it when present.
// Parses JSON WITHOUT joUTF8: with that option (the GetJSON default) the
// scanner pipes string values through the widestring manager, which turns
// non-ASCII into '?' in programs without a UTF-8 manager. Without it the
// scanner copies raw bytes, which is exactly right for UTF-8 files whose
// \uXXXX escapes were already decoded by DecodeUnicodeEscapes.
function ParseJSONRaw(const S: string): TJSONData;
var
  Parser: TJSONParser;
begin
  Parser := TJSONParser.Create(S, []);
  try
    Result := Parser.Parse;
  finally
    Parser.Free;
  end;
end;

function HFParseJSONRaw(const S: string): TJSONData;
begin
  Result := ParseJSONRaw(S);
end;

procedure TNeuralHFTokenizer.DetectKeyMangling();
var
  Probe: TJSONData;
  Key: string;
begin
  Probe := ParseJSONRaw('{"' + #$C3#$A2 + '": 1}');
  try
    Key := TJSONObject(Probe).Names[0];
  finally
    Probe.Free;
  end;
  if Key = #$C3#$A2 then
    FKeysMangled := false
  else if Key = #$C3#$83#$C2#$A2 then
    FKeysMangled := true
  else
    raise EHFTokenizerError.Create('Unsupported string codepage ' +
      'environment: JSON keys decode neither raw nor latin-1 re-encoded. ' +
      'Add the LazUTF8 (or cwstring) unit to your program.');
end;

function TNeuralHFTokenizer.FixJSONKey(const Key: string): string;
var
  Position: integer;
  CP: cardinal;
begin
  if not FKeysMangled then Exit(Key);
  Result := '';
  Position := 1;
  while Position <= Length(Key) do
  begin
    CP := NextCodePoint(Key, Position);
    if CP > 255 then Exit(Key); // not the latin-1 pattern: keep verbatim
    Result := Result + Chr(CP);
  end;
end;

function TNeuralHFTokenizer.VocabFind(const Token: string): integer;
var
  Idx: integer;
begin
  if FVocab.Find(Token, Idx)
  then Result := integer(PtrInt(FVocab.Objects[Idx]))
  else Result := -1;
end;

function TNeuralHFTokenizer.MergeRank(const A, B: string): integer;
var
  Idx: integer;
begin
  if FMerges.Find(A + csMergeSep + B, Idx)
  then Result := integer(PtrInt(FMerges.Objects[Idx]))
  else Result := High(integer);
end;

procedure TNeuralHFTokenizer.LoadFromFile(const FileName: string);
var
  Root: TJSONData;
  RootObj, ModelObj, Entry: TJSONObject;
  VocabObj: TJSONObject;
  MergesArr, AddedArr, SubArr, VocabArr: TJSONArray;
  FS: TFileStream;
  Cnt, TokenId, MaxId: integer;
  Score: double;
  Left, Right, MergeStr, NormType, Content: string;
  SpacePos: integer;
  Node: TJSONData;

  procedure AddNormalizer(NormObj: TJSONObject);
  var
    Kind: string;
    InnerArr: TJSONArray;
    InnerCnt: integer;
  begin
    Kind := NormObj.Get('type', '');
    if Kind = 'Sequence' then
    begin
      InnerArr := NormObj.Arrays['normalizers'];
      for InnerCnt := 0 to InnerArr.Count - 1 do
        AddNormalizer(InnerArr.Objects[InnerCnt]);
    end
    else if Kind = 'Prepend' then
      FNormPrepend := NormObj.Get('prepend', '')
    else if Kind = 'Replace' then
    begin
      SetLength(FNormReplaceFrom, Length(FNormReplaceFrom) + 1);
      SetLength(FNormReplaceTo, Length(FNormReplaceTo) + 1);
      FNormReplaceFrom[High(FNormReplaceFrom)] :=
        NormObj.Objects['pattern'].Get('String', '');
      FNormReplaceTo[High(FNormReplaceTo)] := NormObj.Get('content', '');
    end
    else if Kind = 'BertNormalizer' then
    begin
      FBertCleanText := NormObj.Get('clean_text', true);
      FBertHandleChinese := NormObj.Get('handle_chinese_chars', true);
      FBertLowercase := NormObj.Get('lowercase', true);
      // strip_accents: null means "follow lowercase" (HF semantics)
      Node := NormObj.Find('strip_accents');
      if (Node = nil) or Node.IsNull then
        FBertStripAccents := FBertLowercase
      else
        FBertStripAccents := Node.AsBoolean;
    end
    else if (Kind = 'NFC') or (Kind = 'NFD') then
      // REAL canonical Unicode normalization (Qwen2/3 ship a bare NFC
      // normalizer). Applied to input by UnicodeNormalize before any
      // pre-tokenization, driven by the compact canonical tables in
      // neuralnfctables.inc + algorithmic Hangul. Last writer wins inside a
      // Sequence (HF applies them in order, but canonical NFC/NFD are
      // idempotent so the final form is what matters).
      FNormUnicodeForm := Kind
    else if (Kind = 'NFKC') or (Kind = 'NFKD') then
      // REAL COMPATIBILITY normalization. UnicodeNormalize applies the
      // compatibility decomposition mappings in neuralnfctables.inc
      // (cNFKCDecompIdx/Pool: ligatures, full/half-width forms, super/
      // subscripts, fractions, no-break space, ...) before canonical ordering,
      // then canonical compose for NFKC.
      FNormUnicodeForm := Kind
    else
      raise EHFTokenizerError.Create('Unsupported normalizer type: ' + Kind);
  end;

  // Returns the Split "pattern.Regex" string of a pre_tokenizer child, or ''
  // when the child is not a Split or carries no Regex pattern.
  function ChildRegex(Child: TJSONObject): string;
  var
    PatNode: TJSONData;
  begin
    Result := '';
    if Child.Get('type', '') <> 'Split' then Exit;
    PatNode := Child.Find('pattern');
    if (PatNode <> nil) and (PatNode is TJSONObject) then
      Result := TJSONObject(PatNode).Get('Regex', '');
  end;

  // Detects the DeepSeek-V2 / V2-Lite / V3 pre-tokenizer Sequence as a WHOLE.
  // These ship several Split stages with huge explicit Unicode letter/punct/
  // CJK ranges plus a Digits(individual) stage -- behaviorally a \s?-prefixed
  // letter/number/punct/CJK/newline splitter with single-digit isolation,
  // which SplitDeepSeekPieces reproduces. Matched by distinctive substrings of
  // the shipped patterns (the literals span thousands of codepoints, so a
  // verbatim multi-KB string constant is intentionally avoided).
  function MatchesDeepSeekSequence(Arr: TJSONArray): boolean;
  var
    I: integer;
    HasNL, HasLetters, HasDigits, HasByteLevel: boolean;
    R, Kind2: string;
  begin
    Result := false;
    HasNL := false; HasLetters := false; HasDigits := false;
    HasByteLevel := false;
    for I := 0 to Arr.Count - 1 do
    begin
      Kind2 := Arr.Objects[I].Get('type', '');
      if (Kind2 = 'Digits') and Arr.Objects[I].Get('individual_digits', false)
        then HasDigits := true;
      if Kind2 = 'ByteLevel' then HasByteLevel := true;
      R := ChildRegex(Arr.Objects[I]);
      // The JSON loader unescapes "[\r\n]" to "[" + CR + LF + "]"; accept
      // both the unescaped and the verbatim-backslash forms.
      if (R = '[' + #13 + #10 + ']') or (R = '[\r\n]') then HasNL := true;
      // the V2/V2-Lite/coder letter class starts with "\s?[A-Za-z" / "\s?\p{L}";
      // V3 packs digits first then a CJK class then a punct+letter class.
      if (Pos('\s?[A-Za-z', R) = 1) or (Pos('\s?\p{L}', R) = 1) then
        HasLetters := true;
    end;
    // V2/V2-Lite/coder shape: [\r\n] + \s?letters + ... + Digits + ByteLevel.
    Result := HasNL and HasLetters and HasDigits and HasByteLevel;
  end;

  procedure AddPreTokenizer(PreObj: TJSONObject);
  var
    Kind, Pattern: string;
    InnerArr: TJSONArray;
    InnerCnt: integer;
    SubNode: TJSONData;
  begin
    Kind := PreObj.Get('type', '');
    if Kind = 'Sequence' then
    begin
      InnerArr := PreObj.Arrays['pretokenizers'];
      // The DeepSeek-V2/V2-Lite/V3 pre-tokenizer is a Sequence of several
      // Split stages + a Digits(individual) stage + ByteLevel(use_regex=
      // false). It is NOT a single cl100k-style pattern, so it is detected
      // as a WHOLE (by its distinctive Split-pattern set) and dispatched to
      // the dedicated SplitDeepSeekPieces splitter rather than recursed
      // into child-by-child.
      if MatchesDeepSeekSequence(InnerArr) then
      begin
        FDeepSeekPreTok := true;
        FByteLevel := true;
        FByteLevelUseRegex := false;
        FAddPrefixSpace := false;
        Exit;
      end;
      for InnerCnt := 0 to InnerArr.Count - 1 do
        AddPreTokenizer(InnerArr.Objects[InnerCnt]);
    end
    else if Kind = 'ByteLevel' then
    begin
      FByteLevel := true;
      FAddPrefixSpace := PreObj.Get('add_prefix_space', false);
      // use_regex (default true): a standalone ByteLevel with use_regex=false
      // skips the GPT-2 regex split (one chunk -> byte alphabet). When a
      // Split pre-tokenizer already ran upstream, the ByteLevel's own
      // use_regex is moot (FSplitPreTok/FDeepSeekPreTok take the split path).
      FByteLevelUseRegex := PreObj.Get('use_regex', true);
    end
    else if Kind = 'BertPreTokenizer' then
      // whitespace split + isolated punctuation; keyed off FWordPiece in
      // EncodeSegment (BertPieces), nothing else to configure
    else if Kind = 'Metaspace' then
    begin
      // Llama-2/Mistral SentencePiece-BPE: space -> U+2581 replacement,
      // optional prepend, optional split before each replacement.
      FMetaspacePreTok := true;
      FMSReplacement := PreObj.Get('replacement', csMetaspace);
      SubNode := PreObj.Find('prepend_scheme');
      if (SubNode <> nil) and not SubNode.IsNull then
        FMSPrependScheme := SubNode.AsString
      // legacy serialization (tokenizers < 0.14): add_prefix_space
      else if PreObj.Get('add_prefix_space', true) then
        FMSPrependScheme := 'always'
      else
        FMSPrependScheme := 'never';
      FMSSplit := PreObj.Get('split', true);
    end
    else if Kind = 'Split' then
    begin
      // No regex engine is vendored: the shipped pattern is matched
      // verbatim against the known cl100k-family literals and dispatched
      // to the hand-written splitter (SplitCl100kPieces).
      Pattern := '';
      SubNode := PreObj.Find('pattern');
      if (SubNode <> nil) and (SubNode is TJSONObject) then
      begin
        Pattern := TJSONObject(SubNode).Get('Regex', '');
        if Pattern = '' then
          Pattern := TJSONObject(SubNode).Get('String', '');
      end;
      if (PreObj.Get('behavior', '') <> 'Isolated') or
        PreObj.Get('invert', false) then
        raise EHFTokenizerError.Create(
          'Unsupported Split pre_tokenizer behavior "' +
          PreObj.Get('behavior', '') + '" (only Isolated, invert=false)');
      if Pattern = csQwen2SplitPattern then
      begin
        FSplitDigitsMax := 1;
        FSplitPreTok := true;
      end
      else if Pattern = csCl100kSplitPattern then
      begin
        FSplitDigitsMax := 3;
        FSplitPreTok := true;
      end
      else if Pattern = csO200kSplitPattern then
        // case-aware multi-alternation splitter (distinct from cl100k)
        FO200kPreTok := true
      else
        raise EHFTokenizerError.Create(
          'Unsupported Split pre_tokenizer pattern (only the Qwen2, ' +
          'Llama-3/cl100k and o200k patterns are recognized): ' + Pattern);
    end
    else
      raise EHFTokenizerError.Create(
        'Unsupported pre_tokenizer type: ' + Kind);
  end;

  procedure AddDecoder(DecObj: TJSONObject);
  var
    Kind: string;
    InnerArr: TJSONArray;
    InnerCnt: integer;
  begin
    Kind := DecObj.Get('type', '');
    if Kind = 'Sequence' then
    begin
      InnerArr := DecObj.Arrays['decoders'];
      for InnerCnt := 0 to InnerArr.Count - 1 do
        AddDecoder(InnerArr.Objects[InnerCnt]);
    end
    else if Kind = 'Replace' then
    begin
      SetLength(FDecReplaceFrom, Length(FDecReplaceFrom) + 1);
      SetLength(FDecReplaceTo, Length(FDecReplaceTo) + 1);
      FDecReplaceFrom[High(FDecReplaceFrom)] :=
        DecObj.Objects['pattern'].Get('String', '');
      FDecReplaceTo[High(FDecReplaceTo)] := DecObj.Get('content', '');
    end
    else if Kind = 'Strip' then
      FDecStripLeft := DecObj.Get('start', 0)
    else if Kind = 'Metaspace' then
    begin
      // replacement -> ' ' on every token, then strip the prepended
      // leading space (unless prepend_scheme/add_prefix_space says never).
      SetLength(FDecReplaceFrom, Length(FDecReplaceFrom) + 1);
      SetLength(FDecReplaceTo, Length(FDecReplaceTo) + 1);
      FDecReplaceFrom[High(FDecReplaceFrom)] :=
        DecObj.Get('replacement', csMetaspace);
      FDecReplaceTo[High(FDecReplaceTo)] := ' ';
      Node := DecObj.Find('prepend_scheme');
      if (Node <> nil) and not Node.IsNull then
      begin
        if Node.AsString <> 'never' then FDecStripLeft := 1;
      end
      else if DecObj.Get('add_prefix_space', true) then
        FDecStripLeft := 1;
    end
    else if Kind = 'WordPiece' then
    begin
      FWPPrefix := DecObj.Get('prefix', '##');
      FDecWordPieceCleanup := DecObj.Get('cleanup', true);
    end
    else if (Kind = 'ByteLevel') or (Kind = 'ByteFallback') or
      (Kind = 'Fuse') then
      // ByteLevel/ByteFallback decoding is keyed off the model flags;
      // Fuse only concatenates, which Decode does anyway.
    else
      raise EHFTokenizerError.Create('Unsupported decoder type: ' + Kind);
  end;

var
  RawJson: string;
begin
  // Raw SentencePiece .model protobuf dispatch: by extension, or by sniffing
  // the first byte (every tokenizer.json starts with '{'; a ModelProto
  // starts with the protobuf tag 0x0A for field 1 = pieces).
  if LowerCase(ExtractFileExt(FileName)) = '.model' then
  begin
    LoadSentencePieceModel(FileName);
    Exit;
  end;
  ClearState();
  DetectKeyMangling();
  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    SetLength(RawJson, FS.Size);
    if FS.Size > 0 then FS.ReadBuffer(RawJson[1], FS.Size);
  finally
    FS.Free;
  end;
  // Sniff: a non-'{' first non-whitespace byte means this is a binary
  // ModelProto saved under a non-.model name (e.g. some tokenizer.model).
  if (Length(RawJson) > 0) and (RawJson[1] <> '{') and (RawJson[1] <> #$EF) then
  begin
    LoadSentencePieceModel(FileName);
    Exit;
  end;
  // decode \uXXXX up front (see DecodeUnicodeEscapes) and parse raw
  Root := ParseJSONRaw(DecodeUnicodeEscapes(RawJson));
  try
    if not (Root is TJSONObject) then
      raise EHFTokenizerError.Create('tokenizer.json: root is not an object');
    RootObj := TJSONObject(Root);

    // ---- model -------------------------------------------------
    ModelObj := RootObj.Objects['model'];
    FWordPiece := ModelObj.Get('type', '') = 'WordPiece';
    FUnigram := ModelObj.Get('type', '') = 'Unigram';
    // pre-2021 tokenizer.json files (e.g. openai-community/gpt2) omit
    // model.type entirely; the merges key marks them as BPE
    if (not FWordPiece) and (not FUnigram) and
      (ModelObj.Get('type', '') <> 'BPE') and
      not ((ModelObj.Get('type', '') = '') and
        (ModelObj.IndexOfName('merges') >= 0)) then
      raise EHFTokenizerError.Create('Unsupported model.type "' +
        ModelObj.Get('type', '') +
        '" (only BPE, WordPiece and Unigram are supported)');
    FByteFallback := ModelObj.Get('byte_fallback', false);
    FFuseUnk := ModelObj.Get('fuse_unk', false);
    FIgnoreMerges := ModelObj.Get('ignore_merges', false);
    if FWordPiece then
    begin
      FWPPrefix := ModelObj.Get('continuing_subword_prefix', '##');
      FWPMaxChars := ModelObj.Get('max_input_chars_per_word', 100);
    end;

    // vocab. BPE/WordPiece: an object token -> id. Unigram: an ARRAY of
    // [piece, log_prob] pairs whose 0-based index IS the id.
    if FUnigram then
    begin
      VocabArr := ModelObj.Arrays['vocab'];
      SetLength(FIdToToken, VocabArr.Count);
      SetLength(FUniScore, VocabArr.Count);
      FUniMinScore := 0;
      for Cnt := 0 to VocabArr.Count - 1 do
      begin
        SubArr := TJSONArray(VocabArr.Items[Cnt]);
        Content := FixJSONKey(SubArr.Strings[0]);
        Score := SubArr.Floats[1];
        FVocab.AddObject(Content, TObject(PtrInt(Cnt)));
        FIdToToken[Cnt] := Content;
        FUniScore[Cnt] := Score;
        if (Cnt = 0) or (Score < FUniMinScore) then FUniMinScore := Score;
      end;
      // unk_id is an INDEX into the vocab array (not a token string).
      Node := ModelObj.Find('unk_id');
      if (Node <> nil) and not Node.IsNull then FUnkId := Node.AsInteger;
    end
    else
    begin
      VocabObj := ModelObj.Objects['vocab'];
      MaxId := -1;
      for Cnt := 0 to VocabObj.Count - 1 do
        MaxId := Max(MaxId, VocabObj.Items[Cnt].AsInteger);
      SetLength(FIdToToken, MaxId + 1);
      for Cnt := 0 to VocabObj.Count - 1 do
      begin
        TokenId := VocabObj.Items[Cnt].AsInteger;
        Content := FixJSONKey(VocabObj.Names[Cnt]);
        FVocab.AddObject(Content, TObject(PtrInt(TokenId)));
        FIdToToken[TokenId] := Content;
      end;
    end;

    // merges: ["a b", ...] (old) or [["a","b"], ...] (new).
    // WordPiece (greedy longest-match) and Unigram (Viterbi) have no merges.
    if FWordPiece or FUnigram then
      MergesArr := nil
    else
      MergesArr := ModelObj.Arrays['merges'];
    if MergesArr <> nil then
    for Cnt := 0 to MergesArr.Count - 1 do
    begin
      Node := MergesArr.Items[Cnt];
      if Node is TJSONArray then
      begin
        SubArr := TJSONArray(Node);
        Left := SubArr.Strings[0];
        Right := SubArr.Strings[1];
      end
      else
      begin
        MergeStr := Node.AsString;
        SpacePos := Pos(' ', MergeStr);
        Left := Copy(MergeStr, 1, SpacePos - 1);
        Right := Copy(MergeStr, SpacePos + 1, Length(MergeStr));
      end;
      FMerges.AddObject(Left + csMergeSep + Right, TObject(PtrInt(Cnt)));
    end;

    // unk token
    Node := ModelObj.Find('unk_token');
    if (Node <> nil) and not Node.IsNull then
      FUnkId := VocabFind(Node.AsString);

    // ---- added tokens -------------------------------------------
    Node := RootObj.Find('added_tokens');
    if (Node <> nil) and (Node is TJSONArray) then
    begin
      AddedArr := TJSONArray(Node);
      SetLength(FAddedTokens, AddedArr.Count);
      for Cnt := 0 to AddedArr.Count - 1 do
      begin
        Entry := AddedArr.Objects[Cnt];
        FAddedTokens[Cnt].Content := Entry.Get('content', '');
        FAddedTokens[Cnt].Id := Entry.Get('id', -1);
        FAddedTokens[Cnt].Special := Entry.Get('special', false);
        Content := FAddedTokens[Cnt].Content;
        if FAddedTokens[Cnt].Special then
        begin
          if (Content = '<s>') or (Content = '<|startoftext|>') then
            FBosId := FAddedTokens[Cnt].Id;
          if (Content = '</s>') or (Content = '<|endoftext|>') or
            (Content = '<|im_end|>') then
            FEosId := FAddedTokens[Cnt].Id;
          if (Content = '<unk>') and (FUnkId < 0) then
            FUnkId := FAddedTokens[Cnt].Id;
        end;
      end;
      // GPT-2 family: <|endoftext|> doubles as BOS.
      if (FBosId < 0) and (FEosId >= 0) then FBosId := FEosId;
    end;

    // ---- normalizer / pre_tokenizer / decoder -------------------
    Node := RootObj.Find('normalizer');
    if (Node <> nil) and (Node is TJSONObject) then
      AddNormalizer(TJSONObject(Node));
    Node := RootObj.Find('pre_tokenizer');
    if (Node <> nil) and (Node is TJSONObject) then
      AddPreTokenizer(TJSONObject(Node));
    Node := RootObj.Find('decoder');
    if (Node <> nil) and (Node is TJSONObject) then
      AddDecoder(TJSONObject(Node));
  finally
    Root.Free;
  end;
end;

// Reconstructs the byte-level-BPE merge table (FMerges) from a BPE
// SentencePiece ModelProto's scored pieces. A BPE .model carries NO explicit
// merge list, so we derive it byte-for-byte the way HuggingFace's
// transformers.tokenization_utils_base.generate_merges does for the BPE
// branch of SpmConverter (called with vocab only -> ascending rank order):
//
//   merges = []
//   for piece (in id order):
//     local = [(l, r, piece_id) for each codepoint split where both halves
//              are in the vocab]              # piece_id stands in for "score"
//     local.sort(key=(id_l, id_r))
//     merges += local
//   merges.sort(key=(piece_id, len_cp(l), len_cp(r)))   # STABLE, ascending
//   rank = position in the final list
//
// Both Python len() and slicing are over CODEPOINTS, so the split scan and the
// length keys here iterate codepoints (NextCodePoint), not raw bytes -- this
// matters for the multi-byte learned pieces. Verified id-identical to the
// sentencepiece encoder (TestSentencePieceBPEModelParity) including the
// non-ASCII byte-fallback path; no approximation.
procedure TNeuralHFTokenizer.ReconstructBPEMerges(
  const PieceTextV: array of string; APieceCount: integer);
type
  TMergeCand = record
    Left, Right: string;
    LeftId, RightId, PieceId, LenL, LenR, Seq: integer;
  end;
var
  Cands: array of TMergeCand;
  Boundaries: array of integer; // 1-based byte offsets of codepoint starts
  NumCP, Pos, CnPiece, SplitI, ByteSplit, LId, RId, NLocal, I, J, Seq: integer;
  L, R, Piece: string;
  Tmp: TMergeCand;
  LocalStart: integer;
begin
  Seq := 0;
  SetLength(Cands, 0);
  for CnPiece := 0 to APieceCount - 1 do
  begin
    Piece := PieceTextV[CnPiece];
    if Piece = '' then continue;
    // Codepoint boundary table (1-based byte offsets, plus end sentinel).
    SetLength(Boundaries, Length(Piece) + 1);
    NumCP := 0;
    Pos := 1;
    while Pos <= Length(Piece) do
    begin
      Boundaries[NumCP] := Pos;
      NextCodePoint(Piece, Pos);
      Inc(NumCP);
    end;
    Boundaries[NumCP] := Pos;
    if NumCP < 2 then continue;  // single codepoint -> no split
    LocalStart := Length(Cands);
    // Each codepoint split (1..NumCP-1) where both halves are in the vocab.
    for SplitI := 1 to NumCP - 1 do
    begin
      ByteSplit := Boundaries[SplitI];
      L := Copy(Piece, 1, ByteSplit - 1);
      R := Copy(Piece, ByteSplit, Length(Piece) - ByteSplit + 1);
      LId := VocabFind(L);
      if LId < 0 then continue;
      RId := VocabFind(R);
      if RId < 0 then continue;
      SetLength(Cands, Length(Cands) + 1);
      with Cands[High(Cands)] do
      begin
        Left := L; Right := R;
        LeftId := LId; RightId := RId;
        PieceId := CnPiece;
        LenL := SplitI;            // codepoint length of L
        LenR := NumCP - SplitI;    // codepoint length of R
        Seq := 0;                  // filled after local sort
      end;
    end;
    // local sort by (LeftId, RightId) -- insertion sort over this piece's run.
    NLocal := Length(Cands) - LocalStart;
    for I := LocalStart + 1 to High(Cands) do
    begin
      Tmp := Cands[I];
      J := I - 1;
      while (J >= LocalStart) and
        ((Cands[J].LeftId > Tmp.LeftId) or
         ((Cands[J].LeftId = Tmp.LeftId) and (Cands[J].RightId > Tmp.RightId)))
      do
      begin
        Cands[J + 1] := Cands[J];
        Dec(J);
      end;
      Cands[J + 1] := Tmp;
    end;
    if NLocal = 0 then ;  // silence unused-warning path
  end;

  // Stamp the append/sequence order (after all local sorts) so the final
  // sort can keep ties stable (Python's sort is stable).
  for I := 0 to High(Cands) do Cands[I].Seq := I;

  // Final stable sort by (PieceId, LenL, LenR, Seq) ascending -> merge rank.
  for I := 1 to High(Cands) do
  begin
    Tmp := Cands[I];
    J := I - 1;
    while (J >= 0) and
      ((Cands[J].PieceId > Tmp.PieceId) or
       ((Cands[J].PieceId = Tmp.PieceId) and (Cands[J].LenL > Tmp.LenL)) or
       ((Cands[J].PieceId = Tmp.PieceId) and (Cands[J].LenL = Tmp.LenL) and
        (Cands[J].LenR > Tmp.LenR)) or
       ((Cands[J].PieceId = Tmp.PieceId) and (Cands[J].LenL = Tmp.LenL) and
        (Cands[J].LenR = Tmp.LenR) and (Cands[J].Seq > Tmp.Seq))) do
    begin
      Cands[J + 1] := Cands[J];
      Dec(J);
    end;
    Cands[J + 1] := Tmp;
  end;

  // Position in the final list IS the rank (lower rank = applied first).
  for I := 0 to High(Cands) do
    FMerges.AddObject(Cands[I].Left + csMergeSep + Cands[I].Right,
      TObject(PtrInt(I)));
end;

{ ---------------------------------------------------------------- }
{ Raw SentencePiece ModelProto protobuf reader                      }
{ ---------------------------------------------------------------- }
//
// Hand-decoded protobuf wire format (no vendored proto library). The
// SentencePiece ModelProto (sentencepiece_model.proto) is:
//   field 1  pieces        repeated message SentencePiece
//                            f1 piece (string), f2 score (float/fixed32),
//                            f3 type (enum 1=NORMAL 2=UNKNOWN 3=CONTROL
//                            4=USER_DEFINED 5=UNUSED 6=BYTE)
//   field 2  trainer_spec  message
//                            f3  model_type (1=UNIGRAM 2=BPE 3=WORD 4=CHAR)
//                            f40 unk_id, f41 bos_id, f42 eos_id, f43 pad_id
//   field 3  normalizer_spec (precompiled_charsmap etc. -- ignored here)
// Wire tag = (field_number shl 3) or wire_type; wire_type 0=varint,
// 1=fixed64, 2=length-delimited, 5=fixed32. Negative ids are stored as
// their two's-complement uint64 (e.g. pad_id -1 == high(uint64)).
procedure TNeuralHFTokenizer.LoadSentencePieceModel(const FileName: string);
var
  FS: TFileStream;
  Buf: array of byte;
  BufLen: integer;
  PieceTextV: array of string;  // piece text, id-indexed
  PieceTypeV: array of integer; // piece type, id-indexed
  PieceCount: integer;
  ModelType: integer;           // trainer_spec.model_type (default UNIGRAM)
  SpmUnk, SpmBos, SpmEos: integer;
  Cnt: integer;
  Score: single;

  // Reads a base-128 varint at Pos (0-based into Buf), advances Pos.
  function ReadVarint(var Pos: integer): QWord;
  var
    Shift: integer;
    B: byte;
  begin
    Result := 0;
    Shift := 0;
    repeat
      if Pos >= BufLen then
        raise EHFTokenizerError.Create(
          'SentencePiece .model: truncated varint');
      B := Buf[Pos];
      Inc(Pos);
      Result := Result or (QWord(B and $7F) shl Shift);
      Inc(Shift, 7);
    until (B and $80) = 0;
  end;

  // Sign-recovers a uint64 wire id to a signed integer (HF/SPM convention:
  // an "unset" id of -1 is serialized as 2^64-1).
  function ToSignedId(V: QWord): integer;
  begin
    if V > QWord($7FFFFFFFFFFFFFFF) then
      Result := integer(Int64(V))   // negative (e.g. -1)
    else
      Result := integer(V);
  end;

  // Parses one SentencePiece submessage [Start, Stop) into Text/Typ/Score.
  procedure ParsePiece(Start, Stop: integer; out Text: string;
    out Typ: integer; out PScore: single);
  var
    P, FieldNum, WireType, Len: integer;
    V: QWord;
    U: cardinal;
  begin
    Text := '';
    Typ := 1; // NORMAL
    PScore := 0;
    P := Start;
    while P < Stop do
    begin
      V := ReadVarint(P);
      FieldNum := integer(V shr 3);
      WireType := integer(V and 7);
      case WireType of
        0: // varint  -> f3 type
          begin
            V := ReadVarint(P);
            if FieldNum = 3 then Typ := integer(V);
          end;
        1: Inc(P, 8); // fixed64 (none used in a piece)
        2: // length-delimited -> f1 piece string
          begin
            Len := integer(ReadVarint(P));
            if FieldNum = 1 then
            begin
              SetLength(Text, Len);
              if Len > 0 then Move(Buf[P], Text[1], Len);
            end;
            Inc(P, Len);
          end;
        5: // fixed32 -> f2 score (little-endian IEEE-754 single)
          begin
            if P + 4 > Stop then
              raise EHFTokenizerError.Create(
                'SentencePiece .model: truncated fixed32');
            U := cardinal(Buf[P]) or (cardinal(Buf[P + 1]) shl 8) or
              (cardinal(Buf[P + 2]) shl 16) or (cardinal(Buf[P + 3]) shl 24);
            if FieldNum = 2 then PScore := single((@U)^);
            Inc(P, 4);
          end;
      else
        raise EHFTokenizerError.Create(
          'SentencePiece .model: bad wire type in piece');
      end;
    end;
  end;

  // Parses the trainer_spec submessage [Start, Stop): model_type + ids.
  procedure ParseTrainerSpec(Start, Stop: integer);
  var
    P, FieldNum, WireType, Len: integer;
    V: QWord;
  begin
    P := Start;
    while P < Stop do
    begin
      V := ReadVarint(P);
      FieldNum := integer(V shr 3);
      WireType := integer(V and 7);
      case WireType of
        0:
          begin
            V := ReadVarint(P);
            case FieldNum of
              3:  ModelType := integer(V);
              40: SpmUnk := ToSignedId(V);
              41: SpmBos := ToSignedId(V);
              42: SpmEos := ToSignedId(V);
              // 43 = pad_id (unused here)
            end;
          end;
        1: Inc(P, 8);
        2: begin Len := integer(ReadVarint(P)); Inc(P, Len); end;
        5: Inc(P, 4);
      else
        raise EHFTokenizerError.Create(
          'SentencePiece .model: bad wire type in trainer_spec');
      end;
    end;
  end;

  // Top-level ModelProto walk: collect pieces (field 1) and trainer_spec
  // (field 2). normalizer_spec (field 3) and everything else is skipped.
  procedure ParseModelProto();
  var
    P, FieldNum, WireType, Len: integer;
    V: QWord;
    PText: string;
    PTyp: integer;
  begin
    P := 0;
    while P < BufLen do
    begin
      V := ReadVarint(P);
      FieldNum := integer(V shr 3);
      WireType := integer(V and 7);
      case WireType of
        0: ReadVarint(P);
        1: Inc(P, 8);
        2:
          begin
            Len := integer(ReadVarint(P));
            if FieldNum = 1 then // a SentencePiece
            begin
              ParsePiece(P, P + Len, PText, PTyp, Score);
              if PieceCount >= Length(PieceTextV) then
              begin
                SetLength(PieceTextV, Length(PieceTextV) * 2 + 16);
                SetLength(PieceTypeV, Length(PieceTextV));
                SetLength(FUniScore, Length(PieceTextV));
              end;
              PieceTextV[PieceCount] := PText;
              PieceTypeV[PieceCount] := PTyp;
              FUniScore[PieceCount] := Score;
              Inc(PieceCount);
            end
            else if FieldNum = 2 then // trainer_spec
              ParseTrainerSpec(P, P + Len);
            Inc(P, Len);
          end;
        5: Inc(P, 4);
      else
        raise EHFTokenizerError.Create(
          'SentencePiece .model: bad wire type at top level');
      end;
    end;
  end;

begin
  ClearState();
  // NOTE: DetectKeyMangling() is intentionally NOT called -- the .model
  // reader handles raw bytes itself (no fpjson involved), so no key
  // mangling can occur and the probe (which parses JSON) would be wasted.
  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    BufLen := FS.Size;
    SetLength(Buf, BufLen);
    if BufLen > 0 then FS.ReadBuffer(Buf[0], BufLen);
  finally
    FS.Free;
  end;

  PieceCount := 0;
  ModelType := 1;   // UNIGRAM default (matches SPM proto default)
  SpmUnk := -1;
  SpmBos := -1;
  SpmEos := -1;
  ParseModelProto();

  if (ModelType <> 1) and (ModelType <> 2) then
    raise EHFTokenizerError.Create(
      'SentencePiece .model: only the UNIGRAM and BPE model_types are ' +
      'supported by the raw-.model reader (model_type=' + IntToStr(ModelType) +
      '; WORD/CHAR not supported -- use tokenizer.json instead).');
  if PieceCount = 0 then
    raise EHFTokenizerError.Create('SentencePiece .model: no pieces found.');

  // Populate the SAME internal structures the tokenizer.json path uses:
  // FVocab (token->id), FIdToToken (id->token); for UNIGRAM also FUniScore /
  // FUniMinScore; for BPE the reconstructed FMerges (see below). The 0-based
  // piece index IS the id.
  FUnigram := ModelType = 1;
  SetLength(FIdToToken, PieceCount);
  SetLength(FUniScore, PieceCount);
  FUniMinScore := 0;
  FByteFallback := false;
  for Cnt := 0 to PieceCount - 1 do
  begin
    FVocab.AddObject(PieceTextV[Cnt], TObject(PtrInt(Cnt)));
    FIdToToken[Cnt] := PieceTextV[Cnt];
    if (Cnt = 0) or (FUniScore[Cnt] < FUniMinScore) then
      FUniMinScore := FUniScore[Cnt];
    // SentencePiece BYTE pieces (type 6) are the <0x00>..<0xFF> byte-
    // fallback alphabet. Their presence means byte_fallback=true: unknown
    // bytes route through <0xNN> on encode and those pieces decode back to
    // their raw byte (EmitTokenOrFallback / DecodeToken), so a run of byte
    // pieces reassembles the original multi-byte UTF-8 sequence.
    if PieceTypeV[Cnt] = 6 then FByteFallback := true;
  end;

  // model_type=BPE: the ModelProto stores ONLY scored pieces -- no explicit
  // merge list (unlike a tokenizer.json BPE model). Reconstruct the merges
  // from the pieces, byte-for-byte matching HuggingFace's
  // transformers.tokenization_utils_base.generate_merges (the algorithm
  // SpmConverter uses for model_type==BPE):
  //   * vocab = {piece -> id}; iterate pieces in id order.
  //   * for each piece, for every split point (l, r) with BOTH halves in the
  //     vocab, collect a candidate merge; within one piece the candidates are
  //     ordered by (id_l, id_r).
  //   * concatenate all candidates (pieces in id order), then STABLE-sort the
  //     whole list by (id of the merged piece, len(l), len(r)) ascending --
  //     the resulting order IS the merge rank.
  // The metaspace + reconstructed-merge + byte-fallback path is then routed
  // through the SAME BPEWord / EmitTokenOrFallback machinery the GGUF "gpt2"
  // and tokenizer.json BPE paths use. Verified id-identical to the
  // sentencepiece encoder over ASCII *and* non-ASCII (byte-fallback) inputs
  // by TestSentencePieceBPEModelParity -- no approximation, full parity.
  if ModelType = 2 then
    ReconstructBPEMerges(PieceTextV, PieceCount);

  // Special-token ids straight from trainer_spec (authoritative for SPM).
  if SpmUnk >= 0 then FUnkId := SpmUnk;
  if SpmBos >= 0 then FBosId := SpmBos;
  if SpmEos >= 0 then FEosId := SpmEos;

  // SentencePiece always uses the Metaspace convention: add_dummy_prefix
  // prepends U+2581 and every space becomes U+2581. Wire the SAME
  // pre-tokenizer + decoder the tokenizer.json Unigram path gets from its
  // Metaspace pre_tokenizer / decoder entries (prepend always, split on the
  // metaspace, replacement U+2581 -> ' ' on decode, strip one leading space).
  FMetaspacePreTok := true;
  FMSReplacement := csMetaspace;
  FMSPrependScheme := 'always';
  FMSSplit := true;
  SetLength(FDecReplaceFrom, 1);
  SetLength(FDecReplaceTo, 1);
  FDecReplaceFrom[0] := csMetaspace;
  FDecReplaceTo[0] := ' ';
  FDecStripLeft := 1;

  // Expose the special pieces as added tokens so they round-trip verbatim
  // in the input text and Decode(skip_special_tokens) drops them, matching
  // the tokenizer.json path's added_tokens handling.
  SetLength(FAddedTokens, 0);
  for Cnt := 0 to PieceCount - 1 do
    if (PieceTypeV[Cnt] = 2) or (PieceTypeV[Cnt] = 3) then // UNKNOWN/CONTROL
    begin
      SetLength(FAddedTokens, Length(FAddedTokens) + 1);
      FAddedTokens[High(FAddedTokens)].Content := PieceTextV[Cnt];
      FAddedTokens[High(FAddedTokens)].Id := Cnt;
      FAddedTokens[High(FAddedTokens)].Special := true;
    end;
end;

procedure TNeuralHFTokenizer.LoadFromGGUF(const FileName: string);
// GGUF tokenizer.ggml.token_type enum (llama.cpp llama_token_attr / the
// legacy llama_token_type values shipped in tokenizer.ggml.token_type):
//   1 = NORMAL, 2 = UNKNOWN, 3 = CONTROL, 4 = USER_DEFINED, 5 = UNUSED,
//   6 = BYTE. CONTROL / UNKNOWN / USER_DEFINED pieces are exposed as added
//   tokens so they round-trip verbatim and Decode(skip_special) drops them.
var
  Reader: TNNetGGUFReader;
  Model: string;
  TokCount, Cnt: integer;
  ScoreCount, TypeCount, MergeCount: Int64;
  HasScores, HasTypes: boolean;
  TokType: integer;
  Content, MergeStr, Left, Right: string;
  SpacePos: integer;
  MetaBos, MetaEos, MetaUnk: Int64;
begin
  ClearState();
  // No JSON is involved (the GGUF reader handles raw bytes), so the fpjson
  // key-mangling probe would be wasted, like the .model path.
  Reader := TNNetGGUFReader.Create(FileName);
  try
    Model := Reader.GetMetaString('tokenizer.ggml.model', '');
    if Model = '' then
      raise EHFTokenizerError.Create('GGUF: no tokenizer.ggml.model metadata ' +
        'in ' + FileName + ' (file carries no embedded tokenizer).');

    TokCount := integer(Reader.GetMetaArrayCount('tokenizer.ggml.tokens'));
    if TokCount = 0 then
      raise EHFTokenizerError.Create('GGUF: tokenizer.ggml.tokens is empty ' +
        'in ' + FileName + '.');

    ScoreCount := Reader.GetMetaArrayCount('tokenizer.ggml.scores');
    TypeCount := Reader.GetMetaArrayCount('tokenizer.ggml.token_type');
    HasScores := ScoreCount = TokCount;
    HasTypes := TypeCount = TokCount;

    // ---- vocab (the 0-based array index IS the id) ----
    SetLength(FIdToToken, TokCount);
    for Cnt := 0 to TokCount - 1 do
    begin
      Content := Reader.GetMetaArrayString('tokenizer.ggml.tokens', Cnt);
      FVocab.AddObject(Content, TObject(PtrInt(Cnt)));
      FIdToToken[Cnt] := Content;
    end;

    // ---- special-token ids (authoritative scalar metadata) ----
    MetaBos := Reader.GetMetaInt('tokenizer.ggml.bos_token_id', -1);
    MetaEos := Reader.GetMetaInt('tokenizer.ggml.eos_token_id', -1);
    MetaUnk := Reader.GetMetaInt('tokenizer.ggml.unknown_token_id', -1);
    if MetaBos >= 0 then FBosId := integer(MetaBos);
    if MetaEos >= 0 then FEosId := integer(MetaEos);
    if MetaUnk >= 0 then FUnkId := integer(MetaUnk);

    if Model = 'llama' then
    begin
      // SentencePiece-with-scores -> the Unigram (Viterbi) path, exactly like
      // the raw .model reader: metaspace U+2581, <0xNN> byte fallback.
      FUnigram := true;
      FByteFallback := true;
      SetLength(FUniScore, TokCount);
      FUniMinScore := 0;
      for Cnt := 0 to TokCount - 1 do
      begin
        if HasScores
        then FUniScore[Cnt] :=
          Reader.GetMetaArrayNumber('tokenizer.ggml.scores', Cnt)
        else FUniScore[Cnt] := 0;
        if (Cnt = 0) or (FUniScore[Cnt] < FUniMinScore) then
          FUniMinScore := FUniScore[Cnt];
      end;

      // Metaspace pre-tokenizer + decoder (same wiring the .model path uses).
      FMetaspacePreTok := true;
      FMSReplacement := csMetaspace;
      FMSPrependScheme := 'always';
      FMSSplit := true;
      SetLength(FDecReplaceFrom, 1);
      SetLength(FDecReplaceTo, 1);
      FDecReplaceFrom[0] := csMetaspace;
      FDecReplaceTo[0] := ' ';
      FDecStripLeft := 1;

      // Expose CONTROL / UNKNOWN / USER_DEFINED pieces as added tokens.
      SetLength(FAddedTokens, 0);
      for Cnt := 0 to TokCount - 1 do
      begin
        if HasTypes
        then TokType := integer(Round(
          Reader.GetMetaArrayNumber('tokenizer.ggml.token_type', Cnt)))
        else TokType := 1;
        if (TokType = 2) or (TokType = 3) or (TokType = 4) then
        begin
          SetLength(FAddedTokens, Length(FAddedTokens) + 1);
          FAddedTokens[High(FAddedTokens)].Content := FIdToToken[Cnt];
          FAddedTokens[High(FAddedTokens)].Id := Cnt;
          FAddedTokens[High(FAddedTokens)].Special := true;
        end;
      end;
    end
    else if Model = 'gpt2' then
    begin
      // Byte-level BPE: GPT-2 bytes<->unicode table + the cl100k Split
      // pre-tokenizer + the merges array.
      FByteLevel := true;
      FSplitPreTok := true;
      FSplitDigitsMax := 3;
      MergeCount := Reader.GetMetaArrayCount('tokenizer.ggml.merges');
      for Cnt := 0 to MergeCount - 1 do
      begin
        MergeStr := Reader.GetMetaArrayString('tokenizer.ggml.merges', Cnt);
        SpacePos := Pos(' ', MergeStr);
        if SpacePos <= 0 then continue;
        Left := Copy(MergeStr, 1, SpacePos - 1);
        Right := Copy(MergeStr, SpacePos + 1, Length(MergeStr));
        FMerges.AddObject(Left + csMergeSep + Right, TObject(PtrInt(Cnt)));
      end;

      // Expose CONTROL / USER_DEFINED pieces as added special tokens so they
      // round-trip verbatim (e.g. <|endoftext|>); fall back to id-based
      // special wiring below.
      SetLength(FAddedTokens, 0);
      for Cnt := 0 to TokCount - 1 do
      begin
        if HasTypes
        then TokType := integer(Round(
          Reader.GetMetaArrayNumber('tokenizer.ggml.token_type', Cnt)))
        else TokType := 1;
        if (TokType = 3) or (TokType = 4) then
        begin
          SetLength(FAddedTokens, Length(FAddedTokens) + 1);
          FAddedTokens[High(FAddedTokens)].Content := FIdToToken[Cnt];
          FAddedTokens[High(FAddedTokens)].Id := Cnt;
          FAddedTokens[High(FAddedTokens)].Special := true;
        end;
      end;
      // GPT-2 family: <|endoftext|> doubles as BOS when no explicit bos id.
      if (FBosId < 0) and (FEosId >= 0) then FBosId := FEosId;
    end
    else
      raise EHFTokenizerError.Create('GGUF tokenizer.ggml.model "' + Model +
        '" is not supported (only "llama" SentencePiece-with-scores and ' +
        '"gpt2" byte-level BPE; "bert"/"no_vocab" are deferred).');
  finally
    Reader.Free;
  end;
end;

procedure TNeuralHFTokenizer.SaveTokenizerToGGUF(Writer: TNNetGGUFWriter);
// Inverse of LoadFromGGUF: serialize the in-memory vocab/merges/specials into
// a tokenizer.ggml.* metadata block. token_type follows the llama.cpp enum
// (1=NORMAL, 2=UNKNOWN, 3=CONTROL); added special pieces are tagged CONTROL
// (the unk piece UNKNOWN) so the reader re-exposes them as added tokens.
var
  TokCount, Cnt, SepPos, Rank, MaxRank, Mi: integer;
  Tokens, Merges: array of string;
  Scores: array of single;
  Types: array of Int64;
  Sep, Combined, Left, Right: string;
begin
  if Writer = nil then
    raise EHFTokenizerError.Create('SaveTokenizerToGGUF: nil writer.');
  if FWordPiece then
    raise EHFTokenizerError.Create('SaveTokenizerToGGUF: the WordPiece (BERT) ' +
      'path has no GGUF "bert" tokenizer.ggml.model yet (only byte-level BPE ' +
      '"gpt2" and Unigram-with-scores "llama").');

  TokCount := Length(FIdToToken);
  if TokCount = 0 then
    raise EHFTokenizerError.Create('SaveTokenizerToGGUF: empty vocabulary ' +
      '(load a tokenizer first).');

  // ---- id-indexed tokens / scores / token_type ----
  SetLength(Tokens, TokCount);
  SetLength(Scores, TokCount);
  SetLength(Types, TokCount);
  for Cnt := 0 to TokCount - 1 do
  begin
    Tokens[Cnt] := FIdToToken[Cnt];
    if FUnigram and (Cnt <= High(FUniScore))
    then Scores[Cnt] := FUniScore[Cnt]
    else Scores[Cnt] := 0;
    Types[Cnt] := 1; // NORMAL (overridden below for added/special pieces)
  end;
  // Tag added pieces: special -> CONTROL (3); the unk piece -> UNKNOWN (2).
  for Cnt := 0 to High(FAddedTokens) do
    if (FAddedTokens[Cnt].Id >= 0) and (FAddedTokens[Cnt].Id <= High(Types)) then
    begin
      if FAddedTokens[Cnt].Id = FUnkId then Types[FAddedTokens[Cnt].Id] := 2
      else if FAddedTokens[Cnt].Special then Types[FAddedTokens[Cnt].Id] := 3;
    end;
  if (FUnkId >= 0) and (FUnkId <= High(Types)) and (Types[FUnkId] = 1) then
    Types[FUnkId] := 2;

  if FByteLevel then Writer.AddMetaString('tokenizer.ggml.model', 'gpt2')
  else if FUnigram then Writer.AddMetaString('tokenizer.ggml.model', 'llama')
  else
    raise EHFTokenizerError.Create('SaveTokenizerToGGUF: this tokenizer is ' +
      'neither byte-level BPE nor Unigram; no GGUF model mapping.');

  Writer.AddMetaStringArray('tokenizer.ggml.tokens', Tokens);
  Writer.AddMetaFloat32Array('tokenizer.ggml.scores', Scores);
  Writer.AddMetaInt32Array('tokenizer.ggml.token_type', Types);

  // ---- merges (byte-level BPE only), restored to rank order ----
  if FByteLevel and (FMerges.Count > 0) then
  begin
    Sep := csMergeSep;
    MaxRank := 0;
    for Cnt := 0 to FMerges.Count - 1 do
    begin
      Rank := integer(PtrInt(FMerges.Objects[Cnt]));
      if Rank > MaxRank then MaxRank := Rank;
    end;
    // Ranks are dense 0..N-1 (LoadFromFile/LoadFromGGUF assign the array
    // index); slot each "Left Right" string back into its rank position.
    SetLength(Merges, MaxRank + 1);
    for Cnt := 0 to High(Merges) do Merges[Cnt] := '';
    for Mi := 0 to FMerges.Count - 1 do
    begin
      Combined := FMerges[Mi];
      Rank := integer(PtrInt(FMerges.Objects[Mi]));
      SepPos := Pos(Sep, Combined);
      if SepPos <= 0 then continue;
      Left := Copy(Combined, 1, SepPos - 1);
      Right := Copy(Combined, SepPos + Length(Sep), Length(Combined));
      if (Rank >= 0) and (Rank <= High(Merges)) then
        Merges[Rank] := Left + ' ' + Right;
    end;
    Writer.AddMetaStringArray('tokenizer.ggml.merges', Merges);
  end;

  // ---- scalar special-token ids ----
  if FBosId >= 0 then
    Writer.AddMetaUInt32('tokenizer.ggml.bos_token_id', cardinal(FBosId));
  if FEosId >= 0 then
    Writer.AddMetaUInt32('tokenizer.ggml.eos_token_id', cardinal(FEosId));
  if FUnkId >= 0 then
    Writer.AddMetaUInt32('tokenizer.ggml.unknown_token_id', cardinal(FUnkId));
end;

function TNeuralHFTokenizer.FindAddedToken(const Text: string;
  Position: integer; out TokenIndex: integer): boolean;
var
  Cnt, BestLen, ContentLen: integer;
begin
  Result := false;
  TokenIndex := -1;
  BestLen := 0;
  for Cnt := 0 to High(FAddedTokens) do
  begin
    ContentLen := Length(FAddedTokens[Cnt].Content);
    if (ContentLen > BestLen) and
      (Copy(Text, Position, ContentLen) = FAddedTokens[Cnt].Content) then
    begin
      BestLen := ContentLen;
      TokenIndex := Cnt;
      Result := true;
    end;
  end;
end;

function TNeuralHFTokenizer.IsAddedTokenId(Id: integer;
  out TokenIndex: integer): boolean;
var
  Cnt: integer;
begin
  Result := false;
  TokenIndex := -1;
  for Cnt := 0 to High(FAddedTokens) do
    if FAddedTokens[Cnt].Id = Id then
    begin
      TokenIndex := Cnt;
      Exit(true);
    end;
end;

// Splits a byte-level segment with the GPT-2 pre-tokenization regex:
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
procedure TNeuralHFTokenizer.ByteLevelPieces(const Segment: string;
  Pieces: TStringList);
var
  CPs: array of cardinal;
  CPStr: array of string;
  Total, Position, Idx, RunStart, RunEnd: integer;

  function Collect(StartIdx, EndIdx: integer): string;
  var
    J: integer;
  begin
    Result := '';
    for J := StartIdx to EndIdx do Result := Result + CPStr[J];
  end;

  function MatchContraction(StartIdx: integer): integer; // length or 0
  begin
    Result := 0;
    if CPs[StartIdx] <> Ord('''') then Exit;
    if StartIdx + 1 <= High(CPs) then
    begin
      if (CPs[StartIdx + 1] = Ord('s')) or (CPs[StartIdx + 1] = Ord('t')) or
        (CPs[StartIdx + 1] = Ord('m')) or (CPs[StartIdx + 1] = Ord('d')) then
        Exit(2);
      if StartIdx + 2 <= High(CPs) then
      begin
        if ((CPs[StartIdx + 1] = Ord('r')) and (CPs[StartIdx + 2] = Ord('e')))
          or ((CPs[StartIdx + 1] = Ord('v')) and (CPs[StartIdx + 2] = Ord('e')))
          or ((CPs[StartIdx + 1] = Ord('l')) and (CPs[StartIdx + 2] = Ord('l')))
        then Exit(3);
      end;
    end;
  end;

begin
  // decode UTF-8 into codepoint arrays
  SetLength(CPs, Length(Segment));
  SetLength(CPStr, Length(Segment));
  Total := 0;
  Position := 1;
  while Position <= Length(Segment) do
  begin
    RunStart := Position;
    CPs[Total] := NextCodePoint(Segment, Position);
    CPStr[Total] := Copy(Segment, RunStart, Position - RunStart);
    Inc(Total);
  end;
  SetLength(CPs, Total);
  SetLength(CPStr, Total);

  Idx := 0;
  while Idx < Total do
  begin
    // contractions
    RunEnd := MatchContraction(Idx);
    if RunEnd > 0 then
    begin
      Pieces.Add(Collect(Idx, Idx + RunEnd - 1));
      Inc(Idx, RunEnd);
      continue;
    end;
    if (CPs[Idx] = 32) and (Idx + 1 < Total) and IsLetterCP(CPs[Idx + 1]) then
    begin // " ?\p{L}+"
      RunEnd := Idx + 1;
      while (RunEnd + 1 < Total) and IsLetterCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    else if (CPs[Idx] = 32) and (Idx + 1 < Total) and
      IsNumberCP(CPs[Idx + 1]) then
    begin // " ?\p{N}+"
      RunEnd := Idx + 1;
      while (RunEnd + 1 < Total) and IsNumberCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    else if (CPs[Idx] = 32) and (Idx + 1 < Total) and
      (not IsWhitespaceCP(CPs[Idx + 1])) then
    begin // " ?[^\s\p{L}\p{N}]+"
      RunEnd := Idx + 1;
      while (RunEnd + 1 < Total) and IsOtherCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    else if IsLetterCP(CPs[Idx]) then
    begin
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and IsLetterCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    else if IsNumberCP(CPs[Idx]) then
    begin
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and IsNumberCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    else if not IsWhitespaceCP(CPs[Idx]) then
    begin
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and IsOtherCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    else
    begin // whitespace run: \s+(?!\S) | \s+
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and IsWhitespaceCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      if (RunEnd + 1 < Total) and (RunEnd > Idx) then
      begin
        // leave the last whitespace char to (possibly) attach to the
        // next piece via the " ?" prefixes
        Pieces.Add(Collect(Idx, RunEnd - 1));
        Idx := RunEnd;
      end
      else
      begin
        Pieces.Add(Collect(Idx, RunEnd));
        Idx := RunEnd + 1;
      end;
    end;
  end;
end;

// Splits a segment with the cl100k-style Split pre_tokenizer pattern
// (Qwen2 with \p{N}, Llama-3/cl100k with \p{N}{1,3}):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ |
//   \p{N}{1,FSplitDigitsMax} | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ |
//   \s+(?!\S) | \s+
// Hand-written ordered-alternation matcher (no regex engine), validated
// against HF `tokenizers` Split(behavior=Isolated, invert=false).
procedure TNeuralHFTokenizer.SplitCl100kPieces(const Segment: string;
  Pieces: TStringList);
var
  CPs: array of cardinal;
  CPStr: array of string;
  Total, Position, Idx, RunStart, RunEnd, LastNL, DigitCnt: integer;

  function Collect(StartIdx, EndIdx: integer): string;
  var
    J: integer;
  begin
    Result := '';
    for J := StartIdx to EndIdx do Result := Result + CPStr[J];
  end;

  function LowerAscii(CP: cardinal): cardinal;
  begin
    if (CP >= Ord('A')) and (CP <= Ord('Z')) then Result := CP + 32
    else Result := CP;
  end;

  // (?i:'s|'t|'re|'ve|'m|'ll|'d) -- returns the match length or 0.
  function MatchContractionCI(StartIdx: integer): integer;
  var
    C1, C2: cardinal;
  begin
    Result := 0;
    if CPs[StartIdx] <> Ord('''') then Exit;
    if StartIdx + 1 > High(CPs) then Exit;
    C1 := LowerAscii(CPs[StartIdx + 1]);
    if (C1 = Ord('s')) or (C1 = Ord('t')) or (C1 = Ord('m')) or
      (C1 = Ord('d')) then Exit(2);
    if StartIdx + 2 > High(CPs) then Exit;
    C2 := LowerAscii(CPs[StartIdx + 2]);
    if ((C1 = Ord('r')) and (C2 = Ord('e'))) or
      ((C1 = Ord('v')) and (C2 = Ord('e'))) or
      ((C1 = Ord('l')) and (C2 = Ord('l'))) then Exit(3);
  end;

  function IsNewlineCP(CP: cardinal): boolean;
  begin
    Result := (CP = 10) or (CP = 13);
  end;

begin
  // decode UTF-8 into codepoint arrays
  SetLength(CPs, Length(Segment));
  SetLength(CPStr, Length(Segment));
  Total := 0;
  Position := 1;
  while Position <= Length(Segment) do
  begin
    RunStart := Position;
    CPs[Total] := NextCodePoint(Segment, Position);
    CPStr[Total] := Copy(Segment, RunStart, Position - RunStart);
    Inc(Total);
  end;
  SetLength(CPs, Total);
  SetLength(CPStr, Total);

  Idx := 0;
  while Idx < Total do
  begin
    // 1. case-insensitive contractions
    RunEnd := MatchContractionCI(Idx);
    if RunEnd > 0 then
    begin
      Pieces.Add(Collect(Idx, Idx + RunEnd - 1));
      Inc(Idx, RunEnd);
      continue;
    end;
    // 2. [^\r\n\p{L}\p{N}]?\p{L}+  (optional ONE non-CR/LF non-letter
    //    non-number char glued to a letter run)
    if IsLetterCP(CPs[Idx]) or ((not IsNewlineCP(CPs[Idx])) and
      (not IsNumberCP(CPs[Idx])) and (Idx + 1 < Total) and
      IsLetterCP(CPs[Idx + 1])) then
    begin
      RunEnd := Idx;
      if not IsLetterCP(CPs[Idx]) then Inc(RunEnd); // the optional char
      while (RunEnd + 1 < Total) and IsLetterCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    // 3. \p{N}{1,DigitsMax}
    else if IsNumberCP(CPs[Idx]) then
    begin
      RunEnd := Idx;
      DigitCnt := 1;
      while (RunEnd + 1 < Total) and (DigitCnt < FSplitDigitsMax) and
        IsNumberCP(CPs[RunEnd + 1]) do
      begin
        Inc(RunEnd);
        Inc(DigitCnt);
      end;
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
    else if IsOtherCP(CPs[Idx]) or ((CPs[Idx] = 32) and (Idx + 1 < Total)
      and IsOtherCP(CPs[Idx + 1])) then
    begin
      RunEnd := Idx; // the optional leading space or the first punct char
      while (RunEnd + 1 < Total) and IsOtherCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      while (RunEnd + 1 < Total) and IsNewlineCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    else
    begin // whitespace alternatives over the run [Idx..RunEnd]
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and IsWhitespaceCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      // 5. \s*[\r\n]+ : up to (and including) the LAST newline in the run
      LastNL := -1;
      for Position := Idx to RunEnd do
        if IsNewlineCP(CPs[Position]) then LastNL := Position;
      if LastNL >= 0 then
      begin
        Pieces.Add(Collect(Idx, LastNL));
        Idx := LastNL + 1;
      end
      else if RunEnd + 1 >= Total then
      begin // 6. \s+(?!\S) : trailing whitespace takes the whole run
        Pieces.Add(Collect(Idx, RunEnd));
        Idx := RunEnd + 1;
      end
      else if RunEnd > Idx then
      begin // 6. \s+(?!\S) : leave the last char to attach to what follows
        Pieces.Add(Collect(Idx, RunEnd - 1));
        Idx := RunEnd;
      end
      else
      begin // 7. \s+ : a single whitespace char before non-space
        Pieces.Add(CPStr[Idx]);
        Inc(Idx);
      end;
    end;
  end;
end;

// Splits a segment with the DeepSeek-V2 / V2-Lite pre-tokenizer Sequence,
// which (unlike the cl100k family) is a chain of several Split stages plus a
// Digits(individual) stage. Behaviorally, in left-to-right order:
//   [\r\n]              -> every CR and LF is its OWN single-char piece
//   \s?[\p{L}]+         -> optional ONE leading space + maximal LETTER run
//                          (Latin/Greek/Cyrillic/... but NOT CJK)
//   \s?[punct]+         -> optional ONE leading space + maximal PUNCT run
//   [CJK]+              -> maximal CJK run (no leading-space prefix)
//   Digits(individual)  -> every digit is its OWN single-char piece
//   \s+$ / interior ws  -> a whitespace run with NO eligible following
//                          letter/punct keeps the WHOLE run as one piece;
//                          before a letter/punct the LAST space is peeled off
//                          as that run's \s? prefix (the rest stay as one
//                          piece).
// Validated against HF `tokenizers` for ASCII text, digits, ASCII/fullwidth
// punctuation, whitespace (incl. tabs/CR/LF) and CJK. NON-ASCII LETTERS use
// the same \p{L} approximation as IsLetterCP (see unit header) -- the HF onig
// engine isolates some precomposed Latin-1 letters that this table groups,
// so exact parity is only guaranteed over the validated classes.
procedure TNeuralHFTokenizer.SplitDeepSeekPieces(const Segment: string;
  Pieces: TStringList);
var
  CPs: array of cardinal;
  CPStr: array of string;
  Total, Position, Idx, RunStart, RunEnd: integer;

  function Collect(StartIdx, EndIdx: integer): string;
  var
    J: integer;
  begin
    Result := '';
    for J := StartIdx to EndIdx do Result := Result + CPStr[J];
  end;

  function IsNL(CP: cardinal): boolean;
  begin
    Result := (CP = 10) or (CP = 13);
  end;

  // DeepSeek CJK class: U+0800..U+9FA5 and Hangul U+AC00..U+D7FF.
  function IsCJK(CP: cardinal): boolean;
  begin
    Result := ((CP >= $800) and (CP <= $9FA5)) or
      ((CP >= $AC00) and (CP <= $D7FF));
  end;

  // DeepSeek letter class (NOT CJK -- CJK is its own stage above).
  function IsDSLetter(CP: cardinal): boolean;
  begin
    Result := IsLetterCP(CP) and (not IsCJK(CP));
  end;

  // \s but NOT CR/LF (those are isolated by the [\r\n] stage first).
  function IsSpace(CP: cardinal): boolean;
  begin
    Result := IsWhitespaceCP(CP) and (not IsNL(CP));
  end;

  // punct class: everything that is not space, letter, digit, CJK, newline.
  function IsPunct(CP: cardinal): boolean;
  begin
    Result := (not IsSpace(CP)) and (not IsNL(CP)) and
      (not IsDSLetter(CP)) and (not IsNumberCP(CP)) and (not IsCJK(CP));
  end;

begin
  // decode UTF-8 into codepoint arrays
  SetLength(CPs, Length(Segment));
  SetLength(CPStr, Length(Segment));
  Total := 0;
  Position := 1;
  while Position <= Length(Segment) do
  begin
    RunStart := Position;
    CPs[Total] := NextCodePoint(Segment, Position);
    CPStr[Total] := Copy(Segment, RunStart, Position - RunStart);
    Inc(Total);
  end;
  SetLength(CPs, Total);
  SetLength(CPStr, Total);

  Idx := 0;
  while Idx < Total do
  begin
    // 1. [\r\n] : each CR / LF isolated
    if IsNL(CPs[Idx]) then
    begin
      Pieces.Add(CPStr[Idx]);
      Inc(Idx);
    end
    // 2. \s?[\p{L}]+  (optional ONE leading space glued to a letter run)
    else if IsDSLetter(CPs[Idx]) or ((CPs[Idx] = 32) and (Idx + 1 < Total)
      and IsDSLetter(CPs[Idx + 1])) then
    begin
      RunEnd := Idx;
      if not IsDSLetter(CPs[Idx]) then Inc(RunEnd); // the optional space
      while (RunEnd + 1 < Total) and IsDSLetter(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    // 3. \s?[punct]+  (optional ONE leading space glued to a punct run)
    else if IsPunct(CPs[Idx]) or ((CPs[Idx] = 32) and (Idx + 1 < Total)
      and IsPunct(CPs[Idx + 1])) then
    begin
      RunEnd := Idx;
      if not IsPunct(CPs[Idx]) then Inc(RunEnd); // the optional space
      while (RunEnd + 1 < Total) and IsPunct(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    // 4. [CJK]+  (no leading-space prefix)
    else if IsCJK(CPs[Idx]) then
    begin
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and IsCJK(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    // 5. Digits(individual) : each digit isolated
    else if IsNumberCP(CPs[Idx]) then
    begin
      Pieces.Add(CPStr[Idx]);
      Inc(Idx);
    end
    else
    begin
      // whitespace run (no CR/LF inside -- those were isolated above). The
      // run extends to the next CR/LF or end. If it is followed by a letter
      // or punct, the LAST space becomes that run's \s? prefix; otherwise
      // (digit / CJK / end) the whole run is one piece.
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and IsSpace(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      if (RunEnd + 1 < Total) and (RunEnd > Idx) and
        (IsDSLetter(CPs[RunEnd + 1]) or IsPunct(CPs[RunEnd + 1])) then
      begin
        Pieces.Add(Collect(Idx, RunEnd - 1));
        Idx := RunEnd; // last space attaches to the following run
      end
      else
      begin
        Pieces.Add(Collect(Idx, RunEnd));
        Idx := RunEnd + 1;
      end;
    end;
  end;
end;

// Splits a segment with the o200k_base / GPT-4o-family Split pre_tokenizer
// pattern:
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lo}\p{M}]+
//                                                       (?i:contraction)? |
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lo}\p{M}]*
//                                                       (?i:contraction)? |
//   \p{N}{1,3} |  ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// Hand-written ordered-alternation matcher (no regex engine). EXACT byte
// parity is guaranteed only over ASCII letters/digits/punct/whitespace and
// CJK -- the same approximation stance as SplitCl100kPieces/SplitDeepSeek-
// Pieces. The Lu/Ll case distinction uses the ASCII case tables; non-ASCII
// letters are treated as the Ll/Lo (lowercase) class (CJK is Lo, and most
// accented Latin in the test corpus is lowercase), so the two letter
// alternations collapse to the cl100k single letter run for those scripts.
// \p{M} combining marks are treated as letters (no separate table).
procedure TNeuralHFTokenizer.SplitO200kPieces(const Segment: string;
  Pieces: TStringList);
var
  CPs: array of cardinal;
  CPStr: array of string;
  Total, Position, Idx, RunEnd, LastNL: integer;

  function Collect(StartIdx, EndIdx: integer): string;
  var
    J: integer;
  begin
    Result := '';
    for J := StartIdx to EndIdx do Result := Result + CPStr[J];
  end;

  function LowerAscii(CP: cardinal): cardinal;
  begin
    if (CP >= Ord('A')) and (CP <= Ord('Z')) then Result := CP + 32
    else Result := CP;
  end;

  // \p{Lu}\p{Lt}\p{Lm} approximated by ASCII A-Z.
  function IsUpperLetterCP(CP: cardinal): boolean;
  begin
    Result := (CP >= Ord('A')) and (CP <= Ord('Z'));
  end;

  // \p{Ll}\p{Lo}\p{M} approximated: any letter that is not ASCII-upper.
  function IsLowerLetterCP(CP: cardinal): boolean;
  begin
    Result := IsLetterCP(CP) and (not IsUpperLetterCP(CP));
  end;

  function IsNewlineCP(CP: cardinal): boolean;
  begin
    Result := (CP = 10) or (CP = 13);
  end;

  // (?i:'s|'t|'re|'ve|'m|'ll|'d) at StartIdx -- match length or 0.
  function MatchContractionCI(StartIdx: integer): integer;
  var
    C1, C2: cardinal;
  begin
    Result := 0;
    if StartIdx > High(CPs) then Exit;
    if CPs[StartIdx] <> Ord('''') then Exit;
    if StartIdx + 1 > High(CPs) then Exit;
    C1 := LowerAscii(CPs[StartIdx + 1]);
    if (C1 = Ord('s')) or (C1 = Ord('t')) or (C1 = Ord('m')) or
      (C1 = Ord('d')) then Exit(2);
    if StartIdx + 2 > High(CPs) then Exit;
    C2 := LowerAscii(CPs[StartIdx + 2]);
    if ((C1 = Ord('r')) and (C2 = Ord('e'))) or
      ((C1 = Ord('v')) and (C2 = Ord('e'))) or
      ((C1 = Ord('l')) and (C2 = Ord('l'))) then Exit(3);
  end;

  // Tries the two letter alternations starting at Idx. Returns the index of
  // the LAST matched codepoint (>= Idx) or -1 when neither alternation
  // matches. HasLead := the segment may start with ONE optional
  // [^\r\n\p{L}\p{N}] char glued to the letter run.
  function MatchLetterAlt(Idx: integer): integer;
  var
    P, UpperEnd, LowerEnd: integer;
    LeadOk: boolean;
  begin
    Result := -1;
    P := Idx;
    LeadOk := false;
    // optional leading non-CR/LF non-letter non-number char
    if (P <= High(CPs)) and (not IsNewlineCP(CPs[P])) and
      (not IsLetterCP(CPs[P])) and (not IsNumberCP(CPs[P])) then
    begin
      LeadOk := true;
      Inc(P);
    end;
    // [\p{Lu}...]* greedy uppercase run
    UpperEnd := P - 1;
    while (P <= High(CPs)) and IsUpperLetterCP(CPs[P]) do
    begin
      UpperEnd := P;
      Inc(P);
    end;
    // [\p{Ll}...]+ : alt 1 requires >=1 lowercase letter here
    LowerEnd := P - 1;
    while (P <= High(CPs)) and IsLowerLetterCP(CPs[P]) do
    begin
      LowerEnd := P;
      Inc(P);
    end;
    if LowerEnd >= UpperEnd + 1 then
    begin
      // alt 1 matched (had at least one lowercase letter). End = LowerEnd
      // (+ optional contraction).
      Result := LowerEnd;
      Inc(Result, MatchContractionCI(Result + 1));
      Exit;
    end;
    // alt 2: [\p{Lu}...]+[\p{Ll}...]* -- needs >=1 uppercase letter (no
    // lowercase was consumed because alt 1 failed). The uppercase run above
    // already is the [\p{Lu}...]+; lowercase run is empty.
    if UpperEnd >= Idx then
    begin
      Result := UpperEnd;
      Inc(Result, MatchContractionCI(Result + 1));
      Exit;
    end;
    // Neither letter alternation produced a letter. If the only thing
    // matched was the optional lead char, the alternation does NOT apply
    // (the lead char is part of the punct alternation instead).
    if LeadOk then Result := -1;
  end;

begin
  // decode UTF-8 into codepoint arrays
  SetLength(CPs, Length(Segment));
  SetLength(CPStr, Length(Segment));
  Total := 0;
  Position := 1;
  while Position <= Length(Segment) do
  begin
    LastNL := Position;
    CPs[Total] := NextCodePoint(Segment, Position);
    CPStr[Total] := Copy(Segment, LastNL, Position - LastNL);
    Inc(Total);
  end;
  SetLength(CPs, Total);
  SetLength(CPStr, Total);

  Idx := 0;
  while Idx < Total do
  begin
    // 1+2. letter alternations (case-aware) -- tried before everything else
    RunEnd := MatchLetterAlt(Idx);
    if RunEnd >= Idx then
    begin
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
      continue;
    end;
    // 3. \p{N}{1,3}
    if IsNumberCP(CPs[Idx]) then
    begin
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and (RunEnd - Idx + 1 < 3) and
        IsNumberCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n/]*   (note: trailing run includes '/')
    else if IsOtherCP(CPs[Idx]) or ((CPs[Idx] = 32) and (Idx + 1 < Total)
      and IsOtherCP(CPs[Idx + 1])) then
    begin
      RunEnd := Idx; // optional leading space or first punct char
      while (RunEnd + 1 < Total) and IsOtherCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      while (RunEnd + 1 < Total) and (IsNewlineCP(CPs[RunEnd + 1]) or
        (CPs[RunEnd + 1] = Ord('/'))) do
        Inc(RunEnd);
      Pieces.Add(Collect(Idx, RunEnd));
      Idx := RunEnd + 1;
    end
    else
    begin // whitespace alternatives over the run [Idx..RunEnd]
      RunEnd := Idx;
      while (RunEnd + 1 < Total) and IsWhitespaceCP(CPs[RunEnd + 1]) do
        Inc(RunEnd);
      // 5. \s*[\r\n]+ : up to (and including) the LAST newline in the run
      LastNL := -1;
      for Position := Idx to RunEnd do
        if IsNewlineCP(CPs[Position]) then LastNL := Position;
      if LastNL >= 0 then
      begin
        Pieces.Add(Collect(Idx, LastNL));
        Idx := LastNL + 1;
      end
      else if RunEnd + 1 >= Total then
      begin // 6. \s+(?!\S) : trailing whitespace takes the whole run
        Pieces.Add(Collect(Idx, RunEnd));
        Idx := RunEnd + 1;
      end
      else if RunEnd > Idx then
      begin // 6. \s+(?!\S) : leave the last char to attach to what follows
        Pieces.Add(Collect(Idx, RunEnd - 1));
        Idx := RunEnd;
      end
      else
      begin // 7. \s+ : a single whitespace char before non-space
        Pieces.Add(CPStr[Idx]);
        Inc(Idx);
      end;
    end;
  end;
end;

// Maps a raw piece to its byte-level alphabet symbols (one mapped
// codepoint per input byte). Caller frees the result.
function TNeuralHFTokenizer.MapPieceToByteLevel(
  const Piece: string): TStringList;
var
  Cnt: integer;
begin
  Result := TStringList.Create();
  for Cnt := 1 to Length(Piece) do
    Result.Add(CodePointToUTF8(FByteToCP[Ord(Piece[Cnt])]));
end;

procedure TNeuralHFTokenizer.EmitTokenOrFallback(const Symbol: string;
  Ids: TIntegerList);
var
  TokenId, Cnt: integer;
  ByteToken: string;
begin
  TokenId := VocabFind(Symbol);
  if TokenId >= 0 then
  begin
    Ids.Add(TokenId);
    Exit;
  end;
  if FByteFallback then
  begin
    // check all byte tokens exist before emitting any
    for Cnt := 1 to Length(Symbol) do
      if VocabFind('<0x' + IntToHex(Ord(Symbol[Cnt]), 2) + '>') < 0 then
      begin
        if (FUnkId >= 0) and ((not FFuseUnk) or (Ids.Count = 0) or
          (Ids[Ids.Count - 1] <> FUnkId)) then
          Ids.Add(FUnkId);
        Exit;
      end;
    for Cnt := 1 to Length(Symbol) do
    begin
      ByteToken := '<0x' + IntToHex(Ord(Symbol[Cnt]), 2) + '>';
      Ids.Add(VocabFind(ByteToken));
    end;
  end
  else if (FUnkId >= 0) and ((not FFuseUnk) or (Ids.Count = 0) or
    (Ids[Ids.Count - 1] <> FUnkId)) then
    Ids.Add(FUnkId);
end;

// Ranked BPE merge loop over the symbols of one word/piece.
procedure TNeuralHFTokenizer.BPEWord(const Symbols: TStringList;
  Ids: TIntegerList);
var
  Cnt, BestIdx, BestRank, Rank, WholeId: integer;
  Whole: string;
begin
  if Symbols.Count = 0 then exit;
  // With BPE-dropout active, skip the whole-word fast path so the merge loop
  // below runs and dropout can produce an alternative segmentation.
  if FIgnoreMerges and (FDropoutProb <= 0) then
  begin
    Whole := '';
    for Cnt := 0 to Symbols.Count - 1 do Whole := Whole + Symbols[Cnt];
    WholeId := VocabFind(Whole);
    if WholeId >= 0 then
    begin
      Ids.Add(WholeId);
      Exit;
    end;
  end;
  while Symbols.Count > 1 do
  begin
    BestIdx := -1;
    BestRank := High(integer);
    for Cnt := 0 to Symbols.Count - 2 do
    begin
      Rank := MergeRank(Symbols[Cnt], Symbols[Cnt + 1]);
      // BPE-dropout (Provilkov et al. 2020): with probability FDropoutProb
      // skip an otherwise-applicable merge, so the segmentation falls back to
      // shorter symbols. Train-time only (FDropoutProb > 0); the symbols still
      // concatenate back to the original word, so decode round-trips.
      if (FDropoutProb > 0) and (Rank < High(integer)) and
        (Random < FDropoutProb) then
        continue;
      if Rank < BestRank then
      begin
        BestRank := Rank;
        BestIdx := Cnt;
      end;
    end;
    if BestIdx < 0 then break;
    Symbols[BestIdx] := Symbols[BestIdx] + Symbols[BestIdx + 1];
    Symbols.Delete(BestIdx + 1);
  end;
  for Cnt := 0 to Symbols.Count - 1 do
    EmitTokenOrFallback(Symbols[Cnt], Ids);
end;

// Unigram (SentencePiece) Viterbi segmentation of one pre-tokenized piece.
// Builds the best (maximum total log-prob) segmentation into vocab pieces;
// any single character not starting a vocab piece is covered by a 1-char
// <unk> node with score FUniMinScore - kUnkPenalty (HF's kUnkPenalty=10.0).
// Adjacent <unk> nodes are fused into one emitted <unk> id (HF behavior).
procedure TNeuralHFTokenizer.UnigramWord(const PieceText: string;
  Ids: TIntegerList);
const
  kUnkPenalty = 10.0;
var
  CPStart: array of integer;  // byte offset (1-based) where char i starts
  SegId: array of integer;    // backtracked ids, collected in reverse
  SegStart: array of integer; // backtracked start CHAR index of each segment
  Total, Position, SegCnt, I, J, BestPrev, TokenId: integer;
  BestScore: array of double; // best score to reach char-boundary J (0..Total)
  BackPtr: array of integer;  // start char index of the piece ending at J
  BackId: array of integer;   // vocab id of that piece, or -1 for unk
  CandScore, UnkScore: double;
  Sub: string;
  LastWasUnk: boolean;

  // Emits the SentencePiece <0xNN> BYTE pieces for the raw bytes of the
  // single-char unk span [CharStart, CharEnd) (char indices). Returns True
  // when every needed byte piece exists; False otherwise (caller falls back
  // to <unk>). byte_fallback ALWAYS covers all 256 bytes, so this succeeds
  // whenever FByteFallback is set on a real model.
  function EmitByteFallback(CharStart, CharEnd: integer): boolean;
  var
    K, BId: integer;
  begin
    Result := false;
    if not FByteFallback then Exit;
    // verify all byte pieces exist before emitting any
    for K := CPStart[CharStart] to CPStart[CharEnd] - 1 do
      if VocabFind('<0x' + IntToHex(Ord(PieceText[K]), 2) + '>') < 0 then
        Exit;
    for K := CPStart[CharStart] to CPStart[CharEnd] - 1 do
    begin
      BId := VocabFind('<0x' + IntToHex(Ord(PieceText[K]), 2) + '>');
      Ids.Add(BId);
    end;
    Result := true;
  end;

begin
  if PieceText = '' then exit;
  // Char-boundary table: CPStart[i] is the 1-based byte offset of char i,
  // with a sentinel one past the last byte.
  SetLength(CPStart, Length(PieceText) + 1);
  Total := 0;
  Position := 1;
  while Position <= Length(PieceText) do
  begin
    CPStart[Total] := Position;
    NextCodePoint(PieceText, Position);
    Inc(Total);
  end;
  CPStart[Total] := Position;
  if Total = 0 then exit;

  UnkScore := FUniMinScore - kUnkPenalty;
  SetLength(BestScore, Total + 1);
  SetLength(BackPtr, Total + 1);
  SetLength(BackId, Total + 1);
  for I := 1 to Total do
  begin
    BestScore[I] := NegInfinity;
    BackPtr[I] := -1;
    BackId[I] := -1;
  end;
  BestScore[0] := 0;

  // Forward DP: BestScore[J] = best total log-prob for chars [0..J).
  for J := 1 to Total do
    for I := 0 to J - 1 do
    begin
      if BestScore[I] = NegInfinity then continue;
      Sub := Copy(PieceText, CPStart[I], CPStart[J] - CPStart[I]);
      TokenId := VocabFind(Sub);
      if TokenId >= 0 then
        CandScore := BestScore[I] + FUniScore[TokenId]
      else if J = I + 1 then
      begin
        // single-character unk node (spans exactly one character)
        CandScore := BestScore[I] + UnkScore;
        TokenId := -1;
      end
      else
        continue;
      if CandScore > BestScore[J] then
      begin
        BestScore[J] := CandScore;
        BackPtr[J] := I;
        BackId[J] := TokenId;
      end;
    end;

  // Backtrack (collect ids in reverse), then emit forward, fusing adjacent
  // unk ids into one.
  SetLength(SegId, Total);
  SetLength(SegStart, Total);
  SegCnt := 0;
  J := Total;
  while J > 0 do
  begin
    BestPrev := BackPtr[J];
    if BestPrev < 0 then BestPrev := J - 1; // safety (shouldn't happen)
    SegId[SegCnt] := BackId[J];
    SegStart[SegCnt] := BestPrev; // start char index (unk spans 1 char)
    Inc(SegCnt);
    J := BestPrev;
  end;
  LastWasUnk := false;
  for I := SegCnt - 1 downto 0 do
    if SegId[I] < 0 then
    begin
      // Unknown char: with byte_fallback, decompose it into its <0xNN> BYTE
      // pieces (SentencePiece behavior) instead of emitting one fused <unk>.
      if EmitByteFallback(SegStart[I], SegStart[I] + 1) then
        LastWasUnk := false
      else
      begin
        if (FUnkId >= 0) and (not LastWasUnk) then Ids.Add(FUnkId);
        LastWasUnk := true;
      end;
    end
    else
    begin
      Ids.Add(SegId[I]);
      LastWasUnk := false;
    end;
end;

// BertNormalizer: clean_text -> handle_chinese_chars -> strip_accents ->
// lowercase (HF order). clean_text drops control chars and canonicalizes
// \t\n\r to spaces; handle_chinese_chars pads CJK ideographs with spaces.
function TNeuralHFTokenizer.BertNormalize(const Segment: string): string;
var
  Position: integer;
  CP, Base: cardinal;
begin
  Result := '';
  Position := 1;
  while Position <= Length(Segment) do
  begin
    CP := NextCodePoint(Segment, Position);
    if FBertCleanText then
    begin
      if (CP = 0) or (CP = $FFFD) then continue;
      if (CP = 9) or (CP = 10) or (CP = 13) then CP := 32
      else if (CP < 32) or (CP = 127) then continue; // other control chars
    end;
    if FBertHandleChinese and IsCJKIdeographCP(CP) then
    begin
      Result := Result + ' ' + CodePointToUTF8(CP) + ' ';
      continue;
    end;
    if FBertStripAccents then
    begin
      Base := StripAccentBaseCP(CP);
      if Base = 1 then continue; // bare combining mark: drop
      if Base <> 0 then CP := Base;
    end;
    if FBertLowercase then CP := LowercaseCP(CP);
    Result := Result + CodePointToUTF8(CP);
  end;
end;

// BertPreTokenizer: split on whitespace, then isolate every punctuation
// character as its own piece.
procedure TNeuralHFTokenizer.BertPieces(const Segment: string;
  Pieces: TStringList);
var
  Position, RunStart: integer;
  CP: cardinal;
  Current: string;
begin
  Current := '';
  Position := 1;
  while Position <= Length(Segment) do
  begin
    RunStart := Position;
    CP := NextCodePoint(Segment, Position);
    if IsWhitespaceCP(CP) then
    begin
      if Current <> '' then Pieces.Add(Current);
      Current := '';
    end
    else if IsBertPunctuationCP(CP) then
    begin
      if Current <> '' then Pieces.Add(Current);
      Current := '';
      Pieces.Add(Copy(Segment, RunStart, Position - RunStart));
    end
    else
      Current := Current + Copy(Segment, RunStart, Position - RunStart);
  end;
  if Current <> '' then Pieces.Add(Current);
end;

// Greedy longest-match-first WordPiece over one pre-tokenized word:
// repeatedly take the LONGEST vocab entry matching a prefix of what is
// left ('##' + suffix when not at the word start); any failure marks the
// WHOLE word as [UNK] (HF semantics).
procedure TNeuralHFTokenizer.WordPieceWord(const WordText: string;
  Ids: TIntegerList);
var
  CPStr: array of string;
  Total, Position, RunStart: integer;
  StartIdx, EndIdx, TokenId, Found, Cnt: integer;
  Sub: string;
  Matched: TIntegerList;
begin
  // split the word into codepoints
  SetLength(CPStr, Length(WordText));
  Total := 0;
  Position := 1;
  while Position <= Length(WordText) do
  begin
    RunStart := Position;
    NextCodePoint(WordText, Position);
    CPStr[Total] := Copy(WordText, RunStart, Position - RunStart);
    Inc(Total);
  end;
  if Total = 0 then exit;
  if Total > FWPMaxChars then
  begin
    if FUnkId >= 0 then Ids.Add(FUnkId);
    exit;
  end;
  Matched := TIntegerList.Create();
  try
    StartIdx := 0;
    while StartIdx < Total do
    begin
      Found := -1;
      EndIdx := Total;
      while EndIdx > StartIdx do
      begin
        Sub := '';
        for Cnt := StartIdx to EndIdx - 1 do Sub := Sub + CPStr[Cnt];
        if StartIdx > 0 then Sub := FWPPrefix + Sub;
        TokenId := VocabFind(Sub);
        if TokenId >= 0 then
        begin
          Found := TokenId;
          break;
        end;
        Dec(EndIdx);
      end;
      if Found < 0 then
      begin // whole word becomes [UNK]
        if FUnkId >= 0 then Ids.Add(FUnkId);
        exit;
      end;
      Matched.Add(Found);
      StartIdx := EndIdx;
    end;
    for Cnt := 0 to Matched.Count - 1 do Ids.Add(Matched[Cnt]);
  finally
    Matched.Free;
  end;
end;

procedure TNeuralHFTokenizer.EncodeSegment(const Segment: string;
  Ids: TIntegerList; IsFirstSegment: boolean);
var
  Pieces, Symbols: TStringList;
  Cnt, Position, RunStart, PieceStart: integer;
  Normalized: string;
  Seg: string;

  // BPEs Normalized[PieceFrom..PieceTo] (1-based bytes) over codepoints.
  procedure BPECodePoints(PieceFrom, PieceTo: integer);
  var
    CPSyms: TStringList;
    BytePos, CPStart: integer;
  begin
    if FUnigram then
    begin
      UnigramWord(Copy(Normalized, PieceFrom, PieceTo - PieceFrom + 1), Ids);
      exit;
    end;
    CPSyms := TStringList.Create();
    try
      BytePos := PieceFrom;
      while BytePos <= PieceTo do
      begin
        CPStart := BytePos;
        NextCodePoint(Normalized, BytePos);
        CPSyms.Add(Copy(Normalized, CPStart, BytePos - CPStart));
      end;
      BPEWord(CPSyms, Ids);
    finally
      CPSyms.Free;
    end;
  end;

begin
  if Segment = '' then exit;
  // Unicode normalization runs FIRST, before any pre-tokenization, matching
  // HF where the normalizer precedes the pre_tokenizer. All four forms are
  // real: NFC/NFD canonical, NFKC/NFKD compatibility (compat decompositions
  // applied before canonical ordering -- see UnicodeNormalize).
  if FNormUnicodeForm = 'NFC' then
    Seg := UnicodeNormalize(Segment, True)
  else if FNormUnicodeForm = 'NFD' then
    Seg := UnicodeNormalize(Segment, False)
  else if FNormUnicodeForm = 'NFKC' then
    Seg := UnicodeNormalize(Segment, True, True)
  else if FNormUnicodeForm = 'NFKD' then
    Seg := UnicodeNormalize(Segment, False, True)
  else
    Seg := Segment;
  if FWordPiece then
  begin
    Pieces := TStringList.Create();
    try
      BertPieces(BertNormalize(Seg), Pieces);
      for Cnt := 0 to Pieces.Count - 1 do
        WordPieceWord(Pieces[Cnt], Ids);
    finally
      Pieces.Free;
    end;
  end
  else if FDeepSeekPreTok then
  begin
    // DeepSeek-V2/V2-Lite multi-Split+Digits Sequence, then byte-level
    // alphabet + BPE per piece (NO GPT-2 regex pass).
    Pieces := TStringList.Create();
    try
      SplitDeepSeekPieces(Seg, Pieces);
      for Cnt := 0 to Pieces.Count - 1 do
      begin
        Symbols := MapPieceToByteLevel(Pieces[Cnt]);
        try
          BPEWord(Symbols, Ids);
        finally
          Symbols.Free;
        end;
      end;
    finally
      Pieces.Free;
    end;
  end
  else if FO200kPreTok then
  begin
    // Sequence[Split(o200k-style), ByteLevel(use_regex=false)]:
    // case-aware hand-written splitter, then byte-level alphabet + BPE
    // per piece (NO GPT-2 regex pass).
    Pieces := TStringList.Create();
    try
      SplitO200kPieces(Seg, Pieces);
      for Cnt := 0 to Pieces.Count - 1 do
      begin
        Symbols := MapPieceToByteLevel(Pieces[Cnt]);
        try
          BPEWord(Symbols, Ids);
        finally
          Symbols.Free;
        end;
      end;
    finally
      Pieces.Free;
    end;
  end
  else if FSplitPreTok then
  begin
    // Sequence[Split(cl100k-style), ByteLevel(use_regex=false)]:
    // hand-written splitter, then byte-level alphabet + BPE per piece
    // (NO GPT-2 regex pass).
    Pieces := TStringList.Create();
    try
      SplitCl100kPieces(Seg, Pieces);
      for Cnt := 0 to Pieces.Count - 1 do
      begin
        Symbols := MapPieceToByteLevel(Pieces[Cnt]);
        try
          BPEWord(Symbols, Ids);
        finally
          Symbols.Free;
        end;
      end;
    finally
      Pieces.Free;
    end;
  end
  else if FByteLevel then
  begin
    Normalized := Seg;
    if FAddPrefixSpace and (Length(Normalized) > 0) and
      (Normalized[1] <> ' ') then
      Normalized := ' ' + Normalized;
    Pieces := TStringList.Create();
    try
      if FByteLevelUseRegex then
        ByteLevelPieces(Normalized, Pieces)
      else if Length(Normalized) > 0 then
        // use_regex=false: NO GPT-2 regex split -- the whole segment is a
        // single chunk fed straight to the byte-level alphabet + BPE.
        Pieces.Add(Normalized);
      for Cnt := 0 to Pieces.Count - 1 do
      begin
        Symbols := MapPieceToByteLevel(Pieces[Cnt]);
        try
          BPEWord(Symbols, Ids);
        finally
          Symbols.Free;
        end;
      end;
    finally
      Pieces.Free;
    end;
  end
  else
  begin
    // metaspace family: normalizer chain first (Prepend/Replace --
    // Llama-2 style files do metaspace HERE and have no pre_tokenizer)
    Normalized := FNormPrepend + Seg;
    for Cnt := 0 to High(FNormReplaceFrom) do
      Normalized := StringReplace(Normalized, FNormReplaceFrom[Cnt],
        FNormReplaceTo[Cnt], [rfReplaceAll]);
    if FMetaspacePreTok then
    begin
      // Metaspace pre_tokenizer (Mistral / legacy=false Llama style):
      // space -> replacement, prepend per scheme (never doubled; 'first'
      // only on the segment starting at input offset 0 -- HF semantics),
      // then optionally split before each replacement.
      Normalized := StringReplace(Normalized, ' ', FMSReplacement,
        [rfReplaceAll]);
      if (FMSPrependScheme = 'always') or
        ((FMSPrependScheme = 'first') and IsFirstSegment) then
        if Copy(Normalized, 1, Length(FMSReplacement)) <> FMSReplacement
        then Normalized := FMSReplacement + Normalized;
      if FMSSplit then
      begin
        PieceStart := 1;
        Position := 1;
        while Position <= Length(Normalized) do
        begin
          RunStart := Position;
          NextCodePoint(Normalized, Position);
          if (RunStart > PieceStart) and
            (Copy(Normalized, RunStart, Length(FMSReplacement)) =
             FMSReplacement) then
          begin // new piece starts at each replacement (MergedWithNext)
            BPECodePoints(PieceStart, RunStart - 1);
            PieceStart := RunStart;
          end;
        end;
        BPECodePoints(PieceStart, Length(Normalized));
        exit;
      end;
    end;
    // whole-segment BPE over codepoints
    BPECodePoints(1, Length(Normalized));
  end;
end;

class function TNeuralHFTokenizer.NormalizeUnicodeText(const Text: string;
  Compose: boolean; Compat: boolean): string;
begin
  Result := UnicodeNormalize(Text, Compose, Compat);
end;

procedure TNeuralHFTokenizer.Encode(const Text: string; Ids: TIntegerList);
var
  Position, SegStart, TokenIndex: integer;
begin
  if GetVocabSize() = 0 then
    raise EHFTokenizerError.Create('Encode called before LoadFromFile.');
  Position := 1;
  SegStart := 1;
  while Position <= Length(Text) do
  begin
    if FindAddedToken(Text, Position, TokenIndex) then
    begin
      if Position > SegStart then
        EncodeSegment(Copy(Text, SegStart, Position - SegStart), Ids,
          SegStart = 1);
      Ids.Add(FAddedTokens[TokenIndex].Id);
      Inc(Position, Length(FAddedTokens[TokenIndex].Content));
      SegStart := Position;
    end
    else
      Inc(Position);
  end;
  if Position > SegStart then
    EncodeSegment(Copy(Text, SegStart, Position - SegStart), Ids,
      SegStart = 1);
end;

function TNeuralHFTokenizer.Encode(const Text: string): TNeuralIntegerArray;
var
  Ids: TIntegerList;
  Cnt: integer;
begin
  Ids := TIntegerList.Create();
  try
    Encode(Text, Ids);
    SetLength(Result, Ids.Count);
    for Cnt := 0 to Ids.Count - 1 do Result[Cnt] := Ids[Cnt];
  finally
    Ids.Free;
  end;
end;

function TNeuralHFTokenizer.EncodeWithOffsets(
  const Text: string): TNeuralTokenOffsetArray;
var
  Ids: TIntegerList;
  WordOf: array of integer;   // 1-based byte pos -> 0-based word index
  Cnt, Cursor, TokenIndex, SurfPos, SurfLen, StartPos, WordStart: integer;
  InWord: boolean;
  WordCnt: integer;
  Surface: string;
  C: char;
  Matched: boolean;

  // Is byte C whitespace in the original text / a decoded surface?
  function IsWs(Ch: char): boolean; inline;
  begin
    Result := Ch in [' ', #9, #10, #13];
  end;

begin
  Ids := TIntegerList.Create();
  try
    Encode(Text, Ids);
    SetLength(Result, Ids.Count);

    // Per-byte word index: a word starts at each whitespace -> non-whitespace
    // transition (whitespace-split words, HF word_ids granularity).
    SetLength(WordOf, Length(Text) + 1);
    WordCnt := -1;
    InWord := false;
    for Cnt := 1 to Length(Text) do
    begin
      if IsWs(Text[Cnt]) then
        InWord := false
      else
      begin
        if not InWord then begin Inc(WordCnt); InWord := true; end;
      end;
      if InWord then WordOf[Cnt] := WordCnt else WordOf[Cnt] := -1;
    end;

    // Byte-exact alignment by surface-byte consumption (not a post-hoc whole-
    // surface PosEx search). Every non-special token's decoded surface bytes
    // are matched, byte for byte, against the original text starting at a
    // running Cursor; the matched byte span becomes the token's offset. This
    // guarantees a span for byte-fallback fragments and metaspace tokens that
    // the old surface-PosEx heuristic left unmapped:
    //   * a multi-byte UTF-8 char split across several <0xNN> byte tokens: each
    //     byte token consumes exactly its one source byte, so every fragment
    //     points at its byte sub-range of that same source char (CONVENTION:
    //     split-char tokens get a per-byte span, not the whole-char span).
    //   * metaspace ('_' U+2581) and byte-level leading-space tokens: the
    //     surface space is matched against a source space, or, when it is a
    //     synthetic prepended space (add_dummy_prefix / AddPrefixSpace) with no
    //     source space at the cursor, it is skipped without advancing.
    //   * '##' WordPiece continuation prefix is stripped before matching.
    // A token whose surface still cannot be aligned (true unk with no source
    // bytes, e.g. unk_surface) keeps Start=0/Length=0/WordId=-1.
    Cursor := 1;
    for Cnt := 0 to Ids.Count - 1 do
    begin
      Result[Cnt].Id := Ids[Cnt];
      Result[Cnt].Start := 0;
      Result[Cnt].Length := 0;
      Result[Cnt].WordId := -1;

      // Added/special tokens have no surface text in the input.
      if IsAddedTokenId(Ids[Cnt], TokenIndex) then Continue;

      Surface := DecodeToken(Ids[Cnt]);
      // Strip the WordPiece continuation prefix ('##') so the surface is the
      // raw substring (WordPiece tokens carry no leading-space byte).
      if FWordPiece and (FWPPrefix <> '') and
        (Copy(Surface, 1, Length(FWPPrefix)) = FWPPrefix) then
        Surface := Copy(Surface, Length(FWPPrefix) + 1, MaxInt);
      if Surface = '' then Continue;

      // Consume Surface byte by byte against Text[Cursor..].
      SurfPos := 1;
      SurfLen := Length(Surface);
      StartPos := 0;          // first matched source byte (0 = none yet)
      WordStart := 0;         // first matched NON-whitespace byte (for WordId)
      Matched := true;
      while SurfPos <= SurfLen do
      begin
        C := Surface[SurfPos];
        if IsWs(C) then
        begin
          // A surface whitespace byte: the leading metaspace/prefix space, a
          // real inter-word space, or a pure word-boundary token (whose whole
          // surface is just that space). Claim one source whitespace byte if
          // the source has whitespace at the cursor; if not, it is a synthetic
          // prepended space -- drop it from the surface without advancing.
          if (Cursor <= Length(Text)) and IsWs(Text[Cursor]) then
          begin
            if StartPos = 0 then StartPos := Cursor;
            Inc(Cursor);
            // collapse any further source whitespace into this one span
            while (Cursor <= Length(Text)) and IsWs(Text[Cursor]) do
              Inc(Cursor);
          end;
          Inc(SurfPos);
          Continue;
        end;
        // A non-whitespace surface byte must match the source byte exactly.
        // Skip leading source whitespace once (token order stays monotonic).
        while (Cursor <= Length(Text)) and IsWs(Text[Cursor]) do
          Inc(Cursor);
        if (Cursor <= Length(Text)) and (Text[Cursor] = C) then
        begin
          if StartPos = 0 then StartPos := Cursor;
          if WordStart = 0 then WordStart := Cursor;
          Inc(Cursor);
          Inc(SurfPos);
        end
        else
        begin
          Matched := false;
          break;
        end;
      end;

      if Matched and (StartPos > 0) then
      begin
        Result[Cnt].Start := StartPos;
        Result[Cnt].Length := Cursor - StartPos;
        // Subword -> word alignment off the first real (non-whitespace) byte:
        // a leading-space byte-level/metaspace token (surface " cat") carries
        // the word of "cat", not the inter-word space. A pure word-boundary
        // token (whole surface is whitespace) has no word -> -1.
        if WordStart > 0 then
          Result[Cnt].WordId := WordOf[WordStart]
        else
          Result[Cnt].WordId := -1;
      end;
    end;
  finally
    Ids.Free;
  end;
end;

// Decodes one NON-added token id to its surface text (byte-level unmapping
// or metaspace replacement + byte-fallback included).
function TNeuralHFTokenizer.DecodeToken(Id: integer): string;
var
  Token: string;
  Position, ByteVal, Cnt: integer;
  CP: cardinal;
begin
  Result := '';
  if (Id < 0) or (Id > High(FIdToToken)) then exit;
  Token := FIdToToken[Id];
  if FByteLevel then
  begin
    Position := 1;
    while Position <= Length(Token) do
    begin
      CP := NextCodePoint(Token, Position);
      if (CP <= cardinal(High(FCPToByte))) and (FCPToByte[CP] >= 0)
      then Result := Result + Chr(FCPToByte[CP])
      else Result := Result + CodePointToUTF8(CP);
    end;
  end
  else
  begin
    // byte-fallback token? "<0xNN>"
    if FByteFallback and (Length(Token) = 6) and (Copy(Token, 1, 3) = '<0x')
      and (Token[6] = '>') then
    begin
      ByteVal := StrToIntDef('$' + Copy(Token, 4, 2), -1);
      if ByteVal >= 0 then Exit(Chr(ByteVal));
    end;
    Result := Token;
    for Cnt := 0 to High(FDecReplaceFrom) do
      Result := StringReplace(Result, FDecReplaceFrom[Cnt],
        FDecReplaceTo[Cnt], [rfReplaceAll]);
  end;
end;

function TNeuralHFTokenizer.Decode(const Ids: array of integer;
  SkipSpecialTokens: boolean = true): string;
var
  Cnt, TokenIndex, Stripped: integer;
  Piece: string;
begin
  Result := '';
  for Cnt := 0 to High(Ids) do
  begin
    if IsAddedTokenId(Ids[Cnt], TokenIndex) then
    begin
      if not (SkipSpecialTokens and FAddedTokens[TokenIndex].Special) then
      begin
        if FWordPiece and (Result <> '') then Result := Result + ' ';
        Result := Result + FAddedTokens[TokenIndex].Content;
      end;
    end
    else if FWordPiece then
    begin
      // WordPiece decoder: join with spaces, gluing '##' continuations.
      Piece := DecodeToken(Ids[Cnt]);
      if (FWPPrefix <> '') and
        (Copy(Piece, 1, Length(FWPPrefix)) = FWPPrefix) then
        Result := Result + Copy(Piece, Length(FWPPrefix) + 1, MaxInt)
      else
      begin
        if Result <> '' then Result := Result + ' ';
        Result := Result + Piece;
      end;
    end
    else
      Result := Result + DecodeToken(Ids[Cnt]);
  end;
  if FWordPiece and FDecWordPieceCleanup then
  begin
    // HF WordPiece decoder cleanup: re-attach punctuation/contractions.
    Result := StringReplace(Result, ' .', '.', [rfReplaceAll]);
    Result := StringReplace(Result, ' ?', '?', [rfReplaceAll]);
    Result := StringReplace(Result, ' !', '!', [rfReplaceAll]);
    Result := StringReplace(Result, ' ,', ',', [rfReplaceAll]);
    Result := StringReplace(Result, ' n''t', 'n''t', [rfReplaceAll]);
    Result := StringReplace(Result, ' ''m', '''m', [rfReplaceAll]);
    Result := StringReplace(Result, ' ''s', '''s', [rfReplaceAll]);
    Result := StringReplace(Result, ' ''ve', '''ve', [rfReplaceAll]);
    Result := StringReplace(Result, ' ''re', '''re', [rfReplaceAll]);
  end;
  // decoder Strip(' ', start, 0): drop up to N leading spaces
  Stripped := 0;
  while (Stripped < FDecStripLeft) and (Length(Result) > Stripped) and
    (Result[Stripped + 1] = ' ') do
    Inc(Stripped);
  if Stripped > 0 then Delete(Result, 1, Stripped);
end;

function TNeuralHFTokenizer.Decode(Ids: TIntegerList;
  SkipSpecialTokens: boolean = true): string;
var
  Arr: array of integer;
  Cnt: integer;
begin
  SetLength(Arr, Ids.Count);
  for Cnt := 0 to Ids.Count - 1 do Arr[Cnt] := Ids[Cnt];
  Result := Decode(Arr, SkipSpecialTokens);
end;

function TNeuralHFTokenizer.FragmentToSurface(const Fragment: string): string;
var
  Cnt: integer;
begin
  if FByteLevel then
  begin
    // Each raw byte -> its GPT-2 byte-alphabet codepoint (same mapping
    // MapPieceToByteLevel applies during encoding; DecodeToken inverts it).
    Result := '';
    for Cnt := 1 to Length(Fragment) do
      Result := Result + CodePointToUTF8(FByteToCP[Ord(Fragment[Cnt])]);
  end
  else if FMetaspacePreTok then
    // Metaspace: spaces are stored as the replacement char (usually U+2581).
    Result := StringReplace(Fragment, ' ', FMSReplacement, [rfReplaceAll])
  else
    // WordPiece / plain: vocab surfaces ARE the raw text.
    Result := Fragment;
end;

function TNeuralHFTokenizer.PrefixScanVocab(
  const Fragment: string): TNeuralIntegerArray;
var
  Surface, Cand: string;
  SurfLen, Id, Count: integer;
begin
  SetLength(Result, 0);
  if Fragment = '' then exit;
  Surface := FragmentToSurface(Fragment);
  SurfLen := Length(Surface);
  if SurfLen = 0 then exit;
  SetLength(Result, Length(FIdToToken));
  Count := 0;
  for Id := 0 to High(FIdToToken) do
  begin
    Cand := FIdToToken[Id];
    if (Cand <> '') and (Length(Cand) >= SurfLen) and
       (Copy(Cand, 1, SurfLen) = Surface) then
    begin
      Result[Count] := Id;
      Inc(Count);
    end;
  end;
  SetLength(Result, Count);
end;

function TNeuralHFTokenizer.GetVocabSize(): integer;
begin
  Result := Length(FIdToToken);
end;

function TNeuralHFTokenizer.TokenToId(const Token: string): integer;
var
  TokenIndex: integer;
begin
  Result := VocabFind(Token);
  if Result < 0 then
    for TokenIndex := 0 to High(FAddedTokens) do
      if FAddedTokens[TokenIndex].Content = Token then
        Exit(FAddedTokens[TokenIndex].Id);
end;

function TNeuralHFTokenizer.IdToToken(Id: integer): string;
var
  TokenIndex: integer;
begin
  if (Id >= 0) and (Id <= High(FIdToToken)) and (FIdToToken[Id] <> '')
  then Result := FIdToToken[Id]
  else if IsAddedTokenId(Id, TokenIndex)
  then Result := FAddedTokens[TokenIndex].Content
  else Result := '';
end;

function TNeuralHFTokenizer.AddedTokenCount(): integer;
begin
  Result := Length(FAddedTokens);
end;

function TNeuralHFTokenizer.GetAddedToken(Index: integer): TNeuralAddedToken;
begin
  Result := FAddedTokens[Index];
end;

end.
