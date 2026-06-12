unit neuraldecode;

(*
neuraldecode
Deterministic, sequence-level decoding strategies for char-level next-token
models built with neural-api. The flagship routine is DecodeBeamSearch, the
missing counterpart to the per-token stochastic TNNetSamplerBase family
(Greedy / TopK / TopP).

Beam search keeps the B highest CUMULATIVE-log-probability partial sequences,
expands each by every candidate next token, then re-prunes to the top B. Unlike
a per-token argmax, it can RECOVER from a locally-greedy first-token mistake
that a single argmax would lock in forever. Because it scores whole sequences
rather than one token, it does NOT fit the GetToken(Origin) interface and is a
standalone routine rather than a TNNetSamplerBeam subclass.

Honest v1 scope notes ("what did NOT fit"):
  (a) All scoring is in LOG space and log-probs are SUMMED (never multiply raw
      probabilities -> underflow). Model softmax outputs are converted to
      log-probs with a numerically-safe log of a clamped probability.
  (b) Wu et al. 2016 length penalty:  score = sum_logp / ((5+L)/6)^alpha .
      alpha = 0 reproduces the raw, short-sequence-biased sum; alpha > 0 lifts
      longer beams. See LengthPenaltyDenominator.
  (c) v1 RE-ENCODES each candidate prefix on every step (O(L^2) total forward
      passes). The KV-cache incremental-decode plumbing now lives in
      TNNetStreamingDecoder below, and GenerateTokensStreamed /
      GenerateStringStreamed are the streamed (never-re-encode) greedy/sampled
      generation routines built on it; wiring BEAM search onto the session is
      the remaining follow-up.
  (d) A beam that emits the EOS token is moved to a finished pool and ranked
      there against still-growing beams; growth stops once enough finished
      beams dominate or MaxLen is reached.

The forward-pass / encoding convention matches GenerateStringFromChars in
neuraldatasets: the prompt+generated text is one-hot encoded right-aligned via
OneHotEncodingReversed, NN.Compute produces a single SoftMax distribution over
the vocabulary, and EOS is token 1 (chr(1)), terminating like the
"NextTokenInt < 2" rule used elsewhere in the codebase.

MODERN SAMPLING CONTROLS. The high-level generation routines also expose:
  - Repetition / frequency / presence penalties via an optional
    TNNetTokenHistoryPenalty (neuralvolume). The streamed routines see the
    model's POST-SOFTMAX probabilities, so they call ApplyToProbabilities
    (the probability-domain image of the CTRL logit rule: p := p^r, then
    p := p * exp(-alpha_f*count - alpha_p), then renormalize) before the
    sampler runs; the routine resets the history and registers the prompt
    tokens, then registers every emitted token.
  - Min-p sampling via TNNetSamplerMinP (neuralvolume): keep tokens with
    p >= MinP * max(p), renormalize, weighted draw. It is a plain
    TNNetSamplerBase, so it plugs into the existing Sampler arguments.
  - Stop sequences: GenerateTokensStreamed accepts token-id stop sequences
    (TNNetTokenSequences) matched against the tail of the GENERATED region;
    on a match generation terminates and the stop tokens are trimmed from
    the returned length. GenerateStringStreamed and DecodeGreedy accept stop
    STRINGS (tokenized for early termination in the streamed wrapper, plus a
    string-level scan that trims even when token boundaries differ).
All of these default to "off", leaving the original behavior bit-for-bit
unchanged.

CONSTRAINED (STRUCTURED) DECODING. The generation routines also accept an
optional TNNetTokenConstraint, a caller-supplied "allowed next tokens" hook
applied AFTER the penalties and BEFORE the sampler:
  - The loop calls Reset(PromptTokens) once, then per step MaskAllowed(Probs)
    on the post-softmax probability row (zero every disallowed token, then
    renormalize the survivors to sum 1), and Commit(Token) after each emitted
    token so stateful constraints advance.
  - FALLBACK: when the allowed probability mass of a row is zero (every token
    disallowed, or every allowed token has zero probability), MaskAllowed
    leaves the row UNTOUCHED - generation degrades to unconstrained for that
    step rather than producing an all-zero / NaN row.
  - Constraint = nil is bit-for-bit the unconstrained behavior.
Ready-made constraints:
  - TNNetAllowedTokensConstraint: a static token-id whitelist.
  - TNNetForcedSequenceConstraint: forces generation down one of N candidate
    token sequences (a trie over candidate continuations) - multiple-choice
    answering; once a candidate is fully emitted only special/EOS tokens
    (< 2) are allowed.
  - TNNetJSONConstraint: JSON-mode generation. A character-level JSON
    pushdown automaton (TNNetJSONStateMachine, public for direct use) tracks
    the brace/bracket stack and the object/array/string/number/literal
    context; a token is allowed exactly when feeding its characters through a
    clone of the automaton accepts all of them, so multi-character (BPE)
    tokens are validated transitively and NO INVALID JSON can ever be
    emitted. Special/EOS tokens (< 2) are allowed only once a complete
    top-level value has been emitted. Documented deviations from full JSON:
    the grammar is STRICTER in places (no leading '+' or leading zeros in
    numbers - per spec; raw control characters < #32 are rejected inside
    strings; \u must be followed by exactly 4 hex digits) and it does not
    check UTF-8 multi-byte well-formedness (any byte >= #32 is accepted as
    string content) nor surrogate pairing of \u escapes. Trailing whitespace
    after the top-level value is accepted. Intended for char-level or BPE
    (no-separator) vocabularies: token strings are validated as-is, with no
    separator inserted between tokens.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralnetwork;

type
  // A scored decode candidate: the generated text (excluding the prompt) and
  // its cumulative log-probability.
  TNNetDecodeResult = record
    Text: string;        // generated continuation (prompt NOT included)
    SumLogProb: TNeuralFloat; // sum of per-step log-probabilities
    Score: TNeuralFloat; // length-penalised score actually ranked on
    Finished: boolean;   // True if the beam emitted EOS
  end;

  TNNetDecodeResultArray = array of TNNetDecodeResult;

  // A scored TOKEN-ID decode candidate - the seq2seq (encoder-decoder) beam
  // counterpart of the char-level TNNetDecodeResult above. Tokens holds the
  // GENERATED target ids only (StartTokenId excluded; the EOS token IS
  // included when the beam finished by emitting it, mirroring
  // DecodeSeq2SeqGreedy's "EOS is appended" convention - so a finished
  // beam's Tokens compares token-for-token with a greedy result).
  // Coded by Claude (AI).
  TNNetTokenDecodeResult = record
    Tokens: TNeuralIntegerArray; // generated ids (StartTokenId NOT included)
    SumLogProb: TNeuralFloat;    // sum of per-step log-probabilities
    Score: TNeuralFloat;         // length-penalised score actually ranked on
    Finished: boolean;           // True if the beam emitted EOSTokenId
  end;

  TNNetTokenDecodeResultArray = array of TNNetTokenDecodeResult;

  // A list of token-id stop sequences for GenerateTokensStreamed: generation
  // terminates (and the matched tokens are trimmed) as soon as the tail of
  // the GENERATED region equals one of the entries.
  TNNetTokenSequences = array of TNeuralIntegerArray;

  { TNNetTokenConstraint }
  // Abstract CONSTRAINED-DECODING hook: a caller-supplied "allowed next
  // tokens" filter the generation loop applies to the POST-SOFTMAX
  // probability row AFTER the penalties and BEFORE the sampler. Subclasses
  // implement TokenAllowed (and optionally Reset/Commit for stateful
  // grammars); the base MaskAllowed zeroes every disallowed token and
  // renormalizes the survivors to sum 1.
  // FALLBACK POLICY: when the allowed probability mass is zero (every token
  // disallowed, or every allowed token already at probability zero), the row
  // is left UNTOUCHED - the step degrades to unconstrained sampling instead
  // of producing an all-zero row (whose argmax would degenerate to token 0).
  // Coded by Claude (AI).
  TNNetTokenConstraint = class(TObject)
    public
      // Called once at the start of generation with the prompt token ids so
      // stateful constraints can start a fresh sequence. Default: no-op.
      procedure Reset(const PromptTokens: array of integer); virtual;
      // True when TokenId may be emitted next. Token ids < 2 are the
      // codebase's special/EOS ids; each constraint decides when they are
      // legal (e.g. the JSON constraint allows them only once a complete
      // top-level value has been emitted).
      function TokenAllowed(TokenId: integer): boolean; virtual; abstract;
      // Zeroes the probability of every disallowed token in the row and
      // renormalizes the allowed ones to sum 1 (see the fallback policy
      // above). Element index = token id, as everywhere in the decoder.
      procedure MaskAllowed(Probs: TNNetVolume); virtual;
      // Advances the constraint state after a token was emitted. Default:
      // no-op.
      procedure Commit(TokenId: integer); virtual;
  end;

  { TNNetAllowedTokensConstraint }
  // Static whitelist constraint: only the token ids passed to Create are ever
  // allowed (include the EOS id yourself if generation should be able to
  // stop before the length caps). Stateless - Reset/Commit are no-ops.
  // Coded by Claude (AI).
  TNNetAllowedTokensConstraint = class(TNNetTokenConstraint)
    private
      FAllowed: array of boolean;
    public
      constructor Create(const AllowedTokens: array of integer);
      function TokenAllowed(TokenId: integer): boolean; override;
  end;

  { TNNetForcedSequenceConstraint }
  // Forces generation to follow ONE of N candidate token sequences (a trie
  // over the candidate continuations) - the standard multiple-choice
  // answering constraint. At depth D a token is allowed iff some still-active
  // candidate has it at position D; Commit deactivates the candidates that
  // did not match the emitted token. Once an active candidate has been fully
  // emitted (Completed = True) the special/EOS ids (< 2) become allowed -
  // longer candidates sharing the emitted prefix may still be continued.
  // Build from token sequences directly or from strings via a Dict
  // (Dict.Tokenize per candidate). Reset rewinds to the trie root.
  // Coded by Claude (AI).
  TNNetForcedSequenceConstraint = class(TNNetTokenConstraint)
    private
      FCandidates: TNNetTokenSequences;
      FActive: array of boolean;
      FDepth: integer;
    public
      constructor Create(const Candidates: TNNetTokenSequences); overload;
      constructor Create(Dict: TStringListInt;
        const Candidates: array of string); overload;
      procedure Reset(const PromptTokens: array of integer); override;
      function TokenAllowed(TokenId: integer): boolean; override;
      procedure Commit(TokenId: integer); override;
      // True when some still-active candidate has been emitted in full.
      function Completed(): boolean;
  end;

  // Main states of the character-level JSON automaton (public so tests and
  // diagnostics can inspect where the machine is).
  TNNetJSONMainState = (
    jmsValue,             // expecting a value (top level, after ':' or after ',' in an array)
    jmsValueOrArrayClose, // right after '[': a value or an immediate ']'
    jmsObjectKeyOrClose,  // right after '{': a key string or an immediate '}'
    jmsObjectKey,         // after ',' in an object: a key string only
    jmsObjectColon,       // after a key string: ':' only
    jmsString,            // inside a string (key or value)
    jmsStringEscape,      // right after '\' inside a string
    jmsStringUnicode,     // inside the 4 hex digits of a \u escape
    jmsNumber,            // inside a number (see TNNetJSONNumberState)
    jmsLiteral,           // inside true/false/null
    jmsAfterValue,        // a value completed inside an object/array: ',' or the matching close
    jmsDone);             // a complete top-level value was consumed: whitespace only

  // Sub-states of the JSON number grammar [-] int frac? exp? . The number is
  // a complete value in jnsZero/jnsInt/jnsFrac/jnsExpDigits; a delimiter
  // (whitespace, ',', '}', ']') arriving in those states first completes the
  // number, then is processed in the successor state.
  TNNetJSONNumberState = (jnsMinus, jnsZero, jnsInt, jnsDot, jnsFrac,
    jnsExp, jnsExpSign, jnsExpDigits);

  { TNNetJSONStateMachine }
  // Character-level JSON grammar automaton: a pushdown automaton over the
  // brace/bracket stack with states for the object/array/string/number/
  // literal contexts. FeedChar advances by one character and returns False
  // when the character is not a legal continuation of valid JSON (the state
  // is undefined after a rejected feed - probe with CharAllowed or a copy
  // when unsure). It accepts any single JSON value at top level. Deviations
  // from the full spec are listed in the unit header (stricter in places,
  // never accepting invalid JSON).
  // Coded by Claude (AI).
  TNNetJSONStateMachine = class(TObject)
    private
      FState: TNNetJSONMainState;
      FNumState: TNNetJSONNumberState;
      FHexRemain: integer;     // hex digits left in a \u escape
      FLitRemain: string;      // characters left in true/false/null
      FKeyString: boolean;     // is the current string an object key?
      FStack: array of char;   // '{' / '[' nesting stack
      FStackLen: integer;
      procedure Push(C: char);
      procedure Pop();
      function Top(): char;
      // A value just completed: jmsDone at top level, jmsAfterValue inside
      // an object/array.
      procedure CompleteValue();
      function FeedNumberChar(C: char): boolean;
      // The number ended on a delimiter: complete it, then process the
      // delimiter in the successor state.
      function EndNumberAndRefeed(C: char): boolean;
    public
      constructor Create();
      procedure Reset();
      procedure CopyFrom(Source: TNNetJSONStateMachine);
      // Advances by one character; False = not a legal JSON continuation
      // (state undefined afterwards).
      function FeedChar(C: char): boolean;
      // Feeds every character of S; False at the first rejection.
      function FeedString(const S: string): boolean;
      // Non-destructive probe: would FeedChar(C) succeed from here?
      function CharAllowed(C: char): boolean;
      // True when a complete top-level JSON value has been consumed (jmsDone,
      // or an unterminated but complete top-level number such as "42").
      function IsComplete(): boolean;
      function StackDepth(): integer;
      property State: TNNetJSONMainState read FState;
  end;

  { TNNetJSONConstraint }
  // Grammar-driven JSON-mode constraint: given the tokenizer's id->string
  // mapping, allows exactly the tokens whose string is a legal continuation
  // of valid JSON from the current state. Multi-character tokens are
  // validated transitively: the automaton state is cloned and the token's
  // characters are fed one by one - all must be accepted. Special/EOS ids
  // (< 2) are allowed only when a complete top-level value has been emitted;
  // empty-string tokens are never allowed (they would not advance
  // generation). Intended for char-level or BPE (no-separator) vocabularies;
  // word-level dicts whose detokenizer inserts separators would be validated
  // without the separators. Create(Dict) snapshots Dict.DeTokenize(id) for
  // every id; CreateCharLevel(VocabSize) is the char-level model convention
  // (token id = character code, ids < 2 special).
  // Coded by Claude (AI).
  TNNetJSONConstraint = class(TNNetTokenConstraint)
    private
      FTokenStr: array of string;
      FMachine: TNNetJSONStateMachine;
      FProbe: TNNetJSONStateMachine;
    public
      constructor Create(Dict: TStringListInt); overload;
      constructor CreateCharLevel(VocabSize: integer);
      destructor Destroy(); override;
      procedure Reset(const PromptTokens: array of integer); override;
      function TokenAllowed(TokenId: integer): boolean; override;
      procedure Commit(TokenId: integer); override;
      // The live automaton (after the committed tokens) for inspection.
      property Machine: TNNetJSONStateMachine read FMachine;
  end;

  // TNNetStreamingDecoder: a reusable incremental-decode "streaming session"
  // over a causal next-token net, replacing the hand-rolled step-net plumbing
  // every streaming example repeats (build a short-width twin, CopyWeights,
  // scan layers by class, switch caches/state into incremental mode, set RoPE
  // offsets before every forward).
  //
  // OWNERSHIP. The session does NOT build and does NOT own the net. The
  // caller typically builds the SAME architecture at a short input width
  // (1 for plain token-at-a-time decode, K+1 for a speculative verify
  // window), calls ShortNet.CopyWeights(TrainedNet) - every parameter shape
  // in a streamable model is sequence-length independent, so the layer-by-
  // layer copy is exact - and hands the twin to Create. Destroy switches the
  // collected layers back out of incremental mode but never frees the net.
  //
  // WHAT IS COLLECTED. Create scans pNet.Layers once and collects
  //   - every TNNetScaledDotProductAttention: BeginIncrementalDecode(
  //     pMaxCacheLen) switches it onto the KV-cache path (pMaxCacheLen must
  //     cover the worst transient load: committed context + one whole
  //     window);
  //   - every TNNetDiagonalSSM: BeginIncrementalDecode() switches it onto the
  //     O(1)-per-step persisted-state path (no preallocation budget - the
  //     entire past is one Depth-long state vector h);
  //   - every TNNetRotaryEmbedding: kept so PositionOffset can be advanced
  //     before each forward (below).
  // A net may contain any mix (attention-only, SSM-only, hybrid); the counts
  // are exposed for diagnostics.
  //
  // RoPE EXACTNESS CONTRACT. A streamed window has SizeX = window width, so a
  // rope layer would otherwise always rotate it starting at position 0.
  // StepForward therefore sets PositionOffset := AbsPos on EVERY collected
  // rope layer before EVERY pNet.Compute, where AbsPos is the ABSOLUTE
  // position of the FIRST token in the window (the running committed length).
  // This single rule makes width-1 decode steps and width-K speculative
  // verify windows rotate exactly as the full forward would; skip it and the
  // cached path silently diverges from the full forward.
  //
  // WHICH NORM LAYERS ARE STREAMABLE. TNNetDyT is per-element (tanh of a
  // scaled activation, no cross-token statistics), so cached/streamed decode
  // is exact. TNNetLayerNorm and TNNetRMSNorm normalize over the WHOLE sample
  // INCLUDING the sequence axis: a width-1 window sees different statistics
  // than the same token inside a full-width forward, breaking full-vs-
  // incremental exactness. Build streaming models with TNNetDyT (e.g.
  // AddTransformerEncoderBlock(..., NormClass=TNNetDyT)).
  //
  // TYPICAL LOOP (greedy decode):
  //   Session := TNNetStreamingDecoder.Create(ShortNet, ContextLen + Width);
  //   Session.Reset();
  //   for t := 0 to PromptLen - 2 do            // prefill
  //     begin InV.FData[0] := Toks[t]; Session.StepForward(InV, t); end;
  //   while generating do
  //   begin
  //     InV.FData[0] := Toks[Pos - 1];
  //     Session.StepForward(InV, Pos - 1);
  //     Toks[Pos] := Session.Output().GetClassOnPixel(0, 0); Inc(Pos);
  //   end;
  // Speculative decoding additionally calls TruncateTo(CommittedLen) to roll
  // the KV caches back past rejected draft tokens (pad/draft K/V appended by
  // a verify window is discarded the same way). SSM state cannot be rolled
  // back (it is a folded summary, not a list), so TruncateTo only touches the
  // attention caches - speculative decoding is an attention-family feature.
  // Coded by Claude (AI).
  TNNetStreamingDecoder = class(TObject)
  private
    FNet: TNNet;
    FSDPAs: array of TNNetScaledDotProductAttention;
    FSSMs: array of TNNetDiagonalSSM;
    FRopes: array of TNNetRotaryEmbedding;
    function GetSDPACount(): integer;
    function GetSSMCount(): integer;
    function GetRopeCount(): integer;
  public
    constructor Create(pNet: TNNet; pMaxCacheLen: integer);
    destructor Destroy(); override;
    // Start a fresh sequence: ResetCache on every attention layer, ResetState
    // on every SSM layer. Call before the first prefill token of a sequence.
    procedure Reset();
    // One streamed forward of the window in InV. AbsPos is the absolute
    // position of the FIRST token in the window; it is written to every rope
    // layer's PositionOffset before pNet.Compute (see the exactness contract
    // above).
    procedure StepForward(InV: TNNetVolume; AbsPos: integer);
    // Speculative-decode rollback: TruncateCache(CommittedLen) on every
    // attention layer, discarding the K/V of rejected/pad tokens. No-op when
    // the net has no attention layers.
    procedure TruncateTo(CommittedLen: integer);
    // Convenience: the net's last layer output (e.g. the softmax row(s) of
    // the window just computed).
    function Output(): TNNetVolume;
    property Net: TNNet read FNet;
    property SDPACount: integer read GetSDPACount;
    property SSMCount: integer read GetSSMCount;
    property RopeCount: integer read GetRopeCount;
  end;

const
  csDecodeEOSToken = 1; // chr(1), the codebase end-of-sequence marker.

// Wu et al. 2016 length-penalty denominator ((5+L)/6)^alpha. With alpha=0 this
// is exactly 1.0 (no penalty -> raw sum-log-prob ranking, short-biased).
function LengthPenaltyDenominator(L: integer; Alpha: TNeuralFloat): TNeuralFloat;

// Numerically-safe natural log of a probability (clamps tiny / zero probs so a
// dead-but-not-impossible token never produces -Inf and poisons the sum).
function SafeLogProb(P: TNeuralFloat): TNeuralFloat;

// Deterministic greedy argmax decode, in the same forward-pass / encoding
// convention as DecodeBeamSearch. Returned as a single-element result so its
// SumLogProb is directly comparable to a beam result.
function DecodeGreedy(NN: TNNet; const Prompt: string;
  MaxLen: integer): TNNetDecodeResult; overload;

// DecodeGreedy with STOP STRINGS: generation additionally terminates as soon
// as the generated text ends with any entry of StopStrings; the stop string
// is trimmed from Result.Text and Result.Finished is set True (deliberate
// termination, like EOS). Empty StopStrings reproduces the plain overload
// bit-for-bit (the plain overload delegates here).
function DecodeGreedy(NN: TNNet; const Prompt: string; MaxLen: integer;
  const StopStrings: array of string): TNNetDecodeResult; overload;

// DecodeGreedy with CONSTRAINED DECODING. Constraint (may be nil) restricts
// the per-step argmax to the allowed tokens (DecodeGreedy is char-level:
// token id = character code, so build constraints accordingly, e.g.
// TNNetJSONConstraint.CreateCharLevel). Reset receives the prompt's char
// codes; Commit is called after every emitted token (EOS included). When NO
// token is allowed the step falls back to the plain unconstrained argmax
// (same policy as MaskAllowed). Constraint = nil reproduces the overload
// above bit-for-bit (which delegates here).
function DecodeGreedy(NN: TNNet; const Prompt: string; MaxLen: integer;
  const StopStrings: array of string;
  Constraint: TNNetTokenConstraint): TNNetDecodeResult; overload;

// Beam search. Keeps BeamWidth partial sequences ranked by length-penalised
// cumulative log-prob. Returns the single best (highest Score) result.
//   MaxLen        : maximum number of generated tokens (excludes the prompt).
//   BeamWidth     : B; B=1 with LengthPenalty=0 is exactly greedy argmax.
//   LengthPenalty : alpha in the Wu et al. formula.
function DecodeBeamSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResult;

// Full beam search returning the entire final ranked beam (finished + best
// surviving), so callers can inspect the runners-up. Sorted best-first.
function DecodeBeamSearchAll(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;

// ---------------------------------------------------------------------------
// STREAMED GENERATION: the KV-cache / SSM-state counterpart of neuraldatasets'
// GenerateStringFromCasualNN, driven by a TNNetStreamingDecoder session.
//
// CONTRACT (what the CALLER does, once):
//   1. Build a WIDTH-1 twin of the trained causal next-token net (same Build*
//      function at input width 1 - every parameter shape in a streamable
//      model is sequence-length independent).
//   2. Twin.CopyWeights(TrainedNet) - exact layer-by-layer copy.
//   3. Session := TNNetStreamingDecoder.Create(Twin, MaxTotalLen) (the cache
//      budget must cover the longest sequence ever generated).
// The routines below then NEVER re-encode the prefix: Reset() starts the
// sequence, the prompt is prefilled token-at-a-time (width-1 StepForward,
// each token at its absolute position), and every generated token costs ONE
// width-1 forward - O(cache) per token for attention (one query row over the
// cached K/V), O(1) per token for an SSM - instead of the full O(prefix)
// re-encode GenerateStringFromCasualNN pays per token.
//
// EXACTNESS. Per the TNNetStreamingDecoder header: streamed decode equals the
// full forward exactly when every layer is either per-token (embedding,
// pointwise convs, TNNetDyT) or a collected streamable mixer (SDPA KV cache,
// DiagonalSSM state, RoPE offsets). With that, greedy streamed generation is
// token-for-token identical to a full-re-encode greedy loop.
//
// INPUT ENCODING. The width-1 net must take RAW TOKEN IDS (a (1,1,1) input
// feeding a TNNetEmbedding) - the same csNeuralEncodingMethodInt convention
// GenerateStringFromCasualNN defaults to. One-hot front-ends are not
// supported here (v1).
//
// TOKEN-LEVEL CORE. Tokens[0..PromptLen-1] hold the prompt ids; the array is
// grown as needed and generated ids are appended in place. Per the
// established prefill-then-step idiom, tokens 0..PromptLen-2 are prefilled
// and the LAST prompt token is the first decode step's input (its output row
// predicts the first new token). Greedy argmax when Sampler is nil, otherwise
// Sampler.GetTokenOnPixel over the step's single output row (the model should
// end in a softmax when using stochastic samplers - same caveat as
// GenerateStringFromCasualNN). Generation stops after MaxNewTokens tokens,
// when the total length reaches MaxTotalLen, or when an end-of-sequence token
// is produced (the "NextTokenInt < 2" rule used across the codebase; the EOS
// token IS stored and counted, mirroring GenerateStringFromCasualNN).
// Returns the new TOTAL token count (prompt + generated, EOS included).
// The session must be width-1 (v1); an EArgumentException is raised
// otherwise, and when PromptLen < 1.
// Coded by Claude (AI).
function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase = nil): integer; overload;

// GenerateTokensStreamed with MODERN SAMPLING CONTROLS. Same contract as the
// overload above plus:
//  - Penalty (may be nil): a TNNetTokenHistoryPenalty applied to the step's
//    POST-SOFTMAX probability row via ApplyToProbabilities BEFORE the
//    sampler/argmax reads it. The routine resets the penalty history,
//    registers the prompt tokens (the whole context is penalized, the usual
//    convention), then registers every emitted token.
//  - StopSequences (may be nil/empty): token-id sequences; as soon as the
//    tail of the GENERATED region (never spanning into the prompt) equals an
//    entry, generation stops and the matched tokens are TRIMMED from the
//    returned length. When several entries match, the longest is trimmed.
// Penalty=nil with empty StopSequences is bit-for-bit the plain overload
// (which delegates here).
function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  const StopSequences: TNNetTokenSequences): integer; overload;

// GenerateTokensStreamed with CONSTRAINED DECODING. Same contract as the
// overload above plus:
//  - Constraint (may be nil): a TNNetTokenConstraint applied to the step's
//    POST-SOFTMAX probability row via MaskAllowed AFTER the penalty and
//    BEFORE the sampler/argmax reads it. The routine calls
//    Constraint.Reset(prompt tokens) once, then Constraint.Commit(token)
//    after every emitted token (EOS included) so stateful grammars advance.
//    See TNNetTokenConstraint for the all-masked fallback policy.
// Constraint = nil is bit-for-bit the overload above (which delegates here).
function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  const StopSequences: TNNetTokenSequences;
  Constraint: TNNetTokenConstraint): integer; overload;

// STRING-LEVEL WRAPPER mirroring GenerateStringFromCasualNN's shape
// (dict/tokenizer + prompt + optional sampler; TNeuralTokenizer is a
// TStringListInt subclass with virtual Tokenize/DeTokenize, so both word-dict
// and BPE-tokenizer callers pass the same parameter type). Tokenizes the
// prompt with Dict.Tokenize, runs GenerateTokensStreamed and detokenizes the
// continuation appended to InputString. For display the continuation stops at
// the first special token (< 2), and words are joined with a space only when
// Dict.TokenizerHasSeparator (word-level dicts) - byte-pair vocabularies
// concatenate directly, matching GenerateStringFromCasualNN.
// Coded by Claude (AI).
function GenerateStringStreamed(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase = nil): string; overload;

// GenerateStringStreamed with MODERN SAMPLING CONTROLS. Same contract as the
// overload above plus:
//  - Penalty (may be nil): forwarded to the token core (probability-domain
//    application; see GenerateTokensStreamed).
//  - StopStrings (may be empty): each entry is tokenized with Dict.Tokenize
//    and passed to the token core for EARLY TERMINATION; additionally the
//    detokenized continuation is scanned for the earliest occurrence of any
//    stop string and truncated there - a safety net for vocabularies where
//    the stop text can be emitted across different token boundaries. The
//    stop text never appears in the returned string.
// Penalty=nil with empty StopStrings is bit-for-bit the plain overload
// (which delegates here).
function GenerateStringStreamed(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase; Penalty: TNNetTokenHistoryPenalty;
  const StopStrings: array of string): string; overload;

// GenerateStringStreamed with CONSTRAINED DECODING: forwards Constraint (may
// be nil) to the token core (see GenerateTokensStreamed). Constraint = nil is
// bit-for-bit the overload above (which delegates here).
function GenerateStringStreamed(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase; Penalty: TNNetTokenHistoryPenalty;
  const StopStrings: array of string;
  Constraint: TNNetTokenConstraint): string; overload;

// ---------------------------------------------------------------------------
// SEQ2SEQ (ENCODER-DECODER) GENERATION: the T5/BART-style counterpart of the
// decoder-only routines above, for the two-net pairs returned by
// BuildT5FromSafeTensors / BuildMarianFromSafeTensors (neuralpretrained):
//   - the ENCODER net maps (EncSeqLen,1,1) source token ids to
//     (EncSeqLen,1,d_model) hidden states;
//   - the DECODER net has TWO TNNetInput layers: Layers[0] takes the
//     (DecSeqLen,1,1) target token ids and a SECOND TNNetInput takes the
//     encoder hidden states, read by every block's cross-attention.
// The loop encodes the source ONCE, copies the hidden states into the
// decoder's second input ONCE (they are constant across decode steps), then
// autoregresses the target from StartTokenId (decoder_start_token_id):
// each step re-runs the FULL decoder forward on the growing prefix (no KV
// cache - the decoder was built at a FIXED DecSeqLen; positions past the
// prefix are padded with StartTokenId, which causal self-attention makes
// invisible to the rows actually read) and reads the LOGITS row at the last
// prefix position. Greedy takes the argmax; the sampled variant softmaxes
// the row at Temperature and draws with the usual TNNetSamplerBase family
// (TopK / TopP / MinP / Greedy).
//
// STOPPING. Generation stops when EOSTokenId is emitted (it IS appended to
// the result, mirroring GenerateTokensStreamed's EOS handling), after
// MaxNewTokens tokens, or when the decoder's build-time capacity DecSeqLen
// is exhausted (the last generated token then used every input slot).
// The returned array holds the GENERATED ids only - StartTokenId excluded.
//
// VALIDATION. Length(SourceTokens) must equal the encoder's build-time
// EncSeqLen, the encoder output must match the decoder's second-input size,
// and the decoder must actually have a second TNNetInput - violations raise
// EArgumentException. BEAM SEARCH: the existing DecodeBeamSearch is
// char-level/string-based and does not compose with the two-net token-id
// convention, so DecodeSeq2SeqBeamSearch / DecodeSeq2SeqBeamSearchAll below
// run their own TOKEN-ID candidate loop over the same encode-once /
// per-step-full-decoder-re-forward harness.
// Coded by Claude (AI).

// Returns the decoder net's SECOND TNNetInput - the encoder-hidden-states
// input of a BuildT5FromSafeTensors / BuildMarianFromSafeTensors decoder
// (same convention as neuralpretrained's T5EncoderStatesInput, duplicated
// here so neuraldecode does not depend on the importer unit). Raises
// EArgumentException when the net has no second TNNetInput.
function Seq2SeqEncoderStatesInput(DecoderNet: TNNet): TNNetLayer;

// Deterministic greedy argmax seq2seq decode (see the section header above).
function DecodeSeq2SeqGreedy(EncoderNet, DecoderNet: TNNet;
  const SourceTokens: array of integer;
  StartTokenId, EOSTokenId, MaxNewTokens: integer): TNeuralIntegerArray;

// Stochastic seq2seq decode: the step's logits row is divided by Temperature
// (clamped to >= 1e-6; Temperature -> 0 degenerates to greedy argmax) and
// softmaxed, then Sampler draws from the resulting distribution. Sampler =
// nil is bit-for-bit DecodeSeq2SeqGreedy (which delegates here - argmax over
// logits is Temperature-invariant, so no softmax is computed on that path).
function DecodeSeq2SeqSampled(EncoderNet, DecoderNet: TNNet;
  const SourceTokens: array of integer;
  StartTokenId, EOSTokenId, MaxNewTokens: integer;
  Sampler: TNNetSamplerBase;
  Temperature: TNeuralFloat = 1.0): TNeuralIntegerArray;

// TOKEN-ID seq2seq BEAM SEARCH over the two-net convention above - the
// encoder-decoder counterpart of the char-level DecodeBeamSearch. Keeps
// BeamWidth partial target sequences ranked by length-penalised cumulative
// log-probability (Wu et al. 2016, see LengthPenaltyDenominator; the length
// is Length(Tokens), so an emitted EOS counts). Scoring is in LOG space:
// each step's logits row is converted to log-probs with a numerically-stable
// softmax + SafeLogProb, exactly the per-step distribution the greedy /
// sampled routines act on.
//
// The source is encoded ONCE (cached in the decoder's second TNNetInput,
// constant across all beams and steps); every step then re-runs the FULL
// decoder forward once PER LIVE BEAM on that beam's StartTokenId-padded
// prefix and expands it by every vocabulary token - O(BeamWidth * L) decoder
// forwards for L generated tokens.
//
// FINISHED POOL (same idiom as DecodeBeamSearchAll): a candidate that emits
// EOSTokenId moves to a finished pool (its EOS is INCLUDED in Tokens and its
// log-prob in SumLogProb) and is ranked there against still-growing beams;
// growth stops once BeamWidth finished beams all outscore every live beam,
// when MaxNewTokens tokens were generated, or when the decoder's build-time
// capacity DecSeqLen is exhausted (same caps as DecodeSeq2SeqGreedy). Any
// live beams remaining at the cap join the pool unfinished.
//
// BeamWidth = 1 with LengthPenalty = 0 follows exactly the greedy argmax
// path (log-softmax is monotone in the logits, ties resolve to the lowest
// token id like GetClassOnPixel), so it reproduces DecodeSeq2SeqGreedy
// whenever the greedy path's EOS-terminated beam outranks the shorter
// EOS-truncations collected along the way (raw sum-log-prob ranking is
// short-biased by construction - LengthPenalty exists to counter it).
//
// Validation matches DecodeSeq2SeqSampled (source length, encoder/decoder
// state-size match, second TNNetInput; violations raise EArgumentException).
// MaxNewTokens < 1 returns empty; BeamWidth < 1 is clamped to 1. Pure beam
// search - fully deterministic, no RNG anywhere.
// Coded by Claude (AI).

// Returns the single best (highest Score) beam's generated token ids -
// shaped like DecodeSeq2SeqGreedy's result (EOS included when emitted).
function DecodeSeq2SeqBeamSearch(EncoderNet, DecoderNet: TNNet;
  const SourceTokens: array of integer;
  StartTokenId, EOSTokenId, MaxNewTokens: integer;
  BeamWidth: integer; LengthPenalty: TNeuralFloat): TNeuralIntegerArray;

// Full seq2seq beam search returning the entire final ranked pool (finished
// + surviving live beams), sorted best-first by Score - the token-id
// counterpart of DecodeBeamSearchAll, so callers can inspect runners-up.
function DecodeSeq2SeqBeamSearchAll(EncoderNet, DecoderNet: TNNet;
  const SourceTokens: array of integer;
  StartTokenId, EOSTokenId, MaxNewTokens: integer;
  BeamWidth: integer; LengthPenalty: TNeuralFloat): TNNetTokenDecodeResultArray;

implementation

uses
  Math;

function LengthPenaltyDenominator(L: integer; Alpha: TNeuralFloat): TNeuralFloat;
begin
  if Alpha = 0 then
    Result := 1.0
  else
    Result := Power((5.0 + L) / 6.0, Alpha);
end;

function SafeLogProb(P: TNeuralFloat): TNeuralFloat;
const
  csTinyProb = 1e-30;
begin
  if P < csTinyProb then P := csTinyProb;
  Result := Ln(P);
end;

{ TNNetTokenConstraint }

procedure TNNetTokenConstraint.Reset(const PromptTokens: array of integer);
begin
  // Default: stateless constraint, nothing to rewind.
end;

procedure TNNetTokenConstraint.MaskAllowed(Probs: TNNetVolume);
var
  I: integer;
  AllowedMass: TNeuralFloat;
  Allowed: array of boolean;
  AnyBlocked: boolean;
begin
  SetLength(Allowed, Probs.Size);
  AllowedMass := 0;
  AnyBlocked := false;
  for I := 0 to Probs.Size - 1 do
  begin
    Allowed[I] := TokenAllowed(I);
    if Allowed[I]
    then AllowedMass := AllowedMass + Probs.Raw[I]
    else AnyBlocked := true;
  end;
  // Nothing to mask: every token is allowed.
  if not AnyBlocked then exit;
  // FALLBACK (documented in the class header): zero allowed mass - leave the
  // row untouched so the step degrades to unconstrained sampling instead of
  // an all-zero row whose argmax would degenerate to token 0.
  if AllowedMass <= 0 then exit;
  for I := 0 to Probs.Size - 1 do
    if Allowed[I]
    then Probs.Raw[I] := Probs.Raw[I] / AllowedMass
    else Probs.Raw[I] := 0;
end;

procedure TNNetTokenConstraint.Commit(TokenId: integer);
begin
  // Default: stateless constraint, nothing to advance.
end;

{ TNNetAllowedTokensConstraint }

constructor TNNetAllowedTokensConstraint.Create(
  const AllowedTokens: array of integer);
var
  I, MaxId: integer;
begin
  inherited Create();
  MaxId := -1;
  for I := 0 to High(AllowedTokens) do
    if AllowedTokens[I] > MaxId then MaxId := AllowedTokens[I];
  SetLength(FAllowed, MaxId + 1);
  for I := 0 to MaxId do FAllowed[I] := false;
  for I := 0 to High(AllowedTokens) do
    if AllowedTokens[I] >= 0 then FAllowed[AllowedTokens[I]] := true;
end;

function TNNetAllowedTokensConstraint.TokenAllowed(TokenId: integer): boolean;
begin
  Result := (TokenId >= 0) and (TokenId <= High(FAllowed)) and
    FAllowed[TokenId];
end;

{ TNNetForcedSequenceConstraint }

constructor TNNetForcedSequenceConstraint.Create(
  const Candidates: TNNetTokenSequences);
var
  I: integer;
begin
  inherited Create();
  SetLength(FCandidates, 0);
  // Keep non-empty candidates only (an empty candidate would mean "emit
  // nothing", which cannot guide a generation step).
  for I := 0 to High(Candidates) do
  begin
    if Length(Candidates[I]) = 0 then continue;
    SetLength(FCandidates, Length(FCandidates) + 1);
    FCandidates[High(FCandidates)] :=
      Copy(Candidates[I], 0, Length(Candidates[I]));
  end;
  SetLength(FActive, Length(FCandidates));
  Reset([]);
end;

constructor TNNetForcedSequenceConstraint.Create(Dict: TStringListInt;
  const Candidates: array of string);
var
  Seqs: TNNetTokenSequences;
  Toks: TNeuralIntegerArray;
  I: integer;
begin
  SetLength(Seqs, 0);
  for I := 0 to High(Candidates) do
  begin
    if Candidates[I] = '' then continue;
    Dict.Tokenize(Candidates[I], Toks);
    if Length(Toks) = 0 then continue;
    SetLength(Seqs, Length(Seqs) + 1);
    Seqs[High(Seqs)] := Copy(Toks, 0, Length(Toks));
  end;
  Create(Seqs);
end;

procedure TNNetForcedSequenceConstraint.Reset(
  const PromptTokens: array of integer);
var
  I: integer;
begin
  // The candidates constrain the GENERATED region only; the prompt is just
  // conditioning context, so it is ignored and the trie rewinds to its root.
  FDepth := 0;
  for I := 0 to High(FActive) do FActive[I] := true;
end;

function TNNetForcedSequenceConstraint.TokenAllowed(TokenId: integer): boolean;
var
  I: integer;
begin
  // Special/EOS ids become legal once some candidate has been fully emitted.
  if TokenId < 2 then exit(Completed());
  Result := false;
  for I := 0 to High(FCandidates) do
    if FActive[I] and (FDepth < Length(FCandidates[I])) and
      (FCandidates[I][FDepth] = TokenId) then exit(true);
end;

procedure TNNetForcedSequenceConstraint.Commit(TokenId: integer);
var
  I: integer;
begin
  // Special/EOS ids terminate generation without consuming trie depth, so
  // Completed() remains queryable after the final EOS commit.
  if TokenId < 2 then exit;
  for I := 0 to High(FCandidates) do
    if FActive[I] then
      FActive[I] := (FDepth < Length(FCandidates[I])) and
        (FCandidates[I][FDepth] = TokenId);
  Inc(FDepth);
end;

function TNNetForcedSequenceConstraint.Completed(): boolean;
var
  I: integer;
begin
  Result := false;
  for I := 0 to High(FCandidates) do
    if FActive[I] and (Length(FCandidates[I]) = FDepth) then exit(true);
end;

{ TNNetJSONStateMachine }

// JSON insignificant whitespace (RFC 8259: space, tab, LF, CR).
function JSONIsWS(C: char): boolean;
begin
  Result := (C = ' ') or (C = #9) or (C = #10) or (C = #13);
end;

function JSONIsDigit(C: char): boolean;
begin
  Result := (C >= '0') and (C <= '9');
end;

function JSONIsHexDigit(C: char): boolean;
begin
  Result := JSONIsDigit(C) or ((C >= 'a') and (C <= 'f')) or
    ((C >= 'A') and (C <= 'F'));
end;

constructor TNNetJSONStateMachine.Create();
begin
  inherited Create();
  SetLength(FStack, 8);
  Reset();
end;

procedure TNNetJSONStateMachine.Reset();
begin
  FState := jmsValue;
  FNumState := jnsMinus;
  FHexRemain := 0;
  FLitRemain := '';
  FKeyString := false;
  FStackLen := 0;
end;

procedure TNNetJSONStateMachine.CopyFrom(Source: TNNetJSONStateMachine);
begin
  FState := Source.FState;
  FNumState := Source.FNumState;
  FHexRemain := Source.FHexRemain;
  FLitRemain := Source.FLitRemain;
  FKeyString := Source.FKeyString;
  if Length(FStack) < Source.FStackLen then
    SetLength(FStack, Source.FStackLen);
  if Source.FStackLen > 0 then
    Move(Source.FStack[0], FStack[0], Source.FStackLen * SizeOf(char));
  FStackLen := Source.FStackLen;
end;

procedure TNNetJSONStateMachine.Push(C: char);
begin
  if FStackLen >= Length(FStack) then SetLength(FStack, FStackLen * 2 + 8);
  FStack[FStackLen] := C;
  Inc(FStackLen);
end;

procedure TNNetJSONStateMachine.Pop();
begin
  Dec(FStackLen);
end;

function TNNetJSONStateMachine.Top(): char;
begin
  if FStackLen > 0
  then Result := FStack[FStackLen - 1]
  else Result := #0;
end;

procedure TNNetJSONStateMachine.CompleteValue();
begin
  if FStackLen = 0
  then FState := jmsDone
  else FState := jmsAfterValue;
end;

function TNNetJSONStateMachine.EndNumberAndRefeed(C: char): boolean;
begin
  // Only called from number states where the number is a complete value.
  CompleteValue();
  Result := FeedChar(C);
end;

function TNNetJSONStateMachine.FeedNumberChar(C: char): boolean;
begin
  Result := true;
  case FNumState of
    jnsMinus: // '-' consumed: an int part MUST follow ('-' alone is invalid)
      if C = '0' then FNumState := jnsZero
      else if (C >= '1') and (C <= '9') then FNumState := jnsInt
      else Result := false;
    jnsZero:  // a leading 0 admits no further int digits (per spec)
      if C = '.' then FNumState := jnsDot
      else if (C = 'e') or (C = 'E') then FNumState := jnsExp
      else Result := EndNumberAndRefeed(C);
    jnsInt:
      if JSONIsDigit(C) then // stay
      else if C = '.' then FNumState := jnsDot
      else if (C = 'e') or (C = 'E') then FNumState := jnsExp
      else Result := EndNumberAndRefeed(C);
    jnsDot:   // '.' consumed: at least one fraction digit required
      if JSONIsDigit(C) then FNumState := jnsFrac
      else Result := false;
    jnsFrac:
      if JSONIsDigit(C) then // stay
      else if (C = 'e') or (C = 'E') then FNumState := jnsExp
      else Result := EndNumberAndRefeed(C);
    jnsExp:   // 'e'/'E' consumed: optional sign, then at least one digit
      if JSONIsDigit(C) then FNumState := jnsExpDigits
      else if (C = '+') or (C = '-') then FNumState := jnsExpSign
      else Result := false;
    jnsExpSign:
      if JSONIsDigit(C) then FNumState := jnsExpDigits
      else Result := false;
    jnsExpDigits:
      if JSONIsDigit(C) then // stay
      else Result := EndNumberAndRefeed(C);
  end;
end;

function TNNetJSONStateMachine.FeedChar(C: char): boolean;
begin
  Result := true;
  case FState of
    jmsValue, jmsValueOrArrayClose:
      begin
        if JSONIsWS(C) then exit;
        case C of
          '{': begin Push('{'); FState := jmsObjectKeyOrClose; end;
          '[': begin Push('['); FState := jmsValueOrArrayClose; end;
          '"': begin FKeyString := false; FState := jmsString; end;
          't': begin FLitRemain := 'rue'; FState := jmsLiteral; end;
          'f': begin FLitRemain := 'alse'; FState := jmsLiteral; end;
          'n': begin FLitRemain := 'ull'; FState := jmsLiteral; end;
          '-': begin FNumState := jnsMinus; FState := jmsNumber; end;
          '0': begin FNumState := jnsZero; FState := jmsNumber; end;
          '1'..'9': begin FNumState := jnsInt; FState := jmsNumber; end;
          ']': // legal only right after '[' (empty array; no trailing comma)
            if FState = jmsValueOrArrayClose then
            begin Pop(); CompleteValue(); end
            else Result := false;
          else Result := false;
        end;
      end;
    jmsObjectKeyOrClose:
      begin
        if JSONIsWS(C) then exit;
        if C = '"' then begin FKeyString := true; FState := jmsString; end
        else if C = '}' then begin Pop(); CompleteValue(); end
        else Result := false;
      end;
    jmsObjectKey: // after ',' in an object: a key string is mandatory
      begin
        if JSONIsWS(C) then exit;
        if C = '"' then begin FKeyString := true; FState := jmsString; end
        else Result := false;
      end;
    jmsObjectColon:
      begin
        if JSONIsWS(C) then exit;
        if C = ':' then FState := jmsValue
        else Result := false;
      end;
    jmsString:
      begin
        if C = '"' then
        begin
          if FKeyString
          then FState := jmsObjectColon
          else CompleteValue();
        end
        else if C = '\' then FState := jmsStringEscape
        // Raw control characters are invalid inside JSON strings; any other
        // byte (including >#127 UTF-8 continuation bytes) is string content.
        else if C < #32 then Result := false;
      end;
    jmsStringEscape:
      case C of
        '"', '\', '/', 'b', 'f', 'n', 'r', 't': FState := jmsString;
        'u': begin FHexRemain := 4; FState := jmsStringUnicode; end;
        else Result := false;
      end;
    jmsStringUnicode:
      if JSONIsHexDigit(C) then
      begin
        Dec(FHexRemain);
        if FHexRemain = 0 then FState := jmsString;
      end
      else Result := false;
    jmsNumber:
      Result := FeedNumberChar(C);
    jmsLiteral:
      if (FLitRemain <> '') and (C = FLitRemain[1]) then
      begin
        Delete(FLitRemain, 1, 1);
        if FLitRemain = '' then CompleteValue();
      end
      else Result := false;
    jmsAfterValue: // stack is never empty here (empty stack -> jmsDone)
      begin
        if JSONIsWS(C) then exit;
        if C = ',' then
        begin
          if Top() = '{'
          then FState := jmsObjectKey
          else FState := jmsValue;
        end
        else if (C = '}') and (Top() = '{') then
        begin Pop(); CompleteValue(); end
        else if (C = ']') and (Top() = '[') then
        begin Pop(); CompleteValue(); end
        else Result := false;
      end;
    jmsDone: // trailing whitespace only after the top-level value
      Result := JSONIsWS(C);
  end;
end;

function TNNetJSONStateMachine.FeedString(const S: string): boolean;
var
  I: integer;
begin
  Result := true;
  for I := 1 to Length(S) do
    if not FeedChar(S[I]) then exit(false);
end;

function TNNetJSONStateMachine.CharAllowed(C: char): boolean;
var
  SavedState: TNNetJSONMainState;
  SavedNumState: TNNetJSONNumberState;
  SavedHexRemain, SavedStackLen: integer;
  SavedLitRemain: string;
  SavedKeyString: boolean;
begin
  // One FeedChar performs at most one push (appended above SavedStackLen,
  // discarded by restoring the length) or one pop (the popped element is
  // left intact below SavedStackLen), so saving the scalar fields plus the
  // stack LENGTH restores the exact state.
  SavedState := FState;
  SavedNumState := FNumState;
  SavedHexRemain := FHexRemain;
  SavedLitRemain := FLitRemain;
  SavedKeyString := FKeyString;
  SavedStackLen := FStackLen;
  Result := FeedChar(C);
  FState := SavedState;
  FNumState := SavedNumState;
  FHexRemain := SavedHexRemain;
  FLitRemain := SavedLitRemain;
  FKeyString := SavedKeyString;
  FStackLen := SavedStackLen;
end;

function TNNetJSONStateMachine.IsComplete(): boolean;
begin
  Result := (FState = jmsDone) or
    // An unterminated top-level number that is already a complete value
    // (e.g. "42", "-1.5e3"): valid JSON if generation stops here.
    ((FState = jmsNumber) and (FStackLen = 0) and
     (FNumState in [jnsZero, jnsInt, jnsFrac, jnsExpDigits]));
end;

function TNNetJSONStateMachine.StackDepth(): integer;
begin
  Result := FStackLen;
end;

{ TNNetJSONConstraint }

constructor TNNetJSONConstraint.Create(Dict: TStringListInt);
var
  I: integer;
begin
  inherited Create();
  FMachine := TNNetJSONStateMachine.Create();
  FProbe := TNNetJSONStateMachine.Create();
  SetLength(FTokenStr, Dict.GetVocabCount());
  for I := 0 to High(FTokenStr) do
    if I < 2
    then FTokenStr[I] := '' // special ids carry no text
    else FTokenStr[I] := Dict.DeTokenize(I);
end;

constructor TNNetJSONConstraint.CreateCharLevel(VocabSize: integer);
var
  I: integer;
begin
  inherited Create();
  FMachine := TNNetJSONStateMachine.Create();
  FProbe := TNNetJSONStateMachine.Create();
  SetLength(FTokenStr, VocabSize);
  for I := 0 to High(FTokenStr) do
    if I < 2
    then FTokenStr[I] := ''
    else FTokenStr[I] := Chr(I);
end;

destructor TNNetJSONConstraint.Destroy();
begin
  FProbe.Free;
  FMachine.Free;
  inherited Destroy();
end;

procedure TNNetJSONConstraint.Reset(const PromptTokens: array of integer);
begin
  // The JSON value starts at the generation boundary; the prompt is plain
  // conditioning text and is NOT fed through the automaton.
  FMachine.Reset();
end;

function TNNetJSONConstraint.TokenAllowed(TokenId: integer): boolean;
var
  S: string;
  I: integer;
begin
  if (TokenId < 0) or (TokenId > High(FTokenStr)) then exit(false);
  // Special/EOS ids: legal exactly when a complete top-level value stands.
  if TokenId < 2 then exit(FMachine.IsComplete());
  S := FTokenStr[TokenId];
  if S = '' then exit(false); // would not advance generation
  // Transitive multi-character validation: clone the live state and feed the
  // token's characters one by one; ALL must be legal continuations.
  FProbe.CopyFrom(FMachine);
  for I := 1 to Length(S) do
    if not FProbe.FeedChar(S[I]) then exit(false);
  Result := true;
end;

procedure TNNetJSONConstraint.Commit(TokenId: integer);
begin
  if (TokenId < 2) or (TokenId > High(FTokenStr)) then exit;
  // Tokens reaching Commit were validated by TokenAllowed, so this never
  // rejects in the generation loop; the result is intentionally ignored so
  // direct/driving callers cannot corrupt the automaton silently either way.
  FMachine.FeedString(FTokenStr[TokenId]);
end;

{ TNNetStreamingDecoder }

constructor TNNetStreamingDecoder.Create(pNet: TNNet; pMaxCacheLen: integer);
var
  i, n: integer;
  Layer: TNNetLayer;
begin
  inherited Create();
  FNet := pNet;
  SetLength(FSDPAs, 0);
  SetLength(FSSMs, 0);
  SetLength(FRopes, 0);
  // One class-based scan collects every streamable state holder; the scan is
  // builder-agnostic, so any mix of attention/SSM/hybrid models works.
  for i := 0 to FNet.Layers.Count - 1 do
  begin
    Layer := FNet.Layers[i];
    if Layer is TNNetScaledDotProductAttention then
    begin
      n := Length(FSDPAs);
      SetLength(FSDPAs, n + 1);
      FSDPAs[n] := TNNetScaledDotProductAttention(Layer);
      FSDPAs[n].BeginIncrementalDecode(pMaxCacheLen);
    end;
    if Layer is TNNetDiagonalSSM then
    begin
      n := Length(FSSMs);
      SetLength(FSSMs, n + 1);
      FSSMs[n] := TNNetDiagonalSSM(Layer);
      FSSMs[n].BeginIncrementalDecode();
    end;
    if Layer is TNNetRotaryEmbedding then
    begin
      n := Length(FRopes);
      SetLength(FRopes, n + 1);
      FRopes[n] := TNNetRotaryEmbedding(Layer);
    end;
  end;
end;

destructor TNNetStreamingDecoder.Destroy();
var
  i: integer;
begin
  // Switch the collected layers back onto the normal full-sequence path and
  // restore the default rope offset; the net itself is NOT owned/freed.
  for i := 0 to High(FSDPAs) do FSDPAs[i].EndIncrementalDecode();
  for i := 0 to High(FSSMs) do FSSMs[i].EndIncrementalDecode();
  for i := 0 to High(FRopes) do FRopes[i].PositionOffset := 0;
  SetLength(FSDPAs, 0);
  SetLength(FSSMs, 0);
  SetLength(FRopes, 0);
  inherited Destroy();
end;

procedure TNNetStreamingDecoder.Reset();
var
  i: integer;
begin
  for i := 0 to High(FSDPAs) do FSDPAs[i].ResetCache();
  for i := 0 to High(FSSMs) do FSSMs[i].ResetState();
end;

procedure TNNetStreamingDecoder.StepForward(InV: TNNetVolume; AbsPos: integer);
var
  i: integer;
begin
  // The exactness contract: every rope layer is shifted to the window's
  // ABSOLUTE start position before every forward, so a width-1 step and a
  // width-K speculative verify window both rotate exactly like the full
  // forward.
  for i := 0 to High(FRopes) do FRopes[i].PositionOffset := AbsPos;
  FNet.Compute(InV);
end;

procedure TNNetStreamingDecoder.TruncateTo(CommittedLen: integer);
var
  i: integer;
begin
  for i := 0 to High(FSDPAs) do FSDPAs[i].TruncateCache(CommittedLen);
end;

function TNNetStreamingDecoder.Output(): TNNetVolume;
begin
  Result := FNet.GetLastLayer().Output;
end;

function TNNetStreamingDecoder.GetSDPACount(): integer;
begin
  Result := Length(FSDPAs);
end;

function TNNetStreamingDecoder.GetSSMCount(): integer;
begin
  Result := Length(FSSMs);
end;

function TNNetStreamingDecoder.GetRopeCount(): integer;
begin
  Result := Length(FRopes);
end;

{ Streamed generation }

function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase): integer;
begin
  // Bit-for-bit the original behavior: no penalty, no stop sequences.
  Result := GenerateTokensStreamed(Session, Tokens, PromptLen, MaxNewTokens,
    MaxTotalLen, Sampler, nil, nil);
end;

// Returns the length of the LONGEST StopSequences entry matching the tail
// Tokens[Pos-L..Pos-1], with the whole match inside the generated region
// (start index >= PromptLen, never spanning into the prompt); 0 if none.
function MatchStopSuffix(const Tokens: TNeuralIntegerArray;
  Pos, PromptLen: integer; const StopSequences: TNNetTokenSequences): integer;
var
  S, I, L: integer;
  Match: boolean;
begin
  Result := 0;
  for S := 0 to High(StopSequences) do
  begin
    L := Length(StopSequences[S]);
    if (L > Result) and (L >= 1) and (Pos - L >= PromptLen) then
    begin
      Match := true;
      for I := 0 to L - 1 do
        if Tokens[Pos - L + I] <> StopSequences[S][I] then
        begin
          Match := false;
          Break;
        end;
      if Match then Result := L;
    end;
  end;
end;

function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  const StopSequences: TNNetTokenSequences): integer;
begin
  // Bit-for-bit the unconstrained behavior.
  Result := GenerateTokensStreamed(Session, Tokens, PromptLen, MaxNewTokens,
    MaxTotalLen, Sampler, Penalty, StopSequences, nil);
end;

function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  const StopSequences: TNNetTokenSequences;
  Constraint: TNNetTokenConstraint): integer;
var
  InV: TNNetVolume;
  Pos, CapLen, NextTokenInt, StopLen: integer;
begin
  if Session.Net.GetFirstLayer().Output.SizeX <> 1 then
    raise EArgumentException.Create(
      'GenerateTokensStreamed: the session net must be a WIDTH-1 twin ' +
      '(input SizeX=1); got SizeX=' +
      IntToStr(Session.Net.GetFirstLayer().Output.SizeX) + '.');
  if PromptLen < 1 then
    raise EArgumentException.Create(
      'GenerateTokensStreamed: PromptLen must be >= 1 (the last prompt ' +
      'token is the first decode step''s input); got ' +
      IntToStr(PromptLen) + '.');
  // The hard length ceiling: prompt + MaxNewTokens, clipped by MaxTotalLen
  // (which should not exceed the session's cache budget).
  CapLen := Min(PromptLen + MaxNewTokens, MaxTotalLen);
  if Length(Tokens) < CapLen then SetLength(Tokens, CapLen);
  InV := TNNetVolume.Create(Session.Net.GetFirstLayer().Output);
  try
    InV.Fill(0);
    Session.Reset();
    // A fresh sequence: clear the penalty history and register the prompt
    // tokens, so the penalties see the WHOLE context (usual convention).
    if Assigned(Penalty) then
    begin
      Penalty.ResetHistory();
      for Pos := 0 to PromptLen - 1 do Penalty.RegisterToken(Tokens[Pos]);
    end;
    // A fresh sequence for the constraint, too: stateful grammars rewind and
    // receive the prompt ids (most constraints ignore them - the constrained
    // region is the GENERATED text).
    if Assigned(Constraint) then
      Constraint.Reset(Copy(Tokens, 0, PromptLen));
    // Prefill tokens 0..PromptLen-2 one at a time; the LAST prompt token is
    // the first decode step's input (its output row predicts the first new
    // token) - the established prefill-then-step idiom.
    for Pos := 0 to PromptLen - 2 do
    begin
      InV.FData[0] := Tokens[Pos];
      Session.StepForward(InV, Pos);
    end;
    Pos := PromptLen;
    while Pos < CapLen do
    begin
      InV.FData[0] := Tokens[Pos - 1];
      Session.StepForward(InV, Pos - 1);
      // The streamed step ends in a SoftMax, so the output row holds
      // PROBABILITIES - hence ApplyToProbabilities (p^r + exp() factors +
      // renormalize), not the logit-domain Apply. Mutating the net's output
      // volume in place is safe: the next StepForward recomputes it.
      if Assigned(Penalty) then Penalty.ApplyToProbabilities(Session.Output());
      // Constrained decoding: AFTER the penalty, BEFORE the sampler - zero
      // the disallowed tokens and renormalize (or leave the row untouched
      // when the allowed mass is zero; see TNNetTokenConstraint).
      if Assigned(Constraint) then Constraint.MaskAllowed(Session.Output());
      // The step net is width-1, so the (only) output row is pixel (0,0).
      if Assigned(Sampler)
      then NextTokenInt := Sampler.GetTokenOnPixel(Session.Output(), 0, 0)
      else NextTokenInt := Session.Output().GetClassOnPixel(0, 0);
      Tokens[Pos] := NextTokenInt;
      if Assigned(Penalty) then Penalty.RegisterToken(NextTokenInt);
      if Assigned(Constraint) then Constraint.Commit(NextTokenInt);
      Inc(Pos);
      // Stop sequences: if the GENERATED tail now equals one of the entries,
      // terminate and TRIM the matched tokens from the returned length.
      if Length(StopSequences) > 0 then
      begin
        StopLen := MatchStopSuffix(Tokens, Pos, PromptLen, StopSequences);
        if StopLen > 0 then
        begin
          Pos := Pos - StopLen;
          Break;
        end;
      end;
      // End-of-sequence: the codebase-wide "NextTokenInt < 2" rule (the EOS
      // token is stored and counted, like GenerateStringFromCasualNN).
      if NextTokenInt < 2 then Break;
    end;
    Result := Pos;
  finally
    InV.Free;
  end;
end;

function GenerateStringStreamed(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase): string;
begin
  // Bit-for-bit the original behavior: no penalty, no stop strings.
  Result := GenerateStringStreamed(Session, Dict, InputString, MaxNewTokens,
    MaxTotalLen, oSampler, nil, []);
end;

function GenerateStringStreamed(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase; Penalty: TNNetTokenHistoryPenalty;
  const StopStrings: array of string): string;
begin
  // Bit-for-bit the unconstrained behavior.
  Result := GenerateStringStreamed(Session, Dict, InputString, MaxNewTokens,
    MaxTotalLen, oSampler, Penalty, StopStrings, nil);
end;

function GenerateStringStreamed(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase; Penalty: TNNetTokenHistoryPenalty;
  const StopStrings: array of string;
  Constraint: TNNetTokenConstraint): string;
var
  Tokens, StopToks: TNeuralIntegerArray;
  TokenStops: TNNetTokenSequences;
  PromptLen, TotalLen, Pos, VocabCount, S, CutAt: integer;
  Continuation: string;
begin
  Result := InputString;
  VocabCount := Dict.GetVocabCount();
  Dict.Tokenize(InputString, Tokens);
  PromptLen := Length(Tokens);
  if PromptLen < 1 then Exit; // nothing to condition on
  // Tokenize each stop string into a token-id stop sequence so the token
  // core terminates early; entries that tokenize to nothing are skipped.
  SetLength(TokenStops, 0);
  for S := 0 to High(StopStrings) do
  begin
    if StopStrings[S] = '' then continue;
    Dict.Tokenize(StopStrings[S], StopToks);
    if Length(StopToks) > 0 then
    begin
      SetLength(TokenStops, Length(TokenStops) + 1);
      TokenStops[High(TokenStops)] := Copy(StopToks, 0, Length(StopToks));
    end;
  end;
  TotalLen := GenerateTokensStreamed(Session, Tokens, PromptLen,
    MaxNewTokens, MaxTotalLen, oSampler, Penalty, TokenStops, Constraint);
  // Detokenize the continuation; stop at the first special token (< 2) for
  // display (the TokensToText convention) and join with a space only for
  // separator vocabularies (word dicts; BPE vocabularies concatenate).
  Continuation := '';
  for Pos := PromptLen to TotalLen - 1 do
  begin
    if Tokens[Pos] < 2 then Break;
    if Tokens[Pos] < VocabCount then
    begin
      if Dict.TokenizerHasSeparator
      then Continuation := Continuation + ' ' + Dict.DeTokenize(Tokens[Pos])
      else Continuation := Continuation + Dict.DeTokenize(Tokens[Pos]);
    end;
  end;
  // String-level safety net: even when the stop text was emitted across
  // token boundaries the token core could not match, truncate the
  // continuation at the EARLIEST occurrence of any stop string.
  CutAt := 0;
  for S := 0 to High(StopStrings) do
  begin
    if StopStrings[S] = '' then continue;
    Pos := system.Pos(StopStrings[S], Continuation);
    if (Pos > 0) and ((CutAt = 0) or (Pos < CutAt)) then CutAt := Pos;
  end;
  if CutAt > 0 then Continuation := Copy(Continuation, 1, CutAt - 1);
  Result := Result + Continuation;
  SetLength(Tokens, 0);
end;

// Forward pass: encode (Prompt + Generated) and return the next-token
// distribution as log-probabilities in LogProbs (length = vocab size).
// The output volume is treated as a probability distribution (SoftMax head);
// if it does not sum to ~1 it is re-normalised defensively before the log.
procedure NextLogProbs(NN: TNNet; const Context: string;
  InputVolume, OutputVolume: TNNetVolume; var LogProbs: array of TNeuralFloat);
var
  I: integer;
  Total: TNeuralFloat;
begin
  InputVolume.OneHotEncodingReversed(Context);
  NN.Compute(InputVolume, OutputVolume);
  Total := OutputVolume.GetSum();
  if (Total <= 0) then Total := 1.0;
  for I := 0 to OutputVolume.Size - 1 do
    LogProbs[I] := SafeLogProb(OutputVolume.Raw[I] / Total);
end;

function DecodeGreedy(NN: TNNet; const Prompt: string;
  MaxLen: integer): TNNetDecodeResult;
begin
  // Bit-for-bit the original behavior: no stop strings.
  Result := DecodeGreedy(NN, Prompt, MaxLen, []);
end;

// Returns the length of the longest StopStrings entry that is a suffix of
// Text; 0 when none matches.
function MatchStopStringSuffix(const Text: string;
  const StopStrings: array of string): integer;
var
  S, L: integer;
begin
  Result := 0;
  for S := 0 to High(StopStrings) do
  begin
    L := Length(StopStrings[S]);
    if (L > Result) and (L >= 1) and (L <= Length(Text)) and
      (Copy(Text, Length(Text) - L + 1, L) = StopStrings[S]) then
      Result := L;
  end;
end;

function DecodeGreedy(NN: TNNet; const Prompt: string; MaxLen: integer;
  const StopStrings: array of string): TNNetDecodeResult;
begin
  // Bit-for-bit the unconstrained behavior.
  Result := DecodeGreedy(NN, Prompt, MaxLen, StopStrings, nil);
end;

function DecodeGreedy(NN: TNNet; const Prompt: string; MaxLen: integer;
  const StopStrings: array of string;
  Constraint: TNNetTokenConstraint): TNNetDecodeResult;
var
  InputVolume, OutputVolume: TNNetVolume;
  LogProbs: array of TNeuralFloat;
  PromptIds: TNeuralIntegerArray;
  VocabSize, Step, Best, I, StopLen: integer;
  Context: string;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  VocabSize := OutputVolume.Size;
  SetLength(LogProbs, VocabSize);
  Result.Text := '';
  Result.SumLogProb := 0;
  Result.Finished := False;
  Context := Prompt;
  if Assigned(Constraint) then
  begin
    // DecodeGreedy is char-level: token id = character code.
    SetLength(PromptIds, Length(Prompt));
    for I := 1 to Length(Prompt) do PromptIds[I - 1] := Ord(Prompt[I]);
    Constraint.Reset(PromptIds);
  end;
  try
    for Step := 1 to MaxLen do
    begin
      NextLogProbs(NN, Context, InputVolume, OutputVolume, LogProbs);
      // Constrained step: argmax over the ALLOWED tokens only; when no token
      // is allowed fall back to the plain argmax (same policy as
      // TNNetTokenConstraint.MaskAllowed).
      Best := -1;
      if Assigned(Constraint) then
        for I := 0 to VocabSize - 1 do
          if Constraint.TokenAllowed(I) and
            ((Best < 0) or (LogProbs[I] > LogProbs[Best])) then Best := I;
      if Best < 0 then
      begin
        Best := 0;
        for I := 1 to VocabSize - 1 do
          if LogProbs[I] > LogProbs[Best] then Best := I;
      end;
      Result.SumLogProb := Result.SumLogProb + LogProbs[Best];
      if Assigned(Constraint) then Constraint.Commit(Best);
      if Best = csDecodeEOSToken then
      begin
        Result.Finished := True;
        Break;
      end;
      Result.Text := Result.Text + Chr(Best);
      Context := Context + Chr(Best);
      // Stop strings: terminate when the generated text ends with one and
      // TRIM it from the output (deliberate termination, like EOS).
      if Length(StopStrings) > 0 then
      begin
        StopLen := MatchStopStringSuffix(Result.Text, StopStrings);
        if StopLen > 0 then
        begin
          SetLength(Result.Text, Length(Result.Text) - StopLen);
          Result.Finished := True;
          Break;
        end;
      end;
    end;
  finally
    InputVolume.Free;
    OutputVolume.Free;
  end;
  Result.Score := Result.SumLogProb /
    LengthPenaltyDenominator(Length(Result.Text), 0);
end;

type
  TBeam = record
    Text: string;
    SumLogProb: TNeuralFloat;
    Score: TNeuralFloat;
    Finished: boolean;
  end;
  TBeamArray = array of TBeam;

// Insertion sort beams in DESCENDING Score (small arrays, B is tiny).
procedure SortBeamsByScore(var Beams: TBeamArray);
var
  I, J: integer;
  Tmp: TBeam;
begin
  for I := 1 to High(Beams) do
  begin
    Tmp := Beams[I];
    J := I - 1;
    while (J >= 0) and (Beams[J].Score < Tmp.Score) do
    begin
      Beams[J + 1] := Beams[J];
      Dec(J);
    end;
    Beams[J + 1] := Tmp;
  end;
end;

function DecodeBeamSearchAll(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;
var
  InputVolume, OutputVolume: TNNetVolume;
  LogProbs: array of TNeuralFloat;
  VocabSize, Step, I, T, B: integer;
  Live: TBeamArray;      // still-growing beams
  Finished: TBeamArray;  // beams that emitted EOS
  Cand: TBeamArray;      // expansion candidates for this step
  NewBeam: TBeam;
  CutScore: TNeuralFloat;
  AllDominated: boolean;
begin
  if BeamWidth < 1 then BeamWidth := 1;
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  VocabSize := OutputVolume.Size;
  SetLength(LogProbs, VocabSize);
  try
    SetLength(Live, 1);
    Live[0].Text := '';
    Live[0].SumLogProb := 0;
    Live[0].Score := 0;
    Live[0].Finished := False;
    SetLength(Finished, 0);

    for Step := 1 to MaxLen do
    begin
      if Length(Live) = 0 then Break;

      // (d) Early stop: if we already have BeamWidth finished beams and the
      // best live beam cannot beat the worst kept finished one, stop growing.
      if Length(Finished) >= BeamWidth then
      begin
        SortBeamsByScore(Finished);
        CutScore := Finished[BeamWidth - 1].Score;
        AllDominated := True;
        for I := 0 to High(Live) do
          // A growing beam's sum-log-prob only decreases; its best possible
          // future score is bounded above by its current score (adding more
          // negative log-probs and a >=1 penalty denominator can only lower
          // it once alpha>=0 and tokens are < prob 1). Use current score as
          // an admissible upper bound for the prune decision.
          if Live[I].Score > CutScore then AllDominated := False;
        if AllDominated then Break;
      end;

      // Expand every live beam by every vocabulary token.
      SetLength(Cand, 0);
      for B := 0 to High(Live) do
      begin
        NextLogProbs(NN, Prompt + Live[B].Text,
          InputVolume, OutputVolume, LogProbs);
        for T := 0 to VocabSize - 1 do
        begin
          NewBeam.SumLogProb := Live[B].SumLogProb + LogProbs[T];
          if T = csDecodeEOSToken then
          begin
            NewBeam.Text := Live[B].Text;
            NewBeam.Finished := True;
            NewBeam.Score := NewBeam.SumLogProb /
              LengthPenaltyDenominator(Length(NewBeam.Text), LengthPenalty);
            SetLength(Finished, Length(Finished) + 1);
            Finished[High(Finished)] := NewBeam;
          end
          else
          begin
            NewBeam.Text := Live[B].Text + Chr(T);
            NewBeam.Finished := False;
            NewBeam.Score := NewBeam.SumLogProb /
              LengthPenaltyDenominator(Length(NewBeam.Text), LengthPenalty);
            SetLength(Cand, Length(Cand) + 1);
            Cand[High(Cand)] := NewBeam;
          end;
        end;
      end;

      // Re-prune the survivors to the top BeamWidth by length-penalised score.
      SortBeamsByScore(Cand);
      if Length(Cand) > BeamWidth then SetLength(Cand, BeamWidth);
      Live := Copy(Cand, 0, Length(Cand));
    end;

    // Merge any remaining live beams into the finished pool (MaxLen reached).
    for B := 0 to High(Live) do
    begin
      SetLength(Finished, Length(Finished) + 1);
      Finished[High(Finished)] := Live[B];
    end;

    SortBeamsByScore(Finished);
    SetLength(Result, Length(Finished));
    for I := 0 to High(Finished) do
    begin
      Result[I].Text := Finished[I].Text;
      Result[I].SumLogProb := Finished[I].SumLogProb;
      Result[I].Score := Finished[I].Score;
      Result[I].Finished := Finished[I].Finished;
    end;
  finally
    InputVolume.Free;
    OutputVolume.Free;
  end;
end;

function DecodeBeamSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  LengthPenalty: TNeuralFloat): TNNetDecodeResult;
var
  All: TNNetDecodeResultArray;
begin
  All := DecodeBeamSearchAll(NN, Prompt, MaxLen, BeamWidth, LengthPenalty);
  if Length(All) > 0 then
    Result := All[0]
  else
  begin
    Result.Text := '';
    Result.SumLogProb := 0;
    Result.Score := 0;
    Result.Finished := False;
  end;
end;

{ Seq2seq (encoder-decoder) generation }

function Seq2SeqEncoderStatesInput(DecoderNet: TNNet): TNNetLayer;
var
  LayerCnt, InputCnt: integer;
begin
  Result := nil;
  InputCnt := 0;
  for LayerCnt := 0 to DecoderNet.Layers.Count - 1 do
    if DecoderNet.Layers[LayerCnt] is TNNetInput then
    begin
      Inc(InputCnt);
      if InputCnt = 2 then
      begin
        Result := DecoderNet.Layers[LayerCnt];
        exit;
      end;
    end;
  raise EArgumentException.Create('Seq2SeqEncoderStatesInput: the net has ' +
    'no second TNNetInput - not an encoder-decoder pair''s decoder?');
end;

function DecodeSeq2SeqGreedy(EncoderNet, DecoderNet: TNNet;
  const SourceTokens: array of integer;
  StartTokenId, EOSTokenId, MaxNewTokens: integer): TNeuralIntegerArray;
begin
  // Bit-for-bit the Sampler = nil sampled path (pure argmax over logits, so
  // the Temperature argument is irrelevant there).
  Result := DecodeSeq2SeqSampled(EncoderNet, DecoderNet, SourceTokens,
    StartTokenId, EOSTokenId, MaxNewTokens, nil);
end;

function DecodeSeq2SeqSampled(EncoderNet, DecoderNet: TNNet;
  const SourceTokens: array of integer;
  StartTokenId, EOSTokenId, MaxNewTokens: integer;
  Sampler: TNNetSamplerBase;
  Temperature: TNeuralFloat = 1.0): TNeuralIntegerArray;
const
  csMinTemperature = 1e-6;
var
  EncStates: TNNetLayer;
  EncToks, DecToks, Probs: TNNetVolume;
  Logits: TNNetVolume;          // the decoder's own output volume (not owned)
  Targets: TNeuralIntegerArray; // StartTokenId + generated prefix
  EncSeqLen, DecSeqLen, VocabSize: integer;
  CurLen, Pos, T, Next, Base: integer;
  MaxLogit, SumExp: TNeuralFloat;
begin
  SetLength(Result, 0);
  if MaxNewTokens < 1 then exit;
  EncSeqLen := EncoderNet.GetFirstLayer().Output.Size;
  if Length(SourceTokens) <> EncSeqLen then
    raise EArgumentException.Create('DecodeSeq2SeqSampled: ' +
      IntToStr(Length(SourceTokens)) + ' source tokens but the encoder was ' +
      'built at EncSeqLen ' + IntToStr(EncSeqLen) + '.');
  EncStates := Seq2SeqEncoderStatesInput(DecoderNet);
  if EncoderNet.GetLastLayer().Output.Size <> EncStates.Output.Size then
    raise EArgumentException.Create('DecodeSeq2SeqSampled: encoder output ' +
      'size ' + IntToStr(EncoderNet.GetLastLayer().Output.Size) +
      ' does not match the decoder''s encoder-states input size ' +
      IntToStr(EncStates.Output.Size) + ' (EncSeqLen/d_model mismatch?).');
  DecSeqLen := DecoderNet.GetFirstLayer().Output.Size;
  Logits := DecoderNet.GetLastLayer().Output;
  VocabSize := Logits.Depth;
  if Temperature < csMinTemperature then Temperature := csMinTemperature;
  EncToks := TNNetVolume.Create(EncSeqLen, 1, 1);
  DecToks := TNNetVolume.Create(DecSeqLen, 1, 1);
  Probs := TNNetVolume.Create(VocabSize, 1, 1);
  try
    // (1) Encode the source ONCE and cache the hidden states in the
    // decoder's second input - they are constant across decode steps.
    for Pos := 0 to EncSeqLen - 1 do EncToks.FData[Pos] := SourceTokens[Pos];
    EncoderNet.Compute(EncToks);
    EncStates.Output.Copy(EncoderNet.GetLastLayer().Output);
    // (2) Autoregress the target from StartTokenId. Positions past the
    // prefix are padded with StartTokenId: causal self-attention makes them
    // invisible to the rows at < CurLen, so row CurLen-1 is exactly what a
    // longer real prefix would produce there.
    SetLength(Targets, DecSeqLen);
    Targets[0] := StartTokenId;
    CurLen := 1;
    while True do
    begin
      for Pos := 0 to DecSeqLen - 1 do
        if Pos < CurLen
        then DecToks.FData[Pos] := Targets[Pos]
        else DecToks.FData[Pos] := StartTokenId;
      DecoderNet.Compute(DecToks);
      if Sampler = nil then
        // Greedy: argmax over the logits row (Temperature-invariant).
        Next := Logits.GetClassOnPixel(CurLen - 1, 0)
      else
      begin
        // Stable softmax of the row at Temperature, then the sampler draws.
        Base := (CurLen - 1) * VocabSize;
        MaxLogit := Logits.FData[Base];
        for T := 1 to VocabSize - 1 do
          if Logits.FData[Base + T] > MaxLogit then
            MaxLogit := Logits.FData[Base + T];
        SumExp := 0;
        for T := 0 to VocabSize - 1 do
        begin
          Probs.FData[T] := Exp((Logits.FData[Base + T] - MaxLogit) /
            Temperature);
          SumExp := SumExp + Probs.FData[T];
        end;
        if SumExp <= 0 then SumExp := 1.0;
        for T := 0 to VocabSize - 1 do
          Probs.FData[T] := Probs.FData[T] / SumExp;
        Next := Sampler.GetToken(Probs);
      end;
      SetLength(Result, Length(Result) + 1);
      Result[High(Result)] := Next;
      // EOS is appended and counted (mirroring GenerateTokensStreamed).
      if Next = EOSTokenId then break;
      if Length(Result) >= MaxNewTokens then break;
      // Build-time decoder capacity reached: the token just generated cannot
      // be fed back as input.
      if CurLen >= DecSeqLen then break;
      Targets[CurLen] := Next;
      Inc(CurLen);
    end;
  finally
    Probs.Free;
    DecToks.Free;
    EncToks.Free;
  end;
end;

// Insertion sort token beams in DESCENDING Score. Strictly-less comparison
// keeps earlier-inserted beams ahead on ties (candidates are appended in
// token-id order, so ties resolve to the lowest token id - matching the
// first-max rule of the greedy argmax). Small arrays, B is tiny.
procedure SortTokenBeamsByScore(var Beams: TNNetTokenDecodeResultArray);
var
  I, J: integer;
  Tmp: TNNetTokenDecodeResult;
begin
  for I := 1 to High(Beams) do
  begin
    Tmp := Beams[I];
    J := I - 1;
    while (J >= 0) and (Beams[J].Score < Tmp.Score) do
    begin
      Beams[J + 1] := Beams[J];
      Dec(J);
    end;
    Beams[J + 1] := Tmp;
  end;
end;

function DecodeSeq2SeqBeamSearchAll(EncoderNet, DecoderNet: TNNet;
  const SourceTokens: array of integer;
  StartTokenId, EOSTokenId, MaxNewTokens: integer;
  BeamWidth: integer; LengthPenalty: TNeuralFloat): TNNetTokenDecodeResultArray;
var
  EncStates: TNNetLayer;
  EncToks, DecToks: TNNetVolume;
  Logits: TNNetVolume;          // the decoder's own output volume (not owned)
  LogProbs: array of TNeuralFloat;
  Live: TNNetTokenDecodeResultArray;     // still-growing beams
  Finished: TNNetTokenDecodeResultArray; // beams that emitted EOSTokenId
  Cand: TNNetTokenDecodeResultArray;     // expansion candidates per step
  NewBeam: TNNetTokenDecodeResult;
  EncSeqLen, DecSeqLen, VocabSize, EffMaxNew: integer;
  Step, B, T, Pos, PrevLen, Base: integer;
  MaxLogit, SumExp, CutScore: TNeuralFloat;
  AllDominated: boolean;
begin
  SetLength(Result, 0);
  if MaxNewTokens < 1 then exit;
  if BeamWidth < 1 then BeamWidth := 1;
  EncSeqLen := EncoderNet.GetFirstLayer().Output.Size;
  if Length(SourceTokens) <> EncSeqLen then
    raise EArgumentException.Create('DecodeSeq2SeqBeamSearchAll: ' +
      IntToStr(Length(SourceTokens)) + ' source tokens but the encoder was ' +
      'built at EncSeqLen ' + IntToStr(EncSeqLen) + '.');
  EncStates := Seq2SeqEncoderStatesInput(DecoderNet);
  if EncoderNet.GetLastLayer().Output.Size <> EncStates.Output.Size then
    raise EArgumentException.Create('DecodeSeq2SeqBeamSearchAll: encoder ' +
      'output size ' + IntToStr(EncoderNet.GetLastLayer().Output.Size) +
      ' does not match the decoder''s encoder-states input size ' +
      IntToStr(EncStates.Output.Size) + ' (EncSeqLen/d_model mismatch?).');
  DecSeqLen := DecoderNet.GetFirstLayer().Output.Size;
  Logits := DecoderNet.GetLastLayer().Output;
  VocabSize := Logits.Depth;
  // Build-time decoder capacity: a beam with G generated tokens re-forwards
  // a (1 + G)-token prefix, so generation caps at DecSeqLen tokens - the
  // same stopping rule as DecodeSeq2SeqGreedy's "CurLen >= DecSeqLen" break.
  EffMaxNew := MaxNewTokens;
  if EffMaxNew > DecSeqLen then EffMaxNew := DecSeqLen;
  SetLength(LogProbs, VocabSize);
  EncToks := TNNetVolume.Create(EncSeqLen, 1, 1);
  DecToks := TNNetVolume.Create(DecSeqLen, 1, 1);
  try
    // (1) Encode the source ONCE and cache the hidden states in the
    // decoder's second input - constant across all beams and steps.
    for Pos := 0 to EncSeqLen - 1 do EncToks.FData[Pos] := SourceTokens[Pos];
    EncoderNet.Compute(EncToks);
    EncStates.Output.Copy(EncoderNet.GetLastLayer().Output);

    // (2) Candidate loop. Every live beam at step S carries S-1 generated
    // tokens, so the whole live front shares one prefix length per step.
    SetLength(Live, 1);
    SetLength(Live[0].Tokens, 0);
    Live[0].SumLogProb := 0;
    Live[0].Score := 0;
    Live[0].Finished := False;
    SetLength(Finished, 0);

    for Step := 1 to EffMaxNew do
    begin
      if Length(Live) = 0 then Break;

      // Early stop (same idiom as DecodeBeamSearchAll): once BeamWidth
      // finished beams exist and no live beam outscores the BeamWidth-th
      // finished one, stop growing. A growing beam's sum-log-prob only
      // decreases, so its current score is used as the prune bound.
      if Length(Finished) >= BeamWidth then
      begin
        SortTokenBeamsByScore(Finished);
        CutScore := Finished[BeamWidth - 1].Score;
        AllDominated := True;
        for B := 0 to High(Live) do
          if Live[B].Score > CutScore then AllDominated := False;
        if AllDominated then Break;
      end;

      // Expand every live beam by every vocabulary token.
      SetLength(Cand, 0);
      for B := 0 to High(Live) do
      begin
        PrevLen := Length(Live[B].Tokens); // = Step - 1
        // StartTokenId + generated prefix, padded with StartTokenId: causal
        // self-attention makes the padding invisible to row PrevLen (same
        // re-forward convention as DecodeSeq2SeqSampled).
        DecToks.FData[0] := StartTokenId;
        for Pos := 1 to DecSeqLen - 1 do
          if Pos <= PrevLen
          then DecToks.FData[Pos] := Live[B].Tokens[Pos - 1]
          else DecToks.FData[Pos] := StartTokenId;
        DecoderNet.Compute(DecToks);
        // Stable softmax of the logits row at the last prefix position,
        // then SafeLogProb - the log image of the greedy/sampled row.
        Base := PrevLen * VocabSize;
        MaxLogit := Logits.FData[Base];
        for T := 1 to VocabSize - 1 do
          if Logits.FData[Base + T] > MaxLogit then
            MaxLogit := Logits.FData[Base + T];
        SumExp := 0;
        for T := 0 to VocabSize - 1 do
          SumExp := SumExp + Exp(Logits.FData[Base + T] - MaxLogit);
        if SumExp <= 0 then SumExp := 1.0;
        for T := 0 to VocabSize - 1 do
          LogProbs[T] :=
            SafeLogProb(Exp(Logits.FData[Base + T] - MaxLogit) / SumExp);
        for T := 0 to VocabSize - 1 do
        begin
          NewBeam.SumLogProb := Live[B].SumLogProb + LogProbs[T];
          NewBeam.Tokens := Copy(Live[B].Tokens, 0, PrevLen);
          SetLength(NewBeam.Tokens, PrevLen + 1);
          NewBeam.Tokens[PrevLen] := T; // EOS included, like greedy
          NewBeam.Finished := (T = EOSTokenId);
          NewBeam.Score := NewBeam.SumLogProb /
            LengthPenaltyDenominator(PrevLen + 1, LengthPenalty);
          if NewBeam.Finished then
          begin
            SetLength(Finished, Length(Finished) + 1);
            Finished[High(Finished)] := NewBeam;
          end
          else
          begin
            SetLength(Cand, Length(Cand) + 1);
            Cand[High(Cand)] := NewBeam;
          end;
        end;
      end;

      // Re-prune the survivors to the top BeamWidth by penalised score.
      SortTokenBeamsByScore(Cand);
      if Length(Cand) > BeamWidth then SetLength(Cand, BeamWidth);
      Live := Copy(Cand, 0, Length(Cand));
    end;

    // Merge remaining live beams into the pool (MaxNewTokens/capacity hit).
    for B := 0 to High(Live) do
    begin
      SetLength(Finished, Length(Finished) + 1);
      Finished[High(Finished)] := Live[B];
    end;
    SortTokenBeamsByScore(Finished);
    Result := Finished;
  finally
    DecToks.Free;
    EncToks.Free;
  end;
end;

function DecodeSeq2SeqBeamSearch(EncoderNet, DecoderNet: TNNet;
  const SourceTokens: array of integer;
  StartTokenId, EOSTokenId, MaxNewTokens: integer;
  BeamWidth: integer; LengthPenalty: TNeuralFloat): TNeuralIntegerArray;
var
  All: TNNetTokenDecodeResultArray;
begin
  All := DecodeSeq2SeqBeamSearchAll(EncoderNet, DecoderNet, SourceTokens,
    StartTokenId, EOSTokenId, MaxNewTokens, BeamWidth, LengthPenalty);
  if Length(All) > 0
  then Result := Copy(All[0].Tokens, 0, Length(All[0].Tokens))
  else SetLength(Result, 0);
end;

end.
