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
  - TNNetGrammarConstraint: GENERAL grammar-constrained generation, the
    generalization of the JSON machine to an arbitrary GBNF-subset grammar
    (llama.cpp style). TNNetGrammar compiles GBNF text to a flat pushdown
    representation; TNNetGrammarMachine runs it as an NFA-of-stacks (the active
    set is a SET of parse stacks, so alternation / optional / repetition are
    handled exactly), reusing the same char-by-char token-feasibility walk: a
    token is allowed iff feeding its characters through a FORK of the machine
    accepts all of them, and EOS (< 2) iff the machine is in an accepting
    (complete) state. The grammar constrains ONLY the generated text (the
    prompt is plain conditioning). Supported subset: rule defs name ::= body
    (entry rule MUST be 'root'); sequence + alternation '|'; char literals
    "abc" (\n \r \t \" \\ escapes); classes [a-z], negation [^...], '.' any;
    repetition * + ? and grouping ( ... ); rule references (recursion ->
    the pushdown stack); '#' comments and free whitespace. NOT in v1: bounded
    repetition {m,n}, regex->DFA, and numeric \xNN/\uNNNN class escapes.

LOGITS-PROCESSOR CHAIN + GENERATION CONFIG. The penalties / temperature /
constraints above also compose through TNNetLogitsProcessor, a chainable
per-step distribution transform (the transformers GenerationMixin
"logits processor" idea):
  - DOMAIN CONVENTION: on the causal-LM streamed paths the model ends in a
    SoftMax, so processors receive the POST-SOFTMAX PROBABILITY row, not raw
    logits. Every shipped processor implements the probability-domain image
    of its logit rule (documented per class), and ExpectsProbabilities()
    makes the domain explicit: the streamed loop refuses a processor that
    declares a raw-logit domain instead of silently misapplying it.
  - TNNetTemperatureProcessor is the temperature knob (probability-domain
    image of logits/T: p^(1/T) renormalized, computed stably in log space).
    Temperature is clamped to >= csDecodeMinTemperature and Temperature -> 0
    degenerates to greedy argmax - the same convention as
    DecodeSeq2SeqSampled. Convenience GenerateTokensStreamed /
    GenerateStringStreamed overloads take a Temperature argument directly.
  - TNNetPenaltyProcessor / TNNetConstraintProcessor are thin adapters over
    the existing TNNetTokenHistoryPenalty / TNNetTokenConstraint (which keep
    their public interfaces unchanged).
  - TNNetLogitsProcessorChain runs processors IN ORDER (it is itself a
    processor, so chains nest). The pre-existing penalty/constraint overloads
    now delegate through adapter chains built in the historical order
    (penalty, then constraint), keeping them bit-for-bit unchanged.
  - TGenerationConfig bundles sampler + temperature + penalties + constraint
    + extra processors + stopping criteria (EOS-id list, stop sequences /
    strings, max new tokens) into one record consumed by
    GenerateTokensWithConfig / GenerateStringWithConfig. The implied pipeline
    order is: penalty -> temperature -> extra processors -> constraint ->
    sampler (the constraint runs LAST so its structural guarantees - e.g.
    JSON validity - cannot be broken by a later transform).
  - TNNetCFGProcessor is CLASSIFIER-FREE GUIDANCE (the transformers
    UnbatchedClassifierFreeGuidanceLogitsProcessor): it owns a SECOND width-1
    streaming session for an unconditional / negative prompt, runs it forward
    alongside the conditional loop each step, and combines the two
    distributions in LOGIT space (logits := uncond + scale*(cond-uncond))
    before the sampler. GuidanceScale=1 is bit-for-bit conditional-only
    decoding; GuidanceScale=0 ignores the conditional prompt entirely. As a
    plain TNNetLogitsProcessor it plugs into any processor-accepting overload.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralnetwork;

type
  // Forward declaration: TNNetCFGProcessor (below) holds an unconditional
  // streaming session, but the session class is declared further down.
  TNNetStreamingDecoder = class;

  // External per-candidate scorer for Best-of-N reranking (the Bradley-Terry /
  // reward-model plug). Returns a scalar reward for a generated continuation;
  // Best-of-N returns the candidate with the MAXIMUM scorer value. Plain (not
  // "of object") so a bare function can be passed. Coded by Claude (AI).
  TNNetSequenceScorer = function(const Prompt, Generated: string): TNeuralFloat;

  // DoLa candidate-layer bucket selection (Chuang et al. 2023, the paper's
  // DoLa-low / DoLa-high dynamic premature-layer pools). The lens-compatible
  // earlier layers (those whose Output.Size matches the head-input slot) are
  // ordered by depth and split at the midpoint:
  //   dlbFull - all lens-compatible layers (the v1 fixed bucket; default).
  //   dlbLow  - the SHALLOW half (early layers, index < count div 2).
  //   dlbHigh - the DEEP half (later layers, index >= count div 2).
  // With a single candidate layer dlbLow keeps it and dlbHigh is empty (an
  // empty bucket degrades DoLa to greedy, same invariant as Alpha<=0).
  // Coded by Claude (AI).
  TDoLaLayerBucket = (dlbFull, dlbLow, dlbHigh);

  // Answer-extraction callback for the self-consistency variant: maps a full
  // generated continuation to its canonical ANSWER string (e.g. the final
  // number after "####"). Candidates whose extracted answer is the empty string
  // are ignored in the vote. Coded by Claude (AI).
  TNNetAnswerExtractor = function(const Generated: string): string;

  // A scored decode candidate: the generated text (excluding the prompt) and
  // its cumulative log-probability.
  TNNetDecodeResult = record
    Text: string;        // generated continuation (prompt NOT included)
    SumLogProb: TNeuralFloat; // sum of per-step log-probabilities
    Score: TNeuralFloat; // length-penalised score actually ranked on
    Finished: boolean;   // True if the beam emitted EOS
  end;

  TNNetDecodeResultArray = array of TNNetDecodeResult;

  // -------------------------------------------------------------------------
  // NEEDLE-IN-A-HAYSTACK long-context evaluation (NeedleInHaystackReport).
  //
  // Places a FACT ("needle") at varying depths inside a synthetic long
  // context ("haystack" filler), appends a retrieval question, runs
  // generation, and checks whether the answer was recovered. The result is a
  // (depth x context-length) accuracy grid - the standard probe for whether a
  // position-extrapolation scheme (RoPE scaling) or a KV-cache-eviction policy
  // actually preserves access to information far from the query position.
  //
  // Pluggable callbacks keep it model-agnostic so it can be exercised both by
  // a real char-level TNNet (via the convenience overload that wraps
  // DecodeGreedy) and by a deterministic stand-in in tests:
  //   TNeedleFillerCallback   builds CharCount characters of distractor text
  //                           (must NOT contain the needle answer);
  //   TNeedleGenerateCallback runs the model on a fully-assembled prompt and
  //                           returns its generated continuation.
  // Coded by Claude (AI).
  TNeedleFillerCallback =
    function(CharCount: integer; Data: Pointer): string;
  TNeedleGenerateCallback =
    function(const Prompt: string; Data: Pointer): string;

  // Outcome of a single (length,depth) probe plus the assembled prompt and the
  // raw model output, so callers can inspect individual cells.
  // Coded by Claude (AI).
  TNeedleCell = record
    ContextLen: integer;       // requested filler length in characters
    DepthFraction: TNeuralFloat; // needle insertion depth in [0,1]
    Prompt: string;            // the fully-assembled probe prompt
    Output: string;            // the model's generated continuation
    Hit: boolean;              // True if Output contained the needle answer
  end;

  // Full grid result: Cells[depthRow][lenCol], the requested axes, and the
  // overall hit rate (HitCount/TotalCount). Report is the rendered grid.
  // Coded by Claude (AI).
  TNeedleInHaystackResult = record
    Cells: array of array of TNeedleCell; // [depth][length]
    DepthFractions: TNeuralFloatDynArr;
    ContextLengths: TNeuralIntegerArray;
    HitCount: integer;
    TotalCount: integer;
    Accuracy: TNeuralFloat;    // HitCount / TotalCount (0 if empty)
    Report: string;            // rendered ASCII grid + summary
  end;

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

  { TNNetTokenHealingConstraint }
  // ONE-SHOT whitelist for guidance-style TOKEN HEALING: like
  // TNNetAllowedTokensConstraint, but the restriction applies to the FIRST
  // emitted token only - Commit lifts it, and Reset re-arms it for the next
  // generation run. Token healing backs up over the LAST prompt token and
  // constrains the first generated token to vocabulary entries whose text
  // EXTENDS the dropped token's text (the dropped token itself included, so
  // generation can never get stuck), fixing the classic BPE boundary
  // artifact ("http:" never continuing with "//" because the prompt split
  // mid-merge). Build the allowed set yourself, or use PrepareTokenHealing
  // for the dict-driven prefix scan + prompt trim in one call.
  // Coded by Claude (AI).
  TNNetTokenHealingConstraint = class(TNNetTokenConstraint)
    private
      FAllowed: array of boolean;
      FDone: boolean;
    public
      constructor Create(const AllowedTokens: array of integer);
      procedure Reset(const PromptTokens: array of integer); override;
      function TokenAllowed(TokenId: integer): boolean; override;
      procedure Commit(TokenId: integer); override;
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

  // Flat compiled-grammar element kind (the llama.cpp GBNF encoding: a grammar
  // rule body is a flat array of these, alternates separated by getAlt and
  // terminated by getEnd). Coded by Claude (AI).
  TNNetGrammarElemType = (
    getEnd,        // end of a rule body (an alternate completes here)
    getAlt,        // separates the alternates of a rule body
    getRuleRef,    // a reference to another rule (Value = rule index)
    getChar,       // a single literal character (Value = ord(char))
    getCharSet,    // a character class: Value = index into FCharSets
    getCharSetNot, // a negated character class (Value = index into FCharSets)
    getCharAny);   // '.' : any character (rejects only nothing; impl excludes #0)

  // One compiled grammar element. Coded by Claude (AI).
  TNNetGrammarElem = record
    ElemType: TNNetGrammarElemType;
    Value: integer;
  end;
  TNNetGrammarElemArray = array of TNNetGrammarElem;

  // A character-class entry: an inclusive [Lo..Hi] range (a single char is
  // Lo=Hi). A class is a contiguous run of these in FRanges. Coded by Claude (AI).
  TNNetGrammarRange = record
    Lo, Hi: char;
  end;

  { TNNetGrammar }
  // A GBNF-subset grammar COMPILED to a flat pushdown representation. Supported
  // subset (llama.cpp-style, v1):
  //   - rule definitions:  name ::= body  (one per logical line; the entry
  //     rule MUST be named 'root')
  //   - sequence (concatenation) and alternation '|'
  //   - char literals "abc" (each char a literal; \n \r \t \" \\ \] escapes)
  //   - character classes [a-z0-9_], negation [^...], '.' any-char
  //   - repetition postfix '*', '+', '?' and grouping '( ... )'
  //   - rule references (recursion -> the pushdown stack)
  //   - '#' line comments and free whitespace between tokens
  // NOT in v1 (see the unit header / tasklist): bounded repetition {m,n},
  // regex->DFA, and numeric \xNN / \uNNNN escapes in classes.
  // Repetition/grouping/optional are desugared into anonymous helper rules at
  // compile time, so the run-time machine only ever sees char/charset/ruleref/
  // alt/end elements (exactly the llama.cpp scheme). Coded by Claude (AI).
  TNNetGrammar = class(TObject)
    private
      FSource: string;
      FRuleNames: TStringList;        // index -> rule name
      FRules: array of TNNetGrammarElemArray; // index -> compiled body
      FRanges: array of TNNetGrammarRange;    // pooled char-class ranges
      FCharSets: array of record First, Count: integer; end; // run in FRanges
      FRootRule: integer;
      // --- parser state (over FSource) ---
      FPos: integer;
      function PeekCh(): char;
      procedure NextCh();
      procedure SkipWS();
      function AtEnd(): boolean;
      function AddRule(const Name: string): integer;
      function FindOrAddRule(const Name: string): integer;
      function NewAnonRule(): integer;
      function AddCharSet(const ARanges: array of TNNetGrammarRange): integer;
      // Parses one rule body (alternation of sequences) into Elems, then
      // applies a possible repetition wrapper. Returns via Elems.
      procedure ParseAlternates(var Elems: TNNetGrammarElemArray);
      procedure ParseSequence(var Elems: TNNetGrammarElemArray);
      // Parses ONE element (literal/class/group/ref) plus its */+/? postfix;
      // appends the resulting element(s) to Elems.
      procedure ParseElement(var Elems: TNNetGrammarElemArray);
      // Parses a '[' ... ']' class; sets Negated for '[^'. Returns charset idx.
      function ParseCharClass(out Negated: boolean): integer;
      procedure ExpectCh(C: char);
      // Desugar helpers: append a single getRuleRef to a freshly built
      // anonymous rule implementing the postfix over AtomRule.
      procedure BuildOptRule(AtomRule: integer;
        var Elems: TNNetGrammarElemArray);
      procedure BuildStarRule(AtomRule: integer;
        var Elems: TNNetGrammarElemArray);
      procedure BuildPlusRule(AtomRule: integer;
        var Elems: TNNetGrammarElemArray);
      procedure Compile();
    public
      // Builds and compiles the grammar from GBNF text. Raises EAssertionFailed
      // on a malformed grammar or a missing 'root' rule.
      constructor Create(const GBNFText: string);
      destructor Destroy(); override;
      property RootRule: integer read FRootRule;
  end;

  { TNNetGrammarMachine }
  // Run-time NFA-of-stacks over a compiled TNNetGrammar (does NOT own it). The
  // active set is a list of STACKS; each stack is a list of element positions
  // (a position = rule index * KStride + element index, packed) with the TOP at
  // the end. FeedChar(C) keeps only the stacks whose top element matches C,
  // advances each past the matched element (descending into rule refs along the
  // way so the new tops are again terminals), and dedups. IsComplete() is true
  // when some active stack is empty (a full parse of 'root' stands). CharAllowed
  // is a non-destructive probe; CopyFrom supports copy-on-fork for beams.
  // Coded by Claude (AI).
  TNNetGrammarMachine = class(TObject)
    private
      FGrammar: TNNetGrammar;
      // Active set: FStacks[i] is a stack of packed positions, length in
      // FStackLen[i]; FStackCount stacks are live.
      FStacks: array of array of integer;
      FStackLen: array of integer;
      FStackCount: integer;
      // Scratch for FeedChar/CharAllowed (avoids per-call allocation).
      FScratch: array of array of integer;
      FScratchLen: array of integer;
      FScratchCount: integer;
      function PackPos(Rule, Idx: integer): integer;
      procedure UnpackPos(Pos: integer; out Rule, Idx: integer);
      // Pushes a stack (copy of Src[0..Len-1] then descends rule refs so the
      // new top is a terminal or the stack is empty) into the scratch set,
      // expanding alternates. Dedups against what's already in scratch.
      procedure AddStackExpanded(const Src: array of integer; Len: integer);
      // Forks Base into every alternate of RuleIdx (used by Reset and ruleref
      // expansion).
      procedure PushRuleAlternates(const Base: array of integer;
        BaseLen, RuleIdx: integer);
      procedure AddStackRaw(const Src: array of integer; Len: integer);
      function ScratchHas(const Src: array of integer; Len: integer): boolean;
      // Replaces the active set with the freshly built scratch set.
      procedure CommitScratchToActive();
      // Char matches the terminal element at Pos?
      function ElemMatches(Pos: integer; C: char): boolean;
    public
      constructor Create(AGrammar: TNNetGrammar);
      procedure Reset();
      procedure CopyFrom(Source: TNNetGrammarMachine);
      // Advances by one character; False = no active stack accepts C (the
      // active set is REPLACED by the survivors, which is empty on False - call
      // CharAllowed/use a copy if you must probe non-destructively).
      function FeedChar(C: char): boolean;
      function FeedString(const S: string): boolean;
      function CharAllowed(C: char): boolean;
      // True when a complete parse of the root rule stands (some empty stack).
      function IsComplete(): boolean;
      function ActiveCount(): integer;
  end;

  { TNNetGrammarConstraint }
  // Grammar-driven constrained-decoding hook: allows exactly the tokens whose
  // decoded surface text is a legal continuation of the grammar from the
  // current parse state. Multi-character tokens are validated transitively by
  // feeding their characters through a forked machine (the same char-by-char
  // token-feasibility walk the JSON constraint uses). The grammar constrains
  // ONLY the generated text - the prompt is plain conditioning and is NOT fed
  // through the machine. Special/EOS ids (< 2) are allowed exactly when the
  // grammar is in a complete (accepting) state; empty-string tokens are never
  // allowed. Create(GBNFText, Dict) snapshots Dict.DeTokenize(id) per id;
  // CreateCharLevel(GBNFText, VocabSize) is the char-level convention (token id
  // = character code, ids < 2 special). Owns its TNNetGrammar.
  // Coded by Claude (AI).
  TNNetGrammarConstraint = class(TNNetTokenConstraint)
    private
      FTokenStr: array of string;
      FGrammar: TNNetGrammar;
      FMachine: TNNetGrammarMachine;
      FProbe: TNNetGrammarMachine;
    public
      constructor Create(const GBNFText: string; Dict: TStringListInt); overload;
      constructor CreateCharLevel(const GBNFText: string; VocabSize: integer);
      destructor Destroy(); override;
      procedure Reset(const PromptTokens: array of integer); override;
      function TokenAllowed(TokenId: integer): boolean; override;
      procedure Commit(TokenId: integer); override;
      // The live machine (after the committed tokens) for inspection.
      property Machine: TNNetGrammarMachine read FMachine;
  end;

  { TNNetLogitsProcessor }
  // Chainable next-token distribution transform - the GenerationMixin-style
  // "logits processor" abstraction. The generation loop calls
  // Reset(PromptTokens) once, then per step ProcessRow(Row) on the next-token
  // distribution BEFORE the sampler/argmax reads it, and Commit(Token) after
  // each emitted token so stateful processors advance.
  // DOMAIN CONVENTION: the causal-LM streamed loop works on POST-SOFTMAX
  // PROBABILITIES (the model ends in a SoftMax), so ProcessRow receives a
  // probability row and must leave it a valid distribution (or untouched).
  // ExpectsProbabilities() (default True) declares the domain explicitly; a
  // future raw-logit processor returns False there and the streamed loop
  // rejects it with EArgumentException instead of misapplying it.
  // Coded by Claude (AI).
  TNNetLogitsProcessor = class(TObject)
    public
      // True (default) = ProcessRow expects a POST-SOFTMAX probability row;
      // False = raw logits (not accepted by the streamed loop).
      function ExpectsProbabilities(): boolean; virtual;
      // Called once at the start of generation with the prompt token ids.
      // Default: no-op.
      procedure Reset(const PromptTokens: array of integer); virtual;
      // Transforms the next-token distribution in place. Element index =
      // token id, as everywhere in the decoder.
      procedure ProcessRow(Row: TNNetVolume); virtual; abstract;
      // Advances processor state after a token was emitted. Default: no-op.
      procedure Commit(TokenId: integer); virtual;
  end;

  { TNNetTemperatureProcessor }
  // TEMPERATURE in the probability domain: the exact image of dividing the
  // logits by T before the softmax is p := p^(1/T) renormalized; computed
  // stably in log space as exp((ln p - ln max_p)/T) / sum. Temperature = 1.0
  // is a bit-for-bit no-op; Temperature is clamped to >=
  // csDecodeMinTemperature, so Temperature -> 0 concentrates all mass on the
  // argmax (greedy degeneration - the DecodeSeq2SeqSampled convention).
  // An all-zero row is left untouched (defensive, mirroring MaskAllowed).
  // Coded by Claude (AI).
  TNNetTemperatureProcessor = class(TNNetLogitsProcessor)
    private
      FTemperature: TNeuralFloat;
    public
      constructor Create(Temperature: TNeuralFloat);
      procedure ProcessRow(Row: TNNetVolume); override;
      property Temperature: TNeuralFloat read FTemperature write FTemperature;
  end;

  { TNNetPenaltyProcessor }
  // Thin adapter exposing an existing TNNetTokenHistoryPenalty as a chain
  // processor: Reset clears the history and registers the prompt tokens (the
  // whole context is penalized - the historical convention of the streamed
  // loop), ProcessRow applies the probability-domain rule
  // (ApplyToProbabilities) and Commit registers the emitted token. The
  // wrapped penalty keeps its public interface; it is freed with the adapter
  // only when OwnsPenalty is True.
  // Coded by Claude (AI).
  TNNetPenaltyProcessor = class(TNNetLogitsProcessor)
    private
      FPenalty: TNNetTokenHistoryPenalty;
      FOwnsPenalty: boolean;
    public
      constructor Create(pPenalty: TNNetTokenHistoryPenalty;
        pOwnsPenalty: boolean = false);
      destructor Destroy(); override;
      procedure Reset(const PromptTokens: array of integer); override;
      procedure ProcessRow(Row: TNNetVolume); override;
      procedure Commit(TokenId: integer); override;
      property Penalty: TNNetTokenHistoryPenalty read FPenalty;
  end;

  { TNNetConstraintProcessor }
  // Thin adapter exposing an existing TNNetTokenConstraint as a chain
  // processor: Reset/ProcessRow(=MaskAllowed)/Commit forward 1:1 to the
  // constraint's own Reset/MaskAllowed/Commit (including the zero-mass
  // fallback policy). The wrapped constraint keeps its public interface; it
  // is freed with the adapter only when OwnsConstraint is True.
  // Coded by Claude (AI).
  TNNetConstraintProcessor = class(TNNetLogitsProcessor)
    private
      FConstraint: TNNetTokenConstraint;
      FOwnsConstraint: boolean;
    public
      constructor Create(pConstraint: TNNetTokenConstraint;
        pOwnsConstraint: boolean = false);
      destructor Destroy(); override;
      procedure Reset(const PromptTokens: array of integer); override;
      procedure ProcessRow(Row: TNNetVolume); override;
      procedure Commit(TokenId: integer); override;
      property Constraint: TNNetTokenConstraint read FConstraint;
  end;

  { TNNetLogitsProcessorChain }
  // Ordered chain of processors; Reset/ProcessRow/Commit forward to every
  // item IN INSERTION ORDER (order matters: e.g. penalty-then-temperature
  // sharpens the penalized distribution, temperature-then-penalty penalizes
  // the sharpened one). The chain is itself a TNNetLogitsProcessor, so chains
  // nest. Add returns the chain for fluent building; items added with
  // OwnsProcessor=True are freed with the chain. ExpectsProbabilities is True
  // only when EVERY item agrees (an empty chain is a no-op and returns True).
  // Coded by Claude (AI).
  TNNetLogitsProcessorChain = class(TNNetLogitsProcessor)
    private
      FItems: array of TNNetLogitsProcessor;
      FOwned: array of boolean;
      function GetCount(): integer;
      function GetItem(Index: integer): TNNetLogitsProcessor;
    public
      destructor Destroy(); override;
      function Add(P: TNNetLogitsProcessor;
        OwnsProcessor: boolean = false): TNNetLogitsProcessorChain;
      function ExpectsProbabilities(): boolean; override;
      procedure Reset(const PromptTokens: array of integer); override;
      procedure ProcessRow(Row: TNNetVolume); override;
      procedure Commit(TokenId: integer); override;
      property Count: integer read GetCount;
      property Items[Index: integer]: TNNetLogitsProcessor read GetItem; default;
  end;

  { TNNetCFGProcessor }
  // CLASSIFIER-FREE GUIDANCE (CFG) for autoregressive text generation - the
  // port of transformers' UnbatchedClassifierFreeGuidanceLogitsProcessor. Each
  // decode step the model is run TWICE: the CONDITIONAL branch (the real
  // prompt, owned by the calling generation loop) and an UNCONDITIONAL branch
  // (an empty or NEGATIVE prompt, owned by THIS processor's own width-1
  // streaming session). The two distributions are combined in LOGIT space
  // before the sampler reads them:
  //   logits := uncond + GuidanceScale * (cond - uncond)
  // GuidanceScale = 1 reproduces the conditional-only distribution exactly
  // (the term collapses to cond); GuidanceScale = 0 collapses to uncond,
  // ignoring the conditional prompt entirely; GuidanceScale > 1 SHARPENS the
  // contrast (the usual CFG regime).
  //
  // DOMAIN. The chain feeds POST-SOFTMAX PROBABILITIES (the model ends in a
  // SoftMax). CFG is a logit-space rule, so this processor maps each branch's
  // probabilities to LOG-PROBABILITIES (SafeLogProb), combines, and softmaxes
  // back to a probability row. Using log-probs instead of true pre-softmax
  // logits differs only by a per-branch additive constant; that constant is
  // uniform across tokens and cancels in the final softmax, so the combine is
  // exact (and GuidanceScale = 1 is bit-for-bit the untouched cond row).
  //
  // OWNERSHIP / WIRING (mirrors the HF design that wires an unconditional
  // model+inputs into the processor): Create takes the UNCONDITIONAL width-1
  // streaming session (typically the SAME twin net as the conditional loop, or
  // a separate twin sharing weights) and the negative/unconditional prompt
  // token ids. The session is NOT owned unless OwnsSession is True. The
  // processor drives its own session: Reset prefills the negative prompt,
  // ProcessRow steps the unconditional branch forward one token and combines,
  // Commit feeds the just-emitted token into the unconditional branch (so both
  // branches share the generated suffix, differing only in their prompt).
  // The negative prompt may be empty (pure unconditional / "no prompt"): a
  // single BOS-like seed token is then required so the branch has an input -
  // pass at least one token (e.g. the conditional prompt's first token, or a
  // dedicated BOS id) as the unconditional context.
  // Coded by Claude (AI).
  TNNetCFGProcessor = class(TNNetLogitsProcessor)
    private
      FSession: TNNetStreamingDecoder;
      FOwnsSession: boolean;
      FNegPrompt: TNeuralIntegerArray;
      FInV: TNNetVolume;
      FPendingToken: integer; // next token to feed the uncond branch
      FAbsPos: integer;       // absolute position of FPendingToken
      FGuidanceScale: TNeuralFloat;
    public
      // UncondSession: a WIDTH-1 streaming session for the unconditional
      // branch. NegativePrompt: the unconditional/negative prompt token ids
      // (must be non-empty - it seeds the branch's first input). GuidanceScale
      // = 1 is a no-op; = 0 ignores the conditional prompt.
      constructor Create(UncondSession: TNNetStreamingDecoder;
        const NegativePrompt: array of integer; GuidanceScale: TNeuralFloat;
        OwnsSession: boolean = false);
      destructor Destroy(); override;
      procedure Reset(const PromptTokens: array of integer); override;
      procedure ProcessRow(Row: TNNetVolume); override;
      procedure Commit(TokenId: integer); override;
      property GuidanceScale: TNeuralFloat
        read FGuidanceScale write FGuidanceScale;
  end;

  { TGenerationConfig }
  // One-record bundle of generation knobs - the GenerationConfig counterpart
  // of the parameter piles on the GenerateTokensStreamed overloads, consumed
  // by GenerateTokensWithConfig / GenerateStringWithConfig. Build it with
  // DefaultGenerationConfig (every knob "off") and set what you need. NONE of
  // the referenced objects are owned by the record - the caller frees them.
  // The implied per-step pipeline is:
  //   Penalty -> Temperature -> Processors (extra, in order) -> Constraint
  //   -> Sampler
  // (the constraint runs LAST so its structural guarantees cannot be broken
  // by a later transform). Every field at its default reproduces the plain
  // GenerateTokensStreamed/GenerateStringStreamed paths bit-for-bit.
  // Coded by Claude (AI).
  TGenerationConfig = record
    // Stopping criteria.
    MaxNewTokens: integer;   // cap on generated tokens
    MaxTotalLen: integer;    // prompt+generated ceiling; 0 = prompt+MaxNewTokens
    // Token ids that terminate generation (the emitted EOS is stored and
    // counted, as everywhere). EMPTY = the codebase default "token < 2" rule.
    EOSTokens: TNeuralIntegerArray;
    StopSequences: TNNetTokenSequences; // matched in the generated region
    StopStrings: array of string;       // string paths only; tokenized + scanned
    // Distribution pipeline (see order above). Temperature 1.0 = off.
    Temperature: TNeuralFloat;
    Penalty: TNNetTokenHistoryPenalty;     // nil = off (not owned)
    Processors: TNNetLogitsProcessorChain; // nil = none (not owned)
    Constraint: TNNetTokenConstraint;      // nil = off (not owned)
    // Sampler reading the processed probability row; nil = greedy argmax.
    Sampler: TNNetSamplerBase;             // not owned
    // CLASSIFIER-FREE GUIDANCE (CFG). GuidanceScale = 1.0 is OFF (the no-op
    // default): the conditional distribution is used untouched and CFGUncond
    // is ignored, so the path is bit-for-bit the non-CFG path. When
    // GuidanceScale <> 1.0 the config wires a TNNetCFGProcessor at the FRONT
    // of the per-step pipeline (it combines the two model distributions BEFORE
    // penalty/temperature/processors/constraint see the row). CFGUncond is the
    // WIDTH-1 unconditional streaming twin (typically derived with
    // MakeUnconditionalTwin from the same imported net; not owned by the
    // record). NegativePrompt is the unconditional/negative prompt token ids;
    // it must be non-empty when GuidanceScale <> 1.0 (it seeds the
    // unconditional branch - pass a single BOS-like id for "no prompt").
    GuidanceScale: TNeuralFloat;           // 1.0 = off
    CFGUncond: TNNetStreamingDecoder;      // unconditional twin (not owned)
    NegativePrompt: TNeuralIntegerArray;   // negative/unconditional prompt ids
    // Guidance-style TOKEN HEALING (string paths only - the token-level
    // routines have no dict; call PrepareTokenHealing yourself there): drop
    // the last prompt token and constrain the FIRST generated token to
    // vocabulary entries whose text extends the dropped token's text. A
    // no-op when healing is not applicable (1-token prompt, empty/unknown
    // last-token text, or no strict extension exists in the vocabulary).
    TokenHealing: boolean;
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
  // Temperature clamp shared by every temperature-taking decode path
  // (DecodeSeq2SeqSampled and TNNetTemperatureProcessor): Temperature -> 0
  // degenerates to greedy argmax instead of dividing by zero.
  csDecodeMinTemperature = 1e-6;

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

// DIVERSE beam search (Vijayakumar et al. 2018, "Diverse Beam Search"). The
// BeamWidth beams are partitioned into NumGroups equal groups of size
// BeamWidth div NumGroups; the groups are expanded one after another within a
// step, and when scoring group g (g>0) each candidate token's length-penalised
// score has a HAMMING DIVERSITY PENALTY subtracted:
//     score'(t) = score(t) - Diversity * (count of token t already chosen by
//                                         the earlier groups 0..g-1 this step)
// so later groups are pushed away from tokens the earlier groups already took,
// giving a diverse set of completions instead of B near-duplicates.
// DEGRADE-TO-BEAM invariant: NumGroups=1 (a single group) with Diversity=0 is
// BIT-FOR-BIT the ordinary DecodeBeamSearchAll(NN,Prompt,MaxLen,BeamWidth,..);
// Diversity=0 with any NumGroups removes the penalty (groups then differ only
// through the per-group width split). Returns the whole final ranked beam
// (finished + surviving), sorted best-first, exactly like DecodeBeamSearchAll.
//   NumGroups : G, the number of diversity groups (clamped to [1, BeamWidth]).
//   Diversity : lambda >= 0, the per-collision Hamming penalty (0 = none).
// Coded by Claude (AI).
function DecodeDiverseBeamSearchAll(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer; NumGroups: integer;
  Diversity: TNeuralFloat;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;

// Single-best diverse beam search: the highest-Score result of
// DecodeDiverseBeamSearchAll (same relation as DecodeBeamSearch to
// DecodeBeamSearchAll). Coded by Claude (AI).
function DecodeDiverseBeamSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer; NumGroups: integer;
  Diversity: TNeuralFloat;
  LengthPenalty: TNeuralFloat): TNNetDecodeResult;

// CONSTRAINED beam search with FORCE-WORDS (force_words_ids; Anderson et al.
// 2017 constrained beam search / HF PhrasalConstraint banking). Guarantees
// that EVERY one of the required phrases in ForceTokens APPEARS somewhere in
// the generated output (a phrase is a token sequence; here token id =
// character code, so each phrase is a string). Unlike TNNetTokenConstraint
// (which constrains a PREFIX / per-step allowed set) this forces phrases to
// occur ANYWHERE in the completion, even when an unconstrained beam would
// never emit them.
// V1 GUARANTEE ("force completion before EOS"): each hypothesis tracks its
// progress through every required phrase (how many leading characters of each
// phrase it has emitted as a contiguous run). A beam may only emit EOS / be
// counted finished once ALL phrases have been fully emitted; while phrases
// remain unmet, EOS is blocked and the beam is additionally expanded with the
// next needed phrase character so a satisfying continuation is always present
// in the candidate pool. The standard length-penalised beam ranking then picks
// the best satisfying completion. With ForceTokens empty the routine is
// BIT-FOR-BIT DecodeBeamSearchAll.
//   ForceTokens : the required phrases (strings); empty = ordinary beam search.
// Returns the whole final ranked beam, sorted best-first.
// Coded by Claude (AI).
function DecodeConstrainedBeamSearchAll(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  const ForceTokens: array of string;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;

// Single-best constrained beam search (highest-Score result of
// DecodeConstrainedBeamSearchAll). Coded by Claude (AI).
function DecodeConstrainedBeamSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  const ForceTokens: array of string;
  LengthPenalty: TNeuralFloat): TNNetDecodeResult;

// CONTRASTIVE SEARCH decoding (Su et al. 2022, "A Contrastive Framework for
// Neural Text Generation"; the transformers `penalty_alpha` decoder). Each
// step re-ranks the TopK most probable next-token candidates by
//   score(v) = (1 - PenaltyAlpha) * p(v)
//            - PenaltyAlpha * max_{j<t} cos_sim( h_v , h_j )
// where p(v) is the model probability of candidate v, h_v is the LAST hidden
// state (the input to the final projection / softmax layer) the model produces
// when v is appended, and {h_j} are the hidden states of all previously
// processed tokens (the "degeneration penalty" context). The candidate with
// the highest score is emitted; this favours fluent continuations while
// penalising tokens whose representation collapses onto the existing context
// (the cause of greedy degeneration / repetition).
//   MaxLen       : maximum number of generated tokens (excludes the prompt).
//   TopK         : candidate-set size (Su et al. use 4..8); TopK<=1 is greedy.
//   PenaltyAlpha : the degeneration penalty in [0,1]. PenaltyAlpha=0 degrades
//                  EXACTLY to greedy argmax over the SAME TopK candidates
//                  (i.e. plain greedy argmax, since the top-prob candidate is
//                  always in the TopK set). PenaltyAlpha=1 ranks purely by the
//                  similarity penalty.
// Char-level, same forward-pass / encoding convention as DecodeGreedy (token
// id = character code). The hidden state is read from the layer feeding the
// final projection via an extra forward of each candidate continuation; no
// KV-cache is used (v1, mirrors the char-level DecodeGreedy re-encode loop).
// Coded by Claude (AI).
function DecodeContrastiveSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; TopK: integer; PenaltyAlpha: TNeuralFloat;
  const StopStrings: array of string): TNNetDecodeResult;

// DoLa DECODING (Decoding by Contrasting Layers, Chuang et al. 2023; the
// transformers `generate(dola_layers=...)` path). Improves FACTUALITY (NOT
// speed - distinct from early-exit, and distinct from contrastive SEARCH which
// contrasts against context tokens: DoLa contrasts LAYERS). At each step the
// FINAL distribution p_final is contrasted against a PREMATURE-layer
// distribution p_premature, both read through the SAME LM head:
//   1. A full forward gives p_final (the mature distribution).
//   2. The "logits at layer k" splice (the frozen-body idiom from
//      TNNet.LogitLensReport) takes each candidate premature layer's activation,
//      copies it into the head-input slot, and recomputes ONLY the head
//      sub-stack -> p_premature for that layer.
//   3. The premature layer is chosen PER STEP as the one whose distribution has
//      the MAXIMUM Jensen-Shannon divergence from p_final, over the fixed
//      candidate-layer bucket (v1: every earlier layer whose Output.Size matches
//      the head-input size; HeadInIdx itself is excluded - contrasting against
//      itself is a no-op).
//   4. The next token is argmax over the ADAPTIVE head-candidate set
//      V_head = { t : p_final[t] >= Alpha * max_j p_final[j] } of the contrast
//      score  log p_final[t] - log p_premature[t]  (tokens outside V_head are
//      masked out, exactly as in the paper's APC step).
// DEGRADE-TO-GREEDY invariant: Alpha <= 0 (or an EMPTY candidate-layer bucket,
// e.g. a net with no lens-compatible earlier layer) disables the contrast and
// reproduces plain greedy argmax BIT-FOR-BIT (the same net's DecodeGreedy).
//   MaxLen        : maximum number of generated tokens (excludes the prompt).
//   Alpha         : head-set fraction in [0,1]; the paper uses ~0.1. Alpha<=0
//                   degrades exactly to greedy. Alpha=1 keeps only the
//                   final-argmax token (also greedy, but via the contrast path).
//   HeadStartIdx  : first layer of the LM head sub-stack; -1 auto-detects it as
//                   the last trainable layer (same heuristic as LogitLensReport).
// Char-level, same forward-pass / encoding convention as DecodeGreedy /
// DecodeContrastiveSearch (token id = character code); no KV-cache (v1).
// Coded by Claude (AI).
function DecodeDoLa(NN: TNNet; const Prompt: string;
  MaxLen: integer; Alpha: TNeuralFloat;
  const StopStrings: array of string;
  HeadStartIdx: integer = -1): TNNetDecodeResult; overload;

// DoLa with explicit DoLa-low / DoLa-high candidate-layer bucket selection
// (the paper's dynamic premature-layer pools). Bucket=dlbFull is identical to
// the call above. dlbLow restricts the per-step max-JS premature-layer search
// to the SHALLOW half of the lens-compatible layers, dlbHigh to the DEEP half;
// an empty resulting bucket degrades to greedy exactly as Alpha<=0 does.
// Coded by Claude (AI).
function DecodeDoLa(NN: TNNet; const Prompt: string;
  MaxLen: integer; Alpha: TNeuralFloat;
  const StopStrings: array of string;
  Bucket: TDoLaLayerBucket;
  HeadStartIdx: integer = -1): TNNetDecodeResult; overload;

// SAMPLED char-level decode: the stochastic sibling of DecodeGreedy. Each step
// draws the next token from the softmax row via Sampler.GetToken (Sampler = nil
// -> greedy argmax, bit-identical to DecodeGreedy). Stop strings are trimmed as
// in DecodeGreedy. Result.SumLogProb sums the chosen tokens' log-probs so the
// completion is directly rerankable (the Best-of-N building block). The caller
// seeds the RNG (RandSeed) for reproducibility. Coded by Claude (AI).
function DecodeSampled(NN: TNNet; const Prompt: string; MaxLen: integer;
  Sampler: TNNetSamplerBase;
  const StopStrings: array of string): TNNetDecodeResult;

// BEST-OF-N reranking (the canonical test-time-compute baseline). Draws N
// sampled completions (DecodeSampled with the given Sampler; the caller seeds
// RandSeed) and returns the single best.
//   * Default ranking is by LENGTH-NORMALIZED sequence log-prob:
//       Score = SumLogProb / LengthPenaltyDenominator(Length(Text), LengthPenalty)
//     (LengthPenalty=0 -> raw sum-log-prob; >0 lifts longer completions, Wu et
//     al.). Each returned candidate carries this Score.
//   * When Scorer <> nil the EXTERNAL scorer decides instead: the candidate
//     with the maximum Scorer(Prompt, Text) is returned (the Bradley-Terry
//     reward-model consumer). Score is overwritten with the scorer's value.
// N < 1 is clamped to 1; ties keep the FIRST (lowest-index) candidate.
// Coded by Claude (AI).
function DecodeBestOfN(NN: TNNet; const Prompt: string; MaxLen, N: integer;
  Sampler: TNNetSamplerBase; LengthPenalty: TNeuralFloat;
  const StopStrings: array of string;
  Scorer: TNNetSequenceScorer = nil): TNNetDecodeResult;

// Like DecodeBestOfN but returns ALL N candidates (in draw order, unsorted), so
// callers can inspect/rerank them externally. Each carries its length-normalized
// Score (or, when Scorer<>nil, the scorer value). Coded by Claude (AI).
function SampleNCompletions(NN: TNNet; const Prompt: string; MaxLen, N: integer;
  Sampler: TNNetSamplerBase; LengthPenalty: TNeuralFloat;
  const StopStrings: array of string;
  Scorer: TNNetSequenceScorer = nil): TNNetDecodeResultArray;

// SELF-CONSISTENCY (Wang et al. 2022): draw N sampled completions, extract each
// one's ANSWER via Extract, and return the MODAL (majority-vote) answer string.
// Ties are broken toward the answer that FIRST reached the winning count (the
// earliest-drawn modal answer). Candidates whose extracted answer is empty are
// ignored; if none yields an answer the empty string is returned. The caller
// seeds RandSeed. Coded by Claude (AI).
function DecodeSelfConsistency(NN: TNNet; const Prompt: string; MaxLen, N: integer;
  Sampler: TNNetSamplerBase; Extract: TNNetAnswerExtractor;
  const StopStrings: array of string): string;

// PROMPT-LOOKUP / N-GRAM SPECULATIVE DECODING (Saxena 2023, "prompt lookup
// decoding"; the transformers `prompt_lookup_num_tokens` path). A TRAINING-FREE,
// NO-second-model speculative decode: the draft of the next few tokens is
// produced by a pure STRING LOOKUP instead of a draft network.
//   * Each step, the last MatchLen characters of the running context (prompt +
//     text generated so far) are matched against EARLIER occurrences of that
//     same MatchLen-gram in the context. The MOST RECENT earlier occurrence is
//     preferred; the NumDraft characters that FOLLOWED that occurrence become
//     the speculative draft.
//   * The draft is then VERIFIED against the model with the same width-K greedy
//     verify rule used by speculative decoding: walking the draft left-to-right,
//     a drafted character is ACCEPTED iff it equals the model's greedy argmax at
//     that position; the first mismatch (or draft exhaustion) ends the window.
//     The model's own argmax at the first rejected position is then emitted, so
//     EVERY emitted character is exactly the greedy argmax - acceptance is a
//     SPEEDUP, never a quality change.
//   * DEGRADE-TO-GREEDY GUARANTEE: when no MatchLen-gram match is found (or the
//     whole draft is rejected) exactly one token is emitted, identical to plain
//     DecodeGreedy. The full output is therefore BIT-FOR-BIT DecodeGreedy on the
//     same net for any MatchLen / NumDraft (only the number of forward passes
//     differs). This is the standout win for repetition-heavy NLP (RAG,
//     summarization, "quote the passage").
//   MaxLen   : maximum number of generated tokens (excludes the prompt).
//   MatchLen : length of the suffix n-gram matched into the context (>=1).
//   NumDraft : how many following characters to copy as the draft (>=1).
// Char-level, same forward-pass / encoding convention as DecodeGreedy (token id
// = character code); no KV-cache (v1, re-encode loop), so the verify window is
// checked one prefix at a time - the lookup itself is free. Coded by Claude (AI).
function DecodePromptLookup(NN: TNNet; const Prompt: string;
  MaxLen: integer; MatchLen: integer; NumDraft: integer;
  const StopStrings: array of string): TNNetDecodeResult;

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

// GenerateTokensStreamed with a TEMPERATURE knob - the causal-LM counterpart
// of DecodeSeq2SeqSampled's Temperature argument, with the same clamp
// (>= csDecodeMinTemperature) and the same Temperature -> 0 greedy
// degeneration. The streamed row holds POST-SOFTMAX probabilities, so the
// temperature is applied in the probability domain (p^(1/T), renormalized -
// the exact image of dividing the logits by T; see
// TNNetTemperatureProcessor). Pipeline order: Penalty -> Temperature ->
// Constraint -> Sampler. Temperature = 1.0 is bit-for-bit the overload above.
function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  const StopSequences: TNNetTokenSequences;
  Constraint: TNNetTokenConstraint;
  Temperature: TNeuralFloat): integer; overload;

// GenerateStringStreamed with a TEMPERATURE knob: forwards Temperature to the
// token core (see the GenerateTokensStreamed temperature overload).
// Temperature = 1.0 is bit-for-bit the overload above.
function GenerateStringStreamed(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase; Penalty: TNNetTokenHistoryPenalty;
  const StopStrings: array of string;
  Constraint: TNNetTokenConstraint;
  Temperature: TNeuralFloat): string; overload;

// THE CHAIN-DRIVEN STREAMED CORE every GenerateTokensStreamed overload above
// delegates to (so a nil/empty chain is bit-for-bit the plain path by
// construction). Processors (may be nil; pass a TNNetLogitsProcessorChain
// for several) transforms each step's POST-SOFTMAX probability row IN ORDER
// before the sampler/argmax reads it; it must declare ExpectsProbabilities
// (EArgumentException otherwise). Reset(prompt) is called once and
// Commit(token) after every emitted token. EOSTokens lists the terminating
// token ids; EMPTY = the codebase default "token < 2" rule (the emitted EOS
// is stored and counted either way). Other arguments as in the overloads
// above.
// Coded by Claude (AI).
function GenerateTokensStreamedWithProcessors(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Processors: TNNetLogitsProcessor;
  const StopSequences: TNNetTokenSequences;
  const EOSTokens: TNeuralIntegerArray): integer;

// STRING-LEVEL chain-driven core (the GenerateStringStreamed counterpart of
// GenerateTokensStreamedWithProcessors, and the core every string overload
// delegates to). StopStrings are tokenized for early termination plus the
// string-level truncation safety net; ExtraStopSequences (may be nil) are
// token-id stop sequences appended to the tokenized stop strings; EOSTokens
// as in the token core (also bounds the displayed continuation).
// Coded by Claude (AI).
function GenerateStringStreamedWithProcessors(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase; Processors: TNNetLogitsProcessor;
  const StopStrings: array of string;
  const ExtraStopSequences: TNNetTokenSequences;
  const EOSTokens: TNeuralIntegerArray;
  TokenHealing: boolean = false): string;

// TOKEN HEALING SET-UP for the token-level routines: when healing applies,
// returns a one-shot TNNetTokenHealingConstraint allowing exactly the
// vocabulary entries whose text has the LAST prompt token's text as a
// PREFIX (the dropped token itself included), decrements PromptLen (the
// dropped token stays in Tokens and is simply overwritten by the first
// generated token) and reports the dropped id in DroppedToken. Returns nil
// - and leaves PromptLen untouched - when healing is not applicable:
// PromptLen < 2 (healing would empty the prompt), the last id is outside
// the dict, its text is empty, or NO vocabulary entry STRICTLY extends it
// (the healed run would provably equal the unhealed one). Pass the
// constraint to any Constraint-accepting overload (the caller frees it).
// Linear vocab scan (v1) - fine for the dict sizes the streamed paths use.
// Coded by Claude (AI).
function PrepareTokenHealing(Dict: TStringListInt;
  const Tokens: TNeuralIntegerArray; var PromptLen: integer;
  out DroppedToken: integer): TNNetTokenHealingConstraint;

// A TGenerationConfig with every knob OFF: greedy (nil sampler),
// Temperature 1.0, no penalties/processors/constraint, no stop
// sequences/strings, default EOS rule, MaxTotalLen 0 (= prompt +
// MaxNewTokens). Driving generation with this config reproduces the plain
// GenerateTokensStreamed/GenerateStringStreamed paths bit-for-bit.
function DefaultGenerationConfig(MaxNewTokens: integer;
  MaxTotalLen: integer = 0): TGenerationConfig;

// CONFIG-DRIVEN GENERATION: assembles the pipeline described in
// TGenerationConfig (Penalty -> Temperature -> Processors -> Constraint ->
// Sampler) and runs the chain-driven streamed core. Equivalent to the
// hand-assembled GenerateTokensStreamed calls; see TGenerationConfig for
// ownership and defaults.
function GenerateTokensWithConfig(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen: integer;
  const Config: TGenerationConfig): integer;

// String-level config-driven generation (uses Config.StopStrings AND
// Config.StopSequences; otherwise as GenerateTokensWithConfig).
function GenerateStringWithConfig(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string;
  const Config: TGenerationConfig): string;

// CFG CONVENIENCE: derive the UNCONDITIONAL streaming twin automatically from
// a single (already WIDTH-1) trained net, so CFG callers do not hand-build a
// second branch. SourceWidth1Net must be the same width-1 net the conditional
// streaming loop drives (e.g. an imported inference model, or the twin built
// for the conditional TNNetStreamingDecoder). The function CLONES it through
// SaveToString -> LoadFromString (architecture AND weights survive the
// round-trip, and the width-1 input shape is preserved exactly), giving the
// CFG branch its OWN net with INDEPENDENT KV-cache / SSM state but the SAME
// trained weights - the conditional and unconditional branches must not share
// one cache. The clone is returned in TwinNet and wrapped in a fresh
// TNNetStreamingDecoder (cache budget MaxTotalLen) returned as the result.
//
// OWNERSHIP: the CALLER owns and frees BOTH the returned session AND TwinNet
// (free the session first). A TNNetCFGProcessor created with this session and
// OwnsSession=false leaves them for the caller; with OwnsSession=true the
// processor frees the session but NOT TwinNet (the session never owns its
// net), so the caller still frees TwinNet.
//
// A direct full-width->width-1 clone is intentionally NOT attempted: a TNNet
// cannot be reshaped to a different input width by a string round-trip (the
// saved structure carries the original width). Build the width-1 net once
// (the established Build*(1)+CopyWeights idiom) and pass it here.
// Coded by Claude (AI).
function MakeUnconditionalTwin(SourceWidth1Net: TNNet;
  MaxTotalLen: integer; out TwinNet: TNNet): TNNetStreamingDecoder;

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

// ---------------------------------------------------------------------------
// NEEDLE-IN-A-HAYSTACK long-context eval harness.
//
// For each (ContextLengths[c], DepthFractions[d]) cell the harness:
//   1. builds Filler(ContextLengths[c]) characters of distractor text;
//   2. splices NeedleFact in at byte position round(DepthFraction*len) snapped
//      to a space boundary (depth 0 = very start, 1 = very end);
//   3. appends Question, yielding the probe Prompt;
//   4. calls Generate(Prompt) and records Hit := the output contains
//      NeedleAnswer (case-insensitive substring match);
// then renders a depth x length pass/fail grid plus an overall accuracy line.
//
// The callbacks carry an opaque Data pointer (e.g. a TNNet, a sampler, or a
// test stand-in's state) so no globals are needed. NeedleFact should embed the
// answer; NeedleAnswer is the token actually looked for in the output;
// Question is the retrieval prompt that triggers recall. Empty axis arrays
// yield an empty grid with Accuracy 0. Returns the full grid; .Report holds
// the rendered string (also the function-style overload's Result).
// Coded by Claude (AI).
function NeedleInHaystackReport(
  const ContextLengths: array of integer;
  const DepthFractions: array of TNeuralFloat;
  const NeedleFact, NeedleAnswer, Question: string;
  Filler: TNeedleFillerCallback;
  Generate: TNeedleGenerateCallback;
  Data: Pointer): TNeedleInHaystackResult; overload;

// Convenience overload that drives a real char-level TNNet through DecodeGreedy
// (MaxLen generated chars) and a built-in repeating-lorem filler. The needle
// answer match is the same case-insensitive substring test. This is the path
// used to evaluate RoPE-scaling / KV-cache-eviction on a TinyStories-scale
// char model the repo can run on CPU.
// Coded by Claude (AI).
function NeedleInHaystackReport(NN: TNNet;
  const ContextLengths: array of integer;
  const DepthFractions: array of TNeuralFloat;
  const NeedleFact, NeedleAnswer, Question: string;
  MaxLen: integer): TNeedleInHaystackResult; overload;

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

{ TNNetTokenHealingConstraint }

constructor TNNetTokenHealingConstraint.Create(
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
  FDone := false;
end;

procedure TNNetTokenHealingConstraint.Reset(
  const PromptTokens: array of integer);
begin
  FDone := false; // re-arm: a fresh generation heals its first step again
end;

function TNNetTokenHealingConstraint.TokenAllowed(TokenId: integer): boolean;
begin
  // After the first emitted token the constraint is lifted entirely.
  Result := FDone or ((TokenId >= 0) and (TokenId <= High(FAllowed)) and
    FAllowed[TokenId]);
end;

procedure TNNetTokenHealingConstraint.Commit(TokenId: integer);
begin
  FDone := true;
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

{ TNNetGrammar }

const
  // Packed position = Rule * KGrammarStride + ElementIndex. A rule body of more
  // than KGrammarStride elements would collide; far beyond any practical GBNF.
  KGrammarStride = 1000000;

constructor TNNetGrammar.Create(const GBNFText: string);
begin
  inherited Create();
  FSource := GBNFText;
  FRuleNames := TStringList.Create();
  FRuleNames.CaseSensitive := true;
  FRootRule := -1;
  Compile();
end;

destructor TNNetGrammar.Destroy();
begin
  FRuleNames.Free;
  inherited Destroy();
end;

function TNNetGrammar.PeekCh(): char;
begin
  if FPos <= Length(FSource) then Result := FSource[FPos] else Result := #0;
end;

procedure TNNetGrammar.NextCh();
begin
  Inc(FPos);
end;

function TNNetGrammar.AtEnd(): boolean;
begin
  Result := FPos > Length(FSource);
end;

procedure TNNetGrammar.SkipWS();
// Whitespace AND '#' line comments between grammar tokens.
begin
  while not AtEnd() do
  begin
    if (PeekCh() = ' ') or (PeekCh() = #9) or (PeekCh() = #13) or
       (PeekCh() = #10) then NextCh()
    else if PeekCh() = '#' then
      while (not AtEnd()) and (PeekCh() <> #10) do NextCh()
    else break;
  end;
end;

procedure TNNetGrammar.ExpectCh(C: char);
begin
  if PeekCh() <> C then
    raise EAssertionFailed.Create('TNNetGrammar: expected ''' + C +
      ''' at offset ' + IntToStr(FPos) + ' in grammar');
  NextCh();
end;

function TNNetGrammar.AddRule(const Name: string): integer;
begin
  Result := FRuleNames.Add(Name);
  SetLength(FRules, Length(FRules) + 1);
end;

function TNNetGrammar.FindOrAddRule(const Name: string): integer;
begin
  Result := FRuleNames.IndexOf(Name);
  if Result < 0 then Result := AddRule(Name);
end;

function TNNetGrammar.NewAnonRule(): integer;
begin
  Result := AddRule('__anon' + IntToStr(Length(FRules)));
end;

function TNNetGrammar.AddCharSet(
  const ARanges: array of TNNetGrammarRange): integer;
var
  I, Base: integer;
begin
  Base := Length(FRanges);
  SetLength(FRanges, Base + Length(ARanges));
  for I := 0 to High(ARanges) do FRanges[Base + I] := ARanges[I];
  Result := Length(FCharSets);
  SetLength(FCharSets, Result + 1);
  FCharSets[Result].First := Base;
  FCharSets[Result].Count := Length(ARanges);
end;

// Appends one element to a growing element array.
procedure GrammarAppendElem(var Elems: TNNetGrammarElemArray;
  AType: TNNetGrammarElemType; AValue: integer);
begin
  SetLength(Elems, Length(Elems) + 1);
  Elems[High(Elems)].ElemType := AType;
  Elems[High(Elems)].Value := AValue;
end;

function TNNetGrammar.ParseCharClass(out Negated: boolean): integer;
// Current char is '['. Parses up to and including ']'. Handles '[^...]'.
var
  Ranges: array of TNNetGrammarRange;
  Lo, Hi: char;

  function ReadClassChar(): char;
  begin
    if PeekCh() = '\' then
    begin
      NextCh();
      case PeekCh() of
        'n': Result := #10;
        'r': Result := #13;
        't': Result := #9;
        else Result := PeekCh(); // \\ \] \^ \- \" and any other: literal
      end;
      NextCh();
    end
    else
    begin
      Result := PeekCh();
      NextCh();
    end;
  end;

begin
  ExpectCh('[');
  Negated := false;
  if PeekCh() = '^' then
  begin
    Negated := true;
    NextCh();
  end;
  SetLength(Ranges, 0);
  while (not AtEnd()) and (PeekCh() <> ']') do
  begin
    Lo := ReadClassChar();
    Hi := Lo;
    if (PeekCh() = '-') and (FPos + 1 <= Length(FSource)) and
       (FSource[FPos + 1] <> ']') then
    begin
      NextCh(); // consume '-'
      Hi := ReadClassChar();
    end;
    SetLength(Ranges, Length(Ranges) + 1);
    Ranges[High(Ranges)].Lo := Lo;
    Ranges[High(Ranges)].Hi := Hi;
  end;
  ExpectCh(']');
  Result := AddCharSet(Ranges);
end;

// opt ::= atom | <empty>
procedure TNNetGrammar.BuildOptRule(AtomRule: integer;
  var Elems: TNNetGrammarElemArray);
var
  Body: TNNetGrammarElemArray;
  R: integer;
begin
  R := NewAnonRule();
  SetLength(Body, 0);
  GrammarAppendElem(Body, getRuleRef, AtomRule);
  GrammarAppendElem(Body, getAlt, 0);
  GrammarAppendElem(Body, getEnd, 0); // empty alternate
  FRules[R] := Body;
  GrammarAppendElem(Elems, getRuleRef, R);
end;

// star ::= atom star | <empty>
procedure TNNetGrammar.BuildStarRule(AtomRule: integer;
  var Elems: TNNetGrammarElemArray);
var
  Body: TNNetGrammarElemArray;
  R: integer;
begin
  R := NewAnonRule();
  SetLength(Body, 0);
  GrammarAppendElem(Body, getRuleRef, AtomRule);
  GrammarAppendElem(Body, getRuleRef, R); // self-reference -> repetition
  GrammarAppendElem(Body, getAlt, 0);
  GrammarAppendElem(Body, getEnd, 0);
  FRules[R] := Body;
  GrammarAppendElem(Elems, getRuleRef, R);
end;

// plus ::= atom plus | atom
procedure TNNetGrammar.BuildPlusRule(AtomRule: integer;
  var Elems: TNNetGrammarElemArray);
var
  Body: TNNetGrammarElemArray;
  R: integer;
begin
  R := NewAnonRule();
  SetLength(Body, 0);
  GrammarAppendElem(Body, getRuleRef, AtomRule);
  GrammarAppendElem(Body, getRuleRef, R);
  GrammarAppendElem(Body, getAlt, 0);
  GrammarAppendElem(Body, getRuleRef, AtomRule);
  GrammarAppendElem(Body, getEnd, 0);
  FRules[R] := Body;
  GrammarAppendElem(Elems, getRuleRef, R);
end;

procedure TNNetGrammar.ParseElement(var Elems: TNNetGrammarElemArray);
var
  Inner: TNNetGrammarElemArray;
  AnonRule, SetIdx, RefRule: integer;
  Negated: boolean;
  StartLen, I: integer;
  Name: string;
  Postfix: char;
begin
  SkipWS();
  StartLen := Length(Elems);
  case PeekCh() of
    '"':
      begin
        NextCh();
        while (not AtEnd()) and (PeekCh() <> '"') do
        begin
          if PeekCh() = '\' then
          begin
            NextCh();
            case PeekCh() of
              'n': GrammarAppendElem(Elems, getChar, Ord(#10));
              'r': GrammarAppendElem(Elems, getChar, Ord(#13));
              't': GrammarAppendElem(Elems, getChar, Ord(#9));
              else GrammarAppendElem(Elems, getChar, Ord(PeekCh()));
            end;
            NextCh();
          end
          else
          begin
            GrammarAppendElem(Elems, getChar, Ord(PeekCh()));
            NextCh();
          end;
        end;
        ExpectCh('"');
      end;
    '[':
      begin
        SetIdx := ParseCharClass(Negated);
        if Negated
        then GrammarAppendElem(Elems, getCharSetNot, SetIdx)
        else GrammarAppendElem(Elems, getCharSet, SetIdx);
      end;
    '.':
      begin
        NextCh();
        GrammarAppendElem(Elems, getCharAny, 0);
      end;
    '(':
      begin
        NextCh();
        SetLength(Inner, 0);
        ParseAlternates(Inner);
        SkipWS();
        ExpectCh(')');
        AnonRule := NewAnonRule();
        GrammarAppendElem(Inner, getEnd, 0);
        FRules[AnonRule] := Inner;
        GrammarAppendElem(Elems, getRuleRef, AnonRule);
      end;
    'a'..'z', 'A'..'Z', '_':
      begin
        Name := '';
        while (not AtEnd()) and
          (PeekCh() in ['a'..'z', 'A'..'Z', '0'..'9', '_', '-']) do
        begin
          Name := Name + PeekCh();
          NextCh();
        end;
        RefRule := FindOrAddRule(Name);
        GrammarAppendElem(Elems, getRuleRef, RefRule);
      end;
    else
      raise EAssertionFailed.Create('TNNetGrammar: unexpected ''' + PeekCh() +
        ''' at offset ' + IntToStr(FPos));
  end;

  // Postfix repetition wraps the just-parsed atom (elements StartLen..end).
  SkipWS();
  Postfix := PeekCh();
  if (Postfix = '*') or (Postfix = '+') or (Postfix = '?') then
  begin
    NextCh();
    AnonRule := NewAnonRule();
    SetLength(Inner, 0);
    for I := StartLen to High(Elems) do
      GrammarAppendElem(Inner, Elems[I].ElemType, Elems[I].Value);
    SetLength(Elems, StartLen); // drop the atom; replaced by the wrapper ref
    GrammarAppendElem(Inner, getEnd, 0);
    FRules[AnonRule] := Inner;
    case Postfix of
      '?': BuildOptRule(AnonRule, Elems);
      '*': BuildStarRule(AnonRule, Elems);
      '+': BuildPlusRule(AnonRule, Elems);
    end;
  end;
end;

procedure TNNetGrammar.ParseSequence(var Elems: TNNetGrammarElemArray);
begin
  SkipWS();
  while (not AtEnd()) and (PeekCh() <> '|') and (PeekCh() <> ')') do
  begin
    ParseElement(Elems);
    SkipWS();
  end;
end;

procedure TNNetGrammar.ParseAlternates(var Elems: TNNetGrammarElemArray);
begin
  ParseSequence(Elems);
  SkipWS();
  while PeekCh() = '|' do
  begin
    NextCh();
    GrammarAppendElem(Elems, getAlt, 0);
    ParseSequence(Elems);
    SkipWS();
  end;
end;

procedure TNNetGrammar.Compile();
// Splits the source into 'name ::= body' definitions (continuation lines fold
// into the previous def), pre-registers names so forward refs resolve, parses
// each body, then resolves 'root'.
var
  Lines, Defs: TStringList;
  I, J, ArrowPos, R: integer;
  RawLine, Name, Body, FullBody: string;
  LocalBody: TNNetGrammarElemArray;

  function IsDefLine(const S: string): boolean;
  var
    K: integer;
    T: string;
  begin
    T := TrimLeft(S);
    Result := false;
    if T = '' then exit;
    if not (T[1] in ['a'..'z', 'A'..'Z', '_']) then exit;
    K := 1;
    while (K <= Length(T)) and
      (T[K] in ['a'..'z', 'A'..'Z', '0'..'9', '_', '-']) do Inc(K);
    while (K <= Length(T)) and (T[K] = ' ') do Inc(K);
    Result := (K + 2 <= Length(T)) and (Copy(T, K, 3) = '::=');
  end;

begin
  Defs := TStringList.Create();
  Lines := TStringList.Create();
  try
    Lines.Text := FSource;
    FullBody := '';
    for I := 0 to Lines.Count - 1 do
    begin
      RawLine := Lines[I];
      J := Pos('#', RawLine);
      if J > 0 then RawLine := Copy(RawLine, 1, J - 1);
      if IsDefLine(RawLine) then
      begin
        if FullBody <> '' then Defs.Add(FullBody);
        FullBody := RawLine;
      end
      else if Trim(RawLine) <> '' then
        FullBody := FullBody + ' ' + RawLine;
    end;
    if FullBody <> '' then Defs.Add(FullBody);

    for I := 0 to Defs.Count - 1 do
    begin
      RawLine := TrimLeft(Defs[I]);
      ArrowPos := Pos('::=', RawLine);
      Name := Trim(Copy(RawLine, 1, ArrowPos - 1));
      FindOrAddRule(Name);
    end;

    for I := 0 to Defs.Count - 1 do
    begin
      RawLine := TrimLeft(Defs[I]);
      ArrowPos := Pos('::=', RawLine);
      Name := Trim(Copy(RawLine, 1, ArrowPos - 1));
      Body := Copy(RawLine, ArrowPos + 3, Length(RawLine));
      R := FindOrAddRule(Name);
      FSource := Body;   // parse this body in isolation
      FPos := 1;
      // Parse into a LOCAL array: parsing may create anonymous helper rules,
      // which SetLength(FRules) and would invalidate a var-alias into FRules.
      SetLength(LocalBody, 0);
      ParseAlternates(LocalBody);
      GrammarAppendElem(LocalBody, getEnd, 0);
      FRules[R] := LocalBody;
    end;
  finally
    Lines.Free;
    Defs.Free;
  end;

  FRootRule := FRuleNames.IndexOf('root');
  if FRootRule < 0 then
    raise EAssertionFailed.Create('TNNetGrammar: no ''root'' rule defined');
end;

{ TNNetGrammarMachine }

constructor TNNetGrammarMachine.Create(AGrammar: TNNetGrammar);
begin
  inherited Create();
  FGrammar := AGrammar;
  Reset();
end;

function TNNetGrammarMachine.PackPos(Rule, Idx: integer): integer;
begin
  Result := Rule * KGrammarStride + Idx;
end;

procedure TNNetGrammarMachine.UnpackPos(Pos: integer; out Rule, Idx: integer);
begin
  Rule := Pos div KGrammarStride;
  Idx := Pos mod KGrammarStride;
end;

function TNNetGrammarMachine.ScratchHas(const Src: array of integer;
  Len: integer): boolean;
var
  I, J: integer;
  Same: boolean;
begin
  Result := false;
  for I := 0 to FScratchCount - 1 do
  begin
    if FScratchLen[I] <> Len then continue;
    Same := true;
    for J := 0 to Len - 1 do
      if FScratch[I][J] <> Src[J] then begin Same := false; break; end;
    if Same then exit(true);
  end;
end;

procedure TNNetGrammarMachine.AddStackRaw(const Src: array of integer;
  Len: integer);
var
  J: integer;
begin
  if ScratchHas(Src, Len) then exit;
  if FScratchCount >= Length(FScratch) then
  begin
    SetLength(FScratch, FScratchCount * 2 + 8);
    SetLength(FScratchLen, FScratchCount * 2 + 8);
  end;
  if Length(FScratch[FScratchCount]) < Len then
    SetLength(FScratch[FScratchCount], Len + 8);
  for J := 0 to Len - 1 do FScratch[FScratchCount][J] := Src[J];
  FScratchLen[FScratchCount] := Len;
  Inc(FScratchCount);
end;

procedure TNNetGrammarMachine.AddStackExpanded(const Src: array of integer;
  Len: integer);
// Descends the stack top: rule-refs are expanded (forking on each alternate),
// getEnd/getAlt pop, terminals (or empty stacks) come to rest in the scratch
// set. Recursion depth is bounded by the grammar nesting + recursion depth of
// the partial parse.
var
  Work: array of integer;
  WLen: integer;
  TopPos, Rule, Idx, RefRule, ContPos, K: integer;
  Body: TNNetGrammarElemArray;
begin
  SetLength(Work, Len + 8);
  for K := 0 to Len - 1 do Work[K] := Src[K];
  WLen := Len;

  if WLen = 0 then
  begin
    AddStackRaw(Work, 0);
    exit;
  end;

  TopPos := Work[WLen - 1];
  UnpackPos(TopPos, Rule, Idx);
  Body := FGrammar.FRules[Rule];

  case Body[Idx].ElemType of
    getEnd, getAlt:
      begin
        // End of an alternate/rule: pop and continue with the parent.
        Dec(WLen);
        AddStackExpanded(Work, WLen);
      end;
    getRuleRef:
      begin
        RefRule := Body[Idx].Value;
        ContPos := PackPos(Rule, Idx + 1);
        Work[WLen - 1] := ContPos; // continuation replaces the ref on top
        // Fork into each alternate of the referenced rule.
        PushRuleAlternates(Work, WLen, RefRule);
      end;
    else
      // Terminal top: a valid resting state.
      AddStackRaw(Work, WLen);
  end;
end;

procedure TNNetGrammarMachine.PushRuleAlternates(const Base: array of integer;
  BaseLen, RuleIdx: integer);
// For each top-level alternate of RuleIdx, push its first-element position atop
// Base[0..BaseLen-1] and expand. An empty alternate's first position is its
// getEnd, which AddStackExpanded pops to continue with Base.
var
  Work: array of integer;
  AltIdx, K: integer;
  RefBody: TNNetGrammarElemArray;
begin
  RefBody := FGrammar.FRules[RuleIdx];
  SetLength(Work, BaseLen + 1);
  for K := 0 to BaseLen - 1 do Work[K] := Base[K];
  AltIdx := 0;
  while true do
  begin
    Work[BaseLen] := PackPos(RuleIdx, AltIdx);
    AddStackExpanded(Work, BaseLen + 1);
    K := AltIdx;
    while (RefBody[K].ElemType <> getAlt) and
          (RefBody[K].ElemType <> getEnd) do Inc(K);
    if RefBody[K].ElemType = getEnd then break;
    AltIdx := K + 1;
  end;
end;

procedure TNNetGrammarMachine.CommitScratchToActive();
var
  I, J: integer;
begin
  if Length(FStacks) < FScratchCount then
  begin
    SetLength(FStacks, FScratchCount);
    SetLength(FStackLen, FScratchCount);
  end;
  for I := 0 to FScratchCount - 1 do
  begin
    if Length(FStacks[I]) < FScratchLen[I] then
      SetLength(FStacks[I], FScratchLen[I] + 8);
    for J := 0 to FScratchLen[I] - 1 do FStacks[I][J] := FScratch[I][J];
    FStackLen[I] := FScratchLen[I];
  end;
  FStackCount := FScratchCount;
end;

procedure TNNetGrammarMachine.Reset();
var
  Empty: array of integer;
begin
  FStackCount := 0;
  FScratchCount := 0;
  SetLength(Empty, 0);
  // Seed the active set with every alternate of the root rule.
  PushRuleAlternates(Empty, 0, FGrammar.RootRule);
  CommitScratchToActive();
end;

procedure TNNetGrammarMachine.CopyFrom(Source: TNNetGrammarMachine);
var
  I, J: integer;
begin
  FGrammar := Source.FGrammar;
  if Length(FStacks) < Source.FStackCount then
  begin
    SetLength(FStacks, Source.FStackCount);
    SetLength(FStackLen, Source.FStackCount);
  end;
  for I := 0 to Source.FStackCount - 1 do
  begin
    if Length(FStacks[I]) < Source.FStackLen[I] then
      SetLength(FStacks[I], Source.FStackLen[I] + 8);
    for J := 0 to Source.FStackLen[I] - 1 do
      FStacks[I][J] := Source.FStacks[I][J];
    FStackLen[I] := Source.FStackLen[I];
  end;
  FStackCount := Source.FStackCount;
end;

function TNNetGrammarMachine.ElemMatches(Pos: integer; C: char): boolean;
var
  Rule, Idx, SetIdx, R: integer;
  Body: TNNetGrammarElemArray;
  InSet: boolean;
begin
  UnpackPos(Pos, Rule, Idx);
  Body := FGrammar.FRules[Rule];
  case Body[Idx].ElemType of
    getChar: Result := C = Chr(Body[Idx].Value);
    getCharAny: Result := C <> #0;
    getCharSet, getCharSetNot:
      begin
        SetIdx := Body[Idx].Value;
        InSet := false;
        for R := FGrammar.FCharSets[SetIdx].First to
          FGrammar.FCharSets[SetIdx].First +
          FGrammar.FCharSets[SetIdx].Count - 1 do
          if (C >= FGrammar.FRanges[R].Lo) and
             (C <= FGrammar.FRanges[R].Hi) then
          begin InSet := true; break; end;
        if Body[Idx].ElemType = getCharSet
        then Result := InSet
        else Result := (not InSet) and (C <> #0);
      end;
    else Result := false;
  end;
end;

function TNNetGrammarMachine.FeedChar(C: char): boolean;
var
  I, TopPos, Rule, Idx: integer;
  Adv: array of integer;
begin
  FScratchCount := 0;
  for I := 0 to FStackCount - 1 do
  begin
    if FStackLen[I] = 0 then continue; // a completed stack accepts nothing
    TopPos := FStacks[I][FStackLen[I] - 1];
    if ElemMatches(TopPos, C) then
    begin
      UnpackPos(TopPos, Rule, Idx);
      SetLength(Adv, FStackLen[I]);
      Move(FStacks[I][0], Adv[0], FStackLen[I] * SizeOf(integer));
      Adv[FStackLen[I] - 1] := PackPos(Rule, Idx + 1);
      AddStackExpanded(Adv, FStackLen[I]);
    end;
  end;
  Result := FScratchCount > 0;
  CommitScratchToActive();
end;

function TNNetGrammarMachine.FeedString(const S: string): boolean;
var
  I: integer;
begin
  Result := true;
  for I := 1 to Length(S) do
    if not FeedChar(S[I]) then exit(false);
end;

function TNNetGrammarMachine.CharAllowed(C: char): boolean;
var
  I, TopPos: integer;
begin
  Result := false;
  for I := 0 to FStackCount - 1 do
  begin
    if FStackLen[I] = 0 then continue;
    TopPos := FStacks[I][FStackLen[I] - 1];
    if ElemMatches(TopPos, C) then exit(true);
  end;
end;

function TNNetGrammarMachine.IsComplete(): boolean;
var
  I: integer;
begin
  Result := false;
  for I := 0 to FStackCount - 1 do
    if FStackLen[I] = 0 then exit(true);
end;

function TNNetGrammarMachine.ActiveCount(): integer;
begin
  Result := FStackCount;
end;

{ TNNetGrammarConstraint }

constructor TNNetGrammarConstraint.Create(const GBNFText: string;
  Dict: TStringListInt);
var
  I: integer;
begin
  inherited Create();
  FGrammar := TNNetGrammar.Create(GBNFText);
  FMachine := TNNetGrammarMachine.Create(FGrammar);
  FProbe := TNNetGrammarMachine.Create(FGrammar);
  SetLength(FTokenStr, Dict.GetVocabCount());
  for I := 0 to High(FTokenStr) do
    if I < 2
    then FTokenStr[I] := ''
    else FTokenStr[I] := Dict.DeTokenize(I);
end;

constructor TNNetGrammarConstraint.CreateCharLevel(const GBNFText: string;
  VocabSize: integer);
var
  I: integer;
begin
  inherited Create();
  FGrammar := TNNetGrammar.Create(GBNFText);
  FMachine := TNNetGrammarMachine.Create(FGrammar);
  FProbe := TNNetGrammarMachine.Create(FGrammar);
  SetLength(FTokenStr, VocabSize);
  for I := 0 to High(FTokenStr) do
    if I < 2
    then FTokenStr[I] := ''
    else FTokenStr[I] := Chr(I);
end;

destructor TNNetGrammarConstraint.Destroy();
begin
  FProbe.Free;
  FMachine.Free;
  FGrammar.Free;
  inherited Destroy();
end;

procedure TNNetGrammarConstraint.Reset(const PromptTokens: array of integer);
begin
  // The grammar constrains ONLY the generated text; the prompt is plain
  // conditioning and is NOT fed through the machine.
  FMachine.Reset();
end;

function TNNetGrammarConstraint.TokenAllowed(TokenId: integer): boolean;
var
  S: string;
  I: integer;
begin
  if (TokenId < 0) or (TokenId > High(FTokenStr)) then exit(false);
  // Special/EOS ids: legal exactly when the grammar is in a complete state.
  if TokenId < 2 then exit(FMachine.IsComplete());
  S := FTokenStr[TokenId];
  if S = '' then exit(false);
  // Transitive multi-character validation on a forked machine.
  FProbe.CopyFrom(FMachine);
  for I := 1 to Length(S) do
    if not FProbe.FeedChar(S[I]) then exit(false);
  Result := true;
end;

procedure TNNetGrammarConstraint.Commit(TokenId: integer);
begin
  if (TokenId < 2) or (TokenId > High(FTokenStr)) then exit;
  FMachine.FeedString(FTokenStr[TokenId]);
end;

{ TNNetLogitsProcessor }

function TNNetLogitsProcessor.ExpectsProbabilities(): boolean;
begin
  Result := true;
end;

procedure TNNetLogitsProcessor.Reset(const PromptTokens: array of integer);
begin
  // Default: stateless processor, nothing to rewind.
end;

procedure TNNetLogitsProcessor.Commit(TokenId: integer);
begin
  // Default: stateless processor, nothing to advance.
end;

{ TNNetTemperatureProcessor }

constructor TNNetTemperatureProcessor.Create(Temperature: TNeuralFloat);
begin
  inherited Create();
  FTemperature := Temperature;
end;

procedure TNNetTemperatureProcessor.ProcessRow(Row: TNNetVolume);
var
  I: integer;
  T, MaxP, LogMaxP, Sum: TNeuralFloat;
begin
  T := FTemperature;
  // Temperature 1.0 is a bit-for-bit no-op (p^(1/1) renormalized would
  // already be the identity, but skipping avoids float round-trips).
  if T = 1.0 then exit;
  if T < csDecodeMinTemperature then T := csDecodeMinTemperature;
  MaxP := Row.Raw[0];
  for I := 1 to Row.Size - 1 do
    if Row.Raw[I] > MaxP then MaxP := Row.Raw[I];
  // Defensive: an all-zero (or negative-degenerate) row is left untouched,
  // mirroring MaskAllowed's zero-mass fallback.
  if MaxP <= 0 then exit;
  // Stable log-space exponentiation: exp((ln p - ln max_p)/T). The argument
  // is always <= 0 (SafeLogProb is monotone and clamps zeros), so the max
  // element maps to exactly 1 and nothing overflows; with T at the clamp the
  // row degenerates to one-hot argmax (greedy).
  LogMaxP := SafeLogProb(MaxP);
  Sum := 0;
  for I := 0 to Row.Size - 1 do
  begin
    Row.Raw[I] := Exp((SafeLogProb(Row.Raw[I]) - LogMaxP) / T);
    Sum := Sum + Row.Raw[I];
  end;
  // Sum >= 1 by construction (the max element contributes exactly 1).
  for I := 0 to Row.Size - 1 do
    Row.Raw[I] := Row.Raw[I] / Sum;
end;

{ TNNetPenaltyProcessor }

constructor TNNetPenaltyProcessor.Create(pPenalty: TNNetTokenHistoryPenalty;
  pOwnsPenalty: boolean = false);
begin
  inherited Create();
  FPenalty := pPenalty;
  FOwnsPenalty := pOwnsPenalty;
end;

destructor TNNetPenaltyProcessor.Destroy();
begin
  if FOwnsPenalty then FPenalty.Free;
  inherited Destroy();
end;

procedure TNNetPenaltyProcessor.Reset(const PromptTokens: array of integer);
var
  I: integer;
begin
  // Fresh sequence: clear the history and register the prompt tokens so the
  // penalties see the WHOLE context (the streamed loop's convention).
  FPenalty.ResetHistory();
  for I := 0 to High(PromptTokens) do FPenalty.RegisterToken(PromptTokens[I]);
end;

procedure TNNetPenaltyProcessor.ProcessRow(Row: TNNetVolume);
begin
  // Probability-domain rule (p^r, exp() factors, renormalize) - the row is
  // post-softmax per the chain's domain convention.
  FPenalty.ApplyToProbabilities(Row);
end;

procedure TNNetPenaltyProcessor.Commit(TokenId: integer);
begin
  FPenalty.RegisterToken(TokenId);
end;

{ TNNetConstraintProcessor }

constructor TNNetConstraintProcessor.Create(pConstraint: TNNetTokenConstraint;
  pOwnsConstraint: boolean = false);
begin
  inherited Create();
  FConstraint := pConstraint;
  FOwnsConstraint := pOwnsConstraint;
end;

destructor TNNetConstraintProcessor.Destroy();
begin
  if FOwnsConstraint then FConstraint.Free;
  inherited Destroy();
end;

procedure TNNetConstraintProcessor.Reset(const PromptTokens: array of integer);
begin
  FConstraint.Reset(PromptTokens);
end;

procedure TNNetConstraintProcessor.ProcessRow(Row: TNNetVolume);
begin
  FConstraint.MaskAllowed(Row);
end;

procedure TNNetConstraintProcessor.Commit(TokenId: integer);
begin
  FConstraint.Commit(TokenId);
end;

{ TNNetLogitsProcessorChain }

destructor TNNetLogitsProcessorChain.Destroy();
var
  I: integer;
begin
  for I := 0 to High(FItems) do
    if FOwned[I] then FItems[I].Free;
  SetLength(FItems, 0);
  SetLength(FOwned, 0);
  inherited Destroy();
end;

function TNNetLogitsProcessorChain.GetCount(): integer;
begin
  Result := Length(FItems);
end;

function TNNetLogitsProcessorChain.GetItem(
  Index: integer): TNNetLogitsProcessor;
begin
  Result := FItems[Index];
end;

function TNNetLogitsProcessorChain.Add(P: TNNetLogitsProcessor;
  OwnsProcessor: boolean = false): TNNetLogitsProcessorChain;
var
  N: integer;
begin
  N := Length(FItems);
  SetLength(FItems, N + 1);
  SetLength(FOwned, N + 1);
  FItems[N] := P;
  FOwned[N] := OwnsProcessor;
  Result := Self;
end;

function TNNetLogitsProcessorChain.ExpectsProbabilities(): boolean;
var
  I: integer;
begin
  Result := true;
  for I := 0 to High(FItems) do
    if not FItems[I].ExpectsProbabilities() then exit(false);
end;

procedure TNNetLogitsProcessorChain.Reset(
  const PromptTokens: array of integer);
var
  I: integer;
begin
  for I := 0 to High(FItems) do FItems[I].Reset(PromptTokens);
end;

procedure TNNetLogitsProcessorChain.ProcessRow(Row: TNNetVolume);
var
  I: integer;
begin
  for I := 0 to High(FItems) do FItems[I].ProcessRow(Row);
end;

procedure TNNetLogitsProcessorChain.Commit(TokenId: integer);
var
  I: integer;
begin
  for I := 0 to High(FItems) do FItems[I].Commit(TokenId);
end;

{ TNNetCFGProcessor }

constructor TNNetCFGProcessor.Create(UncondSession: TNNetStreamingDecoder;
  const NegativePrompt: array of integer; GuidanceScale: TNeuralFloat;
  OwnsSession: boolean = false);
var
  I: integer;
begin
  inherited Create();
  if not Assigned(UncondSession) then
    raise EArgumentException.Create(
      'TNNetCFGProcessor: the unconditional session must be assigned.');
  if Length(NegativePrompt) < 1 then
    raise EArgumentException.Create(
      'TNNetCFGProcessor: the negative/unconditional prompt must be ' +
      'non-empty (it seeds the unconditional branch''s first input).');
  if UncondSession.Net.GetFirstLayer().Output.SizeX <> 1 then
    raise EArgumentException.Create(
      'TNNetCFGProcessor: the unconditional session net must be a WIDTH-1 ' +
      'twin (input SizeX=1).');
  FSession := UncondSession;
  FOwnsSession := OwnsSession;
  FGuidanceScale := GuidanceScale;
  SetLength(FNegPrompt, Length(NegativePrompt));
  for I := 0 to High(NegativePrompt) do FNegPrompt[I] := NegativePrompt[I];
  FInV := TNNetVolume.Create(FSession.Net.GetFirstLayer().Output);
  FInV.Fill(0);
end;

destructor TNNetCFGProcessor.Destroy();
begin
  FInV.Free;
  if FOwnsSession then FSession.Free;
  inherited Destroy();
end;

procedure TNNetCFGProcessor.Reset(const PromptTokens: array of integer);
var
  Pos: integer;
begin
  // Fresh sequence: prefill the unconditional branch with the negative prompt
  // (tokens 0..len-2), leaving its LAST token as the first decode step's
  // input - exactly the prefill-then-step idiom the conditional loop uses on
  // its own prompt. The conditional PromptTokens are intentionally ignored:
  // the whole point of CFG is that the two branches differ in their prompt.
  FSession.Reset();
  for Pos := 0 to Length(FNegPrompt) - 2 do
  begin
    FInV.FData[0] := FNegPrompt[Pos];
    FSession.StepForward(FInV, Pos);
  end;
  FPendingToken := FNegPrompt[High(FNegPrompt)];
  FAbsPos := Length(FNegPrompt) - 1;
end;

procedure TNNetCFGProcessor.ProcessRow(Row: TNNetVolume);
var
  UncondRow: TNNetVolume;
  I: integer;
  LCond, LUncond, Combined, MaxL, Sum: TNeuralFloat;
begin
  // GuidanceScale = 1 collapses to the conditional row exactly (uncond + 1 *
  // (cond - uncond) = cond); skip the second forward to keep it bit-for-bit
  // identical to plain decoding AND to avoid the cost.
  if FGuidanceScale = 1.0 then exit;
  // Step the unconditional branch forward one token at its absolute position
  // (the negative prompt's running length + tokens generated so far).
  FInV.FData[0] := FPendingToken;
  FSession.StepForward(FInV, FAbsPos);
  UncondRow := FSession.Output();
  // Combine in LOG space: logits := uncond + scale * (cond - uncond), with
  // log-probs standing in for the pre-softmax logits (the per-branch softmax
  // constant cancels in the final softmax below). Track the max for a stable
  // softmax back to probabilities.
  MaxL := -1e30;
  for I := 0 to Row.Size - 1 do
  begin
    LCond := SafeLogProb(Row.Raw[I]);
    LUncond := SafeLogProb(UncondRow.Raw[I]);
    Combined := LUncond + FGuidanceScale * (LCond - LUncond);
    Row.Raw[I] := Combined;
    if Combined > MaxL then MaxL := Combined;
  end;
  // Softmax the combined logits back into a probability row (the chain's
  // documented domain).
  Sum := 0;
  for I := 0 to Row.Size - 1 do
  begin
    Row.Raw[I] := Exp(Row.Raw[I] - MaxL);
    Sum := Sum + Row.Raw[I];
  end;
  if Sum > 0 then
    for I := 0 to Row.Size - 1 do Row.Raw[I] := Row.Raw[I] / Sum;
end;

procedure TNNetCFGProcessor.Commit(TokenId: integer);
begin
  // The emitted token becomes the next input of BOTH branches; the
  // unconditional branch advances one absolute position. (When GuidanceScale
  // = 1 ProcessRow never stepped the branch, but keeping the bookkeeping in
  // sync is harmless and lets the scale be changed mid-run.)
  FPendingToken := TokenId;
  Inc(FAbsPos);
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

// True when Token terminates generation: membership in EOSTokens when the
// list is non-empty, otherwise the codebase-wide "token < 2" rule.
function TokenIsEOS(Token: integer;
  const EOSTokens: TNeuralIntegerArray): boolean;
var
  I: integer;
begin
  if Length(EOSTokens) = 0 then exit(Token < 2);
  Result := false;
  for I := 0 to High(EOSTokens) do
    if EOSTokens[I] = Token then exit(true);
end;

// Assembles the standard pipeline Penalty -> Temperature -> UserProcessors ->
// Constraint as an adapter chain (only the non-nil / non-default stages are
// added; UserProcessors is appended NOT owned). Returns nil when every stage
// is off, so the caller can take the zero-overhead plain path.
function BuildProcessorPipeline(Penalty: TNNetTokenHistoryPenalty;
  Temperature: TNeuralFloat; UserProcessors: TNNetLogitsProcessorChain;
  Constraint: TNNetTokenConstraint): TNNetLogitsProcessorChain;
begin
  if (Penalty = nil) and (Temperature = 1.0) and
    ((UserProcessors = nil) or (UserProcessors.Count = 0)) and
    (Constraint = nil) then exit(nil);
  Result := TNNetLogitsProcessorChain.Create();
  if Assigned(Penalty) then
    Result.Add(TNNetPenaltyProcessor.Create(Penalty), true);
  if Temperature <> 1.0 then
    Result.Add(TNNetTemperatureProcessor.Create(Temperature), true);
  if Assigned(UserProcessors) and (UserProcessors.Count > 0) then
    Result.Add(UserProcessors, false);
  if Assigned(Constraint) then
    Result.Add(TNNetConstraintProcessor.Create(Constraint), true);
end;

function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  const StopSequences: TNNetTokenSequences;
  Constraint: TNNetTokenConstraint): integer;
begin
  // Delegate through the adapter chain in the historical order (penalty,
  // then constraint) - bit-for-bit the pre-chain behavior.
  Result := GenerateTokensStreamed(Session, Tokens, PromptLen, MaxNewTokens,
    MaxTotalLen, Sampler, Penalty, StopSequences, Constraint, 1.0);
end;

function GenerateTokensStreamed(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Penalty: TNNetTokenHistoryPenalty;
  const StopSequences: TNNetTokenSequences;
  Constraint: TNNetTokenConstraint;
  Temperature: TNeuralFloat): integer;
var
  Chain: TNNetLogitsProcessorChain;
begin
  Chain := BuildProcessorPipeline(Penalty, Temperature, nil, Constraint);
  try
    Result := GenerateTokensStreamedWithProcessors(Session, Tokens,
      PromptLen, MaxNewTokens, MaxTotalLen, Sampler, Chain, StopSequences,
      nil);
  finally
    Chain.Free;
  end;
end;

function GenerateTokensStreamedWithProcessors(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen, MaxNewTokens,
  MaxTotalLen: integer; Sampler: TNNetSamplerBase;
  Processors: TNNetLogitsProcessor;
  const StopSequences: TNNetTokenSequences;
  const EOSTokens: TNeuralIntegerArray): integer;
var
  InV: TNNetVolume;
  Pos, CapLen, NextTokenInt, StopLen: integer;
begin
  // The chain's explicit domain contract: the streamed row is POST-SOFTMAX
  // probabilities, so a raw-logit processor cannot be applied here.
  if Assigned(Processors) and (not Processors.ExpectsProbabilities()) then
    raise EArgumentException.Create(
      'GenerateTokensStreamedWithProcessors: the streamed loop feeds ' +
      'POST-SOFTMAX probabilities, but a processor in the chain declares ' +
      'a raw-logit domain (ExpectsProbabilities=False).');
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
    // A fresh sequence for the whole pipeline: penalties clear their history
    // and register the prompt, stateful grammars rewind (see the adapters).
    if Assigned(Processors) then
      Processors.Reset(Copy(Tokens, 0, PromptLen));
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
      // PROBABILITIES (the chain's documented domain). Processors transform
      // the row IN ORDER before the sampler/argmax reads it. Mutating the
      // net's output volume in place is safe: the next StepForward
      // recomputes it.
      if Assigned(Processors) then Processors.ProcessRow(Session.Output());
      // The step net is width-1, so the (only) output row is pixel (0,0).
      if Assigned(Sampler)
      then NextTokenInt := Sampler.GetTokenOnPixel(Session.Output(), 0, 0)
      else NextTokenInt := Session.Output().GetClassOnPixel(0, 0);
      Tokens[Pos] := NextTokenInt;
      if Assigned(Processors) then Processors.Commit(NextTokenInt);
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
      // End-of-sequence: membership in EOSTokens when provided, else the
      // codebase-wide "NextTokenInt < 2" rule (the EOS token is stored and
      // counted, like GenerateStringFromCasualNN).
      if TokenIsEOS(NextTokenInt, EOSTokens) then Break;
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
begin
  // Delegate through the adapter chain in the historical order (penalty,
  // then constraint) - bit-for-bit the pre-chain behavior.
  Result := GenerateStringStreamed(Session, Dict, InputString, MaxNewTokens,
    MaxTotalLen, oSampler, Penalty, StopStrings, Constraint, 1.0);
end;

function GenerateStringStreamed(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase; Penalty: TNNetTokenHistoryPenalty;
  const StopStrings: array of string;
  Constraint: TNNetTokenConstraint;
  Temperature: TNeuralFloat): string;
var
  Chain: TNNetLogitsProcessorChain;
begin
  Chain := BuildProcessorPipeline(Penalty, Temperature, nil, Constraint);
  try
    Result := GenerateStringStreamedWithProcessors(Session, Dict,
      InputString, MaxNewTokens, MaxTotalLen, oSampler, Chain, StopStrings,
      nil, nil);
  finally
    Chain.Free;
  end;
end;

function PrepareTokenHealing(Dict: TStringListInt;
  const Tokens: TNeuralIntegerArray; var PromptLen: integer;
  out DroppedToken: integer): TNNetTokenHealingConstraint;
var
  LastText, CandText: string;
  Allowed: TNeuralIntegerArray;
  TokenCnt, AllowedCount, VocabCount, LastLen: integer;
  HasStrictExtension: boolean;
begin
  Result := nil;
  DroppedToken := -1;
  if PromptLen < 2 then exit; // healing would empty the prompt
  VocabCount := Dict.GetVocabCount();
  if (Tokens[PromptLen - 1] < 0) or
     (Tokens[PromptLen - 1] >= VocabCount) then exit;
  LastText := Dict.DeTokenize(Tokens[PromptLen - 1]);
  LastLen := Length(LastText);
  if LastLen = 0 then exit; // byte-level/special oddity: no-op fallback
  SetLength(Allowed, VocabCount);
  AllowedCount := 0;
  HasStrictExtension := false;
  for TokenCnt := 0 to VocabCount - 1 do
  begin
    CandText := Dict.DeTokenize(TokenCnt);
    if (Length(CandText) >= LastLen) and
       (Copy(CandText, 1, LastLen) = LastText) then
    begin
      Allowed[AllowedCount] := TokenCnt;
      Inc(AllowedCount);
      if Length(CandText) > LastLen then HasStrictExtension := true;
    end;
  end;
  // Without a strict extension the only allowed continuation re-emits the
  // dropped token - the healed run provably equals the unhealed one, so
  // healing is skipped (PromptLen untouched).
  if not HasStrictExtension then exit;
  SetLength(Allowed, AllowedCount);
  DroppedToken := Tokens[PromptLen - 1];
  Dec(PromptLen);
  Result := TNNetTokenHealingConstraint.Create(Allowed);
end;

function GenerateStringStreamedWithProcessors(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string; MaxNewTokens, MaxTotalLen: integer;
  oSampler: TNNetSamplerBase; Processors: TNNetLogitsProcessor;
  const StopStrings: array of string;
  const ExtraStopSequences: TNNetTokenSequences;
  const EOSTokens: TNeuralIntegerArray;
  TokenHealing: boolean = false): string;
var
  Tokens, StopToks: TNeuralIntegerArray;
  TokenStops: TNNetTokenSequences;
  PromptLen, TotalLen, Pos, VocabCount, S, CutAt: integer;
  Continuation, Prefix, DroppedText: string;
  HealConstraint: TNNetTokenHealingConstraint;
  HealChain: TNNetLogitsProcessorChain;
  EffProcessors: TNNetLogitsProcessor;
  DroppedToken: integer;
begin
  Result := InputString;
  VocabCount := Dict.GetVocabCount();
  Dict.Tokenize(InputString, Tokens);
  PromptLen := Length(Tokens);
  if PromptLen < 1 then Exit; // nothing to condition on
  // Token healing: drop the last prompt token, constrain step 1 to its
  // extensions, and strip its text (plus the display separator for word
  // dicts) from the prompt prefix - the healed first token re-emits the
  // text, possibly extended.
  Prefix := InputString;
  HealConstraint := nil;
  HealChain := nil;
  EffProcessors := Processors;
  if TokenHealing then
  begin
    HealConstraint := PrepareTokenHealing(Dict, Tokens, PromptLen,
      DroppedToken);
    if HealConstraint <> nil then
    begin
      DroppedText := Dict.DeTokenize(DroppedToken);
      if (Length(DroppedText) <= Length(Prefix)) and
         (Copy(Prefix, Length(Prefix) - Length(DroppedText) + 1,
           Length(DroppedText)) = DroppedText) then
      begin
        SetLength(Prefix, Length(Prefix) - Length(DroppedText));
        if Dict.TokenizerHasSeparator and (Prefix <> '') and
           (Prefix[Length(Prefix)] = ' ') then
          SetLength(Prefix, Length(Prefix) - 1);
      end;
      // The healing constraint runs LAST (after any caller pipeline), the
      // TGenerationConfig convention for structural constraints.
      HealChain := TNNetLogitsProcessorChain.Create();
      if Processors <> nil then HealChain.Add(Processors, false);
      HealChain.Add(TNNetConstraintProcessor.Create(HealConstraint,
        {OwnsConstraint=}true), {OwnsProcessor=}true);
      EffProcessors := HealChain;
    end;
  end;
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
  // Token-id stop sequences passed directly (config path) are appended after
  // the tokenized stop strings.
  for S := 0 to High(ExtraStopSequences) do
    if Length(ExtraStopSequences[S]) > 0 then
    begin
      SetLength(TokenStops, Length(TokenStops) + 1);
      TokenStops[High(TokenStops)] :=
        Copy(ExtraStopSequences[S], 0, Length(ExtraStopSequences[S]));
    end;
  TotalLen := GenerateTokensStreamedWithProcessors(Session, Tokens, PromptLen,
    MaxNewTokens, MaxTotalLen, oSampler, EffProcessors, TokenStops, EOSTokens);
  HealChain.Free; // frees the healing adapter+constraint; never Processors
  // Detokenize the continuation; stop at the first terminating token (the
  // EOSTokens list, or the "< 2" rule when it is empty) for display (the
  // TokensToText convention) and join with a space only for separator
  // vocabularies (word dicts; BPE vocabularies concatenate).
  Continuation := '';
  for Pos := PromptLen to TotalLen - 1 do
  begin
    if TokenIsEOS(Tokens[Pos], EOSTokens) then Break;
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
  // Healed runs rebuild from the trimmed prefix (the healed first token
  // re-emits the dropped text). If nothing was generated after all
  // (MaxNewTokens=0, or an immediate EOS through the zero-mass fallback),
  // fall back to the untouched prompt instead of losing the dropped text.
  if (HealConstraint <> nil) and (Continuation = '') then
    Result := InputString
  else
    Result := Prefix + Continuation;
  SetLength(Tokens, 0);
end;

{ Config-driven generation }

function MakeUnconditionalTwin(SourceWidth1Net: TNNet;
  MaxTotalLen: integer; out TwinNet: TNNet): TNNetStreamingDecoder;
begin
  if not Assigned(SourceWidth1Net) then
    raise EArgumentException.Create(
      'MakeUnconditionalTwin: SourceWidth1Net must be assigned.');
  if SourceWidth1Net.GetFirstLayer().Output.SizeX <> 1 then
    raise EArgumentException.Create(
      'MakeUnconditionalTwin: SourceWidth1Net must be a WIDTH-1 net ' +
      '(input SizeX=1); build the width-1 twin first (Build*(1) + ' +
      'CopyWeights). Got SizeX=' +
      IntToStr(SourceWidth1Net.GetFirstLayer().Output.SizeX) + '.');
  // Clone architecture AND weights; the round-trip preserves the width-1
  // input shape, so the clone is a valid streaming twin with its own state.
  TwinNet := TNNet.Create();
  TwinNet.LoadFromString(SourceWidth1Net.SaveToString());
  Result := TNNetStreamingDecoder.Create(TwinNet, MaxTotalLen);
end;

function DefaultGenerationConfig(MaxNewTokens: integer;
  MaxTotalLen: integer = 0): TGenerationConfig;
begin
  Result.MaxNewTokens := MaxNewTokens;
  Result.MaxTotalLen := MaxTotalLen;
  SetLength(Result.EOSTokens, 0);
  SetLength(Result.StopSequences, 0);
  SetLength(Result.StopStrings, 0);
  Result.Temperature := 1.0;
  Result.Penalty := nil;
  Result.Processors := nil;
  Result.Constraint := nil;
  Result.Sampler := nil;
  Result.GuidanceScale := 1.0; // CFG off
  Result.CFGUncond := nil;
  SetLength(Result.NegativePrompt, 0);
  Result.TokenHealing := false;
end;

// Assemble the per-step pipeline implied by a TGenerationConfig. CFG (when
// GuidanceScale <> 1.0) runs FIRST - it combines the conditional and
// unconditional model distributions before any penalty/temperature/processor/
// constraint transform sees the row - then the standard
// Penalty -> Temperature -> Processors -> Constraint pipeline. Returns nil
// when every knob is off (the plain path). The returned chain owns only the
// adapters it created (the CFG processor included); Config's own objects are
// not owned.
function BuildConfigPipeline(
  const Config: TGenerationConfig): TNNetLogitsProcessorChain;
var
  StdChain: TNNetLogitsProcessorChain;
begin
  StdChain := BuildProcessorPipeline(Config.Penalty, Config.Temperature,
    Config.Processors, Config.Constraint);
  if Config.GuidanceScale = 1.0 then exit(StdChain); // CFG off: as before
  if not Assigned(Config.CFGUncond) then
    raise EArgumentException.Create(
      'BuildConfigPipeline: GuidanceScale <> 1 requires Config.CFGUncond ' +
      '(the unconditional width-1 twin) to be assigned.');
  Result := TNNetLogitsProcessorChain.Create();
  // CFG owns its run but NOT the session/twin (OwnsSession=false): the config
  // does not own CFGUncond, so neither does the chain it spawns.
  Result.Add(TNNetCFGProcessor.Create(Config.CFGUncond, Config.NegativePrompt,
    Config.GuidanceScale, {OwnsSession=}false), {OwnsProcessor=}true);
  if Assigned(StdChain) then Result.Add(StdChain, {OwnsProcessor=}true);
end;

function GenerateTokensWithConfig(Session: TNNetStreamingDecoder;
  var Tokens: TNeuralIntegerArray; PromptLen: integer;
  const Config: TGenerationConfig): integer;
var
  Chain: TNNetLogitsProcessorChain;
  MaxTotal: integer;
begin
  MaxTotal := Config.MaxTotalLen;
  if MaxTotal <= 0 then MaxTotal := PromptLen + Config.MaxNewTokens;
  Chain := BuildConfigPipeline(Config);
  try
    Result := GenerateTokensStreamedWithProcessors(Session, Tokens,
      PromptLen, Config.MaxNewTokens, MaxTotal, Config.Sampler, Chain,
      Config.StopSequences, Config.EOSTokens);
  finally
    Chain.Free; // frees the adapters only; Config's objects are NOT owned
  end;
end;

function GenerateStringWithConfig(Session: TNNetStreamingDecoder;
  Dict: TStringListInt; InputString: string;
  const Config: TGenerationConfig): string;
var
  Chain: TNNetLogitsProcessorChain;
  MaxTotal, PromptLen: integer;
  Tokens: TNeuralIntegerArray;
begin
  // MaxTotalLen = 0 resolves against the PROMPT's token count, mirroring
  // GenerateTokensWithConfig (one extra tokenize; the wrapper re-tokenizes).
  MaxTotal := Config.MaxTotalLen;
  if MaxTotal <= 0 then
  begin
    Dict.Tokenize(InputString, Tokens);
    PromptLen := Length(Tokens);
    MaxTotal := PromptLen + Config.MaxNewTokens;
    SetLength(Tokens, 0);
  end;
  Chain := BuildConfigPipeline(Config);
  try
    Result := GenerateStringStreamedWithProcessors(Session, Dict,
      InputString, Config.MaxNewTokens, MaxTotal, Config.Sampler, Chain,
      Config.StopStrings, Config.StopSequences, Config.EOSTokens,
      Config.TokenHealing);
  finally
    Chain.Free; // frees the adapters only; Config's objects are NOT owned
  end;
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

// The "last hidden state" layer for contrastive search: the input to the final
// projection / LM head. The last layer is the logit / probability layer; when
// it is a SoftMax variant the logit-producing layer is the one before it. The
// hidden state is the OUTPUT of the layer FEEDING that logit layer (the
// representation the LM head reads). Falls back to GetLastLayer's PrevLayer
// when the chain is too short to look further back.
// Coded by Claude (AI).
function ContrastiveHiddenLayer(NN: TNNet): TNNetLayer;
var
  Head: TNNetLayer;
begin
  Head := NN.GetLastLayer();
  // Skip a trailing softmax: the LM head (logits) is its predecessor.
  if (Head is TNNetPointwiseSoftMax) and Assigned(Head.PrevLayer) then
    Head := Head.PrevLayer;
  // Hidden state = the LM head's input representation.
  if Assigned(Head.PrevLayer) then
    Result := Head.PrevLayer
  else
    Result := Head;
end;

// Cosine similarity of two equal-length flat vectors (the per-token hidden
// states). Zero magnitude on either side yields 0 (no penalty), keeping the
// score finite for a dead representation.
function ContrastiveCosine(A, B: TNNetVolume): TNeuralFloat;
var
  Denom: TNeuralFloat;
begin
  Denom := A.GetMagnitude() * B.GetMagnitude();
  if Denom <= 0 then
    Result := 0
  else
    Result := A.DotProduct(B) / Denom;
end;

function DecodeContrastiveSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; TopK: integer; PenaltyAlpha: TNeuralFloat;
  const StopStrings: array of string): TNNetDecodeResult;
var
  InputVolume, OutputVolume: TNNetVolume;
  HiddenLayer: TNNetLayer;
  Probs: array of TNeuralFloat;
  Cand: array of integer;          // current top-k candidate token ids
  Past: array of TNNetVolume;      // hidden states of already-processed tokens
  CandHidden: TNNetVolume;         // snapshot of a candidate's hidden state
  VocabSize, Step, I, J, NumCand, Best, StopLen, PastLen: integer;
  Total, MaxSim, Sim, ScoreV, BestScore: TNeuralFloat;
  Context, CandStr: string;
  TmpI: integer;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  HiddenLayer := ContrastiveHiddenLayer(NN);
  VocabSize := OutputVolume.Size;
  SetLength(Probs, VocabSize);
  if TopK < 1 then TopK := 1;
  if TopK > VocabSize then TopK := VocabSize;
  Result.Text := '';
  Result.SumLogProb := 0;
  Result.Finished := False;
  Context := Prompt;
  Past := nil;
  PastLen := 0;
  CandHidden := nil;
  try
    for Step := 1 to MaxLen do
    begin
      // Forward the current context: probabilities for the next token AND the
      // context's own last hidden state (seeds the past set on the first step).
      InputVolume.OneHotEncodingReversed(Context);
      NN.Compute(InputVolume, OutputVolume);
      Total := OutputVolume.GetSum();
      if Total <= 0 then Total := 1.0;
      for I := 0 to VocabSize - 1 do Probs[I] := OutputVolume.Raw[I] / Total;
      // On the first step record the prompt's hidden state as the only past
      // context (so step 1 already has something to penalise against).
      if PastLen = 0 then
      begin
        SetLength(Past, 1);
        Past[0] := TNNetVolume.Create();
        Past[0].Copy(HiddenLayer.Output);
        PastLen := 1;
      end;
      // Top-k candidates by probability (partial selection sort; k is tiny).
      SetLength(Cand, VocabSize);
      for I := 0 to VocabSize - 1 do Cand[I] := I;
      NumCand := TopK;
      for I := 0 to NumCand - 1 do
      begin
        Best := I;
        for J := I + 1 to VocabSize - 1 do
          if Probs[Cand[J]] > Probs[Cand[Best]] then Best := J;
        TmpI := Cand[I]; Cand[I] := Cand[Best]; Cand[Best] := TmpI;
      end;
      SetLength(Cand, NumCand);
      // Re-rank candidates by the contrastive objective. PenaltyAlpha=0 keeps
      // (1-alpha)*p(v) only, so the highest-probability candidate (Cand[0])
      // wins by construction -> exactly greedy argmax over the top-k (= plain
      // greedy argmax, since the global argmax is always in the top-k set).
      Best := Cand[0];
      BestScore := -1e30;
      for I := 0 to NumCand - 1 do
      begin
        MaxSim := 0;
        if PenaltyAlpha > 0 then
        begin
          // Hidden state the model produces when candidate Cand[I] is appended.
          CandStr := Context + Chr(Cand[I]);
          InputVolume.OneHotEncodingReversed(CandStr);
          NN.Compute(InputVolume, OutputVolume);
          if CandHidden = nil then CandHidden := TNNetVolume.Create();
          CandHidden.Copy(HiddenLayer.Output);
          MaxSim := -1e30;
          for J := 0 to PastLen - 1 do
          begin
            Sim := ContrastiveCosine(CandHidden, Past[J]);
            if Sim > MaxSim then MaxSim := Sim;
          end;
        end;
        ScoreV := (1 - PenaltyAlpha) * Probs[Cand[I]] - PenaltyAlpha * MaxSim;
        if ScoreV > BestScore then
        begin
          BestScore := ScoreV;
          Best := Cand[I];
        end;
      end;
      Result.SumLogProb := Result.SumLogProb + SafeLogProb(Probs[Best]);
      if Best = csDecodeEOSToken then
      begin
        Result.Finished := True;
        Break;
      end;
      // Commit: append the token, and add ITS hidden state to the past set so
      // future candidates are penalised against it too. Recompute the chosen
      // continuation's hidden state (cheap, k is small; avoids stashing all k).
      Result.Text := Result.Text + Chr(Best);
      Context := Context + Chr(Best);
      InputVolume.OneHotEncodingReversed(Context);
      NN.Compute(InputVolume, OutputVolume);
      SetLength(Past, PastLen + 1);
      Past[PastLen] := TNNetVolume.Create();
      Past[PastLen].Copy(HiddenLayer.Output);
      Inc(PastLen);
      // Stop strings: terminate and trim, exactly like DecodeGreedy.
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
    if Assigned(CandHidden) then CandHidden.Free;
    for I := 0 to PastLen - 1 do Past[I].Free;
  end;
  Result.Score := Result.SumLogProb /
    LengthPenaltyDenominator(Length(Result.Text), 0);
end;

// Resolves the LM-head start index for the DoLa / logit-lens splice: when
// HeadStartIdx < 0 it is the LAST trainable layer (Neurons.Count > 0), the same
// heuristic TNNet.LogitLensReport uses; clamped to [1, LastLayer] so there is
// always an input slot below the head. Coded by Claude (AI).
function ResolveHeadStartIdx(NN: TNNet; HeadStartIdx: integer): integer;
var
  LayerIdx, LastLayer, LastTrainable: integer;
begin
  LastLayer := NN.GetLastLayerIdx();
  if HeadStartIdx < 0 then
  begin
    LastTrainable := -1;
    for LayerIdx := 0 to LastLayer do
      if NN.Layers[LayerIdx].Neurons.Count > 0 then
        LastTrainable := LayerIdx;
    if LastTrainable <= 0 then
      Result := LastLayer
    else
      Result := LastTrainable;
  end
  else
    Result := HeadStartIdx;
  if Result < 1 then Result := 1;
  if Result > LastLayer then Result := LastLayer;
end;

// Builds the DoLa candidate-layer bucket: every layer 0..HeadInIdx-1 whose
// Output.Size equals the head-input size (so its activation can be spliced into
// the head-input slot and pushed through the SAME LM head). HeadInIdx itself is
// EXCLUDED - contrasting a layer against its own forward is a no-op.
// Mode selects the paper's dynamic premature-layer pool over those (already
// depth-ordered) candidates: dlbFull keeps all, dlbLow the shallow half
// (index < count div 2), dlbHigh the deep half (index >= count div 2).
// Coded by Claude (AI).
procedure BuildDoLaCandidateBucket(NN: TNNet; HeadInIdx: integer;
  Mode: TDoLaLayerBucket; var Bucket: TNeuralIntegerArray);
var
  LayerIdx, HeadInSize, N, Total, Half, Lo, Hi, W: integer;
begin
  HeadInSize := NN.Layers[HeadInIdx].Output.Size;
  N := 0;
  SetLength(Bucket, HeadInIdx);
  for LayerIdx := 0 to HeadInIdx - 1 do
    if NN.Layers[LayerIdx].Output.Size = HeadInSize then
    begin
      Bucket[N] := LayerIdx;
      Inc(N);
    end;
  SetLength(Bucket, N);
  if Mode = dlbFull then Exit;
  // Split the depth-ordered candidates at the midpoint and keep one half.
  Total := N;
  Half := Total div 2;
  case Mode of
    dlbLow:  begin Lo := 0;    Hi := Half - 1; end;
    dlbHigh: begin Lo := Half; Hi := Total - 1; end;
  else
    begin Lo := 0; Hi := Total - 1; end;
  end;
  W := 0;
  for LayerIdx := Lo to Hi do
  begin
    Bucket[W] := Bucket[LayerIdx];
    Inc(W);
  end;
  SetLength(Bucket, W);
end;

function DecodeDoLa(NN: TNNet; const Prompt: string;
  MaxLen: integer; Alpha: TNeuralFloat;
  const StopStrings: array of string;
  HeadStartIdx: integer): TNNetDecodeResult;
begin
  Result := DecodeDoLa(NN, Prompt, MaxLen, Alpha, StopStrings, dlbFull,
    HeadStartIdx);
end;

function DecodeDoLa(NN: TNNet; const Prompt: string;
  MaxLen: integer; Alpha: TNeuralFloat;
  const StopStrings: array of string;
  Bucket: TDoLaLayerBucket;
  HeadStartIdx: integer): TNNetDecodeResult;
var
  InputVolume, OutputVolume: TNNetVolume;
  CandSnap: TNNetVolume;
  PFinal, PLens, MFinalLens: array of TNeuralFloat;  // distributions + 0.5(p+q)
  Cands: TNeuralIntegerArray;
  VocabSize, Step, I, C, L, HeadIdx, HeadInIdx, LastLayer: integer;
  NumCand, BestLayer, Best, StopLen: integer;
  Total, MaxFinal, Threshold, JS, BestJS, Pf, Pl, Pm, ScoreV, BestScore: TNeuralFloat;
  Context: string;
  HaveContrast: boolean;
const
  cEps = 1e-12;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  CandSnap := TNNetVolume.Create();
  VocabSize := OutputVolume.Size;
  SetLength(PFinal, VocabSize);
  SetLength(PLens, VocabSize);
  SetLength(MFinalLens, VocabSize);
  LastLayer := NN.GetLastLayerIdx();
  HeadIdx := ResolveHeadStartIdx(NN, HeadStartIdx);
  HeadInIdx := HeadIdx - 1;
  BuildDoLaCandidateBucket(NN, HeadInIdx, Bucket, Cands);
  NumCand := Length(Cands);
  // The contrast path is active only when Alpha > 0 AND the bucket is non-empty.
  // Otherwise this MUST reproduce plain greedy argmax bit-for-bit, so we take
  // EXACTLY the DecodeGreedy step (argmax over the raw softmax row).
  HaveContrast := (Alpha > 0) and (NumCand > 0);
  Result.Text := '';
  Result.SumLogProb := 0;
  Result.Finished := False;
  Context := Prompt;
  try
    for Step := 1 to MaxLen do
    begin
      // (1) Full forward -> the mature distribution p_final (re-normalised
      //     defensively, same convention as NextLogProbs / contrastive search).
      InputVolume.OneHotEncodingReversed(Context);
      NN.Compute(InputVolume, OutputVolume);
      Total := OutputVolume.GetSum();
      if Total <= 0 then Total := 1.0;
      MaxFinal := 0;
      for I := 0 to VocabSize - 1 do
      begin
        Pf := OutputVolume.Raw[I] / Total;
        if Pf < 0 then Pf := 0;
        PFinal[I] := Pf;
        if Pf > MaxFinal then MaxFinal := Pf;
      end;

      if not HaveContrast then
      begin
        // Greedy argmax over p_final == DecodeGreedy's step (the raw softmax row
        // is monotone in p_final, so this is bit-identical).
        Best := 0;
        for I := 1 to VocabSize - 1 do
          if PFinal[I] > PFinal[Best] then Best := I;
        Result.SumLogProb := Result.SumLogProb + SafeLogProb(PFinal[Best]);
      end
      else
      begin
        // (2)+(3) Pick the premature layer with MAX Jensen-Shannon divergence
        //         from p_final. The candidate activation already lives in the
        //         net after the full forward above (no extra forward needed);
        //         snapshot it, splice into the head-input slot, recompute the
        //         head sub-stack, read p_premature.
        BestLayer := Cands[0];
        BestJS := -1.0;
        for C := 0 to NumCand - 1 do
        begin
          L := Cands[C];
          CandSnap.Copy(NN.Layers[L].Output);
          NN.Layers[HeadInIdx].Output.CopyNoChecks(CandSnap);
          for I := HeadIdx to LastLayer do NN.Layers[I].Compute();
          Total := NN.GetLastLayer().Output.GetSum();
          if Total <= 0 then Total := 1.0;
          // JS(p_final || p_lens) = 0.5 KL(p||m) + 0.5 KL(q||m), m = 0.5(p+q).
          JS := 0;
          for I := 0 to VocabSize - 1 do
          begin
            Pl := NN.GetLastLayer().Output.Raw[I] / Total;
            if Pl < 0 then Pl := 0;
            PLens[I] := Pl;
            MFinalLens[I] := 0.5 * (PFinal[I] + Pl);
          end;
          for I := 0 to VocabSize - 1 do
          begin
            Pm := MFinalLens[I];
            if Pm < cEps then Continue;
            Pf := PFinal[I];
            if Pf >= cEps then JS := JS + 0.5 * Pf * Ln(Pf / Pm);
            Pl := PLens[I];
            if Pl >= cEps then JS := JS + 0.5 * Pl * Ln(Pl / Pm);
          end;
          if JS > BestJS then
          begin
            BestJS := JS;
            BestLayer := L;
          end;
        end;
        // Recompute the chosen premature layer's distribution into PLens.
        CandSnap.Copy(NN.Layers[BestLayer].Output);
        NN.Layers[HeadInIdx].Output.CopyNoChecks(CandSnap);
        for I := HeadIdx to LastLayer do NN.Layers[I].Compute();
        Total := NN.GetLastLayer().Output.GetSum();
        if Total <= 0 then Total := 1.0;
        for I := 0 to VocabSize - 1 do
        begin
          Pl := NN.GetLastLayer().Output.Raw[I] / Total;
          if Pl < 0 then Pl := 0;
          PLens[I] := Pl;
        end;
        // (4) Adaptive plausibility constraint: keep only tokens at/above
        //     Alpha * max(p_final); argmax the contrast score over that set.
        Threshold := Alpha * MaxFinal;
        Best := -1;
        BestScore := -1e30;
        for I := 0 to VocabSize - 1 do
          if PFinal[I] >= Threshold then
          begin
            ScoreV := SafeLogProb(PFinal[I]) - SafeLogProb(PLens[I]);
            if (Best < 0) or (ScoreV > BestScore) then
            begin
              BestScore := ScoreV;
              Best := I;
            end;
          end;
        if Best < 0 then  // degenerate empty head set: fall back to final argmax
        begin
          Best := 0;
          for I := 1 to VocabSize - 1 do
            if PFinal[I] > PFinal[Best] then Best := I;
        end;
        Result.SumLogProb := Result.SumLogProb + SafeLogProb(PFinal[Best]);
      end;

      if Best = csDecodeEOSToken then
      begin
        Result.Finished := True;
        Break;
      end;
      Result.Text := Result.Text + Chr(Best);
      Context := Context + Chr(Best);
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
    CandSnap.Free;
  end;
  Result.Score := Result.SumLogProb /
    LengthPenaltyDenominator(Length(Result.Text), 0);
end;

function DecodeSampled(NN: TNNet; const Prompt: string; MaxLen: integer;
  Sampler: TNNetSamplerBase;
  const StopStrings: array of string): TNNetDecodeResult;
var
  InputVolume, OutputVolume: TNNetVolume;
  VocabSize, Step, I, Best, StopLen: integer;
  Total, Pf: TNeuralFloat;
  Context: string;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  VocabSize := OutputVolume.Size;
  Result.Text := '';
  Result.SumLogProb := 0;
  Result.Finished := False;
  Context := Prompt;
  try
    for Step := 1 to MaxLen do
    begin
      InputVolume.OneHotEncodingReversed(Context);
      NN.Compute(InputVolume, OutputVolume);
      if Assigned(Sampler) then
        Best := Sampler.GetToken(OutputVolume)
      else
      begin
        Best := 0;
        for I := 1 to VocabSize - 1 do
          if OutputVolume.Raw[I] > OutputVolume.Raw[Best] then Best := I;
      end;
      if (Best < 0) or (Best >= VocabSize) then Best := 0;
      // Log-prob of the chosen token (re-normalised row, same convention as
      // NextLogProbs) so completions are directly rerankable.
      Total := OutputVolume.GetSum();
      if Total <= 0 then Total := 1.0;
      Pf := OutputVolume.Raw[Best] / Total;
      Result.SumLogProb := Result.SumLogProb + SafeLogProb(Pf);
      if Best = csDecodeEOSToken then
      begin
        Result.Finished := True;
        Break;
      end;
      Result.Text := Result.Text + Chr(Best);
      Context := Context + Chr(Best);
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

function SampleNCompletions(NN: TNNet; const Prompt: string; MaxLen, N: integer;
  Sampler: TNNetSamplerBase; LengthPenalty: TNeuralFloat;
  const StopStrings: array of string;
  Scorer: TNNetSequenceScorer): TNNetDecodeResultArray;
var
  I: integer;
begin
  if N < 1 then N := 1;
  SetLength(Result, N);
  for I := 0 to N - 1 do
  begin
    Result[I] := DecodeSampled(NN, Prompt, MaxLen, Sampler, StopStrings);
    if Assigned(Scorer) then
      Result[I].Score := Scorer(Prompt, Result[I].Text)
    else
      Result[I].Score := Result[I].SumLogProb /
        LengthPenaltyDenominator(Length(Result[I].Text), LengthPenalty);
  end;
end;

function DecodeBestOfN(NN: TNNet; const Prompt: string; MaxLen, N: integer;
  Sampler: TNNetSamplerBase; LengthPenalty: TNeuralFloat;
  const StopStrings: array of string;
  Scorer: TNNetSequenceScorer): TNNetDecodeResult;
var
  Cands: TNNetDecodeResultArray;
  I, Best: integer;
begin
  Cands := SampleNCompletions(NN, Prompt, MaxLen, N, Sampler, LengthPenalty,
    StopStrings, Scorer);
  Best := 0;
  for I := 1 to High(Cands) do
    if Cands[I].Score > Cands[Best].Score then Best := I; // ties keep first
  Result := Cands[Best];
end;

function DecodeSelfConsistency(NN: TNNet; const Prompt: string; MaxLen, N: integer;
  Sampler: TNNetSamplerBase; Extract: TNNetAnswerExtractor;
  const StopStrings: array of string): string;
var
  Cands: TNNetDecodeResultArray;
  Answers: array of string;
  Counts: array of integer;
  I, J, NumDistinct, Best: integer;
  Ans: string;
  Found: boolean;
begin
  if N < 1 then N := 1;
  Cands := SampleNCompletions(NN, Prompt, MaxLen, N, Sampler, 0.0,
    StopStrings, nil);
  // Tally extracted answers in FIRST-SEEN order so ties resolve toward the
  // earliest-drawn modal answer.
  SetLength(Answers, 0);
  SetLength(Counts, 0);
  NumDistinct := 0;
  for I := 0 to High(Cands) do
  begin
    if Assigned(Extract) then Ans := Extract(Cands[I].Text)
    else Ans := Cands[I].Text;
    if Ans = '' then Continue;          // unparseable: ignore in the vote
    Found := False;
    for J := 0 to NumDistinct - 1 do
      if Answers[J] = Ans then
      begin
        Inc(Counts[J]);
        Found := True;
        Break;
      end;
    if not Found then
    begin
      SetLength(Answers, NumDistinct + 1);
      SetLength(Counts, NumDistinct + 1);
      Answers[NumDistinct] := Ans;
      Counts[NumDistinct] := 1;
      Inc(NumDistinct);
    end;
  end;
  if NumDistinct = 0 then
  begin
    Result := '';
    Exit;
  end;
  Best := 0;
  for J := 1 to NumDistinct - 1 do
    if Counts[J] > Counts[Best] then Best := J; // strict > keeps first-seen tie
  Result := Answers[Best];
end;

// Prompt-lookup draft: find the NumDraft characters that follow the MOST RECENT
// EARLIER occurrence of the last MatchLen characters of Context. Returns '' when
// there is no such earlier occurrence (degrade-to-greedy). The suffix itself
// (the final MatchLen characters) is excluded as a match site so the draft is
// always copied from STRICTLY earlier in the string.
function PromptLookupDraft(const Context: string;
  MatchLen, NumDraft: integer): string;
var
  CtxLen, P, FollowLen: integer;
  Suffix: string;
begin
  Result := '';
  CtxLen := Length(Context);
  if (MatchLen < 1) or (NumDraft < 1) or (CtxLen < MatchLen + 1) then Exit;
  Suffix := Copy(Context, CtxLen - MatchLen + 1, MatchLen);
  // Scan candidate start positions from the most recent earlier one backwards;
  // a match at start position P occupies Context[P .. P+MatchLen-1]. The most
  // recent earlier occurrence has the largest P with P+MatchLen-1 < CtxLen, i.e.
  // P <= CtxLen - MatchLen, EXCLUDING P = CtxLen - MatchLen + 1 (the suffix).
  for P := CtxLen - MatchLen downto 1 do
    if Copy(Context, P, MatchLen) = Suffix then
    begin
      FollowLen := CtxLen - (P + MatchLen) + 1; // chars available after match
      if FollowLen > NumDraft then FollowLen := NumDraft;
      if FollowLen >= 1 then
        Result := Copy(Context, P + MatchLen, FollowLen);
      Exit; // most-recent occurrence wins
    end;
end;

function DecodePromptLookup(NN: TNNet; const Prompt: string;
  MaxLen: integer; MatchLen: integer; NumDraft: integer;
  const StopStrings: array of string): TNNetDecodeResult;
var
  InputVolume, OutputVolume: TNNetVolume;
  LogProbs: array of TNeuralFloat;
  VocabSize, Step, I, Best, StopLen, D: integer;
  Context, Draft: string;
begin
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  VocabSize := OutputVolume.Size;
  SetLength(LogProbs, VocabSize);
  Result.Text := '';
  Result.SumLogProb := 0;
  Result.Finished := False;
  Context := Prompt;
  if MatchLen < 1 then MatchLen := 1;
  if NumDraft < 1 then NumDraft := 1;
  try
    Step := 0;
    while Step < MaxLen do
    begin
      // One forward pass over the current context -> next-token greedy argmax.
      NextLogProbs(NN, Context, InputVolume, OutputVolume, LogProbs);
      Best := 0;
      for I := 1 to VocabSize - 1 do
        if LogProbs[I] > LogProbs[Best] then Best := I;
      Result.SumLogProb := Result.SumLogProb + LogProbs[Best];
      Inc(Step);
      if Best = csDecodeEOSToken then
      begin
        Result.Finished := True;
        Break;
      end;
      Result.Text := Result.Text + Chr(Best);
      Context := Context + Chr(Best);
      // Stop-string check on the freshly emitted greedy token (identical policy
      // to DecodeGreedy: trim and finish).
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

      // SPECULATIVE VERIFY of a prompt-lookup draft. The draft characters that
      // follow the most-recent earlier occurrence of the current suffix are
      // verified one prefix at a time; each is accepted only if it equals the
      // model's greedy argmax at that position (bit-identical to greedy).
      Draft := PromptLookupDraft(Context, MatchLen, NumDraft);
      D := 1;
      while (D <= Length(Draft)) and (Step < MaxLen) and
        (not Result.Finished) do
      begin
        NextLogProbs(NN, Context, InputVolume, OutputVolume, LogProbs);
        Best := 0;
        for I := 1 to VocabSize - 1 do
          if LogProbs[I] > LogProbs[Best] then Best := I;
        // Reject as soon as the model disagrees with the draft (or EOS).
        if (Best = csDecodeEOSToken) or (Best <> Ord(Draft[D])) then
        begin
          // The model's argmax here is the next emitted token; fall back to the
          // outer greedy loop, which will recompute exactly this same argmax on
          // its next iteration over the unchanged Context.
          Break;
        end;
        // Accept: identical to the greedy argmax, so emit it.
        Result.SumLogProb := Result.SumLogProb + LogProbs[Best];
        Inc(Step);
        Result.Text := Result.Text + Chr(Best);
        Context := Context + Chr(Best);
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
        Inc(D);
      end;
      if Result.Finished then Break;
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

function DecodeDiverseBeamSearchAll(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer; NumGroups: integer;
  Diversity: TNeuralFloat;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;
var
  InputVolume, OutputVolume: TNNetVolume;
  LogProbs: array of TNeuralFloat;
  TokenTaken: array of integer;   // per-token collision count, this step
  VocabSize, Step, I, T, B, GroupSize, G, GroupLo, GroupHi: integer;
  Live: TBeamArray;      // still-growing beams, contiguous by group
  NewLive: TBeamArray;   // frontier being assembled this step
  Finished: TBeamArray;  // beams that emitted EOS
  Cand: TBeamArray;      // expansion candidates for the CURRENT group
  NewBeam: TBeam;
  Pen: TNeuralFloat;
begin
  if BeamWidth < 1 then BeamWidth := 1;
  if NumGroups < 1 then NumGroups := 1;
  if NumGroups > BeamWidth then NumGroups := BeamWidth;
  // Single group with no diversity penalty is ordinary beam search; delegate so
  // the degenerate case is BIT-FOR-BIT DecodeBeamSearchAll (incl. early-stop).
  if (NumGroups = 1) and (Diversity = 0) then
  begin
    Result := DecodeBeamSearchAll(NN, Prompt, MaxLen, BeamWidth, LengthPenalty);
    Exit;
  end;
  GroupSize := BeamWidth div NumGroups;
  if GroupSize < 1 then GroupSize := 1;
  InputVolume := TNNetVolume.Create(NN.GetFirstLayer.Output);
  OutputVolume := TNNetVolume.Create(NN.GetLastLayer().Output);
  VocabSize := OutputVolume.Size;
  SetLength(LogProbs, VocabSize);
  SetLength(TokenTaken, VocabSize);
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
      for T := 0 to VocabSize - 1 do TokenTaken[T] := 0;
      SetLength(NewLive, 0);

      // Expand group-by-group; each group contributes up to GroupSize survivors
      // and its chosen first tokens raise TokenTaken so LATER groups are pushed
      // away from them (Hamming diversity penalty).
      for G := 0 to NumGroups - 1 do
      begin
        GroupLo := G * GroupSize;
        GroupHi := GroupLo + GroupSize - 1;
        if GroupLo > High(Live) then
        begin
          // No dedicated parents yet (early steps where the live frontier is
          // smaller than the full beam, e.g. the single seed at step 1): the
          // group still explores from the WHOLE current frontier so the
          // diversity penalty can steer it off the earlier groups' tokens.
          GroupLo := 0;
          GroupHi := High(Live);
        end
        else if GroupHi > High(Live) then
          GroupHi := High(Live);

        SetLength(Cand, 0);
        for B := GroupLo to GroupHi do
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
              // Length-penalised base score, minus the diversity penalty for
              // collisions with earlier groups at THIS step (g=0: no penalty).
              Pen := Diversity * TokenTaken[T];
              NewBeam.Score := NewBeam.SumLogProb /
                LengthPenaltyDenominator(Length(NewBeam.Text), LengthPenalty)
                - Pen;
              SetLength(Cand, Length(Cand) + 1);
              Cand[High(Cand)] := NewBeam;
            end;
          end;
        end;

        SortBeamsByScore(Cand);
        if Length(Cand) > GroupSize then SetLength(Cand, GroupSize);
        // Register this group's chosen first tokens for the diversity penalty
        // and append the survivors to the next frontier.
        for I := 0 to High(Cand) do
        begin
          Inc(TokenTaken[Ord(Cand[I].Text[Length(Cand[I].Text)])]);
          SetLength(NewLive, Length(NewLive) + 1);
          NewLive[High(NewLive)] := Cand[I];
        end;
      end;

      // Pruning is per-group (each group keeps GroupSize); the new frontier is
      // the concatenation. No global re-prune so groups stay independent.
      Live := Copy(NewLive, 0, Length(NewLive));
    end;

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

function DecodeDiverseBeamSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer; NumGroups: integer;
  Diversity: TNeuralFloat;
  LengthPenalty: TNeuralFloat): TNNetDecodeResult;
var
  All: TNNetDecodeResultArray;
begin
  All := DecodeDiverseBeamSearchAll(NN, Prompt, MaxLen, BeamWidth,
    NumGroups, Diversity, LengthPenalty);
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

// True when every required phrase in ForceTokens occurs as a substring of Text.
function AllForcedPhrasesPresent(const Text: string;
  const ForceTokens: array of string): boolean;
var
  K: integer;
begin
  Result := True;
  for K := 0 to High(ForceTokens) do
    if (Length(ForceTokens[K]) > 0) and (Pos(ForceTokens[K], Text) = 0) then
    begin
      Result := False;
      Exit;
    end;
end;

// Of the still-unmet phrases, returns the set of characters that can MAKE
// PROGRESS when appended to Text (the next char of any phrase given Text's
// current longest matching prefix-of-a-phrase suffix). Used to inject
// guaranteed-satisfying continuations into the candidate pool. Returns the
// distinct next-chars as a string (each char at most once).
function NeededNextChars(const Text: string;
  const ForceTokens: array of string): string;
var
  K, P, MatchLen: integer;
  Phrase, Tail: string;
  C: char;
begin
  Result := '';
  for K := 0 to High(ForceTokens) do
  begin
    Phrase := ForceTokens[K];
    if (Length(Phrase) = 0) or (Pos(Phrase, Text) > 0) then Continue; // met
    // Longest prefix of Phrase that is a suffix of Text (how far an in-progress
    // emission of this phrase has got); 0 means "start the phrase fresh".
    MatchLen := 0;
    for P := Length(Phrase) - 1 downto 1 do
      if (Length(Text) >= P) and
         (Copy(Text, Length(Text) - P + 1, P) = Copy(Phrase, 1, P)) then
      begin
        MatchLen := P;
        Break;
      end;
    C := Phrase[MatchLen + 1];
    Tail := Result;
    if Pos(C, Tail) = 0 then Result := Result + C;
  end;
end;

// A monotone PROGRESS metric toward satisfying all forced phrases: for each
// phrase, its full length once it is present as a substring, otherwise the
// length of the longest phrase-prefix that is a suffix of Text (an in-progress
// emission). Summed over phrases. Increases as phrases get emitted and reaches
// Sum(Length(phrase)) exactly when all are satisfied.
function ForcedProgress(const Text: string;
  const ForceTokens: array of string): integer;
var
  K, P: integer;
  Phrase: string;
begin
  Result := 0;
  for K := 0 to High(ForceTokens) do
  begin
    Phrase := ForceTokens[K];
    if Length(Phrase) = 0 then Continue;
    if Pos(Phrase, Text) > 0 then
      Inc(Result, Length(Phrase))
    else
      for P := Length(Phrase) - 1 downto 1 do
        if (Length(Text) >= P) and
           (Copy(Text, Length(Text) - P + 1, P) = Copy(Phrase, 1, P)) then
        begin
          Inc(Result, P);
          Break;
        end;
  end;
end;

function DecodeConstrainedBeamSearchAll(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  const ForceTokens: array of string;
  LengthPenalty: TNeuralFloat): TNNetDecodeResultArray;
var
  InputVolume, OutputVolume: TNNetVolume;
  LogProbs: array of TNeuralFloat;
  VocabSize, Step, I, T, B, RealForced, BestProg, KeptProg: integer;
  Live: TBeamArray;
  Finished: TBeamArray;
  Cand: TBeamArray;
  NewBeam: TBeam;
  Needed: string;
begin
  if BeamWidth < 1 then BeamWidth := 1;
  // Count real (non-empty) phrases; none -> ordinary beam search, bit-identical.
  RealForced := 0;
  for I := 0 to High(ForceTokens) do
    if Length(ForceTokens[I]) > 0 then Inc(RealForced);
  if RealForced = 0 then
  begin
    Result := DecodeBeamSearchAll(NN, Prompt, MaxLen, BeamWidth, LengthPenalty);
    Exit;
  end;

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

      SetLength(Cand, 0);
      for B := 0 to High(Live) do
      begin
        NextLogProbs(NN, Prompt + Live[B].Text,
          InputVolume, OutputVolume, LogProbs);
        // Characters that advance an unmet phrase: each is FORCE-INJECTED as a
        // candidate (even if the model assigns it ~0 probability) so a path that
        // makes progress toward every phrase always exists in the pool.
        Needed := NeededNextChars(Live[B].Text, ForceTokens);
        for I := 1 to Length(Needed) do
        begin
          T := Ord(Needed[I]);
          NewBeam.SumLogProb := Live[B].SumLogProb + LogProbs[T];
          NewBeam.Text := Live[B].Text + Chr(T);
          NewBeam.Finished := False;
          NewBeam.Score := NewBeam.SumLogProb /
            LengthPenaltyDenominator(Length(NewBeam.Text), LengthPenalty);
          SetLength(Cand, Length(Cand) + 1);
          Cand[High(Cand)] := NewBeam;
        end;
        for T := 0 to VocabSize - 1 do
        begin
          NewBeam.SumLogProb := Live[B].SumLogProb + LogProbs[T];
          if T = csDecodeEOSToken then
          begin
            // EOS only allowed once ALL phrases are present; otherwise the
            // hypothesis must keep generating to satisfy the constraint.
            if not AllForcedPhrasesPresent(Live[B].Text, ForceTokens) then
              Continue;
            NewBeam.Text := Live[B].Text;
            NewBeam.Finished := True;
            NewBeam.Score := NewBeam.SumLogProb /
              LengthPenaltyDenominator(Length(NewBeam.Text), LengthPenalty);
            SetLength(Finished, Length(Finished) + 1);
            Finished[High(Finished)] := NewBeam;
          end
          else
          begin
            // Skip a token already added through the force-injection pass above
            // (it is a needed next-char) to avoid a duplicate candidate.
            if Pos(Chr(T), Needed) > 0 then Continue;
            NewBeam.Text := Live[B].Text + Chr(T);
            NewBeam.Finished := False;
            NewBeam.Score := NewBeam.SumLogProb /
              LengthPenaltyDenominator(Length(NewBeam.Text), LengthPenalty);
            SetLength(Cand, Length(Cand) + 1);
            Cand[High(Cand)] := NewBeam;
          end;
        end;
      end;

      // Bank-aware prune. Keep the global top-BeamWidth by score, BUT GUARANTEE
      // MONOTONE PROGRESS toward the forced phrases: compute the maximum
      // ForcedProgress reachable this step (over ALL candidates, including the
      // force-injected needed-char ones) and ensure at least one survivor
      // attains it. If the plain top-B prune drops every max-progress
      // candidate, force-keep the best-scoring one in the last slot. Because a
      // strictly-more-advanced hypothesis thus survives every step (and EOS is
      // blocked until all phrases are present), the progress is non-decreasing
      // and reaches Sum(Length(phrase)) - i.e. full satisfaction - within
      // MaxLen (given MaxLen >= total phrase length).
      SortBeamsByScore(Cand);
      if Length(Cand) > BeamWidth then
      begin
        BestProg := 0;
        for I := 0 to High(Cand) do
          if ForcedProgress(Cand[I].Text, ForceTokens) > BestProg then
            BestProg := ForcedProgress(Cand[I].Text, ForceTokens);
        SetLength(Live, BeamWidth);
        for I := 0 to BeamWidth - 1 do Live[I] := Cand[I];
        KeptProg := 0;
        for I := 0 to High(Live) do
          if ForcedProgress(Live[I].Text, ForceTokens) > KeptProg then
            KeptProg := ForcedProgress(Live[I].Text, ForceTokens);
        if KeptProg < BestProg then
          // Best-scoring candidate that attains the max progress (Cand is
          // score-sorted) takes the final slot.
          for I := BeamWidth to High(Cand) do
            if ForcedProgress(Cand[I].Text, ForceTokens) = BestProg then
            begin
              Live[BeamWidth - 1] := Cand[I];
              Break;
            end;
      end
      else
        Live := Copy(Cand, 0, Length(Cand));
    end;

    // Merge surviving live beams that SATISFY the constraint into the pool; an
    // unsatisfied live beam is dropped (it never met the forced phrases within
    // MaxLen). If nothing satisfied, fall back to keeping the surviving beams so
    // the caller still gets a (best-effort) result rather than empty.
    for B := 0 to High(Live) do
      if AllForcedPhrasesPresent(Live[B].Text, ForceTokens) then
      begin
        SetLength(Finished, Length(Finished) + 1);
        Finished[High(Finished)] := Live[B];
      end;
    if Length(Finished) = 0 then
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

function DecodeConstrainedBeamSearch(NN: TNNet; const Prompt: string;
  MaxLen: integer; BeamWidth: integer;
  const ForceTokens: array of string;
  LengthPenalty: TNeuralFloat): TNNetDecodeResult;
var
  All: TNNetDecodeResultArray;
begin
  All := DecodeConstrainedBeamSearchAll(NN, Prompt, MaxLen, BeamWidth,
    ForceTokens, LengthPenalty);
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
  if Temperature < csDecodeMinTemperature then
    Temperature := csDecodeMinTemperature;
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

// ---------------------------------------------------------------------------
// Needle-in-a-haystack harness implementation.

// Insert NeedleFact into Filler at depth DepthFraction (0..1), snapped to the
// nearest space so a word is never sliced mid-token. Depth 0 prepends, depth 1
// appends.
function NeedleSpliceAt(const Filler, NeedleFact: string;
  DepthFraction: TNeuralFloat): string;
var
  Len, Pos: integer;
begin
  Len := Length(Filler);
  if DepthFraction < 0 then DepthFraction := 0;
  if DepthFraction > 1 then DepthFraction := 1;
  Pos := Round(DepthFraction * Len);
  if Pos < 0 then Pos := 0;
  if Pos > Len then Pos := Len;
  // Snap forward to the next space so we splice on a word boundary.
  while (Pos > 0) and (Pos < Len) and (Filler[Pos] <> ' ') do Inc(Pos);
  Result := Copy(Filler, 1, Pos);
  if (Result <> '') and (Result[Length(Result)] <> ' ') then Result := Result + ' ';
  Result := Result + NeedleFact + ' ';
  Result := Result + Copy(Filler, Pos + 1, Len - Pos);
end;

function NeedleInHaystackReport(
  const ContextLengths: array of integer;
  const DepthFractions: array of TNeuralFloat;
  const NeedleFact, NeedleAnswer, Question: string;
  Filler: TNeedleFillerCallback;
  Generate: TNeedleGenerateCallback;
  Data: Pointer): TNeedleInHaystackResult;
var
  d, c: integer;
  FillerText, Haystack, Prompt, Output, AnswerLow: string;
  Cell: TNeedleCell;
  S: TStringList;
  Line: string;
begin
  SetLength(Result.DepthFractions, Length(DepthFractions));
  for d := 0 to High(DepthFractions) do Result.DepthFractions[d] := DepthFractions[d];
  SetLength(Result.ContextLengths, Length(ContextLengths));
  for c := 0 to High(ContextLengths) do Result.ContextLengths[c] := ContextLengths[c];

  SetLength(Result.Cells, Length(DepthFractions), Length(ContextLengths));
  Result.HitCount := 0;
  Result.TotalCount := 0;
  AnswerLow := LowerCase(NeedleAnswer);

  for d := 0 to High(DepthFractions) do
    for c := 0 to High(ContextLengths) do
    begin
      FillerText := Filler(ContextLengths[c], Data);
      Haystack := NeedleSpliceAt(FillerText, NeedleFact, DepthFractions[d]);
      Prompt := Haystack + ' ' + Question;
      Output := Generate(Prompt, Data);

      Cell.ContextLen := ContextLengths[c];
      Cell.DepthFraction := DepthFractions[d];
      Cell.Prompt := Prompt;
      Cell.Output := Output;
      Cell.Hit := (AnswerLow <> '') and (Pos(AnswerLow, LowerCase(Output)) > 0);
      Result.Cells[d][c] := Cell;

      Inc(Result.TotalCount);
      if Cell.Hit then Inc(Result.HitCount);
    end;

  if Result.TotalCount > 0
  then Result.Accuracy := Result.HitCount / Result.TotalCount
  else Result.Accuracy := 0;

  // Render the grid: rows = depth %, cols = context length.
  S := TStringList.Create;
  try
    S.Add('Needle-in-a-Haystack retrieval grid (. = miss, X = hit)');
    Line := 'depth\len ';
    for c := 0 to High(ContextLengths) do
      Line := Line + Format('%7d', [ContextLengths[c]]);
    S.Add(Line);
    for d := 0 to High(DepthFractions) do
    begin
      Line := Format('%7.0f%% ', [DepthFractions[d] * 100]);
      for c := 0 to High(ContextLengths) do
        if Result.Cells[d][c].Hit
        then Line := Line + '      X'
        else Line := Line + '      .';
      S.Add(Line);
    end;
    S.Add(Format('Overall: %d/%d retrieved (%.1f%% accuracy)',
      [Result.HitCount, Result.TotalCount, Result.Accuracy * 100]));
    Result.Report := S.Text;
  finally
    S.Free;
  end;
end;

type
  // Carries the TNNet and MaxLen through the convenience overload's callbacks.
  TNeedleGreedyContext = record
    NN: TNNet;
    MaxLen: integer;
  end;
  PNeedleGreedyContext = ^TNeedleGreedyContext;

function NeedleLoremFiller(CharCount: integer; Data: Pointer): string;
const
  cLorem = 'the quick brown fox jumps over the lazy dog while a calm river ' +
           'flows past green hills and a small village sleeps under stars ';
begin
  Result := '';
  while Length(Result) < CharCount do Result := Result + cLorem;
  Result := Copy(Result, 1, CharCount);
end;

function NeedleGreedyGenerate(const Prompt: string; Data: Pointer): string;
var
  Ctx: PNeedleGreedyContext;
begin
  Ctx := PNeedleGreedyContext(Data);
  Result := DecodeGreedy(Ctx^.NN, Prompt, Ctx^.MaxLen).Text;
end;

function NeedleInHaystackReport(NN: TNNet;
  const ContextLengths: array of integer;
  const DepthFractions: array of TNeuralFloat;
  const NeedleFact, NeedleAnswer, Question: string;
  MaxLen: integer): TNeedleInHaystackResult;
var
  Ctx: TNeedleGreedyContext;
begin
  Ctx.NN := NN;
  Ctx.MaxLen := MaxLen;
  Result := NeedleInHaystackReport(ContextLengths, DepthFractions,
    NeedleFact, NeedleAnswer, Question,
    @NeedleLoremFiller, @NeedleGreedyGenerate, @Ctx);
end;

end.
