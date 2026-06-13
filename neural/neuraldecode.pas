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

end.
