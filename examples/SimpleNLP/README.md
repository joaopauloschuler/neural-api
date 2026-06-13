# Training a Simple Neural Network Model for Text Generation

## Creating a Vocabulary and Tokenizing a Dataset
You can create your own vocabulary with CAI’s tokenizer from the unit [neuraltokenizer.pas](https://github.com/joaopauloschuler/neural-api/blob/master/neural/neuraltokenizer.pas). A vocabulary with 3000 tokens can be created with:
```
var
  X: TNeuralTokenizer;
begin
  X := TNeuralTokenizer.Create;
  X.FitOnFile('datasets/tinystories-10k.txt', 3000);
  X.SaveToFile('datasets/tinystories-vocab-3k-cai.txt');
  X.Free;
end;
```
The dataset can then be tokenized with:
```
var
  X: TNeuralTokenizer;
begin
  X := TNeuralTokenizer.Create;
  X.LoadVocabularyFromFile('datasets/tinystories-vocab-3k-cai.txt');
  X.TokenizeFileToCsv('datasets/tinystories-100k.txt','datasets/tinystories-100k-tokenized3k.csv');
  X.Free;
end;
```
Above files can be found at the [TinyStories4Pascal-Tokenized-v2](https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2) repository.

## Samplers
Samplers are used to probabilistically select the next token (character) from the probabilities guessed by the neural network. The Greedy, Top-K, and Top-P samplers provide different ways to predict the next character in a sequence.

Greedy Sampling:
* Always selects the token with the highest probability at each step.
* Tends to produce repetitive and deterministic output.

Top-K Sampling:
* Samples from the K most likely next tokens at each step.
* K is a parameter that controls diversity - a bigger K leads to more diverse results.

Top-P Sampling:
* Samples from the smallest possible set of tokens whose cumulative probability exceeds P at each step.
* P is a parameter between 0 and 1 controlling diversity - lower P produces less diversity.

In summary:
Greedy sampling takes the most likely token, leading to less diversity. Top-K and Top-P allow controlling diversity by adjusting their parameters.

These samplers are available in plain pascal code:

```
  { TNNetSamplerGreedy }
  TNNetSamplerGreedy = class (TNNetSamplerBase)
    public
      function GetToken(Origin: TNNetVolume): integer; override;
  end;

  { TNNetSamplerTopK }
  TNNetSamplerTopK = class (TNNetSamplerBase)
    protected
      FTopK: integer;
    public
      constructor Create(TopK: integer);
      function GetToken(Origin: TNNetVolume): integer; override;
  end;

  { TNNetSamplerTopP }
  TNNetSamplerTopP = class (TNNetSamplerBase)
    protected
      FTopP: TNeuralFloat;
    public
      constructor Create(TopP: TNeuralFloat);
      function GetToken(Origin: TNNetVolume): integer; override;
  end;
``` 

In this source code example, the sampler is created with  `FSampler := TNNetSamplerTopP.Create(0.4);`

Then, you can just call the following to see the magic:

```
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'one day', FSampler),'.');
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'once upon', FSampler),'.');
    WriteLn(GenerateStringFromCasualNN(NFit.NN, FDictionary, 'billy', FSampler),'.');
```

The loading and saving of neural networks (NN) can be done with:
```
   NN := TNNet.Create;
   NN.LoadFromFile('MyTrainedNeuralNetwork.nn');
   NN.SaveToFile('MyTrainedNeuralNetwork.nn');
```

A small chat bot can be coded with:

```
procedure TestFromFile;
var
  S: string;
  oSampler: TNNetSamplerBase;
  NN: TNNet;
begin
  oSampler := TNNetSamplerTopP.Create(0.6);
  NN := TNNet.Create();
  WriteLn('Loading neural network.');
  NN.LoadFromFile(csAutosavedFileName);
  NN.DebugStructure();
  WriteLn();
  WriteLn('Write something and I will reply.');
  repeat
    Write('User: ');
    ReadLn(S);
    WriteLn('Neural network: ',GenerateStringFromCasualNN(NN, FDictionary, LowerCase(S), oSampler),'.');
  until S = 'exit';
  NN.Free;
  oSampler.Free;
end;
```

## Example with Transformer Decoder, Vocabulary and Multiple Outputs
The currently leading NLP neural network models use tokenized datasets and vocabulary. In this example, the [tokenized dataset](https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2) and its vocabulary can be downloaded with:
```
git clone https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2
unzip TinyStories4Pascal-Tokenized-v2/tinystories-100k-tokenized3k.csv.zip
unzip TinyStories4Pascal-Tokenized-v2/tinystories-vocab-3k-cai.csv.zip
```

Plenty of models are constructed via a stack of transformer decoder modules. This stack can be created with:
```
for I := 1 to 2 do FNN.AddTransformerBlockCAI(8, 2048, true, false, false);
```

Finally, an output layer with one output per input in the context can be added with:
```
    FNN.AddLayer([
      TNNetPointwiseConvLinear.Create(csModelVocabSize),
      TNNetPointwiseSoftMax.Create(1)
    ]);
```
The full source code can be found at [NLP with Vocabulary](https://colab.research.google.com/github/joaopauloschuler/neural-api/blob/master/examples/SimpleNLP/TransformerWithTokenizer.ipynb).

After training, this model produces sentences such as:
* one day, a little girl named lucy went to the park with her mom. lucy saw a big tree with a hole in it. she wanted to climb the tree and see what was on the other .
* once upon a time, there was a little girl named lily. she loved to play with her toys and her favorite toy was a teddy bear. one day, lily's mom asked her to help with.
* billy was a little boy who loved to play outside. one day, he was playing in the park when he saw a big puddle in the grass. he ran over to it and started to .

An advanced source code example can be found [here](https://github.com/joaopauloschuler/gpt-3-for-pascal) .

## Decode-Efficiency Features Bakeoff
[DecodeFeaturesBakeoff.lpr](DecodeFeaturesBakeoff.lpr) benchmarks the decode-efficiency
features on this same TinyStories tokenized workload, one phase per feature:

```
./DecodeFeaturesBakeoff --phase N      (N in 1..9)
bash run_decode_bakeoff.sh             (all nine phases, logs to decode_bakeoff_phaseN.log)
```

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | KV-cache incremental decode (`TNNetScaledDotProductAttention.BeginIncrementalDecode` + `TNNetRotaryEmbedding.PositionOffset`) vs full re-encode | implemented |
| 2 | MTP-heads self-speculative decode (`TNNet.AddMultiTokenPrediction` heads as a built-in draft) | implemented |
| 3 | DiagonalSSM O(1)-per-step decode (`TNNetDiagonalSSM.BeginIncrementalDecode`) | implemented |
| 4 | MLA decoupled-RoPE latent KV cache (`TNNet.AddMultiHeadLatentAttention`) | implemented |
| 5 | Speculative decoding composed with the KV cache (`TruncateCache` rollback) | implemented |
| 6 | Hybrid 2×DiagonalSSM + 1×MLA trunk, dual-family streamed decode (the configuration phases 1–5 recommend) | implemented |
| 7 | Phase 6's hybrid with grouped pointwise convolutions (`TNNetGroupedPointwiseConvLinear` via `AddAutoGroupedPointwiseConv`, incl. the new `MinChannelsPerGroupCount` argument of `TNNet.AddMultiHeadLatentAttention`) | implemented |
| 8 | Gemma-2 alternating local/global attention with a windowed KV cache (`TNNet.AddAlternatingLocalGlobalBlocks`, the sliding-window `Window` argument of `TNNetScaledDotProductAttention`) | implemented |
| 9 | Parallel attention+FFN block step-cost bake-off (`TNNet.AddParallelTransformerBlock`, GPT-J/PaLM single shared pre-norm) | implemented |

Every phase self-budgets to 270 s of wall clock: training is time-boxed inside the
program (\~185 s in phases 1/4/8, \~195 s in phases 2/3, \~190 s in phases 6/7, \~180 s in
phase 9 — its box pays for a pre-train timed forward comparison — and \~160 s draft
training in phase 5), leaving the rest for
the decode benchmark. The model is a small RoPE transformer decoder
(ctx=48, d_model=128, 2 blocks, 8 heads, FFN 512, 3k vocab, \~1.31M params) with
`TNNetDyT` normalization — RoPE and DyT are chosen because both are exactly
streamable token-at-a-time (learned absolute positions and sequence-wide LayerNorm
statistics are not). Phase 1 saves `bakeoff-phase1.nn`; phase 5 loads it and trains a
cheap attention-free TokenShift draft (d=64, FFN 256), then runs greedy speculative
decoding with cached verification and `TruncateCache` rollback on rejection.
Phase 2 uses MTP (Multi-Token Prediction, Gloeckle et al. 2024 / DeepSeek-V3 —
parallel heads that forecast several future tokens at once instead of only the next
one): it trains the same trunk with `TNNet.AddMultiTokenPrediction(NumFuture=3)`
(\~2.08M params): head 0 is the ordinary next-token head, heads 1..2 forecast t+2/t+3
and double as a built-in draft — no second network. Each self-speculative pass is one
forward that verifies the pending drafts (accept-until-first-mismatch; a rejection
still commits head-0's corrected token; full acceptance yields a bonus token) and
drafts the next block from heads 1..2 at the last committed row. Composing the KV
cache with MTP drafting is NOT attempted (after a rejection the drafts come from a
forward whose window a cache never saw — known open problem), so both phase-2 arms
pay full re-encode forwards and the measured win is forwards/token at equal
per-forward cost.
Phase 3 swaps the sequence mixer: same residual skeleton (DyT pre-norm + SwiGLU FFN,
same d_model/blocks/FFN/head), but the mixer is the recurrent `TNNetDiagonalSSM` —
an SSM (State-Space Model, the S4/Mamba family): a linear per-channel recurrence
`h_t = a·h_{t-1} + b·x_t` whose fixed-size hidden state carries the entire past, so
order needs no positional embedding and decoding is O(1) per step with constant memory
wrapped in token-wise in/out projections (\~1.25M params), with NO RoPE and NO
positional embedding — the recurrence carries order. Its decode benchmark is full
re-encode vs the layer's `BeginIncrementalDecode`/`ResetState` O(1)-per-step path,
where the ENTIRE past is a Depth-long state vector per SSM layer (no per-token cache
growth), with step cost reported per prefix-length bucket to show flatness.
Phase 4 swaps the attention for MLA (Multi-head Latent Attention, DeepSeek-V2 — an
attention variant that shrinks the per-token KV cache by factoring K/V through a tiny
shared latent instead of caching full per-head K and V):
`TNNet.AddMultiHeadLatentAttention(128, 8, LatentDim=32,
CausalMask, RopeDim=8)` in the same pre-norm residual block (\~1.29M params); K/V are
low-rank-factored through a 32-wide per-token latent and position enters ONLY through
the decoupled rotated rope slice (token-only embedding). Its cached decode arm uses
the per-head SDPA caches + `PositionOffset` (proving streamed-MLA token-exactness,
rope slice included); the latent-only bytes/token row in its economics table is the
ANALYTIC paper number — the true latent-only decode loop is run in
[examples/LatentAttention](../LatentAttention), not here.
Phase 6 builds the hybrid that phases 1–5 point to as the constrained-CPU
recommendation: a 3-block trunk of TWO phase-3 DiagonalSSM blocks and ONE phase-4 MLA
block (LatentDim 32, RopeDim 8), each block keeping the identical
[DyT → mixer → residual] + [DyT → SwiGLU FFN 512 → residual] skeleton, token-only
embedding (order = the SSM recurrences + the MLA rope slice), d_model 128, ctx 48,
\~1.50M params (3 blocks vs 2 elsewhere). Block order is SSM → MLA → SSM — attention
in the middle, the Jamba/Zamba interleaving default: the first recurrent block hands
the lone attention layer globally-mixed inputs, and a constant-state recurrence (not
a second cache-growing attention) closes the trunk. Trained at lr=0.001 (phase-2/4
reasoning: the 3-block step is costlier than phase 3's 2-block step, so fewer
examples fit in the box). Its decode benchmark is the novel part — no other example
streams BOTH mixer families in one loop: greedy full re-encode per token vs ONE
streamed single-token step per token driving SIMULTANEOUSLY the DiagonalSSM layers'
O(1) incremental state AND the MLA heads' SDPA caches + rope `PositionOffset`. Plain
greedy only (phase 5 showed speculation does not pay on CPU; phase 2's extra MTP
heads cost two more 128→3000 projections), and the phase ends with a printed
three-assumption verdict (convergence vs the pure-mixer references, sample looping,
decode cost/flatness/cache size).
Phase 7 repeats phase 6 with every large dense 1×1 projection swapped for its grouped
alternative (`TNNetGroupedPointwiseConvLinear` through `AddAutoGroupedPointwiseConv`,
MinChannelsPerGroup=16: block-diagonal weights, plus interleave-based intergroup
remixing where input depth ≥ output depth): the SSM in/out projections, the SwiGLU FFN
projections, both output-head projections, and — through the optional
`MinChannelsPerGroupCount` argument of `TNNet.AddMultiHeadLatentAttention` — the MLA
block's Q and output projections. The MLA latent path stays dense on purpose (c_KV must
compress the whole token; the tiny LatentDim→d_model up-projections would get no
intergroup remix). Grouped pointwise convs are strictly per-token, so the phase-6
dual-family streamed decode applies unchanged, and the hard assert doubles as proof
that grouped layers stream token-exactly. Its verdict compares weight count,
convergence and decode cost against phase 6's dense references.
Phase 8 is phase 1's transformer skeleton EXACTLY — `TNNet.AddAlternatingLocalGlobalBlocks`
with phase 1's arguments (DyT pre-norm, RoPE, SwiGLU FFN 512, CausalMask) expands to the
same `AddTransformerEncoderBlock` calls — with only the attention MASKING changed: block 1
is LOCAL (Gemma-2-style sliding window of 12 keys = ctx/4, the new `Window` argument of
`TNNetScaledDotProductAttention`) and block 2 stays GLOBAL (full attention), the Gemma 2
local-first 1:1 interleave. The decode benchmark mirrors phase 1 (full re-encode vs
KV-cached step net) and the hard assert proves the windowed SDPA path streams token-exactly
through the cache. The headline is the MEMORY table: the window mask makes the last
min(t, W) positions provably sufficient decode state, so a local layer's KV cache is
BOUNDED at 2·d_model·W floats while every global layer grows by 2·d_model floats/token —
printed at several prefix lengths against phase 1's all-global two-layer reference,
together with the quality cost of windowing vs phase 1's reference loss.
Phase 9 swaps phase 1's SEQUENTIAL encoder blocks for the PARALLEL attention+FFN
formulation (GPT-J / PaLM / Falcon, `TNNet.AddParallelTransformerBlock`):
y = x + Attn(DyT(x)) + FFN(DyT(x)) with ONE shared pre-norm and a single 3-input residual
sum — same d_model/heads/FFN/RoPE, one DyT fewer per block, so the weight counts match to
within the norm layers (both are printed). The phase measures the parallel block's selling
point — wall clock per step — with a timed full-context forward comparison of the two nets
plus examples-seen inside the same 185 s-style box, compares final loss against phase 1's
reference, and runs the phase-1 decode showdown (the parallel block composes the same
per-head SDPA layers, so the `TNNetStreamingDecoder` KV-cache path applies unchanged; the
hard assert proves cached decode stays token-exact through the 3-input-sum wiring).
All implemented phases HARD-ASSERT token-exact equality of all
decode arms. Every streamed arm drives its weight-copied short-width twin
through a `TNNetStreamingDecoder` session (`neural/neuraldecode.pas`), which
owns the incremental-mode switching, `ResetCache`/`ResetState`, per-forward
RoPE `PositionOffset`, and the speculative `TruncateTo` rollback — one session
class covers the attention-only, SSM-only and dual-family hybrid twins alike.

Dataset setup is the same as above (run from this directory):
```
git clone https://huggingface.co/datasets/schuler/TinyStories4Pascal-Tokenized-v2
unzip TinyStories4Pascal-Tokenized-v2/tinystories-100k-tokenized3k.csv.zip -d datasets/
unzip TinyStories4Pascal-Tokenized-v2/tinystories-vocab-3k-cai.csv.zip -d datasets/
```
The first run filters 24k rows into `datasets/bakeoff_temp.csv` and later runs reuse it.

Measured on a 4-thread x86_64 CPU (AVX2 build; training loss fell 5.02 → 2.87 in the
185 s phase-1 box, two epochs — enough for a convergence clue, not a fluent model):

Phase 1 (greedy, 40 new tokens per prompt, both arms token-identical):
```
Prompt "one day"          full re-encode 33.61 ms/token  KV-cached 1.50 ms/token  speedup 22.4x
Prompt "once upon a time" full re-encode 45.80 ms/token  KV-cached 2.31 ms/token  speedup 19.9x
```

Phase 2 (greedy, 40 new tokens per prompt, both arms token-identical; \~195 s training
box, loss 5.21 → 4.78 — within the box only the nearest future head converges:
accept rate 97.5% at distance t+2 vs 0.0% at t+3; 1.93 committed tokens per
verification pass):
```
arm                ms/token   target-fwd/token
plain greedy         101.24               1.00
self-speculative      53.18               0.53
speedup            : 1.90x wall clock   1.90x forwards
```
Both phase-2 arms run the SAME MTP net and re-encode fully, so wall clock tracks the
forward-pass ratio exactly. Note the measurement-honesty gap vs phase 1: an MTP
forward costs more than a plain-head forward (phase 2's \~101 ms/token plain-greedy
baseline vs phase 1's \~34-46 ms/token is the price of the two extra 128→3000 head
projections), which is why the comparison is plain vs self-speculative from the same
model, not across models.

Phase 3 (greedy, 40 new tokens per prompt, both arms token-identical; \~195 s training
box, loss 4.47 → 2.34 — the recurrent mixer's O(n) forward fits more examples in the
box than phase 1's O(n²) attention and lands at a comparable loss, though its samples
loop sooner at this budget):
```
Prompt "one day"          full re-encode 35.78 ms/token  O(1) SSM step 1.59 ms/token  speedup 22.6x
Prompt "once upon a time" full re-encode 29.22 ms/token  O(1) SSM step 0.45 ms/token  speedup 65.0x
step cost vs prefix length:  <16 tokens 1.00 ms/step   16-31 tokens 1.05 ms/step   >=32 tokens 0.75 ms/step
```
The per-bucket step cost is FLAT in the prefix length (the whole past is a Depth-long
state vector per SSM layer), whereas a KV-cached attention step grows with the cache.

Phase 4 (greedy, 40 new tokens per prompt, both arms token-identical; \~185 s training
box at lr=0.001 — the latent down/up projections and rope-slice plumbing make an MLA
step costlier than phase 1's MHA step — loss 3.90 → 2.53):
```
Prompt "one day"          full re-encode  8.36 ms/token  KV-cached 0.33 ms/token  speedup 25.3x
Prompt "once upon a time" full re-encode 10.03 ms/token  KV-cached 0.39 ms/token  speedup 25.8x

KV-cache memory per token per attention layer (4-byte floats):
  equivalent MHA full K+V (2*d_model):                  256 floats = 1024 bytes
  MLA via per-head SDPA caches, run here (2*H*(d_k+8)): 384 floats = 1536 bytes
  MLA latent-only, analytic (latent+ropeK = 32+8):       40 floats =  160 bytes
```
Latent-only state is 15.6% of the equivalent-MHA cache (6.4x smaller), independent of
head count; the run above proves streamed-MLA faithfulness via the SDPA caches, and
the latent-only row is the analytic number (that loop runs in examples/LatentAttention,
paying an O(t) K/V up-projection recompute per step).

Phase 5 (greedy, K=4 drafts/pass, all three arms token-identical; accept rate 25.2%,
2.00 committed tokens/pass):
```
arm                    ms/token   target-fwd/token   draft-fwd/token
plain full re-encode      46.38               1.00              0.00
plain KV-cached            1.23               1.10              0.00
speculative KV-cached      5.00               0.54              1.99
speedup vs plain-full  : 37.6x (cached)  9.3x (spec+cached)
```

Note the target forwards per token: speculative verification really does halve them
(0.54 vs 1.00). On CPU the plain KV-cached arm still wins wall-clock because a width-5
verify forward costs \~5x a width-1 step here (compute-bound); speculative decoding pays
off when single-token steps are memory-bandwidth-bound (GPUs/batch-1 LLM serving), and
this phase demonstrates the exactness and the forward-count economics of composing it
with the cache.

Phase 6 (greedy, 40 new tokens per prompt, both arms token-identical; \~190 s training
box at lr=0.001, 1,496,582 params; first-logged loss 3.44 → 1.39 final running mean,
three epochs on an idle machine):
```
Prompt "one day"          full re-encode 10.35 ms/token  dual-family step 0.34 ms/token  speedup 30.9x
Prompt "once upon a time" full re-encode 11.57 ms/token  dual-family step 0.37 ms/token  speedup 31.5x
step cost vs prefix length:  <16 tokens 0.304 ms/step   16-31 tokens 0.306 ms/step   >=32 tokens 0.303 ms/step

Streamed-decode cache memory (4-byte floats):
  SSM state, 2 layers x Depth (total, independent of length):  256 floats = 1024 bytes  (CONSTANT)
  MLA block via per-head SDPA caches, run here (2*H*(d_k+8)):  384 floats = 1536 bytes  per token
  MLA block latent-only, analytic (latent+ropeK = 32+8):        40 floats =  160 bytes  per token
  phase-1 transformer reference, 2 layers x 2*d_model:         512 floats = 2048 bytes  per token
```
The three-assumption verdict (printed by the phase itself):
1. **Convergence: HOLDS.** The hybrid's final loss (\~1.39–1.61 across runs, machine-load
   dependent) lands clearly BELOW pure SSM (2.34), pure MLA (2.53) and the transformer
   (2.87) at the same time budget — the single attention block does not slow the
   SSM-dominant trunk down; it helps.
2. **Quality clue: improved but not eliminated.** Samples like "she loved to play with
   her mom and her mom and her" still drift into short loops at this budget, but less
   than phase 3's "there was a time, there was a time…" (the phase prints a repeated-
   trigram count per prompt as a cheap loop metric: 11/80 generated tokens here vs the
   pure-SSM run's tighter immediate loops).
3. **Decode cost: HOLDS.** The streamed dual-family step lands at \~0.35 ms/token on an
   idle machine (\~1.2–1.9 ms/token under heavy load), flat in prefix length
   (0.303/0.306/0.303 ms across the <16/16–31/≥32 buckets) — the SSM state is a
   constant 256 floats TOTAL and only ONE MLA block's cache grows per token (1536
   bytes/token as run via per-head SDPA caches; 160 bytes/token analytic latent-only),
   vs 2048 bytes/token for the phase-1 transformer.

So the hybrid recommendation holds on this workload: 2×SSM + 1×MLA converges at-or-
better than every pure stack, decodes at KV-cached-transformer speed or better with a
quarter to one-tenth of the growing cache, and needs no speculation machinery.

Phase 7 (greedy, 40 new tokens per prompt, both arms token-identical; same \~190 s box
and lr=0.001 as phase 6; 541,574 params — 36.2% of the dense hybrid's 1,496,582,
2.76× smaller):
```
Prompt "one day"          full re-encode 11.22 ms/token  dual-family step 0.30 ms/token  speedup 38.0x
Prompt "once upon a time" full re-encode 10.97 ms/token  dual-family step 0.32 ms/token  speedup 34.3x
step cost vs prefix length:  <16 tokens 0.275 ms/step   16-31 tokens 0.273 ms/step   >=32 tokens 0.275 ms/step
```
Its verdict (printed by the phase): **convergence HOLDS** — the grouped hybrid's loss
fell 8.13 → 1.58, matching the dense reference's 1.61 at 2.76× fewer weights (the
cheaper grouped step also fits more examples into the same time box: 9,600 seen vs
phase 6's typical \~6,400); the samples carried 0 repeated trigrams across 80 generated
tokens ("one day, there was a little girl named lily. she loved to play with her
friends…"); and **decode cost HOLDS** — 0.31 ms/token streamed, flat across the
prefix buckets (0.273–0.275 ms/step). The hard assert doubles as proof that grouped
pointwise convolutions (block-diagonal weights + interleave intergroup remixing)
stream token-exactly through the dual-family loop.

Phase 8 (greedy, 40 new tokens per prompt, both arms token-identical; same \~185 s box
and lr=0.0005 as phase 1; loss 5.01 → 2.50 vs phase 1's all-global 2.87 — at ctx=48
the 12-key window costs nothing at this budget; longer contexts are where the
local/global trade-off bites):
```
Prompt "one day"          full re-encode 67.46 ms/token  KV-cached 1.13 ms/token  speedup 59.8x
Prompt "once upon a time" full re-encode 68.81 ms/token  KV-cached 2.99 ms/token  speedup 23.1x

KV-cache state by prefix length (4-byte floats; analytic bounded-window state):
  prefix        phase-1: 2 global  phase-8: local+global     saved
  12                  6144 floats            6144 floats      0.0%
  24                 12288 floats            9216 floats     25.0%
  48                 24576 floats           15360 floats     37.5%
  192                98304 floats           52224 floats     46.9%
```
The local layer's decode state is BOUNDED at 2·d_model·W = 3072 floats (12 KiB): the
sliding-window mask discards every key older than 12 positions, so the table's
local+global column converges to HALF of the all-global reference as the prefix grows
(Gemma 2's 1:1 interleave; Gemma 3's 5:1 saves \~5/6 of the growing cache). The hard
assert proves the windowed SDPA streams token-exactly through the KV cache.

Phase 9 (greedy, 40 new tokens per prompt, both arms token-identical; \~180 s box at
lr=0.0005; 1,309,186 params vs the sequential reference's 1,309,700 — equal to within
the 514 norm-layer params; loss 4.70 → 2.82 vs phase 1's sequential 2.87 at the same
budget):
```
Timed full-context forward (12 forwards each, same shapes, untrained):
  sequential (phase 1) : 58.36 ms/forward
  parallel  (phase 9)  : 64.21 ms/forward   (1.10x)
Throughput in the box: 149 batches = 4768 examples in 193 s (24.7 examples/s).

Prompt "one day"          full re-encode 70.64 ms/token  KV-cached 2.71 ms/token  speedup 26.1x
Prompt "once upon a time" full re-encode 82.71 ms/token  KV-cached 3.70 ms/token  speedup 22.4x
```
Quality matches the sequential block at the same budget (2.82 vs 2.87), and the cached
decode stays token-exact through the parallel wiring (the same per-head SDPA layers, so
the `TNNetStreamingDecoder` path applies unchanged). The honest step-cost finding: on
THIS CPU library the parallel block is \~10% SLOWER per forward, not faster — the two
branches execute serially layer-by-layer, so PaLM's \~15% step win (which comes from
running the attention and FFN branches CONCURRENTLY on parallel hardware) does not
materialize; what the parallel form buys here is one norm layer fewer per block at
equal quality.
