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
./DecodeFeaturesBakeoff --phase N      (N in 1..5)
bash run_decode_bakeoff.sh             (all five phases, logs to decode_bakeoff_phaseN.log)
```

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | KV-cache incremental decode (`TNNetScaledDotProductAttention.BeginIncrementalDecode` + `TNNetRotaryEmbedding.PositionOffset`) vs full re-encode | implemented |
| 2 | MTP-heads self-speculative decode (`TNNet.AddMultiTokenPrediction` heads as a built-in draft) | implemented |
| 3 | DiagonalSSM O(1)-per-step decode | stub |
| 4 | MLA decoupled-RoPE latent KV cache | stub |
| 5 | Speculative decoding composed with the KV cache (`TruncateCache` rollback) | implemented |

Every phase self-budgets to 270 s of wall clock: training is time-boxed inside the
program (~185 s in phase 1, ~195 s in phase 2, ~160 s draft training in phase 5),
leaving the rest for
the decode benchmark. The model is a small RoPE transformer decoder
(ctx=48, d_model=128, 2 blocks, 8 heads, FFN 512, 3k vocab, ~1.31M params) with
`TNNetDyT` normalization — RoPE and DyT are chosen because both are exactly
streamable token-at-a-time (learned absolute positions and sequence-wide LayerNorm
statistics are not). Phase 1 saves `bakeoff-phase1.nn`; phase 5 loads it and trains a
cheap attention-free TokenShift draft (d=64, FFN 256), then runs greedy speculative
decoding with cached verification and `TruncateCache` rollback on rejection.
Phase 2 trains the same trunk with `TNNet.AddMultiTokenPrediction(NumFuture=3)`
(~2.08M params): head 0 is the ordinary next-token head, heads 1..2 forecast t+2/t+3
and double as a built-in draft — no second network. Each self-speculative pass is one
forward that verifies the pending drafts (accept-until-first-mismatch; a rejection
still commits head-0's corrected token; full acceptance yields a bonus token) and
drafts the next block from heads 1..2 at the last committed row. Composing the KV
cache with MTP drafting is NOT attempted (after a rejection the drafts come from a
forward whose window a cache never saw — known open problem), so both phase-2 arms
pay full re-encode forwards and the measured win is forwards/token at equal
per-forward cost. All implemented phases HARD-ASSERT token-exact equality of all
decode arms.

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

Phase 2 (greedy, 40 new tokens per prompt, both arms token-identical; ~195 s training
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
forward costs more than a plain-head forward (phase 2's ~101 ms/token plain-greedy
baseline vs phase 1's ~34-46 ms/token is the price of the two extra 128→3000 head
projections), which is why the comparison is plain vs self-speculative from the same
model, not across models.

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
verify forward costs ~5x a width-1 step here (compute-bound); speculative decoding pays
off when single-token steps are memory-bandwidth-bound (GPUs/batch-1 LLM serving), and
this phase demonstrates the exactness and the forward-count economics of composing it
with the cache.
