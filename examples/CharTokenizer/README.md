# CharTokenizer

Smallest possible "tokeniser + learned embedding" demo. Builds a
unique-char vocabulary from a hard-coded in-memory corpus, trains a
`TNNetEmbedding` lookup through a single-token next-char-prediction
task, then prints the top-5 nearest characters in embedding space (by
cosine similarity) for a handful of probe characters.

No external data, no file IO, no Lazarus form. Whole run finishes in
about one second on a single CPU core.

## What it shows

The model is the tiniest end-to-end use of `TNNetEmbedding`:

```
TNNetInput(1, 1, 1)             # one char id per example
  -> TNNetEmbedding(Vocab, 16)  # learned per-char lookup row
  -> TNNetFullConnectLinear(Vocab)
  -> TNNetSoftMax               # next-char distribution
```

The corpus is six short pangram-style sentences concatenated in
memory. For training we draw a random position `i` and use the pair
`(corpus[i], corpus[i+1])` -> `(input, target)`, so each embedding
row gets gradient signal from every next-char context the
corresponding character appears in. After training, characters that
predict similar next-char distributions land near each other in the
embedding matrix.

Nearest-neighbour scoring is **cosine similarity** on the rows of
`NN.Layers[1].Neurons[0].Weights` (the `(VocabSize, EmbeddingSize)`
matrix owned by the embedding layer). Each row is L2-normalised, the
probe itself is excluded from its own neighbour list, and ties are
broken by first-seen.

## Build & run

```
lazbuild CharTokenizer.lpi
../../bin/x86_64-linux/bin/CharTokenizer
```

Or directly with fpc:

```
fpc -dRelease -dUseCThreads -O3 -Fu../../neural CharTokenizer.lpr
./CharTokenizer
```

Pure CPU, no external data. Under a second on anything modern. The
run is non-interactive (no trailing `ReadLn`).

## Expected output sketch

```
CharTokenizer demo
  corpus length : 244 chars
  vocabulary    : 30 unique chars
  embedding dim : 16

Architecture:
 Layers: 4
 ...
Layer 0 ... TNNetInput          Output:1,1,1
Layer 1 ... TNNetEmbedding      Output:1,1,16
Layer 2 ... TNNetFullConnectLinear Output:30,1,1
Layer 3 ... TNNetSoftMax        Output:30,1,1

Training next-char prediction for 600 steps of batch 64...
  step    1 / 600   mean-CE=3.40   elapsed=0.0s
  ...
  step  600 / 600   mean-CE=2.09   elapsed=0.8s

Top-5 nearest characters by cosine similarity:
  'q' -> 'j'(1.000), 'l'(0.121), ...
  'e' -> 'n'(0.90), ',' (0.89), '!'(0.89), ...
  't' -> 'p'(0.66), 'l'(0.28), ...
  '_' -> 'o'(0.30), 'u'(0.29), ...   (here '_' means the space char)
```

A few things worth noting about the printout above:

- The corpus is tiny (six sentences). Linguistically clean clusters
  like "all vowels group together" are NOT expected. The point is to
  show the lookup is real and trainable; the gradient signal is
  unambiguous (final-layer CE drops by ~40% within a second).
- `'q'` and `'j'` typically score very high together because both
  appear almost exclusively before `'u'` in the corpus, so their
  next-char distributions are nearly identical.
- The `'_'` row above is the literal space character; we re-label it
  in the printout so the column lines up.
