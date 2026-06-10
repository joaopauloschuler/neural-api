# Word2Vec skip-gram with negative sampling

This example is a from-scratch implementation of the classic **word2vec
skip-gram with negative sampling** (SGNS; Mikolov et al. 2013) trained on a
tiny **built-in** corpus. It is pure CPU, downloads nothing, is deterministic
(`RandSeed` is fixed), and finishes in a few seconds. It exercises
`TNNetEmbedding` in a **non-transformer, unsupervised** distributional-semantics
setting.

It is distinct from the other NLP examples:

- **`examples/SimpleNLP`** trains a *char-level next-token language model* (a
  small transformer/decoder). That is supervised next-token prediction; its
  embeddings are a by-product.
- **`examples/CharTokenizer`** is about *tokenisation* (text -> ids), not
  representation learning.

Here the **word vectors themselves are the deliverable**, and we evaluate them
with cosine nearest neighbours and the textbook analogy arithmetic.

## What skip-gram with negative sampling is

Skip-gram learns a dense vector per word such that a word predicts the words
around it. For a center word `c` and a context word `w` inside a `+/-window`,
the positive objective wants the dot product `v_c . u_w` to be large. A full
softmax over the vocabulary is expensive, so **negative sampling** replaces it
with cheap binary logistic-regression problems: for each positive `(c, w)` pair
we also draw `K` "noise" words from a **unigram^0.75** table and ask the model
to score them as *not*-a-context. The per-pair loss is the binary
cross-entropy of a sigmoid over the dot product:

```
L = -log sigma( v_c . u_w )  -  sum_{j=1..K} log sigma( -v_c . u_neg_j )
```

so positive dot products are pushed up and negative ones down. The
distributional hypothesis ("words in similar contexts have similar meaning")
then makes the learned vectors cluster semantically.

## How it is wired on `TNNetEmbedding`

Canonical word2vec keeps **two** embedding matrices: an **input** ("center")
matrix `W` and an **output** ("context") matrix `W'`. Each is realised as its
own `Input + TNNetEmbedding(vocab, d)` network (one `(1,1,d)` weight row per
vocabulary id):

```
center net : Input(1, 1, 1)      -> Embedding -> v_c          (1, 1, d)
context net: Input(1 + K, 1, 1)  -> Embedding -> u_w | u_neg  (1+K, 1, d)
```

The SGNS loss and its analytic gradient are computed in Pascal from the two
nets' output slabs (the style of `examples/InfoNCEContrastive` and the repo's
loss-layer pattern). The gradient is seeded with the standard **"target =
output - grad"** trick: `TNNet.Backpropagate(Target)` sets the layer
`OutputError` to `output - target = grad`, and `TNNetEmbedding.Backpropagate`
applies that per-slab gradient straight onto the corresponding word-vector rows.
`EncodeZero=1` is passed so vocabulary id `0` also trains (the layer skips id 0
by default, treating it as padding).

Using **two** matrices rather than one shared matrix is important: with a single
shared matrix the same row is pulled both as a center *and* as a context, and
negative sampling then drives every cosine toward zero (the embeddings collapse
to near-orthogonality and neither neighbours nor analogies emerge). The
two-matrix formulation is also exactly the textbook one. Nearest neighbours and
analogies are read from the **input** matrix `W`.

The gradients used (`sg = sigmoid`):

```
dL/dv_c   = (sg(v_c.u_w) - 1) * u_w  +  sum_j sg(v_c.u_neg_j) * u_neg_j
dL/du_w   = (sg(v_c.u_w) - 1) * v_c
dL/du_neg =  sg(v_c.u_neg_j)      * v_c
```

## The corpus

A small (~400-token, 70-word) but **structured** corpus is embedded as a Pascal
string constant. It deliberately repeats co-occurrence frames over a few word
families so that semantic structure is learnable:

- royalty x gender x age: `king`/`queen`, `prince`/`princess`, `man`/`woman`,
  `boy`/`girl`;
- number: `one`/`two` with singular/plural cue words;
- animals: `dog`/`cat`/`puppy`/`kitten` + `bark`/`meow`;
- food: `bread`/`cheese`/`apple` + `eat`/`eats`.

High-frequency function words (`the`, `a`, `is`, `of`, ...) are kept in the
vocabulary but never used as a center, context, or negative -- the role
word2vec's frequent-word subsampling plays -- which sharply cleans the
co-occurrence signal. Positive pairs are formed **within** a sentence (segments
split by `.`) so the window never crosses sentence boundaries.

## What the example prints

Real output (deterministic):

```
corpus tokens: 401   vocabulary: 70   positive pairs: 242
embed_dim: 24   window: +/-2   negatives K: 8   epochs: 1500   lr: 0.150

Training (two embedding matrices: input/center + output/context)...
  epoch    1   mean SGNS loss =  6.23065
  epoch  100   mean SGNS loss =  1.67044
  ...
  epoch 1500   mean SGNS loss =  1.23030

Nearest neighbours by cosine similarity (input embedding matrix):
  king       ->  queen(0.80)  prince(0.62)  man(0.59)  boy(0.58)  girls(0.57)  women(0.56)
  queen      ->  king(0.80)  prince(0.72)  kingdom(0.70)  princess(0.57)  crown(0.54)  woman(0.50)
  man        ->  woman(0.59)  king(0.59)  boy(0.58)  story(0.56)  prince(0.52)  girls(0.52)
  woman      ->  man(0.59)  prince(0.56)  walk(0.54)  story(0.52)  queen(0.50)  girl(0.47)
  boy        ->  girl(0.72)  prince(0.70)  girls(0.63)  women(0.62)  queens(0.62)  princess(0.59)
  dog        ->  cat(0.71)  cats(0.63)  dogs(0.62)  runs(0.56)  fast(0.56)  girls(0.53)
  cat        ->  dog(0.71)  kitten(0.65)  fast(0.56)  runs(0.56)  likes(0.47)  dogs(0.46)
  bread      ->  eat(0.63)  apple(0.62)  eats(0.57)  fresh(0.56)  cheese(0.52)  people(0.49)
  puppy      ->  kitten(0.80)  bark(0.76)  meow(0.76)  small(0.49)  apples(0.48)  wears(0.48)

Analogy arithmetic (A - B + C ~= ?), top-3 candidates:
  king - man + woman  ->  queen(0.74)  prince(0.57)  crown(0.53)
  prince - man + woman  ->  princess(0.85)  girl(0.76)  queen(0.69)
  puppy - dog + cat  ->  kitten(0.87)  meow(0.77)  bark(0.65)
  queen - woman + man  ->  king(0.80)  kingdom(0.54)  prince(0.53)
  king - prince + princess  ->  eat(0.39)  one(0.38)  bark(0.32)
```

## What worked vs. what didn't at this tiny scale

Honestly, more landed than one might expect from ~400 tokens:

- **Nearest neighbours are all sensible**: `king`<->`queen`, `dog`<->`cat`,
  `cat`->`kitten`, `boy`->`girl`, `bread`->`eat`/`apple`/`cheese`,
  `puppy`->`kitten`. Words sit next to their gender/role/number partners and
  their topical co-occurrents.
- **The textbook analogy lands**: `king - man + woman` returns **`queen`** as
  the top candidate (0.74). So do `prince - man + woman -> princess` (0.85),
  `puppy - dog + cat -> kitten` (0.87), and `queen - woman + man -> king`
  (0.80).
- **What did NOT land**: `king - prince + princess` (an "age/seniority" analogy
  whose intended answer is `queen`) returns unrelated words. Senior/junior is
  the weakest, least-repeated relation in the corpus, so its direction is not
  cleanly encoded. This is exactly the kind of partial result expected at toy
  scale -- the strongly and consistently repeated relations (gender, species)
  generalise; thinly attested ones do not.

Do not over-read the cosine magnitudes: with only 70 words and `d=24` the
geometry is coarse. The point is the *ordering* and the fact that vector
arithmetic recovers the right partner for the well-attested relations.

## Building and running

```
lazbuild Word2VecSkipGram.lpi
../../bin/x86_64-linux/bin/Word2VecSkipGram
```

Pure CPU, deterministic, finishes in well under ten seconds.
