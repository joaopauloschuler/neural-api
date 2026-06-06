# PositionalEncodingDemo

Forward-only ASCII-heatmap inspection of the two additive position-encoding
layers that ship with neural-api:

- `TNNetSinusoidalPositionalEmbedding` (Vaswani et al. sin/cos table)
- `TNNetAddPositionalEmbedding` (despite the name, also a fixed sin/cos table)

The demo builds a tiny `(SeqLen=16, 1, Depth=16)` model around each layer,
feeds an all-zero volume so the layer output IS the encoding table, and
renders the resulting matrix as a 10-bucket ASCII heatmap (` .:-=+*#%@`).

It also runs a short SGD probe against `TNNetAddPositionalEmbedding` and
diffs the table before vs after the probe. The probe confirms empirically
that the table is constant: neither layer registers a `TNNetNeuron` for the
encoding, so `UpdateWeights()` cannot touch it. The two layers are
functionally equivalent additive Vaswani position encoders.

## Build and run

```
cd examples/PositionalEncodingDemo
lazbuild PositionalEncodingDemo.lpi --build-mode=Default
../../bin/x86_64-linux/bin/PositionalEncodingDemo
```

Finishes in well under a second on a single CPU.

## Expected output sketch

```
=== TNNetSinusoidalPositionalEmbedding (Vaswani sin/cos) ===
  shape = (SeqLen=16, Depth=16)   min=-1.0000   max=1.0000
  pos\dep 0123456789012345
    0     =@=@=@=@=@=@=@=@
    1     %*+%=%=%=%=%=%=@
    2     %:#%+%=%=%=%=%=%
    3     + %#+%=%=%=%=%=%
    4     ..%+*%+%=%=%=%=%
    ...
```

The high-frequency oscillation in the leftmost depth columns and the
near-constant high values in the rightmost depth columns are the
hallmark of the sin/cos schedule `1 / base^(2i/D)` with `base=10000`.

The two sanity checks at the bottom of the program output both report
`max |A - B| = 0.00000000`, confirming the table-is-constant and
sinusoidal-equals-add findings.
