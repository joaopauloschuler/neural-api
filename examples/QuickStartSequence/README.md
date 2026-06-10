# Quick-Start Sequence — "hello world" for sequence learning

The shortest possible *"it learns a sequence"* demo: a tiny char-level
next-token model learns a fixed **counting sequence**
`0,1,2,...,9,0,1,2,...` and then continues counting on its own.

## The task

Given the last **two** symbols, predict the next one. Each input is the two
previous digits encoded as concatenated one-hot vectors; the target is the
one-hot of the next digit. The dataset is built programmatically (no files).

## The model

A 4-layer net, a few hundred weights:

```
Input(2*10) -> FullConnectReLU(16) -> FullConnectLinear(10) -> SoftMax
```

Trained with a plain SGD loop (`NN.Compute` + `NN.Backpropagate`, which also
applies the weight update). No `TNeuralFit`, no threads, no data files.

## Build & run

```
lazbuild examples/QuickStartSequence/QuickStartSequence.lpi
./bin/x86_64-linux/bin/QuickStartSequence
```

Pure CPU, trains in ~1–2 seconds.

## Expected output

```
Next-character accuracy: 298/298  (100.0%)

Seed:      "01"
Generated: "012345678901234567890123456789"
Expected:  "012345678901234567890123456789"

SUCCESS: the model learned to continue the counting sequence.
```

Seeded with just `01`, the model regenerates three full cycles by feeding its
own predictions back in — proving it learned the sequence.
