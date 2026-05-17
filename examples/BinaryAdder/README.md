# Binary Adder

Tiny "learns to add two binary numbers" demo. A small MLP learns
`A + B = Sum` over all 256 combinations of two 4-bit operands, with the
5-bit (carry-included) sum encoded as a one-bit-per-channel sigmoid
output. Designed to finish in a few seconds on a single CPU and produce
a deterministic right answer (typically 100% exact-bit accuracy).

## Problem encoding

- Inputs: 8 bits = `[A_3 A_2 A_1 A_0 B_3 B_2 B_1 B_0]`, MSB first.
- Outputs: 5 bits = `[S_4 S_3 S_2 S_1 S_0]`, MSB first; `S_4` is the
  carry-out (the largest sum is `15 + 15 = 30 = 11110b`).
- Training set: the full enumeration of all 256 `(A, B)` pairs.

## Architecture

```
  TNNetInput.Create(8)
  TNNetFullConnectReLU.Create(64)
  TNNetFullConnectReLU.Create(64)
  TNNetFullConnect.Create(5)        // sigmoid activation
```

We frame the problem as five independent per-bit binary regressions
trained with MSE, then read the prediction by rounding each output
channel at 0.5.

## Training loop

This example does NOT use `TNeuralFit`; it calls `NN.Compute`,
`NN.Backpropagate`, `NN.UpdateWeights`, and `NN.ClearDeltas` directly so
it doubles as a tiny reference for hand-rolled mini-batch SGD.

- 256 training pairs (every `(A, B)` combination, enumerated once)
- batch size 16
- 400 epochs, learning rate 0.01, momentum 0.9
- `RandSeed := 42` for reproducibility

MSE and exact-match accuracy on the full 256-case set are printed every
50 epochs. After training, the program prints final loss, a final
exact-match accuracy, and a deterministic spread of 12 sample additions
in `A + B = Sum  predicted: Pred` form.

## Build and run

```
cd examples/BinaryAdder
lazbuild BinaryAdder.lpi
../../bin/x86_64-linux/bin/BinaryAdder
```

Typical run reaches 100% exact-bit accuracy in well under a minute.
