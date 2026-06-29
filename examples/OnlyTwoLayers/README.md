# OnlyTwoLayers — minimal two-layer networks

Two tiny pedagogical examples that train the smallest possible networks on
hand-built tasks, driving the raw `Compute` / `GetOutput` / `Backpropagate` loop
directly (no `TNeuralFit`).

## OnlyTwoLayersXOrOperation

Learns the **XOR** boolean operation with a 2-input → 2-neuron hidden →
1-neuron-output `TNNet` (`TNNetInput` + two `TNNetFullConnect`). It trains on the
4 XOR truth-table rows for 10,000 epochs at learning rate 0.01 / momentum 0.9.
Booleans are encoded FALSE=0.1, TRUE=0.8 (decision threshold 0.45). Every 1000
epochs it prints the computed vs desired output; at the end it dumps the weights
with `DebugWeights` and waits for ENTER.

## OnlyTwoLayersAbs

Learns `f(x,y) = Abs(x + y)` with a 2-input → 2-neuron `TNNetFullConnectReLU`
hidden → 1-neuron `TNNetFullConnectLinear` output net. It trains on 1,000,000
random `(x, y)` pairs drawn from `[-50, +50]` at learning rate 0.00001 with no
momentum (plain SGD). Every 5000 steps it prints output vs target; at the end it
dumps the weights and waits for ENTER.

## Running

No arguments, no dataset, no download. Build either program and run it.

```
cd examples/OnlyTwoLayers
# build with lazbuild (or fpc), then run the chosen program:
./OnlyTwoLayersXOrOperation
./OnlyTwoLayersAbs
```

Coded by Joao Paulo Schwarz Schuler.
