# DeadNeuronReport

Tiny example for `TNNet.DeadNeuronReport`, the ReLU-family dead-unit
diagnostic.

The program builds a 3-hidden-layer ReLU MLP (`8 -> 48 -> 48 -> 48 -> 1`)
and trains it briefly on the synthetic regression target `y = ||x||`.
It runs twice:

1. **RUN 1**: sane learning rate (`LR=0.01`). Expect a healthy network
   with a low dead-unit fraction.
2. **RUN 2**: deliberately ruinous learning rate (`LR=5.0`) which knocks
   many ReLU units permanently into the zero-output regime — the classic
   "dying ReLU" failure mode.

After each training run the example prints
`TNNet.DeadNeuronReport(NN, Probes)`, which reports per ReLU-family
layer:

- output shape and total unit count
- dead unit count and dead fraction (units whose `|activation|` was
  `<= DeadThreshold` for **every** probe)
- mean per-sample zero-activation fraction

Plus a 10-bin ASCII histogram of per-layer dead%, the worst layer, and
the network-wide dead total.

## Build & run

```
cd examples/DeadNeuronReport
lazbuild DeadNeuronReport.lpi
../../bin/x86_64-linux/bin/DeadNeuronReport
```

Total runtime is well under a minute.
