# SEBlockCifar

Minimal demo of `TNNet.AddSEBlock`, the Squeeze-and-Excitation helper.

An SE block recalibrates per-channel responses by:

1. Squeezing each channel to a scalar via global average pooling (`TNNetAvgChannel`).
2. Learning a bottleneck excite path: `FullConnectReLU(C/r)` -> `FullConnectSigmoid(C)`.
3. Multiplying the resulting gating vector back onto every spatial position of the
   original input branch via `TNNetChannelMulByLayer`.

This example wires a 3 -> 8 conv layer, drops in `AddSEBlock(..., 4)`, and trains
on a tiny synthetic 32x32x3 / 2-class problem for two epochs. The goal is to
verify that the SE wiring forward- and back-propagates cleanly, not to produce
state-of-the-art accuracy.

## Build (FPC)

```
fpc -Mobjfpc -Sh -O2 -Fu../../neural \
    -Fu/usr/share/lazarus/<ver>/components/lazutils/lib/<arch> \
    SEBlockCifar.lpr
./SEBlockCifar
```

Or open `SEBlockCifar.lpi` in Lazarus and press Run.
