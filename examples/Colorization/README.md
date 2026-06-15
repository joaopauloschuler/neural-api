# Image Colorization (L → a*b*)

A self-supervised generative-vision example: predict the chroma (CIELAB
`a*`,`b*` channels) of an image from its luminance (`L`) channel alone, so a
grayscale photo can be auto-colorized. No labels are needed — the supervision
is the image's own color.

## How it works

1. Each CIFAR-10 RGB image is converted to **CIELAB** with the existing
   `TNNetVolume.RgbToLab` helper (standard sRGB → linear → XYZ(D65) → Lab; the
   exact inverse is `LabToRgb`).
2. The **L** channel (1 channel, `/100` normalized) is the network **input**.
3. The **a\*,b\*** channels (2 channels, `/110` normalized) are the regression
   **target**.
4. A small conv encoder-decoder built with the reusable
   `TNNet.AddUNet(Depth=2, BaseFeatures=16, OutputChannels=2, …)` is trained
   with plain per-pixel **L2 (MSE)** regression, with a `TNNetHardTanh` head
   keeping the output in the normalized chroma range.
5. After training, a handful of validation images are colorized: the predicted
   `a*,b*` are recombined with the **true L**, mapped back to RGB and written to
   disk as gray-in / color-out pairs.

## Running

The example needs the CIFAR-10 **binary** batches in the working directory:

```
data_batch_1.bin … data_batch_5.bin
test_batch.bin
```

Download from <https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz> and
extract the `*.bin` files next to the binary. Then:

```
lazbuild Colorization.lpi
./Colorization
```

Output `sampleN_gray.png` / `sampleN_color.png` pairs are written under
`./colorized/`. Pure CPU; the default smoke run (2000 training images, 10
epochs) finishes in a few minutes. Raise `csTrainSamples` / epochs for better
quality. For more vivid output, swap the regression head for the classic
quantized-bin classification head (Zhang et al. 2016).

The CIELAB round-trip helper is regression-covered by `TestVolumeLabRoundTrip`
in `tests/TestNeuralVolume.pas`.
