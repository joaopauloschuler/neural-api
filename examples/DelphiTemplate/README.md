# Neural API in Delphi (FireMonkey template)

This is a **starting-point Delphi project** showing how to compile and run
neural-api under **Delphi / FireMonkey (FMX)** rather than only FPC / Lazarus.
It is a minimal GUI skeleton you can copy when embedding the library in a Delphi
desktop application; the example logic itself is intentionally small.

The project is a FMX form (`NeuralInDelphi`, `Unit1`) with two buttons, each
wired to a self-contained demo that uses the library directly:

* **Button1 — `RunNeuralNetwork`**: builds a small sequential CIFAR-10
  convolutional classifier (`TNNetInput(32,32,3)` → a few
  `TNNetConvolutionReLU` layers with pooling/normalization →
  `TNNetFullConnectLinear(10)` → `TNNetSoftMax`) and trains it with
  `TNeuralImageFit` (after `CheckCIFARFile()` fetches the dataset). This mirrors
  the [Simple CIFAR-10 Image Classifier](../SimpleImageClassifier/README.md).
* **Button2 — `RunSimpleLearning`**: trains the smallest possible network
  (`TNNetInput(2)` → `TNNetFullConnectLinear(1)`) to learn the boolean **OR**
  operation, driving the raw `Compute` / `GetOutput` / `Backpropagate` loop by
  hand (no `TNeuralFit`). Booleans are encoded FALSE=0.1, TRUE=0.8. It trains
  for 1200 epochs and prints output vs desired output every 100 epochs, then
  dumps the learned weights with `DebugWeights`.

Both demos write their progress with `WriteLn`, so build the project as a
**console application** (see below) to see the output.

## Delphi project setup

The library has no separate Delphi package — you add its source folder to the
project search path. In the project options:

* In the compiler **search path (`-U`)**, add the `neural` folder: `..\..\neural\`
* Set the **final output directory (`-E`)** to: `..\..\bin\x86_64-win64\bin\`
* Set **"generate console application"** to **true** (so the `WriteLn` output is
  visible).

In your `uses` section, include the neural-api units:

```
  neuralnetwork, neuralvolume, neuraldatasets, neuralfit, neuralthread;
```

## Running

Open `NeuralInDelphi.dproj` in Delphi, build, and run. Click **Button1** to
train the CIFAR-10 classifier or **Button2** to run the OR-operation learning
demo; their output appears in the console window.
