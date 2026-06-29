# CIFAR-10 / CIFAR-100 super-resolution resizer

This program reads the CIFAR-10 (or CIFAR-100) dataset and **resizes every image**
from its native 32×32 up to **64×64 and 128×128** using the repo's learned
super-resolution networks, then writes the results out as PNG files. (See the
companion [`readme.md`](readme.md) for dataset provenance, citation, and links to
the pre-resized image archives on Kaggle.)

## What it does

For each split (train / validation / test) the program builds two cascaded
super-resolution networks and runs them per image:

```
32x32  --CreateResizingNN(32,32)-->  64x64  --CreateResizingNN(64,64)-->  128x128
```

- **`CreateResizingNN`** (`usuperresolutionexample`) builds a `THistoricalNets`
  super-resolution network that doubles each spatial dimension; the weights come
  from the committed example file (`csExampleFileName`). `Compute` / `GetOutput`
  run the forward pass over `TNNetVolume` images.
- Work is split across CPU cores with **`TNeuralThreadList`**
  (`NeuralDefaultThreadCount`): each thread builds its own pair of networks and
  processes its slice of the volume list.
- CIFAR images are loaded with **`CreateCifar10Volumes`** /
  **`CreateCifar100Volumes`** (`neuraldatasets`); the per-channel range is
  rescaled (`Add(2); Mul(64)`) back to 0..255 before saving with
  **`SaveImageFromVolumeIntoFile`**.

CIFAR-10 vs CIFAR-100 is selected by the `IsCifar100` flag in the source (default
`false` = CIFAR-10); flip it and rebuild to resize CIFAR-100.

## Running

```
cd examples/Cifar10Resize
lazbuild Cifar10Resize.lpi --build-mode=Release
./Cifar10Resize
```

(Or compile the `.lpr` with `fpc -Fu../../neural`.) The program checks for the
CIFAR files via `CheckCIFARFile()` and exits if they are missing.

## Inputs / outputs

- **Input:** the CIFAR-10 / CIFAR-100 binary dataset files in the working
  directory (downloaded as the other CIFAR examples expect).
- **Output:** PNGs under `resized/cifar<10|100>-{32,64,128}/{train,test}/class<N>/imgK.png`
  — the original 32×32 alongside the 64×64 and 128×128 upscaled versions, one
  folder per class.

For background on how the resizing network works internally see the
[SuperResolution example](../SuperResolution) and
[issue #26](https://github.com/joaopauloschuler/neural-api/issues/26).
