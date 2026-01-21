# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CAI Neural API is a Pascal-based deep learning neural network library optimized for AVX, AVX2, AVX512 instruction sets and OpenCL-capable devices (AMD, Intel, NVIDIA). It works with Free Pascal/Lazarus and has partial Delphi support.

## Build Commands

**Build tests:**
```bash
mkdir -p bin
cd tests
fpc -Mobjfpc -Fu../neural -FE../bin -o../bin/RunTests RunTests.lpr
```

**Run tests:**
```bash
./bin/RunTests --all --format=plain
```

**Build an example project (from project directory):**
```bash
# Using lazbuild (Lazarus IDE command-line tool)
lazbuild ProjectName.lpi

# Or using fpc directly
fpc -Mobjfpc -Fu../../neural ProjectName.lpr
```

**Enable OpenCL support:**
Add `-dOpenCL` to compiler flags.

## Architecture

### Core Units (in `neural/` folder)

- **neuralvolume.pas** - `TNNetVolume` is the fundamental array structure. Volumes are 3D arrays (width, height, depth) with AVX/SIMD-optimized operations.
- **neuralnetwork.pas** - Main neural network implementation (`TNNet`). Contains all layer classes (convolution, pooling, activation, normalization, etc.).
- **neuralfit.pas** - Training/fitting classes: `TNeuralImageFit` for image classification, `TNeuralDataLoadingFit` for large datasets, `TNeuralFit` for volume pairs.
- **neuraldatasets.pas** - Dataset loaders for CIFAR-10, CIFAR-100, MNIST, Fashion MNIST, and folder-based image datasets.
- **neuralopencl.pas** / **neuralopenclv.pas** - OpenCL integration for GPU acceleration.
- **neuralthread.pas** - Lightweight parallel processing (`TNeuralThreadList`).
- **neuraltokenizer.pas** - Tokenizer for NLP tasks.

### Key Classes

- `TNNet` - The neural network container. Add layers with `AddLayer()` or `AddLayerAfter()` for multi-path architectures.
- `TNNetVolume` - 3D array with SIMD operations. Access via `Data[x,y,d]` or `Raw[index]`.
- `TNeuralImageFit` - Image classification trainer with built-in data augmentation.
- `TNNetVolumeList` / `TNNetVolumePairList` - Collections for training data.

### Layer Naming Convention

Layers follow the pattern `TNNet<Operation><Activation>`:
- `TNNetConvolutionReLU` - Convolution with ReLU activation
- `TNNetFullConnectLinear` - Dense/fully connected without activation
- `TNNetPointwiseConvReLU` - 1x1 convolution with ReLU

### Multi-path Architectures

Use `AddLayerAfter()` to create branches, then merge with `TNNetConcat` or `TNNetDeepConcat`:
```pascal
InputLayer := NN.AddLayer(TNNetInput.Create(32, 32, 3));
NN.AddLayerAfter(TNNetConvolutionReLU.Create(16, 5, 2, 1), InputLayer);
// ... build first path ...
EndPath1 := NN.AddLayer(...);
NN.AddLayerAfter(TNNetConvolutionReLU.Create(16, 3, 1, 1), InputLayer);
// ... build second path ...
EndPath2 := NN.AddLayer(...);
NN.AddLayer(TNNetDeepConcat.Create([EndPath1, EndPath2]));
```

### Model Persistence

Models are portable across CPU/GPU and architectures:
```pascal
NN.SaveToFile('model.nn');
NN.LoadFromFile('model.nn');
```

## Tests

Tests use the FPCUnit framework. Test files are in `tests/`:
- `TestNeuralVolume.pas` - Volume operations
- `TestNeuralLayers.pas` - Layer forward/backward passes
- `TestNeuralNumerical.pas` - Numerical gradient verification
- `TestNeuralTraining.pas` - Training convergence
- `TestNeuralFit.pas` - Fitting classes

## Examples

Examples are in `examples/`. Key starting points:
- `SimpleImageClassifier/` - CIFAR-10 classification
- `Hypotenuse/` - Learning a simple function
- `XorAndOr/` - Boolean function learning
- `IdentityShortcutConnection/` - ResNet building blocks
- `SimpleNLP/` - Transformer-based text generation

## OpenCL Development

### OpenCL Architecture

OpenCL support is conditionally compiled with `-dOpenCL`. Key files:
- **neural/neural.cl** - OpenCL kernel source code (C99-like syntax)
- **neural/neuralopencl.pas** - OpenCL infrastructure classes (`TEasyOpenCL`, `TDotProductKernel`, etc.)

### Key OpenCL Classes

- `TEasyOpenCL` / `TEasyOpenCLV` - Low-level OpenCL wrapper (context, command queue, buffers)
- `TDotProductKernel` - Manages the dot product kernel used for forward pass
- `TDotProductSharedKernel` - Shared kernel instance for forward pass computation
- `TFCBackpropSharedKernel` - Manages backpropagation kernels for fully connected layers

### Weight Interleaving

For GPU-efficient memory access, weights are stored in **interleaved format**:
```
weight[neuron_index + input_index * num_neurons]
```
This allows coalesced memory access when each GPU thread handles one neuron. The `FConcatedWInter` volume in `TNNetLayerConcatedWeights` stores weights in this format.

### Implementing OpenCL for a Layer

Pattern for adding OpenCL support to a layer:

1. **Add kernel to `neural.cl`** - Write the OpenCL kernel function
2. **Create kernel wrapper class in `neuralopencl.pas`** - Manage buffers and kernel execution
3. **Add OpenCL fields to the layer class** (under `{$IFDEF OpenCL}`):
   - Kernel wrapper instance
   - Any GPU-side buffer volumes
   - Preparation flag
4. **Override `EnableOpenCL()`** - Initialize kernel and buffers
5. **Override `DisableOpenCL()`** - Clean up resources
6. **Implement `ComputeOpenCL()` / `BackpropagateOpenCL()`** - Use the kernel

### OpenCL Kernel Conventions

Kernels in `neural.cl` follow these conventions:
- Prefix: `cai_` (CAI = Conscious Artificial Intelligence)
- Use `get_global_id(0)` and `get_global_id(1)` for 2D work items
- Use `mad()` for fused multiply-add operations
- Activation function parameter: 0=none, 1=ReLU
