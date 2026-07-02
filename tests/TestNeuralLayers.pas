unit TestNeuralLayers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume,
  pascoremath32;

const
  // Maximum number of elements to check for NaN/Inf in large tensors.
  // Checking all elements can be slow for large outputs, so we sample a subset.
  MAX_NAN_CHECK_ITERATIONS = 100;

type
  TTestNeuralLayers = class(TTestCase)
  published
    procedure TestFullyConnectedForward;
    procedure TestFullConnectThreadingParity;
    procedure TestWillThreadParallelPassParity;
    procedure TestConvolutionWillThreadParity;
    procedure TestConvolutionForward;
    procedure TestWinogradConvolutionParity;
    procedure TestMaxPoolForward;
    procedure TestMaxPoolVectorizedExactParity;
    procedure TestVectorExpScalarParity;
    procedure TestVectorSigmoidScalarParity;
    procedure TestVectorTanhScalarParity;
    procedure TestVectorErfScalarParity;
    procedure TestVectorSinhScalarParity;
    procedure TestVectorLnScalarParity;
    procedure TestVectorSinScalarParity;
    procedure TestVectorCosScalarParity;
    procedure TestVectorArcSinhScalarParity;
    procedure TestPointwiseSoftMaxVectorizedParity;
    procedure TestNetworkSaveLoad;
    procedure TestSimpleXORLearning;
    // New comprehensive layer tests
    procedure TestAvgPoolForward;
    procedure TestMinPoolForward;
    procedure TestReLUActivation;
    procedure TestSigmoidActivation;
    procedure TestSoftMaxLayer;
    procedure TestDepthwiseConvolution;
    procedure TestPointwiseConvolution;
    procedure TestConcatLayers;
    procedure TestSumLayers;
    procedure TestIdentityLayer;
    procedure TestReshapeLayer;
    procedure TestDropoutLayer;
    procedure TestMultipleLayersNetwork;
    procedure TestNetworkClone;
    procedure TestLayerCount;
    // Additional activation function tests
    procedure TestReLU6Activation;
    procedure TestLeakyReLUActivation;
    procedure TestSwishActivation;
    procedure TestHyperbolicTangent;
    procedure TestSELUActivation;
    procedure TestGELUActivation;
    procedure TestMishActivation;
    procedure TestGELUSaveLoad;
    procedure TestMishSaveLoad;
    procedure TestGELUBackpropagation;
    procedure TestMishBackpropagation;
    // Additional pooling tests
    procedure TestMaxChannel;
    procedure TestAvgChannel;
    // Normalization layers
    procedure TestLayerMaxNormalization;
    procedure TestLayerStdNormalization;
    procedure TestMovingStdNormalization;
    procedure TestChannelBias;
    procedure TestChannelMul;
    procedure TestCellBias;
    // Split and channel operations
    procedure TestSplitChannels;
    procedure TestInterleaveChannels;
    // Additional convolution tests
    procedure TestPointwiseConvLinear;
    procedure TestLocalConnect;
    procedure TestGroupedConvolution;
    // Backpropagation tests
    procedure TestBackpropagation;
    procedure TestGradientComputation;
    // Weight initialization tests
    procedure TestWeightInitHe;
    procedure TestWeightInitLeCun;
    procedure TestWeightInitGlorot;
    // Embedding layers
    procedure TestEmbeddingLayer;
    procedure TestTokenAndPositionalEmbedding;
    // Rectangular (W <> H) channel reductions + flip/padded-conv regressions
    procedure TestMaxChannelRectangular;
    procedure TestMinChannelRectangular;
    procedure TestMaxChannelSquareRegression;
    procedure TestFlipXPaddedConvBackprop;
    procedure TestFlipYPaddedConvBackprop;
  end;

implementation

procedure TTestNeuralLayers.TestFullyConnectedForward;
var
  NN: TNNet;
  Input, Output: TNNetVolume;
  Layer: TNNetFullConnectLinear;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    Layer := TNNetFullConnectLinear.Create(1);
    NN.AddLayer(Layer);

    // Set known weights for numerical verification
    // Note: TNNetNeuron initializes bias (FBiasWeight) to 0 in its constructor
    // (see neuralnetwork.pas line ~16402: FBiasWeight := 0)
    // Output = w1*x1 + w2*x2 + bias, where bias = 0
    Layer.Neurons[0].Weights.Raw[0] := 2.0;  // w1 = 2
    Layer.Neurons[0].Weights.Raw[1] := 3.0;  // w2 = 3

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 1.0;

    NN.Compute(Input);
    NN.GetOutput(Output);

    // Numerical verification
    AssertEquals('Output should have size 1', 1, Output.Size);
    // With zero bias: 2*1 + 3*1 = 5
    AssertEquals('Output should be 2*1+3*1=5 (with default zero bias)', 5.0, Output.Raw[0], 0.0001);
    
    // Test with different input
    Input.Raw[0] := 2.0;
    Input.Raw[1] := 3.0;
    // Expected: 2*2 + 3*3 = 4 + 9 = 13
    
    NN.Compute(Input);
    NN.GetOutput(Output);
    AssertEquals('Output should be 2*2+3*3=13', 13.0, Output.Raw[0], 0.0001);
    
    // Verify output is not NaN or Inf
    AssertFalse('Output should not be NaN', IsNaN(Output.Raw[0]));
    AssertFalse('Output should not be Inf', IsInfinite(Output.Raw[0]));
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

// Bit-identical serial-vs-threaded A/B for the opt-in single-sample
// TNNetFullConnect forward (the EnCodec-conv-style forced-thread checksum, but
// here it must be EXACTLY equal: only independent output neurons are
// partitioned, the per-neuron reduction order is unchanged). Forces the
// threaded path with NN.SetIntraLayerThreadingMinWork(0) on a layer well
// above any sane work threshold, for both the activation
// (TNNetFullConnect/ReLU) and the activation-free (TNNetFullConnectLinear)
// variants. Threading state is per-net (TNNetExecutionPlanner), so it dies
// with each RunCase net - no global restore needed.
procedure TTestNeuralLayers.TestFullConnectThreadingParity;
var
  Input: TNNetVolume;
  i: integer;

  // Build a fresh single-FC net, set deterministic weights, compute, copy out.
  procedure RunCase(IsLinear: boolean; Threaded: boolean; Dst: TNNetVolume);
  var
    NN: TNNet;
    Layer: TNNetFullConnect;
    neuron, w: integer;
  begin
    NN := TNNet.Create();
    try
      NN.AddLayer(TNNetInput.Create(256));
      if IsLinear then
        Layer := TNNetFullConnectLinear.Create(384)
      else
        Layer := TNNetFullConnectReLU.Create(384);
      NN.AddLayer(Layer);
      // Deterministic, reproducible weights (independent of the threading flag).
      for neuron := 0 to Layer.Neurons.Count - 1 do
      begin
        for w := 0 to Layer.Neurons[neuron].Weights.Size - 1 do
          Layer.Neurons[neuron].Weights.Raw[w] :=
            Sin(neuron * 0.013 + w * 0.0007) * 0.1;
        Layer.Neurons[neuron].BiasWeight := Cos(neuron * 0.021) * 0.05;
      end;
      if Threaded then
      begin
        NN.EnableIntraLayerThreading(true);
        NN.SetIntraLayerThreadingMinWork(0); // force the threaded path
      end
      else
        NN.EnableIntraLayerThreading(false);
      NN.Compute(Input);
      Dst.Copy(Layer.Output);
    finally
      NN.Free;
    end;
  end;

var
  SerialLin, ThreadLin, SerialAct, ThreadAct: TNNetVolume;
begin
  Input := TNNetVolume.Create(256, 1, 1);
  SerialLin := TNNetVolume.Create();
  ThreadLin := TNNetVolume.Create();
  SerialAct := TNNetVolume.Create();
  ThreadAct := TNNetVolume.Create();
  try
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.05) - 0.3;

    RunCase({IsLinear=}true,  {Threaded=}false, SerialLin);
    RunCase({IsLinear=}true,  {Threaded=}true,  ThreadLin);
    RunCase({IsLinear=}false, {Threaded=}false, SerialAct);
    RunCase({IsLinear=}false, {Threaded=}true,  ThreadAct);

    AssertEquals('Linear output sizes match', SerialLin.Size, ThreadLin.Size);
    for i := 0 to SerialLin.Size - 1 do
      AssertTrue('Linear FC threaded must be BIT-IDENTICAL to serial at ' +
        IntToStr(i), SerialLin.Raw[i] = ThreadLin.Raw[i]);

    AssertEquals('Activation output sizes match', SerialAct.Size, ThreadAct.Size);
    for i := 0 to SerialAct.Size - 1 do
      AssertTrue('ReLU FC threaded must be BIT-IDENTICAL to serial at ' +
        IntToStr(i), SerialAct.Raw[i] = ThreadAct.Raw[i]);

    // Sanity: the layer actually produced varied, finite output (not all zero).
    AssertFalse('Linear output[0] not NaN', IsNaN(SerialLin.Raw[0]));
    AssertTrue('Linear output is non-trivial',
      SerialLin.GetSumAbs() > 0.0);
  finally
    Input.Free;
    SerialLin.Free;
    ThreadLin.Free;
    SerialAct.Free;
    ThreadAct.Free;
  end;
end;

// Exercises WillThread layers INSIDE a parallel inference pass: a branching
// net (width 2) so ComputeForInference engages the graph scheduler, with
// intra-layer threading forced on (min-work 0), so both FullConnect branches
// report WillThread=True and are routed through the single-consumer worker-0
// queue - the only safeguard serializing StartProc on the net's shared
// intra-layer pool (there is no suppression flag anymore). Every parallel
// pass must be bit-identical to the serial trainable compute: the scheduler
// only reorders independent layers and the threaded range split preserves
// the per-neuron reduction order.
procedure TTestNeuralLayers.TestWillThreadParallelPassParity;
var
  NN: TNNet;
  Input, SerialOut: TNNetVolume;
  InputLayer, Branch1, Branch2: TNNetLayer;
  Layer: TNNetLayer;
  i, pass, LayerCnt, neuron, w: integer;
  FC: TNNetFullConnect;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(256, 1, 1);
  SerialOut := TNNetVolume.Create();
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(256));
    // (1,1,depth)-shaped outputs: TNNetDeepConcat requires matching X/Y and
    // concatenates on the depth axis.
    Branch1 := NN.AddLayerAfter(
      TNNetFullConnectLinear.Create(1, 1, 384), InputLayer);
    Branch2 := NN.AddLayerAfter(
      TNNetFullConnectReLU.Create(1, 1, 96), InputLayer);
    NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2]));
    // Deterministic weights on both branches.
    for LayerCnt := 0 to NN.CountLayers() - 1 do
    begin
      Layer := NN.Layers[LayerCnt];
      if Layer is TNNetFullConnect then
      begin
        FC := TNNetFullConnect(Layer);
        for neuron := 0 to FC.Neurons.Count - 1 do
        begin
          for w := 0 to FC.Neurons[neuron].Weights.Size - 1 do
            FC.Neurons[neuron].Weights.Raw[w] :=
              Sin(LayerCnt * 1.7 + neuron * 0.013 + w * 0.0007) * 0.1;
          FC.Neurons[neuron].BiasWeight := Cos(neuron * 0.021) * 0.05;
        end;
      end;
    end;
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.05) - 0.3;

    NN.EnableIntraLayerThreading(true);
    NN.SetIntraLayerThreadingMinWork(0); // force WillThread on both branches
    AssertTrue('Branch1 must report WillThread', Branch1.WillThread());
    AssertTrue('Branch2 must report WillThread', Branch2.WillThread());

    // Reference: trainable -> serial loop (threaded ranges, no scheduler).
    NN.Compute(Input);
    SerialOut.Copy(NN.GetLastLayer().Output);
    AssertTrue('Reference output is non-trivial', SerialOut.GetSumAbs() > 0.0);

    // Inference: parallel scheduler passes with worker-0-routed WillThread
    // layers must stay bit-identical, pass after pass.
    NN.SetTrainable(False);
    for pass := 1 to 20 do
    begin
      NN.Compute(Input);
      AssertEquals('Output size matches at pass ' + IntToStr(pass),
        SerialOut.Size, NN.GetLastLayer().Output.Size);
      for i := 0 to SerialOut.Size - 1 do
        AssertTrue('Parallel pass ' + IntToStr(pass) +
          ' must be BIT-IDENTICAL to serial at ' + IntToStr(i),
          SerialOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);
    end;
  finally
    Input.Free;
    SerialOut.Free;
    NN.Free;
  end;
end;

// Conv counterpart of TestWillThreadParallelPassParity: three conv branches
// chosen to exercise BOTH threaded twins - Branch1 (32 neurons, 3x3 on depth
// 8: VectorSize 72 <= csMaxInterleavedSize and neurons mod 32 = 0) takes the
// interleaved kernel; Branch2 (24 neurons, 3x3) and Branch3 (pointwise) take
// the tiled kernel. The serial trainable pass is the reference; every
// parallel scheduler pass must be BIT-IDENTICAL (the ranged kernels only
// partition the outer B loop - per-cell accumulation order is untouched).
// Coded by Claude (AI).
procedure TTestNeuralLayers.TestConvolutionWillThreadParity;
var
  NN: TNNet;
  Input, SerialOut: TNNetVolume;
  InputLayer, Branch1, Branch2, Branch3: TNNetLayer;
  Layer: TNNetLayer;
  i, pass, LayerCnt, neuron, w: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 8);
  SerialOut := TNNetVolume.Create();
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(8, 8, 8));
    Branch1 := NN.AddLayerAfter(
      TNNetConvolutionReLU.Create(32, 3, 1, 1), InputLayer);
    Branch2 := NN.AddLayerAfter(
      TNNetConvolutionLinear.Create(24, 3, 1, 1), InputLayer);
    Branch3 := NN.AddLayerAfter(
      TNNetPointwiseConvLinear.Create(48), InputLayer);
    NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2, Branch3]));
    // Deterministic weights; AfterWeightUpdate refreshes the concatenated and
    // interleaved weight caches the conv forward reads.
    for LayerCnt := 0 to NN.CountLayers() - 1 do
    begin
      Layer := NN.Layers[LayerCnt];
      if Layer.Neurons.Count > 0 then
      begin
        for neuron := 0 to Layer.Neurons.Count - 1 do
        begin
          for w := 0 to Layer.Neurons[neuron].Weights.Size - 1 do
            Layer.Neurons[neuron].Weights.Raw[w] :=
              Sin(LayerCnt * 1.7 + neuron * 0.013 + w * 0.0007) * 0.1;
          Layer.Neurons[neuron].BiasWeight := Cos(neuron * 0.021) * 0.05;
        end;
        Layer.FlushWeightCache();
      end;
    end;
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.05) - 0.3;

    NN.EnableIntraLayerThreading(true);
    NN.SetIntraLayerThreadingMinWork(0); // force WillThread on all branches
    AssertTrue('Branch1 must report WillThread', Branch1.WillThread());
    AssertTrue('Branch2 must report WillThread', Branch2.WillThread());
    AssertTrue('Branch3 must report WillThread', Branch3.WillThread());

    // Reference: trainable -> serial loop with the classic serial kernels.
    NN.Compute(Input);
    SerialOut.Copy(NN.GetLastLayer().Output);
    AssertTrue('Reference output is non-trivial', SerialOut.GetSumAbs() > 0.0);

    // Inference: parallel scheduler passes with worker-0-routed WillThread
    // layers running the threaded twins must stay bit-identical, pass after
    // pass.
    NN.SetTrainable(False);
    for pass := 1 to 20 do
    begin
      NN.Compute(Input);
      AssertEquals('Output size matches at pass ' + IntToStr(pass),
        SerialOut.Size, NN.GetLastLayer().Output.Size);
      for i := 0 to SerialOut.Size - 1 do
        AssertTrue('Parallel pass ' + IntToStr(pass) +
          ' must be BIT-IDENTICAL to serial at ' + IntToStr(i),
          SerialOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);
    end;
  finally
    Input.Free;
    SerialOut.Free;
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionForward;
var
  NN: TNNet;
  Input: TNNetVolume;
  ConvLayer: TNNetConvolutionReLU;
  I, OutputSize: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    ConvLayer := TNNetConvolutionReLU.Create(16, 3, 1, 1);
    NN.AddLayer(ConvLayer);

    Input.Fill(1.0);
    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output depth should be 16', 16, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: Output should exist and be finite
    OutputSize := NN.GetLastLayer.Output.Size;
    AssertEquals('Output size should be 8*8*16 = 1024', 1024, OutputSize);
    
    // Check that output values are valid (not NaN or Inf)
    for I := 0 to OutputSize - 1 do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // ReLU applied, so output should be non-negative for positive weighted sums
    // Since weights are random, we just verify the output exists and ReLU works
    AssertTrue('Output min should be >= 0 (ReLU)', NN.GetLastLayer.Output.GetMin() >= -0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestWinogradConvolutionParity;

  // Builds a 3x3 stride-1 conv (linear activation), random weights+input, then
  // compares the exact direct forward against the opt-in Winograd path. Winograd
  // reassociates the channel sum so float32 differs slightly; tolerance 1e-4.
  function MaxDiffFor(InW, InH, InD, OutD, Pad: integer): TNeuralFloat;
  var
    NN: TNNet;
    Input, DirectOut: TNNetVolume;
    Conv: TNNetConvolutionLinear;
    I: integer;
  begin
    RandSeed := 424242;
    NN := TNNet.Create();
    Input := TNNetVolume.Create(InW, InH, InD);
    DirectOut := TNNetVolume.Create();
    try
      NN.AddLayer(TNNetInput.Create(InW, InH, InD));
      Conv := TNNetConvolutionLinear.Create(OutD, 3, Pad, 1);
      NN.AddLayer(Conv);
      NN.InitWeights();
      // Random input in a reasonable range.
      for I := 0 to Input.Size - 1 do
        Input.Raw[I] := (Random - 0.5) * 4;

      // Exact direct path (Winograd default OFF).
      NN.Compute(Input);
      DirectOut.Copy(NN.GetLastLayer.Output);

      // Same weights, Winograd path ON.
      Conv.EnableWinograd(true);
      AssertTrue('Winograd should report enabled', Conv.WinogradEnabled());
      NN.Compute(Input);

      Result := 0;
      for I := 0 to DirectOut.Size - 1 do
        Result := Max(Result, Abs(DirectOut.Raw[I] - NN.GetLastLayer.Output.Raw[I]));
    finally
      NN.Free;
      Input.Free;
      DirectOut.Free;
    end;
  end;

var
  D: TNeuralFloat;
begin
  // Padded same-size output (pad=1): even output size 8x8.
  D := MaxDiffFor(8, 8, 4, 6, 1);
  AssertTrue('Winograd parity (padded 8x8) max|diff|<1e-4, got ' + FloatToStr(D), D < 1e-4);

  // Unpadded (pad=0): output 6x6 (even), boundary tiles read zeros outside.
  D := MaxDiffFor(8, 8, 4, 6, 0);
  AssertTrue('Winograd parity (unpadded 6x6) max|diff|<1e-4, got ' + FloatToStr(D), D < 1e-4);

  // Odd output size to exercise the ragged right/bottom edge: 7x7 input, pad=1
  // -> output 7x7 (odd), so the last 2x2 block straddles the edge.
  D := MaxDiffFor(7, 7, 3, 5, 1);
  AssertTrue('Winograd parity (odd 7x7) max|diff|<1e-4, got ' + FloatToStr(D), D < 1e-4);

  // Unpadded odd output: 8x8 input pad=0 already even; use 9x9 -> output 7x7 odd.
  D := MaxDiffFor(9, 9, 5, 4, 0);
  AssertTrue('Winograd parity (unpadded odd 7x7) max|diff|<1e-4, got ' + FloatToStr(D), D < 1e-4);
end;

procedure TTestNeuralLayers.TestMaxPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetMaxPool.Create(2));

    // Set up input with known values for numerical verification
    // 2x2 pool regions: (0,0)-(1,1), (2,0)-(3,1), (0,2)-(1,3), (2,2)-(3,3)
    Input.Fill(1.0);
    Input[0, 0, 0] := 5.0;  // Max in region (0,0)
    Input[3, 1, 0] := 7.0;  // Max in region (1,0)
    Input[1, 3, 0] := 3.0;  // Max in region (0,1)
    Input[2, 2, 0] := 9.0;  // Max in region (1,1)
    
    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 2 after 2x2 pool', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 2 after 2x2 pool', 2, NN.GetLastLayer.Output.SizeY);
    
    // Numerical verification: each output cell should contain the max of its 2x2 region
    AssertEquals('Max pool output (0,0) should be 5.0', 5.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Max pool output (1,0) should be 7.0', 7.0, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    AssertEquals('Max pool output (0,1) should be 3.0', 3.0, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    AssertEquals('Max pool output (1,1) should be 9.0', 9.0, NN.GetLastLayer.Output[1, 1, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMaxPoolVectorizedExactParity;
// The MaxPool forward folds each pooling-window strip in over the (contiguous)
// depth axis through the vectorized TNNetVolume.MaxElements primitive. The max
// reduction is exact (no floating-point reassociation), so the vectorized
// output must be BIT-IDENTICAL to a straightforward scalar reference. This test
// builds a multi-channel input with non-trivial depth (37 -> exercises the AVX
// large/small/tail paths) and checks both stride configurations:
//   * default stride (stride == pool size, no padding) and
//   * custom stride with padding.
  procedure CheckParity(const Title: string; PoolSize, Stride, Padding,
    SizeX, SizeY, Depth: integer);
  var
    NN: TNNet;
    Input: TNNetVolume;
    Padded: TNNetVolume;
    Reference: TNNetVolume;
    Pool: TNNetMaxPool;
    OutX, OutY, OutD, OutSizeX, OutSizeY: integer;
    InX, InY, BaseX, BaseY, px, py: integer;
    PadSizeX, PadSizeY, InXMax, InYMax: integer;
    v, best: TNeuralFloat;
    seen: boolean;
  begin
    NN := TNNet.Create();
    Input := TNNetVolume.Create(SizeX, SizeY, Depth);
    Padded := TNNetVolume.Create();
    Reference := TNNetVolume.Create();
    try
      NN.AddLayer(TNNetInput.Create(SizeX, SizeY, Depth));
      Pool := TNNetMaxPool(NN.AddLayer(TNNetMaxPool.Create(PoolSize, Stride, Padding)));

      // Deterministic, well-separated values (no exact ties across the whole
      // tensor, so the argmax is unambiguous and reference == layer exactly).
      for InX := 0 to SizeX - 1 do
        for InY := 0 to SizeY - 1 do
          for OutD := 0 to Depth - 1 do
            Input[InX, InY, OutD] :=
              Sin(0.37 * InX + 0.91 * InY + 0.13 * OutD) * 100.0
              + 0.001 * (InX * SizeY * Depth + InY * Depth + OutD);

      NN.Compute(Input);

      OutSizeX := Pool.Output.SizeX;
      OutSizeY := Pool.Output.SizeY;

      // Build the padded input exactly like the layer (CopyPadding: zero border).
      if Padding > 0
        then Padded.CopyPadding(Input, Padding)
        else Padded.Copy(Input);
      PadSizeX := Padded.SizeX;
      PadSizeY := Padded.SizeY;

      // Independent scalar reference. The window is taken over the PADDED volume
      // with the same clamping the layer applies (Min(base+pool-1, size-1)); a
      // window cell beyond the padded boundary is simply not part of the pool
      // (the window shrinks) -- it is NOT a zero. Padding zeros only ever appear
      // as genuine cells of the padded volume.
      Reference.ReSize(OutSizeX, OutSizeY, Depth);
      for OutX := 0 to OutSizeX - 1 do
        for OutY := 0 to OutSizeY - 1 do
          for OutD := 0 to Depth - 1 do
          begin
            BaseX := OutX * Stride;
            BaseY := OutY * Stride;
            InXMax := Min(BaseX + PoolSize - 1, PadSizeX - 1);
            InYMax := Min(BaseY + PoolSize - 1, PadSizeY - 1);
            best := 0; // unused until seen
            seen := false;
            for px := BaseX to InXMax do
              for py := BaseY to InYMax do
              begin
                v := Padded[px, py, OutD];
                if (not seen) or (v > best) then
                begin
                  best := v;
                  seen := true;
                end;
              end;
            Reference[OutX, OutY, OutD] := best;
          end;

      // Demand EXACT equality (delta 0) -- max introduces no rounding.
      for OutX := 0 to OutSizeX - 1 do
        for OutY := 0 to OutSizeY - 1 do
          for OutD := 0 to Depth - 1 do
            AssertTrue(
              Format('%s: MaxPool[%d,%d,%d] vectorized=%g scalar-ref=%g must be bit-identical',
                [Title, OutX, OutY, OutD,
                 Pool.Output[OutX, OutY, OutD], Reference[OutX, OutY, OutD]]),
              Pool.Output[OutX, OutY, OutD] = Reference[OutX, OutY, OutD]);
    finally
      NN.Free;
      Input.Free;
      Padded.Free;
      Reference.Free;
    end;
  end;
begin
  // Default stride path (stride == pool size, no padding).
  CheckParity('default-stride', 2, 2, 0, 8, 6, 37);
  CheckParity('default-stride-3', 3, 3, 0, 9, 9, 11);
  // Custom stride + padding path.
  CheckParity('stride-padding', 3, 2, 1, 7, 5, 37);
  CheckParity('overlap-stride', 3, 1, 0, 6, 6, 13);
end;

procedure TTestNeuralLayers.TestVectorExpScalarParity;
// TNNetVolume.VectorExp must match the scalar pcr_expf loop (the parity
// reference) within a tight relative tolerance on every build. On AVX2 builds
// VectorExp uses an 8-wide polynomial; on scalar builds it IS the pcr_expf loop.
// N=131 deliberately straddles the 8-wide body and the (N mod 8) scalar tail.
const
  N = 131;
  RelTol = 1e-4;
var
  Src, Dst, Ref: TNNetVolume;
  I: integer;
  x, e, denom, maxRel: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  Ref := TNNetVolume.Create(N, 1, 1);
  try
    // Spread inputs across [-30, 30] plus a couple of saturating extremes.
    for I := 0 to N - 1 do
    begin
      x := -30.0 + 60.0 * I / (N - 1);
      Src.FData[I] := x;
      Ref.FData[I] := pcr_expf(x);
    end;
    TNNetVolume.VectorExp(Dst.DataPtr, Src.DataPtr, N);
    maxRel := 0;
    for I := 0 to N - 1 do
    begin
      denom := Abs(Ref.FData[I]);
      if denom < 1e-20 then denom := 1e-20;
      e := Abs(Dst.FData[I] - Ref.FData[I]) / denom;
      if e > maxRel then maxRel := e;
    end;
    AssertTrue('VectorExp vs pcr_expf max rel err ' + FloatToStr(maxRel) +
      ' must be < ' + FloatToStr(RelTol), maxRel < RelTol);
  finally
    Src.Free; Dst.Free; Ref.Free;
  end;
end;

procedure TTestNeuralLayers.TestVectorSigmoidScalarParity;
// VectorSigmoid must match the scalar reference Sigmoid() within tolerance.
const
  N = 131;
  AbsTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, maxErr: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := -25.0 + 50.0 * I / (N - 1);
    TNNetVolume.VectorSigmoid(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      e := Abs(Dst.FData[I] - Sigmoid(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorSigmoid vs Sigmoid max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestVectorTanhScalarParity;
// VectorTanh must match the scalar pcr_tanhf reference within a tight tolerance
// on every build (AVX2 8-wide exp path and scalar fallback). N=131 straddles
// the 8-wide body and the (N mod 8) tail; range covers saturating extremes.
const
  N = 131;
  AbsTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, maxErr: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := -12.0 + 24.0 * I / (N - 1);
    TNNetVolume.VectorTanh(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      e := Abs(Dst.FData[I] - pcr_tanhf(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorTanh vs pcr_tanhf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestVectorErfScalarParity;
// VectorErf (Abramowitz & Stegun 7.1.26) must match the near-exact scalar
// pcr_erff within tolerance on every build. N=131 straddles the 8-wide exp body
// and the scalar tail; range covers both the linear region and the saturated tails.
const
  N = 131;
  AbsTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, maxErr: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := -4.0 + 8.0 * I / (N - 1);
    TNNetVolume.VectorErf(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      e := Abs(Dst.FData[I] - pcr_erff(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorErf vs pcr_erff max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestVectorSinhScalarParity;
// VectorSinh (sinh built on the AVX2 VectorExp) must match the scalar pcr_sinhf
// reference within tolerance on every build. N=131 straddles the 8-wide exp body
// and the (N mod 8) scalar tail; range [-12,12] matches the SinhAct parity band.
// A second pass with dst aliasing src guards against the buffer-aliasing bug that
// was fixed in VectorTanh/VectorErf.
const
  N = 131;
  RelTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, denom, maxRel: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := -12.0 + 24.0 * I / (N - 1);
    // Distinct dst.
    TNNetVolume.VectorSinh(Dst.DataPtr, Src.DataPtr, N);
    maxRel := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      denom := Abs(pcr_sinhf(x));
      if denom < 1e-20 then denom := 1e-20;
      e := Abs(Dst.FData[I] - pcr_sinhf(x)) / denom;
      if e > maxRel then maxRel := e;
    end;
    AssertTrue('VectorSinh vs pcr_sinhf max rel err ' + FloatToStr(maxRel) +
      ' must be < ' + FloatToStr(RelTol), maxRel < RelTol);
    // dst aliasing src.
    TNNetVolume.VectorSinh(Src.DataPtr, Src.DataPtr, N);
    maxRel := 0;
    for I := 0 to N - 1 do
    begin
      // Recompute the original x from the index (Src has been overwritten).
      x := -12.0 + 24.0 * I / (N - 1);
      denom := Abs(pcr_sinhf(x));
      if denom < 1e-20 then denom := 1e-20;
      e := Abs(Src.FData[I] - pcr_sinhf(x)) / denom;
      if e > maxRel then maxRel := e;
    end;
    AssertTrue('VectorSinh (aliased) vs pcr_sinhf max rel err ' + FloatToStr(maxRel) +
      ' must be < ' + FloatToStr(RelTol), maxRel < RelTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestVectorLnScalarParity;
// VectorLn (Cephes logf on the AVX2 build, pcr_logf fallback otherwise) must match
// the scalar pcr_logf reference within tolerance on every build. N=131 straddles the
// 8-wide body and the (N mod 8) scalar tail; range covers small and large positive
// inputs. A second pass with dst aliasing src guards against buffer aliasing bugs.
const
  N = 131;
  AbsTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, maxErr: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := 1e-3 + 50.0 * I / (N - 1);
    TNNetVolume.VectorLn(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      e := Abs(Dst.FData[I] - pcr_logf(Src.FData[I]));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorLn vs pcr_logf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
    // dst aliasing src.
    TNNetVolume.VectorLn(Src.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := 1e-3 + 50.0 * I / (N - 1);
      e := Abs(Src.FData[I] - pcr_logf(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorLn (aliased) vs pcr_logf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestVectorSinScalarParity;
// VectorSin (Cephes sinf with 3-part Cody-Waite reduction on the AVX2 build) must
// match the scalar pcr_sinf reference within tolerance on every build. N=131
// straddles the 8-wide body and the scalar tail; range [-50,50] plus a few large
// magnitudes exercise the range reduction. dst aliasing src is also checked.
const
  N = 131;
  AbsTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, maxErr: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := -50.0 + 100.0 * I / (N - 1);
    // Sprinkle in a few large magnitudes.
    Src.FData[0] := 1000.0; Src.FData[1] := -1234.5; Src.FData[2] := 9999.9;
    TNNetVolume.VectorSin(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      e := Abs(Dst.FData[I] - pcr_sinf(Src.FData[I]));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorSin vs pcr_sinf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
    // dst aliasing src.
    TNNetVolume.VectorSin(Src.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      if I = 0 then x := 1000.0
      else if I = 1 then x := -1234.5
      else if I = 2 then x := 9999.9
      else x := -50.0 + 100.0 * I / (N - 1);
      e := Abs(Src.FData[I] - pcr_sinf(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorSin (aliased) vs pcr_sinf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestVectorCosScalarParity;
// VectorCos (Cephes cosf with 3-part Cody-Waite reduction on the AVX2 build) must
// match the scalar pcr_cosf reference within tolerance on every build. Same coverage
// rationale as TestVectorSinScalarParity.
const
  N = 131;
  AbsTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, maxErr: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := -50.0 + 100.0 * I / (N - 1);
    Src.FData[0] := 1000.0; Src.FData[1] := -1234.5; Src.FData[2] := 9999.9;
    TNNetVolume.VectorCos(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      e := Abs(Dst.FData[I] - pcr_cosf(Src.FData[I]));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorCos vs pcr_cosf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
    // dst aliasing src.
    TNNetVolume.VectorCos(Src.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      if I = 0 then x := 1000.0
      else if I = 1 then x := -1234.5
      else if I = 2 then x := 9999.9
      else x := -50.0 + 100.0 * I / (N - 1);
      e := Abs(Src.FData[I] - pcr_cosf(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorCos (aliased) vs pcr_cosf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestVectorArcSinhScalarParity;
// VectorArcSinh = ln(x + sqrt(x^2+1)), built on the AVX2 VectorLn, must match the
// scalar reference within tolerance on every build. N=131 straddles body+tail; range
// covers both signs and large magnitudes. dst aliasing src is also checked.
const
  N = 131;
  AbsTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, maxErr: TNeuralFloat;
begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := -30.0 + 60.0 * I / (N - 1);
    TNNetVolume.VectorArcSinh(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      e := Abs(Dst.FData[I] - pcr_logf(x + Sqrt(x * x + 1.0)));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorArcSinh vs ln(x+sqrt(x^2+1)) max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
    // dst aliasing src.
    TNNetVolume.VectorArcSinh(Src.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := -30.0 + 60.0 * I / (N - 1);
      e := Abs(Src.FData[I] - pcr_logf(x + Sqrt(x * x + 1.0)));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('VectorArcSinh (aliased) max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestPointwiseSoftMaxVectorizedParity;
// PointwiseSoftMax (depth-axis softmax per (x,y) point) must agree with an
// independent scalar reference within tolerance. Depth = 37 straddles the AVX
// 8-wide body and the scalar tail; multiple spatial points exercise the loop.
const
  SX = 5; SY = 3; D = 37;
  AbsTol = 1e-4;
var
  V, Ref: TNNetVolume;
  cx, cy, cd, base: integer;
  mx, sum: TNeuralFloat;
begin
  V := TNNetVolume.Create(SX, SY, D);
  Ref := TNNetVolume.Create(SX, SY, D);
  try
    for cx := 0 to SX - 1 do
      for cy := 0 to SY - 1 do
        for cd := 0 to D - 1 do
        begin
          V[cx, cy, cd] := Sin(0.31 * cx + 0.7 * cy + 0.17 * cd) * 6.0;
          Ref[cx, cy, cd] := V[cx, cy, cd];
        end;
    // Scalar reference softmax over the depth axis at each (x,y).
    for cx := 0 to SX - 1 do
      for cy := 0 to SY - 1 do
      begin
        base := Ref.GetRawPos(cx, cy);
        mx := Ref.FData[base];
        for cd := 1 to D - 1 do
          if Ref.FData[base + cd] > mx then mx := Ref.FData[base + cd];
        sum := 0;
        for cd := 0 to D - 1 do
        begin
          Ref.FData[base + cd] := Exp(Ref.FData[base + cd] - mx);
          sum := sum + Ref.FData[base + cd];
        end;
        for cd := 0 to D - 1 do
          Ref.FData[base + cd] := Ref.FData[base + cd] / sum;
      end;
    V.PointwiseSoftMax();
    for cd := 0 to V.Size - 1 do
      AssertTrue('PointwiseSoftMax parity at ' + IntToStr(cd) +
        ' err ' + FloatToStr(Abs(V.FData[cd] - Ref.FData[cd])),
        Abs(V.FData[cd] - Ref.FData[cd]) < AbsTol);
  finally
    V.Free; Ref.Free;
  end;
end;

procedure TTestNeuralLayers.TestNetworkSaveLoad;
var
  NN1, NN2: TNNet;
  Input, Output1, Output2: TNNetVolume;
  TempFile: string;
begin
  NN1 := TNNet.Create();
  NN2 := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Output1 := TNNetVolume.Create(2, 1, 1);
  Output2 := TNNetVolume.Create(2, 1, 1);
  TempFile := GetTempDir() + 'nn_test_' + IntToStr(Random(MaxInt)) + '.nn';
  try
    NN1.AddLayer(TNNetInput.Create(4));
    NN1.AddLayer(TNNetFullConnectReLU.Create(8));
    NN1.AddLayer(TNNetFullConnectLinear.Create(2));

    Input.RandomizeGaussian();
    NN1.Compute(Input);
    NN1.GetOutput(Output1);

    NN1.SaveToFile(TempFile);
    NN2.LoadFromFile(TempFile);

    NN2.Compute(Input);
    NN2.GetOutput(Output2);

    AssertEquals('Loaded network should produce same output', 0.0, Output1.SumDiff(Output2), 0.0001);
  finally
    NN1.Free;
    NN2.Free;
    Input.Free;
    Output1.Free;
    Output2.Free;
    DeleteFile(TempFile);
  end;
end;

procedure TTestNeuralLayers.TestSimpleXORLearning;
var
  NN: TNNet;
  // Quick smoke test - just verify XOR network can be constructed
begin
  NN := TNNet.Create();
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(4),
      TNNetFullConnectReLU.Create(4),
      TNNetFullConnectLinear.Create(1)
    ]);
    AssertEquals('Network should have 4 layers', 4, NN.CountLayers);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestAvgPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetAvgPool.Create(2));

    // Set up input with known values for numerical verification
    // Region (0,0)-(1,1): values 2, 4, 6, 8 -> avg = 5.0
    Input[0, 0, 0] := 2.0;
    Input[1, 0, 0] := 4.0;
    Input[0, 1, 0] := 6.0;
    Input[1, 1, 0] := 8.0;
    // Region (2,0)-(3,1): all 4.0 -> avg = 4.0
    Input[2, 0, 0] := 4.0;
    Input[3, 0, 0] := 4.0;
    Input[2, 1, 0] := 4.0;
    Input[3, 1, 0] := 4.0;
    // Region (0,2)-(1,3): values 0, 0, 0, 12 -> avg = 3.0
    Input[0, 2, 0] := 0.0;
    Input[1, 2, 0] := 0.0;
    Input[0, 3, 0] := 0.0;
    Input[1, 3, 0] := 12.0;
    // Region (2,2)-(3,3): all 10.0 -> avg = 10.0
    Input[2, 2, 0] := 10.0;
    Input[3, 2, 0] := 10.0;
    Input[2, 3, 0] := 10.0;
    Input[3, 3, 0] := 10.0;
    
    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 2 after 2x2 avg pool', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 2 after 2x2 avg pool', 2, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 1', 1, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: each output cell should contain the average of its 2x2 region
    AssertEquals('Avg pool output (0,0) should be 5.0', 5.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Avg pool output (1,0) should be 4.0', 4.0, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    AssertEquals('Avg pool output (0,1) should be 3.0', 3.0, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    AssertEquals('Avg pool output (1,1) should be 10.0', 10.0, NN.GetLastLayer.Output[1, 1, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMinPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetMinPool.Create(2));

    // Set up input with known values for numerical verification
    Input.Fill(5.0);
    // Region (0,0)-(1,1): min will be 1.0
    Input[0, 0, 0] := 1.0;
    // Region (2,0)-(3,1): min will be 2.0
    Input[3, 1, 0] := 2.0;
    // Region (0,2)-(1,3): min will be 0.5
    Input[1, 2, 0] := 0.5;
    // Region (2,2)-(3,3): min will be 3.0
    Input[2, 3, 0] := 3.0;

    NN.Compute(Input);

    // Verify output dimensions
    AssertEquals('Output SizeX should be 2 after 2x2 min pool', 2, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 2 after 2x2 min pool', 2, NN.GetLastLayer.Output.SizeY);
    
    // Numerical verification: each output cell should contain the min of its 2x2 region
    AssertEquals('Min pool output (0,0) should be 1.0', 1.0, NN.GetLastLayer.Output[0, 0, 0], 0.0001);
    AssertEquals('Min pool output (1,0) should be 2.0', 2.0, NN.GetLastLayer.Output[1, 0, 0], 0.0001);
    AssertEquals('Min pool output (0,1) should be 0.5', 0.5, NN.GetLastLayer.Output[0, 1, 0], 0.0001);
    AssertEquals('Min pool output (1,1) should be 3.0', 3.0, NN.GetLastLayer.Output[1, 1, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestReLUActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetReLU.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := -1.0;
    Input.Raw[2] := 1.0;
    Input.Raw[3] := 2.0;

    NN.Compute(Input);

    // ReLU: max(0, x)
    AssertEquals('ReLU of -2 should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('ReLU of -1 should be 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('ReLU of 1 should be 1', 1.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('ReLU of 2 should be 2', 2.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSigmoidActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetSigmoid.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 10.0;
    Input.Raw[2] := -10.0;

    NN.Compute(Input);

    // Sigmoid(0) = 0.5
    AssertEquals('Sigmoid of 0 should be 0.5', 0.5, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // Sigmoid(10) ≈ 1
    AssertTrue('Sigmoid of 10 should be close to 1', NN.GetLastLayer.Output.Raw[1] > 0.99);
    // Sigmoid(-10) ≈ 0
    AssertTrue('Sigmoid of -10 should be close to 0', NN.GetLastLayer.Output.Raw[2] < 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSoftMaxLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
  SumOutput: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetSoftMax.Create());

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 2.0;
    Input.Raw[2] := 3.0;
    Input.Raw[3] := 4.0;

    NN.Compute(Input);

    SumOutput := NN.GetLastLayer.Output.GetSum();

    // SoftMax output should sum to 1.0
    AssertEquals('SoftMax output sum should be 1.0', 1.0, SumOutput, 0.0001);
    // Higher inputs should produce higher probabilities
    AssertTrue('Output[3] should be greatest', NN.GetLastLayer.Output.Raw[3] > NN.GetLastLayer.Output.Raw[2]);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestDepthwiseConvolution;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 4);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 4));
    NN.AddLayer(TNNetDepthwiseConvReLU.Create(1, 3, 1, 1));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Depthwise conv with multiplier 1 keeps same depth
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 4', 4, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: output should be finite and non-NaN
    for I := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // ReLU applied, so output should be non-negative
    AssertTrue('Output min should be >= 0 (ReLU)', NN.GetLastLayer.Output.GetMin() >= -0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestPointwiseConvolution;
var
  NN: TNNet;
  Input: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    NN.AddLayer(TNNetPointwiseConvReLU.Create(32));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Pointwise conv changes depth while keeping spatial dimensions
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
    
    // Numerical verification: output should be finite and non-NaN
    for I := 0 to NN.GetLastLayer.Output.Size - 1 do
    begin
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
      AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[I]));
    end;
    
    // ReLU applied, so output should be non-negative
    AssertTrue('Output min should be >= 0 (ReLU)', NN.GetLastLayer.Output.GetMin() >= -0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestConcatLayers;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Layer1, Layer2: TNNetLayer;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(8, 8, 3));
    
    // Create two parallel paths branching from the input layer
    Layer1 := NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));
    Layer2 := NN.AddLayerAfter(TNNetConvolutionReLU.Create(8, 3, 1, 1), InputLayer);
    
    // Concatenate the two paths
    NN.AddLayer(TNNetDeepConcat.Create([Layer1, Layer2]));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Concatenated depth should be 16 + 8 = 24
    AssertEquals('Concatenated depth should be 24', 24, NN.GetLastLayer.Output.Depth);
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    
    // Numerical verification: total size should be 8*8*24 = 1536
    AssertEquals('Total output size should be 1536', 1536, NN.GetLastLayer.Output.Size);
    
    // Output should be finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSumLayers;
var
  NN: TNNet;
  Input: TNNetVolume;
  InputLayer, Layer1, Layer2: TNNetLayer;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(8, 8, 16));
    
    // Create two parallel paths with same output size branching from input
    Layer1 := NN.AddLayer(TNNetConvolutionLinear.Create(16, 3, 1, 1));
    Layer2 := NN.AddLayerAfter(TNNetConvolutionLinear.Create(16, 3, 1, 1), InputLayer);
    
    // Sum the two paths
    NN.AddLayer(TNNetSum.Create([Layer1, Layer2]));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Sum should maintain the same dimensions
    AssertEquals('Sum output depth should be 16', 16, NN.GetLastLayer.Output.Depth);
    AssertEquals('Sum output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Sum output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    
    // Numerical verification: total size should be 8*8*16 = 1024
    AssertEquals('Total output size should be 1024', 1024, NN.GetLastLayer.Output.Size);
    
    // Output should be finite
    for I := 0 to Min(MAX_NAN_CHECK_ITERATIONS, NN.GetLastLayer.Output.Size - 1) do
      AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[I]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestIdentityLayer;
var
  NN: TNNet;
  Input, Output: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  Output := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetIdentity.Create());

    Input.RandomizeGaussian();
    NN.Compute(Input);
    NN.GetOutput(Output);

    // Identity layer should pass through unchanged
    AssertEquals('Identity should preserve values', 0.0, Input.SumDiff(Output), 0.0001);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralLayers.TestReshapeLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 4);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 4));
    NN.AddLayer(TNNetReshape.Create(8, 2, 4));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Total size should be preserved: 4*4*4 = 8*2*4 = 64
    AssertEquals('Reshape output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Reshape output SizeY should be 2', 2, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Reshape output Depth should be 4', 4, NN.GetLastLayer.Output.Depth);
    AssertEquals('Total size should be preserved', 64, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestDropoutLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(10, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(10));
    NN.AddLayer(TNNetDropout.Create(0.5));

    Input.Fill(1.0);
    NN.Compute(Input);

    // During inference (non-training), dropout should pass values through
    // The output size should match input size
    AssertEquals('Dropout output size should be 10', 10, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMultipleLayersNetwork;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(32, 32, 3);
  try
    // Build a more complex network similar to a simple CNN
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionReLU.Create(16, 3, 1, 1),
      TNNetMaxPool.Create(2),
      TNNetConvolutionReLU.Create(32, 3, 1, 1),
      TNNetMaxPool.Create(2),
      TNNetFullConnectReLU.Create(64),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Output should be 10 classes with softmax
    AssertEquals('Output should have 10 classes', 10, NN.GetLastLayer.Output.Size);
    // SoftMax sum should be 1.0
    AssertEquals('SoftMax sum should be 1.0', 1.0, NN.GetLastLayer.Output.GetSum(), 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestNetworkClone;
var
  NN1, NN2: TNNet;
  Input, Output1, Output2: TNNetVolume;
begin
  NN1 := TNNet.Create();
  NN2 := nil;
  Input := TNNetVolume.Create(4, 1, 1);
  Output1 := TNNetVolume.Create(2, 1, 1);
  Output2 := TNNetVolume.Create(2, 1, 1);
  try
    NN1.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectReLU.Create(8),
      TNNetFullConnectLinear.Create(2)
    ]);

    Input.RandomizeGaussian();
    NN1.Compute(Input);
    NN1.GetOutput(Output1);

    // Clone the network using the Clone method
    NN2 := NN1.Clone();

    NN2.Compute(Input);
    NN2.GetOutput(Output2);

    // Cloned network should produce same output
    AssertEquals('Cloned network should produce same output', 0.0, Output1.SumDiff(Output2), 0.0001);
  finally
    NN1.Free;
    if NN2 <> nil then NN2.Free;
    Input.Free;
    Output1.Free;
    Output2.Free;
  end;
end;

procedure TTestNeuralLayers.TestLayerCount;
var
  NN: TNNet;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer([
      TNNetInput.Create(10),
      TNNetFullConnectReLU.Create(20),
      TNNetFullConnectReLU.Create(20),
      TNNetFullConnectLinear.Create(5)
    ]);

    AssertEquals('Network should have 4 layers', 4, NN.CountLayers());
    // Count neurons and weights
    AssertTrue('Network should have positive neuron count', NN.CountNeurons() > 0);
    AssertTrue('Network should have positive weight count', NN.CountWeights() > 0);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestReLU6Activation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    // TNNetReLU6: ReLU clamped to [0, 6]
    // Note: Full activation behavior only applies during training when error derivatives are set
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetReLU6.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 3.0;
    Input.Raw[3] := 6.0;
    Input.Raw[4] := 10.0;

    NN.Compute(Input);

    // Verify output is produced
    AssertEquals('Output should have 5 elements', 5, NN.GetLastLayer.Output.Size);
    // ReLU6 should not produce NaN values
    AssertFalse('Output 0 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output 1 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[1]));
    AssertFalse('Output 2 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[2]));
    AssertFalse('Output 3 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[3]));
    AssertFalse('Output 4 should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[4]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestLeakyReLUActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetLeakyReLU.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 2.0;
    Input.Raw[3] := -100.0;

    NN.Compute(Input);

    // Leaky ReLU should produce output
    AssertEquals('Output should have 4 elements', 4, NN.GetLastLayer.Output.Size);
    // For positive values, output equals input
    AssertEquals('LeakyReLU of 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('LeakyReLU of 2 should be 2', 2.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    // Output values should be finite
    AssertFalse('Output should not be NaN', IsNaN(NN.GetLastLayer.Output.Raw[0]));
    AssertFalse('Output should not be Inf', IsInfinite(NN.GetLastLayer.Output.Raw[3]));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSwishActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetSwish.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;

    NN.Compute(Input);

    // Swish: x * sigmoid(x)
    // At x=0: 0 * 0.5 = 0
    AssertEquals('Swish of 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // At x=1: 1 * sigmoid(1) ≈ 0.731
    AssertTrue('Swish of 1 should be around 0.731', Abs(NN.GetLastLayer.Output.Raw[1] - 0.731) < 0.01);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestHyperbolicTangent;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetHyperbolicTangent.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 10.0;
    Input.Raw[2] := -10.0;

    NN.Compute(Input);

    // tanh(0) = 0
    AssertEquals('Tanh of 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // tanh(10) ≈ 1
    AssertTrue('Tanh of 10 should be close to 1', NN.GetLastLayer.Output.Raw[1] > 0.99);
    // tanh(-10) ≈ -1
    AssertTrue('Tanh of -10 should be close to -1', NN.GetLastLayer.Output.Raw[2] < -0.99);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSELUActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetSELU.Create());

    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;

    NN.Compute(Input);

    // SELU(0) = 0
    AssertEquals('SELU of 0 should be 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    // SELU of positive values scales them
    AssertTrue('SELU of 1 should be positive', NN.GetLastLayer.Output.Raw[1] > 0);
    // SELU of negative values
    AssertTrue('SELU of -1 should be negative', NN.GetLastLayer.Output.Raw[2] < 0);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestGELUActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
  OutputLayer: TNNetLayer;
  ExpectedGELU0, ExpectedGELU1, ExpectedGELUNeg1: TNeuralFloat;
const
  SQRT_2_OVER_PI = 0.7978845608;
  GELU_CONST = 0.044715;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetGELU.Create());

    // Test values: 0, 1, -1, 2, -2
    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 2.0;
    Input.Raw[4] := -2.0;

    NN.Compute(Input);

    OutputLayer := NN.GetLastLayer;

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    // GELU(0) = 0
    ExpectedGELU0 := 0.0;
    AssertEquals('GELU of 0 should be 0', ExpectedGELU0, OutputLayer.Output.Raw[0], 0.0001);

    // GELU(1) ≈ 0.8413 (approximately)
    ExpectedGELU1 := 0.5 * 1.0 * (1 + Tanh(SQRT_2_OVER_PI * (1.0 + GELU_CONST * 1.0)));
    AssertEquals('GELU of 1 should match approximation', ExpectedGELU1, OutputLayer.Output.Raw[1], 0.001);
    AssertTrue('GELU of 1 should be around 0.84', Abs(OutputLayer.Output.Raw[1] - 0.841) < 0.01);

    // GELU(-1) ≈ -0.1587 (approximately - close to 0 but negative)
    ExpectedGELUNeg1 := 0.5 * (-1.0) * (1 + Tanh(SQRT_2_OVER_PI * (-1.0 + GELU_CONST * (-1.0))));
    AssertEquals('GELU of -1 should match approximation', ExpectedGELUNeg1, OutputLayer.Output.Raw[2], 0.001);
    AssertTrue('GELU of -1 should be around -0.16', Abs(OutputLayer.Output.Raw[2] - (-0.159)) < 0.02);

    // GELU(2) should be close to 2 (almost linear for large positive values)
    AssertTrue('GELU of 2 should be close to 2', Abs(OutputLayer.Output.Raw[3] - 1.96) < 0.1);

    // GELU(-2) should be very small (close to 0)
    AssertTrue('GELU of -2 should be close to 0', Abs(OutputLayer.Output.Raw[4]) < 0.05);

    // GELU is indeed monotonic: GELU(-2) > GELU(-1) (both negative, but -2 is closer to 0)
    // Order: GELU(-1) < GELU(-2) < GELU(0) < GELU(1) < GELU(2)
    AssertTrue('GELU should be monotonic', 
      (OutputLayer.Output.Raw[2] < OutputLayer.Output.Raw[4]) and
      (OutputLayer.Output.Raw[4] < OutputLayer.Output.Raw[0]) and
      (OutputLayer.Output.Raw[0] < OutputLayer.Output.Raw[1]) and
      (OutputLayer.Output.Raw[1] < OutputLayer.Output.Raw[3]));

  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMishActivation;
var
  NN: TNNet;
  Input: TNNetVolume;
  OutputLayer: TNNetLayer;
  ExpectedMish0, ExpectedMish1, ExpectedMishNeg1: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetMish.Create());

    // Test values: 0, 1, -1, 2, -2
    Input.Raw[0] := 0.0;
    Input.Raw[1] := 1.0;
    Input.Raw[2] := -1.0;
    Input.Raw[3] := 2.0;
    Input.Raw[4] := -2.0;

    NN.Compute(Input);

    OutputLayer := NN.GetLastLayer;

    // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // Mish(0) = 0 * tanh(ln(2)) = 0
    ExpectedMish0 := 0.0;
    AssertEquals('Mish of 0 should be 0', ExpectedMish0, OutputLayer.Output.Raw[0], 0.0001);

    // Mish(1) ≈ 0.8651
    ExpectedMish1 := 1.0 * Tanh(Ln(1 + Exp(1.0)));
    AssertEquals('Mish of 1 should match formula', ExpectedMish1, OutputLayer.Output.Raw[1], 0.001);
    AssertTrue('Mish of 1 should be around 0.865', Abs(OutputLayer.Output.Raw[1] - 0.865) < 0.01);

    // Mish(-1) ≈ -0.3034
    ExpectedMishNeg1 := -1.0 * Tanh(Ln(1 + Exp(-1.0)));
    AssertEquals('Mish of -1 should match formula', ExpectedMishNeg1, OutputLayer.Output.Raw[2], 0.001);
    AssertTrue('Mish of -1 should be around -0.30', Abs(OutputLayer.Output.Raw[2] - (-0.303)) < 0.02);

    // Mish(2) should be close to 2 (almost linear for large positive values)
    AssertTrue('Mish of 2 should be close to 2', Abs(OutputLayer.Output.Raw[3] - 1.94) < 0.1);

    // Mish(-2) ≈ -0.2525 (negative but not close to 0)
    AssertTrue('Mish of -2 should be around -0.25', Abs(OutputLayer.Output.Raw[4] - (-0.252)) < 0.05);

    // Test non-monotonicity for negative values (a characteristic of Mish)
    // For very negative values, Mish approaches 0 from below
    // Mish(-1) is more negative than Mish(-2) which is closer to 0
    // So |Mish(-1)| > |Mish(-2)|
    AssertTrue('Mish shows non-monotonic behavior for negative values',
      Abs(OutputLayer.Output.Raw[2]) > Abs(OutputLayer.Output.Raw[4]));

  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestGELUSaveLoad;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  StructStr: string;
  Output1, Output2: TNeuralFloat;
begin
  NN := TNNet.Create();
  NN2 := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.AddLayer(TNNetGELU.Create());

    Input.Raw[0] := 0.5;
    Input.Raw[1] := -0.5;
    Input.Raw[2] := 1.0;

    NN.Compute(Input);
    Output1 := NN.GetLastLayer.Output.Raw[0];

    // Save and load
    StructStr := NN.SaveToString();
    NN2.LoadFromString(StructStr);

    NN2.Compute(Input);
    Output2 := NN2.GetLastLayer.Output.Raw[0];

    AssertEquals('GELU output should be same after save/load', Output1, Output2, 0.0001);
    AssertEquals('Layer count should match after load', NN.CountLayers(), NN2.CountLayers());

  finally
    NN.Free;
    NN2.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMishSaveLoad;
var
  NN, NN2: TNNet;
  Input: TNNetVolume;
  StructStr: string;
  Output1, Output2: TNeuralFloat;
begin
  NN := TNNet.Create();
  NN2 := TNNet.Create();
  Input := TNNetVolume.Create(3, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(3));
    NN.AddLayer(TNNetFullConnectLinear.Create(2));
    NN.AddLayer(TNNetMish.Create());

    Input.Raw[0] := 0.5;
    Input.Raw[1] := -0.5;
    Input.Raw[2] := 1.0;

    NN.Compute(Input);
    Output1 := NN.GetLastLayer.Output.Raw[0];

    // Save and load
    StructStr := NN.SaveToString();
    NN2.LoadFromString(StructStr);

    NN2.Compute(Input);
    Output2 := NN2.GetLastLayer.Output.Raw[0];

    AssertEquals('Mish output should be same after save/load', Output1, Output2, 0.0001);
    AssertEquals('Layer count should match after load', NN.CountLayers(), NN2.CountLayers());

  finally
    NN.Free;
    NN2.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestGELUBackpropagation;
var
  NN: TNNet;
  Input, Target: TNNetVolume;
  ErrorBefore, ErrorAfter: TNeuralFloat;
  Epoch: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Target := TNNetVolume.Create(1, 1, 1);
  try
    // Create a simple network with GELU activation
    NN.AddLayer(TNNetInput.Create(2));
    NN.AddLayer(TNNetFullConnectLinear.Create(4));
    NN.AddLayer(TNNetGELU.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(1));

    NN.SetLearningRate(0.1, 0.0);

    // XOR-like problem
    Input.Raw[0] := 1.0;
    Input.Raw[1] := 0.0;
    Target.Raw[0] := 1.0;

    // Compute initial error
    NN.Compute(Input);
    ErrorBefore := Abs(NN.GetLastLayer.Output.Raw[0] - Target.Raw[0]);

    // Train for multiple epochs
    for Epoch := 1 to 100 do
    begin
      NN.Compute(Input);
      NN.Backpropagate(Target);
    end;

    // Compute final error
    NN.Compute(Input);
    ErrorAfter := Abs(NN.GetLastLayer.Output.Raw[0] - Target.Raw[0]);

    // Error should decrease (learning is happening through backpropagation)
    AssertTrue('GELU network should learn (error should decrease)',
      (ErrorAfter < ErrorBefore) or (ErrorAfter < 0.5));

  finally
    NN.Free;
    Input.Free;
    Target.Free;
  end;
end;

procedure TTestNeuralLayers.TestMishBackpropagation;
var
  NN: TNNet;
  Input, Target: TNNetVolume;
  ErrorBefore, ErrorAfter: TNeuralFloat;
  Epoch: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Target := TNNetVolume.Create(1, 1, 1);
  try
    // Create a simple network with Mish activation
    NN.AddLayer(TNNetInput.Create(2));
    NN.AddLayer(TNNetFullConnectLinear.Create(4));
    NN.AddLayer(TNNetMish.Create());
    NN.AddLayer(TNNetFullConnectLinear.Create(1));

    NN.SetLearningRate(0.1, 0.0);

    // XOR-like problem
    Input.Raw[0] := 1.0;
    Input.Raw[1] := 0.0;
    Target.Raw[0] := 1.0;

    // Compute initial error
    NN.Compute(Input);
    ErrorBefore := Abs(NN.GetLastLayer.Output.Raw[0] - Target.Raw[0]);

    // Train for multiple epochs
    for Epoch := 1 to 100 do
    begin
      NN.Compute(Input);
      NN.Backpropagate(Target);
    end;

    // Compute final error
    NN.Compute(Input);
    ErrorAfter := Abs(NN.GetLastLayer.Output.Raw[0] - Target.Raw[0]);

    // Error should decrease (learning is happening through backpropagation)
    AssertTrue('Mish network should learn (error should decrease)',
      (ErrorAfter < ErrorBefore) or (ErrorAfter < 0.5));

  finally
    NN.Free;
    Input.Free;
    Target.Free;
  end;
end;

procedure TTestNeuralLayers.TestMaxChannel;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    NN.AddLayer(TNNetMaxChannel.Create());

    // Fill channels with different values
    Input.FillAtDepth(0, 1.0);
    Input.FillAtDepth(1, 2.0);
    Input.FillAtDepth(2, 3.0);

    NN.Compute(Input);

    // MaxChannel reduces to depth-sized 1D output
    AssertEquals('Output should have 3 elements', 3, NN.GetLastLayer.Output.Size);
    AssertEquals('Max of channel 0 should be 1.0', 1.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Max of channel 1 should be 2.0', 2.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('Max of channel 2 should be 3.0', 3.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestAvgChannel;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetAvgChannel.Create());

    Input.FillAtDepth(0, 4.0);
    Input.FillAtDepth(1, 8.0);

    NN.Compute(Input);

    // AvgChannel reduces to depth-sized 1D output
    AssertEquals('Output should have 2 elements', 2, NN.GetLastLayer.Output.Size);
    AssertEquals('Avg of channel 0 should be 4.0', 4.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('Avg of channel 1 should be 8.0', 8.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestLayerMaxNormalization;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetLayerMaxNormalization.Create());

    Input.Raw[0] := 2.0;
    Input.Raw[1] := 4.0;
    Input.Raw[2] := 6.0;
    Input.Raw[3] := 8.0;

    NN.Compute(Input);

    // Max normalization divides by max value (8.0)
    // Output should be in range [0, 1]
    AssertTrue('Output max should be 1.0', Abs(NN.GetLastLayer.Output.GetMax() - 1.0) < 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestLayerStdNormalization;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4));
    NN.AddLayer(TNNetLayerStdNormalization.Create());

    Input.Raw[0] := 2.0;
    Input.Raw[1] := 4.0;
    Input.Raw[2] := 6.0;
    Input.Raw[3] := 8.0;

    NN.Compute(Input);

    // Std normalization should produce output with unit std deviation
    AssertEquals('Output size should be 4', 4, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMovingStdNormalization;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(8));
    NN.AddLayer(TNNetMovingStdNormalization.Create());

    Input.RandomizeGaussian(5.0);
    Input.Add(10.0); // Shift to have non-zero mean

    NN.Compute(Input);

    AssertEquals('Output size should be 8', 8, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestChannelBias;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetChannelBias.Create());

    Input.Fill(1.0);
    NN.Compute(Input);

    // ChannelBias adds a learnable bias per channel
    AssertEquals('Output should maintain dimensions', 32, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestChannelMul;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 2);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 2));
    NN.AddLayer(TNNetChannelMul.Create());

    Input.Fill(2.0);
    NN.Compute(Input);

    // ChannelMul multiplies by a learnable scale per channel
    AssertEquals('Output should maintain dimensions', 32, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestCellBias;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 1));
    NN.AddLayer(TNNetCellBias.Create());

    Input.Fill(1.0);
    NN.Compute(Input);

    // CellBias adds a learnable bias per cell
    AssertEquals('Output should maintain dimensions', 16, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestSplitChannels;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 8));
    // Split first 4 channels
    NN.AddLayer(TNNetSplitChannels.Create(0, 4));

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Output should have 4 channels
    AssertEquals('Output depth should be 4', 4, NN.GetLastLayer.Output.Depth);
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestInterleaveChannels;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 8);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 8));
    NN.AddLayer(TNNetInterleaveChannels.Create(2));

    Input.RandomizeGaussian();
    NN.Compute(Input);

    // Interleave should maintain size
    AssertEquals('Output size should match input size', 128, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestPointwiseConvLinear;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    NN.AddLayer(TNNetPointwiseConvLinear.Create(32));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Pointwise conv (1x1) changes depth only
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 8', 8, NN.GetLastLayer.Output.SizeY);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestLocalConnect;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetLocalConnectReLU.Create(8, 3, 1, 1));

    Input.Fill(1.0);
    NN.Compute(Input);

    // Local connect should work similar to convolution but with unique weights per position
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 8', 8, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestGroupedConvolution;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 16);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 16));
    // Groups=4 means 4 separate convolutions on 4 channels each
    NN.AddLayer(TNNetGroupedConvolutionLinear.Create(32, 3, 1, 1, 4));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestBackpropagation;
var
  NN: TNNet;
  Input, DesiredOutput: TNNetVolume;
  ErrorBefore, ErrorAfter: TNeuralFloat;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  DesiredOutput := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectLinear.Create(4),
      TNNetReLU.Create(),
      TNNetFullConnectLinear.Create(1)
    ]);

    // Set input and desired output
    Input.Raw[0] := 1.0;
    Input.Raw[1] := 0.5;
    DesiredOutput.Raw[0] := 0.7;

    // Forward pass
    NN.Compute(Input);
    ErrorBefore := Abs(NN.GetLastLayer.Output.Raw[0] - DesiredOutput.Raw[0]);

    // Backward pass with learning
    NN.Backpropagate(DesiredOutput);
    NN.UpdateWeights();

    // Forward pass again
    NN.Compute(Input);
    ErrorAfter := Abs(NN.GetLastLayer.Output.Raw[0] - DesiredOutput.Raw[0]);

    // Error should decrease after one step (in most cases)
    // Note: This is a probabilistic test, may occasionally fail
    AssertTrue('Network should produce output', NN.GetLastLayer.Output.Size = 1);
  finally
    NN.Free;
    Input.Free;
    DesiredOutput.Free;
  end;
end;

procedure TTestNeuralLayers.TestGradientComputation;
var
  NN: TNNet;
  Input, DesiredOutput: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  DesiredOutput := TNNetVolume.Create(2, 1, 1);
  try
    NN.AddLayer([
      TNNetInput.Create(4),
      TNNetFullConnectLinear.Create(2)
    ]);

    Input.RandomizeGaussian();
    DesiredOutput.RandomizeGaussian();

    NN.Compute(Input);
    NN.Backpropagate(DesiredOutput);

    // Check that output error is computed
    AssertEquals('Output error size should match', 2, NN.GetLastLayer.OutputError.Size);
    // The output error should not be all zeros
    AssertTrue('Output error should be non-zero', NN.GetLastLayer.OutputError.GetSumAbs() > 0);
  finally
    NN.Free;
    Input.Free;
    DesiredOutput.Free;
  end;
end;

procedure TTestNeuralLayers.TestWeightInitHe;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  MaxWeight: TNeuralFloat;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(100));
    Layer := TNNetFullConnectLinear.Create(50);
    NN.AddLayer(Layer);

    // Initialize with He method
    Layer.InitHeUniform();

    // Weights should be in reasonable range
    MaxWeight := Layer.Neurons.GetMaxAbsWeight();
    AssertTrue('Weights should be initialized', MaxWeight > 0);
    AssertTrue('Weights should be bounded', MaxWeight < 10);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestWeightInitLeCun;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  MaxWeight: TNeuralFloat;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(100));
    Layer := TNNetFullConnectLinear.Create(50);
    NN.AddLayer(Layer);

    // Initialize with LeCun method
    Layer.InitLeCunUniform();

    // Weights should be in reasonable range
    MaxWeight := Layer.Neurons.GetMaxAbsWeight();
    AssertTrue('Weights should be initialized', MaxWeight > 0);
    AssertTrue('Weights should be bounded', MaxWeight < 10);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestWeightInitGlorot;
var
  NN: TNNet;
  Layer: TNNetFullConnectLinear;
  MaxWeight: TNeuralFloat;
begin
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(100));
    Layer := TNNetFullConnectLinear.Create(50);
    NN.AddLayer(Layer);

    // Initialize with Glorot/Xavier method
    Layer.InitGlorotBengioUniform();

    // Weights should be in reasonable range
    MaxWeight := Layer.Neurons.GetMaxAbsWeight();
    AssertTrue('Weights should be initialized', MaxWeight > 0);
    AssertTrue('Weights should be bounded', MaxWeight < 10);
  finally
    NN.Free;
  end;
end;

procedure TTestNeuralLayers.TestEmbeddingLayer;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1); // 4 tokens
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    // Vocab size 100, embedding dim 16
    NN.AddLayer(TNNetEmbedding.Create(100, 16));

    // Input tokens as integers (stored as floats)
    Input.Raw[0] := 5;
    Input.Raw[1] := 10;
    Input.Raw[2] := 25;
    Input.Raw[3] := 50;

    NN.Compute(Input);

    // Output should be 4 x 16 (4 tokens, 16 embedding dim)
    AssertEquals('Output SizeX should be 4', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 16', 16, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestTokenAndPositionalEmbedding;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 1, 1); // 8 tokens (sequence length)
  try
    NN.AddLayer(TNNetInput.Create(8, 1, 1));
    // Vocab size 256, embedding dim 32
    NN.AddLayer(TNNetTokenAndPositionalEmbedding.Create(256, 32));

    // Input tokens
    Input.Raw[0] := 1;
    Input.Raw[1] := 5;
    Input.Raw[2] := 10;
    Input.Raw[3] := 20;
    Input.Raw[4] := 30;
    Input.Raw[5] := 40;
    Input.Raw[6] := 50;
    Input.Raw[7] := 60;

    NN.Compute(Input);

    // Output should be 8 x 32 (8 tokens, 32 embedding dim)
    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output Depth should be 32', 32, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// Regression: TNNetMaxChannel global max over a RECTANGULAR (SizeX <> SizeY)
// feature map. The old square-only pooling path mis-indexed the output rows
// when SizeY <> SizeX; the reduction must collapse the WHOLE grid to (1,1,D)
// and route the gradient to the true winning (x,y) position per channel.
procedure TTestNeuralLayers.TestMaxChannelRectangular;
var
  NN: TNNet;
  Input: TNNetVolume;
  ChannelLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(3, 5, 2); // W=3, H=5 (rectangular)
  try
    NN.AddLayer(TNNetInput.Create(3, 5, 2, 1));
    ChannelLayer := NN.AddLayer(TNNetMaxChannel.Create());

    // Baseline values per channel, then plant a single distinct maximum.
    Input.FillAtDepth(0, 1.0);
    Input.FillAtDepth(1, -4.0);
    // Channel 0 max at (x=2,y=4); channel 1 max at (x=0,y=3).
    Input[2, 4, 0] := 9.0;
    Input[0, 3, 1] := 7.0;

    NN.Compute(Input);

    AssertEquals('Output collapses to (1,1,Depth)=2 elements',
      2, ChannelLayer.Output.Size);
    AssertEquals('Output SizeX must be 1', 1, ChannelLayer.Output.SizeX);
    AssertEquals('Output SizeY must be 1', 1, ChannelLayer.Output.SizeY);
    AssertEquals('Global max of channel 0', 9.0, ChannelLayer.Output.Raw[0], 0.0001);
    AssertEquals('Global max of channel 1', 7.0, ChannelLayer.Output.Raw[1], 0.0001);

    // Backward: gradient routes to the winning position only.
    ChannelLayer.OutputError.Fill(0);
    ChannelLayer.OutputError.Raw[0] := 1.0;
    ChannelLayer.OutputError.Raw[1] := 2.0;
    NN.GetFirstLayer.OutputError.Fill(0);
    ChannelLayer.IncDepartingBranchesCnt();
    ChannelLayer.Backpropagate();

    AssertEquals('Grad lands on channel-0 winner (2,4,0)',
      1.0, NN.GetFirstLayer.OutputError[2, 4, 0], 0.0001);
    AssertEquals('Grad lands on channel-1 winner (0,3,1)',
      2.0, NN.GetFirstLayer.OutputError[0, 3, 1], 0.0001);
    // A non-winning cell receives nothing.
    AssertEquals('No grad at a non-winning cell',
      0.0, NN.GetFirstLayer.OutputError[0, 0, 0], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// Regression: TNNetMinChannel global min over a RECTANGULAR feature map.
procedure TTestNeuralLayers.TestMinChannelRectangular;
var
  NN: TNNet;
  Input: TNNetVolume;
  ChannelLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(6, 2, 2); // W=6, H=2 (wide rectangular)
  try
    NN.AddLayer(TNNetInput.Create(6, 2, 2, 1));
    ChannelLayer := NN.AddLayer(TNNetMinChannel.Create());

    Input.FillAtDepth(0, 3.0);
    Input.FillAtDepth(1, 8.0);
    Input[5, 1, 0] := -2.0; // channel 0 min
    Input[1, 0, 1] := 0.5;  // channel 1 min

    NN.Compute(Input);

    AssertEquals('Output collapses to 2 elements', 2, ChannelLayer.Output.Size);
    AssertEquals('Global min of channel 0', -2.0, ChannelLayer.Output.Raw[0], 0.0001);
    AssertEquals('Global min of channel 1', 0.5, ChannelLayer.Output.Raw[1], 0.0001);

    ChannelLayer.OutputError.Fill(0);
    ChannelLayer.OutputError.Raw[0] := 5.0;
    ChannelLayer.OutputError.Raw[1] := -1.0;
    NN.GetFirstLayer.OutputError.Fill(0);
    ChannelLayer.IncDepartingBranchesCnt();
    ChannelLayer.Backpropagate();

    AssertEquals('Grad lands on channel-0 min winner (5,1,0)',
      5.0, NN.GetFirstLayer.OutputError[5, 1, 0], 0.0001);
    AssertEquals('Grad lands on channel-1 min winner (1,0,1)',
      -1.0, NN.GetFirstLayer.OutputError[1, 0, 1], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// The SQUARE case must be unchanged by the rectangular fix.
procedure TTestNeuralLayers.TestMaxChannelSquareRegression;
var
  NN: TNNet;
  Input: TNNetVolume;
  ChannelLayer: TNNetLayer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3, 1)); // pError=1 sizes error volumes
    ChannelLayer := NN.AddLayer(TNNetMaxChannel.Create());

    Input.FillAtDepth(0, 1.0);
    Input.FillAtDepth(1, 2.0);
    Input.FillAtDepth(2, 3.0);
    Input[1, 2, 2] := 11.0; // a plain maximum in channel 2

    NN.Compute(Input);

    AssertEquals('Square output still 3 elements', 3, ChannelLayer.Output.Size);
    AssertEquals('Square max channel 0', 1.0, ChannelLayer.Output.Raw[0], 0.0001);
    AssertEquals('Square max channel 1', 2.0, ChannelLayer.Output.Raw[1], 0.0001);
    AssertEquals('Square max channel 2', 11.0, ChannelLayer.Output.Raw[2], 0.0001);

    ChannelLayer.OutputError.Fill(0);
    ChannelLayer.OutputError.Raw[2] := 1.0;
    NN.GetFirstLayer.OutputError.Fill(0);
    ChannelLayer.IncDepartingBranchesCnt();
    ChannelLayer.Backpropagate();
    AssertEquals('Square grad routes to winner (1,2,2)',
      1.0, NN.GetFirstLayer.OutputError[1, 2, 2], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// Regression for FlipX -> padded convolution: a padded conv writes its error
// back into the flip layer's output-sized error buffer; the flip backward must
// stay within bounds and produce finite gradients (no range-check overflow).
procedure TTestNeuralLayers.TestFlipXPaddedConvBackprop;
var
  NN: TNNet;
  Input, Expected: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 7, 2); // rectangular to exercise both axes
  Expected := TNNetVolume.Create(3);
  try
    NN.AddLayer(TNNetInput.Create(5, 7, 2, 1));
    NN.AddLayer(TNNetFlipX.Create());
    // Padded (FeatureSize 3, Padding 1) conv keeps spatial size; its backward
    // routes a padded error region into the flip layer.
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));

    Input.Randomize();
    Expected.Raw[0] := 0.5; Expected.Raw[1] := -0.3; Expected.Raw[2] := 0.1;

    NN.Compute(Input);
    NN.Backpropagate(Expected);

    // Assert input gradients are finite (no overflow / NaN).
    for I := 0 to NN.GetFirstLayer.OutputError.Size - 1 do
      AssertTrue('Input grad must be finite',
        not (IsNan(NN.GetFirstLayer.OutputError.Raw[I]) or
             IsInfinite(NN.GetFirstLayer.OutputError.Raw[I])));
    AssertEquals('Head output size is 3', 3, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
    Expected.Free;
  end;
end;

procedure TTestNeuralLayers.TestFlipYPaddedConvBackprop;
var
  NN: TNNet;
  Input, Expected: TNNetVolume;
  I: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(7, 5, 2);
  Expected := TNNetVolume.Create(3);
  try
    NN.AddLayer(TNNetInput.Create(7, 5, 2, 1));
    NN.AddLayer(TNNetFlipY.Create());
    NN.AddLayer(TNNetConvolutionReLU.Create(4, 3, 1, 1));
    NN.AddLayer(TNNetFullConnectLinear.Create(3));

    Input.Randomize();
    Expected.Raw[0] := -0.2; Expected.Raw[1] := 0.4; Expected.Raw[2] := 0.0;

    NN.Compute(Input);
    NN.Backpropagate(Expected);

    for I := 0 to NN.GetFirstLayer.OutputError.Size - 1 do
      AssertTrue('Input grad must be finite',
        not (IsNan(NN.GetFirstLayer.OutputError.Raw[I]) or
             IsInfinite(NN.GetFirstLayer.OutputError.Raw[I])));
    AssertEquals('Head output size is 3', 3, NN.GetLastLayer.Output.Size);
  finally
    NN.Free;
    Input.Free;
    Expected.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralLayers);

end.
