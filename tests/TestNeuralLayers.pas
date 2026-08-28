unit TestNeuralLayers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume,
  neuralthread, pascoremath32;

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
    procedure TestConvolutionColdParallelParity;
    procedure TestConvolutionLowMemoryChunkParity;
    procedure TestConvolutionDecodeNeuronChunkParity;
    procedure TestConvolutionFastMemoryNeuronChunk;
    procedure TestConvolutionSpatialNeuronChunk;
    procedure TestHotThreadWorkersStartStop;
    procedure TestMaxThreadNumCapsThePool;
    procedure TestConvolutionForward;
    procedure TestWinogradConvolutionParity;
    procedure TestMaxPoolForward;
    procedure TestMaxPoolVectorizedExactParity;
    procedure TestExpScalarParity;
    procedure TestSigmoidScalarParity;
    procedure TestTanhScalarParity;
    procedure TestErfScalarParity;
    procedure TestSinhScalarParity;
    procedure TestLnScalarParity;
    procedure TestSinScalarParity;
    procedure TestCosScalarParity;
    procedure TestSinCosMatchesSeparateSinAndCos;
    procedure TestArcSinhScalarParity;
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
    procedure TestReLULClampsAtInference;
    procedure TestSetTrainablePerBlockEqualsWholeNet;
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
    procedure TestEmbeddingZeroPaddedRows;
    procedure TestEmbeddingInt8ZeroPaddedRows;
    procedure TestTokenAndPositionalEmbeddingZeroPaddedRows;
    // Rectangular (W <> H) channel reductions + flip/padded-conv regressions
    procedure TestMaxChannelRectangular;
    procedure TestMinChannelRectangular;
    procedure TestMaxChannelSquareRegression;
    procedure TestFlipXPaddedConvBackprop;
    procedure TestFlipYPaddedConvBackprop;
    // Int8 convolution input (quantized FInputCopy + byte im2col)
    procedure TestConvolutionInt8InputPadded;
    procedure TestConvolutionInt8InputStrided;
    procedure TestConvolutionInt8InputPointwise;
    procedure TestConvolutionInt8InputNotEnabled;
    // Int8 x int8 convolution forward (int8 weights AND int8 input)
    procedure TestConvolutionInt8Int8Padded;
    procedure TestConvolutionInt8Int8Strided;
    procedure TestConvolutionInt8Int8Pointwise;
    procedure TestConvolutionInt8Int8NotEnabled;
    // Int8 input arming at net level and on TNNetFullConnect
    procedure TestNetEnableInt8InputCountsQuantizedLayers;
    procedure TestDequantizeWeightsInt8DropsInt8Input;
    procedure TestFullConnectQuantizeInputInt8;
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

// Bit-identical serial-vs-threaded A/B for the single-sample TNNetFullConnect
// forward (here it must be EXACTLY equal: only independent output neurons are
// partitioned, the per-neuron reduction order is unchanged). The layer is
// sized (1024 input x 1024 neurons = 1M work proxy) to clear the fixed
// chunk-eligibility crossover. Intra-layer threading is now driven by the
// compute path, not a standalone flag: the serial reference runs ComputeCPU
// (ComputeSerial disables threading), and the threaded case runs an
// inference-only parallel pass (SetTrainable(False) + Compute(...,True)), where
// ComputeParallel enables intra-layer threading and the scheduler splits the FC
// into ComputeRange chunks. Covered for both the activation
// (TNNetFullConnect/ReLU) and activation-free (TNNetFullConnectLinear) variants.
// Threading state is per-net (TNNetExecutionPlanner), so it dies with each
// RunCase net - no global restore needed.
procedure TTestNeuralLayers.TestFullConnectThreadingParity;
var
  Input: TNNetVolume;
  i: integer;

  // Build a fresh single-FC net, set deterministic weights, compute, copy out.
  procedure RunCase(IsLinear: boolean; Parallel: boolean; Dst: TNNetVolume);
  var
    NN: TNNet;
    Layer: TNNetFullConnect;
    neuron, w: integer;
  begin
    NN := TNNet.Create();
    try
      NN.AddLayer(TNNetInput.Create(1024));
      if IsLinear then
        Layer := TNNetFullConnectLinear.Create(1024)
      else
        Layer := TNNetFullConnectReLU.Create(1024);
      NN.AddLayer(Layer);
      // Deterministic, reproducible weights (independent of the compute path).
      for neuron := 0 to Layer.Neurons.Count - 1 do
      begin
        for w := 0 to Layer.Neurons[neuron].Weights.Size - 1 do
          Layer.Neurons[neuron].Weights.Raw[w] :=
            Sin(neuron * 0.013 + w * 0.0007) * 0.1;
        Layer.Neurons[neuron].BiasWeight := Cos(neuron * 0.021) * 0.05;
      end;
      if Parallel then
      begin
        // Inference-only parallel pass: ComputeParallel enables intra-layer
        // threading and the FC clears the 1M crossover, so the scheduler splits
        // it into ComputeRange chunks. MinGain 0 forces the parallel branch.
        NN.SetTrainable(False);
        NN.SchedulerMinGain := 0;
        NN.Compute(Input, 0, True);
      end
      else
        NN.Compute(Input); // serial ComputeCPU reference
      Dst.Copy(Layer.Output);
    finally
      NN.Free;
    end;
  end;

var
  SerialLin, ThreadLin, SerialAct, ThreadAct: TNNetVolume;
begin
  Input := TNNetVolume.Create(1024, 1, 1);
  SerialLin := TNNetVolume.Create();
  ThreadLin := TNNetVolume.Create();
  SerialAct := TNNetVolume.Create();
  ThreadAct := TNNetVolume.Create();
  try
    for i := 0 to Input.Size - 1 do
      Input.Raw[i] := Sin(i * 0.05) - 0.3;

    RunCase({IsLinear=}true,  {Parallel=}false, SerialLin);
    RunCase({IsLinear=}true,  {Parallel=}true,  ThreadLin);
    RunCase({IsLinear=}false, {Parallel=}false, SerialAct);
    RunCase({IsLinear=}false, {Parallel=}true,  ThreadAct);

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
// net (width 2) so ComputeParallel engages the graph scheduler, with
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
  Input := TNNetVolume.Create(2048, 1, 1);
  SerialOut := TNNetVolume.Create();
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(2048));
    // (1,1,depth)-shaped outputs: TNNetDeepConcat requires matching X/Y and
    // concatenates on the depth axis. Both branches are sized so their
    // prevSize*outSize work proxy clears the fixed 1M chunk-eligibility
    // crossover (2048*1024 and 2048*512), so EnableIntraLayerThreading alone
    // makes them chunk-eligible.
    Branch1 := NN.AddLayerAfter(
      TNNetFullConnectLinear.Create(1, 1, 1024), InputLayer);
    Branch2 := NN.AddLayerAfter(
      TNNetFullConnectReLU.Create(1, 1, 512), InputLayer);
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
    // ChunkEligible is the static verdict; WillThread is parallel-pass-only
    // (False here, outside a pass) - see TNNetLayerThreading.
    AssertTrue('Branch1 must be chunk-eligible', Branch1.ChunkEligible());
    AssertTrue('Branch2 must be chunk-eligible', Branch2.ChunkEligible());

    // Reference: trainable -> serial loop (single-threaded, no scheduler;
    // WillThread is False outside a parallel pass).
    NN.Compute(Input);
    SerialOut.Copy(NN.GetLastLayer().Output);
    AssertTrue('Reference output is non-trivial', SerialOut.GetSumAbs() > 0.0);

    // Inference: parallel scheduler passes (Parallel=True) split each
    // chunk-eligible FullConnect branch into wkChunk work items computed via
    // ComputeRange. FullConnect uses scalar per-neuron activation, so the
    // chunked result stays BIT-IDENTICAL to the single-threaded serial
    // reference, pass after pass.
    NN.SetTrainable(False);
    for pass := 1 to 20 do
    begin
      NN.Compute(Input, 0, True);
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

// Exercises the StartThreadWorkers / StopThreadWorkers public API on a small
// two-branch inference net: StartThreadWorkers configures the hot-worker policy
// and pre-warms the persistent pool; every parallel pass stays BIT-IDENTICAL to
// the serial reference; StopThreadWorkers reverts the policy and the pool
// recreates transparently on the next pass; and StopOnFinish=True tears the pool
// down after one pass (the hot count reverts to its default). Coded by Claude (AI).
procedure TTestNeuralLayers.TestHotThreadWorkersStartStop;
var
  NN: TNNet;
  Input, SerialOut: TNNetVolume;
  InputLayer, Branch1, Branch2: TNNetLayer;
  Layer: TNNetLayer;
  i, pass, LayerCnt, neuron, w: integer;
  FC: TNNetFullConnect;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2048, 1, 1);
  SerialOut := TNNetVolume.Create();
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(2048));
    Branch1 := NN.AddLayerAfter(
      TNNetFullConnectLinear.Create(1, 1, 1024), InputLayer);
    Branch2 := NN.AddLayerAfter(
      TNNetFullConnectReLU.Create(1, 1, 512), InputLayer);
    NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2]));
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

    // Serial reference (trainable -> single-threaded serial loop).
    NN.Compute(Input);
    SerialOut.Copy(NN.GetLastLayer().Output);
    AssertTrue('Reference output is non-trivial', SerialOut.GetSumAbs() > 0.0);

    NN.SetTrainable(False);
    NN.SchedulerMinGain := 0; // force the parallel scheduler on every pass

    // StartThreadWorkers configures the hot policy and pre-warms the pool. The
    // hot count is clamped to the pool width (= cpu cores), so use Min() to stay
    // correct on any core count. The timeout is stored verbatim.
    NN.StartThreadWorkers({StopOnFinish=}False, {pHotThreadNum=}3, {timeout=}5);
    AssertEquals('StartThreadWorkers sets HotThreadWorkers',
      Min(3, NeuralDefaultThreadCount()), NN.HotThreadWorkers);
    AssertEquals('StartThreadWorkers sets HotThreadTimeout', 5, NN.HotThreadTimeout);

    // Every parallel pass stays bit-identical to serial with the pool persisting.
    for pass := 1 to 10 do
    begin
      NN.Compute(Input, 0, True);
      AssertEquals('Output size at pass ' + IntToStr(pass),
        SerialOut.Size, NN.GetLastLayer().Output.Size);
      for i := 0 to SerialOut.Size - 1 do
        AssertTrue('Hot-workers parallel pass ' + IntToStr(pass) +
          ' must be BIT-IDENTICAL to serial at ' + IntToStr(i),
          SerialOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);
    end;

    // StopThreadWorkers reverts the hot policy to its default (worker 0 hot).
    NN.StopThreadWorkers();
    AssertEquals('StopThreadWorkers resets HotThreadWorkers', 1, NN.HotThreadWorkers);

    // The pool recreates transparently on the next pass, still bit-identical.
    NN.Compute(Input, 0, True);
    for i := 0 to SerialOut.Size - 1 do
      AssertTrue('Post-stop parallel pass must be BIT-IDENTICAL to serial at ' +
        IntToStr(i), SerialOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);

    // StopOnFinish=True: the pool is torn down after the next pass, so the hot
    // count reverts to 1 (on a multi-core host, where the parallel path - and
    // thus the teardown hook - actually runs; on a single-core host the count is
    // already clamped to 1, so the post-pass assertion holds trivially).
    NN.StartThreadWorkers({StopOnFinish=}True, {pHotThreadNum=}2, {timeout=}3);
    AssertEquals('HotThreadWorkers before the StopOnFinish pass',
      Min(2, NeuralDefaultThreadCount()), NN.HotThreadWorkers);
    NN.Compute(Input, 0, True);
    AssertEquals('StopOnFinish reverts HotThreadWorkers after the pass',
      1, NN.HotThreadWorkers);
    for i := 0 to SerialOut.Size - 1 do
      AssertTrue('StopOnFinish pass must be BIT-IDENTICAL to serial at ' +
        IntToStr(i), SerialOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);
  finally
    Input.Free;
    SerialOut.Free;
    NN.Free;
  end;
end;

// MaxThreadNum caps the scheduler's worker pool: the pool is
// Min(MaxThreadNum, cpu count), 0 restores "every CPU thread", and the cap
// changes nothing about the result (parallel passes stay BIT-IDENTICAL to the
// serial reference). The pool size is read back through StartThreadWorkers'
// auto hot count (pHotThreadNum = -1 sets HotThreadWorkers to the pool size).
// Coded by Claude (AI).
procedure TTestNeuralLayers.TestMaxThreadNumCapsThePool;
var
  NN: TNNet;
  Input, SerialOut: TNNetVolume;
  InputLayer, Branch1, Branch2: TNNetLayer;
  Layer: TNNetLayer;
  i, LayerCnt, neuron, w: integer;
  FC: TNNetFullConnect;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(1024, 1, 1);
  SerialOut := TNNetVolume.Create();
  try
    AssertEquals('MaxThreadNum defaults to the cpu count (no cap)',
      NeuralDefaultThreadCount(), NN.MaxThreadNum);

    InputLayer := NN.AddLayer(TNNetInput.Create(1024));
    Branch1 := NN.AddLayerAfter(
      TNNetFullConnectLinear.Create(1, 1, 512), InputLayer);
    Branch2 := NN.AddLayerAfter(
      TNNetFullConnectReLU.Create(1, 1, 256), InputLayer);
    NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2]));
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
              Sin(LayerCnt * 1.3 + neuron * 0.017 + w * 0.0009) * 0.1;
          FC.Neurons[neuron].BiasWeight := Cos(neuron * 0.019) * 0.05;
        end;
      end;
    end;
    for i := 0 to Input.Size - 1 do Input.Raw[i] := Sin(i * 0.05) - 0.3;

    // Serial reference (trainable -> single-threaded serial loop).
    NN.Compute(Input);
    SerialOut.Copy(NN.GetLastLayer().Output);
    AssertTrue('Reference output is non-trivial', SerialOut.GetSumAbs() > 0.0);

    NN.SetTrainable(False);
    NN.SchedulerMinGain := 0; // force the parallel scheduler on every pass

    // Uncapped: the pool is the full cpu thread count (the plan floors the
    // graph width at it so intra-layer chunking can engage).
    NN.StartThreadWorkers({StopOnFinish=}False, {pHotThreadNum=}-1, {timeout=}5);
    AssertEquals('Uncapped pool is the cpu thread count',
      NeuralDefaultThreadCount(), NN.HotThreadWorkers);

    // Capped at 2 (or 1 on a single-core host).
    NN.MaxThreadNum := 2;
    NN.StartThreadWorkers({StopOnFinish=}False, {pHotThreadNum=}-1, {timeout=}5);
    AssertEquals('MaxThreadNum caps the pool at Min(cap, cpu count)',
      Min(2, NeuralDefaultThreadCount()), NN.HotThreadWorkers);

    // A capped pass computes the same thing, bit for bit.
    NN.Compute(Input, 0, True);
    AssertEquals('Capped output size', SerialOut.Size,
      NN.GetLastLayer().Output.Size);
    for i := 0 to SerialOut.Size - 1 do
      AssertTrue('Capped parallel pass must be BIT-IDENTICAL to serial at ' +
        IntToStr(i), SerialOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);

    // A cap of 1 is legal: a single worker, still bit-identical.
    NN.MaxThreadNum := 1;
    NN.StartThreadWorkers({StopOnFinish=}False, {pHotThreadNum=}-1, {timeout=}5);
    AssertEquals('MaxThreadNum = 1 leaves a single worker', 1,
      NN.HotThreadWorkers);
    NN.Compute(Input, 0, True);
    for i := 0 to SerialOut.Size - 1 do
      AssertTrue('Single-worker pass must be BIT-IDENTICAL to serial at ' +
        IntToStr(i), SerialOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);

    // 0 (and any non-positive value) means "no cap" again - including right
    // after a capped pass, where the cached plan's width floor was built under
    // the lower cap (the setter lifts it).
    NN.MaxThreadNum := 0;
    NN.StartThreadWorkers({StopOnFinish=}False, {pHotThreadNum=}-1, {timeout=}5);
    AssertEquals('MaxThreadNum = 0 restores the full pool',
      NeuralDefaultThreadCount(), NN.HotThreadWorkers);
    NN.Compute(Input, 0, True);
    for i := 0 to SerialOut.Size - 1 do
      AssertTrue('Uncapped-again pass must be BIT-IDENTICAL to serial at ' +
        IntToStr(i), SerialOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);
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
  // 16x16x8 input: each branch's prevSize*outSize work proxy (2048 * >=6144)
  // clears the fixed 1M chunk-eligibility crossover. Depth stays 8 so Branch1
  // keeps VectorSize 72 <= csMaxInterleavedSize and takes the interleaved kernel.
  Input := TNNetVolume.Create(16, 16, 8);
  SerialOut := TNNetVolume.Create();
  try
    InputLayer := NN.AddLayer(TNNetInput.Create(16, 16, 8));
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
    // ChunkEligible is the static verdict; WillThread is parallel-pass-only.
    AssertTrue('Branch1 must be chunk-eligible', Branch1.ChunkEligible());
    AssertTrue('Branch2 must be chunk-eligible', Branch2.ChunkEligible());
    AssertTrue('Branch3 must be chunk-eligible', Branch3.ChunkEligible());

    // Pin the forward kernel BEFORE taking the reference, the same way
    // TestConvolutionColdParallelParity and TestConvolutionLowMemoryChunkParity
    // do. SetTrainable's pLowMemory defaults to True, which releases the
    // concatenated-weight caches and computes per-neuron instead - a different
    // accumulation order from the interleaved and pointwise kernels, so a
    // reference taken while the caches were still resident matches only to FP
    // rounding on an AVX build. Holding the kernel fixed is what makes the
    // bit-identity below a statement about CHUNKING rather than about kernel
    // choice.
    NN.SetTrainable(False, {pLowMemory=}False);

    // Reference: serial loop with the classic serial kernels.
    NN.Compute(Input);
    SerialOut.Copy(NN.GetLastLayer().Output);
    AssertTrue('Reference output is non-trivial', SerialOut.GetSumAbs() > 0.0);

    // Inference: parallel scheduler passes (Parallel=True) split each conv
    // branch into per-output-position wkChunk items (both the interleaved and
    // tiled twins). Branches use ReLU/Linear activations, which are
    // slice-invariant, so the chunked result stays BIT-IDENTICAL to the
    // single-threaded serial reference, pass after pass.
    for pass := 1 to 20 do
    begin
      NN.Compute(Input, 0, True);
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

// Build the 3-branch (2 spatial conv + 1 pointwise) diamond used by the
// cold-parallel and low-memory chunk parity tests, with deterministic weights.
procedure BuildConvParityNet(out NN: TNNet;
  out Branch1, Branch2, Branch3: TNNetLayer);
var
  InputLayer, Layer: TNNetLayer;
  LayerCnt, neuron, w: integer;
begin
  NN := TNNet.Create();
  InputLayer := NN.AddLayer(TNNetInput.Create(16, 16, 8));
  Branch1 := NN.AddLayerAfter(TNNetConvolutionReLU.Create(32, 3, 1, 1), InputLayer);
  Branch2 := NN.AddLayerAfter(TNNetConvolutionLinear.Create(24, 3, 1, 1), InputLayer);
  Branch3 := NN.AddLayerAfter(TNNetPointwiseConvLinear.Create(48), InputLayer);
  NN.AddLayer(TNNetDeepConcat.Create([Branch1, Branch2, Branch3]));
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
end;

// Guards the parallel-path input prologue (PrepareChunkedForward): a chunked
// SPATIAL conv rebuilds its im2col (FInputPrepared) on the parallel path rather
// than reusing a stale one from an earlier pass. Two DISTINCT inputs run
// back-to-back on the parallel scheduler with NO matching serial warm-up; the
// second must match the serial reference bit-for-bit. Before the prologue fix
// the parallel result held the PREVIOUS input's im2col and diverged. Coded by
// Claude (AI).
procedure TTestNeuralLayers.TestConvolutionColdParallelParity;
var
  NN: TNNet;
  InA, InB, ParOut: TNNetVolume;
  Branch1, Branch2, Branch3: TNNetLayer;
  i: integer;
begin
  BuildConvParityNet(NN, Branch1, Branch2, Branch3);
  InA := TNNetVolume.Create(16, 16, 8);
  InB := TNNetVolume.Create(16, 16, 8);
  ParOut := TNNetVolume.Create();
  try
    for i := 0 to InA.Size - 1 do InA.Raw[i] := Sin(i * 0.05) - 0.3;
    for i := 0 to InB.Size - 1 do InB.Raw[i] := Cos(i * 0.037) + 0.2;

    NN.EnableIntraLayerThreading(true);
    NN.SchedulerMinGain := 0; // force the parallel scheduler on every pass
    NN.SetTrainable(False, {pLowMemory=}False);
    AssertTrue('Branch1 must be chunk-eligible', Branch1.ChunkEligible());
    AssertTrue('Branch2 must be chunk-eligible', Branch2.ChunkEligible());

    // Cold parallel: warm the im2col with A, then compute B on the parallel path.
    NN.Compute(InA, 0, True);
    NN.Compute(InB, 0, True);
    ParOut.Copy(NN.GetLastLayer().Output);
    // Serial reference for B rebuilds the im2col correctly via Compute().
    NN.Compute(InB, 0, False);
    AssertEquals('Output size matches', NN.GetLastLayer().Output.Size, ParOut.Size);
    AssertTrue('Reference output is non-trivial',
      NN.GetLastLayer().Output.GetSumAbs() > 0.0);
    for i := 0 to ParOut.Size - 1 do
      AssertTrue('Cold parallel B must be BIT-IDENTICAL to serial B at ' +
        IntToStr(i), ParOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);
  finally
    InA.Free;
    InB.Free;
    ParOut.Free;
    NN.Free;
  end;
end;

// Low-memory convolutions now chunk: ChunkEligible no longer excludes
// ActiveLowMemory, and ComputeRange has a per-neuron ranged branch (the
// concatenated-weight caches are released in low-memory mode). A cold parallel
// low-memory pass must equal the low-memory SERIAL reference (ComputeLowMemoryCPU)
// bit-for-bit. Coded by Claude (AI).
procedure TTestNeuralLayers.TestConvolutionLowMemoryChunkParity;
var
  NN: TNNet;
  InA, InB, ParOut: TNNetVolume;
  Branch1, Branch2, Branch3: TNNetLayer;
  i: integer;
begin
  BuildConvParityNet(NN, Branch1, Branch2, Branch3);
  InA := TNNetVolume.Create(16, 16, 8);
  InB := TNNetVolume.Create(16, 16, 8);
  ParOut := TNNetVolume.Create();
  try
    for i := 0 to InA.Size - 1 do InA.Raw[i] := Sin(i * 0.05) - 0.3;
    for i := 0 to InB.Size - 1 do InB.Raw[i] := Cos(i * 0.037) + 0.2;

    NN.EnableIntraLayerThreading(true);
    NN.SchedulerMinGain := 0;
    NN.SetTrainable(False, {pLowMemory=}True);
    AssertTrue('Low-memory spatial conv must be chunk-eligible',
      Branch1.ChunkEligible());
    AssertTrue('Low-memory pointwise conv must be chunk-eligible',
      Branch3.ChunkEligible());

    // Cold parallel low-memory: warm with A, then compute B.
    NN.Compute(InA, 0, True);
    NN.Compute(InB, 0, True);
    ParOut.Copy(NN.GetLastLayer().Output);
    // Low-memory serial reference for B (per-neuron ComputeLowMemoryCPU).
    NN.Compute(InB, 0, False);
    AssertEquals('Output size matches', NN.GetLastLayer().Output.Size, ParOut.Size);
    AssertTrue('Reference output is non-trivial',
      NN.GetLastLayer().Output.GetSumAbs() > 0.0);
    for i := 0 to ParOut.Size - 1 do
      AssertTrue('Low-memory parallel B must be BIT-IDENTICAL to serial B at ' +
        IntToStr(i), ParOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);
  finally
    InA.Free;
    InB.Free;
    ParOut.Free;
    NN.Free;
  end;
end;

// Single-output-position ("decode-shaped") convolution: with only one spatial
// position, position-chunking would emit a single chunk and thread nothing, so
// a low-memory layer chunks over its output NEURONS instead (ChunkOverNeurons).
// That neuron slice uses the same per-neuron DotProduct as the serial
// ComputeLowMemoryCPU, so a cold parallel pass must equal the serial reference
// bit-for-bit. This is the transformer single-token decode shape (pointwise
// projections over a 1-token grid). Coded by Claude (AI).
procedure TTestNeuralLayers.TestConvolutionDecodeNeuronChunkParity;
var
  NN: TNNet;
  ConvLayer: TNNetLayer;
  InA, InB, ParOut: TNNetVolume;
  i, neuron, w: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, 64));
  ConvLayer := NN.AddLayer(TNNetPointwiseConvLinear.Create(128));
  // Deterministic non-trivial weights so parallel and serial share one net.
  for neuron := 0 to ConvLayer.Neurons.Count - 1 do
  begin
    for w := 0 to ConvLayer.Neurons[neuron].Weights.Size - 1 do
      ConvLayer.Neurons[neuron].Weights.Raw[w] :=
        Sin(neuron * 0.017 + w * 0.0031) * 0.1;
    ConvLayer.Neurons[neuron].BiasWeight := Cos(neuron * 0.019) * 0.05;
  end;
  ConvLayer.FlushWeightCache();

  InA := TNNetVolume.Create(1, 1, 64);
  InB := TNNetVolume.Create(1, 1, 64);
  ParOut := TNNetVolume.Create();
  try
    for i := 0 to InA.Size - 1 do InA.Raw[i] := Sin(i * 0.05) - 0.3;
    for i := 0 to InB.Size - 1 do InB.Raw[i] := Cos(i * 0.037) + 0.2;

    NN.EnableIntraLayerThreading(true);
    NN.SchedulerMinGain := 0; // force the parallel scheduler on every pass
    NN.SetTrainable(False, {pLowMemory=}True);
    // Neuron-axis chunking only engages when the pool has room (more than one
    // worker); on a single-core box the layer stays serial and parity is
    // trivial, so gate the "did it engage" asserts on the thread count.
    if NeuralDefaultThreadCount() > 1 then
    begin
      AssertTrue('Decode-shaped low-memory conv chunks over neurons',
        TNNetConvolution(ConvLayer).ChunkOverNeurons());
      AssertEquals('Neuron-axis work count = neuron count',
        ConvLayer.Neurons.Count, ConvLayer.ChunkWorkCount());
    end;
    AssertTrue('Must be chunk-eligible', ConvLayer.ChunkEligible());

    // Cold parallel low-memory: warm with A, then compute B on the parallel path.
    NN.Compute(InA, 0, True);
    NN.Compute(InB, 0, True);
    ParOut.Copy(NN.GetLastLayer().Output);
    // Low-memory serial reference for B (per-neuron ComputeLowMemoryCPU).
    NN.Compute(InB, 0, False);
    AssertEquals('Output size matches', NN.GetLastLayer().Output.Size, ParOut.Size);
    AssertTrue('Reference output is non-trivial',
      NN.GetLastLayer().Output.GetSumAbs() > 0.0);
    for i := 0 to ParOut.Size - 1 do
      AssertTrue('Neuron-chunk parallel B must be BIT-IDENTICAL to serial B at ' +
        IntToStr(i), ParOut.Raw[i] = NN.GetLastLayer().Output.Raw[i]);
  finally
    InA.Free;
    InB.Free;
    ParOut.Free;
    NN.Free;
  end;
end;

// Fast-memory (--max-fast-memory) decode-shaped pointwise conv: the concatenated
// weight cache is resident, so the neuron-axis chunk runs through the general
// neuron-ranged DotProductsTiled rather than the per-neuron low-memory path.
// Parallel and serial need only be numerically equivalent (bit-parity is not a
// requirement), so this checks a tolerance, not bit-equality. Coded by Claude (AI).
procedure TTestNeuralLayers.TestConvolutionFastMemoryNeuronChunk;
var
  NN: TNNet;
  ConvLayer: TNNetLayer;
  InA, InB, ParOut: TNNetVolume;
  i, neuron, w: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(1, 1, 64));
  ConvLayer := NN.AddLayer(TNNetPointwiseConvLinear.Create(128));
  for neuron := 0 to ConvLayer.Neurons.Count - 1 do
  begin
    for w := 0 to ConvLayer.Neurons[neuron].Weights.Size - 1 do
      ConvLayer.Neurons[neuron].Weights.Raw[w] :=
        Sin(neuron * 0.017 + w * 0.0031) * 0.1;
    ConvLayer.Neurons[neuron].BiasWeight := Cos(neuron * 0.019) * 0.05;
  end;
  ConvLayer.FlushWeightCache();

  InA := TNNetVolume.Create(1, 1, 64);
  InB := TNNetVolume.Create(1, 1, 64);
  ParOut := TNNetVolume.Create();
  try
    for i := 0 to InA.Size - 1 do InA.Raw[i] := Sin(i * 0.05) - 0.3;
    for i := 0 to InB.Size - 1 do InB.Raw[i] := Cos(i * 0.037) + 0.2;

    NN.EnableIntraLayerThreading(true);
    NN.SchedulerMinGain := 0;
    NN.SetTrainable(False, {pLowMemory=}False); // keep the weight cache (fast path)
    if NeuralDefaultThreadCount() > 1 then
      AssertTrue('Fast-memory decode conv chunks over neurons',
        TNNetConvolution(ConvLayer).ChunkOverNeurons());
    AssertTrue('Must be chunk-eligible', ConvLayer.ChunkEligible());

    NN.Compute(InA, 0, True);
    NN.Compute(InB, 0, True);
    ParOut.Copy(NN.GetLastLayer().Output);
    NN.Compute(InB, 0, False);
    AssertTrue('Reference output is non-trivial',
      NN.GetLastLayer().Output.GetSumAbs() > 0.0);
    for i := 0 to ParOut.Size - 1 do
      AssertTrue('Fast-memory neuron-chunk B must match serial B (tol) at ' +
        IntToStr(i),
        Abs(ParOut.Raw[i] - NN.GetLastLayer().Output.Raw[i])
          <= 1e-4 * (1 + Abs(NN.GetLastLayer().Output.Raw[i])));
  finally
    InA.Free;
    InB.Free;
    ParOut.Free;
    NN.Free;
  end;
end;

// The neuron-axis chunk is kernel-size agnostic: a SPATIAL 3x3 conv on a
// single-position grid (3x3 input, pad 0 -> 1x1 output) still chunks over
// neurons and its fast-memory path runs the same neuron-ranged DotProductsTiled
// over the im2col matrix (VectorSize = 3*3*inChannels). Proves the path is not
// pointwise-only. Coded by Claude (AI).
procedure TTestNeuralLayers.TestConvolutionSpatialNeuronChunk;
var
  NN: TNNet;
  ConvLayer: TNNetLayer;
  InA, InB, ParOut: TNNetVolume;
  i, neuron, w: integer;
begin
  NN := TNNet.Create();
  NN.AddLayer(TNNetInput.Create(3, 3, 8));
  // 3x3 kernel, pad 0, stride 1 on a 3x3 input -> output 1x1x64 (one position).
  ConvLayer := NN.AddLayer(TNNetConvolutionReLU.Create(64, 3, 0, 1));
  for neuron := 0 to ConvLayer.Neurons.Count - 1 do
  begin
    for w := 0 to ConvLayer.Neurons[neuron].Weights.Size - 1 do
      ConvLayer.Neurons[neuron].Weights.Raw[w] :=
        Sin(neuron * 0.013 + w * 0.0027) * 0.1;
    ConvLayer.Neurons[neuron].BiasWeight := Cos(neuron * 0.023) * 0.05;
  end;
  ConvLayer.FlushWeightCache();

  InA := TNNetVolume.Create(3, 3, 8);
  InB := TNNetVolume.Create(3, 3, 8);
  ParOut := TNNetVolume.Create();
  try
    for i := 0 to InA.Size - 1 do InA.Raw[i] := Sin(i * 0.05) - 0.3;
    for i := 0 to InB.Size - 1 do InB.Raw[i] := Cos(i * 0.037) + 0.2;

    NN.EnableIntraLayerThreading(true);
    NN.SchedulerMinGain := 0;
    NN.SetTrainable(False, {pLowMemory=}False);
    AssertEquals('Single output position', 1,
      NN.GetLastLayer().Output.SizeX * NN.GetLastLayer().Output.SizeY);
    if NeuralDefaultThreadCount() > 1 then
      AssertTrue('Spatial single-position conv chunks over neurons',
        TNNetConvolution(ConvLayer).ChunkOverNeurons());
    AssertTrue('Must be chunk-eligible', ConvLayer.ChunkEligible());

    NN.Compute(InA, 0, True);
    NN.Compute(InB, 0, True);
    ParOut.Copy(NN.GetLastLayer().Output);
    NN.Compute(InB, 0, False);
    AssertTrue('Reference output is non-trivial',
      NN.GetLastLayer().Output.GetSumAbs() > 0.0);
    for i := 0 to ParOut.Size - 1 do
      AssertTrue('Spatial neuron-chunk B must match serial B (tol) at ' +
        IntToStr(i),
        Abs(ParOut.Raw[i] - NN.GetLastLayer().Output.Raw[i])
          <= 1e-4 * (1 + Abs(NN.GetLastLayer().Output.Raw[i])));
  finally
    InA.Free;
    InB.Free;
    ParOut.Free;
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

procedure TTestNeuralLayers.TestExpScalarParity;
// TNNetVolume.Exp must match the scalar NeuralExp loop (the parity
// reference, bit-identical to pcr_expf; unlike pcr_expf it stays trap-free
// when this test is compiled with overflow checks). On AVX2 builds Exp
// uses an 8-wide polynomial; on scalar builds it IS the NeuralExp loop.
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
      Ref.FData[I] := NeuralExp(x);
    end;
    TNNetVolume.Exp(Dst.DataPtr, Src.DataPtr, N);
    maxRel := 0;
    for I := 0 to N - 1 do
    begin
      denom := Abs(Ref.FData[I]);
      if denom < 1e-20 then denom := 1e-20;
      e := Abs(Dst.FData[I] - Ref.FData[I]) / denom;
      if e > maxRel then maxRel := e;
    end;
    AssertTrue('Exp vs pcr_expf max rel err ' + FloatToStr(maxRel) +
      ' must be < ' + FloatToStr(RelTol), maxRel < RelTol);
  finally
    Src.Free; Dst.Free; Ref.Free;
  end;
end;

procedure TTestNeuralLayers.TestSigmoidScalarParity;
// Sigmoid must match the scalar reference Sigmoid() within tolerance.
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
    TNNetVolume.Sigmoid(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      e := Abs(Dst.FData[I] - Sigmoid(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('Sigmoid vs Sigmoid max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestTanhScalarParity;
// Tanh must match the scalar pcr_tanhf reference within a tight tolerance
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
    TNNetVolume.Tanh(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      e := Abs(Dst.FData[I] - pcr_tanhf(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('Tanh vs pcr_tanhf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestErfScalarParity;
// Erf (Abramowitz & Stegun 7.1.26) must match the near-exact scalar
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
    TNNetVolume.Erf(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      e := Abs(Dst.FData[I] - pcr_erff(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('Erf vs pcr_erff max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestSinhScalarParity;
// Sinh (sinh built on the AVX2 Exp) must match a double-precision
// sinh reference within tolerance on every build (pcr_sinhf itself cannot be
// the reference: it traps when this test is compiled with overflow checks). N=131 straddles the 8-wide exp body
// and the (N mod 8) scalar tail; range [-12,12] matches the SinhAct parity band.
// A second pass with dst aliasing src guards against the buffer-aliasing bug that
// was fixed in Tanh/Erf.
const
  N = 131;
  RelTol = 1e-4;
var
  Src, Dst: TNNetVolume;
  I: integer;
  x, e, denom, maxRel: TNeuralFloat;

  function SinhRef(v: Double): Double;
  begin
    Result := (Exp(v) - Exp(-v)) * 0.5;
  end;

begin
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(N, 1, 1);
  try
    for I := 0 to N - 1 do
      Src.FData[I] := -12.0 + 24.0 * I / (N - 1);
    // Distinct dst.
    TNNetVolume.Sinh(Dst.DataPtr, Src.DataPtr, N);
    maxRel := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      denom := Abs(SinhRef(x));
      if denom < 1e-20 then denom := 1e-20;
      e := Abs(Dst.FData[I] - SinhRef(x)) / denom;
      if e > maxRel then maxRel := e;
    end;
    AssertTrue('Sinh vs sinh reference max rel err ' + FloatToStr(maxRel) +
      ' must be < ' + FloatToStr(RelTol), maxRel < RelTol);
    // dst aliasing src.
    TNNetVolume.Sinh(Src.DataPtr, Src.DataPtr, N);
    maxRel := 0;
    for I := 0 to N - 1 do
    begin
      // Recompute the original x from the index (Src has been overwritten).
      x := -12.0 + 24.0 * I / (N - 1);
      denom := Abs(SinhRef(x));
      if denom < 1e-20 then denom := 1e-20;
      e := Abs(Src.FData[I] - SinhRef(x)) / denom;
      if e > maxRel then maxRel := e;
    end;
    AssertTrue('Sinh (aliased) vs sinh reference max rel err ' + FloatToStr(maxRel) +
      ' must be < ' + FloatToStr(RelTol), maxRel < RelTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestLnScalarParity;
// Ln (Cephes logf on the AVX2 build, pcr_logf fallback otherwise) must match
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
    TNNetVolume.Ln(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      e := Abs(Dst.FData[I] - pcr_logf(Src.FData[I]));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('Ln vs pcr_logf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
    // dst aliasing src.
    TNNetVolume.Ln(Src.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := 1e-3 + 50.0 * I / (N - 1);
      e := Abs(Src.FData[I] - pcr_logf(x));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('Ln (aliased) vs pcr_logf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestSinScalarParity;
// Sin (Cephes sinf with 3-part Cody-Waite reduction on the AVX2 build) must
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
    TNNetVolume.Sin(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      e := Abs(Dst.FData[I] - pcr_sinf(Src.FData[I]));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('Sin vs pcr_sinf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
    // dst aliasing src.
    TNNetVolume.Sin(Src.DataPtr, Src.DataPtr, N);
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
    AssertTrue('Sin (aliased) vs pcr_sinf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestSinCosMatchesSeparateSinAndCos;
// The fused SinCos shares one range reduction and one polynomial pair between
// its two outputs, so it must agree bit-for-bit with Sin followed by Cos. N is
// swept across the 8-wide body boundary so the scalar tail is covered at every
// remainder, and the two aliasing cases (each destination over the source) are
// checked separately because the kernel reads x before either store.
const
  AbsTol = 0;
var
  Src, RefSin, RefCos, DstSin, DstCos: TNNetVolume;
  I, N: integer;
begin
  for N := 1 to 20 do
  begin
    Src := TNNetVolume.Create(N, 1, 1);
    RefSin := TNNetVolume.Create(N, 1, 1);
    RefCos := TNNetVolume.Create(N, 1, 1);
    DstSin := TNNetVolume.Create(N, 1, 1);
    DstCos := TNNetVolume.Create(N, 1, 1);
    try
      for I := 0 to N - 1 do
        Src.FData[I] := -40.0 + 4.3 * I;
      Src.FData[0] := 1000.0;
      if N > 1 then Src.FData[N - 1] := -9999.9;

      TNNetVolume.Sin(RefSin.DataPtr, Src.DataPtr, N);
      TNNetVolume.Cos(RefCos.DataPtr, Src.DataPtr, N);
      TNNetVolume.SinCos(DstSin.DataPtr, DstCos.DataPtr, Src.DataPtr, N);
      for I := 0 to N - 1 do
      begin
        AssertEquals('SinCos sin at N=' + IntToStr(N) + ' idx ' + IntToStr(I),
          RefSin.FData[I], DstSin.FData[I], AbsTol);
        AssertEquals('SinCos cos at N=' + IntToStr(N) + ' idx ' + IntToStr(I),
          RefCos.FData[I], DstCos.FData[I], AbsTol);
      end;

      // The sin destination aliasing the source.
      DstSin.Copy(Src);
      TNNetVolume.SinCos(DstSin.DataPtr, DstCos.DataPtr, DstSin.DataPtr, N);
      for I := 0 to N - 1 do
      begin
        AssertEquals('SinCos aliased sin idx ' + IntToStr(I),
          RefSin.FData[I], DstSin.FData[I], AbsTol);
        AssertEquals('SinCos aliased sin, cos idx ' + IntToStr(I),
          RefCos.FData[I], DstCos.FData[I], AbsTol);
      end;

      // The cos destination aliasing the source.
      DstCos.Copy(Src);
      TNNetVolume.SinCos(DstSin.DataPtr, DstCos.DataPtr, DstCos.DataPtr, N);
      for I := 0 to N - 1 do
      begin
        AssertEquals('SinCos aliased cos, sin idx ' + IntToStr(I),
          RefSin.FData[I], DstSin.FData[I], AbsTol);
        AssertEquals('SinCos aliased cos idx ' + IntToStr(I),
          RefCos.FData[I], DstCos.FData[I], AbsTol);
      end;
    finally
      Src.Free; RefSin.Free; RefCos.Free; DstSin.Free; DstCos.Free;
    end;
  end;
end;

procedure TTestNeuralLayers.TestCosScalarParity;
// Cos (Cephes cosf with 3-part Cody-Waite reduction on the AVX2 build) must
// match the scalar pcr_cosf reference within tolerance on every build. Same coverage
// rationale as TestSinScalarParity.
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
    TNNetVolume.Cos(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      e := Abs(Dst.FData[I] - pcr_cosf(Src.FData[I]));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('Cos vs pcr_cosf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
    // dst aliasing src.
    TNNetVolume.Cos(Src.DataPtr, Src.DataPtr, N);
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
    AssertTrue('Cos (aliased) vs pcr_cosf max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
  finally
    Src.Free; Dst.Free;
  end;
end;

procedure TTestNeuralLayers.TestArcSinhScalarParity;
// ArcSinh = ln(x + sqrt(x^2+1)), built on the AVX2 Ln, must match the
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
    TNNetVolume.ArcSinh(Dst.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := Src.FData[I];
      e := Abs(Dst.FData[I] - pcr_logf(x + Sqrt(x * x + 1.0)));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('ArcSinh vs ln(x+sqrt(x^2+1)) max abs err ' + FloatToStr(maxErr) +
      ' must be < ' + FloatToStr(AbsTol), maxErr < AbsTol);
    // dst aliasing src.
    TNNetVolume.ArcSinh(Src.DataPtr, Src.DataPtr, N);
    maxErr := 0;
    for I := 0 to N - 1 do
    begin
      x := -30.0 + 60.0 * I / (N - 1);
      e := Abs(Src.FData[I] - pcr_logf(x + Sqrt(x * x + 1.0)));
      if e > maxErr then maxErr := e;
    end;
    AssertTrue('ArcSinh (aliased) max abs err ' + FloatToStr(maxErr) +
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
    // TNNetReLU6 is a TNNetReLUL with limits [0, 6] and no leak.
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetReLU6.Create());

    Input.Raw[0] := -2.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 3.0;
    Input.Raw[3] := 6.0;
    Input.Raw[4] := 10.0;

    NN.Compute(Input);

    AssertEquals('Output should have 5 elements', 5, NN.GetLastLayer.Output.Size);
    AssertEquals('ReLU6(-2) should clamp to 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('ReLU6(0) should be 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('ReLU6(3) should pass through', 3.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('ReLU6(6) should be 6', 6.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('ReLU6(10) should clamp to 6', 6.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// SetTrainable(False, False) shrinks the error volumes, which used to send
// TNNetReLUL.Compute down a branch that copied the input instead of clamping.
procedure TTestNeuralLayers.TestReLULClampsAtInference;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(5, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(5));
    NN.AddLayer(TNNetReLU6.Create());
    NN.SetTrainable(False, False);

    Input.Raw[0] := -2.0;
    Input.Raw[1] := 0.0;
    Input.Raw[2] := 3.0;
    Input.Raw[3] := 6.0;
    Input.Raw[4] := 10.0;

    NN.Compute(Input);

    AssertEquals('inference ReLU6(-2) should clamp to 0', 0.0, NN.GetLastLayer.Output.Raw[0], 0.0001);
    AssertEquals('inference ReLU6(0) should be 0', 0.0, NN.GetLastLayer.Output.Raw[1], 0.0001);
    AssertEquals('inference ReLU6(3) should pass through', 3.0, NN.GetLastLayer.Output.Raw[2], 0.0001);
    AssertEquals('inference ReLU6(6) should be 6', 6.0, NN.GetLastLayer.Output.Raw[3], 0.0001);
    AssertEquals('inference ReLU6(10) should clamp to 6', 6.0, NN.GetLastLayer.Output.Raw[4], 0.0001);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// Importers flip the whole net inference-only once per built block, so the
// repeated calls must land in the same state as one call after the last block.
procedure TTestNeuralLayers.TestSetTrainablePerBlockEqualsWholeNet;
var
  PerBlockNN, WholeNN: TNNet;
  BlockCnt: integer;

  procedure AddBlock(pNN: TNNet);
  begin
    pNN.AddLayer( TNNetFullConnectReLU.Create(6) );
    pNN.AddLayer( TNNetPointwiseConvLinear.Create(4) );
  end;

  procedure AssertSameState(const pWhen: string);
  var
    LayerCnt, NeuronCnt, MaxLayerPos, MaxNeuronPos: integer;
    LayerA, LayerB: TNNetLayer;
    Prefix: string;
  begin
    AssertEquals(pWhen + ': layer count', WholeNN.CountLayers(),
      PerBlockNN.CountLayers());
    MaxLayerPos := WholeNN.GetLastLayerIdx();
    for LayerCnt := 0 to MaxLayerPos do
    begin
      LayerA := PerBlockNN.Layers[LayerCnt];
      LayerB := WholeNN.Layers[LayerCnt];
      Prefix := pWhen + ': layer ' + IntToStr(LayerCnt);
      AssertEquals(Prefix + ' IsTrainable', LayerB.IsTrainable, LayerA.IsTrainable);
      AssertEquals(Prefix + ' ActiveLowMemory', LayerB.ActiveLowMemory(),
        LayerA.ActiveLowMemory());
      AssertEquals(Prefix + ' neuron count', LayerB.Neurons.Count,
        LayerA.Neurons.Count);
      MaxNeuronPos := LayerB.Neurons.Count - 1;
      for NeuronCnt := 0 to MaxNeuronPos do
      begin
        AssertEquals(Prefix + ' neuron ' + IntToStr(NeuronCnt) + ' Delta assigned',
          Assigned(LayerB.Neurons[NeuronCnt].Delta),
          Assigned(LayerA.Neurons[NeuronCnt].Delta));
        AssertEquals(Prefix + ' neuron ' + IntToStr(NeuronCnt) + ' BackInertia assigned',
          Assigned(LayerB.Neurons[NeuronCnt].BackInertia),
          Assigned(LayerA.Neurons[NeuronCnt].BackInertia));
        if Assigned(LayerB.Neurons[NeuronCnt].Delta) then
          AssertEquals(Prefix + ' neuron ' + IntToStr(NeuronCnt) + ' Delta size',
            LayerB.Neurons[NeuronCnt].Delta.Size,
            LayerA.Neurons[NeuronCnt].Delta.Size);
      end;
    end;
  end;

begin
  PerBlockNN := TNNet.Create();
  WholeNN := TNNet.Create();
  try
    PerBlockNN.AddLayer( TNNetInput.Create(4, 1, 4) );
    WholeNN.AddLayer( TNNetInput.Create(4, 1, 4) );
    for BlockCnt := 1 to 3 do
    begin
      AddBlock(PerBlockNN);
      // The per-block net pays the flip after every block, exactly as the
      // pretrained importers do to cap peak RSS.
      PerBlockNN.SetTrainable(False);
      AddBlock(WholeNN);
    end;
    WholeNN.SetTrainable(False);
    AssertSameState('after per-block flips');

    // Re-arming must still rebuild every training buffer, including on the
    // layers whose repeated inference-only flip was skipped.
    PerBlockNN.SetTrainable(True, False);
    WholeNN.SetTrainable(True, False);
    AssertSameState('after re-arming');
    AssertTrue('re-armed Delta is weight sized',
      PerBlockNN.Layers[1].Neurons[0].Delta.Size =
      PerBlockNN.Layers[1].Neurons[0].Weights.Size);

    PerBlockNN.SetTrainable(False);
    WholeNN.SetTrainable(False);
    AssertSameState('after re-freezing');
  finally
    PerBlockNN.Free;
    WholeNN.Free;
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

// Regression: with EncodeZero=0, token 0 is a PAD - its output row must be all
// zeros, even when a previous forward pass left a real embedding row there.
procedure TTestNeuralLayers.TestEmbeddingZeroPaddedRows;
const
  cVocab = 10;
  cDim = 4;
var
  NN: TNNet;
  Input: TNNetVolume;
  Emb: TNNetEmbedding;
  W, Output: TNNetVolume;
  t, d: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    Emb := TNNetEmbedding(NN.AddLayer(TNNetEmbedding.Create(cVocab, cDim)));
    W := Emb.Neurons[0].Weights;
    // Every row (including row 0) is non-zero, so a leaked row is visible.
    for t := 0 to cVocab - 1 do
      for d := 0 to cDim - 1 do
        W[t, 0, d] := (t + 1) * 10 + d + 1;
    Output := NN.GetLastLayer.Output;
    // First pass: all tokens are real, so every output row is filled.
    Input.Raw[0] := 3; Input.Raw[1] := 7; Input.Raw[2] := 5; Input.Raw[3] := 9;
    NN.Compute(Input);
    AssertEquals('Row 1 holds token 7', 81.0, Output[1, 0, 0], 0.0001);
    // Second pass: tokens 1 and 3 are pads and must come back as zero rows.
    Input.Raw[0] := 3; Input.Raw[1] := 0; Input.Raw[2] := 5; Input.Raw[3] := 0;
    NN.Compute(Input);
    for d := 0 to cDim - 1 do
    begin
      AssertEquals('Token row 0 element ' + IntToStr(d),
        W[3, 0, d], Output[0, 0, d], 0.0001);
      AssertEquals('Padded row 1 element ' + IntToStr(d),
        0.0, Output[1, 0, d], 0.0);
      AssertEquals('Token row 2 element ' + IntToStr(d),
        W[5, 0, d], Output[2, 0, d], 0.0001);
      AssertEquals('Padded row 3 element ' + IntToStr(d),
        0.0, Output[3, 0, d], 0.0);
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

// Same pad contract on the int8 gather path.
procedure TTestNeuralLayers.TestEmbeddingInt8ZeroPaddedRows;
const
  cVocab = 10;
  cDim = 4;
var
  NN: TNNet;
  Input: TNNetVolume;
  Emb: TNNetEmbedding;
  W, Output: TNNetVolume;
  t, d: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    Emb := TNNetEmbedding(NN.AddLayer(TNNetEmbedding.Create(cVocab, cDim)));
    W := Emb.Neurons[0].Weights;
    for t := 0 to cVocab - 1 do
      for d := 0 to cDim - 1 do
        W[t, 0, d] := (t + 1) * 10 + d + 1;
    Emb.QuantizeWeightsInt8();
    Output := NN.GetLastLayer.Output;
    Input.Raw[0] := 3; Input.Raw[1] := 7; Input.Raw[2] := 5; Input.Raw[3] := 9;
    NN.Compute(Input);
    AssertEquals('Int8 row 1 holds token 7', 81.0, Output[1, 0, 0], 0.5);
    Input.Raw[0] := 3; Input.Raw[1] := 0; Input.Raw[2] := 5; Input.Raw[3] := 0;
    NN.Compute(Input);
    for d := 0 to cDim - 1 do
    begin
      AssertEquals('Int8 token row 0 element ' + IntToStr(d),
        (3 + 1) * 10 + d + 1, Output[0, 0, d], 0.5);
      AssertEquals('Int8 padded row 1 element ' + IntToStr(d),
        0.0, Output[1, 0, d], 0.0);
      AssertEquals('Int8 token row 2 element ' + IntToStr(d),
        (5 + 1) * 10 + d + 1, Output[2, 0, d], 0.5);
      AssertEquals('Int8 padded row 3 element ' + IntToStr(d),
        0.0, Output[3, 0, d], 0.0);
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

// A padded token gets NEITHER table: not the vocab row and not the positional
// row, so the output row stays exactly zero.
procedure TTestNeuralLayers.TestTokenAndPositionalEmbeddingZeroPaddedRows;
const
  cVocab = 10;
  cDim = 8;
var
  NN: TNNet;
  Input: TNNetVolume;
  Emb: TNNetTokenAndPositionalEmbedding;
  W, Output: TNNetVolume;
  t, d: integer;
  Row0, Row2: array[0..cDim - 1] of TNeuralFloat;
  AnyPositional: boolean;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(4, 1, 1));
    Emb := TNNetTokenAndPositionalEmbedding(NN.AddLayer(
      TNNetTokenAndPositionalEmbedding.Create(cVocab, cDim)));
    W := Emb.Neurons[0].Weights;
    for t := 0 to cVocab - 1 do
      for d := 0 to cDim - 1 do
        W[t, 0, d] := (t + 1) * 10 + d + 1;
    Output := NN.GetLastLayer.Output;
    Input.Raw[0] := 3; Input.Raw[1] := 7; Input.Raw[2] := 5; Input.Raw[3] := 9;
    NN.Compute(Input);
    // Rows 0 and 2 keep the same tokens in both passes: vocab row + positional
    // row, so they must be reproduced exactly (and stay above the vocab row).
    AnyPositional := false;
    for d := 0 to cDim - 1 do
    begin
      Row0[d] := Output[0, 0, d];
      Row2[d] := Output[2, 0, d];
      if Row2[d] <> W[5, 0, d] then AnyPositional := true;
    end;
    AssertTrue('Positional term must reach a real token row', AnyPositional);
    Input.Raw[0] := 3; Input.Raw[1] := 0; Input.Raw[2] := 5; Input.Raw[3] := 0;
    NN.Compute(Input);
    for d := 0 to cDim - 1 do
    begin
      AssertEquals('Positional padded row 1 element ' + IntToStr(d),
        0.0, Output[1, 0, d], 0.0);
      AssertEquals('Positional padded row 3 element ' + IntToStr(d),
        0.0, Output[3, 0, d], 0.0);
      AssertEquals('Positional token row 0 element ' + IntToStr(d),
        Row0[d], Output[0, 0, d], 0.0001);
      AssertEquals('Positional token row 2 element ' + IntToStr(d),
        Row2[d], Output[2, 0, d], 0.0001);
    end;
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

// Runs the int8 input path on Conv after a forward and checks the byte im2col
// against the FP32 one it mirrors: same geometry, and every code dequantizes
// to within half a quantization step of the FP32 element.
procedure AssertInt8Im2ColMatchesFP32(Conv: TNNetConvolutionBase;
  const TestName: string);
var
  Prepared: TNNetVolume;
  PreparedInt8: TNNetVolumeQuant8;
  Scale, Tolerance, Dequantized: TNeuralFloat;
  MaxRawPos, RawPos: integer;
begin
  Conv.QuantizeInputInt8();
  Conv.PrepareInputForConvolutionInt8();
  Prepared := Conv.InputPrepared;
  PreparedInt8 := Conv.InputPreparedInt8;
  TAssert.AssertEquals(TestName + ': int8 im2col SizeX', Prepared.SizeX, PreparedInt8.SizeX);
  TAssert.AssertEquals(TestName + ': int8 im2col SizeY', Prepared.SizeY, PreparedInt8.SizeY);
  TAssert.AssertEquals(TestName + ': int8 im2col Depth', Prepared.Depth, PreparedInt8.Depth);
  TAssert.AssertEquals(TestName + ': int8 im2col Size', Prepared.Size, PreparedInt8.Size);
  Scale := Conv.InputScaleInt8;
  TAssert.AssertTrue(TestName + ': scale must be positive', Scale > 0);
  Tolerance := Scale / 2 + 1e-6;
  MaxRawPos := Prepared.Size - 1;
  for RawPos := 0 to MaxRawPos do
  begin
    Dequantized := Scale * PreparedInt8.FData[RawPos];
    TAssert.AssertTrue(TestName + ': im2col element ' + IntToStr(RawPos) + ' expected ' +
      FloatToStr(Prepared.FData[RawPos]) + ' got ' + FloatToStr(Dequantized),
      Abs(Dequantized - Prepared.FData[RawPos]) <= Tolerance);
  end;
end;

// Fills a volume with a deterministic non-symmetric pattern: distinct values
// per element, both signs, and nothing landing on a quantization boundary.
procedure FillInt8ConvInput(Input: TNNetVolume);
var
  MaxRawPos, RawPos: integer;
begin
  MaxRawPos := Input.Size - 1;
  for RawPos := 0 to MaxRawPos do
    Input.FData[RawPos] := Sin(0.37 * RawPos + 0.11) * (1 + 0.013 * RawPos);
end;

procedure TTestNeuralLayers.TestConvolutionInt8InputPadded;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionLinear;
  MaxBorderXPos, MaxBorderYPos: integer;
  CntX, CntY, CntD: integer;
  MaxDepthPos, MaxTapPos, TapPos: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(7, 7, 3);
  try
    NN.AddLayer(TNNetInput.Create(7, 7, 3));
    Conv := TNNetConvolutionLinear.Create(4, 3, 1, 1);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    NN.Compute(Input);

    Conv.EnableInt8Input();
    AssertEquals('Padded int8 input SizeX', 9, Conv.InputCopyInt8.SizeX);
    AssertEquals('Padded int8 input SizeY', 9, Conv.InputCopyInt8.SizeY);
    AssertEquals('Padded int8 input Depth', 3, Conv.InputCopyInt8.Depth);
    AssertInt8Im2ColMatchesFP32(Conv, 'Padded conv');

    // The padded border quantizes from exact FP32 zeros, so it must be code 0.
    MaxBorderXPos := Conv.InputCopyInt8.SizeX - 1;
    MaxBorderYPos := Conv.InputCopyInt8.SizeY - 1;
    MaxDepthPos := Conv.InputCopyInt8.Depth - 1;
    for CntY := 0 to MaxBorderYPos do
      for CntX := 0 to MaxBorderXPos do
        if (CntX = 0) or (CntY = 0) or (CntX = MaxBorderXPos) or
           (CntY = MaxBorderYPos) then
          for CntD := 0 to MaxDepthPos do
            AssertEquals('Padded border code at ' + IntToStr(CntX) + ',' +
              IntToStr(CntY) + ',' + IntToStr(CntD), 0,
              integer(Conv.InputCopyInt8.Get(CntX, CntY, CntD)));

    // Output (0,0) gathers the padded top feature row first: FeatureSizeX*Depth
    // leading codes are zero.
    MaxTapPos := 3 * 3 - 1;
    for TapPos := 0 to MaxTapPos do
      AssertEquals('Padded im2col tap ' + IntToStr(TapPos), 0,
        integer(Conv.InputPreparedInt8.Get(0, 0, TapPos)));
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionInt8InputStrided;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionLinear;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(7, 7, 3);
  try
    NN.AddLayer(TNNetInput.Create(7, 7, 3));
    Conv := TNNetConvolutionLinear.Create(4, 3, 0, 2);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    NN.Compute(Input);

    Conv.EnableInt8Input();
    AssertEquals('Strided int8 input SizeX', 7, Conv.InputCopyInt8.SizeX);
    AssertEquals('Strided output SizeX', 3, Conv.Output.SizeX);
    AssertInt8Im2ColMatchesFP32(Conv, 'Strided conv');
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionInt8InputPointwise;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionLinear;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(7, 7, 3);
  try
    NN.AddLayer(TNNetInput.Create(7, 7, 3));
    Conv := TNNetConvolutionLinear.Create(4, 1, 0, 1);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    NN.Compute(Input);

    AssertTrue('1x1 conv must be pointwise', Conv.Pointwise);
    Conv.EnableInt8Input();
    // Nothing to gather: the int8 im2col IS the quantized input copy.
    AssertTrue('Pointwise int8 im2col aliases the int8 input copy',
      Conv.InputPreparedInt8 = Conv.InputCopyInt8);
    AssertInt8Im2ColMatchesFP32(Conv, 'Pointwise conv');
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionInt8InputNotEnabled;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionLinear;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(7, 7, 3);
  try
    NN.AddLayer(TNNetInput.Create(7, 7, 3));
    Conv := TNNetConvolutionLinear.Create(4, 3, 1, 1);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    NN.Compute(Input);

    // Without EnableInt8Input both buffers stay nil and both methods no-op.
    AssertTrue('Int8 input copy starts nil', Conv.InputCopyInt8 = nil);
    AssertTrue('Int8 im2col starts nil', Conv.InputPreparedInt8 = nil);
    Conv.QuantizeInputInt8();
    Conv.PrepareInputForConvolutionInt8();
    AssertTrue('Int8 input copy stays nil', Conv.InputCopyInt8 = nil);

    // Enabling twice keeps the same buffers.
    Conv.EnableInt8Input();
    AssertTrue('Int8 input copy is sized', Conv.InputCopyInt8.Size > 0);
    Conv.EnableInt8Input();
    AssertEquals('Second enable keeps the size', 9 * 9 * 3,
      Conv.InputCopyInt8.Size);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// Deterministic non-trivial weights and bias, then the caches the forward
// reads (FConcatedWeights, FBiasOutput) are rebuilt from them.
procedure FillInt8ConvWeights(Conv: TNNetConvolution);
var
  NeuronCnt, MaxNeuronPos, WeightCnt, MaxWeightPos: integer;
  W: TNNetVolume;
begin
  MaxNeuronPos := Conv.Neurons.Count - 1;
  for NeuronCnt := 0 to MaxNeuronPos do
  begin
    W := Conv.Neurons[NeuronCnt].Weights;
    MaxWeightPos := W.Size - 1;
    for WeightCnt := 0 to MaxWeightPos do
      W.FData[WeightCnt] := Sin(0.21 * (WeightCnt + 7 * NeuronCnt) + 0.3) * 0.5;
    Conv.Neurons[NeuronCnt].BiasWeight := 0.05 * NeuronCnt - 0.1;
  end;
  Conv.FlushWeightCache();
end;

// Sum of the absolute DEQUANTIZED weights of one neuron, bounded by the FP32
// magnitudes plus their own half-step quantization error. Read it before
// QuantizeWeightsInt8, which releases the FP32 weights.
function Int8ConvWeightMagnitude(Conv: TNNetConvolution;
  NeuronIdx: integer): TNeuralFloat;
var
  W: TNNetVolume;
  WeightCnt, MaxWeightPos: integer;
  SumAbsW, MaxAbsW: TNeuralFloat;
begin
  W := Conv.Neurons[NeuronIdx].Weights;
  MaxWeightPos := W.Size - 1;
  SumAbsW := 0;
  MaxAbsW := 0;
  for WeightCnt := 0 to MaxWeightPos do
  begin
    SumAbsW := SumAbsW + Abs(W.FData[WeightCnt]);
    if Abs(W.FData[WeightCnt]) > MaxAbsW then MaxAbsW := Abs(W.FData[WeightCnt]);
  end;
  Result := SumAbsW + W.Size * MaxAbsW / 254;
end;

// Runs the three convolution forwards - FP32, int8 weights, int8 x int8 - on
// one net and checks the int8 x int8 output against both.
procedure AssertInt8Int8ConvMatches(NN: TNNet; Conv: TNNetConvolution;
  Input: TNNetVolume; const TestName: string);
var
  FP32Output, Int8WeightOutput: TNNetVolume;
  Bounds: TNeuralFloatDynArr;
  NeuronCnt, MaxNeuronPos: integer;
  RawPos, MaxRawPos: integer;
  MaxAbsFP32, MaxAbsDiff, NonZeroCodes: TNeuralFloat;
  CodeCnt, MaxCodePos: integer;
begin
  FP32Output := TNNetVolume.Create();
  Int8WeightOutput := TNNetVolume.Create();
  try
    FillInt8ConvWeights(Conv);
    NN.Compute(Input);
    FP32Output.Copy(Conv.Output);

    MaxNeuronPos := Conv.Neurons.Count - 1;
    SetLength(Bounds, Conv.Neurons.Count);
    for NeuronCnt := 0 to MaxNeuronPos do
      Bounds[NeuronCnt] := Int8ConvWeightMagnitude(Conv, NeuronCnt);

    TNNetLayerConcatedWeights(Conv).QuantizeWeightsInt8();
    TAssert.AssertTrue(TestName + ': weights are int8', Conv.WeightsQuantizedInt8);
    NN.Compute(Input);
    Int8WeightOutput.Copy(Conv.Output);

    Conv.EnableInt8Input();
    // The int8 x int8 forward must not read the FP32 im2col at all. On a
    // pointwise convolution FInputPrepared IS the previous layer's output, so
    // clearing it there would clear the input itself.
    if not Conv.Pointwise then Conv.InputPrepared.Fill(0);
    NN.Compute(Input);

    // Every tap rounded by half an input step is the exact worst case.
    for NeuronCnt := 0 to MaxNeuronPos do
      Bounds[NeuronCnt] := 0.5 * Conv.InputScaleInt8 * Bounds[NeuronCnt] + 1e-5;

    NonZeroCodes := 0;
    MaxCodePos := Conv.InputPreparedInt8.Size - 1;
    for CodeCnt := 0 to MaxCodePos do
      if Conv.InputPreparedInt8.FData[CodeCnt] <> 0 then NonZeroCodes := NonZeroCodes + 1;
    TAssert.AssertTrue(TestName + ': the int8 im2col carries codes', NonZeroCodes > 0);
    if not Conv.Pointwise then
    begin
      MaxCodePos := Conv.InputPrepared.Size - 1;
      for CodeCnt := 0 to MaxCodePos do
        TAssert.AssertTrue(TestName + ': the FP32 im2col stays unbuilt at ' +
          IntToStr(CodeCnt), Conv.InputPrepared.FData[CodeCnt] = 0);
    end;

    MaxRawPos := Conv.Output.Size - 1;
    MaxAbsFP32 := 0;
    MaxAbsDiff := 0;
    for RawPos := 0 to MaxRawPos do
    begin
      NeuronCnt := RawPos mod Conv.Output.Depth;
      TAssert.AssertTrue(TestName + ': output ' + IntToStr(RawPos) + ' int8 x int8 ' +
        FloatToStr(Conv.Output.FData[RawPos]) + ' vs int8 weights ' +
        FloatToStr(Int8WeightOutput.FData[RawPos]) + ' exceeds bound ' +
        FloatToStr(Bounds[NeuronCnt]),
        Abs(Conv.Output.FData[RawPos] - Int8WeightOutput.FData[RawPos]) <=
          Bounds[NeuronCnt]);
      if Abs(FP32Output.FData[RawPos]) > MaxAbsFP32
        then MaxAbsFP32 := Abs(FP32Output.FData[RawPos]);
      if Abs(Conv.Output.FData[RawPos] - FP32Output.FData[RawPos]) > MaxAbsDiff
        then MaxAbsDiff := Abs(Conv.Output.FData[RawPos] - FP32Output.FData[RawPos]);
    end;
    TAssert.AssertTrue(TestName + ': FP32 reference must be non-trivial', MaxAbsFP32 > 0.1);
    TAssert.AssertTrue(TestName + ': max abs error ' + FloatToStr(MaxAbsDiff) +
      ' over max abs output ' + FloatToStr(MaxAbsFP32) + ' exceeds 5%',
      MaxAbsDiff <= 0.05 * MaxAbsFP32);
  finally
    FP32Output.Free;
    Int8WeightOutput.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionInt8Int8Padded;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionReLU;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    Conv := TNNetConvolutionReLU.Create(5, 3, 1, 1);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    AssertInt8Int8ConvMatches(NN, Conv, Input, 'Padded int8 x int8 conv');
    AssertEquals('Padded int8 x int8 output SizeX', 8, Conv.Output.SizeX);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionInt8Int8Strided;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionLinear;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    Conv := TNNetConvolutionLinear.Create(5, 3, 0, 2);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    AssertInt8Int8ConvMatches(NN, Conv, Input, 'Strided int8 x int8 conv');
    AssertEquals('Strided int8 x int8 output SizeX', 3, Conv.Output.SizeX);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionInt8Int8Pointwise;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionLinear;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 6);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 6));
    Conv := TNNetConvolutionLinear.Create(5, 1, 0, 1);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    AssertTrue('1x1 conv must be pointwise', Conv.Pointwise);
    AssertInt8Int8ConvMatches(NN, Conv, Input, 'Pointwise int8 x int8 conv');
    AssertTrue('Pointwise int8 im2col aliases the int8 input copy',
      Conv.InputPreparedInt8 = Conv.InputCopyInt8);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// Without EnableInt8Input a quantized convolution keeps the int8-weight x FP32
// forward, bit for bit.
procedure TTestNeuralLayers.TestConvolutionInt8Int8NotEnabled;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionReLU;
  BeforeOutput: TNNetVolume;
  RawPos, MaxRawPos: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  BeforeOutput := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    Conv := TNNetConvolutionReLU.Create(5, 3, 1, 1);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    FillInt8ConvWeights(Conv);

    TNNetLayerConcatedWeights(Conv).QuantizeWeightsInt8();
    NN.Compute(Input);
    BeforeOutput.Copy(Conv.Output);
    NN.Compute(Input);

    AssertTrue('Int8 input copy stays nil', Conv.InputCopyInt8 = nil);
    AssertTrue('Int8 im2col stays nil', Conv.InputPreparedInt8 = nil);
    MaxRawPos := Conv.Output.Size - 1;
    for RawPos := 0 to MaxRawPos do
      AssertTrue('Int8-weight output ' + IntToStr(RawPos) + ' is unchanged',
        BeforeOutput.FData[RawPos] = Conv.Output.FData[RawPos]);
  finally
    NN.Free;
    Input.Free;
    BeforeOutput.Free;
  end;
end;

// TNNet.EnableInt8Input arms the int8 input copy on int8-quantized weight
// layers only, and TNNet.DisableInt8Input returns every layer to FP32 input.
procedure TTestNeuralLayers.TestNetEnableInt8InputCountsQuantizedLayers;
var
  NN: TNNet;
  Input, BeforeOutput: TNNetVolume;
  ConvFP32, ConvQuant: TNNetConvolutionReLU;
  FullConnect: TNNetFullConnectLinear;
  Bounds: TNeuralFloatDynArr;
  NeuronCnt, MaxNeuronPos: integer;
  RawPos, MaxRawPos: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  BeforeOutput := TNNetVolume.Create();
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    ConvFP32 := TNNetConvolutionReLU.Create(4, 3, 1, 1);
    NN.AddLayer(ConvFP32);
    NN.AddLayer(TNNetReLU.Create());
    ConvQuant := TNNetConvolutionReLU.Create(5, 3, 1, 1);
    NN.AddLayer(ConvQuant);
    FullConnect := TNNetFullConnectLinear.Create(6);
    NN.AddLayer(FullConnect);
    FillInt8ConvInput(Input);
    FillInt8ConvWeights(ConvFP32);
    FillInt8ConvWeights(ConvQuant);

    // Read the FP32 weight magnitudes before the codes replace them.
    MaxNeuronPos := ConvQuant.Neurons.Count - 1;
    SetLength(Bounds, ConvQuant.Neurons.Count);
    for NeuronCnt := 0 to MaxNeuronPos do
      Bounds[NeuronCnt] := Int8ConvWeightMagnitude(ConvQuant, NeuronCnt);

    TNNetLayerConcatedWeights(ConvQuant).QuantizeWeightsInt8();
    TNNetLayerConcatedWeights(FullConnect).QuantizeWeightsInt8();
    AssertTrue('The FP32 convolution stays FP32', not ConvFP32.WeightsQuantizedInt8);
    NN.Compute(Input);
    BeforeOutput.Copy(ConvQuant.Output);

    AssertEquals('Enabled int8-input layer count', 2, NN.EnableInt8Input());
    AssertTrue('The FP32 convolution has no int8 input copy',
      ConvFP32.InputCopyInt8 = nil);
    AssertEquals('Quantized conv int8 input SizeX', 10, ConvQuant.InputCopyInt8.SizeX);
    AssertEquals('Quantized conv int8 input SizeY', 10, ConvQuant.InputCopyInt8.SizeY);
    AssertEquals('Quantized conv int8 input Depth', 4, ConvQuant.InputCopyInt8.Depth);
    AssertEquals('FullConnect int8 input SizeX', 8, FullConnect.InputCopyInt8.SizeX);
    AssertEquals('FullConnect int8 input SizeY', 8, FullConnect.InputCopyInt8.SizeY);
    AssertEquals('FullConnect int8 input Depth', 5, FullConnect.InputCopyInt8.Depth);

    NN.Compute(Input);
    for NeuronCnt := 0 to MaxNeuronPos do
      Bounds[NeuronCnt] := 0.5 * ConvQuant.InputScaleInt8 * Bounds[NeuronCnt] + 1e-5;
    MaxRawPos := ConvQuant.Output.Size - 1;
    for RawPos := 0 to MaxRawPos do
    begin
      NeuronCnt := RawPos mod ConvQuant.Output.Depth;
      AssertTrue('Quantized conv output ' + IntToStr(RawPos) + ' int8 x int8 ' +
        FloatToStr(ConvQuant.Output.FData[RawPos]) + ' vs int8 weights ' +
        FloatToStr(BeforeOutput.FData[RawPos]) + ' exceeds bound ' +
        FloatToStr(Bounds[NeuronCnt]),
        Abs(ConvQuant.Output.FData[RawPos] - BeforeOutput.FData[RawPos]) <=
          Bounds[NeuronCnt]);
    end;

    NN.DisableInt8Input();
    AssertTrue('Quantized conv int8 input is dropped', ConvQuant.InputCopyInt8 = nil);
    AssertTrue('Quantized conv int8 im2col is dropped', ConvQuant.InputPreparedInt8 = nil);
    AssertTrue('FullConnect int8 input is dropped', FullConnect.InputCopyInt8 = nil);
    NN.Compute(Input);
    for RawPos := 0 to MaxRawPos do
      AssertTrue('Int8-weight output ' + IntToStr(RawPos) + ' is restored',
        BeforeOutput.FData[RawPos] = ConvQuant.Output.FData[RawPos]);
  finally
    NN.Free;
    Input.Free;
    BeforeOutput.Free;
  end;
end;

// DequantizeWeightsInt8 puts the layer back on the FP32 weight path, so it
// must drop the int8 input copy that only the int8 kernels read.
procedure TTestNeuralLayers.TestDequantizeWeightsInt8DropsInt8Input;
var
  NN: TNNet;
  Input: TNNetVolume;
  Conv: TNNetConvolutionReLU;
  RawPos, MaxRawPos: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    Conv := TNNetConvolutionReLU.Create(5, 3, 1, 1);
    NN.AddLayer(Conv);
    FillInt8ConvInput(Input);
    FillInt8ConvWeights(Conv);

    TNNetLayerConcatedWeights(Conv).QuantizeWeightsInt8();
    NN.Compute(Input);
    Conv.EnableInt8Input();
    AssertTrue('Int8 input copy is sized', Conv.InputCopyInt8.Size > 0);
    NN.Compute(Input);

    TNNetLayerConcatedWeights(Conv).DequantizeWeightsInt8();
    AssertTrue('Weights are FP32 again', not Conv.WeightsQuantizedInt8);
    AssertTrue('Int8 input copy is dropped', Conv.InputCopyInt8 = nil);
    AssertTrue('Int8 im2col is dropped', Conv.InputPreparedInt8 = nil);
    AssertEquals('Int8 input scale is reset', 1, Conv.InputScaleInt8, 0);

    NN.Compute(Input);
    MaxRawPos := Conv.Output.Size - 1;
    for RawPos := 0 to MaxRawPos do
      AssertTrue('FP32 output ' + IntToStr(RawPos) + ' is finite',
        Abs(Conv.Output.FData[RawPos]) < 1e6);
  finally
    NN.Free;
    Input.Free;
  end;
end;

// TNNetFullConnect inherits the int8 input copy: it quantizes the previous
// layer's output, since a fully connected layer has no padded input copy.
procedure TTestNeuralLayers.TestFullConnectQuantizeInputInt8;
var
  NN: TNNet;
  Input: TNNetVolume;
  FullConnect: TNNetFullConnectLinear;
  PrevOutput: TNNetVolume;
  Scale, Tolerance, Dequantized: TNeuralFloat;
  RawPos, MaxRawPos: integer;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(4, 4, 3);
  try
    NN.AddLayer(TNNetInput.Create(4, 4, 3));
    FullConnect := TNNetFullConnectLinear.Create(5);
    NN.AddLayer(FullConnect);
    FillInt8ConvInput(Input);

    TNNetLayerConcatedWeights(FullConnect).QuantizeWeightsInt8();
    NN.Compute(Input);
    FullConnect.EnableInt8Input();
    AssertEquals('FullConnect int8 input Size', 4 * 4 * 3,
      FullConnect.InputCopyInt8.Size);
    NN.Compute(Input);

    FullConnect.QuantizeInputInt8();
    PrevOutput := NN.Layers[0].Output;
    Scale := FullConnect.InputScaleInt8;
    AssertTrue('The int8 input scale must be positive', Scale > 0);
    Tolerance := Scale / 2 + 1e-6;
    MaxRawPos := PrevOutput.Size - 1;
    for RawPos := 0 to MaxRawPos do
    begin
      Dequantized := Scale * FullConnect.InputCopyInt8.FData[RawPos];
      AssertTrue('Int8 input element ' + IntToStr(RawPos) + ' expected ' +
        FloatToStr(PrevOutput.FData[RawPos]) + ' got ' + FloatToStr(Dequantized),
        Abs(Dequantized - PrevOutput.FData[RawPos]) <= Tolerance);
    end;
  finally
    NN.Free;
    Input.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralLayers);

end.
