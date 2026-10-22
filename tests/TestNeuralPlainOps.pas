unit TestNeuralPlainOps;
(*
Tests for neuralplainops.pas: each plain-volume op against the TNNet layer it
replaces, on small random cases, with the layers in the inference settings
Kokoro uses (SetTrainable(false)). Each threaded op also runs serial and threaded
(small tiles, a 3-thread pool) and the two results must be identical.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralvolume, neuralnetwork, neuralplainops, neuralpretrained, neuralthread;

type
  TTestNeuralPlainOps = class(TTestCase)
  private
    FSerial, FThreaded: TNNetPlainOps;
  protected
    procedure SetUp; override;
    procedure TearDown; override;
  published
    // Conv1D vs TNNetConv1DLinear: Kokoro shapes (k3/k7/k11 dilated, k12 s6,
    // the 1->1 k3 s2 curve conv), Cin <> Cout.
    procedure TestConv1DMatchesLayer;
    // PaddingRight only sets the length: an asymmetric conv is a prefix of the
    // symmetric one.
    procedure TestConv1DAsymmetricPaddingIsPrefix;
    // PointwiseConv vs TNNetPointwiseConvLinear, with and without bias.
    procedure TestPointwiseConvMatchesLayer;
    // ConvTranspose1D vs a dense TNNetDeconv1DLinear (Kokoro ups k20 s10 p5,
    // k12 s6 p3) on its CPU GEMM path.
    procedure TestConvTranspose1DMatchesLayer;
    // DepthwiseConvTranspose1D vs the Kokoro AdainResBlk1d pool (k3 s2 p1 op1,
    // groups = channels).
    procedure TestDepthwiseConvTranspose1DMatchesLayer;
    // UpsampleNearest x2 vs TNNetDeMaxPool(2, 0, 1).
    procedure TestUpsampleNearestMatchesLayer;
    // ReflectPad vs TNNetPadXY with pmReflect, left-only and both sides.
    procedure TestReflectPadMatchesLayer;
    // InstanceNorm + affine vs TNNetInstanceNorm with per-channel gamma/beta.
    procedure TestInstanceNormMatchesLayer;
    // ComputeStyleChannelAffine vs WriteStyleIntoChannelAffine (with and
    // without the norm's own affine).
    procedure TestStyleChannelAffineMatchesWriter;
    // Snake vs a per-channel TNNetSnake.
    procedure TestSnakeMatchesLayer;
    // NormSnakeConv1D (several chunk sizes, write and accumulate, serial and
    // threaded) vs InstanceNorm, Snake and Conv1D run one after the other.
    procedure TestNormSnakeConv1DMatchesSteps;
    // MaxThreads 1 and 2 on a 3-thread pool: Conv1D equals the serial result.
    procedure TestMaxThreadsCapsTheJob;
    // LeakyReLU vs TNNetLeakyReLU.
    procedure TestLeakyReLUMatchesLayer;
    // ConcatChannels vs TNNetDeepConcat (widths 3, 1, 2).
    procedure TestConcatChannelsMatchesLayer;
    // BiLSTM vs TNNet.AddBidirectionalLSTM(H, 1, true, true), and torch-layout
    // weights loaded with LoadTorchLSTM give the same result.
    procedure TestBiLSTMMatchesLayer;
    // TNNetVolume.DotProductsPacked vs a double reference on odd shapes (rows,
    // positions and lengths not multiples of the 16 x 6 block), strides, bias.
    procedure TestDotProductsPackedMatchesReference;
    // Conv1D, PointwiseConv and ConvTranspose1D vs a double reference at
    // generator-like sizes (128 / 256 channels, kernels 3 / 7 / 11, dilations).
    procedure TestConvsMatchReferenceAtGeneratorSizes;
  end;

implementation

const
  // Under AVX the plain ops (TNNetVolume.DotProductsPacked) and the layers
  // order and round their sums differently (a few ulp); scalar builds agree.
  {$IF DEFINED(AVX) or DEFINED(AVX2) or DEFINED(AVX512)}
  csDotKernelTolerance = 1e-5;
  {$ELSE}
  csDotKernelTolerance = 0;
  {$ENDIF}

procedure FillRandom(V: TNNetVolume; Amplitude: TNeuralFloat = 1.0);
var
  ElementPos: integer;
begin
  for ElementPos := 0 to V.Size - 1 do
    V.FData[ElementPos] := (2 * Random - 1) * Amplitude;
end;

procedure RandomizeLayerWeights(Layer: TNNetLayer);
var
  NeuronPos: integer;
begin
  for NeuronPos := 0 to Layer.Neurons.Count - 1 do
  begin
    FillRandom(Layer.Neurons[NeuronPos].Weights, 0.5);
    Layer.Neurons[NeuronPos].BiasWeight := (2 * Random - 1) * 0.5;
  end;
  Layer.FlushWeightCache();
end;

function MaxAbsDiff(A, B: TNNetVolume): TNeuralFloat;
var
  ElementPos: integer;
begin
  if (A.SizeX <> B.SizeX) or (A.SizeY <> B.SizeY) or (A.Depth <> B.Depth) then
    raise Exception.Create('MaxAbsDiff: shapes ' + IntToStr(A.SizeX) + 'x' +
      IntToStr(A.SizeY) + 'x' + IntToStr(A.Depth) + ' and ' +
      IntToStr(B.SizeX) + 'x' + IntToStr(B.SizeY) + 'x' + IntToStr(B.Depth) +
      ' differ.');
  Result := 0;
  for ElementPos := 0 to A.Size - 1 do
    Result := Max(Result, Abs(A.FData[ElementPos] - B.FData[ElementPos]));
end;

// Per element: |Actual - Reference| <= 1e-5 * Magnitude + 1e-6, where
// Magnitude is the sum of |a * b| of that element's dot product.
procedure AssertCloseToReference(const Name: string;
  const Actual, Reference, Magnitude: array of double);
var
  ElementPos: integer;
begin
  TAssert.AssertEquals(Name + ' element count', Length(Reference), Length(Actual));
  for ElementPos := 0 to High(Reference) do
    if Abs(Actual[ElementPos] - Reference[ElementPos]) >
       1e-5 * Magnitude[ElementPos] + 1e-6 then
      raise EAssertionFailedError.Create(Name + ' element ' +
        IntToStr(ElementPos) + ': ' + FloatToStr(Actual[ElementPos]) +
        ' vs reference ' + FloatToStr(Reference[ElementPos]));
end;

procedure TTestNeuralPlainOps.SetUp;
begin
  inherited SetUp;
  RandSeed := 20261017;
  // A 3-thread pool, so the threaded ops split their work on any machine.
  NeuralThreadListFree();
  NeuralThreadListCreate(3);
  FSerial := TNNetPlainOps.Create({pThreaded=}false, {pTilePositions=}5);
  FThreaded := TNNetPlainOps.Create({pThreaded=}true, {pTilePositions=}3);
  FThreaded.MinPositionsToThread := 1;
end;

procedure TTestNeuralPlainOps.TearDown;
begin
  FThreaded.Free;
  FSerial.Free;
  NeuralThreadListFree();
  CreateNeuralThreadListIfRequired();
  inherited TearDown;
end;

procedure TTestNeuralPlainOps.TestConv1DMatchesLayer;
const
  // Cin, Cout, K, padding, stride, dilation, length.
  csCases: array[0..6, 0..6] of integer = (
    (5, 7, 3, 1, 1, 1, 20),
    (4, 6, 7, 3, 1, 1, 23),
    (6, 6, 11, 25, 1, 5, 40),
    (6, 6, 7, 9, 1, 3, 30),
    (3, 5, 12, 3, 6, 1, 61),
    (1, 1, 3, 1, 2, 1, 20),
    (8, 3, 3, 1, 2, 1, 17));
var
  CasePos: integer;
  NN: TNNet;
  Conv: TNNetLayer;
  Input, SerialOut, ThreadedOut: TNNetVolume;
  Weights: TNNetPlainConvWeights;
  Diff: TNeuralFloat;
begin
  Input := TNNetVolume.Create();
  SerialOut := TNNetVolume.Create();
  ThreadedOut := TNNetVolume.Create();
  Weights := TNNetPlainConvWeights.Create();
  try
    for CasePos := 0 to High(csCases) do
    begin
      NN := TNNet.Create();
      try
        NN.AddLayer(TNNetInput.Create(csCases[CasePos, 6], 1,
          csCases[CasePos, 0]));
        Conv := NN.AddLayer(TNNetConv1DLinear.Create(csCases[CasePos, 1],
          csCases[CasePos, 2], csCases[CasePos, 3], csCases[CasePos, 4],
          csCases[CasePos, 5]));
        NN.SetTrainable(false);
        RandomizeLayerWeights(Conv);
        Input.ReSize(csCases[CasePos, 6], 1, csCases[CasePos, 0]);
        FillRandom(Input);
        NN.Compute(Input);
        Weights.LoadConv1DLayer(Conv, true);
        FSerial.Conv1D(Input, SerialOut, Weights, csCases[CasePos, 4],
          csCases[CasePos, 5], csCases[CasePos, 3]);
        FThreaded.Conv1D(Input, ThreadedOut, Weights, csCases[CasePos, 4],
          csCases[CasePos, 5], csCases[CasePos, 3]);
        Diff := MaxAbsDiff(SerialOut, Conv.Output);
        AssertTrue('case ' + IntToStr(CasePos) + ' max diff ' +
          FloatToStr(Diff), Diff <= csDotKernelTolerance);
        AssertEquals('case ' + IntToStr(CasePos) + ' serial vs threaded', 0,
          MaxAbsDiff(SerialOut, ThreadedOut), 0);
      finally
        NN.Free;
      end;
    end;
  finally
    Weights.Free;
    ThreadedOut.Free;
    SerialOut.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestConv1DAsymmetricPaddingIsPrefix;
var
  Input, Symmetric, Asymmetric, TorchWeight, TorchBias: TNNetVolume;
  Weights: TNNetPlainConvWeights;
  Position, ElementPos: integer;
begin
  Input := TNNetVolume.Create(15, 1, 4);
  Symmetric := TNNetVolume.Create();
  Asymmetric := TNNetVolume.Create();
  TorchWeight := TNNetVolume.Create(5 * 4 * 3, 1, 1);
  TorchBias := TNNetVolume.Create(5, 1, 1);
  Weights := TNNetPlainConvWeights.Create();
  try
    FillRandom(Input);
    FillRandom(TorchWeight);
    FillRandom(TorchBias);
    Weights.LoadTorchConv1D(TorchWeight, TorchBias, 5, 4, 3);
    FSerial.Conv1D(Input, Symmetric, Weights, 1, 2, 2, 2);
    FSerial.Conv1D(Input, Asymmetric, Weights, 1, 2, 2, 0);
    AssertEquals('symmetric length', 15, Symmetric.SizeX);
    AssertEquals('asymmetric length', 13, Asymmetric.SizeX);
    for Position := 0 to Asymmetric.SizeX - 1 do
      for ElementPos := 0 to 4 do
        AssertEquals(Symmetric.FData[Position * 5 + ElementPos],
          Asymmetric.FData[Position * 5 + ElementPos], 0);
  finally
    Weights.Free;
    TorchBias.Free;
    TorchWeight.Free;
    Asymmetric.Free;
    Symmetric.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestPointwiseConvMatchesLayer;
var
  SuppressBias: integer;
  NN: TNNet;
  Conv: TNNetLayer;
  Input, SerialOut, ThreadedOut: TNNetVolume;
  Weights: TNNetPlainConvWeights;
  Diff: TNeuralFloat;
begin
  Input := TNNetVolume.Create(19, 1, 6);
  SerialOut := TNNetVolume.Create();
  ThreadedOut := TNNetVolume.Create();
  Weights := TNNetPlainConvWeights.Create();
  try
    for SuppressBias := 0 to 1 do
    begin
      NN := TNNet.Create();
      try
        NN.AddLayer(TNNetInput.Create(19, 1, 6));
        Conv := NN.AddLayer(TNNetPointwiseConvLinear.Create(9, SuppressBias));
        NN.SetTrainable(false);
        RandomizeLayerWeights(Conv);
        FillRandom(Input);
        NN.Compute(Input);
        Weights.LoadConv1DLayer(Conv, SuppressBias = 0);
        FSerial.PointwiseConv(Input, SerialOut, Weights);
        FThreaded.PointwiseConv(Input, ThreadedOut, Weights);
        Diff := MaxAbsDiff(SerialOut, Conv.Output);
        AssertTrue('suppress bias ' + IntToStr(SuppressBias) + ' max diff ' +
          FloatToStr(Diff), Diff <= csDotKernelTolerance);
        AssertEquals('serial vs threaded', 0,
          MaxAbsDiff(SerialOut, ThreadedOut), 0);
      finally
        NN.Free;
      end;
    end;
  finally
    Weights.Free;
    ThreadedOut.Free;
    SerialOut.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestConvTranspose1DMatchesLayer;
const
  // Cin, Cout, K, stride, padding, length.
  csCases: array[0..2, 0..5] of integer = (
    (6, 5, 20, 10, 5, 7),
    (5, 4, 12, 6, 3, 9),
    (3, 2, 4, 2, 1, 6));
var
  CasePos: integer;
  NN: TNNet;
  Deconv: TNNetLayer;
  Input, SerialOut, ThreadedOut: TNNetVolume;
  Weights: TNNetPlainConvWeights;
  Diff: TNeuralFloat;
begin
  Input := TNNetVolume.Create();
  SerialOut := TNNetVolume.Create();
  ThreadedOut := TNNetVolume.Create();
  Weights := TNNetPlainConvWeights.Create();
  try
    for CasePos := 0 to High(csCases) do
    begin
      NN := TNNet.Create();
      try
        NN.AddLayer(TNNetInput.Create(csCases[CasePos, 5], 1,
          csCases[CasePos, 0]));
        Deconv := NN.AddLayer(TNNetDeconv1DLinear.Create(csCases[CasePos, 1],
          csCases[CasePos, 2], csCases[CasePos, 3], csCases[CasePos, 4]));
        // The Kokoro generator settings: the ups run the CPU GEMM.
        NN.SetTrainable(false);
        Deconv.SetTrainable(false, {pLowMemory=}false);
        RandomizeLayerWeights(Deconv);
        Input.ReSize(csCases[CasePos, 5], 1, csCases[CasePos, 0]);
        FillRandom(Input);
        NN.Compute(Input);
        Weights.LoadDeconv1DLayer(Deconv, true);
        FSerial.ConvTranspose1D(Input, SerialOut, Weights, csCases[CasePos, 3],
          csCases[CasePos, 4], 0);
        FThreaded.ConvTranspose1D(Input, ThreadedOut, Weights,
          csCases[CasePos, 3], csCases[CasePos, 4], 0);
        Diff := MaxAbsDiff(SerialOut, Deconv.Output);
        AssertTrue('case ' + IntToStr(CasePos) + ' max diff ' +
          FloatToStr(Diff), Diff <= csDotKernelTolerance);
        AssertEquals('case ' + IntToStr(CasePos) + ' serial vs threaded', 0,
          MaxAbsDiff(SerialOut, ThreadedOut), 0);
      finally
        NN.Free;
      end;
    end;
  finally
    Weights.Free;
    ThreadedOut.Free;
    SerialOut.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestDepthwiseConvTranspose1DMatchesLayer;
var
  NN: TNNet;
  Pool: TNNetLayer;
  Input, SerialOut, ThreadedOut: TNNetVolume;
  Weights: TNNetPlainConvWeights;
  Diff: TNeuralFloat;
begin
  Input := TNNetVolume.Create(11, 1, 6);
  SerialOut := TNNetVolume.Create();
  ThreadedOut := TNNetVolume.Create();
  Weights := TNNetPlainConvWeights.Create();
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(11, 1, 6));
    Pool := NN.AddLayer(TNNetDeconv1DLinear.Create(6, 3, 2, 1, 1, 6));
    NN.SetTrainable(false);
    RandomizeLayerWeights(Pool);
    FillRandom(Input);
    NN.Compute(Input);
    Weights.LoadDepthwiseDeconv1DLayer(Pool, true);
    FSerial.DepthwiseConvTranspose1D(Input, SerialOut, Weights, 2, 1, 1);
    FThreaded.DepthwiseConvTranspose1D(Input, ThreadedOut, Weights, 2, 1, 1);
    AssertEquals('length', 22, SerialOut.SizeX);
    Diff := MaxAbsDiff(SerialOut, Pool.Output);
    AssertTrue('max diff ' + FloatToStr(Diff), Diff <= 1e-6);
    AssertEquals('serial vs threaded', 0,
      MaxAbsDiff(SerialOut, ThreadedOut), 0);
  finally
    NN.Free;
    Weights.Free;
    ThreadedOut.Free;
    SerialOut.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestUpsampleNearestMatchesLayer;
var
  NN: TNNet;
  Upsample: TNNetLayer;
  Input, Output: TNNetVolume;
begin
  Input := TNNetVolume.Create(7, 1, 5);
  Output := TNNetVolume.Create();
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(7, 1, 5));
    Upsample := NN.AddLayer(TNNetDeMaxPool.Create(2, 0, 1));
    NN.SetTrainable(false);
    FillRandom(Input);
    NN.Compute(Input);
    FSerial.UpsampleNearest(Input, Output, 2);
    AssertEquals(0, MaxAbsDiff(Output, Upsample.Output), 0);
  finally
    NN.Free;
    Output.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestReflectPadMatchesLayer;
const
  // left, right
  csCases: array[0..2, 0..1] of integer = ((1, 0), (3, 2), (0, 4));
var
  CasePos: integer;
  NN: TNNet;
  Pad: TNNetLayer;
  Input, Output: TNNetVolume;
begin
  Input := TNNetVolume.Create(6, 1, 3);
  Output := TNNetVolume.Create();
  try
    for CasePos := 0 to High(csCases) do
    begin
      NN := TNNet.Create();
      try
        NN.AddLayer(TNNetInput.Create(6, 1, 3));
        Pad := NN.AddLayer(TNNetPadXY.Create(csCases[CasePos, 0], 0,
          pmReflect, csCases[CasePos, 1]));
        NN.SetTrainable(false);
        FillRandom(Input);
        NN.Compute(Input);
        FSerial.ReflectPad(Input, Output, csCases[CasePos, 0],
          csCases[CasePos, 1]);
        AssertEquals('case ' + IntToStr(CasePos), 0,
          MaxAbsDiff(Output, Pad.Output), 0);
      finally
        NN.Free;
      end;
    end;
  finally
    Output.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestInstanceNormMatchesLayer;
var
  NN: TNNet;
  Norm: TNNetLayer;
  Input, Plain: TNNetVolume;
  Diff: TNeuralFloat;
begin
  Input := TNNetVolume.Create(17, 1, 6);
  Plain := TNNetVolume.Create();
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(17, 1, 6));
    Norm := NN.AddLayer(TNNetInstanceNorm.Create());
    NN.SetTrainable(false);
    FillRandom(Norm.Neurons[0].Weights);
    FillRandom(Norm.Neurons[1].Weights);
    Norm.FlushWeightCache();
    FillRandom(Input, 3);
    NN.Compute(Input);
    Plain.Copy(Input);
    FSerial.InstanceNorm(Plain, Norm.Neurons[0].Weights,
      Norm.Neurons[1].Weights);
    Diff := MaxAbsDiff(Plain, Norm.Output);
    AssertTrue('max diff ' + FloatToStr(Diff), Diff <= 1e-5);
  finally
    NN.Free;
    Plain.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestStyleChannelAffineMatchesWriter;
const
  csChannels = 5;
  csStyleDim = 4;
var
  NN: TNNet;
  Norm: TNNetLayer;
  FcWeight, FcBias, NormWeight, NormBias, Scale, Shift: TNNetVolume;
  Style: array[0..csStyleDim - 1] of TNeuralFloat;
  StylePos, WithNormAffine: integer;
begin
  FcWeight := TNNetVolume.Create(2 * csChannels * csStyleDim, 1, 1);
  FcBias := TNNetVolume.Create(2 * csChannels, 1, 1);
  NormWeight := TNNetVolume.Create(csChannels, 1, 1);
  NormBias := TNNetVolume.Create(csChannels, 1, 1);
  Scale := TNNetVolume.Create();
  Shift := TNNetVolume.Create();
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(3, 1, csChannels));
    Norm := NN.AddLayer(TNNetInstanceNorm.Create());
    FillRandom(FcWeight);
    FillRandom(FcBias);
    FillRandom(NormWeight);
    FillRandom(NormBias);
    for StylePos := 0 to csStyleDim - 1 do Style[StylePos] := 2 * Random - 1;
    for WithNormAffine := 0 to 1 do
    begin
      if WithNormAffine = 1 then
      begin
        WriteStyleIntoChannelAffine(Norm, FcWeight, FcBias, Style, NormWeight,
          NormBias);
        ComputeStyleChannelAffine(FcWeight, FcBias, Style, Scale, Shift,
          NormWeight, NormBias);
      end
      else
      begin
        WriteStyleIntoChannelAffine(Norm, FcWeight, FcBias, Style);
        ComputeStyleChannelAffine(FcWeight, FcBias, Style, Scale, Shift);
      end;
      AssertEquals('scale', 0, MaxAbsDiff(Scale, Norm.Neurons[0].Weights), 0);
      AssertEquals('shift', 0, MaxAbsDiff(Shift, Norm.Neurons[1].Weights), 0);
    end;
  finally
    NN.Free;
    Shift.Free;
    Scale.Free;
    NormBias.Free;
    NormWeight.Free;
    FcBias.Free;
    FcWeight.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestSnakeMatchesLayer;
var
  NN: TNNet;
  SnakeLayer: TNNetLayer;
  Input, SerialOut, ThreadedOut, Alpha: TNNetVolume;
  ChannelPos: integer;
  Diff: TNeuralFloat;
begin
  Input := TNNetVolume.Create(23, 1, 6);
  SerialOut := TNNetVolume.Create();
  ThreadedOut := TNNetVolume.Create();
  Alpha := TNNetVolume.Create(6, 1, 1);
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(23, 1, 6));
    SnakeLayer := NN.AddLayer(TNNetSnake.Create(1, true));
    NN.SetTrainable(false);
    for ChannelPos := 0 to 5 do Alpha.FData[ChannelPos] := 0.5 + Random;
    SnakeLayer.Neurons[0].Weights.Copy(Alpha);
    SnakeLayer.FlushWeightCache();
    FillRandom(Input, 3);
    NN.Compute(Input);
    SerialOut.Copy(Input);
    ThreadedOut.Copy(Input);
    FSerial.Snake(SerialOut, Alpha);
    FThreaded.Snake(ThreadedOut, Alpha);
    Diff := MaxAbsDiff(SerialOut, SnakeLayer.Output);
    // The layer runs Sin over whole blocks, these rows through its scalar tail.
    AssertTrue('max diff ' + FloatToStr(Diff), Diff < 1e-5);
    AssertEquals('serial vs threaded', 0,
      MaxAbsDiff(SerialOut, ThreadedOut), 0);
  finally
    NN.Free;
    Alpha.Free;
    ThreadedOut.Free;
    SerialOut.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestNormSnakeConv1DMatchesSteps;
const
  csLength = 37;
  csInChannels = 5;
  csOutChannels = 4;
  csKernelSize = 7;
  csDilation = 3;
  csChunkSizes: array[0..3] of integer = (1, 8, 36, 64);
var
  Input, Transformed, Expected, Scale, Shift, Alpha, TorchWeight, TorchBias,
    Output, Accumulated, Start: TNNetVolume;
  Weights: TNNetPlainConvWeights;
  ChunkPos, ChannelPos: integer;
  Ops: TNNetPlainOps;
  OpsPos: integer;
  Diff: TNeuralFloat;
begin
  Input := TNNetVolume.Create(csLength, 1, csInChannels);
  Transformed := TNNetVolume.Create();
  Expected := TNNetVolume.Create();
  Scale := TNNetVolume.Create(csInChannels, 1, 1);
  Shift := TNNetVolume.Create(csInChannels, 1, 1);
  Alpha := TNNetVolume.Create(csInChannels, 1, 1);
  TorchWeight := TNNetVolume.Create(csOutChannels * csInChannels * csKernelSize,
    1, 1);
  TorchBias := TNNetVolume.Create(csOutChannels, 1, 1);
  Output := TNNetVolume.Create();
  Accumulated := TNNetVolume.Create();
  Start := TNNetVolume.Create(csLength, 1, csOutChannels);
  Weights := TNNetPlainConvWeights.Create();
  try
    FillRandom(Input, 3);
    FillRandom(Scale);
    FillRandom(Shift);
    for ChannelPos := 0 to csInChannels - 1 do
      Alpha.FData[ChannelPos] := 0.5 + Random;
    FillRandom(TorchWeight, 0.5);
    FillRandom(TorchBias, 0.5);
    FillRandom(Start);
    Weights.LoadTorchConv1D(TorchWeight, TorchBias, csOutChannels,
      csInChannels, csKernelSize);
    Transformed.Copy(Input);
    FSerial.InstanceNorm(Transformed, Scale, Shift);
    FSerial.Snake(Transformed, Alpha);
    FSerial.Conv1D(Transformed, Expected, Weights, 1, csDilation,
      csDilation * (csKernelSize - 1) div 2);
    for OpsPos := 0 to 1 do
    begin
      if OpsPos = 0 then Ops := FSerial else Ops := FThreaded;
      for ChunkPos := 0 to High(csChunkSizes) do
      begin
        Ops.PrepareInstanceNorm(Input, Scale, Shift);
        Ops.NormSnakeConv1D(Input, Output, Weights, Alpha, csDilation,
          {Accumulate=}false, csChunkSizes[ChunkPos]);
        Diff := MaxAbsDiff(Output, Expected);
        AssertTrue('ops ' + IntToStr(OpsPos) + ' chunk ' +
          IntToStr(csChunkSizes[ChunkPos]) + ' max diff ' + FloatToStr(Diff),
          Diff <= 1e-5);
        Accumulated.Copy(Start);
        Ops.NormSnakeConv1D(Input, Accumulated, Weights, Alpha, csDilation,
          {Accumulate=}true, csChunkSizes[ChunkPos]);
        Accumulated.Sub(Start);
        Diff := MaxAbsDiff(Accumulated, Expected);
        AssertTrue('accumulate, ops ' + IntToStr(OpsPos) + ' chunk ' +
          IntToStr(csChunkSizes[ChunkPos]) + ' max diff ' + FloatToStr(Diff),
          Diff <= 1e-5);
      end;
    end;
  finally
    Weights.Free;
    Start.Free;
    Accumulated.Free;
    Output.Free;
    TorchBias.Free;
    TorchWeight.Free;
    Alpha.Free;
    Shift.Free;
    Scale.Free;
    Expected.Free;
    Transformed.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestMaxThreadsCapsTheJob;
const
  csCaps: array[0..1] of integer = (1, 2);
var
  Input, SerialOut, CappedOut, TorchWeight: TNNetVolume;
  Weights: TNNetPlainConvWeights;
  CapPos: integer;
begin
  Input := TNNetVolume.Create(29, 1, 6);
  SerialOut := TNNetVolume.Create();
  CappedOut := TNNetVolume.Create();
  TorchWeight := TNNetVolume.Create(5 * 6 * 3, 1, 1);
  Weights := TNNetPlainConvWeights.Create();
  try
    FillRandom(Input);
    FillRandom(TorchWeight, 0.5);
    Weights.LoadTorchConv1D(TorchWeight, nil, 5, 6, 3);
    FSerial.Conv1D(Input, SerialOut, Weights, 1, 1, 1);
    for CapPos := 0 to High(csCaps) do
    begin
      FThreaded.MaxThreads := csCaps[CapPos];
      FThreaded.Conv1D(Input, CappedOut, Weights, 1, 1, 1);
      AssertEquals('MaxThreads ' + IntToStr(csCaps[CapPos]), 0,
        MaxAbsDiff(SerialOut, CappedOut), 0);
    end;
  finally
    FThreaded.MaxThreads := 0;
    Weights.Free;
    TorchWeight.Free;
    CappedOut.Free;
    SerialOut.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestLeakyReLUMatchesLayer;
var
  NN: TNNet;
  Leaky: TNNetLayer;
  Input, Output: TNNetVolume;
begin
  Input := TNNetVolume.Create(13, 1, 7);
  Output := TNNetVolume.Create();
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(13, 1, 7));
    Leaky := NN.AddLayer(TNNetLeakyReLU.Create(0.2));
    NN.SetTrainable(false);
    FillRandom(Input);
    NN.Compute(Input);
    FSerial.LeakyReLU(Input, Output, 0.2);
    AssertEquals('out of place', 0, MaxAbsDiff(Output, Leaky.Output), 0);
    Output.Copy(Input);
    FSerial.LeakyReLU(Output, Output, 0.2);
    AssertEquals('in place', 0, MaxAbsDiff(Output, Leaky.Output), 0);
  finally
    NN.Free;
    Output.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestConcatChannelsMatchesLayer;
var
  NN: TNNet;
  InputA, InputB, InputC, Concat: TNNetLayer;
  SourceA, SourceB, SourceC, Output: TNNetVolume;
begin
  SourceA := TNNetVolume.Create(6, 1, 3);
  SourceB := TNNetVolume.Create(6, 1, 1);
  SourceC := TNNetVolume.Create(6, 1, 2);
  Output := TNNetVolume.Create();
  NN := TNNet.Create();
  try
    InputA := NN.AddLayer(TNNetInput.Create(6, 1, 3));
    InputB := NN.AddLayer(TNNetInput.Create(6, 1, 1));
    InputC := NN.AddLayer(TNNetInput.Create(6, 1, 2));
    Concat := NN.AddLayer(TNNetDeepConcat.Create([InputA, InputB, InputC]));
    NN.SetTrainable(false);
    FillRandom(SourceA);
    FillRandom(SourceB);
    FillRandom(SourceC);
    NN.Compute([SourceA, SourceB, SourceC]);
    FSerial.ConcatChannels([SourceA, SourceB, SourceC], Output);
    AssertEquals(0, MaxAbsDiff(Output, Concat.Output), 0);
  finally
    NN.Free;
    Output.Free;
    SourceC.Free;
    SourceB.Free;
    SourceA.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestBiLSTMMatchesLayer;
const
  csSteps = 9;
  csInChannels = 7;
  csHidden = 5;
var
  NN: TNNet;
  LSTMOutput: TNNetLayer;
  Cells: array[0..1] of TNNetLayer;
  Input, SerialOut, ThreadedOut, TorchOut: TNNetVolume;
  WeightIH, WeightHH, BiasIH, BiasHH: TNNetVolume;
  Weights, TorchWeights: array[0..1] of TNNetPlainLSTMWeights;
  LayerPos, CellCount, Direction, GatePos: integer;
  Diff: TNeuralFloat;
begin
  Input := TNNetVolume.Create(csSteps, 1, csInChannels);
  SerialOut := TNNetVolume.Create();
  ThreadedOut := TNNetVolume.Create();
  TorchOut := TNNetVolume.Create();
  WeightIH := TNNetVolume.Create();
  WeightHH := TNNetVolume.Create();
  BiasIH := TNNetVolume.Create();
  BiasHH := TNNetVolume.Create();
  for Direction := 0 to 1 do
  begin
    Weights[Direction] := TNNetPlainLSTMWeights.Create();
    TorchWeights[Direction] := TNNetPlainLSTMWeights.Create();
  end;
  NN := TNNet.Create();
  try
    NN.AddLayer(TNNetInput.Create(csSteps, 1, csInChannels));
    LSTMOutput := NN.AddBidirectionalLSTM(csHidden, 1, true, true);
    NN.SetTrainable(false);
    CellCount := 0;
    for LayerPos := 0 to NN.CountLayers() - 1 do
      if NN.Layers[LayerPos] is TNNetLSTMCell then
      begin
        Cells[CellCount] := NN.Layers[LayerPos];
        Inc(CellCount);
      end;
    AssertEquals('cell count', 2, CellCount);
    for Direction := 0 to 1 do
    begin
      RandomizeLayerWeights(Cells[Direction]);
      Weights[Direction].LoadLSTMCell(Cells[Direction]);
      // torch layout: the same rows, the bias split into bias_ih + bias_hh.
      WeightIH.Copy(Weights[Direction].InputRows);
      WeightHH.Copy(Weights[Direction].RecurrentRows);
      BiasIH.Copy(Weights[Direction].Bias);
      BiasHH.Copy(Weights[Direction].Bias);
      for GatePos := 0 to BiasIH.Size - 1 do
      begin
        BiasHH.FData[GatePos] := 0.25 * BiasIH.FData[GatePos];
        BiasIH.FData[GatePos] := BiasIH.FData[GatePos] - BiasHH.FData[GatePos];
      end;
      TorchWeights[Direction].LoadTorchLSTM(WeightIH, WeightHH, BiasIH, BiasHH);
    end;
    FillRandom(Input);
    NN.Compute(Input);
    FSerial.BiLSTM(Input, SerialOut, Weights[0], Weights[1]);
    FThreaded.BiLSTM(Input, ThreadedOut, Weights[0], Weights[1]);
    FSerial.BiLSTM(Input, TorchOut, TorchWeights[0], TorchWeights[1]);
    Diff := MaxAbsDiff(SerialOut, LSTMOutput.Output);
    AssertTrue('max diff ' + FloatToStr(Diff), Diff <= 1e-5);
    AssertEquals('serial vs threaded', 0,
      MaxAbsDiff(SerialOut, ThreadedOut), 0);
    Diff := MaxAbsDiff(TorchOut, SerialOut);
    AssertTrue('torch layout max diff ' + FloatToStr(Diff), Diff <= 1e-6);
  finally
    NN.Free;
    for Direction := 0 to 1 do
    begin
      TorchWeights[Direction].Free;
      Weights[Direction].Free;
    end;
    BiasHH.Free;
    BiasIH.Free;
    WeightHH.Free;
    WeightIH.Free;
    TorchOut.Free;
    ThreadedOut.Free;
    SerialOut.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestDotProductsPackedMatchesReference;
const
  // NumAs, NumBs, VectorSize, extra B stride, extra C stride, with bias.
  csCases: array[0..8, 0..5] of integer = (
    (1, 1, 1, 0, 0, 1),
    (16, 6, 4, 0, 0, 0),
    (5, 7, 3, 2, 3, 1),
    (22, 13, 17, 1, 0, 1),
    (37, 50, 33, 0, 5, 0),
    (128, 49, 384, 0, 0, 1),
    (32, 96, 64, 3, 0, 1),
    (17, 5, 129, 0, 1, 1),
    (48, 61, 7, 0, 0, 0));
  csSentinel = 1234.5;
var
  CasePos, NumAs, NumBs, VectorSize, BStride, CStride: integer;
  RowPos, ColPos, ElementPos, OutPos: integer;
  A, PackedA, B, C, Bias: TNNetVolume;
  BiasPtr: TNeuralFloatArrPtr;
  Actual, Reference, Magnitude: array of double;
  Sum, SumAbs, Product: double;
  {$IF not (DEFINED(AVX) or DEFINED(AVX2) or DEFINED(AVX512))}
  ScalarDot: TNeuralFloat;
  {$ENDIF}
begin
  A := TNNetVolume.Create();
  PackedA := TNNetVolume.Create();
  B := TNNetVolume.Create();
  C := TNNetVolume.Create();
  Bias := TNNetVolume.Create();
  try
    for CasePos := 0 to High(csCases) do
    begin
      NumAs := csCases[CasePos, 0];
      NumBs := csCases[CasePos, 1];
      VectorSize := csCases[CasePos, 2];
      BStride := VectorSize + csCases[CasePos, 3];
      CStride := NumAs + csCases[CasePos, 4];
      A.ReSize(NumAs * VectorSize, 1, 1);
      B.ReSize(NumBs * BStride, 1, 1);
      Bias.ReSize(NumAs, 1, 1);
      FillRandom(A);
      FillRandom(B);
      FillRandom(Bias);
      PackedA.ReSize(TNNetVolume.PackedDotProductFloats(NumAs, VectorSize), 1, 1);
      TNNetVolume.PackDotProductRows(PackedA.DataPtr, A.DataPtr, NumAs,
        VectorSize);
      C.ReSize(NumBs * CStride, 1, 1);
      C.Fill(csSentinel);
      if csCases[CasePos, 5] = 1 then BiasPtr := Bias.DataPtr
      else BiasPtr := nil;
      TNNetVolume.DotProductsPacked(C.DataPtr, CStride, PackedA.DataPtr, NumAs,
        B.DataPtr, BStride, NumBs, VectorSize, BiasPtr);
      SetLength(Actual, NumBs * NumAs);
      SetLength(Reference, NumBs * NumAs);
      SetLength(Magnitude, NumBs * NumAs);
      OutPos := 0;
      for RowPos := 0 to NumBs - 1 do
      begin
        for ColPos := 0 to NumAs - 1 do
        begin
          Sum := 0;
          SumAbs := 0;
          for ElementPos := 0 to VectorSize - 1 do
          begin
            Product := double(A.FData[ColPos * VectorSize + ElementPos]) *
              B.FData[RowPos * BStride + ElementPos];
            Sum := Sum + Product;
            SumAbs := SumAbs + Abs(Product);
          end;
          if Assigned(BiasPtr) then Sum := Sum + Bias.FData[ColPos];
          Reference[OutPos] := Sum;
          Magnitude[OutPos] := SumAbs;
          Actual[OutPos] := C.FData[RowPos * CStride + ColPos];
          Inc(OutPos);
        end;
        // Cells between NumAs and CStride stay untouched.
        for ColPos := NumAs to CStride - 1 do
          AssertEquals('case ' + IntToStr(CasePos) + ' gap', csSentinel,
            C.FData[RowPos * CStride + ColPos], 0);
      end;
      AssertCloseToReference('case ' + IntToStr(CasePos), Actual, Reference,
        Magnitude);
      {$IF not (DEFINED(AVX) or DEFINED(AVX2) or DEFINED(AVX512))}
      // The scalar path sums in element order, as TNNetVolume.DotProduct does.
      for RowPos := 0 to NumBs - 1 do
        for ColPos := 0 to NumAs - 1 do
        begin
          ScalarDot := TNNetVolume.DotProduct(A.GetRawPtr(ColPos * VectorSize),
            B.GetRawPtr(RowPos * BStride), VectorSize);
          if Assigned(BiasPtr) then ScalarDot := ScalarDot + Bias.FData[ColPos];
          AssertEquals('case ' + IntToStr(CasePos) + ' scalar order', ScalarDot,
            C.FData[RowPos * CStride + ColPos], 0);
        end;
      {$ENDIF}
    end;
  finally
    Bias.Free;
    C.Free;
    B.Free;
    PackedA.Free;
    A.Free;
  end;
end;

procedure TTestNeuralPlainOps.TestConvsMatchReferenceAtGeneratorSizes;
const
  // Kind (0 Conv1D, 1 PointwiseConv, 2 ConvTranspose1D), Cin, Cout, K,
  // stride, dilation, padding, input length.
  csCases: array[0..7, 0..7] of integer = (
    (0, 128, 128, 3, 1, 1, 1, 41),
    (0, 128, 128, 7, 1, 3, 9, 50),
    (0, 128, 128, 11, 1, 5, 25, 67),
    (0, 256, 256, 11, 1, 1, 5, 23),
    (0, 64, 22, 7, 1, 1, 3, 37),
    (1, 128, 22, 1, 1, 1, 0, 43),
    (2, 64, 32, 12, 6, 1, 3, 13),
    (2, 128, 64, 20, 10, 1, 5, 7));
var
  CasePos, Kind, InChannels, OutChannels, KernelSize, Stride, Dilation: integer;
  Padding, InLength, OutLength, OpsPos: integer;
  OutPos, OutChannelPos, InChannelPos, TapPos, InPos, ElementPos: integer;
  Input, Output, TorchWeight, TorchBias: TNNetVolume;
  Weights: TNNetPlainConvWeights;
  Ops: TNNetPlainOps;
  Actual, Reference, Magnitude: array of double;
  Product, WeightValue: double;
begin
  Input := TNNetVolume.Create();
  Output := TNNetVolume.Create();
  TorchWeight := TNNetVolume.Create();
  TorchBias := TNNetVolume.Create();
  Weights := TNNetPlainConvWeights.Create();
  try
    for CasePos := 0 to High(csCases) do
    begin
      Kind := csCases[CasePos, 0];
      InChannels := csCases[CasePos, 1];
      OutChannels := csCases[CasePos, 2];
      KernelSize := csCases[CasePos, 3];
      Stride := csCases[CasePos, 4];
      Dilation := csCases[CasePos, 5];
      Padding := csCases[CasePos, 6];
      InLength := csCases[CasePos, 7];
      Input.ReSize(InLength, 1, InChannels);
      FillRandom(Input);
      TorchWeight.ReSize(OutChannels * InChannels * KernelSize, 1, 1);
      FillRandom(TorchWeight, 0.5);
      TorchBias.ReSize(OutChannels, 1, 1);
      FillRandom(TorchBias, 0.5);
      if Kind = 2 then
      begin
        Weights.LoadTorchConvTranspose1D(TorchWeight, TorchBias, InChannels,
          OutChannels, KernelSize);
        OutLength := (InLength - 1) * Stride - 2 * Padding + KernelSize;
      end
      else
      begin
        Weights.LoadTorchConv1D(TorchWeight, TorchBias, OutChannels,
          InChannels, KernelSize);
        OutLength := (InLength + 2 * Padding - Dilation * (KernelSize - 1) - 1)
          div Stride + 1;
      end;
      // Reference in double: torch weight [Cout][Cin][K] (conv) or
      // [Cin][Cout][K] (transposed).
      SetLength(Reference, OutLength * OutChannels);
      SetLength(Magnitude, OutLength * OutChannels);
      for ElementPos := 0 to High(Reference) do
      begin
        Reference[ElementPos] := TorchBias.FData[ElementPos mod OutChannels];
        Magnitude[ElementPos] := 0;
      end;
      for OutChannelPos := 0 to OutChannels - 1 do
        for InChannelPos := 0 to InChannels - 1 do
          for TapPos := 0 to KernelSize - 1 do
          begin
            if Kind = 2 then
              WeightValue := TorchWeight.FData[(InChannelPos * OutChannels +
                OutChannelPos) * KernelSize + TapPos]
            else
              WeightValue := TorchWeight.FData[(OutChannelPos * InChannels +
                InChannelPos) * KernelSize + TapPos];
            for OutPos := 0 to OutLength - 1 do
            begin
              if Kind = 2 then
              begin
                // out[o] += w * in[i] where o = i * Stride - Padding + k.
                InPos := OutPos + Padding - TapPos;
                if (InPos < 0) or (InPos mod Stride <> 0) then continue;
                InPos := InPos div Stride;
              end
              else InPos := OutPos * Stride - Padding + TapPos * Dilation;
              if (InPos < 0) or (InPos >= InLength) then continue;
              Product := WeightValue *
                Input.FData[InPos * InChannels + InChannelPos];
              ElementPos := OutPos * OutChannels + OutChannelPos;
              Reference[ElementPos] := Reference[ElementPos] + Product;
              Magnitude[ElementPos] := Magnitude[ElementPos] + Abs(Product);
            end;
          end;
      for OpsPos := 0 to 1 do
      begin
        if OpsPos = 0 then Ops := FSerial else Ops := FThreaded;
        case Kind of
          0: Ops.Conv1D(Input, Output, Weights, Stride, Dilation, Padding);
          1: Ops.PointwiseConv(Input, Output, Weights);
          2: Ops.ConvTranspose1D(Input, Output, Weights, Stride, Padding, 0);
        end;
        AssertEquals('case ' + IntToStr(CasePos) + ' length', OutLength,
          Output.SizeX);
        SetLength(Actual, Output.Size);
        for ElementPos := 0 to Output.Size - 1 do
          Actual[ElementPos] := Output.FData[ElementPos];
        AssertCloseToReference('case ' + IntToStr(CasePos) + ' ops ' +
          IntToStr(OpsPos), Actual, Reference, Magnitude);
      end;
    end;
  finally
    Weights.Free;
    TorchBias.Free;
    TorchWeight.Free;
    Output.Free;
    Input.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralPlainOps);
end.
