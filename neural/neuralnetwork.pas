(*
neuralnetwork
Copyright (C) 2017 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

unit neuralnetwork;

(*
// coded, adapted and ported by Joao Paulo Schwarz Schuler
// https://sourceforge.net/p/cai/
----------------------------------------------
You can find simple to understand examples at:
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/supersimple/
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/supersimplecorrelation/
----------------------------------------------
There are CIFAR-10 examples at:
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/testcnnalgo/testcnnalgo.lpr
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/visualCifar10BatchUpdate/
https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/experiments/visualCifar10OpenCL/
----------------------------------------------
Example - How to Create Your Network
NumClasses := 10;
NN := TNNet.Create();
NN.AddLayer( TNNetInput.Create(32,32,3) );
NN.AddLayer( TNNetConvolutionReLU.Create( 16,5,0,0) );
NN.AddLayer( TNNetMaxPool.Create(2) );
NN.AddLayer( TNNetConvolutionReLU.Create(128,5,0,0) );
NN.AddLayer( TNNetMaxPool.Create(2) );
NN.AddLayer( TNNetConvolutionReLU.Create(128,5,0,0) );
NN.AddLayer( TNNetLayerFullConnectReLU.Create(64) );
NN.AddLayer( TNNetLayerFullConnect.Create(NumClasses) );
NN.SetLearningRate(0.01,0.8);
----------------------------------------------
Example - How to create a simple fully forward connected network 3x3
NN := TNNet.Create();
NN.AddLayer( TNNetInput.Create(3) );
NN.AddLayer( TNNetLayerFullConnectReLU.Create(3) );
NN.AddLayer( TNNetLayerFullConnectReLU.Create(3) );
NN.SetLearningRate(0.01,0.8);
----------------------------------------------
Example - How to Train Your Network
// InputVolume and vDesiredVolume are of the type TNNetVolume
NN.Compute(InputVolume);
NN.GetOutput(PredictedVolume);
vDesiredVolume.SetClassForReLU(DesiredClass);
NN.Backpropagate(vDesiredVolume);
----------------------------------------------
Interesting links:
http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions

Mario Werner coded examples for CIFAR-10 and MNIST:
https://bitbucket.org/108bits/cai-implementations/src/c8c027b1a0d636713f7ebb70a738f1cd7117a7a4?at=master
*)

{$include neuralnetwork.inc}

interface

uses
  {$IFDEF OpenCL}
    {$IFDEF FPC}
    cl,
    neuralopencl,
    {$ELSE} // For Delphi Compiler
    cl, // https://github.com/CWBudde/PasOpenCL
    neuralopencl,
    {$ENDIF}
  {$ENDIF}
  {$IFDEF FPC}
  fgl,
  {$ENDIF}
  Classes, SysUtils, math, syncobjs, neuralvolume, neuralgeneric,
  neuralbyteprediction, neuralcache, neuralab;

const
  csMaxInterleavedSize: integer = 95;

type
  { TNNetNeuron }
  TNNetNeuron = class (TMObject)
    protected
      FWeights: TNNetVolume;
      FBackInertia: TNNetVolume;
      FDelta: TNNetVolume;

    private
      FBiasWeight: TNeuralFloat;
      FBiasInertia: TNeuralFloat;
      FBiasDelta: TNeuralFloat;

    public
      constructor Create(); override;
      destructor Destroy(); override;
      procedure Fill(Value:TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure AddInertia(); {$IFDEF Release} inline; {$ENDIF}
      procedure UpdateWeights(Inertia:TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      function SaveToString(): string;
      procedure LoadFromString(strData: string);
      procedure ClearDelta; {$IFDEF Release} inline; {$ENDIF}

      // Initializers
      procedure InitUniform(Value: TNeuralFloat = 1);
      procedure InitGaussian(Value: TNeuralFloat = 1);
      procedure InitLeCunUniform(Value: TNeuralFloat = 1);
      procedure InitHeUniform(Value: TNeuralFloat = 1);
      procedure InitHeGaussian(Value: TNeuralFloat = 1);
      procedure InitHeUniformDepthwise(Value: TNeuralFloat = 1);
      procedure InitHeGaussianDepthwise(Value: TNeuralFloat = 1);
      procedure InitSELU(Value: TNeuralFloat = 1);

      property Weights: TNNetVolume read FWeights;
      property BackInertia: TNNetVolume read FBackInertia;
      property Delta: TNNetVolume read FDelta;
  end;

  {$IFDEF FPC}
  TNNetNeuronList = class (specialize TFPGObjectList<TNNetNeuron>)
    public
  {$ELSE}
  TNNetNeuronList = class (TNNetList)
    private
      function GetItem(Index: Integer): TNNetNeuron; inline;
      procedure SetItem(Index: Integer; AObject: TNNetNeuron); inline;
    public
      property Items[Index: Integer]: TNNetNeuron read GetItem write SetItem; default;
  {$ENDIF}
      constructor CreateWithElements(ElementCount: integer);
      function GetMaxWeight(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetMaxAbsWeight(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetMinWeight(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      procedure InitForDebug();
  end;

  const
    csNNetMaxParameterIdx = 7;

  type
  TNNet = class;
  /// neural network layer
  TNNetLayer = class(TMObject)
    protected
      FActivationFn: TNeuralActivationFunction;
      FActivationFnDerivative: TNeuralActivationFunction;
      FForwardTime: double;
      FBackwardTime: double;
      FNeurons: TNNetNeuronList;
      FOutput: TNNetVolume;
      FOutputRaw: TNNetVolume;
      FOutputError: TNNetVolume;
      FOutputErrorDeriv: TNNetVolume;
      FSmoothErrorPropagation: boolean;
      FBatchUpdate: boolean;
      FSuppressBias: integer;
      // Fast access to TNNetNeuron
      FArrNeurons: array of TNNetNeuron;

      FInertia: TNeuralFloat;
      FPrevLayer: TNNetLayer;
      FLearningRate: TNeuralFloat;
      FL2Decay: TNeuralFloat;
      FStruct: array [0..csNNetMaxParameterIdx] of integer;

      //backpropagation properties
      FDepartingBranchesCnt: integer;
      FBackPropCallCurrentCnt: integer;
      FLinkedNeurons: boolean;
      FNN: TNNet;

      procedure InitStruct();
    private
      FLayerIdx: integer;
      {$IFDEF OpenCL}
      FHasOpenCL: boolean;
      FShouldOpenCL: boolean;
      FDotCL: TDotProductSharedKernel;
      FDotProductKernel: TDotProductKernel;
      {$ENDIF}

      procedure ComputeL2Decay(); virtual;
      procedure ComputePreviousLayerError(); virtual;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); virtual;
      procedure ApplyActivationFunctionToOutput(); virtual;
      procedure BuildArrNeurons();
      procedure AfterWeightUpdate(); virtual;
    public
      constructor Create(); override;
      destructor Destroy(); override;

      {$IFDEF OpenCL}
      procedure DisableOpenCL(); virtual;
      procedure EnableOpenCL(DotProductKernel: TDotProductKernel); virtual;
      {$ENDIF}
      procedure Compute(); virtual; abstract;
      procedure Backpropagate(); virtual; abstract;
      procedure ComputeOutputErrorForOneNeuron(NeuronIdx: integer; value: TNeuralFloat);
      procedure ComputeOutputErrorWith(pOutput: TNNetVolume); virtual;
      procedure ComputeOutputErrorForIdx(pOutput: TNNetVolume; const aIdx: array of integer); virtual;
      procedure ComputeErrorDeriv(); {$IFDEF FPC}{$IFDEF Release} inline; {$ENDIF}{$ENDIF}
      procedure Fill(value: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure ClearDeltas(); {$IFDEF Release} inline; {$ENDIF}
      procedure AddNeurons(NeuronNum: integer);
      procedure AddMissingNeurons(NeuronNum: integer);
      procedure SetNumWeightsForAllNeurons(NumWeights: integer); overload;
      procedure SetNumWeightsForAllNeurons(x, y, d: integer); overload;
      procedure SetNumWeightsForAllNeurons(Origin: TNNetVolume); overload;
      function GetMaxWeight(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetMaxAbsWeight(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetMinWeight(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetMaxDelta(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetMinDelta(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function ForceMaxAbsoluteDelta(vMax: TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function ForceMaxAbsoluteWeight(vMax: TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetMaxAbsoluteDelta(): TNeuralFloat; virtual;
      procedure GetMinMaxAtDepth(pDepth: integer; var pMin, pMax: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      function GetWeightSum(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetBiasSum(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetInertiaSum(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function CountWeights(): integer; {$IFDEF Release} inline; {$ENDIF}
      function CountNeurons(): integer; {$IFDEF Release} inline; {$ENDIF}
      procedure MulWeights(V:TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure MulDeltas(V:TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure ClearInertia(); {$IFDEF Release} inline; {$ENDIF}
      procedure ClearTimes(); {$IFDEF Release} inline; {$ENDIF}
      procedure AddTimes(Origin: TNNetLayer); {$IFDEF Release} inline; {$ENDIF}
      procedure CopyTimes(Origin: TNNetLayer); {$IFDEF Release} inline; {$ENDIF}
      procedure MulMulAddWeights(Value1, Value2: TNeuralFloat; Origin: TNNetLayer); {$IFDEF Release} inline; {$ENDIF}
      procedure SumWeights(Origin: TNNetLayer); {$IFDEF Release} inline; {$ENDIF}
      procedure SumDeltas(Origin: TNNetLayer); {$IFDEF Release} inline; {$ENDIF}
      procedure SumDeltasNoChecks(Origin: TNNetLayer); {$IFDEF Release} inline; {$ENDIF}
      procedure CopyWeights(Origin: TNNetLayer); virtual;
      procedure ForceRangeWeights(V:TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure NormalizeWeights(VMax: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      function SaveDataToString(): string; virtual;
      procedure LoadDataFromString(strData: string); virtual;
      function SaveStructureToString(): string; virtual;
      procedure SetBatchUpdate(pBatchUpdate: boolean); {$IFDEF Release} inline; {$ENDIF}
      procedure UpdateWeights(); {$IFDEF Release} inline; {$ENDIF}
      function InitBasicPatterns(): TNNetLayer;

      // Backprop call cnt
      procedure IncDepartingBranchesCnt(); {$IFDEF Release} inline; {$ENDIF}
      procedure ResetBackpropCallCurrCnt(); {$IFDEF Release} inline; {$ENDIF}

      // Initializers
      function InitUniform(Value: TNeuralFloat = 1): TNNetLayer;
      function InitLeCunUniform(Value: TNeuralFloat = 1): TNNetLayer;
      function InitHeUniform(Value: TNeuralFloat = 1): TNNetLayer;
      function InitHeUniformDepthwise(Value: TNeuralFloat = 1): TNNetLayer;
      function InitHeGaussian(Value: TNeuralFloat = 0.5): TNNetLayer;
      function InitHeGaussianDepthwise(Value: TNeuralFloat = 0.5): TNNetLayer;
      function InitGlorotBengioUniform(Value: TNeuralFloat = 1): TNNetLayer;
      function InitSELU(Value: TNeuralFloat = 1): TNNetLayer;
      procedure InitDefault(); virtual;

      property ActivationFn: TNeuralActivationFunction read FActivationFn write FActivationFn;
      property ActivationFnDerivative: TNeuralActivationFunction read FActivationFnDerivative write FActivationFnDerivative;
      property Neurons: TNNetNeuronList read FNeurons;
      property NN:TNNet read FNN write FNN;
      property Output: TNNetVolume read FOutput;
      property OutputRaw: TNNetVolume read FOutputRaw;
      property PrevLayer: TNNetLayer read FPrevLayer write SetPrevLayer;
      property LearningRate: TNeuralFloat read FLearningRate write FLearningRate;
      property L2Decay: TNeuralFloat read FL2Decay write FL2Decay;
      property Inertia: TNeuralFloat read FInertia;
      property OutputError: TNNetVolume read FOutputError write FOutputError;
      property OutputErrorDeriv: TNNetVolume read FOutputErrorDeriv write FOutputErrorDeriv;
      property LayerIdx: integer read FLayerIdx;
      property SmoothErrorPropagation: boolean read FSmoothErrorPropagation write FSmoothErrorPropagation;
      property BackwardTime: double read FBackwardTime write FBackwardTime;
      property ForwardTime: double read FForwardTime write FForwardTime;
      property LinkedNeurons: boolean read FLinkedNeurons;
  end;

  TNNetLayerClass = class of TNNetLayer;

  /// This is a base class. Do not use it directly.
  TNNetLayerConcatedWeights = class(TNNetLayer)
    protected
      FVectorSize, FVectorSizeBytes: integer;
      FNeuronWeightList: TNNetVolumeList;
      FConcatedWeights: TNNetVolume;
      FConcatedWInter: TNNetVolume;
      FBiasOutput: TNNetVolume;
      FShouldConcatWeights: boolean;
      FShouldInterleaveWeights: boolean;
      FAfterWeightUpdateHasBeenCalled:boolean;
      procedure AfterWeightUpdate(); override;
      procedure BuildBiasOutput(); {$IFDEF Release} inline; {$ENDIF}
    public
      constructor Create(); override;
      destructor Destroy(); override;
      procedure RefreshNeuronWeightList();
      {$IFDEF OpenCL}
      procedure EnableOpenCL(DotProductKernel: TDotProductKernel); override;
      {$ENDIF}
  end;

  {$IFDEF FPC}
  TNNetLayerList = specialize TFPGObjectList<TNNetLayer>;
  {$ELSE}
  TNNetLayerList = class (TNNetList)
    private
      function GetItem(Index: Integer): TNNetLayer; inline;
      procedure SetItem(Index: Integer; AObject: TNNetLayer); inline;
    public
      property Items[Index: Integer]: TNNetLayer read GetItem write SetItem; default;
  end;
  {$ENDIF}

  /// This is a base class. Do not use it directly.
  TNNetInputBase = class(TNNetLayer)
  private
    procedure ComputePreviousLayerError(); override;
  public
    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// This is an ideal layer to be used as input layer. In the case that you
  // need to backpropagate errors up to the input, call EnableErrorCollection.
  TNNetInput = class(TNNetInputBase)
    public
      constructor Create(pSize: integer); overload;
      constructor Create(pSizeX, pSizeY, pDepth: integer); overload;
      constructor Create(pSizeX, pSizeY, pDepth, pError: integer); overload;

      function EnableErrorCollection: TNNetInput;
      function DisableErrorCollection: TNNetInput;
  end;

  /// This layer copies the input to the output and can be used as a base class
  // to your new layers.
  TNNetIdentity = class(TNNetLayer)
    private
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This layer allows you to debug activation and backpropagation of an
  TNNetDebug = class(TNNetIdentity)
    public
      constructor Create(hasForward, hasBackward: integer); overload;
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// Padding layer: adds padding to the input.
  // This layer has no trainable parameter. Adding a padding layer may be
  // more efficient than padding at the convolutional layer.
  TNNetPad = class(TNNetLayer)
  private
    FPadding: integer;
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    constructor Create(Padding: integer); overload;
    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// Base class to be used with layers that aren't compatible with L2
  TNNetIdentityWithoutL2 = class(TNNetIdentity)
    private
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      procedure ComputeL2Decay(); override;
  end;

  /// This layer can be used when you need the forward pass but can't let
  // error backpropagation to pass.
  TNNetIdentityWithoutBackprop = class(TNNetIdentity)
    public
      procedure Backpropagate(); override;
  end;

  // Class of activation function layers.
  TNNetActivationFunctionClass = class of TNNetIdentity;

  /// This is a base/abstract class. Do not use it directly.
  TNNetReLUBase = class(TNNetIdentity)
    public
      procedure Backpropagate(); override;
  end;

  TNNetDigital = class(TNNetIdentity)
    private
      FMiddleValue: TNeuralFloat;
      FLowValue, FHighValue: TNeuralFloat;
      FMiddleDist: TNeuralFloat;
    public
      constructor Create(LowValue, HighValue: integer); overload;
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This is a plain Rectified Linear Unit (ReLU) layer.
  // https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  TNNetReLU = class(TNNetReLUBase)
    public
      procedure Compute(); override;
  end;

  /// This is a leaky ReLU with minimum and maximum values. You can
  // scale leakiness via the Leaky parameter.
  TNNetReLUL = class(TNNetReLUBase)
    private
      FScale, FLowLimit, FHighLimit: TNeuralFloat;
    public
      constructor Create(LowLimit, HighLimit, Leakiness: integer); overload;
      procedure Compute(); override;
  end;

  /// This is a Relu with low limit = 0 and high limit = 6. You
  // can optionally make this activation function leaky.
  TNNetReLU6 = class(TNNetReLUL)
    public
      constructor Create(Leakiness: integer = 0); overload;
  end;

  /// Scaled Exponential Linear Unit
  // https://arxiv.org/pdf/1706.02515.pdf
  // You might need to lower your learning rate with SELU.
  TNNetSELU = class(TNNetReLUBase)
    private
      FAlpha: TNeuralFloat;
      FScale: TNeuralFloat;
      FScaleAlpha: TNeuralFloat;
      FThreshold: TNeuralFloat;
    public
      constructor Create(); override;
      procedure Compute(); override;
  end;

  /// Swish activation function
  // https://arxiv.org/abs/1710.05941
  TNNetSwish = class(TNNetReLUBase)
  public
    procedure Compute(); override;
  end;

  /// Swish activation function with maximum limit of 6
  TNNetSwish6 = class(TNNetReLUBase)
  public
    procedure Compute(); override;
  end;

  //Does a ReLU followed by a Square Root
  TNNetReLUSqrt = class(TNNetReLUBase)
    public
      procedure Compute(); override;
  end;

  // Calculates Power(LocalPrevOutput.FData[OutputCnt], iPower).
  TNNetPower = class(TNNetReLUBase)
    private
      FPower: TNeuralFloat;
    public
      constructor Create(iPower: integer); overload;
      procedure Compute(); override;
  end;

  /// Leaky Rectified Linear Unit (ReLU) layer.
  // https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  TNNetLeakyReLU = class(TNNetReLUBase)
    private
      FAlpha: TNeuralFloat;
      FThreshold: TNeuralFloat;
    public
      constructor Create(); override;
      procedure Compute(); override;
  end;

  /// Very Leaky Rectified Linear Unit (ReLU) layer.
  // https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
  TNNetVeryLeakyReLU = class(TNNetLeakyReLU)
    public
      constructor Create(); override;
  end;

  /// This is a plain Sigmoid layer.
  TNNetSigmoid = class(TNNetIdentity)
    private
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(); override;
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This is a plain Hyperbolic Tangent layer.
  TNNetHyperbolicTangent = class(TNNetSigmoid)
    public
      constructor Create(); override;
  end;

  /// This layer multiplies the learning in previous layers. It can speed up
  // learning but can also provoke overflows.
  TNNetMulLearning = class(TNNetIdentity)
    public
      constructor Create(pMul: integer); overload;
      procedure Backpropagate(); override;
  end;

  /// This layer multiplies the output by a constant.
  TNNetMulByConstant = class(TNNetMulLearning)
    public
      //constructor Create(pMul: integer); overload;
      procedure Compute(); override;
  end;

  // This layer multiplies the previous output by -1
  TNNetNegate = class(TNNetMulByConstant)
    public
      constructor Create(); override;
  end;

  /// This is an experimental layer. Do not use it.
  TNNetAddAndDiv = class(TNNetIdentity)
  public
    constructor Create(pAdd, pDiv: integer); overload;
    procedure Compute(); override;
  end;

  TNNetAddNoiseBase = class(TNNetIdentity)
  protected
    FEnabled: boolean;
  public
    property Enabled:boolean read FEnabled write FEnabled;
  end;

  /// Dropout layer. The input parameter is the dropout rate (rate of values
  // that are zeroed).
  TNNetDropout = class(TNNetAddNoiseBase)
    protected
      FRate: integer;
      FDropoutMask: TNNetVolume;
    private
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(Rate: double; OneMaskPerbatch: integer = 1); overload;
      destructor Destroy(); override;
      procedure Compute(); override;
      procedure Backpropagate(); override;
      procedure CopyWeights(Origin: TNNetLayer); override;
      procedure RefreshDropoutMask();
      property DropoutMask: TNNetVolume read FDropoutMask;
  end;

  /// This layer adds a random addition (or bias) and amplifies (multiplies)
  // randomly. Parameter 10 means changes with up to 1%. Parameter 1
  // means 0.1% and 0 means no change. This layer was create to prevent
  // overfitting and force generalization.
  TNNetRandomMulAdd = class(TNNetAddNoiseBase)
  protected
    FRandomBias, FRandomMul: TNeuralFloat;
  public
    constructor Create(AddRate, MulRate: integer); overload;
    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// This layers adds a small random bias (shift) and small
  // random multiplication (scaling).
  TNNetChannelRandomMulAdd = class(TNNetAddNoiseBase)
  protected
    FRandomBias, FRandomMul: TNNetVolume;
  public
    constructor Create(AddRate, MulRate: integer); overload;
    destructor Destroy; override;
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// This layer does a MAX normalization. There are no trainable parameters
  // in this layer.
  TNNetLayerMaxNormalization = class(TNNetIdentity)
    private
      FLastMax: TNeuralFloat;
    public
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This layer does a standard normalization. There are no trainable parameters
  // in this layer.
  TNNetLayerStdNormalization = class(TNNetIdentity)
    private
      FLastStdDev: TNeuralFloat;
    public
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This layer does zero centering and standard normalization with trainable
  // parameters.
  TNNetMovingStdNormalization = class(TNNetIdentityWithoutL2)
    public
      constructor Create(); override;
      procedure Compute(); override;
      procedure Backpropagate(); override;
      procedure InitDefault(); override;
      function GetMaxAbsoluteDelta(): TNeuralFloat; override;
  end;

  // This is an experimental layer. Do not use it.
  TNNetScaleLearning = class(TNNetMovingStdNormalization)
    public
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This is a base class. Do not use it directly.
  TNNetChannelTransformBase = class(TNNetIdentityWithoutL2)
    private
      FAuxDepth: TNNetVolume;
      FOutputChannelSize: TNeuralFloat;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(); override;
      destructor Destroy(); override;
  end;

  /// This is a base class. Do not use it directly.
  TNNetChannelShiftBase = class(TNNetChannelTransformBase)
    public
      procedure Compute(); override;
      procedure InitDefault(); override;
  end;

  /// This layer adds a trainable bias to each channel.
  TNNetChannelBias = class(TNNetChannelShiftBase)
    public
      procedure Backpropagate(); override;
  end;

  /// This layer multiplies (scales) each channel by a trainable number.
  TNNetChannelMul = class(TNNetChannelTransformBase)
    public
      procedure Compute(); override;
      procedure Backpropagate(); override;
      procedure InitDefault(); override;
  end;

  // This is an experimental class. Do not use it.
  TNNetChannelMulByLayer = class(TNNetChannelTransformBase)
    private
      FLayerWithChannelsIdx, FLayerMulIdx: integer;
      FLayerWithChannels, FLayerMul: TNNetLayer;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(LayerWithChannels, LayerMul: TNNetLayer); overload;
      constructor Create(LayerWithChannelsIdx, LayerMulIdx: integer); overload;
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This layer multiplies each cell from one branch to each cell from another
  // branch.
  TNNetCellMulByCell = class(TNNetChannelTransformBase)
    private
      FLayerAIdx, FLayerBIdx: integer;
      FLayerA, FLayerB: TNNetLayer;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(LayerA, LayerB: TNNetLayer); overload;
      constructor Create(LayerAIdx, LayerBIdx: integer); overload;
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This layer adds a trainable bias to each output cell. Placing
  // this layer before and after convolutions can speed up learning.
  // It's useless placing this layer after fully connected layers with bias.
  TNNetCellBias = class(TNNetIdentityWithoutL2)
    private
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      procedure Compute(); override;
      procedure Backpropagate(); override;
      procedure InitDefault(); override;
  end;

  /// This layer multiplies each output cell by a trainable number. Placing
  // this layer before and after convolutions can speed up learning.
  TNNetCellMul = class(TNNetIdentityWithoutL2)
  private
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    procedure Compute(); override;
    procedure Backpropagate(); override;
    procedure InitDefault(); override;
  end;

  /// This layer zero centers the output. This layer placed
  // before convolutional layers can speed up learning A LOT. Use
  // this layer in combination with batch update and NormalizeMaxAbsoluteDelta()
  // as it can produce spikes in the learning provoking overflow.
  TNNetChannelZeroCenter = class(TNNetChannelShiftBase)
    public
      procedure Backpropagate(); override;
      procedure ComputeL2Decay(); override;
  end;

  /// This layer does zero centering and standard normalization per channel with
  // trainable parameters.
  TNNetChannelStdNormalization = class(TNNetChannelZeroCenter)
    private
      FAuxOutput: TNNetVolume;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(); override;
      destructor Destroy(); override;
      procedure Compute(); override;
      procedure Backpropagate(); override;
      procedure InitDefault(); override;
      function GetMaxAbsoluteDelta(): TNeuralFloat; override;
  end;

  /// This layer has no trainable parameter. It does a spacial (per channel)
  // local response normalization.
  TNNetLocalResponseNorm2D = class(TNNetIdentity)
    private
      FLRN: TNNetVolume;
    public
      constructor Create(pSize: integer); overload;
      destructor Destroy(); override;

      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This layer interleaves input channels.
  TNNetInterleaveChannels = class(TNNetIdentity)
    private
      ToChannels: TNeuralIntegerArray;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(StepSize: integer); overload;
      destructor Destroy(); override;

      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This layer has no trainable parameter. It does a cross channel local
  // response normalization.
  TNNetLocalResponseNormDepth = class(TNNetLocalResponseNorm2D)
  public
    procedure Compute(); override;
  end;

  /// This layer reshapes the input into the output.
  TNNetReshape = class(TNNetLayer)
    private
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(pSizeX, pSizeY, pDepth: integer); overload;

      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This is a base class. Do not use it directly.
  TNNetConcatBase = class(TNNetLayer)
  private
    FPrevOutput: TNNetVolumeList;
    FPrevOutputError: TNNetVolumeList;
    FPrevOutputErrorDeriv: TNNetVolumeList;
    FPrevLayerList: TNNetLayerList;
  public
    constructor Create(); override;
    destructor Destroy(); override;

    function SaveStructureToString(): string; override;
    procedure BackpropagateConcat();
  end;

  /// This layer concatenates previous layers into the X axis. Consider using
  // TNNetDeepConcat if you intend to concatenate outputs/volumes with same
  // XY size.
  TNNetConcat = class(TNNetConcatBase)
  public
    constructor Create(pSizeX, pSizeY, pDepth: integer; aL: array of TNNetLayer); overload;
    constructor Create(aL: array of TNNetLayer); overload;

    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// This layer concatenates other layers into the deep/depth dimension.
  // You should concatenate outputs/volumes with same XY size.
  TNNetDeepConcat = class(TNNetConcatBase)
  protected
    FDeepsLayer: TNeuralIntegerArray;
    FDeepsChannel: TNeuralIntegerArray;
    FRemainingChannels: TNeuralIntegerArray;
  public
    constructor Create(aL: array of TNNetLayer); overload;
    destructor Destroy(); override;

    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// This layer sums layers of same size allowing resnet style layers.
  TNNetSum = class(TNNetConcatBase)
  public
    constructor Create(aL: array of TNNetLayer); overload;
    destructor Destroy(); override;

    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// picks/splits from previous layer selected channels.
  TNNetSplitChannels = class(TNNetLayer)
  private
    FChannels: TNeuralIntegerArray;
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    constructor Create(ChannelStart, ChannelLen: integer); overload;
    constructor Create(pChannels: array of integer); overload;
    destructor Destroy(); override;

    procedure Compute(); override;
    procedure Backpropagate(); override;

    function SaveStructureToString(): string; override;
  end;

  TNNetSplitChannelEvery = class(TNNetSplitChannels)
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    constructor Create(GetChannelEvery, ChannelShift: integer); overload;
    constructor Create(pChannels: array of integer); overload;
  end;

  /// Fully connected layer with hyperbolic tangent.
  TNNetFullConnect = class(TNNetLayerConcatedWeights)
    private
      FAuxTransposedW: TNNetVolume;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
      procedure ComputePreviousLayerError(); override;
      procedure ComputePreviousLayerErrorCPU(); virtual;
    public
      constructor Create(pSizeX, pSizeY, pDepth: integer; pSuppressBias: integer = 0); overload; virtual;
      constructor Create(pSize:integer; pSuppressBias: integer = 0); overload;
      procedure Compute(); override;
      procedure ComputeCPU(); virtual;
      procedure Backpropagate(); override;
      procedure BackpropagateCPU(); virtual;
      destructor Destroy(); override;
      {$IFDEF OpenCL}
      procedure EnableOpenCL(DotProductKernel: TDotProductKernel); override;
      procedure ComputeOpenCL(); virtual;
      procedure BackpropagateOpenCL(); virtual;
      {$ENDIF}
  end;

  //FullyConnectedLayers
  TNNetFullConnectClass = class of TNNetFullConnect;

  /// Fully connected layer without activation function. This layer is useful
  // before softmax layers.
  TNNetFullConnectLinear = class(TNNetFullConnect)
  private
    procedure ComputePreviousLayerErrorCPU(); override;
  public
    procedure ComputeCPU(); override;
    procedure BackpropagateCPU(); override;
    constructor Create(pSizeX, pSizeY, pDepth: integer; pSuppressBias: integer = 0); override;
    constructor Create(pSize: integer; pSuppressBias: integer = 0); overload;
  end;

  /// Fully connected layer with Sigmoid activation function.
  TNNetFullConnectSigmoid = class(TNNetFullConnect)
  public
    constructor Create(pSizeX, pSizeY, pDepth: integer; pSuppressBias: integer = 0); override;
    constructor Create(pSize: integer; pSuppressBias: integer = 0); overload;
  end;

  /// Fully connected layer with ReLU.
  TNNetFullConnectReLU = class(TNNetFullConnectLinear)
  private
    procedure ComputePreviousLayerErrorCPU(); override;
  public
    procedure ComputeCPU(); override;
    procedure BackpropagateCPU(); override;
    constructor Create(pSizeX, pSizeY, pDepth: integer; pSuppressBias: integer = 0); override;
    constructor Create(pSize: integer; pSuppressBias: integer = 0); overload;
  end;

  /// Do not use this layer. This is still experimental.
  TNNetFullConnectDiff = class(TNNetFullConnectReLU)
  private
    procedure ComputePreviousLayerError(); override;
  public
    constructor Create(pSizeX, pSizeY, pDepth: integer; pSuppressBias: integer = 0); override;
    constructor Create(pSize: integer; pSuppressBias: integer = 0); overload;

    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// Common softmax layer.
  TNNetSoftMax = class(TNNetIdentity)
    protected
      FSoftTotalSum: TNeuralFloat;
    public
      procedure Compute(); override;
  end;

  TNNetLayerFullConnect = class(TNNetFullConnect);
  TNNetLayerFullConnectReLU = class(TNNetFullConnectReLU);
  TNNetLayerSoftMax = class(TNNetSoftMax);
  TNNetDense = class(TNNetFullConnect);
  TNNetDenseReLU = class(TNNetFullConnectReLU);

  /// This is a base class. Do not use it directly.
  TNNetConvolutionAbstract = class(TNNetLayerConcatedWeights)
    private
      FPadding: integer;
      FStride: integer;
      FOutputSizeX, FOutputSizeY: integer;
      FFeatureSizeX, FFeatureSizeY: integer;
      FFeatureSizeYMinus1, FFeatureSizeXMinus1: integer;
      FInputCopy: TNNetVolume;
      FSizeXDepth: integer;
      FSizeXDepthBytes: integer;
      FPrevSizeXDepthBytes: integer;
      FCalculatePrevLayerError: boolean;
      function CalcOutputSize(pInputSize, pFeatureSize, pInputPadding, pStride: integer) : integer;
      procedure RefreshCalculatePrevLayerError();
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(pFeatureSize, pInputPadding, pStride: integer; pSuppressBias: integer = 0); overload;
      destructor Destroy(); override;
      procedure InitDefault(); override;
  end;

  /// This class does a depthwise convolution.
  TNNetDepthwiseConv = class(TNNetConvolutionAbstract)
  private
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    procedure BackpropagateCPU(); {$IFDEF Release} inline; {$ENDIF}
    procedure BackpropagateCPUFast();
    procedure BackpropagateAtOutputPos(OutputX, OutputY, NeuronIdx, PrevX, PrevY: integer; bCanBackPropagate: boolean); {$IFDEF Release} inline; {$ENDIF}
    procedure ComputeCPU(); {$IFDEF Release} inline; {$ENDIF}
    procedure ComputeCPUAtOutputPos(NeuronIdx, OutputX, OutputY: integer); {$IFDEF Release} inline; {$ENDIF}
    procedure ComputeCPUFast();
  public
    constructor Create(pMultiplier, pFeatureSize, pInputPadding, pStride: integer); overload; virtual;
    procedure Compute(); override;
    procedure Backpropagate(); override;
    procedure InitDefault(); override;
  end;

  /// Depthwise Convolutional layer with Linear activation function.
  TNNetDepthwiseConvLinear = class(TNNetDepthwiseConv)
  public
    constructor Create(pMultiplier, pFeatureSize, pInputPadding, pStride: integer); override;
  end;

  /// Depthwise Convolutional layer with ReLU activation function.
  TNNetDepthwiseConvReLU = class(TNNetDepthwiseConv)
  public
    constructor Create(pMultiplier, pFeatureSize, pInputPadding, pStride: integer); override;
  end;

  /// This is a base class. Do not use it directly.
  TNNetConvolutionBase = class(TNNetConvolutionAbstract)
    private
      FInputPrepared: TNNetVolume;
      //FDotProductResult: TNNetVolume;
      FPointwise: boolean;
      FLearnSmoothener: TNeuralFloat;
      // Tiling
      FMaxTileX, FMaxTileD: integer;
      FTileSizeX, FTileSizeD: integer;

      {$IFDEF Debug}
      procedure PrepareInputForConvolution(); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure PrepareInputForConvolution(OutputX, OutputY: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      {$ENDIF}
      procedure PrepareInputForConvolutionFast();
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
      function ShouldUseInterleavedDotProduct:boolean; {$IFDEF Release} inline; {$ENDIF}
    public
      constructor Create(pNumFeatures, pFeatureSize, pInputPadding, pStride: integer; pSuppressBias: integer = 0); overload; virtual;
      destructor Destroy(); override;
      {$IFDEF OpenCL}
      procedure EnableOpenCL(DotProductKernel: TDotProductKernel); override;
      {$ENDIF}

      property Pointwise: boolean read FPointwise;
  end;

  TNNetConvolutionClass = class of TNNetConvolutionBase;

  /// This layer is under construction. DO NOT USE IT.
  TNNetGroupedConvolutionLinear = class(TNNetConvolutionBase)
    private
      FArrGroupId: array of integer;
      FArrGroupIdStart: array of integer;
      FMaxPrevX, FMaxPrevY: integer;
      procedure PrepareInputForGroupedConvolutionFast();
      procedure ComputeCPU();
      procedure BackpropagateCPU();
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pGroups: integer; pSuppressBias: integer = 0); overload; virtual;
      destructor Destroy(); override;

      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  /// This layer is under construction. DO NOT USE IT.
  TNNetGroupedConvolutionReLU = class(TNNetGroupedConvolutionLinear)
    public
      constructor Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pGroups: integer; pSuppressBias: integer = 0); overload; override;
  end;

  /// Grouped pointwise convolution with Linear activation.
  TNNetGroupedPointwiseConvLinear = class(TNNetGroupedConvolutionLinear)
  public
    constructor Create(pNumFeatures, pGroups: integer; pSuppressBias: integer = 0); virtual;
  end;

  TNNetGroupedPointwiseConvClass = class of TNNetGroupedPointwiseConvLinear;

  /// Grouped pointwise convolution with ReLU activation.
  TNNetGroupedPointwiseConvReLU = class(TNNetGroupedPointwiseConvLinear)
  public
    constructor Create(pNumFeatures, pGroups: integer; pSuppressBias: integer = 0); override;
  end;

  /// Convolutional layer with hyperbolic tangent activation function.
  TNNetConvolution = class(TNNetConvolutionBase)
    protected
      procedure BackpropagateAtOutputPos(pCanBackpropOnPos: boolean; OutputRawPos, OutputX, OutputY, OutputD, PrevX, PrevY: integer); {$IFDEF Release} inline; {$ENDIF}
    private
      procedure ComputeCPU();
      procedure ComputeTiledCPU();
      procedure ComputeInterleaved();
      procedure BackpropagateCPU();
      procedure BackpropagateFastCPU();
      procedure BackpropagateFastTiledCPU();
      procedure BackpropagateFastCPUDev(); // Backprop CPU development version (do not use it)

      {$IFDEF OpenCL}
      procedure ComputeOpenCL();
      {$ENDIF}
      {$IFDEF Debug}
      procedure ComputeNeuronCPU(); {$IFDEF Release} inline; {$ENDIF}
      procedure AddBiasToRawResult(); {$IFDEF Release} inline; {$ENDIF}
      //procedure ComputeNeuronFromResult(NeuronIdx: integer); {$IFDEF Release} inline; {$ENDIF}
      procedure ComputeNeuron(NeuronIdx: integer); {$IFDEF Release} inline; {$ENDIF}
      procedure ComputeNeuronAtOutputPos(NeuronIdx, x, y: integer); {$IFDEF Release} inline; {$ENDIF}
      function ComputeNeuronAtOutputPos3(NeuronIdx, x, y: integer): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function ComputeNeuronAtOutputPos3D3(NeuronIdx, x, y: integer): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function ComputeNeuronAtOutputPosDefault(NeuronIdx, x, y: integer): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function ComputeNeuronAtOutputPosDefaultFast(NeuronIdx, x, y: integer): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function ComputeNeuronAtPreparedInput(NeuronIdx, x, y: integer): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      {$ENDIF}
    public
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  TNNetConvolutionSharedWeights = class(TNNetConvolution)
    private
      FLinkedLayer: TNNetConvolution;
    public
      constructor Create(LinkedLayer: TNNetLayer); overload; virtual;
      destructor Destroy; override;
  end;

  /// Convolutional layer without activation function.
  TNNetConvolutionLinear = class(TNNetConvolution)
  public
    constructor Create(pNumFeatures, pFeatureSize, pInputPadding, pStride: integer; pSuppressBias: integer = 0); override;
  end;

  /// Convolutional layer with ReLU activation function.
  TNNetConvolutionReLU = class(TNNetConvolution)
  public
    constructor Create(pNumFeatures, pFeatureSize, pInputPadding, pStride: integer; pSuppressBias: integer = 0); override;
  end;

  /// Pointwise convolution with tanh activation.
  TNNetPointwiseConv = class(TNNetConvolution)
  public
    constructor Create(pNumFeatures: integer; pSuppressBias: integer = 0); virtual;
  end;

  /// Pointwise convolution with Linear activation.
  TNNetPointwiseConvLinear = class(TNNetConvolutionLinear)
  public
    constructor Create(pNumFeatures: integer; pSuppressBias: integer = 0); virtual;
  end;

  /// Pointwise convolution with ReLU activation.
  TNNetPointwiseConvReLU = class(TNNetConvolutionReLU)
  public
    constructor Create(pNumFeatures: integer; pSuppressBias: integer = 0); virtual;
  end;

  { TNNetDeconvolution }
  TNNetDeconvolution = class(TNNetConvolution)
  public
    constructor Create(pNumFeatures, pFeatureSize: integer; pSuppressBias: integer = 0); overload;
  end;

  { TNNetDeconvolutionReLU }
  TNNetDeconvolutionReLU = class(TNNetConvolutionReLU)
  public
    constructor Create(pNumFeatures, pFeatureSize: integer; pSuppressBias: integer = 0); overload;
  end;

  { TNNetLocalConnect }
  TNNetLocalConnect = class(TNNetConvolutionBase)
    private
      procedure BackpropagateAtOutputPos(OutputX, OutputY, OutputD: integer); {$IFDEF Release} inline; {$ENDIF}
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      procedure Compute(); override;
      procedure ComputeCPU();
      procedure Backpropagate(); override;
      procedure BackpropagateCPU();
  end;

  { TNNetLocalProduct }
  // This is an experimental layer. Do not use it yet.
  TNNetLocalProduct = class(TNNetConvolutionBase)
  private
    procedure BackpropagateAtOutputPos(OutputX, OutputY, OutputD: integer); {$IFDEF Release} inline; {$ENDIF}
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    procedure Compute(); override;
    procedure ComputeCPU();
    procedure Backpropagate(); override;
    procedure BackpropagateCPU();
  end;

  { TNNetDeLocalConnect }
  TNNetDeLocalConnect = class(TNNetLocalConnect)
  public
    constructor Create(pNumFeatures, pFeatureSize: integer; pSuppressBias: integer = 0); overload;
  end;

  { TNNetLocalConnectReLU }
  TNNetLocalConnectReLU = class(TNNetLocalConnect)
  public
    constructor Create(pNumFeatures, pFeatureSize, pInputPadding, pStride: integer; pSuppressBias: integer = 0); override;
  end;

  { TNNetDeLocalConnectReLU }
  TNNetDeLocalConnectReLU = class(TNNetLocalConnectReLU)
  public
    constructor Create(pNumFeatures, pFeatureSize: integer; pSuppressBias: integer = 0); overload;
  end;

  { TNNetPoolBase }
  TNNetPoolBase = class(TNNetLayer)
    private
      FInputCopy: TNNetVolume;
      FMaxPosX, FMaxPosY: array of integer;
      FPoolSize, FStride, FPadding: integer;
      FOutputSizeX, FOutputSizeY, FOutputSizeD: integer;
      FInputDivPool: array of integer;

      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
      function CalcOutputSize(pInputSize: integer): integer; virtual;
      procedure BackpropagateDefaultStride();
      procedure BackpropagateWithStride();
      procedure ComputePreviousLayerError(); override;
    public
      constructor Create(pPoolSize: integer; pStride:integer = 0; pPadding: integer = 0); overload;
      destructor Destroy(); override;
      procedure Backpropagate(); override;
    end;

  /// DEFAULT CAI maxpool layer.
  TNNetMaxPool = class(TNNetPoolBase)
    private
      procedure ComputeDefaultStride();
      procedure ComputeWithStride();
    public
      procedure Compute(); override;
  end;

  /// PORTABLE maxpool layer (similar to other APIs)
  TNNetMaxPoolPortable = class(TNNetMaxPool)
    private
      function CalcOutputSize(pInputSize: integer): integer; override;
    public
      procedure Compute(); override;
  end;

  /// Usual minpool layer.
  TNNetMinPool = class(TNNetPoolBase)
    private
      procedure ComputeDefaultStride();
      procedure ComputeWithStride();
    public
      procedure Compute(); override;
  end;

  /// This layer gets the maximum number from the entire channel.
  TNNetMaxChannel = class(TNNetMaxPool)
  private
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    constructor Create(); override;
  end;

  /// This layer gets the manimum number from the entire channel.
  TNNetMinChannel = class(TNNetMinPool)
  private
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    constructor Create(); override;
  end;

  /// Common avgpool layer.
  TNNetAvgPool = class(TNNetMaxPool)
  private
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    constructor Create(pPoolSize: integer); overload;
    procedure Compute(); override;
    procedure Backpropagate(); override;
  end;

  /// This layer averages the entire channel into only one number.
  TNNetAvgChannel = class(TNNetAvgPool)
  private
    procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
  public
    constructor Create(); override;
  end;

  { TNNetDeMaxPool }
  TNNetDeMaxPool = class(TNNetMaxPool)
    private
      FSpacing: integer;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
      function CalcOutputSize(pInputSize: integer) : integer; override;
    public
      constructor Create(pPoolSize: integer; pSpacing: integer = 0); overload;
      procedure Compute(); override;
      procedure Backpropagate(); override;
      procedure ComputePreviousLayerError(); override;
  end;

  /// This is an experimental layer. Do not use it yet.
  TNNetUpsample = class(TNNetDeMaxPool)
    private
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(); override;
      procedure Compute(); override;
      procedure ComputePreviousLayerError(); override;
  end;

  TNNetDeAvgPool = class(TNNetDeMaxPool);

  /// neural network
  TNNet = class(TMObject)
    protected
      FLayers: TNNetLayerList;
      FLearningRate: TNeuralFloat;
      FForwardTime: double;
      FBackwardTime: double;
      //Layer with Max Delta. You can read after calling GetMaxAbsoluteDelta.
      FMaxDeltaLayer: integer;
      {$IFDEF OpenCL}
      FDotProductKernel: TDotProductKernel;
      {$ENDIF}
    public
      constructor Create(); override;
      destructor Destroy(); override;

      function CreateLayer(strData: string): TNNetLayer;
      function AddLayer(pLayer: TNNetLayer): TNNetLayer; overload;
      function AddLayer(strData: string): TNNetLayer; overload;
      function AddLayer(pLayers: array of TNNetLayer): TNNetLayer; overload;
      function AddLayerAfter(pLayer, pAfterLayer: TNNetLayer): TNNetLayer; overload;
      function AddLayerAfter(pLayer: TNNetLayer; pAfterLayerIdx: integer): TNNetLayer; overload;
      function AddLayerAfter(strData: string; pAfterLayerIdx: integer): TNNetLayer; overload;
      function AddLayerAfter(pLayers: array of TNNetLayer; pLayer: TNNetLayer): TNNetLayer; overload;
      function AddLayerAfter(pLayers: array of TNNetLayer; pAfterLayerIdx: integer): TNNetLayer; overload;
      function AddLayerConcatingInputOutput(pLayers: array of TNNetLayer): TNNetLayer; overload;
      function AddLayerConcatingInputOutput(pLayer: TNNetLayer): TNNetLayer; overload;
      function AddLayerDeepConcatingInputOutput(pLayers: array of TNNetLayer): TNNetLayer; overload;
      function AddLayerDeepConcatingInputOutput(pLayer: TNNetLayer): TNNetLayer; overload;
      // Adds a separable convolution.
      function AddSeparableConv(pNumFeatures{filters}, pFeatureSize, pInputPadding, pStride: integer; pDepthMultiplier: integer = 1; pSuppressBias: integer = 0; pAfterLayer: TNNetLayer = nil): TNNetLayer;
      function AddSeparableConvReLU(pNumFeatures{filters}, pFeatureSize, pInputPadding, pStride: integer; pDepthMultiplier: integer = 1; pSuppressBias: integer = 0; pAfterLayer: TNNetLayer = nil): TNNetLayer;
      function AddSeparableConvLinear(pNumFeatures{filters}, pFeatureSize, pInputPadding, pStride: integer; pDepthMultiplier: integer = 1; pSuppressBias: integer = 0; pAfterLayer: TNNetLayer = nil): TNNetLayer;
      function AddGroupedConvolution(Conv2d: TNNetConvolutionClass;
        Groups, pNumFeatures, pFeatureSize, pInputPadding, pStride: integer;
        pSuppressBias: integer = 0;
        ChannelInterleaving: boolean = True): TNNetLayer;
      /// AddAutoGroupedPointwiseConv implements
      // pointwise convolutions of the kEffNet architecture
      // described on the paper: "Grouped Pointwise Convolutions Significantly
      // Reduces Parameters in EfficientNet" by Joao Paulo Schwarz Schuler,
      // Santiago Romani, Mohamed Abdel-Nasser and Hatem Rashwan.
      function AddAutoGroupedPointwiseConv(
        Conv2d: TNNetGroupedPointwiseConvClass;
        MinChannelsPerGroupCount, pNumFeatures: integer;
        HasNormalization: boolean;
        pSuppressBias: integer = 0;
        HasIntergroup: boolean = true
        ): TNNetLayer;
      function AddAutoGroupedPointwiseConv2(
        Conv2d: TNNetGroupedPointwiseConvClass;
        MinChannelsPerGroupCount, pNumFeatures: integer;
        HasNormalization: boolean;
        pSuppressBias: integer = 0;
        AlwaysIntergroup: boolean = true;
        HasIntergroup: boolean = true
        ): TNNetLayer;
      function AddAutoGroupedConvolution(Conv2d: TNNetConvolutionClass;
        MinChannelsPerGroupCount, pNumFeatures, pFeatureSize, pInputPadding, pStride: integer;
        pSuppressBias: integer = 0;
        ChannelInterleaving: boolean = True): TNNetLayer;
      function AddGroupedFullConnect(FullConnect: TNNetFullConnectClass;
        Groups, pNumFeatures: integer; pSuppressBias: integer = 0;
        ChannelInterleaving: boolean = True): TNNetLayer;
      /// Instead of a batch normalization (or a normalization per batch),
      // this is a moving normalization. It contains zero centering, std. norm.,
      // multiplication and summation. All parameters are trainable. PerCell
      // parameter enables a PerCell bias and multiplication (amplification).
      function AddMovingNorm(PerCell: boolean = false; pAfterLayer: TNNetLayer = nil): TNNetLayer; overload;
      /// This AddMovingNorm implementation adds some randomness simulating what
      // a per batch calculation does. Consider using this normalization as an
      // optimized replacement for batch normalization.
      function AddMovingNorm(PerCell: boolean; RandomBias, RandomAmplifier: integer; pAfterLayer: TNNetLayer = nil): TNNetLayer; overload;
      function AddChannelMovingNorm(PerCell: boolean; RandomBias, RandomAmplifier: integer; pAfterLayer: TNNetLayer = nil): TNNetLayer;
      /// Adds a convolution or a separable convolution. It may also add a
      // ReLU and a ChannelMovingNorm.
      function AddConvOrSeparableConv(IsSeparable, HasReLU, HasNorm: boolean;
        pNumFeatures{filters}, pFeatureSize, pInputPadding, pStride: integer;
        PerCell: boolean = false; pSuppressBias: integer = 0;
        RandomBias: integer = 1; RandomAmplifier: integer = 1;
        pAfterLayer: TNNetLayer = nil): TNNetLayer; overload;
      function AddConvOrSeparableConv(IsSeparable: boolean;
        pNumFeatures{filters}, pFeatureSize, pInputPadding, pStride: integer;
        pSuppressBias: integer = 0;
        pActFn: TNNetActivationFunctionClass = nil;
        pAfterLayer: TNNetLayer = nil): TNNetLayer; overload;
      function AddCompression(Compression: TNeuralFloat = 0.5; supressBias: integer = 1): TNNetLayer;
      function AddGroupedCompression(Compression: TNeuralFloat = 0.5;
        MinGroupSize:integer = 32; supressBias: integer = 1;
        HasIntergroup: boolean = true): TNNetLayer;
      /// This function does both max and min pools and then concatenates results.
      function AddMinMaxPool(pPoolSize: integer; pStride:integer = 0; pPadding: integer = 0): TNNetLayer;
      function AddAvgMaxPool(pPoolSize: integer; pMaxPoolDropout: TNeuralFloat = 0; pKeepDepth:boolean = false; pAfterLayer: TNNetLayer = nil): TNNetLayer;
      function AddMinMaxChannel(pAfterLayer: TNNetLayer = nil): TNNetLayer;
      function AddAvgMaxChannel(pMaxPoolDropout: TNeuralFloat = 0; pKeepDepth:boolean = false; pAfterLayer: TNNetLayer = nil): TNNetLayer;
      procedure AddToExponentialWeightAverage(NewElement: TNNet; Decay: TNeuralFloat);
      procedure AddToWeightAverage(NewElement: TNNet; CurrentElementCount: integer);
      function GetFirstNeuronalLayerIdx(FromLayerIdx:integer = 0): integer; {$IFDEF Release} inline; {$ENDIF}
      function GetFirstImageNeuronalLayerIdx(FromLayerIdx:integer = 0): integer; {$IFDEF Release} inline; {$ENDIF}
      function GetFirstNeuronalLayerIdxWithChannels(FromLayerIdx, Channels:integer): integer; {$IFDEF Release} inline; {$ENDIF}
      function GetLastLayerIdx(): integer; {$IFDEF Release} inline; {$ENDIF}
      function GetLastLayer(): TNNetLayer;
      function GetRandomLayer(): TNNetLayer;
      procedure Compute(pInput, pOutput: TNNetVolumeList; FromLayerIdx:integer = 0); overload;
      procedure Compute(pInput, pOutput: TNNetVolume; FromLayerIdx:integer = 0); overload;
      procedure Compute(pInput: TNNetVolume; FromLayerIdx:integer = 0); overload;
      procedure Compute(pInput: array of TNNetVolume); overload;
      procedure Compute(pInput: array of TNeuralFloatDynArr); overload;
      procedure Compute(pInput: array of TNeuralFloat; FromLayerIdx:integer = 0); overload;
      procedure Backpropagate(pOutput: TNNetVolume); overload;
      procedure BackpropagateForIdx(pOutput: TNNetVolume; const aIdx: array of integer);
      procedure BackpropagateFromLayerAndNeuron(LayerIdx, NeuronIdx: integer; Error: TNeuralFloat);
      procedure Backpropagate(pOutput: array of TNeuralFloat); overload;
      procedure GetOutput(pOutput: TNNetVolume);
      procedure AddOutput(pOutput: TNNetVolume); {$IFDEF Release} inline; {$ENDIF}
      procedure SetActivationFn(ActFn, ActFnDeriv: TNeuralActivationFunction);
      procedure SetLearningRate(pLearningRate, pInertia: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure SetBatchUpdate(pBatchUpdate: boolean); {$IFDEF Release} inline; {$ENDIF}
      procedure InitWeights();
      procedure UpdateWeights(); {$IFDEF Release} inline; {$ENDIF}
      procedure ClearDeltas(); {$IFDEF Release} inline; {$ENDIF}
      procedure ResetBackpropCallCurrCnt(); {$IFDEF Release} inline; {$ENDIF}
      procedure SetL2Decay(pL2Decay: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure SetL2DecayToConvolutionalLayers(pL2Decay: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure ComputeL2Decay(); {$IFDEF Release} inline; {$ENDIF}
      procedure SetSmoothErrorPropagation(p:boolean); {$IFDEF Release} inline; {$ENDIF}
      procedure ClearTime(); {$IFDEF Release} inline; {$ENDIF}
      procedure Clear();
      procedure IdxsToLayers(aIdx: array of integer; var aL: array of TNNetLayer);
      procedure EnableDropouts(pFlag: boolean); {$IFDEF Release} inline; {$ENDIF}
      procedure RefreshDropoutMask(); {$IFDEF Release} inline; {$ENDIF}
      procedure MulMulAddWeights(Value1, Value2: TNeuralFloat; Origin: TNNet); {$IFDEF Release} inline; {$ENDIF}
      procedure MulAddWeights(Value: TNeuralFloat; Origin: TNNet); {$IFDEF Release} inline; {$ENDIF}
      procedure MulWeights(V: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure MulDeltas(V: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      procedure SumWeights(Origin: TNNet); {$IFDEF Release} inline; {$ENDIF}
      procedure SumDeltas(Origin: TNNet); {$IFDEF Release} inline; {$ENDIF}
      procedure SumDeltasNoChecks(Origin: TNNet); {$IFDEF Release} inline; {$ENDIF}
      procedure CopyWeights(Origin: TNNet); {$IFDEF Release} inline; {$ENDIF}
      function ForceMaxAbsoluteDelta(vMax: TNeuralFloat = 0.01): TNeuralFloat;
      function ForceMaxAbsoluteWeight(vMax: TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      function GetMaxAbsoluteDelta(): TNeuralFloat;
      function NormalizeMaxAbsoluteDelta(NewMax: TNeuralFloat = 0.1): TNeuralFloat;
      procedure ClearInertia(); {$IFDEF Release} inline; {$ENDIF}

      {$IFDEF OpenCL}
      procedure DisableOpenCL();
      procedure EnableOpenCL(platform_id: cl_platform_id; device_id: cl_device_id);
      {$ENDIF}

      // debug procedures
      procedure DebugWeights();
      procedure DebugErrors();
      procedure DebugStructure();

      // count/debug functions
      function CountLayers(): integer;
      function CountNeurons(): integer;
      function CountWeights(): integer;
      function GetWeightSum(): TNeuralFloat;
      function GetBiasSum(): TNeuralFloat;

      // load and save functions
      // Save weights to string
      function SaveDataToString(): string;
      // Load weights from string
      procedure LoadDataFromString(strData: string);
      // Load weights from file
      procedure LoadDataFromFile(filename: string);

      // Save architecture to string
      function SaveStructureToString(): string;
      // Load architecture from string
      procedure LoadStructureFromString(strData: string);

      // Save both architecture and weights to string (complete saving).
      function SaveToString(): string;
      // Save both architecture and weights to file (complete saving).
      procedure SaveToFile(filename: string);

      // Save both architecture and weights from string (complete saving).
      procedure LoadFromString(strData: string);
      // Load both architecture and weights from file (complete saving).
      procedure LoadFromFile(filename: string);

      // Returns a cloned neural network
      function Clone(): TNNet;

      // deprecated
      procedure MulWeightsGlorotBengio(V:TNeuralFloat); deprecated;
      procedure MulWeightsHe(V:TNeuralFloat); deprecated;

      // custom layers support
      function ShouldIncDepartingBranchesCnt(pLayer: TNNetLayer):boolean; virtual;

    published
      property BackwardTime: double read FBackwardTime write FBackwardTime;
      property ForwardTime: double read FForwardTime write FForwardTime;
      property Layers: TNNetLayerList read FLayers;
      property LearningRate: TNeuralFloat read FLearningRate;
      property MaxDeltaLayer: integer read FMaxDeltaLayer;
  end;

  { THistoricalNets }
  THistoricalNets = class(TNNet)
    public
      procedure AddLeCunLeNet5(IncludeInput: boolean);
      procedure AddAlexNet(IncludeInput: boolean);
      procedure AddVGGNet(IncludeInput: boolean);
      procedure AddResNetUnit(pNeurons: integer);
      function AddDenseNetBlock(pUnits, k: integer;
        BottleNeck: integer = 0;
        supressBias: integer = 1;
        DropoutRate: TNeuralFloat = 0.0
        ): TNNetLayer;
      function AddDenseNetTransition(Compression: TNeuralFloat = 0.5;
        supressBias: integer = 1;
        HasAvgPool: boolean = true): TNNetLayer;
      function AddDenseNetBlockCAI(pUnits, k, supressBias: integer;
        PointWiseConv: TNNetConvolutionClass {= TNNetConvolutionLinear};
        IsSeparable: boolean = false;
        HasNorm: boolean = true;
        pBefore: TNNetLayerClass = nil;
        pAfter: TNNetLayerClass = nil;
        BottleNeck: integer = 0;
        Compression: integer = 1; // Compression factor. 2 means taking half of channels.
        DropoutRate: TNeuralFloat = 0;
        RandomBias: integer = 1; RandomAmplifier: integer = 1;
        FeatureSize: integer = 3
        ): TNNetLayer; overload;
      function AddDenseNetBlockCAI(pUnits, k, supressBias: integer;
        PointWiseConv: TNNetConvolutionClass {= TNNetConvolutionLinear};
        IsSeparable: boolean = false;
        HasNorm: boolean = true;
        pBeforeBottleNeck: TNNetLayerClass = nil;
        pAfterBottleNeck: TNNetLayerClass = nil;
        pBeforeConv: TNNetLayerClass = nil;
        pAfterConv: TNNetLayerClass = nil;
        BottleNeck: integer = 0;
        Compression: integer = 1; // Compression factor. 2 means taking half of channels.
        DropoutRate: TNeuralFloat = 0;
        RandomBias: integer = 1; RandomAmplifier: integer = 1;
        FeatureSize: integer = 3
        ): TNNetLayer; overload;
      function AddkDenseNetBlock(pUnits, k, supressBias: integer;
        PointWiseConv: TNNetGroupedPointwiseConvClass;
        IsSeparable: boolean = false;
        HasNorm: boolean = true;
        pBeforeBottleNeck: TNNetLayerClass = nil;
        pAfterBottleNeck: TNNetLayerClass = nil;
        pBeforeConv: TNNetLayerClass = nil;
        pAfterConv: TNNetLayerClass = nil;
        BottleNeck: integer = 0;
        Compression: integer = 1; // Compression factor. 2 means taking half of channels.
        DropoutRate: TNeuralFloat = 0;
        RandomBias: integer = 1; RandomAmplifier: integer = 1;
        FeatureSize: integer = 3;
        MinGroupSize: integer = 32
        ): TNNetLayer; overload;
      function AddParallelConvs(
        PointWiseConv: TNNetConvolutionClass {= TNNetConvolutionLinear};
        IsSeparable: boolean = false;
        CopyInput: boolean = false;
        pBeforeBottleNeck: TNNetLayerClass = nil;
        pAfterBottleNeck: TNNetLayerClass = nil;
        pBeforeConv: TNNetLayerClass = nil;
        pAfterConv: TNNetLayerClass = nil;
        PreviousLayer: TNNetLayer = nil;
        BottleNeck: integer = 16;
        p11ConvCount: integer = 4;
        p11FilterCount: integer = 16;
        p33ConvCount: integer = 4;
        p33FilterCount: integer = 16;
        p55ConvCount: integer = 4;
        p55FilterCount: integer = 0;
        p77ConvCount: integer = 4;
        p77FilterCount: integer = 0;
        maxPool: integer = 0;
        minPool: integer = 0
        ): TNNetLayer;
      function AddDenseFullyConnected(pUnits, k, supressBias: integer;
        PointWiseConv: TNNetConvolutionClass {= TNNetConvolutionLinear};
        HasNorm: boolean = true;
        HasReLU: boolean = true;
        pBefore: TNNetLayerClass = nil;
        pAfter: TNNetLayerClass = nil;
        BottleNeck: integer = 0;
        Compression: TNeuralFloat = 1
        ): TNNetLayer;
      function AddSuperResolution(pSizeX, pSizeY, BottleNeck, pNeurons,
        pLayerCnt: integer; IsSeparable:boolean): TNNetLayer;
  end;

  { TNNetDataParallelism }
  {$IFDEF FPC}
  TNNetDataParallelism = class (specialize TFPGObjectList<TNNet>)
  {$ELSE}
  TNNetDataParallelism = class (TNNetList)
    private
      function GetItem(Index: Integer): TNNet; inline;
      procedure SetItem(Index: Integer; AObject: TNNet); inline;
    public
      property Items[Index: Integer]: TNNet read GetItem write SetItem; default;
  {$ENDIF}
    public
      constructor Create(CloneNN: TNNet; pSize: integer; pFreeObjects: Boolean = True); {$IFNDEF FPC} overload; {$ENDIF}
      constructor Create(pSize: integer; pFreeObjects: Boolean = True); {$IFNDEF FPC} overload; {$ENDIF}

      procedure SetLearningRate(pLearningRate, pInertia: TNeuralFloat);
      procedure SetBatchUpdate(pBatchUpdate: boolean);
      procedure SetL2Decay(pL2Decay: TNeuralFloat);
      procedure SetL2DecayToConvolutionalLayers(pL2Decay: TNeuralFloat);
      procedure EnableDropouts(pFlag: boolean);

      procedure CopyWeights(Origin: TNNet); {$IFDEF Release} inline; {$ENDIF}
      procedure SumWeights(Destin: TNNet);  {$IFDEF Release} inline; {$ENDIF}
      procedure SumDeltas(Destin: TNNet);  {$IFDEF Release} inline; {$ENDIF}
      procedure AvgWeights(Destin: TNNet);  {$IFDEF Release} inline; {$ENDIF}
      procedure ReplaceAtIdxAndUpdateWeightAvg(Idx: integer; NewNet, AverageNet: TNNet);

      {$IFDEF OpenCL}
      procedure DisableOpenCL();
      procedure EnableOpenCL(platform_id: cl_platform_id; device_id: cl_device_id);
      {$ENDIF}
  end;

  /// This class is experimental - do not use it.
  TNNetByteProcessing = class(TNNetIdentity)
    private
      FByteLearning: TEasyLearnAndPredictClass;
      FByteInput: array of byte;
      FByteOutput: array of byte;
      FByteOutputFound: array of byte;
      FActionBytes: array of byte;
      procedure SetPrevLayer(pPrevLayer: TNNetLayer); override;
    public
      constructor Create(CacheSize, TestCount, OperationCount: integer); overload;
      destructor Destroy; override;
      procedure Compute(); override;
      procedure Backpropagate(); override;
  end;

  // This class is very experimental - do not use it.
  TNNetForByteProcessing = class(TNNet)
    private
      FInput, FOutput: TNNetVolume;
    public
      constructor Create(); override;
      destructor Destroy(); override;

      procedure AddBasicByteProcessingLayers(InputByteCount, OutputByteCount: integer;
        FullyConnectedLayersCnt: integer = 3; NeuronsPerPath: integer = 16);

      procedure Compute(var pInput: array of byte);
      procedure Backpropagate(var pOutput: array of byte);
      procedure GetOutput(var pOutput: array of byte);
  end;

  // This class is very experimental - do not use it.
  TBytePredictionViaNNet = class(TMObject)
  private
    FNN: TNNet;
    FActions, FStates, FPredictedStates, FOutput: TNNetVolume;
    aActions, aCurrentState, aPredictedState: array of byte;
    FCached: boolean;
    FUseCache: boolean;
  public
    FCache: TCacheMem;

    constructor Create(
      pNN: TNNet;
      pActionByteLen{action array size in bytes},
      pStateByteLen{state array size in bytes}: word;
      // the higher the number, more computations are used on each step. If you don't know what number to use, give 40.
      CacheSize: integer
      // replies the same prediction for the same given state. Use false if you aren't sure.
    );
    destructor Destroy(); override;

    // THIS METHOD WILL PREDICT THE NEXT SATE GIVEN AN ARRAY OF ACTIONS AND STATES.
    // You can understand ACTIONS as a kind of "current state".
    // Returned value "predicted states" contains the neural network prediction.
    procedure Predict(var pActions, pCurrentState: array of byte;
      var pPredictedState: array of byte);

    // Call this method to train the neural network so it can learn from the "found state".
    // Call this method and when the state of your environment changes so the neural
    // network can learn how the state changes from time to time.
    function newStateFound(stateFound: array of byte): extended;

    property NN: TNNet read FNN;
  end;

  // This class is very experimental - do not use it.
  TEasyBytePredictionViaNNet = class(TBytePredictionViaNNet)
  public
    constructor Create(
      pActionByteLen {action array size in bytes},
      pStateByteLen{state array size in bytes}: word;
      // false = creates operation/neurons for non zero entries only.
      NumNeurons: integer;
      // the higher the number, more computations are used on each step. If you don't know what number to use, give 40.
      CacheSize: integer
    );
    destructor Destroy(); override;
  end;

  procedure CompareComputing(NN1, NN2: TNNet);
  procedure CompareNNStructure(NN, NN2: TNNet);
  procedure TestConvolutionAPI();
  procedure TestDataParallelism(NN: TNNet);

  {$IFDEF OpenCL}
  procedure TestConvolutionOpenCL(platform_id: cl_platform_id; device_id: cl_device_id);
  procedure TestFullConnectOpenCL(platform_id: cl_platform_id; device_id: cl_device_id);
  {$ENDIF}

  procedure RebuildPatternOnPreviousPatterns
  (
    Calculated: TNNetVolume;
    LocalWeight: TNNetVolume;
    PrevLayer: TNNetNeuronList;
    PrevStride: integer;
    ReLU: boolean = false;
    Threshold: TNeuralFloat = 0.5
  );

  procedure RebuildNeuronListOnPreviousPatterns
  (
    CalculatedLayer: TNNetNeuronList;
    CurrentLayer, PrevLayer: TNNetNeuronList;
    PrevStride: integer;
    ReLU: boolean = false;
    Threshold: TNeuralFloat = 0.5
  );

implementation

procedure RebuildPatternOnPreviousPatterns
(
  Calculated: TNNetVolume;
  LocalWeight: TNNetVolume;
  PrevLayer: TNNetNeuronList;
  PrevStride: integer;
  ReLU: boolean = false;
  Threshold: TNeuralFloat = 0.5
);
var
  SizeX, SizeY, Depth: integer;
  LocalMaxX, LocalMaxY, LocalMaxD: integer;
  LocalCntX, LocalCntY, NeuronIdx: integer;
  LocalMultiplier: TNeuralFloat;
  PrevMaxX, PrevMaxY, PrevMaxD: integer;
  PrevCntX, PrevCntY, PrevCntD: integer;
  PrevWeight: TNNetVolume;
  PrevWeightValue: TNeuralFloat;
  MinWeightAbs: TNeuralFloat;
begin
  Depth := PrevLayer[0].Weights.Depth;
  SizeX :=
    PrevLayer[0].Weights.SizeX +
    ((LocalWeight.SizeX - 1) * PrevStride);
  SizeY :=
    PrevLayer[0].Weights.SizeY +
    ((LocalWeight.SizeY - 1) * PrevStride);
  if PrevLayer.Count <> LocalWeight.Depth then
  begin
    exit;
  end;
  Calculated.ReSize(SizeX, SizeY, Depth);
  Calculated.Fill(0);
  LocalMaxX := LocalWeight.SizeX - 1;
  LocalMaxY := LocalWeight.SizeY - 1;
  LocalMaxD := LocalWeight.Depth - 1;
  MinWeightAbs := LocalWeight.GetMaxAbs() * Threshold;
  // For each current weight
  for LocalCntX := 0 to LocalMaxX do
  begin
    for LocalCntY := 0 to LocalMaxY do
    begin
      for NeuronIdx := 0 to LocalMaxD do
      begin
        LocalMultiplier := LocalWeight[LocalCntX, LocalCntY, NeuronIdx];
        if MinWeightAbs <= Abs(LocalMultiplier) then
        begin
          // Multiply corresponding weight and add to proper position.
          PrevWeight := PrevLayer[NeuronIdx].Weights;
          PrevMaxX := PrevWeight.SizeX - 1;
          PrevMaxY := PrevWeight.SizeY - 1;
          PrevMaxD := PrevWeight.Depth - 1;
          for PrevCntX := 0 to PrevMaxX do
          begin
            for PrevCntY := 0 to PrevMaxY do
            begin
              for PrevCntD := 0 to PrevMaxD do
              begin
                PrevWeightValue := PrevWeight[PrevCntX, PrevCntY, PrevCntD];
                if (PrevWeightValue > 0) or Not(ReLU) then
                Calculated.Add
                (
                  (LocalCntX * PrevStride) + PrevCntX,
                  (LocalCntY * PrevStride) + PrevCntY,
                  PrevCntD,
                  LocalMultiplier * PrevWeightValue
                );
              end;
            end;
          end; // PrevCntX
        end; //if LocalMultiplier > 0
      end;
    end;
  end; // LocalCntX
end;

procedure RebuildNeuronListOnPreviousPatterns
(
  CalculatedLayer: TNNetNeuronList;
  CurrentLayer, PrevLayer: TNNetNeuronList;
  PrevStride: integer;
  ReLU: boolean = false;
  Threshold: TNeuralFloat = 0.5
);
var
  NeuronCnt: integer;
begin
  if CurrentLayer.Count <> CalculatedLayer.Count then
  begin
    WriteLn(
      'Sizes differ. Current layer: ', CurrentLayer.Count,
      ' Calc layer: ', CalculatedLayer.Count
    );
    exit;
  end;

  for NeuronCnt := 0 to CurrentLayer.Count - 1 do
  begin
    RebuildPatternOnPreviousPatterns
    (
     {Calculated=}CalculatedLayer[NeuronCnt].Weights,
     {LocalWeight=}CurrentLayer[NeuronCnt].Weights,
     {PrevLayer=}PrevLayer,
     {PrevStride=}PrevStride,
     {ReLU=}ReLU,
     {Threshold=}Threshold
    );
  end;
end;

{ TNNetScaleLearning }

procedure TNNetScaleLearning.Compute();
begin
  FOutput.CopyNoChecks(FPrevLayer.FOutput);
end;

procedure TNNetScaleLearning.Backpropagate();
var
  StartTime: double;
  MagnitudeDelta: TNeuralFloat;
  Magnitude: TNeuralFloat;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  if FNeurons[0].Weights.FData[1] > 1 then
  begin
    FOutputError.Mul(FNeurons[0].Weights.FData[1]);
  end;
  Magnitude := FOutput.GetMagnitude();
  MagnitudeDelta := (1-Magnitude);
  if (MagnitudeDelta>0) or (FNeurons[0].Weights.FData[1] > 0) then
  begin
    FNeurons[0].FDelta.Add(0,0,1, NeuronForceRange(MagnitudeDelta, FLearningRate*10) );
  end;
  if (not FBatchUpdate) then
  begin
    FNeurons[0].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
  //if Random(100)=0 then WriteLn(MagnitudeDelta:6:4,' - ',FNeurons[0].Weights.FData[1]:6:4);
  FPrevLayer.FOutputError.Add(FOutputError);
  FPrevLayer.Backpropagate();
  FBackwardTime := FBackwardTime + (Now() - StartTime);
end;

procedure TNNetSwish6.Compute();
var
  SizeM1: integer;
  LocalPrevOutput: TNNetVolume;
  OutputCnt: integer;
  StartTime: double;
  PrevValue: TNeuralFloat;
  SigmoidValue: TNeuralFloat;
  OutputValue: TNeuralFloat;
begin
  StartTime := Now();
  LocalPrevOutput := FPrevLayer.Output;
  SizeM1 := LocalPrevOutput.Size - 1;

  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    for OutputCnt := 0 to SizeM1 do
    begin
      PrevValue := LocalPrevOutput.FData[OutputCnt];
      SigmoidValue := 1 / ( 1 + Exp(-PrevValue) );
      OutputValue := PrevValue * SigmoidValue;
      if OutputValue < 6 then
      begin
        FOutput.FData[OutputCnt] := OutputValue;
        FOutputErrorDeriv.FData[OutputCnt] := OutputValue + SigmoidValue * (1-OutputValue);
      end
      else
      begin
        FOutput.FData[OutputCnt] := 6;
        FOutputErrorDeriv.FData[OutputCnt] := 0;
      end;
    end;
  end
  else
  begin
    // can't calculate error on input layers.
    for OutputCnt := 0 to SizeM1 do
    begin
      PrevValue := LocalPrevOutput.FData[OutputCnt];
      FOutput.FData[OutputCnt] := Min(6.0, PrevValue / ( 1 + Exp(-PrevValue) ));
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

constructor TNNetDebug.Create(hasForward, hasBackward: integer);
begin
  inherited Create();
  FStruct[0] := hasForward;
  FStruct[1] := hasBackward;
end;

{ TNNetDebug }
procedure TNNetDebug.Compute();
begin
  inherited Compute();
  if ((FStruct[0]>0) and (Random(1000)=0)) then
  begin
    Write('Forward:');
    FOutput.PrintDebug();
    WriteLn;
  end;
end;

procedure TNNetDebug.Backpropagate();
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if ((FStruct[1]>0) and (Random(1000)=0)) then
  begin
    Write('Backward:');
    FOutputError.PrintDebug();
    WriteLn;
  end;
  inherited Backpropagate();
end;

{ TNNetReLU6 }
constructor TNNetReLU6.Create(Leakiness: integer);
begin
  inherited Create(0, 6, Leakiness);
end;

{ TNNetSwish }

procedure TNNetSwish.Compute();
var
  SizeM1: integer;
  LocalPrevOutput: TNNetVolume;
  OutputCnt: integer;
  StartTime: double;
  PrevValue: TNeuralFloat;
  SigmoidValue: TNeuralFloat;
  OutputValue: TNeuralFloat;
begin
  StartTime := Now();
  LocalPrevOutput := FPrevLayer.Output;
  SizeM1 := LocalPrevOutput.Size - 1;

  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    for OutputCnt := 0 to SizeM1 do
    begin
      PrevValue := LocalPrevOutput.FData[OutputCnt];
      SigmoidValue := 1 / ( 1 + Exp(-PrevValue) );
      OutputValue := PrevValue * SigmoidValue;
      FOutput.FData[OutputCnt] := OutputValue;
      FOutputErrorDeriv.FData[OutputCnt] := OutputValue + SigmoidValue * (1-OutputValue);
    end;
  end
  else
  begin
    // can't calculate error on input layers.
    for OutputCnt := 0 to SizeM1 do
    begin
      PrevValue := LocalPrevOutput.FData[OutputCnt];
      FOutput.FData[OutputCnt] := PrevValue / ( 1 + Exp(-PrevValue) );
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetInterleaveChannels }

procedure TNNetInterleaveChannels.SetPrevLayer(pPrevLayer: TNNetLayer);
var
  CntDepth, MaxDepth: integer;
begin
  inherited SetPrevLayer(pPrevLayer);
  MaxDepth := FOutput.Depth - 1;
  SetLength(ToChannels, FOutput.Depth);
  for CntDepth := 0 to MaxDepth do
  begin
    ToChannels[CntDepth] :=
      (
        ((CntDepth * FStruct[0]) mod FOutput.Depth) +
        (CntDepth * FStruct[0]) div FOutput.Depth
      ) mod FOutput.Depth;
    // Write(CntDepth,':',ToChannels[CntDepth],'    ');
  end;
  // WriteLn();
end;

constructor TNNetInterleaveChannels.Create(StepSize: integer);
begin
  inherited Create();
  FStruct[0] := StepSize;
end;

destructor TNNetInterleaveChannels.Destroy();
begin
  SetLength(ToChannels, 0);
  inherited Destroy();
end;

procedure TNNetInterleaveChannels.Compute();
var
  CntDepth, MaxDepth: integer;
  StartTime: double;
begin
  StartTime := Now();
  MaxDepth := FOutput.Depth - 1;
  for CntDepth := 0 to MaxDepth do
  begin
    FOutput.CopyFromDepthToDepth(FPrevLayer.FOutput,CntDepth,ToChannels[CntDepth]);
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetInterleaveChannels.Backpropagate();
var
  CntDepth, MaxDepth: integer;
  StartTime, LocalNow: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;

  if FPrevLayer.FOutputError.Size = FOutputError.Size then
  begin
    MaxDepth := FOutput.Depth - 1;
    for CntDepth := 0 to MaxDepth do
    begin
      FPrevLayer.OutputError.AddFromDepthToDepth(FOutputError,ToChannels[CntDepth],CntDepth);
    end;
  end;

  LocalNow := Now();
  FBackwardTime := FBackwardTime + (LocalNow - StartTime);
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

{ TNNetGroupedPointwiseConvReLU }

constructor TNNetGroupedPointwiseConvReLU.Create(pNumFeatures,
  pGroups: integer; pSuppressBias: integer);
begin
  inherited Create(pNumFeatures, pGroups, pSuppressBias);
  FActivationFn := @RectifiedLinearUnit;
  FActivationFnDerivative := @RectifiedLinearUnitDerivative;
end;

{ TNNetGroupedPointwiseConvLinear }

constructor TNNetGroupedPointwiseConvLinear.Create(pNumFeatures,
  pGroups: integer; pSuppressBias: integer);
begin
  inherited Create(pNumFeatures, {pFeatureSize=}1, {pInputPadding=}0,
    {pStride=}1, pGroups, pSuppressBias);
end;

{ TNNetGroupedConvolutionReLU }
constructor TNNetGroupedConvolutionReLU.Create(pNumFeatures, pFeatureSize,
  pInputPadding, pStride, pGroups: integer; pSuppressBias: integer);
begin
  inherited Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pGroups, pSuppressBias);
  FActivationFn := @RectifiedLinearUnit;
  FActivationFnDerivative := @RectifiedLinearUnitDerivative;
end;

procedure TNNetGroupedConvolutionLinear.PrepareInputForGroupedConvolutionFast();
var
  OutputCntX, OutputCntY, OutputD: integer;
  MaxX, MaxY: integer;
  ChannelsPerGroup, ChannelsPerGroupSize: integer;
  yCount, xCount, groupCount: integer;
  InputX, InputY: integer;
  RowSize: integer;
  FeatureSizeXYD: integer;
  {$IFDEF AVXANY}
  SourceRawPos, DestRawPos: pointer;
  {$ENDIF}
begin
  if (FPointwise) then
  begin
    // There is nothing to do. YAY!
  end
  else
  begin
    ChannelsPerGroup := FInputCopy.Depth div FStruct[5];
    RowSize := ChannelsPerGroup;
    ChannelsPerGroupSize := ChannelsPerGroup * SizeOf(TNeuralFloat);
    MaxX := FOutput.SizeX - 1;
    MaxY := FOutput.SizeY - 1;
    FeatureSizeXYD := FFeatureSizeX * FFeatureSizeY * ChannelsPerGroup;
    {$IFDEF Debug}
    if FeatureSizeXYD <> FArrNeurons[0].Weights.Size then
    begin
      FErrorProc('TNNetGroupedConvolutionLinear weight size is incorrect:' +
        IntToStr(FArrNeurons[0].Weights.Size) +
        '. Should be:' +
        IntToStr(FeatureSizeXYD)+'.');
    end;
    {$ENDIF}
    FInputPrepared.ReSize(FOutput.SizeX, FOutput.SizeY, FInputCopy.Depth * FFeatureSizeX * FFeatureSizeY);

    for OutputCntX := 0 to MaxX do
    begin
      for OutputCntY := 0 to MaxY do
      begin
        for yCount := 0 to FFeatureSizeYMinus1 do
        begin
          InputY := OutputCntY * FStride + yCount;
          for xCount := 0 to FFeatureSizeXMinus1 do
          begin
            InputX := OutputCntX * FStride + xCount;
            for groupCount := 0 to FStruct[5] - 1 do
            begin
              OutputD := FeatureSizeXYD * groupCount +
                FArrNeurons[0].Weights.GetRawPos(xCount, yCount);
              {$IFDEF AVXANY}
              SourceRawPos := FInputCopy.GetRawPtr(InputX, InputY, ChannelsPerGroup*groupCount);
              DestRawPos := FInputPrepared.GetRawPtr(OutputCntX, OutputCntY, OutputD);
              asm_dword_copy;
              {$ELSE}
              Move(
                FInputCopy.FData[FInputCopy.GetRawPos(InputX, InputY, ChannelsPerGroup*groupCount)],
                FInputPrepared.FData[FInputPrepared.GetRawPos(OutputCntX, OutputCntY, OutputD)],
                ChannelsPerGroupSize
              );
              {$ENDIF}
            end;
          end;
        end;
      end;
    end;
  end;
end;

{ TNNetGroupedConvolutionLinear }
procedure TNNetGroupedConvolutionLinear.ComputeCPU();
begin
  if FNeurons.Count * FVectorSize <> FConcatedWeights.Size then
  begin
    AfterWeightUpdate();
  end;
  FOutputRaw.GroupedDotProductsTiled(FStruct[5], FNeurons.Count,
    FOutputSizeX * FOutputSizeY, FVectorSize, FConcatedWeights, FInputPrepared,
    FTileSizeD, FTileSizeX);
  if FSuppressBias = 0 then FOutputRaw.Add(FBiasOutput);
  ApplyActivationFunctionToOutput();
end;

procedure TNNetGroupedConvolutionLinear.BackpropagateCPU();
var
  OutputX, OutputY, OutputD: integer;
  MaxX, MaxY, MaxD: integer;
  GroupId, GroupDSize, GroupDStart: integer;
  PrevX, PrevY: integer;
  OutputRawPos: integer;
  CanBackpropOnPos: boolean;
  LocalCntY, LocalCntX: integer;
  LocalLearningErrorDeriv: TNeuralFloat;
  LocalOutputErrorDeriv: TNeuralFloat;
  SmoothLocalOutputErrorDeriv: TNeuralFloat;
  LocalWeight, LocalPrevError: TNNetVolume;
  {SrcPtr,} LocalDestPtr: TNeuralFloatArrPtr;
  //SmoothLocalOutputErrorDerivPtr: pointer;
  //PrevNumElements: integer;
  MissedElements: integer;
  //, PrevMissedElements: integer;
  PtrNeuronDelta: TNeuralFloatArrPtr;
  PtrPreparedInput: TNeuralFloatArrPtr;
  //PrevPtrA, PrevPtrB: TNeuralFloatArrPtr;
  NeuronWeights: integer;
  LocalLearningErrorDerivPtr: pointer;
  localNumElements : integer;
  // Tiling
  TileXCnt, TileDCnt: integer;
  StartTileX, EndTileX, StartTileD, EndTileD: integer;
begin
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxD := OutputError.Depth - 1;
  // Debug code: FOutputError.ForceMaxAbs(1);
  GroupDSize := OutputError.Depth div FStruct[5];
  LocalPrevError := FPrevLayer.OutputError;
  //PrevNumElements := (FSizeXDepth div 4) * 4;
  //PrevMissedElements := FSizeXDepth - PrevNumElements;
  NeuronWeights := FArrNeurons[0].Delta.Size;
  //SmoothLocalOutputErrorDerivPtr := Addr(SmoothLocalOutputErrorDeriv);
  LocalLearningErrorDerivPtr := Addr(LocalLearningErrorDeriv);
  localNumElements := (NeuronWeights div 4) * 4;
  MissedElements := NeuronWeights - localNumElements;
  for OutputY := 0 to MaxY do
  begin
    PrevY := (OutputY*FStride)-FPadding;
    for TileXCnt := 0 to FMaxTileX do
    begin
      StartTileX := TileXCnt * FTileSizeX;
      EndTileX := StartTileX + FTileSizeX - 1;
      for TileDCnt := 0 to FMaxTileD do
      begin
        StartTileD := TileDCnt * FTileSizeD;
        EndTileD := StartTileD + FTileSizeD - 1;
        //WriteLn(StartTileX,' ',EndTileX,' - ',StartTileY,' ',EndTileY,' - ',StartTileD,' ',EndTileD);
        begin
          for OutputX := StartTileX to EndTileX do
          begin
            PrevX := (OutputX*FStride)-FPadding;
            CanBackpropOnPos :=
              (PrevX >= 0) and (PrevY >= 0) and
              (PrevX < FMaxPrevX) and
              (PrevY < FMaxPrevY);
            OutputRawPos := FOutputErrorDeriv.GetRawPos(OutputX, OutputY, StartTileD);
            for OutputD := StartTileD to EndTileD do
            begin
              GroupId := FArrGroupId[OutputD];
              GroupDStart := FArrGroupIdStart[OutputD];
              if (FCalculatePrevLayerError and CanBackpropOnPos)
                then LocalDestPtr := LocalPrevError.GetRawPtr(PrevX, PrevY, GroupDStart);
              {$IFDEF FPC}
              if FActivationFn = @RectifiedLinearUnit then
              begin
                if FOutputRaw.FData[OutputRawPos] >= 0 then
                begin
                  LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
                end
                else
                begin
                  LocalOutputErrorDeriv := 0;
                end;
              end
              else if FActivationFn = @Identity then
              begin
                LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
              end
              else
              begin
                LocalOutputErrorDeriv :=
                  FOutputError.FData[OutputRawPos] *
                  FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
              end;
              {$ELSE}
                LocalOutputErrorDeriv :=
                  FOutputError.FData[OutputRawPos] *
                  FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
              {$ENDIF}

              FOutputErrorDeriv.FData[OutputRawPos] := LocalOutputErrorDeriv;
              LocalLearningErrorDeriv := (-FLearningRate) * LocalOutputErrorDeriv;
              if (LocalLearningErrorDeriv <> 0.0) then
              begin
                  PtrPreparedInput := FInputPrepared.GetRawPtr(OutputX, OutputY, GroupDStart);
                  {$IFNDEF AVX64}
                  FArrNeurons[OutputD].Delta.MulAdd(LocalLearningErrorDeriv, PtrPreparedInput);
                  {$ELSE}
                  PtrNeuronDelta := FArrNeurons[OutputD].Delta.DataPtr;
                  asm_avx64_train_neuron
                  {$ENDIF}

                  {$IFDEF FPC}
                  FArrNeurons[OutputD].FBiasDelta += LocalLearningErrorDeriv;
                  {$ELSE}
                  FArrNeurons[OutputD].FBiasDelta :=
                    FArrNeurons[OutputD].FBiasDelta + LocalLearningErrorDeriv;
                  {$ENDIF}

                  if (FCalculatePrevLayerError) then
                  begin
                    LocalWeight := FArrNeurons[OutputD].Weights;
                    if FPointwise then
                    begin
                      LocalPrevError.MulAdd(LocalDestPtr, LocalWeight.DataPtr, LocalOutputErrorDeriv, GroupDSize);
                    end
                    else
                    begin
                      if CanBackpropOnPos then
                      begin
                        SmoothLocalOutputErrorDeriv := LocalOutputErrorDeriv / FLearnSmoothener;
                        for LocalCntY := 0 to FFeatureSizeYMinus1 do
                        for LocalCntX := 0 to FFeatureSizeXMinus1 do
                        begin
                          LocalPrevError.MulAdd
                          (
                            LocalPrevError.GetRawPtr(PrevX + LocalCntX, PrevY + LocalCntY, GroupDStart), //PrevPtrA
                            LocalWeight.GetRawPtr(LocalCntX, LocalCntY), //PrevPtrB
                            SmoothLocalOutputErrorDeriv,
                            GroupDSize
                          );
                        end;
                      end;
                    end;
                  end; // if (FCalculatePrevLayerError)
              end; // (LocalLearningErrorDeriv <> 0.0)
              Inc(OutputRawPos);
            end;
          end;
        end;
      end;
    end;
  end;

  if (not FBatchUpdate) then
  begin
    for OutputD := 0 to MaxD do FArrNeurons[OutputD].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
end;

procedure TNNetGroupedConvolutionLinear.SetPrevLayer(pPrevLayer: TNNetLayer);
var
  GroupDSize: integer;
  OutputD: integer;
  GroupId, GroupDStart: integer;
begin
  inherited SetPrevLayer(pPrevLayer);
  FVectorSize := FFeatureSizeX*FFeatureSizeY*(pPrevLayer.Output.Depth div FStruct[5]);
  FVectorSizeBytes := FVectorSize * SizeOf(TNeuralFloat);
  GroupDSize := pPrevLayer.Output.Depth div FStruct[5];
  SetNumWeightsForAllNeurons(FFeatureSizeX, FFeatureSizeY, GroupDSize);
  InitDefault();
  AfterWeightUpdate();
  SetLength(FArrGroupId, pPrevLayer.Output.Depth);
  SetLength(FArrGroupIdStart, pPrevLayer.Output.Depth);
  for OutputD := 0 to pPrevLayer.Output.Depth - 1 do
  begin
    GroupId := OutputD div GroupDSize;
    GroupDStart := GroupId * GroupDSize;
    FArrGroupId[OutputD] := GroupId;
    FArrGroupIdStart[OutputD] := GroupDStart;
  end;
  FMaxPrevX := 1 + FPrevLayer.FOutput.SizeX - FFeatureSizeX;
  FMaxPrevY := 1 + FPrevLayer.FOutput.SizeY - FFeatureSizeY;
end;

constructor TNNetGroupedConvolutionLinear.Create(pNumFeatures, pFeatureSize,
  pInputPadding, pStride, pGroups: integer; pSuppressBias: integer);
begin
  inherited Create(pNumFeatures, pFeatureSize, pInputPadding,
    pStride, pSuppressBias);
  FStruct[5] := pGroups;
  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;
end;

destructor TNNetGroupedConvolutionLinear.Destroy();
begin
  SetLength(FArrGroupId, 0);
  SetLength(FArrGroupIdStart, 0);
  inherited Destroy();
end;

procedure TNNetGroupedConvolutionLinear.Compute();
var
  StartTime: double;
begin
  if FNeurons.Count > 0 then
  begin
    StartTime := Now();
    RefreshCalculatePrevLayerError();
    if FPadding > 0
      then FInputCopy.CopyPadding(FPrevLayer.Output, FPadding)
      else FInputCopy := FPrevLayer.Output;

    if FSmoothErrorPropagation then
    begin
      FLearnSmoothener := FFeatureSizeX * FFeatureSizeY;
    end
    else
    begin
      FLearnSmoothener := 1;
    end;

    FSizeXDepth := FFeatureSizeX * FInputCopy.Depth div FStruct[5];
    FSizeXDepthBytes := FSizeXDepth * SizeOf(TNeuralFloat);
    FPrevSizeXDepthBytes := FPrevLayer.Output.IncYSizeBytes();

    PrepareInputForGroupedConvolutionFast();

    //{$IFDEF OpenCL}
    //if (Assigned(FDotCL) and FHasOpenCL and FShouldOpenCL) then
    //begin
    //  ComputeOpenCL();
    //end
    //else
    //begin
    //  ComputeOnCPU;
    //end;
    //{$ELSE}
      ComputeCPU;
    //{$ENDIF}
    FForwardTime := FForwardTime + (Now() - StartTime);
  end
  else
  begin
    FErrorProc('Neuronal layer contains no neuron:'+ IntToStr(FNeurons.Count));
  end;
end;

procedure TNNetGroupedConvolutionLinear.Backpropagate();
var
  StartTime, LocalNow: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FNeurons.Count = FOutput.Depth) and (FPrevLayer.Output.Size > 0) then
  begin
    // ComputeErrorDeriv() isn't required as it's done on BackpropagateAtOutputPos
    // ClearDeltas() is not required as it's done in BackpropagateNTL

    BackpropagateCPU();
  end
  else
  begin
    FErrorProc
    (
      'TNNetConvolution.Backpropagate should have same sizes.' +
      'Neurons:' + IntToStr(FNeurons.Count) +
      ' Output Depth:' + IntToStr(FOutput.Depth) +
      ' PrevLayer:' + IntToStr(FPrevLayer.Output.Size)
    );
  end;
  LocalNow := Now();
  {$IFDEF Debug}
  if LocalNow > StartTime + 1000
    then FErrorProc('TNNetConvolution.Backpropagate bad StartTime.');
  {$ENDIF}
  FBackwardTime := FBackwardTime + (LocalNow - StartTime);
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

{ TNNetNegate }
constructor TNNetNegate.Create();
begin
  inherited Create(-1);
end;

{ TNNetMulByConstant }
procedure TNNetMulByConstant.Compute();
begin
  inherited Compute();
  FOutput.Mul(FStruct[0]);
end;

procedure TNNetCellMulByCell.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  FLayerA := pPrevLayer.NN.Layers[FLayerAIdx];
  FLayerB := pPrevLayer.NN.Layers[FLayerBIdx];
  FLayerA.IncDepartingBranchesCnt();
  FLayerB.IncDepartingBranchesCnt();
  inherited SetPrevLayer(FLayerA);
  SetNumWeightsForAllNeurons(1, 1, 1);
  if FLayerA.Output.Size <> FLayerB.Output.Size then
  begin
    FErrorProc('TNNetCellMulByLayer - A size is ' +
      IntToStr(FLayerA.Output.Size) +
      ' does not match B size ' + IntToStr(FLayerB.Output.Size)
    );
  end;
end;

constructor TNNetCellMulByCell.Create(LayerA, LayerB: TNNetLayer);
begin
  Self.Create(LayerA.LayerIdx, LayerB.LayerIdx);
end;

constructor TNNetCellMulByCell.Create(LayerAIdx, LayerBIdx: integer);
begin
  inherited Create();
  FLayerAIdx := LayerAIdx;
  FLayerBIdx := LayerBIdx;
  FStruct[0] := LayerAIdx;
  FStruct[1] := LayerBIdx;
end;

procedure TNNetCellMulByCell.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  {$IFDEF Debug}
  if FLayerA.Output.Size <> FLayerB.Output.Size then
  begin
    FErrorProc('TNNetCellMulByLayer - A size is ' +
      IntToStr(FLayerA.Output.Size) +
      ' does not match B size ' + IntToStr(FLayerB.Output.Size)
    );
  end;
  {$ENDIF}
  FOutput.Mul(FLayerB.Output);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetCellMulByCell.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  // Calculates the FLayerA Error
  FOutputErrorDeriv.Copy(FOutputError);
  FOutputErrorDeriv.Mul(FLayerA.Output);
  FLayerB.OutputError.Add(FOutputErrorDeriv);
  // Calculates the FLayerB Error
  FOutputErrorDeriv.Copy(FOutputError);
  FOutputErrorDeriv.Mul(FLayerB.Output);
  FLayerA.OutputError.Add(FOutputErrorDeriv);
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  FLayerA.Backpropagate();
  FLayerB.Backpropagate();
end;

{ TNNetForByteProcessing }
constructor TNNetForByteProcessing.Create();
begin
  inherited Create();
  FInput := TNNetVolume.Create();
  FOutput := TNNetVolume.Create();
end;

destructor TNNetForByteProcessing.Destroy();
begin
  FOutput.Free;
  FInput.Free;
  inherited Destroy();
end;

procedure TNNetForByteProcessing.AddBasicByteProcessingLayers(InputByteCount,
  OutputByteCount, FullyConnectedLayersCnt, NeuronsPerPath: integer);
var
  BranchCnt: integer;
  BranchEnd: array of TNNetLayer;
  NNetInputLayer: TNNetLayer;
  //ASide, BSide, ASideEnd, BSideEnd, NewBranch, MainTrunk: TNNetLayer;
  LayerCnt: integer;
begin
  SetLength(BranchEnd, OutputByteCount);
  NNetInputLayer := AddLayer( TNNetInput.Create(InputByteCount*8) );
  for BranchCnt := 0 to OutputByteCount - 1 do
  begin
    AddLayerAfter( TNNetFullConnect.Create( NeuronsPerPath ), NNetInputLayer);
    if FullyConnectedLayersCnt > 1 then
    begin
      for LayerCnt := 2 to FullyConnectedLayersCnt do
      begin
        AddLayer( TNNetFullConnect.Create( NeuronsPerPath ) );
        //MainTrunk := GetLastLayer();
        //ASide := AddLayerAfter( TNNetFullConnect.Create( NeuronsPerPath div 2 ), MainTrunk);
        //ASideEnd := AddLayer( TNNetFullConnect.Create( NeuronsPerPath div 2) );
        //BSide := AddLayerAfter( TNNetFullConnect.Create( NeuronsPerPath div 2 ), MainTrunk);
        //BSideEnd := AddLayer( TNNetFullConnect.Create( NeuronsPerPath div 2) );
        //NewBranch := AddLayer( TNNetCellMulByCell.Create(ASide, BSide) ); // AND block
        //AddLayer( TNNetFullConnectDiff.Create(NeuronsPerPath*2) );
        //AddLayer( TNNetReLU.Create() );
        //AddLayer( TNNetConcat.Create([ ASide, BSide, NewBranch] ) ); // GetLastLayer(),
      end;
    end;
    AddLayer( TNNetFullConnect.Create( 8 ) );
    BranchEnd[BranchCnt] := GetLastLayer();
  end;
  AddLayer( TNNetConcat.Create(BranchEnd) );
  SetLearningRate(0.01, 0.0);
  SetL2Decay(0.0);
  SetLength(BranchEnd, 0);
end;

procedure TNNetForByteProcessing.Compute(var pInput: array of byte);
begin
  FInput.CopyAsBits(pInput, -0.5, +0.5);
  inherited Compute(FInput);
end;

procedure TNNetForByteProcessing.Backpropagate(var pOutput: array of byte);
begin
  FOutput.CopyAsBits(pOutput);
  inherited Backpropagate(FOutput);
end;

procedure TNNetForByteProcessing.GetOutput(var pOutput: array of byte);
begin
  inherited GetOutput(FOutput);
  FOutput.ReadAsBits(pOutput);
end;


{ TNNetByteProcessing }

procedure TNNetByteProcessing.SetPrevLayer(pPrevLayer: TNNetLayer);
var
  ByteLen: integer;
begin
  inherited SetPrevLayer(pPrevLayer);
  ByteLen := pPrevLayer.Output.Size div 8;
  if (pPrevLayer.Output.Size mod 8 > 0) then Inc(ByteLen);
  FByteLearning.Initiate(ByteLen, ByteLen, False{pFullEqual},
    FStruct[1]{relationTableSize}, FStruct[2]{pNumberOfSearches},
    (FStruct[0]>0){pUseCache}, FStruct[0]{CacheSize});
  FOutput.Resize(1, 1, ByteLen*8);
  FOutputError.Resize(FOutput);
  SetLength(FByteInput, ByteLen);
  SetLength(FByteOutput, ByteLen);
  SetLength(FByteOutputFound, ByteLen);
  SetLength(FActionBytes, 0);
  FByteLearning.BytePred.FUseBelief := True;
  FByteLearning.BytePred.FGeneralize := True;
end;

constructor TNNetByteProcessing.Create(CacheSize, TestCount,
  OperationCount: integer);
begin
  inherited Create;
  FStruct[0] := CacheSize;
  FStruct[1] := TestCount;
  FStruct[2] := OperationCount;
end;

destructor TNNetByteProcessing.Destroy;
begin
  SetLength(FByteInput, 0);
  SetLength(FByteOutput, 0);
  SetLength(FByteOutputFound, 0);
  inherited Destroy;
end;

procedure TNNetByteProcessing.Compute();
begin
  FPrevLayer.Output.ReadAsBits(FByteInput, 0.0);
  FByteLearning.Predict(FByteInput, FByteInput, FByteOutput);
  FOutput.CopyAsBits(FByteOutput, -0.5, +0.5);
end;

procedure TNNetByteProcessing.Backpropagate();
begin
  FOutputError.Mul(-1);
  FOutputError.Add(FOutput);
  FOutputError.ReadAsBits(FByteOutputFound, 0.0);
  FByteLearning.newStateFound(FByteOutputFound);
  // This layer doesn't backpropagate.
end;

{ TNNetDigital }

constructor TNNetDigital.Create(LowValue, HighValue: integer);
begin
  inherited Create();
  FLowValue := LowValue;
  FHighValue := HighValue;
  FMiddleValue := (LowValue + HighValue) / 2;
  FMiddleDist := (HighValue - LowValue) / 2;
end;

procedure TNNetDigital.Compute();
var
  PosCnt: integer;
  MaxPos: integer;
  Value: TNeuralFloat;
begin
  MaxPos := FPrevLayer.FOutput.Size - 1;

  for PosCnt := 0 to MaxPos do
  begin
    Value := FPrevLayer.FOutput.FData[PosCnt];
    if Value > FMiddleValue
    then FOutput.FData[PosCnt] := FHighValue
    else FOutput.FData[PosCnt] := FLowValue;
  end;
end;

procedure TNNetDigital.Backpropagate();
var
  MaxOutputCnt: integer;
  OutputCnt: integer;
  LocalPrevError: TNNetVolume;
  LocalError, LocalValue: TNeuralFloat;
begin
  LocalPrevError := FPrevLayer.OutputError;

  MaxOutputCnt := FOutput.Size - 1;
  for OutputCnt := 0 to MaxOutputCnt do
  begin
    LocalError := FOutputError.FData[OutputCnt];
    LocalValue := FOutput.FData[OutputCnt];
    if (Abs(LocalError) > FMiddleDist) then
    begin
      if
        ((LocalValue = FHighValue) and (LocalError > 0)) or
        ((LocalValue = FLowValue) and (LocalError < 0)) then
      begin
        LocalPrevError.FData [OutputCnt] :=
          LocalPrevError.FData[OutputCnt] + LocalError;
      end;
    end;
  end;
  FPrevLayer.Backpropagate();
end;

{ TNNetConvolutionSharedWeights }
constructor TNNetConvolutionSharedWeights.Create(LinkedLayer: TNNetLayer);
begin
  if not(LinkedLayer is TNNetConvolution) then
  begin
    FErrorProc('Linked layer to TNNetConvolutionSharedWeights is not a convolution:'+LinkedLayer.ClassName);
  end;
  inherited Create(LinkedLayer.FStruct[0], LinkedLayer.FStruct[1], LinkedLayer.FStruct[2], LinkedLayer.FStruct[3], LinkedLayer.FStruct[4]);
  FStruct[5] := LinkedLayer.LayerIdx;
  FActivationFn := LinkedLayer.FActivationFn;
  FActivationFnDerivative := LinkedLayer.FActivationFnDerivative;
  FLinkedLayer := TNNetConvolution(LinkedLayer);
  // change the local neural list for the remote neural list
  FNeurons.Free;
  FNeurons := LinkedLayer.FNeurons;
  FLinkedNeurons := true;
end;

destructor TNNetConvolutionSharedWeights.Destroy;
begin
  // recreate a new neural list to allow the destroy to work.
  FNeurons := TNNetNeuronList.Create();
  inherited Destroy;
end;

{ TNNetPad }

procedure TNNetPad.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FOutput.ReSize(pPrevLayer.FOutput.SizeX + FPadding*2, pPrevLayer.FOutput.SizeY + FPadding*2, pPrevLayer.FOutput.Depth);
  if (pPrevLayer.FOutputError.Size = pPrevLayer.FOutput.Size) then
  begin
    FOutputError.ReSize(FOutput);
    FOutputErrorDeriv.ReSize(FOutput);
  end;
end;

constructor TNNetPad.Create(Padding: integer);
begin
  inherited Create();
  FStruct[0] := Padding;
  FPadding := Padding;
end;

procedure TNNetPad.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  if
    (FPrevLayer.FOutputError.Size = FPrevLayer.FOutput.Size) and
    (FOutput.Size <> FOutputError.Size)
    then
  begin
    FOutputError.ReSize(FOutput);
    FOutputErrorDeriv.ReSize(FOutput);
  end;
  FOutput.CopyPadding(FPrevLayer.FOutput, FPadding);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetPad.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FPrevLayer.Output.Size > 0) and (FPrevLayer.Output.Size = FPrevLayer.OutputError.Size) then
  begin
    StartTime := Now();
    FPrevLayer.FOutputError.AddArea
    (
      {DestX=}0,
      {DestY=}0,
      {OriginX=}FPadding,
      {OriginY=}FPadding,
      {LenX=}FPrevLayer.OutputError.SizeX,
      {LenY=}FPrevLayer.OutputError.SizeY,
      FOutputError
    );
    FBackwardTime := FBackwardTime + (Now() - StartTime);
  end;
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

{ TNNetPower }

constructor TNNetPower.Create(iPower: integer);
begin
  inherited Create();
  FPower := iPower;
  FStruct[0] := iPower;
end;

procedure TNNetPower.Compute();
var
  SizeM1: integer;
  LocalPrevOutput: TNNetVolume;
  OutputCnt: integer;
  StartTime: double;
begin
  StartTime := Now();
  LocalPrevOutput := FPrevLayer.Output;
  SizeM1 := LocalPrevOutput.Size - 1;

  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    for OutputCnt := 0 to SizeM1 do
    begin
      FOutput.FData[OutputCnt] := Power(LocalPrevOutput.FData[OutputCnt], FPower);
      FOutputErrorDeriv.FData[OutputCnt] := FPower*Power(LocalPrevOutput.FData[OutputCnt], FPower-1);
    end;
  end
  else
  begin
    // can't calculate error on input layers.
    for OutputCnt := 0 to SizeM1 do
    begin
      FOutput.FData[OutputCnt] := Power(LocalPrevOutput.FData[OutputCnt], FPower);
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetReLUSqrt }

procedure TNNetReLUSqrt.Compute();
var
  SizeM1: integer;
  LocalPrevOutput: TNNetVolume;
  OutputCnt: integer;
  StartTime: double;
  VSqrt: TNeuralFloat;
begin
  StartTime := Now();
  LocalPrevOutput := FPrevLayer.Output;
  SizeM1 := LocalPrevOutput.Size - 1;

  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    for OutputCnt := 0 to SizeM1 do
    begin
      if LocalPrevOutput.FData[OutputCnt] > 0 then
      begin
        VSqrt := Sqrt(LocalPrevOutput.FData[OutputCnt]);
        FOutput.FData[OutputCnt] := VSqrt;
        FOutputErrorDeriv.FData[OutputCnt] := 1/(2*VSqrt);
      end
      else
      begin
        FOutput.FData[OutputCnt] := 0;
        FOutputErrorDeriv.FData[OutputCnt] := 0;
      end;
    end;
  end
  else
  begin
    // can't calculate error on input layers.
    for OutputCnt := 0 to SizeM1 do
    begin
      if LocalPrevOutput.FData[OutputCnt]>0 then
      begin
        FOutput.FData[OutputCnt] := Sqrt(LocalPrevOutput.FData[OutputCnt]);
      end
      else
      begin
        FOutput.FData[OutputCnt] := 0;
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetLocalProduct }

procedure TNNetLocalProduct.BackpropagateAtOutputPos(OutputX, OutputY,
  OutputD: integer);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  LocalOutputError: TNeuralFloat;
  LocalPrevError: TNNetVolume;
  OutputIdx: integer;
  OutputSize, Derivative: TNeuralFloat;
  DestPos: integer;
begin
  OutputIdx := FOutput.GetRawPos(OutputX, OutputY, OutputD);
  LocalOutputError := FOutputError.FData[OutputIdx];

  if (LocalOutputError <> 0) and (FOutput.FData[OutputIdx] > 0) then
  begin
    MaxY := FFeatureSizeY - 1;
    MaxX := FFeatureSizeX - 1;
    LocalPrevError := FPrevLayer.OutputError;

    FSmoothErrorPropagation := true;

    if FSmoothErrorPropagation then
    begin
      OutputSize := FFeatureSizeX * FFeatureSizeY * FOutput.Depth;
    end
    else
    begin
      OutputSize := 1;
    end;

    if
      (OutputX >= FPadding) and (OutputY >= FPadding) and
      (OutputX < OutputError.SizeX - FPadding) and
      (OutputY < OutputError.SizeY - FPadding) then
    begin
      for CntY := 0 to MaxY do
      begin
        for CntX := 0 to MaxX do
        begin
          DestPos := LocalPrevError.GetRawPos( (OutputX-FPadding)*FStride, (OutputY-FPadding)*FStride + CntY, CntX) ;
          Derivative := 1; //(FOutput.FData[OutputIdx] / FPrevLayer.FOutput.FData[DestPos]);
          LocalPrevError.FData[DestPos] := LocalPrevError.FData[DestPos] + LocalOutputError*Derivative/OutputSize;
        end;
      end;
    end;
  end;// of if (LocalOutputErrorDeriv <> 0)
end;

procedure TNNetLocalProduct.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
end;

procedure TNNetLocalProduct.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  if FNeurons.Count > 0 then
  begin
    if FPadding > 0
      then FInputCopy.CopyPadding(FPrevLayer.Output, FPadding)
      else FInputCopy := FPrevLayer.Output;

    FInputPrepared.ReSize(FOutput.SizeX, FOutput.SizeY, FInputCopy.Depth * FFeatureSizeX * FFeatureSizeY);
    PrepareInputForConvolutionFast();
    ComputeCPU();
  end
  else
  begin
    FErrorProc('Neuronal layer contains no neuron:'+ IntToStr(FNeurons.Count));
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetLocalProduct.ComputeCPU();
var
  OutputCntX, OutputCntY, OutputCntD: integer;
  InputCntX, InputCntY: integer;
  MaxX, MaxY, MaxD: integer;
  LocalSize: integer;
  PtrA: TNeuralFloatArrPtr;
  OutputIdx: integer;
  Product: TNeuralFloat;
  CntXYD: integer;
begin
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;
  MaxD := FOutput.Depth - 1;

  LocalSize := FFeatureSizeX*FFeatureSizeY*FInputCopy.Depth;
  InputCntX := 0;
  OutputCntX := 0;
  CntXYD := 0;
  while OutputCntX <= MaxX do
  begin
    InputCntY := 0;
    OutputCntY := 0;
    while OutputCntY <= MaxY do
    begin
      OutputCntD := 0;
      PtrA := FInputPrepared.GetRawPtr(OutputCntX, OutputCntY, 0);
      while OutputCntD <= MaxD do
      begin
        OutputIdx := FOutput.GetRawPos(OutputCntX, OutputCntY, OutputCntD);

        Product := TNNetVolume.Product(PtrA, LocalSize);

        FOutputRaw.FData[OutputIdx] := Product;
        FOutput.FData[OutputIdx] := Product;
        Inc(OutputCntD);
        Inc(CntXYD);
      end;
      Inc(InputCntY, FStride);
      Inc(OutputCntY);
    end;
    Inc(InputCntX, FStride);
    Inc(OutputCntX);
  end;
  (*
  FInputPrepared.Print();WriteLn();
  FPrevLayer.FOutput.Print();WriteLn();
  FOutput.Print();WriteLn();
  WriteLn();
  *)
end;

procedure TNNetLocalProduct.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FPrevLayer.Output.Size > 0) and (FPrevLayer.Output.Size = FPrevLayer.OutputError.Size) then
  begin
    StartTime := Now();
    BackpropagateCPU();
    FBackwardTime := FBackwardTime + (Now() - StartTime);
    {$IFDEF CheckRange}ForceRangeWeights(1000);{$ENDIF}
  end;
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

procedure TNNetLocalProduct.BackpropagateCPU();
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
begin
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxD := OutputError.Depth - 1;

  for CntD := 0 to MaxD do
  begin
    for CntY := 0 to MaxY do
    begin
      for CntX := 0 to MaxX do
      begin
        BackpropagateAtOutputPos(CntX, CntY, CntD);
      end;
    end;
  end;
end;

constructor TNNetPointwiseConv.Create(pNumFeatures: integer;
  pSuppressBias: integer);
begin
  inherited Create(pNumFeatures, {pFeatureSize=}1, {pInputPadding=}0, {pStride=}1, pSuppressBias);
end;

function TNNetMaxPoolPortable.CalcOutputSize(pInputSize: integer): integer;
begin
  Result := ( ( pInputSize - FPoolSize + 2*FPadding ) div FStride) + 1;
end;

procedure TNNetMaxPoolPortable.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  Output.Fill(-1000000);

  if FPadding > 0
    then FInputCopy.CopyPadding(FPrevLayer.Output, FPadding)
    else FInputCopy := FPrevLayer.Output;

  ComputeWithStride();
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetReLUL }

constructor TNNetReLUL.Create(LowLimit, HighLimit, Leakiness: integer);
begin
  inherited Create();
  FScale := 0.001*Leakiness;
  FHighLimit := HighLimit;
  FLowLimit := LowLimit;
  FStruct[0] := LowLimit;
  FStruct[1] := HighLimit;
  FStruct[2] := Leakiness;
end;

procedure TNNetReLUL.Compute();
var
  SizeM1: integer;
  LocalPrevOutput: TNNetVolume;
  OutputCnt: integer;
  StartTime: double;
  CurrValue: TNeuralFloat;
begin
  StartTime := Now();
  LocalPrevOutput := FPrevLayer.Output;
  SizeM1 := LocalPrevOutput.Size - 1;

  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    for OutputCnt := 0 to SizeM1 do
    begin
      CurrValue := LocalPrevOutput.FData[OutputCnt];
      if (CurrValue > FHighLimit) then
      begin
        FOutput.FData[OutputCnt] := FHighLimit + (CurrValue-FHighLimit) * FScale;
        FOutputErrorDeriv.FData[OutputCnt] := FScale;
      end
      else if (CurrValue > FLowLimit) then
      begin
        FOutput.FData[OutputCnt] := CurrValue;
        FOutputErrorDeriv.FData[OutputCnt] := 1;
      end
      else
      begin
        FOutput.FData[OutputCnt] := FLowLimit + (CurrValue-FLowLimit) * FScale;
        FOutputErrorDeriv.FData[OutputCnt] := FScale;
      end;
    end;
  end
  else
  begin
    // not intended for input
    for OutputCnt := 0 to SizeM1 do
    begin
      FOutput.FData[OutputCnt] := LocalPrevOutput.FData[OutputCnt];
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetUpsample }

procedure TNNetUpsample.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FOutputSizeD := pPrevLayer.Output.Depth div (FPoolSize*FPoolSize);

  FOutput.ReSize(FOutputSizeX, FOutputSizeY, FOutputSizeD);
  FOutputError.ReSize(FOutput);
  FOutputErrorDeriv.ReSize(FOutput);
end;

constructor TNNetUpsample.Create();
begin
  inherited Create(2);
end;

procedure TNNetUpsample.Compute();
var
  CntX, CntY, CntD, OutX, OutY, OutD: integer;
  MaxX, MaxY, MaxD: integer;
  StartTime: double;
begin
  StartTime := Now();
  Output.Fill(0);
  MaxX := FPrevLayer.Output.SizeX - 1;
  MaxY := FPrevLayer.Output.SizeY - 1;
  MaxD := Output.Depth - 1;

  for OutD := 0 to MaxD do
  begin
    CntD := OutD shl 2;
    for CntX := 0 to MaxX do
    begin
      OutX := CntX shl 1;
      for CntY := 0 to MaxY do
      begin
        OutY := CntY shl 1;
        //WriteLn(OutX, ' ', OutY, ' ', OutD, ' ', Output.SizeX,' ', Output.SizeY,' ', Output.Depth);
        Output[OutX  ,OutY  ,OutD] := FPrevLayer.Output[CntX,CntY,CntD];
        Output[OutX+1,OutY  ,OutD] := FPrevLayer.Output[CntX,CntY,CntD+1];
        Output[OutX  ,OutY+1,OutD] := FPrevLayer.Output[CntX,CntY,CntD+2];
        Output[OutX+1,OutY+1,OutD] := FPrevLayer.Output[CntX,CntY,CntD+3];
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetUpsample.ComputePreviousLayerError();
var
  CntX, CntY, CntD, OutX, OutY, OutD: integer;
  MaxX, MaxY, MaxD: integer;
  StartTime: double;
begin
  StartTime := Now();
  MaxX := FPrevLayer.Output.SizeX - 1;
  MaxY := FPrevLayer.Output.SizeY - 1;
  MaxD := Output.Depth - 1;

  for OutD := 0 to MaxD do
  begin
    CntD := OutD shl 2;
    for CntX := 0 to MaxX do
    begin
      OutX := CntX shl 1;
      for CntY := 0 to MaxY do
      begin
        OutY := CntY shl 1;
        //WriteLn(OutX, ' ', OutY, ' ', OutD, ' ', Output.SizeX,' ', Output.SizeY,' ', Output.Depth);
        FPrevLayer.OutputError.Add(CntX,CntY,CntD,  OutputError[OutX  ,OutY  ,OutD]);
        FPrevLayer.OutputError.Add(CntX,CntY,CntD+1,OutputError[OutX+1,OutY  ,OutD]);
        FPrevLayer.OutputError.Add(CntX,CntY,CntD+2,OutputError[OutX  ,OutY+1,OutD]);
        FPrevLayer.OutputError.Add(CntX,CntY,CntD+3,OutputError[OutX+1,OutY+1,OutD]);
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

constructor TNNetHyperbolicTangent.Create();
begin
  inherited Create();
  FActivationFn := @HiperbolicTangent;
  FActivationFnDerivative := @HiperbolicTangentDerivative;
end;

{ TNNetNeuronList }
constructor TNNetNeuronList.CreateWithElements(ElementCount: integer);
var
  I: integer;
begin
  Self.Create();
  for I := 1 to ElementCount do
    Self.Add( TNNetNeuron.Create() );
end;

function TNNetNeuronList.GetMaxWeight(): TNeuralFloat;
var
  Cnt: integer;
  MaxValue: TNeuralFloat;
begin
  if Count > 0 then
  begin
    Result := Self[0].Weights.GetMax();
    if Count > 1 then
    begin
      for Cnt := 0 to Count-1 do
      begin
        MaxValue := Self[Cnt].Weights.GetMax();
        if MaxValue > Result then Result := MaxValue;
      end;
    end;
  end
  else
  begin
    Result := -1;
  end;
end;

function TNNetNeuronList.GetMaxAbsWeight(): TNeuralFloat;
var
  Cnt: integer;
  MaxValue: TNeuralFloat;
begin
  if Count > 0 then
  begin
    Result := Self[0].Weights.GetMaxAbs();
    if Count > 1 then
    begin
      for Cnt := 0 to Count-1 do
      begin
        MaxValue := Self[Cnt].Weights.GetMaxAbs();
        if MaxValue > Result then Result := MaxValue;
      end;
    end;
  end
  else
  begin
    Result := -1;
  end;
end;

function TNNetNeuronList.GetMinWeight(): TNeuralFloat;
var
  Cnt: integer;
  MinValue: TNeuralFloat;
begin
  if Count > 0 then
  begin
    Result := Self[0].Weights.GetMin();
    if Count > 1 then
    begin
      for Cnt := 0 to Count-1 do
      begin
        MinValue := Self[Cnt].Weights.GetMin();
        if MinValue < Result then Result := MinValue;
      end;
    end;
  end
  else
  begin
    Result := -1;
  end;
end;

procedure TNNetNeuronList.InitForDebug();
begin
  if (Count >= 3) and (Self[0].Weights.Depth >= 3)then
  begin
    Self[0].Weights.Fill(0);
    Self[1].Weights.Fill(0);
    Self[2].Weights.Fill(0);
    Self[0].Weights[0,0,0] := 1;
    Self[1].Weights[0,0,1] := 1;
    Self[2].Weights[0,0,2] := 1;
  end;
  if (Count >= 6) and (Self[0].Weights.Depth >= 16) and (Self[0].Weights.SizeX >= 3) then
  begin
    Self[3].Weights.Fill(0);
    Self[4].Weights.Fill(0);
    Self[5].Weights.Fill(0);
    Self[3].Weights[0,0,4] := -1;
    Self[4].Weights[1,0,5] := 1;
    Self[5].Weights[2,0,6] := 1;
  end;
end;

{ TNNetSELU }

constructor TNNetSELU.Create();
begin
  inherited Create;
  FAlpha := 1.6733;
  FScale := 1.0507;
  FThreshold := 0.0;
  FScaleAlpha := FAlpha * FScale;
end;

procedure TNNetSELU.Compute();
var
  SizeM1: integer;
  LocalPrevOutput: TNNetVolume;
  OutputCnt: integer;
  StartTime: double;
  ScaleAlphaExp: TNeuralFloat;
begin
  StartTime := Now();
  LocalPrevOutput := FPrevLayer.Output;
  SizeM1 := LocalPrevOutput.Size - 1;

  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    for OutputCnt := 0 to SizeM1 do
    begin
      if LocalPrevOutput.FData[OutputCnt] >= FThreshold then
      begin
        FOutput.FData[OutputCnt] := FScale * LocalPrevOutput.FData[OutputCnt];
        FOutputErrorDeriv.FData[OutputCnt] := FScale;
      end
      else
      begin
        ScaleAlphaExp := FScaleAlpha * Exp(LocalPrevOutput.FData[OutputCnt]);
        FOutput.FData[OutputCnt] := ScaleAlphaExp - FScaleAlpha;
        FOutputErrorDeriv.FData[OutputCnt] := ScaleAlphaExp;
      end;
    end;
  end
  else
  begin
    // can't calculate error on input layers.
    for OutputCnt := 0 to SizeM1 do
    begin
      if LocalPrevOutput.FData[OutputCnt]>FThreshold then
      begin
        FOutput.FData[OutputCnt] := FScale * LocalPrevOutput.FData[OutputCnt];
      end
      else
      begin
        FOutput.FData[OutputCnt] := FScaleAlpha * Exp(LocalPrevOutput.FData[OutputCnt]) - FScaleAlpha;
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetVeryLeakyReLU }
constructor TNNetVeryLeakyReLU.Create();
begin
  inherited Create();
  FAlpha := 1/3;
end;

{ TNNetLeakyReLU }
constructor TNNetLeakyReLU.Create();
begin
  inherited Create();
  FAlpha := 0.01;
  FThreshold := 0.0;
end;

procedure TNNetLeakyReLU.Compute();
var
  SizeM1: integer;
  LocalPrevOutput: TNNetVolume;
  OutputCnt: integer;
  StartTime: double;
begin
  StartTime := Now();
  LocalPrevOutput := FPrevLayer.Output;
  SizeM1 := LocalPrevOutput.Size - 1;

  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    for OutputCnt := 0 to SizeM1 do
    begin
      if LocalPrevOutput.FData[OutputCnt]>FThreshold then
      begin
        FOutput.FData[OutputCnt] := LocalPrevOutput.FData[OutputCnt];
        FOutputErrorDeriv.FData[OutputCnt] := 1;
      end
      else
      begin
        FOutput.FData[OutputCnt] := FOutput.FData[OutputCnt] * FAlpha;
        FOutputErrorDeriv.FData[OutputCnt] := FAlpha;
      end;
    end;
  end
  else
  begin
    // can't calculate error on input layers.
    for OutputCnt := 0 to SizeM1 do
    begin
      if LocalPrevOutput.FData[OutputCnt]>FThreshold then
      begin
        FOutput.FData[OutputCnt] := LocalPrevOutput.FData[OutputCnt];
      end
      else
      begin
        FOutput.FData[OutputCnt] := FOutput.FData[OutputCnt] * FAlpha;
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetIdentityWithoutL2.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  // Layers concerned about L2 should collect errors.
  if FOutputError.Size <> FOutput.Size then
  begin
    FOutputError.ReSize(FOutput);
    FOutputErrorDeriv.ReSize(FOutput);
  end;
end;

{ TNNetIdentityWithoutL2 }
procedure TNNetIdentityWithoutL2.ComputeL2Decay();
begin
  // class intended to hide L2.
end;

{ TNNetSplitChannelEvery }
procedure TNNetSplitChannelEvery.SetPrevLayer(pPrevLayer: TNNetLayer);
var
  ChannelCnt, ChannelMax, CurrentChannelCount: integer;
begin
  if Length(FChannels) = 0 then
  begin
    ChannelMax := pPrevLayer.Output.Depth - 1;
    CurrentChannelCount := 0;
    for ChannelCnt := 0 to ChannelMax do
    begin
      if (ChannelCnt + FStruct[1]) mod FStruct[0] = 0 then
      begin
        Inc(CurrentChannelCount);
        SetLength(FChannels, CurrentChannelCount);
        FChannels[CurrentChannelCount - 1] := ChannelCnt;
        //Debug only: Write('[',(CurrentChannelCount - 1),':',ChannelCnt,']');
      end;
    end;
    //FStruct isn't required anymore
    FStruct[0] := 0;
    FStruct[1] := 0;
  end;
  inherited SetPrevLayer(pPrevLayer);
end;

constructor TNNetSplitChannelEvery.Create(GetChannelEvery, ChannelShift: integer);
begin
  inherited Create([]);
  FStruct[0] := GetChannelEvery;
  FStruct[1] := ChannelShift;
end;

constructor TNNetSplitChannelEvery.Create(pChannels: array of integer);
begin
  inherited Create(pChannels);
end;

{ TNNetIdentityWithoutBackprop }
procedure TNNetIdentityWithoutBackprop.Backpropagate();
begin
  // Doesn't backprop any error
  FPrevLayer.Backpropagate();
end;

{ TNNetChannelRandomMulAdd }

constructor TNNetChannelRandomMulAdd.Create(AddRate, MulRate: integer);
begin
  inherited Create();
  FStruct[0] := AddRate;
  FStruct[1] := MulRate;
  FRandomBias := TNNetVolume.Create(1,1,1);
  FRandomMul := TNNetVolume.Create(1,1,1);
end;

destructor TNNetChannelRandomMulAdd.Destroy;
begin
  FRandomBias.Free;
  FRandomMul.Free;
  inherited Destroy;
end;

procedure TNNetChannelRandomMulAdd.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FRandomBias.ReSize(1,1,FOutput.Depth);
  FRandomMul.ReSize(1,1,FOutput.Depth);
end;

procedure TNNetChannelRandomMulAdd.Compute();
var
  StartTime: double;
  MaxDepth, DepthCount: integer;
begin
  StartTime := Now();
  inherited Compute;
  if FEnabled then
  begin
    MaxDepth := FOutput.Depth - 1;
    for DepthCount := 0 to MaxDepth do
    begin
      FRandomMul.FData[DepthCount] := 1 + (Random(200 * FStruct[1] + 1)/100000) - (0.001 * FStruct[1]);
      FRandomBias.FData[DepthCount] := (Random(200 * FStruct[0] + 1)/100000) - (0.001 * FStruct[0]);
    end;
    FOutput.MulChannels(FRandomMul);
    FOutput.AddToChannels(FRandomBias);
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetChannelRandomMulAdd.Backpropagate();
begin
  if FEnabled then FOutput.MulChannels(FRandomMul);
  inherited Backpropagate();
end;

constructor TNNetRandomMulAdd.Create(AddRate, MulRate: integer);
begin
  inherited Create;
  FStruct[0] := AddRate;
  FStruct[1] := MulRate;
end;

{ TNNetRandomMulAdd }
procedure TNNetRandomMulAdd.Compute();
begin
  inherited Compute();
  if FEnabled then
  begin
    FRandomBias := (Random(200 * FStruct[0] + 1)/100000) - (0.001 * FStruct[0]);
    FRandomMul := 1 + (Random(200 * FStruct[1] + 1)/100000) - (0.001 * FStruct[1]);
    FOutput.Mul(FRandomMul);
    FOutput.Add(FRandomBias);
  end;
end;

procedure TNNetRandomMulAdd.Backpropagate();
begin
  if FEnabled then
  begin
    FOutputError.Mul(FRandomMul);
  end;
  inherited Backpropagate();
end;

{ TNNetPointwiseConvLinear }
constructor TNNetPointwiseConvLinear.Create(pNumFeatures: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures, {pFeatureSize=}1, {pInputPadding=}0, {pStride=}1, pSuppressBias);
end;

{ TNNetPointwiseConvReLU }
constructor TNNetPointwiseConvReLU.Create(pNumFeatures: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures, {pFeatureSize=}1, {pInputPadding=}0, {pStride=}1, pSuppressBias);
end;

{ TNNetDepthwiseConvReLU }
constructor TNNetDepthwiseConvReLU.Create(pMultiplier, pFeatureSize,
  pInputPadding, pStride: integer);
begin
  inherited Create(pMultiplier, pFeatureSize, pInputPadding, pStride);
  FActivationFn := @RectifiedLinearUnit;
  FActivationFnDerivative := @RectifiedLinearUnitDerivative;
end;

{ TNNetDepthwiseConvLinear }
constructor TNNetDepthwiseConvLinear.Create(pMultiplier, pFeatureSize,
  pInputPadding, pStride: integer);
begin
  inherited Create(pMultiplier, pFeatureSize, pInputPadding, pStride);
  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;
end;

{ TNNetDepthwiseConv }
procedure TNNetDepthwiseConv.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FOutput.ReSize(FOutputSizeX, FOutputSizeY, pPrevLayer.Output.Depth * FNeurons.Count);
  FOutputRaw.ReSize(FOutput);
  FOutputError.ReSize(FOutput);
  FOutputErrorDeriv.ReSize(FOutput);
  FVectorSize := FFeatureSizeX*FFeatureSizeY*pPrevLayer.Output.Depth;
  FVectorSizeBytes := FVectorSize * SizeOf(TNeuralFloat);
  RefreshNeuronWeightList();
  FShouldConcatWeights := false;
  BuildArrNeurons();
  InitDefault();
end;

procedure TNNetDepthwiseConv.ComputeCPU();
var
  OutputX, OutputY: integer;
  MaxX, MaxY: integer;
  NeuronIdx, MaxNeurons: integer;
begin
  FOutputRaw.Fill(0.0);
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;
  MaxNeurons := FNeurons.Count - 1;
  for OutputX := 0 to MaxX do
  begin
    for OutputY := 0 to MaxY do
    begin
      for NeuronIdx := 0 to MaxNeurons do
      begin
        ComputeCPUAtOutputPos(NeuronIdx, OutputX, OutputY);
      end;
    end;
  end;
  ApplyActivationFunctionToOutput();
  //Write('Raw:');FOutputRaw.PrintDebug();WriteLn;
  //Write('Output:');FOutput.PrintDebug();WriteLn;
end;

procedure TNNetDepthwiseConv.BackpropagateCPU();
var
  CntX, CntY, NeuronIdx: integer;
  MaxX, MaxY, MaxNeuronIdx: integer;
  PrevX, PrevY: integer;
  bCanBackPropagate: boolean;
begin
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxNeuronIdx := FNeurons.Count - 1;
  bCanBackPropagate :=
    (FPrevLayer.OutputError.Depth = FArrNeurons[0].Weights.Depth) and
    (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size);
  for CntY := 0 to MaxY do
  begin
    PrevY := (CntY*FStride)-FPadding;
    for CntX := 0 to MaxX do
    begin
      PrevX := (CntX*FStride)-FPadding;
      for NeuronIdx := 0 to MaxNeuronIdx do
      begin
        BackpropagateAtOutputPos(CntX, CntY, NeuronIdx, PrevX, PrevY, bCanBackPropagate);
      end;
    end;
  end;
  //Write('Error:');FOutputError.PrintDebug();WriteLn;
  //Write('Error Deriv:');FOutputErrorDeriv.PrintDebug();WriteLn;
  if (not FBatchUpdate) then
  begin
    for NeuronIdx := 0 to MaxNeuronIdx do FNeurons[NeuronIdx].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
end;

procedure TNNetDepthwiseConv.BackpropagateCPUFast();
var
  OutputX, OutputY, NeuronIdx: integer;
  MaxX, MaxY, MaxNeuronIdx: integer;
  PrevX, PrevY: integer;
  bCanBackPropagate: boolean;
  LocalPrevError, LocalDelta, LocalWeight: TNNetVolume;
  OutputErrorDerivLearningPtr, OutputErrorDerivPtr: pointer;
  LocalDepth, LocalWeightDepth: integer;
  FeatureCntX, FeatureCntY: integer;
  PrevPCntX, PrevPCntY: integer;
  MaxFeatureX, MaxFeatureY: integer;
  LocalPrevSizeX, LocalPrevSizeY: integer;
  {$IFDEF AVX64}
  PtrA, PtrB, PtrC: TNeuralFloatArrPtr;
  localNumElements, MissedElements: integer;
  {$ENDIF}
begin
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxFeatureX := FFeatureSizeX - 1;
  MaxFeatureY := FFeatureSizeY - 1;
  MaxNeuronIdx := FNeurons.Count - 1;
  LocalPrevError := FPrevLayer.OutputError;
  LocalPrevSizeX := LocalPrevError.SizeX;
  LocalPrevSizeY := LocalPrevError.SizeY;
  bCanBackPropagate :=
    (FPrevLayer.OutputError.Depth = FArrNeurons[0].Weights.Depth) and
    (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size);
  LocalWeightDepth := FArrNeurons[0].Weights.Depth;
  {$IFDEF AVX64}
  MissedElements := LocalWeightDepth and 3;
  localNumElements := LocalWeightDepth xor MissedElements;
  {$ENDIF}
  if MaxNeuronIdx = 0 then
  begin
    NeuronIdx := 0;
    LocalDelta := FArrNeurons[0].Delta;
    LocalWeight := FArrNeurons[0].Weights;
    for OutputY := 0 to MaxY do
    begin
      PrevY := (OutputY*FStride)-FPadding;
      for OutputX := 0 to MaxX do
      begin
        PrevX := (OutputX*FStride)-FPadding;
        OutputErrorDerivLearningPtr := FOutputError.GetRawPtr(OutputX, OutputY);
        OutputErrorDerivPtr := FOutputErrorDeriv.GetRawPtr(OutputX, OutputY);
          {$IFDEF Debug}
          if LocalDelta.Size <> LocalWeight.Size
          then FErrorProc('TNNetDepthwiseConv.BackpropagateAtOutputPos: deltas don''t match at neuron:'+IntToStr(NeuronIdx));
          if FInputCopy.Depth <> LocalWeight.Depth
          then FErrorProc('TNNetDepthwiseConv.BackpropagateAtOutputPos: dephts don''t match at neuron:'+IntToStr(NeuronIdx));
          {$ENDIF}
          for FeatureCntX := 0 to MaxFeatureX do
          begin
            for FeatureCntY := 0 to MaxFeatureY do
            begin
              {$IFNDEF AVX64}
              TNNetVolume.MulAdd
              (
                LocalDelta.GetRawPtr(FeatureCntX, FeatureCntY),
                OutputErrorDerivLearningPtr,
                FInputCopy.GetRawPtr(OutputX + FeatureCntX, OutputY + FeatureCntY),
                LocalWeightDepth
              );
              {$ELSE}
              PtrA := LocalDelta.GetRawPtr(FeatureCntX, FeatureCntY);
              PtrB := OutputErrorDerivLearningPtr;
              PtrC := FInputCopy.GetRawPtr(OutputX + FeatureCntX, OutputY + FeatureCntY);
              asm_avx64_mulladd_ptra_ptrb_ptrc_num;
              {$ENDIF}
              PrevPCntX := PrevX + FeatureCntX;
              PrevPCntY := PrevY + FeatureCntY;
              if
                bCanBackPropagate and
                (PrevPCntX >= 0) and
                (PrevPCntY >= 0) and
                (PrevPCntX < LocalPrevSizeX) and
                (PrevPCntY < LocalPrevSizeY)
              then
              begin
                {$IFNDEF AVX64}
                TNNetVolume.MulAdd
                (
                  LocalPrevError.GetRawPtr(PrevPCntX, PrevPCntY),
                  LocalWeight.GetRawPtr(FeatureCntX, FeatureCntY),
                  OutputErrorDerivPtr,
                  LocalWeightDepth
                );
                {$ELSE}
                PtrA := LocalPrevError.GetRawPtr(PrevPCntX, PrevPCntY);
                PtrB := LocalWeight.GetRawPtr(FeatureCntX, FeatureCntY);
                PtrC := OutputErrorDerivPtr;
                asm_avx64_mulladd_ptra_ptrb_ptrc_num;
                {$ENDIF}
              end;
            end;
          end;
      end;
    end;
  end
  else
  begin
    for OutputY := 0 to MaxY do
    begin
      PrevY := (OutputY*FStride)-FPadding;
      for OutputX := 0 to MaxX do
      begin
        PrevX := (OutputX*FStride)-FPadding;
        for NeuronIdx := 0 to MaxNeuronIdx do
        begin
          LocalDelta := FArrNeurons[NeuronIdx].Delta;
          LocalWeight := FArrNeurons[NeuronIdx].Weights;
          LocalDepth := LocalWeight.Depth * NeuronIdx;
          OutputErrorDerivLearningPtr := FOutputError.GetRawPtr(OutputX, OutputY, LocalDepth);
          OutputErrorDerivPtr := FOutputErrorDeriv.GetRawPtr(OutputX, OutputY, LocalDepth);
          {$IFDEF Debug}
          if LocalDelta.Size <> LocalWeight.Size
          then FErrorProc('TNNetDepthwiseConv.BackpropagateAtOutputPos: deltas don''t match at neuron:'+IntToStr(NeuronIdx));
          if FInputCopy.Depth <> LocalWeight.Depth
          then FErrorProc('TNNetDepthwiseConv.BackpropagateAtOutputPos: dephts don''t match at neuron:'+IntToStr(NeuronIdx));
          {$ENDIF}
          for FeatureCntX := 0 to MaxFeatureX do
          begin
            for FeatureCntY := 0 to MaxFeatureY do
            begin
              TNNetVolume.MulAdd
              (
                LocalDelta.GetRawPtr(FeatureCntX, FeatureCntY),
                OutputErrorDerivLearningPtr,
                FInputCopy.GetRawPtr(OutputX + FeatureCntX, OutputY + FeatureCntY),
                LocalDelta.Depth
              );
              PrevPCntX := PrevX + FeatureCntX;
              PrevPCntY := PrevY + FeatureCntY;
              if
                bCanBackPropagate and
                (PrevPCntX >= 0) and
                (PrevPCntY >= 0) and
                (PrevPCntX < LocalPrevSizeX) and
                (PrevPCntY < LocalPrevSizeY)
              then
              begin
                TNNetVolume.MulAdd
                (
                  LocalPrevError.GetRawPtr(PrevPCntX, PrevPCntY),
                  LocalWeight.GetRawPtr(FeatureCntX, FeatureCntY),
                  OutputErrorDerivPtr,
                  LocalWeight.Depth
                );
              end;
            end;
          end;
        end;
      end;
    end;
  end;

  //Write('Error:');FOutputError.PrintDebug();WriteLn;
  //Write('Error Deriv:');FOutputErrorDeriv.PrintDebug();WriteLn;
  if (not FBatchUpdate) then
  begin
    for NeuronIdx := 0 to MaxNeuronIdx do FNeurons[NeuronIdx].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
end;

procedure TNNetDepthwiseConv.BackpropagateAtOutputPos(OutputX, OutputY,
  NeuronIdx, PrevX, PrevY: integer; bCanBackPropagate: boolean);
var
  LocalPrevError, LocalDelta, LocalWeight: TNNetVolume;
  OutputErrorDerivLearningPtr, OutputErrorDerivPtr: pointer;
  LocalDepth: integer;
  FeatureCntX, FeatureCntY: integer;
  PrevPCntX, PrevPCntY: integer;
  MaxFeatureX, MaxFeatureY: integer;
  LocalPrevSizeX, LocalPrevSizeY: integer;
begin
  MaxFeatureX := FFeatureSizeX - 1;
  MaxFeatureY := FFeatureSizeY - 1;
  LocalDelta := FArrNeurons[NeuronIdx].Delta;
  LocalWeight := FArrNeurons[NeuronIdx].Weights;
  LocalDepth := LocalWeight.Depth * NeuronIdx;
  OutputErrorDerivLearningPtr := FOutputError.GetRawPtr(OutputX, OutputY, LocalDepth);
  OutputErrorDerivPtr := FOutputErrorDeriv.GetRawPtr(OutputX, OutputY, LocalDepth);
  LocalPrevError := FPrevLayer.OutputError;
  LocalPrevSizeX := LocalPrevError.SizeX;
  LocalPrevSizeY := LocalPrevError.SizeY;
  {$IFDEF Debug}
  if LocalDelta.Size <> LocalWeight.Size
  then FErrorProc('TNNetDepthwiseConv.BackpropagateAtOutputPos: deltas don''t match at neuron:'+IntToStr(NeuronIdx));
  if FInputCopy.Depth <> LocalWeight.Depth
  then FErrorProc('TNNetDepthwiseConv.BackpropagateAtOutputPos: dephts don''t match at neuron:'+IntToStr(NeuronIdx));
  {$ENDIF}
  for FeatureCntX := 0 to MaxFeatureX do
  begin
    for FeatureCntY := 0 to MaxFeatureY do
    begin
      TNNetVolume.MulAdd
      (
        LocalDelta.GetRawPtr(FeatureCntX, FeatureCntY),
        OutputErrorDerivLearningPtr,
        FInputCopy.GetRawPtr(OutputX + FeatureCntX, OutputY + FeatureCntY),
        LocalDelta.Depth
      );
      PrevPCntX := PrevX + FeatureCntX;
      PrevPCntY := PrevY + FeatureCntY;
      if
        bCanBackPropagate and
        (PrevPCntX >= 0) and
        (PrevPCntY >= 0) and
        (PrevPCntX < LocalPrevSizeX) and
        (PrevPCntY < LocalPrevSizeY)
      then
      begin
        TNNetVolume.MulAdd
        (
          LocalPrevError.GetRawPtr(PrevPCntX, PrevPCntY),
          LocalWeight.GetRawPtr(FeatureCntX, FeatureCntY),
          OutputErrorDerivPtr,
          LocalWeight.Depth
        );
      end;
    end;
  end;
end;

procedure TNNetDepthwiseConv.ComputeCPUAtOutputPos(NeuronIdx, OutputX, OutputY: integer);
var
  LocalW: TNNetVolume;
  OutputPtr: pointer;
  OutputDepth: integer;
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  InputX, InputY: integer;
  WeightDepth: integer;
begin
  MaxX := FFeatureSizeX - 1;
  MaxY := FFeatureSizeY - 1;
  LocalW := FArrNeurons[NeuronIdx].Weights;
  OutputDepth := LocalW.Depth * NeuronIdx;
  OutputPtr := FOutputRaw.GetRawPtr(OutputX, OutputY, OutputDepth);
  InputX := OutputX * FStride;
  InputY := OutputY * FStride;
  WeightDepth := LocalW.Depth;
  //WriteLn(InputX,' ', InputY,' ',OutputX,' ',OutputY);
  for CntX := 0 to MaxX do
  begin
    for CntY := 0 to MaxY do
    begin
      TNNetVolume.MulAdd(
        OutputPtr,
        FInputCopy.GetRawPtr(InputX + CntX, InputY + CntY),
        LocalW.GetRawPtr(CntX, CntY),
        WeightDepth
      );
    end;
  end;
end;

procedure TNNetDepthwiseConv.ComputeCPUFast();
var
  OutputX, OutputY: integer;
  MaxX, MaxY: integer;
  NeuronIdx, MaxNeurons: integer;
  LocalW: TNNetVolume;
  OutputPtr: pointer;
  OutputDepth: integer;
  CntX, CntY: integer;
  FeatureSizeXM1, FeatureSizeYM1: integer;
  InputX, InputY: integer;
  WeightDepth: integer;
  {$IFDEF AVX64}
  PtrA, PtrB, PtrC: TNeuralFloatArrPtr;
  localNumElements, MissedElements: integer;
  {$ENDIF}
begin
  FOutputRaw.Fill(0.0);
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;
  FeatureSizeXM1 := FFeatureSizeX - 1;
  FeatureSizeYM1 := FFeatureSizeY - 1;
  MaxNeurons := FNeurons.Count - 1;
  WeightDepth := FArrNeurons[0].Weights.Depth;
  {$IFDEF AVX64}
  MissedElements := WeightDepth and 3;
  localNumElements := WeightDepth xor MissedElements;
  {$ENDIF}
  if MaxNeurons = 0 then
  begin
    LocalW := FArrNeurons[0].Weights;
    for OutputX := 0 to MaxX do
    begin
      InputX := OutputX * FStride;
      for OutputY := 0 to MaxY do
      begin
        InputY := OutputY * FStride;
        OutputPtr := FOutputRaw.GetRawPtr(OutputX, OutputY);
        {$IFDEF AVX64}
        PtrA := OutputPtr;
        {$ENDIF}
          //WriteLn(InputX,' ', InputY,' ',OutputX,' ',OutputY);
        for CntX := 0 to FeatureSizeXM1 do
        begin
          for CntY := 0 to FeatureSizeYM1 do
          begin
            {$IFNDEF AVX64}
            TNNetVolume.MulAdd(
              OutputPtr,
              FInputCopy.GetRawPtr(InputX + CntX, InputY + CntY),
              LocalW.GetRawPtr(CntX, CntY),
              WeightDepth
            );
            {$ELSE}
            PtrB := FInputCopy.GetRawPtr(InputX + CntX, InputY + CntY);
            PtrC := LocalW.GetRawPtr(CntX, CntY);
            asm_avx64_mulladd_ptra_ptrb_ptrc_num;
            {$ENDIF}
          end;
        end;
      end;
    end;
  end
  else
  begin
    for OutputX := 0 to MaxX do
    begin
      InputX := OutputX * FStride;
      for OutputY := 0 to MaxY do
      begin
        InputY := OutputY * FStride;
        for NeuronIdx := 0 to MaxNeurons do
        begin
          LocalW := FArrNeurons[NeuronIdx].Weights;
          OutputDepth := LocalW.Depth * NeuronIdx;
          OutputPtr := FOutputRaw.GetRawPtr(OutputX, OutputY, OutputDepth);
          //WriteLn(InputX,' ', InputY,' ',OutputX,' ',OutputY);
          for CntX := 0 to FeatureSizeXM1 do
          begin
            for CntY := 0 to FeatureSizeYM1 do
            begin
              TNNetVolume.MulAdd(
                OutputPtr,
                FInputCopy.GetRawPtr(InputX + CntX, InputY + CntY),
                LocalW.GetRawPtr(CntX, CntY),
                WeightDepth
              );
            end;
          end;
        end;
      end;
    end;
  end;
  ApplyActivationFunctionToOutput();
  //Write('Raw:');FOutputRaw.PrintDebug();WriteLn;
  //Write('Output:');FOutput.PrintDebug();WriteLn;
end;

constructor TNNetDepthwiseConv.Create(pMultiplier, pFeatureSize, pInputPadding,
  pStride: integer);
begin
  inherited Create(pFeatureSize, pInputPadding, pStride);
  AddNeurons(pMultiplier);
  FOutputError.ReSize(1, 1, 1);
  FOutputErrorDeriv.ReSize(1, 1, 1);
  FStruct[0] := pMultiplier;
  FStruct[1] := pFeatureSize;
  FStruct[2] := pInputPadding;
  FStruct[3] := pStride;
  FActivationFn := @HiperbolicTangent;
  FActivationFnDerivative := @HiperbolicTangentDerivative;
end;

procedure TNNetDepthwiseConv.Compute();
var
    StartTime: double;
begin
  if FNeurons.Count > 0 then
  begin
    StartTime := Now();
    RefreshCalculatePrevLayerError();
    if FPadding > 0
      then FInputCopy.CopyPadding(FPrevLayer.Output, FPadding)
      else FInputCopy := FPrevLayer.Output;
    FSizeXDepth := FFeatureSizeX * FInputCopy.Depth;
    FSizeXDepthBytes := FSizeXDepth * SizeOf(TNeuralFloat);
    FPrevSizeXDepthBytes := FPrevLayer.Output.IncYSizeBytes();
    ComputeCPUFast();
    FForwardTime := FForwardTime + (Now() - StartTime);
  end
  else
  begin
    FErrorProc('TNNetDepthwiseConv.Compute - neuronal layer contains no neuron:'+ IntToStr(FNeurons.Count));
  end;
end;

procedure TNNetDepthwiseConv.Backpropagate();
var
  LocalNow, StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FOutput.Depth = FPrevLayer.Output.Depth * FNeurons.Count) then
  begin
    ComputeErrorDeriv();
    /// In this layer, outputerror contains the error times output derivative
    // time -learning rate.
    FOutputError.Copy(FOutputErrorDeriv);
    FOutputError.Mul(-FLearningRate);
    BackpropagateCPUFast();
  end
  else
  begin
    FErrorProc
    (
      'TNNetDepthwiseConv.Backpropagate sizes are wrong.' +
      'Neurons:' + IntToStr(FNeurons.Count) +
      ' Output Depth:' + IntToStr(FOutput.Depth) +
      ' PrevLayer:' + IntToStr(FPrevLayer.Output.Size)
    );
  end;
  LocalNow := Now();
  {$IFDEF Debug}
  if LocalNow > StartTime + 1000
    then FErrorProc('TNNetDepthwiseConv.Backpropagate bad StartTime.');
  {$ENDIF}
  FBackwardTime := FBackwardTime + (LocalNow - StartTime);
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

procedure TNNetDepthwiseConv.InitDefault();
begin
  InitHeUniformDepthwise(1);
end;

{ TNNetChannelMul }
procedure TNNetChannelMul.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  {$IFDEF Debug}
  if FNeurons[0].FWeights.Size <> FOutput.Depth then
  begin
    FErrorProc('Neuron weight count isn''t compatible with output depth ' +
      'at TNNetChannelMul.');
  end;
  {$ENDIF}
  FOutput.MulChannels(FNeurons[0].FWeights);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetChannelMul.Backpropagate();
var
  StartTime: double;
  localNeuron: TNNetNeuron;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  localNeuron := FNeurons[0];
  FOutputErrorDeriv.Fill(0);
  // activation derivative is 1 in this layer.
  FOutputErrorDeriv.MulAdd(-FLearningRate/(FOutputChannelSize), FOutputError);
  FOutputErrorDeriv.Mul(FPrevLayer.Output);
  FAuxDepth.Fill(0);
  FAuxDepth.AddSumChannel(FOutputErrorDeriv);
  {$IFDEF Debug}
  if localNeuron.Delta.Size <> FAuxDepth.Size then
  begin
    FErrorProc('Neuron weight count isn''t compatible with output depth ' +
      'at TNNetChannelMul backprop.');
  end;
  {$ENDIF}
  localNeuron.Delta.Add(FAuxDepth);
  if (not FBatchUpdate) then
  begin
    localNeuron.UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  if Assigned(FPrevLayer) and (FPrevLayer.FOutputError.Size = FOutputError.Size) then
  begin
    FOutputError.MulChannels(localNeuron.FWeights);
    FPrevLayer.FOutputError.Add(FOutputError);
    FPrevLayer.Backpropagate();
  end;
end;

procedure TNNetChannelMul.InitDefault();
begin
  if FNeurons.Count < 1 then AddMissingNeurons(1);
  inherited InitDefault();
  FNeurons[0].Weights.Fill(1);
  AfterWeightUpdate();
end;

{ TNNetCellMul }

procedure TNNetCellMul.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  if FNeurons.Count < 1 then AddMissingNeurons(1);
  SetNumWeightsForAllNeurons(FOutput);
  FOutputError.ReSize(FOutput);
  FOutputErrorDeriv.ReSize(FOutput);
  InitDefault();
end;

procedure TNNetCellMul.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  FOutput.Mul(FNeurons[0].FWeights);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetCellMul.Backpropagate();
var
  StartTime: double;
  localNeuron: TNNetNeuron;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  localNeuron := FNeurons[0];
  FOutputErrorDeriv.Fill(0);
  // activation derivative is 1 in this layer.
  FOutputErrorDeriv.MulAdd(-FLearningRate, FOutputError);
  localNeuron.Delta.MulAdd(FOutputErrorDeriv, FPrevLayer.Output);
  if (not FBatchUpdate) then
  begin
    localNeuron.UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  if Assigned(FPrevLayer) and (FPrevLayer.FOutputError.Size = FOutputError.Size) then
  begin
    // backpropagating error is simple!
    FPrevLayer.FOutputError.MulAdd(FOutputError, localNeuron.FWeights);
    FPrevLayer.Backpropagate();
  end;
end;

procedure TNNetCellMul.InitDefault();
begin
  FNeurons[0].Weights.Fill(1);
  AfterWeightUpdate();
end;

{ TNNetSigmoid }
procedure TNNetSigmoid.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FOutputRaw.Resize(FOutput);
end;

constructor TNNetSigmoid.Create();
begin
  inherited Create();
  FActivationFn := @Sigmoid;
  FActivationFnDerivative := @SigmoidDerivative;
end;

procedure TNNetSigmoid.Compute();
begin
  FOutputRaw.CopyNoChecks(FPrevLayer.FOutput);
  ApplyActivationFunctionToOutput();
end;

procedure TNNetSigmoid.Backpropagate();
begin
  ComputeErrorDeriv();
  FPrevLayer.FOutputError.Add(FOutputErrorDeriv);
  inherited Backpropagate();
end;

{ TNNetFullConnectSigmoid }
constructor TNNetFullConnectSigmoid.Create(pSizeX, pSizeY, pDepth: integer;
  pSuppressBias: integer);
begin
  inherited Create(pSizeX, pSizeY, pDepth, pSuppressBias);
  FActivationFn := @Sigmoid;
  FActivationFnDerivative := @SigmoidDerivative;
end;

constructor TNNetFullConnectSigmoid.Create(pSize: integer;
  pSuppressBias: integer);
begin
  inherited Create(pSize, pSuppressBias);
  FActivationFn := @Sigmoid;
  FActivationFnDerivative := @SigmoidDerivative;
end;

{ TNNetCellBias }
procedure TNNetCellBias.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  if FNeurons.Count < 1 then AddMissingNeurons(1);
  SetNumWeightsForAllNeurons(FOutput);
  FOutputError.ReSize(FOutput);
  InitDefault();
end;

procedure TNNetChannelMulByLayer.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  FLayerWithChannels := pPrevLayer.NN.Layers[FLayerWithChannelsIdx];
  FLayerMul := pPrevLayer.NN.Layers[FLayerMulIdx];
  FLayerWithChannels.IncDepartingBranchesCnt();
  FLayerMul.IncDepartingBranchesCnt();
  inherited SetPrevLayer(FLayerWithChannels);
  SetNumWeightsForAllNeurons(1, 1, 1);
  if FLayerWithChannels.Output.Depth <> FLayerMul.Output.Size then
  begin
    FErrorProc('TNNetChannelMulByLayer - Channels in origin ' +
      IntToStr(FLayerWithChannels.Output.Depth) +
      ' do not match operand size ' + IntToStr(FLayerMul.Output.Size)
    );
  end;
end;

constructor TNNetChannelMulByLayer.Create(LayerWithChannels, LayerMul: TNNetLayer);
begin
  Self.Create(LayerWithChannels.LayerIdx, LayerMul.LayerIdx);
end;

constructor TNNetChannelMulByLayer.Create(LayerWithChannelsIdx, LayerMulIdx: integer);
begin
  inherited Create();
  FLayerWithChannelsIdx := LayerWithChannelsIdx;
  FLayerMulIdx := LayerMulIdx;
  FStruct[0] := LayerWithChannelsIdx;
  FStruct[1] := LayerMulIdx;
end;

procedure TNNetChannelMulByLayer.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  {$IFDEF Debug}
  if FLayerWithChannels.Output.Depth <> FLayerMul.Output.Size then
  begin
    FErrorProc('TNNetChannelMulByLayer - Channels in origin ' +
      IntToStr(FLayerWithChannels.Output.Depth) +
      ' does not match operand size ' + IntToStr(FLayerMul.Output.Size)
    );
  end;
  {$ENDIF}
  FOutput.MulChannels(FLayerMul.Output);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetChannelMulByLayer.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  // Calculates the FLayerWithChannels Error
  FOutputErrorDeriv.Copy(FOutputError);
  FOutputErrorDeriv.MulChannels(FLayerMul.Output);
  FLayerWithChannels.OutputError.Add(FOutputErrorDeriv);
  // Calculates the FLayerMul Error
  FOutputErrorDeriv.Copy(FOutputError);
  FOutputErrorDeriv.Mul(FLayerWithChannels.Output);
  FAuxDepth.Fill(0);
  FAuxDepth.AddSumChannel(FOutputErrorDeriv);
  FAuxDepth.Divi(FOutputChannelSize);
  FLayerMul.OutputError.Add(FAuxDepth);
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  FLayerWithChannels.Backpropagate();
  FLayerMul.Backpropagate();
end;

procedure TNNetCellBias.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  FOutput.Add(FNeurons[0].FWeights);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetCellBias.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  FNeurons[0].FDelta.MulAdd(-FLearningRate, FOutputError);
  if (not FBatchUpdate) then
  begin
    FNeurons[0].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  if Assigned(FPrevLayer) and (FPrevLayer.FOutputError.Size = FOutputError.Size) then
  begin
    FPrevLayer.FOutputError.Add(FOutputError);
    FPrevLayer.Backpropagate();
  end;
end;

procedure TNNetCellBias.InitDefault();
begin
  FNeurons[0].Weights.Fill(0);
  AfterWeightUpdate();
end;

{ TNNetMinChannel }
procedure TNNetMinChannel.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  FPoolSize := pPrevLayer.Output.SizeX;
  FStride := FPoolSize;
  FPadding := 0;

  inherited SetPrevLayer(pPrevLayer);
end;

constructor TNNetMinChannel.Create();
begin
  inherited Create(2);
end;

{ TNNetMinPool }
procedure TNNetMinPool.ComputeDefaultStride();
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  OutX, OutY: integer;
  OutputRawPos: integer;
  InputRawPtr: TNeuralFloatPtr;
begin
  MaxX := FInputCopy.SizeX - 1;
  MaxY := FInputCopy.SizeY - 1;
  MaxD := FInputCopy.Depth - 1;

  for CntY := 0 to MaxY do
  begin
    OutY := FInputDivPool[CntY]; //CntY div FPoolSize;
    for CntX := 0 to MaxX do
    begin
      OutX := FInputDivPool[CntX]; //CntX div FPoolSize;
      OutputRawPos := FOutput.GetRawPos(OutX, OutY);
      InputRawPtr := FInputCopy.GetRawPtr(CntX, CntY);
      for CntD := 0 to MaxD do
      begin
        if InputRawPtr^ < FOutput.FData[OutputRawPos] then
        begin
          FOutput.FData[OutputRawPos] := InputRawPtr^;
          FMaxPosX[OutputRawPos] := CntX;
          FMaxPosY[OutputRawPos] := CntY;
        end;
        Inc(OutputRawPos);
        Inc(InputRawPtr);
      end;
    end;
  end; // of for CntD
end;

procedure TNNetMinPool.ComputeWithStride();
var
  CntOutputX, CntOutputY, CntD: integer;
  OutputMaxX, OutputMaxY, MaxD: integer;
  InX, InY, InXMax, InYMax: integer;
  CntInputPX, CntInputPY: integer;
  OutputRawPos: integer;
  CurrValue: TNeuralFloat;
  LocalPoolSizeM1, InputSizeXM1, InputSizeYM1: integer;
begin
  OutputMaxX := Output.SizeX - 1;
  OutputMaxY := Output.SizeY - 1;
  MaxD := Output.Depth - 1;
  LocalPoolSizeM1 := FPoolSize - 1;
  InputSizeXM1 := FInputCopy.SizeX - 1;
  InputSizeYM1 := FInputCopy.SizeY - 1;

  for CntOutputY := 0 to OutputMaxY do
  begin
    InY := CntOutputY * FStride;
    InYMax := Min(InY + LocalPoolSizeM1, InputSizeYM1);
    for CntOutputX := 0 to OutputMaxX do
    begin
      InX := CntOutputX * FStride;
      InXMax := Min(InX + LocalPoolSizeM1, InputSizeXM1);
      OutputRawPos := Output.GetRawPos(CntOutputX, CntOutputY);
      for CntD := 0 to MaxD do
      begin
        for CntInputPX := InX to InXMax do
        begin
          for CntInputPY := InY to InYMax do
          begin
            CurrValue := FInputCopy[CntInputPX, CntInputPY, CntD];
            if CurrValue < FOutput.FData[OutputRawPos] then
            begin
              FOutput.FData[OutputRawPos] := CurrValue;
              FMaxPosX[OutputRawPos] := CntInputPX;
              FMaxPosY[OutputRawPos] := CntInputPY;
            end;
          end;
        end;
        Inc(OutputRawPos);
      end;
    end;
  end; // of for CntD
end;

procedure TNNetMinPool.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  Output.Fill(1000000);

  if FPadding > 0
    then FInputCopy.CopyPadding(FPrevLayer.Output, FPadding)
    else FInputCopy := FPrevLayer.Output;

  if ((FStride = FPoolSize) and (FPadding = 0)) then
  begin
    ComputeDefaultStride();
  end
  else
  begin
    ComputeWithStride();
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetChannelZeroCenter }
procedure TNNetChannelZeroCenter.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  FAuxDepth.Fill(0);
  FAuxDepth.AddSumChannel(FOutput);
  FNeurons[0].FDelta.MulAdd(-FLearningRate / FOutputChannelSize, FAuxDepth);
  if (not FBatchUpdate) then
  begin
    FNeurons[0].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate();
end;

procedure TNNetChannelZeroCenter.ComputeL2Decay();
begin
  // it makes no sense to compute L2 decay on a normalization layer.
end;

{ TNNetChannelBias }
procedure TNNetChannelTransformBase.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  SetNumWeightsForAllNeurons(1, 1, FOutput.Depth);
  FAuxDepth.ReSize(1,1,FOutput.Depth);
  InitDefault();
  FOutputChannelSize := FOutput.SizeX * FOutput.SizeY;
end;

constructor TNNetChannelTransformBase.Create();
begin
  inherited Create();
  FAuxDepth := TNNetVolume.Create();
  FOutputChannelSize := 1;
  InitDefault();
end;

destructor TNNetChannelTransformBase.Destroy();
begin
  FAuxDepth.Free;
  inherited Destroy();
end;

procedure TNNetChannelShiftBase.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  {$IFDEF Debug}
  if FNeurons[0].FWeights.Size <> FOutput.Depth then
  begin
    FErrorProc('Neuron weight count isn''t compatible with output depth ' +
      'at TNNetChannelShiftBase.');
  end;
  {$ENDIF}
  FOutput.AddToChannels(FNeurons[0].FWeights);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetChannelShiftBase.InitDefault();
begin
  if FNeurons.Count < 1 then AddMissingNeurons(1);
  inherited InitDefault();
  FNeurons[0].Weights.Fill(0);
end;

procedure TNNetChannelBias.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  FAuxDepth.Fill(0);
  FAuxDepth.AddSumChannel(FOutputError);
  FNeurons[0].FDelta.MulAdd(-FLearningRate / FOutputChannelSize, FAuxDepth);
  if (not FBatchUpdate) then
  begin
    FNeurons[0].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate();
end;

{ TNNetChannelStdNormalization }
procedure TNNetChannelStdNormalization.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  SetNumWeightsForAllNeurons(1, 1, FOutput.Depth);
  FAuxOutput.ReSize(FOutput);
  InitDefault();
end;

constructor TNNetChannelStdNormalization.Create();
begin
  inherited Create();
  FAuxOutput := TNNetVolume.Create();
  InitDefault;
end;

destructor TNNetChannelStdNormalization.Destroy();
begin
  FAuxOutput.Free;
  inherited Destroy();
end;

procedure TNNetChannelStdNormalization.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  FOutput.MulChannels(FNeurons[1].FWeights);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetChannelStdNormalization.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  FAuxOutput.Copy(FOutput);
  FAuxOutput.Mul(FAuxOutput);
  FAuxDepth.Fill(0);
  FAuxDepth.AddSumChannel(FAuxOutput);
  FAuxDepth.Divi(FOutputChannelSize);
  FAuxDepth.VSqrt();
  FAuxDepth.Add(-1);
  // The std deviation learning is 100 times slower than zero centering avoiding
  // overflows.
  FNeurons[1].FDelta.MulAdd(-FLearningRate*0.01 / FOutputChannelSize, FAuxDepth);
  if (not FBatchUpdate) then
  begin
    FNeurons[1].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
  if Assigned(FPrevLayer) and
    (FPrevLayer.OutputError.Size > 0) and
    (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size) then
  begin
    FAuxDepth.Copy(FNeurons[1].FWeights);
    // The direction of the error is more important than its magnitude.
    FAuxDepth.SetMin(1);
    FOutputError.MulChannels(FAuxDepth);
  end;
  (*
  if (Random(100)=0) then
  begin
    FNeurons[1].FWeights.PrintDebug();
    Write(' - ', FOutput.GetStdDeviation());
    WriteLn;
  end;
  *)
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate();
end;

procedure TNNetChannelStdNormalization.InitDefault();
begin
  if FNeurons.Count < 2 then AddMissingNeurons(2);
  FNeurons[0].FWeights.Fill(0);
  FNeurons[1].FWeights.Fill(1);
end;

function TNNetChannelStdNormalization.GetMaxAbsoluteDelta(): TNeuralFloat;
begin
  // channel standard normalization has lower impact on deltas.
  Result := inherited GetMaxAbsoluteDelta() * 0.01;
end;

{ TNNetMovingStdNormalization }
constructor TNNetMovingStdNormalization.Create();
begin
  inherited Create();
  InitDefault;
end;

procedure TNNetMovingStdNormalization.Compute();
var
  StartTime: double;
  StdDev: TNeuralFloat;
begin
  StartTime := Now();
  inherited Compute;
  FOutput.Sub(FNeurons[0].FWeights.FData[0]);
  StdDev := FNeurons[0].FWeights.FData[1];
  if (StdDev > 0) and (StdDev<>1) then
  begin
    FOutput.Divi(StdDev);
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetMovingStdNormalization.Backpropagate();
var
  StartTime: double;
  StdDev, CurrentStdDev: TNeuralFloat;
  StdDevError: TNeuralFloat;
  FloatSize: TNeuralFloat;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  StartTime := Now();
  FloatSize := FOutput.Size;
  CurrentStdDev := Sqrt(FOutput.GetSumSqr()/FloatSize);
  StdDevError := (CurrentStdDev - 1);
  StdDev := FNeurons[0].FWeights.FData[1];
  (*if Random(1000) = 0 then
  begin
    WriteLn(' Current StdDev:',CurrentStdDev:8:6,' Current Avg:',FOutput.GetAvg():8:6,' Current StdDevError:',StdDevError:8:6,' Learned StdDev:',StdDev:8:6);
  end;*)
  FNeurons[0].FDelta.Add(0,0,0, FOutput.GetAvg()*FLearningRate );
  FNeurons[0].FDelta.Add(0,0,1, NeuronForceRange(StdDevError*FLearningRate*0.01, FLearningRate) );
  if (not FBatchUpdate) then
  begin
    FNeurons[0].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
  //WriteLn(FOutput.GetAvg(),' ',StdDevError,' ',CurrentStdDev);
  if StdDev > 1 then
  begin
    // The direction of the error is more important than its magnitude.
    FOutputError.Divi(StdDev);
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate();
end;

procedure TNNetMovingStdNormalization.InitDefault();
begin
  if FNeurons.Count < 1 then AddMissingNeurons(1);
  SetNumWeightsForAllNeurons(1, 1, 2);
  FNeurons[0].FWeights.FData[0] := 0;
  FNeurons[0].FWeights.FData[1] := 1;
end;

function TNNetMovingStdNormalization.GetMaxAbsoluteDelta(): TNeuralFloat;
begin
  // channel standard normalization has lower impact on deltas.
  Result := inherited GetMaxAbsoluteDelta() * 0.01;
end;

{ TNNetLayerConcatedWeights }
procedure TNNetLayerConcatedWeights.AfterWeightUpdate();
begin
  inherited AfterWeightUpdate();
  if FNeuronWeightList.Count > 0 then
  begin
    if FShouldConcatWeights then
    begin
      BuildBiasOutput();
      FNeuronWeightList.ConcatInto(FConcatedWeights);
      if FShouldInterleaveWeights then
      begin
        FConcatedWInter.InterleaveWithXFrom(FConcatedWeights, FVectorSize);
      end;
    end;
  end;
  FAfterWeightUpdateHasBeenCalled := true;
end;

procedure TNNetLayerConcatedWeights.BuildBiasOutput();
var
  OutputCntX, OutputCntY: integer;
  MaxX, MaxY: integer;
  NeuronIdx: integer;
  MaxNeurons: integer;
  BiasValue: TNeuralFloatPtr;
begin
  MaxNeurons := FNeurons.Count - 1;
  FBiasOutput.ReSize(FOutputRaw);
  if High(FArrNeurons) < MaxNeurons then BuildArrNeurons();

  if (Self is TNNetConvolution) or (Self is TNNetGroupedConvolutionLinear) then
  begin
    MaxX := FOutput.SizeX - 1;
    MaxY := FOutput.SizeY - 1;

    for OutputCntX := 0 to MaxX do
    begin
      for OutputCntY := 0 to MaxY do
      begin
        BiasValue := FBiasOutput.GetRawPtr(OutputCntX, OutputCntY);
        for NeuronIdx := 0 to MaxNeurons do
        begin
          BiasValue^ := FArrNeurons[NeuronIdx].FBiasWeight;
          Inc(BiasValue);
        end;
      end;
    end;
  end
  else
  if Self is TNNetFullConnect then
  begin
    BiasValue := FBiasOutput.GetRawPtr(0, 0);
    for NeuronIdx := 0 to MaxNeurons do
    begin
      BiasValue^ := FArrNeurons[NeuronIdx].FBiasWeight;
      Inc(BiasValue);
    end;
  end
  else
  if ( not(Self is TNNetLocalConnect) and not(Self is TNNetLocalProduct) ) then
  begin
    FErrorProc('Error: bias output hasn''t been defined.');
  end;
end;

constructor TNNetLayerConcatedWeights.Create();
begin
  inherited Create();
  FNeuronWeightList := TNNetVolumeList.Create(false);
  FConcatedWeights := TNNetVolume.Create();
  FConcatedWInter := TNNetVolume.Create();
  FBiasOutput := TNNetVolume.Create();
  FShouldConcatWeights := false;
  FShouldInterleaveWeights := false;
  FAfterWeightUpdateHasBeenCalled := false;
end;

destructor TNNetLayerConcatedWeights.Destroy();
begin
  FBiasOutput.Free;
  FConcatedWeights.Free;
  FNeuronWeightList.Free;
  FConcatedWInter.Free;
  inherited Destroy();
end;

procedure TNNetLayerConcatedWeights.RefreshNeuronWeightList();
var
  NeuronCnt: integer;
begin
  FNeuronWeightList.Clear;

  for NeuronCnt := 0 to FNeurons.Count - 1 do
  begin
    FNeuronWeightList.Add(FNeurons[NeuronCnt].Weights);
  end;
end;

{$IFDEF OpenCL}
procedure TNNetConvolutionBase.EnableOpenCL(DotProductKernel: TDotProductKernel);
begin
  inherited EnableOpenCL(DotProductKernel);
  FDotCL.PrepareForCompute(FConcatedWInter, FInputPrepared, FVectorSize);
end;

procedure TNNetLayerConcatedWeights.EnableOpenCL(
  DotProductKernel: TDotProductKernel);
begin
  inherited EnableOpenCL(DotProductKernel);
  (*
  // good for debugging
  WriteLn(
    'Has OpenCL:', FHasOpenCL,
    ' Should OpenCL:', FShouldOpenCL,
    ' Current layer:', Self.LayerIdx
  );
  *)
  if (FHasOpenCL and FShouldOpenCL) then
  begin
    if not Assigned(FDotCL) then
    begin
      FDotCL := TDotProductSharedKernel.Create(DotProductKernel);
      FDotCL.HideMessages();
    end;
    RefreshNeuronWeightList();
    AfterWeightUpdate();

    FConcatedWeights.ReSize(FNeuronWeightList.GetTotalSize(),1,1);

    FConcatedWInter.ReSize(FNeuronWeightList.GetTotalSize(),1,1);

    //WriteLn(' Layer:', Self.LayerIdx,' Vector:',FVectorSize,' Neuron count:',FNeuronWeightList.Count,' Output size:',FOutput.Size);
    FShouldInterleaveWeights := true;
    FShouldConcatWeights := true;

    //FDotProductResult.ReSize(FOutputSizeX, FOutputSizeY, FNeurons.Count);
  end;
  AfterWeightUpdate();
end;
{$ENDIF}

{ TNNetReLU }
procedure TNNetReLU.Compute();
var
  SizeM1: integer;
  LocalPrevOutput: TNNetVolume;
  OutputCnt: integer;
  StartTime: double;
begin
  StartTime := Now();
  LocalPrevOutput := FPrevLayer.Output;
  SizeM1 := LocalPrevOutput.Size - 1;

  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    for OutputCnt := 0 to SizeM1 do
    begin
      if LocalPrevOutput.FData[OutputCnt] >= 0 then
      begin
        FOutput.FData[OutputCnt] := LocalPrevOutput.FData[OutputCnt];
        FOutputErrorDeriv.FData[OutputCnt] := 1;
      end
      else
      begin
        FOutput.FData[OutputCnt] := 0;
        FOutputErrorDeriv.FData[OutputCnt] := 0;
      end;
    end;
  end
  else
  begin
    // can't calculate error on input layers.
    for OutputCnt := 0 to SizeM1 do
    begin
      if LocalPrevOutput.FData[OutputCnt]>0 then
      begin
        FOutput.FData[OutputCnt] := LocalPrevOutput.FData[OutputCnt];
      end
      else
      begin
        FOutput.FData[OutputCnt] := 0;
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetReLUBase.Backpropagate();
var
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FOutput.Size = FOutputError.Size) and (FOutputErrorDeriv.Size = FOutput.Size) then
  begin
    FOutputError.Mul(FOutputErrorDeriv);
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate();
end;

{ TNNetMulLearning }
constructor TNNetMulLearning.Create(pMul: integer);
begin
  inherited Create();
  FStruct[0] := pMul;
end;

procedure TNNetMulLearning.Backpropagate();
var
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  FOutputError.Mul(FStruct[0]);
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate();
end;

{ TNNetSum }
constructor TNNetSum.Create(aL: array of TNNetLayer);
var
  LayerCnt: integer;
  SizeX, SizeY, Deep: integer;
begin
  inherited Create();
  SizeX := aL[0].FOutput.SizeX;
  SizeY := aL[0].FOutput.SizeY;
  Deep  := aL[0].FOutput.Depth;

  if Length(aL) < 1 then
  begin
    FErrorProc('Input layer count is smaller than 1 at TNNetSum.');
  end
  else
  begin
    for LayerCnt := Low(aL) to High(aL) do
    begin
      if
      (
        (aL[LayerCnt].FOutput.SizeX <> SizeX) or
        (aL[LayerCnt].FOutput.SizeY <> SizeY) or
        (aL[LayerCnt].FOutput.Depth <> Deep)
      ) then
      begin
        FErrorProc
        (
          'Size doesn''t match at TNNetSum at index: '+IntToStr(LayerCnt)+
          ' Should be:('+IntToStr(SizeX)+' '+IntToStr(SizeY)+' '+IntToStr(Deep)+' '+')'+
          ' It is:('+IntToStr(aL[LayerCnt].FOutput.SizeX)+' '+IntToStr(aL[LayerCnt].FOutput.SizeY)+' '+IntToStr(aL[LayerCnt].FOutput.Depth)+' '+').'
        );
      end;

      FPrevOutput.Add(aL[LayerCnt].FOutput);
      FPrevOutputError.Add(aL[LayerCnt].FOutputError);
      FPrevOutputErrorDeriv.Add(aL[LayerCnt].FOutputErrorDeriv);
      FPrevLayerList.Add(aL[LayerCnt]);
      aL[LayerCnt].IncDepartingBranchesCnt();
    end;
    Output.Resize(SizeX, SizeY, Deep);
    FOutputError.Resize(SizeX, SizeY, Deep);
    FOutputErrorDeriv.Resize(SizeX, SizeY, Deep);
  end;
  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;
end;

destructor TNNetSum.Destroy();
begin
  inherited Destroy();
end;

procedure TNNetSum.Compute();
var
  LayerCnt: integer;
  StartTime: double;
begin
  StartTime := Now();
  FOutput.Copy(FPrevOutput[0]);
  if FPrevOutput.Count > 1 then
  begin
    for LayerCnt := 1 to FPrevOutput.Count - 1 do
    begin
      FOutput.Add(FPrevOutput[LayerCnt]);
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetSum.Backpropagate();
var
  LayerCnt: integer;
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  //FOutputError.Divi(FPrevOutput.Count);
  for LayerCnt := 0 to FPrevOutput.Count - 1 do
  begin
    FPrevOutputError[LayerCnt].Add(FOutputError);
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  BackpropagateConcat();
end;

{ TNNetMaxChannel }
procedure TNNetMaxChannel.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  FPoolSize := pPrevLayer.Output.SizeX;
  FStride := FPoolSize;
  FPadding := 0;

  inherited SetPrevLayer(pPrevLayer);
end;

constructor TNNetMaxChannel.Create();
begin
  inherited Create(2);
end;

{ TNNetAvgChannel }
procedure TNNetAvgChannel.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  FPoolSize := pPrevLayer.Output.SizeX;
  FStride := FPoolSize;
  FPadding := 0;

  inherited SetPrevLayer(pPrevLayer);
end;

constructor TNNetAvgChannel.Create();
begin
  inherited Create(2);
end;

{ TNNetConvolutionLinear }

constructor TNNetConvolutionLinear.Create(pNumFeatures, pFeatureSize,
  pInputPadding, pStride: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pSuppressBias);
  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;
end;

{ THistoricalNets }
procedure THistoricalNets.AddLeCunLeNet5(IncludeInput: boolean);
begin
  if IncludeInput then AddLayer( TNNetInput.Create(28, 28, 1) );
  AddLayer( TNNetConvolution.Create(6, 5, 0, 1) );
  AddLayer( TNNetMaxPool.Create(2) );
  AddLayer( TNNetConvolution.Create(16, 5, 0, 1) );
  AddLayer( TNNetMaxPool.Create(2) );
  AddLayer( TNNetFullConnect.Create(120) );
  AddLayer( TNNetFullConnect.Create(84) );
  AddLayer( TNNetFullConnectLinear.Create(10) );
  AddLayer( TNNetSoftMax.Create() );
end;

procedure THistoricalNets.AddAlexNet(IncludeInput: boolean);
begin
  if IncludeInput then AddLayer( TNNetInput.Create(227, 227, 3) );

  //Conv1 + ReLU
  AddLayer( TNNetConvolutionReLU.Create(96, 11, 0, 4) );
  AddLayer( TNNetMaxPool.Create(3,2) );
  AddLayer( TNNetLocalResponseNormDepth.Create(5) );

  //Conv2 + ReLU
  AddLayer( TNNetConvolutionReLU.Create(256, 5, 2, 1) );
  AddLayer( TNNetMaxPool.Create(3,2) );
  AddLayer( TNNetLocalResponseNormDepth.Create(5) );

  //Conv3,4,5 + ReLU
  AddLayer( TNNetConvolutionReLU.Create(398, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(398, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(256, 3, 1, 1) );
  AddLayer( TNNetMaxPool.Create(3,2) );

  // Dropouts and Dense Layers
  AddLayer( TNNetDropout.Create(0.5) );
  AddLayer( TNNetFullConnectReLU.Create(4096) );
  AddLayer( TNNetDropout.Create(0.5) );
  AddLayer( TNNetFullConnectReLU.Create(4096) );
  AddLayer( TNNetFullConnectLinear.Create(1000) );
  AddLayer( TNNetSoftMax.Create() );
end;

procedure THistoricalNets.AddVGGNet(IncludeInput: boolean);
begin
  if IncludeInput then AddLayer( TNNetInput.Create(224, 224, 3) );
  AddLayer( TNNetConvolutionReLU.Create(64, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(64, 3, 1, 1) );
  AddLayer( TNNetMaxPool.Create(2) );

  //112x112x64
  AddLayer( TNNetConvolutionReLU.Create(128, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(128, 3, 1, 1) );
  AddLayer( TNNetMaxPool.Create(2) );

  //56x56x128
  AddLayer( TNNetConvolutionReLU.Create(256, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(256, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(256, 3, 1, 1) );
  AddLayer( TNNetMaxPool.Create(2) );

  //28x28x256
  AddLayer( TNNetConvolutionReLU.Create(512, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(512, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(512, 3, 1, 1) );
  AddLayer( TNNetMaxPool.Create(2) );

  //14x14x512
  AddLayer( TNNetConvolutionReLU.Create(512, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(512, 3, 1, 1) );
  AddLayer( TNNetConvolutionReLU.Create(512, 3, 1, 1) );
  AddLayer( TNNetMaxPool.Create(2) );

  //7x7x512
  AddLayer( TNNetFullConnectReLU.Create(4096) );
  AddLayer( TNNetFullConnectReLU.Create(4096) );
  AddLayer( TNNetFullConnectLinear.Create(1000) );
  AddLayer( TNNetSoftMax.Create() );
end;

procedure THistoricalNets.AddResNetUnit(pNeurons: integer);
var
  PreviousLayer, ShortCut, LongPath: TNNetLayer;
  Stride: integer;
begin
  PreviousLayer := GetLastLayer();
  if PreviousLayer.Output.Depth = pNeurons
    then Stride := 1
    else Stride := 2;
  LongPath := AddLayer([
    TNNetConvolutionReLU.Create(pNeurons, {featuresize}3, {padding}1, Stride),
    TNNetConvolutionLinear.Create(pNeurons, {featuresize}3, {padding}1, {stride}1)
  ]);
  if PreviousLayer.Output.Depth = pNeurons then
  begin
    AddLayer( TNNetSum.Create([PreviousLayer, LongPath]) );
  end
  else
  begin
    ShortCut := AddLayerAfter([
      TNNetConvolutionLinear.Create(pNeurons, {featuresize}3, {padding}1, Stride)
    ], PreviousLayer);
    AddLayer( TNNetSum.Create([ShortCut, LongPath]) );
  end;
  AddLayer( TNNetReLU.Create() );
end;

function THistoricalNets.AddDenseNetBlock(pUnits, k: integer;
  BottleNeck: integer = 0;
  supressBias: integer = 1;
  DropoutRate: TNeuralFloat = 0.0): TNNetLayer;
var
  UnitCnt: integer;
  PreviousLayer: TNNetLayer;
begin
  if pUnits > 0 then
  begin
    for UnitCnt := 1 to pUnits do
    begin
      PreviousLayer := GetLastLayer();
      if BottleNeck > 0 then
      begin
        AddMovingNorm(false, 0, 0);
        AddLayer( TNNetSELU.Create() );
        AddLayer( TNNetPointwiseConvLinear.Create(BottleNeck, supressBias) );
      end;
      AddMovingNorm(false, 0, 0);
      AddLayer( TNNetSELU.Create() );
      AddLayer( TNNetConvolutionLinear.Create(k, {featuresize}3, {padding}1, {stride}1, supressBias) );
      if (DropoutRate > 0) then AddLayer( TNNetDropout.Create(DropoutRate) );
      AddLayer( TNNetDeepConcat.Create([PreviousLayer, GetLastLayer()]) );
    end;
  end;
  Result := GetLastLayer();
end;

function THistoricalNets.AddDenseNetTransition(
  Compression: TNeuralFloat = 0.5;
  supressBias: integer = 1;
  HasAvgPool: boolean = true): TNNetLayer;
begin
  AddMovingNorm(false, 0, 0);
  AddLayer( TNNetSELU.Create() );
  AddCompression(Compression, supressBias);
  if HasAvgPool
    then Result := AddLayer( TNNetAvgPool.Create(2) )
    else Result := AddLayer( TNNetMaxPool.Create(2) );
end;

function THistoricalNets.AddDenseNetBlockCAI(pUnits, k, supressBias: integer;
  PointWiseConv: TNNetConvolutionClass {= TNNetConvolutionLinear};
  IsSeparable: boolean = false;
  HasNorm: boolean = true;
  pBefore: TNNetLayerClass = nil;
  pAfter: TNNetLayerClass = nil;
  BottleNeck: integer = 0;
  Compression: integer = 1; // Compression factor. 2 means taking half of channels.
  DropoutRate: TNeuralFloat = 0;
  RandomBias: integer = 1; RandomAmplifier: integer = 1;
  FeatureSize: integer = 3
  ): TNNetLayer;
var
  UnitCnt: integer;
  PreviousLayer, LastLayer: TNNetLayer;
begin
  if pUnits > 0 then
  begin
    for UnitCnt := 1 to pUnits do
    begin
      PreviousLayer := GetLastLayer();
      if BottleNeck > 0 then
      begin
        if (PreviousLayer.Output.Depth > BottleNeck * 2) and (PointWiseConv <> nil) then
        begin
          if pBefore <> nil then AddLayer( pBefore.Create() );
          AddLayer( PointWiseConv.Create(BottleNeck, {featuresize}1, {padding}0, {stride}1, supressBias) );
          //AddAutoGroupedConvolution(PointWiseConv,
          //  {MinGroupSize=}16, {pNumFeatures=}BottleNeck, {pFeatureSize=}1,
          //  {pInputPadding=}0, {pStride=}1, supressBias,
          //  {ChannelInterleaving=} true);
          if pAfter <> nil then AddLayer( pAfter.Create() );
        end;
      end;
      if pBefore <> nil then AddLayer( pBefore.Create() );
      AddConvOrSeparableConv(IsSeparable, {HasReLU=} true, HasNorm, k, FeatureSize, (FeatureSize-1) div 2, 1, {PerCell=}false, supressBias, RandomBias, RandomAmplifier);
      if pAfter <> nil then AddLayer( pAfter.Create() );
      if DropoutRate > 0 then AddLayer( TNNetDropout.Create(DropoutRate) );
      LastLayer := GetLastLayer();
      if (UnitCnt=pUnits) and (Compression > 1) then
      begin
        PreviousLayer := AddLayerAfter( TNNetSplitChannelEvery.Create(Compression, 0),PreviousLayer );
      end;
      AddLayer( TNNetDeepConcat.Create([PreviousLayer, LastLayer]) );
    end;
  end;
  Result := GetLastLayer();
end;

function THistoricalNets.AddDenseNetBlockCAI(pUnits, k, supressBias: integer;
  PointWiseConv: TNNetConvolutionClass; IsSeparable: boolean; HasNorm: boolean;
  pBeforeBottleNeck: TNNetLayerClass; pAfterBottleNeck: TNNetLayerClass;
  pBeforeConv: TNNetLayerClass; pAfterConv: TNNetLayerClass;
  BottleNeck: integer; Compression: integer; DropoutRate: TNeuralFloat;
  RandomBias: integer; RandomAmplifier: integer; FeatureSize: integer
  ): TNNetLayer;
var
  UnitCnt: integer;
  PreviousLayer, LastLayer: TNNetLayer;
begin
  if pUnits > 0 then
  begin
    for UnitCnt := 1 to pUnits do
    begin
      PreviousLayer := GetLastLayer();
      if BottleNeck > 0 then
      begin
        if (PreviousLayer.Output.Depth > BottleNeck * 2) and (PointWiseConv <> nil) then
        begin
          if pBeforeBottleNeck <> nil then AddLayer( pBeforeBottleNeck.Create() );
          AddLayer( PointWiseConv.Create(BottleNeck, {featuresize}1, {padding}0, {stride}1, supressBias) );
          //AddAutoGroupedConvolution(PointWiseConv,
          //  {MinGroupSize=}16, {pNumFeatures=}BottleNeck, {pFeatureSize=}1,
          //  {pInputPadding=}0, {pStride=}1, supressBias,
          //  {ChannelInterleaving=} true);
          if pAfterBottleNeck <> nil then AddLayer( pAfterBottleNeck.Create() );
        end;
      end;
      if pBeforeConv <> nil then AddLayer( pBeforeConv.Create() );
      AddConvOrSeparableConv(IsSeparable, {HasReLU=} false, HasNorm, k, FeatureSize, (FeatureSize-1) div 2, 1, {PerCell=}false, supressBias, RandomBias, RandomAmplifier);
      if pAfterConv <> nil then AddLayer( pAfterConv.Create() );
      if DropoutRate > 0 then AddLayer( TNNetDropout.Create(DropoutRate) );
      LastLayer := GetLastLayer();
      if (UnitCnt=pUnits) and (Compression > 1) then
      begin
        PreviousLayer := AddLayerAfter( TNNetSplitChannelEvery.Create(Compression, 0),PreviousLayer );
      end;
      AddLayer( TNNetDeepConcat.Create([PreviousLayer, LastLayer]) );
    end;
  end;
  Result := GetLastLayer();
end;

function THistoricalNets.AddkDenseNetBlock(pUnits, k, supressBias: integer;
  PointWiseConv: TNNetGroupedPointwiseConvClass; IsSeparable: boolean;
  HasNorm: boolean; pBeforeBottleNeck: TNNetLayerClass;
  pAfterBottleNeck: TNNetLayerClass; pBeforeConv: TNNetLayerClass;
  pAfterConv: TNNetLayerClass; BottleNeck: integer; Compression: integer;
  DropoutRate: TNeuralFloat; RandomBias: integer; RandomAmplifier: integer;
  FeatureSize: integer; MinGroupSize: integer): TNNetLayer;
var
  UnitCnt: integer;
  PreviousLayer, LastLayer: TNNetLayer;
begin
  if pUnits > 0 then
  begin
    for UnitCnt := 1 to pUnits do
    begin
      PreviousLayer := GetLastLayer();
      if BottleNeck > 0 then
      begin
        if (PreviousLayer.Output.Depth > BottleNeck * 2) and (PointWiseConv <> nil) then
        begin
          if pBeforeBottleNeck <> nil then AddLayer( pBeforeBottleNeck.Create() );
          //if UnitCnt > 1 then AddLayer( TNNetInterleaveChannels.Create(UnitCnt) );
          AddAutoGroupedPointwiseConv2( PointWiseConv, MinGroupSize, BottleNeck, HasNorm, supressBias, false, false );
          //AddAutoGroupedConvolution(TNNetConvolutionReLU, MinGroupSize, BottleNeck, 1, 0, 1, supressBias, False);
          if pAfterBottleNeck <> nil then AddLayer( pAfterBottleNeck.Create() );
        end;
      end;
      if pBeforeConv <> nil then AddLayer( pBeforeConv.Create() );
      AddConvOrSeparableConv(IsSeparable, {HasReLU=} false, HasNorm, k, FeatureSize, (FeatureSize-1) div 2, 1, {PerCell=}false, supressBias, RandomBias, RandomAmplifier);
      if pAfterConv <> nil then AddLayer( pAfterConv.Create() );
      if DropoutRate > 0 then AddLayer( TNNetDropout.Create(DropoutRate) );
      LastLayer := GetLastLayer();
      if (UnitCnt=pUnits) and (Compression > 1) then
      begin
        PreviousLayer := AddLayerAfter( TNNetSplitChannelEvery.Create(Compression, 0),PreviousLayer );
      end;
      AddLayer( TNNetDeepConcat.Create([PreviousLayer, LastLayer]) );
    end;
  end;
  Result := GetLastLayer();
end;

function THistoricalNets.AddParallelConvs(PointWiseConv: TNNetConvolutionClass;
  IsSeparable: boolean;
  CopyInput: boolean;
  pBeforeBottleNeck: TNNetLayerClass;
  pAfterBottleNeck: TNNetLayerClass; pBeforeConv: TNNetLayerClass;
  pAfterConv: TNNetLayerClass;
  PreviousLayer: TNNetLayer;
  BottleNeck: integer;
  p11ConvCount: integer;
  p11FilterCount: integer;
  p33ConvCount: integer;
  p33FilterCount: integer;
  p55ConvCount: integer;
  p55FilterCount: integer;
  p77ConvCount: integer;
  p77FilterCount: integer;
  maxPool: integer;
  minPool: integer): TNNetLayer;
var
  UnitCnt: integer;
  ConvCount: integer;
  aL: array of TNNetLayer;
  LastLayer: TNNetLayer;

  procedure LocalAddConv(FilterCount, FilterSize: integer; PrevLayer: TNNetLayer);
  begin
    if pBeforeConv <> nil then PrevLayer := AddLayerAfter( pBeforeConv.Create(), PrevLayer );
    AddConvOrSeparableConv(IsSeparable, {HasReLU=} false, {HasNorm=}false, FilterCount, FilterSize, (FilterSize-1) div 2, 1, {PerCell=}false, {supressBias=}1, {RandomBias=}0, {RandomAmplifier=}0, PrevLayer);
  end;

  function LocalAddBottleNeck(FilterCount: integer; PrevLayer: TNNetLayer): TNNetLayer;
  begin
    if pBeforeBottleNeck <> nil then PrevLayer := AddLayerAfter( pBeforeBottleNeck.Create(), PrevLayer);
    AddLayerAfter( PointWiseConv.Create(FilterCount, {featuresize}1, {padding}0, {stride}1, {supressBias}1), PrevLayer);
    if pAfterBottleNeck <> nil then AddLayer( pAfterBottleNeck.Create() );
    Result := GetLastLayer();
  end;

begin
  if Not(Assigned(PreviousLayer)) then PreviousLayer := GetLastLayer();
  UnitCnt := 0;
  SetLength(aL, 3 + p11ConvCount + p33ConvCount + p55ConvCount + p77ConvCount);
  if (CopyInput) then
  begin
    aL[UnitCnt] := PreviousLayer;
    Inc(UnitCnt);
  end;
  if (maxPool>0) then
  begin
    LastLayer := AddLayerAfter( TNNetMaxPool.Create(3,1,0), PreviousLayer);
    aL[UnitCnt] := LocalAddBottleNeck(maxPool, LastLayer);
    Inc(UnitCnt);
  end;
  if (minPool>0) then
  begin
    LastLayer := AddLayerAfter( TNNetMinPool.Create(3,1,0), PreviousLayer);
    aL[UnitCnt] := LocalAddBottleNeck(minPool, LastLayer);
    Inc(UnitCnt);
  end;
  if ( (p11FilterCount>0) and (p11ConvCount>0) ) then
  begin
    for ConvCount := 1 to p11ConvCount do
    begin
      LastLayer := PreviousLayer;
      if BottleNeck > 0 then LastLayer := LocalAddBottleNeck(BottleNeck, PreviousLayer);
      LocalAddConv(p11FilterCount, 1, LastLayer);
      aL[UnitCnt] := GetLastLayer();
      Inc(UnitCnt);
    end;
  end;
  if ( (p33FilterCount>0) and (p33ConvCount>0) ) then
  begin
    for ConvCount := 1 to p33ConvCount do
    begin
      LastLayer := PreviousLayer;
      if BottleNeck > 0 then LastLayer := LocalAddBottleNeck(BottleNeck, PreviousLayer);
      LocalAddConv(p33FilterCount, 3, LastLayer);
      aL[UnitCnt] := GetLastLayer();
      Inc(UnitCnt);
    end;
  end;
  if ( (p55FilterCount>0) and (p55ConvCount>0) ) then
  begin
    for ConvCount := 1 to p55ConvCount do
    begin
      LastLayer := PreviousLayer;
      if BottleNeck > 0 then LastLayer := LocalAddBottleNeck(BottleNeck, PreviousLayer);
      LocalAddConv(BottleNeck, 3, LastLayer);
      if pAfterConv <> nil then AddLayer( pAfterConv.Create() );
      LocalAddConv(p55FilterCount, 3, GetLastLayer());
      aL[UnitCnt] := GetLastLayer();
      Inc(UnitCnt);
    end;
  end;
  if ( (p77FilterCount>0) and (p77ConvCount>0) ) then
  begin
    for ConvCount := 1 to p77ConvCount do
    begin
      LastLayer := PreviousLayer;
      if BottleNeck > 0 then LastLayer := LocalAddBottleNeck(BottleNeck, PreviousLayer);
      LocalAddConv(BottleNeck, 3, LastLayer);
      if pAfterConv <> nil then AddLayer( pAfterConv.Create() );
      LocalAddConv(BottleNeck, 3, GetLastLayer());
      if pAfterConv <> nil then AddLayer( pAfterConv.Create() );
      LocalAddConv(p77FilterCount, 3, GetLastLayer());
      aL[UnitCnt] := GetLastLayer();
      Inc(UnitCnt);
    end;
  end;
  if UnitCnt > 1 then
  begin
    SetLength(aL, UnitCnt);
    AddLayer( TNNetDeepConcat.Create(aL) );
  end;
  if pAfterConv <> nil then AddLayer( pAfterConv.Create() );
  SetLength(aL, 0);
  Result := GetLastLayer();
end;

function THistoricalNets.AddDenseFullyConnected(pUnits, k,
  supressBias: integer; PointWiseConv: TNNetConvolutionClass;
  HasNorm: boolean; HasReLU: boolean;
  pBefore: TNNetLayerClass; pAfter: TNNetLayerClass; BottleNeck: integer;
  Compression: TNeuralFloat): TNNetLayer;
var
  UnitCnt: integer;
  PreviousLayer: TNNetLayer;
begin
  if pUnits > 0 then
  begin
    for UnitCnt := 1 to pUnits do
    begin
      PreviousLayer := GetLastLayer();
      if BottleNeck > 0 then
      begin
        if (PreviousLayer.Output.Depth > BottleNeck) and (PointWiseConv <> nil) then
        begin
          if pBefore <> nil then AddLayer( pBefore.Create() );
          AddLayer( PointWiseConv.Create(BottleNeck, {featuresize}1, {padding}0, {stride}1, supressBias) );
          if pAfter <> nil then AddLayer( pAfter.Create() );
        end;
      end;
      if pBefore <> nil then AddLayer( pBefore.Create() );
      if HasReLU
      then AddLayer( TNNetFullConnectReLU.Create(k) )
      else AddLayer( TNNetFullConnectLinear.Create(k) );
      if HasNorm then AddMovingNorm(false, 1, 1);
      if pAfter <> nil then AddLayer( pAfter.Create() );
      AddLayer( TNNetConcat.Create([PreviousLayer, GetLastLayer()]) );
    end;
    AddLayer( PointWiseConv.Create(Round(GetLastLayer().Output.Depth * Compression ), {featuresize}1, {padding}0, {stride}1, {suppress bias}1) );
  end;
  Result := GetLastLayer();
end;

function THistoricalNets.AddSuperResolution(pSizeX, pSizeY, BottleNeck, pNeurons,
  pLayerCnt: integer; IsSeparable:boolean): TNNetLayer;
var
  BeforeDeAvgLayerCnt, AfterDeAvgLayerCnt: integer;
begin
  AddLayer( TNNetInput.Create(pSizeX, pSizeY, 3) );
  BeforeDeAvgLayerCnt := (pLayerCnt div 2);
  AfterDeAvgLayerCnt := pLayerCnt - BeforeDeAvgLayerCnt - 1;

  AddDenseNetBlockCAI
  (
    BeforeDeAvgLayerCnt, pNeurons, {supressBias=}0,
    {PointWiseConv=}TNNetConvolutionLinear,
    {IsSeparable=}IsSeparable,
    {HasNorm=}false,
    {pBefore=}nil,
    {pAfter=}nil,
    {BottleNeck=}BottleNeck,
    {Compression=}1, // Compression factor. 2 means taking half of channels.
    {DropoutRate=}0,
    {RandomBias=}0,
    {RandomAmplifier=}0
  );
  AddLayer( TNNetDeAvgPool.Create(2) );
  AddDenseNetBlockCAI
  (
    AfterDeAvgLayerCnt, pNeurons, {supressBias=}0,
    {PointWiseConv=}TNNetConvolutionLinear,
    {IsSeparable=}IsSeparable,
    {HasNorm=}false,
    {pBefore=}nil,
    {pAfter=}nil,
    {BottleNeck=}BottleNeck,
    {Compression=}1, // Compression factor. 2 means taking half of channels.
    {DropoutRate=}0,
    {RandomBias=}0,
    {RandomAmplifier=}0
  );
  Result := AddLayer( TNNetConvolutionLinear.Create(3,1,0,0) );
end;

{ TNNetFullConnectLinear }

procedure TNNetFullConnectLinear.ComputePreviousLayerErrorCPU();
var
  MaxOutputCnt: integer;
  OutputCnt: integer;
  LocalPrevError: TNNetVolume;
begin
  LocalPrevError := FPrevLayer.OutputError;

  MaxOutputCnt := FOutput.Size - 1;
  for OutputCnt := 0 to MaxOutputCnt do
  begin
    if (FOutputError.FData[OutputCnt] <> 0.0) then
    begin
      LocalPrevError.MulAdd(FOutputError.FData[OutputCnt], FArrNeurons[OutputCnt].FWeights);
    end;
  end;
end;

procedure TNNetFullConnectLinear.ComputeCPU();
var
  Cnt, MaxCnt: integer;
begin
  MaxCnt := FNeurons.Count - 1;
  if FSuppressBias = 0 then
  begin
    for Cnt := 0 to MaxCnt do
    begin
      FOutput.FData[Cnt] :=
        FArrNeurons[Cnt].Weights.DotProduct(FPrevLayer.Output) +
        FArrNeurons[Cnt].FBiasWeight;
    end;
  end
  else
  begin
    for Cnt := 0 to MaxCnt do
    begin
      FOutput.FData[Cnt] :=
        FArrNeurons[Cnt].Weights.DotProduct(FPrevLayer.Output);
    end;
  end;
end;

procedure TNNetFullConnectLinear.BackpropagateCPU();
var
  MaxNeurons, NeuronCnt: integer;
  localLearErrorDeriv: TNeuralFloat;
  localNeuron: TNNetNeuron;
begin
  MaxNeurons := FNeurons.Count - 1;
  for NeuronCnt := 0 to MaxNeurons do
  begin
      localNeuron := FArrNeurons[NeuronCnt];
      localLearErrorDeriv := -FLearningRate * FOutputError.FData[NeuronCnt];

      if (FBatchUpdate) then
      begin
        if localLearErrorDeriv <> 0.0 then
        begin
          localNeuron.Delta.MulAdd(localLearErrorDeriv, FPrevLayer.Output);
          localNeuron.FBiasDelta := localNeuron.FBiasDelta + localLearErrorDeriv;
        end;
      end
      else
      begin
        localNeuron.FBackInertia.MulMulAdd(FInertia, localLearErrorDeriv * (1-FInertia), FPrevLayer.Output);

        localNeuron.FBiasInertia :=
          (1-FInertia)*localLearErrorDeriv +
          (  FInertia)*localNeuron.FBiasInertia;

        localNeuron.AddInertia();
      end;
  end;
  if not FBatchUpdate then AfterWeightUpdate();
end;

constructor TNNetFullConnectLinear.Create(pSizeX, pSizeY, pDepth: integer;
  pSuppressBias: integer = 0);
begin
  inherited Create(pSizeX, pSizeY, pDepth, pSuppressBias);
  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;
end;

constructor TNNetFullConnectLinear.Create(pSize: integer; pSuppressBias: integer = 0);
begin
  Self.Create(pSize, 1, 1, pSuppressBias);
end;

{ TNNetAddNumber }

constructor TNNetAddAndDiv.Create(pAdd, pDiv: integer);
begin
  inherited Create;
  FStruct[0] := pAdd;
  FStruct[1] := pDiv;
end;

procedure TNNetAddAndDiv.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute();
  FOutput.Add(FStruct[0]);
  FOutput.Divi(FStruct[1]);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetLocalResponseNormDepth }
procedure TNNetLocalResponseNormDepth.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  FOutput.CopyNoChecks(FPrevLayer.FOutput);
  FLRN.CalculateLocalResponseFromDepth(FOutput, FStruct[0], 0.001 / 9.0, 0.75);
  FOutput.Divi(FLRN);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

{ TNNetLocalResponseNorm2D }
constructor TNNetLocalResponseNorm2D.Create(pSize: integer);
begin
  inherited Create();
  FLRN := TNNetVolume.Create;
  FStruct[0] := pSize;
end;

destructor TNNetLocalResponseNorm2D.Destroy();
begin
  FLRN.Free;
  inherited Destroy();
end;

procedure TNNetLocalResponseNorm2D.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute();
  FLRN.CalculateLocalResponseFrom2D(FOutput, FStruct[0], 0.001 / 9.0, 0.75);
  FOutput.Divi(FLRN);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetLocalResponseNorm2D.Backpropagate();
var
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  FOutputError.Divi(FLRN);
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate();
end;

{ TNNetFullConnectDiff }
procedure TNNetFullConnectDiff.ComputePreviousLayerError();
var
  MaxOutputCnt: integer;
  OutputCnt: integer;
  AuxVolume: TNNetVolume;
  {$IFDEF CheckRange} MaxError: TNeuralFloat; {$ENDIF}
begin
  if ( Assigned(FPrevLayer) and (FPrevLayer.Output.Size = FPrevLayer.OutputError.Size) )then
  begin
    AuxVolume := TNNetVolume.Create();
    MaxOutputCnt := FOutput.Size - 1;
    for OutputCnt := 0 to MaxOutputCnt do
    begin
      AuxVolume.Copy(FPrevLayer.Output);
      AuxVolume.Sub(FNeurons[OutputCnt].FWeights);

      FPrevLayer.OutputError.MulAdd(FOutputErrorDeriv.FData[OutputCnt]/FOutput.Size, AuxVolume);
    end;

    {$IFDEF CheckRange}
    MaxError := FPrevLayer.OutputError.GetMax();
    if MaxError > 1 then
    begin
      FPrevLayer.OutputError.Divi(MaxError);
      FPrevLayer.OutputErrorDeriv.Divi(MaxError);
    end;
    {$ENDIF}
    AuxVolume.Free;
  end;
end;

constructor TNNetFullConnectDiff.Create(pSizeX, pSizeY, pDepth: integer;
  pSuppressBias: integer = 0);
begin
  inherited Create(pSizeX, pSizeY, pDepth, pSuppressBias);
  FActivationFn := @DiffAct;
  FActivationFnDerivative := @DiffActDerivative;
end;

constructor TNNetFullConnectDiff.Create(pSize: integer; pSuppressBias: integer = 0);
begin
  Self.Create(pSize, 1, 1, pSuppressBias);
end;

procedure TNNetFullConnectDiff.Compute();
var
  Cnt, MaxCnt: integer;
  Sum: TNeuralFloat;
  StartTime: double;
begin
  StartTime := Now();
  if (FNeurons.Count = FOutput.Size) and
    (FPrevLayer.Output.Size = FNeurons[0].Weights.Size) then
  begin
    MaxCnt := FNeurons.Count - 1;
    for Cnt := 0 to MaxCnt do
    begin
      {$IFDEF Debug}
      if (FNeurons[Cnt].Weights.Size <> FPrevLayer.Output.Size) then
      begin
        FErrorProc
        (
          'TNNetFullConnectDiff.Compute should have same sizes.'+
          'Neuron:'+IntToStr(Cnt)+
          ' Weights:'+IntToStr(FNeurons[Cnt].Weights.Size)+
          ' Prev Layer Output:'+IntToStr(FPrevLayer.Output.Size)
        );
      end;
      {$ENDIF}
      Sum :=
        (FNeurons[Cnt].Weights.SumDiff(FPrevLayer.Output)/FNeurons.Count);
      FOutputRaw.Raw[Cnt] := Sum;
      FOutput.Raw[Cnt] := FActivationFn(Sum);
    end;
  end else
  begin
    FErrorProc
    (
      'TNNetFullConnectDiff.Compute should have same sizes.'+
      'Neurons:'+IntToStr(FNeurons.Count)+
      ' Prev Layer Output:'+IntToStr(FPrevLayer.Output.Size)+
      ' Output:'+IntToStr(FOutput.Size)
    );
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetFullConnectDiff.Backpropagate();
var
  MaxNeurons, NeuronCnt: integer;
  localLearErrorDeriv: TNeuralFloat;
  localNeuron: TNNetNeuron;
  AuxVolume: TNNetVolume;
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FNeurons.Count = FOutput.Size) and (FPrevLayer.Output.Size>0) then
  begin
    ComputeErrorDeriv();

    AuxVolume := TNNetVolume.Create();
    if FLearningRate <> 0.0 then
    begin
      MaxNeurons := FNeurons.Count - 1;
      for NeuronCnt := 0 to MaxNeurons do
      begin
        localNeuron := FNeurons[NeuronCnt];
        AuxVolume.Copy(FPrevLayer.Output);
        AuxVolume.Sub(localNeuron.FWeights);

        localLearErrorDeriv := FLearningRate * FOutputErrorDeriv.FData[NeuronCnt];

        if (FBatchUpdate) then
        begin
          localNeuron.Delta.MulAdd(localLearErrorDeriv * (1-FInertia), AuxVolume);
        end
        else
        begin
          localNeuron.FBackInertia.Mul(FInertia);
          localNeuron.FBackInertia.MulAdd(localLearErrorDeriv * (1-FInertia), AuxVolume);

          {$IFDEF CheckRange}
          NeuronForceRange(localNeuron.FBiasInertia,FLearningRate);
          {$ENDIF}

          localNeuron.AddInertia();
        end;
      end;
    end; // of FLearningRate <> 0.0
    {$IFDEF CheckRange} ForceRangeWeights(1000); {$ENDIF}
    ComputePreviousLayerError();
    AuxVolume.Free;
  end else
  begin
    FErrorProc
    (
      'TNNetFullConnectDiff.Backpropagate should have same sizes.' +
      'Neurons:' + IntToStr(FNeurons.Count) +
      ' Output:' + IntToStr(FOutput.Size) +
      ' PrevLayer:' + IntToStr(FPrevLayer.Output.Size)
    );
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
  if not FBatchUpdate then AfterWeightUpdate();
end;

{ TNNetDeepConcat }
constructor TNNetDeepConcat.Create(aL: array of TNNetLayer);
var
  LayerCnt: integer;
  ForDeepCnt, DeepCnt: integer;
  SizeX, SizeY: integer;
begin
  inherited Create();
  DeepCnt := 0;

  SizeX := aL[0].FOutput.SizeX;
  SizeY := aL[0].FOutput.SizeY;

  if Length(aL) = 0 then
  begin
    FErrorProc('Input layer count is zero at TNNetDeepConcat.');
  end
  else
  for LayerCnt := Low(aL) to High(aL) do
  begin
    if aL[LayerCnt] is TNNetInput then
    begin
      TNNetInput(aL[LayerCnt]).EnableErrorCollection;
    end;
    if
    (
      (aL[LayerCnt].FOutput.SizeX <> SizeX) or
      (aL[LayerCnt].FOutput.SizeY <> SizeY)
    ) then
    begin
      FErrorProc
      (
        'Size doesn''t match at TNNetDeepConcat. Layer 0 is (' +
        IntToStr(SizeX)+','+IntToStr(SizeY)+') - Layer ' +
        IntToStr(LayerCnt)+' is (' +
        IntToStr(aL[LayerCnt].FOutput.SizeX) +
        ',' +
        IntToStr(aL[LayerCnt].FOutput.SizeY) + ')'
      );
    end;

    FPrevOutput.Add(aL[LayerCnt].FOutput);
    FPrevOutputError.Add(aL[LayerCnt].FOutputError);
    FPrevOutputErrorDeriv.Add(aL[LayerCnt].FOutputErrorDeriv);
    FPrevLayerList.Add(aL[LayerCnt]);
    aL[LayerCnt].IncDepartingBranchesCnt();

    for ForDeepCnt := 0 to Al[LayerCnt].FOutputError.Depth - 1 do
    begin
      Inc(DeepCnt);
      SetLength(FDeepsLayer, DeepCnt);
      SetLength(FDeepsChannel, DeepCnt);
      SetLength(FRemainingChannels, DeepCnt);
      FDeepsLayer[DeepCnt-1] := LayerCnt;
      FDeepsChannel[DeepCnt-1] := ForDeepCnt;
      FRemainingChannels[DeepCnt-1] := Al[LayerCnt].FOutputError.Depth - ForDeepCnt;
    end;
  end;

  Output.Resize(SizeX, SizeY, DeepCnt);
  FOutputError.Resize(SizeX, SizeY, DeepCnt);
  FOutputErrorDeriv.Resize(SizeX, SizeY, DeepCnt);

  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;
end;

destructor TNNetDeepConcat.Destroy();
begin
  SetLength(FRemainingChannels, 0);
  SetLength(FDeepsLayer, 0);
  SetLength(FDeepsChannel, 0);
  inherited Destroy();
end;

procedure TNNetDeepConcat.Compute();
var
  OutputDeepCnt, LocalIdx: integer;
  LocalOutput: TNNetVolume;
  X, Y, MaxX, MaxY, MaxDepth: integer;
  OrigChannel: integer;
  StartTime: double;
  {$IFDEF AVXANY}
  SourceRawPos, DestRawPos: pointer;
  {$ENDIF}
  RowSize, RowSizeBytes: integer;
begin
  StartTime := Now();
  MaxX := Output.SizeX - 1;
  MaxY := Output.SizeY - 1;
  MaxDepth := Output.Depth - 1;

  if Output.Depth <> Length(FDeepsLayer) then
  begin
    FErrorProc('Error at TNNetDeepConcat. Depths do not match '+IntToStr(Output.Depth)+' , '+IntToStr(Length(FDeepsLayer))+'.');
  end;

  OutputDeepCnt := 0;
  while OutputDeepCnt <= MaxDepth do
  begin
    LocalIdx := FDeepsLayer[OutputDeepCnt];
    LocalOutput := FPrevOutput[LocalIdx];
    OrigChannel := FDeepsChannel[OutputDeepCnt];
    RowSize := FRemainingChannels[OutputDeepCnt];
    RowSizeBytes := RowSize * SizeOf(TNeuralFloat);

    for X := 0 to MaxX do
    begin
      for Y := 0 to MaxY do
      begin
        {$IFDEF AVXANY}
        SourceRawPos := LocalOutput.GetRawPtr(X,Y,OrigChannel);
        DestRawPos := FOutput.GetRawPtr(X,Y,OutputDeepCnt);
        asm_dword_copy;
        {$ELSE}
        Move
        (
          LocalOutput.FData[LocalOutput.GetRawPos(X,Y,OrigChannel)],
          FOutput.FData[FOutput.GetRawPos(X,Y,OutputDeepCnt)],
          RowSizeBytes
        );
        {$ENDIF}
      end;
    end;
    Inc(OutputDeepCnt, RowSize);
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetDeepConcat.Backpropagate();
var
  OutputDeepCnt, LocalIdx: integer;
  LocalError: TNNetVolume;
  X, Y, MaxX, MaxY, MaxDepth: integer;
  OrigChannel: integer;
  StartTime: double;
  RowSize: integer;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  MaxX := Output.SizeX - 1;
  MaxY := Output.SizeY - 1;
  MaxDepth := Output.Depth - 1;

  if Output.Depth <> Length(FDeepsLayer) then
  begin
    FErrorProc('Error at TNNetDeepConcat. Depths do not match '+IntToStr(Output.Depth)+' , '+IntToStr(Length(FDeepsLayer))+'.');
  end;

  OutputDeepCnt := 0;
  while OutputDeepCnt <= MaxDepth do
  begin
    LocalIdx    := FDeepsLayer[OutputDeepCnt];
    LocalError  := FPrevOutputError[LocalIdx];
    OrigChannel := FDeepsChannel[OutputDeepCnt];
    RowSize     := FRemainingChannels[OutputDeepCnt];

    for X := 0 to MaxX do
    begin
      for Y := 0 to MaxY do
      begin
        // Debug Only: WriteLn(OutputDeepCnt,' Local Idx:',LocalIdx,' X:',X,' Y:',Y,' Orig Channel:', OrigChannel);
        LocalError.Add
        (
          LocalError.GetRawPtr(X, Y, OrigChannel),
          FOutputError.GetRawPtr(X, Y, OutputDeepCnt),
          RowSize
        );
      end;
    end;
    Inc(OutputDeepCnt, RowSize);
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  BackpropagateConcat();
end;

{ TNNetConcatBase }

constructor TNNetConcatBase.Create();
begin
  inherited Create();
  FPrevOutput := TNNetVolumeList.Create;
  FPrevOutputError := TNNetVolumeList.Create;
  FPrevOutputErrorDeriv := TNNetVolumeList.Create;
  FPrevLayerList := TNNetLayerList.Create;

  FPrevOutput.FreeObjects := false;
  FPrevOutputError.FreeObjects := false;
  FPrevOutputErrorDeriv.FreeObjects := false;
  FPrevLayerList.FreeObjects := false;
end;

destructor TNNetConcatBase.Destroy();
begin
  FPrevOutput.Free;
  FPrevOutputError.Free;
  FPrevOutputErrorDeriv.Free;
  FPrevLayerList.Free;
  inherited Destroy();
end;

function TNNetConcatBase.SaveStructureToString(): string;
var
  I: integer;
begin
  Result := inherited SaveStructureToString + ':';
  for I := 0 to FPrevLayerList.Count - 1 do
  begin
    if I > 0 then Result := Result + ';';
    Result := Result + IntToStr(FPrevLayerList[I].FLayerIdx);
  end;
end;

procedure TNNetConcatBase.BackpropagateConcat();
var
  LayerCnt: integer;
  LocalLayer: TNNetLayer;
begin
  for LayerCnt := 0 to FPrevLayerList.Count - 1 do
  begin
    LocalLayer := FPrevLayerList[LayerCnt];
    if (Assigned(LocalLayer) and (LocalLayer.OutputError.Size = LocalLayer.Output.Size)) then
    begin
      LocalLayer.Backpropagate();
    end;
  end;
end;

{ TNNetSplitChannels }

procedure TNNetSplitChannels.SetPrevLayer(pPrevLayer: TNNetLayer);
var
  SizeX, SizeY, Depth: integer;
begin
  inherited SetPrevLayer(pPrevLayer);
  SizeX := pPrevLayer.Output.SizeX;
  SizeY := pPrevLayer.Output.SizeY;
  Depth := Length(FChannels);

  FOutput.ReSize(SizeX,SizeY,Depth);
  FOutputError.ReSize(FOutput);
  FOutputErrorDeriv.ReSize(FOutput);
end;

constructor TNNetSplitChannels.Create(ChannelStart, ChannelLen: integer);
var
  pChannels: array of integer;
  ChannelCnt: integer;
begin
  SetLength(pChannels, ChannelLen);
  for ChannelCnt := 0 to ChannelLen - 1 do
    pChannels[ChannelCnt] := ChannelStart + ChannelCnt;
  Create(pChannels);
end;

constructor TNNetSplitChannels.Create(pChannels: array of integer);
var
  I: integer;
begin
  inherited Create();
  SetLength(FChannels, Length(pChannels));
  for I := 0 to High(pChannels) do
  begin
    FChannels[I] := pChannels[I];
  end;
  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;
end;

destructor TNNetSplitChannels.Destroy();
begin
  SetLength(FChannels, 0);
  inherited Destroy();
end;

procedure TNNetSplitChannels.Compute();
var
  MaxX, MaxY: integer;
  X, Y, Depth, OutputDepth: integer;
  StartTime: double;
begin
  StartTime := Now();
  MaxX     := FOutput.SizeX - 1;
  MaxY     := FOutput.SizeY - 1;

  for X := 0 to MaxX do
  begin
    for Y := 0 to MaxY do
    begin
      OutputDepth := 0;
      for Depth in FChannels do
      begin
        FOutput[X, Y, OutputDepth] := FPrevLayer.FOutput[X, Y, Depth] ;
        Inc(OutputDepth);
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetSplitChannels.Backpropagate();
var
  MaxX, MaxY: integer;
  X, Y, Depth, OutputDepth: integer;
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (Assigned(FPrevLayer) and (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size)) then
  begin
    StartTime := Now();
    MaxX := FOutput.SizeX - 1;
    MaxY := FOutput.SizeY - 1;

    for X := 0 to MaxX do
    begin
      for Y := 0 to MaxY do
      begin
        OutputDepth := 0;
        for Depth in FChannels do
        begin
          FPrevLayer.FOutputError.Add(X, Y, Depth, FOutputError[X, Y, OutputDepth]);
          Inc(OutputDepth);
        end;
      end;
    end;
    FBackwardTime := FBackwardTime + (Now() - StartTime);
    FPrevLayer.Backpropagate();
  end;
end;

function TNNetSplitChannels.SaveStructureToString(): string;
var
  I, MaxChannels: integer;
begin
  Result := inherited SaveStructureToString + ':';
  MaxChannels := Length(FChannels) - 1;
  for I := 0 to MaxChannels do
  begin
    if I > 0 then Result := Result + ';';
    Result := Result + IntToStr(FChannels[I]);
  end;
end;

procedure TestDataParallelism(NN: TNNet);
var
  I: integer;
  NN2: TNNet;
  Par: TNNetDataParallelism;
  AllGood: boolean;
begin
  WriteLn('Testing Data Parallelism...');

  Par := TNNetDataParallelism.Create(NN, 4);
  NN2 := NN.Clone();
  Par.SumWeights(NN2);
  AllGood := True;
  for I := 0 to NN.Layers.Count - 1 do
  begin
    if NN.Layers[I].Neurons.Count <> NN2.Layers[I].Neurons.Count then
    begin
      WriteLn('Data Parallelism Error: neuron count doesn''t match on layer:',I);
      AllGood := False;
    end;

    if NN.Layers[I].CountWeights() <> NN2.Layers[I].CountWeights() then
    begin
      WriteLn('Data Parallelism Error: weight count doesn''t match on layer:',I);
      AllGood := False;
    end;

    if 4 * NN.Layers[I].GetWeightSum() <> NN2.Layers[I].GetWeightSum() then
    begin
      WriteLn('Data Parallelism Error: weight sum doesn''t match on layer:',I);
      AllGood := False;
    end;

    if 4 * NN.Layers[I].GetBiasSum() <> NN2.Layers[I].GetBiasSum() then
    begin
      WriteLn('Data Parallelism Error: bias sum doesn''t match on layer:',I);
      AllGood := False;
    end;

    if 4 * NN.Layers[I].GetInertiaSum() <> NN2.Layers[I].GetInertiaSum() then
    begin
      WriteLn('Data Parallelism Error: inertial sum doesn''t match on layer:',I);
      AllGood := False;
    end;
  end;
  if AllGood
  then WriteLn('Data Parallelism testing has passed.');
  NN2.Free;
  Par.Free;
end;

procedure CompareNNStructure(NN, NN2: TNNet);
var
  I: integer;
  AllGood: boolean;
begin
  WriteLn('Structural Test Start');
  AllGood := True;

  if NN.SaveToString() <> NN2.SaveToString() then
  begin
    WriteLn('Error: network save to string differs.');
    AllGood := False;
  end
  else
  begin
    WriteLn('Saving to string has passed.');
  end;

  for I := 0 to NN.Layers.Count - 1 do
  begin
    if NN.Layers[I].SaveDataToString() <> NN2.Layers[I].SaveDataToString() then
    begin
      WriteLn('Error: save data to string doesn''t match on layer:',I);
      AllGood := False;
    end;

    if NN.Layers[I].SaveStructureToString() <> NN2.Layers[I].SaveStructureToString() then
    begin
      WriteLn('Error: save structure to string doesn''t match on layer:',I);
      WriteLn('NN Structure:', NN.Layers[I].SaveStructureToString());
      WriteLn('NN2 Structure:', NN2.Layers[I].SaveStructureToString());
      AllGood := False;
    end;

    if NN.Layers[I].ClassName <> NN2.Layers[I].ClassName then
    begin
      WriteLn('Error: class name doesn''t match on layer:',I);
      AllGood := False;
    end;

    if NN.Layers[I].Neurons.Count <> NN2.Layers[I].Neurons.Count then
    begin
      WriteLn('Error: neuron count doesn''t match on layer:',I);
      AllGood := False;
    end;

    if NN.Layers[I].CountWeights() <> NN2.Layers[I].CountWeights() then
    begin
      WriteLn('Error: weight count doesn''t match on layer:',I);
      AllGood := False;
    end;

    if NN.Layers[I].GetWeightSum() <> NN2.Layers[I].GetWeightSum() then
    begin
      WriteLn('Error: weight sum doesn''t match on layer:',I);
      AllGood := False;
    end;
  end;
  if AllGood
  then WriteLn('Structural testing has passed.');
end;

procedure TestBackProp();
var
  NN: TNNet;
  InputVolume, OutputVolume: TNNetVolume;
begin
  NN := TNNet.Create;
  (*
  NN.AddLayer([
    TNNetInput.Create(4,4,1),
    TNNetConvolutionLinear.Create(1,3,1,1,0)
  ]);
  *)
  NN.AddLayer([
    TNNetInput.Create(4,4,1),
    TNNetDepthwiseConvLinear.Create(1,3,1,1)
  ]);

  TNNetInput(NN.Layers[0]).EnableErrorCollection;

  InputVolume := TNNetVolume.Create(NN.Layers[0].Output);
  OutputVolume := TNNetVolume.Create();

  InputVolume.Fill(10);
  NN.Layers[1].Neurons[0].Weights.Fill(0.1);

  NN.Compute(InputVolume);
  NN.GetOutput(OutputVolume);
  OutputVolume.Print();

  OutputVolume.Fill(0.1);
  NN.Backpropagate(OutputVolume);
  NN.Layers[0].OutputError.Print();

  NN.Free;
  OutputVolume.Free;
  InputVolume.Free;
end;


procedure CompareComputing(NN1, NN2: TNNet);
var
  InputVolume, OutputVolume1, OutputVolume2: TNNetVolume;
  SumDiff: TNeuralFloat;
begin
  InputVolume := TNNetVolume.Create(NN1.Layers[0].Output);
  OutputVolume1 := TNNetVolume.Create();
  OutputVolume2 := TNNetVolume.Create();

  if NN1.Layers[0] is TNNetInput then
  begin
    TNNetInput(NN1.Layers[0]).EnableErrorCollection;
  end;
  if NN2.Layers[0] is TNNetInput then
  begin
    TNNetInput(NN2.Layers[0]).EnableErrorCollection;
  end;

  InputVolume.Fill(10);
  NN1.Compute(InputVolume);
  NN2.Compute(InputVolume);

  NN1.GetOutput(OutputVolume1);
  NN2.GetOutput(OutputVolume2);

  SumDiff := OutputVolume1.SumDiff(OutputVolume2);

  if SumDiff <> 0 then
  begin
    WriteLn('Forward computing FAILS: ', SumDiff);
    WriteLn('Output sum 1: ', OutputVolume1.GetSum());
    WriteLn('Output sum 2: ', OutputVolume2.GetSum());
    OutputVolume1.DebugDiff(OutputVolume2, 0.1);
  end
  else
  begin
    WriteLn('Forward computing has passed.');
  end;

  OutputVolume1.Fill(0.001);
  NN1.Backpropagate(OutputVolume1);
  NN2.Backpropagate(OutputVolume1);
  SumDiff := NN1.Layers[0].OutputError.SumDiff(NN2.Layers[0].OutputError);

  if SumDiff <> 0 then
  begin
    WriteLn('Backprop error FAILS:', SumDiff);
    WriteLn('Backprop sum 1: ', NN1.Layers[0].OutputError.GetSum());
    WriteLn('Backprop sum 2: ', NN2.Layers[0].OutputError.GetSum());
    //NN1.Layers[0].OutputError.DebugDiff(NN2.Layers[0].OutputError, 0.001);
  end
  else
  begin
    WriteLn('Backpropagation computing has passed.');
  end;

  OutputVolume2.Free;
  OutputVolume1.Free;
  InputVolume.Free;
end;

procedure TestConvolutionAPI();
var
  NN: THistoricalNets;
  NN2: TNNet;
  AuxVolume: TNNetVolume;
  I: integer;
begin
  NN := THistoricalNets.Create();
  AuxVolume := TNNetVolume.Create;

  NN.AddLayer( TNNetInput.Create(32,32,3) );

  NN.AddLayer( TNNetConvolutionLinear.Create(32, 1, 0, 0) );
  NN.AddDenseNetBlockCAI
  (
        {pUnits=}4, {k=}32, {supressBias=}0,
        {PointWiseConv=}TNNetConvolutionLinear,
        {IsSeparable=}true,
        {HasNorm=}true,
        {pBeforeBottleNeck=}nil,
        {pAfterBottleNeck=}nil,
        {pBeforeConv=}nil,
        {pAfterConv=}TNNetHyperbolicTangent,
        {BottleNeck=}16,
        {Compression=}1,
        {DropoutRate=}0,
        {RandomBias=}0, {RandomAmplifier=}0,
        {FeatureSize=}3
  );
  NN.AddLayer( TNNetConvolutionReLU.Create(16,5,0,0) );
  NN.AddLayer( TNNetMaxPool.Create(2) );
  NN.AddLayer( TNNetConvolutionReLU.Create(128,5,0,0) );
  NN.AddLayer( TNNetMaxPool.Create(2) );
  NN.AddLayer( TNNetCellBias.Create() );
  NN.AddLayer( TNNetConvolutionReLU.Create(128,5,0,0) );
  NN.AddLayer( TNNetConvolutionLinear.Create(32,5,0,0) );
  NN.AddLayer( TNNetGroupedConvolutionLinear.Create(32,1,0,0,4,0) );
  NN.AddLayer( TNNetConvolution.Create(32,5,0,0) );
  NN.AddLayer( TNNetFullConnectReLU.Create(32) );
  NN.AddLayer( TNNetFullConnectReLU.Create(10) );
  NN.AddLayer( TNNetFullConnectLinear.Create(10) );
  NN.AddLayer( TNNetFullConnect.Create(10) );
  NN.AddLayer( TNNetHyperbolicTangent.Create() );
  NN.AddLayer( TNNetReLU.Create() );

  NN2 := NN.Clone();

  CompareNNStructure(NN, NN2);
  TestDataParallelism(NN);

  NN2.Free;
  NN2 := NN.Clone();

  CompareComputing(NN, NN2);

  NN.Clear;
  (*
  WriteLn('Test Grouped Convolution:');

  NN.AddLayer( TNNetInput.Create(3,3,4).EnableErrorCollection() );
  NN.AddLayer( TNNetGroupedConvolutionLinear.Create(4,3,1,1,4,1) );

  AuxVolume.Resize(3,3,4);
  AuxVolume.FillForDebug();
  AuxVolume.Mul(100);AuxVolume.Add(1);
  WriteLn('Input:');
  AuxVolume.PrintWithIndex();

  for I:=0 to 3 do
  begin
    WriteLn(I, ' weights:');
    NN.Layers[1].Neurons[I].Weights.PrintWithIndex();
  end;

  NN.Layers[1].Neurons[0].Weights.FData[0] := 1;
  NN.Layers[1].Neurons[1].Weights.FData[0] := 2;
  NN.Layers[1].Neurons[2].Weights.FData[0] := 3;
  NN.Layers[1].Neurons[3].Weights.FData[0] := 4;
  NN.Layers[1].AfterWeightUpdate();

  NN.Compute(AuxVolume);
  NN.GetOutput(AuxVolume);

  WriteLn('Output:');
  AuxVolume.PrintWithIndex();

  AuxVolume.FillForDebug();AuxVolume.Mul(100);
  NN.Backpropagate(AuxVolume);

  WriteLn('Error at grouped conv:');
  NN.Layers[1].OutputError.PrintWithIndex();

  WriteLn('Error at input:');
  NN.Layers[0].OutputError.PrintWithIndex();
  *)
  (*
  WriteLn('Test DeMaxPool:');

  NN.AddLayer( TNNetInput.Create(2,2,1) );
  NN.AddLayer( TNNetDeMaxPool.Create(2, 1) );
  AuxVolume.Resize(2,2,1);
  AuxVolume.FillForDebug();
  NN.Compute(AuxVolume);
  NN.GetOutput(AuxVolume);

  WriteLn('MaxPool Output:');
  AuxVolume.PrintWithIndex();
  NN.AddLayer( TNNetConvolutionLinear.Create(1,3,0,0) );

  AuxVolume.Resize(2,2,1);
  AuxVolume.FillForDebug();
  NN.Compute(AuxVolume);
  NN.GetOutput(AuxVolume);
  WriteLn('Convolution Output:');
  AuxVolume.PrintWithIndex();

  WriteLn('Test MaxPool:');

  NN.AddLayer( TNNetInput.Create(4,4,1) );
  NN.AddLayer( TNNetMaxPool.Create(3,2) );

  AuxVolume.Resize(4,4,1);
  AuxVolume.FillForDebug();

  AuxVolume.Print();
  AuxVolume.PrintDebug(); WriteLn;

  // 0  1  2  3
  // 4  5  6  7
  // 8  9 10 11
  //12 13 14 15
  AuxVolume.PrintWithIndex();

  NN.Compute(AuxVolume);
  NN.GetOutput(AuxVolume);

  WriteLn('MaxPool Output:');
  AuxVolume.PrintWithIndex();
  AuxVolume.PrintDebug();

  WriteLn;
  *)
  WriteLn('Testing has finished.');
  AuxVolume.Free;
  NN.Free;
  NN2.Free;
end;

{$IFDEF OpenCL}
procedure TestConvolutionOpenCL(platform_id: cl_platform_id; device_id: cl_device_id);
var
  NN: TNNet;
  Input, Output, Output2: TNNetVolume;
  NRelu: TNNetConvolutionReLU;
  Arr1: array[0..0] of Single;
  Arr4: array[0..3] of Single;
  I: integer;
  ErrorCount: integer;
begin
  WriteLn('Test Convolution API with OpenCL');
  WriteLn(' ---------- Test 1 ----------');
  NN := TNNet.Create();
  Input := TNNetVolume.Create([1.0, 2.0, 3.0, 4.0]);
  Output := TNNetVolume.Create(2, 2, 1);

  NRelu := TNNetConvolutionReLU.Create(2,1,0,0);

  NN.AddLayer( TNNetInput.Create(2, 2, 1) );
  NN.AddLayer( NRelu );

  Arr1[0] := 1;
  NRelu.Neurons[0].Weights.Copy(Arr1);

  Arr1[0] := 10;
  NRelu.Neurons[1].Weights.Copy(Arr1);
  NRelu.AfterWeightUpdate();

  NN.Compute(Input);
  NN.GetOutput(Output);
  WriteLn('Test 1) CPU Output:');Output.Print();

  NN.EnableOpenCL(platform_id, device_id);
  NN.Compute(Input);
  NN.GetOutput(Output);
  WriteLn('Test 1) OpenCL Output:');Output.Print();
  Output.Free;
  Input.Free;

  WriteLn(' ---------- Test 2 ----------');
  NN.Clear();
  NN.DisableOpenCL();

  Input := TNNetVolume.Create([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
  Output := TNNetVolume.Create(2, 2, 2);

  NRelu := TNNetConvolutionReLU.Create(2,2,0,0);

  NN.AddLayer( TNNetInput.Create(3, 3, 1) );
  NN.AddLayer( NRelu );

  Arr4[0] := 1;
  Arr4[1] := 2;
  Arr4[2] := 3;
  Arr4[3] := 4;

  NRelu.Neurons[0].Weights.Copy(Arr4);

  Arr4[0] := 10;
  Arr4[1] := 20;
  Arr4[2] := 30;
  Arr4[3] := 40;

  NRelu.Neurons[1].Weights.Copy(Arr4);
  NRelu.AfterWeightUpdate();

  NN.Compute(Input);
  NN.GetOutput(Output);
  WriteLn('Test 2) CPU Output:');Output.Print();

  NN.EnableOpenCL(platform_id, device_id);
  NN.Compute(Input);
  NN.GetOutput(Output);
  WriteLn('Test 2) OpenCL Output:');Output.Print();
  Output.Free;
  Input.Free;

  WriteLn(' ---------- Test 3 ----------');
  ErrorCount :=0;

  //NN.Clear();
  //NN.DisableOpenCL();

  NN.Free;
  NN := TNNet.Create();

  Input := TNNetVolume.Create(32, 32, 16);
  Output := TNNetVolume.Create(32, 32, 16);
  Output2 := TNNetVolume.Create(32, 32, 16);

  NRelu := TNNetConvolutionReLU.Create(16,3,1,1);

  NN.AddLayer( TNNetInput.Create(32, 32, 16) );
  NN.AddLayer( NRelu );
  NN.DebugStructure();

  WriteLn('Test 3) CPU Output:');
  Input.FillForDebug();
  NN.Compute(Input);
  NN.GetOutput(Output);
  Output.PrintDebug(); WriteLn;

  WriteLn('Test 3) OpenCL Output:');
  NN.EnableOpenCL(platform_id, device_id);
  Input.FillForDebug();
  NN.Compute(Input);
  NN.GetOutput(Output2);
  Output2.PrintDebug(); WriteLn;

  for I := 0 to Output.Size - 1 do
  begin
    if Abs( Output.Raw[I] - Output2.Raw[I] ) > 0.1 then
    begin
      if ErrorCount < 10 then WriteLn('Error at pos ',I,':',Output.Raw[I]:10:5,' ',Output2.Raw[I]:10:5);
      ErrorCount := ErrorCount + 1;
    end;
  end;
  WriteLn(' Error Count:', ErrorCount);

  Output.Free;
  Input.Free;
  NN.Free;
end;

procedure TestFullConnectOpenCL(platform_id: cl_platform_id;
  device_id: cl_device_id);
var
  NN: TNNet;
  NReLU: TNNetLayer;
  Input, Output: TNNetVolume;
begin
  WriteLn('Test FULL CONNECT Layer with OpenCL');
  WriteLn(' ---------- Test 1 ----------');
  NN := TNNet.Create();
  Input := TNNetVolume.Create([1.0, 2.0, 3.0, 4.0]);
  NRelu := TNNetFullConnectReLU.Create(1024,0);
  Output := TNNetVolume.Create(8, 1, 1);

  NN.AddLayer( TNNetInput.Create(2, 2, 1) );
  NN.AddLayer( NRelu );
  NN.AddLayer( TNNetFullConnectReLU.Create(8,0) );
  NN.Compute(Input, Output);
  Output.Print();

  NN.EnableOpenCL(platform_id, device_id);
  NN.Compute(Input, Output);
  Output.Print();

  Output.Free;
  Input.Free;
  NN.Free;
end;

{$ENDIF}

{ TNNetDataParallelism }
constructor TNNetDataParallelism.Create(CloneNN: TNNet; pSize: integer; pFreeObjects: Boolean = True);
var
  NNData: String;
  I: integer;
  NN: TNNet;
begin
  inherited Create(pFreeObjects);
  NNData := CloneNN.SaveToString();

  for I := 1 to pSize do
  begin
    NN := TNNet.Create;
    NN.LoadFromString(NNData);
    Self.Add(NN);
  end;
end;

constructor TNNetDataParallelism.Create(pSize: integer; pFreeObjects: Boolean);
var
  I: integer;
  NN: TNNet;
begin
  inherited Create(pFreeObjects);

  for I := 1 to pSize do
  begin
    NN := TNNet.Create;
    Self.Add(NN);
  end;
end;

procedure TNNetDataParallelism.SetLearningRate(pLearningRate, pInertia: TNeuralFloat);
var
  I: integer;
begin
  if Count > 0 then
  begin
    for I := 0 to Count - 1 do
    begin
      Items[I].SetLearningRate(pLearningRate, pInertia);
    end;
  end;
end;

procedure TNNetDataParallelism.SetBatchUpdate(pBatchUpdate: boolean);
var
  I: integer;
begin
  if Count > 0 then
  begin
    for I := 0 to Count - 1 do
    begin
      Items[I].SetBatchUpdate(pBatchUpdate);
    end;
  end;
end;

procedure TNNetDataParallelism.SetL2Decay(pL2Decay: TNeuralFloat);
var
  I: integer;
begin
  if Count > 0 then
  begin
    for I := 0 to Count - 1 do
    begin
      Items[I].SetL2Decay(pL2Decay);
    end;
  end;
end;

procedure TNNetDataParallelism.SetL2DecayToConvolutionalLayers(pL2Decay: TNeuralFloat);
var
  I: integer;
begin
  if Count > 0 then
  begin
    for I := 0 to Count - 1 do
    begin
      Items[I].SetL2DecayToConvolutionalLayers(pL2Decay);
    end;
  end;
end;

procedure TNNetDataParallelism.EnableDropouts(pFlag: boolean);
var
  I: integer;
begin
  if Count > 0 then
  begin
    for I := 0 to Count - 1 do
    begin
      Items[I].EnableDropouts(pFlag);
    end;
  end;
end;

procedure TNNetDataParallelism.CopyWeights(Origin: TNNet);
var
  I: integer;
begin
  if Count > 0 then
  begin
    for I := 0 to Count - 1 do
    begin
      Items[I].CopyWeights(Origin);
    end;
  end;
end;

procedure TNNetDataParallelism.SumWeights(Destin: TNNet);
var
  I: integer;
begin
  if Count > 0 then
  begin
    Destin.CopyWeights( Items[0] );
    if Count > 1 then
    begin
      for I := 1 to Count - 1 do
      begin
        Destin.SumWeights( Items[I] );
      end;
    end;
  end;
end;

procedure TNNetDataParallelism.SumDeltas(Destin: TNNet);
var
  I: integer;
begin
  if Count > 0 then
  begin
    Destin.SumDeltas( Items[0] );
    Items[0].ClearDeltas();
    if Count > 1 then
    begin
      for I := 1 to Count - 1 do
      begin
        Destin.SumDeltas( Items[I] );
        Items[I].ClearDeltas();
      end;
    end;
  end;
end;

procedure TNNetDataParallelism.AvgWeights(Destin: TNNet);
var
  AuxCount: TNeuralFloat;
begin
  if Count > 0 then
  begin
    SumWeights(Destin);
    if (Count > 1) then
    begin
      AuxCount := Count;
      Destin.MulWeights(1/AuxCount);
      //Destin.ClearInertia();
    end;
  end;
end;

procedure TNNetDataParallelism.ReplaceAtIdxAndUpdateWeightAvg(Idx: integer; NewNet,
  AverageNet: TNNet);
var
  OneDivCount: TNeuralFloat;
begin
  if Count > 0 then
  begin
    OneDivCount := 1/Count;
    // Removes the element from the average.
    AverageNet.MulAddWeights(-OneDivCount, Self[Idx]);
    // Adds the new element to the average.
    AverageNet.MulAddWeights(OneDivCount, NewNet);
    Self[Idx].CopyWeights(NewNet);
  end
end;

{$IFDEF OpenCL}
procedure TNNetDataParallelism.DisableOpenCL();
var
  I: integer;
begin
  if Count > 0 then
  begin
    for I := 0 to Count - 1 do
    begin
      Items[I].DisableOpenCL();
    end;
  end;
end;

procedure TNNetDataParallelism.EnableOpenCL(platform_id: cl_platform_id;
  device_id: cl_device_id);
var
  I: integer;
begin
  if Count > 0 then
  begin
    for I := 0 to Count - 1 do
    begin
      Items[I].EnableOpenCL(platform_id, device_id);
    end;
  end;
end;
{$ENDIF}

{ TNNetLayerStdNormalization }
procedure TNNetLayerStdNormalization.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  FOutput.ZeroCenter();
  FLastStdDev := Sqrt(FOutput.GetSumSqr()/FOutput.Size);
  if FLastStdDev > 0 then
  begin
    FOutput.Divi(FLastStdDev);
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetLayerStdNormalization.Backpropagate();
var
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if FLastStdDev <> 0 then
  begin
    FOutputError.Divi(FLastStdDev);
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate;
end;

{ TNNetLayerMaxNormalization }
procedure TNNetLayerMaxNormalization.Compute;
var
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute;
  FLastMax := FOutput.GetMaxAbs();

  if FLastMax <> 0 then
  begin
    FOutput.Divi(FLastMax);
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetLayerMaxNormalization.Backpropagate;
var
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if FLastMax > 0 then
  begin
    FOutputError.Divi(FLastMax);
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate;
end;

{ TNNetDropout }
procedure TNNetDropout.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FDropoutMask.ReSize(FOutput);
end;

constructor TNNetDropout.Create(Rate: double; OneMaskPerbatch: integer = 1);
begin
  inherited Create();
  FStruct[1] := OneMaskPerbatch;
  if (Rate > 0) then
  begin
    FRate := Round(1/Rate);
    FStruct[0] := FRate;
    FEnabled := true;
  end
  else
  begin
    FRate := -1;
    FStruct[0] := -1;
    FEnabled := false;
  end;
  FDropoutMask := TNNetVolume.Create();
end;

destructor TNNetDropout.Destroy();
begin
  FDropoutMask.Free();
  inherited Destroy();
end;

procedure TNNetDropout.CopyWeights(Origin: TNNetLayer);
begin
  inherited CopyWeights(Origin);
  if FBatchUpdate then FDropoutMask.Copy(TNNetDropout(Origin).DropoutMask);
end;

procedure TNNetDropout.Compute();
var
  CntOut, MaxOutput: integer;
  StartTime: double;
begin
  StartTime := Now();
  inherited Compute();

  if ( FEnabled and (FRate>0) ) then
  begin
    if FBatchUpdate and ({OneMaskPerbatch}FStruct[1]>0) then
    begin
      FOutput.Mul(FDropoutMask);
    end
    else
    begin
      FDropoutMask.Fill(1);
      MaxOutput := FOutput.Size - 1;
      for CntOut := 0 to MaxOutput do
      begin
        if (Random(FRate) = 0) then
        begin
          FOutput.FData[CntOut] := 0;
          FDropoutMask.FData[CntOut] := 0;
        end;
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetDropout.Backpropagate();
var
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if ( FEnabled and (FOutputError.Size = FOutput.Size) ) then FOutputError.Mul(FDropoutMask);
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  inherited Backpropagate();
end;

procedure TNNetDropout.RefreshDropoutMask();
var
  CntOut, MaxOutput: integer;
begin
  FDropoutMask.Fill(1);
  MaxOutput := FDropoutMask.Size - 1;
  for CntOut := 0 to MaxOutput do
  begin
    if (Random(FRate) = 0) then
    begin
      FDropoutMask.FData[CntOut] := 0;
    end;
  end;
  // Dropout mask debug: WriteLn('Dropoutmask sum is:', FDropoutMask.GetSum():6:2, ' Size:', MaxOutput + 1);
end;

procedure TNNetAvgPool.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  SetLength(FMaxPosX, 0);
  SetLength(FMaxPosY, 0);
end;

constructor TNNetAvgPool.Create(pPoolSize: integer);
begin
  inherited Create(pPoolSize);
end;

{ TNNetAvgPool }
procedure TNNetAvgPool.Compute();
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  OutX, OutY: integer;
  StartTime: double;
begin
  StartTime := Now();
  Output.Fill(0);
  MaxX := FPrevLayer.Output.SizeX - 1;
  MaxY := FPrevLayer.Output.SizeY - 1;

  for CntX := 0 to MaxX do
  begin
    OutX := CntX div FPoolSize;
    for CntY := 0 to MaxY do
    begin
      OutY := CntY div FPoolSize;
      FOutput.Add
      (
        FOutput.GetRawPtr(OutX, OutY),
        FPrevLayer.Output.GetRawPtr(CntX, CntY),
        FPrevLayer.Output.Depth
      );
    end;
  end;
  Output.Divi(FPoolSize*FPoolSize);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetAvgPool.Backpropagate();
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  OutX, OutY: integer;
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if
    (FBackPropCallCurrentCnt < FDepartingBranchesCnt) or
    (FPrevLayer.FOutput.Size <> FPrevLayer.FOutputError.Size) then exit;
  StartTime := Now();
  OutputError.Divi(FPoolSize*FPoolSize);
  MaxX := FPrevLayer.Output.SizeX - 1;
  MaxY := FPrevLayer.Output.SizeY - 1;
  for CntX := 0 to MaxX do
  begin
    OutX := CntX div FPoolSize;
    for CntY := 0 to MaxY do
    begin
      OutY := CntY div FPoolSize;
      FOutput.Add
      (
        FPrevLayer.OutputError.GetRawPtr(CntX, CntY),
        FOutputError.GetRawPtr(OutX, OutY),
        FPrevLayer.Output.Depth
      );
    end;
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

procedure TNNetDeMaxPool.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  SetLength(FMaxPosX, 0);
  SetLength(FMaxPosY, 0);
end;

{ TNNetDeMaxPool }
function TNNetDeMaxPool.CalcOutputSize(pInputSize: integer): integer;
begin
  Result := pInputSize * FPoolSize;
end;

constructor TNNetDeMaxPool.Create(pPoolSize: integer; pSpacing: integer = 0);
begin
  inherited Create(pPoolSize);
  FSpacing := pSpacing;
  FStruct[7] := FSpacing;
end;

procedure TNNetDeMaxPool.Compute();
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  OutX, OutY: integer;
  CurrValue: TNeuralFloat;
  StartTime: double;
  PrevLayerRawPos, OutputRawPos: integer;
begin
  StartTime := Now();
  Output.Fill(0);
  MaxX := FPrevLayer.Output.SizeX - 1;
  MaxY := FPrevLayer.Output.SizeY - 1;
  MaxD := FPrevLayer.Output.Depth - 1;

  if FSpacing = 1 then
  begin
    for CntX := 0 to MaxX do
    begin
      OutX := CntX*FPoolSize + Random(FPoolSize);
      for CntY := 0 to MaxY do
      begin
        OutY := CntY*FPoolSize + Random(FPoolSize);
        PrevLayerRawPos := FPrevLayer.FOutput.GetRawPos(CntX,CntY,0);
        OutputRawPos := FOutput.GetRawPos(OutX,OutY,0);
        for CntD := 0 to MaxD do
        begin
          //CurrValue := FPrevLayer.FOutput[CntX,CntY,CntD];
          //FOutput[OutX,OutY,CntD] := CurrValue;
          FOutput.FData[OutputRawPos] := FPrevLayer.FOutput.FData[PrevLayerRawPos];
          Inc(PrevLayerRawPos);
          Inc(OutputRawPos);
        end;
      end;
    end;
  end
  else
  begin
    for CntD := 0 to MaxD do
    begin
      for CntX := 0 to MaxX do
      begin
        for CntY := 0 to MaxY do
        begin
          CurrValue := FPrevLayer.Output[CntX,CntY,CntD];
          for OutX := CntX*FPoolSize to CntX*FPoolSize + FPoolSize - 1 do
          begin
            for OutY := CntY*FPoolSize to CntY*FPoolSize + FPoolSize - 1 do
            begin
              Output[OutX,OutY,CntD] := CurrValue;
            end;
          end;
        end;
      end;
    end;
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetDeMaxPool.Backpropagate();
var
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size) then
  begin
    ComputePreviousLayerError;
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

procedure TNNetDeMaxPool.ComputePreviousLayerError();
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  RawPos, PrevRawPos: integer;
  PrevPosX, PrevPosY: integer;
  floatPoolSize: TNeuralFloat;
  OutX, OutY: integer;
begin
  MaxD := Output.Depth - 1;

  floatPoolSize := FPoolSize;

(*
  if FSpacing = 1 then
  begin
    MaxX := FPrevLayer.FOutput.SizeX - 1;
    MaxY := FPrevLayer.FOutput.SizeY - 1;
    for CntX := 0 to MaxX do
    begin
      OutX := CntX*FPoolSize;
      for CntY := 0 to MaxY do
      begin
        OutY := CntY*FPoolSize;
        PrevRawPos := FPrevLayer.FOutputError.GetRawPos(CntX,CntY,0);
        RawPos := FOutputError.GetRawPos(OutX,OutY,0);
        for CntD := 0 to MaxD do
        begin
          {$IFDEF FPC}
          FPrevLayer.FOutputError.FData[PrevRawPos] += FOutputError.FData[RawPos];
          {$ELSE}
          FPrevLayer.FOutputError.FData[PrevRawPos] :=
            FPrevLayer.FOutputError.FData[PrevRawPos] + FOutputError.FData[RawPos];
          {$ENDIF}
          Inc(PrevRawPos);
          Inc(RawPos);
        end;
      end;
    end;
  end
  else *)
  begin
    MaxX := FOutput.SizeX - 1;
    MaxY := FOutput.SizeY - 1;
    if (FSpacing=0) then FOutputError.Divi(floatPoolSize);
    for CntY := 0 to MaxY do
    begin
      PrevPosY := CntY div FPoolSize;
      for CntX := 0 to MaxX do
      begin
        PrevPosX := CntX div FPoolSize;
        RawPos := FOutput.GetRawPos(CntX, CntY, 0);
        PrevRawPos := FPrevLayer.FOutputError.GetRawPos(PrevPosX, PrevPosY, 0);
        for CntD := 0 to MaxD do
        begin
          if (FSpacing=0) or (FOutPut.FData[RawPos]<>0) then
          begin
            {$IFDEF FPC}
            FPrevLayer.FOutputError.FData[PrevRawPos] += FOutPutError.FData[RawPos];
            {$ELSE}
            FPrevLayer.FOutputError.FData[PrevRawPos] :=
              FPrevLayer.FOutputError.FData[PrevRawPos] + FOutPutError.FData[RawPos];
            {$ENDIF}
          end;
          Inc(RawPos);
          Inc(PrevRawPos);
        end;
      end;
    end;
  end;
end;

{ TNNetDeLocalConnectReLU }
constructor TNNetDeLocalConnectReLU.Create(pNumFeatures, pFeatureSize: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures,pFeatureSize,pFeatureSize-1,0,pSuppressBias);
end;

{ TNNetDeLocalConnect }
constructor TNNetDeLocalConnect.Create(pNumFeatures, pFeatureSize: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures,pFeatureSize,pFeatureSize-1,0,pSuppressBias);
end;

{ TNNetDeconvolutionReLU }
constructor TNNetDeconvolutionReLU.Create(pNumFeatures, pFeatureSize: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures,pFeatureSize,pFeatureSize-1,0,pSuppressBias);
end;

{ TNNetDeconvolution }
constructor TNNetDeconvolution.Create(pNumFeatures, pFeatureSize: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures,pFeatureSize,pFeatureSize-1,0,pSuppressBias);
end;

{ TNNetConcat }
constructor TNNetConcat.Create(pSizeX, pSizeY, pDepth: integer;
  aL: array of TNNetLayer);
var
  LayerCnt: integer;
  NewSize: integer;
begin
  inherited Create();

  FStruct[0] := pSizeX;
  FStruct[1] := pSizeY;
  FStruct[2] := pDepth;

  for LayerCnt := Low(aL) to High(aL) do
  begin
    if aL[LayerCnt] is TNNetInput then
    begin
      TNNetInput(aL[LayerCnt]).EnableErrorCollection;
    end;
    FPrevOutput.Add(aL[LayerCnt].FOutput);
    FPrevOutputError.Add(aL[LayerCnt].FOutputError);
    FPrevOutputErrorDeriv.Add(aL[LayerCnt].FOutputErrorDeriv);
    FPrevLayerList.Add(aL[LayerCnt]);
    aL[LayerCnt].IncDepartingBranchesCnt();
  end;

  FActivationFn := aL[0].ActivationFn;
  FActivationFnDerivative := aL[0].ActivationFnDerivative;

  if (pSizeX>0) and (pSizeY>0) and (pDepth>0) then
  begin
    FOutput.Resize(pSizeX, pSizeY, pDepth);
    FOutputError.Resize(pSizeX, pSizeY, pDepth);
    FOutputErrorDeriv.Resize(pSizeX, pSizeY, pDepth);
  end
  else
  begin
    NewSize := FPrevOutput.GetTotalSize();
    FOutput.Resize(NewSize, 1, 1);
    FOutputError.Resize(NewSize, 1, 1);
    FOutputErrorDeriv.Resize(NewSize, 1, 1);
  end;
end;

constructor TNNetConcat.Create(aL: array of TNNetLayer);
begin
  Self.Create(0, 0, 0, aL);
end;

procedure TNNetConcat.Compute;
var
  StartTime: double;
begin
  StartTime := Now();
  FPrevOutput.ConcatInto(FOutput);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetConcat.Backpropagate;
var
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  FPrevOutputError.SplitFrom(FOutputError);
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  BackpropagateConcat();
end;

{ TNNetReshape }
procedure TNNetReshape.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FActivationFn := pPrevLayer.ActivationFn;
  FActivationFnDerivative := pPrevLayer.ActivationFnDerivative;
end;

constructor TNNetReshape.Create(pSizeX, pSizeY, pDepth: integer);
begin
  inherited Create();
  FOutput.Resize(pSizeX, pSizeY, pDepth);
  FOutputError.Resize(pSizeX, pSizeY, pDepth);
  FOutputErrorDeriv.Resize(pSizeX, pSizeY, pDepth);
  FStruct[0] := pSizeX;
  FStruct[1] := pSizeY;
  FStruct[2] := pDepth;
end;

procedure TNNetReshape.Compute;
var
  Len: integer;
  StartTime: double;
begin
  StartTime := Now();
  Len := Min(FOutput.Size, FPrevLayer.FOutput.Size);
  FOutput.Copy(FPrevLayer.FOutput, Len);
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetReshape.Backpropagate;
var
  Len: integer;
  StartTime: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  Len := Min(FOutput.Size, FPrevLayer.FOutput.Size);
  //TODO: check this for possible crash.
  FPrevLayer.FOutputError.Add(FOutputError);
  FBackwardTime := FBackwardTime + (Now() - StartTime);
  FPrevLayer.Backpropagate();
end;

{ TNNetIdentity }
procedure TNNetIdentity.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FOutput.ReSize(pPrevLayer.FOutput);
  FOutputError.ReSize(pPrevLayer.FOutputError);
  FOutputErrorDeriv.ReSize(pPrevLayer.FOutputErrorDeriv);
end;

procedure TNNetIdentity.Compute;
begin
  FOutput.CopyNoChecks(FPrevLayer.FOutput);
end;

procedure TNNetIdentity.Backpropagate;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if Assigned(FPrevLayer) and
    (FPrevLayer.OutputError.Size > 0) and
    (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size) then
  begin
    FPrevLayer.FOutputError.Add(FOutputError);
  end;
  FPrevLayer.Backpropagate();
end;

{ TNNetLocalConnectReLU }
constructor TNNetLocalConnectReLU.Create(pNumFeatures, pFeatureSize,
  pInputPadding, pStride: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pSuppressBias);

  FActivationFn := @RectifiedLinearUnit;
  FActivationFnDerivative := @RectifiedLinearUnitDerivative;
end;

procedure TNNetLocalConnect.BackpropagateAtOutputPos(OutputX, OutputY,
  OutputD: integer);
var
  CntY: integer;
  MaxY: integer;
  LocalLearningErrorDeriv: TNeuralFloat;
  LocalOutputErrorDeriv: TNeuralFloat;
  LocalDelta, LocalWeight, LocalPrevError: TNNetVolume;
  CalculatePrevLayerError, CalculateDelta: boolean;
  OutputSize: TNeuralFloat;
  SrcPtr, DestPtr: TNeuralFloatArrPtr;
  SizeXDepth: integer;
  NeuronIdx: integer;
  LocalNeuron: TNNetNeuron;
begin
  NeuronIdx := FOutput.GetRawPos(OutputX, OutputY, OutputD);
  LocalNeuron := FNeurons[NeuronIdx];
  LocalNeuron.ClearDelta;
  LocalOutputErrorDeriv := OutputErrorDeriv.FData[NeuronIdx];

  if (LocalOutputErrorDeriv <> 0) then
  begin
    MaxY := FFeatureSizeY - 1;
    LocalWeight := LocalNeuron.Weights;
    LocalDelta  := LocalNeuron.Delta;
    LocalPrevError := FPrevLayer.OutputError;

    if FSmoothErrorPropagation then
    begin
      OutputSize := FFeatureSizeX * FFeatureSizeY;//Output.SizeX * Output.SizeY;
    end
    else
    begin
      OutputSize := 1;
    end;

    LocalLearningErrorDeriv := (-FLearningRate) * LocalOutputErrorDeriv;

    {$IFDEF FPC}
    FNeurons[NeuronIdx].FBiasDelta += LocalLearningErrorDeriv;
    {$ELSE}
    FNeurons[NeuronIdx].FBiasDelta := FNeurons[NeuronIdx].FBiasDelta +
      LocalLearningErrorDeriv;
    {$ENDIF}

    CalculatePrevLayerError := Assigned(FPrevLayer) and
      (FPrevLayer.OutputError.Size > 0) and
      (LocalOutputErrorDeriv <> 0.0) and
      (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size);

    CalculateDelta := LocalLearningErrorDeriv <> 0.0;

    if (CalculateDelta) then
    begin
      LocalDelta.MulAdd(LocalLearningErrorDeriv, FInputPrepared.GetRawPtr(OutputX, OutputY, 0));
    end;

    if
      (CalculatePrevLayerError) and
      (OutputX >= FPadding) and (OutputY >= FPadding) and
      (OutputX < OutputError.SizeX - FPadding) and
      (OutputY < OutputError.SizeY - FPadding) then
    begin
      SizeXDepth := FFeatureSizeX * FInputCopy.Depth;
      for CntY := 0 to MaxY do
      begin
        DestPtr  := LocalPrevError.GetRawPtr( (OutputX-FPadding)*FStride, (OutputY-FPadding)*FStride + CntY, 0) ;
        SrcPtr   := LocalWeight.GetRawPtr(0, CntY, 0);
        LocalPrevError.MulAdd(DestPtr, SrcPtr, LocalOutputErrorDeriv / OutputSize, SizeXDepth);
      end;
    end;
  end;// of if (LocalOutputErrorDeriv <> 0)
end;

{ TNNetLocalConnect }
procedure TNNetLocalConnect.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  AddMissingNeurons(Output.Size);
  SetNumWeightsForAllNeurons(FFeatureSizeX, FFeatureSizeY, pPrevLayer.Output.Depth);
  InitDefault();
end;

procedure TNNetLocalConnect.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  if FNeurons.Count > 0 then
  begin
    if FPadding > 0
      then FInputCopy.CopyPadding(FPrevLayer.Output, FPadding)
      else FInputCopy := FPrevLayer.Output;

    FInputPrepared.ReSize(FOutput.SizeX, FOutput.SizeY, FInputCopy.Depth * FFeatureSizeX * FFeatureSizeY);
    PrepareInputForConvolutionFast();
    ComputeCPU();
  end
  else
  begin
    FErrorProc('Neuronal layer contains no neuron:'+ IntToStr(FNeurons.Count));
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetLocalConnect.ComputeCPU();
var
  OutputCntX, OutputCntY, OutputCntD: integer;
  InputCntX, InputCntY: integer;
  MaxX, MaxY, MaxD: integer;
  LocalSize: integer;
  LocalW: TNNetVolume;
  PtrA, PtrB: TNeuralFloatArrPtr;
  NeuronIdx: integer;
  Sum: TNeuralFloat;
  CntXYD: integer;
begin
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;
  MaxD := FOutput.Depth - 1;

  LocalSize := FFeatureSizeX*FFeatureSizeY*FInputCopy.Depth;
  InputCntX := 0;
  OutputCntX := 0;
  CntXYD := 0;
  while OutputCntX <= MaxX do
  begin
    InputCntY := 0;
    OutputCntY := 0;
    while OutputCntY <= MaxY do
    begin
      OutputCntD := 0;
      PtrA := FInputPrepared.GetRawPtr(OutputCntX, OutputCntY, 0);
      while OutputCntD <= MaxD do
      begin
        NeuronIdx := FOutput.GetRawPos(OutputCntX, OutputCntY, OutputCntD);
        LocalW := FNeurons[NeuronIdx].Weights;
        PtrB := LocalW.GetRawPtr(0, 0, 0);

        Sum := LocalW.DotProduct(PtrA, PtrB, LocalSize);
        if FSuppressBias = 0 then Sum := Sum + FNeurons[NeuronIdx].FBiasWeight;

        FOutputRaw.FData[NeuronIdx] := Sum;
        FOutput.FData[NeuronIdx] := FActivationFn(Sum);
        Inc(OutputCntD);
        Inc(CntXYD);
      end;
      Inc(InputCntY, FStride);
      Inc(OutputCntY);
    end;
    Inc(InputCntX, FStride);
    Inc(OutputCntX);
  end;
end;

procedure TNNetLocalConnect.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FNeurons.Count = FOutput.Size) and (FPrevLayer.Output.Size > 0) then
  begin
    StartTime := Now();
    ComputeErrorDeriv();
    BackpropagateCPU();
    FBackwardTime := FBackwardTime + (Now() - StartTime);
    {$IFDEF CheckRange}ForceRangeWeights(1000);{$ENDIF}
  end else
  begin
    FErrorProc
    (
      'TNNetLocalConnect.Backpropagate should have same sizes.' +
      'Neurons:' + IntToStr(FNeurons.Count) +
      ' Output Size:' + IntToStr(FOutput.Size) +
      ' PrevLayer:' + IntToStr(FPrevLayer.Output.Size)
    );
  end;
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

procedure TNNetLocalConnect.BackpropagateCPU();
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  NeuronIdx: integer;
begin
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxD := OutputError.Depth - 1;

  for CntD := 0 to MaxD do
  begin
    for CntY := 0 to MaxY do
    begin
      for CntX := 0 to MaxX do
      begin
        BackpropagateAtOutputPos(CntX, CntY, CntD);
      end;
    end;
  end;

  if (not FBatchUpdate) then
  begin
    for NeuronIdx := 0 to MaxD do FNeurons[NeuronIdx].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
end;

procedure TNNetFullConnectReLU.ComputePreviousLayerErrorCPU();
var
  MaxOutputCnt: integer;
  OutputCnt: integer;
  LocalPrevError: TNNetVolume;
begin
  LocalPrevError := FPrevLayer.OutputError;

  MaxOutputCnt := FOutput.Size - 1;
  for OutputCnt := 0 to MaxOutputCnt do
  begin
    if (FOutputRaw.FData[OutputCnt] > 0.0) and (FOutputError.FData[OutputCnt] <> 0.0) then
    begin
      LocalPrevError.MulAdd(FOutputError.FData[OutputCnt], FArrNeurons[OutputCnt].FWeights);
    end;
  end;
end;

procedure TNNetFullConnectReLU.ComputeCPU();
var
  Cnt, MaxCnt: integer;
  Sum: TNeuralFloat;
begin
  MaxCnt := FNeurons.Count - 1;
  if FSuppressBias = 0 then
  begin
    for Cnt := 0 to MaxCnt do
    begin
      Sum :=
        FArrNeurons[Cnt].Weights.DotProduct(FPrevLayer.Output) +
        FArrNeurons[Cnt].FBiasWeight;
      FOutputRaw.FData[Cnt] := Sum;
      if Sum > 0
        then FOutput.FData[Cnt] := Sum
        else FOutput.FData[Cnt] := 0;
    end;
  end
  else for Cnt := 0 to MaxCnt do
  begin
    Sum :=
      FArrNeurons[Cnt].Weights.DotProduct(FPrevLayer.Output);
    FOutputRaw.FData[Cnt] := Sum;
    if Sum > 0
      then FOutput.FData[Cnt] := Sum
      else FOutput.FData[Cnt] := 0;
  end;
end;

procedure TNNetFullConnectReLU.BackpropagateCPU();
var
  MaxNeurons, NeuronCnt: integer;
  localLearErrorDeriv: TNeuralFloat;
  localNeuron: TNNetNeuron;
begin
  MaxNeurons := FNeurons.Count - 1;
  for NeuronCnt := 0 to MaxNeurons do
  begin
      localNeuron := FArrNeurons[NeuronCnt];

      if FOutputRaw.FData[NeuronCnt] >= 0 then
      begin
        localLearErrorDeriv := -FLearningRate * FOutputError.FData[NeuronCnt];
      end
      else
      begin
        localLearErrorDeriv := 0;
      end;

      if (FBatchUpdate) then
      begin
        if localLearErrorDeriv <> 0.0 then
        begin
          localNeuron.Delta.MulAdd(localLearErrorDeriv, FPrevLayer.Output);
          localNeuron.FBiasDelta := localNeuron.FBiasDelta + localLearErrorDeriv;
        end;
      end
      else
      begin
        localNeuron.FBackInertia.MulMulAdd(FInertia, localLearErrorDeriv * (1-FInertia), FPrevLayer.Output);

        localNeuron.FBiasInertia :=
          (1-FInertia)*localLearErrorDeriv +
          (  FInertia)*localNeuron.FBiasInertia;

        localNeuron.AddInertia();
      end;
  end;

  if not FBatchUpdate then AfterWeightUpdate();
end;

constructor TNNetFullConnectReLU.Create(pSizeX, pSizeY, pDepth: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pSizeX, pSizeY, pDepth, pSuppressBias);
  FActivationFn := @RectifiedLinearUnit;
  FActivationFnDerivative := @RectifiedLinearUnitDerivative;
end;

{ TNNetFullConnectReLU }
constructor TNNetFullConnectReLU.Create(pSize: integer; pSuppressBias: integer = 0);
begin
  Self.Create(pSize, 1, 1, pSuppressBias);
end;

{ TNNetSoftMax }
procedure TNNetSoftMax.Compute();
begin
  inherited Compute();
  FSoftTotalSum := FOutput.SoftMax();
end;

{ TNNetConvolutionReLU }
constructor TNNetConvolutionReLU.Create(pNumFeatures, pFeatureSize,
  pInputPadding, pStride: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pSuppressBias);
  FActivationFn := @RectifiedLinearUnit;
  FActivationFnDerivative := @RectifiedLinearUnitDerivative;
end;

{ TNNetPoolBase }
procedure TNNetPoolBase.SetPrevLayer(pPrevLayer: TNNetLayer);
var
  MaxInputDivPool, MaxInputDivPoolCnt: integer;
begin
  inherited SetPrevLayer(pPrevLayer);
  FOutputSizeX := CalcOutputSize(pPrevLayer.Output.SizeX);
  FOutputSizeY := CalcOutputSize(pPrevLayer.Output.SizeY);
  FOutputSizeD := pPrevLayer.Output.Depth;

  FOutput.ReSize(FOutputSizeX, FOutputSizeY, FOutputSizeD);
  FOutputError.ReSize(FOutputSizeX, FOutputSizeY, FOutputSizeD);
  FOutputErrorDeriv.ReSize(FOutputSizeX, FOutputSizeY, FOutputSizeD);
  SetLength(FMaxPosX, FOutput.Size);
  SetLength(FMaxPosY, FOutput.Size);
  if ((FStride = FPoolSize) and (FPadding = 0)) then
  begin
    MaxInputDivPool := Max(pPrevLayer.Output.SizeX, pPrevLayer.Output.SizeY) + 1;
    SetLength(FInputDivPool, MaxInputDivPool);
    for MaxInputDivPoolCnt := 0 to MaxInputDivPool-1
      do FInputDivPool[MaxInputDivPoolCnt] := MaxInputDivPoolCnt div FPoolSize;
  end;
end;

function TNNetPoolBase.CalcOutputSize(pInputSize: integer): integer;
begin
  Result := (pInputSize + 2*FPadding) div FStride;
  if ((pInputSize + 2*FPadding) mod FStride > 0) then Inc(Result);
end;

constructor TNNetPoolBase.Create(pPoolSize: integer; pStride:integer = 0; pPadding: integer = 0);
begin
  inherited Create;
  FPoolSize := pPoolSize;
  SetLength(FMaxPosX, 0);
  SetLength(FMaxPosY, 0);
  if pStride = 0 then pStride := pPoolSize;
  FStride := pStride;
  FPadding := pPadding;
  FStruct[0] := pPoolSize;
  FStruct[1] := pStride;
  FStruct[2] := pPadding;

  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;

  if FPadding > 0
    then FInputCopy := TNNetVolume.Create;
end;

destructor TNNetPoolBase.Destroy();
begin
  SetLength(FMaxPosX, 0);
  SetLength(FMaxPosY, 0);
  SetLength(FInputDivPool, 0);
  if FPadding > 0
    then FInputCopy.Free;

  inherited Destroy;
end;

procedure TNNetMaxPool.Compute();
var
  StartTime: double;
begin
  StartTime := Now();
  Output.Fill(-1000000);

  if FPadding > 0
    then FInputCopy.CopyPadding(FPrevLayer.Output, FPadding)
    else FInputCopy := FPrevLayer.Output;

  if ((FStride = FPoolSize) and (FPadding = 0)) then
  begin
    ComputeDefaultStride();
  end
  else
  begin
    ComputeWithStride();
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNetMaxPool.ComputeDefaultStride();
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  OutX, OutY: integer;
  OutputRawPos: integer;
  InputRawPtr: TNeuralFloatPtr;
begin
  MaxX := FInputCopy.SizeX - 1;
  MaxY := FInputCopy.SizeY - 1;
  MaxD := FInputCopy.Depth - 1;

  for CntY := 0 to MaxY do
  begin
    OutY := FInputDivPool[CntY]; //CntY div FPoolSize;
    for CntX := 0 to MaxX do
    begin
      OutX := FInputDivPool[CntX]; //CntX div FPoolSize;
      OutputRawPos := FOutput.GetRawPos(OutX, OutY);
      InputRawPtr := FInputCopy.GetRawPtr(CntX, CntY);
      for CntD := 0 to MaxD do
      begin
        if InputRawPtr^ > FOutput.FData[OutputRawPos] then
        begin
          FOutput.FData[OutputRawPos] := InputRawPtr^;
          FMaxPosX[OutputRawPos] := CntX;
          FMaxPosY[OutputRawPos] := CntY;
        end;
        Inc(OutputRawPos);
        Inc(InputRawPtr);
      end;
    end;
  end; // of for CntD
end;

procedure TNNetMaxPool.ComputeWithStride();
var
  CntOutputX, CntOutputY, CntD: integer;
  OutputMaxX, OutputMaxY, MaxD: integer;
  InX, InY, InXMax, InYMax: integer;
  CntInputPX, CntInputPY: integer;
  OutputRawPos: integer;
  CurrValue: TNeuralFloat;
  LocalPoolSizeM1, InputSizeXM1, InputSizeYM1: integer;
begin
  OutputMaxX := Output.SizeX - 1;
  OutputMaxY := Output.SizeY - 1;
  MaxD := Output.Depth - 1;
  LocalPoolSizeM1 := FPoolSize - 1;
  InputSizeXM1 := FInputCopy.SizeX - 1;
  InputSizeYM1 := FInputCopy.SizeY - 1;

  for CntOutputY := 0 to OutputMaxY do
  begin
    InY := CntOutputY * FStride;
    InYMax := Min(InY + LocalPoolSizeM1, InputSizeYM1);
    for CntOutputX := 0 to OutputMaxX do
    begin
      InX := CntOutputX * FStride;
      InXMax := Min(InX + LocalPoolSizeM1, InputSizeXM1);
      OutputRawPos := Output.GetRawPos(CntOutputX, CntOutputY);
      for CntD := 0 to MaxD do
      begin
        for CntInputPX := InX to InXMax do
        begin
          for CntInputPY := InY to InYMax do
          begin
            CurrValue := FInputCopy[CntInputPX, CntInputPY, CntD];
            if CurrValue > FOutput.FData[OutputRawPos] then
            begin
              //WriteLn(CntInputPX, ' ', CntInputPY,' - ',CntOutputX, ' ', CntOutputY,' ', CurrValue);
              FOutput.FData[OutputRawPos] := CurrValue;
              FMaxPosX[OutputRawPos] := CntInputPX;
              FMaxPosY[OutputRawPos] := CntInputPY;
            end;
          end;
        end;
        Inc(OutputRawPos);
      end;
    end;
  end; // of for CntD
end;

procedure TNNetPoolBase.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (Assigned(FPrevLayer) and (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size)) then
  begin
    StartTime := Now();
    //TODO: experiment the following line.
    //FOutputError.Mul(FStride*FStride);

    if ((FStride = FPoolSize) and (FPadding = 0)) then
    begin
      BackpropagateDefaultStride();
    end
    else
    begin
      BackpropagateWithStride();
    end;
    FBackwardTime := FBackwardTime + (Now() - StartTime);
    FPrevLayer.Backpropagate();
  end;
end;

procedure TNNetPoolBase.BackpropagateDefaultStride();
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  OutputRawPos, PrevRawPos: integer;
begin
  MaxX := Output.SizeX - 1;
  MaxY := Output.SizeY - 1;
  MaxD := Output.Depth - 1;
  //Although the below line makes all the sense, it might brake compatibility
  //with existing code.
  //if FStride > 1 then FOutputError.Mul( Min(FStride, 4) );

  for CntY := 0 to MaxY do
  begin
    for CntX := 0 to MaxX do
    begin
      OutputRawPos := FOutput.GetRawPos(CntX, CntY);
      for CntD := 0 to MaxD do
      begin
        PrevRawPos := FPrevLayer.OutputError.GetRawPos(FMaxPosX[OutputRawPos], FMaxPosY[OutputRawPos], CntD);
        {$IFDEF FPC}
        FPrevLayer.OutputError.FData[PrevRawPos] += FOutPutError.FData[OutputRawPos];
        {$ELSE}
        FPrevLayer.OutputError.FData[PrevRawPos] :=
          FPrevLayer.OutputError.FData[PrevRawPos] + FOutPutError.FData[OutputRawPos];
        {$ENDIF}
        Inc(OutputRawPos);
      end;
    end;
  end;
end;

procedure TNNetPoolBase.BackpropagateWithStride();
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  OutputRawPos, PrevRawPos: integer;
  MaxPosX, MaxPosY: integer;
begin
  MaxX := Output.SizeX - 1;
  MaxY := Output.SizeY - 1;
  MaxD := Output.Depth - 1;
  //Although the below line makes all the sense, it might brake compatibility
  //with existing code.
  //if FStride > 1 then FOutputError.Mul( Min(FStride, 4) );

  for CntX := 0 to MaxX do
  begin
    for CntY := 0 to MaxY do
    begin
      OutputRawPos := FOutput.GetRawPos(CntX, CntY);
      for CntD := 0 to MaxD do
      begin
        MaxPosX := FMaxPosX[OutputRawPos] - FPadding;
        MaxPosY := FMaxPosY[OutputRawPos] - FPadding;
        if
        (
          (MaxPosX >= 0) and
          (MaxPosX < FPrevLayer.Output.SizeX) and
          (MaxPosY >= 0) and
          (MaxPosY < FPrevLayer.Output.SizeY)
        ) then
        begin
          PrevRawPos := FPrevLayer.OutputError.GetRawPos(MaxPosX, MaxPosY, CntD);
          {$IFDEF FPC}
          FPrevLayer.OutputError.FData[PrevRawPos] += FOutPutError.FData[OutputRawPos];
          {$ELSE}
          FPrevLayer.OutputError.FData[PrevRawPos] :=
            FPrevLayer.OutputError.FData[PrevRawPos] + FOutPutError.FData[OutputRawPos];
          {$ENDIF}
        end;
        Inc(OutputRawPos);
      end;
    end;
  end;
end;

procedure TNNetPoolBase.ComputePreviousLayerError();
begin
  // Backpropagate already does the job for MaxPool.
end;

{ TNNetConvolution }
procedure TNNetConvolutionBase.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FOutput.ReSize(FOutputSizeX,FOutputSizeY,FNeurons.Count);
  FOutputRaw.ReSize(FOutputSizeX,FOutputSizeY,FNeurons.Count);
  FOutputError.ReSize(FOutputSizeX,FOutputSizeY,FNeurons.Count);
  FOutputErrorDeriv.ReSize(FOutputSizeX,FOutputSizeY,FNeurons.Count);
  FVectorSize := FFeatureSizeX*FFeatureSizeY*pPrevLayer.Output.Depth;
  FVectorSizeBytes := FVectorSize * SizeOf(TNeuralFloat);
  if FPointwise then
  begin
    FInputPrepared := pPrevLayer.Output;
  end
  else
  begin
    FInputPrepared.Resize(FOutputSizeX, FOutputSizeY, FVectorSize);
  end;
  RefreshNeuronWeightList();
  if ShouldUseInterleavedDotProduct then
  begin
    FShouldConcatWeights := true;
    FShouldInterleaveWeights := true;
  end;
  BuildArrNeurons();

  {$IFDEF OpenCL}
  FShouldOpenCL := ( (FVectorSize <= csMaxInterleavedSize) or (FOutput.Size*FVectorSize >= 24*24*3 * 3*3*3) );
  if (FHasOpenCL and FShouldOpenCL) then
  begin
    EnableOpenCL(FDotProductKernel);
  end;
  {$ENDIF}
  FShouldConcatWeights := true;
  InitDefault();

  FTileSizeX := GetMaxDivisor(FOutputSizeX, 16);
  FTileSizeD := GetMaxDivisor(FNeurons.Count, 16);

  if FTileSizeX = 1 then FTileSizeX := GetMaxDivisor(FOutputSizeX, 128);
  if FTileSizeD = 1 then FTileSizeD := GetMaxDivisor(FNeurons.Count, 128);

  FMaxTileX := (FOutputSizeX div FTileSizeX) - 1;
  FMaxTileD := (FNeurons.Count div FTileSizeD) - 1;

  // Debug Tiles
  //WriteLn(FOutputSizeX,' ',FNeurons.Count,
  //  '-->',FTileSizeX,' ',FTileSizeD,
  //  '-->',FMaxTileX,' ',FMaxTileD
  //  );
end;

procedure TNNetConvolutionAbstract.RefreshCalculatePrevLayerError();
begin
  FCalculatePrevLayerError :=
    Assigned(FPrevLayer) and
    (FPrevLayer.OutputError.Size > 0) and
    (FPrevLayer.OutputError.Size = FPrevLayer.Output.Size);
end;

procedure TNNetConvolutionAbstract.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  inherited SetPrevLayer(pPrevLayer);
  FFeatureSizeX := Min(FFeatureSizeX, pPrevLayer.Output.SizeX);
  FFeatureSizeY := Min(FFeatureSizeY, pPrevLayer.Output.SizeY);
  SetNumWeightsForAllNeurons(FFeatureSizeX, FFeatureSizeY, pPrevLayer.Output.Depth);
  FFeatureSizeYMinus1 := FFeatureSizeY - 1;
  FFeatureSizeXMinus1 := FFeatureSizeX - 1;
  RefreshCalculatePrevLayerError();
  FOutputSizeX := CalcOutputSize(pPrevLayer.Output.SizeX, FFeatureSizeX, FPadding, FStride);
  FOutputSizeY := CalcOutputSize(pPrevLayer.Output.SizeY, FFeatureSizeY, FPadding, FStride);
end;

function TNNetConvolutionAbstract.CalcOutputSize(pInputSize, pFeatureSize, pInputPadding,
  pStride: integer): integer;
begin
  Result := ( ( pInputSize - pFeatureSize + 2*pInputPadding ) div pStride) + 1;
end;

procedure TNNetConvolution.BackpropagateAtOutputPos(pCanBackpropOnPos: boolean; OutputRawPos, OutputX, OutputY, OutputD, PrevX, PrevY: integer);
var
  LocalCntY: integer;
  LocalLearningErrorDeriv: TNeuralFloat;
  LocalOutputErrorDeriv: TNeuralFloat;
  SmoothLocalOutputErrorDeriv: TNeuralFloat;
  LocalWeight, LocalPrevError: TNNetVolume;
  {SrcPtr,} LocalDestPtr: TNeuralFloatArrPtr;
begin
  {$IFDEF FPC}
  if FActivationFn = @RectifiedLinearUnit then
  begin
    if FOutput.FData[OutputRawPos] > 0 then
    begin
      LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
    end
    else
    begin
      LocalOutputErrorDeriv := 0;
    end;
  end
  else if FActivationFn = @Identity then
  begin
    LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
  end
  else
  begin
    LocalOutputErrorDeriv :=
      FOutputError.FData[OutputRawPos] *
      FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
  end;
  {$ELSE}
    LocalOutputErrorDeriv :=
      FOutputError.FData[OutputRawPos] *
      FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
  {$ENDIF}

  FOutputErrorDeriv.FData[OutputRawPos] := LocalOutputErrorDeriv;

  if (LocalOutputErrorDeriv <> 0) then
  begin
    LocalLearningErrorDeriv := (-FLearningRate) * LocalOutputErrorDeriv;

    if (LocalLearningErrorDeriv <> 0.0) then
    begin
      FNeurons[OutputD].Delta.MulAdd(LocalLearningErrorDeriv, FInputPrepared.GetRawPtr(OutputX, OutputY));

      {$IFDEF FPC}
      FNeurons[OutputD].FBiasDelta += LocalLearningErrorDeriv;
      {$ELSE}
      FNeurons[OutputD].FBiasDelta :=
        FNeurons[OutputD].FBiasDelta + LocalLearningErrorDeriv;
      {$ENDIF}

      if (FCalculatePrevLayerError) then
      begin
        LocalWeight := FArrNeurons[OutputD].Weights;
        LocalPrevError := FPrevLayer.OutputError;
        if FPointwise then
        begin
          LocalDestPtr := LocalPrevError.GetRawPtr(OutputX, OutputY);
          LocalPrevError.MulAdd(LocalDestPtr, LocalWeight.DataPtr, LocalOutputErrorDeriv, FInputCopy.Depth);
        end
        else
        begin
          if pCanBackpropOnPos then
          begin
            SmoothLocalOutputErrorDeriv := LocalOutputErrorDeriv / FLearnSmoothener;
            for LocalCntY := 0 to FFeatureSizeYMinus1 do
            begin
              (*
              LocalDestPtr  := LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY);
              SrcPtr   := LocalWeight.GetRawPtr(0, LocalCntY);
              LocalPrevError.MulAdd(LocalDestPtr, SrcPtr, SmoothLocalOutputErrorDeriv, FSizeXDepth);
              *)
              LocalPrevError.MulAdd
              (
                LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY),
                LocalWeight.GetRawPtr(0, LocalCntY),
                SmoothLocalOutputErrorDeriv,
                FSizeXDepth
              );
            end;
          end;
        end;
      end; // if (FCalculatePrevLayerError)
    end; // (LocalLearningErrorDeriv <> 0.0)

  end; // (LocalOutputErrorDeriv <> 0)
end;

function TNNetConvolutionBase.ShouldUseInterleavedDotProduct: boolean;
begin
  Result :=
    (
      FOutput.HasAVX and
      (FNeurons.Count mod 32 = 0) and
      ( (FVectorSize <= csMaxInterleavedSize) or (FOutputSizeX>=40) ) and
      ( not(FPointwise) )
    );
end;

procedure TNNetConvolution.ComputeInterleaved();
begin
  if FConcatedWInter.Size < FNeurons[0].Weights.Size * FNeurons.Count then
  begin
    FErrorProc('Error at TNNetConvolution.ComputeInterleaved: ' + IntToStr(FConcatedWInter.Size));
    AfterWeightUpdate();
  end;

  FOutputRaw.InterleavedDotProduct(FConcatedWInter, FInputPrepared, FVectorSize);
  if FSuppressBias = 0 then FOutputRaw.Add(FBiasOutput);
  ApplyActivationFunctionToOutput();
end;

{$IFDEF OpenCL}
procedure TNNetConvolution.ComputeOpenCL();
var
  InputAVolume: TNNetVolume;
begin
  if FShouldInterleaveWeights then
  begin
    if FConcatedWInter.Size < FNeurons[0].Weights.Size * FNeurons.Count then
    begin
      FErrorProc('Error at TNNetConvolution.ComputeOpenCL: ' + IntToStr(FConcatedWInter.Size));
      AfterWeightUpdate();
    end;
    InputAVolume := FConcatedWInter;
  end
  else
  begin
    InputAVolume := FConcatedWeights;
  end;
  FDotCL.Compute(InputAVolume, FInputPrepared, 0, FAfterWeightUpdateHasBeenCalled, true);
  FAfterWeightUpdateHasBeenCalled := false;
  {$IFDEF Linux}
  FDotCL.FinishAndLoadResult(FOutputRaw, 0.75);
  {$ELSE}
  FDotCL.FinishAndLoadResult(FOutputRaw, 0.0);
  {$ENDIF}

  if FSuppressBias = 0 then FOutputRaw.Add(FBiasOutput);
  ApplyActivationFunctionToOutput();
end;
{$ENDIF}

procedure TNNetConvolution.ComputeCPU();
begin
  FOutputRaw.DotProducts(FNeurons.Count, FOutputSizeX * FOutputSizeY, FVectorSize, FConcatedWeights, FInputPrepared);
  if FSuppressBias = 0 then FOutputRaw.Add(FBiasOutput);
  ApplyActivationFunctionToOutput();
end;

procedure TNNetConvolution.ComputeTiledCPU();
begin
  FOutputRaw.DotProductsTiled(FNeurons.Count, FOutputSizeX * FOutputSizeY, FVectorSize, FConcatedWeights, FInputPrepared, FTileSizeD, FTileSizeX);
  if FSuppressBias = 0 then FOutputRaw.Add(FBiasOutput);
  ApplyActivationFunctionToOutput();
end;

procedure TNNetConvolutionBase.PrepareInputForConvolutionFast();
var
  OutputCntX, OutputCntY: integer;
  MaxX, MaxY: integer;
  DepthFSize, SizeOfDepthFSize: integer;
  yCount: integer;
  InputX: integer;
  RowSize: integer;
  {$IFDEF AVXANY}
  SourceRawPos, DestRawPos: pointer;
  {$ENDIF}
begin
  if (FPointwise) then
  begin
    // There is nothing to do. YAY!
  end
  else
  begin
    DepthFSize := FInputCopy.Depth * FFeatureSizeX;
    RowSize := DepthFSize;
    SizeOfDepthFSize := DepthFSize * SizeOf(TNeuralFloat);
    MaxX := FOutput.SizeX - 1;
    MaxY := FOutput.SizeY - 1;

    FInputPrepared.ReSize(FOutput.SizeX, FOutput.SizeY, FInputCopy.Depth * FFeatureSizeX * FFeatureSizeY);

    for OutputCntX := 0 to MaxX do
    begin
      InputX := OutputCntX * FStride;
      for OutputCntY := 0 to MaxY do
      begin
        for yCount := 0 to FFeatureSizeY - 1 do
        begin
          {$IFDEF AVXANY}
          SourceRawPos := FInputCopy.GetRawPtr(InputX, OutputCntY*FStride + yCount , 0);
          DestRawPos := FInputPrepared.GetRawPtr(OutputCntX, OutputCntY, DepthFSize * yCount);
          asm_dword_copy;
          {$ELSE}
          Move(
            FInputCopy.FData[FInputCopy.GetRawPos(InputX, OutputCntY*FStride + yCount , 0)],
            FInputPrepared.FData[FInputPrepared.GetRawPos(OutputCntX, OutputCntY, DepthFSize * yCount)],
            SizeOfDepthFSize
          );
          {$ENDIF}
        end;
      end;
    end;
  end;
end;

constructor TNNetConvolutionBase.Create(pNumFeatures, pFeatureSize, pInputPadding, pStride: integer; pSuppressBias: integer = 0);
begin
  inherited Create(pFeatureSize, pInputPadding, pStride, pSuppressBias);
  FPointwise := (
    (pFeatureSize = 1) and
    (FPadding = 0) and
    (FStride = 1) );
  AddNeurons(pNumFeatures);
  FOutputError.ReSize(pNumFeatures, 1, 1);
  FOutputErrorDeriv.ReSize(pNumFeatures, 1, 1);
  //FDotProductResult := TNNetVolume.Create;

  if not FPointwise
    then FInputPrepared := TNNetVolume.Create;

  FStruct[0] := pNumFeatures;
  FStruct[1] := pFeatureSize;
  FStruct[2] := pInputPadding;
  FStruct[3] := pStride;
  FStruct[4] := FSuppressBias;
  FActivationFn := @HiperbolicTangent;
  FActivationFnDerivative := @HiperbolicTangentDerivative;
end;

destructor TNNetConvolutionBase.Destroy();
begin
  if not FPointwise
    then FInputPrepared.Free;

  //FDotProductResult.Free;
  inherited Destroy;
end;

procedure TNNetConvolution.Compute();
  procedure ComputeOnCPU;
  begin
    // interleaved dot product is faster with small vectors or big convs.
    if ShouldUseInterleavedDotProduct then
    begin
      ComputeInterleaved();
    end
    else
    begin
      ComputeTiledCPU();
      //ComputeCPU();
    end;
  end;
var
    StartTime: double;
begin
  if FNeurons.Count > 0 then
  begin
    StartTime := Now();
    RefreshCalculatePrevLayerError();
    if FPadding > 0
      then FInputCopy.CopyPadding(FPrevLayer.Output, FPadding)
      else FInputCopy := FPrevLayer.Output;

    if FSmoothErrorPropagation then
    begin
      FLearnSmoothener := FFeatureSizeX * FFeatureSizeY;
    end
    else
    begin
      FLearnSmoothener := 1;
    end;

    FSizeXDepth := FFeatureSizeX * FInputCopy.Depth;
    FSizeXDepthBytes := FSizeXDepth * SizeOf(TNeuralFloat);
    FPrevSizeXDepthBytes := FPrevLayer.Output.IncYSizeBytes();

    //FInputPrepared.ReSize(FOutput.SizeX, FOutput.SizeY, FInputCopy.Depth * FFeatureSizeX * FFeatureSizeY);
    PrepareInputForConvolutionFast();

    {$IFDEF OpenCL}
    if (Assigned(FDotCL) and FHasOpenCL and FShouldOpenCL) then
    begin
      ComputeOpenCL();
    end
    else
    begin
      ComputeOnCPU;
    end;
    {$ELSE}
      ComputeOnCPU;
    {$ENDIF}
    FForwardTime := FForwardTime + (Now() - StartTime);
  end
  else
  begin
    FErrorProc('Neuronal layer contains no neuron:'+ IntToStr(FNeurons.Count));
  end;
end;

procedure TNNetConvolution.Backpropagate();
var
  StartTime, LocalNow: double;
begin
  StartTime := Now();
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FNeurons.Count = FOutput.Depth) and (FPrevLayer.Output.Size > 0) then
  begin
    // ComputeErrorDeriv() isn't required as it's done on BackpropagateAtOutputPos
    // ClearDeltas() is not required as it's done in BackpropagateNTL

    //BackpropagateFastCPUDev();
    //BackpropagateFastCPU();
    BackpropagateFastTiledCPU();
    //BackpropagateCPU();

    {$IFDEF CheckRange}ForceRangeWeights(1000);{$ENDIF}
  end
  else
  begin
    FErrorProc
    (
      'TNNetConvolution.Backpropagate should have same sizes.' +
      'Neurons:' + IntToStr(FNeurons.Count) +
      ' Output Depth:' + IntToStr(FOutput.Depth) +
      ' PrevLayer:' + IntToStr(FPrevLayer.Output.Size)
    );
  end;
  LocalNow := Now();
  {$IFDEF Debug}
  if LocalNow > StartTime + 1000
    then FErrorProc('TNNetConvolution.Backpropagate bad StartTime.');
  {$ENDIF}
  FBackwardTime := FBackwardTime + (LocalNow - StartTime);
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

procedure TNNetConvolution.BackpropagateCPU();
var
  CntX, CntY, NeuronIdx: integer;
  MaxX, MaxY, MaxD: integer;
  PrevX, PrevY: integer;
  OutputRawPos: integer;
  CanBackpropOnPos: boolean;
begin
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxD := OutputError.Depth - 1;

  for CntY := 0 to MaxY do
  begin
    PrevY := (CntY*FStride)-FPadding;
    for CntX := 0 to MaxX do
    begin
      PrevX := (CntX*FStride)-FPadding;
      OutputRawPos := FOutputErrorDeriv.GetRawPos(CntX, CntY);
      CanBackpropOnPos :=
        (PrevX >= 0) and (PrevY >= 0) and
        (PrevX < 1 + FPrevLayer.FOutputError.SizeX - FFeatureSizeX) and
        (PrevY < 1 + FPrevLayer.FOutputError.SizeY - FFeatureSizeY);
      for NeuronIdx := 0 to MaxD do
      begin
        BackpropagateAtOutputPos(CanBackpropOnPos, OutputRawPos, CntX, CntY, NeuronIdx, PrevX, PrevY);
        Inc(OutputRawPos);
      end;
    end;
  end;

  if (not FBatchUpdate) then
  begin
    for NeuronIdx := 0 to MaxD do FNeurons[NeuronIdx].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
end;

// code was made monolitic/spaghetti as inline isn't working with ASM
procedure TNNetConvolution.BackpropagateFastCPU();
var
  OutputX, OutputY, OutputD: integer;
  MaxX, MaxY, MaxD: integer;
  PrevX, PrevY: integer;
  OutputRawPos: integer;
  CanBackpropOnPos: boolean;
  LocalCntY: integer;
  LocalLearningErrorDeriv: TNeuralFloat;
  LocalOutputErrorDeriv: TNeuralFloat;
  SmoothLocalOutputErrorDeriv: TNeuralFloat;
  LocalWeight, LocalPrevError: TNNetVolume;
  {SrcPtr,} LocalDestPtr: TNeuralFloatArrPtr;
  SmoothLocalOutputErrorDerivPtr: pointer;
  PrevNumElements, PrevMissedElements: integer;
  PtrNeuronDelta, PtrPreparedInput: TNeuralFloatArrPtr;
  PrevPtrA, PrevPtrB: TNeuralFloatArrPtr;
  NeuronWeights: integer;
  LocalLearningErrorDerivPtr: pointer;
  localNumElements, MissedElements: integer;
  MaxPrevX, MaxPrevY: integer;
begin
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxD := OutputError.Depth - 1;
  MaxPrevX := 1 + FPrevLayer.FOutputError.SizeX - FFeatureSizeX;
  MaxPrevY := 1 + FPrevLayer.FOutputError.SizeY - FFeatureSizeY;
  LocalPrevError := FPrevLayer.OutputError;
  PrevNumElements := (FSizeXDepth div 4) * 4;
  PrevMissedElements := FSizeXDepth - PrevNumElements;
  NeuronWeights := FArrNeurons[0].Delta.Size;
  localNumElements := (NeuronWeights div 4) * 4;
  MissedElements := NeuronWeights - localNumElements;
  SmoothLocalOutputErrorDerivPtr := Addr(SmoothLocalOutputErrorDeriv);
  LocalLearningErrorDerivPtr := Addr(LocalLearningErrorDeriv);
    begin
      for OutputY := 0 to MaxY do
      begin
        PrevY := (OutputY*FStride)-FPadding;
        for OutputX := 0 to MaxX do
        begin
          PrevX := (OutputX*FStride)-FPadding;
          OutputRawPos := FOutputErrorDeriv.GetRawPos(OutputX, OutputY);
          //TODO: the next line is probably wrong.
          if (FCalculatePrevLayerError) then LocalDestPtr  := LocalPrevError.GetRawPtr(OutputX, OutputY);
          PtrPreparedInput := FInputPrepared.GetRawPtr(OutputX, OutputY);
          CanBackpropOnPos :=
            (PrevX >= 0) and (PrevY >= 0) and
            (PrevX < MaxPrevX) and
            (PrevY < MaxPrevY);
          for OutputD := 0 to MaxD do
          begin
            {$IFDEF FPC}
            if FActivationFn = @RectifiedLinearUnit then
            begin
              if FOutputRaw.FData[OutputRawPos] >= 0 then
              begin
                LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
              end
              else
              begin
                LocalOutputErrorDeriv := 0;
              end;
            end
            else if FActivationFn = @Identity then
            begin
              LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
            end
            else
            begin
              LocalOutputErrorDeriv :=
                FOutputError.FData[OutputRawPos] *
                FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
            end;
            {$ELSE}
              LocalOutputErrorDeriv :=
                FOutputError.FData[OutputRawPos] *
                FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
            {$ENDIF}

            FOutputErrorDeriv.FData[OutputRawPos] := LocalOutputErrorDeriv;
            LocalLearningErrorDeriv := (-FLearningRate) * LocalOutputErrorDeriv;
            if (LocalLearningErrorDeriv <> 0.0) then
            begin
                {$IFNDEF AVX64}
                FArrNeurons[OutputD].Delta.MulAdd(LocalLearningErrorDeriv, PtrPreparedInput);
                {$ELSE}
                {$IFDEF Debug}
                if localNumElements + MissedElements <> FArrNeurons[OutputD].Delta.Size
                then FErrorProc('Error at TNNetConvolution.BackpropagateFastCPU(): neuron size doesn''t match.');
                {$ENDIF}
                PtrNeuronDelta := FArrNeurons[OutputD].Delta.DataPtr;
                asm_avx64_train_neuron
                {$ENDIF}

                {$IFDEF FPC}
                FArrNeurons[OutputD].FBiasDelta += LocalLearningErrorDeriv;
                {$ELSE}
                FArrNeurons[OutputD].FBiasDelta :=
                  FArrNeurons[OutputD].FBiasDelta + LocalLearningErrorDeriv;
                {$ENDIF}

                if (FCalculatePrevLayerError) then
                begin
                  LocalWeight := FArrNeurons[OutputD].Weights;
                  if FPointwise then
                  begin
                    {$IFNDEF AVX64}
                    LocalPrevError.MulAdd(LocalDestPtr, LocalWeight.DataPtr, LocalOutputErrorDeriv, FInputCopy.Depth);
                    {$ELSE}
                    {$IFDEF Debug}
                    if PrevNumElements + PrevMissedElements <> FInputCopy.Depth
                    then FErrorProc('Error at TNNetConvolution.BackpropagateFastCPU(): pointwise vector size doesn''t match.');
                    {$ENDIF}
                    PrevPtrA := LocalDestPtr;
                    PrevPtrB := LocalWeight.DataPtr;
                    SmoothLocalOutputErrorDeriv := LocalOutputErrorDeriv;
                    asm_avx64_prev_backprop;
                    {$ENDIF}
                  end
                  else
                  begin
                    if CanBackpropOnPos then
                    begin
                      SmoothLocalOutputErrorDeriv := LocalOutputErrorDeriv / FLearnSmoothener;
                      PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY);
                      PrevPtrB := LocalWeight.DataPtr;
                      for LocalCntY := 0 to FFeatureSizeYMinus1 do
                      begin
                        {$IFNDEF AVX64}
                        LocalPrevError.MulAdd
                        (
                          PrevPtrA, //LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY),
                          PrevPtrB, //LocalWeight.GetRawPtr(0, LocalCntY),
                          SmoothLocalOutputErrorDeriv,
                          FSizeXDepth
                        );
                        {$ELSE}
                        {$IFDEF Debug}
                        if PrevNumElements + PrevMissedElements <> FSizeXDepth
                        then FErrorProc('Error at TNNetConvolution.BackpropagateFastCPU(): vector size doesn''t match.');
                        {$ENDIF}
                        //PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY);
                        //PrevPtrB := LocalWeight.GetRawPtr(0, LocalCntY);
                        asm_avx64_prev_backprop;
                        {$ENDIF}
                        if LocalCntY < FFeatureSizeYMinus1 then
                        begin
                          {$IFDEF FPC}
                          PrevPtrA := (pointer(PrevPtrA) + FPrevSizeXDepthBytes);
                          PrevPtrB := (pointer(PrevPtrB) + FSizeXDepthBytes);
                          {$ELSE}
                          PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY + 1);
                          PrevPtrB := LocalWeight.GetRawPtr(0, LocalCntY + 1);
                          {$ENDIF}
                        end;
                      end;
                    end;
                  end;
                end; // if (FCalculatePrevLayerError)
            end; // (LocalLearningErrorDeriv <> 0.0)
            Inc(OutputRawPos);
          end;
        end;
      end;
    end;

  if (not FBatchUpdate) then
  begin
    for OutputD := 0 to MaxD do FArrNeurons[OutputD].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
end;

procedure TNNetConvolution.BackpropagateFastTiledCPU();
var
  OutputX, OutputY, OutputD: integer;
  MaxX, MaxY, MaxD: integer;
  PrevX, PrevY: integer;
  OutputRawPos: integer;
  CanBackpropOnPos: boolean;
  LocalCntY: integer;
  LocalLearningErrorDeriv: TNeuralFloat;
  LocalOutputErrorDeriv: TNeuralFloat;
  SmoothLocalOutputErrorDeriv: TNeuralFloat;
  LocalWeight, LocalPrevError: TNNetVolume;
  {SrcPtr,} LocalDestPtr: TNeuralFloatArrPtr;
  SmoothLocalOutputErrorDerivPtr: pointer;
  PrevNumElements, PrevMissedElements: integer;
  PtrNeuronDelta, PtrPreparedInput: TNeuralFloatArrPtr;
  PrevPtrA, PrevPtrB: TNeuralFloatArrPtr;
  NeuronWeights: integer;
  LocalLearningErrorDerivPtr: pointer;
  localNumElements, MissedElements: integer;
  MaxPrevX, MaxPrevY: integer;
  // Tiling
  TileXCnt, TileDCnt: integer;
  StartTileX, EndTileX, StartTileD, EndTileD: integer;
begin
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxD := OutputError.Depth - 1;
  MaxPrevX := 1 + FPrevLayer.FOutputError.SizeX - FFeatureSizeX;
  MaxPrevY := 1 + FPrevLayer.FOutputError.SizeY - FFeatureSizeY;
  LocalPrevError := FPrevLayer.OutputError;
  PrevNumElements := (FSizeXDepth div 4) * 4;
  PrevMissedElements := FSizeXDepth - PrevNumElements;
  NeuronWeights := FArrNeurons[0].Delta.Size;
  localNumElements := (NeuronWeights div 4) * 4;
  MissedElements := NeuronWeights - localNumElements;
  SmoothLocalOutputErrorDerivPtr := Addr(SmoothLocalOutputErrorDeriv);
  LocalLearningErrorDerivPtr := Addr(LocalLearningErrorDeriv);
  for OutputY := 0 to MaxY do
  begin
    PrevY := (OutputY*FStride)-FPadding;
    for TileXCnt := 0 to FMaxTileX do
    begin
      StartTileX := TileXCnt * FTileSizeX;
      EndTileX := StartTileX + FTileSizeX - 1;
      for TileDCnt := 0 to FMaxTileD do
      begin
        StartTileD := TileDCnt * FTileSizeD;
        EndTileD := StartTileD + FTileSizeD - 1;
        //WriteLn(StartTileX,' ',EndTileX,' - ',StartTileY,' ',EndTileY,' - ',StartTileD,' ',EndTileD);
        begin
          for OutputX := StartTileX to EndTileX do
          begin
            PrevX := (OutputX*FStride)-FPadding;
            PtrPreparedInput := FInputPrepared.GetRawPtr(OutputX, OutputY);
            CanBackpropOnPos :=
              (PrevX >= 0) and (PrevY >= 0) and
              (PrevX < MaxPrevX) and
              (PrevY < MaxPrevY);
            if (FCalculatePrevLayerError and CanBackpropOnPos) then LocalDestPtr  := LocalPrevError.GetRawPtr(PrevX, PrevY);
            OutputRawPos := FOutputErrorDeriv.GetRawPos(OutputX, OutputY, StartTileD);
            for OutputD := StartTileD to EndTileD do
            begin
              {$IFDEF FPC}
              if FActivationFn = @RectifiedLinearUnit then
              begin
                if FOutputRaw.FData[OutputRawPos] >= 0 then
                begin
                  LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
                end
                else
                begin
                  LocalOutputErrorDeriv := 0;
                end;
              end
              else if FActivationFn = @Identity then
              begin
                LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
              end
              else
              begin
                LocalOutputErrorDeriv :=
                  FOutputError.FData[OutputRawPos] *
                  FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
              end;
              {$ELSE}
                LocalOutputErrorDeriv :=
                  FOutputError.FData[OutputRawPos] *
                  FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
              {$ENDIF}

              FOutputErrorDeriv.FData[OutputRawPos] := LocalOutputErrorDeriv;
              LocalLearningErrorDeriv := (-FLearningRate) * LocalOutputErrorDeriv;
              if (LocalLearningErrorDeriv <> 0.0) then
              begin
                  {$IFNDEF AVX64}
                  FArrNeurons[OutputD].Delta.MulAdd(LocalLearningErrorDeriv, PtrPreparedInput);
                  {$ELSE}
                  {$IFDEF Debug}
                  if localNumElements + MissedElements <> FArrNeurons[OutputD].Delta.Size
                  then FErrorProc('Error at TNNetConvolution.BackpropagateFastCPU(): neuron size doesn''t match.');
                  {$ENDIF}
                  PtrNeuronDelta := FArrNeurons[OutputD].Delta.DataPtr;
                  asm_avx64_train_neuron
                  {$ENDIF}

                  {$IFDEF FPC}
                  FArrNeurons[OutputD].FBiasDelta += LocalLearningErrorDeriv;
                  {$ELSE}
                  FArrNeurons[OutputD].FBiasDelta :=
                    FArrNeurons[OutputD].FBiasDelta + LocalLearningErrorDeriv;
                  {$ENDIF}

                  if (FCalculatePrevLayerError) then
                  begin
                    LocalWeight := FArrNeurons[OutputD].Weights;
                    if FPointwise then
                    begin
                      {$IFNDEF AVX64}
                      LocalPrevError.MulAdd(LocalDestPtr, LocalWeight.DataPtr, LocalOutputErrorDeriv, FInputCopy.Depth);
                      {$ELSE}
                      {$IFDEF Debug}
                      if PrevNumElements + PrevMissedElements <> FInputCopy.Depth
                      then FErrorProc('Error at TNNetConvolution.BackpropagateFastCPU(): pointwise vector size doesn''t match.');
                      {$ENDIF}
                      PrevPtrA := LocalDestPtr;
                      PrevPtrB := LocalWeight.DataPtr;
                      SmoothLocalOutputErrorDeriv := LocalOutputErrorDeriv;
                      asm_avx64_prev_backprop;
                      {$ENDIF}
                    end
                    else
                    begin
                      if CanBackpropOnPos then
                      begin
                        SmoothLocalOutputErrorDeriv := LocalOutputErrorDeriv / FLearnSmoothener;
                        PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY);
                        PrevPtrB := LocalWeight.DataPtr;
                        for LocalCntY := 0 to FFeatureSizeYMinus1 do
                        begin
                          {$IFNDEF AVX64}
                          LocalPrevError.MulAdd
                          (
                            PrevPtrA, //LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY),
                            PrevPtrB, //LocalWeight.GetRawPtr(0, LocalCntY),
                            SmoothLocalOutputErrorDeriv,
                            FSizeXDepth
                          );
                          {$ELSE}
                          {$IFDEF Debug}
                          if PrevNumElements + PrevMissedElements <> FSizeXDepth
                          then FErrorProc('Error at TNNetConvolution.BackpropagateFastCPU(): vector size doesn''t match.');
                          {$ENDIF}
                          //PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY);
                          //PrevPtrB := LocalWeight.GetRawPtr(0, LocalCntY);
                          asm_avx64_prev_backprop;
                          {$ENDIF}
                          if LocalCntY < FFeatureSizeYMinus1 then
                          begin
                            {$IFDEF FPC}
                            PrevPtrA := (pointer(PrevPtrA) + FPrevSizeXDepthBytes);
                            PrevPtrB := (pointer(PrevPtrB) + FSizeXDepthBytes);
                            {$ELSE}
                            PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY + 1);
                            PrevPtrB := LocalWeight.GetRawPtr(0, LocalCntY + 1);
                            {$ENDIF}
                          end;
                        end;
                      end;
                    end;
                  end; // if (FCalculatePrevLayerError)
              end; // (LocalLearningErrorDeriv <> 0.0)
              Inc(OutputRawPos);
            end;
          end;
        end;
      end;
    end;
  end;

  if (not FBatchUpdate) then
  begin
    for OutputD := 0 to MaxD do FArrNeurons[OutputD].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;
end;

procedure TNNetConvolution.BackpropagateFastCPUDev();
var
  OutputX, OutputY, OutputD: integer;
  MaxX, MaxY, MaxD: integer;
  PrevX, PrevY: integer;
  OutputRawPos: integer;
  CanBackpropOnPos: boolean;
  LocalCntY: integer;
  LocalLearningErrorDeriv: TNeuralFloat;
  LocalOutputErrorDeriv: TNeuralFloat;
  SmoothLocalOutputErrorDeriv: TNeuralFloat;
  LocalWeight, LocalPrevError: TNNetVolume;
  {SrcPtr,} LocalDestPtr: TNeuralFloatArrPtr;
  SmoothLocalOutputErrorDerivPtr: pointer;
  PrevNumElements, PrevMissedElements: integer;
  PtrNeuronDelta, PtrPreparedInput: TNeuralFloatArrPtr;
  PrevPtrA, PrevPtrB: TNeuralFloatArrPtr;
  NeuronWeights: integer;
  LocalLearningErrorDerivPtr: pointer;
  localNumElements, MissedElements: integer;
  MaxPrevX, MaxPrevY: integer;
  InterErrorDeriv, InterInput: TNNetVolume;
  NeuronCnt, NeuronPosCnt: integer;
  LocalDelta: TNNetVolume;
begin
  InterErrorDeriv := TNNetVolume.Create();
  InterInput := TNNetVolume.Create();
  MaxX := OutputError.SizeX - 1;
  MaxY := OutputError.SizeY - 1;
  MaxD := OutputError.Depth - 1;
  MaxPrevX := 1 + FPrevLayer.FOutputError.SizeX - FFeatureSizeX;
  MaxPrevY := 1 + FPrevLayer.FOutputError.SizeY - FFeatureSizeY;
  LocalPrevError := FPrevLayer.OutputError;
  PrevNumElements := (FSizeXDepth div 4) * 4;
  PrevMissedElements := FSizeXDepth - PrevNumElements;
  NeuronWeights := FArrNeurons[0].Delta.Size;
  localNumElements := (NeuronWeights div 4) * 4;
  MissedElements := NeuronWeights - localNumElements;
  SmoothLocalOutputErrorDerivPtr := Addr(SmoothLocalOutputErrorDeriv);
  LocalLearningErrorDerivPtr := Addr(LocalLearningErrorDeriv);
    begin
      for OutputY := 0 to MaxY do
      begin
        PrevY := (OutputY*FStride)-FPadding;
        for OutputX := 0 to MaxX do
        begin
          PrevX := (OutputX*FStride)-FPadding;
          OutputRawPos := FOutputErrorDeriv.GetRawPos(OutputX, OutputY);
          if (FCalculatePrevLayerError) then LocalDestPtr  := LocalPrevError.GetRawPtr(OutputX, OutputY);
          PtrPreparedInput := FInputPrepared.GetRawPtr(OutputX, OutputY);
          CanBackpropOnPos :=
            (PrevX >= 0) and (PrevY >= 0) and
            (PrevX < MaxPrevX) and
            (PrevY < MaxPrevY);
          for OutputD := 0 to MaxD do
          begin
            {$IFDEF FPC}
            if FActivationFn = @RectifiedLinearUnit then
            begin
              if FOutput.FData[OutputRawPos] > 0 then
              begin
                LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
              end
              else
              begin
                LocalOutputErrorDeriv := 0;
              end;
            end
            else if FActivationFn = @Identity then
            begin
              LocalOutputErrorDeriv := FOutputError.FData[OutputRawPos];
            end
            else
            begin
              LocalOutputErrorDeriv :=
                FOutputError.FData[OutputRawPos] *
                FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
            end;
            {$ELSE}
              LocalOutputErrorDeriv :=
                FOutputError.FData[OutputRawPos] *
                FActivationFnDerivative(FOutputRaw.FData[OutputRawPos]);
            {$ENDIF}

            FOutputErrorDeriv.FData[OutputRawPos] := LocalOutputErrorDeriv;
            LocalLearningErrorDeriv := (-FLearningRate) * LocalOutputErrorDeriv;
            if (LocalLearningErrorDeriv <> 0.0) then
            begin
                //FArrNeurons[OutputD].Delta.MulAdd(LocalLearningErrorDeriv, PtrPreparedInput);
                {$IFDEF FPC}
                FArrNeurons[OutputD].FBiasDelta += LocalLearningErrorDeriv;
                {$ELSE}
                FArrNeurons[OutputD].FBiasDelta :=
                  FArrNeurons[OutputD].FBiasDelta + LocalLearningErrorDeriv;
                {$ENDIF}

                if (FCalculatePrevLayerError) then
                begin
                  LocalWeight := FArrNeurons[OutputD].Weights;
                  if FPointwise then
                  begin
                    {$IFNDEF AVX64}
                    LocalPrevError.MulAdd(LocalDestPtr, LocalWeight.DataPtr, LocalOutputErrorDeriv, FInputCopy.Depth);
                    {$ELSE}
                    {$IFDEF Debug}
                    if PrevNumElements + PrevMissedElements <> FInputCopy.Depth
                    then FErrorProc('Error at TNNetConvolution.BackpropagateFastCPU(): pointwise vector size doesn''t match.');
                    {$ENDIF}
                    PrevPtrA := LocalDestPtr;
                    PrevPtrB := LocalWeight.DataPtr;
                    SmoothLocalOutputErrorDeriv := LocalOutputErrorDeriv;
                    asm_avx64_prev_backprop;
                    {$ENDIF}
                  end
                  else
                  begin
                    if CanBackpropOnPos then
                    begin
                      SmoothLocalOutputErrorDeriv := LocalOutputErrorDeriv / FLearnSmoothener;
                      PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY);
                      PrevPtrB := LocalWeight.DataPtr;
                      for LocalCntY := 0 to FFeatureSizeYMinus1 do
                      begin
                        {$IFNDEF AVX64}
                        LocalPrevError.MulAdd
                        (
                          PrevPtrA, //LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY),
                          PrevPtrB, //LocalWeight.GetRawPtr(0, LocalCntY),
                          SmoothLocalOutputErrorDeriv,
                          FSizeXDepth
                        );
                        {$ELSE}
                        {$IFDEF Debug}
                        if PrevNumElements + PrevMissedElements <> FSizeXDepth
                        then FErrorProc('Error at TNNetConvolution.BackpropagateFastCPU(): vector size doesn''t match.');
                        {$ENDIF}
                        //PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY);
                        //PrevPtrB := LocalWeight.GetRawPtr(0, LocalCntY);
                        asm_avx64_prev_backprop;
                        {$ENDIF}
                        if LocalCntY < FFeatureSizeYMinus1 then
                        begin
                          {$IFDEF FPC}
                          PrevPtrA := (pointer(PrevPtrA) + FPrevSizeXDepthBytes);
                          PrevPtrB := (pointer(PrevPtrB) + FSizeXDepthBytes);
                          {$ELSE}
                          PrevPtrA := LocalPrevError.GetRawPtr(PrevX, PrevY + LocalCntY + 1);
                          PrevPtrB := LocalWeight.GetRawPtr(0, LocalCntY + 1);
                          {$ENDIF}
                        end;
                      end;
                    end;
                  end;
                end; // if (FCalculatePrevLayerError)
            end; // (LocalLearningErrorDeriv <> 0.0)
            Inc(OutputRawPos);
          end;
        end;
      end;
    end;

  FOutputErrorDeriv.Mul(-FLearningRate);
  InterErrorDeriv.InterleaveWithDepthFrom(FOutputErrorDeriv, FOutputErrorDeriv.SizeX * FOutputErrorDeriv.SizeY);
  InterInput.InterleaveWithDepthFrom(FInputPrepared, FInputPrepared.SizeX * FInputPrepared.SizeY);
  for NeuronCnt := 0 to MaxD do
  begin
    LocalDelta := FArrNeurons[NeuronCnt].Delta;
    for NeuronPosCnt := 0 to NeuronWeights - 1 do
    begin
      {$IFDEF FPC}
      LocalDelta.FData[NeuronPosCnt] +=
      {$ELSE}
      LocalDelta.FData[NeuronPosCnt] :=  LocalDelta.FData[NeuronPosCnt] +
      {$ENDIF}
        TNNetVolume.DotProduct
        (
          InterInput.GetRawPtr(NeuronPosCnt, 0),
          InterErrorDeriv.GetRawPtr(NeuronCnt, 0),
          InterInput.Depth
        );
    end;
  end;

  if (not FBatchUpdate) then
  begin
    for OutputD := 0 to MaxD do FArrNeurons[OutputD].UpdateWeights(FInertia);
    AfterWeightUpdate();
  end;

  InterErrorDeriv.Free;
  InterInput.Free;
end;

constructor TNNetConvolutionAbstract.Create(pFeatureSize, pInputPadding, pStride: integer; pSuppressBias: integer = 0);
begin
  inherited Create();
  FFeatureSizeX := pFeatureSize;
  FFeatureSizeY := pFeatureSize;
  FPadding := pInputPadding;
  FStride := Max(pStride,1);
  FSuppressBias := pSuppressBias;
  if FPadding > 0
    then FInputCopy := TNNetVolume.Create;
end;

destructor TNNetConvolutionAbstract.Destroy();
begin
  if FPadding > 0
    then FInputCopy.Free;
  inherited Destroy();
end;

procedure TNNetConvolutionAbstract.InitDefault();
{$IFDEF Debug}
var
  MaxAbsW: TNeuralFloat;
{$ENDIF}
begin
  InitHeUniform(1);
  (*
  // High values can be usual in small networks.
  {$IFDEF Debug}
  MaxAbsW := FNeurons.GetMaxAbsWeight();
  if MaxAbsW > 0.4 then
  begin
    MulWeights(0.4/MaxAbsW);
    AfterWeightUpdate();
    WriteLn('Too high initial value at layer',Self.LayerIdx,' -> ', MaxAbsW);
  end;
  {$ENDIF}
  *)
end;

{ TNNetFullConnect }
procedure TNNetFullConnect.SetPrevLayer(pPrevLayer: TNNetLayer);
var
  WeightsNum: integer;
begin
  inherited SetPrevLayer(pPrevLayer);
  WeightsNum := pPrevLayer.Output.Size;
  SetNumWeightsForAllNeurons(WeightsNum);
  FVectorSize := FNeurons[0].Weights.Size;
  InitDefault();
  BuildArrNeurons();
  {$IFDEF OpenCL}
  FShouldOpenCL := (FNeurons.Count >= 512) and (pPrevLayer.Output.Size >= 128);
  if (FHasOpenCL and FShouldOpenCL) then
  begin
    FShouldConcatWeights := true;
    FShouldInterleaveWeights := true;
    RefreshNeuronWeightList();
    EnableOpenCL(FDotProductKernel);
  end;
  {$ENDIF}
  AfterWeightUpdate();
end;

procedure TNNetFullConnect.ComputePreviousLayerError();
begin
  if Assigned(FPrevLayer) then
  begin
    if (FPrevLayer.FOutput.Size = FPrevLayer.FOutputError.Size) then
    begin
      ComputePreviousLayerErrorCPU();
    end;
  end;
end;

procedure TNNetFullConnect.ComputePreviousLayerErrorCPU();
var
  MaxOutputCnt: integer;
  OutputCnt: integer;
  LocalPrevError: TNNetVolume;
begin
  LocalPrevError := FPrevLayer.OutputError;

  MaxOutputCnt := FOutput.Size - 1;
  for OutputCnt := 0 to MaxOutputCnt do
  begin
    if (FOutputErrorDeriv.FData[OutputCnt] <> 0.0) then
    begin
      LocalPrevError.MulAdd(FOutputErrorDeriv.FData[OutputCnt], FArrNeurons[OutputCnt].FWeights);
    end;
  end;
end;

(*
procedure TNNetFullConnect.ComputePreviousLayerError;
var
  PrevOutputCnt, MaxCnt, MaxOutputCnt: integer;
  OutputCnt: integer;
  LocalError, LocalErrorDeriv: TNeuralFloat;
  prevFnDeriv: TNeuralActivationFunction;
  {$IFDEF CheckRange} MaxError: TNeuralFloat; {$ENDIF}
begin
  if Assigned(FPrevLayer) then
  begin
    prevFnDeriv := FPrevLayer.ActivationFnDerivative;
    MaxCnt := FPrevLayer.OutputError.Size - 1;
    MaxOutputCnt := FOutput.Size - 1;
    for PrevOutputCnt := 0 to MaxCnt do
    begin
      for OutputCnt := 0 to MaxOutputCnt do
      begin
        FAuxTransposedW.FData[OutputCnt] := FNeurons[OutputCnt].Weights.FData[PrevOutputCnt];
      end;
      LocalError := FAuxTransposedW.DotProduct(FOutputErrorDeriv);

      FPrevLayer.OutputError.FData[PrevOutputCnt] := LocalError;

      LocalErrorDeriv := LocalError *
        prevFnDeriv(FPrevLayer.Output.FData[PrevOutputCnt]);

      FPrevLayer.OutputErrorDeriv.FData[PrevOutputCnt] := LocalErrorDeriv;
    end;

    {$IFDEF CheckRange}
    MaxError := FPrevLayer.OutputError.GetMax();
    if MaxError > 1 then
    begin
      FPrevLayer.OutputError.Divi(MaxError);
      FPrevLayer.OutputErrorDeriv.Divi(MaxError);
    end;
    {$ENDIF}
  end;
end;
*)

constructor TNNetFullConnect.Create(pSizeX, pSizeY, pDepth: integer;
  pSuppressBias: integer);
begin
  inherited Create;
  FOutPut.ReSize(pSizeX, pSizeY, pDepth);
  FOutputRaw.ReSize(pSizeX, pSizeY, pDepth);
  FOutputError.ReSize(pSizeX, pSizeY, pDepth);
  FOutputErrorDeriv.ReSize(pSizeX, pSizeY, pDepth);
  FSuppressBias := pSuppressBias;

  FStruct[0] := pSizeX;
  FStruct[1] := pSizeY;
  FStruct[2] := pDepth;
  FStruct[3] := pSuppressBias;

  AddNeurons(FOutPut.Size);
  FAuxTransposedW := TNNetVolume.Create(FOutPut.Size);
  FActivationFn := @HiperbolicTangent;
  FActivationFnDerivative := @HiperbolicTangentDerivative;
end;

constructor TNNetFullConnect.Create(pSize: integer; pSuppressBias: integer = 0);
begin
  Create(pSize, 1, 1, pSuppressBias);
end;

procedure TNNetFullConnect.Compute();
var
  StartTime: double;
begin
  if (FNeurons.Count = FOutput.Size) and
    (FPrevLayer.Output.Size = FNeurons[0].Weights.Size) then
  begin
    StartTime := Now();
    {$IFDEF OpenCL}
    if Assigned(FDotCL) and FHasOpenCL and FShouldOpenCL then
    begin
      ComputeOpenCL();
    end
    else
    begin
      ComputeCPU();
    end;
    {$ELSE}
    ComputeCPU();
    {$ENDIF}
    FForwardTime := FForwardTime + (Now() - StartTime);
  end else
  begin
    FErrorProc
    (
      'TNNetLayerFullConnect.Compute should have same sizes.'+
      'Neurons:'+IntToStr(FNeurons.Count)+
      ' Prev Layer Output:'+IntToStr(FPrevLayer.Output.Size)+
      ' Output:'+IntToStr(FOutput.Size)
    );
  end;
end;

procedure TNNetFullConnect.ComputeCPU();
var
  Cnt, MaxCnt: integer;
  Sum: TNeuralFloat;
begin
  MaxCnt := FNeurons.Count - 1;
  if FSuppressBias = 0 then
  begin
    for Cnt := 0 to MaxCnt do
    begin
      Sum :=
        FArrNeurons[Cnt].Weights.DotProduct(FPrevLayer.Output) +
        FArrNeurons[Cnt].FBiasWeight;
      FOutputRaw.FData[Cnt] := Sum;
      FOutput.FData[Cnt] := FActivationFn(Sum);
    end;
  end
  else
  begin
    for Cnt := 0 to MaxCnt do
    begin
      Sum :=
        FArrNeurons[Cnt].Weights.DotProduct(FPrevLayer.Output);
      FOutputRaw.FData[Cnt] := Sum;
      FOutput.FData[Cnt] := FActivationFn(Sum);
    end;
  end;
end;

{$IFDEF OpenCL}
procedure TNNetFullConnect.ComputeOpenCL();
var
  InputAVolume: TNNetVolume;
begin
  if FShouldInterleaveWeights then
  begin
    if FConcatedWInter.Size < FNeurons[0].Weights.Size * FNeurons.Count then
    begin
      FErrorProc('Error at TNNetFullConnect.ComputeOpenCL: ' + IntToStr(FConcatedWInter.Size));
      AfterWeightUpdate();
    end;
    InputAVolume := FConcatedWInter;
  end
  else
  begin
    InputAVolume := FConcatedWeights;
  end;
  FDotCL.Compute(InputAVolume, FPrevLayer.FOutput, 0, FAfterWeightUpdateHasBeenCalled, true);
  FAfterWeightUpdateHasBeenCalled := false;
  {$IFDEF Debug}
  if FOutputRaw.Size <> FOutput.Size then
  begin
    FErrorProc('Error at TNNetFullConnect.ComputeOpenCL. Raw output size is:' + IntToStr(FOutputRaw.Size));
    FOutputRaw.Resize(FOutput);
  end;
  if FBiasOutput.Size <> FOutput.Size then
  begin
    FErrorProc('Error at TNNetFullConnect.ComputeOpenCL. Bias size is:' + IntToStr(FOutputRaw.Size));
    FOutputRaw.Resize(FOutput);
  end;
  {$ENDIF}
  {$IFDEF Linux}
  FDotCL.FinishAndLoadResult(FOutputRaw, 0.75);
  {$ELSE}
  FDotCL.FinishAndLoadResult(FOutputRaw, 0.0);
  {$ENDIF}

  if FSuppressBias = 0 then FOutputRaw.Add(FBiasOutput);
  ApplyActivationFunctionToOutput();
end;

procedure TNNetFullConnect.BackpropagateOpenCL();
var
  MaxNeurons, NeuronCnt: integer;
  localLearErrorDeriv: TNeuralFloat;
  localNeuron: TNNetNeuron;
begin
  MaxNeurons := FNeurons.Count - 1;
  for NeuronCnt := 0 to MaxNeurons do
  begin
      OutputErrorDeriv.FData[NeuronCnt] :=
        OutputError.FData[NeuronCnt] *
        FActivationFnDerivative(FOutputRaw.FData[NeuronCnt]);

      localNeuron := FArrNeurons[NeuronCnt];
      localLearErrorDeriv := -FLearningRate * FOutputErrorDeriv.FData[NeuronCnt];

      {$IFDEF Debug}
      if localNeuron.FBackInertia.Size <> FPrevLayer.Output.Size then
      begin
        FErrorProc
        (
          'TNNetLayerFullConnect.Backpropagate should have same sizes.' +
          'Inertia Size:' + IntToStr(localNeuron.FBackInertia.Size) +
          ' PrevLayer Output:' + IntToStr(FPrevLayer.Output.Size)
        );
      end;
      {$ENDIF}
      if (FBatchUpdate) then
      begin
        if localLearErrorDeriv <> 0.0 then
        begin
          localNeuron.Delta.MulAdd(localLearErrorDeriv, FPrevLayer.Output);
          localNeuron.FBiasDelta := localNeuron.FBiasDelta + localLearErrorDeriv;
        end;
      end
      else
      begin
        localNeuron.FBackInertia.MulMulAdd(FInertia, localLearErrorDeriv * (1-FInertia), FPrevLayer.Output);

        localNeuron.FBiasInertia :=
          (1-FInertia)*localLearErrorDeriv +
          (  FInertia)*localNeuron.FBiasInertia;
        {$IFDEF CheckRange}
        NeuronForceRange(localNeuron.FBiasInertia,FLearningRate);
        {$ENDIF}

        localNeuron.AddInertia();
      end;
  end;
  if not FBatchUpdate then AfterWeightUpdate();
end;

procedure TNNetFullConnect.EnableOpenCL(DotProductKernel: TDotProductKernel);
begin
  inherited EnableOpenCL(DotProductKernel);
  FOutputRaw.Resize(FOutput);
  if Assigned(FPrevLayer) and Assigned(FDotCL) then
  begin
    RefreshNeuronWeightList();
    AfterWeightUpdate();
    FDotCL.PrepareForCompute(FConcatedWInter, FPrevLayer.FOutput, FVectorSize);
  end;
end;
{$ENDIF}

procedure TNNetFullConnect.Backpropagate();
var
  StartTime: double;
begin
  Inc(FBackPropCallCurrentCnt);
  if FBackPropCallCurrentCnt < FDepartingBranchesCnt then exit;
  if (FNeurons.Count = FOutput.Size) and (FPrevLayer.FOutput.Size>0) then
  begin
    // ComputeErrorDeriv() is not required as it's done by BackpropagateCPU
    StartTime := Now();
    if FLearningRate <> 0.0 then
    begin
      {$IFDEF OpenCL}
      if Assigned(FDotCL) and FHasOpenCL and FShouldOpenCL then
      begin
        BackpropagateOpenCL();
      end
      else
      begin
        BackpropagateCPU();
      end;
      {$ELSE}
      BackpropagateCPU;
      {$ENDIF}
    end; // of FLearningRate <> 0.0
    {$IFDEF CheckRange} ForceRangeWeights(1000); {$ENDIF}
    ComputePreviousLayerError();
    FBackwardTime := FBackwardTime + (Now() - StartTime);
  end else
  begin
    FErrorProc
    (
      'TNNetLayerFullConnect.Backpropagate should have same sizes.' +
      'Neurons:' + IntToStr(FNeurons.Count) +
      ' Output:' + IntToStr(FOutput.Size) +
      ' PrevLayer:' + IntToStr(FPrevLayer.Output.Size)
    );
  end;
  if Assigned(FPrevLayer) then FPrevLayer.Backpropagate();
end;

procedure TNNetFullConnect.BackpropagateCPU();
var
  MaxNeurons, NeuronCnt: integer;
  localLearErrorDeriv: TNeuralFloat;
  localNeuron: TNNetNeuron;
begin
  MaxNeurons := FNeurons.Count - 1;
  for NeuronCnt := 0 to MaxNeurons do
  begin
      OutputErrorDeriv.FData[NeuronCnt] :=
        OutputError.FData[NeuronCnt] *
        FActivationFnDerivative(FOutputRaw.FData[NeuronCnt]);

      localNeuron := FArrNeurons[NeuronCnt];
      localLearErrorDeriv := -FLearningRate * FOutputErrorDeriv.FData[NeuronCnt];

      {$IFDEF Debug}
      if localNeuron.FBackInertia.Size <> FPrevLayer.Output.Size then
      begin
        FErrorProc
        (
          'TNNetLayerFullConnect.Backpropagate should have same sizes.' +
          'Inertia Size:' + IntToStr(localNeuron.FBackInertia.Size) +
          ' PrevLayer Output:' + IntToStr(FPrevLayer.Output.Size)
        );
      end;
      {$ENDIF}
      if (FBatchUpdate) then
      begin
        if localLearErrorDeriv <> 0.0 then
        begin
          localNeuron.Delta.MulAdd(localLearErrorDeriv, FPrevLayer.Output);
          localNeuron.FBiasDelta := localNeuron.FBiasDelta + localLearErrorDeriv;
        end;
      end
      else
      begin
        if (FInertia<>0.0) or (localLearErrorDeriv<>0.0) then
        begin
          localNeuron.FBackInertia.MulMulAdd(FInertia, localLearErrorDeriv * (1-FInertia), FPrevLayer.Output);

          localNeuron.FBiasInertia :=
            (1-FInertia)*localLearErrorDeriv +
            (  FInertia)*localNeuron.FBiasInertia;
          {$IFDEF CheckRange}
          NeuronForceRange(localNeuron.FBiasInertia,FLearningRate);
          {$ENDIF}

          localNeuron.AddInertia();
        end;
      end;
  end;
  if not FBatchUpdate then AfterWeightUpdate();
end;

destructor TNNetFullConnect.Destroy();
begin
  FAuxTransposedW.Free;
  inherited Destroy;
end;

{ TNNetInput }

procedure TNNetInputBase.ComputePreviousLayerError;
begin
  // Input layer can't compute.
end;

constructor TNNetInput.Create(pSize: integer);
begin
  Create(pSize, 1, 1);
end;

constructor TNNetInput.Create(pSizeX, pSizeY, pDepth: integer);
begin
  inherited Create;
  FOutPut.ReSize(pSizeX, pSizeY, pDepth);

  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;

  FStruct[0] := pSizeX;
  FStruct[1] := pSizeY;
  FStruct[2] := pDepth;
end;

constructor TNNetInput.Create(pSizeX, pSizeY, pDepth, pError: integer);
begin
  Create(pSizeX, pSizeY, pDepth);

  if pError = 1 then
  begin
    EnableErrorCollection();
  end;
end;

function TNNetInput.EnableErrorCollection: TNNetInput;
begin
  if FStruct[3] <> 1 then
  begin
    FStruct[3] := 1;
    FOutputError.ReSize(FOutPut);
    FOutputErrorDeriv.ReSize(FOutPut);
  end;
  Result := Self;
end;

function TNNetInput.DisableErrorCollection: TNNetInput;
begin
  FStruct[3] := 0;
  FOutputError.ReSize(1,1,1);
  FOutputErrorDeriv.ReSize(1,1,1);
  Result := Self;
end;

procedure TNNetInputBase.Compute;
begin
  FOutputError.Fill(0);
  FOutputErrorDeriv.Fill(0);
end;

procedure TNNetInputBase.Backpropagate;
begin
  // Input layer can't backpropagate.
end;

{ TNNet }
procedure TNNet.ComputeL2Decay();
var
  LayerCnt: integer;
begin
  for LayerCnt := 1 to GetLastLayerIdx() do
  begin
    FLayers[LayerCnt].ComputeL2Decay();
  end;
end;

constructor TNNet.Create();
begin
  inherited Create();
  FLayers := TNNetLayerList.Create();
  ClearTime();
  {$IFDEF OpenCL}
  FDotProductKernel := nil;
  {$ENDIF}
end;

destructor TNNet.Destroy();
begin
  {$IFDEF OpenCL}
  if FDotProductKernel <> nil then FDotProductKernel.Free;
  {$ENDIF}
  FLayers.Free;
  inherited Destroy();
end;

function TNNet.CountLayers(): integer;
begin
  Result := FLayers.Count;
end;

function TNNet.CountNeurons(): integer;
var
  LayerCnt: integer;
begin
  Result := 0;
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      Result := Result + FLayers[LayerCnt].CountNeurons();
    end;
  end;
end;

function TNNet.CountWeights(): integer;
var
  LayerCnt: integer;
begin
  Result := 0;
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      Result := Result + FLayers[LayerCnt].CountWeights();
    end;
  end;
end;

function TNNet.GetWeightSum(): TNeuralFloat;
var
  LayerCnt: integer;
begin
  Result := 0;
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      Result := Result + FLayers[LayerCnt].GetWeightSum();
    end;
  end;
end;

function TNNet.GetBiasSum(): TNeuralFloat;
var
  LayerCnt: integer;
begin
  Result := 0;
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      Result := Result + FLayers[LayerCnt].GetBiasSum();
    end;
  end;
end;

function TNNet.CreateLayer(strData: string): TNNetLayer;
var
  S, S2: TStringList;
  St: array [0..csNNetMaxParameterIdx] of integer;
  aL: array of TNNetLayer;
  aIdx: TNeuralIntegerArray;
  IdxCnt: integer;
  I: integer;
begin
  Result := nil;
  S := CreateTokenizedStringList(strData,':');
  S2 := CreateTokenizedStringList(strData,';');

  if S.Count >= 2 then
  begin
    for I := Low(St) to High(St) do St[i] := 0;
    S2.DelimitedText := S[1];
    if S2.Count > 0 then
    begin
      for I := 0 to Min(S2.Count - 1, High(St)) do St[I] := StrToInt(S2[I]);
    end;

    if S.Count = 3 then
    begin
      S2.DelimitedText := S[2];

      if S2.Count > 0 then
      begin
        SetLength(aL, S2.Count);
        SetLength(aIdx, S2.Count);

        for IdxCnt := 0 to S2.Count - 1 do
        begin
          aIdx[IdxCnt] := StrToInt(S2[IdxCnt]);
        end;

        if ( (S[0] = 'TNNetConcat') or (S[0] = 'TNNetDeepConcat') or (S[0] = 'TNNetSum') ) then
        begin
          IdxsToLayers(aIdx, aL);
        end;
      end;
    end;

    {$IFDEF FPC}
    case S[0] of
      'TNNetInput' :                Result := TNNetInput.Create(St[0], St[1], St[2], St[3]);
      'TNNetIdentity' :             Result := TNNetIdentity.Create();
      'TNNetDebug' :                Result := TNNetDebug.Create(St[0], St[1]);
      'TNNetPad' :                  Result := TNNetPad.Create(St[0]);
      'TNNetIdentityWithoutBackprop': Result := TNNetIdentityWithoutBackprop.Create();
      'TNNetReLU' :                 Result := TNNetReLU.Create();
      'TNNetSwish' :                Result := TNNetSwish.Create();
      'TNNetSwish6' :               Result := TNNetSwish6.Create();
      'TNNetReLUSqrt':              Result := TNNetReLUSqrt.Create();
      'TNNetReLUL' :                Result := TNNetReLUL.Create(St[0], St[1], St[2]);
      'TNNetReLU6' :                Result := TNNetReLU6.Create(St[2]);
      'TNNetPower' :                Result := TNNetPower.Create(St[0]);
      'TNNetSELU' :                 Result := TNNetSELU.Create();
      'TNNetLeakyReLU' :            Result := TNNetLeakyReLU.Create();
      'TNNetVeryLeakyReLU' :        Result := TNNetVeryLeakyReLU.Create();
      'TNNetSigmoid' :              Result := TNNetSigmoid.Create();
      'TNNetHyperbolicTangent' :    Result := TNNetHyperbolicTangent.Create();
      'TNNetDropout' :              Result := TNNetDropout.Create(1/St[0], St[1]);
      'TNNetReshape' :              Result := TNNetReshape.Create(St[0], St[1], St[2]);
      'TNNetLayerFullConnect' :     Result := TNNetFullConnect.Create(St[0], St[1], St[2], St[3]);
      'TNNetFullConnect' :          Result := TNNetFullConnect.Create(St[0], St[1], St[2], St[3]);
      'TNNetFullConnectSigmoid':    Result := TNNetFullConnectSigmoid.Create(St[0], St[1], St[2], St[3]);
      'TNNetFullConnectDiff' :      Result := TNNetFullConnectDiff.Create(St[0], St[1], St[2], St[3]);
      'TNNetLayerFullConnectReLU' : Result := TNNetFullConnectReLU.Create(St[0], St[1], St[2], St[3]);
      'TNNetFullConnectReLU' :      Result := TNNetFullConnectReLU.Create(St[0], St[1], St[2], St[3]);
      'TNNetFullConnectLinear' :    Result := TNNetFullConnectLinear.Create(St[0], St[1], St[2], St[3]);
      'TNNetLocalConnect' :         Result := TNNetLocalConnect.Create(St[0], St[1], St[2], St[3], St[4]);
      'TNNetLocalProduct' :         Result := TNNetLocalProduct.Create(St[0], St[1], St[2], St[3], St[4]);
      'TNNetLocalConnectReLU' :     Result := TNNetLocalConnectReLU.Create(St[0], St[1], St[2], St[3], St[4]);
      'TNNetMulLearning'  :         Result := TNNetMulLearning.Create(St[0]);
      'TNNetMulByConstant'  :       Result := TNNetMulByConstant.Create(St[0]);
      'TNNetNegate'  :              Result := TNNetNegate.Create();
      'TNNetLayerSoftMax' :         Result := TNNetSoftMax.Create();
      'TNNetSoftMax' :              Result := TNNetSoftMax.Create();
      'TNNetConvolution' :          Result := TNNetConvolution.Create(St[0], St[1], St[2], St[3], St[4]);
      'TNNetConvolutionReLU' :      Result := TNNetConvolutionReLU.Create(St[0], St[1], St[2], St[3], St[4]);
      'TNNetConvolutionLinear' :    Result := TNNetConvolutionLinear.Create(St[0], St[1], St[2], St[3], St[4]);
      'TNNetGroupedConvolutionLinear' : Result := TNNetGroupedConvolutionLinear.Create(St[0], St[1], St[2], St[3], St[5], St[4]);
      'TNNetGroupedConvolutionReLU'   : Result := TNNetGroupedConvolutionReLU.Create(St[0], St[1], St[2], St[3], St[5], St[4]);
      'TNNetGroupedPointwiseConvLinear' : Result := TNNetGroupedPointwiseConvLinear.Create({pNumFeatures=}St[0], {pGroups=}St[5], {pSuppressBias=}St[4]);
      'TNNetGroupedPointwiseConvReLU'   : Result := TNNetGroupedPointwiseConvReLU.Create({pNumFeatures=}St[0], {pGroups=}St[5], {pSuppressBias=}St[4]);
      'TNNetConvolutionSharedWeights' : Result := TNNetConvolutionSharedWeights.Create(FLayers[St[5]]);
      'TNNetDepthwiseConv' :        Result := TNNetDepthwiseConv.Create(St[0], St[1], St[2], St[3]);
      'TNNetDepthwiseConvReLU' :    Result := TNNetDepthwiseConvReLU.Create(St[0], St[1], St[2], St[3]);
      'TNNetDepthwiseConvLinear' :  Result := TNNetDepthwiseConvLinear.Create(St[0], St[1], St[2], St[3]);
      'TNNetPointwiseConv' :        Result := TNNetPointwiseConv.Create(St[0], St[4]);
      'TNNetPointwiseConvReLU' :    Result := TNNetPointwiseConvReLU.Create(St[0], St[4]);
      'TNNetPointwiseConvLinear' :  Result := TNNetPointwiseConvLinear.Create(St[0], St[4]);
      'TNNetMaxPool' :              Result := TNNetMaxPool.Create(St[0], St[1], St[2]);
      'TNNetMaxPoolPortable' :      Result := TNNetMaxPoolPortable.Create(St[0], St[1], St[2]);
      'TNNetMinPool' :              Result := TNNetMinPool.Create(St[0], St[1], St[2]);
      'TNNetAvgPool' :              Result := TNNetAvgPool.Create(St[0]);
      'TNNetAvgChannel':            Result := TNNetAvgChannel.Create();
      'TNNetMaxChannel':            Result := TNNetMaxChannel.Create();
      'TNNetMinChannel':            Result := TNNetMinChannel.Create();
      'TNNetConcat' :               Result := TNNetConcat.Create(aL);
      'TNNetDeepConcat' :           Result := TNNetDeepConcat.Create(aL);
      'TNNetInterleaveChannels' :   Result := TNNetInterleaveChannels.Create(St[0]);
      'TNNetSum' :                  Result := TNNetSum.Create(aL);
      'TNNetSplitChannels' :        Result := TNNetSplitChannels.Create(aIdx);
      'TNNetSplitChannelEvery' :    Result := TNNetSplitChannelEvery.Create(aIdx);
      'TNNetDeLocalConnect' :       Result := TNNetDeLocalConnect.Create(St[0], St[1], St[4]);
      'TNNetDeLocalConnectReLU' :   Result := TNNetDeLocalConnectReLU.Create(St[0], St[1], St[4]);
      'TNNetDeconvolution' :        Result := TNNetDeconvolution.Create(St[0], St[1], St[4]);
      'TNNetDeconvolutionReLU' :    Result := TNNetDeconvolutionReLU.Create(St[0], St[1], St[4]);
      'TNNetDeMaxPool' :            Result := TNNetDeMaxPool.Create(St[0], St[7]);
      'TNNetDeAvgPool' :            Result := TNNetDeAvgPool.Create(St[0]);
      'TNNetUpsample' :             Result := TNNetUpsample.Create();
      'TNNetLayerMaxNormalization': Result := TNNetLayerMaxNormalization.Create();
      'TNNetLayerStdNormalization': Result := TNNetLayerStdNormalization.Create();
      'TNNetMovingStdNormalization': Result := TNNetMovingStdNormalization.Create();
      'TNNetChannelStdNormalization': Result := TNNetChannelStdNormalization.Create();
      'TNNetScaleLearning' :        Result := TNNetScaleLearning.Create();
      'TNNetChannelBias':           Result := TNNetChannelBias.Create();
      'TNNetChannelMul':            Result := TNNetChannelMul.Create();
      'TNNetChannelMulByLayer':     Result := TNNetChannelMulByLayer.Create(St[0], St[1]);
      'TNNetCellBias':              Result := TNNetCellBias.Create();
      'TNNetCellMul':               Result := TNNetCellMul.Create();
      'TNNetCellMulByCell':         Result := TNNetCellMulByCell.Create(St[0], St[1]);
      'TNNetRandomMulAdd':          Result := TNNetRandomMulAdd.Create(St[0], St[1]);
      'TNNetChannelRandomMulAdd':   Result := TNNetChannelRandomMulAdd.Create(St[0], St[1]);
      'TNNetChannelZeroCenter':     Result := TNNetChannelZeroCenter.Create();
      'TNNetLocalResponseNorm2D':   Result := TNNetLocalResponseNorm2D.Create(St[0]);
      'TNNetLocalResponseNormDepth':Result := TNNetLocalResponseNormDepth.Create(St[0]);
      'TNNetAddAndDiv'             :Result := TNNetAddAndDiv.Create(St[0], St[1]);
    else
       raise Exception.create(strData + ' not allowed in CreateLayer.');
    end;
    {$ELSE}
      if S[0] = 'TNNetInput' then Result := TNNetInput.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetIdentity' then Result := TNNetIdentity.Create() else
      if S[0] = 'TNNetDebug' then Result := TNNetDebug.Create(St[0], St[1]) else
      if S[0] = 'TNNetPad' then Result := TNNetPad.Create(St[0]) else
      if S[0] = 'TNNetIdentityWithoutBackprop' then Result := TNNetIdentityWithoutBackprop.Create() else
      if S[0] = 'TNNetReLU' then Result := TNNetReLU.Create() else
      if S[0] = 'TNNetSwish' then Result := TNNetSwish.Create() else
      if S[0] = 'TNNetSwish6' then Result := TNNetSwish6.Create() else
      if S[0] = 'TNNetReLUSqrt' then Result := TNNetReLUSqrt.Create() else
      if S[0] = 'TNNetReLUL' then Result := TNNetReLUL.Create(St[0], St[1], St[2]) else
      if S[0] = 'TNNetReLU6' then Result := TNNetReLU6.Create(St[2]) else
      if S[0] = 'TNNetPower' then Result := TNNetPower.Create(St[0]) else
      if S[0] = 'TNNetSELU' then Result := TNNetSELU.Create() else
      if S[0] = 'TNNetLeakyReLU' then Result := TNNetLeakyReLU.Create() else
      if S[0] = 'TNNetVeryLeakyReLU' then Result := TNNetVeryLeakyReLU.Create() else
      if S[0] = 'TNNetSigmoid' then Result := TNNetSigmoid.Create() else
      if S[0] = 'TNNetHyperbolicTangent' then Result := TNNetHyperbolicTangent.Create() else
      if S[0] = 'TNNetDropout' then Result := TNNetDropout.Create(1/St[0], St[1]) else
      if S[0] = 'TNNetReshape' then Result := TNNetReshape.Create(St[0], St[1], St[2]) else
      if S[0] = 'TNNetLayerFullConnect' then Result := TNNetFullConnect.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetFullConnect' then Result := TNNetFullConnect.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetFullConnectSigmoid' then Result := TNNetFullConnectSigmoid.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetFullConnectDiff' then Result := TNNetFullConnectDiff.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetLayerFullConnectReLU' then Result := TNNetFullConnectReLU.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetFullConnectReLU' then Result := TNNetFullConnectReLU.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetFullConnectLinear' then Result := TNNetFullConnectLinear.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetLocalConnect' then Result := TNNetLocalConnect.Create(St[0], St[1], St[2], St[3], St[4]) else
      if S[0] = 'TNNetLocalProduct' then Result := TNNetLocalProduct.Create(St[0], St[1], St[2], St[3], St[4]) else
      if S[0] = 'TNNetLocalConnectReLU' then Result := TNNetLocalConnectReLU.Create(St[0], St[1], St[2], St[3], St[4]) else
      if S[0] = 'TNNetMulLearning' then Result := TNNetMulLearning.Create(St[0]) else
      if S[0] = 'TNNetMulByConstant' then Result := TNNetMulByConstant.Create(St[0]) else
      if S[0] = 'TNNetNegate' then Result := TNNetNegate.Create() else
      if S[0] = 'TNNetLayerSoftMax' then Result := TNNetSoftMax.Create() else
      if S[0] = 'TNNetSoftMax' then Result := TNNetSoftMax.Create() else
      if S[0] = 'TNNetConvolution' then Result := TNNetConvolution.Create(St[0], St[1], St[2], St[3], St[4]) else
      if S[0] = 'TNNetConvolutionReLU' then Result := TNNetConvolutionReLU.Create(St[0], St[1], St[2], St[3], St[4]) else
      if S[0] = 'TNNetConvolutionLinear' then Result := TNNetConvolutionLinear.Create(St[0], St[1], St[2], St[3], St[4]) else
      if S[0] = 'TNNetGroupedConvolutionLinear' then Result := TNNetGroupedConvolutionLinear.Create(St[0], St[1], St[2], St[3], St[5], St[4]) else
      if S[0] = 'TNNetGroupedConvolutionReLU' then Result := TNNetGroupedConvolutionReLU.Create(St[0], St[1], St[2], St[3], St[5], St[4]) else
      if S[0] = 'TNNetGroupedPointwiseConvLinear' then Result := TNNetGroupedPointwiseConvLinear.Create({pNumFeatures=}St[0], {pGroups=}St[5], {pSuppressBias=}St[4]) else
      if S[0] = 'TNNetGroupedPointwiseConvReLU' then Result := TNNetGroupedPointwiseConvReLU.Create({pNumFeatures=}St[0], {pGroups=}St[5], {pSuppressBias=}St[4]) else
      if S[0] = 'TNNetConvolutionSharedWeights' then Result := TNNetConvolutionSharedWeights.Create(FLayers[St[5]]) else
      if S[0] = 'TNNetDepthwiseConv' then Result := TNNetDepthwiseConv.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetDepthwiseConvReLU' then Result := TNNetDepthwiseConvReLU.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetDepthwiseConvLinear' then Result := TNNetDepthwiseConvLinear.Create(St[0], St[1], St[2], St[3]) else
      if S[0] = 'TNNetPointwiseConv' then Result := TNNetPointwiseConv.Create(St[0], St[4]) else
      if S[0] = 'TNNetPointwiseConvReLU' then Result := TNNetPointwiseConvReLU.Create(St[0], St[4]) else
      if S[0] = 'TNNetPointwiseConvLinear' then Result := TNNetPointwiseConvLinear.Create(St[0], St[4]) else
      if S[0] = 'TNNetMaxPool' then Result := TNNetMaxPool.Create(St[0], St[1], St[2]) else
      if S[0] = 'TNNetMaxPoolPortable' then Result := TNNetMaxPoolPortable.Create(St[0], St[1], St[2]) else
      if S[0] = 'TNNetMinPool' then Result := TNNetMinPool.Create(St[0], St[1], St[2]) else
      if S[0] = 'TNNetAvgPool' then Result := TNNetAvgPool.Create(St[0]) else
      if S[0] = 'TNNetAvgChannel' then Result := TNNetAvgChannel.Create() else
      if S[0] = 'TNNetMaxChannel' then Result := TNNetMaxChannel.Create() else
      if S[0] = 'TNNetMinChannel' then Result := TNNetMinChannel.Create() else
      if S[0] = 'TNNetConcat' then Result := TNNetConcat.Create(aL) else
      if S[0] = 'TNNetInterleaveChannels' then Result := TNNetInterleaveChannels.Create(St[0]) else
      if S[0] = 'TNNetDeepConcat' then Result := TNNetDeepConcat.Create(aL) else
      if S[0] = 'TNNetSum' then Result := TNNetSum.Create(aL) else
      if S[0] = 'TNNetSplitChannels' then Result := TNNetSplitChannels.Create(aIdx) else
      if S[0] = 'TNNetSplitChannelEvery' then Result := TNNetSplitChannelEvery.Create(aIdx) else
      if S[0] = 'TNNetDeLocalConnect' then Result := TNNetDeLocalConnect.Create(St[0], St[1], St[4]) else
      if S[0] = 'TNNetDeLocalConnectReLU' then Result := TNNetDeLocalConnectReLU.Create(St[0], St[1], St[4]) else
      if S[0] = 'TNNetDeconvolution' then Result := TNNetDeconvolution.Create(St[0], St[1], St[4]) else
      if S[0] = 'TNNetDeconvolutionReLU' then Result := TNNetDeconvolutionReLU.Create(St[0], St[1], St[4]) else
      if S[0] = 'TNNetDeMaxPool' then Result := TNNetDeMaxPool.Create(St[0], St[7]) else
      if S[0] = 'TNNetDeAvgPool' then Result := TNNetDeAvgPool.Create(St[0]) else
      if S[0] = 'TNNetUpsample' then Result := TNNetUpsample.Create() else
      if S[0] = 'TNNetLayerMaxNormalization' then Result := TNNetLayerMaxNormalization.Create() else
      if S[0] = 'TNNetLayerStdNormalization' then Result := TNNetLayerStdNormalization.Create() else
      if S[0] = 'TNNetMovingStdNormalization' then Result := TNNetMovingStdNormalization.Create() else
      if S[0] = 'TNNetChannelStdNormalization' then Result := TNNetChannelStdNormalization.Create() else
      if S[0] = 'TNNetChannelBias' then Result := TNNetChannelBias.Create() else
      if S[0] = 'TNNetChannelMul' then Result := TNNetChannelMul.Create() else
      if S[0] = 'TNNetChannelMulByLayer' then Result := TNNetChannelMulByLayer.Create(St[0], St[1]) else
      if S[0] = 'TNNetCellBias' then Result := TNNetCellBias.Create() else
      if S[0] = 'TNNetCellMul' then Result := TNNetCellMul.Create() else
      if S[0] = 'TNNetCellMulByCell' then Result := TNNetCellMulByCell.Create(St[0], St[1]) else
      if S[0] = 'TNNetRandomMulAdd' then Result := TNNetRandomMulAdd.Create(St[0], St[1]) else
      if S[0] = 'TNNetChannelRandomMulAdd' then Result := TNNetChannelRandomMulAdd.Create(St[0], St[1]) else
      if S[0] = 'TNNetChannelZeroCenter' then Result := TNNetChannelZeroCenter.Create() else
      if S[0] = 'TNNetLocalResponseNorm2D' then   Result := TNNetLocalResponseNorm2D.Create(St[0]) else
      if S[0] = 'TNNetLocalResponseNormDepth' then Result := TNNetLocalResponseNormDepth.Create(St[0]) else
      if S[0] = 'TNNetAddAndDiv' then Result := TNNetAddAndDiv.Create(St[0], St[1]) else
      raise Exception.create(strData + ' not allowed in CreateLayer.');
    {$ENDIF}

  end
  else
  begin
    FErrorProc('Error loading CreateLayer:'+strData+' has '+IntToStr(S.Count)+' parameters.');
  end;
  S2.Free;
  S.Free;
end;

function TNNet.ShouldIncDepartingBranchesCnt(pLayer: TNNetLayer):boolean;
begin
  Result := Not(
    (pLayer is TNNetConcatBase) or
    (pLayer is TNNetChannelMulByLayer) or
    (pLayer is TNNetCellMulByCell)
  );
end;

function TNNet.AddLayer(pLayer: TNNetLayer):TNNetLayer;
var
  AfterLayer: TNNetLayer;
begin
  pLayer.NN := Self;
  if CountLayers()>0 then
  begin
    AfterLayer := FLayers[CountLayers() - 1];
    pLayer.SetPrevLayer(AfterLayer);
    if ShouldIncDepartingBranchesCnt(pLayer)
      then AfterLayer.IncDepartingBranchesCnt();
  end;
  FLayers.Add(pLayer);
  pLayer.FLayerIdx := GetLastLayerIdx();
  Result := pLayer;
end;

function TNNet.AddLayer(strData: string):TNNetLayer;
begin
  Result := AddLayer(CreateLayer(strData));
end;

function TNNet.AddLayer(pLayers: array of TNNetLayer): TNNetLayer;
var
  LocalLayer: TNNetLayer;
begin
  for LocalLayer in pLayers do AddLayer(LocalLayer);
  Result := GetLastLayer();
end;

function TNNet.AddLayerConcatingInputOutput(pLayers: array of TNNetLayer
  ): TNNetLayer;
var
  PrevLayer: TNNetLayer;
begin
  PrevLayer := GetLastLayer();
  AddLayer(pLayers);
  Result := AddLayer(TNNetConcat.Create([PrevLayer, GetLastLayer()]));
end;

function TNNet.AddLayerConcatingInputOutput(pLayer: TNNetLayer): TNNetLayer;
begin
  Result := AddLayerConcatingInputOutput([pLayer]);
end;

function TNNet.AddLayerDeepConcatingInputOutput(pLayers: array of TNNetLayer
  ): TNNetLayer;
var
  PrevLayer: TNNetLayer;
begin
  PrevLayer := GetLastLayer();
  AddLayer(pLayers);
  Result := AddLayer(TNNetDeepConcat.Create([PrevLayer, GetLastLayer()]));
end;

function TNNet.AddLayerDeepConcatingInputOutput(pLayer: TNNetLayer): TNNetLayer;
begin
  Result := AddLayerDeepConcatingInputOutput([pLayer]);
end;

function TNNet.AddSeparableConv(pNumFeatures, pFeatureSize, pInputPadding,
  pStride: integer; pDepthMultiplier: integer; pSuppressBias: integer;
  pAfterLayer: TNNetLayer): TNNetLayer;
begin
  AddLayerAfter( TNNetDepthwiseConvLinear.Create(pDepthMultiplier, pFeatureSize, pInputPadding, pStride), pAfterLayer);
  Result := AddLayer( TNNetPointwiseConv.Create(pNumFeatures, pSuppressBias) );
end;

function TNNet.AddSeparableConvReLU(pNumFeatures, pFeatureSize, pInputPadding,
  pStride: integer; pDepthMultiplier: integer; pSuppressBias: integer;
  pAfterLayer: TNNetLayer): TNNetLayer;
begin
  AddLayerAfter( TNNetDepthwiseConvLinear.Create(pDepthMultiplier, pFeatureSize, pInputPadding, pStride), pAfterLayer);
  Result := AddLayer( TNNetPointwiseConvReLU.Create(pNumFeatures, pSuppressBias) );
end;

function TNNet.AddSeparableConvLinear(pNumFeatures, pFeatureSize,
  pInputPadding, pStride: integer; pDepthMultiplier: integer;
  pSuppressBias: integer; pAfterLayer: TNNetLayer): TNNetLayer;
begin
  AddLayerAfter( TNNetDepthwiseConvLinear.Create(pDepthMultiplier, pFeatureSize, pInputPadding, pStride), pAfterLayer);
  Result := AddLayer( TNNetPointwiseConvLinear.Create(pNumFeatures, pSuppressBias) );
end;

function TNNet.AddGroupedConvolution(Conv2d: TNNetConvolutionClass;
  Groups, pNumFeatures, pFeatureSize, pInputPadding, pStride: integer;
  pSuppressBias: integer; ChannelInterleaving: boolean): TNNetLayer;
var
  PreviousLayer: TNNetLayer;
  FeaturesPerGroup: integer;
  InputChannelsPerGroup: integer;
  EachGroupOutput: array of TNNetLayer;
  GroupCnt: integer;
begin
  PreviousLayer := GetLastLayer();
  Result := PreviousLayer;
  SetLength(EachGroupOutput, Groups);
  FeaturesPerGroup := pNumFeatures div Groups;
  InputChannelsPerGroup := PreviousLayer.FOutput.Depth div Groups;
  if Groups = 1 then
  begin
    Result := AddLayer( Conv2d.Create(FeaturesPerGroup, pFeatureSize, pInputPadding, pStride, pSuppressBias) );
  end;
  if Groups > 1 then
  begin
    for GroupCnt := 0 to Groups - 1 do
    begin
      if ChannelInterleaving
        then AddLayerAfter( TNNetSplitChannelEvery.Create(Groups, GroupCnt), PreviousLayer)
        else AddLayerAfter( TNNetSplitChannels.Create(GroupCnt*InputChannelsPerGroup, InputChannelsPerGroup), PreviousLayer);
      EachGroupOutput[GroupCnt] := AddLayer( Conv2d.Create(FeaturesPerGroup, pFeatureSize, pInputPadding, pStride, pSuppressBias) );
    end;
    Result := AddLayer( TNNetDeepConcat.Create(EachGroupOutput) );
  end;
  SetLength(EachGroupOutput, 0);
end;

function TNNet.AddAutoGroupedPointwiseConv(
  Conv2d: TNNetGroupedPointwiseConvClass;
  MinChannelsPerGroupCount, pNumFeatures: integer;
  HasNormalization: boolean;
  pSuppressBias: integer = 0;
  HasIntergroup: boolean = true
  ): TNNetLayer;
var
  MaxGroupCount: integer;
  GroupCount: integer;
  PrevLayerChannelCount: integer;
  OutputGroupSize: integer;
  FirstLayer: TNNetLayer;
begin
  PrevLayerChannelCount := GetLastLayer().Output.Depth;
  MaxGroupCount := (PrevLayerChannelCount div MinChannelsPerGroupCount);
  GroupCount := GetMaxAcceptableCommonDivisor(
    PrevLayerChannelCount, pNumFeatures, MaxGroupCount);
  FirstLayer := AddLayer(
    Conv2d.Create(
      {Features=}pNumFeatures,
      {Groups=}GroupCount,
      {SupressBias=}pSuppressBias) );
  if HasNormalization then
    FirstLayer := AddLayer( TNNetChannelStdNormalization.Create() );
//  WriteLn(
//      'Group count:',GroupCount,
//      ' Output group size:', pNumFeatures div GroupCount,
//      ' Input group size:', PrevLayerChannelCount div GroupCount
//  );
  if ( (GroupCount > 1) and (HasIntergroup) ) then
  begin
    OutputGroupSize := pNumFeatures div GroupCount;
    AddLayer( TNNetInterleaveChannels.Create(OutputGroupSize) );
    if (PrevLayerChannelCount >= pNumFeatures) then
    begin
      AddLayer(
        Conv2d.Create(
          {Features=}pNumFeatures,
          {Groups=}GroupCount,
          {SupressBias=}pSuppressBias) );
      if HasNormalization then
        AddLayer( TNNetChannelStdNormalization.Create() );
      {$IFDEF Debug}
      if (FirstLayer.Output.Depth <> GetLastLayer().Output.Depth) then
      begin
        WriteLn('AddAutoGroupedPointwiseConv - Bad input channel counts:',
          FirstLayer.Output.Depth,' ',
          GetLastLayer().Output.Depth
        );
      end;
      {$ENDIF}
      AddLayer( TNNetSum.Create([GetLastLayer(), FirstLayer]) );
    end;
  end;

  Result := GetLastLayer();
end;

function TNNet.AddAutoGroupedPointwiseConv2(
  Conv2d: TNNetGroupedPointwiseConvClass; MinChannelsPerGroupCount,
  pNumFeatures: integer; HasNormalization: boolean; pSuppressBias: integer;
  AlwaysIntergroup: boolean;
  HasIntergroup: boolean): TNNetLayer;
var
  MaxGroupCount, SecondMaxGroupCount: integer;
  GroupCount, SecondGroupCount: integer;
  PrevLayerChannelCount: integer;
  FeaturesPerGroup: integer;
  FirstLayer: TNNetLayer;
begin
  PrevLayerChannelCount := GetLastLayer().Output.Depth;
  MaxGroupCount := (PrevLayerChannelCount div MinChannelsPerGroupCount);
  GroupCount := GetMaxAcceptableCommonDivisor(
    PrevLayerChannelCount, pNumFeatures, MaxGroupCount);
  FirstLayer := AddLayer(
    Conv2d.Create(
      {Features=}pNumFeatures,
      {Groups=}GroupCount,
      {SupressBias=}pSuppressBias) );
  if HasNormalization then
    FirstLayer := AddLayer( TNNetChannelStdNormalization.Create() );
  //WriteLn(
  //    'Group count:', GroupCount,
  //    ' Output group size:', pNumFeatures div GroupCount,
  //    ' Input group size:', PrevLayerChannelCount div GroupCount
  //);
  if ( (GroupCount > 1) and (HasIntergroup) ) then
  begin
    SecondMaxGroupCount := (pNumFeatures div MinChannelsPerGroupCount);
    SecondGroupCount := GetMaxAcceptableCommonDivisor(
      pNumFeatures, pNumFeatures, SecondMaxGroupCount);
    if (SecondGroupCount > 1) then
    begin
      if (PrevLayerChannelCount >= pNumFeatures) or (AlwaysIntergroup) then
      begin
        FeaturesPerGroup := pNumFeatures div GroupCount;
        AddLayer( TNNetInterleaveChannels.Create(FeaturesPerGroup) );
        AddLayer(
          Conv2d.Create(
            {Features=}pNumFeatures,
            {Groups=}SecondGroupCount,
            {SupressBias=}pSuppressBias) );
        //WriteLn
        //(
        //  'Second group count:', SecondGroupCount,
        //  ' Input/Output second group size:', pNumFeatures div SecondGroupCount
        //);

        if HasNormalization then
          AddLayer( TNNetChannelStdNormalization.Create() );
        {$IFDEF Debug}
        if (FirstLayer.Output.Depth <> GetLastLayer().Output.Depth) then
        begin
          WriteLn('AddAutoGroupedPointwiseConv - Bad input channel counts:',
            FirstLayer.Output.Depth,' ',
            GetLastLayer().Output.Depth
          );
        end;
        {$ENDIF}
        AddLayer( TNNetSum.Create([GetLastLayer(), FirstLayer]) );
      end;
    end;
  end;

  Result := GetLastLayer();
end;

function TNNet.AddAutoGroupedConvolution(Conv2d: TNNetConvolutionClass;
  MinChannelsPerGroupCount, pNumFeatures, pFeatureSize, pInputPadding, pStride: integer;
  pSuppressBias: integer; ChannelInterleaving: boolean): TNNetLayer;
var
  MaxGroupCount: integer;
  GroupCount: integer;
  PrevLayerChannelCount: integer;
begin
  PrevLayerChannelCount := GetLastLayer().Output.Depth;
  MaxGroupCount := (PrevLayerChannelCount div MinChannelsPerGroupCount);
  GroupCount := GetMaxAcceptableCommonDivisor(
    PrevLayerChannelCount, pNumFeatures, MaxGroupCount);
  Result := AddGroupedConvolution(Conv2d, GroupCount, pNumFeatures,
    pFeatureSize, pInputPadding, pStride, pSuppressBias, ChannelInterleaving);
end;

function TNNet.AddGroupedFullConnect(FullConnect: TNNetFullConnectClass;
  Groups, pNumFeatures: integer; pSuppressBias: integer;
  ChannelInterleaving: boolean): TNNetLayer;
var
  PreviousLayer: TNNetLayer;
  FeaturesPerGroup: integer;
  InputChannelsPerGroup: integer;
  EachGroupOutput: array of TNNetLayer;
  GroupCnt: integer;
begin
  if Groups > 1 then
  begin
    PreviousLayer := AddLayer( TNNetReshape.Create(1, 1, GetLastLayer().Output.Size) );
  end
  else
  begin
    PreviousLayer := GetLastLayer();
  end;
  Result := PreviousLayer;
  SetLength(EachGroupOutput, Groups);
  FeaturesPerGroup := pNumFeatures div Groups;
  InputChannelsPerGroup := PreviousLayer.FOutput.Depth div Groups;
  if Groups = 1 then
  begin
    Result := AddLayer( FullConnect.Create(FeaturesPerGroup, pSuppressBias) );
  end;
  if Groups > 1 then
  begin
    for GroupCnt := 0 to Groups - 1 do
    begin
      if ChannelInterleaving
        then AddLayerAfter( TNNetSplitChannelEvery.Create(Groups, GroupCnt), PreviousLayer)
        else AddLayerAfter( TNNetSplitChannels.Create(GroupCnt*InputChannelsPerGroup, InputChannelsPerGroup), PreviousLayer);
      EachGroupOutput[GroupCnt] := AddLayer( FullConnect.Create(FeaturesPerGroup, pSuppressBias) );
    end;
    Result := AddLayer( TNNetDeepConcat.Create(EachGroupOutput) );
  end;
  SetLength(EachGroupOutput, 0);
end;

function TNNet.AddMovingNorm(PerCell: boolean = false; pAfterLayer:
  TNNetLayer = nil): TNNetLayer;
begin
  AddLayerAfter( TNNetMovingStdNormalization.Create(), pAfterLayer);
  if PerCell then
  begin
    AddLayer( TNNetCellMul.Create() );
    AddMovingNorm := AddLayer( TNNetCellBias.Create() );
  end
  else
  begin
    AddLayer( TNNetChannelMul.Create() );
    AddMovingNorm := AddLayer( TNNetChannelBias.Create() );
  end;
end;

function TNNet.AddMovingNorm(PerCell: boolean; RandomBias,
  RandomAmplifier: integer; pAfterLayer: TNNetLayer): TNNetLayer;
begin
  Self.AddMovingNorm(PerCell, pAfterLayer);
  if (RandomBias>0) and (RandomAmplifier>0) then
    Self.AddLayer( TNNetRandomMulAdd.Create(RandomBias, RandomAmplifier) );
  Result := GetLastLayer();
end;

function TNNet.AddChannelMovingNorm(PerCell: boolean; RandomBias,
  RandomAmplifier: integer; pAfterLayer: TNNetLayer): TNNetLayer;
begin
  AddLayerAfter( TNNetChannelStdNormalization.Create(), pAfterLayer);
  if PerCell then
  begin
    AddLayer( TNNetCellMul.Create() );
    AddLayer( TNNetCellBias.Create() );
  end
  else
  begin
    AddLayer( TNNetChannelMul.Create() );
    AddLayer( TNNetChannelBias.Create() );
  end;
  if (RandomBias>0) and (RandomAmplifier>0) then
    Self.AddLayer( TNNetChannelRandomMulAdd.Create(RandomBias, RandomAmplifier) );
  Result := GetLastLayer();
end;

function TNNet.AddConvOrSeparableConv(IsSeparable, HasReLU, HasNorm: boolean;
  pNumFeatures, pFeatureSize, pInputPadding, pStride: integer;
  PerCell: boolean; pSuppressBias: integer; RandomBias: integer;
  RandomAmplifier: integer; pAfterLayer: TNNetLayer): TNNetLayer;
begin
  if (HasReLU) then
  begin
    if (IsSeparable) then
    begin
      AddConvOrSeparableConv := AddSeparableConvReLU(pNumFeatures, pFeatureSize,
        pInputPadding, pStride, {DepthMultiplier=}1, pSuppressBias,
        pAfterLayer);
    end
    else
    begin
      AddConvOrSeparableConv := AddLayerAfter( TNNetConvolutionReLU.Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pSuppressBias), pAfterLayer);
    end;
  end
  else
  begin
    if (IsSeparable) then
    begin
      AddConvOrSeparableConv := AddSeparableConvLinear(pNumFeatures, pFeatureSize,
        pInputPadding, pStride, {DepthMultiplier=}1, pSuppressBias,
        pAfterLayer);
    end
    else
    begin
      AddConvOrSeparableConv := AddLayerAfter( TNNetConvolutionLinear.Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pSuppressBias), pAfterLayer);
    end;
  end;

  if (HasNorm) then
  begin
    AddConvOrSeparableConv := AddChannelMovingNorm(PerCell, RandomBias, RandomAmplifier);
  end;
end;

function TNNet.AddConvOrSeparableConv(IsSeparable: boolean; pNumFeatures,
  pFeatureSize, pInputPadding, pStride: integer;
  pSuppressBias: integer; pActFn: TNNetActivationFunctionClass;
  pAfterLayer: TNNetLayer): TNNetLayer;
begin
  if (IsSeparable) then
  begin
    AddConvOrSeparableConv := AddSeparableConvLinear(pNumFeatures, pFeatureSize,
      pInputPadding, pStride, {DepthMultiplier=}1, pSuppressBias,
      pAfterLayer);
  end
  else
  begin
    AddConvOrSeparableConv := AddLayerAfter( TNNetConvolutionLinear.Create(pNumFeatures, pFeatureSize, pInputPadding, pStride, pSuppressBias), pAfterLayer);
  end;
  if pActFn <> nil then
  begin
    AddConvOrSeparableConv := AddLayer( pActFn.Create() );
  end;
end;

function TNNet.AddCompression(Compression: TNeuralFloat; supressBias: integer): TNNetLayer;
begin
  AddLayer( TNNetPointwiseConvLinear.Create(Round(GetLastLayer().Output.Depth * Compression ), supressBias) );
  Result := GetLastLayer();
end;

function TNNet.AddGroupedCompression(Compression: TNeuralFloat;
  MinGroupSize:integer; supressBias: integer;
  HasIntergroup: boolean): TNNetLayer;
var
  FilterCount: integer;
begin
  FilterCount := Round(GetLastLayer().Output.Depth * Compression );
  if Odd(FilterCount) then Inc(FilterCount);
  //AddAutoGroupedConvolution(TNNetPointwiseConvLinear,
  //  {MinGroupSize=}MinGroupSize, {pNumFeatures=}FilterCount, {pFeatureSize=}1,
  //  {pInputPadding=}0, {pStride=}1, supressBias,
  //  {ChannelInterleaving=} true);
  AddAutoGroupedPointwiseConv2(TNNetGroupedPointwiseConvLinear, MinGroupSize,
    FilterCount, False, supressBias, False, HasIntergroup);
  Result := GetLastLayer();
end;

function TNNet.AddMinMaxPool(pPoolSize: integer; pStride:integer = 0; pPadding: integer = 0): TNNetLayer;
var
  PreviousLayer, MinPool, MaxPool: TNNetLayer;
begin
  PreviousLayer := GetLastLayer();
  if pPoolSize > 0 then
  begin
    MinPool := AddLayerAfter(TNNetMinPool.Create(pPoolSize, pStride, pPadding), PreviousLayer);
    MaxPool := AddLayerAfter(TNNetMaxPool.Create(pPoolSize, pStride, pPadding), PreviousLayer);
    Result := AddLayer( TNNetDeepConcat.Create([MinPool, MaxPool]) );
  end
  else Result := PreviousLayer;
end;

function TNNet.AddAvgMaxPool(pPoolSize: integer; pMaxPoolDropout: TNeuralFloat;
  pKeepDepth:boolean; pAfterLayer: TNNetLayer): TNNetLayer;
var
  PreviousLayer, AvgPool, MaxPool: TNNetLayer;
begin
  if pAfterLayer = nil
  then PreviousLayer := GetLastLayer()
  else PreviousLayer := pAfterLayer;
  if pPoolSize > 0 then
  begin
    AvgPool := AddLayerAfter(TNNetAvgPool.Create(pPoolSize), PreviousLayer);
    if pMaxPoolDropout > 0 then
    begin
      AddLayerAfter(TNNetDropout.Create(pMaxPoolDropout), PreviousLayer);
      MaxPool := AddLayer( TNNetMaxPool.Create(pPoolSize) );
    end
    else
    begin
      MaxPool := AddLayerAfter( TNNetMaxPool.Create(pPoolSize), PreviousLayer);
    end;
    Result := AddLayer( TNNetDeepConcat.Create([AvgPool, MaxPool]) );
    if pKeepDepth
      then Result := AddLayer( TNNetPointwiseConvLinear.Create(PreviousLayer.Output.Depth) )
  end
  else Result := PreviousLayer;
end;

function TNNet.AddMinMaxChannel(pAfterLayer: TNNetLayer = nil): TNNetLayer;
var
  PreviousLayer, MinPool, MaxPool: TNNetLayer;
begin
  if pAfterLayer = nil
  then PreviousLayer := GetLastLayer()
  else PreviousLayer := pAfterLayer;
  MinPool := AddLayerAfter(TNNetMinChannel.Create(), PreviousLayer);
  MaxPool := AddLayerAfter(TNNetMaxChannel.Create(), PreviousLayer);
  Result := AddLayer( TNNetDeepConcat.Create([MinPool, MaxPool]) );
end;

function TNNet.AddAvgMaxChannel(pMaxPoolDropout: TNeuralFloat; pKeepDepth:
  boolean; pAfterLayer:TNNetLayer): TNNetLayer;
var
  PreviousLayer, AvgPool, MaxPool: TNNetLayer;
begin
  if pAfterLayer = nil
  then PreviousLayer := GetLastLayer()
  else PreviousLayer := pAfterLayer;
  AvgPool := AddLayerAfter(TNNetAvgChannel.Create(), PreviousLayer);
  if pMaxPoolDropout > 0 then
  begin
    AddLayerAfter(TNNetDropout.Create(pMaxPoolDropout), PreviousLayer);
    MaxPool := AddLayer( TNNetMaxChannel.Create() );
  end
  else
  begin
    MaxPool := AddLayerAfter( TNNetMaxChannel.Create(), PreviousLayer);
  end;
  Result := AddLayer( TNNetDeepConcat.Create([AvgPool, MaxPool]) );
  if pKeepDepth
    then Result := AddLayer( TNNetPointwiseConvLinear.Create(PreviousLayer.Output.Depth) )
end;

procedure TNNet.AddToExponentialWeightAverage(NewElement: TNNet; Decay: TNeuralFloat);
begin
  MulMulAddWeights(Decay, 1 - Decay, NewElement);
end;

procedure TNNet.AddToWeightAverage(NewElement: TNNet; CurrentElementCount: integer);
begin
  MulMulAddWeights(CurrentElementCount/(CurrentElementCount+1), 1/(CurrentElementCount+1), NewElement);
end;

function TNNet.AddLayerAfter(pLayer, pAfterLayer: TNNetLayer): TNNetLayer;
begin
  if Assigned(pAfterLayer) then
  begin
    pLayer.NN := Self;
    pLayer.SetPrevLayer(pAfterLayer);
    FLayers.Add(pLayer);
    pLayer.FLayerIdx := GetLastLayerIdx();
    if Not(pLayer is TNNetConcatBase) then pAfterLayer.IncDepartingBranchesCnt();
    Result := pLayer;
  end
  else
  begin
    Result := AddLayer(pLayer);
  end;
end;

function TNNet.AddLayerAfter(pLayer: TNNetLayer; pAfterLayerIdx: integer
  ): TNNetLayer;
begin
  pLayer.NN := Self;
  if pAfterLayerIdx >= 0 then
  begin
    pLayer.SetPrevLayer(FLayers[pAfterLayerIdx]);
    if Not(pLayer is TNNetConcatBase) then FLayers[pAfterLayerIdx].IncDepartingBranchesCnt();
  end;
  FLayers.Add(pLayer);
  pLayer.FLayerIdx := GetLastLayerIdx();
  Result := pLayer;
end;

function TNNet.AddLayerAfter(strData: string; pAfterLayerIdx: integer
  ): TNNetLayer;
begin
  Result := AddLayerAfter(CreateLayer(strData), pAfterLayerIdx);
end;

function TNNet.AddLayerAfter(pLayers: array of TNNetLayer; pLayer: TNNetLayer
  ): TNNetLayer;
var
  LocalLayer: TNNetLayer;
  LayerCnt: integer;
begin
  LayerCnt := 0;
  for LocalLayer in pLayers do
  begin
    if LayerCnt = 0
    then AddLayerAfter(LocalLayer, pLayer)
    else AddLayer(LocalLayer);
    Inc(LayerCnt);
  end;
  Result := GetLastLayer();
end;

function TNNet.AddLayerAfter(pLayers: array of TNNetLayer;
  pAfterLayerIdx: integer): TNNetLayer;
begin
  Result := AddLayerAfter(pLayers, FLayers[pAfterLayerIdx]);
end;

function TNNet.GetFirstNeuronalLayerIdx(FromLayerIdx:integer = 0): integer;
var
  LayerCnt: integer;
begin
  Result := -1;
  if FLayers.Count > FromLayerIdx then
  begin
    for LayerCnt := FromLayerIdx to GetLastLayerIdx() do
    begin
      if (FLayers[LayerCnt].Neurons.Count > 0) then
      begin
        Result := LayerCnt;
        Break;
      end;
    end;
  end;
end;

function TNNet.GetFirstImageNeuronalLayerIdx(FromLayerIdx: integer): integer;
var
  LayerCnt: integer;
  WeightDepth: integer;
begin
  Result := -1;
  if FLayers.Count > FromLayerIdx then
  begin
    for LayerCnt := FromLayerIdx to GetLastLayerIdx() do
    begin
      if (FLayers[LayerCnt].Neurons.Count > 0) then
      begin
        WeightDepth := FLayers[LayerCnt].Neurons[0].Weights.Depth;
        if ( (WeightDepth>=1) and (WeightDepth<=3) ) then
        begin
          Result := LayerCnt;
          Break;
        end;
      end;
    end;
  end;
end;

function TNNet.GetFirstNeuronalLayerIdxWithChannels(FromLayerIdx,
  Channels: integer): integer;
var
  LayerCnt: integer;
  WeightDepth: integer;
begin
  Result := -1;
  if FLayers.Count > FromLayerIdx then
  begin
    for LayerCnt := FromLayerIdx to GetLastLayerIdx() do
    begin
      if (FLayers[LayerCnt].Neurons.Count > 0) then
      begin
        WeightDepth := FLayers[LayerCnt].Neurons[0].Weights.Depth;
        if (WeightDepth = Channels) then
        begin
          Result := LayerCnt;
          Break;
        end;
      end;
    end;
  end;
end;

procedure TNNet.Compute(pInput: TNNetVolume; FromLayerIdx:integer = 0);
var
  LayerCnt: integer;
  LastLayer: integer;
  StartTime: double;
begin
  StartTime := Now();
  if FLayers.Count > FromLayerIdx + 1 then
  begin
    if FLayers[FromLayerIdx].FOutput.Size = pInput.Size then
    begin
      FLayers[FromLayerIdx].FOutput.CopyNoChecks(pInput);
      LastLayer := GetLastLayerIdx();
      for LayerCnt := FromLayerIdx to LastLayer do
      begin
        FLayers[LayerCnt].Compute();
      end;
    end else
    begin
      FErrorProc
      (
        'Compute - Wrong Input Size:'+IntToStr(pInput.Size) +
        ' Expected size is:' + IntToStr(FLayers[FromLayerIdx].Output.Size) +
        ' Have you missed the TNNetInput layer?'
      );
    end;
  end else
  begin
    FErrorProc('Compute - Neural Network doesn''t have suficcient layers.');
  end;
  FForwardTime := FForwardTime + (Now() - StartTime);
end;

procedure TNNet.Compute(pInput: array of TNNetVolume);
var
  MaxInputs: integer;
  InputCnt: integer;
begin
  MaxInputs := Length(pInput) - 1;
  if MaxInputs >= 1 then
  begin
    for InputCnt := 1 to MaxInputs do
    begin
      if FLayers[InputCnt].FOutput.Size = pInput[InputCnt].Size then
      begin
        FLayers[InputCnt].FOutput.CopyNoChecks(pInput[InputCnt]);
      end
      else
      begin
        FErrorProc
        (
          'Compute - Wrong Input Size:'+IntToStr(pInput[InputCnt].Size) + ' on layer '+
          IntToStr(InputCnt) +
          ' Expected size is:' + IntToStr(FLayers[InputCnt].Output.Size) +
          ' Have you missed the TNNetInput layer?'
        );
      end;
    end;
  end;
  if MaxInputs >= 0 then
  begin
    Compute(pInput[0]);
  end;
end;

procedure TNNet.Compute(pInput: array of TNeuralFloatDynArr);
var
  MaxInputs: integer;
  InputCnt: integer;
begin
  MaxInputs := Length(pInput) - 1;
  if MaxInputs >= 1 then
  begin
    for InputCnt := 1 to MaxInputs do
    begin
      if FLayers[InputCnt].FOutput.Size = Length(pInput[InputCnt]) then
      begin
        FLayers[InputCnt].FOutput.Copy(pInput[InputCnt]);
      end
      else
      begin
        FErrorProc
        (
          'Compute - Wrong Input Size:'+IntToStr(Length(pInput[InputCnt])) + ' on layer '+
          IntToStr(InputCnt) +
          ' Expected size is:' + IntToStr(FLayers[InputCnt].Output.Size) +
          ' Have you missed the TNNetInput layer?'
        );
      end;
    end;
  end;
  if MaxInputs >= 0 then
  begin
    Compute(pInput[0], 0);
  end;
end;

procedure TNNet.Compute(pInput: array of TNeuralFloat; FromLayerIdx:integer = 0);
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(pInput);
  Compute(V, FromLayerIdx);
  V.Free;
end;

procedure TNNet.Compute(pInput, pOutput: TNNetVolumeList; FromLayerIdx: integer
  );
var
  AuxOutput: TNNetVolume;
  MaxIdxInput, IdxInput: integer;
begin
  MaxIdxInput := pInput.Count - 1;
  if MaxIdxInput >=0 then
  begin
    AuxOutput := TNNetVolume.Create();
    for IdxInput := 0 to MaxIdxInput do
    begin
      Self.Compute(pInput[IdxInput], FromLayerIdx);
      Self.GetOutput(AuxOutput);
      if (pOutput.Count > IdxInput) then
      begin
        pOutput[IdxInput].Copy(AuxOutput);
      end
      else
      begin
        pOutput.AddCopy(AuxOutput);
      end;
      pOutput[IdxInput].Tags[0] := pInput[IdxInput].Tags[0];
      pOutput[IdxInput].Tags[1] := pInput[IdxInput].Tags[1];
      if (IdxInput mod 1000 = 0) and (IdxInput > 0) then
      begin
        MessageProc(IntToStr(IdxInput)+' processed.');
      end;
    end;
    AuxOutput.Free;
  end;
end;

procedure TNNet.Compute(pInput, pOutput: TNNetVolume; FromLayerIdx: integer);
begin
  Self.Compute(pInput, FromLayerIdx);
  Self.GetOutput(pOutput);
end;

procedure TNNet.Backpropagate(pOutput: TNNetVolume);
var
  LastLayer: integer;
  StartTime: double;
begin
  StartTime := Now();
  if FLayers.Count > 1 then
  begin
    ResetBackpropCallCurrCnt();
    LastLayer := GetLastLayerIdx();
    FLayers[LastLayer].ComputeOutputErrorWith(pOutput);
    FLayers[LastLayer].Backpropagate();
    ComputeL2Decay();
  end else
  begin
    FErrorProc('Backpropagate - Neural Network doesn''t have suficcient layers.');
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
end;

procedure TNNet.BackpropagateForIdx(pOutput: TNNetVolume;
  const aIdx: array of integer);
var
  LastLayer: integer;
  StartTime: double;
begin
  StartTime := Now();
  if FLayers.Count > 1 then
  begin
    ResetBackpropCallCurrCnt();
    LastLayer := GetLastLayerIdx();
    FLayers[LastLayer].ComputeOutputErrorForIdx(pOutput, aIdx);
    FLayers[LastLayer].Backpropagate();
    ComputeL2Decay();
  end else
  begin
    FErrorProc('Backpropagate - Neural Network doesn''t have suficcient layers.');
  end;
  FBackwardTime := FBackwardTime + (Now() - StartTime);
end;

procedure TNNet.BackpropagateFromLayerAndNeuron(LayerIdx, NeuronIdx: integer; Error: TNeuralFloat);
begin
  ResetBackpropCallCurrCnt();
  Layers[LayerIdx].ComputeOutputErrorForOneNeuron(NeuronIdx, Error);
  Layers[LayerIdx].Backpropagate();
end;

procedure TNNet.Backpropagate(pOutput: array of TNeuralFloat);
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(pOutput);
  Backpropagate(V);
  V.Free;
end;

procedure TNNet.GetOutput(pOutput: TNNetVolume);
begin
  if FLayers.Count > 0 then
  begin
    pOutput.Copy( FLayers[GetLastLayerIdx()].Output );
  end;
end;

procedure TNNet.AddOutput(pOutput: TNNetVolume);
begin
  if FLayers.Count > 0 then
  begin
    pOutput.Add( FLayers[GetLastLayerIdx()].Output );
  end;
end;

procedure TNNet.SetActivationFn(ActFn, ActFnDeriv: TNeuralActivationFunction);
var
  LayerCnt: integer;
begin
  if FLayers.Count > 1 then
  begin
    for LayerCnt := GetLastLayerIdx() downto 0 do
    begin
      FLayers[LayerCnt].ActivationFn := ActFn;
      FLayers[LayerCnt].ActivationFnDerivative := ActFnDeriv;
    end;
  end;
end;

function TNNet.GetLastLayerIdx(): integer;
begin
  Result := CountLayers() - 1;
end;

function TNNet.GetLastLayer(): TNNetLayer;
begin
  Result := FLayers[GetLastLayerIdx()];
end;

function TNNet.GetRandomLayer(): TNNetLayer;
begin
  Result := FLayers[Random(FLayers.Count)];
end;

procedure TNNet.InitWeights();
var
  LayerCnt: integer;
begin
  if FLayers.Count > 1 then
  begin
    for LayerCnt := GetLastLayerIdx() downto 0 do
    begin
      FLayers[LayerCnt].InitDefault();
    end;
  end;
  ClearInertia();
  ClearDeltas();
end;

procedure TNNet.MulWeights(V: TNeuralFloat);
var
  LayerCnt: integer;
begin
  if FLayers.Count > 1 then
  begin
    for LayerCnt := 1 to GetLastLayerIdx() do
    begin
      if not(FLayers[LayerCnt].LinkedNeurons) then FLayers[LayerCnt].MulWeights( V );
    end;
  end;
end;

procedure TNNet.MulDeltas(V: TNeuralFloat);
var
  LayerCnt: integer;
begin
  if FLayers.Count > 1 then
  begin
    for LayerCnt := 1 to GetLastLayerIdx() do
    begin
      if not(FLayers[LayerCnt].LinkedNeurons) then FLayers[LayerCnt].MulDeltas( V );
    end;
  end;
end;

procedure TNNet.SumWeights(Origin: TNNet);
var
  LayerCnt: integer;
begin
  FForwardTime := FForwardTime + Origin.FForwardTime;
  FBackwardTime := FBackwardTime + Origin.FBackwardTime;
  if FLayers.Count = Origin.Layers.Count then
  begin
    if FLayers.Count > 1 then
    begin
      for LayerCnt := 1 to GetLastLayerIdx() do
      begin
        if not(FLayers[LayerCnt].LinkedNeurons) then FLayers[LayerCnt].SumWeights(Origin.Layers[LayerCnt]);
      end;
    end;
  end
  else
  begin
    FErrorProc
    (
      'TNNet.SumWeights - SumWeights does not match: ' + IntToStr(Origin.Layers.Count)
    );
  end;
end;

procedure TNNet.SumDeltas(Origin: TNNet);
var
  LayerCnt: integer;
  MaxLayerIdx: integer;
begin
  if FLayers.Count = Origin.Layers.Count then
  begin
    if FLayers.Count > 1 then
    begin
      FForwardTime := FForwardTime + Origin.FForwardTime;
      FBackwardTime := FBackwardTime + Origin.FBackwardTime;
      MaxLayerIdx := GetLastLayerIdx();
      for LayerCnt := 1 to MaxLayerIdx do
      begin
        if not(FLayers[LayerCnt].LinkedNeurons) then
        begin
          FLayers[LayerCnt].SumDeltas(Origin.Layers[LayerCnt]);
          FLayers[LayerCnt].AddTimes(Origin.Layers[LayerCnt]);
        end;
      end;
    end;
  end
  else
  begin
    FErrorProc
    (
      'TNNet.SumDelta - SumWeights does not match: ' + IntToStr(Origin.Layers.Count)
    );
  end;
end;

procedure TNNet.SumDeltasNoChecks(Origin: TNNet);
var
  LayerCnt: integer;
  MaxLayerIdx: integer;
begin
  FForwardTime := FForwardTime + Origin.FForwardTime;
  FBackwardTime := FBackwardTime + Origin.FBackwardTime;
  MaxLayerIdx := GetLastLayerIdx();
  for LayerCnt := 1 to MaxLayerIdx do
  begin
    if not(FLayers[LayerCnt].LinkedNeurons) then
    begin
      FLayers[LayerCnt].SumDeltasNoChecks(Origin.Layers[LayerCnt]);
      FLayers[LayerCnt].AddTimes(Origin.Layers[LayerCnt]);
    end;
  end;
end;

procedure TNNet.CopyWeights(Origin: TNNet);
var
  LayerCnt: integer;
  MaxLayerIdx: integer;
begin
  FForwardTime := Origin.FForwardTime;
  FBackwardTime := Origin.FBackwardTime;
  if FLayers.Count = Origin.Layers.Count then
  begin
    if FLayers.Count > 1 then
    begin
      MaxLayerIdx := GetLastLayerIdx();
      for LayerCnt := 1 to MaxLayerIdx do
      begin
        if not(FLayers[LayerCnt].LinkedNeurons) then
        begin
          FLayers[LayerCnt].CopyWeights(Origin.Layers[LayerCnt]);
          FLayers[LayerCnt].CopyTimes(Origin.Layers[LayerCnt]);
        end;
      end;
    end;
  end
  else
  begin
    FErrorProc
    (
      'TNNet.CopyWeights does not match. Origin: ' + IntToStr(Origin.Layers.Count) +
      ' Self: ' + IntToStr(FLayers.Count)
    );
  end;
end;

function TNNet.ForceMaxAbsoluteDelta(vMax: TNeuralFloat): TNeuralFloat;
var
  LayerCnt: integer;
  LayerMul: TNeuralFloat;
begin
  Result := 1;
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      if not(FLayers[LayerCnt].LinkedNeurons) then
      begin
        LayerMul := FLayers[LayerCnt].ForceMaxAbsoluteDelta(vMax);
        {$IFDEF Debug}
        if LayerMul < Result then
        begin
          Result := LayerMul;
          MessageProc('Deltas have been multiplied by '+FloatToStr(LayerMul)+
            ' on layer '+IntToStr(LayerCnt)+' - '+
            FLayers[LayerCnt].ClassName+'.');
        end;
        {$ENDIF}
      end;
    end;
  end;
end;

function TNNet.ForceMaxAbsoluteWeight(vMax: TNeuralFloat): TNeuralFloat;
var
  LayerCnt: integer;
  LayerMax: TNeuralFloat;
begin
  Result := 0;
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      if (
         (not(FLayers[LayerCnt].LinkedNeurons)) and
         (not(FLayers[LayerCnt] is TNNetChannelShiftBase)) and
         (not(FLayers[LayerCnt] is TNNetScaleLearning))
         ) then
      begin
        LayerMax := FLayers[LayerCnt].ForceMaxAbsoluteWeight(vMax);
        if LayerMax > Result then
        begin
          Result := LayerMax;
        end;
      end;
    end;
  end;
end;

function TNNet.GetMaxAbsoluteDelta(): TNeuralFloat;
var
  LayerCnt: integer;
  LayerDelta: TNeuralFloat;
begin
  Result := 0;
  FMaxDeltaLayer := 0;
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      LayerDelta := FLayers[LayerCnt].GetMaxAbsoluteDelta();
      if Result < LayerDelta then
      begin
        Result := LayerDelta;
        FMaxDeltaLayer := LayerCnt;
      end;
    end;
  end;
end;

function TNNet.NormalizeMaxAbsoluteDelta(NewMax: TNeuralFloat): TNeuralFloat;
var
  MaxAbsolute: TNeuralFloat;
begin
  MaxAbsolute := GetMaxAbsoluteDelta();
  if MaxAbsolute > NewMax then
  begin
    Result := NewMax / MaxAbsolute;
    MulDeltas(Result);
  end
  else
  begin
    Result := 1;
  end;
end;

procedure TNNet.ClearInertia();
var
  LayerCnt: integer;
begin
  if FLayers.Count > 1 then
  begin
    for LayerCnt := 1 to GetLastLayerIdx() do
    begin
      FLayers[LayerCnt].ClearInertia();
    end;
  end;
end;

{$IFDEF OpenCL}
procedure TNNet.DisableOpenCL();
var
  LayerCnt: integer;
begin
  for LayerCnt := 0 to GetLastLayerIdx() do
  begin
    FLayers[LayerCnt].DisableOpenCL();
  end;
end;

procedure TNNet.EnableOpenCL(platform_id: cl_platform_id;
  device_id: cl_device_id);
var
  LayerCnt: integer;
begin
  FDotProductKernel := TDotProductCL.Create(platform_id, device_id);
  for LayerCnt := 0 to GetLastLayerIdx() do
  begin
    FLayers[LayerCnt].EnableOpenCL(FDotProductKernel);
  end;
end;
{$ENDIF}

procedure TNNet.MulWeightsGlorotBengio(V: TNeuralFloat);
var
  LayerCnt: integer;
  MulAux: Single;
  NeuronCnt: integer;
begin
  // This implementation is inspired on:
  // Understanding the difficulty of training deep feedforward neural networks
  // Xavier Glorot, Yoshua Bengio ; Proceedings of the Thirteenth International
  // Conference on Artificial Intelligence and Statistics, PMLR 9:249-256, 2010.
  // http://proceedings.mlr.press/v9/glorot10a.html
  if FLayers.Count > 1 then
  begin
    for LayerCnt := 1 to GetLastLayerIdx() do
    begin
      NeuronCnt := FLayers[LayerCnt].Neurons.Count;
      if NeuronCnt>0 then
      begin
        MulAux := V*Sqrt(6/(FLayers[LayerCnt].Neurons[0].Weights.Size+NeuronCnt));
        FLayers[LayerCnt].MulWeights( MulAux );
      end;
    end;
  end;
end;

procedure TNNet.MulWeightsHe(V: TNeuralFloat);
var
  LayerCnt: integer;
  MulAux: Single;
  NeuronCnt: integer;
begin
  if FLayers.Count > 1 then
  begin
    for LayerCnt := 1 to GetLastLayerIdx() do
    begin
      NeuronCnt := FLayers[LayerCnt].Neurons.Count;
      if NeuronCnt>0 then
      begin
        MulAux := V*Sqrt(2/(FLayers[LayerCnt].Neurons[0].Weights.Size));
        FLayers[LayerCnt].MulWeights( MulAux );
      end;
    end;
  end;
end;

procedure TNNet.SetLearningRate(pLearningRate, pInertia: TNeuralFloat);
var
  LayerCnt: integer;
begin
  FLearningRate := pLearningRate;
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      FLayers[LayerCnt].FLearningRate := pLearningRate;
      FLayers[LayerCnt].FInertia := pInertia;
    end;
  end;
end;

procedure TNNet.SetL2Decay(pL2Decay: TNeuralFloat);
var
  LayerCnt: integer;
begin
  if ( (pL2Decay > 0) and (FLayers.Count > 1) )  then
  begin
    for LayerCnt := 1 to GetLastLayerIdx() do
    begin
      FLayers[LayerCnt].L2Decay := pL2Decay;
    end;
  end;
end;

procedure TNNet.SetBatchUpdate(pBatchUpdate: boolean);
var
  LayerCnt: integer;
begin
  for LayerCnt := 0 to GetLastLayerIdx() do
  begin
    FLayers[LayerCnt].SetBatchUpdate(pBatchUpdate);
  end;
end;

procedure TNNet.UpdateWeights();
var
  LayerCnt: integer;
begin
  for LayerCnt := 0 to GetLastLayerIdx() do
  begin
    FLayers[LayerCnt].UpdateWeights();
  end;
end;

procedure TNNet.ClearDeltas();
var
  LayerCnt: integer;
begin
  for LayerCnt := 0 to GetLastLayerIdx() do
  begin
    FLayers[LayerCnt].ClearDeltas();
  end;
end;

procedure TNNet.ResetBackpropCallCurrCnt();
var
  LayerCnt: integer;
begin
  for LayerCnt := 0 to GetLastLayerIdx() do
  begin
    FLayers[LayerCnt].ResetBackpropCallCurrCnt();
  end;
end;

procedure TNNet.SetL2DecayToConvolutionalLayers(pL2Decay: TNeuralFloat);
var
  LayerCnt: integer;
begin
  if ( (pL2Decay > 0) and (FLayers.Count > 1) )  then
  begin
    for LayerCnt := 1 to GetLastLayerIdx() do
    begin
      if FLayers[LayerCnt] is TNNetConvolutionBase
        then FLayers[LayerCnt].L2Decay := pL2Decay;
    end;
  end;
end;

procedure TNNet.SetSmoothErrorPropagation(p: boolean);
var
  LayerCnt: integer;
begin
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      FLayers[LayerCnt].SmoothErrorPropagation := p;
    end;
  end;
end;

procedure TNNet.ClearTime();
var
  LayerCnt: integer;
begin
  FForwardTime := 0;
  FBackwardTime := 0;
  for LayerCnt := 0 to GetLastLayerIdx() do
  begin
    FLayers[LayerCnt].ClearTimes();
  end;
end;

procedure TNNet.Clear();
begin
  FLayers.Free;
  FLayers := TNNetLayerList.Create();
  ClearTime();
end;

procedure TNNet.DebugWeights();
var
  LayerCnt, NeuronCount: integer;
begin
  if FLayers.Count > 1 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      NeuronCount := FLayers[LayerCnt].FNeurons.Count;
      if NeuronCount > 0 then
      begin
        Write
        (
          'Layer ',LayerCnt:2,
          ' Neurons:',NeuronCount:3,
          ' Max Weight: ',FLayers[LayerCnt].GetMaxWeight():7:3,
          ' Min Weight: ',FLayers[LayerCnt].GetMinWeight():7:3,
          ' Max Output: ',FLayers[LayerCnt].Output.GetMax():6:3,
          ' Min Output: ',FLayers[LayerCnt].Output.GetMin():6:3,
          ' ' + FLayers[LayerCnt].ClassName + ' ' +
          IntToStr(FLayers[LayerCnt].Output.SizeX) + ',' +
          IntToStr(FLayers[LayerCnt].Output.SizeY) + ',' +
          IntToStr(FLayers[LayerCnt].Output.Depth)
        );
      end
      else
      begin
        Write
        (
          'Layer ',LayerCnt:2,
          '             ',' ':14,
          '             ',' ':13,
           'Max Output: ',FLayers[LayerCnt].Output.GetMax():6:3,
          ' Min Output: ',FLayers[LayerCnt].Output.GetMin():6:3,
          ' ' + FLayers[LayerCnt].ClassName + ' ' +
          IntToStr(FLayers[LayerCnt].Output.SizeX) + ',' +
          IntToStr(FLayers[LayerCnt].Output.SizeY) + ',' +
          IntToStr(FLayers[LayerCnt].Output.Depth)
        );
      end;
      Write
      (
          ' Times: ',
          (FLayers[LayerCnt].FForwardTime  * 24 * 60 * 60):4:2, 's ',
          (FLayers[LayerCnt].FBackwardTime * 24 * 60 * 60):4:2, 's'
      );

      if Assigned(FLayers[LayerCnt].PrevLayer) then
      begin
        WriteLn(' Parent:',FLayers[LayerCnt].PrevLayer.LayerIdx);
      end
      else
      begin
        WriteLn;
      end;

    end; // of for
  end;
end;

procedure TNNet.DebugErrors();
var
  LayerCnt: integer;
begin
  if FLayers.Count > 1 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
        Write
        (
          'Layer ',LayerCnt:2,
          ' Max Error: ', FLayers[LayerCnt].OutputError.GetMax():12:7,
          ' Min Error: ', FLayers[LayerCnt].OutputError.GetMin():12:7,
          ' Max ErrorD: ',FLayers[LayerCnt].OutputErrorDeriv.GetMax():6:3,
          ' Min ErrorD: ',FLayers[LayerCnt].OutputErrorDeriv.GetMin():6:3,
          ' ' + FLayers[LayerCnt].ClassName + ' ' +
          IntToStr(FLayers[LayerCnt].Output.SizeX) + ',' +
          IntToStr(FLayers[LayerCnt].Output.SizeY) + ',' +
          IntToStr(FLayers[LayerCnt].Output.Depth)
        );

      if Assigned(FLayers[LayerCnt].PrevLayer) then
      begin
        WriteLn(' Parent:',FLayers[LayerCnt].PrevLayer.LayerIdx);
      end
      else
      begin
        WriteLn;
      end;

    end; // of for
  end;
end;

procedure TNNet.DebugStructure();
var
  LayerCnt, NeuronCount, WeightCount: integer;
begin
  WriteLn(' Layers: ', CountLayers()  );
  WriteLn(' Neurons:', CountNeurons() );
  WriteLn(' Weights:' ,CountWeights(), ' Sum:', GetWeightSum():12:6 );

  if FLayers.Count > 1 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      WeightCount := FLayers[LayerCnt].CountWeights();
      NeuronCount := FLayers[LayerCnt].CountNeurons();
      Write
        (
          'Layer ',LayerCnt:2,
          ' Neurons:',NeuronCount:4,
          ' Weights:',WeightCount:6,
          ' ' + FLayers[LayerCnt].ClassName + '(' +
          IntToStr(FLayers[LayerCnt].FStruct[0]) + ',' +
          IntToStr(FLayers[LayerCnt].FStruct[1]) + ',' +
          IntToStr(FLayers[LayerCnt].FStruct[2]) + ',' +
          IntToStr(FLayers[LayerCnt].FStruct[3]) + ',' +
          IntToStr(FLayers[LayerCnt].FStruct[4]) + ') Output:' +
          IntToStr(FLayers[LayerCnt].Output.SizeX) + ',' +
          IntToStr(FLayers[LayerCnt].Output.SizeY) + ',' +
          IntToStr(FLayers[LayerCnt].Output.Depth),
          ' Learning Rate:',FLayers[LayerCnt].LearningRate:6:4,
          ' Inertia:',FLayers[LayerCnt].Inertia:4:2,
          ' Weight Sum:', FLayers[LayerCnt].GetWeightSum():8:4
        );

      if Assigned(FLayers[LayerCnt].PrevLayer) then
      begin
        Write(' Parent:',FLayers[LayerCnt].PrevLayer.LayerIdx);
      end;
      WriteLn(' Branches:',FLayers[LayerCnt].FDepartingBranchesCnt );
    end; // of for
  end;
end;

procedure TNNet.IdxsToLayers(aIdx: array of integer; var aL: array of TNNetLayer);
var
  IdxCnt: integer;
  LayerIdx: integer;
begin
  for IdxCnt := Low(aIdx) to High(aIdx) do
  begin
    LayerIdx := aIdx[IdxCnt];
    aL[IdxCnt] := FLayers[LayerIdx];
  end;
end;

procedure TNNet.EnableDropouts(pFlag: boolean);
var
  LayerCnt: integer;
begin
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      if (FLayers[LayerCnt] is TNNetAddNoiseBase) then
      begin
        TNNetAddNoiseBase(FLayers[LayerCnt]).Enabled := pFlag;
      end;
    end;
  end;
end;

procedure TNNet.RefreshDropoutMask();
var
  LayerCnt: integer;
begin
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      if (FLayers[LayerCnt] is TNNetDropout) then
      begin
        TNNetDropout(FLayers[LayerCnt]).RefreshDropoutMask();
      end;
    end;
  end;
end;

procedure TNNet.MulMulAddWeights(Value1, Value2: TNeuralFloat; Origin: TNNet);
var
  LayerCnt: integer;
begin
  FForwardTime := FForwardTime * Value1 + Origin.FForwardTime * Value2;
  FBackwardTime := FBackwardTime * Value1 + Origin.FBackwardTime * Value2;
  if FLayers.Count = Origin.Layers.Count then
  begin
    if FLayers.Count > 1 then
    begin
      for LayerCnt := 1 to GetLastLayerIdx() do
      begin
        FLayers[LayerCnt].MulMulAddWeights(Value1, Value2, Origin.Layers[LayerCnt]);
      end;
    end;
  end
  else
  begin
    FErrorProc
    (
      'TNNet.SumWeights - SumWeights does not match: ' + IntToStr(Origin.Layers.Count)
    );
  end;
end;

procedure TNNet.MulAddWeights(Value: TNeuralFloat; Origin: TNNet);
begin
  MulMulAddWeights(1, Value, Origin);
end;

function TNNet.SaveDataToString(): string;
var
  LayerCnt: integer;
  S: TStringList;
begin
  S := CreateTokenizedStringList('!');
  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      S.Add( FLayers[LayerCnt].SaveDataToString() );
    end;
  end;
  Result := S.DelimitedText;
  S.Free;
end;

function TNNet.SaveStructureToString(): string;
var
  LayerCnt: integer;
  S: TStringList;
  PrevLayerIdx: integer;
begin
  S := CreateTokenizedStringList('#');

  if FLayers.Count > 0 then
  begin
    for LayerCnt := 0 to GetLastLayerIdx() do
    begin
      PrevLayerIdx := -1;
      if Assigned(FLayers[LayerCnt].PrevLayer) then
      begin
        PrevLayerIdx := FLayers[LayerCnt].PrevLayer.FLayerIdx;
      end;
      S.Add( IntToStr(PrevLayerIdx) + ')' + FLayers[LayerCnt].SaveStructureToString() );
    end;
  end;
  Result := S.DelimitedText;
  S.Free;
end;

procedure TNNet.LoadStructureFromString(strData: string);
var
  S, S2: TStringList;
  Cnt: integer;
begin
  Clear();
  S := CreateTokenizedStringList(strData,'#');
  S2 := CreateTokenizedStringList(')');
  if S.Count > 0 then
  begin
    for Cnt := 0 to S.Count - 1 do
    begin
      S2.DelimitedText := S[Cnt];
      AddLayerAfter(S2[1], StrToInt(S2[0]));
    end;
  end;

  S2.Free;
  S.Free;
end;

function TNNet.SaveToString(): string;
begin
  Result :=
    SaveStructureToString() + '>' +
    SaveDataToString();
end;

procedure TNNet.SaveToFile(filename: string);
var
  S: TStringList;
begin
  S := CreateTokenizedStringList(SaveToString(),'>');
  S.SaveToFile(filename);
  S.Free;
end;

procedure TNNet.LoadFromString(strData: string);
var
  S: TStringList;
begin
  S := CreateTokenizedStringList(strData, '>');

  if (S.Count = 2) then
  begin
    LoadStructureFromString(S[0]);
    LoadDataFromString(S[1]);
  end
  else
  begin
    FErrorProc
    (
      'TNNet.LoadFromString - wrong number of arguments: ' + IntToStr(S.Count)
    );
  end;

  S.Free;
end;

procedure TNNet.LoadFromFile(filename: string);
var
  S: TStringList;
begin
  S := CreateTokenizedStringList('>');
  S.LoadFromFile(filename);

  if (S.Count = 2) then
  begin
    LoadStructureFromString(S[0]);
    LoadDataFromString(S[1]);
  end
  else
  begin
    FErrorProc
    (
      'TNNet.LoadFromString - wrong number of arguments: ' + IntToStr(S.Count)
    );
  end;

  S.Free;
end;

function TNNet.Clone(): TNNet;
var
  NNData: String;
begin
  NNData := SaveToString();

  Result := TNNet.Create;
  Result.LoadFromString(NNData);
end;

procedure TNNet.LoadDataFromString(strData: string);
var
  S: TStringList;
  Cnt: integer;
begin
  S := CreateTokenizedStringList(strData,'!');

  if S.Count = FLayers.Count then
  begin
    if S.Count > 0 then
    begin
      for Cnt := 0 to S.Count - 1 do
      begin
        FLayers[Cnt].LoadDataFromString(S[Cnt]);
      end;
    end;
  end
  else
  begin
    FErrorProc
    (
      'Error while loading network: number of structure layers '+
      IntToStr(FLayers.Count)+' differ from data loaded layers '+
      IntToStr(S.Count)
    );

    {$IFDEF Debug}
    WriteLn('Loaded Data Layers:');
    if S.Count > 0 then
    begin
      for Cnt := 0 to S.Count - 1 do
      begin
        Writeln(Cnt, ':', Copy(S[Cnt],1,20) );
      end;
    end;

    WriteLn('Structure Layers:');
    if FLayers.Count>0 then
    begin
      for Cnt := 0 to FLayers.Count - 1 do
      begin
        Writeln(Cnt, ':', FLayers[Cnt].ClassName );
      end;
    end;
    {$ENDIF}
  end;

  S.Free;
end;

procedure TNNet.LoadDataFromFile(filename: string);
var
  S: TStringList;
begin
  S := CreateTokenizedStringList('>');
  S.LoadFromFile(filename);

  if (S.Count = 2) then
  begin
    LoadDataFromString(S[1]);
  end
  else
  begin
    FErrorProc
    (
      'TNNet.LoadFromString - wrong number of arguments: ' + IntToStr(S.Count)
    );
  end;

  S.Free;
end;

procedure TNNetLayer.InitStruct();
var
  I: integer;
begin
  for I := Low(FStruct) to High(FStruct) do
  begin
    FStruct[I] := 0;
  end;
end;

procedure TNNetLayer.ComputePreviousLayerError();
begin
  // to be implemented by inherited classes
end;

procedure TNNetLayer.SetPrevLayer(pPrevLayer: TNNetLayer);
begin
  FPrevLayer := pPrevLayer;
end;

procedure TNNetLayer.ApplyActivationFunctionToOutput();
var
  OutputCnt, OutputMax: integer;
begin
  OutputMax := FOutput.Size - 1;
  if OutputMax >= 0 then
  begin
    {$IFDEF DEBUG}
    if FOutput.Size <> FOutputRaw.Size then
    begin
      FErrorProc
      (
        'Output size ' + IntToStr(FOutput.Size) +
        ' differs from raw output size '+IntToStr(FOutputRaw.Size)
      );
    end;
    {$ENDIF}
    {$IFDEF FPC}
    if FActivationFn = @Identity then
    begin
      FOutput.Copy(FOutputRaw);
    end
    else
    if FActivationFn = @RectifiedLinearUnit then
    begin
      FOutput.CopyRelu(FOutputRaw);
    end
    else
    begin
      for OutputCnt := 0 to OutputMax do
      begin
        FOutput.FData[OutputCnt] := FActivationFn(FOutputRaw.FData[OutputCnt]);
      end;
    end;
    {$ELSE}
      for OutputCnt := 0 to OutputMax do
      begin
        FOutput.FData[OutputCnt] := FActivationFn(FOutputRaw.FData[OutputCnt]);
      end;
    {$ENDIF}
  end;
end;

procedure TNNetLayer.BuildArrNeurons();
var
  NeuronIdx: integer;
begin
  SetLength(FArrNeurons, FNeurons.Count);
  for NeuronIdx := 0 to FNeurons.Count - 1 do
  begin
    FArrNeurons[NeuronIdx] := FNeurons[NeuronIdx];
  end;
end;

{ TNNetLayer }
constructor TNNetLayer.Create();
begin
  inherited Create();
  InitStruct();
  FOutput := TNNetVolume.Create(1,1,1);
  FOutputRaw := TNNetVolume.Create(1,1,1);
  FOutputError := TNNetVolume.Create(1,1,1);
  FOutputErrorDeriv := TNNetVolume.Create(1,1,1);

  FNeurons := TNNetNeuronList.Create();
  FLinkedNeurons := false;
  FActivationFn := @Identity;
  FActivationFnDerivative := @IdentityDerivative;
  FLearningRate := 0.01;
  FL2Decay := 0;
  FPrevLayer := nil;
  FInertia := 0.9;
  FLayerIdx := -1;
  FBatchUpdate := false;
  FSmoothErrorPropagation := false;
  FDepartingBranchesCnt := 0;
  FBackPropCallCurrentCnt := 0;
  FBackwardTime := 0;
  FForwardTime := 0;
  FSuppressBias := 0;
  {$IFDEF OpenCL}
  FDotCL := nil;
  DisableOpenCL();
  {$ENDIF}
end;

destructor TNNetLayer.Destroy();
begin
  {$IFDEF OpenCL}
  if Assigned(FDotCL) then
  begin
    FDotCL.Free;
    FDotCL := nil;
  end;
  {$ENDIF}
  FOutputError.Free;
  FOutputErrorDeriv.Free;
  FOutputRaw.Free;
  FOutput.Free;
  FNeurons.Free;
  inherited Destroy();
end;

{$IFDEF OpenCL}
procedure TNNetLayer.DisableOpenCL();
begin
  if Assigned(FDotCL) then
  begin
    FDotCL.Free;
    FDotCL := nil;
  end;

  FHasOpenCL := false;
  FShouldOpenCL := false;
end;

procedure TNNetLayer.EnableOpenCL(DotProductKernel: TDotProductKernel);
begin
  FHasOpenCL := true;
  FDotProductKernel := DotProductKernel;
end;
{$ENDIF}

procedure TNNetLayer.ComputeL2Decay();
begin
  if ( (FNeurons.Count > 0) and (FL2Decay > 0) ) then
  begin
    MulWeights( 1 - ( FL2Decay * FLearningRate) );
  end;
end;

procedure TNNetLayer.ComputeOutputErrorForOneNeuron(NeuronIdx: integer;
  value: TNeuralFloat);
begin
  FOutputError.Fill(0);
  FOutputErrorDeriv.Fill(0);
  if
    (FOutputError.Size = FOutput.Size) and
    (FOutputErrorDeriv.Size = FOutputError.Size)
    then
  begin
    if (FOutputError.Depth > 1) then
    begin
      FOutputError.AddAtDepth(NeuronIdx, FOutput);
      FOutputError.AddAtDepth(NeuronIdx, -value);
      ComputeErrorDeriv();
    end
    else
    begin
      {$IFDEF FPC}
        FOutputError.FData[NeuronIdx] += FOutput.FData[NeuronIdx] - value;
        FOutputErrorDeriv.FData[NeuronIdx] += FOutputError.FData[NeuronIdx] *
          FActivationFnDerivative(FOutput.FData[NeuronIdx]);
      {$ELSE}
        FOutputError.FData[NeuronIdx] := FOutputError.FData[NeuronIdx] + FOutput.FData[NeuronIdx] - value;
        FOutputErrorDeriv.FData[NeuronIdx] := FOutputErrorDeriv.FData[NeuronIdx] + FOutputError.FData[NeuronIdx] *
          FActivationFnDerivative(FOutput.FData[NeuronIdx]);
      {$ENDIF}
    end;
  end else
  begin
    FErrorProc
    (
      'ComputeOutputErrorWith should have same sizes.' +
      'Neurons:' + IntToStr(FNeurons.Count) +
      ' Output:' + IntToStr(FOutput.Size) +
      ' Error:' + IntToStr(FOutputError.Size) +
      ' Deriv:' + IntToStr(FOutputErrorDeriv.Size)
    );
  end;
end;

procedure TNNetLayer.ComputeOutputErrorWith(pOutput: TNNetVolume);
  {$IFDEF CheckRange}var MaxError:TNeuralFloat; {$ENDIF}
begin
  if
    (pOutput.Size = FOutput.Size) and
    (pOutput.Size = FOutputError.Size) then
  begin
    FOutputError.CopyNoChecks(FOutput);
    FOutputError.Sub(pOutput);

    {$IFDEF CheckRange}
    MaxError := FPrevLayer.OutputError.GetMax();
    if MaxError > 1 then
    begin
      FOutputError.Divi(MaxError);
    end;
    {$ENDIF}
  end else
  begin
    FErrorProc
    (
      'ComputeOutputErrorWith should have same sizes.' +
      'Neurons:' + IntToStr(FNeurons.Count) +
      ' Output:' + IntToStr(FOutput.Size) +
      ' Expected output:' + IntToStr(pOutput.Size) +
      ' Error:' + IntToStr(FOutputError.Size) +
      ' Error times Deriv:' + IntToStr(FOutputErrorDeriv.Size)
    );
  end;
end;

procedure TNNetLayer.ComputeOutputErrorForIdx(pOutput: TNNetVolume;
  const aIdx: array of integer);
var
  Idx: integer;
begin
  for Idx in aIdx do
  begin
    FOutputError.FData[Idx] := (FOutput.FData[Idx] - pOutput.FData[Idx]);
  end;
end;

procedure TNNetLayer.ComputeErrorDeriv();
  procedure FallbackComputeErrorDeriv();
  var
    MaxOutput, OutputCnt: integer;
  begin
    MaxOutput := OutputError.Size - 1;
    for OutputCnt := 0 to MaxOutput do
    begin
      OutputErrorDeriv.FData[OutputCnt] :=
        OutputError.FData[OutputCnt] *
        FActivationFnDerivative(FOutputRaw.FData[OutputCnt]);
    end;
  end;
begin
  {$IFDEF FPC}
  if FActivationFn = @RectifiedLinearUnit then
  begin
    FallbackComputeErrorDeriv();
  end
  else if FActivationFn = @Identity then
  begin
    FOutputErrorDeriv.Copy(FOutputError);
  end
  else
  begin
    FallbackComputeErrorDeriv();
  end;
  {$ELSE}
  FallbackComputeErrorDeriv();
  {$ENDIF}
end;

function TNNetLayer.InitUniform(Value: TNeuralFloat): TNNetLayer;
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].InitUniform(Value);
    end;
    AfterWeightUpdate();
  end;
  Result := Self;
end;

function TNNetLayer.InitLeCunUniform(Value: TNeuralFloat): TNNetLayer;
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].InitLeCunUniform(Value);
    end;
    AfterWeightUpdate();
  end;
  Result := Self;
end;

function TNNetLayer.InitHeUniform(Value: TNeuralFloat): TNNetLayer;
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].InitHeUniform(Value);
    end;
    AfterWeightUpdate();
  end;
  Result := Self;
end;

function TNNetLayer.InitHeUniformDepthwise(Value: TNeuralFloat): TNNetLayer;
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].InitHeUniformDepthwise(Value);
    end;
    AfterWeightUpdate();
  end;
  Result := Self;
end;

function TNNetLayer.InitHeGaussian(Value: TNeuralFloat): TNNetLayer;
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].InitHeGaussian(Value);
    end;
    AfterWeightUpdate();
  end;
  Result := Self;
end;

function TNNetLayer.InitHeGaussianDepthwise(Value: TNeuralFloat): TNNetLayer;
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].InitHeGaussianDepthwise(Value);
    end;
    AfterWeightUpdate();
  end;
  Result := Self;
end;

function TNNetLayer.InitGlorotBengioUniform(Value: TNeuralFloat): TNNetLayer;
var
  Cnt: integer;
  MulAux: Single;
begin
  // This implementation is inspired on:
  // Understanding the difficulty of training deep feedforward neural networks
  // Xavier Glorot, Yoshua Bengio ; Proceedings of the Thirteenth International
  // Conference on Artificial Intelligence and Statistics, PMLR 9:249-256, 2010.
  // http://proceedings.mlr.press/v9/glorot10a.html
  if (FNeurons.Count > 0) then
  begin
    InitUniform(Value);
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      MulAux := Sqrt(6/(FNeurons[Cnt].Weights.Size + FNeurons.Count));
      FNeurons[Cnt].Weights.Mul( MulAux );
    end;
    AfterWeightUpdate();
  end;
  Result := Self;
end;

function TNNetLayer.InitSELU(Value: TNeuralFloat): TNNetLayer;
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].InitSELU(Value);
    end;
    AfterWeightUpdate();
  end;
  Result := Self;
end;

procedure TNNetLayer.InitDefault();
begin
  InitGlorotBengioUniform();
end;

procedure TNNetLayer.Fill(value: TNeuralFloat);
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].Fill(Value);
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.ClearDeltas();
var
  Cnt: integer;
begin
  if (FNeurons.Count > 0) then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].ClearDelta();
    end;
  end;
end;

procedure TNNetLayer.AddNeurons(NeuronNum: integer);
var
  I: integer;
begin
  for I := 1 to NeuronNum do
  begin
    FNeurons.Add(TNNetNeuron.Create());
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.AddMissingNeurons(NeuronNum: integer);
begin
  if FNeurons.Count < NeuronNum then
  begin
    AddNeurons(NeuronNum - FNeurons.Count);
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.SetNumWeightsForAllNeurons(NumWeights: integer);
var
  Cnt: integer;
begin
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].Weights.ReSize(NumWeights,1,1);
      FNeurons[Cnt].BackInertia.ReSize(NumWeights,1,1);
      FNeurons[Cnt].Delta.ReSize(NumWeights,1,1);
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.SetNumWeightsForAllNeurons(x, y, d: integer);
var
  Cnt: integer;
begin
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].Weights.ReSize(x,y,d);
      FNeurons[Cnt].BackInertia.ReSize(x,y,d);
      FNeurons[Cnt].Delta.ReSize(x,y,d);
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.SetNumWeightsForAllNeurons(Origin: TNNetVolume);
begin
  SetNumWeightsForAllNeurons(Origin.SizeX, Origin.SizeY, Origin.Depth);
end;

function TNNetLayer.GetMaxWeight(): TNeuralFloat;
begin
  Result := FNeurons.GetMaxWeight();
end;

function TNNetLayer.GetMaxAbsWeight(): TNeuralFloat;
begin
  Result := FNeurons.GetMaxAbsWeight();
end;

function TNNetLayer.GetMinWeight(): TNeuralFloat;
begin
  Result := FNeurons.GetMinWeight();
end;

function TNNetLayer.GetMaxDelta(): TNeuralFloat;
var
  Cnt: integer;
  MaxValue: TNeuralFloat;
begin
  if FNeurons.Count > 0 then
  begin
    Result := FNeurons[0].Delta.GetMax();
    if FNeurons.Count > 1 then
    begin
      for Cnt := 0 to FNeurons.Count-1 do
      begin
        MaxValue := FNeurons[Cnt].Delta.GetMax();
        if MaxValue > Result then Result := MaxValue;
      end;
    end;
  end
  else
  begin
    Result := 0;
  end;
end;

function TNNetLayer.GetMaxAbsoluteDelta(): TNeuralFloat;
var
  Cnt: integer;
  MaxValue: TNeuralFloat;
begin
  if FNeurons.Count > 0 then
  begin
    Result := FNeurons[0].Delta.GetMaxAbs();
    if FNeurons.Count > 1 then
    begin
      for Cnt := 0 to FNeurons.Count-1 do
      begin
        MaxValue := FNeurons[Cnt].Delta.GetMaxAbs();
        if MaxValue > Result then Result := MaxValue;
      end;
    end;
  end
  else
  begin
    Result := 0;
  end;
end;

function TNNetLayer.GetMinDelta(): TNeuralFloat;
var
  Cnt: integer;
  MinValue: TNeuralFloat;
begin
  if FNeurons.Count > 0 then
  begin
    Result := FNeurons[0].Delta.GetMin();
    if FNeurons.Count > 1 then
    begin
      for Cnt := 0 to FNeurons.Count-1 do
      begin
        MinValue := FNeurons[Cnt].Delta.GetMin();
        if MinValue < Result then Result := MinValue;
      end;
    end;
  end
  else
  begin
    Result := 0;
  end;
end;

function TNNetLayer.ForceMaxAbsoluteDelta(vMax: TNeuralFloat): TNeuralFloat;
var
  Cnt: integer;
  MaxValue, MulValue: TNeuralFloat;
begin
  Result := 1;
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count - 1 do
    begin
      MaxValue := FNeurons[Cnt].Delta.GetMaxAbs();
      if MaxValue > vMax then
      begin
        MulValue := vMax/MaxValue;
        FNeurons[Cnt].Delta.Mul(MulValue);
        if MulValue < Result then Result := MulValue;
      end;
    end;
  end
end;

function TNNetLayer.ForceMaxAbsoluteWeight(vMax: TNeuralFloat): TNeuralFloat;
var
  V: TNeuralFloat;
begin
  V := Self.GetMaxAbsWeight();
  if V > vMax then
  begin
    Self.MulWeights(vMax/V);
  end;
  Result := V;
end;

procedure TNNetLayer.GetMinMaxAtDepth(pDepth: integer; var pMin, pMax: TNeuralFloat);
var
  Cnt: integer;
  localMin, localMax: TNeuralFloat;
begin
  if FNeurons.Count > 0 then
  begin
    FNeurons[0].Weights.GetMinMaxAtDepth(pDepth, localMin, localMax);
    pMin := localMin;
    pMax := localMax;
    if FNeurons.Count > 1 then
    begin
      for Cnt := 1 to FNeurons.Count-1 do
      begin
        FNeurons[Cnt].Weights.GetMinMaxAtDepth(pDepth, localMin, localMax);
        pMin := Min(pMin, localMin);
        pMax := Max(pMax, localMax);
      end;
    end;
  end
end;

function TNNetLayer.GetWeightSum(): TNeuralFloat;
var
  Cnt: integer;
begin
  Result := 0;
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      Result := Result + FNeurons[Cnt].Weights.GetSum();
    end;
  end
end;

function TNNetLayer.GetBiasSum(): TNeuralFloat;
var
  Cnt: integer;
begin
  Result := 0;
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      Result := Result + FNeurons[Cnt].FBiasWeight;
    end;
  end
end;

function TNNetLayer.GetInertiaSum(): TNeuralFloat;
var
  Cnt: integer;
begin
  Result := 0;
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      Result := Result + FNeurons[Cnt].FBackInertia.GetSum() + FNeurons[Cnt].FBiasInertia;
    end;
  end
end;

function TNNetLayer.CountWeights(): integer;
var
  Cnt: integer;
begin
  Result := 0;
  if FLinkedNeurons then exit;
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      Result := Result + FNeurons[Cnt].Weights.Size;
    end;
  end
end;

function TNNetLayer.CountNeurons(): integer;
begin
  if LinkedNeurons
  then Result := 0
  else Result := FNeurons.Count;
end;

procedure TNNetLayer.MulWeights(V: TNeuralFloat);
var
  Cnt: integer;
begin
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].Weights.Mul(V);
      FNeurons[Cnt].BackInertia.Mul(V);
      {$IFDEF FPC}
      FNeurons[Cnt].FBiasWeight *= V;
      FNeurons[Cnt].FBiasInertia *= V;
      {$ELSE}
      FNeurons[Cnt].FBiasWeight := FNeurons[Cnt].FBiasWeight * V;
      FNeurons[Cnt].FBiasInertia := FNeurons[Cnt].FBiasInertia * V;
      {$ENDIF}
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.MulDeltas(V: TNeuralFloat);
var
  Cnt: integer;
begin
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].Delta.Mul(V);
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.ClearInertia();
var
  Cnt: integer;
begin
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].BackInertia.Fill(0);
      FNeurons[Cnt].FBiasInertia := 0;
    end;
  end
end;

procedure TNNetLayer.ClearTimes();
begin
  FBackwardTime := 0;
  FForwardTime  := 0;
end;

procedure TNNetLayer.AddTimes(Origin: TNNetLayer);
begin
  FBackwardTime := FBackwardTime + Origin.FBackwardTime;
  FForwardTime  := FForwardTime  + Origin.FForwardTime;
end;

procedure TNNetLayer.CopyTimes(Origin: TNNetLayer);
begin
  FBackwardTime := Origin.FBackwardTime;
  FForwardTime  := Origin.FForwardTime;
end;

procedure TNNetLayer.MulMulAddWeights(Value1, Value2: TNeuralFloat; Origin: TNNetLayer);
var
  Cnt: integer;
begin
  if FLinkedNeurons then exit;
  if Neurons.Count = Origin.Neurons.Count then
  begin
    if FNeurons.Count > 0 then
    begin
      for Cnt := 0 to FNeurons.Count-1 do
      begin
        FNeurons[Cnt].Weights.MulMulAdd(Value1, Value2, Origin.Neurons[Cnt].Weights);
        FNeurons[Cnt].BackInertia.MulMulAdd(Value1, Value2, Origin.Neurons[Cnt].BackInertia);
        FNeurons[Cnt].FBiasWeight := FNeurons[Cnt].FBiasWeight * Value1 + Origin.Neurons[Cnt].FBiasWeight * Value2;
        FNeurons[Cnt].FBiasInertia := FNeurons[Cnt].FBiasInertia * Value1 + Origin.Neurons[Cnt].FBiasInertia * Value2;
      end;
    end
  end
  else
  begin
    begin
      FErrorProc
      (
        'Error while adding neuron weights layer: '+
        IntToStr(FNeurons.Count)+' differs from '+
        IntToStr(Origin.Neurons.Count)
      );
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.SumWeights(Origin: TNNetLayer);
var
  Cnt: integer;
begin
  if FLinkedNeurons then exit;
  if Neurons.Count = Origin.Neurons.Count then
  begin
    if FNeurons.Count > 0 then
    begin
      for Cnt := 0 to FNeurons.Count-1 do
      begin
        FNeurons[Cnt].Weights.Add(Origin.Neurons[Cnt].Weights);
        FNeurons[Cnt].BackInertia.Add(Origin.Neurons[Cnt].BackInertia);
        {$IFDEF FPC}
        FNeurons[Cnt].FBiasWeight += Origin.Neurons[Cnt].FBiasWeight;
        FNeurons[Cnt].FBiasInertia += Origin.Neurons[Cnt].FBiasInertia;
        {$ELSE}
        FNeurons[Cnt].FBiasWeight := FNeurons[Cnt].FBiasWeight + Origin.Neurons[Cnt].FBiasWeight;
        FNeurons[Cnt].FBiasInertia := FNeurons[Cnt].FBiasInertia + Origin.Neurons[Cnt].FBiasInertia;
        {$ENDIF}
      end;
    end
  end
  else
  begin
    begin
      FErrorProc
      (
        'Error while adding neuron weights layer: '+
        IntToStr(FNeurons.Count)+' differs from '+
        IntToStr(Origin.Neurons.Count)
      );
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.SumDeltas(Origin: TNNetLayer);
begin
  if FLinkedNeurons then exit;
  if Neurons.Count = Origin.Neurons.Count then
  begin
    SumDeltasNoChecks(Origin);
  end
  else
  begin
    begin
      FErrorProc
      (
        'Error while adding neuron deltas layer: '+
        IntToStr(FNeurons.Count)+' differs from '+
        IntToStr(Origin.Neurons.Count)
      );
    end;
  end;
end;

procedure TNNetLayer.SumDeltasNoChecks(Origin: TNNetLayer);
var
  Cnt: integer;
  NeuronCount, NeuronCountM1: integer;
begin
  if FLinkedNeurons then exit;
  NeuronCount := Neurons.Count;
  if NeuronCount > 0 then
  begin
    NeuronCountM1 := NeuronCount - 1;
    for Cnt := 0 to NeuronCountM1 do
    begin
      FNeurons[Cnt].FDelta.Add(Origin.FNeurons[Cnt].FDelta);
      {$IFDEF FPC}
      FNeurons[Cnt].FBiasDelta += Origin.FNeurons[Cnt].FBiasDelta;
      {$ELSE}
      FNeurons[Cnt].FBiasDelta := FNeurons[Cnt].FBiasDelta + Origin.FNeurons[Cnt].FBiasDelta;
      {$ENDIF}
    end;
  end
end;

procedure TNNetLayer.CopyWeights(Origin: TNNetLayer);
var
  Cnt: integer;
begin
  if FLinkedNeurons then exit;
  if Neurons.Count = Origin.Neurons.Count then
  begin
    if FNeurons.Count > 0 then
    begin
      for Cnt := 0 to FNeurons.Count-1 do
      begin
        if FNeurons[Cnt].Weights.Size = Origin.Neurons[Cnt].Weights.Size then
        begin
          FNeurons[Cnt].Weights.CopyNoChecks(Origin.Neurons[Cnt].Weights);
          FNeurons[Cnt].BackInertia.CopyNoChecks(Origin.Neurons[Cnt].BackInertia);
          FNeurons[Cnt].FBiasWeight := Origin.Neurons[Cnt].FBiasWeight;
          FNeurons[Cnt].FBiasInertia := Origin.Neurons[Cnt].FBiasInertia;
        end
        else
        begin
          FErrorProc
          (
            'Error while copying neuron weights layer: '+
            IntToStr(FNeurons[Cnt].Weights.Size)+' neuron differs from '+
            IntToStr(Origin.Neurons[Cnt].Weights.Size)
          );
        end;
      end;
    end
  end
  else
  begin
    begin
      FErrorProc
      (
        'Error while copying neuron weights layer: '+
        IntToStr(FNeurons.Count)+' differs from '+
        IntToStr(Origin.Neurons.Count)
      );
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.ForceRangeWeights(V: TNeuralFloat);
var
  Cnt: integer;
begin
  if FLinkedNeurons then exit;
  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      FNeurons[Cnt].Weights.ForceMaxRange(V);
    end;
  end;
  AfterWeightUpdate();
end;

procedure TNNetLayer.NormalizeWeights(VMax: TNeuralFloat);
var
  MaxV: TNeuralFloat;
begin
  if FLinkedNeurons then exit;
  MaxV := GetMaxWeight();
  if MaxV > VMax then
  begin
    Self.MulWeights(VMax/MaxV);
  end;
  AfterWeightUpdate();
end;

function TNNetLayer.SaveDataToString(): string;
var
  S: TStringList;
  Cnt: integer;
begin
  S := TStringList.Create;
  S.Sorted := false;
  S.Delimiter := '[';
  S.StrictDelimiter := true;

  if FNeurons.Count > 0 then
  begin
    for Cnt := 0 to FNeurons.Count-1 do
    begin
      S.Add(FNeurons[Cnt].SaveToString());
    end;
  end;

  Result := S.DelimitedText;
  S.Free;
end;

procedure TNNetLayer.LoadDataFromString(strData: string);
var
  S: TStringList;
  Cnt: integer;
begin
  S := CreateTokenizedStringList(strData,'[');

  if S.Count = FNeurons.Count then
  begin
    if S.Count > 0 then
    begin
      for Cnt := 0 to S.Count-1 do
      begin
        FNeurons[Cnt].LoadFromString(S[Cnt]);
      end;
    end;
  end
  else
  begin
    FErrorProc
    (
      'Error while loading layer: number of neurons '+
      IntToStr(FNeurons.Count)+' differ from number of loaded neurons '+
      IntToStr(S.Count)
    );
  end;
  AfterWeightUpdate();
  S.Free;
end;

function TNNetLayer.SaveStructureToString(): string;
var
  I: integer;
begin
  Result := ClassName + ':';

  for I := Low(FStruct) to High(FStruct) do
  begin
    if I > 0 then Result := Result + ';';
    Result := Result + IntToStr(FStruct[I]);
  end;
end;

procedure TNNetLayer.SetBatchUpdate(pBatchUpdate: boolean);
begin
  FBatchUpdate := pBatchUpdate;
end;

procedure TNNetLayer.UpdateWeights();
var
  Cnt, MaxNeurons: integer;
begin
  MaxNeurons := FNeurons.Count - 1;
  if MaxNeurons >= 0 then
  begin
    for Cnt := 0 to MaxNeurons do
    begin
      FNeurons[Cnt].UpdateWeights(FInertia);
    end;
  end;
  AfterWeightUpdate();
end;

function TNNetLayer.InitBasicPatterns(): TNNetLayer;
var
  CntNeurons, MaxNeurons: integer;
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
begin
  MaxNeurons := FNeurons.Count - 1;
  if MaxNeurons >= 16 then
  begin
    MaxX := FNeurons[0].Weights.SizeX - 1;
    MaxY := FNeurons[0].Weights.SizeY - 1;
    MaxD := FNeurons[0].Weights.Depth - 1;
    for CntNeurons := 0 to MaxD do
    begin
      FNeurons[CntNeurons].Weights.Fill(0);
      FNeurons[CntNeurons].Weights.FillAtDepth(CntNeurons,0.1);
    end;
    for CntNeurons := 1 to 5 do
    begin
      FNeurons[CntNeurons + MaxD].Weights.Fill(0.1);
    end;
    for CntX := 0 to MaxX do
    begin
      for CntD := 0 to MaxD do
      begin
        FNeurons[MaxD + 1].Weights[CntX, CntX, CntD] := -0.1;
        FNeurons[MaxD + 2].Weights[CntX, MaxX - CntX, CntD] := -0.1;
        FNeurons[MaxD + 3].Weights[CntX, MaxY div 2, CntD] := -0.1;
        FNeurons[MaxD + 4].Weights[MaxX div 2, CntX, CntD] := -0.1;
        FNeurons[MaxD + 5].Weights[MaxX div 2, MaxY div 2, CntD] := -0.1;
      end;
    end;
    FNeurons[MaxD + 6].Weights.Fill(-0.1);
    FNeurons[MaxD + 7].Weights.Fill( 0.1);
    CntNeurons := MaxD + 8;

    for CntX := 0 to MaxX do
    begin
      for CntY := 0 to MaxY do
      begin
        for CntD := 0 to MaxD do
        begin
          if CntX <= CntY then
          begin
            FNeurons[CntNeurons + 0].Weights[CntX, CntY, CntD] := -0.1;
          end
          else
          begin
            FNeurons[CntNeurons + 0].Weights[CntX, CntY, CntD] := +0.1;
          end;
          if CntX <= MaxX div 2 then
          begin
            FNeurons[CntNeurons + 1].Weights[CntX, CntY, CntD] := -0.1;
          end
          else
          begin
            FNeurons[CntNeurons + 1].Weights[CntX, CntY, CntD] := +0.1;
          end;
          if CntY <= MaxY div 2 then
          begin
            FNeurons[CntNeurons + 2].Weights[CntX, CntY, CntD] := -0.1;
          end
          else
          begin
            FNeurons[CntNeurons + 2].Weights[CntX, CntY, CntD] := +0.1;
          end;
          if CntX <= MaxY - CntY then
          begin
            FNeurons[CntNeurons + 3].Weights[CntX, CntY, CntD] := -0.1;
          end
          else
          begin
            FNeurons[CntNeurons + 3].Weights[CntX, CntY, CntD] := +0.1;
          end;
        end;
      end;
    end;
  end;

  if MaxNeurons >= 55 then
  begin
    for CntNeurons := 0 to 13 do
    begin
      FNeurons[CntNeurons + 14].Weights.Copy(FNeurons[CntNeurons].Weights);
      FNeurons[CntNeurons + 14].Weights.Mul(-1);
    end;
  end;

  AfterWeightUpdate();
  Result := Self;
end;

procedure TNNetLayer.IncDepartingBranchesCnt();
begin
  Inc(FDepartingBranchesCnt);
end;

procedure TNNetLayer.ResetBackpropCallCurrCnt();
begin
  FBackPropCallCurrentCnt := 0;
  FOutputError.Fill(0);
end;

procedure TNNetLayer.AfterWeightUpdate();
begin
  // to be implemented in descending classes.
end;

{ TNNetNeuron }
constructor TNNetNeuron.Create();
begin
  inherited Create();
  FBiasWeight := 0;
  FBiasInertia := 0;
  FBiasDelta := 0;
  FWeights := TNNetVolume.Create(1,1,1);
  FBackInertia := TNNetVolume.Create(1,1,1);
  FDelta := TNNetVolume.Create(1,1,1);
end;

destructor TNNetNeuron.Destroy();
begin
  FDelta.Free;
  FBackInertia.Free;
  FWeights.Free;
  inherited Destroy();
end;

procedure TNNetNeuron.InitUniform(Value: TNeuralFloat = 1);
begin
  FWeights.InitUniform(Value);
  FBiasWeight := 0;
  FBackInertia.Fill(0);
  FDelta.Fill(0);
  FBiasInertia := 0;
  FBiasDelta := 0;
end;

procedure TNNetNeuron.InitGaussian(Value: TNeuralFloat);
begin
  FWeights.InitGaussian(Value);
  FBiasWeight := 0;
  FBackInertia.Fill(0);
  FDelta.Fill(0);
  FBiasInertia := 0;
  FBiasDelta := 0;
end;

procedure TNNetNeuron.InitLeCunUniform(Value: TNeuralFloat = 1);
var
  MulAux: Single;
begin
  // LeCun 98, Efficient Backprop
  // http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  InitUniform();
  MulAux := Value*Sqrt(2/(FWeights.Size));
  FWeights.Mul(MulAux);
end;

procedure TNNetNeuron.InitHeUniform(Value: TNeuralFloat = 1);
var
  MulAux: Single;
begin
  // This implementation is inspired on:
  // Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
  // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  // https://arxiv.org/abs/1502.01852
  InitUniform();
  MulAux := Value*Sqrt(3/(FWeights.Size));
  FWeights.Mul(MulAux);
end;

procedure TNNetNeuron.InitHeGaussian(Value: TNeuralFloat);
var
  MulAux: Single;
begin
  InitGaussian();
  MulAux := Value*Sqrt(3/(FWeights.Size));
  FWeights.Mul(MulAux);
end;

procedure TNNetNeuron.InitHeUniformDepthwise(Value: TNeuralFloat);
var
  MulAux: Single;
begin
  InitUniform();
  MulAux := Value*Sqrt(3/(FWeights.SizeX * FWeights.SizeY));
  FWeights.Mul(MulAux);
end;

procedure TNNetNeuron.InitHeGaussianDepthwise(Value: TNeuralFloat);
var
  MulAux: Single;
begin
  InitGaussian();
  MulAux := Value*Sqrt(3/(FWeights.SizeX * FWeights.SizeY));
  FWeights.Mul(MulAux);
end;

procedure TNNetNeuron.InitSELU(Value: TNeuralFloat);
begin
  InitGaussian( Value * Sqrt(1/FWeights.Size) );
end;

procedure TNNetNeuron.Fill(Value: TNeuralFloat);
begin
  FWeights.Fill(Value) ;
end;

procedure TNNetNeuron.AddInertia();
begin
  FWeights.Add(FBackInertia);
  FBiasWeight := FBiasWeight + FBiasInertia;
  {$IFDEF CheckRange}
  FWeights.ForceMaxRange(10000);
  NeuronForceRange(FBiasWeight,10000);
  {$ENDIF}
end;

// (BackInertia*Inertia) + (Delta*(1-Inertia))
procedure TNNetNeuron.UpdateWeights(Inertia:TNeuralFloat);
begin
  FBiasDelta := FBiasDelta * ( 1 - Inertia );
  FBiasInertia := FBiasInertia * Inertia;
  FBiasInertia := FBiasInertia + FBiasDelta;
  FBiasWeight := FBiasWeight + FBiasInertia;

  FBackInertia.MulMulAdd(Inertia, 1-Inertia, FDelta);
  FWeights.Add(FBackInertia);
  ClearDelta();
end;

function TNNetNeuron.SaveToString(): string;
begin
  Result := NeuralFloatToStr(FBiasWeight) + ']' + FWeights.SaveToString();
end;

procedure TNNetNeuron.LoadFromString(strData: string);
var
  S: TStringList;
begin
  S := CreateTokenizedStringList(strData,']');

  FBiasWeight := NeuralStrToFloat(S[0]);
  FWeights.LoadFromString(S[1]);
  S.Free;
end;

procedure TNNetNeuron.ClearDelta;
begin
  FDelta.Fill(0);
  FBiasDelta := 0;
end;

constructor TEasyBytePredictionViaNNet.Create(pActionByteLen,
  pStateByteLen: word; NumNeurons: integer;
  CacheSize: integer);
var
  NNetInputLayer1, NNetInputLayer2, RootLayer: TNNetLayer;
  BranchCnt: integer;
  BranchEnd: array of TNNetLayer;
begin
  inherited Create(FNN, pActionByteLen, pStateByteLen, CacheSize);
  SetLength(BranchEnd, pStateByteLen);
  FNN := TNNet.Create;
  NNetInputLayer1 := NN.AddLayer( TNNetInput.Create(pActionByteLen*8) );
  NNetInputLayer2 := NN.AddLayer( TNNetInput.Create(pStateByteLen*8) );
  RootLayer := NN.AddLayer( TNNetConcat.Create([NNetInputLayer1, NNetInputLayer2]) );
  //Experimental implementation with TNNetByteProcessing
  //NN.AddLayer( TNNetByteProcessing.Create(0, NumNeurons, 40) );
  //NN.AddLayer( TNNetSplitChannels.Create(pActionByteLen*8, pStateByteLen*8) );
  // A traditional NN - one branch for each output byte
  for BranchCnt := 0 to pStateByteLen - 1 do
  begin
    NN.AddLayerAfter( TNNetFullConnect.Create( NumNeurons ), RootLayer);
    NN.AddLayer( TNNetFullConnect.Create( NumNeurons ) );
    NN.AddLayer( TNNetFullConnect.Create( NumNeurons ) );
    NN.AddLayer( TNNetFullConnect.Create( 8 ) );
    BranchEnd[BranchCnt] := NN.GetLastLayer(); // NN.AddLayer( TNNetDigital.Create(-1, +1) );
  end;
  NN.AddLayer( TNNetConcat.Create(BranchEnd) );
  NN.SetLearningRate(0.01, 0.0);
  NN.SetL2Decay(0.0);
  NN.DebugStructure();
end;

destructor TEasyBytePredictionViaNNet.Destroy();
begin
  NN.Free;
  inherited Destroy();
end;

constructor TBytePredictionViaNNet.Create(pNN: TNNet; pActionByteLen,
  pStateByteLen: word; CacheSize: integer);
begin
  inherited Create();
  FNN := pNN;
  FActions := TNNetVolume.Create();
  FStates := TNNetVolume.Create();
  FPredictedStates := TNNetVolume.Create();
  FOutput := TNNetVolume.Create();
  SetLength(aActions, pActionByteLen);
  SetLength(aCurrentState, pStateByteLen);
  SetLength(aPredictedState, pStateByteLen);
  if (CacheSize>0)
  then FCache.Init(pActionByteLen, pStateByteLen, CacheSize)
  else FCache.Init(1, 1, 1);
  FUseCache := (CacheSize>0);
end;

destructor TBytePredictionViaNNet.Destroy();
begin
  FCache.DeInit;
  FOutput.Free;
  FActions.Free;
  FStates.Free;
  FPredictedStates.Free;
  inherited Destroy();
end;

procedure TBytePredictionViaNNet.Predict(var pActions,
  pCurrentState: array of byte; var pPredictedState: array of byte);
var
  idxCache: longint;
  Equal: boolean;
begin
  ABCopy(aActions, pActions);
  ABCopy(aCurrentState, pCurrentState);
  if FUseCache then
    idxCache := FCache.Read(pActions, pPredictedState);
  Equal := ABCmp(pActions, pCurrentState);
  if FUseCache and (idxCache <> -1) and Equal then
  begin
    FCached := True;
  end
  else
  begin
    //BytePred.Prediction(aActions, aCurrentState, pPredictedState, FRelationProbability, FVictoryIndex);
    FActions.CopyAsBits(aActions, -1, 1);
    FStates.CopyAsBits(pCurrentState, -1, 1);
    NN.Compute([FActions, FStates]);
    NN.GetOutput(FPredictedStates);
    FPredictedStates.ReadAsBits(pPredictedState);
    FCached := False;
  end;
  ABCopy(aPredictedState, pPredictedState);
end;

function TBytePredictionViaNNet.newStateFound(stateFound: array of byte): extended;
begin
  Result := ABCountDif(stateFound, aPredictedState);
  // Do we have a cached prediction
  if Not(FCached) then
  begin
    FPredictedStates.CopyAsBits(stateFound, -1, 1);
    // backpropagates only when fails
    //if Result > 0 then
    NN.Backpropagate(FPredictedStates);
    //NN.GetOutput(FOutput);
    //newStateFound := FOutput.SumDiff(FPredictedStates);
  end;
  if FUseCache and (Result = 0) then
    FCache.Include(aActions, stateFound);
end;


{$IFDEF Debug}
procedure TNNetConvolutionBase.PrepareInputForConvolution();
var
  OutputCntX, OutputCntY: integer;
  MaxX, MaxY: integer;
begin
  if (FPointwise) then
  begin
    // There is nothing to do. YAY!
  end
  else
  begin
    MaxX := FOutput.SizeX - 1;
    MaxY := FOutput.SizeY - 1;

    FInputPrepared.ReSize(FOutput.SizeX, FOutput.SizeY, FInputCopy.Depth * FFeatureSizeX * FFeatureSizeY);

    for OutputCntY := 0 to MaxY do
    begin
      for OutputCntX := 0 to MaxX do
      begin
        PrepareInputForConvolution(OutputCntX, OutputCntY);
      end;
    end;
  end;
end;

procedure TNNetConvolutionBase.PrepareInputForConvolution(OutputX, OutputY: integer);
var
  DepthFSize, SizeOfDepthFSize: integer;
  yCount: integer;
  InputX: integer;
begin
  InputX := OutputX * FStride;
  DepthFSize := FInputCopy.Depth * FFeatureSizeX;
  SizeOfDepthFSize := DepthFSize * SizeOf(TNeuralFloat);

  for yCount := 0 to FFeatureSizeY - 1 do
  begin
    (*
    fi00 := FInputCopy.GetRawPos(InputX, OutputY*FStride + yCount , 0);
    lo00 := FInputPrepared.GetRawPos(OutputX, OutputY, DepthFSize * yCount);
    Move(FInputCopy.FData[fi00], FInputPrepared.FData[lo00], SizeOfDepthFSize);
    *)
    Move
    (
      FInputCopy.FData[FInputCopy.GetRawPos(InputX, OutputY * FStride + yCount)],
      FInputPrepared.FData[FInputPrepared.GetRawPos(OutputX, OutputY, DepthFSize * yCount)],
      SizeOfDepthFSize
    );
  end;
end;

procedure TNNetConvolution.ComputeNeuronCPU();
var
  OutputCntX, OutputCntY: integer;
  MaxX, MaxY: integer;
  LocalSize: integer;
  LocalNeuron: TNNetNeuron;
  PtrA: TNeuralFloatArrPtr;
  NeuronIdx: integer;
  MaxNeurons: integer;
  ConvResult: TNeuralFloatPtr;
begin
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;
  MaxNeurons := FNeurons.Count - 1;
  LocalSize := FFeatureSizeX*FFeatureSizeY*FInputCopy.Depth;

  OutputCntX := 0;
  while OutputCntX <= MaxX do
  begin
    OutputCntY := 0;
    while OutputCntY <= MaxY do
    begin
      PtrA := FInputPrepared.GetRawPtr(OutputCntX, OutputCntY);
      ConvResult := FOutputRaw.GetRawPtr(OutputCntX, OutputCntY);
      for NeuronIdx := 0 to MaxNeurons do
      begin
        LocalNeuron := FArrNeurons[NeuronIdx];

        ConvResult^ :=
          LocalNeuron.Weights.DotProduct(PtrA, LocalNeuron.Weights.DataPtr, LocalSize) +
          LocalNeuron.FBiasWeight;

        Inc(ConvResult);
      end;
      Inc(OutputCntY);
    end;
    Inc(OutputCntX);
  end;
end;

procedure TNNetConvolution.AddBiasToRawResult();
var
  OutputCntX, OutputCntY: integer;
  MaxX, MaxY: integer;
  NeuronIdx: integer;
  MaxNeurons: integer;
  ConvResult: TNeuralFloatPtr;
begin
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;
  MaxNeurons := FNeurons.Count - 1;

  for OutputCntX := 0 to MaxX do
  begin
    for OutputCntY := 0 to MaxY do
    begin
      ConvResult := FOutputRaw.GetRawPtr(OutputCntX, OutputCntY);
      for NeuronIdx := 0 to MaxNeurons do
      begin
        {$IFDEF FPC}
        ConvResult^ += FArrNeurons[NeuronIdx].FBiasWeight;
        {$ELSE}
        ConvResult^ := ConvResult^ + FArrNeurons[NeuronIdx].FBiasWeight;
        {$ENDIF}
        Inc(ConvResult);
      end;
    end;
  end;
end;

procedure TNNetConvolution.ComputeNeuron(NeuronIdx: integer);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
begin
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;

  for CntX := 0 to MaxX do
  begin
    for CntY := 0 to MaxY do
    begin
      ComputeNeuronAtOutputPos(NeuronIdx, CntX, CntY);
    end;
  end;
end;

(*
procedure TNNetConvolution.ComputeNeuronFromResult(NeuronIdx: integer);
var
  OutputCntX, OutputCntY: integer;
  MaxX, MaxY: integer;
  Sum: TNeuralFloat;
begin
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;

  OutputCntX := 0;
  while OutputCntX <= MaxX do
  begin
    OutputCntY := 0;
    while OutputCntY <= MaxY do
    begin
      //Sum := FDotProductResult.FData[ FOutputRaw.GetRawPos(OutputCntX, OutputCntY, NeuronIdx) ];
      Sum := FDotProductResult[OutputCntX, OutputCntY, NeuronIdx];

      FOutputRaw[OutputCntX, OutputCntY, NeuronIdx] := Sum;
      FOutput[OutputCntX, OutputCntY, NeuronIdx] := FActivationFn(Sum);

      Inc(OutputCntY);
    end;
    Inc(OutputCntX);
  end;
end;
*)
(*
procedure TNNetConvolution.ComputeNeuronFastAVX32(NeuronIdx: integer);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  LocalSize: integer;
  FloatLocalSize: TNeuralFloat;
  Total: TNeuralFloat;

  LocalW: TNNetVolume;
  PtrA, PtrB: TNeuralFloatArrPtr;

  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  MaxX := FOutput.SizeX - 1;
  MaxY := FOutput.SizeY - 1;
  LocalW := FNeurons[NeuronIdx].Weights;
  LocalSize := FFeatureSize*FFeatureSize*FInputCopy.Depth;
  FloatLocalSize := LocalSize;
  PtrB := LocalW.GetRawPtr(0, 0, 0);

  localNumElements := (LocalSize div 4) * 4;
  MissedElements := LocalSize - localNumElements;

  for CntX := 0 to MaxX do
  begin
    for CntY := 0 to MaxY do
    begin
      PtrA := FInputPrepared.GetRawPtr(CntX, CntY, 0);

      if localNumElements > 0 then
      begin
      asm
      mov ecx, localNumElements
      mov eax, PtrB
      mov edx, PtrA
      vxorps ymm0, ymm0, ymm0

      push ecx
      shr ecx,4  // number of large iterations = number of elements / 16
      jz @SkipLargeAddLoop
      vxorps ymm1, ymm1, ymm1
    @LargeAddLoop:

      vmovups ymm2, [eax]
      vmovups ymm6, [edx]

      vmovups ymm3, [eax+32]
      vmovups ymm7, [edx+32]

      {$IFDEF AVX2}
      vfmadd231ps ymm0, ymm2, ymm6
      vfmadd231ps ymm1, ymm3, ymm7
      {$ELSE}
      vmulps  ymm2, ymm2, ymm6
      vmulps  ymm3, ymm3, ymm7

      vaddps  ymm0, ymm0, ymm2
      vaddps  ymm1, ymm1, ymm3
      {$ENDIF}

      add eax, 64
      add edx, 64
      dec ecx
      jnz @LargeAddLoop

      vaddps ymm0, ymm0, ymm1
      VEXTRACTF128 xmm2, ymm0, 1

      vzeroupper
      addps xmm0, xmm2

    @SkipLargeAddLoop:
      pop ecx
      and ecx,$0000000F
      jz @EndAdd
      shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
    @SmallAddLoop:
      vzeroupper

      movups xmm2, [eax]
      movups xmm3, [edx]
      mulps xmm2, xmm3
      addps xmm0, xmm2

      add eax, 16
      add edx, 16
      dec ecx
      jnz @SmallAddLoop

    @EndAdd:
      // Sums all elements of xmm0 into the first position
      HADDPS xmm0,xmm0
      HADDPS xmm0,xmm0

      movups vRes, xmm0
      end
      [
        'EAX', 'ECX', 'EDX',
        'xmm0', 'xmm1', 'xmm2', 'xmm3',
        'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm6', 'ymm7'
      ];

        Total :=
          vRes[0] +
          FNeurons[NeuronIdx].FBiasWeight;
      end else
      begin
        Total := FNeurons[NeuronIdx].FBiasWeight;
      end;

      if MissedElements>0 then
      begin
        if MissedElements = 1
        then Total += PtrA^[localNumElements] * PtrB^[localNumElements]
        else if MissedElements = 2
        then Total +=
               PtrA^[localNumElements] * PtrB^[localNumElements] +
               PtrA^[localNumElements+1] * PtrB^[localNumElements+1]
        else Total +=
               PtrA^[localNumElements] * PtrB^[localNumElements] +
               PtrA^[localNumElements+1] * PtrB^[localNumElements+1] +
               PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
      end;

      Total :=
        LocalW.DotProduct(PtrA, PtrB, LocalSize) +
        FNeurons[NeuronIdx].FBiasWeight;

      if (FBalanceOutput) then
      begin
        Output[CntX, CntY, NeuronIdx] := FActivationFn( Total / (FloatLocalSize) );
      end
      else
      begin
        Output[CntX, CntY, NeuronIdx] := FActivationFn( Total );
      end;
    end;
  end;
end;
*)

procedure TNNetConvolution.ComputeNeuronAtOutputPos(NeuronIdx, x, y: integer);
var
  Total: TNeuralFloat;
begin
  Total := ComputeNeuronAtPreparedInput(NeuronIdx, x, y);

  Total := Total + FNeurons[NeuronIdx].FBiasWeight;

  Output[x,y,NeuronIdx] := FActivationFn( Total );
end;

function TNNetConvolution.ComputeNeuronAtPreparedInput(NeuronIdx, x, y: integer
  ):TNeuralFloat;
var
  LocalW: TNNetVolume;
  fi00, lo00: pointer;
begin
  LocalW := FNeurons[NeuronIdx].Weights;

  {$IFDEF Debug}
  if (FInputPrepared.Depth <> LocalW.Size) then
  begin
    WriteLn
    (
      'Prepared input Depth doesn''t match weight size at neuron ',NeuronIdx,':',
      'Weight Size:', LocalW.Size,
      'Depth:', FInputPrepared.Depth
    );
  end;
  {$ENDIF}

  fi00 := FInputPrepared.GetRawPtr(x, y , 0);
  lo00 := LocalW.GetRawPtr(0, 0, 0);

  Result := LocalW.DotProduct(fi00, lo00, LocalW.Size);
end;

function TNNetConvolution.ComputeNeuronAtOutputPos3(NeuronIdx, x, y: integer): TNeuralFloat;
var
  CntD: integer;
  MaxD: integer;
  x1, x2, y1, y2: integer;
  Total: TNeuralFloat;
  LocalW: TNNetVolume;
  fi00, fi10, fi20,
  fi01, fi11, fi21,
  fi02, fi12, fi22: integer;
  lo00, lo10, lo20,
  lo01, lo11, lo21,
  lo02, lo12, lo22: integer;
  DepthX: integer;
begin
  Total := 0;
  LocalW := FNeurons[NeuronIdx].Weights;

  x1 := x + 1;
  x2 := x + 2;
  y1 := y + 1;
  y2 := y + 2;
  fi00 := FInputCopy.GetRawPos(x ,y ,0);
  fi01 := FInputCopy.GetRawPos(x ,y1,0);
  fi02 := FInputCopy.GetRawPos(x ,y2,0);
  lo00 := LocalW.GetRawPos(0,0,0);
  lo01 := LocalW.GetRawPos(0,1,0);
  lo02 := LocalW.GetRawPos(0,2,0);

  DepthX := FInputCopy.Depth * LocalW.SizeX;

  if DepthX >= 220 then
  begin

    Total :=
      LocalW.DotProduct(addr(FInputCopy.FData[fi00]), addr(LocalW.FData[lo00]), DepthX) +
      LocalW.DotProduct(addr(FInputCopy.FData[fi01]), addr(LocalW.FData[lo01]), DepthX) +
      LocalW.DotProduct(addr(FInputCopy.FData[fi02]), addr(LocalW.FData[lo02]), DepthX);

  end else
  begin
    MaxD := FInputCopy.Depth - 1;

    fi10 := FInputCopy.GetRawPos(x1,y ,0);
    fi20 := FInputCopy.GetRawPos(x2,y ,0);

    fi11 := FInputCopy.GetRawPos(x1,y1,0);
    fi21 := FInputCopy.GetRawPos(x2,y1,0);

    fi12 := FInputCopy.GetRawPos(x1,y2,0);
    fi22 := FInputCopy.GetRawPos(x2,y2,0);

    lo10 := LocalW.GetRawPos(1,0,0);
    lo20 := LocalW.GetRawPos(2,0,0);

    lo11 := LocalW.GetRawPos(1,1,0);
    lo21 := LocalW.GetRawPos(2,1,0);

    lo12 := LocalW.GetRawPos(1,2,0);
    lo22 := LocalW.GetRawPos(2,2,0);

    for CntD := 0 to MaxD do
    begin
      Total := Total +
        FInputCopy.FData[fi00 + CntD] * LocalW.FData[lo00 + CntD] +
        FInputCopy.FData[fi10 + CntD] * LocalW.FData[lo10 + CntD] +
        FInputCopy.FData[fi20 + CntD] * LocalW.FData[lo20 + CntD] +
        FInputCopy.FData[fi01 + CntD] * LocalW.FData[lo01 + CntD] +
        FInputCopy.FData[fi11 + CntD] * LocalW.FData[lo11 + CntD] +
        FInputCopy.FData[fi21 + CntD] * LocalW.FData[lo21 + CntD] +
        FInputCopy.FData[fi02 + CntD] * LocalW.FData[lo02 + CntD] +
        FInputCopy.FData[fi12 + CntD] * LocalW.FData[lo12 + CntD] +
        FInputCopy.FData[fi22 + CntD] * LocalW.FData[lo22 + CntD];
    end;
  end;

  Result := Total;
end;

function TNNetConvolution.ComputeNeuronAtOutputPos3D3(NeuronIdx, x, y: integer
  ): TNeuralFloat;
var
  x1, x2, y1, y2: integer;
  LocalW: TNNetVolume;
  fi00, fi10, fi20,
  fi01, fi11, fi21,
  fi02, fi12, fi22: integer;
  lo00, lo10, lo20,
  lo01, lo11, lo21,
  lo02, lo12, lo22: integer;

begin
  x1 := x + 1;
  x2 := x + 2;
  y1 := y + 1;
  y2 := y + 2;
  LocalW := FNeurons[NeuronIdx].Weights;

  fi00 := FInputCopy.GetRawPos(x ,y ,0);
  fi10 := FInputCopy.GetRawPos(x1,y ,0);
  fi20 := FInputCopy.GetRawPos(x2,y ,0);

  fi01 := FInputCopy.GetRawPos(x ,y1,0);
  fi11 := FInputCopy.GetRawPos(x1,y1,0);
  fi21 := FInputCopy.GetRawPos(x2,y1,0);

  fi02 := FInputCopy.GetRawPos(x ,y2,0);
  fi12 := FInputCopy.GetRawPos(x1,y2,0);
  fi22 := FInputCopy.GetRawPos(x2,y2,0);

  lo00 := LocalW.GetRawPos(0,0,0);
  lo10 := LocalW.GetRawPos(1,0,0);
  lo20 := LocalW.GetRawPos(2,0,0);

  lo01 := LocalW.GetRawPos(0,1,0);
  lo11 := LocalW.GetRawPos(1,1,0);
  lo21 := LocalW.GetRawPos(2,1,0);

  lo02 := LocalW.GetRawPos(0,2,0);
  lo12 := LocalW.GetRawPos(1,2,0);
  lo22 := LocalW.GetRawPos(2,2,0);

  begin
    Result :=
      FInputCopy.FData[fi00] * LocalW.FData[lo00] +
      FInputCopy.FData[fi10] * LocalW.FData[lo10] +
      FInputCopy.FData[fi20] * LocalW.FData[lo20] +
      FInputCopy.FData[fi01] * LocalW.FData[lo01] +
      FInputCopy.FData[fi11] * LocalW.FData[lo11] +
      FInputCopy.FData[fi21] * LocalW.FData[lo21] +
      FInputCopy.FData[fi02] * LocalW.FData[lo02] +
      FInputCopy.FData[fi12] * LocalW.FData[lo12] +
      FInputCopy.FData[fi22] * LocalW.FData[lo22] +

      FInputCopy.FData[fi00+1] * LocalW.FData[lo00+1] +
      FInputCopy.FData[fi10+1] * LocalW.FData[lo10+1] +
      FInputCopy.FData[fi20+1] * LocalW.FData[lo20+1] +
      FInputCopy.FData[fi01+1] * LocalW.FData[lo01+1] +
      FInputCopy.FData[fi11+1] * LocalW.FData[lo11+1] +
      FInputCopy.FData[fi21+1] * LocalW.FData[lo21+1] +
      FInputCopy.FData[fi02+1] * LocalW.FData[lo02+1] +
      FInputCopy.FData[fi12+1] * LocalW.FData[lo12+1] +
      FInputCopy.FData[fi22+1] * LocalW.FData[lo22+1] +

      FInputCopy.FData[fi00+2] * LocalW.FData[lo00+2] +
      FInputCopy.FData[fi10+2] * LocalW.FData[lo10+2] +
      FInputCopy.FData[fi20+2] * LocalW.FData[lo20+2] +
      FInputCopy.FData[fi01+2] * LocalW.FData[lo01+2] +
      FInputCopy.FData[fi11+2] * LocalW.FData[lo11+2] +
      FInputCopy.FData[fi21+2] * LocalW.FData[lo21+2] +
      FInputCopy.FData[fi02+2] * LocalW.FData[lo02+2] +
      FInputCopy.FData[fi12+2] * LocalW.FData[lo12+2] +
      FInputCopy.FData[fi22+2] * LocalW.FData[lo22+2];
  end;

end;

function TNNetConvolution.ComputeNeuronAtOutputPosDefault(NeuronIdx, x,
  y: integer): TNeuralFloat;
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  Total: TNeuralFloat;
  LocalW: TNNetVolume;
begin
  // This is the default behaviour
  Total := 0;
  MaxX := x + FFeatureSizeX - 1;
  MaxY := y + FFeatureSizeY - 1;
  MaxD := FInputCopy.Depth - 1;
  LocalW := FNeurons[NeuronIdx].Weights;

  for CntX := x to MaxX do
  begin
    for CntY := y to MaxY do
    begin
      for CntD := 0 to MaxD do
      begin
        Total := Total + FInputCopy[CntX,CntY,CntD] * LocalW[CntX-x, CntY-y, CntD];
      end;
    end;
  end;

  Result := Total;
end;

function TNNetConvolution.ComputeNeuronAtOutputPosDefaultFast(NeuronIdx, x,
  y: integer): TNeuralFloat;
var
  Total: TNeuralFloat;
  LocalW: TNNetVolume;
  fi00, lo00: integer;
  DepthX: integer;
  yCount: integer;
begin
  LocalW := FNeurons[NeuronIdx].Weights;
  Total := 0;

  DepthX := FInputCopy.Depth * LocalW.SizeX;

  for yCount := 0 to LocalW.SizeY - 1 do
  begin
    fi00 := FInputCopy.GetRawPos(x, y + yCount, 0);
    lo00 := LocalW.GetRawPos(0, yCount, 0);
    Total := Total +
      LocalW.DotProduct(addr(FInputCopy.FData[fi00]), addr(LocalW.FData[lo00]), DepthX);
  end;
  Result := Total;
end;

{$ENDIF}

{$IFNDEF FPC}
{ TNNetNeuronList }
function TNNetNeuronList.GetItem(Index: Integer): TNNetNeuron;
begin
  Result := TNNetNeuron(Get(Index));
end;

procedure TNNetNeuronList.SetItem(Index: Integer; AObject: TNNetNeuron);
begin
  Put(Index,AObject);
end;

{ TNNetLayerList }
function TNNetLayerList.GetItem(Index: Integer): TNNetLayer;
begin
  Result := TNNetLayer(Get(Index));
end;

procedure TNNetLayerList.SetItem(Index: Integer; AObject: TNNetLayer);
begin
  Put(Index,AObject);
end;

function TNNetDataParallelism.GetItem(Index: Integer): TNNet;
begin
  Result := TNNet(Get(Index));
end;

procedure TNNetDataParallelism.SetItem(Index: Integer; AObject: TNNet);
begin
  Put(Index,AObject);
end;
{$ENDIF}

end.

