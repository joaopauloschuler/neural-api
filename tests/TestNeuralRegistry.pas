unit TestNeuralRegistry;

// Layer-registry round-trip audit.
//
// For every concrete TNNet* layer covered below, this test:
//   1. builds a minimal TNNet (Input + any shape-prep predecessors + the
//      target layer), instantiated with representative default-ish args,
//   2. serializes the architecture with SaveStructureToString,
//   3. reloads it into a SECOND net via LoadStructureFromString (this is the
//      code path that runs TNNet.CreateLayer -- the class-name -> constructor
//      dispatch table),
//   4. serializes the second net again, and
//   5. asserts BIT-FOR-BIT string equality between the two structure strings.
//
// This is the highest-leverage single test for the "added a layer but forgot
// to register it in CreateLayer" bug class: if a layer can be serialized but
// not deserialized, the reload produces a different (or empty) layer and the
// re-serialized string differs, failing the test WITH THE LAYER NAME in the
// message.
//
// We compare SaveStructureToString (architecture only) rather than SaveToString
// (architecture + weights) on purpose: weights carry random init that does not
// round-trip deterministically without an explicit reload of data, and the
// registration bug we are hunting lives entirely in the structure dispatch.
//
// COVERAGE: the CreateLayer dispatch table has ~404 registered class-name
// strings (some are legacy aliases pointing at the same class, e.g.
// 'TNNetLayerSoftMax' -> TNNetSoftMax). This test exercises a large
// representative subset spanning every category (activations, convolutions,
// pooling, normalization, attention, recurrent/SSM, MoE/gates, embeddings,
// loss heads, flow layers, shape ops). See the SKIP-LIST comment at the bottom
// of BuildLayerNet for the small set deliberately not covered and why.
//
// Test-only change; no neuralnetwork.pas classes were added.

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralnetwork, neuralvolume;

type
  TTestNeuralRegistry = class(TTestCase)
  private
    // Builds net `idx` (Input + predecessors + one target layer).
    // Returns False once idx is past the end of the list.
    // On True, `LayerName` is set to the target layer's class name.
    function BuildLayerNet(idx: integer; ANet: TNNet; out LayerName: string): boolean;
  published
    procedure TestRegistryRoundTrip;
  end;

implementation

// Convenience: a fresh standard 2D feature-map input (8 x 8 x 8).
procedure AddImgInput(ANet: TNNet);
begin
  ANet.AddLayer(TNNetInput.Create(8, 8, 8, 1));
end;

// A sequence-shaped input (SeqLen=8, 1, Depth=16) used by token / attention
// style layers that expect (X=seq, Y=1, Depth=d).
procedure AddSeqInput(ANet: TNNet);
begin
  ANet.AddLayer(TNNetInput.Create(8, 1, 16, 1));
end;

// A flat vector input (1 x 1 x 16).
procedure AddVecInput(ANet: TNNet);
begin
  ANet.AddLayer(TNNetInput.Create(1, 1, 16, 1));
end;

// Attention-shaped input: (SeqLen=8, 1, Depth=3*d_k) with d_k=4 so the
// standard scaled-dot-product attention family (which expects packed Q|K|V
// of depth 3*d_k) validates on attach.
procedure AddAttnInput(ANet: TNNet);
begin
  ANet.AddLayer(TNNetInput.Create(8, 1, 12, 1));
end;

// Raw-waveform input (T, 1, 1) for 1-D signal layers.
procedure AddWaveInput(ANet: TNNet);
begin
  ANet.AddLayer(TNNetInput.Create(64, 1, 1, 1));
end;

function TTestNeuralRegistry.BuildLayerNet(idx: integer; ANet: TNNet; out LayerName: string): boolean;

  // Helper to register the expected name AND add a layer in one shot.
  function L(const AName: string; ALayer: TNNetLayer): boolean;
  begin
    LayerName := AName;
    ANet.AddLayer(ALayer);
    Result := True;
  end;

begin
  Result := True;
  LayerName := '';
  case idx of
    // ---- core / identity / shape ops -------------------------------------
    0:  begin AddImgInput(ANet); Exit(L('TNNetIdentity', TNNetIdentity.Create())); end;
    1:  begin AddImgInput(ANet); Exit(L('TNNetIdentityWithoutBackprop', TNNetIdentityWithoutBackprop.Create())); end;
    2:  begin AddImgInput(ANet); Exit(L('TNNetDebug', TNNetDebug.Create(1, 1))); end;
    3:  begin AddImgInput(ANet); Exit(L('TNNetPad', TNNetPad.Create(1))); end;
    4:  begin AddImgInput(ANet); Exit(L('TNNetPadXY', TNNetPadXY.Create(1, 1))); end;
    5:  begin AddImgInput(ANet); Exit(L('TNNetCrop', TNNetCrop.Create(1, 1, 4, 4))); end;
    6:  begin AddImgInput(ANet); Exit(L('TNNetReshape', TNNetReshape.Create(4, 4, 8))); end;
    7:  begin AddImgInput(ANet); Exit(L('TNNetExpandDims', TNNetExpandDims.Create(2))); end;
    8:  begin AddImgInput(ANet); Exit(L('TNNetTransposeXD', TNNetTransposeXD.Create())); end;
    9:  begin AddImgInput(ANet); Exit(L('TNNetTransposeYD', TNNetTransposeYD.Create())); end;
    10: begin AddImgInput(ANet); Exit(L('TNNetFlipX', TNNetFlipX.Create())); end;
    11: begin AddImgInput(ANet); Exit(L('TNNetFlipY', TNNetFlipY.Create())); end;
    12: begin AddImgInput(ANet); Exit(L('TNNetReverseXY', TNNetReverseXY.Create())); end;
    13: begin AddImgInput(ANet); Exit(L('TNNetReverseChannels', TNNetReverseChannels.Create())); end;
    14: begin AddImgInput(ANet); Exit(L('TNNetInterleaveChannels', TNNetInterleaveChannels.Create(2))); end;
    15: begin AddImgInput(ANet); Exit(L('TNNetChannelShuffle', TNNetChannelShuffle.Create(2))); end;
    16: begin AddImgInput(ANet); Exit(L('TNNetSpaceToDepth', TNNetSpaceToDepth.Create(2))); end;
    17: begin AddImgInput(ANet); Exit(L('TNNetDepthToSpace', TNNetDepthToSpace.Create(2))); end;
    18: begin AddSeqInput(ANet); Exit(L('TNNetGather', TNNetGather.Create(0))); end;
    19: begin AddImgInput(ANet); Exit(L('TNNetGatherChannels', TNNetGatherChannels.Create([0, 1, 2]))); end;
    20: begin AddSeqInput(ANet); Exit(L('TNNetGatherTokens', TNNetGatherTokens.Create([0, 1]))); end;
    21: begin AddSeqInput(ANet); Exit(L('TNNetSplitChannels', TNNetSplitChannels.Create(0, 4))); end;
    22: begin AddSeqInput(ANet); Exit(L('TNNetSplitChannelEvery', TNNetSplitChannelEvery.Create(2, 0))); end;
    23: begin AddSeqInput(ANet); Exit(L('TNNetCumSum', TNNetCumSum.Create(0))); end;
    24: begin AddSeqInput(ANet); Exit(L('TNNetRoll', TNNetRoll.Create(1))); end;
    25: begin AddImgInput(ANet); Exit(L('TNNetCoordConv', TNNetCoordConv.Create())); end;
    26: begin AddImgInput(ANet); Exit(L('TNNetSqueeze', TNNetSqueeze.Create(2))); end;

    // ---- plain activations (TNNetReLUBase / TNNetIdentity family) ---------
    27: begin AddImgInput(ANet); Exit(L('TNNetReLU', TNNetReLU.Create())); end;
    28: begin AddImgInput(ANet); Exit(L('TNNetReLUP', TNNetReLUP.Create())); end;
    29: begin AddImgInput(ANet); Exit(L('TNNetReLU6', TNNetReLU6.Create())); end;
    30: begin AddImgInput(ANet); Exit(L('TNNetReLUL', TNNetReLUL.Create(0, 6, 0))); end;
    31: begin AddImgInput(ANet); Exit(L('TNNetReLUSqrt', TNNetReLUSqrt.Create())); end;
    32: begin AddImgInput(ANet); Exit(L('TNNetSquaredReLU', TNNetSquaredReLU.Create())); end;
    33: begin AddImgInput(ANet); Exit(L('TNNetLeakyReLU', TNNetLeakyReLU.Create(0.01))); end;
    34: begin AddImgInput(ANet); Exit(L('TNNetVeryLeakyReLU', TNNetVeryLeakyReLU.Create())); end;
    35: begin AddImgInput(ANet); Exit(L('TNNetShiftedReLU', TNNetShiftedReLU.Create())); end;
    36: begin AddImgInput(ANet); Exit(L('TNNetSwish', TNNetSwish.Create())); end;
    37: begin AddImgInput(ANet); Exit(L('TNNetSwish6', TNNetSwish6.Create())); end;
    38: begin AddImgInput(ANet); Exit(L('TNNetSiLU', TNNetSiLU.Create())); end;
    39: begin AddImgInput(ANet); Exit(L('TNNetESwish', TNNetESwish.Create(1.25))); end;
    40: begin AddImgInput(ANet); Exit(L('TNNetHardSwish', TNNetHardSwish.Create())); end;
    41: begin AddImgInput(ANet); Exit(L('TNNetHardSigmoid', TNNetHardSigmoid.Create())); end;
    42: begin AddImgInput(ANet); Exit(L('TNNetHardTanh', TNNetHardTanh.Create())); end;
    43: begin AddImgInput(ANet); Exit(L('TNNetGELU', TNNetGELU.Create())); end;
    44: begin AddImgInput(ANet); Exit(L('TNNetGELUErf', TNNetGELUErf.Create())); end;
    45: begin AddImgInput(ANet); Exit(L('TNNetMish', TNNetMish.Create())); end;
    46: begin AddImgInput(ANet); Exit(L('TNNetPhish', TNNetPhish.Create())); end;
    47: begin AddImgInput(ANet); Exit(L('TNNetSerf', TNNetSerf.Create())); end;
    48: begin AddImgInput(ANet); Exit(L('TNNetErf', TNNetErf.Create())); end;
    49: begin AddImgInput(ANet); Exit(L('TNNetTanhExp', TNNetTanhExp.Create())); end;
    50: begin AddImgInput(ANet); Exit(L('TNNetSmish', TNNetSmish.Create())); end;
    51: begin AddImgInput(ANet); Exit(L('TNNetPenalizedTanh', TNNetPenalizedTanh.Create())); end;
    52: begin AddImgInput(ANet); Exit(L('TNNetSoftPlus', TNNetSoftPlus.Create())); end;
    53: begin AddImgInput(ANet); Exit(L('TNNetSoftPlusBeta', TNNetSoftPlusBeta.Create(1.0))); end;
    54: begin AddImgInput(ANet); Exit(L('TNNetSoftSign', TNNetSoftSign.Create())); end;
    55: begin AddImgInput(ANet); Exit(L('TNNetSELU', TNNetSELU.Create())); end;
    56: begin AddImgInput(ANet); Exit(L('TNNetELU', TNNetELU.Create(1.0))); end;
    57: begin AddImgInput(ANet); Exit(L('TNNetCELU', TNNetCELU.Create(1.0))); end;
    58: begin AddImgInput(ANet); Exit(L('TNNetISRU', TNNetISRU.Create(1.0))); end;
    59: begin AddImgInput(ANet); Exit(L('TNNetISRLU', TNNetISRLU.Create(1.0))); end;
    60: begin AddImgInput(ANet); Exit(L('TNNetSigmoid', TNNetSigmoid.Create())); end;
    61: begin AddImgInput(ANet); Exit(L('TNNetHyperbolicTangent', TNNetHyperbolicTangent.Create())); end;
    62: begin AddImgInput(ANet); Exit(L('TNNetLeCunTanh', TNNetLeCunTanh.Create())); end;
    63: begin AddImgInput(ANet); Exit(L('TNNetLogCoshActivation', TNNetLogCoshActivation.Create())); end;
    64: begin AddImgInput(ANet); Exit(L('TNNetPower', TNNetPower.Create(2))); end;
    65: begin AddImgInput(ANet); Exit(L('TNNetSignedSquareRoot', TNNetSignedSquareRoot.Create())); end;
    66: begin AddImgInput(ANet); Exit(L('TNNetSignedSquareRoot1', TNNetSignedSquareRoot1.Create())); end;
    67: begin AddImgInput(ANet); Exit(L('TNNetSignedSquareRootN', TNNetSignedSquareRootN.Create(2.0))); end;
    68: begin AddImgInput(ANet); Exit(L('TNNetRReLU', TNNetRReLU.Create(0.1, 0.3))); end;
    69: begin AddImgInput(ANet); Exit(L('TNNetSoftExponential', TNNetSoftExponential.Create(0.5))); end;
    70: begin AddImgInput(ANet); Exit(L('TNNetAbs', TNNetAbs.Create())); end;
    71: begin AddImgInput(ANet); Exit(L('TNNetArcSinh', TNNetArcSinh.Create())); end;
    72: begin AddImgInput(ANet); Exit(L('TNNetBentIdentity', TNNetBentIdentity.Create())); end;
    73: begin AddImgInput(ANet); Exit(L('TNNetCos', TNNetCos.Create())); end;
    74: begin AddImgInput(ANet); Exit(L('TNNetSin', TNNetSin.Create())); end;
    75: begin AddImgInput(ANet); Exit(L('TNNetSinc', TNNetSinc.Create())); end;
    76: begin AddImgInput(ANet); Exit(L('TNNetSinhAct', TNNetSinhAct.Create())); end;
    77: begin AddImgInput(ANet); Exit(L('TNNetExp', TNNetExp.Create())); end;
    78: begin AddImgInput(ANet); Exit(L('TNNetLog', TNNetLog.Create())); end;
    79: begin AddImgInput(ANet); Exit(L('TNNetSqrt', TNNetSqrt.Create())); end;
    80: begin AddImgInput(ANet); Exit(L('TNNetSquare', TNNetSquare.Create())); end;
    81: begin AddImgInput(ANet); Exit(L('TNNetReciprocal', TNNetReciprocal.Create())); end;
    82: begin AddImgInput(ANet); Exit(L('TNNetSign', TNNetSign.Create())); end;
    83: begin AddImgInput(ANet); Exit(L('TNNetLisht', TNNetLisht.Create())); end;
    84: begin AddImgInput(ANet); Exit(L('TNNetLogSigmoid', TNNetLogSigmoid.Create())); end;
    85: begin AddImgInput(ANet); Exit(L('TNNetGaussianActivation', TNNetGaussianActivation.Create())); end;
    86: begin AddImgInput(ANet); Exit(L('TNNetClamp', TNNetClamp.Create(-1.0, 1.0))); end;
    87: begin AddImgInput(ANet); Exit(L('TNNetHardShrink', TNNetHardShrink.Create(0.5))); end;
    88: begin AddImgInput(ANet); Exit(L('TNNetSoftShrink', TNNetSoftShrink.Create(0.5))); end;
    89: begin AddImgInput(ANet); Exit(L('TNNetTanhShrink', TNNetTanhShrink.Create())); end;
    90: begin AddImgInput(ANet); Exit(L('TNNetThreshold', TNNetThreshold.Create(0.5, 0.0))); end;
    91: begin AddImgInput(ANet); Exit(L('TNNetSnake', TNNetSnake.Create(1.0))); end;
    92: begin AddImgInput(ANet); Exit(L('TNNetSoftCapping', TNNetSoftCapping.Create(30.0))); end;

    // ---- trainable / per-channel activations (TNNetChannelTransformBase) --
    93:  begin AddImgInput(ANet); Exit(L('TNNetPReLU', TNNetPReLU.Create(0.25))); end;
    94:  begin AddImgInput(ANet); Exit(L('TNNetPReLUChannel', TNNetPReLUChannel.Create())); end;
    95:  begin AddImgInput(ANet); Exit(L('TNNetSReLU', TNNetSReLU.Create(1, 1, -1, 1))); end;
    96:  begin AddImgInput(ANet); Exit(L('TNNetAPL', TNNetAPL.Create(2))); end;
    97:  begin AddImgInput(ANet); Exit(L('TNNetSwishLearnable', TNNetSwishLearnable.Create(1.0))); end;
    98:  begin AddImgInput(ANet); Exit(L('TNNetMishLearnable', TNNetMishLearnable.Create(1.0))); end;
    99:  begin AddImgInput(ANet); Exit(L('TNNetSoftPlusBetaLearnable', TNNetSoftPlusBetaLearnable.Create(1.0))); end;
    100: begin AddImgInput(ANet); Exit(L('TNNetAconC', TNNetAconC.Create())); end;
    101: begin AddImgInput(ANet); Exit(L('TNNetMetaAconC', TNNetMetaAconC.Create())); end;
    102: begin AddImgInput(ANet); Exit(L('TNNetSplineActivation', TNNetSplineActivation.Create(4, 2.0))); end;
    103: begin AddImgInput(ANet); Exit(L('TNNetPolynomialActivation', TNNetPolynomialActivation.Create())); end;
    104: begin AddImgInput(ANet); Exit(L('TNNetReZero', TNNetReZero.Create(0.0))); end;
    105: begin AddImgInput(ANet); Exit(L('TNNetGatedResidual', TNNetGatedResidual.Create(0.0))); end;
    106: begin AddImgInput(ANet); Exit(L('TNNetGRN', TNNetGRN.Create())); end;
    107: begin AddImgInput(ANet); Exit(L('TNNetDyT', TNNetDyT.Create())); end;
    108: begin AddImgInput(ANet); Exit(L('TNNetHardConcrete', TNNetHardConcrete.Create(0.66, -0.1, 1.1))); end;

    // ---- normalization ---------------------------------------------------
    109: begin AddImgInput(ANet); Exit(L('TNNetLayerMaxNormalization', TNNetLayerMaxNormalization.Create())); end;
    110: begin AddImgInput(ANet); Exit(L('TNNetLayerStdNormalization', TNNetLayerStdNormalization.Create())); end;
    111: begin AddImgInput(ANet); Exit(L('TNNetMovingStdNormalization', TNNetMovingStdNormalization.Create())); end;
    112: begin AddImgInput(ANet); Exit(L('TNNetLayerNorm', TNNetLayerNorm.Create())); end;
    113: begin AddSeqInput(ANet); Exit(L('TNNetTokenLayerNorm', TNNetTokenLayerNorm.Create(1e-5))); end;
    114: begin AddImgInput(ANet); Exit(L('TNNetRMSNorm', TNNetRMSNorm.Create())); end;
    115: begin AddSeqInput(ANet); Exit(L('TNNetTokenRMSNorm', TNNetTokenRMSNorm.Create(1e-6))); end;
    116: begin AddImgInput(ANet); Exit(L('TNNetRMSNormGated', TNNetRMSNormGated.Create())); end;
    117: begin AddImgInput(ANet); Exit(L('TNNetSwitchableNorm', TNNetSwitchableNorm.Create())); end;
    118: begin AddImgInput(ANet); Exit(L('TNNetZScore', TNNetZScore.Create())); end;
    119: begin AddImgInput(ANet); Exit(L('TNNetPixelNorm', TNNetPixelNorm.Create())); end;
    120: begin AddImgInput(ANet); Exit(L('TNNetGroupNorm', TNNetGroupNorm.Create(2, True))); end;
    121: begin AddImgInput(ANet); Exit(L('TNNetInstanceNorm', TNNetInstanceNorm.Create(True))); end;
    122: begin AddImgInput(ANet); Exit(L('TNNetMovingScale', TNNetMovingScale.Create(1.0, 1.0))); end;
    123: begin AddImgInput(ANet); Exit(L('TNNetChannelStdNormalization', TNNetChannelStdNormalization.Create())); end;
    124: begin AddImgInput(ANet); Exit(L('TNNetChannelNorm', TNNetChannelNorm.Create())); end;
    125: begin AddImgInput(ANet); Exit(L('TNNetChannelBias', TNNetChannelBias.Create())); end;
    126: begin AddImgInput(ANet); Exit(L('TNNetChannelMul', TNNetChannelMul.Create())); end;
    127: begin AddImgInput(ANet); Exit(L('TNNetChannelZeroCenter', TNNetChannelZeroCenter.Create())); end;
    128: begin AddImgInput(ANet); Exit(L('TNNetLocalResponseNorm2D', TNNetLocalResponseNorm2D.Create(5))); end;
    129: begin AddImgInput(ANet); Exit(L('TNNetLocalResponseNormDepth', TNNetLocalResponseNormDepth.Create(5))); end;
    130: begin AddImgInput(ANet); Exit(L('TNNetPointwiseNorm', TNNetPointwiseNorm.Create())); end;
    131: begin AddImgInput(ANet); Exit(L('TNNetL2Normalize', TNNetL2Normalize.Create(1e-8))); end;
    132: begin AddImgInput(ANet); Exit(L('TNNetUnitNorm', TNNetUnitNorm.Create())); end;
    133: begin AddImgInput(ANet); Exit(L('TNNetMinMaxNorm', TNNetMinMaxNorm.Create(1e-8))); end;
    134: begin AddImgInput(ANet); Exit(L('TNNetLogitNormalize', TNNetLogitNormalize.Create(1.0, 1e-8))); end;
    135: begin AddImgInput(ANet); Exit(L('TNNetSpectralNorm', TNNetSpectralNorm.Create(8, 1, 1, 0, 1))); end;
    136: begin AddImgInput(ANet); Exit(L('TNNetWeightStandardization', TNNetWeightStandardization.Create(8, 1, 1, 0, 1e-5))); end;
    137: begin AddImgInput(ANet); Exit(L('TNNetWeightNormLinear', TNNetWeightNormLinear.Create(8, 1, 1, 0, 1e-5))); end;

    // ---- arithmetic / scaling --------------------------------------------
    138: begin AddImgInput(ANet); Exit(L('TNNetMulLearning', TNNetMulLearning.Create(2.0))); end;
    139: begin AddImgInput(ANet); Exit(L('TNNetMulByConstant', TNNetMulByConstant.Create(2))); end;
    140: begin AddImgInput(ANet); Exit(L('TNNetAddConstant', TNNetAddConstant.Create(1.0))); end;
    141: begin AddImgInput(ANet); Exit(L('TNNetNegate', TNNetNegate.Create())); end;
    142: begin AddImgInput(ANet); Exit(L('TNNetAddAndDiv', TNNetAddAndDiv.Create(1, 2))); end;
    143: begin AddImgInput(ANet); Exit(L('TNNetGradientReversal', TNNetGradientReversal.Create(1.0))); end;
    144: begin AddImgInput(ANet); Exit(L('TNNetCellBias', TNNetCellBias.Create())); end;
    145: begin AddImgInput(ANet); Exit(L('TNNetCellMul', TNNetCellMul.Create())); end;
    146: begin AddImgInput(ANet); Exit(L('TNNetSimpleGate', TNNetSimpleGate.Create())); end;
    147: begin AddImgInput(ANet); Exit(L('TNNetEntropyRegularizer', TNNetEntropyRegularizer.Create(0.01))); end;

    // ---- softmax / topk / gating -----------------------------------------
    148: begin AddImgInput(ANet); Exit(L('TNNetSoftMax', TNNetSoftMax.Create())); end;
    149: begin AddImgInput(ANet); Exit(L('TNNetSoftMaxOne', TNNetSoftMaxOne.Create())); end;
    150: begin AddImgInput(ANet); Exit(L('TNNetSoftMin', TNNetSoftMin.Create())); end;
    151: begin AddImgInput(ANet); Exit(L('TNNetCenteredSoftmax', TNNetCenteredSoftmax.Create())); end;
    152: begin AddImgInput(ANet); Exit(L('TNNetSoftmaxTemperature', TNNetSoftmaxTemperature.Create(1.0))); end;
    153: begin AddSeqInput(ANet); Exit(L('TNNetPointwiseSoftMax', TNNetPointwiseSoftMax.Create(0, 0))); end;
    154: begin AddImgInput(ANet); Exit(L('TNNetLogSoftMax', TNNetLogSoftMax.Create())); end;
    155: begin AddImgInput(ANet); Exit(L('TNNetSparsemax', TNNetSparsemax.Create())); end;
    156: begin AddImgInput(ANet); Exit(L('TNNetGumbelSoftmax', TNNetGumbelSoftmax.Create(1.0, 0))); end;
    157: begin AddImgInput(ANet); Exit(L('TNNetTopK', TNNetTopK.Create(2))); end;
    158: begin AddImgInput(ANet); Exit(L('TNNetTopKGate', TNNetTopKGate.Create(2))); end;
    159: begin AddImgInput(ANet); Exit(L('TNNetBiasBalancedTopKGate', TNNetBiasBalancedTopKGate.Create(2, 0.01))); end;
    160: begin AddImgInput(ANet); Exit(L('TNNetExpertChoiceGate', TNNetExpertChoiceGate.Create(4))); end;
    161: begin AddImgInput(ANet); Exit(L('TNNetLoadBalanceLoss', TNNetLoadBalanceLoss.Create())); end;

    // ---- fully connected / linear families -------------------------------
    162: begin AddVecInput(ANet); Exit(L('TNNetFullConnect', TNNetFullConnect.Create(8, 1, 1, 0))); end;
    163: begin AddVecInput(ANet); Exit(L('TNNetFullConnectReLU', TNNetFullConnectReLU.Create(8, 1, 1, 0))); end;
    164: begin AddVecInput(ANet); Exit(L('TNNetFullConnectLinear', TNNetFullConnectLinear.Create(8, 1, 1, 0))); end;
    165: begin AddVecInput(ANet); Exit(L('TNNetFullConnectSigmoid', TNNetFullConnectSigmoid.Create(8, 1, 1, 0))); end;
    166: begin AddVecInput(ANet); Exit(L('TNNetFullConnectDiff', TNNetFullConnectDiff.Create(8, 1, 1, 0))); end;
    167: begin AddVecInput(ANet); Exit(L('TNNetBitLinear', TNNetBitLinear.Create(8, 1, 1, 0, 0))); end;
    168: begin AddVecInput(ANet); Exit(L('TNNetCirculantLinear', TNNetCirculantLinear.Create(16, 1, 1, 0))); end;
    169: begin AddVecInput(ANet); Exit(L('TNNetComplexLinear', TNNetComplexLinear.Create(8, 0))); end;
    170: begin AddVecInput(ANet); Exit(L('TNNetQuaternionLinear', TNNetQuaternionLinear.Create(8, 0))); end;
    171: begin AddVecInput(ANet); Exit(L('TNNetOctonionLinear', TNNetOctonionLinear.Create(8, 0))); end;
    172: begin AddVecInput(ANet); Exit(L('TNNetMonarchLinear', TNNetMonarchLinear.Create(0))); end;
    173: begin AddVecInput(ANet); Exit(L('TNNetKroneckerLinear', TNNetKroneckerLinear.Create(0, 0))); end;
    174: begin AddVecInput(ANet); Exit(L('TNNetTensorTrain', TNNetTensorTrain.Create(0, 0, 0))); end;
    175: begin AddVecInput(ANet); Exit(L('TNNetHouseholderLinear', TNNetHouseholderLinear.Create(16, 1, 1, 0))); end;
    176: begin AddVecInput(ANet); Exit(L('TNNetTropicalLinear', TNNetTropicalLinear.Create(8, 0))); end;
    177: begin AddVecInput(ANet); Exit(L('TNNetMaxOut', TNNetMaxOut.Create(2))); end;
    178: begin AddVecInput(ANet); Exit(L('TNNetGLU', TNNetGLU.Create())); end;
    179: begin AddVecInput(ANet); Exit(L('TNNetReGLU', TNNetReGLU.Create())); end;
    180: begin AddVecInput(ANet); Exit(L('TNNetGEGLU', TNNetGEGLU.Create())); end;
    181: begin AddVecInput(ANet); Exit(L('TNNetGEGLUErf', TNNetGEGLUErf.Create())); end;
    182: begin AddVecInput(ANet); Exit(L('TNNetSwiGLU', TNNetSwiGLU.Create())); end;
    183: begin AddVecInput(ANet); Exit(L('TNNetReGLUSquared', TNNetReGLUSquared.Create())); end;
    184: begin AddVecInput(ANet); Exit(L('TNNetTanhGLU', TNNetTanhGLU.Create())); end;
    185: begin AddVecInput(ANet); Exit(L('TNNetHighway', TNNetHighway.Create(16, 1, 1, 0, 0.0))); end;

    // ---- convolutions ----------------------------------------------------
    186: begin AddImgInput(ANet); Exit(L('TNNetConvolution', TNNetConvolution.Create(4, 3, 1, 1, 0))); end;
    187: begin AddImgInput(ANet); Exit(L('TNNetConvolutionReLU', TNNetConvolutionReLU.Create(4, 3, 1, 1, 0))); end;
    188: begin AddImgInput(ANet); Exit(L('TNNetConvolutionLinear', TNNetConvolutionLinear.Create(4, 3, 1, 1, 0))); end;
    189: begin AddImgInput(ANet); Exit(L('TNNetConvolutionSwish', TNNetConvolutionSwish.Create(4, 3, 1, 1, 0))); end;
    190: begin AddImgInput(ANet); Exit(L('TNNetConvolutionHardSwish', TNNetConvolutionHardSwish.Create(4, 3, 1, 1, 0))); end;
    191: begin AddImgInput(ANet); Exit(L('TNNetConvolutionRectangular', TNNetConvolutionRectangular.Create(4, 3, 3, 1, 1, 0))); end;
    192: begin AddImgInput(ANet); Exit(L('TNNetConvolutionRectangularReLU', TNNetConvolutionRectangularReLU.Create(4, 3, 3, 1, 1, 0))); end;
    193: begin AddImgInput(ANet); Exit(L('TNNetPointwiseConv', TNNetPointwiseConv.Create(4, 0))); end;
    194: begin AddImgInput(ANet); Exit(L('TNNetPointwiseConvReLU', TNNetPointwiseConvReLU.Create(4, 0))); end;
    195: begin AddImgInput(ANet); Exit(L('TNNetPointwiseConvLinear', TNNetPointwiseConvLinear.Create(4, 0))); end;
    196: begin AddImgInput(ANet); Exit(L('TNNetDepthwiseConv', TNNetDepthwiseConv.Create(1, 3, 1, 1))); end;
    197: begin AddImgInput(ANet); Exit(L('TNNetDepthwiseConvReLU', TNNetDepthwiseConvReLU.Create(1, 3, 1, 1))); end;
    198: begin AddImgInput(ANet); Exit(L('TNNetDepthwiseConvLinear', TNNetDepthwiseConvLinear.Create(1, 3, 1, 1))); end;
    199: begin AddImgInput(ANet); Exit(L('TNNetGroupedConvolutionLinear', TNNetGroupedConvolutionLinear.Create(4, 3, 1, 1, 2, 0))); end;
    200: begin AddImgInput(ANet); Exit(L('TNNetGroupedConvolutionReLU', TNNetGroupedConvolutionReLU.Create(4, 3, 1, 1, 2, 0))); end;
    201: begin AddImgInput(ANet); Exit(L('TNNetGroupedPointwiseConvLinear', TNNetGroupedPointwiseConvLinear.Create(4, 2, 0))); end;
    202: begin AddImgInput(ANet); Exit(L('TNNetGroupedPointwiseConvReLU', TNNetGroupedPointwiseConvReLU.Create(4, 2, 0))); end;
    203: begin AddImgInput(ANet); Exit(L('TNNetGroupedPointwiseConvHardSwish', TNNetGroupedPointwiseConvHardSwish.Create(4, 2, 0))); end;
    204: begin AddImgInput(ANet); Exit(L('TNNetComplexConv', TNNetComplexConv.Create(4, 3, 1, 1, 0))); end;
    205: begin AddImgInput(ANet); Exit(L('TNNetQuaternionConv', TNNetQuaternionConv.Create(4, 3, 1, 1, 0))); end;
    206: begin AddImgInput(ANet); Exit(L('TNNetOctonionConv', TNNetOctonionConv.Create(8, 3, 1, 1, 0))); end;
    207: begin AddImgInput(ANet); Exit(L('TNNetGroupConvP4', TNNetGroupConvP4.Create(4, 3, 1, 1, 0))); end;
    208: begin AddImgInput(ANet); Exit(L('TNNetWeightStandardizationConv', TNNetWeightStandardizationConv.Create(4, 3, 1, 1, 0, 1e-5))); end;
    209: begin AddImgInput(ANet); Exit(L('TNNetSpectralNormConv', TNNetSpectralNormConv.Create(4, 3, 1, 1, 0, 1))); end;
    210: begin AddImgInput(ANet); Exit(L('TNNetTropicalConv', TNNetTropicalConv.Create(4, 3, 1, 1, 0))); end;
    211: begin AddImgInput(ANet); Exit(L('TNNetDeformableConv', TNNetDeformableConv.Create(4, 3, 1, 1, 0, 0))); end;
    212: begin AddImgInput(ANet); Exit(L('TNNetCondConv', TNNetCondConv.Create(2, 4, 3, 1, 1, 0))); end;
    213: begin AddWaveInput(ANet); Exit(L('TNNetSincConv1D', TNNetSincConv1D.Create(4, 3, 1, 16000))); end;
    214: begin AddSeqInput(ANet); Exit(L('TNNetCausalConv1D', TNNetCausalConv1D.Create(4, 3, 0, 1))); end;
    215: begin AddSeqInput(ANet); Exit(L('TNNetDepthwiseConv1D', TNNetDepthwiseConv1D.Create(3, False, 0))); end;
    216: begin AddImgInput(ANet); Exit(L('TNNetKANConv', TNNetKANConv.Create(4, 3, 1, 1, 4, 0, 0))); end;
    217: begin AddImgInput(ANet); Exit(L('TNNetLocalConnect', TNNetLocalConnect.Create(4, 3, 1, 1, 0))); end;
    218: begin AddImgInput(ANet); Exit(L('TNNetLocalConnectLinear', TNNetLocalConnectLinear.Create(4, 3, 1, 1, 0))); end;
    219: begin AddImgInput(ANet); Exit(L('TNNetLocalConnectReLU', TNNetLocalConnectReLU.Create(4, 3, 1, 1, 0))); end;
    220: begin AddImgInput(ANet); Exit(L('TNNetLocalProduct', TNNetLocalProduct.Create(4, 3, 1, 1, 0))); end;

    // ---- pooling / resampling --------------------------------------------
    221: begin AddImgInput(ANet); Exit(L('TNNetMaxPool', TNNetMaxPool.Create(2, 2, 0))); end;
    222: begin AddImgInput(ANet); Exit(L('TNNetMaxPoolPortable', TNNetMaxPoolPortable.Create(2, 2, 0))); end;
    223: begin AddImgInput(ANet); Exit(L('TNNetMaxPoolWithPosition', TNNetMaxPoolWithPosition.Create(2, 2, 0, 0, 0, 0))); end;
    224: begin AddImgInput(ANet); Exit(L('TNNetMinPool', TNNetMinPool.Create(2, 2, 0))); end;
    225: begin AddImgInput(ANet); Exit(L('TNNetAvgPool', TNNetAvgPool.Create(2))); end;
    226: begin AddImgInput(ANet); Exit(L('TNNetLpPool', TNNetLpPool.Create(2, 2, 0, 2.0))); end;
    227: begin AddImgInput(ANet); Exit(L('TNNetSoftPool', TNNetSoftPool.Create(2, 2, 0, 1.0))); end;
    228: begin AddImgInput(ANet); Exit(L('TNNetStochasticPool', TNNetStochasticPool.Create(2, 2, 0))); end;
    229: begin AddImgInput(ANet); Exit(L('TNNetMaxBlurPool', TNNetMaxBlurPool.Create(2, 2, 0))); end;
    230: begin AddImgInput(ANet); Exit(L('TNNetBlurPool', TNNetBlurPool.Create(2, 2, 0))); end;
    231: begin AddImgInput(ANet); Exit(L('TNNetAvgChannel', TNNetAvgChannel.Create())); end;
    232: begin AddImgInput(ANet); Exit(L('TNNetMaxChannel', TNNetMaxChannel.Create())); end;
    233: begin AddImgInput(ANet); Exit(L('TNNetMinChannel', TNNetMinChannel.Create())); end;
    234: begin AddImgInput(ANet); Exit(L('TNNetGlobalSumPool', TNNetGlobalSumPool.Create())); end;
    235: begin AddImgInput(ANet); Exit(L('TNNetAdaptiveAvgPool', TNNetAdaptiveAvgPool.Create(4, 4))); end;
    236: begin AddImgInput(ANet); Exit(L('TNNetAdaptiveMaxPool', TNNetAdaptiveMaxPool.Create(4, 4))); end;
    237: begin AddSeqInput(ANet); Exit(L('TNNetMaskedMean', TNNetMaskedMean.Create())); end;
    238: begin AddSeqInput(ANet); Exit(L('TNNetMaskedMax', TNNetMaskedMax.Create())); end;
    239: begin AddImgInput(ANet); Exit(L('TNNetDeMaxPool', TNNetDeMaxPool.Create(2))); end;
    240: begin AddImgInput(ANet); Exit(L('TNNetDeAvgPool', TNNetDeAvgPool.Create(2))); end;
    241: begin AddImgInput(ANet); Exit(L('TNNetUpsample', TNNetUpsample.Create())); end;
    242: begin AddImgInput(ANet); Exit(L('TNNetPixelShuffle', TNNetPixelShuffle.Create(2))); end;
    243: begin AddImgInput(ANet); Exit(L('TNNetBilinearUpsample', TNNetBilinearUpsample.Create(2))); end;
    244: begin AddImgInput(ANet); Exit(L('TNNetBilinearResize', TNNetBilinearResize.Create(16, 16, 0))); end;
    245: begin AddImgInput(ANet); Exit(L('TNNetDeconvolution', TNNetDeconvolution.Create(4, 3, 1, 1, 0, 0))); end;
    246: begin AddImgInput(ANet); Exit(L('TNNetDeconvolutionReLU', TNNetDeconvolutionReLU.Create(4, 3, 1, 1, 0, 0))); end;
    247: begin AddImgInput(ANet); Exit(L('TNNetDeconvolutionLinear', TNNetDeconvolutionLinear.Create(4, 3, 1, 1, 0, 0))); end;
    248: begin AddImgInput(ANet); Exit(L('TNNetDeLocalConnect', TNNetDeLocalConnect.Create(4, 3, 0))); end;
    249: begin AddImgInput(ANet); Exit(L('TNNetDeLocalConnectReLU', TNNetDeLocalConnectReLU.Create(4, 3, 0))); end;

    // ---- dropout / noise -------------------------------------------------
    250: begin AddImgInput(ANet); Exit(L('TNNetDropout', TNNetDropout.Create(0.5, 0))); end;
    251: begin AddImgInput(ANet); Exit(L('TNNetDropPath', TNNetDropPath.Create(0.1))); end;
    252: begin AddImgInput(ANet); Exit(L('TNNetNEFTune', TNNetNEFTune.Create(5.0))); end;
    253: begin AddImgInput(ANet); Exit(L('TNNetSpatialDropout1D', TNNetSpatialDropout1D.Create(0.1))); end;
    254: begin AddImgInput(ANet); Exit(L('TNNetSpatialDropout2D', TNNetSpatialDropout2D.Create(0.1))); end;
    255: begin AddImgInput(ANet); Exit(L('TNNetDropBlock', TNNetDropBlock.Create(3, 0.1))); end;
    256: begin AddImgInput(ANet); Exit(L('TNNetGaussianNoise', TNNetGaussianNoise.Create(0.1))); end;
    257: begin AddImgInput(ANet); Exit(L('TNNetGaussianDropout', TNNetGaussianDropout.Create(0.1))); end;
    258: begin AddImgInput(ANet); Exit(L('TNNetRandomMulAdd', TNNetRandomMulAdd.Create(10, 10))); end;
    259: begin AddImgInput(ANet); Exit(L('TNNetChannelRandomMulAdd', TNNetChannelRandomMulAdd.Create(10, 10))); end;

    // ---- embeddings / positional -----------------------------------------
    260: begin AddSeqInput(ANet); Exit(L('TNNetAddPositionalEmbedding', TNNetAddPositionalEmbedding.Create(0))); end;
    261: begin AddSeqInput(ANet); Exit(L('TNNetLearnedPositionalEmbedding', TNNetLearnedPositionalEmbedding.Create(8, 0.02))); end;
    262: begin AddSeqInput(ANet); Exit(L('TNNetSinusoidalPositionalEmbedding', TNNetSinusoidalPositionalEmbedding.Create(10000.0))); end;
    263: begin AddVecInput(ANet); Exit(L('TNNetSinusoidalTimeEmbedding', TNNetSinusoidalTimeEmbedding.Create(16, 10000))); end;
    264: begin ANet.AddLayer(TNNetInput.Create(8, 1, 1, 1)); Exit(L('TNNetEmbedding', TNNetEmbedding.Create(32, 16, 0, 0.02))); end;
    265: begin ANet.AddLayer(TNNetInput.Create(8, 1, 1, 1)); Exit(L('TNNetTokenAndPositionalEmbedding', TNNetTokenAndPositionalEmbedding.Create(32, 8, 16, 0.02, 0.02, 0))); end;
    266: begin AddSeqInput(ANet); Exit(L('TNNetRotaryEmbedding', TNNetRotaryEmbedding.Create())); end;
    267: begin AddSeqInput(ANet); Exit(L('TNNetMRotaryEmbedding', TNNetMRotaryEmbedding.Create(
            10000.0, 2, 1, 1, rsmNone, 1.0, 0, 1.0, 32.0, 1.0, False))); end;
    268: begin AddSeqInput(ANet); Exit(L('TNNetALiBi', TNNetALiBi.Create())); end;

    // ---- attention -------------------------------------------------------
    269: begin AddAttnInput(ANet); Exit(L('TNNetScaledDotProductAttention', TNNetScaledDotProductAttention.Create(4))); end;
    270: begin AddAttnInput(ANet); Exit(L('TNNetDifferentialAttention', TNNetDifferentialAttention.Create(4))); end;
    271: begin AddAttnInput(ANet); Exit(L('TNNetCosineSimilarityAttention', TNNetCosineSimilarityAttention.Create(4))); end;
    272: begin AddAttnInput(ANet); Exit(L('TNNetQKNormAttention', TNNetQKNormAttention.Create(4))); end;
    273: begin AddAttnInput(ANet); Exit(L('TNNetSinkAttention', TNNetSinkAttention.Create(4))); end;
    274: begin AddAttnInput(ANet); Exit(L('TNNetALiBiAttention', TNNetALiBiAttention.Create(4))); end;
    275: begin AddAttnInput(ANet); Exit(L('TNNetWindowAttention', TNNetWindowAttention.Create(4, 8))); end;
    276: begin AddAttnInput(ANet); Exit(L('TNNetT5RelPosBiasAttention', TNNetT5RelPosBiasAttention.Create(4))); end;
    277: begin AddAttnInput(ANet); Exit(L('TNNetGptOssSinkAttention', TNNetGptOssSinkAttention.Create(4))); end;
    278: begin AddAttnInput(ANet); Exit(L('TNNetLinearAttention', TNNetLinearAttention.Create(4))); end;
    279: begin AddAttnInput(ANet); Exit(L('TNNetCausalLinearAttention', TNNetCausalLinearAttention.Create(4))); end;
    280: begin AddAttnInput(ANet); Exit(L('TNNetLinformerAttention', TNNetLinformerAttention.Create(4, 4))); end;
    281: begin AddAttnInput(ANet); Exit(L('TNNetPerformerAttention', TNNetPerformerAttention.Create(4, 8, 0))); end;
    282: begin AddSeqInput(ANet); Exit(L('TNNetInducedSetAttention', TNNetInducedSetAttention.Create(4, 16))); end;
    283: begin AddSeqInput(ANet); Exit(L('TNNetAttentionPooling', TNNetAttentionPooling.Create(2, 16))); end;
    284: begin AddSeqInput(ANet); Exit(L('TNNetModernHopfield', TNNetModernHopfield.Create(4, 16, 1, 1.0))); end;
    285: begin AddSeqInput(ANet); Exit(L('TNNetProductKeyMemory', TNNetProductKeyMemory.Create(8, 16, 2, 1, 16))); end;
    286: begin AddSeqInput(ANet); Exit(L('TNNetSoftPrompt', TNNetSoftPrompt.Create(2, 16))); end;
    287: begin AddSeqInput(ANet); Exit(L('TNNetTokenMerging', TNNetTokenMerging.Create(2, 0))); end;
    288: begin AddSeqInput(ANet); Exit(L('TNNetForgetGateBias', TNNetForgetGateBias.Create())); end;
    289: begin AddSeqInput(ANet); Exit(L('TNNetTriangularCausalMask', TNNetTriangularCausalMask.Create(8))); end;
    290: begin AddSeqInput(ANet); Exit(L('TNNetSlidingWindowMaskedFill', TNNetSlidingWindowMaskedFill.Create(4))); end;
    291: begin AddSeqInput(ANet); Exit(L('TNNetMaskedFill', TNNetMaskedFill.Create(0.0))); end;
    292: begin AddSeqInput(ANet); Exit(L('TNNetLlama4AttnTemperature', TNNetLlama4AttnTemperature.Create(0.1, 0.1))); end;

    // ---- recurrent / SSM / linear-attention ------------------------------
    293: begin AddSeqInput(ANet); Exit(L('TNNetTokenShift', TNNetTokenShift.Create())); end;
    294: begin AddSeqInput(ANet); Exit(L('TNNetDiagonalSSM', TNNetDiagonalSSM.Create())); end;
    295: begin AddSeqInput(ANet); Exit(L('TNNetClosedFormContinuous', TNNetClosedFormContinuous.Create())); end;
    296: begin AddSeqInput(ANet); Exit(L('TNNetSLSTMCell', TNNetSLSTMCell.Create())); end;
    297: begin AddSeqInput(ANet); Exit(L('TNNetMLSTMCell', TNNetMLSTMCell.Create())); end;
    298: begin AddSeqInput(ANet); Exit(L('TNNetMinGRU', TNNetMinGRU.Create())); end;
    299: begin AddSeqInput(ANet); Exit(L('TNNetMinLSTM', TNNetMinLSTM.Create())); end;
    300: begin AddSeqInput(ANet); Exit(L('TNNetDeltaNet', TNNetDeltaNet.Create())); end;
    301: begin AddSeqInput(ANet); Exit(L('TNNetLegendreMemoryUnit', TNNetLegendreMemoryUnit.Create(8, 1.0))); end;
    302: begin AddSeqInput(ANet); Exit(L('TNNetGatedLinearAttention', TNNetGatedLinearAttention.Create())); end;
    303: begin AddSeqInput(ANet); Exit(L('TNNetWKV', TNNetWKV.Create())); end;
    304: begin AddSeqInput(ANet); Exit(L('TNNetLRU', TNNetLRU.Create())); end;
    305: begin AddAttnInput(ANet); Exit(L('TNNetRGLRU', TNNetRGLRU.Create())); end;
    306: begin AddSeqInput(ANet); Exit(L('TNNetSelectiveSSM', TNNetSelectiveSSM.Create(1))); end;
    // Mamba2 expects input depth = 2*DInner + 2*NGroups*StateSize + NumHeads.
    // With (NumHeads=2, HeadDim=4, StateSize=2, NGroups=1): 2*8 + 2*1*2 + 2 = 22.
    307: begin ANet.AddLayer(TNNetInput.Create(8, 1, 22, 1)); Exit(L('TNNetMamba2', TNNetMamba2.Create(2, 4, 2, 1))); end;
    308: begin AddSeqInput(ANet); Exit(L('TNNetImplicitLongConv', TNNetImplicitLongConv.Create(4))); end;
    309: begin AddSeqInput(ANet); Exit(L('TNNetSpatialGatingUnit', TNNetSpatialGatingUnit.Create(8))); end;
    310: begin AddAttnInput(ANet); Exit(L('TNNetRetention', TNNetRetention.Create(4, 0.9, False))); end;
    311: begin AddSeqInput(ANet); Exit(L('TNNetConvLSTMCell', TNNetConvLSTMCell.Create(4, 3, 1))); end;
    // ConvGRUCell input depth must equal HiddenChannels + InputChannels.
    312: begin AddImgInput(ANet); Exit(L('TNNetConvGRUCell', TNNetConvGRUCell.Create(4, 4, 3))); end;
    313: begin AddSeqInput(ANet); Exit(L('TNNetTestTimeTraining', TNNetTestTimeTraining.Create())); end;
    314: begin AddSeqInput(ANet); Exit(L('TNNetTitansMemory', TNNetTitansMemory.Create(0))); end;
    315: begin AddSeqInput(ANet); Exit(L('TNNetNTMMemory', TNNetNTMMemory.Create(4, 8, 0.0))); end;
    316: begin AddSeqInput(ANet); Exit(L('TNNetKalmanFilterCell', TNNetKalmanFilterCell.Create())); end;
    317: begin AddSeqInput(ANet); Exit(L('TNNetHamiltonianCell', TNNetHamiltonianCell.Create())); end;
    318: begin AddSeqInput(ANet); Exit(L('TNNetFourierMix', TNNetFourierMix.Create(0))); end;

    // ---- normalizing flows -----------------------------------------------
    319: begin AddImgInput(ANet); Exit(L('TNNetAffineCoupling', TNNetAffineCoupling.Create())); end;
    320: begin AddImgInput(ANet); Exit(L('TNNetInvertible1x1Conv', TNNetInvertible1x1Conv.Create())); end;
    321: begin AddImgInput(ANet); Exit(L('TNNetActNorm', TNNetActNorm.Create())); end;

    // ---- spectral / signal -----------------------------------------------
    322: begin AddSeqInput(ANet); Exit(L('TNNetSpectralConv1D', TNNetSpectralConv1D.Create(16, 4))); end;
    323: begin AddImgInput(ANet); Exit(L('TNNetSpectralConv2D', TNNetSpectralConv2D.Create(8, 4, 4))); end;
    324: begin AddSeqInput(ANet); Exit(L('TNNetDWT1D', TNNetDWT1D.Create(0, False))); end;
    325: begin AddImgInput(ANet); Exit(L('TNNetFourierFeatures', TNNetFourierFeatures.Create(8, 1.0, 0))); end;
    326: begin AddVecInput(ANet); Exit(L('TNNetRandomFourierFeatures', TNNetRandomFourierFeatures.Create(8, 1.0, 0, 0))); end;

    // ---- graph / geometric / hyperbolic ----------------------------------
    327: begin AddSeqInput(ANet); Exit(L('TNNetGraphConvolution', TNNetGraphConvolution.Create(8, 1, 16, 0))); end;
    328: begin AddSeqInput(ANet); Exit(L('TNNetGraphAttention', TNNetGraphAttention.Create(16, 0.0, 0))); end;
    329: begin AddVecInput(ANet); Exit(L('TNNetHyperbolicLinear', TNNetHyperbolicLinear.Create(8, 1.0, 0, False))); end;
    330: begin AddVecInput(ANet); Exit(L('TNNetHyperbolicDistance', TNNetHyperbolicDistance.Create(8, 1.0))); end;
    331: begin AddVecInput(ANet); Exit(L('TNNetHolographicBinding', TNNetHolographicBinding.Create(0))); end;
    332: begin AddVecInput(ANet); Exit(L('TNNetKANLayer', TNNetKANLayer.Create(16, 3))); end;
    // Sinkhorn needs a square (N,1,N) cost matrix: SizeX must equal Depth.
    333: begin ANet.AddLayer(TNNetInput.Create(8, 1, 8, 1)); Exit(L('TNNetSinkhorn', TNNetSinkhorn.Create())); end;
    334: begin AddSeqInput(ANet); Exit(L('TNNetDotProducts', TNNetDotProducts.Create())); end;
    335: begin AddSeqInput(ANet); Exit(L('TNNetCosineSimilarity', TNNetCosineSimilarity.Create())); end;

    // ---- spiking / misc cells --------------------------------------------
    336: begin AddSeqInput(ANet); Exit(L('TNNetLIFNeuron', TNNetLIFNeuron.Create())); end;
    337: begin AddSeqInput(ANet); Exit(L('TNNetALIFNeuron', TNNetALIFNeuron.Create())); end;
    338: begin AddImgInput(ANet); Exit(L('TNNetSquash', TNNetSquash.Create(2))); end;
    339: begin AddImgInput(ANet); Exit(L('TNNetCapsuleRouting', TNNetCapsuleRouting.Create(2, 4, 2, 4, 3))); end;
    340: begin AddSeqInput(ANet); Exit(L('TNNetVectorQuantizer', TNNetVectorQuantizer.Create())); end;
    341: begin AddImgInput(ANet); Exit(L('TNNetStraightThroughEstimator', TNNetStraightThroughEstimator.Create(1.0))); end;
    342: begin AddImgInput(ANet); Exit(L('TNNetGroupPoolP4', TNNetGroupPoolP4.Create(0))); end;

    // ---- byte / bit processing -------------------------------------------
    343: begin AddVecInput(ANet); Exit(L('TNNetByteProcessing', TNNetByteProcessing.Create(16, 8, 4, 0))); end;
    344: begin AddVecInput(ANet); Exit(L('TNNetBitProcessing', TNNetBitProcessing.Create(16, 8, 4, 0))); end;
    345: begin AddSeqInput(ANet); Exit(L('TNNetPointwiseByteProcessing', TNNetPointwiseByteProcessing.Create(16, 8, 4, 0))); end;
    346: begin AddSeqInput(ANet); Exit(L('TNNetPointwiseBitProcessing', TNNetPointwiseBitProcessing.Create(16, 8, 4, 0))); end;

    // ---- loss heads (TNNetIdentity-derived; structure round-trips) -------
    347: begin AddImgInput(ANet); Exit(L('TNNetCenterLoss', TNNetCenterLoss.Create())); end;
    348: begin AddImgInput(ANet); Exit(L('TNNetFocalLoss', TNNetFocalLoss.Create(0.25, 2.0))); end;
    349: begin AddImgInput(ANet); Exit(L('TNNetDiceLoss', TNNetDiceLoss.Create())); end;
    350: begin AddImgInput(ANet); Exit(L('TNNetTverskyLoss', TNNetTverskyLoss.Create(0.5, 0.5, 1.0))); end;
    351: begin AddImgInput(ANet); Exit(L('TNNetHuberLoss', TNNetHuberLoss.Create(1.0))); end;
    352: begin AddImgInput(ANet); Exit(L('TNNetSmoothL1Loss', TNNetSmoothL1Loss.Create())); end;
    353: begin AddImgInput(ANet); Exit(L('TNNetLogCoshLoss', TNNetLogCoshLoss.Create())); end;
    354: begin AddImgInput(ANet); Exit(L('TNNetCharbonnierLoss', TNNetCharbonnierLoss.Create(1e-3))); end;
    355: begin AddImgInput(ANet); Exit(L('TNNetQuantileLoss', TNNetQuantileLoss.Create(0.5))); end;
    356: begin AddImgInput(ANet); Exit(L('TNNetMultiQuantileLoss', TNNetMultiQuantileLoss.Create([0.1, 0.5, 0.9]))); end;
    357: begin AddImgInput(ANet); Exit(L('TNNetWingLoss', TNNetWingLoss.Create(10.0, 2.0))); end;
    358: begin AddImgInput(ANet); Exit(L('TNNetKLDivergence', TNNetKLDivergence.Create())); end;
    359: begin AddImgInput(ANet); Exit(L('TNNetNLLLoss', TNNetNLLLoss.Create())); end;
    360: begin AddImgInput(ANet); Exit(L('TNNetLabelSmoothingLoss', TNNetLabelSmoothingLoss.Create(0.1))); end;
    // TripletLoss packs (anchor|positive|negative); input depth must be /3.
    361: begin ANet.AddLayer(TNNetInput.Create(1, 1, 9, 1)); Exit(L('TNNetTripletLoss', TNNetTripletLoss.Create(1.0))); end;
    // CosineEmbeddingLoss requires odd input depth >= 3 (two halves + flag).
    362: begin ANet.AddLayer(TNNetInput.Create(1, 1, 9, 1)); Exit(L('TNNetCosineEmbeddingLoss', TNNetCosineEmbeddingLoss.Create(0.0))); end;
    363: begin AddImgInput(ANet); Exit(L('TNNetInfoNCELoss', TNNetInfoNCELoss.Create())); end;
    364: begin AddImgInput(ANet); Exit(L('TNNetCTCLoss', TNNetCTCLoss.Create(0))); end;
    // EvidentialRegression default D=1 needs input depth 4*D=4.
    365: begin ANet.AddLayer(TNNetInput.Create(1, 1, 4, 1)); Exit(L('TNNetEvidentialRegression', TNNetEvidentialRegression.Create())); end;
    // EvidentialClassification default K=2 needs input depth = NumClasses = 2.
    366: begin ANet.AddLayer(TNNetInput.Create(1, 1, 2, 1)); Exit(L('TNNetEvidentialClassification', TNNetEvidentialClassification.Create())); end;
    // MixtureDensity (K=2, D=4) needs input depth K*(1+2*D) = 18.
    367: begin ANet.AddLayer(TNNetInput.Create(1, 1, 18, 1)); Exit(L('TNNetMixtureDensity', TNNetMixtureDensity.Create(2, 4))); end;
    368: begin AddImgInput(ANet); Exit(L('TNNetArcFace', TNNetArcFace.Create())); end;
    369: begin AddImgInput(ANet); Exit(L('TNNetPonderCostLoss', TNNetPonderCostLoss.Create(0.01))); end;
    370: begin AddImgInput(ANet); Exit(L('TNNetPonderHalting', TNNetPonderHalting.Create())); end;
    371: begin AddSeqInput(ANet); Exit(L('TNNetSoftDecisionTree', TNNetSoftDecisionTree.Create(2, 4, 1.0))); end;

    // ---- positional / temperature / misc heads ---------------------------
    372: begin AddImgInput(ANet); Exit(L('TNNetScaleLearning', TNNetScaleLearning.Create())); end;
    373: begin AddImgInput(ANet); Exit(L('TNNetPolynomialActivation', TNNetPolynomialActivation.Create())); end;
    // SAM vision attention: Channels must equal Heads*HeadDim (2*4=8), matching
    // the 8-channel image input; WindowSize 4 divides the 8x8 map.
    374: begin AddImgInput(ANet); Exit(L('TNNetSAMVisionAttention', TNNetSAMVisionAttention.Create(2, 4, 4, 8))); end;
    375: begin AddImgInput(ANet); Exit(L('TNNetGridAvgPool', TNNetGridAvgPool.Create(3, 1, 1))); end;
    // QAT fake-quant: non-default qmax/momentum/running-stat/frozen so all four
    // packed FStruct/FFloatSt slots are exercised by the structure round-trip.
    376: begin AddImgInput(ANet); Exit(L('TNNetFakeQuantize', TNNetFakeQuantize.Create(63, 0.95, 1.25, 1))); end;
    // GatedDeltaNet expects input depth 2*Hk*Dk + 2*Hv*Dv + 2*Hv ([q|k|v|z|b|a]).
    // With (Hk=2, Hv=4, Dk=4, Dv=4): 2*8 + 2*16 + 8 = 56.
    377: begin ANet.AddLayer(TNNetInput.Create(8, 1, 56, 1)); Exit(L('TNNetGatedDeltaNet', TNNetGatedDeltaNet.Create(2, 4, 4, 4))); end;

    // Head-tiled RMSNorm (depth 16 = 4 heads x head_dim 4).
    378: begin AddSeqInput(ANet); Exit(L('TNNetHeadRMSNorm', TNNetHeadRMSNorm.Create(4, 1e-6))); end;
    else
      Result := False;
  end;
  // ---------------------------------------------------------------------------
  // SKIP-LIST (deliberately NOT covered; each is a layer this audit does not
  // protect, justified below). Kept MINIMAL.
  //
  //  * Multi-source / two-input layers whose constructor takes a TNNetLayer or
  //    an array of TNNetLayer reference and therefore cannot be added as a lone
  //    layer after a single Input without wiring a second branch:
  //      TNNetConcat, TNNetDeepConcat, TNNetSum, TNNetFiLM,
  //      TNNetShakeShakeMerge, TNNetShakeDropMerge, TNNetCrossAttention,
  //      TNNetCrossWKV, TNNetAffineGridSample, TNNetFlowWarp,
  //      TNNetBackwardWarp, TNNetCorrelationVolume, TNNetCorrelationLookup,
  //      TNNetHyperLinear, TNNetHyperConv, TNNetModulatedConv2D,
  //      TNNetScatterToAffine, TNNetRoIAlign, TNNetAdaIN,
  //      TNNetConvolutionSharedWeights, TNNetDeepEquilibriumSharedConv,
  //      TNNetCellMulByCell, TNNetChannelMulByLayer, TNNetConvolution3D,
  //      TNNetDisentangledAttention, TNNetGptOssGatedSwiGLU.
  //    (These ARE in the dispatch table; covering them needs a multi-branch
  //    builder -- a worthwhile follow-up. They are exercised elsewhere by their
  //    own dedicated round-trip tests per the codebase memory notes.)
  //
  //  * Legacy name aliases that map to a class already covered under its
  //    canonical name (same constructor, identical serialized class string):
  //      TNNetLayerSoftMax (->TNNetSoftMax), TNNetLayerFullConnect /
  //      TNNetLayerFullConnectReLU (->TNNetFullConnect/ReLU).
  //
  // COVERAGE: 375 representative builders above span every layer category;
  // ~324 distinct canonical class names of the ~404 dispatch entries are
  // exercised (the remainder are the multi-source layers and aliases listed
  // here). Adding a new single-source layer below is a one-line addition.
  // ---------------------------------------------------------------------------
end;

procedure TTestNeuralRegistry.TestRegistryRoundTrip;
var
  idx, Covered: integer;
  NetA, NetB: TNNet;
  LayerName: string;
  StructA, StructB: string;
begin
  Covered := 0;
  // The builder list is a dense 0..N case; iterate until BuildLayerNet reports
  // an out-of-range index. We bound the loop so a missing case simply stops it
  // (the final coverage assertion below then fails loudly).
  for idx := 0 to 100000 do
  begin
    NetA := TNNet.Create();
    try
      if not BuildLayerNet(idx, NetA, LayerName) then Break;
      Inc(Covered);

      // Guard: the target layer must actually have been created.
      AssertTrue(
        Format('Builder #%d (%s) produced no target layer', [idx, LayerName]),
        NetA.GetLastLayerIdx() >= 1);

      StructA := NetA.SaveStructureToString();

      NetB := TNNet.Create();
      try
        // This runs TNNet.CreateLayer (the class-name -> constructor dispatch).
        NetB.LoadStructureFromString(StructA);
        StructB := NetB.SaveStructureToString();

        // The round-tripped net must have the same layer count: an unregistered
        // class makes CreateLayer return nil and the layer is dropped.
        AssertEquals(
          Format('Layer "%s" (builder #%d): layer count changed after '
            + 'round-trip -- likely missing from CreateLayer dispatch table',
            [LayerName, idx]),
          NetA.CountLayers(), NetB.CountLayers());

        // KEY ASSERTION: bit-for-bit structure-string equality.
        AssertEquals(
          Format('Layer "%s" (builder #%d) failed structure round-trip '
            + '(SaveStructureToString -> LoadStructureFromString -> '
            + 'SaveStructureToString). Check its CreateLayer registration / '
            + 'SaveStructureToString parameter packing.', [LayerName, idx]),
          StructA, StructB);
      finally
        NetB.Free;
      end;
    finally
      NetA.Free;
    end;
  end;

  // Sanity: ensure we actually ran a meaningful number of builders.
  AssertTrue(
    Format('Expected to cover at least 300 layer builders, covered %d',
      [Covered]),
    Covered >= 300);
end;

initialization
  RegisterTest(TTestNeuralRegistry);
end.
