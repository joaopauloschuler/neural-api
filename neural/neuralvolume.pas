(*
neuralvolume
Copyright (C) 2016 Joao Paulo Schwarz Schuler

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

unit neuralvolume;
// Coded, adapted and ported by Joao Paulo Schwarz Schuler
// https://sourceforge.net/p/cai/

// This class allows you to create an array that is at the same time 1D and 3D.
// This is useful in A.I. as sometimes the same data needs both representations.
// This is also useful when preparing data to OpenCL code.

// This class has extremelly fast methods for Single Precision floating point
// operations using AVX assembler. AVX instructions can be enabled with either
// AVX or AVX2 defines. Have a look at neuralnetwork.inc file.

// TVolume was inspired on and extended from convnet_vol:
// https://github.com/karpathy/convnetjs/blob/master/src/convnet_vol.js

// TVolume has also been inpired on Exentia
// http://www.tommesani.com/ExentiaWhatsNew.html

{$IFDEF FPC}
{$mode objfpc}
{$ENDIF}

interface

uses {$IFDEF FPC}fgl,{$ELSE}Contnrs,Generics.Collections,{$ENDIF} classes, sysutils, pascoremath32, pascoremathtypes, pascoremathhelperfuncs;

{$include neuralnetwork.inc}

const csMinAvxSize = 16;

const
  csEncodeRGB  = 0;
  csEncodeHSV  = 1;
  csEncodeHSL  = 2;
  csEncodeLAB  = 3;
  csEncodeGray = 4;

type
  TNeuralFloat = Single;
  TNeuralFloatDynArr = array of TNeuralFloat;
  TInt8DynArr = array of ShortInt;
  TNeuralFloatPtr = ^TNeuralFloat;

  TNeuralFloat4 = array[0..3] of TNeuralFloat;
  {$IFDEF FPC}
    {$IFDEF CPU32}
    TNeuralFloatArr = array[0..1024*2048] of TNeuralFloat;
    {$ELSE}
    TNeuralFloatArr = array[0..Maxint div SizeOf(TNeuralFloat)] of TNeuralFloat;
    {$ENDIF}
  {$ELSE}
    {$IFDEF CPUX86}
    TNeuralFloatArr = array[0..1024*2048] of TNeuralFloat;
    {$ELSE}
    TNeuralFloatArr = array[0..Maxint div SizeOf(TNeuralFloat) div 8] of TNeuralFloat; // Modified by Max 30/12/2019 [Data type too large: exceeds 2 GB]
    {$ENDIF}
  {$ENDIF}

  TNeuralFloatArrPtr = ^TNeuralFloatArr;
  TNeuralIntegerArray = array of integer;
  // Unbounded-view type over int8 quantized weight codes (never allocated as
  // such - only used to pointer-index into TInt8DynArr storage, mirroring how
  // TNeuralFloatArrPtr views float buffers). Coded by Claude (AI).
  TNeuralInt8Arr = array[0..Maxint div 2] of ShortInt;
  TNeuralInt8ArrPtr = ^TNeuralInt8Arr;

const
  csNeuralFloatSize = SizeOf(TNeuralFloat);
  csNeuralFloat4Size = SizeOf(TNeuralFloat4);
  csLongintSize = SizeOf(Longint);
  csIntegerSize = SizeOf(Integer);
  csShortIntSize = SizeOf(ShortInt);
  csNeuralFloat4Zero : TNeuralFloat4 = (0,0,0,0);
  csNeuralFloat4One : TNeuralFloat4  = (1,1,1,1);

type
  TNeuralActivationFunction = function(x:TNeuralFloat): TNeuralFloat;

  { TVolume }
  {$IFDEF FPC}
  TIntegerList = class (specialize TFPGList<integer>);
  generic TVolume<T> = class(TObject)
  {$ELSE}
  TIntegerList = TList<Integer>;
  T = TNeuralFloat;
  PtrInt = Integer;
  // This is a hack to allow compilation with other compilers
  TNNetList = class(TList)
    public
      FreeObjects: boolean;
      constructor Create(pFreeObjects: boolean = true);
      destructor Destroy(); override;
  end;

  TVolume = class(TObject)
  {$ENDIF}
    // T has to be a numerical/float type
  protected
    FSize: integer;
    FSizeX: integer;
    FSizeY: integer;
    FDepth: integer;
    FTag: array[0..1] of integer;
    FFormatSettings: TFormatSettings;
    FLastPos: integer;
    function GetTag: integer; {$IFDEF Release} inline; {$ENDIF}
    procedure SetTag(I: integer); {$IFDEF Release} inline; {$ENDIF}
    function GetTags(x: integer): integer; {$IFDEF Release} inline; {$ENDIF}
    procedure SetTags(x: integer; AValue: integer); {$IFDEF Release} inline; {$ENDIF}
    class procedure MulAddPPVS(PtrA, PtrB: TNeuralFloatArrPtr; Value: T;
      pSize: integer); {$IFDEF Release} inline; {$ENDIF}
  public
    // FData was made public to allow other fast operations
    FData: array of T;
    constructor Create(pSizeX, pSizeY, pDepth: integer; c: T = 0); {$IFNDEF FPC} overload; {$ENDIF}
    constructor Create(pInput: array of T); {$IFNDEF FPC} overload; {$ENDIF}
    constructor Create(Original: array of byte); {$IFNDEF FPC} overload; {$ENDIF}
    constructor Create(Original: TVolume); {$IFNDEF FPC} overload; {$ENDIF}
    constructor Create(Original: TBits; pFalse: T = -0.5; pTrue: T = +0.5); {$IFNDEF FPC} overload; {$ENDIF}
    constructor CreateAsBits(Original: array of byte; pFalse: T = -0.5; pTrue: T = +0.5); {$IFNDEF FPC} overload; {$ENDIF}
    constructor Create(pSize: integer; c: T = 0); {$IFNDEF FPC} overload; {$ENDIF}
    constructor Create(); {$IFNDEF FPC} overload; {$ENDIF}
    destructor Destroy(); override;
    procedure Fill(c: T = 0); {$IFDEF Release} inline; {$ENDIF}
    procedure FillForIdx(c: T; const aIdx: array of integer);
    procedure FillAtDepth(pDepth: integer; Value: T); {$IFDEF Release} inline; {$ENDIF}
    procedure FillForDebug();
    procedure Resize(pSize: integer); overload; virtual;
    procedure ReSize(pSizeX, pSizeY, pDepth: integer); overload; virtual;
    procedure ReSize(Original: TVolume); overload; virtual;
    function Get(x, y, d: integer): T; {$IFDEF Release} inline; {$ENDIF}
    function GetAsByte(x, y, d: integer): byte; {$IFDEF Release} inline; {$ENDIF}
    function GetRaw(x: integer): T; {$IFDEF Release} inline; {$ENDIF}
    procedure SetRaw(X: integer; Value: T); {$IFDEF Release} inline; {$ENDIF}
    procedure Store(x, y, d: integer; Value: T); {$IFDEF Release} inline; {$ENDIF}
    procedure Add(x, y, d: integer; Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Add(Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Add(Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    class procedure Add(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure AddAtDepth(pDepth: integer; Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure AddAtDepth(pDepth: integer; Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure AddFromDepthToDepth(Original: TVolume; FromDepth, ToDepth: integer); {$IFDEF Release} inline; {$ENDIF}
    procedure AddTransposingXD(Original: TVolume); {$IFDEF Release} inline; {$ENDIF}
    procedure AddTransposingYD(Original: TVolume); {$IFDEF Release} inline; {$ENDIF}
    procedure AddTransposingAs2D(Original: TVolume); {$IFDEF Release} inline; {$ENDIF}
    procedure CopyFromDepthToDepth(Original: TVolume; FromDepth, ToDepth: integer); {$IFDEF Release} inline; {$ENDIF}
    procedure AddLayers(A,B: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Sub(x, y, d: integer; Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Sub(Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Sub(Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Diff(Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure InterleaveWithDepthFrom(Original: TVolume; NewDepth: integer);{$IFDEF Release} inline; {$ENDIF}
    procedure InterleaveWithXFrom(Original: TVolume; NewX: integer); {$IFDEF Release} inline; {$ENDIF}
    function IncYSize(): integer; inline;
    function IncYSizeBytes(): integer; inline;
    function SameSize(Original: TVolume): boolean;
    procedure DeInterleaveWithXFrom(Original: TVolume; NewX: integer); {$IFDEF Release} inline; {$ENDIF}
    procedure DeInterleaveWithDepthFrom(Original: TVolume; NewDepth: integer);{$IFDEF Release} inline; {$ENDIF}
    procedure SetMin(Value: TNeuralFloat); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure SetMax(Value: TNeuralFloat); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Mul(x, y, d: integer; Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Mul(Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    class procedure Mul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat; pSize: integer); overload;
    class procedure Mul(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer); overload;
    // Element-wise depth-contiguous maximum: PtrA[i] := max(PtrA[i], PtrB[i]).
    // Scalar base; overridden by TNNetVolume with an AVX implementation.
    class procedure MaxElements(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer); overload;
    procedure Mul(Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure MulAtDepth(pDepth: integer; Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Pow(Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure PowMinus1();
    procedure VSqrt(); {$IFDEF Release} inline; {$ENDIF}
    procedure MulAdd(Value: T; Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure MulMulAdd(Value1, Value2: T; Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    class procedure MulMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value1, Value2: T; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure MulAdd(Value: T; PtrB: TNeuralFloatArrPtr); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure MulAdd(Original1, Original2: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    class procedure MulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value: T; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
    class procedure MulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
    // Rank-1 state/weight carry: Dst[i] := AlphaScale*Prev[i] + BScale*B[i].
    // Prev may be nil (the t=0 case: Prev is treated as the zero row), in which
    // case Dst[i] := BScale*B[i]. Prev may alias Dst (in-place carry). Routes
    // through the AVX Mul/MulAdd primitives so each row update is vectorized over
    // its (contiguous) inner axis. Shared by the rank-1 linear-attention state
    // updates (TNNetDeltaNet / TNNetGatedLinearAttention) and the test-time
    // inner-optimizer weight updates (TNNetTestTimeTraining / TNNetTitansMemory).
    class procedure RankOneUpdateRow(PtrDst, PtrPrev, PtrB: TNeuralFloatArrPtr;
      AlphaScale, BScale: T; pSize: integer); {$IFDEF Release} inline; {$ENDIF}
    procedure Divi(x, y, d: integer; Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Divi(Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Divi(Value: T); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure ForceMinRange(Value: T); {$IFDEF Release} inline; {$ENDIF}
    procedure ForceMaxRange(Value: T); {$IFDEF Release} inline; {$ENDIF}
    procedure ForceMaxMagnitude(Value: T); {$IFDEF Release} inline; {$ENDIF}
    procedure ForceMaxAbs(Value: T); {$IFDEF Release} inline; {$ENDIF}
    // Returns true if any element is NaN or +/-Inf (non-finite).
    function HasNonFinite(): boolean;
    procedure ForcePositive(); {$IFDEF Release} inline; {$ENDIF}
    procedure Randomize(a:integer=10000; b:integer=5000; c:integer=5000); {$IFDEF Release} inline; {$ENDIF}
    procedure RandomizeGaussian(pMul: TNeuralFloat = 1.0); {$IFDEF Release} inline; {$ENDIF}
    procedure AddGaussianNoise(pMul: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
    procedure AddSaltAndPepper(pNum: integer; pSalt: T = 1.0; pPepper: T = -1.0; pColor:boolean = false); {$IFDEF Release} inline; {$ENDIF}
    function RandomGaussianValue(): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
    // Copy
    procedure Copy(Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure CopyRelu(Original: TVolume); overload; {$IFDEF Release} inline; {$ENDIF}
    procedure Copy(Original: TVolume; Len: integer); {$IFNDEF FPC} overload; {$ENDIF} {$IFDEF Release} inline; {$ENDIF}
    procedure Copy(var Original: array of T); overload;
    procedure Copy(var Original: array of byte); overload;
    procedure Copy(Original: TBits; pFlase: T = -0.5; pTrue: T = +0.5); overload;
    procedure CopyPadding(Original: TVolume; Padding: integer); overload;
    procedure CopyPadding(Original: TVolume; PaddingX, PaddingY: integer); overload;
    procedure CopyCropping(Original: TVolume; StartX, StartY, pSizeX, pSizeY: integer);
    procedure CopyResizing(Original: TVolume; NewSizeX, NewSizeY: integer);
    procedure CopyNoChecks(Original: TVolume); overload;
    procedure CopyNoChecks(var Original: array of byte); overload;
    procedure CopyNoChecksIntArr(var Original: array of integer); overload;
    procedure CopyReversedNoChecksIntArr(var Original: array of integer); overload;
    procedure CopyNoChecks(var Original: string); overload;
    procedure CopyReversedNoChecks(var Original: string); overload;
    procedure CopyChannels(Original: TVolume; aChannels: array of integer);
    // Transpose Copying
    procedure CopyTransposingXD(Original: TVolume);
    procedure CopyTransposingYD(Original: TVolume);
    procedure CopyTransposingAs2D(Original: TVolume);
    procedure Define(Original: array of T);
    function DotProduct(Original: TVolume): T; overload; {$IFDEF Release} inline; {$ENDIF}
    class function DotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single; overload; {$IFDEF Release} inline; {$ENDIF}
    class function Product(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single; overload; {$IFDEF Release} inline; {$ENDIF}
    function SumDiff(Original: TVolume): T;  {$IFDEF Release} inline; {$ENDIF}
    procedure DebugDiff(Original: TVolume; Limit: Single = 0);
    procedure SumToPos(Original: TVolume);
    function GetDistanceSqr(Original: TVolume): T;  overload; {$IFDEF Release} inline; {$ENDIF}
    function GetDistance(Original: TVolume): T;  overload; {$IFDEF Release} inline; {$ENDIF}
    function SumAtDepth(pDepth: integer): T; {$IFDEF Release} inline; {$ENDIF}
    function AvgAtDepth(pDepth: integer): T; {$IFDEF Release} inline; {$ENDIF}
    function GetRawPos(x, y, d: integer): integer; overload; {$IFDEF Release} inline; {$ENDIF}
    function GetRawPos(x, y: integer): integer; overload; {$IFDEF Release} inline; {$ENDIF}
    function GetRawPtr(x, y, d: integer): pointer; overload; {$IFDEF Release} inline; {$ENDIF}
    function GetRawPtr(x, y: integer): pointer; overload; {$IFDEF Release} inline; {$ENDIF}
    function GetRawPtr(x: integer): pointer; overload; {$IFDEF Release} inline; {$ENDIF}
    function GetRawPtr(): pointer; overload; {$IFDEF Release} inline; {$ENDIF}
    function GetMin(): T; {$IFDEF Release} inline; {$ENDIF}
    function GetMax(): T; {$IFDEF Release} inline; {$ENDIF}
    function GetNonZero(): integer; {$IFDEF Release} inline; {$ENDIF}
    function GetMaxAbs(): T; {$IFDEF Release} inline; {$ENDIF}
    procedure GetMinMaxAtDepth(pDepth: integer; out pMin, pMax: T);
    function GetSum(): T; virtual;
    function GetSumAbs(): T; virtual;
    function GetSumSqr(): T; virtual;
    function GetAvg(): T; {$IFDEF Release} inline; {$ENDIF}
    function GetVariance(): T; {$IFDEF Release} inline; {$ENDIF}
    function GetValueCount(Value: T): integer;
    function GetSmallestIdxInRange(StartPos, Len: integer): integer;
    function GetStdDeviation(): T; {$IFDEF Release} inline; {$ENDIF}
    function GetMagnitude(): T; {$IFDEF Release} inline; {$ENDIF}
    function GetEntropy(): T;
    function GetPerplexity(): T;
    // Cross-entropy of this volume (treated as predicted probabilities)
    // against Target, along the depth axis at pixel (X, Y):
    //   -sum_d Target[X,Y,d] * Ln(Self[X,Y,d]).
    // Predicted values are clamped to >= 1e-12 before Ln to avoid log(0).
    // Mirrors GetClassOnPixel: it operates per pixel along the depth axis.
    function CrossEntropyOnPixel(Target: TVolume; X, Y: integer): T;
    // Mean of CrossEntropyOnPixel over every (X, Y) pixel of the volume.
    function MeanCrossEntropy(Target: TVolume): T;
    procedure FlipX();
    procedure FlipY();
    procedure IncTag(); {$IFDEF Release} inline; {$ENDIF}
    procedure ClearTag(); {$IFDEF Release} inline; {$ENDIF}
    function NeuralToStr(V: TNeuralFloat): string;

    // create lists with positions that are non zeros.
    procedure LoadNonZeroPosIntoTIntegerList(Ints: TIntegerList;
      IncludePositive: boolean=true; IncludeNegative:boolean = true);
    function CreateIntegerListWithNonZeroPos(IncludePositive: boolean=true;
      IncludeNegative:boolean = true): TIntegerList;

    // Color and Neuronal Weights Transformations
    procedure RgbImgToNeuronalInput(color_encoding: integer);
    procedure NeuronalInputToRgbImg(color_encoding: integer);
    procedure NeuronalWeightToImg(color_encoding: integer); overload;
    procedure NeuronalWeightToImg(MaxW, MinW:TNeuralFloat; color_encoding: integer); overload;
    procedure NeuronalWeightToImg3Channel(MaxW0, MinW0, MaxW1, MinW1, MaxW2, MinW2:TNeuralFloat; color_encoding: integer);

    procedure ZeroCenter();

    procedure Print();
    procedure PrintXD(Digits:integer=9; Decimals: integer=5);
    procedure PrintWithIndex();
    procedure PrintDebug();
    procedure PrintDebugChannel();

    // initializers
    procedure InitUniform(Value: T = 1);
    procedure InitGaussian(Value: T = 1);
    procedure InitLeCunUniform(Value: T = 1);
    procedure InitHeUniform(Value: T = 1);
    procedure InitLeCunGaussian(Value: T = 1);
    procedure InitHeGaussian(Value: T = 1);
    procedure InitSELU(Value: T = 1);

    // load and save functions
    function SaveToString(): string;
    procedure LoadFromString(strData: string);

    // bit operations
    procedure CopyAsBits(var Original: array of byte; pFalse: T = -0.5; pTrue: T = +0.5; CanResize: boolean = True); overload;
    procedure CopyAsBits(Original: string; pFalse: T = -0.5; pTrue: T = +0.5; CanResize: boolean = True); overload;
    procedure CopyAsBitsReversed(Original: string; pFalse: T = -0.5; pTrue: T = +0.5);
    procedure ReadAsBits(var Dest: array of byte; Threshold: T = 0.0);

    // Classification Functions (SetClass is similar to One Hot Encoding)
    procedure SetClass(pClass: integer; value: T); {$IFNDEF FPC} overload; {$ENDIF}
    procedure SetClass(pClass: integer; TrueValue, FalseValue: T); {$IFNDEF FPC} overload; {$ENDIF}
    procedure SetClassForHiperbolicTangent(pClass: integer);
    procedure SetClassForReLU(pClass: integer);
    procedure SetClassForSoftMax(pClass: integer);
    // GetClass is similar to argmax over the whole volume (returns the flat
    // index of the maximum element). Prefer it instead of hand-rolling an
    // argmax loop over Raw/FData.
    function GetClass(): integer;
    // GetClassOnPixel is the per-position argmax along the depth axis at pixel
    // (X, Y): it returns the depth index with the maximum value. This is
    // exactly the "argmax over the depth/vocab axis at a sequence position"
    // pattern (e.g. ArgMaxDepth(V, Pos) == V.GetClassOnPixel(Pos, 0)); reuse
    // it rather than re-implementing such a loop in callers/examples.
    function GetClassOnPixel(X, Y: integer): integer;
    function SoftMax(): T;
    procedure PointwiseSoftMax(NoForward: boolean = false);
    procedure GroupedPointwiseSoftMax(Groups: integer);

    // Encoding Functions
    // Sets the depth column at pixel (X, Y) to a one-hot of Token: writes 1 at
    // depth Token and 0 at every other depth of that pixel, leaving the rest of
    // the volume untouched. Inverse of GetClassOnPixel. Unlike the array/string
    // OneHotEncoding overloads it does NOT Fill(0) the whole volume nor pad
    // other positions, so it is the right primitive for per-position sequence
    // targets and for single-position one-hots.
    procedure OneHotEncodingOnPixel(X, Y, Token: integer);
    procedure OneHotEncoding(aTokens: array of integer); overload;
    procedure GroupedOneHotEncoding(aTokens: array of integer; Groups: integer); overload;
    procedure ReverseGroupedOneHotEncoding(out aTokens: TNeuralIntegerArray; Groups: integer);
    function ReverseGroupedOneHotEncodingOnPixel(Groups, X, Y: integer):integer;
    procedure OneHotEncoding(aTokens: string); overload;
    procedure OneHotEncodingAtEnd(aTokens: string); overload;
    procedure OneHotEncodingReversed(aTokens: string); overload;
    procedure OneHotEncodingReversed(var aTokens: array of integer); overload;
    // Sets positional embedding as per paper "Attention Is All You Need".
    // https://arxiv.org/abs/1706.03762 .
    // Fills the volume with the Vaswani sin/cos positional-encoding table.
    // PositionOffset (default 0, additive API) shifts every position by a
    // constant: PE(pos + PositionOffset, i). Used by streamed/incremental
    // decoding, where a short window of tokens must be encoded at their
    // ABSOLUTE sequence positions rather than at window-local positions.
    procedure PositionalEncoding(n: integer = 10000; PositionOffset: integer = 0);

    // Color Encoding Functions
    procedure RgbToHsv(); {$IFDEF Release} inline; {$ENDIF}
    procedure HsvToRgb(); {$IFDEF Release} inline; {$ENDIF}
    procedure RgbToHsl(); {$IFDEF Release} inline; {$ENDIF}
    procedure HslToRgb(); {$IFDEF Release} inline; {$ENDIF}
    procedure RgbToLab(); {$IFDEF Release} inline; {$ENDIF}
    procedure LabToRgb(); {$IFDEF Release} inline; {$ENDIF}
    procedure RgbToGray(); {$IFDEF Release} inline; {$ENDIF}
    procedure GetGrayFromRgb(Rgb: TVolume); {$IFDEF Release} inline; {$ENDIF}
    procedure MakeGray(color_encoding: integer);

    // Shift Functions
    procedure ShiftRight(Positions: integer = 1);
    procedure ShiftLeft();

    property Data[x, y, d: integer]: T read Get write Store; default;
    property AsByte[x, y, d: integer]: byte read GetAsByte;
    property Raw[x: integer]: T read GetRaw write SetRaw;
    property Tag: integer read GetTag write SetTag;
    property Tags[x: integer]:integer read GetTags write SetTags;
    property Size: integer read FSize;
    property SizeX: integer read FSizeX;
    property SizeY: integer read FSizeY;
    property Depth: integer read FDepth;
    property FormatSettings: TFormatSettings read FFormatSettings;
  end;

  TNNetToken = record
    Token: integer;
    Score: TNeuralFloat;
  end;

  TNNetGroupInfo = record
    GroupId: integer;
    GroupIdVectorSize: integer;
    PtrA: TNeuralFloatArrPtr;
  end;

  TNNetTokenArray = array of TNNetToken;

  TNNetGroupInfoArray = array of TNNetGroupInfo;

  { TNNetVolume }
  {$IFDEF FPC}
  TNNetVolume = class (specialize TVolume<TNeuralFloat>)
  {$ELSE}
  TNNetVolume = class (TVolume)
  {$ENDIF}
    private
      FDataPtr: TNeuralFloatArrPtr;
    public
      procedure ReSize(pSizeX, pSizeY, pDepth: integer); override;
      function GetMemSize(): integer; {$IFDEF Release} inline; {$ENDIF}
      procedure CalculateLocalResponseFrom2D(Original: TNNetVolume; pSize:integer; alpha, beta: TNeuralFloat );
      procedure CalculateLocalResponseFromDepth(Original: TNNetVolume; pSize:integer; alpha, beta: TNeuralFloat );
      procedure GetTokenArray(var TokenArray: TNNetTokenArray);
      procedure GetTokenArrayOnPixel(var TokenArray: TNNetTokenArray; X,Y: integer);
      (*
      Assume that "As" and "Bs" contain lists of vectors "A" and "B".
      "NumAs and NumBs" are the number of elements in the
      The DotProducts function runs dot products for all combinations of "As" and "Bs".
      "Convolutions" are "dot products".
      Assume 3 matrixes 2x2 of the type TNNetVolume: A, B and B transposed (BT)
      Assume c,d,e,f,x,y,z,w are of the type TNeuralFloat.

      These are the matrixes A, B and BT (B Transposed):
      A       B       BT
      c  d    x  y    x  z
      e  f    z  w    y  w

      A = [c, d, e, f]
      B = [x, y, z, w]

      a1  = [c, d]
      a2  = [e, f]

      b1  = [x, y]
      b2  = [z, w]

      bt1 = [x, z]
      bt2 = [y, w]

      A  = [a1 ,  a2]
      B  = [b1 ,  b2]
      BT = [bt1, bt2]

      * denotes "dot product".
      The result of DotProducts (2, 2, 2, A, B) will be: [a1* b1, a2* b1, a1* b2, a2* b2]
      The result of a matrix multiplicaton would be:     [a1*bt1, a1*bt2, a2*bt1, a2*bt2]
      The result of DotProducts (2, 2, 2, A, BT)will be: [a1*bt1, a2*bt1, a1*bt2, a2*bt2]
      The transposed result of DotProducts (2, 2, 4, A, BT) will be the same as a matrix multiplication AB.
      OR
      Given that (A B)T = (BT AT),
      The result of DotProducts (2, 2, 2, BT, A) is the same as a matrix multiplication AB.
      This interpretation is valid for the functions:
      * InterleavedDotProduct
      * DotProducts
      * DotProductsTiled
      *)
      procedure InterleavedDotProduct(InterleavedAs, B:TNNetVolume);  overload;
      procedure InterleavedDotProduct(InterleavedAs, Bs:TNNetVolume; VectorSize: integer); overload;
      procedure InterleavedDotProduct(InterleavedAs, Bs:TNNetVolume; BStart, BFinish, VectorSize: integer); overload;
      procedure DotProducts(NumAs, NumBs, VectorSize: integer; VAs, VBs: TNNetVolume; NoForward:boolean = false); overload;
      // Ranged variant computing only the B rows [BStart..BFinish]. Output cells
      // keep their absolute positions (FData[CntB*NumAs + CntA]), so concurrent
      // callers on disjoint B ranges write disjoint slices of the same volume.
      // Threaded callers must NOT pass NoForward=true (its Fill(0) clears the
      // WHOLE volume, racing with the other ranges).
      procedure DotProducts(NumAs, BStart, BFinish, VectorSize: integer; VAs, VBs: TNNetVolume; NoForward:boolean = false); overload;
      procedure DotProductsPointwise(VAs, VBs: TNNetVolume; NoForward:boolean = false); overload;
      // Ranged variant over the B rows [BStart..BFinish]; same absolute-position
      // guarantees as the ranged DotProducts. Never resizes Self - the caller
      // must have presized it (concurrent resize would race).
      procedure DotProductsPointwise(VAs, VBs: TNNetVolume; BStart, BFinish: integer; NoForward:boolean = false); overload;
      procedure DotProductsTiled(NumAs, NumBs, VectorSize: integer; VAs, VBs: TNNetVolume; TileSizeA, TileSizeB: integer); overload;
      // Ranged variant over the B rows [BStart..BFinish]; same absolute-position
      // guarantees as the ranged DotProducts. Tiles are anchored at BStart and the
      // last B tile may be PARTIAL (clamped to BFinish), so arbitrary thread
      // ranges are safe even when TileSizeB does not divide the range length.
      // Optional AStart..AFinish restricts the OUTPUT rows (A / neurons) this
      // call writes, keeping NumAs as the destination stride so the written
      // slice lands at FData[CntB*NumAs + CntA]. Default (AFinish<0) = the full
      // 0..NumAs-1 range, i.e. unchanged for every existing caller. A disjoint
      // A slice over all B is the neuron-axis intra-layer chunk; kernel-size
      // agnostic (VBs is the im2col matrix, so VectorSize folds the kernel), so
      // it covers spatial convs, not just pointwise. Reuses the same inline
      // kernel - no per-element call overhead. Coded by Claude (AI).
      procedure DotProductsTiled(NumAs, BStart, BFinish, VectorSize: integer; VAs, VBs: TNNetVolume; TileSizeA, TileSizeB: integer; AStart: integer = 0; AFinish: integer = -1); overload;
      // Fused int8-weight x float32-input dot product: returns the RAW code
      // sum (sum of code_i * b_i) with NO scale applied - the caller multiplies
      // by the per-row quantization scale once, so the kernel never touches a
      // dequantized FP32 weight copy (weights stream at 1 byte/element).
      // AVX2/x86-64 builds use an asm kernel (sign-extend + convert + FMA in
      // registers); every other build runs the pure Pascal loop.
      // Coded by Claude (AI).
      class function DotProductInt8(PtrA: TNeuralInt8ArrPtr; PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
      // Fused int8-weight x float32-input elementwise multiply-accumulate:
      // PtrA[i] += PtrCodes[i] * PtrB[i], with NO scale applied - the caller
      // multiplies the accumulated result by the per-row quantization scale
      // once (every tap of a row shares it), so the codes are never
      // dequantized to memory. Channelwise sibling of DotProductInt8 for the
      // depthwise convolution (product per element instead of a reduction).
      // Coded by Claude (AI).
      class procedure MulAddInt8(PtrA, PtrB: TNeuralFloatArrPtr; PtrCodes: TNeuralInt8ArrPtr; pSize: integer); static;
      // Fused int8 axpy: PtrA[i] += W * PtrCodes[i], the int8-code twin of
      // MulAdd(PtrA, PtrB, W, N). The caller folds every per-row scalar
      // (attention weight, softmax normalizer, the row's quantization scale)
      // into W, so the codes are never dequantized to memory. Built for the
      // int8 KV-cache decode value sum. Coded by Claude (AI).
      class procedure MulAddInt8Scalar(PtrA: TNeuralFloatArrPtr; PtrCodes: TNeuralInt8ArrPtr; W: TNeuralFloat; pSize: integer); static;
      // Int8-weight twin of DotProductsTiled: A rows are int8 codes laid out
      // exactly like the concatenated weights (row r at Codes[r*VectorSize]),
      // Scales[r] is row r's quantization scale (applied once per dot product,
      // fused into the output store). Same tiling and same output layout
      // (FData[CntB*NumAs + CntA]) as the FP32 version. Coded by Claude (AI).
      procedure DotProductsTiledInt8(NumAs, NumBs, VectorSize: integer; const Codes: array of ShortInt; const Scales: array of TNeuralFloat; VBs: TNNetVolume; TileSizeA, TileSizeB: integer); overload;
      // Ranged twin (same contract as the ranged DotProductsTiled): computes
      // only B columns [BStart..BFinish] and A rows [AStart..AFinish], with
      // ceil-division tiling anchored at the range start and a clamped trailing
      // partial tile. NumAs stays the output row stride, so a sliced call
      // writes exactly its own output elements - this is what the intra-layer
      // chunk scheduler calls (position-axis chunks range B, neuron-axis
      // chunks range A). AFinish < 0 means all rows. Coded by Claude (AI).
      procedure DotProductsTiledInt8(NumAs, BStart, BFinish, VectorSize: integer; const Codes: array of ShortInt; const Scales: array of TNeuralFloat; VBs: TNNetVolume; TileSizeA, TileSizeB: integer; AStart: integer = 0; AFinish: integer = -1); overload;
      procedure PointwiseNorm(pNorms: TNNetVolume = nil);
      procedure PointwiseMul(pNorms: TNNetVolume);
      // VectorExp writes dst[0..N-1] := exp(src[0..N-1]). On an AVX2 build it
      // uses an 8-wide polynomial approximation (AVXExp) with a scalar NeuralExp
      // remainder; on a non-AVX build it is a plain NeuralExp loop. Buffers may
      // alias (dst = src) since the read happens before the write per lane/element.
      class procedure VectorExp(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // VectorSigmoid writes dst[0..N-1] := 1/(1+exp(-src)). AVX2-accelerated
      // path built on VectorExp; numerically stable scalar form on the tail.
      class procedure VectorSigmoid(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // VectorTanh writes dst[0..N-1] := tanh(src[0..N-1]). Built on VectorExp
      // via tanh(x) = 1 - 2/(exp(2x)+1) so it inherits VectorExp's AVX2 path.
      // Matches pcr_tanhf to ~1e-6. Buffers may alias (dst = src).
      class procedure VectorTanh(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // VectorErf writes dst[0..N-1] := erf(src[0..N-1]) using the Abramowitz &
      // Stegun 7.1.26 approximation (|err| < 1.5e-7, i.e. matches pcr_erff to
      // ~1e-6). Built on VectorExp so it inherits the AVX2 path. dst may alias src.
      class procedure VectorErf(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // VectorSinh writes dst[0..N-1] := sinh(src[0..N-1]) via
      // sinh(x) = (exp(x) - exp(-x)) / 2, so it inherits VectorExp's AVX2 path.
      // exp(x) and exp(-x) are produced by two vectorized VectorExp passes into a
      // local scratch (NOT pDst) so pSrc is never clobbered; hence dst may alias
      // src. The clamped arg keeps exp finite; sinh stays accurate to ~1e-6 vs
      // pcr_sinhf over the activation parity range.
      class procedure VectorSinh(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // VectorLn writes dst[0..N-1] := ln(src[0..N-1]). On an AVX2 build it uses an
      // 8-wide Cephes logf polynomial (AVXLn) with a scalar pcr_logf remainder; on a
      // non-AVX build it is a plain pcr_logf loop. Matches pcr_logf to ~1e-6 over the
      // positive normal range. Buffers may alias (read-before-write per lane/element).
      class procedure VectorLn(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // VectorSin / VectorCos write dst[0..N-1] := sin/cos(src[0..N-1]). On an AVX2
      // build they use an 8-wide Cephes sinf/cosf polynomial (AVXSinCos) with a
      // 3-part Cody-Waite range reduction (accurate to large magnitudes) and a scalar
      // pcr_sinf/pcr_cosf remainder; non-AVX builds are plain RTL loops. ~1e-6 vs RTL.
      // Buffers may alias (dst = src).
      class procedure VectorSin(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      class procedure VectorCos(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // VectorArcSinh writes dst[0..N-1] := arcsinh(src) = ln(x + sqrt(x^2 + 1)).
      // Built on VectorLn (and a vectorized sqrt in the prep pass) so it inherits the
      // AVX2 path. The sqrt argument is always >= 1 so ln stays in its accurate range
      // and dst may alias src (the prep pass reads src into a scratch before VectorLn).
      class procedure VectorArcSinh(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      procedure AddArea(DestX, DestY, OriginX, OriginY, LenX, LenY: integer; Original: TNNetVolume);
      function HasAVX: boolean; {$IFDEF Release} inline; {$ENDIF}
      function HasAVX2: boolean; {$IFDEF Release} inline; {$ENDIF}
      function HasAVX512: boolean; {$IFDEF Release} inline; {$ENDIF}
      function PearsonCorrelation(Y : TNNetVolume): TNeuralFloat;
      // AddSumChannel adds the sum of each channel to the current 1D array.
      procedure AddSumChannel(Original: TNNetVolume); {$IFDEF Release} inline; {$ENDIF}
      // AddSumSqrChannel is designed to compute the sum of the squares of elements
      // channel-wise from Original and add this sum to the current volume.
      procedure AddSumSqrChannel(Original: TNNetVolume); {$IFDEF Release} inline; {$ENDIF}
      // AddToChannels receives an 1D array (Original). Each element in Original
      // will be summed to the entire XY 2D slice at the same depth.
      procedure AddToChannels(Original: TNNetVolume); {$IFDEF Release} inline; {$ENDIF}
      // MulChannels receives an 1D array (Original). Each element in Original
      // will multiply the entire XY 2D slice at the same depth.
      procedure MulChannels(Original: TNNetVolume); {$IFDEF Release} inline; {$ENDIF}
      procedure Mul(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure NormalizeMax(Value: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      /// Calculates the recurrence plot from the input volume
      // https://en.wikipedia.org/wiki/Recurrence_plot
      procedure RecurrencePlot(Original: TNNetVolume; Threshold: TNeuralFloat);
      /// This function creates one output channel for each input channel.
      // The recurrence plot is calculated from Original's X axis.
      // Output size is: Original.SizeX, Original.SizeX, Original.Depth.
      procedure RecurrencePlotCAI(Original: TNNetVolume);
      {$IFDEF AVXANY}
      procedure Fill(c: Single = 0); {$IFDEF Release} inline; {$ENDIF}
      procedure Add(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure Add(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Sub(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      function DotProduct(Original: TNNetVolume): TNeuralFloat; overload; {$IFDEF Release} inline; {$ENDIF}
      class function DotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single; overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Mul(Value: Single); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure Mul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat; pSize: integer); overload;
      class procedure Mul(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure MaxElements(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulAdd(Value: TNeuralFloat; Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulAdd(Original1, Original2: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulMulAdd(Value1, Value2: TNeuralFloat; Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulAdd(Value: TNeuralFloat; PtrB: TNeuralFloatArrPtr); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure MulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value: TNeuralFloat; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure MulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Divi(Value: Single); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Copy(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure CopyRelu(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure CopyPadding(Original: TNNetVolume; Padding: integer); overload;
      procedure CopyPadding(Original: TNNetVolume; PaddingX, PaddingY: integer); {$IFDEF Release} inline; {$ENDIF} overload;
      procedure CopyNoChecks(Original: TNNetVolume);
      function GetSum(): TNeuralFloat; override;
      function GetSumSqr(): TNeuralFloat; override;
      function GetDistanceSqr(Original: TNNetVolume): TNeuralFloat;  overload; {$IFDEF Release} inline; {$ENDIF}
      function GetDistance(Original: TNNetVolume): TNeuralFloat;  overload; {$IFDEF Release} inline; {$ENDIF}
      function SumDiff(Original: TNNetVolume): TNeuralFloat; overload; {$IFDEF Release} inline; {$ENDIF}
      {$ENDIF}
    property
      DataPtr: TNeuralFloatArrPtr read FDataPtr;
  end;

  { TNNetGroupedVolume }

  TNNetGroupedVolume = class(TNNetVolume)
    protected
      FGrInfoArray: TNNetGroupInfoArray;
    public
      destructor Destroy(); override;
      procedure GroupedDotProductsTiled(Groups, NumAs, NumBs, VectorSize: integer; VAs, VBs: TNNetVolume; TileSizeA, TileSizeB: integer);
      // Int8-weight twin of GroupedDotProductsTiled: A rows are int8 codes
      // laid out exactly like the concatenated weights (row r at
      // Codes[r*VectorSize]), Scales[r] applied once per dot product. Same
      // grouped B addressing (input vectors hold VectorSize*Groups, neuron r
      // reads its group's slice) and same output layout
      // (FData[CntB*NumAs + CntA]) as the FP32 version. Coded by Claude (AI).
      procedure GroupedDotProductsTiledInt8(Groups, NumAs, NumBs, VectorSize: integer; const Codes: array of ShortInt; const Scales: array of TNeuralFloat; VBs: TNNetVolume; TileSizeA, TileSizeB: integer);
  end;

  { TNNetSamplerBase }

  TNNetSamplerBase = class(TObject)
    protected
      FTokenArr: TNNetTokenArray;
    public
      function GetToken(Origin: TNNetVolume): integer; virtual; abstract;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; virtual; abstract;
      procedure SortTokenArray();
      // State-init hook for STATEFUL samplers (e.g. TNNetSamplerMirostat carries
      // a running mu across the generation). The streamed decode path calls this
      // at the start of every fresh sequence (right where it Reset()s the
      // session). Stateless samplers inherit the no-op default.
      procedure Reset(); virtual;
      destructor Destroy(); override;
  end;

  { TNNetSamplerGreedy }
  TNNetSamplerGreedy = class (TNNetSamplerBase)
    public
      function GetToken(Origin: TNNetVolume): integer; override;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; override;
  end;

  { TNNetSamplerTopK }
  TNNetSamplerTopK = class (TNNetSamplerBase)
    protected
      FTopK: integer;
    public
      constructor Create(TopK: integer);
      function GetToken(Origin: TNNetVolume): integer; override;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; override;
  end;

  { TNNetSamplerTopP }
  TNNetSamplerTopP = class (TNNetSamplerBase)
    protected
      FTopP: TNeuralFloat;
    public
      constructor Create(TopP: TNeuralFloat);
      function GetToken(Origin: TNNetVolume): integer; override;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; override;
  end;

  { TNNetSamplerMinP }
  // Min-p sampling (Nguyen et al. 2024, "Turning Up the Heat: Min-p Sampling
  // for Creative and Coherent LLM Outputs"). Operates on PROBABILITIES (a
  // post-softmax volume, same convention as TNNetSamplerTopP): keeps every
  // token whose probability satisfies p >= MinP * max(p), renormalizes the
  // kept mass and draws PROPORTIONALLY to the renormalized probabilities (a
  // true weighted draw). MinP = 1.0 keeps only the argmax (greedy);
  // MinP -> 0 approaches full ancestral sampling.
  // Coded by Claude (AI).
  TNNetSamplerMinP = class (TNNetSamplerBase)
    protected
      FMinP: TNeuralFloat;
      // Weighted draw over the (descending-sorted) FTokenArr entries that
      // pass the p >= MinP * max(p) cut.
      function SampleFromSorted(): integer;
    public
      constructor Create(MinP: TNeuralFloat);
      function GetToken(Origin: TNNetVolume): integer; override;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; override;
  end;

  { TNNetSamplerWeightedTopK }
  // HF-semantics top-k sampling. Operates on PROBABILITIES (a post-softmax
  // volume, same convention as TNNetSamplerTopP / TNNetSamplerMinP): keeps the
  // TopK highest-probability tokens, renormalizes their mass and draws
  // PROPORTIONALLY to the renormalized probabilities. This differs from the
  // legacy TNNetSamplerTopK, which draws UNIFORMLY (1/K each) among the top K
  // and is deliberately left unchanged for reproducibility. TopK <= 0 or
  // TopK >= vocab degenerates to full ancestral sampling over the whole row.
  // Coded by Claude (AI).
  TNNetSamplerWeightedTopK = class (TNNetSamplerBase)
    protected
      FTopK: integer;
      // Weighted draw over the top-K entries of the (descending-sorted)
      // FTokenArr, proportional to their renormalized probability mass.
      function SampleFromSorted(): integer;
    public
      constructor Create(TopK: integer);
      function GetToken(Origin: TNNetVolume): integer; override;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; override;
  end;

  { TNNetSamplerTypical }
  // Locally-typical sampling (Meister et al. 2023, "Locally Typical Sampling").
  // Operates on PROBABILITIES (a post-softmax volume, same convention as
  // TNNetSamplerTopP / TNNetSamplerMinP). Unlike top-k / top-p (truncate by RANK
  // or CUMULATIVE MASS), typical sampling truncates by how close each token's
  // surprise -log p is to the distribution's conditional (Shannon) entropy
  // H = -sum_t p_t log p_t: it keeps the SMALLEST set of tokens (sorted by
  // ascending |(-log p) - H|) whose cumulative probability first reaches FMass,
  // then draws PROPORTIONALLY to the renormalized kept mass. FMass in (0,1];
  // FMass >= 1 keeps the whole row (full ancestral sampling). The kept set is
  // the "locally typical" set: tokens that are neither surprisingly likely nor
  // surprisingly unlikely given the model's own uncertainty.
  // Coded by Claude (AI).
  TNNetSamplerTypical = class (TNNetSamplerBase)
    protected
      FMass: TNeuralFloat;
      // Build the typical set from FTokenArr (any order) and draw from it.
      function SampleTypical(): integer;
    public
      constructor Create(Mass: TNeuralFloat);
      function GetToken(Origin: TNNetVolume): integer; override;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; override;
  end;

  // Mirostat version selector for TNNetSamplerMirostat.
  TNNetMirostatVersion = (mvV1, mvV2);

  { TNNetSamplerMirostat }
  // Mirostat sampling (Basu et al. 2021, "Mirostat: A Neural Text Decoding
  // Algorithm that Directly Controls Perplexity"). A STATEFUL sampler: it
  // carries a running estimate Mu across the generation and, each step, picks a
  // truncation that drives the OBSERVED surprise -log p(chosen) toward the
  // target FTau (target surprise / cross-entropy in nats; perplexity = e^tau).
  // After each draw it updates Mu := Mu - FEta * (observedSurprise - FTau), a
  // simple feedback controller, so output entropy is held near FTau over time.
  // Operates on PROBABILITIES (post-softmax volume, same convention as the other
  // probability samplers). TWO versions:
  //   mvV1: estimates the Zipf exponent s from the top tokens, computes a
  //         target truncation size k from (Mu, s, vocab) and samples uniformly-
  //         then-weighted among the top-k (the original paper's algorithm).
  //   mvV2: the version-2 simplification - keep every token whose surprise
  //         -log p <= Mu, draw proportionally from that kept set (no Zipf
  //         estimate; the common llama.cpp default). FEta and FTau identical.
  // Mu is initialized to 2*FTau by Reset() (the paper's init) and the streamed
  // decode path calls Reset() at the start of each fresh sequence.
  // Coded by Claude (AI).
  TNNetSamplerMirostat = class (TNNetSamplerBase)
    protected
      FTau: TNeuralFloat;        // target surprise (nats)
      FEta: TNeuralFloat;        // learning rate of the Mu feedback loop
      FMu: TNeuralFloat;         // running surprise budget (state)
      FVersion: TNNetMirostatVersion;
      // Draw on the (descending-sorted) FTokenArr, update FMu from the chosen
      // token's surprise, return the token id.
      function SampleAndUpdate(): integer;
    public
      // Tau = target surprise in nats (e.g. 3.0). Eta = feedback learning rate
      // (e.g. 0.1). Version selects v1 / v2 (default v2).
      constructor Create(Tau: TNeuralFloat; Eta: TNeuralFloat = 0.1;
        Version: TNNetMirostatVersion = mvV2);
      // Re-arm the controller (Mu := 2*Tau) for a fresh generation.
      procedure Reset(); override;
      function GetToken(Origin: TNNetVolume): integer; override;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; override;
      // Read-only state introspection (tests assert Mu converges toward Tau).
      property Mu: TNeuralFloat read FMu;
      property Tau: TNeuralFloat read FTau;
  end;

  { TNNetTokenHistoryPenalty }
  // Stateful logit processor that sits BETWEEN the model output and the
  // TNNetSamplerBase family (Greedy / TopK / TopP). It is NOT a sampler: it
  // owns a per-token occurrence count over the tokens emitted so far and
  // rewrites the next-step logit volume in place (Apply) before a sampler
  // reads it, implementing three standard, distinct knobs:
  // (a) repetition penalty (Keskar et al. CTRL 2019) - divide a logit by
  //     FRepetition>1 if its token has appeared, in the sign-correct CTRL
  //     form (l := l/r for l>0, l := l*r for l<0) so a penalty always lowers
  //     the score;
  // (b) frequency penalty - subtract FFrequency * count[t] (scales with how
  //     OFTEN the token was used);
  // (c) presence penalty - subtract FPresence once for any token used at
  //     least once (a flat "encourage new tokens" push).
  // Typical caller usage:
  //   Penalty.Apply(Logits); tok := Sampler.GetToken(Logits);
  //   Penalty.RegisterToken(tok);
  // Coded by Claude (AI).
  TNNetTokenHistoryPenalty = class(TObject)
    protected
      FRepetition: TNeuralFloat;
      FFrequency: TNeuralFloat;
      FPresence: TNeuralFloat;
      FCounts: array of integer;
      procedure EnsureSize(NewSize: integer);
    public
      // Defaults are NO-OP: r=1.0, alpha_f=0.0, alpha_p=0.0.
      constructor Create(Repetition: TNeuralFloat = 1.0;
        Frequency: TNeuralFloat = 0.0; Presence: TNeuralFloat = 0.0);
      destructor Destroy(); override;
      // Increments the occurrence count of TokenId (call after each emit).
      procedure RegisterToken(TokenId: integer);
      // Clears all counts for a fresh sequence.
      procedure ResetHistory();
      // Mutates the logit volume in place; each element index is a token id.
      procedure Apply(Logits: TNNetVolume);
      // Probability-domain (POST-SOFTMAX) variant of Apply for callers whose
      // next-token volume holds probabilities rather than raw logits (e.g.
      // the streamed generation loop in neuraldecode, where the model ends
      // in a SoftMax). Works in log space (ln p = logit - logsumexp):
      //  (a) repetition: ln p is always <= 0, so the sign-correct CTRL rule
      //      reduces to ln p := ln p * r, i.e. p := p^r - the standard
      //      "power then renormalize" probability adaptation;
      //  (b/c) frequency/presence: subtracting alpha_f*count + alpha_p from
      //      the log multiplies p by exp(-alpha_f*count - alpha_p).
      // The volume is renormalized to sum 1 afterwards. Bit-for-bit no-op
      // when all knobs are at their defaults or no token has been seen.
      procedure ApplyToProbabilities(Probs: TNNetVolume);
  end;

  /// Implements a pair of volumes
  TNNetVolumePair = class(TObject)
    protected
      FA: TNNetVolume;
      FB: TNNetVolume;
    public
      constructor Create(); overload;
      constructor Create(pA, pB: TNNetVolume); overload;
      constructor CreateCopying(pA, pB: TNNetVolume); overload;

      destructor Destroy(); override;

      property A:TNNetVolume read FA;
      property B:TNNetVolume read FB;
      property I:TNNetVolume read FA;
      property O:TNNetVolume read FB;
  end;

  /// Class with string message events
  {$IFNDEF FPC} {$M+} {$ENDIF}
  TMObject = class(TObject)
    protected
      FMessageProc: TGetStrProc;
      FErrorProc: TGetStrProc;

    public
      constructor Create(); virtual;
      destructor Destroy(); override;

      procedure DefaultMessageProc(const S: string);
      procedure DefaultErrorProc(const S: string);
      procedure DefaultHideMessages(const S: string);
      procedure HideMessages();

    published
      property MessageProc: TGetStrProc read FMessageProc write FMessageProc;
      property ErrorProc: TGetStrProc read FErrorProc write FErrorProc;
  end;
  {$IFNDEF FPC} {$M-} {$ENDIF}

  /// TNNetVolume list
  {$IFDEF FPC}
  TNNetVolumeList = class (specialize TFPGObjectList<TNNetVolume>)
  {$ELSE}
  TNNetVolumeList = class (TNNetList)
    private
      function GetItem(Index: Integer): TNNetVolume; inline;
      procedure SetItem(Index: Integer; AObject: TNNetVolume); inline;
  {$ENDIF}
    public
      function GetTotalSize(): integer;
      function GetSum(): TNeuralFloat;
      function GetAvg(): TNeuralFloat;
      procedure AddValue(Value: TNeuralFloat);
      procedure Mul(Value: TNeuralFloat);
      procedure Divi(Value: TNeuralFloat);
      function GetClosestId(Original: TNNetVolume; var MinDist: TNeuralFloat): integer;
      function GetManhattanClosestId(Original: TNNetVolume; var MinDist: TNeuralFloat): integer;
      procedure Fill(c: Single = 0);
      procedure ClearTag();
      procedure FillTag(TagId, TagValue: integer);
      procedure ConcatInto(V: TNNetVolume);
      procedure InterleaveInto(V: TNNetVolume);
      procedure SplitFrom(V: TNNetVolume);
      procedure AddVolumes(pVolNum, pSizeX, pSizeY, pDepth: integer; c: TNeuralFloat = 0); overload;
      procedure AddVolumes(Origin: TNNetVolumeList); overload;
      procedure AddCopy(Origin: TNNetVolume);
      procedure AddInto(Original: TNNetVolume);
      procedure SortByTagAsc;
      procedure SortByTagDesc;
      procedure GetColumn(V: TNNetVolume; colIdx: integer);
      procedure ResizeImage(NewSizeX, NewSizeY: integer);
      procedure AddPadding(Padding: integer);
     {$IFNDEF FPC}
      property Items[Index: Integer]: TNNetVolume read GetItem write SetItem; default;
     {$ENDIF}
  end;

  /// A list of TNNetVolume pairs.
  {$IFDEF FPC}
  TNNetVolumePairList = class (specialize TFPGObjectList<TNNetVolumePair>);
  {$ELSE}
  TNNetVolumePairList = class (TNNetList)
    private
      function GetItem(Index: Integer): TNNetVolumePair; inline;
      procedure SetItem(Index: Integer; AObject: TNNetVolumePair); inline;
    public
      property Items[Index: Integer]: TNNetVolumePair read GetItem write SetItem; default;
  end;
  {$ENDIF}

  { TNNetKMeans }
  TNNetKMeans = class(TMObject)
    protected
      FManhattanDistance: boolean;
      FSample: TNNetVolumeList;
      FClusters: TNNetVolumeList;
      FClusterSums: TNNetVolumeList;
      FLastStepTime: double;
      FLastDistance: TNeuralFloat;
    public
      constructor Create(pVolNum, pSizeX, pSizeY, pDepth: integer; pManhattan: boolean = true); reintroduce;
      destructor Destroy(); override;

      procedure RunStep(RepositionClusters: boolean = true);
      procedure Resize(pVolNum, pSizeX, pSizeY, pDepth: integer);

      procedure Randomize();
      procedure RandomizeEmptyClusters();
      procedure AddSample(Original: TNNetVolume); {$IFDEF Release} inline; {$ENDIF}
      function GetClusterId(Original: TNNetVolume): integer; {$IFDEF Release} inline; {$ENDIF}
      function GetTotalSize(): integer;

      property Sample: TNNetVolumeList read FSample;
      property Clusters: TNNetVolumeList read FClusters;
      property LastStepTime: double read FLastStepTime;
      property LastDistance: TNeuralFloat read FLastDistance;
      property ManhattanDistance: boolean read FManhattanDistance write FManhattanDistance;
  end;

  { TNNetStringList }
  TNNetStringList = class(TStringList)
    public
      function GetRandomIndex():integer; {$IFDEF Release} inline; {$ENDIF}
      procedure KeepFirst(Cnt: integer);
      procedure KeepLast(Cnt: integer);
      procedure DeleteFirst(Cnt: integer);
      procedure DeleteLast(Cnt: integer);
      procedure SetCapacity(NewCapacity: Integer); override;
      function GetDelimitedTextFast: string;
      procedure LoadLargeFile(Filename: string);
  end;

  { TStringListInt }
  TStringListInt = class(TNNetStringList)
    private
      FTokenizer: TStringList;
      FIntegerToStr: array of string;

      function GetInteger(Index: Integer): PtrInt; {$IFDEF Release} inline; {$ENDIF}
      procedure PutInteger(Index: Integer; AValue: PtrInt); {$IFDEF Release} inline; {$ENDIF}
    public
      constructor Create;
      destructor Destroy; override;
      procedure LoadVocabularyFromFile(const filename: string);

      procedure SortByIntegerAsc;
      procedure SortByIntegerDesc;
      function AddInteger(const S: string; AValue: PtrInt): integer; {$IFDEF Release} inline; {$ENDIF}
      function WordToIndex(pWord:string): integer;
      function WordToInteger(pWord:string): integer;
      function IntegerToWord(pInteger: integer): string;
      procedure SaveCurrentPosition();
      procedure SaveCurrentPositionAndSort();
      procedure StringToIndexArray(pString: string; var IntArr: TNeuralIntegerArray);
      procedure StringToIntegerArray(pString: string; var IntArr: TNeuralIntegerArray);
      function IndexArrayToString(var IntArr: TNeuralIntegerArray): string;
      function IntegerArrayToString(var IntArr: TNeuralIntegerArray): string;
      function IntegerListToCsv(IL: TIntegerList; pDelimiter: char = ','): string;

      function DeTokenize(TokenId: integer): string; virtual;
      procedure Tokenize(pString: string; var IntArr: TNeuralIntegerArray); overload; virtual;
      function GetVocabCount(): integer; virtual;
      function TokenizerHasSeparator: boolean; virtual;

      property Integers[Index: Integer]: PtrInt read GetInteger write PutInteger;
  end;

  {$IFDEF FPC}
  { TStringsObj }
  generic TStringsObj<TObj> = class(TNNetStringList)
    private
      FSortedList: boolean;
      function GetList(Index: Integer): TObj; {$IFDEF Release} inline; {$ENDIF}
    public
      constructor Create;
      function AddObject(const S: string; AObject: TObject): Integer; override;
      procedure FixObjects();
      procedure AddStringObj(const S: string); {$IFDEF Release} inline; {$ENDIF}

      property List[Index: Integer]: TObj read GetList;
      property SortedList: boolean read FSortedList write FSortedList;
  end;

  TStringIntegerList = class (specialize TStringsObj<TIntegerList>);

  { TStringStringList }

  TStringStringList = class (specialize TStringsObj<TStringList>)
    public
      procedure LoadFromCsv(filename: string;
        SkipFirstLine:boolean = true;
        KeyId: integer = -1;
        Separator: char = ',');
      procedure SaveToCsv(filename: string;
        Separator: char = ',');
  end;

  TStringVolumeList = class (specialize TStringsObj<TNNetVolume>)
    public
      function CreateNonZeroPositionLists(): TStringIntegerList;
  end;

  TStringStringListVolume = class (specialize TStringsObj<TStringVolumeList>);

  {$ELSE}
  TStringsObj = class(TNNetStringList)
    private
      function GetList(Index: Integer): TObject;
      function CreateObject: TObject; virtual; abstract;
    public
      constructor Create;
      function AddObject(const S: string; AObject: TObject): Integer; override;
      procedure FixObjects();

      procedure AddStringObj(const S: string);
      property List[Index: Integer]: TObject read GetList;
  end;

  TStringIntegerList = class (TStringsObj)
    private
      function GetList(Index: Integer): TIntegerList;
      function CreateObject: TObject; override;
    public
      property List[Index: Integer]: TIntegerList read GetList;
  end;

  TStringStringList = class(TStringsObj)
    private
      function GetList(Index: Integer): TStringList;
      function CreateObject: TObject; override;
    public
      property List[Index: Integer]: TStringList read GetList;
  end;

  TStringVolumeList = class(TStringsObj)
    private
      function GetList(Index: Integer): TNNetVolume;
      function CreateObject: TObject;  override;
    public
      function CreateNonZeroPositionLists(): TStringIntegerList;

      property List[Index: Integer]: TNNetVolume read GetList;
    end;

  TStringStringListVolume = class(TStringsObj)
    private
      function GetList(Index: Integer): TStringVolumeList;
      function CreateObject: TObject;  override;
    public
      property List[Index: Integer]: TStringVolumeList read GetList;
    end;
  {$ENDIF}

  { TNNetDictionary }
  // This class creates a dictionary where integers contains the frequency.
  TNNetDictionary = class(TStringListInt)
    protected
      FMaxSize: integer;
    public
      constructor Create(pMaxSize: integer);

      function AddWordToDictionary(pWord:string): boolean;
      function AddWordsToDictionary(pString:string): boolean;
      procedure AddWordFromCsvField(filename: string; fieldId: integer;
        SkipFirstLine: boolean = True; Separator:char = ',');
      procedure RemoveAllStringsWithLessThen(I:integer);
      procedure StringToVolume(pString: string; Volume: TNNetVolume);
      function VolumeToString(Volume: TNNetVolume; Threshold: TNeuralFloat = 0.2): string;
      procedure CsvToTStringVolumeList(filename: string;
        GroupByFieldId, DataFieldId: integer; SVL: TStringVolumeList;
        SkipFirstLine: boolean = True; Separator:char = ',');
      procedure PrintDebug(FirstElements: integer);
      procedure SaveDictionaryToFile(Filename: string; Separator:char = ',');
      procedure LoadDictionaryFromFile(Filename: string; Separator:char = ',');
  end;

  function CreateTokenizedStringList(str: string; c:char):TNNetStringList; overload;
  function CreateTokenizedStringList(c:char):TNNetStringList; overload;

  function CreateQuotedTokenizedStringList(Str: string; Separator:char; QuoteChar: char):TNNetStringList; overload;
  function CreateQuotedTokenizedStringList(Separator:char; QuoteChar: char):TNNetStringList; overload;

  function HiperbolicTangent(x: TNeuralFloat): TNeuralFloat;
  function HiperbolicTangentDerivative(x: TNeuralFloat): TNeuralFloat;

  function RectifiedLinearUnit(x: TNeuralFloat): TNeuralFloat;
  function RectifiedLinearUnitDerivative(x: TNeuralFloat): TNeuralFloat;

  function Swish(x: TNeuralFloat): TNeuralFloat;
  function SwishDerivative(x: TNeuralFloat): TNeuralFloat;

  function HardSwish(x: TNeuralFloat): TNeuralFloat;
  function HardSwishDerivative(x: TNeuralFloat): TNeuralFloat;

  function RectifiedLinearUnitLeaky(x: TNeuralFloat): TNeuralFloat;
  function RectifiedLinearUnitLeakyDerivative(x: TNeuralFloat): TNeuralFloat;

  function SignedSquareRoot1(x: TNeuralFloat): TNeuralFloat;
  function SignedSquareRoot1Derivative(x: TNeuralFloat): TNeuralFloat;

  function ReLULeakyBound(x: TNeuralFloat): TNeuralFloat;
  function ReLULeakyBoundDerivative(x: TNeuralFloat): TNeuralFloat;

  function NeuralExp(x: TNeuralFloat): TNeuralFloat; {$IFDEF FPC} inline; {$ENDIF}
  function Sigmoid(x: TNeuralFloat): TNeuralFloat;
  function SigmoidDerivative(x: TNeuralFloat): TNeuralFloat;

  function Identity(x: TNeuralFloat): TNeuralFloat;
  function IdentityDerivative(x: TNeuralFloat): TNeuralFloat;
  function SoftmaxDerivative(x: TNeuralFloat): TNeuralFloat;

  function DiffAct(x: TNeuralFloat): TNeuralFloat;
  function DiffActDerivative(x: TNeuralFloat): TNeuralFloat;

  function NeuronForceMinMax(x, pMin, pMax: TNeuralFloat): TNeuralFloat; overload; {$IFDEF Release} inline; {$ENDIF}
  function NeuronForceMinMax(x, pMin, pMax: integer): integer; overload; {$IFDEF Release} inline; {$ENDIF}
  function NeuronForceRange(x, range:TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
  function NeuronForceMinRange(x, range:TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}

  procedure rgb2hsv(r,g,b: TNeuralFloat; var h,s,v: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
  procedure hsv2rgb(h,s,v: TNeuralFloat; var r,g,b: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}

  function hue2rgb(p, q, t: TNeuralFloat): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
  procedure rgb2hsl(r,g,b: TNeuralFloat; var h,s,l: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
  procedure hsl2rgb(h,s,l: TNeuralFloat; var r,g,b: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}

  procedure lab2rgb(l, a, b: TNeuralFloat; var r, g, bb: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
  procedure rgb2lab(r, g, b: TNeuralFloat; var l, a, bb: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}

  function RoundAsByte(x: TNeuralFloat): byte; {$IFDEF Release} inline; {$ENDIF}

  function CompareStringListIntegerAsc(List: TStringList; Index1, Index2: Integer): Integer;
  function CompareStringListIntegerDesc(List: TStringList; Index1, Index2: Integer): Integer;

  function CompareNNetVolumeListAsc(const Item1, Item2: TNNetVolume): Integer;
  function CompareNNetVolumeListDesc(const Item1, Item2: TNNetVolume): Integer;

  function NeuralFloatToStr(V: TNeuralFloat): string;
  function NeuralStrToFloat(V: String): TNeuralFloat;

  { AssertFinite scans every element of V for NaN/Inf and raises a
    labelled exception on the first offending value. Useful for catching
    numerical instability in forward/backward passes. }
  procedure AssertFinite(V: TNNetVolume; const Where: string);

  { RowSoftMax replaces Row in place with the numerically-stable softmax over
    all Row.Size elements (subtract the row max, exponentiate, divide by the
    sum). This is post-network host math over a flat logits/score row; it is
    NOT a TNNetSoftMax layer. A zero sum is left untouched. }
  procedure RowSoftMax(Row: TNNetVolume);

  { RowCosineSimilarity returns the cosine similarity between two equally-sized
    volumes (dot(A,B) / (||A||*||B||)), treating each as a flat vector of
    A.Size elements. Returns 0 when either vector has zero norm or the sizes
    differ. }
  function RowCosineSimilarity(A, B: TNNetVolume): TNeuralFloat;

  { NormalizeRowsL2 L2-normalizes each row of a (Rows,1,Dim) volume in place:
    every row vector of length Dim is divided by its own L2 norm (rows with
    zero norm are left untouched). For a single (1,1,Dim) embedding this
    normalizes that one vector. }
  procedure NormalizeRowsL2(Mat: TNNetVolume);

  { NeuralLinearSolve solves the dense linear system A*X = B in place by
    Gauss-Jordan elimination with partial pivoting (single precision). A is a
    row-major n x n matrix, B is a row-major n x m matrix; on return B holds the
    solution X and A is destroyed. Both arrays are flat TNeuralFloat arrays
    (A indexed A[row*n+col], B indexed B[row*m+col]). Returns False when A is
    singular (a near-zero pivot is encountered), True otherwise. This is the
    single shared dense solver used by the closed-form least-squares /
    ridge-regression callers across the library and examples. }
  function NeuralLinearSolve(var A: array of TNeuralFloat;
    var B: array of TNeuralFloat; n, m: integer): boolean;

  { NeuralBoxIoU returns the Intersection-over-Union of two axis-aligned boxes
    given in corner (x1,y1,x2,y2) format (x2>=x1, y2>=y1; pixel or any
    consistent unit). Degenerate boxes are clamped to zero area, and a zero
    union yields 0. This is the single shared box-IoU used by the object-
    detection NMS / matching code across the importers and examples. }
  function NeuralBoxIoU(AX1, AY1, AX2, AY2,
    BX1, BY1, BX2, BY2: TNeuralFloat): TNeuralFloat;

  { NeuralBoxGIoU returns the Generalized Intersection-over-Union (Rezatofighi
    et al. 2019) of two axis-aligned boxes in corner (x1,y1,x2,y2) format.
    GIoU = IoU - (area(C) - union) / area(C), where C is the smallest enclosing
    box of A and B. It lies in (-1,1], equals IoU when the boxes overlap, and
    stays a useful signal (negative) even when the boxes are disjoint, which is
    why DETR-style set-prediction matching/loss uses it instead of plain IoU.
    Degenerate boxes are clamped to zero area; a zero enclosing area yields 0. }
  function NeuralBoxGIoU(AX1, AY1, AX2, AY2,
    BX1, BY1, BX2, BY2: TNeuralFloat): TNeuralFloat;

  { NeuralGreedyNMS runs greedy, score-sorted, class-aware Non-Max-Suppression
    over Count boxes. Boxes are passed as four parallel flat arrays in corner
    (x1,y1,x2,y2) format; Scores[i] is box i's confidence; Classes[i] is its
    integer class id. The routine does NOT mutate any input array. It returns
    the kept box indices, ORDERED by descending score (ties keep the original
    relative order, because the internal sort is a stable selection sort over
    the index permutation). A box j is suppressed by an earlier (higher-score)
    kept box i ONLY when Classes[j] = Classes[i] AND IoU(i,j) > IoUThreshold
    (strictly greater, matching the YOLO post-process). Pass a class array of
    all-equal ids for class-agnostic NMS. }
  function NeuralGreedyNMS(
    const BX1, BY1, BX2, BY2, Scores: array of TNeuralFloat;
    const Classes: array of integer; Count: integer;
    IoUThreshold: TNeuralFloat): TNeuralIntegerArray;

  { RandomBetaValue draws a sample from a Beta(Alpha, Alpha) distribution
    using the repo's global Random RNG. Implemented via two Gamma(Alpha,1)
    draws: Beta = Ga/(Ga+Gb). For Alpha=1 this reduces to Uniform(0,1), the
    common practical Mixup default. The Gamma sampler uses the Marsaglia &
    Tsang (2000) method, supporting any Alpha > 0. }
  function RandomGammaValue(Alpha: TNeuralFloat): TNeuralFloat;
  function RandomBetaValue(Alpha: TNeuralFloat): TNeuralFloat;

  { MixVolumes computes the convex combination
      Output := Lambda*A + (1-Lambda)*B
    Output is resized to match A. A and B must have matching sizes. }
  procedure MixVolumes(Output, A, B: TNNetVolume; Lambda: TNeuralFloat);

  { CreateMixedVolumePairList returns a NEW TNNetVolumePairList (owning copies)
    where each pair is the Mixup convex combination of an original pair with a
    randomly-permuted partner pair. Lambda is drawn per pair from
    Beta(Alpha, Alpha). The input list is NOT mutated. The caller owns the
    result and must Free it. Pass a fixed FixedLambda >= 0 to override the
    Beta draw (handy for tests / deterministic runs); FixedLambda < 0 (default)
    uses the Beta sampler. }
  function CreateMixedVolumePairList(Original: TNNetVolumePairList;
    Alpha: TNeuralFloat = 1.0; FixedLambda: TNeuralFloat = -1.0): TNNetVolumePairList;

  { ComputeCutMixBox computes the standard CutMix rand_bbox for an image of
    size W x H. The cut ratio is r = sqrt(1 - Lambda); the box has size
    (r*W) x (r*H) centered at (CenterFracX*W, CenterFracY*H) and is clamped to
    the image bounds. CenterFracX/Y are in [0,1] (the caller draws them
    uniformly; exposing them keeps the geometry deterministic for tests).
    Returns the top-left corner (X0,Y0) and the clamped box size (BoxW,BoxH). }
  procedure ComputeCutMixBox(W, H: integer;
    Lambda, CenterFracX, CenterFracY: TNeuralFloat;
    out X0, Y0, BoxW, BoxH: integer);

  { CreateCutMixVolumePairList returns a NEW TNNetVolumePairList (owning copies)
    implementing CutMix (Yun et al. 2019): for each pair, a random rectangle of
    a randomly-permuted partner's input is pasted into a copy of this input
    (across the full depth), and the targets are mixed by the TRUE pasted-area
    fraction: target := LambdaAdj*target_a + (1-LambdaAdj)*target_b, where
    LambdaAdj = 1 - PastedArea/(W*H). Lambda ~ Beta(Alpha,Alpha) per pair; the
    box center is drawn uniformly. The input list is NOT mutated; the caller
    owns the result and must Free it. Pass FixedLambda >= 0 to override the Beta
    draw (handy for tests / deterministic runs); FixedLambda < 0 (default) uses
    the Beta sampler. }
  function CreateCutMixVolumePairList(Original: TNNetVolumePairList;
    Alpha: TNeuralFloat = 1.0; FixedLambda: TNeuralFloat = -1.0): TNNetVolumePairList;

  function GetLastChars(const InputStr: string; LenStr: Integer): string;

  procedure TestTNNetVolume();
  procedure TestKMeans();

  function GetDefaultNumericFormat: TFormatSettings;

  {$IFDEF AVXANY}
  // AVXExp writes pDst[0..N-1] := exp(pSrc[0..N-1]) using an 8-wide AVX2
  // polynomial approximation (scalar NeuralExp remainder for the N mod 8 tail).
  // Declared at interface scope so the generic TVolume.PointwiseSoftMax may call
  // it (a generic template cannot reference an implementation-private symbol).
  // Implemented in the AVX32 / AVX64 asm blocks. Buffers may alias.
  procedure AVXExp(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer);
  // AVXLn writes pDst[0..N-1] := ln(pSrc[0..N-1]) via an 8-wide Cephes logf
  // polynomial (scalar pcr_logf remainder). Buffers may alias.
  procedure AVXLn(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer);
  // AVXSinCos writes pDst[0..N-1] := sin or cos of pSrc[0..N-1] via an 8-wide Cephes
  // sinf/cosf polynomial with 3-part Cody-Waite range reduction (scalar RTL
  // remainder). DoCos selects cos (true) vs sin (false). Buffers may alias.
  procedure AVXSinCos(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer; DoCos: boolean);
  // AVXGetSum returns the sum of pSrc[0..N-1] via an 8-wide AVX2 reduction
  // (scalar tail for the N mod 4 remainder). Declared at interface scope so the
  // generic TVolume.SoftMax / PointwiseSoftMax may reduce the exp pass with it.
  function AVXGetSum(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
  {$ENDIF}

implementation

uses
  Math, neuralbit, strutils;

{$IFDEF AVX64}
{$IFDEF AVX2}
// Fused int8 x float32 dot product (raw code sum, no scale): sign-extends 8
// codes at a time to dwords (vpmovsxbd), converts to floats in-register
// (vcvtdq2ps) and FMAs against the float input - the dequantized weight never
// exists in memory, so the weight stream costs 1 byte/element. Same loop
// skeleton, accumulator discipline and 4-wide tail as AVXDotProduct.
// Coded by Claude (AI).
function AVXDotProductInt8(PtrA: TNeuralInt8ArrPtr; PtrB: TNeuralFloatArrPtr;
  NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB
  vxorps ymm0, ymm0, ymm0

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
  vxorps ymm1, ymm1, ymm1

@LargeAddLoop:
  vpmovsxbd ymm2, [rax]
  vpmovsxbd ymm3, [rax+8]
  vpmovsxbd ymm4, [rax+16]
  vpmovsxbd ymm5, [rax+24]

  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3
  vcvtdq2ps ymm4, ymm4
  vcvtdq2ps ymm5, ymm5

  vfmadd231ps ymm0, ymm2, [rdx]
  vfmadd231ps ymm1, ymm3, [rdx+32]
  vfmadd231ps ymm0, ymm4, [rdx+64]
  vfmadd231ps ymm1, ymm5, [rdx+96]

  add rax, 32
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

  vaddps ymm0, ymm0, ymm1
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps xmm0, xmm2

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 32) / 4
@SmallAddLoop:
  vzeroupper

  vpmovsxbd xmm2, [rax]
  vcvtdq2ps xmm2, xmm2
  movups xmm3, [rdx]
  mulps xmm2, xmm3
  addps xmm0, xmm2

  add rax, 4
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];

    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements > 0 then
  begin
    if MissedElements = 1
    then Result += PtrA^[localNumElements] * PtrB^[localNumElements]
    else if MissedElements = 2
    then Result +=
           PtrA^[localNumElements] * PtrB^[localNumElements] +
           PtrA^[localNumElements+1] * PtrB^[localNumElements+1]
    else Result +=
           PtrA^[localNumElements] * PtrB^[localNumElements] +
           PtrA^[localNumElements+1] * PtrB^[localNumElements+1] +
           PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
  end;
end;

// Fused int8 axpy PtrA[i] += W * PtrCodes[i]: AVXDotProductInt8's byte->float
// front end (vpmovsxbd + vcvtdq2ps, the codes stream at 1 byte/element and the
// dequantized value never exists in memory) grafted onto the scalar
// AVXMulAdd's broadcast-FMA + store-back back end. Note the asymmetric
// strides: 32 elements advance the code pointer 32 BYTES but the float
// pointer 128. Scalar remainder (N mod 4) in Pascal. Coded by Claude (AI).
procedure AVXMulAddInt8Scalar(PtrA: TNeuralFloatArrPtr;
  PtrCodes: TNeuralInt8ArrPtr; W: TNeuralFloat; NumElements: integer);
var
  WPtr: pointer;
  localNumElements, MissedElements: integer;
  i, NumElementsM1: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    WPtr := Addr(W);
  asm
  mov ecx, localNumElements
  mov rdx, WPtr
  VBROADCASTSS ymm5, [rdx]
  mov rax, PtrCodes
  mov rdx, PtrA

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:
  vpmovsxbd ymm0, [rax]
  vpmovsxbd ymm1, [rax+8]
  vpmovsxbd ymm2, [rax+16]
  vpmovsxbd ymm3, [rax+24]

  vcvtdq2ps ymm0, ymm0
  vcvtdq2ps ymm1, ymm1
  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3

  vmovups ymm6, [rdx]
  vmovups ymm7, [rdx+32]
  vfmadd231ps ymm6, ymm0, ymm5
  vfmadd231ps ymm7, ymm1, ymm5
  vmovups [rdx],    ymm6
  vmovups [rdx+32], ymm7

  vmovups ymm6, [rdx+64]
  vmovups ymm7, [rdx+96]
  vfmadd231ps ymm6, ymm2, ymm5
  vfmadd231ps ymm7, ymm3, ymm5
  vmovups [rdx+64], ymm6
  vmovups [rdx+96], ymm7

  add rax, 32
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 32) / 4
@SmallAddLoop:
  vpmovsxbd xmm0, [rax]
  vcvtdq2ps xmm0, xmm0
  vmovups xmm6, [rdx]
  vfmadd231ps xmm6, xmm0, xmm5
  vmovups [rdx], xmm6

  add rax, 4
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm5', 'ymm6', 'ymm7'
  ];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
    PtrA^[i] := PtrA^[i] + W * PtrCodes^[i];
end;

// Fused int8 elementwise multiply-accumulate PtrA[i] += PtrCodes[i] * PtrB[i]
// (the depthwise-conv tap kernel): same byte->float front end as above, with
// the float input as the FMA memory operand (three streams, RBX for the codes
// like the 3-pointer AVXMulAdd macro). Scalar remainder (N mod 4) in Pascal.
// Coded by Claude (AI).
procedure AVXMulAddInt8(PtrA, PtrB: TNeuralFloatArrPtr;
  PtrCodes: TNeuralInt8ArrPtr; NumElements: integer);
var
  localNumElements, MissedElements: integer;
  i, NumElementsM1: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rdx, PtrA
  mov rax, PtrB
  mov rbx, PtrCodes

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:
  vpmovsxbd ymm0, [rbx]
  vpmovsxbd ymm1, [rbx+8]
  vpmovsxbd ymm2, [rbx+16]
  vpmovsxbd ymm3, [rbx+24]

  vcvtdq2ps ymm0, ymm0
  vcvtdq2ps ymm1, ymm1
  vcvtdq2ps ymm2, ymm2
  vcvtdq2ps ymm3, ymm3

  vmovups ymm6, [rdx]
  vmovups ymm7, [rdx+32]
  vfmadd231ps ymm6, ymm0, [rax]
  vfmadd231ps ymm7, ymm1, [rax+32]
  vmovups [rdx],    ymm6
  vmovups [rdx+32], ymm7

  vmovups ymm6, [rdx+64]
  vmovups ymm7, [rdx+96]
  vfmadd231ps ymm6, ymm2, [rax+64]
  vfmadd231ps ymm7, ymm3, [rax+96]
  vmovups [rdx+64], ymm6
  vmovups [rdx+96], ymm7

  add rbx, 32
  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 32) / 4
@SmallAddLoop:
  vpmovsxbd xmm0, [rbx]
  vcvtdq2ps xmm0, xmm0
  vmovups xmm6, [rdx]
  vfmadd231ps xmm6, xmm0, [rax]
  vmovups [rdx], xmm6

  add rbx, 4
  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  end
  [
    'RAX', 'RBX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm6', 'ymm7'
  ];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
    PtrA^[i] := PtrA^[i] + PtrCodes^[i] * PtrB^[i];
end;
{$ENDIF}
{$ENDIF}

{$IFDEF AVX2}
// Constants for the AVX2 8-wide exp() polynomial approximation (AVXExp).
// exp(x) = 2^(x*log2e); split t=x*log2e into k=round(t) and f=t-k in [-0.5,0.5];
// 2^k is built from the float exponent bits, 2^f = exp(f*ln2) via a degree-6
// minimax-style Taylor/Horner polynomial. Max relative error ~1e-6, far below
// the 1e-4 parity target against the scalar pcr_expf reference.
const
  cAVXExpHi:  Single  = 88.3762626647949;
  cAVXExpLo:  Single  = -88.3762626647949;
  cAVXLog2e:  Single  = 1.44269504088896341;
  cAVXLn2:    Single  = 0.6931471805599453;
  cAVXExpP0:  Single  = 1.0;
  cAVXExpP1:  Single  = 1.0;
  cAVXExpP2:  Single  = 0.5;
  cAVXExpP3:  Single  = 0.16666666666666666;
  cAVXExpP4:  Single  = 0.041666666666666664;
  cAVXExpP5:  Single  = 0.008333333333333333;
  cAVXExpP6:  Single  = 0.001388888888888889;
  cAVXExp127: Integer = 127;

// Constants for the AVX2 8-wide ln() approximation (AVXLn), Cephes single-precision
// logf. Decompose x = m * 2^e with m in [sqrt(0.5), sqrt(2)); ln(x) = ln(m) + e*ln2,
// where ln(m) is a degree-8 minimax polynomial in (m-1). Max relative error ~2e-7
// over the normal positive range, far below the 1e-4 parity target vs pcr_logf.
  cAVXLnP0:   Single  =  7.0376836292E-2;
  cAVXLnP1:   Single  = -1.1514610310E-1;
  cAVXLnP2:   Single  =  1.1676998740E-1;
  cAVXLnP3:   Single  = -1.2420140846E-1;
  cAVXLnP4:   Single  =  1.4249322787E-1;
  cAVXLnP5:   Single  = -1.6668057665E-1;
  cAVXLnP6:   Single  =  2.0000714765E-1;
  cAVXLnP7:   Single  = -2.4999993993E-1;
  cAVXLnP8:   Single  =  3.3333331174E-1;
  cAVXLnQ1:   Single  = -2.12194440E-4;     // ln2 correction tail
  cAVXLnQ2:   Single  =  0.693359375;       // ln2 lead
  cAVXLnSqrtHf: Single = 0.707106781186547524; // sqrt(0.5)
  cAVXLnHalf:  Single = 0.5;
  cAVXLnOne:   Single = 1.0;
  cAVXLnMinNorm: Integer = $00800000;       // smallest positive normal float bits
  cAVXLnInvMant: uInt32  = $807fffff;       // sign + mantissa mask (clears exponent)

// Constants for the AVX2 8-wide sin()/cos() approximation (AVXSinCos), Cephes
// single-precision sinf/cosf. Range-reduce x by q = round(x * 4/pi); the low 3 bits
// of q select the octant and the sin/cos polynomial + sign. Max abs error ~1e-7 over
// a wide range; we extend the reduction with a 3-part Cody-Waite pi/4 subtraction so
// it stays accurate out to large magnitudes (|x| up to ~1e5).
  cAVXSC_FOPI:  Single =  1.27323954473516;   // 4/pi
  cAVXSC_DP1:   Single = -0.78515625;
  cAVXSC_DP2:   Single = -2.4187564849853515625E-4;
  cAVXSC_DP3:   Single = -3.77489497744594108E-8;
  cAVXSC_SinP0: Single = -1.9515295891E-4;
  cAVXSC_SinP1: Single =  8.3321608736E-3;
  cAVXSC_SinP2: Single = -1.6666654611E-1;
  cAVXSC_CosP0: Single =  2.443315711809948E-5;
  cAVXSC_CosP1: Single = -1.388731625493765E-3;
  cAVXSC_CosP2: Single =  4.166664568298827E-2;
  cAVXSC_Half:  Single =  0.5;
  cAVXSC_One:   Single =  1.0;
  cAVXSC_1i:    Integer = 1;
  cAVXSC_2i:    Integer = 2;
  cAVXSC_4i:    Integer = 4;
  cAVXSC_NOT1i: Integer = -2;                 // not(1) = $FFFFFFFE
{$ENDIF}

function CreateTokenizedStringList(str: string; c:char):TNNetStringList;
begin
  Result := CreateTokenizedStringList(c);
  Result.DelimitedText := str;
end;

function CreateTokenizedStringList(c: char): TNNetStringList;
begin
  Result := TNNetStringList.Create;
  Result.Sorted := false;
  Result.Delimiter := c;
  Result.StrictDelimiter := true;
end;

function RectifiedLinearUnitLeaky(x: TNeuralFloat): TNeuralFloat;
begin
  if x>0
    then Result := x
    else Result := x * 0.01;

  if x<-1 then Result := -1;
end;

function RectifiedLinearUnitLeakyDerivative(x: TNeuralFloat): TNeuralFloat;
begin
  if x>0
    then Result := 1
    else Result := 0.01;

  if x<-1 then Result := 0;
end;

function SignedSquareRoot1(x: TNeuralFloat): TNeuralFloat;
begin
  if x > 1 then
  begin
    Result := Sqrt(x);
  end
  else
  if x < -1 then
  begin
    Result := Sqrt(-x);
  end
  else
  begin
    Result := x;
  end;
end;

function SignedSquareRoot1Derivative(x: TNeuralFloat): TNeuralFloat;
begin
  if x > 1 then
  begin
    Result := 1/(2*Sqrt(x));
  end
  else
  if x < -1 then
  begin
    Result := 1/(2*Sqrt(-x));
  end
  else
  begin
    Result := 1;
  end;
end;

function ReLULeakyBound(x: TNeuralFloat): TNeuralFloat;
begin
  if x>0
    then Result := x
    else Result := x * 0.01;

  Result := NeuronForceRange(Result, 1);
end;

function ReLULeakyBoundDerivative(x: TNeuralFloat): TNeuralFloat;
begin
  if Abs(x)>=1 then
  begin
    Result := 0;
  end
  else
  begin
    if x>0
      then Result := 1
      else Result := 0.01;
  end;
end;

{ NeuralExp is a clone of pas-core-math's pcr_expf with two local changes.
  pcr_expf itself cannot be called from code built with debug checks: its
  bit-twiddling relies on intentional UInt64 wraparound, which raises
  EIntOverflow for ordinary negative inputs when its unit is compiled with
  -Co/-Cr (project-wide checks in Lazarus "Debug" build modes). The clone is
  compiled with checks pushed off, and the x > 88.72 overflow branch builds
  +Inf from its bit pattern instead of a deliberate Single overflow (which
  traps under FPC's default unmasked SSE overflow exception). Every other
  input returns the identical correctly-rounded pcr_expf result. }
{$PUSH}
{$Q-}{$R-}
function NeuralExp(x: TNeuralFloat): TNeuralFloat;
const
  c_exp_0: Double = 0.69314718055994529;
  c_exp_1: Double = 0.24022650695910072;
  c_exp_2: Double = 0.055504108664026088;
  c_exp_3: Double = 0.0096181291075005358;
  c_exp_4: Double = 0.001333362331326638;
  c_exp_5: Double = 0.00015403602972146417;
  b_exp_0: Double = 1;
  b_exp_1: Double = 0.69314718052023927;
  b_exp_2: Double = 0.2402288551437867;
  b_exp_3: Double = 0.055504596827996931;
  tb_exp: array[0..63] of UInt64 = (
    UInt64($3FF0000000000000), UInt64($3FF02C9A3E778061), UInt64($3FF059B0D3158574), UInt64($3FF0874518759BC8),
    UInt64($3FF0B5586CF9890F), UInt64($3FF0E3EC32D3D1A2), UInt64($3FF11301D0125B51), UInt64($3FF1429AAEA92DE0),
    UInt64($3FF172B83C7D517B), UInt64($3FF1A35BEB6FCB75), UInt64($3FF1D4873168B9AA), UInt64($3FF2063B88628CD6),
    UInt64($3FF2387A6E756238), UInt64($3FF26B4565E27CDD), UInt64($3FF29E9DF51FDEE1), UInt64($3FF2D285A6E4030B),
    UInt64($3FF306FE0A31B715), UInt64($3FF33C08B26416FF), UInt64($3FF371A7373AA9CB), UInt64($3FF3A7DB34E59FF7),
    UInt64($3FF3DEA64C123422), UInt64($3FF4160A21F72E2A), UInt64($3FF44E086061892D), UInt64($3FF486A2B5C13CD0),
    UInt64($3FF4BFDAD5362A27), UInt64($3FF4F9B2769D2CA7), UInt64($3FF5342B569D4F82), UInt64($3FF56F4736B527DA),
    UInt64($3FF5AB07DD485429), UInt64($3FF5E76F15AD2148), UInt64($3FF6247EB03A5585), UInt64($3FF6623882552225),
    UInt64($3FF6A09E667F3BCD), UInt64($3FF6DFB23C651A2F), UInt64($3FF71F75E8EC5F74), UInt64($3FF75FEB564267C9),
    UInt64($3FF7A11473EB0187), UInt64($3FF7E2F336CF4E62), UInt64($3FF82589994CCE13), UInt64($3FF868D99B4492ED),
    UInt64($3FF8ACE5422AA0DB), UInt64($3FF8F1AE99157736), UInt64($3FF93737B0CDC5E5), UInt64($3FF97D829FDE4E50),
    UInt64($3FF9C49182A3F090), UInt64($3FFA0C667B5DE565), UInt64($3FFA5503B23E255D), UInt64($3FFA9E6B5579FDBF),
    UInt64($3FFAE89F995AD3AD), UInt64($3FFB33A2B84F15FB), UInt64($3FFB7F76F2FB5E47), UInt64($3FFBCC1E904BC1D2),
    UInt64($3FFC199BDD85529C), UInt64($3FFC67F12E57D14B), UInt64($3FFCB720DCEF9069), UInt64($3FFD072D4A07897C),
    UInt64($3FFD5818DCFBA487), UInt64($3FFDA9E603DB3285), UInt64($3FFDFC97337B9B5F), UInt64($3FFE502EE78B3FF6),
    UInt64($3FFEA4AFA2A490DA), UInt64($3FFEFA1BEE615A27), UInt64($3FFF50765B6E4540), UInt64($3FFFA7C1819E90D8));
  k1_exp: Double = 1.4426950408889634;
  k2_exp: Double = 105553116266496;
  k6_exp: Double = 1.4012984643248171e-45;
  k10_exp: Double = 0.5;
  k11_exp: Double = 1.0;
  k13_exp: Double = 0.0;
  k14_exp: Double = 103.27892990343184;
  k15_exp: Double = 1.0108231726433641e-45;
  k16_exp: Double = 3.5032461608120427e-46;
  k18_exp: Double = 1.45e-10;
  k19_exp: Double = 1.442695040255785;
  k20_exp: Double = 6.3317841895660438e-10;
var
  te: Tb32u32;
  ux_exp: UInt32;
  z_exp: Double;
  a_exp: Double;
  u_exp: Tb64u64;
  sv: Tb64u64;
  ia_exp, h_exp, h2_exp, r_exp: Double;
  ub_exp: Single;
  lb_exp: Single;
  w_exp, s_exp: Double;
begin
  te.f := x;
  //TODO: this is a hack to be fixed
  if (te.u = UInt32($C16912CD)) then
  begin
    te.u := UInt32($34FD331B);
    Result := te.f;
    Exit;
  end;
  ux_exp := te.u shl 1;
  z_exp := x;
  a_exp := k1_exp * z_exp;
  u_exp.f := a_exp + k2_exp;
  if (ux_exp > $8562E42E) or (ux_exp < $6F93813E) then begin
    if ux_exp < $6F93813E then begin  // |x| < 0x1.93813ep-16
      Result := Single(k11_exp + z_exp*(k11_exp + z_exp*k10_exp)); Exit;
    end;
    if ux_exp >= UInt32($FF shl 24) then begin
      if ux_exp > UInt32($FF shl 24) then begin Result := x + x; Exit; end;  // nan
      if (te.u shr 31) <> 0 then Result := k13_exp else Result := x; Exit;  // +-inf
    end;
    if te.u > $C2CE8EC0 then begin  // x < -0x1.9d1d8p+6
      if k6_exp + (z_exp + k14_exp)*k15_exp > k16_exp then
        Result := Single(k6_exp + (z_exp + k14_exp)*k15_exp)
      else
        Result := Single(k16_exp); Exit;
    end;
    if ((te.u shr 31) = 0) and (te.u > $42B17217) then begin  // x > 0x1.62e42ep+6
      // pcr_expf overflows Single(3.4e38*3.4e38) here on purpose; building the
      // same +Inf from its bit pattern avoids the hardware SSE overflow trap.
      te.u := UInt32($7F800000);
      Result := te.f; Exit;
    end;
  end;
  ia_exp := k2_exp - u_exp.f;
  h_exp := a_exp + ia_exp;
  sv.u := tb_exp[u_exp.u and $3F] + ((u_exp.u shr 6) shl 52);
  h2_exp := h_exp * h_exp;
  r_exp := ((b_exp_0 + h_exp*b_exp_1) + h2_exp*(b_exp_2 + h_exp*b_exp_3)) * sv.f;
  ub_exp := Single(r_exp);
  lb_exp := Single(r_exp - r_exp*k18_exp);
  if ub_exp <> lb_exp then begin
    h_exp := (k19_exp*z_exp + ia_exp) + k20_exp*z_exp;
    s_exp := sv.f;
    h2_exp := h_exp * h_exp;
    w_exp := s_exp * h_exp;
    r_exp := s_exp + w_exp*((c_exp_0 + h_exp*c_exp_1) + h2_exp*((c_exp_2 + h_exp*c_exp_3) + h2_exp*(c_exp_4 + h_exp*c_exp_5)));
    ub_exp := Single(r_exp);
  end;
  Result := ub_exp;
end;
{$POP}

// https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
function Sigmoid(x: TNeuralFloat): TNeuralFloat;
var
  S: TNeuralFloat;
begin
  if x > 0 then
  begin
    Result := 1 / ( 1 + NeuralExp(-x) )
  end
  else
  begin
    S := NeuralExp(x);
    Result := S / (1 + S);
  end;
end;

// https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
function SigmoidDerivative(x: TNeuralFloat): TNeuralFloat;
var
  S: TNeuralFloat;
begin
  S := Sigmoid(x);
  Result := S * (1 - S);
end;

function Identity(x: TNeuralFloat): TNeuralFloat;
begin
  Result := x;
end;

function IdentityDerivative(x: TNeuralFloat): TNeuralFloat;
begin
  Result := 1;
end;

function SoftmaxDerivative(x: TNeuralFloat): TNeuralFloat;
begin
  // https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
  // https://github.com/neuroph/neuroph/blob/master/neuroph-2.9/Contrib/src/main/java/org/neuroph/contrib/learning/SoftMax.java
  Result := x*(1-x);
end;

function DiffAct(x: TNeuralFloat): TNeuralFloat;
begin
  NeuronForceRange(x, 1);
  Result := 1 - Abs(x);
end;

function DiffActDerivative(x: TNeuralFloat): TNeuralFloat;
begin
  if ( (x < -1) or (x > 1) ) then
  begin
    Result := 0
  end
  else if (x > 0) then
  begin
    Result := -1
  end
  else Result := 1;
end;

function NeuronForceMinMax(x, pMin, pMax: TNeuralFloat): TNeuralFloat;
begin
  if (x>pMax) then Result := pMax
  else if (x<pMin) then Result := pMin
  else Result := x;
end;

function NeuronForceMinMax(x, pMin, pMax: integer): integer;
begin
  if (x>pMax) then Result := pMax
  else if (x<pMin) then Result := pMin
  else Result := x;
end;

function NeuronForceRange(x, range:TNeuralFloat): TNeuralFloat;
begin
  if (x>range) then Result := range
  else if (x<-range) then Result := -range
  else Result := x;
end;

function NeuronForceMinRange(x, range: TNeuralFloat): TNeuralFloat;
begin
  if (x>0) then
  begin
    Result := Max(x, range);
  end
  else if (x<0) then
  begin
    Result := Min(x, -range);
  end
  else
  begin
    Result := 0;
  end;
end;

// converts rgb to hsv
// ported from https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
// modified make output values to stay in range 0..1.
procedure rgb2hsv(r, g, b: TNeuralFloat; var h, s, v: TNeuralFloat);
var
  min, max, delta: TNeuralFloat;
begin
  {$IFDEF FPC}
  r /= 255;
  g /= 255;
  b /= 255;
  {$ELSE}
  r := r/255;
  g := g/255;
  b := b/255;
  {$ENDIF}

  min := Math.Min( Math.Min(r,g), b);
  max := Math.Max( Math.Max(r,g), b);

  v := max;
  delta := max - min;

  if (delta < 0.00001) then
  begin
    s := 0;
    h := 0; // its now undefined
  end
  else if( max > 0.0 ) then
  begin
    s := (delta / max);

    if( r >= max ) then // > is bogus, just keeps compiler happy
    begin
      h := ( g - b ) / delta; // between yellow & magenta
    end
    else
    begin
      if( g >= max )
        then h := 2.0 + ( b - r ) / delta   // between cyan & yellow
        else h := 4.0 + ( r - g ) / delta;  // between magenta & cyan
    end;

    {$IFDEF FPC}
    h /= 6.0;                               // times 60 = degrees
    {$ELSE}
    h := h / 6.0;
    {$ENDIF}

    if( h < 0.0 ) then h := h + 1.0;
  end
  else
  begin
    s := 0.0;
    h := 0.0; // its now undefined
  end;
end;

// converts hsv to rgb
// ported from https://stackoverflow.com/questions/3018313/algorithm-to-convert-rgb-to-hsv-and-hsv-to-rgb-in-range-0-255-for-both
// modified make input values to stay in range 0..1.
procedure hsv2rgb(h, s, v: TNeuralFloat; var r, g, b: TNeuralFloat);
var
  hh, p, q, t, ff: TNeuralFloat;
  i: integer;
begin
  if (s <= 0.0) then // < is bogus, just shuts up warnings
  begin
    r := v;
    g := v;
    b := v;
  end
  else
  begin
    hh := h*360;
    if (hh >= 360.0) then hh := 0.0;
    hh := hh / 60.0;
    i := Round(hh);
    ff := hh - i;
    p := v * (1.0 - s);
    q := v * (1.0 - (s * ff));
    t := v * (1.0 - (s * (1.0 - ff)));

    case i of
    0:
    begin
      r := v;
      g := t;
      b := p;
    end;
    1:
    begin
      r := q;
      g := v;
      b := p;
    end;
    2:
    begin
      r := p;
      g := v;
      b := t;
    end;
    3:
    begin
      r := p;
      g := q;
      b := v;
    end;
    4:
    begin
      r := t;
      g := p;
      b := v;
    end;
    else
    begin
      r := v;
      g := p;
      b := q;
    end; // of else
    end; // of case

  end; // of if

  {$IFDEF FPC}
  r *= 255;
  g *= 255;
  b *= 255;
  {$ELSE}
  r := r * 255;
  g := g * 255;
  b := b * 255;
  {$ENDIF}

end; // of procedure

{
// ported from https://gist.github.com/mjackson/5311256
* Converts an RGB color value to HSL. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
* Assumes r, g, and b are contained in the set [0, 255] and
* returns h, s, and l in the set [0, 1].
}
procedure rgb2hsl(r, g, b: TNeuralFloat; var h, s, l: TNeuralFloat);
var
  min, max, delta: TNeuralFloat;
begin
  {$IFDEF FPC}
  r /= 255;
  g /= 255;
  b /= 255;
  {$ELSE}
  r := r/255;
  g := g/255;
  b := b/255;
  {$ENDIF}

  min := Math.Min( Math.Min(r,g), b);
  max := Math.Max( Math.Max(r,g), b);

  h := (max + min) / 2;
  s := h;
  l := h;

  if (max = min) then
  begin
    h := 0; // achromatic
    s := 0;
  end
  else
  begin
    delta := max - min;
    if l > 0.5
      then s := delta / (2 - max - min)
      else s := delta / (max + min);

    if r = max then
    begin
      if (g < b)
        then h := (g - b) / (delta + 6)
        else h := (g - b) / delta;
    end
    else if g = max then
    begin
      h := (b - r) / delta + 2;
    end
    else
    begin
      h := (r - g) / delta + 4
    end;

    h := h/6;
  end;
end;

// ported from https://gist.github.com/mjackson/5311256
function hue2rgb(p, q, t: TNeuralFloat): TNeuralFloat;
begin
  if (t < 0) then t := t + 1;
  if (t > 1) then t := t - 1;
  if (t < 1/6) then
  begin
    Result := p + (q - p) * 6 * t;
  end
  else if (t < 1/2) then
  begin
    Result := q;
  end
  else if (t < 2/3) then
  begin
    Result := p + (q - p) * (2/3 - t) * 6;
  end
  else
  begin
    Result := p;
  end;
end;

{
// ported from https://gist.github.com/mjackson/5311256
* Converts an HSL color value to RGB. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
* Assumes h, s, and l are contained in the set [0, 1] and
* returns r, g, and b in the set [0, 255].
}
procedure hsl2rgb(h, s, l: TNeuralFloat; var r, g, b: TNeuralFloat);
var
  p, q: TNeuralFloat;
begin
  if (s = 0) then
  begin
    r := 1; // achromatic
    g := 1;
    b := l;
  end
  else
  begin
    if l < 0.5
      then q := l * (1 + s)
      else q := l + s - l * s;

    p := 2 * l - q;

    r := hue2rgb(p, q, h + 1/3);
    g := hue2rgb(p, q, h);
    b := hue2rgb(p, q, h - 1/3);
  end;

  {$IFDEF FPC}
  r *= 255;
  g *= 255;
  b *= 255;
  {$ELSE}
  r := r * 255;
  g := g * 255;
  b := b * 255;
  {$ENDIF}
end;

// ported from:
// https://github.com/antimatter15/rgb-lab/blob/master/color.js
procedure lab2rgb(l, a, b: TNeuralFloat; var r, g, bb: TNeuralFloat);
var
  x, y, z: TNeuralFloat;
begin
  y := (l + 16) / 116;
  x := a / 500 + y;
  z := y - b / 200;

  if (x * x * x > 0.008856) then
  begin
    x := 0.95047 * x * x * x;
  end
  else
  begin
    x := 0.95047 * (x - 16/116) / 7.787;
  end;

  if (y * y * y > 0.008856) then
  begin
    y := y * y * y;
  end
  else
  begin
    y := (y - 16/116) / 7.787;
  end;

  if (z * z * z > 0.008856) then
  begin
    z := 1.08883 * z * z * z;
  end
  else
  begin
    z := 1.08883 * (z - 16/116) / 7.787;
  end;

  r  := x *  3.2406 + y * -1.5372 + z * -0.4986;
  g  := x * -0.9689 + y *  1.8758 + z *  0.0415;
  bb := x *  0.0557 + y * -0.2040 + z *  1.0570;

  if (r > 0.0031308) then
  begin
    r  := (1.055 * pcr_powf(r, 1/2.4) - 0.055);
  end
  else
  begin
    r  := 12.92 * r;
  end;

  if (g > 0.0031308) then
  begin
    g  := (1.055 * pcr_powf(g, 1/2.4) - 0.055);
  end
  else
  begin
    g  := 12.92 * g;
  end;

  if (bb > 0.0031308) then
  begin
    bb := (1.055 * pcr_powf(bb, 1/2.4) - 0.055);
  end
  else
  begin
    bb := 12.92 * bb;
  end;

  r  := Max(0, Min(1, r)) * 255;
  g  := Max(0, Min(1, g)) * 255;
  bb := Max(0, Min(1, bb)) * 255;
end;


// ported from:
// https://github.com/antimatter15/rgb-lab/blob/master/color.js
procedure rgb2lab(r, g, b: TNeuralFloat; var l, a, bb: TNeuralFloat);
var
  x, y, z: TNeuralFloat;
begin
  {$IFDEF FPC}
  r /= 255;
  g /= 255;
  b /= 255;
  {$ELSE}
  r := r/255;
  g := g/255;
  b := b/255;
  {$ENDIF}

  if (r > 0.04045) then
  begin
    r := pcr_powf((r + 0.055) / 1.055, 2.4)
  end
  else
  begin
    r := r / 12.92;
  end;

  if (g > 0.04045) then
  begin
    g := pcr_powf((g + 0.055) / 1.055, 2.4);
  end
  else
  begin
    g := g / 12.92;
  end;

  if (b > 0.04045) then
  begin
    b := pcr_powf((b + 0.055) / 1.055, 2.4);
  end
  else
  begin
    b := b / 12.92;
  end;

  x := (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
  y := (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
  z := (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

  if (x > 0.008856) then
  begin
    x := pcr_powf(x, 1/3);
  end
  else
  begin
    x := (7.787 * x) + 16/116;
  end;

  if (y > 0.008856) then
  begin
    y := pcr_powf(y, 1/3);
  end
  else
  begin
    y := (7.787 * y) + 16/116;
  end;

  if (z > 0.008856) then
  begin
    z := pcr_powf(z, 1/3);
  end
  else
  begin
    z := (7.787 * z) + 16/116;
  end;

  l  := (116 * y) - 16;
  a  := 500 * (x - y);
  bb := 200 * (y - z);
end;

function RoundAsByte(x: TNeuralFloat): byte;
begin
  Result := Round(Min(Max(x,0),255));
end;

function CompareStringListIntegerAsc(List: TStringList; Index1, Index2: Integer
  ): Integer;
begin
  Result := (PtrInt(List.Objects[Index1]) - PtrInt(List.Objects[Index2]));
end;

function CompareStringListIntegerDesc(List: TStringList; Index1, Index2: Integer
  ): Integer;
begin
  Result := (PtrInt(List.Objects[Index2]) - PtrInt(List.Objects[Index1]));
end;

function CompareNNetVolumeListAsc(const Item1, Item2: TNNetVolume): Integer;
begin
  Result := Item1.Tag - Item2.Tag;
end;

function CompareNNetVolumeListDesc(const Item1, Item2: TNNetVolume): Integer;
begin
  Result := Item2.Tag - Item1.Tag;
end;

function NeuralFloatToStr(V: TNeuralFloat): string;
var
  LocalFormatSettings: TFormatSettings;
begin
  {$IFDEF FPC} LocalFormatSettings := DefaultFormatSettings; {$ENDIF}
  LocalFormatSettings.DecimalSeparator := '.';
  Result := FloatToStr(V,LocalFormatSettings);
end;

function NeuralStrToFloat(V: String): TNeuralFloat;
var
  LocalFormatSettings: TFormatSettings;
begin
  {$IFDEF FPC} LocalFormatSettings := DefaultFormatSettings; {$ENDIF}
  LocalFormatSettings.DecimalSeparator := '.';
  Result := StrToFloat(V,LocalFormatSettings);
end;

procedure AssertFinite(V: TNNetVolume; const Where: string);
var
  I, MaxN: integer;
  Val: TNeuralFloat;
begin
  if V = nil then
    raise Exception.Create('AssertFinite(' + Where + '): volume is nil');
  MaxN := V.Size - 1;
  for I := 0 to MaxN do
  begin
    Val := V.FData[I];
    if IsNan(Val) then
      raise Exception.Create('AssertFinite(' + Where +
        '): non-finite value at index ' + IntToStr(I) +
        ': NaN (' + FloatToStr(Val) + ')');
    if IsInfinite(Val) then
      raise Exception.Create('AssertFinite(' + Where +
        '): non-finite value at index ' + IntToStr(I) +
        ': Inf (' + FloatToStr(Val) + ')');
  end;
end;

procedure RowSoftMax(Row: TNNetVolume);
var
  SizeM1: integer;
  MaxV, Sum: TNeuralFloat;
begin
  SizeM1 := Row.Size - 1;
  if SizeM1 < 0 then exit;
  MaxV := Row.GetMax();
  Row.Sub(MaxV);
  TNNetVolume.VectorExp(Row.DataPtr, Row.DataPtr, Row.Size);
  Sum := Row.GetSum();
  if Sum > 0 then Row.Mul(1 / Sum);
end;

function RowCosineSimilarity(A, B: TNNetVolume): TNeuralFloat;
var
  Dot, NormA, NormB: TNeuralFloat;
begin
  if A.Size <> B.Size then exit(0);
  Dot := TNNetVolume.DotProduct(A.DataPtr, B.DataPtr, A.Size);
  NormA := A.GetSumSqr();
  NormB := B.GetSumSqr();
  if (NormA <= 0) or (NormB <= 0) then
    Result := 0
  else
    Result := Dot / (Sqrt(NormA) * Sqrt(NormB));
end;

procedure NormalizeRowsL2(Mat: TNNetVolume);
var
  Rows, Dim, R: integer;
  RowsM1: integer;
  Norm: TNeuralFloat;
  RowPtr: TNeuralFloatArrPtr;
begin
  Rows := Mat.SizeX;
  Dim := Mat.Depth;
  RowsM1 := Rows - 1;
  for R := 0 to RowsM1 do
  begin
    RowPtr := Mat.GetRawPtr(R * Dim);
    Norm := Sqrt(TNNetVolume.DotProduct(RowPtr, RowPtr, Dim));
    if Norm > 0 then
      TNNetVolume.Mul(RowPtr, 1 / Norm, Dim);
  end;
end;

function NeuralLinearSolve(var A: array of TNeuralFloat;
  var B: array of TNeuralFloat; n, m: integer): boolean;
var
  col, row, piv, k: integer;
  nM1, mM1, rowStart: integer;
  colBaseN, colBaseM, pivBaseN, pivBaseM, rowBaseN, rowBaseM: integer;
  maxAbs, v, factor, diag, tmp: TNeuralFloat;
begin
  Result := True;
  nM1 := n - 1;
  mM1 := m - 1;
  for col := 0 to nM1 do
  begin
    colBaseN := col * n;
    colBaseM := col * m;
    // Partial pivot: pick the row (>= col) with the largest |A[row,col]|.
    piv := col;
    maxAbs := Abs(A[colBaseN + col]);
    rowStart := col + 1;
    for row := rowStart to nM1 do
    begin
      v := Abs(A[row * n + col]);
      if v > maxAbs then begin maxAbs := v; piv := row; end;
    end;
    if maxAbs < 1e-30 then begin Result := False; Exit; end;

    // Swap the pivot row into place (in both A and B).
    if piv <> col then
    begin
      pivBaseN := piv * n;
      pivBaseM := piv * m;
      for k := 0 to nM1 do
      begin
        tmp := A[colBaseN + k]; A[colBaseN + k] := A[pivBaseN + k]; A[pivBaseN + k] := tmp;
      end;
      for k := 0 to mM1 do
      begin
        tmp := B[colBaseM + k]; B[colBaseM + k] := B[pivBaseM + k]; B[pivBaseM + k] := tmp;
      end;
    end;

    // Normalise the pivot row so A[col,col] = 1.
    diag := A[colBaseN + col];
    TNNetVolume.Mul(@A[colBaseN], 1 / diag, n);
    TNNetVolume.Mul(@B[colBaseM], 1 / diag, m);

    // Eliminate the pivot column from every other row.
    for row := 0 to nM1 do
    begin
      if row = col then Continue;
      rowBaseN := row * n;
      rowBaseM := row * m;
      factor := A[rowBaseN + col];
      if factor = 0 then Continue;
      TNNetVolume.MulAdd(@A[rowBaseN], @A[colBaseN], -factor, n);
      TNNetVolume.MulAdd(@B[rowBaseM], @B[colBaseM], -factor, m);
    end;
  end;
end;

function NeuralBoxIoU(AX1, AY1, AX2, AY2,
  BX1, BY1, BX2, BY2: TNeuralFloat): TNeuralFloat;
var
  IX1, IY1, IX2, IY2, IW, IH, Inter, Area1, Area2, UnionA: TNeuralFloat;
begin
  Area1 := Max(0, AX2 - AX1) * Max(0, AY2 - AY1);
  Area2 := Max(0, BX2 - BX1) * Max(0, BY2 - BY1);
  IX1 := Max(AX1, BX1);
  IY1 := Max(AY1, BY1);
  IX2 := Min(AX2, BX2);
  IY2 := Min(AY2, BY2);
  IW := Max(0, IX2 - IX1);
  IH := Max(0, IY2 - IY1);
  Inter := IW * IH;
  UnionA := Area1 + Area2 - Inter;
  if UnionA > 0 then Result := Inter / UnionA else Result := 0;
end;

function NeuralBoxGIoU(AX1, AY1, AX2, AY2,
  BX1, BY1, BX2, BY2: TNeuralFloat): TNeuralFloat;
var
  IX1, IY1, IX2, IY2, IW, IH, Inter, Area1, Area2, UnionA: TNeuralFloat;
  CX1, CY1, CX2, CY2, AreaC, IoU: TNeuralFloat;
begin
  Area1 := Max(0, AX2 - AX1) * Max(0, AY2 - AY1);
  Area2 := Max(0, BX2 - BX1) * Max(0, BY2 - BY1);
  IX1 := Max(AX1, BX1);
  IY1 := Max(AY1, BY1);
  IX2 := Min(AX2, BX2);
  IY2 := Min(AY2, BY2);
  IW := Max(0, IX2 - IX1);
  IH := Max(0, IY2 - IY1);
  Inter := IW * IH;
  UnionA := Area1 + Area2 - Inter;
  if UnionA > 0 then IoU := Inter / UnionA else IoU := 0;
  // Smallest axis-aligned box C enclosing both A and B.
  CX1 := Min(AX1, BX1);
  CY1 := Min(AY1, BY1);
  CX2 := Max(AX2, BX2);
  CY2 := Max(AY2, BY2);
  AreaC := Max(0, CX2 - CX1) * Max(0, CY2 - CY1);
  if AreaC > 0 then
    Result := IoU - (AreaC - UnionA) / AreaC
  else
    Result := 0;
end;

function NeuralGreedyNMS(
  const BX1, BY1, BX2, BY2, Scores: array of TNeuralFloat;
  const Classes: array of integer; Count: integer;
  IoUThreshold: TNeuralFloat): TNeuralIntegerArray;
var
  Order: TNeuralIntegerArray;
  Keep: array of boolean;
  i, jj, oi, oj, tmp, HiCand, KeptCnt: integer;
  jjStart: integer;
  IoU: TNeuralFloat;
begin
  SetLength(Result, 0);
  if Count <= 0 then Exit;
  HiCand := Count - 1;
  // Index permutation sorted by descending score (stable selection sort over
  // the indices - candidate counts in detection are small).
  SetLength(Order, Count);
  for i := 0 to HiCand do Order[i] := i;
  for i := 0 to HiCand do
  begin
    jjStart := i + 1;
    for jj := jjStart to HiCand do
      if Scores[Order[jj]] > Scores[Order[i]] then
      begin tmp := Order[i]; Order[i] := Order[jj]; Order[jj] := tmp; end;
  end;
  // Greedy NMS over the sorted order: a later box is suppressed only by an
  // earlier (higher-score) kept box of the SAME class with IoU > threshold.
  SetLength(Keep, Count);
  for i := 0 to HiCand do Keep[i] := True;
  for i := 0 to HiCand do
  begin
    if not Keep[i] then Continue;
    oi := Order[i];
    jjStart := i + 1;
    for jj := jjStart to HiCand do
    begin
      oj := Order[jj];
      if (not Keep[jj]) or (Classes[oj] <> Classes[oi]) then Continue;
      IoU := NeuralBoxIoU(BX1[oi], BY1[oi], BX2[oi], BY2[oi],
        BX1[oj], BY1[oj], BX2[oj], BY2[oj]);
      if IoU > IoUThreshold then Keep[jj] := False;
    end;
  end;
  // Emit kept original indices in descending-score order.
  SetLength(Result, Count);
  KeptCnt := 0;
  for i := 0 to HiCand do
    if Keep[i] then
    begin
      Result[KeptCnt] := Order[i];
      Inc(KeptCnt);
    end;
  SetLength(Result, KeptCnt);
end;

procedure WriteLnPassIfZero(x: TNeuralFloat; Tolerance: TNeuralFloat=0.0001);
begin
  if Abs(x) < Tolerance
  then WriteLn(' Passed.')
  else WriteLn(' FAILED.');
end;

// Marsaglia & Tsang (2000) "A Simple Method for Generating Gamma Variables".
// Generates a Gamma(Alpha, 1) sample using the repo's global Random RNG.
// Standard normal sample (Marsaglia polar) using the global Random RNG.
function RandomStdNormal(): TNeuralFloat;
var
  r, x, y: TNeuralFloat;
begin
  r := 0;
  while (r > 1) or (r = 0) do
  begin
    x := 2.0 * Random() - 1.0;
    y := 2.0 * Random() - 1.0;
    r := x * x + y * y;
  end;
  Result := x * Sqrt(-2.0 * pcr_logf(r) / r);
end;

function RandomGammaValue(Alpha: TNeuralFloat): TNeuralFloat;
var
  d, c, x, v, u: TNeuralFloat;
begin
  Result := 0;
  if Alpha <= 0 then Exit;
  // Boost: for Alpha < 1 use Gamma(Alpha) = Gamma(Alpha+1) * U^(1/Alpha).
  if Alpha < 1 then
  begin
    u := Random();
    // Guard against log(0) below by avoiding a zero draw.
    while u <= 0 do u := Random();
    Result := RandomGammaValue(Alpha + 1.0) * pcr_powf(u, 1.0 / Alpha);
    Exit;
  end;
  d := Alpha - 1.0 / 3.0;
  c := 1.0 / Sqrt(9.0 * d);
  while True do
  begin
    repeat
      x := RandomStdNormal();
      v := 1.0 + c * x;
    until v > 0;
    v := v * v * v;
    u := Random();
    if u < 1.0 - 0.0331 * (x * x) * (x * x) then
    begin
      Result := d * v;
      Exit;
    end;
    if pcr_logf(u) < 0.5 * x * x + d * (1.0 - v + pcr_logf(v)) then
    begin
      Result := d * v;
      Exit;
    end;
  end;
end;

function RandomBetaValue(Alpha: TNeuralFloat): TNeuralFloat;
var
  ga, gb: TNeuralFloat;
begin
  // Beta(1,1) == Uniform(0,1): fast path and exact.
  if Alpha = 1.0 then
  begin
    Result := Random();
    Exit;
  end;
  ga := RandomGammaValue(Alpha);
  gb := RandomGammaValue(Alpha);
  if ga + gb <= 0
  then Result := 0.5
  else Result := ga / (ga + gb);
end;

procedure MixVolumes(Output, A, B: TNNetVolume; Lambda: TNeuralFloat);
begin
  // Output := Lambda*A + (1-Lambda)*B, reusing AVX-backed volume ops.
  Output.Copy(A);
  Output.Mul(Lambda);
  Output.MulAdd(1.0 - Lambda, B);
end;

function CreateMixedVolumePairList(Original: TNNetVolumePairList;
  Alpha: TNeuralFloat; FixedLambda: TNeuralFloat): TNNetVolumePairList;
var
  Cnt, CntM1, I, J, Tmp, Partner: integer;
  Perm: array of integer;
  Lambda: TNeuralFloat;
  MixedA, MixedB: TNNetVolume;
  PartnerPair: TNNetVolumePair;
begin
  Result := TNNetVolumePairList.Create();
  if Original = nil then Exit;
  Cnt := Original.Count;
  if Cnt = 0 then Exit;
  CntM1 := Cnt - 1;

  // Build a random derangement-ish permutation (Fisher-Yates) so each sample
  // is paired with another sample from the same list (minibatch mixup).
  SetLength(Perm, Cnt);
  for I := 0 to CntM1 do Perm[I] := I;
  for I := CntM1 downto 1 do
  begin
    J := Random(I + 1);
    Tmp := Perm[I]; Perm[I] := Perm[J]; Perm[J] := Tmp;
  end;

  for I := 0 to CntM1 do
  begin
    Partner := Perm[I];
    PartnerPair := Original[Partner];
    if FixedLambda >= 0
    then Lambda := FixedLambda
    else Lambda := RandomBetaValue(Alpha);

    MixedA := TNNetVolume.Create();
    MixedB := TNNetVolume.Create();
    MixVolumes(MixedA, Original[I].A, PartnerPair.A, Lambda);
    MixVolumes(MixedB, Original[I].B, PartnerPair.B, Lambda);
    // TNNetVolumePair.Create takes ownership of the volumes.
    Result.Add(TNNetVolumePair.Create(MixedA, MixedB));
  end;
end;

procedure ComputeCutMixBox(W, H: integer;
  Lambda, CenterFracX, CenterFracY: TNeuralFloat;
  out X0, Y0, BoxW, BoxH: integer);
var
  CutRatio: TNeuralFloat;
  CutW, CutH, Cx, Cy, X1, Y1: integer;
begin
  X0 := 0; Y0 := 0; BoxW := 0; BoxH := 0;
  if (W <= 0) or (H <= 0) then Exit;
  // Standard CutMix rand_bbox: cut size proportional to sqrt(1 - lambda).
  if Lambda < 0 then Lambda := 0;
  if Lambda > 1 then Lambda := 1;
  CutRatio := Sqrt(1.0 - Lambda);
  CutW := Round(CutRatio * W);
  CutH := Round(CutRatio * H);
  // Uniform center, then clamp the corners to the image bounds.
  Cx := Round(CenterFracX * W);
  Cy := Round(CenterFracY * H);
  X0 := Cx - CutW div 2;
  Y0 := Cy - CutH div 2;
  X1 := Cx + (CutW - CutW div 2);
  Y1 := Cy + (CutH - CutH div 2);
  if X0 < 0 then X0 := 0;
  if Y0 < 0 then Y0 := 0;
  if X1 > W then X1 := W;
  if Y1 > H then Y1 := H;
  if X1 < X0 then X1 := X0;
  if Y1 < Y0 then Y1 := Y0;
  BoxW := X1 - X0;
  BoxH := Y1 - Y0;
end;

function CreateCutMixVolumePairList(Original: TNNetVolumePairList;
  Alpha: TNeuralFloat; FixedLambda: TNeuralFloat): TNNetVolumePairList;
var
  Cnt, CntM1, I, J, Tmp, Partner: integer;
  Perm: array of integer;
  Lambda, LambdaAdj: TNeuralFloat;
  X0, Y0, BoxW, BoxH, X, Y, D, W, H, DepthMax, XMax, YMax: integer;
  PastePos: integer;
  CutA, MixedB: TNNetVolume;
  SrcA, SrcB: TNNetVolume;
  PartnerPair: TNNetVolumePair;
begin
  Result := TNNetVolumePairList.Create();
  if Original = nil then Exit;
  Cnt := Original.Count;
  if Cnt = 0 then Exit;
  CntM1 := Cnt - 1;

  // Random partner permutation (Fisher-Yates) -> minibatch CutMix.
  SetLength(Perm, Cnt);
  for I := 0 to CntM1 do Perm[I] := I;
  for I := CntM1 downto 1 do
  begin
    J := Random(I + 1);
    Tmp := Perm[I]; Perm[I] := Perm[J]; Perm[J] := Tmp;
  end;

  for I := 0 to CntM1 do
  begin
    Partner := Perm[I];
    PartnerPair := Original[Partner];
    SrcA := Original[I].A;
    SrcB := PartnerPair.A;
    if FixedLambda >= 0
    then Lambda := FixedLambda
    else Lambda := RandomBetaValue(Alpha);

    W := SrcA.SizeX;
    H := SrcA.SizeY;
    ComputeCutMixBox(W, H, Lambda, Random(), Random(), X0, Y0, BoxW, BoxH);

    // Start from a copy of this sample's input, then paste the partner's box.
    CutA := TNNetVolume.Create();
    CutA.Copy(SrcA);
    // Only paste when the partner shares the same XY/depth geometry; otherwise
    // fall back to lambda=1 (no paste) so mismatched shapes are still safe.
    if (SrcB.SizeX = W) and (SrcB.SizeY = H) and (SrcB.Depth = SrcA.Depth) then
    begin
      DepthMax := SrcA.Depth - 1;
      XMax := X0 + BoxW - 1;
      YMax := Y0 + BoxH - 1;
      for X := X0 to XMax do
        for Y := Y0 to YMax do
        begin
          // CutA (a copy of SrcA) and SrcB share XY/depth geometry here, so a
          // single base indexes both FData arrays.
          PastePos := CutA.GetRawPos(X, Y, 0);
          Move(SrcB.FData[PastePos], CutA.FData[PastePos], (DepthMax + 1) * csNeuralFloatSize);
        end;
      // True pasted-area fraction after clamping.
      LambdaAdj := 1.0 - (BoxW * BoxH) / (W * H);
    end
    else
      LambdaAdj := 1.0;

    // Mix targets by the actual pasted-area fraction.
    MixedB := TNNetVolume.Create();
    MixVolumes(MixedB, Original[I].B, PartnerPair.B, LambdaAdj);

    // TNNetVolumePair.Create takes ownership of the volumes.
    Result.Add(TNNetVolumePair.Create(CutA, MixedB));
  end;
end;

// https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
// Expected result is:
// [[ 0.          1.          0.          1.        ]
//  [ 0.84147098  0.54030231  0.09983342  0.99500417]
//  [ 0.90929743 -0.41614684  0.19866933  0.98006658]
//  [ 0.14112001 -0.9899925   0.29552021  0.95533649]]
procedure TestTNNetVolumePositionalEncoding;
var
  X: TNNetVolume;
begin
  X := TNNetVolume.Create(4,1,4);
  X.PositionalEncoding(100);
  X.Print();
  X.Free;
  readln;
end;

procedure TestTNNetVolume();
var
  TestSize: integer;
  I, SizeMax: integer;
  Result, Aux: TNeuralFloat;
  Min0, Max0, Min1, Max1, Min2, Max2: TNeuralFloat;
  A, B: TNNetVolume;
  R: TNNetGroupedVolume;
begin
  TestSize := 1+Random(2630);
  WriteLn(' TestTNNetVolume Testing size:', TestSize);
  A := TNNetVolume.Create(TestSize);
  B := TNNetVolume.Create(TestSize);
  R := TNNetGroupedVolume.Create(TestSize);

  A.Randomize();
  B.Randomize();

  R.Fill(2);

  Write('Fill/Inner sum:', (R.GetSum() - 2*TestSize));
  WriteLnPassIfZero(R.GetSum() - 2*TestSize);

  R.Copy(A);
  R.Add(B);
  SizeMax := A.Size - 1;
  Result := 0;
  for I := 0 to SizeMax do
  begin
    Result := Result + Abs( R.Raw[I] - (A.Raw[I]+B.Raw[I]) );
  end;
  Write(' A + B:',Result);
  WriteLnPassIfZero(Result);

  Result := 0;
  for I := 0 to SizeMax do
  begin
    Result := Result + ( A.Raw[I] * B.Raw[I]);
  end;
  Write(' A . B:',Result - A.DotProduct(B));
  WriteLnPassIfZero(Result - A.DotProduct(B));

  R.Copy(A);
  R.Sub(B);
  Result := 0;
  for I := 0 to SizeMax do
  begin
    Result := Result + Abs( A.Raw[I] - B.Raw[I] );
  end;
  Write(' Sum( Abs(A - B) ):', Result - A.SumDiff(B),' ',Result,' ',A.SumDiff(B));
  WriteLnPassIfZero(Result - A.SumDiff(B));

  R.Copy(A);
  R.Sub(B);
  Result := 0;
  for I := 0 to SizeMax do
  begin
    Result := Result + Sqr( A.Raw[I] - B.Raw[I] );
  end;
  Write(' Sum( Sqr(A - B) ):', Result - A.GetDistanceSqr(B),' ',Result,' ',A.GetDistanceSqr(B));
  WriteLnPassIfZero(Result - A.GetDistanceSqr(B));

  R.Copy(A);
  R.Mul(B);
  Result := 0;
  for I := 0 to SizeMax do
  begin
    Result := Result + Abs( R.Raw[I] - (A.Raw[I]*B.Raw[I]) );
  end;
  Write(' A * B:',Result);
  WriteLnPassIfZero(Result);

  Result := 0;
  R.Randomize();
  for I := 0 to SizeMax do
  begin
    R.Raw[I] := Abs(R.Raw[I]);
    Result := Result + R.Raw[I];
  end;
  Write(' Inner Sum(A):', (Result - R.GetSum()),' ', Result,' ', R.GetSum());
  WriteLnPassIfZero(Result - R.GetSum());

  Result := 0;
  for I := 0 to SizeMax do
  begin
    Result := Result + Sqr(R.Raw[I]);
  end;
  Write(' Inner SumSqr(A):', (Result - R.GetSumSqr()),' ', Result,' ', R.GetSumSqr());
  WriteLnPassIfZero(Result - R.GetSumSqr());

  Result := 0;
  A.Randomize();
  R.Copy(A);
  R.Mul(3);
  for I := 0 to SizeMax do
  begin
    Result := Result + Abs( R.Raw[I] - 3*A.Raw[I] );
  end;
  Write(' A * 3:', Result);
  WriteLnPassIfZero(Result);

  R.Copy(A);
  R.MulAdd(3,B);
  Result := 0;
  for I := 0 to SizeMax do
  begin
    Aux := Abs( R.Raw[I] - (A.Raw[I]+ 3*B.Raw[I]) );
    Result := Result + Aux;
    if Aux > 0.0001 then
    begin
      WriteLn(' A + 3B ERROR: ',I,' : ',Aux, ' :: ', R.Raw[I], ' ?= ', A.Raw[I], ' + 3*', B.Raw[I] );
    end;
  end;
  Write(' A + 3B:',Result);
  WriteLnPassIfZero(Result);

  R.Copy(A);
  R.MulMulAdd(2,3,B);
  Result := 0;
  for I := 0 to SizeMax do
  begin
    Result := Result + Abs( R.Raw[I] - (2*A.Raw[I] + 3*B.Raw[I]) );
  end;
  Write(' 2A + 3B:',Result);
  WriteLnPassIfZero(Result);

  R.Fill(10);
  TNNetVolume.MulAdd(R.DataPtr, A.DataPtr, B.DataPtr, R.Size);
  Result := 0;
  for I := 0 to SizeMax do
  begin
    Result := Result + Abs( R.Raw[I] - (10 + A.Raw[I] * B.Raw[I]) );
  end;
  Write(' R += A * B:',Result);
  WriteLnPassIfZero(Result);

  WriteLn('Channel testing:');
  A.Resize(32, 32, 3);
  A.Fill(0);
  B.Resize(1, 1, 3);
  B.Define([2.0,3.0,4.0]);
  A.AddToChannels(B);
  A.GetMinMaxAtDepth(0, Min0, Max0);
  A.GetMinMaxAtDepth(1, Min1, Max1);
  A.GetMinMaxAtDepth(2, Min2, Max2);

  Write
  (
    'Min/Max at 0:',  Min0:4:0,' ',Max0:4:0,
    ' at 1:', Min1:4:0,' ',Max1:4:0,
    ' at 2:', Min2:4:0,' ',Max2:4:0
  );
  WriteLnPassIfZero(Abs(Min0-2)+Abs(Min1-3)+Abs(Min2-4));

  B.Fill(0);
  B.AddSumChannel(A);
  B.Print();

  WriteLn('Interleave testing:');
  A.Resize(1,1,12);
  A.FillForDebug();
  A.Mul(100);
  Write(' Original:'); A.Print();
  B.InterleaveWithDepthFrom(A,2);
  Write(' Interleave With Depth 2:'); B.Print();
  R.DeInterleaveWithDepthFrom(B,2);
  Write(' DeInterleave B With Depth 2:');R.Print();
  B.InterleaveWithXFrom(A,2);
  Write(' Interleave With X 2:');B.Print();
  R.DeInterleaveWithXFrom(B,2);
  Write(' DeInterleave B With X 2:');R.Print();
  A.Resize(1,1,128);
  B.Resize(A);
  A.FillForDebug();
  B.Copy(A);

  Write('Pearson Correlation (A,A):', A.PearsonCorrelation(B) );
  WriteLnPassIfZero(A.PearsonCorrelation(B)-1);
  B.Mul(-1);
  Write('Pearson Correlation (A,-A):', A.PearsonCorrelation(B) );
  WriteLnPassIfZero(A.PearsonCorrelation(B)+1);
  B.Randomize();
  Write('Pearson Correlation (A,Random):', A.PearsonCorrelation(B) );
  WriteLnPassIfZero(A.PearsonCorrelation(B), 0.1);

  // Testing Grouped Dot Product
  // 2 vectors of 2 elements
  A.Resize(1,2,2); // 0 1 and 2 3
  // 2 vectors of 4 elements
  B.Resize(1,2,4); // 0 1 2 3 and 4 5 6 7
  // 1 resulting vector with 4 elements
  R.Resize(1,1,4);

  A.FillForDebug();
  B.FillForDebug();
  A.Mul(100);
  B.Mul(100);
  Write('Grouped dot product result:');
  R.GroupedDotProductsTiled({Groups=}2, {NumAs=}2, {NumBs=}2,{VectorSize=}2, A, B, {TileSizeA=}1, {TileSizeB=}1);
  //R.Print();
  WriteLnPassIfZero(R.GetSum() -1 -5 -13 -33);
  R.Free;
  B.Free;
  A.Free;
  WriteLn('TestTNNetVolume has finished.');
end;

procedure TestKMeans();
var
  KMeans: TNNetKMeans;
  Clusters, ClusterSize, Samples, SamplesM1: integer;
  SampleCnt, StepCnt, ClusterCnt: integer;
  SampleVolume: TNNetVolume;
  ClustersWithElements: integer;
  ClusteredElements: integer;
  ClusterMax: integer;
begin
  Clusters := Random(128) + 1;
  ClusterSize := Random(128) + 1;
  Samples := Random(1280) + 1;
  WriteLn('Testing KMeans - Clusters:', Clusters, ' Cluster Size:', ClusterSize,
    ' Samples:', Samples);
  KMeans := TNNetKMeans.Create(Clusters, 1, 1, ClusterSize);
  // Creates the sample for clustering.
  SamplesM1 := Samples - 1;
  for SampleCnt := 0 to SamplesM1 do
  begin
    SampleVolume := TNNetVolume.Create(1, 1, ClusterSize);
    SampleVolume.FillForDebug();
    SampleVolume.Mul(Random(Clusters));
    SampleVolume.Add(Random(100)/100);
    KMeans.AddSample( SampleVolume );
  end;
  // Runs the clusteting.
  KMeans.Randomize();
  for StepCnt := 1 to 20 do
  begin
    KMeans.RunStep();
    KMeans.RandomizeEmptyClusters();
  end;
  KMeans.RunStep(False);
  // Counts how many clusters have elements.
  ClustersWithElements := 0;
  ClusteredElements := 0;
  ClusterMax := KMeans.Clusters.Count - 1;
  for ClusterCnt := 0 to ClusterMax do
  begin
    if KMeans.Clusters[ClusterCnt].Tag > 0 then Inc(ClustersWithElements);
    Inc(ClusteredElements, KMeans.Clusters[ClusterCnt].Tag);
  end;
  Write(ClustersWithElements, ' clusters have ', ClusteredElements,
    ' elements.');
  WriteLnPassIfZero(ClusteredElements-Samples);

  WriteLn('KMeans testing has finished.');
  KMeans.Free;
end;

function GetDefaultNumericFormat: TFormatSettings;
begin
  {$IFDEF FPC} Result := DefaultFormatSettings; {$ENDIF}
  Result.DecimalSeparator := '.';
end;

function CreateQuotedTokenizedStringList(Str: string; Separator:char; QuoteChar: char): TNNetStringList;
begin
  Result := CreateQuotedTokenizedStringList(Separator, QuoteChar);
  Result.DelimitedText := Str;
end;

function CreateQuotedTokenizedStringList(Separator:char; QuoteChar: char): TNNetStringList;
begin
  Result := CreateTokenizedStringList(Separator);
  Result.QuoteChar := QuoteChar;
end;

function HiperbolicTangent(x: TNeuralFloat): TNeuralFloat;
var
  exp2x: TNeuralFloat;
begin
  x := NeuronForceRange(x, 10);
  exp2x := exp(-2 * x);
  Result := (1 - exp2x) / (1 + exp2x);
end;

function HiperbolicTangentDerivative(x: TNeuralFloat): TNeuralFloat;
begin
  Result := 1 - sqr(HiperbolicTangent(x));
end;

function RectifiedLinearUnit(x: TNeuralFloat): TNeuralFloat;
begin
  if x>0
    then Result := x
    else Result := 0;
end;

function RectifiedLinearUnitDerivative(x: TNeuralFloat): TNeuralFloat;
begin
  if x>0
    then Result := 1
    else Result := 0;
end;

function Swish(x: TNeuralFloat): TNeuralFloat;
begin
  Result := x / ( 1 + NeuralExp(-x) );
end;

function SwishDerivative(x: TNeuralFloat): TNeuralFloat;
var
  SigmoidValue, OutputValue: TNeuralFloat;
begin
  SigmoidValue := 1 / ( 1 + NeuralExp(-x) ); {Swish(x)}
  OutputValue := x * SigmoidValue;
  Result :=  OutputValue + SigmoidValue * (1-OutputValue);
end;

// https://paperswithcode.com/method/hard-swish
function HardSwish(x: TNeuralFloat): TNeuralFloat;
begin
  if x > 3 then
  begin
    Result := x;
  end
  else if x < -3 then
  begin
    Result := 0;
  end
  else
  begin
    Result := x*(x + 3)/6;
  end;
end;

function HardSwishDerivative(x: TNeuralFloat): TNeuralFloat;
begin
  if x<-3 then
  begin
    Result := 0;
  end
  else if x>3 then
  begin
    Result := 1;
  end
  else
  begin
    Result := 0.3333*x + 0.5;
  end;
end;

procedure QuickSortTokenArray(var A: TNNetTokenArray; iLo, iHi: Integer);
var
  Lo, Hi: Integer;
  Mid, T: TNNetToken;
begin
  Lo := iLo;
  Hi := iHi;
  Mid := A[(Lo + Hi) div 2];
  repeat
    while A[Lo].Score > Mid.Score do Inc(Lo);
    while A[Hi].Score < Mid.Score do Dec(Hi);
    if Lo <= Hi then
    begin
      T := A[Lo];
      A[Lo] := A[Hi];
      A[Hi] := T;
      Inc(Lo);
      Dec(Hi);
    end;
  until Lo > Hi;
  if Hi > iLo then QuickSortTokenArray(A, iLo, Hi);
  if Lo < iHi then QuickSortTokenArray(A, Lo, iHi);
end;

{ TNNetSamplerTopP }

constructor TNNetSamplerTopP.Create(TopP: TNeuralFloat);
begin
  inherited Create();
  FTopP := TopP;
end;

function TNNetSamplerTopP.GetToken(Origin: TNNetVolume): integer;
var
  CumulativeSum: TNeuralFloat;
  I, Threshold, Hi, Lo: Integer;
begin
  Origin.GetTokenArray(FTokenArr);
  SortTokenArray();
  CumulativeSum := 0;
  Threshold := 0;
  Hi := High(FTokenArr);
  Lo := Low(FTokenArr);
  for I := Lo to Hi do
  begin
    CumulativeSum := CumulativeSum + FTokenArr[i].Score;
    if CumulativeSum > FTopP then
    begin
      Threshold := I;
      Break;
    end;
  end;

  // Randomly select one of the top tokens within the threshold.
  if Threshold > 0 then
    Result := FTokenArr[Random(Threshold)].Token
  else
    Result := FTokenArr[0].Token; // Fallback in case P is too low.
end;

function TNNetSamplerTopP.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
var
  CumulativeSum: TNeuralFloat;
  I, Threshold, Hi, Lo: Integer;
begin
  Origin.GetTokenArrayOnPixel(FTokenArr, PixelX, PixelY);
  SortTokenArray();
  CumulativeSum := 0;
  Threshold := 0;
  Hi := High(FTokenArr);
  Lo := Low(FTokenArr);
  for I := Lo to Hi do
  begin
    CumulativeSum := CumulativeSum + FTokenArr[i].Score;
    if CumulativeSum > FTopP then
    begin
      Threshold := I;
      Break;
    end;
  end;

  // Randomly select one of the top tokens within the threshold.
  if Threshold > 0 then
    Result := FTokenArr[Random(Threshold)].Token
  else
    Result := FTokenArr[0].Token; // Fallback in case P is too low.
end;

{ TNNetSamplerMinP }

constructor TNNetSamplerMinP.Create(MinP: TNeuralFloat);
begin
  inherited Create();
  FMinP := MinP;
end;

function TNNetSamplerMinP.SampleFromSorted(): integer;
var
  Threshold, KeptSum, Roll, Cumulative: TNeuralFloat;
  I, KeptCount, KeptCountM1, Hi, Lo: integer;
begin
  if Length(FTokenArr) = 0 then
  begin
    Result := 0; // defensive: empty distribution
    exit;
  end;
  // FTokenArr is sorted DESCENDING, so [0] holds the max probability.
  Threshold := FMinP * FTokenArr[0].Score;
  KeptCount := 0;
  KeptSum := 0;
  Hi := High(FTokenArr);
  Lo := Low(FTokenArr);
  for I := Lo to Hi do
  begin
    if FTokenArr[I].Score >= Threshold then
    begin
      Inc(KeptCount);
      KeptSum := KeptSum + FTokenArr[I].Score;
    end
    else Break; // sorted descending: nothing later can pass the cut
  end;
  if (KeptCount = 0) or (KeptSum <= 0) then
  begin
    Result := FTokenArr[0].Token; // fallback: degenerate distribution
    exit;
  end;
  // Weighted draw proportional to the renormalized kept mass.
  Roll := Random * KeptSum;
  Cumulative := 0;
  KeptCountM1 := KeptCount - 1;
  Result := FTokenArr[KeptCountM1].Token; // numeric-safety fallback
  for I := 0 to KeptCountM1 do
  begin
    Cumulative := Cumulative + FTokenArr[I].Score;
    if Roll < Cumulative then
    begin
      Result := FTokenArr[I].Token;
      exit;
    end;
  end;
end;

function TNNetSamplerMinP.GetToken(Origin: TNNetVolume): integer;
begin
  Origin.GetTokenArray(FTokenArr);
  SortTokenArray();
  Result := SampleFromSorted();
end;

function TNNetSamplerMinP.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  Origin.GetTokenArrayOnPixel(FTokenArr, PixelX, PixelY);
  SortTokenArray();
  Result := SampleFromSorted();
end;

{ TNNetSamplerTopK }

constructor TNNetSamplerTopK.Create(TopK: integer);
begin
  inherited Create();
  FTopK := TopK;
end;

function TNNetSamplerTopK.GetToken(Origin: TNNetVolume): integer;
begin
  Origin.GetTokenArray(FTokenArr);
  SortTokenArray();
  Result := FTokenArr[Random(FTopK)].Token;
end;

function TNNetSamplerTopK.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  Origin.GetTokenArrayOnPixel(FTokenArr, PixelX, PixelY);
  SortTokenArray();
  Result := FTokenArr[Random(FTopK)].Token;
end;

{ TNNetSamplerWeightedTopK }

constructor TNNetSamplerWeightedTopK.Create(TopK: integer);
begin
  inherited Create();
  FTopK := TopK;
end;

function TNNetSamplerWeightedTopK.SampleFromSorted(): integer;
var
  KeptSum, Roll, Cumulative: TNeuralFloat;
  I, KeptCount, KeptCountM1: integer;
begin
  if Length(FTokenArr) = 0 then
  begin
    Result := 0; // defensive: empty distribution
    exit;
  end;
  // FTokenArr is sorted DESCENDING, so [0..KeptCount-1] are the top-K tokens.
  KeptCount := FTopK;
  if (KeptCount <= 0) or (KeptCount > Length(FTokenArr)) then
    KeptCount := Length(FTokenArr); // <=0 or >=vocab => whole row
  KeptSum := 0;
  KeptCountM1 := KeptCount - 1;
  for I := 0 to KeptCountM1 do
    KeptSum := KeptSum + FTokenArr[I].Score;
  if KeptSum <= 0 then
  begin
    Result := FTokenArr[0].Token; // fallback: degenerate distribution
    exit;
  end;
  // Weighted draw proportional to the renormalized kept mass.
  Roll := Random * KeptSum;
  Cumulative := 0;
  Result := FTokenArr[KeptCountM1].Token; // numeric-safety fallback
  for I := 0 to KeptCountM1 do
  begin
    Cumulative := Cumulative + FTokenArr[I].Score;
    if Roll < Cumulative then
    begin
      Result := FTokenArr[I].Token;
      exit;
    end;
  end;
end;

function TNNetSamplerWeightedTopK.GetToken(Origin: TNNetVolume): integer;
begin
  Origin.GetTokenArray(FTokenArr);
  SortTokenArray();
  Result := SampleFromSorted();
end;

function TNNetSamplerWeightedTopK.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  Origin.GetTokenArrayOnPixel(FTokenArr, PixelX, PixelY);
  SortTokenArray();
  Result := SampleFromSorted();
end;

{ TNNetSamplerTypical }

constructor TNNetSamplerTypical.Create(Mass: TNeuralFloat);
begin
  inherited Create();
  FMass := Mass;
end;

function TNNetSamplerTypical.SampleTypical(): integer;
var
  Entropy, P, Surprise, KeptSum, Roll, Cumulative: TNeuralFloat;
  Dist: array of TNeuralFloat; // |surprise - entropy| per FTokenArr entry
  Order: array of integer;     // FTokenArr indices sorted by ascending Dist
  I, J, KeptCount, KeptCountM1, Tmp, N, NM1, NM2, JStart: integer;
begin
  N := Length(FTokenArr);
  if N = 0 then
  begin
    Result := 0; // defensive: empty distribution
    exit;
  end;
  NM1 := N - 1;
  NM2 := N - 2;
  // Conditional (Shannon) entropy of the row, in nats.
  Entropy := 0;
  for I := 0 to NM1 do
  begin
    P := FTokenArr[I].Score;
    if P > 0 then Entropy := Entropy - P * pcr_logf(P);
  end;
  // Per-token distance |(-log p) - H|.
  SetLength(Dist, N);
  SetLength(Order, N);
  for I := 0 to NM1 do
  begin
    P := FTokenArr[I].Score;
    if P > 0 then Surprise := -pcr_logf(P) else Surprise := 1e30; // p=0 => infinite
    Dist[I] := Abs(Surprise - Entropy);
    Order[I] := I;
  end;
  // Selection sort of Order by ascending Dist (vocab-sized but only run once
  // per step; mirrors the simple sort style used elsewhere in this unit).
  for I := 0 to NM2 do
  begin
    JStart := I + 1;
    for J := JStart to NM1 do
      if Dist[Order[J]] < Dist[Order[I]] then
      begin
        Tmp := Order[I]; Order[I] := Order[J]; Order[J] := Tmp;
      end;
  end;
  // Smallest prefix (by ascending distance) whose cumulative mass reaches FMass.
  KeptCount := 0;
  KeptSum := 0;
  for I := 0 to NM1 do
  begin
    Inc(KeptCount);
    KeptSum := KeptSum + FTokenArr[Order[I]].Score;
    if KeptSum >= FMass then Break;
  end;
  if (KeptCount = 0) or (KeptSum <= 0) then
  begin
    Result := FTokenArr[Order[0]].Token; // fallback: degenerate distribution
    exit;
  end;
  // Weighted draw proportional to the renormalized kept mass.
  Roll := Random * KeptSum;
  Cumulative := 0;
  KeptCountM1 := KeptCount - 1;
  Result := FTokenArr[Order[KeptCountM1]].Token; // numeric-safety fallback
  for I := 0 to KeptCountM1 do
  begin
    Cumulative := Cumulative + FTokenArr[Order[I]].Score;
    if Roll < Cumulative then
    begin
      Result := FTokenArr[Order[I]].Token;
      exit;
    end;
  end;
end;

function TNNetSamplerTypical.GetToken(Origin: TNNetVolume): integer;
begin
  Origin.GetTokenArray(FTokenArr);
  Result := SampleTypical();
end;

function TNNetSamplerTypical.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  Origin.GetTokenArrayOnPixel(FTokenArr, PixelX, PixelY);
  Result := SampleTypical();
end;

{ TNNetSamplerMirostat }

constructor TNNetSamplerMirostat.Create(Tau: TNeuralFloat; Eta: TNeuralFloat;
  Version: TNNetMirostatVersion);
begin
  inherited Create();
  FTau := Tau;
  FEta := Eta;
  FVersion := Version;
  FMu := 2 * FTau; // paper init; Reset() re-arms it per generation
end;

procedure TNNetSamplerMirostat.Reset();
begin
  FMu := 2 * FTau;
end;

function TNNetSamplerMirostat.SampleAndUpdate(): integer;
var
  KeptSum, Roll, Cumulative, P, Surprise: TNeuralFloat;
  SumLogP, SumLogRank, SumLogPLogRank, SumLogRankSq, LogRank, LogP: TNeuralFloat;
  S, Epsilon, KFloat, ChosenScore: TNeuralFloat;
  I, KeptCount, KeptCountM1, N, NM1, NumFit, NumFitM2, K: integer;
begin
  N := Length(FTokenArr);
  if N = 0 then
  begin
    Result := 0; // defensive
    exit;
  end;
  NM1 := N - 1;
  // FTokenArr is sorted DESCENDING: [0] is the max probability.
  if FVersion = mvV2 then
  begin
    // v2: keep every token with surprise -log p <= Mu.
    KeptCount := 0;
    KeptSum := 0;
    for I := 0 to NM1 do
    begin
      P := FTokenArr[I].Score;
      if P <= 0 then Break; // descending: nothing later is larger
      if -pcr_logf(P) <= FMu then
      begin
        Inc(KeptCount);
        KeptSum := KeptSum + P;
      end
      else Break; // surprise grows as p shrinks => monotone in this sorted order
    end;
    if KeptCount = 0 then
    begin
      KeptCount := 1; // always keep the most-likely token
      KeptSum := FTokenArr[0].Score;
    end;
  end
  else
  begin
    // v1: estimate Zipf exponent s from the head of the distribution, then a
    // target truncation size k = ((eps * 2^Mu) / (1 - N^(-eps)))^(1/s).
    NumFit := N;
    if NumFit > 100 then NumFit := 100; // fit on the head (paper uses ~100)
    SumLogP := 0; SumLogRank := 0; SumLogPLogRank := 0; SumLogRankSq := 0;
    K := 0;
    NumFitM2 := NumFit - 2;
    for I := 0 to NumFitM2 do
    begin
      P := FTokenArr[I].Score;
      if (P <= 0) or (FTokenArr[I + 1].Score <= 0) then Break;
      // t_i = log(p_i / p_{i+1}) regressed on log((i+2)/(i+1)) gives s.
      LogP := pcr_logf(P / FTokenArr[I + 1].Score);
      LogRank := pcr_logf((I + 2) / (I + 1));
      SumLogP := SumLogP + LogP;
      SumLogRank := SumLogRank + LogRank;
      SumLogPLogRank := SumLogPLogRank + LogP * LogRank;
      SumLogRankSq := SumLogRankSq + LogRank * LogRank;
      Inc(K);
    end;
    if (K > 0) and (SumLogRankSq > 0) then
      S := SumLogPLogRank / SumLogRankSq
    else
      S := 1.0;
    if S <= 0 then S := 1.0;
    Epsilon := S - 1.0;
    // k from the paper's closed form; clamp into [1, N].
    if Abs(Epsilon) < 1e-6 then
      KFloat := Exp(FMu)            // s ~ 1 limit
    else
      KFloat := Exp( pcr_logf( (Epsilon * Exp(FMu * pcr_logf(2.0))) /
                         (1 - pcr_powf(N, -Epsilon)) ) / S );
    if KFloat < 1 then KFloat := 1;
    KeptCount := Round(KFloat);
    if KeptCount < 1 then KeptCount := 1;
    if KeptCount > N then KeptCount := N;
    KeptSum := 0;
    KeptCountM1 := KeptCount - 1;
    for I := 0 to KeptCountM1 do
      KeptSum := KeptSum + FTokenArr[I].Score;
  end;

  if KeptSum <= 0 then
  begin
    Result := FTokenArr[0].Token;
    ChosenScore := FTokenArr[0].Score;
  end
  else
  begin
    // Weighted draw proportional to the renormalized kept mass.
    Roll := Random * KeptSum;
    Cumulative := 0;
    KeptCountM1 := KeptCount - 1;
    Result := FTokenArr[KeptCountM1].Token;        // numeric-safety fallback
    ChosenScore := FTokenArr[KeptCountM1].Score;
    for I := 0 to KeptCountM1 do
    begin
      Cumulative := Cumulative + FTokenArr[I].Score;
      if Roll < Cumulative then
      begin
        Result := FTokenArr[I].Token;
        ChosenScore := FTokenArr[I].Score;
        Break;
      end;
    end;
  end;
  // Feedback update: drive observed surprise toward Tau.
  if ChosenScore > 0 then Surprise := -pcr_logf(ChosenScore) else Surprise := FMu;
  FMu := FMu - FEta * (Surprise - FTau);
end;

function TNNetSamplerMirostat.GetToken(Origin: TNNetVolume): integer;
begin
  Origin.GetTokenArray(FTokenArr);
  SortTokenArray();
  Result := SampleAndUpdate();
end;

function TNNetSamplerMirostat.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  Origin.GetTokenArrayOnPixel(FTokenArr, PixelX, PixelY);
  SortTokenArray();
  Result := SampleAndUpdate();
end;

{ TNNetSamplerBase }

procedure TNNetSamplerBase.SortTokenArray;
begin
  QuickSortTokenArray(FTokenArr, Low(FTokenArr), High(FTokenArr));
end;

procedure TNNetSamplerBase.Reset();
begin
  // No-op default: stateless samplers have nothing to re-arm.
end;

destructor TNNetSamplerBase.Destroy;
begin
  SetLength(FTokenArr, 0);
  inherited Destroy;
end;

{ TNNetSamplerGreedy }

function TNNetSamplerGreedy.GetToken(Origin: TNNetVolume): integer;
begin
  Result := Origin.GetClass();
end;

function TNNetSamplerGreedy.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  Result := Origin.GetClassOnPixel(PixelX, PixelY);
end;

{ TNNetTokenHistoryPenalty }

constructor TNNetTokenHistoryPenalty.Create(Repetition: TNeuralFloat = 1.0;
  Frequency: TNeuralFloat = 0.0; Presence: TNeuralFloat = 0.0);
begin
  inherited Create();
  FRepetition := Repetition;
  FFrequency := Frequency;
  FPresence := Presence;
  SetLength(FCounts, 0);
end;

destructor TNNetTokenHistoryPenalty.Destroy();
begin
  SetLength(FCounts, 0);
  inherited Destroy();
end;

procedure TNNetTokenHistoryPenalty.EnsureSize(NewSize: integer);
var
  OldSize, NewSizeM1, I: integer;
begin
  OldSize := Length(FCounts);
  if NewSize > OldSize then
  begin
    SetLength(FCounts, NewSize);
    NewSizeM1 := NewSize - 1;
    for I := OldSize to NewSizeM1 do FCounts[I] := 0;
  end;
end;

procedure TNNetTokenHistoryPenalty.RegisterToken(TokenId: integer);
begin
  if TokenId < 0 then exit;
  EnsureSize(TokenId + 1);
  Inc(FCounts[TokenId]);
end;

procedure TNNetTokenHistoryPenalty.ResetHistory();
var
  I, MaxI: integer;
begin
  MaxI := Length(FCounts) - 1;
  for I := 0 to MaxI do FCounts[I] := 0;
end;

procedure TNNetTokenHistoryPenalty.Apply(Logits: TNNetVolume);
var
  I, MaxToken, Count: integer;
  Logit: TNeuralFloat;
begin
  // The history can never be larger than the logit volume of interest.
  MaxToken := Length(FCounts) - 1;
  if MaxToken >= Logits.Size then MaxToken := Logits.Size - 1;
  for I := 0 to MaxToken do
  begin
    Count := FCounts[I];
    if Count > 0 then
    begin
      Logit := Logits.FData[I];
      // (a) repetition penalty - sign-correct CTRL form.
      if FRepetition <> 1.0 then
      begin
        if Logit > 0 then Logit := Logit / FRepetition
        else Logit := Logit * FRepetition;
      end;
      // (b) frequency penalty - scales with the occurrence count.
      Logit := Logit - FFrequency * Count;
      // (c) presence penalty - flat push for any token used at least once.
      Logit := Logit - FPresence;
      Logits.FData[I] := Logit;
    end;
  end;
end;

procedure TNNetTokenHistoryPenalty.ApplyToProbabilities(Probs: TNNetVolume);
var
  I, MaxToken, Count: integer;
  P, Total: TNeuralFloat;
  Changed: boolean;
begin
  // Guaranteed bit-for-bit no-op when every knob is at its default.
  if (FRepetition = 1.0) and (FFrequency = 0.0) and (FPresence = 0.0) then exit;
  Changed := false;
  MaxToken := Length(FCounts) - 1;
  if MaxToken >= Probs.Size then MaxToken := Probs.Size - 1;
  for I := 0 to MaxToken do
  begin
    Count := FCounts[I];
    if Count > 0 then
    begin
      P := Probs.FData[I];
      // (a) repetition penalty: p := p^r ("power then renormalize", the
      // probability-domain image of the sign-correct CTRL logit rule -
      // ln p <= 0 always, so the negative branch ln p * r applies).
      if (FRepetition <> 1.0) and (P > 0) then P := pcr_powf(P, FRepetition);
      // (b) frequency + (c) presence: log-space subtraction is a
      // multiplicative exp() factor on the probability.
      if (FFrequency <> 0.0) or (FPresence <> 0.0) then
        P := P * Exp(-(FFrequency * Count + FPresence));
      Probs.FData[I] := P;
      Changed := true;
    end;
  end;
  // Renormalize to a proper distribution (only if something changed, so an
  // empty history remains a bit-for-bit no-op).
  if Changed then
  begin
    Total := Probs.GetSum();
    if Total > 0 then Probs.Divi(Total);
  end;
end;

{ TStringVolumeList }

function TStringVolumeList.CreateNonZeroPositionLists: TStringIntegerList;
var
  ElementCnt: integer;
  MaxCnt: integer;
begin
  Result := TStringIntegerList.Create;
  if Count > 0 then
  begin
    MaxCnt := Count - 1;
    for ElementCnt := 0 to MaxCnt do
    begin
      Result.AddObject(Self[ElementCnt], Self.List[ElementCnt].CreateIntegerListWithNonZeroPos() );
    end;
  end;
end;

constructor TNNetVolumePair.Create();
begin
  inherited Create();
  FA := TNNetVolume.Create();
  FB := TNNetVolume.Create();
end;

constructor TNNetVolumePair.Create(pA, pB: TNNetVolume);
begin
  inherited Create();
  FA := pA;
  FB := pB;
end;

constructor TNNetVolumePair.CreateCopying(pA, pB: TNNetVolume);
begin
  inherited Create();
  FA := TNNetVolume.Create(pA);
  FB := TNNetVolume.Create(pB);
  FA.Copy(pA);
  FB.Copy(pB);
end;

destructor TNNetVolumePair.Destroy();
begin
  FA.Free;
  FB.Free;
  inherited Destroy();
end;

{ TNNetStringList }

function TNNetStringList.GetRandomIndex(): integer;
begin
  if Self.Count > 0 then
  begin
    Result := Random(Self.Count);
  end
  else
  begin
    Result := -1;
  end;
end;

procedure TNNetStringList.KeepFirst(Cnt: integer);
begin
  DeleteLast(Count-Cnt);
end;

procedure TNNetStringList.KeepLast(Cnt: integer);
begin
  DeleteFirst(Count-Cnt);
end;

procedure TNNetStringList.DeleteFirst(Cnt: integer);
var
  I: integer;
begin
  if Cnt >= Count then
  begin
    Clear;
  end
  else
  begin
    for I := 1 to Cnt do Delete(0);
  end;
end;

procedure TNNetStringList.DeleteLast(Cnt: integer);
var
  I: integer;
begin
  if Cnt >= Count then
  begin
    Clear;
  end
  else
  begin
    for I := 1 to Cnt do Delete(Count-1);
  end;
end;

procedure TNNetStringList.SetCapacity(NewCapacity: Integer);
begin
  inherited SetCapacity(NewCapacity);
end;

/// Helper function to check if a string contains any character from a set
// This function was coded by chatGPT4.
function StrHasChars(const Str: string; Strict: Boolean; const Chars: TSysCharSet): Boolean;
var
  P: PChar;
begin
  P := PChar(Str);
  while (P^ <> #0) and (not CharInSet(P^, Chars) or Strict) do Inc(P);
  Result := P^ <> #0;
end;

// This function was coded by chatGPT4.
function TNNetStringList.GetDelimitedTextFast: string;
{$IFDEF FPC}
var
  I, MaxIdx: Integer;
  S: String;
  BreakChars: set of Char;
  DoQuote: Boolean;
  StringBuilder: TAnsiStringBuilder;
begin
  CheckSpecialChars;
  if StrictDelimiter then
    BreakChars := [#0, QuoteChar, Delimiter]
  else
    BreakChars := [#0..' ', QuoteChar, Delimiter];

  StringBuilder := TAnsiStringBuilder.Create();
  MaxIdx := Count - 1;
  try
    for I := 0 to MaxIdx do
    begin
      S := Strings[I];
      DoQuote := AlwaysQuote;
      if not DoQuote then
      begin
        // Quote strings that include BreakChars
        DoQuote := StrHasChars(S, True, BreakChars);
      end;
      if DoQuote and (QuoteChar <> #0) then
        StringBuilder.Append(AnsiQuotedStr(S, QuoteChar))
      else
        StringBuilder.Append(S);

      if I < Count - 1 then
        StringBuilder.Append(Delimiter);
    end;

    // Quote empty string
    if (StringBuilder.Length = 0) and (Count = 1) and (QuoteChar <> #0) then
      StringBuilder.Append(QuoteChar).Append(QuoteChar);

    Result := StringBuilder.ToString;
  finally
    StringBuilder.Free;
  end;
end;
{$ELSE}
begin
  Result := DelimitedText;
end;
{$ENDIF}

procedure TNNetStringList.LoadLargeFile(Filename: string);
var
  LargeFile: TextFile;
  StrLine: string;
begin
  AssignFile(LargeFile, Filename);
  Reset(LargeFile);
  while not Eof(LargeFile) do
  begin
    ReadLn(LargeFile, StrLine);
    Self.Add(StrLine);
  end;
  CloseFile(LargeFile);
end;

{$IFDEF FPC}
{ TStringsObj }
function TStringsObj.GetList(Index: Integer): TObj;
begin
  Result := TObj(Self.Objects[Index]);
end;

constructor TStringsObj.Create;
begin
  inherited Create;
  Self.OwnsObjects := true;
  Self.Sorted := true;
  Self.FSortedList := true;
end;

function TStringsObj.AddObject(const S: string; AObject: TObject): Integer;
begin
  if not Assigned(AObject) then
  begin
    AObject := TObj.Create;
  end;

  if (FSortedList) and (AObject is TStringList) then
  begin
    TStringList(AObject).Sorted := true;
  end;

  Result := inherited AddObject(S, AObject);
end;

procedure TStringsObj.FixObjects();
var
  ElementId, MaxIdx: integer;
begin
  if Count > 0 then
  begin
    MaxIdx := Count - 1;
    for ElementId := 0 to MaxIdx do
    begin
      if not Assigned(Self.List[ElementId]) then
      begin
        Self.Objects[ElementId] := TObj.Create;
      end;

      if (FSortedList) and (Self.Objects[ElementId] is TStringList) then
      begin
        TStringList(Self.Objects[ElementId]).Sorted := true;
      end;
    end;
  end;
end;

procedure TStringsObj.AddStringObj(const S: string);
begin
  Self.AddObject(S, TObj.Create);
end;

{ TStringStringList }

procedure TStringStringList.LoadFromCsv(filename: string;
  SkipFirstLine:boolean = true;
  KeyId: integer = -1;
  Separator: char = ',');
var
  Sep: TStringList;
  CurrentLine: string;
  KeyStr: string;
  FileHandler: TextFile;
  LineCnt: integer;
begin
  Self.Sorted := false;
  Self.SortedList := false;
  AssignFile(FileHandler, filename);
  Reset(FileHandler);
  LineCnt := 0;
  while (not Eof(FileHandler)) do // and (LineCnt<10000)
  begin
    ReadLn(FileHandler, CurrentLine);
    if not( (LineCnt = 0) and (SkipFirstLine) ) then
    begin
      Sep := CreateTokenizedStringList(Separator);
      Sep.DelimitedText := CurrentLine;
      if (KeyId = -1) then
      begin
        KeyStr := IntToStr(LineCnt);
      end
      else
      begin
        KeyStr := Sep[KeyId];
      end;
      AddObject(KeyStr, TObject(Sep));
    end;
    LineCnt := LineCnt + 1;
    // debug line only:
    //if LineCnt mod 100000 = 0 then WriteLn(LineCnt);
  end;
  CloseFile(FileHandler);
end;

procedure TStringStringList.SaveToCsv(filename: string;
  Separator: char = ',');
var
  RowCnt: integer;
  MaxCnt: integer;
  FileHandler: TextFile;
begin
  MaxCnt := Count - 1;
  if MaxCnt > -1 then
  begin
    AssignFile(FileHandler, filename);
    ReWrite(FileHandler);
    for RowCnt := 0 to MaxCnt do
    begin
      List[RowCnt].Delimiter := Separator;
      WriteLn(FileHandler, List[RowCnt].DelimitedText);
    end;
    CloseFile(FileHandler);
  end;
end;

{$ELSE}
function TStringsObj.GetList(Index: Integer): TObject;
begin
  Result := Self.Objects[Index];
end;

constructor TStringsObj.Create;
begin
  inherited Create;
  Self.OwnsObjects := true;
  Self.Sorted := true;
end;

function TStringsObj.AddObject(const S: string; AObject: TObject): Integer;
begin
  if not Assigned(AObject) then
  begin
    AObject := CreateObject;
  end;

  if AObject is TStringList then
  begin
    TStringList(AObject).Sorted := true;
  end;

  Result := inherited AddObject(S, AObject);
end;

procedure TStringsObj.FixObjects();
var
  ElementId, MaxIdx: integer;
begin
  if Count > 0 then
  begin
    MaxIdx := Count - 1;
    for ElementId := 0 to MaxIdx do
    begin
      if not Assigned(Self.List[ElementId]) then
      begin
        Self.Objects[ElementId] := CreateObject;
      end;

      if Self.Objects[ElementId] is TStringList then
      begin
        TStringList(Self.Objects[ElementId]).Sorted := true;
      end;
    end;
  end;
end;

procedure TStringsObj.AddStringObj(const S: string);
begin
  Self.AddObject(S, CreateObject);
end;

{ TStringStringList }
function TStringStringList.CreateObject: TObject;
begin
  Result := TStringList.Create();
end;

function TStringStringList.GetList(Index: Integer): TStringList;
begin
  Result := TStringList(inherited GetList(Index) );
end;

{ TStringVolumeList }
function TStringVolumeList.CreateObject: TObject;
begin
  Result := TNNetVolume.Create();
end;

function TStringVolumeList.GetList(Index: Integer): TNNetVolume;
begin
  Result := TNNetVolume(inherited GetList(Index) );
end;

{ TStringStringListVolume }
function TStringStringListVolume.CreateObject: TObject;
begin
  Result := TStringVolumeList.Create;
end;

function TStringStringListVolume.GetList(Index: Integer): TStringVolumeList;
begin
  Result := TStringVolumeList(inherited GetList(Index) );
end;

{ TStringIntegerList }

function TStringIntegerList.CreateObject: TObject;
begin
  Result := TIntegerList.Create();
end;

function TStringIntegerList.GetList(Index: Integer): TIntegerList;
begin
  Result := TIntegerList(inherited GetList(Index) );
end;

{$ENDIF}

{ TStringListInt }
function TStringListInt.GetInteger(Index: Integer): PtrInt;
begin
  Result := PtrInt(Self.Objects[Index]);
end;

procedure TStringListInt.PutInteger(Index: Integer; AValue: PtrInt);
begin
  Objects[Index] := TObject(AValue);
end;

constructor TStringListInt.Create;
begin
  inherited Create;
  Self.OwnsObjects := false;
  FTokenizer := CreateTokenizedStringList(' ');
  SetLength(FIntegerToStr, 0);
end;

destructor TStringListInt.Destroy;
begin
  SetLength(FIntegerToStr, 0);
  FTokenizer.Free;
  inherited Destroy;
end;

procedure TStringListInt.LoadVocabularyFromFile(const filename: string);
begin
  Self.LoadFromFile(filename);
  SaveCurrentPositionAndSort();
end;

procedure TStringListInt.SortByIntegerAsc;
begin
  Sorted := false;
  CustomSort(@CompareStringListIntegerAsc);
end;

procedure TStringListInt.SortByIntegerDesc;
begin
  Sorted := false;
  CustomSort(@CompareStringListIntegerDesc);
end;

function TStringListInt.AddInteger(const S: string; AValue: PtrInt): integer;
begin
  Result := AddObject(S, TObject(AValue));
end;

{ TNNetDictionary }
constructor TNNetDictionary.Create(pMaxSize: integer);
begin
  inherited Create;
  Self.Sorted := true;
  Self.CaseSensitive := false;

  FMaxSize := pMaxSize;
end;


function TNNetDictionary.AddWordToDictionary(pWord: string): boolean;
var
  Index: integer;
begin
  if Count < FMaxSize then
  begin
    Result := true;
    if Length(pWord) > 0 then
    begin
      if not(Self.Find(pWord, Index)) then
      begin
        Self.AddInteger(pWord, 1);
      end
      else
      begin
        Self.Integers[Index] := Self.Integers[Index] + 1;
      end;
    end;
  end
  else
  begin
    Result := false;
  end;
end;

function TNNetDictionary.AddWordsToDictionary(pString: string): boolean;
var
  WordCount, TokenMax: integer;
begin
  Result := false;
  FTokenizer.Delimiter := ' ';
  FTokenizer.DelimitedText := pString;

  if FTokenizer.Count > 0 then
  begin
    TokenMax := FTokenizer.Count - 1;
    for WordCount := 0  to TokenMax do
    begin
      Result := AddWordToDictionary(Trim(FTokenizer[WordCount]));
    end;
  end;
end;

procedure TNNetDictionary.AddWordFromCsvField(filename: string; fieldId: integer
  ; SkipFirstLine: boolean = True; Separator:char = ',');
var
  Sep: TStringList;
  CurrentLine: string;
  WordToAdd: string;
  FileHandler: TextFile;
  LineCnt: integer;
begin
  Sep := CreateTokenizedStringList(Separator);
  AssignFile(FileHandler, filename);
  Reset(FileHandler);
  LineCnt := 0;
  while not Eof(FileHandler) do
  begin
    ReadLn(FileHandler, CurrentLine);
    if not( (LineCnt = 0) and (SkipFirstLine) ) then
    begin
      Sep.DelimitedText := CurrentLine;
      if Sep.Count > fieldId then
      begin
        WordToAdd := Sep[fieldId];
        {$IFDEF FPC}
        AddWordToDictionary(TrimSet(WordToAdd,['"',' ']));
        {$ELSE}
        AddWordToDictionary(Trim(WordToAdd));
        {$ENDIF}
      end;
    end;
    LineCnt := LineCnt + 1;
    //Debug line:
    //if LineCnt mod 100000 = 0 then WriteLn(LineCnt);
  end;
  CloseFile(FileHandler);
  Sep.Free;
end;

procedure TNNetDictionary.RemoveAllStringsWithLessThen(I: integer);
var
  MaxPos, CurrentPos: integer;
begin
  MaxPos := Count - 1;
  if MaxPos > -1 then
  begin
    Self.Sorted := false;
    Self.SortByIntegerDesc;
    CurrentPos := 0;
    while CurrentPos <= MaxPos do
    begin
      if Self.Integers[CurrentPos] < I then
      begin
        Self.KeepFirst(CurrentPos);
        MaxPos := -1; // exit the while loop
      end;
      CurrentPos := CurrentPos + 1;
    end;
    Self.Sort;
    Self.Sorted := true;
  end;
end;

function TStringListInt.WordToIndex(pWord: string): integer;
begin
  if not(Self.Find(pWord, Result)) then Result := -1;
end;

function TStringListInt.WordToInteger(pWord: string): integer;
var
  Position: integer;
begin
  if Self.Find(pWord, Position) then
  begin
    Result := Integers[Position];
  end
  else
  begin
    Result := -1;
  end;
end;

function TStringListInt.IntegerToWord(pInteger: integer): string;
begin
  Result := FIntegerToStr[pInteger];
end;

function TStringListInt.DeTokenize(TokenId: integer): string;
begin
  Result := IntegerToWord(TokenId);
end;

procedure TStringListInt.Tokenize(pString: string;
  var IntArr: TNeuralIntegerArray);
begin
  StringToIntegerArray(pString, IntArr);
end;

function TStringListInt.GetVocabCount(): integer;
begin
  Result := Count;
end;

function TStringListInt.TokenizerHasSeparator: boolean;
begin
  Result := true;
end;

procedure TStringListInt.SaveCurrentPosition();
var
  RowCnt, RowMax: integer;
begin
  SetLength(FIntegerToStr, Self.Count);
  RowMax := Self.Count - 1;
  for RowCnt := 0 to RowMax do
  begin
    Self.Integers[RowCnt] := RowCnt;
    FIntegerToStr[RowCnt] := Self[RowCnt];
  end;
end;

procedure TStringListInt.StringToIndexArray(pString: string;
  var IntArr: TNeuralIntegerArray);
var
  WordCount: integer;
  WordIndex: integer;
  TokenMax: integer;
begin
  FTokenizer.Delimiter := ' ';
  FTokenizer.DelimitedText := pString;

  if FTokenizer.Count > 0 then
  begin
    SetLength(IntArr, FTokenizer.Count);
    TokenMax := FTokenizer.Count - 1;
    for WordCount := 0  to TokenMax do
    begin
      WordIndex := Self.WordToIndex(FTokenizer[WordCount]);
      //WriteLn(WordIndex,':',FTokenizer[WordCount]);
      if WordIndex >= 0 then
      begin
        IntArr[WordCount] := WordIndex;
      end;
    end;
  end;
end;

procedure TStringListInt.StringToIntegerArray(pString: string;
  var IntArr: TNeuralIntegerArray);
var
  WordCount: integer;
  WordInteger: integer;
  TokenMax: integer;
begin
  FTokenizer.Delimiter := ' ';
  FTokenizer.DelimitedText := pString;

  if FTokenizer.Count > 0 then
  begin
    SetLength(IntArr, FTokenizer.Count);
    TokenMax := FTokenizer.Count - 1;
    for WordCount := 0  to TokenMax do
    begin
      WordInteger := Self.WordToInteger(FTokenizer[WordCount]);
      //WriteLn(WordIndex,':',FTokenizer[WordCount]);
      if WordInteger >= 0 then
      begin
        IntArr[WordCount] := WordInteger;
      end;
    end;
  end;
end;

function TStringListInt.IndexArrayToString(var IntArr: TNeuralIntegerArray
  ): string;
var
  WordCount, WordMax: integer;
  WordIndex: integer;
begin
  FTokenizer.Clear;
  FTokenizer.Delimiter := ' ';
  WordMax := Length(IntArr) - 1;
  if WordMax >= 0 then
  begin
    for WordCount := 0 to WordMax do
    begin
      WordIndex := IntArr[WordCount];
      //WriteLn(WordIndex,':',FTokenizer[WordCount]);
      if WordIndex >= 0 then
      begin
        FTokenizer.Add(Self[WordIndex]);
      end;
    end;
  end;
  Result := FTokenizer.DelimitedText;
end;

function TStringListInt.IntegerArrayToString(var IntArr: TNeuralIntegerArray
  ): string;
var
  WordCount, WordMax: integer;
  WordInteger: integer;
begin
  FTokenizer.Clear;
  FTokenizer.Delimiter := ' ';
  WordMax := Length(IntArr) - 1;
  if WordMax >= 0 then
  begin
    for WordCount := 0 to WordMax do
    begin
      WordInteger := IntArr[WordCount];
      //WriteLn(WordIndex,':',FTokenizer[WordCount]);
      if WordInteger >= 0 then
      begin
        FTokenizer.Add(FIntegerToStr[WordInteger]);
      end;
    end;
  end;
  Result := FTokenizer.DelimitedText;
end;

function TStringListInt.IntegerListToCsv(IL: TIntegerList; pDelimiter: char = ','): string;
var
  WordCount, WordMax: integer;
begin
  FTokenizer.Clear;
  FTokenizer.Delimiter := Delimiter;
  WordMax := IL.Count - 1;
  if WordMax >= 0 then
  begin
    for WordCount := 0 to WordMax do
    begin
      FTokenizer.Add(IntToStr(IL[WordCount]));
    end;
  end;
  Result := FTokenizer.DelimitedText;
end;

procedure TStringListInt.SaveCurrentPositionAndSort();
begin
  SaveCurrentPosition();
  Self.Sort();
  Self.Sorted := true;
end;

procedure TNNetDictionary.StringToVolume(pString: string; Volume: TNNetVolume);
var
  WordCount: integer;
  WordIndex: integer;
  TokenMax: integer;
begin
  if Volume.Size <> Count then Volume.Resize(Count,1,1);

  Volume.Fill(0);

  FTokenizer.DelimitedText := pString;

  if FTokenizer.Count > 0 then
  begin
    TokenMax := FTokenizer.Count - 1;
    for WordCount := 0  to TokenMax do
    begin
      WordIndex := Self.WordToIndex(FTokenizer[WordCount]);
      //WriteLn(WordIndex,':',FTokenizer[WordCount]);
      if WordIndex >= 0 then
        {$IFDEF FPC}
        Volume.FData[WordIndex] += 1.0;
        {$ELSE}
        Volume.FData[WordIndex] := Volume.FData[WordIndex] + 1.0;
        {$ENDIF}
    end;
  end;
end;

function TNNetDictionary.VolumeToString(Volume: TNNetVolume;
  Threshold: TNeuralFloat): string;
var
  I: integer;
  vHigh: integer;
begin
  FTokenizer.Text := '';
  if Length(Volume.FData) > 0 then
  begin
    vHigh := High(Volume.FData);
    if vHigh > 0 then
    begin
      for I := 0 to vHigh do
      begin
        if Volume.FData[I] > Threshold then
        begin
          FTokenizer.Add(Self[I]+':'+Volume.NeuralToStr(Volume.FData[I]));
        end;
      end;
    end;
  end;

  Result := FTokenizer.DelimitedText;
end;

procedure TNNetDictionary.CsvToTStringVolumeList(filename: string;
  GroupByFieldId, DataFieldId: integer; SVL: TStringVolumeList;
  SkipFirstLine: boolean = True; Separator:char = ',');
var
  Sep: TStringList;
  CurrentLine: string;
  KeyStr, DataStr: string;
  DataId, KeyId: integer;
  FileHandler: TextFile;
  LineCnt: integer;
  V: TNNetVolume;
begin
  Sep := CreateTokenizedStringList(Separator);
  AssignFile(FileHandler, filename);
  Reset(FileHandler);
  LineCnt := 0;
  while not Eof(FileHandler) do
  begin
    ReadLn(FileHandler, CurrentLine);
    if not( (LineCnt = 0) and (SkipFirstLine) ) then
    begin
      Sep.DelimitedText := CurrentLine;
      if (Sep.Count > GroupByFieldId) and (Sep.Count > DataFieldId) then
      begin
        KeyStr := Sep[GroupByFieldId];
        DataStr := Sep[DataFieldId];
        DataId := IndexOf(DataStr);
        if DataId > -1 then
        begin
          KeyId := SVL.IndexOf(KeyStr);
          if KeyId > -1 then
          begin
            V := SVL.List[KeyId];
            V.FData[DataId] := 1;
          end
          else
          begin
            V := TNNetVolume.Create(Count);
            V.FData[DataId] := 1;
            SVL.AddObject(KeyStr, V);
          end;
        end;
      end;
    end;
    LineCnt := LineCnt + 1;
    // debug line only:
    //if LineCnt mod 100000 = 0 then WriteLn(LineCnt);
  end;
  CloseFile(FileHandler);
  Sep.Free;
end;

procedure TNNetDictionary.PrintDebug(FirstElements: integer);
var
  ElementCnt, MaxIdx: integer;
begin
  WriteLn('Number of elements: ', Count);
  if Count > 0 then
  begin
    if FirstElements > Count then FirstElements := Count;
    WriteLn('Showing first ',FirstElements,' elements.');
    MaxIdx := FirstElements - 1;
    for ElementCnt := 0 to MaxIdx do
    begin
      WriteLn(ElementCnt,': ',Self[ElementCnt],' -> ', Self.Integers[ElementCnt]);
    end;
  end;
end;

procedure TNNetDictionary.SaveDictionaryToFile(Filename: string; Separator: char
  );
var
  RowCnt: integer;
  MaxCnt: integer;
  FileHandler: TextFile;
begin
  MaxCnt := Count - 1;
  if MaxCnt > -1 then
  begin
    AssignFile(FileHandler, Filename);
    ReWrite(FileHandler);
    for RowCnt := 0 to MaxCnt do
    begin
      WriteLn(FileHandler, Self[RowCnt]+Separator+IntToStr(Self.Integers[RowCnt]));
    end;
    CloseFile(FileHandler);
  end;
end;

procedure TNNetDictionary.LoadDictionaryFromFile(Filename: string;
  Separator: char);
var
  Sep: TStringList;
  CurrentLine: string;
  Word: string;
  WordCount: string;
  FileHandler: TextFile;
  //LineCnt: integer;
begin
  Clear;
  Sep := CreateTokenizedStringList(Separator);
  AssignFile(FileHandler, Filename);
  Reset(FileHandler);
  //LineCnt := 0;
  while not Eof(FileHandler) do
  begin
    ReadLn(FileHandler, CurrentLine);
    Sep.DelimitedText := CurrentLine;
    if Sep.Count = 2 then
    begin
      {$IFDEF Debug}
      Word := Sep[0];
      WordCount := Sep[1];
      Self.AddInteger(Word,StrToInt(WordCount));
      {$ELSE}
      Self.AddInteger(Sep[0],StrToInt(Sep[1]));
      {$ENDIF}
      //LineCnt := LineCnt + 1;
    end
    else
    begin
      WriteLn('Bad dictionary entry:', CurrentLine);
    end;

    // debug line only:
    //if LineCnt mod 100000 = 0 then WriteLn(LineCnt);
  end;
  CloseFile(FileHandler);
  Sep.Free;
end;

{ TNNetKMeans }
constructor TNNetKMeans.Create(pVolNum, pSizeX, pSizeY, pDepth: integer; pManhattan: boolean = true);
begin
  inherited Create();

  FSample := TNNetVolumeList.Create();
  FClusters := TNNetVolumeList.Create();
  FClusterSums := TNNetVolumeList.Create();

  Resize(pVolNum, pSizeX, pSizeY, pDepth);

  FManhattanDistance := pManhattan;
end;

destructor TNNetKMeans.Destroy();
begin
  FSample.Free;
  FClusters.Free;
  FClusterSums.Free;

  inherited Destroy();
end;

procedure TNNetKMeans.RunStep(RepositionClusters: boolean = true);
var
  SampleCount, MaxSampleCount: integer;
  ClusterCount, MaxClusterCount: integer;
  ClosestId: integer;
  StartTime: double;
begin
  StartTime := Now();
  MaxSampleCount := FSample.Count - 1;
  MaxClusterCount := FClusters.Count - 1;

  FClusterSums.Fill(0);
  FClusterSums.ClearTag();

  if ( (MaxSampleCount > 0) and (MaxClusterCount > 0) ) then
  begin
    for SampleCount := 0 to MaxSampleCount do
    begin
      ClosestId := GetClusterId(FSample[SampleCount]);
      FClusterSums[ClosestId].Add(FSample[SampleCount]);
      FClusterSums[ClosestId].IncTag();
      FSample[SampleCount].Tag := ClosestId;
    end;

    for ClusterCount := 0 to MaxClusterCount do
    begin
      if FClusterSums[ClusterCount].Tag > 0 then
      begin
        FClusterSums[ClusterCount].Divi(FClusterSums[ClusterCount].Tag);
        if RepositionClusters then
        begin
          FClusters[ClusterCount].Copy(FClusterSums[ClusterCount]);
        end;
        FClusters[ClusterCount].Tag := FClusterSums[ClusterCount].Tag;
      end
      else
      begin
        FClusters[ClusterCount].Tag := 0;
      end;
    end;
  end;
  FLastStepTime := ( Now() - StartTime );
end;

procedure TNNetKMeans.Resize(pVolNum, pSizeX, pSizeY, pDepth: integer);
begin
  FClusters.Clear();
  FClusterSums.Clear();
  FSample.Clear();

  FClusters.AddVolumes(pVolNum, pSizeX, pSizeY, pDepth);
  FClusterSums.AddVolumes(pVolNum, pSizeX, pSizeY, pDepth);
end;

procedure TNNetKMeans.Randomize();
var
  ClusterCount, MaxCount: integer;
begin
  MaxCount := FClusters.Count - 1;
  if MaxCount >= 0 then
  begin
    for ClusterCount := 0 to MaxCount do
    begin
      {$IFDEF Debug}
      if FClusters[ClusterCount].Size <> FClusters[0].Size then
      begin
        FErrorProc('Cluster sizes differ at TNNetKMeans.Randomize.');
      end;
      {$ENDIF}
      FClusters[ClusterCount].Copy(FSample[Random(FSample.Count)]);
    end;
  end;
end;

procedure TNNetKMeans.RandomizeEmptyClusters();
var
  ClusterCount, MaxCount: integer;
begin
  MaxCount := FClusters.Count - 1;
  if MaxCount >= 0 then
  begin
    for ClusterCount := 0 to MaxCount do
    begin
      {$IFDEF Debug}
      if FClusters[ClusterCount].Size <> FClusters[0].Size then
      begin
        FErrorProc('Cluster sizes differ at TNNetKMeans.RandomizeEmptyClusters.');
      end;
      {$ENDIF}
      if FClusters[ClusterCount].Tag = 0 then
      begin
        FClusters[ClusterCount].Copy(FSample[Random(FSample.Count)]);
      end;
    end;
  end;
end;

procedure TNNetKMeans.AddSample(Original: TNNetVolume);
begin
  {$IFDEF Debug}
  if FClusters.Count > 0 then
  begin
    if Original.Size = FClusters[0].Size then
    begin
  {$ENDIF}
      FSample.Add(Original);
  {$IFDEF Debug}
    end
    else
    begin
      FErrorProc('Sample size ' + IntToStr(Original.Size) + ' differs from ' +
       'cluster size ' + IntToStr(FClusters[0].Size) );
    end;
  end
  else
    FErrorProc('Clusters need to be allocated before adding samples');
  {$ENDIF}
end;

function TNNetKMeans.GetClusterId(Original: TNNetVolume): integer;
begin
  if FManhattanDistance then
  begin
    Result := FClusters.GetManhattanClosestId(Original, FLastDistance);
  end
  else
  begin
    Result := FClusters.GetClosestId(Original, FLastDistance);
  end;
end;

function TNNetKMeans.GetTotalSize(): integer;
begin
  Result :=
    FSample.GetTotalSize() +
    FClusters.GetTotalSize() +
    FClusterSums.GetTotalSize();
end;

{ TNNetVolumeList }

function TNNetVolumeList.GetTotalSize(): integer;
var
  I, MaxIdx: integer;
begin
  Result := 0;
  if (Count>0) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Result := Result + Self[I].Size;
    end;
  end;
end;

function TNNetVolumeList.GetSum(): TNeuralFloat;
var
  I, MaxIdx: integer;
begin
  Result := 0;
  if (Count>0) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Result := Result + Self[I].GetSum();
    end;
  end;
end;

function TNNetVolumeList.GetAvg(): TNeuralFloat;
var
  floatSize: Single;
begin
  floatSize := GetTotalSize();
  if (floatSize > 0.1) then
  begin
    Result := GetSum() / floatSize;
  end
  else
  begin
    Result := 0;
  end;
end;

procedure TNNetVolumeList.AddValue(Value: TNeuralFloat);
var
  I, MaxIdx: integer;
begin
  if (Count>0) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Self[I].Add(Value);
    end;
  end;
end;

procedure TNNetVolumeList.Mul(Value: TNeuralFloat);
var
  I, MaxIdx: integer;
begin
  if (Count>0) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Self[I].Mul(Value);
    end;
  end;
end;

procedure TNNetVolumeList.Divi(Value: TNeuralFloat);
var
  I, MaxIdx: integer;
begin
  if (Count>0) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Self[I].Divi(Value);
    end;
  end;
end;

function TNNetVolumeList.GetClosestId(Original: TNNetVolume; var MinDist: TNeuralFloat): integer;
var
  I: integer;
  MaxCount: integer;
  CurrentDist: TNeuralFloat;
begin
  Result := 0;
  MaxCount := Count - 1;
  if (MaxCount > 0) then
  begin
    MinDist := Original.GetDistance(Self[0]);
    for I := 1 to MaxCount do
    begin
      CurrentDist := Original.GetDistance(Self[I]);
      if (CurrentDist < MinDist) then
      begin
        Result := I;
        MinDist := CurrentDist;
      end;
      if MinDist <= 0 then Break;
    end;
  end;
end;

function TNNetVolumeList.GetManhattanClosestId(Original: TNNetVolume; var MinDist: TNeuralFloat): integer;
var
  I: integer;
  MaxCount: integer;
  CurrentDist: TNeuralFloat;
begin
  Result := 0;
  MaxCount := Count - 1;
  if (MaxCount > 0) then
  begin
    MinDist := Original.SumDiff(Self[0]);
    for I := 1 to MaxCount do
    begin
      CurrentDist := Original.SumDiff(Self[I]);
      if (CurrentDist < MinDist) then
      begin
        Result := I;
        MinDist := CurrentDist;
      end;
      if MinDist <= 0 then Break;
    end;
  end;
end;

procedure TNNetVolumeList.Fill(c: Single);
var
  I, MaxIdx: integer;
begin
  if (Count>0) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Self[I].Fill(c);
    end;
  end;
end;

procedure TNNetVolumeList.ClearTag();
var
  I, MaxIdx: integer;
begin
  if (Count>0) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Self[I].ClearTag();
    end;
  end;
end;

procedure TNNetVolumeList.FillTag(TagId, TagValue: integer);
var
  I, MaxIdx: integer;
begin
  if (Count>0) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Self[I].Tags[TagId] := TagValue;
    end;
  end;
end;

procedure TNNetVolumeList.ConcatInto(V: TNNetVolume);
var
  TotalSize: integer;
  I, MaxIdx: integer;
  CurrPos: integer;
  Vol: TNNetVolume;
  VolSize: integer;
begin
  if (Count>0) then
  begin

    TotalSize := Self.GetTotalSize();
    if V.Size <> TotalSize then
    begin
      if TotalSize = Count * Self[0].Size
      then V.ReSize(Count,1,Self[0].Size)
      else V.ReSize(TotalSize,1,1);
    end;

    CurrPos := 0;
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Vol := Self[I];
      VolSize := Vol.Size;
      system.Move(Vol.FData[0], V.FData[CurrPos], VolSize * csNeuralFloatSize);
      Inc(CurrPos, VolSize);
    end;
  end;
end;

procedure TNNetVolumeList.InterleaveInto(V: TNNetVolume);
var
  CountVolume, CountElement: integer;
  MaxVolume, MaxElement: integer;
  CurrPos: integer;
begin
  if (Count>0) then
  begin
    MaxVolume := Count - 1;
    MaxElement := Self[0].Size - 1;
    CurrPos := 0;

    for CountElement := 0 to MaxElement do
    begin
      for CountVolume := 0 to MaxVolume do
      begin
        V.FData[CurrPos] := Self[CountVolume].FData[CountElement];
        CurrPos := CurrPos + 1;
      end;
    end;
  end;
end;

procedure TNNetVolumeList.SplitFrom(V: TNNetVolume);
var
  TotalSize: integer;
  I, MaxIdx: integer;
  CurrPos: integer;
  Vol: TNNetVolume;
  VolSize: integer;
begin
  if (Count>0) then
  begin

    TotalSize := Self.GetTotalSize();
    if V.Size < TotalSize then
    begin
      V.ReSize(TotalSize,1,1);
    end;

    CurrPos := 0;
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      Vol := Self[I];
      VolSize := Vol.Size;
      system.Move(V.FData[CurrPos], Vol.FData[0], VolSize * csNeuralFloatSize);
      Inc(CurrPos, VolSize);
    end;
  end;
end;

procedure TNNetVolumeList.AddVolumes(pVolNum, pSizeX, pSizeY, pDepth: integer;
  c: TNeuralFloat);
var
  I: integer;
begin
  for I := 1 to pVolNum do
  begin
    Self.Add( TNNetVolume.Create(pSizeX, pSizeY, pDepth,c) );
  end;
end;

procedure TNNetVolumeList.AddVolumes(Origin: TNNetVolumeList);
var
  I, OriginMax: integer;
  NewVolume: TNNetVolume;
begin
  if Origin.Count > 0 then
  begin
    OriginMax := Origin.Count - 1;
    for I := 0 to OriginMax do
    begin
      NewVolume := TNNetVolume.Create();
      NewVolume.Copy(Origin[I]);
      Self.Add( NewVolume );
    end;
  end;
end;

procedure TNNetVolumeList.AddCopy(Origin: TNNetVolume);
var
  NewVolume: TNNetVolume;
begin
  NewVolume := TNNetVolume.Create();
  NewVolume.Copy(Origin);
  NewVolume.Tags[0] := Origin.Tags[0];
  NewVolume.Tags[1] := Origin.Tags[1];
  Self.Add( NewVolume );
end;

procedure TNNetVolumeList.AddInto(Original: TNNetVolume);
var
  MaxVolumes, I: integer;
begin
  MaxVolumes := Count - 1;
  for I := 0 to MaxVolumes do
  begin
    Original.Add(Self.Items[I]);
  end;
end;

{$IFNDEF FPC}
function TNNetVolumeList.GetItem(Index: Integer): TNNetVolume;
begin
  Result := TNNetVolume(Get(Index));
end;

procedure TNNetVolumeList.SetItem(Index: Integer; AObject: TNNetVolume);
begin
  Put(Index,AObject);
end;
{$ENDIF}

procedure TNNetVolumeList.SortByTagAsc;
begin
  Sort(@CompareNNetVolumeListAsc);
end;

procedure TNNetVolumeList.SortByTagDesc;
begin
  Sort(@CompareNNetVolumeListDesc);
end;

procedure TNNetVolumeList.GetColumn(V: TNNetVolume; colIdx: integer);
var
  I, MaxIdx: integer;
begin
  if (Count>0) then
  begin
    if V.Size <> Self.Count then
    begin
      V.ReSize(1, 1, Self.Count);
    end;

    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      V.FData[I] := Self[I].FData[colIdx];
    end;
  end;
end;

procedure TNNetVolumeList.ResizeImage(NewSizeX, NewSizeY: integer);
var
  I, MaxIdx: integer;
  AuxVolume: TNNetVolume;
begin
  if (Count>0) then
  begin
    AuxVolume := TNNetVolume.Create();
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      AuxVolume.Copy(Self[I]);
      Self[I].CopyResizing(AuxVolume, NewSizeX, NewSizeY);
    end;
    AuxVolume.Free;
  end;
end;

procedure TNNetVolumeList.AddPadding(Padding: integer);
var
  I, MaxIdx: integer;
  AuxVolume: TNNetVolume;
begin
  if (Count>0) then
  begin
    AuxVolume := TNNetVolume.Create();
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      AuxVolume.Copy(Self[I]);
      Self[I].CopyPadding(AuxVolume, Padding);
    end;
    AuxVolume.Free;
  end;
end;

{ TMObject }
procedure TMObject.DefaultMessageProc(const S: string);
begin
  WriteLn(S);
end;

procedure TMObject.DefaultErrorProc(const S: string);
begin
  WriteLn(S);
end;

procedure TMObject.DefaultHideMessages(const S: string);
begin
  // do nothing !!!
end;

procedure TMObject.HideMessages();
begin
  MessageProc := {$IFDEF FPC}@{$ENDIF}Self.DefaultHideMessages;
end;

constructor TMObject.Create();
begin
  inherited Create();
  MessageProc := {$IFDEF FPC}@{$ENDIF}Self.DefaultMessageProc;
  ErrorProc := {$IFDEF FPC}@{$ENDIF}Self.DefaultErrorProc;
end;

destructor TMObject.Destroy();
begin
  inherited Destroy;
end;

function TVolume.GetTags(x: integer): integer;
begin
  GetTags := FTag[x];
end;

procedure TVolume.SetTags(x: integer; AValue: integer);
begin
  FTag[x] := AValue;
end;

procedure TVolume.SetTag(I: integer);
begin
  FTag[0] := I;
end;

function TVolume.GetTag: integer;
begin
  GetTag := FTag[0];
end;

{ TVolume }
constructor TVolume.Create(pSizeX, pSizeY, pDepth: integer; c: T);
begin
  inherited Create();
  FSize := 0;

  ReSize(pSizeX, pSizeY, pDepth);
  Fill(c);
  ClearTag();

  FFormatSettings := GetDefaultNumericFormat;
end;

constructor TVolume.Create(pInput: array of T);
begin
  Create(Length(pInput), 1, 1);
  Self.Copy(pInput);
end;

constructor TVolume.Create(Original: array of byte);
begin
  Create(Length(Original), 1, 1);
  Self.Copy(Original);
end;

constructor TVolume.Create(Original: TVolume);
begin
  Create(Original.SizeX, Original.SizeY, Original.Depth);
  Copy(Original);
end;

constructor TVolume.Create(Original: TBits; pFalse: T; pTrue: T);
begin
  Create();
  Self.Copy(Original, pFalse, pTrue);
end;

constructor TVolume.CreateAsBits(Original: array of byte; pFalse: T; pTrue: T);
begin
  Create();
  Self.CopyAsBits(Original, pFalse, pTrue);
end;

constructor TVolume.Create(pSize: integer; c: T);
begin
  Create(pSize,1,1,c);
end;

constructor TVolume.Create();
begin
  Create(1, 1, 1);
end;

procedure TVolume.Randomize(a:integer=10000; b:integer=5000; c:integer=5000);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := (random(a) - b) / c;
end;

procedure TVolume.RandomizeGaussian(pMul: TNeuralFloat = 1.0);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := RandomGaussianValue() * pMul;
end;

procedure TVolume.AddGaussianNoise(pMul: TNeuralFloat);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] += RandomGaussianValue() * pMul;
    {$ELSE}
    FData[I] := FData[I] + RandomGaussianValue() * pMul;
    {$ENDIF}
end;

// inspired on
// https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
procedure TVolume.AddSaltAndPepper(pNum: integer; pSalt: T = 1.0;
  pPepper: T = -1.0; pColor:boolean = false);
var
  I: integer;
  CntDepth, DepthM1: integer;
  SaltPosX, SaltPosY, PepperPosX, PepperPosY: integer;
  SaltBase, PepperBase: integer;
begin
  DepthM1 := FDepth - 1;
  for I := 1 to pNum do
  begin
    SaltPosX := Random(FSizeX);
    SaltPosY := Random(FSizeY);
    PepperPosX := Random(FSizeX);
    PepperPosY := Random(FSizeY);

    SaltBase := GetRawPos(SaltPosX, SaltPosY, 0);
    PepperBase := GetRawPos(PepperPosX, PepperPosY, 0);
    for CntDepth := 0 to DepthM1 do
    begin
      if (Not(pColor) or (Random(100) < 50) ) then
      begin
        FData[SaltBase + CntDepth] := pSalt;
        FData[PepperBase + CntDepth] := pPepper;
      end;
    end;
  end;
end;

// returns a random gaussivan value. This implementation is inspired on:
//http://www.cs.princeton.edu/courses/archive/fall12/cos126/assignments/StdGaussian.java.html
function TVolume.RandomGaussianValue(): TNeuralFloat;
var
  r, x, y: TNeuralFloat;
begin
  r := 0;
  // loop executed 4 / pi = 1.273.. times on average
  while ( (r > 1) or (r = 0) ) do
  begin
    // find a uniform random point (x, y) inside unit circle
    x := 2.0 * Random() - 1.0;
    y := 2.0 * Random() - 1.0;
    r := x*x + y*y;
  end;

  RandomGaussianValue := x * Sqrt(-2.0 * pcr_logf(r) / r);
end;

procedure TVolume.Add(Original: TVolume);
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at Add: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] += Original.FData[I];
    {$ELSE}
    FData[I] := FData[I] + Original.FData[I];
    {$ENDIF}
end;

procedure TVolume.Add(Value: T);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] += Value;
    {$ELSE}
    FData[I] := FData[I] + Value;
    {$ENDIF}
end;

class procedure TVolume.Add(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer
  );
var
  I: integer;
  vHigh: integer;
begin
  vHigh := NumElements - 1;
  for I := 0 to vHigh do
    {$IFDEF FPC}
    PtrA^[I] += PtrB^[I];
    {$ELSE}
    PtrA^[I] := PtrA^[I] + PtrB^[I];
    {$ENDIF}
end;

class procedure TVolume.MaxElements(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := pSize - 1;
  for I := 0 to vHigh do
    if PtrB^[I] > PtrA^[I] then PtrA^[I] := PtrB^[I];
end;

procedure TVolume.AddAtDepth(pDepth: integer; Value: T);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  RawPos, RowStride: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step

  for CntX := 0 to MaxX do
  begin
    RawPos := Self.GetRawPos(CntX, 0, pDepth);
    for CntY := 0 to MaxY do
    begin
      {$IFDEF FPC}
      FData[RawPos] += Value;
      {$ELSE}
      FData[RawPos] := FData[RawPos] + Value;
      {$ENDIF}
      Inc(RawPos, RowStride);
    end;
  end;
end;

procedure TVolume.AddAtDepth(pDepth: integer; Original: TVolume);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  RawPos, RowStride: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  if Self.Size = Original.Size then
  begin
    RowStride := FSizeX * FDepth; // per-CntY step; same shape indexes both
    for CntX := 0 to MaxX do
    begin
      RawPos := Self.GetRawPos(CntX, 0, pDepth);
      for CntY := 0 to MaxY do
      begin
        {$IFDEF FPC}
        FData[RawPos] += Original.FData[RawPos];
        {$ELSE}
        FData[RawPos] := FData[RawPos] + Original.FData[RawPos];
        {$ENDIF}
        Inc(RawPos, RowStride);
      end;
    end;
  end
  else
  begin
    WriteLn('To Be Implemented.');
  end;
end;

procedure TVolume.AddFromDepthToDepth(Original: TVolume; FromDepth,
  ToDepth: integer);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  RawPos, SrcPos, RowStride: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  if Self.Size = Original.Size then
  begin
    RowStride := FSizeX * FDepth; // per-CntY step (same shape for both volumes)
    for CntX := 0 to MaxX do
    begin
      RawPos := Self.GetRawPos(CntX, 0, ToDepth);
      SrcPos := Original.GetRawPos(CntX, 0, FromDepth);
      for CntY := 0 to MaxY do
      begin
        {$IFDEF FPC}
        FData[RawPos] += Original.FData[SrcPos];
        {$ELSE}
        FData[RawPos] := FData[RawPos] + Original.FData[SrcPos];
        {$ENDIF}
        Inc(RawPos, RowStride);
        Inc(SrcPos, RowStride);
      end;
    end;
  end
  else
  begin
    WriteLn('To Be Implemented.');
  end;
end;

procedure TVolume.AddTransposingXD(Original: TVolume);
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  DestBase, SrcPos, SrcStride: integer;
begin
  ReSize(Original.Depth, Original.SizeY, Original.SizeX);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  SrcStride := Original.FDepth; // Original X-slot step per CntD
  if MaxY > 0 then
  begin
    for CntX := 0 to MaxX do
    begin
      for CntY := 0 to MaxY do
      begin
        DestBase := GetRawPos(CntX, CntY, 0);
        SrcPos := Original.GetRawPos(0, CntY, CntX);
        for CntD := 0 to MaxD do
        begin
          FData[DestBase + CntD] := FData[DestBase + CntD] + Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
        end;
      end;
    end;
  end
  else
  begin
    for CntX := 0 to MaxX do
    begin
      DestBase := GetRawPos(CntX, 0, 0);
      SrcPos := Original.GetRawPos(0, 0, CntX);
      for CntD := 0 to MaxD do
      begin
          FData[DestBase + CntD] := FData[DestBase + CntD] + Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
      end;
    end;
  end;
end;

procedure TVolume.AddTransposingYD(Original: TVolume);
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  DestBase, SrcPos, SrcStride: integer;
begin
  ReSize(Original.SizeX, Original.Depth, Original.SizeY);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  SrcStride := Original.FSizeX * Original.FDepth; // Original Y-slot step per CntD
  if MaxX > 0 then
  begin
    for CntX := 0 to MaxX do
    begin
      for CntY := 0 to MaxY do
      begin
        DestBase := GetRawPos(CntX, CntY, 0);
        SrcPos := Original.GetRawPos(CntX, 0, CntY);
        for CntD := 0 to MaxD do
        begin
          FData[DestBase + CntD] := FData[DestBase + CntD] + Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
        end;
      end;
    end;
  end
  else
  begin
    for CntY := 0 to MaxY do
    begin
      DestBase := GetRawPos(0, CntY, 0);
      SrcPos := Original.GetRawPos(0, 0, CntY);
      for CntD := 0 to MaxD do
      begin
        FData[DestBase + CntD] := FData[DestBase + CntD] + Original.FData[SrcPos];
        Inc(SrcPos, SrcStride);
      end;
    end;
  end;
end;

procedure TVolume.AddTransposingAs2D(Original: TVolume);
var
  OriginalSizeX, OriginalSizeY, OriginalDepth: integer;
begin
  OriginalSizeX := Original.SizeX;
  OriginalSizeY := Original.SizeY;
  OriginalDepth := Original.Depth;
  Original.ReSize(OriginalSizeX*OriginalSizeY, 1, OriginalDepth);
  AddTransposingXD(Original);
  Original.ReSize(OriginalSizeX, OriginalSizeY, OriginalDepth);
end;

procedure TVolume.CopyFromDepthToDepth(Original: TVolume; FromDepth,
  ToDepth: integer);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  RawPos, SrcPos, RowStride: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  if Self.Size = Original.Size then
  begin
    RowStride := FSizeX * FDepth; // per-CntY step (same shape for both volumes)
    for CntX := 0 to MaxX do
    begin
      RawPos := Self.GetRawPos(CntX, 0, ToDepth);
      SrcPos := Original.GetRawPos(CntX, 0, FromDepth);
      for CntY := 0 to MaxY do
      begin
        FData[RawPos] := Original.FData[SrcPos];
        Inc(RawPos, RowStride);
        Inc(SrcPos, RowStride);
      end;
    end;
  end
  else
  begin
    WriteLn('To Be Implemented.');
  end;
end;

procedure TVolume.AddLayers(A,B: TVolume);
var
  I,J,K: integer;
  MaxX, MaxY, MaxD: integer;
  ASizeXM1, ASizeYM1, ADepthM1, BSizeXM1, BSizeYM1, BDepthM1: integer;
  SelfBase, SrcBase: integer;
begin
  MaxX := Max(A.FSizeX, B.FSizeX);
  MaxY := Max(A.FSizeX, B.FSizeX);
  MaxD := A.FDepth + B.FDepth;
  Resize(MaxX,MaxY,MaxD);

  if (A.FDepth>0) and (A.FSizeX > 0) and (A.FSizeY > 0) then
  begin
    ASizeXM1 := A.FSizeX - 1;
    ASizeYM1 := A.FSizeY - 1;
    ADepthM1 := A.FDepth - 1;
    for I := 0 to ASizeXM1 do
    begin
      for J := 0 to ASizeYM1 do
      begin
        SelfBase := GetRawPos(I, J, 0);
        SrcBase := A.GetRawPos(I, J, 0);
        Move(A.FData[SrcBase], FData[SelfBase], A.FDepth * csNeuralFloatSize);
      end;
    end;
  end;

  if (B.FDepth>0) and (B.FSizeX > 0) and (B.FSizeY > 0) then
  begin
    BSizeXM1 := B.FSizeX - 1;
    BSizeYM1 := B.FSizeY - 1;
    BDepthM1 := B.FDepth - 1;
    for I := 0 to BSizeXM1 do
    begin
      for J := 0 to BSizeYM1 do
      begin
        SelfBase := GetRawPos(I, J, 0);
        SrcBase := B.GetRawPos(I, J, 0);
        Move(B.FData[SrcBase], FData[SelfBase + A.FDepth], B.FDepth * csNeuralFloatSize);
      end;
    end;
  end;
end;

procedure TVolume.Sub(Original: TVolume);
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at Sub: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] -= Original.FData[I];
    {$ELSE}
    FData[I] := FData[I] - Original.FData[I];
    {$ENDIF}
end;

procedure TVolume.Sub(Value: T);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] -= Value;
    {$ELSE}
    FData[I] := FData[I] - Value;
    {$ENDIF}
end;

procedure TVolume.Diff(Original: TVolume);
var
  I: integer;
  vHigh: integer;
  AuxSingle: Single;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at Diff: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  vHigh := High(FData);
  for I := 0 to vHigh do
  begin
    AuxSingle := FData[I] - Original.FData[I];
    FData[I] := Abs(AuxSingle);
  end;
end;

procedure TVolume.InterleaveWithDepthFrom(Original: TVolume; NewDepth: integer);
var
  NewX: integer;
  I: integer;
  vHigh: integer;
  posX, posD, maxPosX: integer;
begin
  NewX := Original.FSize div NewDepth;
  Resize(NewX,1,NewDepth);

  vHigh := High(FData);

  posX := 0;
  posD := 0;

  maxPosX := NewX * NewDepth;

  for I := 0 to vHigh do
  begin
    //posX := I mod NewX;
    //posD := I div NewX;
    //Self.Data[posX, 0, posD] := Original.FData[I];

    FData[posX + posD] := Original.FData[I];

    {$IFDEF FPC}
    posX += NewDepth;
    {$ELSE}
    posX := posX + NewDepth;
    {$ENDIF}

    if posX >= maxPosX then
    begin
      posX := 0;
      posD := posD + 1;
    end;
  end;
end;
(*
// this is a new version to be validated.
var
  NewX: integer;
  I: integer;
  vHigh: integer;
  posX, posD, maxPosX: integer;
  NewDepth2, NewDepth3, NewDepth4, vHighM4: integer;
  SourcePtr, DestPtr: TNeuralFloatPtr;
begin
  NewX := Original.FSize div NewDepth;
  Resize(NewX,1,NewDepth);
  NewDepth2 := NewDepth  + NewDepth;
  NewDepth3 := NewDepth2 + NewDepth;
  NewDepth4 := NewDepth3 + NewDepth;

  vHigh := High(FData);
  vHighM4 := vHigh - 4;

  posX := 0;
  posD := 0;

  maxPosX := NewX * NewDepth;

  SourcePtr := Addr(Original.FData[0]);
  DestPtr := Addr(FData[posX + posD]);

  //for I := 0 to vHigh do
  I := 0;
  while I <= vHigh do
  begin
    //posX := I mod NewX;
    //posD := I div NewX;
    //Self.Data[posX, 0, posD] := Original.FData[I];
    while ( (I<vHighM4) and (posX + NewDepth4 < maxPosX) ) do
    begin
      (DestPtr            )^ := (SourcePtr)^;
      (DestPtr + NewDepth )^ := (SourcePtr+1)^;
      (DestPtr + NewDepth2)^ := (SourcePtr+2)^;
      (DestPtr + NewDepth3)^ := (SourcePtr+3)^;
      Inc(I, 4);
      Inc(posX, NewDepth4);
      Inc(SourcePtr,4);
      Inc(DestPtr, NewDepth4);
    end;

    (DestPtr)^ := (SourcePtr)^;
    Inc(SourcePtr, 1);
    Inc(posX, NewDepth);
    Inc(I);

    if I <= vHigh then
    begin
      if posX >= maxPosX then
      begin
        posX := 0;
        posD := posD + 1;
        DestPtr := Addr(FData[posX + posD]);
      end
      else
      begin
        Inc(DestPtr, NewDepth);
      end;
    end;
  end;
end;
*)

procedure TVolume.InterleaveWithXFrom(Original: TVolume; NewX: integer);
begin
  InterleaveWithDepthFrom(Original, Original.FSize div NewX);
end;

function TVolume.IncYSize(): integer;
begin
  Result := GetRawPos(0, 1);
end;

function TVolume.IncYSizeBytes(): integer;
begin
  Result := IncYSize() * csNeuralFloatSize;
end;

function TVolume.SameSize(Original: TVolume): boolean;
begin
  Result :=
    (Self.SizeX = Original.SizeX) and
    (Self.SizeY = Original.SizeY) and
    (Self.Depth = Original.Depth);
end;

procedure TVolume.DeInterleaveWithXFrom(Original: TVolume; NewX: integer);
begin
  InterleaveWithDepthFrom(Original, NewX);
end;

procedure TVolume.DeInterleaveWithDepthFrom(Original: TVolume; NewDepth: integer
  );
begin
  InterleaveWithXFrom(Original, NewDepth);
end;

procedure TVolume.SetMin(Value: TNeuralFloat);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := Min(FData[I], Value);
end;

procedure TVolume.SetMax(Value: TNeuralFloat);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := Max(FData[I], Value);
end;

procedure TVolume.Mul(Original: TVolume);
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at Mul: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] *= Original.FData[I];
    {$ELSE}
    FData[I] := FData[I] * Original.FData[I];
    {$ENDIF}
end;

class procedure TVolume.Mul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat;
  pSize: integer);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := pSize - 1;
  for I := 0 to vHigh do
    {$IFDEF FPC}
    PtrA^[I] *= MulOp;
    {$ELSE}
    PtrA^[I] := PtrA^[I] * MulOp;
    {$ENDIF}
end;

class procedure TVolume.Mul(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := pSize - 1;
  for I := 0 to vHigh do
    {$IFDEF FPC}
    PtrA^[I] *= PtrB^[I];
    {$ELSE}
    PtrA^[I] := PtrA^[I] * PtrB^[I];
    {$ENDIF}
end;

procedure TVolume.Mul(Value: T);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] *= Value;
    {$ELSE}
    FData[I] := FData[I] * Value;
    {$ENDIF}
end;

procedure TVolume.MulAtDepth(pDepth: integer; Value: T);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  RawPos, RowStride: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step

  for CntX := 0 to MaxX do
  begin
    RawPos := Self.GetRawPos(CntX, 0, pDepth);
    for CntY := 0 to MaxY do
    begin
      {$IFDEF FPC}
      FData[RawPos] *= Value;
      {$ELSE}
      FData[RawPos] := FData[RawPos] * Value;
      {$ENDIF}
      Inc(RawPos, RowStride);
    end;
  end;
end;

procedure TVolume.Pow(Value: T);
var
  I: integer;
  vHigh: integer;
begin
  if Value <> 1 then
  begin
    vHigh := High(FData);
    for I := 0 to vHigh do
      FData[I] := pcr_powf(FData[I],Value);
  end;
end;

procedure TVolume.PowMinus1();
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
  begin
    if FData[I] <> 0 then FData[I] := (1/FData[I]);
  end;
end;

procedure TVolume.VSqrt();
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := pcr_sqrtf(FData[I]);
end;

procedure TVolume.MulAdd(Value: T; Original: TVolume);
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at MulAdd: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  MulAdd(Value, Addr(Original.FData[0]));
end;

procedure TVolume.MulMulAdd(Value1, Value2: T; Original: TVolume);
begin
  MulMulAdd(Addr(Self.FData[0]), Addr(Original.FData[0]), Value1, Value2, Self.Size);
end;

procedure TVolume.MulAdd(Value: T; PtrB: TNeuralFloatArrPtr);
begin
  MulAddPPVS(TNeuralFloatArrPtr(Addr(Self.FData[0])), PtrB, Value, Self.Size);
end;

procedure TVolume.MulAdd(Original1, Original2: TVolume);
begin
  {$IFDEF Debug}
  if Original1.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at MulAdd parameter 1: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original1.Size) + ' .');
  if Original2.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at MulAdd parameter 2: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original2.Size) + ' .');
  {$ENDIF}
  MulAdd(Addr(Self.FData[0]), Addr(Original1.FData[0]), Addr(Original2.FData[0]), Self.Size);
end;

class procedure TVolume.MulAddPPVS(PtrA, PtrB: TNeuralFloatArrPtr; Value: T;
  pSize: integer);
var
  I: integer;
  vHigh: integer;
  BasePos: integer;
  {$IFDEF FPC}
  AddrA, AddrB: TNeuralFloatPtr;
  {$ENDIF}
begin
  BasePos := 0;
  vHigh := pSize - 1;

  {$IFDEF FPC}
  AddrA := pointer(PtrA);
  AddrB := pointer(PtrB);
  while BasePos <= vHigh - 7 do
  begin
    (AddrA)^   := (AddrA)^   + (AddrB)^   * Value;
    (AddrA+1)^ := (AddrA+1)^ + (AddrB+1)^ * Value;
    (AddrA+2)^ := (AddrA+2)^ + (AddrB+2)^ * Value;
    (AddrA+3)^ := (AddrA+3)^ + (AddrB+3)^ * Value;
    (AddrA+4)^ := (AddrA+4)^ + (AddrB+4)^ * Value;
    (AddrA+5)^ := (AddrA+5)^ + (AddrB+5)^ * Value;
    (AddrA+6)^ := (AddrA+6)^ + (AddrB+6)^ * Value;
    (AddrA+7)^ := (AddrA+7)^ + (AddrB+7)^ * Value;
    BasePos := BasePos + 8;
    AddrA := AddrA + 8;
    AddrB := AddrB + 8;
  end;

  while BasePos <= vHigh - 3 do
  begin
    (AddrA)^   := (AddrA)^   + (AddrB)^   * Value;
    (AddrA+1)^ := (AddrA+1)^ + (AddrB+1)^ * Value;
    (AddrA+2)^ := (AddrA+2)^ + (AddrB+2)^ * Value;
    (AddrA+3)^ := (AddrA+3)^ + (AddrB+3)^ * Value;
    BasePos := BasePos + 4;
    AddrA := AddrA + 4;
    AddrB := AddrB + 4;
  end;
  {$ENDIF}

  if BasePos <= vHigh then for I := BasePos to vHigh do
  begin
    //Write(PtrA^[I],' ', PtrB^[I],' ', Value,'->');
    {$IFDEF FPC}
    PtrA^[I] += PtrB^[I]*Value;
    {$ELSE}
    PtrA^[I] := PtrA^[I] + PtrB^[I]*Value;
    {$ENDIF}
    //WriteLn(PtrA^[I]);
  end;
end;

class procedure TVolume.MulMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value1,
  Value2: T; pSize: integer);
var
  I: integer;
  vHigh: integer;
  BasePos: integer;
  {$IFDEF FPC}
  AddrA, AddrB: TNeuralFloatPtr;
  {$ENDIF}
begin
  BasePos := 0;
  vHigh := pSize - 1;
  {$IFDEF FPC}
  AddrA := pointer(PtrA);
  AddrB := pointer(PtrB);
  while BasePos <= vHigh - 7 do
  begin
    (AddrA)^   := (AddrA)^   * Value1 + (AddrB)^   * Value2;
    (AddrA+1)^ := (AddrA+1)^ * Value1 + (AddrB+1)^ * Value2;
    (AddrA+2)^ := (AddrA+2)^ * Value1 + (AddrB+2)^ * Value2;
    (AddrA+3)^ := (AddrA+3)^ * Value1 + (AddrB+3)^ * Value2;
    (AddrA+4)^ := (AddrA+4)^ * Value1 + (AddrB+4)^ * Value2;
    (AddrA+5)^ := (AddrA+5)^ * Value1 + (AddrB+5)^ * Value2;
    (AddrA+6)^ := (AddrA+6)^ * Value1 + (AddrB+6)^ * Value2;
    (AddrA+7)^ := (AddrA+7)^ * Value1 + (AddrB+7)^ * Value2;
    BasePos := BasePos + 8;
    AddrA := AddrA + 8;
    AddrB := AddrB + 8;
  end;

  while BasePos <= vHigh - 3 do
  begin
    (AddrA)^   := (AddrA)^   * Value1 + (AddrB)^   * Value2;
    (AddrA+1)^ := (AddrA+1)^ * Value1 + (AddrB+1)^ * Value2;
    (AddrA+2)^ := (AddrA+2)^ * Value1 + (AddrB+2)^ * Value2;
    (AddrA+3)^ := (AddrA+3)^ * Value1 + (AddrB+3)^ * Value2;
    BasePos := BasePos + 4;
    AddrA := AddrA + 4;
    AddrB := AddrB + 4;
  end;
  {$ENDIF}
  if BasePos <= vHigh then for I := BasePos to vHigh do
    PtrA^[I] := PtrA^[I] * Value1 + PtrB^[I] * Value2;
end;


class procedure TVolume.MulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value: T;
  pSize: integer);
begin
  Self.MulAddPPVS(PtrA, PtrB, Value, pSize);
end;

class procedure TVolume.RankOneUpdateRow(PtrDst, PtrPrev, PtrB: TNeuralFloatArrPtr;
  AlphaScale, BScale: T; pSize: integer);
begin
  // Dst := AlphaScale*Prev + BScale*B, with Prev=nil meaning the zero row.
  if (PtrPrev = nil) or (AlphaScale = 0) then
  begin
    // Dst := BScale*B (no prev carry).
    Move(PtrB^, PtrDst^, pSize * SizeOf(T));
    TVolume.Mul(PtrDst, BScale, pSize);
  end
  else
  begin
    if PtrPrev <> PtrDst then Move(PtrPrev^, PtrDst^, pSize * SizeOf(T));
    TVolume.Mul(PtrDst, AlphaScale, pSize);  // Dst := AlphaScale*Prev
    TVolume.MulAdd(PtrDst, PtrB, BScale, pSize);  // Dst += BScale*B
  end;
end;

class procedure TVolume.MulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr;
  pSize: integer);
var
  I: integer;
  vHigh: integer;
  BasePos: integer;
  {$IFDEF FPC}
  AddrA, AddrB, AddrC: TNeuralFloatPtr;
  {$ENDIF}
begin
  BasePos := 0;
  {$IFDEF FPC}
  AddrA := pointer(PtrA);
  AddrB := pointer(PtrB);
  AddrC := pointer(PtrC);
  {$ENDIF}
  vHigh := pSize - 1;
  {$IFDEF FPC}
  while BasePos <= vHigh - 7 do
  begin
    (AddrA)^   := (AddrA)^   + (AddrB)^   * (AddrC)^;
    (AddrA+1)^ := (AddrA+1)^ + (AddrB+1)^ * (AddrC+1)^;
    (AddrA+2)^ := (AddrA+2)^ + (AddrB+2)^ * (AddrC+2)^;
    (AddrA+3)^ := (AddrA+3)^ + (AddrB+3)^ * (AddrC+3)^;
    (AddrA+4)^ := (AddrA+4)^ + (AddrB+4)^ * (AddrC+4)^;
    (AddrA+5)^ := (AddrA+5)^ + (AddrB+5)^ * (AddrC+5)^;
    (AddrA+6)^ := (AddrA+6)^ + (AddrB+6)^ * (AddrC+6)^;
    (AddrA+7)^ := (AddrA+7)^ + (AddrB+7)^ * (AddrC+7)^;
    BasePos := BasePos + 8;
    AddrA := AddrA + 8;
    AddrB := AddrB + 8;
    AddrC := AddrC + 8;
  end;

  while BasePos <= vHigh - 3 do
  begin
    (AddrA)^   := (AddrA)^   + (AddrB)^   * (AddrC)^;
    (AddrA+1)^ := (AddrA+1)^ + (AddrB+1)^ * (AddrC+1)^;
    (AddrA+2)^ := (AddrA+2)^ + (AddrB+2)^ * (AddrC+2)^;
    (AddrA+3)^ := (AddrA+3)^ + (AddrB+3)^ * (AddrC+3)^;
    BasePos := BasePos + 4;
    AddrA := AddrA + 4;
    AddrB := AddrB + 4;
    AddrC := AddrC + 4;
  end;
  {$ENDIF}
  if BasePos <= vHigh then for I := BasePos to vHigh do
    {$IFDEF FPC}
    PtrA^[I] += PtrB^[I]*PtrC^[I];
    {$ELSE}
    PtrA^[I] := PtrA^[I] + PtrB^[I]*PtrC^[I];
    {$ENDIF}
end;

procedure TVolume.Divi(Original: TVolume);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] /= Original.FData[I];
    {$ELSE}
    FData[I] := FData[I] / Original.FData[I];
    {$ENDIF}
end;

procedure TVolume.Divi(Value: T);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    {$IFDEF FPC}
    FData[I] /= Value;
    {$ELSE}
    FData[I] := FData[I] / Value;
    {$ENDIF}
end;

procedure TVolume.ForceMinRange(Value: T);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := NeuronForceMinRange(FData[I], Value);
end;

procedure TVolume.ForceMaxRange(Value: T);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := NeuronForceRange(FData[I], Value);
end;

procedure TVolume.ForceMaxMagnitude(Value: T);
var
  VNorm: Single;
begin
  VNorm := GetMagnitude();
  if VNorm > Value then Mul(Value/VNorm);
end;

procedure TVolume.ForceMaxAbs(Value: T);
var
  VMaxAbs, VFix: Single;
begin
  VMaxAbs := GetMaxAbs();
  if VMaxAbs > Value then
  begin
    VFix := Value/VMaxAbs;
    Self.Mul( VFix );
    WriteLn(VMaxAbs:6:2);
  end;
end;

function TVolume.HasNonFinite(): boolean;
var
  I, MaxIdx: integer;
begin
  Result := false;
  MaxIdx := FSize - 1;
  for I := 0 to MaxIdx do
  begin
    if IsNan(FData[I]) or IsInfinite(FData[I]) then
    begin
      Result := true;
      Exit;
    end;
  end;
end;

procedure TVolume.ForcePositive();
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    if FData[I] < 0 then FData[I] := -FData[I];
end;

destructor TVolume.Destroy();
begin
  SetLength(FData, 0);
  inherited Destroy;
end;

procedure TVolume.Fill(c: T);
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := c;
end;

procedure TVolume.FillForIdx(c: T; const aIdx: array of integer);
var
  Idx: integer;
begin
  for Idx in aIdx do
  begin
    FData[Idx] := c;
  end;
end;

procedure TVolume.FillAtDepth(pDepth: integer; Value: T);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  RawPos, RowStride: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step

  for CntX := 0 to MaxX do
  begin
    RawPos := Self.GetRawPos(CntX, 0, pDepth);
    for CntY := 0 to MaxY do
    begin
      FData[RawPos] := Value;
      Inc(RawPos, RowStride);
    end;
  end;
end;

procedure TVolume.FillForDebug();
var
  I: integer;
  vHigh: integer;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
    FData[I] := I/100;
end;

procedure TVolume.Resize(pSize: integer);
begin
  ReSize(1, 1, pSize);
end;

procedure TVolume.ReSize(pSizeX, pSizeY, pDepth: integer);
var
  NewSize: integer;
begin
  if (pSizeX<>FSizeX) or (pSizeY<>FSizeY) or (pDepth<>FDepth) then
  begin
    NewSize := pSizeX * pSizeY * pDepth;
    if (NewSize <> FSize) then
    begin
      FSize := NewSize;
      SetLength(FData, FSize);
    end;
    FSizeX := pSizeX;
    FSizeY := pSizeY;
    FDepth := pDepth;
  end;
end;

procedure TVolume.ReSize(Original: TVolume);
begin
  Resize(Original.SizeX, Original.SizeY, Original.Depth);
end;

function TVolume.Get(x, y, d: integer): T;
begin
  Result := FData[((FSizeX * y) + x) * FDepth + d];
end;

function TVolume.GetAsByte(x, y, d: integer): byte;
begin
  Result := RoundAsByte(Get(x, y, d));
end;

function TVolume.GetRaw(x: integer): T;
begin
  Result := FData[x];
end;

procedure TVolume.SetRaw(X: integer; Value: T);
begin
  FData[x] := Value;
end;

procedure TVolume.Store(x, y, d: integer; Value: T);
begin
  FData[((FSizeX * y) + x) * FDepth + d] := Value;
end;

procedure TVolume.Add(x, y, d: integer; Value: T);
{$IFDEF FPC}
begin
  FData[((FSizeX * y) + x) * FDepth + d] += Value;
end;
{$ELSE}
var
  Idx: integer;
begin
  Idx := ((FSizeX * y) + x) * FDepth + d;
  FData[Idx] := FData[Idx] + Value;
end;
{$ENDIF}

procedure TVolume.Sub(x, y, d: integer; Value: T);
{$IFDEF FPC}
begin
  FData[((FSizeX * y) + x) * FDepth + d] -= Value;
end;
{$ELSE}
var
  Idx: integer;
begin
  Idx := ((FSizeX * y) + x) * FDepth + d;
  FData[Idx] := FData[Idx] - Value;
end;
{$ENDIF}

procedure TVolume.Mul(x, y, d: integer; Value: T);
{$IFDEF FPC}
begin
  FData[((FSizeX * y) + x) * FDepth + d] *= Value;
end;
{$ELSE}
var
  Idx: integer;
begin
  Idx := ((FSizeX * y) + x) * FDepth + d;
  FData[Idx] := FData[Idx] * Value;
end;
{$ENDIF}

procedure TVolume.Divi(x, y, d: integer; Value: T);
{$IFDEF FPC}
begin
  FData[((FSizeX * y) + x) * FDepth + d] /= Value;
end;
{$ELSE}
var
  Idx: integer;
begin
  Idx := ((FSizeX * y) + x) * FDepth + d;
  FData[Idx] := FData[Idx] / Value;
end;
{$ENDIF}

procedure TVolume.Copy(Original: TVolume);
begin
  if Original.Size > 0 then
  begin
    if Original.Size <> Self.Size then
    begin
      Self.ReSize(Original);
    end;
    CopyNoChecks(Original);
  end;
end;

procedure TVolume.CopyRelu(Original: TVolume);
var
  OriginalCnt, OriginalMax: integer;
begin
  OriginalMax := Original.Size - 1;
  if OriginalMax >= 0 then
  begin
    if Original.Size <> Self.Size then
    begin
      Self.ReSize(Original);
    end;
    for OriginalCnt := 0 to OriginalMax do
    begin
      if Original.FData[OriginalCnt] > 0.0
        then FData[OriginalCnt] := Original.FData[OriginalCnt]
        else FData[OriginalCnt] := 0;
    end;
  end;
end;


procedure TVolume.Copy(Original: TVolume; Len: integer);
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at Copy: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Move(Original.FData[0], Self.FData[0], Len * SizeOf(T));
end;

procedure TVolume.CopyNoChecks(Original: TVolume);
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at CopyNoChecks: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Move(Original.FData[0], Self.FData[0], Self.Size * SizeOf(T));
end;

procedure TVolume.CopyNoChecks(var Original: array of byte);
var
  I: integer;
  vHigh: integer;
begin
  if Length(Original) > 0 then
  begin
    vHigh := High(Original);
    for I := 0 to vHigh do
    begin
      FData[I] := Original[I];
    end;
  end;
end;

procedure TVolume.CopyNoChecksIntArr(var Original: array of integer);
var
  I: integer;
  vHigh: integer;
begin
  if Length(Original) > 0 then
  begin
    vHigh := High(Original);
    for I := 0 to vHigh do
    begin
      FData[I] := Original[I];
    end;
  end;
end;

procedure TVolume.CopyReversedNoChecksIntArr(var Original: array of integer);
var
  I: integer;
  MaxLen: integer;
begin
  MaxLen := Length(Original) - 1;
  if MaxLen >= 0 then
  begin
    for I := 0 to MaxLen do
    begin
      FData[I] := Original[MaxLen - I];
    end;
  end;
end;

procedure TVolume.CopyNoChecks(var Original: string);
var
  I: integer;
  LenOriginal: integer;
begin
  LenOriginal := Length(Original);
  if LenOriginal > 0 then
  begin
    for I := 1 to LenOriginal do
    begin
      FData[I-1] := Ord(Original[I]);
    end;
  end;
end;

procedure TVolume.CopyReversedNoChecks(var Original: string);
var
  I: integer;
  LenOriginal: integer;
begin
  LenOriginal := Length(Original);
  if LenOriginal > 0 then
  begin
    for I := 1 to LenOriginal do
    begin
      FData[I-1] := Ord(Original[LenOriginal - I + 1]);
    end;
  end;
end;

procedure TVolume.CopyChannels(Original: TVolume; aChannels: array of integer);
var
  MaxX, MaxY: integer;
  X, Y, InputDepth, OutputDepth: integer;
  SelfBase, OrigBase: integer;
begin
  Resize(Original.SizeX, Original.SizeY, Length(aChannels));

  MaxX := SizeX - 1;
  MaxY := SizeY - 1;

  for X := 0 to MaxX do
  begin
    for Y := 0 to MaxY do
    begin
      SelfBase := GetRawPos(X, Y, 0);
      OrigBase := Original.GetRawPos(X, Y, 0);
      OutputDepth := 0;
      for InputDepth in aChannels do
      begin
        FData[SelfBase + OutputDepth] := Original.FData[OrigBase + InputDepth];
        Inc(OutputDepth);
      end;
    end;
  end;
end;

procedure TVolume.Define(Original: array of T);
begin
  Copy(Original);
end;

// This function doesn't check for sizes - use it with care
procedure TVolume.ReadAsBits(var Dest: array of byte; Threshold: T);
var
  I: integer;
  vHigh: integer;
begin
  if Length(Dest) > 0 then
  begin
    BAClear(Dest);
    vHigh := Self.FSize - 1;
    for I := 0 to vHigh do
    begin
      if ( FData[I] > Threshold ) then
      begin
        BAWrite(Dest,I,1);
      end;
    end;
  end;
end;

procedure TVolume.Copy(var Original: array of T);
begin
  if Length(Original) > 0 then
  begin
    if (Length(Original) <> Self.Size) then
    begin
      Self.ReSize(Length(Original), 1, 1);
    end;
    Move(Original[0], Self.FData[0], Self.Size * SizeOf(T));
  end;
end;

procedure TVolume.Copy(var Original: array of byte);
var
  I: integer;
  vHigh: integer;
begin
  if Length(Original) > 0 then
  begin
    if (Length(Original) <> Self.Size) then
    begin
      Self.ReSize(Length(Original), 1, 1);
    end;

    vHigh := High(Original);

    for I := 0 to vHigh do
    begin
      FData[I] := Original[I];
    end;
  end;
end;

procedure TVolume.Copy(Original: TBits; pFlase: T = -0.5; pTrue: T = +0.5);
var
  I: integer;
  vHigh: integer;
  aTranslate: array [false..true] of T;
begin
  if Original.Size > 0 then
  begin
    aTranslate[false] := pFlase;
    aTranslate[true]  := pTrue;

    if (Original.Size <> Self.Size) then
    begin
      if Original.Size mod 8 = 0 then
      begin
        Self.ReSize(Original.Size div 8, 1, 8);
      end else
      begin
        Self.ReSize(Original.Size, 1, 1);
      end;
    end;

    vHigh := Original.Size - 1;

    for I := 0 to vHigh do
    begin
      FData[I] := aTranslate[Original[I]];
    end;
  end;
end;

procedure TVolume.CopyAsBits(var Original: array of byte; pFalse: T = -0.5; pTrue: T = +0.5; CanResize:boolean = True);
var
  I: integer;
  vHigh: integer;
  LenOriginal: integer;
  aTranslate: array [0..1] of T;
begin
  LenOriginal := Length(Original);
  if LenOriginal > 0 then
  begin
    if CanResize and (LenOriginal*8 <> Self.Size) then
    begin
      Self.ReSize(LenOriginal, 1, 8);
    end;

    vHigh := LenOriginal * 8 - 1;
    aTranslate[0] := pFalse;
    aTranslate[1] := pTrue;

    for I := 0 to vHigh do
    begin
      FData[I] := aTranslate[BARead(Original,I)];
    end;
  end;
end;

procedure TVolume.CopyAsBits(Original: string; pFalse: T; pTrue: T; CanResize:boolean);
var
  AB: array of byte;
  I: integer;
  vHigh: integer;
  LenOriginal: integer;
begin
  LenOriginal := Length(Original);
  if LenOriginal > 0 then
  begin
    SetLength(AB, LenOriginal);
    vHigh := LenOriginal;
    for I := 1 to vHigh do
    begin
      AB[I-1] := Min(Ord(Original[I]), 255);
    end;
    Self.CopyAsBits(AB, pFalse, pTrue, CanResize);
  end;
end;

procedure TVolume.CopyAsBitsReversed(Original: string; pFalse: T; pTrue: T);
var
  AB: array of byte;
  I: integer;
  vHigh: integer;
  LenOriginal: integer;
begin
  LenOriginal := Length(Original);
  if LenOriginal > 0 then
  begin
    SetLength(AB, LenOriginal);
    vHigh := LenOriginal;
    for I := 1 to vHigh do
    begin
      AB[I-1] := Min(Ord(Original[vHigh-I+1]), 255);
    end;
    Self.CopyAsBits(AB, pFalse, pTrue, False);
    SetLength(AB, 0);
  end;
end;

(*
procedure TVolume.CopyPadding(Original: TVolume; Padding: integer);
var
  CntX, CntY, CntD: integer;
  NewSizeX, NewSizeY: integer;
  MaxX, MaxY, MaxD: integer;
begin
  NewSizeX := Original.SizeX + Padding * 2;
  NewSizeY := Original.SizeY + Padding * 2;
  MaxX := Original.SizeX - 1;
  MaxY := Original.SizeY - 1;
  MaxD := Original.Depth - 1;

  Resize(NewSizeX, NewSizeY, Original.Depth);
  Fill(0);

  for CntX := 0 to MaxX do
  begin
    for CntY := 0 to MaxY do
    begin
      for CntD := 0 to MaxD do
      begin
        Data[CntX + Padding, CntY + Padding, CntD] :=
          Original[CntX, CntY, CntD];
      end;
    end;
  end;
end;
*)

procedure TVolume.CopyPadding(Original: TVolume; Padding: integer);
var
  CntY: integer;
  NewSizeX, NewSizeY: integer;
  MaxY: integer;
  RowSize: integer;
  SourceRawPos, DestRawPos: integer;
begin
  NewSizeX := Original.SizeX + Padding * 2;
  NewSizeY := Original.SizeY + Padding * 2;
  MaxY := Original.SizeY - 1;
  RowSize := Original.SizeX * Original.Depth * csNeuralFloatSize;

  Resize(NewSizeX, NewSizeY, Original.Depth);
  Fill(0);

  for CntY := 0 to MaxY do
  begin
    SourceRawPos := Original.GetRawPos(0, CntY, 0);
    DestRawPos := GetRawPos(Padding, CntY + Padding, 0);
    Move(Original.FData[SourceRawPos], Self.FData[DestRawPos], RowSize);
  end;
end;

procedure TVolume.CopyPadding(Original: TVolume; PaddingX, PaddingY: integer);
var
  CntY: integer;
  NewSizeX, NewSizeY: integer;
  MaxY: integer;
  RowSize: integer;
  SourceRawPos, DestRawPos: integer;
begin
  NewSizeX := Original.SizeX + PaddingX * 2;
  NewSizeY := Original.SizeY + PaddingY * 2;
  MaxY := Original.SizeY - 1;
  RowSize := Original.SizeX * Original.Depth * csNeuralFloatSize;

  Resize(NewSizeX, NewSizeY, Original.Depth);
  Fill(0);

  for CntY := 0 to MaxY do
  begin
    SourceRawPos := Original.GetRawPos(0, CntY, 0);
    DestRawPos := GetRawPos(PaddingX, CntY + PaddingY, 0);
    Move(Original.FData[SourceRawPos], Self.FData[DestRawPos], RowSize);
  end;
end;

procedure TVolume.CopyCropping(Original: TVolume; StartX, StartY, pSizeX,
  pSizeY: integer);
var
  CountY: integer;
  MaxY, MoveSizeBytes: integer;
  RawPostDest, RawPosSource: integer;
begin
  Resize(pSizeX, pSizeY, Original.Depth);

  MaxY := SizeY - 1;
  MoveSizeBytes := Depth * SizeOf(T) * SizeX;

  for CountY := 0 to MaxY do
  begin
    RawPostDest := GetRawPos(0, CountY);
    RawPosSource := Original.GetRawPos(StartX, CountY+StartY);
    Move(Original.FData[RawPosSource], FData[RawPostDest], MoveSizeBytes);
  end;
end;

procedure TVolume.CopyResizing(Original: TVolume; NewSizeX, NewSizeY: integer);
var
  RatioX, RatioY: TNeuralFloat;
  InvRatioX, InvRatioY: TNeuralFloat;
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  OrigMaxX, OrigMaxY: integer;
  OrigPosX, OrigPosY: integer;
  MoveSizeBytes: integer;
  RawPostDest, RawPosSource: integer;
begin
  if (NewSizeX=Original.SizeX) and (NewSizeY=Original.SizeY) then
  begin
    Copy(Original);
  end
  else
  begin
    ReSize(NewSizeX, NewSizeY, Original.Depth);
    RatioX := NewSizeX / Original.SizeX;
    RatioY := NewSizeY / Original.SizeY;
    InvRatioX := 1 / RatioX;
    InvRatioY := 1 / RatioY;

    MaxX := SizeX - 1;
    MaxY := SizeY - 1;
    OrigMaxX := Original.SizeX - 1;
    OrigMaxY := Original.SizeY - 1;
    MoveSizeBytes := Depth * SizeOf(T);

    for CntX := 0 to MaxX do
    begin
      OrigPosX := Min(OrigMaxX, Round(CntX * InvRatioX));
      for CntY := 0 to MaxY do
      begin
        OrigPosY := Min(OrigMaxY, Round(CntY * InvRatioY));
        RawPostDest := GetRawPos(CntX, CntY);
        RawPosSource := Original.GetRawPos(OrigPosX, OrigPosY);
        Move(Original.FData[RawPosSource], FData[RawPostDest], MoveSizeBytes);
      end;
    end;
  end;
end;

procedure TVolume.CopyTransposingXD(Original: TVolume);
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  DestBase, SrcPos, SrcStride: integer;
begin
  ReSize(Original.Depth, Original.SizeY, Original.SizeX);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  SrcStride := Original.FDepth; // Original X-slot step per CntD
  if MaxY > 0 then
  begin
    for CntX := 0 to MaxX do
    begin
      for CntY := 0 to MaxY do
      begin
        DestBase := GetRawPos(CntX, CntY, 0);
        SrcPos := Original.GetRawPos(0, CntY, CntX);
        for CntD := 0 to MaxD do
        begin
          FData[DestBase + CntD] := Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
        end;
      end;
    end;
  end
  else
  begin
    for CntX := 0 to MaxX do
    begin
      DestBase := GetRawPos(CntX, 0, 0);
      SrcPos := Original.GetRawPos(0, 0, CntX);
      for CntD := 0 to MaxD do
      begin
        FData[DestBase + CntD] := Original.FData[SrcPos];
        Inc(SrcPos, SrcStride);
      end;
    end;
  end;
end;

procedure TVolume.CopyTransposingYD(Original: TVolume);
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  DestBase, SrcPos, SrcStride: integer;
begin
  ReSize(Original.SizeX, Original.Depth, Original.SizeY);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  SrcStride := Original.FSizeX * Original.FDepth; // Original Y-slot step per CntD
  if MaxX > 0 then
  begin
    for CntX := 0 to MaxX do
    begin
      for CntY := 0 to MaxY do
      begin
        DestBase := GetRawPos(CntX, CntY, 0);
        SrcPos := Original.GetRawPos(CntX, 0, CntY);
        for CntD := 0 to MaxD do
        begin
          FData[DestBase + CntD] := Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
        end;
      end;
    end;
  end
  else
  begin
    for CntY := 0 to MaxY do
    begin
      DestBase := GetRawPos(0, CntY, 0);
      SrcPos := Original.GetRawPos(0, 0, CntY);
      for CntD := 0 to MaxD do
      begin
        FData[DestBase + CntD] := Original.FData[SrcPos];
        Inc(SrcPos, SrcStride);
      end;
    end;
  end;
end;

procedure TVolume.CopyTransposingAs2D(Original: TVolume);
var
  OriginalSizeX, OriginalSizeY, OriginalDepth: integer;
begin
  OriginalSizeY := Original.SizeY;
  if OriginalSizeY <> 1 then
  begin
    OriginalSizeX := Original.SizeX;
    OriginalDepth := Original.Depth;
    Original.ReSize(OriginalSizeX*OriginalSizeY, 1, OriginalDepth);
    CopyTransposingXD(Original);
    Original.ReSize(OriginalSizeX, OriginalSizeY, OriginalDepth);
  end
  else
  begin
    CopyTransposingXD(Original);
  end;
end;

function TVolume.DotProduct(Original: TVolume): T;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at DotProduct: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Result := Self.DotProduct(Addr(Self.FData[0]), Addr(Original.FData[0]), Self.Size);
end;

function TVolume.SumDiff(Original: TVolume): T;
var
  I: integer;
  vHigh: integer;
  AuxDiff: Single;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at SumDiff: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Result := 0;
  vHigh := High(FData);
  for I := 0 to vHigh do
  begin
    AuxDiff := FData[I] - Original.FData[I];
    Result := Result + Abs(AuxDiff);
  end;
end;

procedure TVolume.DebugDiff(Original: TVolume; Limit: Single);
var
  I: integer;
  vHigh: integer;
  AuxDiff: Single;
begin
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at DebugDiff: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  vHigh := High(FData);
  for I := 0 to vHigh do
  begin
    AuxDiff := FData[I] - Original.FData[I];
    if AuxDiff > Limit then
    begin
      WriteLn('Diff at pos ', I, ':', AuxDiff,'. Self:', FData[I], ' Original:', Original.FData[I]);
    end;
  end;
end;

procedure TVolume.SumToPos(Original: TVolume);
var
  I: integer;
  vHigh: integer;
begin
  if Size <> Original.FSize then Resize(Original);
  if Length(Original.FData) > 0 then
  begin
    vHigh := High(Original.FData);
    FData[0] := Original.FData[0];
    if vHigh > 0 then
    begin
      for I := 1 to vHigh do
        FData[I] := Original.FData[I] + Original.FData[I-1];
    end;
  end;
end;

function TVolume.GetDistanceSqr(Original: TVolume): T;
var
  I: integer;
  vHigh: integer;
  AuxDiff: Single;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at GetDistanceSqr: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Result := 0;
  vHigh := High(FData);
  for I := 0 to vHigh do
  begin
    AuxDiff := FData[I] - Original.FData[I];
    Result := Result + AuxDiff * AuxDiff;
  end;
end;

function TVolume.GetDistance(Original: TVolume): T;
begin
  Result := GetDistanceSqr(Original);
  if Result > 0 then Result := Sqrt(Result) else Result := 0;
end;

function TVolume.SumAtDepth(pDepth: integer): T;
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  RawPos, RowStride: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step
  Result := 0;
  for CntX := 0 to MaxX do
  begin
    RawPos := Self.GetRawPos(CntX, 0, pDepth);
    for CntY := 0 to MaxY do
    begin
      Result := Result + FData[RawPos];
      Inc(RawPos, RowStride);
    end;
  end;
end;

function TVolume.AvgAtDepth(pDepth: integer): T;
begin
  Result := SumAtDepth(pDepth)/(SizeX*SizeY);
end;

function TVolume.GetRawPos(x, y, d: integer): integer;
begin
  Result := ((FSizeX * y) + x) * FDepth + d;
end;

function TVolume.GetRawPos(x, y: integer): integer;
begin
  Result := ((FSizeX * y) + x) * FDepth;
end;

function TVolume.GetRawPtr(x, y, d: integer): pointer;
begin
  Result := Addr(FData[GetRawPos(x, y, d)]);
end;

function TVolume.GetRawPtr(x, y: integer): pointer;
begin
  Result := Addr(FData[GetRawPos(x, y)]);
end;

function TVolume.GetRawPtr(x: integer): pointer;
begin
  Result := Addr(FData[x]);
end;

function TVolume.GetRawPtr(): pointer;
begin
  Result := Addr(FData[0]);
end;

function TVolume.GetMax(): T;
var
  I: integer;
  vHigh: integer;
begin
  if Length(FData) > 0 then
  begin
    Result := FData[0];
    FLastPos := 0;
    vHigh := High(FData);
    if vHigh > 0 then
    begin
      for I := 1 to vHigh do
      begin
        if FData[I] > Result then
        begin
          Result := FData[I];
          FLastPos := I;
        end;
      end;
    end;
  end
  else
  begin
    Result := -1;
  end;
end;

function TVolume.GetNonZero(): integer;
var
  I: integer;
  vHigh: integer;
begin
  Result := 0;
  if Length(FData) > 0 then
  begin
    vHigh := High(FData);
    for I := 0 to vHigh do
    begin
      if FData[I] <> 0 then Inc(Result);
    end;
  end;
end;

function TVolume.GetMaxAbs(): T;
var
  I: integer;
  vHigh: integer;
  auxSingle: single;
begin
  if Length(FData) > 0 then
  begin
    auxSingle := FData[0];
    FLastPos := 0;
    if auxSingle < 0 then auxSingle := -auxSingle;
    // Seed the running max with the MAGNITUDE of element 0, not its signed
    // value: a negative element 0 of largest magnitude would otherwise be
    // missed (the returned max-abs would be too small for the scale users -
    // ForceMaxAbs / NormalizeMax / int8 quantization / backprop overflow
    // protection - that all expect a true non-negative magnitude).
    Result := auxSingle;
    vHigh := High(FData);
    if vHigh > 0 then
    begin
      for I := 1 to vHigh do
      begin
        auxSingle := FData[I];
        if auxSingle < 0 then auxSingle := -auxSingle;
        if auxSingle > Result then
        begin
          Result := auxSingle;
          FLastPos := I;
        end;
      end;
    end;
  end
  else
  begin
    Result := 0;
  end;
end;

function TVolume.GetMin(): T;
var
  I: integer;
  vHigh: integer;
begin
  if Length(FData) > 0 then
  begin
    Result := FData[0];
    FLastPos := 0;
    vHigh := High(FData);
    if vHigh > 0 then
    begin
      for I := 1 to vHigh do
      begin
        if FData[I] < Result then
        begin
          Result := FData[I];
          FLastPos := I;
        end;
      end;
    end;
  end
  else
  begin
    Result := -1;
  end;
end;

// this function returns the minimum and maximum values of a channel.
procedure TVolume.GetMinMaxAtDepth(pDepth: integer; out pMin, pMax: T);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  Aux: T;
  RawPos, RowStride: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step

  pMin := Self.Data[0, 0, pDepth];
  pMax := Self.Data[0, 0, pDepth];

  for CntX := 0 to MaxX do
  begin
    RawPos := Self.GetRawPos(CntX, 0, pDepth);
    for CntY := 0 to MaxY do
    begin
      Aux := FData[RawPos];

      if Aux < pMin
      then pMin := Aux
      else if Aux > pMax then pMax := Aux;
      Inc(RawPos, RowStride);
    end;
  end;
end;

function TVolume.GetSum(): T;
var
  I: integer;
  vHigh: integer;
begin
  if Length(FData) > 0 then
  begin
    Result := FData[0];
    vHigh := High(FData);
    if vHigh > 0 then
    begin
      for I := 1 to vHigh do
      begin
        Result := Result + FData[I];
      end;
    end;
  end
  else
  begin
    Result := 0;
  end;
end;

function TVolume.GetSumAbs(): T;
var
  I: integer;
  vHigh: integer;
begin
  if Length(FData) > 0 then
  begin
    if FData[0] >0 then Result := FData[0] else Result := -FData[0];
    vHigh := High(FData);
    if vHigh > 0 then
    begin
      for I := 1 to vHigh do
      begin
        if FData[I] > 0
          then Result := Result + FData[I]
          else Result := Result - FData[I];
      end;
    end;
  end
  else
  begin
    Result := 0;
  end;
end;

function TVolume.GetSumSqr(): T;
var
  I: integer;
  vHigh: integer;
begin
  if Length(FData) > 0 then
  begin
    Result := FData[0] * FData[0];
    vHigh := High(FData);
    if vHigh > 0 then
    begin
      for I := 1 to vHigh do
      begin
        Result := Result + FData[I] * FData[I];
      end;
    end;
  end
  else
  begin
    Result := 0;
  end;
end;

function TVolume.GetAvg(): T;
var
  floatSize: Single;
begin
  if (FSize > 0) then
  begin
    floatSize := FSize;
    Result := GetSum() / floatSize;
  end
  else
  begin
    Result := 0;
  end;
end;

procedure TVolume.ZeroCenter();
var
  localAvg: Single;
begin
  localAvg := GetAvg();
  Sub(localAvg);
end;

function TVolume.GetVariance(): T;
var
  Avg: T;
  I: integer;
  vHigh: integer;
  AuxDif: Single;
  floatSize: Single;
begin
  Result := 0;
  if (FSize > 1) then
  begin
    Avg := GetAvg();
    vHigh := High(FData);

    for I := 0 to vHigh do
    begin
      AuxDif := FData[I] - Avg;
      Result := Result + Sqr(AuxDif);
    end;
    floatSize := FSize;
    Result := Result / floatSize;
  end
end;

function TVolume.GetValueCount(Value: T): integer;
var
  I, vHigh: integer;
begin
  Result := 0;
  if FSize > 0 then
  begin
    vHigh := FSize - 1;
    for I := 0 to vHigh do
    begin
      if FData[I]=Value then Inc(Result);
    end;
  end;
end;

function TVolume.GetSmallestIdxInRange(StartPos, Len: integer): integer;
var
  FinishPos: integer;
  PosCnt: integer;
  SmallestValue: T;
begin
  Result := 0;
  if StartPos < FSize then
  begin
    FinishPos := Min(FSize - 1, StartPos + Len - 1);
    if FinishPos >= StartPos then
    begin
      SmallestValue := FData[StartPos];
      Result := StartPos;
      if FinishPos > StartPos then
      begin
        for PosCnt := StartPos to FinishPos do
        begin
          if FData[PosCnt] < SmallestValue then
          begin
            SmallestValue := FData[PosCnt];
            Result := PosCnt;
          end;
        end;
      end;
    end;
  end;
end;

function TVolume.GetStdDeviation(): T;
var
  Aux: Single;
begin
  Aux := GetVariance();
  Result := Sqrt( Aux );
end;

function TVolume.GetMagnitude(): T;
var
  Aux: Single;
begin
  Aux := GetSumSqr();
  Result := Sqrt( Aux );
end;

function TVolume.GetEntropy: T;
var
  I, vHigh: integer;
  vSum: TNeuralFloat;
begin
  vSum := 0;
  if FSize > 0 then
  begin
    vHigh := FSize - 1;
    for I := 0 to vHigh do
    begin
      if FData[I] > 0 then // To avoid log(0) which is undefined
        vSum := vSum + (FData[i] * pcr_log2f(FData[i]));
    end;
  end;
  Result := -vSum;
end;

function TVolume.GetPerplexity: T;
begin
  Result := pcr_exp2f(GetEntropy());
end;

function TVolume.CrossEntropyOnPixel(Target: TVolume; X, Y: integer): T;
var
  d, MaxD: integer;
  BaseT, BaseS: integer;
  P, Tgt: T;
begin
  Result := 0;
  MaxD := FDepth - 1;
  BaseT := Target.GetRawPos(X, Y);
  BaseS := GetRawPos(X, Y);
  for d := 0 to MaxD do
  begin
    Tgt := Target.FData[BaseT + d];
    if Tgt > 0 then
    begin
      P := FData[BaseS + d];
      if P < 1e-12 then P := 1e-12;
      Result := Result - Tgt * Ln(P);
    end;
  end;
end;

function TVolume.MeanCrossEntropy(Target: TVolume): T;
var
  X, Y, MaxX, MaxY, PixelCount: integer;
begin
  Result := 0;
  PixelCount := FSizeX * FSizeY;
  if PixelCount = 0 then Exit;
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
      Result := Result + CrossEntropyOnPixel(Target, X, Y);
  Result := Result / PixelCount;
end;

procedure TVolume.FlipX();
var
  iFrom, iTo: integer;
  iRawPos1, iRawPos2: integer;
  iBase1, iBase2: integer;
  MaxY, MaxD: integer;
  CountX, CountY, CountD: integer;
  Aux: TNeuralFloat;
begin
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;

  iTo := FSizeX div 2 - 1;
  iFrom := 0;

  for CountX := iFrom to iTo do
  begin
    for CountY := 0 to MaxY do
    begin
      iBase1 := GetRawPos(CountX, CountY, 0);
      iBase2 := GetRawPos(FSizeX-CountX-1, CountY, 0);
      for CountD := 0 to MaxD do
      begin
        iRawPos1 := iBase1 + CountD;
        iRawPos2 := iBase2 + CountD;
        Aux := FData[iRawPos1];
        FData[iRawPos1] := FData[iRawPos2];
        FData[iRawPos2] := Aux;
      end;
    end;
  end;
end;

procedure TVolume.FlipY();
var
  iFrom, iTo: integer;
  iRawPos1, iRawPos2: integer;
  iBase1, iBase2: integer;
  MaxX, MaxD: integer;
  CountX, CountY, CountD: integer;
  Aux: TNeuralFloat;
begin
  MaxX := FSizeX - 1;
  MaxD := FDepth - 1;

  iTo := FSizeY div 2 - 1;
  iFrom := 0;

  for CountX := 0 to MaxX do
  begin
    for CountY := iFrom to iTo do
    begin
      iBase1 := GetRawPos(CountX, CountY, 0);
      iBase2 := GetRawPos(CountX, FSizeY-CountY-1, 0);
      for CountD := 0 to MaxD do
      begin
        iRawPos1 := iBase1 + CountD;
        iRawPos2 := iBase2 + CountD;
        Aux := FData[iRawPos1];
        FData[iRawPos1] := FData[iRawPos2];
        FData[iRawPos2] := Aux;
      end;
    end;
  end;
end;

procedure TVolume.IncTag();
begin
  Inc(FTag[0]);
end;

procedure TVolume.ClearTag();
var
  I, Hi, Lo: integer;
begin
  Hi := High(FTag);
  Lo := Low(FTag);
  for I := Lo to Hi do FTag[I] := 0;
end;

function TVolume.NeuralToStr(V: TNeuralFloat): string;
begin
  Result := FloatToStr(V, FFormatSettings);
end;

procedure TVolume.LoadNonZeroPosIntoTIntegerList(Ints: TIntegerList;
  IncludePositive: boolean=true; IncludeNegative:boolean = true);
var
  I: integer;
  vHigh: integer;
  Value: TNeuralFloat;
begin
  vHigh := High(FData);
  for I := 0 to vHigh do
  begin
    Value := FData[I];
    if IncludePositive and (value > 0) then Ints.Add(I)
    else if IncludeNegative and (value < 0) then Ints.Add(I);
  end;
end;

function TVolume.CreateIntegerListWithNonZeroPos(IncludePositive: boolean;
  IncludeNegative: boolean): TIntegerList;
begin
  Result := TIntegerList.Create();
  LoadNonZeroPosIntoTIntegerList(Result, IncludePositive, IncludeNegative);
end;


procedure TVolume.RgbImgToNeuronalInput(color_encoding: integer);
begin
  // In all color encodings, values vary from -2 to 2.
  if ( (color_encoding = csEncodeRGB) or (color_encoding = csEncodeGray) ) then
  begin
    Sub(128);
    Divi(64);
  end
  else if (color_encoding = csEncodeHSV) then
  begin
    RgbToHsv();
    Mul(4);
    Sub(2);
    //MulAtDepth(0,2);
    //MulAtDepth(1,4);
    //MulAtDepth(2,4);
    //AddAtDepth(1,-2);
    //AddAtDepth(2,-2);
  end
  else if (color_encoding = csEncodeHSL) then
  begin
    RgbToHsl();
    Mul(4);
    Sub(2);
  end
  else if (color_encoding = csEncodeLAB) then
  begin
    RgbToLab();
    MulAtDepth(0,1/25);
    AddAtDepth(0,-2);
    MulAtDepth(1,1/50);
    MulAtDepth(2,1/50);
  end
  else
  begin
    WriteLn('No color encoding has been found:', color_encoding);
  end;
end;

procedure TVolume.NeuronalInputToRgbImg(color_encoding: integer);
begin
  if ( (color_encoding = csEncodeRGB) or (color_encoding = csEncodeGray) ) then
  begin
    Mul(64);
    Add(128);
  end
  else if (color_encoding = csEncodeHSV) then
  begin
    Add(2);
    Mul(0.25);
    //AddAtDepth(1,2);
    //AddAtDepth(2,2);
    //MulAtDepth(1,1/4);
    //MulAtDepth(2,1/4);
    //MulAtDepth(0,1/2);
    HsvToRgb();
  end
  else if (color_encoding = csEncodeHSL) then
  begin
    Add(2);
    Mul(0.25);
    HslToRgb();
  end
  else if (color_encoding = csEncodeLAB) then
  begin
    MulAtDepth(1,50);
    MulAtDepth(2,50);
    AddAtDepth(0, 2);
    MulAtDepth(0,25);
    LabToRgb();
  end
  else
  begin
    WriteLn('Bad color encoding:', color_encoding);
  end;
end;

procedure TVolume.NeuronalWeightToImg(color_encoding: integer);
begin
  NeuronalWeightToImg(Self.GetMax(), Self.GetMin(), color_encoding);
end;

procedure TVolume.NeuronalWeightToImg(MaxW, MinW: TNeuralFloat; color_encoding: integer);
var
  MaxAbs: TNeuralFloat;
begin
  MaxAbs := Max(Abs(MinW), Abs(MaxW));
  if MaxAbs = 0.0 then exit;
  if ( (color_encoding = csEncodeRGB) or (color_encoding = csEncodeGray) ) then
  begin
    Mul(128/MaxAbs);
    Add(128);
  end
  else if color_encoding = csEncodeLAB then
  begin
    if FDepth = 3 then
    begin
      MulAtDepth(0,50/MaxAbs);
      AddAtDepth(0,50);
      MulAtDepth(1,100/MaxAbs);
      MulAtDepth(2,100/MaxAbs);
    end
    else if FDepth = 2 then
    begin
      // AB channels only
      Mul(100/MaxAbs);
    end
    else if FDepth = 1 then
    begin
      // L channel only
      Mul(50/MaxAbs);
      Add(50);
    end;
  end
  else
  begin
    // HSL and HSV
    Mul(0.5/MaxAbs);
    Add(0.5);
  end;
end;

procedure TVolume.NeuronalWeightToImg3Channel(MaxW0, MinW0, MaxW1, MinW1,
  MaxW2, MinW2: TNeuralFloat; color_encoding: integer);
var
  MaxAbs0, MaxAbs1, MaxAbs2:TNeuralFloat;
begin
  MaxAbs0 := Max(Abs(MinW0), Abs(MaxW0));
  MaxAbs1 := Max(Abs(MinW1), Abs(MaxW1));
  MaxAbs2 := Max(Abs(MinW2), Abs(MaxW2));

  if ( (color_encoding = csEncodeRGB) ) then
  begin
    MulAtDepth(0,128/MaxAbs0);
    if FDepth > 1 then MulAtDepth(1,128/MaxAbs1);
    if FDepth > 2 then MulAtDepth(2,128/MaxAbs2);
    Add(128);
  end
  else if color_encoding = csEncodeGray then
  begin
    Mul(128/MaxAbs0);
    Add(128);
  end
  else if color_encoding = csEncodeLAB then
  begin
    if FDepth = 3 then
    begin
      MulAtDepth(0,50/MaxAbs0);
      AddAtDepth(0,50);
      MulAtDepth(1,100/MaxAbs1);
      MulAtDepth(2,100/MaxAbs2);
    end
    else if FDepth = 2 then
    begin
      // AB channels only
      MulAtDepth(0,100/MaxAbs0);
      MulAtDepth(1,100/MaxAbs1);
    end
    else if FDepth = 1 then
    begin
      // L channel only
      Mul(50/MaxAbs0);
      Add(50);
    end;
  end
  else
  begin
    // HSL and HSV
    MulAtDepth(0,0.5/MaxAbs0);
    if FDepth > 1 then MulAtDepth(1,0.5/MaxAbs1);
    if FDepth > 2 then MulAtDepth(2,0.5/MaxAbs2);
    Add(0.5);
  end;
end;

procedure TVolume.SetClass(pClass: integer; value: T);
begin
  if (pClass >= 0) and (pClass <= High(FData)) then
  begin
    Fill(-value);
    FData[pClass] := value;
  end
  else
  begin
    //TODO: add error treatment here.
  end;
end;

procedure TVolume.SetClass(pClass: integer; TrueValue, FalseValue: T);
begin
  if (pClass >= 0) and (pClass <= High(FData)) then
  begin
    Fill(FalseValue);
    FData[pClass] := TrueValue;
  end
  else
  begin
    //TODO: add error treatment here.
  end;
end;

procedure TVolume.SetClassForHiperbolicTangent(pClass: integer);
begin
  // Bipolar result works better with hiperbolic tangent output
  Self.SetClass(pClass, 0.5, -0.5);
end;

procedure TVolume.SetClassForReLU(pClass: integer);
begin
  // Pure ReLU (without softmax) works better with all positive outputs
  Self.SetClass(pClass, 1.6, 0.2);
end;

procedure TVolume.SetClassForSoftMax(pClass: integer);
begin
  Self.SetClass(pClass, 1, 0);
end;

function TVolume.GetClass(): integer;
var
  I: integer;
  vHigh: integer;
  vMax: T;
begin
  vHigh := High(FData);
  if (vHigh>0) then
  begin
    Result := 0;
    vMax := FData[Result];
    for I := 0 to vHigh do
    begin
      if FData[I] > vMax then
      begin
        Result := I;
        vMax := FData[I];
      end;
    end;
  end else
  begin
    Result := -1;
  end;
end;

function TVolume.GetClassOnPixel(X, Y: integer): integer;
var
  I: integer;
  vHigh: integer;
  vMax: T;
  Pos: integer;
  Value: T;
begin
  vHigh := Depth - 1;
  if (vHigh>=0) then
  begin
    Result := 0;
    Pos := GetRawPos(X, Y);
    vMax := FData[Pos];
    for I := 1 to vHigh do
    begin
      Inc(Pos);
      Value := FData[Pos];
      if Value > vMax then
      begin
        Result := I;
        vMax := Value;
      end;
    end;
  end else
  begin
    Result := -1;
  end;
end;

function TVolume.SoftMax(): T;
var
  I: integer;
  vHigh: integer;
  LocalValue: T;
  TotalSum: TNeuralFloat;
  MinValue, MaxValue: T;
begin
  MaxValue := GetMax();
  if MaxValue <> 0 then Sub(MaxValue);
  MinValue := GetMin();

  TotalSum := 0;

  // forces range [-1000,0]
  if MinValue <> 0 then
  begin
    if MinValue < -1000 then Mul( -1000/MinValue );
    vHigh := High(FData);

    {$IFDEF AVXANY}
    // FData has already been shifted by Sub(MaxValue), so every element is <= 0
    // (and floored to [-1000,0] above), well inside AVXExp's safe range. Clamp
    // in place for bit-parity with the scalar NeuronForceRange path, then
    // exponentiate the whole flat buffer 8-wide and sum it with the AVX
    // reduction (parity with the scalar Exp/accumulate loop within ~1e-6).
    for I := 0 to vHigh do
      FData[I] := NeuronForceRange(FData[I], 4000);
    AVXExp(TNeuralFloatArrPtr(@FData[0]),
           TNeuralFloatArrPtr(@FData[0]), vHigh + 1);
    TotalSum := AVXGetSum(TNeuralFloatArrPtr(@FData[0]), vHigh + 1);
    {$ELSE}
    for I := 0 to vHigh do
    begin
      // FData has already been shifted by Sub(MaxValue) above, so do not
      // subtract MaxValue again here (that would underflow Exp to zero).
      LocalValue := NeuralExp( NeuronForceRange(FData[I], 4000) );
      // LocalValue := pcr_expf( FData[I] );
      FData[I] := LocalValue;
      TotalSum := TotalSum + FData[I];
    end;
    {$ENDIF}

    if TotalSum > 0 then
    begin
      Divi(TotalSum);
    end;
  end;

  Result := TotalSum;
end;

procedure TVolume.PointwiseSoftMax(NoForward: boolean = false);
var
  I, StartPointPos: integer;
  MaxX, MaxY, MaxD, FDepthM1, MaxDP1: integer;
  CountX, CountY, CountD: integer;
  MaxValue: T;
  LocalValue: T;
  TotalSum: TNeuralFloat;
begin
  // TODO: This portion of code can be optimized
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  FDepthM1 := FDepth - 1;

  if MaxD > 0 then
  begin
    {$IFDEF AVXANY}
    if not NoForward then
    begin
      // Whole-volume fast path (NoForward=false): every (x,y) position owns a
      // full, contiguous FDepth-length span, so the entire FData buffer is a
      // back-to-back sequence of equal-length depth spans. Because exp() is a
      // pure element-wise map, the numerically-stabilized exponentiation can be
      // dispatched to AVXExp ONCE over the whole volume instead of re-dispatching
      // per position. Only the max-find / clamp (pass 1) and the sum+normalize
      // (pass 2) reductions reset at each Depth boundary, preserving the exact
      // per-position max-subtraction stabilization. This avoids FSizeX*FSizeY
      // AVXExp/AVXGetSum dispatches per call.
      // Pass 1: per-span max, then in-place clamp of (x - max) over the span.
      for CountX := 0 to MaxX do
      begin
        for CountY := 0 to MaxY do
        begin
          StartPointPos := GetRawPos(CountX, CountY);
          I := StartPointPos;
          MaxValue := FData[I];
          for CountD := 1 to MaxD do
          begin
            Inc(I);
            if FData[I] > MaxValue then MaxValue := FData[I];
          end;
          I := StartPointPos;
          for CountD := 0 to MaxD do
          begin
            FData[I] := NeuronForceRange(FData[I] - MaxValue, 4000);
            Inc(I);
          end;
        end;
      end;
      // Single whole-volume exponentiation (parity with the scalar NeuralExp loop
      // within ~1e-6 relative error). Spatial spans are contiguous so the entire
      // FSizeX*FSizeY*FDepth range is one AVXExp call.
      AVXExp(TNeuralFloatArrPtr(@FData[0]),
             TNeuralFloatArrPtr(@FData[0]), FSizeX * FSizeY * FDepth);
      // Pass 2: per-span sum + normalize.
      for CountX := 0 to MaxX do
      begin
        for CountY := 0 to MaxY do
        begin
          StartPointPos := GetRawPos(CountX, CountY);
          TotalSum := AVXGetSum(TNeuralFloatArrPtr(@FData[StartPointPos]), MaxD + 1);
          if TotalSum > 0 then
            TNNetVolume.Mul(Addr(FData[StartPointPos]), 1.0 / TotalSum, MaxD + 1);
        end;
      end;
      Exit;
    end;
    {$ENDIF}
    for CountX := 0 to MaxX do
    begin
      if NoForward then MaxD := Min(FDepth - 1, CountX);
      for CountY := 0 to MaxY do
      begin
        StartPointPos := GetRawPos(CountX, CountY);
        if NoForward and (MaxD < FDepth - 1) then
        begin
          MaxDP1 := MaxD + 1;
          I := StartPointPos + MaxDP1;
          for CountD := MaxDP1 to FDepthM1 do
          begin
            FData[I] := 0;
            Inc(I);
          end;
        end;
        I := StartPointPos;
        // Find the point max value.
        MaxValue := FData[I];
        for CountD := 1 to MaxD do
        begin
          Inc(I);
          if FData[I] > MaxValue
            then MaxValue := FData[I];
        end;
        TotalSum := 0;
        I := StartPointPos;
        {$IFDEF AVXANY}
        // Contiguous-depth fast path: clamp (x - max) in place over the depth
        // segment, then exponentiate it 8-wide via AVXExp (parity with the scalar
        // pcr_expf loop within ~1e-6 relative error), then accumulate the sum.
        for CountD := 0 to MaxD do
        begin
          FData[I] := NeuronForceRange(FData[I] - MaxValue, 4000);
          Inc(I);
        end;
        AVXExp(TNeuralFloatArrPtr(@FData[StartPointPos]),
               TNeuralFloatArrPtr(@FData[StartPointPos]), MaxD + 1);
        TotalSum := AVXGetSum(TNeuralFloatArrPtr(@FData[StartPointPos]), MaxD + 1);
        {$ELSE}
        for CountD := 0 to MaxD do
        begin
          // LocalValue := pcr_expf( NeuronForceRange(FData[I] - MaxValue, 4000) );
          LocalValue := NeuralExp( NeuronForceRange(FData[I] - MaxValue, 4000) );
          FData[I] := LocalValue;
          TotalSum := TotalSum + LocalValue;
          Inc(I);
        end;
        {$ENDIF}
        if TotalSum > 0 then
          TNNetVolume.Mul(Addr(FData[StartPointPos]), 1.0 / TotalSum, MaxD + 1);
      end;
    end;
  end;
end;

procedure TNNetVolume.PointwiseNorm(pNorms: TNNetVolume = nil);
var
  StartPointPtr: pointer;
  MaxX, MaxY: integer;
  CountX, CountY: integer;
  Modulus, Multiplier: TNeuralFloat;
begin
  if Assigned(pNorms) then
  begin
    pNorms.ReSize(SizeX, SizeY, 1);
    pNorms.Fill(1);
  end;
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  for CountX := 0 to MaxX do
  begin
    for CountY := 0 to MaxY do
    begin
      StartPointPtr := GetRawPtr(CountX, CountY);
      Modulus := Sqrt(DotProduct(StartPointPtr, StartPointPtr, FDepth));
      if Modulus > 0 then
      begin
        Multiplier := 1/Modulus;
        if Assigned(pNorms) then pNorms[CountX, CountY, 0] := Multiplier;
        Mul(StartPointPtr, Multiplier, FDepth);
      end;
    end;
  end;
end;

procedure TNNetVolume.PointwiseMul(pNorms: TNNetVolume);
var
  StartPointPtr: pointer;
  MaxX, MaxY: integer;
  CountX, CountY: integer;
  Modulus: TNeuralFloat;
begin
  if Assigned(pNorms) then pNorms.ReSize(SizeX, SizeY, 1);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  for CountX := 0 to MaxX do
  begin
    for CountY := 0 to MaxY do
    begin
      StartPointPtr := GetRawPtr(CountX, CountY);
      Modulus := pNorms[CountX, CountY, 0];
      if Modulus <> 1 then
      begin
        Mul(StartPointPtr, Modulus, FDepth);
      end;
    end;
  end;
end;

class procedure TNNetVolume.VectorExp(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
{$IFDEF AVXANY}
begin
  if N <= 0 then exit;
  AVXExp(pDst, pSrc, N);
end;
{$ELSE}
var
  I, NM1: integer;
begin
  NM1 := N - 1;
  for I := 0 to NM1 do
    pDst^[I] := NeuralExp(pSrc^[I]);
end;
{$ENDIF}

class procedure TNNetVolume.VectorSigmoid(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
var
  I, NM1: integer;
  S: TNeuralFloat;
begin
  if N <= 0 then exit;
  NM1 := N - 1;
  // sigmoid(x) = 1/(1+exp(-x)). Compute exp(-x) into the destination buffer in a
  // single vectorized pass, then finish elementwise. The scalar form below mirrors
  // the reference Sigmoid() (avoids overflow for very negative x).
  for I := 0 to NM1 do
    pDst^[I] := -pSrc^[I];
  VectorExp(pDst, pDst, N);
  for I := 0 to NM1 do
  begin
    S := pDst^[I]; // S = exp(-x)
    pDst^[I] := 1 / (1 + S);
  end;
end;

class procedure TNNetVolume.VectorTanh(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
var
  I, NM1: integer;
  X, E: TNeuralFloat;
begin
  if N <= 0 then exit;
  NM1 := N - 1;
  // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x)). Compute E = exp(-2x) in a single
  // vectorized pass, clamping -2x into [-88, 88] so exp neither overflows nor
  // underflows (tanh saturates to +/-1 there, matching the scalar pcr_tanhf).
  // No sign read in the finishing pass, so buffers may alias (dst = src).
  for I := 0 to NM1 do
  begin
    X := -2 * pSrc^[I];
    if X > 88 then X := 88
    else if X < -88 then X := -88;
    pDst^[I] := X;
  end;
  VectorExp(pDst, pDst, N);
  for I := 0 to NM1 do
  begin
    E := pDst^[I]; // E = exp(-2x) in [exp(-88), exp(88)]
    pDst^[I] := (1 - E) / (1 + E);
  end;
end;

class procedure TNNetVolume.VectorErf(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
const
  // Abramowitz & Stegun 7.1.26 coefficients (|err| < 1.5e-7).
  cErfA1: TNeuralFloat =  0.254829592;
  cErfA2: TNeuralFloat = -0.284496736;
  cErfA3: TNeuralFloat =  1.421413741;
  cErfA4: TNeuralFloat = -1.453152027;
  cErfA5: TNeuralFloat =  1.061405429;
  cErfP:  TNeuralFloat =  0.3275911;
var
  I, NM1: integer;
  X, AX, T, Poly, E: TNeuralFloat;
  ExpBuf: array of TNeuralFloat;
begin
  if N <= 0 then exit;
  NM1 := N - 1;
  // erf(x) = sign(x) * (1 - poly(t)*exp(-x^2)), t = 1/(1+p*|x|).
  // exp(-x^2) is produced by a single vectorized VectorExp pass into a scratch
  // buffer (NOT pDst) so that pSrc -- which still holds x for the |x| and sign
  // terms in the finishing pass -- is never clobbered. Hence dst may alias src.
  SetLength(ExpBuf, N);
  for I := 0 to NM1 do
  begin
    X := pSrc^[I];
    ExpBuf[I] := -X * X;
  end;
  VectorExp(TNeuralFloatArrPtr(@ExpBuf[0]), TNeuralFloatArrPtr(@ExpBuf[0]), N);
  for I := 0 to NM1 do
  begin
    X := pSrc^[I];
    if X < 0 then AX := -X else AX := X;
    T := 1 / (1 + cErfP * AX);
    // Horner: (((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
    Poly := ((((cErfA5 * T + cErfA4) * T + cErfA3) * T + cErfA2) * T + cErfA1) * T;
    E := ExpBuf[I];
    if X < 0 then
      pDst^[I] := -(1 - Poly * E)
    else
      pDst^[I] := 1 - Poly * E;
  end;
end;

class procedure TNNetVolume.VectorSinh(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
var
  I, NM1: integer;
  X, EPos, ENeg: TNeuralFloat;
  PosBuf, NegBuf: array of TNeuralFloat;
begin
  if N <= 0 then exit;
  NM1 := N - 1;
  // sinh(x) = (exp(x) - exp(-x)) / 2. Both exponentials are produced by single
  // vectorized VectorExp passes into local scratch buffers (NOT pDst), so pSrc --
  // read for nothing past the fill below -- is never clobbered and dst may alias
  // src. The arg is clamped into [-88, 88] so exp neither overflows nor underflows
  // (sinh would overflow to +/-Inf there anyway, matching the scalar pcr_sinhf).
  SetLength(PosBuf, N);
  SetLength(NegBuf, N);
  for I := 0 to NM1 do
  begin
    X := pSrc^[I];
    if X > 88 then X := 88
    else if X < -88 then X := -88;
    PosBuf[I] := X;
    NegBuf[I] := -X;
  end;
  VectorExp(TNeuralFloatArrPtr(@PosBuf[0]), TNeuralFloatArrPtr(@PosBuf[0]), N);
  VectorExp(TNeuralFloatArrPtr(@NegBuf[0]), TNeuralFloatArrPtr(@NegBuf[0]), N);
  for I := 0 to NM1 do
  begin
    EPos := PosBuf[I]; // exp(x)
    ENeg := NegBuf[I]; // exp(-x)
    pDst^[I] := (EPos - ENeg) * 0.5;
  end;
end;

class procedure TNNetVolume.VectorLn(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
{$IFDEF AVXANY}
begin
  if N <= 0 then exit;
  AVXLn(pDst, pSrc, N);
end;
{$ELSE}
var
  I, NM1: integer;
begin
  NM1 := N - 1;
  for I := 0 to NM1 do
    pDst^[I] := pcr_logf(pSrc^[I]);
end;
{$ENDIF}

class procedure TNNetVolume.VectorSin(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
{$IFDEF AVXANY}
begin
  if N <= 0 then exit;
  AVXSinCos(pDst, pSrc, N, False);
end;
{$ELSE}
var
  I, NM1: integer;
begin
  NM1 := N - 1;
  for I := 0 to NM1 do
    pDst^[I] := pcr_sinf(pSrc^[I]);
end;
{$ENDIF}

class procedure TNNetVolume.VectorCos(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
{$IFDEF AVXANY}
begin
  if N <= 0 then exit;
  AVXSinCos(pDst, pSrc, N, True);
end;
{$ELSE}
var
  I, NM1: integer;
begin
  NM1 := N - 1;
  for I := 0 to NM1 do
    pDst^[I] := pcr_cosf(pSrc^[I]);
end;
{$ENDIF}

class procedure TNNetVolume.VectorArcSinh(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
var
  I, NM1: integer;
  X: TNeuralFloat;
  ArgBuf: array of TNeuralFloat;
begin
  if N <= 0 then exit;
  NM1 := N - 1;
  // arcsinh(x) = ln(x + sqrt(x^2 + 1)). The argument x + sqrt(x^2+1) is always >= 1
  // and is built into a scratch buffer (NOT pDst) so pSrc is never clobbered before
  // it is read -- hence dst may alias src. VectorLn then supplies the AVX2 ln pass.
  SetLength(ArgBuf, N);
  for I := 0 to NM1 do
  begin
    X := pSrc^[I];
    ArgBuf[I] := X + Sqrt(X * X + 1.0);
  end;
  VectorLn(TNeuralFloatArrPtr(@ArgBuf[0]), TNeuralFloatArrPtr(@ArgBuf[0]), N);
  for I := 0 to NM1 do
    pDst^[I] := ArgBuf[I];
end;

procedure TVolume.GroupedPointwiseSoftMax(Groups: integer);
var
  I, StartPointPos: integer;
  MaxX, MaxY: integer;
  CountX, CountY, CountD: integer;
  MaxValue: T;
  LocalValue: T;
  TotalSum: TNeuralFloat;
  GroupCnt, StartD, PointBase: integer;
  ChannelsPerGroup, ChannelsPerGroupM1, GroupsM1: integer;
begin
  // TODO: This portion of code can be optimized
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  ChannelsPerGroup := FDepth div Groups;
  ChannelsPerGroupM1 := ChannelsPerGroup - 1;
  GroupsM1 := Groups - 1;
  if ChannelsPerGroup > 1 then
  begin
    for CountX := 0 to MaxX do
    begin
      for CountY := 0 to MaxY do
      begin
        PointBase := GetRawPos(CountX, CountY, 0);
        for GroupCnt := 0 to GroupsM1 do
        begin
          StartD := ChannelsPerGroup * GroupCnt;
          //EndD := StartD + ChannelsPerGroup - 1;
          StartPointPos := PointBase + StartD;
          I := StartPointPos;
          // Find the point max value.
          MaxValue := FData[I];
          for CountD := 1 to ChannelsPerGroupM1 do
          begin
            Inc(I);
            if FData[I] > MaxValue
              then MaxValue := FData[I];
          end;
          TotalSum := 0;
          I := StartPointPos;
          for CountD := 0 to ChannelsPerGroupM1 do
          begin
            //LocalValue := pcr_expf( NeuronForceRange(FData[I] - MaxValue, 4000) );
            LocalValue := Exp( NeuronForceRange(FData[I] - MaxValue, 4000) );
            FData[I] := LocalValue;
            TotalSum := TotalSum + LocalValue;
            Inc(I);
          end;
          if TotalSum > 0 then
            TNNetVolume.Mul(Addr(FData[StartPointPos]), 1.0 / TotalSum, ChannelsPerGroupM1 + 1);
        end;
      end;
    end;
  end;
end;

procedure TVolume.OneHotEncodingOnPixel(X, Y, Token: integer);
var
  Base: integer;
begin
  if (Token < 0) or (Token >= FDepth) then
  begin
    WriteLn('Token '+IntToStr(Token)+' is out of range [0,'+IntToStr(FDepth)+
      ') at OneHotEncodingOnPixel.');
    Exit;
  end;
  Base := GetRawPos(X, Y);
  FillChar(FData[Base], FDepth * csNeuralFloatSize, 0);
  FData[Base + Token] := 1;
end;

procedure TVolume.OneHotEncoding(aTokens: array of integer);
var
  CntToken, MaxToken, Token, SizeXM1, MaxTokenP1: integer;
begin
  MaxToken := Length(aTokens) - 1;
  SizeXM1 := SizeX - 1;
  Self.Fill(0);
  if MaxToken < SizeX then
  begin
    for CntToken := 0 to MaxToken do
    begin
      Token := aTokens[CntToken];
      if Token < FDepth then
      begin
        Self[CntToken, 0, Token] := 1;
      end
      else
      begin
        WriteLn('Token '+IntToStr(Token)+' is bigger than Depth '+IntToStr(FDepth)+' at OneHotEncoding.');
      end;
    end;
    if MaxToken < SizeX - 1 then
    begin
      MaxTokenP1 := MaxToken + 1;
      for CntToken := MaxTokenP1 to SizeXM1 do
      begin
        Self[CntToken, 0, 0] := 1;
      end;
    end;
  end
  else
  begin
    WriteLn('Token length '+IntToStr(MaxToken + 1)+' is bigger than Size X '+IntToStr(SizeX)+' at OneHotEncoding.');
  end;
end;

procedure TVolume.GroupedOneHotEncoding(aTokens: array of integer;
  Groups: integer);
var
  CntToken, MaxToken, Token: integer;
  GroupSize, GroupCnt, MaxGroup, TokenPos, TokenMod, TokenDiv: integer;
begin
  MaxToken := Length(aTokens) - 1;
  GroupSize := FDepth div Groups;
  MaxGroup := Groups - 1;
  Self.Fill(0);
  if MaxToken <= SizeX then
  begin
    for CntToken := 0 to MaxToken do
    begin
      Token := aTokens[CntToken];
      for GroupCnt := 0 to MaxGroup do
      begin
        TokenDiv := Token div GroupSize;
        TokenMod := Token mod GroupSize;
        TokenPos := GroupCnt*GroupSize + TokenMod;
        if TokenPos < FDepth then
        begin
          Self[CntToken, 0, TokenPos] := 1;
        end
        else
        begin
          WriteLn('GroupedOneHotEncoding - ' +
            IntToStr(TokenPos)+' is bigger than depth ' + IntToStr(FDepth) +
            '.');
        end;
        Token := TokenDiv;
      end;
    end;
  end
  else
  begin
    WriteLn('Token length '+IntToStr(MaxToken + 1)+' is bigger than Size X '+IntToStr(SizeX)+' at GroupedOneHotEncoding.');
  end;
end;

procedure TVolume.ReverseGroupedOneHotEncoding(out aTokens: TNeuralIntegerArray; Groups: integer);
var
  CntToken, MaxToken, Token: integer;
  GroupSize, MaxGroupSize, GroupCnt, MaxGroup, TokenMod: integer;
  GroupSizePower: integer;
  InitTokenPos: integer;
  RawTokenPos: integer;
  TokenBase: integer;
  MaxValue: TNeuralFloat;
  MaxTokenMod: integer;
begin
  // Calculate maximum token index
  MaxToken := FSizeX - 1;
  // Calculate size of each group
  GroupSize := FDepth div Groups;
  MaxGroupSize := GroupSize - 1;
  // Calculate maximum group index
  MaxGroup := Groups - 1;
  // Initialize the tokens array with zeros
  SetLength(aTokens, FSizeX);
  for CntToken := 0 to MaxToken do
    aTokens[CntToken] := 0;
  // Iterate through the volume data to reconstruct tokens
  for CntToken := 0 to MaxToken do
  begin
    Token := 0;
    GroupSizePower := 1;
    TokenBase := GetRawPos(CntToken, 0, 0);
    for GroupCnt := 0 to MaxGroup do
    begin
      InitTokenPos := GroupCnt * GroupSize;
      RawTokenPos := TokenBase + InitTokenPos;
      MaxValue := FData[RawTokenPos];
      MaxTokenMod := 0;
      // Calculate the position within the group
      for TokenMod := 1 to MaxGroupSize do
      begin
        if FData[RawTokenPos + TokenMod] > MaxValue then
        begin
          MaxValue := FData[RawTokenPos + TokenMod];
          MaxTokenMod := TokenMod;
        end;
      end;
      // Reconstruct the token by reversing the modulus and division
      Token := Token + MaxTokenMod * GroupSizePower;
      GroupSizePower := GroupSizePower * GroupSize;
    end;
    // Store the reconstructed token
    aTokens[CntToken] := Token;
  end;
end;

function TVolume.ReverseGroupedOneHotEncodingOnPixel(Groups, X, Y: integer): integer;
var
  Token: integer;
  //MaxToken: integer;
  GroupSize, MaxGroupSize, GroupCnt, MaxGroup, TokenMod: integer;
  GroupSizePower: integer;
  InitTokenPos: integer;
  RawTokenPos: integer;
  MaxValue: TNeuralFloat;
  MaxTokenMod: integer;
begin
  // Calculate maximum token index
  //MaxToken := FSizeX - 1;
  // Calculate size of each group
  GroupSize := FDepth div Groups;
  MaxGroupSize := GroupSize - 1;
  // Calculate maximum group index
  MaxGroup := Groups - 1;
  begin
    Token := 0;
    GroupSizePower := 1;
    for GroupCnt := 0 to MaxGroup do
    begin
      InitTokenPos := GroupCnt * GroupSize;
      RawTokenPos := GetRawPos(X, Y, InitTokenPos);
      MaxValue := FData[RawTokenPos];
      MaxTokenMod := 0;
      // Calculate the position within the group
      for TokenMod := 1 to MaxGroupSize do
      begin
        if FData[RawTokenPos + TokenMod] > MaxValue then
        begin
          MaxValue := FData[RawTokenPos + TokenMod];
          MaxTokenMod := TokenMod;
        end;
      end;
      // Reconstruct the token by reversing the modulus and division
      Token := Token + MaxTokenMod * GroupSizePower;
      GroupSizePower := GroupSizePower * GroupSize;
    end;
  end;
  Result := Token;
end;

procedure TVolume.OneHotEncoding(aTokens: string);
var
  CntToken, MaxToken, Token: integer;
begin
  MaxToken := Length(aTokens);
  Self.Fill(0);
  if MaxToken <= SizeX then
  begin
    for CntToken := 1 to MaxToken do
    begin
      Token := Ord(aTokens[CntToken]);
      if Token < FDepth then
      begin
        Self[CntToken-1, 0, Token] := 1;
      end
    end;
  end
  else
  begin
    WriteLn('Token length '+IntToStr(MaxToken + 1)+' is bigger than Size X '+IntToStr(SizeX)+' at OneHotEncoding.');
  end;
end;

procedure TVolume.OneHotEncodingAtEnd(aTokens: string);
var
  CntToken, MaxToken, Token, Offset: integer;
begin
  MaxToken := Length(aTokens);
  Offset := SizeX - MaxToken;
  Self.Fill(0);
  if MaxToken <= SizeX then
  begin
    for CntToken := 1 to MaxToken do
    begin
      Token := Ord(aTokens[CntToken]);
      if Token < FDepth then
      begin
        Self[Offset+CntToken-1, 0, Token] := 1;
      end
    end;
  end
  else
  begin
    WriteLn('Token length '+IntToStr(MaxToken + 1)+' is bigger than Size X '+IntToStr(SizeX)+' at OneHotEncodingAtEnd.');
  end;
end;

function GetLastChars(const InputStr: string; LenStr: Integer): string;
begin
  if Length(InputStr) > LenStr then
    Result := Copy(InputStr, Length(InputStr) - LenStr + 1, LenStr)
  else
    Result := InputStr;
end;

procedure TVolume.OneHotEncodingReversed(aTokens: string);
var
  CntToken, MaxToken, Token: integer;
  LocalTokens: string;
begin
  MaxToken := Length(aTokens);
  if MaxToken > SizeX then
  begin
    LocalTokens := GetLastChars(aTokens, SizeX);
    MaxToken := Length(aTokens);
  end
  else
  begin
    LocalTokens := aTokens;
  end;
  Self.Fill(0);
  if MaxToken > 0 then
  begin
    {$IFDEF DEBUG}
    if Ord(LocalTokens[MaxToken]) < 2 then
    begin
      WriteLn('A string for prediction should not end with terminal symbol.');
    end;
    if Ord(LocalTokens[1]) < 2 then
    begin
      WriteLn('A string for prediction should not start with terminal symbol.');
    end;
    {$ENDIF}
    if MaxToken <= SizeX then
    begin
      for CntToken := 1 to MaxToken do
      begin
        Token := Ord(LocalTokens[CntToken]);
        if Token < FDepth then
        begin
          Self[MaxToken-CntToken, 0, Token] := 1;
        end;
      end;
    end
    else
    begin
      WriteLn('This should never happend. Token length '+IntToStr(MaxToken)+' is bigger than Size X '+IntToStr(SizeX)+' at OneHotEncodingReversed.');
    end;
  end
  else
  begin
    {$IFDEF DEBUG}
    WriteLn('Zero len at OneHotEncodingReversed');
    {$ENDIF}
  end;
end;

procedure TVolume.OneHotEncodingReversed(var aTokens: array of integer);
var
  CntToken, MaxToken, Token: integer;
begin
  MaxToken := Length(aTokens) - 1;
  Self.Fill(0);
  if MaxToken < SizeX then
  begin
    for CntToken := 0 to MaxToken do
    begin
      Token := aTokens[CntToken];
      if Token < FDepth then
      begin
        Self[MaxToken-CntToken, 0, Token] := 1;
      end
      else
      begin
        WriteLn('Token '+IntToStr(Token)+' is bigger than Depth '+IntToStr(FDepth)+' at OneHotEncodingReversed.');
      end;
    end;
  end
  else
  begin
    WriteLn('Token length '+IntToStr(MaxToken + 1)+' is bigger than Size X '+IntToStr(SizeX)+' at OneHotEncodingReversed.');
  end;
end;

procedure TVolume.PositionalEncoding(n: integer; PositionOffset: integer);
var
  Position: Integer;
  divTerm: Double;
  MaxX, MaxY, MaxDepth: integer;
  CntX, CntY, CntDepth: integer;
  EmbeddingSize: integer;
  RawPos, RowStride: integer;
  IsEvenDepth: boolean;
begin
  EmbeddingSize := FDepth;
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxDepth := FDepth - 1;
  RowStride := FSizeX * FDepth; // per-CntY step
  for CntDepth := 0 to MaxDepth do
  begin
    divTerm := pcr_powf(n, (2 * (CntDepth div 2)) / EmbeddingSize);
    IsEvenDepth := (CntDepth mod 2 = 0);
    for CntX := 0 to MaxX do
    begin
      RawPos := GetRawPos(CntX, 0, CntDepth);
      for CntY := 0 to MaxY do
      begin
        Position := CntY*FSizeX + CntX + PositionOffset;
        if IsEvenDepth
          then FData[RawPos] := pcr_sinf(Position / divTerm)
          else FData[RawPos] := pcr_cosf(Position / divTerm);
        Inc(RawPos, RowStride);
      end;
    end;
  end;
end;

procedure TVolume.RgbToHsv();
var
  I, J: integer;
  MaxX, MaxY: integer;
  h, s, v: TNeuralFloat;
  base: integer;
begin
  h := 0;
  s := 0;
  v := 0;

  // this function can only be used if the first 3 layers contain RGB
  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;

    for I := 0 to MaxX do
    begin
      for J := 0 to MaxY do
      begin
        base := Self.GetRawPos(I, J, 0);
        rgb2hsv(FData[base], FData[base+1], FData[base+2], h, s, v);
        FData[base] := h;
        FData[base+1] := s;
        FData[base+2] := v;
      end;
    end;
  end;
end;

procedure TVolume.HsvToRgb();
var
  I, J: integer;
  MaxX, MaxY: integer;
  r, g, b: TNeuralFloat;
  base: integer;
begin
  r := 0;
  g := 0;
  b := 0;

  // this function can only be used if the first 3 layers contain RGB
  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;

    for I := 0 to MaxX do
    begin
      for J := 0 to MaxY do
      begin
        base := Self.GetRawPos(I, J, 0);
        hsv2rgb(FData[base], FData[base+1], FData[base+2], r, g, b);
        FData[base] := r;
        FData[base+1] := g;
        FData[base+2] := b;
      end;
    end;
  end;
end;

procedure TVolume.RgbToHsl();
var
  I, J: integer;
  MaxX, MaxY: integer;
  h, s, l: TNeuralFloat;
  base: integer;
begin
  h := 0;
  s := 0;
  l := 0;

  // this function can only be used if the first 3 layers contain RGB
  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;

    for I := 0 to MaxX do
    begin
      for J := 0 to MaxY do
      begin
        base := Self.GetRawPos(I, J, 0);
        rgb2hsl(FData[base], FData[base+1], FData[base+2], h, s, l);
        FData[base] := h;
        FData[base+1] := s;
        FData[base+2] := l;
      end;
    end;
  end;
end;

procedure TVolume.HslToRgb();
var
  I, J: integer;
  MaxX, MaxY: integer;
  r, g, b: TNeuralFloat;
  base: integer;
begin
  r := 0;
  g := 0;
  b := 0;

  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;

    for I := 0 to MaxX do
    begin
      for J := 0 to MaxY do
      begin
        base := Self.GetRawPos(I, J, 0);
        hsl2rgb(FData[base], FData[base+1], FData[base+2], r, g, b);
        FData[base] := r;
        FData[base+1] := g;
        FData[base+2] := b;
      end;
    end;
  end;
end;

procedure TVolume.RgbToLab();
var
  I, J: integer;
  MaxX, MaxY: integer;
  l, a, b: TNeuralFloat;
  base: integer;
begin
  l := 0;
  a := 0;
  b := 0;

  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;

    for I := 0 to MaxX do
    begin
      for J := 0 to MaxY do
      begin
        base := Self.GetRawPos(I, J, 0);
        rgb2lab(FData[base], FData[base+1], FData[base+2], l, a, b);
        FData[base] := l;
        FData[base+1] := a;
        FData[base+2] := b;

      end;
    end;
  end;
end;

procedure TVolume.LabToRgb();
var
  I, J: integer;
  MaxX, MaxY: integer;
  r, g, b: TNeuralFloat;
  base: integer;
begin
  r := 0;
  g := 0;
  b := 0;

  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;

    for I := 0 to MaxX do
    begin
      for J := 0 to MaxY do
      begin
        base := Self.GetRawPos(I, J, 0);
        lab2rgb(FData[base], FData[base+1], FData[base+2], r, g, b);
        FData[base] := r;
        FData[base+1] := g;
        FData[base+2] := b;
      end;
    end;
  end;
end;

procedure TVolume.RgbToGray();
var
  I, J: integer;
  MaxX, MaxY: integer;
  aux: TNeuralFloat;
  base: integer;
begin
  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;

    for I := 0 to MaxX do
    begin
      for J := 0 to MaxY do
      begin
        base := Self.GetRawPos(I, J, 0);
        aux := (FData[base] + FData[base+1] + FData[base+2]) / 3;
        FData[base] := aux;
        FData[base+1] := aux;
        FData[base+2] := aux;
      end;
    end;
  end;
end;

procedure TVolume.GetGrayFromRgb(Rgb: TVolume);
var
  I, J: integer;
  MaxX, MaxY: integer;
  rgbBase: integer;
begin
  ReSize(Rgb.SizeX, Rgb.SizeY, 1);
  if Rgb.Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;

    for I := 0 to MaxX do
    begin
      for J := 0 to MaxY do
      begin
        rgbBase := Rgb.GetRawPos(I, J, 0);
        Self.Data[I, J, 0] :=
          (Rgb.FData[rgbBase] + Rgb.FData[rgbBase+1] + Rgb.FData[rgbBase+2]) / 3;
      end;
    end;
  end;
end;

procedure TVolume.MakeGray(color_encoding: integer);
begin
  if color_encoding = csEncodeRGB then
  begin
    RgbToGray();
  end
  else if ( (color_encoding = csEncodeHSL) or (color_encoding = csEncodeHSV) ) then
  begin
    FillAtDepth(0, 0);
    FillAtDepth(1, 0);
  end
  else if color_encoding = csEncodeLAB then
  begin
    FillAtDepth(1, 0);
    FillAtDepth(2, 0);
  end;
end;

procedure TVolume.ShiftRight(Positions: integer = 1);
var
  VMax, VMin: longint;
begin
  if ( (FSize > 0) and (Positions > 0) ) then
  begin
    if FSize > 1 then
    begin
      VMax := High(FData);
      VMin := Low(FData) + Positions;
      if ( (VMin <= VMax) and (VMin > 0) ) then
      begin
        // memmove semantics handle the overlapping copy toward higher indices.
        Move(FData[0], FData[Positions], (FSize - Positions) * csNeuralFloatSize);
        FillChar(FData[0], Positions * csNeuralFloatSize, 0);
      end;
    end;
  end;
end;

procedure TVolume.ShiftLeft();
begin
  if FSize > 0 then
  begin
    if FSize > 1 then
      // memmove semantics handle the overlapping copy toward lower indices.
      Move(FData[1], FData[0], (FSize - 1) * csNeuralFloatSize);
    FData[High(FData)] := 0;
  end;
end;

procedure TVolume.Print();
var
  I: integer;
  vHigh: integer;
  AuxData: Single;
begin
  vHigh := High(FData);
  Write('(',SizeX,',',SizeY,',',Depth,') - ');
  for I := 0 to vHigh do
  begin
    AuxData := FData[I];
    Write(FloatToStr(AuxData), ' ');
  end;
  WriteLn;
end;

procedure TVolume.PrintXD(Digits: integer; Decimals: integer);
var
  CX, CD, DepthM1, SizeXM1: integer;
  AUX: TNeuralFloat;
begin
  DepthM1 := Depth - 1;
  SizeXM1 := SizeX - 1;
  for CD := 0 to DepthM1 do
  begin
    for CX := 0 to SizeXM1 do
    begin
      AUX := Self[CX, 0, CD];
      Write(AUX:Digits:Decimals);
    end;
    WriteLn;
  end;
end;

procedure TVolume.PrintWithIndex();
var
  CX, CY, CD, SizeXM1, SizeYM1, DepthM1: integer;
begin
  SizeXM1 := SizeX - 1;
  SizeYM1 := SizeY - 1;
  DepthM1 := Depth - 1;
  for CX := 0 to SizeXM1 do
  begin
    for CY := 0 to SizeYM1 do
    begin
      for CD := 0 to DepthM1 do
      begin
        WriteLn(CX,' ',CY,' ',CD,':',Self[CX, CY, CD]);
      end;
    end;
  end;
end;

procedure TVolume.PrintDebug();
var
  MinVal, MaxVal: TNeuralFloat;
  MinPos, MaxPos: integer;
begin
  MinVal :=  GetMin();
  MinPos := FLastPos;
  MaxVal :=  GetMax();
  MaxPos := FLastPos;

  Write(
    '(',SizeX,',',SizeY,',',Depth,') - ',
    'Min: ',MinVal,' Min Pos:',MinPos,
    ' Max:',MaxVal,' Max Pos:',MaxPos,
    ' Avg:',GetAvg(),' Non Zero:',GetNonZero(),' Size:', FSize);
end;

procedure TVolume.PrintDebugChannel();
var
  CntD, MaxD: integer;
  AuxMax, AuxMin, AuxAvg: TNeuralFloat;
begin
  MaxD := Depth - 1;
  AuxMin := 0;
  AuxMax := 0;

  for CntD := 0 to MaxD do
  begin
    GetMinMaxAtDepth(CntD, AuxMin, AuxMax);
    AuxAvg := AvgAtDepth(CntD);

    WriteLn('[',CntD,':',AuxMin,' ',AuxMax,' ',AuxAvg,']');
  end;

end;

procedure TVolume.InitUniform(Value: T = 1);
var
  MulAux: Single;
begin
  Randomize();
  if (Value <> 1) then
  begin
    MulAux := Value;
    Mul(MulAux);
  end;
end;

procedure TVolume.InitGaussian(Value: T);
begin
  RandomizeGaussian(Value);
end;

procedure TVolume.InitLeCunUniform(Value: T);
var
  MulAux: Single;
begin
  // LeCun 98, Efficient Backprop
  // http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  InitUniform();
  MulAux := Value*Sqrt(2/(Size));
  Mul(MulAux);
end;

procedure TVolume.InitHeUniform(Value: T);
var
  MulAux: Single;
begin
  // This implementation is inspired on:
  // Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
  // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  // https://arxiv.org/abs/1502.01852
  InitUniform();
  MulAux := Value*Sqrt(3/(Size));
  Mul(MulAux);
end;

procedure TVolume.InitLeCunGaussian(Value: T);
var
  MulAux: Single;
begin
  // LeCun 98, Efficient Backprop
  // http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
  InitGaussian();
  MulAux := Value*Sqrt(2/(Size));
  Mul(MulAux);
end;

procedure TVolume.InitHeGaussian(Value: T);
var
  MulAux: Single;
begin
  // This implementation is inspired on:
  // Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
  // Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  // https://arxiv.org/abs/1502.01852
  InitGaussian();
  MulAux := Value*Sqrt(3/(Size));
  Mul(MulAux);
end;

procedure TVolume.InitSELU(Value: T);
begin
  InitGaussian( Value * Sqrt(1/Size) );
end;

function TVolume.SaveToString(): string;
var
  S: TNNetStringList;
  I, Hi, Lo: integer;
  version: integer;
  AuxFloat: Single;
begin
  version := 1;
  S := CreateTokenizedStringList(';');
  S.SetCapacity(FSize+10);
  S.Add( IntToStr(version) );
  S.Add( IntToStr(FSizeX) );
  S.Add( IntToStr(FSizeY) );
  S.Add( IntToStr(FDepth) );

  Hi := High(FData);
  Lo := Low(FData);
  for I := Lo to Hi do
  begin
    AuxFloat := FData[I];
    S.Add( FloatToStr(AuxFloat, FFormatSettings) );
  end;

  Result := S.GetDelimitedTextFast();
  //Result := S.DelimitedText;
  S.Free;
end;

procedure TVolume.LoadFromString(strData: string);
var
  S: TStringList;
  //version: integer;
  pSizeX, pSizeY, pDepth: integer;
  I, SCountMax: integer;
  AuxFloat: Single;
begin
  //version := 1;
  S := CreateTokenizedStringList(strData,';');

  //version := StrToInt(S[0]);
  pSizeX  := StrToInt(S[1]);
  pSizeY  := StrToInt(S[2]);
  pDepth  := StrToInt(S[3]);

  Resize(pSizeX, pSizeY, pDepth);

  {$IFDEF Debug}
  if (pSizeX * pSizeY * pDepth + 4 <> S.Count) then
  begin
    WriteLn
    (
      'Error while loading neuron from string. ',
      'SizeX: ',
      'SizeY: ',
      'SizeZ: ',
      'String Count: ', S.Count
    );
  end;
  {$ENDIF}

  if (S.Count>4) then
  begin
    SCountMax := S.Count-1;
    for I := 4 to SCountMax do
    begin
      AuxFloat := StrToFloat(S[I], FFormatSettings);
      FData[I-4] := AuxFloat;
    end;
  end;
  S.Free;
end;

{ TNNetVolume }

procedure TNNetVolume.ReSize(pSizeX, pSizeY, pDepth: integer);
begin
  inherited ReSize(pSizeX, pSizeY, pDepth);
  FDataPtr := addr(FData[0]);
end;

function TNNetVolume.GetMemSize(): integer;
begin
  Result := FSize * csNeuralFloatSize;
end;

// inspired on: http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
procedure TNNetVolume.CalculateLocalResponseFrom2D(Original: TNNetVolume;
  pSize: integer; alpha, beta: TNeuralFloat);
var
  SqrElements: TNNetVolume;
  iFrom, iTo, CountIX, CountIY: integer;
  iRawPos: integer;
  MaxX, MaxY, MaxD: integer;
  MinIX, MaxIX, MinIY, MaxIY: integer;
  CountX, CountY, CountD: integer;
  iBase: integer;
begin
  ReSize(Original);
  Fill(1);

  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;

  iTo := pSize div 2;
  iFrom := -iTo;
  SqrElements := TNNetVolume.Create();
  SqrElements.Copy(Original);
  SqrElements.Mul(SqrElements);
  SqrElements.Mul(alpha/(pSize*pSize));

  for CountX := 0 to MaxX do
  begin
    MinIX := Max(CountX + iFrom,0);
    MaxIX := Min(CountX + iTo, MaxX);
    for CountY := 0 to MaxY do
    begin
      iBase := GetRawPos(CountX, CountY, 0);
      MinIY := Max(CountY + iFrom,0);
      MaxIY := Min(CountY + iTo, MaxY);
      for CountD := 0 to MaxD do
      begin
        iRawPos := iBase + CountD;

        for CountIX := MinIX to MaxIX do
        begin
          for CountIY := MinIY to MaxIY do
          begin
            {$IFDEF FPC}
            FData[iRawPos] += SqrElements[CountIX, CountIY, CountD];
            {$ELSE}
            FData[iRawPos] := FData[iRawPos] + SqrElements[CountIX, CountIY, CountD];
            {$ENDIF}
          end;
        end;
      end;
    end;
  end;

  Pow(beta);

  SqrElements.Free;
end;

procedure TNNetVolume.CalculateLocalResponseFromDepth(Original: TNNetVolume;
  pSize: integer; alpha, beta: TNeuralFloat);
var
  SqrElements: TNNetVolume;
  iFrom, iTo, CountID: integer;
  iRawPos: integer;
  MaxX, MaxY, MaxD: integer;
  MinID, MaxID: integer;
  CountX, CountY, CountD: integer;
  sqrBase: integer;
  iBase: integer;
begin
  ReSize(Original);
  Fill(1);

  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;

  iTo := pSize div 2;
  iFrom := -iTo;
  SqrElements := TNNetVolume.Create();
  SqrElements.Copy(Original);
  SqrElements.Mul(SqrElements);
  SqrElements.Mul(alpha/pSize);

  for CountX := 0 to MaxX do
  begin
    for CountY := 0 to MaxY do
    begin
      // Self and SqrElements are both shaped like Original, so one base indexes both.
      iBase := GetRawPos(CountX, CountY, 0);
      for CountD := 0 to MaxD do
      begin
        MinID := Max(CountD + iFrom,0);
        MaxID := Min(CountD + iTo, MaxD);
        iRawPos := iBase + CountD;

        //WriteLn('CountX:', CountX,' CountY:', CountY, ' CountD:',CountD, ' MinID:',MinID, ' MaxID:', MaxID);

        sqrBase := iBase + MinID;
        for CountID := MinID to MaxID do
        begin
          {$IFDEF FPC}
          FData[iRawPos] += SqrElements.FData[sqrBase];
          {$ELSE}
          FData[iRawPos] := FData[iRawPos] + SqrElements.FData[sqrBase];
          {$ENDIF}
          Inc(sqrBase);
        end;

      end;
    end;
  end;

  Pow(beta);

  SqrElements.Free;
end;

procedure TNNetVolume.GetTokenArray(var TokenArray: TNNetTokenArray);
var
  I, vHigh: integer;
begin
  if (Length(TokenArray) <> FSize) then SetLength(TokenArray, FSize);
  if FSize > 0 then
  begin
    vHigh := FSize - 1;
    for I := 0 to vHigh do
    begin
      TokenArray[I].Token := I;
      TokenArray[I].Score := FData[I];
    end;
  end;
end;

procedure TNNetVolume.GetTokenArrayOnPixel(var TokenArray: TNNetTokenArray; X,
  Y: integer);
var
  I, vHigh, Base: integer;
begin
  if (Length(TokenArray) <> FDepth) then SetLength(TokenArray, FDepth);
  if FDepth > 0 then
  begin
    vHigh := FDepth - 1;
    Base := GetRawPos(X, Y);
    for I := 0 to vHigh do
    begin
      TokenArray[I].Token := I;
      TokenArray[I].Score := FData[Base + I];
    end;
  end;
end;

procedure TNNetVolume.InterleavedDotProduct(InterleavedAs,
  B: TNNetVolume);
var
  CntBPos, MaxBPos: integer;
  NumOriginalInterleaved: integer;
  Ofs: integer;
begin
  MaxBPos := B.FSize - 1;
  NumOriginalInterleaved := InterleavedAs.Size div B.Size;

  if FSize <> NumOriginalInterleaved then
  begin
    Resize(NumOriginalInterleaved,1,1);
  end;

  Fill(0);

  Ofs := 0;
  for CntBPos := 0 to MaxBPos do
  begin
    MulAdd(FDataPtr, InterleavedAs.GetRawPtr(Ofs), B.FData[CntBPos], NumOriginalInterleaved);
    Inc(Ofs, NumOriginalInterleaved);
  end;
end;

procedure TNNetVolume.InterleavedDotProduct(InterleavedAs, Bs: TNNetVolume;
  VectorSize: integer);
var
  CntB, CntBPos, MaxBPos: integer;
  NumA, NumB, NumBM1: integer;
  DestPointer: pointer;
  CntBVectorSizePlusCntBPos: integer;
  AOfs: integer;
begin
  NumA := InterleavedAs.Size div VectorSize;
  NumB := Bs.Size div VectorSize;
  NumBM1 := NumB - 1;

  MaxBPos := VectorSize - 1;

  if FSize <> NumA * NumB then
  begin
    Resize(1, NumB, NumA);
  end;

  Fill(0);
  for CntB := 0 to NumBM1 do
  begin
    DestPointer := Self.GetRawPtr(NumA*CntB);
    CntBVectorSizePlusCntBPos := CntB*VectorSize;
    AOfs := 0;
    for CntBPos := 0 to MaxBPos do
    begin
      //MulAdd(DestPointer, InterleavedAs.GetRawPtr(CntBPos*NumA), Bs.FData[CntB*VectorSize + CntBPos], NumA);
      MulAdd(DestPointer, InterleavedAs.GetRawPtr(AOfs), Bs.FData[CntBVectorSizePlusCntBPos], NumA);
      Inc(CntBVectorSizePlusCntBPos);
      Inc(AOfs, NumA);
    end;
  end;
end;

procedure TNNetVolume.InterleavedDotProduct(InterleavedAs, Bs: TNNetVolume;
  BStart, BFinish, VectorSize: integer);
var
  CntB, CntBPos, MaxBPos: integer;
  NumA, NumB: integer;
  DestPointer: pointer;
  CntBVectorSizePlusCntBPos: integer;
  AOfs: integer;
begin
  NumA := InterleavedAs.Size div VectorSize;
  NumB := Bs.Size div VectorSize;

  MaxBPos := VectorSize - 1;

  if FSize <> NumA * NumB then
  begin
    Resize(1, NumB, NumA);
  end;

  for CntB := BStart to BFinish do
  begin
    DestPointer := Self.GetRawPtr(NumA*CntB);
    CntBVectorSizePlusCntBPos := CntB*VectorSize;
    AOfs := 0;
    for CntBPos := 0 to MaxBPos do
    begin
      //MulAdd(DestPointer, InterleavedAs.GetRawPtr(CntBPos*NumA), Bs.FData[CntB*VectorSize + CntBPos], NumA);
      MulAdd(DestPointer, InterleavedAs.GetRawPtr(AOfs), Bs.FData[CntBVectorSizePlusCntBPos], NumA);
      Inc(CntBVectorSizePlusCntBPos);
      Inc(AOfs, NumA);
    end;
  end;
end;

procedure TNNetVolume.DotProductsPointwise(VAs, VBs: TNNetVolume;
  NoForward: boolean);
var
  VAsCount, VBsCount: integer;
begin
  VAsCount := VAs.SizeX * VAs.SizeY;
  VBsCount := VBs.SizeX * VBs.SizeY;
  if (VAsCount*VBsCount <> FSize) then
  begin
    Resize(VBsCount, 1, VAsCount);
  end;
  DotProductsPointwise(VAs, VBs, 0, VBsCount-1, NoForward);
end;

procedure TNNetVolume.DotProductsPointwise(VAs, VBs: TNNetVolume;
  BStart, BFinish: integer; NoForward: boolean);
var
  VAsCount, VBsCount: integer;
begin
  VAsCount := VAs.SizeX * VAs.SizeY;
  VBsCount := VBs.SizeX * VBs.SizeY;
  if (VAsCount*VBsCount <> FSize) then
  begin
    WriteLn(
      'TNNetVolume.DotProductsPointwise (ranged) - Self is not presized: '+
      IntToStr(FSize) + ' <> ' +
      IntToStr(VAsCount*VBsCount) + '.'
    );
    exit;
  end;

  if (VAs.Depth = VBs.Depth) then
  begin
    DotProducts(VAsCount, BStart, BFinish, VAs.Depth, VAs, VBs, NoForward);
  end
  else
  begin
    WriteLn(
      'TNNetVolume.DotProductsPointwise - Depths differ '+
      IntToStr(VAs.Depth) + ' ' +
      IntToStr(VBs.Depth) + '.'
    );
  end;
end;

(*
// A reference implementation of DotProducts is:
for CntB := 0 to MaxB do
begin
  PtrB := VBs.GetRawPtr(CntB*VectorSize);
  for CntA := 0 to MaxA do
  begin
    PtrA := VAs.GetRawPtr(CntA*VectorSize);
    Result := DotProduct(PtrA, PtrB, VectorSize);
    FData[CntB * NumAs + CntA] := Result;
  end;
end;
*)

procedure TNNetVolume.DotProducts(NumAs, NumBs, VectorSize: integer;
  VAs, VBs: TNNetVolume;
  NoForward:boolean = false);
begin
  DotProducts(NumAs, 0, NumBs-1, VectorSize, VAs, VBs, NoForward);
end;

procedure TNNetVolume.DotProducts(NumAs, BStart, BFinish, VectorSize: integer;
  VAs, VBs: TNNetVolume;
  NoForward:boolean = false);
var
  CntA, CntB, MaxA, LocalMaxA: integer;
  RowBase, AOfs: integer;
  //DestPointer: pointer;
  //CntBVectorSizePlusCntBPos: integer;
  {$IFDEF AVXANY}
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
  {$ENDIF}
  PtrA, PtrB: TNeuralFloatArrPtr;
  Result: TNeuralFloat;
  //PointwiseMinValue: TNeuralFloat;
begin
  MaxA := NumAs - 1;

  //localNumElements := (VectorSize div 4) * 4;
  //MissedElements := VectorSize - localNumElements;

  {$IFDEF AVXANY}
  MissedElements := VectorSize and 3;
  localNumElements := VectorSize xor MissedElements;
  {$ENDIF}

  if NoForward then Fill(0);

  for CntB := BStart to BFinish do
  begin
    PtrB := VBs.GetRawPtr(CntB*VectorSize);
    if NoForward
      then LocalMaxA := Min(MaxA, CntB)
      else LocalMaxA := MaxA;
    if LocalMaxA >= 0 then
    begin
      RowBase := CntB * NumAs;
      AOfs := 0;
      for CntA := 0 to LocalMaxA do
      begin
        {$IFDEF DEBUG}
        if NoForward and (CntB < CntA) then
        begin
          WriteLn('This should never happen.');
        end;
        {$ENDIF}
        PtrA := VAs.GetRawPtr(AOfs);

        {$IFDEF AVXANY}
        {$IFDEF AVX32}
        if localNumElements > 0 then
        begin
        asm
        mov ecx, localNumElements
        mov eax, PtrA
        mov edx, PtrB
        vxorps ymm0, ymm0, ymm0

        push ecx
        shr ecx,5  // number of large iterations = number of elements / 32
        jz @SkipLargeAddLoop
        vxorps ymm1, ymm1, ymm1
        vxorps ymm2, ymm2, ymm2
        vxorps ymm3, ymm3, ymm3
      @LargeAddLoop:

        vmovups ymm4, [eax]
        vmovups ymm5, [eax+32]
        vmovups ymm6, [eax+64]
        vmovups ymm7, [eax+96]

        {$IFDEF AVX2}
        vfmadd231ps ymm0, ymm4, [edx]
        vfmadd231ps ymm1, ymm5, [edx+32]
        vfmadd231ps ymm2, ymm6, [edx+64]
        vfmadd231ps ymm3, ymm7, [edx+96]
        {$ELSE}
        vmulps  ymm4, ymm4, [edx]
        vmulps  ymm5, ymm5, [edx+32]
        vmulps  ymm6, ymm6, [edx+64]
        vmulps  ymm7, ymm7, [edx+96]

        vaddps  ymm0, ymm0, ymm4
        vaddps  ymm1, ymm1, ymm5
        vaddps  ymm2, ymm2, ymm6
        vaddps  ymm3, ymm3, ymm7
        {$ENDIF}

        add eax, 128
        add edx, 128
        dec ecx
        jnz @LargeAddLoop

        vaddps ymm2, ymm2, ymm3
        vaddps ymm0, ymm0, ymm1
        vaddps ymm0, ymm0, ymm2
        VEXTRACTF128 xmm2, ymm0, 1

        vzeroupper
        addps xmm0, xmm2

      @SkipLargeAddLoop:
        pop ecx
        and ecx,$0000001F
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
          'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7'
        ];

          Result := vRes[0];
        end else
        begin
          Result := 0;
        end;
        {$ENDIF}
        {$IFDEF AVX64}
        //Write(localNumElements,' ',MissedElements);
        if localNumElements > 0 then
        begin
        asm
        mov ecx, localNumElements
        mov rax, PtrA
        mov rdx, PtrB
        {$IFDEF AVX512}
        vxorps zmm0, zmm0, zmm0
        {$ELSE}
        vxorps ymm0, ymm0, ymm0
        {$ENDIF}

        push rcx
        shr ecx,5  // number of large iterations = number of elements / 32
        jz @SkipLargeAddLoop

        {$IFDEF AVX512}
        vxorps zmm1, zmm1, zmm1
        {$ELSE}
        vxorps ymm1, ymm1, ymm1
        {$ENDIF}

      @LargeAddLoop:

        {$IFDEF AVX512}
        vmovups zmm2, [rax]
        vmovups zmm3, [rax+64]

        vmulps  zmm2, zmm2, [rdx]
        vmulps  zmm3, zmm3, [rdx+64]

        vaddps  zmm0, zmm0, zmm2
        vaddps  zmm1, zmm1, zmm3
        {$ELSE}
          vmovups ymm2, [rax]
          vmovups ymm3, [rax+32]
          vmovups ymm4, [rax+64]
          vmovups ymm5, [rax+96]

          {$IFDEF AVX2}
          vfmadd231ps ymm0, ymm2, [rdx]
          vfmadd231ps ymm1, ymm3, [rdx+32]
          vfmadd231ps ymm0, ymm4, [rdx+64]
          vfmadd231ps ymm1, ymm5, [rdx+96]
          {$ELSE}
          vmulps  ymm2, ymm2, [rdx]
          vmulps  ymm3, ymm3, [rdx+32]
          vmulps  ymm4, ymm4, [rdx+64]
          vmulps  ymm5, ymm5, [rdx+96]

          vaddps  ymm0, ymm0, ymm2
          vaddps  ymm1, ymm1, ymm3
          vaddps  ymm0, ymm0, ymm4
          vaddps  ymm1, ymm1, ymm5
          {$ENDIF}
        {$ENDIF}

        add rax, 128
        add rdx, 128
        dec ecx
        jnz @LargeAddLoop

        {$IFDEF AVX512}
        vaddps zmm0, zmm0, zmm1
        VEXTRACTF32x4 xmm2, zmm0, 1
        VEXTRACTF32x4 xmm3, zmm0, 2
        VEXTRACTF32x4 xmm4, zmm0, 3
        vzeroupper
        addps  xmm0, xmm2
        addps  xmm0, xmm3
        addps  xmm0, xmm4
        {$ELSE}
        vaddps ymm0, ymm0, ymm1
        VEXTRACTF128 xmm2, ymm0, 1
        vzeroupper
        addps  xmm0, xmm2
        {$ENDIF}

      @SkipLargeAddLoop:
        pop rcx
        and ecx,$0000001F
        jz @EndAdd
        shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
      @SmallAddLoop:
        vzeroupper

        movups xmm2, [rax]
        movups xmm3, [rdx]
        mulps xmm2, xmm3
        addps xmm0, xmm2

        add rax, 16
        add rdx, 16
        dec ecx
        jnz @SmallAddLoop

      @EndAdd:
        vzeroupper
        // Sums all elements of xmm0 into the first position
        HADDPS xmm0,xmm0
        HADDPS xmm0,xmm0

        movups vRes, xmm0
        end
        [
          'RAX', 'RCX', 'RDX',
          'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
          {$IFDEF AVX512},'zmm0', 'zmm1'{$ENDIF}
        ];

          Result := vRes[0];
        end else
        begin
          Result := 0;
        end;
        {$ENDIF}
        //Write(' A:', PtrA^[0],' B:', PtrB^[0],' -> ',Result);
        if MissedElements>0 then
        begin
          if MissedElements = 1
          then Result += PtrA^[localNumElements] * PtrB^[localNumElements]
          else if MissedElements = 2
          then Result +=
                 PtrA^[localNumElements] * PtrB^[localNumElements] +
                 PtrA^[localNumElements+1] * PtrB^[localNumElements+1]
          else Result +=
                 PtrA^[localNumElements] * PtrB^[localNumElements] +
                 PtrA^[localNumElements+1] * PtrB^[localNumElements+1] +
                 PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
        end;
        //WriteLn(' ', Result);
        {$ENDIF}
        {$IFNDEF AVXANY}
        Result := DotProduct(PtrA, PtrB, VectorSize);
        {$ENDIF}
        FData[RowBase + CntA] := Result;
        Inc(AOfs, VectorSize);
        (*
        if NoForward then
        begin
          if CntA = 0
            then PointwiseMinValue := Result
            else PointwiseMinValue := Min(Result, PointwiseMinValue);
        end; // NoForward
        *)
      end; // CntA
      (*
      if NoForward and (LocalMaxA < MaxA) then
      begin
        for CntA := LocalMaxA+1 to MaxA do
        FData[CntB * NumAs + CntA] := PointwiseMinValue;
      end;
      *)
    end; // MaxA >= 0
  end; // CntB
end;

procedure TNNetVolume.DotProductsTiled(NumAs, NumBs, VectorSize: integer; VAs, VBs: TNNetVolume; TileSizeA, TileSizeB: integer);
begin
  DotProductsTiled(NumAs, 0, NumBs-1, VectorSize, VAs, VBs, TileSizeA, TileSizeB);
end;

procedure TNNetVolume.DotProductsTiled(NumAs, BStart, BFinish, VectorSize: integer; VAs, VBs: TNNetVolume; TileSizeA, TileSizeB: integer; AStart: integer = 0; AFinish: integer = -1);
var
  CntA, CntB: Integer;
  RowBase, AOfs: integer;
  //DestPointer: pointer;
  //CntBVectorSizePlusCntBPos: integer;
  {$IFDEF AVXANY}
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
  {$ENDIF}
  PtrA, PtrB: TNeuralFloatArrPtr;
  Result: TNeuralFloat;
  // Tiling
  TileACnt, TileBCnt: integer;
  StartTileA, EndTileA, StartTileB, EndTileB: integer;
  MaxTileA, MaxTileB: integer;
begin
  //localNumElements := (VectorSize div 4) * 4;
  //MissedElements := VectorSize - localNumElements;
  {$IFDEF AVXANY}
  MissedElements := VectorSize and 3;
  localNumElements := VectorSize xor MissedElements;
  {$ENDIF}
  // A tiles are anchored at AStart with a trailing PARTIAL tile (ceil division,
  // clamped to AFinish below), so an arbitrary neuron range - the neuron-axis
  // intra-layer chunk - is safe even when TileSizeA does not divide it. AFinish
  // < 0 means "all rows" (AFinish := NumAs-1); with AStart=0 and TileSizeA
  // dividing NumAs (every non-ranged caller - conv tile sizes come from
  // GetMaxDivisor) this reduces to the original tiling. NumAs stays the output
  // row stride, so a sliced call writes exactly its neuron columns.
  if AFinish < 0 then AFinish := NumAs - 1;
  MaxTileA := ((AFinish - AStart + 1) + TileSizeA - 1) div TileSizeA - 1;
  // B tiles are anchored at BStart; ceil division so a trailing PARTIAL tile
  // (clamped to BFinish below) covers ranges TileSizeB does not divide. With
  // BStart=0 and TileSizeB dividing NumBs (every non-ranged caller - the conv
  // tile sizes come from GetMaxDivisor), this is the original tiling.
  MaxTileB := ((BFinish - BStart + 1) + TileSizeB - 1) div TileSizeB - 1;
  for TileBCnt := 0 to MaxTileB do
  begin
    StartTileB := BStart + TileBCnt * TileSizeB;
    EndTileB := Min(StartTileB + TileSizeB - 1, BFinish);
    for TileACnt := 0 to MaxTileA do
    begin
      StartTileA := AStart + TileACnt * TileSizeA;
      EndTileA := Min(StartTileA + TileSizeA - 1, AFinish);
      for CntB := StartTileB to EndTileB do
      begin
        PtrB := VBs.GetRawPtr(CntB*VectorSize);
        RowBase := CntB * NumAs;
        AOfs := StartTileA * VectorSize;
        for CntA := StartTileA to EndTileA do
        begin
          PtrA := VAs.GetRawPtr(AOfs);

          {$IFDEF AVXANY}
          {$IFDEF AVX32}
          if localNumElements > 0 then
          begin
          asm
          mov ecx, localNumElements
          mov eax, PtrA
          mov edx, PtrB
          vxorps ymm0, ymm0, ymm0

          push ecx
          shr ecx,5  // number of large iterations = number of elements / 32
          jz @SkipLargeAddLoop
          vxorps ymm1, ymm1, ymm1
          vxorps ymm2, ymm2, ymm2
          vxorps ymm3, ymm3, ymm3
        @LargeAddLoop:

          vmovups ymm4, [eax]
          vmovups ymm5, [eax+32]
          vmovups ymm6, [eax+64]
          vmovups ymm7, [eax+96]

          {$IFDEF AVX2}
          vfmadd231ps ymm0, ymm4, [edx]
          vfmadd231ps ymm1, ymm5, [edx+32]
          vfmadd231ps ymm2, ymm6, [edx+64]
          vfmadd231ps ymm3, ymm7, [edx+96]
          {$ELSE}
          vmulps  ymm4, ymm4, [edx]
          vmulps  ymm5, ymm5, [edx+32]
          vmulps  ymm6, ymm6, [edx+64]
          vmulps  ymm7, ymm7, [edx+96]

          vaddps  ymm0, ymm0, ymm4
          vaddps  ymm1, ymm1, ymm5
          vaddps  ymm2, ymm2, ymm6
          vaddps  ymm3, ymm3, ymm7
          {$ENDIF}

          add eax, 128
          add edx, 128
          dec ecx
          jnz @LargeAddLoop

          vaddps ymm2, ymm2, ymm3
          vaddps ymm0, ymm0, ymm1
          vaddps ymm0, ymm0, ymm2
          VEXTRACTF128 xmm2, ymm0, 1

          vzeroupper
          addps xmm0, xmm2

        @SkipLargeAddLoop:
          pop ecx
          and ecx,$0000001F
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
            'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7'
          ];

            Result := vRes[0];
          end else
          begin
            Result := 0;
          end;
          {$ENDIF}
          {$IFDEF AVX64}
          //Write(localNumElements,' ',MissedElements);
          if localNumElements > 0 then
          begin
          asm
          mov ecx, localNumElements
          mov rax, PtrA
          mov rdx, PtrB
          {$IFDEF AVX512}
          vxorps zmm0, zmm0, zmm0
          {$ELSE}
          vxorps ymm0, ymm0, ymm0
          {$ENDIF}

          push rcx
          shr ecx,5  // number of large iterations = number of elements / 32
          jz @SkipLargeAddLoop

          {$IFDEF AVX512}
          vxorps zmm1, zmm1, zmm1
          {$ELSE}
          vxorps ymm1, ymm1, ymm1
          {$ENDIF}

        @LargeAddLoop:

          {$IFDEF AVX512}
          vmovups zmm2, [rax]
          vmovups zmm3, [rax+64]

          vmulps  zmm2, zmm2, [rdx]
          vmulps  zmm3, zmm3, [rdx+64]

          vaddps  zmm0, zmm0, zmm2
          vaddps  zmm1, zmm1, zmm3
          {$ELSE}
            vmovups ymm2, [rax]
            vmovups ymm3, [rax+32]
            vmovups ymm4, [rax+64]
            vmovups ymm5, [rax+96]

            {$IFDEF AVX2}
            vfmadd231ps ymm0, ymm2, [rdx]
            vfmadd231ps ymm1, ymm3, [rdx+32]
            vfmadd231ps ymm0, ymm4, [rdx+64]
            vfmadd231ps ymm1, ymm5, [rdx+96]
            {$ELSE}
            vmulps  ymm2, ymm2, [rdx]
            vmulps  ymm3, ymm3, [rdx+32]
            vmulps  ymm4, ymm4, [rdx+64]
            vmulps  ymm5, ymm5, [rdx+96]

            vaddps  ymm0, ymm0, ymm2
            vaddps  ymm1, ymm1, ymm3
            vaddps  ymm0, ymm0, ymm4
            vaddps  ymm1, ymm1, ymm5
            {$ENDIF}
          {$ENDIF}

          add rax, 128
          add rdx, 128
          dec ecx
          jnz @LargeAddLoop

          {$IFDEF AVX512}
          vaddps zmm0, zmm0, zmm1
          VEXTRACTF32x4 xmm2, zmm0, 1
          VEXTRACTF32x4 xmm3, zmm0, 2
          VEXTRACTF32x4 xmm4, zmm0, 3
          vzeroupper
          addps  xmm0, xmm2
          addps  xmm0, xmm3
          addps  xmm0, xmm4
          {$ELSE}
          vaddps ymm0, ymm0, ymm1
          VEXTRACTF128 xmm2, ymm0, 1
          vzeroupper
          addps  xmm0, xmm2
          {$ENDIF}

        @SkipLargeAddLoop:
          pop rcx
          and ecx,$0000001F
          jz @EndAdd
          shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
        @SmallAddLoop:
          vzeroupper

          movups xmm2, [rax]
          movups xmm3, [rdx]
          mulps xmm2, xmm3
          addps xmm0, xmm2

          add rax, 16
          add rdx, 16
          dec ecx
          jnz @SmallAddLoop

        @EndAdd:
          vzeroupper
          // Sums all elements of xmm0 into the first position
          HADDPS xmm0,xmm0
          HADDPS xmm0,xmm0

          movups vRes, xmm0
          end
          [
            'RAX', 'RCX', 'RDX',
            'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
            {$IFDEF AVX512},'zmm0', 'zmm1'{$ENDIF}
          ];

            Result := vRes[0];
          end else
          begin
            Result := 0;
          end;
          {$ENDIF}
          //Write(' A:', PtrA^[0],' B:', PtrB^[0],' -> ',Result);
          if MissedElements>0 then
          begin
            if MissedElements = 1
            then Result += PtrA^[localNumElements] * PtrB^[localNumElements]
            else if MissedElements = 2
            then Result +=
                   PtrA^[localNumElements] * PtrB^[localNumElements] +
                   PtrA^[localNumElements+1] * PtrB^[localNumElements+1]
            else Result +=
                   PtrA^[localNumElements] * PtrB^[localNumElements] +
                   PtrA^[localNumElements+1] * PtrB^[localNumElements+1] +
                   PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
          end;
          //WriteLn(' ', Result);
          {$ENDIF}
          {$IFNDEF AVXANY}
          Result := DotProduct(PtrA, PtrB, VectorSize);
          {$ENDIF}
          FData[RowBase + CntA] := Result;
          Inc(AOfs, VectorSize);
        end;
      end;

    end; // A Tiling.
  end; // B Tiling.
end;

class function TNNetVolume.DotProductInt8(PtrA: TNeuralInt8ArrPtr;
  PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  if NumElements >= csMinAvxSize then
  begin
    Result := AVXDotProductInt8(PtrA, PtrB, NumElements);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  Result := 0;
  vHigh := NumElements - 1;
  for I := 0 to vHigh do
    Result += PtrA^[I] * PtrB^[I];
end;

class procedure TNNetVolume.MulAddInt8(PtrA, PtrB: TNeuralFloatArrPtr;
  PtrCodes: TNeuralInt8ArrPtr; pSize: integer);
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  if pSize >= csMinAvxSize then
  begin
    AVXMulAddInt8(PtrA, PtrB, PtrCodes, pSize);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  vHigh := pSize - 1;
  for I := 0 to vHigh do
    PtrA^[I] := PtrA^[I] + PtrCodes^[I] * PtrB^[I];
end;

class procedure TNNetVolume.MulAddInt8Scalar(PtrA: TNeuralFloatArrPtr;
  PtrCodes: TNeuralInt8ArrPtr; W: TNeuralFloat; pSize: integer);
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  if pSize >= csMinAvxSize then
  begin
    AVXMulAddInt8Scalar(PtrA, PtrCodes, W, pSize);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  vHigh := pSize - 1;
  for I := 0 to vHigh do
    PtrA^[I] := PtrA^[I] + W * PtrCodes^[I];
end;

procedure TNNetVolume.DotProductsTiledInt8(NumAs, NumBs, VectorSize: integer;
  const Codes: array of ShortInt; const Scales: array of TNeuralFloat;
  VBs: TNNetVolume; TileSizeA, TileSizeB: integer);
begin
  DotProductsTiledInt8(NumAs, 0, NumBs - 1, VectorSize, Codes, Scales, VBs,
    TileSizeA, TileSizeB);
end;

procedure TNNetVolume.DotProductsTiledInt8(NumAs, BStart, BFinish,
  VectorSize: integer;
  const Codes: array of ShortInt; const Scales: array of TNeuralFloat;
  VBs: TNNetVolume; TileSizeA, TileSizeB: integer;
  AStart: integer = 0; AFinish: integer = -1);
var
  CntA, CntB: integer;
  RowBase, AOfs: integer;
  PtrA: TNeuralInt8ArrPtr;
  PtrB: TNeuralFloatArrPtr;
  // Tiling
  TileACnt, TileBCnt: integer;
  StartTileA, EndTileA, StartTileB, EndTileB: integer;
  MaxTileA, MaxTileB: integer;
begin
  // Ceil-division tiling anchored at the range start with a clamped trailing
  // PARTIAL tile (same contract as the ranged DotProductsTiled), so tile sizes
  // that do not divide the range are safe. NumAs stays the output row stride,
  // so a sliced call writes exactly its own output elements.
  if AFinish < 0 then AFinish := NumAs - 1;
  MaxTileA := ((AFinish - AStart + 1) + TileSizeA - 1) div TileSizeA - 1;
  MaxTileB := ((BFinish - BStart + 1) + TileSizeB - 1) div TileSizeB - 1;
  for TileBCnt := 0 to MaxTileB do
  begin
    StartTileB := BStart + TileBCnt * TileSizeB;
    EndTileB := Min(StartTileB + TileSizeB - 1, BFinish);
    for TileACnt := 0 to MaxTileA do
    begin
      StartTileA := AStart + TileACnt * TileSizeA;
      EndTileA := Min(StartTileA + TileSizeA - 1, AFinish);
      for CntB := StartTileB to EndTileB do
      begin
        PtrB := VBs.GetRawPtr(CntB*VectorSize);
        RowBase := CntB * NumAs;
        AOfs := StartTileA * VectorSize;
        for CntA := StartTileA to EndTileA do
        begin
          PtrA := TNeuralInt8ArrPtr(@Codes[AOfs]);
          // Deferred per-row scale: fused into the store so the inner kernel
          // stays a pure raw-code reduction.
          FData[RowBase + CntA] := DotProductInt8(PtrA, PtrB, VectorSize)
            * Scales[CntA];
          Inc(AOfs, VectorSize);
        end;
      end;
    end; // A Tiling.
  end; // B Tiling.
end;

procedure TNNetGroupedVolume.GroupedDotProductsTiledInt8(Groups, NumAs, NumBs,
  VectorSize: integer; const Codes: array of ShortInt;
  const Scales: array of TNeuralFloat; VBs: TNNetVolume;
  TileSizeA, TileSizeB: integer);
var
  CntA, CntB: integer;
  GroupASize, VectorBSize, GroupIdVectorSize: integer;
  RowBase, BOfs, AOfs, InGroupLeft: integer;
  PtrA: TNeuralInt8ArrPtr;
  PtrB: TNeuralFloatArrPtr;
  // Tiling
  TileACnt, TileBCnt: integer;
  StartTileA, EndTileA, StartTileB, EndTileB: integer;
  MaxTileA, MaxTileB: integer;
begin
  GroupASize := NumAs div Groups;
  VectorBSize := VectorSize * Groups;
  // Ceil-division tiling with a clamped trailing PARTIAL tile (same contract
  // as DotProductsTiledInt8), so tile sizes that do not divide the range are
  // safe.
  MaxTileA := (NumAs + TileSizeA - 1) div TileSizeA - 1;
  MaxTileB := (NumBs + TileSizeB - 1) div TileSizeB - 1;
  for TileBCnt := 0 to MaxTileB do
  begin
    StartTileB := TileBCnt * TileSizeB;
    EndTileB := Min(StartTileB + TileSizeB - 1, NumBs - 1);
    for TileACnt := 0 to MaxTileA do
    begin
      StartTileA := TileACnt * TileSizeA;
      EndTileA := Min(StartTileA + TileSizeA - 1, NumAs - 1);
      for CntB := StartTileB to EndTileB do
      begin
        RowBase := CntB * NumAs;
        BOfs := CntB * VectorBSize;
        AOfs := StartTileA * VectorSize;
        GroupIdVectorSize := (StartTileA div GroupASize) * VectorSize;
        InGroupLeft := GroupASize - (StartTileA mod GroupASize);
        for CntA := StartTileA to EndTileA do
        begin
          PtrA := Addr(Codes[AOfs]);
          PtrB := VBs.GetRawPtr(BOfs + GroupIdVectorSize);
          FData[RowBase + CntA] := DotProductInt8(PtrA, PtrB, VectorSize)
            * Scales[CntA];
          Inc(AOfs, VectorSize);
          Dec(InGroupLeft);
          if InGroupLeft = 0 then
          begin
            Inc(GroupIdVectorSize, VectorSize);
            InGroupLeft := GroupASize;
          end;
        end;
      end;
    end;
  end;
end;

/// In this function, "As" should be weights, "VectorSize" should be the number
// of weights from each neuron. "VBs" contains input vectors. Input vectors
// should have VectorSize * Groups.
procedure TNNetGroupedVolume.GroupedDotProductsTiled(Groups, NumAs, NumBs,
  VectorSize: integer; VAs, VBs: TNNetVolume; TileSizeA, TileSizeB: integer);
var
  CntA, CntB, CntAPos, CntBPos, MaxA, MaxB: integer;
  GroupASize: integer;
  VectoreBSize: integer;
  DestPointer: pointer;
  CntBVectorSizePlusCntBPos: integer;
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
  PtrA, PtrB: TNeuralFloatArrPtr;
  Result: TNeuralFloat;
  // Tiling
  TileACnt, TileBCnt: integer;
  StartTileA, EndTileA, StartTileB, EndTileB: integer;
  MaxTileA, MaxTileB: integer;
  LocalGroupInfo: TNNetGroupInfo;
begin
  MaxA := NumAs - 1;
  MaxB := NumBs - 1;
  GroupASize := NumAs div Groups;
  VectoreBSize := VectorSize * Groups;

  {$IFDEF Debug}
  if NumAs * VectorSize <> VAs.Size then
  begin
    WriteLn('TNNetVolume.GroupedDotProductsTiled VAs size has failed.');
  end;

  if NumBs * VectoreBSize <> VBs.Size then
  begin
    WriteLn('TNNetVolume.GroupedDotProductsTiled VBs size has failed.');
  end;
  {$ENDIF}

  // is group info not cached?
  if Length(FGrInfoArray) <> NumAs then
  begin
    SetLength(FGrInfoArray, NumAs);
    for CntA := 0 to MaxA do
    begin
      LocalGroupInfo.GroupId := CntA div GroupASize;
      LocalGroupInfo.GroupIdVectorSize := LocalGroupInfo.GroupId*VectorSize;
      LocalGroupInfo.PtrA := VAs.GetRawPtr(CntA*VectorSize);
      FGrInfoArray[CntA] := LocalGroupInfo;
    end;
  end;

  //localNumElements := (VectorSize div 4) * 4;
  //MissedElements := VectorSize - localNumElements;
  MissedElements := VectorSize and 3;
  localNumElements := VectorSize xor MissedElements;
  MaxTileA := (NumAs div TileSizeA) - 1;
  MaxTileB := (NumBs div TileSizeB) - 1;
  for TileBCnt := 0 to MaxTileB do
  begin
    StartTileB := TileBCnt * TileSizeB;
    EndTileB := StartTileB + TileSizeB - 1;
    for TileACnt := 0 to MaxTileA do
    begin
      StartTileA := TileACnt * TileSizeA;
      EndTileA := StartTileA + TileSizeA - 1;
      for CntA := StartTileA to EndTileA do
      begin
        //GroupId := CntA div GroupASize;
        //GroupIdVectorSize := GroupId*VectorSize;
        //PtrA := VAs.GetRawPtr(CntA*VectorSize);
        LocalGroupInfo := FGrInfoArray[CntA];
        PtrA := LocalGroupInfo.PtrA;
        for CntB := StartTileB to EndTileB do
        begin
          PtrB := VBs.GetRawPtr(CntB*VectoreBSize + LocalGroupInfo.GroupIdVectorSize);
          {$IFDEF AVXANY}
          {$IFDEF AVX32}
          if localNumElements > 0 then
          begin
          asm
          mov ecx, localNumElements
          mov eax, PtrA
          mov edx, PtrB
          vxorps ymm0, ymm0, ymm0

          push ecx
          shr ecx,5  // number of large iterations = number of elements / 32
          jz @SkipLargeAddLoop
          vxorps ymm1, ymm1, ymm1
          vxorps ymm2, ymm2, ymm2
          vxorps ymm3, ymm3, ymm3
        @LargeAddLoop:

          vmovups ymm4, [eax]
          vmovups ymm5, [eax+32]
          vmovups ymm6, [eax+64]
          vmovups ymm7, [eax+96]

          {$IFDEF AVX2}
          vfmadd231ps ymm0, ymm4, [edx]
          vfmadd231ps ymm1, ymm5, [edx+32]
          vfmadd231ps ymm2, ymm6, [edx+64]
          vfmadd231ps ymm3, ymm7, [edx+96]
          {$ELSE}
          vmulps  ymm4, ymm4, [edx]
          vmulps  ymm5, ymm5, [edx+32]
          vmulps  ymm6, ymm6, [edx+64]
          vmulps  ymm7, ymm7, [edx+96]

          vaddps  ymm0, ymm0, ymm4
          vaddps  ymm1, ymm1, ymm5
          vaddps  ymm2, ymm2, ymm6
          vaddps  ymm3, ymm3, ymm7
          {$ENDIF}

          add eax, 128
          add edx, 128
          dec ecx
          jnz @LargeAddLoop

          vaddps ymm2, ymm2, ymm3
          vaddps ymm0, ymm0, ymm1
          vaddps ymm0, ymm0, ymm2
          VEXTRACTF128 xmm2, ymm0, 1

          vzeroupper
          addps xmm0, xmm2

        @SkipLargeAddLoop:
          pop ecx
          and ecx,$0000001F
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
            'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7'
          ];

            Result := vRes[0];
          end else
          begin
            Result := 0;
          end;
          {$ENDIF}
          {$IFDEF AVX64}
          //Write(localNumElements,' ',MissedElements);
          if localNumElements > 0 then
          begin
          asm
          mov ecx, localNumElements
          mov rax, PtrA
          mov rdx, PtrB
          {$IFDEF AVX512}
          vxorps zmm0, zmm0, zmm0
          {$ELSE}
          vxorps ymm0, ymm0, ymm0
          {$ENDIF}

          push rcx
          shr ecx,5  // number of large iterations = number of elements / 32
          jz @SkipLargeAddLoop

          {$IFDEF AVX512}
          vxorps zmm1, zmm1, zmm1
          {$ELSE}
          vxorps ymm1, ymm1, ymm1
          {$ENDIF}

        @LargeAddLoop:

          {$IFDEF AVX512}
          vmovups zmm2, [rax]
          vmovups zmm3, [rax+64]

          vmulps  zmm2, zmm2, [rdx]
          vmulps  zmm3, zmm3, [rdx+64]

          vaddps  zmm0, zmm0, zmm2
          vaddps  zmm1, zmm1, zmm3
          {$ELSE}
            vmovups ymm2, [rax]
            vmovups ymm3, [rax+32]
            vmovups ymm4, [rax+64]
            vmovups ymm5, [rax+96]

            {$IFDEF AVX2}
            vfmadd231ps ymm0, ymm2, [rdx]
            vfmadd231ps ymm1, ymm3, [rdx+32]
            vfmadd231ps ymm0, ymm4, [rdx+64]
            vfmadd231ps ymm1, ymm5, [rdx+96]
            {$ELSE}
            vmulps  ymm2, ymm2, [rdx]
            vmulps  ymm3, ymm3, [rdx+32]
            vmulps  ymm4, ymm4, [rdx+64]
            vmulps  ymm5, ymm5, [rdx+96]

            vaddps  ymm0, ymm0, ymm2
            vaddps  ymm1, ymm1, ymm3
            vaddps  ymm0, ymm0, ymm4
            vaddps  ymm1, ymm1, ymm5
            {$ENDIF}
          {$ENDIF}

          add rax, 128
          add rdx, 128
          dec ecx
          jnz @LargeAddLoop

          {$IFDEF AVX512}
          vaddps zmm0, zmm0, zmm1
          VEXTRACTF32x4 xmm2, zmm0, 1
          VEXTRACTF32x4 xmm3, zmm0, 2
          VEXTRACTF32x4 xmm4, zmm0, 3
          vzeroupper
          addps  xmm0, xmm2
          addps  xmm0, xmm3
          addps  xmm0, xmm4
          {$ELSE}
          vaddps ymm0, ymm0, ymm1
          VEXTRACTF128 xmm2, ymm0, 1
          vzeroupper
          addps  xmm0, xmm2
          {$ENDIF}

        @SkipLargeAddLoop:
          pop rcx
          and ecx,$0000001F
          jz @EndAdd
          shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
        @SmallAddLoop:
          vzeroupper

          movups xmm2, [rax]
          movups xmm3, [rdx]
          mulps xmm2, xmm3
          addps xmm0, xmm2

          add rax, 16
          add rdx, 16
          dec ecx
          jnz @SmallAddLoop

        @EndAdd:
          vzeroupper
          // Sums all elements of xmm0 into the first position
          HADDPS xmm0,xmm0
          HADDPS xmm0,xmm0

          movups vRes, xmm0
          end
          [
            'RAX', 'RCX', 'RDX',
            'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
            {$IFDEF AVX512},'zmm0', 'zmm1'{$ENDIF}
          ];

            Result := vRes[0];
          end else
          begin
            Result := 0;
          end;
          {$ENDIF}
          //Write(' A:', PtrA^[0],' B:', PtrB^[0],' -> ',Result);
          if MissedElements>0 then
          begin
            if MissedElements = 1
            then Result += PtrA^[localNumElements] * PtrB^[localNumElements]
            else if MissedElements = 2
            then Result +=
                   PtrA^[localNumElements] * PtrB^[localNumElements] +
                   PtrA^[localNumElements+1] * PtrB^[localNumElements+1]
            else Result +=
                   PtrA^[localNumElements] * PtrB^[localNumElements] +
                   PtrA^[localNumElements+1] * PtrB^[localNumElements+1] +
                   PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
          end;
          //WriteLn(' ', Result);
          {$ENDIF}
          {$IFNDEF AVXANY}
          Result := DotProduct(PtrA, PtrB, VectorSize);
          {$ENDIF}
          // Use for debug only: WriteLn('Grouped dot product result [', CntB,' ',NumAs,' ',CntA,' Pos:',CntB * NumAs + CntA,']:',Result);
          FData[CntB * NumAs + CntA] := Result;
        end;
      end;

    end; // A Tiling.
  end; // B Tiling.
end;

procedure TNNetVolume.AddArea(DestX, DestY, OriginX, OriginY, LenX,
  LenY: integer; Original: TNNetVolume);
var
  CntY: integer;
  SizeXDepth: integer;
  PtrA, PtrB: Pointer;
  MaxLenY: integer;
begin
  if Self.Depth = Original.Depth then
  begin
    SizeXDepth := LenX * Self.Depth;
    MaxLenY := LenY - 1;
    for CntY := 0 to MaxLenY do
    begin
      PtrA := Self.GetRawPtr(DestX, DestY+CntY);
      PtrB := Original.GetRawPtr(OriginX, OriginY+CntY);
      Add(PtrA, PtrB, SizeXDepth);
    end;
  end
  {$IFDEF Debug}
  else
  begin
    WriteLn('Error at TNNetVolume.AddArea: depth size doesn''t match. ',
      Self.Depth, ' ',Original.Depth);
  end
  {$ENDIF};
end;

function TNNetVolume.HasAVX: boolean;
begin
  {$IFDEF AVXANY}
  Result := true;
  {$ELSE}
  Result := false;
  {$ENDIF}
end;

function TNNetVolume.HasAVX2: boolean;
begin
  {$IFDEF AVX2}
  Result := true;
  {$ELSE}
  Result := false;
  {$ENDIF}
end;

function TNNetVolume.HasAVX512: boolean;
begin
  {$IFDEF AVX512}
  Result := true;
  {$ELSE}
  Result := false;
  {$ENDIF}
end;

function TNNetVolume.PearsonCorrelation(Y: TNNetVolume): TNeuralFloat;
var
  X, XMinusAvg, YMinusAvg: TNNetVolume;
  SumX, SumY: TNeuralFloat;
  AvgX, AvgY: TNeuralFloat;
  VarianceX, VarianceY: TNeuralFloat;
  StdDevX, StdDevY: TNeuralFloat;
  Covariance: TNeuralFloat;
  SizeFloat: TNeuralFloat;
begin
  X := Self;
  if (X.Size < 1) or (Y.Size < 1) or (X.Size <> Y.Size) then
  begin
    Result := 0;
    exit;
  end;

  SizeFloat := X.Size;

  SumX := X.GetSum();
  SumY := Y.GetSum();

  AvgX := SumX / SizeFloat;
  AvgY := SumY / SizeFloat;

  XMinusAvg := TNNetVolume.Create(1, 1, X.Size, -AvgX);
  YMinusAvg := TNNetVolume.Create(1, 1, Y.Size, -AvgY);

  XMinusAvg.Add(X);
  YMinusAvg.Add(Y);

  VarianceX := XMinusAvg.GetSumSqr() / SizeFloat;
  VarianceY := YMinusAvg.GetSumSqr() / SizeFloat;

  StdDevX := Sqrt( VarianceX );
  StdDevY := Sqrt( VarianceY );

  (*
  // Debug code
  WriteLn('Sum X:', SumX, ' Avg X:', AvgX, ' Variance X:', VarianceX, ' StdDev X:', StdDevX);
  WriteLn('Sum Y:', SumY, ' Avg Y:', AvgY, ' Variance Y:', VarianceX, ' StdDev Y:', StdDevY);
  WriteLn('Variance X:', X.GetVariance() );
  WriteLn('Variance Y:', Y.GetVariance() );
  *)

  if (StdDevX <> 0) and (StdDevY<>0) then
  begin
    Covariance := XMinusAvg.DotProduct(YMinusAvg) / SizeFloat;
    Result := (Covariance) / (StdDevX * StdDevY);
    Result := NeuronForceRange(Result, 1);
  end
  else
  begin
    Result := 0;
  end;

  YMinusAvg.Free;
  XMinusAvg.Free;
end;

procedure TNNetVolume.AddSumChannel(Original: TNNetVolume);
var
  MaxXY, CntXY: integer;
  PtrDest: TNeuralFloatArrPtr;
  PtrSource: TNeuralFloatPtr;
  NumElements: integer;
begin
  MaxXY := (Original.SizeX * Original.SizeY) - 1;
  NumElements := Original.Depth;
  if Size <> NumElements then Resize(1,1,NumElements);
  PtrDest := FDataPtr;
  PtrSource := TNeuralFloatPtr(Original.DataPtr);
  for CntXY := 0 to MaxXY do
  begin
    Add(PtrDest, TNeuralFloatArrPtr(PtrSource), NumElements);
    Inc(PtrSource, NumElements);
  end;
end;

procedure TNNetVolume.AddSumSqrChannel(Original: TNNetVolume);
var
  MaxXY, CntXY: integer;
  PtrDest: TNeuralFloatArrPtr;
  PtrSource: TNeuralFloatPtr;
  NumElements: integer;
begin
  MaxXY := (Original.SizeX * Original.SizeY) - 1;
  NumElements := Original.Depth;
  if Size <> NumElements then Resize(1,1,NumElements);
  PtrDest := FDataPtr;
  PtrSource := TNeuralFloatPtr(Original.DataPtr);
  for CntXY := 0 to MaxXY do
  begin
    MulAdd(PtrDest, TNeuralFloatArrPtr(PtrSource), TNeuralFloatArrPtr(PtrSource), NumElements);
    Inc(PtrSource, NumElements);
  end;
end;

procedure TNNetVolume.AddToChannels(Original: TNNetVolume);
var
  MaxXY, CntXY: integer;
  PtrDest: TNeuralFloatPtr;
  PtrSource: TNeuralFloatArrPtr;
  NumElements: integer;
begin
  MaxXY := (SizeX * SizeY) - 1;
  NumElements := Depth;
  if Original.Size <> NumElements then
  begin
    raise Exception.Create('AddToChannels: volumes aren''t compatible.');
  end
  else
  begin
    PtrDest := TNeuralFloatPtr(FDataPtr);
    PtrSource := Original.DataPtr;
    for CntXY := 0 to MaxXY do
    begin
      Add(TNeuralFloatArrPtr(PtrDest), PtrSource, NumElements);
      Inc(PtrDest, NumElements);
    end;
  end;
end;

procedure TNNetVolume.MulChannels(Original: TNNetVolume);
var
  MaxXY, CntXY: integer;
  PtrDest: TNeuralFloatPtr;
  PtrSource: TNeuralFloatArrPtr;
  NumElements: integer;
begin
  MaxXY := (SizeX * SizeY) - 1;
  NumElements := Depth;
  if Original.Size <> NumElements then
  begin
    raise Exception.Create('MulChannels: volumes aren''t compatible: ' +
      IntToStr(Original.Size) + ' , ' + IntToStr(NumElements));
  end
  else
  begin
    PtrDest := TNeuralFloatPtr(FDataPtr);
    PtrSource := Original.DataPtr;
    for CntXY := 0 to MaxXY do
    begin
      Mul(TNeuralFloatArrPtr(PtrDest), PtrSource, NumElements);
      Inc(PtrDest, NumElements);
    end;
  end;
end;

procedure TNNetVolume.Mul(Original: TNNetVolume);
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at TNNetVolume.Mul: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Mul(FDataPtr, Original.DataPtr, Size);
end;

procedure TNNetVolume.NormalizeMax(Value: TNeuralFloat);
var
  MaxValue: TNeuralFloat;
begin
  MaxValue := GetMaxAbs();
  if MaxValue > 0 then
  begin
    Mul( Value/MaxValue );
  end;
end;

// https://en.wikipedia.org/wiki/Recurrence_plot
procedure TNNetVolume.RecurrencePlot(Original: TNNetVolume; Threshold: TNeuralFloat);
var
  MaxX, CntX, CntY: integer;
  LocalDiff: TNeuralFloat;
  Dst1Pos, Dst2Pos: integer;
  StrideXSelf, StrideYSelf: integer;
begin
  if Original.Size > 0 then
  begin
    Resize(Original.Size, Original.Size, 1);
    MaxX := SizeX - 1;
    StrideXSelf := FDepth;          // Self X-slot step per CntY (Depth=1)
    StrideYSelf := FSizeX * FDepth; // Self Y-slot step per CntY
    for CntX := 0 to MaxX do
    begin
      Dst1Pos := GetRawPos(CntX, 0, 0); // Self[CntX, CntY, 0]
      Dst2Pos := GetRawPos(0, CntX, 0); // Self[CntY, CntX, 0]
      for CntY := 0 to CntX do
      begin
        if Abs(Original.FData[CntX] - Original.FData[CntY]) <= Threshold
        then LocalDiff := 1
        else LocalDiff := 0;
        FData[Dst1Pos] := LocalDiff;
        FData[Dst2Pos] := LocalDiff;
        Inc(Dst1Pos, StrideYSelf);
        Inc(Dst2Pos, StrideXSelf);
      end;
    end;
  end;
end;

procedure TNNetVolume.RecurrencePlotCAI(Original: TNNetVolume);
var
  MaxX, MaxD, CntX, CntY, CntD: integer;
  LocalDiff: TNeuralFloat;
  OrigA: TNeuralFloat;
  SrcBPos, Dst1Pos, Dst2Pos: integer;
  StrideXOrig, StrideXSelf, StrideYSelf: integer;
begin
  if Original.Size > 0 then
  begin
    Resize(Original.SizeX, Original.SizeX, Original.Depth);
    MaxX := SizeX - 1;
    MaxD := Depth - 1;
    StrideXOrig := Original.FDepth;      // Original X-slot step per CntY
    StrideXSelf := FDepth;               // Self X-slot step per CntY
    StrideYSelf := FSizeX * FDepth;      // Self Y-slot step per CntY
    for CntD := 0 to MaxD do
    begin
      for CntX := 0 to MaxX do
      begin
        OrigA := Original.FData[Original.GetRawPos(CntX, 0, CntD)];
        SrcBPos := Original.GetRawPos(0, 0, CntD);
        Dst1Pos := GetRawPos(CntX, 0, CntD);
        Dst2Pos := GetRawPos(0, CntX, CntD);
        for CntY := 0 to CntX do
        begin
          LocalDiff := OrigA - Original.FData[SrcBPos];
          FData[Dst1Pos] := LocalDiff;
          FData[Dst2Pos] := Abs(LocalDiff);
          Inc(SrcBPos, StrideXOrig);
          Inc(Dst1Pos, StrideYSelf);
          Inc(Dst2Pos, StrideXSelf);
        end;
      end;
    end;
  end;
end;

{$RANGECHECKS OFF}
{$OVERFLOWCHECKS OFF}

{$IFDEF AVX32}
procedure AVXFill(PtrA: TNeuralFloatArrPtr; FillOp: TNeuralFloat; NumElements: integer);
var
  I: integer;
  localNumElements, MissedElements: integer;
  FillOpPtr: pointer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    FillOpPtr := Addr(FillOp);
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  mov edx, FillOpPtr

  VBROADCASTSS ymm7, [edx]

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:

  vmovups [eax],    ymm7
  vmovups [eax+32], ymm7
  vmovups [eax+64], ymm7
  vmovups [eax+96], ymm7

  add eax, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups [eax], xmm7

  add eax, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'EAX', 'ECX', 'EDX',
    'ymm7'
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := FillOp;
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := FillOp;
      if MissedElements>2 then PtrA^[localNumElements+2] := FillOp;
    end;
  end;
end;

// PtrA := PtrA * MulOp1 + PtrB * MulOp2
// RDX  := RDX  * ymm5   + RAX  * ymm6
procedure AVXMulMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; MulOp1, MulOp2: TNeuralFloat; NumElements: integer);
var
  MulOpPtr1, MulOpPtr2: pointer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr1 := Addr(MulOp1);
    MulOpPtr2 := Addr(MulOp2);
  asm
  mov ecx, localNumElements
  mov eax, PtrB

  mov edx, MulOpPtr1
  VBROADCASTSS ymm5, [edx]

  mov edx, MulOpPtr2
  VBROADCASTSS ymm6, [edx]

  mov edx, PtrA

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:
  vmulps  ymm0, ymm6, [eax]
  vmulps  ymm1, ymm6, [eax+32]

  vmulps  ymm2, ymm5, [edx]
  vmulps  ymm3, ymm5, [edx+32]

  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm1, ymm1, ymm3

  vmovups [edx],    ymm0
  vmovups [edx+32], ymm1

  vmulps  ymm0, ymm6, [eax+64]
  vmulps  ymm1, ymm6, [eax+96]

  vmulps  ymm2, ymm5, [edx+64]
  vmulps  ymm3, ymm5, [edx+96]

  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm1, ymm1, ymm3

  vmovups [edx+64], ymm0
  vmovups [edx+96], ymm1

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [eax]
  movups  xmm4, [edx]

  mulps   xmm2, xmm6
  mulps   xmm4, xmm5

  addps   xmm4, xmm2
  movups  [edx], xmm4

  add eax, 16
  add edx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'EAX', 'ECX', 'EDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6'
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements]*MulOp1 + MulOp2*PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1]*MulOp1 + MulOp2*PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2]*MulOp1 + MulOp2*PtrB^[localNumElements+2];
    end;
  end;
end;


procedure AVXMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; MulOp: TNeuralFloat; NumElements: integer);
var
  MulOpPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr := Addr(MulOp);
  asm
  mov ecx, localNumElements
  mov eax, PtrB
  mov edx, MulOpPtr

  VBROADCASTSS ymm7, [edx]
  mov edx, PtrA

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:
  {$IFDEF AVX2}
  vmovups ymm0, [edx]
  vmovups ymm1, [edx+32]
  vmovups ymm2, [edx+64]
  vmovups ymm3, [edx+96]

  vfmadd231ps ymm0, ymm7, [eax]
  vfmadd231ps ymm1, ymm7, [eax+32]
  vfmadd231ps ymm2, ymm7, [eax+64]
  vfmadd231ps ymm3, ymm7, [eax+96]
  {$ELSE}
  vmulps  ymm0, ymm7, [eax]
  vmulps  ymm1, ymm7, [eax+32]
  vmulps  ymm2, ymm7, [eax+64]
  vmulps  ymm3, ymm7, [eax+96]

  vaddps  ymm0, ymm0, [edx]
  vaddps  ymm1, ymm1, [edx+32]
  vaddps  ymm2, ymm2, [edx+64]
  vaddps  ymm3, ymm3, [edx+96]
  {$ENDIF}

  vmovups [edx],    ymm0
  vmovups [edx+32], ymm1
  vmovups [edx+64], ymm2
  vmovups [edx+96], ymm3

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [eax]
  movups  xmm4, [edx]

  mulps   xmm2, xmm7
  addps   xmm4, xmm2

  movups  [edx], xmm4

  add eax, 16
  add edx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'EAX', 'ECX', 'EDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm7'
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] += MulOp*PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] += MulOp*PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] += MulOp*PtrB^[localNumElements+2];
    end;
  end;
end;

procedure AVXMulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr; NumElements: integer);  overload;
var
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov edx, PtrA
  mov eax, PtrB
  mov ebx, PtrC

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:
    vmovups ymm4, [ebx]
    vmovups ymm5, [ebx+32]

    vmulps  ymm0, ymm4, [eax]
    vmulps  ymm1, ymm5, [eax+32]

    vaddps  ymm0, ymm0, [edx]
    vaddps  ymm1, ymm1, [edx+32]

    vmovups [edx],    ymm0
    vmovups [edx+32], ymm1

    vmovups ymm4, [ebx+64]
    vmovups ymm5, [ebx+96]

    vmulps  ymm2, ymm4, [eax+64]
    vmulps  ymm3, ymm5, [eax+96]

    vaddps  ymm2, ymm2, [edx+64]
    vaddps  ymm3, ymm3, [edx+96]

    vmovups [edx+64], ymm2
    vmovups [edx+96], ymm3

  add eax, 128
  add edx, 128
  add ebx, 128

  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [eax]
  movups  xmm5, [ebx]
  movups  xmm4, [edx]

  mulps   xmm2, xmm5
  addps   xmm4, xmm2

  movups  [edx], xmm4

  add eax, 16
  add ebx, 16
  add edx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'EAX', 'EBX', 'ECX', 'EDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] += PtrB^[localNumElements]*PtrC^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] += PtrB^[localNumElements+1]*PtrC^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] += PtrB^[localNumElements+2]*PtrC^[localNumElements+2];
    end;
  end;
end;


procedure AVXCopyRelu(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
var
  ZeroVar: TNeuralFloat;
  ZeroVarPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  ZeroVar := 0;
  if localNumElements > 0 then
  begin
    ZeroVarPtr := Addr(ZeroVar);
  asm
  mov ecx, localNumElements
  mov eax, PtrB
  mov edx, ZeroVarPtr

  VBROADCASTSS ymm5, [edx]

  mov edx, PtrA

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:
    VMAXPS ymm0, ymm5, [eax]
    VMAXPS ymm1, ymm5, [eax+32]
    VMAXPS ymm2, ymm5, [eax+64]
    VMAXPS ymm3, ymm5, [eax+96]

    vmovups [edx],    ymm0
    vmovups [edx+32], ymm1
    vmovups [edx+64], ymm2
    vmovups [edx+96], ymm3

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [eax]
  MAXPS   xmm2, xmm5

  movups  [edx], xmm2

  add eax, 16
  add edx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'EAX', 'ECX', 'EDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm0','zmm1','zmm5'{$ENDIF}
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := Max(0,PtrB^[localNumElements]);
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := Max(0,PtrB^[localNumElements+1]);
      if MissedElements>2 then PtrA^[localNumElements+2] := Max(0,PtrB^[localNumElements+2]);
    end;
  end;
end;

procedure AVXMul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat; NumElements: integer); overload;
var
  MulOpPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr := Addr(MulOp);
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  mov edx, MulOpPtr

  VBROADCASTSS ymm7, [edx]

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:

  vmulps  ymm2, ymm7, [eax]
  vmulps  ymm3, ymm7, [eax+32]
  vmulps  ymm4, ymm7, [eax+64]
  vmulps  ymm5, ymm7, [eax+96]

  vmovups [eax],    ymm2
  vmovups [eax+32], ymm3
  vmovups [eax+64], ymm4
  vmovups [eax+96], ymm5

  add eax, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [eax]
  mulps   xmm2, xmm7
  movups [eax], xmm2

  add eax, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'EAX', 'ECX', 'EDX',
    'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm7'
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] *= MulOp;
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] *= MulOp;
      if MissedElements>2 then PtrA^[localNumElements+2] *= MulOp;
    end;
  end;
end;

procedure AVXMul(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer); overload;
var
  MulOpPtr1, MulOpPtr2: pointer;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrB
  mov edx, PtrA

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:

  vmovups  ymm0, [eax]
  vmovups  ymm1, [eax+32]
  vmovups  ymm2, [eax+64]
  vmovups  ymm3, [eax+96]

  vmulps  ymm0, ymm0, [edx]
  vmulps  ymm1, ymm1, [edx+32]
  vmulps  ymm2, ymm2, [edx+64]
  vmulps  ymm3, ymm3, [edx+96]

  vmovups [edx],    ymm0
  vmovups [edx+32], ymm1
  vmovups [edx+64], ymm2
  vmovups [edx+96], ymm3

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations

@SmallAddLoop:

  movups  xmm2, [eax]
  movups  xmm4, [edx]

  mulps   xmm2, xmm4
  movups  [edx], xmm2

  add eax, 16
  add edx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'EAX', 'ECX', 'EDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3'
    {$IFDEF AVX512},'zmm0', 'zmm1'{$ENDIF}
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements] * PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1] * PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
    end;
  end;
end;

procedure AVXAdd(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
var
  I: integer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  mov edx, PtrB

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
@LargeAddLoop:

  vmovups ymm2, [eax]
  vmovups ymm3, [eax+32]
  vmovups ymm4, [eax+64]
  vmovups ymm5, [eax+96]

  vaddps  ymm2, ymm2, [edx]
  vaddps  ymm3, ymm3, [edx+32]
  vaddps  ymm4, ymm4, [edx+64]
  vaddps  ymm5, ymm5, [edx+96]

  vmovups [eax],    ymm2
  vmovups [eax+32], ymm3
  vmovups [eax+64], ymm4
  vmovups [eax+96], ymm5

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

  vzeroupper

@SkipLargeAddLoop:
  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [eax]
  movups xmm3, [edx]
  addps xmm2, xmm3
  movups [eax], xmm2

  add eax, 16
  add edx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end
  [
    'EAX', 'ECX', 'EDX',
    'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];
  end;

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] += PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] += PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] += PtrB^[localNumElements+2];
    end;
  end;
end;

procedure AVXMax(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
var
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  mov edx, PtrB

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeMaxLoop
@LargeMaxLoop:

  vmovups ymm2, [eax]
  vmovups ymm3, [eax+32]
  vmovups ymm4, [eax+64]
  vmovups ymm5, [eax+96]

  vmaxps  ymm2, ymm2, [edx]
  vmaxps  ymm3, ymm3, [edx+32]
  vmaxps  ymm4, ymm4, [edx+64]
  vmaxps  ymm5, ymm5, [edx+96]

  vmovups [eax],    ymm2
  vmovups [eax+32], ymm3
  vmovups [eax+64], ymm4
  vmovups [eax+96], ymm5

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeMaxLoop

  vzeroupper

@SkipLargeMaxLoop:
  pop ecx
  and ecx,$0000001F
  jz @EndMax
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallMaxLoop:
  vzeroupper

  movups xmm2, [eax]
  movups xmm3, [edx]
  maxps  xmm2, xmm3
  movups [eax], xmm2

  add eax, 16
  add edx, 16
  dec ecx
  jnz @SmallMaxLoop

@EndMax:
  end
  [
    'EAX', 'ECX', 'EDX',
    'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];
  end;

  if MissedElements>0 then
  begin
    if PtrB^[localNumElements] > PtrA^[localNumElements] then
      PtrA^[localNumElements] := PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      if PtrB^[localNumElements+1] > PtrA^[localNumElements+1] then
        PtrA^[localNumElements+1] := PtrB^[localNumElements+1];
      if MissedElements>2 then
        if PtrB^[localNumElements+2] > PtrA^[localNumElements+2] then
          PtrA^[localNumElements+2] := PtrB^[localNumElements+2];
    end;
  end;
end;

function AVXSumDiff(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  mov edx, PtrB

  vxorps ymm0, ymm0, ymm0

  {$IFDEF AVX2}
  VPCMPEQD  ymm1, ymm1, ymm1
  VPSRLD    ymm1, ymm1, 1
  {$ELSE}
  VPCMPEQD  xmm2, xmm2, xmm2
  VPCMPEQD  xmm3, xmm3, xmm3
  VPSRLD    xmm2, xmm2, 1
  VPSRLD    xmm3, xmm3, 1
  VPERM2F128 ymm1, ymm2, ymm3, 0
  {$ENDIF}

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
@LargeAddLoop:

  vmovups ymm2, [eax]
  vmovups ymm3, [eax+32]
  vmovups ymm4, [eax+64]
  vmovups ymm5, [eax+96]

  vsubps  ymm2, ymm2, [edx]
  vsubps  ymm3, ymm3, [edx+32]
  vsubps  ymm4, ymm4, [edx+64]
  vsubps  ymm5, ymm5, [edx+96]

  // absolute values
  vandps  ymm2, ymm2, ymm1
  vandps  ymm3, ymm3, ymm1
  vandps  ymm4, ymm4, ymm1
  vandps  ymm5, ymm5, ymm1

  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm0, ymm0, ymm3
  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm0, ymm0, ymm5

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2

@SkipLargeAddLoop:
  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [eax]
  movups xmm3, [edx]
  subps  xmm2, xmm3
  andps  xmm2, xmm1
  addps  xmm0, xmm2

  add eax, 16
  add edx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'EAX', 'ECX', 'EDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];
    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += Abs(PtrA^[localNumElements]-PtrB^[localNumElements])
    else if MissedElements = 2
    then Result +=
           Abs(PtrA^[localNumElements]-PtrB^[localNumElements]) +
           Abs(PtrA^[localNumElements+1]-PtrB^[localNumElements+1])
    else Result +=
           Abs(PtrA^[localNumElements]-PtrB^[localNumElements]) +
           Abs(PtrA^[localNumElements+1]-PtrB^[localNumElements+1]) +
           Abs(PtrA^[localNumElements+2]-PtrB^[localNumElements+2]);
  end;
end;

function AVXDistanceSqr(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  mov edx, PtrB

  vxorps ymm0, ymm0, ymm0

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
@LargeAddLoop:

  vmovups ymm2, [eax]
  vmovups ymm3, [eax+32]
  vmovups ymm4, [eax+64]
  vmovups ymm5, [eax+96]

  vsubps  ymm2, ymm2, [edx]
  vsubps  ymm3, ymm3, [edx+32]
  vsubps  ymm4, ymm4, [edx+64]
  vsubps  ymm5, ymm5, [edx+96]

  vmulps  ymm2, ymm2, ymm2
  vmulps  ymm3, ymm3, ymm3
  vmulps  ymm4, ymm4, ymm4
  vmulps  ymm5, ymm5, ymm5

  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm0, ymm0, ymm3
  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm0, ymm0, ymm5

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2

@SkipLargeAddLoop:
  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [eax]
  movups xmm3, [edx]
  subps  xmm2, xmm3
  mulps  xmm2, xmm2
  addps  xmm0, xmm2

  add eax, 16
  add edx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'EAX', 'ECX', 'EDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];
    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += Sqr(PtrA^[localNumElements]-PtrB^[localNumElements])
    else if MissedElements = 2
    then Result +=
           Sqr(PtrA^[localNumElements]-PtrB^[localNumElements]) +
           Sqr(PtrA^[localNumElements+1]-PtrB^[localNumElements+1])
    else Result +=
           Sqr(PtrA^[localNumElements]-PtrB^[localNumElements]) +
           Sqr(PtrA^[localNumElements+1]-PtrB^[localNumElements+1]) +
           Sqr(PtrA^[localNumElements+2]-PtrB^[localNumElements+2]);
  end;
end;

procedure AVXSub(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
var
  I: integer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  mov edx, PtrB

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
@LargeAddLoop:

  vmovups ymm2, [eax]
  vmovups ymm3, [eax+32]
  vmovups ymm4, [eax+64]
  vmovups ymm5, [eax+96]

  vsubps  ymm2, ymm2, [edx]
  vsubps  ymm3, ymm3, [edx+32]
  vsubps  ymm4, ymm4, [edx+64]
  vsubps  ymm5, ymm5, [edx+96]

  vmovups [eax],    ymm2
  vmovups [eax+32], ymm3
  vmovups [eax+64], ymm4
  vmovups [eax+96], ymm5

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

  vzeroupper

@SkipLargeAddLoop:
  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [eax]
  movups xmm3, [edx]
  subps xmm2, xmm3
  movups [eax], xmm2

  add eax, 16
  add edx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end
  [
    'EAX', 'ECX', 'EDX',
    'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];
  end;

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] -= PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] -= PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] -= PtrB^[localNumElements+2];
    end;
  end;
end;

function AVXGetSum(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
   if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  vxorps ymm0, ymm0, ymm0

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
  vxorps ymm1, ymm1, ymm1
  vxorps ymm2, ymm2, ymm2
  vxorps ymm3, ymm3, ymm3
@LargeAddLoop:

  vaddps  ymm0, ymm0, [eax]
  vaddps  ymm1, ymm1, [eax+32]
  vaddps  ymm2, ymm2, [eax+64]
  vaddps  ymm3, ymm3, [eax+96]

  add eax, 128
  dec ecx
  jnz @LargeAddLoop

  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm1
  vaddps ymm0, ymm0, ymm2
  VEXTRACTF128 xmm2, ymm0, 1

  vzeroupper
  addps xmm0, xmm2

@SkipLargeAddLoop:
  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [eax]
  addps xmm0, xmm2

  add eax, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'EAX', 'ECX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7'
  ];

    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += PtrA^[localNumElements]
    else if MissedElements = 2
    then Result +=
           PtrA^[localNumElements] +
           PtrA^[localNumElements+1]
    else Result +=
           PtrA^[localNumElements] +
           PtrA^[localNumElements+1] +
           PtrA^[localNumElements+2] ;
  end;
end;

function AVXGetSumSqr(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  vxorps ymm0, ymm0, ymm0

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
  vxorps ymm1, ymm1, ymm1
  vxorps ymm2, ymm2, ymm2
  vxorps ymm3, ymm3, ymm3
@LargeAddLoop:

  vmovups ymm4, [eax]
  vmovups ymm5, [eax+32]
  vmovups ymm6, [eax+64]
  vmovups ymm7, [eax+96]

  {$IFDEF AVX2}
  vfmadd231ps ymm0, ymm4, ymm4
  vfmadd231ps ymm1, ymm5, ymm5
  vfmadd231ps ymm2, ymm6, ymm6
  vfmadd231ps ymm3, ymm7, ymm7
  {$ELSE}
  vmulps  ymm4, ymm4, ymm4
  vmulps  ymm5, ymm5, ymm5
  vmulps  ymm6, ymm6, ymm6
  vmulps  ymm7, ymm7, ymm7

  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm1, ymm1, ymm5
  vaddps  ymm2, ymm2, ymm6
  vaddps  ymm3, ymm3, ymm7
  {$ENDIF}

  add eax, 128
  dec ecx
  jnz @LargeAddLoop

  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm1
  vaddps ymm0, ymm0, ymm2
  VEXTRACTF128 xmm2, ymm0, 1

  vzeroupper
  addps xmm0, xmm2

@SkipLargeAddLoop:
  pop ecx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [eax]
  mulps xmm2, xmm2
  addps xmm0, xmm2

  add eax, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'EAX', 'ECX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7'
  ];

    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += Sqr(PtrA^[localNumElements])
    else if MissedElements = 2
    then Result +=
           Sqr(PtrA^[localNumElements]) +
           Sqr(PtrA^[localNumElements+1])
    else Result +=
           Sqr(PtrA^[localNumElements]) +
           Sqr(PtrA^[localNumElements+1]) +
           Sqr(PtrA^[localNumElements+2]);
  end;
end;

{ AVXExp (32-bit): dst[0..N-1] := exp(src[0..N-1]). 8-wide AVX2 body using only
  ymm0..ymm7 (no extended regs in 32-bit), scalar NeuralExp remainder. Under
  plain-AVX it degrades to a scalar NeuralExp loop. }
procedure AVXExp(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer);
{$IFDEF AVX2}
var
  localNumElements, MissedElements, I, NumElementsM1: integer;
begin
  MissedElements := NumElements and 7;
  localNumElements := NumElements xor MissedElements;
  NumElementsM1 := NumElements - 1;
  if localNumElements > 0 then
  begin
  asm
  mov eax, pSrc
  mov ecx, pDst
  mov edx, localNumElements
  shr edx, 3
  jz @DoneAVXExp32
@LoopAVXExp32:
  vmovups ymm0, [eax]
  vbroadcastss ymm6, dword ptr [cAVXExpHi]
  vminps  ymm0, ymm0, ymm6
  vbroadcastss ymm6, dword ptr [cAVXExpLo]
  vmaxps  ymm0, ymm0, ymm6
  vbroadcastss ymm6, dword ptr [cAVXLog2e]
  vmulps  ymm1, ymm0, ymm6          // t = x*log2e
  vroundps ymm2, ymm1, 0            // k = round(t)
  vsubps  ymm1, ymm1, ymm2          // f = t-k
  vbroadcastss ymm6, dword ptr [cAVXLn2]
  vmulps  ymm3, ymm1, ymm6          // g = f*ln2
  vbroadcastss ymm4, dword ptr [cAVXExpP6]
  vbroadcastss ymm5, dword ptr [cAVXExpP5]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, dword ptr [cAVXExpP4]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, dword ptr [cAVXExpP3]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, dword ptr [cAVXExpP2]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, dword ptr [cAVXExpP1]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, dword ptr [cAVXExpP0]
  vfmadd213ps ymm4, ymm3, ymm5      // ymm4 = 2^f
  vcvtps2dq ymm2, ymm2              // k -> int32
  vbroadcastss ymm6, dword ptr [cAVXExp127]
  vpaddd ymm2, ymm2, ymm6
  vpslld ymm2, ymm2, 23            // 2^k as float bits
  vmulps ymm0, ymm4, ymm2
  vmovups [ecx], ymm0
  add eax, 32
  add ecx, 32
  dec edx
  jnz @LoopAVXExp32
@DoneAVXExp32:
  vzeroupper
  end ['eax','ecx','edx',
       'ymm0','ymm1','ymm2','ymm3','ymm4','ymm5','ymm6'];
  end;
  for I := localNumElements to NumElementsM1 do
    pDst^[I] := NeuralExp(pSrc^[I]);
end;
{$ELSE}
var
  I, NumElementsM1: integer;
begin
  NumElementsM1 := NumElements - 1;
  for I := 0 to NumElementsM1 do
    pDst^[I] := NeuralExp(pSrc^[I]);
end;
{$ENDIF}

{ AVXLn (32-bit): scalar pcr_logf loop. The Cephes log bit-tricks need many ymm
  registers (only ymm0..7 are usable in 32-bit asm), so the 32-bit build falls back
  to the RTL while the 64-bit build provides the 8-wide vectorized AVXLn. }
procedure AVXLn(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer);
var
  I, NumElementsM1: integer;
begin
  NumElementsM1 := NumElements - 1;
  for I := 0 to NumElementsM1 do
    pDst^[I] := pcr_logf(pSrc^[I]);
end;

{ AVXSinCos (32-bit): scalar RTL loop (see AVXLn note on the register pressure). }
procedure AVXSinCos(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer; DoCos: boolean);
var
  I, NumElementsM1: integer;
begin
  NumElementsM1 := NumElements - 1;
  if DoCos then
    for I := 0 to NumElementsM1 do
      pDst^[I] := pcr_cosf(pSrc^[I])
  else
    for I := 0 to NumElementsM1 do
      pDst^[I] := pcr_sinf(pSrc^[I]);
end;

function AVXDotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov eax, PtrA
  mov edx, PtrB
  vxorps ymm0, ymm0, ymm0

  push ecx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
  vxorps ymm1, ymm1, ymm1
  vxorps ymm2, ymm2, ymm2
  vxorps ymm3, ymm3, ymm3
@LargeAddLoop:

  vmovups ymm4, [eax]
  vmovups ymm5, [eax+32]
  vmovups ymm6, [eax+64]
  vmovups ymm7, [eax+96]

  {$IFDEF AVX2}
  vfmadd231ps ymm0, ymm4, [edx]
  vfmadd231ps ymm1, ymm5, [edx+32]
  vfmadd231ps ymm2, ymm6, [edx+64]
  vfmadd231ps ymm3, ymm7, [edx+96]
  {$ELSE}
  vmulps  ymm4, ymm4, [edx]
  vmulps  ymm5, ymm5, [edx+32]
  vmulps  ymm6, ymm6, [edx+64]
  vmulps  ymm7, ymm7, [edx+96]

  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm1, ymm1, ymm5
  vaddps  ymm2, ymm2, ymm6
  vaddps  ymm3, ymm3, ymm7
  {$ENDIF}

  add eax, 128
  add edx, 128
  dec ecx
  jnz @LargeAddLoop

  vaddps ymm2, ymm2, ymm3
  vaddps ymm0, ymm0, ymm1
  vaddps ymm0, ymm0, ymm2
  VEXTRACTF128 xmm2, ymm0, 1

  vzeroupper
  addps xmm0, xmm2

@SkipLargeAddLoop:
  pop ecx
  and ecx,$0000001F
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
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7'
  ];

    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += PtrA^[localNumElements] * PtrB^[localNumElements]
    else if MissedElements = 2
    then Result +=
           PtrA^[localNumElements] * PtrB^[localNumElements] +
           PtrA^[localNumElements+1] * PtrB^[localNumElements+1]
    else Result +=
           PtrA^[localNumElements] * PtrB^[localNumElements] +
           PtrA^[localNumElements+1] * PtrB^[localNumElements+1] +
           PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
  end;
end;
{$ENDIF}

{$IFDEF AVX64}
procedure AVXFill(PtrA: TNeuralFloatArrPtr; FillOp: TNeuralFloat; NumElements: integer);
var
  FillOpPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    FillOpPtr := Addr(FillOp);
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, FillOpPtr

  {$IFDEF AVX512}
  VBROADCASTSS zmm0, [rdx]
  {$ELSE}
  VBROADCASTSS ymm0, [rdx]
  {$ENDIF}

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:

  {$IFDEF AVX512}
  vmovups [rax],    zmm0
  vmovups [rax+64], zmm0
  {$ELSE}
  vmovups [rax],    ymm0
  vmovups [rax+32], ymm0
  vmovups [rax+64], ymm0
  vmovups [rax+96], ymm0
  {$ENDIF}

  add rax, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop rcx
  and ecx,$0000001F

  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups [rax], xmm0

  add rax, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'RAX', 'RCX', 'RDX'
    {$IFDEF AVX512} ,'ymm0', 'zmm0' {$ENDIF}
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := FillOp;
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := FillOp;
      if MissedElements>2 then PtrA^[localNumElements+2] := FillOp;
    end;
  end;
end;

procedure AVXMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; MulOp: TNeuralFloat; NumElements: integer);  overload;
var
  MulOpPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr := Addr(MulOp);
  asm
  mov ecx, localNumElements
  mov rax, PtrB
  mov rdx, MulOpPtr

  {$IFDEF AVX512}
  VBROADCASTSS zmm5, [rdx]
  {$ELSE}
  VBROADCASTSS ymm5, [rdx]
  {$ENDIF}

  mov rdx, PtrA

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:
  {$IFDEF AVX512}
  vmulps  zmm0, zmm5, [rax]
  vmulps  zmm1, zmm5, [rax+64]

  vaddps  zmm0, zmm0, [rdx]
  vaddps  zmm1, zmm1, [rdx+64]

  vmovups [rdx],    zmm0
  vmovups [rdx+64], zmm1
  {$ELSE}
    {$IFDEF AVX2}
    vmovups ymm0, [rdx]
    vmovups ymm1, [rdx+32]
    vmovups ymm2, [rdx+64]
    vmovups ymm3, [rdx+96]

    vfmadd231ps ymm0, ymm5, [rax]
    vfmadd231ps ymm1, ymm5, [rax+32]
    vfmadd231ps ymm2, ymm5, [rax+64]
    vfmadd231ps ymm3, ymm5, [rax+96]
    {$ELSE}
    vmulps  ymm0, ymm5, [rax]
    vmulps  ymm1, ymm5, [rax+32]
    vmulps  ymm2, ymm5, [rax+64]
    vmulps  ymm3, ymm5, [rax+96]

    vaddps  ymm0, ymm0, [rdx]
    vaddps  ymm1, ymm1, [rdx+32]
    vaddps  ymm2, ymm2, [rdx+64]
    vaddps  ymm3, ymm3, [rdx+96]
    {$ENDIF}

    vmovups [rdx],    ymm0
    vmovups [rdx+32], ymm1
    vmovups [rdx+64], ymm2
    vmovups [rdx+96], ymm3
  {$ENDIF}

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [rax]
  movups  xmm4, [rdx]

  mulps   xmm2, xmm5
  addps   xmm4, xmm2

  movups  [rdx], xmm4

  add rax, 16
  add rdx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm0','zmm1','zmm5'{$ENDIF}
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] += MulOp*PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] += MulOp*PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] += MulOp*PtrB^[localNumElements+2];
    end;
  end;
end;

procedure AVXMulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr; NumElements: integer);  overload;
var
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  asm_avx64_mulladd_ptra_ptrb_ptrc_num;
end;

procedure AVXCopyRelu(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
var
  ZeroVar: TNeuralFloat;
  ZeroVarPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  ZeroVar := 0;
  if localNumElements > 0 then
  begin
    ZeroVarPtr := Addr(ZeroVar);
  asm
  mov ecx, localNumElements
  mov rax, PtrB
  mov rdx, ZeroVarPtr

  VBROADCASTSS ymm5, [rdx]

  mov rdx, PtrA

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:
    VMAXPS ymm0, ymm5, [rax]
    VMAXPS ymm1, ymm5, [rax+32]
    VMAXPS ymm2, ymm5, [rax+64]
    VMAXPS ymm3, ymm5, [rax+96]

    vmovups [rdx],    ymm0
    vmovups [rdx+32], ymm1
    vmovups [rdx+64], ymm2
    vmovups [rdx+96], ymm3

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [rax]
  MAXPS   xmm2, xmm5

  movups  [rdx], xmm2

  add rax, 16
  add rdx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm0','zmm1','zmm5'{$ENDIF}
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := Max(0,PtrB^[localNumElements]);
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := Max(0,PtrB^[localNumElements+1]);
      if MissedElements>2 then PtrA^[localNumElements+2] := Max(0,PtrB^[localNumElements+2]);
    end;
  end;
end;


// PtrA := PtrA * MulOp1 + PtrB * MulOp2
// RDX  := RDX  * ymm5   + RAX  * ymm4
procedure AVXMulMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; MulOp1, MulOp2: TNeuralFloat; NumElements: integer);
var
  MulOpPtr1, MulOpPtr2: pointer;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr1 := Addr(MulOp1);
    MulOpPtr2 := Addr(MulOp2);
  asm
  mov ecx, localNumElements
  mov rax, PtrB

  mov rdx, MulOpPtr1
  {$IFDEF AVX512}
  VBROADCASTSS zmm5, [rdx]
  {$ELSE}
  VBROADCASTSS ymm5, [rdx]
  {$ENDIF}

  mov rdx, MulOpPtr2
  {$IFDEF AVX512}
  VBROADCASTSS zmm4, [rdx]
  {$ELSE}
  VBROADCASTSS ymm4, [rdx]
  {$ENDIF}

  mov rdx, PtrA

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:

  {$IFDEF AVX512}
  vmulps  zmm0, zmm4, [rax]
  vmulps  zmm1, zmm4, [rax+64]

  vmulps  zmm2, zmm5, [rdx]
  vmulps  zmm3, zmm5, [rdx+64]

  vaddps  zmm0, zmm0, zmm2
  vaddps  zmm1, zmm1, zmm3

  vmovups [rdx],    zmm0
  vmovups [rdx+64], zmm1
  {$ELSE}
  vmulps  ymm0, ymm4, [rax]
  vmulps  ymm1, ymm4, [rax+32]

  vmulps  ymm2, ymm5, [rdx]
  vmulps  ymm3, ymm5, [rdx+32]

  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm1, ymm1, ymm3

  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1

  vmulps  ymm0, ymm4, [rax+64]
  vmulps  ymm1, ymm4, [rax+96]

  vmulps  ymm2, ymm5, [rdx+64]
  vmulps  ymm3, ymm5, [rdx+96]

  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm1, ymm1, ymm3

  vmovups [rdx+64], ymm0
  vmovups [rdx+96], ymm1
  {$ENDIF}

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [rax]
  movups  xmm1, [rdx]

  mulps   xmm2, xmm4
  mulps   xmm1, xmm5

  addps   xmm1, xmm2
  movups  [rdx], xmm1

  add rax, 16
  add rdx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm0', 'zmm1', 'zmm2', 'zmm3', 'zmm4', 'zmm5'{$ENDIF}
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements]*MulOp1 + MulOp2*PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1]*MulOp1 + MulOp2*PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2]*MulOp1 + MulOp2*PtrB^[localNumElements+2];
    end;
  end;
end;

procedure AVXMul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat; NumElements: integer); overload;
var
  MulOpPtr: pointer;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    MulOpPtr := Addr(MulOp);
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, MulOpPtr

  {$IFDEF AVX512}
  VBROADCASTSS zmm0, [rdx]
  {$ELSE}
  VBROADCASTSS ymm0, [rdx]
  {$ENDIF}

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:

  {$IFDEF AVX512}
  vmulps  zmm2, zmm0, [rax]
  vmulps  zmm3, zmm0, [rax+64]

  vmovups [rax],    zmm2
  vmovups [rax+64], zmm3
  {$ELSE}
  vmulps  ymm2, ymm0, [rax]
  vmulps  ymm3, ymm0, [rax+32]
  vmulps  ymm4, ymm0, [rax+64]
  vmulps  ymm5, ymm0, [rax+96]

  vmovups [rax],    ymm2
  vmovups [rax+32], ymm3
  vmovups [rax+64], ymm4
  vmovups [rax+96], ymm5
  {$ENDIF}

  add rax, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4

@SmallAddLoop:

  movups  xmm2, [rax]
  mulps   xmm2, xmm0
  movups [rax], xmm2

  add rax, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'RAX', 'RCX', 'RDX',
    'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm0'
    {$IFDEF AVX512},'zmm2', 'zmm3', 'zmm0'{$ENDIF}
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] *= MulOp;
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] *= MulOp;
      if MissedElements>2 then PtrA^[localNumElements+2] *= MulOp;
    end;
  end;
end;

procedure AVXMul(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer); overload;
var
  MulOpPtr1, MulOpPtr2: pointer;
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrB
  mov rdx, PtrA

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

@LargeAddLoop:

  {$IFDEF AVX512}
  vmovups  zmm0, [rax]
  vmovups  zmm1, [rax+64]

  vmulps  zmm0, zmm0, [rdx]
  vmulps  zmm1, zmm1, [rdx+64]

  vmovups [rdx],    zmm0
  vmovups [rdx+64], zmm1
  {$ELSE}
  vmovups  ymm0, [rax]
  vmovups  ymm1, [rax+32]
  vmovups  ymm2, [rax+64]
  vmovups  ymm3, [rax+96]

  vmulps  ymm0, ymm0, [rdx]
  vmulps  ymm1, ymm1, [rdx+32]
  vmulps  ymm2, ymm2, [rdx+64]
  vmulps  ymm3, ymm3, [rdx+96]

  vmovups [rdx],    ymm0
  vmovups [rdx+32], ymm1
  vmovups [rdx+64], ymm2
  vmovups [rdx+96], ymm3
  {$ENDIF}

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

@SkipLargeAddLoop:
  vzeroupper

  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations

@SmallAddLoop:

  movups  xmm2, [rax]
  movups  xmm4, [rdx]

  mulps   xmm2, xmm4
  movups  [rdx], xmm2

  add rax, 16
  add rdx, 16

  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3'
    {$IFDEF AVX512},'zmm0', 'zmm1'{$ENDIF}
  ];
  end; // of if

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] := PtrA^[localNumElements] * PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] := PtrA^[localNumElements+1] * PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] := PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
    end;
  end;
end;

procedure AVXAdd(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
var
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
@LargeAddLoop:

  {$IFDEF AVX512}
  vmovups zmm2, [rax]
  vmovups zmm3, [rax+64]

  vaddps  zmm2, zmm2, [rdx]
  vaddps  zmm3, zmm3, [rdx+64]

  vmovups [rax],    zmm2
  vmovups [rax+64], zmm3
  {$ELSE}
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]

  vaddps  ymm2, ymm2, [rdx]
  vaddps  ymm3, ymm3, [rdx+32]
  vaddps  ymm4, ymm4, [rdx+64]
  vaddps  ymm5, ymm5, [rdx+96]

  vmovups [rax],    ymm2
  vmovups [rax+32], ymm3
  vmovups [rax+64], ymm4
  vmovups [rax+96], ymm5
  {$ENDIF}

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

  vzeroupper

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [rax]
  movups xmm3, [rdx]
  addps xmm2, xmm3
  movups [rax], xmm2

  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end
  [
    'RAX', 'RCX', 'RDX',
    'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm2', 'zmm3'{$ENDIF}
  ];
  end;

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] += PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] += PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] += PtrB^[localNumElements+2];
    end;
  end;
end;

procedure AVXMax(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
var
  localNumElements, MissedElements: integer;
begin
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeMaxLoop
@LargeMaxLoop:

  {$IFDEF AVX512}
  vmovups zmm2, [rax]
  vmovups zmm3, [rax+64]

  vmaxps  zmm2, zmm2, [rdx]
  vmaxps  zmm3, zmm3, [rdx+64]

  vmovups [rax],    zmm2
  vmovups [rax+64], zmm3
  {$ELSE}
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]

  vmaxps  ymm2, ymm2, [rdx]
  vmaxps  ymm3, ymm3, [rdx+32]
  vmaxps  ymm4, ymm4, [rdx+64]
  vmaxps  ymm5, ymm5, [rdx+96]

  vmovups [rax],    ymm2
  vmovups [rax+32], ymm3
  vmovups [rax+64], ymm4
  vmovups [rax+96], ymm5
  {$ENDIF}

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeMaxLoop

  vzeroupper

@SkipLargeMaxLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndMax
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallMaxLoop:
  vzeroupper

  movups xmm2, [rax]
  movups xmm3, [rdx]
  maxps  xmm2, xmm3
  movups [rax], xmm2

  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallMaxLoop

@EndMax:
  end
  [
    'RAX', 'RCX', 'RDX',
    'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm2', 'zmm3'{$ENDIF}
  ];
  end;

  if MissedElements>0 then
  begin
    if PtrB^[localNumElements] > PtrA^[localNumElements] then
      PtrA^[localNumElements] := PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      if PtrB^[localNumElements+1] > PtrA^[localNumElements+1] then
        PtrA^[localNumElements+1] := PtrB^[localNumElements+1];
      if MissedElements>2 then
        if PtrB^[localNumElements+2] > PtrA^[localNumElements+2] then
          PtrA^[localNumElements+2] := PtrB^[localNumElements+2];
    end;
  end;
end;

function AVXSumDiff(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB

  vxorps ymm0, ymm0, ymm0

  {$IFDEF AVX2}
  VPCMPEQD  ymm1, ymm1, ymm1
  VPSRLD    ymm1, ymm1, 1
  {$ELSE}
  VPCMPEQD  xmm2, xmm2, xmm2
  VPCMPEQD  xmm3, xmm3, xmm3
  VPSRLD    xmm2, xmm2, 1
  VPSRLD    xmm3, xmm3, 1
  VPERM2F128 ymm1, ymm2, ymm3, 0
  {$ENDIF}

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
@LargeAddLoop:

  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]

  vsubps  ymm2, ymm2, [rdx]
  vsubps  ymm3, ymm3, [rdx+32]
  vsubps  ymm4, ymm4, [rdx+64]
  vsubps  ymm5, ymm5, [rdx+96]

  // absolute values
  vandps  ymm2, ymm2, ymm1
  vandps  ymm3, ymm3, ymm1
  vandps  ymm4, ymm4, ymm1
  vandps  ymm5, ymm5, ymm1

  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm0, ymm0, ymm3
  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm0, ymm0, ymm5

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [rax]
  movups xmm3, [rdx]
  subps  xmm2, xmm3
  andps  xmm2, xmm1
  addps  xmm0, xmm2

  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];
    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += Abs(PtrA^[localNumElements]-PtrB^[localNumElements])
    else if MissedElements = 2
    then Result +=
           Abs(PtrA^[localNumElements]-PtrB^[localNumElements]) +
           Abs(PtrA^[localNumElements+1]-PtrB^[localNumElements+1])
    else Result +=
           Abs(PtrA^[localNumElements]-PtrB^[localNumElements]) +
           Abs(PtrA^[localNumElements+1]-PtrB^[localNumElements+1]) +
           Abs(PtrA^[localNumElements+2]-PtrB^[localNumElements+2]);
  end;
end;

function AVXDistanceSqr(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB

  vxorps ymm0, ymm0, ymm0

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
@LargeAddLoop:

  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]

  vsubps  ymm2, ymm2, [rdx]
  vsubps  ymm3, ymm3, [rdx+32]
  vsubps  ymm4, ymm4, [rdx+64]
  vsubps  ymm5, ymm5, [rdx+96]

  vmulps  ymm2, ymm2, ymm2
  vmulps  ymm3, ymm3, ymm3
  vmulps  ymm4, ymm4, ymm4
  vmulps  ymm5, ymm5, ymm5

  vaddps  ymm0, ymm0, ymm2
  vaddps  ymm0, ymm0, ymm3
  vaddps  ymm0, ymm0, ymm4
  vaddps  ymm0, ymm0, ymm5

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [rax]
  movups xmm3, [rdx]
  subps  xmm2, xmm3
  mulps  xmm2, xmm2
  addps  xmm0, xmm2

  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
  ];
    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += Sqr(PtrA^[localNumElements]-PtrB^[localNumElements])
    else if MissedElements = 2
    then Result +=
           Sqr(PtrA^[localNumElements]-PtrB^[localNumElements]) +
           Sqr(PtrA^[localNumElements+1]-PtrB^[localNumElements+1])
    else Result +=
           Sqr(PtrA^[localNumElements]-PtrB^[localNumElements]) +
           Sqr(PtrA^[localNumElements+1]-PtrB^[localNumElements+1]) +
           Sqr(PtrA^[localNumElements+2]-PtrB^[localNumElements+2]);
  end;
end;

procedure AVXSub(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer);
var
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
@LargeAddLoop:

  {$IFDEF AVX512}
  vmovups zmm2, [rax]
  vmovups zmm3, [rax+64]

  vsubps  zmm2, zmm2, [rdx]
  vsubps  zmm3, zmm3, [rdx+64]

  vmovups [rax],    zmm2
  vmovups [rax+64], zmm3
  {$ELSE}
  vmovups ymm2, [rax]
  vmovups ymm3, [rax+32]
  vmovups ymm4, [rax+64]
  vmovups ymm5, [rax+96]

  vsubps  ymm2, ymm2, [rdx]
  vsubps  ymm3, ymm3, [rdx+32]
  vsubps  ymm4, ymm4, [rdx+64]
  vsubps  ymm5, ymm5, [rdx+96]

  vmovups [rax],    ymm2
  vmovups [rax+32], ymm3
  vmovups [rax+64], ymm4
  vmovups [rax+96], ymm5
  {$ENDIF}

  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

  vzeroupper

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [rax]
  movups xmm3, [rdx]
  subps  xmm2, xmm3
  movups [rax], xmm2

  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  end
  [
    'RAX', 'RCX', 'RDX',
    'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm2', 'zmm3'{$ENDIF}
  ];
  end;

  if MissedElements>0 then
  begin
    PtrA^[localNumElements] -= PtrB^[localNumElements];
    if MissedElements>1 then
    begin
      PtrA^[localNumElements+1] -= PtrB^[localNumElements+1];
      if MissedElements>2 then PtrA^[localNumElements+2] -= PtrB^[localNumElements+2];
    end;
  end;
end;

function AVXGetSum(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA

  {$IFDEF AVX512}
  vxorps zmm0, zmm0, zmm0
  {$ELSE}
  vxorps ymm0, ymm0, ymm0
  {$ENDIF}

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop
  {$IFDEF AVX512}
  vxorps zmm1, zmm1, zmm1
  {$ELSE}
  vxorps ymm1, ymm1, ymm1
  {$ENDIF}

@LargeAddLoop:

  {$IFDEF AVX512}
  vaddps  zmm0, zmm0, [rax]
  vaddps  zmm1, zmm1, [rax+64]
  {$ELSE}
  vaddps  ymm0, ymm0, [rax]
  vaddps  ymm1, ymm1, [rax+32]
  vaddps  ymm0, ymm0, [rax+64]
  vaddps  ymm1, ymm1, [rax+96]
  {$ENDIF}

  add rax, 128
  dec ecx
  jnz @LargeAddLoop

  {$IFDEF AVX512}
  vaddps zmm0, zmm0, zmm1
  VEXTRACTF32x4 xmm2, zmm0, 1
  VEXTRACTF32x4 xmm3, zmm0, 2
  VEXTRACTF32x4 xmm4, zmm0, 3
  vzeroupper
  addps  xmm0, xmm2
  addps  xmm0, xmm3
  addps  xmm0, xmm4
  {$ELSE}
  vaddps ymm0, ymm0, ymm1
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2
  {$ENDIF}

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [rax]
  addps xmm0, xmm2

  add rax, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'RAX', 'RCX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm0', 'zmm1'{$ENDIF}
  ];

    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += PtrA^[localNumElements]
    else if MissedElements = 2
    then Result +=
           PtrA^[localNumElements] +
           PtrA^[localNumElements+1]
    else Result +=
           PtrA^[localNumElements] +
           PtrA^[localNumElements+1] +
           PtrA^[localNumElements+2];
  end;
end;

function AVXGetSumSqr(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  {$IFDEF AVX512}
  vxorps zmm0, zmm0, zmm0
  {$ELSE}
  vxorps ymm0, ymm0, ymm0
  {$ENDIF}

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

  {$IFDEF AVX512}
  vxorps zmm1, zmm1, zmm1
  {$ELSE}
  vxorps ymm1, ymm1, ymm1
  {$ENDIF}

@LargeAddLoop:

  {$IFDEF AVX512}
  vmovups zmm2, [rax]
  vmovups zmm3, [rax+64]

  vmulps  zmm2, zmm2, zmm2
  vmulps  zmm3, zmm3, zmm3

  vaddps  zmm0, zmm0, zmm2
  vaddps  Zmm1, zmm1, zmm3
  {$ELSE}
    vmovups ymm2, [rax]
    vmovups ymm3, [rax+32]
    vmovups ymm4, [rax+64]
    vmovups ymm5, [rax+96]
    {$IFDEF AVX2}
    vfmadd231ps ymm0, ymm2, ymm2
    vfmadd231ps ymm1, ymm3, ymm3
    vfmadd231ps ymm0, ymm4, ymm4
    vfmadd231ps ymm1, ymm5, ymm5
    {$ELSE}
    vmulps  ymm2, ymm2, ymm2
    vmulps  ymm3, ymm3, ymm3
    vmulps  ymm4, ymm4, ymm4
    vmulps  ymm5, ymm5, ymm5

    vaddps  ymm0, ymm0, ymm2
    vaddps  ymm1, ymm1, ymm3
    vaddps  ymm0, ymm0, ymm4
    vaddps  ymm1, ymm1, ymm5
    {$ENDIF}
  {$ENDIF}

  add rax, 128
  dec ecx
  jnz @LargeAddLoop

  {$IFDEF AVX512}
  vaddps zmm0, zmm0, zmm1
  VEXTRACTF32x4 xmm2, zmm0, 1
  VEXTRACTF32x4 xmm3, zmm0, 2
  VEXTRACTF32x4 xmm4, zmm0, 3
  vzeroupper
  addps  xmm0, xmm2
  addps  xmm0, xmm3
  addps  xmm0, xmm4
  {$ELSE}
  vaddps ymm0, ymm0, ymm1
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2
  {$ENDIF}

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [rax]
  mulps xmm2, xmm2
  addps xmm0, xmm2

  add rax, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'RAX', 'RCX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm0', 'zmm1'{$ENDIF}
  ];

    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += Sqr(PtrA^[localNumElements])
    else if MissedElements = 2
    then Result +=
           Sqr(PtrA^[localNumElements]) +
           Sqr(PtrA^[localNumElements+1])
    else Result +=
           Sqr(PtrA^[localNumElements]) +
           Sqr(PtrA^[localNumElements+1]) +
           Sqr(PtrA^[localNumElements+2]);
  end;
end;

{ AVXExp: dst[0..N-1] := exp(src[0..N-1]). 8-wide AVX2 polynomial body plus a
  scalar NeuralExp remainder for the (N mod 8) tail. Under plain-AVX (no AVX2)
  the whole thing degrades to a scalar NeuralExp loop. }
procedure AVXExp(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer);
{$IFDEF AVX2}
var
  localNumElements, MissedElements, I, NumElementsM1: integer;
begin
  MissedElements := NumElements and 7;
  localNumElements := NumElements xor MissedElements;
  NumElementsM1 := NumElements - 1;
  if localNumElements > 0 then
  begin
  asm
  mov rax, pSrc
  mov rcx, pDst
  mov r8d, localNumElements
  shr r8d, 3
  jz @DoneAVXExp
  vbroadcastss ymm10, [rip+cAVXExpHi]
  vbroadcastss ymm11, [rip+cAVXExpLo]
  vbroadcastss ymm12, [rip+cAVXLog2e]
  vbroadcastss ymm13, [rip+cAVXLn2]
  vmovd xmm14, dword ptr [rip+cAVXExp127]
  vpbroadcastd ymm14, xmm14
@LoopAVXExp:
  vmovups ymm0, [rax]
  vminps  ymm0, ymm0, ymm10
  vmaxps  ymm0, ymm0, ymm11
  vmulps  ymm1, ymm0, ymm12        // t = x*log2e
  vroundps ymm2, ymm1, 0           // k = round(t)
  vsubps  ymm1, ymm1, ymm2         // f = t-k in [-0.5,0.5]
  vmulps  ymm3, ymm1, ymm13        // g = f*ln2
  vbroadcastss ymm4, [rip+cAVXExpP6]
  vbroadcastss ymm5, [rip+cAVXExpP5]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, [rip+cAVXExpP4]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, [rip+cAVXExpP3]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, [rip+cAVXExpP2]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, [rip+cAVXExpP1]
  vfmadd213ps ymm4, ymm3, ymm5
  vbroadcastss ymm5, [rip+cAVXExpP0]
  vfmadd213ps ymm4, ymm3, ymm5     // ymm4 = 2^f
  vcvtps2dq ymm2, ymm2             // k -> int32
  vpaddd ymm2, ymm2, ymm14
  vpslld ymm2, ymm2, 23            // 2^k as float bits
  vmulps ymm0, ymm4, ymm2
  vmovups [rcx], ymm0
  add rax, 32
  add rcx, 32
  dec r8d
  jnz @LoopAVXExp
@DoneAVXExp:
  vzeroupper
  end ['rax','rcx','r8',
       'ymm0','ymm1','ymm2','ymm3','ymm4','ymm5',
       'ymm10','ymm11','ymm12','ymm13','ymm14'];
  end;
  for I := localNumElements to NumElementsM1 do
    pDst^[I] := NeuralExp(pSrc^[I]);
end;
{$ELSE}
var
  I, NumElementsM1: integer;
begin
  NumElementsM1 := NumElements - 1;
  for I := 0 to NumElementsM1 do
    pDst^[I] := NeuralExp(pSrc^[I]);
end;
{$ENDIF}

{ AVXLn: dst[0..N-1] := ln(src[0..N-1]). 8-wide AVX2 Cephes logf body plus a scalar
  pcr_logf remainder for the (N mod 8) tail. Decomposes x = m*2^e with m in
  [sqrt(0.5),sqrt(2)) and evaluates ln(m) as a degree-8 polynomial in (m-1). }
procedure AVXLn(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer);
{$IFDEF AVX2}
var
  localNumElements, MissedElements, I, NumElementsM1: integer;
begin
  MissedElements := NumElements and 7;
  localNumElements := NumElements xor MissedElements;
  NumElementsM1 := NumElements - 1;
  if localNumElements > 0 then
  begin
  asm
  mov rax, pSrc
  mov rcx, pDst
  mov r8d, localNumElements
  shr r8d, 3
  jz @DoneAVXLn
@LoopAVXLn:
  vmovups ymm0, [rax]
  // clamp to smallest positive normal so denormals/zero do not poison the bit tricks
  vbroadcastss ymm15, [rip+cAVXLnMinNorm]
  vmaxps  ymm0, ymm0, ymm15
  // e = (float)(((bits >> 23) & 0xff) - 0x7f) + 1   (mantissa rescaled to [0.5,1))
  vpsrld  ymm2, ymm0, 23
  vmovd   xmm15, dword ptr [rip+cAVXExp127]
  vpbroadcastd ymm15, xmm15            // 0x7f = 127
  vpsubd  ymm2, ymm2, ymm15            // unbiased exponent
  vcvtdq2ps ymm2, ymm2
  vbroadcastss ymm15, [rip+cAVXLnOne]
  vaddps  ymm2, ymm2, ymm15            // e = exp + 1 (0.5*2^e convention)
  // mantissa in [0.5,1): bits = (bits & invMant) | 0.5bits
  vbroadcastss ymm15, [rip+cAVXLnInvMant]
  vandps  ymm0, ymm0, ymm15
  vbroadcastss ymm15, [rip+cAVXLnHalf]
  vorps   ymm0, ymm0, ymm15            // x = mantissa in [0.5,1)
  // mask: m < sqrt(0.5) ?
  vbroadcastss ymm15, [rip+cAVXLnSqrtHf]
  vcmpltps ymm3, ymm0, ymm15           // mask = (x < SQRTHF)
  vandps  ymm4, ymm0, ymm3             // tmp = (x<sqrthf)? x : 0
  vbroadcastss ymm15, [rip+cAVXLnOne]
  vsubps  ymm0, ymm0, ymm15            // x = x - 1
  vaddps  ymm0, ymm0, ymm4             // if x<sqrthf: x = 2x - 1
  vandps  ymm5, ymm15, ymm3            // (x<sqrthf)? 1.0 : 0.0
  vsubps  ymm2, ymm2, ymm5             // e -= 1 where x<sqrthf
  // z = x*x
  vmulps  ymm1, ymm0, ymm0             // z
  // Horner polynomial in x: P0..P8
  vbroadcastss ymm4, [rip+cAVXLnP0]
  vbroadcastss ymm5, [rip+cAVXLnP1]
  vfmadd213ps ymm4, ymm0, ymm5
  vbroadcastss ymm5, [rip+cAVXLnP2]
  vfmadd213ps ymm4, ymm0, ymm5
  vbroadcastss ymm5, [rip+cAVXLnP3]
  vfmadd213ps ymm4, ymm0, ymm5
  vbroadcastss ymm5, [rip+cAVXLnP4]
  vfmadd213ps ymm4, ymm0, ymm5
  vbroadcastss ymm5, [rip+cAVXLnP5]
  vfmadd213ps ymm4, ymm0, ymm5
  vbroadcastss ymm5, [rip+cAVXLnP6]
  vfmadd213ps ymm4, ymm0, ymm5
  vbroadcastss ymm5, [rip+cAVXLnP7]
  vfmadd213ps ymm4, ymm0, ymm5
  vbroadcastss ymm5, [rip+cAVXLnP8]
  vfmadd213ps ymm4, ymm0, ymm5         // ymm4 = poly
  vmulps  ymm4, ymm4, ymm0             // poly *= x
  vmulps  ymm4, ymm4, ymm1             // poly *= z   (= y)
  // y += e*Q1
  vbroadcastss ymm5, [rip+cAVXLnQ1]
  vfmadd231ps ymm4, ymm2, ymm5
  // y -= 0.5*z
  vbroadcastss ymm5, [rip+cAVXLnHalf]
  vmulps  ymm6, ymm1, ymm5
  vsubps  ymm4, ymm4, ymm6
  // x = x + y
  vaddps  ymm0, ymm0, ymm4
  // x += e*Q2
  vbroadcastss ymm5, [rip+cAVXLnQ2]
  vfmadd231ps ymm0, ymm2, ymm5
  vmovups [rcx], ymm0
  add rax, 32
  add rcx, 32
  dec r8d
  jnz @LoopAVXLn
@DoneAVXLn:
  vzeroupper
  end ['rax','rcx','r8',
       'ymm0','ymm1','ymm2','ymm3','ymm4','ymm5','ymm6','ymm15'];
  end;
  for I := localNumElements to NumElementsM1 do
    pDst^[I] := pcr_logf(pSrc^[I]);
end;
{$ELSE}
var
  I, NumElementsM1: integer;
begin
  NumElementsM1 := NumElements - 1;
  for I := 0 to NumElementsM1 do
    pDst^[I] := pcr_logf(pSrc^[I]);
end;
{$ENDIF}

{ AVXSinCos: dst[0..N-1] := sin or cos of src[0..N-1]. 8-wide AVX2 Cephes sinf/cosf
  body (3-part Cody-Waite pi/4 range reduction) plus a scalar RTL remainder. }
procedure AVXSinCos(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer; DoCos: boolean);
{$IFDEF AVX2}
var
  localNumElements, MissedElements, I, NumElementsM1: integer;
begin
  MissedElements := NumElements and 7;
  localNumElements := NumElements xor MissedElements;
  NumElementsM1 := NumElements - 1;
  if localNumElements > 0 then
  begin
  if DoCos then
  begin
  asm
  mov rax, pSrc
  mov rcx, pDst
  mov r8d, localNumElements
  shr r8d, 3
  jz @DoneAVXCos
@LoopAVXCos:
  vmovups ymm0, [rax]               // x
  vpcmpeqd ymm14, ymm14, ymm14
  vpsrld  ymm14, ymm14, 1           // 0x7fffffff
  vandps  ymm1, ymm0, ymm14         // |x|
  vbroadcastss ymm15, [rip+cAVXSC_FOPI]
  vmulps  ymm2, ymm1, ymm15
  vcvttps2dq ymm3, ymm2             // j = trunc(|x|*4/pi)
  vmovd   xmm15, dword ptr [rip+cAVXSC_1i]
  vpbroadcastd ymm15, xmm15
  vpaddd  ymm3, ymm3, ymm15         // j+1
  vmovd   xmm15, dword ptr [rip+cAVXSC_NOT1i]
  vpbroadcastd ymm15, xmm15
  vpand   ymm3, ymm3, ymm15         // j &= ~1
  vcvtdq2ps ymm2, ymm3              // y = (float)j
  vbroadcastss ymm15, [rip+cAVXSC_DP1]
  vfmadd231ps ymm1, ymm2, ymm15
  vbroadcastss ymm15, [rip+cAVXSC_DP2]
  vfmadd231ps ymm1, ymm2, ymm15
  vbroadcastss ymm15, [rip+cAVXSC_DP3]
  vfmadd231ps ymm1, ymm2, ymm15     // reduced x
  vmovd   xmm15, dword ptr [rip+cAVXSC_2i]
  vpbroadcastd ymm15, xmm15
  vpsubd  ymm4, ymm3, ymm15         // m = j-2
  vmovd   xmm15, dword ptr [rip+cAVXSC_4i]
  vpbroadcastd ymm15, xmm15
  vpandn  ymm5, ymm4, ymm15         // (~m)&4   (Cephes cos sign convention)
  vpslld  ymm5, ymm5, 29            // sign = ((~m)&4)<<29
  vmovd   xmm15, dword ptr [rip+cAVXSC_2i]
  vpbroadcastd ymm15, xmm15
  vpand   ymm6, ymm4, ymm15
  vpxor   ymm15, ymm15, ymm15
  vpcmpeqd ymm6, ymm6, ymm15        // polymask: (m&2)==0 -> sin poly (Cephes cos)
  vmulps  ymm7, ymm1, ymm1          // z
  vbroadcastss ymm8,  [rip+cAVXSC_CosP0]
  vbroadcastss ymm9,  [rip+cAVXSC_CosP1]
  vfmadd213ps ymm8, ymm7, ymm9
  vbroadcastss ymm9,  [rip+cAVXSC_CosP2]
  vfmadd213ps ymm8, ymm7, ymm9
  vmulps  ymm8, ymm8, ymm7
  vmulps  ymm8, ymm8, ymm7
  vbroadcastss ymm9,  [rip+cAVXSC_Half]
  vmulps  ymm10, ymm7, ymm9
  vsubps  ymm8, ymm8, ymm10
  vbroadcastss ymm9,  [rip+cAVXSC_One]
  vaddps  ymm8, ymm8, ymm9          // cos candidate
  vbroadcastss ymm11, [rip+cAVXSC_SinP0]
  vbroadcastss ymm12, [rip+cAVXSC_SinP1]
  vfmadd213ps ymm11, ymm7, ymm12
  vbroadcastss ymm12, [rip+cAVXSC_SinP2]
  vfmadd213ps ymm11, ymm7, ymm12
  vmulps  ymm11, ymm11, ymm7
  vmulps  ymm11, ymm11, ymm1
  vaddps  ymm11, ymm11, ymm1        // sin candidate
  vblendvps ymm0, ymm8, ymm11, ymm6 // (m&2)? sin : cos
  vxorps  ymm0, ymm0, ymm5          // sign
  vmovups [rcx], ymm0
  add rax, 32
  add rcx, 32
  dec r8d
  jnz @LoopAVXCos
@DoneAVXCos:
  vzeroupper
  end ['rax','rcx','r8',
       'ymm0','ymm1','ymm2','ymm3','ymm4','ymm5','ymm6','ymm7','ymm8',
       'ymm9','ymm10','ymm11','ymm12','ymm14','ymm15'];
  end
  else
  begin
  asm
  mov rax, pSrc
  mov rcx, pDst
  mov r8d, localNumElements
  shr r8d, 3
  jz @DoneAVXSin
@LoopAVXSin:
  vmovups ymm0, [rax]               // x
  vpcmpeqd ymm14, ymm14, ymm14
  vpslld  ymm13, ymm14, 31          // 0x80000000
  vandps  ymm5, ymm0, ymm13         // sign_x
  vpsrld  ymm14, ymm14, 1           // 0x7fffffff
  vandps  ymm1, ymm0, ymm14         // |x|
  vbroadcastss ymm15, [rip+cAVXSC_FOPI]
  vmulps  ymm2, ymm1, ymm15
  vcvttps2dq ymm3, ymm2             // j
  vmovd   xmm15, dword ptr [rip+cAVXSC_1i]
  vpbroadcastd ymm15, xmm15
  vpaddd  ymm3, ymm3, ymm15
  vmovd   xmm15, dword ptr [rip+cAVXSC_NOT1i]
  vpbroadcastd ymm15, xmm15
  vpand   ymm3, ymm3, ymm15         // j = (j+1)&~1
  vcvtdq2ps ymm2, ymm3              // y
  vbroadcastss ymm15, [rip+cAVXSC_DP1]
  vfmadd231ps ymm1, ymm2, ymm15
  vbroadcastss ymm15, [rip+cAVXSC_DP2]
  vfmadd231ps ymm1, ymm2, ymm15
  vbroadcastss ymm15, [rip+cAVXSC_DP3]
  vfmadd231ps ymm1, ymm2, ymm15     // reduced x
  vmovd   xmm15, dword ptr [rip+cAVXSC_4i]
  vpbroadcastd ymm15, xmm15
  vpand   ymm4, ymm3, ymm15
  vpslld  ymm4, ymm4, 29            // (j&4)<<29
  vxorps  ymm5, ymm5, ymm4          // combined sign
  vmovd   xmm15, dword ptr [rip+cAVXSC_2i]
  vpbroadcastd ymm15, xmm15
  vpand   ymm6, ymm3, ymm15
  vpcmpeqd ymm6, ymm6, ymm15        // polymask: (j&2)==2 -> cos poly
  vmulps  ymm7, ymm1, ymm1          // z
  vbroadcastss ymm8,  [rip+cAVXSC_CosP0]
  vbroadcastss ymm9,  [rip+cAVXSC_CosP1]
  vfmadd213ps ymm8, ymm7, ymm9
  vbroadcastss ymm9,  [rip+cAVXSC_CosP2]
  vfmadd213ps ymm8, ymm7, ymm9
  vmulps  ymm8, ymm8, ymm7
  vmulps  ymm8, ymm8, ymm7
  vbroadcastss ymm9,  [rip+cAVXSC_Half]
  vmulps  ymm10, ymm7, ymm9
  vsubps  ymm8, ymm8, ymm10
  vbroadcastss ymm9,  [rip+cAVXSC_One]
  vaddps  ymm8, ymm8, ymm9          // cos candidate
  vbroadcastss ymm11, [rip+cAVXSC_SinP0]
  vbroadcastss ymm12, [rip+cAVXSC_SinP1]
  vfmadd213ps ymm11, ymm7, ymm12
  vbroadcastss ymm12, [rip+cAVXSC_SinP2]
  vfmadd213ps ymm11, ymm7, ymm12
  vmulps  ymm11, ymm11, ymm7
  vmulps  ymm11, ymm11, ymm1
  vaddps  ymm11, ymm11, ymm1        // sin candidate
  vblendvps ymm0, ymm11, ymm8, ymm6 // (j&2)? cos : sin
  vxorps  ymm0, ymm0, ymm5          // sign
  vmovups [rcx], ymm0
  add rax, 32
  add rcx, 32
  dec r8d
  jnz @LoopAVXSin
@DoneAVXSin:
  vzeroupper
  end ['rax','rcx','r8',
       'ymm0','ymm1','ymm2','ymm3','ymm4','ymm5','ymm6','ymm7','ymm8',
       'ymm9','ymm10','ymm11','ymm12','ymm13','ymm14','ymm15'];
  end;
  end;
  if DoCos then
    for I := localNumElements to NumElementsM1 do
      pDst^[I] := pcr_cosf(pSrc^[I])
  else
    for I := localNumElements to NumElementsM1 do
      pDst^[I] := pcr_sinf(pSrc^[I]);
end;
{$ELSE}
var
  I, NumElementsM1: integer;
begin
  NumElementsM1 := NumElements - 1;
  if DoCos then
    for I := 0 to NumElementsM1 do
      pDst^[I] := pcr_cosf(pSrc^[I])
  else
    for I := 0 to NumElementsM1 do
      pDst^[I] := pcr_sinf(pSrc^[I]);
end;
{$ENDIF}

function AVXDotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single;
var
  vRes: array[0..3] of Single;
  localNumElements, MissedElements: integer;
begin
  //localNumElements := (NumElements div 4) * 4;
  //MissedElements := NumElements - localNumElements;
  MissedElements := NumElements and 3;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
  asm
  mov ecx, localNumElements
  mov rax, PtrA
  mov rdx, PtrB
  {$IFDEF AVX512}
  vxorps zmm0, zmm0, zmm0
  {$ELSE}
  vxorps ymm0, ymm0, ymm0
  {$ENDIF}

  push rcx
  shr ecx,5  // number of large iterations = number of elements / 32
  jz @SkipLargeAddLoop

  {$IFDEF AVX512}
  vxorps zmm1, zmm1, zmm1
  {$ELSE}
  vxorps ymm1, ymm1, ymm1
  {$ENDIF}

@LargeAddLoop:

  {$IFDEF AVX512}
  vmovups zmm2, [rax]
  vmovups zmm3, [rax+64]

  vmulps  zmm2, zmm2, [rdx]
  vmulps  zmm3, zmm3, [rdx+64]

  vaddps  zmm0, zmm0, zmm2
  vaddps  zmm1, zmm1, zmm3
  {$ELSE}
    vmovups ymm2, [rax]
    vmovups ymm3, [rax+32]
    vmovups ymm4, [rax+64]
    vmovups ymm5, [rax+96]

    {$IFDEF AVX2}
    vfmadd231ps ymm0, ymm2, [rdx]
    vfmadd231ps ymm1, ymm3, [rdx+32]
    vfmadd231ps ymm0, ymm4, [rdx+64]
    vfmadd231ps ymm1, ymm5, [rdx+96]
    {$ELSE}
    vmulps  ymm2, ymm2, [rdx]
    vmulps  ymm3, ymm3, [rdx+32]
    vmulps  ymm4, ymm4, [rdx+64]
    vmulps  ymm5, ymm5, [rdx+96]

    vaddps  ymm0, ymm0, ymm2
    vaddps  ymm1, ymm1, ymm3
    vaddps  ymm0, ymm0, ymm4
    vaddps  ymm1, ymm1, ymm5
    {$ENDIF}
  {$ENDIF}


  add rax, 128
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

  {$IFDEF AVX512}
  vaddps zmm0, zmm0, zmm1
  VEXTRACTF32x4 xmm2, zmm0, 1
  VEXTRACTF32x4 xmm3, zmm0, 2
  VEXTRACTF32x4 xmm4, zmm0, 3
  vzeroupper
  addps  xmm0, xmm2
  addps  xmm0, xmm3
  addps  xmm0, xmm4
  {$ELSE}
  vaddps ymm0, ymm0, ymm1
  VEXTRACTF128 xmm2, ymm0, 1
  vzeroupper
  addps  xmm0, xmm2
  {$ENDIF}

@SkipLargeAddLoop:
  pop rcx
  and ecx,$0000001F
  jz @EndAdd
  shr ecx, 2 // number of small iterations = (number of elements modulo 16) / 4
@SmallAddLoop:
  vzeroupper

  movups xmm2, [rax]
  movups xmm3, [rdx]
  mulps xmm2, xmm3
  addps xmm0, xmm2

  add rax, 16
  add rdx, 16
  dec ecx
  jnz @SmallAddLoop

@EndAdd:
  vzeroupper
  // Sums all elements of xmm0 into the first position
  HADDPS xmm0,xmm0
  HADDPS xmm0,xmm0

  movups vRes, xmm0
  end
  [
    'RAX', 'RCX', 'RDX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5'
    {$IFDEF AVX512},'zmm0', 'zmm1'{$ENDIF}
  ];

    Result := vRes[0];
  end else
  begin
    Result := 0;
  end;

  if MissedElements>0 then
  begin
    if MissedElements = 1
    then Result += PtrA^[localNumElements] * PtrB^[localNumElements]
    else if MissedElements = 2
    then Result +=
           PtrA^[localNumElements] * PtrB^[localNumElements] +
           PtrA^[localNumElements+1] * PtrB^[localNumElements+1]
    else Result +=
           PtrA^[localNumElements] * PtrB^[localNumElements] +
           PtrA^[localNumElements+1] * PtrB^[localNumElements+1] +
           PtrA^[localNumElements+2] * PtrB^[localNumElements+2];
  end;
end;
{$ENDIF}

{$IFDEF AVXANY}
procedure TNNetVolume.Fill(c: Single);
begin
  AVXFill(FDataPtr, c, FSize);
end;

function TNNetVolume.DotProduct(Original: TNNetVolume): TNeuralFloat; overload; inline;
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at DotProduct: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  if FSize >= csMinAvxSize
    then Result := AVXDotProduct(FDataPtr, Original.FDataPtr, FSize)
    else
    begin
      Result := 0;
      vHigh := High(FData);
      for I := 0 to vHigh do
        Result += FData[I] * Original.FData[I];
    end;
end;

function TNNetVolume.GetSum(): TNeuralFloat;
var
  I: integer;
  vHigh: integer;
begin
  if FSize >= csMinAvxSize
    then Result := AVXGetSum(FDataPtr, FSize)
    else
    begin
      Result := 0;
      vHigh := High(FData);
      for I := 0 to vHigh do
        Result += FData[I];
    end;
end;

function TNNetVolume.GetSumSqr(): TNeuralFloat;
begin
  if FSize >= csMinAvxSize
    then Result := AVXGetSumSqr(FDataPtr, FSize)
    else
    begin
      Result := DotProduct(Self);
    end;
end;

function TNNetVolume.GetDistanceSqr(Original: TNNetVolume): TNeuralFloat;
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at GetDistanceSqr: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Result := 0;
  if FSize >= csMinAvxSize
    then Result := AVXDistanceSqr(FDataPtr, Original.FDataPtr, FSize)
    else
    begin
      vHigh := High(FData);
      for I := 0 to vHigh do
        Result += Sqr(Original.FData[I]-FData[I]);
    end;
end;

function TNNetVolume.GetDistance(Original: TNNetVolume): TNeuralFloat;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at GetDistance: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Result := Self.GetDistanceSqr(Original);
  if Result > 0 then Result := Sqrt(Result) else Result := 0;
end;

function TNNetVolume.SumDiff(Original: TNNetVolume): TNeuralFloat;
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at SumDiff: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  Result := 0;
  if FSize >= csMinAvxSize
    then Result := AVXSumDiff(FDataPtr, Original.FDataPtr, FSize)
    else
    begin
      vHigh := High(FData);
      for I := 0 to vHigh do
        Result += Abs(Original.FData[I]-FData[I]);
    end;
end;

class function TNNetVolume.DotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer): Single; overload; inline;
var
  I: integer;
  vHigh: integer;
begin
  if NumElements >= csMinAvxSize
    then Result := AVXDotProduct(PtrA, PtrB, NumElements)
    else
    begin
      Result := 0;
      vHigh := NumElements - 1;
      for I := 0 to vHigh do
        Result += PtrA^[I] * PtrB^[I];
    end;
end;

procedure TNNetVolume.Mul(Value: Single);
var
  I: integer;
  vHigh: integer;
begin
  if FSize >= csMinAvxSize
    then AVXMul(FDataPtr, Value, FSize)
    else
    begin
      vHigh := High(FData);
      for I := 0 to vHigh do
        FData[I] *= Value;
    end;
end;

class procedure TNNetVolume.Mul(PtrA: TNeuralFloatArrPtr; MulOp: TNeuralFloat;
  pSize: integer);
begin
  AVXMul(PtrA, MulOp, pSize);
end;

class procedure TNNetVolume.Mul(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer);
begin
  AVXMul(PtrA, PtrB, pSize);
end;

class procedure TNNetVolume.MaxElements(PtrA, PtrB: TNeuralFloatArrPtr; pSize: integer);
begin
  AVXMax(PtrA, PtrB, pSize);
end;

procedure TNNetVolume.MulAdd(Value: TNeuralFloat; Original: TNNetVolume);
begin
  {$IFDEF Debug}
  if (Original.Size <> Self.Size) then
  begin
    raise Exception.Create('Sizes don''t match at MulAdd: ' +
      IntToStr(Self.Size) + ' and ' +
      IntToStr(Original.Size) +
      '.');
  end;
  {$ENDIF}
  AVXMulAdd(FDataPtr, Original.FDataPtr, Value, FSize);
end;

procedure TNNetVolume.MulAdd(Original1, Original2: TNNetVolume);
begin
  {$IFDEF Debug}
  if (Original1.Size <> Self.Size) or (Original2.Size <> Self.Size) then
  begin
    raise Exception.Create('Sizes don''t match at MulAdd: ' +
      IntToStr(Self.Size) + ', ' +
      IntToStr(Original1.Size) + ' and ' +
      IntToStr(Original2.Size) +
      '.');
  end;
  {$ENDIF}
  AVXMulAdd(FDataPtr, Original1.DataPtr, Original2.DataPtr, FSize);
end;

procedure TNNetVolume.MulMulAdd(Value1, Value2: TNeuralFloat;
  Original: TNNetVolume);
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at TNNetVolume.MulMulAdd: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + '.');
  {$ENDIF}
  AVXMulMulAdd(FDataPtr, Original.FDataPtr, Value1, Value2, FSize);
end;

procedure TNNetVolume.MulAdd(Value: TNeuralFloat; PtrB: TNeuralFloatArrPtr);
begin
  AVXMulAdd(FDataPtr, PtrB, Value, FSize);
end;

class procedure TNNetVolume.MulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value: TNeuralFloat;
  pSize: integer);
begin
  AVXMulAdd(PtrA, PtrB, Value, pSize);
end;

class procedure TNNetVolume.MulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr;
  pSize: integer);
begin
  AVXMulAdd(PtrA, PtrB, PtrC, pSize);
end;

procedure TNNetVolume.Divi(Value: Single);
begin
  Self.Mul(1/Value);
end;

procedure TNNetVolume.Copy(Original: TNNetVolume);
begin
  if Original.Size > 0 then
  begin
    if Original.Size <> Self.Size then
    begin
      Self.ReSize(Original);
    end;
    Self.CopyNoChecks(Original);
  end;
end;

procedure TNNetVolume.CopyRelu(Original: TNNetVolume);
begin
  if Original.Size <> Self.Size then
  begin
    Self.ReSize(Original);
  end;
  AVXCopyRelu(Self.FDataPtr, Original.FDataPtr, FSize);
end;

procedure TNNetVolume.Add(Original: TNNetVolume);
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at Add: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  if FSize >= csMinAvxSize
    then AVXAdd(FDataPtr, Original.FDataPtr, FSize)
    else
    begin
      vHigh := High(FData);
      for I := 0 to vHigh do
        FData[I] += Original.FData[I];
    end;
end;

class procedure TNNetVolume.Add(PtrA, PtrB: TNeuralFloatArrPtr;
  NumElements: integer);
begin
  AVXAdd(PtrA, PtrB, NumElements);
end;

procedure TNNetVolume.Sub(Original: TNNetVolume);
var
  I: integer;
  vHigh: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at Sub: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  if FSize >= csMinAvxSize
    then AVXSub(FDataPtr, Original.FDataPtr, FSize)
    else
    begin
      vHigh := High(FData);
      for I := 0 to vHigh do
        FData[I] -= Original.FData[I];
    end;
end;

procedure TNNetVolume.CopyPadding(Original: TNNetVolume; Padding: integer);
var
  CntY: integer;
  NewSizeX, NewSizeY: integer;
  MaxY: integer;
  RowSize: integer;
  SourceRawPos, DestRawPos: pointer;
begin
  NewSizeX := Original.SizeX + Padding * 2;
  NewSizeY := Original.SizeY + Padding * 2;
  MaxY := Original.SizeY - 1;
  RowSize := Original.SizeX * Original.Depth;

  Resize(NewSizeX, NewSizeY, Original.Depth);
  Fill(0);

  for CntY := 0 to MaxY do
  begin
    SourceRawPos := Original.GetRawPtr(0, CntY, 0);
    DestRawPos := GetRawPtr(Padding, CntY + Padding, 0);
    asm_dword_copy;
  end;
end;

procedure TNNetVolume.CopyPadding(Original: TNNetVolume; PaddingX, PaddingY: integer
  );
var
  CntY: integer;
  NewSizeX, NewSizeY: integer;
  MaxY: integer;
  RowSize: integer;
  SourceRawPos, DestRawPos: pointer;
begin
  NewSizeX := Original.SizeX + PaddingX * 2;
  NewSizeY := Original.SizeY + PaddingY * 2;
  MaxY := Original.SizeY - 1;
  RowSize := Original.SizeX * Original.Depth;

  Resize(NewSizeX, NewSizeY, Original.Depth);
  Fill(0);

  for CntY := 0 to MaxY do
  begin
    SourceRawPos := Original.GetRawPtr(0, CntY, 0);
    DestRawPos := GetRawPtr(PaddingX, CntY + PaddingY, 0);
    asm_dword_copy;
  end;
end;

procedure TNNetVolume.CopyNoChecks(Original: TNNetVolume);
var
  SourceRawPos, DestRawPos: pointer;
  RowSize: integer;
begin
  {$IFDEF Debug}
  if Original.Size <> Self.Size then
    raise Exception.Create('Sizes don''t match at TNNetVolume.CopyNoChecks: ' +
      IntToStr(Self.Size) + ' and ' + IntToStr(Original.Size) + ' .');
  {$ENDIF}
  RowSize := Size;
  SourceRawPos := Addr(Original.FData[0]);
  DestRawPos := Addr(FData[0]);
  asm_dword_copy;
end;

{$ENDIF} // of AVXANY

{ TNNetGroupedVolume }

destructor TNNetGroupedVolume.Destroy;
begin
  SetLength(FGrInfoArray, 0);
  inherited Destroy;
end;

class function TVolume.DotProduct(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer
  ): Single;
var
  I: integer;
  BasePos, vHigh: integer;
  {$IFDEF FPC}
  AddrA, AddrB: TNeuralFloatPtr;
  {$ENDIF}
begin
  Result := 0;
  BasePos := 0;
  vHigh := NumElements - 1;

  {$IFDEF FPC}
  AddrA := pointer(PtrA);
  AddrB := pointer(PtrB);
  while BasePos <= vHigh - 7 do
  begin
    Result := Result +
      (AddrA)^   * (AddrB)^ +
      (AddrA+1)^ * (AddrB+1)^ +
      (AddrA+2)^ * (AddrB+2)^ +
      (AddrA+3)^ * (AddrB+3)^ +
      (AddrA+4)^ * (AddrB+4)^ +
      (AddrA+5)^ * (AddrB+5)^ +
      (AddrA+6)^ * (AddrB+6)^ +
      (AddrA+7)^ * (AddrB+7)^ ;
    BasePos := BasePos + 8;
    AddrA := AddrA + 8;
    AddrB := AddrB + 8;
  end;

  while BasePos <= vHigh - 3 do
  begin
    Result := Result +
      (AddrA)^   * (AddrB)^ +
      (AddrA+1)^ * (AddrB+1)^ +
      (AddrA+2)^ * (AddrB+2)^ +
      (AddrA+3)^ * (AddrB+3)^;
    BasePos := BasePos + 4;
    AddrA := AddrA + 4;
    AddrB := AddrB + 4;
  end;
  {$ENDIF}

  if BasePos <= vHigh then for I := BasePos to vHigh do
  begin
    Result := Result + PtrA^[I] * PtrB^[I];
    //Uncomment for debugging only: WriteLn(PtrA^[I]:8:6,' # ', PtrB^[I]:8:6,' # ', Result:8:6);
  end;
  //WriteLn('Hello: ', Result);
  //ReadLn();
end;

class function TVolume.Product(PtrA: TNeuralFloatArrPtr;
  NumElements: integer): Single;
var
  I: integer;
  vHigh: integer;
begin
  Result := 1;
  vHigh := NumElements - 1;
  for I := 0 to vHigh do
    Result := Result * PtrA^[I];
end;

{$IFNDEF FPC}
{ TNNetList }
constructor TNNetList.Create(pFreeObjects: boolean);
begin
  FreeObjects := pFreeObjects;
  inherited Create;
end;

destructor TNNetList.Destroy;
var
  I, MaxIdx: integer;
begin
  if (FreeObjects and (Count>0)) then
  begin
    MaxIdx := Count - 1;
    for I := 0 to MaxIdx do
    begin
      TObject(Self[I]).Free;
    end;
  end;
  inherited;
end;

function TNNetVolumePairList.GetItem(Index: Integer): TNNetVolumePair;
begin
  Result := TNNetVolumePair(Get(Index));
end;

procedure TNNetVolumePairList.SetItem(Index: Integer; AObject: TNNetVolumePair);
begin
  Put(Index,AObject);
end;
{$ENDIF}

end.
