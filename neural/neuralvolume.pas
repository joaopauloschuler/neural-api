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
  // Unbounded-view type over 16-bit half / bfloat16 source data (never
  // allocated as such - the checkpoint readers aim it at their staging
  // buffers, mirroring how TNeuralInt8ArrPtr views int8 codes).
  // Coded by Claude (AI).
  TNeuralHalfArr = array[0..Maxint div 4] of Word;
  TNeuralHalfArrPtr = ^TNeuralHalfArr;

const
  csNeuralFloatSize = SizeOf(TNeuralFloat);
  csNeuralFloat4Size = SizeOf(TNeuralFloat4);
  csLongintSize = SizeOf(Longint);
  csIntegerSize = SizeOf(Integer);
  csShortIntSize = SizeOf(ShortInt);
  csDoubleSize = SizeOf(Double);
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
    // GetMin/GetMax/GetMaxAbs also record the flat index of the element they
    // returned in FLastPos (first occurrence wins on ties). They are virtual
    // so TNNetVolume can serve them from a vectorized kernel.
    function GetMin(): T; virtual;
    function GetMax(): T; virtual;
    function GetNonZero(): integer; {$IFDEF Release} inline; {$ENDIF}
    function GetMaxAbs(): T; virtual;
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
    function GetClass(): integer; virtual;
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

  // Forward: the int8 tiled kernels below take a TNNetVolumeQuant8 table
  // (declared after TNNetVolume, since it owns one as its scale plane).
  TNNetVolumeQuant8 = class;

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
      // SqrElements is a caller-owned scratch volume shaped like Original; it
      // is resized here only when the shape changes, so nothing is allocated
      // per call (rule #17).
      procedure CalculateLocalResponseFrom2D(Original, SqrElements: TNNetVolume; pSize:integer; alpha, beta: TNeuralFloat );
      procedure CalculateLocalResponseFromDepth(Original, SqrElements: TNNetVolume; pSize:integer; alpha, beta: TNeuralFloat );
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
      // TNNetVolumeQuant8 twin of the two calls above: one table carries the
      // codes and the per-row scales together, shaped (NumAs, 1, VectorSize) -
      // exactly the layout the open-array versions document, so these forward
      // straight to them. Coded by Claude (AI).
      procedure DotProductsTiledInt8(NumAs, NumBs, VectorSize: integer; Codes: TNNetVolumeQuant8; VBs: TNNetVolume; TileSizeA, TileSizeB: integer); overload;
      // Ranged twin (same contract as the ranged DotProductsTiled): computes
      // only B columns [BStart..BFinish] and A rows [AStart..AFinish], with
      // ceil-division tiling anchored at the range start and a clamped trailing
      // partial tile. NumAs stays the output row stride, so a sliced call
      // writes exactly its own output elements - this is what the intra-layer
      // chunk scheduler calls (position-axis chunks range B, neuron-axis
      // chunks range A). AFinish < 0 means all rows. Coded by Claude (AI).
      procedure DotProductsTiledInt8(NumAs, BStart, BFinish, VectorSize: integer; const Codes: array of ShortInt; const Scales: array of TNeuralFloat; VBs: TNNetVolume; TileSizeA, TileSizeB: integer; AStart: integer = 0; AFinish: integer = -1); overload;
      procedure DotProductsTiledInt8(NumAs, BStart, BFinish, VectorSize: integer; Codes: TNNetVolumeQuant8; VBs: TNNetVolume; TileSizeA, TileSizeB: integer; AStart: integer = 0; AFinish: integer = -1); overload;
      procedure PointwiseNorm(pNorms: TNNetVolume = nil);
      procedure PointwiseMul(pNorms: TNNetVolume);
      // Exp writes dst[0..N-1] := exp(src[0..N-1]). On an AVX2 build it
      // uses an 8-wide polynomial approximation (AVXExp) with a scalar NeuralExp
      // remainder; on a non-AVX build it is a plain NeuralExp loop. Buffers may
      // alias (dst = src) since the read happens before the write per lane/element.
      // AddScalar adds the same Value to dst[0..N-1] in place - the uniform
      // scalar accumulate that rule #13's table has no entry for (it lists
      // dst+=src and dst+=src*k, but not dst+=k). AVX2/64-bit builds broadcast
      // the value and add 32 elements per iteration; every other build runs the
      // scalar loop. Bit-exact either way: a float add is a float add.
      class procedure AddScalar(PtrA: TNeuralFloatArrPtr; Value: TNeuralFloat; pSize: integer); static;
      class procedure Exp(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // ExpShiftSum writes dst[0..N-1] := exp(src[0..N-1] - Shift) and
      // returns the sum of everything it wrote - the numerator and the
      // denominator of a numerically stable softmax in a single pass. On an
      // AVX2/64-bit build one fused kernel (AVXExpShiftSum) applies the
      // broadcast subtract, runs the 8-wide exp polynomial and reduces the
      // result while the exponentials are still in registers; every other build
      // runs the equivalent scalar loop. Buffers may alias (dst = src).
      // Arguments far below -88 exponentiate to EXACTLY 0 on both paths, so an
      // additive -1e9 attention mask still yields a hard zero weight.
      class function ExpShiftSum(pDst, pSrc: TNeuralFloatArrPtr; Shift: TNeuralFloat; N: integer): TNeuralFloat; static;
      // MaxPos returns the largest of src[0..N-1] and writes the index of its
      // FIRST occurrence into Pos - the pointer-and-count form of GetMax, so a
      // slice of a volume (a depth span, one row of a matrix) can be reduced
      // without wrapping it in a volume. Ties go to the lower index on every
      // build, matching the scalar loop exactly. N <= 0 yields 0 with Pos = -1.
      // AVX2/64-bit builds run AVXGetMaxPos (sixteen elements per iteration);
      // every other build runs the scalar loop. NaN never wins on either path.
      class function MaxPos(pSrc: TNeuralFloatArrPtr; N: integer; out Pos: integer): TNeuralFloat; static;
      // MaxValue is MaxPos when the caller wants only the value - the softmax
      // shift, a range check - and does not care where it came from.
      class function MaxValue(pSrc: TNeuralFloatArrPtr; N: integer): TNeuralFloat; static;
      // Sigmoid writes dst[0..N-1] := 1/(1+exp(-src)). AVX2-accelerated
      // path built on Exp; numerically stable scalar form on the tail.
      class procedure Sigmoid(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // Tanh writes dst[0..N-1] := tanh(src[0..N-1]). Built on Exp
      // via tanh(x) = 1 - 2/(exp(2x)+1) so it inherits Exp's AVX2 path.
      // Matches pcr_tanhf to ~1e-6. Buffers may alias (dst = src).
      class procedure Tanh(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // AdamDelta runs one whole Adam step over a weight row in a single pass:
      //   m := Beta1*m + OmBeta1*g
      //   v := Beta2*v + OmBeta2*(g*g)
      //   g := (kLR*m) / (sqrt(v*InvOmB2D) + Epsilon)
      // where g is PtrDelta in and the finished increment out, and the two
      // bias-correction denominators arrive pre-folded as scalars: InvOmB2D is
      // 1/(1-Beta2^t) and kLR is LearningRate/(1-Beta1^t). Composing this from
      // Copy/Mul/MulMulAdd/VSqrt/Add/Fill/Divi takes eleven passes and a
      // scratch row. AVX2/64-bit builds run AVXAdamDelta; every other build
      // runs the equivalent scalar loop. Neither path uses FMA, so both round
      // exactly where the composed form rounds and the results are
      // bit-identical to it.
      class procedure AdamDelta(PtrDelta, PtrM, PtrV: TNeuralFloatArrPtr;
        Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR: TNeuralFloat;
        N: integer); static;
      // Relu writes dst[0..N-1] := max(src[0..N-1], 0). AVX2-accelerated
      // (AVXCopyRelu) with a scalar fallback on non-AVX builds. Bit-exact vs the
      // scalar relu-copy. Buffers may alias (dst = src).
      class procedure Relu(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // ReluGateMask writes dst[0..N-1] := 1 where src[i] >= 0 and 0 elsewhere,
      // the ReLU derivative gate. AVX2/64-bit builds run AVXReluGateMask; every
      // other build runs the equivalent scalar loop. Bit-identical either way,
      // the >= 0 boundary included. Buffers may alias (dst = src).
      class procedure ReluGateMask(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // LeakyRelu writes dst[0..N-1] := src[i] when src[i] >= 0 and
      // Slope*src[i] otherwise - the activation every HiFi-GAN / vocoder
      // resblock applies over its whole channel x timestep signal.
      // AVX2/64-bit builds run AVXLeakyRelu (eight elements per iteration);
      // every other build runs the equivalent scalar loop. Bit-identical either
      // way, the >= 0 boundary included. Buffers may alias (dst = src).
      class procedure LeakyRelu(pDst, pSrc: TNeuralFloatArrPtr;
        Slope: TNeuralFloat; N: integer); static;
      // Largest FINITE magnitude of src[0..N-1], or 0 when nothing there is
      // finite and non-zero: NaN is skipped and +/-Inf excluded, so the result
      // is always a usable quantization range. This is the pointer-and-count
      // form of GetMaxAbs (an instance method over a whole volume) that the
      // int8 quantizers need for a row slice of a caller's buffer.
      // AVX2/64-bit builds run AVXMaxAbsFinite; every other build runs
      // the equivalent scalar loop. Coded by Claude (AI).
      class function MaxAbsFinite(pSrc: TNeuralFloatArrPtr; N: integer): TNeuralFloat; static;
      // EXACT centered sum of squares of src[0..N-1] about Mean:
      // sum_i (src[i] - Mean)^2. Divide by N for the population variance.
      // AVX2/64-bit builds run AVXSumSqrCentered; every other build runs the
      // equivalent scalar loop.
      //
      // Deliberately NOT the algebraic sum(x^2) - N*Mean^2 shortcut, which is
      // one pass cheaper but cancels catastrophically once |Mean| >> Std -- the
      // regime a weight-standardization layer lives in. The vector and scalar
      // paths differ only in summation ORDER, and every term is non-negative,
      // so the reordering is benign (unlike that shortcut). Coded by Claude (AI).
      class function SumSqrCentered(pSrc: TNeuralFloatArrPtr; Mean: TNeuralFloat; N: integer): TNeuralFloat; static;
      // Quantizes src[0..N-1] to symmetric int8 codes against a KNOWN row max:
      // dst[i] = clamp(Round(src[i] * 127/MaxAbs), -127, 127), with NaN coding
      // as 0 and +/-Inf clamping to +/-127. MaxAbs must be the value
      // MaxAbsFinite returned for this slice; MaxAbs <= 0 writes nothing
      // (a zero row has no scale - the caller owns that convention).
      //
      // NOT bit-exact against a scalar double-precision reference: the AVX2
      // path scales in single precision and may differ by one code where a
      // product lands on a rounding boundary. Quantization is lossy by
      // construction, so this is a deliberate trade for ~8 codes per
      // iteration - do not build a bit-parity test on it. Rows whose MaxAbs is
      // denormal take a double-precision scalar path instead, because the
      // single reciprocal of a denormal overflows to Inf (and would trap under
      // FPC's unmasked SSE exceptions). Coded by Claude (AI).
      class procedure QuantizeInt8(pDst: TNeuralInt8ArrPtr; pSrc: TNeuralFloatArrPtr; N: integer; MaxAbs: TNeuralFloat); static;
      // Expands symmetric int8 codes back to floats: dst[i] := Scale*src[i].
      // The inverse of QuantizeInt8, and the inner loop every block-quantized
      // checkpoint reader runs (ggml Q8_0 directly; the k-quants and the legacy
      // Q4/Q5 forms once their nibbles are unpacked into codes).
      // Bit-exact against the scalar loop on every build - one single-precision
      // multiply per element either way, so unlike QuantizeInt8 this one CAN be
      // parity-tested. AVX2/64-bit builds convert 8 codes per iteration.
      // Coded by Claude (AI).
      class procedure DequantizeInt8(pDst: TNeuralFloatArrPtr; pSrc: TNeuralInt8ArrPtr; N: integer; Scale: TNeuralFloat); static;
      // Widens N bfloat16 values to single. A bfloat16 IS the top half of a
      // single, so this is a 16-bit left shift: exact for every input, Inf and
      // NaN included, and bit-exact on all builds. AVX2/64-bit builds widen 8
      // per iteration. Coded by Claude (AI).
      class procedure DecodeBF16(pDst: TNeuralFloatArrPtr; pSrc: TNeuralHalfArrPtr; N: integer); static;
      // Widens N IEEE-754 half values to single. Every half - subnormals and
      // NaN included - is exactly representable as a single, so the conversion
      // is lossless, and it never traps whatever the input bits say.
      // Bit-exact across builds for every value a checkpoint can hold; the one
      // exception is a SIGNALLING NaN, which the AVX2 path quiets (same
      // payload, mantissa MSB set) while the scalar path passes it through.
      //
      // The AVX2/64-bit path is F16C's vcvtph2ps, 8 per iteration. F16C is a
      // CPUID bit distinct from AVX2, but no shipping AVX2 part lacks it (Intel
      // >= Haswell, AMD >= Excavator all carry both), so it rides the AVX2
      // define rather than adding a third build flavour; -dNOF16C forces the
      // scalar path for anyone who needs it. Coded by Claude (AI).
      class procedure DecodeF16(pDst: TNeuralFloatArrPtr; pSrc: TNeuralHalfArrPtr; N: integer); static;
      // Erf writes dst[0..N-1] := erf(src[0..N-1]) using the Abramowitz &
      // Stegun 7.1.26 approximation (|err| < 1.5e-7, i.e. matches pcr_erff to
      // ~1e-6). Built on Exp so it inherits the AVX2 path. dst may alias src.
      class procedure Erf(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // Sinh writes dst[0..N-1] := sinh(src[0..N-1]) via
      // sinh(x) = (exp(x) - exp(-x)) / 2, so it inherits Exp's AVX2 path.
      // exp(x) and exp(-x) are produced by two vectorized Exp passes into a
      // local scratch (NOT pDst) so pSrc is never clobbered; hence dst may alias
      // src. The clamped arg keeps exp finite; sinh stays accurate to ~1e-6 vs
      // pcr_sinhf over the activation parity range.
      class procedure Sinh(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // Ln writes dst[0..N-1] := ln(src[0..N-1]). On an AVX2 build it uses an
      // 8-wide Cephes logf polynomial (AVXLn) with a scalar pcr_logf remainder; on a
      // non-AVX build it is a plain pcr_logf loop. Matches pcr_logf to ~1e-6 over the
      // positive normal range. Buffers may alias (read-before-write per lane/element).
      class procedure Ln(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // Sin / Cos write dst[0..N-1] := sin/cos(src[0..N-1]). On an AVX2
      // build they use an 8-wide Cephes sinf/cosf polynomial (AVXSinCos) with a
      // 3-part Cody-Waite range reduction (accurate to large magnitudes) and a scalar
      // pcr_sinf/pcr_cosf remainder; non-AVX builds are plain RTL loops. ~1e-6 vs RTL.
      // Buffers may alias (dst = src).
      class procedure Sin(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      class procedure Cos(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
      // ArcSinh writes dst[0..N-1] := arcsinh(src) = ln(x + sqrt(x^2 + 1)).
      // Built on Ln (and a vectorized sqrt in the prep pass) so it inherits the
      // AVX2 path. The sqrt argument is always >= 1 so ln stays in its accurate range
      // and dst may alias src (the prep pass reads src into a scratch before Ln).
      class procedure ArcSinh(pDst, pSrc: TNeuralFloatArrPtr; N: integer); static;
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
      // Uniform scalar accumulate over the whole volume, routed through the
      // AddScalar kernel instead of TVolume's element loop. Sub adds -Value:
      // x - v and x + (-v) are the same IEEE-754 result for every input, so
      // this is bit-exact against the inherited loop.
      procedure Add(Value: Single); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Sub(Value: Single); overload; {$IFDEF Release} inline; {$ENDIF}
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
      class procedure MulMulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value1, Value2: TNeuralFloat; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure MulAdd(Value: TNeuralFloat; PtrB: TNeuralFloatArrPtr); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure MulAdd(PtrA, PtrB: TNeuralFloatArrPtr; Value: TNeuralFloat; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      class procedure MulAdd(PtrA, PtrB, PtrC: TNeuralFloatArrPtr; pSize: integer); overload; {$IFDEF Release} inline; {$ENDIF}
      // Dst := AlphaScale*Prev + BScale*B over a raw row. Same contract as
      // TVolume.RankOneUpdateRow (Prev = nil means the zero row, Prev may alias
      // Dst) but routed through the AVX kernels instead of the scalar element
      // loops, which is what every recurrent-state caller needs.
      class procedure RankOneUpdateRow(PtrDst, PtrPrev, PtrB: TNeuralFloatArrPtr;
        AlphaScale, BScale: TNeuralFloat; pSize: integer);
      procedure Divi(Value: Single); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure Copy(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure CopyRelu(Original: TNNetVolume); overload; {$IFDEF Release} inline; {$ENDIF}
      procedure CopyPadding(Original: TNNetVolume; Padding: integer); overload;
      procedure CopyPadding(Original: TNNetVolume; PaddingX, PaddingY: integer); {$IFDEF Release} inline; {$ENDIF} overload;
      procedure CopyNoChecks(Original: TNNetVolume);
      function GetSum(): TNeuralFloat; override;
      function GetSumSqr(): TNeuralFloat; override;
      {$IFDEF AVX2}
      // Served by the AVXGetMaxPos family. They keep the scalar contract in
      // full: same value, same FLastPos (first occurrence wins), same GetClass
      // tie-break. Only AVX64 assembles those kernels, so a 32-bit AVX2 build
      // keeps the inherited scalar loops.
      {$IFDEF AVX64}
      function GetMin(): TNeuralFloat; override;
      function GetMax(): TNeuralFloat; override;
      function GetMaxAbs(): TNeuralFloat; override;
      function GetClass(): integer; override;
      {$ENDIF}
      {$ENDIF}
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
      procedure GroupedDotProductsTiledInt8(Groups, NumAs, NumBs, VectorSize: integer; const Codes: array of ShortInt; const Scales: array of TNeuralFloat; VBs: TNNetVolume; TileSizeA, TileSizeB: integer); overload;
      // TNNetVolumeQuant8 twin: same (NumAs, 1, VectorSize) table shape as the
      // ungrouped kernel. Coded by Claude (AI).
      procedure GroupedDotProductsTiledInt8(Groups, NumAs, NumBs, VectorSize: integer; Codes: TNNetVolumeQuant8; VBs: TNNetVolume; TileSizeA, TileSizeB: integer); overload;
  end;

  { TNNetVolumeQuant8 }

  // Symmetric int8 storage carrying the geometry contract of TNNetVolume:
  // code (x,y,d) lives at ((SizeX*y) + x) * Depth + d. Every (x,y) pair owns
  // one scale and the dequantized value is code*Scale[x,y]. The scale plane is
  // itself a TNNetVolume of shape (SizeX, SizeY, 1), so a scale is addressed
  // by the very same formula with d = 0.
  //
  // Axis convention across the int8 call sites: Depth is always the quantized
  // vector, X the primary row index and Y a secondary grouping, 1 when there
  // is none. So a vocab table is (VocabSize, 1, EmbeddingSize), concatenated
  // layer weights are (NumNeurons, 1, VectorSize), a single-head KV cache is
  // (MaxContext, 1, d_k) and a grouped-query KV cache is
  // (MaxContext, KVHeads, d_k) - one contiguous Y-plane per KV head, which is
  // the head-major layout that cache already had. This matches how the FP32
  // volumes beside them are shaped and indexed (row index first,
  // GetRawPtr(Row, 0)). Note DeleteRows works on Y, so it drops groups, not
  // rows, under this convention.
  //
  // ReSize is the only member that changes lengths. SetLength may move the
  // buffer, so a SetLength on FData anywhere else would leave DataPtr
  // dangling. ReSize does not fill: after a growth the contents are whatever
  // SetLength left behind, and callers wanting a known state fill it
  // themselves. The empty state is Size = 0, a nil DataPtr and an empty (never
  // nil) scale plane. Coded by Claude (AI).
  TNNetVolumeQuant8 = class(TObject)
    private
      FScaleData: TNNetVolume;
      FDataPtr: TNeuralInt8ArrPtr;
      FSizeX, FSizeY, FDepth, FSize: integer;
      function GetScale(x, y: integer): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      procedure SetScale(x, y: integer; Value: TNeuralFloat); {$IFDEF Release} inline; {$ENDIF}
      function GetScalePtr(): TNeuralFloatArrPtr; {$IFDEF Release} inline; {$ENDIF}
      function GetScaleCount(): integer; {$IFDEF Release} inline; {$ENDIF}
    public
      // Exposed exactly like TNNetVolume.FData: the int8 kernels and the
      // loaders take raw pointers into it. Never SetLength it - call ReSize.
      FData: TInt8DynArr;

      constructor Create(); {$IFNDEF FPC} overload; {$ENDIF}
      constructor Create(pSizeX, pSizeY, pDepth: integer); {$IFNDEF FPC} overload; {$ENDIF}
      destructor Destroy(); override;

      procedure ReSize(pSizeX, pSizeY, pDepth: integer); overload;
      procedure ReSize(Original: TNNetVolumeQuant8); overload;

      function GetRawPos(x, y, d: integer): integer; overload; {$IFDEF Release} inline; {$ENDIF}
      function GetRawPos(x, y: integer): integer; overload; {$IFDEF Release} inline; {$ENDIF}
      // Base of the (x,y) row, or of the element (x,y,d) - convolutional taps
      // index mid-row. Both are pure arithmetic on DataPtr.
      function GetRawPtr(x, y: integer): TNeuralInt8ArrPtr; overload; {$IFDEF Release} inline; {$ENDIF}
      function GetRawPtr(x, y, d: integer): TNeuralInt8ArrPtr; overload; {$IFDEF Release} inline; {$ENDIF}

      function Get(x, y, d: integer): ShortInt; {$IFDEF Release} inline; {$ENDIF}
      procedure Store(x, y, d: integer; Value: ShortInt); {$IFDEF Release} inline; {$ENDIF}
      function GetRaw(p: integer): ShortInt; {$IFDEF Release} inline; {$ENDIF}
      procedure SetRaw(p: integer; Value: ShortInt); {$IFDEF Release} inline; {$ENDIF}

      function Dequantize(x, y, d: integer): TNeuralFloat; {$IFDEF Release} inline; {$ENDIF}
      // Expands the (x,y) row into Depth floats at Dest.
      procedure DequantizeRowTo(x, y: integer; Dest: TNeuralFloatArrPtr);
      // Expands every row into Dest, resized to (SizeX, SizeY, Depth). Leaves
      // Dest untouched when this volume is empty.
      procedure DequantizeTo(Dest: TNNetVolume);

      procedure Fill(c: ShortInt = 0);
      procedure CopyFrom(Original: TNNetVolumeQuant8);
      // Drops Count rows from row StartY on and shifts the rows above them
      // down, leaving the last Count rows stale. Capacity is untouched: this
      // is the rolling-window eviction primitive, not a resize.
      procedure DeleteRows(StartY: integer; Count: integer = 1);
      // Plain copies of both planes, for callers that export or serialize.
      procedure GetQuantData(out pCodes: TInt8DynArr; out pScales: TNeuralFloatDynArr);
      function GetMemSize(): int64;

      property SizeX: integer read FSizeX;
      property SizeY: integer read FSizeY;
      property Depth: integer read FDepth;
      property Size: integer read FSize;
      property ScaleCount: integer read GetScaleCount;
      property DataPtr: TNeuralInt8ArrPtr read FDataPtr;
      property ScalePtr: TNeuralFloatArrPtr read GetScalePtr;
      // Exposed for Fill/Copy/inspection. Never ReSize it directly: ReSize
      // keeps it in step with FData.
      property ScaleData: TNNetVolume read FScaleData;
      property Scale[x, y: integer]: TNeuralFloat read GetScale write SetScale;
  end;

  { TNNetSamplerBase }

  TNNetSamplerBase = class(TObject)
    protected
      FTokenArr: TNNetTokenArray;
      // Live candidate window: only FTokenArr[0..FCount-1] is meaningful.
      // FTokenArr itself stays vocabulary-sized so the load path never
      // reallocates (rule #17); a truncating stage just lowers FCount, the
      // shape llama.cpp's llama_token_data_array{size, sorted} uses. FSorted
      // records whether that window is already in descending Score order, so
      // a later stage can skip a redundant sort.
      FCount: integer;
      FSorted: boolean;
      // Fill the candidate window from a whole volume / from one pixel and
      // arm it as untruncated + unsorted.
      procedure LoadCandidates(Origin: TNNetVolume);
      procedure LoadCandidatesOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer);
    public
      function GetToken(Origin: TNNetVolume): integer; virtual; abstract;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; virtual; abstract;
      // Sorts the live window descending by Score. A no-op when FSorted is
      // already set.
      procedure SortTokenArray();
      // Reduces the window to its K highest-Score entries, sorted descending,
      // WITHOUT sorting the whole vocabulary: an O(n) average quickselect
      // partition followed by a sort of the K-element prefix. The prefix is
      // identical to the first K entries of a full descending sort (ties
      // aside), so callers that only read [0..K-1] are unaffected.
      // Sorted=False skips that prefix sort. It is for callers that draw
      // UNIFORMLY inside the window and so do not depend on the order within
      // it; every caller that walks the window in cumulative order (top-p,
      // min-p, Mirostat, weighted top-k) needs the default.
      procedure SelectTopCandidates(K: integer; Sorted: boolean = True);
      // Re-arms the full window and sorts it. Used as the fallback when a
      // truncated window turned out to be too small to answer the query.
      procedure RestoreFullWindowSorted();
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
      // Uniform draw over the live window (see the implementation note).
      function DrawFromWindow(): integer;
    public
      constructor Create(TopK: integer);
      function GetToken(Origin: TNNetVolume): integer; override;
      function GetTokenOnPixel(Origin: TNNetVolume; PixelX, PixelY: integer): integer; override;
  end;

  { TNNetSamplerTopP }
  TNNetSamplerTopP = class (TNNetSamplerBase)
    protected
      FTopP: TNeuralFloat;
      // Cumulative-mass cut over the candidate window, with an adaptive
      // top-k pre-truncation so the whole vocabulary is not sorted to find
      // a nucleus that is typically a few dozen tokens wide.
      function SampleFromNucleus(): integer;
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
      // Counts the p >= MinP * max(p) survivors in linear passes and reduces
      // the window to exactly that many, so the full row is never sorted.
      procedure TruncateToMinP();
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
      // Persistent scratch, lazily sized to the vocab and reused across tokens
      // so SampleTypical allocates ~once over the sampler's lifetime, not per
      // call (rule #17 amortized-field form).
      FDist: array of TNeuralFloat;  // |surprise - entropy| per FTokenArr entry
      FOrder: array of integer;      // FTokenArr indices sorted by ascending FDist
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
      // Compact list of the DISTINCT token ids with a non-zero count, in
      // first-seen order (mirrors llama.cpp's penalty token_count map). Apply
      // and ApplyToProbabilities walk this instead of scanning FCounts up to
      // the highest id ever registered - the history is a few hundred tokens
      // while FCounts spans the vocabulary (152k on Qwen2.5). Rule #17: grown
      // only by RegisterToken, never inside the per-token apply path.
      FSeen: array of integer;
      FSeenCount: integer;
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
  // Implemented in the AVX32 / AVX64 asm blocks. Buffers may alias. Call sites
  // outside this unit want TNNetVolume.Exp, which dispatches on the build.
  procedure AVXExp(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer);
  // AVXLn writes pDst[0..N-1] := ln(pSrc[0..N-1]) via an 8-wide Cephes logf
  // polynomial (scalar pcr_logf remainder). Buffers may alias.
  procedure AVXLn(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer);
  // AVXSinCos writes pDst[0..N-1] := sin or cos of pSrc[0..N-1] via an 8-wide Cephes
  // sinf/cosf polynomial with 3-part Cody-Waite range reduction (scalar RTL
  // remainder). DoCos selects cos (true) vs sin (false). Buffers may alias.
  procedure AVXSinCos(pDst, pSrc: TNeuralFloatArrPtr; NumElements: integer; DoCos: boolean);
  // AVXGetSum returns the sum of pSrc[0..N-1] via an 8-wide AVX2 reduction
  // (scalar tail for the N mod 4 remainder). Call sites outside this unit want
  // TNNetVolume.GetSum, which dispatches on the build.
  function AVXGetSum(PtrA: TNeuralFloatArrPtr; NumElements: integer): Single;
  {$ENDIF}

implementation

uses
  Math, neuralbit, strutils;

// Scalar IEEE-754 half -> single. Pure bit surgery, so no floating-point
// compare happens and nothing here can trap: a signalling NaN half widens to
// the matching signalling NaN single untouched. Used by DecodeF16's non-AVX
// build and by the AVX tail. Coded by Claude (AI).
function NeuralHalfToSingle(Bits: Word): Single;
var
  Sign, Exponent, Mantissa, OutBits: Cardinal;
begin
  Sign := (Bits shr 15) and $1;
  Exponent := (Bits shr 10) and $1F;
  Mantissa := Bits and $3FF;
  if Exponent = 0 then
  begin
    if Mantissa = 0 then
    begin
      OutBits := Sign shl 31;                  // signed zero
      Result := PSingle(@OutBits)^;
    end
    else
    begin
      // Subnormal half: value = (-1)^s * m * 2^-24, exact in single.
      Result := Mantissa * 5.9604644775390625e-8;
      if Sign <> 0 then Result := -Result;
    end;
  end
  else if Exponent = $1F then
  begin
    // Inf / NaN: rebuild against single's all-ones exponent.
    OutBits := (Sign shl 31) or ($FF shl 23) or (Mantissa shl 13);
    Result := PSingle(@OutBits)^;
  end
  else
  begin
    // Normal: rebias the exponent 15 -> 127, widen the mantissa 10 -> 23 bits.
    OutBits := (Sign shl 31) or ((Exponent + 112) shl 23) or (Mantissa shl 13);
    Result := PSingle(@OutBits)^;
  end;
end;

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
  vxorps ymm6, ymm6, ymm6
  vxorps ymm7, ymm7, ymm7

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
  vfmadd231ps ymm6, ymm4, [rdx+64]
  vfmadd231ps ymm7, ymm5, [rdx+96]

  add rax, 32
  add rdx, 128
  dec ecx
  jnz @LargeAddLoop

  vaddps ymm0, ymm0, ymm1
  vaddps ymm6, ymm6, ymm7
  vaddps ymm0, ymm0, ymm6
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
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6', 'ymm7'
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

// Largest FINITE magnitude of NumElements floats: the sign bit is cleared with
// a broadcast $7FFFFFFF mask and each vector is masked against MaxSingle with
// the NON-SIGNALING LE_OQ predicate (18), which is false for both NaN and
// +/-Inf - so a non-finite lane contributes 0 and no compare can raise
// EInvalidOp under FPC's unmasked SSE exceptions. Eight lanes at a time; the
// fold and the tail repeat the same masking in Pascal.
//
// Both vector constants are broadcast from a LOCAL pair reached through a
// pointer (the AVXAddScalar idiom) rather than from a global const table: no
// [rip+label] relocation means nothing here can break position-independent
// linking of the examples. Coded by Claude (AI).
function AVXMaxAbsFinite(PtrA: TNeuralFloatArrPtr;
  NumElements: integer): Single;
var
  vMax: array[0..7] of Single;
  // [0] = the $7FFFFFFF sign-clearing mask, [1] = MaxSingle.
  Consts: array[0..1] of Single;
  ConstsPtr: pointer;
  localNumElements, i, NumElementsM1: integer;
  v, AbsV: Single;
begin
  PLongWord(@Consts[0])^ := $7FFFFFFF;
  Consts[1] := MaxSingle;
  localNumElements := NumElements and (not 7);
  Result := 0;
  if localNumElements > 0 then
  begin
    ConstsPtr := Addr(Consts[0]);
  asm
  mov rax, PtrA
  mov rdx, ConstsPtr
  mov ecx, localNumElements
  shr ecx, 3
  vbroadcastss ymm3, [rdx]
  vbroadcastss ymm2, [rdx+4]
  vxorps    ymm4, ymm4, ymm4
@Loop:
  vmovups   ymm0, [rax]
  vandps    ymm0, ymm0, ymm3   // |x|; a NaN stays a NaN
  vcmpps    ymm1, ymm0, ymm2, 18  // LE_OQ: false for NaN and for +Inf
  vandps    ymm0, ymm0, ymm1   // non-finite lanes -> 0
  vmaxps    ymm4, ymm4, ymm0
  add rax, 32
  dec ecx
  jnz @Loop
  vmovups   vMax, ymm4
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX', 'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4'
  ];
    Result := vMax[0];
    for i := 1 to 7 do
      if vMax[i] > Result then Result := vMax[i];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
  begin
    v := PtrA^[i];
    if IsNan(v) then continue;
    AbsV := Abs(v);
    if (AbsV > Result) and (AbsV <= MaxSingle) then Result := AbsV;
  end;
end;

// Symmetric int8 quantization of NumElements floats against a known row max:
// code = clamp(Round(v * (1/MaxAbs) * 127), -127, 127), NaN -> 0.
// MaxAbs must be > 0 and normal (the caller routes denormal maxima to the
// double-precision scalar path, whose reciprocal cannot overflow).
//
// The two multiplies are NOT folded into one 127/MaxAbs constant on purpose:
// that product overflows single for the tiny-magnitude rows real checkpoints
// pad their vocab with (max ~1.18e-37 -> 1.07e39). Multiplying by 1/MaxAbs
// FIRST bounds every intermediate by 1 in magnitude, so 127*that is bounded by
// 127 and single precision suffices throughout.
//
// Per 8 lanes: an ORD_Q compare (7, non-signaling) zeroes NaN, the two
// multiplies scale, vminps/vmaxps clamp to +/-127 (which is also what turns
// +/-Inf into +/-127), vcvtps2dq rounds to nearest-even in the default MXCSR
// mode - the same rounding FPC's Round() emits - and the two saturating packs
// narrow 8 dwords to 8 bytes in lane order via an xmm extract, so no
// cross-lane fixup is needed. Coded by Claude (AI).
procedure AVXQuantizeInt8(PtrDst: TNeuralInt8ArrPtr;
  PtrSrc: TNeuralFloatArrPtr; NumElements: integer; MaxAbs: Single);
var
  localNumElements, i, NumElementsM1, Code: integer;
  Recip, Scaled, v: Single;
  // [0] = 1/MaxAbs, [1] = +127, [2] = -127. One pointer, three broadcasts -
  // locals only, so no [rip+label] relocation (see AVXMaxAbsFinite).
  Consts: array[0..2] of Single;
  ConstsPtr: pointer;
begin
  Recip := 1 / MaxAbs;
  localNumElements := NumElements and (not 7);
  if localNumElements > 0 then
  begin
    Consts[0] := Recip;
    Consts[1] := 127;
    Consts[2] := -127;
    ConstsPtr := Addr(Consts[0]);
  asm
  mov rax, PtrSrc
  mov rdx, PtrDst
  mov r8, ConstsPtr
  mov ecx, localNumElements
  shr ecx, 3
  vbroadcastss ymm5, [r8]
  vbroadcastss ymm6, [r8+4]
  vbroadcastss ymm7, [r8+8]
@Loop:
  vmovups   ymm0, [rax]
  vcmpps    ymm1, ymm0, ymm0, 7   // ORD_Q: false only for NaN
  vandps    ymm0, ymm0, ymm1      // NaN -> 0
  vmulps    ymm0, ymm0, ymm5      // * 1/MaxAbs  (|.| <= 1, Inf stays Inf)
  vmulps    ymm0, ymm0, ymm6      // * 127
  vminps    ymm0, ymm0, ymm6      // clamp +127 (+Inf -> 127)
  vmaxps    ymm0, ymm0, ymm7      // clamp -127 (-Inf -> -127)
  vcvtps2dq ymm0, ymm0            // round to nearest even
  vextracti128 xmm1, ymm0, 1
  vpackssdw xmm0, xmm0, xmm1      // 8 dwords -> 8 words, in lane order
  vpacksswb xmm0, xmm0, xmm0      // low 8 bytes = the 8 codes
  vmovq     [rdx], xmm0
  add rax, 32
  add rdx, 8
  dec ecx
  jnz @Loop
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX', 'R8',
    'ymm0', 'ymm1', 'ymm5', 'ymm6', 'ymm7'
  ];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
  begin
    v := PtrSrc^[i];
    if IsNan(v) then
    begin
      PtrDst^[i] := 0;
      continue;
    end;
    Scaled := v * Recip * 127;
    if Scaled > 127 then Scaled := 127
    else if Scaled < -127 then Scaled := -127;
    Code := Round(Scaled);
    PtrDst^[i] := ShortInt(Code);
  end;
end;

// dst[i] := Scale * src[i] over NumElements symmetric int8 codes. Per 8 lanes:
// vpmovsxbd sign-extends 8 bytes to dwords, vcvtdq2ps converts, one broadcast
// vmulps applies the scale - the same three steps AVXDotProductInt8 uses to
// materialize weights in-register, here written out to memory instead.
//
// Bit-exact against the scalar tail: both round exactly one single-precision
// product. The scale is broadcast from a LOCAL through a pointer (the
// AVXAddScalar idiom) so no [rip+label] relocation appears and
// position-independent linking of the examples keeps working - see
// AVXMaxAbsFinite. Coded by Claude (AI).
procedure AVXDequantizeInt8(PtrDst: TNeuralFloatArrPtr;
  PtrSrc: TNeuralInt8ArrPtr; NumElements: integer; Scale: Single);
var
  localNumElements, i, NumElementsM1: integer;
  localScale: Single;
  ScalePtr: pointer;
begin
  localNumElements := NumElements and (not 7);
  if localNumElements > 0 then
  begin
    localScale := Scale;
    ScalePtr := Addr(localScale);
  asm
  mov rax, PtrSrc
  mov rdx, PtrDst
  mov r8, ScalePtr
  mov ecx, localNumElements
  shr ecx, 3
  vbroadcastss ymm2, [r8]
@Loop:
  vpmovsxbd ymm0, [rax]      // 8 codes -> 8 sign-extended dwords
  vcvtdq2ps ymm0, ymm0
  vmulps    ymm0, ymm0, ymm2
  vmovups   [rdx], ymm0
  add rax, 8
  add rdx, 32
  dec ecx
  jnz @Loop
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX', 'R8', 'ymm0', 'ymm2'
  ];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
    PtrDst^[i] := Scale * PtrSrc^[i];
end;

// dst[i] := bfloat16(src[i]) widened to single. A bfloat16 is literally the
// high 16 bits of the single, so per 8 lanes vpmovzxwd spreads the halves into
// dwords (value in the LOW half of each) and one vpslld shifts each up into
// place. No floating-point operation executes at all, so this cannot trap on
// NaN payloads and is exact by construction. Coded by Claude (AI).
procedure AVXDecodeBF16(PtrDst: TNeuralFloatArrPtr;
  PtrSrc: TNeuralHalfArrPtr; NumElements: integer);
var
  localNumElements, i, NumElementsM1: integer;
  OutBits: Cardinal;
begin
  localNumElements := NumElements and (not 7);
  if localNumElements > 0 then
  begin
  asm
  mov rax, PtrSrc
  mov rdx, PtrDst
  mov ecx, localNumElements
  shr ecx, 3
@Loop:
  vpmovzxwd ymm0, [rax]      // 8 bfloat16 -> 8 dwords, value in bits 0..15
  vpslld    ymm0, ymm0, 16   // shift into bits 16..31 = the single's bits
  vmovups   [rdx], ymm0
  add rax, 16
  add rdx, 32
  dec ecx
  jnz @Loop
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX', 'ymm0'
  ];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
  begin
    OutBits := Cardinal(PtrSrc^[i]) shl 16;
    PtrDst^[i] := PSingle(@OutBits)^;
  end;
end;

// dst[i] := 1.0 when src[i] >= 0, else 0.0 -- the ReLU derivative gate mask.
// The non-signaling GE_OQ predicate (29) is false for NaN and true for -0.0,
// which is exactly what the scalar `src >= 0` test does, and the vandps against
// a broadcast 1.0 leaves 1.0 and 0.0 as the only possible outputs. So this is
// bit-identical to the scalar loop, boundary included.
//
// The 1.0 is broadcast from a LOCAL reached through a pointer (the AVXAddScalar
// idiom) so no [rip+label] relocation appears and position-independent linking
// of the examples keeps working - see AVXMaxAbsFinite. Coded by Claude (AI).
procedure AVXReluGateMask(PtrDst, PtrSrc: TNeuralFloatArrPtr;
  NumElements: integer);
var
  localNumElements, i, NumElementsM1: integer;
  localOne: Single;
  OnePtr: pointer;
begin
  localNumElements := NumElements and (not 7);
  if localNumElements > 0 then
  begin
    localOne := 1.0;
    OnePtr := Addr(localOne);
  asm
  mov rax, PtrSrc
  mov rdx, PtrDst
  mov r8, OnePtr
  mov ecx, localNumElements
  shr ecx, 3
  vbroadcastss ymm2, [r8]
  vxorps    ymm3, ymm3, ymm3
@Loop:
  vmovups   ymm0, [rax]
  vcmpps    ymm1, ymm0, ymm3, 29   // GE_OQ: false for NaN, true for -0.0
  vandps    ymm1, ymm1, ymm2
  vmovups   [rdx], ymm1
  add rax, 32
  add rdx, 32
  dec ecx
  jnz @Loop
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX', 'R8', 'ymm0', 'ymm1', 'ymm2', 'ymm3'
  ];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
    if PtrSrc^[i] >= 0 then PtrDst^[i] := 1 else PtrDst^[i] := 0;
end;

// dst[i] := src[i] when src[i] >= 0, else Slope * src[i] - eight elements per
// iteration as a compare, a multiply and a blend. The slope is broadcast from a
// local, so the kernel references no global constant and stays position
// independent. Bit-exact against the scalar tail: the taken branch is the same
// single-precision multiply, GE_OQ selects -0.0 as non-negative exactly as the
// scalar >= 0 does, and NaN falls to the multiply on both paths.
procedure AVXLeakyRelu(PtrDst, PtrSrc: TNeuralFloatArrPtr;
  Slope: TNeuralFloat; NumElements: integer);
var
  localNumElements, i, NumElementsM1: integer;
  localSlope: Single;
  SlopePtr: pointer;
begin
  localNumElements := NumElements and (not 7);
  if localNumElements > 0 then
  begin
    localSlope := Slope;
    SlopePtr := Addr(localSlope);
  asm
  mov rax, PtrSrc
  mov rdx, PtrDst
  mov r8, SlopePtr
  mov ecx, localNumElements
  shr ecx, 3
  vbroadcastss ymm2, [r8]
  vxorps    ymm3, ymm3, ymm3
@Loop:
  vmovups   ymm0, [rax]
  vmulps    ymm1, ymm0, ymm2       // Slope * x
  vcmpps    ymm4, ymm0, ymm3, 29   // GE_OQ: false for NaN, true for -0.0
  vblendvps ymm1, ymm1, ymm0, ymm4 // x where x >= 0, else Slope * x
  vmovups   [rdx], ymm1
  add rax, 32
  add rdx, 32
  dec ecx
  jnz @Loop
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX', 'R8', 'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4'
  ];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
    if PtrSrc^[i] >= 0 then PtrDst^[i] := PtrSrc^[i]
    else PtrDst^[i] := Slope * PtrSrc^[i];
end;


{$IFNDEF NOF16C}
// dst[i] := half(src[i]) widened to single, 8 per iteration through F16C's
// vcvtph2ps. Every half is exactly representable as a single, so the
// instruction is a lossless widening and agrees bit-for-bit with the scalar
// NeuralHalfToSingle tail.
//
// vcvtph2ps raises #I for a SIGNALLING NaN input, and FPC leaves the SSE
// invalid-operation exception unmasked - so a corrupt file carrying one would
// crash an AVX2 build while the pure bit-surgery scalar path decoded it
// happily. The three integer ops before the conversion close that gap: any
// half whose |value| exceeds $7C00 is a NaN (all-ones exponent, non-zero
// mantissa), and setting its mantissa MSB makes it quiet. Inf is exactly
// $7C00 so it is left alone, and no finite value can reach the compare.
// The one visible consequence is that a signalling NaN decodes to the QUIET
// NaN of the same payload here, where the scalar path preserves it verbatim.
//
// The conversion is emitted as raw bytes because FPC 3.2.2's assembler has no
// F16C mnemonics at all - "Unrecognized opcode VCVTPH2PS". The five bytes are
// "vcvtph2ps ymm0, xmm0" = VEX.256.66.0F38.W0 13 /r:
//   C4 E2 7D   3-byte VEX - R/X/B all unset (ymm0, xmm0), mmmmm=0F38, W=0,
//              vvvv unused, L=1 (256-bit), pp=66
//   13         the opcode
//   C0         ModRM mod=11 reg=000 (ymm0) rm=000 (xmm0)
// TestDecodeF16 and TestDecodeF16SpecialValues cover the results, and the
// encoding is verified by disassembling the built binary. Coded by Claude (AI).
procedure AVXDecodeF16(PtrDst: TNeuralFloatArrPtr;
  PtrSrc: TNeuralHalfArrPtr; NumElements: integer);
var
  localNumElements, i, NumElementsM1: integer;
  // Three word-wide vector constants, held in LOCALS and reached through a
  // pointer so no [rip+label] relocation appears (see AVXMaxAbsFinite):
  // [0..7] = $7FFF sign mask, [8..15] = $7C00 (Inf), [16..23] = $0200 quiet bit.
  Consts: array[0..23] of Word;
  ConstsPtr: pointer;
begin
  localNumElements := NumElements and (not 7);
  if localNumElements > 0 then
  begin
    for i := 0 to 7 do
    begin
      Consts[i] := $7FFF;
      Consts[i + 8] := $7C00;
      Consts[i + 16] := $0200;
    end;
    ConstsPtr := Addr(Consts[0]);
  asm
  mov rax, PtrSrc
  mov rdx, PtrDst
  mov r8, ConstsPtr
  mov ecx, localNumElements
  shr ecx, 3
  vmovups   xmm2, [r8]
  vmovups   xmm3, [r8+16]
  vmovups   xmm4, [r8+32]
@Loop:
  vmovups   xmm0, [rax]          // 8 halves
  vpand     xmm1, xmm0, xmm2     // |h|
  vpcmpgtw  xmm1, xmm1, xmm3     // |h| > $7C00  <=>  h is NaN
  vpand     xmm1, xmm1, xmm4     // quiet bit, only on the NaN lanes
  vpor      xmm0, xmm0, xmm1     // signalling NaN -> quiet NaN
  db $C4, $E2, $7D, $13, $C0     // vcvtph2ps ymm0, xmm0: 8 halves -> 8 singles
  vmovups   [rdx], ymm0
  add rax, 16
  add rdx, 32
  dec ecx
  jnz @Loop
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX', 'R8', 'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4'
  ];
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
    PtrDst^[i] := NeuralHalfToSingle(PtrSrc^[i]);
end;

// EXACT centered sum of squares: sum_i (PtrA[i] - Mean)^2, eight lanes at a
// time. Each vector is loaded, the broadcast mean is subtracted and the
// difference is FMA'd into one of two accumulators; the scalar tail repeats the
// same subtract-then-square in Pascal.
//
// The centered form is the point: the algebraic shortcut sum(x^2) - N*Mean^2
// cancels catastrophically once |Mean| >> Std, which is exactly the regime a
// weight-standardization layer sits in.
//
// The mean is broadcast from a LOCAL reached through a pointer (the
// AVXMaxAbsFinite idiom) rather than a global, so no [rip+label] relocation
// appears and position-independent linking of the examples is unaffected.
// Coded by Claude (AI).
function AVXSumSqrCentered(PtrA: TNeuralFloatArrPtr; Mean: Single;
  NumElements: integer): Single;
var
  vRes: array[0..7] of Single;
  vMean: Single;
  vMeanPtr: pointer;
  localNumElements, i, NumElementsM1: integer;
  Centered: Single;
begin
  vMean := Mean;
  // 16 elements per iteration across two accumulators, so the FMA latency is
  // covered; the tail (up to 15) is folded scalar below.
  localNumElements := NumElements and (not 15);
  Result := 0;
  if localNumElements > 0 then
  begin
    vMeanPtr := Addr(vMean);
  asm
  mov rax, PtrA
  mov rdx, vMeanPtr
  mov ecx, localNumElements
  shr ecx, 4
  vbroadcastss ymm3, [rdx]
  vxorps    ymm0, ymm0, ymm0
  vxorps    ymm1, ymm1, ymm1
@Loop:
  vmovups   ymm2, [rax]
  vmovups   ymm4, [rax+32]
  vsubps    ymm2, ymm2, ymm3
  vsubps    ymm4, ymm4, ymm3
  vfmadd231ps ymm0, ymm2, ymm2
  vfmadd231ps ymm1, ymm4, ymm4
  add rax, 64
  dec ecx
  jnz @Loop
  vaddps    ymm0, ymm0, ymm1
  vmovups   vRes, ymm0
  vzeroupper
  end
  [
    'RAX', 'RCX', 'RDX', 'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4'
  ];
    Result := ((vRes[0] + vRes[1]) + (vRes[2] + vRes[3])) +
              ((vRes[4] + vRes[5]) + (vRes[6] + vRes[7]));
  end;
  NumElementsM1 := NumElements - 1;
  for i := localNumElements to NumElementsM1 do
  begin
    Centered := PtrA^[i] - Mean;
    Result := Result + Centered * Centered;
  end;
end;
{$ENDIF}
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
  // Subtracting the row max leaves every element at <= 0, so the fused
  // shift-exp-sum kernel covers the whole stabilized numerator and its
  // denominator in one pass instead of subtract, exp and reduce in three.
  Sum := TNNetVolume.ExpShiftSum(Row.DataPtr, Row.DataPtr, MaxV, Row.Size);
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
  RowsM1, RowOfs: integer;
  Norm: TNeuralFloat;
  RowPtr: TNeuralFloatArrPtr;
begin
  Rows := Mat.SizeX;
  Dim := Mat.Depth;
  RowsM1 := Rows - 1;
  RowOfs := 0;
  for R := 0 to RowsM1 do
  begin
    RowPtr := Mat.GetRawPtr(RowOfs);
    Norm := Sqrt(TNNetVolume.DotProduct(RowPtr, RowPtr, Dim));
    if Norm > 0 then
      TNNetVolume.Mul(RowPtr, 1 / Norm, Dim);
    Inc(RowOfs, Dim);
  end;
end;

function NeuralLinearSolve(var A: array of TNeuralFloat;
  var B: array of TNeuralFloat; n, m: integer): boolean;
var
  col, row, piv, k: integer;
  nM1, mM1, rowStart, rowBase, idxC, idxP: integer;
  colBaseN, colBaseM, pivBaseN, pivBaseM, rowBaseN, rowBaseM: integer;
  maxAbs, v, factor, diag, tmp, InvDiag: TNeuralFloat;
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
    rowBase := rowStart * n;
    for row := rowStart to nM1 do
    begin
      v := Abs(A[rowBase + col]);
      if v > maxAbs then begin maxAbs := v; piv := row; end;
      Inc(rowBase, n);
    end;
    if maxAbs < 1e-30 then begin Result := False; Exit; end;

    // Swap the pivot row into place (in both A and B).
    if piv <> col then
    begin
      pivBaseN := piv * n;
      pivBaseM := piv * m;
      for k := 0 to nM1 do
      begin
        idxC := colBaseN + k; idxP := pivBaseN + k;
        tmp := A[idxC]; A[idxC] := A[idxP]; A[idxP] := tmp;
      end;
      for k := 0 to mM1 do
      begin
        idxC := colBaseM + k; idxP := pivBaseM + k;
        tmp := B[idxC]; B[idxC] := B[idxP]; B[idxP] := tmp;
      end;
    end;

    // Normalise the pivot row so A[col,col] = 1.
    diag := A[colBaseN + col];
    InvDiag := 1 / diag;
    TNNetVolume.Mul(@A[colBaseN], InvDiag, n);
    TNNetVolume.Mul(@B[colBaseM], InvDiag, m);

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
  best: TNeuralFloat;
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
    best := Scores[Order[i]]; // #4: pivot keyed value, refreshed on swap
    for jj := jjStart to HiCand do
      if Scores[Order[jj]] > best then
      begin tmp := Order[i]; Order[i] := Order[jj]; Order[jj] := tmp; best := Scores[Order[i]]; end;
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
  X0 := Cx - (CutW shr 1);
  Y0 := Cy - (CutH shr 1);
  X1 := Cx + (CutW - (CutW shr 1));
  Y1 := Cy + (CutH - (CutH shr 1));
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
          PastePos := CutA.GetRawPos(X, Y);
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
  Mid := A[(Lo + Hi) shr 1];
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

// In-place quicksort of the index array Order so that Dist[Order[0..]] ends up
// ascending. Sorts the caller's existing Order/Dist buffers by reference and
// recurses on the CPU stack only, so it adds no heap allocation (rule #17). Used
// by SampleTypical to replace an O(N^2) selection sort over the full vocab.
procedure QuickSortOrderByDist(var Order: array of integer;
  const Dist: array of TNeuralFloat; iLo, iHi: Integer);
var
  Lo, Hi, T: Integer;
  Mid: TNeuralFloat;
begin
  Lo := iLo;
  Hi := iHi;
  Mid := Dist[Order[(Lo + Hi) shr 1]];
  repeat
    while Dist[Order[Lo]] < Mid do Inc(Lo);
    while Dist[Order[Hi]] > Mid do Dec(Hi);
    if Lo <= Hi then
    begin
      T := Order[Lo];
      Order[Lo] := Order[Hi];
      Order[Hi] := T;
      Inc(Lo);
      Dec(Hi);
    end;
  until Lo > Hi;
  if Hi > iLo then QuickSortOrderByDist(Order, Dist, iLo, Hi);
  if Lo < iHi then QuickSortOrderByDist(Order, Dist, Lo, iHi);
end;

// Hoare-partition quickselect over the index array: permutes Order[0..Count-1]
// so that the K entries with the SMALLEST Dist occupy Order[0..K-1] (in no
// particular order). Average O(Count) and, like QuickSortOrderByDist, it works
// entirely inside the caller's buffers - no heap allocation (rule #17). The
// median-of-three pivot keeps already-ordered and all-equal inputs off the
// quadratic path. Coded by Claude (AI).
procedure PartialSelectOrderByDist(var Order: array of integer;
  const Dist: array of TNeuralFloat; Count, K: integer);
var
  Lo, Hi, I, J, Mid, Bound, T: integer;
  Pivot: TNeuralFloat;
begin
  if (K <= 0) or (K >= Count) then exit;
  Lo := 0;
  Hi := Count - 1;
  Bound := K - 1;
  while Lo < Hi do
  begin
    // Median of Dist[Order[Lo]], Dist[Order[Mid]], Dist[Order[Hi]], left at Mid.
    Mid := (Lo + Hi) shr 1;
    if Dist[Order[Lo]] > Dist[Order[Mid]] then
    begin
      T := Order[Lo]; Order[Lo] := Order[Mid]; Order[Mid] := T;
    end;
    if Dist[Order[Lo]] > Dist[Order[Hi]] then
    begin
      T := Order[Lo]; Order[Lo] := Order[Hi]; Order[Hi] := T;
    end;
    if Dist[Order[Mid]] > Dist[Order[Hi]] then
    begin
      T := Order[Mid]; Order[Mid] := Order[Hi]; Order[Hi] := T;
    end;
    Pivot := Dist[Order[Mid]];
    I := Lo;
    J := Hi;
    repeat
      while Dist[Order[I]] < Pivot do Inc(I);
      while Dist[Order[J]] > Pivot do Dec(J);
      if I <= J then
      begin
        T := Order[I]; Order[I] := Order[J]; Order[J] := T;
        Inc(I);
        Dec(J);
      end;
    until I > J;
    // Recurse into the side that still holds the K-th boundary; when the
    // boundary falls inside the equal-to-pivot gap the split is already exact.
    if Bound <= J then Hi := J
    else if Bound >= I then Lo := I
    else Break;
  end;
end;

{ TNNetSamplerTopP }

constructor TNNetSamplerTopP.Create(TopP: TNeuralFloat);
begin
  inherited Create();
  FTopP := TopP;
end;

function TNNetSamplerTopP.SampleFromNucleus(): integer;
const
  // Below this window size a full sort is already cheap, so skip the
  // partial-selection machinery entirely.
  csTopPAdaptiveMin = 1024;
  // First guess at the nucleus width. A 0.8-0.95 nucleus over a peaked
  // next-token distribution is far narrower than this in practice; when it
  // is not, the retry below pays one full sort and is still correct.
  csTopPAdaptiveK = 256;
var
  CumulativeSum: TNeuralFloat;
  I, Threshold, Hi: Integer;
  Found, Truncated: boolean;
begin
  if FCount = 0 then
  begin
    Result := 0; // defensive: empty distribution
    exit;
  end;
  // Sorting 152k tokens to consume the first few dozen is the dominant cost
  // of this sampler, so try a bounded prefix first (llama.cpp's adaptive
  // top-k scheme) and only widen when the mass was not reached.
  Truncated := FCount > csTopPAdaptiveMin;
  if Truncated then SelectTopCandidates(csTopPAdaptiveK)
  else SortTokenArray();
  repeat
    CumulativeSum := 0;
    Threshold := 0;
    Found := false;
    Hi := FCount - 1;
    for I := 0 to Hi do
    begin
      CumulativeSum := CumulativeSum + FTokenArr[I].Score;
      if CumulativeSum > FTopP then
      begin
        Threshold := I;
        Found := true;
        Break;
      end;
    end;
    if Found or (not Truncated) then Break;
    // The prefix did not hold FTopP of the mass: re-arm the full row (the
    // partition only permuted it) and redo the scan exactly once.
    RestoreFullWindowSorted();
    Truncated := false;
  until false;

  // Randomly select one of the top tokens within the threshold.
  if Threshold > 0 then
    Result := FTokenArr[Random(Threshold)].Token
  else
    Result := FTokenArr[0].Token; // Fallback in case P is too low.
end;

function TNNetSamplerTopP.GetToken(Origin: TNNetVolume): integer;
begin
  LoadCandidates(Origin);
  Result := SampleFromNucleus();
end;

function TNNetSamplerTopP.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  LoadCandidatesOnPixel(Origin, PixelX, PixelY);
  Result := SampleFromNucleus();
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
  if FCount = 0 then
  begin
    Result := 0; // defensive: empty distribution
    exit;
  end;
  // FTokenArr is sorted DESCENDING, so [0] holds the max probability.
  Threshold := FMinP * FTokenArr[0].Score;
  KeptCount := 0;
  KeptSum := 0;
  Hi := FCount - 1;
  Lo := 0;
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

procedure TNNetSamplerMinP.TruncateToMinP();
var
  I, Hi, KeepCount: integer;
  MaxScore, Threshold: TNeuralFloat;
begin
  if FCount = 0 then exit;
  // The p >= MinP * max(p) cut does not need a sorted row to be COUNTED, so
  // find the max and the survivor count in two linear passes and select
  // exactly that many. Avoids sorting the whole vocabulary for a kept set
  // that is normally a handful of tokens.
  Hi := FCount - 1;
  MaxScore := FTokenArr[0].Score;
  for I := 1 to Hi do
    if FTokenArr[I].Score > MaxScore then MaxScore := FTokenArr[I].Score;
  Threshold := FMinP * MaxScore;
  KeepCount := 0;
  for I := 0 to Hi do
    if FTokenArr[I].Score >= Threshold then Inc(KeepCount);
  if KeepCount < 1 then KeepCount := 1;
  SelectTopCandidates(KeepCount);
end;

function TNNetSamplerMinP.GetToken(Origin: TNNetVolume): integer;
begin
  LoadCandidates(Origin);
  TruncateToMinP();
  Result := SampleFromSorted();
end;

function TNNetSamplerMinP.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  LoadCandidatesOnPixel(Origin, PixelX, PixelY);
  TruncateToMinP();
  Result := SampleFromSorted();
end;

{ TNNetSamplerTopK }

constructor TNNetSamplerTopK.Create(TopK: integer);
begin
  inherited Create();
  FTopK := TopK;
end;

// Uniform draw over the truncated window. FCount is the live window size, so
// it is already Min(FTopK, vocabulary size): drawing on FTopK instead would
// index past the loaded candidates whenever FTopK exceeds the vocabulary.
function TNNetSamplerTopK.DrawFromWindow(): integer;
begin
  if FCount > 0
    then Result := FTokenArr[Random(FCount)].Token
    else Result := 0; // defensive: empty distribution
end;

function TNNetSamplerTopK.GetToken(Origin: TNNetVolume): integer;
begin
  LoadCandidates(Origin);
  // The draw is uniform over the window, so the order inside it is irrelevant
  // and the prefix sort is pure waste here.
  SelectTopCandidates(FTopK, {Sorted=}False);
  Result := DrawFromWindow();
end;

function TNNetSamplerTopK.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  LoadCandidatesOnPixel(Origin, PixelX, PixelY);
  SelectTopCandidates(FTopK, {Sorted=}False);
  Result := DrawFromWindow();
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
  if FCount = 0 then
  begin
    Result := 0; // defensive: empty distribution
    exit;
  end;
  // FTokenArr is sorted DESCENDING, so [0..KeptCount-1] are the top-K tokens.
  KeptCount := FTopK;
  if (KeptCount <= 0) or (KeptCount > FCount) then
    KeptCount := FCount; // <=0 or >=candidate count => whole window
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
  LoadCandidates(Origin);
  // FTopK <= 0 keeps its documented "whole row" meaning, so only a positive
  // K may pre-truncate the window.
  if FTopK > 0 then SelectTopCandidates(FTopK) else SortTokenArray();
  Result := SampleFromSorted();
end;

function TNNetSamplerWeightedTopK.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  LoadCandidatesOnPixel(Origin, PixelX, PixelY);
  if FTopK > 0 then SelectTopCandidates(FTopK) else SortTokenArray();
  Result := SampleFromSorted();
end;

{ TNNetSamplerTypical }

constructor TNNetSamplerTypical.Create(Mass: TNeuralFloat);
begin
  inherited Create();
  FMass := Mass;
end;

function TNNetSamplerTypical.SampleTypical(): integer;
const
  // Below this window size a full sort is already cheap, so skip the
  // partial-selection machinery entirely.
  csTypicalAdaptiveMin = 1024;
  // First guess at the typical-set width. Over a peaked next-token row the
  // set that first reaches FMass is far narrower than this; when it is not,
  // the retry below pays one full sort and is still correct.
  csTypicalAdaptiveK = 256;
var
  Entropy, P, LogP, KeptSum, Roll, Cumulative: TNeuralFloat;
  Dist: array of TNeuralFloat; // |surprise - entropy| per FTokenArr entry
  Order: array of integer;     // FTokenArr indices sorted by ascending Dist
  I, KeptCount, KeptCountM1, N, NM1, Limit: integer;
  Truncated: boolean;
begin
  N := FCount;
  if N = 0 then
  begin
    Result := 0; // defensive: empty distribution
    exit;
  end;
  NM1 := N - 1;
  // Reuse the persistent scratch fields; resize only when the vocab size
  // changes (rule #17: amortized, no per-call heap allocation).
  if Length(FDist) <> N then SetLength(FDist, N);
  if Length(FOrder) <> N then SetLength(FOrder, N);
  Dist := FDist;   // reference share (refcount bump only, no new buffer)
  Order := FOrder;
  // Conditional (Shannon) entropy of the row, in nats. log p is stashed in
  // Dist on the way through so the distance pass below needs no second log.
  Entropy := 0;
  for I := 0 to NM1 do
  begin
    P := FTokenArr[I].Score;
    if P > 0 then
    begin
      LogP := pcr_logf(P);
      Dist[I] := LogP;
      Entropy := Entropy - P * LogP;
    end
    else Dist[I] := -1e30; // p = 0 => surprise +infinite
    Order[I] := I;
  end;
  // Per-token distance |(-log p) - H|: surprise is -LogP, so this is purely
  // arithmetic over the cached logs.
  for I := 0 to NM1 do
    Dist[I] := Abs(-Dist[I] - Entropy);
  // Sorting the whole vocabulary to consume a prefix of a few hundred entries
  // dominates this sampler, so select a bounded prefix first (the adaptive
  // scheme TNNetSamplerTopP already uses) and widen only if the mass was not
  // reached. Both branches sort in place over the existing index buffer, so
  // there is still no heap allocation (rule #17).
  Truncated := N > csTypicalAdaptiveMin;
  if Truncated then
  begin
    PartialSelectOrderByDist(Order, Dist, N, csTypicalAdaptiveK);
    Limit := csTypicalAdaptiveK - 1;
  end
  else Limit := NM1;
  if Limit > 0 then QuickSortOrderByDist(Order, Dist, 0, Limit);
  // Smallest prefix (by ascending distance) whose cumulative mass reaches FMass.
  repeat
    KeptCount := 0;
    KeptSum := 0;
    for I := 0 to Limit do
    begin
      Inc(KeptCount);
      KeptSum := KeptSum + FTokenArr[Order[I]].Score;
      if KeptSum >= FMass then Break;
    end;
    if (KeptSum >= FMass) or (not Truncated) then Break;
    // The prefix did not hold FMass: re-arm the identity order over the whole
    // row and sort it in full, exactly once.
    for I := 0 to NM1 do Order[I] := I;
    QuickSortOrderByDist(Order, Dist, 0, NM1);
    Truncated := false;
    Limit := NM1;
  until false;
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
  LoadCandidates(Origin);
  Result := SampleTypical();
end;

function TNNetSamplerTypical.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  LoadCandidatesOnPixel(Origin, PixelX, PixelY);
  Result := SampleTypical();
end;

{ TNNetSamplerMirostat }

const
  // 2^Mu in the Mirostat v1 closed form is exp(Mu * ln 2).
  csLn2 = 0.693147180559945309417;
  // Highest loop index of the v1 Zipf fit: the fit window is capped at
  // csMirostatMaxFitIdx + 2 candidates and the regression pairs candidate i
  // with i + 1, so i never exceeds csMirostatMaxFitIdx.
  csMirostatMaxFitIdx = 98;

var
  // ln((i+2)/(i+1)) for i in [0, csMirostatMaxFitIdx]. The regressor depends
  // only on the loop index, never on the logits, so it is tabulated once at
  // unit start-up instead of recomputed on every sampled token.
  vMirostatLogRank: array[0..csMirostatMaxFitIdx] of TNeuralFloat;

procedure BuildMirostatLogRankTable();
var
  I: integer;
begin
  for I := 0 to csMirostatMaxFitIdx do
    vMirostatLogRank[I] := pcr_logf((I + 2) / (I + 1));
end;

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
  KeptSum, Roll, Cumulative, P, Surprise, SurpriseCut: TNeuralFloat;
  SumLogP, SumLogRank, SumLogPLogRank, SumLogRankSq, LogRank, LogP: TNeuralFloat;
  S, Epsilon, KFloat, ChosenScore: TNeuralFloat;
  I, KeptCount, KeptCountM1, N, NM1, NumFit, NumFitM2, K: integer;
begin
  N := FCount;
  if N = 0 then
  begin
    Result := 0; // defensive
    exit;
  end;
  NM1 := N - 1;
  // FTokenArr is sorted DESCENDING: [0] is the max probability.
  if FVersion = mvV2 then
  begin
    // v2: keep every token with surprise -log p <= Mu. Rule #14: that cut is
    // exactly p >= exp(-Mu) (exp is strictly increasing), so one exp on the
    // pre-image replaces a log per candidate - and a typical Tau keeps
    // thousands of candidates. A huge Mu underflows the cut to ~0 and keeps
    // everything; a very negative Mu overflows it to +Inf, keeps nothing, and
    // falls through to the "always keep the most-likely token" guard below.
    SurpriseCut := NeuralExp(-FMu);
    KeptCount := 0;
    KeptSum := 0;
    for I := 0 to NM1 do
    begin
      P := FTokenArr[I].Score;
      if P <= 0 then Break; // descending: nothing later is larger
      if P >= SurpriseCut then
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
    // fit on the head (paper uses ~100); the cap is what bounds the
    // vMirostatLogRank lookup below.
    if NumFit > csMirostatMaxFitIdx + 2 then NumFit := csMirostatMaxFitIdx + 2;
    SumLogP := 0; SumLogRank := 0; SumLogPLogRank := 0; SumLogRankSq := 0;
    K := 0;
    NumFitM2 := NumFit - 2;
    for I := 0 to NumFitM2 do
    begin
      P := FTokenArr[I].Score;
      if (P <= 0) or (FTokenArr[I + 1].Score <= 0) then Break;
      // t_i = log(p_i / p_{i+1}) regressed on log((i+2)/(i+1)) gives s.
      LogP := pcr_logf(P / FTokenArr[I + 1].Score);
      LogRank := vMirostatLogRank[I];
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
      KFloat := NeuralExp(FMu)            // s ~ 1 limit
    else
      KFloat := NeuralExp( pcr_logf( (Epsilon * NeuralExp(FMu * csLn2)) /
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
  LoadCandidates(Origin);
  SortTokenArray();
  Result := SampleAndUpdate();
end;

function TNNetSamplerMirostat.GetTokenOnPixel(Origin: TNNetVolume; PixelX,
  PixelY: integer): integer;
begin
  LoadCandidatesOnPixel(Origin, PixelX, PixelY);
  SortTokenArray();
  Result := SampleAndUpdate();
end;

{ TNNetSamplerBase }

// Hoare-partition quickselect: permutes A[0..Count-1] so that the K highest
// Scores occupy A[0..K-1] (in no particular order). Average O(Count); the
// median-of-three pivot keeps the already-descending and all-equal inputs
// (a post-softmax row that a previous stage sorted, or a uniform one) off
// the quadratic path. Coded by Claude (AI).
procedure PartialSelectTokenArray(var A: TNNetTokenArray; Count, K: integer);
var
  Lo, Hi, I, J, Mid, Bound: integer;
  Pivot: TNeuralFloat;
  T: TNNetToken;
begin
  if (K <= 0) or (K >= Count) then exit;
  Lo := 0;
  Hi := Count - 1;
  Bound := K - 1;
  while Lo < Hi do
  begin
    // Median of A[Lo], A[Mid], A[Hi] as the pivot value.
    Mid := (Lo + Hi) shr 1;
    if A[Mid].Score > A[Lo].Score then
    begin
      T := A[Mid]; A[Mid] := A[Lo]; A[Lo] := T;
    end;
    if A[Hi].Score > A[Lo].Score then
    begin
      T := A[Hi]; A[Hi] := A[Lo]; A[Lo] := T;
    end;
    if A[Mid].Score > A[Hi].Score then
    begin
      T := A[Mid]; A[Mid] := A[Hi]; A[Hi] := T;
    end;
    Pivot := A[Mid].Score;
    I := Lo;
    J := Hi;
    repeat
      while A[I].Score > Pivot do Inc(I);
      while A[J].Score < Pivot do Dec(J);
      if I <= J then
      begin
        T := A[I]; A[I] := A[J]; A[J] := T;
        Inc(I);
        Dec(J);
      end;
    until I > J;
    // Recurse into the side that still contains the K-th boundary; when the
    // boundary falls inside the equal-to-pivot gap the split is exact.
    if Bound <= J then Hi := J
    else if Bound >= I then Lo := I
    else Break;
  end;
end;

procedure TNNetSamplerBase.LoadCandidates(Origin: TNNetVolume);
begin
  Origin.GetTokenArray(FTokenArr);
  FCount := Length(FTokenArr);
  FSorted := false;
end;

procedure TNNetSamplerBase.LoadCandidatesOnPixel(Origin: TNNetVolume;
  PixelX, PixelY: integer);
begin
  Origin.GetTokenArrayOnPixel(FTokenArr, PixelX, PixelY);
  FCount := Length(FTokenArr);
  FSorted := false;
end;

procedure TNNetSamplerBase.SortTokenArray;
begin
  if FSorted then exit;
  if FCount > 1 then QuickSortTokenArray(FTokenArr, 0, FCount - 1);
  FSorted := true;
end;

procedure TNNetSamplerBase.SelectTopCandidates(K: integer; Sorted: boolean);
begin
  if K <= 0 then K := 1;
  if K >= FCount then
  begin
    // Nothing to truncate - the caller wants the whole window.
    if Sorted then SortTokenArray();
    exit;
  end;
  if FSorted then
  begin
    // Already descending: the top K are exactly the current prefix.
    FCount := K;
    exit;
  end;
  PartialSelectTokenArray(FTokenArr, FCount, K);
  FCount := K;
  if Sorted then SortTokenArray();
end;

procedure TNNetSamplerBase.RestoreFullWindowSorted();
begin
  // PartialSelectTokenArray only PERMUTES the array, so the entries beyond
  // the truncated window are still the rest of the vocabulary and re-arming
  // the full length is sound - no reload from the volume needed.
  FCount := Length(FTokenArr);
  FSorted := false;
  SortTokenArray();
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
  SetLength(FSeen, 0);
  FSeenCount := 0;
end;

destructor TNNetTokenHistoryPenalty.Destroy();
begin
  SetLength(FCounts, 0);
  SetLength(FSeen, 0);
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
  // A zero count means this id is not in FSeen yet: append it before the
  // increment so FSeen holds exactly the ids with FCounts > 0.
  if FCounts[TokenId] = 0 then
  begin
    if FSeenCount >= Length(FSeen) then
      SetLength(FSeen, (FSeenCount + 1) * 2);
    FSeen[FSeenCount] := TokenId;
    Inc(FSeenCount);
  end;
  Inc(FCounts[TokenId]);
end;

procedure TNNetTokenHistoryPenalty.ResetHistory();
var
  I, SeenM1: integer;
begin
  // Only the ids actually registered can be non-zero, so clearing them is
  // proportional to the history rather than to the vocabulary. FSeen itself
  // keeps its capacity for the next sequence (no per-sequence realloc).
  SeenM1 := FSeenCount - 1;
  for I := 0 to SeenM1 do FCounts[FSeen[I]] := 0;
  FSeenCount := 0;
end;

procedure TNNetTokenHistoryPenalty.Apply(Logits: TNNetVolume);
var
  I, Tok, MaxToken, SeenM1, Count: integer;
  Logit: TNeuralFloat;
begin
  // Walk the distinct-id history, not the vocabulary-sized FCounts array.
  // The history can never be larger than the logit volume of interest.
  MaxToken := Logits.Size - 1;
  SeenM1 := FSeenCount - 1;
  for I := 0 to SeenM1 do
  begin
    Tok := FSeen[I];
    if Tok <= MaxToken then
    begin
      Count := FCounts[Tok];
      Logit := Logits.FData[Tok];
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
      Logits.FData[Tok] := Logit;
    end;
  end;
end;

procedure TNNetTokenHistoryPenalty.ApplyToProbabilities(Probs: TNNetVolume);
var
  I, Tok, MaxToken, SeenM1, Count: integer;
  P, Total: TNeuralFloat;
  Changed: boolean;
begin
  // Guaranteed bit-for-bit no-op when every knob is at its default.
  if (FRepetition = 1.0) and (FFrequency = 0.0) and (FPresence = 0.0) then exit;
  Changed := false;
  // Walk the distinct-id history, not the vocabulary-sized FCounts array.
  MaxToken := Probs.Size - 1;
  SeenM1 := FSeenCount - 1;
  for I := 0 to SeenM1 do
  begin
    Tok := FSeen[I];
    if Tok <= MaxToken then
    begin
      Count := FCounts[Tok];
      P := Probs.FData[Tok];
      // (a) repetition penalty: p := p^r ("power then renormalize", the
      // probability-domain image of the sign-correct CTRL logit rule -
      // ln p <= 0 always, so the negative branch ln p * r applies).
      if (FRepetition <> 1.0) and (P > 0) then P := pcr_powf(P, FRepetition);
      // (b) frequency + (c) presence: log-space subtraction is a
      // multiplicative exp() factor on the probability.
      if (FFrequency <> 0.0) or (FPresence <> 0.0) then
        P := P * NeuralExp(-(FFrequency * Count + FPresence));
      Probs.FData[Tok] := P;
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
  // Single guarded accessor for FIntegerToStr. The array is sized only by
  // SaveCurrentPosition, so it is empty until that runs, and a sampled token id
  // can exceed the dictionary whenever the net output is wider than the vocab.
  // Generation must not die on either, so an unknown id yields no text.
  if (pInteger >= 0) and (pInteger < Length(FIntegerToStr)) then
  begin
    Result := FIntegerToStr[pInteger];
  end
  else
  begin
    Result := '';
    {$IFDEF DEBUG}
    WriteLn('Token '+IntToStr(pInteger)+' is bigger than dictionary '+IntToStr(Length(FIntegerToStr))+' at IntegerToWord.');
    {$ENDIF}
  end;
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
        FTokenizer.Add(IntegerToWord(WordInteger));
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
  Smp, CS, Cl: TNNetVolume; // #7: bind repeatedly-indexed list elements
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
      Smp := FSample[SampleCount];      // #7: bound once
      ClosestId := GetClusterId(Smp);
      CS := FClusterSums[ClosestId];    // #7: bound once
      CS.Add(Smp);
      CS.IncTag();
      Smp.Tag := ClosestId;
    end;

    for ClusterCount := 0 to MaxClusterCount do
    begin
      CS := FClusterSums[ClusterCount];   // #7: bound once
      Cl := FClusters[ClusterCount];      // #7: bound once
      if CS.Tag > 0 then
      begin
        CS.Divi(CS.Tag);
        if RepositionClusters then
        begin
          Cl.Copy(CS);
        end;
        Cl.Tag := CS.Tag;
      end
      else
      begin
        Cl.Tag := 0;
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
  CurrentDist, MinSqr: TNeuralFloat;
begin
  Result := 0;
  MaxCount := Count - 1;
  if (MaxCount > 0) then
  begin
    // #14: rank on GetDistanceSqr (argmin identical, Sqrt strictly increasing);
    // take the one Sqrt on the winner after the loop. MinSqr<=0 <=> MinDist<=0.
    MinSqr := Original.GetDistanceSqr(Self[0]);
    for I := 1 to MaxCount do
    begin
      CurrentDist := Original.GetDistanceSqr(Self[I]);
      if (CurrentDist < MinSqr) then
      begin
        Result := I;
        MinSqr := CurrentDist;
      end;
      if MinSqr <= 0 then Break;
    end;
    if MinSqr > 0 then MinDist := Sqrt(MinSqr) else MinDist := 0;
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
  Vol: TNNetVolume;
  Pos, Stride: integer;
begin
  if (Count>0) then
  begin
    MaxVolume := Count - 1;
    MaxElement := Self[0].Size - 1;
    Stride := MaxVolume + 1; // volume count == output interleave stride

    // Interleaved layout is V[CountElement*Stride + CountVolume]. Iterating
    // volume-outer binds each list element once (#7, no Items[] getter per cell)
    // and carries the strided write offset by addition (#12).
    for CountVolume := 0 to MaxVolume do
    begin
      Vol := Self[CountVolume];
      Pos := CountVolume;
      for CountElement := 0 to MaxElement do
      begin
        V.FData[Pos] := Vol.FData[CountElement];
        Inc(Pos, Stride);
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

    SaltBase := GetRawPos(SaltPosX, SaltPosY);
    PepperBase := GetRawPos(PepperPosX, PepperPosY);
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
  RawPos, RowStride, colPos: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step

  colPos := pDepth; // #12: carried GetRawPos(CntX, 0, pDepth)
  for CntX := 0 to MaxX do
  begin
    RawPos := colPos;
    for CntY := 0 to MaxY do
    begin
      {$IFDEF FPC}
      FData[RawPos] += Value;
      {$ELSE}
      FData[RawPos] := FData[RawPos] + Value;
      {$ENDIF}
      Inc(RawPos, RowStride);
    end;
    Inc(colPos, FDepth);
  end;
end;

procedure TVolume.AddAtDepth(pDepth: integer; Original: TVolume);
var
  CntX, CntY: integer;
  MaxX, MaxY: integer;
  RawPos, RowStride, colPos: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  if Self.Size = Original.Size then
  begin
    RowStride := FSizeX * FDepth; // per-CntY step; same shape indexes both
    colPos := pDepth; // #12: carried GetRawPos(CntX, 0, pDepth)
    for CntX := 0 to MaxX do
    begin
      RawPos := colPos;
      for CntY := 0 to MaxY do
      begin
        {$IFDEF FPC}
        FData[RawPos] += Original.FData[RawPos];
        {$ELSE}
        FData[RawPos] := FData[RawPos] + Original.FData[RawPos];
        {$ENDIF}
        Inc(RawPos, RowStride);
      end;
      Inc(colPos, FDepth);
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
  RawPos, SrcPos, RowStride, colPos, srcColPos: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  if Self.Size = Original.Size then
  begin
    RowStride := FSizeX * FDepth; // per-CntY step (same shape for both volumes)
    colPos := ToDepth;      // #12: carried GetRawPos(CntX, 0, ToDepth)
    srcColPos := FromDepth; // #12: carried GetRawPos(CntX, 0, FromDepth)
    for CntX := 0 to MaxX do
    begin
      RawPos := colPos;
      SrcPos := srcColPos;
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
      Inc(colPos, FDepth);
      Inc(srcColPos, FDepth);
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
  DestRowStride, SrcRowStride, SrcRowPos, idx: integer;
begin
  ReSize(Original.Depth, Original.SizeY, Original.SizeX);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  SrcStride := Original.FDepth; // Original X-slot step per CntD
  DestRowStride := FSizeX * FDepth;                  // #12: per-CntY dest step
  SrcRowStride := Original.FSizeX * Original.FDepth;  // #12: per-CntY src step
  if MaxY > 0 then
  begin
    for CntX := 0 to MaxX do
    begin
      DestBase := CntX * FDepth; // #12: GetRawPos(CntX, 0), carried across CntY
      SrcRowPos := CntX;         // #12: GetRawPos(0, 0, CntX), carried across CntY
      for CntY := 0 to MaxY do
      begin
        SrcPos := SrcRowPos;
        idx := DestBase; // #4: DestBase + CntD carried
        for CntD := 0 to MaxD do
        begin
          FData[idx] := FData[idx] + Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
          Inc(idx);
        end;
        Inc(DestBase, DestRowStride);
        Inc(SrcRowPos, SrcRowStride);
      end;
    end;
  end
  else
  begin
    for CntX := 0 to MaxX do
    begin
      DestBase := GetRawPos(CntX, 0);
      SrcPos := Original.GetRawPos(0, 0, CntX);
      idx := DestBase; // #4: DestBase + CntD carried
      for CntD := 0 to MaxD do
      begin
          FData[idx] := FData[idx] + Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
          Inc(idx);
      end;
    end;
  end;
end;

procedure TVolume.AddTransposingYD(Original: TVolume);
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  DestBase, SrcPos, SrcStride: integer;
  DestRowStride, OrigDepth, SrcRowPos, idx: integer;
begin
  ReSize(Original.SizeX, Original.Depth, Original.SizeY);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  SrcStride := Original.FSizeX * Original.FDepth; // Original Y-slot step per CntD
  DestRowStride := FSizeX * FDepth; // #12: per-CntY dest step
  OrigDepth := Original.FDepth;
  if MaxX > 0 then
  begin
    for CntX := 0 to MaxX do
    begin
      DestBase := CntX * FDepth;     // #12: GetRawPos(CntX, 0), carried across CntY
      SrcRowPos := CntX * OrigDepth; // #12: GetRawPos(CntX, 0, 0), carried across CntY
      for CntY := 0 to MaxY do
      begin
        SrcPos := SrcRowPos;
        idx := DestBase; // #4: DestBase + CntD carried
        for CntD := 0 to MaxD do
        begin
          FData[idx] := FData[idx] + Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
          Inc(idx);
        end;
        Inc(DestBase, DestRowStride);
        Inc(SrcRowPos); // per-CntY src step is 1 (Original depth axis)
      end;
    end;
  end
  else
  begin
    DestBase := 0;  // #12: GetRawPos(0, 0) seed, carried across CntY
    SrcRowPos := 0; // #12: GetRawPos(0, 0, 0) seed, carried across CntY
    for CntY := 0 to MaxY do
    begin
      SrcPos := SrcRowPos;
      idx := DestBase; // #4: DestBase + CntD carried
      for CntD := 0 to MaxD do
      begin
        FData[idx] := FData[idx] + Original.FData[SrcPos];
        Inc(SrcPos, SrcStride);
        Inc(idx);
      end;
      Inc(DestBase, DestRowStride);
      Inc(SrcRowPos);
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
  RawPos, SrcPos, RowStride, colPos, srcColPos: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  if Self.Size = Original.Size then
  begin
    RowStride := FSizeX * FDepth; // per-CntY step (same shape for both volumes)
    colPos := ToDepth;      // #12: carried GetRawPos(CntX, 0, ToDepth)
    srcColPos := FromDepth; // #12: carried GetRawPos(CntX, 0, FromDepth)
    for CntX := 0 to MaxX do
    begin
      RawPos := colPos;
      SrcPos := srcColPos;
      for CntY := 0 to MaxY do
      begin
        FData[RawPos] := Original.FData[SrcPos];
        Inc(RawPos, RowStride);
        Inc(SrcPos, RowStride);
      end;
      Inc(colPos, FDepth);
      Inc(srcColPos, FDepth);
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
  RowBytesA, RowBytesB, ADepth: integer;
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
    RowBytesA := A.FDepth * csNeuralFloatSize; // #5: invariant byte count
    for I := 0 to ASizeXM1 do
    begin
      for J := 0 to ASizeYM1 do
      begin
        SelfBase := GetRawPos(I, J);
        SrcBase := A.GetRawPos(I, J);
        Move(A.FData[SrcBase], FData[SelfBase], RowBytesA);
      end;
    end;
  end;

  if (B.FDepth>0) and (B.FSizeX > 0) and (B.FSizeY > 0) then
  begin
    BSizeXM1 := B.FSizeX - 1;
    BSizeYM1 := B.FSizeY - 1;
    BDepthM1 := B.FDepth - 1;
    RowBytesB := B.FDepth * csNeuralFloatSize; // #5: invariant byte count
    ADepth := A.FDepth;
    for I := 0 to BSizeXM1 do
    begin
      for J := 0 to BSizeYM1 do
      begin
        SelfBase := GetRawPos(I, J);
        SrcBase := B.GetRawPos(I, J);
        Move(B.FData[SrcBase], FData[SelfBase + ADepth], RowBytesB);
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
  RawPos, RowStride, colPos: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step

  colPos := pDepth; // #12: carried GetRawPos(CntX, 0, pDepth)
  for CntX := 0 to MaxX do
  begin
    RawPos := colPos;
    for CntY := 0 to MaxY do
    begin
      {$IFDEF FPC}
      FData[RawPos] *= Value;
      {$ELSE}
      FData[RawPos] := FData[RawPos] * Value;
      {$ENDIF}
      Inc(RawPos, RowStride);
    end;
    Inc(colPos, FDepth);
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
  RawPos, RowStride, colPos: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step

  colPos := pDepth; // #12: carried GetRawPos(CntX, 0, pDepth)
  for CntX := 0 to MaxX do
  begin
    RawPos := colPos;
    for CntY := 0 to MaxY do
    begin
      FData[RawPos] := Value;
      Inc(RawPos, RowStride);
    end;
    Inc(colPos, FDepth);
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
  v: T;
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
      v := Original.FData[OriginalCnt]; // #4: read source once
      if v > 0.0
        then FData[OriginalCnt] := v
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
  SelfRowStride, OrigRowStride: integer;
begin
  Resize(Original.SizeX, Original.SizeY, Length(aChannels));

  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  SelfRowStride := FSizeX * FDepth;                  // #12: per-Y self step
  OrigRowStride := Original.FSizeX * Original.FDepth; // #12: per-Y src step

  for X := 0 to MaxX do
  begin
    SelfBase := GetRawPos(X, 0);
    OrigBase := Original.GetRawPos(X, 0);
    for Y := 0 to MaxY do
    begin
      OutputDepth := 0;
      for InputDepth in aChannels do
      begin
        FData[SelfBase + OutputDepth] := Original.FData[OrigBase + InputDepth];
        Inc(OutputDepth);
      end;
      Inc(SelfBase, SelfRowStride);
      Inc(OrigBase, OrigRowStride);
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
      if Original.Size and 7 = 0 then
      begin
        Self.ReSize(Original.Size shr 3, 1, 8);
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
    SourceRawPos := Original.GetRawPos(0, CntY);
    DestRawPos := GetRawPos(Padding, CntY + Padding);
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
    SourceRawPos := Original.GetRawPos(0, CntY);
    DestRawPos := GetRawPos(PaddingX, CntY + PaddingY);
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
  DestRowStride, SrcRowStride, SrcColBase: integer;
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
    DestRowStride := FSizeX * FDepth;                  // #12: per-CntY dest step
    SrcRowStride := Original.FSizeX * Original.FDepth;  // #5: invariant per call

    for CntX := 0 to MaxX do
    begin
      OrigPosX := Min(OrigMaxX, Round(CntX * InvRatioX));
      SrcColBase := OrigPosX * Original.FDepth; // #11: invariant across CntY
      RawPostDest := GetRawPos(CntX, 0);        // #12: carried across CntY
      for CntY := 0 to MaxY do
      begin
        OrigPosY := Min(OrigMaxY, Round(CntY * InvRatioY));
        RawPosSource := SrcRowStride * OrigPosY + SrcColBase;
        Move(Original.FData[RawPosSource], FData[RawPostDest], MoveSizeBytes);
        Inc(RawPostDest, DestRowStride);
      end;
    end;
  end;
end;

procedure TVolume.CopyTransposingXD(Original: TVolume);
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  DestBase, SrcPos, SrcStride: integer;
  DestRowStride, SrcRowStride, SrcRowPos, idx: integer;
begin
  ReSize(Original.Depth, Original.SizeY, Original.SizeX);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  SrcStride := Original.FDepth; // Original X-slot step per CntD
  DestRowStride := FSizeX * FDepth;                  // #12: per-CntY dest step
  SrcRowStride := Original.FSizeX * Original.FDepth;  // #12: per-CntY src step
  if MaxY > 0 then
  begin
    for CntX := 0 to MaxX do
    begin
      DestBase := CntX * FDepth; // #12: GetRawPos(CntX, 0), carried across CntY
      SrcRowPos := CntX;         // #12: GetRawPos(0, 0, CntX), carried across CntY
      for CntY := 0 to MaxY do
      begin
        SrcPos := SrcRowPos;
        idx := DestBase;
        for CntD := 0 to MaxD do
        begin
          FData[idx] := Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
          Inc(idx);
        end;
        Inc(DestBase, DestRowStride);
        Inc(SrcRowPos, SrcRowStride);
      end;
    end;
  end
  else
  begin
    for CntX := 0 to MaxX do
    begin
      DestBase := GetRawPos(CntX, 0);
      SrcPos := Original.GetRawPos(0, 0, CntX);
      idx := DestBase;
      for CntD := 0 to MaxD do
      begin
        FData[idx] := Original.FData[SrcPos];
        Inc(SrcPos, SrcStride);
        Inc(idx);
      end;
    end;
  end;
end;

procedure TVolume.CopyTransposingYD(Original: TVolume);
var
  CntX, CntY, CntD: integer;
  MaxX, MaxY, MaxD: integer;
  DestBase, SrcPos, SrcStride: integer;
  DestRowStride, OrigDepth, SrcRowPos, idx: integer;
begin
  ReSize(Original.SizeX, Original.Depth, Original.SizeY);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  SrcStride := Original.FSizeX * Original.FDepth; // Original Y-slot step per CntD
  DestRowStride := FSizeX * FDepth; // #12: per-CntY dest step
  OrigDepth := Original.FDepth;
  if MaxX > 0 then
  begin
    for CntX := 0 to MaxX do
    begin
      DestBase := CntX * FDepth;     // #12: GetRawPos(CntX, 0), carried across CntY
      SrcRowPos := CntX * OrigDepth; // #12: GetRawPos(CntX, 0, 0), carried across CntY
      for CntY := 0 to MaxY do
      begin
        SrcPos := SrcRowPos;
        idx := DestBase;
        for CntD := 0 to MaxD do
        begin
          FData[idx] := Original.FData[SrcPos];
          Inc(SrcPos, SrcStride);
          Inc(idx);
        end;
        Inc(DestBase, DestRowStride);
        Inc(SrcRowPos); // per-CntY src step is 1 (Original depth axis)
      end;
    end;
  end
  else
  begin
    DestBase := 0;  // #12: GetRawPos(0, 0) seed, carried across CntY
    SrcRowPos := 0; // #12: GetRawPos(0, 0, 0) seed, carried across CntY
    for CntY := 0 to MaxY do
    begin
      SrcPos := SrcRowPos;
      idx := DestBase;
      for CntD := 0 to MaxD do
      begin
        FData[idx] := Original.FData[SrcPos];
        Inc(SrcPos, SrcStride);
        Inc(idx);
      end;
      Inc(DestBase, DestRowStride);
      Inc(SrcRowPos);
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
  RawPos, RowStride, colPos: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step
  Result := 0;
  colPos := pDepth; // #12: carried GetRawPos(CntX, 0, pDepth)
  for CntX := 0 to MaxX do
  begin
    RawPos := colPos;
    for CntY := 0 to MaxY do
    begin
      Result := Result + FData[RawPos];
      Inc(RawPos, RowStride);
    end;
    Inc(colPos, FDepth);
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
  v: T;
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
        v := FData[I]; // #4: read element once
        if v > Result then
        begin
          Result := v;
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
  v: T;
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
        v := FData[I]; // #4: read element once
        if v < Result then
        begin
          Result := v;
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
  RawPos, RowStride, colPos: integer;
begin
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  RowStride := FSizeX * FDepth; // per-CntY step

  pMin := Self.Data[0, 0, pDepth];
  pMax := Self.Data[0, 0, pDepth];

  colPos := pDepth; // #12: carried GetRawPos(CntX, 0, pDepth)
  for CntX := 0 to MaxX do
  begin
    RawPos := colPos;
    for CntY := 0 to MaxY do
    begin
      Aux := FData[RawPos];

      if Aux < pMin
      then pMin := Aux
      else if Aux > pMax then pMax := Aux;
      Inc(RawPos, RowStride);
    end;
    Inc(colPos, FDepth);
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
    // Eight accumulations per iteration into the one left-associated chain, so
    // the summation order - and therefore the result - is exactly the
    // element-at-a-time loop's, at a fraction of the loop overhead. This is the
    // fallback used on every build without AVXANY (AArch64, PowerPC, Delphi).
    I := 1;
    while I <= vHigh - 7 do
    begin
      Result := Result + FData[I] + FData[I+1] + FData[I+2] + FData[I+3] +
        FData[I+4] + FData[I+5] + FData[I+6] + FData[I+7];
      Inc(I, 8);
    end;
    while I <= vHigh do
    begin
      Result := Result + FData[I];
      Inc(I);
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
  v: T;
begin
  vSum := 0;
  if FSize > 0 then
  begin
    vHigh := FSize - 1;
    for I := 0 to vHigh do
    begin
      v := FData[I];
      if v > 0 then // To avoid log(0) which is undefined
        vSum := vSum + (v * pcr_log2f(v));
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
      Result := Result - Tgt * pcr_logf(P);
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
  RowStride: integer;
  MaxY, MaxD: integer;
  CountX, CountY, CountD: integer;
  Aux: TNeuralFloat;
begin
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;

  iTo := (FSizeX shr 1) - 1;
  iFrom := 0;
  RowStride := FSizeX * FDepth;

  for CountX := iFrom to iTo do
  begin
    iBase1 := GetRawPos(CountX, 0);
    iBase2 := GetRawPos(FSizeX-CountX-1, 0);
    for CountY := 0 to MaxY do
    begin
      for CountD := 0 to MaxD do
      begin
        iRawPos1 := iBase1 + CountD;
        iRawPos2 := iBase2 + CountD;
        Aux := FData[iRawPos1];
        FData[iRawPos1] := FData[iRawPos2];
        FData[iRawPos2] := Aux;
      end;
      Inc(iBase1, RowStride);
      Inc(iBase2, RowStride);
    end;
  end;
end;

procedure TVolume.FlipY();
var
  iFrom, iTo: integer;
  iRawPos1, iRawPos2: integer;
  iBase1, iBase2: integer;
  RowStride: integer;
  MaxX, MaxD: integer;
  CountX, CountY, CountD: integer;
  Aux: TNeuralFloat;
begin
  MaxX := FSizeX - 1;
  MaxD := FDepth - 1;

  iTo := (FSizeY shr 1) - 1;
  iFrom := 0;
  RowStride := FSizeX * FDepth;

  for CountX := 0 to MaxX do
  begin
    iBase1 := GetRawPos(CountX, iFrom);
    iBase2 := GetRawPos(CountX, FSizeY-iFrom-1);
    for CountY := iFrom to iTo do
    begin
      for CountD := 0 to MaxD do
      begin
        iRawPos1 := iBase1 + CountD;
        iRawPos2 := iBase2 + CountD;
        Aux := FData[iRawPos1];
        FData[iRawPos1] := FData[iRawPos2];
        FData[iRawPos2] := Aux;
      end;
      Inc(iBase1, RowStride);
      Dec(iBase2, RowStride);
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
  v: T;
begin
  vHigh := High(FData);
  if (vHigh>0) then
  begin
    Result := 0;
    vMax := FData[Result];
    for I := 0 to vHigh do
    begin
      v := FData[I]; // #4: read element once
      if v > vMax then
      begin
        Result := I;
        vMax := v;
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
  vHigh: integer;
  TotalSum: TNeuralFloat;
  MinValue, MaxValue, ShiftedMin: T;
begin
  MaxValue := GetMax();
  MinValue := GetMin();
  // Value the smallest element takes once the max is subtracted off. Reading it
  // before the shift rather than after lets the shift itself be folded into the
  // exponentiation below.
  ShiftedMin := MinValue - MaxValue;

  TotalSum := 0;

  // forces range [-1000,0]
  if ShiftedMin <> 0 then
  begin
    vHigh := High(FData);
    if ShiftedMin < -1000 then
    begin
      // The rescale has to see the shifted values, so here the subtraction is a
      // pass of its own and the fused kernel exponentiates with a zero shift.
      Sub(MaxValue);
      Mul( -1000/ShiftedMin );
      TotalSum := TNNetVolume.ExpShiftSum(Addr(FData[0]), Addr(FData[0]), 0, vHigh + 1);
    end
    else
      // Everything already sits in [ShiftedMin, 0] once the max comes off, well
      // inside exp's safe range, so subtract-and-exponentiate-and-total is one
      // fused pass over the buffer instead of three.
      TotalSum := TNNetVolume.ExpShiftSum(Addr(FData[0]), Addr(FData[0]), MaxValue, vHigh + 1);

    if TotalSum > 0 then
    begin
      Divi(TotalSum);
    end;
  end;

  Result := TotalSum;
end;

procedure TVolume.PointwiseSoftMax(NoForward: boolean = false);
var
  StartPointPos: integer;
  MaxX, MaxY, MaxD, FDepthM1, MaxDP1: integer;
  CountX, CountY: integer;
  SpanMax: TNeuralFloat;
  TotalSum: TNeuralFloat;
  RowStride, colBase: integer;
begin
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;
  FDepthM1 := FDepth - 1;
  RowStride := FSizeX * FDepth; // #12: per-CountY GetRawPos step (carried below)

  if MaxD > 0 then
  begin
    // Every (x,y) position owns a contiguous depth span, so the three softmax
    // reductions each run as one vectorized primitive over that span: find the
    // stabilizing max, then exponentiate the shifted span and total it in the
    // single fused pass, then normalize. Subtracting the span max leaves every
    // element at <= 0, which is why no clamp is needed - the exp of anything
    // far below -88 is a hard zero on both the AVX and the scalar path.
    colBase := 0; // #12: carried GetRawPos(CountX, 0)
    for CountX := 0 to MaxX do
    begin
      if NoForward then MaxD := Min(FDepthM1, CountX);
      StartPointPos := colBase;
      for CountY := 0 to MaxY do
      begin
        if NoForward and (MaxD < FDepthM1) then
        begin
          MaxDP1 := MaxD + 1;
          FillChar(FData[StartPointPos + MaxDP1], (FDepthM1 - MaxDP1 + 1) * csNeuralFloatSize, 0);
        end;
        SpanMax := TNNetVolume.MaxValue(Addr(FData[StartPointPos]), MaxD + 1);
        TotalSum := TNNetVolume.ExpShiftSum(Addr(FData[StartPointPos]),
          Addr(FData[StartPointPos]), SpanMax, MaxD + 1);
        if TotalSum > 0 then
          TNNetVolume.Mul(Addr(FData[StartPointPos]), 1.0 / TotalSum, MaxD + 1);
        Inc(StartPointPos, RowStride);
      end;
      Inc(colBase, FDepth);
    end;
  end;
end;

procedure TNNetVolume.PointwiseNorm(pNorms: TNNetVolume = nil);
var
  StartPointPtr: pointer;
  MaxX, MaxY: integer;
  CountX, CountY: integer;
  Modulus, Multiplier: TNeuralFloat;
  RowStride, colBase, pos: integer;
begin
  if Assigned(pNorms) then
  begin
    pNorms.ReSize(SizeX, SizeY, 1);
    pNorms.Fill(1);
  end;
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  RowStride := FSizeX * FDepth; // #12: per-CountY GetRawPos step (carried below)
  colBase := 0;
  for CountX := 0 to MaxX do
  begin
    pos := colBase;
    for CountY := 0 to MaxY do
    begin
      StartPointPtr := GetRawPtr(pos);
      Modulus := Sqrt(DotProduct(StartPointPtr, StartPointPtr, FDepth));
      if Modulus > 0 then
      begin
        Multiplier := 1/Modulus;
        if Assigned(pNorms) then pNorms[CountX, CountY, 0] := Multiplier;
        Mul(StartPointPtr, Multiplier, FDepth);
      end;
      Inc(pos, RowStride);
    end;
    Inc(colBase, FDepth);
  end;
end;

procedure TNNetVolume.PointwiseMul(pNorms: TNNetVolume);
var
  StartPointPtr: pointer;
  MaxX, MaxY: integer;
  CountX, CountY: integer;
  Modulus: TNeuralFloat;
  RowStride, colBase, pos: integer;
begin
  if Assigned(pNorms) then pNorms.ReSize(SizeX, SizeY, 1);
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  RowStride := FSizeX * FDepth; // #12: per-CountY GetRawPos step (carried below)
  colBase := 0;
  for CountX := 0 to MaxX do
  begin
    pos := colBase;
    for CountY := 0 to MaxY do
    begin
      StartPointPtr := GetRawPtr(pos);
      Modulus := pNorms[CountX, CountY, 0];
      if Modulus <> 1 then
      begin
        Mul(StartPointPtr, Modulus, FDepth);
      end;
      Inc(pos, RowStride);
    end;
    Inc(colBase, FDepth);
  end;
end;

// The broadcast-add kernel is assembled only in the AVX64 block below, so a
// 32-bit AVX2 build takes the scalar path like every non-AVX build.
{$IFDEF AVX2}{$IFDEF AVX64}{$DEFINE HASAVXADDSCALAR}{$ENDIF}{$ENDIF}
{$IFDEF HASAVXADDSCALAR}
// AVXAddScalar is defined later in this file inside the AVX64 asm block;
// forward-declare it so AddScalar can dispatch to it from here.
procedure AVXAddScalar(PtrA: TNeuralFloatArrPtr; Value: TNeuralFloat;
  NumElements: integer); forward;
{$ENDIF}
class procedure TNNetVolume.AddScalar(PtrA: TNeuralFloatArrPtr;
  Value: TNeuralFloat; pSize: integer);
{$IFDEF HASAVXADDSCALAR}
begin
  if pSize <= 0 then exit;
  AVXAddScalar(PtrA, Value, pSize);
end;
{$ELSE}
var
  I, pSizeM1: integer;
begin
  if pSize <= 0 then exit;
  pSizeM1 := pSize - 1;
  for I := 0 to pSizeM1 do
    PtrA^[I] := PtrA^[I] + Value;
end;
{$ENDIF}
{$UNDEF HASAVXADDSCALAR}

class procedure TNNetVolume.Exp(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
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

// The fused exp-shift-sum kernel is assembled only in the AVX64 block below, so
// a 32-bit AVX2 build has to take the scalar path like every non-AVX build.
{$IFDEF AVX2}{$IFDEF AVX64}{$DEFINE HASAVXEXPSHIFTSUM}{$ENDIF}{$ENDIF}
{$IFDEF HASAVXEXPSHIFTSUM}
// AVXExpShiftSum is defined later in this file inside the AVX64 asm block;
// forward-declare it so ExpShiftSum can dispatch to it from here.
function AVXExpShiftSum(pDst, pSrc: TNeuralFloatArrPtr; Shift: TNeuralFloat;
  NumElements: integer): TNeuralFloat; forward;
{$ENDIF}
class function TNNetVolume.ExpShiftSum(pDst, pSrc: TNeuralFloatArrPtr;
  Shift: TNeuralFloat; N: integer): TNeuralFloat;
{$IFDEF HASAVXEXPSHIFTSUM}
begin
  if N <= 0 then exit(0);
  Result := AVXExpShiftSum(pDst, pSrc, Shift, N);
end;
{$ELSE}
var
  I, NM1: integer;
  V, Sum: TNeuralFloat;
begin
  if N <= 0 then exit(0);
  NM1 := N - 1;
  Sum := 0;
  for I := 0 to NM1 do
  begin
    V := NeuralExp(pSrc^[I] - Shift);
    pDst^[I] := V;
    Sum := Sum + V;
  end;
  Result := Sum;
end;
{$ENDIF}
{$UNDEF HASAVXEXPSHIFTSUM}

// AVXGetMaxPos is assembled only in the AVX64 block, so a 32-bit AVX2 build
// takes the scalar path like every non-AVX build.
{$IFDEF AVX2}{$IFDEF AVX64}{$DEFINE HASAVXMAXPOS}{$ENDIF}{$ENDIF}
{$IFDEF HASAVXMAXPOS}
// AVXGetMaxPos is defined later in this file inside the AVX64 asm block;
// forward-declare it so MaxPos can dispatch to it from here.
function AVXGetMaxPos(PtrA: TNeuralFloatArrPtr; NumElements: integer;
  out Pos: integer): Single; forward;
{$ENDIF}
class function TNNetVolume.MaxPos(pSrc: TNeuralFloatArrPtr; N: integer;
  out Pos: integer): TNeuralFloat;
{$IFNDEF HASAVXMAXPOS}
var
  I, NM1: integer;
  V: TNeuralFloat;
{$ENDIF}
begin
  if N <= 0 then
  begin
    Pos := -1;
    exit(0);
  end;
  {$IFDEF HASAVXMAXPOS}
  Result := AVXGetMaxPos(pSrc, N, Pos);
  {$ELSE}
  Result := pSrc^[0];
  Pos := 0;
  NM1 := N - 1;
  for I := 1 to NM1 do
  begin
    V := pSrc^[I];
    if V > Result then
    begin
      Result := V;
      Pos := I;
    end;
  end;
  {$ENDIF}
end;
{$UNDEF HASAVXMAXPOS}

class function TNNetVolume.MaxValue(pSrc: TNeuralFloatArrPtr; N: integer): TNeuralFloat;
var
  Pos: integer;
begin
  Result := MaxPos(pSrc, N, Pos);
end;

class procedure TNNetVolume.Sigmoid(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
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
  Exp(pDst, pDst, N);
  for I := 0 to NM1 do
  begin
    S := pDst^[I]; // S = exp(-x)
    pDst^[I] := 1 / (1 + S);
  end;
end;

class procedure TNNetVolume.Tanh(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
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
  Exp(pDst, pDst, N);
  for I := 0 to NM1 do
  begin
    E := pDst^[I]; // E = exp(-2x) in [exp(-88), exp(88)]
    pDst^[I] := (1 - E) / (1 + E);
  end;
end;

{$IFDEF AVX2}
// AVXAdamDelta is defined later in this section under {$IFDEF AVX64};
// forward-declare it so AdamDelta can call it here.
procedure AVXAdamDelta(PtrDelta, PtrM, PtrV: TNeuralFloatArrPtr;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR: TNeuralFloat;
  NumElements: integer); forward;
{$ENDIF}
class procedure TNNetVolume.AdamDelta(PtrDelta, PtrM, PtrV: TNeuralFloatArrPtr;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR: TNeuralFloat;
  N: integer);
{$IFNDEF AVX2}
var
  I: integer;
  g, m, v, t1, t2: TNeuralFloat;
{$ENDIF}
begin
  if N <= 0 then exit;
  {$IFDEF AVX2}
  AVXAdamDelta(PtrDelta, PtrM, PtrV,
    Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR, N);
  {$ELSE}
  // Every intermediate lands in a TNeuralFloat before it is used again, so each
  // operation rounds exactly once - the rounding sequence the composed
  // Copy/Mul/MulMulAdd/VSqrt/Add/Fill/Divi form performs, and the one the AVX
  // kernel performs.
  for I := 0 to N - 1 do
  begin
    g  := PtrDelta^[I];
    t1 := Beta1 * PtrM^[I];
    t2 := OmBeta1 * g;
    m  := t1 + t2;
    t1 := g * g;
    t2 := OmBeta2 * t1;
    t1 := Beta2 * PtrV^[I];
    v  := t2 + t1;
    PtrM^[I] := m;
    PtrV^[I] := v;
    t1 := v * InvOmB2D;
    t1 := Sqrt(t1);
    t1 := t1 + Epsilon;
    t2 := kLR * m;
    PtrDelta^[I] := t2 / t1;
  end;
  {$ENDIF}
end;

{$IFDEF AVXANY}
// AVXCopyRelu (dst := max(src,0)) is defined later in this section under
// {$IFDEF AVXANY}; forward-declare it so Relu can call it here.
procedure AVXCopyRelu(PtrA, PtrB: TNeuralFloatArrPtr; NumElements: integer); forward;
{$ENDIF}
class procedure TNNetVolume.Relu(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
{$IFNDEF AVXANY}
var
  I: integer;
{$ENDIF}
begin
  // dst[i] := max(src[i], 0). On an AVX build this is the vectorized
  // AVXCopyRelu kernel; otherwise a plain scalar relu-copy loop. Bit-exact
  // either way (no float arithmetic, just a compare-and-select).
  if N <= 0 then exit;
  {$IFDEF AVXANY}
  AVXCopyRelu(pDst, pSrc, N);
  {$ELSE}
  for I := 0 to N - 1 do
    if pSrc^[I] > 0 then pDst^[I] := pSrc^[I] else pDst^[I] := 0;
  {$ENDIF}
end;

class procedure TNNetVolume.ReluGateMask(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
{$IFNDEF AVX2}
var
  I: integer;
{$ELSE}
{$IFNDEF AVX64}
var
  I: integer;
{$ENDIF}
{$ENDIF}
begin
  if N <= 0 then exit;
  {$IFDEF AVX2}
  {$IFDEF AVX64}
  AVXReluGateMask(pDst, pSrc, N);
  {$ELSE}
  for I := 0 to N - 1 do
    if pSrc^[I] >= 0 then pDst^[I] := 1 else pDst^[I] := 0;
  {$ENDIF}
  {$ELSE}
  for I := 0 to N - 1 do
    if pSrc^[I] >= 0 then pDst^[I] := 1 else pDst^[I] := 0;
  {$ENDIF}
end;

class procedure TNNetVolume.LeakyRelu(pDst, pSrc: TNeuralFloatArrPtr;
  Slope: TNeuralFloat; N: integer);
{$IFNDEF AVX2}
var
  I: integer;
{$ELSE}
{$IFNDEF AVX64}
var
  I: integer;
{$ENDIF}
{$ENDIF}
begin
  if N <= 0 then exit;
  {$IFDEF AVX2}
  {$IFDEF AVX64}
  AVXLeakyRelu(pDst, pSrc, Slope, N);
  {$ELSE}
  for I := 0 to N - 1 do
    if pSrc^[I] >= 0 then pDst^[I] := pSrc^[I] else pDst^[I] := Slope * pSrc^[I];
  {$ENDIF}
  {$ELSE}
  for I := 0 to N - 1 do
    if pSrc^[I] >= 0 then pDst^[I] := pSrc^[I] else pDst^[I] := Slope * pSrc^[I];
  {$ENDIF}
end;

class procedure TNNetVolume.Erf(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
const
  // Abramowitz & Stegun 7.1.26 coefficients (|err| < 1.5e-7).
  cErfA1: TNeuralFloat =  0.254829592;
  cErfA2: TNeuralFloat = -0.284496736;
  cErfA3: TNeuralFloat =  1.421413741;
  cErfA4: TNeuralFloat = -1.453152027;
  cErfA5: TNeuralFloat =  1.061405429;
  cErfP:  TNeuralFloat =  0.3275911;
var
  I, ChunkStart, ChunkLen, ChunkLenM1, J: integer;
  X, AX, T, Poly, E: TNeuralFloat;
  ExpBuf: array[0..255] of TNeuralFloat;
begin
  if N <= 0 then exit;
  // erf(x) = sign(x) * (1 - poly(t)*exp(-x^2)), t = 1/(1+p*|x|).
  // exp(-x^2) is produced by a single vectorized Exp pass into a fixed
  // stack scratch buffer (NOT pDst) so that pSrc -- which still holds x for the
  // |x| and sign terms in the finishing pass -- is never clobbered. Hence dst
  // may alias src. The buffer is processed in chunks of at most 256 to stay
  // allocation-free.
  ChunkStart := 0;
  while ChunkStart < N do
  begin
    ChunkLen := N - ChunkStart;
    if ChunkLen > 256 then ChunkLen := 256;
    ChunkLenM1 := ChunkLen - 1;   // #2: hoist the for-bound (both loops)
    for J := 0 to ChunkLenM1 do
    begin
      X := pSrc^[ChunkStart + J];
      ExpBuf[J] := -X * X;
    end;
    Exp(TNeuralFloatArrPtr(@ExpBuf[0]), TNeuralFloatArrPtr(@ExpBuf[0]), ChunkLen);
    for J := 0 to ChunkLenM1 do
    begin
      I := ChunkStart + J;
      X := pSrc^[I];
      if X < 0 then AX := -X else AX := X;
      T := 1 / (1 + cErfP * AX);
      // Horner: (((a5*t + a4)*t + a3)*t + a2)*t + a1) * t
      Poly := ((((cErfA5 * T + cErfA4) * T + cErfA3) * T + cErfA2) * T + cErfA1) * T;
      E := ExpBuf[J];
      if X < 0 then
        pDst^[I] := -(1 - Poly * E)
      else
        pDst^[I] := 1 - Poly * E;
    end;
    Inc(ChunkStart, ChunkLen);
  end;
end;

class procedure TNNetVolume.Sinh(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
var
  I, NM1: integer;
  X, E: TNeuralFloat;
begin
  if N <= 0 then exit;
  NM1 := N - 1;
  // sinh(x) = (exp(x) - exp(-x)) / 2. The clamped x is written into pDst and a
  // single in-place Exp pass turns it into exp(x); pSrc is not read past
  // that fill so dst may alias src. In [-88, 88] exp neither overflows nor
  // underflows, so exp(-x) = 1/exp(x) is exact and sinh = (E - 1/E)*0.5 (sinh
  // would overflow to +/-Inf outside the clamp anyway, matching pcr_sinhf).
  for I := 0 to NM1 do
  begin
    X := pSrc^[I];
    if X > 88 then X := 88
    else if X < -88 then X := -88;
    pDst^[I] := X;
  end;
  Exp(pDst, pDst, N);
  for I := 0 to NM1 do
  begin
    E := pDst^[I]; // exp(x)
    pDst^[I] := (E - 1 / E) * 0.5;
  end;
end;

class procedure TNNetVolume.Ln(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
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

class procedure TNNetVolume.Sin(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
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

class procedure TNNetVolume.Cos(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
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

class procedure TNNetVolume.ArcSinh(pDst, pSrc: TNeuralFloatArrPtr; N: integer);
var
  I, NM1: integer;
  X: TNeuralFloat;
begin
  if N <= 0 then exit;
  NM1 := N - 1;
  // arcsinh(x) = ln(x + sqrt(x^2 + 1)). The argument x + sqrt(x^2+1) is always >= 1
  // and is built directly into pDst; pSrc^[I] is read before pDst^[I] is written at
  // the same index, so dst may alias src elementwise. Ln then supplies the
  // AVX2 ln pass in place.
  for I := 0 to NM1 do
  begin
    X := pSrc^[I];
    pDst^[I] := X + Sqrt(X * X + 1.0);
  end;
  Ln(pDst, pDst, N);
end;

procedure TVolume.GroupedPointwiseSoftMax(Groups: integer);
var
  StartPointPos: integer;
  MaxX, MaxY: integer;
  CountX, CountY: integer;
  SpanMax: TNeuralFloat;
  TotalSum: TNeuralFloat;
  GroupCnt, StartD, PointBase: integer;
  ChannelsPerGroup, GroupsM1: integer;
  RowStride, colBase: integer;
begin
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  ChannelsPerGroup := FDepth div Groups;
  GroupsM1 := Groups - 1;
  RowStride := FSizeX * FDepth; // #12: per-CountY GetRawPos step (carried below)
  if ChannelsPerGroup > 1 then
  begin
    colBase := 0;
    for CountX := 0 to MaxX do
    begin
      PointBase := colBase;
      for CountY := 0 to MaxY do
      begin
        StartD := 0;
        for GroupCnt := 0 to GroupsM1 do
        begin
          //EndD := StartD + ChannelsPerGroup - 1;
          StartPointPos := PointBase + StartD;
          // A group is ChannelsPerGroup contiguous elements from StartPointPos,
          // so it takes the same three vectorized reductions PointwiseSoftMax
          // applies to a depth span: span max, fused shift-exp-sum, normalize.
          SpanMax := TNNetVolume.MaxValue(Addr(FData[StartPointPos]), ChannelsPerGroup);
          TotalSum := TNNetVolume.ExpShiftSum(Addr(FData[StartPointPos]),
            Addr(FData[StartPointPos]), SpanMax, ChannelsPerGroup);
          if TotalSum > 0 then
            TNNetVolume.Mul(Addr(FData[StartPointPos]), 1.0 / TotalSum, ChannelsPerGroup);
          Inc(StartD, ChannelsPerGroup);
        end;
        Inc(PointBase, RowStride);
      end;
      Inc(colBase, FDepth);
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
  CntToken, MaxToken, Token, SizeXM1, MaxTokenP1, rowBase: integer;
begin
  MaxToken := Length(aTokens) - 1;
  SizeXM1 := SizeX - 1;
  Self.Fill(0);
  if MaxToken < SizeX then
  begin
    rowBase := 0; // #12: GetRawPos(CntToken,0), steps FDepth per token
    for CntToken := 0 to MaxToken do
    begin
      Token := aTokens[CntToken];
      if Token < FDepth then
      begin
        FData[rowBase + Token] := 1;
      end
      else
      begin
        WriteLn('Token '+IntToStr(Token)+' is bigger than Depth '+IntToStr(FDepth)+' at OneHotEncoding.');
      end;
      Inc(rowBase, FDepth);
    end;
    if MaxToken < SizeX - 1 then
    begin
      MaxTokenP1 := MaxToken + 1;
      // rowBase now equals MaxTokenP1*FDepth = GetRawPos(MaxTokenP1,0)
      for CntToken := MaxTokenP1 to SizeXM1 do
      begin
        FData[rowBase] := 1;
        Inc(rowBase, FDepth);
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
  groupBase: integer;
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
      groupBase := 0; // #6: carried GroupCnt*GroupSize
      for GroupCnt := 0 to MaxGroup do
      begin
        TokenDiv := Token div GroupSize;
        TokenMod := Token mod GroupSize;
        TokenPos := groupBase + TokenMod;
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
        Inc(groupBase, GroupSize); // #6: next GroupCnt*GroupSize
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
  v: TNeuralFloat;
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
    TokenBase := GetRawPos(CntToken, 0);
    InitTokenPos := 0; // #6: carried GroupCnt*GroupSize
    for GroupCnt := 0 to MaxGroup do
    begin
      RawTokenPos := TokenBase + InitTokenPos;
      MaxValue := FData[RawTokenPos];
      MaxTokenMod := 0;
      // Calculate the position within the group
      for TokenMod := 1 to MaxGroupSize do
      begin
        v := FData[RawTokenPos + TokenMod]; // #4: read element once
        if v > MaxValue then
        begin
          MaxValue := v;
          MaxTokenMod := TokenMod;
        end;
      end;
      // Reconstruct the token by reversing the modulus and division
      Token := Token + MaxTokenMod * GroupSizePower;
      GroupSizePower := GroupSizePower * GroupSize;
      Inc(InitTokenPos, GroupSize); // #6: next GroupCnt*GroupSize
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
  RawTokenPos: integer;
  MaxValue: TNeuralFloat;
  MaxTokenMod: integer;
  v: TNeuralFloat;
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
    RawTokenPos := GetRawPos(X, Y); // #6: carried GetRawPos(X,Y,GroupCnt*GroupSize)
    for GroupCnt := 0 to MaxGroup do
    begin
      MaxValue := FData[RawTokenPos];
      MaxTokenMod := 0;
      // Calculate the position within the group
      for TokenMod := 1 to MaxGroupSize do
      begin
        v := FData[RawTokenPos + TokenMod]; // #4: read element once
        if v > MaxValue then
        begin
          MaxValue := v;
          MaxTokenMod := TokenMod;
        end;
      end;
      // Reconstruct the token by reversing the modulus and division
      Token := Token + MaxTokenMod * GroupSizePower;
      GroupSizePower := GroupSizePower * GroupSize;
      Inc(RawTokenPos, GroupSize); // #6: next group base
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
  // GetLastChars returns the input unchanged when it already fits, so a single
  // unconditional truncation covers both cases and keeps MaxToken derived from
  // the string that is actually encoded.
  LocalTokens := GetLastChars(aTokens, SizeX);
  MaxToken := Length(LocalTokens);
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
  CntToken, MaxToken, Token, rowBase: integer;
begin
  MaxToken := Length(aTokens) - 1;
  Self.Fill(0);
  if MaxToken < SizeX then
  begin
    rowBase := MaxToken * FDepth; // #12: GetRawPos(MaxToken-CntToken,0), Dec per token
    for CntToken := 0 to MaxToken do
    begin
      Token := aTokens[CntToken];
      if Token < FDepth then
      begin
        FData[rowBase + Token] := 1;
      end
      else
      begin
        WriteLn('Token '+IntToStr(Token)+' is bigger than Depth '+IntToStr(FDepth)+' at OneHotEncodingReversed.');
      end;
      Dec(rowBase, FDepth);
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
  RawPos, RowStride, colPos: integer;
  IsEvenDepth: boolean;
begin
  EmbeddingSize := FDepth;
  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxDepth := FDepth - 1;
  RowStride := FSizeX * FDepth; // per-CntY step
  for CntDepth := 0 to MaxDepth do
  begin
    divTerm := pcr_powf(n, (CntDepth and (not 1)) / EmbeddingSize); // 2*(CntDepth div 2), CntDepth>=0
    IsEvenDepth := ((CntDepth and 1) = 0);
    colPos := CntDepth; // #12: carried GetRawPos(CntX, 0, CntDepth)
    for CntX := 0 to MaxX do
    begin
      RawPos := colPos;
      Position := CntX + PositionOffset; // #6: Position at CntY=0, carried across CntY
      for CntY := 0 to MaxY do
      begin
        if IsEvenDepth
          then FData[RawPos] := pcr_sinf(Position / divTerm)
          else FData[RawPos] := pcr_cosf(Position / divTerm);
        Inc(RawPos, RowStride);
        Inc(Position, FSizeX); // #6: next CntY position
      end;
      Inc(colPos, FDepth);
    end;
  end;
end;

procedure TVolume.RgbToHsv();
var
  I, J: integer;
  MaxX, MaxY: integer;
  h, s, v: TNeuralFloat;
  base, RowStride, colBase: integer;
begin
  h := 0;
  s := 0;
  v := 0;

  // this function can only be used if the first 3 layers contain RGB
  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;
    RowStride := FSizeX * FDepth; // #12: per-J GetRawPos step
    colBase := 0;
    for I := 0 to MaxX do
    begin
      base := colBase;
      for J := 0 to MaxY do
      begin
        rgb2hsv(FData[base], FData[base+1], FData[base+2], h, s, v);
        FData[base] := h;
        FData[base+1] := s;
        FData[base+2] := v;
        Inc(base, RowStride);
      end;
      Inc(colBase, FDepth);
    end;
  end;
end;

procedure TVolume.HsvToRgb();
var
  I, J: integer;
  MaxX, MaxY: integer;
  r, g, b: TNeuralFloat;
  base, RowStride, colBase: integer;
begin
  r := 0;
  g := 0;
  b := 0;

  // this function can only be used if the first 3 layers contain RGB
  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;
    RowStride := FSizeX * FDepth; // #12: per-J GetRawPos step
    colBase := 0;
    for I := 0 to MaxX do
    begin
      base := colBase;
      for J := 0 to MaxY do
      begin
        hsv2rgb(FData[base], FData[base+1], FData[base+2], r, g, b);
        FData[base] := r;
        FData[base+1] := g;
        FData[base+2] := b;
        Inc(base, RowStride);
      end;
      Inc(colBase, FDepth);
    end;
  end;
end;

procedure TVolume.RgbToHsl();
var
  I, J: integer;
  MaxX, MaxY: integer;
  h, s, l: TNeuralFloat;
  base, RowStride, colBase: integer;
begin
  h := 0;
  s := 0;
  l := 0;

  // this function can only be used if the first 3 layers contain RGB
  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;
    RowStride := FSizeX * FDepth; // #12: per-J GetRawPos step
    colBase := 0;
    for I := 0 to MaxX do
    begin
      base := colBase;
      for J := 0 to MaxY do
      begin
        rgb2hsl(FData[base], FData[base+1], FData[base+2], h, s, l);
        FData[base] := h;
        FData[base+1] := s;
        FData[base+2] := l;
        Inc(base, RowStride);
      end;
      Inc(colBase, FDepth);
    end;
  end;
end;

procedure TVolume.HslToRgb();
var
  I, J: integer;
  MaxX, MaxY: integer;
  r, g, b: TNeuralFloat;
  base, RowStride, colBase: integer;
begin
  r := 0;
  g := 0;
  b := 0;

  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;
    RowStride := FSizeX * FDepth; // #12: per-J GetRawPos step
    colBase := 0;
    for I := 0 to MaxX do
    begin
      base := colBase;
      for J := 0 to MaxY do
      begin
        hsl2rgb(FData[base], FData[base+1], FData[base+2], r, g, b);
        FData[base] := r;
        FData[base+1] := g;
        FData[base+2] := b;
        Inc(base, RowStride);
      end;
      Inc(colBase, FDepth);
    end;
  end;
end;

procedure TVolume.RgbToLab();
var
  I, J: integer;
  MaxX, MaxY: integer;
  l, a, b: TNeuralFloat;
  base, RowStride, colBase: integer;
begin
  l := 0;
  a := 0;
  b := 0;

  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;
    RowStride := FSizeX * FDepth; // #12: per-J GetRawPos step
    colBase := 0;
    for I := 0 to MaxX do
    begin
      base := colBase;
      for J := 0 to MaxY do
      begin
        rgb2lab(FData[base], FData[base+1], FData[base+2], l, a, b);
        FData[base] := l;
        FData[base+1] := a;
        FData[base+2] := b;
        Inc(base, RowStride);
      end;
      Inc(colBase, FDepth);
    end;
  end;
end;

procedure TVolume.LabToRgb();
var
  I, J: integer;
  MaxX, MaxY: integer;
  r, g, b: TNeuralFloat;
  base, RowStride, colBase: integer;
begin
  r := 0;
  g := 0;
  b := 0;

  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;
    RowStride := FSizeX * FDepth; // #12: per-J GetRawPos step
    colBase := 0;
    for I := 0 to MaxX do
    begin
      base := colBase;
      for J := 0 to MaxY do
      begin
        lab2rgb(FData[base], FData[base+1], FData[base+2], r, g, b);
        FData[base] := r;
        FData[base+1] := g;
        FData[base+2] := b;
        Inc(base, RowStride);
      end;
      Inc(colBase, FDepth);
    end;
  end;
end;

procedure TVolume.RgbToGray();
var
  I, J: integer;
  MaxX, MaxY: integer;
  aux: TNeuralFloat;
  base, RowStride, colBase: integer;
begin
  if Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;
    RowStride := FSizeX * FDepth; // #12: per-J GetRawPos step
    colBase := 0;
    for I := 0 to MaxX do
    begin
      base := colBase;
      for J := 0 to MaxY do
      begin
        aux := (FData[base] + FData[base+1] + FData[base+2]) / 3;
        FData[base] := aux;
        FData[base+1] := aux;
        FData[base+2] := aux;
        Inc(base, RowStride);
      end;
      Inc(colBase, FDepth);
    end;
  end;
end;

procedure TVolume.GetGrayFromRgb(Rgb: TVolume);
var
  I, J: integer;
  MaxX, MaxY: integer;
  rgbBase, rgbRowStride, rgbColBase: integer;
  selfPos, selfRowStride, selfColBase: integer;
begin
  ReSize(Rgb.SizeX, Rgb.SizeY, 1);
  if Rgb.Depth >= 3 then
  begin
    MaxX := FSizeX - 1;
    MaxY := FSizeY - 1;
    rgbRowStride := Rgb.FSizeX * Rgb.FDepth; // #12: per-J Rgb GetRawPos step
    selfRowStride := FSizeX * FDepth;        // #12: per-J Self GetRawPos step
    rgbColBase := 0;
    selfColBase := 0;
    for I := 0 to MaxX do
    begin
      rgbBase := rgbColBase;
      selfPos := selfColBase;
      for J := 0 to MaxY do
      begin
        FData[selfPos] :=
          (Rgb.FData[rgbBase] + Rgb.FData[rgbBase+1] + Rgb.FData[rgbBase+2]) / 3;
        Inc(rgbBase, rgbRowStride);
        Inc(selfPos, selfRowStride);
      end;
      Inc(rgbColBase, Rgb.FDepth);
      Inc(selfColBase, FDepth);
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
// The pSize x pSize spatial box sum is read off a 2-D summed-area table built
// in place in the caller's scratch volume, so every output costs at most three
// adds instead of pSize*pSize. The summands are all alpha*x^2 >= 0 and the
// window is a fraction of the plane, so the four-corner subtraction loses only
// a few ULP of a quantity that is then added to 1.
procedure TNNetVolume.CalculateLocalResponseFrom2D(Original, SqrElements: TNNetVolume;
  pSize: integer; alpha, beta: TNeuralFloat);
var
  iFrom, iTo: integer;
  MaxX, MaxY: integer;
  MinIX, MaxIX, MinIY, MaxIY: integer;
  CountX, CountY: integer;
  iBase: integer;
  RowStrideSq, SqDepth: integer;
  HasLeft, HasTop: boolean;
begin
  ReSize(Original);
  Fill(1);
  SqrElements.ReSize(Original); // no-op once the shape settles (rule #17)

  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;

  iTo := pSize shr 1;
  iFrom := -iTo;
  SqrElements.Copy(Original);
  SqrElements.Mul(SqrElements);
  SqrElements.Mul(alpha/(pSize*pSize));
  SqDepth := SqrElements.FDepth;               // one CountX step in SqrElements
  RowStrideSq := SqrElements.FSizeX * SqDepth; // one CountY step in SqrElements

  // Inclusive summed-area table, in place: prefix along X, then along Y.
  for CountY := 0 to MaxY do
  begin
    iBase := SqrElements.GetRawPos(0, CountY);
    for CountX := 1 to MaxX do
    begin
      TNNetVolume.Add(SqrElements.GetRawPtr(iBase + SqDepth),
        SqrElements.GetRawPtr(iBase), SqDepth);
      Inc(iBase, SqDepth);
    end;
  end;
  iBase := 0;
  for CountY := 1 to MaxY do
  begin
    // A whole row is contiguous, so one call carries the Y prefix.
    TNNetVolume.Add(SqrElements.GetRawPtr(iBase + RowStrideSq),
      SqrElements.GetRawPtr(iBase), RowStrideSq);
    Inc(iBase, RowStrideSq);
  end;

  // Self is filled with 1, so each box sum accumulates on top of it. Self and
  // SqrElements share Original's shape, so one base indexes both.
  for CountX := 0 to MaxX do
  begin
    MinIX := Max(CountX + iFrom,0);
    MaxIX := Min(CountX + iTo, MaxX);
    HasLeft := MinIX > 0;
    for CountY := 0 to MaxY do
    begin
      MinIY := Max(CountY + iFrom,0);
      MaxIY := Min(CountY + iTo, MaxY);
      HasTop := MinIY > 0;
      iBase := GetRawPos(CountX, CountY);
      TNNetVolume.Add(GetRawPtr(iBase),
        SqrElements.GetRawPtr(SqrElements.GetRawPos(MaxIX, MaxIY)), FDepth);
      if HasLeft then
        TNNetVolume.MulAdd(GetRawPtr(iBase),
          SqrElements.GetRawPtr(SqrElements.GetRawPos(MinIX - 1, MaxIY)), -1, FDepth);
      if HasTop then
      begin
        TNNetVolume.MulAdd(GetRawPtr(iBase),
          SqrElements.GetRawPtr(SqrElements.GetRawPos(MaxIX, MinIY - 1)), -1, FDepth);
        if HasLeft then
          TNNetVolume.Add(GetRawPtr(iBase),
            SqrElements.GetRawPtr(SqrElements.GetRawPos(MinIX - 1, MinIY - 1)), FDepth);
      end;
    end;
  end;

  Pow(beta);
end;

// The MinID..MaxID depth window is a sliding box sum, so it is read off an
// inclusive prefix sum built in place along the depth axis of the caller's
// scratch volume: one subtraction per output instead of pSize adds. All the
// summands are alpha*x^2 >= 0 and the result is then added to 1, so the
// prefix difference is numerically harmless.
procedure TNNetVolume.CalculateLocalResponseFromDepth(Original, SqrElements: TNNetVolume;
  pSize: integer; alpha, beta: TNeuralFloat);
var
  iFrom, iTo: integer;
  MaxX, MaxY, MaxD: integer;
  MinID, MaxID: integer;
  CountX, CountY, CountD: integer;
  sqrPos: integer;
  iBase: integer;
  WindowSum: TNeuralFloat;
begin
  ReSize(Original);
  SqrElements.ReSize(Original); // no-op once the shape settles (rule #17)

  MaxX := FSizeX - 1;
  MaxY := FSizeY - 1;
  MaxD := FDepth - 1;

  iTo := pSize shr 1;
  iFrom := -iTo;
  SqrElements.Copy(Original);
  SqrElements.Mul(SqrElements);
  SqrElements.Mul(alpha/pSize);

  for CountX := 0 to MaxX do
  begin
    for CountY := 0 to MaxY do
    begin
      // Self and SqrElements are both shaped like Original, so one base indexes both.
      iBase := GetRawPos(CountX, CountY);
      // Inclusive prefix along the depth axis of this (X, Y) column, in place.
      for CountD := 1 to MaxD do
      begin
        sqrPos := iBase + CountD;
        SqrElements.FData[sqrPos] :=
          SqrElements.FData[sqrPos] + SqrElements.FData[sqrPos - 1];
      end;
      for CountD := 0 to MaxD do
      begin
        MinID := CountD + iFrom;
        MaxID := Min(CountD + iTo, MaxD);
        WindowSum := SqrElements.FData[iBase + MaxID];
        // MinID <= 0 means the window starts at depth 0: nothing to subtract.
        if MinID > 0 then
          WindowSum := WindowSum - SqrElements.FData[iBase + MinID - 1];
        FData[iBase + CountD] := 1 + WindowSum;
      end;
    end;
  end;

  Pow(beta);
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
  AOfs, DestOfs: integer;
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
  DestOfs := 0;
  CntBVectorSizePlusCntBPos := 0;
  for CntB := 0 to NumBM1 do
  begin
    DestPointer := Self.GetRawPtr(DestOfs);
    AOfs := 0;
    for CntBPos := 0 to MaxBPos do
    begin
      //MulAdd(DestPointer, InterleavedAs.GetRawPtr(CntBPos*NumA), Bs.FData[CntB*VectorSize + CntBPos], NumA);
      MulAdd(DestPointer, InterleavedAs.GetRawPtr(AOfs), Bs.FData[CntBVectorSizePlusCntBPos], NumA);
      Inc(CntBVectorSizePlusCntBPos);
      Inc(AOfs, NumA);
    end;
    Inc(DestOfs, NumA);
  end;
end;

procedure TNNetVolume.InterleavedDotProduct(InterleavedAs, Bs: TNNetVolume;
  BStart, BFinish, VectorSize: integer);
var
  CntB, CntBPos, MaxBPos: integer;
  NumA, NumB: integer;
  DestPointer: pointer;
  CntBVectorSizePlusCntBPos: integer;
  AOfs, DestOfs: integer;
begin
  NumA := InterleavedAs.Size div VectorSize;
  NumB := Bs.Size div VectorSize;

  MaxBPos := VectorSize - 1;

  if FSize <> NumA * NumB then
  begin
    Resize(1, NumB, NumA);
  end;

  DestOfs := NumA*BStart;
  CntBVectorSizePlusCntBPos := BStart*VectorSize;
  for CntB := BStart to BFinish do
  begin
    DestPointer := Self.GetRawPtr(DestOfs);
    AOfs := 0;
    for CntBPos := 0 to MaxBPos do
    begin
      //MulAdd(DestPointer, InterleavedAs.GetRawPtr(CntBPos*NumA), Bs.FData[CntB*VectorSize + CntBPos], NumA);
      MulAdd(DestPointer, InterleavedAs.GetRawPtr(AOfs), Bs.FData[CntBVectorSizePlusCntBPos], NumA);
      Inc(CntBVectorSizePlusCntBPos);
      Inc(AOfs, NumA);
    end;
    Inc(DestOfs, NumA);
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
  RowBase, AOfs, BPos: integer;
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

  BPos := BStart * VectorSize;   // #6: carried CntB*VectorSize
  RowBase := BStart * NumAs;     // #6: carried CntB*NumAs
  for CntB := BStart to BFinish do
  begin
    PtrB := VBs.GetRawPtr(BPos);
    if NoForward
      then LocalMaxA := Min(MaxA, CntB)
      else LocalMaxA := MaxA;
    if LocalMaxA >= 0 then
    begin
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
        vxorps ymm6, ymm6, ymm6
        vxorps ymm7, ymm7, ymm7
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
          vfmadd231ps ymm6, ymm4, [rdx+64]
          vfmadd231ps ymm7, ymm5, [rdx+96]
          {$ELSE}
          vmulps  ymm2, ymm2, [rdx]
          vmulps  ymm3, ymm3, [rdx+32]
          vmulps  ymm4, ymm4, [rdx+64]
          vmulps  ymm5, ymm5, [rdx+96]

          vaddps  ymm0, ymm0, ymm2
          vaddps  ymm1, ymm1, ymm3
          vaddps  ymm6, ymm6, ymm4
          vaddps  ymm7, ymm7, ymm5
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
        vaddps ymm6, ymm6, ymm7
        vaddps ymm0, ymm0, ymm6
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
          {$IFDEF AVX512},'zmm0', 'zmm1'{$ELSE},'ymm6', 'ymm7'{$ENDIF}
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
    Inc(BPos, VectorSize); // #6: next CntB*VectorSize
    Inc(RowBase, NumAs);   // #6: next CntB*NumAs
  end; // CntB
end;

procedure TNNetVolume.DotProductsTiled(NumAs, NumBs, VectorSize: integer; VAs, VBs: TNNetVolume; TileSizeA, TileSizeB: integer);
begin
  DotProductsTiled(NumAs, 0, NumBs-1, VectorSize, VAs, VBs, TileSizeA, TileSizeB);
end;

procedure TNNetVolume.DotProductsTiled(NumAs, BStart, BFinish, VectorSize: integer; VAs, VBs: TNNetVolume; TileSizeA, TileSizeB: integer; AStart: integer = 0; AFinish: integer = -1);
var
  CntA, CntB: Integer;
  RowBase, AOfs, BPos: integer;
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
  TileAOfs0: integer;
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
      TileAOfs0 := StartTileA * VectorSize; // #5: hoisted, invariant across CntB
      BPos := StartTileB * VectorSize;   // #12: carried CntB*VectorSize
      RowBase := StartTileB * NumAs;     // #12: carried CntB*NumAs
      for CntB := StartTileB to EndTileB do
      begin
        PtrB := VBs.GetRawPtr(BPos);
        AOfs := TileAOfs0;
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
          vxorps ymm6, ymm6, ymm6
          vxorps ymm7, ymm7, ymm7
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
            vfmadd231ps ymm6, ymm4, [rdx+64]
            vfmadd231ps ymm7, ymm5, [rdx+96]
            {$ELSE}
            vmulps  ymm2, ymm2, [rdx]
            vmulps  ymm3, ymm3, [rdx+32]
            vmulps  ymm4, ymm4, [rdx+64]
            vmulps  ymm5, ymm5, [rdx+96]

            vaddps  ymm0, ymm0, ymm2
            vaddps  ymm1, ymm1, ymm3
            vaddps  ymm6, ymm6, ymm4
            vaddps  ymm7, ymm7, ymm5
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
          vaddps ymm6, ymm6, ymm7
          vaddps ymm0, ymm0, ymm6
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
            {$IFDEF AVX512},'zmm0', 'zmm1'{$ELSE},'ymm6', 'ymm7'{$ENDIF}
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
        Inc(BPos, VectorSize); // #12: next CntB*VectorSize
        Inc(RowBase, NumAs);   // #12: next CntB*NumAs
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

class function TNNetVolume.SumSqrCentered(pSrc: TNeuralFloatArrPtr;
  Mean: TNeuralFloat; N: integer): TNeuralFloat;
var
  I, vHigh: integer;
  Centered: TNeuralFloat;
begin
  Result := 0;
  if N <= 0 then exit;
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  if N >= csMinAvxSize then
  begin
    Result := AVXSumSqrCentered(pSrc, Mean, N);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  vHigh := N - 1;
  for I := 0 to vHigh do
  begin
    Centered := pSrc^[I] - Mean;
    Result := Result + Centered * Centered;
  end;
end;

class function TNNetVolume.MaxAbsFinite(pSrc: TNeuralFloatArrPtr;
  N: integer): TNeuralFloat;
var
  I, vHigh: integer;
  V, AbsV: TNeuralFloat;
begin
  Result := 0;
  if N <= 0 then exit;
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  if N >= csMinAvxSize then
  begin
    Result := AVXMaxAbsFinite(pSrc, N);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  vHigh := N - 1;
  for I := 0 to vHigh do
  begin
    V := pSrc^[I];
    // IsNan is a bit test: FPC emits SIGNALING compares, so even comparing a
    // NaN would raise EInvalidOp. +/-Inf compares fine and the MaxSingle test
    // excludes it.
    if IsNan(V) then continue;
    AbsV := Abs(V);
    if (AbsV > Result) and (AbsV <= MaxSingle) then Result := AbsV;
  end;
end;

class procedure TNNetVolume.QuantizeInt8(pDst: TNeuralInt8ArrPtr;
  pSrc: TNeuralFloatArrPtr; N: integer; MaxAbs: TNeuralFloat);
var
  I, vHigh, Code: integer;
  V, Scaled, Recip: TNeuralFloat;
  InvScale: double;
begin
  if (N <= 0) or (MaxAbs <= 0) then exit;
  if MaxAbs < MinSingle then
  begin
    // Denormal row max: 1/MaxAbs overflows single, so scale in DOUBLE. Real
    // checkpoints reach here only on untrained vocab padding, so the scalar
    // cost is irrelevant. This is the arithmetic the whole quantizer used
    // before the vectorized path existed.
    InvScale := 127.0 / Double(MaxAbs);
    vHigh := N - 1;
    for I := 0 to vHigh do
    begin
      V := pSrc^[I];
      if IsNan(V) then Code := 0
      else
      begin
        if V > MaxAbs then V := MaxAbs
        else if V < -MaxAbs then V := -MaxAbs;
        Code := Round(V * InvScale);
        if Code > 127 then Code := 127;
        if Code < -127 then Code := -127;
      end;
      pDst^[I] := ShortInt(Code);
    end;
    exit;
  end;
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  if N >= csMinAvxSize then
  begin
    AVXQuantizeInt8(pDst, pSrc, N, MaxAbs);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  // Scalar twin of the AVX2 kernel, in the same order: multiply by 1/MaxAbs
  // FIRST so every intermediate is bounded by 1 (never form 127/MaxAbs, which
  // overflows single for tiny rows), then by 127, then clamp - which is also
  // what maps +/-Inf onto +/-127.
  Recip := 1 / MaxAbs;
  vHigh := N - 1;
  for I := 0 to vHigh do
  begin
    V := pSrc^[I];
    if IsNan(V) then
    begin
      pDst^[I] := 0;
      continue;
    end;
    Scaled := V * Recip * 127;
    if Scaled > 127 then Scaled := 127
    else if Scaled < -127 then Scaled := -127;
    Code := Round(Scaled);
    pDst^[I] := ShortInt(Code);
  end;
end;

class procedure TNNetVolume.DequantizeInt8(pDst: TNeuralFloatArrPtr;
  pSrc: TNeuralInt8ArrPtr; N: integer; Scale: TNeuralFloat);
var
  I, vHigh: integer;
begin
  if N <= 0 then exit;
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  if N >= csMinAvxSize then
  begin
    AVXDequantizeInt8(pDst, pSrc, N, Scale);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  vHigh := N - 1;
  for I := 0 to vHigh do
    pDst^[I] := Scale * pSrc^[I];
end;

class procedure TNNetVolume.DecodeBF16(pDst: TNeuralFloatArrPtr;
  pSrc: TNeuralHalfArrPtr; N: integer);
var
  I, vHigh: integer;
  OutBits: Cardinal;
begin
  if N <= 0 then exit;
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  if N >= csMinAvxSize then
  begin
    AVXDecodeBF16(pDst, pSrc, N);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  vHigh := N - 1;
  for I := 0 to vHigh do
  begin
    OutBits := Cardinal(pSrc^[I]) shl 16;
    pDst^[I] := PSingle(@OutBits)^;
  end;
end;

class procedure TNNetVolume.DecodeF16(pDst: TNeuralFloatArrPtr;
  pSrc: TNeuralHalfArrPtr; N: integer);
var
  I, vHigh: integer;
begin
  if N <= 0 then exit;
  {$IFDEF AVX64}
  {$IFDEF AVX2}
  {$IFNDEF NOF16C}
  if N >= csMinAvxSize then
  begin
    AVXDecodeF16(pDst, pSrc, N);
    exit;
  end;
  {$ENDIF}
  {$ENDIF}
  {$ENDIF}
  vHigh := N - 1;
  for I := 0 to vHigh do
    pDst^[I] := NeuralHalfToSingle(pSrc^[I]);
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

procedure TNNetVolume.DotProductsTiledInt8(NumAs, NumBs, VectorSize: integer;
  Codes: TNNetVolumeQuant8; VBs: TNNetVolume; TileSizeA, TileSizeB: integer);
begin
  DotProductsTiledInt8(NumAs, 0, NumBs - 1, VectorSize, Codes, VBs,
    TileSizeA, TileSizeB);
end;

procedure TNNetVolume.DotProductsTiledInt8(NumAs, BStart, BFinish,
  VectorSize: integer; Codes: TNNetVolumeQuant8; VBs: TNNetVolume;
  TileSizeA, TileSizeB: integer; AStart: integer = 0; AFinish: integer = -1);
begin
  DotProductsTiledInt8(NumAs, BStart, BFinish, VectorSize, Codes.FData,
    Codes.ScaleData.FData, VBs, TileSizeA, TileSizeB, AStart, AFinish);
end;

procedure TNNetVolume.DotProductsTiledInt8(NumAs, BStart, BFinish,
  VectorSize: integer;
  const Codes: array of ShortInt; const Scales: array of TNeuralFloat;
  VBs: TNNetVolume; TileSizeA, TileSizeB: integer;
  AStart: integer = 0; AFinish: integer = -1);
var
  CntA, CntB: integer;
  RowBase, AOfs, BPos: integer;
  PtrA: TNeuralInt8ArrPtr;
  PtrB: TNeuralFloatArrPtr;
  // Tiling
  TileACnt, TileBCnt: integer;
  StartTileA, EndTileA, StartTileB, EndTileB: integer;
  MaxTileA, MaxTileB: integer;
  TileAOfs0: integer;
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
      TileAOfs0 := StartTileA * VectorSize; // #5: hoisted, invariant across CntB
      BPos := StartTileB * VectorSize;   // #12: carried CntB*VectorSize
      RowBase := StartTileB * NumAs;     // #12: carried CntB*NumAs
      for CntB := StartTileB to EndTileB do
      begin
        PtrB := VBs.GetRawPtr(BPos);
        AOfs := TileAOfs0;
        for CntA := StartTileA to EndTileA do
        begin
          PtrA := TNeuralInt8ArrPtr(@Codes[AOfs]);
          // Deferred per-row scale: fused into the store so the inner kernel
          // stays a pure raw-code reduction.
          FData[RowBase + CntA] := DotProductInt8(PtrA, PtrB, VectorSize)
            * Scales[CntA];
          Inc(AOfs, VectorSize);
        end;
        Inc(BPos, VectorSize); // #12: next CntB*VectorSize
        Inc(RowBase, NumAs);   // #12: next CntB*NumAs
      end;
    end; // A Tiling.
  end; // B Tiling.
end;

procedure TNNetGroupedVolume.GroupedDotProductsTiledInt8(Groups, NumAs, NumBs,
  VectorSize: integer; Codes: TNNetVolumeQuant8; VBs: TNNetVolume;
  TileSizeA, TileSizeB: integer);
begin
  GroupedDotProductsTiledInt8(Groups, NumAs, NumBs, VectorSize, Codes.FData,
    Codes.ScaleData.FData, VBs, TileSizeA, TileSizeB);
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
  TileAOfs0, TileGIVS0, TileInGroup0: integer;
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
      RowBase := StartTileB * NumAs;     // #12: carried CntB*NumAs
      BOfs := StartTileB * VectorBSize;  // #12: carried CntB*VectorBSize
      TileAOfs0 := StartTileA * VectorSize; // #5: hoisted, invariant across CntB
      TileGIVS0 := (StartTileA div GroupASize) * VectorSize; // #5: div hoisted
      TileInGroup0 := GroupASize - (StartTileA mod GroupASize); // #5: mod hoisted
      for CntB := StartTileB to EndTileB do
      begin
        AOfs := TileAOfs0;
        GroupIdVectorSize := TileGIVS0;
        InGroupLeft := TileInGroup0;
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
        Inc(RowBase, NumAs);     // #12: next CntB*NumAs
        Inc(BOfs, VectorBSize);  // #12: next CntB*VectorBSize
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
  CntA, CntB, CntAPos, CntBPos, MaxA, MaxB, BOfs, DestIdx: integer;
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
        BOfs := StartTileB * VectoreBSize + LocalGroupInfo.GroupIdVectorSize; // #12: carried CntB*VectoreBSize
        DestIdx := StartTileB * NumAs + CntA;                                 // #12: carried CntB*NumAs
        for CntB := StartTileB to EndTileB do
        begin
          PtrB := VBs.GetRawPtr(BOfs);
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
          vxorps ymm6, ymm6, ymm6
          vxorps ymm7, ymm7, ymm7
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
            vfmadd231ps ymm6, ymm4, [rdx+64]
            vfmadd231ps ymm7, ymm5, [rdx+96]
            {$ELSE}
            vmulps  ymm2, ymm2, [rdx]
            vmulps  ymm3, ymm3, [rdx+32]
            vmulps  ymm4, ymm4, [rdx+64]
            vmulps  ymm5, ymm5, [rdx+96]

            vaddps  ymm0, ymm0, ymm2
            vaddps  ymm1, ymm1, ymm3
            vaddps  ymm6, ymm6, ymm4
            vaddps  ymm7, ymm7, ymm5
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
          vaddps ymm6, ymm6, ymm7
          vaddps ymm0, ymm0, ymm6
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
            {$IFDEF AVX512},'zmm0', 'zmm1'{$ELSE},'ymm6', 'ymm7'{$ENDIF}
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
          FData[DestIdx] := Result;
          Inc(BOfs, VectoreBSize); // #12: next CntB*VectoreBSize
          Inc(DestIdx, NumAs);     // #12: next CntB*NumAs
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
  MaxLenY: integer;
  PosA, PosB, StrideA, StrideB: integer;
begin
  if Self.Depth = Original.Depth then
  begin
    SizeXDepth := LenX * Self.Depth;
    MaxLenY := LenY - 1;
    PosA := Self.GetRawPos(DestX, DestY);
    PosB := Original.GetRawPos(OriginX, OriginY);
    StrideA := FSizeX * FDepth;
    StrideB := Original.FSizeX * Original.FDepth;
    for CntY := 0 to MaxLenY do
    begin
      Add(Self.GetRawPtr(PosA), Original.GetRawPtr(PosB), SizeXDepth);
      Inc(PosA, StrideA);
      Inc(PosB, StrideB);
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

// The two centered copies this used to materialize were two heap allocations
// per call inside neuralvolume.pas (rule #17) and three extra traversals. The
// centered second moments need no buffer: one pass accumulating the three
// centered sums replaces the copies, the two GetSumSqr passes and the
// DotProduct pass. Centering is still done against the measured means (rather
// than the E[X^2]-E[X]^2 shortcut), so the numerics are unchanged.
function TNNetVolume.PearsonCorrelation(Y: TNNetVolume): TNeuralFloat;
var
  I, vHigh: integer;
  DX, DY: TNeuralFloat;
  SumXX, SumYY, SumXY: TNeuralFloat;
  AvgX, AvgY: TNeuralFloat;
  VarianceX, VarianceY: TNeuralFloat;
  StdDevX, StdDevY: TNeuralFloat;
  Covariance: TNeuralFloat;
  SizeFloat: TNeuralFloat;
begin
  if (FSize < 1) or (Y.FSize < 1) or (FSize <> Y.FSize) then
  begin
    Result := 0;
    exit;
  end;

  SizeFloat := FSize;
  AvgX := GetSum() / SizeFloat;
  AvgY := Y.GetSum() / SizeFloat;

  SumXX := 0;
  SumYY := 0;
  SumXY := 0;
  vHigh := FSize - 1;
  for I := 0 to vHigh do
  begin
    DX := FData[I] - AvgX;
    DY := Y.FData[I] - AvgY;
    SumXX := SumXX + DX * DX;
    SumYY := SumYY + DY * DY;
    SumXY := SumXY + DX * DY;
  end;

  VarianceX := SumXX / SizeFloat;
  VarianceY := SumYY / SizeFloat;

  StdDevX := Sqrt( VarianceX );
  StdDevY := Sqrt( VarianceY );

  if (StdDevX <> 0) and (StdDevY<>0) then
  begin
    Covariance := SumXY / SizeFloat;
    Result := (Covariance) / (StdDevX * StdDevY);
    Result := NeuronForceRange(Result, 1);
  end
  else
  begin
    Result := 0;
  end;
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
  CurrentMaxAbs: TNeuralFloat;
begin
  CurrentMaxAbs := GetMaxAbs();
  if CurrentMaxAbs > 0 then
  begin
    Mul( Value/CurrentMaxAbs );
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
      Dst1Pos := GetRawPos(CntX, 0); // Self[CntX, CntY, 0]
      Dst2Pos := GetRawPos(0, CntX); // Self[CntY, CntX, 0]
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
  SrcB0, SrcBPos, Dst1Pos, Dst2Pos: integer;
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
      SrcB0 := Original.GetRawPos(0, 0, CntD); // #11: no CntX in args
      for CntX := 0 to MaxX do
      begin
        // Self was resized to (Original.SizeX, Original.SizeX, Original.Depth),
        // so Self and Original share X-stride and depth: GetRawPos(CntX,0,CntD)
        // is the same flat offset in both, so compute it once.
        Dst1Pos := GetRawPos(CntX, 0, CntD);
        OrigA := Original.FData[Dst1Pos];
        SrcBPos := SrcB0;
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


// One fused Adam step over a weight row, eight lanes at a time:
//   m := Beta1*m + OmBeta1*g
//   v := Beta2*v + OmBeta2*(g*g)
//   g := (kLR*m) / (sqrt(v*InvOmB2D) + Epsilon)
// The composed form needs eleven passes and a scratch row; this reads delta, m
// and v once and writes each once. No FMA: every multiply and every add rounds
// separately, which is what keeps the result bit-identical to the composed
// kernels. The seven scalars are broadcast from their own addresses, so the
// code stays position independent.
procedure AVXAdamDelta(PtrDelta, PtrM, PtrV: TNeuralFloatArrPtr;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR: TNeuralFloat;
  NumElements: integer);
var
  pB1, pOmB1, pB2, pOmB2, pInv, pEps, pLR: pointer;
  localNumElements, MissedElements, I: integer;
  g, m, v, t1, t2: TNeuralFloat;
begin
  MissedElements := NumElements and 7;
  localNumElements := NumElements xor MissedElements;
  if localNumElements > 0 then
  begin
    pB1   := Addr(Beta1);
    pOmB1 := Addr(OmBeta1);
    pB2   := Addr(Beta2);
    pOmB2 := Addr(OmBeta2);
    pInv  := Addr(InvOmB2D);
    pEps  := Addr(Epsilon);
    pLR   := Addr(kLR);
  asm
  mov rcx, pB1
  vbroadcastss ymm9, [rcx]
  mov rcx, pOmB1
  vbroadcastss ymm10, [rcx]
  mov rcx, pB2
  vbroadcastss ymm11, [rcx]
  mov rcx, pOmB2
  vbroadcastss ymm12, [rcx]
  mov rcx, pInv
  vbroadcastss ymm13, [rcx]
  mov rcx, pEps
  vbroadcastss ymm14, [rcx]
  mov rcx, pLR
  vbroadcastss ymm15, [rcx]

  mov rax, PtrDelta
  mov rdx, PtrM
  mov r8,  PtrV
  mov ecx, localNumElements
  shr ecx, 3

@AdamLoop:
  vmovups ymm0, [rax]
  vmulps  ymm1, ymm0, ymm10
  vmulps  ymm2, ymm9, [rdx]
  vaddps  ymm1, ymm1, ymm2
  vmovups [rdx], ymm1

  vmulps  ymm3, ymm0, ymm0
  vmulps  ymm3, ymm3, ymm12
  vmulps  ymm4, ymm11, [r8]
  vaddps  ymm3, ymm3, ymm4
  vmovups [r8], ymm3

  vmulps  ymm3, ymm3, ymm13
  vsqrtps ymm3, ymm3
  vaddps  ymm3, ymm3, ymm14
  vmulps  ymm1, ymm1, ymm15
  vdivps  ymm0, ymm1, ymm3
  vmovups [rax], ymm0

  add rax, 32
  add rdx, 32
  add r8, 32
  dec ecx
  jnz @AdamLoop

  vzeroupper
  end [
    'RAX', 'RCX', 'RDX', 'R8',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4',
    'ymm9', 'ymm10', 'ymm11', 'ymm12', 'ymm13', 'ymm14', 'ymm15'
  ];
  end; // of if

  // Scalar tail. Every intermediate lands in a TNeuralFloat before it is used
  // again, so each operation rounds exactly once - the same rounding sequence
  // the kernel above performs.
  if MissedElements > 0 then
  for I := localNumElements to NumElements - 1 do
  begin
    g  := PtrDelta^[I];
    t1 := Beta1 * PtrM^[I];
    t2 := OmBeta1 * g;
    m  := t1 + t2;
    t1 := g * g;
    t2 := OmBeta2 * t1;
    t1 := Beta2 * PtrV^[I];
    v  := t2 + t1;
    PtrM^[I] := m;
    PtrV^[I] := v;
    t1 := v * InvOmB2D;
    t1 := Sqrt(t1);
    t1 := t1 + Epsilon;
    t2 := kLR * m;
    PtrDelta^[I] := t2 / t1;
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


{$IFDEF AVX2}
// Lane-index seeds for the argmax/argmin kernels below: cAVXArgLaneSeed is the
// flat index of each of the 16 lanes in the first block, cAVXArgLaneStep is the
// per-iteration increment (16 elements are consumed per iteration).
// cAVXArgAbsMask clears the sign bit of eight Singles.
const
  cAVXArgLaneSeed: array[0..15] of integer =
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  cAVXArgLaneStep: array[0..15] of integer =
    (16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16);
  cAVXArgAbsMask: array[0..7] of longword =
    ($7FFFFFFF, $7FFFFFFF, $7FFFFFFF, $7FFFFFFF,
     $7FFFFFFF, $7FFFFFFF, $7FFFFFFF, $7FFFFFFF);

{ AVXGetMaxPos returns the largest of PtrA[0..NumElements-1] and writes the flat
  index of its FIRST occurrence into Pos - the exact contract of the scalar
  TVolume.GetMax loop, ties included.

  Sixteen elements are consumed per iteration through two independent
  accumulator pairs (value + winning index), which halves the vcmpps->vblendvps
  latency chain. The compare is _CMP_GT_OQ (predicate 30): ordered, so a NaN
  never wins, and quiet, so it does not raise the invalid-operation exception
  FPC leaves unmasked. Both the value and the index are moved by the same mask,
  so they can never disagree, and because the compare is strict the first
  occurrence within a lane is the one that survives. Cross-lane ties are broken
  towards the lower index in the scalar fold, which is what makes the overall
  result the FIRST maximum.

  The (NumElements mod 16) tail is folded by the scalar loop at the end, which
  also uses a strict compare and therefore cannot displace an equal earlier
  winner. NaN inputs are outside the contract: a NaN among the first 16 elements
  pins its lane, exactly as a NaN in element 0 pins the scalar loop. }
function AVXGetMaxPos(PtrA: TNeuralFloatArrPtr; NumElements: integer;
  out Pos: integer): Single;
var
  vMax: array[0..15] of Single;
  vIdx: array[0..15] of integer;
  I, J, localNumElements: integer;
  v: Single;
begin
  localNumElements := NumElements and (not 15);
  if localNumElements >= 16 then
  begin
  asm
  mov rax, PtrA
  mov ecx, localNumElements
  shr ecx, 4
  dec ecx                          // remaining iterations after the seed block
  vmovups   ymm0, [rax]            // seed values, lanes 0..7
  vmovups   ymm1, [rax+32]         // seed values, lanes 8..15
  vmovdqu   ymm4, [rip+cAVXArgLaneSeed]
  vmovdqu   ymm5, [rip+cAVXArgLaneSeed+32]
  vmovdqa   ymm2, ymm4             // seed winning indices = 0..15
  vmovdqa   ymm3, ymm5
  vmovdqu   ymm6, [rip+cAVXArgLaneStep]
  add rax, 64
  test ecx, ecx
  jz @Fold
@Loop:
  vmovups   ymm7, [rax]
  vmovups   ymm8, [rax+32]
  vpaddd    ymm4, ymm4, ymm6
  vpaddd    ymm5, ymm5, ymm6
  vcmpps    ymm9,  ymm7, ymm0, 30  // 30 = _CMP_GT_OQ
  vcmpps    ymm10, ymm8, ymm1, 30
  vblendvps ymm2, ymm2, ymm4, ymm9
  vblendvps ymm3, ymm3, ymm5, ymm10
  vblendvps ymm0, ymm0, ymm7, ymm9
  vblendvps ymm1, ymm1, ymm8, ymm10
  add rax, 64
  dec ecx
  jnz @Loop
@Fold:
  vmovups   vMax, ymm0
  vmovups   vMax+32, ymm1
  vmovdqu   vIdx, ymm2
  vmovdqu   vIdx+32, ymm3
  vzeroupper
  end
  [
    'RAX', 'RCX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6',
    'ymm7', 'ymm8', 'ymm9', 'ymm10'
  ];
    Result := vMax[0];
    Pos := vIdx[0];
    for J := 1 to 15 do
      if (vMax[J] > Result) or ((vMax[J] = Result) and (vIdx[J] < Pos)) then
      begin
        Result := vMax[J];
        Pos := vIdx[J];
      end;
  end
  else
  begin
    Result := PtrA^[0];
    Pos := 0;
    localNumElements := 1;
  end;
  for I := localNumElements to NumElements - 1 do
  begin
    v := PtrA^[I];
    if v > Result then
    begin
      Result := v;
      Pos := I;
    end;
  end;
end;

{ AVXGetMinPos is AVXGetMaxPos with the compare inverted (predicate 17 =
  _CMP_LT_OQ) and the fold taking the smaller value: it returns the smallest
  element and the flat index of its first occurrence, matching TVolume.GetMin. }
function AVXGetMinPos(PtrA: TNeuralFloatArrPtr; NumElements: integer;
  out Pos: integer): Single;
var
  vMin: array[0..15] of Single;
  vIdx: array[0..15] of integer;
  I, J, localNumElements: integer;
  v: Single;
begin
  localNumElements := NumElements and (not 15);
  if localNumElements >= 16 then
  begin
  asm
  mov rax, PtrA
  mov ecx, localNumElements
  shr ecx, 4
  dec ecx
  vmovups   ymm0, [rax]
  vmovups   ymm1, [rax+32]
  vmovdqu   ymm4, [rip+cAVXArgLaneSeed]
  vmovdqu   ymm5, [rip+cAVXArgLaneSeed+32]
  vmovdqa   ymm2, ymm4
  vmovdqa   ymm3, ymm5
  vmovdqu   ymm6, [rip+cAVXArgLaneStep]
  add rax, 64
  test ecx, ecx
  jz @Fold
@Loop:
  vmovups   ymm7, [rax]
  vmovups   ymm8, [rax+32]
  vpaddd    ymm4, ymm4, ymm6
  vpaddd    ymm5, ymm5, ymm6
  vcmpps    ymm9,  ymm7, ymm0, 17  // 17 = _CMP_LT_OQ
  vcmpps    ymm10, ymm8, ymm1, 17
  vblendvps ymm2, ymm2, ymm4, ymm9
  vblendvps ymm3, ymm3, ymm5, ymm10
  vblendvps ymm0, ymm0, ymm7, ymm9
  vblendvps ymm1, ymm1, ymm8, ymm10
  add rax, 64
  dec ecx
  jnz @Loop
@Fold:
  vmovups   vMin, ymm0
  vmovups   vMin+32, ymm1
  vmovdqu   vIdx, ymm2
  vmovdqu   vIdx+32, ymm3
  vzeroupper
  end
  [
    'RAX', 'RCX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6',
    'ymm7', 'ymm8', 'ymm9', 'ymm10'
  ];
    Result := vMin[0];
    Pos := vIdx[0];
    for J := 1 to 15 do
      if (vMin[J] < Result) or ((vMin[J] = Result) and (vIdx[J] < Pos)) then
      begin
        Result := vMin[J];
        Pos := vIdx[J];
      end;
  end
  else
  begin
    Result := PtrA^[0];
    Pos := 0;
    localNumElements := 1;
  end;
  for I := localNumElements to NumElements - 1 do
  begin
    v := PtrA^[I];
    if v < Result then
    begin
      Result := v;
      Pos := I;
    end;
  end;
end;

{ AVXGetMaxAbsPos is AVXGetMaxPos over |x|: each loaded vector has its sign bits
  cleared by cAVXArgAbsMask before the compare, so the returned value is a
  magnitude and Pos is the flat index of the first element carrying it - the
  contract of TVolume.GetMaxAbs. Clearing the sign bit costs one vandps per
  vector and replaces the per-element compare-and-negate branch the scalar loop
  mispredicts on roughly half of a zero-mean tensor. }
function AVXGetMaxAbsPos(PtrA: TNeuralFloatArrPtr; NumElements: integer;
  out Pos: integer): Single;
var
  vMax: array[0..15] of Single;
  vIdx: array[0..15] of integer;
  I, J, localNumElements: integer;
  v: Single;
begin
  localNumElements := NumElements and (not 15);
  if localNumElements >= 16 then
  begin
  asm
  mov rax, PtrA
  mov ecx, localNumElements
  shr ecx, 4
  dec ecx
  vmovdqu   ymm11, [rip+cAVXArgAbsMask]
  vmovups   ymm0, [rax]
  vmovups   ymm1, [rax+32]
  vandps    ymm0, ymm0, ymm11
  vandps    ymm1, ymm1, ymm11
  vmovdqu   ymm4, [rip+cAVXArgLaneSeed]
  vmovdqu   ymm5, [rip+cAVXArgLaneSeed+32]
  vmovdqa   ymm2, ymm4
  vmovdqa   ymm3, ymm5
  vmovdqu   ymm6, [rip+cAVXArgLaneStep]
  add rax, 64
  test ecx, ecx
  jz @Fold
@Loop:
  vmovups   ymm7, [rax]
  vmovups   ymm8, [rax+32]
  vandps    ymm7, ymm7, ymm11
  vandps    ymm8, ymm8, ymm11
  vpaddd    ymm4, ymm4, ymm6
  vpaddd    ymm5, ymm5, ymm6
  vcmpps    ymm9,  ymm7, ymm0, 30
  vcmpps    ymm10, ymm8, ymm1, 30
  vblendvps ymm2, ymm2, ymm4, ymm9
  vblendvps ymm3, ymm3, ymm5, ymm10
  vblendvps ymm0, ymm0, ymm7, ymm9
  vblendvps ymm1, ymm1, ymm8, ymm10
  add rax, 64
  dec ecx
  jnz @Loop
@Fold:
  vmovups   vMax, ymm0
  vmovups   vMax+32, ymm1
  vmovdqu   vIdx, ymm2
  vmovdqu   vIdx+32, ymm3
  vzeroupper
  end
  [
    'RAX', 'RCX',
    'ymm0', 'ymm1', 'ymm2', 'ymm3', 'ymm4', 'ymm5', 'ymm6',
    'ymm7', 'ymm8', 'ymm9', 'ymm10', 'ymm11'
  ];
    Result := vMax[0];
    Pos := vIdx[0];
    for J := 1 to 15 do
      if (vMax[J] > Result) or ((vMax[J] = Result) and (vIdx[J] < Pos)) then
      begin
        Result := vMax[J];
        Pos := vIdx[J];
      end;
  end
  else
  begin
    Result := Abs(PtrA^[0]);
    Pos := 0;
    localNumElements := 1;
  end;
  for I := localNumElements to NumElements - 1 do
  begin
    v := Abs(PtrA^[I]);
    if v > Result then
    begin
      Result := v;
      Pos := I;
    end;
  end;
end;
{$ENDIF}

{ AVXAddScalar: dst[0..N-1] += Value. Thirty-two elements per iteration through
  four independent ymm adds off one broadcast register, with a scalar
  (N mod 32) remainder. Bit-exact against the scalar loop: every element takes
  the same single add, in any order. }
{$IFDEF AVX2}
{$IFDEF AVX64}
procedure AVXAddScalar(PtrA: TNeuralFloatArrPtr; Value: TNeuralFloat;
  NumElements: integer);
var
  localNumElements, MissedElements, I, NumElementsM1: integer;
  ValuePtr: pointer;
begin
  MissedElements := NumElements and 31;
  localNumElements := NumElements xor MissedElements;
  NumElementsM1 := NumElements - 1;
  if localNumElements > 0 then
  begin
    ValuePtr := Addr(Value);
  asm
  mov rax, PtrA
  mov rdx, ValuePtr
  mov r8d, localNumElements
  shr r8d, 5
  vbroadcastss ymm7, [rdx]
@LoopAVXAddScalar:
  vaddps ymm0, ymm7, [rax]
  vaddps ymm1, ymm7, [rax+32]
  vaddps ymm2, ymm7, [rax+64]
  vaddps ymm3, ymm7, [rax+96]
  vmovups [rax], ymm0
  vmovups [rax+32], ymm1
  vmovups [rax+64], ymm2
  vmovups [rax+96], ymm3
  add rax, 128
  dec r8d
  jnz @LoopAVXAddScalar
  vzeroupper
  end ['rax','rdx','r8','ymm0','ymm1','ymm2','ymm3','ymm7'];
  end;
  for I := localNumElements to NumElementsM1 do
    PtrA^[I] := PtrA^[I] + Value;
end;
{$ENDIF}
{$ENDIF}

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

{ AVXExpShiftSum: dst[0..N-1] := exp(src[0..N-1] - Shift), returning the sum of
  what was written - the whole numerator-and-denominator half of a numerically
  stable softmax in one pass over the row.

  It is the AVXExp body with two additions that cost one instruction each: a
  broadcast vsubps of the row max on the way in, and a vaddps of the finished
  exponentials into an accumulator on the way out. That is why this is fused
  rather than composed: the shift-then-exp-then-sum spelling reads and writes
  the row three times and dispatches three kernels per softmax row, whereas an
  attention softmax row is short (the live cache length during decode) and is
  run once per head per layer per token, so the per-call overhead is as much of
  the cost as the arithmetic.

  The eight lane partials are folded in a fixed order, so the sum is
  reproducible but not identical to the scalar left-to-right accumulation (it is
  in fact better conditioned). exp() itself matches the scalar NeuralExp to
  ~1e-6 relative, except that arguments below about -88 return exactly +0 here
  (the 2^k bit assembly leaves a zero exponent field) where the scalar returns a
  denormal below 4e-39 - which is what keeps an additive -1e9 attention mask at
  a hard zero weight. }
{$IFDEF AVX2}
{$IFDEF AVX64}
function AVXExpShiftSum(pDst, pSrc: TNeuralFloatArrPtr; Shift: TNeuralFloat;
  NumElements: integer): TNeuralFloat;
var
  localNumElements, MissedElements, I, NumElementsM1: integer;
  LaneSums: array[0..7] of Single;
  ShiftPtr, LaneSumsPtr: pointer;
  V, Sum: TNeuralFloat;
begin
  Sum := 0;
  MissedElements := NumElements and 7;
  localNumElements := NumElements xor MissedElements;
  NumElementsM1 := NumElements - 1;
  if localNumElements > 0 then
  begin
    // localNumElements is a non-zero multiple of 8 here, so the loop below
    // always runs at least once and always fills LaneSums.
    ShiftPtr := Addr(Shift);
    LaneSumsPtr := Addr(LaneSums[0]);
  asm
  mov rax, pSrc
  mov rcx, pDst
  mov r8d, localNumElements
  shr r8d, 3
  vbroadcastss ymm10, [rip+cAVXExpHi]
  vbroadcastss ymm11, [rip+cAVXExpLo]
  vbroadcastss ymm12, [rip+cAVXLog2e]
  vbroadcastss ymm13, [rip+cAVXLn2]
  vmovd xmm14, dword ptr [rip+cAVXExp127]
  vpbroadcastd ymm14, xmm14
  mov rdx, ShiftPtr
  vbroadcastss ymm15, [rdx]
  vxorps ymm8, ymm8, ymm8
@LoopAVXExpShiftSum:
  vmovups ymm0, [rax]
  vsubps  ymm0, ymm0, ymm15        // x = src - Shift
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
  vaddps ymm8, ymm8, ymm0          // per-lane running sum
  add rax, 32
  add rcx, 32
  dec r8d
  jnz @LoopAVXExpShiftSum
  mov rdx, LaneSumsPtr
  vmovups [rdx], ymm8
  vzeroupper
  end ['rax','rcx','rdx','r8',
       'ymm0','ymm1','ymm2','ymm3','ymm4','ymm5',
       'ymm8','ymm10','ymm11','ymm12','ymm13','ymm14','ymm15'];
    for I := 0 to 7 do
      Sum := Sum + LaneSums[I];
  end;
  for I := localNumElements to NumElementsM1 do
  begin
    V := NeuralExp(pSrc^[I] - Shift);
    pDst^[I] := V;
    Sum := Sum + V;
  end;
  Result := Sum;
end;
{$ENDIF}
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
  vxorps ymm6, ymm6, ymm6
  vxorps ymm7, ymm7, ymm7
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
    vfmadd231ps ymm6, ymm4, [rdx+64]
    vfmadd231ps ymm7, ymm5, [rdx+96]
    {$ELSE}
    vmulps  ymm2, ymm2, [rdx]
    vmulps  ymm3, ymm3, [rdx+32]
    vmulps  ymm4, ymm4, [rdx+64]
    vmulps  ymm5, ymm5, [rdx+96]

    vaddps  ymm0, ymm0, ymm2
    vaddps  ymm1, ymm1, ymm3
    vaddps  ymm6, ymm6, ymm4
    vaddps  ymm7, ymm7, ymm5
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
  vaddps ymm6, ymm6, ymm7
  vaddps ymm0, ymm0, ymm6
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
    {$IFDEF AVX512},'zmm0', 'zmm1'{$ELSE},'ymm6', 'ymm7'{$ENDIF}
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

{$IFDEF AVX2}
{$IFDEF AVX64}
// GetMin/GetMax/GetMaxAbs/GetClass below hand the whole buffer to one
// vectorized argmax/argmin pass. The kernel returns the winning index as well,
// so FLastPos keeps the meaning the scalar loops gave it and GetClass is the
// same pass with the index as the result instead of the value. Below
// csMinAvxSize the per-call setup outweighs the work, so the inherited scalar
// loop runs.
function TNNetVolume.GetMin(): TNeuralFloat;
begin
  if FSize >= csMinAvxSize
    then Result := AVXGetMinPos(FDataPtr, FSize, FLastPos)
    else Result := inherited GetMin();
end;

function TNNetVolume.GetMax(): TNeuralFloat;
begin
  if FSize >= csMinAvxSize
    then Result := AVXGetMaxPos(FDataPtr, FSize, FLastPos)
    else Result := inherited GetMax();
end;

function TNNetVolume.GetMaxAbs(): TNeuralFloat;
begin
  if FSize >= csMinAvxSize
    then Result := AVXGetMaxAbsPos(FDataPtr, FSize, FLastPos)
    else Result := inherited GetMaxAbs();
end;

function TNNetVolume.GetClass(): integer;
begin
  // The scalar GetClass answers -1 for a volume of one element or less; the
  // AVX path only runs well above that, so the guard only has to keep the
  // small sizes on the inherited loop.
  if FSize >= csMinAvxSize
    then AVXGetMaxPos(FDataPtr, FSize, Result)
    else Result := inherited GetClass();
end;
{$ENDIF}
{$ENDIF}

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

// A := A*Value1 + B*Value2 over a raw run. The AVX kernel keeps two separate
// vmulps and one vaddps (no FMA), so it is bit-identical to the inherited
// scalar loop.
class procedure TNNetVolume.MulMulAdd(PtrA, PtrB: TNeuralFloatArrPtr;
  Value1, Value2: TNeuralFloat; pSize: integer);
begin
  AVXMulMulAdd(PtrA, PtrB, Value1, Value2, pSize);
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

// Dst := AlphaScale*Prev + BScale*B, the rank-one state carry every recurrent
// scan performs per token. TVolume's version composes the scalar Mul/MulAdd
// class methods; this one keeps the identical rounding -- round(Prev*Alpha)
// then round(that + round(B*BScale)), which is exactly what the MulMulAdd
// kernel emits with its separate vmulps pair and vaddps -- while running eight
// lanes at a time. Results are bit-identical to the inherited version.
// PtrPrev = PtrDst is the common in-place case and skips the copy.
class procedure TNNetVolume.RankOneUpdateRow(PtrDst, PtrPrev, PtrB: TNeuralFloatArrPtr;
  AlphaScale, BScale: TNeuralFloat; pSize: integer);
begin
  if (PtrPrev = nil) or (AlphaScale = 0) then
  begin
    Move(PtrB^, PtrDst^, pSize * SizeOf(TNeuralFloat));
    TNNetVolume.Mul(PtrDst, BScale, pSize);
  end
  else
  begin
    if PtrPrev <> PtrDst then
      Move(PtrPrev^, PtrDst^, pSize * SizeOf(TNeuralFloat));
    TNNetVolume.MulMulAdd(PtrDst, PtrB, AlphaScale, BScale, pSize);
  end;
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

procedure TNNetVolume.Add(Value: Single);
begin
  AddScalar(FDataPtr, Value, FSize);
end;

procedure TNNetVolume.Sub(Value: Single);
begin
  AddScalar(FDataPtr, -Value, FSize);
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
    SourceRawPos := Original.GetRawPtr(0, CntY);
    DestRawPos := GetRawPtr(Padding, CntY + Padding);
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
    SourceRawPos := Original.GetRawPtr(0, CntY);
    DestRawPos := GetRawPtr(PaddingX, CntY + PaddingY);
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

{ TNNetVolumeQuant8 }

constructor TNNetVolumeQuant8.Create();
begin
  Create(0, 0, 0);
end;

constructor TNNetVolumeQuant8.Create(pSizeX, pSizeY, pDepth: integer);
begin
  inherited Create();
  FScaleData := TNNetVolume.Create(1, 1, 1);
  FSizeX := 0;
  FSizeY := 0;
  FDepth := 0;
  FSize := 0;
  FDataPtr := nil;
  ReSize(pSizeX, pSizeY, pDepth);
end;

destructor TNNetVolumeQuant8.Destroy();
begin
  SetLength(FData, 0);
  FDataPtr := nil;
  FScaleData.Free;
  inherited Destroy();
end;

procedure TNNetVolumeQuant8.ReSize(pSizeX, pSizeY, pDepth: integer);
var
  NewSize: integer;
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
  // One scale per (x,y). The scale plane keeps Depth 1 so that GetRawPos with
  // d = 0 addresses both planes. It is never emptied: TNNetVolume.ReSize takes
  // addr(FData[0]) unconditionally, which range-checks under -Cr on a
  // zero-length array, so an empty volume parks the plane at (1,1,1). Read
  // ScaleCount, not ScaleData.Size, for the number of live scales.
  if (pSizeX * pSizeY) > 0
  then FScaleData.ReSize(pSizeX, pSizeY, 1)
  else FScaleData.ReSize(1, 1, 1);
  if FSize > 0
  then FDataPtr := addr(FData[0])
  else FDataPtr := nil;
end;

procedure TNNetVolumeQuant8.ReSize(Original: TNNetVolumeQuant8);
begin
  ReSize(Original.SizeX, Original.SizeY, Original.Depth);
end;

function TNNetVolumeQuant8.GetRawPos(x, y, d: integer): integer;
begin
  Result := ((FSizeX * y) + x) * FDepth + d;
end;

function TNNetVolumeQuant8.GetRawPos(x, y: integer): integer;
begin
  Result := ((FSizeX * y) + x) * FDepth;
end;

function TNNetVolumeQuant8.GetRawPtr(x, y: integer): TNeuralInt8ArrPtr;
begin
  Result := TNeuralInt8ArrPtr(@FDataPtr^[((FSizeX * y) + x) * FDepth]);
end;

function TNNetVolumeQuant8.GetRawPtr(x, y, d: integer): TNeuralInt8ArrPtr;
begin
  Result := TNeuralInt8ArrPtr(@FDataPtr^[((FSizeX * y) + x) * FDepth + d]);
end;

function TNNetVolumeQuant8.Get(x, y, d: integer): ShortInt;
begin
  Result := FData[((FSizeX * y) + x) * FDepth + d];
end;

procedure TNNetVolumeQuant8.Store(x, y, d: integer; Value: ShortInt);
begin
  FData[((FSizeX * y) + x) * FDepth + d] := Value;
end;

function TNNetVolumeQuant8.GetRaw(p: integer): ShortInt;
begin
  Result := FData[p];
end;

procedure TNNetVolumeQuant8.SetRaw(p: integer; Value: ShortInt);
begin
  FData[p] := Value;
end;

function TNNetVolumeQuant8.GetScale(x, y: integer): TNeuralFloat;
begin
  Result := FScaleData.FData[(FSizeX * y) + x];
end;

procedure TNNetVolumeQuant8.SetScale(x, y: integer; Value: TNeuralFloat);
begin
  FScaleData.FData[(FSizeX * y) + x] := Value;
end;

function TNNetVolumeQuant8.GetScalePtr(): TNeuralFloatArrPtr;
begin
  Result := FScaleData.DataPtr;
end;

function TNNetVolumeQuant8.GetScaleCount(): integer;
begin
  Result := FSizeX * FSizeY;
end;

function TNNetVolumeQuant8.Dequantize(x, y, d: integer): TNeuralFloat;
begin
  Result := FData[((FSizeX * y) + x) * FDepth + d] *
    FScaleData.FData[(FSizeX * y) + x];
end;

procedure TNNetVolumeQuant8.DequantizeRowTo(x, y: integer;
  Dest: TNeuralFloatArrPtr);
var
  RowIdx: integer;
begin
  RowIdx := (FSizeX * y) + x;
  // Rule #18: the widening multiply is a TNNetVolume primitive, so an AVX2
  // build does 8 codes per iteration. This is the inner loop of
  // TNNetLayerConcatedWeights.DequantizeWeightsInt8 (every int8 layer an
  // importer reopens) and of the int8 embedding row lookup, so it is both a
  // load-time and a decode-time path.
  TNNetVolume.DequantizeInt8(Dest,
    TNeuralInt8ArrPtr(@FDataPtr^[RowIdx * FDepth]), FDepth,
    FScaleData.FData[RowIdx]);
end;

procedure TNNetVolumeQuant8.DequantizeTo(Dest: TNNetVolume);
var
  XCnt, YCnt, SizeXM1, SizeYM1: integer;
begin
  if FSize = 0 then exit;
  Dest.ReSize(FSizeX, FSizeY, FDepth);
  SizeXM1 := FSizeX - 1;
  SizeYM1 := FSizeY - 1;
  for YCnt := 0 to SizeYM1 do
  begin
    for XCnt := 0 to SizeXM1 do
    begin
      DequantizeRowTo(XCnt, YCnt,
        TNeuralFloatArrPtr(Dest.GetRawPtr(XCnt, YCnt, 0)));
    end;
  end;
end;

procedure TNNetVolumeQuant8.Fill(c: ShortInt);
begin
  if FSize > 0 then FillChar(FData[0], FSize * csShortIntSize, byte(c));
end;

procedure TNNetVolumeQuant8.CopyFrom(Original: TNNetVolumeQuant8);
begin
  ReSize(Original.SizeX, Original.SizeY, Original.Depth);
  if FSize > 0
  then Move(Original.FData[0], FData[0], FSize * csShortIntSize);
  FScaleData.Copy(Original.ScaleData);
end;

procedure TNNetVolumeQuant8.DeleteRows(StartY: integer; Count: integer);
var
  RowCodes, RowScales, MoveRows: integer;
begin
  if (Count <= 0) or (StartY < 0) or (StartY + Count > FSizeY) then exit;
  MoveRows := FSizeY - StartY - Count;
  if MoveRows <= 0 then exit;
  RowCodes := FSizeX * FDepth;
  RowScales := FSizeX;
  Move(FData[(StartY + Count) * RowCodes], FData[StartY * RowCodes],
    MoveRows * RowCodes * csShortIntSize);
  Move(FScaleData.FData[(StartY + Count) * RowScales],
    FScaleData.FData[StartY * RowScales],
    MoveRows * RowScales * csNeuralFloatSize);
end;

procedure TNNetVolumeQuant8.GetQuantData(out pCodes: TInt8DynArr;
  out pScales: TNeuralFloatDynArr);
var
  ScaleCnt: integer;
begin
  ScaleCnt := FSizeX * FSizeY;
  SetLength(pCodes, FSize);
  SetLength(pScales, ScaleCnt);
  if FSize > 0 then Move(FData[0], pCodes[0], FSize * csShortIntSize);
  if ScaleCnt > 0
  then Move(FScaleData.FData[0], pScales[0], ScaleCnt * csNeuralFloatSize);
end;

function TNNetVolumeQuant8.GetMemSize(): int64;
begin
  Result := int64(FSize) * csShortIntSize +
    int64(FSizeX) * int64(FSizeY) * csNeuralFloatSize;
end;

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

initialization
  BuildMirostatLogRankTable();

end.
