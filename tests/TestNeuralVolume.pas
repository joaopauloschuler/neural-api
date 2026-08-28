unit TestNeuralVolume;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralvolume;

type
  TTestNeuralVolume = class(TTestCase)
  published
    procedure TestVolumeCreation;
    procedure TestVolumeFill;
    procedure TestVolumeDotProduct;
    procedure TestVolumeAddSub;
    procedure TestVolumeCopy;
    procedure TestVolumeSaveLoad;
    // New comprehensive tests
    procedure TestVolumeMul;
    procedure TestVolumeDiv;
    procedure TestVolumeResize;
    procedure TestVolumeStatistics;
    procedure TestVolumeMinMax;
    procedure TestVolumeMaxAbsNegativeFirst;
    procedure TestVolumeMinMaxClassParity;
    procedure TestVolumeExpShiftSumParity;
    procedure TestVolumeAddScalarParity;
    procedure TestVolumeSumSqrCenteredParity;
    procedure TestVolumeReluGateMaskParity;
    procedure TestVolumeReluGradParity;
    procedure TestVolumeLeakyReluParity;
    procedure TestVolumeReluLParity;
    procedure TestVolumeMaxPosParity;
    procedure TestVolumeAddSubValueParity;
    procedure TestVolumeRankOneUpdateRowParity;
    procedure TestVolumeAdamDeltaParity;
    procedure TestVolumeAdafactorDeltaParity;
    procedure TestVolumeClampAbsParity;
    procedure TestVolumeForceMaxRangeParity;
    procedure TestVolumeForceMaxAbs;
    procedure TestVolumeHasNonFiniteBitTest;
    procedure TestVolumeLionDeltaParity;
    procedure TestVolumeFlip;
    procedure TestVolumeClassification;
    procedure TestVolumeSoftMax;
    procedure TestVolumeSoftMaxParity;
    procedure TestVolumeSoftMaxConstantInput;
    procedure TestVolumePointwiseSoftMaxParity;
    procedure TestVolumeGroupedPointwiseSoftMaxParity;
    procedure TestGroupedDotProductsTiledRebuildsOnNewSource;
    procedure TestGroupedDotProductsTiledPartialTile;
    procedure TestVolumePadding;
    procedure TestVolumePaddingBorderIsZeroed;
    procedure TestVolumeTranspose;
    // Additional volume tests
    procedure TestVolumeNormalization;
    procedure TestVolumePointwiseNormAndMul;
    procedure TestVolumePointwiseMulWithoutNorms;
    procedure TestVolumeMagnitude;
    procedure TestVolumeEntropy;
    procedure TestVolumeCrossEntropy;
    procedure TestVolumeOneHotEncodingOnPixel;
    procedure TestVolumeOneHotEncoding;
    procedure TestVolumeOneHotEncodingReversedString;
    procedure TestVolumeGroupedOneHotEncoding;
    procedure TestVolumePositionalEncoding;
    procedure TestVolumeColorConversions;
    procedure TestVolumeLabRoundTrip;
    procedure TestVolumeGaussianNoise;
    procedure TestVolumeCopyResizing;
    procedure TestVolumeCopyResizingMatchesReference;
    procedure TestVolumeCopyCropping;
    procedure TestVolumeShift;
    procedure TestVolumeSumToPos;
    procedure TestVolumeSmallestIdxInRange;
    procedure TestVolumeRawPosAndPtr;
    procedure TestVolumeDepthOperations;
    // AssertFinite tests
    procedure TestAssertFiniteAllFinite;
    procedure TestAssertFiniteDetectsNaN;
    procedure TestAssertFiniteDetectsInf;
    procedure TestAssertFiniteNilVolume;
    procedure TestNeuralBoxIoU;
    procedure TestNeuralGreedyNMS;
    procedure TestLocalResponse2DMatchesReference;
    procedure TestLocalResponseDepthMatchesReference;
  end;

  // TNNetVolumeQuant8 holds int8 codes under the TNNetVolume geometry
  // contract, with one scale per (x,y). These tests pin the layout agreement
  // with TNNetVolume, the empty state and the eviction primitive.
  TTestNeuralVolumeQuant8 = class(TTestCase)
  published
    procedure TestQuant8EmptyState;
    procedure TestQuant8ResizeGeometry;
    procedure TestQuant8LayoutMatchesVolume;
    procedure TestQuant8StoreAndGet;
    procedure TestQuant8RawPointers;
    procedure TestQuant8ScaleAccess;
    procedure TestQuant8Dequantize;
    procedure TestQuant8DequantizeTo;
    procedure TestQuant8CopyFrom;
    procedure TestQuant8DeleteRows;
    procedure TestQuant8DeleteRowsGuards;
    procedure TestQuant8GetQuantData;
    procedure TestQuant8MemSize;
    procedure TestQuant8FillAndReshapeCycles;
    procedure TestQuant8TiledDotProductMatchesArrays;
    procedure TestQuant8GroupedTiledDotProductMatchesArrays;
    procedure TestMaxAbsFinite;
    procedure TestMaxAbsFiniteLengthSweep;
    procedure TestQuantizeInt8;
    procedure TestQuantizeInt8LengthSweep;
    procedure TestDequantizeInt8LengthSweep;
    procedure TestQuantizeInt8NonFinite;
    procedure TestQuantizeInt8TinyAndDenormalRows;
    procedure TestQuantizeInt8MatchesScalarReference;
    procedure TestDequantizeInt8;
    procedure TestDequantizeInt8RoundTrip;
    procedure TestDotProductInt8Int8LengthSweep;
    procedure TestDotProductInt8Int8MatchesFloatPath;
    procedure TestDecodeBF16;
    procedure TestDecodeBF16LengthSweep;
    procedure TestDecodeF16;
    procedure TestDecodeF16SpecialValues;
    procedure TestEncodeF16;
    procedure TestEncodeF16SpecialValues;
    procedure TestEncodeF16MatchesScalar;
    procedure TestDecodeF16MatchesScalar;
    procedure TestEncodeBF16;
    procedure TestEncodeBF16SpecialValues;
    procedure TestEncodeBF16MatchesScalar;
  end;

implementation

procedure TTestNeuralVolume.TestVolumeCreation;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(32, 32, 3);
  try
    AssertEquals('SizeX should be 32', 32, V.SizeX);
    AssertEquals('SizeY should be 32', 32, V.SizeY);
    AssertEquals('Depth should be 3', 3, V.Depth);
    AssertEquals('Total size should be 3072', 3072, V.Size);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeFill;
var
  V: TNNetVolume;
  I: integer;
begin
  V := TNNetVolume.Create(10, 10, 1);
  try
    V.Fill(5.0);
    for I := 0 to V.Size - 1 do
      AssertEquals('All values should be 5.0', 5.0, V.Raw[I], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeDotProduct;
var
  V1, V2: TNNetVolume;
  DotProd: TNeuralFloat;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(2.0);
    V2.Fill(3.0);
    DotProd := V1.DotProduct(V2);
    AssertEquals('Dot product of [2,2,2,2] and [3,3,3,3] should be 24', 24.0, DotProd, 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeAddSub;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(5.0);
    V2.Fill(3.0);
    V1.Add(V2);
    AssertEquals('After adding, values should be 8.0', 8.0, V1.Raw[0], 0.0001);
    V1.Sub(V2);
    AssertEquals('After subtracting, values should be 5.0', 5.0, V1.Raw[0], 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeCopy;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(10, 10, 3);
  V2 := TNNetVolume.Create(10, 10, 3);
  try
    V1.RandomizeGaussian();
    V2.Copy(V1);
    AssertEquals('Copied volume should match', 0.0, V1.SumDiff(V2), 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSaveLoad;
var
  V1, V2: TNNetVolume;
  SavedStr: string;
begin
  V1 := TNNetVolume.Create(5, 5, 2);
  V2 := TNNetVolume.Create(1, 1, 1);
  try
    V1.RandomizeGaussian();
    SavedStr := V1.SaveToString();
    V2.LoadFromString(SavedStr);
    AssertEquals('Loaded volume should match saved', 0.0, V1.SumDiff(V2), 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeMul;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(5.0);
    V1.Mul(2.0);
    AssertEquals('After multiplying by 2, values should be 10.0', 10.0, V1.Raw[0], 0.0001);
    
    V1.Fill(3.0);
    V2.Fill(4.0);
    V1.Mul(V2);
    AssertEquals('After element-wise multiplication, values should be 12.0', 12.0, V1.Raw[0], 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeDiv;
var
  V1, V2: TNNetVolume;
begin
  V1 := TNNetVolume.Create(4, 1, 1);
  V2 := TNNetVolume.Create(4, 1, 1);
  try
    V1.Fill(10.0);
    V1.Divi(2.0);
    AssertEquals('After dividing by 2, values should be 5.0', 5.0, V1.Raw[0], 0.0001);
    
    V1.Fill(12.0);
    V2.Fill(4.0);
    V1.Divi(V2);
    AssertEquals('After element-wise division, values should be 3.0', 3.0, V1.Raw[0], 0.0001);
  finally
    V1.Free;
    V2.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeResize;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(10, 10, 3);
  try
    AssertEquals('Initial SizeX should be 10', 10, V.SizeX);
    AssertEquals('Initial SizeY should be 10', 10, V.SizeY);
    AssertEquals('Initial Depth should be 3', 3, V.Depth);
    
    V.ReSize(20, 15, 5);
    AssertEquals('After resize SizeX should be 20', 20, V.SizeX);
    AssertEquals('After resize SizeY should be 15', 15, V.SizeY);
    AssertEquals('After resize Depth should be 5', 5, V.Depth);
    AssertEquals('After resize total size should be 1500', 1500, V.Size);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeStatistics;
var
  V: TNNetVolume;
  Avg, Sum, Variance, StdDev: TNeuralFloat;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    // Set values: 2, 4, 6, 8
    V.Raw[0] := 2.0;
    V.Raw[1] := 4.0;
    V.Raw[2] := 6.0;
    V.Raw[3] := 8.0;
    
    Sum := V.GetSum();
    AssertEquals('Sum should be 20.0', 20.0, Sum, 0.0001);
    
    Avg := V.GetAvg();
    AssertEquals('Average should be 5.0', 5.0, Avg, 0.0001);
    
    Variance := V.GetVariance();
    // Variance of [2,4,6,8] = E[(X-5)^2] = (9+1+1+9)/4 = 5
    AssertEquals('Variance should be 5.0', 5.0, Variance, 0.0001);
    
    StdDev := V.GetStdDeviation();
    // StdDev = sqrt(5) ≈ 2.236
    AssertEquals('StdDeviation should be ~2.236', 2.236, StdDev, 0.01);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeMinMax;
var
  V: TNNetVolume;
  MinVal, MaxVal, MaxAbsVal: TNeuralFloat;
begin
  V := TNNetVolume.Create(5, 1, 1);
  try
    V.Raw[0] := -3.0;
    V.Raw[1] := 1.0;
    V.Raw[2] := 5.0;
    V.Raw[3] := -7.0;
    V.Raw[4] := 2.0;
    
    MinVal := V.GetMin();
    MaxVal := V.GetMax();
    MaxAbsVal := V.GetMaxAbs();
    
    AssertEquals('Min should be -7.0', -7.0, MinVal, 0.0001);
    AssertEquals('Max should be 5.0', 5.0, MaxVal, 0.0001);
    AssertEquals('MaxAbs should be 7.0', 7.0, MaxAbsVal, 0.0001);
  finally
    V.Free;
  end;
end;

// Regression: GetMaxAbs used to seed its running max with the SIGNED first
// element, so a negative element 0 of largest magnitude was missed and the
// returned max-abs was too small (it would have returned 2.0 below). The
// pinned vector has element 0 = -8.0 as the unique largest magnitude. This
// FAILS against the pre-fix code and passes after seeding with abs(FData[0]).
procedure TTestNeuralVolume.TestVolumeMaxAbsNegativeFirst;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Raw[0] := -8.0; // largest magnitude AND negative AND first
    V.Raw[1] := 2.0;
    V.Raw[2] := -1.0;
    V.Raw[3] := 0.5;
    AssertEquals('MaxAbs must be 8.0 (negative element 0)', 8.0,
      V.GetMaxAbs(), 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeMinMaxClassParity;
// GetMax / GetMin / GetMaxAbs / GetClass are served by a vectorized kernel on
// an AVX2 64-bit build and by a scalar loop everywhere else. Both must agree
// EXACTLY with an independent scalar reference - value and, for GetClass, the
// argmax index including the "first occurrence wins" tie-break. Sizes straddle
// csMinAvxSize (16), the 16-element block width and its tail; the data patterns
// cover all-negative buffers (which a max-abs kernel that forgets the sign gets
// wrong), constant buffers (every element ties) and single elements.
const
  Sizes: array[0..13] of integer =
    (1, 2, 7, 15, 16, 17, 23, 31, 32, 33, 47, 64, 65, 1000);
var
  V: TNNetVolume;
  SI, Pattern, K, N, RefClass: integer;
  RefMax, RefMin, RefAbs, Val, AbsVal: TNeuralFloat;
  Tag: string;
begin
  RandSeed := 271828;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    for Pattern := 0 to 4 do
    begin
      V := TNNetVolume.Create(N, 1, 1);
      try
        for K := 0 to N - 1 do
          case Pattern of
            0: V.Raw[K] := (Random - 0.5) * 20;      // mixed signs
            1: V.Raw[K] := -(Random + 0.01) * 20;    // all negative
            2: V.Raw[K] := (Random + 0.01) * 20;     // all positive
            3: V.Raw[K] := 3.0;                      // every element ties
            4: V.Raw[K] := Round((Random - 0.5) * 6); // many repeated values
          end;

        RefMax := V.Raw[0];
        RefMin := V.Raw[0];
        RefAbs := Abs(V.Raw[0]);
        RefClass := 0;
        for K := 1 to N - 1 do
        begin
          Val := V.Raw[K];
          AbsVal := Abs(Val);
          if Val > RefMax then
          begin
            RefMax := Val;
            RefClass := K;
          end;
          if Val < RefMin then RefMin := Val;
          if AbsVal > RefAbs then RefAbs := AbsVal;
        end;

        Tag := ' (N=' + IntToStr(N) + ', pattern ' + IntToStr(Pattern) + ')';
        AssertEquals('GetMax' + Tag, RefMax, V.GetMax(), 0.0);
        AssertEquals('GetMin' + Tag, RefMin, V.GetMin(), 0.0);
        AssertEquals('GetMaxAbs' + Tag, RefAbs, V.GetMaxAbs(), 0.0);
        // GetClass answers -1 for a volume of one element or less.
        if N > 1 then
          AssertEquals('GetClass' + Tag, RefClass, V.GetClass())
        else
          AssertEquals('GetClass' + Tag, -1, V.GetClass());
      finally
        V.Free;
      end;
    end;
  end;
end;

procedure TTestNeuralVolume.TestVolumeAddScalarParity;
// AddScalar is a broadcast add on an AVX2/64-bit build and a scalar loop
// everywhere else, and it must be BIT-exact against the loop it replaces on
// both - every element takes one float add, so there is no reassociation to
// excuse a difference. Sizes straddle the 32-element block width and its tail.
const
  Sizes: array[0..10] of integer = (1, 7, 8, 31, 32, 33, 63, 64, 65, 97, 1000);
  cAddend = -1e9;
var
  Buf, Ref: array of TNeuralFloat;
  SI, K, N: integer;
begin
  RandSeed := 161803;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(Buf, N);
    SetLength(Ref, N);
    for K := 0 to N - 1 do
    begin
      Buf[K] := (Random - 0.5) * 8;
      Ref[K] := Buf[K] + cAddend;
    end;
    TNNetVolume.AddScalar(TNeuralFloatArrPtr(@Buf[0]), cAddend, N);
    for K := 0 to N - 1 do
      AssertEquals('AddScalar[' + IntToStr(K) + '] (N=' + IntToStr(N) + ')',
        Ref[K], Buf[K], 0.0);
  end;
  // A zero-length run must leave the buffer untouched.
  Buf[0] := 5.0;
  TNNetVolume.AddScalar(TNeuralFloatArrPtr(@Buf[0]), 1.0, 0);
  AssertEquals('empty run', 5.0, Buf[0], 0.0);
end;

procedure TTestNeuralVolume.TestVolumeSumSqrCenteredParity;
// SumSqrCentered is a 16-wide FMA reduction on an AVX2/64-bit build and a plain
// loop everywhere else. The two differ only in summation ORDER and every term
// is non-negative, so they cannot disagree by more than accumulated rounding --
// checked here against a DOUBLE-precision reference, which also pins down that
// the result really is the exact centered sum of squares. Sizes straddle the
// 16-element block width and its tail, and the csMinAvxSize dispatch threshold.
const
  Sizes: array[0..11] of integer = (1, 7, 8, 15, 16, 17, 31, 32, 33, 64, 97, 1000);
var
  Buf: array of TNeuralFloat;
  Ref: double;
  Got, Mean: TNeuralFloat;
  SI, K, N: integer;
begin
  RandSeed := 271828;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(Buf, N);
    Mean := 0;
    for K := 0 to N - 1 do
    begin
      Buf[K] := (Random - 0.5) * 8;
      Mean := Mean + Buf[K];
    end;
    Mean := Mean / N;
    Ref := 0;
    for K := 0 to N - 1 do
      Ref := Ref + double(Buf[K] - Mean) * double(Buf[K] - Mean);
    Got := TNNetVolume.SumSqrCentered(TNeuralFloatArrPtr(@Buf[0]), Mean, N);
    AssertEquals('SumSqrCentered (N=' + IntToStr(N) + ')', Ref, Got, 1e-3);
  end;

  // The whole point of the centered form: with |Mean| >> Std the algebraic
  // shortcut sum(x^2) - N*Mean^2 cancels away every significant digit, while
  // the centered kernel stays accurate. 1e6 +/- 1 has variance 1.
  N := 512;
  SetLength(Buf, N);
  for K := 0 to N - 1 do
    if K mod 2 = 0 then Buf[K] := 1e6 + 1 else Buf[K] := 1e6 - 1;
  Got := TNNetVolume.SumSqrCentered(TNeuralFloatArrPtr(@Buf[0]), 1e6, N);
  AssertEquals('SumSqrCentered on a large-offset run', N * 1.0, Got, 1e-2);

  // A zero-length run has no terms.
  AssertEquals('empty run', 0.0,
    TNNetVolume.SumSqrCentered(TNeuralFloatArrPtr(@Buf[0]), 1.0, 0), 0.0);
end;

procedure TTestNeuralVolume.TestVolumeReluGateMaskParity;
// ReluGateMask is AVXReluGateMask on an AVX2/64-bit build and a scalar loop
// everywhere else. The output is only ever 1.0 or 0.0, so both paths must agree
// BIT-exactly -- including at the boundary, where the contract is >= 0 (so +0.0
// and -0.0 both gate open) and NaN gates shut. The sizes separate the three
// parts of the vectorized routine - the 32-element unrolled body, the
// 8-element remainder loop and the scalar tail - and combine them.
const
  Sizes: array[0..15] of integer =
    (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 39, 40, 47, 64, 128, 1000);
var
  Src, Dst, Ref: array of TNeuralFloat;
  SI, K, N: integer;
begin
  RandSeed := 271828;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(Src, N);
    SetLength(Dst, N + 1);   // one guard slot past the run
    SetLength(Ref, N);
    for K := 0 to N - 1 do
    begin
      case K mod 7 of
        0: Src[K] := 0.0;
        1: Src[K] := -0.0;
        2: Src[K] := -1e-30;
        3: Src[K] := 1e-30;
      else
        Src[K] := (Random - 0.5) * 8;
      end;
      Dst[K] := 12345;
      if Src[K] >= 0 then Ref[K] := 1 else Ref[K] := 0;
    end;
    Dst[N] := 12345;
    TNNetVolume.ReluGateMask(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Src[0]), N);
    for K := 0 to N - 1 do
      AssertEquals('ReluGateMask[' + IntToStr(K) + '] (N=' + IntToStr(N) + ')',
        Ref[K], Dst[K], 0.0);
    AssertEquals('ReluGateMask wrote past N=' + IntToStr(N), 12345.0,
      Dst[N], 0.0);
  end;
  // In-place (dst = src) must produce the same mask.
  N := 40;
  SetLength(Src, N);
  for K := 0 to N - 1 do Src[K] := (Random - 0.5) * 8;
  SetLength(Ref, N);
  for K := 0 to N - 1 do
    if Src[K] >= 0 then Ref[K] := 1 else Ref[K] := 0;
  TNNetVolume.ReluGateMask(TNeuralFloatArrPtr(@Src[0]),
    TNeuralFloatArrPtr(@Src[0]), N);
  for K := 0 to N - 1 do
    AssertEquals('ReluGateMask in-place[' + IntToStr(K) + ']', Ref[K], Src[K], 0.0);
  // A zero-length run must leave the buffer untouched.
  Src[0] := 5.0;
  TNNetVolume.ReluGateMask(TNeuralFloatArrPtr(@Src[0]),
    TNeuralFloatArrPtr(@Src[0]), 0);
  AssertEquals('empty run', 5.0, Src[0], 0.0);
end;

procedure TTestNeuralVolume.TestVolumeReluGradParity;
// ReluGrad is AVXReluGrad on an AVX2/64-bit build and a scalar loop everywhere
// else. It is a pure select with no arithmetic, so both paths must agree
// BIT-exactly -- including at the boundary, where the contract is > 0 (so both
// +0.0 and -0.0 gate SHUT, unlike ReluGateMask's >= 0) and NaN gates shut. The
// sizes separate the three parts of the vectorized routine - the 32-element
// unrolled body, the 8-element remainder loop and the scalar tail.
const
  Sizes: array[0..15] of integer =
    (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 39, 40, 47, 64, 128, 1000);
var
  Raw, Err, Dst, Ref: array of TNeuralFloat;
  SI, K, N: integer;
begin
  RandSeed := 314159;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(Raw, N);
    SetLength(Err, N);
    SetLength(Dst, N + 1);   // one guard slot past the run
    SetLength(Ref, N);
    for K := 0 to N - 1 do
    begin
      case K mod 7 of
        0: Raw[K] := 0.0;
        1: Raw[K] := -0.0;
        2: Raw[K] := -1e-30;
        3: Raw[K] := 1e-30;
      else
        Raw[K] := (Random - 0.5) * 8;
      end;
      Err[K] := (Random - 0.5) * 20;
      Dst[K] := 12345;
      if Raw[K] > 0 then Ref[K] := Err[K] else Ref[K] := 0;
    end;
    Dst[N] := 12345;
    TNNetVolume.ReluGrad(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Err[0]), TNeuralFloatArrPtr(@Raw[0]), N);
    for K := 0 to N - 1 do
      AssertEquals('ReluGrad[' + IntToStr(K) + '] (N=' + IntToStr(N) + ')',
        Ref[K], Dst[K], 0.0);
    AssertEquals('ReluGrad wrote past N=' + IntToStr(N), 12345.0, Dst[N], 0.0);
  end;
  // In-place (dst = err) must produce the same gated errors.
  N := 40;
  SetLength(Raw, N);
  SetLength(Err, N);
  SetLength(Ref, N);
  for K := 0 to N - 1 do
  begin
    Raw[K] := (Random - 0.5) * 8;
    Err[K] := (Random - 0.5) * 20;
    if Raw[K] > 0 then Ref[K] := Err[K] else Ref[K] := 0;
  end;
  TNNetVolume.ReluGrad(TNeuralFloatArrPtr(@Err[0]),
    TNeuralFloatArrPtr(@Err[0]), TNeuralFloatArrPtr(@Raw[0]), N);
  for K := 0 to N - 1 do
    AssertEquals('ReluGrad in-place[' + IntToStr(K) + ']', Ref[K], Err[K], 0.0);
  // A zero-length run must leave the buffer untouched.
  Err[0] := 5.0;
  TNNetVolume.ReluGrad(TNeuralFloatArrPtr(@Err[0]),
    TNeuralFloatArrPtr(@Err[0]), TNeuralFloatArrPtr(@Raw[0]), 0);
  AssertEquals('empty run', 5.0, Err[0], 0.0);
end;

procedure TTestNeuralVolume.TestVolumeLeakyReluParity;
// LeakyRelu is AVXLeakyRelu on an AVX2/64-bit build and a scalar loop everywhere
// else. Both paths must agree BIT-exactly: the negative branch is the same
// single-precision multiply, and at the boundary the contract is >= 0, so +0.0
// and -0.0 both pass through unscaled. The sizes separate the three parts of
// the vectorized routine - the 32-element unrolled body, the 8-element
// remainder loop and the scalar tail - and combine them.
const
  Sizes: array[0..15] of integer =
    (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 39, 40, 47, 64, 128, 1000);
  // TYPED: an untyped 0.1 would be a Double here, so the reference multiply
  // would not be the Single one the kernel performs.
  Slope: TNeuralFloat = 0.1;
var
  Src, Dst, Ref: array of TNeuralFloat;
  SI, K, N: integer;
begin
  RandSeed := 141421;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(Src, N);
    SetLength(Dst, N + 1);   // one guard slot past the run
    SetLength(Ref, N);
    for K := 0 to N - 1 do
    begin
      case K mod 7 of
        0: Src[K] := 0.0;
        1: Src[K] := -0.0;
        2: Src[K] := -1e-30;
        3: Src[K] := 1e-30;
      else
        Src[K] := (Random - 0.5) * 8;
      end;
      Dst[K] := 12345;
      if Src[K] >= 0 then Ref[K] := Src[K] else Ref[K] := Slope * Src[K];
    end;
    Dst[N] := 12345;
    TNNetVolume.LeakyRelu(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Src[0]), Slope, N);
    for K := 0 to N - 1 do
      AssertEquals('LeakyRelu[' + IntToStr(K) + '] (N=' + IntToStr(N) + ')',
        Ref[K], Dst[K], 0.0);
    AssertEquals('LeakyRelu wrote past N=' + IntToStr(N), 12345.0, Dst[N], 0.0);
  end;
  // In-place (dst = src) must produce the same result.
  N := 40;
  SetLength(Src, N);
  for K := 0 to N - 1 do Src[K] := (Random - 0.5) * 8;
  SetLength(Ref, N);
  for K := 0 to N - 1 do
    if Src[K] >= 0 then Ref[K] := Src[K] else Ref[K] := Slope * Src[K];
  TNNetVolume.LeakyRelu(TNeuralFloatArrPtr(@Src[0]),
    TNeuralFloatArrPtr(@Src[0]), Slope, N);
  for K := 0 to N - 1 do
    AssertEquals('LeakyRelu in-place[' + IntToStr(K) + ']', Ref[K], Src[K], 0.0);
  // A slope of zero degenerates to a plain relu.
  N := 20;
  SetLength(Src, N);
  for K := 0 to N - 1 do Src[K] := (Random - 0.5) * 8;
  SetLength(Dst, N);
  TNNetVolume.LeakyRelu(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralFloatArrPtr(@Src[0]), 0.0, N);
  for K := 0 to N - 1 do
    if Src[K] >= 0 then
      AssertEquals('LeakyRelu slope 0 pos[' + IntToStr(K) + ']',
        Src[K], Dst[K], 0.0)
    else
      AssertEquals('LeakyRelu slope 0 neg[' + IntToStr(K) + ']',
        0.0, Abs(Dst[K]), 0.0);
  // A zero-length run must leave the buffer untouched.
  Src[0] := 5.0;
  TNNetVolume.LeakyRelu(TNeuralFloatArrPtr(@Src[0]),
    TNeuralFloatArrPtr(@Src[0]), Slope, 0);
  AssertEquals('empty run', 5.0, Src[0], 0.0);
end;

procedure TTestNeuralVolume.TestVolumeReluLParity;
// ReluL and ReluLGateMask are AVXReluL / AVXReluLGateMask on an AVX2/64-bit
// build and scalar loops everywhere else. Both paths must agree BIT-exactly:
// each clamped form is the same subtract-multiply-add, and the boundary
// contract is strict >, so LowLimit itself takes the low form while HighLimit
// passes through. The sizes separate the three parts of the vectorized routines
// - the 32-element unrolled body, the 8-element remainder loop and the scalar
// tail - and combine them.
const
  Sizes: array[0..15] of integer =
    (1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 39, 40, 47, 64, 128, 1000);
  // TYPED: an untyped literal would be a Double here, so the reference
  // arithmetic would not be the Single the kernel performs.
  LowLimit: TNeuralFloat = -3.0;
  HighLimit: TNeuralFloat = 3.0;
  Slope: TNeuralFloat = 0.01;
var
  Src, Dst, Ref, RefDeriv: array of TNeuralFloat;
  SI, K, N: integer;
begin
  RandSeed := 271828;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(Src, N);
    SetLength(Dst, N + 1);   // one guard slot past the run
    SetLength(Ref, N);
    SetLength(RefDeriv, N);
    for K := 0 to N - 1 do
    begin
      case K mod 8 of
        0: Src[K] := LowLimit;   // exactly on the low limit: takes the low form
        1: Src[K] := HighLimit;  // exactly on the high limit: passes through
        2: Src[K] := 0.0;
        3: Src[K] := -0.0;
      else
        Src[K] := (Random - 0.5) * 16; // spans both limits comfortably
      end;
      Dst[K] := 12345;
      if Src[K] > HighLimit then
        Ref[K] := HighLimit + (Src[K] - HighLimit) * Slope
      else if Src[K] > LowLimit then Ref[K] := Src[K]
      else Ref[K] := LowLimit + (Src[K] - LowLimit) * Slope;
      if (Src[K] > LowLimit) and not (Src[K] > HighLimit) then RefDeriv[K] := 1
      else RefDeriv[K] := Slope;
    end;
    Dst[N] := 12345;
    TNNetVolume.ReluL(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Src[0]), LowLimit, HighLimit, Slope, N);
    for K := 0 to N - 1 do
      AssertEquals('ReluL[' + IntToStr(K) + '] (N=' + IntToStr(N) + ')',
        Ref[K], Dst[K], 0.0);
    AssertEquals('ReluL wrote past N=' + IntToStr(N), 12345.0, Dst[N], 0.0);
    Dst[N] := 12345;
    TNNetVolume.ReluLGateMask(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Src[0]), LowLimit, HighLimit, Slope, N);
    for K := 0 to N - 1 do
      AssertEquals('ReluLGateMask[' + IntToStr(K) + '] (N=' + IntToStr(N) + ')',
        RefDeriv[K], Dst[K], 0.0);
    AssertEquals('ReluLGateMask wrote past N=' + IntToStr(N), 12345.0,
      Dst[N], 0.0);
  end;
  // In-place (dst = src) must produce the same result.
  N := 40;
  SetLength(Src, N);
  SetLength(Ref, N);
  for K := 0 to N - 1 do Src[K] := (Random - 0.5) * 16;
  for K := 0 to N - 1 do
    if Src[K] > HighLimit then Ref[K] := HighLimit + (Src[K] - HighLimit) * Slope
    else if Src[K] > LowLimit then Ref[K] := Src[K]
    else Ref[K] := LowLimit + (Src[K] - LowLimit) * Slope;
  TNNetVolume.ReluL(TNeuralFloatArrPtr(@Src[0]),
    TNeuralFloatArrPtr(@Src[0]), LowLimit, HighLimit, Slope, N);
  for K := 0 to N - 1 do
    AssertEquals('ReluL in-place[' + IntToStr(K) + ']', Ref[K], Src[K], 0.0);
  // A slope of zero degenerates to a hard clamp.
  N := 20;
  SetLength(Src, N);
  SetLength(Dst, N);
  for K := 0 to N - 1 do Src[K] := (Random - 0.5) * 16;
  TNNetVolume.ReluL(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralFloatArrPtr(@Src[0]), LowLimit, HighLimit, 0.0, N);
  for K := 0 to N - 1 do
  begin
    AssertTrue('ReluL slope 0 not above high[' + IntToStr(K) + ']',
      Dst[K] <= HighLimit);
    AssertTrue('ReluL slope 0 not below low[' + IntToStr(K) + ']',
      Dst[K] >= LowLimit);
  end;
  // A zero-length run must leave the buffer untouched.
  Src[0] := 5.0;
  TNNetVolume.ReluL(TNeuralFloatArrPtr(@Src[0]),
    TNeuralFloatArrPtr(@Src[0]), LowLimit, HighLimit, Slope, 0);
  AssertEquals('empty run', 5.0, Src[0], 0.0);
end;

procedure TTestNeuralVolume.TestVolumeMaxPosParity;
// MaxPos is AVXGetMaxPos on an AVX2/64-bit build and a scalar loop everywhere
// else. Both must agree exactly on the value AND on the index, and both must
// hand ties to the FIRST occurrence - the softmax spans and the argmax callers
// depend on that. Sizes straddle the 16-element block width and its tail.
const
  Sizes: array[0..11] of integer = (1, 2, 7, 15, 16, 17, 31, 32, 33, 64, 65, 517);
var
  Buf: array of TNeuralFloat;
  SI, K, N, GotPos, RefPos: integer;
  GotVal, RefVal: TNeuralFloat;
  Tag: string;
begin
  RandSeed := 271828;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    Tag := ' (N=' + IntToStr(N) + ')';
    SetLength(Buf, N);
    for K := 0 to N - 1 do Buf[K] := (Random - 0.5) * 20;

    RefVal := Buf[0];
    RefPos := 0;
    for K := 1 to N - 1 do
      if Buf[K] > RefVal then
      begin
        RefVal := Buf[K];
        RefPos := K;
      end;

    GotVal := TNNetVolume.MaxPos(TNeuralFloatArrPtr(@Buf[0]), N, GotPos);
    AssertEquals('max value' + Tag, RefVal, GotVal, 0.0);
    AssertEquals('max index' + Tag, RefPos, GotPos);
    AssertEquals('MaxValue agrees' + Tag, RefVal,
      TNNetVolume.MaxValue(TNeuralFloatArrPtr(@Buf[0]), N), 0.0);

    // All-equal: the first index has to win on every path.
    for K := 0 to N - 1 do Buf[K] := -3.5;
    GotVal := TNNetVolume.MaxPos(TNeuralFloatArrPtr(@Buf[0]), N, GotPos);
    AssertEquals('flat value' + Tag, -3.5, GotVal, 0.0);
    AssertEquals('flat ties go to index 0' + Tag, 0, GotPos);

    // A duplicated maximum: still the earlier of the two.
    if N >= 4 then
    begin
      Buf[1] := 9.25;
      Buf[N - 1] := 9.25;
      GotVal := TNNetVolume.MaxPos(TNeuralFloatArrPtr(@Buf[0]), N, GotPos);
      AssertEquals('duplicate max value' + Tag, 9.25, GotVal, 0.0);
      AssertEquals('duplicate max takes the first' + Tag, 1, GotPos);
    end;
  end;
  // Empty run: no element to point at.
  AssertEquals('empty run value', 0.0,
    TNNetVolume.MaxPos(TNeuralFloatArrPtr(@Buf[0]), 0, GotPos), 0.0);
  AssertEquals('empty run index', -1, GotPos);
end;

procedure TTestNeuralVolume.TestVolumeAddSubValueParity;
// TNNetVolume routes the whole-volume Add(Value)/Sub(Value) through the
// AddScalar kernel while TVolume keeps its element loop. The two must stay
// BIT-identical: Sub adds the negated value, and x - v equals x + (-v) exactly
// in IEEE-754. Sizes straddle the kernel's 32-element block and its tail.
const
  Sizes: array[0..7] of integer = (1, 8, 31, 32, 33, 63, 100, 1000);
  cValue = 0.7853981634;
var
  V: TNNetVolume;
  RefAdd, RefBack: array of TNeuralFloat;
  SI, K, N: integer;
  Delta: TNeuralFloat;
  Tag: string;
begin
  RandSeed := 6180339;
  Delta := cValue;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    Tag := ' (N=' + IntToStr(N) + ')';
    SetLength(RefAdd, N);
    SetLength(RefBack, N);
    V := TNNetVolume.Create(1, 1, N);
    try
      // Every reference value lands in a TNeuralFloat before the comparison, so
      // the expectation is single-precision arithmetic, not the double the
      // expression would otherwise be evaluated in.
      for K := 0 to N - 1 do
      begin
        V.FData[K] := (Random - 0.5) * 40;
        RefAdd[K] := V.FData[K] + Delta;
        RefBack[K] := RefAdd[K] - Delta;
      end;
      V.Add(Delta);
      for K := 0 to N - 1 do
        AssertEquals('Add[' + IntToStr(K) + ']' + Tag, RefAdd[K],
          V.FData[K], 0.0);
      V.Sub(Delta);
      for K := 0 to N - 1 do
        AssertEquals('Add then Sub[' + IntToStr(K) + ']' + Tag,
          RefBack[K], V.FData[K], 0.0);
    finally
      V.Free;
    end;
  end;
end;

procedure TTestNeuralVolume.TestVolumeRankOneUpdateRowParity;
// TNNetVolume.RankOneUpdateRow runs the AVX kernels while TVolume composes its
// scalar element loops. Both must land BIT-exactly on the reference rounding
// sequence built here: round(Prev*Alpha), round(B*BScale), then round of their
// sum -- which is what the composed scalar loops do and what the MulMulAdd
// kernel's separate vmulps pair plus vaddps do. All three contract cases are
// covered: no previous row (Prev = nil, and a zero Alpha), the disjoint carry,
// and the aliased PtrPrev = PtrDst in-place carry every recurrent scan uses.
// Sizes straddle the kernels' 32-element block, their 4-element small loop and
// the scalar tail.
const
  Sizes: array[0..8] of integer = (1, 3, 4, 8, 31, 32, 33, 100, 1000);
  cAlpha: TNeuralFloat = 0.96875;
  cBScale: TNeuralFloat = -0.3125;
var
  Dst, Ref, Prev, B: array of TNeuralFloat;
  SI, K, N, Rep: integer;
  T1, T2: TNeuralFloat;
  Tag: string;
begin
  RandSeed := 24011966;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    Tag := ' (N=' + IntToStr(N) + ')';
    SetLength(Dst, N);
    SetLength(Ref, N);
    SetLength(Prev, N);
    SetLength(B, N);
    for K := 0 to N - 1 do
    begin
      Prev[K] := (Random - 0.5) * 6;
      B[K] := (Random - 0.5) * 6;
      Dst[K] := -999;
    end;

    // Case 1 - no previous row at all.
    for K := 0 to N - 1 do Ref[K] := B[K] * cBScale;
    TNNetVolume.RankOneUpdateRow(TNeuralFloatArrPtr(@Dst[0]), nil,
      TNeuralFloatArrPtr(@B[0]), cAlpha, cBScale, N);
    for K := 0 to N - 1 do
      AssertEquals('nil prev[' + IntToStr(K) + ']' + Tag, Ref[K], Dst[K], 0.0);

    // Case 2 - a zero carry scale takes the same branch with a real Prev.
    TNNetVolume.RankOneUpdateRow(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Prev[0]), TNeuralFloatArrPtr(@B[0]), 0, cBScale, N);
    for K := 0 to N - 1 do
      AssertEquals('zero alpha[' + IntToStr(K) + ']' + Tag, Ref[K], Dst[K], 0.0);

    // Case 3 - disjoint Prev and Dst.
    for K := 0 to N - 1 do
    begin
      T1 := Prev[K] * cAlpha;
      T2 := B[K] * cBScale;
      Ref[K] := T1 + T2;
    end;
    TNNetVolume.RankOneUpdateRow(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Prev[0]), TNeuralFloatArrPtr(@B[0]),
      cAlpha, cBScale, N);
    for K := 0 to N - 1 do
      AssertEquals('carry[' + IntToStr(K) + ']' + Tag, Ref[K], Dst[K], 0.0);

    // Case 4 - the in-place form, run three times so a state carry that drifts
    // from the reference cannot cancel itself out.
    for K := 0 to N - 1 do
    begin
      Ref[K] := Prev[K];
      Dst[K] := Prev[K];
    end;
    for Rep := 1 to 3 do
    begin
      for K := 0 to N - 1 do
      begin
        T1 := Ref[K] * cAlpha;
        T2 := B[K] * cBScale;
        Ref[K] := T1 + T2;
      end;
      TNNetVolume.RankOneUpdateRow(TNeuralFloatArrPtr(@Dst[0]),
        TNeuralFloatArrPtr(@Dst[0]), TNeuralFloatArrPtr(@B[0]),
        cAlpha, cBScale, N);
    end;
    for K := 0 to N - 1 do
      AssertEquals('in-place carry[' + IntToStr(K) + ']' + Tag,
        Ref[K], Dst[K], 0.0);
  end;
end;

procedure TTestNeuralVolume.TestVolumeAdamDeltaParity;
// AdamDelta fuses the eleven-pass Adam composition that TNNetNeuron.CalcAdamDelta
// used to run inline. The reference here is that exact composition, built from
// the same TNNetVolume primitives in the same order, so the assertion is
// BIT-identity at tolerance 0.0 rather than a numeric tolerance: neither path
// uses FMA, so every multiply and every add rounds at the same point.
// Five consecutive steps are driven with both moments accumulating and the
// bias-correction denominators moving, so a fused kernel that got the moment
// recurrence subtly wrong could not hide behind a single step. Sizes straddle
// the kernel's 8-element block and its scalar tail.
const
  Sizes: array[0..8] of integer = (1, 3, 7, 8, 9, 16, 31, 64, 517);
  cB1 = 0.9;
  cB2 = 0.999;
  cEps = 1e-8;
  cLR = 0.01;
  cSteps = 5;
var
  D, M, V, RefD, RefM, RefV, Scratch: TNNetVolume;
  G: array of TNeuralFloat;
  SI, K, N, Step: integer;
  B1Decay, B2Decay, OmB1D, OmB2D: TNeuralFloat;
  Tag: string;
begin
  RandSeed := 31415926;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(G, N);
    D := TNNetVolume.Create(1, 1, N);
    M := TNNetVolume.Create(1, 1, N);
    V := TNNetVolume.Create(1, 1, N);
    RefD := TNNetVolume.Create(1, 1, N);
    RefM := TNNetVolume.Create(1, 1, N);
    RefV := TNNetVolume.Create(1, 1, N);
    Scratch := TNNetVolume.Create(1, 1, N);
    try
      // Both moments start non-zero so the recurrence is exercised from the
      // first step, and one gradient slot is an exact zero.
      for K := 0 to N - 1 do
      begin
        M.FData[K] := (Random - 0.5) * 0.4;
        V.FData[K] := Random * 0.3;
        RefM.FData[K] := M.FData[K];
        RefV.FData[K] := V.FData[K];
      end;
      B1Decay := 1;
      B2Decay := 1;

      for Step := 1 to cSteps do
      begin
        Tag := ' (N=' + IntToStr(N) + ' step ' + IntToStr(Step) + ')';
        for K := 0 to N - 1 do
        begin
          if K mod 9 = 4 then G[K] := 0
          else G[K] := ((K mod 7) - 3) * 0.25 * Step;
          D.FData[K] := G[K];
          RefD.FData[K] := G[K];
        end;
        B1Decay := B1Decay * cB1;
        B2Decay := B2Decay * cB2;
        OmB1D := 1 - B1Decay;
        OmB2D := 1 - B2Decay;

        // Reference: the original composition, primitive for primitive.
        Scratch.Copy(RefD);
        Scratch.Mul(Scratch);
        RefM.MulMulAdd(cB1, 1 - cB1, RefD);
        RefV.MulMulAdd(cB2, 1 - cB2, Scratch);
        Scratch.Copy(RefV);
        Scratch.Mul(1.0 / OmB2D);
        Scratch.VSqrt();
        Scratch.Add(cEps);
        RefD.Fill(cLR / OmB1D);
        RefD.Mul(RefM);
        RefD.Divi(Scratch);

        TNNetVolume.AdamDelta(D.DataPtr, M.DataPtr, V.DataPtr,
          cB1, 1 - cB1, cB2, 1 - cB2, 1.0 / OmB2D, cEps, cLR / OmB1D, N);

        for K := 0 to N - 1 do
        begin
          AssertEquals('first moment[' + IntToStr(K) + ']' + Tag,
            RefM.FData[K], M.FData[K], 0.0);
          AssertEquals('second moment[' + IntToStr(K) + ']' + Tag,
            RefV.FData[K], V.FData[K], 0.0);
          AssertEquals('delta[' + IntToStr(K) + ']' + Tag,
            RefD.FData[K], D.FData[K], 0.0);
        end;
      end;

      // A zero-length run must touch nothing.
      D.FData[0] := 5;
      TNNetVolume.AdamDelta(D.DataPtr, M.DataPtr, V.DataPtr,
        cB1, 1 - cB1, cB2, 1 - cB2, 1.0, cEps, cLR, 0);
      AssertEquals('empty run', 5.0, D.FData[0], 0.0);
    finally
      D.Free; M.Free; V.Free;
      RefD.Free; RefM.Free; RefV.Free; Scratch.Free;
    end;
  end;
end;

procedure TTestNeuralVolume.TestVolumeAdafactorDeltaParity;
// AdafactorDelta fuses the per-element sqrt/divide loop that
// TNNetNeuron.CalcAdafactorDelta runs on its unfactored branch (every fully
// connected / embedding layer). The reference here is that loop written out
// element by element in the SAME operation order, so the assertion is
// BIT-identity at tolerance 0.0: neither path uses FMA and every intermediate
// is a TNeuralFloat, so both round at the same points. Five consecutive steps
// let the second moment accumulate, so a kernel that got the recurrence wrong
// could not hide behind one step. Sizes straddle the 8-element block and the
// scalar tail; one delta slot per row is an exact zero.
const
  Sizes: array[0..8] of integer = (1, 3, 7, 8, 9, 16, 31, 64, 517);
  cB2 = 0.999;
  cEps = 1e-8;
  cLR = 0.01;
  cSteps = 5;
var
  D, V, RefD, RefV: TNNetVolume;
  SI, K, N, Step: integer;
  invNegLr, kAF, cAF, d0, t1, t2, vNew, B2, Eps: TNeuralFloat;
  Tag: string;
begin
  RandSeed := 27182818;
  // The reference loop must see the SAME single-precision scalars the kernel
  // broadcasts; an untyped const would let FPC evaluate its multiply in Double
  // and round once more than the kernel does.
  B2 := cB2;
  Eps := cEps;
  invNegLr := -1.0 / cLR;
  kAF := (1 - B2) * (invNegLr * invNegLr);
  cAF := (1 - B2) * Eps;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    D := TNNetVolume.Create(1, 1, N);
    V := TNNetVolume.Create(1, 1, N);
    RefD := TNNetVolume.Create(1, 1, N);
    RefV := TNNetVolume.Create(1, 1, N);
    try
      for K := 0 to N - 1 do
      begin
        V.FData[K] := Random * 0.3;
        RefV.FData[K] := V.FData[K];
      end;

      for Step := 1 to cSteps do
      begin
        Tag := ' (N=' + IntToStr(N) + ' step ' + IntToStr(Step) + ')';
        for K := 0 to N - 1 do
        begin
          if K mod 9 = 4 then D.FData[K] := 0
          else D.FData[K] := ((K mod 7) - 3) * 0.25 * Step;
          RefD.FData[K] := D.FData[K];
        end;

        // Reference: the fused recurrence, spelled out one element at a time.
        for K := 0 to N - 1 do
        begin
          d0 := RefD.FData[K];
          t1 := d0 * d0;
          t1 := kAF * t1;
          t1 := t1 + cAF;
          t2 := B2 * RefV.FData[K];
          vNew := t1 + t2;
          RefV.FData[K] := vNew;
          t1 := Sqrt(vNew);
          t1 := t1 + Eps;
          RefD.FData[K] := d0 / t1;
        end;

        TNNetVolume.AdafactorDelta(D.DataPtr, V.DataPtr,
          B2, kAF, cAF, Eps, N);

        for K := 0 to N - 1 do
        begin
          AssertEquals('second moment[' + IntToStr(K) + ']' + Tag,
            RefV.FData[K], V.FData[K], 0.0);
          AssertEquals('delta[' + IntToStr(K) + ']' + Tag,
            RefD.FData[K], D.FData[K], 0.0);
        end;
      end;

      // A zero-length run must touch nothing.
      D.FData[0] := 5;
      TNNetVolume.AdafactorDelta(D.DataPtr, V.DataPtr, B2, kAF, cAF, Eps, 0);
      AssertEquals('empty run', 5.0, D.FData[0], 0.0);
    finally
      D.Free; V.Free; RefD.Free; RefV.Free;
    end;
  end;
end;

procedure TTestNeuralVolume.TestVolumeClampAbsParity;
// ClampAbs is a vmaxps/vminps pair on an AVX2/64-bit build and a two-branch
// scalar loop everywhere else; the two must agree bit for bit, which is why the
// kernel puts the bound in the FIRST operand of both instructions (x86 min/max
// return the second operand when a compare is unordered, so a NaN passes
// through untouched, exactly as the scalar "if v > b / else if v < -b" form
// leaves it). The inputs below therefore cover: values inside the band, values
// on both saturating sides, the exact boundaries +/-Value (which must NOT be
// rewritten), a signed zero and an infinity on each side. Sizes straddle the
// 8-element block and the scalar tail. NaN is deliberately NOT fed in: MAXPS
// signals Invalid Operation on a QNaN source exactly as the scalar COMISS
// does, so both paths raise EInvalidOp under FPC's default unmasked exceptions
// - identical behaviour, but not something a test can assert a value for.
const
  Sizes: array[0..8] of integer = (1, 3, 7, 8, 9, 16, 31, 64, 517);
  cBound = 0.75;
var
  Buf, Ref: array of TNeuralFloat;
  SI, K, N: integer;
  v: TNeuralFloat;
  Tag: string;
begin
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(Buf, N);
    SetLength(Ref, N);
    Tag := ' (N=' + IntToStr(N) + ')';
    for K := 0 to N - 1 do
    begin
      case K mod 8 of
        0: v := 0.1;
        1: v := -0.1;
        2: v := 3.5;
        3: v := -3.5;
        4: v := cBound;         // exactly the bound: must survive untouched
        5: v := -cBound;
        6: v := 0.0;
        else v := -0.0;
      end;
      Buf[K] := v;
      Ref[K] := v;
    end;
    if N > 16 then
    begin
      Buf[10] := Infinity;  Ref[10] := Infinity;
      Buf[11] := -Infinity; Ref[11] := -Infinity;
    end;

    // Reference: the scalar clamp the layer used to run inline.
    for K := 0 to N - 1 do
    begin
      if Ref[K] > cBound then Ref[K] := cBound
      else if Ref[K] < -cBound then Ref[K] := -cBound;
    end;

    TNNetVolume.ClampAbs(TNeuralFloatArrPtr(@Buf[0]), cBound, N);

    for K := 0 to N - 1 do
      AssertEquals('ClampAbs[' + IntToStr(K) + ']' + Tag,
        Ref[K], Buf[K], 0.0);

    // A non-positive bound and a zero count are both no-ops.
    Buf[0] := 9;
    TNNetVolume.ClampAbs(TNeuralFloatArrPtr(@Buf[0]), 0, N);
    AssertEquals('zero bound is a no-op' + Tag, 9.0, Buf[0], 0.0);
    TNNetVolume.ClampAbs(TNeuralFloatArrPtr(@Buf[0]), cBound, 0);
    AssertEquals('empty run' + Tag, 9.0, Buf[0], 0.0);
  end;
end;

procedure TTestNeuralVolume.TestVolumeForceMaxRangeParity;
// ForceMaxRange now hands a positive bound to the ClampAbs kernel. The
// reference is NeuronForceRange's own two-branch chain, so the assertion is
// bit-identity: values inside the bound, exactly on it, and beyond it on both
// signs, plus the infinities. NaN is deliberately absent: the debug build traps
// invalid FP compares, so NaN handling is asserted where it can be observed
// without one - see TestVolumeHasNonFiniteBitTest. A non-positive bound is
// outside the kernel's contract and
// still takes the scalar path, which zeroes the whole volume.
var
  Vol: TNNetVolume;
  K, N: integer;
  Ref: array of TNeuralFloat;
  v: TNeuralFloat;
begin
  N := 37;
  Vol := TNNetVolume.Create(N, 1, 1);
  SetLength(Ref, N);
  try
    for K := 0 to N - 1 do
    begin
      case K mod 7 of
        0: v := 0.25;
        1: v := -0.25;
        2: v := 9.0;
        3: v := -9.0;
        4: v := 2.0;          // exactly the bound
        5: v := -2.0;
        else v := 0.0;
      end;
      Vol.Raw[K] := v;
      Ref[K] := NeuronForceRange(v, 2.0);
    end;
    Vol.Raw[10] := Infinity;   Ref[10] := NeuronForceRange(Infinity, 2.0);
    Vol.Raw[11] := -Infinity;  Ref[11] := NeuronForceRange(-Infinity, 2.0);

    Vol.ForceMaxRange(2.0);
    for K := 0 to N - 1 do
    begin
      AssertEquals('ForceMaxRange[' + IntToStr(K) + ']', Ref[K], Vol.Raw[K], 0.0);
    end;

    // A zero bound keeps the historical scalar behaviour: everything collapses.
    Vol.Raw[0] := 5;
    Vol.Raw[1] := -5;
    Vol.ForceMaxRange(0);
    AssertEquals('zero bound clamps up', 0.0, Vol.Raw[0], 0.0);
    AssertEquals('zero bound clamps down', 0.0, Vol.Raw[1], 0.0);
  finally
    Vol.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeForceMaxAbs;
// ForceMaxAbs rescales the whole volume so that the largest magnitude lands on
// the bound, keeping every ratio between cells; a volume already inside the
// bound is left byte for byte as it was.
var
  Vol: TNNetVolume;
begin
  Vol := TNNetVolume.Create(4, 1, 1);
  try
    Vol.Raw[0] := 2.0;
    Vol.Raw[1] := -8.0;
    Vol.Raw[2] := 0.0;
    Vol.Raw[3] := 4.0;
    Vol.ForceMaxAbs(2.0);
    AssertEquals('scaled max abs', 2.0, Vol.GetMaxAbs(), 0.0001);
    AssertEquals('scaled cell 0', 0.5, Vol.Raw[0], 0.0001);
    AssertEquals('scaled cell 1', -2.0, Vol.Raw[1], 0.0001);
    AssertEquals('scaled cell 2', 0.0, Vol.Raw[2], 0.0);
    AssertEquals('scaled cell 3', 1.0, Vol.Raw[3], 0.0001);

    // Already within the bound: no scaling at all.
    Vol.Raw[0] := 0.25;
    Vol.Raw[1] := -1.5;
    Vol.Raw[2] := 0.0;
    Vol.Raw[3] := 1.0;
    Vol.ForceMaxAbs(2.0);
    AssertEquals('in range cell 0', 0.25, Vol.Raw[0], 0.0);
    AssertEquals('in range cell 1', -1.5, Vol.Raw[1], 0.0);
    AssertEquals('in range cell 2', 0.0, Vol.Raw[2], 0.0);
    AssertEquals('in range cell 3', 1.0, Vol.Raw[3], 0.0);
  finally
    Vol.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeHasNonFiniteBitTest;
// HasNonFinite classifies binary32 by masking the exponent field instead of
// calling IsNan/IsInfinite per element. The two must agree on every class the
// scan can meet: normals, zeros, denormals (finite - the exponent field is 0),
// the largest finite value, both infinities and a NaN.
const
  cProbes: array[0..8] of TNeuralFloat =
    (0.0, -0.0, 1.0, -1.0, 3.4e38, 1.0e-40, -1.0e-40, 1.17549435e-38, 123456.75);
var
  Vol: TNNetVolume;
  K: integer;
begin
  Vol := TNNetVolume.Create(Length(cProbes), 1, 1);
  try
    for K := 0 to High(cProbes) do Vol.Raw[K] := cProbes[K];
    AssertFalse('finite probes (denormals included) are finite', Vol.HasNonFinite());

    for K := 0 to High(cProbes) do
    begin
      // One slot at a time goes non-finite, so the scan has to find it wherever
      // it sits rather than only at the head of the buffer.
      Vol.Raw[K] := NaN;
      AssertTrue('NaN at ' + IntToStr(K), Vol.HasNonFinite());
      Vol.Raw[K] := Infinity;
      AssertTrue('+Inf at ' + IntToStr(K), Vol.HasNonFinite());
      Vol.Raw[K] := -Infinity;
      AssertTrue('-Inf at ' + IntToStr(K), Vol.HasNonFinite());
      Vol.Raw[K] := cProbes[K];
      AssertFalse('restored at ' + IntToStr(K), Vol.HasNonFinite());
    end;
  finally
    Vol.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeLionDeltaParity;
// LionDelta fuses the interpolation, the momentum EMA and the three-valued sign
// select that TNNetNeuron.CalcLionDelta ran element by element. The reference
// here is that loop in the SAME operation order, so the assertion is
// BIT-identity at tolerance 0.0: neither path uses FMA, and the kernel's two
// vcmpps masks reproduce the "> 0 / < 0 / else 0" chain exactly, +0.0 included.
// The gradient pattern deliberately drives c through zero and back so all three
// select arms fire, and one delta slot per row is an exact zero. Five
// consecutive steps let the single momentum buffer accumulate. Sizes straddle
// the 8-element block and the scalar tail.
const
  Sizes: array[0..8] of integer = (1, 3, 7, 8, 9, 16, 31, 64, 517);
  cB1 = 0.9;
  cB2 = 0.99;
  cLR = 0.01;
  cSteps = 5;
var
  D, M, RefD, RefM: TNNetVolume;
  SI, K, N, Step: integer;
  B1, B2, invNegLr, k1, k2, negLr, posLr, dv, mv, cv: TNeuralFloat;
  Tag: string;
begin
  RandSeed := 16180339;
  // Typed singles: an untyped const would let FPC evaluate the reference's
  // multiplies in Double and round once more than the kernel does.
  B1 := cB1;
  B2 := cB2;
  posLr := cLR;
  negLr := -cLR;
  invNegLr := -1.0 / cLR;
  k1 := (1 - B1) * invNegLr;
  k2 := (1 - B2) * invNegLr;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    D := TNNetVolume.Create(1, 1, N);
    M := TNNetVolume.Create(1, 1, N);
    RefD := TNNetVolume.Create(1, 1, N);
    RefM := TNNetVolume.Create(1, 1, N);
    try
      for K := 0 to N - 1 do
      begin
        M.FData[K] := (Random - 0.5) * 0.4;
        RefM.FData[K] := M.FData[K];
      end;

      for Step := 1 to cSteps do
      begin
        Tag := ' (N=' + IntToStr(N) + ' step ' + IntToStr(Step) + ')';
        for K := 0 to N - 1 do
        begin
          if K mod 9 = 4 then D.FData[K] := 0
          else if Step mod 2 = 0 then D.FData[K] := ((K mod 7) - 3) * 0.05
          else D.FData[K] := (3 - (K mod 7)) * 0.05;
          RefD.FData[K] := D.FData[K];
        end;

        // Reference: the loop this kernel replaced.
        for K := 0 to N - 1 do
        begin
          dv := RefD.FData[K];
          mv := RefM.FData[K];
          cv := B1 * mv + k1 * dv;
          RefM.FData[K] := B2 * mv + k2 * dv;
          if cv > 0 then RefD.FData[K] := negLr
          else if cv < 0 then RefD.FData[K] := posLr
          else RefD.FData[K] := 0;
        end;

        TNNetVolume.LionDelta(D.DataPtr, M.DataPtr,
          B1, k1, B2, k2, negLr, posLr, N);

        for K := 0 to N - 1 do
        begin
          AssertEquals('momentum[' + IntToStr(K) + ']' + Tag,
            RefM.FData[K], M.FData[K], 0.0);
          AssertEquals('delta[' + IntToStr(K) + ']' + Tag,
            RefD.FData[K], D.FData[K], 0.0);
        end;
      end;

      // An exactly-zero c must select the zero arm, not a learning rate.
      M.Fill(0);
      D.Fill(0);
      TNNetVolume.LionDelta(D.DataPtr, M.DataPtr,
        B1, k1, B2, k2, negLr, posLr, N);
      for K := 0 to N - 1 do
        AssertEquals('zero c selects zero' + Tag, 0.0, D.FData[K], 0.0);

      // A zero-length run must touch nothing.
      D.FData[0] := 5;
      TNNetVolume.LionDelta(D.DataPtr, M.DataPtr,
        B1, k1, B2, k2, negLr, posLr, 0);
      AssertEquals('empty run', 5.0, D.FData[0], 0.0);
    finally
      D.Free; M.Free; RefD.Free; RefM.Free;
    end;
  end;
end;

procedure TTestNeuralVolume.TestVolumeExpShiftSumParity;
// ExpShiftSum is one fused AVX2/64-bit kernel (broadcast subtract,
// 8-wide exp, in-register reduction) and a plain scalar loop on every other
// build; both must reproduce the stable-softmax numerator/denominator the
// attention layers used to compute element by element. Sizes straddle the
// 8-element block width and its tail. Two properties are load-bearing beyond
// raw accuracy: an argument far below the exp underflow point must come back
// EXACTLY zero (that is what keeps an additive -1e9 attention mask at a hard
// zero weight, which the value-sum loops skip on), and dst may alias src.
const
  Sizes: array[0..10] of integer = (1, 2, 7, 8, 9, 15, 16, 17, 31, 64, 517);
  cShift = 1.75;
var
  Src, Dst, Ref: array of TNeuralFloat;
  SI, K, N: integer;
  RefSum, GotSum, V: TNeuralFloat;
  Tag: string;
begin
  RandSeed := 314159;
  for SI := 0 to High(Sizes) do
  begin
    N := Sizes[SI];
    SetLength(Src, N);
    SetLength(Dst, N);
    SetLength(Ref, N);
    for K := 0 to N - 1 do Src[K] := (Random - 0.5) * 12;
    // Two masked entries (the additive attention mask sentinel).
    if N > 6 then
    begin
      Src[1] := -1e9;
      Src[N - 2] := -1e9;
    end;
    RefSum := 0;
    for K := 0 to N - 1 do
    begin
      Ref[K] := NeuralExp(Src[K] - cShift);
      RefSum := RefSum + Ref[K];
    end;

    Tag := ' (N=' + IntToStr(N) + ')';
    GotSum := TNNetVolume.ExpShiftSum(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Src[0]), cShift, N);
    for K := 0 to N - 1 do
      AssertEquals('exp[' + IntToStr(K) + ']' + Tag, Ref[K], Dst[K],
        1e-6 * Ref[K] + 1e-30);
    AssertEquals('sum' + Tag, RefSum, GotSum, 1e-5 * RefSum);
    if N > 6 then
    begin
      AssertEquals('masked head is a hard zero' + Tag, 0.0, Dst[1], 0.0);
      AssertEquals('masked tail is a hard zero' + Tag, 0.0, Dst[N - 2], 0.0);
    end;

    // In place: dst = src must give the same answer.
    for K := 0 to N - 1 do Dst[K] := Src[K];
    GotSum := TNNetVolume.ExpShiftSum(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralFloatArrPtr(@Dst[0]), cShift, N);
    for K := 0 to N - 1 do
    begin
      V := Ref[K];
      AssertEquals('in-place exp[' + IntToStr(K) + ']' + Tag, V, Dst[K],
        1e-6 * V + 1e-30);
    end;
    AssertEquals('in-place sum' + Tag, RefSum, GotSum, 1e-5 * RefSum);
  end;
  // A zero-length run is a no-op that sums to zero.
  AssertEquals('empty run', 0.0, TNNetVolume.ExpShiftSum(
    TNeuralFloatArrPtr(@Src[0]), TNeuralFloatArrPtr(@Src[0]), cShift, 0), 0.0);
end;

procedure TTestNeuralVolume.TestVolumeFlip;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(3, 1, 1);
  try
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 3.0;
    
    V.FlipX();
    
    AssertEquals('After FlipX, first value should be 3.0', 3.0, V.Raw[0], 0.0001);
    AssertEquals('After FlipX, middle value should be 2.0', 2.0, V.Raw[1], 0.0001);
    AssertEquals('After FlipX, last value should be 1.0', 1.0, V.Raw[2], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeClassification;
var
  V: TNNetVolume;
  PredictedClass: integer;
begin
  V := TNNetVolume.Create(5, 1, 1);
  try
    // Simulate classification output with class 2 having highest probability
    V.Raw[0] := 0.1;
    V.Raw[1] := 0.2;
    V.Raw[2] := 0.5;
    V.Raw[3] := 0.15;
    V.Raw[4] := 0.05;
    
    PredictedClass := V.GetClass();
    AssertEquals('Predicted class should be 2', 2, PredictedClass);
    
    // Test SetClass with single value parameter
    // SetClass(class, value) fills non-class elements with -value
    // This is useful for hyperbolic tangent activations (-1 to +1 range)
    V.SetClass(3, 1.0);
    AssertEquals('After SetClass(3), class 3 should be 1.0', 1.0, V.Raw[3], 0.0001);
    AssertEquals('After SetClass(3), class 0 should be -1.0', -1.0, V.Raw[0], 0.0001);
    
    // Test SetClass with explicit true/false values (two-parameter overload)
    // This allows standard one-hot encoding (0 for false, 1 for true)
    V.SetClass(2, 1.0, 0.0);
    AssertEquals('After SetClass(2, 1.0, 0.0), class 2 should be 1.0', 1.0, V.Raw[2], 0.0001);
    AssertEquals('After SetClass(2, 1.0, 0.0), class 0 should be 0.0', 0.0, V.Raw[0], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSoftMax;
var
  V: TNNetVolume;
  SumAfterSoftMax: TNeuralFloat;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 3.0;
    V.Raw[3] := 4.0;
    
    V.SoftMax();
    
    SumAfterSoftMax := V.GetSum();
    // SoftMax output should sum to 1.0
    AssertEquals('SoftMax output should sum to 1.0', 1.0, SumAfterSoftMax, 0.0001);
    
    // Higher input values should have higher probabilities
    AssertTrue('V[3] should be greater than V[0]', V.Raw[3] > V.Raw[0]);
    AssertTrue('V[3] should be greater than V[1]', V.Raw[3] > V.Raw[1]);
    AssertTrue('V[3] should be greater than V[2]', V.Raw[3] > V.Raw[2]);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSoftMaxConstantInput;
// A constant vector has no preferred element, so softmax must return the
// uniform distribution 1/N and a total sum of N (N * exp(0)).
var
  V: TNNetVolume;
  N, I: integer;
  TotalSum: TNeuralFloat;
begin
  N := 5;
  V := TNNetVolume.Create(N, 1, 1);
  try
    V.Fill(3.5);
    TotalSum := V.SoftMax();
    AssertEquals('Constant vector SoftMax total sum', N * 1.0, TotalSum, 1e-6);
    for I := 0 to N - 1 do
      AssertEquals('Constant vector SoftMax at ' + IntToStr(I),
        1.0 / N, V.Raw[I], 1e-6);
    AssertEquals('Constant vector SoftMax sums to 1', 1.0, V.GetSum(), 1e-6);

    // An all-zero vector is the same degenerate case and must not stay at zero.
    V.Fill(0);
    V.SoftMax();
    for I := 0 to N - 1 do
      AssertEquals('Zero vector SoftMax at ' + IntToStr(I),
        1.0 / N, V.Raw[I], 1e-6);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSoftMaxParity;
// Verifies the (possibly AVX) TVolume.SoftMax against an independent scalar
// stable-softmax reference, element by element, within 1e-4.
var
  V: TNNetVolume;
  Ref: array of TNeuralFloat;
  N, I: integer;
  MaxV, MinV, S: TNeuralFloat;
begin
  N := 37; // not a multiple of 8 to exercise the AVXExp remainder tail
  V := TNNetVolume.Create(N, 1, 1);
  SetLength(Ref, N);
  try
    RandSeed := 424242;
    for I := 0 to N - 1 do
    begin
      V.Raw[I] := (Random - 0.5) * 20.0;
      Ref[I] := V.Raw[I];
    end;

    // Independent scalar reference mirroring TVolume.SoftMax semantics.
    MaxV := Ref[0];
    for I := 1 to N - 1 do if Ref[I] > MaxV then MaxV := Ref[I];
    if MaxV <> 0 then for I := 0 to N - 1 do Ref[I] := Ref[I] - MaxV;
    MinV := Ref[0];
    for I := 1 to N - 1 do if Ref[I] < MinV then MinV := Ref[I];
    if MinV <> 0 then
    begin
      if MinV < -1000 then
        for I := 0 to N - 1 do Ref[I] := Ref[I] * (-1000 / MinV);
      S := 0;
      for I := 0 to N - 1 do
      begin
        Ref[I] := Exp(NeuronForceRange(Ref[I], 4000));
        S := S + Ref[I];
      end;
      if S > 0 then for I := 0 to N - 1 do Ref[I] := Ref[I] / S;
    end;

    V.SoftMax();

    for I := 0 to N - 1 do
      AssertEquals('SoftMax parity at ' + IntToStr(I), Ref[I], V.Raw[I], 1e-4);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumePointwiseSoftMaxParity;
// Verifies TVolume.PointwiseSoftMax (per-(x,y) over the depth axis) against an
// independent scalar stable-softmax reference, within 1e-4.
var
  V: TNNetVolume;
  SX, SY, D, X, Y, K, Base: integer;
  Ref: array of TNeuralFloat;
  MaxV, S: TNeuralFloat;
begin
  SX := 3; SY := 2; D := 13; // depth not a multiple of 8 -> AVXExp tail
  V := TNNetVolume.Create(SX, SY, D);
  SetLength(Ref, SX * SY * D);
  try
    RandSeed := 99;
    for K := 0 to SX * SY * D - 1 do
    begin
      V.Raw[K] := (Random - 0.5) * 16.0;
      Ref[K] := V.Raw[K];
    end;

    // Independent per-(x,y) scalar reference over the contiguous depth span.
    for X := 0 to SX - 1 do
      for Y := 0 to SY - 1 do
      begin
        Base := V.GetRawPos(X, Y, 0);
        MaxV := Ref[Base];
        for K := 1 to D - 1 do
          if Ref[Base + K] > MaxV then MaxV := Ref[Base + K];
        S := 0;
        for K := 0 to D - 1 do
        begin
          Ref[Base + K] := Exp(NeuronForceRange(Ref[Base + K] - MaxV, 4000));
          S := S + Ref[Base + K];
        end;
        if S > 0 then
          for K := 0 to D - 1 do Ref[Base + K] := Ref[Base + K] / S;
      end;

    V.PointwiseSoftMax();

    for K := 0 to SX * SY * D - 1 do
      AssertEquals('PointwiseSoftMax parity at ' + IntToStr(K),
        Ref[K], V.Raw[K], 1e-4);
  finally
    V.Free;
  end;
end;

// GroupedDotProductsTiled caches a raw row pointer per A row. The cache must
// follow the source volume: a second call with a DIFFERENT VAs of the same row
// count has to read the new rows, not the pointers cached from the first one.
procedure TTestNeuralVolume.TestGroupedDotProductsTiledRebuildsOnNewSource;
const
  Groups = 2;
  NumAs = 4;
  NumBs = 3;
  VectorSize = 3;
var
  R: TNNetGroupedVolume;
  A1, A2, B: TNNetVolume;
  i: integer;
  First: array[0..NumAs * NumBs - 1] of TNeuralFloat;
begin
  R := TNNetGroupedVolume.Create(NumAs * NumBs, 1, 1);
  // Both sources stay alive, so they are guaranteed to be distinct buffers.
  A1 := TNNetVolume.Create(NumAs, 1, VectorSize);
  A2 := TNNetVolume.Create(NumAs, 1, VectorSize);
  B := TNNetVolume.Create(NumBs, 1, VectorSize * Groups);
  try
    A1.Fill(1);
    A2.Fill(2);
    for i := 0 to B.Size - 1 do B.FData[i] := 0.25 * (i + 1);
    R.Fill(0);
    R.GroupedDotProductsTiled(Groups, NumAs, NumBs, VectorSize, A1, B, 2, 3);
    for i := 0 to NumAs * NumBs - 1 do First[i] := R.FData[i];
    AssertTrue('First source produced non-zero output', R.GetSumAbs() > 0);
    R.Fill(0);
    R.GroupedDotProductsTiled(Groups, NumAs, NumBs, VectorSize, A2, B, 2, 3);
    // A2 is exactly twice A1, so every dot product must double.
    for i := 0 to NumAs * NumBs - 1 do
      AssertEquals('Second source element ' + IntToStr(i),
        2 * First[i], R.FData[i], 1e-5);
  finally
    B.Free;
    A2.Free;
    A1.Free;
    R.Free;
  end;
end;

// A tile size that does not divide the range leaves a partial trailing tile;
// the kernel must still cover the trailing rows/columns (it never zero-fills,
// so a skipped output keeps whatever was there before).
procedure TTestNeuralVolume.TestGroupedDotProductsTiledPartialTile;
const
  Groups = 1;
  NumAs = 3;
  NumBs = 3;
  VectorSize = 2;
var
  R: TNNetGroupedVolume;
  A, B: TNNetVolume;
  CntA, CntB, k: integer;
  Expected: TNeuralFloat;
begin
  R := TNNetGroupedVolume.Create(NumAs * NumBs, 1, 1);
  A := TNNetVolume.Create(NumAs, 1, VectorSize);
  B := TNNetVolume.Create(NumBs, 1, VectorSize * Groups);
  try
    for k := 0 to A.Size - 1 do A.FData[k] := 0.5 * (k + 1);
    for k := 0 to B.Size - 1 do B.FData[k] := 0.25 * (k + 2);
    R.Fill(-999);
    // 3 rows with tile 2 and 3 columns with tile 2: both axes end in a
    // one-wide partial tile.
    R.GroupedDotProductsTiled(Groups, NumAs, NumBs, VectorSize, A, B, 2, 2);
    for CntB := 0 to NumBs - 1 do
      for CntA := 0 to NumAs - 1 do
      begin
        Expected := 0;
        for k := 0 to VectorSize - 1 do
          Expected := Expected +
            A.FData[CntA * VectorSize + k] * B.FData[CntB * VectorSize + k];
        AssertEquals('Partial tile A=' + IntToStr(CntA) + ' B=' + IntToStr(CntB),
          Expected, R.FData[CntB * NumAs + CntA], 1e-5);
      end;
  finally
    B.Free;
    A.Free;
    R.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeGroupedPointwiseSoftMaxParity;
// Verifies TVolume.GroupedPointwiseSoftMax (per-(x,y), per-group over a
// contiguous slice of the depth axis) against an independent scalar stable
// softmax reference, within 1e-4. Depth is split so that ChannelsPerGroup is
// neither a multiple of 8 nor above csMinAvxSize in one of the cases, which
// exercises the vectorized group path and its scalar tail.
var
  V: TNNetVolume;
  SX, SY, D, Groups, ChPerGroup, X, Y, G, K, Base, Total: integer;
  Ref: array of TNeuralFloat;
  MaxV, S: TNeuralFloat;
begin
  SX := 3; SY := 2; Groups := 4; ChPerGroup := 11; // 11 is not a multiple of 8
  D := Groups * ChPerGroup;
  Total := SX * SY * D;
  V := TNNetVolume.Create(SX, SY, D);
  SetLength(Ref, Total);
  try
    RandSeed := 101;
    for K := 0 to Total - 1 do
    begin
      V.Raw[K] := (Random - 0.5) * 16.0;
      Ref[K] := V.Raw[K];
    end;

    // Independent per-(x,y), per-group scalar reference.
    for X := 0 to SX - 1 do
      for Y := 0 to SY - 1 do
        for G := 0 to Groups - 1 do
        begin
          Base := V.GetRawPos(X, Y, 0) + G * ChPerGroup;
          MaxV := Ref[Base];
          for K := 1 to ChPerGroup - 1 do
            if Ref[Base + K] > MaxV then MaxV := Ref[Base + K];
          S := 0;
          for K := 0 to ChPerGroup - 1 do
          begin
            Ref[Base + K] := Exp(NeuronForceRange(Ref[Base + K] - MaxV, 4000));
            S := S + Ref[Base + K];
          end;
          if S > 0 then
            for K := 0 to ChPerGroup - 1 do Ref[Base + K] := Ref[Base + K] / S;
        end;

    V.GroupedPointwiseSoftMax(Groups);

    for K := 0 to Total - 1 do
      AssertEquals('GroupedPointwiseSoftMax parity at ' + IntToStr(K),
        Ref[K], V.Raw[K], 1e-4);

    // Every group must sum to 1 - catches a group whose normalization used the
    // wrong element count.
    for X := 0 to SX - 1 do
      for Y := 0 to SY - 1 do
        for G := 0 to Groups - 1 do
        begin
          Base := V.GetRawPos(X, Y, 0) + G * ChPerGroup;
          S := 0;
          for K := 0 to ChPerGroup - 1 do S := S + V.Raw[Base + K];
          AssertEquals('Group sums to one', 1.0, S, 1e-4);
        end;
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumePadding;
var
  Original, Padded: TNNetVolume;
begin
  Original := TNNetVolume.Create(3, 3, 1);
  Padded := TNNetVolume.Create(1, 1, 1);
  try
    Original.Fill(1.0);
    
    Padded.CopyPadding(Original, 1);
    
    // After padding by 1, size should be 5x5
    AssertEquals('Padded SizeX should be 5', 5, Padded.SizeX);
    AssertEquals('Padded SizeY should be 5', 5, Padded.SizeY);
    
    // Center should have original values
    AssertEquals('Center value should be 1.0', 1.0, Padded[1, 1, 0], 0.0001);
    
    // Padding areas should be 0
    AssertEquals('Top-left corner should be 0.0', 0.0, Padded[0, 0, 0], 0.0001);
    AssertEquals('Bottom-right corner should be 0.0', 0.0, Padded[4, 4, 0], 0.0001);
  finally
    Original.Free;
    Padded.Free;
  end;
end;

// CopyPadding zeroes only the border and lets the row copies rewrite the
// interior, so a destination that already holds data must come back with every
// border cell zeroed and every interior cell taken from the source.
procedure TTestNeuralVolume.TestVolumePaddingBorderIsZeroed;
var
  Original, Padded: TNNetVolume;
  X, Y, D, PadX, PadY: integer;
  Expected: TNeuralFloat;
begin
  for PadX := 0 to 2 do
    for PadY := 0 to 2 do
    begin
      Original := TNNetVolume.Create(4, 3, 2);
      // Sized to the padded result up front so ReSize keeps the dirty content.
      Padded := TNNetVolume.Create(4 + PadX * 2, 3 + PadY * 2, 2);
      try
        for X := 0 to Original.SizeX - 1 do
          for Y := 0 to Original.SizeY - 1 do
            for D := 0 to Original.Depth - 1 do
              Original[X, Y, D] := 1 + X + Y * 10 + D * 100;
        Padded.Fill(-7.0);

        if PadX = PadY
          then Padded.CopyPadding(Original, PadX)
          else Padded.CopyPadding(Original, PadX, PadY);

        AssertEquals('Padded SizeX', 4 + PadX * 2, Padded.SizeX);
        AssertEquals('Padded SizeY', 3 + PadY * 2, Padded.SizeY);
        for X := 0 to Padded.SizeX - 1 do
          for Y := 0 to Padded.SizeY - 1 do
            for D := 0 to Padded.Depth - 1 do
            begin
              if (X < PadX) or (Y < PadY) or
                 (X >= PadX + Original.SizeX) or (Y >= PadY + Original.SizeY)
                then Expected := 0
                else Expected := Original[X - PadX, Y - PadY, D];
              AssertEquals('Padded cell', Expected, Padded[X, Y, D], 0.0001);
            end;
      finally
        Original.Free;
        Padded.Free;
      end;
    end;
end;

procedure TTestNeuralVolume.TestVolumeTranspose;
var
  Original, Transposed: TNNetVolume;
begin
  // Test transpose of X and D dimensions
  Original := TNNetVolume.Create(4, 1, 2);
  Transposed := TNNetVolume.Create(1, 1, 1);
  try
    // Set some values
    Original[0, 0, 0] := 1.0;
    Original[1, 0, 0] := 2.0;
    Original[2, 0, 0] := 3.0;
    Original[3, 0, 0] := 4.0;
    Original[0, 0, 1] := 5.0;
    Original[1, 0, 1] := 6.0;
    Original[2, 0, 1] := 7.0;
    Original[3, 0, 1] := 8.0;
    
    Transposed.CopyTransposingXD(Original);
    
    // After transposing X and D, SizeX and Depth should be swapped
    AssertEquals('Transposed SizeX should be 2', 2, Transposed.SizeX);
    AssertEquals('Transposed Depth should be 4', 4, Transposed.Depth);
  finally
    Original.Free;
    Transposed.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeNormalization;
var
  V: TNNetVolume;
  Magnitude: TNeuralFloat;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Raw[0] := 3.0;
    V.Raw[1] := 4.0;
    V.Raw[2] := 0.0;
    V.Raw[3] := 0.0;
    
    Magnitude := V.GetMagnitude();
    // Magnitude of [3, 4, 0, 0] = 5
    AssertEquals('Magnitude should be 5.0', 5.0, Magnitude, 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeMagnitude;
var
  V: TNNetVolume;
  Magnitude: TNeuralFloat;
begin
  V := TNNetVolume.Create(3, 1, 1);
  try
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 2.0;
    
    Magnitude := V.GetMagnitude();
    // Magnitude of [1, 2, 2] = sqrt(1 + 4 + 4) = 3
    AssertEquals('Magnitude should be 3.0', 3.0, Magnitude, 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeEntropy;
var
  V: TNNetVolume;
  Entropy: TNeuralFloat;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    // Uniform distribution
    V.Raw[0] := 0.25;
    V.Raw[1] := 0.25;
    V.Raw[2] := 0.25;
    V.Raw[3] := 0.25;
    
    Entropy := V.GetEntropy();
    // Entropy of uniform distribution over 4 elements = log2(4) = 2
    AssertTrue('Entropy of uniform dist should be around 2', Abs(Entropy - 2.0) < 0.1);
    
    // Deterministic distribution
    V.Raw[0] := 1.0;
    V.Raw[1] := 0.0;
    V.Raw[2] := 0.0;
    V.Raw[3] := 0.0;
    
    Entropy := V.GetEntropy();
    // Entropy of deterministic distribution = 0
    AssertEquals('Entropy of deterministic dist should be 0', 0.0, Entropy, 0.001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeCrossEntropy;
var
  Output, Target: TNNetVolume;
begin
  // Sequence of 2 positions (X axis), vocab size 3 (depth axis).
  Output := TNNetVolume.Create(2, 1, 3);
  Target := TNNetVolume.Create(2, 1, 3);
  try
    Target.Fill(0);
    // Position 0: true class 1, predicted perfectly -> CE = -ln(1) = 0.
    Target[0, 0, 1] := 1.0;
    Output[0, 0, 0] := 0.0; Output[0, 0, 1] := 1.0; Output[0, 0, 2] := 0.0;
    AssertEquals('CE of perfect prediction is 0', 0.0,
      Output.CrossEntropyOnPixel(Target, 0, 0), 0.0001);

    // Position 1: true class 2, predicted prob 0.7 -> CE = -ln(0.7).
    Target[1, 0, 2] := 1.0;
    Output[1, 0, 0] := 0.1; Output[1, 0, 1] := 0.2; Output[1, 0, 2] := 0.7;
    AssertEquals('CE matches -ln(p) of the true class', -Ln(0.7),
      Output.CrossEntropyOnPixel(Target, 1, 0), 0.0001);

    // Mean over the two positions.
    AssertEquals('Mean CE averages over all pixels', (0.0 + (-Ln(0.7))) / 2,
      Output.MeanCrossEntropy(Target), 0.0001);

    // Zero predicted probability on the true class is clamped to 1e-12.
    Output[1, 0, 2] := 0.0;
    AssertEquals('Zero probability is clamped before Ln', -Ln(1e-12),
      Output.CrossEntropyOnPixel(Target, 1, 0), 0.0001);
  finally
    Output.Free;
    Target.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeOneHotEncodingOnPixel;
var
  V: TNNetVolume;
begin
  // 3 positions (X axis), vocab size 4 (depth axis), single row (Y = 0).
  V := TNNetVolume.Create(3, 1, 4);
  try
    V.Fill(9);  // non-zero garbage to prove the column is cleared
    V.OneHotEncodingOnPixel(1, 0, 2);
    // The targeted pixel becomes a clean one-hot of class 2 ...
    AssertEquals('one-hot bit set', 1.0, V[1, 0, 2], 0.0001);
    AssertEquals('other depth 0 cleared', 0.0, V[1, 0, 0], 0.0001);
    AssertEquals('other depth 1 cleared', 0.0, V[1, 0, 1], 0.0001);
    AssertEquals('other depth 3 cleared', 0.0, V[1, 0, 3], 0.0001);
    // ... and GetClassOnPixel is its inverse.
    AssertEquals('GetClassOnPixel inverts it', 2, V.GetClassOnPixel(1, 0));
    // Neighbouring pixels are left untouched (still the garbage fill).
    AssertEquals('neighbour pixel untouched', 9.0, V[0, 0, 0], 0.0001);
    AssertEquals('neighbour pixel untouched', 9.0, V[2, 0, 3], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeOneHotEncoding;
var
  V: TNNetVolume;
  Tokens: array[0..2] of integer;
begin
  // OneHotEncoding: SizeX = number of tokens, Depth = vocab size
  // It sets Self[TokenIndex, 0, TokenValue] := 1
  V := TNNetVolume.Create(3, 1, 10);  // 3 tokens, vocab size 10
  try
    Tokens[0] := 1;
    Tokens[1] := 5;
    Tokens[2] := 8;
    
    V.Fill(0);
    V.OneHotEncoding(Tokens);
    
    // Check that the correct positions are set
    // Token 0 has value 1, so V[0, 0, 1] = 1
    AssertEquals('V[0,0,1] should be 1.0', 1.0, V[0, 0, 1], 0.0001);
    // Token 1 has value 5, so V[1, 0, 5] = 1
    AssertEquals('V[1,0,5] should be 1.0', 1.0, V[1, 0, 5], 0.0001);
    // Token 2 has value 8, so V[2, 0, 8] = 1
    AssertEquals('V[2,0,8] should be 1.0', 1.0, V[2, 0, 8], 0.0001);
    // Other positions should be 0
    AssertEquals('V[0,0,0] should be 0.0', 0.0, V[0, 0, 0], 0.0001);
    AssertEquals('V[1,0,0] should be 0.0', 0.0, V[1, 0, 0], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeGroupedOneHotEncoding;
var
  V: TNNetVolume;
  Tokens: array[0..2] of integer;
  TooManyTokens: array[0..3] of integer;
begin
  // 3 positions, depth 8 split into 2 groups of 4: group 0 holds Token mod 4
  // and group 1 (offset 4) holds Token div 4.
  V := TNNetVolume.Create(3, 1, 8);
  try
    Tokens[0] := 1;
    Tokens[1] := 5;
    Tokens[2] := 8;
    V.GroupedOneHotEncoding(Tokens, 2);
    AssertEquals('token 1 low group', 1.0, V[0, 0, 1], 0.0001);
    AssertEquals('token 1 high group', 1.0, V[0, 0, 4], 0.0001);
    AssertEquals('token 5 low group', 1.0, V[1, 0, 1], 0.0001);
    AssertEquals('token 5 high group', 1.0, V[1, 0, 5], 0.0001);
    AssertEquals('token 8 low group', 1.0, V[2, 0, 0], 0.0001);
    AssertEquals('token 8 high group', 1.0, V[2, 0, 6], 0.0001);
    AssertEquals('two bits per position', 6.0, V.GetSum(), 0.0001);

    // One token more than SizeX must be rejected instead of writing a depth row
    // past the end of the volume.
    TooManyTokens[0] := 1;
    TooManyTokens[1] := 2;
    TooManyTokens[2] := 3;
    TooManyTokens[3] := 4;
    V.GroupedOneHotEncoding(TooManyTokens, 2);
    AssertEquals('oversized token list encodes nothing', 0.0, V.GetSum(), 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeOneHotEncodingReversedString;
const
  csSizeX = 8;
  csPrompt = 'ABCDEFGHIJKL';
var
  V, VRef: TNNetVolume;
  Tail: string;
begin
  // The string overload keeps only the last SizeX characters. Encoding an
  // over-long prompt must match encoding its manually truncated tail.
  V := TNNetVolume.Create(csSizeX, 1, 256);
  VRef := TNNetVolume.Create(csSizeX, 1, 256);
  try
    Tail := Copy(csPrompt, Length(csPrompt) - csSizeX + 1, csSizeX);
    AssertEquals('Truncated tail', 'EFGHIJKL', Tail);

    V.OneHotEncodingReversed(csPrompt);
    AssertEquals('Over-long prompt must not encode to an all zero volume',
      csSizeX, Round(V.GetSumAbs()));

    VRef.OneHotEncodingReversed(Tail);
    AssertEquals('Over-long prompt must encode as its last SizeX characters',
      0, Round(V.SumDiff(VRef)));

    // Exactly SizeX characters.
    V.OneHotEncodingReversed(Tail);
    AssertEquals('Exact length encodes one hit per position',
      csSizeX, Round(V.GetSumAbs()));
    AssertEquals('Last character lands at position 0',
      1, Round(V[0, 0, Ord('L')]));
    AssertEquals('First character lands at position SizeX-1',
      1, Round(V[csSizeX - 1, 0, Ord('E')]));

    // Shorter than SizeX: only the used positions are set.
    V.OneHotEncodingReversed('XY');
    AssertEquals('Short prompt sets one hit per character',
      2, Round(V.GetSumAbs()));
    AssertEquals('Short prompt last character at position 0',
      1, Round(V[0, 0, Ord('Y')]));
    AssertEquals('Short prompt first character at position 1',
      1, Round(V[1, 0, Ord('X')]));

    // Empty prompt clears the volume.
    V.OneHotEncodingReversed('');
    AssertEquals('Empty prompt yields an empty volume', 0, Round(V.GetSumAbs()));
  finally
    VRef.Free;
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumePositionalEncoding;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(8, 1, 16);
  try
    V.Fill(0);
    V.PositionalEncoding();
    
    // Positional encoding should produce values in [-1, 1] range
    AssertTrue('Max value should be <= 1', V.GetMax() <= 1.001);
    AssertTrue('Min value should be >= -1', V.GetMin() >= -1.001);
    // Should have non-zero values
    AssertTrue('Should have non-zero values', V.GetSumAbs() > 0);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeColorConversions;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(1, 1, 3);
  try
    // Test RGB to HSV and back
    // Pure red: RGB(1, 0, 0)
    V[0, 0, 0] := 1.0;
    V[0, 0, 1] := 0.0;
    V[0, 0, 2] := 0.0;
    
    V.RgbToHsv();
    // After conversion, we should have HSV values
    AssertTrue('HSV converted values should be valid', V.GetSum() >= 0);
    
    V.HsvToRgb();
    // After conversion back, should approximately be red
    AssertEquals('R should be approximately 1.0', 1.0, V[0, 0, 0], 0.01);
    AssertEquals('G should be approximately 0.0', 0.0, V[0, 0, 1], 0.01);
    AssertEquals('B should be approximately 0.0', 0.0, V[0, 0, 2], 0.01);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeLabRoundTrip;
// Regression cover for the CIELAB (sRGB<->Lab, D65) helper used by the
// Colorization example. RGB channels are in the 0..255 range here.
var
  V: TNNetVolume;
  R, G, B: array[0..6] of integer;
  I: integer;
  MaxDiff, Diff: TNeuralFloat;
begin
  // A spread of colors including the grays the in-gamut clamp can bite.
  R[0]:=255; G[0]:=0;   B[0]:=0;     // red
  R[1]:=0;   G[1]:=255; B[1]:=0;     // green
  R[2]:=0;   G[2]:=0;   B[2]:=255;   // blue
  R[3]:=128; G[3]:=128; B[3]:=128;   // mid gray
  R[4]:=10;  G[4]:=200; B[4]:=90;    // arbitrary
  R[5]:=240; G[5]:=130; B[5]:=40;    // orange
  R[6]:=17;  G[6]:=17;  B[6]:=17;    // near black

  V := TNNetVolume.Create(1, 1, 3);
  try
    MaxDiff := 0;
    for I := 0 to 6 do
    begin
      V[0, 0, 0] := R[I];
      V[0, 0, 1] := G[I];
      V[0, 0, 2] := B[I];
      V.RgbToLab();
      // L in [0,100], a/b roughly [-128,127]: sanity on L.
      AssertTrue('L within [0,100]', (V[0,0,0] >= -0.5) and (V[0,0,0] <= 100.5));
      V.LabToRgb();
      Diff := Max(Abs(V[0,0,0]-R[I]), Max(Abs(V[0,0,1]-G[I]), Abs(V[0,0,2]-B[I])));
      if Diff > MaxDiff then MaxDiff := Diff;
    end;
    // Round-trip error should be well under 1 of 255 (8-bit ulp territory).
    AssertTrue('RGB->Lab->RGB max|diff| should be tiny: ' + FloatToStr(MaxDiff),
      MaxDiff < 1.0);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeGaussianNoise;
var
  V: TNNetVolume;
  SumBefore, SumAfter: TNeuralFloat;
begin
  V := TNNetVolume.Create(100, 1, 1);
  try
    V.Fill(5.0);
    SumBefore := V.GetSum();
    
    V.AddGaussianNoise(0.1);
    SumAfter := V.GetSum();
    
    // Sum should change after adding noise
    AssertTrue('Values should change after adding noise', Abs(SumAfter - SumBefore) > 0.001);
    // Average should still be around 5.0 with small noise
    AssertTrue('Average should be approximately 5.0', Abs(V.GetAvg() - 5.0) < 1.0);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeCopyResizing;
var
  Original, Resized: TNNetVolume;
begin
  Original := TNNetVolume.Create(4, 4, 1);
  Resized := TNNetVolume.Create(1, 1, 1);
  try
    Original.Fill(1.0);
    
    Resized.CopyResizing(Original, 8, 8);
    
    AssertEquals('Resized SizeX should be 8', 8, Resized.SizeX);
    AssertEquals('Resized SizeY should be 8', 8, Resized.SizeY);
  finally
    Original.Free;
    Resized.Free;
  end;
end;

// CopyResizing is a nearest-neighbour gather; the destination is walked in
// storage order, which must not change which source pixel each output takes.
procedure TTestNeuralVolume.TestVolumeCopyResizingMatchesReference;
var
  Original, Resized: TNNetVolume;
  RatioX, RatioY, InvRatioX, InvRatioY: TNeuralFloat;
  I, CntX, CntY, CntD, OrigPosX, OrigPosY: integer;
  NewSizeX, NewSizeY, Cnt: integer;
begin
  Original := TNNetVolume.Create(7, 5, 3);
  Resized := TNNetVolume.Create(1, 1, 1);
  try
    RandSeed := 987;
    for I := 0 to Original.Size - 1 do
      Original.FData[I] := (Random - 0.5) * 10;
    for Cnt := 0 to 3 do
    begin
      case Cnt of
        0: begin NewSizeX := 13; NewSizeY := 11; end;  // upscale
        1: begin NewSizeX := 3;  NewSizeY := 2;  end;  // downscale
        2: begin NewSizeX := 13; NewSizeY := 2;  end;  // mixed
        else begin NewSizeX := 1; NewSizeY := 1; end;  // degenerate
      end;
      Resized.CopyResizing(Original, NewSizeX, NewSizeY);
      AssertEquals('SizeX', NewSizeX, Resized.SizeX);
      AssertEquals('SizeY', NewSizeY, Resized.SizeY);
      AssertEquals('Depth', Original.Depth, Resized.Depth);
      RatioX := NewSizeX / Original.SizeX;
      RatioY := NewSizeY / Original.SizeY;
      InvRatioX := 1 / RatioX;
      InvRatioY := 1 / RatioY;
      for CntX := 0 to NewSizeX - 1 do
      begin
        OrigPosX := Min(Original.SizeX - 1, Round(CntX * InvRatioX));
        for CntY := 0 to NewSizeY - 1 do
        begin
          OrigPosY := Min(Original.SizeY - 1, Round(CntY * InvRatioY));
          for CntD := 0 to Original.Depth - 1 do
            AssertEquals('Resized element',
              Original[OrigPosX, OrigPosY, CntD], Resized[CntX, CntY, CntD], 0);
        end;
      end;
    end;
  finally
    Original.Free;
    Resized.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeCopyCropping;
var
  Original, Cropped: TNNetVolume;
begin
  Original := TNNetVolume.Create(8, 8, 1);
  Cropped := TNNetVolume.Create(1, 1, 1);
  try
    Original.Fill(1.0);
    // Set a specific value in the center
    Original[3, 3, 0] := 5.0;
    
    Cropped.CopyCropping(Original, 2, 2, 4, 4);
    
    AssertEquals('Cropped SizeX should be 4', 4, Cropped.SizeX);
    AssertEquals('Cropped SizeY should be 4', 4, Cropped.SizeY);
    // The value at (3,3) in original should be at (1,1) in cropped
    AssertEquals('Cropped value at (1,1) should be 5.0', 5.0, Cropped[1, 1, 0], 0.0001);
  finally
    Original.Free;
    Cropped.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSumToPos;
var
  Source, Prefix: TNNetVolume;
begin
  Source := TNNetVolume.Create(5, 1, 1);
  Prefix := TNNetVolume.Create(1, 1, 1);
  try
    Source.Raw[0] := 1.0;
    Source.Raw[1] := 2.0;
    Source.Raw[2] := -3.0;
    Source.Raw[3] := 4.0;
    Source.Raw[4] := 0.5;

    Prefix.SumToPos(Source);

    AssertEquals('Prefix takes the source shape', Source.Size, Prefix.Size);
    AssertEquals('Position 0 repeats the first element', 1.0, Prefix.Raw[0], 0.0001);
    AssertEquals('Position 1', 3.0, Prefix.Raw[1], 0.0001);
    AssertEquals('Position 2', 0.0, Prefix.Raw[2], 0.0001);
    AssertEquals('Position 3', 4.0, Prefix.Raw[3], 0.0001);
    AssertEquals('Last position sums everything', 4.5, Prefix.Raw[4], 0.0001);

    // In place over itself must give the same prefix sums.
    Source.SumToPos(Source);
    AssertEquals('In place position 0', 1.0, Source.Raw[0], 0.0001);
    AssertEquals('In place position 2', 0.0, Source.Raw[2], 0.0001);
    AssertEquals('In place last position', 4.5, Source.Raw[4], 0.0001);
  finally
    Prefix.Free;
    Source.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeSmallestIdxInRange;
var
  Source: TNNetVolume;
begin
  Source := TNNetVolume.Create(6, 1, 1);
  try
    Source.Raw[0] := 5.0;
    Source.Raw[1] := 2.0;
    Source.Raw[2] := 9.0;
    Source.Raw[3] := 2.0;
    Source.Raw[4] := 7.0;
    Source.Raw[5] := 1.0;

    AssertEquals('Single element range returns its own position',
      2, Source.GetSmallestIdxInRange(2, 1));
    AssertEquals('Normal range finds the minimum',
      1, Source.GetSmallestIdxInRange(0, 4));
    AssertEquals('A tie keeps the first position',
      1, Source.GetSmallestIdxInRange(1, 3));
    AssertEquals('The minimum at the start position survives the scan',
      1, Source.GetSmallestIdxInRange(1, 4));
    AssertEquals('The range is clipped to the volume size',
      5, Source.GetSmallestIdxInRange(4, 100));
    AssertEquals('A start position beyond the volume returns zero',
      0, Source.GetSmallestIdxInRange(6, 3));
    AssertEquals('An empty range returns zero',
      0, Source.GetSmallestIdxInRange(2, 0));
  finally
    Source.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeShift;
var
  V: TNNetVolume;
begin
  V := TNNetVolume.Create(4, 1, 1);
  try
    V.Raw[0] := 1.0;
    V.Raw[1] := 2.0;
    V.Raw[2] := 3.0;
    V.Raw[3] := 4.0;
    
    V.ShiftRight(1);
    
    // After shifting right by 1: [0, 1, 2, 3]
    AssertEquals('Position 0 should be 0 after shift', 0.0, V.Raw[0], 0.0001);
    AssertEquals('Position 1 should be 1.0', 1.0, V.Raw[1], 0.0001);
    AssertEquals('Position 2 should be 2.0', 2.0, V.Raw[2], 0.0001);
    AssertEquals('Position 3 should be 3.0', 3.0, V.Raw[3], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeRawPosAndPtr;
var
  V: TNNetVolume;
  Pos: integer;
  Ptr: pointer;
begin
  V := TNNetVolume.Create(4, 4, 3);
  try
    // Test GetRawPos - storage is (x, y, depth) in interleaved format
    // The data is stored as: for each (x,y) position, all depths are consecutive
    Pos := V.GetRawPos(0, 0, 0);
    AssertEquals('RawPos(0,0,0) should be 0', 0, Pos);
    
    // Depth is interleaved, so (0,0,1) is at position 1
    Pos := V.GetRawPos(0, 0, 1);
    AssertEquals('RawPos(0,0,1) should be 1', 1, Pos);
    
    // Position (1,0,0) is at position Depth (which is 3)
    Pos := V.GetRawPos(1, 0, 0);
    AssertEquals('RawPos(1,0,0) should be 3 (depth)', 3, Pos);
    
    // Test GetRawPtr
    Ptr := V.GetRawPtr(0, 0, 0);
    AssertTrue('RawPtr should not be nil', Ptr <> nil);
    
    Ptr := V.GetRawPtr();
    AssertTrue('RawPtr() should not be nil', Ptr <> nil);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumeDepthOperations;
var
  V: TNNetVolume;
  SumD0, AvgD0: TNeuralFloat;
begin
  V := TNNetVolume.Create(2, 2, 3);
  try
    // Fill each depth with different values
    V.FillAtDepth(0, 2.0);
    V.FillAtDepth(1, 4.0);
    V.FillAtDepth(2, 6.0);
    
    // Test SumAtDepth
    SumD0 := V.SumAtDepth(0);
    AssertEquals('Sum at depth 0 should be 8.0 (4 * 2.0)', 8.0, SumD0, 0.0001);
    
    // Test AvgAtDepth
    AvgD0 := V.AvgAtDepth(0);
    AssertEquals('Avg at depth 0 should be 2.0', 2.0, AvgD0, 0.0001);
    
    // Test AddAtDepth
    V.AddAtDepth(0, 1.0);
    AssertEquals('After AddAtDepth, value should be 3.0', 3.0, V[0, 0, 0], 0.0001);
    
    // Test MulAtDepth
    V.MulAtDepth(1, 0.5);
    AssertEquals('After MulAtDepth, value should be 2.0', 2.0, V[0, 0, 1], 0.0001);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestAssertFiniteAllFinite;
var
  V: TNNetVolume;
  I: integer;
begin
  V := TNNetVolume.Create(8, 1, 1);
  try
    for I := 0 to V.Size - 1 do
      V.Raw[I] := I * 0.5;
    try
      AssertFinite(V, 'AllFinite');
    except
      on E: Exception do
        Fail('AssertFinite raised on finite values: ' + E.Message);
    end;
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestAssertFiniteDetectsNaN;
var
  V: TNNetVolume;
  Raised: boolean;
  Msg: string;
begin
  V := TNNetVolume.Create(8, 1, 1);
  try
    V.Fill(1.0);
    V.FData[3] := NaN;
    Raised := False;
    Msg := '';
    try
      AssertFinite(V, 'NaNCheck');
    except
      on E: Exception do
      begin
        Raised := True;
        Msg := E.Message;
      end;
    end;
    AssertTrue('Exception should have been raised for NaN', Raised);
    AssertTrue('Message should contain label NaNCheck: ' + Msg,
      Pos('NaNCheck', Msg) > 0);
    AssertTrue('Message should contain NaN: ' + Msg,
      Pos('NaN', Msg) > 0);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestAssertFiniteDetectsInf;
var
  V: TNNetVolume;
  Raised: boolean;
  Msg: string;
begin
  V := TNNetVolume.Create(8, 1, 1);
  try
    V.Fill(1.0);
    V.FData[5] := Infinity;
    Raised := False;
    Msg := '';
    try
      AssertFinite(V, 'InfCheck');
    except
      on E: Exception do
      begin
        Raised := True;
        Msg := E.Message;
      end;
    end;
    AssertTrue('Exception should have been raised for Inf', Raised);
    AssertTrue('Message should contain label InfCheck: ' + Msg,
      Pos('InfCheck', Msg) > 0);
    AssertTrue('Message should contain Inf: ' + Msg,
      Pos('Inf', Msg) > 0);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolume.TestAssertFiniteNilVolume;
var
  Raised: boolean;
begin
  Raised := False;
  try
    AssertFinite(nil, 'NilCheck');
  except
    on E: Exception do
      Raised := True;
  end;
  AssertTrue('Exception should have been raised for nil volume', Raised);
end;

procedure TTestNeuralVolume.TestNeuralBoxIoU;
var
  IoU: TNeuralFloat;
begin
  // Identical boxes -> IoU 1.
  AssertEquals('Identical boxes IoU', 1.0,
    NeuralBoxIoU(0, 0, 10, 10, 0, 0, 10, 10), 1e-5);
  // Disjoint boxes -> IoU 0.
  AssertEquals('Disjoint boxes IoU', 0.0,
    NeuralBoxIoU(0, 0, 10, 10, 100, 100, 110, 110), 1e-5);
  // Half overlap: A=(0,0,10,10), B=(5,0,15,10). inter=5*10=50,
  // union=100+100-50=150 -> IoU = 1/3.
  IoU := NeuralBoxIoU(0, 0, 10, 10, 5, 0, 15, 10);
  AssertEquals('Half-overlap IoU', 1.0 / 3.0, IoU, 1e-5);
  // Degenerate (zero-area) box -> 0.
  AssertEquals('Degenerate box IoU', 0.0,
    NeuralBoxIoU(5, 5, 5, 5, 0, 0, 10, 10), 1e-5);
end;

procedure TTestNeuralVolume.TestNeuralGreedyNMS;
var
  BX1, BY1, BX2, BY2, Scores: array of TNeuralFloat;
  Classes: TNeuralIntegerArray;
  Kept: TNeuralIntegerArray;
begin
  // Four candidate boxes:
  //   0: (0,0,10,10)       class 0  score 0.90  -> KEPT (highest)
  //   1: (1,1,11,11)       class 0  score 0.80  -> SUPPRESSED by 0 (IoU>0.45)
  //   2: (1,1,11,11)       class 1  score 0.85  -> KEPT (different class)
  //   3: (100,100,110,110) class 0  score 0.70  -> KEPT (no overlap)
  SetLength(BX1, 4); SetLength(BY1, 4); SetLength(BX2, 4); SetLength(BY2, 4);
  SetLength(Scores, 4); SetLength(Classes, 4);
  BX1[0] := 0;   BY1[0] := 0;   BX2[0] := 10;  BY2[0] := 10;  Scores[0] := 0.90; Classes[0] := 0;
  BX1[1] := 1;   BY1[1] := 1;   BX2[1] := 11;  BY2[1] := 11;  Scores[1] := 0.80; Classes[1] := 0;
  BX1[2] := 1;   BY1[2] := 1;   BX2[2] := 11;  BY2[2] := 11;  Scores[2] := 0.85; Classes[2] := 1;
  BX1[3] := 100; BY1[3] := 100; BX2[3] := 110; BY2[3] := 110; Scores[3] := 0.70; Classes[3] := 0;

  Kept := NeuralGreedyNMS(BX1, BY1, BX2, BY2, Scores, Classes, 4, 0.45);

  // Expected kept indices in descending-score order: [0, 2, 3].
  AssertEquals('Kept count', 3, Length(Kept));
  AssertEquals('Kept[0] (score 0.90)', 0, Kept[0]);
  AssertEquals('Kept[1] (score 0.85, class 1)', 2, Kept[1]);
  AssertEquals('Kept[2] (score 0.70, disjoint)', 3, Kept[2]);

  // Empty input -> empty result, no crash.
  Kept := NeuralGreedyNMS(BX1, BY1, BX2, BY2, Scores, Classes, 0, 0.45);
  AssertEquals('Empty NMS count', 0, Length(Kept));
end;

{ TTestNeuralVolumeQuant8 }

// Fills a (2,3,4) volume with a distinct code per element and a distinct
// scale per row, so any layout mistake shows up as a mismatched value.
procedure FillQuant8Sample(V: TNNetVolumeQuant8);
var
  x, y, d: integer;
begin
  V.ReSize(2, 3, 4);
  for y := 0 to 2 do
  begin
    for x := 0 to 1 do
    begin
      V.Scale[x, y] := 0.5 + y + 10 * x;
      for d := 0 to 3 do V.Store(x, y, d, ShortInt(x * 10 + y * 2 + d));
    end;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8EmptyState;
var
  V: TNNetVolumeQuant8;
begin
  V := TNNetVolumeQuant8.Create();
  try
    AssertEquals('Empty size', 0, V.Size);
    AssertEquals('Empty scale count', 0, V.ScaleCount);
    AssertEquals('Empty SizeX', 0, V.SizeX);
    AssertEquals('Empty SizeY', 0, V.SizeY);
    AssertEquals('Empty Depth', 0, V.Depth);
    AssertTrue('Empty DataPtr is nil', V.DataPtr = nil);
    // The scale plane is parked at (1,1,1) rather than emptied: TNNetVolume
    // takes addr(FData[0]) unconditionally, which range-checks on a
    // zero-length array. ScaleCount, not ScaleData.Size, is the live count.
    AssertTrue('Scale plane is never nil', V.ScaleData <> nil);
    AssertEquals('Empty memory size', 0, V.GetMemSize());
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8ResizeGeometry;
var
  V: TNNetVolumeQuant8;
begin
  V := TNNetVolumeQuant8.Create(2, 3, 4);
  try
    AssertEquals('Size', 24, V.Size);
    AssertEquals('SizeX', 2, V.SizeX);
    AssertEquals('SizeY', 3, V.SizeY);
    AssertEquals('Depth', 4, V.Depth);
    AssertEquals('Scale count', 6, V.ScaleCount);
    AssertEquals('Scale plane SizeX', 2, V.ScaleData.SizeX);
    AssertEquals('Scale plane SizeY', 3, V.ScaleData.SizeY);
    AssertEquals('Scale plane Depth', 1, V.ScaleData.Depth);
    AssertTrue('DataPtr is armed', V.DataPtr <> nil);
    AssertEquals('FData length', 24, Length(V.FData));
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8LayoutMatchesVolume;
var
  V: TNNetVolumeQuant8;
  F: TNNetVolume;
  x, y, d: integer;
begin
  V := TNNetVolumeQuant8.Create(2, 3, 4);
  F := TNNetVolume.Create(2, 3, 4);
  try
    // The whole point of the type: identical addressing to TNNetVolume, so
    // migrated call sites keep their byte order.
    for y := 0 to 2 do
      for x := 0 to 1 do
      begin
        AssertEquals('Row base at ' + IntToStr(x) + ',' + IntToStr(y),
          F.GetRawPos(x, y), V.GetRawPos(x, y));
        for d := 0 to 3 do
          AssertEquals('Pos at ' + IntToStr(x) + ',' + IntToStr(y) + ',' +
            IntToStr(d), F.GetRawPos(x, y, d), V.GetRawPos(x, y, d));
      end;
  finally
    F.Free;
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8StoreAndGet;
var
  V: TNNetVolumeQuant8;
  x, y, d: integer;
begin
  V := TNNetVolumeQuant8.Create();
  try
    FillQuant8Sample(V);
    for y := 0 to 2 do
      for x := 0 to 1 do
        for d := 0 to 3 do
        begin
          AssertEquals('Get', x * 10 + y * 2 + d, integer(V.Get(x, y, d)));
          AssertEquals('GetRaw agrees with Get', integer(V.Get(x, y, d)),
            integer(V.GetRaw(V.GetRawPos(x, y, d))));
        end;
    V.SetRaw(0, -128);
    AssertEquals('SetRaw lower bound', -128, integer(V.Get(0, 0, 0)));
    V.Store(1, 2, 3, 127);
    AssertEquals('Store upper bound', 127, integer(V.Get(1, 2, 3)));
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8RawPointers;
var
  V: TNNetVolumeQuant8;
  P: TNeuralInt8ArrPtr;
begin
  V := TNNetVolumeQuant8.Create();
  try
    FillQuant8Sample(V);
    // Row base.
    P := V.GetRawPtr(1, 2);
    AssertEquals('Row base element 0', integer(V.Get(1, 2, 0)), integer(P^[0]));
    AssertEquals('Row base element 3', integer(V.Get(1, 2, 3)), integer(P^[3]));
    // Mid-row, as the convolutional taps index it.
    P := V.GetRawPtr(1, 2, 3);
    AssertEquals('Mid-row element', integer(V.Get(1, 2, 3)), integer(P^[0]));
    // DataPtr is the base of the whole buffer.
    AssertEquals('DataPtr element 0', integer(V.GetRaw(0)),
      integer(V.DataPtr^[0]));
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8ScaleAccess;
var
  V: TNNetVolumeQuant8;
  x, y: integer;
begin
  V := TNNetVolumeQuant8.Create();
  try
    FillQuant8Sample(V);
    for y := 0 to 2 do
      for x := 0 to 1 do
      begin
        AssertEquals('Scale property', 0.5 + y + 10 * x, V.Scale[x, y], 1e-6);
        // ScalePtr delegates to the scale plane and must see the same writes.
        AssertEquals('ScalePtr agrees', V.Scale[x, y],
          V.ScalePtr^[V.SizeX * y + x], 1e-6);
      end;
    V.Scale[0, 1] := -3.25;
    AssertEquals('Scale write', -3.25, V.Scale[0, 1], 1e-6);
    AssertEquals('Scale write via plane', -3.25, V.ScalePtr^[2], 1e-6);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8Dequantize;
var
  V: TNNetVolumeQuant8;
  x, y, d: integer;
begin
  V := TNNetVolumeQuant8.Create();
  try
    FillQuant8Sample(V);
    for y := 0 to 2 do
      for x := 0 to 1 do
        for d := 0 to 3 do
          AssertEquals('Dequantize', V.Get(x, y, d) * V.Scale[x, y],
            V.Dequantize(x, y, d), 1e-5);
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8DequantizeTo;
var
  V: TNNetVolumeQuant8;
  F: TNNetVolume;
  x, y, d: integer;
begin
  V := TNNetVolumeQuant8.Create();
  F := TNNetVolume.Create(1, 1, 1);
  try
    FillQuant8Sample(V);
    V.DequantizeTo(F);
    AssertEquals('Dest SizeX', 2, F.SizeX);
    AssertEquals('Dest SizeY', 3, F.SizeY);
    AssertEquals('Dest Depth', 4, F.Depth);
    for y := 0 to 2 do
      for x := 0 to 1 do
        for d := 0 to 3 do
          AssertEquals('Dequantized element',
            V.Get(x, y, d) * V.Scale[x, y], F[x, y, d], 1e-5);
    // A row expansion writes exactly the row it was asked for.
    F.Fill(0);
    V.DequantizeRowTo(1, 2, TNeuralFloatArrPtr(F.GetRawPtr(1, 2, 0)));
    for d := 0 to 3 do
      AssertEquals('Row expansion', V.Get(1, 2, d) * V.Scale[1, 2],
        F[1, 2, d], 1e-5);
    AssertEquals('Untouched neighbour', 0.0, F[0, 0, 0], 1e-5);
  finally
    F.Free;
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8CopyFrom;
var
  V, C: TNNetVolumeQuant8;
  x, y, d: integer;
begin
  V := TNNetVolumeQuant8.Create();
  C := TNNetVolumeQuant8.Create(1, 1, 1);
  try
    FillQuant8Sample(V);
    C.CopyFrom(V);
    AssertEquals('Copied size', V.Size, C.Size);
    AssertEquals('Copied scale count', V.ScaleCount, C.ScaleCount);
    for y := 0 to 2 do
      for x := 0 to 1 do
      begin
        AssertEquals('Copied scale', V.Scale[x, y], C.Scale[x, y], 1e-6);
        for d := 0 to 3 do
          AssertEquals('Copied code', integer(V.Get(x, y, d)),
            integer(C.Get(x, y, d)));
      end;
    // Deep copy: writing the copy must not reach the original.
    C.Store(0, 0, 0, 99);
    C.Scale[0, 0] := 42;
    AssertEquals('Original code untouched', 0, integer(V.Get(0, 0, 0)));
    AssertEquals('Original scale untouched', 0.5, V.Scale[0, 0], 1e-6);
  finally
    C.Free;
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8DeleteRows;
var
  V, C: TNNetVolumeQuant8;
  x, d: integer;
begin
  V := TNNetVolumeQuant8.Create();
  C := TNNetVolumeQuant8.Create();
  try
    FillQuant8Sample(V);
    C.CopyFrom(V);
    // Rolling-window eviction: drop row 0, rows 1 and 2 shift down.
    C.DeleteRows(0, 1);
    AssertEquals('Capacity unchanged', 24, C.Size);
    AssertEquals('SizeY unchanged', 3, C.SizeY);
    for x := 0 to 1 do
    begin
      AssertEquals('Scale row 0', V.Scale[x, 1], C.Scale[x, 0], 1e-6);
      AssertEquals('Scale row 1', V.Scale[x, 2], C.Scale[x, 1], 1e-6);
      for d := 0 to 3 do
      begin
        AssertEquals('Code row 0', integer(V.Get(x, 1, d)),
          integer(C.Get(x, 0, d)));
        AssertEquals('Code row 1', integer(V.Get(x, 2, d)),
          integer(C.Get(x, 1, d)));
      end;
    end;
    // Default Count is 1, and two rows can go at once.
    C.CopyFrom(V);
    C.DeleteRows(0);
    for x := 0 to 1 do
      AssertEquals('Default count drops one row', V.Scale[x, 1],
        C.Scale[x, 0], 1e-6);
    C.CopyFrom(V);
    C.DeleteRows(0, 2);
    for x := 0 to 1 do
    begin
      AssertEquals('Two rows dropped', V.Scale[x, 2], C.Scale[x, 0], 1e-6);
      for d := 0 to 3 do
        AssertEquals('Two rows dropped, codes', integer(V.Get(x, 2, d)),
          integer(C.Get(x, 0, d)));
    end;
  finally
    C.Free;
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8DeleteRowsGuards;
var
  V, C: TNNetVolumeQuant8;
  x, y, d: integer;

  procedure AssertUnchanged(const Msg: string);
  var
    xi, yi, di: integer;
  begin
    for yi := 0 to 2 do
      for xi := 0 to 1 do
      begin
        AssertEquals(Msg + ' scale', V.Scale[xi, yi], C.Scale[xi, yi], 1e-6);
        for di := 0 to 3 do
          AssertEquals(Msg + ' code', integer(V.Get(xi, yi, di)),
            integer(C.Get(xi, yi, di)));
      end;
  end;

begin
  V := TNNetVolumeQuant8.Create();
  C := TNNetVolumeQuant8.Create();
  try
    FillQuant8Sample(V);
    C.CopyFrom(V);
    C.DeleteRows(0, 0);        // nothing to drop
    AssertUnchanged('Zero count');
    C.DeleteRows(-1, 1);       // negative start
    AssertUnchanged('Negative start');
    C.DeleteRows(2, 5);        // runs past the end
    AssertUnchanged('Overrun');
    C.DeleteRows(2, 1);        // last row: nothing above it to shift
    AssertUnchanged('Last row');
  finally
    C.Free;
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8GetQuantData;
var
  V: TNNetVolumeQuant8;
  Codes: TInt8DynArr;
  Scales: TNeuralFloatDynArr;
  i: integer;
begin
  V := TNNetVolumeQuant8.Create();
  try
    FillQuant8Sample(V);
    V.GetQuantData(Codes, Scales);
    AssertEquals('Exported code count', 24, Length(Codes));
    AssertEquals('Exported scale count', 6, Length(Scales));
    for i := 0 to 23 do
      AssertEquals('Exported code ' + IntToStr(i), integer(V.GetRaw(i)),
        integer(Codes[i]));
    for i := 0 to 5 do
      AssertEquals('Exported scale ' + IntToStr(i), V.ScalePtr^[i],
        Scales[i], 1e-6);
    // An empty volume exports empty arrays instead of failing.
    V.ReSize(0, 0, 0);
    V.GetQuantData(Codes, Scales);
    AssertEquals('Empty export codes', 0, Length(Codes));
    AssertEquals('Empty export scales', 0, Length(Scales));
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8MemSize;
var
  V: TNNetVolumeQuant8;
begin
  V := TNNetVolumeQuant8.Create(2, 3, 4);
  try
    // 24 codes at one byte plus 6 scales at four bytes.
    AssertEquals('Memory size', 24 + 6 * 4, V.GetMemSize());
    // A weight-shaped volume: one scale per neuron row.
    V.ReSize(1, 4, 6);
    AssertEquals('Weight-shaped memory size', 24 + 4 * 4, V.GetMemSize());
  finally
    V.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8FillAndReshapeCycles;
var
  V: TNNetVolumeQuant8;
begin
  V := TNNetVolumeQuant8.Create();
  try
    FillQuant8Sample(V);
    V.Fill(-7);
    AssertEquals('Filled negative', -7, integer(V.Get(1, 1, 1)));
    AssertEquals('Filled negative, first', -7, integer(V.GetRaw(0)));
    V.Fill(0);
    AssertEquals('Filled zero', 0, integer(V.Get(1, 1, 1)));
    // Shrink to empty and grow again: ReSize alone keeps every field in step.
    V.ReSize(0, 0, 0);
    AssertEquals('Shrunk size', 0, V.Size);
    AssertEquals('Shrunk scale count', 0, V.ScaleCount);
    AssertTrue('Shrunk DataPtr is nil', V.DataPtr = nil);
    V.ReSize(5, 1, 3);
    AssertEquals('Regrown size', 15, V.Size);
    AssertEquals('Regrown scale count', 5, V.ScaleCount);
    AssertTrue('Regrown DataPtr is armed', V.DataPtr <> nil);
    AssertEquals('Regrown FData length', 15, Length(V.FData));
    // Weight shape: X = neuron, Y = 1, Depth = vector size. Row r starts at
    // r*Depth, which is how the concatenated weights are laid out today.
    V.ReSize(4, 1, 6);
    AssertEquals('Weight shape size', 24, V.Size);
    AssertEquals('Weight shape scale count', 4, V.ScaleCount);
    AssertEquals('Weight row base', 3 * 6, V.GetRawPos(3, 0));
    AssertEquals('Weight element', 3 * 6 + 5, V.GetRawPos(3, 0, 5));
    // Same element count, different geometry: the buffer is kept, the scale
    // plane follows the new (x,y) count.
    V.ReSize(2, 4, 3);
    AssertEquals('Reshaped size', 24, V.Size);
    AssertEquals('Reshaped scale count', 8, V.ScaleCount);
    AssertEquals('Reshaped scale plane', 8, V.ScaleData.Size);
  finally
    V.Free;
  end;
end;

// Fills a (NumAs, 1, VectorSize) weight table - the shape the concatenated
// layer weights use - with a distinct code per element and a distinct scale
// per row, then hands back the loose-array view of the same data.
procedure FillQuant8Table(V: TNNetVolumeQuant8; NumAs, VectorSize: integer;
  out pCodes: TInt8DynArr; out pScales: TNeuralFloatDynArr);
var
  a, e: integer;
begin
  V.ReSize(NumAs, 1, VectorSize);
  for a := 0 to NumAs - 1 do
  begin
    V.Scale[a, 0] := 0.125 * (a + 1);
    for e := 0 to VectorSize - 1 do
      V.Store(a, 0, e, ShortInt(((a * 7 + e * 3) mod 61) - 30));
  end;
  V.GetQuantData(pCodes, pScales);
end;

// Fills VBs with distinct, sign-varying inputs.
procedure FillQuant8Inputs(VBs: TNNetVolume);
var
  i: integer;
begin
  for i := 0 to VBs.Size - 1 do
    VBs.FData[i] := ((i mod 9) - 4) * 0.25;
end;

// The TNNetVolumeQuant8 overloads forward to the open-array kernels, so they
// must agree element for element - not merely within a tolerance.
procedure TTestNeuralVolumeQuant8.TestQuant8TiledDotProductMatchesArrays;
const
  NumAs = 5;
  NumBs = 4;
  VectorSize = 6;
var
  Q: TNNetVolumeQuant8;
  Codes: TInt8DynArr;
  Scales: TNeuralFloatDynArr;
  VBs, OutArr, OutVol: TNNetVolume;
  i: integer;
begin
  Q := TNNetVolumeQuant8.Create();
  VBs := TNNetVolume.Create(NumBs, 1, VectorSize);
  OutArr := TNNetVolume.Create(NumAs * NumBs, 1, 1);
  OutVol := TNNetVolume.Create(NumAs * NumBs, 1, 1);
  try
    FillQuant8Table(Q, NumAs, VectorSize, Codes, Scales);
    FillQuant8Inputs(VBs);
    // Full range. Tile sizes deliberately do not divide the ranges.
    OutArr.Fill(0);
    OutVol.Fill(0);
    OutArr.DotProductsTiledInt8(NumAs, NumBs, VectorSize, Codes, Scales, VBs,
      3, 3);
    OutVol.DotProductsTiledInt8(NumAs, NumBs, VectorSize, Q, VBs, 3, 3);
    for i := 0 to NumAs * NumBs - 1 do
      AssertEquals('Full range element ' + IntToStr(i),
        OutArr.FData[i], OutVol.FData[i]);
    // Guards against a vacuous comparison of two all-zero outputs.
    AssertTrue('Full range produced non-zero output',
      OutVol.GetSumAbs() > 0);
    // Ranged twin: a neuron slice crossed with a position slice.
    OutArr.Fill(0);
    OutVol.Fill(0);
    OutArr.DotProductsTiledInt8(NumAs, {BStart}1, {BFinish}2, VectorSize,
      Codes, Scales, VBs, 2, 2, {AStart}1, {AFinish}3);
    OutVol.DotProductsTiledInt8(NumAs, {BStart}1, {BFinish}2, VectorSize,
      Q, VBs, 2, 2, {AStart}1, {AFinish}3);
    for i := 0 to NumAs * NumBs - 1 do
      AssertEquals('Ranged element ' + IntToStr(i),
        OutArr.FData[i], OutVol.FData[i]);
    // The slice really did write only its own rows.
    AssertEquals('Outside the slice stays zero', 0, OutVol.FData[0]);
  finally
    OutVol.Free;
    OutArr.Free;
    VBs.Free;
    Q.Free;
  end;
end;

procedure TTestNeuralVolumeQuant8.TestQuant8GroupedTiledDotProductMatchesArrays;
const
  Groups = 2;
  NumAs = 4;
  NumBs = 3;
  VectorSize = 3;
var
  Q: TNNetVolumeQuant8;
  Codes: TInt8DynArr;
  Scales: TNeuralFloatDynArr;
  VBs: TNNetVolume;
  OutArr, OutVol: TNNetGroupedVolume;
  i: integer;
begin
  Q := TNNetVolumeQuant8.Create();
  // Grouped inputs hold VectorSize*Groups per position.
  VBs := TNNetVolume.Create(NumBs, 1, VectorSize * Groups);
  OutArr := TNNetGroupedVolume.Create(NumAs * NumBs, 1, 1);
  OutVol := TNNetGroupedVolume.Create(NumAs * NumBs, 1, 1);
  try
    FillQuant8Table(Q, NumAs, VectorSize, Codes, Scales);
    FillQuant8Inputs(VBs);
    OutArr.Fill(0);
    OutVol.Fill(0);
    OutArr.GroupedDotProductsTiledInt8(Groups, NumAs, NumBs, VectorSize,
      Codes, Scales, VBs, 3, 2);
    OutVol.GroupedDotProductsTiledInt8(Groups, NumAs, NumBs, VectorSize,
      Q, VBs, 3, 2);
    for i := 0 to NumAs * NumBs - 1 do
      AssertEquals('Grouped element ' + IntToStr(i),
        OutArr.FData[i], OutVol.FData[i]);
    AssertTrue('Grouped produced non-zero output', OutVol.GetSumAbs() > 0);
  finally
    OutVol.Free;
    OutArr.Free;
    VBs.Free;
    Q.Free;
  end;
end;

// MaxAbsFinite is the pointer-and-count max|slice| the int8 quantizers
// scan with. It must return the largest FINITE magnitude: NaN is skipped and
// +/-Inf excluded, so a garbage row still yields a usable range. Sizes cross
// the AVX gate (csMinAvxSize = 16) and the 8-lane block, so the vector body,
// its fold and the scalar tail all get exercised.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestMaxAbsFinite;
const
  N = 37;                    // > 16 (AVX path) and not a multiple of 8 (tail)
var
  V: TNNetVolume;
  i: integer;
  P: TNeuralFloatArrPtr;
begin
  V := TNNetVolume.Create(N, 1, 1);
  try
    P := TNeuralFloatArrPtr(@V.FData[0]);
    AssertEquals('all-zero slice has no range', 0,
      TNNetVolume.MaxAbsFinite(P, N), 0);
    // A negative element carries the max: the sign must be cleared, not
    // compared away.
    for i := 0 to N - 1 do V.FData[i] := 0.5;
    V.FData[20] := -3.25;
    AssertEquals('negative element carries the max', 3.25,
      TNNetVolume.MaxAbsFinite(P, N), 0);
    // In the tail (index >= 32), so the scalar remainder must see it too.
    V.FData[35] := -9.5;
    AssertEquals('tail element carries the max', 9.5,
      TNNetVolume.MaxAbsFinite(P, N), 0);
    // Non-finite values are excluded, not propagated.
    V.FData[3] := math.NaN;
    V.FData[9] := math.Infinity;
    V.FData[34] := math.NegInfinity;
    AssertEquals('NaN and +/-Inf excluded from the max', 9.5,
      TNNetVolume.MaxAbsFinite(P, N), 0);
    // Nothing finite non-zero at all -> 0, the zero-row signal.
    for i := 0 to N - 1 do V.FData[i] := math.NaN;
    AssertEquals('all-NaN slice reports no range', 0,
      TNNetVolume.MaxAbsFinite(P, N), 0);
    // Below the AVX gate the scalar path must agree.
    for i := 0 to N - 1 do V.FData[i] := 0;
    V.FData[5] := -2;
    AssertEquals('short slice takes the scalar path', 2,
      TNNetVolume.MaxAbsFinite(P, 8), 0);
  finally
    V.Free;
  end;
end;

// AVXMaxAbsFinite folds 32 floats per iteration into four accumulators, then 8
// at a time, then a scalar tail. The lengths below take each part alone and in
// combination, and the max is planted at every position in turn, so a lane the
// fold drops or an accumulator it forgets shows up wherever it hides. The slot
// past the run holds a larger value that must not be read. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestMaxAbsFiniteLengthSweep;
const
  cLengths: array[0..12] of integer =
    (16, 24, 31, 32, 33, 39, 40, 47, 64, 65, 128, 1000, 1024);
  N = 1024;
var
  Buf: array of TNeuralFloat;
  i, L, Len, Plant: integer;
begin
  SetLength(Buf, N + 1);
  for L := 0 to High(cLengths) do
  begin
    Len := cLengths[L];
    for Plant := 0 to Len - 1 do
    begin
      for i := 0 to Len - 1 do Buf[i] := 0.25;
      Buf[Plant] := -7.5;
      Buf[Len] := 1000;   // past the run: reading it would win the max
      AssertEquals('max planted at ' + IntToStr(Plant) + ' of N=' +
        IntToStr(Len), 7.5,
        TNNetVolume.MaxAbsFinite(TNeuralFloatArrPtr(@Buf[0]), Len), 0);
    end;
  end;
end;

// QuantizeInt8 against a known row max: the row max itself must land on
// +/-127, zero on 0, and every code must dequantize back within half a step.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestQuantizeInt8;
const
  N = 37;
var
  V: TNNetVolume;
  Codes: TInt8DynArr;
  MaxAbs, Scale, Deq: TNeuralFloat;
  i: integer;
begin
  V := TNNetVolume.Create(N, 1, 1);
  SetLength(Codes, N);
  try
    for i := 0 to N - 1 do V.FData[i] := Sin(0.7 * i) * 2.5;
    V.FData[11] := 2.5;     // the positive max
    V.FData[29] := -2.5;    // the negative max, in the AVX body
    V.FData[36] := 0;       // a zero in the tail
    MaxAbs := TNNetVolume.MaxAbsFinite(
      TNeuralFloatArrPtr(@V.FData[0]), N);
    AssertEquals('row max', 2.5, MaxAbs, 1e-6);
    Scale := MaxAbs / 127;
    TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@Codes[0]),
      TNeuralFloatArrPtr(@V.FData[0]), N, MaxAbs);
    AssertEquals('positive row max codes as +127', 127, Codes[11]);
    AssertEquals('negative row max codes as -127', -127, Codes[29]);
    AssertEquals('zero codes as 0', 0, Codes[36]);
    for i := 0 to N - 1 do
    begin
      Deq := Codes[i] * Scale;
      AssertTrue('code ' + IntToStr(i) + ' dequantizes within half a step: ' +
        FloatToStr(Deq) + ' vs ' + FloatToStr(V.FData[i]),
        Abs(Deq - V.FData[i]) <= Scale * 0.5 + 1e-9);
    end;
    // MaxAbs <= 0 is the caller's zero-row branch: nothing may be written.
    for i := 0 to N - 1 do Codes[i] := 99;
    TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@Codes[0]),
      TNeuralFloatArrPtr(@V.FData[0]), N, 0);
    for i := 0 to N - 1 do
      AssertEquals('zero max writes nothing ' + IntToStr(i), 99, Codes[i]);
  finally
    V.Free;
  end;
end;

// The non-finite convention the checkpoint importers rely on: NaN codes as 0
// and +/-Inf clamp to +/-127 (the finite row max), in both the vector body and
// the scalar tail, without raising EInvalidOp under FPC's unmasked SSE
// exceptions. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestQuantizeInt8NonFinite;
const
  N = 20;
var
  V: TNNetVolume;
  Codes: TInt8DynArr;
  MaxAbs: TNeuralFloat;
  i: integer;
begin
  V := TNNetVolume.Create(N, 1, 1);
  SetLength(Codes, N);
  try
    for i := 0 to N - 1 do V.FData[i] := 1;
    V.FData[0] := 4;                   // the finite max
    V.FData[2] := math.NaN;            // vector body
    V.FData[3] := math.Infinity;
    V.FData[4] := math.NegInfinity;
    V.FData[17] := math.NaN;           // scalar tail
    V.FData[18] := math.Infinity;
    V.FData[19] := math.NegInfinity;
    MaxAbs := TNNetVolume.MaxAbsFinite(
      TNeuralFloatArrPtr(@V.FData[0]), N);
    AssertEquals('finite max survives the non-finite lanes', 4, MaxAbs, 0);
    TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@Codes[0]),
      TNeuralFloatArrPtr(@V.FData[0]), N, MaxAbs);
    AssertEquals('body NaN codes as 0', 0, Codes[2]);
    AssertEquals('body +Inf clamps to +127', 127, Codes[3]);
    AssertEquals('body -Inf clamps to -127', -127, Codes[4]);
    AssertEquals('tail NaN codes as 0', 0, Codes[17]);
    AssertEquals('tail +Inf clamps to +127', 127, Codes[18]);
    AssertEquals('tail -Inf clamps to -127', -127, Codes[19]);
    AssertEquals('finite max codes as +127', 127, Codes[0]);
  finally
    V.Free;
  end;
end;

// Tiny-magnitude rows: the single-precision path forms 1/MaxAbs (never
// 127/MaxAbs, which would overflow), so the real Qwen2.5-7B vocab padding
// value - exactly the smallest NORMAL single - quantizes on the fast path.
// A DENORMAL row max has no finite single reciprocal at all and must route to
// the double-precision scalar path instead of trapping. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestQuantizeInt8TinyAndDenormalRows;
const
  N = 24;
  TinyV: single = 1.1754943508222875e-37;  // the actual Qwen2.5-7B pad value
  DenormV: single = 3.0e-39;               // denormal: 1/x overflows single
var
  V: TNNetVolume;
  Codes: TInt8DynArr;
  MaxAbs: TNeuralFloat;
  i: integer;
begin
  V := TNNetVolume.Create(N, 1, 1);
  SetLength(Codes, N);
  try
    // (a) the smallest normal single, alternating signs as the real pad rows do
    for i := 0 to N - 1 do
      if (i mod 2) = 0 then V.FData[i] := TinyV else V.FData[i] := -TinyV;
    MaxAbs := TNNetVolume.MaxAbsFinite(
      TNeuralFloatArrPtr(@V.FData[0]), N);
    AssertEquals('tiny row max', TinyV, MaxAbs, 0);
    TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@Codes[0]),
      TNeuralFloatArrPtr(@V.FData[0]), N, MaxAbs);
    for i := 0 to N - 1 do
      if (i mod 2) = 0 then
        AssertEquals('tiny +value saturates ' + IntToStr(i), 127, Codes[i])
      else
        AssertEquals('tiny -value saturates ' + IntToStr(i), -127, Codes[i]);
    // (b) a denormal row max: must not trap, and must still quantize its own
    // max onto +/-127 through the double-precision path.
    for i := 0 to N - 1 do
      if (i mod 2) = 0 then V.FData[i] := DenormV else V.FData[i] := -DenormV;
    MaxAbs := TNNetVolume.MaxAbsFinite(
      TNeuralFloatArrPtr(@V.FData[0]), N);
    AssertEquals('denormal row max', DenormV, MaxAbs, 0);
    TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@Codes[0]),
      TNeuralFloatArrPtr(@V.FData[0]), N, MaxAbs);
    for i := 0 to N - 1 do
      if (i mod 2) = 0 then
        AssertEquals('denormal +value saturates ' + IntToStr(i), 127, Codes[i])
      else
        AssertEquals('denormal -value saturates ' + IntToStr(i), -127, Codes[i]);
  finally
    V.Free;
  end;
end;

// The vectorized path scales in SINGLE precision, so it is deliberately NOT
// bit-exact against a double-precision scalar reference - quantization is lossy
// and one code of disagreement on a rounding boundary is acceptable. This pins
// the tolerance at exactly that: every code within 1 of the reference, which is
// tight enough to catch a genuine lane/order/rounding bug.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestQuantizeInt8MatchesScalarReference;
const
  N = 1024;
var
  V: TNNetVolume;
  Codes: TInt8DynArr;
  MaxAbs, Val: TNeuralFloat;
  InvScale: double;
  i, Ref, Diff, Differing: integer;
begin
  RandSeed := 191919;
  V := TNNetVolume.Create(N, 1, 1);
  SetLength(Codes, N);
  try
    for i := 0 to N - 1 do V.FData[i] := (Random - 0.5) * 7;
    MaxAbs := TNNetVolume.MaxAbsFinite(
      TNeuralFloatArrPtr(@V.FData[0]), N);
    AssertTrue('random row has a range', MaxAbs > 0);
    TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@Codes[0]),
      TNeuralFloatArrPtr(@V.FData[0]), N, MaxAbs);
    InvScale := 127.0 / Double(MaxAbs);
    Differing := 0;
    for i := 0 to N - 1 do
    begin
      Val := V.FData[i];
      Ref := Round(Val * InvScale);
      if Ref > 127 then Ref := 127;
      if Ref < -127 then Ref := -127;
      Diff := Abs(Codes[i] - Ref);
      AssertTrue('code ' + IntToStr(i) + ' within 1 of the double reference: ' +
        IntToStr(Codes[i]) + ' vs ' + IntToStr(Ref), Diff <= 1);
      if Diff <> 0 then Inc(Differing);
    end;
    // Sanity: the vector path is not silently falling back to the reference
    // arithmetic for every element, nor wrong for most of them.
    AssertTrue('disagreement stays rare: ' + IntToStr(Differing) + '/' +
      IntToStr(N), Differing * 10 <= N);
  finally
    V.Free;
  end;
end;

// DequantizeInt8 is one single multiply per element on both paths, so unlike
// QuantizeInt8 it IS bit-exact and the reference below uses delta 0. N = 21
// crosses the AVX body (16) into the scalar tail (5); N = 5 stays under
// csMinAvxSize and exercises the pure scalar method. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDequantizeInt8;
const
  N = 21;
  // TYPED on purpose: an untyped const would keep full precision in the
  // expectation below while the call rounds it to single, so the two would
  // disagree in the last bit for no good reason.
  Scale: TNeuralFloat = 0.0125;
var
  Codes: TInt8DynArr;
  Dst: array of TNeuralFloat;
  // The expectation must round to SINGLE exactly once, like the kernel does.
  // Passing "Scale * Codes[i]" straight to AssertEquals would evaluate it in
  // double and disagree in the last bit.
  Expected: TNeuralFloat;
  i: integer;
begin
  SetLength(Codes, N);
  SetLength(Dst, N);
  // A spread that includes both saturation ends and zero.
  for i := 0 to N - 1 do Codes[i] := ShortInt(((i * 13) mod 255) - 127);
  Codes[0] := 127;
  Codes[1] := -127;
  Codes[2] := 0;
  for i := 0 to N - 1 do Dst[i] := 12345;   // poison: every slot must be written
  TNNetVolume.DequantizeInt8(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralInt8ArrPtr(@Codes[0]), N, Scale);
  for i := 0 to N - 1 do
  begin
    Expected := Scale * Codes[i];
    AssertEquals('element ' + IntToStr(i) + ' (code ' + IntToStr(Codes[i]) + ')',
      Expected, Dst[i], 0);
  end;
  // Under csMinAvxSize: the scalar method, same contract.
  for i := 0 to 4 do Dst[i] := 12345;
  TNNetVolume.DequantizeInt8(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralInt8ArrPtr(@Codes[0]), 5, Scale);
  for i := 0 to 4 do
  begin
    Expected := Scale * Codes[i];
    AssertEquals('short-run element ' + IntToStr(i), Expected, Dst[i], 0);
  end;
  // N <= 0 must write nothing at all.
  Dst[0] := 999;
  TNNetVolume.DequantizeInt8(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralInt8ArrPtr(@Codes[0]), 0, Scale);
  AssertEquals('N=0 writes nothing', 999, Dst[0], 0);
end;

// The vectorized run and the scalar run must produce the same codes: the same
// values are quantized once in bulk (the AVX2 path) and once one element at a
// time (always scalar, being under csMinAvxSize). The lengths take the
// 32-element unrolled body, the 8-element remainder loop and the scalar tail
// alone and in combination. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestQuantizeInt8LengthSweep;
const
  cLengths: array[0..12] of integer =
    (16, 24, 31, 32, 33, 39, 40, 47, 64, 65, 128, 1000, 1024);
  N = 1024;
var
  V: TNNetVolume;
  Bulk, OneByOne: TInt8DynArr;
  MaxAbs: TNeuralFloat;
  i, L, Len: integer;
begin
  RandSeed := 313131;
  V := TNNetVolume.Create(N, 1, 1);
  SetLength(Bulk, N + 1);      // one guard slot past the longest run
  SetLength(OneByOne, N);
  try
    for i := 0 to N - 1 do V.FData[i] := (Random - 0.5) * 7;
    MaxAbs := TNNetVolume.MaxAbsFinite(TNeuralFloatArrPtr(@V.FData[0]), N);
    AssertTrue('random row has a range', MaxAbs > 0);
    for i := 0 to N - 1 do
      TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@OneByOne[i]),
        TNeuralFloatArrPtr(@V.FData[i]), 1, MaxAbs);
    for L := 0 to High(cLengths) do
    begin
      Len := cLengths[L];
      for i := 0 to N do Bulk[i] := 99;
      TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@Bulk[0]),
        TNeuralFloatArrPtr(@V.FData[0]), Len, MaxAbs);
      for i := 0 to Len - 1 do
        AssertEquals('code ' + IntToStr(i) + ' at N=' + IntToStr(Len),
          OneByOne[i], Bulk[i]);
      AssertEquals('wrote past N=' + IntToStr(Len), 99, Bulk[Len]);
    end;
  finally
    V.Free;
  end;
end;

// The DequantizeInt8 twin of TestQuantizeInt8LengthSweep. One single-precision
// multiply per element on both paths, so the reference below is exact and the
// delta is 0. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDequantizeInt8LengthSweep;
const
  cLengths: array[0..12] of integer =
    (16, 24, 31, 32, 33, 39, 40, 47, 64, 65, 128, 1000, 1024);
  N = 1024;
  // TYPED on purpose: an untyped const would keep full precision in the
  // expectation below while the call rounds it to single.
  Scale: TNeuralFloat = 0.0125;
var
  Codes: TInt8DynArr;
  Dst: array of TNeuralFloat;
  Expected: TNeuralFloat;
  i, L, Len: integer;
begin
  SetLength(Codes, N);
  SetLength(Dst, N + 1);       // one guard slot past the longest run
  for i := 0 to N - 1 do Codes[i] := ShortInt(((i * 13) mod 255) - 127);
  for L := 0 to High(cLengths) do
  begin
    Len := cLengths[L];
    for i := 0 to N do Dst[i] := 12345;
    TNNetVolume.DequantizeInt8(TNeuralFloatArrPtr(@Dst[0]),
      TNeuralInt8ArrPtr(@Codes[0]), Len, Scale);
    for i := 0 to Len - 1 do
    begin
      Expected := Scale * Codes[i];
      AssertEquals('element ' + IntToStr(i) + ' at N=' + IntToStr(Len),
        Expected, Dst[i], 0);
    end;
    AssertEquals('wrote past N=' + IntToStr(Len), 12345.0, Dst[Len], 0);
  end;
end;

// Exact int32 oracle over lengths around the AVX2 32-element block and its
// tail, with the extreme codes +-127 placed so consecutive pairs hit the
// vpmaddubsw worst case (2 * 127 * 127). Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDotProductInt8Int8LengthSweep;
const
  cLengths: array[0..15] of integer =
    (0, 1, 3, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100, 1000, 1024);
  N = 1024;
var
  A, B: TInt8DynArr;
  Expected, Got: integer;
  i, L, Len: integer;
begin
  SetLength(A, N);
  SetLength(B, N);
  for i := 0 to N - 1 do
  begin
    A[i] := ShortInt(((i * 37) mod 255) - 127);
    B[i] := ShortInt(((i * 91 + 5) mod 255) - 127);
  end;
  // first 8 elements: all four sign combinations of the +-127 extremes
  A[0] := 127;  B[0] := 127;  A[1] := 127;  B[1] := 127;
  A[2] := -127; B[2] := -127; A[3] := -127; B[3] := -127;
  A[4] := 127;  B[4] := -127; A[5] := 127;  B[5] := -127;
  A[6] := -127; B[6] := 127;  A[7] := -127; B[7] := 127;
  for L := 0 to High(cLengths) do
  begin
    Len := cLengths[L];
    Expected := 0;
    for i := 0 to Len - 1 do Expected := Expected + A[i] * B[i];
    Got := TNNetVolume.DotProductInt8Int8(TNeuralInt8ArrPtr(@A[0]),
      TNeuralInt8ArrPtr(@B[0]), Len);
    AssertEquals('N=' + IntToStr(Len), Expected, Got);
  end;
end;

// The int8 x int8 sum times both scales must agree with the existing
// int8-weight x FP32-input path fed the dequantized B. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDotProductInt8Int8MatchesFloatPath;
const
  N = 333;
  ScaleA: TNeuralFloat = 0.02;
  ScaleB: TNeuralFloat = 0.5;
var
  A, B: TInt8DynArr;
  BFloat: array of TNeuralFloat;
  ViaFloat, ViaInt: TNeuralFloat;
  i: integer;
begin
  SetLength(A, N);
  SetLength(B, N);
  SetLength(BFloat, N);
  for i := 0 to N - 1 do
  begin
    A[i] := ShortInt(((i * 53) mod 255) - 127);
    B[i] := ShortInt(((i * 17 + 9) mod 255) - 127);
  end;
  TNNetVolume.DequantizeInt8(TNeuralFloatArrPtr(@BFloat[0]),
    TNeuralInt8ArrPtr(@B[0]), N, ScaleB);
  ViaFloat := ScaleA * TNNetVolume.DotProductInt8(TNeuralInt8ArrPtr(@A[0]),
    TNeuralFloatArrPtr(@BFloat[0]), N);
  ViaInt := ScaleA * ScaleB * TNNetVolume.DotProductInt8Int8(
    TNeuralInt8ArrPtr(@A[0]), TNeuralInt8ArrPtr(@B[0]), N);
  AssertEquals('int8 x int8 vs int8 x float', ViaFloat, ViaInt,
    Abs(ViaFloat) * 1e-5);
end;

// QuantizeInt8 then DequantizeInt8 must land within half a code of the input,
// which is the whole accuracy claim of the int8 weight path. Binds the two
// primitives together so a lane-order bug in either shows up here.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDequantizeInt8RoundTrip;
const
  N = 300;
var
  V: TNNetVolume;
  Codes: TInt8DynArr;
  Back: array of TNeuralFloat;
  MaxAbs, Scale: TNeuralFloat;
  i: integer;
begin
  RandSeed := 24680;
  V := TNNetVolume.Create(N, 1, 1);
  SetLength(Codes, N);
  SetLength(Back, N);
  try
    for i := 0 to N - 1 do V.FData[i] := (Random - 0.5) * 3.2;
    MaxAbs := TNNetVolume.MaxAbsFinite(TNeuralFloatArrPtr(@V.FData[0]), N);
    AssertTrue('random row has a range', MaxAbs > 0);
    Scale := MaxAbs / 127;
    TNNetVolume.QuantizeInt8(TNeuralInt8ArrPtr(@Codes[0]),
      TNeuralFloatArrPtr(@V.FData[0]), N, MaxAbs);
    TNNetVolume.DequantizeInt8(TNeuralFloatArrPtr(@Back[0]),
      TNeuralInt8ArrPtr(@Codes[0]), N, Scale);
    for i := 0 to N - 1 do
      AssertTrue('element ' + IntToStr(i) + ' recovers within half a code: ' +
        FloatToStr(V.FData[i]) + ' -> ' + FloatToStr(Back[i]),
        Abs(Back[i] - V.FData[i]) <= Scale * 0.5 + 1e-7);
  finally
    V.Free;
  end;
end;

// A bfloat16 IS the high half of a single, so decoding is exactly that shift -
// which is what the expectation below computes. Exact on every build, hence
// delta 0. N = 21 crosses the AVX body into the tail. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDecodeBF16;
const
  // 1.0, -2.0, +0.0, -0.0, ~3.14, -0.5, ~3.39e38, a denormal single, 0.1-ish,
  // then filler - none of them Inf or NaN, so AssertEquals stays safe.
  Pats: array[0..20] of Word = ($3F80, $C000, $0000, $8000, $4049, $BF00,
    $7F7F, $0001, $3DCC, $4120, $C2F0, $3F00, $BE80, $4780, $C780, $3E00,
    $B800, $4400, $C400, $3B00, $BB80);
  N = 21;
var
  Dst: array of TNeuralFloat;
  Expected: TNeuralFloat;
  OutBits: Cardinal;
  i: integer;
begin
  SetLength(Dst, N);
  for i := 0 to N - 1 do Dst[i] := 12345;
  TNNetVolume.DecodeBF16(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralHalfArrPtr(@Pats[0]), N);
  for i := 0 to N - 1 do
  begin
    OutBits := Cardinal(Pats[i]) shl 16;
    Expected := PSingle(@OutBits)^;
    AssertEquals('bf16 $' + IntToHex(Pats[i], 4), Expected, Dst[i], 0);
  end;
  // Under csMinAvxSize: the scalar method, same answers.
  for i := 0 to 4 do Dst[i] := 12345;
  TNNetVolume.DecodeBF16(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralHalfArrPtr(@Pats[0]), 5);
  for i := 0 to 4 do
  begin
    OutBits := Cardinal(Pats[i]) shl 16;
    AssertEquals('short-run bf16 ' + IntToStr(i), PSingle(@OutBits)^, Dst[i], 0);
  end;
end;

// The length sweep TestDecodeBF16 lacks: its single N = 21 never reaches the
// 32-element unrolled body. Every length here is checked against the same shift
// the scalar tail performs, so the delta is 0. The patterns keep bit 7 clear,
// so the exponent is never all ones and no Inf or NaN reaches AssertEquals.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDecodeBF16LengthSweep;
const
  cLengths: array[0..14] of integer =
    (1, 7, 8, 16, 24, 31, 32, 33, 39, 40, 47, 64, 128, 1000, 1024);
  N = 1024;
var
  Bits: array of Word;
  Bulk: array of TNeuralFloat;
  Expected: TNeuralFloat;
  OutBits: Cardinal;
  i, L, Len: integer;
begin
  SetLength(Bits, N);
  SetLength(Bulk, N + 1);   // one guard slot past the longest run
  for i := 0 to N - 1 do
  begin
    Bits[i] := Word((i * 61) and $7F7F);       // exponent never all ones
    if (i and 1) = 1 then Bits[i] := Bits[i] or $8000;
  end;
  for L := 0 to High(cLengths) do
  begin
    Len := cLengths[L];
    for i := 0 to N do Bulk[i] := 12345;
    TNNetVolume.DecodeBF16(TNeuralFloatArrPtr(@Bulk[0]),
      TNeuralHalfArrPtr(@Bits[0]), Len);
    for i := 0 to Len - 1 do
    begin
      OutBits := Cardinal(Bits[i]) shl 16;
      Expected := PSingle(@OutBits)^;
      AssertEquals('element ' + IntToStr(i) + ' at N=' + IntToStr(Len),
        Expected, Bulk[i], 0);
    end;
    AssertEquals('wrote past N=' + IntToStr(Len), 12345.0, Bulk[Len], 0);
  end;
end;

// Every IEEE half is exactly representable as a single, so these are exact
// values, not approximations - delta 0. Includes a subnormal (2^-24) and the
// largest finite half (65504), the two places a widening implementation
// usually breaks. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDecodeF16;
const
  N = 21;
  Bits: array[0..20] of Word = (
    $3C00, $BC00, $0000, $8000, $4000, $C000, $3800, $B800,
    $7BFF, $FBFF, $0001, $8001, $03FF, $0400, $3555, $4248,
    $5140, $D140, $1400, $6400, $E400);
  Vals: array[0..20] of TNeuralFloat = (
    1.0, -1.0, 0.0, -0.0, 2.0, -2.0, 0.5, -0.5,
    65504.0, -65504.0,
    5.9604644775390625e-8,      // smallest subnormal half, 2^-24
    -5.9604644775390625e-8,
    6.0975551605224609e-5,      // largest subnormal half
    6.103515625e-5,             // smallest normal half, 2^-14
    0.333251953125,             // nearest half to 1/3
    3.140625,                   // nearest half to pi
    42.0, -42.0,
    0.0009765625,               // 2^-10
    1024.0, -1024.0);
var
  Dst: array of TNeuralFloat;
  i: integer;
begin
  SetLength(Dst, N);
  for i := 0 to N - 1 do Dst[i] := 12345;
  TNNetVolume.DecodeF16(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralHalfArrPtr(@Bits[0]), N);
  for i := 0 to N - 1 do
    AssertEquals('f16 $' + IntToHex(Bits[i], 4), Vals[i], Dst[i], 0);
  // Under csMinAvxSize: the scalar method, same answers.
  for i := 0 to 4 do Dst[i] := 12345;
  TNNetVolume.DecodeF16(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralHalfArrPtr(@Bits[0]), 5);
  for i := 0 to 4 do
    AssertEquals('short-run f16 ' + IntToStr(i), Vals[i], Dst[i], 0);
end;

// Inf and NaN halves must widen to Inf and NaN singles rather than trapping -
// FPC leaves the SSE invalid-operation exception unmasked, so a decode that
// touched these values arithmetically would raise EInvalidOp here.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDecodeF16SpecialValues;
const
  N = 20;
  // +Inf, -Inf, quiet NaN, a NaN with a payload, then finite filler out to a
  // length that still runs the vector body.
  Bits: array[0..19] of Word = (
    $7C00, $FC00, $7E00, $7DAB, $3C00, $0000, $BC00, $4000,
    $3C00, $0000, $BC00, $4000, $3C00, $0000, $BC00, $4000,
    $7C00, $FC00, $7E00, $3C00);
var
  Dst: array of TNeuralFloat;
  i: integer;
begin
  SetLength(Dst, N);
  for i := 0 to N - 1 do Dst[i] := 12345;
  TNNetVolume.DecodeF16(TNeuralFloatArrPtr(@Dst[0]),
    TNeuralHalfArrPtr(@Bits[0]), N);
  for i := 0 to N - 1 do
    case Bits[i] of
      $7C00: AssertTrue('slot ' + IntToStr(i) + ' is +Inf',
               IsInfinite(Dst[i]) and (Dst[i] > 0));
      $FC00: AssertTrue('slot ' + IntToStr(i) + ' is -Inf',
               IsInfinite(Dst[i]) and (Dst[i] < 0));
      $7E00, $7DAB: AssertTrue('slot ' + IntToStr(i) + ' is NaN',
               IsNan(Dst[i]));
      $3C00: AssertEquals('slot ' + IntToStr(i), 1.0, Dst[i], 0);
      $BC00: AssertEquals('slot ' + IntToStr(i), -1.0, Dst[i], 0);
      $4000: AssertEquals('slot ' + IntToStr(i), 2.0, Dst[i], 0);
      $0000: AssertEquals('slot ' + IntToStr(i), 0.0, Dst[i], 0);
    end;
end;

// Narrowing is lossy, so these are the exact half bit patterns IEEE
// round-to-nearest-even produces. The interesting rows are the ties: 65520 is
// halfway between the largest finite half and the next power of two and must
// round UP to Inf, 2^-25 is halfway between zero and the smallest subnormal
// and must round DOWN to zero, and 1.5*2^-24 must round to the EVEN
// subnormal $0002. 1e30 overflows the half range, which is also the input
// that traps a vcvtps2ph loop running with FPC's default MXCSR.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestEncodeF16;
const
  N = 21;
  Vals: array[0..20] of TNeuralFloat = (
    1.0, -1.0, 0.0, -0.0, 2.0, 0.5,
    65504.0,                    // largest finite half
    65520.0,                    // tie above it -> Inf
    1e30, -1e30,                // overflow -> +/-Inf
    0.333251953125,             // nearest half to 1/3
    3.140625,                   // nearest half to pi
    5.9604644775390625e-8,      // smallest subnormal half, 2^-24
    2.98023223876953125e-8,     // 2^-25: tie to even -> zero
    8.940696716308594e-8,       // 1.5 * 2^-24: tie to even -> $0002
    6.0975551605224609e-5,      // largest subnormal half
    6.103515625e-5,             // smallest normal half, 2^-14
    1e-10,                      // far below the subnormal range -> zero
    42.0, -42.0, 1024.0);
  Bits: array[0..20] of Word = (
    $3C00, $BC00, $0000, $8000, $4000, $3800,
    $7BFF, $7C00, $7C00, $FC00,
    $3555, $4248,
    $0001, $0000, $0002, $03FF, $0400, $0000,
    $5140, $D140, $6400);
var
  Dst: array of Word;
  i: integer;
begin
  SetLength(Dst, N);
  for i := 0 to N - 1 do Dst[i] := $DEAD;
  TNNetVolume.EncodeF16(TNeuralHalfArrPtr(@Dst[0]),
    TNeuralFloatArrPtr(@Vals[0]), N);
  for i := 0 to N - 1 do
    AssertEquals('half of ' + FloatToStr(Vals[i]),
      IntToHex(Bits[i], 4), IntToHex(Dst[i], 4));
  // Under csMinAvxSize: the scalar method, same answers.
  for i := 0 to 4 do Dst[i] := $DEAD;
  TNNetVolume.EncodeF16(TNeuralHalfArrPtr(@Dst[0]),
    TNeuralFloatArrPtr(@Vals[0]), 5);
  for i := 0 to 4 do
    AssertEquals('short-run half ' + IntToStr(i),
      IntToHex(Bits[i], 4), IntToHex(Dst[i], 4));
end;

// Inf and NaN singles must narrow to Inf and NaN halves rather than trapping.
// FPC leaves the SSE invalid-operation exception unmasked, so a signalling NaN
// reaching vcvtps2ph unmasked would raise EInvalidOp here. A NaN narrows to the
// quiet NaN of the same top-10 payload bits, so the assertion checks the class
// (all-ones exponent, non-zero mantissa) rather than one pattern.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestEncodeF16SpecialValues;
const
  N = 20;
var
  Vals: array[0..19] of TNeuralFloat;
  Dst: array of Word;
  SrcBits: Cardinal;
  i: integer;
begin
  for i := 0 to N - 1 do Vals[i] := 1.0;
  SrcBits := $7F800000; Vals[0] := PSingle(@SrcBits)^;   // +Inf
  SrcBits := $FF800000; Vals[1] := PSingle(@SrcBits)^;   // -Inf
  SrcBits := $7FC00000; Vals[2] := PSingle(@SrcBits)^;   // quiet NaN
  SrcBits := $7F800001; Vals[3] := PSingle(@SrcBits)^;   // signalling NaN
  SrcBits := $FFABCDEF; Vals[4] := PSingle(@SrcBits)^;   // NaN with a payload
  Vals[5] := -0.0;
  Vals[6] := 1e-45;                                      // subnormal single
  Vals[7] := 3.4028235e38;                               // largest finite single
  SetLength(Dst, N);
  for i := 0 to N - 1 do Dst[i] := $DEAD;
  TNNetVolume.EncodeF16(TNeuralHalfArrPtr(@Dst[0]),
    TNeuralFloatArrPtr(@Vals[0]), N);
  AssertEquals('+Inf', IntToHex($7C00, 4), IntToHex(Dst[0], 4));
  AssertEquals('-Inf', IntToHex($FC00, 4), IntToHex(Dst[1], 4));
  for i := 2 to 4 do
    AssertTrue('slot ' + IntToStr(i) + ' is a NaN half',
      ((Dst[i] and $7C00) = $7C00) and ((Dst[i] and $03FF) <> 0));
  AssertEquals('-0.0', IntToHex($8000, 4), IntToHex(Dst[5], 4));
  AssertEquals('subnormal single', IntToHex($0000, 4), IntToHex(Dst[6], 4));
  AssertEquals('largest single', IntToHex($7C00, 4), IntToHex(Dst[7], 4));
  for i := 8 to N - 1 do
    AssertEquals('filler ' + IntToStr(i), IntToHex($3C00, 4), IntToHex(Dst[i], 4));
end;

// The vectorized run and the scalar run must agree bit-for-bit: the same values
// are encoded once in bulk (which takes the F16C path on an AVX2 build) and
// once one element at a time (always the scalar path, being under
// csMinAvxSize). The values sweep nine decades, so they cross the overflow,
// normal, subnormal and flush-to-zero regions.
//
// The lengths are the ones that separate the three parts of the vectorized
// routine: under 32 uses only the 8-at-a-time loop, a multiple of 32 uses only
// the unrolled body, and the rest exercise a body-plus-remainder-plus-tail
// combination. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestEncodeF16MatchesScalar;
const
  cLengths: array[0..12] of integer =
    (1, 7, 8, 16, 31, 32, 33, 39, 40, 47, 128, 1000, 1024);
  N = 1024;
var
  Vals: array of TNeuralFloat;
  Bulk, OneByOne: array of Word;
  i, L, Len, Mismatches: integer;
begin
  SetLength(Vals, N);
  SetLength(Bulk, N + 1);   // one guard slot past the longest run
  SetLength(OneByOne, N);
  for i := 0 to N - 1 do
    Vals[i] := (i - 512) * 0.0011 * Power(10, (i mod 19) - 9);
  for i := 0 to N - 1 do
    TNNetVolume.EncodeF16(TNeuralHalfArrPtr(@OneByOne[i]),
      TNeuralFloatArrPtr(@Vals[i]), 1);
  for L := 0 to High(cLengths) do
  begin
    Len := cLengths[L];
    for i := 0 to N do Bulk[i] := $DEAD;
    TNNetVolume.EncodeF16(TNeuralHalfArrPtr(@Bulk[0]),
      TNeuralFloatArrPtr(@Vals[0]), Len);
    Mismatches := 0;
    for i := 0 to Len - 1 do
      if Bulk[i] <> OneByOne[i] then Inc(Mismatches);
    AssertEquals('bulk vs scalar half mismatches at N=' + IntToStr(Len),
      0, Mismatches);
    AssertEquals('write past N=' + IntToStr(Len),
      IntToHex($DEAD, 4), IntToHex(Bulk[Len], 4));
  end;
end;

// The decode twin of TestEncodeF16MatchesScalar, over the same lengths. The
// patterns walk the half range with the all-ones exponent excluded: a
// signalling NaN is the one input where the two paths legitimately differ (the
// F16C path quiets it, the scalar path passes it through), and
// TestDecodeF16SpecialValues covers that case instead. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestDecodeF16MatchesScalar;
const
  cLengths: array[0..12] of integer =
    (1, 7, 8, 16, 31, 32, 33, 39, 40, 47, 128, 1000, 1024);
  N = 1024;
var
  Bits: array of Word;
  Bulk, OneByOne: array of TNeuralFloat;
  i, L, Len, Mismatches: integer;
begin
  SetLength(Bits, N);
  SetLength(Bulk, N + 1);   // one guard slot past the longest run
  SetLength(OneByOne, N);
  for i := 0 to N - 1 do
  begin
    Bits[i] := Word((i * 61) and $7BFF);       // exponent never all ones
    if (i and 1) = 1 then Bits[i] := Bits[i] or $8000;
  end;
  for i := 0 to N - 1 do
    TNNetVolume.DecodeF16(TNeuralFloatArrPtr(@OneByOne[i]),
      TNeuralHalfArrPtr(@Bits[i]), 1);
  for L := 0 to High(cLengths) do
  begin
    Len := cLengths[L];
    for i := 0 to N do Bulk[i] := 12345;
    TNNetVolume.DecodeF16(TNeuralFloatArrPtr(@Bulk[0]),
      TNeuralHalfArrPtr(@Bits[0]), Len);
    Mismatches := 0;
    for i := 0 to Len - 1 do
      if Bulk[i] <> OneByOne[i] then Inc(Mismatches);
    AssertEquals('bulk vs scalar single mismatches at N=' + IntToStr(Len),
      0, Mismatches);
    AssertEquals('write past N=' + IntToStr(Len), 12345.0, Bulk[Len], 0);
  end;
end;

// Narrowing to bfloat16 drops the low 16 bits of the single, so these are the
// exact words round-to-nearest-even produces. The values are built from bit
// patterns rather than decimal literals so the ties are exact. The interesting
// rows are the three ties (low half exactly $8000), which must go to the EVEN
// kept word in both directions, and the largest finite single, whose round-up
// leaves the bfloat16 range and lands on Inf. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestEncodeBF16;
const
  N = 21;
  SrcBits: array[0..20] of Cardinal = (
    $3F800000, $BF800000, $00000000, $80000000, $40000000, $3F000000,
    $3EAAAAAB,                  // nearest single to 1/3, rounds up
    $40490FDB,                  // nearest single to pi, rounds down
    $3F808000,                  // tie, kept word even -> stays
    $3F818000,                  // tie, kept word odd  -> rounds up
    $7F7FFFFF, $FF7FFFFF,       // largest finite singles -> +/-Inf
    $7F800000, $FF800000,       // +/-Inf
    $00000001,                  // smallest subnormal single -> zero
    $00008000,                  // tie at zero, rounds down
    $00018000,                  // tie, kept word odd -> $0002
    $42280000, $C2280000, $44800000,
    $3F7FFFFF);                 // just under 1.0, rounds up to it
  Bits: array[0..20] of Word = (
    $3F80, $BF80, $0000, $8000, $4000, $3F00,
    $3EAB, $4049, $3F80, $3F82, $7F80, $FF80,
    $7F80, $FF80, $0000, $0000, $0002,
    $4228, $C228, $4480, $3F80);
var
  Vals: array[0..20] of TNeuralFloat;
  Dst: array of Word;
  i: integer;
begin
  for i := 0 to N - 1 do Vals[i] := PSingle(@SrcBits[i])^;
  SetLength(Dst, N);
  for i := 0 to N - 1 do Dst[i] := $DEAD;
  TNNetVolume.EncodeBF16(TNeuralHalfArrPtr(@Dst[0]),
    TNeuralFloatArrPtr(@Vals[0]), N);
  for i := 0 to N - 1 do
    AssertEquals('bfloat16 of ' + IntToHex(SrcBits[i], 8),
      IntToHex(Bits[i], 4), IntToHex(Dst[i], 4));
  // Under csMinAvxSize: the scalar method, same answers.
  for i := 0 to 4 do Dst[i] := $DEAD;
  TNNetVolume.EncodeBF16(TNeuralHalfArrPtr(@Dst[0]),
    TNeuralFloatArrPtr(@Vals[0]), 5);
  for i := 0 to 4 do
    AssertEquals('short-run bfloat16 ' + IntToStr(i),
      IntToHex(Bits[i], 4), IntToHex(Dst[i], 4));
end;

// A NaN must stay a NaN. Rounding alone would carry $7F800001 up to $7F80,
// which DecodeBF16 reads back as an Inf, so both paths force the quiet bit
// instead. Nothing here may trap either: the AVX2 kernel is integer-only, so
// unlike EncodeF16 it runs with FPC's default MXCSR untouched.
// Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestEncodeBF16SpecialValues;
const
  N = 20;
var
  Vals: array[0..19] of TNeuralFloat;
  Dst: array of Word;
  SrcBits: Cardinal;
  i: integer;
begin
  for i := 0 to N - 1 do Vals[i] := 1.0;
  SrcBits := $7F800000; Vals[0] := PSingle(@SrcBits)^;   // +Inf
  SrcBits := $FF800000; Vals[1] := PSingle(@SrcBits)^;   // -Inf
  SrcBits := $7FC00000; Vals[2] := PSingle(@SrcBits)^;   // quiet NaN
  SrcBits := $7F800001; Vals[3] := PSingle(@SrcBits)^;   // signalling NaN
  SrcBits := $FFABCDEF; Vals[4] := PSingle(@SrcBits)^;   // NaN with a payload
  Vals[5] := -0.0;
  Vals[6] := 1e-45;                                      // subnormal single
  Vals[7] := 3.4028235e38;                               // largest finite single
  SetLength(Dst, N);
  for i := 0 to N - 1 do Dst[i] := $DEAD;
  TNNetVolume.EncodeBF16(TNeuralHalfArrPtr(@Dst[0]),
    TNeuralFloatArrPtr(@Vals[0]), N);
  AssertEquals('+Inf', IntToHex($7F80, 4), IntToHex(Dst[0], 4));
  AssertEquals('-Inf', IntToHex($FF80, 4), IntToHex(Dst[1], 4));
  for i := 2 to 4 do
    AssertTrue('slot ' + IntToStr(i) + ' is a NaN bfloat16',
      ((Dst[i] and $7F80) = $7F80) and ((Dst[i] and $007F) <> 0));
  AssertEquals('-0.0', IntToHex($8000, 4), IntToHex(Dst[5], 4));
  AssertEquals('subnormal single', IntToHex($0000, 4), IntToHex(Dst[6], 4));
  AssertEquals('largest single', IntToHex($7F80, 4), IntToHex(Dst[7], 4));
  for i := 8 to N - 1 do
    AssertEquals('filler ' + IntToStr(i), IntToHex($3F80, 4), IntToHex(Dst[i], 4));
end;

// The vectorized run and the scalar run must agree bit-for-bit, over the same
// lengths TestEncodeF16MatchesScalar uses: under 32 exercises only the
// 8-at-a-time loop, a multiple of 32 only the unrolled body, and the rest a
// body-plus-remainder-plus-tail combination. The values sweep nine decades in
// both signs. Coded by Claude (AI).
procedure TTestNeuralVolumeQuant8.TestEncodeBF16MatchesScalar;
const
  cLengths: array[0..12] of integer =
    (1, 7, 8, 16, 31, 32, 33, 39, 40, 47, 128, 1000, 1024);
  N = 1024;
var
  Vals: array of TNeuralFloat;
  Bulk, OneByOne: array of Word;
  i, L, Len, Mismatches: integer;
begin
  SetLength(Vals, N);
  SetLength(Bulk, N + 1);   // one guard slot past the longest run
  SetLength(OneByOne, N);
  for i := 0 to N - 1 do
    Vals[i] := (i - 512) * 0.0011 * Power(10, (i mod 19) - 9);
  for i := 0 to N - 1 do
    TNNetVolume.EncodeBF16(TNeuralHalfArrPtr(@OneByOne[i]),
      TNeuralFloatArrPtr(@Vals[i]), 1);
  for L := 0 to High(cLengths) do
  begin
    Len := cLengths[L];
    for i := 0 to N do Bulk[i] := $DEAD;
    TNNetVolume.EncodeBF16(TNeuralHalfArrPtr(@Bulk[0]),
      TNeuralFloatArrPtr(@Vals[0]), Len);
    Mismatches := 0;
    for i := 0 to Len - 1 do
      if Bulk[i] <> OneByOne[i] then Inc(Mismatches);
    AssertEquals('bulk vs scalar bfloat16 mismatches at N=' + IntToStr(Len),
      0, Mismatches);
    AssertEquals('write past N=' + IntToStr(Len),
      IntToHex($DEAD, 4), IntToHex(Bulk[Len], 4));
  end;
end;

// Straightforward O(pSize*pSize) box sum: the definition the summed-area-table
// implementation in TNNetVolume.CalculateLocalResponseFrom2D has to reproduce.
procedure ReferenceLocalResponse2D(Dest, Original: TNNetVolume;
  pSize: integer; alpha, beta: TNeuralFloat);
var
  iFrom, iTo, CountIX, CountIY: integer;
  MaxX, MaxY, MaxD: integer;
  MinIX, MaxIX, MinIY, MaxIY: integer;
  CountX, CountY, CountD: integer;
  Sum, Scale: TNeuralFloat;
begin
  Dest.ReSize(Original);
  MaxX := Original.SizeX - 1;
  MaxY := Original.SizeY - 1;
  MaxD := Original.Depth - 1;
  iTo := pSize shr 1;
  iFrom := -iTo;
  Scale := alpha / (pSize * pSize);
  for CountX := 0 to MaxX do
  begin
    MinIX := Max(CountX + iFrom, 0);
    MaxIX := Min(CountX + iTo, MaxX);
    for CountY := 0 to MaxY do
    begin
      MinIY := Max(CountY + iFrom, 0);
      MaxIY := Min(CountY + iTo, MaxY);
      for CountD := 0 to MaxD do
      begin
        Sum := 1;
        for CountIX := MinIX to MaxIX do
          for CountIY := MinIY to MaxIY do
            Sum := Sum + Scale * Sqr(Original[CountIX, CountIY, CountD]);
        Dest[CountX, CountY, CountD] := Power(Sum, beta);
      end;
    end;
  end;
end;

// Straightforward O(pSize) depth window: the definition the prefix-sum
// implementation in TNNetVolume.CalculateLocalResponseFromDepth reproduces.
procedure ReferenceLocalResponseDepth(Dest, Original: TNNetVolume;
  pSize: integer; alpha, beta: TNeuralFloat);
var
  iFrom, iTo, CountID: integer;
  MaxX, MaxY, MaxD: integer;
  MinID, MaxID: integer;
  CountX, CountY, CountD: integer;
  Sum, Scale: TNeuralFloat;
begin
  Dest.ReSize(Original);
  MaxX := Original.SizeX - 1;
  MaxY := Original.SizeY - 1;
  MaxD := Original.Depth - 1;
  iTo := pSize shr 1;
  iFrom := -iTo;
  Scale := alpha / pSize;
  for CountX := 0 to MaxX do
    for CountY := 0 to MaxY do
      for CountD := 0 to MaxD do
      begin
        MinID := Max(CountD + iFrom, 0);
        MaxID := Min(CountD + iTo, MaxD);
        Sum := 1;
        for CountID := MinID to MaxID do
          Sum := Sum + Scale * Sqr(Original[CountX, CountY, CountID]);
        Dest[CountX, CountY, CountD] := Power(Sum, beta);
      end;
end;

procedure TTestNeuralVolume.TestLocalResponse2DMatchesReference;
var
  Original, Got, Want, Scratch: TNNetVolume;
  I, pSize: integer;
begin
  Original := TNNetVolume.Create(7, 5, 6);
  Got      := TNNetVolume.Create(1, 1, 1);
  Want     := TNNetVolume.Create(1, 1, 1);
  Scratch  := TNNetVolume.Create(1, 1, 1);
  try
    RandSeed := 1234;
    for I := 0 to Original.Size - 1 do
      Original.FData[I] := (Random - 0.5) * 20;
    for pSize := 1 to 5 do
    begin
      ReferenceLocalResponse2D(Want, Original, pSize, 0.001 / 9.0, 0.75);
      Got.CalculateLocalResponseFrom2D(Original, Scratch, pSize, 0.001 / 9.0, 0.75);
      AssertEquals('Size for pSize ' + IntToStr(pSize), Want.Size, Got.Size);
      for I := 0 to Want.Size - 1 do
        AssertEquals('pSize ' + IntToStr(pSize) + ' element ' + IntToStr(I),
          Want.FData[I], Got.FData[I], 1e-5);
    end;
    // The scratch is reused across calls without reallocation.
    AssertEquals('Scratch keeps the input shape', Original.Size, Scratch.Size);
  finally
    Scratch.Free;
    Want.Free;
    Got.Free;
    Original.Free;
  end;
end;

procedure TTestNeuralVolume.TestLocalResponseDepthMatchesReference;
var
  Original, Got, Want, Scratch: TNNetVolume;
  I, pSize: integer;
begin
  Original := TNNetVolume.Create(4, 3, 9);
  Got      := TNNetVolume.Create(1, 1, 1);
  Want     := TNNetVolume.Create(1, 1, 1);
  Scratch  := TNNetVolume.Create(1, 1, 1);
  try
    RandSeed := 4321;
    for I := 0 to Original.Size - 1 do
      Original.FData[I] := (Random - 0.5) * 20;
    // pSize runs past twice the depth so the window-clamp ranges the
    // implementation splits the depth axis into are each driven empty in turn.
    for pSize := 1 to 21 do
    begin
      ReferenceLocalResponseDepth(Want, Original, pSize, 0.001 / 9.0, 0.75);
      Got.CalculateLocalResponseFromDepth(Original, Scratch, pSize, 0.001 / 9.0, 0.75);
      AssertEquals('Size for pSize ' + IntToStr(pSize), Want.Size, Got.Size);
      for I := 0 to Want.Size - 1 do
        AssertEquals('pSize ' + IntToStr(pSize) + ' element ' + IntToStr(I),
          Want.FData[I], Got.FData[I], 1e-5);
    end;
    AssertEquals('Scratch keeps the input shape', Original.Size, Scratch.Size);
  finally
    Scratch.Free;
    Want.Free;
    Got.Free;
    Original.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumePointwiseNormAndMul;
var
  Normalized, Scaled, Norms: TNNetVolume;
  CountX, CountY: integer;
begin
  Normalized := TNNetVolume.Create(2, 3, 2);
  Scaled := TNNetVolume.Create(2, 3, 2);
  Norms := TNNetVolume.Create(1, 1, 1);
  try
    for CountX := 0 to Normalized.Size - 1 do Normalized.FData[CountX] := CountX + 1;
    Scaled.Copy(Normalized);
    Normalized.PointwiseNorm(Norms);
    AssertEquals('Norms size X', 2, Norms.SizeX);
    AssertEquals('Norms size Y', 3, Norms.SizeY);
    for CountY := 0 to 2 do
      for CountX := 0 to 1 do
        AssertEquals('Unit modulus at ' + IntToStr(CountX) + ',' + IntToStr(CountY),
          1.0,
          Sqrt(Sqr(Normalized[CountX, CountY, 0]) + Sqr(Normalized[CountX, CountY, 1])),
          0.0001);
    // PointwiseMul reapplies the recorded multipliers, so it reproduces PointwiseNorm.
    Scaled.PointwiseMul(Norms);
    for CountX := 0 to Normalized.Size - 1 do
      AssertEquals('Element ' + IntToStr(CountX),
        Normalized.FData[CountX], Scaled.FData[CountX], 0.0001);
  finally
    Norms.Free;
    Scaled.Free;
    Normalized.Free;
  end;
end;

procedure TTestNeuralVolume.TestVolumePointwiseMulWithoutNorms;
var
  Scaled: TNNetVolume;
  CountElement: integer;
begin
  Scaled := TNNetVolume.Create(2, 3, 2);
  try
    for CountElement := 0 to Scaled.Size - 1 do Scaled.FData[CountElement] := CountElement + 1;
    Scaled.PointwiseMul(nil);
    for CountElement := 0 to Scaled.Size - 1 do
      AssertEquals('Element ' + IntToStr(CountElement),
        CountElement + 1, Scaled.FData[CountElement], 0.0001);
  finally
    Scaled.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralVolume);
  RegisterTest(TTestNeuralVolumeQuant8);

end.
