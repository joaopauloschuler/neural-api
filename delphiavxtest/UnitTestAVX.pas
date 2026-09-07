unit UnitTestAVX;

interface

uses
  TestFramework,
  System.SysUtils,
  System.Math,
  {$IFDEF WIN32} neuralavx32w, {$ENDIF}
  {$IFDEF WIN64} neuralavx64w, {$ENDIF}
  neuralavx;          // Unit containing all AVX functions

type
  TTestAVX = class(TTestCase)
  private
    // Helper comparison functions
    function CompareSingleArrays(const A, B: array of Single; ACount: Integer; AbsTol: Single = 1e-5; RelTol: Single = 1e-6): Boolean;
    function CompareWordArrays(const A, B: array of Word; ACount: Integer; Epsilon: Word = 0): Boolean;
    function CompareShortIntArrays(const A, B: array of ShortInt; ACount: Integer): Boolean;
    procedure RandomSingleArray(var Arr: array of Single);
    procedure FillWithConst(var Arr: array of Single; const Value: Single);
    procedure FillWithSeq(var Arr: array of Single; Start, Step: Single);

    // Conversion helpers for half and bfloat16 (IEEE-754 binary16 / bfloat16)
    function SingleToHalf(Value: Single): Word;
    function HalfToSingle(Value: Word): Single;
    function SingleToBFloat16(Value: Single): Word;
    function BFloat16ToSingle(Value: Word): Single;

  published
    // Test methods for each AVX function
    procedure TestFillMem;
    procedure TestCopyRelu;
    procedure TestMulAdd;
    procedure TestMulAddF;
    procedure TestMulMulAdd;
    procedure TestMulF;
    procedure TestMul;
    procedure TestAdd;
    procedure TestMax;
    procedure TestSumDiff;
    procedure TestDistanceSqr;
    procedure TestSub;
    procedure TestGetSum;
    procedure TestGetSumSqr;
    procedure TestExp;
    procedure TestDotProd;
    procedure TestDotProdInt8;
    procedure TestMulAddInt8Scalar;
    procedure TestMulAddInt8;
    procedure TestMaxAbsFinite;
    procedure TestQuantizeInt8;
    procedure TestDequantizeInt8;
    procedure TestDecodeBF16;
    procedure TestReluGateMask;
    procedure TestLeakyRelu;
    procedure TestDecodeF16;
    procedure TestSumSqrCentered;
    procedure TestAdamDelta;
    procedure TestAdafactorDelta;
    procedure TestClampAbs;
    procedure TestLionDelta;
    procedure TestGetMaxPos;
    procedure TestGetMinPos;
    procedure TestGetMaxAbsPos;
    procedure TestAddScalar;
    procedure TestExpShiftSum;
    procedure TestLn;
    procedure TestSinCos;
    procedure TestEncodeBF16;
    procedure TestReluL;
    procedure TestEncodeF16;
    procedure TestReluLGateMask;
  end;

implementation

uses
  System.RTLConsts;

{------------------------------------------------------------------------------
  Helper functions
-----------------------------------------------------------------------------}

function TTestAVX.CompareSingleArrays(const A, B: array of Single; ACount: Integer; AbsTol: Single = 1e-5; RelTol: Single = 1e-6): Boolean;
var
  i: Integer;
  Diff, MaxVal: Single;
begin
  Result := (Length(A) >= ACount) and (Length(B) >= ACount);
  if not Result then Exit;
  for i := 0 to ACount - 1 do
  begin
    Diff := Abs(A[i] - B[i]);
    MaxVal := Max(Abs(A[i]), Abs(B[i]));
    if Diff > Max(AbsTol, MaxVal * RelTol) then
    begin
      Result := False;
      Exit;
    end;
  end;
end;

function TTestAVX.CompareWordArrays(const A, B: array of Word; ACount: Integer; Epsilon: Word = 0): Boolean;
var
  i: Integer;
begin
  Result := (Length(A) >= ACount) and (Length(B) >= ACount);
  if not Result then Exit;

  for i := 0 to ACount - 1 do
    if Abs(A[i] - B[i]) > Epsilon then Exit(False);
  Result := True;
end;

function TTestAVX.CompareShortIntArrays(const A, B: array of ShortInt; ACount: Integer): Boolean;
var
  i: Integer;
begin
  Result := (Length(A) >= ACount) and (Length(B) >= ACount);
  if not Result then Exit;

  for i := 0 to ACount - 1 do
    if A[i] <> B[i] then
    begin
      Result := False;
      Exit;
    end;
end;

procedure TTestAVX.RandomSingleArray(var Arr: array of Single);
var
  i: Integer;
begin
  for i := 0 to High(Arr) do
    Arr[i] := (Random(20000) - 10000) / 1000;  // -10 .. +10
  // Insert special values for edge testing
  if Length(Arr) > 0 then
  begin
    Arr[0] := 0;
    if Length(Arr) > 1 then Arr[1] := 1e-30;
    if Length(Arr) > 2 then Arr[2] := 1e30;
    if Length(Arr) > 3 then Arr[3] := NaN;
    if Length(Arr) > 4 then Arr[4] := Infinity;
    if Length(Arr) > 5 then Arr[5] := -Infinity;
  end;
end;

procedure TTestAVX.FillWithConst(var Arr: array of Single; const Value: Single);
var
  i: Integer;
begin
  for i := 0 to High(Arr) do
    Arr[i] := Value;
end;

procedure TTestAVX.FillWithSeq(var Arr: array of Single; Start, Step: Single);
var
  i: Integer;
begin
  for i := 0 to High(Arr) do
    Arr[i] := Start + i * Step;
end;

{-----------------------------------------------------------------------------
  Convert single-precision float to half-precision (binary16) with
  round-to-nearest-even, matching hardware vcvtps2ph (imm8=0).

  Algorithm:
    - Extract sign, exponent, mantissa.
    - Handle zero, denormal, infinity, NaN.
    - Adjust exponent bias (127 -> 15).
    - Apply round-to-nearest-even on the low 13 bits of mantissa.
    - Pack result.
-----------------------------------------------------------------------------}
function TTestAVX.SingleToHalf(Value: Single): Word;
var
  Bits: Cardinal absolute Value;
  Sign: Cardinal;
  Exp: Integer;       // Integer to avoid underflow
  Mant: Cardinal;
  Low13: Cardinal;    // bits to be discarded
  RoundUp: Boolean;
begin
  Sign := Bits shr 31;
  Exp := (Bits shr 23) and $FF;
  Mant := Bits and $7FFFFF;

  // ---- Zero or denormal (flush to zero, preserving sign) ----
  if Exp = 0 then
  begin
    Result := Sign shl 15;      // +0 or -0
    Exit;
  end;

  // ---- Infinity or NaN ----
  if Exp = $FF then
  begin
    Result := (Sign shl 15) or $7C00;
    if Mant <> 0 then
      Result := Result or (Mant shr 13);   // preserve NaN payload
    Exit;
  end;

  // ---- Normal number: adjust exponent bias ----
  Exp := Exp - 127 + 15;        // now Exp in range -14..31

  // ---- Overflow to infinity ----
  if Exp >= 31 then
  begin
    Result := (Sign shl 15) or $7C00;
    Exit;
  end;

  // ---- Underflow to zero (flush to zero, preserving sign) ----
  if Exp <= 0 then
  begin
    Result := Sign shl 15;
    Exit;
  end;

  // ---- Round-to-nearest-even on the low 13 bits ----
  Low13 := Mant and $1FFF;      // lower 13 bits (0x1FFF = 8191)

  // Determine if we need to round up
  if Low13 > $1000 then
    RoundUp := True
  else if Low13 < $1000 then
    RoundUp := False
  else
    // Exactly halfway: round to even (LSB of result mantissa should be 0)
    RoundUp := (Mant and $2000) <> 0;   // if bit 13 (0-indexed) is 1, round up

  // Apply rounding
  if RoundUp then
  begin
    Mant := Mant + $2000;
    // Check for carry into exponent
    if Mant >= $800000 then
    begin
      Mant := Mant and $7FFFFF;
      Inc(Exp);
      if Exp >= 31 then
      begin
        Result := (Sign shl 15) or $7C00;   // overflow to infinity
        Exit;
      end;
    end;
  end;

  // ---- Pack result ----
  Result := (Sign shl 15) or (Cardinal(Exp) shl 10) or (Mant shr 13);
end;

function TTestAVX.HalfToSingle(Value: Word): Single;
var
  Bits: Cardinal;
  Sign, Mant: Cardinal;
  Exp, shift: Integer;
begin
  Sign := (Value shr 15) and 1;
  Exp := (Value shr 10) and $1F;
  Mant := Value and $3FF;

  if Exp = 0 then
  begin
    // Zero or denormal
    if Mant = 0 then
      Bits := Sign shl 31
    else
    begin
      // Denormal: normalize by shifting mantissa left until bit 10 is set
      shift := 0;
      while (Mant and $400) = 0 do
      begin
        Mant := Mant shl 1;
        Inc(shift);
      end;
      // Remove the implicit leading 1 (bit 10)
      Mant := Mant and $3FF;
      // Single precision exponent = 127 - 14 - shift = 113 - shift
      // Since shift can be up to 9, exponent is positive
      Bits := (Sign shl 31) or (Cardinal(113 - shift) shl 23) or (Mant shl 13);
    end;
  end
  else if Exp = $1F then
  begin
    // Infinity or NaN
    Bits := (Sign shl 31) or $7F800000;
    if Mant <> 0 then
      Bits := Bits or (Mant shl 13);
  end
  else
  begin
    // Normal number: single exponent = half_exp - 15 + 127 = half_exp + 112
    Bits := (Sign shl 31) or (Cardinal(Exp + 112) shl 23) or (Mant shl 13);
  end;

  // Interpret the constructed bit pattern as a Single
  Result := PSingle(@Bits)^;
end;

{------------------------------------------------------------------------------
  Bfloat16 conversion helpers (truncated IEEE-754 single)
-----------------------------------------------------------------------------}

function TTestAVX.SingleToBFloat16(Value: Single): Word;
var
  Bits: Cardinal absolute Value;
begin
  // Round-to-nearest-even on the low 16 bits.
  // Simple version: just take the high 16 bits and round based on the low bits.
  // For the purpose of testing we can use a straightforward conversion.
  // We use the same algorithm as in the AVX implementation: add half-ulp and shift.
  // This is bit-exact to the hardware method if MXCSR is set correctly.
  // To keep it simple, we just truncate (which may produce slightly different results).
  // For reliable testing, we rely on the fact that the AVX implementation uses
  // round-to-nearest-even and our scalar helper must match exactly.
  // We'll implement the official round-to-nearest-even algorithm:
  // Add 0x7FFF + (high_word & 1) to the full bits, then shift right 16.
  // This matches the integer arithmetic used by the vectorized code.
  Result := (Bits + $7FFF + ((Bits shr 16) and 1)) shr 16;
  // Additionally, handle NaN by quieting
  if (Bits and $7F800000) = $7F800000 then
    if (Bits and $007FFFFF) <> 0 then
      Result := Result or $0040;  // set quiet bit
end;

function TTestAVX.BFloat16ToSingle(Value: Word): Single;
var
  Bits: Cardinal;
begin
  // Expand 16-bit bfloat16 to 32-bit single by shifting left 16 bits.
  Bits := Value shl 16;
  // If the half is NaN, we need to ensure the single becomes a quiet NaN.
  // The standard bfloat16 format is just the high 16 bits of the single.
  Result := PSingle(@Bits)^;
end;

{------------------------------------------------------------------------------
  Test methods
-----------------------------------------------------------------------------}

procedure TTestAVX.TestFillMem;
const
  N = 257;
var
  Dst, Ref: array[0..N-1] of Single;
  FillVal: Single;
begin
  FillVal := 3.14159;
  FillWithConst(Ref, FillVal);
  _AVXFillMem(@Dst[0], FillVal, N);
  Check(CompareSingleArrays(Dst, Ref, N), 'FillMem failed');
end;

procedure TTestAVX.TestCopyRelu;
const
  N = 257;
var
  Src, Dst, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  RandomSingleArray(Src);
  // Reference scalar implementation
  for i := 0 to N-1 do
    if Src[i] > 0 then Ref[i] := Src[i] else Ref[i] := 0;
  _AVXCopyRelu(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'CopyRelu failed');
end;

procedure TTestAVX.TestMulAdd;
const
  N = 257;
var
  Dst, Src, Z, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  RandomSingleArray(Src);
  RandomSingleArray(Z);
  FillWithSeq(Dst, 1.0, 0.1);
  for i := 0 to N-1 do Ref[i] := Dst[i] + Src[i] * Z[i];
  _AVXMulAdd(@Dst[0], @Src[0], @Z[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'MulAdd failed');
end;

procedure TTestAVX.TestMulAddF;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  fact: Single;
  i: Integer;
begin
  fact := 2.718;
  RandomSingleArray(Src);
  FillWithConst(Dst, 1.0);
  for i := 0 to N-1 do Ref[i] := Dst[i] + Src[i] * fact;
  _AVXMulAddF(@Dst[0], @Src[0], N, fact);
  Check(CompareSingleArrays(Dst, Ref, N), 'MulAddF failed');
end;

procedure TTestAVX.TestMulMulAdd;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  m1, m2: Single;
  i: Integer;
begin
  m1 := 1.5; m2 := 2.5;
  RandomSingleArray(Src);
  FillWithConst(Dst, 0.5);
  for i := 0 to N-1 do
    Ref[i] := Dst[i] * m1 + Src[i] * m2;
  _AVXMulMulAdd(@Dst[0], @Src[0], N, m1, m2);
  Check(CompareSingleArrays(Dst, Ref, N), 'MulMulAdd failed');
end;

procedure TTestAVX.TestMulF;
const
  N = 257;
var
  Dst, Ref: array[0..N-1] of Single;
  factor: Single;
  i: Integer;
begin
  factor := 0.5;
  FillWithSeq(Dst, 1.0, 0.1);
  for i := 0 to N-1 do Ref[i] := Dst[i] * factor;
  _AVXMulF(@Dst[0], N, factor);
  Check(CompareSingleArrays(Dst, Ref, N), 'MulF failed');
end;

procedure TTestAVX.TestMul;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1);
  FillWithSeq(Src, 2.0, 0.2);
  for i := 0 to N-1 do Ref[i] := Dst[i] * Src[i];
  _AVXMul(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'Mul failed');
end;

procedure TTestAVX.TestAdd;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1);
  FillWithSeq(Src, 2.0, 0.2);
  for i := 0 to N-1 do Ref[i] := Dst[i] + Src[i];
  _AVXAdd(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'Add failed');
end;

procedure TTestAVX.TestMax;
const
  N = 256;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1);
  FillWithSeq(Src, 2.0, 0.2);
  for i := 0 to N-1 do Ref[i] := Max(Dst[i], Src[i]);
  _AVXMax(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'Max failed');
end;

procedure TTestAVX.TestSumDiff;
const
  N = 300;
var
  Dst, Src: array[0..N-1] of Single;
  RefSum, AVXSum: Single;
  LCount: Integer;
  i: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1);
  FillWithSeq(Src, 2.0, 0.2);

  LCount := 256;
  RefSum := 0;
  for i := 0 to LCount-1 do
    RefSum := RefSum + Abs(Dst[i] - Src[i]);
  AVXSum := _AVX2SumDiff(@Dst[0], @Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-3, Format('SumDiff %d failed', [LCount]));

  LCount := 257;
  RefSum := 0;
  for i := 0 to LCount-1 do
    RefSum := RefSum + Abs(Dst[i] - Src[i]);
  AVXSum := _AVX2SumDiff(@Dst[0], @Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-3, Format('SumDiff %d failed', [LCount]));
end;

procedure TTestAVX.TestDistanceSqr;
const
  N = 300;
var
  Dst, Src: array[0..N-1] of Single;
  RefSum, AVXSum: Single;
  i: Integer;
  LCount: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1);
  FillWithSeq(Src, 2.0, 0.2);

  LCount := 256;
  RefSum := 0;
  for i := 0 to LCount-1 do RefSum := RefSum + Sqr(Dst[i] - Src[i]);
  AVXSum := _AVXDistanceSqr(@Dst[0], @Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-2, Format('DistanceSqr %d failed', [LCount]));


  LCount := 257;
  RefSum := 0;
  for i := 0 to LCount-1 do RefSum := RefSum + Sqr(Dst[i] - Src[i]);
  AVXSum := _AVXDistanceSqr(@Dst[0], @Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-2, Format('DistanceSqr %d failed', [LCount]));

  LCount := 259;
  RefSum := 0;
  for i := 0 to LCount-1 do RefSum := RefSum + Sqr(Dst[i] - Src[i]);
  AVXSum := _AVXDistanceSqr(@Dst[0], @Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-2, Format('DistanceSqr %d failed', [LCount]));
end;

procedure TTestAVX.TestSub;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1);
  FillWithSeq(Src, 2.0, 0.2);
  for i := 0 to N-1 do Ref[i] := Dst[i] - Src[i];
  _AVXSub(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'Sub failed');
end;

procedure TTestAVX.TestGetSum;
const
  N = 300;
var
  Src: array[0..N-1] of Single;
  RefSum, AVXSum: Single;
  i: Integer;
  LCount: Integer;
begin
  FillWithSeq(Src, 1.0, 0.1);

  LCount := 256;
  RefSum := 0;
  for i := 0 to LCount-1 do RefSum := RefSum + Src[i];
  AVXSum := _AVXGetSum(@Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-3, Format('GetSum %d failed', [LCount]));

  LCount := 259;
  RefSum := 0;
  for i := 0 to LCount-1 do RefSum := RefSum + Src[i];
  AVXSum := _AVXGetSum(@Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-3, Format('GetSum %d failed', [LCount]));
end;

procedure TTestAVX.TestGetSumSqr;
const
  N = 300;
var
  Src: array[0..N-1] of Single;
  RefSum, AVXSum: Single;
  i: Integer;
  LCount: Integer;
begin
  FillWithSeq(Src, 1.0, 0.1);

  LCount := 256;
  RefSum := 0;
  for i := 0 to LCount-1 do RefSum := RefSum + Src[i] * Src[i];
  AVXSum := _AVXGetSumSqr(@Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-2, Format('GetSumSqr %d failed', [LCount]));

  LCount := 259;
  RefSum := 0;
  for i := 0 to LCount-1 do RefSum := RefSum + Src[i] * Src[i];
  AVXSum := _AVXGetSumSqr(@Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-2, Format('GetSumSqr %d failed', [LCount]));
end;

procedure TTestAVX.TestExp;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Src, -10.0, 0.1);
  for i := 0 to N-1 do Ref[i] := Exp(Src[i]);
  _AVXExp(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N, 1e-5), 'Exp failed');
end;

procedure TTestAVX.TestDotProd;
const
  N = 257;
var
  Dst, Src: array[0..N-1] of Single;
  RefSum, AVXSum: Single;
  i: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1);
  FillWithSeq(Src, 2.0, 0.2);
  RefSum := 0;
  for i := 0 to N-1 do RefSum := RefSum + Dst[i] * Src[i];
  AVXSum := _AVXDotProd(@Dst[0], @Src[0], N);
  Check(Abs(RefSum - AVXSum) < 1e-2, 'DotProd failed');
end;

procedure TTestAVX.TestDotProdInt8;
const
  N = 257;
var
  Dst: array[0..N-1] of ShortInt;
  Src: array[0..N-1] of Single;
  RefSum, AVXSum: Single;
  i: Integer;
begin
  for i := 0 to N-1 do Dst[i] := Random(256) - 128;
  for i := 0 to N-1 do Src[i] := (Random(20000) - 10000) / 1000;
  RefSum := 0;
  for i := 0 to N-1 do RefSum := RefSum + Dst[i] * Src[i];
  AVXSum := _AVXDotProdInt8(@Dst[0], @Src[0], N);
  Check(Abs(RefSum - AVXSum) < 1e-2, 'DotProdInt8 failed');
end;

procedure TTestAVX.TestMulAddInt8Scalar;
const
  N = 257;
var
  Dst, Ref: array[0..N-1] of Single;
  Codes: array[0..N-1] of ShortInt;
  W: Single;
  i: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1);
  for i := 0 to N-1 do Codes[i] := Random(256) - 128;
  W := 0.75;
  for i := 0 to N-1 do Ref[i] := Dst[i] + Codes[i] * W;
  _AVXMulAddInt8Scalar(@Dst[0], @Codes[0], W, N);
  Check(CompareSingleArrays(Dst, Ref, N, 1e-2), 'MulAddInt8Scalar failed');
end;

procedure TTestAVX.TestMulAddInt8;
const
  N = 300;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  Codes: array[0..N-1] of ShortInt;
  i: Integer;
  LCount: Integer;
begin
  FillWithSeq(Src, 2.0, 0.2);
  for i := 0 to N-1 do Codes[i] := Random(256) - 128;

  // Test 256
  LCount := 256;
  FillWithSeq(Dst, 1.0, 0.1);
  for i := 0 to LCount-1 do Ref[i] := Dst[i] + Src[i] * Codes[i];
  _AVXMulAddInt8(@Dst[0], @Src[0], @Codes[0], LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount, 1e-3), Format('MulAddInt8 %d failed', [LCount]));

  // Test 259 - reuse Dst but reinitialize for this test
  LCount := 259;
  FillWithSeq(Dst, 1.0, 0.1);
  for i := 0 to LCount-1 do Ref[i] := Dst[i] + Src[i] * Codes[i];
  _AVXMulAddInt8(@Dst[0], @Src[0], @Codes[0], LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount, 1e-3), Format('MulAddInt8 %d failed', [LCount]));
end;

procedure TTestAVX.TestMaxAbsFinite;
const
  N = 257;
var
  Src: array[0..N-1] of Single;
  Ref, AVX: Single;
  i: Integer;
begin
  FillWithSeq(Src, -10.0, 0.1);
  Ref := 0;
  for i := 0 to N-1 do
    if Abs(Src[i]) > Ref then Ref := Abs(Src[i]);
  AVX := _AVXMaxAbsFinite(@Src[0], N);
  Check(Abs(Ref - AVX) < 1e-5, 'MaxAbsFinite failed');
end;

procedure TTestAVX.TestQuantizeInt8;
const
  N = 257;
var
  Src: array[0..N-1] of Single;
  Dst, Ref: array[0..N-1] of ShortInt;
  MaxAbs: Single;
  i: Integer;
begin
  FillWithSeq(Src, -5.0, 0.04);
  MaxAbs := 5.0;
  for i := 0 to N-1 do
    Ref[i] := Round(EnsureRange(Src[i] / MaxAbs * 127, -127, 127));

  _AVXQuantizeInt8(@Dst[0], @Src[0], N, MaxAbs);
  Check(CompareShortIntArrays(Dst, Ref, N), 'QuantizeInt8 failed');
end;

procedure TTestAVX.TestDequantizeInt8;
const
  N = 257;
var
  Src: array[0..N-1] of ShortInt;
  Dst, Ref: array[0..N-1] of Single;
  Scale: Single;
  i: Integer;
begin
  for i := 0 to N-1 do Src[i] := Random(256) - 128;
  Scale := 0.1;
  for i := 0 to N-1 do Ref[i] := Src[i] * Scale;
  _AVXDequantizeInt8(@Dst[0], @Src[0], N, Scale);
  Check(CompareSingleArrays(Dst, Ref, N), 'DequantizeInt8 failed');
end;

procedure TTestAVX.TestDecodeBF16;
const
  N = 257;
var
  Src: array[0..N-1] of Word;
  Dst, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  for i := 0 to N-1 do Src[i] := Random(65536);
  for i := 0 to N-1 do Ref[i] := BFloat16ToSingle(Src[i]);
  _AVXDecodeBF16(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'DecodeBF16 failed');
end;

procedure TTestAVX.TestReluGateMask;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  //FillWithSeq(Src, -2.0, 0.02);     // generates values from -2.0 to 3.1
  FillWithSeq(Src, 3.2, -0.02);
  for i := 0 to N-1 do
    if Src[i] >= 0 then Ref[i] := 1.0
    else Ref[i] := 0.0;
  _AVXReluGateMask(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N, 1e-6), 'ReluGateMask failed');
end;

procedure TTestAVX.TestLeakyRelu;
const
  N = 256;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  Slope: Single;
  i: Integer;
begin
  Slope := 0.01;
  FillWithSeq(Src, -2.0, 0.02);
  for i := 0 to N-1 do
    if Src[i] >= 0 then Ref[i] := Src[i] else Ref[i] := Src[i] * Slope;
  _AVXLeakyRelu(@Dst[0], @Src[0], N, Slope);
  Check(CompareSingleArrays(Dst, Ref, N, 1e-6), 'LeakyRelu failed');
end;

procedure TTestAVX.TestDecodeF16;
const
  N = 257;
var
  Src: array[0..N-1] of Word;
  Dst, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  for i := 0 to N-1 do Src[i] := Random(65536);
  for i := 0 to N-1 do Ref[i] := HalfToSingle(Src[i]);
  _AVXDecodeF16(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'DecodeF16 failed');
end;

procedure TTestAVX.TestSumSqrCentered;
const
  N = 257;
var
  Src: array[0..N-1] of Single;
  Mean, Ref, AVX: Single;
  i: Integer;
begin
  FillWithSeq(Src, 1.0, 0.1);
  Mean := 5.0;
  Ref := 0;
  for i := 0 to N-1 do Ref := Ref + Sqr(Src[i] - Mean);
  AVX := _AVXSumSqrCentered(@Src[0], Mean, N);
  Check(Abs(Ref - AVX) < 1e-5, 'SumSqrCentered failed');
end;

procedure TTestAVX.TestAdamDelta;
const
  N = 257;
var
  Delta, M, V, RefDelta, RefM, RefV: array[0..N-1] of Single;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR: Single;
  i: Integer;
begin
  // Initialize parameters and inputs
  Beta1 := 0.9; OmBeta1 := 0.1; Beta2 := 0.999; OmBeta2 := 0.001;
  Epsilon := 1e-8; kLR := 0.001;
  // Simulate t=10 steps for bias correction
  InvOmB2D := 1.0 / (1 - Power(Beta2, 10));

  FillWithSeq(Delta, 1.0, 0.1);
  FillWithSeq(M, 0.0, 0.1);
  FillWithSeq(V, 0.0, 0.2);

  // Reference scalar implementation
  for i := 0 to N-1 do
  begin
    RefM[i] := Beta1 * M[i] + OmBeta1 * Delta[i];
    RefV[i] := Beta2 * V[i] + OmBeta2 * Delta[i] * Delta[i];
    RefDelta[i] := kLR * (RefM[i] / (Sqrt(RefV[i]) + Epsilon));
  end;

  _AVXAdamDelta(@Delta[0], @M[0], @V[0],
                Beta1, OmBeta1, Beta2, OmBeta2,
                InvOmB2D, Epsilon, kLR, N);

  Check(CompareSingleArrays(Delta, RefDelta, N) and
        CompareSingleArrays(M, RefM, N) and
        CompareSingleArrays(V, RefV, N), 'AdamDelta failed');
end;

procedure TTestAVX.TestAdafactorDelta;
const
  N = 257;
var
  Delta, V, RefDelta, RefV: array[0..N-1] of Single;
  Beta2, k, c, Epsilon: Single;
  i: Integer;
begin
  Beta2 := 0.999; k := 0.01; c := 0.001; Epsilon := 1e-8;
  FillWithSeq(Delta, 1.0, 0.1);
  FillWithSeq(V, 0.0, 0.2);

  for i := 0 to N-1 do
  begin
    RefV[i] := Beta2 * V[i] + (k * Delta[i] * Delta[i] + c);
    RefDelta[i] := k * Delta[i] / (Sqrt(RefV[i]) + Epsilon);
  end;

  _AVXAdafactorDelta(@Delta[0], @V[0], Beta2, k, c, Epsilon, N);
  Check(CompareSingleArrays(Delta, RefDelta, N) and
        CompareSingleArrays(V, RefV, N), 'AdafactorDelta failed');
end;

procedure TTestAVX.TestClampAbs;
const
  N = 257;
var
  A, Ref: array[0..N-1] of Single;
  Value: Single;
  i: Integer;
begin
  Value := 2.5;
  FillWithSeq(A, -5.0, 0.04);
  for i := 0 to N-1 do
    if Abs(A[i]) > Value then Ref[i] := Sign(A[i]) * Value else Ref[i] := A[i];
  _AVXClampAbs(@A[0], Value, N);
  Check(CompareSingleArrays(A, Ref, N), 'ClampAbs failed');
end;

procedure TTestAVX.TestLionDelta;
const
  N = 257;
var
  Delta, M, RefDelta, RefM: array[0..N-1] of Single;
  Beta1, k1, Beta2, k2, NegLR, PosLR: Single;
  c: Single;
  i: Integer;
begin
  Beta1 := 0.9; k1 := 0.1; Beta2 := 0.99; k2 := 0.1; NegLR := -0.01; PosLR := 0.01;
  FillWithSeq(Delta, 1.0, 0.1);
  FillWithSeq(M, 0.0, 0.2);

  for i := 0 to N-1 do
  begin
    c := Beta1 * M[i] + k1 * Delta[i];
    RefM[i] := Beta2 * M[i] + k2 * Delta[i];
    if c > 0 then RefDelta[i] := NegLR
    else if c < 0 then RefDelta[i] := PosLR
    else RefDelta[i] := 0;
  end;

  _AVXLionDelta(@Delta[0], @M[0], Beta1, k1, Beta2, k2, NegLR, PosLR, N);
  Check(CompareSingleArrays(Delta, RefDelta, N) and
        CompareSingleArrays(M, RefM, N), 'LionDelta failed');
end;

procedure TTestAVX.TestGetMaxPos;
const
  N = 257;
var
  A: array[0..N-1] of Single;
  RefVal, AVXVal: Single;
  RefPos, AVXPos: Integer;
  i: Integer;
begin
  FillWithSeq(A, -5.0, 0.04);
  RefVal := A[0]; RefPos := 0;
  for i := 1 to N-1 do
    if A[i] > RefVal then
    begin
      RefVal := A[i]; RefPos := i;
    end;
  AVXVal := _AVXGetMaxPos(@A[0], N, AVXPos);
  Check((Abs(RefVal - AVXVal) < 1e-5) and (RefPos = AVXPos), 'GetMaxPos failed');
end;

procedure TTestAVX.TestGetMinPos;
const
  N = 257;
var
  A: array[0..N-1] of Single;
  RefVal, AVXVal: Single;
  RefPos, AVXPos: Integer;
  i: Integer;
begin
  FillWithSeq(A, -5.0, 0.04);
  RefVal := A[0]; RefPos := 0;
  for i := 1 to N-1 do
    if A[i] < RefVal then
    begin
      RefVal := A[i]; RefPos := i;
    end;
  AVXVal := _AVXGetMinPos(@A[0], N, AVXPos);
  Check((Abs(RefVal - AVXVal) < 1e-5) and (RefPos = AVXPos), 'GetMinPos failed');
end;

procedure TTestAVX.TestGetMaxAbsPos;
const
  N = 257;
var
  A: array[0..N-1] of Single;
  RefVal, AVXVal: Single;
  RefPos, AVXPos: Integer;
  i: Integer;
begin
  FillWithSeq(A, -5.0, 0.04);
  RefVal := Abs(A[0]); RefPos := 0;
  for i := 1 to N-1 do
    if Abs(A[i]) > RefVal then
    begin
      RefVal := Abs(A[i]); RefPos := i;
    end;
  AVXVal := _AVXGetMaxAbsPos(@A[0], N, AVXPos);
  Check((Abs(RefVal - AVXVal) < 1e-5) and (RefPos = AVXPos), 'GetMaxAbsPos failed');
end;

procedure TTestAVX.TestAddScalar;
const
  N = 257;
var
  A, Ref: array[0..N-1] of Single;
  Value: Single;
  i: Integer;
begin
  Value := 3.14;
  FillWithSeq(A, 1.0, 0.1);
  for i := 0 to N-1 do Ref[i] := A[i] + Value;
  _AVXAddScalar(@A[0], Value, N);
  Check(CompareSingleArrays(A, Ref, N), 'AddScalar failed');
end;

procedure TTestAVX.TestExpShiftSum;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  Shift, RefSum, AVXSum: Single;
  i: Integer;
begin
  Shift := 1.0;
  FillWithSeq(Src, -5.0, 0.04);
  RefSum := 0;
  for i := 0 to N-1 do
  begin
    Ref[i] := Exp(Src[i] - Shift);
    RefSum := RefSum + Ref[i];
  end;
  AVXSum := _AVXExpShiftSum(@Dst[0], @Src[0], Shift, N);
  Check(CompareSingleArrays(Dst, Ref, N) and (Abs(RefSum - AVXSum) < 1e-3),
        'ExpShiftSum failed');
end;

procedure TTestAVX.TestLn;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Src, 0.1, 0.02);
  for i := 0 to N-1 do Ref[i] := Ln(Src[i]);
  _AVXLn(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N), 'Ln failed');
end;

procedure TTestAVX.TestSinCos;
const
  N = 257;
var
  Dst, Src, RefSin, RefCos: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Src, -2*Pi, 0.1);
  for i := 0 to N-1 do
  begin
    RefSin[i] := Sin(Src[i]);
    RefCos[i] := Cos(Src[i]);
  end;

  // Test sin
  _AVXSinCos(@Dst[0], @Src[0], N, 0);
  Check(CompareSingleArrays(Dst, RefSin, N), 'Sin failed');

  // Test cos
  _AVXSinCos(@Dst[0], @Src[0], N, 1);
  Check(CompareSingleArrays(Dst, RefCos, N), 'Cos failed');
end;

procedure TTestAVX.TestEncodeBF16;
const
  N = 257;
var
  Src: array[0..N-1] of Single;
  Dst, Ref: array[0..N-1] of Word;
  i: Integer;
begin
  FillWithSeq(Src, -10.0, 0.08);
  for i := 0 to N-1 do Ref[i] := SingleToBFloat16(Src[i]);
  _AVXEncodeBF16(@Dst[0], @Src[0], N);
  Check(CompareWordArrays(Dst, Ref, 5), 'EncodeBF16 failed');
end;

procedure TTestAVX.TestReluL;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  Low, High, Slope: Single;
  i: Integer;
begin
  Low := -0.5; High := 0.5; Slope := 0.01;
  FillWithSeq(Src, -1.0, 0.008);
  for i := 0 to N-1 do
  begin
    if Src[i] > High then
      Ref[i] := High + (Src[i] - High) * Slope
    else if Src[i] > Low then
      Ref[i] := Src[i]
    else
      Ref[i] := Low + (Src[i] - Low) * Slope;
  end;
  _AVXReluL(@Dst[0], @Src[0], Low, High, Slope, N);
  Check(CompareSingleArrays(Dst, Ref, N), 'ReluL failed');
end;

procedure TTestAVX.TestEncodeF16;
const
  N = 257;
var
  Src: array[0..N-1] of Single;
  Dst, Ref: array[0..N-1] of Word;
  i: Integer;
begin
  FillWithSeq(Src, -10.0, 0.08);
  for i := 0 to N-1 do Ref[i] := SingleToHalf(Src[i]);
  _AVXEncodeF16(@Dst[0], @Src[0], N);
  Check(CompareWordArrays(Dst, Ref, 5), 'EncodeF16 failed');
end;

procedure TTestAVX.TestReluLGateMask;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  Low, High, Slope: Single;
  i: Integer;
begin
  Low := -0.5; High := 0.5; Slope := 0.01;
  FillWithSeq(Src, -1.0, 0.008);
  for i := 0 to N-1 do
    if (Src[i] > Low) and not (Src[i] > High) then Ref[i] := 1.0
    else Ref[i] := Slope;
  _AVXReluLGateMask(@Dst[0], @Src[0], Low, High, Slope, N);
  Check(CompareSingleArrays(Dst, Ref, N, 1e-6), 'ReluLGateMask failed');
end;

initialization
  Randomize;
  RegisterTest(TTestAVX.Suite);
end.
