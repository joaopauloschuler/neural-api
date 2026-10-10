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
  TSingle8 = array[0..7] of Single;
  PSingle8 = ^TSingle8;

  TTestAVX = class(TTestCase)
  private
    // Helper comparison functions
    function CompareSingleArrays(const A, B: array of Single; ACount: Integer; AbsTol: Single = 1e-5; RelTol: Single = 1e-6): Boolean;
    function CompareWordArrays(const A, B: array of Word; ACount: Integer; Epsilon: Word = 0): Boolean;
    function CompareShortIntArrays(const A, B: array of ShortInt; ACount: Integer): Boolean;
    procedure RandomSingleArray(var Arr: array of Single);
    procedure FillWithConst(var Arr: array of Single; const Value: Single; N: Integer);
    procedure FillWithSeq(var Arr: array of Single; Start, Step: Single; N: Integer);

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
    procedure TestDotProductInt8Int8;
    procedure TestDotProductInt4Int8;
    procedure TestReluGrad;
    procedure TestSinCosBoth;
    //
    procedure TestMulAddF_ParamProbe;
    procedure TestMulAddF_BoundaryN;
    procedure TestMulMulAdd_ParamProbe;
    procedure TestMulF_ParamProbe;
    procedure TestSumSqrCentered_ParamProbe;
    procedure TestClampAbs_ParamProbe;
    procedure TestAddScalar_ParamProbe;
    procedure TestExpShiftSum_ParamProbe;
    procedure TestAdamDelta_ParamProbe;
    procedure TestAdafactorDelta_ParamProbe;
    procedure TestLionDelta_ParamProbe;
    procedure TestReluL_ParamProbe;
    procedure TestReluLGateMask_ParamProbe;
    procedure TestDotProductInt4Int8_ParamProbe;
    procedure TestMulAddInt8Scalar_ParamProbe;
    procedure TestUnalignedBuffers;
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
    if IsNan(A[i]) or IsNan(B[i]) then
    begin
      if not (IsNan(A[i]) and IsNan(B[i])) then
        Exit(False);
      Continue;
    end;
    if IsInfinite(A[i]) or IsInfinite(B[i]) then
    begin
      if (A[i] <> B[i]) then
        Exit(False);
      Continue;
    end;

    Diff := Abs(A[i] - B[i]);
    MaxVal := Max(Abs(A[i]), Abs(B[i]));
    if Diff > Max(AbsTol, MaxVal * RelTol) then
      Exit(False);
  end;
  Result := True;
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

procedure TTestAVX.FillWithConst(var Arr: array of Single; const Value: Single; N: Integer);
var
  i: Integer;
begin
  for i := 0 to N - 1 do
    Arr[i] := Value;
end;

procedure TTestAVX.FillWithSeq(var Arr: array of Single; Start, Step: Single; N: Integer);
var
  i: Integer;
begin
  for i := 0 to N - 1 do
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
  N = 300;
var
  Dst, Ref: array[0..N-1] of Single;
  FillVal: Single;
  LCount: Integer;
begin
  FillVal := 3.14159;

  LCount := 256;
  FillWithConst(Ref, FillVal, LCount);
  _AVXFillMem(@Dst[0], FillVal, LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount), Format('FillMem %d failed', [LCount]));

  LCount := 257;
  FillWithConst(Ref, FillVal + 1, LCount);
  _AVXFillMem(@Dst[0], FillVal + 1, LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount), Format('FillMem %d failed', [LCount]));
end;

procedure TTestAVX.TestCopyRelu;
const
  N = 300;
var
  Src, Dst, Ref: array[0..N-1] of Single;
  i: Integer;
  LCount: Integer;
begin
  LCount := 256;
  RandomSingleArray(Src);
  for i := 0 to LCount-1 do
    if Src[i] > 0 then Ref[i] := Src[i] else Ref[i] := 0;
  _AVXCopyRelu(@Dst[0], @Src[0], LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount), Format('CopyRelu %d failed', [LCount]));

  LCount := 257;
  RandomSingleArray(Src);
  for i := 0 to LCount-1 do
    if Src[i] > 0 then Ref[i] := Src[i] else Ref[i] := 0;
  _AVXCopyRelu(@Dst[0], @Src[0], LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount), Format('CopyRelu %d failed', [LCount]));
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
  FillWithSeq(Dst, 1.0, 0.1, N);
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
  FillWithConst(Dst, 1.0, N);
  for i := 0 to N-1 do Ref[i] := Dst[i] + Src[i] * fact;
  _AVXMulAddF(@Dst[0], @Src[0], N, fact);
  Check(CompareSingleArrays(Dst, Ref, N), 'MulAddF failed');
end;

procedure TTestAVX.TestMulAddF_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;                       // Not a multiple of 8, exercises the tail
var
  Dst, Src, Ref: array[0..N+15] of Single;
  fact: Single;
  i: Integer;
begin
  fact := -12345.678;
  for i := 0 to N+15 do
  begin
    Src[i] := (i + 1) * 1.5;
    Dst[i] := 1000.0 + i;
    Ref[i] := Dst[i] + Src[i] * fact;
  end;

  for i := 0 to 7 do
  begin
    Dst[i] := CANARY; Dst[N+8+i] := CANARY;
    Src[i] := CANARY; Src[N+8+i] := CANARY;
  end;

  _AVXMulAddF(@Dst[8], @Src[8], N, fact);

  for i := 0 to N-1 do
    Check(Abs(Dst[8+i] - Ref[8+i]) < 1e-3,
      Format('MulAddF param probe: idx %d expected %.3f actual %.3f',
             [i, Ref[8+i], Dst[8+i]]));

  for i := 0 to 7 do
  begin
    Check(Dst[i] = CANARY, 'MulAddF out-of-bounds write before dst');
    Check(Dst[N+8+i] = CANARY, 'MulAddF out-of-bounds write after dst');
    Check(Src[i] = CANARY, 'MulAddF out-of-bounds write before src');
    Check(Src[N+8+i] = CANARY, 'MulAddF out-of-bounds write after src');
  end;
end;

procedure TTestAVX.TestMulAddF_BoundaryN;
const
  Sizes: array[0..20] of Integer =
    (0,1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,63,64,65,127,257);
var
  Dst, Src, Ref: array[0..300] of Single;
  fact: Single;
  k, i, N: Integer;
begin
  fact := 1.2345;
  for k := 0 to High(Sizes) do
  begin
    N := Sizes[k];
    if N = 0 then Continue;
    for i := 0 to 300 do
    begin
      Src[i] := i * 0.7 - 5.0;
      Dst[i] := i * 0.3 + 1.0;
      Ref[i] := Dst[i] + Src[i] * fact;
    end;
    _AVXMulAddF(@Dst[0], @Src[0], N, fact);
    for i := 0 to N-1 do
      Check(Abs(Dst[i] - Ref[i]) < 1e-3,
        Format('N=%d idx=%d Dst=%.6f Ref=%.6f diff=%.6g',
               [N, i, Dst[i], Ref[i], Dst[i] - Ref[i]]));
  end;
end;

procedure TTestAVX.TestMulMulAdd;
const
  N = 300;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  m1, m2: Single;
  i: Integer;
  LCount: Integer;
begin
  m1 := 1.5; m2 := 2.5;
  RandomSingleArray(Src);
  FillWithConst(Dst, 0.5, N);

  LCount := 256;
  for i := 0 to LCount-1 do
    Ref[i] := Dst[i] * m1 + Src[i] * m2;
  _AVXMulMulAdd(@Dst[0], @Src[0], LCount, m1, m2);
  Check(CompareSingleArrays(Dst, Ref, LCount), Format('MulMulAdd %d failed', [LCount]));

  LCount := 257;
  for i := 0 to LCount-1 do
    Ref[i] := Dst[i] * m1 + Src[i] * m2;
  _AVXMulMulAdd(@Dst[0], @Src[0], LCount, m1, m2);
  Check(CompareSingleArrays(Dst, Ref, LCount), Format('MulMulAdd %d failed', [LCount]));
end;

procedure TTestAVX.TestMulMulAdd_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Dst, Src, Ref: array[0..N+15] of Single;
  m1, m2: Single;
  i: Integer;
begin
  // The two multipliers have extremely different magnitudes, so any
  // argument mix-up will be immediately exposed.
  m1 := 1e-8;
  m2 := 1e+8;
  for i := 0 to N+15 do
  begin
    Src[i] := (i + 1) * 2.0;
    Dst[i] := 100.0 + i;
    Ref[i] := Dst[i] * m1 + Src[i] * m2;
  end;
  for i := 0 to 7 do
  begin
    Dst[i] := CANARY; Dst[N+8+i] := CANARY;
    Src[i] := CANARY; Src[N+8+i] := CANARY;
  end;

  _AVXMulMulAdd(@Dst[8], @Src[8], N, m1, m2);

  for i := 0 to N-1 do
    Check(Abs(Dst[8+i] - Ref[8+i]) < 1e-2,
      Format('MulMulAdd param probe: idx %d expected %.4e actual %.4e',
             [i, Ref[8+i], Dst[8+i]]));
  for i := 0 to 7 do
  begin
    Check(Dst[i] = CANARY, 'MulMulAdd out-of-bounds write before dst');
    Check(Dst[N+8+i] = CANARY, 'MulMulAdd out-of-bounds write after dst');
    Check(Src[i] = CANARY, 'MulMulAdd out-of-bounds write before src');
    Check(Src[N+8+i] = CANARY, 'MulMulAdd out-of-bounds write after src');
  end;
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
  FillWithSeq(Dst, 1.0, 0.1, N);
  for i := 0 to N-1 do Ref[i] := Dst[i] * factor;
  _AVXMulF(@Dst[0], N, factor);
  Check(CompareSingleArrays(Dst, Ref, N), 'MulF failed');
end;

procedure TTestAVX.TestMulF_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Dst, Ref: array[0..N+15] of Single;
  factor: Single;
  i: Integer;
begin
  factor := -9876.543;   // Extreme value
  for i := 0 to N+15 do
  begin
    Dst[i] := 1.0 + i * 0.25;
    Ref[i] := Dst[i] * factor;
  end;
  for i := 0 to 7 do
  begin
    Dst[i] := CANARY; Dst[N+8+i] := CANARY;
  end;

  _AVXMulF(@Dst[8], N, factor);

  for i := 0 to N-1 do
    Check(Abs(Dst[8+i] - Ref[8+i]) < 1e-2,
      Format('MulF param probe: idx %d expected %.3f actual %.3f',
             [i, Ref[8+i], Dst[8+i]]));
  for i := 0 to 7 do
  begin
    Check(Dst[i] = CANARY, 'MulF out-of-bounds write before canary');
    Check(Dst[N+8+i] = CANARY, 'MulF out-of-bounds write after canary');
  end;
end;

procedure TTestAVX.TestMul;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Dst, 1.0, 0.1, N);
  FillWithSeq(Src, 2.0, 0.2, N);
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
  FillWithSeq(Dst, 1.0, 0.1, N);
  FillWithSeq(Src, 2.0, 0.2, N);
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
  FillWithSeq(Dst, 1.0, 0.1, N);
  FillWithSeq(Src, 2.0, 0.2, N);
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
  FillWithSeq(Dst, 1.0, 0.1, N);
  FillWithSeq(Src, 2.0, 0.2, N);

  LCount := 256;
  RefSum := 0;
  for i := 0 to LCount-1 do
    RefSum := RefSum + Abs(Dst[i] - Src[i]);
  AVXSum := _AVXSumDiff(@Dst[0], @Src[0], LCount);
  Check(Abs(RefSum - AVXSum) < 1e-3, Format('SumDiff %d failed', [LCount]));

  LCount := 257;
  RefSum := 0;
  for i := 0 to LCount-1 do
    RefSum := RefSum + Abs(Dst[i] - Src[i]);
  AVXSum := _AVXSumDiff(@Dst[0], @Src[0], LCount);
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
  FillWithSeq(Dst, 1.0, 0.1, N);
  FillWithSeq(Src, 2.0, 0.2, N);

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
  FillWithSeq(Dst, 1.0, 0.1, N);
  FillWithSeq(Src, 2.0, 0.2, N);
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
  FillWithSeq(Src, 1.0, 0.1, N);

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
  FillWithSeq(Src, 1.0, 0.1, N);

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
  FillWithSeq(Src, -10.0, 0.1, N);
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
  FillWithSeq(Dst, 1.0, 0.1, N);
  FillWithSeq(Src, 2.0, 0.2, N);
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

procedure TTestAVX.TestDotProductInt4Int8;
const
  MAX_BLOCKS = 10;
var
  BytePacked: array[0..(MAX_BLOCKS * 16) - 1] of Byte;
  B: array[0..(MAX_BLOCKS * 32) - 1] of ShortInt;
  Scales: array[0..MAX_BLOCKS - 1] of Single;
  BlockSum8: array[0..MAX_BLOCKS - 1] of Single;
  LCount, i: Integer;
  Expected, Actual: Single;
begin
  for LCount := 1 to MAX_BLOCKS do
  begin
    // Each packed byte holds low nibble = 1 and high nibble = 1
    FillChar(BytePacked, SizeOf(BytePacked), $11);
    // All Int8 inputs = 1 (never negative)
    FillChar(B, SizeOf(B), 1);
    for i := 0 to LCount - 1 do
    begin
      Scales[i] := 1.0;
      BlockSum8[i] := 0.0;
    end;

    // 32 elements per block, each contributing 1 * 1 * scale = 1
    Expected := 32.0 * LCount;

    Actual := _AVXDotProductInt4Int8(
      @BytePacked[0], @Scales[0], @B[0], @BlockSum8[0], LCount);

    Check(Abs(Actual - Expected) < 1e-3,
      Format('DotProductInt4Int8 %d blocks: expected %.4f, got %.4f',
             [LCount, Expected, Actual]));
  end;
end;

procedure TTestAVX.TestDotProductInt8Int8;
const
  N = 300;
  TestSizes: array[0..8] of Integer = (0, 1, 7, 31, 32, 33, 100, 256, 257);
var
  A, B: array[0..N-1] of ShortInt;
  Expected, Actual: Integer;
  LCount, i, k: Integer;
begin
  for k := 0 to High(TestSizes) do
  begin
    LCount := TestSizes[k];
    if LCount > N then LCount := N;

    // Fill with values in [-127, 127] to avoid the -128 abs overflow case
    for i := 0 to LCount - 1 do
    begin
      A[i] := ShortInt(Random(255) - 127);
      B[i] := ShortInt(Random(255) - 127);
    end;

    Expected := 0;
    for i := 0 to LCount - 1 do
      Inc(Expected, Integer(A[i]) * Integer(B[i]));

    Actual := _AVXDotProductInt8Int8(@A[0], @B[0], LCount);
    Check(Expected = Actual,
      Format('DotProductInt8Int8 %d: expected %d, got %d',
             [LCount, Expected, Actual]));
  end;
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
  FillWithSeq(Dst, 1.0, 0.1, N);
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
  FillWithSeq(Src, 2.0, 0.2, N);
  for i := 0 to N-1 do Codes[i] := Random(256) - 128;

  // Test 256
  LCount := 256;
  FillWithSeq(Dst, 1.0, 0.1, N);
  for i := 0 to LCount-1 do Ref[i] := Dst[i] + Src[i] * Codes[i];
  _AVXMulAddInt8(@Dst[0], @Src[0], @Codes[0], LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount, 1e-3), Format('MulAddInt8 %d failed', [LCount]));

  // Test 259 - reuse Dst but reinitialize for this test
  LCount := 259;
  FillWithSeq(Dst, 1.0, 0.1, N);
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
  FillWithSeq(Src, -10.0, 0.1, N);
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
  FillWithSeq(Src, -5.0, 0.04, N);
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
  FillWithSeq(Src, 3.2, -0.02, N);
  for i := 0 to N-1 do
    if Src[i] >= 0 then Ref[i] := 1.0
    else Ref[i] := 0.0;
  _AVXReluGateMask(@Dst[0], @Src[0], N);
  Check(CompareSingleArrays(Dst, Ref, N, 1e-6), 'ReluGateMask failed');
end;

procedure TTestAVX.TestReluGrad;
const
  N = 300;
  TestSizes: array[0..7] of Integer = (0, 1, 7, 8, 9, 32, 100, 257);
var
  Dst, Err, Raw, Ref: array[0..N-1] of Single;
  LCount, i, k: Integer;
begin
  for k := 0 to High(TestSizes) do
  begin
    LCount := TestSizes[k];
    if LCount > N then LCount := N;

    // Mixed positive / negative / zero / NaN / Inf values
    for i := 0 to LCount - 1 do
    begin
      case i mod 8 of
        0: Raw[i] :=  1.5;
        1: Raw[i] := -1.5;
        2: Raw[i] :=  0.0;
        3: Raw[i] := -0.0;
        4: Raw[i] :=  NaN;
        5: Raw[i] :=  Infinity;
        6: Raw[i] := -Infinity;
        7: Raw[i] :=  (Random(2000) - 1000) / 100.0;
      end;
      Err[i] := (Random(2000) - 1000) / 100.0;
    end;

    // Reference: exactly the scalar test
    for i := 0 to LCount - 1 do
      if Raw[i] > 0 then Ref[i] := Err[i]
      else Ref[i] := 0.0;

    _AVXReluGrad(@Dst[0], @Err[0], @Raw[0], LCount);

    Check(CompareSingleArrays(Dst, Ref, LCount),
      Format('ReluGrad %d failed', [LCount]));
  end;
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
  FillWithSeq(Src, -2.0, 0.02, N);
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
  FillWithSeq(Src, 1.0, 0.1, N);
  Mean := 5.0;
  Ref := 0;
  for i := 0 to N-1 do Ref := Ref + Sqr(Src[i] - Mean);
  AVX := _AVXSumSqrCentered(@Src[0], Mean, N);
  Check(Abs(Ref - AVX) < 1e-5, 'SumSqrCentered failed');
end;

procedure TTestAVX.TestSumSqrCentered_ParamProbe;
const
  N = 35;
var
  Src: array[0..N-1] of Single;
  Mean, Ref, AVX: Single;
  i: Integer;
begin
  for i := 0 to N-1 do
    Src[i] := i * 0.37 - 3.0;

  Mean := 2.5;

  Ref := 0;
  for i := 0 to N-1 do
    Ref := Ref + Sqr(Src[i] - Mean);

  AVX := _AVXSumSqrCentered(@Src[0], Mean, N);

  Check(Abs(Ref - AVX) < 1e-3,
    Format('SumSqrCentered parameter probe: expected %.6f actual %.6f',
      [Ref, AVX]));
end;

procedure TTestAVX.TestClampAbs_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  A, Ref: array[0..N+15] of Single;
  Value: Single;
  i: Integer;
begin
  Value := 2.5;

  for i := 0 to N+15 do
  begin
    A[i] := (i - 20) * 0.4;

    if Abs(A[i]) > Value then
      Ref[i] := Sign(A[i]) * Value
    else
      Ref[i] := A[i];
  end;

  for i := 0 to 7 do
  begin
    A[i] := CANARY;
    A[N+8+i] := CANARY;
  end;

  _AVXClampAbs(@A[8], Value, N);

  for i := 0 to N-1 do
    Check(Abs(A[8+i] - Ref[8+i]) < 1e-5,
      Format('ClampAbs parameter probe: idx %d expected %.3f actual %.3f',
        [i, Ref[8+i], A[8+i]]));

  for i := 0 to 7 do
  begin
    Check(A[i] = CANARY, 'ClampAbs out-of-bounds write front canary');
    Check(A[N+8+i] = CANARY, 'ClampAbs out-of-bounds write rear canary');
  end;
end;
procedure TTestAVX.TestAddScalar_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  A, Ref: array[0..N+15] of Single;
  Value: Single;
  i: Integer;
begin
  Value := -777.7;
  for i := 0 to N+15 do
  begin
    A[i] := i * 0.5;
    Ref[i] := A[i] + Value;
  end;
  for i := 0 to 7 do
  begin
    A[i] := CANARY; A[N+8+i] := CANARY;
  end;

  _AVXAddScalar(@A[8], Value, N);

  for i := 0 to N-1 do
    Check(Abs(A[8+i] - Ref[8+i]) < 1e-3,
      Format('AddScalar parameter probe: idx %d expected %.3f actual %.3f', [i, Ref[8+i], A[8+i]]));
  for i := 0 to 7 do
  begin
    Check(A[i] = CANARY, 'AddScalar out-of-bounds write front canary');
    Check(A[N+8+i] = CANARY, 'AddScalar out-of-bounds write rear canary');
  end;
end;

procedure TTestAVX.TestExpShiftSum_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Dst, Src, Ref: array[0..N+15] of Single;
  Shift, RefSum, AVXSum: Single;
  i: Integer;
begin
  Shift := 3.75;                      // Very different from N
  for i := 0 to N+15 do
  begin
    Src[i] := (i - 20) * 0.3;
    Ref[i] := Exp(Src[i] - Shift);
  end;
  RefSum := 0;
  for i := 0 to N-1 do RefSum := RefSum + Ref[8+i];
  for i := 0 to 7 do
  begin
    Dst[i] := CANARY; Dst[N+8+i] := CANARY;
    Src[i] := CANARY; Src[N+8+i] := CANARY;
  end;

  AVXSum := _AVXExpShiftSum(@Dst[8], @Src[8], Shift, N);

  for i := 0 to N-1 do
    Check(Abs(Dst[8+i] - Ref[8+i]) < 1e-4,
      Format('ExpShiftSum parameter probe: idx %d expected %.6f actual %.6f', [i, Ref[8+i], Dst[8+i]]));
  Check(Abs(RefSum - AVXSum) < 1e-2,
    Format('ExpShiftSum return sum: expected %.6f actual %.6f', [RefSum, AVXSum]));
  for i := 0 to 7 do
  begin
    Check(Dst[i] = CANARY, 'ExpShiftSum out-of-bounds write dst front canary');
    Check(Dst[N+8+i] = CANARY, 'ExpShiftSum out-of-bounds write dst rear canary');
    Check(Src[i] = CANARY, 'ExpShiftSum out-of-bounds write src front canary');
    Check(Src[N+8+i] = CANARY, 'ExpShiftSum out-of-bounds write src rear canary');
  end;
end;

procedure TTestAVX.TestAdamDelta_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Delta, M, V, RefDelta, RefM, RefV: array[0..N+15] of Single;
  Beta1, OmBeta1, Beta2, OmBeta2, InvOmB2D, Epsilon, kLR: Single;
  i: Integer;
begin
  // Give each parameter a unique value; any misalignment will cause drastic deviation
  Beta1    := 0.111;
  OmBeta1  := 0.222;
  Beta2    := 0.333;
  OmBeta2  := 0.444;
  InvOmB2D := 1.555;
  Epsilon  := 1e-7;
  kLR      := 0.00777;

  for i := 0 to N+15 do
  begin
    Delta[i] := (i + 1) * 0.1;
    M[i]     := (i + 1) * 0.01;
    V[i]     := (i + 1) * 0.02;
    RefM[i]     := Beta1 * M[i] + OmBeta1 * Delta[i];
    RefV[i]     := Beta2 * V[i] + OmBeta2 * Delta[i] * Delta[i];
    RefDelta[i] := kLR * (RefM[i] / (Sqrt(RefV[i]) + Epsilon));
  end;
  for i := 0 to 7 do
  begin
    Delta[i] := CANARY; Delta[N+8+i] := CANARY;
    M[i]     := CANARY; M[N+8+i]     := CANARY;
    V[i]     := CANARY; V[N+8+i]     := CANARY;
  end;

  _AVXAdamDelta(@Delta[8], @M[8], @V[8],
                 Beta1, OmBeta1, Beta2, OmBeta2,
                 InvOmB2D, Epsilon, kLR, N);

  for i := 0 to N-1 do
  begin
    Check(Abs(Delta[8+i] - RefDelta[8+i]) < 1e-4,
      Format('AdamDelta Delta idx %d: expected %.6f actual %.6f', [i, RefDelta[8+i], Delta[8+i]]));
    Check(Abs(M[8+i] - RefM[8+i]) < 1e-4,
      Format('AdamDelta M idx %d: expected %.6f actual %.6f', [i, RefM[8+i], M[8+i]]));
    Check(Abs(V[8+i] - RefV[8+i]) < 1e-4,
      Format('AdamDelta V idx %d: expected %.6f actual %.6f', [i, RefV[8+i], V[8+i]]));
  end;
  for i := 0 to 7 do
  begin
    Check(Delta[i] = CANARY, 'AdamDelta out-of-bounds write Delta front canary');
    Check(Delta[N+8+i] = CANARY, 'AdamDelta out-of-bounds write Delta rear canary');
    Check(M[i] = CANARY, 'AdamDelta out-of-bounds write M front canary');
    Check(M[N+8+i] = CANARY, 'AdamDelta out-of-bounds write M rear canary');
    Check(V[i] = CANARY, 'AdamDelta out-of-bounds write V front canary');
    Check(V[N+8+i] = CANARY, 'AdamDelta out-of-bounds write V rear canary');
  end;
end;

procedure TTestAVX.TestAdafactorDelta_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Delta, V, RefDelta, RefV: array[0..N+15] of Single;
  Beta2, k, c, Epsilon: Single;
  i: Integer;
begin
  Beta2   := 0.777;
  k       := 0.123;
  c       := 0.000456;
  Epsilon := 1e-9;

  for i := 0 to N+15 do
  begin
    Delta[i] := (i + 1) * 0.11;
    V[i]     := (i + 1) * 0.02;
    RefV[i]     := Beta2 * V[i] + (k * Delta[i] * Delta[i] + c);
    RefDelta[i] := k * Delta[i] / (Sqrt(RefV[i]) + Epsilon);
  end;
  for i := 0 to 7 do
  begin
    Delta[i] := CANARY; Delta[N+8+i] := CANARY;
    V[i]     := CANARY; V[N+8+i]     := CANARY;
  end;

  _AVXAdafactorDelta(@Delta[8], @V[8], Beta2, k, c, Epsilon, N);

  for i := 0 to N-1 do
  begin
    Check(Abs(Delta[8+i] - RefDelta[8+i]) < 1e-4,
      Format('AdafactorDelta Delta idx %d failed', [i]));
    Check(Abs(V[8+i] - RefV[8+i]) < 1e-4,
      Format('AdafactorDelta V idx %d failed', [i]));
  end;
  for i := 0 to 7 do
  begin
    Check(Delta[i] = CANARY, 'AdafactorDelta out-of-bounds write Delta front canary');
    Check(Delta[N+8+i] = CANARY, 'AdafactorDelta out-of-bounds write Delta rear canary');
    Check(V[i] = CANARY, 'AdafactorDelta out-of-bounds write V front canary');
    Check(V[N+8+i] = CANARY, 'AdafactorDelta out-of-bounds write V rear canary');
  end;
end;

procedure TTestAVX.TestLionDelta_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Delta, M, RefDelta, RefM: array[0..N+15] of Single;
  Beta1, k1, Beta2, k2, NegLR, PosLR: Single;
  c: Single;
  i: Integer;
begin
  Beta1 := 0.111;
  k1    := 0.222;
  Beta2 := 0.333;
  k2    := 0.444;
  NegLR := -0.00555;
  PosLR :=  0.00666;

  for i := 0 to N+15 do
  begin
    Delta[i] := (i - 20) * 0.13;     // Both positive and negative to trigger both branches
    M[i]     := (i - 20) * 0.07;
    c := Beta1 * M[i] + k1 * Delta[i];
    RefM[i] := Beta2 * M[i] + k2 * Delta[i];
    if c > 0 then RefDelta[i] := NegLR
    else if c < 0 then RefDelta[i] := PosLR
    else RefDelta[i] := 0;
  end;
  for i := 0 to 7 do
  begin
    Delta[i] := CANARY; Delta[N+8+i] := CANARY;
    M[i]     := CANARY; M[N+8+i]     := CANARY;
  end;

  _AVXLionDelta(@Delta[8], @M[8],
                 Beta1, k1, Beta2, k2, NegLR, PosLR, N);

  for i := 0 to N-1 do
  begin
    Check(Abs(Delta[8+i] - RefDelta[8+i]) < 1e-5,
      Format('LionDelta Delta idx %d: expected %.5f actual %.5f',
             [i, RefDelta[8+i], Delta[8+i]]));
    Check(Abs(M[8+i] - RefM[8+i]) < 1e-5,
      Format('LionDelta M idx %d failed', [i]));
  end;
  for i := 0 to 7 do
  begin
    Check(Delta[i] = CANARY, 'LionDelta out-of-bounds write Delta front canary');
    Check(Delta[N+8+i] = CANARY, 'LionDelta out-of-bounds write Delta rear canary');
    Check(M[i] = CANARY, 'LionDelta out-of-bounds write M front canary');
    Check(M[N+8+i] = CANARY, 'LionDelta out-of-bounds write M rear canary');
  end;
end;

procedure TTestAVX.TestReluL_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Dst, Src, Ref: array[0..N+15] of Single;
  Low, High, Slope: Single;
  i: Integer;
begin
  Low   := -0.5;
  High  :=  0.7;
  Slope :=  0.03;
  for i := 0 to N+15 do
  begin
    Src[i] := (i - 20) * 0.15;       // Cover below Low, within range, above High
    if Src[i] > High then
      Ref[i] := High + (Src[i] - High) * Slope
    else if Src[i] > Low then
      Ref[i] := Src[i]
    else
      Ref[i] := Low + (Src[i] - Low) * Slope;
  end;
  for i := 0 to 7 do
  begin
    Dst[i] := CANARY; Dst[N+8+i] := CANARY;
    Src[i] := CANARY; Src[N+8+i] := CANARY;
  end;

  _AVXReluL(@Dst[8], @Src[8], Low, High, Slope, N);

  for i := 0 to N-1 do
    Check(Abs(Dst[8+i] - Ref[8+i]) < 1e-4,
      Format('ReluL parameter probe: idx %d expected %.5f actual %.5f', [i, Ref[8+i], Dst[8+i]]));
  for i := 0 to 7 do
  begin
    Check(Dst[i] = CANARY, 'ReluL out-of-bounds write dst front canary');
    Check(Dst[N+8+i] = CANARY, 'ReluL out-of-bounds write dst rear canary');
    Check(Src[i] = CANARY, 'ReluL out-of-bounds write src front canary');
    Check(Src[N+8+i] = CANARY, 'ReluL out-of-bounds write src rear canary');
  end;
end;

procedure TTestAVX.TestReluLGateMask_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Dst, Src, Ref: array[0..N+15] of Single;
  Low, High, Slope: Single;
  i: Integer;
begin
  Low   := -0.5;
  High  :=  0.7;
  Slope :=  0.123;
  for i := 0 to N+15 do
  begin
    Src[i] := (i - 20) * 0.15;
    if (Src[i] > Low) and not (Src[i] > High) then Ref[i] := 1.0
    else Ref[i] := Slope;
  end;
  for i := 0 to 7 do
  begin
    Dst[i] := CANARY; Dst[N+8+i] := CANARY;
    Src[i] := CANARY; Src[N+8+i] := CANARY;
  end;

  _AVXReluLGateMask(@Dst[8], @Src[8], Low, High, Slope, N);

  for i := 0 to N-1 do
    Check(Abs(Dst[8+i] - Ref[8+i]) < 1e-6,
      Format('ReluLGateMask parameter probe: idx %d expected %.4f actual %.4f', [i, Ref[8+i], Dst[8+i]]));
  for i := 0 to 7 do
  begin
    Check(Dst[i] = CANARY, 'ReluLGateMask out-of-bounds write dst front canary');
    Check(Dst[N+8+i] = CANARY, 'ReluLGateMask out-of-bounds write dst rear canary');
    Check(Src[i] = CANARY, 'ReluLGateMask out-of-bounds write src front canary');
    Check(Src[N+8+i] = CANARY, 'ReluLGateMask out-of-bounds write src rear canary');
  end;
end;

procedure TTestAVX.TestDotProductInt4Int8_ParamProbe;
const
  MAX_BLOCKS = 5;
var
  PackedArr: array[0..(MAX_BLOCKS*16)-1] of Byte;
  B: array[0..(MAX_BLOCKS*32)-1] of ShortInt;
  Scales, BlockSum8: array[0..MAX_BLOCKS-1] of Single;
  LCount, i: Integer;
  Ref, Actual: Single;
begin
  for LCount := 1 to MAX_BLOCKS do
  begin
    // Fill low/high nibble of each byte with 2, B all 1, scales differ
    FillChar(PackedArr, SizeOf(PackedArr), $22);
    FillChar(B, SizeOf(B), 1);
    for i := 0 to LCount-1 do
    begin
      Scales[i] := 1.0 + i * 0.5;
      BlockSum8[i] := 0.0;
    end;
    // Reference: 32 elements per block, each contributes 2 * 1 * scale
    Ref := 0;
    for i := 0 to LCount-1 do
      Ref := Ref + 32.0 * 2.0 * Scales[i];

    Actual := _AVXDotProductInt4Int8(
      @PackedArr[0], @Scales[0], @B[0], @BlockSum8[0], LCount);

    Check(Abs(Actual - Ref) < 1e-3,
      Format('DotProductInt4Int8 %d blocks: expected %.4f actual %.4f',
             [LCount, Ref, Actual]));
  end;
end;

procedure TTestAVX.TestMulAddInt8Scalar_ParamProbe;
const
  CANARY: Single = 1.2345678e30;
  N = 35;
var
  Dst, Ref: array[0..N+15] of Single;
  Codes: array[0..N+15] of ShortInt;
  W: Single;
  i: Integer;
begin
  W := -1357.9;                       // Extreme value
  for i := 0 to N+15 do
  begin
    Dst[i] := 1.0 + i * 0.01;
    Codes[i] := ShortInt((i mod 255) - 127);
    Ref[i] := Dst[i] + Codes[i] * W;
  end;
  for i := 0 to 7 do
  begin
    Dst[i] := CANARY; Dst[N+8+i] := CANARY;
    Codes[i] := 0; Codes[N+8+i] := 0;
  end;

  _AVXMulAddInt8Scalar(@Dst[8], @Codes[8], W, N);

  for i := 0 to N-1 do
    Check(Abs(Dst[8+i] - Ref[8+i]) < 1e-2,
      Format('MulAddInt8Scalar parameter probe: idx %d expected %.3f actual %.3f', [i, Ref[8+i], Dst[8+i]]));
  for i := 0 to 7 do
  begin
    Check(Dst[i] = CANARY, 'MulAddInt8Scalar out-of-bounds write front canary');
    Check(Dst[N+8+i] = CANARY, 'MulAddInt8Scalar out-of-bounds write rear canary');
  end;
end;

procedure TTestAVX.TestUnalignedBuffers;
const
  // Sentinel value: large enough that no computation can accidentally
  // produce it, and exactly representable as a Single.
  CANARY: Single = 1.2345678e30;
  FACTOR: Single = 2.5;

  N       = 33;   // Not a multiple of 8, exercises the tail path
  MAX_OFF = 3;    // Test offsets 0..3 floats (0/4/8/12 bytes)
  PAD     = 8;    // 8 floats of sentinel padding on each side

  BUF_SIZE = PAD + MAX_OFF + N + PAD + 4;
var
  Buf, Src, RefDst, SrcOrig: array[0..BUF_SIZE-1] of Single;
  Off, i, StartIdx, EndIdx: Integer;
begin
  for Off := 0 to MAX_OFF do
  begin
    // ---- 1. Fill the entire buffer with sentinels ----
    for i := 0 to BUF_SIZE - 1 do
    begin
      Buf[i] := CANARY;
      Src[i] := CANARY;
    end;

    // ---- 2. Compute the working region for this iteration ----
    // StartIdx is intentionally shifted by Off so that the pointer passed
    // to the AVX function is misaligned relative to a 32-byte boundary.
    StartIdx := PAD + Off;
    EndIdx   := StartIdx + N - 1;

    // ---- 3. Initialize the working region and snapshot the reference ----
    for i := StartIdx to EndIdx do
    begin
      Src[i] := (i - StartIdx + 1) * 0.7;
      Buf[i] := 10.0 + (i - StartIdx);
      RefDst[i] := Buf[i] + Src[i] * FACTOR;
      SrcOrig[i] := Src[i];
    end;

    // ---- 4. Call the function under test with misaligned pointers ----
    _AVXMulAddF(@Buf[StartIdx], @Src[StartIdx], N, FACTOR);

    // ---- 5. Verify the working region results ----
    for i := StartIdx to EndIdx do
      Check(Abs(Buf[i] - RefDst[i]) < 1e-3,
        Format('Unaligned offset=%d idx=%d: expected %.4f, actual %.4f',
               [Off, i - StartIdx, RefDst[i], Buf[i]]));

    // ---- 6. Verify the src working region was not modified ----
    for i := StartIdx to EndIdx do
      Check(Src[i] = SrcOrig[i],
        Format('Unaligned offset=%d: src[%d] unexpectedly modified',
               [Off, i - StartIdx]));

    // ---- 7. Verify dst front sentinels ----
    for i := 0 to StartIdx - 1 do
      Check(Buf[i] = CANARY,
        Format('offset=%d: dst front sentinel corrupted at index %d (value=%g)',
               [Off, i, Buf[i]]));

    // ---- 8. Verify dst rear sentinels ----
    for i := EndIdx + 1 to BUF_SIZE - 1 do
      Check(Buf[i] = CANARY,
        Format('offset=%d: dst rear sentinel corrupted at index %d (value=%g)',
               [Off, i, Buf[i]]));

    // ---- 9. Verify src front sentinels ----
    for i := 0 to StartIdx - 1 do
      Check(Src[i] = CANARY,
        Format('offset=%d: src front sentinel corrupted at index %d (value=%g)',
               [Off, i, Src[i]]));

    // ---- 10. Verify src rear sentinels ----
    for i := EndIdx + 1 to BUF_SIZE - 1 do
      Check(Src[i] = CANARY,
        Format('offset=%d: src rear sentinel corrupted at index %d (value=%g)',
               [Off, i, Src[i]]));
  end;
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

  FillWithSeq(Delta, 1.0, 0.1, N);
  FillWithSeq(M, 0.0, 0.1, N);
  FillWithSeq(V, 0.0, 0.2, N);

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

  FillWithSeq(Delta, 1.0, 0.1, N);
  FillWithSeq(V, 0.0, 0.2, N);

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
  FillWithSeq(A, -5.0, 0.04, N);
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
  FillWithSeq(Delta, 1.0, 0.1, N);
  FillWithSeq(M, 0.0, 0.2, N);

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
  N = 300;
var
  A: array[0..N-1] of Single;
  RefVal, AVXVal: Single;
  RefPos, AVXPos: Integer;
  i, LCount: Integer;
begin
  FillWithSeq(A, -5.0, 0.04, N);

  LCount := 256;
  RefVal := A[0]; RefPos := 0;
  for i := 1 to LCount-1 do
    if A[i] > RefVal then
    begin
      RefVal := A[i];
      RefPos := i;
    end;
  AVXVal := _AVXGetMaxPos(@A[0], LCount, AVXPos);
  Check((Abs(RefVal - AVXVal) < 1e-5) and (RefPos = AVXPos), Format('GetMaxPos %d failed', [LCount]));

  LCount := 259;
  RefVal := A[0]; RefPos := 0;
  for i := 1 to LCount-1 do
    if A[i] > RefVal then
    begin
      RefVal := A[i];
      RefPos := i;
    end;
  AVXVal := _AVXGetMaxPos(@A[0], LCount, AVXPos);
  Check((Abs(RefVal - AVXVal) < 1e-5) and (RefPos = AVXPos), Format('GetMaxPos %d failed', [LCount]));
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
  FillWithSeq(A, -5.0, 0.04, N);
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
  N = 300;
var
  A: array[0..N-1] of Single;
  RefVal, AVXVal: Single;
  RefPos, AVXPos: Integer;
  i, LCount: Integer;
begin
  FillWithSeq(A, -5.0, 0.04, N);

  LCount := 256;
  RefVal := Abs(A[0]); RefPos := 0;
  for i := 1 to LCount-1 do
    if Abs(A[i]) > RefVal then
    begin
      RefVal := Abs(A[i]);
      RefPos := i;
    end;
  AVXVal := _AVXGetMaxAbsPos(@A[0], LCount, AVXPos);
  Check((Abs(RefVal - AVXVal) < 1e-5) and (RefPos = AVXPos), Format('GetMaxAbsPos %d failed', [LCount]));

  LCount := 257;
  RefVal := Abs(A[0]); RefPos := 0;
  for i := 1 to LCount-1 do
    if Abs(A[i]) > RefVal then
    begin
      RefVal := Abs(A[i]);
      RefPos := i;
    end;
  AVXVal := _AVXGetMaxAbsPos(@A[0], LCount, AVXPos);
  Check((Abs(RefVal - AVXVal) < 1e-5) and (RefPos = AVXPos), Format('GetMaxAbsPos %d failed', [LCount]));
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
  FillWithSeq(A, 1.0, 0.1, N);
  for i := 0 to N-1 do Ref[i] := A[i] + Value;
  _AVXAddScalar(@A[0], Value, N);
  Check(CompareSingleArrays(A, Ref, N), 'AddScalar failed');
end;

procedure TTestAVX.TestExpShiftSum;
const
  N = 300;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  Shift, RefSum, AVXSum: Single;
  LCount, i: Integer;
begin
  Shift := 1.0;
  FillWithSeq(Src, -5.0, 0.04, N);

  LCount := 257;
  RefSum := 0;
  for i := 0 to LCount-1 do
  begin
    Ref[i] := Exp(Src[i] - Shift);
    RefSum := RefSum + Ref[i];
  end;
  AVXSum := _AVXExpShiftSum(@Dst[0], @Src[0], Shift, LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount) and (Abs(RefSum - AVXSum) < 1e-3), Format('ExpShiftSum %d failed', [LCount]));

  LCount := 256;
  RefSum := 0;
  for i := 0 to LCount-1 do
  begin
    Ref[i] := Exp(Src[i] - Shift);
    RefSum := RefSum + Ref[i];
  end;
  AVXSum := _AVXExpShiftSum(@Dst[0], @Src[0], Shift, LCount);
  Check(CompareSingleArrays(Dst, Ref, LCount) and (Abs(RefSum - AVXSum) < 1e-3), Format('ExpShiftSum %d failed', [LCount]));
end;

procedure TTestAVX.TestLn;
const
  N = 257;
var
  Dst, Src, Ref: array[0..N-1] of Single;
  i: Integer;
begin
  FillWithSeq(Src, 0.1, 0.02, N);
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
  FillWithSeq(Src, -2*Pi, 0.1, N);
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

procedure TTestAVX.TestSinCosBoth;
const
  N = 300;
  TestSizes: array[0..8] of Integer = (0, 1, 7, 8, 9, 15, 32, 257, 300);
var
  Src, DstSin, DstCos, RefSin, RefCos: array[0..N-1] of Single;
  LCount, i, k: Integer;
begin
  for k := 0 to High(TestSizes) do
  begin
    LCount := TestSizes[k];
    if LCount > N then LCount := N;

    // Cover [-2pi, 2pi] with a fine step so we get many quadrant transitions
    FillWithSeq(Src, -6.28, 0.0419, LCount);

    for i := 0 to LCount - 1 do
    begin
      RefSin[i] := Sin(Src[i]);
      RefCos[i] := Cos(Src[i]);
    end;

    _AVXSinCosBoth(@DstSin[0], @DstCos[0], @Src[0], LCount);

    Check(CompareSingleArrays(DstSin, RefSin, LCount, 1e-5, 1e-6),
      Format('SinCosBoth sin %d failed', [LCount]));
    Check(CompareSingleArrays(DstCos, RefCos, LCount, 1e-5, 1e-6),
      Format('SinCosBoth cos %d failed', [LCount]));
  end;
end;

procedure TTestAVX.TestEncodeBF16;
const
  N = 257;
var
  Src: array[0..N-1] of Single;
  Dst, Ref: array[0..N-1] of Word;
  i: Integer;
begin
  FillWithSeq(Src, -10.0, 0.08, N);
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
  FillWithSeq(Src, -1.0, 0.008, N);
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
  Check(CompareSingleArrays(Dst, Ref, N, 1e-2), 'ReluL failed');
end;

procedure TTestAVX.TestEncodeF16;
const
  N = 257;
var
  Src: array[0..N-1] of Single;
  Dst, Ref: array[0..N-1] of Word;
  i: Integer;
begin
  FillWithSeq(Src, -10.0, 0.08, N);
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
  FillWithSeq(Src, -1.0, 0.008, N);
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
