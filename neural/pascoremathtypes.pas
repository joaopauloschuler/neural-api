// pas-core-math - Pascal port of CORE-MATH
// https://github.com/joaopauloschuler/pas-core-math
//
// Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and contributors.
// Refer to the git commit history for individual authorship.
// SPDX-License-Identifier: MIT
{$I pascoremath.inc}
unit pascoremathtypes;

interface

uses Math;

type
  {$IFDEF FPC}
  {$ELSE}
  QWord = UInt64;
  DWord = UInt32;
  {$ENDIF}

  TUInt128 = record
    lo, hi: UInt64;
    {$IFDEF FPC}
    {$ELSE}
    class operator Add(const a: TUInt128; b: UInt64): TUInt128; inline;
    {$ENDIF}
  end;

  Tb32u32 = record
    case boolean of
      false: (f: Single);
      true:  (u: UInt32);
  end;

  Tb64u64 = record
    case boolean of
      false: (f: Double);
      true:  (u: UInt64);
  end;

  // 128-bit significand type used as slow path in every binary64 function.
  // Fields match the little-endian C union in sin.c / dint.h.
  TDInt64 = record
    hi:  UInt64;   // high 64 bits of significand (MSB always 1 when non-zero)
    lo:  UInt64;   // low 64 bits of significand
    ex:  Int64;    // binary exponent (signed)
    sgn: Byte;     // sign: 0 = positive, 1 = negative
  end;

  // 256-bit significand type used exclusively by pow (third Ziv iteration).
  // Arithmetic ported separately in Phase 5.01.
  TQInt64 = record
    r0, r1, r2, r3: UInt64;  // 256-bit significand, r0 = most significant
    ex:             Int64;
    sgn:            Byte;
  end;

  // 192-bit significand type used by atan2 / atan2pi (Phase 2.07 / 2.11).
  // Field order matches the little-endian C union in atan2/tint.h:
  //   m at offset 0 (low 64 of u128 _h), h at offset 8 (high 64 of u128 _h),
  //   l at offset 16 (= u64 _l). Value = (-1)^sgn * (h/2^64 + m/2^128 + l/2^192) * 2^ex.
  // h is the most significant limb; when non-zero, MSB of h is 1 (normalized).
  TInt64 = record
    m:   UInt64;   // middle 64 bits of significand
    h:   UInt64;   // high 64 bits of significand (MSB always 1 when non-zero)
    l:   UInt64;   // low 64 bits of significand
    ex:  Int64;    // binary exponent
    sgn: UInt64;   // sign: 0 = positive, 1 = negative
  end;

{$IFDEF FPC}
operator +(const a: TUInt128; b: UInt64): TUInt128; inline;
{$ENDIF}

procedure AddU128(out r: TUInt128; const a, b: TUInt128); inline;
procedure SubU128(out r: TUInt128; const a, b: TUInt128); inline;
procedure ShlU128(var a: TUInt128; sh: Integer); inline;
procedure ShrU128(var a: TUInt128; sh: Integer); inline;

function Mulu64u64(a, b: UInt64): TUInt128; {$IFNDEF AVX2} inline; {$ENDIF}

// ------- dint64_t arithmetic -------
// All operations ported faithfully from core-math/src/binary64/sin/sin.c.

// Copy
procedure CpDInt(out r: TDInt64; const a: TDInt64); inline;
// True if value is zero (hi = 0)
function DIntZeroP(const a: TDInt64): Boolean; inline;
// Compare absolute values: -1 / 0 / +1
function CmpDIntAbs(const a, b: TDInt64): Integer; inline;
// Add two TDInt64 values (error bounded by 2 ulp_128)
procedure AddDInt(out r: TDInt64; const a, b: TDInt64); inline;
// Multiply two TDInt64 values (error bounded by 6 ulp_128)
procedure MulDInt(out r: TDInt64; const a, b: TDInt64); inline;
// Multiply two TDInt64 values, assuming b.lo = 0 (error bounded by 2 ulp_128)
procedure MulDInt21(out r: TDInt64; const a, b: TDInt64); inline;
// Multiply two TDInt64 values, assuming both a.lo = b.lo = 0; the 128-bit
// product is exact (ported from mul_dint_11 in core-math/src/binary64/pow/dint.h).
procedure MulDInt11(out r: TDInt64; const a, b: TDInt64); inline;
// Add two TDInt64 values, assuming both a.lo = b.lo = 0 (error 1-2 ulp_64;
// ported from add_dint_11 in core-math/src/binary64/pow/dint.h).
procedure AddDInt11(out r: TDInt64; const a, b: TDInt64); inline;
// Truncate a TDInt64 to a signed 64-bit integer (rounding toward zero)
// — ported from dint_toi in core-math/src/binary64/pow/pow.h. Note that
// the dint format here uses ex=1 for value 1.0, so the C formula
// `hi >> (63 - ex)` becomes `hi >> (64 - ex)` in our convention.
function DIntToI(const a: TDInt64): Int64; inline;
// Multiply a TDInt64 by a signed 64-bit integer (ported from mul_dint_2 in
// core-math/src/binary64/log/dint.h). Caller must ensure |b| fits the dint
// range — currently safe for log/log10 where |b| <= 1074.
procedure MulDIntInt(out r: TDInt64; b: Int64; const a: TDInt64); inline;
// Normalize X so that X.hi has its most significant bit set (if X <> 0).
// Used by the 2*pi range-reduction routines (sin/cos/tan/sincos).
procedure NormalizeDInt(var X: TDInt64); inline;
// Convert Double to TDInt64
procedure DIntFromD(out a: TDInt64; b: Double); inline;
// Convert TDInt64 to Double (modifies a via subnormalise; pass a copy if const needed)
function DToD(var a: TDInt64): Double; inline;

// ------- tint64_t (192-bit) arithmetic -------
// All operations ported faithfully from core-math/src/binary64/atan2/tint.h.

procedure CpTInt(out r: TInt64; const a: TInt64); inline;
function TIntZeroP(const a: TInt64): Boolean; inline;
function CmpTIntAbs(const a, b: TInt64): Integer; inline;
// Right shift the significand (h, m, l) by k bits. Does not touch ex/sgn.
procedure RShiftTInt(var r: TInt64; const b: TInt64; k: Integer); inline;
// Left shift the significand (h, m, l) by k bits. Does not touch ex/sgn.
procedure LShiftTInt(var r: TInt64; const b: TInt64; k: Integer); inline;
// r := a + b   (error bounded by 2 ulps in 192-bit)
procedure AddTInt(out r: TInt64; const a, b: TInt64); inline;
// Same as AddTInt but caller GUARANTEES r is distinct from a and b.
// Skips the 80-byte entry copies and the swap copy. Wrong results if r aliases.
procedure AddTInt_NoAlias(out r: TInt64; const a, b: TInt64); inline;
// r := a * b   (error bounded by 10 ulps in 192-bit; alias-safe)
procedure MulTInt(out r: TInt64; const a, b: TInt64); {$IFNDEF AVX2} inline; {$ENDIF}
// Convert Double to TInt64 (exact for finite, non-NaN inputs; defined for 0)
procedure TIntFromD(out a: TInt64; b: Double); inline;
// Convert TInt64 to Double with directed rounding driven by err (in ulps of l).
// y, x are pass-through inputs used only for the worst-case panic message.
function TIntToD(const a: TInt64; err: UInt64; y, x: Double): Double; {$IFDEF FPC} inline; {$ENDIF}
// r := 1 / A   (relative error < 2^-103.9; A must be non-zero)
procedure InvTInt(out r: TInt64; const A: TInt64); inline;
// r := b / a   (relative error < 2^-185.53)
procedure DivTInt(out r: TInt64; const b, a: TInt64); inline;
// Convenience: r := bd / ad, both Doubles
procedure DivTIntD(out r: TInt64; bd, ad: Double); inline;

// ------- qint64_t (256-bit) arithmetic -------
// All operations ported faithfully from core-math/src/binary64/pow/qint.h.
// Field mapping: r0 = hh (most significant), r1 = hl, r2 = lh, r3 = ll.

// Copy
procedure CpQInt(out r: TQInt64; const a: TQInt64); inline;
// True if value is zero (r0 = r1 = 0; matches C check on rh)
function QIntZeroP(const a: TQInt64): Boolean; inline;
// Compare absolute values: -1 / 0 / +1 (full 256-bit precision)
function CmpQIntAbs(const a, b: TQInt64): Integer; inline;
// Compare absolute values using only the upper 128 bits
function CmpQIntAbs22(const a, b: TQInt64): Integer; inline;
// r := a + b (error bounded by 2 ulps in 256-bit; alias-safe)
procedure AddQInt(out r: TQInt64; const a, b: TQInt64); inline;
// Same as AddQInt but only considers upper 2 limbs (error < 2 ulps in 128-bit)
procedure AddQInt22(out r: TQInt64; const a, b: TQInt64); inline;
// r := a * b (error < 14 ulps in 256-bit)
procedure MulQInt(out r: TQInt64; const a, b: TQInt64); inline;
// Same as MulQInt but only considers upper 3 limbs of a and b (error < 6 ulps)
procedure MulQInt33(out r: TQInt64; const a, b: TQInt64); inline;
// Same as MulQInt but only considers upper limb of b (error < 2 ulps)
procedure MulQInt41(out r: TQInt64; const a, b: TQInt64); inline;
// Same as MulQInt but uses upper 3 limbs of a, upper limb of b (no error)
procedure MulQInt31(out r: TQInt64; const a, b: TQInt64); inline;
// Same as MulQInt but uses upper 2 limbs of a and b (no error)
procedure MulQInt22(out r: TQInt64; const a, b: TQInt64); inline;
// Same as MulQInt but uses upper 2 limbs of a, upper limb of b (no error)
procedure MulQInt21(out r: TQInt64; const a, b: TQInt64); inline;
// Same as MulQInt but uses upper limb of a and b only (no error)
procedure MulQInt11(out r: TQInt64; const a, b: TQInt64); inline;
// Multiply integer b by qint a (error < 2 ulps); ported from mul_qint_2
procedure MulQIntInt(out r: TQInt64; b: Int64; const a: TQInt64); inline;
// Truncate a TQInt64 to a signed 64-bit integer (rounding toward zero).
// Ported from qint_toi in core-math/src/binary64/pow/pow.h. QInt convention
// here matches C exactly (ex of QINT_ONE is 0), so the C formula
// `hh >> (63 - ex)` applies as-is.
function QIntToI(const a: TQInt64): Int64; inline;

const
  cNaNSingle: Single = 0.0/0.0;       // x86 indefinite: 0xFFC00000 (negative quiet NaN)
  cNaNDouble: Double = 0.0/0.0;
  // Positive quiet NaNs matching __builtin_nanf("") and __builtin_nanf("1") in C.
  cNaNSinglePos:  Tb32u32 = (u: $7FC00000); // positive quiet NaN, payload 0
  cNaNSinglePos1: Tb32u32 = (u: $7FC00001); // positive quiet NaN, payload 1
  // Positive quiet NaNs (Double) matching __builtin_nan("") and __builtin_nan("1") in C.
  cNaNDoublePos:  Tb64u64 = (u: $7FF8000000000000); // positive quiet NaN, payload 0
  cNaNDoublePos1: Tb64u64 = (u: $7FF8000000000001); // positive quiet NaN, payload 1
  // 2^(-127): subnormal Single used to trigger IEEE 754 underflow via multiplication
  cUnderflowSingle: Single = 5.877471754111438e-39;

  // Sentinel dint constants (see sin.c ZERO / ONE)
  DINT_ZERO: TDInt64 = (hi: 0; lo: 0; ex: -1076; sgn: 0);
  // hi = $8000000000000000 = 2^63 (MSB set, normalised 1.0 in dint format)
  // ex=1: value = hi/2^64 * 2^ex = (2^63/2^64) * 2^1 = 0.5 * 2 = 1.0
  DINT_ONE:  TDInt64 = (hi: QWord($8000000000000000); lo: 0; ex: 1; sgn: 0);
  // -1 (used by pow log_2 / log_3); same magnitude as DINT_ONE, sgn=1.
  DINT_M_ONE: TDInt64 = (hi: QWord($8000000000000000); lo: 0; ex: 1; sgn: 1);
  // log(2) to absolute error < 2^-129.97 (ex shifted by +1 vs C convention).
  DINT_LOG2: TDInt64 = (hi: QWord($B17217F7D1CF79AB); lo: QWord($C9E3B39803F2F6AF);
                        ex: 0; sgn: 0);
  // 2^12/log(2) to absolute error < 2^-52.96 (ex shifted by +1 vs C).
  DINT_LOG2_INV: TDInt64 = (hi: QWord($B8AA3B295C17F0BC); lo: 0; ex: 13; sgn: 0);

  // Sentinel TInt64 constants (see atan2/tint.h ZERO / ONE / PI / PI2)
  TINT_ZERO: TInt64 = (m: 0; h: 0; l: 0; ex: -1076; sgn: 0);
  TINT_ONE:  TInt64 = (m: 0; h: QWord($8000000000000000); l: 0; ex: 1; sgn: 0);
  // pi to error < 2^-196.96
  TINT_PI:   TInt64 = (m: QWord($C4C6628B80DC1CD1); h: QWord($C90FDAA22168C234);
                       l: $29024E088A67CC74; ex: 2; sgn: 0);
  // pi/2 to error < 2^-197.96
  TINT_PI2:  TInt64 = (m: QWord($C4C6628B80DC1CD1); h: QWord($C90FDAA22168C234);
                       l: $29024E088A67CC74; ex: 1; sgn: 0);
  // 1/2 (used by atan2pi)
  TINT_ONE_HALF: TInt64 = (m: 0; h: QWord($8000000000000000); l: 0; ex: 0; sgn: 0);
  // 1/pi to relative error < 2^-198.59 (used by atan2pi)
  TINT_ONE_OVER_PI: TInt64 = (m: QWord($FC2757D1F534DDC0); h: QWord($A2F9836E4E441529);
                              l: QWord($DB6295993C439042); ex: -1; sgn: 0);

  // Helpers used inside InvTInt
  cTI_1pm1022: Tb64u64 = (u: $0010000000000000); // 0x1p-1022 (smallest normal)
  cTI_1p53:    Tb64u64 = (u: $4340000000000000); // 0x1p+53

  // Sentinel TQInt64 constants (see pow/qint.h ZERO_Q / ONE_Q / M_ONE_Q / LOG2_Q / LOG2_INV_Q)
  QINT_ZERO: TQInt64 = (r0: 0; r1: 0; r2: 0; r3: 0; ex: 0; sgn: 0);
  // r0 = $8000... (MSB set, value 1.0 exactly)
  QINT_ONE:  TQInt64 = (r0: QWord($8000000000000000); r1: 0; r2: 0; r3: 0; ex: 0; sgn: 0);
  QINT_M_ONE: TQInt64 = (r0: QWord($8000000000000000); r1: 0; r2: 0; r3: 0; ex: 0; sgn: 1);
  // log(2) to absolute error < 2^-256.14
  QINT_LOG2: TQInt64 = (r0: QWord($B17217F7D1CF79AB); r1: QWord($C9E3B39803F2F6AF);
                        r2: $40F343267298B62D; r3: QWord($8A0D175B8BAAFA2B);
                        ex: -1; sgn: 0);
  // 2^12/log(2) to absolute error < 2^-52.96
  QINT_LOG2_INV: TQInt64 = (r0: QWord($B8AA3B295C17F0BC); r1: 0; r2: 0; r3: 0;
                            ex: 12; sgn: 0);

  // Hex-float constants used inside DToD / subnormalise_dint
  cDTD_1pm53:    Tb64u64 = (u: $3CA0000000000000); // 0x1p-53
  cDTD_1pm54:    Tb64u64 = (u: $3C90000000000000); // 0x1p-54
  cDTD_1p1023:   Tb64u64 = (u: $7FE0000000000000); // 0x1p+1023
  cDTD_MaxNorm:  Tb64u64 = (u: $7FEFFFFFFFFFFFFF); // 0x1.fffffffffffffp+1023 = DBL_MAX
  cDTD_MinSub:   Tb64u64 = (u: $0000000000000001); // 0x1p-1074 = smallest subnormal

implementation

// ---------------------------------------------------------------------------
// Internal 64-bit clz helper (no external dependencies, used by dint ops).
// Identical implementation to pcr_clzll in pascoremathhelperfuncs.pas.
// ---------------------------------------------------------------------------
function clzll64(x: UInt64): Integer; inline;
{$IFDEF AVX2}
begin
  Result := 63 - BsrQWord(x);
end;
{$ELSE}
var n: Integer;
begin
  if x = 0 then begin Result := 64; Exit; end;
  n := 0;
  if (x and UInt64($FFFFFFFF00000000)) = 0 then begin n := n + 32; x := x shl 32; end;
  if (x and UInt64($FFFF000000000000)) = 0 then begin n := n + 16; x := x shl 16; end;
  if (x and UInt64($FF00000000000000)) = 0 then begin n := n +  8; x := x shl  8; end;
  if (x and UInt64($F000000000000000)) = 0 then begin n := n +  4; x := x shl  4; end;
  if (x and UInt64($C000000000000000)) = 0 then begin n := n +  2; x := x shl  2; end;
  if (x and UInt64($8000000000000000)) = 0 then n := n + 1;
  Result := n;
end;
{$ENDIF}

// ---------------------------------------------------------------------------
// Internal 128-bit arithmetic helpers (private, used by dint ops)
// ---------------------------------------------------------------------------

// r := a + b  (128-bit addition)
procedure AddU128(out r: TUInt128; const a, b: TUInt128); inline;
var alo: UInt64;
begin
  alo := a.lo;
  r.lo := alo + b.lo;
  r.hi := a.hi + b.hi + UInt64(r.lo < alo);
end;

// r := a - b  (128-bit subtraction)
procedure SubU128(out r: TUInt128; const a, b: TUInt128); inline;
var alo: UInt64;
begin
  alo := a.lo;
  r.lo := alo - b.lo;
  r.hi := a.hi - b.hi - UInt64(alo < b.lo);
end;

// a <<= sh  (0 <= sh; in-place 128-bit left shift)
procedure ShlU128(var a: TUInt128; sh: Integer); inline;
begin
  if sh <= 0 then Exit;
  if sh < 64 then begin
    a.hi := (a.hi shl sh) or (a.lo shr (64 - sh));
    a.lo := a.lo shl sh;
  end else if sh < 128 then begin
    a.hi := a.lo shl (sh - 64);
    a.lo := 0;
  end else begin
    a.hi := 0; a.lo := 0;
  end;
end;

// a >>= sh  (0 <= sh; in-place 128-bit logical right shift)
procedure ShrU128(var a: TUInt128; sh: Integer); inline;
begin
  if sh <= 0 then Exit;
  if sh < 64 then begin
    a.lo := (a.lo shr sh) or (a.hi shl (64 - sh));
    a.hi := a.hi shr sh;
  end else if sh < 128 then begin
    a.lo := a.hi shr (sh - 64);
    a.hi := 0;
  end else begin
    a.lo := 0; a.hi := 0;
  end;
end;

// ---------------------------------------------------------------------------
// TUInt128 operator and Mulu64u64 (existing)
// ---------------------------------------------------------------------------

{$IFDEF FPC}
operator +(const a: TUInt128; b: UInt64): TUInt128; inline;
begin
  Result.lo := a.lo + b;
  Result.hi := a.hi + UInt64(Result.lo < b);
end;
{$ENDIF}

function Mulu64u64(a, b: UInt64): TUInt128;
{$IFDEF AVX2}
var
  rlo, rhi: UInt64;
begin
  //Result := Default(TUInt128);
  asm
    mov  rax, a
    mul  b           // rdx:rax = a * b
    mov  rlo, rax
    mov  rhi, rdx
  end {$IFDEF FPC} ['rax', 'rdx']{$ENDIF};
  Result.lo := rlo;
  Result.hi := rhi;
end;
{$ELSE}
// Portable fallback: four 32-bit partial products
// done by nanobit in the Lazarus forum: https://forum.lazarus.freepascal.org/index.php/topic,73881.0.html
var
  MulLo, Temp1, Temp2: UInt64;
begin
  //Result := Default(TUInt128);
  MulLo := uint64(uint32(a)) * uint64(uint32(b));
  Temp1 := (a shr 32) * uint64(uint32(b)) + (MulLo shr 32);
  Temp2 := uint64(uint32(a)) * (b shr 32) + uint64(uint32(Temp1));
  Result.lo := (Temp2 shl 32) or (MulLo and $FFFFFFFF);
  Result.hi := (a shr 32) * (b shr 32) + (Temp1 shr 32) + (Temp2 shr 32);
end;
{$ENDIF}

// ---------------------------------------------------------------------------
// dint64_t arithmetic
// ---------------------------------------------------------------------------

procedure CpDInt(out r: TDInt64; const a: TDInt64); inline;
begin
  r := a;
end;

function DIntZeroP(const a: TDInt64): Boolean; inline;
begin
  Result := a.hi = 0;
end;

function CmpDIntAbs(const a, b: TDInt64): Integer; inline;
begin
  if a.hi = 0 then begin
    if b.hi = 0 then Result := 0 else Result := -1;
    Exit;
  end;
  if b.hi = 0 then begin Result := 1; Exit; end;
  if a.ex > b.ex then begin Result := 1; Exit; end;
  if a.ex < b.ex then begin Result := -1; Exit; end;
  // same exponent: compare 128-bit significands as unsigned
  if a.hi > b.hi then Result := 1
  else if a.hi < b.hi then Result := -1
  else if a.lo > b.lo then Result := 1
  else if a.lo < b.lo then Result := -1
  else Result := 0;
end;

// Subnormal rounding used inside DToD — modifies a in place.
// Ported faithfully from subnormalize_dint in sin.c.
procedure SubnormalizeDInt(var a: TDInt64); inline;
var
  ex: UInt64;
  hi, md, lo: UInt64;
  rmode: TFPURoundingMode;
begin
  if a.ex > -1023 then Exit;

  ex := UInt64(-(1011 + a.ex));
  hi := a.hi shr ex;
  md := (a.hi shr (ex - 1)) and 1;
  // logical OR: lo = 1 if any low/residual bits are non-zero
  lo := UInt64(((a.hi and (UInt64($FFFFFFFFFFFFFFFF) shr ex)) <> 0) or (a.lo <> 0));

  rmode := GetRoundMode;
  case rmode of
    rmNearest:
      if lo <> 0 then hi := hi + md
      else hi := hi + (hi and md);
    rmDown:
      if (a.sgn <> 0) and ((md or lo) <> 0) then Inc(hi);
    rmUp:
      if (a.sgn = 0) and ((md or lo) <> 0) then Inc(hi);
    // rmTruncate: truncate towards zero — no correction needed
  end;

  a.hi := hi shl ex;
  a.lo := 0;

  if a.hi = 0 then begin
    Inc(a.ex);
    a.hi := UInt64(1) shl 63;
  end;
end;

// Ported from add_dint in sin.c.
// NOTE: Pascal identifiers are case-insensitive; local 128-bit vars are
// named vA/vB/vC/vD/vE to avoid collision with the parameters a/b.
procedure AddDInt(out r: TDInt64; const a, b: TDInt64); inline;
var
  pa, pb, ptmp: TDInt64;
  vA, vB, vBorig, vC, vD, vE: TUInt128;
  k, ex: UInt64;
  sgn: Byte;
  ch: UInt64;
  cmp: Integer;
  sh: Integer;
begin
  pa := a; pb := b;  // local copies handle aliasing (r may alias a or b)

  // if a is zero (both hi and lo are 0), return b
  if (pa.hi or pa.lo) = 0 then begin
    r := pb;
    Exit;
  end;

  cmp := CmpDIntAbs(pa, pb);

  case cmp of
    0:
      begin
        if (pa.sgn xor pb.sgn) <> 0 then
          r := DINT_ZERO
        else begin
          r := pa;
          Inc(r.ex);
        end;
        Exit;
      end;
    -1:
      begin
        ptmp := pa; pa := pb; pb := ptmp;  // swap so |pa| >= |pb|
      end;
  end;

  // From here |pa| >= |pb|
  vA.hi := pa.hi; vA.lo := pa.lo;
  vB.hi := pb.hi; vB.lo := pb.lo;
  vBorig := vB;  // save original pb.r for Sterbenz case
  k := UInt64(pa.ex - pb.ex);

  if k > 0 then begin
    if k < 128 then
      ShrU128(vB, Integer(k))
    else begin
      vB.hi := 0; vB.lo := 0;
    end;
  end;

  sgn := pa.sgn;
  r.ex := pa.ex;

  if (pa.sgn xor pb.sgn) <> 0 then begin
    // Different signs: vC = vA - vB
    SubU128(vC, vA, vB);

    ch := vC.hi;
    if ch <> 0 then
      ex := UInt64(clzll64(ch))
    else
      ex := 64 + UInt64(clzll64(vC.lo));

    if ex > 0 then begin
      sh := Integer(ex);
      if k = 1 then begin
        // Sterbenz case: vC = (vA << ex) - (vBorig << (ex - 1))
        vD := vA; ShlU128(vD, sh);
        vE := vBorig; ShlU128(vE, sh - 1);
        SubU128(vC, vD, vE);
      end else begin
        vD := vA; ShlU128(vD, sh);
        vE := vB; ShlU128(vE, sh);
        SubU128(vC, vD, vE);
      end;
      Dec(r.ex, Int64(ex));
      ex := UInt64(clzll64(vC.hi));  // now 0 or 1
    end;

    // Final normalization
    ShlU128(vC, Integer(ex));
    Dec(r.ex, Int64(ex));
  end else begin
    // Same signs: vC = vA + vB
    AddU128(vC, vA, vB);

    // Detect 128-bit overflow: vC < vA
    if (vC.hi < vA.hi) or ((vC.hi = vA.hi) and (vC.lo < vA.lo)) then begin
      // vC = (1 << 127) | (vC >> 1)
      vC.lo := (vC.lo shr 1) or (vC.hi shl 63);
      vC.hi := UInt64($8000000000000000) or (vC.hi shr 1);
      Inc(r.ex);
    end;
  end;

  r.sgn := sgn;
  r.hi := vC.hi;
  r.lo := vC.lo;
end;

// Ported from mul_dint in sin.c.
// NOTE: overlap between r and a is allowed (inputs saved to locals first).
procedure MulDInt(out r: TDInt64; const a, b: TDInt64); inline;
var
  m1, m2, rr: TUInt128;
  ah, bh, al, bl: UInt64;
  ex: UInt64;
  rex_a, rex_b: Int64;
  rsgn: Byte;
begin
  // Save inputs before any write to r (r may alias a)
  ah := a.hi; al := a.lo;
  bh := b.hi; bl := b.lo;
  rex_a := a.ex; rex_b := b.ex;
  rsgn := a.sgn xor b.sgn;

  // hi * hi
  rr := Mulu64u64(ah, bh);

  // middle terms: add high 64 bits of (hi*lo) and (lo*hi)
  m1 := Mulu64u64(ah, bl);
  m2 := Mulu64u64(al, bh);

  // rr += (m1 >> 64) + (m2 >> 64)  — no overflow (see C comment)
  rr := rr + m1.hi;
  rr := rr + m2.hi;

  // Normalize: ensure MSB of rr.hi is set
  ex := rr.hi shr 63;
  if ex = 0 then begin
    rr.hi := (rr.hi shl 1) or (rr.lo shr 63);
    rr.lo := rr.lo shl 1;
  end;

  r.hi  := rr.hi;
  r.lo  := rr.lo;
  r.ex  := rex_a + rex_b + Int64(ex) - 1;
  r.sgn := rsgn;
end;

// Ported from mul_dint_21 in core-math/src/binary64/cos/cos.c:
// "Multiply two dint64_t numbers, assuming the low part of b is zero,
//  with error bounded by 2 ulps."
procedure MulDInt21(out r: TDInt64; const a, b: TDInt64); inline;
var
  hi, lo: TUInt128;
  ah, al, bh: UInt64;
  rex_a, rex_b: Int64;
  rsgn: Byte;
  ex: UInt64;
begin
  ah := a.hi; al := a.lo;
  bh := b.hi;
  rex_a := a.ex; rex_b := b.ex;
  rsgn := a.sgn xor b.sgn;

  hi := Mulu64u64(ah, bh);
  lo := Mulu64u64(al, bh);

  // r.r = hi + (lo >> 64)
  hi := hi + lo.hi;

  ex := hi.hi shr 63;
  if ex = 0 then begin
    hi.hi := (hi.hi shl 1) or (hi.lo shr 63);
    hi.lo := hi.lo shl 1;
  end;

  r.hi  := hi.hi;
  r.lo  := hi.lo;
  r.ex  := rex_a + rex_b + Int64(ex) - 1;
  r.sgn := rsgn;
end;

// Ported from mul_dint_11 in core-math/src/binary64/pow/dint.h.
// Both operand low limbs are assumed zero; the 128-bit product is exact.
procedure MulDInt11(out r: TDInt64; const a, b: TDInt64); inline;
var
  hi: TUInt128;
  ex: UInt64;
begin
  hi := Mulu64u64(a.hi, b.hi);
  ex := hi.hi shr 63;
  if ex = 0 then begin
    hi.hi := (hi.hi shl 1) or (hi.lo shr 63);
    hi.lo := hi.lo shl 1;
  end;
  r.hi  := hi.hi;
  r.lo  := hi.lo;
  r.ex  := a.ex + b.ex + Int64(ex) - 1;
  r.sgn := a.sgn xor b.sgn;
end;

// Ported from add_dint_11 in core-math/src/binary64/pow/dint.h.
// Both operand low limbs are assumed zero; error bounded by 2 ulps_64.
procedure AddDInt11(out r: TDInt64; const a, b: TDInt64); inline;
var
  pa, pb: TDInt64;
  uA, uB, uC, tmp64: UInt64;
  k, ex_shift: UInt64;
  cmp: Integer;
  sgn, tmpSgn: Byte;
  tmpEx: Int64;
begin
  pa := a; pb := b;
  if pa.hi = 0 then begin r := pb; Exit; end;
  if pb.hi = 0 then begin r := pa; Exit; end;

  // cmp_dint_11: compare ex first, then hi
  if pa.ex > pb.ex then cmp := 1
  else if pa.ex < pb.ex then cmp := -1
  else if pa.hi > pb.hi then cmp := 1
  else if pa.hi < pb.hi then cmp := -1
  else cmp := 0;

  case cmp of
    0:
      begin
        if (pa.sgn xor pb.sgn) <> 0 then r := DINT_ZERO
        else begin r := pa; Inc(r.ex); end;
        Exit;
      end;
    -1:
      begin
        // swap so |pa| > |pb|
        tmp64 := pa.hi; pa.hi := pb.hi; pb.hi := tmp64;
        tmpEx := pa.ex; pa.ex := pb.ex; pb.ex := tmpEx;
        tmpSgn := pa.sgn; pa.sgn := pb.sgn; pb.sgn := tmpSgn;
      end;
  end;

  uA := pa.hi; uB := pb.hi;
  if pa.ex > pb.ex then begin
    k := UInt64(pa.ex - pb.ex);
    if k < 64 then uB := uB shr k else uB := 0;
  end;

  sgn := pa.sgn;
  r.ex := pa.ex;

  if (pa.sgn xor pb.sgn) <> 0 then begin
    // different signs: uC = uA - uB; uA > uB since |pa| > |pb|
    uC := uA - uB;
    ex_shift := UInt64(clzll64(uC));
    if ex_shift > 0 then begin
      uC := (uA shl ex_shift) - (uB shl ex_shift);
      Dec(r.ex, Int64(ex_shift));
      ex_shift := UInt64(clzll64(uC));  // 0 or 1
    end;
    uC := uC shl ex_shift;
    Dec(r.ex, Int64(ex_shift));
  end else begin
    // same signs: uC = uA + uB
    uC := uA + uB;
    if uC < uA then begin
      // overflow: shift right and set MSB, bump exponent
      uC := (UInt64($8000000000000000)) or (uC shr 1);
      Inc(r.ex);
    end;
  end;

  r.sgn := sgn;
  r.hi  := uC;
  r.lo  := 0;
end;

// Ported from dint_toi in core-math/src/binary64/pow/pow.h.
// Truncates toward zero; assumes |a| < 2^63 (caller guards).
function DIntToI(const a: TDInt64): Int64; inline;
var
  shift: Integer;
  r: UInt64;
begin
  if a.ex < 1 then begin Result := 0; Exit; end;
  // C uses shift (63 - ex_C); our ex is C-ex + 1, so shift = 64 - our_ex.
  shift := Integer(64 - a.ex);
  if shift < 0 then shift := 0;
  if shift >= 64 then r := 0
  else r := a.hi shr shift;
  if a.sgn <> 0 then Result := -Int64(r) else Result := Int64(r);
end;

// Ported from mul_dint_2 in core-math/src/binary64/log/dint.h.
// Multiplies a dint by an Int64; result has same convention as input.
procedure MulDIntInt(out r: TDInt64; b: Int64; const a: TDInt64); inline;
var
  c: UInt64;
  t, l, sum: TUInt128;
  m: Integer;
  carry: Boolean;
begin
  if b = 0 then begin
    CpDInt(r, DINT_ZERO);
    Exit;
  end;
  if b < 0 then begin
    c := UInt64(-b);
    r.sgn := Byte(a.sgn xor 1);
  end else begin
    c := UInt64(b);
    r.sgn := a.sgn;
  end;

  // t = a.hi * c (128-bit)
  t := Mulu64u64(a.hi, c);
  if t.hi <> 0 then m := clzll64(t.hi)
  else m := 64;

  // t <<= m
  ShlU128(t, m);

  // l = a.lo * c
  l := Mulu64u64(a.lo, c);
  // l = (l << (m-1)) >> 63  -- round bit alignment.
  // m >= 1 in all reachable cases (a.hi != 0).
  if m >= 1 then ShlU128(l, m - 1);
  ShrU128(l, 63);

  // t = l + t (track carry)
  sum.lo := l.lo + t.lo;
  carry  := sum.lo < l.lo;
  sum.hi := l.hi + t.hi + UInt64(carry);
  carry  := (sum.hi < l.hi) or ((sum.hi = l.hi) and carry);
  t := sum;

  if carry then begin
    // round-half-to-even on the bottom bit, then shift right 1 and set MSB
    if (t.lo and 1) <> 0 then begin
      // t += 1 (with carry-out into bit 128 ignored — already overflowed)
      Inc(t.lo);
      if t.lo = 0 then Inc(t.hi);
    end;
    // t >>= 1
    t.lo := (t.lo shr 1) or (t.hi shl 63);
    t.hi := (t.hi shr 1) or (UInt64(1) shl 63);
    Dec(m);
  end;

  r.hi := t.hi;
  r.lo := t.lo;
  r.ex := a.ex + 64 - Int64(m);
end;

// Ported from normalize() in core-math/src/binary64/cos/cos.c:
// shift left so X.hi has its MSB set (if X <> 0); adjust ex accordingly.
procedure NormalizeDInt(var X: TDInt64); inline;
var
  cnt: Integer;
begin
  if X.hi <> 0 then begin
    cnt := clzll64(X.hi);
    if cnt <> 0 then begin
      X.hi := (X.hi shl cnt) or (X.lo shr (64 - cnt));
      X.lo := X.lo shl cnt;
    end;
    X.ex := X.ex - Int64(cnt);
  end else if X.lo <> 0 then begin
    cnt := clzll64(X.lo);
    X.hi := X.lo shl cnt;
    X.lo := 0;
    X.ex := X.ex - Int64(64 + cnt);
  end;
end;

// Ported from dint_fromd / fast_extract in sin.c.
procedure DIntFromD(out a: TDInt64; b: Double); inline;
var
  xu: Tb64u64;
  e: Int64;
  m: UInt64;
  t: Integer;
begin
  xu.f := b;
  e := Int64((xu.u shr 52) and $7FF);
  m := xu.u and UInt64($000FFFFFFFFFFFFF);
  if e <> 0 then m := m or (UInt64(1) shl 52);
  e := e - $3FE;  // biased_exp - 1022

  t := clzll64(m);
  a.sgn := Byte(b < 0.0);
  a.hi  := m shl t;
  if t > 11 then a.ex := e - Int64(t - 12)
  else a.ex := e;
  a.lo  := 0;
end;

// Ported from dint_tod in sin.c.
// Calls SubnormalizeDInt which may modify a.
function DToD(var a: TDInt64): Double; inline;
var
  ru, eu: Tb64u64;
  rd: Double;
  ex_val: Int64;
begin
  SubnormalizeDInt(a);

  ru.u := (a.hi shr 11) or (UInt64($3FF) shl 52);

  rd := 0.0;
  if ((a.hi shr 10) and 1) <> 0 then
    rd := rd + cDTD_1pm53.f;   // 0x1p-53

  if ((a.hi and $3FF) <> 0) or (a.lo <> 0) then
    rd := rd + cDTD_1pm54.f;   // 0x1p-54

  if a.sgn <> 0 then rd := -rd;

  ru.u := ru.u or (UInt64(a.sgn) shl 63);
  ru.f := ru.f + rd;

  ex_val := a.ex;

  if ex_val > -1022 then begin
    // Normal double result
    if ex_val > 1024 then begin
      if ex_val = 1025 then begin
        ru.f := ru.f * 2.0;
        eu.f := cDTD_1p1023.f;    // 0x1p+1023
      end else begin
        ru.f := cDTD_MaxNorm.f;   // DBL_MAX
        eu.f := cDTD_MaxNorm.f;
      end;
    end else
      eu.u := UInt64((ex_val + 1022) and $7FF) shl 52;
  end else begin
    // Subnormal range
    if ex_val < -1073 then begin
      if ex_val = -1074 then begin
        ru.f := ru.f * 0.5;
        eu.f := cDTD_MinSub.f;    // 0x1p-1074
      end else begin
        ru.f := cDTD_MinSub.f;
        eu.f := cDTD_MinSub.f;
      end;
    end else
      eu.u := UInt64(1) shl UInt64(ex_val + 1073);
  end;

  Result := ru.f * eu.f;
end;

// ---------------------------------------------------------------------------
// TInt64 (192-bit) arithmetic — ported from core-math/src/binary64/atan2/tint.h
// ---------------------------------------------------------------------------

procedure CpTInt(out r: TInt64; const a: TInt64); inline;
begin
  r := a;
end;

function TIntZeroP(const a: TInt64): Boolean; inline;
begin
  Result := a.h = 0;
end;

function CmpTIntAbs(const a, b: TInt64): Integer; inline;
begin
  if a.h = 0 then begin
    if b.h = 0 then Result := 0 else Result := -1;
    Exit;
  end;
  if b.h = 0 then begin Result := 1; Exit; end;
  if a.ex > b.ex then begin Result := 1; Exit; end;
  if a.ex < b.ex then begin Result := -1; Exit; end;
  // same exponent: compare 192-bit significands as unsigned (h:m:l)
  if a.h > b.h then Result := 1
  else if a.h < b.h then Result := -1
  else if a.m > b.m then Result := 1
  else if a.m < b.m then Result := -1
  else if a.l > b.l then Result := 1
  else if a.l < b.l then Result := -1
  else Result := 0;
end;

// Right shift only the (h, m, l) significand. Caller manages ex/sgn.
procedure RShiftTInt(var r: TInt64; const b: TInt64; k: Integer); inline;
var bh, bm, bl: UInt64;
begin
  bh := b.h; bm := b.m; bl := b.l;
  if k = 0 then begin r.h := bh; r.m := bm; r.l := bl; end
  else if k < 64 then begin
    r.h := bh shr k;
    r.m := (bm shr k) or (bh shl (64 - k));
    r.l := (bl shr k) or (bm shl (64 - k));
  end
  else if k = 64 then begin
    r.h := 0;
    r.m := bh;
    r.l := bm;
  end
  else if k < 128 then begin
    r.h := 0;
    r.m := bh shr (k - 64);
    r.l := (bm shr (k - 64)) or (bh shl (128 - k));
  end
  else if k < 192 then begin
    r.h := 0;
    r.m := 0;
    r.l := bh shr (k - 128);
  end
  else begin
    r.h := 0; r.m := 0; r.l := 0;
  end;
end;

// Left shift only the (h, m, l) significand. Caller manages ex/sgn.
procedure LShiftTInt(var r: TInt64; const b: TInt64; k: Integer); inline;
var bh, bm, bl: UInt64;
begin
  bh := b.h; bm := b.m; bl := b.l;
  if k = 0 then begin r.h := bh; r.m := bm; r.l := bl; end
  else if k < 64 then begin
    r.h := (bh shl k) or (bm shr (64 - k));
    r.m := (bm shl k) or (bl shr (64 - k));
    r.l := bl shl k;
  end
  else if k = 64 then begin
    r.h := bm;
    r.m := bl;
    r.l := 0;
  end
  else if k < 128 then begin
    r.h := (bm shl (k - 64)) or (bl shr (128 - k));
    r.m := bl shl (k - 64);
    r.l := 0;
  end
  else if k < 192 then begin
    r.h := bl shl (k - 128);
    r.m := 0;
    r.l := 0;
  end
  else begin
    r.h := 0; r.m := 0; r.l := 0;
  end;
end;

// Internal: 192-bit clz of (h:m:l) treated as unsigned.
// Returns 0..192. Undefined-but-defined: returns 192 for the all-zero input.
function clz192(h, m, l: UInt64): Integer; inline;
begin
  if h <> 0 then Result := clzll64(h)
  else if m <> 0 then Result := 64 + clzll64(m)
  else Result := 128 + clzll64(l);
end;

// Unsigned 64x64 -> 128 multiply with the high/low halves written through
// out parameters (no record round-trip). Used by MulTInt to keep the 6
// partial products in plain UInt64 registers/locals.
{$IFDEF AVX2}
{$IFDEF MSWINDOWS}
// Win64 ABI: rcx=@hi, rdx=@lo, r8=a, r9=b. mul writes rdx:rax, so we must
// move @lo out of rdx before issuing mul.
procedure Mul64x64(out hi, lo: UInt64; a, b: UInt64); assembler; nostackframe;
asm
  mov r10, rdx           // preserve @lo (rdx is about to be clobbered)
  mov rax, r8            // a
  mul r9                 // rdx:rax = a*b
  mov [rcx], rdx         // *hi = high
  mov [r10], rax         // *lo = low
end;
{$ELSE}
// SysV: rdi=@hi, rsi=@lo, rdx=a, rcx=b. mul leaves product in rdx:rax.
procedure Mul64x64(out hi, lo: UInt64; a, b: UInt64); assembler; nostackframe;
asm
  mov rax, rdx           // a
  mul rcx                // a * b -> rdx:rax (high:low)
  mov [rdi], rdx         // *hi = high
  mov [rsi], rax         // *lo = low
end;
{$ENDIF}
{$ELSE}
procedure Mul64x64(out hi, lo: UInt64; a, b: UInt64); inline;
var
  MulLo, Temp1, Temp2: UInt64;
begin
  MulLo := uint64(uint32(a)) * uint64(uint32(b));
  Temp1 := (a shr 32) * uint64(uint32(b)) + (MulLo shr 32);
  Temp2 := uint64(uint32(a)) * (b shr 32) + uint64(uint32(Temp1));
  lo := (Temp2 shl 32) or (MulLo and $FFFFFFFF);
  hi := (a shr 32) * (b shr 32) + (Temp1 shr 32) + (Temp2 shr 32);
end;
{$ENDIF}

// 192-bit unsigned subtract on the (m,h,l) triple of TInt64.
// Caller guarantees |a| >= |b| so there is no borrow out. ex/sgn untouched.
// Self-aliasing of r with a or b is safe (reads of a/b at offset N happen
// before writes to r at offset N).
{$IFDEF AVX2}
{$IFDEF MSWINDOWS}
// Win64 ABI: rcx=@r, rdx=@a, r8=@b. TInt64 layout: m@0, h@8, l@16.
procedure Sub192(out r: TInt64; const a, b: TInt64); assembler; nostackframe;
asm
  mov rax, [rdx+16]      // a.l
  sub rax, [r8+16]       // - b.l
  mov [rcx+16], rax      // r.l
  mov rax, [rdx]         // a.m
  sbb rax, [r8]          // - b.m - borrow
  mov [rcx], rax         // r.m
  mov rax, [rdx+8]       // a.h
  sbb rax, [r8+8]        // - b.h - borrow
  mov [rcx+8], rax       // r.h
end;
{$ELSE}
// SysV x86-64 ABI: rdi=@r, rsi=@a, rdx=@b. TInt64 layout: m@0, h@8, l@16.
procedure Sub192(out r: TInt64; const a, b: TInt64); assembler; nostackframe;
asm
  mov rax, [rsi+16]      // a.l
  sub rax, [rdx+16]      // - b.l
  mov [rdi+16], rax      // r.l
  mov rax, [rsi]         // a.m
  sbb rax, [rdx]         // - b.m - borrow
  mov [rdi], rax         // r.m
  mov rax, [rsi+8]       // a.h
  sbb rax, [rdx+8]       // - b.h - borrow
  mov [rdi+8], rax       // r.h
end;
{$ENDIF}
{$ELSE}
procedure Sub192(out r: TInt64; const a, b: TInt64); inline;
var
  th, tm, tl, borrow: UInt64;
begin
  tl := a.l - b.l;
  borrow := UInt64(b.l > a.l);
  tm := a.m - b.m - borrow;
  if a.m < b.m then borrow := 1
  else if (a.m = b.m) and (borrow = 1) then borrow := 1
  else borrow := 0;
  th := a.h - b.h - borrow;
  r.l := tl; r.m := tm; r.h := th;
end;
{$ENDIF}

// 192-bit unsigned add on the (m,h,l) triple of TInt64; returns 1 if the
// sum overflows out of the top bit, 0 otherwise. ex/sgn untouched.
// Self-aliasing of r with a or b is safe.
{$IFDEF AVX2}
{$IFDEF MSWINDOWS}
// Win64 ABI: rcx=@r, rdx=@a, r8=@b. Result returned in rax.
function Add192Cy(out r: TInt64; const a, b: TInt64): UInt64; assembler; nostackframe;
asm
  mov rax, [rdx+16]      // a.l
  add rax, [r8+16]       // + b.l
  mov [rcx+16], rax      // r.l
  mov rax, [rdx]         // a.m
  adc rax, [r8]          // + b.m + carry
  mov [rcx], rax         // r.m
  mov rax, [rdx+8]       // a.h
  adc rax, [r8+8]        // + b.h + carry
  mov [rcx+8], rax       // r.h
  setc al                // carry-out into AL
  movzx rax, al          // zero-extend to 64-bit Result (RAX)
end;
{$ELSE}
// SysV x86-64 ABI: rdi=@r, rsi=@a, rdx=@b. Result returned in rax.
function Add192Cy(out r: TInt64; const a, b: TInt64): UInt64; assembler; nostackframe;
asm
  mov rax, [rsi+16]      // a.l
  add rax, [rdx+16]      // + b.l
  mov [rdi+16], rax      // r.l
  mov rax, [rsi]         // a.m
  adc rax, [rdx]         // + b.m + carry
  mov [rdi], rax         // r.m
  mov rax, [rsi+8]       // a.h
  adc rax, [rdx+8]       // + b.h + carry
  mov [rdi+8], rax       // r.h
  setc al                // carry-out into AL
  movzx rax, al          // zero-extend to 64-bit Result (RAX)
end {$IFDEF FPC} ['rax']{$ENDIF};
{$ENDIF}
{$ELSE}
function Add192Cy(out r: TInt64; const a, b: TInt64): UInt64; inline;
var
  th, tm, tl, c: UInt64;
begin
  tl := a.l + b.l;
  c := UInt64(tl < a.l);
  tm := a.m + b.m + c;
  // carry-out of m: tm < a.m, OR (tm = a.m AND c = 1)
  if tm < a.m then c := 1
  else if (tm = a.m) and (c = 1) then c := 1
  else c := 0;
  th := a.h + b.h + c;
  if th < a.h then Result := 1
  else if (th = a.h) and (c = 1) then Result := 1
  else Result := 0;
  r.l := tl; r.m := tm; r.h := th;
end;
{$ENDIF}

// Ported from add_tint in tint.h.
procedure AddTInt(out r: TInt64; const a, b: TInt64); inline;
var
  pa, pb, ptmp, t: TInt64;
  sh: UInt64;
  ex, ex1: Integer;
  ch: UInt64;
  cmp: Integer;
begin
  pa := a; pb := b;  // local copies handle aliasing

  cmp := CmpTIntAbs(pa, pb);
  case cmp of
    0:
      begin
        if (pa.sgn xor pb.sgn) <> 0 then begin
          CpTInt(r, TINT_ZERO);
          Exit;
        end;
        CpTInt(r, pa);
        Inc(r.ex);
        Exit;
      end;
    -1:
      begin
        ptmp := pa; pa := pb; pb := ptmp;  // swap so |pa| >= |pb|
      end;
  end;

  // From here |pa| > |pb|, so pa.ex >= pb.ex
  sh := UInt64(pa.ex - pb.ex);
  // rshift writes only h/m/l; preserve t.ex and t.sgn (unused after).
  t.ex := 0; t.sgn := 0;
  if sh < 192 then RShiftTInt(t, pb, Integer(sh))
  else begin t.h := 0; t.m := 0; t.l := 0; end;

  if (pa.sgn xor pb.sgn) <> 0 then begin
    Sub192(t, pa, t);   // t := pa - t (no borrow out, |pa| > |pb|)
    ex := clz192(t.h, t.m, t.l);
    if (ex <= 1) or (sh = 0) then begin
      LShiftTInt(r, t, ex);
      r.ex := pa.ex - ex;
    end
    else begin
      // ex >= 2 and sh >= 1: redo with no neglected low bits of pb
      LShiftTInt(t, pb, ex - Integer(sh));
      LShiftTInt(r, pa, ex);
      Sub192(t, r, t);   // t := r - t
      ex1 := clz192(t.h, t.m, t.l);
      LShiftTInt(r, t, ex1);
      r.ex := pa.ex - (ex + ex1);
    end;
  end
  else begin
    // Same signs: 192-bit add into t, with ch = carry-out of bit 192.
    ch := Add192Cy(t, pa, t);
    if ch <> 0 then begin
      // 193-bit overflow: shift result right by 1, insert ch=1 at MSB of h.
      r.l := (t.m shl 63) or (t.l shr 1);
      r.m := (t.h shl 63) or (t.m shr 1);
      r.h := (ch shl 63) or (t.h shr 1);
      r.ex := pa.ex + 1;
    end
    else begin
      r.l := t.l;
      r.m := t.m;
      r.h := t.h;
      r.ex := pa.ex;
    end;
  end;
  r.sgn := pa.sgn;
end;

// Same body as AddTInt but the |a|>=|b| selection is done with pointers
// (no record copies on entry, no swap copy). Caller must ensure r does not
// alias a or b.
procedure AddTInt_NoAlias(out r: TInt64; const a, b: TInt64); inline;
var
  pa, pb: ^TInt64;
  t: TInt64;
  sh: UInt64;
  ex, ex1: Integer;
  ch: UInt64;
  cmp: Integer;
begin
  cmp := CmpTIntAbs(a, b);
  case cmp of
    0:
      begin
        if (a.sgn xor b.sgn) <> 0 then begin
          CpTInt(r, TINT_ZERO);
          Exit;
        end;
        CpTInt(r, a);
        Inc(r.ex);
        Exit;
      end;
    -1:
      begin pa := @b; pb := @a; end;
  else
    begin pa := @a; pb := @b; end;
  end;

  // From here |pa^| > |pb^|, so pa^.ex >= pb^.ex
  sh := UInt64(pa^.ex - pb^.ex);
  t.ex := 0; t.sgn := 0;
  if sh < 192 then RShiftTInt(t, pb^, Integer(sh))
  else begin t.h := 0; t.m := 0; t.l := 0; end;

  if (pa^.sgn xor pb^.sgn) <> 0 then begin
    Sub192(t, pa^, t);   // t := pa - t
    ex := clz192(t.h, t.m, t.l);
    if (ex <= 1) or (sh = 0) then begin
      LShiftTInt(r, t, ex);
      r.ex := pa^.ex - ex;
    end
    else begin
      LShiftTInt(t, pb^, ex - Integer(sh));
      LShiftTInt(r, pa^, ex);
      Sub192(t, r, t);
      ex1 := clz192(t.h, t.m, t.l);
      LShiftTInt(r, t, ex1);
      r.ex := pa^.ex - (ex + ex1);
    end;
  end
  else begin
    ch := Add192Cy(t, pa^, t);
    if ch <> 0 then begin
      r.l := (t.m shl 63) or (t.l shr 1);
      r.m := (t.h shl 63) or (t.m shr 1);
      r.h := (ch shl 63) or (t.h shr 1);
      r.ex := pa^.ex + 1;
    end
    else begin
      r.l := t.l;
      r.m := t.m;
      r.h := t.h;
      r.ex := pa^.ex;
    end;
  end;
  r.sgn := pa^.sgn;
end;

// Ported from mul_tint in tint.h. AVX2 path: a single asm block doing the
// six 64x64 partial products with chained ADC, then SHLD-based normalize.
{$IFDEF AVX2}
{$IFDEF MSWINDOWS}
procedure MulTInt(out r: TInt64; const a, b: TInt64); assembler; nostackframe;
asm
  // Win64 ABI: rcx=@r, rdx=@a, r8=@b. We need 8 live registers but Win64
  // exposes only 7 volatile ones, so we save rsi/rdi and reuse them as @a/@r
  // — that lets the body match the SysV variant exactly.
  // TInt64 layout: m@0, h@8, l@16, ex@24, sgn@32
  push rsi
  push rdi
  mov rdi, rcx                  // @r
  mov rsi, rdx                  // @a (rdx will be clobbered by mul)
  mov rcx, r8                   // @b (rcx as base for [rcx+N])

  xor r8, r8                    // rl_v = 0
  xor r9, r9                    // rm_v = 0
  xor r10, r10                  // rh_v = 0

  // 1) ah * bh  -> bits 256..383
  mov rax, [rsi+8]
  mul qword ptr [rcx+8]
  add r9, rax
  adc r10, rdx

  // 2) ah * bm  -> bits 192..319
  mov rax, [rsi+8]
  mul qword ptr [rcx+0]
  add r8, rax
  adc r9, rdx
  adc r10, 0

  // 3) am * bh  -> bits 192..319
  mov rax, [rsi+0]
  mul qword ptr [rcx+8]
  add r8, rax
  adc r9, rdx
  adc r10, 0

  // 4) ah * bl  -> bits 128..255 (only HIGH 64 contributes)
  mov rax, [rsi+8]
  mul qword ptr [rcx+16]
  add r8, rdx
  adc r9, 0
  adc r10, 0

  // 5) am * bm
  mov rax, [rsi+0]
  mul qword ptr [rcx+0]
  add r8, rdx
  adc r9, 0
  adc r10, 0

  // 6) al * bh
  mov rax, [rsi+16]
  mul qword ptr [rcx+8]
  add r8, rdx
  adc r9, 0
  adc r10, 0

  // r.ex = a.ex + b.ex; r.sgn = a.sgn xor b.sgn
  mov rax, [rsi+24]
  add rax, [rcx+24]
  mov rdx, [rsi+32]
  xor rdx, [rcx+32]
  mov [rdi+32], rdx

  // Normalize (left shift 1, dec ex) if top bit of r10 (rh_v) is 0.
  test r10, r10
  js @@no_norm
  shld r10, r9, 1
  shld r9, r8, 1
  shl r8, 1
  dec rax
@@no_norm:
  mov [rdi+24], rax             // r.ex
  mov [rdi+8],  r10             // r.h
  mov [rdi+0],  r9              // r.m
  mov [rdi+16], r8              // r.l

  pop rdi
  pop rsi
end {$IFDEF FPC} ['rax', 'rcx', 'rdx', 'r8', 'r9', 'r10', 'rdi', 'rsi']{$ENDIF};
{$ELSE}
procedure MulTInt(out r: TInt64; const a, b: TInt64); assembler; nostackframe;
asm
  // SysV ABI: rdi=@r, rsi=@a, rdx=@b
  // TInt64 layout: m@0, h@8, l@16, ex@24, sgn@32
  // mul writes rdx:rax, so move @b out of rdx first.
  mov rcx, rdx
  xor r8, r8                    // rl_v = 0  (bits 192..255 of conceptual product)
  xor r9, r9                    // rm_v = 0  (bits 256..319)
  xor r10, r10                  // rh_v = 0  (bits 320..383)

  // 1) ah * bh  -> bits 256..383
  mov rax, [rsi+8]              // ah
  mul qword ptr [rcx+8]         // rdx:rax = ah*bh
  add r9, rax
  adc r10, rdx

  // 2) ah * bm  -> bits 192..319
  mov rax, [rsi+8]
  mul qword ptr [rcx+0]
  add r8, rax
  adc r9, rdx
  adc r10, 0

  // 3) am * bh  -> bits 192..319
  mov rax, [rsi+0]
  mul qword ptr [rcx+8]
  add r8, rax
  adc r9, rdx
  adc r10, 0

  // 4) ah * bl  -> bits 128..255 (only HIGH 64 contributes; LOW 64 dropped)
  mov rax, [rsi+8]
  mul qword ptr [rcx+16]
  add r8, rdx
  adc r9, 0
  adc r10, 0

  // 5) am * bm
  mov rax, [rsi+0]
  mul qword ptr [rcx+0]
  add r8, rdx
  adc r9, 0
  adc r10, 0

  // 6) al * bh
  mov rax, [rsi+16]
  mul qword ptr [rcx+8]
  add r8, rdx
  adc r9, 0
  adc r10, 0

  // r.ex = a.ex + b.ex; r.sgn = a.sgn xor b.sgn
  mov rax, [rsi+24]
  add rax, [rcx+24]
  mov rdx, [rsi+32]
  xor rdx, [rcx+32]
  mov [rdi+32], rdx

  // Normalize (left shift 1, dec ex) if top bit of r10 (rh_v) is 0.
  test r10, r10
  js @@no_norm
  shld r10, r9, 1
  shld r9, r8, 1
  shl r8, 1
  dec rax
@@no_norm:
  mov [rdi+24], rax             // r.ex
  mov [rdi+8],  r10             // r.h
  mov [rdi+0],  r9              // r.m
  mov [rdi+16], r8              // r.l
end {$IFDEF FPC} ['rax', 'rcx', 'rdx', 'r8', 'r9', 'r10']{$ENDIF};
{$ENDIF}
{$ELSE}
procedure MulTInt(out r: TInt64; const a, b: TInt64); inline;
var
  ah, am, al, bh, bm, bl: UInt64;
  rh_hi, rh_lo, rm1_hi, rm1_lo, rm2_hi, rm2_lo: UInt64;
  rl1_hi, rl1_lo, rl2_hi, rl2_lo, rl3_hi, rl3_lo: UInt64;
  rh_v, rm_v, rl_v: UInt64;
  hh, lo, cm, sum_lo, sum_hi: UInt64;
  rex_a, rex_b: Int64;
  rsgn: UInt64;
begin
  rex_a := a.ex; rex_b := b.ex;
  rsgn := a.sgn xor b.sgn;
  ah := a.h; am := a.m; al := a.l;
  bh := b.h; bm := b.m; bl := b.l;

  Mul64x64(rh_hi,  rh_lo,  ah, bh);
  Mul64x64(rm1_hi, rm1_lo, ah, bm);
  Mul64x64(rm2_hi, rm2_lo, am, bh);
  Mul64x64(rl1_hi, rl1_lo, ah, bl);
  Mul64x64(rl2_hi, rl2_lo, am, bm);
  Mul64x64(rl3_hi, rl3_lo, al, bh);
  // rl1_lo, rl2_lo, rl3_lo are discarded; only the high words contribute.

  rh_v := rh_hi;
  rm_v := rh_lo;
  rl_v := rm1_lo;

  // Accumulate rm1's high part into rm_v (carry to rh_v)
  hh := rm1_hi;
  rm_v := rm_v + hh;
  if rm_v < hh then Inc(rh_v);

  // Accumulate rm2 (lo into rl_v with carry-out cm; hi into rm_v)
  lo := rm2_lo;
  hh := rm2_hi;
  rl_v := rl_v + lo;
  cm := UInt64(rl_v < lo);
  rm_v := rm_v + hh;
  if rm_v < hh then Inc(rh_v);

  // Accumulate (rl1_hi + rl2_hi + rl3_hi) into (rl_v, cm)
  sum_lo := rl1_hi;
  sum_hi := 0;
  sum_lo := sum_lo + rl2_hi;
  if sum_lo < rl2_hi then Inc(sum_hi);
  sum_lo := sum_lo + rl3_hi;
  if sum_lo < rl3_hi then Inc(sum_hi);
  cm := cm + sum_hi;
  rl_v := rl_v + sum_lo;
  if rl_v < sum_lo then Inc(cm);

  // Accumulate cm into rm_v (carry to rh_v)
  rm_v := rm_v + cm;
  if rm_v < cm then Inc(rh_v);

  r.ex := rex_a + rex_b;
  r.sgn := rsgn;
  if (rh_v shr 63) = 0 then begin
    // Normalize: shift left 1
    rh_v := (rh_v shl 1) or (rm_v shr 63);
    rm_v := (rm_v shl 1) or (rl_v shr 63);
    rl_v := rl_v shl 1;
    Dec(r.ex);
  end;
  r.h := rh_v; r.m := rm_v; r.l := rl_v;
end;
{$ENDIF}

// Ported from tint_fromd in tint.h. Defined for 0 (yields h=m=l=0).
procedure TIntFromD(out a: TInt64; b: Double); inline;
var
  u: Tb64u64;
  ax: UInt64;
  e: Int64;
  cnt: Integer;
begin
  u.f := b;
  a.sgn := u.u shr 63;
  ax := u.u and UInt64($7FFFFFFFFFFFFFFF);
  e := Int64(ax shr 52);
  if e <> 0 then begin
    a.ex := e - $3FE;
    a.h := (UInt64(1) shl 63) or (ax shl 11);
  end
  else begin
    cnt := clzll64(ax);
    a.ex := -$3F2 - cnt;
    if cnt < 64 then a.h := ax shl cnt else a.h := 0;
  end;
  a.m := 0; a.l := 0;
end;

// Ported from tint_tod in tint.h. Calls Math.Ldexp for the final exponent fold.
function TIntToD(const a: TInt64; err: UInt64; y, x: Double): Double; {$IFDEF FPC} inline; {$ENDIF}
const
  S: array[0..1] of Double = (1.0, -1.0);
var
  hh, mm, ll, low, notmm, notll: UInt64;
  ex_val: Int64;
  sh: Integer;
  hf, lf, sf: Double;
  worst: Boolean;
  mid: Boolean;
begin
  // Defined extension over the C: zero significand returns +/-0.0 cleanly.
  if (a.h = 0) and (a.m = 0) and (a.l = 0) then begin
    if a.sgn <> 0 then Result := -0.0 else Result := 0.0;
    Exit;
  end;
  if a.ex >= 1025 then begin
    // overflow
    if a.sgn <> 0 then Result := -1.7976931348623157e+308 - 1.7976931348623157e+308
    else Result := 1.7976931348623157e+308 + 1.7976931348623157e+308;
    Exit;
  end;
  if a.ex <= -1074 then begin
    if a.ex < -1074 then begin
      // |a| < 2^-1075 — round-to-nearest-even yields +/-0. The C source
      // computes `0x1p-1074 * 0.5` and relies on runtime IEEE rounding to
      // produce 0; FPC's compile-time constant folder rounds half-up to
      // 0x1p-1074 (smallest subnormal) instead, so we hard-code 0 here.
      if a.sgn <> 0 then Result := -0.0 else Result := 0.0;
      Exit;
    end;
    mid := (a.h = (UInt64(1) shl 63)) and (a.m = 0) and (a.l = 0);
    if a.sgn <> 0 then begin
      if mid then Result := -5e-324 * 0.5 else Result := -5e-324 * 0.75;
    end
    else begin
      if mid then Result := 5e-324 * 0.5 else Result := 5e-324 * 0.75;
    end;
    Exit;
  end;

  hh := a.h; mm := a.m; ll := a.l;
  ex_val := a.ex;
  low := hh and $7FF;
  notmm := not mm;
  notll := not ll;

  // Worst-case detection — we cannot determine correct rounding.
  if (mm = 0) or (notmm = 0) then begin
    worst :=
      ((mm = 0) and ((low = 0) or (low = $400)) and (ll < err)) or
      ((notmm = 0) and ((low = $3FF) or (low = $7FF)) and (notll < err));
    if worst then begin
      WriteLn('Unexpected worst-case found, please report to core-math@inria.fr:');
      WriteLn('Worst-case of atan2 found: y,x=', y, ',', x);
      Halt(1);
    end;
  end;

  if ex_val <= -1022 then begin
    sh := -1021 - ex_val;  // 1 <= sh <= 52
    ll := (mm shl (64 - sh)) or (ll shr sh) or UInt64(ll > 0);
    mm := (hh shl (64 - sh)) or (mm shr sh);
    hh := hh shr sh;
    low := hh and $7FF;
    ex_val := ex_val + sh;
  end;

  hf := Double(hh shr 11);  // 53-bit significand value
  if err = 0 then lf := 0.0
  else if low < $400 then lf := 0.25
  else if low > $400 then lf := 0.75
  else begin
    if (mm = 0) and (ll = 0) then lf := 0.5
    else lf := 0.75;
  end;

  sf := S[a.sgn];
  // h = fma(l, s, s*h) ; h *= 2^-52 ; result = h * 2^(ex_val-1)
  hf := lf * sf + sf * hf;
  hf := hf * 2.220446049250313e-16;  // 0x1p-52
  Result := hf * Math.Ldexp(1.0, Integer(ex_val - 1));
end;

// Ported from inv_tint in tint.h.
procedure InvTInt(out r: TInt64; const A: TInt64); inline;
var
  q: TInt64;
  ad: Double;
  subnormal: Boolean;
begin
  ad := TIntToD(A, 0, 0.0, 0.0);
  subnormal := Abs(ad) < cTI_1pm1022.f;
  if subnormal then ad := ad * cTI_1p53.f;
  TIntFromD(r, 1.0 / ad);
  if subnormal then Inc(r.ex, 53);
  MulTInt(q, A, r);
  q.sgn := 1 - q.sgn;
  AddTInt(q, TINT_ONE, q);
  MulTInt(q, r, q);
  AddTInt(r, r, q);
end;

// Ported from div_tint in tint.h.
procedure DivTInt(out r: TInt64; const b, a: TInt64); inline;
var
  Y, Z: TInt64;
begin
  InvTInt(Y, a);
  MulTInt(r, Y, b);
  MulTInt(Z, a, r);
  Z.sgn := 1 - Z.sgn;
  AddTInt(Z, b, Z);
  MulTInt(Z, Y, Z);
  AddTInt(r, r, Z);
end;

// Ported from div_tint_d in tint.h.
procedure DivTIntD(out r: TInt64; bd, ad: Double); inline;
var
  A, B: TInt64;
begin
  TIntFromD(A, ad);
  TIntFromD(B, bd);
  DivTInt(r, B, A);
end;

// ---------------------------------------------------------------------------
// qint64_t (256-bit) arithmetic — ported from core-math/src/binary64/pow/qint.h
// ---------------------------------------------------------------------------
// Internal representation: a 256-bit significand stored as (rh, rl) where
//   rh.hi = r0 (hh, most significant), rh.lo = r1 (hl),
//   rl.hi = r2 (lh),                   rl.lo = r3 (ll, least significant).
// Helpers below use (rh, rl) as TUInt128 pairs to mirror the C u128 code.

// 128-bit add returning carry (0 or 1)
function AddU128Cy(out r: TUInt128; const a, b: TUInt128): UInt64; inline;
var alo: UInt64;
begin
  alo := a.lo;
  r.lo := alo + b.lo;
  r.hi := a.hi + b.hi + UInt64(r.lo < alo);
  // Carry out of 128-bit add: result.hi < a.hi (or equal with low-half overflow)
  if (r.hi < a.hi) or ((r.hi = a.hi) and (r.lo < alo)) then
    Result := 1
  else
    Result := 0;
end;

// 128-bit subtract returning borrow (0 or 1)
function SubU128Bo(out r: TUInt128; const a, b: TUInt128): UInt64; inline;
var alo: UInt64;
begin
  alo := a.lo;
  r.lo := alo - b.lo;
  r.hi := a.hi - b.hi - UInt64(alo < b.lo);
  if (a.hi < b.hi) or ((a.hi = b.hi) and (alo < b.lo)) then
    Result := 1
  else
    Result := 0;
end;

// Decrement a 128-bit value (used after subu128 borrow)
procedure DecU128(var a: TUInt128); inline;
begin
  if a.lo = 0 then Dec(a.hi);
  Dec(a.lo);
end;

// Count leading zeros across a 128-bit value (returns 0..128)
function ClzU128(const a: TUInt128): Integer; inline;
begin
  if a.hi <> 0 then
    Result := clzll64(a.hi)
  else if a.lo <> 0 then
    Result := 64 + clzll64(a.lo)
  else
    Result := 128;
end;

procedure CpQInt(out r: TQInt64; const a: TQInt64); inline;
begin
  r := a;
end;

function QIntZeroP(const a: TQInt64): Boolean; inline;
begin
  // Matches C check `a->rh == 0` (top 128 bits zero ⇒ value zero, since
  // a normalised qint has MSB of r0 set).
  Result := (a.r0 = 0) and (a.r1 = 0);
end;

function CmpQIntAbs(const a, b: TQInt64): Integer; inline;
begin
  if a.ex > b.ex then begin Result := 1; Exit; end;
  if a.ex < b.ex then begin Result := -1; Exit; end;
  if a.r0 > b.r0 then begin Result := 1; Exit; end;
  if a.r0 < b.r0 then begin Result := -1; Exit; end;
  if a.r1 > b.r1 then begin Result := 1; Exit; end;
  if a.r1 < b.r1 then begin Result := -1; Exit; end;
  if a.r2 > b.r2 then begin Result := 1; Exit; end;
  if a.r2 < b.r2 then begin Result := -1; Exit; end;
  if a.r3 > b.r3 then begin Result := 1; Exit; end;
  if a.r3 < b.r3 then begin Result := -1; Exit; end;
  Result := 0;
end;

function CmpQIntAbs22(const a, b: TQInt64): Integer; inline;
begin
  if a.ex > b.ex then begin Result := 1; Exit; end;
  if a.ex < b.ex then begin Result := -1; Exit; end;
  if a.r0 > b.r0 then begin Result := 1; Exit; end;
  if a.r0 < b.r0 then begin Result := -1; Exit; end;
  if a.r1 > b.r1 then begin Result := 1; Exit; end;
  if a.r1 < b.r1 then begin Result := -1; Exit; end;
  Result := 0;
end;

// Ported from add_qint in qint.h.
procedure AddQInt(out r: TQInt64; const a, b: TQInt64); inline;
var
  pa, pb, ptmp: TQInt64;
  ah, al, bh, bl, ch, cl, t: TUInt128;
  m_ex, k: Int64;
  sgn: Byte;
  ex: UInt64;
  sh: Integer;
  cy: UInt64;
begin
  pa := a; pb := b;  // local copies handle aliasing

  if (pa.r0 = 0) and (pa.r1 = 0) then begin
    r := pb;
    Exit;
  end;
  if (pb.r0 = 0) and (pb.r1 = 0) then begin
    r := pa;
    Exit;
  end;

  case CmpQIntAbs(pa, pb) of
    0:
      begin
        if (pa.sgn xor pb.sgn) <> 0 then
          r := QINT_ZERO
        else begin
          r := pa;
          Inc(r.ex);
        end;
        Exit;
      end;
    -1:
      begin
        ptmp := pa; pa := pb; pb := ptmp;  // swap so |pa| >= |pb|
      end;
  end;

  // From now on, |pa| > |pb|
  ah.hi := pa.r0; ah.lo := pa.r1; al.hi := pa.r2; al.lo := pa.r3;
  bh.hi := pb.r0; bh.lo := pb.r1; bl.hi := pb.r2; bl.lo := pb.r3;

  m_ex := pa.ex;
  k := pa.ex - pb.ex;

  if k > 0 then begin
    if k >= 128 then begin
      if k < 256 then begin
        // bl := bh >> (k-128); bh := 0
        bl := bh; ShrU128(bl, Integer(k - 128));
        bh.hi := 0; bh.lo := 0;
      end else begin
        bl.hi := 0; bl.lo := 0;
        bh.hi := 0; bh.lo := 0;
      end;
    end else begin
      // 1 <= k <= 127: bl := (bl >> k) | (bh << (128-k)); bh := bh >> k
      // emulate via shifted copy of bh combined with shifted bl
      ShrU128(bl, Integer(k));
      t := bh;
      ShlU128(t, Integer(128 - k));
      bl.hi := bl.hi or t.hi;
      bl.lo := bl.lo or t.lo;
      ShrU128(bh, Integer(k));
    end;
  end;

  sgn := pa.sgn;
  r.ex := m_ex;

  if (pa.sgn xor pb.sgn) <> 0 then begin
    // subtraction case: C = A + (-B)
    SubU128(ch, ah, bh);
    if SubU128Bo(cl, al, bl) <> 0 then DecU128(ch);
    // |A| > |B| guarantees C <> 0
    ex := UInt64(ClzU128(ch));
    if ex = 128 then ex := 128 + UInt64(ClzU128(cl));
    // ex < 256

    if ex > 0 then begin
      // shift A by ex bits to the left, B by ex-k bits to the left
      if ex >= 128 then begin
        ah := al; ShlU128(ah, Integer(ex - 128));
        al.hi := 0; al.lo := 0;
      end else begin
        // 1 <= ex < 128
        t := al; ShrU128(t, Integer(128 - ex));
        ShlU128(ah, Integer(ex));
        ah.hi := ah.hi or t.hi;
        ah.lo := ah.lo or t.lo;
        ShlU128(al, Integer(ex));
      end;
      sh := Integer(ex) - Integer(k);
      bh.hi := pb.r0; bh.lo := pb.r1;
      bl.hi := pb.r2; bl.lo := pb.r3;
      if sh >= 0 then begin
        if sh >= 128 then begin
          bh := bl; ShlU128(bh, sh - 128);
          bl.hi := 0; bl.lo := 0;
        end else if sh > 0 then begin
          // 1 <= sh < 128
          t := bl; ShrU128(t, 128 - sh);
          ShlU128(bh, sh);
          bh.hi := bh.hi or t.hi;
          bh.lo := bh.lo or t.lo;
          ShlU128(bl, sh);
        end;
      end else begin
        sh := -sh;  // 0 < sh
        if sh >= 128 then begin
          bl := bh; ShrU128(bl, sh - 128);
          bh.hi := 0; bh.lo := 0;
        end else begin
          // 0 < sh < 128
          t := bh; ShlU128(t, 128 - sh);
          ShrU128(bl, sh);
          bl.hi := bl.hi or t.hi;
          bl.lo := bl.lo or t.lo;
          ShrU128(bh, sh);
        end;
      end;
      r.ex := r.ex - Int64(ex);
      SubU128(ch, ah, bh);
      if SubU128Bo(cl, al, bl) <> 0 then DecU128(ch);
      ex := UInt64(ClzU128(ch));
      if ex = 128 then ex := 128 + UInt64(ClzU128(cl));
    end;

    if ex <> 0 then begin
      // ch := (ch << ex) | (cl >> (128 - ex)); cl := cl << ex
      // ex is in 1..255 here.
      if ex < 128 then begin
        t := cl; ShrU128(t, Integer(128 - ex));
        ShlU128(ch, Integer(ex));
        ch.hi := ch.hi or t.hi;
        ch.lo := ch.lo or t.lo;
        ShlU128(cl, Integer(ex));
      end else begin
        // 128 <= ex < 256: top half becomes shifted cl, bottom becomes 0
        ch := cl; ShlU128(ch, Integer(ex - 128));
        cl.hi := 0; cl.lo := 0;
      end;
    end;
    r.ex := r.ex - Int64(ex);
  end else begin
    // addition case
    cy := AddU128Cy(ch, ah, bh);
    if AddU128Cy(cl, al, bl) <> 0 then begin
      // ++ch and check whether ch wrapped to zero (which would imply another carry)
      if (ch.lo = QWord($FFFFFFFFFFFFFFFF)) and (ch.hi = QWord($FFFFFFFFFFFFFFFF)) then begin
        ch.lo := 0; ch.hi := 0;
        Inc(cy);
      end else begin
        Inc(ch.lo);
        if ch.lo = 0 then Inc(ch.hi);
      end;
    end;

    if cy <> 0 then begin
      // cl := (ch << 127) | (cl >> 1)
      t := ch; ShlU128(t, 127);
      ShrU128(cl, 1);
      cl.hi := cl.hi or t.hi;
      cl.lo := cl.lo or t.lo;
      // ch := (1 << 127) | (ch >> 1)
      ShrU128(ch, 1);
      ch.hi := ch.hi or UInt64($8000000000000000);
      Inc(r.ex);
    end;
  end;

  r.sgn := sgn;
  r.r0 := ch.hi; r.r1 := ch.lo;
  r.r2 := cl.hi; r.r3 := cl.lo;
end;

// Ported from add_qint_22 in qint.h.
procedure AddQInt22(out r: TQInt64; const a, b: TQInt64); inline;
var
  pa, pb, ptmp: TQInt64;
  ah, bh, ch: TUInt128;
  m_ex: Int64;
  k: UInt64;
  sgn: Byte;
  ex: UInt64;
  cy: UInt64;
begin
  pa := a; pb := b;

  if (pa.r0 = 0) and (pa.r1 = 0) then begin r := pb; Exit; end;
  if (pb.r0 = 0) and (pb.r1 = 0) then begin r := pa; Exit; end;

  case CmpQIntAbs22(pa, pb) of
    0:
      begin
        if (pa.sgn xor pb.sgn) <> 0 then
          r := QINT_ZERO
        else begin
          r := pa;
          Inc(r.ex);
        end;
        Exit;
      end;
    -1:
      begin
        ptmp := pa; pa := pb; pb := ptmp;
      end;
  end;

  ah.hi := pa.r0; ah.lo := pa.r1;
  bh.hi := pb.r0; bh.lo := pb.r1;

  m_ex := pa.ex;
  k := UInt64(pa.ex - pb.ex);

  if k > 0 then begin
    if k >= 128 then begin
      bh.hi := 0; bh.lo := 0;
    end else
      ShrU128(bh, Integer(k));
  end;

  sgn := pa.sgn;
  r.ex := m_ex;

  if (pa.sgn xor pb.sgn) <> 0 then begin
    SubU128(ch, ah, bh);
    ex := UInt64(ClzU128(ch));  // < 128 since |A| > |B|

    if ex > 0 then begin
      ShlU128(ah, Integer(ex));
      bh.hi := pb.r0; bh.lo := pb.r1;
      if ex >= k then
        ShlU128(bh, Integer(ex - k))
      else
        ShrU128(bh, Integer(k - ex));
      r.ex := r.ex - Int64(ex);
      SubU128(ch, ah, bh);
      ex := UInt64(ClzU128(ch));
    end;
    ShlU128(ch, Integer(ex));
    r.ex := r.ex - Int64(ex);
  end else begin
    cy := AddU128Cy(ch, ah, bh);
    if cy <> 0 then begin
      ShrU128(ch, 1);
      ch.hi := ch.hi or UInt64($8000000000000000);
      Inc(r.ex);
    end;
  end;

  r.sgn := sgn;
  r.r0 := ch.hi; r.r1 := ch.lo;
  r.r2 := 0; r.r3 := 0;
end;

// Helper: compose (rh, rl) from t6 (128b) + lowt5 (64b) + lowt4 (64b),
// applying the ex (0/1) renormalisation step shared by all the mul_qint*
// variants. ulow5 = low 64 bits of t5 (post-shift), ulow4 = low 64 bits of t4.
procedure QMulFinish(var r: TQInt64;
                     const t6: TUInt128; ulow5, ulow4: UInt64;
                     a_ex, b_ex: Int64; a_sgn, b_sgn: Byte); inline;
var
  ex: UInt64;
  rh, rl: TUInt128;
begin
  // build t5 in 128 bits: (lowt5 in high half, ulow4 in low half)
  rh := t6;
  rl.hi := ulow5; rl.lo := ulow4;

  if (rh.hi and UInt64($8000000000000000)) <> 0 then
    ex := 0
  else
    ex := 1;

  if ex <> 0 then begin
    // shift left by 1
    rh.hi := (rh.hi shl 1) or (rh.lo shr 63);
    rh.lo := (rh.lo shl 1) or (rl.hi shr 63);
    rl.hi := (rl.hi shl 1) or (rl.lo shr 63);
    rl.lo := rl.lo shl 1;
  end;

  r.r0 := rh.hi; r.r1 := rh.lo;
  r.r2 := rl.hi; r.r3 := rl.lo;
  r.ex := a_ex + b_ex + 1 - Int64(ex);
  r.sgn := a_sgn xor b_sgn;
end;

// Ported from mul_qint in qint.h. Error < 14 ulps.
procedure MulQInt(out r: TQInt64; const a, b: TQInt64); inline;
var
  r33, r32, r23, r31, r13, r22, r30, r03, r21, r12: TUInt128;
  t6, t5, t4, t3: TUInt128;
  c5, c4: UInt64;
  tmp: TUInt128;
  ulow4: UInt64;
begin
  r33 := Mulu64u64(a.r0, b.r0);
  r32 := Mulu64u64(a.r0, b.r1);
  r23 := Mulu64u64(a.r1, b.r0);
  r31 := Mulu64u64(a.r0, b.r2);
  r13 := Mulu64u64(a.r2, b.r0);
  r22 := Mulu64u64(a.r1, b.r1);
  r30 := Mulu64u64(a.r0, b.r3);
  r03 := Mulu64u64(a.r3, b.r0);
  r21 := Mulu64u64(a.r1, b.r2);
  r12 := Mulu64u64(a.r2, b.r1);

  // t3 = (r12 >> 64) + (r21 >> 64) + (r03 >> 64) + (r30 >> 64)  (sum < 2^66)
  t3.hi := 0; t3.lo := r12.hi;
  Inc(t3.hi, Ord(t3.lo + r21.hi < t3.lo));
  t3.lo := t3.lo + r21.hi;
  Inc(t3.hi, Ord(t3.lo + r03.hi < t3.lo));
  t3.lo := t3.lo + r03.hi;
  Inc(t3.hi, Ord(t3.lo + r30.hi < t3.lo));
  t3.lo := t3.lo + r30.hi;

  // c4 = addu128(r22, t3, &t4); c4 += addu128(r13, t4, &t4); c4 += addu128(r31, t4, &t4);
  c4 := AddU128Cy(t4, r22, t3);
  c4 := c4 + AddU128Cy(t4, r13, t4);
  c4 := c4 + AddU128Cy(t4, r31, t4);

  // c5 = addu128(r23, t4 >> 64, &t5); c5 += addu128(r32, t5, &t5);
  tmp.lo := t4.hi; tmp.hi := 0;
  c5 := AddU128Cy(t5, r23, tmp);
  c5 := c5 + AddU128Cy(t5, r32, t5);

  // t6 = r33 + ((c5 << 64) | (t5 >> 64)) + c4
  tmp.hi := c5; tmp.lo := t5.hi;
  AddU128(t6, r33, tmp);
  tmp.hi := 0; tmp.lo := c4;
  AddU128(t6, t6, tmp);

  ulow4 := t4.lo;
  // t5 = (t5 << 64) | (low64(t4))
  t5.hi := t5.lo; t5.lo := ulow4;

  QMulFinish(r, t6, t5.hi, t5.lo, a.ex, b.ex, a.sgn, b.sgn);
end;

// Ported from mul_qint_33 in qint.h. Error < 6 ulps.
procedure MulQInt33(out r: TQInt64; const a, b: TQInt64); inline;
var
  r33, r32, r23, r31, r13, r22, r21, r12: TUInt128;
  t6, t5, t4, t3, tmp: TUInt128;
  c5, c4: UInt64;
  ulow4: UInt64;
begin
  r33 := Mulu64u64(a.r0, b.r0);
  r32 := Mulu64u64(a.r0, b.r1);
  r23 := Mulu64u64(a.r1, b.r0);
  r31 := Mulu64u64(a.r0, b.r2);
  r13 := Mulu64u64(a.r2, b.r0);
  r22 := Mulu64u64(a.r1, b.r1);
  r21 := Mulu64u64(a.r1, b.r2);
  r12 := Mulu64u64(a.r2, b.r1);

  // t3 = (r12 >> 64) + (r21 >> 64)
  t3.hi := 0; t3.lo := r12.hi;
  Inc(t3.hi, Ord(t3.lo + r21.hi < t3.lo));
  t3.lo := t3.lo + r21.hi;

  c4 := AddU128Cy(t4, r22, t3);
  c4 := c4 + AddU128Cy(t4, r13, t4);
  c4 := c4 + AddU128Cy(t4, r31, t4);

  tmp.lo := t4.hi; tmp.hi := 0;
  c5 := AddU128Cy(t5, r23, tmp);
  c5 := c5 + AddU128Cy(t5, r32, t5);

  tmp.hi := c5; tmp.lo := t5.hi;
  AddU128(t6, r33, tmp);
  tmp.hi := 0; tmp.lo := c4;
  AddU128(t6, t6, tmp);

  ulow4 := t4.lo;
  t5.hi := t5.lo; t5.lo := ulow4;
  QMulFinish(r, t6, t5.hi, t5.lo, a.ex, b.ex, a.sgn, b.sgn);
end;

// Ported from mul_qint_41 in qint.h. Error < 2 ulps.
procedure MulQInt41(out r: TQInt64; const a, b: TQInt64); inline;
var
  r33, r23, r13, r03: TUInt128;
  t6, t5, t4, t3, tmp: TUInt128;
  c5, c4: UInt64;
  ulow4: UInt64;
begin
  r33 := Mulu64u64(a.r0, b.r0);
  r23 := Mulu64u64(a.r1, b.r0);
  r13 := Mulu64u64(a.r2, b.r0);
  r03 := Mulu64u64(a.r3, b.r0);

  // t3 = r03 >> 64
  t3.hi := 0; t3.lo := r03.hi;

  c4 := AddU128Cy(t4, r13, t3);

  tmp.lo := t4.hi; tmp.hi := 0;
  c5 := AddU128Cy(t5, r23, tmp);

  tmp.hi := c5; tmp.lo := t5.hi;
  AddU128(t6, r33, tmp);
  tmp.hi := 0; tmp.lo := c4;
  AddU128(t6, t6, tmp);

  ulow4 := t4.lo;
  t5.hi := t5.lo; t5.lo := ulow4;
  QMulFinish(r, t6, t5.hi, t5.lo, a.ex, b.ex, a.sgn, b.sgn);
end;

// Ported from mul_qint_31 in qint.h. Exact (no error).
procedure MulQInt31(out r: TQInt64; const a, b: TQInt64); inline;
var
  r33, r23, r13: TUInt128;
  t6, t5, t4, tmp: TUInt128;
  c5: UInt64;
  ulow4: UInt64;
begin
  r33 := Mulu64u64(a.r0, b.r0);
  r23 := Mulu64u64(a.r1, b.r0);
  r13 := Mulu64u64(a.r2, b.r0);

  t4 := r13;

  tmp.lo := t4.hi; tmp.hi := 0;
  c5 := AddU128Cy(t5, r23, tmp);

  tmp.hi := c5; tmp.lo := t5.hi;
  AddU128(t6, r33, tmp);

  ulow4 := t4.lo;
  t5.hi := t5.lo; t5.lo := ulow4;
  QMulFinish(r, t6, t5.hi, t5.lo, a.ex, b.ex, a.sgn, b.sgn);
end;

// Ported from mul_qint_22 in qint.h. Exact.
procedure MulQInt22(out r: TQInt64; const a, b: TQInt64); inline;
var
  r33, r32, r23, r22: TUInt128;
  t6, t5, t4, tmp: TUInt128;
  c5: UInt64;
  ulow4: UInt64;
begin
  r33 := Mulu64u64(a.r0, b.r0);
  r32 := Mulu64u64(a.r0, b.r1);
  r23 := Mulu64u64(a.r1, b.r0);
  r22 := Mulu64u64(a.r1, b.r1);

  t4 := r22;

  tmp.lo := t4.hi; tmp.hi := 0;
  c5 := AddU128Cy(t5, r23, tmp);
  c5 := c5 + AddU128Cy(t5, r32, t5);

  tmp.hi := c5; tmp.lo := t5.hi;
  AddU128(t6, r33, tmp);

  ulow4 := t4.lo;
  t5.hi := t5.lo; t5.lo := ulow4;
  QMulFinish(r, t6, t5.hi, t5.lo, a.ex, b.ex, a.sgn, b.sgn);
end;

// Ported from mul_qint_21 in qint.h. Exact.
procedure MulQInt21(out r: TQInt64; const a, b: TQInt64); inline;
var
  r33, r23: TUInt128;
  t6, t5, tmp: TUInt128;
begin
  r33 := Mulu64u64(a.r0, b.r0);
  r23 := Mulu64u64(a.r1, b.r0);

  // t6 = r33 + (r23 >> 64)
  tmp.hi := 0; tmp.lo := r23.hi;
  AddU128(t6, r33, tmp);

  // t5 = r23 << 64  (only low 64 of r23 contribute as the new high half)
  t5.hi := r23.lo; t5.lo := 0;

  QMulFinish(r, t6, t5.hi, t5.lo, a.ex, b.ex, a.sgn, b.sgn);
end;

// Ported from mul_qint_11 in qint.h. Exact.
procedure MulQInt11(out r: TQInt64; const a, b: TQInt64); inline;
var
  t6: TUInt128;
  ex: UInt64;
begin
  t6 := Mulu64u64(a.r0, b.r0);
  if (t6.hi and UInt64($8000000000000000)) <> 0 then ex := 0 else ex := 1;
  if ex <> 0 then begin
    t6.hi := (t6.hi shl 1) or (t6.lo shr 63);
    t6.lo := t6.lo shl 1;
  end;
  r.r0 := t6.hi; r.r1 := t6.lo;
  r.r2 := 0; r.r3 := 0;
  r.ex := a.ex + b.ex + 1 - Int64(ex);
  r.sgn := a.sgn xor b.sgn;
end;

// Ported from mul_qint_2 in qint.h. Error < 2 ulps.
procedure MulQIntInt(out r: TQInt64; b: Int64; const a: TQInt64); inline;
var
  c: UInt64;
  k: Integer;
  t0, t1, t2, t3, t, tmp: TUInt128;
  cy: UInt64;
  ex: UInt32;
  ulow1: UInt64;
begin
  if b = 0 then begin
    r := QINT_ZERO;
    Exit;
  end;

  if b < 0 then c := UInt64(-b) else c := UInt64(b);
  if c = 1 then begin
    r := a;
    if b < 0 then r.sgn := r.sgn xor 1;
    Exit;
  end;

  if b < 0 then r.sgn := a.sgn xor 1 else r.sgn := a.sgn;
  r.ex := a.ex + 64;

  // scale c so that 2^63 <= c < 2^64
  k := clzll64(c);
  c := c shl k;
  r.ex := r.ex - Int64(k);

  t3 := Mulu64u64(a.r0, c);
  t2 := Mulu64u64(a.r1, c);
  t1 := Mulu64u64(a.r2, c);
  t0 := Mulu64u64(a.r3, c);

  // t = t0 >> 64
  t.hi := 0; t.lo := t0.hi;

  cy := AddU128Cy(t1, t, t1);
  // t = (cy << 64) | (t1 >> 64)
  tmp.hi := cy; tmp.lo := t1.hi;
  cy := AddU128Cy(t2, tmp, t2);
  // t3 += (cy << 64) | (t2 >> 64)
  tmp.hi := cy; tmp.lo := t2.hi;
  AddU128(t3, t3, tmp);

  ex := UInt32(clzll64(t3.hi));  // 0 or 1

  ulow1 := t1.lo;
  t2.hi := t2.lo; t2.lo := ulow1;

  if ex <> 0 then begin
    r.r0 := (t3.hi shl 1) or (t3.lo shr 63);
    r.r1 := (t3.lo shl 1) or (t2.hi shr 63);
    r.r2 := (t2.hi shl 1) or (t2.lo shr 63);
    r.r3 := t2.lo shl 1;
    Dec(r.ex);
  end else begin
    r.r0 := t3.hi; r.r1 := t3.lo;
    r.r2 := t2.hi; r.r3 := t2.lo;
  end;
end;

// Ported from qint_toi in pow.h. Truncates toward zero. ex < 0 → 0.
function QIntToI(const a: TQInt64): Int64; inline;
var
  r: UInt64;
begin
  if a.ex < 0 then begin
    QIntToI := 0;
    Exit;
  end;
  r := a.r0 shr (63 - a.ex);
  if a.sgn = 1 then QIntToI := -Int64(r) else QIntToI := Int64(r);
end;

{ TUInt128 }

{$IFNDEF FPC}
class operator TUInt128.Add(const a: TUInt128; b: UInt64): TUInt128;
begin
  Result.lo := a.lo + b;
  Result.hi := a.hi + UInt64(Result.lo < b);
end;
{$ENDIF}

end.
