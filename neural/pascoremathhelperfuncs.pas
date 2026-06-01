// pas-core-math - Pascal port of CORE-MATH
// https://github.com/joaopauloschuler/pas-core-math
//                                                                                                                                                                                                      
// Copyright (c) 2024-2026 Joao Paulo Schwarz Schuler and contributors.
// Refer to the git commit history for individual authorship.
// SPDX-License-Identifier: MIT
{$I pascoremath.inc}
unit pascoremathhelperfuncs;

interface

uses
  Math, SysUtils, pascoremathtypes;

// Fused multiply-add — hardware FMA3 on x86-64; 80-bit approximation elsewhere.
// Pure-asm body uses System V AMD64 ABI (params in xmm0/1/2).
function pcr_fmaf(x, y, z: Single): Single; {$IFNDEF AVX2} inline; {$ENDIF}
function pcr_fma(x, y, z: Double): Double; {$IFNDEF AVX2} inline; {$ENDIF}
function pcr_fma_pascal( a,b,c: Double ):Double; inline;

// Absolute value
function pcr_fabsf(x: Single): Single; inline;
function pcr_fabs(x: Double): Double; inline;

// Copy sign of y to magnitude of x
function pcr_copysignf(x, y: Single): Single; inline;
function pcr_copysign(x, y: Double): Double; inline;

// Square root
function pcr_sqrtf(x: Single): Single; inline;
function pcr_sqrt(x: Double): Double; inline;

// Round to nearest even integer — SSE4.1 ROUNDSD/ROUNDSS on x86-64; bit-manip elsewhere.
// Pure-asm body uses System V AMD64 ABI (param in xmm0, result in xmm0).
function pcr_roundevenf(x: Single): Single; {$IFNDEF AVX2} inline; {$ENDIF}
function pcr_roundeven(x: Double): Double; {$IFNDEF AVX2} inline; {$ENDIF}

// NaN-aware maximum — MAXSS/MAXSD on x86-64; branch fallback elsewhere.
// Pure-asm body uses System V AMD64 ABI (params in xmm0/xmm1).
function pcr_fmaxf(x, y: Single): Single; {$IFNDEF AVX2} inline; {$ENDIF}
function pcr_fmax(x, y: Double): Double; {$IFNDEF AVX2} inline; {$ENDIF}

// NaN-aware minimum — MINSS/MINSD on x86-64; branch fallback elsewhere.
// Pure-asm body uses System V AMD64 ABI (params in xmm0/xmm1).
function pcr_fminf(x, y: Single): Single; {$IFNDEF AVX2} inline; {$ENDIF}
function pcr_fmin(x, y: Double): Double; {$IFNDEF AVX2} inline; {$ENDIF}

// Return a NaN (tagp is ignored, matches C nan()/nanf() signature)
function pcr_nanf(const tagp: PAnsiChar): Single; inline;
function pcr_nan(const tagp: PAnsiChar): Double; inline;

// Raise floating-point exceptions
function pcr_feraiseexcept_invalid():Single; inline;
function pcr_feraiseexcept_divbyzero():Single; inline;

// Count leading zeros in a 64-bit value (__builtin_clzll equivalent).
// Result is undefined when x = 0.
function pcr_clzll(x: UInt64): Integer; inline;

// Current FPU rounding mode (wraps Math.GetRoundMode for use in ported functions).
function pcr_GetRoundMode: TFPURoundingMode; inline;

// ------- binary64 bit-pattern helpers (task 0.10) -------
// Double equivalents of the cf_*/isint_pf/isodd_pf helpers in pascoremath32.pas.

// Detect signaling NaN via bit-flip at $0008000000000000.
function pcr_is_signaling(x: Double): Boolean; inline;

// Returns true if y is an exact integer (general use).
function pcr_isint_d(y: Double): Boolean; inline;

// pow-variant of integer test (same logic as pcr_isint_d).
function pcr_isint_pd(y: Double): Boolean; inline;

// pow-variant of odd-integer test.
function pcr_isodd_pd(y: Double): Boolean; inline;

// pow-variant of signaling-NaN test.
function pcr_is_signaling_pd(x: Double): Boolean; inline;

// Double x double-double multiply: returns (xh+xl)*ch with error in l.
function pcr_mulddd_pd(xh, xl, ch: Double; out l: Double): Double; inline;

// MXCSR flag save/restore (AVX2: hardware register; otherwise no-op).
function pcr_get_mxcsr: DWord;
procedure pcr_set_mxcsr(flag: DWord);

// ------- double-double and polynomial helpers (task 0.9, promoted from pascoremath32) -------

// Degree-12 polynomial evaluator (used by acosf, asinf and their binary64 analogues).
function pcr_poly12(z: Double; const c: array of Double): Double; inline;

// Double-double × double-double product: returns xh*ch + mixed terms, error in l.
function pcr_muldd(xh, xl, ch, cl: Double; out l: Double): Double; inline;

// Horner evaluation of a flat-array double-double polynomial.
// c is flat: c[k*2] = high part, c[k*2+1] = low part.
function pcr_polydd(xh, xl: Double; n: Int32; const c: array of Double; out l: Double): Double; inline;

// All four primitives below write their var outputs LAST (after all value-param
// reads) so that callers may safely alias value params with var params.

// Error-free sum: s + t = a + b exactly.
procedure pcr_fasttwosum(out s, t: Double; a, b: Double); inline;

// Error-free product: hi + lo = a * b exactly (via FMA).
procedure pcr_a_mul(out hi, lo: Double; a, b: Double); inline;

// Scalar × double-double: (hi + lo) = a * (bh + bl).
procedure pcr_s_mul(out hi, lo: Double; a, bh, bl: Double); inline;

// Double-double × double-double: (hi + lo) = (ah + al) * (bh + bl).
procedure pcr_d_mul(out hi, lo: Double; ah, al, bh, bl: Double); inline;

implementation

function pcr_fmaf(x, y, z: Single): Single;
{$IFDEF AVX2}
// Pure-asm: System V AMD64 ABI passes x→xmm0, y→xmm1, z→xmm2; result in xmm0.
// VFMADD213SS: xmm0 = xmm0 * xmm1 + xmm2  (correctly rounded IEEE 754 FMA).
assembler;
asm
  vfmadd213ss xmm0, xmm1, xmm2
end ['xmm0'];
{$ELSE}
begin
  // 80-bit fallback: correctly rounded for singles (Extended has enough mantissa bits).
  Result := Single(Extended(x) * Extended(y) + Extended(z));
end;
{$ENDIF}

// =============================================================================
// Correctly rounded FMA3 emulation
// Coded by MathMan in the forum: https://forum.lazarus.freepascal.org/index.php/topic,73881.30.html
//
// The implementation is based on https://hal.science/hal-04575249/document
// ============================================================================= 
type
  DW = record
    h: Double;
    l: Double;
  end;

function split( x: Double ):DW;
const
  K: Double = Double(134217729.0);
var
  splittedx: DW;
  gamma: Double;
begin
  gamma := K * x;
  splittedx.h := ( gamma+( x-gamma ) );
  splittedx.l := x - splittedx.h;
  Result := splittedx;
end;

function DekkerProd( a,b: Double ):DW;
var
  splitteda: DW;
  splittedb: DW;
  product: DW;
begin
  splitteda := split( a );
  splittedb := split( b );
  product.h := a * b;
  product.l := ((( -product.h+splitteda.h*splittedb.h )+( splitteda.h*splittedb.l ))
            + splitteda.l*splittedb.h ) + splitteda.l*splittedb.l;
  Result := product;
end;

function TwoSum( a,b: Double ):DW;
var
  z: DW;
  aprime: Double;
begin
  z.h := a + b;
  aprime := z.h - b;
  z.l := ( a-aprime ) + ( b-( z.h-aprime ) );

  Result := z;
end;

function IsNot1or3timesPowerOf2( x: Double ):Boolean;
const
  P: Double = Double(2251799813685249.0);
  Q: Double = Double(2251799813685248.0);
var
  Delta: Double;
begin
  Delta := ( P*x ) - ( Q*x );
  Result := ( Delta<>x );
end;

// Boldo-Melquiond intermediates: after this, a*b + c = s.h + v.h + v.l exactly.
procedure pcr_fma_decompose(a, b, c: Double; out sh, vh, vl: Double);
var x, s, v: DW;
begin
  x := DekkerProd(a, b);
  s := TwoSum(x.h, c);
  v := TwoSum(x.l, s.l);
  sh := s.h; vh := v.h; vl := v.l;
end;

function pcr_fma_pascal_core( a,b,c: Double ):Double;
var
  sh, vh, vl: Double;
begin
  pcr_fma_decompose(a, b, c, sh, vh, vl);
  if IsNot1or3timesPowerOf2(vh) or (vl = 0) then
    Result := sh + vh
  else if (UInt8(vl < 0) xor UInt8(vh < 0)) <> 0 then
    Result := sh + (0.875 * vh)
  else
    Result := sh + (1.125 * vh);
end;

// FMA via integer mantissa arithmetic — the fallback used whenever the Dekker
// product in the Boldo-Melquiond core would be lossy, i.e. whenever a*b
// rounded to double would be subnormal. Handles both subnormal and normal
// outputs; the subnormal-output case arises when c is also subnormal-scale
// (atanpi's AtanpiSmall path), and the normal-output case arises when c is
// a small normal dominating the subnormal product.
//
// Algorithm:
//   1. Extract (sign, 53-bit integer mantissa, binary exp) from a, b, c.
//   2. Compute mp = ma * mb as an exact TUInt128 (106 bits).
//   3. Align mp and mc at a common fine scale 2^(ec - G). mc is left-shifted
//      by G; mp is shifted by (ep - ec + G) — right-shift collects a sticky
//      bit for bits dropped below the common scale.
//   4. Add or subtract (by relative signs).
//   5. Normalize the 128-bit integer back to a double: find the leading bit,
//      determine biased exponent, right-shift to 53-bit mantissa with round
//      and sticky, apply round-to-nearest-even, rebuild the IEEE bit pattern.
function pcr_fma_integer(a, b, c: Double): Double;
const
  G = 20;  // guard bits below scale 2^ec — fits comfortably in TUInt128
  EXP_MASK_D  : UInt64 = $7FF0000000000000;
  MANT_MASK_D : UInt64 = $000FFFFFFFFFFFFF;
  IMPLICIT_D  : UInt64 = $0010000000000000;
  SIGN_MASK_D : UInt64 = QWord($8000000000000000);
var
 va, vb, vc: Tb64u64;
  ma, mb, mc: UInt64;
  ea, eb, ec, ep: Int32;
  sa, sb, sc, sp, sr: Boolean;
  aNaN, aZ, bNaN, bZ, cNaN, cZ: Boolean;
  mp, mc_wide, sum: TUInt128;
  shift_p, shift_c: Integer;
  sticky_extra: Boolean;

  procedure ExtractParts(u: UInt64; out m: UInt64; out e: Int32; out s: Boolean; out isNanInf, isZero: Boolean);
  var b: UInt64;
  begin
    s := (u and SIGN_MASK_D) <> 0;
    b := (u and EXP_MASK_D) shr 52;
    isNanInf := b = $7FF;
    if b = 0 then begin
      m := u and MANT_MASK_D;
      isZero := m = 0;
      e := -1074;
    end else begin
      m := (u and MANT_MASK_D) or IMPLICIT_D;
      isZero := False;
      e := Int32(b) - 1023 - 52;
    end;
  end;

  procedure RShiftCollectSticky(var x: TUInt128; sh: Integer; var sticky: Boolean);
  begin
    if sh <= 0 then Exit;
    if sh >= 128 then begin
      if (x.lo <> 0) or (x.hi <> 0) then sticky := True;
      x.lo := 0; x.hi := 0;
      Exit;
    end;
    if sh < 64 then begin
      if (x.lo and ((UInt64(1) shl sh) - 1)) <> 0 then sticky := True;
      x.lo := (x.lo shr sh) or (x.hi shl (64 - sh));
      x.hi := x.hi shr sh;
    end else if sh = 64 then begin
      if x.lo <> 0 then sticky := True;
      x.lo := x.hi;
      x.hi := 0;
    end else begin
      if (x.lo <> 0) or ((x.hi and ((UInt64(1) shl (sh - 64)) - 1)) <> 0) then sticky := True;
      x.lo := x.hi shr (sh - 64);
      x.hi := 0;
    end;
  end;

  function U128GE(const x, y: TUInt128): Boolean;
  begin
    if x.hi <> y.hi then Result := x.hi > y.hi
    else Result := x.lo >= y.lo;
  end;

  function FallbackCore: Double;
  begin Result := pcr_fma_pascal_core(a, b, c); end;

  // Round 128-bit |sum| (at scale 2^(ec - G)) to a double with sign sr,
  // using sticky_extra as an additional below-the-guard sticky bit.
  function FinalizeDouble: Double;
  var
    k: Integer;           // leading-bit position of sum (0..127)
    e_unbiased: Int32;    // unbiased exponent of the result
    shift_right: Integer; // bits to shift out to form 53-bit mantissa
    keep_bits: Integer;   // bits of sum that survive the shift
    round_bit, sticky_bit: Boolean;
    lost_mask: UInt64;
    out_mant: UInt64;
    final_biased: Int32;
    r: Tb64u64;
  begin
    if (sum.hi = 0) and (sum.lo = 0) then begin
      // Exact zero (ignoring sticky_extra which represents <1-ulp residue).
      if not sticky_extra then begin
        r.u := 0;
        if sr then r.u := r.u or SIGN_MASK_D;
        Result := r.f; Exit;
      end;
      // Sub-guard positive residue with empty top: magnitude < 2^-1074, rounds to zero.
      // In round-to-nearest-even, magnitude 0..ulp/2 rounds to 0.
      r.u := 0;
      if sr then r.u := r.u or SIGN_MASK_D;
      Result := r.f; Exit;
    end;

    // Leading bit k (0-indexed).
    if sum.hi <> 0 then k := 127 - pcr_clzll(sum.hi)
    else                k := 63  - pcr_clzll(sum.lo);

    // Value = sum * 2^(ec - G) has magnitude in [2^(k + ec - G), 2^(k + 1 + ec - G)).
    e_unbiased  := k + ec - G;
    final_biased := e_unbiased + 1023;

    if final_biased >= 1 then begin
      // Normal result: mantissa needs to be 53-bit with leading 1 at bit 52.
      shift_right := k - 52;
    end else begin
      // Subnormal result: target grid is 2^-1074, so bit alignment = -1074 - (ec - G) = G - ec - 1074.
      // (This equals shift_right we'd want to apply so sum shr shift_right lands on the subnormal grid.)
      shift_right := G - ec - 1074;
      if shift_right < 0 then shift_right := 0;
    end;

    // Extract round/sticky bits from the bits about to be shifted out.
    if shift_right <= 0 then begin
      round_bit  := False;
      sticky_bit := sticky_extra;
      keep_bits  := 0;
    end else begin
      keep_bits := shift_right;
      // round_bit = bit (keep_bits - 1) of sum
      if keep_bits <= 64 then
        round_bit := ((sum.lo shr (keep_bits - 1)) and 1) <> 0
      else
        round_bit := ((sum.hi shr (keep_bits - 1 - 64)) and 1) <> 0;
      // sticky_bit = OR of bits [0 .. keep_bits-2]
      sticky_bit := sticky_extra;
      if keep_bits - 1 > 0 then begin
        if keep_bits - 1 <= 64 then begin
          lost_mask := (UInt64(1) shl (keep_bits - 1)) - 1;
          if (sum.lo and lost_mask) <> 0 then sticky_bit := True;
        end else begin
          if sum.lo <> 0 then sticky_bit := True;
          lost_mask := (UInt64(1) shl (keep_bits - 1 - 64)) - 1;
          if (sum.hi and lost_mask) <> 0 then sticky_bit := True;
        end;
      end;
    end;

    // Shift right by shift_right to get the to-be-rounded mantissa in sum.lo.
    if shift_right > 0 then begin
      if shift_right < 64 then begin
        sum.lo := (sum.lo shr shift_right) or (sum.hi shl (64 - shift_right));
        sum.hi := sum.hi shr shift_right;
      end else if shift_right = 64 then begin
        sum.lo := sum.hi;
        sum.hi := 0;
      end else begin
        if shift_right >= 128 then begin sum.lo := 0; sum.hi := 0; end
        else begin
          sum.lo := sum.hi shr (shift_right - 64);
          sum.hi := 0;
        end;
      end;
    end;

    if sum.hi <> 0 then begin Result := FallbackCore; Exit; end;  // should not happen
    out_mant := sum.lo;

    // Round-to-nearest-even.
    if round_bit and (sticky_bit or ((out_mant and 1) <> 0)) then
      out_mant := out_mant + 1;

    if final_biased >= 1 then begin
      // Normal. Rounding may have bumped mantissa from (1<<53)-1 to (1<<53), promoting exponent.
      if (out_mant shr 53) <> 0 then begin
        out_mant := out_mant shr 1;
        Inc(final_biased);
      end;
      if final_biased >= 2047 then begin
        // Overflow -> infinity
        r.u := UInt64($7FF) shl 52;
      end else begin
        r.u := (UInt64(final_biased) shl 52) or (out_mant and MANT_MASK_D);
      end;
    end else begin
      // Subnormal result. If rounding promoted to 2^52, it became smallest normal.
      if (out_mant shr 52) <> 0 then
        r.u := UInt64($0010000000000000)          // smallest normal
      else
        r.u := out_mant;                          // subnormal mantissa
    end;
    if sr then r.u := r.u or SIGN_MASK_D;
    Result := r.f;
  end;

begin
  Result := 0;
  va.f := a; vb.f := b; vc.f := c;

  ExtractParts(va.u, ma, ea, sa, aNaN, aZ);
  ExtractParts(vb.u, mb, eb, sb, bNaN, bZ);
  ExtractParts(vc.u, mc, ec, sc, cNaN, cZ);
  if aNaN or bNaN or cNaN then begin Result := FallbackCore; Exit; end;
  if aZ or bZ then begin Result := c; Exit; end;
  if cZ then begin
    // a*b in subnormal range with c=0: result IS a*b, potentially subnormal.
    // Rare for our callers; defer to core.
    Result := FallbackCore; Exit;
  end;

  mp := Mulu64u64(ma, mb);
  ep := ea + eb;
  sp := sa xor sb;

  // Align both at scale 2^(ec - G). mc gets a pure left-shift by G bits.
  // mp shifts by (ep - ec + G): may be positive (left) or negative (right + sticky).
  shift_p := ep - ec + G;
  sticky_extra := False;
  if shift_p > 0 then begin
    // mp is up to 106 bits. Require shift_p + 106 <= 128 -> shift_p <= 22.
    if shift_p > 22 then begin Result := FallbackCore; Exit; end;
    ShlU128(mp, shift_p);
  end else if shift_p < 0 then
    RShiftCollectSticky(mp, -shift_p, sticky_extra);

  shift_c := G;
  mc_wide.lo := mc;
  mc_wide.hi := 0;
  if shift_c > 0 then ShlU128(mc_wide, shift_c);  // mc is 53-bit, 53+G <= 128 for G<=75

  if sp = sc then begin
    AddU128(sum, mp, mc_wide);
    sr := sp;
  end else begin
    if U128GE(mp, mc_wide) then begin
      SubU128(sum, mp, mc_wide);
      sr := sp;
    end else begin
      // Subtrahend (mp) has lost tail bits represented by sticky_extra. True
      // value = mc_wide - (mp_stored + epsilon) = (sum - epsilon) where
      // epsilon ∈ [0, 1) in our integer units at scale 2^(ec-G).
      // Model this as (sum - 1) + (1 - epsilon): sum-1 is the integer part and
      // (1 - epsilon) ∈ (0, 1) lives below the LSB, so it only contributes a
      // sticky bit at finalization. Decrement sum by 1 with borrow; sticky
      // stays True.
      SubU128(sum, mc_wide, mp);
      sr := sc;
      if sticky_extra then begin
        if sum.lo = 0 then begin
          sum.lo := UInt64($FFFFFFFFFFFFFFFF);
          if sum.hi = 0 then begin
            // sum was exactly 0 before the borrow; that means mc_wide == mp
            // in the TUInt128 part and epsilon > 0, so true value is negative
            // of sign sc i.e. sign = sp. Magnitude is epsilon at scale 2^(ec-G),
            // less than half a subnormal-ulp. Round-to-nearest-even gives +/-0.
            sum.lo := 0;
            sum.hi := 0;
            sr := sp;
            sticky_extra := False;  // already consumed
          end else begin
            sum.hi := sum.hi - 1;
          end;
        end else begin
          sum.lo := sum.lo - 1;
        end;
        // sticky_extra remains True → FinalizeDouble will OR it into sticky_bit.
      end;
    end;
  end;

  Result := FinalizeDouble;
end;

function pcr_fma_pascal( a,b,c: Double ):Double; inline;
const
  cFmaDblMin: Tb64u64 = (u:$0010000000000000);  // 2^-1022 = DBL_MIN
var
  ab: Double;
begin
  // Boldo-Melquiond's Dekker product rounds a*b to a double; when that rounded
  // value is subnormal the low word can no longer recover the discarded bits
  // and the whole algorithm loses precision. Redirect those cases to the
  // integer-mantissa path.
  ab := a * b;
  if (ab <> 0.0) and (Abs(ab) < cFmaDblMin.f) and
     (ab = ab) and (ab - ab = 0.0) then
  begin
    Result := pcr_fma_integer(a, b, c);
    Exit;
  end;
  Result := pcr_fma_pascal_core(a, b, c);
end;

// =============================================================================
// END OF Correctly rounded FMA3 emulation
// =============================================================================

function pcr_fma(x, y, z: Double): Double;
{$IFDEF AVX2}
// Pure-asm: System V AMD64 ABI passes x→xmm0, y→xmm1, z→xmm2; result in xmm0.
// VFMADD213SD: xmm0 = xmm0 * xmm1 + xmm2  (correctly rounded IEEE 754 FMA).
assembler;
asm
  vfmadd213sd xmm0, xmm1, xmm2
end ['xmm0'];
{$ELSE}
begin
  // 80-bit fallback (double-rounding — not true FMA; may lose 1 ULP in rare cases).
  // Result := Double(Extended(x) * Extended(y) + Extended(z));
  Result := pcr_fma_pascal(x, y, z);
end;
{$ENDIF}

function pcr_fabsf(x: Single): Single;
begin
  if x >= Single(0.0) then Result := x else Result := -x;
end;

function pcr_fabs(x: Double): Double;
begin
  if x >= Single(0.0) then Result := x else Result := -x;
end;

function pcr_copysignf(x, y: Single): Single;
var
  vx, vy: Tb32u32;
begin
  vx.f := x;
  vy.f := y;
  vx.u := (vx.u and UInt32($7FFFFFFF)) or (vy.u and UInt32($80000000));
  Result := vx.f;
end;

function pcr_copysign(x, y: Double): Double;
var
  vx, vy: Tb64u64;
begin
  vx.f := x;
  vy.f := y;
  vx.u := (vx.u and UInt64($7FFFFFFFFFFFFFFF)) or (vy.u and UInt64($8000000000000000));
  Result := vx.f;
end;

function pcr_sqrtf(x: Single): Single;
begin
  Result := Sqrt(x);
end;

function pcr_sqrt(x: Double): Double;
begin
  Result := Sqrt(x);
end;

function pcr_roundevenf(x: Single): Single;
{$IFDEF AVX2}
// Pure-asm: System V AMD64 ABI passes x→xmm0; result in xmm0.
// ROUNDSS imm8=12 (0x0C): override MXCSR with round-to-nearest-even, suppress PE.
assembler;
asm
  roundss xmm0, xmm0, 12
end ['xmm0'];
{$ELSE}
// Portable fallback: round to nearest even using bit manipulation.
// For |x| >= 2^23 the value is already an integer.
var
  v: Tb32u32;
  e, shift: Int32;
  mask, frac, half: UInt32;
begin
  v.f := x;
  e := Int32((v.u shr 23) and $FF) - 127;  // unbiased exponent
  if e >= 23 then
  begin
    // Already an integer (or inf/nan)
    Result := x;
    Exit;
  end;
  if e < 0 then
  begin
    // |x| < 1: round to 0 or +-1
    if e = -1 then
    begin
      // |x| in [0.5, 1): round to nearest even => 0 if exactly 0.5, else +-1
      // Check if it's exactly +-0.5
      if (v.u and $7FFFFFFF) = $3F000000 then
        Result := 0.0  // exact half => round to even (0)
      else if Abs(x) < 0.5 then
        Result := 0.0
      else
        Result := pcr_copysignf(1.0, x);
    end
    else
      Result := 0.0;
    Exit;
  end;
  // e in [0, 22]: some fractional bits present
  shift := 23 - e;                    // number of fractional bits
  mask  := (1 shl shift) - 1;         // mask for fractional bits
  frac  := v.u and mask;
  half  := 1 shl (shift - 1);         // 0.5 in fractional position

  if frac < half then
  begin
    // Round down: clear fractional bits
    v.u := v.u and (not mask);
  end
  else if frac > half then
  begin
    // Round up
    v.u := (v.u and (not mask)) + (1 shl shift);
  end
  else
  begin
    // Exactly halfway: round to even (check integer bit)
    if (v.u and (1 shl shift)) <> 0 then
      // Int32 part is odd => round up
      v.u := (v.u and (not mask)) + (1 shl shift)
    else
      // Int32 part is even => round down
      v.u := v.u and (not mask);
  end;
  Result := v.f;
end;
{$ENDIF}

function pcr_roundeven(x: Double): Double;
{$IFDEF AVX2}
// Pure-asm: System V AMD64 ABI passes x→xmm0; result in xmm0.
// ROUNDSD imm8=12 (0x0C): override MXCSR with round-to-nearest-even, suppress PE.
assembler;
asm
  roundsd xmm0, xmm0, 12
end ['xmm0'];
{$ELSE}
// Portable fallback: round to nearest even using bit manipulation.
var
  v: Tb64u64;
  e, shift: Int32;
  mask, half: UInt64;
  frac: UInt64;
begin
  v.f := x;
  e := Int32((v.u shr 52) and $7FF) - 1023;
  if e >= 52 then begin Result := x; Exit; end;
  if e < 0 then begin
    if e = -1 then begin
      if (v.u and $7FFFFFFFFFFFFFFF) = $3FE0000000000000 then Result := 0.0
      else if Abs(x) < 0.5 then Result := 0.0
      else Result := pcr_copysign(1.0, x);
    end else Result := 0.0;
    Exit;
  end;
  shift := 52 - e;
  mask  := (UInt64(1) shl shift) - 1;
  frac  := v.u and mask;
  half  := UInt64(1) shl (shift - 1);
  if frac < half then v.u := v.u and (not mask)
  else if frac > half then v.u := (v.u and (not mask)) + (UInt64(1) shl shift)
  else begin
    if (v.u and (UInt64(1) shl shift)) <> 0 then
      v.u := (v.u and (not mask)) + (UInt64(1) shl shift)
    else v.u := v.u and (not mask);
  end;
  Result := v.f;
end;
{$ENDIF}

function pcr_fmaxf(x, y: Single): Single;
{$IFDEF AVX2}
// Pure-asm: System V AMD64 ABI passes x→xmm0, y→xmm1; result in xmm0.
// MAXSS returns the larger value; if x (first operand) is NaN, returns y.
assembler;
asm
  maxss xmm0, xmm1
end ['xmm0'];
{$ELSE}
begin
  if IsNan(x) then Result := y
  else if IsNan(y) then Result := x
  else if x > y then Result := x
  else Result := y;
end;
{$ENDIF}

function pcr_fmax(x, y: Double): Double;
{$IFDEF AVX2}
// Pure-asm: System V AMD64 ABI passes x→xmm0, y→xmm1; result in xmm0.
// MAXSD returns the larger value; if x (first operand) is NaN, returns y.
assembler;
asm
  maxsd xmm0, xmm1
end ['xmm0'];
{$ELSE}
begin
  if IsNan(x) then Result := y
  else if IsNan(y) then Result := x
  else if x > y then Result := x
  else Result := y;
end;
{$ENDIF}

function pcr_fminf(x, y: Single): Single;
{$IFDEF AVX2}
// Pure-asm: System V AMD64 ABI passes x→xmm0, y→xmm1; result in xmm0.
// MINSS returns the smaller value; if x (first operand) is NaN, returns y.
assembler;
asm
  minss xmm0, xmm1
end ['xmm0'];
{$ELSE}
begin
  if IsNan(x) then Result := y
  else if IsNan(y) then Result := x
  else if x < y then Result := x
  else Result := y;
end;
{$ENDIF}

function pcr_fmin(x, y: Double): Double;
{$IFDEF AVX2}
// Pure-asm: System V AMD64 ABI passes x→xmm0, y→xmm1; result in xmm0.
// MINSD returns the smaller value; if x (first operand) is NaN, returns y.
assembler;
asm
  minsd xmm0, xmm1
end ['xmm0'];
{$ELSE}
begin
  if IsNan(x) then Result := y
  else if IsNan(y) then Result := x
  else if x < y then Result := x
  else Result := y;
end;
{$ENDIF}

function pcr_nanf(const tagp: PAnsiChar): Single;
begin
  Result := Single(NaN);
end;

function pcr_nan(const tagp: PAnsiChar): Double;
begin
  Result := NaN;
end;

function pcr_feraiseexcept_invalid(): Single;
var
  x: Single;
begin
  // Raise FE_INVALID by computing 0/0
  x := 0.0;
  Result := x / x;
end;

function pcr_feraiseexcept_divbyzero(): Single;
var
  x: Single;
begin
  // Raise FE_DIVBYZERO by computing 1/0
  x := 0.0;
  Result := 1.0 / x;
end;

function pcr_clzll(x: UInt64): Integer; inline;
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

function pcr_GetRoundMode: TFPURoundingMode; inline;
begin
  Result := GetRoundMode;
end;

// ---------------------------------------------------------------------------
// binary64 bit-pattern helpers (task 0.10)
// ---------------------------------------------------------------------------

function pcr_is_signaling(x: Double): Boolean; inline;
var u: Tb64u64;
begin
  u.f := x;
  u.u := u.u xor UInt64($0008000000000000);
  Result := (u.u and UInt64($7FFFFFFFFFFFFFFF)) > UInt64($7FF8000000000000);
end;

function pcr_isint_d(y: Double): Boolean; inline;
var wy: Tb64u64;
    ey, s: Int32;
begin
  wy.f := y;
  ey := Int32((wy.u shr 52) and $7FF) - 1023;
  s  := ey + 12;
  if ey >= 0 then begin
    if s >= 64 then begin Result := True; Exit; end;
    Result := (wy.u shl s) = 0;
    Exit;
  end;
  Result := (wy.u shl 1) = 0;
end;

function pcr_isint_pd(y: Double): Boolean; inline;
var wy: Tb64u64;
    ey, s: Int32;
begin
  wy.f := y;
  ey := Int32((wy.u shr 52) and $7FF) - 1023;
  s  := ey + 12;
  if ey >= 0 then begin
    if s >= 64 then begin Result := True; Exit; end;
    Result := (wy.u shl s) = 0;
    Exit;
  end;
  Result := (wy.u shl 1) = 0;
end;

function pcr_isodd_pd(y: Double): Boolean; inline;
var wy: Tb64u64;
    ey, s: Int32;
    oddb: UInt64;
begin
  wy.f := y;
  ey := Int32((wy.u shr 52) and $7FF) - 1023;
  s  := ey + 12;
  oddb := 0;
  if ey >= 0 then begin
    if (s < 64) and ((wy.u shl s) = 0) then
      oddb := (wy.u shr (64 - s)) and 1;
    if s = 64 then
      oddb := wy.u and 1;
  end;
  Result := oddb <> 0;
end;

function pcr_is_signaling_pd(x: Double): Boolean; inline;
var u: Tb64u64;
begin
  u.f := x;
  u.u := u.u xor UInt64($0008000000000000);
  Result := (u.u and UInt64($7FFFFFFFFFFFFFFF)) > UInt64($7FF8000000000000);
end;

function pcr_mulddd_pd(xh, xl, ch: Double; out l: Double): Double; inline;
var ahlh, ahhh, ahhl: Double;
begin
  ahlh := ch * xl;
  ahhh := ch * xh;
  ahhl := pcr_fma(ch, xh, -ahhh);
  ahhl := ahhl + ahlh;
  ch   := ahhh + ahhl;
  l    := (ahhh - ch) + ahhl;
  Result := ch;
end;

function pcr_get_mxcsr: DWord;
{$IFDEF AVX2}
var r: DWord;
begin
  asm
    stmxcsr r
  end [];
  Result := r;
end;
{$ELSE}
begin
  Result := 0;
end;
{$ENDIF}

procedure pcr_set_mxcsr(flag: DWord);
{$IFDEF AVX2}
begin
  asm
    ldmxcsr flag
  end [];
end;
{$ELSE}
begin
end;
{$ENDIF}

// ---------------------------------------------------------------------------
// Double-double and polynomial helpers (promoted from pascoremath32, task 0.9)
// ---------------------------------------------------------------------------

function pcr_poly12(z: Double; const c: array of Double): Double; inline;
var z2, z4, c0, c2, c4, c6, c8, c10: Double;
begin
  z2 := z * z; z4 := z2 * z2;
  c0  := c[0]  + z * c[1];
  c2  := c[2]  + z * c[3];
  c4  := c[4]  + z * c[5];
  c6  := c[6]  + z * c[7];
  c8  := c[8]  + z * c[9];
  c10 := c[10] + z * c[11];
  c0 := c0 + c2 * z2;
  c4 := c4 + c6 * z2;
  c8 := c8 + z2 * c10;
  c0 := c0 + z4 * (c4 + z4 * c8);
  Result := c0;
end;

function pcr_muldd(xh, xl, ch, cl: Double; out l: Double): Double; inline;
var
  ahlh, alhh, ahhh, ahhl: Double;
begin
  ahlh := ch * xl;
  alhh := cl * xh;
  ahhh := ch * xh;
  ahhl := pcr_fma(ch, xh, -ahhh);
  ahhl := ahhl + alhh + ahlh;
  ch := ahhh + ahhl;
  l := (ahhh - ch) + ahhl;
  Result := ch;
end;

function pcr_polydd(xh, xl: Double; n: Int32; const c: array of Double; out l: Double): Double; inline;
var
  i, i2: Int32;
  ch, cl, th, tl: Double;
begin
  i := n - 1;
  i2 := i * 2;
  ch := c[i2];
  cl := c[i2 + 1];
  Dec(i2,2);
  while i2 >= 8 do begin
    ch := pcr_muldd(xh, xl, ch, cl, cl);
    th := ch + c[i2];
    tl := (c[i2] - th) + ch;
    ch := th;
    cl := cl + tl + c[i2 + 1];
    Dec(i2,2);
    ch := pcr_muldd(xh, xl, ch, cl, cl);
    th := ch + c[i2];
    tl := (c[i2] - th) + ch;
    ch := th;
    cl := cl + tl + c[i2 + 1];
    Dec(i2,2);
    ch := pcr_muldd(xh, xl, ch, cl, cl);
    th := ch + c[i2];
    tl := (c[i2] - th) + ch;
    ch := th;
    cl := cl + tl + c[i2 + 1];
    Dec(i2,2);
    ch := pcr_muldd(xh, xl, ch, cl, cl);
    th := ch + c[i2];
    tl := (c[i2] - th) + ch;
    ch := th;
    cl := cl + tl + c[i2 + 1];
    Dec(i2,2);
  end;
  while i2 >= 0 do begin
    ch := pcr_muldd(xh, xl, ch, cl, cl);
    th := ch + c[i2];
    tl := (c[i2] - th) + ch;
    ch := th;
    cl := cl + tl + c[i2 + 1];
    Dec(i2,2);
  end;
  l := cl;
  Result := ch;
end;

procedure pcr_fasttwosum(out s, t: Double; a, b: Double); inline;
var s_tmp: Double;
begin
  s_tmp := a + b;
  t     := b - (s_tmp - a);
  s     := s_tmp;
end;

procedure pcr_a_mul(out hi, lo: Double; a, b: Double); inline;
var t_am: Double;
begin
  t_am := a * b;
  lo   := pcr_fma(a, b, -t_am);
  hi   := t_am;
end;

procedure pcr_s_mul(out hi, lo: Double; a, bh, bl: Double); inline;
var bl_sm: Double;
begin
  bl_sm := bl;             // save bl before pcr_a_mul may overwrite lo
  pcr_a_mul(hi, lo, a, bh);
  lo := pcr_fma(a, bl_sm, lo);
end;

procedure pcr_d_mul(out hi, lo: Double; ah, al, bh, bl: Double); inline;
var s_dm, t_dm, ah_dm: Double;
begin
  ah_dm := ah;             // save ah before pcr_a_mul may overwrite hi
  pcr_a_mul(hi, s_dm, ah_dm, bh);
  t_dm := pcr_fma(al, bh, s_dm);
  lo   := pcr_fma(ah_dm, bl, t_dm);
end;

end.
