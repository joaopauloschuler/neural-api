unit neuralmxfp4;
// Pure-Pascal MXFP4 (micro-scaled FP4) dequant-at-load helper, the format the
// gpt-oss MoE expert matrices ship in. MXFP4 is the OCP Microscaling (MX) spec
// E2M1 element with an E8M0 shared block scale:
//
//   - The values are grouped in BLOCKS of 32. Each block shares one uint8
//     E8M0 power-of-two scale: the stored byte is the exponent and the scale
//     value is 2^(byte - 127). The byte 0xFF is the reserved NaN scale (per
//     the OCP MX spec); a block carrying it dequantizes to NaN for all 32
//     elements.
//   - The 32 mantissas are packed as 16 bytes of 4-bit pairs. Within each
//     byte the LOW nibble is the EVEN-indexed element and the HIGH nibble is
//     the ODD-indexed element (matching transformers' _convert_moe_packed_
//     tensors: out[..., 0::2] = lut[blk & 0x0F]; out[..., 1::2] = lut[blk>>4]).
//   - Each 4-bit element is FP4 E2M1 (1 sign, 2 exp, 1 mantissa). The 16
//     representable values are the standard MXFP4 lookup table
//     [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] for nibbles 0..7 and their
//     negations for nibbles 8..15 (sign bit = high bit of the nibble).
//
//   Dequantized element = MXFP4_LUT[nibble] * 2^(scale_byte - 127).
//
// gpt-oss stores the packed mantissas and the per-block scales as TWO SEPARATE
// safetensors tensors (*_blocks uint8 and *_scales uint8). This helper takes
// both raw byte streams and expands them into a caller-provided FP32 buffer,
// the same dequant-at-load contract as the GGUF Q8_0/Q4_K helpers in
// neuralgguf.pas, but kept in its own unit because MXFP4 is not a ggml dtype
// and carries no GGUF dependency. The full BuildGptOssFromSafeTensors importer
// will consume this when expanding gate_up_proj / down_proj expert banks.
//
// Verified against transformers 5.11 integrations/mxfp4.py
// (FP4_VALUES table + _convert_moe_packed_tensors) and the OCP MX spec.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  SysUtils, Math;

const
  // Number of FP4 elements per MXFP4 block (share one E8M0 scale).
  MXFP4_BLOCK_ELEMS = 32;
  // Packed mantissa bytes per block (32 nibbles = 16 bytes).
  MXFP4_BLOCK_BYTES = 16;
  // E8M0 exponent bias: scale value = 2^(stored_byte - bias).
  MXFP4_E8M0_BIAS = 127;
  // Reserved E8M0 NaN scale byte (OCP MX spec).
  MXFP4_E8M0_NAN = $FF;

type
  EMXFP4Error = class(Exception);

// The 16 representable FP4 E2M1 values, indexed by the raw 4-bit nibble:
// nibbles 0..7 are the positive magnitudes, 8..15 their negations.
function MXFP4Lut(Nibble: byte): single;

// Dequantizes NumBlocks MXFP4 blocks into Dest (NumBlocks*32 FP32 elements).
//   Blocks: packed mantissa bytes, NumBlocks*16 bytes (16 bytes per block).
//   Scales: one E8M0 scale byte per block, NumBlocks bytes.
//   Dest:   caller-provided FP32 buffer, NumBlocks*32 singles.
// A block whose scale byte is 0xFF dequantizes to NaN for all 32 elements.
// Raises EMXFP4Error on a nil pointer or non-positive NumBlocks.
procedure DequantizeMXFP4(const Blocks: PByte; const Scales: PByte;
  NumBlocks: Int64; Dest: PSingle);

implementation

const
  // [0, 0.5, 1, 1.5, 2, 3, 4, 6] then negated half.
  cMXFP4Lut: array[0..15] of single = (
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
  );

function MXFP4Lut(Nibble: byte): single;
begin
  Result := cMXFP4Lut[Nibble and $0F];
end;

procedure DequantizeMXFP4(const Blocks: PByte; const Scales: PByte;
  NumBlocks: Int64; Dest: PSingle);
var
  b, i: Int64;
  BlkPtr: PByte;
  Outp: PSingle;
  ScaleByte: byte;
  ScaleVal, NaNVal: single;
  PackedByte: byte;
  NumBlocksM1, ElemsM1, BytesM1: Int64;
begin
  if (Blocks = nil) or (Scales = nil) or (Dest = nil) then
    raise EMXFP4Error.Create('DequantizeMXFP4: nil pointer.');
  if NumBlocks <= 0 then
    raise EMXFP4Error.Create('DequantizeMXFP4: NumBlocks must be positive.');
  NaNVal := NaN;
  NumBlocksM1 := NumBlocks - 1;
  ElemsM1 := MXFP4_BLOCK_ELEMS - 1;
  BytesM1 := MXFP4_BLOCK_BYTES - 1;
  for b := 0 to NumBlocksM1 do
  begin
    BlkPtr := Blocks + b * MXFP4_BLOCK_BYTES;
    Outp   := Dest + b * MXFP4_BLOCK_ELEMS;
    ScaleByte := (Scales + b)^;
    if ScaleByte = MXFP4_E8M0_NAN then
    begin
      // Reserved NaN scale: whole block is NaN.
      for i := 0 to ElemsM1 do
        Outp[i] := NaNVal;
      Continue;
    end;
    // 2^(byte - 127). Ldexp keeps it exact and avoids Exp/Power rounding.
    ScaleVal := Ldexp(1.0, Integer(ScaleByte) - MXFP4_E8M0_BIAS);
    for i := 0 to BytesM1 do
    begin
      PackedByte := (BlkPtr + i)^;
      // Low nibble -> even element, high nibble -> odd element.
      Outp[2 * i]     := cMXFP4Lut[PackedByte and $0F] * ScaleVal;
      Outp[2 * i + 1] := cMXFP4Lut[(PackedByte shr 4) and $0F] * ScaleVal;
    end;
  end;
end;

end.
