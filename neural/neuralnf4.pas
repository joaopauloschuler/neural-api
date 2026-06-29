unit neuralnf4;
// Pure-Pascal NF4 (bitsandbytes 4-bit NormalFloat) dequant-at-load helper, the
// format `bnb-4bit` HF checkpoints (Linear4bit, e.g. a 4-bit Llama/Qwen) ship
// their weight matrices in. NF4 is bitsandbytes' nonlinear 4-bit quantization:
//
//   - Values are grouped in BLOCKS (default 64 elements). Each block shares one
//     FP32 `absmax` scale = max(|w|) over the block. The 4-bit code is an index
//     into a fixed 16-level nonlinear codebook (the standard-normal quantiles
//     baked into bitsandbytes create_normal_map()); index 7 is exactly 0.0.
//
//       Dequantized element = NF4_CODE[nibble] * absmax_block.
//
//   - Two 4-bit indices are packed per byte, HIGH nibble FIRST: for packed byte
//     B, element 2*i comes from (B shr 4) and element 2*i+1 from (B and $0F)
//     (matching bitsandbytes' kDequantizeBlockwise nibble order and the HF
//     Linear4bit storage; this is the OPPOSITE order to MXFP4, whose low nibble
//     is the even element).
//
//   - bitsandbytes optionally DOUBLE-QUANTIZES the absmax values themselves (an
//     int8 quant of the absmax with a nested absmax + a mean offset). That path
//     is NOT yet supported here: callers must pass the already-dequantized FP32
//     absmax. DequantizeNF4 only consumes single-quant FP32 absmax; the
//     importer is responsible for detecting `*.nested_absmax` /
//     `*.quant_state.*` double-quant metadata and raising a clear error rather
//     than silently producing garbage.
//
// The HF safetensors for bnb-4bit store the packed weight as `weight` (uint8,
// shape [ceil(n/2),1]) plus `weight.absmax` (the per-block scales, fp32 when
// not double-quantized) and `weight.quant_map` (the 16-level codebook). This
// helper takes the packed byte stream + the FP32 absmax array and expands them
// into a caller-provided FP32 buffer, the same dequant-at-load contract as the
// MXFP4 helper in neuralmxfp4.pas and the GGUF Q8_0/Q4_K helpers in
// neuralgguf.pas, but kept in its own unit (NF4 is a bitsandbytes format, not a
// ggml dtype and not MX). BuildLlamaFromSafeTensors family consumes this when a
// checkpoint stores its projection/MLP weights as bnb-4bit blocks.
//
// Verified against a numpy reconstruction of the published bitsandbytes NF4
// codebook + block-absmax formula (bitsandbytes is not importable in the CPU
// venv); see tools/make_pico_nf4_fixture.py and the TestNF4Dequant* tests.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  SysUtils;

const
  // Default NF4 block size in bitsandbytes (elements sharing one absmax).
  NF4_DEFAULT_BLOCKSIZE = 64;

type
  ENF4Error = class(Exception);

// The 16-level NF4 codebook value for a raw 4-bit nibble (index 0..15).
function NF4Code(Nibble: byte): single;

// Dequantizes a packed NF4 weight into Dest (NumElements FP32 values).
//   Packed:      packed nibble bytes, ceil(NumElements/2) bytes. Within each
//                byte the HIGH nibble is the even element, the LOW nibble the
//                odd element.
//   Absmax:      one FP32 scale per block, ceil(NumElements/BlockSize) entries.
//                Must already be FP32 (double-quant must be expanded upstream).
//   NumElements: number of dequantized FP32 values to produce.
//   BlockSize:   elements per absmax block (bitsandbytes default 64).
//   Dest:        caller-provided FP32 buffer, NumElements singles.
// Raises ENF4Error on a nil pointer, non-positive count or non-positive block.
procedure DequantizeNF4(const Codes: PByte; const Absmax: PSingle;
  NumElements: Int64; Dest: PSingle; BlockSize: Int64 = NF4_DEFAULT_BLOCKSIZE);

implementation

const
  // bitsandbytes NF4 codebook (functional.py create_normal_map, quant_type
  // "nf4"). 16 ascending nonlinear levels; index 7 is exactly 0.0.
  cNF4Code: array[0..15] of single = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
     0.0,
     0.07958029955625534,
     0.16093020141124725,
     0.24611230194568634,
     0.33791524171829224,
     0.44070982933044434,
     0.5626170039176941,
     0.7229568362236023,
     1.0
  );

function NF4Code(Nibble: byte): single;
begin
  Result := cNF4Code[Nibble and $0F];
end;

procedure DequantizeNF4(const Codes: PByte; const Absmax: PSingle;
  NumElements: Int64; Dest: PSingle; BlockSize: Int64 = NF4_DEFAULT_BLOCKSIZE);
var
  i, NumElementsM1: Int64;
  PackedByte: byte;
  ScaleVal: single;
  SrcByte: PByte;
begin
  if (Codes = nil) or (Absmax = nil) or (Dest = nil) then
    raise ENF4Error.Create('DequantizeNF4: nil pointer.');
  if NumElements <= 0 then
    raise ENF4Error.Create('DequantizeNF4: NumElements must be positive.');
  if BlockSize <= 0 then
    raise ENF4Error.Create('DequantizeNF4: BlockSize must be positive.');
  NumElementsM1 := NumElements - 1;
  for i := 0 to NumElementsM1 do
  begin
    // Two elements per byte; HIGH nibble is the even element.
    SrcByte := Codes + (i shr 1);
    PackedByte := SrcByte^;
    ScaleVal := (Absmax + (i div BlockSize))^;
    if (i and 1) = 0 then
      Dest[i] := cNF4Code[(PackedByte shr 4) and $0F] * ScaleVal
    else
      Dest[i] := cNF4Code[PackedByte and $0F] * ScaleVal;
  end;
end;

end.
