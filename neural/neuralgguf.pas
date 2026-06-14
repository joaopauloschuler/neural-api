unit neuralgguf;
// Pure-Pascal reader for the GGUF checkpoint format (llama.cpp / ggml -
// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), the other
// de-facto LLM checkpoint container next to safetensors. GGUF ships typed
// metadata key/value pairs (the config travels INSIDE the file - no
// config.json) and, crucially, PRE-quantized tensor data.
//
// File layout (v2/v3, little-endian):
//   u32 magic 'GGUF', u32 version, u64 tensor_count, u64 metadata_kv_count
//   metadata_kv_count KV pairs: string key (u64 length + bytes), u32 value
//     type, value. Types: u8/i8/u16/i16/u32/i32/f32/bool/string/array/
//     u64/i64/f64; array = u32 element type + u64 count + packed elements.
//   tensor_count tensor infos: string name, u32 n_dims, u64 dims[n_dims]
//     (ggml order: dims[0] is the FASTEST-varying/contiguous axis, so an
//     HF row-major [out, in] matrix appears as dims [in, out]), u32 ggml
//     type, u64 offset (relative to the data section, aligned).
//   padding to general.alignment (default 32), then the tensor data blob.
//
// The reader subclasses TNNetSafeTensorsReader (the TNNetTorchBinReader
// pattern) and populates the same tensor table with the dims REVERSED into
// row-major order, so the whole public API - HasTensor/DimCount/DimSize/
// ShapeAsString/LoadTensorFlat/... - matches the safetensors reader and a
// [out, in] nn.Linear matrix reports the same shape from either format.
// LoadTensorFlat is overridden to decode the ggml dtypes:
//   F32  (type id 0) - native singles;
//   F16  (type id 1) - via neuralsafetensors.DecodeF16;
//   Q8_0 (type id 8) - blocks of 32 elements along the contiguous axis,
//        each block an f16 scale d + 32 int8 quants, x = d * q. Decoded
//        (dequantized) to FP32 on load - the int8-direct path into
//        TNNet.QuantizeWeightsInt8 storage is a possible follow-up.
//   Q2_K (10) / Q4_K (12) / Q5_K (13) / Q6_K (14) - the k-quant members of
//        the dominant community mixes. ggml k-quant 256-element super-blocks:
//        Q4_K/Q5_K/Q6_K use 8 sub-blocks of 32 with a block-level f16 d
//        (plus f16 d_min for Q4_K/Q5_K), 6-bit packed sub-scales/sub-mins
//        and 4/5/6-bit packed quants; Q2_K uses 16 sub-blocks of 16 with
//        4-bit packed sub-scales/sub-mins and 2-bit quants. Dequantized to
//        FP32 on load, mirroring ggml's reference dequant_row_q2_K /
//        dequant_row_q4_K / dequant_row_q5_K / dequant_row_q6_K unpacking.
// Anything else (Q3_K, Q4_0, ...) raises EGGUFError with the type name.
//
// Importer hooks (used by BuildLlamaFromGGUF in neuralpretrained.pas):
//   RenameTensor          - GGUF names tensors per the ggml convention
//                           (token_embd.weight, blk.N.attn_q.weight, ...);
//                           the importer renames them to their HF
//                           equivalents and reuses the safetensors path.
//   RegisterRowDeinterleave - llama.cpp's convert script PERMUTES the q/k
//                           projection rows of rotary models from HF's
//                           rotate_half layout into the interleaved-pair
//                           layout (per head: new[2p] = old[p],
//                           new[2p+1] = old[p + head_dim/2]). Registering
//                           a tensor makes LoadTensorFlat serve the rows
//                           de-interleaved, i.e. back in HF order.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume, neuralsafetensors;

const
  // GGUF metadata value-type ids (the wire format).
  GGUF_TYPE_UINT8   = 0;
  GGUF_TYPE_INT8    = 1;
  GGUF_TYPE_UINT16  = 2;
  GGUF_TYPE_INT16   = 3;
  GGUF_TYPE_UINT32  = 4;
  GGUF_TYPE_INT32   = 5;
  GGUF_TYPE_FLOAT32 = 6;
  GGUF_TYPE_BOOL    = 7;
  GGUF_TYPE_STRING  = 8;
  GGUF_TYPE_ARRAY   = 9;
  GGUF_TYPE_UINT64  = 10;
  GGUF_TYPE_INT64   = 11;
  GGUF_TYPE_FLOAT64 = 12;

  // ggml tensor dtype ids (the ones this reader decodes).
  GGML_TYPE_F32  = 0;
  GGML_TYPE_F16  = 1;
  GGML_TYPE_Q8_0 = 8;
  GGML_TYPE_Q2_K = 10;
  GGML_TYPE_Q4_K = 12;
  GGML_TYPE_Q5_K = 13;
  GGML_TYPE_Q6_K = 14;

  // Q8_0 block geometry: 32 elements stored as f16 scale + 32 int8.
  GGUF_Q8_0_BLOCK_ELEMS = 32;
  GGUF_Q8_0_BLOCK_BYTES = 34;

  // k-quant super-block geometry. Every k-quant packs 256 elements
  // (QK_K) into one super-block of 8 sub-blocks of 32. Q4_K and Q6_K are
  // the two members of the dominant Q4_K_M mix.
  GGUF_QK_K            = 256;
  // block_q4_K = 144 bytes: f16 d, f16 d_min, 12 bytes of 6-bit packed
  // sub-scales/sub-mins, then 128 bytes of 4-bit quants (256 nibbles).
  GGUF_Q4_K_BLOCK_BYTES = 144;
  GGUF_K_SCALE_SIZE     = 12;
  // block_q5_K = 176 bytes: f16 d, f16 d_min, 12 bytes of 6-bit packed
  // sub-scales/sub-mins (same as Q4_K), 32 bytes qh (the 5th-bit plane),
  // then 128 bytes of 4-bit low quants (256 nibbles).
  GGUF_Q5_K_BLOCK_BYTES = 176;
  // block_q2_K = 84 bytes: 16 bytes of 4-bit packed sub-scales|sub-mins,
  // 64 bytes of 2-bit quants (256 fields), then f16 d and f16 d_min.
  GGUF_Q2_K_BLOCK_BYTES = 84;
  // block_q6_K = 210 bytes: 128 bytes ql (4-bit low), 64 bytes qh (2-bit
  // high), 16 int8 group scales, then f16 d.
  GGUF_Q6_K_BLOCK_BYTES = 210;

type
  EGGUFError = class(ESafeTensorsError);

  // One parsed metadata KV pair. Scalars land in the matching field
  // (integers of every width in IntVal, f32/f64 in FloatVal); arrays keep
  // their element type in ArrElemType with the values in ArrStr (string
  // arrays) or ArrInt+ArrNum (numeric/bool arrays, both filled).
  TGGUFMetaValue = record
    Key: string;
    ValueType: integer;          // GGUF_TYPE_* of the value
    IntVal: Int64;               // u8..i64 scalars (and bool as 0/1)
    FloatVal: double;            // f32/f64 scalars
    BoolVal: boolean;
    StrVal: string;
    ArrElemType: integer;        // GGUF_TYPE_* of the elements (arrays)
    ArrStr: array of string;
    ArrInt: array of Int64;
    ArrNum: array of double;
  end;

  { TNNetGGUFReader }
  // Opens a .gguf file, parses and validates the header, metadata and
  // tensor table, and serves named tensors through the inherited
  // TNNetSafeTensorsReader API (shapes reported in row-major order, data
  // decoded to FP32). Raises EGGUFError (an ESafeTensorsError, so existing
  // importer handlers catch it) on truncated, malformed or unsupported
  // input - it never silently returns garbage.
  // Coded by Claude (AI).
  TNNetGGUFReader = class(TNNetSafeTensorsReader)
  private
    FVersion: integer;
    FAlignment: integer;
    FMeta: array of TGGUFMetaValue;
    FGGMLTypes: array of integer;        // parallel to inherited FTensors
    FDeinterleaveNames: array of string; // RegisterRowDeinterleave targets
    FDeinterleaveHeadDim: array of integer;
    procedure ParseFile(Stream: TFileStream);
    function ReadHeaderString(Stream: TFileStream): string;
    procedure ReadMetaValue(Stream: TFileStream; var Meta: TGGUFMetaValue);
    function FindMeta(const pKey: string): integer;
    function DeinterleaveHeadDimFor(const pName: string): integer;
  public
    constructor Create(const pFileName: string);

    // ---- metadata access ----
    function MetaCount: integer;
    function MetaKey(Index: integer): string;
    function HasMetaKey(const pKey: string): boolean;
    // Typed scalar getters: a missing key returns Default; a key holding an
    // incompatible type raises EGGUFError (integers convert to float, bools
    // to 0/1 integers - the GGUF ecosystem is loose about scalar widths).
    function GetMetaInt(const pKey: string; Default: Int64): Int64;
    function GetMetaFloat(const pKey: string; Default: double): double;
    function GetMetaString(const pKey: string;
      const Default: string): string;
    function GetMetaBool(const pKey: string; Default: boolean): boolean;
    // Array getters: count of a missing key is 0; element getters raise on
    // a bad index or element type.
    function GetMetaArrayCount(const pKey: string): Int64;
    function GetMetaArrayString(const pKey: string; Index: integer): string;
    function GetMetaArrayNumber(const pKey: string; Index: integer): double;

    // ---- tensor access beyond the inherited API ----
    // The ggml dtype id of the named tensor (GGML_TYPE_*).
    function TensorGGMLType(const pName: string): integer;
    // Renames a tensor (importer hook - GGUF ggml names -> HF names).
    // Raises if pOldName is missing or pNewName already exists.
    procedure RenameTensor(const pOldName, pNewName: string);
    // Marks a 2-D tensor whose rows llama.cpp permuted into the
    // interleaved-rotary order; LoadTensorFlat will undo the permutation
    // (per HeadDim-row head: hf_row[p] = stored[2p],
    // hf_row[p + HeadDim/2] = stored[2p+1]) and serve HF-order rows.
    procedure RegisterRowDeinterleave(const pName: string;
      HeadDim: integer);

    procedure LoadTensorFlat(const pName: string;
      Dest: TNNetVolume); override;

    property Version: integer read FVersion;
    property Alignment: integer read FAlignment;
  end;

type
  // The ggml dtype a writer encodes a 2-D matrix tensor as. F32 is the
  // lossless default; F16 halves the file (round-to-nearest-even via
  // EncodeF16); Q8_0 quantizes 32-element blocks to an f16 scale + 32 int8
  // (~4x smaller, the dominant weight-only quant). 1-D norm gains always
  // stay F32 (the llama.cpp convention), whatever the matrix dtype.
  TGGUFWriteDType = (gwF32, gwF16, gwQ8_0);

  // One tensor queued in a TNNetGGUFWriter: its ggml name, the row-major
  // shape (the writer reverses it to the ggml contiguous-first order on
  // disk), the chosen ggml dtype and the encoded little-endian bytes.
  TGGUFPendingTensor = record
    Name: string;
    Shape: array of Int64;       // row-major (as the reader serves it)
    GGMLType: integer;           // GGML_TYPE_* actually written
    Data: TBytes;                // already-encoded payload
  end;

  { TNNetGGUFWriter }
  // Builds a llama.cpp/ggml-loadable .gguf (the byte-for-byte inverse of
  // TNNetGGUFReader): magic 'GGUF', version 3, the typed metadata KV block
  // and the tensor section with ggml's REVERSED dimension order and 32-byte
  // data alignment. Metadata is queued through the typed Add* methods
  // (insertion order is preserved); tensors through AddTensorFlat (which
  // encodes F32/F16/Q8_0 and reverses the shape). SaveToFile emits the file
  // once. A file written here re-imports through TNNetGGUFReader /
  // BuildLlamaFromGGUF and loads with the Python "gguf" package.
  // Coded by Claude (AI).
  TNNetGGUFWriter = class
  private
    FFileName: string;
    FMeta: array of TGGUFMetaValue;
    FTensors: array of TGGUFPendingTensor;
    FAlignment: integer;
    FSaved: boolean;
    function FindMeta(const pKey: string): integer;
    function FindTensor(const pName: string): integer;
    procedure AddMeta(const Meta: TGGUFMetaValue);
    procedure WriteHeaderString(Stream: TStream; const S: string);
    procedure WriteMetaValue(Stream: TStream; const Meta: TGGUFMetaValue);
  public
    constructor Create(const pFileName: string);
    destructor Destroy; override;

    // ---- typed metadata (queued in call order, keys must be unique) ----
    procedure AddMetaUInt32(const pKey: string; Value: cardinal);
    procedure AddMetaInt32(const pKey: string; Value: integer);
    procedure AddMetaUInt64(const pKey: string; Value: QWord);
    procedure AddMetaFloat32(const pKey: string; Value: single);
    procedure AddMetaBool(const pKey: string; Value: boolean);
    procedure AddMetaString(const pKey, pValue: string);
    // A string array (e.g. tokenizer.ggml.tokens / .merges).
    procedure AddMetaStringArray(const pKey: string;
      const Values: array of string);
    // A float32 array (e.g. tokenizer.ggml.scores).
    procedure AddMetaFloat32Array(const pKey: string;
      const Values: array of single);
    // An int32 array (e.g. tokenizer.ggml.token_type).
    procedure AddMetaInt32Array(const pKey: string;
      const Values: array of Int64);

    // Queues a named tensor. pShape is the row-major shape (a 2-D matrix is
    // [out, in], the same view the reader serves); the writer reverses it to
    // ggml's contiguous-first order on disk. Src supplies the elements in
    // flat row-major order (prod(pShape) must equal Src.Size). pDType picks
    // the ggml encoding: gwF32/gwF16/gwQ8_0. Q8_0 requires the contiguous
    // (last) dimension to be a multiple of 32. Data is copied, so Src may be
    // freed/reused immediately.
    procedure AddTensorFlat(const pName: string; const pShape: array of Int64;
      Src: TNNetVolume; pDType: TGGUFWriteDType = gwF32);
    // Queues a Q8_0 tensor built DIRECTLY from int8 weight-only storage
    // (TNNetLayerConcatedWeights.QuantizeWeightsInt8: per-output-row symmetric
    // codes + one FP32 scale per row, dequant value = code * RowScale[r]),
    // without materializing the FP32 matrix. pShape is [out, in] (row-major,
    // as the reader serves it); the contiguous (last = in) dimension must be a
    // multiple of 32 AND each int8 row is exactly one output channel, so a
    // Q8_0 32-block never crosses a row boundary. pCodes holds NumRows * VS
    // codes in row-major order (row r at pCodes[r*VS .. r*VS+VS-1]); pScales
    // holds NumRows scales. The FAITHFUL mapping is NOT a byte copy (Q8_0 uses
    // one f16 scale per 32-block, int8 storage uses one scale per row): for
    // each block we recompute blockscale = max|code|*RowScale/127 and emit
    // q[i] = round(code[i]*127 / max|code|) (the row scale cancels in the
    // quants, surviving only in the f16 d). This reproduces the SAME bytes the
    // gwQ8_0 path would emit from the DEQUANTIZED FP32 row, within Q8_0
    // rounding. Coded by Claude (AI).
    procedure AddTensorFlatInt8(const pName: string;
      const pShape: array of Int64; const pCodes: array of ShortInt;
      const pScales: array of single; NumRows, VS: integer);
    function Count: integer;
    // Writes the queued metadata and tensors to FileName. Call once.
    procedure SaveToFile;

    property FileName: string read FFileName;
    property Alignment: integer read FAlignment write FAlignment;
  end;

// Human-readable name of a ggml tensor dtype id (for error messages).
function GGMLTypeName(TypeId: integer): string;

implementation

function GGMLTypeName(TypeId: integer): string;
begin
  case TypeId of
    0: Result := 'F32';
    1: Result := 'F16';
    2: Result := 'Q4_0';
    3: Result := 'Q4_1';
    6: Result := 'Q5_0';
    7: Result := 'Q5_1';
    8: Result := 'Q8_0';
    9: Result := 'Q8_1';
    10: Result := 'Q2_K';
    11: Result := 'Q3_K';
    12: Result := 'Q4_K';
    13: Result := 'Q5_K';
    14: Result := 'Q6_K';
    15: Result := 'Q8_K';
    24: Result := 'I8';
    25: Result := 'I16';
    26: Result := 'I32';
    27: Result := 'I64';
    28: Result := 'F64';
    30: Result := 'BF16';
    else Result := 'GGML_TYPE_' + IntToStr(TypeId);
  end;
end;

// Byte length of NumElements elements of the given ggml dtype; 0 for
// dtypes this reader cannot size (unsupported anyway).
function GGMLByteSize(TypeId: integer; NumElements: Int64): Int64;
begin
  case TypeId of
    GGML_TYPE_F32: Result := NumElements * 4;
    GGML_TYPE_F16: Result := NumElements * 2;
    GGML_TYPE_Q8_0:
      Result := (NumElements div GGUF_Q8_0_BLOCK_ELEMS) *
        GGUF_Q8_0_BLOCK_BYTES;
    GGML_TYPE_Q2_K:
      Result := (NumElements div GGUF_QK_K) * GGUF_Q2_K_BLOCK_BYTES;
    GGML_TYPE_Q4_K:
      Result := (NumElements div GGUF_QK_K) * GGUF_Q4_K_BLOCK_BYTES;
    GGML_TYPE_Q5_K:
      Result := (NumElements div GGUF_QK_K) * GGUF_Q5_K_BLOCK_BYTES;
    GGML_TYPE_Q6_K:
      Result := (NumElements div GGUF_QK_K) * GGUF_Q6_K_BLOCK_BYTES;
    else Result := 0;
  end;
end;

// Unpacks the 12 packed bytes of a Q4_K super-block into the 8 6-bit
// sub-block scales (Sc) and 8 6-bit sub-block mins (Mn). This mirrors
// ggml's get_scale_min_k4 EXACTLY. The 12 bytes hold, per the layout
// (groups d=bytes[0..3], m=bytes[4..7], m_d=bytes[8..11]):
//   sc[0..3] = d & 0x3F
//   sc[4..7] = (m_d & 0x0F) | ((d >> 2) & 0x30)
//   mn[0..3] = m & 0x3F
//   mn[4..7] = (m_d >> 4) | ((m >> 2) & 0x30)
procedure DequantizeQ4KScaleMin(const PackedSc: PByte;
  out Sc, Mn: array of byte);
var
  j: integer;
  d, m, m_d: byte;
begin
  for j := 0 to 3 do
  begin
    d   := PackedSc[j];
    m   := PackedSc[j + 4];
    m_d := PackedSc[j + 8];
    Sc[j]     := d and $3F;
    Sc[j + 4] := (m_d and $0F) or ((d shr 2) and $30);
    Mn[j]     := m and $3F;
    Mn[j + 4] := (m_d shr 4) or ((m shr 2) and $30);
  end;
end;

// Dequantizes a Q4_K tensor (NumBlocks super-blocks of 256) from Raw into
// Dest in flat row-major order. Block layout (144 bytes): f16 d, f16
// d_min, 12 packed scale bytes, then 128 bytes of 4-bit quants. The 256
// nibbles are 4 byte-chunks of 32; chunk c carries sub-block 2c in the low
// nibble and sub-block 2c+1 in the high nibble. Reconstruction per
// sub-block j (0..7): x = d*Sc[j]*q - d_min*Mn[j], q in [0,15].
procedure DequantizeQ4K(const Raw: PByte; NumBlocks: Int64;
  Dest: PSingle);
var
  b, j, c, i: Int64;
  Base: PByte;
  d, dmin, dsc, dm: single;
  Sc, Mn: array[0..7] of byte;
  q: byte;
  Outp: PSingle;
begin
  for b := 0 to NumBlocks - 1 do
  begin
    Base := Raw + b * GGUF_Q4_K_BLOCK_BYTES;
    d    := DecodeF16(PWord(Base)^);
    dmin := DecodeF16(PWord(Base + 2)^);
    DequantizeQ4KScaleMin(Base + 4, Sc, Mn);
    Outp := Dest + b * GGUF_QK_K;
    // Sub-blocks come in low/high-nibble pairs sharing one 32-byte chunk.
    for c := 0 to 3 do
    begin
      // even sub-block: low nibble; odd sub-block: high nibble.
      j := 2 * c;
      dsc := d * Sc[j];
      dm  := dmin * Mn[j];
      for i := 0 to 31 do
      begin
        q := (Base + 16 + c * 32 + i)^ and $0F;
        Outp[j * 32 + i] := dsc * q - dm;
      end;
      j := 2 * c + 1;
      dsc := d * Sc[j];
      dm  := dmin * Mn[j];
      for i := 0 to 31 do
      begin
        q := ((Base + 16 + c * 32 + i)^ shr 4) and $0F;
        Outp[j * 32 + i] := dsc * q - dm;
      end;
    end;
  end;
end;

// Dequantizes a Q5_K tensor (NumBlocks super-blocks of 256) from Raw into
// Dest in flat row-major order. Block layout (176 bytes): f16 d, f16 d_min,
// 12 packed scale bytes (same get_scale_min_k4 packing as Q4_K), 32 bytes
// qh (the 5th-bit plane), then 128 bytes of 4-bit low quants. Like Q4_K but
// each quant is 5-bit: q = ql | (qh_bit << 4) in [0,31]. Reconstruction per
// sub-block j (0..7): x = d*Sc[j]*q - d_min*Mn[j].
//
// ql layout (128 bytes): 4 byte-chunks of 32; chunk c carries sub-block 2c
// in the low nibble and 2c+1 in the high nibble (identical to Q4_K).
// qh layout (32 bytes): one bit-plane shared by all 8 sub-blocks - sub-block
// j takes bit j of byte (i mod 32). This mirrors ggml's
// qh.reshape(-1,1,32) >> [0..7] & 1 in dequantize_row_q5_K.
procedure DequantizeQ5K(const Raw: PByte; NumBlocks: Int64;
  Dest: PSingle);
var
  b, j, c, i: Int64;
  Base, QhPtr, QlPtr: PByte;
  d, dmin, dsc, dm: single;
  Sc, Mn: array[0..7] of byte;
  ql, qh, q: byte;
  Outp: PSingle;
begin
  for b := 0 to NumBlocks - 1 do
  begin
    Base := Raw + b * GGUF_Q5_K_BLOCK_BYTES;
    d    := DecodeF16(PWord(Base)^);
    dmin := DecodeF16(PWord(Base + 2)^);
    DequantizeQ4KScaleMin(Base + 4, Sc, Mn);
    QhPtr := Base + 16;        // 32-byte 5th-bit plane
    QlPtr := Base + 48;        // 128 bytes of 4-bit low quants
    Outp := Dest + b * GGUF_QK_K;
    // Sub-blocks come in low/high-nibble pairs sharing one 32-byte chunk;
    // the 5th bit of sub-block j is bit j of the qh byte at (i mod 32).
    for c := 0 to 3 do
    begin
      // even sub-block: low nibble; odd sub-block: high nibble.
      j := 2 * c;
      dsc := d * Sc[j];
      dm  := dmin * Mn[j];
      for i := 0 to 31 do
      begin
        ql := (QlPtr + c * 32 + i)^ and $0F;
        qh := ((QhPtr + i)^ shr j) and $01;
        q  := ql or (qh shl 4);
        Outp[j * 32 + i] := dsc * q - dm;
      end;
      j := 2 * c + 1;
      dsc := d * Sc[j];
      dm  := dmin * Mn[j];
      for i := 0 to 31 do
      begin
        ql := ((QlPtr + c * 32 + i)^ shr 4) and $0F;
        qh := ((QhPtr + i)^ shr j) and $01;
        q  := ql or (qh shl 4);
        Outp[j * 32 + i] := dsc * q - dm;
      end;
    end;
  end;
end;

// Dequantizes a Q2_K tensor (NumBlocks super-blocks of 256) from Raw into
// Dest in flat row-major order. Block layout (84 bytes): 16 bytes of 4-bit
// packed sub-scales|sub-mins (one byte per sub-block, scale in the low
// nibble, min in the high nibble), 64 bytes of 2-bit quants, then f16 d and
// f16 d_min. There are 16 sub-blocks of 16 elements. Reconstruction per
// sub-block sb (0..15): x = d*(scale[sb])*q - d_min*(min[sb]), q in [0,3].
//
// qs layout (64 bytes): output element e splits as c = e div 128,
// s = (e mod 128) div 32, p = e mod 32; the 2-bit field is at shift 2*s of
// byte (c*32 + p). This mirrors ggml's qs.reshape(-1,1,32) >> [0,2,4,6] & 3
// in dequantize_row_q2_K. The sub-block of element e is sb = e div 16.
procedure DequantizeQ2K(const Raw: PByte; NumBlocks: Int64;
  Dest: PSingle);
var
  b, e, c, s, p, sb: Int64;
  Base, ScPtr, QsPtr: PByte;
  d, dmin: single;
  scale, mn, q: byte;
  Outp: PSingle;
begin
  for b := 0 to NumBlocks - 1 do
  begin
    Base   := Raw + b * GGUF_Q2_K_BLOCK_BYTES;
    ScPtr  := Base;            // 16 bytes scale|min
    QsPtr  := Base + 16;       // 64 bytes of 2-bit quants
    d    := DecodeF16(PWord(Base + 80)^);
    dmin := DecodeF16(PWord(Base + 82)^);
    Outp := Dest + b * GGUF_QK_K;
    for e := 0 to GGUF_QK_K - 1 do
    begin
      c := e div 128;
      s := (e mod 128) div 32;
      p := e mod 32;
      q := ((QsPtr + (c * 32 + p))^ shr (2 * s)) and $03;
      sb := e div 16;
      scale := (ScPtr + sb)^ and $0F;
      mn    := ((ScPtr + sb)^ shr 4) and $0F;
      Outp[e] := d * scale * q - dmin * mn;
    end;
  end;
end;

// Dequantizes a Q6_K tensor (NumBlocks super-blocks of 256) from Raw into
// Dest in flat row-major order. Block layout (210 bytes): 128 bytes ql
// (4-bit low), 64 bytes qh (2-bit high), 16 int8 group scales, then f16 d.
// The 6-bit quant q = (ql | (qh << 4)) - 32 in [-32,31]; there are 16
// scale groups of 16 elements (scale group g = element index div 16).
// Reconstruction: x = d * scales[g] * q.
//
// ql layout: 2 byte-chunks of 64; chunk c carries sub-block 2c (low
// nibble) and sub-block 2c+1 (high nibble), sub-blocks 64 wide.
// qh layout: 2 byte-chunks of 32; chunk c packs four 32-wide sub-blocks at
// bit offsets 0,2,4,6.
procedure DequantizeQ6K(const Raw: PByte; NumBlocks: Int64;
  Dest: PSingle);
var
  b, i, g: Int64;
  Base, QlPtr, QhPtr: PByte;
  d: single;
  Scales: PShortInt;
  Outp: PSingle;
  ql, qh, idx, qv: integer;
begin
  for b := 0 to NumBlocks - 1 do
  begin
    Base := Raw + b * GGUF_Q6_K_BLOCK_BYTES;
    QlPtr := Base;
    QhPtr := Base + 128;
    Scales := PShortInt(Base + 192);
    d := DecodeF16(PWord(Base + 208)^);
    Outp := Dest + b * GGUF_QK_K;
    // Walk the 256 element indices; recompute each byte/bit position from
    // ggml's ql reshape((-1,1,64))>>[0,4] and qh reshape((-1,1,32))>>
    // [0,2,4,6] super-block decomposition (the two 128-element halves
    // n in {0,128} of dequantize_row_q6_K). Verified against the gguf
    // package's dequantize_blocks reshape semantics:
    //   ql byte  = (i div 128)*64 + (i mod 64); low nibble when (i mod 128)
    //              < 64, else high nibble.
    //   qh byte  = (i div 128)*32 + (i mod 32); 2-bit field at shift
    //              2*((i mod 128) div 32).
    for i := 0 to GGUF_QK_K - 1 do
    begin
      ql := (QlPtr + ((i div 128) * 64 + (i mod 64)))^;
      if (i mod 128) < 64 then
        ql := ql and $0F
      else
        ql := (ql shr 4) and $0F;
      idx := (i div 128) * 32 + (i mod 32);
      qh := (QhPtr + idx)^;
      qh := (qh shr (2 * ((i mod 128) div 32))) and $03;
      qv := (ql or (qh shl 4)) - 32;
      g := i div 16;
      Outp[i] := d * Scales[g] * qv;
    end;
  end;
end;

{ TNNetGGUFReader }

constructor TNNetGGUFReader.Create(const pFileName: string);
var
  Stream: TFileStream;
begin
  inherited CreateBare; // Create(pFileName) would parse as safetensors
  FFileName := pFileName;
  if not FileExists(pFileName) then
    raise EGGUFError.CreateFmt('gguf: file not found: %s', [pFileName]);
  Stream := TFileStream.Create(pFileName, fmOpenRead or fmShareDenyWrite);
  // Single "shard": the inherited LoadTensorFlat machinery (and Destroy)
  // sees one open stream whose data section starts at the aligned offset
  // computed by ParseFile.
  SetLength(FStreams, 1);
  SetLength(FShardNames, 1);
  SetLength(FDataStarts, 1);
  SetLength(FDataSizes, 1);
  FStreams[0] := Stream; // owned (freed by the inherited Destroy)
  FShardNames[0] := pFileName;
  ParseFile(Stream);
end;

function TNNetGGUFReader.ReadHeaderString(Stream: TFileStream): string;
var
  Len: QWord;
begin
  Stream.ReadBuffer(Len, 8);
  if Len > QWord(Stream.Size) - QWord(Stream.Position) then
    raise EGGUFError.CreateFmt(
      'gguf: string length %d at offset %d runs past the end of file: %s',
      [Len, Stream.Position - 8, FFileName]);
  SetLength(Result, Len);
  if Len > 0 then Stream.ReadBuffer(Result[1], Len);
end;

procedure TNNetGGUFReader.ReadMetaValue(Stream: TFileStream;
  var Meta: TGGUFMetaValue);
var
  U8: byte;
  I8: shortint;
  U16: word;
  I16: smallint;
  U32: cardinal;
  I32: integer;
  U64: QWord;
  I64: Int64;
  F32: single;
  F64: double;
  Cnt: QWord;
  i: integer;

  // Reads one scalar of type TypeId into IntOut/FloatOut/StrOut.
  procedure ReadScalar(TypeId: integer; out IntOut: Int64;
    out FloatOut: double; out StrOut: string);
  begin
    IntOut := 0;
    FloatOut := 0;
    StrOut := '';
    case TypeId of
      GGUF_TYPE_UINT8:   begin Stream.ReadBuffer(U8, 1); IntOut := U8; end;
      GGUF_TYPE_INT8:    begin Stream.ReadBuffer(I8, 1); IntOut := I8; end;
      GGUF_TYPE_UINT16:  begin Stream.ReadBuffer(U16, 2); IntOut := U16; end;
      GGUF_TYPE_INT16:   begin Stream.ReadBuffer(I16, 2); IntOut := I16; end;
      GGUF_TYPE_UINT32:  begin Stream.ReadBuffer(U32, 4); IntOut := U32; end;
      GGUF_TYPE_INT32:   begin Stream.ReadBuffer(I32, 4); IntOut := I32; end;
      GGUF_TYPE_UINT64:  begin Stream.ReadBuffer(U64, 8);
                           IntOut := Int64(U64); end;
      GGUF_TYPE_INT64:   begin Stream.ReadBuffer(I64, 8); IntOut := I64; end;
      GGUF_TYPE_FLOAT32: begin Stream.ReadBuffer(F32, 4); FloatOut := F32; end;
      GGUF_TYPE_FLOAT64: begin Stream.ReadBuffer(F64, 8); FloatOut := F64; end;
      GGUF_TYPE_BOOL:    begin Stream.ReadBuffer(U8, 1); IntOut := U8; end;
      GGUF_TYPE_STRING:  StrOut := ReadHeaderString(Stream);
      else
        raise EGGUFError.CreateFmt(
          'gguf: metadata key "%s" has unsupported value type %d: %s',
          [Meta.Key, TypeId, FFileName]);
    end;
    if (TypeId <> GGUF_TYPE_FLOAT32) and (TypeId <> GGUF_TYPE_FLOAT64) then
      FloatOut := IntOut;
  end;

begin
  if Meta.ValueType = GGUF_TYPE_ARRAY then
  begin
    Stream.ReadBuffer(U32, 4);
    Meta.ArrElemType := integer(U32);
    if Meta.ArrElemType = GGUF_TYPE_ARRAY then
      raise EGGUFError.CreateFmt(
        'gguf: metadata key "%s" is a nested array (not supported): %s',
        [Meta.Key, FFileName]);
    Stream.ReadBuffer(Cnt, 8);
    if Cnt > QWord(High(integer)) then
      raise EGGUFError.CreateFmt(
        'gguf: metadata array "%s" is too large (%d elements): %s',
        [Meta.Key, Cnt, FFileName]);
    if Meta.ArrElemType = GGUF_TYPE_STRING then
      SetLength(Meta.ArrStr, Cnt)
    else
    begin
      SetLength(Meta.ArrInt, Cnt);
      SetLength(Meta.ArrNum, Cnt);
    end;
    for i := 0 to integer(Cnt) - 1 do
    begin
      ReadScalar(Meta.ArrElemType, I64, F64, Meta.StrVal);
      if Meta.ArrElemType = GGUF_TYPE_STRING then
        Meta.ArrStr[i] := Meta.StrVal
      else
      begin
        Meta.ArrInt[i] := I64;
        Meta.ArrNum[i] := F64;
      end;
    end;
    Meta.StrVal := '';
  end
  else
  begin
    ReadScalar(Meta.ValueType, Meta.IntVal, Meta.FloatVal, Meta.StrVal);
    Meta.BoolVal := Meta.IntVal <> 0;
  end;
end;

procedure TNNetGGUFReader.ParseFile(Stream: TFileStream);
var
  Magic, U32: cardinal;
  TensorCount, KVCount, U64: QWord;
  i, j, NDims: integer;
  Dims: array of Int64;
  NumElements, ByteSize, DataStart: Int64;
begin
  if Stream.Size < 24 then
    raise EGGUFError.CreateFmt(
      'gguf: file too small (%d bytes; need at least the 24-byte header): %s',
      [Stream.Size, FFileName]);
  Stream.ReadBuffer(Magic, 4);
  if Magic <> $46554747 then // 'GGUF' little-endian
    raise EGGUFError.CreateFmt(
      'gguf: bad magic 0x%s (expected "GGUF"): %s',
      [IntToHex(Magic, 8), FFileName]);
  Stream.ReadBuffer(U32, 4);
  FVersion := integer(U32);
  if (FVersion <> 2) and (FVersion <> 3) then
    raise EGGUFError.CreateFmt(
      'gguf: unsupported version %d (supported: 2, 3): %s',
      [FVersion, FFileName]);
  Stream.ReadBuffer(TensorCount, 8);
  Stream.ReadBuffer(KVCount, 8);
  if (TensorCount > QWord(High(integer))) or
     (KVCount > QWord(High(integer))) then
    raise EGGUFError.CreateFmt(
      'gguf: implausible header counts (%d tensors, %d KV pairs): %s',
      [TensorCount, KVCount, FFileName]);

  // ---- metadata KV pairs ----
  SetLength(FMeta, KVCount);
  for i := 0 to integer(KVCount) - 1 do
  begin
    FMeta[i].Key := ReadHeaderString(Stream);
    if FMeta[i].Key = '' then
      raise EGGUFError.CreateFmt(
        'gguf: empty metadata key (pair %d): %s', [i, FFileName]);
    if FindMeta(FMeta[i].Key) <> i then
      raise EGGUFError.CreateFmt(
        'gguf: duplicate metadata key "%s": %s', [FMeta[i].Key, FFileName]);
    Stream.ReadBuffer(U32, 4);
    FMeta[i].ValueType := integer(U32);
    ReadMetaValue(Stream, FMeta[i]);
  end;

  // ---- tensor infos ----
  SetLength(FTensors, TensorCount);
  SetLength(FGGMLTypes, TensorCount);
  Dims := nil;
  for i := 0 to integer(TensorCount) - 1 do
  begin
    FTensors[i].Name := ReadHeaderString(Stream);
    FTensors[i].Shard := 0;
    if FTensors[i].Name = '' then
      raise EGGUFError.CreateFmt(
        'gguf: empty tensor name (tensor %d): %s', [i, FFileName]);
    if FindTensor(FTensors[i].Name) <> i then
      raise EGGUFError.CreateFmt(
        'gguf: duplicate tensor name "%s": %s',
        [FTensors[i].Name, FFileName]);
    Stream.ReadBuffer(U32, 4);
    NDims := integer(U32);
    if (NDims < 1) or (NDims > 8) then
      raise EGGUFError.CreateFmt(
        'gguf: tensor "%s" has implausible n_dims %d: %s',
        [FTensors[i].Name, NDims, FFileName]);
    SetLength(Dims, NDims);
    NumElements := 1;
    for j := 0 to NDims - 1 do
    begin
      Stream.ReadBuffer(U64, 8);
      if (U64 = 0) or (U64 > QWord(High(integer))) then
        raise EGGUFError.CreateFmt(
          'gguf: tensor "%s" has implausible dimension %d: %s',
          [FTensors[i].Name, U64, FFileName]);
      Dims[j] := Int64(U64);
      NumElements := NumElements * Dims[j];
    end;
    // ggml stores the contiguous axis FIRST; the inherited API reports
    // row-major (contiguous axis LAST), so the dims are reversed: a HF
    // [out, in] matrix arrives as [in, out] and is served as [out, in].
    SetLength(FTensors[i].Shape, NDims);
    for j := 0 to NDims - 1 do
      FTensors[i].Shape[j] := Dims[NDims - 1 - j];
    Stream.ReadBuffer(U32, 4);
    FGGMLTypes[i] := integer(U32);
    if (FGGMLTypes[i] = GGML_TYPE_Q8_0) and
       ((Dims[0] mod GGUF_Q8_0_BLOCK_ELEMS) <> 0) then
      raise EGGUFError.CreateFmt(
        'gguf: tensor "%s" is Q8_0 but its contiguous dimension %d is not ' +
        'a multiple of the block size %d: %s',
        [FTensors[i].Name, Dims[0], GGUF_Q8_0_BLOCK_ELEMS, FFileName]);
    if ((FGGMLTypes[i] = GGML_TYPE_Q2_K) or
        (FGGMLTypes[i] = GGML_TYPE_Q4_K) or
        (FGGMLTypes[i] = GGML_TYPE_Q5_K) or
        (FGGMLTypes[i] = GGML_TYPE_Q6_K)) and
       ((Dims[0] mod GGUF_QK_K) <> 0) then
      raise EGGUFError.CreateFmt(
        'gguf: tensor "%s" is %s but its contiguous dimension %d is not a ' +
        'multiple of the k-quant super-block size %d: %s',
        [FTensors[i].Name, GGMLTypeName(FGGMLTypes[i]), Dims[0],
         GGUF_QK_K, FFileName]);
    Stream.ReadBuffer(U64, 8); // offset relative to the data section
    FTensors[i].DataBegin := Int64(U64);
    ByteSize := GGMLByteSize(FGGMLTypes[i], NumElements);
    // Unsupported dtypes keep DataEnd = DataBegin; LoadTensorFlat raises a
    // descriptive error for them, but the table (names/shapes/types) stays
    // inspectable.
    FTensors[i].DataEnd := FTensors[i].DataBegin + ByteSize;
    FTensors[i].DType := GGMLTypeName(FGGMLTypes[i]);
  end;

  // ---- alignment padding, then the data section ----
  FAlignment := integer(GetMetaInt('general.alignment', 32));
  if (FAlignment < 1) or ((FAlignment and (FAlignment - 1)) <> 0) then
    raise EGGUFError.CreateFmt(
      'gguf: general.alignment %d is not a positive power of two: %s',
      [FAlignment, FFileName]);
  DataStart := ((Stream.Position + FAlignment - 1) div FAlignment) *
    FAlignment;
  if DataStart > Stream.Size then
    raise EGGUFError.CreateFmt(
      'gguf: header ends at %d past the end of the %d-byte file: %s',
      [DataStart, Stream.Size, FFileName]);
  FDataStarts[0] := DataStart;
  FDataSizes[0] := Stream.Size - DataStart;
  for i := 0 to High(FTensors) do
    if (FTensors[i].DataBegin < 0) or
       (FTensors[i].DataEnd > FDataSizes[0]) then
      raise EGGUFError.CreateFmt(
        'gguf: tensor "%s" data [%d, %d) falls outside the data section ' +
        '(size %d): %s', [FTensors[i].Name, FTensors[i].DataBegin,
         FTensors[i].DataEnd, FDataSizes[0], FFileName]);
end;

function TNNetGGUFReader.FindMeta(const pKey: string): integer;
var
  i: integer;
begin
  for i := 0 to High(FMeta) do
    if FMeta[i].Key = pKey then exit(i);
  Result := -1;
end;

function TNNetGGUFReader.MetaCount: integer;
begin
  Result := Length(FMeta);
end;

function TNNetGGUFReader.MetaKey(Index: integer): string;
begin
  if (Index < 0) or (Index > High(FMeta)) then
    raise EGGUFError.CreateFmt(
      'gguf: metadata index %d out of range (0..%d): %s',
      [Index, High(FMeta), FFileName]);
  Result := FMeta[Index].Key;
end;

function TNNetGGUFReader.HasMetaKey(const pKey: string): boolean;
begin
  Result := FindMeta(pKey) >= 0;
end;

function TNNetGGUFReader.GetMetaInt(const pKey: string;
  Default: Int64): Int64;
var
  Idx: integer;
begin
  Idx := FindMeta(pKey);
  if Idx < 0 then exit(Default);
  case FMeta[Idx].ValueType of
    GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_UINT64, GGUF_TYPE_INT64,
    GGUF_TYPE_BOOL:
      Result := FMeta[Idx].IntVal;
    else
      raise EGGUFError.CreateFmt(
        'gguf: metadata key "%s" (type %d) is not an integer: %s',
        [pKey, FMeta[Idx].ValueType, FFileName]);
  end;
end;

function TNNetGGUFReader.GetMetaFloat(const pKey: string;
  Default: double): double;
var
  Idx: integer;
begin
  Idx := FindMeta(pKey);
  if Idx < 0 then exit(Default);
  case FMeta[Idx].ValueType of
    GGUF_TYPE_FLOAT32, GGUF_TYPE_FLOAT64:
      Result := FMeta[Idx].FloatVal;
    GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_UINT64, GGUF_TYPE_INT64:
      Result := FMeta[Idx].IntVal;
    else
      raise EGGUFError.CreateFmt(
        'gguf: metadata key "%s" (type %d) is not a number: %s',
        [pKey, FMeta[Idx].ValueType, FFileName]);
  end;
end;

function TNNetGGUFReader.GetMetaString(const pKey: string;
  const Default: string): string;
var
  Idx: integer;
begin
  Idx := FindMeta(pKey);
  if Idx < 0 then exit(Default);
  if FMeta[Idx].ValueType <> GGUF_TYPE_STRING then
    raise EGGUFError.CreateFmt(
      'gguf: metadata key "%s" (type %d) is not a string: %s',
      [pKey, FMeta[Idx].ValueType, FFileName]);
  Result := FMeta[Idx].StrVal;
end;

function TNNetGGUFReader.GetMetaBool(const pKey: string;
  Default: boolean): boolean;
var
  Idx: integer;
begin
  Idx := FindMeta(pKey);
  if Idx < 0 then exit(Default);
  if FMeta[Idx].ValueType <> GGUF_TYPE_BOOL then
    raise EGGUFError.CreateFmt(
      'gguf: metadata key "%s" (type %d) is not a bool: %s',
      [pKey, FMeta[Idx].ValueType, FFileName]);
  Result := FMeta[Idx].BoolVal;
end;

function TNNetGGUFReader.GetMetaArrayCount(const pKey: string): Int64;
var
  Idx: integer;
begin
  Idx := FindMeta(pKey);
  if Idx < 0 then exit(0);
  if FMeta[Idx].ValueType <> GGUF_TYPE_ARRAY then
    raise EGGUFError.CreateFmt(
      'gguf: metadata key "%s" (type %d) is not an array: %s',
      [pKey, FMeta[Idx].ValueType, FFileName]);
  if FMeta[Idx].ArrElemType = GGUF_TYPE_STRING then
    Result := Length(FMeta[Idx].ArrStr)
  else
    Result := Length(FMeta[Idx].ArrInt);
end;

function TNNetGGUFReader.GetMetaArrayString(const pKey: string;
  Index: integer): string;
var
  Idx: integer;
begin
  Idx := FindMeta(pKey);
  if (Idx < 0) or (FMeta[Idx].ValueType <> GGUF_TYPE_ARRAY) or
     (FMeta[Idx].ArrElemType <> GGUF_TYPE_STRING) then
    raise EGGUFError.CreateFmt(
      'gguf: metadata key "%s" is not a string array: %s', [pKey, FFileName]);
  if (Index < 0) or (Index > High(FMeta[Idx].ArrStr)) then
    raise EGGUFError.CreateFmt(
      'gguf: array index %d out of range for metadata key "%s": %s',
      [Index, pKey, FFileName]);
  Result := FMeta[Idx].ArrStr[Index];
end;

function TNNetGGUFReader.GetMetaArrayNumber(const pKey: string;
  Index: integer): double;
var
  Idx: integer;
begin
  Idx := FindMeta(pKey);
  if (Idx < 0) or (FMeta[Idx].ValueType <> GGUF_TYPE_ARRAY) or
     (FMeta[Idx].ArrElemType = GGUF_TYPE_STRING) then
    raise EGGUFError.CreateFmt(
      'gguf: metadata key "%s" is not a numeric array: %s',
      [pKey, FFileName]);
  if (Index < 0) or (Index > High(FMeta[Idx].ArrNum)) then
    raise EGGUFError.CreateFmt(
      'gguf: array index %d out of range for metadata key "%s": %s',
      [Index, pKey, FFileName]);
  Result := FMeta[Idx].ArrNum[Index];
end;

function TNNetGGUFReader.TensorGGMLType(const pName: string): integer;
var
  Idx: integer;
begin
  Idx := FindTensor(pName);
  if Idx < 0 then
    raise EGGUFError.CreateFmt(
      'gguf: tensor "%s" not found in %s', [pName, FFileName]);
  Result := FGGMLTypes[Idx];
end;

procedure TNNetGGUFReader.RenameTensor(const pOldName, pNewName: string);
var
  Idx: integer;
begin
  Idx := FindTensor(pOldName);
  if Idx < 0 then
    raise EGGUFError.CreateFmt(
      'gguf: cannot rename missing tensor "%s": %s', [pOldName, FFileName]);
  if FindTensor(pNewName) >= 0 then
    raise EGGUFError.CreateFmt(
      'gguf: cannot rename "%s" to "%s" - the target name already exists: %s',
      [pOldName, pNewName, FFileName]);
  FTensors[Idx].Name := pNewName;
end;

procedure TNNetGGUFReader.RegisterRowDeinterleave(const pName: string;
  HeadDim: integer);
var
  Idx, Rows: integer;
begin
  Idx := FindTensor(pName);
  if Idx < 0 then
    raise EGGUFError.CreateFmt(
      'gguf: cannot register de-interleave for missing tensor "%s": %s',
      [pName, FFileName]);
  if (HeadDim < 2) or Odd(HeadDim) then
    raise EGGUFError.CreateFmt(
      'gguf: de-interleave head_dim %d for "%s" must be an even number ' +
      '>= 2: %s', [HeadDim, pName, FFileName]);
  if Length(FTensors[Idx].Shape) <> 2 then
    raise EGGUFError.CreateFmt(
      'gguf: de-interleave target "%s" must be 2-D, got %s: %s',
      [pName, ShapeAsString(pName), FFileName]);
  Rows := integer(FTensors[Idx].Shape[0]);
  if (Rows mod HeadDim) <> 0 then
    raise EGGUFError.CreateFmt(
      'gguf: de-interleave target "%s" has %d rows - not a multiple of ' +
      'head_dim %d: %s', [pName, Rows, HeadDim, FFileName]);
  SetLength(FDeinterleaveNames, Length(FDeinterleaveNames) + 1);
  SetLength(FDeinterleaveHeadDim, Length(FDeinterleaveHeadDim) + 1);
  FDeinterleaveNames[High(FDeinterleaveNames)] := pName;
  FDeinterleaveHeadDim[High(FDeinterleaveHeadDim)] := HeadDim;
end;

function TNNetGGUFReader.DeinterleaveHeadDimFor(
  const pName: string): integer;
var
  i: integer;
begin
  for i := 0 to High(FDeinterleaveNames) do
    if FDeinterleaveNames[i] = pName then
      exit(FDeinterleaveHeadDim[i]);
  Result := 0;
end;

procedure TNNetGGUFReader.LoadTensorFlat(const pName: string;
  Dest: TNNetVolume);
var
  Idx, GGMLType, HeadDim, HalfDim: integer;
  NumElements, i, BlockCnt, NumBlocks: Int64;
  Rows, RowLen, r, RowInHead, SrcRow: Int64;
  RawBytes: TBytes;
  Scale: single;
  WordPtr: PWord;
  SinglePtr: PSingle;
  QuantPtr: PShortInt;
  Tmp: array of TNeuralFloat;
begin
  Idx := FindTensor(pName);
  if Idx < 0 then
    raise EGGUFError.CreateFmt(
      'gguf: tensor "%s" not found in %s', [pName, FFileName]);
  GGMLType := FGGMLTypes[Idx];
  if (GGMLType <> GGML_TYPE_F32) and (GGMLType <> GGML_TYPE_F16) and
     (GGMLType <> GGML_TYPE_Q8_0) and (GGMLType <> GGML_TYPE_Q2_K) and
     (GGMLType <> GGML_TYPE_Q4_K) and (GGMLType <> GGML_TYPE_Q5_K) and
     (GGMLType <> GGML_TYPE_Q6_K) then
    raise EGGUFError.CreateFmt(
      'gguf: tensor "%s" has unsupported ggml dtype %s (supported: F32, ' +
      'F16, Q8_0, Q2_K, Q4_K, Q5_K, Q6_K): %s',
      [pName, GGMLTypeName(GGMLType), FFileName]);
  NumElements := ElementCount(pName);
  if NumElements > High(integer) then
    raise EGGUFError.CreateFmt(
      'gguf: tensor "%s" is too large (%d elements): %s',
      [pName, NumElements, FFileName]);
  Dest.ReSize(integer(NumElements), 1, 1);
  if NumElements = 0 then exit;
  SetLength(RawBytes, FTensors[Idx].DataEnd - FTensors[Idx].DataBegin);
  FStreams[0].Position := FDataStarts[0] + FTensors[Idx].DataBegin;
  FStreams[0].ReadBuffer(RawBytes[0], Length(RawBytes));
  case GGMLType of
    GGML_TYPE_F32:
    begin
      SinglePtr := PSingle(@RawBytes[0]);
      for i := 0 to NumElements - 1 do
      begin
        Dest.FData[i] := SinglePtr^;
        Inc(SinglePtr);
      end;
    end;
    GGML_TYPE_F16:
    begin
      WordPtr := PWord(@RawBytes[0]);
      for i := 0 to NumElements - 1 do
      begin
        Dest.FData[i] := DecodeF16(WordPtr^);
        Inc(WordPtr);
      end;
    end;
    GGML_TYPE_Q8_0:
    begin
      // Blocks of 32 elements along the contiguous axis: f16 scale d,
      // then 32 int8 quants; x = d * q. The contiguous dimension is a
      // multiple of 32 (validated at parse), so blocks never straddle
      // rows and a sequential sweep decodes the flat row-major order.
      NumBlocks := NumElements div GGUF_Q8_0_BLOCK_ELEMS;
      for BlockCnt := 0 to NumBlocks - 1 do
      begin
        Scale := DecodeF16(
          PWord(@RawBytes[BlockCnt * GGUF_Q8_0_BLOCK_BYTES])^);
        QuantPtr := PShortInt(
          @RawBytes[BlockCnt * GGUF_Q8_0_BLOCK_BYTES + 2]);
        for i := 0 to GGUF_Q8_0_BLOCK_ELEMS - 1 do
        begin
          Dest.FData[BlockCnt * GGUF_Q8_0_BLOCK_ELEMS + i] :=
            Scale * QuantPtr^;
          Inc(QuantPtr);
        end;
      end;
    end;
    GGML_TYPE_Q4_K:
    begin
      // k-quant super-blocks of 256 along the contiguous axis (a multiple
      // of 256, validated at parse), so a sequential sweep decodes the
      // flat row-major order. Each block: f16 d, f16 d_min, 12 packed
      // 6-bit sub-scales/sub-mins, 128 bytes of 4-bit quants.
      NumBlocks := NumElements div GGUF_QK_K;
      DequantizeQ4K(PByte(@RawBytes[0]), NumBlocks, PSingle(@Dest.FData[0]));
    end;
    GGML_TYPE_Q5_K:
    begin
      // Like Q4_K plus a 32-byte 5th-bit plane: 176-byte super-blocks.
      NumBlocks := NumElements div GGUF_QK_K;
      DequantizeQ5K(PByte(@RawBytes[0]), NumBlocks, PSingle(@Dest.FData[0]));
    end;
    GGML_TYPE_Q2_K:
    begin
      // 2-bit quants with 4-bit packed sub-scales/sub-mins: 84-byte blocks.
      NumBlocks := NumElements div GGUF_QK_K;
      DequantizeQ2K(PByte(@RawBytes[0]), NumBlocks, PSingle(@Dest.FData[0]));
    end;
    GGML_TYPE_Q6_K:
    begin
      NumBlocks := NumElements div GGUF_QK_K;
      DequantizeQ6K(PByte(@RawBytes[0]), NumBlocks, PSingle(@Dest.FData[0]));
    end;
  end;
  // Registered q/k projections: undo llama.cpp's per-head interleaved-
  // rotary row permutation so the served rows are in HF rotate_half order
  // (hf_row[p] = stored[2p], hf_row[p + HeadDim/2] = stored[2p+1]).
  HeadDim := DeinterleaveHeadDimFor(pName);
  if HeadDim > 0 then
  begin
    Rows := FTensors[Idx].Shape[0];
    RowLen := NumElements div Rows;
    HalfDim := HeadDim div 2;
    SetLength(Tmp, NumElements);
    for i := 0 to NumElements - 1 do Tmp[i] := Dest.FData[i];
    for r := 0 to Rows - 1 do
    begin
      RowInHead := r mod HeadDim;
      if RowInHead < HalfDim then
        SrcRow := (r - RowInHead) + 2 * RowInHead
      else
        SrcRow := (r - RowInHead) + 2 * (RowInHead - HalfDim) + 1;
      for i := 0 to RowLen - 1 do
        Dest.FData[r * RowLen + i] := Tmp[SrcRow * RowLen + i];
    end;
  end;
end;

{ TNNetGGUFWriter }

constructor TNNetGGUFWriter.Create(const pFileName: string);
begin
  inherited Create;
  FFileName := pFileName;
  FAlignment := 32;
  FSaved := false;
end;

destructor TNNetGGUFWriter.Destroy;
begin
  inherited Destroy;
end;

function TNNetGGUFWriter.FindMeta(const pKey: string): integer;
var
  i: integer;
begin
  for i := 0 to High(FMeta) do
    if FMeta[i].Key = pKey then exit(i);
  Result := -1;
end;

function TNNetGGUFWriter.FindTensor(const pName: string): integer;
var
  i: integer;
begin
  for i := 0 to High(FTensors) do
    if FTensors[i].Name = pName then exit(i);
  Result := -1;
end;

procedure TNNetGGUFWriter.AddMeta(const Meta: TGGUFMetaValue);
begin
  if Meta.Key = '' then
    raise EGGUFError.Create('gguf writer: empty metadata key.');
  if FindMeta(Meta.Key) >= 0 then
    raise EGGUFError.CreateFmt(
      'gguf writer: duplicate metadata key "%s".', [Meta.Key]);
  SetLength(FMeta, Length(FMeta) + 1);
  FMeta[High(FMeta)] := Meta;
end;

procedure TNNetGGUFWriter.AddMetaUInt32(const pKey: string; Value: cardinal);
var
  M: TGGUFMetaValue;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_UINT32; M.IntVal := Value;
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddMetaInt32(const pKey: string; Value: integer);
var
  M: TGGUFMetaValue;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_INT32; M.IntVal := Value;
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddMetaUInt64(const pKey: string; Value: QWord);
var
  M: TGGUFMetaValue;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_UINT64; M.IntVal := Int64(Value);
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddMetaFloat32(const pKey: string; Value: single);
var
  M: TGGUFMetaValue;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_FLOAT32; M.FloatVal := Value;
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddMetaBool(const pKey: string; Value: boolean);
var
  M: TGGUFMetaValue;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_BOOL;
  M.BoolVal := Value;
  if Value then M.IntVal := 1 else M.IntVal := 0;
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddMetaString(const pKey, pValue: string);
var
  M: TGGUFMetaValue;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_STRING; M.StrVal := pValue;
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddMetaStringArray(const pKey: string;
  const Values: array of string);
var
  M: TGGUFMetaValue;
  i: integer;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_ARRAY;
  M.ArrElemType := GGUF_TYPE_STRING;
  SetLength(M.ArrStr, Length(Values));
  for i := 0 to High(Values) do M.ArrStr[i] := Values[i];
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddMetaFloat32Array(const pKey: string;
  const Values: array of single);
var
  M: TGGUFMetaValue;
  i: integer;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_ARRAY;
  M.ArrElemType := GGUF_TYPE_FLOAT32;
  SetLength(M.ArrNum, Length(Values));
  SetLength(M.ArrInt, Length(Values));
  for i := 0 to High(Values) do M.ArrNum[i] := Values[i];
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddMetaInt32Array(const pKey: string;
  const Values: array of Int64);
var
  M: TGGUFMetaValue;
  i: integer;
begin
  M := Default(TGGUFMetaValue);
  M.Key := pKey; M.ValueType := GGUF_TYPE_ARRAY;
  M.ArrElemType := GGUF_TYPE_INT32;
  SetLength(M.ArrInt, Length(Values));
  SetLength(M.ArrNum, Length(Values));
  for i := 0 to High(Values) do M.ArrInt[i] := Values[i];
  AddMeta(M);
end;

procedure TNNetGGUFWriter.AddTensorFlat(const pName: string;
  const pShape: array of Int64; Src: TNNetVolume; pDType: TGGUFWriteDType);
var
  Idx, i, ContigDim: integer;
  NumElements, NumBlocks, b, e: Int64;
  WordPtr: PWord;
  SinglePtr: PSingle;
  QuantPtr: PShortInt;
  ScalePtr: PWord;
  AbsMax, V, Scale, InvScale, Q: single;
  Pending: TGGUFPendingTensor;
begin
  if FSaved then
    raise EGGUFError.Create('gguf writer: AddTensorFlat after SaveToFile.');
  if pName = '' then
    raise EGGUFError.Create('gguf writer: empty tensor name.');
  if FindTensor(pName) >= 0 then
    raise EGGUFError.CreateFmt(
      'gguf writer: duplicate tensor name "%s".', [pName]);
  if Length(pShape) = 0 then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" needs at least one dimension.', [pName]);
  NumElements := 1;
  for i := 0 to High(pShape) do
  begin
    if pShape[i] <= 0 then
      raise EGGUFError.CreateFmt(
        'gguf writer: tensor "%s" has a non-positive dimension %d.',
        [pName, pShape[i]]);
    NumElements := NumElements * pShape[i];
  end;
  if NumElements <> Src.Size then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" shape implies %d elements but the source ' +
      'holds %d.', [pName, NumElements, Src.Size]);

  Pending := Default(TGGUFPendingTensor);
  Pending.Name := pName;
  SetLength(Pending.Shape, Length(pShape));
  for i := 0 to High(pShape) do Pending.Shape[i] := pShape[i];

  case pDType of
    gwF32:
    begin
      Pending.GGMLType := GGML_TYPE_F32;
      SetLength(Pending.Data, NumElements * 4);
      SinglePtr := PSingle(@Pending.Data[0]);
      for i := 0 to NumElements - 1 do
      begin
        SinglePtr^ := Src.FData[i];
        Inc(SinglePtr);
      end;
    end;
    gwF16:
    begin
      Pending.GGMLType := GGML_TYPE_F16;
      SetLength(Pending.Data, NumElements * 2);
      WordPtr := PWord(@Pending.Data[0]);
      for i := 0 to NumElements - 1 do
      begin
        WordPtr^ := EncodeF16(Src.FData[i]);
        Inc(WordPtr);
      end;
    end;
    gwQ8_0:
    begin
      ContigDim := integer(pShape[High(pShape)]);
      if (ContigDim mod GGUF_Q8_0_BLOCK_ELEMS) <> 0 then
        raise EGGUFError.CreateFmt(
          'gguf writer: tensor "%s" is Q8_0 but its contiguous dimension ' +
          '%d is not a multiple of the block size %d.',
          [pName, ContigDim, GGUF_Q8_0_BLOCK_ELEMS]);
      Pending.GGMLType := GGML_TYPE_Q8_0;
      NumBlocks := NumElements div GGUF_Q8_0_BLOCK_ELEMS;
      SetLength(Pending.Data, NumBlocks * GGUF_Q8_0_BLOCK_BYTES);
      for b := 0 to NumBlocks - 1 do
      begin
        // ggml quantize_row_q8_0: scale = max|x| / 127, quants = round(x/scale)
        // clamped to int8; the scale is stored as f16. Mirrors gguf's
        // quants.quantize(..., Q8_0) so a reader round-trip is bit-stable.
        AbsMax := 0;
        for i := 0 to GGUF_Q8_0_BLOCK_ELEMS - 1 do
        begin
          V := Abs(Src.FData[b * GGUF_Q8_0_BLOCK_ELEMS + i]);
          if V > AbsMax then AbsMax := V;
        end;
        Scale := AbsMax / 127.0;
        if Scale = 0 then InvScale := 0 else InvScale := 1.0 / Scale;
        ScalePtr := PWord(@Pending.Data[b * GGUF_Q8_0_BLOCK_BYTES]);
        ScalePtr^ := EncodeF16(Scale);
        QuantPtr := PShortInt(@Pending.Data[b * GGUF_Q8_0_BLOCK_BYTES + 2]);
        for i := 0 to GGUF_Q8_0_BLOCK_ELEMS - 1 do
        begin
          Q := Src.FData[b * GGUF_Q8_0_BLOCK_ELEMS + i] * InvScale;
          // round-half-away-from-zero then clamp to [-127, 127]
          if Q >= 0 then e := Trunc(Q + 0.5) else e := Trunc(Q - 0.5);
          if e > 127 then e := 127;
          if e < -127 then e := -127;
          QuantPtr^ := ShortInt(e);
          Inc(QuantPtr);
        end;
      end;
    end;
  end;

  Idx := Length(FTensors);
  SetLength(FTensors, Idx + 1);
  FTensors[Idx] := Pending;
end;

procedure TNNetGGUFWriter.AddTensorFlatInt8(const pName: string;
  const pShape: array of Int64; const pCodes: array of ShortInt;
  const pScales: array of single; NumRows, VS: integer);
var
  Idx, i, ContigDim, r, AbsMaxCode, c: integer;
  NumElements, NumBlocks, RowBlocks, b, RowBase, BlockBase, e: Int64;
  QuantPtr: PShortInt;
  ScalePtr: PWord;
  Scale, Q: single;
  Pending: TGGUFPendingTensor;
begin
  if FSaved then
    raise EGGUFError.Create('gguf writer: AddTensorFlatInt8 after SaveToFile.');
  if pName = '' then
    raise EGGUFError.Create('gguf writer: empty tensor name.');
  if FindTensor(pName) >= 0 then
    raise EGGUFError.CreateFmt(
      'gguf writer: duplicate tensor name "%s".', [pName]);
  if Length(pShape) = 0 then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" needs at least one dimension.', [pName]);
  if (NumRows <= 0) or (VS <= 0) then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" int8 source has non-positive geometry ' +
      '(%d rows x %d).', [pName, NumRows, VS]);
  NumElements := 1;
  for i := 0 to High(pShape) do
  begin
    if pShape[i] <= 0 then
      raise EGGUFError.CreateFmt(
        'gguf writer: tensor "%s" has a non-positive dimension %d.',
        [pName, pShape[i]]);
    NumElements := NumElements * pShape[i];
  end;
  if NumElements <> Int64(NumRows) * VS then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" shape implies %d elements but the int8 ' +
      'source holds %d rows x %d = %d.',
      [pName, NumElements, NumRows, VS, Int64(NumRows) * VS]);
  if Length(pCodes) < NumRows * VS then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" int8 source has %d codes, expected %d.',
      [pName, Length(pCodes), NumRows * VS]);
  if Length(pScales) < NumRows then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" int8 source has %d scales, expected %d.',
      [pName, Length(pScales), NumRows]);
  // The int8 storage is per-row symmetric; a Q8_0 32-block must stay inside
  // one row so its f16 scale derives from a single row scale. The contiguous
  // (last) dimension MUST equal the int8 vector size AND be a 32 multiple.
  ContigDim := integer(pShape[High(pShape)]);
  if ContigDim <> VS then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" Q8_0-from-int8 needs the contiguous ' +
      'dimension (%d) to equal the int8 vector size (%d) so 32-blocks do ' +
      'not cross row boundaries.', [pName, ContigDim, VS]);
  if (VS mod GGUF_Q8_0_BLOCK_ELEMS) <> 0 then
    raise EGGUFError.CreateFmt(
      'gguf writer: tensor "%s" is Q8_0 but its contiguous dimension %d is ' +
      'not a multiple of the block size %d.',
      [pName, VS, GGUF_Q8_0_BLOCK_ELEMS]);

  Pending := Default(TGGUFPendingTensor);
  Pending.Name := pName;
  SetLength(Pending.Shape, Length(pShape));
  for i := 0 to High(pShape) do Pending.Shape[i] := pShape[i];
  Pending.GGMLType := GGML_TYPE_Q8_0;
  NumBlocks := NumElements div GGUF_Q8_0_BLOCK_ELEMS;
  SetLength(Pending.Data, NumBlocks * GGUF_Q8_0_BLOCK_BYTES);
  RowBlocks := VS div GGUF_Q8_0_BLOCK_ELEMS;
  for r := 0 to NumRows - 1 do
  begin
    RowBase := Int64(r) * VS;
    for b := 0 to RowBlocks - 1 do
    begin
      BlockBase := RowBase + b * GGUF_Q8_0_BLOCK_ELEMS;
      // Per-block absmax over the int8 CODES; the true FP32 value is
      // code * RowScale, so max|x| = AbsMaxCode * RowScale and the Q8_0
      // block scale = max|x|/127. The row scale rides into d only; the
      // quants q[i] = round(code[i]*127 / AbsMaxCode) are scale-invariant.
      AbsMaxCode := 0;
      for i := 0 to GGUF_Q8_0_BLOCK_ELEMS - 1 do
      begin
        c := pCodes[BlockBase + i];
        if c < 0 then c := -c;
        if c > AbsMaxCode then AbsMaxCode := c;
      end;
      Scale := (AbsMaxCode * pScales[r]) / 127.0;
      ScalePtr := PWord(@Pending.Data[(Int64(r) * RowBlocks + b) *
        GGUF_Q8_0_BLOCK_BYTES]);
      ScalePtr^ := EncodeF16(Scale);
      QuantPtr := PShortInt(@Pending.Data[(Int64(r) * RowBlocks + b) *
        GGUF_Q8_0_BLOCK_BYTES + 2]);
      for i := 0 to GGUF_Q8_0_BLOCK_ELEMS - 1 do
      begin
        if AbsMaxCode = 0 then
          e := 0
        else
        begin
          Q := (pCodes[BlockBase + i] * 127.0) / AbsMaxCode;
          // round-half-away-from-zero then clamp to [-127, 127]
          if Q >= 0 then e := Trunc(Q + 0.5) else e := Trunc(Q - 0.5);
          if e > 127 then e := 127;
          if e < -127 then e := -127;
        end;
        QuantPtr^ := ShortInt(e);
        Inc(QuantPtr);
      end;
    end;
  end;

  Idx := Length(FTensors);
  SetLength(FTensors, Idx + 1);
  FTensors[Idx] := Pending;
end;

function TNNetGGUFWriter.Count: integer;
begin
  Result := Length(FTensors);
end;

procedure TNNetGGUFWriter.WriteHeaderString(Stream: TStream;
  const S: string);
var
  Len: QWord;
begin
  Len := Length(S);
  Stream.WriteBuffer(Len, 8);
  if Len > 0 then Stream.WriteBuffer(S[1], Len);
end;

procedure TNNetGGUFWriter.WriteMetaValue(Stream: TStream;
  const Meta: TGGUFMetaValue);
var
  U8: byte;
  U32: cardinal;
  U64: QWord;
  Cnt: QWord;
  i: integer;
  ElemType: cardinal;

  procedure WriteScalar(TypeId: integer; const M: TGGUFMetaValue;
    StrIdx, NumIdx: integer);
  var
    lU8: byte;
    lU32: cardinal;
    lU64: QWord;
    lF32: single;
  begin
    case TypeId of
      GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL:
        begin lU8 := byte(M.IntVal); Stream.WriteBuffer(lU8, 1); end;
      GGUF_TYPE_UINT16, GGUF_TYPE_INT16:
        begin lU32 := cardinal(M.IntVal);
          Stream.WriteBuffer(lU32, 2); end;
      GGUF_TYPE_UINT32, GGUF_TYPE_INT32:
        begin lU32 := cardinal(M.IntVal);
          Stream.WriteBuffer(lU32, 4); end;
      GGUF_TYPE_UINT64, GGUF_TYPE_INT64:
        begin lU64 := QWord(M.IntVal); Stream.WriteBuffer(lU64, 8); end;
      GGUF_TYPE_FLOAT32:
        begin
          if NumIdx >= 0 then lF32 := M.ArrNum[NumIdx] else lF32 := M.FloatVal;
          Stream.WriteBuffer(lF32, 4);
        end;
      GGUF_TYPE_STRING:
        if StrIdx >= 0 then WriteHeaderString(Stream, M.ArrStr[StrIdx])
        else WriteHeaderString(Stream, M.StrVal);
      else
        raise EGGUFError.CreateFmt(
          'gguf writer: cannot write metadata "%s" value type %d.',
          [M.Key, TypeId]);
    end;
  end;

begin
  if Meta.ValueType = GGUF_TYPE_ARRAY then
  begin
    ElemType := cardinal(Meta.ArrElemType);
    Stream.WriteBuffer(ElemType, 4);
    if Meta.ArrElemType = GGUF_TYPE_STRING then
      Cnt := Length(Meta.ArrStr)
    else
      Cnt := Length(Meta.ArrInt);
    if Meta.ArrElemType = GGUF_TYPE_FLOAT32 then
      Cnt := Length(Meta.ArrNum);
    Stream.WriteBuffer(Cnt, 8);
    for i := 0 to integer(Cnt) - 1 do
    begin
      case Meta.ArrElemType of
        GGUF_TYPE_STRING: WriteScalar(GGUF_TYPE_STRING, Meta, i, -1);
        GGUF_TYPE_FLOAT32: WriteScalar(GGUF_TYPE_FLOAT32, Meta, -1, i);
        GGUF_TYPE_UINT32, GGUF_TYPE_INT32:
          begin U32 := cardinal(Meta.ArrInt[i]); Stream.WriteBuffer(U32, 4); end;
        GGUF_TYPE_UINT64, GGUF_TYPE_INT64:
          begin U64 := QWord(Meta.ArrInt[i]); Stream.WriteBuffer(U64, 8); end;
        GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL:
          begin U8 := byte(Meta.ArrInt[i]); Stream.WriteBuffer(U8, 1); end;
        else
          raise EGGUFError.CreateFmt(
            'gguf writer: cannot write array element type %d for "%s".',
            [Meta.ArrElemType, Meta.Key]);
      end;
    end;
  end
  else
    WriteScalar(Meta.ValueType, Meta, -1, -1);
end;

procedure TNNetGGUFWriter.SaveToFile;
var
  Stream: TFileStream;
  Magic, U32: cardinal;
  U64, Offset: QWord;
  TensorCount, KVCount: QWord;
  i, j, NDims: integer;
  Pad: array of byte;
  PadLen: Int64;
  HeaderEnd, DataStart: Int64;
begin
  if FSaved then
    raise EGGUFError.Create('gguf writer: SaveToFile called twice.');
  FSaved := true;
  if (FAlignment < 1) or ((FAlignment and (FAlignment - 1)) <> 0) then
    raise EGGUFError.CreateFmt(
      'gguf writer: alignment %d is not a positive power of two.',
      [FAlignment]);
  Stream := TFileStream.Create(FFileName, fmCreate);
  try
    // ---- header ----
    Magic := $46554747; // 'GGUF' little-endian
    Stream.WriteBuffer(Magic, 4);
    U32 := 3; // version 3
    Stream.WriteBuffer(U32, 4);
    TensorCount := Length(FTensors);
    KVCount := Length(FMeta);
    Stream.WriteBuffer(TensorCount, 8);
    Stream.WriteBuffer(KVCount, 8);

    // ---- metadata KV pairs ----
    for i := 0 to High(FMeta) do
    begin
      WriteHeaderString(Stream, FMeta[i].Key);
      U32 := cardinal(FMeta[i].ValueType);
      Stream.WriteBuffer(U32, 4);
      WriteMetaValue(Stream, FMeta[i]);
    end;

    // ---- tensor infos (offsets computed below, relative to data section) ----
    // First pass: compute each tensor's aligned offset into the data blob.
    Offset := 0;
    for i := 0 to High(FTensors) do
    begin
      WriteHeaderString(Stream, FTensors[i].Name);
      NDims := Length(FTensors[i].Shape);
      U32 := cardinal(NDims);
      Stream.WriteBuffer(U32, 4);
      // ggml stores the contiguous axis FIRST: reverse the row-major shape.
      for j := 0 to NDims - 1 do
      begin
        U64 := QWord(FTensors[i].Shape[NDims - 1 - j]);
        Stream.WriteBuffer(U64, 8);
      end;
      U32 := cardinal(FTensors[i].GGMLType);
      Stream.WriteBuffer(U32, 4);
      Stream.WriteBuffer(Offset, 8);
      // Advance the data offset, aligning each tensor to FAlignment.
      Inc(Offset, QWord(Length(FTensors[i].Data)));
      Offset := ((Offset + QWord(FAlignment) - 1) div QWord(FAlignment)) *
        QWord(FAlignment);
    end;

    // ---- alignment padding, then the tensor data blob ----
    HeaderEnd := Stream.Position;
    DataStart := ((HeaderEnd + FAlignment - 1) div FAlignment) * FAlignment;
    PadLen := DataStart - HeaderEnd;
    if PadLen > 0 then
    begin
      SetLength(Pad, PadLen);
      FillChar(Pad[0], PadLen, 0);
      Stream.WriteBuffer(Pad[0], PadLen);
    end;
    // Write each tensor's payload, padding between tensors to FAlignment.
    for i := 0 to High(FTensors) do
    begin
      if Length(FTensors[i].Data) > 0 then
        Stream.WriteBuffer(FTensors[i].Data[0], Length(FTensors[i].Data));
      if i < High(FTensors) then
      begin
        PadLen := Int64(FAlignment) -
          (Int64(Length(FTensors[i].Data)) mod FAlignment);
        if PadLen = FAlignment then PadLen := 0;
        if PadLen > 0 then
        begin
          SetLength(Pad, PadLen);
          FillChar(Pad[0], PadLen, 0);
          Stream.WriteBuffer(Pad[0], PadLen);
        end;
      end;
    end;
  finally
    Stream.Free;
  end;
end;

end.
