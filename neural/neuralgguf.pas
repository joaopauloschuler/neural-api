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
// Anything else (Q4_K & friends) raises EGGUFError with the type name.
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

  // Q8_0 block geometry: 32 elements stored as f16 scale + 32 int8.
  GGUF_Q8_0_BLOCK_ELEMS = 32;
  GGUF_Q8_0_BLOCK_BYTES = 34;

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
    else Result := 0;
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
     (GGMLType <> GGML_TYPE_Q8_0) then
    raise EGGUFError.CreateFmt(
      'gguf: tensor "%s" has unsupported ggml dtype %s (supported: F32, ' +
      'F16, Q8_0): %s', [pName, GGMLTypeName(GGMLType), FFileName]);
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

end.
