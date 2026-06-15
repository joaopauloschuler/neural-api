unit neuralsafetensors;
// Pure-Pascal reader AND writer for the HuggingFace "safetensors"
// tensor-storage format (https://github.com/huggingface/safetensors).
// No external dependencies beyond FPC's bundled fpjson.
//
// File layout:
//   bytes 0..7  : little-endian uint64 N = byte length of the JSON header
//   bytes 8..8+N-1 : UTF-8 JSON object mapping tensor name ->
//        {"dtype": "F32", "shape": [d0, d1, ...], "data_offsets": [b, e]}
//     (an optional "__metadata__" string->string object is ignored)
//   bytes 8+N.. : raw tensor data; each tensor's bytes live at
//        [8+N+b, 8+N+e) and hold prod(shape) elements in ROW-MAJOR order
//        (C order: the LAST dimension is contiguous).
//
// Supported dtypes: F32 (native), F16 and BF16 (decoded by hand to single
// via bit manipulation), I64 (converted to single). All loads return 32-bit
// singles in the same row-major element order as stored.
//
// SHARDED CHECKPOINTS: HF repos above ~2B params ship as
// model-00001-of-000NN.safetensors shards plus a model.safetensors.index.json
// of the form {"metadata": {...}, "weight_map": {"tensor.name":
// "model-00001-of-000NN.safetensors", ...}}. Pass the path of the
// index .json to Create (detected by its ".json" extension) and the reader
// opens every referenced shard (resolved relative to the index's directory)
// behind the same API: tensor names span all shards transparently and
// LoadTensorFlat reads from the owning shard's stream. The weight_map is
// validated against the shard headers - a tensor mapped to a shard that
// does not contain it, a duplicate tensor name across shards, or a missing
// shard file all raise ESafeTensorsError.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, jsonparser, neuralvolume, neuralnetwork;

type
  ESafeTensorsError = class(Exception);

  TSafeTensorInfo = record
    Name: string;
    DType: string;               // 'F32', 'F16', 'BF16', 'I64', ...
    Shape: array of Int64;       // row-major dims as stored in the file
    DataBegin, DataEnd: Int64;   // byte offsets RELATIVE to the data section
    Shard: integer;              // index of the owning shard file (0 if single)
  end;

  { TNNetSafeTensorsReader }
  // Opens a safetensors file - OR a model.safetensors.index.json describing
  // a sharded checkpoint (see the unit header) - parses and validates the
  // header(s), and loads named tensors on demand. Raises ESafeTensorsError
  // (with a descriptive message) on truncated, malformed or inconsistent
  // files - it never silently returns garbage.
  TNNetSafeTensorsReader = class
  protected
    // Protected (not private) so format siblings can reuse the stream +
    // tensor-table + LoadTensorFlat machinery: TNNetTorchBinReader
    // (neuraltorchbin.pas) subclasses this reader and populates the same
    // fields from a torch.save zip instead of a safetensors header.
    FFileName: string;           // path given to Create (file or index.json)
    FStreams: array of TFileStream;  // one open stream per shard
    FShardNames: array of string;    // shard file paths (error messages)
    FDataStarts: array of Int64; // per shard: absolute offset of data section
    FDataSizes: array of Int64;  // per shard: byte length of data section
    FTensors: array of TSafeTensorInfo;
    function FindTensor(const pName: string): integer;
    // Allocates WITHOUT parsing anything - the subclass constructor fills
    // the fields itself (Create(pFileName) would parse as safetensors).
    constructor CreateBare;
  private
    // Opens one .safetensors file, validates and parses its header and
    // appends its tensors (tagged with the new shard index) to FTensors.
    // Returns the shard index.
    function OpenShard(const pShardFileName: string): integer;
    procedure ParseHeader(const HeaderJson: string; ShardIdx: integer);
    // Parses a model.safetensors.index.json and opens every referenced shard.
    procedure OpenFromIndex(const pIndexFileName: string);
  public
    constructor Create(const pFileName: string);
    destructor Destroy; override;

    function Count: integer;
    // Number of open shard files (1 for a plain single-file checkpoint).
    function ShardCount: integer;
    function TensorName(Index: integer): string;
    function HasTensor(const pName: string): boolean;
    function GetInfo(const pName: string): TSafeTensorInfo;
    function GetDType(const pName: string): string;
    // Number of dimensions and the size of dimension Dim (0-based).
    function DimCount(const pName: string): integer;
    function DimSize(const pName: string; Dim: integer): Int64;
    function ElementCount(const pName: string): Int64;
    function ShapeAsString(const pName: string): string;
    // Loads the named tensor as a FLAT row-major array of singles into Dest:
    // Dest is resized to (ElementCount, 1, 1) and Dest.FData[i] receives the
    // i-th element in the stored row-major order (last dim contiguous).
    // Virtual so format siblings with non-raw storage (TNNetGGUFReader's
    // quantized blocks, neuralgguf.pas) can decode their own dtypes.
    procedure LoadTensorFlat(const pName: string;
      Dest: TNNetVolume); virtual;
    // Loads the named tensor's RAW on-disk bytes verbatim into Dest (no dtype
    // decoding). Used by the MXFP4 dequant-at-load path (gpt-oss), whose
    // packed-nibble "*_blocks" and E8M0 "*_scales" tensors ship as U8 and must
    // be read byte-for-byte. Dest is set to Length(Dest) = the tensor's byte
    // length. Coded by Claude (AI).
    procedure LoadTensorRawBytes(const pName: string; out Dest: TBytes);
    // Renames a tensor in place (the on-disk file is untouched; only this
    // reader's name table changes). Used by composite importers that load a
    // sub-net through a builder expecting a different name prefix - e.g. the
    // LLaVA importer feeds the language_model.* sub-tree to the stock Llama
    // builder by aliasing 'model.language_model.X' -> 'model.X'. Raises if
    // pOldName is absent or pNewName already exists. Coded by Claude (AI).
    procedure RenameTensor(const pOldName, pNewName: string);
    // Renames every tensor whose name starts with pOldPrefix, replacing that
    // leading prefix with pNewPrefix (e.g. 'model.language_model.' ->
    // 'model.'). No-op when nothing matches. Returns the count renamed.
    // Coded by Claude (AI).
    function RenameTensorPrefix(const pOldPrefix, pNewPrefix: string): integer;
    // Drops every tensor whose name starts with pPrefix from this reader's
    // name table (the on-disk file is untouched). Composite importers use it
    // to hide a sub-net's tensors from a downstream builder's strict
    // all-tensors-consumed check after that sub-net has been loaded - e.g.
    // the LLaVA importer drops the vision-tower + projector tensors before
    // handing the reader to the stock Llama builder. Returns the count
    // dropped. Coded by Claude (AI).
    function RemoveTensorsWithPrefix(const pPrefix: string): integer;
    property FileName: string read FFileName;
  end;

  // The on-disk dtype a writer encodes a tensor as. F32 is the lossless
  // default; F16/BF16 halve the file at the cost of precision (encoded on
  // write via EncodeF16/EncodeBF16, read back through DecodeF16/DecodeBF16).
  TSafeTensorsWriteDType = (stwF32, stwF16, stwBF16);

  // One tensor queued in a TNNetSafeTensorsWriter (name + shape + the chosen
  // on-disk dtype + a private copy of its raw little-endian encoded bytes).
  TSafeTensorPending = record
    Name: string;
    Shape: array of Int64;
    DType: TSafeTensorsWriteDType;
    Data: TBytes;
  end;

  { TNNetSafeTensorsWriter }
  // Collects named F32 tensors and writes a spec-compliant .safetensors
  // file: 8-byte little-endian uint64 header length, a plain-ASCII JSON
  // header mapping name -> {"dtype","shape","data_offsets"} (plus an
  // optional "__metadata__" string->string object), then the raw tensor
  // bytes packed contiguously in insertion order. The header is padded
  // with spaces to an 8-byte boundary so the data section is aligned,
  // matching the reference Python serializer. Files written here read
  // back bit-exact through TNNetSafeTensorsReader and load with the
  // Python "safetensors" library. F32 is the default; AddTensorFlat also
  // accepts stwF16/stwBF16 to encode-on-write a half-width tensor (via
  // EncodeF16/EncodeBF16) for smaller exported checkpoints - the header
  // dtype string and data_offsets reflect the chosen 2-byte encoding.
  // Raises ESafeTensorsError on invalid input (duplicate or empty names,
  // shape/element-count mismatches, writing twice).
  TNNetSafeTensorsWriter = class
  private
    FFileName: string;
    FTensors: array of TSafeTensorPending;
    FMetaKeys, FMetaValues: TStringList;  // parallel "__metadata__" pairs
    FSaved: boolean;
    function FindTensor(const pName: string): integer;
    function BuildHeaderJson: string;
  public
    constructor Create(const pFileName: string);
    destructor Destroy; override;

    // Adds/overwrites one "__metadata__" key/value pair (string->string,
    // as the spec requires). Call before SaveToFile.
    procedure SetMetadata(const pKey, pValue: string);
    // Queues a named tensor: pShape are the row-major dims declared in the
    // header (an empty array declares a 0-dim scalar) and Src supplies the
    // elements - Src.FData[0..Size-1] is copied verbatim in its flat order,
    // so prod(pShape) must equal Src.Size. pDType selects the on-disk
    // encoding (stwF32 default = lossless; stwF16/stwBF16 encode-on-write to
    // half the bytes). The data is copied, so Src may be freed or reused
    // immediately.
    procedure AddTensorFlat(const pName: string; const pShape: array of Int64;
      Src: TNNetVolume; pDType: TSafeTensorsWriteDType = stwF32);
    function Count: integer;
    // Writes the collected tensors to FileName. May be called once.
    procedure SaveToFile;
    property FileName: string read FFileName;
  end;

// IEEE 754 half precision (binary16) -> single, by bit manipulation.
function DecodeF16(Bits: Word): Single;
// bfloat16 -> single (bf16 is the top 16 bits of a single's bit pattern).
function DecodeBF16(Bits: Word): Single;
// single -> IEEE 754 half (binary16), round-to-nearest-even, saturating to
// +/-Inf on overflow; the inverse of DecodeF16 (neuralnumpy reuses this).
function EncodeF16(Value: Single): Word;
// single -> bfloat16, round-to-nearest-even on the dropped low 16 bits
// (DecodeBF16 reconstructs by zero-extension, so RNE here yields the nearest
// representable bf16). NaN is mapped to a canonical quiet NaN; Inf is kept.
function EncodeBF16(Value: Single): Word;

// Exports every parameterized layer of NN as named F32 tensors so
// Pascal-trained models round-trip into PyTorch/transformers (and back via
// LoadNNetFromSafeTensors). Naming scheme, per layer L at index I with
// Neurons.Count > 0 (layers with LinkedNeurons = true share another layer's
// neurons and are skipped):
//   layer_<I>.<ClassName>.weights : [NeuronCount, SizeY, SizeX, Depth]
//     - all neurons' weight volumes packed neuron-major, each in CAI's
//       native row-major (Y, X, Depth) memory order. If neurons disagree
//       on their weight-volume shape (rare), one tensor per neuron is
//       written instead: layer_<I>.<ClassName>.neuron_<J>.weights with
//       shape [SizeY, SizeX, Depth].
//   layer_<I>.<ClassName>.biases  : [NeuronCount]
//     - the per-neuron scalar bias weights.
// Layers whose parameters live OUTSIDE their neuron list (e.g. the symbolic
// program of TNNetByteProcessing) are not covered - use the .nn serializer
// for those. "__metadata__" records format=cai-neural-api/v1.
procedure SaveNNetToSafeTensors(NN: TNNet; const pFileName: string);
// As SaveNNetToSafeTensors but encodes every weight/bias tensor as the given
// on-disk dtype (stwF16/stwBF16 produce ~half-size checkpoints at reduced
// precision; stwF32 is identical to SaveNNetToSafeTensors). The reloaded net
// is within the dtype's precision of the original, not bit-exact.
procedure SaveNNetToSafeTensorsEx(NN: TNNet; const pFileName: string;
  pDType: TSafeTensorsWriteDType);
// Loads tensors written by SaveNNetToSafeTensors back into an
// IDENTICALLY-STRUCTURED net, matching by name. Raises ESafeTensorsError
// if an expected tensor is missing or its element count disagrees with the
// layer. Refreshes each touched layer's derived weight caches and clears
// inertia/deltas, so the net is immediately ready for inference/training.
procedure LoadNNetFromSafeTensors(NN: TNNet; const pFileName: string);

implementation

function DecodeF16(Bits: Word): Single;
var
  Sign, Exponent, Mantissa: Cardinal;
  OutBits: Cardinal;
begin
  Sign := (Bits shr 15) and $1;
  Exponent := (Bits shr 10) and $1F;
  Mantissa := Bits and $3FF;
  if Exponent = 0 then
  begin
    if Mantissa = 0 then
    begin
      // signed zero
      OutBits := Sign shl 31;
      Result := PSingle(@OutBits)^;
    end
    else
    begin
      // subnormal half: value = (-1)^s * m * 2^-24 (exact in single)
      Result := Mantissa * 5.9604644775390625e-8; // 2^-24
      if Sign <> 0 then Result := -Result;
    end;
  end
  else if Exponent = $1F then
  begin
    // Inf / NaN: rebuild with single's all-ones exponent.
    OutBits := (Sign shl 31) or ($FF shl 23) or (Mantissa shl 13);
    Result := PSingle(@OutBits)^;
  end
  else
  begin
    // Normal: rebias exponent 15 -> 127, widen mantissa 10 -> 23 bits.
    OutBits := (Sign shl 31) or ((Exponent + 112) shl 23) or (Mantissa shl 13);
    Result := PSingle(@OutBits)^;
  end;
end;

function DecodeBF16(Bits: Word): Single;
var
  OutBits: Cardinal;
begin
  OutBits := Cardinal(Bits) shl 16;
  Result := PSingle(@OutBits)^;
end;

function EncodeF16(Value: Single): Word;
var
  Bits: Cardinal;
  Sign, Exp, Mant: Cardinal;
  E: integer;
  Half: Cardinal;
begin
  Bits := PCardinal(@Value)^;
  Sign := (Bits shr 16) and $8000;
  Exp := (Bits shr 23) and $FF;
  Mant := Bits and $7FFFFF;
  if Exp = $FF then
  begin
    // Inf or NaN
    if Mant <> 0 then
      Result := Word(Sign or $7E00) // canonical quiet NaN
    else
      Result := Word(Sign or $7C00); // Inf
    exit;
  end;
  E := integer(Exp) - 127 + 15; // rebias 127 -> 15
  if E >= $1F then
  begin
    // overflow -> Inf
    Result := Word(Sign or $7C00);
    exit;
  end
  else if E <= 0 then
  begin
    // subnormal or zero
    if E < -10 then
    begin
      Result := Word(Sign); // too small -> signed zero
      exit;
    end;
    // add implicit leading 1, then shift into subnormal position with rounding
    Mant := Mant or $800000;
    Half := Mant shr (14 - E);
    // round to nearest even using the bit just shifted out
    if (Mant shr (13 - E)) and 1 = 1 then
      Inc(Half);
    Result := Word(Sign or Half);
    exit;
  end
  else
  begin
    // normal: 10-bit mantissa = top 10 bits of the 23-bit mantissa
    Half := (Cardinal(E) shl 10) or (Mant shr 13);
    // round to nearest even on the dropped 13 bits
    if (Mant and $1000) <> 0 then // guard bit set
    begin
      if ((Mant and $FFF) <> 0) or ((Half and 1) = 1) then
        Inc(Half); // may carry into exponent, which is the correct behavior
    end;
    Result := Word(Sign or Half);
    exit;
  end;
end;

function EncodeBF16(Value: Single): Word;
var
  Bits, Exp, Mant, Rounded: Cardinal;
begin
  Bits := PCardinal(@Value)^;
  Exp := (Bits shr 23) and $FF;
  Mant := Bits and $7FFFFF;
  if (Exp = $FF) and (Mant <> 0) then
  begin
    // NaN: preserve sign, force a non-zero mantissa (canonical quiet NaN) so
    // the zero-extending DecodeBF16 cannot turn it into an Inf.
    Result := Word((Bits shr 16) or $0040);
    exit;
  end;
  // Round-to-nearest-even on the low 16 bits that DecodeBF16 discards.
  // Add half an ULP (0x8000) plus the round-bias (bit 16 of the kept part)
  // before truncating, the standard RNE-by-bias trick.
  Rounded := Bits + $7FFF + ((Bits shr 16) and 1);
  Result := Word(Rounded shr 16);
end;

function DTypeByteSize(const DType: string): integer;
begin
  if DType = 'F32' then Result := 4
  else if DType = 'F16' then Result := 2
  else if DType = 'BF16' then Result := 2
  else if DType = 'I64' then Result := 8
  else Result := 0; // unsupported
end;

{ TNNetSafeTensorsReader }

constructor TNNetSafeTensorsReader.CreateBare;
begin
  inherited Create;
end;

constructor TNNetSafeTensorsReader.Create(const pFileName: string);
begin
  inherited Create;
  FFileName := pFileName;
  if not FileExists(pFileName) then
    raise ESafeTensorsError.CreateFmt('safetensors: file not found: %s',
      [pFileName]);
  // A ".json" extension means a sharded-checkpoint index
  // (model.safetensors.index.json); anything else is a plain single-file
  // safetensors checkpoint.
  if LowerCase(ExtractFileExt(pFileName)) = '.json' then
    OpenFromIndex(pFileName)
  else
    OpenShard(pFileName);
end;

destructor TNNetSafeTensorsReader.Destroy;
var
  i: integer;
begin
  for i := 0 to High(FStreams) do
    FStreams[i].Free;
  inherited Destroy;
end;

function TNNetSafeTensorsReader.OpenShard(
  const pShardFileName: string): integer;
var
  HeaderLen: QWord;
  HeaderBytes: TBytes;
  HeaderJson: string;
  Stream: TFileStream;
begin
  if not FileExists(pShardFileName) then
    raise ESafeTensorsError.CreateFmt('safetensors: file not found: %s',
      [pShardFileName]);
  Stream := TFileStream.Create(pShardFileName,
    fmOpenRead or fmShareDenyWrite);
  Result := Length(FStreams);
  SetLength(FStreams, Result + 1);
  SetLength(FShardNames, Result + 1);
  SetLength(FDataStarts, Result + 1);
  SetLength(FDataSizes, Result + 1);
  FStreams[Result] := Stream; // owned (freed by Destroy) from here on
  FShardNames[Result] := pShardFileName;
  if Stream.Size < 8 then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: file too small (%d bytes; need at least the 8-byte ' +
      'header length): %s', [Stream.Size, pShardFileName]);
  Stream.ReadBuffer(HeaderLen, 8); // little-endian uint64 (x86/ARM-LE host)
  {$IFDEF ENDIAN_BIG}
  HeaderLen := SwapEndian(HeaderLen);
  {$ENDIF}
  if (HeaderLen = 0) or (HeaderLen > QWord(Stream.Size) - 8) then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: invalid header length %d (file size %d): %s',
      [HeaderLen, Stream.Size, pShardFileName]);
  SetLength(HeaderBytes, HeaderLen);
  Stream.ReadBuffer(HeaderBytes[0], HeaderLen);
  SetString(HeaderJson, PAnsiChar(@HeaderBytes[0]), HeaderLen);
  FDataStarts[Result] := 8 + Int64(HeaderLen);
  FDataSizes[Result] := Stream.Size - FDataStarts[Result];
  ParseHeader(HeaderJson, Result);
end;

procedure TNNetSafeTensorsReader.OpenFromIndex(const pIndexFileName: string);
var
  IndexText: TStringList;
  Root: TJSONData;
  WeightMap: TJSONData;
  WeightMapObj: TJSONObject;
  BaseDir, ShardFile, MappedTensor: string;
  ShardFiles: TStringList;
  i, ShardIdx, TensorIdx: integer;
begin
  BaseDir := ExtractFilePath(pIndexFileName);
  IndexText := TStringList.Create;
  ShardFiles := TStringList.Create;
  Root := nil;
  try
    IndexText.LoadFromFile(pIndexFileName);
    try
      Root := GetJSON(IndexText.Text);
    except
      on E: Exception do
        raise ESafeTensorsError.CreateFmt(
          'safetensors: index is not valid JSON (%s): %s',
          [E.Message, pIndexFileName]);
    end;
    if not (Root is TJSONObject) then
      raise ESafeTensorsError.CreateFmt(
        'safetensors: index JSON is not an object: %s', [pIndexFileName]);
    WeightMap := TJSONObject(Root).Find('weight_map');
    if not (WeightMap is TJSONObject) then
      raise ESafeTensorsError.CreateFmt(
        'safetensors: index has no "weight_map" object: %s',
        [pIndexFileName]);
    WeightMapObj := TJSONObject(WeightMap);
    if WeightMapObj.Count = 0 then
      raise ESafeTensorsError.CreateFmt(
        'safetensors: index "weight_map" is empty: %s', [pIndexFileName]);
    // Open each distinct shard once, in first-mention order.
    ShardFiles.Sorted := False;
    for i := 0 to WeightMapObj.Count - 1 do
    begin
      if not (WeightMapObj.Items[i].JSONType = jtString) then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: index weight_map entry "%s" is not a string: %s',
          [WeightMapObj.Names[i], pIndexFileName]);
      ShardFile := WeightMapObj.Items[i].AsString;
      if ShardFiles.IndexOf(ShardFile) < 0 then
      begin
        ShardFiles.Add(ShardFile);
        OpenShard(BaseDir + ShardFile);
      end;
    end;
    // Validate the weight_map against the shard headers: every mapped
    // tensor must exist and live in the shard the index claims.
    for i := 0 to WeightMapObj.Count - 1 do
    begin
      MappedTensor := WeightMapObj.Names[i];
      ShardFile := WeightMapObj.Items[i].AsString;
      ShardIdx := ShardFiles.IndexOf(ShardFile);
      TensorIdx := FindTensor(MappedTensor);
      if TensorIdx < 0 then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: index maps tensor "%s" to shard "%s" but no shard ' +
          'contains it: %s', [MappedTensor, ShardFile, pIndexFileName]);
      if FTensors[TensorIdx].Shard <> ShardIdx then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: index maps tensor "%s" to shard "%s" but it lives ' +
          'in "%s": %s', [MappedTensor, ShardFile,
           FShardNames[FTensors[TensorIdx].Shard], pIndexFileName]);
    end;
  finally
    Root.Free;
    ShardFiles.Free;
    IndexText.Free;
  end;
end;

procedure TNNetSafeTensorsReader.ParseHeader(const HeaderJson: string;
  ShardIdx: integer);
var
  Root: TJSONData;
  Obj, TensorObj: TJSONObject;
  ShapeArr, OffsArr: TJSONArray;
  i, j, TensorCnt: integer;
  ExpectedBytes, NumElements: Int64;
  ByteSize: integer;
  ShardName: string;
begin
  ShardName := FShardNames[ShardIdx];
  try
    Root := GetJSON(HeaderJson);
  except
    on E: Exception do
      raise ESafeTensorsError.CreateFmt(
        'safetensors: header is not valid JSON (%s): %s',
        [E.Message, ShardName]);
  end;
  try
    if not (Root is TJSONObject) then
      raise ESafeTensorsError.CreateFmt(
        'safetensors: header JSON is not an object: %s', [ShardName]);
    Obj := TJSONObject(Root);
    // Appends to FTensors: with a sharded checkpoint each shard's header
    // contributes its own tensors (TensorCnt is the global write cursor).
    TensorCnt := Length(FTensors);
    SetLength(FTensors, TensorCnt + Obj.Count);
    for i := 0 to Obj.Count - 1 do
    begin
      if Obj.Names[i] = '__metadata__' then continue;
      if not (Obj.Items[i] is TJSONObject) then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: entry "%s" is not an object: %s',
          [Obj.Names[i], ShardName]);
      TensorObj := TJSONObject(Obj.Items[i]);
      if FindTensor(Obj.Names[i]) >= 0 then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: tensor "%s" in shard "%s" duplicates a tensor from ' +
          'shard "%s".', [Obj.Names[i], ShardName,
           FShardNames[FTensors[FindTensor(Obj.Names[i])].Shard]]);
      FTensors[TensorCnt].Name := Obj.Names[i];
      FTensors[TensorCnt].Shard := ShardIdx;
      if TensorObj.IndexOfName('dtype') < 0 then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: tensor "%s" has no dtype: %s',
          [Obj.Names[i], ShardName]);
      FTensors[TensorCnt].DType := TensorObj.Get('dtype', '');
      if (TensorObj.IndexOfName('shape') < 0) or
         not (TensorObj.Find('shape') is TJSONArray) then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: tensor "%s" has no shape array: %s',
          [Obj.Names[i], ShardName]);
      ShapeArr := TJSONArray(TensorObj.Find('shape'));
      SetLength(FTensors[TensorCnt].Shape, ShapeArr.Count);
      NumElements := 1;
      for j := 0 to ShapeArr.Count - 1 do
      begin
        FTensors[TensorCnt].Shape[j] := ShapeArr.Items[j].AsInt64;
        if FTensors[TensorCnt].Shape[j] < 0 then
          raise ESafeTensorsError.CreateFmt(
            'safetensors: tensor "%s" has a negative dimension: %s',
            [Obj.Names[i], ShardName]);
        NumElements := NumElements * FTensors[TensorCnt].Shape[j];
      end;
      if (TensorObj.IndexOfName('data_offsets') < 0) or
         not (TensorObj.Find('data_offsets') is TJSONArray) then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: tensor "%s" has no data_offsets array: %s',
          [Obj.Names[i], ShardName]);
      OffsArr := TJSONArray(TensorObj.Find('data_offsets'));
      if OffsArr.Count <> 2 then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: tensor "%s" data_offsets must have 2 entries: %s',
          [Obj.Names[i], ShardName]);
      FTensors[TensorCnt].DataBegin := OffsArr.Items[0].AsInt64;
      FTensors[TensorCnt].DataEnd := OffsArr.Items[1].AsInt64;
      if (FTensors[TensorCnt].DataBegin < 0) or
         (FTensors[TensorCnt].DataEnd < FTensors[TensorCnt].DataBegin) or
         (FTensors[TensorCnt].DataEnd > FDataSizes[ShardIdx]) then
        raise ESafeTensorsError.CreateFmt(
          'safetensors: tensor "%s" data_offsets [%d, %d) fall outside the ' +
          'data section (size %d): %s',
          [Obj.Names[i], FTensors[TensorCnt].DataBegin,
           FTensors[TensorCnt].DataEnd, FDataSizes[ShardIdx], ShardName]);
      ByteSize := DTypeByteSize(FTensors[TensorCnt].DType);
      if ByteSize > 0 then
      begin
        ExpectedBytes := NumElements * ByteSize;
        if FTensors[TensorCnt].DataEnd - FTensors[TensorCnt].DataBegin <>
           ExpectedBytes then
          raise ESafeTensorsError.CreateFmt(
            'safetensors: tensor "%s" (%s, %d elements) expects %d bytes ' +
            'but data_offsets span %d bytes: %s',
            [Obj.Names[i], FTensors[TensorCnt].DType, NumElements,
             ExpectedBytes,
             FTensors[TensorCnt].DataEnd - FTensors[TensorCnt].DataBegin,
             ShardName]);
      end;
      Inc(TensorCnt);
    end;
    SetLength(FTensors, TensorCnt);
  finally
    Root.Free;
  end;
end;

function TNNetSafeTensorsReader.FindTensor(const pName: string): integer;
var
  i: integer;
begin
  for i := 0 to High(FTensors) do
    if FTensors[i].Name = pName then exit(i);
  Result := -1;
end;

function TNNetSafeTensorsReader.Count: integer;
begin
  Result := Length(FTensors);
end;

procedure TNNetSafeTensorsReader.RenameTensor(const pOldName, pNewName: string);
var
  Idx: integer;
begin
  Idx := FindTensor(pOldName);
  if Idx < 0 then
    raise ESafeTensorsError.CreateFmt(
      'safetensors RenameTensor: tensor "%s" not found: %s',
      [pOldName, FFileName]);
  if (pNewName <> pOldName) and (FindTensor(pNewName) >= 0) then
    raise ESafeTensorsError.CreateFmt(
      'safetensors RenameTensor: target name "%s" already exists: %s',
      [pNewName, FFileName]);
  FTensors[Idx].Name := pNewName;
end;

function TNNetSafeTensorsReader.RenameTensorPrefix(
  const pOldPrefix, pNewPrefix: string): integer;
var
  i, OldLen: integer;
begin
  Result := 0;
  OldLen := Length(pOldPrefix);
  for i := 0 to High(FTensors) do
    if (Length(FTensors[i].Name) >= OldLen) and
       (Copy(FTensors[i].Name, 1, OldLen) = pOldPrefix) then
    begin
      FTensors[i].Name := pNewPrefix +
        Copy(FTensors[i].Name, OldLen + 1, MaxInt);
      Inc(Result);
    end;
end;

function TNNetSafeTensorsReader.RemoveTensorsWithPrefix(
  const pPrefix: string): integer;
var
  i, PfxLen, WriteIdx: integer;
begin
  Result := 0;
  PfxLen := Length(pPrefix);
  WriteIdx := 0;
  for i := 0 to High(FTensors) do
  begin
    if (Length(FTensors[i].Name) >= PfxLen) and
       (Copy(FTensors[i].Name, 1, PfxLen) = pPrefix) then
    begin
      Inc(Result);
      continue;   // drop this tensor (do not copy it forward)
    end;
    if WriteIdx <> i then FTensors[WriteIdx] := FTensors[i];
    Inc(WriteIdx);
  end;
  SetLength(FTensors, WriteIdx);
end;

function TNNetSafeTensorsReader.ShardCount: integer;
begin
  Result := Length(FStreams);
end;

function TNNetSafeTensorsReader.TensorName(Index: integer): string;
begin
  if (Index < 0) or (Index > High(FTensors)) then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: tensor index %d out of range (0..%d): %s',
      [Index, High(FTensors), FFileName]);
  Result := FTensors[Index].Name;
end;

function TNNetSafeTensorsReader.HasTensor(const pName: string): boolean;
begin
  Result := FindTensor(pName) >= 0;
end;

function TNNetSafeTensorsReader.GetInfo(const pName: string): TSafeTensorInfo;
var
  Idx: integer;
begin
  Idx := FindTensor(pName);
  if Idx < 0 then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: tensor "%s" not found in %s', [pName, FFileName]);
  Result := FTensors[Idx];
end;

function TNNetSafeTensorsReader.GetDType(const pName: string): string;
begin
  Result := GetInfo(pName).DType;
end;

function TNNetSafeTensorsReader.DimCount(const pName: string): integer;
begin
  Result := Length(GetInfo(pName).Shape);
end;

function TNNetSafeTensorsReader.DimSize(const pName: string;
  Dim: integer): Int64;
var
  Info: TSafeTensorInfo;
begin
  Info := GetInfo(pName);
  if (Dim < 0) or (Dim > High(Info.Shape)) then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: dimension %d out of range for tensor "%s" (%d dims): %s',
      [Dim, pName, Length(Info.Shape), FFileName]);
  Result := Info.Shape[Dim];
end;

function TNNetSafeTensorsReader.ElementCount(const pName: string): Int64;
var
  Info: TSafeTensorInfo;
  i: integer;
begin
  Info := GetInfo(pName);
  Result := 1;
  for i := 0 to High(Info.Shape) do
    Result := Result * Info.Shape[i];
end;

function TNNetSafeTensorsReader.ShapeAsString(const pName: string): string;
var
  Info: TSafeTensorInfo;
  i: integer;
begin
  Info := GetInfo(pName);
  Result := '[';
  for i := 0 to High(Info.Shape) do
  begin
    if i > 0 then Result := Result + ', ';
    Result := Result + IntToStr(Info.Shape[i]);
  end;
  Result := Result + ']';
end;

procedure TNNetSafeTensorsReader.LoadTensorFlat(const pName: string;
  Dest: TNNetVolume);
var
  Info: TSafeTensorInfo;
  NumElements, i: Int64;
  RawBytes: TBytes;
  WordPtr: PWord;
  SinglePtr: PSingle;
  Int64Ptr: PInt64;
begin
  Info := GetInfo(pName);
  NumElements := ElementCount(pName);
  if DTypeByteSize(Info.DType) = 0 then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: tensor "%s" has unsupported dtype "%s" (supported: ' +
      'F32, F16, BF16, I64): %s', [pName, Info.DType, FFileName]);
  if NumElements > High(integer) then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: tensor "%s" is too large (%d elements): %s',
      [pName, NumElements, FFileName]);
  Dest.ReSize(integer(NumElements), 1, 1);
  if NumElements = 0 then exit;
  SetLength(RawBytes, Info.DataEnd - Info.DataBegin);
  FStreams[Info.Shard].Position := FDataStarts[Info.Shard] + Info.DataBegin;
  FStreams[Info.Shard].ReadBuffer(RawBytes[0], Length(RawBytes));
  if Info.DType = 'F32' then
  begin
    SinglePtr := PSingle(@RawBytes[0]);
    for i := 0 to NumElements - 1 do
    begin
      Dest.FData[i] := SinglePtr^;
      Inc(SinglePtr);
    end;
  end
  else if Info.DType = 'F16' then
  begin
    WordPtr := PWord(@RawBytes[0]);
    for i := 0 to NumElements - 1 do
    begin
      Dest.FData[i] := DecodeF16(WordPtr^);
      Inc(WordPtr);
    end;
  end
  else if Info.DType = 'BF16' then
  begin
    WordPtr := PWord(@RawBytes[0]);
    for i := 0 to NumElements - 1 do
    begin
      Dest.FData[i] := DecodeBF16(WordPtr^);
      Inc(WordPtr);
    end;
  end
  else // I64
  begin
    Int64Ptr := PInt64(@RawBytes[0]);
    for i := 0 to NumElements - 1 do
    begin
      Dest.FData[i] := Int64Ptr^;
      Inc(Int64Ptr);
    end;
  end;
end;

procedure TNNetSafeTensorsReader.LoadTensorRawBytes(const pName: string;
  out Dest: TBytes);
var
  Info: TSafeTensorInfo;
  Len: Int64;
begin
  Info := GetInfo(pName);
  Len := Info.DataEnd - Info.DataBegin;
  SetLength(Dest, Len);
  if Len = 0 then exit;
  FStreams[Info.Shard].Position := FDataStarts[Info.Shard] + Info.DataBegin;
  FStreams[Info.Shard].ReadBuffer(Dest[0], Len);
end;

{ TNNetSafeTensorsWriter }

// Escapes S for embedding in a JSON string literal: backslash, double
// quote and control characters (< U+0020) are escaped; everything else
// (including UTF-8 multi-byte sequences) passes through verbatim.
function JsonEscapeString(const S: string): string;
var
  i: integer;
  C: char;
begin
  Result := '';
  for i := 1 to Length(S) do
  begin
    C := S[i];
    case C of
      '"': Result := Result + '\"';
      '\': Result := Result + '\\';
      #8: Result := Result + '\b';
      #9: Result := Result + '\t';
      #10: Result := Result + '\n';
      #12: Result := Result + '\f';
      #13: Result := Result + '\r';
      else
        if C < #32 then
          Result := Result + '\u' + IntToHex(Ord(C), 4)
        else
          Result := Result + C;
    end;
  end;
end;

constructor TNNetSafeTensorsWriter.Create(const pFileName: string);
begin
  inherited Create;
  FFileName := pFileName;
  FMetaKeys := TStringList.Create;
  FMetaValues := TStringList.Create;
  FSaved := false;
end;

destructor TNNetSafeTensorsWriter.Destroy;
begin
  FMetaValues.Free;
  FMetaKeys.Free;
  inherited Destroy;
end;

function TNNetSafeTensorsWriter.FindTensor(const pName: string): integer;
var
  i: integer;
begin
  for i := 0 to High(FTensors) do
    if FTensors[i].Name = pName then exit(i);
  Result := -1;
end;

procedure TNNetSafeTensorsWriter.SetMetadata(const pKey, pValue: string);
var
  Idx: integer;
begin
  Idx := FMetaKeys.IndexOf(pKey);
  if Idx >= 0 then
    FMetaValues[Idx] := pValue
  else
  begin
    FMetaKeys.Add(pKey);
    FMetaValues.Add(pValue);
  end;
end;

procedure TNNetSafeTensorsWriter.AddTensorFlat(const pName: string;
  const pShape: array of Int64; Src: TNNetVolume;
  pDType: TSafeTensorsWriteDType = stwF32);
var
  NumElements: Int64;
  i, Idx, ElemBytes: integer;
  SinglePtr: PSingle;
  WordPtr: PWord;
begin
  if pName = '' then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: cannot write a tensor with an empty name: %s',
      [FFileName]);
  if pName = '__metadata__' then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: "__metadata__" is reserved (use SetMetadata): %s',
      [FFileName]);
  if FindTensor(pName) >= 0 then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: duplicate tensor name "%s": %s', [pName, FFileName]);
  NumElements := 1;
  for i := 0 to High(pShape) do
  begin
    if pShape[i] < 0 then
      raise ESafeTensorsError.CreateFmt(
        'safetensors: tensor "%s" has a negative dimension: %s',
        [pName, FFileName]);
    NumElements := NumElements * pShape[i];
  end;
  if NumElements > High(integer) then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: tensor "%s" is too large (%d elements): %s',
      [pName, NumElements, FFileName]);
  if NumElements <> Src.Size then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: tensor "%s" shape declares %d elements but the source ' +
      'volume holds %d: %s', [pName, NumElements, Src.Size, FFileName]);
  Idx := Length(FTensors);
  SetLength(FTensors, Idx + 1);
  FTensors[Idx].Name := pName;
  FTensors[Idx].DType := pDType;
  SetLength(FTensors[Idx].Shape, Length(pShape));
  for i := 0 to High(pShape) do
    FTensors[Idx].Shape[i] := pShape[i];
  if pDType = stwF32 then ElemBytes := 4 else ElemBytes := 2;
  SetLength(FTensors[Idx].Data, NumElements * ElemBytes);
  if NumElements > 0 then
  begin
    case pDType of
      stwF32:
        begin
          // Element-wise copy through PSingle so the bytes are F32 even if
          // TNeuralFloat is ever widened.
          SinglePtr := PSingle(@FTensors[Idx].Data[0]);
          for i := 0 to NumElements - 1 do
          begin
            SinglePtr^ := Src.FData[i];
            Inc(SinglePtr);
          end;
        end;
      stwF16:
        begin
          WordPtr := PWord(@FTensors[Idx].Data[0]);
          for i := 0 to NumElements - 1 do
          begin
            WordPtr^ := EncodeF16(Src.FData[i]);
            Inc(WordPtr);
          end;
        end;
      stwBF16:
        begin
          WordPtr := PWord(@FTensors[Idx].Data[0]);
          for i := 0 to NumElements - 1 do
          begin
            WordPtr^ := EncodeBF16(Src.FData[i]);
            Inc(WordPtr);
          end;
        end;
    end;
  end;
end;

function TNNetSafeTensorsWriter.Count: integer;
begin
  Result := Length(FTensors);
end;

function TNNetSafeTensorsWriter.BuildHeaderJson: string;
var
  i, j: integer;
  Offset: Int64;
  Entry: string;
begin
  // Built by hand (not fpjson) so the output is deterministic plain ASCII
  // with no locale-dependent formatting; offsets/shapes are integers and
  // strings go through JsonEscapeString.
  Result := '{';
  if FMetaKeys.Count > 0 then
  begin
    Result := Result + '"__metadata__":{';
    for i := 0 to FMetaKeys.Count - 1 do
    begin
      if i > 0 then Result := Result + ',';
      Result := Result + '"' + JsonEscapeString(FMetaKeys[i]) + '":"' +
        JsonEscapeString(FMetaValues[i]) + '"';
    end;
    Result := Result + '}';
    if Length(FTensors) > 0 then Result := Result + ',';
  end;
  Offset := 0;
  for i := 0 to High(FTensors) do
  begin
    if i > 0 then Result := Result + ',';
    case FTensors[i].DType of
      stwF16: Entry := '"' + JsonEscapeString(FTensors[i].Name) +
        '":{"dtype":"F16","shape":[';
      stwBF16: Entry := '"' + JsonEscapeString(FTensors[i].Name) +
        '":{"dtype":"BF16","shape":[';
      else Entry := '"' + JsonEscapeString(FTensors[i].Name) +
        '":{"dtype":"F32","shape":[';
    end;
    for j := 0 to High(FTensors[i].Shape) do
    begin
      if j > 0 then Entry := Entry + ',';
      Entry := Entry + IntToStr(FTensors[i].Shape[j]);
    end;
    Entry := Entry + '],"data_offsets":[' + IntToStr(Offset) + ',' +
      IntToStr(Offset + Length(FTensors[i].Data)) + ']}';
    Offset := Offset + Length(FTensors[i].Data);
    Result := Result + Entry;
  end;
  Result := Result + '}';
  // Pad to an 8-byte boundary with spaces (valid JSON whitespace) so the
  // data section is aligned, like the reference Python serializer.
  while (8 + Length(Result)) mod 8 <> 0 do
    Result := Result + ' ';
end;

procedure TNNetSafeTensorsWriter.SaveToFile;
var
  HeaderJson: string;
  HeaderLen: QWord;
  Stream: TFileStream;
  i: integer;
begin
  if FSaved then
    raise ESafeTensorsError.CreateFmt(
      'safetensors: file already written (SaveToFile called twice): %s',
      [FFileName]);
  HeaderJson := BuildHeaderJson;
  Stream := TFileStream.Create(FFileName, fmCreate);
  try
    HeaderLen := Length(HeaderJson);
    {$IFDEF ENDIAN_BIG}
    HeaderLen := SwapEndian(HeaderLen);
    {$ENDIF}
    Stream.WriteBuffer(HeaderLen, 8); // little-endian uint64
    Stream.WriteBuffer(HeaderJson[1], Length(HeaderJson));
    for i := 0 to High(FTensors) do
      if Length(FTensors[i].Data) > 0 then
        Stream.WriteBuffer(FTensors[i].Data[0], Length(FTensors[i].Data));
  finally
    Stream.Free;
  end;
  FSaved := true;
end;

{ Model export/import helpers }

// True when every neuron in L has the same weight-volume shape as
// Neurons[0] - the packed [NeuronCount, SizeY, SizeX, Depth] layout applies.
function LayerHasUniformNeurons(L: TNNetLayer): boolean;
var
  j: integer;
begin
  for j := 1 to L.Neurons.Count - 1 do
    if (L.Neurons[j].Weights.SizeX <> L.Neurons[0].Weights.SizeX) or
       (L.Neurons[j].Weights.SizeY <> L.Neurons[0].Weights.SizeY) or
       (L.Neurons[j].Weights.Depth <> L.Neurons[0].Weights.Depth) then
      exit(false);
  Result := true;
end;

function LayerTensorBaseName(NN: TNNet; LayerIdx: integer): string;
begin
  Result := 'layer_' + IntToStr(LayerIdx) + '.' +
    NN.Layers[LayerIdx].ClassName;
end;

procedure SaveNNetToSafeTensors(NN: TNNet; const pFileName: string);
begin
  SaveNNetToSafeTensorsEx(NN, pFileName, stwF32);
end;

procedure SaveNNetToSafeTensorsEx(NN: TNNet; const pFileName: string;
  pDType: TSafeTensorsWriteDType);
var
  Writer: TNNetSafeTensorsWriter;
  Tmp: TNNetVolume;
  L: TNNetLayer;
  W: TNNetVolume;
  LayerIdx, j, i, Cursor: integer;
  Base: string;
begin
  Writer := TNNetSafeTensorsWriter.Create(pFileName);
  Tmp := TNNetVolume.Create;
  try
    Writer.SetMetadata('format', 'cai-neural-api/v1');
    for LayerIdx := 0 to NN.Layers.Count - 1 do
    begin
      L := NN.Layers[LayerIdx];
      if (L.Neurons.Count = 0) or L.LinkedNeurons then continue;
      Base := LayerTensorBaseName(NN, LayerIdx);
      if LayerHasUniformNeurons(L) then
      begin
        W := L.Neurons[0].Weights;
        Tmp.ReSize(L.Neurons.Count * W.Size, 1, 1);
        Cursor := 0;
        for j := 0 to L.Neurons.Count - 1 do
          for i := 0 to W.Size - 1 do
          begin
            Tmp.FData[Cursor] := L.Neurons[j].Weights.FData[i];
            Inc(Cursor);
          end;
        Writer.AddTensorFlat(Base + '.weights',
          [L.Neurons.Count, W.SizeY, W.SizeX, W.Depth], Tmp, pDType);
      end
      else
      begin
        for j := 0 to L.Neurons.Count - 1 do
        begin
          W := L.Neurons[j].Weights;
          Writer.AddTensorFlat(
            Base + '.neuron_' + IntToStr(j) + '.weights',
            [W.SizeY, W.SizeX, W.Depth], W, pDType);
        end;
      end;
      Tmp.ReSize(L.Neurons.Count, 1, 1);
      for j := 0 to L.Neurons.Count - 1 do
        Tmp.FData[j] := L.Neurons[j].BiasWeight;
      Writer.AddTensorFlat(Base + '.biases', [L.Neurons.Count], Tmp, pDType);
    end;
    Writer.SaveToFile;
  finally
    Tmp.Free;
    Writer.Free;
  end;
end;

procedure LoadNNetFromSafeTensors(NN: TNNet; const pFileName: string);
var
  Reader: TNNetSafeTensorsReader;
  Tmp: TNNetVolume;
  L: TNNetLayer;
  LayerIdx, j, i, Cursor: integer;
  Base, TensorName: string;

  procedure LoadExpected(const pName: string; ExpectedElements: integer);
  begin
    if not Reader.HasTensor(pName) then
      raise ESafeTensorsError.CreateFmt(
        'safetensors: net layout expects tensor "%s" but the file does ' +
        'not contain it (was it saved from an identically-structured ' +
        'net?): %s', [pName, pFileName]);
    Reader.LoadTensorFlat(pName, Tmp);
    if Tmp.Size <> ExpectedElements then
      raise ESafeTensorsError.CreateFmt(
        'safetensors: tensor "%s" holds %d elements but the layer ' +
        'expects %d: %s', [pName, Tmp.Size, ExpectedElements, pFileName]);
  end;

begin
  Reader := TNNetSafeTensorsReader.Create(pFileName);
  Tmp := TNNetVolume.Create;
  try
    for LayerIdx := 0 to NN.Layers.Count - 1 do
    begin
      L := NN.Layers[LayerIdx];
      if (L.Neurons.Count = 0) or L.LinkedNeurons then continue;
      Base := LayerTensorBaseName(NN, LayerIdx);
      if LayerHasUniformNeurons(L) then
      begin
        LoadExpected(Base + '.weights',
          L.Neurons.Count * L.Neurons[0].Weights.Size);
        Cursor := 0;
        for j := 0 to L.Neurons.Count - 1 do
          for i := 0 to L.Neurons[j].Weights.Size - 1 do
          begin
            L.Neurons[j].Weights.FData[i] := Tmp.FData[Cursor];
            Inc(Cursor);
          end;
      end
      else
      begin
        for j := 0 to L.Neurons.Count - 1 do
        begin
          TensorName := Base + '.neuron_' + IntToStr(j) + '.weights';
          LoadExpected(TensorName, L.Neurons[j].Weights.Size);
          for i := 0 to L.Neurons[j].Weights.Size - 1 do
            L.Neurons[j].Weights.FData[i] := Tmp.FData[i];
        end;
      end;
      LoadExpected(Base + '.biases', L.Neurons.Count);
      for j := 0 to L.Neurons.Count - 1 do
        L.Neurons[j].BiasWeight := Tmp.FData[j];
      L.ClearInertia();
      L.ClearDeltas();
      L.FlushWeightCache();
    end;
  finally
    Tmp.Free;
    Reader.Free;
  end;
end;

end.
