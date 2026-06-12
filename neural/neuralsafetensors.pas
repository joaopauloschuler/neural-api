unit neuralsafetensors;
// Pure-Pascal reader for the HuggingFace "safetensors" tensor-storage format
// (https://github.com/huggingface/safetensors). No external dependencies
// beyond FPC's bundled fpjson.
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
  Classes, SysUtils, fpjson, jsonparser, neuralvolume;

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
  private
    FFileName: string;           // path given to Create (file or index.json)
    FStreams: array of TFileStream;  // one open stream per shard
    FShardNames: array of string;    // shard file paths (error messages)
    FDataStarts: array of Int64; // per shard: absolute offset of data section
    FDataSizes: array of Int64;  // per shard: byte length of data section
    FTensors: array of TSafeTensorInfo;
    function FindTensor(const pName: string): integer;
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
    procedure LoadTensorFlat(const pName: string; Dest: TNNetVolume);
    property FileName: string read FFileName;
  end;

// IEEE 754 half precision (binary16) -> single, by bit manipulation.
function DecodeF16(Bits: Word): Single;
// bfloat16 -> single (bf16 is the top 16 bits of a single's bit pattern).
function DecodeBF16(Bits: Word): Single;

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

function DTypeByteSize(const DType: string): integer;
begin
  if DType = 'F32' then Result := 4
  else if DType = 'F16' then Result := 2
  else if DType = 'BF16' then Result := 2
  else if DType = 'I64' then Result := 8
  else Result := 0; // unsupported
end;

{ TNNetSafeTensorsReader }

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

end.
