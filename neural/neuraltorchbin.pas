unit neuraltorchbin;
// Pure-Pascal reader for PyTorch "pytorch_model.bin" checkpoints - the
// torch.save zip container (the long tail of older / fine-tuned HF repos
// that never got converted to safetensors, e.g. cerebras/Cerebras-GPT-*
// and roneneldan/TinyStories-*).
//
// File layout (torch.save, _use_new_zipfile_serialization=True, the default
// since torch 1.6): a ZIP archive containing
//   <archive>/data.pkl     a pickle (protocol 2) of the state_dict
//   <archive>/data/<N>     one raw little-endian storage blob per tensor
//   <archive>/version, byteorder, ...   small metadata entries (ignored,
//                          except byteorder which must be "little")
// All entries are STORED (uncompressed), so tensor data can be read straight
// from the container at an absolute file offset.
//
// SECURITY: pickle is a Turing-complete VM and unpickling untrusted data is
// arbitrary code execution - the hazard that motivated safetensors. A
// state_dict however only ever uses a dozen data-shaped opcodes plus
// persistent-id storage references. TNNetTorchBinReader implements EXACTLY
// that subset as a RESTRICTED unpickler:
//   * only the opcodes listed in the dispatcher below are accepted;
//   * GLOBAL/STACK_GLOBAL must name a whitelisted symbol (collections
//     OrderedDict, torch._utils _rebuild_tensor_v2 / _rebuild_parameter,
//     and the torch.*Storage dtype classes) - anything else (e.g. the
//     classic `os system` payload) raises ETorchBinError;
//   * REDUCE only ever calls those whitelisted callables, and BUILD /
//     INST / OBJ / NEWOBJ are not implemented at all.
// Nothing is ever executed; the unpickler only builds a passive object tree.
//
// The reader subclasses TNNetSafeTensorsReader and populates the same
// tensor table (same dtype strings, same flat row-major semantics), so the
// whole public API - HasTensor/DimSize/LoadTensorFlat/... including the
// F16/BF16 decoding - is inherited unchanged and the neuralpretrained.pas
// importer builders accept either format transparently.
//
// SHARDED CHECKPOINTS: large .bin repos ship as
// pytorch_model-00001-of-000NN.bin shards plus a
// pytorch_model.bin.index.json with the SAME {"metadata": {...},
// "weight_map": {"tensor.name": "shard file", ...}} layout as the
// safetensors index. Pass the index path to Create (detected by its
// ".json" extension, mirroring TNNetSafeTensorsReader) and the reader
// opens every referenced shard (resolved relative to the index's
// directory) behind the same API. The weight_map is validated: a tensor
// mapped to a shard that does not contain it, a duplicate tensor name
// across shards, or a missing shard file all raise ETorchBinError.
//
// Out of scope (raises a descriptive error): the pre-1.6 non-zip legacy
// format, DEFLATE-compressed entries, and non-contiguous (stride-permuted)
// tensors - state_dict tensors are normally contiguous.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpjson, jsonparser, neuralvolume, neuralsafetensors;

type
  ETorchBinError = class(Exception);

  { TNNetTorchBinReader }
  // Opens a torch.save zip checkpoint, parses the central directory, runs
  // the restricted unpickler over <archive>/data.pkl and exposes every
  // tensor of the state_dict through the inherited TNNetSafeTensorsReader
  // API. Raises ETorchBinError on anything outside the state_dict pickle
  // subset - it never executes pickled code and never silently returns
  // garbage.
  TNNetTorchBinReader = class(TNNetSafeTensorsReader)
  private
    // FZip*/FArchivePrefix describe the shard CURRENTLY being parsed
    // (FCurShard); each OpenBinShard call overwrites them after baking
    // absolute byte offsets into the inherited tensor table.
    FZipNames: array of string;   // entry names from the central directory
    FZipMethod: array of word;    // compression method (0 = stored)
    FZipCompSize: array of QWord; // compressed byte size
    FZipSize: array of QWord;     // uncompressed byte size
    FZipLocalOfs: array of QWord; // local-header offset
    FArchivePrefix: string;       // "<archive>/" before data.pkl and data/
    FCurShard: integer;           // shard index currently being parsed
    // Current shard's file path - for error messages.
    function ShardPath: string;
    procedure ParseZipDirectory(Stream: TFileStream);
    function FindZipEntry(const pEntryName: string): integer;
    // Absolute file offset of a STORED entry's first data byte (resolved
    // through its local header: filename/extra lengths differ from the
    // central directory copies - torch pads extras for 64-byte alignment).
    function ZipEntryDataOffset(Stream: TFileStream;
      EntryIdx: integer): Int64;
    function ReadZipEntry(Stream: TFileStream; EntryIdx: integer): TBytes;
    procedure Unpickle(const Pickle: TBytes);
    // Opens one torch.save zip, parses it and appends its tensors (tagged
    // with the new shard index) to the inherited tensor table. Returns the
    // shard index. Mirrors TNNetSafeTensorsReader.OpenShard.
    function OpenBinShard(const pShardFileName: string): integer;
    // Parses a pytorch_model.bin.index.json and opens every referenced
    // shard. Mirrors TNNetSafeTensorsReader.OpenFromIndex.
    procedure OpenFromBinIndex(const pIndexFileName: string);
  public
    constructor Create(const pFileName: string);
  end;

implementation

// ---------------------------------------------------------------------------
// Restricted-unpickler object model: a passive tree of tagged nodes. Every
// node is owned by one TFPList and freed in a single sweep - pickle graphs
// share nodes via the memo, so per-node ownership would double-free.
// ---------------------------------------------------------------------------
type
  TPickleKind = (pkNone, pkBool, pkInt, pkFloat, pkStr, pkTuple, pkList,
    pkDict, pkGlobal, pkPersId, pkTensor, pkMark);

  TPickleNode = class
  public
    Kind: TPickleKind;
    IntVal: Int64;
    FloatVal: double;
    StrVal: string;                  // pkStr value; pkGlobal module
    NameVal: string;                 // pkGlobal symbol name
    Items: array of TPickleNode;     // pkTuple/pkList items; pkPersId payload
    Keys, Vals: array of TPickleNode; // pkDict entries (insertion order)
    // pkTensor payload (from _rebuild_tensor_v2 + its persistent id)
    StorageKey: string;
    StorageDType: string;            // safetensors dtype string ('F32', ...)
    StorageNumel: Int64;             // total elements in the backing storage
    StorageOffset: Int64;            // element offset of this view
    Shape: array of Int64;
    Stride: array of Int64;
  end;

function DTypeOfStorageClass(const pClassName: string;
  out ByteSize: integer): string;
begin
  ByteSize := 0;
  if pClassName = 'FloatStorage' then begin ByteSize := 4; exit('F32'); end;
  if pClassName = 'HalfStorage' then begin ByteSize := 2; exit('F16'); end;
  if pClassName = 'BFloat16Storage' then
    begin ByteSize := 2; exit('BF16'); end;
  if pClassName = 'LongStorage' then begin ByteSize := 8; exit('I64'); end;
  if pClassName = 'DoubleStorage' then begin ByteSize := 8; exit('F64'); end;
  if pClassName = 'IntStorage' then begin ByteSize := 4; exit('I32'); end;
  if pClassName = 'ShortStorage' then begin ByteSize := 2; exit('I16'); end;
  if pClassName = 'CharStorage' then begin ByteSize := 1; exit('I8'); end;
  if pClassName = 'ByteStorage' then begin ByteSize := 1; exit('U8'); end;
  if pClassName = 'BoolStorage' then begin ByteSize := 1; exit('BOOL'); end;
  Result := '';
end;

function GlobalIsWhitelisted(const pModule, pName: string): boolean;
var
  Ignored: integer;
begin
  if (pModule = 'collections') and (pName = 'OrderedDict') then exit(true);
  if (pModule = 'torch._utils') and
     ((pName = '_rebuild_tensor_v2') or (pName = '_rebuild_parameter')) then
    exit(true);
  // The storage dtype classes ('torch FloatStorage', ...) only ever appear
  // inside persistent-id tuples, but they are loaded via GLOBAL like any
  // other symbol.
  if (pModule = 'torch') and
     (DTypeOfStorageClass(pName, Ignored) <> '') then exit(true);
  Result := false;
end;

{ TNNetTorchBinReader }

constructor TNNetTorchBinReader.Create(const pFileName: string);
begin
  inherited CreateBare; // Create(pFileName) would parse as safetensors
  FFileName := pFileName;
  if not FileExists(pFileName) then
    raise ETorchBinError.CreateFmt('torch.bin: file not found: %s',
      [pFileName]);
  // A ".json" extension means a sharded-checkpoint index
  // (pytorch_model.bin.index.json - the same weight_map layout as the
  // safetensors index); anything else is a single torch.save zip.
  if LowerCase(ExtractFileExt(pFileName)) = '.json' then
    OpenFromBinIndex(pFileName)
  else
    OpenBinShard(pFileName);
end;

function TNNetTorchBinReader.ShardPath: string;
begin
  Result := FShardNames[FCurShard];
end;

function TNNetTorchBinReader.OpenBinShard(
  const pShardFileName: string): integer;
var
  Stream: TFileStream;
  PklIdx, ByteOrderIdx: integer;
  PickleBytes, ByteOrderBytes: TBytes;
  ByteOrderStr: string;
begin
  if not FileExists(pShardFileName) then
    raise ETorchBinError.CreateFmt('torch.bin: file not found: %s',
      [pShardFileName]);
  Stream := TFileStream.Create(pShardFileName,
    fmOpenRead or fmShareDenyWrite);
  // Register the stream in the inherited shard table immediately so the
  // inherited destructor owns it even if parsing fails below. Tensor
  // offsets are stored ABSOLUTE, so the shard data section is the whole
  // file (FDataStarts[shard] = 0).
  Result := Length(FStreams);
  SetLength(FStreams, Result + 1);
  SetLength(FShardNames, Result + 1);
  SetLength(FDataStarts, Result + 1);
  SetLength(FDataSizes, Result + 1);
  FStreams[Result] := Stream;
  FShardNames[Result] := pShardFileName;
  FDataStarts[Result] := 0;
  FDataSizes[Result] := Stream.Size;
  FCurShard := Result;
  ParseZipDirectory(Stream);
  ByteOrderIdx := FindZipEntry(FArchivePrefix + 'byteorder');
  if ByteOrderIdx >= 0 then
  begin
    ByteOrderBytes := ReadZipEntry(Stream, ByteOrderIdx);
    SetString(ByteOrderStr, PAnsiChar(@ByteOrderBytes[0]),
      Length(ByteOrderBytes));
    if Trim(ByteOrderStr) <> 'little' then
      raise ETorchBinError.CreateFmt(
        'torch.bin: byteorder "%s" not supported (little-endian only): %s',
        [Trim(ByteOrderStr), pShardFileName]);
  end;
  PklIdx := FindZipEntry(FArchivePrefix + 'data.pkl');
  PickleBytes := ReadZipEntry(Stream, PklIdx);
  Unpickle(PickleBytes);
end;

procedure TNNetTorchBinReader.OpenFromBinIndex(const pIndexFileName: string);
var
  IndexText: TStringList;
  Root: TJSONData;
  WeightMap: TJSONData;
  WeightMapObj: TJSONObject;
  BaseDir, ShardFile, MappedTensor: string;
  ShardFiles: TStringList;
  i, ShardIdx, TensorIdx, WeightMapCount: integer;
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
        raise ETorchBinError.CreateFmt(
          'torch.bin: index is not valid JSON (%s): %s',
          [E.Message, pIndexFileName]);
    end;
    if not (Root is TJSONObject) then
      raise ETorchBinError.CreateFmt(
        'torch.bin: index JSON is not an object: %s', [pIndexFileName]);
    WeightMap := TJSONObject(Root).Find('weight_map');
    if not (WeightMap is TJSONObject) then
      raise ETorchBinError.CreateFmt(
        'torch.bin: index has no "weight_map" object: %s',
        [pIndexFileName]);
    WeightMapObj := TJSONObject(WeightMap);
    WeightMapCount := WeightMapObj.Count;
    if WeightMapCount = 0 then
      raise ETorchBinError.CreateFmt(
        'torch.bin: index "weight_map" is empty: %s', [pIndexFileName]);
    // Open each distinct shard once, in first-mention order.
    ShardFiles.Sorted := False;
    for i := 0 to WeightMapCount - 1 do
    begin
      if not (WeightMapObj.Items[i].JSONType = jtString) then
        raise ETorchBinError.CreateFmt(
          'torch.bin: index weight_map entry "%s" is not a string: %s',
          [WeightMapObj.Names[i], pIndexFileName]);
      ShardFile := WeightMapObj.Items[i].AsString;
      if ShardFiles.IndexOf(ShardFile) < 0 then
      begin
        ShardFiles.Add(ShardFile);
        OpenBinShard(BaseDir + ShardFile);
      end;
    end;
    // Validate the weight_map against the shard state_dicts: every mapped
    // tensor must exist and live in the shard the index claims.
    for i := 0 to WeightMapCount - 1 do
    begin
      MappedTensor := WeightMapObj.Names[i];
      ShardFile := WeightMapObj.Items[i].AsString;
      ShardIdx := ShardFiles.IndexOf(ShardFile);
      TensorIdx := FindTensor(MappedTensor);
      if TensorIdx < 0 then
        raise ETorchBinError.CreateFmt(
          'torch.bin: index maps tensor "%s" to shard "%s" but no shard ' +
          'contains it: %s', [MappedTensor, ShardFile, pIndexFileName]);
      if FTensors[TensorIdx].Shard <> ShardIdx then
        raise ETorchBinError.CreateFmt(
          'torch.bin: index maps tensor "%s" to shard "%s" but it lives ' +
          'in "%s": %s', [MappedTensor, ShardFile,
           FShardNames[FTensors[TensorIdx].Shard], pIndexFileName]);
    end;
  finally
    Root.Free;
    ShardFiles.Free;
    IndexText.Free;
  end;
end;

procedure TNNetTorchBinReader.ParseZipDirectory(Stream: TFileStream);
const
  EOCD_SIG = $06054B50;       // end of central directory
  Z64_LOCATOR_SIG = $07064B50; // zip64 EOCD locator
  Z64_EOCD_SIG = $06064B50;   // zip64 end of central directory
  CDIR_SIG = $02014B50;       // central directory file header
var
  Tail: TBytes;
  TailLen, ScanPos, EocdPos: Int64;
  EntryCount: QWord;
  CDirOfs: QWord;
  Z64EocdOfs: QWord;
  EntryCnt: QWord;
  Sig: Cardinal;
  Method: word;
  CompSize32, Size32, LocalOfs32: Cardinal;
  CompSize, Size, LocalOfs: QWord;
  NameLen, ExtraLen, CommentLen: word;
  EntryName: string;
  Extra: TBytes;
  ExtraPos, FieldId, FieldLen: integer;
  ZipIdx: integer;
  SlashPos: integer;

  function ReadWordAt(const Buf: TBytes; Ofs: Int64): word;
  begin
    Result := Buf[Ofs] or (word(Buf[Ofs + 1]) shl 8);
  end;

  function ReadDWordAt(const Buf: TBytes; Ofs: Int64): Cardinal;
  begin
    Result := Buf[Ofs] or (Cardinal(Buf[Ofs + 1]) shl 8) or
      (Cardinal(Buf[Ofs + 2]) shl 16) or (Cardinal(Buf[Ofs + 3]) shl 24);
  end;

  function ReadQWordAt(const Buf: TBytes; Ofs: Int64): QWord;
  begin
    Result := QWord(ReadDWordAt(Buf, Ofs)) or
      (QWord(ReadDWordAt(Buf, Ofs + 4)) shl 32);
  end;

begin
  if Stream.Size < 22 then
    raise ETorchBinError.CreateFmt(
      'torch.bin: file too small to be a zip archive (%d bytes; the ' +
      'pre-1.6 non-zip legacy torch format is not supported): %s',
      [Stream.Size, ShardPath]);
  // The EOCD record sits in the last 22..22+65535 bytes (variable comment).
  TailLen := 22 + 65535 + 20; // + room for the zip64 locator just before it
  if TailLen > Stream.Size then TailLen := Stream.Size;
  SetLength(Tail, TailLen);
  Stream.Position := Stream.Size - TailLen;
  Stream.ReadBuffer(Tail[0], TailLen);
  EocdPos := -1;
  ScanPos := TailLen - 22;
  while ScanPos >= 0 do
  begin
    if ReadDWordAt(Tail, ScanPos) = EOCD_SIG then
    begin
      EocdPos := ScanPos;
      break;
    end;
    Dec(ScanPos);
  end;
  if EocdPos < 0 then
    raise ETorchBinError.CreateFmt(
      'torch.bin: zip end-of-central-directory record not found (not a ' +
      'torch.save zip checkpoint? the pre-1.6 non-zip legacy format is ' +
      'not supported): %s', [ShardPath]);
  EntryCount := ReadWordAt(Tail, EocdPos + 10);
  CDirOfs := ReadDWordAt(Tail, EocdPos + 16);
  if (EntryCount = $FFFF) or (CDirOfs = $FFFFFFFF) then
  begin
    // zip64: a locator record sits 20 bytes before the EOCD and points at
    // the zip64 EOCD, which carries the real 64-bit values.
    if (EocdPos < 20) or
       (ReadDWordAt(Tail, EocdPos - 20) <> Z64_LOCATOR_SIG) then
      raise ETorchBinError.CreateFmt(
        'torch.bin: zip64 sizes announced but no zip64 locator found: %s',
        [ShardPath]);
    Z64EocdOfs := ReadQWordAt(Tail, EocdPos - 20 + 8);
    SetLength(Tail, 56);
    Stream.Position := Z64EocdOfs;
    Stream.ReadBuffer(Tail[0], 56);
    if ReadDWordAt(Tail, 0) <> Z64_EOCD_SIG then
      raise ETorchBinError.CreateFmt(
        'torch.bin: zip64 end-of-central-directory signature missing: %s',
        [ShardPath]);
    EntryCount := ReadQWordAt(Tail, 32);
    CDirOfs := ReadQWordAt(Tail, 48);
  end;
  if EntryCount = 0 then
    raise ETorchBinError.CreateFmt('torch.bin: zip archive is empty: %s',
      [ShardPath]);
  // Walk the central directory.
  Stream.Position := CDirOfs;
  SetLength(FZipNames, EntryCount);
  SetLength(FZipMethod, EntryCount);
  SetLength(FZipCompSize, EntryCount);
  SetLength(FZipSize, EntryCount);
  SetLength(FZipLocalOfs, EntryCount);
  for EntryCnt := 0 to EntryCount - 1 do
  begin
    SetLength(Tail, 46);
    Stream.ReadBuffer(Tail[0], 46);
    Sig := ReadDWordAt(Tail, 0);
    if Sig <> CDIR_SIG then
      raise ETorchBinError.CreateFmt(
        'torch.bin: bad central-directory entry signature $%x at entry ' +
        '%d: %s', [Sig, EntryCnt, ShardPath]);
    Method := ReadWordAt(Tail, 10);
    CompSize32 := ReadDWordAt(Tail, 20);
    Size32 := ReadDWordAt(Tail, 24);
    NameLen := ReadWordAt(Tail, 28);
    ExtraLen := ReadWordAt(Tail, 30);
    CommentLen := ReadWordAt(Tail, 32);
    LocalOfs32 := ReadDWordAt(Tail, 42);
    SetLength(EntryName, NameLen);
    if NameLen > 0 then Stream.ReadBuffer(EntryName[1], NameLen);
    CompSize := CompSize32;
    Size := Size32;
    LocalOfs := LocalOfs32;
    SetLength(Extra, ExtraLen);
    if ExtraLen > 0 then Stream.ReadBuffer(Extra[0], ExtraLen);
    // zip64 extended-information extra field (id $0001): 64-bit values for
    // exactly those header fields that saturated at $FFFFFFFF, in the
    // fixed order uncompressed size, compressed size, local-header offset.
    ExtraPos := 0;
    while ExtraPos + 4 <= ExtraLen do
    begin
      FieldId := ReadWordAt(Extra, ExtraPos);
      FieldLen := ReadWordAt(Extra, ExtraPos + 2);
      if FieldId = $0001 then
      begin
        ExtraPos := ExtraPos + 4;
        if Size32 = $FFFFFFFF then
        begin
          Size := ReadQWordAt(Extra, ExtraPos);
          ExtraPos := ExtraPos + 8;
        end;
        if CompSize32 = $FFFFFFFF then
        begin
          CompSize := ReadQWordAt(Extra, ExtraPos);
          ExtraPos := ExtraPos + 8;
        end;
        if LocalOfs32 = $FFFFFFFF then
          LocalOfs := ReadQWordAt(Extra, ExtraPos);
        break;
      end;
      ExtraPos := ExtraPos + 4 + FieldLen;
    end;
    if CommentLen > 0 then
      Stream.Position := Stream.Position + CommentLen;
    FZipNames[EntryCnt] := EntryName;
    FZipMethod[EntryCnt] := Method;
    FZipCompSize[EntryCnt] := CompSize;
    FZipSize[EntryCnt] := Size;
    FZipLocalOfs[EntryCnt] := LocalOfs;
  end;
  // The archive prefix is whatever precedes "data.pkl" ("<archive>/" for
  // torch.save; "" is tolerated for hand-rolled containers).
  FArchivePrefix := '';
  ZipIdx := -1;
  for ScanPos := 0 to High(FZipNames) do
    if (FZipNames[ScanPos] = 'data.pkl') or
       (Copy(FZipNames[ScanPos],
         Length(FZipNames[ScanPos]) - 8, 9) = '/data.pkl') then
    begin
      ZipIdx := ScanPos;
      break;
    end;
  if ZipIdx < 0 then
    raise ETorchBinError.CreateFmt(
      'torch.bin: no data.pkl entry in the zip (not a torch.save ' +
      'checkpoint?): %s', [ShardPath]);
  SlashPos := Length(FZipNames[ZipIdx]) - Length('data.pkl');
  FArchivePrefix := Copy(FZipNames[ZipIdx], 1, SlashPos);
end;

function TNNetTorchBinReader.FindZipEntry(const pEntryName: string): integer;
var
  i: integer;
begin
  for i := 0 to High(FZipNames) do
    if FZipNames[i] = pEntryName then exit(i);
  Result := -1;
end;

function TNNetTorchBinReader.ZipEntryDataOffset(Stream: TFileStream;
  EntryIdx: integer): Int64;
const
  LOCAL_SIG = $04034B50;
var
  Hdr: array[0..29] of byte;
  Sig: Cardinal;
  NameLen, ExtraLen: word;
begin
  Stream.Position := FZipLocalOfs[EntryIdx];
  Stream.ReadBuffer(Hdr[0], 30);
  Sig := Hdr[0] or (Cardinal(Hdr[1]) shl 8) or (Cardinal(Hdr[2]) shl 16) or
    (Cardinal(Hdr[3]) shl 24);
  if Sig <> LOCAL_SIG then
    raise ETorchBinError.CreateFmt(
      'torch.bin: bad local-header signature for zip entry "%s": %s',
      [FZipNames[EntryIdx], ShardPath]);
  // The LOCAL name/extra lengths can differ from the central directory's
  // (torch pads the local extra field to 64-byte-align tensor data).
  NameLen := Hdr[26] or (word(Hdr[27]) shl 8);
  ExtraLen := Hdr[28] or (word(Hdr[29]) shl 8);
  Result := FZipLocalOfs[EntryIdx] + 30 + NameLen + ExtraLen;
end;

function TNNetTorchBinReader.ReadZipEntry(Stream: TFileStream;
  EntryIdx: integer): TBytes;
begin
  Result := nil;
  if FZipMethod[EntryIdx] <> 0 then
    raise ETorchBinError.CreateFmt(
      'torch.bin: zip entry "%s" uses compression method %d; only STORED ' +
      '(0) entries are supported (torch.save never compresses): %s',
      [FZipNames[EntryIdx], FZipMethod[EntryIdx], ShardPath]);
  SetLength(Result, FZipSize[EntryIdx]);
  if FZipSize[EntryIdx] = 0 then exit;
  Stream.Position := ZipEntryDataOffset(Stream, EntryIdx);
  Stream.ReadBuffer(Result[0], Length(Result));
end;

procedure TNNetTorchBinReader.Unpickle(const Pickle: TBytes);
var
  Owned: TFPList;       // every allocated node (single-sweep ownership)
  Stack: array of TPickleNode;
  StackTop: integer;
  Memo: array of TPickleNode;
  MemoNext: integer;    // MEMOIZE writes here (= #memo entries so far)
  Pos: Int64;           // read cursor into Pickle
  PickleLen: Int64;
  Root: TPickleNode;
  i, j, TensorCnt: integer;
  OwnedCount: integer;
  EntryIdx: integer;
  ElemSize: integer;
  NumElements, ExpectedStride, ByteBegin, ByteEnd: Int64;
  Node, ValNode: TPickleNode;
  EntryName: string;

  procedure Fail(const Msg: string);
  begin
    raise ETorchBinError.CreateFmt('torch.bin: %s (pickle offset %d): %s',
      [Msg, Pos, ShardPath]);
  end;

  function NewNode(pKind: TPickleKind): TPickleNode;
  begin
    Result := TPickleNode.Create;
    Result.Kind := pKind;
    Owned.Add(Result);
  end;

  procedure Push(pNode: TPickleNode);
  begin
    if StackTop + 1 >= Length(Stack) then
      SetLength(Stack, (StackTop + 2) * 2);
    Inc(StackTop);
    Stack[StackTop] := pNode;
  end;

  function Pop: TPickleNode;
  begin
    if StackTop < 0 then Fail('pickle stack underflow');
    Result := Stack[StackTop];
    Dec(StackTop);
  end;

  function NeedBytes(Count: Int64): Int64; // returns start, advances Pos
  begin
    if Pos + Count > PickleLen then Fail('truncated pickle stream');
    Result := Pos;
    Pos := Pos + Count;
  end;

  function ReadByte: byte;
  begin
    Result := Pickle[NeedBytes(1)];
  end;

  function ReadUInt2: word;
  var
    P: Int64;
  begin
    P := NeedBytes(2);
    Result := Pickle[P] or (word(Pickle[P + 1]) shl 8);
  end;

  function ReadUInt4: Cardinal;
  var
    P: Int64;
  begin
    P := NeedBytes(4);
    Result := Pickle[P] or (Cardinal(Pickle[P + 1]) shl 8) or
      (Cardinal(Pickle[P + 2]) shl 16) or (Cardinal(Pickle[P + 3]) shl 24);
  end;

  function ReadString(Len: Int64): string;
  var
    P: Int64;
  begin
    P := NeedBytes(Len);
    SetLength(Result, Len);
    if Len > 0 then Move(Pickle[P], Result[1], Len);
  end;

  function ReadLine: string; // newline-terminated GLOBAL argument
  var
    StartPos: Int64;
  begin
    StartPos := Pos;
    while (Pos < PickleLen) and (Pickle[Pos] <> 10) do Inc(Pos);
    if Pos >= PickleLen then Fail('unterminated GLOBAL line');
    SetLength(Result, Pos - StartPos);
    if Pos > StartPos then
      Move(Pickle[StartPos], Result[1], Pos - StartPos);
    Inc(Pos); // consume the newline
  end;

  procedure MemoPut(Idx: Cardinal);
  begin
    if StackTop < 0 then Fail('BINPUT/MEMOIZE on an empty stack');
    if Idx >= Cardinal(Length(Memo)) then SetLength(Memo, (Idx + 1) * 2);
    Memo[Idx] := Stack[StackTop];
    if Int64(Idx) + 1 > MemoNext then MemoNext := Idx + 1;
  end;

  function MemoGet(Idx: Cardinal): TPickleNode;
  begin
    if (Idx >= Cardinal(Length(Memo))) or (Memo[Idx] = nil) then
      Fail('BINGET of an unset memo slot');
    Result := Memo[Idx];
  end;

  // Pops stack entries down to the topmost MARK into a fresh node of
  // pKind (preserving order); pops the MARK itself too.
  function PopToMark(pKind: TPickleKind): TPickleNode;
  var
    MarkPos, n: integer;
  begin
    MarkPos := StackTop;
    while (MarkPos >= 0) and (Stack[MarkPos].Kind <> pkMark) do
      Dec(MarkPos);
    if MarkPos < 0 then Fail('no MARK on the pickle stack');
    Result := NewNode(pKind);
    SetLength(Result.Items, StackTop - MarkPos);
    for n := MarkPos + 1 to StackTop do
      Result.Items[n - MarkPos - 1] := Stack[n];
    StackTop := MarkPos - 1;
  end;

  procedure DictSetItem(Dict, Key, Value: TPickleNode);
  var
    n: integer;
  begin
    if Dict.Kind <> pkDict then Fail('SETITEM(S) on a non-dict');
    n := Length(Dict.Keys);
    SetLength(Dict.Keys, n + 1);
    SetLength(Dict.Vals, n + 1);
    Dict.Keys[n] := Key;
    Dict.Vals[n] := Value;
  end;

  function TupleN(Count: integer): TPickleNode;
  var
    n: integer;
  begin
    Result := NewNode(pkTuple);
    SetLength(Result.Items, Count);
    for n := Count - 1 downto 0 do
      Result.Items[n] := Pop;
  end;

  function NodeAsInt(pNode: TPickleNode; const What: string): Int64;
  begin
    if pNode = nil then Fail(What + ' is missing');
    if not (pNode.Kind in [pkInt, pkBool]) then
      Fail(What + ' is not an integer');
    Result := pNode.IntVal;
  end;

  // REDUCE dispatcher - the only "calls" the restricted unpickler performs,
  // each one a hard-coded constructor for a whitelisted symbol.
  function ApplyReduce(Callable, Args: TPickleNode): TPickleNode;
  var
    PersId, SizeT, StrideT, StorageT: TPickleNode;
    n: integer;
  begin
    if Callable.Kind <> pkGlobal then
      Fail('REDUCE callable is not a GLOBAL');
    if Args.Kind <> pkTuple then Fail('REDUCE args are not a tuple');
    if (Callable.StrVal = 'collections') and
       (Callable.NameVal = 'OrderedDict') then
      // OrderedDict(): a fresh dict; pickle fills it via SETITEM(S).
      exit(NewNode(pkDict));
    if (Callable.StrVal = 'torch._utils') and
       (Callable.NameVal = '_rebuild_parameter') then
    begin
      // _rebuild_parameter(tensor, requires_grad, hooks) -> the tensor.
      if (Length(Args.Items) < 1) or (Args.Items[0].Kind <> pkTensor) then
        Fail('_rebuild_parameter expects a tensor argument');
      exit(Args.Items[0]);
    end;
    if (Callable.StrVal = 'torch._utils') and
       (Callable.NameVal = '_rebuild_tensor_v2') then
    begin
      // _rebuild_tensor_v2(storage_persid, storage_offset, size, stride,
      //                    requires_grad, backward_hooks[, metadata])
      if Length(Args.Items) < 6 then
        Fail('_rebuild_tensor_v2 expects at least 6 arguments');
      PersId := Args.Items[0];
      if (PersId.Kind <> pkPersId) or (PersId.Items[0].Kind <> pkTuple) then
        Fail('_rebuild_tensor_v2 storage is not a persistent id');
      // The torch persistent id is ('storage', StorageClass, key, location,
      // numel) - see torch.serialization persistent_id.
      StorageT := PersId.Items[0];
      if (Length(StorageT.Items) <> 5) or
         (StorageT.Items[0].Kind <> pkStr) or
         (StorageT.Items[0].StrVal <> 'storage') or
         (StorageT.Items[1].Kind <> pkGlobal) or
         (StorageT.Items[2].Kind <> pkStr) then
        Fail('unrecognized persistent id (expected the 5-element ' +
          '''storage'' tuple)');
      Result := NewNode(pkTensor);
      Result.StorageKey := StorageT.Items[2].StrVal;
      Result.StorageDType := StorageT.Items[1].NameVal; // class name for now
      Result.StorageNumel :=
        NodeAsInt(StorageT.Items[4], 'storage numel');
      Result.StorageOffset := NodeAsInt(Args.Items[1], 'storage_offset');
      SizeT := Args.Items[2];
      StrideT := Args.Items[3];
      if (SizeT.Kind <> pkTuple) or (StrideT.Kind <> pkTuple) or
         (Length(SizeT.Items) <> Length(StrideT.Items)) then
        Fail('_rebuild_tensor_v2 size/stride are not matching tuples');
      SetLength(Result.Shape, Length(SizeT.Items));
      SetLength(Result.Stride, Length(StrideT.Items));
      for n := 0 to High(SizeT.Items) do
      begin
        Result.Shape[n] := NodeAsInt(SizeT.Items[n], 'tensor size');
        Result.Stride[n] := NodeAsInt(StrideT.Items[n], 'tensor stride');
      end;
      exit;
    end;
    Fail('REDUCE on non-whitelisted callable "' + Callable.StrVal + ' ' +
      Callable.NameVal + '"');
    Result := nil; // unreachable
  end;

  procedure HandleGlobal(const Module, Name: string);
  var
    G: TPickleNode;
  begin
    if not GlobalIsWhitelisted(Module, Name) then
      Fail('refusing GLOBAL "' + Module + ' ' + Name + '" - only the ' +
        'state_dict whitelist (collections OrderedDict, torch._utils ' +
        '_rebuild_tensor_v2/_rebuild_parameter, torch *Storage) is allowed');
    G := NewNode(pkGlobal);
    G.StrVal := Module;
    G.NameVal := Name;
    Push(G);
  end;

var
  Op: byte;
  Module, SymName: string;
  Cnt: Cardinal;
  LongLen: byte;
  Accum: QWord;
  Stopped: boolean;
  FloatBits: QWord;
begin
  Owned := TFPList.Create;
  try
    PickleLen := Length(Pickle);
    Pos := 0;
    StackTop := -1;
    SetLength(Stack, 64);
    SetLength(Memo, 256);
    for i := 0 to High(Memo) do Memo[i] := nil;
    MemoNext := 0;
    Root := nil;
    Stopped := false;
    while not Stopped do
    begin
      Op := ReadByte;
      case Op of
        $80: // PROTO
          if ReadByte > 5 then Fail('unsupported pickle protocol');
        $2E: // STOP '.'
          begin
            Root := Pop;
            Stopped := true;
          end;
        $7D: Push(NewNode(pkDict));   // EMPTY_DICT '}'
        $5D: Push(NewNode(pkList));   // EMPTY_LIST ']'
        $29: Push(NewNode(pkTuple));  // EMPTY_TUPLE ')'
        $28: Push(NewNode(pkMark));   // MARK '('
        $4E: Push(NewNode(pkNone));   // NONE 'N'
        $88, $89: // NEWTRUE / NEWFALSE
          begin
            Node := NewNode(pkBool);
            Node.IntVal := Ord(Op = $88);
            Push(Node);
          end;
        $4B: // BININT1 'K'
          begin
            Node := NewNode(pkInt);
            Node.IntVal := ReadByte;
            Push(Node);
          end;
        $4D: // BININT2 'M'
          begin
            Node := NewNode(pkInt);
            Node.IntVal := ReadUInt2;
            Push(Node);
          end;
        $4A: // BININT 'J' (signed 32-bit)
          begin
            Node := NewNode(pkInt);
            Node.IntVal := Int32(ReadUInt4);
            Push(Node);
          end;
        $8A: // LONG1: n bytes little-endian two's complement
          begin
            LongLen := ReadByte;
            if LongLen > 8 then
              Fail('LONG1 integer wider than 64 bits');
            Accum := 0;
            for i := 0 to LongLen - 1 do
              Accum := Accum or (QWord(ReadByte) shl (8 * i));
            // Sign-extend from the top bit of the last byte.
            if (LongLen > 0) and (LongLen < 8) and
               (Accum and (QWord(1) shl (8 * LongLen - 1)) <> 0) then
              Accum := Accum or (QWord($FFFFFFFFFFFFFFFF) shl (8 * LongLen));
            Node := NewNode(pkInt);
            Node.IntVal := Int64(Accum);
            Push(Node);
          end;
        $47: // BINFLOAT 'G': 8-byte BIG-endian IEEE double
          begin
            FloatBits := 0;
            for i := 0 to 7 do
              FloatBits := (FloatBits shl 8) or ReadByte;
            Node := NewNode(pkFloat);
            Node.FloatVal := PDouble(@FloatBits)^;
            Push(Node);
          end;
        $58: // BINUNICODE 'X'
          begin
            Node := NewNode(pkStr);
            Node.StrVal := ReadString(ReadUInt4);
            Push(Node);
          end;
        $8C: // SHORT_BINUNICODE (protocol 4)
          begin
            Node := NewNode(pkStr);
            Node.StrVal := ReadString(ReadByte);
            Push(Node);
          end;
        $55: // SHORT_BINSTRING 'U'
          begin
            Node := NewNode(pkStr);
            Node.StrVal := ReadString(ReadByte);
            Push(Node);
          end;
        $71: MemoPut(ReadByte);    // BINPUT 'q'
        $72: MemoPut(ReadUInt4);   // LONG_BINPUT 'r'
        $94: // MEMOIZE (protocol 4)
          begin
            Cnt := MemoNext;
            MemoPut(Cnt);
          end;
        $68: Push(MemoGet(ReadByte));  // BINGET 'h'
        $6A: Push(MemoGet(ReadUInt4)); // LONG_BINGET 'j'
        $63: // GLOBAL 'c': two newline-terminated lines
          begin
            Module := ReadLine;
            SymName := ReadLine;
            HandleGlobal(Module, SymName);
          end;
        $93: // STACK_GLOBAL (protocol 4): pops name, module
          begin
            Node := Pop; // name
            ValNode := Pop; // module
            if (Node.Kind <> pkStr) or (ValNode.Kind <> pkStr) then
              Fail('STACK_GLOBAL arguments are not strings');
            HandleGlobal(ValNode.StrVal, Node.StrVal);
          end;
        $51: // BINPERSID 'Q': wrap the popped tuple as a persistent id
          begin
            Node := NewNode(pkPersId);
            SetLength(Node.Items, 1);
            Node.Items[0] := Pop;
            Push(Node);
          end;
        $74: Push(PopToMark(pkTuple)); // TUPLE 't'
        $85: Push(TupleN(1));          // TUPLE1
        $86: Push(TupleN(2));          // TUPLE2
        $87: Push(TupleN(3));          // TUPLE3
        $52: // REDUCE 'R'
          begin
            Node := Pop;    // args
            ValNode := Pop; // callable
            Push(ApplyReduce(ValNode, Node));
          end;
        $73: // SETITEM 's'
          begin
            Node := Pop;    // value
            ValNode := Pop; // key
            if StackTop < 0 then Fail('SETITEM on an empty stack');
            DictSetItem(Stack[StackTop], ValNode, Node);
          end;
        $75: // SETITEMS 'u': alternating key/value pairs down to MARK
          begin
            Node := PopToMark(pkTuple);
            if Length(Node.Items) mod 2 <> 0 then
              Fail('SETITEMS with an odd number of stack entries');
            if StackTop < 0 then Fail('SETITEMS on an empty stack');
            i := 0;
            while i < Length(Node.Items) do
            begin
              DictSetItem(Stack[StackTop], Node.Items[i],
                Node.Items[i + 1]);
              i := i + 2;
            end;
          end;
        $61: // APPEND 'a'
          begin
            Node := Pop;
            if (StackTop < 0) or (Stack[StackTop].Kind <> pkList) then
              Fail('APPEND on a non-list');
            ValNode := Stack[StackTop];
            SetLength(ValNode.Items, Length(ValNode.Items) + 1);
            ValNode.Items[High(ValNode.Items)] := Node;
          end;
        $65: // APPENDS 'e'
          begin
            Node := PopToMark(pkTuple);
            if (StackTop < 0) or (Stack[StackTop].Kind <> pkList) then
              Fail('APPENDS on a non-list');
            ValNode := Stack[StackTop];
            j := Length(ValNode.Items);
            SetLength(ValNode.Items, j + Length(Node.Items));
            for i := 0 to High(Node.Items) do
              ValNode.Items[j + i] := Node.Items[i];
          end;
        $95: // FRAME (protocol 4): 8-byte length hint, ignored
          NeedBytes(8);
        else
          Fail(Format('unsupported pickle opcode $%2.2x - outside the ' +
            'restricted state_dict subset', [Op]));
      end;
    end;
    if (Root = nil) or (Root.Kind <> pkDict) then
      raise ETorchBinError.CreateFmt(
        'torch.bin: the pickled object is not a dict/state_dict: %s',
        [ShardPath]);

    // ---- materialize the tensor table from the unpickled state_dict ----
    // Appends to FTensors: with a sharded checkpoint each shard's
    // state_dict contributes its own tensors (duplicate names across
    // shards are rejected via FindTensor below).
    TensorCnt := Length(FTensors);
    SetLength(FTensors, TensorCnt + Length(Root.Keys));
    for i := 0 to High(Root.Keys) do
    begin
      if (Root.Keys[i].Kind <> pkStr) then
        raise ETorchBinError.CreateFmt(
          'torch.bin: state_dict key %d is not a string: %s',
          [i, ShardPath]);
      ValNode := Root.Vals[i];
      if ValNode.Kind <> pkTensor then continue; // tolerate metadata values
      Node := ValNode;
      if FindTensor(Root.Keys[i].StrVal) >= 0 then
        raise ETorchBinError.CreateFmt(
          'torch.bin: duplicate state_dict tensor "%s": %s',
          [Root.Keys[i].StrVal, ShardPath]);
      // Contiguity: state_dict tensors are normally contiguous; reject the
      // stride-permuted exception loudly rather than stride-walk in v1.
      // (PyTorch semantics: strides of size<=1 dims are irrelevant.)
      NumElements := 1;
      ExpectedStride := 1;
      for j := High(Node.Shape) downto 0 do
      begin
        if (Node.Shape[j] > 1) and (Node.Stride[j] <> ExpectedStride) then
          raise ETorchBinError.CreateFmt(
            'torch.bin: tensor "%s" is not contiguous (dim %d stride %d, ' +
            'expected %d) - non-contiguous tensors are not supported: %s',
            [Root.Keys[i].StrVal, j, Node.Stride[j], ExpectedStride,
             ShardPath]);
        ExpectedStride := ExpectedStride * Node.Shape[j];
        NumElements := NumElements * Node.Shape[j];
      end;
      FTensors[TensorCnt].Name := Root.Keys[i].StrVal;
      FTensors[TensorCnt].Shard := FCurShard;
      FTensors[TensorCnt].DType :=
        DTypeOfStorageClass(Node.StorageDType, ElemSize);
      SetLength(FTensors[TensorCnt].Shape, Length(Node.Shape));
      for j := 0 to High(Node.Shape) do
        FTensors[TensorCnt].Shape[j] := Node.Shape[j];
      // Resolve the storage zip entry to an ABSOLUTE byte range
      // (FDataStarts[FCurShard] = 0, so the inherited LoadTensorFlat reads
      // it directly; views share a storage via their element offsets).
      EntryName := FArchivePrefix + 'data/' + Node.StorageKey;
      EntryIdx := FindZipEntry(EntryName);
      if EntryIdx < 0 then
        raise ETorchBinError.CreateFmt(
          'torch.bin: tensor "%s" references missing storage entry "%s": %s',
          [Root.Keys[i].StrVal, EntryName, ShardPath]);
      if FZipMethod[EntryIdx] <> 0 then
        raise ETorchBinError.CreateFmt(
          'torch.bin: storage entry "%s" is compressed (method %d); only ' +
          'STORED entries are supported: %s',
          [EntryName, FZipMethod[EntryIdx], ShardPath]);
      if QWord(Node.StorageNumel) * QWord(ElemSize) > FZipSize[EntryIdx] then
        raise ETorchBinError.CreateFmt(
          'torch.bin: storage "%s" declares %d %s elements but the zip ' +
          'entry holds only %d bytes: %s',
          [EntryName, Node.StorageNumel, FTensors[TensorCnt].DType,
           FZipSize[EntryIdx], ShardPath]);
      ByteBegin := ZipEntryDataOffset(FStreams[FCurShard], EntryIdx) +
        Node.StorageOffset * ElemSize;
      ByteEnd := ByteBegin + NumElements * ElemSize;
      if (Node.StorageOffset < 0) or
         (Node.StorageOffset + NumElements > Node.StorageNumel) then
        raise ETorchBinError.CreateFmt(
          'torch.bin: tensor "%s" (offset %d + %d elements) overruns its ' +
          'storage (%d elements): %s',
          [Root.Keys[i].StrVal, Node.StorageOffset, NumElements,
           Node.StorageNumel, ShardPath]);
      FTensors[TensorCnt].DataBegin := ByteBegin;
      FTensors[TensorCnt].DataEnd := ByteEnd;
      Inc(TensorCnt);
    end;
    SetLength(FTensors, TensorCnt);
  finally
    OwnedCount := Owned.Count;
    for i := 0 to OwnedCount - 1 do
      TObject(Owned[i]).Free;
    Owned.Free;
  end;
end;

end.
