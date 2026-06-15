unit neuralnumpy;
// Pure-Pascal reader AND writer for NumPy's .npy / .npz interchange formats -
// the universal escape hatch for moving raw tensors between this library and
// the Python/NumPy ecosystem (np.save / np.load / np.savez).
//
// .npy FILE LAYOUT (format versions 1.0, 2.0 and 3.0):
//   bytes 0..5   : the 6-byte magic string  \x93NUMPY
//   byte  6      : major version (1, 2 or 3)
//   byte  7      : minor version (0)
//   v1.0         : bytes 8..9   = little-endian uint16 HEADER_LEN
//   v2.0/v3.0    : bytes 8..11  = little-endian uint32 HEADER_LEN
//   then HEADER_LEN bytes of an ASCII Python-dict literal, e.g.
//     {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }
//   padded with spaces (and a trailing \n) so the raw data that follows starts
//   on a 64-byte boundary (16-byte for the v1 era, but always 64 in practice).
//   then prod(shape) elements in the declared dtype, ROW-MAJOR (C order).
//
// .npz FILE LAYOUT: a ZIP archive whose entries are named "<key>.npy". Each
// entry is one .npy blob as above; the dictionary key is the entry name with
// the trailing ".npy" stripped. np.savez writes STORED (uncompressed) entries;
// np.savez_compressed writes DEFLATE entries. This reader handles STORED for
// sure and DEFLATE when FPC's paszlib is available (it is, by default); a
// compressed entry on a build without paszlib raises a clear error.
//
// SUPPORTED dtypes (little-endian and byte-order-agnostic '|' for 1-byte):
//   floats : <f4 / =f4 (F32), <f8 (F64), <f2 (F16, decoded to single)
//   ints   : <i1 |i1 (int8), <i2 (int16), <i4 (int32), <i8 (int64),
//            <u1 |u1 (uint8), <u2/<u4/<u8 (uint16/32/64)
//   bool   : |b1 (read as 0.0 / 1.0)
// Every load returns 32-bit singles (this library's native element type) in
// the stored row-major element order. Big-endian dtypes ('>...'), Fortran
// order ('fortran_order': True) and pickled-object arrays ('descr': '|O...')
// are explicitly REJECTED with a descriptive ENumpyError - the reader never
// silently returns garbage.
//
// SHAPE <-> TNNetVolume CONVENTION: the flat row-major data is loaded verbatim
// into Dest.FData; the volume dimensions are set so element addressing matches
// numpy's C order:
//   0-D scalar  -> ReSize(1, 1, 1)
//   1-D (N,)    -> ReSize(N, 1, 1)            FData[i]      = a[i]
//   2-D (R, C)  -> ReSize(C, R, 1)            FData[r*C+c]  = a[r, c]
//   3-D (D,H,W) -> ReSize(W, H, D)            (X=W, Y=H, Depth... see below)
//   n-D         -> flat ReSize(prod, 1, 1); use GetOriginalShape to reshape.
// For 3-D the natural CAI image layout is (Depth, H, W) with Depth contiguous,
// which numpy stores as the LEADING axis, not the trailing one; to avoid a
// surprising transpose we map 3-D as SizeX=last, SizeY=mid, Depth=first ONLY
// when that preserves the row-major flat order, which it does NOT in general -
// so 3-D+ arrays are loaded FLAT (prod,1,1) and the caller reshapes from
// GetOriginalShape. 1-D and 2-D (the common interchange cases) map naturally.
//
// This unit is coded by Claude (AI).
{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, StrUtils, neuralvolume, neuralsafetensors;

type
  ENumpyError = class(Exception);

  TInt64DynArray = array of Int64;

  // A named tensor parsed from a .npy blob, kept alongside its original numpy
  // shape (the volume itself may be a flat reshape - see the unit header).
  TNumpyTensor = record
    Name: string;          // key (npz entry name without ".npy"); '' for .npy
    DType: string;         // normalized descr, e.g. 'f4','f8','f2','i4','i8'
    Shape: array of Int64; // numpy shape, leading axis first (C order)
  end;

// --- .npy single-array API --------------------------------------------------

// Loads a .npy file into a freshly-allocated TNNetVolume (caller frees it).
// Dimensions follow the SHAPE convention in the unit header.
function LoadVolumeFromNpy(const pFileName: string): TNNetVolume;
// As above but loads from an already-open stream (used by the .npz reader).
// On return Stream.Position sits just past the consumed array. OutShape and
// OutDType receive the original numpy shape / dtype. Loads into Dest (resized).
procedure LoadNpyFromStream(Stream: TStream; Dest: TNNetVolume;
  out OutShape: TInt64DynArray; out OutDType: string);

// Writes Dest to a .npy file using the given numpy shape (which must have the
// same element count as Dest.Size). pDType selects the on-disk dtype:
//   'f4' (default), 'f8', 'f2', 'i1','i2','i4','i8','u1'. Data is written
// row-major in Dest.FData order. Use an empty pShape to infer (Dest.Size,).
procedure SaveVolumeToNpy(const pFileName: string; Src: TNNetVolume;
  const pShape: array of Int64; const pDType: string = 'f4');

// --- .npz dictionary API ----------------------------------------------------

type
  { TNNetNpzReader }
  // Opens a .npz archive, lists its arrays and loads them by key on demand.
  // STORED and (paszlib-backed) DEFLATE entries are supported; anything else
  // raises ENumpyError.
  TNNetNpzReader = class
  private
    FFileName: string;
    FStream: TFileStream;
    FNames: array of string;       // entry names WITHOUT the ".npy" suffix
    FMethod: array of word;        // 0 = stored, 8 = deflate
    FCompSize: array of QWord;
    FSize: array of QWord;
    FLocalOfs: array of QWord;
    procedure ParseZipDirectory;
    function EntryRawBytes(Index: integer): TBytes;
  public
    constructor Create(const pFileName: string);
    destructor Destroy; override;
    function Count: integer;
    function Name(Index: integer): string;
    function HasKey(const pKey: string): boolean;
    function IndexOfKey(const pKey: string): integer;
    // Loads the named array into Dest (resized) and returns its numpy shape.
    function LoadVolume(const pKey: string; Dest: TNNetVolume): TInt64DynArray;
    // Convenience: a freshly allocated volume (caller frees).
    function LoadVolume(const pKey: string): TNNetVolume;
    property FileName: string read FFileName;
  end;

  { TNNetNpzWriter }
  // Collects named volumes and writes a STORED (uncompressed) .npz archive -
  // exactly what np.savez produces and np.load reads back as a dict. Doubles
  // as fixture tooling for parity tests.
  TNNetNpzWriter = class
  private
    FFileName: string;
    FKeys: TStringList;
    FBlobs: array of TBytes;   // each is a complete .npy blob
    FSaved: boolean;
  public
    constructor Create(const pFileName: string);
    destructor Destroy; override;
    // Queues Src under pKey with the given numpy shape and dtype (see
    // SaveVolumeToNpy). Empty shape -> (Src.Size,).
    procedure AddVolume(const pKey: string; Src: TNNetVolume;
      const pShape: array of Int64; const pDType: string = 'f4');
    // Writes the archive. Call once.
    procedure Save;
    property FileName: string read FFileName;
  end;

// --- low-level helpers (exposed for tests / fixture tooling) ----------------

// IEEE-754 single -> half (round-to-nearest-even), the inverse of
// neuralsafetensors.DecodeF16. Saturates to +/-Inf on overflow.
function EncodeF16(Value: Single): Word;
// Builds a complete .npy blob (header + raw data) in memory.
function BuildNpyBlob(Src: TNNetVolume; const pShape: array of Int64;
  const pDType: string): TBytes;

implementation

uses
  Math
  {$IFDEF FPC}, zstream {$ENDIF};

const
  NPY_MAGIC: array[0..5] of byte = ($93, $4E, $55, $4D, $50, $59); // \x93NUMPY

// ---------------------------------------------------------------------------
// F16 encode (the canonical implementation, like DecodeF16, now lives in
// neuralsafetensors so the .npy writer and the safetensors writer share one
// copy; this thin forwarder keeps neuralnumpy.EncodeF16 a stable entry point).
// ---------------------------------------------------------------------------
function EncodeF16(Value: Single): Word;
begin
  Result := neuralsafetensors.EncodeF16(Value);
end;

// ---------------------------------------------------------------------------
// dtype helpers
// ---------------------------------------------------------------------------

// Returns the 2-char numpy type code (e.g. 'f4') for a normalized dtype, and
// the byte size. Raises on unsupported / rejected dtypes.
function DTypeSize(const DType: string): integer;
begin
  if (DType = 'f4') then Result := 4
  else if DType = 'f8' then Result := 8
  else if DType = 'f2' then Result := 2
  else if (DType = 'i1') or (DType = 'u1') or (DType = 'b1') then Result := 1
  else if (DType = 'i2') or (DType = 'u2') then Result := 2
  else if (DType = 'i4') or (DType = 'u4') then Result := 4
  else if (DType = 'i8') or (DType = 'u8') then Result := 8
  else
    raise ENumpyError.CreateFmt('numpy: unsupported dtype "%s"', [DType]);
  if Result = 0 then ; // silence hint
end;

// Parses a numpy 'descr' string (e.g. '<f4', '=f8', '|u1', '|b1') into our
// normalized 2-char code, rejecting big-endian, Fortran order and object
// dtypes. Byte order: '<' (LE) and '=' (native, assumed LE here) and '|'
// (not-applicable, 1-byte) are accepted; '>' big-endian is rejected.
function NormalizeDescr(const Descr: string): string;
var
  ByteOrder: char;
  Kind: char;
  Rest: string;
begin
  if Length(Descr) < 2 then
    raise ENumpyError.CreateFmt('numpy: malformed descr "%s"', [Descr]);
  ByteOrder := Descr[1];
  if (ByteOrder = '<') or (ByteOrder = '=') or (ByteOrder = '|') then
  begin
    Kind := Descr[2];
    Rest := Copy(Descr, 2, MaxInt);
  end
  else if (ByteOrder >= 'a') then
  begin
    // no byte-order prefix (e.g. just 'f4'); rare but legal
    Kind := Descr[1];
    Rest := Descr;
    ByteOrder := '|';
  end
  else if ByteOrder = '>' then
    raise ENumpyError.CreateFmt(
      'numpy: big-endian arrays are not supported (descr "%s"); ' +
      're-save with a little-endian dtype', [Descr])
  else
    raise ENumpyError.CreateFmt('numpy: unrecognized byte order in "%s"',
      [Descr]);
  if (Kind = 'O') then
    raise ENumpyError.Create(
      'numpy: object/pickled arrays (dtype ''O'') are not supported');
  Result := LowerCase(Rest);
  // validate by asking for its size (raises if unsupported)
  DTypeSize(Result);
end;

// ---------------------------------------------------------------------------
// little-endian readers over a byte buffer
// ---------------------------------------------------------------------------
function ReadU16LE(const B: TBytes; Ofs: integer): Word; inline;
begin
  Result := B[Ofs] or (Word(B[Ofs + 1]) shl 8);
end;

function ReadU32LE(const B: TBytes; Ofs: integer): Cardinal; inline;
begin
  Result := B[Ofs] or (Cardinal(B[Ofs + 1]) shl 8) or
    (Cardinal(B[Ofs + 2]) shl 16) or (Cardinal(B[Ofs + 3]) shl 24);
end;

// ---------------------------------------------------------------------------
// header dict parsing (a tiny, restricted Python-literal scanner)
// ---------------------------------------------------------------------------

// Extracts the descr string, fortran_order flag and shape tuple from the
// header dict text. Deliberately permissive about whitespace and quote style.
procedure ParseHeaderDict(const Header: string; out Descr: string;
  out FortranOrder: boolean; out Shape: TInt64DynArray);
var
  P, Q: integer;
  ShapeStr, Num: string;
  Vals: array of Int64;
  i: integer;
  C: char;

  function FindKey(const Key: string): integer;
  var idx: integer;
  begin
    idx := Pos(Key, Header);
    Result := idx;
  end;

begin
  // --- descr ---
  P := Pos('''descr''', Header);
  if P = 0 then P := Pos('"descr"', Header);
  if P = 0 then
    raise ENumpyError.Create('numpy: header has no ''descr'' key');
  // find the value: skip to the colon, then the opening quote
  Q := PosEx(':', Header, P);
  // find first quote after colon
  i := Q + 1;
  while (i <= Length(Header)) and not (Header[i] in ['''', '"']) do Inc(i);
  if i > Length(Header) then
    raise ENumpyError.Create('numpy: malformed ''descr'' value');
  C := Header[i];
  Inc(i);
  Descr := '';
  while (i <= Length(Header)) and (Header[i] <> C) do
  begin
    Descr := Descr + Header[i];
    Inc(i);
  end;

  // --- fortran_order ---
  P := Pos('fortran_order', Header);
  if P = 0 then
    raise ENumpyError.Create('numpy: header has no ''fortran_order'' key');
  Q := PosEx(':', Header, P);
  // look for "True" / "False" token after the colon
  if Pos('True', Copy(Header, Q, 8)) > 0 then
    FortranOrder := true
  else if Pos('False', Copy(Header, Q, 9)) > 0 then
    FortranOrder := false
  else
    raise ENumpyError.Create('numpy: malformed ''fortran_order'' value');

  // --- shape ---
  P := Pos('''shape''', Header);
  if P = 0 then P := Pos('"shape"', Header);
  if P = 0 then
    raise ENumpyError.Create('numpy: header has no ''shape'' key');
  Q := PosEx('(', Header, P);
  if Q = 0 then
    raise ENumpyError.Create('numpy: malformed ''shape'' value (no tuple)');
  i := Q + 1;
  ShapeStr := '';
  while (i <= Length(Header)) and (Header[i] <> ')') do
  begin
    ShapeStr := ShapeStr + Header[i];
    Inc(i);
  end;
  // ShapeStr is like "3, 4" or "5" or "" (0-D scalar) or "5," (1-tuple)
  SetLength(Vals, 0);
  Num := '';
  for i := 1 to Length(ShapeStr) + 1 do
  begin
    if (i <= Length(ShapeStr)) and (ShapeStr[i] in ['0'..'9']) then
      Num := Num + ShapeStr[i]
    else
    begin
      if Num <> '' then
      begin
        SetLength(Vals, Length(Vals) + 1);
        Vals[High(Vals)] := StrToInt64(Num);
        Num := '';
      end;
    end;
  end;
  SetLength(Shape, Length(Vals));
  for i := 0 to High(Vals) do Shape[i] := Vals[i];
end;

// ---------------------------------------------------------------------------
// raw-bytes -> volume decode (shared by .npy and .npz)
// ---------------------------------------------------------------------------
procedure DecodeRawToVolume(const Raw: TBytes; NumElements: Int64;
  const DType: string; Dest: TNNetVolume);
var
  i: Int64;
  pf4: PSingle;
  pf8: PDouble;
  pu16: PWord;
  pi1: PShortInt;
  pu1: PByte;
  pi2: PSmallInt;
  pi4: PLongInt;
  pi8: PInt64;
begin
  // Dest is expected pre-sized to NumElements by the caller.
  if DType = 'f4' then
  begin
    pf4 := PSingle(@Raw[0]);
    for i := 0 to NumElements - 1 do begin Dest.FData[i] := pf4^; Inc(pf4); end;
  end
  else if DType = 'f8' then
  begin
    pf8 := PDouble(@Raw[0]);
    for i := 0 to NumElements - 1 do begin Dest.FData[i] := pf8^; Inc(pf8); end;
  end
  else if DType = 'f2' then
  begin
    pu16 := PWord(@Raw[0]);
    for i := 0 to NumElements - 1 do
      begin Dest.FData[i] := DecodeF16(pu16^); Inc(pu16); end;
  end
  else if DType = 'i1' then
  begin
    pi1 := PShortInt(@Raw[0]);
    for i := 0 to NumElements - 1 do begin Dest.FData[i] := pi1^; Inc(pi1); end;
  end
  else if (DType = 'u1') or (DType = 'b1') then
  begin
    pu1 := PByte(@Raw[0]);
    for i := 0 to NumElements - 1 do begin Dest.FData[i] := pu1^; Inc(pu1); end;
  end
  else if DType = 'i2' then
  begin
    pi2 := PSmallInt(@Raw[0]);
    for i := 0 to NumElements - 1 do begin Dest.FData[i] := pi2^; Inc(pi2); end;
  end
  else if DType = 'u2' then
  begin
    pu16 := PWord(@Raw[0]);
    for i := 0 to NumElements - 1 do begin Dest.FData[i] := pu16^; Inc(pu16); end;
  end
  else if DType = 'i4' then
  begin
    pi4 := PLongInt(@Raw[0]);
    for i := 0 to NumElements - 1 do begin Dest.FData[i] := pi4^; Inc(pi4); end;
  end
  else if DType = 'u4' then
  begin
    pi8 := nil; // unused
    for i := 0 to NumElements - 1 do
      Dest.FData[i] := Cardinal(ReadU32LE(Raw, i * 4));
    if pi8 = nil then ;
  end
  else if (DType = 'i8') or (DType = 'u8') then
  begin
    pi8 := PInt64(@Raw[0]);
    for i := 0 to NumElements - 1 do begin Dest.FData[i] := pi8^; Inc(pi8); end;
  end
  else
    raise ENumpyError.CreateFmt('numpy: unsupported dtype "%s"', [DType]);
end;

// Maps a numpy C-order shape onto the volume dimensions (see unit header).
procedure ApplyShapeToVolume(Dest: TNNetVolume; const Shape: TInt64DynArray;
  NumElements: Int64);
begin
  case Length(Shape) of
    0: Dest.ReSize(1, 1, 1);                       // 0-D scalar
    1: Dest.ReSize(integer(Shape[0]), 1, 1);       // (N,)
    2: Dest.ReSize(integer(Shape[1]), integer(Shape[0]), 1); // (R,C)->X=C,Y=R
  else
    Dest.ReSize(integer(NumElements), 1, 1);       // n-D: flat, caller reshapes
  end;
end;

// ---------------------------------------------------------------------------
// .npy stream reader
// ---------------------------------------------------------------------------
procedure LoadNpyFromStream(Stream: TStream; Dest: TNNetVolume;
  out OutShape: TInt64DynArray; out OutDType: string);
var
  Magic: array[0..5] of byte;
  Major, Minor: byte;
  HeaderLen: Cardinal;
  Lo16: Word;
  HeaderBytes: TBytes;
  Header: string;
  Descr: string;
  FortranOrder: boolean;
  Shape: TInt64DynArray;
  NumElements: Int64;
  i: integer;
  ElemSize: integer;
  Raw: TBytes;
begin
  if Stream.Read(Magic, 6) <> 6 then
    raise ENumpyError.Create('numpy: stream too short for magic');
  for i := 0 to 5 do
    if Magic[i] <> NPY_MAGIC[i] then
      raise ENumpyError.Create('numpy: bad magic (not a .npy stream)');
  if Stream.Read(Major, 1) <> 1 then
    raise ENumpyError.Create('numpy: truncated version');
  Stream.Read(Minor, 1);
  if (Major < 1) or (Major > 3) then
    raise ENumpyError.CreateFmt('numpy: unsupported format version %d.%d',
      [Major, Minor]);
  if Major = 1 then
  begin
    if Stream.Read(Lo16, 2) <> 2 then
      raise ENumpyError.Create('numpy: truncated header length');
    HeaderLen := LEtoN(Lo16);
  end
  else
  begin
    if Stream.Read(HeaderLen, 4) <> 4 then
      raise ENumpyError.Create('numpy: truncated header length');
    HeaderLen := LEtoN(HeaderLen);
  end;
  SetLength(HeaderBytes, HeaderLen);
  if HeaderLen > 0 then
    if Stream.Read(HeaderBytes[0], HeaderLen) <> integer(HeaderLen) then
      raise ENumpyError.Create('numpy: truncated header dict');
  SetLength(Header, HeaderLen);
  for i := 0 to integer(HeaderLen) - 1 do
    Header[i + 1] := Chr(HeaderBytes[i]);

  ParseHeaderDict(Header, Descr, FortranOrder, Shape);
  OutDType := NormalizeDescr(Descr);
  OutShape := Shape;

  if FortranOrder then
    raise ENumpyError.Create(
      'numpy: Fortran-order (column-major) arrays are not supported; ' +
      're-save with np.ascontiguousarray');

  NumElements := 1;
  for i := 0 to High(Shape) do NumElements := NumElements * Shape[i];
  if Length(Shape) = 0 then NumElements := 1;
  if NumElements > High(integer) then
    raise ENumpyError.CreateFmt('numpy: array too large (%d elements)',
      [NumElements]);

  ElemSize := DTypeSize(OutDType);
  SetLength(Raw, NumElements * ElemSize);
  if NumElements > 0 then
    if Stream.Read(Raw[0], Length(Raw)) <> Length(Raw) then
      raise ENumpyError.Create('numpy: truncated data section');

  ApplyShapeToVolume(Dest, Shape, NumElements);
  if Dest.Size <> NumElements then
    raise ENumpyError.CreateFmt(
      'numpy: internal shape/volume mismatch (%d vs %d)',
      [NumElements, Dest.Size]);
  if NumElements > 0 then
    DecodeRawToVolume(Raw, NumElements, OutDType, Dest);
end;

function LoadVolumeFromNpy(const pFileName: string): TNNetVolume;
var
  Stream: TFileStream;
  Shape: TInt64DynArray;
  DType: string;
begin
  if not FileExists(pFileName) then
    raise ENumpyError.CreateFmt('numpy: file not found: %s', [pFileName]);
  Result := TNNetVolume.Create(1, 1, 1);
  Stream := TFileStream.Create(pFileName, fmOpenRead or fmShareDenyWrite);
  try
    try
      LoadNpyFromStream(Stream, Result, Shape, DType);
    except
      Result.Free;
      raise;
    end;
  finally
    Stream.Free;
  end;
end;

// ---------------------------------------------------------------------------
// .npy writer
// ---------------------------------------------------------------------------

function ShapeTupleStr(const Shape: array of Int64): string;
var i: integer;
begin
  if Length(Shape) = 0 then
    Result := '()'
  else if Length(Shape) = 1 then
    Result := '(' + IntToStr(Shape[0]) + ',)'  // numpy writes 1-tuples w/ comma
  else
  begin
    Result := '(';
    for i := 0 to High(Shape) do
    begin
      Result := Result + IntToStr(Shape[i]);
      if i < High(Shape) then Result := Result + ', ';
    end;
    Result := Result + ')';
  end;
end;

procedure AppendElement(var Buf: TBytes; var Pos: integer;
  const DType: string; Value: Single);
var
  u: Cardinal;
  d: Double;
  pd: PCardinal;
  w: Word;
  i64: Int64;
begin
  if DType = 'f4' then
  begin
    u := PCardinal(@Value)^;
    Buf[Pos] := u and $FF; Buf[Pos+1] := (u shr 8) and $FF;
    Buf[Pos+2] := (u shr 16) and $FF; Buf[Pos+3] := (u shr 24) and $FF;
    Inc(Pos, 4);
  end
  else if DType = 'f8' then
  begin
    d := Value;
    pd := PCardinal(@d);
    Buf[Pos] := pd^ and $FF; Buf[Pos+1] := (pd^ shr 8) and $FF;
    Buf[Pos+2] := (pd^ shr 16) and $FF; Buf[Pos+3] := (pd^ shr 24) and $FF;
    Inc(pd);
    Buf[Pos+4] := pd^ and $FF; Buf[Pos+5] := (pd^ shr 8) and $FF;
    Buf[Pos+6] := (pd^ shr 16) and $FF; Buf[Pos+7] := (pd^ shr 24) and $FF;
    Inc(Pos, 8);
  end
  else if DType = 'f2' then
  begin
    w := EncodeF16(Value);
    Buf[Pos] := w and $FF; Buf[Pos+1] := (w shr 8) and $FF;
    Inc(Pos, 2);
  end
  else if (DType = 'i1') or (DType = 'u1') then
  begin
    Buf[Pos] := Round(Value) and $FF; Inc(Pos);
  end
  else if (DType = 'i2') or (DType = 'u2') then
  begin
    i64 := Round(Value);
    Buf[Pos] := i64 and $FF; Buf[Pos+1] := (i64 shr 8) and $FF; Inc(Pos, 2);
  end
  else if (DType = 'i4') or (DType = 'u4') then
  begin
    i64 := Round(Value);
    Buf[Pos] := i64 and $FF; Buf[Pos+1] := (i64 shr 8) and $FF;
    Buf[Pos+2] := (i64 shr 16) and $FF; Buf[Pos+3] := (i64 shr 24) and $FF;
    Inc(Pos, 4);
  end
  else if (DType = 'i8') or (DType = 'u8') then
  begin
    i64 := Round(Value);
    Buf[Pos]   := i64 and $FF;        Buf[Pos+1] := (i64 shr 8) and $FF;
    Buf[Pos+2] := (i64 shr 16) and $FF; Buf[Pos+3] := (i64 shr 24) and $FF;
    Buf[Pos+4] := (i64 shr 32) and $FF; Buf[Pos+5] := (i64 shr 40) and $FF;
    Buf[Pos+6] := (i64 shr 48) and $FF; Buf[Pos+7] := (i64 shr 56) and $FF;
    Inc(Pos, 8);
  end
  else
    raise ENumpyError.CreateFmt('numpy: cannot write dtype "%s"', [DType]);
end;

function BuildNpyBlob(Src: TNNetVolume; const pShape: array of Int64;
  const pDType: string): TBytes;
var
  Shape: array of Int64;
  NumElements: Int64;
  DType, Descr, HeaderText: string;
  ElemSize, i: integer;
  HeaderLen, TotalHeader, PadTo: integer;
  PreambleLen: integer;
  DataPos: integer;
begin
  // normalize dtype (lower, strip byte-order if caller passed one)
  DType := LowerCase(pDType);
  if (Length(DType) >= 1) and (DType[1] in ['<', '=', '|', '>']) then
    DType := Copy(DType, 2, MaxInt);
  ElemSize := DTypeSize(DType);

  if Length(pShape) = 0 then
  begin
    SetLength(Shape, 1);
    Shape[0] := Src.Size;
  end
  else
  begin
    SetLength(Shape, Length(pShape));
    for i := 0 to High(pShape) do Shape[i] := pShape[i];
  end;

  NumElements := 1;
  for i := 0 to High(Shape) do NumElements := NumElements * Shape[i];
  if NumElements <> Src.Size then
    raise ENumpyError.CreateFmt(
      'numpy: shape declares %d elements but volume holds %d',
      [NumElements, Src.Size]);

  Descr := '<' + DType;
  if ElemSize = 1 then Descr := '|' + DType; // byte order N/A for 1-byte
  HeaderText := '{''descr'': ''' + Descr +
    ''', ''fortran_order'': False, ''shape'': ' + ShapeTupleStr(Shape) + ', }';

  // v1.0: 6 magic + 2 version + 2 header-len = 10-byte preamble. Pad the
  // header (with spaces, terminated by \n) so preamble+header is a multiple
  // of 64. NumPy's modern alignment target is 64 bytes.
  PreambleLen := 10;
  PadTo := 64;
  // +1 for the trailing newline that numpy always appends
  TotalHeader := Length(HeaderText) + 1;
  HeaderLen := ((PreambleLen + TotalHeader + PadTo - 1) div PadTo) * PadTo
    - PreambleLen;
  // pad HeaderText with spaces up to HeaderLen-1, then newline
  while Length(HeaderText) < HeaderLen - 1 do HeaderText := HeaderText + ' ';
  HeaderText := HeaderText + #10;
  if Length(HeaderText) <> HeaderLen then
    HeaderLen := Length(HeaderText); // safety (should match)

  SetLength(Result, PreambleLen + HeaderLen + NumElements * ElemSize);
  for i := 0 to 5 do Result[i] := NPY_MAGIC[i];
  Result[6] := 1; // major
  Result[7] := 0; // minor
  Result[8] := HeaderLen and $FF;
  Result[9] := (HeaderLen shr 8) and $FF;
  for i := 1 to Length(HeaderText) do
    Result[PreambleLen + i - 1] := Ord(HeaderText[i]);

  DataPos := PreambleLen + HeaderLen;
  for i := 0 to NumElements - 1 do
    AppendElement(Result, DataPos, DType, Src.FData[i]);
end;

procedure SaveVolumeToNpy(const pFileName: string; Src: TNNetVolume;
  const pShape: array of Int64; const pDType: string = 'f4');
var
  Blob: TBytes;
  Stream: TFileStream;
begin
  Blob := BuildNpyBlob(Src, pShape, pDType);
  Stream := TFileStream.Create(pFileName, fmCreate);
  try
    Stream.WriteBuffer(Blob[0], Length(Blob));
  finally
    Stream.Free;
  end;
end;

// ---------------------------------------------------------------------------
// ZIP CRC-32 (for the writer's central directory)
// ---------------------------------------------------------------------------
var
  CrcTable: array[0..255] of Cardinal;
  CrcReady: boolean = false;

procedure InitCrcTable;
var i, j: integer; c: Cardinal;
begin
  for i := 0 to 255 do
  begin
    c := i;
    for j := 0 to 7 do
      if (c and 1) <> 0 then c := $EDB88320 xor (c shr 1)
      else c := c shr 1;
    CrcTable[i] := c;
  end;
  CrcReady := true;
end;

function Crc32Of(const B: TBytes): Cardinal;
var i: integer; c: Cardinal;
begin
  if not CrcReady then InitCrcTable;
  c := $FFFFFFFF;
  for i := 0 to High(B) do
    c := CrcTable[(c xor B[i]) and $FF] xor (c shr 8);
  Result := c xor $FFFFFFFF;
end;

// ---------------------------------------------------------------------------
// TNNetNpzReader
// ---------------------------------------------------------------------------
constructor TNNetNpzReader.Create(const pFileName: string);
begin
  inherited Create;
  FFileName := pFileName;
  if not FileExists(pFileName) then
    raise ENumpyError.CreateFmt('numpy: .npz file not found: %s', [pFileName]);
  FStream := TFileStream.Create(pFileName, fmOpenRead or fmShareDenyWrite);
  ParseZipDirectory;
end;

destructor TNNetNpzReader.Destroy;
begin
  FStream.Free;
  inherited Destroy;
end;

function ReadDWordAt(const B: TBytes; Ofs: integer): Cardinal; inline;
begin
  Result := B[Ofs] or (Cardinal(B[Ofs+1]) shl 8) or
    (Cardinal(B[Ofs+2]) shl 16) or (Cardinal(B[Ofs+3]) shl 24);
end;

function ReadWordAt(const B: TBytes; Ofs: integer): Word; inline;
begin
  Result := B[Ofs] or (Word(B[Ofs+1]) shl 8);
end;

procedure TNNetNpzReader.ParseZipDirectory;
const
  EOCD_SIG = $06054B50;
  CDH_SIG  = $02014B50;
var
  Tail: TBytes;
  TailLen, ScanPos: integer;
  Found: boolean;
  CdOfs, CdSize: Cardinal;
  EntryCount, EntryCnt: integer;
  CdBytes: TBytes;
  Pos, NameLen, ExtraLen, CommentLen: integer;
  Method: word;
  CompSize, USize, LocalOfs: Cardinal;
  EntryName: string;
  i: integer;
begin
  if FStream.Size < 22 then
    raise ENumpyError.Create('numpy: .npz too small to be a ZIP');
  TailLen := Min(FStream.Size, 22 + 65535);
  SetLength(Tail, TailLen);
  FStream.Position := FStream.Size - TailLen;
  FStream.ReadBuffer(Tail[0], TailLen);
  Found := false;
  ScanPos := TailLen - 22;
  while ScanPos >= 0 do
  begin
    if ReadDWordAt(Tail, ScanPos) = EOCD_SIG then begin Found := true; break; end;
    Dec(ScanPos);
  end;
  if not Found then
    raise ENumpyError.Create('numpy: ZIP end-of-central-directory not found ' +
      '(zip64 or comment-heavy archives unsupported)');
  EntryCount := ReadWordAt(Tail, ScanPos + 10);
  CdSize := ReadDWordAt(Tail, ScanPos + 12);
  CdOfs := ReadDWordAt(Tail, ScanPos + 16);
  if (CdOfs = $FFFFFFFF) or (CdSize = $FFFFFFFF) then
    raise ENumpyError.Create('numpy: zip64 .npz archives are not supported');

  SetLength(CdBytes, CdSize);
  FStream.Position := CdOfs;
  if CdSize > 0 then FStream.ReadBuffer(CdBytes[0], CdSize);

  SetLength(FNames, EntryCount);
  SetLength(FMethod, EntryCount);
  SetLength(FCompSize, EntryCount);
  SetLength(FSize, EntryCount);
  SetLength(FLocalOfs, EntryCount);

  Pos := 0;
  EntryCnt := 0;
  while (EntryCnt < EntryCount) and (Pos + 46 <= integer(CdSize)) do
  begin
    if ReadDWordAt(CdBytes, Pos) <> CDH_SIG then
      raise ENumpyError.Create('numpy: bad central-directory header in .npz');
    Method := ReadWordAt(CdBytes, Pos + 10);
    CompSize := ReadDWordAt(CdBytes, Pos + 20);
    USize := ReadDWordAt(CdBytes, Pos + 24);
    NameLen := ReadWordAt(CdBytes, Pos + 28);
    ExtraLen := ReadWordAt(CdBytes, Pos + 30);
    CommentLen := ReadWordAt(CdBytes, Pos + 32);
    LocalOfs := ReadDWordAt(CdBytes, Pos + 42);
    SetLength(EntryName, NameLen);
    for i := 0 to NameLen - 1 do
      EntryName[i + 1] := Chr(CdBytes[Pos + 46 + i]);
    // strip trailing ".npy" to form the dict key
    if (Length(EntryName) > 4) and
       (LowerCase(Copy(EntryName, Length(EntryName) - 3, 4)) = '.npy') then
      FNames[EntryCnt] := Copy(EntryName, 1, Length(EntryName) - 4)
    else
      FNames[EntryCnt] := EntryName;
    FMethod[EntryCnt] := Method;
    FCompSize[EntryCnt] := CompSize;
    FSize[EntryCnt] := USize;
    FLocalOfs[EntryCnt] := LocalOfs;
    Inc(EntryCnt);
    Pos := Pos + 46 + NameLen + ExtraLen + CommentLen;
  end;
  SetLength(FNames, EntryCnt);
  SetLength(FMethod, EntryCnt);
  SetLength(FCompSize, EntryCnt);
  SetLength(FSize, EntryCnt);
  SetLength(FLocalOfs, EntryCnt);
end;

function TNNetNpzReader.EntryRawBytes(Index: integer): TBytes;
var
  LocalHdr: array[0..29] of byte;
  NameLen, ExtraLen: integer;
  DataOfs: Int64;
  Comp: TBytes;
  {$IFDEF FPC}
  MemIn, MemOut: TMemoryStream;
  Decompressor: TDecompressionStream;
  ReadCnt: integer;
  ChunkBuf: array[0..65535] of byte;
  {$ENDIF}
begin
  FStream.Position := FLocalOfs[Index];
  FStream.ReadBuffer(LocalHdr, 30);
  if (LocalHdr[0] or (LocalHdr[1] shl 8) or (LocalHdr[2] shl 16) or
      (LocalHdr[3] shl 24)) <> $04034B50 then
    raise ENumpyError.Create('numpy: bad local-header signature in .npz');
  NameLen := LocalHdr[26] or (LocalHdr[27] shl 8);
  ExtraLen := LocalHdr[28] or (LocalHdr[29] shl 8);
  DataOfs := FLocalOfs[Index] + 30 + NameLen + ExtraLen;
  SetLength(Comp, FCompSize[Index]);
  FStream.Position := DataOfs;
  if FCompSize[Index] > 0 then FStream.ReadBuffer(Comp[0], FCompSize[Index]);

  if FMethod[Index] = 0 then
    Result := Comp                 // STORED
  else if FMethod[Index] = 8 then  // DEFLATE
  begin
    {$IFDEF FPC}
    MemIn := TMemoryStream.Create;
    MemOut := TMemoryStream.Create;
    try
      if Length(Comp) > 0 then MemIn.WriteBuffer(Comp[0], Length(Comp));
      MemIn.Position := 0;
      // raw DEFLATE (no zlib header) -> negative window via skip_header
      Decompressor := TDecompressionStream.Create(MemIn, true);
      try
        repeat
          ReadCnt := Decompressor.Read(ChunkBuf, SizeOf(ChunkBuf));
          if ReadCnt > 0 then MemOut.WriteBuffer(ChunkBuf, ReadCnt);
        until ReadCnt = 0;
      finally
        Decompressor.Free;
      end;
      SetLength(Result, MemOut.Size);
      if MemOut.Size > 0 then
      begin
        MemOut.Position := 0;
        MemOut.ReadBuffer(Result[0], MemOut.Size);
      end;
    finally
      MemOut.Free;
      MemIn.Free;
    end;
    {$ELSE}
    raise ENumpyError.Create(
      'numpy: DEFLATE-compressed .npz entries require paszlib/zstream; ' +
      're-save with np.savez (uncompressed)');
    {$ENDIF}
  end
  else
    raise ENumpyError.CreateFmt(
      'numpy: unsupported ZIP compression method %d in .npz', [FMethod[Index]]);
end;

function TNNetNpzReader.Count: integer;
begin
  Result := Length(FNames);
end;

function TNNetNpzReader.Name(Index: integer): string;
begin
  Result := FNames[Index];
end;

function TNNetNpzReader.IndexOfKey(const pKey: string): integer;
var i: integer;
begin
  for i := 0 to High(FNames) do
    if FNames[i] = pKey then exit(i);
  Result := -1;
end;

function TNNetNpzReader.HasKey(const pKey: string): boolean;
begin
  Result := IndexOfKey(pKey) >= 0;
end;

function TNNetNpzReader.LoadVolume(const pKey: string;
  Dest: TNNetVolume): TInt64DynArray;
var
  Idx: integer;
  Raw: TBytes;
  MemStream: TMemoryStream;
  DType: string;
begin
  Idx := IndexOfKey(pKey);
  if Idx < 0 then
    raise ENumpyError.CreateFmt('numpy: key "%s" not in %s', [pKey, FFileName]);
  Raw := EntryRawBytes(Idx);
  MemStream := TMemoryStream.Create;
  try
    if Length(Raw) > 0 then MemStream.WriteBuffer(Raw[0], Length(Raw));
    MemStream.Position := 0;
    LoadNpyFromStream(MemStream, Dest, Result, DType);
  finally
    MemStream.Free;
  end;
end;

function TNNetNpzReader.LoadVolume(const pKey: string): TNNetVolume;
begin
  Result := TNNetVolume.Create(1, 1, 1);
  try
    LoadVolume(pKey, Result);
  except
    Result.Free;
    raise;
  end;
end;

// ---------------------------------------------------------------------------
// TNNetNpzWriter (STORED-only)
// ---------------------------------------------------------------------------
constructor TNNetNpzWriter.Create(const pFileName: string);
begin
  inherited Create;
  FFileName := pFileName;
  FKeys := TStringList.Create;
  FSaved := false;
end;

destructor TNNetNpzWriter.Destroy;
begin
  FKeys.Free;
  inherited Destroy;
end;

procedure TNNetNpzWriter.AddVolume(const pKey: string; Src: TNNetVolume;
  const pShape: array of Int64; const pDType: string = 'f4');
var
  Idx: integer;
begin
  if pKey = '' then
    raise ENumpyError.Create('numpy: cannot write an array with empty key');
  if FKeys.IndexOf(pKey) >= 0 then
    raise ENumpyError.CreateFmt('numpy: duplicate key "%s"', [pKey]);
  Idx := FKeys.Add(pKey);
  SetLength(FBlobs, Idx + 1);
  FBlobs[Idx] := BuildNpyBlob(Src, pShape, pDType);
end;

procedure TNNetNpzWriter.Save;
var
  Stream: TFileStream;
  i, n: integer;
  EntryName: string;
  NameBytes: TBytes;
  Crc, CompSize: Cardinal;
  LocalOfs: array of Cardinal;
  CdStart: Cardinal;
  Hdr: TBytes;

  procedure PutW(var B: TBytes; var P: integer; V: Word);
  begin B[P] := V and $FF; B[P+1] := (V shr 8) and $FF; Inc(P, 2); end;
  procedure PutD(var B: TBytes; var P: integer; V: Cardinal);
  begin
    B[P] := V and $FF; B[P+1] := (V shr 8) and $FF;
    B[P+2] := (V shr 16) and $FF; B[P+3] := (V shr 24) and $FF; Inc(P, 4);
  end;

var
  P: integer;
begin
  if FSaved then
    raise ENumpyError.Create('numpy: .npz writer already saved');
  FSaved := true;
  n := FKeys.Count;
  SetLength(LocalOfs, n);
  Stream := TFileStream.Create(FFileName, fmCreate);
  try
    // local file headers + data
    for i := 0 to n - 1 do
    begin
      LocalOfs[i] := Stream.Position;
      EntryName := FKeys[i] + '.npy';
      SetLength(NameBytes, Length(EntryName));
      for P := 0 to Length(EntryName) - 1 do
        NameBytes[P] := Ord(EntryName[P + 1]);
      Crc := Crc32Of(FBlobs[i]);
      CompSize := Length(FBlobs[i]);
      SetLength(Hdr, 30);
      P := 0;
      PutD(Hdr, P, $04034B50);     // local file header sig
      PutW(Hdr, P, 20);            // version needed
      PutW(Hdr, P, 0);             // flags
      PutW(Hdr, P, 0);             // method = STORED
      PutW(Hdr, P, 0);             // mod time
      PutW(Hdr, P, $21);           // mod date (arbitrary valid)
      PutD(Hdr, P, Crc);
      PutD(Hdr, P, CompSize);      // compressed size
      PutD(Hdr, P, CompSize);      // uncompressed size
      PutW(Hdr, P, Length(EntryName));
      PutW(Hdr, P, 0);             // extra len
      Stream.WriteBuffer(Hdr[0], 30);
      Stream.WriteBuffer(NameBytes[0], Length(NameBytes));
      if Length(FBlobs[i]) > 0 then
        Stream.WriteBuffer(FBlobs[i][0], Length(FBlobs[i]));
    end;
    // central directory
    CdStart := Stream.Position;
    for i := 0 to n - 1 do
    begin
      EntryName := FKeys[i] + '.npy';
      SetLength(NameBytes, Length(EntryName));
      for P := 0 to Length(EntryName) - 1 do
        NameBytes[P] := Ord(EntryName[P + 1]);
      Crc := Crc32Of(FBlobs[i]);
      CompSize := Length(FBlobs[i]);
      SetLength(Hdr, 46);
      P := 0;
      PutD(Hdr, P, $02014B50);     // central dir header sig
      PutW(Hdr, P, 20);            // version made by
      PutW(Hdr, P, 20);            // version needed
      PutW(Hdr, P, 0);             // flags
      PutW(Hdr, P, 0);             // method STORED
      PutW(Hdr, P, 0);             // mod time
      PutW(Hdr, P, $21);           // mod date
      PutD(Hdr, P, Crc);
      PutD(Hdr, P, CompSize);
      PutD(Hdr, P, CompSize);
      PutW(Hdr, P, Length(EntryName));
      PutW(Hdr, P, 0);             // extra len
      PutW(Hdr, P, 0);             // comment len
      PutW(Hdr, P, 0);             // disk number start
      PutW(Hdr, P, 0);             // internal attrs
      PutD(Hdr, P, 0);             // external attrs
      PutD(Hdr, P, LocalOfs[i]);   // local header offset
      Stream.WriteBuffer(Hdr[0], 46);
      Stream.WriteBuffer(NameBytes[0], Length(NameBytes));
    end;
    // end of central directory
    SetLength(Hdr, 22);
    P := 0;
    PutD(Hdr, P, $06054B50);
    PutW(Hdr, P, 0);               // this disk
    PutW(Hdr, P, 0);               // cd start disk
    PutW(Hdr, P, n);               // entries this disk
    PutW(Hdr, P, n);               // total entries
    PutD(Hdr, P, Cardinal(Stream.Position) - CdStart); // cd size
    PutD(Hdr, P, CdStart);         // cd offset
    PutW(Hdr, P, 0);               // comment len
    Stream.WriteBuffer(Hdr[0], 22);
  finally
    Stream.Free;
  end;
end;

end.
