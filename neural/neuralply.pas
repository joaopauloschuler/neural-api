unit neuralply;
{
  neuralply Free Pascal/Lazarus Library by Joao Paulo Schwarz Schuler
  Conscious Artificial Intelligence Project
  https://sourceforge.net/projects/cai/

  PLY (Stanford polygon file) reader and writer: ASCII header, ascii /
  binary_little_endian / binary_big_endian bodies, typed scalar and list
  properties. Rows stream one at a time so a large point cloud is never held
  twice in memory. Coded by Claude (AI).
}

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume;

type
  TNNetPLYFormat = (plyAscii, plyBinaryLittleEndian, plyBinaryBigEndian);
  TNNetPLYValueType = (plyInt8, plyUInt8, plyInt16, plyUInt16, plyInt32,
    plyUInt32, plyFloat32, plyFloat64);

  TNNetPLYProperty = record
    Name: string;
    ValueType: TNNetPLYValueType;
    IsList: boolean;
    ListCountType: TNNetPLYValueType;
  end;

  TNNetPLYElement = record
    Name: string;
    Count: integer;
    Properties: array of TNNetPLYProperty;
  end;
  TNNetPLYElementArray = array of TNNetPLYElement;

  // Parses the header on Create; ReadRow then returns rows in file order. ReadRow
  // drops list items (the column holds the length); ReadFlatRow keeps them.
  TNNetPLYReader = class
  private
    FStream: TStream;
    FOwnsStream: boolean;
    FFormat: TNNetPLYFormat;
    FElements: TNNetPLYElementArray;
    FComments: TStringList;
    FBuffer: array[0..65535] of byte;
    FBufferPos, FBufferLen: integer;
    FEndOfStream: boolean;
    FCurrentElement: integer;
    FRowsReadInElement: integer;
    function ReadNextByte(out Value: byte): boolean;
    procedure ReadBytes(var Dest; ByteCount: integer);
    function ReadHeaderLine: string;
    function ReadToken: string;
    function ReadValue(ValueType: TNNetPLYValueType): Double;
    procedure ParseHeader;
    procedure SkipFinishedElements;
  public
    constructor Create(const FileName: string);
    constructor CreateOnStream(Stream: TStream; pOwnsStream: boolean = false);
    destructor Destroy; override;
    function FindElement(const ElementName: string): integer;
    function PropertyIndex(ElementIndex: integer; const PropertyName: string): integer;
    // Reads and drops every remaining row before element ElementIndex.
    procedure SkipToElement(ElementIndex: integer);
    // Reads the next row of CurrentElement; Row needs one slot per property.
    procedure ReadRow(var Row: array of TNeuralFloat);
    // Reads the next row in the WriteRow flat layout (a list is its length, then its
    // items); grows Row when short and returns the slot count.
    function ReadFlatRow(var Row: TNeuralFloatDynArr): integer;
    property Format: TNNetPLYFormat read FFormat;
    property Elements: TNNetPLYElementArray read FElements;
    property Comments: TStringList read FComments;
    property CurrentElement: integer read FCurrentElement;
  end;

  // Declare elements and properties, call WriteHeader, then WriteRow Count times
  // per element in declaration order; Destroy flushes.
  TNNetPLYWriter = class
  private
    FStream: TStream;
    FOwnsStream: boolean;
    FFormat: TNNetPLYFormat;
    FElements: TNNetPLYElementArray;
    FComments: TStringList;
    FBuffer: array[0..65535] of byte;
    FBufferLen: integer;
    FHeaderWritten: boolean;
    FCurrentElement: integer;
    FRowsWrittenInElement: integer;
    FFormatSettings: TFormatSettings;
    procedure WriteBytes(const Source; ByteCount: integer);
    procedure WriteText(const Text: string);
    procedure WriteValue(Value: TNeuralFloat; ValueType: TNNetPLYValueType);
    procedure Flush;
  public
    constructor Create(const FileName: string; pFormat: TNNetPLYFormat);
    constructor CreateOnStream(Stream: TStream; pFormat: TNNetPLYFormat;
      pOwnsStream: boolean = false);
    destructor Destroy; override;
    procedure AddComment(const Comment: string);
    procedure AddElement(const ElementName: string; Count: integer);
    procedure AddProperty(const PropertyName: string; ValueType: TNNetPLYValueType);
    procedure AddListProperty(const PropertyName: string;
      CountType, ValueType: TNNetPLYValueType);
    procedure WriteHeader;
    // Row is flat: one slot per scalar; a list takes its length, then its items.
    procedure WriteRow(const Row: array of TNeuralFloat);
    property Elements: TNNetPLYElementArray read FElements;
  end;

  function PLYValueTypeByteSize(ValueType: TNNetPLYValueType): integer;
  function PLYValueTypeName(ValueType: TNNetPLYValueType): string;
  // Accepts both the classic (uchar, float) and sized (uint8, float32) names.
  function PLYValueTypeFromName(const TypeName: string; out ValueType: TNNetPLYValueType): boolean;

implementation

const
  PLYValueTypeSizes: array[TNNetPLYValueType] of integer = (1, 1, 2, 2, 4, 4, 4, 8);
  PLYValueTypeNames: array[TNNetPLYValueType] of string =
    ('char', 'uchar', 'short', 'ushort', 'int', 'uint', 'float', 'double');
  PLYValueTypeSizedNames: array[TNNetPLYValueType] of string =
    ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64');
  PLYFormatNames: array[TNNetPLYFormat] of string =
    ('ascii', 'binary_little_endian', 'binary_big_endian');

function PLYValueTypeByteSize(ValueType: TNNetPLYValueType): integer;
begin
  Result := PLYValueTypeSizes[ValueType];
end;

function PLYValueTypeName(ValueType: TNNetPLYValueType): string;
begin
  Result := PLYValueTypeNames[ValueType];
end;

function PLYValueTypeFromName(const TypeName: string; out ValueType: TNNetPLYValueType): boolean;
var
  CandidateType: TNNetPLYValueType;
begin
  Result := false;
  ValueType := plyFloat32;
  for CandidateType := Low(TNNetPLYValueType) to High(TNNetPLYValueType) do
    if (TypeName = PLYValueTypeNames[CandidateType]) or
      (TypeName = PLYValueTypeSizedNames[CandidateType]) then
    begin
      ValueType := CandidateType;
      Exit(true);
    end;
end;

function HostIsLittleEndian: boolean; inline;
begin
  {$IFDEF ENDIAN_LITTLE}
  Result := true;
  {$ELSE}
  Result := false;
  {$ENDIF}
end;

procedure ReverseBytes(var Bytes: array of byte; ByteCount: integer);
var
  Lo, Hi: integer;
  Swap: byte;
begin
  Lo := 0;
  Hi := ByteCount - 1;
  while Lo < Hi do
  begin
    Swap := Bytes[Lo];
    Bytes[Lo] := Bytes[Hi];
    Bytes[Hi] := Swap;
    Inc(Lo);
    Dec(Hi);
  end;
end;

{ TNNetPLYReader }

constructor TNNetPLYReader.Create(const FileName: string);
begin
  CreateOnStream(TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite), true);
end;

constructor TNNetPLYReader.CreateOnStream(Stream: TStream; pOwnsStream: boolean);
begin
  inherited Create;
  FStream := Stream;
  FOwnsStream := pOwnsStream;
  FComments := TStringList.Create;
  FBufferPos := 0;
  FBufferLen := 0;
  FEndOfStream := false;
  ParseHeader;
  FCurrentElement := 0;
  FRowsReadInElement := 0;
  SkipFinishedElements;
end;

destructor TNNetPLYReader.Destroy;
begin
  FComments.Free;
  if FOwnsStream then FStream.Free;
  inherited Destroy;
end;

function TNNetPLYReader.ReadNextByte(out Value: byte): boolean;
begin
  if FBufferPos >= FBufferLen then
  begin
    FBufferLen := FStream.Read(FBuffer[0], SizeOf(FBuffer));
    FBufferPos := 0;
    if FBufferLen <= 0 then
    begin
      FBufferLen := 0;
      FEndOfStream := true;
      Value := 0;
      Exit(false);
    end;
  end;
  Value := FBuffer[FBufferPos];
  Inc(FBufferPos);
  Result := true;
end;

procedure TNNetPLYReader.ReadBytes(var Dest; ByteCount: integer);
var
  DestBytes: PByte;
  ChunkSize: integer;
begin
  DestBytes := @Dest;
  while ByteCount > 0 do
  begin
    if FBufferPos >= FBufferLen then
    begin
      FBufferLen := FStream.Read(FBuffer[0], SizeOf(FBuffer));
      FBufferPos := 0;
      if FBufferLen <= 0 then
      begin
        FBufferLen := 0;
        raise EReadError.Create('PLY: unexpected end of file');
      end;
    end;
    ChunkSize := FBufferLen - FBufferPos;
    if ChunkSize > ByteCount then ChunkSize := ByteCount;
    Move(FBuffer[FBufferPos], DestBytes^, ChunkSize);
    Inc(FBufferPos, ChunkSize);
    Inc(DestBytes, ChunkSize);
    Dec(ByteCount, ChunkSize);
  end;
end;

function TNNetPLYReader.ReadHeaderLine: string;
var
  NextByte: byte;
begin
  Result := '';
  while ReadNextByte(NextByte) do
  begin
    if NextByte = 10 then Break;
    if NextByte <> 13 then Result := Result + Chr(NextByte);
  end;
  Result := Trim(Result);
end;

function TNNetPLYReader.ReadToken: string;
var
  NextByte: byte;
begin
  Result := '';
  repeat
    if not ReadNextByte(NextByte) then
      raise EReadError.Create('PLY: unexpected end of ascii body');
  until NextByte > 32;
  repeat
    Result := Result + Chr(NextByte);
  until (not ReadNextByte(NextByte)) or (NextByte <= 32);
end;

function TNNetPLYReader.ReadValue(ValueType: TNNetPLYValueType): Double;
var
  Bytes: array[0..7] of byte;
  ByteCount, ParseErrorPos: integer;
  Token: string;
begin
  if FFormat = plyAscii then
  begin
    Token := ReadToken;
    Val(Token, Result, ParseErrorPos);
    if ParseErrorPos <> 0 then
      raise EConvertError.CreateFmt('PLY: bad ascii value "%s"', [Token]);
    Exit;
  end;
  ByteCount := PLYValueTypeSizes[ValueType];
  ReadBytes(Bytes[0], ByteCount);
  if (FFormat = plyBinaryLittleEndian) <> HostIsLittleEndian then
    ReverseBytes(Bytes, ByteCount);
  case ValueType of
    plyInt8:    Result := PShortInt(@Bytes[0])^;
    plyUInt8:   Result := Bytes[0];
    plyInt16:   Result := PSmallInt(@Bytes[0])^;
    plyUInt16:  Result := PWord(@Bytes[0])^;
    plyInt32:   Result := PLongInt(@Bytes[0])^;
    plyUInt32:  Result := PLongWord(@Bytes[0])^;
    plyFloat32: Result := PSingle(@Bytes[0])^;
  else
    Result := PDouble(@Bytes[0])^;
  end;
end;

procedure TNNetPLYReader.ParseHeader;
var
  Line, FormatName: string;
  Words: TStringList;
  ElementPos, PropertyPos: integer;
  ParsedFormat: TNNetPLYFormat;
  FormatFound: boolean;
begin
  if ReadHeaderLine <> 'ply' then
    raise EReadError.Create('PLY: missing "ply" magic line');
  FormatFound := false;
  ElementPos := -1;
  Words := TStringList.Create;
  try
    Words.Delimiter := ' ';
    Words.StrictDelimiter := false;
    repeat
      Line := ReadHeaderLine;
      if FEndOfStream and (Line = '') then
        raise EReadError.Create('PLY: header has no end_header');
      if Line = 'end_header' then Break;
      if (Copy(Line, 1, 8) = 'comment ') or (Line = 'comment') then
      begin
        FComments.Add(Trim(Copy(Line, 9, Length(Line))));
        Continue;
      end;
      Words.DelimitedText := Line;
      if Words.Count = 0 then Continue;
      if Words[0] = 'format' then
      begin
        if Words.Count < 2 then raise EReadError.Create('PLY: bad format line');
        FormatName := Words[1];
        for ParsedFormat := Low(TNNetPLYFormat) to High(TNNetPLYFormat) do
          if FormatName = PLYFormatNames[ParsedFormat] then
          begin
            FFormat := ParsedFormat;
            FormatFound := true;
          end;
        if not FormatFound then
          raise EReadError.CreateFmt('PLY: unknown format "%s"', [FormatName]);
      end
      else if Words[0] = 'element' then
      begin
        if Words.Count < 3 then raise EReadError.Create('PLY: bad element line');
        ElementPos := Length(FElements);
        SetLength(FElements, ElementPos + 1);
        FElements[ElementPos].Name := Words[1];
        FElements[ElementPos].Count := StrToInt(Words[2]);
      end
      else if Words[0] = 'property' then
      begin
        if ElementPos < 0 then raise EReadError.Create('PLY: property before element');
        PropertyPos := Length(FElements[ElementPos].Properties);
        SetLength(FElements[ElementPos].Properties, PropertyPos + 1);
        with FElements[ElementPos].Properties[PropertyPos] do
        begin
          IsList := (Words.Count >= 5) and (Words[1] = 'list');
          if IsList then
          begin
            if not PLYValueTypeFromName(Words[2], ListCountType) or
              not PLYValueTypeFromName(Words[3], ValueType) then
              raise EReadError.CreateFmt('PLY: bad list property "%s"', [Line]);
            Name := Words[4];
          end
          else
          begin
            if (Words.Count < 3) or not PLYValueTypeFromName(Words[1], ValueType) then
              raise EReadError.CreateFmt('PLY: bad property "%s"', [Line]);
            ListCountType := plyUInt8;
            Name := Words[2];
          end;
        end;
      end;
      // obj_info and unknown keywords carry no layout and are ignored.
    until false;
  finally
    Words.Free;
  end;
  if not FormatFound then raise EReadError.Create('PLY: header has no format line');
end;

procedure TNNetPLYReader.SkipFinishedElements;
begin
  while (FCurrentElement < Length(FElements)) and
    (FRowsReadInElement >= FElements[FCurrentElement].Count) do
  begin
    Inc(FCurrentElement);
    FRowsReadInElement := 0;
  end;
end;

function TNNetPLYReader.FindElement(const ElementName: string): integer;
var
  ElementPos, MaxElementPos: integer;
begin
  MaxElementPos := High(FElements);
  for ElementPos := 0 to MaxElementPos do
    if FElements[ElementPos].Name = ElementName then Exit(ElementPos);
  Result := -1;
end;

function TNNetPLYReader.PropertyIndex(ElementIndex: integer; const PropertyName: string): integer;
var
  PropertyPos, MaxPropertyPos: integer;
begin
  MaxPropertyPos := High(FElements[ElementIndex].Properties);
  for PropertyPos := 0 to MaxPropertyPos do
    if FElements[ElementIndex].Properties[PropertyPos].Name = PropertyName then
      Exit(PropertyPos);
  Result := -1;
end;

procedure TNNetPLYReader.SkipToElement(ElementIndex: integer);
var
  DroppedRow: TNeuralFloatDynArr;
begin
  DroppedRow := nil;
  if ElementIndex < FCurrentElement then
    raise EReadError.Create('PLY: cannot seek back to an earlier element');
  while FCurrentElement < ElementIndex do
  begin
    if Length(DroppedRow) < Length(FElements[FCurrentElement].Properties) then
      SetLength(DroppedRow, Length(FElements[FCurrentElement].Properties));
    ReadRow(DroppedRow);
  end;
end;

procedure TNNetPLYReader.ReadRow(var Row: array of TNeuralFloat);
var
  PropertyPos, MaxPropertyPos, ListLength, MaxListPos, ListPos: integer;
begin
  if FCurrentElement >= Length(FElements) then
    raise EReadError.Create('PLY: no rows left');
  MaxPropertyPos := High(FElements[FCurrentElement].Properties);
  if Length(Row) <= MaxPropertyPos then
    raise ERangeError.Create('PLY: row buffer shorter than the property count');
  for PropertyPos := 0 to MaxPropertyPos do
    with FElements[FCurrentElement].Properties[PropertyPos] do
    begin
      if IsList then
      begin
        ListLength := Round(ReadValue(ListCountType));
        MaxListPos := ListLength - 1;
        for ListPos := 0 to MaxListPos do ReadValue(ValueType);
        Row[PropertyPos] := ListLength;
      end
      else
        Row[PropertyPos] := ReadValue(ValueType);
    end;
  Inc(FRowsReadInElement);
  SkipFinishedElements;
end;

function TNNetPLYReader.ReadFlatRow(var Row: TNeuralFloatDynArr): integer;
var
  PropertyPos, MaxPropertyPos, ListLength, MaxListPos, ListPos: integer;

  procedure EnsureSlots(SlotCount: integer);
  begin
    if Length(Row) < SlotCount then SetLength(Row, 2 * SlotCount);
  end;

begin
  if FCurrentElement >= Length(FElements) then
    raise EReadError.Create('PLY: no rows left');
  MaxPropertyPos := High(FElements[FCurrentElement].Properties);
  Result := 0;
  for PropertyPos := 0 to MaxPropertyPos do
    with FElements[FCurrentElement].Properties[PropertyPos] do
    begin
      if IsList then
      begin
        ListLength := Round(ReadValue(ListCountType));
        EnsureSlots(Result + 1 + ListLength);
        Row[Result] := ListLength;
        Inc(Result);
        MaxListPos := ListLength - 1;
        for ListPos := 0 to MaxListPos do
          Row[Result + ListPos] := ReadValue(ValueType);
        Inc(Result, ListLength);
      end
      else
      begin
        EnsureSlots(Result + 1);
        Row[Result] := ReadValue(ValueType);
        Inc(Result);
      end;
    end;
  Inc(FRowsReadInElement);
  SkipFinishedElements;
end;

{ TNNetPLYWriter }

constructor TNNetPLYWriter.Create(const FileName: string; pFormat: TNNetPLYFormat);
begin
  CreateOnStream(TFileStream.Create(FileName, fmCreate), pFormat, true);
end;

constructor TNNetPLYWriter.CreateOnStream(Stream: TStream; pFormat: TNNetPLYFormat;
  pOwnsStream: boolean);
begin
  inherited Create;
  FStream := Stream;
  FOwnsStream := pOwnsStream;
  FFormat := pFormat;
  FComments := TStringList.Create;
  FBufferLen := 0;
  FHeaderWritten := false;
  FFormatSettings := DefaultFormatSettings;
  FFormatSettings.DecimalSeparator := '.';
  FFormatSettings.ThousandSeparator := #0;
end;

destructor TNNetPLYWriter.Destroy;
begin
  Flush;
  FComments.Free;
  if FOwnsStream then FStream.Free;
  inherited Destroy;
end;

procedure TNNetPLYWriter.Flush;
begin
  if FBufferLen > 0 then FStream.WriteBuffer(FBuffer[0], FBufferLen);
  FBufferLen := 0;
end;

procedure TNNetPLYWriter.WriteBytes(const Source; ByteCount: integer);
var
  SourceBytes: PByte;
  ChunkSize: integer;
begin
  SourceBytes := @Source;
  while ByteCount > 0 do
  begin
    if FBufferLen >= SizeOf(FBuffer) then Flush;
    ChunkSize := SizeOf(FBuffer) - FBufferLen;
    if ChunkSize > ByteCount then ChunkSize := ByteCount;
    Move(SourceBytes^, FBuffer[FBufferLen], ChunkSize);
    Inc(FBufferLen, ChunkSize);
    Inc(SourceBytes, ChunkSize);
    Dec(ByteCount, ChunkSize);
  end;
end;

procedure TNNetPLYWriter.WriteText(const Text: string);
begin
  if Length(Text) > 0 then WriteBytes(Text[1], Length(Text));
end;

procedure TNNetPLYWriter.AddComment(const Comment: string);
begin
  FComments.Add(Comment);
end;

procedure TNNetPLYWriter.AddElement(const ElementName: string; Count: integer);
var
  ElementPos: integer;
begin
  if FHeaderWritten then raise EWriteError.Create('PLY: header already written');
  ElementPos := Length(FElements);
  SetLength(FElements, ElementPos + 1);
  FElements[ElementPos].Name := ElementName;
  FElements[ElementPos].Count := Count;
end;

procedure TNNetPLYWriter.AddProperty(const PropertyName: string; ValueType: TNNetPLYValueType);
var
  ElementPos, PropertyPos: integer;
begin
  if FHeaderWritten then raise EWriteError.Create('PLY: header already written');
  ElementPos := High(FElements);
  if ElementPos < 0 then raise EWriteError.Create('PLY: property before element');
  PropertyPos := Length(FElements[ElementPos].Properties);
  SetLength(FElements[ElementPos].Properties, PropertyPos + 1);
  FElements[ElementPos].Properties[PropertyPos].Name := PropertyName;
  FElements[ElementPos].Properties[PropertyPos].ValueType := ValueType;
  FElements[ElementPos].Properties[PropertyPos].IsList := false;
  FElements[ElementPos].Properties[PropertyPos].ListCountType := plyUInt8;
end;

procedure TNNetPLYWriter.AddListProperty(const PropertyName: string;
  CountType, ValueType: TNNetPLYValueType);
var
  PropertyPos: integer;
begin
  AddProperty(PropertyName, ValueType);
  PropertyPos := High(FElements[High(FElements)].Properties);
  FElements[High(FElements)].Properties[PropertyPos].IsList := true;
  FElements[High(FElements)].Properties[PropertyPos].ListCountType := CountType;
end;

procedure TNNetPLYWriter.WriteHeader;
var
  CommentPos, MaxCommentPos, ElementPos, MaxElementPos, PropertyPos, MaxPropertyPos: integer;
begin
  if FHeaderWritten then raise EWriteError.Create('PLY: header already written');
  WriteText('ply'#10'format ' + PLYFormatNames[FFormat] + ' 1.0'#10);
  MaxCommentPos := FComments.Count - 1;
  for CommentPos := 0 to MaxCommentPos do
    WriteText('comment ' + FComments[CommentPos] + #10);
  MaxElementPos := High(FElements);
  for ElementPos := 0 to MaxElementPos do
  begin
    WriteText('element ' + FElements[ElementPos].Name + ' ' +
      IntToStr(FElements[ElementPos].Count) + #10);
    MaxPropertyPos := High(FElements[ElementPos].Properties);
    for PropertyPos := 0 to MaxPropertyPos do
      with FElements[ElementPos].Properties[PropertyPos] do
        if IsList then
          WriteText('property list ' + PLYValueTypeNames[ListCountType] + ' ' +
            PLYValueTypeNames[ValueType] + ' ' + Name + #10)
        else
          WriteText('property ' + PLYValueTypeNames[ValueType] + ' ' + Name + #10);
  end;
  WriteText('end_header'#10);
  FHeaderWritten := true;
  FCurrentElement := 0;
  FRowsWrittenInElement := 0;
  while (FCurrentElement < Length(FElements)) and (FElements[FCurrentElement].Count = 0) do
    Inc(FCurrentElement);
end;

procedure TNNetPLYWriter.WriteValue(Value: TNeuralFloat; ValueType: TNNetPLYValueType);
var
  Bytes: array[0..7] of byte;
  ByteCount: integer;
begin
  if FFormat = plyAscii then
  begin
    if ValueType in [plyFloat32, plyFloat64] then
      WriteText(FloatToStrF(Value, ffGeneral, 9, 0, FFormatSettings))
    else
      WriteText(IntToStr(Round(Value)));
    Exit;
  end;
  ByteCount := PLYValueTypeSizes[ValueType];
  case ValueType of
    plyInt8:    PShortInt(@Bytes[0])^ := Round(Value);
    plyUInt8:   Bytes[0] := Round(Value);
    plyInt16:   PSmallInt(@Bytes[0])^ := Round(Value);
    plyUInt16:  PWord(@Bytes[0])^ := Round(Value);
    plyInt32:   PLongInt(@Bytes[0])^ := Round(Value);
    plyUInt32:  PLongWord(@Bytes[0])^ := Round(Value);
    plyFloat32: PSingle(@Bytes[0])^ := Value;
  else
    PDouble(@Bytes[0])^ := Value;
  end;
  if (FFormat = plyBinaryLittleEndian) <> HostIsLittleEndian then
    ReverseBytes(Bytes, ByteCount);
  WriteBytes(Bytes[0], ByteCount);
end;

procedure TNNetPLYWriter.WriteRow(const Row: array of TNeuralFloat);
var
  PropertyPos, MaxPropertyPos, SlotPos, ListLength, MaxListPos, ListPos: integer;

  procedure WriteSlot(ValueType: TNNetPLYValueType);
  begin
    if SlotPos > High(Row) then
      raise ERangeError.Create('PLY: row shorter than its properties and lists');
    if (FFormat = plyAscii) and (SlotPos > 0) then WriteText(' ');
    WriteValue(Row[SlotPos], ValueType);
    Inc(SlotPos);
  end;

begin
  if not FHeaderWritten then raise EWriteError.Create('PLY: WriteHeader not called');
  if FCurrentElement >= Length(FElements) then
    raise EWriteError.Create('PLY: every declared row is already written');
  MaxPropertyPos := High(FElements[FCurrentElement].Properties);
  SlotPos := 0;
  for PropertyPos := 0 to MaxPropertyPos do
    with FElements[FCurrentElement].Properties[PropertyPos] do
    begin
      if IsList then
      begin
        if SlotPos > High(Row) then
          raise ERangeError.Create('PLY: row shorter than its properties and lists');
        ListLength := Round(Row[SlotPos]);
        WriteSlot(ListCountType);
        MaxListPos := ListLength - 1;
        for ListPos := 0 to MaxListPos do WriteSlot(ValueType);
      end
      else
        WriteSlot(ValueType);
    end;
  if FFormat = plyAscii then WriteText(#10);
  Inc(FRowsWrittenInElement);
  while (FCurrentElement < Length(FElements)) and
    (FRowsWrittenInElement >= FElements[FCurrentElement].Count) do
  begin
    Inc(FCurrentElement);
    FRowsWrittenInElement := 0;
  end;
end;

end.
