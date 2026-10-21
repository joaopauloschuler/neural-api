unit neuralgif;
{
  neuralgif Free Pascal/Lazarus Library by Joao Paulo Schwarz Schuler
  Conscious Artificial Intelligence Project
  https://sourceforge.net/projects/cai/

  Animated GIF writer for (W,H,C) volumes: a fixed global 256-entry palette
  (6x7x6 RGB cube or 256 grays), LZW-coded frames and a NETSCAPE2.0 loop.
  Coded by Claude (AI).
}

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, neuralvolume;

type
  TNNetGIFPalette = (gpColorCube, gpGrayscale);

  // Writes frames one by one; Finish (or Destroy) writes the GIF trailer.
  // pLoopCount 0 loops forever; delays are in hundredths of a second.
  TNNetAnimatedGIFWriter = class
  private
    FStream: TStream;
    FOwnsStream: boolean;
    FWidth, FHeight: integer;
    FDelayCentiseconds: integer;
    FPalette: TNNetGIFPalette;
    FFrameCount: integer;
    FFinished: boolean;
    FIndices: array of byte;
    FHashKeys: array of integer;
    FHashCodes: array of word;
    FBitBuffer: cardinal;
    FBitCount: integer;
    FBlock: array[0..254] of byte;
    FBlockLen: integer;
    procedure Init(Stream: TStream; pOwnsStream: boolean;
      pWidth, pHeight, pDelayCentiseconds: integer; pPalette: TNNetGIFPalette;
      pLoopCount: integer);
    procedure WriteWord(Value: word);
    procedure PutByte(Value: byte);
    procedure EmitCode(Code, CodeSize: integer);
    procedure FlushCodes;
    procedure EncodeIndices;
  public
    constructor Create(const FileName: string; pWidth, pHeight: integer;
      pDelayCentiseconds: integer = 10; pPalette: TNNetGIFPalette = gpColorCube;
      pLoopCount: integer = 0);
    // The caller keeps ownership of Stream; writing starts at its position.
    constructor CreateOnStream(Stream: TStream; pWidth, pHeight: integer;
      pDelayCentiseconds: integer = 10; pPalette: TNNetGIFPalette = gpColorCube;
      pLoopCount: integer = 0);
    destructor Destroy; override;
    // Frame is (Width,Height,1) gray or (Width,Height,3+) RGB; values in
    // [ValueMin,ValueMax] map to 0..255 before palette quantization.
    procedure AddFrame(Frame: TNNetVolume; ValueMin: TNeuralFloat = 0;
      ValueMax: TNeuralFloat = 255);
    procedure Finish;
    property FrameCount: integer read FFrameCount;
  end;

// Nearest entry of the 6x7x6 cube palette (indices 0..251).
function GIFColorCubeIndex(R, G, B: byte): byte;
// RGB of a palette entry; cube indices 252..255 are black.
procedure GIFPaletteColor(Palette: TNNetGIFPalette; PaletteIndex: byte;
  out R, G, B: byte);

implementation

const
  csGIFClearCode = 256;
  csGIFEndCode = 257;
  csGIFMaxCode = 4096;
  csGIFHashSize = 8191;

function GIFColorCubeIndex(R, G, B: byte): byte;
begin
  Result := (((R * 5 + 127) div 255) * 7 + (G * 6 + 127) div 255) * 6 +
    (B * 5 + 127) div 255;
end;

procedure GIFPaletteColor(Palette: TNNetGIFPalette; PaletteIndex: byte;
  out R, G, B: byte);
begin
  if Palette = gpGrayscale then
  begin
    R := PaletteIndex; G := PaletteIndex; B := PaletteIndex;
  end
  else if PaletteIndex >= 252 then
  begin
    R := 0; G := 0; B := 0;
  end
  else
  begin
    R := ((PaletteIndex div 42) * 255 + 2) div 5;
    G := (((PaletteIndex div 6) mod 7) * 255 + 3) div 6;
    B := ((PaletteIndex mod 6) * 255 + 2) div 5;
  end;
end;

constructor TNNetAnimatedGIFWriter.Create(const FileName: string; pWidth,
  pHeight: integer; pDelayCentiseconds: integer; pPalette: TNNetGIFPalette;
  pLoopCount: integer);
begin
  inherited Create;
  Init(TFileStream.Create(FileName, fmCreate), true, pWidth, pHeight,
    pDelayCentiseconds, pPalette, pLoopCount);
end;

constructor TNNetAnimatedGIFWriter.CreateOnStream(Stream: TStream; pWidth,
  pHeight: integer; pDelayCentiseconds: integer; pPalette: TNNetGIFPalette;
  pLoopCount: integer);
begin
  inherited Create;
  Init(Stream, false, pWidth, pHeight, pDelayCentiseconds, pPalette, pLoopCount);
end;

procedure TNNetAnimatedGIFWriter.Init(Stream: TStream; pOwnsStream: boolean;
  pWidth, pHeight, pDelayCentiseconds: integer; pPalette: TNNetGIFPalette;
  pLoopCount: integer);
const
  cNetscape: AnsiString = 'NETSCAPE2.0';
  cSignature: AnsiString = 'GIF89a';
var
  PaletteIndex: integer;
  R, G, B: byte;
begin
  FStream := Stream;
  FOwnsStream := pOwnsStream;
  if (pWidth < 1) or (pHeight < 1) or (pWidth > 65535) or (pHeight > 65535) then
  begin
    FFinished := true;
    raise Exception.CreateFmt('GIF size %dx%d out of range', [pWidth, pHeight]);
  end;
  FWidth := pWidth;
  FHeight := pHeight;
  FDelayCentiseconds := pDelayCentiseconds;
  FPalette := pPalette;
  SetLength(FIndices, FWidth * FHeight);
  SetLength(FHashKeys, csGIFHashSize);
  SetLength(FHashCodes, csGIFHashSize);
  FStream.WriteBuffer(cSignature[1], Length(cSignature));
  WriteWord(FWidth);
  WriteWord(FHeight);
  // Global color table of 256 entries, 8 bits of color resolution.
  FStream.WriteByte($F7);
  FStream.WriteByte(0);
  FStream.WriteByte(0);
  for PaletteIndex := 0 to 255 do
  begin
    GIFPaletteColor(FPalette, PaletteIndex, R, G, B);
    FStream.WriteByte(R);
    FStream.WriteByte(G);
    FStream.WriteByte(B);
  end;
  FStream.WriteByte($21);
  FStream.WriteByte($FF);
  FStream.WriteByte(Length(cNetscape));
  FStream.WriteBuffer(cNetscape[1], Length(cNetscape));
  FStream.WriteByte(3);
  FStream.WriteByte(1);
  WriteWord(pLoopCount);
  FStream.WriteByte(0);
end;

destructor TNNetAnimatedGIFWriter.Destroy;
begin
  if Assigned(FStream) then
  begin
    Finish;
    if FOwnsStream then FStream.Free;
  end;
  inherited Destroy;
end;

procedure TNNetAnimatedGIFWriter.WriteWord(Value: word);
begin
  FStream.WriteByte(Value and $FF);
  FStream.WriteByte(Value shr 8);
end;

procedure TNNetAnimatedGIFWriter.PutByte(Value: byte);
begin
  FBlock[FBlockLen] := Value;
  Inc(FBlockLen);
  if FBlockLen = Length(FBlock) then
  begin
    FStream.WriteByte(FBlockLen);
    FStream.WriteBuffer(FBlock[0], FBlockLen);
    FBlockLen := 0;
  end;
end;

procedure TNNetAnimatedGIFWriter.EmitCode(Code, CodeSize: integer);
begin
  FBitBuffer := FBitBuffer or (cardinal(Code) shl FBitCount);
  Inc(FBitCount, CodeSize);
  while FBitCount >= 8 do
  begin
    PutByte(FBitBuffer and $FF);
    FBitBuffer := FBitBuffer shr 8;
    Dec(FBitCount, 8);
  end;
end;

procedure TNNetAnimatedGIFWriter.FlushCodes;
begin
  if FBitCount > 0 then PutByte(FBitBuffer and $FF);
  FBitBuffer := 0;
  FBitCount := 0;
  if FBlockLen > 0 then
  begin
    FStream.WriteByte(FBlockLen);
    FStream.WriteBuffer(FBlock[0], FBlockLen);
    FBlockLen := 0;
  end;
  FStream.WriteByte(0);
end;

// Variable-width LZW (8-bit roots, codes up to 12 bits, deferred-free clear
// when the table is full); the code width grows on the same step the decoder's does.
procedure TNNetAnimatedGIFWriter.EncodeIndices;
var
  CodeSize, NextCode, Prefix, PixelIndex, Key, HashPos, HashStep: integer;
  MaxPixelPos: integer;
  Found: boolean;

  procedure EmitAndWiden(Code: integer);
  begin
    EmitCode(Code, CodeSize);
    if (NextCode >= (1 shl CodeSize)) and (CodeSize < 12) then Inc(CodeSize);
  end;

  procedure ResetTable(out pCodeSize, pNextCode: integer);
  begin
    FillDWord(FHashKeys[0], csGIFHashSize, $FFFFFFFF);
    pCodeSize := 9;
    pNextCode := csGIFEndCode + 1;
  end;

begin
  FStream.WriteByte(8);
  FBitBuffer := 0;
  FBitCount := 0;
  FBlockLen := 0;
  ResetTable(CodeSize, NextCode);
  EmitCode(csGIFClearCode, CodeSize);
  Prefix := FIndices[0];
  MaxPixelPos := High(FIndices);
  for PixelIndex := 1 to MaxPixelPos do
  begin
    Key := (integer(FIndices[PixelIndex]) shl 12) or Prefix;
    HashPos := (integer(FIndices[PixelIndex]) shl 4) xor Prefix;
    if HashPos = 0 then HashStep := 1 else HashStep := csGIFHashSize - HashPos;
    Found := false;
    while FHashKeys[HashPos] >= 0 do
    begin
      if FHashKeys[HashPos] = Key then
      begin
        Found := true;
        break;
      end;
      Dec(HashPos, HashStep);
      if HashPos < 0 then Inc(HashPos, csGIFHashSize);
    end;
    if Found then
    begin
      Prefix := FHashCodes[HashPos];
      continue;
    end;
    EmitAndWiden(Prefix);
    if NextCode < csGIFMaxCode then
    begin
      FHashKeys[HashPos] := Key;
      FHashCodes[HashPos] := NextCode;
      Inc(NextCode);
    end
    else
    begin
      EmitCode(csGIFClearCode, CodeSize);
      ResetTable(CodeSize, NextCode);
    end;
    Prefix := FIndices[PixelIndex];
  end;
  EmitAndWiden(Prefix);
  EmitCode(csGIFEndCode, CodeSize);
  FlushCodes;
end;

procedure TNNetAnimatedGIFWriter.AddFrame(Frame: TNNetVolume;
  ValueMin: TNeuralFloat; ValueMax: TNeuralFloat);
var
  X, Y, RawPos, PixelPos, Depth, MaxX, MaxY: integer;
  Scale: TNeuralFloat;
  R, G, B: byte;

  function ToByte(Value: TNeuralFloat): byte;
  begin
    Value := (Value - ValueMin) * Scale;
    if Value >= 255 then Result := 255
    else if Value > 0 then Result := Round(Value)
    else Result := 0;
  end;

begin
  if FFinished then raise Exception.Create('AddFrame after Finish');
  if (Frame.SizeX <> FWidth) or (Frame.SizeY <> FHeight) then
    raise Exception.CreateFmt('GIF frame is %dx%d, expected %dx%d',
      [Frame.SizeX, Frame.SizeY, FWidth, FHeight]);
  if ValueMax > ValueMin then Scale := 255 / (ValueMax - ValueMin) else Scale := 0;
  Depth := Frame.Depth;
  MaxX := FWidth - 1;
  MaxY := FHeight - 1;
  PixelPos := 0;
  for Y := 0 to MaxY do
  begin
    RawPos := Frame.GetRawPos(0, Y, 0);
    for X := 0 to MaxX do
    begin
      R := ToByte(Frame.FData[RawPos]);
      if Depth >= 3 then
      begin
        G := ToByte(Frame.FData[RawPos + 1]);
        B := ToByte(Frame.FData[RawPos + 2]);
      end
      else
      begin
        G := R;
        B := R;
      end;
      if FPalette = gpGrayscale
      then FIndices[PixelPos] := (R * 299 + G * 587 + B * 114 + 500) div 1000
      else FIndices[PixelPos] := GIFColorCubeIndex(R, G, B);
      Inc(PixelPos);
      Inc(RawPos, Depth);
    end;
  end;
  // Graphic control extension: disposal 1 (keep), delay, no transparency.
  FStream.WriteByte($21);
  FStream.WriteByte($F9);
  FStream.WriteByte(4);
  FStream.WriteByte($04);
  WriteWord(FDelayCentiseconds);
  FStream.WriteByte(0);
  FStream.WriteByte(0);
  // Image descriptor: full canvas, no local color table, not interlaced.
  FStream.WriteByte($2C);
  WriteWord(0);
  WriteWord(0);
  WriteWord(FWidth);
  WriteWord(FHeight);
  FStream.WriteByte(0);
  EncodeIndices;
  Inc(FFrameCount);
end;

procedure TNNetAnimatedGIFWriter.Finish;
begin
  if FFinished then exit;
  FStream.WriteByte($3B);
  FFinished := true;
end;

end.
