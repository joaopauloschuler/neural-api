unit TestNeuralGIF;
(*
Tests for TNNetAnimatedGIFWriter (neuralgif.pas): the block structure of a
multi-frame GIF is walked byte by byte, and fcl-image's TFPReaderGif (which
decodes the first frame) checks the LZW stream pixel by pixel.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, FPImage, FPReadGif,
  neuralvolume, neuralgif;

type
  TTestNeuralGIF = class(TTestCase)
  published
    // 3 RGB frames: header, palette, NETSCAPE2.0 loop, 3 delays, 3 image
    // descriptors, trailer; frame 0 decodes to the cube-quantized colors.
    procedure TestAnimatedGIFStructureAndFirstFrame;
    // A 120x90 noise frame overflows the 4096-code table several times; the
    // 256-gray palette makes the decode exact.
    procedure TestGrayNoiseFrameDecodesExactly;
  end;

implementation

function DecodeFirstFrame(Stream: TStream): TFPMemoryImage;
var
  Reader: TFPReaderGif;
begin
  Result := TFPMemoryImage.Create(1, 1);
  Reader := TFPReaderGif.Create;
  try
    Stream.Position := 0;
    Result.LoadFromStream(Stream, Reader);
  finally
    Reader.Free;
  end;
end;

procedure SkipSubBlocks(Stream: TStream);
var
  BlockLen: byte;
begin
  repeat
    BlockLen := Stream.ReadByte;
    Stream.Seek(BlockLen, soCurrent);
  until BlockLen = 0;
end;

procedure TTestNeuralGIF.TestAnimatedGIFStructureAndFirstFrame;
const
  ImgW = 7;
  ImgH = 5;
  FrameTotal = 3;
var
  Stream: TMemoryStream;
  Writer: TNNetAnimatedGIFWriter;
  Frame: TNNetVolume;
  Img: TFPMemoryImage;
  Signature: array[0..5] of AnsiChar;
  AppId: array[0..10] of AnsiChar;
  Introducer, ExtLabel, BlockLen: byte;
  FrameCount, DelayCount, X, Y, C, FramePos: integer;
  FoundLoop, FoundTrailer: boolean;
  R, G, B: byte;
  Expected: array[0..2] of byte;
begin
  Stream := TMemoryStream.Create;
  Frame := TNNetVolume.Create(ImgW, ImgH, 3);
  Img := nil;
  try
    Writer := TNNetAnimatedGIFWriter.CreateOnStream(Stream, ImgW, ImgH, 25);
    try
      for FramePos := 0 to FrameTotal - 1 do
      begin
        for Y := 0 to ImgH - 1 do
          for X := 0 to ImgW - 1 do
            for C := 0 to 2 do
              Frame[X, Y, C] := ((X * 40 + Y * 60 + C * 90 + FramePos * 70) mod 256)
                / 127.5 - 1;
        Writer.AddFrame(Frame, -1, 1);
      end;
      AssertEquals('frame count', FrameTotal, Writer.FrameCount);
    finally
      Writer.Free;
    end;

    Stream.Position := 0;
    Stream.ReadBuffer(Signature, 6);
    AssertEquals('signature', 'GIF89a', string(Signature));
    AssertEquals('width', ImgW, Stream.ReadWord);
    AssertEquals('height', ImgH, Stream.ReadWord);
    AssertEquals('global 256-color table', $F7, Stream.ReadByte);
    Stream.Seek(2 + 768, soCurrent);
    FrameCount := 0;
    DelayCount := 0;
    FoundLoop := false;
    FoundTrailer := false;
    while (Stream.Position < Stream.Size) and not FoundTrailer do
    begin
      Introducer := Stream.ReadByte;
      case Introducer of
        $21:
          begin
            ExtLabel := Stream.ReadByte;
            if ExtLabel = $FF then
            begin
              AssertEquals('app block length', 11, Stream.ReadByte);
              Stream.ReadBuffer(AppId, 11);
              AssertEquals('app id', 'NETSCAPE2.0', string(AppId));
              BlockLen := Stream.ReadByte;
              AssertEquals('loop sub-block length', 3, BlockLen);
              AssertEquals('loop sub-block id', 1, Stream.ReadByte);
              AssertEquals('loop forever', 0, Stream.ReadWord);
              AssertEquals('loop terminator', 0, Stream.ReadByte);
              FoundLoop := true;
            end
            else if ExtLabel = $F9 then
            begin
              AssertEquals('GCE length', 4, Stream.ReadByte);
              Stream.ReadByte;
              AssertEquals('delay', 25, Stream.ReadWord);
              Stream.ReadByte;
              AssertEquals('GCE terminator', 0, Stream.ReadByte);
              Inc(DelayCount);
            end
            else
              SkipSubBlocks(Stream);
          end;
        $2C:
          begin
            AssertEquals('left', 0, Stream.ReadWord);
            AssertEquals('top', 0, Stream.ReadWord);
            AssertEquals('frame width', ImgW, Stream.ReadWord);
            AssertEquals('frame height', ImgH, Stream.ReadWord);
            AssertEquals('no local table', 0, Stream.ReadByte);
            AssertEquals('LZW min code size', 8, Stream.ReadByte);
            SkipSubBlocks(Stream);
            Inc(FrameCount);
          end;
        $3B: FoundTrailer := true;
      else
        Fail('unexpected block introducer ' + IntToStr(Introducer));
      end;
    end;
    AssertTrue('NETSCAPE2.0 loop block', FoundLoop);
    AssertTrue('trailer', FoundTrailer);
    AssertEquals('trailer is the last byte', Stream.Size, Stream.Position);
    AssertEquals('image descriptors', FrameTotal, FrameCount);
    AssertEquals('graphic control blocks', FrameTotal, DelayCount);

    Img := DecodeFirstFrame(Stream);
    AssertEquals('decoded width', ImgW, Img.Width);
    AssertEquals('decoded height', ImgH, Img.Height);
    for Y := 0 to ImgH - 1 do
      for X := 0 to ImgW - 1 do
      begin
        for C := 0 to 2 do
          Expected[C] := (X * 40 + Y * 60 + C * 90) mod 256;
        GIFPaletteColor(gpColorCube,
          GIFColorCubeIndex(Expected[0], Expected[1], Expected[2]), R, G, B);
        AssertEquals('red ' + IntToStr(X) + ',' + IntToStr(Y), R,
          Img.Colors[X, Y].Red shr 8);
        AssertEquals('green ' + IntToStr(X) + ',' + IntToStr(Y), G,
          Img.Colors[X, Y].Green shr 8);
        AssertEquals('blue ' + IntToStr(X) + ',' + IntToStr(Y), B,
          Img.Colors[X, Y].Blue shr 8);
      end;
  finally
    Img.Free;
    Frame.Free;
    Stream.Free;
  end;
end;

procedure TTestNeuralGIF.TestGrayNoiseFrameDecodesExactly;
const
  ImgW = 120;
  ImgH = 90;
var
  Stream: TMemoryStream;
  Writer: TNNetAnimatedGIFWriter;
  Frame: TNNetVolume;
  Img: TFPMemoryImage;
  X, Y: integer;
  Seed: cardinal;
begin
  Stream := TMemoryStream.Create;
  Frame := TNNetVolume.Create(ImgW, ImgH, 1);
  Img := nil;
  try
    Seed := 12345;
    for Y := 0 to ImgH - 1 do
      for X := 0 to ImgW - 1 do
      begin
        Seed := (Seed * 1103515245 + 12345) and $7FFFFFFF;
        // Half noise, half a run-heavy gradient, so long codes appear too.
        if Y < ImgH div 2 then Frame[X, Y, 0] := (Seed shr 16) and 255
        else Frame[X, Y, 0] := X div 8;
      end;
    Writer := TNNetAnimatedGIFWriter.CreateOnStream(Stream, ImgW, ImgH, 10,
      gpGrayscale);
    try
      Writer.AddFrame(Frame);
    finally
      Writer.Free;
    end;
    Img := DecodeFirstFrame(Stream);
    for Y := 0 to ImgH - 1 do
      for X := 0 to ImgW - 1 do
        AssertEquals('gray ' + IntToStr(X) + ',' + IntToStr(Y),
          Round(Frame[X, Y, 0]), Img.Colors[X, Y].Red shr 8);
  finally
    Img.Free;
    Frame.Free;
    Stream.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralGIF);
end.
