unit TestNeuralAudio;
(*
Tests for neuralaudio.pas WAV I/O.

SaveVolumeToWav16 is the inverse of LoadWav16ToVolume: a synthetic mono
waveform (sine + a touch of deterministic noise) is written to a temp 16-bit
PCM WAV, read back, and asserted to round-trip to within 1 LSB
(1/32767 ~ 3.06e-5). The header is also independently checked to be a
canonical mono 16-bit PCM RIFF/WAVE at the requested sample rate.

The temp file is written under GetTempDir (never the repo tree) and deleted
in the test teardown.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry,
  neuralvolume, neuralaudio;

type
  TTestNeuralAudio = class(TTestCase)
  private
    FTmpWav: string;
  protected
    procedure TearDown; override;
  published
    // Write a synthetic waveform, read it back, assert max abs diff <= 1 LSB.
    procedure TestWavRoundTrip;
    // The emitted header is a canonical mono 16-bit PCM WAV at the given rate.
    procedure TestWavHeader;
    // An empty (zero-length) waveform writes a valid header and reads back 0.
    procedure TestWavEmpty;
  end;

implementation

const
  csLSB = 1.0 / 32767.0;

procedure TTestNeuralAudio.TearDown;
begin
  if (FTmpWav <> '') and FileExists(FTmpWav) then
    DeleteFile(FTmpWav);
  FTmpWav := '';
  inherited TearDown;
end;

procedure TTestNeuralAudio.TestWavRoundTrip;
var
  Src, Dst: TNNetVolume;
  N, i, Sr: integer;
  t, v, MaxDiff, d: double;
begin
  FTmpWav := GetTempDir(false) + 'cai_audio_roundtrip_' +
    IntToStr(Random(1000000)) + '.wav';
  N := 4000;
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := TNNetVolume.Create(1, 1, 1);
  try
    RandSeed := 20260615;
    for i := 0 to N - 1 do
    begin
      t := i / 16000.0;
      // sine at 220 Hz, 0.6 amplitude, plus a little deterministic noise,
      // kept well inside [-1, 1] so no clamping happens.
      v := 0.6 * Sin(2.0 * Pi * 220.0 * t) + 0.05 * (Random - 0.5);
      Src.FData[i] := v;
    end;
    SaveVolumeToWav16(Src, FTmpWav, 16000);
    Sr := LoadWav16ToVolume(FTmpWav, Dst);
    AssertEquals('sample rate round-trips', 16000, Sr);
    AssertEquals('frame count round-trips', N, Dst.Size);
    MaxDiff := 0;
    for i := 0 to N - 1 do
    begin
      d := Abs(Src.FData[i] - Dst.FData[i]);
      if d > MaxDiff then MaxDiff := d;
    end;
    AssertTrue('max abs diff <= 1 LSB (' + FloatToStr(MaxDiff) + ' vs ' +
      FloatToStr(csLSB) + ')', MaxDiff <= csLSB + 1e-9);
  finally
    Src.Free;
    Dst.Free;
  end;
end;

procedure TTestNeuralAudio.TestWavHeader;
var
  Src: TNNetVolume;
  N, i: integer;
  FS: TFileStream;
  Tag: array[0..3] of AnsiChar;
  ChunkSize, SampleRate, ByteRate, DataBytes: longword;
  AudioFormat, NumChannels, BlockAlign, BitsPerSample: word;
begin
  FTmpWav := GetTempDir(false) + 'cai_audio_header_' +
    IntToStr(Random(1000000)) + '.wav';
  N := 100;
  Src := TNNetVolume.Create(N, 1, 1);
  try
    for i := 0 to N - 1 do Src.FData[i] := Sin(i * 0.1);
    SaveVolumeToWav16(Src, FTmpWav, 22050);
    FS := TFileStream.Create(FTmpWav, fmOpenRead or fmShareDenyWrite);
    try
      FS.ReadBuffer(Tag, 4); AssertEquals('RIFF tag', 'RIFF', string(Tag));
      FS.ReadBuffer(ChunkSize, 4);
      AssertEquals('RIFF size', longword(4 + 8 + 16 + 8 + N * 2), ChunkSize);
      FS.ReadBuffer(Tag, 4); AssertEquals('WAVE tag', 'WAVE', string(Tag));
      FS.ReadBuffer(Tag, 4); AssertEquals('fmt tag', 'fmt ', string(Tag));
      FS.ReadBuffer(ChunkSize, 4); AssertEquals('fmt size', longword(16), ChunkSize);
      FS.ReadBuffer(AudioFormat, 2); AssertEquals('PCM format', 1, AudioFormat);
      FS.ReadBuffer(NumChannels, 2); AssertEquals('mono', 1, NumChannels);
      FS.ReadBuffer(SampleRate, 4); AssertEquals('rate', longword(22050), SampleRate);
      FS.ReadBuffer(ByteRate, 4); AssertEquals('byte rate', longword(22050 * 2), ByteRate);
      FS.ReadBuffer(BlockAlign, 2); AssertEquals('block align', 2, BlockAlign);
      FS.ReadBuffer(BitsPerSample, 2); AssertEquals('16-bit', 16, BitsPerSample);
      FS.ReadBuffer(Tag, 4); AssertEquals('data tag', 'data', string(Tag));
      FS.ReadBuffer(DataBytes, 4);
      AssertEquals('data bytes', longword(N * 2), DataBytes);
    finally
      FS.Free;
    end;
  finally
    Src.Free;
  end;
end;

procedure TTestNeuralAudio.TestWavEmpty;
var
  Src, Dst: TNNetVolume;
begin
  FTmpWav := GetTempDir(false) + 'cai_audio_empty_' +
    IntToStr(Random(1000000)) + '.wav';
  Src := TNNetVolume.Create(0, 1, 1);
  Dst := TNNetVolume.Create(1, 1, 1);
  try
    SaveVolumeToWav16(Src, FTmpWav, 16000);
    AssertEquals('empty rate', 16000, LoadWav16ToVolume(FTmpWav, Dst));
    AssertEquals('empty frame count', 0, Dst.Size);
  finally
    Src.Free;
    Dst.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralAudio);
end.
