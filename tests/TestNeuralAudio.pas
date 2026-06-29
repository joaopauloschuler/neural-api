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
    // Forward STFT -> ISTFTOverlapAdd round-trips the interior near-exactly
    // under a perfect-reconstruction Hann window with 75% overlap.
    procedure TestISTFTRoundTrip;
    // Resampling 16 kHz -> 16 kHz is a bit-identical passthrough.
    procedure TestResampleIdentity;
    // Output length matches the round(N * target/source) ratio (up and down).
    procedure TestResampleLength;
    // A pure low-frequency sine survives 44.1k -> 16k and 8k -> 16k with its
    // fundamental (zero-crossing rate) preserved within tolerance.
    procedure TestResampleSinePreservesFrequency;
    // LoadWavResampledToVolume on an 8 kHz WAV returns a 16 kHz volume of the
    // right length, round-tripped through a temp WAV on disk.
    procedure TestLoadWavResampledRoundTrip;
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

// Forward real STFT mirroring neuralaudio's CosTab/SinTab convention exactly:
// Re[k] = sum_t (x[FrameStart+t]*Window[t]) * cos(2*pi*k*t/NFFT)
// Im[k] = sum_t (x[FrameStart+t]*Window[t]) * sin(2*pi*k*t/NFFT)
// (no centering: FrameStart = frame*Hop), periodic Hann analysis window.
// Produces Mag/Phase volumes (NumFrames, 1, NFFT div 2 + 1) consumable by
// ISTFTOverlapAdd. This is the inverse contract under test.
procedure ForwardSTFT(Sig: TNNetVolume; Mag, Phase: TNNetVolume;
  NFFT, Hop: integer);
var
  NumBins, NumFrames, f, k, t, start, src: integer;
  Window: array of double;
  ReAcc, ImAcc, v: double;
begin
  NumBins := NFFT div 2 + 1;
  NumFrames := (Sig.Size - NFFT) div Hop + 1;
  SetLength(Window, NFFT);
  for t := 0 to NFFT - 1 do
    Window[t] := 0.5 - 0.5 * Cos(2.0 * Pi * t / NFFT);
  Mag.ReSize(NumFrames, 1, NumBins);
  Phase.ReSize(NumFrames, 1, NumBins);
  for f := 0 to NumFrames - 1 do
  begin
    start := f * Hop;
    for k := 0 to NumBins - 1 do
    begin
      ReAcc := 0.0;
      ImAcc := 0.0;
      for t := 0 to NFFT - 1 do
      begin
        src := start + t;
        v := Sig.FData[src] * Window[t];
        ReAcc := ReAcc + v * Cos(2.0 * Pi * k * t / NFFT);
        ImAcc := ImAcc + v * Sin(2.0 * Pi * k * t / NFFT);
      end;
      Mag.FData[f * NumBins + k] := Sqrt(ReAcc * ReAcc + ImAcc * ImAcc);
      Phase.FData[f * NumBins + k] := ArcTan2(ImAcc, ReAcc);
    end;
  end;
end;

procedure TTestNeuralAudio.TestISTFTRoundTrip;
const
  csNFFT = 16;
  csHop = csNFFT div 4;   // 75% overlap -> Hann satisfies COLA
  csN = 256;
var
  Sig, Mag, Phase, Rec: TNNetVolume;
  i, lo, hi: integer;
  t, MaxDiff, d: double;
begin
  Sig := TNNetVolume.Create(csN, 1, 1);
  Mag := TNNetVolume.Create(1, 1, 1);
  Phase := TNNetVolume.Create(1, 1, 1);
  Rec := TNNetVolume.Create(1, 1, 1);
  try
    // deterministic sum of two sinusoids, well inside [-1, 1]
    for i := 0 to csN - 1 do
    begin
      t := i / csN;
      Sig.FData[i] := 0.5 * Sin(2.0 * Pi * 5.0 * t) +
                      0.3 * Cos(2.0 * Pi * 11.0 * t);
    end;
    ForwardSTFT(Sig, Mag, Phase, csNFFT, csHop);
    ISTFTOverlapAdd(Mag, Phase, Rec, csNFFT, csHop);

    // Interior only: skip the first/last (NFFT - Hop) samples where the
    // overlap-add envelope is partial (COLA not satisfied at the edges).
    // Tolerance 1e-9: the DSP is all double precision; only TNNetVolume's
    // float32 storage of Mag/Phase/Rec limits the round-trip (~1e-6), so we
    // keep a comfortable 1e-4 bound the float32 path easily clears.
    lo := csNFFT - csHop;
    hi := Rec.Size - (csNFFT - csHop) - 1;
    AssertTrue('reconstruction length covers the interior', hi > lo);
    MaxDiff := 0;
    for i := lo to hi do
    begin
      d := Abs(Sig.FData[i] - Rec.FData[i]);
      if d > MaxDiff then MaxDiff := d;
    end;
    AssertTrue('interior ISTFT round-trip max abs diff < 1e-4 (' +
      FloatToStr(MaxDiff) + ')', MaxDiff < 1e-4);
  finally
    Sig.Free;
    Mag.Free;
    Phase.Free;
    Rec.Free;
  end;
end;

// Counts sign changes (zero crossings) of a mono waveform - twice the
// frequency in cycles per the buffer's duration, robust to amplitude.
function CountZeroCrossings(V: TNNetVolume): integer;
var
  i: integer;
  Prev, Cur: double;
begin
  Result := 0;
  if V.Size < 2 then exit;
  Prev := V.FData[0];
  for i := 1 to V.Size - 1 do
  begin
    Cur := V.FData[i];
    if ((Prev < 0) and (Cur >= 0)) or ((Prev >= 0) and (Cur < 0)) then
      Inc(Result);
    Prev := Cur;
  end;
end;

procedure TTestNeuralAudio.TestResampleIdentity;
var
  Src, Dst: TNNetVolume;
  N, i: integer;
begin
  N := 2000;
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := nil;
  try
    RandSeed := 314159;
    for i := 0 to N - 1 do
      Src.FData[i] := 0.7 * Sin(2.0 * Pi * 300.0 * i / 16000.0)
        + 0.1 * (Random - 0.5);
    Dst := ResampleVolume(Src, 16000, 16000);
    AssertEquals('identity resample length', N, Dst.Size);
    for i := 0 to N - 1 do
      AssertTrue('identity resample is bit-identical at ' + IntToStr(i),
        Src.FData[i] = Dst.FData[i]);
  finally
    Src.Free;
    Dst.Free;
  end;
end;

procedure TTestNeuralAudio.TestResampleLength;
var
  Src, Dst: TNNetVolume;
  N: integer;
begin
  N := 4410;
  Src := TNNetVolume.Create(N, 1, 1);
  Dst := nil;
  try
    // Downsample 44100 -> 16000 : expect round(4410 * 16000/44100) = 1600.
    Dst := ResampleVolume(Src, 44100, 16000);
    AssertEquals('downsample length', Round(N * 16000.0 / 44100.0), Dst.Size);
    AssertEquals('downsample length value', 1600, Dst.Size);
    Dst.Free; Dst := nil;
    // Upsample 8000 -> 16000 : expect exactly 2*N.
    Dst := ResampleVolume(Src, 8000, 16000);
    AssertEquals('upsample length', 2 * N, Dst.Size);
  finally
    Src.Free;
    Dst.Free;
  end;
end;

procedure TTestNeuralAudio.TestResampleSinePreservesFrequency;
var
  Src, Dst: TNNetVolume;
  N, i, ZcSrc, ZcDst: integer;
  ExpectedZc, Tol: double;

  procedure CheckRate(SrcRate, DstRate: integer; Freq: double);
  var DurSec: double; k: integer;
  begin
    DurSec := 0.25; // 250 ms of signal
    N := Round(SrcRate * DurSec);
    Src.ReSize(N, 1, 1);
    for k := 0 to N - 1 do
      Src.FData[k] := Sin(2.0 * Pi * Freq * k / SrcRate);
    ZcSrc := CountZeroCrossings(Src);
    Dst := ResampleVolume(Src, SrcRate, DstRate);
    try
      ZcDst := CountZeroCrossings(Dst);
      // Both should be ~ 2 * Freq * DurSec crossings; allow a couple of edge
      // crossings of slack (the resampled buffer covers the same duration).
      ExpectedZc := 2.0 * Freq * DurSec;
      Tol := 4;
      AssertTrue('src zero-crossings near expected (' + IntToStr(ZcSrc) +
        ' vs ' + FloatToStr(ExpectedZc) + ')', Abs(ZcSrc - ExpectedZc) <= Tol);
      AssertTrue('resampled zero-crossings preserved (' + IntToStr(ZcDst) +
        ' vs src ' + IntToStr(ZcSrc) + ')', Abs(ZcDst - ZcSrc) <= Tol);
    finally
      Dst.Free; Dst := nil;
    end;
  end;

begin
  Src := TNNetVolume.Create(1, 1, 1);
  Dst := nil;
  try
    CheckRate(44100, 16000, 440.0);  // downsample, A4
    CheckRate(8000, 16000, 440.0);   // upsample, A4
  finally
    Src.Free;
    Dst.Free;
  end;
end;

procedure TTestNeuralAudio.TestLoadWavResampledRoundTrip;
var
  Src, Loaded: TNNetVolume;
  N, i, Sr: integer;
begin
  FTmpWav := GetTempDir(false) + 'cai_audio_resample_' +
    IntToStr(Random(1000000)) + '.wav';
  N := 2000; // 0.25 s at 8 kHz
  Src := TNNetVolume.Create(N, 1, 1);
  Loaded := TNNetVolume.Create(1, 1, 1);
  try
    for i := 0 to N - 1 do
      Src.FData[i] := 0.6 * Sin(2.0 * Pi * 300.0 * i / 8000.0);
    SaveVolumeToWav16(Src, FTmpWav, 8000);
    Sr := LoadWavResampledToVolume(FTmpWav, Loaded, 16000);
    AssertEquals('returns target rate', 16000, Sr);
    AssertEquals('resampled length', Round(N * 16000.0 / 8000.0), Loaded.Size);
    AssertEquals('resampled length value', 4000, Loaded.Size);
    // 300 Hz over 0.25 s => ~150 zero crossings, preserved through disk+resample.
    AssertTrue('fundamental preserved through load+resample',
      Abs(CountZeroCrossings(Loaded) - 150) <= 4);
  finally
    Src.Free;
    Loaded.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralAudio);
end.
