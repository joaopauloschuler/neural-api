(*
neuralaudio
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

// AUDIO FRONTEND for speech models - the first audio unit in the library.
//
// Two building blocks, both consumed by the Whisper import target
// (neural/neuralpretrained.pas BuildWhisperFromSafeTensors and the
// examples/WhisperTranscribe demo):
//
//   - LoadWav16ToVolume: a minimal RIFF/WAVE reader for 16-bit PCM files
//     (mono or multi-channel - channels are averaged to mono). Returns the
//     sample rate; samples land in [-1, 1) floats, the torchaudio/soundfile
//     convention (s / 32768).
//
//   - ComputeWhisperLogMel: the EXACT log-mel spectrogram of HF transformers'
//     WhisperFeatureExtractor (verified to ~1e-5 against the float64 oracle in
//     tests/fixtures/whisper_mel_oracle.json):
//       * pad/truncate the waveform to NumFrames*160 samples (30 s = 480000
//         at the canonical NumFrames = 3000),
//       * center-REFLECT-pad by n_fft/2 = 200 samples on both sides,
//       * 400-point STFT (PERIODIC hann window, hop 160) - computed here as a
//         direct real DFT against precomputed cos/sin tables because 400 is
//         not a power of two (the radix-2 FFTs in neuralnetwork.pas do not
//         apply); all-zero frames (the 30 s padding tail) are skipped,
//       * power spectrum (201 bins) through the SLANEY-scale, slaney-NORMED
//         80-filter triangular mel bank (mel_filter_bank(num_frequency_bins=
//         201, num_mel_filters=80, min=0, max=8000, norm="slaney",
//         mel_scale="slaney")),
//       * log10 floored at 1e-10, the LAST frame dropped (3001 -> 3000),
//       * global max-minus-8 clamp, then (x + 4) / 4.
//     Output volume layout: (NumFrames, 1, NumMelBins) - the time axis along
//     SizeX, mel bins along Depth, matching the (SeqLen, 1, Channels)
//     convention of every sequence layer and the Whisper encoder input.
//     Everything is computed in double precision; only the final write into
//     the volume rounds to float32.
//
// Coded by Claude (AI).
unit neuralaudio;

{$IFDEF FPC}
{$mode objfpc}
{$H+}
{$ENDIF}

interface

uses
  Classes, SysUtils, Math, neuralvolume;

// Reads a 16-bit PCM RIFF/WAVE file into Samples as mono floats in [-1, 1)
// (multi-channel input is averaged). Samples is resized to (N, 1, 1).
// Returns the sample rate in Hz. Raises Exception on anything that is not
// an uncompressed 16-bit PCM WAV.
function LoadWav16ToVolume(const FileName: string;
  Samples: TNNetVolume): integer;

// Writes Samples (a mono waveform, floats in [-1, 1], laid out along FData as
// LoadWav16ToVolume reads them) to a canonical mono 16-bit PCM RIFF/WAVE file.
// Samples are clamped to [-1, 1] and scaled to int16 (x * 32768, rounded and
// clamped to the int16 range). This is the inverse of LoadWav16ToVolume
// (which divides by 32768) to within 1 LSB.
procedure SaveVolumeToWav16(Samples: TNNetVolume; const FileName: string;
  SampleRate: integer = 16000);

// Sample-rate conversion of a mono waveform (floats in [-1, 1], laid out
// along FData as LoadWav16ToVolume reads them) from SourceRate to TargetRate.
// Returns a NEWLY created TNNetVolume of shape (round(N * TargetRate /
// SourceRate), 1, 1) - the CALLER OWNS and must Free it.
//
// Quality: a windowed-sinc (Lanczos, default A = csResampleLanczosA lobes)
// polyphase-style kernel evaluated per output sample. For DOWNSAMPLING the
// sinc cutoff is lowered to the output Nyquist (cutoff = TargetRate /
// SourceRate in input-sample units), so the low-pass anti-alias filter is
// folded into the interpolation kernel - no separate pre-filter needed. For
// UPSAMPLING the cutoff stays at the input Nyquist (band-limited interp).
// This is materially better than naive linear interpolation for downsampling
// (linear leaves audible aliasing); the cost is O(N_out * 2*A/ratio) which is
// negligible next to the STFT/mel that follows.
//
// FAST PATH: when SourceRate = TargetRate the input is copied bit-identically
// (no kernel evaluated), so a 16 kHz -> 16 kHz call is a pure passthrough.
function ResampleVolume(Wave: TNNetVolume; SourceRate, TargetRate: integer): TNNetVolume;

// Convenience: ResampleVolume(Wave, SourceRate, 16000). Caller owns Result.
function ResampleVolumeTo16k(Wave: TNNetVolume; SourceRate: integer): TNNetVolume;

// Like LoadWav16ToVolume but the loaded waveform is resampled to TargetRate
// (default 16000 Hz) mono. When the file is already at TargetRate this is
// bit-identical to LoadWav16ToVolume. Returns TargetRate (always); Samples is
// the resampled mono waveform (N, 1, 1).
function LoadWavResampledToVolume(const FileName: string; Samples: TNNetVolume;
  TargetRate: integer = 16000): integer;

// HF WhisperFeatureExtractor log-mel spectrogram (see the unit header).
// Samples: mono waveform at 16 kHz, any length (padded/truncated to
// NumFrames*160 samples). Mel is resized to (NumFrames, 1, NumMelBins).
procedure ComputeWhisperLogMel(Samples: TNNetVolume; Mel: TNNetVolume;
  NumMelBins: integer = 80; NumFrames: integer = 3000);

// Convenience wrapper: load a WAV + ComputeWhisperLogMel. By default
// (Resample = True) a WAV at any sample rate is accepted and resampled to
// 16000 Hz mono via ResampleVolume (a 16 kHz file passes through untouched).
// Pass Resample = False to keep the strict v1 behaviour (raise Exception if
// the file is not already 16000 Hz).
procedure WhisperLogMelFromWavFile(const FileName: string; Mel: TNNetVolume;
  NumMelBins: integer = 80; NumFrames: integer = 3000;
  Resample: boolean = True);

// MusicGen Melody chroma front-end (HF transformers
// MusicgenMelodyFeatureExtractor): a 12-bin CHROMAGRAM of a reference waveform
// that conditions music generation in BuildMusicGenMelodyFromSafeTensors. The
// pipeline matches the feature extractor bit-for-bit:
//   - a POWER spectrogram (torchaudio.transforms.Spectrogram with
//     n_fft=NFFT, win_length=NFFT, hop_length=Hop, a PERIODIC Hann window,
//     center=True / reflect pad, NORMALIZED by the window L2 energy
//     sqrt(sum(window^2)), power=2);
//   - a chroma_filter_bank (librosa-style, see BuildChromaFilterBank) mapping
//     the NFFT div 2 + 1 one-sided FFT bins onto NumChroma pitch classes;
//   - per-frame inf-norm normalization (divide each chroma row by its max,
//     eps 1e-6);
//   - per-frame ARGMAX one-hot: the dominant pitch class is set to 1, the rest
//     to 0.
// Samples is a mono waveform at SampleRate (32000 for the real model; any
// length - padded to NFFT if shorter, matching the extractor). Chroma is
// resized to (NumFrames, 1, NumChroma): time along SizeX, pitch classes along
// Depth (the same (SeqLen,1,Channels) layout ComputeWhisperLogMel emits), where
// NumFrames = 1 + Samples.Size div Hop (the center-padded STFT frame count).
// Coded by Claude (AI).
procedure ComputeMusicgenMelodyChroma(Samples: TNNetVolume; Chroma: TNNetVolume;
  SampleRate: integer = 32000; NFFT: integer = 16384; Hop: integer = 4096;
  NumChroma: integer = 12);

// Builds the MusicGen Melody chroma filter bank (librosa/transformers
// chroma_filter_bank with tuning=0, power=2, weighting_parameters=(5,2),
// start_at_c_chroma=True): a (NumChroma x (NFFT div 2 + 1)) matrix projecting
// one-sided power-spectrum bins onto pitch classes. Filt is resized to
// (NumChroma, 1, NFFT div 2 + 1): Filt[c][b] is the weight of FFT bin b in
// chroma c. Depends only on (SampleRate, NFFT, NumChroma) - a fixed constant
// matrix for a given model. Coded by Claude (AI).
procedure BuildChromaFilterBank(Filt: TNNetVolume;
  SampleRate, NFFT, NumChroma: integer);

// Inverse STFT overlap-add (OLA) synthesis - the exact inverse of the forward
// real STFT used by ComputeWhisperLogMel (periodic Hann analysis window, same
// one-sided rDFT bin convention). Reconstructs a mono waveform from a complex
// spectrogram given as a magnitude + phase pair (ISTFTOverlapAdd) or as a
// real + imaginary pair (ISTFTOverlapAddReIm).
//
// Mag/Phase (or Re/Im) are (NumFrames, 1, NumFreqBins) volumes, time along
// SizeX and frequency bins along Depth - the same (SeqLen, 1, Channels) layout
// ComputeWhisperLogMel emits. NumFreqBins MUST equal NFFT div 2 + 1 (one-sided
// real-FFT layout). The bin convention mirrors the forward DFT tables exactly:
//   Re[k] = Mag[k]*cos(Phase[k]),  Im[k] = Mag[k]*sin(Phase[k])
// where the stored Im is +sum(x*sin(2*pi*k*t/NFFT)) (i.e. the negated standard
// imaginary part, identical to the forward CosTab/SinTab convention here), so
// a forward STFT computed with the file's tables fed straight back in is a true
// round-trip.
//
// Each frame is inverse-rDFT'd, multiplied by the synthesis Hann window, and
// overlap-added; the accumulated signal is then divided by the overlap-added
// squared-window envelope (the COLA / window_sumsquare normalization that
// torch/librosa apply), with a small epsilon guard against divide-by-zero at
// edge samples covered by no window. With a perfect-reconstruction Hann window
// and 75% overlap (HopLength = NFFT div 4) the interior reconstructs the
// original waveform to ~1e-12.
//
// Wave is resized to (NFFT + (NumFrames-1)*HopLength, 1, 1) - the full
// uncentred OLA length (NO librosa center-trim; trim NFFT div 2 from each end
// yourself if you fed a centre-padded forward STFT). The first and last
// ~NFFT-HopLength samples are partial-overlap (COLA not satisfied) and should
// be treated as edge transients.
procedure ISTFTOverlapAdd(Mag, Phase, Wave: TNNetVolume;
  NFFT: integer; HopLength: integer);

// As ISTFTOverlapAdd but the spectrogram is supplied as real + imaginary parts
// (same bin/sign convention as the forward CosTab/SinTab tables here) instead
// of magnitude + phase. ISTFTOverlapAdd is a thin wrapper over this.
procedure ISTFTOverlapAddReIm(Re, Im, Wave: TNNetVolume;
  NFFT: integer; HopLength: integer);

implementation

const
  csWhisperNFFT = 400;      // STFT window (25 ms at 16 kHz)
  csWhisperHop = 160;       // hop (10 ms at 16 kHz)
  csWhisperSampleRate = 16000;
  csWhisperMaxFreq = 8000.0;
  csResampleLanczosA = 4;   // sinc lobes per side for the resampler kernel
  csWhisperMelFloor = 1e-10;

function LoadWav16ToVolume(const FileName: string;
  Samples: TNNetVolume): integer;
var
  FS: TFileStream;
  ChunkId: array[0..3] of AnsiChar;
  ChunkSize, RiffType: longword;
  AudioFormat, NumChannels, BitsPerSample: word;
  SampleRate, ByteRate: longword;
  BlockAlign: word;
  HaveFmt: boolean;
  DataBytes, FrameCnt, NumFrames, ChCnt: integer;
  NumFramesM1, NumChannelsM1: integer;
  Raw: array of smallint;
  Acc: double;

  procedure ReadExact(var Buf; Count: integer);
  begin
    if FS.Read(Buf, Count) <> Count then
      raise Exception.Create('LoadWav16ToVolume: unexpected end of file in ' +
        FileName);
  end;

begin
  Result := 0;
  AudioFormat := 0; NumChannels := 0; BitsPerSample := 0; SampleRate := 0;
  HaveFmt := false;
  FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
  try
    ReadExact(ChunkId, 4);
    if ChunkId <> 'RIFF' then
      raise Exception.Create('LoadWav16ToVolume: not a RIFF file: ' +
        FileName);
    ReadExact(ChunkSize, 4);
    ReadExact(RiffType, 4);
    if RiffType <> $45564157 then // 'WAVE' little-endian
      raise Exception.Create('LoadWav16ToVolume: not a WAVE file: ' +
        FileName);
    // Walk the chunks: "fmt " must precede "data".
    while FS.Position + 8 <= FS.Size do
    begin
      ReadExact(ChunkId, 4);
      ReadExact(ChunkSize, 4);
      if ChunkId = 'fmt ' then
      begin
        if ChunkSize < 16 then
          raise Exception.Create('LoadWav16ToVolume: malformed fmt chunk in ' +
            FileName);
        ReadExact(AudioFormat, 2);
        ReadExact(NumChannels, 2);
        ReadExact(SampleRate, 4);
        ReadExact(ByteRate, 4);
        ReadExact(BlockAlign, 2);
        ReadExact(BitsPerSample, 2);
        FS.Seek(ChunkSize - 16, soFromCurrent);
        if AudioFormat <> 1 then
          raise Exception.Create('LoadWav16ToVolume: only uncompressed PCM ' +
            '(format 1) is supported, got format ' + IntToStr(AudioFormat) +
            ' in ' + FileName);
        if BitsPerSample <> 16 then
          raise Exception.Create('LoadWav16ToVolume: only 16-bit PCM is ' +
            'supported, got ' + IntToStr(BitsPerSample) + '-bit in ' +
            FileName);
        if NumChannels < 1 then
          raise Exception.Create('LoadWav16ToVolume: zero channels in ' +
            FileName);
        HaveFmt := true;
      end
      else if ChunkId = 'data' then
      begin
        if not HaveFmt then
          raise Exception.Create('LoadWav16ToVolume: "data" before "fmt " ' +
            'in ' + FileName);
        DataBytes := ChunkSize;
        if FS.Position + DataBytes > FS.Size then
          DataBytes := FS.Size - FS.Position; // tolerate a truncated tail
        NumFrames := DataBytes div (2 * NumChannels);
        SetLength(Raw, NumFrames * NumChannels);
        if NumFrames > 0 then
          ReadExact(Raw[0], NumFrames * NumChannels * 2);
        Samples.ReSize(NumFrames, 1, 1);
        NumFramesM1 := NumFrames - 1;
        NumChannelsM1 := NumChannels - 1;
        for FrameCnt := 0 to NumFramesM1 do
        begin
          Acc := 0;
          for ChCnt := 0 to NumChannelsM1 do
            Acc := Acc + Raw[FrameCnt * NumChannels + ChCnt];
          Samples.FData[FrameCnt] := (Acc / NumChannels) / 32768.0;
        end;
        Result := SampleRate;
        exit;
      end
      else
        FS.Seek(longint(ChunkSize + (ChunkSize and 1)), soFromCurrent);
    end;
    raise Exception.Create('LoadWav16ToVolume: no "data" chunk in ' +
      FileName);
  finally
    FS.Free;
  end;
end;

function ResampleVolume(Wave: TNNetVolume; SourceRate, TargetRate: integer): TNNetVolume;
var
  NIn, NOut, OutCnt, J, JLo, JHi: integer;
  NOutM1: integer;
  Ratio, Cutoff, SrcPos, Center, X, Acc, WSum, W, PiX, PiXA: double;

  // Lanczos-windowed sinc at offset T (in input-sample units), low-passed at
  // Cutoff (cycles per input sample, <= 0.5). Returns the kernel weight.
  function LanczosKernel(T: double): double;
  begin
    if Abs(T) < 1e-12 then
    begin
      Result := 2.0 * Cutoff;
      exit;
    end;
    if Abs(T) >= csResampleLanczosA then
    begin
      Result := 0.0;
      exit;
    end;
    // sinc(2*Cutoff*T) low-pass, windowed by the Lanczos lobe sinc(T/A).
    PiX := Pi * 2.0 * Cutoff * T;
    PiXA := Pi * T / csResampleLanczosA;
    Result := 2.0 * Cutoff * (Sin(PiX) / PiX) * (Sin(PiXA) / PiXA);
  end;

begin
  if (SourceRate <= 0) or (TargetRate <= 0) then
    raise Exception.Create('ResampleVolume: sample rates must be positive.');
  Result := TNNetVolume.Create;
  NIn := Wave.Size;
  // FAST PATH: identical rates -> bit-identical copy, no kernel evaluated.
  if SourceRate = TargetRate then
  begin
    Result.ReSize(NIn, 1, 1);
    if NIn > 0 then
      Move(Wave.FData[0], Result.FData[0], NIn * SizeOf(Wave.FData[0]));
    exit;
  end;
  NOut := Round(NIn * (TargetRate / SourceRate));
  Result.ReSize(NOut, 1, 1);
  if (NIn = 0) or (NOut = 0) then exit;

  Ratio := TargetRate / SourceRate;
  // Anti-alias cutoff: input Nyquist when upsampling (Ratio>=1, Cutoff=0.5),
  // lowered to the OUTPUT Nyquist when downsampling (Cutoff=0.5*Ratio). The
  // kernel half-width grows by 1/min(1,Ratio) so it still spans A output lobes.
  if Ratio >= 1.0 then
    Cutoff := 0.5
  else
    Cutoff := 0.5 * Ratio;

  NOutM1 := NOut - 1;
  for OutCnt := 0 to NOutM1 do
  begin
    // Position of this output sample expressed in INPUT-sample coordinates.
    SrcPos := OutCnt / Ratio;
    Center := SrcPos;
    // Kernel support in input samples: A lobes scaled by the cutoff stretch.
    if Ratio >= 1.0 then
    begin
      JLo := Floor(Center) - csResampleLanczosA + 1;
      JHi := Floor(Center) + csResampleLanczosA;
    end
    else
    begin
      JLo := Floor(Center - csResampleLanczosA / Ratio);
      JHi := Ceil(Center + csResampleLanczosA / Ratio);
    end;
    Acc := 0.0;
    WSum := 0.0;
    for J := JLo to JHi do
    begin
      if (J < 0) or (J > NIn - 1) then continue;
      X := Center - J;
      W := LanczosKernel(X);
      if W = 0.0 then continue;
      Acc := Acc + W * Wave.FData[J];
      WSum := WSum + W;
    end;
    if WSum <> 0.0 then
      Result.FData[OutCnt] := Acc / WSum
    else
      Result.FData[OutCnt] := 0.0;
  end;
end;

function ResampleVolumeTo16k(Wave: TNNetVolume; SourceRate: integer): TNNetVolume;
begin
  Result := ResampleVolume(Wave, SourceRate, csWhisperSampleRate);
end;

function LoadWavResampledToVolume(const FileName: string; Samples: TNNetVolume;
  TargetRate: integer = 16000): integer;
var
  SrcRate: integer;
  Resampled: TNNetVolume;
begin
  SrcRate := LoadWav16ToVolume(FileName, Samples);
  if SrcRate <> TargetRate then
  begin
    Resampled := ResampleVolume(Samples, SrcRate, TargetRate);
    try
      Samples.Copy(Resampled);
    finally
      Resampled.Free;
    end;
  end;
  Result := TargetRate;
end;

procedure SaveVolumeToWav16(Samples: TNNetVolume; const FileName: string;
  SampleRate: integer = 16000);
var
  FS: TFileStream;
  NumFrames, i: integer;
  NumFramesM1: integer;
  DataBytes, ByteRate, FileTail: longword;
  ChunkSize16: longword;
  NumChannels, BlockAlign, BitsPerSample, AudioFormat: word;
  SR: longword;
  Raw: array of smallint;
  V: TNeuralFloat;
  Scaled: double;

  procedure WriteBytes(const Buf; Count: integer);
  begin
    FS.WriteBuffer(Buf, Count);
  end;

  procedure WriteTag(const Tag: string);
  var
    A: array[0..3] of AnsiChar;
    k: integer;
  begin
    for k := 0 to 3 do A[k] := AnsiChar(Tag[k + 1]);
    WriteBytes(A, 4);
  end;

begin
  NumFrames := Samples.Size;
  NumChannels := 1;
  BitsPerSample := 16;
  AudioFormat := 1;            // PCM
  SR := longword(SampleRate);
  BlockAlign := NumChannels * (BitsPerSample div 8);
  ByteRate := SR * BlockAlign;
  DataBytes := longword(NumFrames) * BlockAlign;
  ChunkSize16 := 16;
  // RIFF chunk size = 4 ("WAVE") + (8 + 16) fmt + (8 + DataBytes) data.
  FileTail := 4 + (8 + 16) + (8 + DataBytes);

  SetLength(Raw, NumFrames);
  NumFramesM1 := NumFrames - 1;
  for i := 0 to NumFramesM1 do
  begin
    V := Samples.FData[i];
    if V > 1.0 then V := 1.0
    else if V < -1.0 then V := -1.0;
    // Scale by 32768 to be the exact inverse of LoadWav16ToVolume (which
    // divides by 32768), clamped to the int16 range so +1.0 does not overflow.
    Scaled := Round(V * 32768.0);
    if Scaled > 32767 then Scaled := 32767
    else if Scaled < -32768 then Scaled := -32768;
    Raw[i] := smallint(Round(Scaled));
  end;

  FS := TFileStream.Create(FileName, fmCreate);
  try
    WriteTag('RIFF');
    WriteBytes(FileTail, 4);
    WriteTag('WAVE');
    WriteTag('fmt ');
    WriteBytes(ChunkSize16, 4);
    WriteBytes(AudioFormat, 2);
    WriteBytes(NumChannels, 2);
    WriteBytes(SR, 4);
    WriteBytes(ByteRate, 4);
    WriteBytes(BlockAlign, 2);
    WriteBytes(BitsPerSample, 2);
    WriteTag('data');
    WriteBytes(DataBytes, 4);
    if NumFrames > 0 then
      WriteBytes(Raw[0], NumFrames * 2);
  finally
    FS.Free;
  end;
end;

// Slaney-scale frequency conversion (transformers.audio_utils
// hertz_to_mel/mel_to_hertz with mel_scale="slaney"): linear below 1 kHz
// (3 mel per 200 Hz), logarithmic above (27 mel per factor of 6.4).
function HertzToMelSlaney(Freq: double): double;
begin
  if Freq >= 1000.0 then
    Result := 15.0 + Ln(Freq / 1000.0) * (27.0 / Ln(6.4))
  else
    Result := 3.0 * Freq / 200.0;
end;

function MelToHertzSlaney(Mels: double): double;
begin
  if Mels >= 15.0 then
    Result := 1000.0 * Exp((Ln(6.4) / 27.0) * (Mels - 15.0))
  else
    Result := 200.0 * Mels / 3.0;
end;

procedure ComputeWhisperLogMel(Samples: TNNetVolume; Mel: TNNetVolume;
  NumMelBins: integer = 80; NumFrames: integer = 3000);
var
  NumSamples, NumBins, NumStftFrames: integer;
  NumSamplesM1, NumBinsM1, NumMelBinsM1, NumMelBinsP1, NFFTM1: integer;
  NumStftFramesM1, NumFramesM1: integer;
  Wave: array of double;          // padded/truncated waveform
  Window: array of double;        // periodic hann
  CosTab, SinTab: array of double; // (NumBins x NFFT) DFT twiddles
  FilterBank: array of double;    // (NumBins x NumMelBins) triangular bank
  FilterFreqs: array of double;   // NumMelBins + 2 triangle corner freqs
  Power: array of double;         // one frame's power spectrum
  LogMel: array of double;        // (NumFrames x NumMelBins)
  SampleCnt, FrameCnt, BinCnt, MelCnt, TapCnt, FrameStart, SrcIdx: integer;
  MelMin, MelMax, FFTFreq, DownSlope, UpSlope, Tri: double;
  ReAcc, ImAcc, Acc, MaxLog, V: double;
  AllZero: boolean;

  // np.pad reflect indexing for the center pad (mirror WITHOUT the edge
  // sample). Only single-bounce reflection is needed: the pad (200) is
  // always shorter than the waveform (NumFrames*160 >= 400 in practice).
  function WaveAt(Idx: integer): double;
  begin
    if Idx < 0 then Idx := -Idx;
    if Idx >= NumSamples then Idx := 2 * NumSamples - 2 - Idx;
    Result := Wave[Idx];
  end;

begin
  if NumFrames < 3 then
    raise Exception.Create('ComputeWhisperLogMel: NumFrames must be >= 3.');
  if NumMelBins < 2 then
    raise Exception.Create('ComputeWhisperLogMel: NumMelBins must be >= 2.');
  NumSamples := NumFrames * csWhisperHop;
  NumBins := csWhisperNFFT div 2 + 1; // 201 one-sided rfft bins
  NumSamplesM1 := NumSamples - 1;
  NumBinsM1 := NumBins - 1;
  NumMelBinsM1 := NumMelBins - 1;
  NumMelBinsP1 := NumMelBins + 1;
  NFFTM1 := csWhisperNFFT - 1;
  NumFramesM1 := NumFrames - 1;

  // ---- pad / truncate to the fixed 30 s context ----
  SetLength(Wave, NumSamples);
  for SampleCnt := 0 to NumSamplesM1 do
    if SampleCnt < Samples.Size then
      Wave[SampleCnt] := Samples.FData[SampleCnt]
    else
      Wave[SampleCnt] := 0.0;

  // ---- periodic hann window (transformers window_function default) ----
  SetLength(Window, csWhisperNFFT);
  for TapCnt := 0 to NFFTM1 do
    Window[TapCnt] := 0.5 - 0.5 * Cos(2.0 * Pi * TapCnt / csWhisperNFFT);

  // ---- DFT twiddle tables (400 is not a power of two - direct rDFT) ----
  SetLength(CosTab, NumBins * csWhisperNFFT);
  SetLength(SinTab, NumBins * csWhisperNFFT);
  for BinCnt := 0 to NumBinsM1 do
    for TapCnt := 0 to NFFTM1 do
    begin
      CosTab[BinCnt * csWhisperNFFT + TapCnt] :=
        Cos(2.0 * Pi * BinCnt * TapCnt / csWhisperNFFT);
      SinTab[BinCnt * csWhisperNFFT + TapCnt] :=
        Sin(2.0 * Pi * BinCnt * TapCnt / csWhisperNFFT);
    end;

  // ---- slaney mel filter bank (triangles in Hz, slaney-normed) ----
  MelMin := HertzToMelSlaney(0.0);
  MelMax := HertzToMelSlaney(csWhisperMaxFreq);
  SetLength(FilterFreqs, NumMelBins + 2);
  for MelCnt := 0 to NumMelBinsP1 do
    FilterFreqs[MelCnt] := MelToHertzSlaney(
      MelMin + (MelMax - MelMin) * MelCnt / (NumMelBins + 1));
  SetLength(FilterBank, NumBins * NumMelBins);
  for BinCnt := 0 to NumBinsM1 do
  begin
    FFTFreq := (csWhisperSampleRate div 2) * BinCnt / (NumBins - 1);
    for MelCnt := 0 to NumMelBinsM1 do
    begin
      DownSlope := (FFTFreq - FilterFreqs[MelCnt]) /
        (FilterFreqs[MelCnt + 1] - FilterFreqs[MelCnt]);
      UpSlope := (FilterFreqs[MelCnt + 2] - FFTFreq) /
        (FilterFreqs[MelCnt + 2] - FilterFreqs[MelCnt + 1]);
      Tri := DownSlope;
      if UpSlope < Tri then Tri := UpSlope;
      if Tri < 0.0 then Tri := 0.0;
      // slaney norm: ~constant energy per channel.
      FilterBank[BinCnt * NumMelBins + MelCnt] := Tri *
        (2.0 / (FilterFreqs[MelCnt + 2] - FilterFreqs[MelCnt]));
    end;
  end;

  // ---- framed power spectrum -> mel -> log10 ----
  // The center pad gives 1 + NumSamples/hop frames; the LAST one is
  // dropped by WhisperFeatureExtractor (log_spec[:, :-1]), so only the
  // first NumFrames frames are ever computed here.
  NumStftFrames := NumFrames;
  NumStftFramesM1 := NumStftFrames - 1;
  SetLength(Power, NumBins);
  SetLength(LogMel, NumStftFrames * NumMelBins);
  MaxLog := -1e30;
  for FrameCnt := 0 to NumStftFramesM1 do
  begin
    FrameStart := FrameCnt * csWhisperHop - (csWhisperNFFT div 2);
    // The 30 s zero-padding tail produces all-zero frames whose mel rows
    // are exactly log10(mel_floor) - skip their O(bins*taps) DFT.
    AllZero := true;
    for TapCnt := 0 to NFFTM1 do
    begin
      SrcIdx := FrameStart + TapCnt;
      if (SrcIdx >= 0) and (SrcIdx < NumSamples) then
      begin
        if Wave[SrcIdx] <> 0.0 then begin AllZero := false; break; end;
      end
      else if WaveAt(SrcIdx) <> 0.0 then begin AllZero := false; break; end;
    end;
    if AllZero then
    begin
      for MelCnt := 0 to NumMelBinsM1 do
        LogMel[FrameCnt * NumMelBins + MelCnt] := Ln(csWhisperMelFloor) / Ln(10.0);
    end
    else
    begin
      for BinCnt := 0 to NumBinsM1 do
      begin
        ReAcc := 0.0;
        ImAcc := 0.0;
        for TapCnt := 0 to NFFTM1 do
        begin
          V := WaveAt(FrameStart + TapCnt) * Window[TapCnt];
          ReAcc := ReAcc + V * CosTab[BinCnt * csWhisperNFFT + TapCnt];
          ImAcc := ImAcc + V * SinTab[BinCnt * csWhisperNFFT + TapCnt];
        end;
        Power[BinCnt] := ReAcc * ReAcc + ImAcc * ImAcc;
      end;
      for MelCnt := 0 to NumMelBinsM1 do
      begin
        Acc := 0.0;
        for BinCnt := 0 to NumBinsM1 do
          Acc := Acc + FilterBank[BinCnt * NumMelBins + MelCnt] *
            Power[BinCnt];
        if Acc < csWhisperMelFloor then Acc := csWhisperMelFloor;
        LogMel[FrameCnt * NumMelBins + MelCnt] := Ln(Acc) / Ln(10.0);
      end;
    end;
    for MelCnt := 0 to NumMelBinsM1 do
      if LogMel[FrameCnt * NumMelBins + MelCnt] > MaxLog then
        MaxLog := LogMel[FrameCnt * NumMelBins + MelCnt];
  end;

  // ---- global max-8 clamp + (x + 4) / 4 ----
  Mel.ReSize(NumFrames, 1, NumMelBins);
  for FrameCnt := 0 to NumFramesM1 do
    for MelCnt := 0 to NumMelBinsM1 do
    begin
      V := LogMel[FrameCnt * NumMelBins + MelCnt];
      if V < MaxLog - 8.0 then V := MaxLog - 8.0;
      Mel.FData[FrameCnt * NumMelBins + MelCnt] := (V + 4.0) / 4.0;
    end;
end;

procedure WhisperLogMelFromWavFile(const FileName: string; Mel: TNNetVolume;
  NumMelBins: integer = 80; NumFrames: integer = 3000;
  Resample: boolean = True);
var
  Samples: TNNetVolume;
  SampleRate: integer;
begin
  Samples := TNNetVolume.Create;
  try
    if Resample then
      SampleRate := LoadWavResampledToVolume(FileName, Samples,
        csWhisperSampleRate)
    else
      SampleRate := LoadWav16ToVolume(FileName, Samples);
    if SampleRate <> csWhisperSampleRate then
      raise Exception.Create('WhisperLogMelFromWavFile: ' + FileName +
        ' is sampled at ' + IntToStr(SampleRate) + ' Hz - Whisper needs ' +
        '16000 Hz mono (pass Resample=True, or convert first, e.g. ' +
        'ffmpeg -i in.wav -ar 16000 -ac 1 out.wav).');
    ComputeWhisperLogMel(Samples, Mel, NumMelBins, NumFrames);
  finally
    Samples.Free;
  end;
end;

procedure BuildChromaFilterBank(Filt: TNNetVolume;
  SampleRate, NFFT, NumChroma: integer);
var
  NumBins, NumKept, c, b, NumChromaM1, NumBinsM1, NumKeptM1: integer;
  NumBinsM2: integer;
  Freq, Octave, StuttgartOver16, NumChroma2, Center, HalfWidth, FreqBin0: double;
  FreqBins: array of double;     // NFFT (the 0-Hz bin + NFFT-1 real bins)
  BinsWidth: array of double;    // NFFT
  Raw: array of double;          // (NumChroma x NFFT) before normalization
  ColNorm, Diff, Val, W, Tmp: double;
  Rolled: array of double;       // (NumChroma x NumKept) after roll + slice
  RollBy: integer;
begin
  // chroma_filter_bank(tuning=0, power=2, weighting=(5,2), start_at_c=True),
  // transformers.audio_utils. The filter bank has one column per FREQ BIN of
  // the full NFFT spectrum (NFFT entries: a synthetic 0-Hz bin prepended to the
  // NFFT-1 bins of linspace(0,sr,NFFT,endpoint=False)[1:]); the final result is
  // sliced to the first NFFT div 2 + 1 one-sided bins.
  NumBins := NFFT;                 // freq_bins length after the 0-Hz prepend
  NumKept := NFFT div 2 + 1;       // one-sided bins kept
  NumChromaM1 := NumChroma - 1;
  NumBinsM1 := NumBins - 1;
  NumKeptM1 := NumKept - 1;
  StuttgartOver16 := 440.0 / 16.0; // stuttgart_pitch (tuning 0) / 16

  // HF freq_bins = concatenate([fb0], NumChroma*octave(linspace[1:])): slot 0 is
  // the synthetic 0-Hz bin = fb(at freq index 1) - 1.5*NumChroma; slot i (>=1)
  // = NumChroma*octave(freq index i), freq index i = sr*i/NFFT.
  SetLength(FreqBins, NumBins);
  for b := 1 to NumBinsM1 do
  begin
    Freq := SampleRate * (b / NFFT);  // linspace(0,sr,NFFT,endpoint=False)[b]
    Octave := Ln(Freq / StuttgartOver16) / Ln(2.0);
    FreqBins[b] := NumChroma * Octave;
  end;
  FreqBin0 := FreqBins[1] - 1.5 * NumChroma;
  FreqBins[0] := FreqBin0;

  // bins_width = concatenate([max(diff(freq_bins),1.0), [1]])
  SetLength(BinsWidth, NumBins);
  NumBinsM2 := NumBinsM1 - 1;
  for b := 0 to NumBinsM2 do
  begin
    Diff := FreqBins[b + 1] - FreqBins[b];
    if Diff < 1.0 then Diff := 1.0;
    BinsWidth[b] := Diff;
  end;
  BinsWidth[NumBinsM1] := 1.0;

  NumChroma2 := Round(NumChroma / 2.0);

  // chroma_filters[c][b] = exp(-0.5 * (2*D/binwidth[b])^2) where
  //   D = ((freq_bins[b] - c + NumChroma2 + 10*NumChroma) mod NumChroma)
  //       - NumChroma2
  SetLength(Raw, NumChroma * NumBins);
  for c := 0 to NumChromaM1 do
    for b := 0 to NumBinsM1 do
    begin
      Val := FreqBins[b] - c;
      Val := Val + NumChroma2 + 10 * NumChroma;
      // python float remainder (result has sign of divisor; here divisor > 0)
      Val := Val - NumChroma * Floor(Val / NumChroma);
      Val := Val - NumChroma2;
      W := 2.0 * Val / BinsWidth[b];
      Raw[c * NumBins + b] := Exp(-0.5 * W * W);
    end;

  // normalize each COLUMN by its L2 norm (power=2)
  for b := 0 to NumBinsM1 do
  begin
    ColNorm := 0.0;
    for c := 0 to NumChromaM1 do
      ColNorm := ColNorm + Raw[c * NumBins + b] * Raw[c * NumBins + b];
    ColNorm := Sqrt(ColNorm);
    if ColNorm > 0.0 then
      for c := 0 to NumChromaM1 do
        Raw[c * NumBins + b] := Raw[c * NumBins + b] / ColNorm;
  end;

  // Gaussian frequency weighting: *= exp(-0.5*((freq_bins/NumChroma - 5)/2)^2)
  Center := 5.0;
  HalfWidth := 2.0;
  for b := 0 to NumBinsM1 do
  begin
    Tmp := (FreqBins[b] / NumChroma - Center) / HalfWidth;
    W := Exp(-0.5 * Tmp * Tmp);
    for c := 0 to NumChromaM1 do
      Raw[c * NumBins + b] := Raw[c * NumBins + b] * W;
  end;

  // start_at_c_chroma: roll rows by -3*(NumChroma div 12) along the chroma axis
  RollBy := 3 * (NumChroma div 12);   // shift UP by RollBy (np.roll(-RollBy))
  SetLength(Rolled, NumChroma * NumKept);
  for c := 0 to NumChromaM1 do
    for b := 0 to NumKeptM1 do
      Rolled[c * NumKept + b] :=
        Raw[(((c + RollBy) mod NumChroma)) * NumBins + b];

  Filt.ReSize(NumChroma, 1, NumKept);
  for c := 0 to NumChromaM1 do
    for b := 0 to NumKeptM1 do
      Filt.FData[c * NumKept + b] := Rolled[c * NumKept + b];
end;

procedure ComputeMusicgenMelodyChroma(Samples: TNNetVolume; Chroma: TNNetVolume;
  SampleRate: integer = 32000; NFFT: integer = 16384; Hop: integer = 4096;
  NumChroma: integer = 12);
var
  NumSamplesIn, NumSamples, PadLeft, NumFrames, NumBins: integer;
  NumSamplesM1, NFFTM1, NumBinsM1, NumChromaM1, NumFramesM1: integer;
  Window: array of double;        // periodic hann
  WinNorm: double;                // sqrt(sum(window^2))
  Filt: TNNetVolume;
  Re, Im: array of double;        // FFT scratch (NFFT)
  Power: array of double;         // one frame's power spectrum (NumBins)
  RawChroma: array of double;     // (NumFrames x NumChroma)
  FrameStart, FrameCnt, TapCnt, BinCnt, ChCnt, SrcIdx, ArgMax: integer;
  Acc, MaxVal, V: double;

  // reflect pad indexing (np pad mode 'reflect' = torch center pad): mirror
  // WITHOUT repeating the edge sample. Single bounce suffices (PadLeft < len).
  function WaveAt(Idx: integer): double;
  begin
    if Idx < 0 then Idx := -Idx;
    if Idx >= NumSamplesIn then Idx := 2 * NumSamplesIn - 2 - Idx;
    if Idx < 0 then Idx := 0;
    if Idx >= NumSamplesIn then Idx := NumSamplesIn - 1;
    Result := Samples.FData[Idx];
  end;

  // In-place radix-2 FFT (forward: X[k] = sum x[n] exp(-2*pi*i*k*n/N)). NFFT
  // is a power of two for the melody front-end (16384).
  procedure FFTRadix2;
  var
    i2, j2, lenF, halfF, k2, a2, b2: integer;
    halfFM1: integer;
    ang, wRe, wIm, wpRe, wpIm, tmpF: double;
    uRe, uIm, vRe, vIm: double;
  begin
    j2 := 0;
    for i2 := 1 to NFFTM1 do
    begin
      k2 := NFFT shr 1;
      while (j2 and k2) <> 0 do
      begin
        j2 := j2 and (not k2);
        k2 := k2 shr 1;
      end;
      j2 := j2 or k2;
      if i2 < j2 then
      begin
        tmpF := Re[i2]; Re[i2] := Re[j2]; Re[j2] := tmpF;
        tmpF := Im[i2]; Im[i2] := Im[j2]; Im[j2] := tmpF;
      end;
    end;
    lenF := 2;
    while lenF <= NFFT do
    begin
      halfF := lenF shr 1;
      halfFM1 := halfF - 1;
      ang := -2.0 * Pi / lenF;
      wpRe := Cos(ang);
      wpIm := Sin(ang);
      i2 := 0;
      while i2 < NFFT do
      begin
        wRe := 1.0; wIm := 0.0;
        for k2 := 0 to halfFM1 do
        begin
          a2 := i2 + k2;
          b2 := a2 + halfF;
          uRe := Re[a2]; uIm := Im[a2];
          vRe := Re[b2] * wRe - Im[b2] * wIm;
          vIm := Re[b2] * wIm + Im[b2] * wRe;
          Re[a2] := uRe + vRe; Im[a2] := uIm + vIm;
          Re[b2] := uRe - vRe; Im[b2] := uIm - vIm;
          tmpF := wRe * wpRe - wIm * wpIm;
          wIm := wRe * wpIm + wIm * wpRe;
          wRe := tmpF;
        end;
        i2 := i2 + lenF;
      end;
      lenF := lenF shl 1;
    end;
  end;

begin
  if NFFT < 2 then
    raise Exception.Create('ComputeMusicgenMelodyChroma: NFFT must be >= 2.');
  if (NFFT and (NFFT - 1)) <> 0 then
    raise Exception.Create('ComputeMusicgenMelodyChroma: NFFT must be a power of two.');
  NumSamplesIn := Samples.Size;
  // The feature extractor pads a too-short waveform to NFFT with zeros first
  // (centered). Mirror that by treating Samples as already >= NFFT here: in
  // practice the reference melody is longer than NFFT. If shorter, the reflect
  // pad below still produces a valid (if degenerate) spectrogram.
  if NumSamplesIn < 1 then
    raise Exception.Create('ComputeMusicgenMelodyChroma: empty waveform.');
  NFFTM1 := NFFT - 1;
  NumBins := NFFT div 2 + 1;
  NumBinsM1 := NumBins - 1;
  NumChromaM1 := NumChroma - 1;
  // center=True: torch reflect-pads NFFT div 2 on each side, then frames hop by
  // Hop. Frame count = 1 + NumSamplesIn div Hop.
  PadLeft := NFFT div 2;
  NumFrames := 1 + NumSamplesIn div Hop;
  NumFramesM1 := NumFrames - 1;
  NumSamples := NumSamplesIn;
  NumSamplesM1 := NumSamples - 1;

  // periodic hann window + its L2 energy (the torchaudio normalized=True norm)
  SetLength(Window, NFFT);
  WinNorm := 0.0;
  for TapCnt := 0 to NFFTM1 do
  begin
    Window[TapCnt] := 0.5 - 0.5 * Cos(2.0 * Pi * TapCnt / NFFT);
    WinNorm := WinNorm + Window[TapCnt] * Window[TapCnt];
  end;
  WinNorm := Sqrt(WinNorm);
  if WinNorm <= 0.0 then WinNorm := 1.0;

  Filt := TNNetVolume.Create;
  SetLength(Re, NFFT);
  SetLength(Im, NFFT);
  SetLength(Power, NumBins);
  SetLength(RawChroma, NumFrames * NumChroma);
  try
    BuildChromaFilterBank(Filt, SampleRate, NFFT, NumChroma);

    for FrameCnt := 0 to NumFramesM1 do
    begin
      // frame center at FrameCnt*Hop in the centered (padded) signal -> start at
      // FrameCnt*Hop - PadLeft in the ORIGINAL signal.
      FrameStart := FrameCnt * Hop - PadLeft;
      for TapCnt := 0 to NFFTM1 do
      begin
        SrcIdx := FrameStart + TapCnt;
        if (SrcIdx >= 0) and (SrcIdx <= NumSamplesM1) then
          V := Samples.FData[SrcIdx]
        else
          V := WaveAt(SrcIdx);
        Re[TapCnt] := V * Window[TapCnt];
        Im[TapCnt] := 0.0;
      end;
      FFTRadix2;
      // one-sided power spectrum, normalized by the window L2 energy
      for BinCnt := 0 to NumBinsM1 do
      begin
        V := (Re[BinCnt] / WinNorm);
        Acc := (Im[BinCnt] / WinNorm);
        Power[BinCnt] := V * V + Acc * Acc;
      end;
      // raw_chroma[c] = sum_b Filt[c][b] * Power[b]
      MaxVal := -1e30;
      for ChCnt := 0 to NumChromaM1 do
      begin
        Acc := 0.0;
        for BinCnt := 0 to NumBinsM1 do
          Acc := Acc + Filt.FData[ChCnt * NumBins + BinCnt] * Power[BinCnt];
        RawChroma[FrameCnt * NumChroma + ChCnt] := Acc;
        if Acc > MaxVal then MaxVal := Acc;
      end;
      // inf-norm normalize is monotonic, so the ARGMAX is unchanged by it; the
      // one-hot only needs the dominant chroma index.
      ArgMax := 0;
      MaxVal := RawChroma[FrameCnt * NumChroma];
      for ChCnt := 1 to NumChromaM1 do
        if RawChroma[FrameCnt * NumChroma + ChCnt] > MaxVal then
        begin
          MaxVal := RawChroma[FrameCnt * NumChroma + ChCnt];
          ArgMax := ChCnt;
        end;
      for ChCnt := 0 to NumChromaM1 do
        RawChroma[FrameCnt * NumChroma + ChCnt] := 0.0;
      RawChroma[FrameCnt * NumChroma + ArgMax] := 1.0;
    end;

    Chroma.ReSize(NumFrames, 1, NumChroma);
    for FrameCnt := 0 to NumFramesM1 do
      for ChCnt := 0 to NumChromaM1 do
        Chroma.FData[FrameCnt * NumChroma + ChCnt] :=
          RawChroma[FrameCnt * NumChroma + ChCnt];
  finally
    Filt.Free;
    SetLength(Re, 0); SetLength(Im, 0); SetLength(Power, 0);
    SetLength(RawChroma, 0); SetLength(Window, 0);
  end;
end;

procedure ISTFTOverlapAddReIm(Re, Im, Wave: TNNetVolume;
  NFFT: integer; HopLength: integer);
var
  NumFrames, NumBins, OutLen: integer;
  NFFTM1, NumBinsM1, OutLenM1, NumFramesM1: integer;
  FrameCnt, BinCnt, TapCnt, OutIdx, FrameStart: integer;
  Window: array of double;        // periodic hann (matches forward analysis)
  CosTab, SinTab: array of double; // (NumBins x NFFT) inverse-DFT twiddles
  AccSig: array of double;        // overlap-added windowed frames
  AccEnv: array of double;        // overlap-added squared-window envelope
  ReVal, ImVal, Sample, Scale, WinTap: double;
const
  csEnvEps = 1e-12;               // divide-by-zero guard for the COLA envelope
begin
  if NFFT < 2 then
    raise Exception.Create('ISTFTOverlapAdd: NFFT must be >= 2.');
  if HopLength < 1 then
    raise Exception.Create('ISTFTOverlapAdd: HopLength must be >= 1.');
  NumBins := NFFT div 2 + 1;
  NFFTM1 := NFFT - 1;
  NumBinsM1 := NumBins - 1;
  if (Re.Depth <> NumBins) or (Im.Depth <> NumBins) then
    raise Exception.Create('ISTFTOverlapAdd: Re/Im Depth must be NFFT div 2 + 1.');
  NumFrames := Re.SizeX;
  if Im.SizeX <> NumFrames then
    raise Exception.Create('ISTFTOverlapAdd: Re and Im must have the same frame count.');
  if NumFrames < 1 then
    raise Exception.Create('ISTFTOverlapAdd: at least one frame is required.');

  // ---- periodic hann synthesis window (matches the forward analysis) ----
  SetLength(Window, NFFT);
  for TapCnt := 0 to NFFTM1 do
    Window[TapCnt] := 0.5 - 0.5 * Cos(2.0 * Pi * TapCnt / NFFT);

  // ---- inverse-rDFT twiddle tables (same cos/sin convention as forward) ----
  // x[t] = (1/NFFT) * ( Re[0]
  //        + 2*sum_{k=1..NumBins-1}( Re[k]*cos + Im[k]*sin )  [k=NFFT/2 halved]
  //        ). The Im sign matches the forward SinTab (stored Im = +sum x*sin),
  // so feeding a forward STFT computed with this file's tables back in inverts.
  SetLength(CosTab, NumBins * NFFT);
  SetLength(SinTab, NumBins * NFFT);
  for BinCnt := 0 to NumBinsM1 do
    for TapCnt := 0 to NFFTM1 do
    begin
      CosTab[BinCnt * NFFT + TapCnt] :=
        Cos(2.0 * Pi * BinCnt * TapCnt / NFFT);
      SinTab[BinCnt * NFFT + TapCnt] :=
        Sin(2.0 * Pi * BinCnt * TapCnt / NFFT);
    end;

  OutLen := NFFT + (NumFrames - 1) * HopLength;
  OutLenM1 := OutLen - 1;
  NumFramesM1 := NumFrames - 1;
  SetLength(AccSig, OutLen);
  SetLength(AccEnv, OutLen);
  for OutIdx := 0 to OutLenM1 do
  begin
    AccSig[OutIdx] := 0.0;
    AccEnv[OutIdx] := 0.0;
  end;

  for FrameCnt := 0 to NumFramesM1 do
  begin
    FrameStart := FrameCnt * HopLength;
    for TapCnt := 0 to NFFTM1 do
    begin
      // inverse real DFT of this frame at sample TapCnt
      Sample := 0.0;
      for BinCnt := 0 to NumBinsM1 do
      begin
        ReVal := Re.FData[FrameCnt * NumBins + BinCnt];
        ImVal := Im.FData[FrameCnt * NumBins + BinCnt];
        // bins 0 and NFFT/2 are self-conjugate -> weight 1, the rest weight 2
        // (they stand in for their negative-frequency conjugate partner).
        if (BinCnt = 0) or ((BinCnt = NumBins - 1) and (NFFT mod 2 = 0)) then
          Scale := 1.0
        else
          Scale := 2.0;
        Sample := Sample + Scale *
          (ReVal * CosTab[BinCnt * NFFT + TapCnt] +
           ImVal * SinTab[BinCnt * NFFT + TapCnt]);
      end;
      Sample := Sample / NFFT;
      WinTap := Window[TapCnt];
      OutIdx := FrameStart + TapCnt;
      AccSig[OutIdx] := AccSig[OutIdx] + Sample * WinTap;
      AccEnv[OutIdx] := AccEnv[OutIdx] + WinTap * WinTap;
    end;
  end;

  // ---- COLA / window_sumsquare normalization (guarded) ----
  Wave.ReSize(OutLen, 1, 1);
  for OutIdx := 0 to OutLenM1 do
    if AccEnv[OutIdx] > csEnvEps then
      Wave.FData[OutIdx] := AccSig[OutIdx] / AccEnv[OutIdx]
    else
      Wave.FData[OutIdx] := 0.0;
end;

procedure ISTFTOverlapAdd(Mag, Phase, Wave: TNNetVolume;
  NFFT: integer; HopLength: integer);
var
  Re, Im: TNNetVolume;
  Idx: integer;
  MagSizeM1: integer;
  M, P: double;
begin
  if (Mag.SizeX <> Phase.SizeX) or (Mag.Depth <> Phase.Depth) then
    raise Exception.Create('ISTFTOverlapAdd: Mag and Phase must have the same shape.');
  Re := TNNetVolume.Create(Mag.SizeX, 1, Mag.Depth);
  Im := TNNetVolume.Create(Mag.SizeX, 1, Mag.Depth);
  try
    // Re[k] = Mag*cos(Phase), Im[k] = Mag*sin(Phase) - the forward table
    // convention (stored Im = +sum x*sin), so the round-trip is exact.
    MagSizeM1 := Mag.Size - 1;
    for Idx := 0 to MagSizeM1 do
    begin
      M := Mag.FData[Idx];
      P := Phase.FData[Idx];
      Re.FData[Idx] := M * Cos(P);
      Im.FData[Idx] := M * Sin(P);
    end;
    ISTFTOverlapAddReIm(Re, Im, Wave, NFFT, HopLength);
  finally
    Re.Free;
    Im.Free;
  end;
end;

end.
