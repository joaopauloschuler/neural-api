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
  Classes, SysUtils, neuralvolume;

// Reads a 16-bit PCM RIFF/WAVE file into Samples as mono floats in [-1, 1)
// (multi-channel input is averaged). Samples is resized to (N, 1, 1).
// Returns the sample rate in Hz. Raises Exception on anything that is not
// an uncompressed 16-bit PCM WAV.
function LoadWav16ToVolume(const FileName: string;
  Samples: TNNetVolume): integer;

// HF WhisperFeatureExtractor log-mel spectrogram (see the unit header).
// Samples: mono waveform at 16 kHz, any length (padded/truncated to
// NumFrames*160 samples). Mel is resized to (NumFrames, 1, NumMelBins).
procedure ComputeWhisperLogMel(Samples: TNNetVolume; Mel: TNNetVolume;
  NumMelBins: integer = 80; NumFrames: integer = 3000);

// Convenience wrapper: LoadWav16ToVolume + ComputeWhisperLogMel. The WAV
// must already be sampled at 16000 Hz (no resampler in this v1 - raises
// Exception otherwise; convert first, e.g. ffmpeg -ar 16000 -ac 1).
procedure WhisperLogMelFromWavFile(const FileName: string; Mel: TNNetVolume;
  NumMelBins: integer = 80; NumFrames: integer = 3000);

implementation

const
  csWhisperNFFT = 400;      // STFT window (25 ms at 16 kHz)
  csWhisperHop = 160;       // hop (10 ms at 16 kHz)
  csWhisperSampleRate = 16000;
  csWhisperMaxFreq = 8000.0;
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
        for FrameCnt := 0 to NumFrames - 1 do
        begin
          Acc := 0;
          for ChCnt := 0 to NumChannels - 1 do
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

  // ---- pad / truncate to the fixed 30 s context ----
  SetLength(Wave, NumSamples);
  for SampleCnt := 0 to NumSamples - 1 do
    if SampleCnt < Samples.Size then
      Wave[SampleCnt] := Samples.FData[SampleCnt]
    else
      Wave[SampleCnt] := 0.0;

  // ---- periodic hann window (transformers window_function default) ----
  SetLength(Window, csWhisperNFFT);
  for TapCnt := 0 to csWhisperNFFT - 1 do
    Window[TapCnt] := 0.5 - 0.5 * Cos(2.0 * Pi * TapCnt / csWhisperNFFT);

  // ---- DFT twiddle tables (400 is not a power of two - direct rDFT) ----
  SetLength(CosTab, NumBins * csWhisperNFFT);
  SetLength(SinTab, NumBins * csWhisperNFFT);
  for BinCnt := 0 to NumBins - 1 do
    for TapCnt := 0 to csWhisperNFFT - 1 do
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
  for MelCnt := 0 to NumMelBins + 1 do
    FilterFreqs[MelCnt] := MelToHertzSlaney(
      MelMin + (MelMax - MelMin) * MelCnt / (NumMelBins + 1));
  SetLength(FilterBank, NumBins * NumMelBins);
  for BinCnt := 0 to NumBins - 1 do
  begin
    FFTFreq := (csWhisperSampleRate div 2) * BinCnt / (NumBins - 1);
    for MelCnt := 0 to NumMelBins - 1 do
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
  SetLength(Power, NumBins);
  SetLength(LogMel, NumStftFrames * NumMelBins);
  MaxLog := -1e30;
  for FrameCnt := 0 to NumStftFrames - 1 do
  begin
    FrameStart := FrameCnt * csWhisperHop - (csWhisperNFFT div 2);
    // The 30 s zero-padding tail produces all-zero frames whose mel rows
    // are exactly log10(mel_floor) - skip their O(bins*taps) DFT.
    AllZero := true;
    for TapCnt := 0 to csWhisperNFFT - 1 do
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
      for MelCnt := 0 to NumMelBins - 1 do
        LogMel[FrameCnt * NumMelBins + MelCnt] := Ln(csWhisperMelFloor) / Ln(10.0);
    end
    else
    begin
      for BinCnt := 0 to NumBins - 1 do
      begin
        ReAcc := 0.0;
        ImAcc := 0.0;
        for TapCnt := 0 to csWhisperNFFT - 1 do
        begin
          V := WaveAt(FrameStart + TapCnt) * Window[TapCnt];
          ReAcc := ReAcc + V * CosTab[BinCnt * csWhisperNFFT + TapCnt];
          ImAcc := ImAcc + V * SinTab[BinCnt * csWhisperNFFT + TapCnt];
        end;
        Power[BinCnt] := ReAcc * ReAcc + ImAcc * ImAcc;
      end;
      for MelCnt := 0 to NumMelBins - 1 do
      begin
        Acc := 0.0;
        for BinCnt := 0 to NumBins - 1 do
          Acc := Acc + FilterBank[BinCnt * NumMelBins + MelCnt] *
            Power[BinCnt];
        if Acc < csWhisperMelFloor then Acc := csWhisperMelFloor;
        LogMel[FrameCnt * NumMelBins + MelCnt] := Ln(Acc) / Ln(10.0);
      end;
    end;
    for MelCnt := 0 to NumMelBins - 1 do
      if LogMel[FrameCnt * NumMelBins + MelCnt] > MaxLog then
        MaxLog := LogMel[FrameCnt * NumMelBins + MelCnt];
  end;

  // ---- global max-8 clamp + (x + 4) / 4 ----
  Mel.ReSize(NumFrames, 1, NumMelBins);
  for FrameCnt := 0 to NumFrames - 1 do
    for MelCnt := 0 to NumMelBins - 1 do
    begin
      V := LogMel[FrameCnt * NumMelBins + MelCnt];
      if V < MaxLog - 8.0 then V := MaxLog - 8.0;
      Mel.FData[FrameCnt * NumMelBins + MelCnt] := (V + 4.0) / 4.0;
    end;
end;

procedure WhisperLogMelFromWavFile(const FileName: string; Mel: TNNetVolume;
  NumMelBins: integer = 80; NumFrames: integer = 3000);
var
  Samples: TNNetVolume;
  SampleRate: integer;
begin
  Samples := TNNetVolume.Create;
  try
    SampleRate := LoadWav16ToVolume(FileName, Samples);
    if SampleRate <> csWhisperSampleRate then
      raise Exception.Create('WhisperLogMelFromWavFile: ' + FileName +
        ' is sampled at ' + IntToStr(SampleRate) + ' Hz - Whisper needs ' +
        '16000 Hz mono (convert first, e.g. ffmpeg -i in.wav -ar 16000 ' +
        '-ac 1 out.wav).');
    ComputeWhisperLogMel(Samples, Mel, NumMelBins, NumFrames);
  finally
    Samples.Free;
  end;
end;

end.
