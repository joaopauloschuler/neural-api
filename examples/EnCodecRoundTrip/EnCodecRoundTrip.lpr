program EnCodecRoundTrip;
(*
Copyright (C) 2026 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Coded by Claude (AI).
*)

// EnCodecRoundTrip -- the FIRST audio-GENERATIVE demo: a NEURAL AUDIO CODEC
// that compresses a waveform to a stack of discrete codes and reconstructs
// it (waveform -> codes -> waveform), the inverse of the analysis-only
// Whisper / Wav2Vec2 demos. The new primitive is RESIDUAL VECTOR
// QUANTIZATION (RVQ): a cascade of codebooks where each stage quantizes the
// residual left by the previous one (the single-codebook
// TNNetVectorQuantizer is the one-stage special case).
//
//   waveform (mono)
//     -> conv ENCODER (causal weight-norm Conv1d + ELU + resnet blocks +
//        a residual LSTM bottleneck) -> latent frames
//     -> RVQ: per frame, argmin-L2 over codebook 0, subtract, argmin over
//        codebook 1, ... -> a [num_codebooks][num_frames] integer code grid
//     -> RVQ decode (sum of the chosen codebook vectors) -> latent
//     -> conv DECODER (mirror, ConvTranspose1d upsamplers) -> waveform
//
// With NO arguments this runs a SELF-CONTAINED pico smoke test on the
// committed random fixture (tests/fixtures/tiny_encodec.*) and a procedurally
// generated tone -- no download, pure CPU, a couple of seconds. With a real
// facebook/encodec_24khz checkpoint directory as the first argument it builds
// the full codec (the architecture is identical; only the widths differ) and
// round-trips a synthesized 24 kHz tone, reporting the compression ratio and
// reconstruction error.
//
// Usage:
//   EnCodecRoundTrip                         # pico smoke test (no download)
//   EnCodecRoundTrip /path/to/encodec_24khz  # real checkpoint directory

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained;

const
  PicoFixtureDir = '../../tests/fixtures/';

procedure GenerateTone(out Wave: TNeuralFloatDynArr; NumSamples: integer;
  SampleRate, Freq: integer);
var
  i: integer;
begin
  SetLength(Wave, NumSamples);
  for i := 0 to NumSamples - 1 do
    Wave[i] := 0.3 * Sin(2 * Pi * Freq * i / SampleRate);
end;

procedure RoundTrip(Model: TEnCodecModel; const Wave: TNeuralFloatDynArr;
  const Title: string);
var
  Codes: array of TNeuralIntegerArray;
  Recon: TNeuralFloatDynArr;
  Frames, i, q, NonZero: integer;
  MaxDiff, RMS, Diff: double;
  RawBits, CodeBits: int64;
begin
  WriteLn('--- ', Title, ' ---');
  WriteLn('  input samples : ', Length(Wave));
  Model.EncodeAudioToCodes(Wave, Codes, Frames);
  WriteLn('  RVQ codebooks : ', Length(Codes));
  WriteLn('  latent frames : ', Frames);
  // First codebook codes (a peek at the discrete representation).
  Write('  codebook 0 [t=0..', Min(7, Frames - 1), ']:');
  for i := 0 to Min(7, Frames - 1) do Write(' ', Codes[0][i]);
  WriteLn;
  Model.DecodeCodesToAudio(Codes, Recon, 0);
  // Reconstruction error.
  MaxDiff := 0; RMS := 0; NonZero := 0;
  for i := 0 to Min(Length(Wave), Length(Recon)) - 1 do
  begin
    Diff := Abs(Recon[i] - Wave[i]);
    if Diff > MaxDiff then MaxDiff := Diff;
    RMS := RMS + Diff * Diff;
    if Recon[i] <> 0 then Inc(NonZero);
  end;
  if Length(Recon) > 0 then RMS := Sqrt(RMS / Length(Recon));
  // Rough compression accounting: codes need ceil(log2(codebook_size)) bits
  // each, raw audio 16 bits/sample.
  RawBits := int64(Length(Wave)) * 16;
  CodeBits := int64(Length(Codes)) * int64(Frames) *
    Ceil(Log2(Model.Config.CodebookSize));
  WriteLn('  recon samples : ', Length(Recon), ' (', NonZero, ' non-zero)');
  WriteLn('  raw bits      : ', RawBits, '   code bits: ', CodeBits);
  if CodeBits > 0 then
    WriteLn('  compression   : ', (RawBits / CodeBits):0:2, 'x');
  WriteLn('  recon |diff|  : max=', MaxDiff:0:6, '  rms=', RMS:0:6);
  WriteLn;
end;

var
  Model: TEnCodecModel;
  Config: TEnCodecConfig;
  Wave: TNeuralFloatDynArr;
  CkptDir, SafePath, CfgPath: string;
begin
  WriteLn('EnCodec neural audio codec - round-trip demo');
  WriteLn('============================================');
  if ParamCount >= 1 then
  begin
    // Real checkpoint directory.
    CkptDir := IncludeTrailingPathDelimiter(ParamStr(1));
    SafePath := CkptDir + 'model.safetensors';
    CfgPath := CkptDir + 'config.json';
    WriteLn('Loading real EnCodec checkpoint from ', CkptDir);
    Model := BuildEnCodecFromSafeTensors(SafePath, Config, CfgPath);
    try
      WriteLn(EnCodecConfigToString(Config));
      WriteLn;
      // A short tone at the model sampling rate (about 1/20 s to keep CPU
      // and memory bounded). 320-sample multiple keeps every conv positive.
      GenerateTone(Wave, 3200, Config.SamplingRate, 440);
      RoundTrip(Model, Wave, 'real checkpoint, 440 Hz tone');
    finally
      Model.Free;
    end;
  end
  else
  begin
    // Pico smoke test on the committed random fixture (no download).
    SafePath := PicoFixtureDir + 'tiny_encodec.safetensors';
    CfgPath := PicoFixtureDir + 'tiny_encodec_config.json';
    if not FileExists(SafePath) then
    begin
      WriteLn('Pico fixture not found at ', SafePath);
      WriteLn('Run from the example directory, or pass a real EnCodec ',
        'checkpoint directory as the first argument.');
      Halt(1);
    end;
    WriteLn('No checkpoint argument: running the self-contained pico smoke ',
      'test');
    WriteLn('on the committed RANDOM fixture (weights are untrained, so the ',
      'reconstruction is not meant to resemble the input - this only ',
      'exercises');
    WriteLn('the full encode->RVQ->decode pipeline end to end).');
    WriteLn;
    Model := BuildEnCodecFromSafeTensors(SafePath, Config, CfgPath);
    try
      WriteLn(EnCodecConfigToString(Config));
      WriteLn;
      GenerateTone(Wave, 56, Config.SamplingRate, 200);
      RoundTrip(Model, Wave, 'pico fixture, short tone');
      WriteLn('Pico smoke test complete (the round trip ran without error).');
    finally
      Model.Free;
    end;
  end;
end.
