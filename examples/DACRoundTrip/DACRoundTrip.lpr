program DACRoundTrip;
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

// DACRoundTrip -- a NEURAL AUDIO CODEC demo for DAC (the Descript Audio Codec,
// model_type "dac", the RVQGAN-lineage codec). It compresses a waveform to a
// stack of discrete codes and reconstructs it (waveform -> codes -> waveform),
// then writes the reconstruction to a 16-bit WAV.
//
// DAC differs from the EnCodec / Mimi codecs in three ways:
//   * SNAKE activations (x + (1/(alpha+1e-9))*sin(alpha*x)^2) with a learnable
//     per-channel alpha, throughout the conv stacks;
//   * SYMMETRIC (non-causal) conv padding;
//   * a FACTORIZED, L2-NORMALIZED Residual Vector Quantizer: each quantizer
//     projects the latent down to a small codebook_dim, L2-normalizes it and
//     the codebook, looks up the nearest entry (argmax cosine), then projects
//     it back up; the residual is subtracted in the full hidden space.
//
//   waveform (mono)
//     -> conv ENCODER (Snake + symmetric Conv1d + residual units) -> latent
//     -> factorized L2-normalized RVQ -> [num_codebooks][num_frames] code grid
//     -> RVQ decode (sum of out_proj(codebook[code])) -> latent
//     -> conv DECODER (Snake + ConvTranspose1d upsamplers + Tanh) -> waveform
//
// With NO arguments this runs a SELF-CONTAINED pico smoke test on the committed
// random fixture (tests/fixtures/tiny_dac.*) and a procedurally generated tone
// -- no download, pure CPU, a couple of seconds. With a real descript/dac_*
// checkpoint directory as the first argument it builds the full codec (the
// architecture is identical; only the widths differ) and round-trips a
// synthesized tone, reporting the compression ratio and reconstruction error.
// The reconstruction is written to dac_recon.wav in the current directory.
//
// Usage:
//   DACRoundTrip                        # pico smoke test (no download)
//   DACRoundTrip /path/to/dac_16khz     # real checkpoint directory

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained, neuralaudio;

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

procedure SaveWav(const Wave: TNeuralFloatDynArr; const FileName: string;
  SampleRate: integer);
var
  Vol: TNNetVolume;
  i: integer;
begin
  Vol := TNNetVolume.Create(Length(Wave), 1, 1);
  try
    for i := 0 to Length(Wave) - 1 do Vol.FData[i] := Wave[i];
    SaveVolumeToWav16(Vol, FileName, SampleRate);
    WriteLn('  wrote WAV     : ', FileName, ' (', Length(Wave),
      ' samples @ ', SampleRate, ' Hz)');
  finally
    Vol.Free;
  end;
end;

procedure RoundTrip(Model: TNNetDAC; const Wave: TNeuralFloatDynArr;
  const Title, WavName: string; SampleRate: integer);
var
  Codes: array of TNeuralIntegerArray;
  Recon: TNeuralFloatDynArr;
  Frames, i, NonZero: integer;
  MaxDiff, RMS, Diff: double;
  RawBits, CodeBits: int64;
begin
  WriteLn('--- ', Title, ' ---');
  WriteLn('  input samples : ', Length(Wave));
  Model.Encode(Wave, Codes, Frames);
  WriteLn('  RVQ codebooks : ', Length(Codes));
  WriteLn('  latent frames : ', Frames);
  Write('  codebook 0 [t=0..', Min(7, Frames - 1), ']:');
  for i := 0 to Min(7, Frames - 1) do Write(' ', Codes[0][i]);
  WriteLn;
  Model.Decode(Codes, Recon, 0);
  MaxDiff := 0; RMS := 0; NonZero := 0;
  for i := 0 to Min(Length(Wave), Length(Recon)) - 1 do
  begin
    Diff := Abs(Recon[i] - Wave[i]);
    if Diff > MaxDiff then MaxDiff := Diff;
    RMS := RMS + Diff * Diff;
    if Recon[i] <> 0 then Inc(NonZero);
  end;
  if Length(Recon) > 0 then RMS := Sqrt(RMS / Length(Recon));
  RawBits := int64(Length(Wave)) * 16;
  CodeBits := int64(Length(Codes)) * int64(Frames) *
    Ceil(Log2(Model.Config.CodebookSize));
  WriteLn('  recon samples : ', Length(Recon), ' (', NonZero, ' non-zero)');
  WriteLn('  raw bits      : ', RawBits, '   code bits: ', CodeBits);
  if CodeBits > 0 then
    WriteLn('  compression   : ', (RawBits / CodeBits):0:2, 'x');
  WriteLn('  recon |diff|  : max=', MaxDiff:0:6, '  rms=', RMS:0:6);
  if Length(Recon) > 0 then SaveWav(Recon, WavName, SampleRate);
  WriteLn;
end;

var
  Model: TNNetDAC;
  Config: TDACConfig;
  Wave: TNeuralFloatDynArr;
  CkptDir, SafePath, CfgPath: string;
begin
  WriteLn('DAC (Descript Audio Codec) neural audio codec - round-trip demo');
  WriteLn('===============================================================');
  if ParamCount >= 1 then
  begin
    CkptDir := IncludeTrailingPathDelimiter(ParamStr(1));
    SafePath := CkptDir + 'model.safetensors';
    CfgPath := CkptDir + 'config.json';
    WriteLn('Loading real DAC checkpoint from ', CkptDir);
    Model := BuildDACFromSafeTensors(SafePath, Config, CfgPath);
    try
      WriteLn(DACConfigToString(Config));
      WriteLn;
      // ~1/20 s tone; a hop_length multiple keeps every conv positive.
      GenerateTone(Wave, Config.HopLength * 20, Config.SamplingRate, 440);
      RoundTrip(Model, Wave, 'real checkpoint, 440 Hz tone', 'dac_recon.wav',
        Config.SamplingRate);
    finally
      Model.Free;
    end;
  end
  else
  begin
    SafePath := PicoFixtureDir + 'tiny_dac.safetensors';
    CfgPath := PicoFixtureDir + 'tiny_dac_config.json';
    if not FileExists(SafePath) then
    begin
      WriteLn('Pico fixture not found at ', SafePath);
      WriteLn('Run from the example directory, or pass a real DAC ',
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
    Model := BuildDACFromSafeTensors(SafePath, Config, CfgPath);
    try
      WriteLn(DACConfigToString(Config));
      WriteLn;
      GenerateTone(Wave, Config.HopLength * 12, Config.SamplingRate, 200);
      RoundTrip(Model, Wave, 'pico fixture, short tone', 'dac_recon.wav',
        Config.SamplingRate);
      WriteLn('Pico smoke test complete (the round trip ran without error).');
    finally
      Model.Free;
    end;
  end;
end.
