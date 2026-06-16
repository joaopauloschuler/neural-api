program MimiCodec;
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

// MimiCodec -- a round-trip demo of the Mimi streaming neural audio codec
// (kyutai/mimi, the 12.5 Hz tokenizer behind Moshi / Kyutai-TTS / Sesame CSM).
// Mimi extends the convolutional SEANet + RVQ EnCodec design with a small
// CAUSAL RoPE transformer bottleneck (one after the conv encoder, one before
// the conv decoder) and a SPLIT residual vector quantizer (a semantic codebook
// concatenated with an acoustic RVQ stack):
//
//   waveform (mono)
//     -> conv ENCODER (causal Conv1d + ELU + resnet blocks)
//     -> RoPE TRANSFORMER (pre-norm attention + GELU MLP, LayerScale)
//     -> downsample Conv1d (-> 12.5 Hz)
//     -> SPLIT RVQ: semantic codebook(s) + acoustic RVQ cascade
//          -> a [num_quantizers][num_frames] integer code grid
//     -> RVQ decode -> upsample ConvTranspose1d -> RoPE TRANSFORMER
//     -> conv DECODER (ConvTranspose1d upsamplers) -> waveform
//
// With NO arguments this runs a SELF-CONTAINED pico smoke test on the
// committed random fixture (tests/fixtures/tiny_mimi.*) and a procedurally
// generated tone -- no download, pure CPU, a couple of seconds -- and writes
// the resynthesized clip to a 16-bit WAV via SaveVolumeToWav16. With a real
// kyutai/mimi checkpoint directory as the first argument it builds the full
// codec (same architecture, wider) and round-trips a synthesized tone.
//
// Usage:
//   MimiCodec                       # pico smoke test (no download)
//   MimiCodec /path/to/kyutai_mimi  # real checkpoint directory

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

procedure RoundTrip(Model: TNNetMimi; const Wave: TNeuralFloatDynArr;
  const Title, WavOut: string);
var
  Codes: TNNetIntArr2D;
  Recon: TNeuralFloatDynArr;
  Vol: TNNetVolume;
  Frames, i, NSem, NAco: integer;
  MaxDiff, RMS, Diff: double;
  RawBits, CodeBits: int64;
begin
  WriteLn('--- ', Title, ' ---');
  WriteLn('  input samples : ', Length(Wave));
  Model.Encode(Wave, Codes, Frames);
  NSem := Model.Config.NumSemanticQuantizers;
  NAco := Length(Codes) - NSem;
  WriteLn('  RVQ codebooks : ', Length(Codes), ' (', NSem, ' semantic + ',
    NAco, ' acoustic)');
  WriteLn('  latent frames : ', Frames, '  (frame rate ',
    Model.Config.FrameRate:0:1, ' Hz)');
  Write('  semantic code [t=0..', Min(7, Frames - 1), ']:');
  for i := 0 to Min(7, Frames - 1) do Write(' ', Codes[0][i]);
  WriteLn;
  Model.Decode(Codes, Recon);
  MaxDiff := 0; RMS := 0;
  for i := 0 to Min(Length(Wave), Length(Recon)) - 1 do
  begin
    Diff := Abs(Recon[i] - Wave[i]);
    if Diff > MaxDiff then MaxDiff := Diff;
    RMS := RMS + Diff * Diff;
  end;
  if Length(Recon) > 0 then RMS := Sqrt(RMS / Length(Recon));
  RawBits := int64(Length(Wave)) * 16;
  CodeBits := int64(Length(Codes)) * int64(Frames) *
    Ceil(Log2(Model.Config.CodebookSize));
  WriteLn('  recon samples : ', Length(Recon));
  WriteLn('  raw bits      : ', RawBits, '   code bits: ', CodeBits);
  if CodeBits > 0 then
    WriteLn('  compression   : ', (RawBits / CodeBits):0:2, 'x');
  WriteLn('  recon |diff|  : max=', MaxDiff:0:6, '  rms=', RMS:0:6);
  // Resynthesize to a WAV file.
  if WavOut <> '' then
  begin
    Vol := TNNetVolume.Create;
    try
      Vol.ReSize(Length(Recon), 1, 1);
      for i := 0 to Length(Recon) - 1 do Vol.FData[i] := Recon[i];
      SaveVolumeToWav16(Vol, WavOut, Model.Config.SamplingRate);
      WriteLn('  wrote WAV     : ', WavOut, ' (', Length(Recon), ' samples @ ',
        Model.Config.SamplingRate, ' Hz)');
    finally
      Vol.Free;
    end;
  end;
  WriteLn;
end;

var
  Model: TNNetMimi;
  Config: TMimiConfig;
  Wave: TNeuralFloatDynArr;
  CkptDir, SafePath, CfgPath: string;
begin
  WriteLn('Mimi streaming neural audio codec - round-trip demo');
  WriteLn('===================================================');
  if ParamCount >= 1 then
  begin
    CkptDir := IncludeTrailingPathDelimiter(ParamStr(1));
    SafePath := CkptDir + 'model.safetensors';
    CfgPath := CkptDir + 'config.json';
    WriteLn('Loading real Mimi checkpoint from ', CkptDir);
    Model := BuildMimiFromSafeTensors(SafePath, Config, CfgPath);
    try
      WriteLn(MimiConfigToString(Config));
      WriteLn;
      GenerateTone(Wave, 24000, Config.SamplingRate, 440);
      RoundTrip(Model, Wave, 'real checkpoint, 440 Hz tone',
        'mimi_roundtrip.wav');
    finally
      Model.Free;
    end;
  end
  else
  begin
    SafePath := PicoFixtureDir + 'tiny_mimi.safetensors';
    CfgPath := PicoFixtureDir + 'tiny_mimi_config.json';
    if not FileExists(SafePath) then
    begin
      WriteLn('Pico fixture not found at ', SafePath);
      WriteLn('Run from the example directory, or pass a real kyutai/mimi ',
        'checkpoint directory as the first argument.');
      Halt(1);
    end;
    WriteLn('No checkpoint argument: running the self-contained pico smoke ',
      'test on the');
    WriteLn('committed RANDOM fixture (untrained weights, so the ',
      'reconstruction is not');
    WriteLn('meant to resemble the input - this exercises the full ',
      'encode->split-RVQ->decode pipeline end to end and writes a WAV).');
    WriteLn;
    Model := BuildMimiFromSafeTensors(SafePath, Config, CfgPath);
    try
      WriteLn(MimiConfigToString(Config));
      WriteLn;
      GenerateTone(Wave, 144, Config.SamplingRate, 100);
      RoundTrip(Model, Wave, 'pico fixture, short tone',
        'mimi_pico_roundtrip.wav');
      WriteLn('Pico smoke test complete (the round trip ran without error).');
    finally
      Model.Free;
    end;
  end;
end.
