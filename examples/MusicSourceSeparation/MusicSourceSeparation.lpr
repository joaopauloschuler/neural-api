program MusicSourceSeparation;
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

// MusicSourceSeparation -- the FIRST audio SOURCE-SEPARATION demo: one MIXED
// stereo track in, FOUR stems out (drums / bass / other / vocals). This is the
// time-domain Demucs (Defossez et al. 2019, arXiv:1911.13254) imported by
// BuildDemucsFromSafeTensors -- a symmetric 1-D conv U-Net:
//
//   mixed waveform (stereo)
//     -> encoder: depth blocks of (strided Conv1d -> ReLU -> 1x1 Conv1d -> GLU)
//        each saving a skip
//     -> bi-LSTM bottleneck over time + Linear(2C -> C)
//     -> decoder: depth blocks of (skip-add -> Conv1d -> GLU ->
//        ConvTranspose1d -> ReLU) mirroring the encoder
//     -> 4 stems x (audio_channels, time)
//
// With NO arguments this runs a SELF-CONTAINED pico smoke test on the
// committed random fixture (tests/fixtures/tiny_demucs.*) and a procedurally
// generated stereo mix -- no download, pure CPU, a fraction of a second --
// writing each separated stem to a 16-bit WAV (the stem channels are averaged
// to mono for the writer). With a real Demucs checkpoint directory as the
// first argument it builds the full model (same architecture, wider) and
// separates a synthesized stereo clip.
//
// Usage:
//   MusicSourceSeparation                  # pico smoke test (no download)
//   MusicSourceSeparation /path/to/demucs  # real checkpoint directory

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained, neuralaudio;

const
  PicoFixtureDir = '../../tests/fixtures/';
  StemNames: array[0..3] of string = ('drums', 'bass', 'other', 'vocals');

procedure GenerateMix(out Mix: array of TNeuralFloatDynArr;
  NumSamples, AudioChannels, SampleRate: integer);
var
  ch, i: integer;
begin
  // A simple two-tone stereo mix: a low "bass" sine plus a higher overtone,
  // slightly detuned per channel so the two channels differ.
  for ch := 0 to AudioChannels - 1 do
  begin
    SetLength(Mix[ch], NumSamples);
    for i := 0 to NumSamples - 1 do
      Mix[ch][i] :=
        0.25 * Sin(2 * Pi * (110 + ch * 3) * i / SampleRate) +
        0.15 * Sin(2 * Pi * (440 + ch * 5) * i / SampleRate);
  end;
end;

// Average a stem's channels to mono and write a 16-bit WAV via SaveVolumeToWav16.
procedure WriteStemWav(const Stem: TNNetFloatDynArr2D; const FileName: string;
  SampleRate: integer);
var
  V: TNNetVolume;
  AudioCh, Tn, c, t: integer;
  Acc: TNeuralFloat;
begin
  AudioCh := Length(Stem);
  if AudioCh = 0 then Exit;
  Tn := Length(Stem[0]);
  V := TNNetVolume.Create(Tn, 1, 1);
  try
    for t := 0 to Tn - 1 do
    begin
      Acc := 0;
      for c := 0 to AudioCh - 1 do Acc := Acc + Stem[c][t];
      V.FData[t] := Acc / AudioCh;
    end;
    SaveVolumeToWav16(V, FileName, SampleRate);
  finally
    V.Free;
  end;
end;

procedure Separate(Model: TNNetDemucs; const Mix: array of TNeuralFloatDynArr;
  const Title, OutPrefix: string);
var
  Stems: array of TNNetFloatDynArr2D;
  src, c, t, T2: integer;
  Energy: double;
  Fn: string;
begin
  WriteLn('--- ', Title, ' ---');
  WriteLn('  input channels: ', Length(Mix), '   samples: ', Length(Mix[0]));
  SetLength(Stems, Model.Config.Sources);
  Model.Separate(Mix, Stems);
  WriteLn('  stems out     : ', Length(Stems));
  for src := 0 to Length(Stems) - 1 do
  begin
    T2 := Length(Stems[src][0]);
    Energy := 0;
    for c := 0 to Length(Stems[src]) - 1 do
      for t := 0 to T2 - 1 do Energy := Energy + Sqr(Stems[src][c][t]);
    if (Length(Stems[src]) * T2) > 0 then
      Energy := Sqrt(Energy / (Length(Stems[src]) * T2));
    Fn := OutPrefix + '_' + StemNames[src] + '.wav';
    WriteStemWav(Stems[src], Fn, Model.Config.SamplingRate);
    WriteLn('  stem ', src, ' (', StemNames[src], '): ', T2,
      ' samples  rms=', Energy:0:6, '  -> ', Fn);
  end;
  WriteLn;
end;

var
  Model: TNNetDemucs;
  Config: TDemucsConfig;
  Mix: array of TNeuralFloatDynArr;
  CkptDir, SafePath, CfgPath: string;
begin
  WriteLn('Demucs music source separation - 4-stem demo');
  WriteLn('============================================');
  if ParamCount >= 1 then
  begin
    CkptDir := IncludeTrailingPathDelimiter(ParamStr(1));
    SafePath := CkptDir + 'model.safetensors';
    CfgPath := CkptDir + 'config.json';
    WriteLn('Loading real Demucs checkpoint from ', CkptDir);
    Model := BuildDemucsFromSafeTensors(SafePath, Config, CfgPath);
    try
      WriteLn(DemucsConfigToString(Config));
      WriteLn;
      // ~0.1 s of stereo audio; a stride^depth multiple keeps the U-Net aligned.
      SetLength(Mix, Config.AudioChannels);
      GenerateMix(Mix, Config.SamplingRate div 10, Config.AudioChannels,
        Config.SamplingRate);
      Separate(Model, Mix, 'real checkpoint, synthesized stereo mix', 'stem');
    finally
      Model.Free;
    end;
  end
  else
  begin
    SafePath := PicoFixtureDir + 'tiny_demucs.safetensors';
    CfgPath := PicoFixtureDir + 'tiny_demucs_config.json';
    if not FileExists(SafePath) then
    begin
      WriteLn('Pico fixture not found at ', SafePath);
      WriteLn('Run from the example directory, or pass a real Demucs ',
        'checkpoint directory as the first argument.');
      Halt(1);
    end;
    WriteLn('No checkpoint argument: running the self-contained pico smoke ',
      'test');
    WriteLn('on the committed RANDOM fixture (weights are untrained, so the ',
      'stems are not meant to resemble real instruments - this only ',
      'exercises');
    WriteLn('the full encoder -> bi-LSTM -> decoder separation pipeline).');
    WriteLn;
    Model := BuildDemucsFromSafeTensors(SafePath, Config, CfgPath);
    try
      WriteLn(DemucsConfigToString(Config));
      WriteLn;
      SetLength(Mix, Config.AudioChannels);
      GenerateMix(Mix, 56, Config.AudioChannels, Config.SamplingRate);
      Separate(Model, Mix, 'pico fixture, short stereo mix', 'stem');
      WriteLn('Pico smoke test complete (separation ran and wrote 4 stems).');
    finally
      Model.Free;
    end;
  end;
end.
