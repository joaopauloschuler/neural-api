program KokoroTTS;
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

// KokoroTTS -- a phonemes -> waveform smoke demo of the Kokoro / StyleTTS2
// text-to-speech importer (hexgrad/Kokoro-82M, Apache-2.0). Kokoro is a
// StyleTTS2 model: a StyleDim-d voice/style vector is AdaIN-injected into a
// duration predictor, F0/energy predictors and an iSTFTNet decoder that
// predicts a magnitude + phase spectrogram and runs an INVERSE STFT to the
// waveform:
//
//   phoneme ids + style vector
//     -> text encoder (embed -> conv -> ReLU)
//     -> duration predictor (AdaIN(s_pred) -> conv -> proj) -> durations
//     -> length regulator (monotonic expansion along time)
//     -> F0 + energy predictors (AdaIN(s_pred) -> conv -> proj)
//     -> iSTFTNet decoder (AdaIN(s_dec) -> magnitude + phase -> ISTFT)
//     -> waveform -> 16-bit WAV (SaveVolumeToWav16)
//
// SCOPE v1: a single deterministic forward graph with the reference style
// vector as an EXPLICIT input. The grapheme->phoneme (misaki/espeak)
// front-end is OUT OF SCOPE - pre-phonemized integer ids are the input.
//
// With NO arguments this runs a SELF-CONTAINED pico smoke test on the
// committed RANDOM fixture (tests/fixtures/tiny_kokoro.*) - no download, pure
// CPU, a fraction of a second - and writes the synthesized clip to a WAV.
//
// Usage:
//   KokoroTTS   # pico smoke test (no download), writes kokoro_pico.wav

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained, neuralaudio;

const
  PicoFixtureDir = '../../tests/fixtures/';

var
  Model: TNNetKokoro;
  Config: TKokoroConfig;
  SafePath, CfgPath: string;
  Ids: array of integer;
  Style, Wave: TNeuralFloatDynArr;
  i: integer;
  MaxAbs: double;
begin
  WriteLn('Kokoro / StyleTTS2 text-to-speech - phonemes -> waveform smoke');
  WriteLn('==============================================================');
  SafePath := PicoFixtureDir + 'tiny_kokoro.safetensors';
  CfgPath := PicoFixtureDir + 'tiny_kokoro_config.json';
  if not FileExists(SafePath) then
  begin
    WriteLn('Pico fixture not found at ', SafePath);
    WriteLn('Run from the example directory (examples/KokoroTTS).');
    Halt(1);
  end;
  WriteLn('Running the self-contained pico smoke test on the committed RANDOM');
  WriteLn('fixture (untrained weights, so the audio is not meant to be ',
    'intelligible -');
  WriteLn('this exercises the full StyleTTS2 forward graph end to end and ',
    'writes a WAV).');
  WriteLn;

  Model := BuildKokoroFromSafeTensors(SafePath, Config, CfgPath);
  try
    WriteLn(KokoroConfigToString(Config));
    WriteLn;
    // Pre-phonemized integer ids (the front-end is out of scope) + an
    // explicit StyleDim-d reference style vector.
    SetLength(Ids, 6);
    Ids[0] := 3; Ids[1] := 7; Ids[2] := 1;
    Ids[3] := 12; Ids[4] := 5; Ids[5] := 9;
    SetLength(Style, Config.StyleDim);
    for i := 0 to Config.StyleDim - 1 do
      Style[i] := 0.5 * Sin(0.7 * i + 1.0);   // a deterministic pseudo-voice

    Write('  phoneme ids   :');
    for i := 0 to Length(Ids) - 1 do Write(' ', Ids[i]);
    WriteLn;
    Model.Synthesize(Ids, Style, Wave);
    MaxAbs := 0;
    for i := 0 to Length(Wave) - 1 do
      if Abs(Wave[i]) > MaxAbs then MaxAbs := Abs(Wave[i]);
    WriteLn('  output samples: ', Length(Wave), '  (max|wave| = ',
      MaxAbs:0:4, ')');

    Model.SynthesizeToWav(Ids, Style, 'kokoro_pico.wav');
    WriteLn('  wrote WAV     : kokoro_pico.wav (', Length(Wave),
      ' samples @ ', Config.SamplingRate, ' Hz)');
    WriteLn;
    WriteLn('Pico smoke test complete (the forward graph ran without error).');
  finally
    Model.Free;
  end;
end.
