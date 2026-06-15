program TextToSpeech;
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

// TextToSpeech -- the FIRST text-to-speech demo: VITS / MMS-TTS
// (facebook/mms-tts-*, kakao-enterprise/vits-ljs) synthesizes a waveform
// end-to-end from a sequence of token ids:
//
//   token ids
//     -> text_encoder (relative-position transformer) -> per-token prior
//        mean/log-variance + hidden;
//     -> deterministic duration_predictor -> frames-per-token;
//     -> length regulator EXPANDS the prior along time;
//     -> prior_latents = mean + z*exp(logvar)*noise_scale (z = prior noise);
//     -> normalizing FLOW (RealNVP/Glow additive coupling, run in reverse);
//     -> HiFi-GAN DECODER (the same generator as BuildHiFiGANFromSafeTensors)
//     -> raw mono waveform written to a 16-bit PCM WAV.
//
// With NO arguments this runs a SELF-CONTAINED pico smoke test on the
// committed RANDOM fixture (tests/fixtures/tiny_vits.*): it builds the model,
// synthesizes a short utterance from a fixed token sequence and a fixed noise
// tensor, and writes the waveform to /tmp/tts_smoke.wav -- no download, pure
// CPU, well under a minute. The weights are untrained random, so the output
// is NOISE, not intelligible speech; this only exercises the importer +
// synthesis end to end. The ~5 min / ulimit budget note in the README applies
// to real downloaded checkpoints (this smoke is a fraction of a second).
//
// Usage:
//   TextToSpeech                 # pico smoke test (no download)

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained,
  neuralaudio;

const
  PicoFixtureDir = '../../tests/fixtures/';
  OutWav = '/tmp/tts_smoke.wav';

var
  Model: TNNetVits;
  Config: TVitsConfig;
  SafePath, CfgPath: string;
  Ids: array of integer;
  Z: array of TNeuralFloatDynArr;
  Wave: TNeuralFloatDynArr;
  WaveVol: TNNetVolume;
  Durations: array of integer;
  PriorMean, PriorLogVar: array of TNeuralFloatDynArr;
  OutLen, Flow, i, c, t, TotalDur: integer;
begin
  WriteLn('VITS / MMS-TTS text-to-speech - end-to-end synthesis smoke test');
  WriteLn('===============================================================');
  SafePath := PicoFixtureDir + 'tiny_vits.safetensors';
  CfgPath := PicoFixtureDir + 'tiny_vits_config.json';
  if not FileExists(SafePath) then
  begin
    WriteLn('Pico fixture not found at ', SafePath);
    WriteLn('Run from the example directory (examples/TextToSpeech).');
    Halt(1);
  end;
  WriteLn('No checkpoint argument: running the self-contained pico smoke test');
  WriteLn('on the committed RANDOM fixture (weights are untrained, so the');
  WriteLn('output is noise, not speech - this only exercises the importer +');
  WriteLn('synthesis end to end).');
  WriteLn;

  RandSeed := 424242;
  Model := nil;
  WaveVol := TNNetVolume.Create;
  try
    Model := BuildVitsFromSafeTensors(SafePath, Config, CfgPath);
    WriteLn(VitsConfigToString(Config));
    WriteLn;

    // A short fixed token sequence (vocab is pico-sized).
    SetLength(Ids, 6);
    Ids[0] := 2; Ids[1] := 5; Ids[2] := 7; Ids[3] := 3; Ids[4] := 9; Ids[5] := 1;

    // Inspect the alignment the duration predictor chose.
    SetLength(Durations, Length(Ids));
    Model.Analyze(Ids, PriorMean, PriorLogVar, Durations);
    TotalDur := 0;
    Write('Durations (frames/token):');
    for i := 0 to High(Ids) do
    begin
      Write(' ', Durations[i]);
      TotalDur := TotalDur + Durations[i];
    end;
    WriteLn;
    OutLen := TotalDur;
    if OutLen < 1 then OutLen := 1;
    Flow := Config.FlowSize;

    // Standard-normal prior noise z [flow][out_len] (Box-Muller).
    SetLength(Z, Flow);
    for c := 0 to Flow - 1 do
    begin
      SetLength(Z[c], OutLen);
      for t := 0 to OutLen - 1 do
        Z[c][t] := Sqrt(-2.0 * Ln(Random)) * Cos(2.0 * Pi * Random);
    end;

    WriteLn('Synthesizing ', OutLen, ' latent frames -> waveform...');
    Model.Synthesize(Ids, Z, Wave);
    WriteLn('Waveform: ', Length(Wave), ' samples at ',
      Config.SamplingRate, ' Hz (', (Length(Wave) / Config.SamplingRate):0:3,
      ' s).');

    // Write a 16-bit PCM WAV (the waveform is already in [-1,1] via tanh).
    WaveVol.ReSize(Length(Wave), 1, 1);
    for i := 0 to High(Wave) do WaveVol.FData[i] := Wave[i];
    SaveVolumeToWav16(WaveVol, OutWav, Config.SamplingRate);
    WriteLn('Wrote ', OutWav, '.');
    WriteLn;
    WriteLn('Smoke test complete (the full text -> waveform pipeline ran).');
    WriteLn('For real speech, import a downloaded facebook/mms-tts-eng or');
    WriteLn('kakao-enterprise/vits-ljs checkpoint with BuildVitsFromSafeTensors');
    WriteLn('and tokenize the text with the model''s VitsTokenizer.');
  finally
    Model.Free;
    WaveVol.Free;
  end;
end.
