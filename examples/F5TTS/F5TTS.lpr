program F5TTS;
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

// F5TTS -- a voice-clone smoke demo of the F5-TTS flow-matching text-to-speech
// importer (SWivid/F5-TTS; Chen et al. 2024). F5-TTS is a NON-autoregressive,
// NON-GAN voice cloner that regresses a mel-spectrogram by integrating a
// conditional-flow-matching ODE through a DiT velocity field, conditioned
// IN-CONTEXT on a masked reference mel + an embedded character sequence:
//
//   reference mel (cond) + character ids + noised mel x_t + time t
//     -> text branch:  char-embed -> ConvNeXt-V2 1D blocks
//     -> input embed:  concat([x_t, cond, text_emb]) -> Linear + conv-pos res
//     -> time branch:  sinusoidal(t*1000) -> SiLU-MLP -> conditioning vector c
//     -> DiT trunk:    adaLN-zero blocks with RoPE self-attention
//     -> adaLN norm-out -> proj_out -> VELOCITY field v_theta(x_t, ..., t)
//
// SAMPLER: the mel is produced by the flow-matching Euler ODE driver (the same
// machinery as examples/FlowMatching / neuraldiffusion.pas): start at x_0 ~
// N(0,I), integrate x_{t+dt} = x_t + dt * v_theta(x_t, cond, text, t) from
// t=0 to t=1. NON-autoregressive -- there is no KV-cache (every step is a full
// parallel forward over the whole sequence; KV-cache would not help).
//
// SCOPE v1: imports the DiT VELOCITY FIELD (the genuinely new importable
// piece). The reference mel and character ids are EXPLICIT inputs; the
// grapheme front-end (raw chars / pinyin, no phonemizer) and the mel->waveform
// VOCODER (Vocos / HiFi-GAN) are out of scope here -- v1 outputs a MEL, not a
// waveform. Pair the mel with an already-landed vocoder downstream.
//
// Runs a SELF-CONTAINED pico smoke with no arguments (committed random fixture
// tests/fixtures/tiny_f5.*) -- pure CPU, a fraction of a second. The weights
// are untrained random so the mel is noise, not speech.

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralpretrained;

var
  NN: TNNet;
  Config: TF5Config;
  Xt, Cond, Text, Time, Vel: TNNetVolume;
  FixDir, SafePath, CfgPath: string;
  S, NMel, NumSteps, step, i: integer;
  t, dt: TNeuralFloat;
  MelMin, MelMax, MelMean: TNeuralFloat;

begin
  WriteLn('F5-TTS flow-matching voice-clone smoke (DiT velocity field).');
  // The pico fixture lives beside the repo's tests/fixtures.
  FixDir := IncludeTrailingPathDelimiter(
    ExpandFileName('..' + PathDelim + '..' + PathDelim + 'tests' +
      PathDelim + 'fixtures'));
  SafePath := FixDir + 'tiny_f5.safetensors';
  CfgPath := FixDir + 'tiny_f5_config.json';
  if not FileExists(SafePath) then
  begin
    WriteLn('Missing fixture ', SafePath);
    WriteLn('Generate it with:  /home/bpsa/x/bin/python tools/f5_tiny_fixture.py');
    Halt(1);
  end;

  // Sequence length for the static graph: the reference + target mel frames.
  S := 5;
  NN := BuildF5TTSFromSafeTensors(SafePath, S, Config, CfgPath);
  try
    NMel := Config.NMelChannels;
    WriteLn('Loaded: ', F5ConfigToString(Config));
    WriteLn('DiT velocity field: ', NN.CountLayers(), ' layers, ',
      NN.GetLastLayer().Output.Size, ' mel outputs (', S, ' frames x ',
      NMel, ' mels).');

    Xt   := TNNetVolume.Create(S, 1, NMel);
    Cond := TNNetVolume.Create(S, 1, NMel);
    Text := TNNetVolume.Create(S, 1, 1);
    Time := TNNetVolume.Create(1, 1, 1);
    Vel  := TNNetVolume.Create();
    try
      RandSeed := 1234;
      // Reference (cond) mel: pretend we have a masked reference speaker clip.
      // Untrained weights -> any reference works; use a smooth synthetic one.
      for i := 0 to Cond.Size - 1 do
        Cond.FData[i] := Sin(i * 0.21) * 0.5;
      // Character ids of the target text (raw chars / pinyin ids; here a
      // deterministic toy id sequence within the tiny vocab).
      for i := 0 to S - 1 do
        Text.FData[i] := (3 * i + 2) mod Config.TextNumEmbeds;

      // ---- flow-matching Euler ODE sampler ----
      NumSteps := 8;
      dt := 1.0 / NumSteps;
      // x_0 ~ N(0,I)
      for i := 0 to Xt.Size - 1 do Xt.FData[i] := RandG(0, 1);
      for step := 0 to NumSteps - 1 do
      begin
        t := step * dt;
        // TNNetSinusoidalTimeEmbedding consumes the input verbatim; F5 /
        // FlowMatching scale continuous t in [0,1] by 1000 first.
        Time.FData[0] := t * 1000.0;
        NN.Compute([Xt, Cond, Text, Time]);
        NN.GetOutput(Vel);
        for i := 0 to Xt.Size - 1 do
          Xt.FData[i] := Xt.FData[i] + dt * Vel.FData[i];
      end;

      // The integrated x_1 is the predicted MEL spectrogram (S x NMel).
      MelMin := Xt.FData[0]; MelMax := Xt.FData[0]; MelMean := 0;
      for i := 0 to Xt.Size - 1 do
      begin
        if Xt.FData[i] < MelMin then MelMin := Xt.FData[i];
        if Xt.FData[i] > MelMax then MelMax := Xt.FData[i];
        MelMean := MelMean + Xt.FData[i];
      end;
      MelMean := MelMean / Xt.Size;
      WriteLn('Sampled mel after ', NumSteps,
        ' Euler ODE steps: min=', MelMin:0:4, ' max=', MelMax:0:4,
        ' mean=', MelMean:0:4);
      WriteLn('First mel frame:');
      Write('  ');
      for i := 0 to NMel - 1 do Write(Xt.FData[i]:0:4, ' ');
      WriteLn;
      WriteLn('NOTE: v1 outputs a MEL spectrogram. Pair it with a landed ',
        'vocoder (Vocos / HiFi-GAN) to reach a waveform. Untrained random ',
        'weights -> the mel is noise, not speech.');
      WriteLn('Done.');
    finally
      Vel.Free; Time.Free; Text.Free; Cond.Free; Xt.Free;
    end;
  finally
    NN.Free;
  end;
end.
