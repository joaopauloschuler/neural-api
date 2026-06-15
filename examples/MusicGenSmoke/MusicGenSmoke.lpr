program MusicGenSmoke;
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

// MusicGenSmoke -- a text-to-AUDIO generation demo: the MusicGen LM
// decoder predicts a stack of EnCodec codes autoregressively using the
// DELAY-PATTERN codebook interleaving, the inverse of the analysis-only audio
// demos. The genuinely new piece over the seq2seq enc-dec convention is the
// delay pattern: each of the K codebooks is offset by one decode step so a
// single set of K LM heads can predict them causally.
//
//   text tokens
//     -> T5 text ENCODER (BuildT5FromSafeTensors)        [not in this smoke]
//     -> enc_to_dec_proj -> cross-attention conditioning
//     -> MusicGen DECODER (pre-norm cross-attention blocks, K summed code
//        embeddings, sinusoidal positions, K LM heads)
//     -> greedy delay-pattern decode -> a [K][frames] EnCodec code stack
//     -> EnCodec audio DECODER (BuildEnCodecFromSafeTensors) -> waveform
//
// With NO arguments this runs a SELF-CONTAINED pico smoke test on the
// committed random fixture (tests/fixtures/tiny_musicgen.*): it builds the
// decoder, feeds a fixed pseudo-encoder-state tensor (standing in for the T5
// text encoder), and greedily generates a short code stack through the delay
// pattern -- no download, pure CPU, a fraction of a second. The weights are
// untrained random, so the codes are not meant to sound like music; this only
// exercises the importer + delay-pattern generation end to end.
//
// Usage:
//   MusicGenSmoke                # pico smoke test (no download)

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained;

const
  PicoFixtureDir = '../../tests/fixtures/';

var
  Model: TMusicGenModel;
  Config: TMusicGenConfig;
  EncStates: TNNetVolume;
  Codes: TNNetIntArr2D;
  SafePath, CfgPath: string;
  EncSeq, DecSeq, NumFrames, k_i, t, i: integer;
begin
  WriteLn('MusicGen text-to-music - delay-pattern generation smoke test');
  WriteLn('============================================================');
  SafePath := PicoFixtureDir + 'tiny_musicgen.safetensors';
  CfgPath := PicoFixtureDir + 'tiny_musicgen_config.json';
  if not FileExists(SafePath) then
  begin
    WriteLn('Pico fixture not found at ', SafePath);
    WriteLn('Run from the example directory (examples/MusicGenSmoke).');
    Halt(1);
  end;
  WriteLn('No checkpoint argument: running the self-contained pico smoke test');
  WriteLn('on the committed RANDOM fixture (weights are untrained, so the');
  WriteLn('generated codes are not meant to sound like music - this only');
  WriteLn('exercises the importer + delay-pattern generation end to end).');
  WriteLn;

  // The decoder needs room for NumFrames + (K - 1) delay steps.
  EncSeq := 5;
  NumFrames := 6;
  // Build with a DecSeqLen comfortably above NumFrames + K - 1.
  Model := nil;
  EncStates := TNNetVolume.Create;
  try
    // First peek at the config to size the sequence length / encoder states.
    Config := ReadMusicGenConfigFromJSONFile(CfgPath);
    DecSeq := NumFrames + Config.NumCodebooks - 1 + 1;
    Model := BuildMusicGenFromSafeTensors(SafePath, Config, EncSeq, DecSeq,
      {pInferenceOnly=}true, CfgPath);
    WriteLn(MusicGenConfigToString(Config));
    WriteLn;

    // A fixed pseudo-encoder-state tensor (EncSeq x text_d_model) standing in
    // for the T5 text encoder output. A deterministic ramp keeps it simple.
    EncStates.ReSize(EncSeq, 1, Config.TextDModel);
    for t := 0 to EncSeq - 1 do
      for i := 0 to Config.TextDModel - 1 do
        EncStates.FData[t * Config.TextDModel + i] :=
          0.1 * Sin(0.3 * (t + 1) + 0.2 * i);

    WriteLn('Generating ', NumFrames, ' frames over ',
      Config.NumCodebooks, ' codebooks via the delay pattern...');
    Model.Generate(EncStates, NumFrames, Codes);

    WriteLn('Generated code stack (codebook x frame):');
    for k_i := 0 to Config.NumCodebooks - 1 do
    begin
      Write('  cb', k_i, ':');
      for t := 0 to NumFrames - 1 do Write(' ', Codes[k_i][t]:3);
      WriteLn;
    end;
    WriteLn;
    WriteLn('Smoke test complete (the delay-pattern decode ran without error).');
    WriteLn('Feed these codes to BuildEnCodecFromSafeTensors''');
    WriteLn('DecodeCodesToAudio to synthesize a waveform with a real codec.');
  finally
    Model.Free;
    EncStates.Free;
  end;
end.
