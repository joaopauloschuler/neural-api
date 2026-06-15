program MusicGenText;
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

// MusicGenText -- the END-TO-END TEXT-CONDITIONED MusicGen demo, the successor
// to the landed MusicGenSmoke (which STUBBED the text encoder with a fixed
// pseudo encoder-state tensor). Here the REAL T5 text ENCODER is wired in, so
// a free-text prompt (tokenized to ids) actually steers the generated music:
//
//   text token ids
//     -> T5 text ENCODER (BuildT5FromSafeTensors)            [REAL, not stubbed]
//     -> the encoder's final hidden states (EncSeq x text_d_model)
//     -> MusicGen enc_to_dec_proj -> cross-attention conditioning
//     -> MusicGen DECODER greedy delay-pattern decode -> [K][frames] code stack
//     -> EnCodec audio DECODER (BuildEnCodecFromSafeTensors) -> mono waveform
//     -> SaveVolumeToWav16 -> a SHORT .wav clip
//
// The T5 encoder hidden states feed the SAME slot MusicGenSmoke filled with a
// hand-built ramp; the difference is that those states are now produced by a
// genuine encoder run, so changing the prompt ids changes the music.
//
// With NO arguments this runs a SELF-CONTAINED pico demo on the committed
// random fixtures (tests/fixtures/tiny_musicgen*, tiny_musicgen_t5enc*,
// tiny_musicgen_encodec*): build the T5 encoder, run it on a fixed prompt id
// sequence, build the MusicGen decoder, greedily generate a short code stack
// through the delay pattern, decode it to audio with the EnCodec decoder, and
// write the clip to musicgen_text_demo.wav -- no download, pure CPU, a fraction
// of a second. The weights are untrained random, so the clip is not meant to
// sound like music; this only exercises the FULL text->audio wiring end to end.
//
// Usage:
//   MusicGenText                 # pico text-conditioned demo (no download)
//
// Deferred follow-ups (see tasklist.md): classifier-free guidance (the
// unconditional/null-prompt branch), stereo (the 2K-codebook layout), and
// real large checkpoints / a real tokenizer for the prompt.

{$mode objfpc}{$H+}

uses
  SysUtils, Math,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained,
  neuralaudio;

const
  PicoFixtureDir = '../../tests/fixtures/';
  OutWavName = 'musicgen_text_demo.wav';

var
  T5Enc, T5Dec: TNNet;
  T5Cfg: TT5Config;
  Model: TMusicGenModel;
  Config: TMusicGenConfig;
  Codec: TEnCodecModel;
  CodecCfg: TEnCodecConfig;
  Tokens, EncStates, Wave: TNNetVolume;
  Codes: TNNetIntArr2D;
  Waveform: TNeuralFloatDynArr;
  EncSeq, DecSeq, NumFrames, k_i, t, i: integer;
  MgSafe, MgCfg, T5Safe, T5CfgPath, EcSafe, EcCfg: string;
  Ids: array of integer;
begin
  // A fixed pseudo-prompt: a short list of token ids standing in for a real
  // tokenizer's encoding of a text prompt. Any ids in [0, T5 vocab) are valid.
  Ids := [3, 8, 1, 5, 2];
  WriteLn('MusicGen TEXT-CONDITIONED generation - prompt -> T5 -> music -> WAV');
  WriteLn('==================================================================');

  MgSafe := PicoFixtureDir + 'tiny_musicgen.safetensors';
  MgCfg := PicoFixtureDir + 'tiny_musicgen_config.json';
  T5Safe := PicoFixtureDir + 'tiny_musicgen_t5enc.safetensors';
  T5CfgPath := PicoFixtureDir + 'tiny_musicgen_t5enc_config.json';
  EcSafe := PicoFixtureDir + 'tiny_musicgen_encodec.safetensors';
  EcCfg := PicoFixtureDir + 'tiny_musicgen_encodec_config.json';
  if not FileExists(MgSafe) or not FileExists(T5Safe) or not FileExists(EcSafe)
  then
  begin
    WriteLn('Pico fixtures not found under ', PicoFixtureDir);
    WriteLn('Run from the example directory (examples/MusicGenText), or run');
    WriteLn('  python tools/musicgen_tiny_fixture.py  to (re)generate them.');
    Halt(1);
  end;
  WriteLn('No checkpoint argument: running the self-contained pico demo on the');
  WriteLn('committed RANDOM fixtures (untrained weights, so the clip is not');
  WriteLn('meant to sound like music - this exercises the full text->audio');
  WriteLn('wiring end to end).');
  WriteLn;

  EncSeq := Length(Ids);
  NumFrames := 6;

  T5Enc := nil; T5Dec := nil; Model := nil; Codec := nil;
  Tokens := TNNetVolume.Create;
  EncStates := TNNetVolume.Create;
  Wave := TNNetVolume.Create;
  try
    // --- 1. Build the REAL T5 text encoder and run it on the prompt ids. ---
    BuildT5FromSafeTensors(T5Safe, T5Enc, T5Dec, T5Cfg, EncSeq, 1,
      {pInferenceOnly=}true, T5CfgPath);
    WriteLn('T5 text encoder: ', T5ConfigToString(T5Cfg));

    Tokens.ReSize(EncSeq, 1, 1);
    Write('Prompt token ids:');
    for i := 0 to EncSeq - 1 do
    begin
      Tokens.FData[i] := Ids[i];
      Write(' ', Ids[i]);
    end;
    WriteLn;
    T5Enc.Compute(Tokens);
    // The encoder's final hidden states (EncSeq x d_model) are the conditioning
    // signal; copy them into a (EncSeq,1,d_model) volume for MusicGen.
    EncStates.Copy(T5Enc.GetLastLayer.Output);
    WriteLn('Encoder hidden states: ', EncStates.SizeX, 'x', EncStates.Depth,
      ' (seq x d_model)');
    WriteLn;

    // --- 2. Build the MusicGen decoder and generate the code stack. ---
    Config := ReadMusicGenConfigFromJSONFile(MgCfg);
    if Config.TextDModel <> EncStates.Depth then
    begin
      WriteLn('FATAL: T5 d_model (', EncStates.Depth, ') != MusicGen ',
        'text_d_model (', Config.TextDModel, ')');
      Halt(1);
    end;
    // The decoder needs room for NumFrames + (K - 1) delay steps (+1 headroom).
    DecSeq := NumFrames + Config.NumCodebooks - 1 + 1;
    Model := BuildMusicGenFromSafeTensors(MgSafe, Config, EncSeq, DecSeq,
      {pInferenceOnly=}true, MgCfg);
    WriteLn(MusicGenConfigToString(Config));

    WriteLn('Generating ', NumFrames, ' frames over ', Config.NumCodebooks,
      ' codebooks via the delay pattern (conditioned on the T5 prompt)...');
    Model.Generate(EncStates, NumFrames, Codes);
    WriteLn('Generated code stack (codebook x frame):');
    for k_i := 0 to Config.NumCodebooks - 1 do
    begin
      Write('  cb', k_i, ':');
      for t := 0 to NumFrames - 1 do Write(' ', Codes[k_i][t]:3);
      WriteLn;
    end;
    WriteLn;

    // --- 3. Decode the code stack to a waveform with the EnCodec decoder. ---
    Codec := BuildEnCodecFromSafeTensors(EcSafe, CodecCfg, EcCfg);
    WriteLn('EnCodec decoder: ', EnCodecConfigToString(CodecCfg));
    if Codec.NumCodebooks < Config.NumCodebooks then
    begin
      WriteLn('FATAL: EnCodec has ', Codec.NumCodebooks,
        ' quantizers, fewer than MusicGen K=', Config.NumCodebooks);
      Halt(1);
    end;
    // Decode using exactly the K codebooks MusicGen produced.
    Codec.DecodeCodesToAudio(Codes, Waveform, Config.NumCodebooks);
    WriteLn('Decoded waveform: ', Length(Waveform), ' samples at ',
      CodecCfg.SamplingRate, ' Hz');

    // --- 4. Write the clip to a 16-bit PCM WAV. ---
    Wave.ReSize(Length(Waveform), 1, 1);
    for i := 0 to Length(Waveform) - 1 do Wave.FData[i] := Waveform[i];
    SaveVolumeToWav16(Wave, OutWavName, CodecCfg.SamplingRate);
    WriteLn('Wrote ', OutWavName, ' (', Length(Waveform), ' samples).');
    WriteLn;
    WriteLn('Text-conditioned generation complete: the prompt ids drove a real');
    WriteLn('T5 encoder, which conditioned the MusicGen decoder, whose codes');
    WriteLn('were synthesized to audio by the EnCodec decoder.');
  finally
    T5Enc.Free;
    T5Dec.Free;
    Model.Free;
    Codec.Free;
    Tokens.Free;
    EncStates.Free;
    Wave.Free;
  end;
end.
