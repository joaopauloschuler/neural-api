program MusicGenMelody;
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

// MusicGenMelody -- the END-TO-END MELODY-CONDITIONED MusicGen demo. A
// reference MELODY waveform (a few deterministic pure tones) is turned into a
// 12-bin one-hot CHROMAGRAM that conditions the MusicGen Melody DECODER,
// optionally alongside a fixed text condition; the predicted EnCodec code
// stack is synthesized back to a waveform:
//
//   reference melody waveform (synthesized here)
//     -> ComputeMusicgenMelodyChroma (neuralaudio) -> one-hot chromagram
//     -> audio_enc_to_dec_proj                      -> chroma conditioning
//     -> (concat with the projected text condition, CHROMA FIRST)
//     -> PREPENDED to the MusicGen Melody causal self-attention decoder
//     -> greedy delay-pattern decode                -> [K][frames] code stack
//     -> EnCodec audio DECODER (BuildEnCodecFromSafeTensors) -> mono waveform
//     -> SaveVolumeToWav16                          -> a .wav clip
//
// Unlike text-MusicGen (cross-attention decoder), MusicGen Melody is a
// DECODER-ONLY model: the chroma + text conditioning is PREPENDED to the
// decoder sequence and attended through causal self-attention. The decoder,
// delay-interleave, and EnCodec codec are the landed text-MusicGen path; the
// genuinely new pieces are the chroma front-end and the dual conditioning
// concat.
//
// PICO SMOKE DEMO (no network)
// ----------------------------
// This v1 example runs the self-contained PICO fixtures (committed RANDOM
// weights: tests/fixtures/tiny_musicgen_melody* for the decoder, the matched
// tiny_musicgen_encodec* codec). The weights are untrained, so the produced
// clip is NOISE -- this exercises the full melody->audio wiring on CPU in a
// fraction of a second, NOT musical output. (A --download real-checkpoint mode
// for facebook/musicgen-melody is a deferred follow-up.)
//
// Usage:
//   MusicGenMelody                # pico smoke run (random weights -> noise WAV)
//   MusicGenMelody --frames 6     # explicit decoder frame count
//   MusicGenMelody --no-text      # chroma-only conditioning (zeroed text)

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  SysUtils, Math, StrUtils,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained,
  neuralaudio;

const
  PicoFixtureDir = '../../tests/fixtures/';
  OutWavName = 'musicgen_melody_demo.wav';

var
  Model: TMusicGenMelodyModel;
  Config: TMusicGenMelodyConfig;
  Codec: TEnCodecModel;
  CodecCfg: TEnCodecConfig;
  RefWave, Chroma, EncStates, Prefix, Wave: TNNetVolume;
  Codes: TNNetIntArr2D;
  Waveform: TNeuralFloatDynArr;
  EncSeq, DecSeq, NumFrames, k_i, t, i: integer;
  NSamp: integer;
  MelSafe, MelCfg, EcSafe, EcCfg: string;
  UseText: boolean;
  tt, ww: double;

  function HasArg(const Name: string): boolean;
  var a: integer;
  begin
    Result := false;
    for a := 1 to ParamCount do if ParamStr(a) = Name then Exit(true);
  end;

  function ParseIntArg(const Name: string; Def: integer): integer;
  var a: integer;
  begin
    Result := Def;
    for a := 1 to ParamCount - 1 do
      if ParamStr(a) = Name then Exit(StrToIntDef(ParamStr(a + 1), Def));
  end;

begin
  UseText := not HasArg('--no-text');
  NumFrames := ParseIntArg('--frames', 6);

  MelSafe := PicoFixtureDir + 'tiny_musicgen_melody.safetensors';
  MelCfg := PicoFixtureDir + 'tiny_musicgen_melody_config.json';
  EcSafe := PicoFixtureDir + 'tiny_musicgen_encodec.safetensors';
  EcCfg := PicoFixtureDir + 'tiny_musicgen_encodec_config.json';
  if not (FileExists(MelSafe) and FileExists(EcSafe)) then
  begin
    WriteLn('Pico fixtures not found under ', PicoFixtureDir);
    WriteLn('Generate them with:');
    WriteLn('  /home/bpsa/x/bin/python tools/make_pico_musicgen_melody_fixture.py');
    WriteLn('  /home/bpsa/x/bin/python tools/musicgen_tiny_fixture.py');
    Halt(1);
  end;

  WriteLn('MusicGenMelody pico smoke demo (random weights -> noise).');
  WriteLn;

  Model := nil;
  Codec := nil;
  RefWave := TNNetVolume.Create;
  Chroma := TNNetVolume.Create;
  EncStates := TNNetVolume.Create;
  Prefix := TNNetVolume.Create;
  Wave := TNNetVolume.Create;
  try
    // ---- 0. Read config to learn the chroma front-end + sequence widths. ----
    Config := ReadMusicGenMelodyConfigFromJSONFile(MelCfg);
    WriteLn(MusicGenMelodyConfigToString(Config));
    EncSeq := 5;             // matched to the pico T5 / fixture enc_states width
    DecSeq := NumFrames + Config.NumCodebooks - 1 + 1;  // delay-pattern headroom

    // ---- 1. Synthesize a short reference MELODY and extract its chroma. ------
    // A deterministic A4 (440 Hz) + E5 (660 Hz) mixture, > n_fft samples so the
    // STFT sees a couple of frames. The real model uses 32 kHz; the chroma
    // front-end takes the sample rate as a parameter.
    NSamp := 16384 + 4096;   // n_fft + one hop -> a couple of chroma frames
    RefWave.ReSize(NSamp, 1, 1);
    for i := 0 to NSamp - 1 do
    begin
      tt := i / 32000.0;
      ww := 0.6 * Sin(2 * Pi * 440.0 * tt) + 0.3 * Sin(2 * Pi * 660.0 * tt);
      RefWave.FData[i] := ww;
    end;
    ComputeMusicgenMelodyChroma(RefWave, Chroma, 32000, 16384, 4096,
      Config.NumChroma);
    WriteLn('Reference melody: ', NSamp, ' samples -> chroma ',
      Chroma.SizeX, ' frames x ', Chroma.Depth, ' pitch classes.');
    Write('Dominant pitch class per frame:');
    for t := 0 to Chroma.SizeX - 1 do
    begin
      k_i := 0;
      for i := 1 to Config.NumChroma - 1 do
        if Chroma.FData[t * Config.NumChroma + i] >
           Chroma.FData[t * Config.NumChroma + k_i] then k_i := i;
      Write(' ', k_i);
    end;
    WriteLn;

    // ---- 2. Optional text condition (zeroed when --no-text). ----------------
    // The pico melody fixture ships no text encoder; a deterministic small
    // text-state tensor stands in (or zeros for chroma-only conditioning).
    EncStates.ReSize(EncSeq, 1, Config.TextDModel);
    if UseText then
      for t := 0 to EncSeq - 1 do
        for i := 0 to Config.TextDModel - 1 do
          EncStates.FData[t * Config.TextDModel + i] :=
            0.25 * Sin(0.7 * (t + 1) + 0.3 * i)
    else
      EncStates.Fill(0);
    WriteLn('Text conditioning: ', IfThen(UseText, 'ON (fixed pseudo-states)',
      'OFF (zeroed)'), '.');
    WriteLn;

    // ---- 3. Build the melody decoder and generate the code stack. -----------
    Model := BuildMusicGenMelodyFromSafeTensors(MelSafe, Config, EncSeq, DecSeq,
      {pTrainable=}false, MelCfg);
    Model.BuildConditioningPrefix(Chroma, EncStates, Prefix);
    WriteLn('Conditioning prefix: ', Prefix.SizeX, ' frames (',
      Config.ChromaLength, ' chroma + ', EncSeq, ' text) prepended to the ',
      'decoder.');
    WriteLn('Generating ', NumFrames, ' frames over ', Config.NumCodebooks,
      ' codebooks via the delay pattern...');
    Model.Generate(Prefix, NumFrames, Codes);
    WriteLn('Generated code stack (codebook x frame):');
    for k_i := 0 to Config.NumCodebooks - 1 do
    begin
      Write('  cb', k_i, ':');
      for t := 0 to NumFrames - 1 do Write(' ', Codes[k_i][t]:3);
      WriteLn;
    end;
    WriteLn;

    // ---- 4. Decode the code stack to a waveform with the EnCodec decoder. ---
    Codec := BuildEnCodecFromSafeTensors(EcSafe, CodecCfg, EcCfg);
    WriteLn('EnCodec decoder: ', EnCodecConfigToString(CodecCfg));
    if Codec.NumCodebooks < Config.NumCodebooks then
    begin
      WriteLn('FATAL: EnCodec has ', Codec.NumCodebooks,
        ' quantizers, fewer than MusicGen K=', Config.NumCodebooks);
      Halt(1);
    end;
    Codec.DecodeCodesToAudio(Codes, Waveform, Config.NumCodebooks);
    WriteLn('Decoded waveform: ', Length(Waveform), ' samples at ',
      CodecCfg.SamplingRate, ' Hz');

    // ---- 5. Write the clip to a 16-bit PCM WAV. -----------------------------
    Wave.ReSize(Length(Waveform), 1, 1);
    for i := 0 to Length(Waveform) - 1 do Wave.FData[i] := Waveform[i];
    SaveVolumeToWav16(Wave, OutWavName, CodecCfg.SamplingRate);
    WriteLn('Wrote ', OutWavName, ' (', Length(Waveform), ' samples, ',
      (Length(Waveform) / CodecCfg.SamplingRate):0:2, ' s).');
    WriteLn;
    WriteLn('Done (pico demo): full melody->chroma->audio wiring exercised on ',
      'random weights.');
  finally
    Model.Free;
    Codec.Free;
    RefWave.Free;
    Chroma.Free;
    EncStates.Free;
    Prefix.Free;
    Wave.Free;
  end;
end.
