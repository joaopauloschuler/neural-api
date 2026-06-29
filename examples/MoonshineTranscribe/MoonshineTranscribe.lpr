program MoonshineTranscribe;
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

// MoonshineTranscribe -- the Moonshine STREAMING-ASR encoder (UsefulSensors/
// moonshine-tiny | moonshine-base), a SECOND speech-to-text architecture
// distinct from Whisper. Whisper pads every clip to a FIXED 30 s log-mel
// spectrogram (so a 1 s utterance costs the same as a 30 s one); Moonshine
// has NO mel frontend - it convolves RoPE-positioned features DIRECTLY off
// the raw 16 kHz waveform with a small strided-conv stem, so the encoder
// compute scales with the ACTUAL audio length. The pipeline here:
//
//   raw 16 kHz waveform (samples,1,1)
//     -> conv stem  (conv1 k127 s64 + tanh; GroupNorm(1); conv2 k7 s3 + GELU;
//        conv3 k3 s2 + GELU)  -- ~13 ms hop, length-proportional frame count
//     -> partial-RoPE BIDIRECTIONAL pre-norm transformer encoder
//     -> encoder hidden states (frames,1,hidden)
//
// The decoder (RoPE + SwiGLU, cross-attending these states) is the autoregressive
// transformer that turns the encoder hidden states into text. It is built by
// BuildMoonshineEncoderDecoderFromSafeTensors and driven by
// DecodeMoonshineGreedyCached: a KV-CACHE incremental decode (self-attn K/V
// cached across steps, constant cross-attn states re-read each step) that runs
// the whole transcript in O(L) instead of the O(L^2) re-encode-the-prefix loop.
//
// Usage:
//   MoonshineTranscribe [<checkpoint-dir>]
//
//   With NO argument it runs a DETERMINISTIC self-contained smoke off the
//   committed pico fixture (tests/fixtures/tiny_moonshine*), no download:
//   it builds the encoder, encodes a synthetic waveform at two lengths, and
//   prints the per-length frame count + latency so the Whisper-vs-Moonshine
//   contrast (fixed 30 s cost vs length-proportional cost) is visible.
//
//   With a checkpoint-dir holding model.safetensors (or pytorch_model.bin)
//   + config.json (e.g. a download of huggingface.co/UsefulSensors/
//   moonshine-tiny) it imports that real encoder AND decoder, then - if a
//   tokenizer.json is present - greedily decodes a real transcription of a
//   synthetic clip (swap in a TNNetVolume of your own 16 kHz samples for a
//   real utterance).

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils, Math,
  neuralvolume, neuralnetwork, neuralpretrained, neuraldecode,
  neuralhftokenizer;

// Builds the encoder for NumSamples samples, times one forward over a
// deterministic synthetic waveform, and reports frames + latency.
procedure EncodeAndReport(const WeightsPath, ConfigPath: string;
  NumSamples: integer);
var
  NN: TNNet;
  Config: TMoonshineConfig;
  Wave: TNNetVolume;
  i: integer;
  BuildTicks, EncTicks: QWord;
begin
  BuildTicks := GetTickCount64;
  NN := BuildMoonshineFromSafeTensorsEx(WeightsPath, Config, NumSamples,
    {pTrainable=}false, ConfigPath);
  try
    BuildTicks := GetTickCount64 - BuildTicks;
    Wave := TNNetVolume.Create;
    try
      Wave.ReSize(NumSamples, 1, 1);
      // Deterministic synthetic clip: a 220 Hz tone at 16 kHz, dyadic-ish.
      for i := 0 to NumSamples - 1 do
        Wave.FData[i] := 0.5 * Sin(2.0 * Pi * 220.0 * i / 16000.0);
      EncTicks := GetTickCount64;
      NN.Compute(Wave);
      EncTicks := GetTickCount64 - EncTicks;
      WriteLn(Format('  %6d samples (%5.2f s) -> %3d frames | '
        + 'build %4d ms, encode %4d ms',
        [NumSamples, NumSamples / 16000.0,
         NN.GetLastLayer().Output.SizeX, BuildTicks, EncTicks]));
    finally
      Wave.Free;
    end;
  finally
    NN.Free;
  end;
end;

// Imports the full encoder+decoder and, when tokenizer.json is present,
// greedily transcribes a synthetic clip to real text. SourceTokens for a
// seq2seq audio model is the ENCODER input grid (one slot per sample); we
// feed it a synthetic waveform here - replace Wave with your own samples.
procedure TranscribeReal(const WeightsPath, ConfigPath, TokPath: string;
  NumSamples: integer);
var
  Enc, Dec: TNNet;
  Config: TMoonshineConfig;
  Tok: TNeuralHFTokenizer;
  Wave: TNNetVolume;
  Gen: TNeuralIntegerArray;
  i, MaxNewTokens, StartId, EOSId: integer;
begin
  // A small decode budget keeps the smoke fast; raise for longer clips.
  MaxNewTokens := 63;
  StartId := 1;  // decoder_start_token_id (Moonshine default)
  EOSId := 2;    // eos_token_id (Moonshine default)
  // KV-cache incremental decode feeds ONE token per step, so the decoder's
  // token input is built at width 1 (the growing context lives in the cache).
  BuildMoonshineEncoderDecoderFromSafeTensors(WeightsPath, Enc, Dec, Config,
    NumSamples, {DecSeqLen=}1, {pTrainable=}false, ConfigPath);
  Tok := TNeuralHFTokenizer.Create();
  Wave := TNNetVolume.Create;
  SetLength(Gen, 0);
  try
    Tok.LoadFromFile(TokPath);
    Wave.ReSize(NumSamples, 1, 1);
    for i := 0 to NumSamples - 1 do
      Wave.FData[i] := 0.5 * Sin(2.0 * Pi * 220.0 * i / 16000.0);
    // The audio encoder's input is the RAW WAVEFORM (not token ids), so we
    // cannot use DecodeSeq2SeqGreedy (which assumes a token-id encoder).
    // DecodeMoonshineGreedyCached encodes once, caches the states in the
    // decoder's 2nd input, then drives a KV-CACHE incremental decode: each
    // generated token is fed one at a time, appending only its self-attn K/V
    // and re-reading the constant cross-attn states - O(L) total instead of
    // the O(L^2) re-run-the-whole-prefix loop this example used to drive.
    Gen := DecodeMoonshineGreedyCached(Enc, Dec, Wave, StartId, EOSId,
      MaxNewTokens);
    // EOS is appended by the decoder; trim it before detokenizing.
    if (Length(Gen) > 0) and (Gen[High(Gen)] = EOSId) then
      SetLength(Gen, Length(Gen) - 1);
    Write('  transcription: "');
    Write(Tok.Decode(Gen, {SkipSpecialTokens=}true));
    WriteLn('"');
  finally
    Wave.Free;
    Tok.Free;
    Dec.Free;
    Enc.Free;
  end;
end;

var
  CheckpointDir, WeightsPath, ConfigPath, FixDir, TokPath: string;
  Config: TMoonshineConfig;
begin
  TokPath := '';
  if ParamCount >= 1 then
  begin
    CheckpointDir := IncludeTrailingPathDelimiter(ParamStr(1));
    WeightsPath := CheckpointDir + 'model.safetensors';
    if not FileExists(WeightsPath) then
      WeightsPath := CheckpointDir + 'pytorch_model.bin';
    ConfigPath := CheckpointDir + 'config.json';
    if not FileExists(WeightsPath) then
    begin
      WriteLn('No model.safetensors / pytorch_model.bin in ', CheckpointDir);
      Halt(1);
    end;
    if FileExists(CheckpointDir + 'tokenizer.json') then
      TokPath := CheckpointDir + 'tokenizer.json';
  end
  else
  begin
    // Deterministic smoke off the committed pico fixture (run from the repo
    // root or the example directory).
    FixDir := 'tests/fixtures/';
    if not FileExists(FixDir + 'tiny_moonshine.safetensors') then
      FixDir := '../../tests/fixtures/';
    WeightsPath := FixDir + 'tiny_moonshine.safetensors';
    ConfigPath := FixDir + 'tiny_moonshine_config.json';
    if not FileExists(WeightsPath) then
    begin
      WriteLn('Fixture not found; run from the repo root or pass a ' +
        'checkpoint dir. Looked in ', FixDir);
      Halt(1);
    end;
    WriteLn('No checkpoint given - running the deterministic pico-fixture ' +
      'smoke.');
  end;

  Config := ReadMoonshineConfigFromJSONFile(ConfigPath);
  WriteLn('Loaded ', MoonshineConfigToString(Config));
  WriteLn;
  WriteLn('Encoder compute scales with the ACTUAL waveform length ' +
    '(unlike Whisper''s fixed 30 s mel cost):');
  // Two lengths that both clear the conv stem (conv1 k127 s64, conv2 k7 s3,
  // conv3 k3 s2): the frame count grows roughly linearly with the samples.
  EncodeAndReport(WeightsPath, ConfigPath, 1719);
  EncodeAndReport(WeightsPath, ConfigPath, 5000);
  WriteLn;
  if TokPath <> '' then
  begin
    WriteLn('Full encoder+decoder greedy transcription (RoPE + cross-attn + ' +
      'SwiGLU decoder, tokenizer.json):');
    TranscribeReal(WeightsPath, ConfigPath, TokPath, 16000);
    WriteLn;
    WriteLn('Done. (Swap the synthetic clip for your own 16 kHz samples to ' +
      'transcribe real audio.)');
  end
  else
  begin
    WriteLn('Done. (No tokenizer.json found - the decoder builds and runs ' +
      'but real text needs a checkpoint with a tokenizer; the pico fixture ' +
      'has none. See TestMoonshineDecoderLogitParity for decoder parity.)');
  end;
end.
