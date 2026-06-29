program ParlerTTS;
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

// ParlerTTS -- an inference smoke of the Parler-TTS importer (model_type
// "parler_tts", parler-tts/parler-tts-mini-v1; Lyth & King 2024). Parler-TTS is
// a DESCRIPTION-conditioned text-to-speech model:
//
//   * a (By)T5 text ENCODER encodes a free-text STYLE DESCRIPTION ("a female
//     speaker with a slightly low-pitched voice, very clear audio") whose hidden
//     states condition the decoder via CROSS-ATTENTION;
//   * a codec-LM DECODER autoregressively predicts the DELAY-PATTERNED
//     multi-codebook DAC code stack, conditioned on BOTH the description
//     (cross-attention) and the TRANSCRIPT to speak (its prompt token ids,
//     embedded by embed_prompts and PREPENDED on the sequence axis);
//   * a DAC decoder renders the waveform from the predicted codes.
//
// This smoke drives the genuinely new piece -- the prefix-prepended,
// cross-attended codec DECODER -- end to end on the committed pico fixture
// (tests/fixtures/tiny_parler.*), pure CPU, a fraction of a second. The decoder
// is fed a FIXED description-encoder hidden-state tensor (what enc_to_dec_proj(
// T5) would emit) and a FIXED transcript prompt, then autoregressively
// generates DAC codes with the SDPA KV-CACHE incremental-decode fast path. The
// weights are untrained random so the codes are not real speech; the smoke
// exercises the wiring (prefix prepend + cross-attention + delay pattern +
// KV-cache).
//
// SCOPE: imports the description-conditioned codec DECODER (the new importable
// piece). Pairing it with the REAL (By)T5 encoder (BuildT5FromSafeTensors) and
// the landed DAC decoder (BuildDACFromSafeTensors) to a waveform is a documented
// follow-up -- all three pieces are already importable in-tree.

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils, Math,
  neuralnetwork, neuralvolume, neuralpretrained;

var
  Model: TParlerTTSModel;
  Config: TParlerConfig;
  EncStates: TNNetVolume;
  Codes: TNNetIntArr2D;
  PromptIds: array of integer;
  FixDir, SafePath, CfgPath: string;
  EncSeq, PromptLen, CodecLen, NumFrames, i, k_i, t: integer;

begin
  WriteLn('Parler-TTS description-conditioned codec-LM decode smoke.');
  FixDir := IncludeTrailingPathDelimiter(
    ExpandFileName('..' + PathDelim + '..' + PathDelim + 'tests' +
      PathDelim + 'fixtures'));
  SafePath := FixDir + 'tiny_parler.safetensors';
  CfgPath := FixDir + 'tiny_parler_config.json';
  if not FileExists(SafePath) then
  begin
    WriteLn('Missing fixture ', SafePath);
    WriteLn('Generate it with:  /home/bpsa/x/bin/python tools/parler_tiny_fixture.py');
    Halt(1);
  end;

  // Sequence budgets for the static decoder graph: the description length
  // (cross-attention), the transcript prefix length, and the codec frame budget.
  EncSeq := 5;
  PromptLen := 3;
  CodecLen := 8;

  Model := BuildParlerTTSFromSafeTensors(SafePath, Config, EncSeq, PromptLen,
    CodecLen, {pTrainable=}false, CfgPath);
  try
    WriteLn('Loaded: ', ParlerConfigToString(Config));
    WriteLn('Codec decoder: ', Model.Decoder.CountLayers(), ' layers, K=',
      Config.NumCodebooks, ' codebooks, prefix=', PromptLen, ' transcript tokens.');

    EncStates := TNNetVolume.Create(EncSeq, 1, Config.TextDModel);
    try
      RandSeed := 1234;
      // Description encoder hidden states (what the (By)T5 encoder WOULD emit for
      // the style description). Untrained weights -> any conditioning works; use
      // a deterministic synthetic tensor here.
      for i := 0 to EncStates.Size - 1 do
        EncStates.FData[i] := Sin(i * 0.17) * 0.5;
      // Transcript prompt token ids (what to SAY), within the prompt vocabulary.
      SetLength(PromptIds, PromptLen);
      for i := 0 to PromptLen - 1 do
        PromptIds[i] := (2 * i + 1) mod Config.PromptVocabSize;

      // Autoregressive DAC-code generation with the KV-cache fast path. The
      // budget is CodecLen - K so the delay-pattern extraction stays in range.
      NumFrames := CodecLen - Config.NumCodebooks;
      WriteLn('Generating ', NumFrames, ' DAC frames (KV-cache incremental decode)...');
      Model.Generate(EncStates, PromptIds, NumFrames, {UseCache=}true, Codes);

      WriteLn('Predicted DAC code stack (', Config.NumCodebooks, ' codebooks x ',
        NumFrames, ' frames):');
      for k_i := 0 to Config.NumCodebooks - 1 do
      begin
        Write('  codebook ', k_i, ':');
        for t := 0 to NumFrames - 1 do Write(' ', Codes[k_i][t]);
        WriteLn;
      end;
      WriteLn('Smoke OK. (Untrained pico weights -> codes are not real speech.)');
      WriteLn('Pair with the real (By)T5 encoder + DAC decoder for a waveform.');
    finally
      EncStates.Free;
    end;
  finally
    Model.Free;
  end;
end.
