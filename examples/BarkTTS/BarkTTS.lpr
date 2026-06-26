program BarkTTS;
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

// BarkTTS -- a text-to-SPEECH generation smoke for the Bark importer
// (model_type "bark", suno/bark[-small]). Bark chains THREE GPT-2-style
// decoders then the LANDED EnCodec decoder (reused from the MusicGen path):
//
//   text+semantic tokens
//     -> SEMANTIC model (BarkCausalModel)  -> semantic tokens
//     -> COARSE model   (BarkCausalModel)  -> coarse EnCodec codebooks
//     -> FINE model     (BarkFineModel, NON-causal over the codebook axis)
//        -> the remaining EnCodec codebooks, conditioning on codebooks 0..idx
//     -> EnCodec audio DECODER (BuildEnCodecFromSafeTensors) -> waveform -> WAV
//
// With NO arguments this runs a SELF-CONTAINED pico smoke on the committed
// random fixtures (tests/fixtures/tiny_bark_*.safetensors + the matched
// tiny_musicgen_encodec.* codec). It runs all three stages end-to-end and
// writes a SHORT bark_tts_demo.wav -- the weights are UNTRAINED random so the
// clip is noise, not speech; the point is to exercise the full wiring under
// the time/memory budget. Real suno/bark key-mapping (one nested checkpoint
// with "semantic"/"coarse_acoustics"/"fine_acoustics" prefixes) and a real
// tokenizer/voice prompt are documented follow-ups.

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork, neuralsafetensors, neuralpretrained,
  neuralaudio;

const
  FixDir = '../../tests/fixtures/';
  OutWavName = 'bark_tts_demo.wav';
  SemSeqLen = 6;     // semantic decode window
  CoarseSeqLen = 6;  // coarse decode window
  FineSeqLen = 6;    // fine frames (== coarse frame count)

// Greedy argmax over one position's logits of a (SeqLen*Vocab) volume.
function ArgMaxAt(Logits: TNNetVolume; Pos, Vocab: integer): integer;
var
  t, best: integer;
  bestv: TNeuralFloat;
begin
  best := 0;
  bestv := Logits.FData[Pos * Vocab + 0];
  for t := 1 to Vocab - 1 do
    if Logits.FData[Pos * Vocab + t] > bestv then
    begin
      bestv := Logits.FData[Pos * Vocab + t];
      best := t;
    end;
  Result := best;
end;

var
  Model: TBarkModel;
  Config: TBarkConfig;
  Codec: TEnCodecModel;
  CodecCfg: TEnCodecConfig;
  Logits: TNNetVolume;
  Wave: TNNetVolume;
  Waveform: TNeuralFloatDynArr;
  SemIds, CoarseIds: array of integer;
  Codes: TNNetIntArr2D;       // [n_codes_total][frames] for the EnCodec decode
  FineInput: TNNetIntArr2D;   // [frame][n_codes_total] for the fine model
  i, p, cb, NCodes, FineVocab: integer;
begin
  WriteLn('BarkTTS -- pico text-to-speech smoke (untrained random weights).');
  WriteLn;

  Logits := TNNetVolume.Create;
  Wave := TNNetVolume.Create;
  Model := nil;
  Codec := nil;
  try
    // ---- Build the three Bark GPT sub-models. ----
    Model := BuildBarkFromSafeTensors(
      FixDir + 'tiny_bark_semantic.safetensors',
      FixDir + 'tiny_bark_coarse.safetensors',
      FixDir + 'tiny_bark_fine.safetensors',
      Config, SemSeqLen, CoarseSeqLen, FineSeqLen, {pTrainable=}false,
      FixDir + 'tiny_bark_config.json');
    WriteLn(BarkConfigToString(Config));
    WriteLn;

    NCodes := Config.Fine.NCodesTotal;
    FineVocab := Config.Fine.OutVocab;

    // ---- Stage 1: SEMANTIC. Seed a window of text+semantic ids, run one
    // forward, take the argmax at the last position as the "next" token. ----
    SetLength(SemIds, SemSeqLen);
    for i := 0 to SemSeqLen - 1 do SemIds[i] := i mod Config.Semantic.InVocab;
    Model.Semantic.ComputeLogits(SemIds, Logits);
    WriteLn('Stage 1 SEMANTIC: next-token argmax = ',
      ArgMaxAt(Logits, SemSeqLen - 1, Config.Semantic.OutVocab));

    // ---- Stage 2: COARSE. Feed the semantic ids (clamped to coarse vocab),
    // emit per-frame coarse codebook-0 codes via argmax. ----
    SetLength(CoarseIds, CoarseSeqLen);
    for i := 0 to CoarseSeqLen - 1 do
      CoarseIds[i] := SemIds[i] mod Config.Coarse.InVocab;
    Model.Coarse.ComputeLogits(CoarseIds, Logits);
    SetLength(Codes, NCodes);
    for cb := 0 to NCodes - 1 do SetLength(Codes[cb], FineSeqLen);
    for p := 0 to FineSeqLen - 1 do
      Codes[0][p] := ArgMaxAt(Logits, p, Config.Coarse.OutVocab) mod FineVocab;
    Write('Stage 2 COARSE: codebook 0 codes =');
    for p := 0 to FineSeqLen - 1 do Write(' ', Codes[0][p]);
    WriteLn;

    // ---- Stage 3: FINE. NON-causal over the codebook axis: given codebooks
    // 0..idx-1, predict codebook idx for every frame at once. Build the
    // (frame x n_codes_total) input, fill unknown codebooks with 0, then walk
    // idx = n_codes_given..n_codes_total-1 filling each predicted codebook. ----
    SetLength(FineInput, FineSeqLen);
    for p := 0 to FineSeqLen - 1 do
    begin
      SetLength(FineInput[p], NCodes);
      for cb := 0 to NCodes - 1 do FineInput[p][cb] := 0;
      FineInput[p][0] := Codes[0][p];  // coarse-provided codebook 0
    end;
    for cb := Config.Fine.NCodesGiven to NCodes - 1 do
    begin
      Model.Fine.ComputeFineLogits(FineInput, cb, Logits);
      for p := 0 to FineSeqLen - 1 do
      begin
        Codes[cb][p] := ArgMaxAt(Logits, p, FineVocab);
        FineInput[p][cb] := Codes[cb][p];  // condition the next codebook on it
      end;
    end;
    WriteLn('Stage 3 FINE: filled codebooks ', Config.Fine.NCodesGiven,
      '..', NCodes - 1, ' (non-causal codebook-conditioned).');
    WriteLn;

    // ---- Stage 4: EnCodec decode the [n_codes_total][frames] stack -> WAV. --
    Codec := BuildEnCodecFromSafeTensors(
      FixDir + 'tiny_musicgen_encodec.safetensors', CodecCfg,
      FixDir + 'tiny_musicgen_encodec_config.json');
    if Codec.NumCodebooks < NCodes then
    begin
      WriteLn('FATAL: EnCodec has ', Codec.NumCodebooks,
        ' quantizers, fewer than Bark n_codes_total=', NCodes);
      Halt(1);
    end;
    Codec.DecodeCodesToAudio(Codes, Waveform, NCodes);
    Wave.ReSize(Length(Waveform), 1, 1);
    for i := 0 to Length(Waveform) - 1 do Wave.FData[i] := Waveform[i];
    SaveVolumeToWav16(Wave, OutWavName, CodecCfg.SamplingRate);
    WriteLn('Wrote ', OutWavName, ' (', Length(Waveform), ' samples, ',
      (Length(Waveform) / CodecCfg.SamplingRate):0:3, ' s at ',
      CodecCfg.SamplingRate, ' Hz).');
    WriteLn;
    WriteLn('Done (pico smoke): full SEMANTIC -> COARSE -> FINE -> EnCodec ',
      'wiring exercised on random weights (the clip is noise, not speech).');
  finally
    Model.Free;
    Codec.Free;
    Logits.Free;
    Wave.Free;
  end;
end.
