program Wav2Vec2Transcribe;
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

// Wav2Vec2Transcribe -- SPEECH-TO-TEXT with a pretrained Wav2Vec2 / HuBERT
// CTC checkpoint (facebook/wav2vec2-base-960h, facebook/hubert-large-
// ls960-ft and siblings): an audio demo and a
// SELF-SUPERVISED-encoder ASR. Unlike WhisperTranscribe (a mel-
// spectrogram encoder-decoder that autoregresses tokens), this pipeline is
// a RAW-WAVEFORM conv feature extractor -> transformer encoder -> linear
// CTC head, decoded in ONE forward pass with greedy CTC collapse:
//
//   16-bit PCM WAV (16 kHz mono)
//     -> raw float samples            (neuralaudio.LoadWav16ToVolume:
//        mono floats in [-1, 1), (N,1,1)), optionally zero-mean/unit-var
//        normalized (the base-960h feature extractor's do_normalize=true);
//     -> Wav2Vec2 / HuBERT CTC net    (BuildWav2Vec2FromSafeTensors:
//        strided 1-D conv feature extractor with first-conv GroupNorm,
//        feature projection, a grouped-conv relative positional embedding,
//        a post-LN bidirectional transformer encoder, and a linear CTC
//        head over the char/phoneme vocab);
//     -> CTC GREEDY decode            (neuraldecode.DecodeCTCGreedy:
//        per-frame argmax, collapse repeats, drop the blank);
//     -> vocab.json id->token detokenization (the word delimiter "|" is
//        rendered as a space; ids are a tiny char vocab, NOT SentencePiece).
//
// Usage:
//   Wav2Vec2Transcribe <checkpoint-dir> <audio.wav> [--no-normalize]
//
//   checkpoint-dir - directory holding model.safetensors (or
//                    pytorch_model.bin), config.json and vocab.json,
//                    e.g. a download of facebook/wav2vec2-base-960h.
//   audio.wav      - 16 kHz MONO 16-bit PCM WAV. Convert anything with:
//                      ffmpeg -i in.ext -ar 16000 -ac 1 out.wav
//   --no-normalize - skip the zero-mean/unit-variance normalization
//                    (use for checkpoints whose feature extractor pins
//                    do_normalize=false).
//
// Download a checkpoint (about 360 MB) with:
//   mkdir -p /tmp/w2v && cd /tmp/w2v
//   wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors
//   wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json
//   wget https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json
//
// With NO arguments the demo runs a SELF-CONTAINED smoke test on a
// synthesized tone using the committed pico fixture, so it builds and runs
// with zero downloads (the transcription is gibberish - the fixture is
// random - but it proves the whole raw-audio -> CTC path end to end).

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils, Math, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained, neuralaudio, neuraldecode;

var
  CheckpointDir, WeightsPath, WavPath, VocabPath: string;
  NN: TNNet;
  Config: TWav2Vec2Config;
  Samples, Logits: TNNetVolume;
  Decoded: TNeuralIntegerArray;
  Vocab: array of string;
  Normalize: boolean;
  i, BlankId, NumSamples: integer;
  StartTicks: QWord;
  Transcript: string;

// Loads vocab.json ({"<token>": id, ...}) into Vocab[id] = token. The
// special "|" word-delimiter is rendered as a space at decode time.
procedure LoadVocab(const FileName: string);
var
  RefJson: TStringList;
  Root: TJSONData;
  Obj: TJSONObject;
  k, Id, MaxId: integer;
begin
  RefJson := TStringList.Create;
  Root := nil;
  try
    RefJson.LoadFromFile(FileName);
    Root := GetJSON(RefJson.Text);
    Obj := TJSONObject(Root);
    MaxId := -1;
    for k := 0 to Obj.Count - 1 do
    begin
      Id := Obj.Items[k].AsInteger;
      if Id > MaxId then MaxId := Id;
    end;
    SetLength(Vocab, MaxId + 1);
    for k := 0 to Obj.Count - 1 do
    begin
      Id := Obj.Items[k].AsInteger;
      if (Id >= 0) and (Id <= MaxId) then Vocab[Id] := Obj.Names[k];
    end;
  finally
    Root.Free;
    RefJson.Free;
  end;
end;

// Renders a decoded CTC id sequence to text via Vocab; "|" -> space, the
// pad/<s>/</s>/<unk> specials are skipped.
function DecodeToText(const Ids: TNeuralIntegerArray): string;
var
  k: integer;
  Tok: string;
begin
  Result := '';
  for k := 0 to High(Ids) do
  begin
    if (Ids[k] < 0) or (Ids[k] > High(Vocab)) then continue;
    Tok := Vocab[Ids[k]];
    if Tok = '|' then Result := Result + ' '
    else if (Tok = '<pad>') or (Tok = '<s>') or (Tok = '</s>') or
            (Tok = '<unk>') then continue
    else Result := Result + Tok;
  end;
end;

// Zero-mean / unit-variance per-utterance normalization (the base-960h
// feature extractor's do_normalize).
procedure NormalizeSamples(V: TNNetVolume);
var
  k: integer;
  Mean, Variance, Std: TNeuralFloat;
begin
  if V.Size < 1 then exit;
  Mean := 0;
  for k := 0 to V.Size - 1 do Mean := Mean + V.FData[k];
  Mean := Mean / V.Size;
  Variance := 0;
  for k := 0 to V.Size - 1 do Variance := Variance + Sqr(V.FData[k] - Mean);
  Variance := Variance / V.Size;
  Std := Sqrt(Variance + 1e-7);
  for k := 0 to V.Size - 1 do V.FData[k] := (V.FData[k] - Mean) / Std;
end;

begin
  Samples := TNNetVolume.Create;
  Logits := TNNetVolume.Create;
  NN := nil;
  try
    Normalize := True;
    if ParamCount < 2 then
    begin
      // ---- self-contained smoke test on the committed pico fixture ----
      WriteLn('No <checkpoint-dir> <audio.wav> given - running the ',
        'self-contained pico smoke test (synthetic tone, random fixture; ',
        'the transcript is gibberish but proves the raw-audio -> CTC path).');
      CheckpointDir := IncludeTrailingPathDelimiter(
        ExtractFilePath(ParamStr(0)) + '../../tests/fixtures');
      WeightsPath := CheckpointDir + 'tiny_wav2vec2.safetensors';
      if not FileExists(WeightsPath) then
        WeightsPath := 'tests/fixtures/tiny_wav2vec2.safetensors';
      NumSamples := 200;
      Samples.ReSize(NumSamples, 1, 1);
      for i := 0 to NumSamples - 1 do
        Samples.FData[i] := Sin(0.07 * i) * 0.5;
      Config := ReadWav2Vec2ConfigFromJSONFile(
        ExtractFilePath(WeightsPath) + 'tiny_wav2vec2_config.json');
      NN := BuildWav2Vec2FromSafeTensorsWithConfig(WeightsPath, Config,
        NumSamples, {pInferenceOnly=}true);
      WriteLn('Built ', Wav2Vec2ConfigToString(Config));
      NN.Compute(Samples);
      NN.GetOutput(Logits);
      Decoded := DecodeCTCGreedy(Logits, {Blank=}Config.VocabSize - 1);
      Write('CTC greedy ids:');
      for i := 0 to High(Decoded) do Write(' ', Decoded[i]);
      WriteLn;
      WriteLn('(', NN.GetLastLayer().Output.SizeX,
        ' CTC frames over a ', NumSamples, '-sample clip)');
      Halt(0);
    end;

    CheckpointDir := IncludeTrailingPathDelimiter(ParamStr(1));
    WavPath := ParamStr(2);
    if (ParamCount >= 3) and (ParamStr(3) = '--no-normalize') then
      Normalize := False;

    WeightsPath := CheckpointDir + 'model.safetensors';
    if not FileExists(WeightsPath) then
      WeightsPath := CheckpointDir + 'pytorch_model.bin';
    if not FileExists(WeightsPath) then
    begin
      WriteLn('No model.safetensors / pytorch_model.bin in ', CheckpointDir);
      Halt(1);
    end;
    VocabPath := CheckpointDir + 'vocab.json';
    if not FileExists(VocabPath) then
    begin
      WriteLn('No vocab.json in ', CheckpointDir,
        ' - needed to render the CTC ids to text.');
      Halt(1);
    end;

    // ---- WAV -> raw float samples ----
    if LoadWav16ToVolume(WavPath, Samples) <> 16000 then
    begin
      WriteLn('WAV must be 16 kHz mono 16-bit PCM (convert with ',
        'ffmpeg -ar 16000 -ac 1).');
      Halt(1);
    end;
    if Normalize then NormalizeSamples(Samples);
    NumSamples := Samples.SizeX;
    WriteLn('Loaded ', NumSamples, ' samples (',
      (NumSamples div 16000), ' s) from ', WavPath);

    // ---- read config, build the net at this clip length, load weights ----
    Config := ReadWav2Vec2ConfigFromJSONFile(CheckpointDir + 'config.json');
    WriteLn('Loaded ', Wav2Vec2ConfigToString(Config));
    LoadVocab(VocabPath);
    StartTicks := GetTickCount64;
    NN := BuildWav2Vec2FromSafeTensorsWithConfig(WeightsPath, Config,
      NumSamples, {pInferenceOnly=}true);
    WriteLn('Built ', NN.CountLayers, ' layers in ',
      (GetTickCount64 - StartTicks) div 1000, ' s');

    // ---- forward pass -> CTC logits ----
    StartTicks := GetTickCount64;
    NN.Compute(Samples);
    NN.GetOutput(Logits);
    WriteLn('Encoded ', NN.GetLastLayer().Output.SizeX,
      ' CTC frames in ', (GetTickCount64 - StartTicks) div 1000, ' s');

    // ---- CTC greedy decode (blank = last vocab id, the wav2vec2 <pad>) ----
    BlankId := Config.VocabSize - 1;
    Decoded := DecodeCTCGreedy(Logits, BlankId);
    Transcript := DecodeToText(Decoded);
    WriteLn;
    WriteLn('Transcription: ', UpperCase(Trim(Transcript)));
  finally
    NN.Free;
    Logits.Free;
    Samples.Free;
  end;
end.
