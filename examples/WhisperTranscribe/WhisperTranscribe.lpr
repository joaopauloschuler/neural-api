program WhisperTranscribe;
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

// WhisperTranscribe -- SPEECH-TO-TEXT with a pretrained OpenAI Whisper
// checkpoint (openai/whisper-tiny and siblings): an audio demo.
// The pipeline is pure CAI Pascal end to end:
//
//   16-bit PCM WAV (16 kHz mono)
//     -> log-mel spectrogram          (neuralaudio.WhisperLogMelFromWavFile:
//        the exact HF WhisperFeatureExtractor recipe - 400-pt hann STFT,
//        hop 160, 80 slaney mel bins, log10, max-8 clamp, (x+4)/4)
//     -> Whisper ENCODER              (conv1d+GELU x2 with a stride-2
//        halving, sinusoidal positions, pre-norm blocks, final LayerNorm)
//     -> Whisper DECODER              (learned positions, causal
//        self-attention + cross-attention over the 1500-frame encoder
//        states, tied LM head) greedily autoregressed from the prologue
//        <|startoftranscript|><|en|><|transcribe|><|notimestamps|>
//     -> GPT-2-style byte-level BPE detokenization (neuralhftokenizer).
//
// Usage:
//   WhisperTranscribe <checkpoint-dir> <audio.wav> [MaxNewTokens]
//
//   checkpoint-dir - directory holding model.safetensors (or
//                    pytorch_model.bin), config.json and tokenizer.json,
//                    e.g. a download of huggingface.co/openai/whisper-tiny.
//   audio.wav      - 16 kHz MONO 16-bit PCM WAV, up to 30 s (longer input
//                    is truncated; shorter is zero-padded, like HF).
//                    Convert anything with:
//                      ffmpeg -i in.ext -ar 16000 -ac 1 out.wav
//   MaxNewTokens   - decode budget after the prologue (default 32; the
//                    decoder net is built at prologue+MaxNewTokens
//                    positions, capped at max_target_positions).
//
// Download the checkpoint (about 151 MB) with:
//   mkdir -p /tmp/whisper-tiny && cd /tmp/whisper-tiny
//   wget https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors
//   wget https://huggingface.co/openai/whisper-tiny/resolve/main/config.json
//   wget https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json
//
// The language/task prologue is resolved from the tokenizer: multilingual
// checkpoints get <|startoftranscript|><|en|><|transcribe|><|notimestamps|>;
// the English-only *.en checkpoints have no language/task tokens, so the
// prologue degrades gracefully to whatever subset exists in the vocabulary.

{$mode objfpc}{$H+}

uses
  {$IFDEF UNIX}cthreads, cmem,{$ENDIF}
  Classes, SysUtils,
  neuralvolume, neuralnetwork,
  neuralpretrained, neuralhftokenizer, neuralaudio;

var
  CheckpointDir, WeightsPath, WavPath: string;
  Enc, Dec: TNNet;
  Config: TWhisperConfig;
  Tokenizer: TNeuralHFTokenizer;
  Mel, DecToks, Logits: TNNetVolume;
  EncStates: TNNetLayer;
  Prologue: array of integer;
  Generated: array of integer;
  MaxNewTokens, DecSeqLen: integer;
  StepCnt, PosCnt, TokCnt, BestId, CurLen: integer;
  BestLogit: TNeuralFloat;
  StartTicks: QWord;

// Appends the id of Token to Prologue when the tokenizer knows it (the
// *.en checkpoints have no language/task specials).
procedure TryAppendSpecial(const Token: string);
var
  Id: integer;
begin
  Id := Tokenizer.TokenToId(Token);
  if Id >= 0 then
  begin
    SetLength(Prologue, Length(Prologue) + 1);
    Prologue[High(Prologue)] := Id;
  end
  else
    WriteLn('  (vocabulary has no ', Token, ' - skipped)');
end;

begin
  if ParamCount < 2 then
  begin
    WriteLn('Usage: WhisperTranscribe <checkpoint-dir> <audio.wav> ' +
      '[MaxNewTokens]');
    WriteLn('  checkpoint-dir: model.safetensors + config.json + ' +
      'tokenizer.json');
    WriteLn('  audio.wav:      16 kHz mono 16-bit PCM, up to 30 s');
    Halt(1);
  end;
  CheckpointDir := IncludeTrailingPathDelimiter(ParamStr(1));
  WavPath := ParamStr(2);
  MaxNewTokens := 32;
  if ParamCount >= 3 then MaxNewTokens := StrToIntDef(ParamStr(3), 32);

  WeightsPath := CheckpointDir + 'model.safetensors';
  if not FileExists(WeightsPath) then
    WeightsPath := CheckpointDir + 'pytorch_model.bin';
  if not FileExists(WeightsPath) then
  begin
    WriteLn('No model.safetensors / pytorch_model.bin in ', CheckpointDir);
    Halt(1);
  end;

  // ---- tokenizer (needed for the prologue ids and the final text) ----
  Tokenizer := TNeuralHFTokenizer.Create;
  Tokenizer.LoadFromFile(CheckpointDir + 'tokenizer.json');

  // ---- read the config first to size the decoder ----
  Config := ReadWhisperConfigFromJSONFile(CheckpointDir + 'config.json');
  WriteLn('Loaded ', WhisperConfigToString(Config));

  // The decode prologue: <|startoftranscript|> is decoder_start_token_id;
  // language/task/timestamps come from the tokenizer's added specials.
  SetLength(Prologue, 1);
  Prologue[0] := Config.DecoderStartTokenId;
  TryAppendSpecial('<|en|>');
  TryAppendSpecial('<|transcribe|>');
  TryAppendSpecial('<|notimestamps|>');
  Write('Prologue ids:');
  for TokCnt := 0 to High(Prologue) do Write(' ', Prologue[TokCnt]);
  WriteLn;

  DecSeqLen := Length(Prologue) + MaxNewTokens;
  if DecSeqLen > Config.MaxTargetPositions then
    DecSeqLen := Config.MaxTargetPositions;

  // ---- build the two nets and load the weights ----
  StartTicks := GetTickCount64;
  BuildWhisperFromSafeTensorsWithConfig(WeightsPath, Config, Enc, Dec,
    DecSeqLen, {pTrainable=}false);
  WriteLn('Built encoder (', Enc.CountLayers, ' layers) + decoder (',
    Dec.CountLayers, ' layers) in ',
    (GetTickCount64 - StartTicks) div 1000, ' s');

  // ---- WAV -> log-mel -> encoder ----
  Mel := TNNetVolume.Create;
  DecToks := TNNetVolume.Create;
  Logits := TNNetVolume.Create;
  try
    StartTicks := GetTickCount64;
    WhisperLogMelFromWavFile(WavPath, Mel, Config.NumMelBins,
      2 * Config.MaxSourcePositions);
    WriteLn('Log-mel spectrogram (', Mel.SizeX, ' frames x ', Mel.Depth,
      ' mel bins) in ', (GetTickCount64 - StartTicks) div 1000, ' s');
    StartTicks := GetTickCount64;
    Enc.Compute(Mel);
    WriteLn('Encoded to ', Config.MaxSourcePositions,
      ' hidden states in ', (GetTickCount64 - StartTicks) div 1000, ' s');
    // The encoder states are constant across decode steps: copy them into
    // the decoder's second input ONCE (the RunT5/T5EncoderStatesInput
    // convention shared with the T5/Marian importers).
    EncStates := T5EncoderStatesInput(Dec);
    EncStates.Output.Copy(Enc.GetLastLayer().Output);

    // ---- greedy decode from the multi-token prologue ----
    // Each step re-runs the full decoder on the growing prefix; unused
    // positions are padded with the start id, which the causal mask makes
    // invisible to the rows actually read.
    StartTicks := GetTickCount64;
    SetLength(Generated, 0);
    CurLen := Length(Prologue);
    DecToks.ReSize(DecSeqLen, 1, 1);
    Write('Tokens:');
    while CurLen < DecSeqLen do
    begin
      for PosCnt := 0 to DecSeqLen - 1 do
        if PosCnt < Length(Prologue) then
          DecToks.FData[PosCnt] := Prologue[PosCnt]
        else if PosCnt - Length(Prologue) <= High(Generated) then
          DecToks.FData[PosCnt] := Generated[PosCnt - Length(Prologue)]
        else
          DecToks.FData[PosCnt] := Config.DecoderStartTokenId;
      Dec.Compute(DecToks);
      Dec.GetOutput(Logits);
      // argmax over the logits row at the last prefix position
      BestId := 0;
      BestLogit := Logits.FData[(CurLen - 1) * Config.VocabSize];
      for TokCnt := 1 to Config.VocabSize - 1 do
        if Logits.FData[(CurLen - 1) * Config.VocabSize + TokCnt] >
           BestLogit then
        begin
          BestLogit := Logits.FData[(CurLen - 1) * Config.VocabSize +
            TokCnt];
          BestId := TokCnt;
        end;
      Write(' ', BestId);
      if BestId = Config.EosTokenId then break;
      SetLength(Generated, Length(Generated) + 1);
      Generated[High(Generated)] := BestId;
      Inc(CurLen);
    end;
    WriteLn;
    WriteLn('Decoded ', Length(Generated), ' tokens in ',
      (GetTickCount64 - StartTicks) div 1000, ' s');
    WriteLn;
    WriteLn('Transcription:', Tokenizer.Decode(Generated));
  finally
    Logits.Free;
    DecToks.Free;
    Mel.Free;
    Dec.Free;
    Enc.Free;
    Tokenizer.Free;
  end;
end.
