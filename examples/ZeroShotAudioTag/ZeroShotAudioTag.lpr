program ZeroShotAudioTag;
// CLAP zero-shot audio tagging structure on the committed pico fixture (or
// any clap-htsat-UNfused checkpoint passed as argument 1): embed one audio
// clip's log-mel spectrogram through the HTS-AT (Swin) audio tower and N
// text prompts through the RoBERTa text tower with the TWO nets returned by
// BuildClapFromSafeTensors (neural/neuralpretrained.pas - the audio-domain
// analogue of the CLIP dual encoder), score them in the shared space with
// exp(logit_scale_a) * cosine, and print the cosine-similarity matrix plus
// the top-1 zero-shot label.
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_clap.* pico fixture):
//   examples/ZeroShotAudioTag/ZeroShotAudioTag
//   examples/ZeroShotAudioTag/ZeroShotAudioTag /path/to/clap-htsat-unfused/model.safetensors
//
// The clip's log-mel is SYNTHETIC (a deterministic pattern; a real pipeline
// would compute it with the log-mel frontend in neural/neuralaudio.pas) and
// the text "prompts" are synthetic token-id sequences (the RoBERTa BPE
// tokenizer is out of this demo's scope): the point is the zero-shot
// SCORING STRUCTURE - real audio + text just swap in the real frontends.
// v1 supports freq_ratio = 1 (spec_size = num_mel_bins) only.
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, neuralvolume, neuralnetwork, neuralsafetensors,
  neuralpretrained;

const
  NumPrompts = 3;
  // Each "prompt" is a token-id sequence in the pico vocab (no pad id 1).
  PicoPrompts: array[0..NumPrompts - 1, 0..7] of integer = (
    (5, 12, 23,  7, 31,  2,  9, 38),
    (8, 19,  3, 27, 14,  6, 22, 30),
    (11, 4, 33, 21, 16, 28, 17,  9));
  PicoLabels: array[0..NumPrompts - 1] of string = (
    'a dog barking',
    'a vacuum cleaner',
    'rain on a window');

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_clap.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_clap.safetensors';
end;

var
  AudioNet, TextNet: TNNet;
  Reader: TNNetSafeTensorsReader;
  Config: TClapConfig;
  CheckpointPath, ConfigPath: string;
  RawMel, AudioImage, TextInput: TNNetVolume;
  AudioEmb: TNNetVolume;
  AudioEmbs, TextEmbs: array of TNNetVolume;
  SimMatrix: TNNetVolume;
  Mel, Time, SeqLen, PromptCnt, PosCnt, t, f, TokenId, BestIdx: integer;
  BestSim, Sim: TNeuralFloat;
begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a CLAP .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_clap.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_clap_config.json'
  else ConfigPath := '';

  WriteLn('Loading ', CheckpointPath, ' ...');
  BuildClapFromSafeTensors(CheckpointPath, AudioNet, TextNet, Config,
    {TextSeqLen=}8, {pInferenceOnly=}true, ConfigPath);
  WriteLn(ClapConfigToString(Config));
  WriteLn;

  Mel := Config.Audio.NumMelBins;
  Time := Config.Audio.SpecSize;          // freq_ratio = 1
  SeqLen := TextNet.Layers[0].Output.SizeX;
  // ClapBatchNormMelImage needs the checkpoint's batch_norm stats.
  Reader := CreatePretrainedTensorReader(CheckpointPath);
  RawMel := TNNetVolume.Create(Time, 1, Mel);
  AudioImage := TNNetVolume.Create;
  AudioEmb := TNNetVolume.Create;
  TextInput := TNNetVolume.Create(SeqLen, 1, 2);
  SimMatrix := TNNetVolume.Create;
  SetLength(AudioEmbs, 1);
  SetLength(TextEmbs, NumPrompts);
  try
    // ---- the clip: a deterministic synthetic log-mel (time, 1, mel) ----
    for t := 0 to Time - 1 do
      for f := 0 to Mel - 1 do
        RawMel.FData[t * Mel + f] := (((t * 31 + f * 7) mod 23) - 11) / 11.0;
    // apply the HTS-AT batch_norm + mel2img affine, then the audio tower.
    ClapBatchNormMelImage(Reader, RawMel, AudioImage, Config.Audio);
    AudioNet.Compute(AudioImage);
    ClipExtractEmbedding(AudioNet.GetLastLayer().Output, 0, AudioEmb);
    AudioEmbs[0] := AudioEmb;

    // ---- the N text prompts through the text tower (token-type 0) ----
    for PromptCnt := 0 to NumPrompts - 1 do
    begin
      for PosCnt := 0 to SeqLen - 1 do
      begin
        TokenId := PicoPrompts[PromptCnt, PosCnt mod 8] mod Config.Text.VocabSize;
        TextInput.FData[PosCnt * 2] := TokenId;
        TextInput.FData[PosCnt * 2 + 1] := 0;
      end;
      TextNet.Compute(TextInput);
      TextEmbs[PromptCnt] := TNNetVolume.Create;
      ClipExtractEmbedding(TextNet.GetLastLayer().Output, 0, TextEmbs[PromptCnt]);
    end;

    // ---- cosine-similarity matrix (1 audio row x N text cols) ----
    ClapSimilarityMatrix(AudioEmbs, TextEmbs, Config.LogitScaleA, SimMatrix);
    WriteLn('Audio<->text cosine similarity (raw, pre-scale):');
    BestIdx := 0;
    BestSim := -2;
    for PromptCnt := 0 to NumPrompts - 1 do
    begin
      Sim := ClipSimilarity(AudioEmb, TextEmbs[PromptCnt]);
      WriteLn(Format('  clip  vs  %-20s cosine = %7.4f   logit = %8.4f',
        [PicoLabels[PromptCnt], Sim, SimMatrix[PromptCnt, 0, 0]]));
      if Sim > BestSim then begin BestSim := Sim; BestIdx := PromptCnt; end;
    end;
    WriteLn;
    WriteLn(Format('Top-1 zero-shot label: "%s" (cosine %.4f)',
      [PicoLabels[BestIdx], BestSim]));
  finally
    for PromptCnt := 0 to High(TextEmbs) do TextEmbs[PromptCnt].Free;
    SimMatrix.Free;
    TextInput.Free;
    AudioEmb.Free;
    AudioImage.Free;
    RawMel.Free;
    Reader.Free;
    TextNet.Free;
    AudioNet.Free;
  end;
end.
