program MusicTagging;
// MERT music-representation embedding on the committed pico fixture (or any
// m-a-p/MERT-v1-95M checkpoint passed as argument 1). MERT is a self-
// supervised MUSIC understanding encoder - the audio analogue of a frozen
// vision backbone: a HuBERT-style raw-waveform conv feature extractor + a
// transformer trunk whose deep WEIGHTED-LAYER-SUM of hidden states is a
// fixed music embedding for tagging / genre / similarity.
//
// This example builds the MERT encoder with BuildMERTFromSafeTensorsEx
// (neural/neuralpretrained.pas), runs TWO synthesized raw "clips" through it,
// pools each into the fixed music embedding with MERTWeightedLayerSum (the
// learned per-layer softmax weights from the config, then a mean over the
// time frames - the HF *ForSequenceClassification pooled_output), prints the
// embeddings, and scores their CosineSimilarity (the same cosine the CLAP /
// ClipSimilarity examples use - a higher value = more musically similar).
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_mert pico fixture):
//   examples/MusicTagging/MusicTagging
//   examples/MusicTagging/MusicTagging /path/to/MERT-v1-95M/model.safetensors
//
// The two "clips" are SYNTHETIC raw waveforms (deterministic sine + noise
// patterns; a real pipeline would feed a 24 kHz mono waveform): the point is
// the EMBEDDING + SIMILARITY structure - a real clip just swaps in the audio.
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, neuralvolume, neuralnetwork, neuralsafetensors,
  neuralpretrained;

const
  RawLen = 200;          // raw samples per synthetic clip (pico-scale)
  NumClips = 2;

// Builds a deterministic synthetic raw waveform clip (sine of frequency
// proportional to Variant, plus a small periodic noise floor).
procedure SynthClip(Variant: integer; V: TNNetVolume);
var
  t: integer;
begin
  V.ReSize(RawLen, 1, 1);
  for t := 0 to RawLen - 1 do
    V.FData[t] := Sin((Variant + 1) * 0.07 * t) * 0.5 +
      0.1 * (((7 * t + 3 * Variant) mod 5) - 2);
end;

procedure PrintEmbedding(const Tag: string; V: TNNetVolume);
var
  c, ShowN: integer;
  Line: string;
begin
  ShowN := V.Size;
  if ShowN > 8 then ShowN := 8;
  Line := Tag + ' embedding [' + IntToStr(V.Size) + 'd]: ';
  for c := 0 to ShowN - 1 do
    Line := Line + FormatFloat('0.000', V.FData[c]) + ' ';
  if V.Size > ShowN then Line := Line + '...';
  WriteLn(Line);
end;

var
  WeightsPath, ConfigPath: string;
  NN: TNNet;
  Config: TMERTConfig;
  HiddenLayers: TMERTHiddenStateArray;
  Input: TNNetVolume;
  Emb: array[0..NumClips - 1] of TNNetVolume;
  ClipCnt: integer;
  Cos: TNeuralFloat;
begin
  if ParamCount >= 1 then WeightsPath := ParamStr(1)
  else WeightsPath := 'tests/fixtures/tiny_mert.safetensors';
  if not FileExists(WeightsPath) then
  begin
    WriteLn('Checkpoint not found: ', WeightsPath);
    WriteLn('Run from the repo root, or pass a MERT-v1-95M model.safetensors.');
    Halt(1);
  end;
  // The pico fixture keeps its config next to the weights as *_config.json.
  ConfigPath := ChangeFileExt(WeightsPath, '') + '_config.json';
  if not FileExists(ConfigPath) then
    ConfigPath := ExtractFilePath(WeightsPath) + 'config.json';

  WriteLn('MERT music-representation embedding example');
  WriteLn('  checkpoint : ', WeightsPath);
  WriteLn('  config     : ', ConfigPath);

  Input := TNNetVolume.Create;
  for ClipCnt := 0 to NumClips - 1 do Emb[ClipCnt] := TNNetVolume.Create;
  NN := nil;
  try
    NN := BuildMERTFromSafeTensorsEx(WeightsPath, Config, RawLen,
      HiddenLayers, {pTrainable=}false, ConfigPath);
    WriteLn('  ', MERTConfigToString(Config));
    WriteLn('  hidden states for weighted-layer-sum: ',
      Length(HiddenLayers), ' (num_hidden_layers+1)');
    WriteLn('  encoder frames: ', MERTEncoderLength(Config, RawLen));
    WriteLn;

    for ClipCnt := 0 to NumClips - 1 do
    begin
      SynthClip(ClipCnt, Input);
      NN.Compute(Input);
      // Pool the N+1 hidden states into the fixed music embedding.
      MERTWeightedLayerSum(Config, HiddenLayers, Emb[ClipCnt], {MeanPool=}true);
      PrintEmbedding('clip ' + IntToStr(ClipCnt), Emb[ClipCnt]);
    end;
    WriteLn;

    Cos := CosineSimilarity(Emb[0], Emb[1]);
    WriteLn('cosine similarity(clip 0, clip 1) = ', FormatFloat('0.0000', Cos));
    if Cos > 0.95 then WriteLn('  -> the two clips are musically very similar.')
    else if Cos > 0.5 then WriteLn('  -> the two clips share some structure.')
    else WriteLn('  -> the two clips are musically distinct.');
  finally
    for ClipCnt := 0 to NumClips - 1 do Emb[ClipCnt].Free;
    Input.Free;
    NN.Free;
  end;
end.
