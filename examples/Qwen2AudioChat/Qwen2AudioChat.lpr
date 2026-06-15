program Qwen2AudioChat;
// Qwen2-Audio audio-understanding demo on the committed pico fixture (or any
// Qwen2-Audio checkpoint passed as argument 1): an audio-in / text-out demo
// (transcribe / caption / answer a question about a clip). It loads the FOUR
// nets returned by BuildQwen2AudioFromSafeTensors (neural/neuralpretrained.pas)
// - the Whisper audio tower, the post-avg_pooler LayerNorm tail, the single
// linear projector, and the Qwen2 language decoder - splices the projected
// audio frames into the decoder's embedding sequence at the <|AUDIO|>
// placeholder slots, and greedily decodes a short answer.
//
// The decode loop is the canonical Qwen2-Audio forward: Qwen2AudioRunLogits
// runs the audio tower + avg_pooler + final LayerNorm + projector once,
// splices the projected audio frames into the decoder's embedding sequence at
// the audio_token_index placeholder slots, and runs the decoder causally; the
// next token is the argmax of the LAST logit row, appended, and the whole
// prompt is re-run (a clear, KV-cache-free reference - the cached fast path is
// a follow-up).
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_qwen2audio.* pico fixture, randomly
// initialized, so the "answer" is gibberish: the point is the AUDIO->TEXT
// PLUMBING):
//   examples/Qwen2AudioChat/Qwen2AudioChat
//   examples/Qwen2AudioChat/Qwen2AudioChat /path/to/qwen2audio/model.safetensors
//
// Memory: cap the process when pointing at a real checkpoint, e.g.
//   ulimit -v 14000000; examples/Qwen2AudioChat/Qwen2AudioChat <real.safetensors>
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, neuralvolume, neuralnetwork, neuralpretrained;

const
  MaxNewTokens = 6;   // short answer (pico LM output is gibberish)

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_qwen2audio.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_qwen2audio.safetensors';
end;

var
  TowerNet, PoolNormNet, ProjectorNet, TextNet: TNNet;
  Config: TQwen2AudioConfig;
  CheckpointPath, ConfigPath: string;
  Mel, MelInput, Logits: TNNetVolume;
  TokenIds: array of integer;
  b, t, Step, BestTok, MelLen: integer;

  // Greedy argmax over the LAST logit row.
  function ArgmaxLastRow(L: TNNetVolume): integer;
  var
    vv, Vocab, Rows: integer;
    Best: TNeuralFloat;
  begin
    Vocab := L.Depth;
    Rows := L.SizeX;
    Result := 0;
    Best := L.FData[(Rows - 1) * Vocab + 0];
    for vv := 1 to Vocab - 1 do
      if L.FData[(Rows - 1) * Vocab + vv] > Best then
      begin
        Best := L.FData[(Rows - 1) * Vocab + vv];
        Result := vv;
      end;
  end;

  // Assembles the prompt ids: 2 text ids, NumAudioTokens <|AUDIO|> tokens,
  // 2 text ids, then every already-generated token.
  procedure AssemblePrompt(const Generated: array of integer);
  var
    i, n, p: integer;
  begin
    n := 4 + Config.NumAudioTokens + Length(Generated);
    SetLength(TokenIds, n);
    p := 0;
    TokenIds[p] := 1; Inc(p);
    TokenIds[p] := 7; Inc(p);
    for i := 0 to Config.NumAudioTokens - 1 do
    begin
      TokenIds[p] := Config.AudioTokenIndex; Inc(p);
    end;
    TokenIds[p] := 12 mod Config.Text.VocabSize; Inc(p);
    TokenIds[p] := 3; Inc(p);
    for i := 0 to High(Generated) do
    begin
      TokenIds[p] := Generated[i]; Inc(p);
    end;
  end;

var
  Generated: array of integer;
begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a Qwen2-Audio .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_qwen2audio.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) +
      'tiny_qwen2audio_config.json'
  else ConfigPath := '';

  WriteLn('Loading ', CheckpointPath, ' ...');
  // First build just to read the config (NumAudioTokens etc.); the decode loop
  // rebuilds at each step's exact prompt length. pSeqLen=1 is the smallest
  // valid context for the config read.
  BuildQwen2AudioFromSafeTensors(CheckpointPath, TowerNet, PoolNormNet,
    ProjectorNet, TextNet, Config, {pSeqLen=}1, {pInferenceOnly=}true,
    ConfigPath);
  WriteLn(Qwen2AudioConfigToString(Config));
  WriteLn;

  MelLen := 2 * Config.Audio.MaxSourcePositions;
  Mel := TNNetVolume.Create(Config.Audio.NumMelBins, 1, MelLen);
  MelInput := TNNetVolume.Create;
  Logits := TNNetVolume.Create;
  SetLength(Generated, 0);
  try
    // ---- the log-mel spectrogram: the fixture's deterministic dyadic test
    // pattern (a real pipeline would compute it from a waveform via the
    // neuralaudio log-mel frontend, padded to 2*max_source_positions frames).
    // Layout is HF input_features (num_mel_bins, mel_len). ----
    for b := 0 to Config.Audio.NumMelBins - 1 do
      for t := 0 to MelLen - 1 do
        Mel.FData[b * MelLen + t] := (((b * 13 + t * 7) * 5) mod 17 - 8) / 8.0;
    Qwen2AudioMelToInput(Mel, MelInput, Config.Audio.NumMelBins, MelLen);

    WriteLn('Greedy answer (', MaxNewTokens, ' tokens, pico = gibberish):');
    Write('  generated token ids:');
    for Step := 0 to MaxNewTokens - 1 do
    begin
      AssemblePrompt(Generated);
      // Rebuild the decoder at the EXACT current prompt length so the causal
      // mask and positions line up (the KV-cache fast path is a follow-up).
      TowerNet.Free; PoolNormNet.Free; ProjectorNet.Free; TextNet.Free;
      BuildQwen2AudioFromSafeTensors(CheckpointPath, TowerNet, PoolNormNet,
        ProjectorNet, TextNet, Config, {pSeqLen=}Length(TokenIds),
        {pInferenceOnly=}true, ConfigPath);
      Qwen2AudioRunLogits(TowerNet, PoolNormNet, ProjectorNet, TextNet,
        MelInput, TokenIds, Config.AudioTokenIndex, Config.NumAudioTokens,
        Logits);
      BestTok := ArgmaxLastRow(Logits);
      SetLength(Generated, Length(Generated) + 1);
      Generated[High(Generated)] := BestTok;
      Write(' ', BestTok);
    end;
    WriteLn;
    WriteLn;
    WriteLn('Done. (The plumbing - audio tower -> avg_pooler -> LayerNorm -> ',
      'projector -> spliced audio frames -> causal decode - is what this demo ',
      'exercises.)');
  finally
    Logits.Free;
    MelInput.Free;
    Mel.Free;
    TowerNet.Free;
    PoolNormNet.Free;
    ProjectorNet.Free;
    TextNet.Free;
  end;
end.
