program PaliGemmaCaption;
// PaliGemma generative vision-language captioning on the committed pico
// fixture (or any PaliGemma checkpoint passed as argument 1): the FIRST
// PREFIX-LM vision-language demo in the repo. It loads the THREE nets returned
// by BuildPaliGemmaFromSafeTensors (neural/neuralpretrained.pas) - the SigLIP
// vision tower (last_hidden_state, WITH post_layernorm), the single-linear
// multimodal projector, and the Gemma language decoder - assembles a
// multimodal prompt ([<image>*NumPatches | prompt-text | generated-suffix]),
// and greedily decodes a short caption under the PREFIX-LM attention mask.
//
// What makes this DIFFERENT from LlavaDescribe is the attention regime:
// PaliGemma is a PREFIX-LM. The image tokens AND the prompt tokens (the
// "prefix") see each other with FULL BIDIRECTIONAL attention; ONLY the
// generated suffix is causal. PaliGemmaRunLogits sets the prefix-LM
// bidirectional block to PrefixLen on every SDPA layer
// (TNNet.SetAttentionPrefixLen) for the duration of the forward, then restores
// pure causal. PrefixLen stays FIXED at the original image+prompt length while
// the generated suffix grows.
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_paligemma.* pico fixture, randomly
// initialized, so the "caption" is gibberish: the point is the IMAGE->TEXT
// PREFIX-LM PLUMBING):
//   examples/PaliGemmaCaption/PaliGemmaCaption
//   examples/PaliGemmaCaption/PaliGemmaCaption /path/to/paligemma/model.safetensors
//
// Memory: cap the process when pointing at a real checkpoint, e.g.
//   ulimit -v 12000000; examples/PaliGemmaCaption/PaliGemmaCaption <real.safetensors>
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, neuralvolume, neuralnetwork, neuralpretrained;

const
  MaxNewTokens = 5;   // short caption (pico LM output is gibberish)

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_paligemma.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_paligemma.safetensors';
end;

var
  VisionNet, ProjectorNet, TextNet: TNNet;
  Config: TPaliGemmaConfig;
  CheckpointPath, ConfigPath: string;
  ImageInput, Logits: TNNetVolume;
  TokenIds: array of integer;
  Generated: array of integer;
  C, Y, X, Step, BestTok, PrefixLen: integer;

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

  // Assembles the prompt ids: NumPatches image tokens, then 2 prompt-text ids
  // (the PREFIX, token_type_id 0 = bidirectional), then every already-generated
  // suffix token (token_type_id 1 = causal). PaliGemma lays the image FIRST.
  procedure AssemblePrompt(const Generated: array of integer);
  var
    i, n, p: integer;
  begin
    n := Config.NumPatches + 2 + Length(Generated);
    SetLength(TokenIds, n);
    p := 0;
    for i := 0 to Config.NumPatches - 1 do
    begin
      TokenIds[p] := Config.ImageTokenIndex; Inc(p);
    end;
    TokenIds[p] := 7; Inc(p);
    TokenIds[p] := 12 mod Config.Text.VocabSize; Inc(p);
    for i := 0 to High(Generated) do
    begin
      TokenIds[p] := Generated[i]; Inc(p);
    end;
  end;

begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a PaliGemma .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_paligemma.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_paligemma_config.json'
  else ConfigPath := '';

  WriteLn('Loading ', CheckpointPath, ' ...');
  BuildPaliGemmaFromSafeTensors(CheckpointPath, VisionNet, ProjectorNet,
    TextNet, Config, {pSeqLen=}1, {pInferenceOnly=}true, ConfigPath);
  WriteLn(PaliGemmaConfigToString(Config));
  WriteLn;

  // The PREFIX is the image block + the 2 prompt-text tokens; it stays FIXED
  // while the generated suffix grows. Every prefix position attends
  // bidirectionally; the suffix is causal.
  PrefixLen := Config.NumPatches + 2;

  ImageInput := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumChannels);
  Logits := TNNetVolume.Create;
  SetLength(Generated, 0);
  try
    // ---- the image: the fixture's deterministic dyadic test pattern (a real
    // pipeline would load + SigLIP-normalize a photo). ----
    for C := 0 to Config.NumChannels - 1 do
      for Y := 0 to Config.ImageSize - 1 do
        for X := 0 to Config.ImageSize - 1 do
          ImageInput[X, Y, C] :=
            (((C * 256 + Y * 16 + X) * 5) mod 17 - 8) / 8.0;

    WriteLn('Greedy caption (', MaxNewTokens, ' tokens, prefix-LM mask, ',
      'pico = gibberish):');
    WriteLn('  prefix_len = ', PrefixLen, ' (', Config.NumPatches,
      ' image + 2 prompt tokens, bidirectional)');
    Write('  generated token ids:');
    for Step := 0 to MaxNewTokens - 1 do
    begin
      AssemblePrompt(Generated);
      // Rebuild the decoder at the EXACT current prompt length so the mask and
      // positions line up (the KV-cache fast path is a follow-up).
      VisionNet.Free; ProjectorNet.Free; TextNet.Free;
      BuildPaliGemmaFromSafeTensors(CheckpointPath, VisionNet, ProjectorNet,
        TextNet, Config, {pSeqLen=}Length(TokenIds), {pInferenceOnly=}true,
        ConfigPath);
      PaliGemmaRunLogits(VisionNet, ProjectorNet, TextNet, ImageInput, TokenIds,
        Config.ImageTokenIndex, Config.NumPatches, PrefixLen, Logits);
      BestTok := ArgmaxLastRow(Logits);
      SetLength(Generated, Length(Generated) + 1);
      Generated[High(Generated)] := BestTok;
      Write(' ', BestTok);
    end;
    WriteLn;
    WriteLn;
    WriteLn('Done. (The plumbing - SigLIP tower -> linear projector -> spliced ',
      'visual tokens -> PREFIX-LM decode - is what this demo exercises.)');
  finally
    Logits.Free;
    ImageInput.Free;
    VisionNet.Free;
    ProjectorNet.Free;
    TextNet.Free;
  end;
end.
