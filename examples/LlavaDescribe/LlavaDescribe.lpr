program LlavaDescribe;
// LLaVA generative vision-language captioning on the committed pico fixture
// (or any LLaVA checkpoint passed as argument 1): an image-in /
// text-out demo. It loads the THREE nets returned by
// BuildLlavaFromSafeTensors (neural/neuralpretrained.pas) - the SigLIP/CLIP
// vision tower, the 2-layer gelu projector, and the Qwen2/Llama language
// decoder - assembles a multimodal prompt with the cfLlava chat template
// (neuralchat.pas: "...USER: <image>\n<question> ASSISTANT:"), and greedily
// decodes a short caption.
//
// The decode loop is the canonical LLaVA forward: LlavaRunLogits runs the
// vision tower + projector once, splices the projected visual tokens into the
// decoder's embedding sequence at the image_token_index placeholder slots,
// and runs the decoder causally; the next token is the argmax of the LAST
// logit row, appended, and the whole prompt is re-run (a clear, KV-cache-free
// reference - the cached fast path is a follow-up).
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_llava.* pico fixture, randomly initialized,
// so the "caption" is gibberish: the point is the IMAGE->TEXT PLUMBING):
//   examples/LlavaDescribe/LlavaDescribe
//   examples/LlavaDescribe/LlavaDescribe /path/to/llava/model.safetensors
//
// Memory: cap the process when pointing at a real checkpoint, e.g.
//   ulimit -v 12000000; examples/LlavaDescribe/LlavaDescribe <real.safetensors>
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, neuralvolume, neuralnetwork, neuralpretrained, neuralchat;

const
  MaxNewTokens = 6;   // short caption (pico LM output is gibberish)

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_llava.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_llava.safetensors';
end;

var
  VisionNet, ProjectorNet, TextNet: TNNet;
  Config: TLlavaConfig;
  CheckpointPath, ConfigPath, Prompt: string;
  ImageInput, Logits: TNNetVolume;
  Messages: TChatMessages;
  TokenIds: array of integer;
  C, Y, X, Step, BestTok: integer;

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

  // Assembles the fixed prompt ids: 2 text ids, NumPatches image tokens,
  // 2 text ids, then every already-generated token.
  procedure AssemblePrompt(const Generated: array of integer);
  var
    i, n, p: integer;
  begin
    n := 4 + Config.NumPatches + Length(Generated);
    SetLength(TokenIds, n);
    p := 0;
    TokenIds[p] := 1; Inc(p);
    TokenIds[p] := 7; Inc(p);
    for i := 0 to Config.NumPatches - 1 do
    begin
      TokenIds[p] := Config.ImageTokenIndex; Inc(p);
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
    WriteLn('Pass a LLaVA .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_llava.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_llava_config.json'
  else ConfigPath := '';

  // ---- the multimodal chat prompt (cfLlava). The user content carries the
  // "<image>\n" placeholder; in a real pipeline the tokenizer expands it to
  // NumPatches image_token ids. Here we print the rendered template to show
  // the format, then drive the decode with explicit ids below. ----
  SetLength(Messages, 1);
  Messages[0] := ChatMessage('user', '<image>' + LineEnding +
    'Describe the image.');
  Prompt := ApplyChatTemplate(cfLlava, Messages, {AddGenerationPrompt=}true);
  WriteLn('Chat prompt (cfLlava):');
  WriteLn('  ', Prompt);
  WriteLn;

  WriteLn('Loading ', CheckpointPath, ' ...');
  // First build just to read the config (NumPatches etc.); the decode loop
  // rebuilds at each step's exact prompt length. pSeqLen=1 is the smallest
  // valid context for the config read.
  BuildLlavaFromSafeTensors(CheckpointPath, VisionNet, ProjectorNet, TextNet,
    Config, {pSeqLen=}1, {pInferenceOnly=}true, ConfigPath);
  WriteLn(LlavaConfigToString(Config));
  WriteLn;

  ImageInput := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumChannels);
  Logits := TNNetVolume.Create;
  SetLength(Generated, 0);
  try
    // ---- the image: the fixture's deterministic dyadic test pattern (a real
    // pipeline would load + CLIP/SigLIP-normalize a photo via
    // ReadClipImageProcessorConfig + ClipPreprocessImage). ----
    for C := 0 to Config.NumChannels - 1 do
      for Y := 0 to Config.ImageSize - 1 do
        for X := 0 to Config.ImageSize - 1 do
          ImageInput[X, Y, C] :=
            (((C * 256 + Y * 16 + X) * 5) mod 17 - 8) / 8.0;

    WriteLn('Greedy caption (', MaxNewTokens, ' tokens, pico = gibberish):');
    Write('  generated token ids:');
    for Step := 0 to MaxNewTokens - 1 do
    begin
      AssemblePrompt(Generated);
      // Rebuild the decoder at the EXACT current prompt length so the causal
      // mask and positions line up (the KV-cache fast path is a follow-up).
      VisionNet.Free; ProjectorNet.Free; TextNet.Free;
      BuildLlavaFromSafeTensors(CheckpointPath, VisionNet, ProjectorNet,
        TextNet, Config, {pSeqLen=}Length(TokenIds), {pInferenceOnly=}true,
        ConfigPath);
      LlavaRunLogits(VisionNet, ProjectorNet, TextNet, ImageInput, TokenIds,
        Config.ImageTokenIndex, Config.NumPatches, Logits);
      BestTok := ArgmaxLastRow(Logits);
      SetLength(Generated, Length(Generated) + 1);
      Generated[High(Generated)] := BestTok;
      Write(' ', BestTok);
    end;
    WriteLn;
    WriteLn;
    WriteLn('Done. (The plumbing - vision tower -> projector -> spliced ',
      'visual tokens -> causal decode - is what this demo exercises.)');
  finally
    Logits.Free;
    ImageInput.Free;
    VisionNet.Free;
    ProjectorNet.Free;
    TextNet.Free;
  end;
end.
