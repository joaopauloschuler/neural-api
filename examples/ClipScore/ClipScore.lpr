program ClipScoreDemo;
// CLIPScore (Hessel et al. 2021, "CLIPScore: A Reference-free Evaluation
// Metric for Image Captioning") on the committed pico CLIP fixture (or any
// CLIP checkpoint passed as argument 1). CLIPScore is the standard
// REFERENCE-FREE text<->image alignment metric for text-to-image / caption
// quality - how well a generated image matches its prompt. It complements
// the IMAGE-only FID / IS / KID (neuralimagemetrics.pas) with a SEMANTIC
// image<->text score.
//
//   CLIPScore = w * max(0, cos(image_embed, text_embed)),  w = 2.5
//
// using the TWO nets BuildClipFromSafeTensors returns (neuralpretrained.pas).
// This demo scores ONE image against several prompts and shows the
// MISMATCHED prompts score LOWER than the matching one. It also prints
// RefCLIPScore (the captioning variant: the harmonic mean of CLIPScore with
// the candidate<->reference-caption cosine).
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_clip.* pico fixture):
//   examples/ClipScore/ClipScore
//   examples/ClipScore/ClipScore /path/to/clip-vit-base-patch32/model.safetensors
//
// With a real checkpoint the "prompts" below are still synthetic token-id
// sequences (the CLIP byte-level BPE tokenizer and the resize/normalize
// image preprocessing are out of this demo's scope): the point is the
// SCORING STRUCTURE - real text just swaps in real token ids.
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, neuralvolume, neuralnetwork, neuralpretrained;

const
  // The pico fixture's "prompts": each is a token-id sequence in the tiny
  // vocab (33 ids, eot = 32 the HIGHEST id - the legacy eos_token_id = 2
  // ARGMAX pooling of every published OpenAI CLIP). Prompt 0 is the
  // fixture's text sequence 1 (the BETTER match: positive oracle cosine);
  // prompts 1 and 2 are deliberately MISMATCHED (negative / lower cosine).
  NumPrompts = 3;
  Prompts: array[0..NumPrompts - 1, 0..7] of integer = (
    (0, 31,  8, 30, 17, 26,  2, 32),   // fixture text seq 1 (matching)
    (0,  7, 23, 11, 32,  5,  9,  3),   // fixture text seq 0 (mismatched)
    (0,  4, 19,  6, 28, 14, 21, 32));  // a synthetic unrelated prompt
  Labels: array[0..NumPrompts - 1] of string = (
    'matching prompt   ',
    'mismatched prompt ',
    'unrelated prompt  ');

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_clip.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_clip.safetensors';
end;

var
  TextNet, VisionNet: TNNet;
  Config: TClipConfig;
  CheckpointPath, ConfigPath: string;
  TextInput, ImageInput, ImageEmb: TNNetVolume;
  TextEmbs: array of TNNetVolume;
  Scores: array of TNeuralFloat;
  SeqLen, PromptCnt, ChanCnt, YCnt, XCnt, EosPos: integer;
  RefScore: TNeuralFloat;

procedure FillPrompt(P: integer);
var
  K, Tok: integer;
begin
  for K := 0 to SeqLen - 1 do
  begin
    Tok := Prompts[P, K mod 8];
    // Clamp ids into a real checkpoint's vocab; map the pico eot to the
    // checkpoint's own top id so ARGMAX pooling stays valid.
    if Tok = 32 then Tok := Config.TextVocabSize - 1
    else Tok := Tok mod Config.TextVocabSize;
    TextInput.FData[K] := Tok;
  end;
end;

begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a CLIP .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_clip.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_clip_config.json'
  else ConfigPath := '';

  WriteLn('Loading ', CheckpointPath, ' ...');
  BuildClipFromSafeTensors(CheckpointPath, TextNet, VisionNet, Config,
    {TextSeqLen=}8, {pInferenceOnly=}true, ConfigPath);
  WriteLn(ClipConfigToString(Config));
  WriteLn;

  SeqLen := TextNet.Layers[0].Output.SizeX;
  TextInput := TNNetVolume.Create(SeqLen, 1, 1);
  ImageInput := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumChannels);
  ImageEmb := TNNetVolume.Create;
  SetLength(TextEmbs, NumPrompts);
  SetLength(Scores, NumPrompts);
  try
    // ---- the image: the fixture generator's deterministic dyadic test
    // pattern (a real pipeline would load + CLIP-normalize a photo) ----
    for ChanCnt := 0 to Config.NumChannels - 1 do
      for YCnt := 0 to Config.ImageSize - 1 do
        for XCnt := 0 to Config.ImageSize - 1 do
          ImageInput[XCnt, YCnt, ChanCnt] :=
            (((ChanCnt * 256 + YCnt * 16 + XCnt) * 5) mod 17 - 8) / 8.0;

    WriteLn('CLIPScore = 2.5 * max(0, cosine(image, prompt))');
    WriteLn('  (higher = the prompt better describes the image)');
    WriteLn;
    for PromptCnt := 0 to NumPrompts - 1 do
    begin
      FillPrompt(PromptCnt);
      // End-to-end CLIPScore: runs both towers, pools + L2-normalizes,
      // returns 2.5 * max(0, cosine).
      Scores[PromptCnt] := ClipScore(TextNet, VisionNet, ImageInput,
        TextInput, Config.EosTokenId);
      // Stash the unit-L2 embeddings for the RefCLIPScore demo below.
      ClipExtractEmbedding(VisionNet.GetLastLayer().Output, 0, ImageEmb);
      EosPos := ClipTextEosPosition(TextInput, Config.EosTokenId);
      TextEmbs[PromptCnt] := TNNetVolume.Create;
      ClipExtractEmbedding(TextNet.GetLastLayer().Output, EosPos,
        TextEmbs[PromptCnt]);
      WriteLn(Format('  %s  CLIPScore = %7.4f',
        [Labels[PromptCnt], Scores[PromptCnt]]));
    end;
    WriteLn;
    if (Scores[1] <= Scores[0]) and (Scores[1] <= Scores[2]) then
      WriteLn('OK: the MISMATCHED prompt scores LOWEST (its negative ',
        'cosine clips CLIPScore to 0); the matching prompt scores higher. ',
        'On a real trained CLIP the matching caption also ranks first - ',
        'this pico checkpoint is randomly initialized.')
    else
      WriteLn('NOTE: ranking differs (this pico checkpoint is randomly ',
        'initialized; on a real CLIP the matching caption ranks first).');

    // ---- RefCLIPScore (captioning variant): harmonic mean of the
    // candidate's CLIPScore with the candidate<->reference cosine. Here the
    // "candidate" is prompt 0 and the "reference caption" is prompt 2. ----
    WriteLn;
    RefScore := RefClipScoreFromEmbeddings(ImageEmb, TextEmbs[0],
      TextEmbs[2]);
    // ImageEmb currently holds the last image embedding (the image is fixed
    // across prompts, so it is the same for every prompt).
    WriteLn(Format('RefCLIPScore(candidate=prompt0, reference=prompt2) ' +
      '= %7.4f', [RefScore]));
    WriteLn('  (harmonic mean of the image CLIPScore and the ',
      'candidate<->reference cosine - the captioning metric).');
  finally
    for PromptCnt := 0 to High(TextEmbs) do TextEmbs[PromptCnt].Free;
    ImageEmb.Free;
    ImageInput.Free;
    TextInput.Free;
    VisionNet.Free;
    TextNet.Free;
  end;
end.
