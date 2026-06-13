program ClipZeroShot;
// CLIP zero-shot classification structure on the committed pico fixture
// (or any CLIP checkpoint passed as argument 1): embed N class-prompt
// token sequences and one image with the TWO nets returned by
// BuildClipFromSafeTensors (neural/neuralpretrained.pas - the first
// vision-language importer), score them in the shared space with
// exp(logit_scale) * cosine, and print the softmaxed class probabilities.
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_clip.* pico fixture):
//   examples/ClipZeroShot/ClipZeroShot
//   examples/ClipZeroShot/ClipZeroShot /path/to/clip-vit-base-patch32/model.safetensors
//
// With a real checkpoint the "prompts" below are still synthetic token-id
// sequences (the CLIP byte-level BPE tokenizer and the resize/normalize
// image preprocessing are out of this demo's scope): the point is the
// zero-shot SCORING STRUCTURE - real text just swaps in real token ids.
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, neuralvolume, neuralnetwork, neuralpretrained;

const
  // The pico fixture's classes: each "prompt" is a token-id sequence in
  // the tiny vocab (33 ids, eot = 32 the HIGHEST id - the legacy
  // eos_token_id = 2 ARGMAX pooling of every published OpenAI CLIP).
  // Sequence 0 carries its eot MID-sequence on purpose (positions after
  // it are pad-like and, with the causal text tower, provably ignored).
  NumClasses = 3;
  PicoPrompts: array[0..NumClasses - 1, 0..7] of integer = (
    (0,  7, 23, 11, 32,  5,  9,  3),   // the fixture's text sequence 0
    (0, 31,  8, 30, 17, 26,  2, 32),   // the fixture's text sequence 1
    (0,  4, 19,  6, 28, 14, 21, 32));  // a third synthetic class prompt
  PicoLabels: array[0..NumClasses - 1] of string = (
    'class prompt #0 (eot mid-sequence)',
    'class prompt #1 (eot last)',
    'class prompt #2 (synthetic)');

function DefaultCheckpoint(): string;
begin
  // Repo root or examples/ClipZeroShot as the working directory.
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
  TextInput, ImageInput: TNNetVolume;
  ImageEmb: TNNetVolume;
  TextEmbs: array of TNNetVolume;
  Logits, Probs: array of TNeuralFloat;
  SeqLen, ClassCnt, PosCnt, ChanCnt, YCnt, XCnt, EosPos, TokenId: integer;
  MaxLogit, SumExp: TNeuralFloat;
begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a CLIP .safetensors path or run from the repo root.');
    Halt(1);
  end;
  // Optional argument 2 = config.json path. The pico fixture names its
  // config tiny_clip_config.json (the tests/fixtures convention); real
  // checkpoints keep the default config.json next to the weights ('').
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_clip.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_clip_config.json'
  else ConfigPath := '';

  WriteLn('Loading ', CheckpointPath, ' ...');
  // SeqLen 8 covers the pico prompts; real checkpoints allow up to
  // text max_position_embeddings (77). pInferenceOnly keeps the full
  // clip-vit-base-patch32 import comfortably in memory.
  BuildClipFromSafeTensors(CheckpointPath, TextNet, VisionNet, Config,
    {TextSeqLen=}8, {pInferenceOnly=}true, ConfigPath);
  WriteLn(ClipConfigToString(Config));
  WriteLn;

  SeqLen := TextNet.Layers[0].Output.SizeX;
  TextInput := TNNetVolume.Create(SeqLen, 1, 1);
  ImageInput := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumChannels);
  ImageEmb := TNNetVolume.Create;
  SetLength(TextEmbs, NumClasses);
  SetLength(Logits, NumClasses);
  SetLength(Probs, NumClasses);
  try
    // ---- the image: the fixture generator's deterministic dyadic test
    // pattern (a real pipeline would load + CLIP-normalize a photo) ----
    for ChanCnt := 0 to Config.NumChannels - 1 do
      for YCnt := 0 to Config.ImageSize - 1 do
        for XCnt := 0 to Config.ImageSize - 1 do
          ImageInput[XCnt, YCnt, ChanCnt] :=
            (((ChanCnt * 256 + YCnt * 16 + XCnt) * 5) mod 17 - 8) / 8.0;
    VisionNet.Compute(ImageInput);
    // Row 0 = the class token: HF's image_embeds, L2-normalized here.
    ClipExtractEmbedding(VisionNet.GetLastLayer().Output, 0, ImageEmb);

    // ---- the N class prompts through the text tower ----
    for ClassCnt := 0 to NumClasses - 1 do
    begin
      for PosCnt := 0 to SeqLen - 1 do
      begin
        TokenId := PicoPrompts[ClassCnt, PosCnt mod 8];
        // Clamp ids into a real checkpoint's vocab; map the pico eot to
        // the checkpoint's own top id so ARGMAX pooling stays valid.
        if TokenId = 32 then TokenId := Config.TextVocabSize - 1
        else TokenId := TokenId mod Config.TextVocabSize;
        TextInput.FData[PosCnt] := TokenId;
      end;
      TextNet.Compute(TextInput);
      EosPos := ClipTextEosPosition(TextInput, Config.EosTokenId);
      TextEmbs[ClassCnt] := TNNetVolume.Create;
      ClipExtractEmbedding(TextNet.GetLastLayer().Output, EosPos,
        TextEmbs[ClassCnt]);
      // HF's logits_per_image: exp(logit_scale) * cosine similarity.
      Logits[ClassCnt] := Exp(Config.LogitScale) *
        ClipSimilarity(ImageEmb, TextEmbs[ClassCnt]);
    end;

    // ---- softmax over the class logits = zero-shot probabilities ----
    MaxLogit := Logits[0];
    for ClassCnt := 1 to NumClasses - 1 do
      MaxLogit := Max(MaxLogit, Logits[ClassCnt]);
    SumExp := 0;
    for ClassCnt := 0 to NumClasses - 1 do
      SumExp := SumExp + Exp(Logits[ClassCnt] - MaxLogit);
    WriteLn('Zero-shot class probabilities for the test image:');
    for ClassCnt := 0 to NumClasses - 1 do
    begin
      Probs[ClassCnt] := Exp(Logits[ClassCnt] - MaxLogit) / SumExp;
      WriteLn(Format('  %-36s cosine*scale = %8.4f   p = %6.2f%%',
        [PicoLabels[ClassCnt], Logits[ClassCnt],
         100.0 * Probs[ClassCnt]]));
    end;
  finally
    for ClassCnt := 0 to High(TextEmbs) do TextEmbs[ClassCnt].Free;
    ImageEmb.Free;
    ImageInput.Free;
    TextInput.Free;
    VisionNet.Free;
    TextNet.Free;
  end;
end.
