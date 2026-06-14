program SigLIPZeroShot;
// SigLIP zero-shot classification structure on the committed pico fixture
// (or any SigLIP / siglip2 checkpoint passed as argument 1): embed N
// class-prompt token sequences and one image with the TWO nets returned by
// BuildSigLIPFromSafeTensors (neural/neuralpretrained.pas), score them in
// the shared space with SigLIP's SIGMOID head - exp(logit_scale)*cosine +
// logit_bias - and print BOTH the per-class sigmoid probabilities (the
// native SigLIP output: each class is an INDEPENDENT yes/no, unlike CLIP's
// softmax-over-classes) AND a softmax ranking for convenience.
//
// SigLIP is architecturally DISTINCT from CLIP (handled on its own path):
//   - the image embedding is the Multihead Attention Pooling head's output
//     (row 0 of the vision net), NOT a CLS token;
//   - the text embedding is the LAST token's hidden state through a biased
//     head (NOT CLIP's eos-argmax pooling);
//   - the score adds a learnable logit_bias to the scaled cosine and uses a
//     per-pair SIGMOID (the sigmoid pairwise loss), so the demo reports a
//     standalone match probability per class.
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_siglip.* pico fixture):
//   examples/SigLIPZeroShot/SigLIPZeroShot
//   examples/SigLIPZeroShot/SigLIPZeroShot /path/to/siglip-base-patch16-224/model.safetensors
//
// With a real checkpoint the "prompts" below are still synthetic token-id
// sequences (the SigLIP SentencePiece tokenizer and the resize/normalize
// image preprocessing are out of this demo's scope): the point is the
// zero-shot SCORING STRUCTURE - real text just swaps in real token ids
// (SigLIP pads to max_length, so the LAST token is meaningful).
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, Math, neuralvolume, neuralnetwork, neuralpretrained;

const
  // The pico fixture's classes: each "prompt" is a token-id sequence in
  // the tiny vocab (33 ids). SigLIP pools the LAST position, so the last
  // id is the one that matters most for the pooled embedding.
  NumClasses = 3;
  PicoPrompts: array[0..NumClasses - 1, 0..7] of integer = (
    (0,  7, 23, 11, 14,  5,  9,  3),   // the fixture's text sequence 0
    (0, 31,  8, 30, 17, 26,  2, 19),   // the fixture's text sequence 1
    (0,  4, 19,  6, 28, 14, 21, 12));  // a third synthetic class prompt
  PicoLabels: array[0..NumClasses - 1] of string = (
    'class prompt #0',
    'class prompt #1',
    'class prompt #2 (synthetic)');

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_siglip.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_siglip.safetensors';
end;

var
  TextNet, VisionNet: TNNet;
  Config: TSigLIPConfig;
  CheckpointPath, ConfigPath: string;
  TextInput, ImageInput: TNNetVolume;
  ImageEmb: TNNetVolume;
  TextEmbs: array of TNNetVolume;
  Logits, Probs, SigmoidProbs: array of TNeuralFloat;
  SeqLen, ClassCnt, PosCnt, ChanCnt, YCnt, XCnt, TokenId: integer;
  MaxLogit, SumExp: TNeuralFloat;
begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a SigLIP .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_siglip.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_siglip_config.json'
  else ConfigPath := '';

  WriteLn('Loading ', CheckpointPath, ' ...');
  BuildSigLIPFromSafeTensors(CheckpointPath, TextNet, VisionNet, Config,
    {TextSeqLen=}8, {pInferenceOnly=}true, ConfigPath);
  WriteLn(SigLIPConfigToString(Config));
  WriteLn;

  SeqLen := TextNet.Layers[0].Output.SizeX;
  TextInput := TNNetVolume.Create(SeqLen, 1, 1);
  ImageInput := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumChannels);
  ImageEmb := TNNetVolume.Create;
  SetLength(TextEmbs, NumClasses);
  SetLength(Logits, NumClasses);
  SetLength(Probs, NumClasses);
  SetLength(SigmoidProbs, NumClasses);
  try
    // ---- the image: the fixture generator's deterministic dyadic test
    // pattern (a real pipeline would load + SigLIP-normalize a photo) ----
    for ChanCnt := 0 to Config.NumChannels - 1 do
      for YCnt := 0 to Config.ImageSize - 1 do
        for XCnt := 0 to Config.ImageSize - 1 do
          ImageInput[XCnt, YCnt, ChanCnt] :=
            (((ChanCnt * 256 + YCnt * 16 + XCnt) * 5) mod 17 - 8) / 8.0;
    VisionNet.Compute(ImageInput);
    // Row 0 = the Multihead Attention Pooling output: HF's image_embeds,
    // L2-normalized here for the cosine score.
    ClipExtractEmbedding(VisionNet.GetLastLayer().Output, 0, ImageEmb);

    // ---- the N class prompts through the BIDIRECTIONAL text tower ----
    for ClassCnt := 0 to NumClasses - 1 do
    begin
      for PosCnt := 0 to SeqLen - 1 do
      begin
        TokenId := PicoPrompts[ClassCnt, PosCnt mod 8];
        TokenId := TokenId mod Config.TextVocabSize;
        TextInput.FData[PosCnt] := TokenId;
      end;
      TextNet.Compute(TextInput);
      TextEmbs[ClassCnt] := TNNetVolume.Create;
      // SigLIP pools the LAST token (row SeqLen-1).
      ClipExtractEmbedding(TextNet.GetLastLayer().Output, SeqLen - 1,
        TextEmbs[ClassCnt]);
      // HF's logits_per_image entry: exp(logit_scale)*cosine + logit_bias.
      Logits[ClassCnt] := SigLIPLogit(ImageEmb, TextEmbs[ClassCnt],
        Config.LogitScale, Config.LogitBias);
      // SigLIP's native output: an INDEPENDENT sigmoid match probability.
      SigmoidProbs[ClassCnt] := 1.0 / (1.0 + Exp(-Logits[ClassCnt]));
    end;

    // ---- softmax over the class logits (a convenience ranking) ----
    MaxLogit := Logits[0];
    for ClassCnt := 1 to NumClasses - 1 do
      MaxLogit := Max(MaxLogit, Logits[ClassCnt]);
    SumExp := 0;
    for ClassCnt := 0 to NumClasses - 1 do
      SumExp := SumExp + Exp(Logits[ClassCnt] - MaxLogit);
    WriteLn('Zero-shot scoring for the test image:');
    WriteLn('  (SigLIP uses a per-pair SIGMOID, so the sigmoid column is the');
    WriteLn('   native standalone match probability; softmax is a ranking.)');
    for ClassCnt := 0 to NumClasses - 1 do
    begin
      Probs[ClassCnt] := Exp(Logits[ClassCnt] - MaxLogit) / SumExp;
      WriteLn(Format('  %-30s logit = %8.4f  sigmoid = %6.2f%%  softmax = %6.2f%%',
        [PicoLabels[ClassCnt], Logits[ClassCnt],
         100.0 * SigmoidProbs[ClassCnt], 100.0 * Probs[ClassCnt]]));
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
