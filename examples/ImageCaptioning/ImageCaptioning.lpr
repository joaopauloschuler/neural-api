program ImageCaptioning;
// BLIP image captioning on the committed pico fixture (or any
// BlipForConditionalGeneration checkpoint passed as argument 1): a
// generative ENCODER-DECODER vision-language importer. A ViT
// image encoder feeds a BERT-style causal text DECODER through cross-attention
// (BuildBlipForCaptioningFromSafeTensors, neural/neuralpretrained.pas), which
// autoregressively generates a caption with DecodeBlipCaptionGreedy.
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_blip.* pico fixture):
//   examples/ImageCaptioning/ImageCaptioning
//   examples/ImageCaptioning/ImageCaptioning /path/to/blip-image-captioning-base/model.safetensors
//
// The pico fixture's "image" is the deterministic test pattern its generator
// used; a real pipeline would load + BLIP-normalize a photo and decode the
// generated ids with the BLIP (BERT WordPiece) tokenizer - both out of this
// demo's scope. The point is the end-to-end CAPTION GENERATION STRUCTURE:
// encode the image once, then greedily roll out the decoder over its
// cross-attention to the patch features.
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  SysUtils, neuralvolume, neuralnetwork, neuralpretrained;

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_blip.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_blip.safetensors';
end;

var
  VisionNet, TextNet: TNNet;
  Config: TBlipConfig;
  CheckpointPath, ConfigPath, CaptionStr: string;
  PixelValues: TNNetVolume;
  Caption: TNeuralIntegerArray;
  ChanCnt, YCnt, XCnt, i: integer;
begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a BLIP .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_blip.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_blip_config.json'
  else ConfigPath := '';

  WriteLn('Loading ', CheckpointPath, ' ...');
  BuildBlipForCaptioningFromSafeTensors(CheckpointPath, VisionNet, TextNet,
    Config, {DecSeqLen=}20, {pInferenceOnly=}true, ConfigPath);
  WriteLn(BlipConfigToString(Config));
  WriteLn;

  PixelValues := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumChannels);
  try
    // The pico "image": a deterministic dyadic test pattern (a real pipeline
    // would load + BLIP-normalize a photo).
    for ChanCnt := 0 to Config.NumChannels - 1 do
      for YCnt := 0 to Config.ImageSize - 1 do
        for XCnt := 0 to Config.ImageSize - 1 do
          PixelValues[XCnt, YCnt, ChanCnt] :=
            (((ChanCnt * 256 + YCnt * 16 + XCnt) * 5) mod 17 - 8) / 8.0;

    // Encode the image once, then greedily generate the caption ids from BOS.
    Caption := DecodeBlipCaptionGreedy(VisionNet, TextNet, PixelValues,
      Config, {MaxNewTokens=}18);

    WriteLn('Generated caption token ids (BOS=', Config.BosTokenId,
      ' excluded; EOS=', Config.EosTokenId, ' terminates):');
    CaptionStr := '';
    for i := 0 to High(Caption) do
    begin
      if i > 0 then CaptionStr := CaptionStr + ' ';
      CaptionStr := CaptionStr + IntToStr(Caption[i]);
    end;
    WriteLn('  [', CaptionStr, ']');
    WriteLn('  (', Length(Caption), ' tokens; decode them with the BLIP ',
      'WordPiece tokenizer for text.)');
  finally
    PixelValues.Free;
    VisionNet.Free;
    TextNet.Free;
  end;
end.
