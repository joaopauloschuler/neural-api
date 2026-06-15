program OCRTranscribe;
(*
  TrOCR optical-character-recognition on the committed pico fixture (or any
  TrOCR VisionEncoderDecoder checkpoint passed as argument 1): the FIRST OCR /
  image-to-text vertical in this repo - a cropped text-line image -> a
  transcribed string. A DeiT/ViT image encoder feeds a BART-style causal text
  DECODER through cross-attention (BuildTrOCRFromSafeTensors,
  neural/neuralpretrained.pas), which autoregressively transcribes the line
  with DecodeTrOCRGreedy.

  Run from the repo root (works OFFLINE - the default checkpoint is the
  committed tests/fixtures/tiny_trocr.* pico fixture):
    examples/OCRTranscribe/OCRTranscribe
    examples/OCRTranscribe/OCRTranscribe /path/to/trocr-small-printed/model.safetensors

  The pico fixture's "image" is the deterministic test pattern its generator
  used; a real pipeline would load + TrOCR-normalize a text-line crop and
  decode the generated ids with the TrOCR (GPT-2 byte-level BPE) tokenizer -
  both out of this demo's scope. The point is the end-to-end TRANSCRIPTION
  STRUCTURE: encode the image once, then greedily roll out the decoder over its
  cross-attention to ALL of the patch features (cls + distillation + patches).

  Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses
  SysUtils, neuralvolume, neuralnetwork, neuralpretrained;

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_trocr.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_trocr.safetensors';
end;

var
  EncoderNet, DecoderNet: TNNet;
  Config: TTrOCRConfig;
  CheckpointPath, ConfigPath, TextStr: string;
  PixelValues: TNNetVolume;
  Transcription: TNeuralIntegerArray;
  ChanCnt, YCnt, XCnt, i: integer;
begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a TrOCR .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_trocr.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_trocr_config.json'
  else ConfigPath := '';

  WriteLn('Loading ', CheckpointPath, ' ...');
  BuildTrOCRFromSafeTensors(CheckpointPath, EncoderNet, DecoderNet,
    Config, {DecSeqLen=}20, {pInferenceOnly=}true, ConfigPath);
  WriteLn(TrOCRConfigToString(Config));
  WriteLn;

  PixelValues := TNNetVolume.Create(Config.ImageSize, Config.ImageSize,
    Config.NumChannels);
  try
    // The pico "image": a deterministic dyadic test pattern (a real pipeline
    // would load + TrOCR-normalize a text-line crop).
    for ChanCnt := 0 to Config.NumChannels - 1 do
      for YCnt := 0 to Config.ImageSize - 1 do
        for XCnt := 0 to Config.ImageSize - 1 do
          PixelValues[XCnt, YCnt, ChanCnt] :=
            (((ChanCnt * 256 + YCnt * 16 + XCnt) * 5) mod 17 - 8) / 8.0;

    // Encode the image once, then greedily transcribe from the decoder start
    // token over the cross-attention to the patch features.
    Transcription := DecodeTrOCRGreedy(EncoderNet, DecoderNet, PixelValues,
      Config, {MaxNewTokens=}18);

    WriteLn('Transcribed token ids (start=', Config.DecoderStartTokenId,
      ' excluded; EOS=', Config.EosTokenId, ' terminates):');
    TextStr := '';
    for i := 0 to High(Transcription) do
    begin
      if i > 0 then TextStr := TextStr + ' ';
      TextStr := TextStr + IntToStr(Transcription[i]);
    end;
    WriteLn('  [', TextStr, ']');
    WriteLn('  (', Length(Transcription), ' tokens; decode them with the ',
      'TrOCR GPT-2 byte-level BPE tokenizer for text.)');
  finally
    PixelValues.Free;
    EncoderNet.Free;
    DecoderNet.Free;
  end;
end.
