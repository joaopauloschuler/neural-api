program OpenVocabDetection;
// OWL-ViT open-vocabulary (zero-shot) object detection on the committed pico
// fixture (or any OWL-ViT checkpoint passed as argument 1). Unlike DETR (a
// FIXED, closed label set), OWL-ViT scores every IMAGE PATCH against arbitrary
// FREE-TEXT query embeddings by cosine similarity - so the "classes" are
// whatever text you encode, decided at inference time.
//
// The demo loads the two nets returned by BuildOwlViTFromSafeTensors
// (neural/neuralpretrained.pas - the repo's first zero-shot detector), runs one
// tiny image through the CLIP ViT image + detection head, embeds a couple of
// free-text query token-id sequences through the CLIP text tower
// (OwlViTQueryEmbedding), and reads off the (patch, query) match scores and
// boxes with DecodeOwlViTDetections. For each query it prints the best-matching
// patch (its sigmoid match score and the predicted cxcywh box on the 0..1
// image).
//
// Run from the repo root (works OFFLINE - the default checkpoint is the
// committed tests/fixtures/tiny_owlvit.* pico fixture; the image and the query
// token ids are read straight from the fixture's io.json so the numbers match
// the parity test):
//   examples/OpenVocabDetection/OpenVocabDetection
//   examples/OpenVocabDetection/OpenVocabDetection /path/to/owlvit/model.safetensors
//
// With a real checkpoint the queries are still synthetic token-id sequences
// (the OWL-ViT BPE tokenizer and the resize/normalize image preprocessing are
// out of this demo's scope): the point is the open-vocabulary SCORING
// STRUCTURE - real prompts just swap in real token ids.
//
// This example is coded by Claude (AI).
{$mode objfpc}{$H+}

uses
  Classes, SysUtils, Math, fpjson, jsonparser, neuralvolume, neuralnetwork,
  neuralpretrained;

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_owlvit.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_owlvit.safetensors';
end;

var
  TextNet, VisionNet: TNNet;
  Config: TOwlViTConfig;
  CheckpointPath, ConfigPath, IoPath: string;
  ImageInput, Tokens, VisOut, QueryEmbeds, OneQuery: TNNetVolume;
  RefRoot: TJSONData;
  RefJson: TStringList;
  CasesArr, InArr, IdArr: TJSONArray;
  CaseObj: TJSONObject;
  Dets: TOwlViTDetectionArray;
  W, H, NumCh, SeqLen, NQ, NP, q, c, x, yy, ch, FlatIdx, d, i: integer;
  BestIdx: integer;
  BestScore: TNeuralFloat;
begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass an OWL-ViT .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ParamCount >= 2 then ConfigPath := ParamStr(2)
  else if ExtractFileName(CheckpointPath) = 'tiny_owlvit.safetensors' then
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_owlvit_config.json'
  else ConfigPath := '';
  IoPath := ExtractFilePath(CheckpointPath) + 'tiny_owlvit_io.json';

  WriteLn('Loading ', CheckpointPath, ' ...');
  BuildOwlViTFromSafeTensors(CheckpointPath, TextNet, VisionNet, Config,
    {TextSeqLen=}0, {pInferenceOnly=}true, ConfigPath);
  WriteLn(OwlViTConfigToString(Config));
  WriteLn;

  W := Config.ImageSize; H := Config.ImageSize; NumCh := Config.NumChannels;
  SeqLen := Config.TextMaxPositions;
  NP := Config.GridH * Config.GridW;

  ImageInput := TNNetVolume.Create(W, H, NumCh);
  Tokens := TNNetVolume.Create(SeqLen, 1, 1);
  VisOut := TNNetVolume.Create;
  QueryEmbeds := TNNetVolume.Create;
  OneQuery := TNNetVolume.Create;
  RefJson := TStringList.Create;
  RefRoot := nil;
  try
    // The image + the query token ids come from the fixture's io.json so the
    // demo runs offline AND reproduces the parity test's numbers. A real
    // pipeline would load+normalize a photo and BPE-tokenize the text prompts.
    if not FileExists(IoPath) then
    begin
      WriteLn('io.json not found beside the checkpoint: ', IoPath);
      WriteLn('(only the committed pico fixture ships one)');
      Halt(1);
    end;
    RefJson.LoadFromFile(IoPath);
    RefRoot := GetJSON(RefJson.Text);
    CasesArr := TJSONArray(TJSONObject(RefRoot).Find('cases'));
    CaseObj := TJSONObject(CasesArr.Items[0]);
    InArr := TJSONArray(CaseObj.Find('input'));
    IdArr := TJSONArray(CaseObj.Find('input_ids'));
    NQ := CaseObj.Get('num_queries', 0);

    // image: flat (y, x, c)
    for yy := 0 to H - 1 do
      for x := 0 to W - 1 do
        for ch := 0 to NumCh - 1 do
        begin
          FlatIdx := (yy * W + x) * NumCh + ch;
          ImageInput[x, yy, ch] := InArr.Items[FlatIdx].AsFloat;
        end;
    VisionNet.Compute(ImageInput);
    VisOut.Copy(VisionNet.GetLastLayer().Output);

    // embed each free-text query through the text tower.
    QueryEmbeds.ReSize(NQ, 1, Config.ProjectionDim);
    for q := 0 to NQ - 1 do
    begin
      for c := 0 to SeqLen - 1 do
        Tokens.FData[c] := IdArr.Items[q * SeqLen + c].AsFloat;
      TextNet.Compute(Tokens);
      OwlViTQueryEmbedding(TextNet, Tokens, OneQuery);
      for d := 0 to Config.ProjectionDim - 1 do
        QueryEmbeds[q, 0, d] := OneQuery.FData[d];
    end;

    // score every (patch, query) pair; threshold 0 keeps them all.
    Dets := DecodeOwlViTDetections(VisOut, QueryEmbeds,
      Config.ProjectionDim, Config.GridH, Config.GridW, {Threshold=}0.0);

    WriteLn(Format('Image: %dx%d, %d patches (%dx%d grid), %d free-text queries',
      [W, H, NP, Config.GridH, Config.GridW, NQ]));
    WriteLn('Best-matching patch per query (sigmoid score + cxcywh box):');
    for q := 0 to NQ - 1 do
    begin
      BestIdx := -1;
      BestScore := -1;
      for i := 0 to High(Dets) do
        if (Dets[i].QueryIndex = q) and (Dets[i].Score > BestScore) then
        begin
          BestScore := Dets[i].Score;
          BestIdx := i;
        end;
      if BestIdx >= 0 then
        WriteLn(Format('  query #%d -> patch %d  score=%6.2f%%  ' +
          'box=(cx %.3f, cy %.3f, w %.3f, h %.3f)',
          [q, Dets[BestIdx].PatchIndex, 100.0 * Dets[BestIdx].Score,
           Dets[BestIdx].Cx, Dets[BestIdx].Cy, Dets[BestIdx].W,
           Dets[BestIdx].H]));
    end;
  finally
    RefRoot.Free;
    RefJson.Free;
    OneQuery.Free;
    QueryEmbeds.Free;
    VisOut.Free;
    Tokens.Free;
    ImageInput.Free;
    VisionNet.Free;
    TextNet.Free;
  end;
end.
