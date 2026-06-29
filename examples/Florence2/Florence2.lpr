program Florence2;
(*
  Florence-2 unified-vision demo on the committed pico fixture: ONE task-
  prompted seq2seq head that does captioning AND box detection, the repo's
  first "spatial-output-as-text" VLM (BuildFlorence2FromSafeTensors,
  neural/neuralpretrained.pas). The input is an image (here: the DaViT vision
  tower's feature map, supplied precomputed by the fixture - the tower itself
  is the deferred gap) + a short TASK TOKEN stream (<CAPTION>, <OD>, ...). The
  multimodal projector turns the feature map into visual tokens, those are
  prepended to the embedded task-prompt text, the BART encoder runs over the
  whole [visual; text] sequence, and a BART decoder cross-attends to it and
  emits the answer token stream.

  The genuinely new idea: boxes/polygons are emitted as QUANTIZED LOCATION
  tokens <loc_0..loc_999> in the vocabulary (spatial outputs as text). This
  demo shows the location-token (de)quantization round-trip on a sample box.

  Run from the repo root (works OFFLINE - the default checkpoint is the
  committed tests/fixtures/tiny_florence2.* pico fixture):
    examples/Florence2/Florence2
    examples/Florence2/Florence2 /path/to/Florence-2-base/model.safetensors

  The pico checkpoint is randomly initialized, so the generated caption ids are
  gibberish: the demo exercises the image+task -> token-stream PLUMBING and the
  box<->loc-token coordinate mapping. A real run needs a real Florence-2
  download + its tokenizer + the DaViT vision tower (deferred) feeding real
  feature maps. Pure CPU, <1 s on the fixture.

  Coded by Claude (AI).
*)
{$mode objfpc}{$H+}

uses
  SysUtils, Classes, fpjson, jsonparser,
  neuralvolume, neuralnetwork, neuralpretrained;

function DefaultCheckpoint(): string;
begin
  Result := 'tests' + DirectorySeparator + 'fixtures' +
    DirectorySeparator + 'tiny_florence2.safetensors';
  if not FileExists(Result) then
    Result := '..' + DirectorySeparator + '..' + DirectorySeparator +
      'tests' + DirectorySeparator + 'fixtures' + DirectorySeparator +
      'tiny_florence2.safetensors';
end;

var
  EncoderNet, DecoderNet: TNNet;
  Projector: TFlorence2Projector;
  Config: TFlorence2Config;
  CheckpointPath, ConfigPath, IOPath, TextStr: string;
  Root: TJSONData;
  JsonText: TStringList;
  FmapArr, TaskArr, BoxArr: TJSONArray;
  FeatureMap, TaskTokens, DecToks, Logits: TNNetVolume;
  C, H, W, NumTask, EncSeqLen, ch, yy, xx, i, NextId, LocId: integer;
  Coord, BestVal: double;
begin
  if ParamCount >= 1 then CheckpointPath := ParamStr(1)
  else CheckpointPath := DefaultCheckpoint();
  if not FileExists(CheckpointPath) then
  begin
    WriteLn('Checkpoint not found: ', CheckpointPath);
    WriteLn('Pass a Florence-2 .safetensors path or run from the repo root.');
    Halt(1);
  end;
  if ExtractFileName(CheckpointPath) = 'tiny_florence2.safetensors' then
  begin
    ConfigPath := ExtractFilePath(CheckpointPath) + 'tiny_florence2_config.json';
    IOPath := ExtractFilePath(CheckpointPath) + 'tiny_florence2_io.json';
  end
  else
  begin
    ConfigPath := '';
    IOPath := '';
  end;

  // The pico fixture supplies the DaViT feature map + a sample task prompt in
  // its io.json (the tower is the deferred gap - we take its output directly).
  if (IOPath = '') or (not FileExists(IOPath)) then
  begin
    WriteLn('This demo needs the pico io.json (the precomputed DaViT feature ',
      'map). Run it on the committed fixture.');
    Halt(1);
  end;

  JsonText := TStringList.Create;
  Root := nil;
  try
    JsonText.LoadFromFile(IOPath);
    Root := GetJSON(JsonText.Text);
    C := TJSONObject(Root).Get('feature_c', 0);
    H := TJSONObject(Root).Get('feature_h', 0);
    W := TJSONObject(Root).Get('feature_w', 0);
    FmapArr := TJSONArray(TJSONObject(Root).Find('feature_map_chw'));
    TaskArr := TJSONArray(TJSONObject(Root).Find('task_ids'));
    NumTask := TaskArr.Count;
    // visual tokens = H*W + 1 (spatial-mean token), + the task tokens.
    EncSeqLen := H * W + 1 + NumTask;

    WriteLn('Loading ', CheckpointPath, ' ...');
    BuildFlorence2FromSafeTensors(CheckpointPath, EncoderNet, DecoderNet,
      Projector, Config, EncSeqLen, {DecSeqLen=}1, {pTrainable=}false,
      ConfigPath);
    WriteLn(Florence2ConfigToString(Config));
    WriteLn;

    FeatureMap := TNNetVolume.Create(W, H, C);
    TaskTokens := TNNetVolume.Create(NumTask, 1, 1);
    DecToks := TNNetVolume.Create(1, 1, 1);
    Logits := TNNetVolume.Create;
    try
      for ch := 0 to C - 1 do
        for yy := 0 to H - 1 do
          for xx := 0 to W - 1 do
            FeatureMap[xx, yy, ch] :=
              TJSONArray(TJSONArray(FmapArr.Items[ch]).Items[yy]).Items[xx].AsFloat;
      TextStr := '';
      for i := 0 to NumTask - 1 do
      begin
        TaskTokens.FData[i] := TaskArr.Items[i].AsInteger;
        if i > 0 then TextStr := TextStr + ' ';
        TextStr := TextStr + IntToStr(TaskArr.Items[i].AsInteger);
      end;
      WriteLn('Task-prompt token ids: [', TextStr, ']  (e.g. <OD> detection)');

      // One greedy decoder step from the start token: the next predicted id.
      DecToks.FData[0] := Config.Bart.DecoderStartTokenId;
      RunFlorence2Logits(EncoderNet, DecoderNet, Config, Projector,
        FeatureMap, TaskTokens, DecToks, Logits);
      NextId := 0;
      BestVal := Logits[0, 0, 0];
      for i := 1 to Config.Bart.VocabSize - 1 do
        if Logits[0, 0, i] > BestVal then
        begin
          BestVal := Logits[0, 0, i];
          NextId := i;
        end;
      WriteLn('First generated token id (argmax): ', NextId,
        '  (decode with the Florence-2 BART tokenizer for text).');
      WriteLn;

      // Detection demo: a box's coordinates ARE text via <loc_> tokens.
      WriteLn('Location-token (de)quantization (boxes/polygons as text):');
      BoxArr := TJSONArray(TJSONObject(Root).Find('ref_box'));
      TextStr := '';
      for i := 0 to BoxArr.Count - 1 do
      begin
        Coord := BoxArr.Items[i].AsFloat;
        LocId := Florence2QuantizeCoord(Coord, Config.LocNumBins,
          Config.LocBase);
        if i > 0 then TextStr := TextStr + ' ';
        TextStr := TextStr + '<loc_' + IntToStr(LocId - Config.LocBase) + '>';
      end;
      WriteLn('  box ', '[x0 y0 x1 y1] -> ', TextStr);
      Write('  decoded back -> [');
      for i := 0 to BoxArr.Count - 1 do
      begin
        LocId := Florence2QuantizeCoord(BoxArr.Items[i].AsFloat,
          Config.LocNumBins, Config.LocBase);
        Coord := Florence2DequantizeCoord(LocId, Config.LocNumBins,
          Config.LocBase);
        if i > 0 then Write(' ');
        Write(FormatFloat('0.000', Coord));
      end;
      WriteLn(']');
    finally
      Logits.Free;
      DecToks.Free;
      TaskTokens.Free;
      FeatureMap.Free;
      FreeFlorence2Projector(Projector);
      EncoderNet.Free;
      DecoderNet.Free;
    end;
  finally
    Root.Free;
    JsonText.Free;
  end;
end.
