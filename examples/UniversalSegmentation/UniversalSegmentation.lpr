(*
  UniversalSegmentation - semantic, instance or panoptic inference with an
  imported Mask2Former model.

  Mask2Former (Cheng et al. 2022, "Masked-attention Mask Transformer for
  Universal Image Segmentation", https://arxiv.org/abs/2112.01527) does MASK
  CLASSIFICATION set-prediction: a fixed set of learned object queries, each
  predicting ONE binary mask + a class distribution, unifying semantic /
  instance / panoptic in a SINGLE head. It is distinct from per-PIXEL argmax
  (SegFormer) and from region-proposal RoIAlign (Mask R-CNN): there are NO
  proposals and NO per-pixel classifier. The conceptual core is MASKED
  ATTENTION - each decoder layer's cross-attention is restricted to the
  FOREGROUND of the mask predicted by the previous layer.

  This example loads the committed PICO Mask2Former parity fixture
  (the tiny_mask2former.safetensors masked-attention decoder + heads) and
  runs it on the fixture's precomputed pixel-decoder feature maps
  (mask_features + the 3 multi-scale memory levels), which v1 takes as inputs
  (mirroring how Mask R-CNN v1 took FPN feature maps directly). It folds the
  per-query class logits and mask logits into a per-pixel map selected by
  --task semantic|instance|panoptic (DecodeMask2FormerSemantic /
  DecodeMask2FormerInstance / DecodeMask2FormerPanoptic, ports of the
  transformers Mask2FormerImageProcessor post-processors), prints it as a
  colored ASCII palette (plus the segment list for instance/panoptic), and
  writes the map as a colored PPM image. A real run would
  load facebook/mask2former-swin-tiny-*-semantic the same way and segment a
  photograph; the math is identical. Inference-only, CPU, finishes instantly.

  Coded by Joao Paulo Schwarz Schuler with Claude (AI).
  https://github.com/joaopauloschuler/neural-api

  Coded by Claude (AI).
*)
program UniversalSegmentation;

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, fpjson, jsonparser,
  neuralnetwork, neuralvolume, neuralpretrained;

const
  // Palette glyphs / RGB, one per class (the pico fixture has 4 classes).
  csPalette: array[0..7] of char = ('.', ':', '+', '#', '@', '*', 'o', '=');
  csRGB: array[0..7, 0..2] of byte = (
    (40, 40, 60), (220, 60, 60), (60, 200, 90), (70, 120, 230),
    (230, 200, 60), (200, 70, 210), (60, 210, 210), (200, 200, 200));

type
  TSegmentationTask = (stSemantic, stInstance, stPanoptic);

function FixturePath(const FileName: string): string;
var
  Base: string;
begin
  Base := '../../tests/fixtures/' + FileName;
  if FileExists(Base) then Exit(Base);
  Base := 'tests/fixtures/' + FileName;
  if FileExists(Base) then Exit(Base);
  Result := FileName;
end;

// Palette slot of a map value: semantic maps index by label; segment maps put
// everything below FirstId (background) on slot 0 and cycle ids over 1..7.
function PaletteIndex(Value, FirstId: integer; IsSegmentMap: boolean): integer;
begin
  if not IsSegmentMap then Exit(Value and 7);
  if Value < FirstId then Exit(0);
  Result := 1 + (Value - FirstId) mod 7;
end;

// Prints Map (W x H, row-major) as palette glyphs.
procedure PrintMap(const Map: TMask2FormerLabelMap; W, H, FirstId: integer;
  IsSegmentMap: boolean);
var
  x, y: integer;
  Line: string;
begin
  for y := 0 to H - 1 do
  begin
    Line := '  ';
    for x := 0 to W - 1 do
      Line := Line + csPalette[PaletteIndex(Map[y * W + x], FirstId,
        IsSegmentMap)] + ' ';
    WriteLn(Line);
  end;
end;

// Writes Map (W x H, row-major) as a colored P3 PPM.
procedure WriteMapPPM(const Map: TMask2FormerLabelMap; W, H, FirstId: integer;
  IsSegmentMap: boolean; const PPMFileName: string);
var
  x, y, Slot: integer;
  PPM: TextFile;
begin
  AssignFile(PPM, PPMFileName);
  Rewrite(PPM);
  WriteLn(PPM, 'P3');
  WriteLn(PPM, W, ' ', H);
  WriteLn(PPM, 255);
  for y := 0 to H - 1 do
    for x := 0 to W - 1 do
    begin
      Slot := PaletteIndex(Map[y * W + x], FirstId, IsSegmentMap);
      WriteLn(PPM, csRGB[Slot, 0], ' ', csRGB[Slot, 1], ' ', csRGB[Slot, 2]);
    end;
  CloseFile(PPM);
  WriteLn;
  WriteLn('Wrote ', PPMFileName, ' (', W, 'x', H, ' colored label map).');
end;

// Parses "1,3" into label ids; an empty string gives no ids.
function ParseLabelList(const Text: string): TMask2FormerLabelMap;
var
  Parts: TStringArray;
  PartPos: integer;
begin
  Result := nil;
  if Text = '' then Exit;
  Parts := Text.Split([',']);
  SetLength(Result, Length(Parts));
  for PartPos := 0 to High(Parts) do
    Result[PartPos] := StrToInt(Trim(Parts[PartPos]));
end;

var
  NN: TNNet;
  Config: TMask2FormerConfig;
  Root: TJSONData;
  JsonText: TStringList;
  Obj: TJSONObject;
  MaskFeatArr, LvArr, Lv, PosLvArr, PosLv: TJSONArray;
  MaskFeatures, ClassLogits, MaskLogits: TNNetVolume;
  SrcArr, PosArr: array of TNNetVolume;
  LabelMap: TMask2FormerLabelMap;
  Segmentation: TMask2FormerSegmentation;
  i, c, p, s, NumPix, NumKeys, Hidden, W, H, FirstId, ArgPos: integer;
  Counts: array of integer;
  Task: TSegmentationTask;
  TaskName, FuseText: string;
  Threshold, MaskThreshold, OverlapThreshold: TNeuralFloat;
begin
  TaskName := 'semantic';
  FuseText := '';
  Threshold := 0.5;
  MaskThreshold := 0.5;
  OverlapThreshold := 0.8;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    if (ArgPos < ParamCount) and (ParamStr(ArgPos) = '--task') then
      TaskName := LowerCase(ParamStr(ArgPos + 1))
    else if (ArgPos < ParamCount) and (ParamStr(ArgPos) = '--threshold') then
      Threshold := StrToFloat(ParamStr(ArgPos + 1))
    else if (ArgPos < ParamCount) and (ParamStr(ArgPos) = '--mask-threshold') then
      MaskThreshold := StrToFloat(ParamStr(ArgPos + 1))
    else if (ArgPos < ParamCount) and (ParamStr(ArgPos) = '--overlap-threshold') then
      OverlapThreshold := StrToFloat(ParamStr(ArgPos + 1))
    else if (ArgPos < ParamCount) and (ParamStr(ArgPos) = '--fuse') then
      FuseText := ParamStr(ArgPos + 1)
    else
    begin
      WriteLn('usage: UniversalSegmentation [--task semantic|instance|panoptic]');
      WriteLn('  [--threshold X] [--mask-threshold X] [--overlap-threshold X]');
      WriteLn('  [--fuse label,label]');
      Halt(1);
    end;
    Inc(ArgPos, 2);
  end;
  if TaskName = 'semantic' then Task := stSemantic
  else if TaskName = 'instance' then Task := stInstance
  else if TaskName = 'panoptic' then Task := stPanoptic
  else
  begin
    WriteLn('unknown --task "', TaskName, '" (semantic, instance or panoptic)');
    Halt(1);
  end;
  WriteLn('Mask2Former universal-segmentation example (', TaskName, ')');
  WriteLn('-----------------------------------------------------');
  NN := BuildMask2FormerFromSafeTensors(
    FixturePath('tiny_mask2former.safetensors'), Config,
    {pTrainable=}false, FixturePath('tiny_mask2former_config.json'));
  JsonText := TStringList.Create;
  Root := nil;
  MaskFeatures := TNNetVolume.Create;
  ClassLogits := TNNetVolume.Create;
  MaskLogits := TNNetVolume.Create;
  Hidden := Config.Hidden;
  W := Config.MaskWidth; H := Config.MaskHeight;
  NumPix := W * H;
  SetLength(SrcArr, Length(Config.Levels));
  SetLength(PosArr, Length(Config.Levels));
  for i := 0 to Length(Config.Levels) - 1 do
  begin
    SrcArr[i] := TNNetVolume.Create;
    PosArr[i] := TNNetVolume.Create;
  end;
  try
    WriteLn(Mask2FormerConfigToString(Config));
    // Load the precomputed pixel-decoder feature maps from the fixture io.json.
    JsonText.LoadFromFile(FixturePath('tiny_mask2former_io.json'));
    Root := GetJSON(JsonText.Text);
    Obj := TJSONObject(Root);
    // mask_features: io is channel-major [HIDDEN, H, W]; volume (W,H,HIDDEN) is
    // pixel-major FData[p*HIDDEN + c]. Transpose.
    MaskFeatArr := TJSONArray(Obj.Find('mask_features'));
    MaskFeatures.ReSize(W, H, Hidden);
    for c := 0 to Hidden - 1 do
      for p := 0 to NumPix - 1 do
        MaskFeatures.FData[p * Hidden + c] :=
          MaskFeatArr.Items[c * NumPix + p].AsFloat;
    // value memory + keys (= memory + sine pos) per level.
    LvArr := TJSONArray(Obj.Find('level_sources'));
    PosLvArr := TJSONArray(Obj.Find('level_pos'));
    for i := 0 to Length(Config.Levels) - 1 do
    begin
      NumKeys := Config.Levels[i].Width * Config.Levels[i].Height;
      SrcArr[i].ReSize(NumKeys, 1, Hidden);
      PosArr[i].ReSize(NumKeys, 1, Hidden);
      Lv := TJSONArray(LvArr.Items[i]);
      PosLv := TJSONArray(PosLvArr.Items[i]);
      for s := 0 to NumKeys - 1 do
        for c := 0 to Hidden - 1 do
        begin
          SrcArr[i].FData[s * Hidden + c] := Lv.Items[s * Hidden + c].AsFloat;
          PosArr[i].FData[s * Hidden + c] :=
            Lv.Items[s * Hidden + c].AsFloat + PosLv.Items[s * Hidden + c].AsFloat;
        end;
    end;

    // Run the masked-attention decoder (drives the per-layer mask feedback).
    RunMask2FormerSemantic(NN, Config, MaskFeatures, SrcArr, PosArr,
      ClassLogits, MaskLogits);

    if Task = stSemantic then
    begin
      // Fold the per-query class + mask logits into a per-pixel semantic map.
      LabelMap := DecodeMask2FormerSemantic(ClassLogits, MaskLogits,
        Config.NumLabels, W, H);
      WriteLn;
      WriteLn('Semantic label map (', W, 'x', H, ', one glyph per class):');
      PrintMap(LabelMap, W, H, 0, {IsSegmentMap=}false);
      SetLength(Counts, Config.NumLabels);
      for p := 0 to NumPix - 1 do
        Inc(Counts[LabelMap[p]]);
      WriteLn;
      WriteLn('Per-class pixel counts:');
      for c := 0 to Config.NumLabels - 1 do
        WriteLn('  class ', c, ' (', csPalette[c and 7], '): ', Counts[c]);
      WriteMapPPM(LabelMap, W, H, 0, {IsSegmentMap=}false, 'segmentation.ppm');
    end
    else
    begin
      // transformers' 384x384 processed grid, resized back to the mask grid.
      if Task = stInstance then
      begin
        Segmentation := DecodeMask2FormerInstance(ClassLogits, MaskLogits,
          Config.NumLabels, W, H, Threshold, W, H);
        FirstId := 0;
      end
      else
      begin
        Segmentation := DecodeMask2FormerPanoptic(ClassLogits, MaskLogits,
          Config.NumLabels, W, H, ParseLabelList(FuseText), Threshold,
          MaskThreshold, OverlapThreshold, W, H);
        FirstId := 1;
      end;
      WriteLn;
      WriteLn(TaskName, ' segment map (', W, 'x', H, ', "',
        csPalette[0], '" = no segment, ids cycle over the other glyphs):');
      PrintMap(Segmentation.SegmentMap, W, H, FirstId, {IsSegmentMap=}true);
      WriteLn;
      WriteLn('Segments (', Length(Segmentation.Segments), '):');
      for i := 0 to High(Segmentation.Segments) do
        WriteLn(Format('  id %d (%s) label %d was_fused %s score %.6f',
          [Segmentation.Segments[i].Id,
           csPalette[PaletteIndex(Segmentation.Segments[i].Id, FirstId, true)],
           Segmentation.Segments[i].LabelId,
           BoolToStr(Segmentation.Segments[i].WasFused, 'true', 'false'),
           Segmentation.Segments[i].Score]));
      WriteMapPPM(Segmentation.SegmentMap, W, H, FirstId, {IsSegmentMap=}true,
        'segmentation.ppm');
    end;
  finally
    Root.Free;
    JsonText.Free;
    MaskFeatures.Free;
    ClassLogits.Free;
    MaskLogits.Free;
    for i := 0 to Length(SrcArr) - 1 do begin SrcArr[i].Free; PosArr[i].Free; end;
    FreeMask2Former(NN);
  end;
end.
