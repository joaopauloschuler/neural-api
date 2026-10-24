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

  --model eomt runs an EoMT checkpoint folder (config.json + model.safetensors
  + preprocessor_config.json, e.g. tue-mps/coco_panoptic_eomt_large_640) on an
  image file: EoMTPreprocessImage (resize, pad or split into windows), one
  forward per window (BuildEoMTFromSafeTensors), SplitEoMTOutput, then
  DecodeEoMTSemantic / DecodeEoMTInstance / DecodeEoMTPanoptic at the original
  image size, with the same printout and PPM.

  Coded by Joao Paulo Schwarz Schuler with Claude (AI).
  https://github.com/joaopauloschuler/neural-api

  Coded by Claude (AI).
*)
program UniversalSegmentation;

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, fpjson, jsonparser,
  neuralnetwork, neuralvolume, neuralpretrained, neuraldatasets;

const
  // Palette glyphs / RGB, one per class (the pico fixture has 4 classes).
  csPalette: array[0..7] of char = ('.', ':', '+', '#', '@', '*', 'o', '=');
  csRGB: array[0..7, 0..2] of byte = (
    (40, 40, 60), (220, 60, 60), (60, 200, 90), (70, 120, 230),
    (230, 200, 60), (200, 70, 210), (60, 210, 210), (200, 200, 200));
  // Larger maps are written to the PPM only, not printed as glyphs.
  csMaxPrintedWidth = 80;

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


type
  TSegmentationOptions = record
    Task: TSegmentationTask;
    TaskName, FuseText: string;
    Threshold, MaskThreshold, OverlapThreshold: TNeuralFloat;
  end;

// Prints the semantic map (when small), per-class pixel counts, writes the PPM.
procedure ReportSemantic(const LabelMap: TMask2FormerLabelMap;
  W, H, NumLabels: integer);
var
  Counts: array of integer;
  p, c: integer;
begin
  WriteLn;
  WriteLn('Semantic label map (', W, 'x', H, ', one glyph per class):');
  if W <= csMaxPrintedWidth then
    PrintMap(LabelMap, W, H, 0, {IsSegmentMap=}false);
  SetLength(Counts, NumLabels);
  for p := 0 to High(LabelMap) do
    Inc(Counts[LabelMap[p]]);
  WriteLn;
  WriteLn('Per-class pixel counts:');
  for c := 0 to NumLabels - 1 do
    WriteLn('  class ', c, ' (', csPalette[c and 7], '): ', Counts[c]);
  WriteMapPPM(LabelMap, W, H, 0, {IsSegmentMap=}false, 'segmentation.ppm');
end;

// Prints a segment map (when small) and its segment list, writes the PPM.
procedure ReportSegmentation(const Segmentation: TMask2FormerSegmentation;
  const TaskName: string; FirstId: integer);
var
  i: integer;
begin
  WriteLn;
  WriteLn(TaskName, ' segment map (', Segmentation.Width, 'x',
    Segmentation.Height, ', "', csPalette[0],
    '" = no segment, ids cycle over the other glyphs):');
  if Segmentation.Width <= csMaxPrintedWidth then
    PrintMap(Segmentation.SegmentMap, Segmentation.Width, Segmentation.Height,
      FirstId, {IsSegmentMap=}true);
  WriteLn;
  WriteLn('Segments (', Length(Segmentation.Segments), '):');
  for i := 0 to High(Segmentation.Segments) do
    WriteLn(Format('  id %d (%s) label %d was_fused %s score %.6f',
      [Segmentation.Segments[i].Id,
       csPalette[PaletteIndex(Segmentation.Segments[i].Id, FirstId, true)],
       Segmentation.Segments[i].LabelId,
       BoolToStr(Segmentation.Segments[i].WasFused, 'true', 'false'),
       Segmentation.Segments[i].Score]));
  WriteMapPPM(Segmentation.SegmentMap, Segmentation.Width, Segmentation.Height,
    FirstId, {IsSegmentMap=}true, 'segmentation.ppm');
end;

// The committed pico Mask2Former decoder on its precomputed feature maps.
procedure RunMask2Former(const Options: TSegmentationOptions);
var
  NN: TNNet;
  Config: TMask2FormerConfig;
  Root: TJSONData;
  JsonText: TStringList;
  Obj: TJSONObject;
  MaskFeatArr, LvArr, Lv, PosLvArr, PosLv: TJSONArray;
  MaskFeatures, ClassLogits, MaskLogits: TNNetVolume;
  SrcArr, PosArr: array of TNNetVolume;
  i, c, p, s, NumPix, NumKeys, Hidden, W, H: integer;
begin
  WriteLn('Mask2Former universal-segmentation example (', Options.TaskName, ')');
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

    // transformers' 384x384 processed grid, resized back to the mask grid.
    case Options.Task of
      stSemantic:
        ReportSemantic(DecodeMask2FormerSemantic(ClassLogits, MaskLogits,
          Config.NumLabels, W, H, W, H), W, H, Config.NumLabels);
      stInstance:
        ReportSegmentation(DecodeMask2FormerInstance(ClassLogits, MaskLogits,
          Config.NumLabels, W, H, Options.Threshold, W, H), Options.TaskName, 0);
      stPanoptic:
        ReportSegmentation(DecodeMask2FormerPanoptic(ClassLogits, MaskLogits,
          Config.NumLabels, W, H, ParseLabelList(Options.FuseText),
          Options.Threshold, Options.MaskThreshold, Options.OverlapThreshold,
          W, H), Options.TaskName, 1);
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
end;

// The task a tue-mps folder name implies: do_split_image checkpoints are
// semantic; otherwise "instance" / "semantic" in the name, else panoptic.
function EoMTDefaultTask(const ModelDir: string;
  const Processor: TEoMTProcessorConfig): TSegmentationTask;
var
  FolderName: string;
begin
  FolderName := LowerCase(ExtractFileName(ExcludeTrailingPathDelimiter(ModelDir)));
  if Processor.DoSplitImage or (Pos('semantic', FolderName) > 0) then
    Result := stSemantic
  else if Pos('instance', FolderName) > 0 then Result := stInstance
  else Result := stPanoptic;
end;

// An EoMT checkpoint folder on an image file, for Options.Task.
procedure RunEoMT(const ModelDir, ImagePath: string;
  const Options: TSegmentationOptions);
const
  csTaskNames: array[TSegmentationTask] of string =
    ('semantic', 'instance', 'panoptic');
var
  NN: TNNet;
  Config: TEoMTConfig;
  Processor: TEoMTProcessorConfig;
  Layout: TEoMTImageLayout;
  Image: TNNetVolume;
  Windows: TNNetVolumeList;
  ClassLogits, MaskLogits: array of TNNetVolume;
  MaskGrid, WindowPos, ImageSize: integer;
  StartTime: TDateTime;
begin
  WriteLn('EoMT universal-segmentation example (', csTaskNames[Options.Task], ')');
  WriteLn('-----------------------------------------------------');
  Processor := ReadEoMTProcessorConfigFromJSONFile(
    ModelDir + 'preprocessor_config.json');
  if Processor.DoSplitImage and (Options.Task <> stSemantic) then
  begin
    WriteLn('This checkpoint splits images into windows (do_split_image); ',
      'only --task semantic merges them.');
    Halt(1);
  end;
  NN := BuildEoMTFromSafeTensors(ModelDir + 'model.safetensors', Config,
    {pTrainable=}false, ModelDir + 'config.json');
  Image := TNNetVolume.Create;
  Windows := TNNetVolumeList.Create();
  ClassLogits := nil;
  MaskLogits := nil;
  try
    WriteLn(EoMTConfigToString(Config));
    if not LoadImageFromFileIntoVolume(ImagePath, Image) then
    begin
      WriteLn('Could not load image: ', ImagePath);
      Halt(1);
    end;
    EoMTPreprocessImage(Image, Processor, Windows, Layout);
    ImageSize := Config.Backbone.ImageSize;
    WriteLn('Image ', ImagePath, ' (', Image.SizeX, 'x', Image.SizeY,
      ') resized to ', Layout.ResizedWidth, 'x', Layout.ResizedHeight, ', ',
      Windows.Count, ' window(s) of ', Windows[0].SizeX, 'x', Windows[0].SizeY);
    MaskGrid := EoMTMaskGridSize(Config);
    SetLength(ClassLogits, Windows.Count);
    SetLength(MaskLogits, Windows.Count);
    StartTime := Now;
    for WindowPos := 0 to Windows.Count - 1 do
    begin
      if (Windows[WindowPos].SizeX <> ImageSize) or
        (Windows[WindowPos].SizeY <> ImageSize) then
      begin
        WriteLn('Window ', WindowPos, ' is ', Windows[WindowPos].SizeX, 'x',
          Windows[WindowPos].SizeY, ', the net takes ', ImageSize, 'x',
          ImageSize, ' (preprocessor_config.json needs do_pad or ',
          'do_split_image).');
        Halt(1);
      end;
      ClassLogits[WindowPos] := TNNetVolume.Create;
      MaskLogits[WindowPos] := TNNetVolume.Create;
      NN.Compute(Windows[WindowPos]);
      SplitEoMTOutput(NN.GetLastLayer.Output, Config, ClassLogits[WindowPos],
        MaskLogits[WindowPos]);
    end;
    WriteLn('Forward: ', Windows.Count, ' window(s) in ',
      FormatFloat('0.000', (Now - StartTime) * 86400), ' s');
    case Options.Task of
      stSemantic:
        ReportSemantic(DecodeEoMTSemantic(ClassLogits, MaskLogits,
          Config.NumLabels, MaskGrid, Layout), Layout.OriginalWidth,
          Layout.OriginalHeight, Config.NumLabels);
      stInstance:
        ReportSegmentation(DecodeEoMTInstance(ClassLogits[0], MaskLogits[0],
          Config.NumLabels, MaskGrid, Layout, Options.Threshold),
          csTaskNames[Options.Task], 0);
      stPanoptic:
        ReportSegmentation(DecodeEoMTPanoptic(ClassLogits[0], MaskLogits[0],
          Config.NumLabels, MaskGrid, Layout, ParseLabelList(Options.FuseText),
          Options.Threshold, Options.MaskThreshold, Options.OverlapThreshold),
          csTaskNames[Options.Task], 0);
    end;
  finally
    for WindowPos := 0 to High(ClassLogits) do
    begin
      ClassLogits[WindowPos].Free;
      MaskLogits[WindowPos].Free;
    end;
    Windows.Free;
    Image.Free;
    NN.Free;
  end;
end;

procedure PrintUsage;
begin
  WriteLn('usage: UniversalSegmentation [--model mask2former|eomt]');
  WriteLn('  [--task semantic|instance|panoptic] [--threshold X]');
  WriteLn('  [--mask-threshold X] [--overlap-threshold X] [--fuse label,label]');
  WriteLn('  eomt only: --model-dir DIR --image FILE');
  Halt(1);
end;

var
  Options: TSegmentationOptions;
  ModelName, ModelDir, ImagePath: string;
  ArgPos: integer;
  HasTask, HasThreshold: boolean;
begin
  ModelName := 'mask2former';
  ModelDir := '';
  ImagePath := '';
  Options.TaskName := 'semantic';
  Options.FuseText := '';
  Options.Threshold := 0.5;
  Options.MaskThreshold := 0.5;
  Options.OverlapThreshold := 0.8;
  HasTask := false;
  HasThreshold := false;
  ArgPos := 1;
  while ArgPos <= ParamCount do
  begin
    if ArgPos >= ParamCount then PrintUsage;
    if ParamStr(ArgPos) = '--model' then
      ModelName := LowerCase(ParamStr(ArgPos + 1))
    else if ParamStr(ArgPos) = '--model-dir' then
      ModelDir := IncludeTrailingPathDelimiter(ParamStr(ArgPos + 1))
    else if ParamStr(ArgPos) = '--image' then
      ImagePath := ParamStr(ArgPos + 1)
    else if ParamStr(ArgPos) = '--task' then
    begin
      Options.TaskName := LowerCase(ParamStr(ArgPos + 1));
      HasTask := true;
    end
    else if ParamStr(ArgPos) = '--threshold' then
    begin
      Options.Threshold := StrToFloat(ParamStr(ArgPos + 1));
      HasThreshold := true;
    end
    else if ParamStr(ArgPos) = '--mask-threshold' then
      Options.MaskThreshold := StrToFloat(ParamStr(ArgPos + 1))
    else if ParamStr(ArgPos) = '--overlap-threshold' then
      Options.OverlapThreshold := StrToFloat(ParamStr(ArgPos + 1))
    else if ParamStr(ArgPos) = '--fuse' then
      Options.FuseText := ParamStr(ArgPos + 1)
    else PrintUsage;
    Inc(ArgPos, 2);
  end;
  if Options.TaskName = 'semantic' then Options.Task := stSemantic
  else if Options.TaskName = 'instance' then Options.Task := stInstance
  else if Options.TaskName = 'panoptic' then Options.Task := stPanoptic
  else
  begin
    WriteLn('unknown --task "', Options.TaskName,
      '" (semantic, instance or panoptic)');
    Halt(1);
  end;
  if ModelName = 'mask2former' then RunMask2Former(Options)
  else if ModelName = 'eomt' then
  begin
    if (ModelDir = '') or (ImagePath = '') then PrintUsage;
    if not HasTask then
      Options.Task := EoMTDefaultTask(ModelDir,
        ReadEoMTProcessorConfigFromJSONFile(ModelDir + 'preprocessor_config.json'));
    // the EomtImageProcessor default threshold.
    if not HasThreshold then Options.Threshold := 0.8;
    RunEoMT(ModelDir, ImagePath, Options);
  end
  else
  begin
    WriteLn('unknown --model "', ModelName, '" (mask2former or eomt)');
    Halt(1);
  end;
end.
