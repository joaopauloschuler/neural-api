(*
  UniversalSegmentation - semantic inference with an imported Mask2Former model.

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
  per-query class logits and mask logits into a per-pixel SEMANTIC label map
  (DecodeMask2FormerSemantic: softmax the classes, drop the no-object slot,
  sigmoid the masks, class-weighted argmax), prints it as a colored ASCII
  palette, and writes the label map as a colored PPM image. A real run would
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
  i, c, p, s, NumPix, NumKeys, Hidden, W, H, x, y, lab: integer;
  Line: string;
  Counts: array of integer;
  PPM: TextFile;
begin
  WriteLn('Mask2Former universal-segmentation example (semantic)');
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

    // Fold the per-query class + mask logits into a per-pixel semantic map.
    LabelMap := DecodeMask2FormerSemantic(ClassLogits, MaskLogits,
      Config.NumLabels, W, H);

    WriteLn;
    WriteLn('Semantic label map (', W, 'x', H, ', one glyph per class):');
    for y := 0 to H - 1 do
    begin
      Line := '  ';
      for x := 0 to W - 1 do
        Line := Line + csPalette[LabelMap[y * W + x] and 7] + ' ';
      WriteLn(Line);
    end;

    SetLength(Counts, Config.NumLabels);
    for p := 0 to NumPix - 1 do
      Inc(Counts[LabelMap[p]]);
    WriteLn;
    WriteLn('Per-class pixel counts:');
    for c := 0 to Config.NumLabels - 1 do
      WriteLn('  class ', c, ' (', csPalette[c and 7], '): ', Counts[c]);

    // Write the colored label map as a PPM (P3 ASCII) overlay.
    AssignFile(PPM, 'segmentation.ppm');
    Rewrite(PPM);
    WriteLn(PPM, 'P3');
    WriteLn(PPM, W, ' ', H);
    WriteLn(PPM, 255);
    for y := 0 to H - 1 do
      for x := 0 to W - 1 do
      begin
        lab := LabelMap[y * W + x] and 7;
        WriteLn(PPM, csRGB[lab, 0], ' ', csRGB[lab, 1], ' ', csRGB[lab, 2]);
      end;
    CloseFile(PPM);
    WriteLn;
    WriteLn('Wrote segmentation.ppm (', W, 'x', H, ' colored label map).');
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
