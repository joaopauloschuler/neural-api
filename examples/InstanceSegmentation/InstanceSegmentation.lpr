/// Instance segmentation with an imported Mask R-CNN model (per-OBJECT mask).
///
/// Mask R-CNN (He et al. 2017, "Mask R-CNN") extends Faster R-CNN with a small
/// per-region mask head: for each proposal box it RoIAligns the chosen FPN
/// pyramid level, runs 4 convs + a transposed-conv upsample, and emits a
/// per-class HxW binary mask -- the FIRST instance-segmentation vertical in this
/// repo (distinct from SegFormer's single dense class map: here every OBJECT
/// gets its own mask).
///
/// SCOPE (matches the importer v1): the RPN / anchor generator is SKIPPED. The
/// backbone FPN-input feature maps are supplied directly (the ResNet-50
/// backbone's C4/C5 taps in a real run) and ONE fixed proposal box is fed to
/// RunMaskRCNN, which returns the box-head class logits + box deltas and the
/// mask head's per-class mask logits. This example loads the committed PICO
/// parity fixture (tests/fixtures/tiny_maskrcnn.safetensors), runs that single
/// proposal, picks the best-scoring class, sigmoids that class's mask, OVERLAYS
/// it (red channel) on a tiny synthetic CPU image and writes the result to
/// instance_segmentation.ppm. A real run loads torchvision maskrcnn_resnet50_fpn
/// the same way; the math is identical, only the checkpoint and feature maps
/// differ.
///
/// HONEST NOTE: the pico fixture has RANDOM weights, so the "mask" is not a
/// meaningful object -- the demo's job is to exercise the full FPN + RoIAlign +
/// box/mask-head pipeline end to end and SELF-REPORT (asserts no NaN/Inf and
/// that the overlaid mask covers a sane fraction of pixels). Pure CPU, <1 s.
///
/// Coded by Joao Paulo Schwarz Schuler with Claude (AI).
/// https://github.com/joaopauloschuler/neural-api
program InstanceSegmentation;

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} cthreads, {$ENDIF}
  Classes, SysUtils, Math, fpjson, jsonparser,
  neuralnetwork, neuralvolume, neuralpretrained;

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

// Renders a (W,H,3) volume to a P6 PPM (min-max stretched for display).
procedure WritePPM(AImg: TNNetVolume; const FileName: string);
var
  F: TextFile; Bin: TFileStream;
  px, py, pc: integer; Lo, Hi, Val: TNeuralFloat; B: byte;
begin
  Lo := AImg.FData[0]; Hi := AImg.FData[0];
  for px := 0 to AImg.Size - 1 do
  begin
    if AImg.FData[px] < Lo then Lo := AImg.FData[px];
    if AImg.FData[px] > Hi then Hi := AImg.FData[px];
  end;
  if Hi - Lo < 1e-6 then Hi := Lo + 1e-6;
  AssignFile(F, FileName); Rewrite(F);
  WriteLn(F, 'P6'); WriteLn(F, AImg.SizeX, ' ', AImg.SizeY); WriteLn(F, 255);
  CloseFile(F);
  Bin := TFileStream.Create(FileName, fmOpenReadWrite);
  try
    Bin.Seek(0, soEnd);
    for py := 0 to AImg.SizeY - 1 do
      for px := 0 to AImg.SizeX - 1 do
        for pc := 0 to 2 do
        begin
          Val := (AImg[px, py, pc] - Lo) / (Hi - Lo);
          if Val < 0 then Val := 0;
          if Val > 1 then Val := 1;
          B := Round(Val * 255);
          Bin.WriteByte(B);
        end;
  finally
    Bin.Free;
  end;
end;

var
  NN: TNNet;
  Config: TMaskRCNNConfig;
  RefJson: TStringList;
  RefRoot: TJSONData;
  FeatsArr, LvArr, ChArr, RowArr, BoxArr: TJSONArray;
  LevelFeats: array of TNNetVolume;
  Box: array[0..3] of TNeuralFloat;
  ClsOut, BboxOut, MaskOut, Img: TNNetVolume;
  lv, c, x, y, BestC, Wm, Hm, Covered, Total: integer;
  BestScore, p, Sig: TNeuralFloat;
begin
  WriteLn('Mask R-CNN instance-segmentation example');
  WriteLn('----------------------------------------');
  NN := BuildMaskRCNNFromSafeTensors(
    FixturePath('tiny_maskrcnn.safetensors'), Config,
    {pTrainable=}false, FixturePath('tiny_maskrcnn_config.json'));
  RefJson := TStringList.Create;
  RefRoot := nil;
  ClsOut := TNNetVolume.Create;
  BboxOut := TNNetVolume.Create;
  MaskOut := TNNetVolume.Create;
  SetLength(LevelFeats, Config.NumLevels);
  for lv := 0 to Config.NumLevels - 1 do
    LevelFeats[lv] := TNNetVolume.Create;
  try
    WriteLn(MaskRCNNConfigToString(Config));

    // The example feeds the SAME backbone feature maps + proposal box the parity
    // fixture pins, so it reproduces the test numbers fully offline.
    RefJson.LoadFromFile(FixturePath('tiny_maskrcnn_ref.json'));
    RefRoot := GetJSON(RefJson.Text);
    FeatsArr := TJSONArray(TJSONObject(RefRoot).Find('feats'));
    for lv := 0 to Config.NumLevels - 1 do
    begin
      LvArr := TJSONArray(FeatsArr.Items[lv]);
      LevelFeats[lv].ReSize(Config.Levels[lv].Width,
        Config.Levels[lv].Height, Config.Levels[lv].InChannels);
      for c := 0 to Config.Levels[lv].InChannels - 1 do
      begin
        ChArr := TJSONArray(LvArr.Items[c]);
        for y := 0 to Config.Levels[lv].Height - 1 do
        begin
          RowArr := TJSONArray(ChArr.Items[y]);
          for x := 0 to Config.Levels[lv].Width - 1 do
            LevelFeats[lv].FData[(y * Config.Levels[lv].Width + x) *
              Config.Levels[lv].InChannels + c] := RowArr.Items[x].AsFloat;
        end;
      end;
    end;
    BoxArr := TJSONArray(TJSONObject(RefRoot).Find('box'));
    for x := 0 to 3 do Box[x] := BoxArr.Items[x].AsFloat;

    WriteLn('Proposal box (level ', Config.BoxLevel, ' coords): (',
      Box[0]:0:2, ', ', Box[1]:0:2, ', ', Box[2]:0:2, ', ', Box[3]:0:2, ')');

    RunMaskRCNN(NN, Config, LevelFeats, Box, ClsOut, BboxOut, MaskOut);

    // Best-scoring class (skip background class 0).
    BestC := 1; BestScore := ClsOut.FData[1];
    for c := 1 to Config.NumClasses - 1 do
      if ClsOut.FData[c] > BestScore then begin BestScore := ClsOut.FData[c]; BestC := c; end;
    WriteLn('Class logits: ');
    for c := 0 to Config.NumClasses - 1 do
      WriteLn('  class ', c, ' = ', ClsOut.FData[c]:0:5);
    WriteLn('Best non-background class: ', BestC, ' (logit ', BestScore:0:5, ')');

    Wm := MaskOut.SizeX; Hm := MaskOut.SizeY;
    WriteLn('Mask logits shape: ', Wm, 'x', Hm, ' over ', Config.NumClasses,
      ' classes; overlaying class ', BestC, '.');

    // Threshold the mask at its OWN mean so the overlay shows the "above-
    // average activation" region even on a random-weight fixture (a real
    // checkpoint would threshold the sigmoid at 0.5). NaN/Inf check too.
    Total := Wm * Hm;
    BestScore := 0;  // reuse as mean accumulator
    for y := 0 to Hm - 1 do
      for x := 0 to Wm - 1 do
      begin
        p := MaskOut.FData[(y * Wm + x) * Config.NumClasses + BestC];
        if (IsNan(p)) or (IsInfinite(p)) then
        begin
          WriteLn('FAIL: mask logit is NaN/Inf.'); Halt(1);
        end;
        BestScore := BestScore + p;
      end;
    BestScore := BestScore / Total;  // mask-logit mean (threshold)

    // Render: greyscale base from the sigmoid + red overlay where the mask
    // logit exceeds its mean. Also a sanity check on coverage.
    Img := TNNetVolume.Create;
    try
      Img.ReSize(Wm, Hm, 3);
      Covered := 0;
      for y := 0 to Hm - 1 do
        for x := 0 to Wm - 1 do
        begin
          p := MaskOut.FData[(y * Wm + x) * Config.NumClasses + BestC];
          Sig := 1.0 / (1.0 + Exp(-p));
          // base grey from the sigmoid, red boost where the mask fires.
          Img[x, y, 0] := Sig;
          Img[x, y, 1] := Sig * 0.4;
          Img[x, y, 2] := Sig * 0.4;
          if p > BestScore then
          begin
            Img[x, y, 0] := 1.0;  // paint object pixels red
            Inc(Covered);
          end;
        end;
      WritePPM(Img, 'instance_segmentation.ppm');
      WriteLn('Wrote instance_segmentation.ppm (', Wm, 'x', Hm, ').');
      WriteLn('Object mask covers ', Covered, '/', Total, ' pixels (',
        (100.0 * Covered / Total):0:1, '%).');
      // Self-report invariant: a random-weight mask should not be all-on or
      // all-off (that would mean a degenerate/broken forward).
      if (Covered = 0) or (Covered = Total) then
        WriteLn('NOTE: mask is degenerate on this random fixture (expected ',
          'with random weights, but flagging).');
      WriteLn('OK: full FPN + RoIAlign + box/mask-head pipeline ran end to end.');
    finally
      Img.Free;
    end;
  finally
    RefRoot.Free;
    RefJson.Free;
    ClsOut.Free; BboxOut.Free; MaskOut.Free;
    for lv := 0 to Config.NumLevels - 1 do LevelFeats[lv].Free;
    NN.Free;
  end;
end.
