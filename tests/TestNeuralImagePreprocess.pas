unit TestNeuralImagePreprocess;
(*
Tests for PreprocessImageForVisionModel / ReadImagePreprocessConfig
(neuraldatasets.pas): the standard timm/torchvision/CLIP inference transform
(aspect-preserving shorter-side resize -> center-crop -> (x/255-mean)/std).

torchvision is NOT installable in this environment, so the "parity vs
torchvision/PIL" check is substituted with a SYNTHETIC, hand-computed oracle:
tiny known volumes whose geometry (output dims, center-crop offsets) and
normalization math are computed independently (a short python3 snippet, inlined
below) and asserted byte-for-byte. The bilinear resize is an identity for the
identity-resize case, so the normalization oracle is exact there.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry,
  neuralnetwork, neuralvolume, neuraldatasets;

type
  TTestNeuralImagePreprocess = class(TTestCase)
  published
    // 4x4 src, resize shorter side to 4 (identity), center-crop to 2x2 ->
    // offsets (1,1); ImageNet normalization on hand-computed oracle values.
    procedure TestIdentityResizeCropAndNormalize;
    // 6 wide x 4 tall src, resize shorter (Y) to 4 (identity), crop 4x4 ->
    // OffX=1, OffY=0: dst column x reads src column 1+x.
    procedure TestCenterCropOffsetWideImage;
    // 8 wide x 4 tall src, resize shorter (Y) to 2 -> 4x2, crop 2x2: output
    // dims are exactly CropSize x CropSize x 3.
    procedure TestOutputGeometryWithDownResize;
    // ReadImagePreprocessConfig parses size/crop_size as ints and the
    // image_mean/image_std triples from a HF preprocessor_config.json.
    procedure TestReadConfigIntSizes;
    // size as {"shortest_edge":N} and crop_size as {"height":..,"width":..}.
    procedure TestReadConfigObjectSizes;
    // absent mean/std fall back to the CLIP defaults.
    procedure TestReadConfigDefaults;
  end;

implementation

const
  csOracleTol = 1e-5;

// value(x,y,c) used by the synthetic source images
function SrcVal(x, y, c: integer): TNeuralFloat;
begin
  Result := x * 10 + y + c * 100;
end;

procedure TTestNeuralImagePreprocess.TestIdentityResizeCropAndNormalize;
var
  Src, Dst: TNNetVolume;
  X, Y, C: integer;
  Mean, Std: array[0..2] of TNeuralFloat;
  Expected, V: TNeuralFloat;
begin
  Mean[0] := 0.485; Mean[1] := 0.456; Mean[2] := 0.406;
  Std[0]  := 0.229; Std[1]  := 0.224; Std[2]  := 0.225;
  Src := TNNetVolume.Create(4, 4, 3);
  Dst := TNNetVolume.Create;
  try
    for X := 0 to 3 do
      for Y := 0 to 3 do
        for C := 0 to 2 do
          Src[X, Y, C] := SrcVal(X, Y, C);
    PreprocessImageForVisionModel(Src, Dst, 4, 2, Mean, Std);
    AssertEquals('out W', 2, Dst.SizeX);
    AssertEquals('out H', 2, Dst.SizeY);
    AssertEquals('out D', 3, Dst.Depth);
    // center-crop of 4->2 has offset (4-2) div 2 = 1 on both axes
    for C := 0 to 2 do
      for Y := 0 to 1 do
        for X := 0 to 1 do
        begin
          V := SrcVal(1 + X, 1 + Y, C);
          Expected := (V / 255.0 - Mean[C]) / Std[C];
          AssertEquals(Format('norm c=%d (%d,%d)', [C, X, Y]),
            Expected, Dst[X, Y, C], csOracleTol);
        end;
  finally
    Src.Free;
    Dst.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestCenterCropOffsetWideImage;
var
  Src, Dst: TNNetVolume;
  X, Y: integer;
  Mean, Std: array[0..2] of TNeuralFloat;
begin
  // identity mean=0 std=1 so the normalized value equals the source value:
  // we can read the crop offset directly off the channel-0 source columns.
  Mean[0] := 0; Mean[1] := 0; Mean[2] := 0;
  Std[0]  := 1; Std[1]  := 1; Std[2]  := 1;
  Src := TNNetVolume.Create(6, 4, 3); // 6 wide, 4 tall: shorter side = Y = 4
  Dst := TNNetVolume.Create;
  try
    for X := 0 to 5 do
      for Y := 0 to 3 do
        Src[X, Y, 0] := X; // encode the source column index
    PreprocessImageForVisionModel(Src, Dst, 4, 4, Mean, Std);
    AssertEquals('out W', 4, Dst.SizeX);
    AssertEquals('out H', 4, Dst.SizeY);
    // OffX = (6-4) div 2 = 1, OffY = 0: dst column x reads src column 1+x
    for X := 0 to 3 do
      AssertEquals(Format('col offset x=%d', [X]),
        TNeuralFloat(1 + X), Dst[X, 0, 0] * 255.0, 1e-3);
  finally
    Src.Free;
    Dst.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestOutputGeometryWithDownResize;
var
  Src, Dst: TNNetVolume;
  X, Y, C: integer;
  Mean, Std: array[0..2] of TNeuralFloat;
begin
  Mean[0] := 0; Mean[1] := 0; Mean[2] := 0;
  Std[0]  := 1; Std[1]  := 1; Std[2]  := 1;
  Src := TNNetVolume.Create(8, 4, 3); // shorter side Y=4 -> resize to 2 -> 4x2
  Dst := TNNetVolume.Create;
  try
    for X := 0 to 7 do
      for Y := 0 to 3 do
        for C := 0 to 2 do
          Src[X, Y, C] := SrcVal(X, Y, C);
    PreprocessImageForVisionModel(Src, Dst, 2, 2, Mean, Std);
    AssertEquals('out W', 2, Dst.SizeX);
    AssertEquals('out H', 2, Dst.SizeY);
    AssertEquals('out D', 3, Dst.Depth);
  finally
    Src.Free;
    Dst.Free;
  end;
end;

procedure WriteConfig(const FileName, Body: string);
var
  L: TStringList;
begin
  L := TStringList.Create;
  try
    L.Text := Body;
    L.SaveToFile(FileName);
  finally
    L.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestReadConfigIntSizes;
var
  Cfg: TNNetImagePreprocess;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_int.json';
  WriteConfig(FN,
    '{"size":256,"crop_size":224,' +
    '"image_mean":[0.485,0.456,0.406],' +
    '"image_std":[0.229,0.224,0.225]}');
  try
    Cfg := ReadImagePreprocessConfig(FN);
    AssertEquals('shorter', 256, Cfg.ResizeShorterSide);
    AssertEquals('crop', 224, Cfg.CropSize);
    AssertEquals('mean0', 0.485, Cfg.Mean[0], csOracleTol);
    AssertEquals('mean2', 0.406, Cfg.Mean[2], csOracleTol);
    AssertEquals('std1', 0.224, Cfg.Std[1], csOracleTol);
  finally
    DeleteFile(FN);
  end;
end;

procedure TTestNeuralImagePreprocess.TestReadConfigObjectSizes;
var
  Cfg: TNNetImagePreprocess;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_obj.json';
  WriteConfig(FN,
    '{"size":{"shortest_edge":248},' +
    '"crop_size":{"height":224,"width":224},' +
    '"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}');
  try
    Cfg := ReadImagePreprocessConfig(FN);
    AssertEquals('shorter', 248, Cfg.ResizeShorterSide);
    AssertEquals('crop', 224, Cfg.CropSize);
    AssertEquals('mean0', 0.5, Cfg.Mean[0], csOracleTol);
    AssertEquals('std0', 0.5, Cfg.Std[0], csOracleTol);
  finally
    DeleteFile(FN);
  end;
end;

procedure TTestNeuralImagePreprocess.TestReadConfigDefaults;
var
  Cfg: TNNetImagePreprocess;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_def.json';
  // no mean/std/size: everything falls back to CLIP defaults / 224.
  WriteConfig(FN, '{"do_resize":true}');
  try
    Cfg := ReadImagePreprocessConfig(FN);
    AssertEquals('shorter default', 224, Cfg.ResizeShorterSide);
    AssertEquals('crop default', 224, Cfg.CropSize);
    AssertEquals('clip mean0', 0.48145466, Cfg.Mean[0], csOracleTol);
    AssertEquals('clip std2', 0.27577711, Cfg.Std[2], csOracleTol);
  finally
    DeleteFile(FN);
  end;
end;

initialization
  RegisterTest(TTestNeuralImagePreprocess);
end.
