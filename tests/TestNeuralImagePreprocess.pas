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
identity-resize case, so the normalization oracle is exact there. Shrinks are
checked against hand-computed PIL BILINEAR values and against ResizeImagePIL.
Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, FPImage,
  neuralnetwork, neuralvolume, neuraldatasets, neuralimageresize;

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
    // 4x4 -> 2x2 default (bilinear) shrink: PIL's antialiased taps are
    // (3,3,1)/7 and (1,3,3)/7 per axis, hand-computed.
    procedure TestBilinearAntialiasShrinkHandComputed;
    // 10x7 -> 5x4 bicubic shrink (long side Trunc(4*10/7) = 5, not 6) equals
    // ResizeImagePIL with uint8 rounding, center-cropped and normalized.
    procedure TestBicubicShrinkMatchesResizeImagePIL;
    // ReadImagePreprocessConfig parses size/crop_size as ints and the
    // image_mean/image_std triples from a HF preprocessor_config.json.
    procedure TestReadConfigIntSizes;
    // size as {"shortest_edge":N} and crop_size as {"height":..,"width":..}.
    procedure TestReadConfigObjectSizes;
    // absent mean/std fall back to the CLIP defaults.
    procedure TestReadConfigDefaults;
    // A (W,H,4) volume saved as PNG is 8-bit RGBA (colour type 6) and reloads
    // with its alpha; a 3-channel volume reloads opaque.
    procedure TestSaveImageRGBAPngRoundTrip;
    // GaussianBlurImage of a centred hot pixel keeps its mass of 1, is
    // symmetric, peaks at 1/S^2 (S = the 1-D tap sum); a flat channel stays flat.
    procedure TestGaussianBlurImageHotPixel;
    // Radius 0 copies Source into Dest; in place it changes nothing.
    procedure TestGaussianBlurImageZeroRadiusIsIdentity;
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

procedure TTestNeuralImagePreprocess.TestBilinearAntialiasShrinkHandComputed;
const
  // Per-axis PIL BILINEAR 4 -> 2 of 14*x (x) and 70*y (y).
  ExpectedX: array[0..1] of TNeuralFloat = (10, 32);
  ExpectedY: array[0..1] of TNeuralFloat = (50, 160);
var
  Src, Dst: TNNetVolume;
  X, Y, C: integer;
  Mean, Std: array[0..2] of TNeuralFloat;
begin
  Mean[0] := 0; Mean[1] := 0; Mean[2] := 0;
  Std[0]  := 1; Std[1]  := 1; Std[2]  := 1;
  Src := TNNetVolume.Create(4, 4, 3);
  Dst := TNNetVolume.Create;
  try
    for X := 0 to 3 do
      for Y := 0 to 3 do
        for C := 0 to 2 do
          Src[X, Y, C] := 14 * X + 70 * Y + C;
    PreprocessImageForVisionModel(Src, Dst, 2, 2, Mean, Std);
    AssertEquals('out W', 2, Dst.SizeX);
    AssertEquals('out H', 2, Dst.SizeY);
    for Y := 0 to 1 do
      for X := 0 to 1 do
        for C := 0 to 2 do
          AssertEquals(Format('bilinear (%d,%d,%d)', [X, Y, C]),
            (ExpectedX[X] + ExpectedY[Y] + C) / 255.0, Dst[X, Y, C],
            csOracleTol);
  finally
    Src.Free;
    Dst.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestBicubicShrinkMatchesResizeImagePIL;
var
  Src, Dst, Resized: TNNetVolume;
  X, Y, C: integer;
  Mean, Std: array[0..2] of TNeuralFloat;
  Expected: TNeuralFloat;
begin
  Mean[0] := 0.485; Mean[1] := 0.456; Mean[2] := 0.406;
  Std[0]  := 0.229; Std[1]  := 0.224; Std[2]  := 0.225;
  Src := TNNetVolume.Create(10, 7, 3);
  Dst := TNNetVolume.Create;
  Resized := TNNetVolume.Create;
  try
    // Hard 0/255 edges in channel 2 make the bicubic overshoot clamp.
    for X := 0 to 9 do
      for Y := 0 to 6 do
      begin
        Src[X, Y, 0] := (X * 35 + Y * 16) mod 256;
        Src[X, Y, 1] := (X * 37 + Y * 91) mod 256;
        Src[X, Y, 2] := 255 * ((X + Y) mod 2);
      end;
    PreprocessImageForVisionModel(Src, Dst, 4, 4, Mean, Std, irfBicubic);
    ResizeImagePIL(Src, Resized, 5, 4, irfBicubic, {RoundToBytes=}true);
    AssertEquals('out W', 4, Dst.SizeX);
    AssertEquals('out H', 4, Dst.SizeY);
    // 5x4 -> 4x4 center crop: OffX = 0, OffY = 0.
    for Y := 0 to 3 do
      for X := 0 to 3 do
        for C := 0 to 2 do
        begin
          AssertTrue('uint8 rounded',
            (Resized[X, Y, C] = Round(Resized[X, Y, C])) and
            (Resized[X, Y, C] >= 0) and (Resized[X, Y, C] <= 255));
          Expected := (Resized[X, Y, C] / 255.0 - Mean[C]) / Std[C];
          AssertEquals(Format('bicubic (%d,%d,%d)', [X, Y, C]),
            Expected, Dst[X, Y, C], csOracleTol);
        end;
  finally
    Src.Free;
    Dst.Free;
    Resized.Free;
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
    '{"size":256,"crop_size":224,"resample":2,' +
    '"image_mean":[0.485,0.456,0.406],' +
    '"image_std":[0.229,0.224,0.225]}');
  try
    Cfg := ReadImagePreprocessConfig(FN);
    AssertEquals('shorter', 256, Cfg.ResizeShorterSide);
    AssertEquals('crop', 224, Cfg.CropSize);
    AssertEquals('mean0', 0.485, Cfg.Mean[0], csOracleTol);
    AssertEquals('mean2', 0.406, Cfg.Mean[2], csOracleTol);
    AssertEquals('std1', 0.224, Cfg.Std[1], csOracleTol);
    AssertTrue('resample 2 = bilinear', Cfg.Resample = irfBilinear);
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
    AssertTrue('resample default bicubic', Cfg.Resample = irfBicubic);
  finally
    DeleteFile(FN);
  end;
end;

procedure TTestNeuralImagePreprocess.TestSaveImageRGBAPngRoundTrip;
var
  Src, Reloaded: TNNetVolume;
  Img: TFPMemoryImage;
  FN: string;
  PngHeader: TFileStream;
  HeaderBytes: array[0..25] of byte;
  X, Y, C: integer;
begin
  FN := GetTempDir + 'cai_rgba_roundtrip.png';
  Src := TNNetVolume.Create(5, 3, 4);
  Reloaded := TNNetVolume.Create;
  Img := TFPMemoryImage.Create(1, 1);
  try
    // Fractional values: the saver rounds to the nearest byte.
    for Y := 0 to 2 do
      for X := 0 to 4 do
        for C := 0 to 3 do
          Src[X, Y, C] := (X * 37 + Y * 71 + C * 53) mod 256 + 0.3;
    Src[0, 0, 3] := 255;
    Src[1, 0, 3] := 0;
    AssertTrue('saved', SaveImageFromVolumeIntoFile(Src, FN));
    PngHeader := TFileStream.Create(FN, fmOpenRead);
    try
      PngHeader.ReadBuffer(HeaderBytes, SizeOf(HeaderBytes));
    finally
      PngHeader.Free;
    end;
    AssertEquals('IHDR bit depth', 8, HeaderBytes[24]);
    AssertEquals('IHDR colour type 6 = RGBA', 6, HeaderBytes[25]);
    AssertTrue('reloaded', Img.LoadFromFile(FN));
    LoadImageIntoVolume(Img, Reloaded, {pWithAlpha=}true);
    AssertEquals('width', 5, Reloaded.SizeX);
    AssertEquals('height', 3, Reloaded.SizeY);
    AssertEquals('4 channels', 4, Reloaded.Depth);
    for Y := 0 to 2 do
      for X := 0 to 4 do
        for C := 0 to 3 do
          AssertEquals('pixel ' + IntToStr(X) + ',' + IntToStr(Y) + ' c' +
            IntToStr(C), Src[X, Y, C], Reloaded[X, Y, C], 0.5);
    // Fully opaque RGBA is still written as RGBA.
    for Y := 0 to 2 do
      for X := 0 to 4 do Src[X, Y, 3] := 255;
    AssertTrue('saved opaque', SaveImageFromVolumeIntoFile(Src, FN));
    PngHeader := TFileStream.Create(FN, fmOpenRead);
    try
      PngHeader.ReadBuffer(HeaderBytes, SizeOf(HeaderBytes));
    finally
      PngHeader.Free;
    end;
    AssertEquals('opaque RGBA: colour type 6', 6, HeaderBytes[25]);
    Src.ReSize(5, 3, 3);
    Src.Fill(10);
    AssertTrue('saved RGB', SaveImageFromVolumeIntoFile(Src, FN));
    AssertTrue('reloaded RGB', Img.LoadFromFile(FN));
    LoadImageIntoVolume(Img, Reloaded, {pWithAlpha=}true);
    AssertEquals('RGB file reloads opaque', 255, Reloaded[4, 2, 3], 0);
    AssertEquals('RGB value', 10, Reloaded[4, 2, 1], 0);
  finally
    DeleteFile(FN);
    Img.Free;
    Reloaded.Free;
    Src.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestGaussianBlurImageHotPixel;
const
  csSide = 15;
  csCenter = 7;
  csRadius = 1.5;
  csFlatValue = 5;
var
  Image: TNNetVolume;
  X, Y, OffsetX, OffsetY, Distance: integer;
  MassSum, TapSum, Value: double;
begin
  Image := TNNetVolume.Create(csSide, csSide, 2);
  try
    Image.Fill(0);
    for Y := 0 to csSide - 1 do
      for X := 0 to csSide - 1 do Image[X, Y, 1] := csFlatValue;
    Image[csCenter, csCenter, 0] := 1;
    GaussianBlurImage(Image, Image, csRadius);
    MassSum := 0;
    for Y := 0 to csSide - 1 do
      for X := 0 to csSide - 1 do
      begin
        MassSum := MassSum + Image[X, Y, 0];
        AssertEquals('flat channel at ' + IntToStr(X) + ',' + IntToStr(Y),
          csFlatValue, Image[X, Y, 1], 1e-5);
      end;
    AssertEquals('mass', 1, MassSum, 1e-5);
    // Taps reach Ceil(3 * 1.5) = 5 pixels, so the centre value is 1/TapSum^2.
    TapSum := 0;
    for Distance := -5 to 5 do
      TapSum := TapSum + Exp(-Distance * Distance / (2 * csRadius * csRadius));
    AssertEquals('peak', 1 / (TapSum * TapSum),
      Image[csCenter, csCenter, 0], 1e-6);
    AssertEquals('no taps beyond 5 pixels', 0, Image[csCenter + 6, csCenter, 0], 0);
    for OffsetY := -6 to 6 do
      for OffsetX := -6 to 6 do
      begin
        Value := Image[csCenter + OffsetX, csCenter + OffsetY, 0];
        AssertEquals('mirror X', Value,
          Image[csCenter - OffsetX, csCenter + OffsetY, 0], 1e-7);
        AssertEquals('mirror Y', Value,
          Image[csCenter + OffsetX, csCenter - OffsetY, 0], 1e-7);
        AssertEquals('transpose', Value,
          Image[csCenter + OffsetY, csCenter + OffsetX, 0], 1e-7);
        AssertTrue('peak is the maximum',
          Value <= Image[csCenter, csCenter, 0]);
      end;
  finally
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestGaussianBlurImageZeroRadiusIsIdentity;
var
  Source, Dest: TNNetVolume;
  ElementPos: integer;
begin
  Source := TNNetVolume.Create(4, 3, 3);
  Dest := TNNetVolume.Create;
  try
    for ElementPos := 0 to Source.Size - 1 do
      Source.FData[ElementPos] := (ElementPos * 37) mod 256;
    GaussianBlurImage(Source, Dest, 0);
    AssertEquals('width', 4, Dest.SizeX);
    AssertEquals('height', 3, Dest.SizeY);
    AssertEquals('depth', 3, Dest.Depth);
    for ElementPos := 0 to Source.Size - 1 do
      AssertEquals('copied element ' + IntToStr(ElementPos),
        Source.FData[ElementPos], Dest.FData[ElementPos], 0);
    GaussianBlurImage(Dest, Dest, 0);
    for ElementPos := 0 to Source.Size - 1 do
      AssertEquals('in-place element ' + IntToStr(ElementPos),
        Source.FData[ElementPos], Dest.FData[ElementPos], 0);
  finally
    Dest.Free;
    Source.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralImagePreprocess);
end.
