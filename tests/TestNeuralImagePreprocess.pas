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
  Classes, SysUtils, Math, fpcunit, testregistry, FPImage,
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
    // size {"height","width"} parses to the direct resize; the crop is (H,W)
    // unless crop_size is given and do_center_crop is on.
    procedure TestReadConfigHeightWidthSize;
    // 13x7 src, direct (H,W) = (4,5) config: Dst is 5x4, uncropped, equal to
    // ResizeImagePIL(5,4) with uint8 rounding then normalized.
    procedure TestHeightWidthConfigResizesWithoutCrop;
    // shortest_edge config through the config overload equals the scalar
    // call bit for bit.
    procedure TestShortestEdgeConfigMatchesScalarCall;
    // square src: height/width (N,N) and shortest_edge N give identical Dst.
    procedure TestHeightWidthSquareImageMatchesShortestEdge;
    // A (W,H,4) volume saved as PNG is 8-bit RGBA (colour type 6) and reloads
    // with its alpha; a 3-channel volume reloads opaque.
    procedure TestSaveImageRGBAPngRoundTrip;
    // GaussianBlurImage of a centred hot pixel keeps its mass of 1, is
    // symmetric, peaks at 1/S^2 (S = the 1-D tap sum); a flat channel stays flat.
    procedure TestGaussianBlurImageHotPixel;
    // Radius 0 copies Source into Dest; in place it changes nothing.
    procedure TestGaussianBlurImageZeroRadiusIsIdentity;
    // A binary P6 written like the examples write it (larger than the FPC
    // reader buffer, with a comment), a P5 and a saved .ppm reload exactly.
    procedure TestLoadBinaryPPMAndPGM;
    // BuildOutpaintCanvas with asymmetric sides 2,1,0,3 and overlap 1: the
    // image at (2,1), clamp-to-edge border, mask band only on extended sides.
    procedure TestOutpaintCanvasPlacementEdgeFillAndMask;
    // Zero sides: the canvas is the image and the mask is all 0, whatever the
    // overlap; an overlap wider than the image masks all of it.
    procedure TestOutpaintCanvasZeroSidesAndWideOverlap;
    // PasteOutpaintOriginal writes the image's channels where the mask is 0
    // and nothing else; a result of another size is refused.
    procedure TestPasteOutpaintOriginal;
    // GrowOutpaintSidesToMultiple and ParseOutpaintSides on valid and invalid
    // input.
    procedure TestOutpaintSidesGrowAndParse;
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

procedure TTestNeuralImagePreprocess.TestReadConfigHeightWidthSize;
var
  Cfg: TNNetImagePreprocess;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_hw.json';
  try
    WriteConfig(FN, '{"size":{"height":6,"width":10}}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertFalse('direct mode', Cfg.ResizeKeepsAspect);
    AssertEquals('resize H', 6, Cfg.ResizeHeight);
    AssertEquals('resize W', 10, Cfg.ResizeWidth);
    AssertEquals('crop H = resize H', 6, Cfg.CropHeight);
    AssertEquals('crop W = resize W', 10, Cfg.CropWidth);

    WriteConfig(FN, '{"size":{"height":256,"width":256},' +
      '"crop_size":{"height":224,"width":200},"do_center_crop":true}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertFalse('direct mode with crop', Cfg.ResizeKeepsAspect);
    AssertEquals('crop H', 224, Cfg.CropHeight);
    AssertEquals('crop W', 200, Cfg.CropWidth);

    WriteConfig(FN, '{"size":{"height":6,"width":10},' +
      '"crop_size":{"height":4,"width":4},"do_center_crop":false}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertEquals('no center crop H', 6, Cfg.CropHeight);
    AssertEquals('no center crop W', 10, Cfg.CropWidth);

    WriteConfig(FN, '{"size":{"shortest_edge":8},"crop_size":5}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertTrue('shortest_edge keeps aspect', Cfg.ResizeKeepsAspect);
    AssertEquals('shorter', 8, Cfg.ResizeShorterSide);
    AssertEquals('crop square H', 5, Cfg.CropHeight);
    AssertEquals('crop square W', 5, Cfg.CropWidth);
  finally
    DeleteFile(FN);
  end;
end;

procedure FillTestPhoto(Img: TNNetVolume);
var
  X, Y: integer;
begin
  for X := 0 to Img.SizeX - 1 do
    for Y := 0 to Img.SizeY - 1 do
    begin
      Img[X, Y, 0] := (X * 35 + Y * 16) mod 256;
      Img[X, Y, 1] := (X * 37 + Y * 91) mod 256;
      Img[X, Y, 2] := 255 * ((X + Y) mod 2);
    end;
end;

procedure TTestNeuralImagePreprocess.TestHeightWidthConfigResizesWithoutCrop;
var
  Cfg: TNNetImagePreprocess;
  Src, Dst, Resized: TNNetVolume;
  FN: string;
  X, Y, C: integer;
  Expected: TNeuralFloat;
begin
  FN := GetTempDir + 'cai_pp_hw_resize.json';
  WriteConfig(FN, '{"size":{"height":4,"width":5},"resample":3,' +
    '"image_mean":[0.5,0.4,0.3],"image_std":[0.2,0.25,0.3]}');
  Src := TNNetVolume.Create(13, 7, 3);
  Dst := TNNetVolume.Create;
  Resized := TNNetVolume.Create;
  try
    Cfg := ReadImagePreprocessConfig(FN);
    FillTestPhoto(Src);
    PreprocessImageForVisionModel(Src, Dst, Cfg);
    ResizeImagePIL(Src, Resized, 5, 4, irfBicubic, {RoundToBytes=}true);
    AssertEquals('out W', 5, Dst.SizeX);
    AssertEquals('out H', 4, Dst.SizeY);
    AssertEquals('out D', 3, Dst.Depth);
    for Y := 0 to 3 do
      for X := 0 to 4 do
        for C := 0 to 2 do
        begin
          Expected := (Resized[X, Y, C] / 255.0 - Cfg.Mean[C]) / Cfg.Std[C];
          AssertEquals(Format('direct (%d,%d,%d)', [X, Y, C]),
            Expected, Dst[X, Y, C], csOracleTol);
        end;
  finally
    DeleteFile(FN);
    Resized.Free;
    Dst.Free;
    Src.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestShortestEdgeConfigMatchesScalarCall;
var
  Cfg: TNNetImagePreprocess;
  Src, Dst, ScalarDst: TNNetVolume;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_se.json';
  WriteConfig(FN, '{"size":{"shortest_edge":4},"crop_size":3,' +
    '"resample":2,"image_mean":[0.5,0.4,0.3],"image_std":[0.2,0.25,0.3]}');
  Src := TNNetVolume.Create(10, 7, 3);
  Dst := TNNetVolume.Create;
  ScalarDst := TNNetVolume.Create;
  try
    Cfg := ReadImagePreprocessConfig(FN);
    FillTestPhoto(Src);
    PreprocessImageForVisionModel(Src, Dst, Cfg);
    PreprocessImageForVisionModel(Src, ScalarDst, 4, 3, Cfg.Mean, Cfg.Std,
      irfBilinear);
    AssertEquals('out W', 3, Dst.SizeX);
    AssertEquals('out H', 3, Dst.SizeY);
    AssertEquals('bit identical', 0.0, Dst.SumDiff(ScalarDst), 0.0);
  finally
    DeleteFile(FN);
    ScalarDst.Free;
    Dst.Free;
    Src.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestHeightWidthSquareImageMatchesShortestEdge;
var
  Cfg: TNNetImagePreprocess;
  Src, Dst, ShortestEdgeDst: TNNetVolume;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_sq.json';
  Src := TNNetVolume.Create(9, 9, 3);
  Dst := TNNetVolume.Create;
  ShortestEdgeDst := TNNetVolume.Create;
  try
    FillTestPhoto(Src);
    WriteConfig(FN, '{"size":{"height":4,"width":4},"resample":3}');
    Cfg := ReadImagePreprocessConfig(FN);
    PreprocessImageForVisionModel(Src, Dst, Cfg);
    WriteConfig(FN, '{"size":{"shortest_edge":4},"resample":3}');
    Cfg := ReadImagePreprocessConfig(FN);
    PreprocessImageForVisionModel(Src, ShortestEdgeDst, Cfg);
    AssertEquals('out W', 4, Dst.SizeX);
    AssertEquals('bit identical', 0.0, Dst.SumDiff(ShortestEdgeDst), 0.0);
  finally
    DeleteFile(FN);
    ShortestEdgeDst.Free;
    Dst.Free;
    Src.Free;
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

procedure TTestNeuralImagePreprocess.TestLoadBinaryPPMAndPGM;
const
  ImgW = 40;
  ImgH = 30;
var
  FN: string;
  F: TFileStream;
  Header: AnsiString;
  Raster: array of byte;
  Src, Loaded: TNNetVolume;
  X, Y, C: integer;
begin
  FN := GetTempDir + 'cai_p6_roundtrip.ppm';
  Src := TNNetVolume.Create(ImgW, ImgH, 3);
  Loaded := TNNetVolume.Create;
  try
    SetLength(Raster, ImgW * ImgH * 3);
    for Y := 0 to ImgH - 1 do
      for X := 0 to ImgW - 1 do
        for C := 0 to 2 do
        begin
          Src[X, Y, C] := (X * 7 + Y * 13 + C * 101) mod 256;
          Raster[(Y * ImgW + X) * 3 + C] := Round(Src[X, Y, C]);
        end;
    Header := 'P6'#10'# made by a test'#10 + IntToStr(ImgW) + ' ' + IntToStr(ImgH) +
      #10'255'#10;
    F := TFileStream.Create(FN, fmCreate);
    try
      F.WriteBuffer(Header[1], Length(Header));
      F.WriteBuffer(Raster[0], Length(Raster));
    finally
      F.Free;
    end;
    AssertTrue('P6 loads', LoadImageFromFileIntoVolume(FN, Loaded));
    AssertEquals('P6 width', ImgW, Loaded.SizeX);
    AssertEquals('P6 height', ImgH, Loaded.SizeY);
    AssertEquals('P6 depth', 3, Loaded.Depth);
    for Y := 0 to ImgH - 1 do
      for X := 0 to ImgW - 1 do
        for C := 0 to 2 do
          AssertEquals('P6 pixel ' + IntToStr(X) + ',' + IntToStr(Y) + ' c' +
            IntToStr(C), Src[X, Y, C], Loaded[X, Y, C], 0);
    // The repo's own .ppm saver round-trips through the same loader.
    AssertTrue('saved ppm', SaveImageFromVolumeIntoFile(Src, FN));
    AssertTrue('saved ppm loads', LoadImageFromFileIntoVolume(FN, Loaded));
    AssertEquals('saved ppm pixel', Src[ImgW - 1, ImgH - 1, 2],
      Loaded[ImgW - 1, ImgH - 1, 2], 0);
    AssertEquals('saved ppm pixel 2', Src[3, 5, 1], Loaded[3, 5, 1], 0);
    DeleteFile(FN);
    // Binary grayscale: every channel carries the gray byte.
    FN := GetTempDir + 'cai_p5_roundtrip.pgm';
    Header := 'P5 3 2 255'#10;
    F := TFileStream.Create(FN, fmCreate);
    try
      F.WriteBuffer(Header[1], Length(Header));
      F.WriteBuffer(Raster[0], 6);
    finally
      F.Free;
    end;
    AssertTrue('P5 loads', LoadImageFromFileIntoVolume(FN, Loaded));
    AssertEquals('P5 width', 3, Loaded.SizeX);
    AssertEquals('P5 height', 2, Loaded.SizeY);
    for X := 0 to 5 do
      for C := 0 to 2 do
        AssertEquals('P5 pixel ' + IntToStr(X) + ' c' + IntToStr(C),
          Raster[X], Loaded[X mod 3, X div 3, C], 0);
  finally
    DeleteFile(FN);
    Loaded.Free;
    Src.Free;
  end;
end;

// Image (SizeX, SizeY, Depth) with value X*100 + Y*10 + Channel.
procedure FillOutpaintImage(Image: TNNetVolume; SizeX, SizeY, Depth: integer);
var
  X, Y, Channel: integer;
begin
  Image.ReSize(SizeX, SizeY, Depth);
  for Y := 0 to SizeY - 1 do
    for X := 0 to SizeX - 1 do
      for Channel := 0 to Depth - 1 do
        Image[X, Y, Channel] := X * 100 + Y * 10 + Channel;
end;

procedure TTestNeuralImagePreprocess.TestOutpaintCanvasPlacementEdgeFillAndMask;
var
  Image, Canvas, Mask: TNNetVolume;
  X, Y, Channel: integer;
  IsKept: boolean;
begin
  Image := TNNetVolume.Create;
  Canvas := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
  try
    FillOutpaintImage(Image, 4, 5, 3);
    BuildOutpaintCanvas(Image, {Left=}2, {Top=}1, {Right=}0, {Bottom=}3,
      {Overlap=}1, Canvas, Mask);
    AssertEquals('canvas width', 6, Canvas.SizeX);
    AssertEquals('canvas height', 9, Canvas.SizeY);
    AssertEquals('canvas depth', 3, Canvas.Depth);
    AssertEquals('mask width', 6, Mask.SizeX);
    AssertEquals('mask height', 9, Mask.SizeY);
    AssertEquals('mask depth', 1, Mask.Depth);
    for Y := 0 to 8 do
      for X := 0 to 5 do
      begin
        for Channel := 0 to 2 do
          AssertEquals(Format('canvas (%d,%d,%d)', [X, Y, Channel]),
            Image[Min(Max(X - 2, 0), 3), Min(Max(Y - 1, 0), 4), Channel],
            Canvas[X, Y, Channel], 0);
        // Left and top bands start one pixel in; the right side is not
        // extended, so the image keeps its last column.
        IsKept := (X >= 3) and (X <= 5) and (Y >= 2) and (Y <= 4);
        if IsKept then
          AssertEquals(Format('mask kept (%d,%d)', [X, Y]), 0, Mask[X, Y, 0], 0)
        else
          AssertEquals(Format('mask hole (%d,%d)', [X, Y]), 1, Mask[X, Y, 0], 0);
      end;
  finally
    Mask.Free;
    Canvas.Free;
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestOutpaintCanvasZeroSidesAndWideOverlap;
var
  Image, Canvas, Mask: TNNetVolume;
  ElementPos: integer;
begin
  Image := TNNetVolume.Create;
  Canvas := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
  try
    FillOutpaintImage(Image, 5, 3, 4);
    BuildOutpaintCanvas(Image, 0, 0, 0, 0, {Overlap=}2, Canvas, Mask);
    AssertEquals('canvas width', 5, Canvas.SizeX);
    AssertEquals('canvas height', 3, Canvas.SizeY);
    AssertEquals('canvas depth', 4, Canvas.Depth);
    for ElementPos := 0 to Image.Size - 1 do
      AssertEquals('canvas element ' + IntToStr(ElementPos),
        Image.FData[ElementPos], Canvas.FData[ElementPos], 0);
    for ElementPos := 0 to Mask.Size - 1 do
      AssertEquals('mask element ' + IntToStr(ElementPos), 0,
        Mask.FData[ElementPos], 0);
    BuildOutpaintCanvas(Image, 1, 1, 1, 1, {Overlap=}2, Canvas, Mask);
    AssertEquals('wide-overlap mask size', 7 * 5, Mask.Size);
    for ElementPos := 0 to Mask.Size - 1 do
      AssertEquals('wide-overlap mask element ' + IntToStr(ElementPos), 1,
        Mask.FData[ElementPos], 0);
  finally
    Mask.Free;
    Canvas.Free;
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestPasteOutpaintOriginal;
var
  Image, Canvas, Mask, Dest: TNNetVolume;
  X, Y, Channel: integer;
  IsRefused: boolean;
begin
  Image := TNNetVolume.Create;
  Canvas := TNNetVolume.Create;
  Mask := TNNetVolume.Create;
  Dest := TNNetVolume.Create;
  try
    FillOutpaintImage(Image, 6, 4, 3);
    BuildOutpaintCanvas(Image, 3, 0, 2, 1, {Overlap=}2, Canvas, Mask);
    Dest.ReSize(Canvas.SizeX, Canvas.SizeY, 4);
    Dest.Fill(-1);
    PasteOutpaintOriginal(Image, 3, 0, 2, 1, 2, Dest);
    for Y := 0 to Dest.SizeY - 1 do
      for X := 0 to Dest.SizeX - 1 do
        for Channel := 0 to 3 do
          if (Mask[X, Y, 0] = 0) and (Channel < 3) then
            AssertEquals(Format('pasted (%d,%d,%d)', [X, Y, Channel]),
              Image[X - 3, Y, Channel], Dest[X, Y, Channel], 0)
          else
            AssertEquals(Format('untouched (%d,%d,%d)', [X, Y, Channel]), -1,
              Dest[X, Y, Channel], 0);
    Dest.ReSize(Canvas.SizeX + 1, Canvas.SizeY, 3);
    IsRefused := false;
    try
      PasteOutpaintOriginal(Image, 3, 0, 2, 1, 2, Dest);
    except
      on Exception do IsRefused := true;
    end;
    AssertTrue('a result of another size is refused', IsRefused);
  finally
    Dest.Free;
    Mask.Free;
    Canvas.Free;
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestOutpaintSidesGrowAndParse;
var
  Left, Top, Right, Bottom: integer;
begin
  // 40 + 8 = 48 -> 64: only Left is extended, so Left takes the 16; 24 -> 32
  // with no vertical extension: Bottom takes the 8.
  Left := 8; Top := 0; Right := 0; Bottom := 0;
  GrowOutpaintSidesToMultiple(40, 24, 32, Left, Top, Right, Bottom);
  AssertEquals('left grows', 24, Left);
  AssertEquals('right stays', 0, Right);
  AssertEquals('top stays', 0, Top);
  AssertEquals('bottom grows', 8, Bottom);
  Left := 4; Top := 3; Right := 5; Bottom := 0;
  GrowOutpaintSidesToMultiple(23, 29, 32, Left, Top, Right, Bottom);
  // 23 + 9 = 32 and 29 + 3 = 32 are multiples already: nothing grows.
  AssertEquals('left already a multiple', 4, Left);
  AssertEquals('right already a multiple', 5, Right);
  AssertEquals('top already a multiple', 3, Top);
  AssertEquals('bottom already a multiple', 0, Bottom);

  AssertEquals('valid', '', ParseOutpaintSides(' 1, 2,3 ,4', Left, Top, Right,
    Bottom));
  AssertEquals('left', 1, Left);
  AssertEquals('top', 2, Top);
  AssertEquals('right', 3, Right);
  AssertEquals('bottom', 4, Bottom);
  AssertTrue('three values', ParseOutpaintSides('1,2,3', Left, Top, Right,
    Bottom) <> '');
  AssertTrue('five values', ParseOutpaintSides('1,2,3,4,5', Left, Top, Right,
    Bottom) <> '');
  AssertTrue('all zero', ParseOutpaintSides('0,0,0,0', Left, Top, Right,
    Bottom) <> '');
  AssertTrue('negative', ParseOutpaintSides('-1,0,4,0', Left, Top, Right,
    Bottom) <> '');
  AssertTrue('not numbers', ParseOutpaintSides('a,b,c,d', Left, Top, Right,
    Bottom) <> '');
end;

initialization
  RegisterTest(TTestNeuralImagePreprocess);
end.
