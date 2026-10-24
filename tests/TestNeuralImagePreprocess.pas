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
    procedure TestSaveGray16PngRoundTrip;
    procedure TestLoadImage16BitFullPrecision;
    // GaussianBlurImage of a centred hot pixel keeps its mass of 1, is
    // symmetric, peaks at 1/S^2 (S = the 1-D tap sum); a flat channel stays flat.
    procedure TestGaussianBlurImageHotPixel;
    // Radius 0 copies Source into Dest; in place it changes nothing.
    procedure TestGaussianBlurImageZeroRadiusIsIdentity;
    // The (Sigma, KernelSize) overload: a delta yields torchvision's 9 taps of
    // sigma 1; a 6x5x2 image matches diffusers gaussian_blur_2d (reflect).
    procedure TestGaussianBlurImageTorchvisionKernel;
    // A binary P6 written like the examples write it (larger than the FPC
    // reader buffer, with a comment), a P5 and a saved .ppm reload exactly.
    procedure TestLoadBinaryPPMAndPGM;
    // ListImageFilesInFolder keeps PNG / JPEG / PPM files of any case, no
    // directories, and sorts digit runs by value, leading zeros ignored.
    procedure TestListImageFilesInFolder;
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
    // GetMaskCropRegion vs hand-traced diffusers get_crop_region: padding,
    // growth to the aspect, the shift off the bottom, growth past the image.
    procedure TestMaskCropRegion;
    // RGB mask read as PIL 'L' (luma 0 not masked); an empty mask is refused.
    procedure TestMaskCropRegionLumaAndEmptyMask;
    // ResizeImageAndFill / ResizeImageAndCrop at identity scale: placement,
    // the edge-repeated fill rows/columns and the centred crop offsets.
    procedure TestResizeImageAndFillAndCrop;
    // PasteInpaintCrop: the crop blended by mask / 255, the original elsewhere,
    // RGBA alpha kept, Dest aliasing Generated; bad crops and masks refused.
    procedure TestPasteInpaintCrop;
    // HFImageProcessorFamily strips Fast/Pil/ImageProcessor/FeatureExtractor;
    // ViT/DeiT read an int size as a square, CLIP/ConvNext/unknown do not.
    procedure TestHFProcessorFamilyAndIntSizeSquare;
    // int size: (size,size) for ViT/DeiT/BEiT types, shortest_edge for CLIP
    // and for configs with no or an unknown processor type.
    procedure TestReadConfigIntSizeByProcessorType;
    // crop_pct: ConvNext below/at 384 and its default, PoolFormer scaling and
    // its 224 crop default; ignored by other processors.
    procedure TestReadConfigCropPct;
    // 13x7 photo through ConvNext (size 7, crop_pct 0.875) and PoolFormer
    // (size 9, crop_pct 0.9) configs vs transformers' PIL processors.
    procedure TestCropPctPreprocessMatchesTransformers;
    // 13x7 photo through ViT int size 5 and DeiT size 6 / crop 4 configs vs
    // transformers' PIL processors.
    procedure TestIntSizeSquarePreprocessMatchesTransformers;
    // CannyEdges of a vertical step: one-pixel edge on the dark side, the
    // strict > thresholds, swapped thresholds, L2 gradient, RGB input.
    procedure TestCannyEdgesStepEdge;
    // CannyEdges of a filled square: hand-computed Sobel + 4-direction NMS.
    procedure TestCannyEdgesSquare;
    // Hysteresis keeps a weak edge 8-connected to a strong one and drops a
    // weak edge with no strong pixel.
    procedure TestCannyEdgesHysteresis;
    // A known rotation + scale + shift is recovered from 5 points; the inverse
    // round-trips; with clean points the LMedS variant equals least squares.
    procedure TestSimilarityTransformRecoversKnown;
    // 5 face landmarks onto the FFHQ template: cv2.estimateAffinePartial2D
    // (LMEDS) numbers, also with one landmark moved far enough to be dropped.
    procedure TestSimilarityTransformMatchesCV2;
    // WarpImageAffine of a 7x5 RGB image with a gray border against a float
    // cv2.warpAffine oracle (cv2 itself rounds coordinates to 1/32 pixel).
    procedure TestWarpImageAffineMatchesFloatOracle;
    // cv2.resize INTER_LANCZOS4 of a float32 5x4 image to 10x8, 15x12 and 3x3.
    procedure TestResizeImageLanczos4CV2MatchesCV2;
    // cv2.erode (0 = 3x3, 2, 4) and cv2.GaussianBlur(k, 0) (3, 9, 11) of a
    // 9x7 float mask.
    procedure TestErodeAndGaussianBlurCV2MatchCV2;
    // DrawBitmapText of 'A' (3 channels), '1' at scale 2, lower case as upper
    // case and an unknown character as '?' against hand-drawn 5x7 patterns.
    procedure TestBitmapTextGlyphPixels;
    // A UTF-8 multi-byte character is one '?' glyph wide; & [ { < ] } > " ` ; | \
    // draw as their look-alikes + ( ( ( ) ) ) ' ' : / /.
    procedure TestBitmapTextUTF8AndLookAlikes;
    // BitmapTextWidth / the DrawBitmapText result: (6 * Length - 1) * Scale.
    procedure TestBitmapTextWidth;
    // Text and FillRectangleCV2 across or far past the image border are
    // clipped pixel-exactly without exceptions; a wrong Color length raises.
    procedure TestBitmapTextClipsAtBorder;
    // DrawRectangleOutline: swapped corners, a 2 px ring inwards, border clipping.
    procedure TestDrawRectangleOutline;
    // ConvexHullOfPoints of a square with interior, edge and repeated points;
    // collinear, single and empty point sets.
    procedure TestConvexHullOfKnownPoints;
    // MinAreaRectOfConvexHull of a 20 x 10 rectangle, its DB unclip, a 45 degree
    // lattice rectangle and a vertical bar.
    procedure TestMinAreaRectAndUnclip;
    // CropRotatedRect of a linear image (exact bilinear) and of a 45 degree bar.
    procedure TestCropRotatedRect;
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

procedure TTestNeuralImagePreprocess.TestSaveGray16PngRoundTrip;
const
  cValues: array[0..5] of TNeuralFloat = (0, 1, 0.5, 0.123456, 1.5, -0.2);
  cExpected: array[0..5] of word = (0, 65535, 32768, 8091, 65535, 0);
var
  Src: TNNetVolume;
  Img: TFPMemoryImage;
  FN: string;
  PngHeader: TFileStream;
  HeaderBytes: array[0..25] of byte;
  X: integer;
begin
  FN := GetTempDir + 'cai_gray16_roundtrip.png';
  Src := TNNetVolume.Create(6, 1, 1);
  Img := TFPMemoryImage.Create(1, 1);
  try
    for X := 0 to 5 do Src[X, 0, 0] := cValues[X];
    AssertTrue('saved', SaveGray16PNGFromVolume(Src, FN));
    PngHeader := TFileStream.Create(FN, fmOpenRead);
    try
      PngHeader.ReadBuffer(HeaderBytes, SizeOf(HeaderBytes));
    finally
      PngHeader.Free;
    end;
    AssertEquals('IHDR bit depth', 16, HeaderBytes[24]);
    AssertEquals('IHDR colour type 0 = gray', 0, HeaderBytes[25]);
    AssertTrue('reloaded', Img.LoadFromFile(FN));
    AssertEquals('width', 6, Img.Width);
    for X := 0 to 5 do
      AssertEquals('gray ' + IntToStr(X), cExpected[X], Img.Colors[X, 0].red);
  finally
    DeleteFile(FN);
    Img.Free;
    Src.Free;
  end;
end;

// 16-bit RGB / gray PNGs whose samples differ only in the low byte load as
// sample / 257 with pFullPrecision; 8-bit PNG and JPEG load bit-identically.
procedure TTestNeuralImagePreprocess.TestLoadImage16BitFullPrecision;
const
  cWords: array[0..3] of word = ($8001, $8002, $80FE, $1234);
var
  Src, Loaded, Plain: TNNetVolume;
  FN, Ext: string;
  X, C, ExtCnt: integer;
begin
  FN := GetTempDir + 'cai_rgb16_roundtrip.png';
  Src := TNNetVolume.Create(4, 1, 3);
  Loaded := TNNetVolume.Create;
  Plain := TNNetVolume.Create;
  try
    for X := 0 to 3 do
      for C := 0 to 2 do Src[X, 0, C] := cWords[(X + C) mod 4] / 65535;
    AssertTrue('saved rgb16', SavePNG16FromVolume(Src, FN, {pGrayScale=}false));
    AssertTrue('loaded full', LoadImageFromFileIntoVolume(FN, Loaded, false, true));
    AssertTrue('loaded plain', LoadImageFromFileIntoVolume(FN, Plain, false));
    AssertEquals('depth', 3, Loaded.Depth);
    for X := 0 to 3 do
      for C := 0 to 2 do
      begin
        AssertEquals('full ' + IntToStr(X) + ',' + IntToStr(C),
          cWords[(X + C) mod 4] / 257, Loaded[X, 0, C], 1e-4);
        AssertEquals('plain keeps the high byte', cWords[(X + C) mod 4] shr 8,
          Plain[X, 0, C], 0);
      end;
    AssertTrue('low byte kept', Loaded[0, 0, 0] <> Loaded[1, 0, 0]);
    AssertTrue('saved gray16', SavePNG16FromVolume(Src, FN, {pGrayScale=}true));
    AssertTrue('loaded gray', LoadImageFromFileIntoVolume(FN, Loaded, false, true));
    for X := 0 to 3 do
      for C := 0 to 2 do
        AssertEquals('gray ' + IntToStr(X), cWords[X] / 257, Loaded[X, 0, C], 1e-4);
    DeleteFile(FN);
    for X := 0 to 3 do
      for C := 0 to 2 do Src[X, 0, C] := (X * 61 + C * 97) mod 256;
    for ExtCnt := 0 to 1 do
    begin
      if ExtCnt = 0 then Ext := '.png' else Ext := '.jpg';
      FN := GetTempDir + 'cai_rgb8_fullprecision' + Ext;
      AssertTrue('saved 8-bit' + Ext, SaveImageFromVolumeIntoFile(Src, FN));
      AssertTrue('loaded full' + Ext,
        LoadImageFromFileIntoVolume(FN, Loaded, false, true));
      AssertTrue('loaded plain' + Ext, LoadImageFromFileIntoVolume(FN, Plain));
      AssertEquals('8-bit' + Ext + ' bit identical', 0.0, Loaded.SumDiff(Plain), 0);
      DeleteFile(FN);
    end;
  finally
    DeleteFile(FN);
    Plain.Free;
    Loaded.Free;
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

procedure TTestNeuralImagePreprocess.TestGaussianBlurImageTorchvisionKernel;
const
  // torch.exp(-0.5 * torch.linspace(-4, 4, 9) ** 2), normalized (float32).
  csTorchTaps: array[0..8] of double = (0.0001338306174147874,
    0.004431861452758312, 0.05399112403392792, 0.2419714331626892,
    0.3989434540271759, 0.2419714331626892, 0.05399112403392792,
    0.004431861452758312, 0.0001338306174147874);
  // diffusers gaussian_blur_2d(x, 9, 1.0) of x[c, y, x] = (c*30 + y*6 + x)^1.5
  // / 10 on a 2x5x6 (C, H, W) image, float64, in CHW order.
  csDiffusersBlur: array[0..59] of double = (1.447148256, 1.576111393,
    1.87754503, 2.242229786, 2.593834433, 2.758309889, 2.428190341,
    2.585578317, 2.947038492, 3.373757746, 3.777771359, 3.965219622,
    4.912830307, 5.123746565, 5.599476091, 6.146241864, 6.652994145,
    6.88574718, 7.864269332, 8.118602838, 8.688342659, 9.336000571,
    9.930558196, 10.202334236, 9.337348855, 9.609310279, 10.217478592,
    10.906803509, 11.537923906, 11.8260133, 20.898076909, 21.254616434,
    22.048343934, 22.941020797, 23.752320344, 24.121206355, 23.112787812,
    23.480883852, 24.300031977, 25.220694719, 26.05690036, 26.436981325,
    28.12371562, 28.516372423, 29.389563998, 30.369758291, 31.258974096,
    31.662889748, 33.370474497, 33.786589395, 34.711422688, 35.748550891,
    36.688505537, 37.115244857, 35.83300808, 36.259519912, 37.207248174,
    38.269630676, 39.232103049, 39.668974228);
var
  Image: TNNetVolume;
  X, Y, Channel: integer;
begin
  Image := TNNetVolume.Create(17, 1, 1);
  try
    Image.Fill(0);
    Image[8, 0, 0] := 1;
    // One row reflects onto itself, so the Y pass keeps the X taps.
    GaussianBlurImage(Image, Image, 1.0, 9);
    for X := 0 to 16 do
      if (X < 4) or (X > 12) then
        AssertEquals('no tap at ' + IntToStr(X), 0, Image[X, 0, 0], 0)
      else AssertEquals('tap ' + IntToStr(X - 4), csTorchTaps[X - 4],
        Image[X, 0, 0], 1e-7);
    Image.ReSize(6, 5, 2);
    for Channel := 0 to 1 do
      for Y := 0 to 4 do
        for X := 0 to 5 do
          Image[X, Y, Channel] := Power(Channel * 30 + Y * 6 + X, 1.5) / 10;
    GaussianBlurImage(Image, Image, 1.0, 9);
    for Channel := 0 to 1 do
      for Y := 0 to 4 do
        for X := 0 to 5 do
          AssertEquals('blur at ' + IntToStr(X) + ',' + IntToStr(Y) + ',' +
            IntToStr(Channel), csDiffusersBlur[Channel * 30 + Y * 6 + X],
            Image[X, Y, Channel], 2e-5);
  finally
    Image.Free;
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

procedure TTestNeuralImagePreprocess.TestMaskCropRegion;
var
  Mask: TNNetVolume;
  X1, Y1, X2, Y2: integer;

  procedure AssertRegion(const Name: string; ExpectedX1, ExpectedY1,
    ExpectedX2, ExpectedY2: integer);
  begin
    AssertEquals(Name + ' x1', ExpectedX1, X1);
    AssertEquals(Name + ' y1', ExpectedY1, Y1);
    AssertEquals(Name + ' x2', ExpectedX2, X2);
    AssertEquals(Name + ' y2', ExpectedY2, Y2);
  end;

  procedure SetMaskRect(SizeX, SizeY, FirstX, FirstY, LastX, LastY: integer);
  var
    X, Y: integer;
  begin
    Mask.ReSize(SizeX, SizeY, 1);
    Mask.Fill(0);
    for Y := FirstY to LastY do
      for X := FirstX to LastX do
        Mask[X, Y, 0] := 255;
  end;

begin
  Mask := TNNetVolume.Create;
  try
    // Box x 8..9, y 4..5 padded by 1: 4x4, already 1:1.
    SetMaskRect(20, 10, 8, 4, 9, 5);
    GetMaskCropRegion(Mask, 32, 32, 1, X1, Y1, X2, Y2);
    AssertRegion('square', 7, 3, 11, 7);
    // Box x 17..19, y 0 padded by 2: 5x3 grown to 6 wide at 2:1, x2 = 21
    // shifted back to 20.
    SetMaskRect(20, 10, 17, 0, 19, 0);
    GetMaskCropRegion(Mask, 64, 32, 2, X1, Y1, X2, Y2);
    AssertRegion('right edge', 14, 0, 20, 3);
    // Row 9, x 0..5: 6x1 grown to 6 high (7..13), shifted up to 4..10.
    SetMaskRect(10, 10, 0, 9, 5, 9);
    GetMaskCropRegion(Mask, 1, 1, 0, X1, Y1, X2, Y2);
    AssertRegion('bottom edge', 0, 4, 6, 10);
    // A full 10-wide row in a 10x4 mask cannot grow to 10 high: whole image.
    SetMaskRect(10, 4, 0, 0, 9, 0);
    GetMaskCropRegion(Mask, 1, 1, 0, X1, Y1, X2, Y2);
    AssertRegion('past the image', 0, 0, 10, 4);
  finally
    Mask.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestMaskCropRegionLumaAndEmptyMask;
var
  Mask: TNNetVolume;
  X1, Y1, X2, Y2: integer;
  IsRefused: boolean;
begin
  Mask := TNNetVolume.Create(6, 4, 3);
  try
    Mask.Fill(0);
    // PIL 'L' of (0,0,2) is (2*7471 + 32768) shr 16 = 0; of (0,0,5) it is 1.
    Mask[1, 1, 2] := 2;
    Mask[4, 2, 2] := 5;
    GetMaskCropRegion(Mask, 1, 1, 0, X1, Y1, X2, Y2);
    AssertEquals('x1', 4, X1);
    AssertEquals('y1', 2, Y1);
    AssertEquals('x2', 5, X2);
    AssertEquals('y2', 3, Y2);
    Mask[4, 2, 2] := 0;
    IsRefused := false;
    try
      GetMaskCropRegion(Mask, 1, 1, 0, X1, Y1, X2, Y2);
    except
      on Exception do IsRefused := true;
    end;
    AssertTrue('an all-zero mask is refused', IsRefused);
  finally
    Mask.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestResizeImageAndFillAndCrop;
var
  Source, Dest: TNNetVolume;
  X, Y, Channel: integer;
begin
  Source := TNNetVolume.Create;
  Dest := TNNetVolume.Create;
  try
    // 6x3 into 6x6: fits 6x3 at y = 3 - 1 = 2; rows 0..1 repeat row 0, row 5
    // repeats row 2.
    FillOutpaintImage(Source, 6, 3, 2);
    ResizeImageAndFill(Source, Dest, 6, 6, irfNearest);
    AssertEquals('fill rows width', 6, Dest.SizeX);
    AssertEquals('fill rows height', 6, Dest.SizeY);
    for Y := 0 to 5 do
      for X := 0 to 5 do
        for Channel := 0 to 1 do
          AssertEquals(Format('fill rows (%d,%d,%d)', [X, Y, Channel]),
            Source[X, Min(Max(Y - 2, 0), 2), Channel], Dest[X, Y, Channel], 0);
    // 2x4 into 4x4: fits 2x4 at x = 2 - 1 = 1; column 0 and 3 repeat edges.
    FillOutpaintImage(Source, 2, 4, 1);
    ResizeImageAndFill(Source, Dest, 4, 4, irfNearest);
    for Y := 0 to 3 do
      for X := 0 to 3 do
        AssertEquals(Format('fill columns (%d,%d)', [X, Y]),
          Source[Min(Max(X - 1, 0), 1), Y, 0], Dest[X, Y, 0], 0);
    // 5x3 into 3x3: covers at 5x3, crop x from 5 div 2 - 3 div 2 = 1.
    FillOutpaintImage(Source, 5, 3, 2);
    ResizeImageAndCrop(Source, Dest, 3, 3, irfNearest);
    AssertEquals('crop width', 3, Dest.SizeX);
    AssertEquals('crop height', 3, Dest.SizeY);
    for Y := 0 to 2 do
      for X := 0 to 2 do
        for Channel := 0 to 1 do
          AssertEquals(Format('crop wide (%d,%d,%d)', [X, Y, Channel]),
            Source[X + 1, Y, Channel], Dest[X, Y, Channel], 0);
    // 3x6 into 3x3: covers at 3x6, crop y from 6 div 2 - 3 div 2 = 2; Dest
    // is Source.
    FillOutpaintImage(Source, 3, 6, 1);
    Dest.ReSize(Source);
    Dest.Copy(Source);
    ResizeImageAndCrop(Dest, Dest, 3, 3, irfNearest);
    for Y := 0 to 2 do
      for X := 0 to 2 do
        AssertEquals(Format('crop tall (%d,%d)', [X, Y]),
          Source[X, Y + 2, 0], Dest[X, Y, 0], 0);
  finally
    Dest.Free;
    Source.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestPasteInpaintCrop;
var
  Original, Generated, Mask, Dest: TNNetVolume;
  X, Y, Channel: integer;
  Weight, Expected: TNeuralFloat;
  IsRefused: boolean;
begin
  Original := TNNetVolume.Create;
  Generated := TNNetVolume.Create(2, 2, 3);
  Mask := TNNetVolume.Create(5, 4, 1);
  Dest := TNNetVolume.Create;
  try
    FillOutpaintImage(Original, 5, 4, 4);
    Generated.Fill(250);
    Mask.Fill(0);
    Mask[2, 1, 0] := 255;
    Mask[3, 1, 0] := 128;
    Mask[3, 2, 0] := 51;
    // Outside the crop: never blended.
    Mask[0, 0, 0] := 200;
    PasteInpaintCrop(Generated, Original, Mask, 2, 1, 4, 3, irfNearest,
      Generated);
    AssertEquals('width', 5, Generated.SizeX);
    AssertEquals('height', 4, Generated.SizeY);
    AssertEquals('depth', 4, Generated.Depth);
    for Y := 0 to 3 do
      for X := 0 to 4 do
        for Channel := 0 to 3 do
        begin
          Weight := 0;
          if (X >= 2) and (Y >= 1) and (Y <= 2) and (Channel < 3) then
            Weight := Mask[X, Y, 0] / 255;
          Expected := Original[X, Y, Channel] +
            Weight * (250 - Original[X, Y, Channel]);
          AssertEquals(Format('(%d,%d,%d)', [X, Y, Channel]), Expected,
            Generated[X, Y, Channel], 1e-4);
        end;
    // (3,1): 310 + 128/255 * (250 - 310), hand-computed.
    AssertEquals('half blend', 279.882353, Generated[3, 1, 0], 1e-4);
    Generated.ReSize(2, 2, 3);
    Generated.Fill(250);
    // A Dest of the original's element count (10x4x2 = 5x4x4) takes its shape.
    Dest.ReSize(10, 4, 2);
    PasteInpaintCrop(Generated, Original, Mask, 2, 1, 4, 3, irfNearest, Dest);
    AssertEquals('shaped width', 5, Dest.SizeX);
    AssertEquals('shaped depth', 4, Dest.Depth);
    AssertEquals('shaped pixel', 279.882353, Dest[3, 1, 0], 1e-4);
    IsRefused := false;
    try
      PasteInpaintCrop(Generated, Original, Mask, 3, 1, 6, 3, irfNearest,
        Generated);
    except
      on Exception do IsRefused := true;
    end;
    AssertTrue('a crop past the image is refused', IsRefused);
    Mask.ReSize(4, 4, 1);
    IsRefused := false;
    try
      PasteInpaintCrop(Generated, Original, Mask, 2, 1, 4, 3, irfNearest,
        Generated);
    except
      on Exception do IsRefused := true;
    end;
    AssertTrue('a mask of another size is refused', IsRefused);
  finally
    Dest.Free;
    Mask.Free;
    Generated.Free;
    Original.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestHFProcessorFamilyAndIntSizeSquare;
begin
  AssertEquals('fast', 'ViT', HFImageProcessorFamily('ViTImageProcessorFast'));
  AssertEquals('pil', 'DeiT', HFImageProcessorFamily('DeiTImageProcessorPil'));
  AssertEquals('legacy', 'ConvNext',
    HFImageProcessorFamily('ConvNextFeatureExtractor'));
  AssertEquals('empty', '', HFImageProcessorFamily(''));
  AssertTrue('ViT square', HFImageProcessorIntSizeIsSquare('ViTImageProcessor'));
  AssertTrue('ViT legacy square',
    HFImageProcessorIntSizeIsSquare('ViTFeatureExtractor'));
  AssertTrue('Beit square',
    HFImageProcessorIntSizeIsSquare('BeitImageProcessorFast'));
  AssertFalse('CLIP shortest edge',
    HFImageProcessorIntSizeIsSquare('CLIPImageProcessor'));
  AssertFalse('ConvNext shortest edge',
    HFImageProcessorIntSizeIsSquare('ConvNextImageProcessor'));
  AssertFalse('SigLIP shortest edge',
    HFImageProcessorIntSizeIsSquare('SiglipImageProcessor'));
  AssertFalse('unknown', HFImageProcessorIntSizeIsSquare('FooImageProcessor'));
  AssertFalse('none', HFImageProcessorIntSizeIsSquare(''));
end;

procedure TTestNeuralImagePreprocess.TestReadConfigIntSizeByProcessorType;
var
  Cfg: TNNetImagePreprocess;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_type.json';
  try
    WriteConfig(FN, '{"image_processor_type":"ViTImageProcessor","size":224}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertFalse('ViT direct', Cfg.ResizeKeepsAspect);
    AssertEquals('ViT H', 224, Cfg.ResizeHeight);
    AssertEquals('ViT W', 224, Cfg.ResizeWidth);
    AssertEquals('ViT crop H', 224, Cfg.CropHeight);
    AssertEquals('ViT crop W', 224, Cfg.CropWidth);

    WriteConfig(FN, '{"feature_extractor_type":"ViTFeatureExtractor",' +
      '"size":32}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertFalse('legacy ViT direct', Cfg.ResizeKeepsAspect);
    AssertEquals('legacy ViT W', 32, Cfg.ResizeWidth);

    WriteConfig(FN, '{"feature_extractor_type":"DeiTFeatureExtractor",' +
      '"size":256,"crop_size":224,"do_center_crop":true}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertFalse('DeiT direct', Cfg.ResizeKeepsAspect);
    AssertEquals('DeiT H', 256, Cfg.ResizeHeight);
    AssertEquals('DeiT W', 256, Cfg.ResizeWidth);
    AssertEquals('DeiT crop H', 224, Cfg.CropHeight);
    AssertEquals('DeiT crop W', 224, Cfg.CropWidth);

    WriteConfig(FN, '{"image_processor_type":"BeitImageProcessor",' +
      '"size":224,"crop_size":200,"do_center_crop":false}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertEquals('BEiT uncropped', 224, Cfg.CropHeight);

    WriteConfig(FN, '{"image_processor_type":"CLIPImageProcessor",' +
      '"size":224,"crop_size":224}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertTrue('CLIP shortest edge', Cfg.ResizeKeepsAspect);
    AssertEquals('CLIP edge', 224, Cfg.ResizeShorterSide);

    WriteConfig(FN, '{"image_processor_type":"FooImageProcessor",' +
      '"size":256,"crop_size":224}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertTrue('unknown shortest edge', Cfg.ResizeKeepsAspect);
    AssertEquals('unknown edge', 256, Cfg.ResizeShorterSide);
    AssertEquals('unknown crop', 224, Cfg.CropSize);
  finally
    DeleteFile(FN);
  end;
end;

procedure TTestNeuralImagePreprocess.TestReadConfigCropPct;
var
  Cfg: TNNetImagePreprocess;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_croppct.json';
  try
    WriteConfig(FN, '{"feature_extractor_type":"ConvNextFeatureExtractor",' +
      '"size":224,"crop_pct":0.875}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertTrue('ConvNext 224 keeps aspect', Cfg.ResizeKeepsAspect);
    AssertEquals('ConvNext 224/0.875', 256, Cfg.ResizeShorterSide);
    AssertEquals('ConvNext crop H', 224, Cfg.CropHeight);
    AssertEquals('ConvNext crop W', 224, Cfg.CropWidth);
    AssertEquals('ConvNext crop', 224, Cfg.CropSize);

    WriteConfig(FN, '{"image_processor_type":"ConvNextImageProcessor",' +
      '"size":{"shortest_edge":224}}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertEquals('ConvNext default crop_pct', 256, Cfg.ResizeShorterSide);

    WriteConfig(FN, '{"image_processor_type":"ConvNextImageProcessor",' +
      '"size":{"shortest_edge":384},"crop_pct":0.875}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertFalse('ConvNext 384 warps', Cfg.ResizeKeepsAspect);
    AssertEquals('ConvNext 384 H', 384, Cfg.ResizeHeight);
    AssertEquals('ConvNext 384 W', 384, Cfg.ResizeWidth);
    AssertEquals('ConvNext 384 crop', 384, Cfg.CropHeight);

    WriteConfig(FN, '{"feature_extractor_type":"PoolFormerFeatureExtractor",' +
      '"size":224,"crop_pct":0.9}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertTrue('PoolFormer keeps aspect', Cfg.ResizeKeepsAspect);
    AssertEquals('PoolFormer 224/0.9', 248, Cfg.ResizeShorterSide);
    AssertEquals('PoolFormer crop', 224, Cfg.CropHeight);

    WriteConfig(FN, '{"image_processor_type":"PoolFormerImageProcessor",' +
      '"size":{"height":10,"width":20},"crop_pct":0.5,' +
      '"crop_size":{"height":8,"width":16}}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertFalse('PoolFormer (H,W) direct', Cfg.ResizeKeepsAspect);
    AssertEquals('PoolFormer H/0.5', 20, Cfg.ResizeHeight);
    AssertEquals('PoolFormer W/0.5', 40, Cfg.ResizeWidth);
    AssertEquals('PoolFormer crop W', 16, Cfg.CropWidth);

    WriteConfig(FN, '{"image_processor_type":"CLIPImageProcessor",' +
      '"size":224,"crop_pct":0.5}');
    Cfg := ReadImagePreprocessConfig(FN);
    AssertEquals('CLIP ignores crop_pct', 224, Cfg.ResizeShorterSide);
  finally
    DeleteFile(FN);
  end;
end;

// Dst against transformers' pixel_values: its size and sum, then
// (X, Y, C, Value) quadruples in Samples.
procedure AssertTransformersOracle(const Name: string; Dst: TNNetVolume;
  ExpectedWidth, ExpectedHeight: integer; ExpectedSum: double;
  const Samples: array of double);
var
  SampleIdx: integer;
begin
  TAssert.AssertEquals(Name + ' W', ExpectedWidth, Dst.SizeX);
  TAssert.AssertEquals(Name + ' H', ExpectedHeight, Dst.SizeY);
  TAssert.AssertEquals(Name + ' D', 3, Dst.Depth);
  TAssert.AssertEquals(Name + ' sum', ExpectedSum, Dst.GetSum, 1e-3);
  SampleIdx := 0;
  while SampleIdx + 3 <= High(Samples) do
  begin
    TAssert.AssertEquals(Format('%s (%g,%g,%g)', [Name, Samples[SampleIdx],
      Samples[SampleIdx + 1], Samples[SampleIdx + 2]]), Samples[SampleIdx + 3],
      Dst[Round(Samples[SampleIdx]), Round(Samples[SampleIdx + 1]),
      Round(Samples[SampleIdx + 2])], csOracleTol);
    Inc(SampleIdx, 4);
  end;
end;

procedure TTestNeuralImagePreprocess.TestCropPctPreprocessMatchesTransformers;
const
  csNorm = '"resample":3,"image_mean":[0.5,0.4,0.3],"image_std":[0.2,0.25,0.3]';
var
  Cfg: TNNetImagePreprocess;
  Src, Dst: TNNetVolume;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_croppct_px.json';
  Src := TNNetVolume.Create(13, 7, 3);
  Dst := TNNetVolume.Create;
  try
    FillTestPhoto(Src);
    WriteConfig(FN, '{"image_processor_type":"ConvNextImageProcessor",' +
      '"size":7,"crop_pct":0.875,' + csNorm + '}');
    Cfg := ReadImagePreprocessConfig(FN);
    PreprocessImageForVisionModel(Src, Dst, Cfg);
    // ConvNextImageProcessorPil(size=7, crop_pct=0.875, resample=3, ...).
    AssertTransformersOracle('ConvNext', Dst, 7, 7, 51.462094,
      [0, 0, 0, -0.617647,  6, 0, 1, -0.831373,  0, 6, 2, -0.267974,
       3, 3, 0, 0.656863,  1, 2, 1, 0.109804,  6, 6, 2, 1.366013,
       2, 1, 2, 0.228758,  5, 4, 0, -1.343137]);

    WriteConfig(FN, '{"image_processor_type":"PoolFormerImageProcessor",' +
      '"size":9,"crop_pct":0.9,"crop_size":9,' + csNorm + '}');
    Cfg := ReadImagePreprocessConfig(FN);
    PreprocessImageForVisionModel(Src, Dst, Cfg);
    // PoolFormerImageProcessorPil(size=9, crop_pct=0.9, crop_size=9, ...).
    AssertTransformersOracle('PoolFormer', Dst, 9, 9, 83.783664,
      [0, 0, 0, -0.637255,  8, 0, 1, -0.752941,  0, 8, 2, 0.437908,
       4, 4, 0, 0.009804,  1, 2, 1, 1.600000,  8, 8, 2, 0.633987,
       2, 1, 2, 0.921569,  7, 6, 0, -0.892157]);
  finally
    DeleteFile(FN);
    Dst.Free;
    Src.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestIntSizeSquarePreprocessMatchesTransformers;
const
  csNorm = '"resample":3,"image_mean":[0.5,0.4,0.3],"image_std":[0.2,0.25,0.3]';
var
  Cfg: TNNetImagePreprocess;
  Src, Dst: TNNetVolume;
  FN: string;
begin
  FN := GetTempDir + 'cai_pp_square_px.json';
  Src := TNNetVolume.Create(13, 7, 3);
  Dst := TNNetVolume.Create;
  try
    FillTestPhoto(Src);
    WriteConfig(FN, '{"image_processor_type":"ViTImageProcessor",' +
      '"size":5,' + csNorm + '}');
    Cfg := ReadImagePreprocessConfig(FN);
    PreprocessImageForVisionModel(Src, Dst, Cfg);
    // ViTImageProcessorPil(size=5, resample=3, ...).
    AssertTransformersOracle('ViT', Dst, 5, 5, 22.592812,
      [0, 0, 0, -1.833333,  4, 0, 1, 0.956863,  0, 4, 2, 0.607843,
       2, 2, 0, -0.637255,  1, 2, 1, 0.376471,  4, 4, 2, 0.607843,
       2, 1, 2, 0.673203,  3, 2, 0, -0.813725]);

    WriteConfig(FN, '{"feature_extractor_type":"DeiTFeatureExtractor",' +
      '"size":6,"crop_size":4,' + csNorm + '}');
    Cfg := ReadImagePreprocessConfig(FN);
    PreprocessImageForVisionModel(Src, Dst, Cfg);
    // DeiTImageProcessorPil(size=6, crop_size=4, resample=3, ...).
    AssertTransformersOracle('DeiT', Dst, 4, 4, 14.956864,
      [0, 0, 0, -0.264706,  3, 0, 1, 0.815686,  0, 3, 2, 0.660131,
       2, 2, 0, -1.754902,  1, 2, 1, 0.062745,  3, 3, 2, 0.660131,
       2, 1, 2, 0.660131,  2, 1, 0, -1.264706]);
  finally
    DeleteFile(FN);
    Dst.Free;
    Src.Free;
  end;
end;

// Asserts Edges (W,H,D) is 255 where Expected[Y][X] = '#' and 0 elsewhere.
procedure AssertCannyEdges(const Context: string; Edges: TNNetVolume;
  const Expected: array of string);
var
  X, Y, Channel, MaxX, MaxY, MaxChannel: integer;
  ExpectedValue: TNeuralFloat;
begin
  MaxY := High(Expected);
  TAssert.AssertEquals(Context + ' height', MaxY + 1, Edges.SizeY);
  MaxX := Length(Expected[0]) - 1;
  TAssert.AssertEquals(Context + ' width', MaxX + 1, Edges.SizeX);
  MaxChannel := Edges.Depth - 1;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      if Expected[Y][X + 1] = '#' then ExpectedValue := 255
      else ExpectedValue := 0;
      for Channel := 0 to MaxChannel do
        TAssert.AssertEquals(Context + ' at ' + IntToStr(X) + ',' + IntToStr(Y),
          ExpectedValue, Edges[X, Y, Channel], 0);
    end;
end;

procedure TTestNeuralImagePreprocess.TestCannyEdgesStepEdge;
const
  // Sobel gives |dx| = 400 at X = 3 and X = 4; NMS keeps X = 3 (> left,
  // >= right).
  csEdge: array[0..5] of string = ('...#....', '...#....', '...#....',
    '...#....', '...#....', '...#....');
  csNone: array[0..5] of string = ('........', '........', '........',
    '........', '........', '........');
var
  Image, Edges: TNNetVolume;
  X, Y: integer;
begin
  Image := TNNetVolume.Create(8, 6, 1);
  Edges := TNNetVolume.Create;
  try
    Image.Fill(0);
    for Y := 0 to 5 do
      for X := 4 to 7 do Image[X, Y, 0] := 100;
    CannyEdges(Image, Edges, 100, 200);
    AssertEquals('depth', 1, Edges.Depth);
    AssertCannyEdges('100/200', Edges, csEdge);
    CannyEdges(Image, Edges, 100, 399);
    AssertCannyEdges('high 399', Edges, csEdge);
    CannyEdges(Image, Edges, 100, 400);
    AssertCannyEdges('high 400 is not > 400', Edges, csNone);
    CannyEdges(Image, Edges, 399, 10);
    AssertCannyEdges('swapped thresholds', Edges, csEdge);
    CannyEdges(Image, Edges, 100, 399, true);
    AssertCannyEdges('L2 400^2 > 399^2', Edges, csEdge);
    CannyEdges(Image, Edges, 100, 400, true);
    AssertCannyEdges('L2 400^2 is not > 400^2', Edges, csNone);
    // RGB: the step only in the blue channel; the strongest channel wins.
    Image.ReSize(8, 6, 3);
    Image.Fill(0);
    for Y := 0 to 5 do
      for X := 4 to 7 do
      begin
        Image[X, Y, 0] := 5;
        Image[X, Y, 2] := 100;
      end;
    CannyEdges(Image, Edges, 100, 200, false, 3);
    AssertEquals('dest depth', 3, Edges.Depth);
    AssertCannyEdges('RGB', Edges, csEdge);
    // Dest may be Source.
    CannyEdges(Image, Image, 100, 200);
    AssertCannyEdges('in place', Image, csEdge);
  finally
    Edges.Free;
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestCannyEdgesSquare;
const
  // L1 magnitudes / 100 (rows mirror about 3.5):
  //   0 2 4 4 4 4 2 0 / 0 4 6 4 4 6 4 0 / 0 4 4 0 0 4 4 0
  // NMS ties keep the left / upper pixel of each pair, so the ring is not
  // mirror-symmetric, as in cv2.
  csEdges: array[0..7] of string = (
    '........',
    '...##...',
    '..#..#..',
    '.#...#..',
    '.#...#..',
    '..####..',
    '........',
    '........');
var
  Image, Edges: TNNetVolume;
  X, Y: integer;
begin
  Image := TNNetVolume.Create(8, 8, 1);
  Edges := TNNetVolume.Create;
  try
    Image.Fill(0);
    for Y := 2 to 5 do
      for X := 2 to 5 do Image[X, Y, 0] := 100;
    CannyEdges(Image, Edges, 100, 200);
    AssertCannyEdges('square', Edges, csEdges);
  finally
    Edges.Free;
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestCannyEdgesHysteresis;
const
  // Step at X = 4 of height 100 in rows 0..4 and 30 in rows 5..9: the upper
  // rows (400) and the horizontal step (280, 400, 540) are strong; rows 6..9
  // (120) are weak and join through (4,5).
  csLinked: array[0..9] of string = (
    '...#......',
    '...#......',
    '...#......',
    '...#......',
    '....######',
    '....#.....',
    '...#......',
    '...#......',
    '...#......',
    '...#......');
  csNone: array[0..5] of string = ('........', '........', '........',
    '........', '........', '........');
  csWeakAsStrong: array[0..5] of string = ('...#....', '...#....',
    '...#....', '...#....', '...#....', '...#....');
var
  Image, Edges: TNNetVolume;
  X, Y: integer;
begin
  Image := TNNetVolume.Create(10, 10, 1);
  Edges := TNNetVolume.Create;
  try
    Image.Fill(0);
    for Y := 0 to 9 do
      for X := 4 to 9 do
        if Y <= 4 then Image[X, Y, 0] := 100
        else Image[X, Y, 0] := 30;
    CannyEdges(Image, Edges, 100, 200);
    AssertCannyEdges('linked', Edges, csLinked);
    // No pixel above 540: nothing seeds the hysteresis.
    CannyEdges(Image, Edges, 100, 540);
    AssertEquals('no strong pixel', 0, Edges.GetSum(), 0);
    // An isolated weak step (|dx| = 120).
    Image.ReSize(8, 6, 1);
    Image.Fill(0);
    for Y := 0 to 5 do
      for X := 4 to 7 do Image[X, Y, 0] := 30;
    CannyEdges(Image, Edges, 100, 200);
    AssertCannyEdges('isolated weak', Edges, csNone);
    CannyEdges(Image, Edges, 100, 119);
    AssertCannyEdges('weak above high', Edges, csWeakAsStrong);
    CannyEdges(Image, Edges, 120, 200);
    AssertCannyEdges('120 is not > low 120', Edges, csNone);
  finally
    Edges.Free;
    Image.Free;
  end;
end;

// value(x, y, c) = (x*MulX + y*MulY + c*MulC) mod 256 on a W x H x Depth image.
procedure FillModuloImage(Image: TNNetVolume; SizeX, SizeY, Depth, MulX, MulY,
  MulC: integer);
var
  X, Y, Channel: integer;
begin
  Image.ReSize(SizeX, SizeY, Depth);
  for Y := 0 to SizeY - 1 do
    for X := 0 to SizeX - 1 do
      for Channel := 0 to Depth - 1 do
        Image[X, Y, Channel] := (X * MulX + Y * MulY + Channel * MulC) mod 256;
end;

procedure AssertAffineMatrix(const Msg: string; const Expected: array of double;
  const Actual: TImageAffineMatrix; Tolerance: double);
var
  ElementPos: integer;
begin
  for ElementPos := 0 to 5 do
    TAssert.AssertEquals(Msg + ' [' + IntToStr(ElementPos) + ']', Expected[ElementPos],
      Actual[ElementPos], Tolerance);
end;

procedure TTestNeuralImagePreprocess.TestSimilarityTransformRecoversKnown;
const
  csX: array[0..4] of double = (61.3, 93.7, 78.2, 64.9, 92.2);
  csY: array[0..4] of double = (72.8, 70.1, 91.5, 108.4, 106.6);
var
  Known, Estimated, Robust, RoundTrip, Inverse: TImageAffineMatrix;
  DestX, DestY: array[0..4] of double;
  PointPos: integer;
begin
  Known[0] := 1.3 * Cos(0.2); Known[1] := -1.3 * Sin(0.2); Known[2] := 5;
  Known[3] := 1.3 * Sin(0.2); Known[4] := 1.3 * Cos(0.2); Known[5] := -7;
  for PointPos := 0 to 4 do
  begin
    DestX[PointPos] := Known[0] * csX[PointPos] + Known[1] * csY[PointPos] + Known[2];
    DestY[PointPos] := Known[3] * csX[PointPos] + Known[4] * csY[PointPos] + Known[5];
  end;
  Estimated := EstimateSimilarityTransform(csX, csY, DestX, DestY);
  AssertAffineMatrix('least squares', Known, Estimated, 1e-9);
  Robust := EstimateSimilarityTransformLMedS(csX, csY, DestX, DestY);
  AssertAffineMatrix('LMedS', Known, Robust, 1e-9);
  Inverse := InvertImageAffineMatrix(Known);
  // Inverse after Known is the identity.
  RoundTrip[0] := Inverse[0] * Known[0] + Inverse[1] * Known[3];
  RoundTrip[1] := Inverse[0] * Known[1] + Inverse[1] * Known[4];
  RoundTrip[2] := Inverse[0] * Known[2] + Inverse[1] * Known[5] + Inverse[2];
  RoundTrip[3] := Inverse[3] * Known[0] + Inverse[4] * Known[3];
  RoundTrip[4] := Inverse[3] * Known[1] + Inverse[4] * Known[4];
  RoundTrip[5] := Inverse[3] * Known[2] + Inverse[4] * Known[5] + Inverse[5];
  AssertAffineMatrix('inverse', [1, 0, 0, 0, 1, 0], RoundTrip, 1e-12);
end;

procedure TTestNeuralImagePreprocess.TestSimilarityTransformMatchesCV2;
const
  csX: array[0..4] of double = (61.3, 93.7, 78.2, 64.9, 92.2);
  csY: array[0..4] of double = (72.8, 70.1, 91.5, 108.4, 106.6);
  csTemplateX: array[0..4] of double = (192.98138, 318.90277, 256.63416,
    201.26117, 313.08905);
  csTemplateY: array[0..4] of double = (239.94708, 240.1936, 314.01935,
    371.41043, 371.15118);
  // cv2 4.10 estimateAffinePartial2D(method=LMEDS)[0]; cv2 rounds the points
  // to float32, hence the 5e-5 tolerance.
  csCV2Clean: array[0..5] of double = (3.7661942007, -0.1656064972,
    -22.5306949058, 0.1656064972, 3.7661942007, -44.0884489685);
  // The nose moved by (+30, -25): cv2 drops it as an outlier.
  csCV2Outlier: array[0..5] of double = (3.7657039691, -0.1653803211,
    -22.4630491176, 0.1653803211, 3.7657039691, -44.1645881508);
  // numpy closed-form least squares of all 5 points (float64).
  csLeastSquaresClean: array[0..5] of double = (3.7661940602, -0.1656061338,
    -22.5307230344, 0.1656061338, 3.7661940602, -44.0884089307);
var
  OutlierX, OutlierY: array[0..4] of double;
  PointPos: integer;
begin
  AssertAffineMatrix('least squares = numpy', csLeastSquaresClean,
    EstimateSimilarityTransform(csX, csY, csTemplateX, csTemplateY), 1e-8);
  AssertAffineMatrix('LMedS clean = cv2', csCV2Clean,
    EstimateSimilarityTransformLMedS(csX, csY, csTemplateX, csTemplateY), 5e-5);
  for PointPos := 0 to 4 do
  begin
    OutlierX[PointPos] := csX[PointPos];
    OutlierY[PointPos] := csY[PointPos];
  end;
  OutlierX[2] := OutlierX[2] + 30;
  OutlierY[2] := OutlierY[2] - 25;
  AssertAffineMatrix('LMedS outlier = cv2', csCV2Outlier,
    EstimateSimilarityTransformLMedS(OutlierX, OutlierY, csTemplateX,
    csTemplateY), 5e-5);
  AssertTrue('plain least squares keeps the outlier',
    Abs(EstimateSimilarityTransform(OutlierX, OutlierY, csTemplateX,
    csTemplateY)[2] - csCV2Outlier[2]) > 10);
end;

procedure TTestNeuralImagePreprocess.TestWarpImageAffineMatchesFloatOracle;
const
  // numpy float64 cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT) semantics.
  csPixelX: array[0..5] of integer = (0, 3, 5, 7, 2, 6);
  csPixelY: array[0..5] of integer = (0, 2, 4, 6, 5, 1);
  csPixels: array[0..5, 0..2] of double = (
    (149.6306504611, 174.2673169921, 90.9922432448),
    (59.9845893993, 112.9845893993, 165.9845893993),
    (105.5284658581, 158.5284658581, 59.0110767035),
    (96.4228186634, 129.3196019307, 162.6029855198),
    (132, 133, 135),
    (236.0013977954, 39.6308037767, 92.603291955));
  csSum = 21975.829929980282;
var
  Image, Warped: TNNetVolume;
  Matrix: TImageAffineMatrix;
  PixelPos, Channel: integer;
begin
  Image := TNNetVolume.Create;
  Warped := TNNetVolume.Create;
  try
    FillModuloImage(Image, 7, 5, 3, 37, 91, 53);
    Matrix[0] := 1.3 * Cos(0.4); Matrix[1] := -1.3 * Sin(0.4); Matrix[2] := 1.7;
    Matrix[3] := 1.3 * Sin(0.4); Matrix[4] := 1.3 * Cos(0.4); Matrix[5] := -2.2;
    WarpImageAffine(Image, Warped, Matrix, 8, 7, [132, 133, 135]);
    AssertEquals('width', 8, Warped.SizeX);
    AssertEquals('height', 7, Warped.SizeY);
    AssertEquals('depth', 3, Warped.Depth);
    for PixelPos := 0 to 5 do
      for Channel := 0 to 2 do
        AssertEquals(Format('pixel (%d,%d,%d)', [csPixelX[PixelPos],
          csPixelY[PixelPos], Channel]), csPixels[PixelPos, Channel],
          Warped[csPixelX[PixelPos], csPixelY[PixelPos], Channel], 1e-4);
    // GetSum accumulates in single.
    AssertEquals('sum', csSum, Warped.GetSum(), csSum * 1e-6);
  finally
    Warped.Free;
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestResizeImageLanczos4CV2MatchesCV2;
const
  csSizes: array[0..2, 0..1] of integer = ((10, 8), (15, 12), (3, 3));
  csSums: array[0..2] of double = (28332.796250224113, 63760.29971283674,
    3156.4869422912598);
  // cv2 4.10 at (2, 1), at the last pixel and at (0, 0).
  csPixels: array[0..2, 0..2, 0..2] of double = (
    ((33.498577, 69.972664, 131.45172), (92.07607, 133.42291, 162.21442),
     (-8.501168, 38.410713, 82.78556)),
    ((7.293399, 50.2934, 93.2934), (95.13215, 135.21355, 162.35149),
     (-9.177448, 37.92198, 82.648476)),
    ((134.94568, 116.59711, 62.53772), (69.02842, 86.17246, 143.34836),
     (15.592037, 54.181152, 101.65164)));
var
  Image, Resized: TNNetVolume;
  SizePos, Channel, LastX, LastY: integer;
begin
  Image := TNNetVolume.Create;
  Resized := TNNetVolume.Create;
  try
    FillModuloImage(Image, 5, 4, 3, 29, 71, 43);
    for SizePos := 0 to 2 do
    begin
      ResizeImageLanczos4CV2(Image, Resized, csSizes[SizePos, 0],
        csSizes[SizePos, 1]);
      LastX := csSizes[SizePos, 0] - 1;
      LastY := csSizes[SizePos, 1] - 1;
      AssertEquals('width', csSizes[SizePos, 0], Resized.SizeX);
      AssertEquals('height', csSizes[SizePos, 1], Resized.SizeY);
      AssertEquals('sum ' + IntToStr(SizePos), csSums[SizePos],
        Resized.GetSum(), csSums[SizePos] * 1e-6);
      for Channel := 0 to 2 do
      begin
        AssertEquals('(2,1)', csPixels[SizePos, 0, Channel],
          Resized[2, 1, Channel], 1e-3);
        AssertEquals('last', csPixels[SizePos, 1, Channel],
          Resized[LastX, LastY, Channel], 1e-3);
        AssertEquals('(0,0)', csPixels[SizePos, 2, Channel],
          Resized[0, 0, Channel], 1e-3);
      end;
    end;
    ResizeImageLanczos4CV2(Image, Resized, 5, 4);
    AssertEquals('same size copies', 0, Resized.SumDiff(Image), 0);
  finally
    Resized.Free;
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestErodeAndGaussianBlurCV2MatchCV2;
const
  // cv2 4.10 sums and values at (4, 3), (8, 0), (0, 6).
  csErodeSizes: array[0..3] of integer = (0, 2, 4, 3);
  csErodeSums: array[0..3] of double = (18.03529462404549, 23.05098100192845,
    11.956863041967154, 18.03529462404549);
  csErodePixels: array[0..3, 0..2] of double = (
    (0.42745098, 0.46666667, 0.5686275), (0.42745098, 0.46666667, 0.5686275),
    (0.24705882, 0.4, 0.45490196), (0.42745098, 0.46666667, 0.5686275));
  // cv2 4.10 GaussianBlur sums and values at (4, 3), (0, 0), (8, 6).
  csBlurSizes: array[0..2] of integer = (3, 9, 11);
  csBlurSums: array[0..2] of double = (32.77255008369684, 33.63855728507042,
    33.78414648771286);
  csBlurPixels: array[0..2, 0..2] of double = (
    (0.60784316, 0.09019608, 0.121568635), (0.5794731, 0.23394608, 0.500674),
    (0.5666165, 0.27718127, 0.55454963));
var
  Mask, Filtered: TNNetVolume;
  SizePos: integer;
begin
  Mask := TNNetVolume.Create;
  Filtered := TNNetVolume.Create;
  try
    FillModuloImage(Mask, 9, 7, 1, 17, 29, 0);
    Mask.Divi(255);
    for SizePos := 0 to 3 do
    begin
      ErodeImage(Mask, Filtered, csErodeSizes[SizePos], csErodeSizes[SizePos]);
      AssertEquals('erode sum', csErodeSums[SizePos], Filtered.GetSum(), 1e-5);
      AssertEquals('erode (4,3)', csErodePixels[SizePos, 0], Filtered[4, 3, 0], 1e-6);
      AssertEquals('erode (8,0)', csErodePixels[SizePos, 1], Filtered[8, 0, 0], 1e-6);
      AssertEquals('erode (0,6)', csErodePixels[SizePos, 2], Filtered[0, 6, 0], 1e-6);
    end;
    for SizePos := 0 to 2 do
    begin
      GaussianBlurImageCV2(Mask, Filtered, csBlurSizes[SizePos]);
      AssertEquals('blur sum', csBlurSums[SizePos], Filtered.GetSum(), 1e-5);
      AssertEquals('blur (4,3)', csBlurPixels[SizePos, 0], Filtered[4, 3, 0], 1e-6);
      AssertEquals('blur (0,0)', csBlurPixels[SizePos, 1], Filtered[0, 0, 0], 1e-6);
      AssertEquals('blur (8,6)', csBlurPixels[SizePos, 2], Filtered[8, 6, 0], 1e-6);
    end;
  finally
    Filtered.Free;
    Mask.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestListImageFilesInFolder;
const
  cFileNames: array[0..5] of string = ('f10.png', 'f2.PNG', 'f1.jpeg',
    'f007.png', 'f3.ppm', 'notes.txt');
  cExpected: array[0..4] of string = ('f1.jpeg', 'f2.PNG', 'f3.ppm',
    'f007.png', 'f10.png');
var
  Dir: string;
  FileNames: TStringList;
  FilePos: integer;
  Stream: TFileStream;
begin
  Dir := IncludeTrailingPathDelimiter(GetTempDir + 'cai_list_image_files');
  ForceDirectories(Dir + 'sub.png');
  FileNames := TStringList.Create;
  try
    for FilePos := 0 to High(cFileNames) do
    begin
      Stream := TFileStream.Create(Dir + cFileNames[FilePos], fmCreate);
      Stream.Free;
    end;
    ListImageFilesInFolder(Dir, FileNames);
    AssertEquals('count', Length(cExpected), FileNames.Count);
    for FilePos := 0 to High(cExpected) do
      AssertEquals('name ' + IntToStr(FilePos), cExpected[FilePos],
        FileNames[FilePos]);
  finally
    FileNames.Free;
    for FilePos := 0 to High(cFileNames) do
      DeleteFile(Dir + cFileNames[FilePos]);
    RemoveDir(Dir + 'sub.png');
    RemoveDir(Dir);
  end;
end;

// '' when every pixel of Image is Color inside the '#' cells of Pattern (font
// pixels Scale x Scale from (Left, Top)) and 0 elsewhere; else the first mismatch.
function BitmapPatternMismatch(Image: TNNetVolume; Left, Top, Scale: integer;
  const Pattern: array of string; const Color: array of TNeuralFloat): string;
var
  X, Y, Channel, PatternColumn, PatternRow: integer;
  Expected: TNeuralFloat;
begin
  Result := '';
  for Y := 0 to Image.SizeY - 1 do
    for X := 0 to Image.SizeX - 1 do
      for Channel := 0 to Image.Depth - 1 do
      begin
        Expected := 0;
        if (X >= Left) and (Y >= Top) then
        begin
          PatternColumn := (X - Left) div Scale;
          PatternRow := (Y - Top) div Scale;
          if (PatternRow <= High(Pattern)) and
            (PatternColumn < Length(Pattern[PatternRow])) and
            (Pattern[PatternRow][PatternColumn + 1] = '#') then
            Expected := Color[Channel];
        end;
        if Image[X, Y, Channel] <> Expected then
          exit(Format('pixel (%d, %d, %d) is %g, expected %g',
            [X, Y, Channel, Image[X, Y, Channel], Expected]));
      end;
end;

procedure TTestNeuralImagePreprocess.TestBitmapTextGlyphPixels;
const
  csGlyphA: array[0..6] of string = ('.###.', '#...#', '#...#', '#...#',
    '#####', '#...#', '#...#');
  csGlyphOne: array[0..6] of string = ('..#..', '.##..', '..#..', '..#..',
    '..#..', '..#..', '.###.');
  csGlyphQuestion: array[0..6] of string = ('.###.', '#...#', '....#',
    '...#.', '..#..', '.....', '..#..');
  csRGB: array[0..2] of TNeuralFloat = (10, 20, 30);
  csWhite: array[0..0] of TNeuralFloat = (255);
var
  Image: TNNetVolume;
begin
  Image := TNNetVolume.Create(8, 10, 3);
  try
    Image.Fill(0);
    AssertEquals('A width', 5, DrawBitmapText(Image, 1, 2, 'A', csRGB));
    AssertEquals('A', '', BitmapPatternMismatch(Image, 1, 2, 1, csGlyphA, csRGB));
    Image.Fill(0);
    DrawBitmapText(Image, 1, 2, 'a', csRGB);
    AssertEquals('a as A', '', BitmapPatternMismatch(Image, 1, 2, 1, csGlyphA,
      csRGB));
    Image.ReSize(12, 16, 1);
    Image.Fill(0);
    AssertEquals('1 width', 10, DrawBitmapText(Image, 1, 1, '1', csWhite, 2));
    AssertEquals('1 at scale 2', '', BitmapPatternMismatch(Image, 1, 1, 2,
      csGlyphOne, csWhite));
    Image.Fill(0);
    DrawBitmapText(Image, 3, 4, '~', csWhite);
    AssertEquals('unknown as ?', '', BitmapPatternMismatch(Image, 3, 4, 1,
      csGlyphQuestion, csWhite));
  finally
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestBitmapTextUTF8AndLookAlikes;
const
  csWhite: array[0..0] of TNeuralFloat = (255);
  csLookAlikeFrom = '&[{<]}>"`;|\';
  csLookAlikeTo = '+((()))''''://';
var
  Got, Want: TNNetVolume;
  CharPos: integer;
begin
  // 'cafe' with U+00E9 (two UTF-8 bytes) and U+20AC (three) is 5 glyphs.
  AssertEquals('UTF-8 width', 29,
    BitmapTextWidth('caf' + #$C3#$A9 + #$E2#$82#$AC));
  Got := TNNetVolume.Create(40, 9, 1);
  Want := TNNetVolume.Create(40, 9, 1);
  try
    Got.Fill(0);
    Want.Fill(0);
    AssertEquals('UTF-8 drawn width', 29, DrawBitmapText(Got, 0, 1,
      'caf' + #$C3#$A9 + #$E2#$82#$AC, csWhite));
    DrawBitmapText(Want, 0, 1, 'CAF??', csWhite);
    AssertTrue('one ? per UTF-8 character', Got.GetDistanceSqr(Want) = 0);
    for CharPos := 1 to Length(csLookAlikeFrom) do
    begin
      Got.Fill(0);
      Want.Fill(0);
      DrawBitmapText(Got, 1, 1, csLookAlikeFrom[CharPos], csWhite);
      DrawBitmapText(Want, 1, 1, csLookAlikeTo[CharPos], csWhite);
      AssertTrue(csLookAlikeFrom[CharPos] + ' drawn as ' +
        csLookAlikeTo[CharPos], Got.GetDistanceSqr(Want) = 0);
    end;
  finally
    Got.Free;
    Want.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestBitmapTextWidth;
const
  csWhite: array[0..0] of TNeuralFloat = (1);
var
  Image: TNNetVolume;
  HasRaised: boolean;
begin
  AssertEquals('empty', 0, BitmapTextWidth(''));
  AssertEquals('one glyph', 5, BitmapTextWidth('A'));
  AssertEquals('ID:12 at scale 2', 58, BitmapTextWidth('ID:12', 2));
  Image := TNNetVolume.Create(4, 4, 1);
  try
    AssertEquals('drawn width', 33, DrawBitmapText(Image, 0, 0, 'AB', csWhite, 3));
    AssertEquals('drawn empty', 0, DrawBitmapText(Image, 0, 0, '', csWhite));
  finally
    Image.Free;
  end;
  HasRaised := false;
  try
    BitmapTextWidth('A', 0);
  except
    on EArgumentException do HasRaised := true;
  end;
  AssertTrue('scale 0 raises', HasRaised);
end;

procedure TTestNeuralImagePreprocess.TestBitmapTextClipsAtBorder;
const
  csGlyphMinus: array[0..6] of string = ('.....', '.....', '.....', '#####',
    '.....', '.....', '.....');
  csAllOn: array[0..0] of string = ('#####');
  csOne: array[0..0] of TNeuralFloat = (1);
  csSeven: array[0..0] of TNeuralFloat = (7);
  csTwoChannels: array[0..1] of TNeuralFloat = (1, 2);
var
  Image: TNNetVolume;
  HasRaised: boolean;
begin
  Image := TNNetVolume.Create(4, 4, 1);
  try
    Image.Fill(0);
    DrawBitmapText(Image, -2, -3, '-', csOne);
    AssertEquals('minus across the top-left corner', '',
      BitmapPatternMismatch(Image, -2, -3, 1, csGlyphMinus, csOne));
    Image.Fill(0);
    DrawBitmapText(Image, 1000, 1000, 'HELLO', csOne);
    DrawBitmapText(Image, -1000, -1000, 'HELLO', csOne, 50);
    DrawBitmapText(Image, 3, -1000, 'W%#', csOne, 20);
    AssertEquals('far outside draws nothing', 0, Image.GetSum(), 0);
    // W row 3 is #.#.#: at scale 100 its middle cell covers the whole image.
    DrawBitmapText(Image, -200, -300, 'W', csOne, 100);
    AssertEquals('one font pixel over all of it', '',
      BitmapPatternMismatch(Image, -1, -1, 100, csAllOn, csOne));
    Image.Fill(0);
    FillRectangleCV2(Image, 5, 3, -2, 1, csSeven);
    AssertEquals('rectangle rows 1-3', 7 * 12, Image.GetSum(), 0);
    AssertEquals('rectangle row 0', 0, Image[2, 0, 0], 0);
    AssertEquals('rectangle row 1', 7, Image[0, 1, 0], 0);
    HasRaised := false;
    try
      DrawBitmapText(Image, 0, 0, 'A', csTwoChannels);
    except
      on EArgumentException do HasRaised := true;
    end;
    AssertTrue('colour per channel', HasRaised);
  finally
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestDrawRectangleOutline;
var
  Image: TNNetVolume;
begin
  Image := TNNetVolume.Create(8, 6, 1);
  try
    Image.Fill(0);
    DrawRectangleOutline(Image, 6, 4, 1, 1, [1]);
    // 6 x 4 outline: 2 * 6 + 2 * 2 pixels.
    AssertEquals('1 px ring', 16, Image.GetSum(), 0);
    AssertEquals('corner', 1, Image[1, 1, 0], 0);
    AssertEquals('inside', 0, Image[3, 2, 0], 0);
    Image.Fill(0);
    DrawRectangleOutline(Image, 0, 0, 7, 5, [1], 2);
    // Rings 8 x 6 and 6 x 4: 24 + 16 pixels, the 4 x 2 centre stays empty.
    AssertEquals('2 px ring', 40, Image.GetSum(), 0);
    AssertEquals('second ring', 1, Image[1, 1, 0], 0);
    AssertEquals('centre', 0, Image[2, 2, 0], 0);
    Image.Fill(0);
    DrawRectangleOutline(Image, -3, 2, 20, 30, [1]);
    // Only the top edge (row 2) is inside the image.
    AssertEquals('clipped to the top edge', 8, Image.GetSum(), 0);
    AssertEquals('row 2', 1, Image[7, 2, 0], 0);
  finally
    Image.Free;
  end;
end;

procedure TTestNeuralImagePreprocess.TestConvexHullOfKnownPoints;
const
  csPointX: array[0..7] of TNeuralFloat = (2, 0, 4, 1, 4, 0, 2, 4);
  csPointY: array[0..7] of TNeuralFloat = (2, 0, 0, 3, 4, 4, 0, 4);
var
  HullX, HullY: TNeuralFloatDynArr;
  HullPos, CornerPos, Matches: integer;
begin
  // A 4 x 4 square with an interior point, a point on an edge and a duplicate.
  AssertEquals('hull points', 4, ConvexHullOfPoints(csPointX, csPointY, HullX,
    HullY));
  for CornerPos := 0 to 3 do
  begin
    Matches := 0;
    for HullPos := 0 to 3 do
      if (HullX[HullPos] = 4 * (CornerPos mod 2)) and
        (HullY[HullPos] = 4 * (CornerPos div 2)) then Inc(Matches);
    AssertEquals('corner ' + IntToStr(CornerPos) + ' once', 1, Matches);
  end;
  AssertEquals('area', 16, PolygonArea(HullX, HullY), 1e-6);
  AssertEquals('perimeter', 16, PolygonPerimeter(HullX, HullY), 1e-6);
  AssertEquals('collinear points keep the ends', 2,
    ConvexHullOfPoints([0, 1, 2, 3], [0, 1, 2, 3], HullX, HullY));
  AssertEquals('repeated point', 1, ConvexHullOfPoints([5, 5], [7, 7], HullX,
    HullY));
  AssertEquals('no points', 0, ConvexHullOfPoints([], [], HullX, HullY));
end;

procedure TTestNeuralImagePreprocess.TestMinAreaRectAndUnclip;
var
  PointX, PointY, HullX, HullY: TNeuralFloatDynArr;
  Rect: TImageRotatedRect;
  OffsetX, OffsetY, PointCount: integer;
  Distance: TNeuralFloat;
begin
  // 20 x 10 rectangle: itself, then the DB unclip D = 200 * 1.5 / 60 = 5.
  ConvexHullOfPoints([20, 0, 20, 0, 10], [10, 0, 0, 10, 5], HullX, HullY);
  Rect := MinAreaRectOfConvexHull(HullX, HullY);
  AssertEquals('TL x', 0, Rect.CornerX[0], 1e-5);
  AssertEquals('TL y', 0, Rect.CornerY[0], 1e-5);
  AssertEquals('TR x', 20, Rect.CornerX[1], 1e-5);
  AssertEquals('TR y', 0, Rect.CornerY[1], 1e-5);
  AssertEquals('BR x', 20, Rect.CornerX[2], 1e-5);
  AssertEquals('BR y', 10, Rect.CornerY[2], 1e-5);
  AssertEquals('BL x', 0, Rect.CornerX[3], 1e-5);
  AssertEquals('BL y', 10, Rect.CornerY[3], 1e-5);
  Distance := PolygonArea(HullX, HullY) * 1.5 / PolygonPerimeter(HullX, HullY);
  AssertEquals('unclip distance', 5, Distance, 1e-6);
  OffsetRotatedRect(Rect, Distance);
  AssertEquals('unclipped TL x', -5, Rect.CornerX[0], 1e-5);
  AssertEquals('unclipped TL y', -5, Rect.CornerY[0], 1e-5);
  AssertEquals('unclipped BR x', 25, Rect.CornerX[2], 1e-5);
  AssertEquals('unclipped BR y', 15, Rect.CornerY[2], 1e-5);
  AssertEquals('unclipped width', 30, RotatedRectWidth(Rect), 1e-5);
  AssertEquals('unclipped height', 20, RotatedRectHeight(Rect), 1e-5);
  // Lattice points with |dx + dy| <= 20 and |dx - dy| <= 6 around (30, 30): a
  // 45 degree rectangle 40 / sqrt(2) long and 12 / sqrt(2) wide.
  SetLength(PointX, 41 * 41);
  SetLength(PointY, 41 * 41);
  PointCount := 0;
  for OffsetY := -20 to 20 do
    for OffsetX := -20 to 20 do
      if (Abs(OffsetX + OffsetY) <= 20) and (Abs(OffsetX - OffsetY) <= 6) then
      begin
        PointX[PointCount] := 30 + OffsetX;
        PointY[PointCount] := 30 + OffsetY;
        Inc(PointCount);
      end;
  SetLength(PointX, PointCount);
  SetLength(PointY, PointCount);
  AssertEquals('45 degree hull corners', 4, ConvexHullOfPoints(PointX, PointY,
    HullX, HullY));
  Rect := MinAreaRectOfConvexHull(HullX, HullY);
  AssertEquals('45 degree width', 40 / Sqrt(2), RotatedRectWidth(Rect), 1e-4);
  AssertEquals('45 degree height', 12 / Sqrt(2), RotatedRectHeight(Rect), 1e-4);
  AssertEquals('45 degree angle', Pi / 4, ArcTan2(Rect.CornerY[1] -
    Rect.CornerY[0], Rect.CornerX[1] - Rect.CornerX[0]), 1e-5);
  AssertEquals('TL is (23, 17)', 23, Rect.CornerX[0], 1e-4);
  AssertEquals('TL y', 17, Rect.CornerY[0], 1e-4);
  AssertEquals('BR is (37, 43)', 37, Rect.CornerX[2], 1e-4);
  AssertEquals('BR y', 43, Rect.CornerY[2], 1e-4);
  // A vertical bar: the long side runs straight down.
  ConvexHullOfPoints([3, 5, 5, 3], [2, 2, 30, 30], HullX, HullY);
  Rect := MinAreaRectOfConvexHull(HullX, HullY);
  AssertEquals('vertical width', 28, RotatedRectWidth(Rect), 1e-5);
  AssertEquals('vertical TL x', 5, Rect.CornerX[0], 1e-5);
  AssertEquals('vertical TL y', 2, Rect.CornerY[0], 1e-5);
  AssertEquals('vertical TR y', 30, Rect.CornerY[1], 1e-5);
end;

procedure TTestNeuralImagePreprocess.TestCropRotatedRect;
var
  Image, Crop: TNNetVolume;
  Rect: TImageRotatedRect;
  HullX, HullY: TNeuralFloatDynArr;
  X, Y, Channel, OffsetX, OffsetY: integer;
  SourceX, SourceY: TNeuralFloat;
begin
  Image := TNNetVolume.Create(48, 40, 3);
  Crop := TNNetVolume.Create;
  try
    // A linear image: bilinear sampling reproduces 2x + 3y + channel exactly.
    for Y := 0 to 39 do
      for X := 0 to 47 do
        for Channel := 0 to 2 do Image[X, Y, Channel] := 2 * X + 3 * Y + Channel;
    // 20 x 10 at angle atan(3/4): width axis (0.8, 0.6), height axis (-0.6, 0.8).
    Rect.CornerX[0] := 10;  Rect.CornerY[0] := 8;
    Rect.CornerX[1] := 26;  Rect.CornerY[1] := 20;
    Rect.CornerX[2] := 20;  Rect.CornerY[2] := 28;
    Rect.CornerX[3] := 4;   Rect.CornerY[3] := 16;
    CropRotatedRect(Image, Crop, Rect, [255, 255, 255]);
    AssertEquals('crop width', 20, Crop.SizeX);
    AssertEquals('crop height', 10, Crop.SizeY);
    AssertEquals('crop depth', 3, Crop.Depth);
    for Y := 0 to 9 do
      for X := 0 to 19 do
      begin
        SourceX := 10 + X * 16 / 19 - Y * 6 / 9;
        SourceY := 8 + X * 12 / 19 + Y * 8 / 9;
        for Channel := 0 to 2 do
          AssertEquals('crop pixel ' + IntToStr(X) + ',' + IntToStr(Y),
            2 * SourceX + 3 * SourceY + Channel, Crop[X, Y, Channel], 1e-3);
      end;
    // A dark bar 3 px wide along 45 degrees on white: the upright crop of its
    // 28.3 x 8.5 box is dark along the middle and white along the top and bottom.
    Image.ReSize(64, 64, 3);
    Image.Fill(255);
    for OffsetY := -20 to 20 do
      for OffsetX := -20 to 20 do
        if (Abs(OffsetX - OffsetY) <= 2) and (Abs(OffsetX + OffsetY) <= 20) then
          for Channel := 0 to 2 do Image[32 + OffsetX, 32 + OffsetY, Channel] := 0;
    ConvexHullOfPoints([45, 39, 19, 25], [39, 45, 25, 19], HullX, HullY);
    Rect := MinAreaRectOfConvexHull(HullX, HullY);
    CropRotatedRect(Image, Crop, Rect, [255, 255, 255]);
    AssertEquals('bar crop width', 28, Crop.SizeX);
    AssertEquals('bar crop height', 8, Crop.SizeY);
    for X := 5 to 22 do
    begin
      AssertEquals('bar middle row ' + IntToStr(X), 0, Crop[X, 3, 0], 1e-3);
      AssertEquals('bar middle row 4 ' + IntToStr(X), 0, Crop[X, 4, 1], 1e-3);
      AssertEquals('top row white ' + IntToStr(X), 255, Crop[X, 0, 0], 1e-3);
      AssertEquals('bottom row white ' + IntToStr(X), 255, Crop[X, 7, 2], 1e-3);
    end;
    // A clipped, non-rectangular quadrilateral: corners 0, 1, 2 set the affine map
    // and |c2 - c1| = 6 the height (|c3 - c0| is 4).
    Image.ReSize(48, 40, 3);
    for Y := 0 to 39 do
      for X := 0 to 47 do
        for Channel := 0 to 2 do Image[X, Y, Channel] := 2 * X + 3 * Y + Channel;
    Rect.CornerX[0] := 10;  Rect.CornerY[0] := 8;
    Rect.CornerX[1] := 30;  Rect.CornerY[1] := 8;
    Rect.CornerX[2] := 30;  Rect.CornerY[2] := 14;
    Rect.CornerX[3] := 10;  Rect.CornerY[3] := 12;
    CropRotatedRect(Image, Crop, Rect, [255, 255, 255]);
    AssertEquals('quad crop width', 20, Crop.SizeX);
    AssertEquals('quad crop height', 6, Crop.SizeY);
    for Y := 0 to 5 do
      for X := 0 to 19 do
        AssertEquals('quad pixel ' + IntToStr(X) + ',' + IntToStr(Y),
          2 * (10 + X * 20 / 19) + 3 * (8 + Y * 6 / 5), Crop[X, Y, 0], 1e-3);
  finally
    Crop.Free;
    Image.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralImagePreprocess);
end.
