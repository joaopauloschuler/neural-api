unit neuralimageresize;

(*
neuralimageresize: image resampling, warping and filtering on TNNetVolume
with no fcl-image dependency, so importers can use it.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  Types, neuralvolume;

type
  // PIL's resample filters (Image.Resampling); the PIL integer codes map via
  // PILResampleCodeToFilter.
  TImageResampleFilter = (irfNearest, irfBox, irfBilinear, irfHamming,
    irfBicubic, irfLanczos);

// Source (W,H,D) to Dest (NewSizeX,NewSizeY,D) as PIL Image.resize: mode 'F',
// or 8-bit (fixed point, uint8 per pass) if RoundToBytes; Dest may be Source.
procedure ResizeImagePIL(Source, Dest: TNNetVolume; NewSizeX,
  NewSizeY: integer; Filter: TImageResampleFilter;
  RoundToBytes: boolean = false);

// ResizeImagePIL with irfLanczos. Not PIL's 8-bit rounding.
procedure ResizeImageLanczos(Source, Dest: TNNetVolume; NewSizeX,
  NewSizeY: integer);

// PIL convert to L/RGB/RGBA (Channels 1/3/4) + resize, RGBA premultiplied, in
// float; 0..255 maps to [OutputMin, OutputMax]. Dest may be Source.
procedure ConvertAndResizeImagePIL(Source, Dest: TNNetVolume; Width, Height,
  Channels: integer; ClipToByteRange: boolean; OutputMin,
  OutputMax: TNeuralFloat; Filter: TImageResampleFilter = irfLanczos);

// HF/torchvision shortest-edge resize of a 0..255 image (long side
// Trunc(ShortestEdge*Long/Short)) with PIL's uint8 rounding.
procedure ResizeByteImageShortestEdge(Source, Dest: TNNetVolume;
  ShortestEdge: integer; Filter: TImageResampleFilter);

// controlnet_aux util.resize_image output size: scale so the short side is
// Resolution, then np.round (half to even) each side to a multiple of 64.
procedure ControlNetAuxResizeImageSize(Width, Height, Resolution: integer;
  out ResizedWidth, ResizedHeight: integer);

// Separable Gaussian blur of Source (W,H,D) per channel, sigma = Radius (PIL
// GaussianBlur radius), clamped edges. Dest may be Source; Radius <= 0 copies.
procedure GaussianBlurImage(Source, Dest: TNNetVolume; Radius: TNeuralFloat); overload;
// torchvision gaussian_blur: KernelSize (odd) taps of std Sigma per axis,
// reflect padding (no edge repeat). Dest may be Source.
procedure GaussianBlurImage(Source, Dest: TNNetVolume; Sigma: TNeuralFloat;
  KernelSize: integer); overload;
// scipy.ndimage.gaussian_filter(sigma=Sigma, radius=TapRadius) on x and y of each
// channel, mode "reflect" (the edge pixel repeats). Dest may be Source.
procedure GaussianFilterImageScipy(Source, Dest: TNNetVolume;
  Sigma: TNeuralFloat; TapRadius: integer);

// cv2.Canny(Source, Low, High, apertureSize=3, L2gradient) of a 0..255 image
// (no blur; strongest channel per pixel): Dest (W,H,DestDepth) holds 0 or 255.
procedure CannyEdges(Source, Dest: TNNetVolume; LowThreshold,
  HighThreshold: TNeuralFloat; L2Gradient: boolean = false;
  DestDepth: integer = 1);

// Labels[Y * SizeX + X] := 1..Result for the 8-connected regions of Map channel
// Channel above Threshold, numbered in raster order of their first pixel; 0 elsewhere.
function LabelEightConnectedComponents(Map: TNNetVolume; Channel: integer;
  Threshold: TNeuralFloat; var Labels: TNeuralIntegerArray): integer;

// X1..X2, Y1..Y2 (inclusive) bound the pixels of Map channel Channel above
// Threshold; False and all zero when there are none.
function MaskChannelBounds(Map: TNNetVolume; Channel: integer;
  Threshold: TNeuralFloat; out X1, Y1, X2, Y2: integer): boolean;

// Where Mask channel MaskChannel > Threshold, Image := (1 - Alpha) * Image +
// Alpha * Color (one value per channel of Image); Mask has Image's SizeX, SizeY.
procedure BlendColorWhereMask(Image, Mask: TNNetVolume; MaskChannel: integer;
  const Color: array of TNeuralFloat; Alpha: TNeuralFloat;
  Threshold: TNeuralFloat = 0);

// Canvas := Image with Left/Top/Right/Bottom pixels added on each side, each
// new pixel copying the nearest edge pixel (torch F.pad mode="replicate").
procedure PadImageClampToEdge(Image: TNNetVolume; Left, Top, Right,
  Bottom: integer; Canvas: TNNetVolume);

// Outpaint canvas: Image at (Left, Top) on a larger Canvas, the new border
// clamp-to-edge; Mask (W, H, 1) is 1 there and Overlap pixels in, 0 elsewhere.
procedure BuildOutpaintCanvas(Image: TNNetVolume; Left, Top, Right, Bottom,
  Overlap: integer; Canvas, Mask: TNNetVolume);

// Copies Image over Dest (its BuildOutpaintCanvas size) where that Mask is 0,
// the channels both have, so the kept pixels are exact.
procedure PasteOutpaintOriginal(Image: TNNetVolume; Left, Top, Right,
  Bottom, Overlap: integer; Dest: TNNetVolume);

// Grows Right (Left when only Left is extended) and Bottom (or Top) so the
// outpaint canvas sides are multiples of Multiple.
procedure GrowOutpaintSidesToMultiple(ImageSizeX, ImageSizeY,
  Multiple: integer; var Left, Top, Right, Bottom: integer);

// diffusers get_crop_region: [X1, X2) x [Y1, Y2) bounds the Mask's nonzero PIL
// 'L' pixels (0..255) plus Padding, grown to Width:Height, kept in the mask.
procedure GetMaskCropRegion(Mask: TNNetVolume; Width, Height,
  Padding: integer; out X1, Y1, X2, Y2: integer);

// diffusers _resize_and_fill: Source fit inside Width x Height at its aspect,
// centred; the rest repeats its edge pixels (PIL samples them with the filter).
procedure ResizeImageAndFill(Source, Dest: TNNetVolume; Width,
  Height: integer; Filter: TImageResampleFilter; RoundToBytes: boolean = false);

// diffusers _resize_and_crop: Source covering Width x Height at its aspect,
// centred, the excess cropped. Dest may be Source.
procedure ResizeImageAndCrop(Source, Dest: TNNetVolume; Width,
  Height: integer; Filter: TImageResampleFilter; RoundToBytes: boolean = false);

// diffusers apply_overlay: Dest := Original, then Generated resized into
// [X1, X2) x [Y1, Y2) (ResizeImageAndCrop) blended by Mask's PIL 'L' / 255.
procedure PasteInpaintCrop(Generated, Original, Mask: TNNetVolume;
  X1, Y1, X2, Y2: integer; Filter: TImageResampleFilter; Dest: TNNetVolume);

// Parses "L,T,R,B" (pixels added on each side, >= 0, not all 0); returns ''
// or what is wrong.
function ParseOutpaintSides(const Text: string;
  out Left, Top, Right, Bottom: integer): string;

// PIL code (0 NEAREST, 1 LANCZOS, 2 BILINEAR, 3 BICUBIC, 4 BOX, 5 HAMMING),
// as in a HF preprocessor_config.json "resample"; raises on other codes.
function PILResampleCodeToFilter(Code: integer): TImageResampleFilter;

// cv2.resize(INTER_LANCZOS4) in float: 8 taps around (Dest + 0.5) * scale - 0.5,
// edge pixels repeated, no antialias; the same size copies. Dest may be Source.
procedure ResizeImageLanczos4CV2(Source, Dest: TNNetVolume; NewSizeX,
  NewSizeY: integer);

// cv2.GaussianBlur(Source, (KernelSize, KernelSize), 0): sigma from KernelSize
// (cv2's fixed tables up to 9 taps), reflect-101 edges. Dest may be Source.
procedure GaussianBlurImageCV2(Source, Dest: TNNetVolume; KernelSize: integer);

// cv2.erode with a KernelSizeX x KernelSizeY rectangle anchored at its centre;
// pixels past the edge are ignored. A 0 size is cv2's empty kernel (3 x 3).
procedure ErodeImage(Source, Dest: TNNetVolume; KernelSizeX,
  KernelSizeY: integer);

type
  TImagePointArray = array of TPoint;

// cv2.ellipse2Poly((CenterX, CenterY), (AxisX, AxisY), Angle, 0, 360, Delta):
// cv2's float sine table, lrint rounding, repeated points dropped.
function EllipsePolygonCV2(CenterX, CenterY, AxisX, AxisY, Angle: integer;
  Delta: integer = 1): TImagePointArray;

// cv2.line(Image, (X1, Y1), (X2, Y2), Color, 1, LINE_8): Bresenham from the
// left end after cv2.clipLine. Color holds one value per channel of Image.
procedure DrawLineCV2(Image: TNNetVolume; X1, Y1, X2, Y2: integer;
  const Color: array of TNeuralFloat);

// cv2.fillConvexPoly(Image, Points, Color) (LINE_8, shift 0): the edges as
// DrawLineCV2, then cv2's 16.16 fixed-point scanline fill, clipped.
procedure FillConvexPolygonCV2(Image: TNNetVolume;
  const Points: TImagePointArray; const Color: array of TNeuralFloat);

// cv2.circle(Image, (CenterX, CenterY), Radius, Color, thickness=-1) (LINE_8):
// cv2's midpoint-circle spans, clipped.
procedure FillCircleCV2(Image: TNNetVolume; CenterX, CenterY, Radius: integer;
  const Color: array of TNeuralFloat);

// cv2.rectangle(Image, (X1, Y1), (X2, Y2), Color, thickness=-1): the corners in
// any order, both inclusive, clipped.
procedure FillRectangleCV2(Image: TNNetVolume; X1, Y1, X2, Y2: integer;
  const Color: array of TNeuralFloat);

// The outline of the (X1, Y1)-(X2, Y2) rectangle (corners in any order, both
// inclusive, clipped), Thickness pixels wide inwards.
procedure DrawRectangleOutline(Image: TNNetVolume; X1, Y1, X2, Y2: integer;
  const Color: array of TNeuralFloat; Thickness: integer = 1);

const
  // DrawBitmapText glyph size in font pixels; glyphs sit 1 font pixel apart.
  csBitmapFontGlyphWidth = 5;
  csBitmapFontGlyphHeight = 7;

// Width in pixels of Text drawn by DrawBitmapText: (6 * characters - 1) * Scale,
// a UTF-8 multi-byte character counting as one.
function BitmapTextWidth(const Text: string; Scale: integer = 1): integer;

// Draws Text in a 5x7 font, top-left at (Left, Top), each font pixel Scale x Scale,
// clipped; lower case as upper case, & [ ] " ; | as look-alikes, others '?'; returns the width.
function DrawBitmapText(Image: TNNetVolume; Left, Top: integer;
  const Text: string; const Color: array of TNeuralFloat;
  Scale: integer = 1): integer;

type
  // 2 x 3 affine matrix [a b c; d e f], row major, as cv2: (x, y) -> (ax+by+c, dx+ey+f).
  TImageAffineMatrix = array[0..5] of double;

// cv2.invertAffineTransform.
function InvertImageAffineMatrix(const Matrix: TImageAffineMatrix): TImageAffineMatrix;

// Least-squares 4-DOF similarity (rotation, uniform scale, shift; the Umeyama
// closed form) taking the Source points onto the Dest points.
function EstimateSimilarityTransform(const SourceX, SourceY, DestX,
  DestY: array of double): TImageAffineMatrix;

// cv2.estimateAffinePartial2D(method=LMEDS) made deterministic: the point pair
// whose similarity has the least median squared error picks the inliers, then least squares.
function EstimateSimilarityTransformLMedS(const SourceX, SourceY, DestX,
  DestY: array of double): TImageAffineMatrix;

// cv2.warpAffine(Source, Matrix, (DestSizeX, DestSizeY)) bilinear in float with
// BORDER_CONSTANT: an outside neighbour reads BorderValue (missing channels 0).
// BlendsBorder false is scipy affine_transform(order=1, mode="constant"): a
// sample outside [0, Size - 1] reads BorderValue unblended.
procedure WarpImageAffine(Source, Dest: TNNetVolume;
  const Matrix: TImageAffineMatrix; DestSizeX, DestSizeY: integer;
  const BorderValue: array of TNeuralFloat; BlendsBorder: boolean = true);

type
  // Rotated rectangle corners: top-left, top-right, bottom-right, bottom-left; corner 0 -> 1
  // is the long side, pointing right (dx > 0; straight down when vertical).
  TImageRotatedRect = record
    CornerX, CornerY: array[0..3] of TNeuralFloat;
  end;

// Andrew's monotone chain: the convex hull of the points in boundary order, without
// repeated or collinear points; returns the hull point count.
function ConvexHullOfPoints(const PointX, PointY: array of TNeuralFloat;
  out HullX, HullY: TNeuralFloatDynArr): integer;

// Shoelace area (absolute) and closed-boundary length of a polygon.
function PolygonArea(const PointX, PointY: array of TNeuralFloat): TNeuralFloat;
function PolygonPerimeter(const PointX, PointY: array of TNeuralFloat): TNeuralFloat;

// cv2.minAreaRect of a convex hull (ConvexHullOfPoints) by rotating calipers;
// one or two points give a zero-height rectangle.
function MinAreaRectOfConvexHull(const HullX,
  HullY: array of TNeuralFloat): TImageRotatedRect;

function RotatedRectWidth(const Rect: TImageRotatedRect): TNeuralFloat;
function RotatedRectHeight(const Rect: TImageRotatedRect): TNeuralFloat;

// Moves each side of Rect outward by Distance (inward when negative).
procedure OffsetRotatedRect(var Rect: TImageRotatedRect; Distance: TNeuralFloat);

// Crop := Rect of Image made upright, Round(width) x Round(height) (at least 1):
// corners 0, 1, 2 go to (0, 0), (W - 1, 0), (W - 1, H - 1); WarpImageAffine bilinear.
procedure CropRotatedRect(Image, Crop: TNNetVolume; const Rect: TImageRotatedRect;
  const BorderValue: array of TNeuralFloat);

type
  // Resize + crop of a HF image processor: shortest-edge (KeepsAspect) or
  // direct (ResizeHeight,ResizeWidth) resize, then a (CropHeight,CropWidth) crop.
  THFImageResize = record
    KeepsAspect: boolean;
    ShortestEdge: integer;
    ResizeHeight, ResizeWidth: integer;
    CropHeight, CropWidth: integer;
    DoCenterCrop: boolean;
  end;

// "ViTImageProcessorFast" / "ViTFeatureExtractor" / "ViTImageProcessorPil"
// -> "ViT": the processor family of an image_processor_type.
function HFImageProcessorFamily(const ProcessorType: string): string;

// True when transformers reads an int "size" of this processor type as
// (size,size) (default_to_square); unknown types read it as shortest_edge.
function HFImageProcessorIntSizeIsSquare(const ProcessorType: string): boolean;

// Applies the processor rules to the raw size/crop_size fields in Resize: int
// size as a square, crop = resize when uncropped, ConvNext/PoolFormer crop_pct.
procedure ResolveHFImageResize(const ProcessorType: string;
  SizeIsInt, HasCropSize: boolean; CropPct: double; var Resize: THFImageResize);

implementation

uses
  SysUtils, Math, neuralnetwork;

function PILResampleCodeToFilter(Code: integer): TImageResampleFilter;
begin
  case Code of
    0: Result := irfNearest;
    1: Result := irfLanczos;
    2: Result := irfBilinear;
    3: Result := irfBicubic;
    4: Result := irfBox;
    5: Result := irfHamming;
  else
    raise Exception.Create('PILResampleCodeToFilter: unknown PIL resample ' +
      'code ' + IntToStr(Code) + '.');
  end;
end;

const
  // transformers 5.x image processor families whose default_to_square is True.
  csHFSquareIntSizeFamilies: array[0..74] of string = (
    'Aria', 'Beit', 'Blip', 'CHMv2', 'Cohere2Vision', 'DINOv3ViT', 'DPT',
    'DeepseekOcr2', 'DeepseekVL', 'DeepseekVLHybrid', 'DeiT', 'DepthPro',
    'Donut', 'EfficientNet', 'Emu3', 'Flava', 'Fuyu', 'GLPN', 'Gemma3',
    'Gemma4', 'Gemma4Unified', 'GotOcr2', 'Idefics', 'Idefics3', 'ImageGPT',
    'Janus', 'Kimi_K25', 'Kosmos2_5', 'LayoutLMv2', 'LayoutLMv3', 'Lfm2Vl',
    'Llama4', 'MiniCPMV4_6', 'Mllama', 'NeoMME', 'Nougat', 'Ovis2', 'OwlViT',
    'Owlv2', 'PPChart2Table', 'PPDocLayoutV2', 'PPDocLayoutV3',
    'PPFormulaNet', 'PPLCNet', 'PPOCRV5ServerDet', 'PPOCRV5ServerRec',
    'PPOCRV6SmallRec', 'Perceiver', 'PerceptionLM', 'Phi4Multimodal',
    'Pix2Struct', 'Pixtral', 'PromptDepthAnything', 'Pvt', 'SLANeXt', 'Sam',
    'Sam2', 'Sam3', 'Sapiens2', 'SegGpt', 'Segformer', 'Siglip2', 'SmolVLM',
    'Step3p7', 'Swin2SR', 'TimmWrapper', 'Tipsv2', 'Tipsv2Dpt', 'UVDoc',
    'ViT', 'VideoLlava', 'VitMatte', 'VitPose', 'Vivit', 'ZoeDepth');

function HFImageProcessorFamily(const ProcessorType: string): string;

  procedure StripSuffix(const Suffix: string);
  var
    Start: integer;
  begin
    Start := Length(Result) - Length(Suffix) + 1;
    if (Start > 1) and (Copy(Result, Start, Length(Suffix)) = Suffix) then
      SetLength(Result, Start - 1);
  end;

begin
  Result := Trim(ProcessorType);
  StripSuffix('Fast');
  StripSuffix('Pil');
  StripSuffix('ImageProcessor');
  StripSuffix('FeatureExtractor');
end;

function HFImageProcessorIntSizeIsSquare(const ProcessorType: string): boolean;
var
  Family: string;
  FamilyIdx: integer;
begin
  Result := false;
  Family := HFImageProcessorFamily(ProcessorType);
  if Family = '' then Exit;
  for FamilyIdx := Low(csHFSquareIntSizeFamilies) to
    High(csHFSquareIntSizeFamilies) do
    if SameText(Family, csHFSquareIntSizeFamilies[FamilyIdx]) then
      Exit(true);
end;

procedure ResolveHFImageResize(const ProcessorType: string;
  SizeIsInt, HasCropSize: boolean; CropPct: double; var Resize: THFImageResize);
var
  Family: string;
  Edge: integer;
begin
  if SizeIsInt and HFImageProcessorIntSizeIsSquare(ProcessorType) then
  begin
    Resize.KeepsAspect := false;
    Resize.ResizeHeight := Resize.ShortestEdge;
    Resize.ResizeWidth := Resize.ShortestEdge;
  end;
  Family := HFImageProcessorFamily(ProcessorType);
  if SameText(Family, 'ConvNext') then
  begin
    // ConvNextImageProcessor.resize: crop_pct (default 224/256) below 384,
    // a plain (edge,edge) warp from 384 up.
    if CropPct <= 0 then CropPct := 224 / 256;
    Edge := Resize.ShortestEdge;
    Resize.CropHeight := Edge;
    Resize.CropWidth := Edge;
    Resize.DoCenterCrop := true;
    Resize.KeepsAspect := Edge < 384;
    if Resize.KeepsAspect then
      Resize.ShortestEdge := Trunc(Edge / CropPct)
    else
    begin
      Resize.ResizeHeight := Edge;
      Resize.ResizeWidth := Edge;
    end;
    Exit;
  end;
  if SameText(Family, 'PoolFormer') then
  begin
    // PoolFormerImageProcessor.resize: the size scaled by 1/crop_pct, then
    // the usual crop_size center crop (class defaults 0.9 and 224x224).
    if CropPct <= 0 then CropPct := 0.9;
    if not HasCropSize then
    begin
      Resize.CropHeight := 224;
      Resize.CropWidth := 224;
      HasCropSize := true;
    end;
    if (not Resize.KeepsAspect) and
      (Resize.ResizeHeight = Resize.ResizeWidth) then
    begin
      Resize.KeepsAspect := true;
      Resize.ShortestEdge := Resize.ResizeHeight;
    end;
    if Resize.KeepsAspect then
      Resize.ShortestEdge := Trunc(Resize.ShortestEdge / CropPct)
    else
    begin
      Resize.ResizeHeight := Trunc(Resize.ResizeHeight / CropPct);
      Resize.ResizeWidth := Trunc(Resize.ResizeWidth / CropPct);
    end;
  end;
  // HF crops only when do_center_crop; ViT/SigLIP have no crop_size at all.
  if (not Resize.KeepsAspect) and
    ((not HasCropSize) or (not Resize.DoCenterCrop)) then
  begin
    Resize.CropHeight := Resize.ResizeHeight;
    Resize.CropWidth := Resize.ResizeWidth;
  end;
end;

procedure ResizeByteImageShortestEdge(Source, Dest: TNNetVolume;
  ShortestEdge: integer; Filter: TImageResampleFilter);
var
  NewSizeX, NewSizeY: integer;
begin
  if Source.SizeX <= Source.SizeY then
  begin
    NewSizeX := ShortestEdge;
    NewSizeY := Trunc(ShortestEdge * Source.SizeY / Source.SizeX);
  end
  else
  begin
    NewSizeY := ShortestEdge;
    NewSizeX := Trunc(ShortestEdge * Source.SizeX / Source.SizeY);
  end;
  if NewSizeX < 1 then NewSizeX := 1;
  if NewSizeY < 1 then NewSizeY := 1;
  ResizeImagePIL(Source, Dest, NewSizeX, NewSizeY, Filter,
    {RoundToBytes=}true);
end;

procedure ControlNetAuxResizeImageSize(Width, Height, Resolution: integer;
  out ResizedWidth, ResizedHeight: integer);
var
  Scale: double;
begin
  // Double, as numpy; Round is half to even like np.round.
  Scale := Resolution / Min(Width, Height);
  ResizedWidth := Round(Width * Scale / 64.0) * 64;
  ResizedHeight := Round(Height * Scale / 64.0) * 64;
end;

procedure ResizeImageLanczos(Source, Dest: TNNetVolume; NewSizeX,
  NewSizeY: integer);
begin
  ResizeImagePIL(Source, Dest, NewSizeX, NewSizeY, irfLanczos);
end;

procedure ConvertAndResizeImagePIL(Source, Dest: TNNetVolume; Width, Height,
  Channels: integer; ClipToByteRange: boolean; OutputMin,
  OutputMax: TNeuralFloat; Filter: TImageResampleFilter);
const
  MaxChannelValue: TNeuralFloat = 255;
  // PIL 'L' = (R*19595 + G*38470 + B*7471 + 0x8000) >> 16.
  csLumaR = 19595;
  csLumaG = 38470;
  csLumaB = 7471;
  csLumaRound = 32768;
var
  Converted: TNNetVolume;
  PixelPos, MaxPixelPos, SourcePos, DestPos, Channel, MaxColorChannel,
    MaxChannel, SourceDepth: integer;
  Alpha, AlphaScale, Value, Scale: TNeuralFloat;
  IsResized, IsPremultiplied, OwnsConverted: boolean;
begin
  SourceDepth := Source.Depth;
  if (SourceDepth <> 1) and (SourceDepth <> 3) and (SourceDepth <> 4) then
    raise Exception.Create('ConvertAndResizeImagePIL: the image has ' +
      IntToStr(SourceDepth) + ' channels, expected gray, RGB or RGBA.');
  if (Channels <> 1) and (Channels <> 3) and (Channels <> 4) then
    raise Exception.Create('ConvertAndResizeImagePIL: ' + IntToStr(Channels) +
      ' output channels; 1, 3 or 4 are supported.');
  if (Width < 1) or (Height < 1) then
    raise Exception.Create('ConvertAndResizeImagePIL: target size ' +
      IntToStr(Width) + 'x' + IntToStr(Height) + ' is empty.');
  IsResized := (Source.SizeX <> Width) or (Source.SizeY <> Height);
  MaxPixelPos := Source.SizeX * Source.SizeY - 1;
  // PIL resizes RGBA as premultiplied "RGBa"; an opaque image skips it.
  IsPremultiplied := false;
  if IsResized and (Channels = 4) and (SourceDepth = 4) then
    for PixelPos := 0 to MaxPixelPos do
      if Source.FData[PixelPos * 4 + 3] < MaxChannelValue then
      begin
        IsPremultiplied := true;
        break;
      end;
  Converted := Source;
  OwnsConverted := false;
  try
    if (SourceDepth <> Channels) or IsPremultiplied then
    begin
      if IsResized or (Source = Dest) then
      begin
        Converted := TNNetVolume.Create();
        OwnsConverted := true;
      end
      else Converted := Dest;
      Converted.ReSize(Source.SizeX, Source.SizeY, Channels);
      MaxColorChannel := Min(Channels, 3) - 1;
      for PixelPos := 0 to MaxPixelPos do
      begin
        SourcePos := PixelPos * SourceDepth;
        DestPos := PixelPos * Channels;
        if SourceDepth = 4 then Alpha := Source.FData[SourcePos + 3]
        else Alpha := MaxChannelValue;
        if SourceDepth = 1 then
          for Channel := 0 to MaxColorChannel do
            Converted.FData[DestPos + Channel] := Source.FData[SourcePos]
        else if Channels = 1 then
          Converted.FData[DestPos] := Trunc((Source.FData[SourcePos] * csLumaR +
            Source.FData[SourcePos + 1] * csLumaG +
            Source.FData[SourcePos + 2] * csLumaB + csLumaRound) / 65536)
        else if IsPremultiplied then
        begin
          AlphaScale := Alpha / MaxChannelValue;
          for Channel := 0 to MaxColorChannel do
            Converted.FData[DestPos + Channel] :=
              Source.FData[SourcePos + Channel] * AlphaScale;
        end
        else
          for Channel := 0 to MaxColorChannel do
            Converted.FData[DestPos + Channel] :=
              Source.FData[SourcePos + Channel];
        if Channels = 4 then Converted.FData[DestPos + 3] := Alpha;
      end;
    end;
    if IsResized then
    begin
      ResizeImagePIL(Converted, Dest, Width, Height, Filter);
      if ClipToByteRange or IsPremultiplied then
      begin
        MaxPixelPos := Width * Height - 1;
        MaxChannel := Channels - 1;
        for PixelPos := 0 to MaxPixelPos do
        begin
          DestPos := PixelPos * Channels;
          if ClipToByteRange then
            for Channel := 0 to MaxChannel do
            begin
              Value := Dest.FData[DestPos + Channel];
              if Value < 0 then Dest.FData[DestPos + Channel] := 0
              else if Value > MaxChannelValue then
                Dest.FData[DestPos + Channel] := MaxChannelValue;
            end;
          if IsPremultiplied then
          begin
            Alpha := Dest.FData[DestPos + 3];
            // PIL's rgba2rgbA leaves a pixel with alpha 0 premultiplied.
            if (Alpha > 0) and (Alpha < MaxChannelValue) then
              for Channel := 0 to 2 do
              begin
                Value := Dest.FData[DestPos + Channel] * MaxChannelValue /
                  Alpha;
                if Value > MaxChannelValue then Value := MaxChannelValue;
                Dest.FData[DestPos + Channel] := Value;
              end;
          end;
        end;
      end;
    end
    else if Converted <> Dest then
    begin
      Dest.ReSize(Converted);
      Dest.Copy(Converted);
    end;
  finally
    if OwnsConverted then Converted.Free;
  end;
  Scale := (OutputMax - OutputMin) / MaxChannelValue;
  if Scale <> 1 then Dest.Mul(Scale);
  if OutputMin <> 0 then Dest.Add(OutputMin);
end;

procedure ResizeImagePIL(Source, Dest: TNNetVolume; NewSizeX,
  NewSizeY: integer; Filter: TImageResampleFilter; RoundToBytes: boolean);
type
  // Per output coordinate of one axis: the first source coordinate, the tap
  // count, and the normalised weights at Weight[OutPos * MaxTaps].
  TResampleAxis = record
    First, Count: array of integer;
    Weight: array of double;
    MaxTaps: integer;
  end;
var
  AxisX, AxisY: TResampleAxis;
  // Source resized along X only: (NewSizeX, Source.SizeY, Depth).
  Rows: array of TNeuralFloat;
  ChannelCount, SourceSizeY, RowStride, SourceRowStride: integer;
  X, Y, Channel, TapPos, ElementPos: integer;
  MaxX, MaxY, MaxSourceY, MaxChannel, MaxTapPos, MaxElementPos: integer;
  TapBase, FirstPos, SourcePos, DestPos: integer;
  Sum: double;

  // PIL's *_filter kernels (Resample.c).
  function FilterTap(Distance: double): double;

    function Sinc(Value: double): double;
    begin
      if Value = 0 then exit(1);
      Value := Value * Pi;
      Result := Sin(Value) / Value;
    end;

  begin
    Result := 0;
    case Filter of
      irfBox:
        if (Distance > -0.5) and (Distance <= 0.5) then Result := 1;
      irfBilinear:
        begin
          Distance := Abs(Distance);
          if Distance < 1 then Result := 1 - Distance;
        end;
      irfHamming:
        begin
          Distance := Abs(Distance);
          if Distance = 0 then Result := 1
          else if Distance < 1 then
          begin
            Distance := Distance * Pi;
            Result := Sin(Distance) / Distance *
              (single(0.54) + single(0.46) * Cos(Distance));
          end;
        end;
      irfBicubic:
        begin
          // Keys cubic with a = -0.5.
          Distance := Abs(Distance);
          if Distance < 1 then
            Result := (1.5 * Distance - 2.5) * Distance * Distance + 1
          else if Distance < 2 then
            Result := (((Distance - 5) * Distance + 8) * Distance - 4) * -0.5;
        end;
      irfLanczos:
        if (Distance >= -3) and (Distance < 3) then
          Result := Sinc(Distance) * Sinc(Distance / 3);
    end;
  end;

  // PIL's clip8 of a fixed-point sum: round half up, clamp to 0..255.
  function ClipToByte(Value: double): double;
  begin
    if Value <= 0 then exit(0);
    if Value >= 255 then exit(255);
    Result := Floor(Value + 0.5);
  end;

  // PIL's precompute_coeffs for an InSize -> OutSize axis; NEAREST
  // (ImagingScaleAffine) is one tap at an accumulated source coordinate.
  procedure BuildAxis(InSize, OutSize: integer; var Axis: TResampleAxis);
  var
    Scale, FilterScale, InvFilterScale, Support, Center, Total: double;
    OutPos, MaxOutPos, Last, WeightPos, AxisTapPos, MaxAxisTapPos: integer;
  begin
    Scale := InSize / OutSize;
    MaxOutPos := OutSize - 1;
    SetLength(Axis.First, OutSize);
    SetLength(Axis.Count, OutSize);
    if Filter = irfNearest then
    begin
      Axis.MaxTaps := 1;
      SetLength(Axis.Weight, OutSize);
      Center := Scale * 0.5;
      for OutPos := 0 to MaxOutPos do
      begin
        Axis.First[OutPos] := Min(Trunc(Center), InSize - 1);
        Axis.Count[OutPos] := 1;
        Axis.Weight[OutPos] := 1;
        Center := Center + Scale;
      end;
      exit;
    end;
    FilterScale := Scale;
    if FilterScale < 1 then FilterScale := 1;
    InvFilterScale := 1 / FilterScale;
    case Filter of
      irfBox: Support := 0.5;
      irfBilinear, irfHamming: Support := 1;
      irfBicubic: Support := 2;
    else
      Support := 3;
    end;
    Support := Support * FilterScale;
    Axis.MaxTaps := Ceil(Support) * 2 + 1;
    SetLength(Axis.Weight, OutSize * Axis.MaxTaps);
    for OutPos := 0 to MaxOutPos do
    begin
      Center := (OutPos + 0.5) * Scale;
      Axis.First[OutPos] := Max(Trunc(Center - Support + 0.5), 0);
      Last := Min(Trunc(Center + Support + 0.5), InSize);
      Axis.Count[OutPos] := Last - Axis.First[OutPos];
      WeightPos := OutPos * Axis.MaxTaps;
      MaxAxisTapPos := Axis.Count[OutPos] - 1;
      Total := 0;
      for AxisTapPos := 0 to MaxAxisTapPos do
      begin
        Axis.Weight[WeightPos + AxisTapPos] := FilterTap((AxisTapPos +
          Axis.First[OutPos] - Center + 0.5) * InvFilterScale);
        Total := Total + Axis.Weight[WeightPos + AxisTapPos];
      end;
      if Total <> 0 then
        for AxisTapPos := 0 to MaxAxisTapPos do
          Axis.Weight[WeightPos + AxisTapPos] :=
            Axis.Weight[WeightPos + AxisTapPos] / Total;
      // PIL's 8-bit path: weights in 22-bit fixed point, (int)(w*2^22 +- 0.5).
      if RoundToBytes then
        for AxisTapPos := 0 to MaxAxisTapPos do
          Axis.Weight[WeightPos + AxisTapPos] :=
            Trunc(Axis.Weight[WeightPos + AxisTapPos] * 4194304 +
            Sign(Axis.Weight[WeightPos + AxisTapPos]) * 0.5) / 4194304;
    end;
  end;

begin
  if (NewSizeX = Source.SizeX) and (NewSizeY = Source.SizeY) then
  begin
    if Source <> Dest then Dest.Copy(Source);
    exit;
  end;
  ChannelCount := Source.Depth;
  SourceSizeY := Source.SizeY;
  BuildAxis(Source.SizeX, NewSizeX, AxisX);
  BuildAxis(SourceSizeY, NewSizeY, AxisY);
  RowStride := NewSizeX * ChannelCount;
  SourceRowStride := Source.SizeX * ChannelCount;
  SetLength(Rows, RowStride * SourceSizeY);
  MaxX := NewSizeX - 1;
  MaxSourceY := SourceSizeY - 1;
  MaxChannel := ChannelCount - 1;
  for Y := 0 to MaxSourceY do
    for X := 0 to MaxX do
    begin
      TapBase := X * AxisX.MaxTaps;
      MaxTapPos := AxisX.Count[X] - 1;
      FirstPos := Y * SourceRowStride + AxisX.First[X] * ChannelCount;
      DestPos := Y * RowStride + X * ChannelCount;
      for Channel := 0 to MaxChannel do
      begin
        Sum := 0;
        SourcePos := FirstPos + Channel;
        for TapPos := 0 to MaxTapPos do
        begin
          Sum := Sum + AxisX.Weight[TapBase + TapPos] *
            Source.FData[SourcePos];
          Inc(SourcePos, ChannelCount);
        end;
        if RoundToBytes then Sum := ClipToByte(Sum);
        Rows[DestPos + Channel] := Sum;
      end;
    end;
  // Source is fully read: Dest may be the same volume.
  Dest.ReSize(NewSizeX, NewSizeY, ChannelCount);
  MaxY := NewSizeY - 1;
  MaxElementPos := RowStride - 1;
  for Y := 0 to MaxY do
  begin
    TapBase := Y * AxisY.MaxTaps;
    MaxTapPos := AxisY.Count[Y] - 1;
    FirstPos := AxisY.First[Y] * RowStride;
    DestPos := Y * RowStride;
    for ElementPos := 0 to MaxElementPos do
    begin
      Sum := 0;
      SourcePos := FirstPos + ElementPos;
      for TapPos := 0 to MaxTapPos do
      begin
        Sum := Sum + AxisY.Weight[TapBase + TapPos] * Rows[SourcePos];
        Inc(SourcePos, RowStride);
      end;
      if RoundToBytes then Sum := ClipToByte(Sum);
      Dest.FData[DestPos + ElementPos] := Sum;
    end;
  end;
end;


type
  // Past the edge: repeat the edge pixel, mirror without it (d c b | a b c d,
  // torchvision/cv2 reflect-101) or mirror with it (b a | a b c d, scipy "reflect").
  TBlurEdgeMode = (bemClamp, bemReflect101, bemSymmetric);

// Separable blur of Source by Taps (odd count, tap TapPos weighs the pixel
// TapPos - Length(Taps) div 2 away) on both axes; EdgeMode reads past the edge.
procedure BlurImageSeparableTaps(Source, Dest: TNNetVolume;
  const Taps: array of double; EdgeMode: TBlurEdgeMode);
var
  // NeighboursX/Y[Pos * TapCount + TapPos]: the pixel tap TapPos of Pos reads.
  NeighboursX, NeighboursY: array of integer;
  // Source blurred along X only, in the TNNetVolume layout.
  Rows: array of TNeuralFloat;
  TapPos, MaxTapPos, TapCount, TapRadius, SizeX, SizeY, Depth, RowStride: integer;
  X, Y, Channel, MaxX, MaxY, MaxChannel, PixelPos, NeighbourBase: integer;
  Sum: double;

  procedure BuildNeighbours(MaxPos: integer; var Neighbours: array of integer);
  var
    Pos, NeighbourTapPos, Offset, Period: integer;
  begin
    Period := 2 * MaxPos;
    for Pos := 0 to MaxPos do
      for NeighbourTapPos := 0 to MaxTapPos do
      begin
        Offset := Pos + NeighbourTapPos - TapRadius;
        if EdgeMode = bemClamp then Offset := Min(Max(Offset, 0), MaxPos)
        else if EdgeMode = bemSymmetric then
        begin
          if Offset < 0 then Offset := -Offset - 1;
          Offset := Offset mod (Period + 2);
          if Offset > MaxPos then Offset := Period + 1 - Offset;
        end
        else if Period = 0 then Offset := 0
        else
        begin
          Offset := Abs(Offset) mod Period;
          if Offset > MaxPos then Offset := Period - Offset;
        end;
        Neighbours[Pos * TapCount + NeighbourTapPos] := Offset;
      end;
  end;

begin
  TapCount := Length(Taps);
  MaxTapPos := TapCount - 1;
  TapRadius := TapCount div 2;
  SizeX := Source.SizeX;
  SizeY := Source.SizeY;
  Depth := Source.Depth;
  RowStride := SizeX * Depth;
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  MaxChannel := Depth - 1;
  SetLength(NeighboursX, SizeX * TapCount);
  SetLength(NeighboursY, SizeY * TapCount);
  BuildNeighbours(MaxX, NeighboursX);
  BuildNeighbours(MaxY, NeighboursY);
  SetLength(Rows, SizeX * SizeY * Depth);
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      PixelPos := Y * RowStride + X * Depth;
      NeighbourBase := X * TapCount;
      for Channel := 0 to MaxChannel do
      begin
        Sum := 0;
        for TapPos := 0 to MaxTapPos do
          Sum := Sum + Taps[TapPos] * Source.FData[Y * RowStride +
            NeighboursX[NeighbourBase + TapPos] * Depth + Channel];
        Rows[PixelPos + Channel] := Sum;
      end;
    end;
  Dest.ReSize(SizeX, SizeY, Depth);
  for Y := 0 to MaxY do
  begin
    NeighbourBase := Y * TapCount;
    for X := 0 to MaxX do
    begin
      PixelPos := Y * RowStride + X * Depth;
      for Channel := 0 to MaxChannel do
      begin
        Sum := 0;
        for TapPos := 0 to MaxTapPos do
          Sum := Sum + Taps[TapPos] * Rows[NeighboursY[NeighbourBase + TapPos] *
            RowStride + X * Depth + Channel];
        Dest.FData[PixelPos + Channel] := Sum;
      end;
    end;
  end;
end;

// Separable blur of Source by the normalized Gaussian of Sigma over taps
// -TapRadius..TapRadius; EdgeMode reads past the edge.
procedure BlurImageSeparable(Source, Dest: TNNetVolume; Sigma: TNeuralFloat;
  TapRadius: integer; EdgeMode: TBlurEdgeMode);
var
  Taps: array of double;
  TapPos, MaxTapPos: integer;
  Distance, TapSum: double;
begin
  MaxTapPos := 2 * TapRadius;
  SetLength(Taps, MaxTapPos + 1);
  TapSum := 0;
  for TapPos := 0 to MaxTapPos do
  begin
    Distance := TapPos - TapRadius;
    Taps[TapPos] := Exp(-Distance * Distance / (2 * Sigma * Sigma));
    TapSum := TapSum + Taps[TapPos];
  end;
  for TapPos := 0 to MaxTapPos do
    Taps[TapPos] := Taps[TapPos] / TapSum;
  BlurImageSeparableTaps(Source, Dest, Taps, EdgeMode);
end;

procedure GaussianBlurImageCV2(Source, Dest: TNNetVolume; KernelSize: integer);
const
  // cv2 getGaussianKernel's tables (x 256) for 1 to 9 taps when sigma <= 0.
  csSmallTaps: array[0..4, 0..8] of double = (
    (256, 0, 0, 0, 0, 0, 0, 0, 0),
    (64, 128, 64, 0, 0, 0, 0, 0, 0),
    (16, 64, 96, 64, 16, 0, 0, 0, 0),
    (8, 28, 56, 72, 56, 28, 8, 0, 0),
    (4, 13, 30, 51, 60, 51, 30, 13, 4));
var
  Taps: array of double;
  TapPos, MaxTapPos: integer;
begin
  if (KernelSize < 1) or not Odd(KernelSize) then
    raise EArgumentException.Create('GaussianBlurImageCV2: KernelSize must ' +
      'be odd and >= 1.');
  if KernelSize = 1 then
  begin
    if Dest <> Source then Dest.Copy(Source);
    exit;
  end;
  if KernelSize > 9 then
  begin
    BlurImageSeparable(Source, Dest, 0.3 * ((KernelSize - 1) * 0.5 - 1) + 0.8,
      KernelSize div 2, bemReflect101);
    exit;
  end;
  MaxTapPos := KernelSize - 1;
  SetLength(Taps, KernelSize);
  for TapPos := 0 to MaxTapPos do
    Taps[TapPos] := csSmallTaps[KernelSize div 2, TapPos] / 256;
  BlurImageSeparableTaps(Source, Dest, Taps, bemReflect101);
end;

procedure ErodeImage(Source, Dest: TNNetVolume; KernelSizeX,
  KernelSizeY: integer);
var
  // Source eroded along X only, in the TNNetVolume layout.
  Rows: array of TNeuralFloat;
  SizeX, SizeY, Depth, RowStride, X, Y, MaxX, MaxY, MaxElementPos: integer;
  FirstX, LastX, FirstY, LastY, NeighbourPos, MaxNeighbourPos: integer;
  ElementPos, AnchorX, AnchorY, PixelPos: integer;
  MinValue: TNeuralFloat;
begin
  if (KernelSizeX < 0) or (KernelSizeY < 0) then
    raise EArgumentException.Create('ErodeImage: negative kernel size.');
  if (KernelSizeX = 0) or (KernelSizeY = 0) then
  begin
    KernelSizeX := 3;
    KernelSizeY := 3;
  end;
  SizeX := Source.SizeX;
  SizeY := Source.SizeY;
  Depth := Source.Depth;
  RowStride := SizeX * Depth;
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  MaxElementPos := Depth - 1;
  AnchorX := KernelSizeX div 2;
  AnchorY := KernelSizeY div 2;
  SetLength(Rows, Source.Size);
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      FirstX := Max(X - AnchorX, 0);
      LastX := Min(X - AnchorX + KernelSizeX - 1, MaxX);
      PixelPos := Y * RowStride + X * Depth;
      for ElementPos := 0 to MaxElementPos do
      begin
        MinValue := Source.FData[Y * RowStride + FirstX * Depth + ElementPos];
        MaxNeighbourPos := Y * RowStride + LastX * Depth + ElementPos;
        NeighbourPos := Y * RowStride + (FirstX + 1) * Depth + ElementPos;
        while NeighbourPos <= MaxNeighbourPos do
        begin
          if Source.FData[NeighbourPos] < MinValue then
            MinValue := Source.FData[NeighbourPos];
          Inc(NeighbourPos, Depth);
        end;
        Rows[PixelPos + ElementPos] := MinValue;
      end;
    end;
  Dest.ReSize(SizeX, SizeY, Depth);
  for Y := 0 to MaxY do
  begin
    FirstY := Max(Y - AnchorY, 0);
    LastY := Min(Y - AnchorY + KernelSizeY - 1, MaxY);
    for X := 0 to MaxX do
    begin
      PixelPos := Y * RowStride + X * Depth;
      for ElementPos := 0 to MaxElementPos do
      begin
        MinValue := Rows[FirstY * RowStride + X * Depth + ElementPos];
        MaxNeighbourPos := LastY * RowStride + X * Depth + ElementPos;
        NeighbourPos := (FirstY + 1) * RowStride + X * Depth + ElementPos;
        while NeighbourPos <= MaxNeighbourPos do
        begin
          if Rows[NeighbourPos] < MinValue then MinValue := Rows[NeighbourPos];
          Inc(NeighbourPos, RowStride);
        end;
        Dest.FData[PixelPos + ElementPos] := MinValue;
      end;
    end;
  end;
end;

var
  // cv2 drawing.cpp SinTable: sin of 0..450 degrees as 7-decimal floats.
  CV2SinTable: array[0..450] of single;

procedure BuildCV2SinTable();
var
  Degree: integer;
  SinValue: double;
begin
  for Degree := 0 to High(CV2SinTable) do
  begin
    SinValue := Round(Sin(Degree * Pi / 180) * 1e7);
    CV2SinTable[Degree] := SinValue / 1e7;
  end;
end;

procedure CheckDrawColor(Image: TNNetVolume; const Color: array of TNeuralFloat;
  const Caller: string);
begin
  if Length(Color) <> Image.Depth then
    raise EArgumentException.Create(Caller + ': Color needs one value per ' +
      'channel of Image.');
end;

// Image row Y, columns FirstX..LastX (already inside the image) := Color.
procedure FillImageRowSpan(Image: TNNetVolume; Y, FirstX, LastX: integer;
  const Color: array of TNeuralFloat);
var
  X, ChannelPos, MaxChannelPos, PixelPos, Depth: integer;
begin
  Depth := Image.Depth;
  MaxChannelPos := Depth - 1;
  PixelPos := Image.GetRawPos(FirstX, Y);
  for X := FirstX to LastX do
  begin
    for ChannelPos := 0 to MaxChannelPos do
      Image.FData[PixelPos + ChannelPos] := Color[ChannelPos];
    Inc(PixelPos, Depth);
  end;
end;

function EllipsePolygonCV2(CenterX, CenterY, AxisX, AxisY, Angle: integer;
  Delta: integer): TImagePointArray;
var
  CosAngle, SinAngle: single;
  ArcAngle, PointCount: integer;
  EllipseX, EllipseY: double;
  Point, PrevPoint: TPoint;
begin
  if (Delta <= 0) or (Delta > 180) then
    raise EArgumentException.Create('EllipsePolygonCV2: Delta must be 1..180.');
  while Angle < 0 do Inc(Angle, 360);
  while Angle > 360 do Dec(Angle, 360);
  CosAngle := CV2SinTable[450 - Angle];
  SinAngle := CV2SinTable[Angle];
  SetLength(Result, 360 div Delta + 2);
  PointCount := 0;
  PrevPoint.X := Low(integer);
  PrevPoint.Y := Low(integer);
  ArcAngle := 0;
  while ArcAngle < 360 + Delta do
  begin
    EllipseX := AxisX * double(CV2SinTable[450 - Min(ArcAngle, 360)]);
    EllipseY := AxisY * double(CV2SinTable[Min(ArcAngle, 360)]);
    Point.X := Round(CenterX + EllipseX * CosAngle - EllipseY * SinAngle);
    Point.Y := Round(CenterY + EllipseX * SinAngle + EllipseY * CosAngle);
    if (Point.X <> PrevPoint.X) or (Point.Y <> PrevPoint.Y) then
    begin
      Result[PointCount] := Point;
      Inc(PointCount);
      PrevPoint := Point;
    end;
    Inc(ArcAngle, Delta);
  end;
  if PointCount = 1 then
  begin
    Result[0].X := CenterX;
    Result[0].Y := CenterY;
    Result[1] := Result[0];
    PointCount := 2;
  end;
  SetLength(Result, PointCount);
end;

// cv2.clipLine to [0, SizeX - 1] x [0, SizeY - 1]; false when nothing is left.
function ClipLineCV2(SizeX, SizeY: integer; var X1, Y1, X2, Y2: integer): boolean;
var
  Code1, Code2, Edge, Right, Bottom: integer;

  function OutCode(X, Y: integer): integer;
  begin
    Result := Ord(X < 0) + Ord(X > Right) * 2 + Ord(Y < 0) * 4 +
      Ord(Y > Bottom) * 8;
  end;

begin
  Right := SizeX - 1;
  Bottom := SizeY - 1;
  Code1 := OutCode(X1, Y1);
  Code2 := OutCode(X2, Y2);
  if ((Code1 and Code2) = 0) and ((Code1 or Code2) <> 0) then
  begin
    if (Code1 and 12) <> 0 then
    begin
      if Code1 < 8 then Edge := 0 else Edge := Bottom;
      X1 := X1 + Trunc(double(Edge - Y1) * (X2 - X1) / (Y2 - Y1));
      Y1 := Edge;
      Code1 := Ord(X1 < 0) + Ord(X1 > Right) * 2;
    end;
    if (Code2 and 12) <> 0 then
    begin
      if Code2 < 8 then Edge := 0 else Edge := Bottom;
      X2 := X2 + Trunc(double(Edge - Y2) * (X2 - X1) / (Y2 - Y1));
      Y2 := Edge;
      Code2 := Ord(X2 < 0) + Ord(X2 > Right) * 2;
    end;
    if ((Code1 and Code2) = 0) and ((Code1 or Code2) <> 0) then
    begin
      if Code1 <> 0 then
      begin
        if Code1 = 1 then Edge := 0 else Edge := Right;
        Y1 := Y1 + Trunc(double(Edge - X1) * (Y2 - Y1) / (X2 - X1));
        X1 := Edge;
        Code1 := 0;
      end;
      if Code2 <> 0 then
      begin
        if Code2 = 1 then Edge := 0 else Edge := Right;
        Y2 := Y2 + Trunc(double(Edge - X2) * (Y2 - Y1) / (X2 - X1));
        X2 := Edge;
        Code2 := 0;
      end;
    end;
  end;
  Result := (Code1 or Code2) = 0;
end;

// DrawLineCV2 without the Color check.
procedure DrawLineCV2Unchecked(Image: TNNetVolume; X1, Y1, X2, Y2: integer;
  const Color: array of TNeuralFloat);
var
  DeltaMajor, DeltaMinor, StepX, StepY, Error, X, Y, Swap, PixelCount,
    PixelPos, ChannelPos, MaxChannelPos: integer;
  IsSteep, MinorStep: boolean;
begin
  if (X1 < 0) or (X1 >= Image.SizeX) or (X2 < 0) or (X2 >= Image.SizeX) or
    (Y1 < 0) or (Y1 >= Image.SizeY) or (Y2 < 0) or (Y2 >= Image.SizeY) then
    if not ClipLineCV2(Image.SizeX, Image.SizeY, X1, Y1, X2, Y2) then exit;
  if X2 < X1 then
  begin
    Swap := X1; X1 := X2; X2 := Swap;
    Swap := Y1; Y1 := Y2; Y2 := Swap;
  end;
  DeltaMajor := X2 - X1;
  DeltaMinor := Y2 - Y1;
  StepX := 1;
  StepY := 1;
  if DeltaMinor < 0 then
  begin
    DeltaMinor := -DeltaMinor;
    StepY := -1;
  end;
  IsSteep := DeltaMinor > DeltaMajor;
  if IsSteep then
  begin
    Swap := DeltaMajor; DeltaMajor := DeltaMinor; DeltaMinor := Swap;
  end;
  Error := DeltaMajor - 2 * DeltaMinor;
  X := X1;
  Y := Y1;
  MaxChannelPos := Image.Depth - 1;
  for PixelCount := 0 to DeltaMajor do
  begin
    PixelPos := Image.GetRawPos(X, Y);
    for ChannelPos := 0 to MaxChannelPos do
      Image.FData[PixelPos + ChannelPos] := Color[ChannelPos];
    MinorStep := Error < 0;
    Dec(Error, 2 * DeltaMinor);
    if MinorStep then Inc(Error, 2 * DeltaMajor);
    if IsSteep then
    begin
      Inc(Y, StepY);
      if MinorStep then Inc(X, StepX);
    end
    else
    begin
      Inc(X, StepX);
      if MinorStep then Inc(Y, StepY);
    end;
  end;
end;

procedure DrawLineCV2(Image: TNNetVolume; X1, Y1, X2, Y2: integer;
  const Color: array of TNeuralFloat);
begin
  CheckDrawColor(Image, Color, 'DrawLineCV2');
  DrawLineCV2Unchecked(Image, X1, Y1, X2, Y2, Color);
end;

procedure FillConvexPolygonCV2(Image: TNNetVolume;
  const Points: TImagePointArray; const Color: array of TNeuralFloat);
const
  csFixedShift = 16;
  csFixedHalf = 1 shl (csFixedShift - 1);
type
  // One side of the polygon walked down from the top vertex, x in 16.16.
  TPolygonEdge = record
    PointPos, PointStep, EndY: integer;
    X, StepX: int64;
  end;
var
  Edges: array[0..1] of TPolygonEdge;
  PointCount, PointPos, MaxPointPos, TopPos, PrevPos, EdgePos, NextPos,
    EdgesLeft, NextY, Y, MinX, MaxX, MinY, MaxY, LeftEdge, FirstX, LastX: integer;
  StartX, EndX: int64;
begin
  CheckDrawColor(Image, Color, 'FillConvexPolygonCV2');
  PointCount := Length(Points);
  if PointCount = 0 then exit;
  MaxPointPos := PointCount - 1;
  MinX := Points[0].X;
  MaxX := MinX;
  MinY := Points[0].Y;
  MaxY := MinY;
  TopPos := 0;
  PrevPos := MaxPointPos;
  for PointPos := 0 to MaxPointPos do
  begin
    if Points[PointPos].Y < MinY then
    begin
      MinY := Points[PointPos].Y;
      TopPos := PointPos;
    end;
    MaxY := Max(MaxY, Points[PointPos].Y);
    MaxX := Max(MaxX, Points[PointPos].X);
    MinX := Min(MinX, Points[PointPos].X);
    DrawLineCV2Unchecked(Image, Points[PrevPos].X, Points[PrevPos].Y,
      Points[PointPos].X, Points[PointPos].Y, Color);
    PrevPos := PointPos;
  end;
  if (PointCount < 3) or (MaxX < 0) or (MaxY < 0) or (MinX >= Image.SizeX) or
    (MinY >= Image.SizeY) then exit;
  MaxY := Min(MaxY, Image.SizeY - 1);
  for EdgePos := 0 to 1 do
  begin
    Edges[EdgePos].PointPos := TopPos;
    Edges[EdgePos].EndY := MinY;
    Edges[EdgePos].X := -(int64(1) shl csFixedShift);
    Edges[EdgePos].StepX := 0;
  end;
  Edges[0].PointStep := 1;
  Edges[1].PointStep := PointCount - 1;
  EdgesLeft := PointCount;
  Y := MinY;
  repeat
    for EdgePos := 0 to 1 do
    begin
      if Y < Edges[EdgePos].EndY then continue;
      PrevPos := Edges[EdgePos].PointPos;
      NextPos := (PrevPos + Edges[EdgePos].PointStep) mod PointCount;
      while true do
      begin
        Dec(EdgesLeft);
        if EdgesLeft < 0 then break;
        NextY := Points[NextPos].Y;
        if NextY > Y then
        begin
          StartX := int64(Points[PrevPos].X) shl csFixedShift;
          EndX := int64(Points[NextPos].X) shl csFixedShift;
          Edges[EdgePos].EndY := NextY;
          Edges[EdgePos].StepX := ((EndX - StartX) * 2 + (NextY - Y)) div
            (2 * int64(NextY - Y));
          Edges[EdgePos].X := StartX;
          Edges[EdgePos].PointPos := NextPos;
          break;
        end;
        PrevPos := NextPos;
        NextPos := (NextPos + Edges[EdgePos].PointStep) mod PointCount;
      end;
    end;
    if EdgesLeft < 0 then break;
    if Y >= 0 then
    begin
      if Edges[0].X > Edges[1].X then LeftEdge := 1 else LeftEdge := 0;
      FirstX := SarInt64(Edges[LeftEdge].X + csFixedHalf, csFixedShift);
      LastX := SarInt64(Edges[1 - LeftEdge].X + csFixedHalf, csFixedShift);
      if (LastX >= 0) and (FirstX < Image.SizeX) then
        FillImageRowSpan(Image, Y, Max(FirstX, 0), Min(LastX, Image.SizeX - 1),
          Color);
    end;
    Inc(Edges[0].X, Edges[0].StepX);
    Inc(Edges[1].X, Edges[1].StepX);
    Inc(Y);
  until Y > MaxY;
end;

procedure FillCircleCV2(Image: TNNetVolume; CenterX, CenterY, Radius: integer;
  const Color: array of TNeuralFloat);
var
  Error, OffsetX, OffsetY, Plus, Minus, SizeX, SizeY: integer;

  procedure FillClippedRow(Y, FirstX, LastX: integer);
  begin
    if (Y < 0) or (Y >= SizeY) then exit;
    FirstX := Max(FirstX, 0);
    LastX := Min(LastX, SizeX - 1);
    if FirstX <= LastX then FillImageRowSpan(Image, Y, FirstX, LastX, Color);
  end;

begin
  CheckDrawColor(Image, Color, 'FillCircleCV2');
  SizeX := Image.SizeX;
  SizeY := Image.SizeY;
  Error := 0;
  OffsetX := Radius;
  OffsetY := 0;
  Plus := 1;
  Minus := 2 * Radius - 1;
  while OffsetX >= OffsetY do
  begin
    if (CenterX - OffsetX < SizeX) and (CenterX + OffsetX >= 0) and
      (CenterY - OffsetX < SizeY) and (CenterY + OffsetX >= 0) then
    begin
      FillClippedRow(CenterY - OffsetY, CenterX - OffsetX, CenterX + OffsetX);
      FillClippedRow(CenterY + OffsetY, CenterX - OffsetX, CenterX + OffsetX);
      if (CenterX - OffsetY < SizeX) and (CenterX + OffsetY >= 0) then
      begin
        FillClippedRow(CenterY - OffsetX, CenterX - OffsetY, CenterX + OffsetY);
        FillClippedRow(CenterY + OffsetX, CenterX - OffsetY, CenterX + OffsetY);
      end;
    end;
    Inc(OffsetY);
    Inc(Error, Plus);
    Inc(Plus, 2);
    if Error > 0 then
    begin
      Dec(Error, Minus);
      Dec(OffsetX);
      Dec(Minus, 2);
    end;
  end;
end;

// FillRectangleCV2 without the Color check.
procedure FillRectangleUnchecked(Image: TNNetVolume; X1, Y1, X2, Y2: integer;
  const Color: array of TNeuralFloat);
var
  Swap, Y: integer;
begin
  if X2 < X1 then
  begin
    Swap := X1; X1 := X2; X2 := Swap;
  end;
  if Y2 < Y1 then
  begin
    Swap := Y1; Y1 := Y2; Y2 := Swap;
  end;
  X1 := Max(X1, 0);
  Y1 := Max(Y1, 0);
  X2 := Min(X2, Image.SizeX - 1);
  Y2 := Min(Y2, Image.SizeY - 1);
  if X1 > X2 then exit;
  for Y := Y1 to Y2 do FillImageRowSpan(Image, Y, X1, X2, Color);
end;

procedure FillRectangleCV2(Image: TNNetVolume; X1, Y1, X2, Y2: integer;
  const Color: array of TNeuralFloat);
begin
  CheckDrawColor(Image, Color, 'FillRectangleCV2');
  FillRectangleUnchecked(Image, X1, Y1, X2, Y2, Color);
end;

procedure DrawRectangleOutline(Image: TNNetVolume; X1, Y1, X2, Y2: integer;
  const Color: array of TNeuralFloat; Thickness: integer);
var
  Left, Top, Right, Bottom, MaxRingPos, RingPos: integer;
begin
  CheckDrawColor(Image, Color, 'DrawRectangleOutline');
  Left := Min(X1, X2);
  Right := Max(X1, X2);
  Top := Min(Y1, Y2);
  Bottom := Max(Y1, Y2);
  MaxRingPos := Thickness - 1;
  for RingPos := 0 to MaxRingPos do
  begin
    if (Left > Right) or (Top > Bottom) then exit;
    FillRectangleUnchecked(Image, Left, Top, Right, Top, Color);
    FillRectangleUnchecked(Image, Left, Bottom, Right, Bottom, Color);
    FillRectangleUnchecked(Image, Left, Top, Left, Bottom, Color);
    FillRectangleUnchecked(Image, Right, Top, Right, Bottom, Color);
    Inc(Left);
    Inc(Top);
    Dec(Right);
    Dec(Bottom);
  end;
end;

const
  csBitmapFontCharacters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ .,:-_%#/()!?+=''';
  csBitmapFontGlyphCount = Length(csBitmapFontCharacters);
  csBitmapFontUnknownGlyphPos = 48; // '?'
  // One byte per glyph row, top row first; bit 4 is the left column.
  csBitmapFontRows: array[0..csBitmapFontGlyphCount - 1,
    0..csBitmapFontGlyphHeight - 1] of byte = (
    ($0E, $11, $13, $15, $19, $11, $0E), ($04, $0C, $04, $04, $04, $04, $0E),
    ($0E, $11, $01, $02, $04, $08, $1F), ($1F, $02, $04, $02, $01, $11, $0E),
    ($02, $06, $0A, $12, $1F, $02, $02), ($1F, $10, $1E, $01, $01, $11, $0E),
    ($06, $08, $10, $1E, $11, $11, $0E), ($1F, $01, $02, $04, $08, $08, $08),
    ($0E, $11, $11, $0E, $11, $11, $0E), ($0E, $11, $11, $0F, $01, $02, $0C),
    ($0E, $11, $11, $11, $1F, $11, $11), ($1E, $11, $11, $1E, $11, $11, $1E),
    ($0E, $11, $10, $10, $10, $11, $0E), ($1C, $12, $11, $11, $11, $12, $1C),
    ($1F, $10, $10, $1E, $10, $10, $1F), ($1F, $10, $10, $1E, $10, $10, $10),
    ($0E, $11, $10, $17, $11, $11, $0F), ($11, $11, $11, $1F, $11, $11, $11),
    ($0E, $04, $04, $04, $04, $04, $0E), ($07, $02, $02, $02, $02, $12, $0C),
    ($11, $12, $14, $18, $14, $12, $11), ($10, $10, $10, $10, $10, $10, $1F),
    ($11, $1B, $15, $15, $11, $11, $11), ($11, $11, $19, $15, $13, $11, $11),
    ($0E, $11, $11, $11, $11, $11, $0E), ($1E, $11, $11, $1E, $10, $10, $10),
    ($0E, $11, $11, $11, $15, $12, $0D), ($1E, $11, $11, $1E, $14, $12, $11),
    ($0F, $10, $10, $0E, $01, $01, $1E), ($1F, $04, $04, $04, $04, $04, $04),
    ($11, $11, $11, $11, $11, $11, $0E), ($11, $11, $11, $11, $11, $0A, $04),
    ($11, $11, $11, $15, $15, $15, $0A), ($11, $11, $0A, $04, $0A, $11, $11),
    ($11, $11, $11, $0A, $04, $04, $04), ($1F, $01, $02, $04, $08, $10, $1F),
    ($00, $00, $00, $00, $00, $00, $00), ($00, $00, $00, $00, $00, $0C, $0C),
    ($00, $00, $00, $00, $0C, $04, $08), ($00, $0C, $0C, $00, $0C, $0C, $00),
    ($00, $00, $00, $1F, $00, $00, $00), ($00, $00, $00, $00, $00, $00, $1F),
    ($18, $19, $02, $04, $08, $13, $03), ($0A, $0A, $1F, $0A, $1F, $0A, $0A),
    ($00, $01, $02, $04, $08, $10, $00), ($02, $04, $08, $08, $08, $04, $02),
    ($08, $04, $02, $02, $02, $04, $08), ($04, $04, $04, $04, $00, $00, $04),
    ($0E, $11, $01, $02, $04, $00, $04), ($00, $04, $04, $1F, $04, $04, $00),
    ($00, $00, $1F, $00, $1F, $00, $00), ($0C, $04, $08, $00, $00, $00, $00));

// Glyph of Character: lower case as upper case, a few look-alikes, else '?'.
function BitmapFontGlyphPos(Character: char): integer;
const
  csLookAlikeFrom = '&[{<]}>"`;|\';
  csLookAlikeTo = '+((()))''''://';
var
  LookAlikePos: integer;
begin
  Result := Pos(UpCase(Character), csBitmapFontCharacters) - 1;
  if Result >= 0 then exit;
  LookAlikePos := Pos(Character, csLookAlikeFrom);
  if LookAlikePos > 0 then
    Result := Pos(csLookAlikeTo[LookAlikePos], csBitmapFontCharacters) - 1
  else
    Result := csBitmapFontUnknownGlyphPos;
end;

// A UTF-8 continuation byte belongs to the glyph of its lead byte.
function IsUTF8ContinuationByte(Character: char): boolean;
  {$IFDEF Release} inline; {$ENDIF}
begin
  Result := (Ord(Character) and $C0) = $80;
end;

function BitmapTextWidth(const Text: string; Scale: integer): integer;
var
  CharPos, GlyphCount: integer;
begin
  if Scale < 1 then
    raise EArgumentException.Create('BitmapTextWidth: Scale must be >= 1.');
  GlyphCount := 0;
  for CharPos := 1 to Length(Text) do
    if not IsUTF8ContinuationByte(Text[CharPos]) then Inc(GlyphCount);
  if GlyphCount = 0 then exit(0);
  Result := ((csBitmapFontGlyphWidth + 1) * GlyphCount - 1) * Scale;
end;

function DrawBitmapText(Image: TNNetVolume; Left, Top: integer;
  const Text: string; const Color: array of TNeuralFloat;
  Scale: integer): integer;
var
  CharPos, GlyphPos, RowPos, ColumnPos, RowBits, GlyphLeft, CellLeft,
    CellTop: integer;
begin
  CheckDrawColor(Image, Color, 'DrawBitmapText');
  Result := BitmapTextWidth(Text, Scale);
  GlyphLeft := Left;
  for CharPos := 1 to Length(Text) do
  begin
    if IsUTF8ContinuationByte(Text[CharPos]) then continue;
    GlyphPos := BitmapFontGlyphPos(Text[CharPos]);
    for RowPos := 0 to csBitmapFontGlyphHeight - 1 do
    begin
      RowBits := csBitmapFontRows[GlyphPos, RowPos];
      CellTop := Top + RowPos * Scale;
      for ColumnPos := 0 to csBitmapFontGlyphWidth - 1 do
        if (RowBits shr (csBitmapFontGlyphWidth - 1 - ColumnPos)) and 1 = 1 then
        begin
          CellLeft := GlyphLeft + ColumnPos * Scale;
          FillRectangleUnchecked(Image, CellLeft, CellTop, CellLeft + Scale - 1,
            CellTop + Scale - 1, Color);
        end;
    end;
    Inc(GlyphLeft, (csBitmapFontGlyphWidth + 1) * Scale);
  end;
end;

procedure ResizeImageLanczos4CV2(Source, Dest: TNNetVolume; NewSizeX,
  NewSizeY: integer);
const
  csTapCount = 8;
  csMaxTapPos = csTapCount - 1;
type
  // Per destination column or row: the 8 source indices and their weights.
  TLanczos4Axis = record
    Index: array of integer;
    Weight: array of double;
  end;
var
  AxisX, AxisY: TLanczos4Axis;
  // Source resized along X only, (NewSizeX, SizeY, Depth).
  Rows: array of TNeuralFloat;
  SizeX, SizeY, Depth, X, Y, MaxX, MaxY, ElementPos, MaxElementPos: integer;
  TapPos, TapBase, SourceRowStride, DestRowStride, DestPos, ColumnPos: integer;
  Sum: double;

  procedure BuildAxis(SourceSize, DestSize: integer; var Axis: TLanczos4Axis);
  var
    DestPos, MaxDestPos, SourcePos, AxisTapPos: integer;
    Scale, Fraction, Distance, WeightSum: double;
  begin
    SetLength(Axis.Index, DestSize * csTapCount);
    SetLength(Axis.Weight, DestSize * csTapCount);
    Scale := 1 / (DestSize / SourceSize);
    MaxDestPos := DestSize - 1;
    for DestPos := 0 to MaxDestPos do
    begin
      Fraction := single((DestPos + 0.5) * Scale - 0.5);
      SourcePos := Floor(Fraction);
      Fraction := Fraction - SourcePos;
      WeightSum := 0;
      for AxisTapPos := 0 to csMaxTapPos do
      begin
        Axis.Index[DestPos * csTapCount + AxisTapPos] :=
          Min(Max(SourcePos - 3 + AxisTapPos, 0), SourceSize - 1);
        if Fraction < 1.1920929e-07 then
        begin
          if AxisTapPos = 3 then Axis.Weight[DestPos * csTapCount + AxisTapPos] := 1
          else Axis.Weight[DestPos * csTapCount + AxisTapPos] := 0;
        end
        else
        begin
          Distance := (Fraction + 3 - AxisTapPos) * Pi;
          Axis.Weight[DestPos * csTapCount + AxisTapPos] :=
            Sin(Distance) * Sin(Distance * 0.25) / (Distance * Distance);
        end;
        WeightSum := WeightSum + Axis.Weight[DestPos * csTapCount + AxisTapPos];
      end;
      for AxisTapPos := 0 to csMaxTapPos do
        Axis.Weight[DestPos * csTapCount + AxisTapPos] :=
          Axis.Weight[DestPos * csTapCount + AxisTapPos] / WeightSum;
    end;
  end;

begin
  if (NewSizeX < 1) or (NewSizeY < 1) then
    raise EArgumentException.Create('ResizeImageLanczos4CV2: size must be ' +
      'positive.');
  SizeX := Source.SizeX;
  SizeY := Source.SizeY;
  Depth := Source.Depth;
  if (NewSizeX = SizeX) and (NewSizeY = SizeY) then
  begin
    if Dest <> Source then Dest.Copy(Source);
    exit;
  end;
  BuildAxis(SizeX, NewSizeX, AxisX);
  BuildAxis(SizeY, NewSizeY, AxisY);
  SourceRowStride := SizeX * Depth;
  DestRowStride := NewSizeX * Depth;
  MaxX := NewSizeX - 1;
  MaxY := SizeY - 1;
  MaxElementPos := Depth - 1;
  SetLength(Rows, NewSizeX * SizeY * Depth);
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      TapBase := X * csTapCount;
      DestPos := Y * DestRowStride + X * Depth;
      for ElementPos := 0 to MaxElementPos do
      begin
        Sum := 0;
        for TapPos := 0 to csMaxTapPos do
          Sum := Sum + AxisX.Weight[TapBase + TapPos] * Source.FData[Y *
            SourceRowStride + AxisX.Index[TapBase + TapPos] * Depth + ElementPos];
        Rows[DestPos + ElementPos] := Sum;
      end;
    end;
  Dest.ReSize(NewSizeX, NewSizeY, Depth);
  MaxY := NewSizeY - 1;
  for Y := 0 to MaxY do
  begin
    TapBase := Y * csTapCount;
    for X := 0 to MaxX do
    begin
      ColumnPos := X * Depth;
      DestPos := Y * DestRowStride + ColumnPos;
      for ElementPos := 0 to MaxElementPos do
      begin
        Sum := 0;
        for TapPos := 0 to csMaxTapPos do
          Sum := Sum + AxisY.Weight[TapBase + TapPos] *
            Rows[AxisY.Index[TapBase + TapPos] * DestRowStride + ColumnPos +
            ElementPos];
        Dest.FData[DestPos + ElementPos] := Sum;
      end;
    end;
  end;
end;

function InvertImageAffineMatrix(const Matrix: TImageAffineMatrix): TImageAffineMatrix;
var
  Determinant: double;
begin
  Determinant := Matrix[0] * Matrix[4] - Matrix[1] * Matrix[3];
  if Determinant <> 0 then Determinant := 1 / Determinant;
  Result[0] := Matrix[4] * Determinant;
  Result[1] := -Matrix[1] * Determinant;
  Result[3] := -Matrix[3] * Determinant;
  Result[4] := Matrix[0] * Determinant;
  Result[2] := -Result[0] * Matrix[2] - Result[1] * Matrix[5];
  Result[5] := -Result[3] * Matrix[2] - Result[4] * Matrix[5];
end;

function EstimateSimilarityTransform(const SourceX, SourceY, DestX,
  DestY: array of double): TImageAffineMatrix;
var
  PointPos, MaxPointPos, PointCount: integer;
  SourceMeanX, SourceMeanY, DestMeanX, DestMeanY: double;
  CenteredX, CenteredY, Spread, CosTerm, SinTerm, ScaledCos, ScaledSin: double;
begin
  PointCount := Length(SourceX);
  if (PointCount < 2) or (Length(SourceY) <> PointCount) or
     (Length(DestX) <> PointCount) or (Length(DestY) <> PointCount) then
    raise EArgumentException.Create('EstimateSimilarityTransform needs at ' +
      'least 2 points and equal-length arrays.');
  MaxPointPos := PointCount - 1;
  SourceMeanX := 0; SourceMeanY := 0; DestMeanX := 0; DestMeanY := 0;
  for PointPos := 0 to MaxPointPos do
  begin
    SourceMeanX := SourceMeanX + SourceX[PointPos];
    SourceMeanY := SourceMeanY + SourceY[PointPos];
    DestMeanX := DestMeanX + DestX[PointPos];
    DestMeanY := DestMeanY + DestY[PointPos];
  end;
  SourceMeanX := SourceMeanX / PointCount;
  SourceMeanY := SourceMeanY / PointCount;
  DestMeanX := DestMeanX / PointCount;
  DestMeanY := DestMeanY / PointCount;
  Spread := 0; CosTerm := 0; SinTerm := 0;
  for PointPos := 0 to MaxPointPos do
  begin
    CenteredX := SourceX[PointPos] - SourceMeanX;
    CenteredY := SourceY[PointPos] - SourceMeanY;
    Spread := Spread + CenteredX * CenteredX + CenteredY * CenteredY;
    CosTerm := CosTerm + CenteredX * (DestX[PointPos] - DestMeanX) +
      CenteredY * (DestY[PointPos] - DestMeanY);
    SinTerm := SinTerm + CenteredX * (DestY[PointPos] - DestMeanY) -
      CenteredY * (DestX[PointPos] - DestMeanX);
  end;
  if Spread = 0 then
    raise EArgumentException.Create('EstimateSimilarityTransform: the source ' +
      'points coincide.');
  ScaledCos := CosTerm / Spread;
  ScaledSin := SinTerm / Spread;
  Result[0] := ScaledCos;
  Result[1] := -ScaledSin;
  Result[2] := DestMeanX - ScaledCos * SourceMeanX + ScaledSin * SourceMeanY;
  Result[3] := ScaledSin;
  Result[4] := ScaledCos;
  Result[5] := DestMeanY - ScaledSin * SourceMeanX - ScaledCos * SourceMeanY;
end;

function EstimateSimilarityTransformLMedS(const SourceX, SourceY, DestX,
  DestY: array of double): TImageAffineMatrix;
var
  Candidate: TImageAffineMatrix;
  // SquaredErrors[PointPos]: the squared residual of point PointPos (cv2 keeps float).
  SquaredErrors, SortedErrors: array of single;
  InlierSourceX, InlierSourceY, InlierDestX, InlierDestY: array of double;
  PointCount, MaxPointPos, FirstPos, SecondPos, PointPos, InlierCount: integer;
  MedianError, BestMedian, Threshold: double;

  procedure ComputeErrors(const Model: TImageAffineMatrix);
  var
    ErrorPos: integer;
  begin
    for ErrorPos := 0 to MaxPointPos do
      SquaredErrors[ErrorPos] := Sqr(Model[0] * SourceX[ErrorPos] +
        Model[1] * SourceY[ErrorPos] + Model[2] - DestX[ErrorPos]) +
        Sqr(Model[3] * SourceX[ErrorPos] + Model[4] * SourceY[ErrorPos] +
        Model[5] - DestY[ErrorPos]);
  end;

  function MedianOfErrors(): double;
  var
    SortPos, InsertPos: integer;
    Value: single;
  begin
    for SortPos := 0 to MaxPointPos do
    begin
      Value := SquaredErrors[SortPos];
      InsertPos := SortPos;
      while (InsertPos > 0) and (SortedErrors[InsertPos - 1] > Value) do
      begin
        SortedErrors[InsertPos] := SortedErrors[InsertPos - 1];
        Dec(InsertPos);
      end;
      SortedErrors[InsertPos] := Value;
    end;
    Result := SortedErrors[PointCount div 2];
  end;

begin
  Result := EstimateSimilarityTransform(SourceX, SourceY, DestX, DestY);
  PointCount := Length(SourceX);
  if PointCount <= 2 then exit;
  MaxPointPos := PointCount - 1;
  SetLength(SquaredErrors, PointCount);
  SetLength(SortedErrors, PointCount);
  BestMedian := MaxDouble;
  for FirstPos := 0 to MaxPointPos - 1 do
    for SecondPos := FirstPos + 1 to MaxPointPos do
    begin
      if (SourceX[FirstPos] = SourceX[SecondPos]) and
         (SourceY[FirstPos] = SourceY[SecondPos]) then continue;
      Candidate := EstimateSimilarityTransform(
        [SourceX[FirstPos], SourceX[SecondPos]],
        [SourceY[FirstPos], SourceY[SecondPos]],
        [DestX[FirstPos], DestX[SecondPos]],
        [DestY[FirstPos], DestY[SecondPos]]);
      ComputeErrors(Candidate);
      MedianError := MedianOfErrors();
      if MedianError < BestMedian then
      begin
        BestMedian := MedianError;
        Result := Candidate;
      end;
    end;
  if BestMedian = MaxDouble then exit;
  // cv2 LMeDSPointSetRegistrator: sigma = 2.5 * 1.4826 * (1 + 5 / (n - 2)) * sqrt(median).
  Threshold := Max(2.5 * 1.4826 * (1 + 5 / (PointCount - 2)) * Sqrt(BestMedian),
    0.001);
  Threshold := single(Threshold * Threshold);
  ComputeErrors(Result);
  SetLength(InlierSourceX, PointCount);
  SetLength(InlierSourceY, PointCount);
  SetLength(InlierDestX, PointCount);
  SetLength(InlierDestY, PointCount);
  InlierCount := 0;
  for PointPos := 0 to MaxPointPos do
    if SquaredErrors[PointPos] <= Threshold then
    begin
      InlierSourceX[InlierCount] := SourceX[PointPos];
      InlierSourceY[InlierCount] := SourceY[PointPos];
      InlierDestX[InlierCount] := DestX[PointPos];
      InlierDestY[InlierCount] := DestY[PointPos];
      Inc(InlierCount);
    end;
  if InlierCount < 2 then exit;
  Result := EstimateSimilarityTransform(Copy(InlierSourceX, 0, InlierCount),
    Copy(InlierSourceY, 0, InlierCount), Copy(InlierDestX, 0, InlierCount),
    Copy(InlierDestY, 0, InlierCount));
end;

procedure WarpImageAffine(Source, Dest: TNNetVolume;
  const Matrix: TImageAffineMatrix; DestSizeX, DestSizeY: integer;
  const BorderValue: array of TNeuralFloat; BlendsBorder: boolean);
var
  Inverse: TImageAffineMatrix;
  // Border[Channel]: what an outside neighbour reads.
  Border: array of TNeuralFloat;
  SizeX, SizeY, Depth, X, Y, MaxX, MaxY, MaxChannel, Channel: integer;
  LeftX, TopY: integer;
  RowSourceX, RowSourceY: double;
  SourceX, SourceY, FractionX, FractionY, InsideWeight: TNeuralFloat;
  OutPtr: TNeuralFloatArrPtr;
begin
  SizeX := Source.SizeX;
  SizeY := Source.SizeY;
  Depth := Source.Depth;
  MaxChannel := Depth - 1;
  SetLength(Border, Depth);
  for Channel := 0 to MaxChannel do
    if Channel < Length(BorderValue) then Border[Channel] := BorderValue[Channel]
    else Border[Channel] := 0;
  Inverse := InvertImageAffineMatrix(Matrix);
  Dest.ReSize(DestSizeX, DestSizeY, Depth);
  MaxX := DestSizeX - 1;
  MaxY := DestSizeY - 1;
  for Y := 0 to MaxY do
  begin
    RowSourceX := Inverse[1] * Y + Inverse[2];
    RowSourceY := Inverse[4] * Y + Inverse[5];
    for X := 0 to MaxX do
    begin
      SourceX := Inverse[0] * X + RowSourceX;
      SourceY := Inverse[3] * X + RowSourceY;
      LeftX := Floor(SourceX);
      TopY := Floor(SourceY);
      OutPtr := Dest.GetRawPtr(X, Y);
      if (LeftX >= 0) and (LeftX < SizeX - 1) and (TopY >= 0) and
         (TopY < SizeY - 1) then
        NeuralBilinearGatherColumn(Source, OutPtr, SourceX, SourceY, SizeX,
          SizeY, Depth, {BorderPad=}false)
      else if not BlendsBorder then
      begin
        if (SourceX < 0) or (SourceX > SizeX - 1) or (SourceY < 0) or
           (SourceY > SizeY - 1) then
        begin
          for Channel := 0 to MaxChannel do OutPtr^[Channel] := Border[Channel];
        end
        else
          NeuralBilinearGatherColumn(Source, OutPtr, SourceX, SourceY, SizeX,
            SizeY, Depth, {BorderPad=}false);
      end
      else if (LeftX < -1) or (LeftX > SizeX - 1) or (TopY < -1) or
         (TopY > SizeY - 1) then
      begin
        for Channel := 0 to MaxChannel do OutPtr^[Channel] := Border[Channel];
      end
      else
      begin
        NeuralBilinearGatherColumn(Source, OutPtr, SourceX, SourceY, SizeX,
          SizeY, Depth, {BorderPad=}false);
        FractionX := SourceX - LeftX;
        FractionY := SourceY - TopY;
        InsideWeight := 0;
        if (TopY >= 0) and (LeftX >= 0) then
          InsideWeight := InsideWeight + (1 - FractionX) * (1 - FractionY);
        if (TopY >= 0) and (LeftX < SizeX - 1) then
          InsideWeight := InsideWeight + FractionX * (1 - FractionY);
        if (TopY < SizeY - 1) and (LeftX >= 0) then
          InsideWeight := InsideWeight + (1 - FractionX) * FractionY;
        if (TopY < SizeY - 1) and (LeftX < SizeX - 1) then
          InsideWeight := InsideWeight + FractionX * FractionY;
        for Channel := 0 to MaxChannel do
          OutPtr^[Channel] := OutPtr^[Channel] + Border[Channel] *
            (1 - InsideWeight);
      end;
    end;
  end;
end;

function ConvexHullOfPoints(const PointX, PointY: array of TNeuralFloat;
  out HullX, HullY: TNeuralFloatDynArr): integer;
var
  Order: TNeuralIntegerArray;
  PointCount, MaxPointPos, PointPos, InsertPos, MovingPos, LowerCount: integer;
  HullCount: integer;

  // Pops the hull tail while it does not turn strictly left at PointPos.
  procedure PushHullPoint(PointPos, MinCount: integer);
  var
    OriginX, OriginY: double;
  begin
    while HullCount >= MinCount do
    begin
      OriginX := HullX[HullCount - 2];
      OriginY := HullY[HullCount - 2];
      if (HullX[HullCount - 1] - OriginX) * (PointY[PointPos] - OriginY) -
        (HullY[HullCount - 1] - OriginY) * (PointX[PointPos] - OriginX) > 0 then
        break;
      Dec(HullCount);
    end;
    HullX[HullCount] := PointX[PointPos];
    HullY[HullCount] := PointY[PointPos];
    Inc(HullCount);
  end;

begin
  PointCount := Length(PointX);
  if Length(PointY) <> PointCount then
    raise Exception.Create('ConvexHullOfPoints: ' + IntToStr(PointCount) +
      ' X and ' + IntToStr(Length(PointY)) + ' Y coordinates.');
  SetLength(HullX, 2 * PointCount);
  SetLength(HullY, 2 * PointCount);
  HullCount := 0;
  if PointCount > 0 then
  begin
    MaxPointPos := PointCount - 1;
    SetLength(Order, PointCount);
    for PointPos := 0 to MaxPointPos do Order[PointPos] := PointPos;
    QuickSortOrderByDist(Order, PointX, 0, MaxPointPos);
    // Ties in X end up in Y order; this pass moves only inside equal-X runs.
    for PointPos := 1 to MaxPointPos do
    begin
      MovingPos := Order[PointPos];
      InsertPos := PointPos;
      while (InsertPos > 0) and (PointX[Order[InsertPos - 1]] = PointX[MovingPos]) and
        (PointY[Order[InsertPos - 1]] > PointY[MovingPos]) do
      begin
        Order[InsertPos] := Order[InsertPos - 1];
        Dec(InsertPos);
      end;
      Order[InsertPos] := MovingPos;
    end;
    for PointPos := 0 to MaxPointPos do PushHullPoint(Order[PointPos], 2);
    LowerCount := HullCount + 1;
    for PointPos := MaxPointPos - 1 downto 0 do
      PushHullPoint(Order[PointPos], LowerCount);
    // The upper chain ends on the first point again.
    if PointCount > 1 then Dec(HullCount);
    if (HullCount = 2) and (HullX[0] = HullX[1]) and (HullY[0] = HullY[1]) then
      HullCount := 1;
  end;
  SetLength(HullX, HullCount);
  SetLength(HullY, HullCount);
  Result := HullCount;
end;

function PolygonArea(const PointX, PointY: array of TNeuralFloat): TNeuralFloat;
var
  PointPos, PreviousPos, MaxPointPos: integer;
  DoubleArea: double;
begin
  DoubleArea := 0;
  MaxPointPos := High(PointX);
  PreviousPos := MaxPointPos;
  for PointPos := 0 to MaxPointPos do
  begin
    DoubleArea := DoubleArea + double(PointX[PreviousPos]) * PointY[PointPos] -
      double(PointX[PointPos]) * PointY[PreviousPos];
    PreviousPos := PointPos;
  end;
  Result := Abs(DoubleArea) * 0.5;
end;

function PolygonPerimeter(const PointX, PointY: array of TNeuralFloat): TNeuralFloat;
var
  PointPos, PreviousPos, MaxPointPos: integer;
  BoundaryLength: double;
begin
  BoundaryLength := 0;
  MaxPointPos := High(PointX);
  PreviousPos := MaxPointPos;
  for PointPos := 0 to MaxPointPos do
  begin
    BoundaryLength := BoundaryLength + Hypot(PointX[PointPos] - PointX[PreviousPos],
      PointY[PointPos] - PointY[PreviousPos]);
    PreviousPos := PointPos;
  end;
  Result := BoundaryLength;
end;

// The rectangle centred at (CentreX, CentreY) with sides LengthA along (AxisX, AxisY)
// and LengthB across it, in TImageRotatedRect corner order; a zero axis means X.
function RotatedRectFromAxis(CentreX, CentreY, AxisX, AxisY, LengthA,
  LengthB: double): TImageRotatedRect;
var
  AxisLength, WidthX, WidthY, PreviousWidthX, HalfWidth, HalfHeight: double;
begin
  AxisLength := Hypot(AxisX, AxisY);
  if AxisLength > 0 then
  begin
    WidthX := AxisX / AxisLength;
    WidthY := AxisY / AxisLength;
  end
  else
  begin
    WidthX := 1;
    WidthY := 0;
  end;
  if LengthB > LengthA then
  begin
    PreviousWidthX := WidthX;
    WidthX := -WidthY;
    WidthY := PreviousWidthX;
    HalfWidth := LengthB * 0.5;
    HalfHeight := LengthA * 0.5;
  end
  else
  begin
    HalfWidth := LengthA * 0.5;
    HalfHeight := LengthB * 0.5;
  end;
  if (WidthX < -1e-6) or ((Abs(WidthX) <= 1e-6) and (WidthY < 0)) then
  begin
    WidthX := -WidthX;
    WidthY := -WidthY;
  end;
  // The height axis (-WidthY, WidthX) points down for a left-to-right width axis.
  Result.CornerX[0] := CentreX - WidthX * HalfWidth + WidthY * HalfHeight;
  Result.CornerY[0] := CentreY - WidthY * HalfWidth - WidthX * HalfHeight;
  Result.CornerX[1] := CentreX + WidthX * HalfWidth + WidthY * HalfHeight;
  Result.CornerY[1] := CentreY + WidthY * HalfWidth - WidthX * HalfHeight;
  Result.CornerX[2] := CentreX + WidthX * HalfWidth - WidthY * HalfHeight;
  Result.CornerY[2] := CentreY + WidthY * HalfWidth + WidthX * HalfHeight;
  Result.CornerX[3] := CentreX - WidthX * HalfWidth - WidthY * HalfHeight;
  Result.CornerY[3] := CentreY - WidthY * HalfWidth + WidthX * HalfHeight;
end;

function MinAreaRectOfConvexHull(const HullX,
  HullY: array of TNeuralFloat): TImageRotatedRect;
var
  PointCount, MaxPointPos, EdgePos, PointPos, MaxDotPos, MinDotPos, FarPos: integer;
  EdgeX, EdgeY, EdgeLength, AxisX, AxisY, BaseX, BaseY: double;
  MaxDot, MinDot, FarHeight, Area, BestArea: double;
  BestCentreX, BestCentreY, BestAxisX, BestAxisY, BestLength, BestHeight: double;

  function DotAt(PointPos: integer): double;
  begin
    Result := (HullX[PointPos] - BaseX) * AxisX + (HullY[PointPos] - BaseY) * AxisY;
  end;

  // Signed distance from the edge line, along (-AxisY, AxisX).
  function HeightAt(PointPos: integer): double;
  begin
    Result := (HullY[PointPos] - BaseY) * AxisX - (HullX[PointPos] - BaseX) * AxisY;
  end;

  function NextPos(PointPos: integer): integer;
  begin
    if PointPos = MaxPointPos then Result := 0 else Result := PointPos + 1;
  end;

begin
  PointCount := Length(HullX);
  MaxPointPos := PointCount - 1;
  if PointCount = 0 then
    exit(RotatedRectFromAxis(0, 0, 1, 0, 0, 0));
  if PointCount < 3 then
    exit(RotatedRectFromAxis((HullX[0] + HullX[MaxPointPos]) * 0.5,
      (HullY[0] + HullY[MaxPointPos]) * 0.5, HullX[MaxPointPos] - HullX[0],
      HullY[MaxPointPos] - HullY[0], Hypot(HullX[MaxPointPos] - HullX[0],
      HullY[MaxPointPos] - HullY[0]), 0));
  BestArea := MaxDouble;
  MaxDotPos := 0;
  MinDotPos := 0;
  FarPos := 0;
  BestCentreX := 0;
  BestCentreY := 0;
  BestAxisX := 1;
  BestAxisY := 0;
  BestLength := 0;
  BestHeight := 0;
  for EdgePos := 0 to MaxPointPos do
  begin
    BaseX := HullX[EdgePos];
    BaseY := HullY[EdgePos];
    EdgeX := HullX[NextPos(EdgePos)] - BaseX;
    EdgeY := HullY[NextPos(EdgePos)] - BaseY;
    EdgeLength := Hypot(EdgeX, EdgeY);
    if EdgeLength = 0 then continue;
    AxisX := EdgeX / EdgeLength;
    AxisY := EdgeY / EdgeLength;
    if EdgePos = 0 then
      for PointPos := 1 to MaxPointPos do
      begin
        if DotAt(PointPos) > DotAt(MaxDotPos) then MaxDotPos := PointPos;
        if DotAt(PointPos) < DotAt(MinDotPos) then MinDotPos := PointPos;
        if Abs(HeightAt(PointPos)) > Abs(HeightAt(FarPos)) then FarPos := PointPos;
      end;
    // Rotating calipers: each support point only moves forward along the hull.
    while DotAt(NextPos(MaxDotPos)) > DotAt(MaxDotPos) do
      MaxDotPos := NextPos(MaxDotPos);
    while DotAt(NextPos(MinDotPos)) < DotAt(MinDotPos) do
      MinDotPos := NextPos(MinDotPos);
    while Abs(HeightAt(NextPos(FarPos))) > Abs(HeightAt(FarPos)) do
      FarPos := NextPos(FarPos);
    MaxDot := DotAt(MaxDotPos);
    MinDot := DotAt(MinDotPos);
    FarHeight := HeightAt(FarPos);
    Area := (MaxDot - MinDot) * Abs(FarHeight);
    if Area < BestArea then
    begin
      BestArea := Area;
      BestCentreX := BaseX + AxisX * (MaxDot + MinDot) * 0.5 - AxisY * FarHeight * 0.5;
      BestCentreY := BaseY + AxisY * (MaxDot + MinDot) * 0.5 + AxisX * FarHeight * 0.5;
      BestAxisX := AxisX;
      BestAxisY := AxisY;
      BestLength := MaxDot - MinDot;
      BestHeight := Abs(FarHeight);
    end;
  end;
  Result := RotatedRectFromAxis(BestCentreX, BestCentreY, BestAxisX, BestAxisY,
    BestLength, BestHeight);
end;

function RotatedRectWidth(const Rect: TImageRotatedRect): TNeuralFloat;
begin
  Result := Hypot(Rect.CornerX[1] - Rect.CornerX[0],
    Rect.CornerY[1] - Rect.CornerY[0]);
end;

function RotatedRectHeight(const Rect: TImageRotatedRect): TNeuralFloat;
begin
  Result := Hypot(Rect.CornerX[3] - Rect.CornerX[0],
    Rect.CornerY[3] - Rect.CornerY[0]);
end;

procedure OffsetRotatedRect(var Rect: TImageRotatedRect; Distance: TNeuralFloat);
begin
  Rect := RotatedRectFromAxis((Rect.CornerX[0] + Rect.CornerX[2]) * 0.5,
    (Rect.CornerY[0] + Rect.CornerY[2]) * 0.5, Rect.CornerX[1] - Rect.CornerX[0],
    Rect.CornerY[1] - Rect.CornerY[0],
    Max(RotatedRectWidth(Rect) + 2 * Distance, 0),
    Max(RotatedRectHeight(Rect) + 2 * Distance, 0));
end;

procedure CropRotatedRect(Image, Crop: TNNetVolume; const Rect: TImageRotatedRect;
  const BorderValue: array of TNeuralFloat);
var
  CropSizeX, CropSizeY: integer;
  RectWidth, UnitX, UnitY: TNeuralFloat;
  CropToImage: TImageAffineMatrix;
begin
  RectWidth := RotatedRectWidth(Rect);
  CropSizeX := Max(1, Round(RectWidth));
  CropSizeY := Max(1, Round(RotatedRectHeight(Rect)));
  if RectWidth > 0 then
  begin
    UnitX := (Rect.CornerX[1] - Rect.CornerX[0]) / RectWidth;
    UnitY := (Rect.CornerY[1] - Rect.CornerY[0]) / RectWidth;
  end
  else
  begin
    UnitX := 1;
    UnitY := 0;
  end;
  // Columns: one crop pixel step along the width and along the height.
  if CropSizeX > 1 then
  begin
    CropToImage[0] := (Rect.CornerX[1] - Rect.CornerX[0]) / (CropSizeX - 1);
    CropToImage[3] := (Rect.CornerY[1] - Rect.CornerY[0]) / (CropSizeX - 1);
  end
  else
  begin
    CropToImage[0] := UnitX;
    CropToImage[3] := UnitY;
  end;
  if CropSizeY > 1 then
  begin
    CropToImage[1] := (Rect.CornerX[2] - Rect.CornerX[1]) / (CropSizeY - 1);
    CropToImage[4] := (Rect.CornerY[2] - Rect.CornerY[1]) / (CropSizeY - 1);
  end
  else
  begin
    CropToImage[1] := -UnitY;
    CropToImage[4] := UnitX;
  end;
  CropToImage[2] := Rect.CornerX[0];
  CropToImage[5] := Rect.CornerY[0];
  WarpImageAffine(Image, Crop, InvertImageAffineMatrix(CropToImage), CropSizeX,
    CropSizeY, BorderValue);
end;

procedure GaussianBlurImage(Source, Dest: TNNetVolume; Radius: TNeuralFloat);
begin
  if Radius <= 0 then
  begin
    if Dest <> Source then Dest.Copy(Source);
    exit;
  end;
  // 3 sigma holds all but 0.3% of the Gaussian's mass.
  BlurImageSeparable(Source, Dest, Radius, Ceil(3 * Radius), bemClamp);
end;

procedure GaussianBlurImage(Source, Dest: TNNetVolume; Sigma: TNeuralFloat;
  KernelSize: integer);
begin
  if (KernelSize < 1) or not Odd(KernelSize) or (Sigma <= 0) then
    raise EArgumentException.Create('GaussianBlurImage: KernelSize must be ' +
      'odd and >= 1 and Sigma > 0.');
  BlurImageSeparable(Source, Dest, Sigma, KernelSize div 2, bemReflect101);
end;

procedure GaussianFilterImageScipy(Source, Dest: TNNetVolume;
  Sigma: TNeuralFloat; TapRadius: integer);
begin
  if (TapRadius < 0) or (Sigma <= 0) then
    raise EArgumentException.Create('GaussianFilterImageScipy: TapRadius ' +
      'must be >= 0 and Sigma > 0.');
  BlurImageSeparable(Source, Dest, Sigma, TapRadius, bemSymmetric);
end;

procedure CannyEdges(Source, Dest: TNNetVolume; LowThreshold,
  HighThreshold: TNeuralFloat; L2Gradient: boolean; DestDepth: integer);
const
  // cv2's fixed-point tangent tests: tan(22.5) * 2^15 and tan(67.5) * 2^15.
  csTan22Fixed = 13573;
  csTan67Fixed = 13573 + 65536;
  csFixedOne = 32768;
  csMapCandidate = 0;
  csMapNotEdge = 1;
  csMapEdge = 2;
var
  // Gradient magnitudes framed by a one-pixel zero border, row stride PaddedX.
  Magnitudes: array of TNeuralFloat;
  // Sobel dx and dy of each pixel's strongest channel, row stride SizeX.
  GradientX, GradientY: array of TNeuralFloat;
  // Padded like Magnitudes; the border stays csMapNotEdge.
  EdgeMap: array of byte;
  // Padded positions of edge pixels whose neighbours are still unvisited.
  Stack: array of integer;
  NeighbourOffsets: array[0..7] of integer;
  SizeX, SizeY, Depth, MaxX, MaxY, MaxChannel, PaddedX, RowStride: integer;
  X, Y, Channel, RowAbove, RowHere, RowBelow, ColumnLeft, ColumnRight,
    ColumnHere, GradientPos, PaddedPos, StackCount, NeighbourPos,
    NeighbourIndex, DiagonalStep, PixelPos, MaxDestChannel: integer;
  ChannelGradientX, ChannelGradientY, ChannelMagnitude, BestMagnitude,
    Magnitude: TNeuralFloat;
  LowFloor, HighFloor, Swap, AbsGradientX, ScaledAbsGradientY: double;
  IsMaximum: boolean;
  EdgeValue: TNeuralFloat;
begin
  SizeX := Source.SizeX;
  SizeY := Source.SizeY;
  Depth := Source.Depth;
  MaxX := SizeX - 1;
  MaxY := SizeY - 1;
  MaxChannel := Depth - 1;
  RowStride := SizeX * Depth;
  PaddedX := SizeX + 2;
  LowFloor := LowThreshold;
  HighFloor := HighThreshold;
  if LowFloor > HighFloor then
  begin
    Swap := LowFloor;
    LowFloor := HighFloor;
    HighFloor := Swap;
  end;
  if L2Gradient then
  begin
    LowFloor := Min(32767.0, LowFloor);
    HighFloor := Min(32767.0, HighFloor);
    if LowFloor > 0 then LowFloor := LowFloor * LowFloor;
    if HighFloor > 0 then HighFloor := HighFloor * HighFloor;
  end;
  LowFloor := Floor(LowFloor);
  HighFloor := Floor(HighFloor);
  SetLength(Magnitudes, PaddedX * (SizeY + 2));
  FillChar(Magnitudes[0], Length(Magnitudes) * SizeOf(TNeuralFloat), 0);
  SetLength(GradientX, SizeX * SizeY);
  SetLength(GradientY, SizeX * SizeY);
  // 3x3 Sobel with replicated borders, as cv2.Canny calls cv::Sobel.
  GradientPos := 0;
  for Y := 0 to MaxY do
  begin
    RowAbove := Max(Y - 1, 0) * RowStride;
    RowHere := Y * RowStride;
    RowBelow := Min(Y + 1, MaxY) * RowStride;
    PaddedPos := (Y + 1) * PaddedX + 1;
    for X := 0 to MaxX do
    begin
      ColumnLeft := Max(X - 1, 0) * Depth;
      ColumnHere := X * Depth;
      ColumnRight := Min(X + 1, MaxX) * Depth;
      BestMagnitude := -1;
      for Channel := 0 to MaxChannel do
      begin
        ChannelGradientX :=
          (Source.FData[RowAbove + ColumnRight + Channel] -
           Source.FData[RowAbove + ColumnLeft + Channel]) +
          2 * (Source.FData[RowHere + ColumnRight + Channel] -
           Source.FData[RowHere + ColumnLeft + Channel]) +
          (Source.FData[RowBelow + ColumnRight + Channel] -
           Source.FData[RowBelow + ColumnLeft + Channel]);
        ChannelGradientY :=
          (Source.FData[RowBelow + ColumnLeft + Channel] +
           2 * Source.FData[RowBelow + ColumnHere + Channel] +
           Source.FData[RowBelow + ColumnRight + Channel]) -
          (Source.FData[RowAbove + ColumnLeft + Channel] +
           2 * Source.FData[RowAbove + ColumnHere + Channel] +
           Source.FData[RowAbove + ColumnRight + Channel]);
        if L2Gradient then
          ChannelMagnitude := ChannelGradientX * ChannelGradientX +
            ChannelGradientY * ChannelGradientY
        else
          ChannelMagnitude := Abs(ChannelGradientX) + Abs(ChannelGradientY);
        // cv2 keeps the first channel on a tie.
        if ChannelMagnitude > BestMagnitude then
        begin
          BestMagnitude := ChannelMagnitude;
          GradientX[GradientPos] := ChannelGradientX;
          GradientY[GradientPos] := ChannelGradientY;
        end;
      end;
      Magnitudes[PaddedPos] := BestMagnitude;
      Inc(GradientPos);
      Inc(PaddedPos);
    end;
  end;
  // Non-maximum suppression and the double threshold; strong pixels seed Stack.
  SetLength(EdgeMap, Length(Magnitudes));
  FillChar(EdgeMap[0], Length(EdgeMap), csMapNotEdge);
  SetLength(Stack, SizeX * SizeY);
  StackCount := 0;
  GradientPos := 0;
  for Y := 0 to MaxY do
  begin
    PaddedPos := (Y + 1) * PaddedX + 1;
    for X := 0 to MaxX do
    begin
      Magnitude := Magnitudes[PaddedPos];
      if Magnitude > LowFloor then
      begin
        AbsGradientX := Abs(GradientX[GradientPos]);
        ScaledAbsGradientY := Abs(GradientY[GradientPos]) * csFixedOne;
        if ScaledAbsGradientY < AbsGradientX * csTan22Fixed then
          IsMaximum := (Magnitude > Magnitudes[PaddedPos - 1]) and
            (Magnitude >= Magnitudes[PaddedPos + 1])
        else if ScaledAbsGradientY > AbsGradientX * csTan67Fixed then
          IsMaximum := (Magnitude > Magnitudes[PaddedPos - PaddedX]) and
            (Magnitude >= Magnitudes[PaddedPos + PaddedX])
        else
        begin
          if (GradientX[GradientPos] < 0) <> (GradientY[GradientPos] < 0) then
            DiagonalStep := -1
          else
            DiagonalStep := 1;
          IsMaximum :=
            (Magnitude > Magnitudes[PaddedPos - PaddedX - DiagonalStep]) and
            (Magnitude > Magnitudes[PaddedPos + PaddedX + DiagonalStep]);
        end;
        if IsMaximum then
        begin
          if Magnitude > HighFloor then
          begin
            EdgeMap[PaddedPos] := csMapEdge;
            Stack[StackCount] := PaddedPos;
            Inc(StackCount);
          end
          else
            EdgeMap[PaddedPos] := csMapCandidate;
        end;
      end;
      Inc(GradientPos);
      Inc(PaddedPos);
    end;
  end;
  // Hysteresis: grow the strong edges through 8-connected candidates.
  NeighbourOffsets[0] := -PaddedX - 1;
  NeighbourOffsets[1] := -PaddedX;
  NeighbourOffsets[2] := -PaddedX + 1;
  NeighbourOffsets[3] := -1;
  NeighbourOffsets[4] := 1;
  NeighbourOffsets[5] := PaddedX - 1;
  NeighbourOffsets[6] := PaddedX;
  NeighbourOffsets[7] := PaddedX + 1;
  while StackCount > 0 do
  begin
    Dec(StackCount);
    PaddedPos := Stack[StackCount];
    for NeighbourIndex := 0 to 7 do
    begin
      NeighbourPos := PaddedPos + NeighbourOffsets[NeighbourIndex];
      if EdgeMap[NeighbourPos] = csMapCandidate then
      begin
        EdgeMap[NeighbourPos] := csMapEdge;
        Stack[StackCount] := NeighbourPos;
        Inc(StackCount);
      end;
    end;
  end;
  Dest.ReSize(SizeX, SizeY, DestDepth);
  MaxDestChannel := DestDepth - 1;
  PixelPos := 0;
  for Y := 0 to MaxY do
  begin
    PaddedPos := (Y + 1) * PaddedX + 1;
    for X := 0 to MaxX do
    begin
      if EdgeMap[PaddedPos] = csMapEdge then EdgeValue := 255
      else EdgeValue := 0;
      for Channel := 0 to MaxDestChannel do
        Dest.FData[PixelPos + Channel] := EdgeValue;
      Inc(PixelPos, DestDepth);
      Inc(PaddedPos);
    end;
  end;
end;

function MaskChannelBounds(Map: TNNetVolume; Channel: integer;
  Threshold: TNeuralFloat; out X1, Y1, X2, Y2: integer): boolean;
var
  X, Y, MaxX, MaxY, PixelPos, Depth: integer;
begin
  if (Channel < 0) or (Channel >= Map.Depth) then
    raise Exception.Create('MaskChannelBounds: channel ' + IntToStr(Channel) +
      ' is outside depth ' + IntToStr(Map.Depth) + '.');
  X1 := Map.SizeX;
  Y1 := Map.SizeY;
  X2 := -1;
  Y2 := -1;
  Depth := Map.Depth;
  MaxX := Map.SizeX - 1;
  MaxY := Map.SizeY - 1;
  PixelPos := Channel;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      if Map.FData[PixelPos] > Threshold then
      begin
        if X < X1 then X1 := X;
        if X > X2 then X2 := X;
        if Y < Y1 then Y1 := Y;
        Y2 := Y;
      end;
      Inc(PixelPos, Depth);
    end;
  Result := X2 >= 0;
  if not Result then
  begin
    X1 := 0;
    Y1 := 0;
    X2 := 0;
    Y2 := 0;
  end;
end;

procedure BlendColorWhereMask(Image, Mask: TNNetVolume; MaskChannel: integer;
  const Color: array of TNeuralFloat; Alpha: TNeuralFloat;
  Threshold: TNeuralFloat);
var
  PixelPos, MaxPixelPos, ChannelPos, MaxChannelPos, ImagePos, MaskPos: integer;
  MaskDepth: integer;
  KeepWeight: TNeuralFloat;
  BlendedColor: array of TNeuralFloat;
begin
  CheckDrawColor(Image, Color, 'BlendColorWhereMask');
  if (Mask.SizeX <> Image.SizeX) or (Mask.SizeY <> Image.SizeY) or
    (MaskChannel < 0) or (MaskChannel >= Mask.Depth) then
    raise EArgumentException.Create('BlendColorWhereMask: Mask must match ' +
      'the image size and hold channel ' + IntToStr(MaskChannel) + '.');
  MaxChannelPos := Image.Depth - 1;
  SetLength(BlendedColor, Image.Depth);
  for ChannelPos := 0 to MaxChannelPos do
    BlendedColor[ChannelPos] := Alpha * Color[ChannelPos];
  KeepWeight := 1 - Alpha;
  MaskDepth := Mask.Depth;
  MaxPixelPos := Image.SizeX * Image.SizeY - 1;
  MaskPos := MaskChannel;
  ImagePos := 0;
  for PixelPos := 0 to MaxPixelPos do
  begin
    if Mask.FData[MaskPos] > Threshold then
      for ChannelPos := 0 to MaxChannelPos do
        Image.FData[ImagePos + ChannelPos] := KeepWeight *
          Image.FData[ImagePos + ChannelPos] + BlendedColor[ChannelPos];
    Inc(MaskPos, MaskDepth);
    Inc(ImagePos, MaxChannelPos + 1);
  end;
end;

function LabelEightConnectedComponents(Map: TNNetVolume; Channel: integer;
  Threshold: TNeuralFloat; var Labels: TNeuralIntegerArray): integer;
var
  Stack: TNeuralIntegerArray;
  StackCount, PixelCount, SeedPos, PixelPos, NeighbourPos, Depth: integer;
  MaxSeedPos, X, Y, NeighbourX, NeighbourY, MaxX, MaxY: integer;
begin
  if (Channel < 0) or (Channel >= Map.Depth) then
    raise Exception.Create('LabelEightConnectedComponents: channel ' +
      IntToStr(Channel) + ' is outside depth ' + IntToStr(Map.Depth) + '.');
  PixelCount := Map.SizeX * Map.SizeY;
  SetLength(Labels, PixelCount);
  if PixelCount = 0 then Exit(0);
  FillChar(Labels[0], PixelCount * SizeOf(integer), 0);
  SetLength(Stack, PixelCount);
  Depth := Map.Depth;
  MaxX := Map.SizeX - 1;
  MaxY := Map.SizeY - 1;
  MaxSeedPos := PixelCount - 1;
  Result := 0;
  for SeedPos := 0 to MaxSeedPos do
  begin
    if (Labels[SeedPos] <> 0) or
      not (Map.FData[SeedPos * Depth + Channel] > Threshold) then continue;
    Inc(Result);
    Labels[SeedPos] := Result;
    Stack[0] := SeedPos;
    StackCount := 1;
    // A pixel is labelled when pushed, so the stack never exceeds PixelCount.
    while StackCount > 0 do
    begin
      Dec(StackCount);
      PixelPos := Stack[StackCount];
      Y := PixelPos div Map.SizeX;
      X := PixelPos - Y * Map.SizeX;
      for NeighbourY := Max(Y - 1, 0) to Min(Y + 1, MaxY) do
        for NeighbourX := Max(X - 1, 0) to Min(X + 1, MaxX) do
        begin
          NeighbourPos := NeighbourY * Map.SizeX + NeighbourX;
          if (Labels[NeighbourPos] = 0) and
            (Map.FData[NeighbourPos * Depth + Channel] > Threshold) then
          begin
            Labels[NeighbourPos] := Result;
            Stack[StackCount] := NeighbourPos;
            Inc(StackCount);
          end;
        end;
    end;
  end;
end;

// The canvas rectangle the outpaint mask keeps: the image without an Overlap
// band on each extended side; empty (Last < First) when the bands meet.
procedure GetOutpaintKeptArea(Image: TNNetVolume; Left, Top, Right, Bottom,
  Overlap: integer; out FirstX, LastX, FirstY, LastY: integer);
begin
  if (Left < 0) or (Top < 0) or (Right < 0) or (Bottom < 0) or
    (Overlap < 0) then
    raise Exception.Create('outpaint: sides ' + IntToStr(Left) + ',' +
      IntToStr(Top) + ',' + IntToStr(Right) + ',' + IntToStr(Bottom) +
      ' and overlap ' + IntToStr(Overlap) + ' must not be negative.');
  FirstX := Left;
  LastX := Left + Image.SizeX - 1;
  FirstY := Top;
  LastY := Top + Image.SizeY - 1;
  if Left > 0 then Inc(FirstX, Overlap);
  if Right > 0 then Dec(LastX, Overlap);
  if Top > 0 then Inc(FirstY, Overlap);
  if Bottom > 0 then Dec(LastY, Overlap);
end;

procedure PadImageClampToEdge(Image: TNNetVolume; Left, Top, Right,
  Bottom: integer; Canvas: TNNetVolume);
var
  CanvasSizeX, CanvasSizeY, Depth, X, Y, MaxX, MaxY, MaxImageX, MaxImageY,
    SourceY, PixelBytes, RowBytes: integer;
begin
  if (Left < 0) or (Top < 0) or (Right < 0) or (Bottom < 0) then
    raise Exception.Create('PadImageClampToEdge: sides ' + IntToStr(Left) +
      ',' + IntToStr(Top) + ',' + IntToStr(Right) + ',' + IntToStr(Bottom) +
      ' must not be negative.');
  if Canvas = Image then
    raise Exception.Create('PadImageClampToEdge: Canvas must not be Image.');
  CanvasSizeX := Image.SizeX + Left + Right;
  CanvasSizeY := Image.SizeY + Top + Bottom;
  Depth := Image.Depth;
  Canvas.ReSize(CanvasSizeX, CanvasSizeY, Depth);
  PixelBytes := Depth * SizeOf(TNeuralFloat);
  RowBytes := Image.SizeX * PixelBytes;
  MaxX := CanvasSizeX - 1;
  MaxY := CanvasSizeY - 1;
  MaxImageX := Image.SizeX - 1;
  MaxImageY := Image.SizeY - 1;
  for Y := 0 to MaxY do
  begin
    SourceY := Min(Max(Y - Top, 0), MaxImageY);
    for X := 0 to Left - 1 do
      Move(Image.FData[Image.GetRawPos(0, SourceY)],
        Canvas.FData[Canvas.GetRawPos(X, Y)], PixelBytes);
    Move(Image.FData[Image.GetRawPos(0, SourceY)],
      Canvas.FData[Canvas.GetRawPos(Left, Y)], RowBytes);
    for X := Left + Image.SizeX to MaxX do
      Move(Image.FData[Image.GetRawPos(MaxImageX, SourceY)],
        Canvas.FData[Canvas.GetRawPos(X, Y)], PixelBytes);
  end;
end;

procedure BuildOutpaintCanvas(Image: TNNetVolume; Left, Top, Right, Bottom,
  Overlap: integer; Canvas, Mask: TNNetVolume);
var
  X, Y, FirstX, LastX, FirstY, LastY: integer;
begin
  GetOutpaintKeptArea(Image, Left, Top, Right, Bottom, Overlap, FirstX, LastX,
    FirstY, LastY);
  PadImageClampToEdge(Image, Left, Top, Right, Bottom, Canvas);
  Mask.ReSize(Canvas.SizeX, Canvas.SizeY, 1);
  Mask.Fill(1);
  for Y := FirstY to LastY do
    for X := FirstX to LastX do
      Mask.FData[Mask.GetRawPos(X, Y)] := 0;
end;

procedure PasteOutpaintOriginal(Image: TNNetVolume; Left, Top, Right,
  Bottom, Overlap: integer; Dest: TNNetVolume);
var
  X, Y, FirstX, LastX, FirstY, LastY, Channel, MaxChannel, SourcePos,
    DestPos: integer;
begin
  GetOutpaintKeptArea(Image, Left, Top, Right, Bottom, Overlap, FirstX, LastX,
    FirstY, LastY);
  if (Dest.SizeX <> Image.SizeX + Left + Right) or
    (Dest.SizeY <> Image.SizeY + Top + Bottom) then
    raise Exception.Create('PasteOutpaintOriginal: the result is ' +
      IntToStr(Dest.SizeX) + 'x' + IntToStr(Dest.SizeY) + ', the canvas ' +
      IntToStr(Image.SizeX + Left + Right) + 'x' +
      IntToStr(Image.SizeY + Top + Bottom) + '.');
  MaxChannel := Min(Image.Depth, Dest.Depth) - 1;
  for Y := FirstY to LastY do
    for X := FirstX to LastX do
    begin
      SourcePos := Image.GetRawPos(X - Left, Y - Top);
      DestPos := Dest.GetRawPos(X, Y);
      for Channel := 0 to MaxChannel do
        Dest.FData[DestPos + Channel] := Image.FData[SourcePos + Channel];
    end;
end;

procedure GrowOutpaintSidesToMultiple(ImageSizeX, ImageSizeY,
  Multiple: integer; var Left, Top, Right, Bottom: integer);
var
  ExtraX, ExtraY: integer;
begin
  ExtraX := (Multiple - (ImageSizeX + Left + Right) mod Multiple) mod Multiple;
  ExtraY := (Multiple - (ImageSizeY + Top + Bottom) mod Multiple) mod Multiple;
  if (Right = 0) and (Left > 0) then Inc(Left, ExtraX)
  else Inc(Right, ExtraX);
  if (Bottom = 0) and (Top > 0) then Inc(Top, ExtraY)
  else Inc(Bottom, ExtraY);
end;

// PIL 'L' (0..255) of the mask pixel at PixelPos: the channels rounded to
// bytes, then R*0.299 + G*0.587 + B*0.114 in PIL's fixed point; gray reads as is.
function MaskPixelLuma(Mask: TNNetVolume; PixelPos: integer): integer;
const
  csLumaR = 19595;
  csLumaG = 38470;
  csLumaB = 7471;
  csLumaRound = 32768;
var
  SourcePos: integer;

  function ChannelByte(Value: TNeuralFloat): integer;
  begin
    if Value <= 0 then exit(0);
    if Value >= 255 then exit(255);
    Result := Floor(Value + 0.5);
  end;

begin
  SourcePos := PixelPos * Mask.Depth;
  if Mask.Depth < 3 then exit(ChannelByte(Mask.FData[SourcePos]));
  Result := (ChannelByte(Mask.FData[SourcePos]) * csLumaR +
    ChannelByte(Mask.FData[SourcePos + 1]) * csLumaG +
    ChannelByte(Mask.FData[SourcePos + 2]) * csLumaB + csLumaRound) shr 16;
end;

procedure GetMaskCropRegion(Mask: TNNetVolume; Width, Height,
  Padding: integer; out X1, Y1, X2, Y2: integer);
var
  X, Y, MaxX, MaxY, PixelPos, Left, Right, Top, Bottom: integer;
  ProcessingRatio: double;

  // Grows [First, Last) by Trunc(DesiredSize) - its size, half before, then
  // shifts it back inside [0, Limit) and clips it, as get_crop_region does.
  procedure GrowAxis(var First, Last: integer; DesiredSize: double;
    Limit: integer);
  var
    Growth: integer;
  begin
    Growth := Trunc(DesiredSize - (Last - First));
    Dec(First, Growth div 2);
    Inc(Last, Growth - Growth div 2);
    if Last > Limit then
    begin
      Dec(First, Last - Limit);
      Last := Limit;
    end;
    if First < 0 then
    begin
      Dec(Last, First);
      First := 0;
    end;
    if Last > Limit then Last := Limit;
  end;

begin
  if (Width < 1) or (Height < 1) or (Padding < 0) then
    raise Exception.Create('GetMaskCropRegion: size ' + IntToStr(Width) +
      'x' + IntToStr(Height) + ' and padding ' + IntToStr(Padding) +
      ' must be positive (padding >= 0).');
  Left := Mask.SizeX;
  Right := -1;
  Top := Mask.SizeY;
  Bottom := -1;
  MaxX := Mask.SizeX - 1;
  MaxY := Mask.SizeY - 1;
  PixelPos := 0;
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      if MaskPixelLuma(Mask, PixelPos) > 0 then
      begin
        Left := Min(Left, X);
        Right := Max(Right, X);
        Top := Min(Top, Y);
        Bottom := Y;
      end;
      Inc(PixelPos);
    end;
  if Right < 0 then
    raise Exception.Create('GetMaskCropRegion: the mask has no nonzero pixel.');
  X1 := Max(Left - Padding, 0);
  Y1 := Max(Top - Padding, 0);
  X2 := Min(Right + 1 + Padding, Mask.SizeX);
  Y2 := Min(Bottom + 1 + Padding, Mask.SizeY);
  ProcessingRatio := Width / Height;
  if (X2 - X1) / (Y2 - Y1) > ProcessingRatio then
    GrowAxis(Y1, Y2, (X2 - X1) / ProcessingRatio, Mask.SizeY)
  else
    GrowAxis(X1, X2, (Y2 - Y1) * ProcessingRatio, Mask.SizeX);
end;

procedure ResizeImageAndFill(Source, Dest: TNNetVolume; Width,
  Height: integer; Filter: TImageResampleFilter; RoundToBytes: boolean);
var
  Resized: TNNetVolume;
  FitWidth, FitHeight, Left, Top: integer;
begin
  if Width / Height < Source.SizeX / Source.SizeY then
  begin
    FitWidth := Width;
    FitHeight := Max(Source.SizeY * Width div Source.SizeX, 1);
  end
  else
  begin
    FitWidth := Max(Source.SizeX * Height div Source.SizeY, 1);
    FitHeight := Height;
  end;
  Left := Width div 2 - FitWidth div 2;
  Top := Height div 2 - FitHeight div 2;
  Resized := TNNetVolume.Create();
  try
    ResizeImagePIL(Source, Resized, FitWidth, FitHeight, Filter, RoundToBytes);
    PadImageClampToEdge(Resized, Left, Top, Width - FitWidth - Left,
      Height - FitHeight - Top, Dest);
  finally
    Resized.Free;
  end;
end;

procedure ResizeImageAndCrop(Source, Dest: TNNetVolume; Width,
  Height: integer; Filter: TImageResampleFilter; RoundToBytes: boolean);
var
  Resized: TNNetVolume;
  CoverWidth, CoverHeight: integer;
begin
  if Width / Height > Source.SizeX / Source.SizeY then
  begin
    CoverWidth := Width;
    CoverHeight := Source.SizeY * Width div Source.SizeX;
  end
  else
  begin
    CoverWidth := Source.SizeX * Height div Source.SizeY;
    CoverHeight := Height;
  end;
  Resized := TNNetVolume.Create();
  try
    ResizeImagePIL(Source, Resized, CoverWidth, CoverHeight, Filter,
      RoundToBytes);
    Dest.CopyCropping(Resized, CoverWidth div 2 - Width div 2,
      CoverHeight div 2 - Height div 2, Width, Height);
  finally
    Resized.Free;
  end;
end;

procedure PasteInpaintCrop(Generated, Original, Mask: TNNetVolume;
  X1, Y1, X2, Y2: integer; Filter: TImageResampleFilter; Dest: TNNetVolume);
var
  Resized: TNNetVolume;
  X, Y, MaxX, MaxY, Channel, MaxChannel, DestPos, ResizedPos,
    MaskPos: integer;
  Weight: TNeuralFloat;
begin
  if (Mask.SizeX <> Original.SizeX) or (Mask.SizeY <> Original.SizeY) then
    raise Exception.Create('PasteInpaintCrop: the mask is ' +
      IntToStr(Mask.SizeX) + 'x' + IntToStr(Mask.SizeY) + ', the image ' +
      IntToStr(Original.SizeX) + 'x' + IntToStr(Original.SizeY) + '.');
  if (X1 < 0) or (Y1 < 0) or (X2 > Original.SizeX) or
    (Y2 > Original.SizeY) or (X2 <= X1) or (Y2 <= Y1) then
    raise Exception.Create('PasteInpaintCrop: crop ' + IntToStr(X1) + ',' +
      IntToStr(Y1) + ',' + IntToStr(X2) + ',' + IntToStr(Y2) +
      ' is not inside the image.');
  Resized := TNNetVolume.Create();
  try
    ResizeImageAndCrop(Generated, Resized, X2 - X1, Y2 - Y1, Filter);
    if Dest <> Original then
    begin
      // Copy alone keeps Dest's shape when only the element counts match.
      Dest.ReSize(Original);
      Dest.Copy(Original);
    end;
    MaxChannel := Min(Dest.Depth, Resized.Depth) - 1;
    MaxX := X2 - 1;
    MaxY := Y2 - 1;
    for Y := Y1 to MaxY do
    begin
      MaskPos := Y * Mask.SizeX + X1;
      for X := X1 to MaxX do
      begin
        Weight := MaskPixelLuma(Mask, MaskPos) / 255;
        Inc(MaskPos);
        if Weight = 0 then continue;
        DestPos := Dest.GetRawPos(X, Y);
        ResizedPos := Resized.GetRawPos(X - X1, Y - Y1);
        for Channel := 0 to MaxChannel do
          Dest.FData[DestPos + Channel] := Dest.FData[DestPos + Channel] +
            Weight * (Resized.FData[ResizedPos + Channel] -
            Dest.FData[DestPos + Channel]);
      end;
    end;
  finally
    Resized.Free;
  end;
end;

function ParseOutpaintSides(const Text: string;
  out Left, Top, Right, Bottom: integer): string;
var
  Sides: array[0..3] of integer;
  Rest: string;
  SidePos, CommaPos: integer;
begin
  Result := '"' + Text + '" is not L,T,R,B: four pixel counts >= 0, not ' +
    'all 0.';
  Left := 0; Top := 0; Right := 0; Bottom := 0;
  Rest := Text;
  for SidePos := 0 to 3 do
  begin
    CommaPos := Pos(',', Rest);
    if (SidePos < 3) = (CommaPos = 0) then exit;
    if CommaPos = 0 then CommaPos := Length(Rest) + 1;
    Sides[SidePos] := StrToIntDef(Trim(Copy(Rest, 1, CommaPos - 1)), -1);
    if Sides[SidePos] < 0 then exit;
    Rest := Copy(Rest, CommaPos + 1, MaxInt);
  end;
  if Sides[0] + Sides[1] + Sides[2] + Sides[3] = 0 then exit;
  Left := Sides[0];
  Top := Sides[1];
  Right := Sides[2];
  Bottom := Sides[3];
  Result := '';
end;


initialization
  BuildCV2SinTable();

end.
