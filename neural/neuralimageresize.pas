unit neuralimageresize;

(*
neuralimageresize: image resampling and filtering on TNNetVolume with no
unit dependency beyond neuralvolume (no fcl-image), so importers can use it.

Coded by Claude (AI).
*)

{$mode objfpc}{$H+}

interface

uses
  neuralvolume;

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

// Separable Gaussian blur of Source (W,H,D) per channel, sigma = Radius (PIL
// GaussianBlur radius), clamped edges. Dest may be Source; Radius <= 0 copies.
procedure GaussianBlurImage(Source, Dest: TNNetVolume; Radius: TNeuralFloat); overload;
// torchvision gaussian_blur: KernelSize (odd) taps of std Sigma per axis,
// reflect padding (no edge repeat). Dest may be Source.
procedure GaussianBlurImage(Source, Dest: TNNetVolume; Sigma: TNeuralFloat;
  KernelSize: integer); overload;

// cv2.Canny(Source, Low, High, apertureSize=3, L2gradient) of a 0..255 image
// (no blur; strongest channel per pixel): Dest (W,H,DestDepth) holds 0 or 255.
procedure CannyEdges(Source, Dest: TNNetVolume; LowThreshold,
  HighThreshold: TNeuralFloat; L2Gradient: boolean = false;
  DestDepth: integer = 1);

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
  SysUtils, Math;

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


// Separable blur of Source by the normalized Gaussian of Sigma over taps
// -TapRadius..TapRadius; past the edge it clamps or reflects.
procedure BlurImageSeparable(Source, Dest: TNNetVolume; Sigma: TNeuralFloat;
  TapRadius: integer; IsReflect: boolean);
var
  // Taps[TapPos] weighs the pixel TapPos - TapRadius away; they sum to 1.
  Taps: array of double;
  // NeighboursX/Y[Pos * TapCount + TapPos]: the pixel tap TapPos of Pos reads.
  NeighboursX, NeighboursY: array of integer;
  // Source blurred along X only, in the TNNetVolume layout.
  Rows: array of TNeuralFloat;
  TapPos, MaxTapPos, TapCount, SizeX, SizeY, Depth, RowStride: integer;
  X, Y, Channel, MaxX, MaxY, MaxChannel, PixelPos, NeighbourBase: integer;
  Distance, TapSum, Sum: double;

  procedure BuildNeighbours(MaxPos: integer; var Neighbours: array of integer);
  var
    Pos, NeighbourTapPos, Offset, Period: integer;
  begin
    Period := 2 * MaxPos;
    for Pos := 0 to MaxPos do
      for NeighbourTapPos := 0 to MaxTapPos do
      begin
        Offset := Pos + NeighbourTapPos - TapRadius;
        if not IsReflect then Offset := Min(Max(Offset, 0), MaxPos)
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
  MaxTapPos := 2 * TapRadius;
  TapCount := MaxTapPos + 1;
  SetLength(Taps, TapCount);
  TapSum := 0;
  for TapPos := 0 to MaxTapPos do
  begin
    Distance := TapPos - TapRadius;
    Taps[TapPos] := Exp(-Distance * Distance / (2 * Sigma * Sigma));
    TapSum := TapSum + Taps[TapPos];
  end;
  for TapPos := 0 to MaxTapPos do
    Taps[TapPos] := Taps[TapPos] / TapSum;
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

procedure GaussianBlurImage(Source, Dest: TNNetVolume; Radius: TNeuralFloat);
begin
  if Radius <= 0 then
  begin
    if Dest <> Source then Dest.Copy(Source);
    exit;
  end;
  // 3 sigma holds all but 0.3% of the Gaussian's mass.
  BlurImageSeparable(Source, Dest, Radius, Ceil(3 * Radius), false);
end;

procedure GaussianBlurImage(Source, Dest: TNNetVolume; Sigma: TNeuralFloat;
  KernelSize: integer);
begin
  if (KernelSize < 1) or not Odd(KernelSize) or (Sigma <= 0) then
    raise EArgumentException.Create('GaussianBlurImage: KernelSize must be ' +
      'odd and >= 1 and Sigma > 0.');
  BlurImageSeparable(Source, Dest, Sigma, KernelSize div 2, true);
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


end.
