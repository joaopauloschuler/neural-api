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

// HF/torchvision shortest-edge resize of a 0..255 image (long side
// Trunc(ShortestEdge*Long/Short)) with PIL's uint8 rounding.
procedure ResizeByteImageShortestEdge(Source, Dest: TNNetVolume;
  ShortestEdge: integer; Filter: TImageResampleFilter);

// Separable Gaussian blur of Source (W,H,D) per channel, sigma = Radius (PIL
// GaussianBlur radius), clamped edges. Dest may be Source; Radius <= 0 copies.
procedure GaussianBlurImage(Source, Dest: TNNetVolume; Radius: TNeuralFloat);

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


procedure GaussianBlurImage(Source, Dest: TNNetVolume; Radius: TNeuralFloat);
var
  // Taps[TapPos] weighs the pixel TapPos - TapRadius away; they sum to 1.
  Taps: array of double;
  // Source blurred along X only, in the TNNetVolume layout.
  Rows: array of TNeuralFloat;
  TapRadius, TapPos, MaxTapPos, SizeX, SizeY, Depth, RowStride: integer;
  X, Y, Channel, MaxX, MaxY, MaxChannel, NeighbourPos, PixelPos,
    SourcePixelPos: integer;
  Distance, TapSum, Sum: double;
begin
  if Radius <= 0 then
  begin
    if Dest <> Source then Dest.Copy(Source);
    exit;
  end;
  // 3 sigma holds all but 0.3% of the Gaussian's mass.
  TapRadius := Ceil(3 * Radius);
  MaxTapPos := 2 * TapRadius;
  SetLength(Taps, MaxTapPos + 1);
  TapSum := 0;
  for TapPos := 0 to MaxTapPos do
  begin
    Distance := TapPos - TapRadius;
    Taps[TapPos] := Exp(-Distance * Distance / (2 * Radius * Radius));
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
  SetLength(Rows, SizeX * SizeY * Depth);
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      PixelPos := Y * RowStride + X * Depth;
      for Channel := 0 to MaxChannel do
      begin
        Sum := 0;
        for TapPos := 0 to MaxTapPos do
        begin
          NeighbourPos := Min(Max(X + TapPos - TapRadius, 0), MaxX);
          Sum := Sum + Taps[TapPos] *
            Source.FData[Y * RowStride + NeighbourPos * Depth + Channel];
        end;
        Rows[PixelPos + Channel] := Sum;
      end;
    end;
  Dest.ReSize(SizeX, SizeY, Depth);
  for Y := 0 to MaxY do
    for X := 0 to MaxX do
    begin
      PixelPos := Y * RowStride + X * Depth;
      for Channel := 0 to MaxChannel do
      begin
        Sum := 0;
        for TapPos := 0 to MaxTapPos do
        begin
          NeighbourPos := Min(Max(Y + TapPos - TapRadius, 0), MaxY);
          SourcePixelPos := NeighbourPos * RowStride + X * Depth;
          Sum := Sum + Taps[TapPos] * Rows[SourcePixelPos + Channel];
        end;
        Dest.FData[PixelPos + Channel] := Sum;
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
