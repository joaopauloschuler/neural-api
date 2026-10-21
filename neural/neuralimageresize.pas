unit neuralimageresize;

(*
neuralimageresize: image resampling on TNNetVolume with no unit dependency
beyond neuralvolume (no fcl-image), so importers can use it.

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

// PIL code (0 NEAREST, 1 LANCZOS, 2 BILINEAR, 3 BICUBIC, 4 BOX, 5 HAMMING),
// as in a HF preprocessor_config.json "resample"; raises on other codes.
function PILResampleCodeToFilter(Code: integer): TImageResampleFilter;

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


end.
