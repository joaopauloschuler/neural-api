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

// Source (W,H,D) to Dest (NewSizeX,NewSizeY,D): PIL's LANCZOS coefficients in
// float, equal to PIL mode 'F'; Dest may be Source. Not PIL's 8-bit rounding.
procedure ResizeImageLanczos(Source, Dest: TNNetVolume; NewSizeX,
  NewSizeY: integer);

implementation

uses
  Math;

procedure ResizeImageLanczos(Source, Dest: TNNetVolume; NewSizeX,
  NewSizeY: integer);
type
  // Per output coordinate of one axis: the first source coordinate, the tap
  // count, and the normalised weights at Weight[OutPos * MaxTaps].
  TLanczosAxis = record
    First, Count: array of integer;
    Weight: array of double;
    MaxTaps: integer;
  end;
var
  AxisX, AxisY: TLanczosAxis;
  // Source resized along X only: (NewSizeX, Source.SizeY, Depth).
  Rows: array of TNeuralFloat;
  ChannelCount, SourceSizeY, RowStride, SourceRowStride: integer;
  X, Y, Channel, TapPos, ElementPos: integer;
  MaxX, MaxY, MaxSourceY, MaxChannel, MaxTapPos, MaxElementPos: integer;
  TapBase, FirstPos, SourcePos, DestPos: integer;
  Sum: double;

  // PIL's lanczos_filter: sinc(x) * sinc(x / 3) on [-3, 3).
  function LanczosTap(Distance: double): double;

    function Sinc(Value: double): double;
    begin
      if Value = 0 then exit(1);
      Value := Value * Pi;
      Result := Sin(Value) / Value;
    end;

  begin
    if (Distance < -3) or (Distance >= 3) then exit(0);
    Result := Sinc(Distance) * Sinc(Distance / 3);
  end;

  // PIL's precompute_coeffs for an InSize -> OutSize axis.
  procedure BuildAxis(InSize, OutSize: integer; var Axis: TLanczosAxis);
  var
    Scale, FilterScale, InvFilterScale, Support, Center, Total: double;
    OutPos, MaxOutPos, Last, WeightPos, AxisTapPos, MaxAxisTapPos: integer;
  begin
    Scale := InSize / OutSize;
    FilterScale := Scale;
    if FilterScale < 1 then FilterScale := 1;
    InvFilterScale := 1 / FilterScale;
    Support := 3 * FilterScale;
    Axis.MaxTaps := Ceil(Support) * 2 + 1;
    SetLength(Axis.First, OutSize);
    SetLength(Axis.Count, OutSize);
    SetLength(Axis.Weight, OutSize * Axis.MaxTaps);
    MaxOutPos := OutSize - 1;
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
        Axis.Weight[WeightPos + AxisTapPos] := LanczosTap((AxisTapPos +
          Axis.First[OutPos] - Center + 0.5) * InvFilterScale);
        Total := Total + Axis.Weight[WeightPos + AxisTapPos];
      end;
      if Total <> 0 then
        for AxisTapPos := 0 to MaxAxisTapPos do
          Axis.Weight[WeightPos + AxisTapPos] :=
            Axis.Weight[WeightPos + AxisTapPos] / Total;
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
      Dest.FData[DestPos + ElementPos] := Sum;
    end;
  end;
end;


end.
