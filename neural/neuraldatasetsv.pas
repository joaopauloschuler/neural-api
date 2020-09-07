(*
neuraldatasetsv
Copyright (C) 2018 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)

unit neuraldatasetsv;

{$IFDEF FPC} {$mode objfpc}{$H+} {$ENDIF}
{$DEFINE HASTHREADS}

interface

uses
  neuraldatasets, Classes, SysUtils, ExtCtrls, Graphics,
  neuralvolume, neuralnetwork, StdCtrls
  {$IFNDEF FPC}, Windows{$ENDIF}
  ;

type
  TImageDynArr = array of TImage;
  TLabelDynArr = array of TLabel;

/// Loads a TinyImage into FCL TImage.
procedure LoadTinyImageIntoTImage(var TI: TTinyImage; var Image: TImage);

/// Loads a single channel tiny image into image.
procedure LoadTISingleChannelIntoImage(var TI: TTinySingleChannelImage;
  var Image: TImage);

/// Show Neuronal Patterns (pNeuronList) in an array of images (pImage).
procedure ShowNeurons(
  pNeuronList: TNNetNeuronList;
  var pImage: TImageDynArr;
  startImage, filterSize, color_encoding: integer;
  ScalePerImage: boolean);

/// Create Images for Gradient Ascent
procedure CreateAscentImages
(
  GrBoxNeurons: TGroupBox;
  var pImage: TImageDynArr;
  var pLabelX, pLabelY: TLabelDynArr;
  ImagesNum: integer;
  inputSize, displaySize, imagesPerRow: integer
);

/// Creates images to display neuronal patterns inside of a group box.
procedure CreateNeuronImages
(
  GrBoxNeurons: TGroupBox;
  var pImage: TImageDynArr;
  var pLabelX, pLabelY: TLabelDynArr;
  pNeuronList: TNNetNeuronList;
  filterSize, imagesPerRow, NeuronNum: integer
);

/// Frees images displaying neuronal patterns.
procedure FreeNeuronImages
(
  var pImage: TImageDynArr;
  var pLabelX, pLabelY: TLabelDynArr
);

/// Loads neuronal network layer names into a ComboBox.
procedure LoadNNLayersIntoCombo(NN: TNNet; Combo: TComboBox);

implementation

uses
  neuralvolumev, math;

procedure LoadNNLayersIntoCombo(NN: TNNet; Combo: TComboBox);
var
  LayerCnt: integer;
  LayerStr: string;
begin
  Combo.Enabled := false;
  Combo.Items.Clear;
  if NN.GetLastLayerIdx() > 0 then
  begin
    for LayerCnt := 0 to NN.GetLastLayerIdx() do
    begin
      LayerStr := IntToStr(LayerCnt) + ' - ' + NN.Layers[LayerCnt].ClassName;
      Combo.Items.Add(LayerStr);
    end;
    Combo.Enabled := true;
    Combo.ItemIndex := NN.GetLastLayerIdx();
  end;
end;

procedure CreateLabels
(
  GrBoxNeurons: TGroupBox;
  ColNum, RowNum, filterSize, PosTop, PosLeft: integer;
  var pLabelX, pLabelY: TLabelDynArr
);
var
  RowCount, ColCount: integer;
begin
  for ColCount := 0 to ColNum - 1 do
  begin
    pLabelX[ColCount] := TLabel.Create(GrBoxNeurons.Parent);
    pLabelX[ColCount].Parent  := GrBoxNeurons;
    pLabelX[ColCount].Top     := (0)        * (filterSize+4) + PosTop - 14;
    pLabelX[ColCount].Left    := (ColCount) * (filterSize+4) + PosLeft;
    pLabelX[ColCount].Caption := Chr(Ord('A') + ColCount);
  end;

  for RowCount := 0 to RowNum - 1 do
  begin
    pLabelY[RowCount] := TLabel.Create(GrBoxNeurons.Parent);
    pLabelY[RowCount].Parent  := GrBoxNeurons;
    pLabelY[RowCount].Top     := (RowCount) * (filterSize+4) + PosTop;
    pLabelY[RowCount].Left    := (0)        * (filterSize+4) + PosLeft - 16;
    pLabelY[RowCount].Caption := IntToStr(RowCount);
  end;
end;

procedure CreateAscentImages
(
  GrBoxNeurons: TGroupBox;
  var pImage: TImageDynArr;
  var pLabelX, pLabelY: TLabelDynArr;
  ImagesNum: integer;
  inputSize, displaySize, imagesPerRow: integer
);
var
  NeuronCount: integer;
  RowNum, ColNum: integer;
  PosTop, PosLeft: integer;
  MaxTop, MaxLeft: integer;
begin
  PosTop  := 14;
  PosLeft := 22;
  MaxTop  := 0;
  MaxLeft := 0;

  RowNum := ImagesNum div imagesPerRow;
  ColNum := imagesPerRow;

  if (ImagesNum mod imagesPerRow > 0) then
  begin
    Inc(RowNum);
  end;

  SetLength(pImage,  ImagesNum);
  SetLength(pLabelY, RowNum);
  SetLength(pLabelX, ColNum);

  for NeuronCount := 0 to ImagesNum - 1 do
  begin
    pImage[NeuronCount] := TImage.Create(GrBoxNeurons.Parent);
    pImage[NeuronCount].Parent  := GrBoxNeurons;
    pImage[NeuronCount].Width   := inputSize;
    pImage[NeuronCount].Height  := inputSize;
    pImage[NeuronCount].Top     := (NeuronCount div imagesPerRow) * (displaySize+4) + PosTop;
    pImage[NeuronCount].Left    := (NeuronCount mod imagesPerRow) * (displaySize+4) + PosLeft;
    pImage[NeuronCount].Stretch := true;
    MaxTop                      := Max(MaxTop, pImage[NeuronCount].Top);
    MaxLeft                     := Max(MaxLeft, pImage[NeuronCount].Left);
  end;

  GrBoxNeurons.Height := MaxTop  + displaySize + 24;
  GrBoxNeurons.Width  := MaxLeft + displaySize + 10;

  CreateLabels(GrBoxNeurons, ColNum, RowNum, displaySize, PosTop, PosLeft, pLabelX, pLabelY);
end;

procedure CreateNeuronImages
(
  GrBoxNeurons: TGroupBox;
  var pImage: TImageDynArr;
  var pLabelX, pLabelY: TLabelDynArr;
  pNeuronList: TNNetNeuronList;
  filterSize, imagesPerRow, NeuronNum: integer
);
var
  NeuronCount: integer;
  RowNum, ColNum: integer;
  PosTop, PosLeft: integer;
  MaxTop, MaxLeft: integer;
begin
  PosTop  := 14;
  PosLeft := 22;
  MaxTop  := 0;
  MaxLeft := 0;

  RowNum := NeuronNum div imagesPerRow;
  ColNum := imagesPerRow;

  if (NeuronNum mod imagesPerRow > 0) then
  begin
    Inc(RowNum);
  end;

  SetLength(pImage,  NeuronNum);
  SetLength(pLabelY, RowNum);
  SetLength(pLabelX, ColNum);

  for NeuronCount := 0 to NeuronNum - 1 do
  begin
    pImage[NeuronCount] := TImage.Create(GrBoxNeurons.Parent);
    pImage[NeuronCount].Parent  := GrBoxNeurons;
    pImage[NeuronCount].Width   := pNeuronList[0].Weights.SizeX;
    pImage[NeuronCount].Height  := pNeuronList[0].Weights.SizeY;
    pImage[NeuronCount].Top     := (NeuronCount div imagesPerRow) * (filterSize+4) + PosTop;
    pImage[NeuronCount].Left    := (NeuronCount mod imagesPerRow) * (filterSize+4) + PosLeft;
    pImage[NeuronCount].Stretch := true;
    MaxTop                      := Max(MaxTop, pImage[NeuronCount].Top);
    MaxLeft                     := Max(MaxLeft, pImage[NeuronCount].Left);
  end;

  GrBoxNeurons.Height := MaxTop  + filterSize + 24;
  GrBoxNeurons.Width  := MaxLeft + filterSize + 10;

  CreateLabels(GrBoxNeurons, ColNum, RowNum, filterSize, PosTop, PosLeft, pLabelX, pLabelY);
end;

procedure FreeNeuronImages
(
  var pImage: TImageDynArr;
  var pLabelX, pLabelY: TLabelDynArr
);
var
  NeuronCount, RowCount, ColCount: integer;
begin
  if Length(pImage) > 0 then
  begin
    for NeuronCount := Low(pImage) to High(pImage) do
    begin
      pImage[NeuronCount].Free;
    end;
  end;

  if Length(pLabelX) > 0 then
  begin
    for RowCount := Low(pLabelX) to High(pLabelX) do
    begin
      pLabelX[RowCount].Free;
    end;
  end;

  if Length(pLabelY) > 0 then
  begin
    for ColCount := Low(pLabelY) to High(pLabelY) do
    begin
      pLabelY[ColCount].Free;
    end;
  end;

  SetLength(pImage,  0);
  SetLength(pLabelX, 0);
  SetLength(pLabelY, 0);
end;

procedure ShowNeurons(
  pNeuronList: TNNetNeuronList;
  var pImage: TImageDynArr;
  startImage, filterSize, color_encoding: integer;
  ScalePerImage: boolean);
var
  NeuronCount: integer;
  MaxW, MinW: TNeuralFloat;
  vDisplay: TNNetVolume;
begin
  vDisplay := TNNetVolume.Create();
  MaxW := 0.0;
  MinW := 0.0;

  if Not(ScalePerImage) then
  begin
    MaxW := pNeuronList.GetMaxWeight();
    MinW := pNeuronList.GetMinWeight();
  end;

  for NeuronCount := 0 to pNeuronList.Count - 1 do
  begin
    vDisplay.Copy(pNeuronList[NeuronCount].Weights);
    if (ScalePerImage) then
    begin
      MaxW := pNeuronList[NeuronCount].Weights.GetMax();
      MinW := pNeuronList[NeuronCount].Weights.GetMin();
    end;

    if MinW < MaxW then
    begin
      vDisplay.NeuronalWeightToImg(MaxW, MinW, color_encoding);

      LoadVolumeIntoTImage(vDisplay, pImage[NeuronCount + startImage], color_encoding);
      pImage[NeuronCount + startImage].Width := filterSize;
      pImage[NeuronCount + startImage].Height := filterSize;
    end;
  end;
  vDisplay.Free;
end;

procedure LoadTinyImageIntoTImage(var TI: TTinyImage; var Image: TImage);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Image.Canvas.Pixels[J, I] := {$IFDEF FPC}RGBToColor{$ELSE}RGB{$ENDIF}(TI.R[I, J], TI.G[I, J], TI.B[I, J]);
    end;
  end;
end;

procedure LoadTISingleChannelIntoImage(var TI: TTinySingleChannelImage;
  var Image: TImage);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Image.Canvas.Pixels[J, I] :=
        {$IFDEF FPC}RGBToColor{$ELSE}RGB{$ENDIF}(TI.Grey[I, J], TI.Grey[I, J], TI.Grey[I, J]);
    end;
  end;
end;

end.

