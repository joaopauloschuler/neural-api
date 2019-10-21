(*
ucifar10
Copyright (C) 2017 Joao Paulo Schwarz Schuler

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

unit neuraldatasets;
{
  neuraldatasets Free Pascal/Lazarus Library by Joao Paulo Schwarz Schuler
  Conscious Artificial Intelligence Project
  https://sourceforge.net/projects/cai/

// This is an example showing how to use pascal unit ucifar10:
procedure TForm1.Button1Click(Sender: TObject);
var
  Img: TTinyImage;
  cifarFile: TTInyImageFile;
begin
  AssignFile(cifarFile, 'C:\cifar-10\data_batch_1.bin');
  Reset(cifarFile);

  while not EOF(cifarFile) do
  begin
    Read(cifarFile, Img);
    // Image1 is a TImage component.
    LoadTinyImageIntoTImage(Img, Image1);
    Label1.Caption := csTinyImageLabel[Img.bLabel];
    Application.ProcessMessages;
    Sleep(5000);
  end;
  CloseFile(cifarFile);
end;
}

interface

uses
  neuralvolume, neuralnetwork;

type
  TTinyImageChannel = packed array [0..31, 0..31] of byte;
  TTinyImageChannel1D = packed array [0..32 * 32 - 1] of byte;
  TMNistImage = packed array [0..27, 0..27] of byte;

  TTinyImage = packed record
    bLabel: byte;
    R, G, B: TTinyImageChannel;
  end;

  TCifar100Image = packed record
    bCoarseLabel: byte;
    bFineLabel: byte;
    R, G, B: TTinyImageChannel;
  end;

  TTInyImageFile = file of TTinyImage;
  TCifar100File = file of TCifar100Image;

  // Useful for gray scale image
  TTinySingleChannelImage = packed record
    bLabel: byte;
    Grey: TTinyImageChannel;
  end;

  TTinySingleChannelImage1D = packed record
    bLabel: byte;
    Grey: TTinyImageChannel1D;
  end;

  TTinySingleChannelImagePtr = ^TTinySingleChannelImage;
  TTinySingleChannelImage1DPtr = ^TTinySingleChannelImage1D;

const
  csTinyImageLabel: array[0..9] of string =
    (
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
    );

  csMachineAnimalCifar10Pos: array[0..9] of integer =
  (
    0,
    1,
    8, // new bird position is 8. This position is now ship.
    9, // new cat position  is 9. This position is now truck.
    4,
    5,
    6,
    7,
    2, // new ship position is 2. This position is now bird.
    3  // new truck         is 3. This position is now cat.
  );

  csMachineAnimalCifar10Labels: array[0..9] of string =
  (
    'airplane',
    'automobile',
    'ship',  // used to be bird
    'truck', // used to be cat
    'deer',  // used to be deer
    'dog',   // used to be dog
    'frog',  // used to be frog
    'horse', // used to be horse
    'bird',  // used to be ship
    'cat'    // used to be truck
  );

// Writes the header of a confusion matrix into a CSV file
procedure ConfusionWriteCSVHeader(var CSVConfusion: TextFile; Labels: array of string);

// Writes a confusion matrix into CSV file
procedure ConfusionWriteCSV(var CSVConfusion: TextFile; Vol: TNNetVolume; Digits: integer);

// Loads a TinyImage into TNNetVolume.
procedure LoadTinyImageIntoNNetVolume(var TI: TTinyImage; Vol: TNNetVolume); overload;
procedure LoadTinyImageIntoNNetVolume(var TI: TCifar100Image; Vol: TNNetVolume); overload;
procedure LoadTinyImageIntoNNetVolume(var TI: TMNistImage; Vol: TNNetVolume); overload;

// Loads a volume into a tiny image.
procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TTinyImage); overload;
procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TCifar100Image); overload;

// Loads a SingleChannelTinyImage Into TNNetVolue.
procedure LoadTinySingleChannelIntoNNetVolume(var SC: TTinySingleChannelImage; Vol: TNNetVolume);

// Creates a gray scale tiny image
procedure TinyImageCreateGrey(var TI: TTinyImage; var TIGrey: TTinySingleChannelImage);

// Calculates Horizontal Edges
procedure TinyImageHE(var TI, TIHE: TTinySingleChannelImage);

// Calculates Vertical Edges
procedure TinyImageVE(var TI, TIVE: TTinySingleChannelImage);

//Zeroes all pixels that have a small distance to the number 128
procedure TinyImageRemoveZeroGradient(var TI: TTinySingleChannelImage; distance: byte);

// Calculates Horizontal and Vertical Edges
procedure TinyImageHVE(var TI, TIHE: TTinySingleChannelImage);

// This function transforms a 2D TinyImage into 1D TinyImage
function TinyImageTo1D(var TI: TTinySingleChannelImage): TTinySingleChannelImage1D;

// creates CIFAR10 volumes required for training, testing and validation
procedure CreateCifar10Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList);

// creates CIFAR100 volumes required for training, testing and validation
procedure CreateCifar100Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList; color_encoding: byte = csEncodeRGB;
  Verbose:boolean = true);

procedure CreateMNISTVolumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList;
  TrainFileName, TestFileName: string;
  Verbose:boolean = true;
  IsFashion:boolean = false);

// loads a CIFAR10 into TNNetVolumeList
procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; idx:integer; base_pos:integer = 0; color_encoding: byte = csEncodeRGB); overload;
procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; fileName:string; base_pos:integer = 0; color_encoding: byte = csEncodeRGB); overload;

// loads a CIFAR100 into TNNetVolumeList
procedure loadCifar100Dataset(ImgVolumes: TNNetVolumeList; fileName:string;
  color_encoding: byte = csEncodeRGB; Verbose:boolean = true);

// loads MNIST pair of files (image and labels) into ImgVolumes.
procedure loadMNISTDataset(ImgVolumes: TNNetVolumeList; fileName:string;
  Verbose:boolean = true; IsFashion:boolean = false;
  MaxLabel: integer = 10); overload;

// These functions return TRUE if the dataset is found or an error message otherwise
function CheckCIFARFile():boolean;
function CheckCIFAR100File():boolean;
function CheckMNISTFile(fileName:string; IsFasion:boolean = false):boolean;

// This function tests a neural network on the passed ImgVolumes
procedure TestBatch
(
  NN: TNNet; ImgVolumes: TNNetVolumeList; SampleSize: integer;
  out Rate, Loss, ErrorSum: TNeuralFloat
);

// This function translates the original CIFAR10 labels to Animal/Machine labels.
procedure TranslateCifar10VolumesToMachineAnimal(VolumeList: TNNetVolumeList);

{$IFNDEF FPC}
function SwapEndian(I:integer):integer;
{$ENDIF}

implementation

uses SysUtils, math;

{$IFNDEF FPC}
function SwapEndian(I:integer):integer;
begin
  result := Swap(I)
end;
{$ENDIF}

procedure TranslateCifar10VolumesToMachineAnimal(VolumeList: TNNetVolumeList);
var
  Volume: TNNetVolume;
begin
  for Volume in VolumeList do
  begin
    Volume.Tag := csMachineAnimalCifar10Pos[Volume.Tag];
  end;
end;

procedure CreateCifar10Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList);
var
  I: integer;
begin
  ImgTrainingVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();

  // creates required volumes to store images
  for I := 0 to 39999 do
  begin
    ImgTrainingVolumes.Add(TNNetVolume.Create());
  end;

  for I := 0 to 9999 do
  begin
    ImgValidationVolumes.Add(TNNetVolume.Create());
    ImgTestVolumes.Add(TNNetVolume.Create());
  end;

  loadCifar10Dataset(ImgTrainingVolumes, 1, 0, 0);
  loadCifar10Dataset(ImgTrainingVolumes, 2, 10000, 0);
  loadCifar10Dataset(ImgTrainingVolumes, 3, 20000, 0);
  loadCifar10Dataset(ImgTrainingVolumes, 4, 30000, 0);
  loadCifar10Dataset(ImgValidationVolumes, 5, 0, 0);
  loadCifar10Dataset(ImgTestVolumes, 'test_batch.bin', 0);
end;

procedure CreateCifar100Volumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList; color_encoding: byte = csEncodeRGB;
  Verbose:boolean = true);
var
  I, HalfSize, LastElement: integer;
begin
  ImgTrainingVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();
  loadCifar100Dataset(ImgTrainingVolumes, 'train.bin', color_encoding, Verbose);
  loadCifar100Dataset(ImgValidationVolumes, 'test.bin', color_encoding, Verbose);
  ImgValidationVolumes.FreeObjects := false;
  HalfSize := ImgValidationVolumes.Count div 2;
  LastElement := ImgValidationVolumes.Count - 1;
  for I := LastElement downto HalfSize do
  begin
    ImgTestVolumes.Add(ImgValidationVolumes[I]);
    ImgValidationVolumes.Delete(I);
  end;
  ImgValidationVolumes.FreeObjects := true;
end;

procedure CreateMNISTVolumes(out ImgTrainingVolumes, ImgValidationVolumes,
  ImgTestVolumes: TNNetVolumeList; TrainFileName, TestFileName: string;
  Verbose: boolean = true; IsFashion:boolean = false);
var
  I, HalfSize, LastElement: integer;
begin
  ImgTrainingVolumes := TNNetVolumeList.Create();
  ImgValidationVolumes := TNNetVolumeList.Create();
  ImgTestVolumes := TNNetVolumeList.Create();
  loadMNISTDataset(ImgTrainingVolumes, TrainFileName, Verbose, IsFashion);
  loadMNISTDataset(ImgValidationVolumes, TestFileName, Verbose, IsFashion);
  ImgValidationVolumes.FreeObjects := false;
  HalfSize := ImgValidationVolumes.Count div 2;
  LastElement := ImgValidationVolumes.Count - 1;
  for I := LastElement downto HalfSize do
  begin
    ImgTestVolumes.Add(ImgValidationVolumes[I]);
    ImgValidationVolumes.Delete(I);
  end;
  ImgValidationVolumes.FreeObjects := true;
end;

// loads a CIFAR10 into TNNetVolumeList
procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; idx:integer; base_pos:integer = 0; color_encoding: byte = csEncodeRGB);
var
  fileName: string;
begin
  fileName := 'data_batch_'+IntToStr(idx)+'.bin';
  loadCifar10Dataset(ImgVolumes, fileName, base_pos, color_encoding);
end;

// loads a CIFAR10 into TNNetVolumeList
procedure loadCifar10Dataset(ImgVolumes: TNNetVolumeList; fileName:string; base_pos:integer = 0; color_encoding: byte = csEncodeRGB);
var
  I, ImgPos: integer;
  Img: TTinyImage;
  cifarFile: TTInyImageFile;
  AuxVolume: TNNetVolume;
  pMin, pMax: TNeuralFloat;
  globalMin0, globalMax0: TNeuralFloat;
  globalMin1, globalMax1: TNeuralFloat;
  globalMin2, globalMax2: TNeuralFloat;

begin
  Write('Loading 10K images from file "'+fileName+'" ...');
  AssignFile(cifarFile, fileName);
  Reset(cifarFile);
  AuxVolume := TNNetVolume.Create();

  globalMin0 := 0;
  globalMax0 := 0;
  globalMin1 := 0;
  globalMax1 := 0;
  globalMin2 := 0;
  globalMax2 := 0;

  // binary CIFAR 10 file contains 10K images
  for I := 0 to 9999 do
  begin
    Read(cifarFile, Img);
    ImgPos := I + base_pos;
    LoadTinyImageIntoNNetVolume(Img, ImgVolumes[ImgPos]);

    if (color_encoding = csEncodeGray) then
    begin
      AuxVolume.Copy(ImgVolumes[ImgPos]);
      ImgVolumes[ImgPos].GetGrayFromRgb(AuxVolume);
    end;

    ImgVolumes[ImgPos].RgbImgToNeuronalInput(color_encoding);

    ImgVolumes[ImgPos].GetMinMaxAtDepth(0, pMin, pMax); //WriteLn  (I:8,' - #0 Min:',pMin, ' Max:',pMax);

    globalMin0 := Math.Min(pMin, globalMin0);
    globalMax0 := Math.Max(pMax, globalMax0);

    if (ImgVolumes[ImgPos].Depth >= 2) then
    begin
      ImgVolumes[ImgPos].GetMinMaxAtDepth(1, pMin, pMax); //Write  (' #1 Min:',pMin, ' Max:',pMax);

      globalMin1 := Math.Min(pMin, globalMin1);
      globalMax1 := Math.Max(pMax, globalMax1);
    end;

    if (ImgVolumes[ImgPos].Depth >= 3) then
    begin
      ImgVolumes[ImgPos].GetMinMaxAtDepth(2, pMin, pMax); //WriteLn(' #2 Min:',pMin, ' Max:',pMax);

      globalMin2 := Math.Min(pMin, globalMin2);
      globalMax2 := Math.Max(pMax, globalMax2);
    end;
  end;

  Write(' GLOBAL MIN MAX ', globalMin0:8:4,globalMax0:8:4,globalMin1:8:4,globalMax1:8:4,globalMin2:8:4,globalMax2:8:4);

  AuxVolume.Free;
  CloseFile(cifarFile);
  WriteLn(' Done.');
end;

procedure loadCifar100Dataset(ImgVolumes: TNNetVolumeList; fileName: string;
  color_encoding: byte = csEncodeRGB; Verbose:boolean = true);
var
  Img: TCifar100Image;
  cifarFile: TCifar100File;
  AuxVolume: TNNetVolume;
begin
  if Verbose then Write('Loading images from CIFAR-100 file "'+fileName+'" ...');
  AssignFile(cifarFile, fileName);
  Reset(cifarFile);

  // binary CIFAR 10 file contains 10K images
  while not(Eof(cifarFile)) do
  begin
    AuxVolume := TNNetVolume.Create();
    Read(cifarFile, Img);
    LoadTinyImageIntoNNetVolume(Img, AuxVolume);
    AuxVolume.RgbImgToNeuronalInput(color_encoding);
    ImgVolumes.Add(AuxVolume);

    //TODO: add treatment for CIFAR100
    (*if (color_encoding = csEncodeGray) then
    begin
      AuxVolume.Copy(ImgVolumes[ImgPos]);
      ImgVolumes[ImgPos].GetGrayFromRgb(AuxVolume);
    end;*)
  end;

  CloseFile(cifarFile);
  if Verbose then WriteLn(' ',ImgVolumes.Count, ' images loaded.');
end;

procedure loadMNISTDataset(ImgVolumes: TNNetVolumeList; fileName: string;
  Verbose: boolean = true; IsFashion:boolean = false;
  MaxLabel: integer = 10);
var
  fileNameLabels, fileNameImg: string;
  fileLabels, fileImg: THandle;
  LabelItems, LabelMagic: integer;
  ImgMagic, ImgItems, ImgRows, ImgCols: Integer;
  MNistImg: TMNistImage;
  ImgCnt: integer;
  LabelByte: byte;
  Vol: TNNetVolume;
  Separator: char;
begin
  if IsFashion
    then Separator := '-'
    else Separator := '.';

  fileNameLabels := fileName + '-labels'+Separator+'idx1-ubyte';
  fileNameImg := fileName + '-images'+Separator+'idx3-ubyte';

  if not FileExists(fileNameLabels) then
  begin
    if Verbose then WriteLn('Labels file not found:', fileNameLabels);
    exit;
  end;

  if not FileExists(fileNameImg) then
  begin
    if Verbose then WriteLn('Image file not found:', fileNameImg);
    exit;
  end;

  fileLabels := FileOpen(fileNameLabels, fmOpenRead);
  fileImg := FileOpen(fileNameImg, fmOpenRead);

  FileRead(fileLabels, LabelMagic, 4);
  FileRead(fileLabels, LabelItems, 4);

  FileRead(fileImg, ImgMagic, 4);
  FileRead(fileImg, ImgItems, 4);
  FileRead(fileImg, ImgRows, 4);
  FileRead(fileImg, ImgCols, 4);

  LabelItems := SwapEndian(LabelItems);
  LabelMagic := SwapEndian(LabelMagic);
  ImgMagic := SwapEndian(ImgMagic);
  ImgItems := SwapEndian(ImgItems);
  ImgRows := SwapEndian(ImgRows);
  ImgCols := SwapEndian(ImgCols);

  if Verbose then
    WriteLn
    (
      'File:', fileName,
      ' Labels:', LabelItems, ' Images:', ImgItems,
      ' Rows:', ImgRows, ' Cols:', ImgCols
    );

  for ImgCnt := 1 to ImgItems do
  begin
    FileRead(fileImg, MNistImg, SizeOf(TMNistImage));
    FileRead(fileLabels, LabelByte, 1);
    Vol := TNNetVolume.Create();
    LoadTinyImageIntoNNetVolume(MNistImg, Vol);
    Vol.Divi(64);
    Vol.Add(-2);
    Vol.Tag := LabelByte;
    if LabelByte >= MaxLabel then
    begin
      WriteLn('Error loading Label:', LabelByte,' at index ', ImgCnt);
      Vol.Free;
    end
    else
    begin
      ImgVolumes.Add(Vol);
    end;
  end;

  FileClose(fileLabels);
  FileClose(fileImg);
end;

// This function returns TRUE if data_batch_1.bin and error message otherwise
function CheckCIFARFile():boolean;
begin
  Result := true;
  if not (FileExists('data_batch_1.bin')) then
  begin
    WriteLn('File Not Fount: data_batch_1.bin');
    WriteLn('Please download it from here: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz');
    //TODO: automatically download file
    Result := false;
  end;
end;

function CheckCIFAR100File(): boolean;
begin
  Result := true;
  if not (FileExists('train.bin')) then
  begin
    WriteLn('File Not Fount: train.bin');
    WriteLn('Please download it from here: https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz');
    //TODO: automatically download file
    Result := false;
  end
  else if not (FileExists('test.bin')) then
  begin
    WriteLn('File Not Fount: test.bin');
    WriteLn('Please download it from here: https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz');
    //TODO: automatically download file
    Result := false;
  end;
end;

function CheckMNISTFile(fileName: string; IsFasion:boolean = false): boolean;
var
  Separator: char;
  Link: string;
  FullFileName: string;
begin
  Result := true;
  if IsFasion then
  begin
    Separator := '-';
    Link := 'https://github.com/zalandoresearch/fashion-mnist';
  end
  else
  begin
    Separator := '.';
    Link := 'http://yann.lecun.com/exdb/mnist/';
  end;
  FullFileName := fileName+'-labels'+Separator+'idx1-ubyte';
  if not (FileExists(FullFileName)) then
  begin
    WriteLn('File Not Fount:', FullFileName);
    WriteLn('Please download from ', Link);
    //TODO: automatically download file
    Result := false;
    exit;
  end;
  FullFileName := fileName+'-images'+Separator+'idx3-ubyte';
  if not (FileExists(FullFileName)) then
  begin
    WriteLn('File Not Fount:', FullFileName);
    WriteLn('Please download from ', Link);
    //TODO: automatically download file
    Result := false;
  end;
end;

procedure ConfusionWriteCSVHeader(var CSVConfusion: TextFile; Labels: array of string);
var
  I: integer;
begin
  for I := Low(Labels) to High(Labels) do
  begin
    if I > 0 then Write(CSVConfusion, ',');
    Write(CSVConfusion, Labels[I]);
  end;
  WriteLn(CSVConfusion);
end;

procedure ConfusionWriteCSV(var CSVConfusion: TextFile; Vol: TNNetVolume; Digits: integer);
var
  I, J: integer;
begin
  for I := 0 to Vol.SizeY - 1 do
  begin
    for J := 0 to Vol.Depth - 1 do
    begin
      if J > 0 then Write(CSVConfusion, ',');
      Write(CSVConfusion, Round(Vol[0, I, J]):Digits);
    end;
    WriteLn(CSVConfusion);
  end;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TTinyImage; Vol: TNNetVolume);
var
  I, J: integer;
begin
  Vol.ReSize(32,32,3);
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Vol[J, I, 0] := TI.R[I, J];
      Vol[J, I, 1] := TI.G[I, J];
      Vol[J, I, 2] := TI.B[I, J];
    end;
  end;
  Vol.Tag := TI.bLabel;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TCifar100Image; Vol: TNNetVolume);
var
  I, J: integer;
begin
  Vol.ReSize(32,32,3);
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Vol[J, I, 0] := TI.R[I, J];
      Vol[J, I, 1] := TI.G[I, J];
      Vol[J, I, 2] := TI.B[I, J];
    end;
  end;
  Vol.Tags[0] := TI.bFineLabel;
  Vol.Tags[1] := TI.bCoarseLabel;
end;

procedure LoadTinyImageIntoNNetVolume(var TI: TMNistImage; Vol: TNNetVolume);
var
  I, J: integer;
begin
  Vol.ReSize(28, 28, 1);
  for I := 0 to 27 do
  begin
    for J := 0 to 27 do
    begin
      Vol[J, I, 0] := TI[I, J];
    end;
  end;
end;

procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TTinyImage);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      TI.R[I, J] := Vol.AsByte[J, I, 0];
      TI.G[I, J] := Vol.AsByte[J, I, 1];
      TI.B[I, J] := Vol.AsByte[J, I, 2];
    end;
  end;
  TI.bLabel := Vol.Tag;
end;

procedure LoadNNetVolumeIntoTinyImage(Vol: TNNetVolume; var TI: TCifar100Image);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      TI.R[I, J] := Vol.AsByte[J, I, 0];
      TI.G[I, J] := Vol.AsByte[J, I, 1];
      TI.B[I, J] := Vol.AsByte[J, I, 2];
    end;
  end;
  TI.bCoarseLabel := Vol.Tags[0];
  TI.bFineLabel := Vol.Tags[1];
end;

procedure LoadTinySingleChannelIntoNNetVolume(var SC: TTinySingleChannelImage;
  Vol: TNNetVolume);
var
  I, J: integer;
begin
  Vol.ReSize(32,32,1);
  Vol.Tag := SC.bLabel;
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      Vol[I, J, 0] := SC.Grey[I,J];
    end;
  end;
end;

procedure TinyImageCreateGrey(var TI: TTinyImage; var TIGrey: TTinySingleChannelImage);
var
  I, J: integer;
begin
  TIGrey.bLabel := TI.bLabel;
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      TIGrey.Grey[I, J] := (TI.R[I, J] + TI.G[I, J] + TI.B[I, J]) div 3;
    end;
  end;
end;

procedure TinyImageVE(var TI, TIVE: TTinySingleChannelImage);
var
  I, J: integer;
  aux: integer;
begin
  TIVE.bLabel := TI.bLabel;
  for I := 1 to 31 do
  begin
    TIVE.Grey[0, I] := 128;
    for J := 0 to 31 do
    begin
      aux := (TI.Grey[I, J] - TI.Grey[I - 1, J]) * 3 + 128;

      if (aux < 0) then
        aux := 0;
      if (aux > 255) then
        aux := 255;

      TIVE.Grey[I, J] := aux;
    end;
  end;
end;

procedure TinyImageHE(var TI, TIHE: TTinySingleChannelImage);
var
  I, J: integer;
  aux: integer;
begin
  TIHE.bLabel := TI.bLabel;
  for I := 0 to 31 do
  begin
    TIHE.Grey[I, 0] := 128;
    for J := 1 to 31 do
    begin
      aux := (TI.Grey[I, J] - TI.Grey[I, J - 1]) * 3 + 128;

      if (aux < 0) then
        aux := 0;
      if (aux > 255) then
        aux := 255;

      TIHE.Grey[I, J] := aux;
    end;
  end;
end;

procedure TinyImageRemoveZeroGradient(var TI: TTinySingleChannelImage; distance: byte);
var
  I, J: integer;
begin
  for I := 0 to 31 do
  begin
    for J := 0 to 31 do
    begin
      if abs(TI.Grey[I, J] - 128) < distance then
        TI.Grey[I, J] := 0;
    end;
  end;
end;

procedure TinyImageHVE(var TI, TIHE: TTinySingleChannelImage);
var
  I, J: integer;
  aux, aux1, aux2: integer;
begin
  TIHE.bLabel := TI.bLabel;
  for I := 1 to 31 do
  begin
    TIHE.Grey[I, 0] := 128;
    TIHE.Grey[0, I] := 128;
    for J := 1 to 31 do
    begin
      aux1 := (TI.Grey[I, J] - TI.Grey[I, J - 1]) + 128;
      aux2 := (TI.Grey[I, J] - TI.Grey[I - 1, J]) + 128;

      if (abs(aux1 - 128) > abs(aux2 - 128)) then
      begin
        aux := aux1;
      end
      else
      begin
        aux := aux2;
      end;

      if (aux < 0) then
        aux := 0;
      if (aux > 255) then
        aux := 255;

      TIHE.Grey[I, J] := aux;
    end;
  end;
end;

function TinyImageTo1D(var TI: TTinySingleChannelImage): TTinySingleChannelImage1D;
var
  TIPtr: TTinySingleChannelImage1DPtr;
begin
  TIPtr := addr(TI);
  Result := TIPtr^;
end;

// this function tests a neural network on the passed ImgVolumes
procedure TestBatch
(
  NN: TNNet; ImgVolumes: TNNetVolumeList; SampleSize: integer;
  out Rate, Loss, ErrorSum: TNeuralFloat
);
var
  I, ImgIdx: integer;
  hit, miss: integer;
  pOutput, vOutput: TNNetVolume;
  bIsSoftmax: boolean;
  CurrentLoss : TNeuralFloat;
  OutputValue: TNeuralFloat;
  MaxID: integer;
begin
  pOutput := TNNetVolume.Create;
  vOutput := TNNetVolume.Create;

  hit  := 0;
  miss := 0;
  ErrorSum := 0;
  Loss := 0;
  Rate := 0;
  bIsSoftmax := false;

  if NN.Layers[NN.GetLastLayerIdx()] is TNNetSoftMax then
  begin
    bIsSoftmax := true;
  end;

  if SampleSize = 0 then
  begin
    MaxID := ImgVolumes.Count;
  end
  else
  begin
    MaxID := SampleSize;
  end;

  for I := 1 to MaxID do
  begin
    if SampleSize = 0 then
    begin
      ImgIdx := I - 1;
    end
    else
    begin
      ImgIdx := Random(ImgVolumes.Count);
    end;

    NN.Compute(ImgVolumes[ImgIdx]);
    NN.GetOutput(pOutput);

    ImgVolumes[ImgIdx].FlipX();
    NN.AddOutput(pOutput);

    if pOutput.GetClass() = ImgVolumes[ImgIdx].Tag then
    begin
      Inc(Hit);
    end
    else
    begin
      Inc(Miss);
    end;

    if (bIsSoftmax) then
    begin
      vOutput.SetClassForSoftMax( ImgVolumes[ImgIdx].Tag );
    end
    else
    begin
      vOutput.SetClassForReLU( ImgVolumes[ImgIdx].Tag );
    end;

    ErrorSum := ErrorSum + vOutput.SumDiff(pOutput);

    if (bIsSoftmax) then
    begin
      OutputValue := pOutput.FData[ ImgVolumes[ImgIdx].Tag ];
      if (OutputValue > 0) then
      begin
        CurrentLoss := -Ln(OutputValue);
      end
      else
      begin
        WriteLn('Error: invalid output value',OutputValue);
        CurrentLoss := 1;
      end;

      Loss := Loss + CurrentLoss;
    end;
  end;

  if (Hit > 0) then
  begin
    Rate := Hit / (Hit + Miss);
    Loss := Loss / (Hit + Miss);
  end;

  vOutput.Free;
  pOutput.Free;
end;

end.
