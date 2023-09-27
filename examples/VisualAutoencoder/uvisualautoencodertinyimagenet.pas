{
unit uvisualgan
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
}

unit uvisualautoencodertinyimagenet;

{$mode objfpc}{$H+}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, Menus, neuralnetwork, neuralvolumev, neuraldatasets, neuraldatasetsv,
  neuralvolume, MTProcs, math, neuralfit

  {$ifdef OpenCL}
  , neuralopencl
  {$endif}
  ;

const
  csLearningRates: array[0..2] of TNeuralFloat = (1, 0.1, 0.01);
  csGeneratorInputSize = 4;
  csMaxDiscriminatorError = 0.1;

type

  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    ChkRunOnGPU: TCheckBox;
    ChkBigNetwork: TCheckBox;
    ComboLearningRate: TComboBox;
    ComboComplexity: TComboBox;
    GrBoxNeurons: TGroupBox;
    ImgSample: TImage;
    LabClassRate: TLabel;
    LabComplexity: TLabel;
    LabLearningRate: TLabel;
    RadLAB: TRadioButton;
    RadRGB: TRadioButton;
    procedure ButLearnClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    FDisplay: TNNetVolume;
    FImageCnt: integer;
    iEpochCount, iEpochCountAfterLoading: integer;
    FAutoencoder: THistoricalNets;
    aImage: array of TImage;
    aLabelX, aLabelY: array of TLabel;
    FBaseName: string;
    FColorEncoding: byte;
    {$ifdef OpenCL}
    FEasyOpenCL: TEasyOpenCL;
    FHasOpenCL: boolean;
    {$endif}

    FCritSec: TRTLCriticalSection;
    FFit: TNeuralDataLoadingFit;
    FTrainImages: TClassesAndElements;
    procedure GetTrainingData(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure AutoencoderOnAfterEpoch(Sender: TObject);
    procedure AutoencoderOnAfterStep(Sender: TObject);
    procedure AutoencoderOnStart(Sender: TObject);
    procedure AutoencoderAugmentation(pInput: TNNetVolume; ThreadId: integer);
    procedure Learn(Sender: TObject);
    procedure SaveScreenshot(filename: string);
    procedure DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
    procedure SendStop;
  public
    procedure ProcessMessages();
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

uses strutils, LCLIntf, LCLType;


{ TFormVisualLearning }

procedure TFormVisualLearning.ButLearnClick(Sender: TObject);
begin
  if (FRunning) then
  begin
    SendStop;
  end
  else
  begin
    FRunning := true;
    ButLearn.Caption := 'Stop';
    ChkBigNetwork.Enabled := false;
    Learn(Sender);
    ChkBigNetwork.Enabled := true;
    ButLearn.Caption := 'Restart';
    FRunning := false;
  end;
end;

procedure TFormVisualLearning.FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  SendStop;
end;

procedure TFormVisualLearning.FormCreate(Sender: TObject);
begin
  FRunning := false;
  FFit := TNeuralDataLoadingFit.Create();
  FTrainImages := TClassesAndElements.Create();
  InitCriticalSection(FCritSec);
  FDisplay := TNNetVolume.Create();
  FImageCnt := 0;
  CreateAscentImages
  (
    GrBoxNeurons,
    aImage, aLabelX, aLabelY,
    {ImageCount=}32,
    {InputSize=}64, {displaySize=}128, {ImagesPerRow=}8
  );
  {$ifdef OpenCL}
  FEasyOpenCL := TEasyOpenCL.Create();
  {$else}
  ChkRunOnGPU.Visible := false;
  {$endif}
end;

procedure TFormVisualLearning.FormDestroy(Sender: TObject);
begin
  SendStop;
  while FFit.Running do Application.ProcessMessages;
  while FRunning do Application.ProcessMessages;
  FreeNeuronImages(aImage, aLabelX, aLabelY);
  DoneCriticalSection(FCritSec);
  FDisplay.Free;
  {$ifdef OpenCL}FEasyOpenCL.Free;{$endif}
  FFit.Free;
  FTrainImages.Free;
end;

procedure TFormVisualLearning.DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
begin
  FDisplay.Resize(ImgInput);
  FDisplay.Copy(ImgInput);
  FDisplay.ForceMaxRange(2);

  //Debug only: FDisplay.PrintDebugChannel();

  FDisplay.NeuronalInputToRgbImg(color_encoding);
  LoadVolumeIntoTImage(FDisplay, aImage[FImageCnt]);
  aImage[FImageCnt].Width := 128;
  aImage[FImageCnt].Height := 128;
  ProcessMessages();
  FImageCnt := (FImageCnt + 1) mod Length(aImage);
end;

procedure TFormVisualLearning.AutoencoderOnStart(Sender: TObject);
begin
  // TODO
end;

procedure TFormVisualLearning.SendStop;
begin
  WriteLn('Sending STOP request');
  FFit.ShouldQuit := true;
end;

procedure TFormVisualLearning.Learn( Sender: TObject);
var
  NeuronMultiplier: integer;
begin
  {$ifdef OpenCL}
  FHasOpenCL := false;
  if ChkRunOnGPU.Checked then
  begin
    if FEasyOpenCL.GetPlatformCount() > 0 then
    begin
      FEasyOpenCL.SetCurrentPlatform(FEasyOpenCL.PlatformIds[0]);
      if FEasyOpenCL.GetDeviceCount() > 0 then
      begin
        FHasOpenCL := true;
      end;
    end;
  end;
  {$endif}
  if ChkBigNetwork.Checked
    then NeuronMultiplier := 2
    else NeuronMultiplier := 1;
  FBaseName := 'IMAGEART-v1.1-'+IntToStr(NeuronMultiplier)+'-';
  if RadRGB.Checked then
  begin
    FColorEncoding := csEncodeRGB;
    FBaseName += 'RGB-';
  end
  else
  begin
    FColorEncoding := csEncodeLAB;
    FBaseName += 'LAB-';
  end;
  Self.Height := GrBoxNeurons.Top + GrBoxNeurons.Height + 10;
  Self.Width  := GrBoxNeurons.Left + GrBoxNeurons.Width + 10;
  ProcessMessages();
  if FTrainImages.Count = 0 then
  begin
    WriteLn('Loading Tiny ImageNet 200 file names.');
    FTrainImages.LoadFoldersAsClassesProportional('tiny-imagenet-200/train/','images',0,0.2);
    WriteLn('Tiny ImageNet 200 loaded classes: ',FTrainImages.Count,'. File names: ', FTrainImages.CountElements(),'.');
    WriteLn('Loading Tiny ImageNet 200 images.');
    FTrainImages.LoadImages(FColorEncoding);
    WriteLn('Loaded.');
  end;
  iEpochCount := 0;
  iEpochCountAfterLoading := 0;

  writeln('Creating Neural Networks...');
  FAutoencoder := THistoricalNets.Create();

  if Not(FileExists(FBaseName+'autoencoder.nn')) then
  begin
    WriteLn('Creating auto encoder.');
    FAutoencoder.AddLayer([
      TNNetInput.Create(64, 64, 3),
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}2,{SuppressBias=}1), //32x32
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}2,{SuppressBias=}1), //16x16
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}2,{SuppressBias=}1), //8x8
      TNNetConvolution.Create({Features=}64 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}2,{SuppressBias=}1), //4x4
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),

      TNNetUpsample.Create(), //8x8
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetUpsample.Create(), //16x16
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetUpsample.Create(), //32x32
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}128 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetUpsample.Create(), //64x64
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolution.Create({Features=}32 * NeuronMultiplier,{FeatureSize=}3,{Padding=}1,{Stride=}1,{SuppressBias=}1),
      TNNetConvolutionLinear.Create({Features=}3,{FeatureSize=}1,{Padding=}0,{Stride=}1,{SuppressBias=}0),
      TNNetReLUL.Create(-40, +40, 0) // Protection against overflow
    ]);
  end
  else
  begin
    WriteLn('Loading auto encoder.');
    FAutoencoder.LoadFromFile(FBaseName+'autoencoder.nn');
  end;
  FAutoencoder.DebugStructure();

  FFit.OnAfterEpoch := @Self.AutoencoderOnAfterEpoch;
  FFit.OnAfterStep := @Self.AutoencoderOnAfterStep;
  FFit.OnStart := @Self.AutoencoderOnStart;
  FFit.LearningRateDecay := 0.0;
  FFit.L2Decay := 0.0;
  FFit.AvgWeightEpochCount := 1;
  FFit.InitialLearningRate := 0.0001;
  FFit.ClipDelta := 0.01;
  FFit.FileNameBase := FBaseName+'autoencoder';
  FFit.EnableBipolar99HitComparison();
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FFit.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
    FAutoencoder.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  //Debug only: FFit.MaxThreadNum := 1;
  FFit.FitLoading(FAutoencoder, {EpochSize=}FTrainImages.CountElements(), 0, 0, {Batch=}64, {Epochs=}35000, @GetTrainingData, nil, nil); // This line does the same as above

  FAutoencoder.Free;
end;

procedure TFormVisualLearning.GetTrainingData(Idx: integer;
  ThreadId: integer; pInput, pOutput: TNNetVolume);
var
  ClassId, ImageId: integer;
begin
  ClassId := FTrainImages.GetRandomClassId();
  ImageId := FTrainImages.List[ClassId].GetRandomIndex();
  pInput.Copy(FTrainImages.List[ClassId].List[ImageId]);
  if Random(1000)>500 then pInput.FlipX();
  pOutput.Copy(pInput);
end;

procedure TFormVisualLearning.AutoencoderOnAfterEpoch(Sender: TObject);
begin
  WriteLn('Finished epoch number: ', FFit.CurrentEpoch);
end;

procedure TFormVisualLearning.AutoencoderOnAfterStep(Sender: TObject);
var
  ClassId, ImageId: integer;
begin
  LabClassRate.Caption := PadLeft(IntToStr(Round(FFit.TrainingAccuracy*100))+'%',4);
  ProcessMessages();
  if FFit.CurrentStep mod FFit.ThreadNN.Count = 0 then
  begin
    ClassId := FTrainImages.GetRandomClassId();
    ImageId := FTrainImages.List[ClassId].GetRandomIndex();
    FFit.NN.Compute(FTrainImages.List[ClassId].List[ImageId]);
    DisplayInputImage(FFit.NN.GetLastLayer().Output, 0);
    //Debug only: FFit.NN.GetLastLayer().Output.PrintDebug();
  end;
end;

procedure TFormVisualLearning.AutoencoderAugmentation(pInput: TNNetVolume;
  ThreadId: integer);
begin
end;

procedure TFormVisualLearning.SaveScreenshot(filename: string);
begin
  try
    WriteLn(' Saving ',filename,'.');
    SaveHandleToBitmap(filename, Self.Handle);
  except
    // Nothing can be done if this fails.
  end;
end;

procedure TFormVisualLearning.ProcessMessages();
begin
  Application.ProcessMessages();
end;

end.
