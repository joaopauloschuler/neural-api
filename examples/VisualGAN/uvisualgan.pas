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

unit uvisualgan;

{$mode objfpc}{$H+}

interface

uses
  {$ifdef unix}
  cmem, // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  ExtCtrls, Menus, neuralnetwork, neuralvolumev, neuraldatasets,
  neuralvolume, MTProcs, math, neuralfit

  {$ifdef OpenCL}
  , neuralopencl
  {$endif}
  ;

type

  { TFormVisualLearning }
  TFormVisualLearning = class(TForm)
    ButLearn: TButton;
    ChkRunOnGPU: TCheckBox;
    ChkBigNetwork: TCheckBox;
    ComboComplexity: TComboBox;
    GrBoxNeurons: TGroupBox;
    ImgSample: TImage;
    LabClassRate: TLabel;
    LabComplexity: TLabel;
    RadLAB: TRadioButton;
    RadRGB: TRadioButton;
    procedure ButLearnClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  private
    { private declarations }
    FRunning: boolean;
    vDisplay: TNNetVolume;
    ImgTrainingVolumes, ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;
    FRealPairs, FGeneratedPairs: TNNetVolumePairList;
    FImageCnt: integer;
    iEpochCount, iEpochCountAfterLoading: integer;
    Generative: THistoricalNets;
    FGeneratives: TNNetDataParallelism;
    Discriminator, DiscriminatorClone: TNNet;
    aImage: array of TImage;
    aLabelX, aLabelY: array of TLabel;
    FBaseName: string;
    FColorEncoding: byte;
    FRandomSizeX, FRandomSizeY, FRandomDepth: integer;
    {$ifdef OpenCL}FEasyOpenCL: TEasyOpenCL;{$endif}
    FHasOpenCL: boolean;

    FCritSec: TRTLCriticalSection;
    FFit: TNeuralDataLoadingFit;
    function GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
    procedure DiscriminatorOnAfterEpoch(Sender: TObject);
    procedure DiscriminatorOnAfterStep(Sender: TObject);
    procedure DiscriminatorAugmentation(pInput: TNNetVolume; ThreadId: integer);
    procedure Learn(Sender: TObject);
    procedure SaveScreenshot(filename: string);
    procedure BuildTrainingPairs();
    procedure DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
    procedure DiscriminatorOnStart(Sender: TObject);
    procedure SendStop;
  public
    procedure ProcessMessages();
  end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
{$R *.lfm}

uses strutils, LCLIntf, LCLType, neuraldatasetsv;


{ TFormVisualLearning }

procedure TFormVisualLearning.ButLearnClick(Sender: TObject);
begin
  if not CheckCIFARFile() then exit;

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
  InitCriticalSection(FCritSec);
  ImgTrainingVolumes := nil;
  ImgTestVolumes := nil;
  ImgValidationVolumes := nil;
  FGeneratives := nil;
  FRealPairs := TNNetVolumePairList.Create();
  FGeneratedPairs := TNNetVolumePairList.Create();
  vDisplay := TNNetVolume.Create();
  FImageCnt := 0;
  CreateAscentImages
  (
    GrBoxNeurons,
    aImage, aLabelX, aLabelY,
    {ImageCount=}128,
    {InputSize=}32, {FFilterSize=}64, {ImagesPerRow=}16
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
  if Assigned(ImgValidationVolumes) then ImgValidationVolumes.Free;
  if Assigned(ImgTestVolumes) then ImgTestVolumes.Free;
  if Assigned(ImgTrainingVolumes) then ImgTrainingVolumes.Free;
  FreeNeuronImages(aImage, aLabelX, aLabelY);
  DoneCriticalSection(FCritSec);
  FRealPairs.Free;
  FGeneratedPairs.Free;
  vDisplay.Free;
  {$ifdef OpenCL}FEasyOpenCL.Free;{$endif}
  FFit.Free;
end;

procedure TFormVisualLearning.DisplayInputImage(ImgInput: TNNetVolume; color_encoding: integer);
var
  pMin0, pMax0: TNeuralFloat;
  pMin1, pMax1: TNeuralFloat;
  pMin2, pMax2: TNeuralFloat;
begin
  vDisplay := TNNetVolume.Create(ImgInput);
  vDisplay.Copy(ImgInput);

  if color_encoding = csEncodeLAB then
  begin
    vDisplay.GetMinMaxAtDepth(0, pMin0, pMax0);
    vDisplay.GetMinMaxAtDepth(1, pMin1, pMax1);
    vDisplay.GetMinMaxAtDepth(2, pMin2, pMax2);
    pMax0 := Max(Abs(pMin0), Abs(pMax0));
    pMax1 := Max(Abs(pMin1), Abs(pMax1));
    pMax2 := Max(Abs(pMin2), Abs(pMax2));

    if pMax0 > 2 then
    begin
      vDisplay.MulAtDepth(0, 2/pMax0);
    end;

    if pMax1 > 2 then
    begin
      vDisplay.MulAtDepth(1, 2/pMax1);
    end;

    if pMax2 > 2 then
    begin
      vDisplay.MulAtDepth(2, 2/pMax2);
    end;
  end
  else if vDisplay.GetMaxAbs() > 2 then
  begin
    vDisplay.NormalizeMax(2);
  end;

  vDisplay.PrintDebugChannel();

  vDisplay.NeuronalInputToRgbImg(color_encoding);

  LoadVolumeIntoTImage(vDisplay, aImage[FImageCnt]);
  aImage[FImageCnt].Width := 64;
  aImage[FImageCnt].Height := 64;
  ProcessMessages();
  FImageCnt := (FImageCnt + 1) mod Length(aImage);
end;

procedure TFormVisualLearning.DiscriminatorOnStart(Sender: TObject);
begin
  FGeneratives := TNNetDataParallelism.Create(Generative, FFit.MaxThreadNum);
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FGeneratives.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  BuildTrainingPairs();
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
  FRandomSizeX :=  4;
  FRandomSizeY :=  4;
  FRandomDepth := StrToInt(ComboComplexity.Text);
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
  FBaseName := 'ART'+IntToStr(FRandomDepth)+'-'+IntToStr(NeuronMultiplier)+'-';
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
  if not(Assigned(ImgValidationVolumes)) then
  begin
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, FColorEncoding);
  end;

  iEpochCount := 0;
  iEpochCountAfterLoading := 0;

  writeln('Creating Neural Network...');
  Generative := THistoricalNets.Create();
  Discriminator := TNNet.Create();

  if Not(FileExists(FBaseName+'generative.nn')) then
  begin
    WriteLn('Creating generative.');
    Generative.AddLayer([
      TNNetInput.Create(FRandomSizeX, FRandomSizeY, FRandomDepth),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,1,1,0), //4x4
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetDeMaxPool.Create(2),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,1,1,0), //8x8
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetDeMaxPool.Create(4),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,5,2,1,0), //32x32
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionLinear.Create(3,3,1,1,0),
      TNNetReLUL.Create(-40, +40) // Protection against overflow
    ]);
    (*
    Generative.AddLayer([
       TNNetInput.Create(40, 40, 3),
       TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,0,1,0),
       TNNetMovingStdNormalization.Create(),
       TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,0,1,0),
       TNNetMovingStdNormalization.Create(),
       TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,0,1,0),
       TNNetMovingStdNormalization.Create(),
       TNNetConvolutionLinear.Create(3,3,0,1,0),
       TNNetReLUL.Create(-40, +40) // Protection against overflow
    ]);
    *)
    Generative.Layers[Generative.GetFirstImageNeuronalLayerIdx()].InitBasicPatterns();
  end
  else
  begin
    WriteLn('Loading generative.');
    Generative.LoadFromFile(FBaseName+'generative.nn');
  end;
  Generative.DebugStructure();
  Generative.SetLearningRate(0.001,0.9);
  Generative.SetL2Decay(0.0);

  if Not(FileExists(FBaseName+'discriminator.nn')) then
  begin
    WriteLn('Creating discriminator.');
    (*
    Discriminator.AddLayer([
      TNNetInput.Create(32,32,3),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), // downsample to 16x16
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), // downsample to 8x8
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), // downsample to 4x4
      TNNetSELU.Create(),
//      TNNetDropout.Create(0.4),
      TNNetFullConnectLinear.Create(2),
      TNNetSoftMax.Create()
    ]);*)
    Discriminator.AddLayer([
      TNNetInput.Create(32,32,3),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,0,1,0), // downsample to 15x15
      TNNetMaxPool.Create(2),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,0,1,0), // downsample to 7x7
      TNNetMaxPool.Create(2),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,0,1,0), // downsample to 3x3
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(128),
      TNNetFullConnectLinear.Create(2),
      TNNetSoftMax.Create()
    ]);

    Discriminator.Layers[Discriminator.GetFirstImageNeuronalLayerIdx()].InitBasicPatterns();
  end
  else
  begin
    WriteLn('Loading discriminator.');
    Discriminator.LoadFromFile(FBaseName+'discriminator.nn');
    TNNetInput(Discriminator.Layers[0]).EnableErrorCollection;
    Discriminator.DebugStructure();
    Discriminator.DebugWeights();
  end;
  Discriminator.DebugStructure();

  DiscriminatorClone := Discriminator.Clone();
  TNNetInput(DiscriminatorClone.Layers[0]).EnableErrorCollection;

  FFit.EnableClassComparison();
  FFit.OnAfterEpoch := @Self.DiscriminatorOnAfterEpoch;
  FFit.OnAfterStep := @Self.DiscriminatorOnAfterStep;
  FFit.OnStart := @Self.DiscriminatorOnStart;
  FFit.DataAugmentationFn := @Self.DiscriminatorAugmentation;
  FFit.LearningRateDecay := 0.00001;
  FFit.AvgWeightEpochCount := 1;
  FFit.InitialLearningRate := 0.001;
  FFit.FileNameBase := FBaseName+'GenerativeNeuralAPI';
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FFit.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
    Generative.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  FFit.FitLoading(Discriminator, 64*10, 500, 500, 64, 3500, @GetDiscriminatorTrainingPair, nil, nil);

  if Assigned(FGeneratives) then FreeAndNil(FGeneratives);
  vDisplay.Free;
  Generative.Free;
  Discriminator.Free;
  DiscriminatorClone.Free;
end;

function TFormVisualLearning.GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
var
  RandomValue: integer;
  LocalPair: TNNetVolumePair;
begin
  if (FRealPairs.Count = 0) then
  begin
    WriteLn('Error: discriminator real pairs have no element');
    Result := nil;
    exit;
  end;

  if FGeneratedPairs.Count = 0 then
  begin
    WriteLn('Error: discriminator generated/fake pairs have no element');
    Result := nil;
    exit;
  end;

  RandomValue := Random(1000);
  if RandomValue < 500 then
  begin
    Result := FRealPairs[Random(FRealPairs.Count)];
    Result.O.SetClassForSoftMax(1);
    // Debug Only: if (Random(100)=0) then DisplayInputImage(Result.I, FColorEncoding);
  end
  else
  begin
    LocalPair := FGeneratedPairs[ThreadId];
    LocalPair.I.Resize(FRandomSizeX, FRandomSizeY, FRandomDepth);
    LocalPair.I.Randomize();
    LocalPair.I.NormalizeMax(2);
    FGeneratives[ThreadId].Compute(LocalPair.I);
    FGeneratives[ThreadId].GetOutput(LocalPair.I);
    Result := LocalPair;
    Result.O.SetClassForSoftMax(0);
  end;
end;

procedure TFormVisualLearning.DiscriminatorOnAfterEpoch(Sender: TObject);
var
  LoopCnt: integer;
  RandomIdx: integer;
  LocalPair: TNNetVolumePair;
  ExpectedDiscriminatorOutput, Transitory, DiscriminatorFound: TNNetVolume;
  Error: TNeuralFloat;
begin
  if FFit.TrainingAccuracy <= 0.745 //Default is: 0.745
  then exit;
  WriteLn('Training Generative Start.');
  ExpectedDiscriminatorOutput := TNNetVolume.Create(2, 1, 1);
  ExpectedDiscriminatorOutput.SetClassForSoftMax(1);
  DiscriminatorFound := TNNetVolume.Create(ExpectedDiscriminatorOutput);
  Transitory := TNNetVolume.Create(DiscriminatorClone.Layers[0].OutputError);
  DiscriminatorClone.CopyWeights(Discriminator);
  Generative.SetBatchUpdate(true);
  Generative.SetLearningRate(FFit.CurrentLearningRate*1, 0); //Also good values are: *0.1, 0.0
  DiscriminatorClone.SetBatchUpdate(true);
  DiscriminatorClone.SetL2Decay(0.0);
  Generative.SetL2Decay(0.00001);
  begin
    Error := 0;
    DiscriminatorClone.RefreshDropoutMask();
    for LoopCnt := 1 to 100 do
    begin
      DiscriminatorClone.ClearDeltas();
      DiscriminatorClone.ClearInertia();
      RandomIdx := Random(FGeneratedPairs.Count);
      LocalPair := FGeneratedPairs[RandomIdx];
      LocalPair.I.Resize(FRandomSizeX, FRandomSizeY, FRandomDepth);
      LocalPair.I.Randomize();
      LocalPair.I.NormalizeMax(2);
      Generative.Compute(LocalPair.I);
      Generative.GetOutput(LocalPair.I);
      DiscriminatorClone.Compute(LocalPair.I);
      DiscriminatorClone.GetOutput(DiscriminatorFound);
      DiscriminatorClone.Backpropagate(ExpectedDiscriminatorOutput);
      Error += ExpectedDiscriminatorOutput.SumDiff(DiscriminatorFound);
      Transitory.Copy(LocalPair.I);
      Transitory.Sub(DiscriminatorClone.Layers[0].OutputError);
      Generative.Backpropagate(Transitory);
      Generative.NormalizeMaxAbsoluteDelta(0.0001);
      Generative.UpdateWeights();
      if LoopCnt mod 10 = 0 then ProcessMessages();
    end;
    DisplayInputImage(LocalPair.I, FColorEncoding);
    DiscriminatorClone.Layers[0].OutputError.PrintDebug();WriteLn();
    WriteLn('Generative error:', Error:6:4);
  end;
  Generative.DebugErrors();
  Generative.DebugWeights();
  FGeneratives.CopyWeights(Generative);
  DiscriminatorClone.DebugWeights();
  ExpectedDiscriminatorOutput.Free;
  Transitory.Free;
  DiscriminatorFound.Free;
  if FFit.CurrentEpoch mod 100 = 0 then
  begin
    WriteLn('Saving ', FBaseName);
    Generative.SaveToFile(FBaseName+'generative.nn');
    Discriminator.SaveToFile(FBaseName+'discriminator.nn');
    SaveScreenshot(FBaseName+'cai-neural-gan.bmp');
  end;
  WriteLn('Training Generative Finish with:', Error:6:4);
  //DisplayInputImage(FRealPairs[Random(FRealPairs.Count)].I, FColorEncoding);
end;

procedure TFormVisualLearning.DiscriminatorOnAfterStep(Sender: TObject);
begin
  LabClassRate.Caption := PadLeft(IntToStr(Round(FFit.TrainingAccuracy*100))+'%',4);
  ProcessMessages();
end;

procedure TFormVisualLearning.DiscriminatorAugmentation(pInput: TNNetVolume;
  ThreadId: integer);
begin
  if Random(1000)>500 then pInput.FlipX();
  //if Random(1000)>750 then pInput.MakeGray(FColorEncoding);
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

procedure TFormVisualLearning.BuildTrainingPairs();
var
  FakePairCnt: integer;
  ImgTrainingVolume: TNNetVolume;
  DiscriminatorOutput, GenerativeOutput: TNNetVolume;
begin
  DiscriminatorOutput := TNNetVolume.Create(2, 1, 1);
  GenerativeOutput := TNNetVolume.Create(32, 32, 3);
  if FRealPairs.Count = 0 then
  begin
    for ImgTrainingVolume in ImgTrainingVolumes do
    begin
      if ImgTrainingVolume.Tag = 5 then //5
      begin
        DiscriminatorOutput.SetClassForSoftMax(1);
        FRealPairs.Add
        (
          TNNetVolumePair.CreateCopying
          (
            ImgTrainingVolume,
            DiscriminatorOutput
          )
        );
        //Debug only: if (Random(100)=0) then DisplayInputImage(ImgTrainingVolume, FColorEncoding);
      end
    end;
  end;

  DiscriminatorOutput.SetClassForSoftMax(0);

  if FGeneratedPairs.Count < FFit.MaxThreadNum then
  begin
    for FakePairCnt := 1 to FFit.MaxThreadNum do
    begin
      FGeneratedPairs.Add
      (
        TNNetVolumePair.CreateCopying
        (
          GenerativeOutput,
          DiscriminatorOutput
        )
      );
    end;
  end;
  ImgTrainingVolumes.Clear;
  ImgValidationVolumes.Clear;
  ImgTestVolumes.Clear;
  GenerativeOutput.Free;
  DiscriminatorOutput.Free;
end;

procedure TFormVisualLearning.ProcessMessages();
begin
  Application.ProcessMessages();
end;

end.
