{
Copyright (C) 18 Joao Paulo Schwarz Schuler

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

unit uvisualgan_mX476;

//{$mode objfpc}{$H+}

interface

//uses
  {$ifdef unix}
  cmem; // the c memory manager is on some systems much faster for multi-threading
  {$endif}
  //Classes, SysUtils, FileUtil, Forms, Controls, Graphics, Dialogs, StdCtrls,
  //ExtCtrls, neuralnetwork, neuralvolumev, neuraldatasets,
  //neuralvolume;
  
var csLearningRates: array[0..2] of TNeuralFloat; // = (1, 0.1, 0.01);  

type
  { TFormVisualLearning }
  TFormVisualLearning = {class(}TForm;
  var
    ButLearn: TButton;
    ButLoadFile: TButton;
    ChkForceInputRange: TCheckBox;
    ChkStrongInput: TCheckBox;
    ComboLayer: TComboBox;
    GrBoxNeurons: TGroupBox;
    ImgSample, imgdirect: TImage;
    LabLayer, lblfilename, lblfile: TLabel;
    //ButLearn: TButton;
    ChkRunOnGPU: TCheckBox;
    ChkBigNetwork: TCheckBox;
    ComboLearningRate: TComboBox;
    ComboComplexity: TComboBox;
    //GrBoxNeurons: TGroupBox;
    //ImgSample: TImage;
    LabClassRate: TLabel;
    LabComplexity: TLabel;
    LabLearningRate: TLabel;
    RadLAB: TRadioButton;
    RadRGB: TRadioButton;
    OpenDialogNN: TOpenDialog;
    procedure ButLearnClick(Sender: TObject);
    procedure ButLoadFileClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
  //private
    { private declarations }
   var 
    FRunning: boolean;
    aImage: array of TImage;
    aLabelX, aLabelY: array of TLabel;
    FileName: string;
    FDisplay: TNNetVolume;
    ImgTrainingVolumes, ImgTestVolumes, ImgValidationVolumes: TNNetVolumeList;
    FRealPairs, FGeneratedPairs: TNNetVolumePairList;
    FImageCnt: integer;
    iEpochCount, iEpochCountAfterLoading: integer;
    FGenerative: THistoricalNets;
    FGeneratives: TNNetDataParallelism;
    FDiscriminator, FDiscriminatorClone: TNNet;
    FBaseName: string;
    FColorEncoding: byte;
    FRandomSizeX, FRandomSizeY, FRandomDepth: integer;
    FLearningRateProportion: TNeuralFloat;

    //iEpochCount, iEpochCountAfterLoading: integer;

    FNN: TNNet;
    FLastLayerIdx: integer;
    FFilterSize: integer;
    FImagesPerRow: integer;
    FOutputSize: integer;
    FCritSec: TRTLCriticalSection;
    FFit: TNeuralDataLoadingFit;
    
    procedure Learn(Sender: TObject);
    procedure TFormVisualLearningLearn2( Sender: TObject);
    procedure TFormVisualLearningEnableComponents(flag: boolean);
    procedure SaveScreenshot(pfilename: string);
    procedure SaveNeuronsImage(pfilename: string);
    
    function GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
    procedure GetDiscriminatorTrainingProc(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure DiscriminatorOnAfterEpoch(Sender: TObject);
    procedure DiscriminatorOnAfterStep(Sender: TObject);
    procedure DiscriminatorAugmentation(pInput: TNNetVolume; ThreadId: integer);
    //procedure Learn(Sender: TObject);
    //procedure SaveScreenshot(filename: string);
    procedure BuildTrainingPairs();
    procedure TFormVisualLearningDisplayInputImage(ImgInput: TNNetVolume;
                                   color_encoding: integer);
    procedure DiscriminatorOnStart(Sender: TObject);
    procedure TFormVisualLearningSendStop;
  //public
    procedure TFormVisualLearningProcessMessages();
 // end;

var
  FormVisualLearning: TFormVisualLearning;

implementation
//{$R *.lfm}

//uses LCLIntf, LCLType, neuraldatasetsv;

{ TFormVisualLearning }

procedure TFormVisualLearningLearn2( Sender: TObject);
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
    //FBaseName += 'RGB-';
    FBaseName:= fbasename +'RGB-';
  end
  else
  begin
    FColorEncoding := csEncodeLAB;
    FBaseName := fbasename+ 'LAB-';
  end;
  Self.Height := GrBoxNeurons.Top + GrBoxNeurons.Height + 10;
  Self.Width  := GrBoxNeurons.Left + GrBoxNeurons.Width + 10;
  TFormVisualLearningProcessMessages();
  if not(Assigned(ImgValidationVolumes)) then
  begin
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, FColorEncoding);
  end;

  iEpochCount := 0;
  iEpochCountAfterLoading := 0;

  writeln('Creating Neural Networks...');
  FGenerative := THistoricalNets.Create();
  FDiscriminator := TNNet.Create();

  FLearningRateProportion := csLearningRates[ComboLearningRate.ItemIndex];

  if Not(FileExists(FBaseName+'generative.nn')) then begin
    WriteLn('Creating generative.');
    FGenerative.AddLayer49([
      TNNetInput.Create4(FRandomSizeX, FRandomSizeY, FRandomDepth),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,1,1,0), //4x4
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetDeMaxPool.Create44(2,0,0),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,1,1,0), //8x8
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetDeMaxPool.Create44(4,0,0),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,5,2,1,0), //32x32
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create(32 * NeuronMultiplier,3,1,1,0),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionLinear.Create(3,3,1,1,0),
      TNNetReLUL.Create(-40, +40) // Protection against overflow
    ]);
    (*
    FGenerative.AddLayer([
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
  FGenerative.Layers[FGenerative.GetFirstImageNeuronalLayerIdx(0)].InitBasicPatterns();
  end
  else
  begin
    WriteLn('Loading generative.');
    FGenerative.LoadFromFile(FBaseName+'generative.nn');
  end;
  FGenerative.DebugStructure();
  FGenerative.SetLearningRate(0.001,0.9);
  FGenerative.SetL2Decay(0.0);

  if Not(FileExists(FBaseName+'discriminator.nn')) then
  begin
    WriteLn('Creating discriminator.');
    (*
    FDiscriminator.AddLayer([
      TNNetInput.Create(32,32,3),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), 
      // downsample to 16x16
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), 
      // downsample to 8x8
      TNNetSELU.Create(),
      TNNetConvolutionLinear.Create(64 * NeuronMultiplier,3,1,2,0), 
      // downsample to 4x4
      TNNetSELU.Create(),
//      TNNetDropout.Create(0.4),
      TNNetFullConnectLinear.Create(2),
      TNNetSoftMax.Create()
    ]);*)
    FDiscriminator.AddLayer49([
      TNNetInput.Create4(32,32,3),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,0,1,0), 
      // downsample to 15x15
      TNNetMaxPool.Create44(2,0,0),
      TNNetConvolutionReLU.Create(64 * NeuronMultiplier,3,0,1,0), 
      // downsample to 7x7
      TNNetMaxPool.Create44(2,0,0),
      TNNetConvolutionReLU.Create(128 * NeuronMultiplier,3,0,1,0), 
      // downsample to 3x3
      TNNetDropout.Create12(0.5,1),
      TNNetMaxPool.Create44(2,0,0),
      TNNetFullConnectLinear.Create28(128,0),
      TNNetFullConnectLinear.Create28(2,0),
      TNNetSoftMax.Create()
    ]);

    FDiscriminator.Layers[FDiscriminator.GetFirstImageNeuronalLayerIdx(0)].InitBasicPatterns();
  end
  else
  begin
    WriteLn('Loading discriminator.');
    FDiscriminator.LoadFromFile(FBaseName+'discriminator.nn');
    TNNetInput(FDiscriminator.Layers[0]).EnableErrorCollection;
    FDiscriminator.DebugStructure();
    FDiscriminator.DebugWeights();
  end;
  FDiscriminator.DebugStructure();

  FDiscriminatorClone := FDiscriminator.Clone();
  TNNetInput(FDiscriminatorClone.Layers[0]).EnableErrorCollection;

  FFit.EnableClassComparison();
  FFit.OnAfterEpoch := @{Self.}DiscriminatorOnAfterEpoch;
  FFit.OnAfterStep := @{Self.}DiscriminatorOnAfterStep;
  FFit.OnStart := @{Self.}DiscriminatorOnStart;
  FFit.DataAugmentationFn := @{Self.}DiscriminatorAugmentation;
  FFit.LearningRateDecay := 0.00001;
  FFit.AvgWeightEpochCount := 1;
  FFit.InitialLearningRate := 0.001;
  FFit.FileNameBase := FBaseName+'GenerativeNeuralAPI';
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FFit.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
    FGenerative.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  //Debug only: FFit.MaxThreadNum := 1;
  //FFit.FitLoading(FDiscriminator, 64*10, 500, 500, 64, 35000, @GetDiscriminatorTrainingPair, nil, nil); // This line does the same as below
  FFit.FitLoading1(FDiscriminator, 64*10, 500, 500, 64, 35000,
   @GetDiscriminatorTrainingProc, nil, nil); // This line does the same as above

  if Assigned(FGeneratives) then begin
     //FreeAndNil(FGeneratives);
     FGeneratives.Free;
     FGeneratives:= Nil;
   end;  
  FGenerative.Free;
  FDiscriminator.Free;
  FDiscriminatorClone.Free;
end;


procedure TFormVisualLearningSendStop;
begin
  WriteLn('Sending STOP request');
  FFit.ShouldQuit := true;
end;


procedure DiscriminatorOnStart(Sender: TObject);             
begin
  FGeneratives := TNNetDataParallelism.Create74(FGenerative, FFit.MaxThreadNum, true);
  {$ifdef OpenCL}
  if FHasOpenCL then
  begin
    FGeneratives.EnableOpenCL(FEasyOpenCL.PlatformIds[0], FEasyOpenCL.Devices[0]);
  end;
  {$endif}
  BuildTrainingPairs();
end;

procedure TFormVisualLearningDisplayInputImage(ImgInput: TNNetVolume; 
                                   color_encoding: integer);
var
  pMin0, pMax0: TNeuralFloat;
  pMin1, pMax1: TNeuralFloat;
  pMin2, pMax2: TNeuralFloat;
  //FDisplay : TNNetVolume;
begin
  //FDisplay := TNNetVolume.Create();
  FDisplay.Resize(ImgInput, 0,0);
  FDisplay.Copy76(ImgInput);

  if color_encoding = csEncodeLAB then begin
    FDisplay.GetMinMaxAtDepth(0, pMin0, pMax0);
    FDisplay.GetMinMaxAtDepth(1, pMin1, pMax1);
    FDisplay.GetMinMaxAtDepth(2, pMin2, pMax2);
    pMax0 := MaxF(Abs(pMin0), Abs(pMax0));
    pMax1 := MaxF(Abs(pMin1), Abs(pMax1));
    pMax2 := MaxF(Abs(pMin2), Abs(pMax2));

    if pMax0 > 2 then
    begin
      FDisplay.MulAtDepth27(0, 2/pMax0);
    end;

    if pMax1 > 2 then
    begin
      FDisplay.MulAtDepth27(1, 2/pMax1);
    end;

    if pMax2 > 2 then
    begin
      FDisplay.MulAtDepth27(2, 2/pMax2);
    end;
  end
  else if FDisplay.GetMaxAbs() > 2 then
  begin
    FDisplay.NormalizeMax(2);
  end;

  //Debug only: FDisplay.PrintDebugChannel();
  FDisplay.NeuronalInputToRgbImg(color_encoding);
  LoadVolumeIntoTImage(FDisplay, aImage[FImageCnt],color_encoding);
  aImage[FImageCnt].Width := 64;
  aImage[FImageCnt].Height := 64;
  TFormVisualLearningProcessMessages();
  FImageCnt := (FImageCnt + 1) mod Length(aImage);
  //FDisplay.Free;
end;  


function GetDiscriminatorTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
var
  RandomValue, RandomPos: integer;
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
    RandomPos := Random(FRealPairs.Count);
    Result := FRealPairs[RandomPos];
    Result.O.SetClassForSoftMax(1);
    if Result.I.Size <> 32*32*3 then
    begin
      WriteLn('ERROR: Real Pair index '+itoa(RandomPos)+
            'has wrong size:'+itoa( Result.I.Size));
    end;
    //Debug Only: if (Random(100)=0) then DisplayInputImage(Result.I, FColorEncoding);
  end
  else
  begin
    LocalPair := FGeneratedPairs[ThreadId];
    LocalPair.I.Resize(FRandomSizeX, FRandomSizeY, FRandomDepth);
    LocalPair.I.Randomize(0,0,0);
    LocalPair.I.NormalizeMax(2);
    //FGeneratives: TNNetDataParallelism;
    //RegisterProperty('Items', 'TNNet Integer', iptrw);
    //LocalPair: TNNetVolumePair;
    //RegisterMethod('Procedure Compute65(pInput:TNNetVolume;FromLayerIdx:integer);');
    FGeneratives[ThreadId].Compute65(LocalPair.I,0);
    FGeneratives[ThreadId].GetOutput(LocalPair.I);
    Result := LocalPair;
    Result.O.SetClassForSoftMax(0);
    if Result.I.Size <> 32*32*3 then
    begin
      WriteLn('ERROR: Generated Pair has wrong size:'+itoa( Result.I.Size));
    end;
  end;
end;

procedure GetDiscriminatorTrainingProc(Idx: integer;
  ThreadId: integer; pInput, pOutput: TNNetVolume);
var
  LocalPair: TNNetVolumePair;
begin
  LocalPair := GetDiscriminatorTrainingPair(Idx, ThreadId);
  pInput.Copy76(LocalPair.I);
  pOutput.Copy76(LocalPair.O);
end;

procedure DiscriminatorOnAfterEpoch(Sender: TObject);
var
  LoopCnt, MaxLoop: integer;
  ExpectedDiscriminatorOutput, Transitory, DiscriminatorFound, GenerativeInput: TNNetVolume;
  Error: TNeuralFloat;
begin
  if (FFit.TrainingAccuracy <= 0.745) or FFit.ShouldQuit
  then exit;
  WriteLn('Training Generative Start.');
  ExpectedDiscriminatorOutput := TNNetVolume.Create0(2, 1, 1,0);
  ExpectedDiscriminatorOutput.SetClassForSoftMax(1);
  DiscriminatorFound := TNNetVolume.Create3(ExpectedDiscriminatorOutput);
  //Create3( Original : TVolume);');
  //Create6( pSize : integer; c : TNeuralFloat);')
  Transitory := TNNetVolume.Create3(FDiscriminatorClone.Layers[0].OutputError);
  GenerativeInput := TNNetVolume.Create0(FRandomSizeX, FRandomSizeY, FRandomDepth,0);
  FDiscriminatorClone.CopyWeights(FDiscriminator);
  FGenerative.SetBatchUpdate(true);
  FGenerative.SetLearningRate(FFit.CurrentLearningRate*FLearningRateProportion, 0);
  FGenerative.SetL2Decay(0.00001);
  FDiscriminatorClone.SetBatchUpdate(true);
  FDiscriminatorClone.SetL2Decay(0.0);
  MaxLoop := Round(100 * (1/FLearningRateProportion));
  begin
    Error := 0;
    FDiscriminatorClone.RefreshDropoutMask();
    for LoopCnt := 1 to MaxLoop do
    begin
      if FFit.ShouldQuit then break;
      FDiscriminatorClone.ClearDeltas();
      FDiscriminatorClone.ClearInertia();
      GenerativeInput.Randomize(0,0,0);
      GenerativeInput.NormalizeMax(2);
      if FGenerative.Layers[0].Output.Size<>GenerativeInput.Size then
      begin
        Write('.');
        FGenerative.Layers[0].Output.ReSize10(GenerativeInput);
        FGenerative.Layers[0].OutputError.ReSize10(GenerativeInput);
      end;
      FGenerative.Compute65(GenerativeInput,0);
      FGenerative.GetOutput(Transitory);
      FDiscriminatorClone.Compute65(Transitory,0);
      FDiscriminatorClone.GetOutput(DiscriminatorFound);
      FDiscriminatorClone.Backpropagate69(ExpectedDiscriminatorOutput);
      //Error += ExpectedDiscriminatorOutput.SumDiff(DiscriminatorFound);
      Error:= error +ExpectedDiscriminatorOutput.SumDiff(DiscriminatorFound);
      Transitory.Sub64(FDiscriminatorClone.Layers[0].OutputError);
      FGenerative.Backpropagate69(Transitory);
      FGenerative.NormalizeMaxAbsoluteDelta(0.001);
      FGenerative.UpdateWeights();
      if LoopCnt mod 10 = 0 then TFormVisualLearningProcessMessages();
      if LoopCnt mod 100 = 0 then TFormVisualLearningDisplayInputImage(Transitory, FColorEncoding);
    end;
    FDiscriminatorClone.Layers[0].OutputError.PrintDebug();
    WriteLn('Clone.Layers end');
    WriteLn('Generative error:'+flots(Error))//:6:4);         
  end;
  //Debug:
  //FGenerative.DebugErrors();
  //FGenerative.DebugWeights();
  //FDiscriminatorClone.DebugWeights();
  FGeneratives.CopyWeights(FGenerative);
  GenerativeInput.Free;
  ExpectedDiscriminatorOutput.Free;
  Transitory.Free;
  DiscriminatorFound.Free;
  if FFit.CurrentEpoch mod 100 = 0 then
  begin
    WriteLn('Saving '+ FBaseName);
    FGenerative.SaveToFile(FBaseName+'generative.nn');
    FDiscriminator.SaveToFile(FBaseName+'discriminator.nn');
    SaveScreenshot(FBaseName+'cai-neural-gan.bmp');
  end;
  WriteLn('Training Generative Finish with:'+flots( Error)); //:6:4);
  //DisplayInputImage(FRealPairs[Random(FRealPairs.Count)].I, FColorEncoding);
end;

procedure DiscriminatorOnAfterStep(Sender: TObject);
begin
  LabClassRate.Caption := 
       StrPadLeft(IntToStr(Round(FFit.TrainingAccuracy*100))+'%',4,'-');
  TFormVisualLearningProcessMessages();
end;

procedure DiscriminatorAugmentation(pInput: TNNetVolume;
  ThreadId: integer);
begin
  if Random(1000)>500 then pInput.FlipX();
  //if Random(1000)>750 then pInput.MakeGray(FColorEncoding);
end;

procedure BuildTrainingPairs();
var
  FakePairCnt: integer;
  ImgTrainingVolume: TNNetVolume;
  DiscriminatorOutput, GenerativeOutput: TNNetVolume;
begin
  DiscriminatorOutput := TNNetVolume.Create0(2, 1, 1,0);
  GenerativeOutput := TNNetVolume.Create0(32, 32, 3,0);
  //if FRealPairs.Count = 0 then
  //begin
  (*
    for ImgTrainingVolume in ImgTrainingVolumes do   fix
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
  end;*)

  DiscriminatorOutput.SetClassForSoftMax(0);

  if FGeneratedPairs.Count < FFit.MaxThreadNum then
  begin
    for FakePairCnt := 1 to FFit.MaxThreadNum do
    begin
      FGeneratedPairs.Add
      (
        TNNetVolumePair.CreateCopying83
        (
          GenerativeOutput,
          DiscriminatorOutput
        )
      );
    end;
  end;
  ImgTrainingVolumes.Cleartag;
  ImgValidationVolumes.Cleartag;
  ImgTestVolumes.Cleartag;
  GenerativeOutput.Free;
  DiscriminatorOutput.Free;
end;



procedure ButLearnClick(Sender: TObject);
begin
  if (FRunning) then
  begin
    FRunning := false;
  end
  else
  begin
    FRunning := true;
    ButLearn.Caption := 'Stop';
    TFormVisualLearningEnableComponents(false);
    Learn(Sender);
    TFormVisualLearningEnableComponents(true);
    FRunning := false;
  end;
  ButLearn.Caption := 'Restart';
end;

procedure ButLoadFileClick(Sender: TObject);
begin
  if (OpenDialogNN.Execute()) then begin
    if FileExists(OpenDialogNN.FileName) then begin
      ButLearn.Enabled := false;
      FileName := OpenDialogNN.FileName;
      lblfile.caption:= extractfilename(filename);
      if not(FileExists(FileName)) then begin
        if FileName = 'SimpleImageClassifier.nn' then begin
          WriteLn('Please run the Simple Image Classifier example before running this example.');
          WriteLn('https://github.com/joaopauloschuler/neural-api/tree/master/examples/SimpleImageClassifier');
        end
        else begin
          WriteLn('File not found:'+FileName);
        end;
        exit;
      end;

      FNN.LoadFromFile(FileName);
      FNN.DebugStructure();
      FNN.DebugWeights();
      FNN.SetBatchUpdate( true );
      FNN.SetLearningRate(0.01,0.0);
      TNNetInput(FNN.Layers[0]).EnableErrorCollection();
      WriteLn('Neural network has: ');
      WriteLn(' Layers: '+ itoa(FNN.CountLayers()  ));
      WriteLn(' Neurons:'+ itoa(FNN.CountNeurons() ));
      WriteLn(' Weights:'+ itoa(FNN.CountWeights() ));
      ButLearn.Enabled := true;
      LoadNNLayersIntoCombo(FNN, ComboLayer);
    end;
  end;
end;

procedure FormClose(Sender: TObject;
  var CloseAction: TCloseAction);
begin
  FRunning := false;
  closeaction:= cafree;
  writeln('form closed...');
end;

procedure FormCreate(Sender: TObject);
var ab: TBitmap;
begin
  WriteLn('Creating Neural Network...');
  FNN := TNNet.Create();
  FRunning := false;
  SetLength(aImage, 0);
  SetLength(aLabelX, 0);
  SetLength(aLabelY, 0);
  ab:= TBitmap.create;
  ab.loadfromresourcename(0,'MOON_FULL');
  imgsample.picture.bitmap:= ab;
  ab.loadfromresourcename(0,'MOON_FULL');
  imgdirect.picture.bitmap:= ab;
  ab.Free;
  
  FRunning := false;
  FFit := TNeuralDataLoadingFit.Create();
  NeuralInitCriticalSection(FCritSec);
  ImgTrainingVolumes := nil;
  ImgTestVolumes := nil;
  ImgValidationVolumes := nil;
  FGeneratives := nil;
  FRealPairs := TNNetVolumePairList.Create(true);
  FGeneratedPairs := TNNetVolumePairList.Create(true);
  FDisplay := TNNetVolume.Create();             
  FImageCnt := 0;
 (* CreateAscentImages
  (
    GrBoxNeurons,
    aImage, aLabelX, aLabelY,
    {ImageCount=}128,
    {InputSize=}32, {FFilterSize=}64, {ImagesPerRow=}16
  );     *)
  {$ifdef OpenCL}
  FEasyOpenCL := TEasyOpenCL.Create();
  {$else}
    //ChkRunOnGPU.Visible := false;
  {$endif}
end;

procedure FormDestroy(Sender: TObject);
begin
  FRunning := false;
  FreeNeuronImages(aImage,  aLabelX,  aLabelY);
  FNN.Free;
  writeln('destroy neural images');
  TFormVisualLearningSendStop;
  while FFit.Running do Application.ProcessMessages;
  while FRunning do Application.ProcessMessages;
  if Assigned(ImgValidationVolumes) then ImgValidationVolumes.Free;
  if Assigned(ImgTestVolumes) then ImgTestVolumes.Free;
  if Assigned(ImgTrainingVolumes) then ImgTrainingVolumes.Free;
  FreeNeuronImages(aImage, aLabelX, aLabelY);
  NeuralDoneCriticalSection(FCritSec);
  FRealPairs.Free;
  FGeneratedPairs.Free;
  FDisplay.Free;
  {$ifdef OpenCL}FEasyOpenCL.Free;{$endif}
  FFit.Free;
  writeln('destroy neural neurons all done');      
end;

procedure Learn(Sender: TObject);
var
  vInput, pOutput, vDisplay: TNNetVolume;
  OutputCount, K: integer;
  InputSize: integer;
  StatingSum, CurrentSum, InputForce: TNeuralFloat;
  LayerHasDepth: boolean;
  StopRatio: TNeuralFloat;
begin
  iEpochCount := 0;
  iEpochCountAfterLoading := 0;
  FLastLayerIdx := ComboLayer.ItemIndex;
  LayerHasDepth := (FNN.Layers[FLastLayerIdx].OutputError.Depth > 1);
  if LayerHasDepth
    then FOutputSize := FNN.Layers[FLastLayerIdx].OutputError.Depth
    else FOutputSize := FNN.Layers[FLastLayerIdx].OutputError.Size;

  if FOutputSize > 32 then
  begin
    FFilterSize := 64;
    FImagesPerRow := 16;
  end
  else
  begin
    FFilterSize := 128;
    FImagesPerRow := 8;
  end;

  //RegisterMethod('Constructor Create3( Original : TVolume);');
  vInput  := TNNetVolume.Create3(FNN.Layers[0].Output);
  vDisplay := TNNetVolume.Create3(vInput);
  pOutput := TNNetVolume.Create3(FNN.GetLastLayer().Output);
  InputSize := FNN.Layers[0].Output.SizeX;

  FreeNeuronImages(aImage,  aLabelX,  aLabelY);
  CreateAscentImages
  (
    GrBoxNeurons,
    aImage, aLabelX, aLabelY,
    FOutputSize,
    InputSize, FFilterSize, FImagesPerRow
  );

  FormVisualLearning.Width := GrBoxNeurons.Left + GrBoxNeurons.Width + 10;
  FormVisualLearning.Height := GrBoxNeurons.Top + GrBoxNeurons.Height + 50;
  Application.ProcessMessages;

  if ChkForceInputRange.Checked
  then StopRatio := 100
  else StopRatio := vInput.Size;

  for OutputCount := 0 to FOutputSize - 1 do
  begin
    //vInput.Randomize(10000, 5000, 5000000);
    if LayerHasDepth then
    begin
      vInput.Fill(0.0);
      vInput.AddSaltAndPepper(4, 2, -2, true);
    end
    else
    begin
    // RegisterMethod('Procedure Sub19( Value : TNeuralFloat);');
      vInput.RandomizeGaussian(2); vInput.Sub19(1);
    end;
    StatingSum := vInput.GetSumAbs();
    if ChkStrongInput.Checked
      then InputForce := 2
      else InputForce := 0.002;
    vInput.NormalizeMax(InputForce);
    Write('Random Input: '); vInput.PrintDebug(); WriteLn('');
    for K := 1 to 10000 do
    begin
      FNN.Compute65(vInput,0);
      FNN.GetOutput(pOutput);
      FNN.BackpropagateFromLayerAndNeuron(FLastLayerIdx, OutputCount, 20);
      vInput.MulAdd69(-1, FNN.Layers[0].OutputError);
      if ChkForceInputRange.Checked then vInput.ForceMaxRange(2);
      FNN.ClearDeltas();
      FNN.ClearInertia();
      if K mod 100 = 0 then
      begin
        vDisplay.Copy76(vInput);
        vDisplay.NeuronalWeightToImg54(0);
        LoadVolumeIntoTImage(vDisplay, aImage[OutputCount], 0);
        aImage[OutputCount].Width   := FFilterSize;
        aImage[OutputCount].Height  := FFilterSize;
        Application.ProcessMessages();
        CurrentSum := vInput.GetSumAbs();
        WriteLn(flots(StatingSum)+' - '+flots(CurrentSum)+' - '+
                                           flots((CurrentSum/StatingSum)));
        if (CurrentSum > StopRatio*StatingSum) or Not(LayerHasDepth)
          then break
          else vInput.AddSaltAndPepper(4, InputForce/10, -InputForce/10, true);
      end;
      Application.ProcessMessages();
      if Not(FRunning) then break;
    end;
    //FNN.DebugWeights();
    writeln('Neural learn part has been finished!')
  end;

  vDisplay.Free;
  vInput.Free;
  pOutput.Free;
end;

procedure TFormVisualLearningEnableComponents(flag: boolean);
var
  i : Integer;
begin
  for i := 0 to FormVisualLearning.ComponentCount-1 do
  begin
    if (FormVisualLearning.Components[i] is TEdit) then
      TEdit(FormVisualLearning.Components[i]).Enabled := flag;

    if (FormVisualLearning.Components[i] is TComboBox) then
       TComboBox(FormVisualLearning.Components[i]).Enabled := flag;

    if (FormVisualLearning.Components[i] is TCheckBox) then
       TCheckBox(FormVisualLearning.Components[i]).Enabled := flag;

    if (FormVisualLearning.Components[i] is TRadioButton) then
       TRadioButton(FormVisualLearning.Components[i]).Enabled := flag;
  end;

  Application.ProcessMessages;
end;

procedure SaveScreenshot(pfilename: string);
begin
  WriteLn(' Saving '+pfilename+'.');
  SaveHandleToBitmap(pfilename, Self.Handle);
end;

procedure SaveNeuronsImage(pfilename: string);
begin
  WriteLn(' Saving '+pfilename+'.');
  SaveHandleToBitmap(pfilename, GrBoxNeurons.Handle);
end;

procedure TFormVisualLearningProcessMessages();
begin
  Application.ProcessMessages();
end;


procedure loadvisualneuralForm;
begin
FormVisualLearning:= TFormVisualLearning.create(self);
ImgSample:= TImage.create(self);
imgdirect:= TImage.create(self);
with  FormVisualLearning do begin
  Left := 88
  Height := 704
  Top := 61
  Width := 1106
  Caption := 'CAI Visual Gradient Ascent Example by maXbox4'
  ClientHeight := 702
  ClientWidth := 1104
  //DesignTimePPI := 120
  OnClose := @FormClose;
  OnCreate := @FormCreate;
  OnDestroy := @FormDestroy;
  //show;
  FormCreate(Self);
  ShoW;
  Position := poScreenCenter
  //LCLVersion := '2.0.2.0'
  ButLearn:= TButton.create(self)
  with butlearn do begin
   parent:= formvisuallearning;
    Left := 624
    Height := 30
    Top := 15
    Width := 112
    Caption := 'Start'
    Enabled := False
    OnClick := @ButLearnClick;
    ParentFont := False
    TabOrder := 0
  end;
  //ImgSample:= TImage.create(self)
  with imgsample do begin
   parent:= formvisuallearning;
    Left := 445
    Height := 42
    Top := 120
    Width := 42
    Stretch := True
  end;
 //imgdirect:= TImage.create(self);
  with imgdirect do begin
   parent:= formvisuallearning;
    Left := 545
    Height := 142
    Top := 120
    Width := 142
    Stretch := True
  end;
  
  GrBoxNeurons:= TGroupBox.create(self)
  with grboxneurons do begin
   parent:= formvisuallearning;
    Left := 24
    Height := 312
    Top := 64
    Width := 344
    ParentFont := False
    TabOrder := 1
  end;
  //object ButLoadFile: TButton
  ButLoadFile:= TButton.create(self)
  with ButLoadFile do begin
   parent:= formvisuallearning;  
    Left := 30
    Height := 30
    Top := 16
    Width := 160
    Caption := 'Load Neural Network...'
    OnClick := @ButLoadFileClick
    ParentFont := False
    TabOrder := 2
  end;
  LabLayer:= TLabel.create(self)
  with lablayer do begin
   parent:= formvisuallearning;
    Left := 296
    Height := 20
    Top := 21
    Width := 38
    Caption := 'Layer:'
    ParentColor := False
  end ;
  lblfilename:= TLabel.create(self)
  with lblfilename do begin
   parent:= formvisuallearning;
    Left := 890
    Height := 20
    Top := 21
    Width := 38
    Caption := 'Filename:'
    ParentColor := False
  end ;
  lblfile:= TLabel.create(self)
  with lblfile do begin
   parent:= formvisuallearning;
    Left := 940
    Height := 20
    font.size:= 10;
    Top := 19
    Width := 38
    Caption := 'Filename:'
    ParentColor := False
  end ;
  ComboLayer:= TComboBox.create(self)
  with combolayer do begin
   parent:= formvisuallearning;
    Left := 344
    Height := 30
    Top := 17
    Width := 264
    Enabled := False
    font.size:= 11;
    ItemHeight := 25
    Style := csDropDownList
    TabOrder := 3
  end ;
  ChkStrongInput:= TCheckBox.create(self)
  with chkstronginput do begin
   parent:= formvisuallearning;
    Left := 768
    Height := 24
    Top := 8
    Width := 106
    Caption := 'Strong Input'
    Checked := True
    State := cbChecked
    TabOrder := 4
  end;
  //object ChkForceInputRange: TCheckBox
  ChkForceInputRange:= TCheckBox.create(self)
  with ChkForceInputRange do begin
   parent:= formvisuallearning;
    Left := 768
    Height := 24
    Top := 32
    Width := 144
    Caption := 'Force Input Range'
    TabOrder := 5
  end;
  OpenDialogNN:= TOpenDialog.create(self);
  with opendialogNN do begin
    Title := 'Open existing Neural Network File'
    Filter := 'Neural Network|*.nn'
    //left := 488
    //top := 120
  end;
 end;
end; 

procedure TFormVisualLearningDisplayInputImage2(ImgInput: TNNetVolume; 
                                  fimagecnt, color_encoding: integer);
var
  pMin0, pMax0: TNeuralFloat;
  pMin1, pMax1: TNeuralFloat;
  pMin2, pMax2: TNeuralFloat;
  FDisplay : TNNetVolume;
begin
  FDisplay := TNNetVolume.Create();
  FDisplay.Resize(ImgInput, 0,0);
  FDisplay.Copy76(ImgInput);

  if color_encoding = csEncodeLAB then begin
    FDisplay.GetMinMaxAtDepth(0, pMin0, pMax0);
    FDisplay.GetMinMaxAtDepth(1, pMin1, pMax1);
    FDisplay.GetMinMaxAtDepth(2, pMin2, pMax2);
    pMax0 := MaxF(Abs(pMin0), Abs(pMax0));
    pMax1 := MaxF(Abs(pMin1), Abs(pMax1));
    pMax2 := MaxF(Abs(pMin2), Abs(pMax2));

    if pMax0 > 2 then
    begin
      FDisplay.MulAtDepth27(0, 2/pMax0);
    end;

    if pMax1 > 2 then
    begin
      FDisplay.MulAtDepth27(1, 2/pMax1);
    end;

    if pMax2 > 2 then
    begin
      FDisplay.MulAtDepth27(2, 2/pMax2);
    end;
  end
  else if FDisplay.GetMaxAbs() > 2 then
  begin
    FDisplay.NormalizeMax(2);
  end;

  //Debug only: FDisplay.PrintDebugChannel();
  FDisplay.NeuronalInputToRgbImg(color_encoding);
  LoadVolumeIntoTImage(FDisplay, aImage[FImageCnt],color_encoding);
  aImage[FImageCnt].Width := 64;
  aImage[FImageCnt].Height := 64;
  TFormVisualLearningProcessMessages();
  FImageCnt := (FImageCnt + 1) mod Length(aImage);
  FDisplay.Free;
end;  

procedure TForm1ConvertIcon2BitmapClick(Sender: TObject);
var 
  s : string;
  Icon: TIcon; OpenDialog1: TOpenDialog; Image1: TImage;
begin
  OpenDialog1.DefaultExt := '.ICO';
  OpenDialog1.Filter := 'icons (*.ico)|*.ICO';
  OpenDialog1.Options := [ofOverwritePrompt, ofFileMustExist, ofHideReadOnly ];
  if OpenDialog1.Execute then
  begin
    Icon := TIcon.Create;
    try
      Icon.Loadfromfile(OpenDialog1.FileName);
      s:= ChangeFileExt(OpenDialog1.FileName,'.BMP');
      Image1.Width := Icon.Width;
      Image1.Height := Icon.Height;
      Image1.Canvas.Draw(0,0,Icon);
      Image1.Picture.SaveToFile(s);
      ShowMessage(OpenDialog1.FileName + ' Saved to ' + s);
    finally
      Icon.Free;
    end;
  end;
end;

function NeuronForceMinMax(x, pMin, pMax: TNeuralFloat): TNeuralFloat;
begin
  if (x>pMax) then Result := pMax
  else if (x<pMin) then Result := pMin
  else Result := x;
end;

procedure LoadVolumeIntoImage2(Vol: TNNetVolume; var M: TImage);
var
  CountX, CountY, MaxX, MaxY: integer;
  LocalColor: TColor;
  RawPos: integer;
  R,G,B: Byte;
  Data:  TNeuralFloatArray;
  abit: TBitmap;
begin
  MaxX := Vol.SizeX - 1;
  MaxY := Vol.SizeY - 1;
  //M.Size(Vol.SizeX, Vol.SizeY);
  M.height:=  Vol.SizeY;
  m.Width:=   Vol.SizeX;
  writeln('debug mheight'+itoa(m.height));
  for CountX := 0 to MaxX do
  begin
    for CountY := 0 to MaxY do
    begin
      RawPos := Vol.GetRawPos48(CountX, CountY, 0);
      data:= Vol.FData;
      //NeuronForceMinMax88(Round(Data[RawPos])) ;
       R := round(NeuronForceMinMax(Round(Data[RawPos]),0,255));
      G:=  round(NeuronForceMinMax(Round(Data[RawPos + 1]),0,255));
     B := round(NeuronForceMinMax(Round(Data[RawPos + 2]),0, 255));
      writeln(itoa(R)+flots(data[county]))
      //M.Colors[CountX, CountY] := rgbtocolor(r,g,b);
      M.Canvas.Pixels[countX, countY] := rgbtocolor(r,g,b);
      writeln('debug width color '+itoa(rgbtocolor(r,g,b))+'  ');
    end;
  end;
  writeln('debug width '+itoa(m.width)+'  ');
  //Imgdirect.Canvas.Draw(0,0,m.canvas);
 // abit:= TBitmap.create;
  //abit.assign(m)
  // abit:= m.picture.bitmap;
 
  //M.picture.savetofile(exepath+'cifar10.bmp');
  //abit.savetofile (exepath+'cifar10.bmp');
  //openfile(exepath+'cifar10.bmp');
  //abit.free;
end;

procedure TTestCNNAlgoDoRunMNIST;
  var
    NN: THistoricalNets;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
  begin
    if Not(CheckMNISTFile('train', false)) or Not(CheckMNISTFile('t10k', false)) then
    begin
      //Terminate;
      //exit;
      writeln('mnist not found data ');
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    try
    NN.AddLayer49([                         
      TNNetInput.Create4(28, 28, 1),
      TNNetConvolutionLinear.Create(32, 5, 2, 1, 1),
      TNNetMaxPool.Create44(4,0,0),
      TNNetConvolutionReLU.Create(32, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(32, 3, 1, 1, 1),
      TNNetFullConnectReLU.Create28(32,0),
      TNNetFullConnectReLU.Create28(32,0),
      TNNetDropout.Create12(0.2,1),
      TNNetMaxPool.Create44(2,0,0),
      TNNetFullConnectLinear.Create28(10,0),
      TNNetSoftMax.Create()
    ]);
    CreateMNISTVolumes(ImgTrainingVolumes, 
        ImgValidationVolumes, ImgTestVolumes, 'train', 't10k', true, false);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleMNist_mX4_fakeNN';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.HasFlipX := false;
    NeuralFit.HasFlipY := false;
    NeuralFit.MaxCropSize := 4;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,    {NumClasses=} 10, {batchsize=}128, {epochs=}20); 
   finally
    NeuralFit.Free;
    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    //Terminate;
    writeln('MNIST classifier done...')
  end;  //(/}
end;  

  
var ImgVolumes: TNNetVolumeList;
    Volume: TNNetVolume;  TI : TTinyImage;  img: Tbitmap;  img1: TImage;

begin  //@main

  writeln('CPUSpeed '+cpuspeed);
  
  loadvisualneuralForm;
  
 { if CheckCIFARFile() then begin
    ImgVolumes:= TNNetVolumeList.create(true);
    // creates required volumes to store images
  for it := 0 to 9999 do begin
    Volume := TNNetVolume.Create();
    ImgVolumes.Add(Volume);
  end;
  //------------
    loadCifar10Dataset(ImgVolumes,
        'C:\maXbox\EKON_BASTA\EKON24\cifar-10-batches-bin\data_batch_1.bin',0, csEncodeRGB);
    //imgdirect.clear;
    LoadVolumeIntoImage( ImgVolumes[50],  imgdirect);
    writeln(itoa(imgvolumes[50].tag ))
    //imgsample.refresh;
    writeln(itoa(imgdirect.width))
    //imgsample.picture.savetofile (exepath+'cifarpic.bmp')
    //openfile( exepath+'cifarpic.bmp');
    //SetLength(aImage, 9999);
    //TFormVisualLearningDisplayInputImage(ImgVolumes[1], 1, csEncodeRGB);
  end; 
  ImgVolumes.Free;    //}
  
  CreateMNISTVolumes(ImgTrainingVolumes, ImgValidationVolumes, 
                ImgTestVolumes, 'train', 't10k', false, false);   
   if Not(CheckMNISTFile('train', false)) or Not(CheckMNISTFile('t10k', false)) then
    writeln('mnist not found');  // } 
    
    //TTestCNNAlgoDoRunMNIST; 
    
    {  CreateCifar10Volumes(ImgTrainingVolumes, ImgTrainingVolumes, ImgTrainingVolumes, FColorEncoding); 
      //ImgTrainingVolumes.free;    
      //LoadVolumeIntoImage2( ImgTrainingVolumes[2],  imgdirect); 
      img1:= TImage.create(self);
      LoadNNetVolumeIntoTinyImage4(ImgTrainingVolumes[5], TI );
      writeln('debug label TI'+itoa(TI.blabel)+'  ');
      LoadTinyImageIntoTImage(TI, img1);
      writeln('debug width '+itoa(img1.width)+'  ');
      img1.picture.savetofile (exepath+'cifarpic.bmp')  }

End.

{   https://www.youtube.com/c/EmbarcaderoTechnologies/videos

 196320 Examples seen. Accuracy: 0.9919 Error: 0.01550 Loss: 0.01474 Threads: 1 Forward time: 1.32s Backward time: 0.20s Step time: 7.82s
1197600 Examples seen. Accuracy: 0.9919 Error: 0.00800 Loss: 0.00429 Threads: 1 Forward time: 1.29s Backward time: 0.21s Step time: 7.66s
1198880 Examples seen. Accuracy: 0.9918 Error: 0.01036 Loss: 0.00615 Threads: 1 Forward time: 1.30s Backward time: 0.20s Step time: 7.55s
60000 of samples have been processed.
Starting Validation.
Epochs: 20 Examples seen:1200000 Validation Accuracy: 0.9908 Validation Error: 0.0237 Validation Loss: 0.0255 Total time: 137.05min
Image mX4 FThreadNN[0].DebugWeights(); skipped...
Starting Testing.
Epochs: 20 Examples seen:1200000 Test Accuracy: 0.9966 Test Error: 0.0093 Test Loss: 0.0117 Total time: 137.90min
Epoch time: 5.9000 minutes. 20 epochs: 2.0000 hours.
Epochs: 20. Working time: 2.30 hours.
CAI maXbox Neural Fit Finished.
mnist classifier done...
 mX4 executed: 19/11/2021 16:43:30  Runtime: 2:18:1.308  Memload: 46% use
PascalScript maXbox4 - RemObjects & SynEdit




Doc: # Gradient Ascent Example
<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/gradientascent.jpg" height="192">

It's usually very hard to understand neuron by neuron how a neural network dedicated to image classification internally works. 
One technique used to help with the understanding about what individual neurons represent is called Gradient Ascent.

In this technique, an arbitrary neuron is required to activate and then the same backpropagation method used for learning is
applied to an input image producing an image that this neuron expects to see.

To be able to run this example, you'll need to load an already trained neural network file and then select the layer you intend to visualize.

Deeper convolutional layers tend to produce more complex patterns. Above image was produced from the a first convolutional layer. The following image was produced from a third convolutional layer. Notice that patterns are more complex.

<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/gradientascent3layer.jpg" height="192">

This is the API method used for an arbitrary neuron backpropagation (Gradient Ascent):
```
procedure TNNet.BackpropagateFromLayerAndNeuron(LayerIdx, NeuronIdx: integer; Error: TNeuralFloat);
```

Errors on the input image aren't enabled by default. In this example, errors regarding the input image are enabled with this:
```
TNNetInput(FNN.Layers[0]).EnableErrorCollection();
```

Then, errors are added to the input with this:
```
vInput.MulAdd(-1, FNN.Layers[0].OutputError);
FNN.ClearDeltas();
FNN.ClearInertia();
```

You can find more about Gradient Ascent at:
* [Lecture 12: Visualizing and Understanding - CS231n - Stanford](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf)
* [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)

Starting Testing.
Epochs: 50 Examples seen:2400000 Test Accuracy: 0.8481 Test Error: 0.4238 Test Loss: 0.4620 Total time: 188.44min
Epoch time: 3.2887 minutes. 50 epochs: 2.7406 hours.
Epochs: 50. Working time: 3.14 hours.
Loading SimpleImageClassifier-60.nn for final test.
Starting Testing.
Epochs: 50 Examples seen:2400000 Test Accuracy: 0.8481 Test Error: 0.4238 Test Loss: 0.4620 Total time: 189.71min
Loading best performing results SimpleImageClassifier-60.nn.
Finished.

 procedure TTestCNNAlgo.DoRun;
  var
    NN: THistoricalNets;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
  begin
    if Not(CheckMNISTFile('train')) or Not(CheckMNISTFile('t10k')) then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(28, 28, 1),
      TNNetConvolutionLinear.Create(32, 5, 2, 1, 1),
      TNNetMaxPool.Create(4),
      TNNetConvolutionReLU.Create(32, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(32, 3, 1, 1, 1),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetDropout.Create(0.2),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    CreateMNISTVolumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, 'train', 't10k');

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleMNist';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.HasFlipX := false;
    NeuralFit.HasFlipY := false;
    NeuralFit.MaxCropSize := 4;
 { NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}//10, {batchsize=}128, {epochs=}20); }
  {  NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    Terminate;
  end;  }


//}
