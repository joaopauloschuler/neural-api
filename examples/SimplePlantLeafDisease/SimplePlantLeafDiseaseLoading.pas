///This file has an implementation to classify
//plant leaf diseases. You can get the dataset at
//https://data.mendeley.com/datasets/tywbtsjrjv/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/Plant_leaf_diseases_dataset_without_augmentation.zip?dl=1 .
//Folders with plant diseases will need to be stored inside of a folder named "plant".
program SimplePlantLeafDiseaseLoading;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets,
  neuralfit;

type

  TTestCNNAlgo = class(TCustomApplication)
  protected
    FSizeX, FSizeY: integer;
    FTrainingFileNames, FValidationFileNames, FTestFileNames: TFileNameList;
    procedure DoRun; override;
  public
    procedure GetTrainingProc(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationProc(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestProc(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    NN: TNNet;
    NeuralFit: TNeuralDataLoadingFit;
    ProportionToLoad: Single;
  begin
    FSizeX := 128;
    FSizeY := 128;
    ProportionToLoad := 1;
    WriteLn('Loading ', Round(ProportionToLoad*100), '% of the Plant leave disease file names into memory.');

    CreateFileNameListsFromImagesFromFolder(
      FTrainingFileNames, FValidationFileNames, FTestFileNames,
      {FolderName=}'plant', {pImageSubFolder=}'',
      {TrainingProp=}0.9*ProportionToLoad,
      {ValidationProp=}0.05*ProportionToLoad,
      {TestProp=}0.05*ProportionToLoad
    );

    WriteLn('Creating Neural Network...');
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(FSizeX, FSizeY, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}4, {Stride=}2),
      TNNetMaxPool.Create(2),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}2),
      TNNetMaxPool.Create(2),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}2),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(FTrainingFileNames.ClassCount),
      TNNetSoftMax.Create()
    ]);
    NN.DebugStructure();
    WriteLn
    (
      'Training Images:', FTrainingFileNames.Count,
      ' Validation Images:', FValidationFileNames.Count,
      ' Test Images:', FTestFileNames.Count,
      ' Classes:', FTrainingFileNames.ClassCount
    );

    NeuralFit := TNeuralDataLoadingFit.Create;
    NeuralFit.EnableDefaultImageTreatment();
    NeuralFit.FileNameBase := 'SimplePlantLeafDiseaseLoading';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    //NeuralFit.MaxThreadNum := 1;
    //NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}39, {batchsize=}64, {epochs=}50);
    NeuralFit.FitLoading(NN, FTrainingFileNames.Count, FValidationFileNames.Count, FTestFileNames.Count, 64, 10, @GetTrainingProc, @GetValidationProc, @GetTestProc);
    NeuralFit.Free;

    FTrainingFileNames.Free;
    FValidationFileNames.Free;
    FTestFileNames.Free;
    NN.Free;
    Terminate;
  end;

  procedure TTestCNNAlgo.GetTrainingProc(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    AuxVolume: TNNetVolume;
  begin
    AuxVolume := TNNetVolume.Create();
    FTrainingFileNames.GetRandomImagePair(AuxVolume, pOutput);
    pInput.CopyResizing(AuxVolume, FSizeX, FSizeY);
    AuxVolume.Free;
  end;

  procedure TTestCNNAlgo.GetValidationProc(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    AuxVolume: TNNetVolume;
  begin
    AuxVolume := TNNetVolume.Create();
    FValidationFileNames.GetImageVolumePairFromId(Idx, AuxVolume, pOutput);
    pInput.CopyResizing(AuxVolume, FSizeX, FSizeY);
    AuxVolume.Free;
  end;

  procedure TTestCNNAlgo.GetTestProc(Idx: integer; ThreadId: integer; pInput,
    pOutput: TNNetVolume);
  var
    AuxVolume: TNNetVolume;
  begin
    AuxVolume := TNNetVolume.Create();
    FTestFileNames.GetImageVolumePairFromId(Idx, AuxVolume, pOutput);
    pInput.CopyResizing(AuxVolume, FSizeX, FSizeY);
    AuxVolume.Free;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='Plant Leaf Disease Classification';
  Application.Run;
  Application.Free;
end.
