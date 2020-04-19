///This example trains a super resolution neural network with
//the TinyImageNet 200 dataset: https://tiny-imagenet.herokuapp.com/ .
program SuperResolutionTrain;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets, neuralfit;

type

  { TTestCNNAlgo }

  TTestCNNAlgo = class(TCustomApplication)
  protected
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    ImgTrainingSmall, ImgValidationSmall, ImgTestSmall: TNNetVolumeList;
    procedure DoRun; override;
  public
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    NN: THistoricalNets;
    NeuralFit: TNeuralDataLoadingFit;
    ProportionToLoad: Single;
    BaseFileName: string;
    FinalFileName: string;
  begin
    WriteLn('Creating Neural Network...');
    BaseFileName := 'super-resolution-tiny-image-net-200';
    FinalFileName := BaseFileName+'-final.nn';
    NN := THistoricalNets.Create();
    if FileExists(FinalFileName) then
    begin
      NN.LoadFromFile(FinalFileName);
    end
    else
    begin
      NN.AddSuperResolution({pSizeX=}32, {pSizeY=}32, {pNeurons=}16, {pLayerCnt=}7);
    end;
    NN.DebugStructure();

    ProportionToLoad := 1;
    WriteLn('Loading ', Round(ProportionToLoad*100), '% of the Tiny ImageNet 200 dataset into memory.');
    CreateVolumesFromImagesFromFolder
    (
      ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      {FolderName=}'tiny-imagenet-200/train', {pImageSubFolder=}'images',
      {color_encoding=}0{RGB},
      {TrainingProp=}0.9*ProportionToLoad,
      {ValidationProp=}0.05*ProportionToLoad,
      {TestProp=}0.05*ProportionToLoad
    );

    ImgTrainingSmall := TNNetVolumeList.Create();
    ImgValidationSmall := TNNetVolumeList.Create();
    ImgTestSmall := TNNetVolumeList.Create();

    ImgTrainingSmall.AddVolumes(ImgTrainingVolumes);
    ImgValidationSmall.AddVolumes(ImgValidationVolumes);
    ImgTestSmall.AddVolumes(ImgTestVolumes);

    ImgTrainingSmall.ResizeImage(32, 32);
    ImgValidationSmall.ResizeImage(32, 32);
    ImgTestSmall.ResizeImage(32, 32);

    NeuralFit := TNeuralDataLoadingFit.Create;
    NeuralFit.FileNameBase := BaseFileName;
    NeuralFit.InitialLearningRate := 0.001/(32*32);
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Verbose := true;
    NeuralFit.EnableBipolar99HitComparison();
    NeuralFit.AvgWeightEpochCount := 1;
    //NeuralFit.MaxThreadNum := 8;
    NeuralFit.FitLoading(NN,
      ImgTrainingVolumes.Count, ImgValidationVolumes.Count, ImgTestVolumes.Count,
      {batchsize=}64, {epochs=}10,
      @GetTrainingPair, @GetValidationPair, @GetTestPair);
    NeuralFit.Free;

    NN.SaveToFile(FinalFileName);
    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    ImgTrainingSmall.Free;
    ImgValidationSmall.Free;
    ImgTestSmall.Free;
    Terminate;
  end;

  procedure TTestCNNAlgo.GetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    pInput.Copy(ImgTrainingSmall[Idx]);
    pOutput.Copy(ImgTrainingVolumes[Idx]);
  end;

  procedure TTestCNNAlgo.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    pInput.Copy(ImgValidationSmall[Idx]);
    pOutput.Copy(ImgValidationVolumes[Idx]);
  end;

  procedure TTestCNNAlgo.GetTestPair(Idx: integer; ThreadId: integer; pInput,
    pOutput: TNNetVolume);
  begin
    pInput.Copy(ImgTestSmall[Idx]);
    pOutput.Copy(ImgTestVolumes[Idx]);
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-10 Classification Example';
  Application.Run;
  Application.Free;
end.
