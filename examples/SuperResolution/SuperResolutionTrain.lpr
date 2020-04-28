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
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets,
  neuralfit, usuperresolutionexample;

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
    NNMaxPool: TNNet;
    NeuralFit: TNeuralDataLoadingFit;
  begin
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddSuperResolution({pSizeX=}16, {pSizeY=}16,
      {BottleNeck=}csExampleBottleNeck, {pNeurons=}csExampleNeuronCount,
      {pLayerCnt=}csExampleLayerCount, {IsSeparable=}csExampleIsSeparable);
    LoadResizingWeights(NN, csExampleFileName);
    NN.DebugStructure();

    // Small Neural Network to resize 32x32 images into 16x16 images.
    NNMaxPool := TNNet.Create();
    NNMaxPool.AddLayer( TNNetInput.Create(32, 32, 3) );
    NNMaxPool.AddLayer( TNNetMaxPool.Create(2) );

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    ImgTrainingSmall := TNNetVolumeList.Create();
    ImgValidationSmall := TNNetVolumeList.Create();
    ImgTestSmall := TNNetVolumeList.Create();

    NNMaxPool.Compute(ImgTrainingVolumes, ImgTrainingSmall);
    NNMaxPool.Compute(ImgValidationVolumes, ImgValidationSmall);
    NNMaxPool.Compute(ImgTestVolumes, ImgTestSmall);

    NeuralFit := TNeuralDataLoadingFit.Create;
    NeuralFit.FileNameBase := csExampleBaseFileName;
    NeuralFit.InitialLearningRate := 0.001/(32*32);
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0;
    NeuralFit.Verbose := true;
    NeuralFit.EnableBipolar99HitComparison();
    NeuralFit.AvgWeightEpochCount := 1;
    //NeuralFit.MaxThreadNum := 16;
    NeuralFit.FitLoading(NN,
      ImgTrainingVolumes.Count, ImgValidationVolumes.Count, ImgTestVolumes.Count,
      {batchsize=}64, {epochs=}50,
      @GetTrainingPair, @GetValidationPair, @GetTestPair);
    NeuralFit.Free;

    NN.Free;
    NNMaxPool.Free;
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
  var
    LocalIdx: integer;
  begin
    LocalIdx := Random(ImgTrainingSmall.Count);
    pInput.Copy(ImgTrainingSmall[LocalIdx]);
    pOutput.Copy(ImgTrainingVolumes[LocalIdx]);
    // insert data augmentation
    if Random(1000)>500 then
    begin
      pInput.FlipX();
      pOutput.FlipX();
    end;
    if Random(1000)>500 then
    begin
      pInput.FlipY();
      pOutput.FlipY();
    end;
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
  Application.Title:='CIFAR-10 Super Resolution Train';
  Application.Run;
  Application.Free;
end.
