///This file has an implementation to classify
//the TinyImageNet 200 dataset: https://tiny-imagenet.herokuapp.com/ .
program SimpleTinyImageNet;
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
    procedure DoRun; override;
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    NN: TNNet;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    ProportionToLoad: Single;
  begin
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(64, 64, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}2, {Stride=}2),
      TNNetMaxPool.Create(2),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetMaxPool.Create(2),
      TNNetConvolutionReLU.Create({Features=}128, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionReLU.Create({Features=}128, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionReLU.Create({Features=}256, {FeatureSize=}3, {Padding=}1, {Stride=}2),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(200),
      TNNetSoftMax.Create()
    ]);
    NN.DebugStructure();
    // change ProportionToLoad to a smaller number if you don't have available 6GB of RAM.
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

    WriteLn
    (
      'Training Images:', ImgTrainingVolumes.Count,
      ' Validation Images:', ImgValidationVolumes.Count,
      ' Test Images:', ImgTestVolumes.Count
    );

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleTinyImageNet200';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    //NeuralFit.MaxThreadNum := 8;
    NeuralFit.CyclicalLearningRateLen := 100;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}200, {batchsize=}64, {epochs=}400);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    Terminate;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='Tiny ImageNet 200 Classification Example';
  Application.Run;
  Application.Free;
end.
