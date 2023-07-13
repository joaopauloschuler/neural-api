///This file has an implementation to classify
// the Colorectal Cancer Dataset:
// https://zenodo.org/record/53169/
// https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1
// https://www.tensorflow.org/datasets/catalog/colorectal_histology

// Change ProportionToLoad to a smaller number if you don't have available 4GB of RAM.

program ColorectalImageClassification;
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
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(128, 128, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}4, {Stride=}2),
      TNNetMaxPool.Create(2),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetMaxPool.Create(2),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}2),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(8),
      TNNetSoftMax.Create()
    ]);
    NN.DebugStructure();
    // change ProportionToLoad to a smaller number if you don't have available 8GB of RAM.
    ProportionToLoad := 1;
    WriteLn('Loading ', Round(ProportionToLoad*100), '% of the Plant leave disease dataset into memory.');
    CreateVolumesFromImagesFromFolder
    (
      ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      {FolderName=}'Kather_texture_2016_image_tiles_5000', {pImageSubFolder=}'',
      {color_encoding=}0{RGB},
      {TrainingProp=}0.9*ProportionToLoad,
      {ValidationProp=}0.05*ProportionToLoad,
      {TestProp=}0.05*ProportionToLoad,
      {NewSizeX=}128, {NewSizeY=}128
    );

    WriteLn
    (
      'Training Images:', ImgTrainingVolumes.Count,
      ' Validation Images:', ImgValidationVolumes.Count,
      ' Test Images:', ImgTestVolumes.Count
    );

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'Colorectal';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.CyclicalLearningRateLen := 10;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}8, {batchsize=}64, {epochs=}250);
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
  Application.Title:='Colorectal Cancer Image Classification';
  Application.Run;
  Application.Free;
end.
