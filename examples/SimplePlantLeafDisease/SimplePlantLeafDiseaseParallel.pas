///This file has an implementation to classify
//plant leaf diseases. You can get the dataset at
//https://data.mendeley.com/datasets/tywbtsjrjv/1/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/Plant_leaf_diseases_dataset_without_augmentation.zip?dl=1 .
//Folders with plant diseases will need to be stored inside of a folder named "plant".
program SimplePlantLeafDisease;
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
    NN: THistoricalNets;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    ProportionToLoad: Single;
  begin
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(128, 128, 3),
      TNNetConvolutionLinear.Create({Features=}32, {FeatureSize=}3, {Padding=}0, {Stride=}2),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}3, {Padding=}0, {Stride=}1),
      TNNetMaxPool.Create(2)
    ]);
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}false,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {PreviousLayer}nil,
        {BottleNeck}32,
        {p11ConvCount}1,
        {p11FilterCount}16,
        {p33ConvCount}1,
        {p33FilterCount}16,
        {p55ConvCount}1,
        {p55FilterCount}16,
        {p77ConvCount}1,
        {p77FilterCount}16,
        {maxPool}16
    );
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}false,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {PreviousLayer}nil,
        {BottleNeck}32,
        {p11ConvCount}1,
        {p11FilterCount}48,
        {p33ConvCount}1,
        {p33FilterCount}16,
        {p55ConvCount}1,
        {p55FilterCount}16,
        {p77ConvCount}1,
        {p77FilterCount}16,
        {maxPool}8,
        {minPool}8
    );
    NN.AddLayer(TNNetMaxPool.Create(2));
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}true,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {PreviousLayer}nil,
        {BottleNeck}32,
        {p11ConvCount}1,
        {p11FilterCount}48,
        {p33ConvCount}1,
        {p33FilterCount}16,
        {p55ConvCount}1,
        {p55FilterCount}16,
        {p77ConvCount}1,
        {p77FilterCount}16,
        {maxPool}8
    );
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}true,
        {CopyInput}true,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {PreviousLayer}nil,
        {BottleNeck}32,
        {p11ConvCount}1,
        {p11FilterCount}0,
        {p33ConvCount}1,
        {p33FilterCount}16,
        {p55ConvCount}1,
        {p55FilterCount}16,
        {p77ConvCount}1,
        {p77FilterCount}16,
        {maxPool}8
    );
    NN.AddLayer([
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(39),
      TNNetSoftMax.Create({SkipBackpropDerivative=}1)
    ]);
    NN.DebugStructure();
    // change ProportionToLoad to a smaller number if you don't have available 32GB of RAM.
    ProportionToLoad := 1;
    WriteLn('Loading ', Round(ProportionToLoad*100), '% of the Plant leave disease dataset into memory.');
    CreateVolumesFromImagesFromFolder
    (
      ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      {FolderName=}'plant', {pImageSubFolder=}'',
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
    NeuralFit.FileNameBase := 'SimplePlantLeafDiseaseParallel';
    NeuralFit.MaxThreadNum := 32; // batch size divided by 2
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.02;
    NeuralFit.CyclicalLearningRateLen := 50;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}39, {batchsize=}64, {epochs=}50);
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
  Application.Title:='Plant Leaf Disease Classification';
  Application.Run;
  Application.Free;
end.
