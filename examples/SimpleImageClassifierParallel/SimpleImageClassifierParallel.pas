program Cifar10Parallel;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets, neuralfit;

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
    I: integer;
  begin
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}0, {Stride=}2, {SuppressBias=}0)
    ]);
    NN.AddLayer( TNNetMulLearning.Create(2) );
    for I := 1 to 2 do
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}false,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {PreviousLayer}nil,
        {BottleNeck}16,
        {p11ConvCount}2,
        {p11FilterCount}16,
        {p33ConvCount}2,
        {p33FilterCount}16,
        {p55ConvCount}2,
        {p55FilterCount}16,
        {p77ConvCount}2,
        {p77FilterCount}16,
        {maxPool}0
    );
    NN.AddLayer( TNNetMaxPool.Create(2) );
    for I := 1 to 2 do
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}false,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {PreviousLayer}nil,
        {BottleNeck}16,
        {p11ConvCount}2,
        {p11FilterCount}16,
        {p33ConvCount}2,
        {p33FilterCount}16,
        {p55ConvCount}2,
        {p55FilterCount}16,
        {p77ConvCount}2,
        {p77FilterCount}16,
        {maxPool}16
    );
    NN.AddLayer([
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    NN.DebugStructure();
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifierParallel';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.MaxThreadNum := 32;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}64, {epochs=}50);
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
  Application.Title:='CIFAR-10 Classification Example';
  Application.Run;
  Application.Free;
end.
