program SimpleImageClassifierGroupedConv;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume,
  Math, neuraldatasets, neuralfit, neuralthread;

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
  begin
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}2, {Stride=}1, {SuppressBias=}1),
      TNNetMaxPool.Create(4)
    ]);
    NN.AddGroupedConvolution(TNNetConvolutionHardSwish,
      {Groups=}4, {Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1);
    NN.AddGroupedConvolution(TNNetConvolutionHardSwish,
      {Groups=}4, {Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1);
    NN.AddGroupedConvolution(TNNetConvolutionHardSwish,
      {Groups=}4, {Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1);
    NN.AddGroupedConvolution(TNNetConvolutionHardSwish,
      {Groups=}4, {Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1);
    NN.AddLayer([
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    NN.DebugStructure();

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifierGroupedConv-'+IntToStr(GetProcessId());
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0;
    //NeuralFit.MaxThreadNum := 1;
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
