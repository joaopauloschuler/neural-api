program SimpleImageClassifierParallel;
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
  begin
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    NN.AddLayer( TNNetInput.Create(32, 32, 3) );
    NN.AddLayer([
      TNNetConvolutionLinear.Create({Features=}32, {FeatureSize=}3, {Padding=}0, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}0, {Stride=}1, {SuppressBias=}1)
    ]);
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}false,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {BottleNeck}32,
        {p11count}16,
        {p33count}16,
        {p55count}16,
        {p77count}16,
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
        {BottleNeck}32,
        {p11count}16,
        {p33count}16,
        {p55count}16,
        {p77count}16,
        {maxPool}16
    );
    NN.AddLayer( TNNetMaxPool.Create(2) );
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}false,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {BottleNeck}32,
        {p11count}48,
        {p33count}16,
        {p55count}16,
        {p77count}16,
        {maxPool}8,
        {minPool}8
    );
    NN.AddParallelConvs(
        {PointWiseConv}TNNetConvolutionLinear,
        {IsSeparable}false,
        {CopyInput}true,
        {pBeforeBottleNeck}nil,
        {pAfterBottleNeck}nil,
        {pBeforeConv}nil,
        {pAfterConv}TNNetReLU,
        {BottleNeck}32,
        {p11count}48,
        {p33count}16,
        {p55count}16,
        {p77count}16,
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
        {BottleNeck}32,
        {p11count}0,
        {p33count}16,
        {p55count}16,
        {p77count}16,
        {maxPool}8
    );
    NN.AddLayer([
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(4),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    NN.DebugStructure();
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifierParallel';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.03;
    NeuralFit.CyclicalLearningRateLen := 100;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.MaxThreadNum := 32;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}64, {epochs=}300);
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
