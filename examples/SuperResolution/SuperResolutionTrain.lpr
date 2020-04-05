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
  begin
    if not CheckCIFARFile() then
    begin
      Terminate;
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := THistoricalNets.Create();
    if FileExists('super-resolution-cifar10.nn') then
    begin
      NN.LoadFromFile('super-resolution-cifar10.nn');
    end
    else
    begin
      NN.AddSuperResolution({pSizeX=}16, {pSizeY=}16, {pNeurons=}64, {pLayerCnt=}7);
    end;
    NN.DebugStructure();

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    ImgTrainingSmall := TNNetVolumeList.Create();
    ImgValidationSmall := TNNetVolumeList.Create();
    ImgTestSmall := TNNetVolumeList.Create();

    ImgTrainingSmall.AddVolumes(ImgTrainingVolumes);
    ImgValidationSmall.AddVolumes(ImgValidationVolumes);
    ImgTestSmall.AddVolumes(ImgTestVolumes);

    ImgTrainingSmall.ResizeImage(16, 16);
    ImgValidationSmall.ResizeImage(16, 16);
    ImgTestSmall.ResizeImage(16, 16);

    NeuralFit := TNeuralDataLoadingFit.Create;
    NeuralFit.FileNameBase := 'super-resolution-cifar10';
    NeuralFit.InitialLearningRate := 0.01/(32*32);
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.Verbose := true;
    NeuralFit.EnableBipolarHitComparison();
    NeuralFit.FitLoading(NN,
      ImgTrainingVolumes.Count, ImgValidationVolumes.Count, ImgTestVolumes.Count,
      {batchsize=}64, {epochs=}50,
      @GetTrainingPair, @GetValidationPair, @GetTestPair);

    NeuralFit.Free;

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
