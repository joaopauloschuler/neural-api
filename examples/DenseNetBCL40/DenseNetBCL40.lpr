program DenseNetBCL40;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
 This command line tool trains a DenseNetBC L40 with CIFAR-10.
 This code is inspired on:
 https://github.com/liuzhuang13/DenseNet
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  SysUtils,
  CustApp,
  neuralnetwork,
  neuralvolume,
  Math,
  neuraldatasets,
  neuralfit;

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    fLearningRate, fInertia, fTarget, fDropout: single;
    procedure DoRun; override;
    procedure Train();
  public
    constructor Create(TheOwner: TComponent); override;
    procedure WriteHelp; virtual;

    function DenseNetLearningRateSchedule(Epoch: integer): single;
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    LearningRate, Inertia, Target: string;
  begin
    // parse parameters
    if HasOption('h', 'help') then
    begin
      WriteHelp;
      Terminate;
      Exit;
    end;

    fLearningRate := 0.001;
    if HasOption('l', 'learningrate') then
    begin
      LearningRate := GetOptionValue('l', 'learningrate');
      fLearningRate := StrToFloat(LearningRate);
    end;

    fInertia := 0.9; //0.75 should also be tested.
    if HasOption('i', 'inertia') then
    begin
      Inertia := GetOptionValue('i', 'inertia');
      fInertia := StrToFloat(Inertia);
    end;

    fTarget := 1;
    if HasOption('t', 'target') then
    begin
      Target := GetOptionValue('t', 'target');
      fTarget := StrToFloat(Target);
    end;

    fDropout := 0.0;
    if HasOption('d', 'dropout') then
    begin
      Target := GetOptionValue('d', 'dropout');
      fDropout := StrToFloat(Target);
    end;

    Train();

    Terminate;
  end;

  procedure TTestCNNAlgo.Train();
  var
    NN: THistoricalNets;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
    NumClasses: integer;
    fileNameBase: string;
  begin
    if not CheckCIFARFile() then exit;
    WriteLn('Creating Neural Network...');
    NumClasses  := 10;
    NN := THistoricalNets.Create();
    fileNameBase := 'DenseNetBCL40';

    NN.AddLayer( TNNetInput.Create(32, 32, 3) );
    NN.AddLayer( TNNetConvolutionLinear.Create(64, 3, 1, 1, 1) ).InitBasicPatterns();
    NN.AddDenseNetBlock(6, 12, 48, 0, fDropout);
    NN.AddDenseNetTransition(0.5, 1, false);
    NN.AddDenseNetBlock(6, 12, 48, 0, fDropout);
    NN.AddDenseNetTransition(0.5, 1, false);
    NN.AddDenseNetBlock(6, 12, 48, 0, fDropout);
    NN.AddMovingNorm(false, 0, 0);
    NN.AddLayer( TNNetSELU.Create() );
    NN.AddLayer( TNNetMaxChannel.Create() );
    NN.AddLayer( TNNetFullConnectLinear.Create(NumClasses) );
    NN.AddLayer( TNNetSoftMax.Create() );

    WriteLn('Learning rate set to: [',fLearningRate:7:5,']');
    WriteLn('Inertia set to: [',fInertia:7:5,']');
    WriteLn('Target set to: [',fTarget:7:5,']');

    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    WriteLn('Neural Network will minimize error with:');
    WriteLn(' Layers: ', NN.CountLayers());
    WriteLn(' Neurons:', NN.CountNeurons());
    WriteLn(' Weights:' ,NN.CountWeights());
    NN.DebugWeights();
    NN.DebugStructure();

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := fileNameBase;
    NeuralFit.InitialLearningRate := fLearningRate;
    NeuralFit.Inertia := fInertia;
    NeuralFit.TargetAccuracy := fTarget;
    NeuralFit.CustomLearningRateScheduleObjFn := @Self.DenseNetLearningRateSchedule;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, NumClasses, {batchsize=}64, {epochs=}300);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
  end;

  constructor TTestCNNAlgo.Create(TheOwner: TComponent);
  begin
    inherited Create(TheOwner);
    StopOnException := True;
  end;

  procedure TTestCNNAlgo.WriteHelp;
  begin
    WriteLn
    (
      'CIFAR-10 DenseNetBC L40 Classification Example by Joao Paulo Schwarz Schuler',sLineBreak,
      'Command Line Example: DenseNetBCL40 -i 0.8', sLineBreak,
      ' -h : displays this help. ', sLineBreak,
      ' -i : defines inertia. Default is -i 0.9.', sLineBreak,
      ' -l : defines learing rate. Default is -l 0.001. ', sLineBreak,
      ' -d : defines dropout rate. Default is -d 0.0. ', sLineBreak,
      ' https://sourceforge.net/p/cai/svncode/HEAD/tree/trunk/lazarus/examples/DenseNetBCL40/',sLineBreak,
      ' More info at:',sLineBreak,
      '   https://github.com/joaopauloschuler/neural-api'
    );
  end;

  function TTestCNNAlgo.DenseNetLearningRateSchedule(Epoch: integer): single;
  begin
    if Epoch < 150
      then Result := fLearningRate
    else if epoch < 225
      then Result :=  fLearningRate * 0.1
    else
      Result := fLearningRate * 0.01;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);
  Application.Title:='CIFAR-10 DenseNetBC L40 Example';
  Application.Run;
  Application.Free;
end.
