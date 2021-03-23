(*
neuralfit
Copyright (C) 2017 Joao Paulo Schwarz Schuler

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*)
unit neuralfit;

{$include neuralnetwork.inc}
{$DEFINE HASTHREADS}

{$IFDEF FPC}
{$H+}
{$ENDIF}

interface

uses
  Classes, SysUtils, neuralnetwork, neuralvolume, neuralthread
  {$IFDEF OpenCL}, cl
  {$IFNDEF FPC}, neuralopencl {$ENDIF}
  {$ENDIF}
  {$IFNDEF FPC}, Windows {$ENDIF}
  ;

type
  TCustomLearningRateScheduleFn = function(Epoch: integer): single;
  TCustomLearningRateScheduleObjFn = function(Epoch: integer): single of object;

  {$IFNDEF HASTHREADS}
  TMultiThreadProcItem = TObject;
  {$ENDIF}

  /// This is a base class. Do not use it directly.
  TNeuralFitBase = class(TMObject)
    protected
      FBatchSize: integer;
      FMaxThreadNum, FDefaultThreadCount, FThreadNum: integer;
      FThreadNN: TNNetDataParallelism;
      FAvgWeights: TNNetDataParallelism;
      FAvgWeight: TNNet;
      FAvgWeightEpochCount: integer;
      FCurrentEpoch: integer;
      FCurrentStep: integer;
      FNN: TNNet;
      FGlobalHit: integer;
      FGlobalMiss: integer;
      FGlobalTotal: integer;
      FGlobalTotalLoss: single;
      FGlobalErrorSum: single;
      FFinishedThread: TNNetVolume;
      {$IFDEF HASTHREADS}FCritSec: TRTLCriticalSection;{$ENDIF}
      FMultipleSamplesAtValidation: boolean;
      FDataAugmentation: boolean;
      FVerbose: boolean;
      FStaircaseEpochs: integer;
      FStepSize: integer;
      FLearningRateDecay: single;
      FInitialLearningRate: single;
      FCyclicalLearningRateLen: integer;
      FInitialEpoch: integer;
      FMaxEpochs: integer;
      FMinLearnRate: single;
      FCurrentLearningRate: single;
      FInertia: single;
      FL2Decay: TNeuralFloat;
      FFileNameBase: string;
      FClipDelta: single;
      FTargetAccuracy: single;
      FCustomLearningRateScheduleFn: TCustomLearningRateScheduleFn;
      FCustomLearningRateScheduleObjFn: TCustomLearningRateScheduleObjFn;
      FOnAfterStep, FOnAfterEpoch, FOnStart: TNotifyEvent;
      FRunning, FShouldQuit: boolean;
      FTrainingAccuracy, FValidationAccuracy, FTestAccuracy: TNeuralFloat;
      {$IFDEF OpenCL}
      FPlatformId: cl_platform_id;
      FDeviceId: cl_device_id;
      {$ENDIF}
      FProcs: TNeuralThreadList;
      procedure CheckLearningRate(iEpochCount: integer);
    public
      constructor Create();
      destructor Destroy(); override;
      procedure WaitUntilFinished;
      {$IFDEF OpenCL}
      procedure DisableOpenCL();
      procedure EnableOpenCL(platform_id: cl_platform_id; device_id: cl_device_id);
      {$ENDIF}

      property AvgWeightEpochCount: integer read FAvgWeightEpochCount write FAvgWeightEpochCount;
      property AvgNN: TNNet read FAvgWeight;
      property ClipDelta: single read FClipDelta write FClipDelta;
      property CurrentEpoch: integer read FCurrentEpoch;
      property CurrentStep: integer read FCurrentStep;
      property CurrentLearningRate: single read FCurrentLearningRate;
      property CustomLearningRateScheduleFn: TCustomLearningRateScheduleFn read FCustomLearningRateScheduleFn write FCustomLearningRateScheduleFn;
      property CustomLearningRateScheduleObjFn: TCustomLearningRateScheduleObjFn read FCustomLearningRateScheduleObjFn write FCustomLearningRateScheduleObjFn;
      property CyclicalLearningRateLen: integer read FCyclicalLearningRateLen write FCyclicalLearningRateLen;
      property FileNameBase: string read FFileNameBase write FFileNameBase;
      property Inertia: single read FInertia write FInertia;
      property InitialEpoch: integer read FInitialEpoch write FInitialEpoch;
      property InitialLearningRate: single read FInitialLearningRate write FInitialLearningRate;
      property LearningRateDecay: single read FLearningRateDecay write FLearningRateDecay;
      property L2Decay: single read FL2Decay write FL2Decay;
      property MaxThreadNum: integer read FMaxThreadNum write FMaxThreadNum;
      property Momentum: single read FInertia write FInertia;
      property MultipleSamplesAtValidation: boolean read FMultipleSamplesAtValidation write FMultipleSamplesAtValidation;
      property NN: TNNet read FNN;
      property OnAfterStep: TNotifyEvent read FOnAfterStep write FOnAfterStep;
      property OnAfterEpoch: TNotifyEvent read FOnAfterEpoch write FOnAfterEpoch;
      property OnStart: TNotifyEvent read FOnStart write FOnStart;
      property StaircaseEpochs: integer read FStaircaseEpochs write FStaircaseEpochs;
      property TargetAccuracy: single read FTargetAccuracy write FTargetAccuracy;
      property ValidationAccuracy: TNeuralFloat read FValidationAccuracy;
      property Verbose: boolean read FVerbose write FVerbose;
      property TestAccuracy: TNeuralFloat read FTestAccuracy;
      property ThreadNN: TNNetDataParallelism read FThreadNN;
      property TrainingAccuracy: TNeuralFloat read FTrainingAccuracy;
      property Running: boolean read FRunning;
      property ShouldQuit: boolean read FShouldQuit write FShouldQuit;
  end;

  TNNetDataAugmentationFn = procedure(pInput: TNNetVolume; ThreadId: integer) of object;
  TNNetLossFn = function(ExpectedOutput, FoundOutput: TNNetVolume; ThreadId: integer): TNeuralFloat of object;
  TNNetInferHitFn = function(A, B: TNNetVolume; ThreadId: integer): boolean;
  TNNetGetPairFn = function(Idx: integer; ThreadId: integer): TNNetVolumePair of object;
  TNNetGet2VolumesProc = procedure(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume) of object;

  /// Fitting algorithm with data (pairs) loading
  TNeuralDataLoadingFit = class(TNeuralFitBase)
    protected
      FDataAugmentationFn: TNNetDataAugmentationFn;
      FInferHitFn: TNNetInferHitFn;
      FLossFn: TNNetLossFn;
      FTestSize: integer;

      FLocalTestPair: TNNetGetPairFn;
      FLocalTestProc: TNNetGet2VolumesProc;
      FGetTrainingPair, FGetValidationPair, FGetTestPair: TNNetGetPairFn;
      FGetTrainingProc, FGetValidationProc, FGetTestProc: TNNetGet2VolumesProc;
    public
      constructor Create();
      procedure FitLoading(pNN: TNNet;
        TrainingCnt, ValidationCnt, TestCnt, pBatchSize, Epochs: integer;
        pGetTrainingPair, pGetValidationPair, pGetTestPair: TNNetGetPairFn); overload;
      procedure FitLoading(pNN: TNNet;
        TrainingCnt, ValidationCnt, TestCnt, pBatchSize, Epochs: integer;
        pGetTrainingProc, pGetValidationProc, pGetTestProc: TNNetGet2VolumesProc); overload;

      procedure EnableMonopolarHitComparison();
      procedure EnableBipolarHitComparison();
      procedure EnableBipolar99HitComparison();
      procedure EnableClassComparison();

      // On most cases, you should never call the following methods directly
      procedure RunNNThread(index, threadnum: integer);
      procedure TestNNThread(index, threadnum: integer);

      procedure AllocateMemory(pNN: TNNet;
        pBatchSize: integer;
        pGetTrainingPair, pGetValidationPair, pGetTestPair: TNNetGetPairFn); overload;
      procedure AllocateMemory(pNN: TNNet;
        pBatchSize: integer;
        pGetTrainingProc, pGetValidationProc, pGetTestProc: TNNetGet2VolumesProc); overload;
      procedure FreeMemory();
      procedure RunTrainingBatch();
      procedure RunValidationBatch(ValidationSize: integer);
      procedure RunTestBatch(TestSize: integer);

      property DataAugmentationFn: TNNetDataAugmentationFn read FDataAugmentationFn write FDataAugmentationFn;
      property InferHitFn: TNNetInferHitFn read FInferHitFn write FInferHitFn;
      property LossFn: TNNetLossFn read FLossFn write FLossFn;
  end;

  /// Generic Neural Network Fitting Algorithm
  TNeuralFit = class(TNeuralDataLoadingFit)
    protected
      FTrainingVolumes, FValidationVolumes, FTestVolumes: TNNetVolumePairList;
      FWorkingVolumes: TNNetVolumePairList;

      function FitTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
      function FitValidationPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
      function FitTestPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
    public
      constructor Create();
      destructor Destroy(); override;

      procedure Fit(pNN: TNNet;
        pTrainingVolumes, pValidationVolumes, pTestVolumes: TNNetVolumePairList;
        pBatchSize, Epochs: integer);
  end;

  /// Image Classification Fitting Algorithm
  TNeuralImageFit = class(TNeuralFitBase)
    protected
      FNumClasses: integer;
      FImgVolumes, FImgValidationVolumes, FImgTestVolumes: TNNetVolumeList;
      FMaxCropSize: integer;
      FHasImgCrop: boolean;
      FHasMakeGray: boolean;
      FHasFlipX: boolean;
      FHasFlipY: boolean;
      FIsSoftmax: boolean;
      FColorEncoding: integer;
      FWorkingVolumes: TNNetVolumeList;
    public
      constructor Create();
      destructor Destroy(); override;

      procedure Fit(pNN: TNNet;
        pImgVolumes, pImgValidationVolumes, pImgTestVolumes: TNNetVolumeList;
        pNumClasses, pBatchSize, Epochs: integer);
      procedure RunNNThread(index, threadnum: integer);
      procedure TestNNThread(index, threadnum: integer);
      procedure ClassifyImage(pNN: TNNet; pImgInput, pOutput: TNNetVolume);

      property HasImgCrop: boolean read FHasImgCrop write FHasImgCrop;
      property HasMakeGray: boolean read FHasMakeGray write FHasMakeGray;
      property HasFlipX: boolean read FHasFlipX write FHasFlipX;
      property HasFlipY: boolean read FHasFlipY write FHasFlipY;
      property MaxCropSize: integer read FMaxCropSize write FMaxCropSize;
  end;


  function MonopolarCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
  function BipolarCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
  function BipolarCompare99(A, B: TNNetVolume; ThreadId: integer): boolean;
  function ClassCompare(A, B: TNNetVolume; ThreadId: integer): boolean;

implementation
uses
  math;

function MonopolarCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
var
  Pos: integer;
  ACount, BCount: integer;
begin
  ACount := A.Size;
  BCount := B.Size;
  Result := (ACount>0) and (ACount = BCount);
  Pos := 0;
  while Result and (Pos < ACount) do
  begin
    Result := Result and (
      ( (A.FData[Pos]>0.5) and (B.FData[Pos]>0.5) ) or
      ( (A.FData[Pos]<0.5) and (B.FData[Pos]<0.5) )
    );
    Inc(Pos);
  end;
end;

function BipolarCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
var
  Pos: integer;
  ACount, BCount: integer;
begin
  ACount := A.Size;
  BCount := B.Size;
  Result := (ACount>0) and (ACount = BCount);
  Pos := 0;
  while Result and (Pos < ACount) do
  begin
    Result := Result and (
      ( (A.FData[Pos]>0) and (B.FData[Pos]>0) ) or
      ( (A.FData[Pos]<0) and (B.FData[Pos]<0) )
    );
    Inc(Pos);
  end;
end;

function BipolarCompare99(A, B: TNNetVolume; ThreadId: integer): boolean;
var
  Pos, Hits: integer;
  ACount: integer;
begin
  ACount := Min(A.Size, B.Size);
  Pos := 0;
  Hits := 0;
  while (Pos < ACount) do
  begin
    if (
      ( (A.FData[Pos]>0) and (B.FData[Pos]>0) ) or
      ( (A.FData[Pos]<0) and (B.FData[Pos]<0) )
    ) then Inc(Hits);
    Inc(Pos);
  end;
  Result := (Hits > Round(ACount*0.99));
end;

function ClassCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
begin
  Result := (A.GetClass() = B.GetClass());
end;
{ TNeuralDataLoadingFit }

constructor TNeuralDataLoadingFit.Create();
begin
  inherited Create;
  FGetTrainingPair := nil;
  FGetValidationPair := nil;
  FGetTestPair := nil;
  FLossFn := nil;
  FTestSize := 0;
end;

procedure TNeuralDataLoadingFit.FitLoading(pNN: TNNet; TrainingCnt,
  ValidationCnt, TestCnt, pBatchSize, Epochs: integer;
  pGetTrainingPair, pGetValidationPair,
  pGetTestPair: TNNetGetPairFn);
var
  I: integer;
  startTime, totalTimeSeconds: double;
  globalStartTime: double;
  fileName, FileNameCSV: string;
  TrainingError, TrainingLoss: TNeuralFloat;
  ValidationError, ValidationLoss, ValidationRate: TNeuralFloat;
  TestError, TestLoss, TestRate: TNeuralFloat;
  CSVFile: TextFile;
  CurrentAccuracy, AccuracyWithInertia: TNeuralFloat;
  ValidationRecord: TNeuralFloat;
begin
  AllocateMemory(pNN, pBatchSize, pGetTrainingPair,
    pGetValidationPair, pGetTestPair);
  FMaxEpochs := Epochs;
  AccuracyWithInertia := 0;
  TrainingError := 0;
  TrainingLoss := 0;
  ValidationRate := 0;
  ValidationLoss := 0;
  ValidationError := 0;
  TestRate := 0;
  TestLoss := 0;
  TestError := 0;
  ValidationRecord := 0;
  totalTimeSeconds := 0;

  FMessageProc('File name is: ' + FFileNameBase);
  FileNameCSV := FFileNameBase + '.csv';
  FileName := FFileNameBase + '.nn';

  AssignFile(CSVFile, FileNameCSV);
  ReWrite(CSVFile);
  WriteLn(CSVFile, 'epoch,training accuracy,training loss,training error,validation accuracy,validation loss,validation error,learning rate,time,test accuracy,test loss,test error');

  if FVerbose then
  begin
    WriteLn('Learning rate:',FCurrentLearningRate:8:6,
      ' L2 decay:', FL2Decay:8:6,
      ' Inertia:', FInertia:8:6,
      ' Batch size:', FBatchSize,
      ' Step size:', FStepSize,
      ' Staircase ephocs:',FStaircaseEpochs);
    if TrainingCnt > 0 then WriteLn('Training volumes:', TrainingCnt);
    if ValidationCnt > 0 then WriteLn('Validation volumes: ', ValidationCnt);
    if TestCnt > 0 then WriteLn('Test volumes: ', TestCnt);
  end;

  if FVerbose then
  begin
    MessageProc('Computing...');
  end;
  if Assigned(FOnStart) then FOnStart(Self);
  globalStartTime := Now();
  while ( (FMaxEpochs > FCurrentEpoch) and Not(FShouldQuit) ) do
  begin
    FGlobalErrorSum := 0;
    startTime := Now();
    CheckLearningRate(FCurrentEpoch);
    for I := 1 to (TrainingCnt div FStepSize) {$IFDEF MakeQuick}div 10{$ENDIF} do
    begin
      if FShouldQuit then break;
      RunTrainingBatch();

      FGlobalTotal := (FGlobalHit + FGlobalMiss);
      if (FGlobalTotal > 0) then
      begin
        TrainingError := FGlobalErrorSum / FGlobalTotal;
        TrainingLoss  := FGlobalTotalLoss / FGlobalTotal;
        CurrentAccuracy := (FGlobalHit*100) div FGlobalTotal;
        if (FCurrentEpoch = FInitialEpoch) and (I = 1) then
        begin
          AccuracyWithInertia := CurrentAccuracy;
        end else if (FStepSize < 100) then
        begin
          AccuracyWithInertia := AccuracyWithInertia*0.99 + CurrentAccuracy*0.01;
        end
        else if (FStepSize < 500) then
        begin
          AccuracyWithInertia := AccuracyWithInertia*0.9 + CurrentAccuracy*0.1;
        end
        else
        begin
          AccuracyWithInertia := CurrentAccuracy;
        end;
        FTrainingAccuracy := AccuracyWithInertia/100;
      end;

      if ( (FGlobalTotal > 0) and (I mod 10 = 0) ) then
      begin
        totalTimeSeconds := (Now() - startTime) * 24 * 60 * 60;
        if FVerbose then WriteLn
        (
          (FGlobalHit + FGlobalMiss)*I + FCurrentEpoch*TrainingCnt,
          ' Examples seen. Accuracy:', (FTrainingAccuracy):6:4,
          ' Error:', TrainingError:10:5,
          ' Loss:', TrainingLoss:7:5,
          ' Threads: ', FThreadNum,
          ' Forward time:', (FNN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Backward time:', (FNN.BackwardTime * 24 * 60 * 60):6:2,'s',
          ' Step time:', totalTimeSeconds:6:2,'s'
        );

        startTime := Now();
      end;
      if Assigned(FOnAfterStep) then FOnAfterStep(Self);
      Inc(FCurrentStep);
    end;

    Inc(FCurrentEpoch);

    if not Assigned(FAvgWeights) then
    begin
      FAvgWeight.CopyWeights(FNN);
      if FAvgWeightEpochCount > 1
      then FAvgWeights := TNNetDataParallelism.Create(FNN, FAvgWeightEpochCount)
      else FAvgWeights := TNNetDataParallelism.Create(FNN, 1)
    end;

    if Not(FShouldQuit) and (Assigned(FGetValidationPair) or Assigned(FGetValidationProc)) then
    begin
      if FAvgWeightEpochCount > 1 then
      begin
        FAvgWeights.ReplaceAtIdxAndUpdateWeightAvg(FCurrentEpoch mod FAvgWeightEpochCount, FNN, FAvgWeight);
      end
      else
      begin
        FAvgWeight.CopyWeights(FNN);
      end;

      if (ValidationCnt > 0) and Not(FShouldQuit) then
      begin
        RunValidationBatch(ValidationCnt);

        if FGlobalTotal > 0 then
        begin
          ValidationRate  := FGlobalHit / FGlobalTotal;
          ValidationLoss  := FGlobalTotalLoss / FGlobalTotal;
          ValidationError := FGlobalErrorSum / FGlobalTotal;
          FValidationAccuracy := ValidationRate;
        end;

        if (ValidationRate > ValidationRecord) then
        begin
          ValidationRecord := ValidationRate;
          FMessageProc('VALIDATION RECORD! Saving NN at '+fileName);
          FAvgWeight.SaveToFile(fileName);
        end;

        if (FGlobalTotal > 0) and (FVerbose) then
        begin
          WriteLn(
            'Epochs: ',FCurrentEpoch,
            ' Examples seen:', FCurrentEpoch * TrainingCnt,
            ' Validation Accuracy: ', ValidationRate:6:4,
            ' Validation Error: ', ValidationError:6:4,
            ' Validation Loss: ', ValidationLoss:6:4,
            ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
          );
        end;

        if (ValidationRate >= FTargetAccuracy) then
        begin
          FMessageProc('ValidationRate passed TargetAccuracy.');
          FShouldQuit := true;
          break;
        end;
      end;// Assigned(pGetValidationPair)

      if (FCurrentEpoch mod FThreadNN.Count = 0) and (FVerbose) then
      begin
        FThreadNN[0].DebugWeights();
      end;

      if (TestCnt > 0) and Not(FShouldQuit) and (Assigned(FGetTestPair) or Assigned(FGetTestProc)) then
      begin
        if ( (FCurrentEpoch mod 10 = 0) and (FCurrentEpoch > 0) ) then
        begin
          RunTestBatch(TestCnt);
          if FGlobalTotal > 0 then
          begin
            TestRate  := FGlobalHit / FGlobalTotal;
            TestLoss  := FGlobalTotalLoss / FGlobalTotal;
            TestError := FGlobalErrorSum / FGlobalTotal;
            FTestAccuracy := TestRate;
          end;
          if (FGlobalTotal > 0) and (FVerbose) then
          begin
            WriteLn(
              'Epochs: ', FCurrentEpoch,
              ' Examples seen:', FCurrentEpoch * TrainingCnt,
              ' Test Accuracy: ', TestRate:6:4,
              ' Test Error: ', TestError:6:4,
              ' Test Loss: ', TestLoss:6:4,
              ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
            );
          end;
        end;
        if (TestRate >= FTargetAccuracy) then
        begin
          FMessageProc('TestRate passed TargetAccuracy.');
          FShouldQuit := true;
          break;
        end;
      end;

      if ( (FCurrentEpoch mod 10 = 0) and (FCurrentEpoch > 0) ) then
      begin
        WriteLn
        (
          CSVFile,
          FCurrentEpoch,',',
          (AccuracyWithInertia/100):6:4,',',
          TrainingLoss:6:4,',',
          TrainingError:6:4,',',
          ValidationRate:6:4,',',
          ValidationLoss:6:4,',',
          ValidationError:6:4,',',
          FCurrentLearningRate:9:7,',',
          Round( (Now() - globalStartTime) * 24 * 60 * 60),',',
          TestRate:6:4,',',
          TestLoss:6:4,',',
          TestError:6:4
        );
      end
      else
      begin
        WriteLn
        (
          CSVFile,
          FCurrentEpoch,',',
          (AccuracyWithInertia/100):6:4,',',
          TrainingLoss:6:4,',',
          TrainingError:6:4,',',
          ValidationRate:6:4,',',
          ValidationLoss:6:4,',',
          ValidationError:6:4,',',
          FCurrentLearningRate:9:7,',',
          Round( (Now() - globalStartTime) * 24 * 60 * 60),',,,'
        );
      end;

      CloseFile(CSVFile);
      AssignFile(CSVFile, FileNameCSV);
      Append(CSVFile);

      MessageProc(
        'Epoch time: ' + FloatToStrF( totalTimeSeconds*(TrainingCnt/(FStepSize*10))/60,ffGeneral,1,4)+' minutes.' +
        ' '+IntToStr(Epochs)+' epochs: ' + FloatToStrF( Epochs*totalTimeSeconds*(TrainingCnt/(FStepSize*10))/3600,ffGeneral,1,4)+' hours.');

      MessageProc(
        'Epochs: '+IntToStr(FCurrentEpoch)+
        '. Working time: '+FloatToStrF(Round((Now() - globalStartTime)*2400)/100,ffGeneral,4,2)+' hours.');
    end;
    if Assigned(FOnAfterEpoch) then FOnAfterEpoch(Self);
  end;

  CloseFile(CSVFile);
  FMessageProc('Finished.');
  FreeMemory();
end;

procedure TNeuralDataLoadingFit.FitLoading(pNN: TNNet; TrainingCnt,
  ValidationCnt, TestCnt, pBatchSize, Epochs: integer; pGetTrainingProc,
  pGetValidationProc, pGetTestProc: TNNetGet2VolumesProc);
var
  vGetTrainingPair, vGetValidationPair, vGetTestPair: TNNetGetPairFn;
begin
  vGetTrainingPair := nil;
  vGetValidationPair := nil;
  vGetTestPair := nil;
  FGetTrainingProc := pGetTrainingProc;
  FGetValidationProc := pGetValidationProc;
  FGetTestProc := pGetTestProc;

  FitLoading(pNN, TrainingCnt, ValidationCnt, TestCnt, pBatchSize, Epochs,
    vGetTrainingPair, vGetValidationPair, vGetTestPair);

  FGetTrainingProc := nil;
  FGetValidationProc := nil;
  FGetTestProc := nil;
end;

function TNeuralFit.FitTrainingPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
var
  ElementIdx: integer;
begin
  ElementIdx := Random(FTrainingVolumes.Count);
  FitTrainingPair := FTrainingVolumes[ElementIdx];
end;

function TNeuralFit.FitValidationPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
begin
  FitValidationPair := FValidationVolumes[Idx];
end;

function TNeuralFit.FitTestPair(Idx: integer; ThreadId: integer): TNNetVolumePair;
begin
  FitTestPair := FTestVolumes[Idx];
end;

constructor TNeuralFit.Create();
begin
  inherited Create();
  FInferHitFn := nil;
  FDataAugmentationFn := nil;
  FLossFn := nil;
  {$IFDEF FPC}
  FGetTrainingPair := @Self.FitTrainingPair;
  FGetValidationPair := @Self.FitTestPair;
  FGetTestPair := @Self.FitTestPair;
  {$ELSE}
  FGetTrainingPair := Self.FitTrainingPair;
  FGetValidationPair := Self.FitTestPair;
  FGetTestPair := Self.FitTestPair;
  {$ENDIF}
end;

destructor TNeuralFit.Destroy();
begin
  inherited Destroy();
end;

procedure TNeuralFit.Fit(pNN: TNNet; pTrainingVolumes, pValidationVolumes,
  pTestVolumes: TNNetVolumePairList; pBatchSize, Epochs: integer);
var
  TrainingCnt, ValidationCnt, TestCnt: integer;
begin
  TrainingCnt := 0;
  ValidationCnt := 0;
  TestCnt := 0;

  FTrainingVolumes := pTrainingVolumes;
  FValidationVolumes := pValidationVolumes;
  FTestVolumes := pTestVolumes;

  if Assigned(FTrainingVolumes) then TrainingCnt := FTrainingVolumes.Count;
  if Assigned(FValidationVolumes) then ValidationCnt := FValidationVolumes.Count;
  if Assigned(FTestVolumes) then TestCnt := FTestVolumes.Count;

  {$IFDEF FPC}
  FitLoading(pNN, TrainingCnt, ValidationCnt, TestCnt,
    pBatchSize, Epochs, @Self.FitTrainingPair,
    @Self.FitValidationPair, @Self.FitTestPair);
  {$ELSE}
  FitLoading(pNN, TrainingCnt, ValidationCnt, TestCnt,
    pBatchSize, Epochs, Self.FitTrainingPair,
    Self.FitValidationPair, Self.FitTestPair);
  {$ENDIF}
end;

procedure TNeuralDataLoadingFit.RunNNThread(index, threadnum: integer);
var
  BlockSize, BlockSizeRest: integer;
  LocalNN: TNNet;
  vInput: TNNetVolume;
  pOutput, vOutput: TNNetVolume;
  I: integer;
  CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
  LocalTrainingPair: TNNetVolumePair;
  {$IFDEF DEBUG}
  InitialWeightSum, FinalWeightSum: TNeuralFloat;
  {$ENDIF}
begin
  vInput  := TNNetVolume.Create();
  pOutput := TNNetVolume.Create();
  vOutput := TNNetVolume.Create();

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := FBatchSize div FThreadNum;
  BlockSizeRest := FBatchSize mod FThreadNum;

  if (Index < BlockSizeRest) then Inc(BlockSize);

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(FNN);
  LocalNN.ClearTime();
  LocalNN.ClearDeltas();
  LocalNN.EnableDropouts(true);
  {$IFDEF DEBUG}
  InitialWeightSum := LocalNN.GetWeightSum();
  {$ENDIF}
  for I := 1 to BlockSize do
  begin
    if FShouldQuit then Break;

    if Assigned(FGetTrainingPair) then
    begin
      LocalTrainingPair := FGetTrainingPair(I, Index);
      vInput.Copy( LocalTrainingPair.I );
      vOutput.Copy( LocalTrainingPair.O );
    end
    else
    begin
      FGetTrainingProc(I, Index, vInput, vOutput)
    end;

    if Assigned(FDataAugmentationFn)
      then FDataAugmentationFn(vInput, Index);

    LocalNN.Compute( vInput );
    LocalNN.GetOutput( pOutput );
    LocalNN.Backpropagate( vOutput );

    LocalErrorSum := LocalErrorSum + vOutput.SumDiff( pOutput );

    CurrentLoss := 0;
    if Assigned(FLossFn) then
    begin
      CurrentLoss := FLossFn(vOutput, pOutput, Index);
    end;

    LocalTotalLoss := LocalTotalLoss + CurrentLoss;

    if Assigned(FInferHitFn) then
    begin
      if FInferHitFn(vOutput, pOutput, Index) then
      begin
        Inc(LocalHit);
      end
      else
      begin
        Inc(LocalMiss);
      end;
    end;
  end; // of for

  if (Index and 1 = 0) and Not(FShouldQuit) then
  begin
    if Index + 1 < FThreadNum then
    begin
      while (FFinishedThread.FData[Index + 1] = 0) and Not(FShouldQuit) do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 1]);
      {$IFDEF FPC}
      FFinishedThread.FData[Index] += FFinishedThread.FData[Index + 1];
      {$ELSE}
      FFinishedThread.FData[Index] := FFinishedThread.FData[Index] +
        FFinishedThread.FData[Index + 1];
      {$ENDIF}
    end;
  end;
  FFinishedThread.FData[Index] := FFinishedThread.FData[Index] + 1;
  if (Index and 3 = 0) and Not(FShouldQuit) then
  begin
    if Index + 2 < FThreadNum then
    begin
      while (FFinishedThread.FData[Index + 2] = 0) and Not(FShouldQuit) do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 2]);
      {$IFDEF FPC}
      FFinishedThread.FData[Index] += FFinishedThread.FData[Index + 2];
      {$ELSE}
      FFinishedThread.FData[Index] := FFinishedThread.FData[Index] +
        FFinishedThread.FData[Index + 2];
      {$ENDIF}
    end;
  end;
  {$IFDEF DEBUG}
  FinalWeightSum := LocalNN.GetWeightSum();
  if InitialWeightSum <> FinalWeightSum then
  begin
    FErrorProc('Weights changed on thread:'+FloatToStr(FinalWeightSum - InitialWeightSum));
  end;
  {$ENDIF}
  {$IFDEF HASTHREADS}EnterCriticalSection(FCritSec);{$ENDIF}
  FGlobalHit       := FGlobalHit + LocalHit;
  FGlobalMiss      := FGlobalMiss + LocalMiss;
  FGlobalTotalLoss := FGlobalTotalLoss + LocalTotalLoss;
  FGlobalErrorSum  := FGlobalErrorSum + LocalErrorSum;

  FNN.ForwardTime := FNN.ForwardTime + LocalNN.ForwardTime;
  FNN.BackwardTime := FNN.BackwardTime + LocalNN.BackwardTime;
  {$IFDEF Debug}
  if Index and 3 = 0 then FNN.SumDeltas(LocalNN);
  {$ELSE}
  if Index and 3 = 0 then FNN.SumDeltasNoChecks(LocalNN);
  {$ENDIF}
  {$IFDEF HASTHREADS}LeaveCriticalSection(FCritSec);{$ENDIF}
  vInput.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TNeuralDataLoadingFit.TestNNThread(index, threadnum: integer);
var
  BlockSize: integer;
  LocalNN: TNNet;
  vInput: TNNetVolume;
  pOutput, vOutput, LocalFrequency: TNNetVolume;
  I: integer;
  StartPos, FinishPos: integer;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
  LocalTestPair: TNNetVolumePair;
begin
  vInput  := TNNetVolume.Create();
  pOutput := TNNetVolume.Create();
  vOutput := TNNetVolume.Create();
  LocalFrequency := TNNetVolume.Create();

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := (FTestSize div FThreadNum) {$IFDEF MakeQuick}div 10{$ENDIF};
  StartPos  := BlockSize * index;
  FinishPos := BlockSize * (index + 1) - 1;

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(FAvgWeight);
  LocalNN.EnableDropouts(false);
  for I := StartPos to FinishPos - 1 do
  begin
    if FShouldQuit then Break;

    if Assigned(FLocalTestPair) then
    begin
      LocalTestPair := FLocalTestPair(I, Index);
      vInput.Copy(LocalTestPair.I);
      vOutput.Copy(LocalTestPair.O);
    end
    else
    begin
      FLocalTestProc(I, Index, vInput, vOutput);
    end;

    LocalNN.Compute( vInput );
    LocalNN.GetOutput( pOutput );

    LocalErrorSum := LocalErrorSum + vOutput.SumDiff( pOutput );

    if Assigned(FInferHitFn) then
    begin
      if FInferHitFn(vOutput, pOutput, Index) then
      begin
        Inc(LocalHit);
      end
      else
      begin
        Inc(LocalMiss);
      end;
    end;

    if Assigned(FLossFn)
      then LocalTotalLoss := LocalTotalLoss + FLossFn(vOutput, pOutput, Index);
  end; // of for
  LocalNN.EnableDropouts(true);

  {$IFDEF HASTHREADS}EnterCriticalSection(FCritSec);{$ENDIF}
  FGlobalHit       := FGlobalHit + LocalHit;
  FGlobalMiss      := FGlobalMiss + LocalMiss;
  FGlobalTotalLoss := FGlobalTotalLoss + LocalTotalLoss;
  FGlobalErrorSum  := FGlobalErrorSum + LocalErrorSum;
  {$IFDEF HASTHREADS}LeaveCriticalSection(FCritSec);{$ENDIF}

  LocalFrequency.Free;
  vInput.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TNeuralDataLoadingFit.AllocateMemory(pNN: TNNet;
  pBatchSize: integer; pGetTrainingPair,
  pGetValidationPair, pGetTestPair: TNNetGetPairFn);
begin
  FRunning := true;
  FShouldQuit := false;
  FNN := pNN;
  FGetTrainingPair := pGetTrainingPair;
  FGetValidationPair := pGetValidationPair;
  FGetTestPair := pGetTestPair;
  {$IFNDEF HASTHREADS}
  FMaxThreadNum := 1;
  {$ENDIF}
  FCurrentEpoch := FInitialEpoch;
  FCurrentLearningRate := FInitialLearningRate;
  FThreadNum := Min(FMaxThreadNum, FDefaultThreadCount);
  FBatchSize := pBatchSize;
  if FBatchSize >= FThreadNum then
  begin
    FBatchSize := (FBatchSize div FThreadNum) * FThreadNum;
  end
  else if FBatchSize > 0 then
  begin
    FThreadNum := FBatchSize;
  end
  else
  begin
    FErrorProc('Batch size has to be bigger than zero.');
    FRunning := false;
    exit;
  end;

  if Not(Assigned(FGetTrainingPair)) and Not(Assigned(FGetTrainingProc)) then
  begin
    FErrorProc('Training function hasn''t been defined.');
    exit;
  end;
  FProcs := TNeuralThreadList.Create(FThreadNum);
  FStepSize := FBatchSize;
  if Assigned(FThreadNN) then FThreadNN.Free;
  FThreadNN := TNNetDataParallelism.Create(FNN, FThreadNum);
  {$IFDEF OpenCL}
  if (FPlatformId <> nil) and (FDeviceId <> nil) then
    FThreadNN.EnableOpenCL(FPlatformId, FDeviceId);
  {$ENDIF}
  FFinishedThread.Resize(1, 1, FThreadNum);
  FAvgWeights := nil;
  FAvgWeight := FNN.Clone();
  FThreadNN.SetLearningRate(FCurrentLearningRate, FInertia);
  FThreadNN.SetBatchUpdate(true);
  // in batch update mode, threaded NN should not apply L2 (L2 is applied in the main thread).
  FThreadNN.SetL2Decay(0);

  FNN.SetL2Decay(FL2Decay);
  FNN.SetLearningRate(FCurrentLearningRate, FInertia);
end;

procedure TNeuralDataLoadingFit.AllocateMemory(pNN: TNNet; pBatchSize: integer;
  pGetTrainingProc, pGetValidationProc,
  pGetTestProc: TNNetGet2VolumesProc);
var
  vGetTrainingPair, vGetValidationPair, vGetTestPair: TNNetGetPairFn;
begin
  vGetTrainingPair := nil;
  vGetValidationPair := nil;
  vGetTestPair := nil;
  FGetTrainingProc := pGetTrainingProc;
  FGetValidationProc := pGetValidationProc;
  FGetTestProc := pGetTestProc;
  AllocateMemory(pNN, pBatchSize, vGetTrainingPair, vGetValidationPair, vGetTestPair);
end;

procedure TNeuralDataLoadingFit.FreeMemory();
begin
  FProcs.Free;
  FGetTrainingPair := nil;
  FGetValidationPair := nil;
  FGetTestPair := nil;
  FreeAndNil(FAvgWeight);
  if Assigned(FAvgWeights) then FreeAndNil(FAvgWeights);
  FreeAndNil(FThreadNN);
  FRunning := false;
end;

procedure TNeuralDataLoadingFit.RunTrainingBatch();
var
  MaxDelta: TNeuralFloat;
begin
  FGlobalHit       := 0;
  FGlobalMiss      := 0;
  FGlobalTotalLoss := 0;
  FGlobalErrorSum  := 0;
  FFinishedThread.Fill(0);
  FNN.ClearTime();
  FNN.RefreshDropoutMask();
  {$IFDEF HASTHREADS}
  //ProcThreadPool.DoParallel(@RunNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
  FProcs.StartProc({$IFDEF FPC}@RunNNThread{$ELSE}RunNNThread{$ENDIF});
  {$ELSE}
  RunNNThread(0, 1);
  {$ENDIF}
  if FClipDelta > 0 then
  begin
    MaxDelta := FNN.ForceMaxAbsoluteDelta(FClipDelta);
  end
  else
  begin
    MaxDelta := FNN.NormalizeMaxAbsoluteDelta();
    if MaxDelta < 1 then
    begin
      MessageProc('Deltas have been multiplied by: '+FloatToStr(MaxDelta)+'.'+
      ' Max delta on layer: '+IntToStr(FNN.MaxDeltaLayer)+' - '+
      FNN.Layers[FNN.MaxDeltaLayer].ClassName+'.');
    end;
  end;
  FNN.UpdateWeights();
  if FL2Decay > 0.0 then FNN.ComputeL2Decay();
end;

procedure TNeuralDataLoadingFit.RunValidationBatch(ValidationSize: integer);
begin
  FGlobalHit       := 0;
  FGlobalMiss      := 0;
  FGlobalTotalLoss := 0;
  FGlobalErrorSum  := 0;
  FTestSize        := ValidationSize;
  FLocalTestPair   := FGetValidationPair;
  FLocalTestProc   := FGetValidationProc;
  FMessageProc('Starting Validation.');
  {$IFDEF HASTHREADS}
  //ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
  FProcs.StartProc({$IFDEF FPC}@TestNNThread{$ELSE}TestNNThread{$ENDIF});
  {$ELSE}
  TestNNThread(0, 1);
  {$ENDIF}
  FGlobalTotal := (FGlobalHit + FGlobalMiss);
end;

procedure TNeuralDataLoadingFit.RunTestBatch(TestSize: integer);
begin
  FGlobalHit       := 0;
  FGlobalMiss      := 0;
  FGlobalTotalLoss := 0;
  FGlobalErrorSum  := 0;
  FTestSize        := TestSize;
  FLocalTestPair   := FGetTestPair;
  FLocalTestProc   := FGetTestProc;
  FMessageProc('Starting Testing.');
  {$IFDEF HASTHREADS}
  //ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
  FProcs.StartProc({$IFDEF FPC}@TestNNThread{$ELSE}TestNNThread{$ENDIF});
  {$ELSE}
  TestNNThread(0, 1);
  {$ENDIF}
  FGlobalTotal := (FGlobalHit + FGlobalMiss);
end;

procedure TNeuralDataLoadingFit.EnableMonopolarHitComparison();
begin
  FInferHitFn := @MonopolarCompare;
end;

procedure TNeuralDataLoadingFit.EnableBipolarHitComparison();
begin
  FInferHitFn := @BipolarCompare;
end;

procedure TNeuralDataLoadingFit.EnableBipolar99HitComparison();
begin
  FInferHitFn := @BipolarCompare99;
end;

procedure TNeuralDataLoadingFit.EnableClassComparison();
begin
  FInferHitFn := @ClassCompare;
end;

{ TNeuralFitBase }

constructor TNeuralFitBase.Create();
begin
  inherited Create();
  {$IFDEF OpenCL}
  DisableOpenCL();
  {$ENDIF}
  FFinishedThread := TNNetVolume.Create();
  {$IFDEF HASTHREADS}
  FMaxThreadNum := NeuralDefaultThreadCount();
    //{$IFDEF OpenCL}
    // This optimization works very well on some systems but not all :-(
    // This optimization fails on google colab.
    //if FMaxThreadNum <=8 then
    //FMaxThreadNum := ProcThreadPool.MaxThreadCount * 2;
    //{$ENDIF}
  NeuralInitCriticalSection(FCritSec);
  {$ELSE}
  FMaxThreadNum := 1;
  {$ENDIF}
  FDefaultThreadCount := FMaxThreadNum;
  FDataAugmentation := true;
  FMultipleSamplesAtValidation := true;
  FVerbose := true;
  FStaircaseEpochs := 1;
  FStepSize := 1;
  FAvgWeightEpochCount := 10;
  FLearningRateDecay := 0.01;
  FInitialLearningRate := 0.001;
  FCyclicalLearningRateLen := 0; // not cyclical by default.
  FInitialEpoch := 0;
  fMinLearnRate := FInitialLearningRate * 0.01;
  FInertia := 0.9;
  FClipDelta := 0.0;
  FFileNameBase := 'autosave';
  FL2Decay := 0.0000001;
  FTargetAccuracy := 1;
  FCustomLearningRateScheduleFn := nil;
  FCustomLearningRateScheduleObjFn := nil;
  FOnAfterStep := nil;
  FOnAfterEpoch := nil;
  FOnStart := nil;
  FTrainingAccuracy := 0;
  FValidationAccuracy := 0;
  FTestAccuracy := 0;
  FRunning := false;
  FShouldQuit := false;
  FCurrentEpoch := 0;
  FCurrentStep := 0;
end;

destructor TNeuralFitBase.Destroy();
begin
  {$IFDEF HASTHREADS}
  NeuralDoneCriticalSection(FCritSec);
  {$ENDIF}
  FFinishedThread.Free;
  inherited Destroy();
end;

procedure TNeuralFitBase.WaitUntilFinished;
begin
  FShouldQuit := true;
  while FRunning do
  begin
    Sleep(1);
  end;
end;

{$IFDEF OpenCL}
procedure TNeuralFitBase.DisableOpenCL();
begin
  FPlatformId := nil;
  FDeviceId := nil;
end;

procedure TNeuralFitBase.EnableOpenCL(platform_id: cl_platform_id;
  device_id: cl_device_id);
begin
  FPlatformId := platform_id;
  FDeviceId := device_id;
end;
{$ENDIF}

procedure TNeuralFitBase.CheckLearningRate(iEpochCount: integer);
var
  iStairCount: integer;
  fNewLearningRate: single;
begin
  if FCyclicalLearningRateLen > 0 then
  begin
    iEpochCount := iEpochCount mod FCyclicalLearningRateLen;
  end;
  if (FLearningRateDecay >= 0.5) and (FLearningRateDecay <= 1) then
  begin
    // This is a hack for backward compatibility. Weight decays should be
    // expressed with 0.01 and not 0.99
    FLearningRateDecay := 1 - FLearningRateDecay;
    if FVerbose then
    WriteLn
    (
      'Learning rate decay set to:',FLearningRateDecay:7:5
    );
  end;
  if Assigned(FCustomLearningRateScheduleFn) then
  begin
    fNewLearningRate := FCustomLearningRateScheduleFn(iEpochCount);
  end
  else if Assigned(FCustomLearningRateScheduleObjFn) then
  begin
    fNewLearningRate := FCustomLearningRateScheduleObjFn(iEpochCount);
  end
  else if FStaircaseEpochs > 1 then
  begin
    iStairCount := iEpochCount div FStaircaseEpochs;
    fNewLearningRate := (fInitialLearningRate * power(1-FLearningRateDecay, iStairCount * FStaircaseEpochs));
  end
  else
  begin
    fNewLearningRate := (fInitialLearningRate * power(1-FLearningRateDecay,iEpochCount));
  end;
  if ( ( fNewLearningRate >= fMinLearnRate ) and (fNewLearningRate <> FCurrentLearningRate) ) then
  begin
    FCurrentLearningRate := fNewLearningRate;
    FThreadNN.SetLearningRate(FCurrentLearningRate, FInertia);
    FNN.SetLearningRate(FCurrentLearningRate, FInertia);
    FNN.ClearInertia();
    if FVerbose then
    WriteLn
    (
      'Learning rate set to:',FCurrentLearningRate:7:5
    );
  end;
end;

{ TNeuralImageFit }

constructor TNeuralImageFit.Create();
begin
  inherited Create();
  FMaxCropSize := 8;
  FHasImgCrop := false;
  FHasFlipX := true;
  FHasFlipY := false;
  FHasMakeGray := true;
  FIsSoftmax := true;
  FColorEncoding := 0;
  FMultipleSamplesAtValidation := true;
end;

destructor TNeuralImageFit.Destroy();
begin
  inherited Destroy;
end;

procedure TNeuralImageFit.Fit(pNN: TNNet;
  pImgVolumes, pImgValidationVolumes, pImgTestVolumes: TNNetVolumeList;
  pNumClasses, pBatchSize, Epochs: integer);
var
  I: integer;
  startTime, totalTimeSeconds: double;
  globalStartTime: double;
  fileName, FileNameCSV: string;
  TrainingError, TrainingLoss: TNeuralFloat;
  ValidationError, ValidationLoss, ValidationRate: TNeuralFloat;
  TestError, TestLoss, TestRate: TNeuralFloat;
  CSVFile: TextFile;
  CurrentAccuracy, AccuracyWithInertia: TNeuralFloat;
  MaxDelta: TNeuralFloat;
  ValidationRecord: TNeuralFloat;
begin
  FRunning := true;
  FShouldQuit := false;
  FNN := pNN;
  {$IFNDEF HASTHREADS}
  FMaxThreadNum := 1;
  {$ENDIF}
  FCurrentEpoch := FInitialEpoch;
  FCurrentStep := 0;
  FCurrentLearningRate := FInitialLearningRate;
  FImgVolumes := pImgVolumes;
  FImgValidationVolumes := pImgValidationVolumes;
  FImgTestVolumes := pImgTestVolumes;
  FThreadNum := Min(FMaxThreadNum, FDefaultThreadCount);
  FBatchSize := pBatchSize;
  FMaxEpochs := Epochs;
  if FBatchSize >= FThreadNum then
  begin
    FBatchSize := (FBatchSize div FThreadNum) * FThreadNum;
  end
  else if FBatchSize > 0 then
  begin
    FThreadNum := FBatchSize;
  end
  else
  begin
    FErrorProc('Batch size has to be bigger than zero.');
    FRunning := false;
    exit;
  end;
  FProcs := TNeuralThreadList.Create(FThreadNum);
  FStepSize := FBatchSize;
  if Assigned(FThreadNN) then FThreadNN.Free;
  FThreadNN := TNNetDataParallelism.Create(FNN, FThreadNum);
  {$IFDEF OpenCL}
  if (FPlatformId <> nil) and (FDeviceId <> nil) then
    FThreadNN.EnableOpenCL(FPlatformId, FDeviceId);
  {$ENDIF}
  FFinishedThread.Resize(1, 1, FThreadNum);
  FNumClasses := pNumClasses;
  AccuracyWithInertia := 100 / FNumClasses;
  TrainingError := 0;
  TrainingLoss := 0;
  ValidationRate := 0;
  ValidationLoss := 0;
  ValidationError := 0;
  TestRate := 0;
  TestLoss := 0;
  TestError := 0;
  ValidationRecord := 0;
  totalTimeSeconds := 0;

  FMessageProc('File name is: ' + FFileNameBase);

  FileNameCSV := FFileNameBase + '.csv';
  FileName := FFileNameBase + '.nn';

  AssignFile(CSVFile, FileNameCSV);
  ReWrite(CSVFile);
  WriteLn(CSVFile, 'epoch,training accuracy,training loss,training error,validation accuracy,validation loss,validation error,learning rate,time,test accuracy,test loss,test error');

  FAvgWeights := nil;
  FAvgWeight := FNN.Clone();

  if FVerbose then
  begin
    WriteLn('Learning rate:',FCurrentLearningRate:8:6,
      ' L2 decay:', FL2Decay:8:6,
      ' Inertia:', FInertia:8:6,
      ' Batch size:', FBatchSize,
      ' Step size:', FStepSize,
      ' Staircase ephocs:',FStaircaseEpochs);
    if Assigned(FImgVolumes) then WriteLn('Training images:', FImgVolumes.Count);
    if Assigned(FImgValidationVolumes) then WriteLn('Validation images:', FImgValidationVolumes.Count);
    if Assigned(FImgTestVolumes) then WriteLn('Test images:', FImgTestVolumes.Count);
  end;

  FThreadNN.SetLearningRate(FCurrentLearningRate, FInertia);
  FThreadNN.SetBatchUpdate(true);
  // in batch update mode, threaded NN should not apply L2 (L2 is applied in the main thread).
  FThreadNN.SetL2Decay(0);

  FNN.SetL2Decay(FL2Decay);
  FNN.SetLearningRate(FCurrentLearningRate, FInertia);

  if FVerbose then
  begin
    MessageProc('Computing...');
  end;
  if Assigned(FOnStart) then FOnStart(Self);
  globalStartTime := Now();
  while ( (FMaxEpochs > FCurrentEpoch) and Not(FShouldQuit) ) do
  begin
    FGlobalErrorSum := 0;
    startTime := Now();
    CheckLearningRate(FCurrentEpoch);
    for I := 1 to (FImgVolumes.Count div FStepSize) {$IFDEF MakeQuick}div 10{$ENDIF} do
    begin
      if (FShouldQuit) then break;
      FGlobalHit       := 0;
      FGlobalMiss      := 0;
      FGlobalTotalLoss := 0;
      FGlobalErrorSum  := 0;
      FFinishedThread.Fill(0);
      FNN.ClearTime();
      FNN.RefreshDropoutMask();
      {$IFDEF HASTHREADS}
      //ProcThreadPool.DoParallel(@RunNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
      FProcs.StartProc({$IFDEF FPC}@RunNNThread{$ELSE}RunNNThread{$ENDIF});
      {$ELSE}
      RunNNThread(0, 1);
      {$ENDIF}
      if FClipDelta > 0 then
      begin
        MaxDelta := FNN.ForceMaxAbsoluteDelta(FClipDelta);
      end
      else
      begin
        MaxDelta := FNN.NormalizeMaxAbsoluteDelta();
        if MaxDelta < 1 then
        begin
          MessageProc('Deltas have been multiplied by: '+FloatToStr(MaxDelta)+'.'+
          ' Max delta on layer: '+IntToStr(FNN.MaxDeltaLayer)+' - '+
          FNN.Layers[FNN.MaxDeltaLayer].ClassName+'.');
        end;
      end;
      FNN.UpdateWeights();
      if FL2Decay > 0.0 then FNN.ComputeL2Decay();

      FGlobalTotal := (FGlobalHit + FGlobalMiss);
      if (FGlobalTotal > 0) then
      begin
        TrainingError := FGlobalErrorSum / FGlobalTotal;
        TrainingLoss  := FGlobalTotalLoss / FGlobalTotal;
        CurrentAccuracy := (FGlobalHit*100) div FGlobalTotal;
        if (FCurrentEpoch = FInitialEpoch) and (I = 1) then
        begin
          AccuracyWithInertia := CurrentAccuracy;
        end else if (FStepSize < 100) then
        begin
          AccuracyWithInertia := AccuracyWithInertia*0.99 + CurrentAccuracy*0.01;
        end
        else if (FStepSize < 500) then
        begin
          AccuracyWithInertia := AccuracyWithInertia*0.9 + CurrentAccuracy*0.1;
        end
        else
        begin
          AccuracyWithInertia := CurrentAccuracy;
        end;
        FTrainingAccuracy := AccuracyWithInertia/100;
      end;

      if ( (FGlobalTotal > 0) and (I mod 10 = 0) ) then
      begin
        totalTimeSeconds := (Now() - startTime) * 24 * 60 * 60;
        if FVerbose then WriteLn
        (
          (FGlobalHit + FGlobalMiss)*I + FCurrentEpoch*FImgVolumes.Count,
          ' Examples seen. Accuracy:', (TrainingAccuracy):6:4,
          ' Error:', TrainingError:10:5,
          ' Loss:', TrainingLoss:7:5,
          ' Threads: ', FThreadNum,
          ' Forward time:', (FNN.ForwardTime * 24 * 60 * 60):6:2,'s',
          ' Backward time:', (FNN.BackwardTime * 24 * 60 * 60):6:2,'s',
          ' Step time:', totalTimeSeconds:6:2,'s'
        );

        startTime := Now();
      end;
      if Assigned(FOnAfterStep) then FOnAfterStep(Self);
      Inc(FCurrentStep);
    end;

    Inc(FCurrentEpoch);

    if not Assigned(FAvgWeights) then
    begin
      FAvgWeight.CopyWeights(FNN);
      if FAvgWeightEpochCount > 1
      then FAvgWeights := TNNetDataParallelism.Create(FNN, FAvgWeightEpochCount)
      else FAvgWeights := TNNetDataParallelism.Create(FNN, 1)
    end;

    if Not(FShouldQuit) then
    begin
      if FAvgWeightEpochCount > 1 then
      begin
        FAvgWeights.ReplaceAtIdxAndUpdateWeightAvg(FCurrentEpoch mod FAvgWeightEpochCount, FNN, FAvgWeight);
      end
      else
      begin
        FAvgWeight.CopyWeights(FNN);
      end;

      if Assigned(FImgValidationVolumes) and Not(FShouldQuit) then
      begin
        FWorkingVolumes := FImgValidationVolumes;
        FGlobalHit       := 0;
        FGlobalMiss      := 0;
        FGlobalTotalLoss := 0;
        FGlobalErrorSum  := 0;
        FMessageProc('Starting Validation.');
        {$IFDEF HASTHREADS}
        //ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
        FProcs.StartProc({$IFDEF FPC}@TestNNThread{$ELSE}TestNNThread{$ENDIF});
        {$ELSE}
        TestNNThread(0, 1);
        {$ENDIF}

        FGlobalTotal := (FGlobalHit + FGlobalMiss);
        if (FGlobalTotal > 0) then
        begin
          ValidationRate  := FGlobalHit / FGlobalTotal;
          ValidationLoss  := FGlobalTotalLoss / FGlobalTotal;
          ValidationError := FGlobalErrorSum / FGlobalTotal;
          FValidationAccuracy := ValidationRate;
        end;

        if (ValidationRate > ValidationRecord) then
        begin
          ValidationRecord := ValidationRate;
          FMessageProc('VALIDATION RECORD! Saving NN at '+fileName);
          FAvgWeight.SaveToFile(fileName);
        end;

        if (FGlobalTotal > 0) and (FVerbose) then
        begin
          WriteLn(
            'Epochs: ',FCurrentEpoch,
            ' Examples seen:', FCurrentEpoch * FImgVolumes.Count,
            ' Validation Accuracy: ', ValidationRate:6:4,
            ' Validation Error: ', ValidationError:6:4,
            ' Validation Loss: ', ValidationLoss:6:4,
            ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
          );
        end;

        if (ValidationRate >= FTargetAccuracy) then
        begin
          FMessageProc('ValidationRate passed TargetAccuracy.');
          FShouldQuit := true;
          break;
        end;
      end;// Assigned(FImgValidationVolumes)

      if (FCurrentEpoch mod FThreadNN.Count = 0) and (FVerbose) then
      begin
        FThreadNN[0].DebugWeights();
      end;

      if Assigned(FImgTestVolumes) and Not(FShouldQuit) then
      begin
        if ( (FCurrentEpoch mod 10 = 0) and (FCurrentEpoch > 0) ) then
        begin
          FWorkingVolumes := FImgTestVolumes;
          FGlobalHit       := 0;
          FGlobalMiss      := 0;
          FGlobalTotalLoss := 0;
          FGlobalErrorSum  := 0;
          FMessageProc('Starting Testing.');
          {$IFDEF HASTHREADS}
          //ProcThreadPool.DoParallel(@TestNNThread, 0, FThreadNN.Count-1, Nil, FThreadNN.Count);
          FProcs.StartProc({$IFDEF FPC}@TestNNThread{$ELSE}TestNNThread{$ENDIF});
          {$ELSE}
          TestNNThread(0, 1);
          {$ENDIF}

          FGlobalTotal := (FGlobalHit + FGlobalMiss);
          if (FGlobalTotal > 0) then
          begin
            TestRate  := FGlobalHit / FGlobalTotal;
            TestLoss  := FGlobalTotalLoss / FGlobalTotal;
            TestError := FGlobalErrorSum / FGlobalTotal;
            FTestAccuracy := TestRate;
          end;

          if (FGlobalTotal > 0) and (FVerbose) then
          begin
            WriteLn(
              'Epochs: ',FCurrentEpoch,
              ' Examples seen:', FCurrentEpoch * FImgVolumes.Count,
              ' Test Accuracy: ', TestRate:6:4,
              ' Test Error: ', TestError:6:4,
              ' Test Loss: ', TestLoss:6:4,
              ' Total time: ', (((Now() - globalStartTime)) * 24 * 60): 6: 2, 'min'
            );
          end;
        end;
        if (TestRate >= FTargetAccuracy) then
        begin
          FMessageProc('TestRate passed TargetAccuracy.');
          FShouldQuit := true;
          break;
        end;
      end;

      if ( (FCurrentEpoch mod 10 = 0) and (FCurrentEpoch > 0) ) then
      begin
        WriteLn
        (
          CSVFile,
          FCurrentEpoch,',',
          (AccuracyWithInertia/100):6:4,',',
          TrainingLoss:6:4,',',
          TrainingError:6:4,',',
          ValidationRate:6:4,',',
          ValidationLoss:6:4,',',
          ValidationError:6:4,',',
          FCurrentLearningRate:9:7,',',
          Round( (Now() - globalStartTime) * 24 * 60 * 60),',',
          TestRate:6:4,',',
          TestLoss:6:4,',',
          TestError:6:4
        );
      end
      else
      begin
        WriteLn
        (
          CSVFile,
          FCurrentEpoch,',',
          (AccuracyWithInertia/100):6:4,',',
          TrainingLoss:6:4,',',
          TrainingError:6:4,',',
          ValidationRate:6:4,',',
          ValidationLoss:6:4,',',
          ValidationError:6:4,',',
          FCurrentLearningRate:9:7,',',
          Round( (Now() - globalStartTime) * 24 * 60 * 60),',,,'
        );
      end;
      if Assigned(FOnAfterEpoch) then FOnAfterEpoch(Self);

      CloseFile(CSVFile);
      AssignFile(CSVFile, FileNameCSV);
      Append(CSVFile);

      MessageProc(
        'Epoch time: ' + FloatToStrF( totalTimeSeconds*(pImgVolumes.Count/(FStepSize*10))/60,ffGeneral,1,4)+' minutes.' +
        ' '+IntToStr(Epochs)+' epochs: ' + FloatToStrF( Epochs*totalTimeSeconds*(pImgVolumes.Count/(FStepSize*10))/3600,ffGeneral,1,4)+' hours.');

      MessageProc(
        'Epochs: '+IntToStr(FCurrentEpoch)+
        '. Working time: '+FloatToStrF(Round((Now() - globalStartTime)*2400)/100,ffGeneral,4,2)+' hours.');
    end;
  end;

  FProcs.Free;
  FreeAndNil(FAvgWeight);
  if Assigned(FAvgWeights) then FreeAndNil(FAvgWeights);
  FreeAndNil(FThreadNN);
  CloseFile(CSVFile);
  FMessageProc('Finished.');
  FRunning := false;
end;

procedure TNeuralImageFit.RunNNThread(index, threadnum: integer);
var
  BlockSize, BlockSizeRest, CropSizeX, CropSizeY: integer;
  LocalNN: TNNet;
  ImgInput, ImgInputCp: TNNetVolume;
  pOutput, vOutput: TNNetVolume;
  I, ImgIdx: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
begin
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(FNumClasses,1,1);
  vOutput := TNNetVolume.Create(FNumClasses,1,1);

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := FBatchSize div FThreadNum;
  BlockSizeRest := FBatchSize mod FThreadNum;

  if (Index < BlockSizeRest) then Inc(BlockSize);

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(FNN);
  LocalNN.ClearTime();
  LocalNN.ClearDeltas();
  LocalNN.EnableDropouts(true);
  for I := 1 to BlockSize do
  begin
    if (FShouldQuit) then Break;
    ImgIdx := Random(FImgVolumes.Count);

    if FDataAugmentation then
    begin
      if FHasImgCrop then
      begin
        ImgInput.CopyCropping(FImgVolumes[ImgIdx], random(FMaxCropSize), random(FMaxCropSize), FImgVolumes[ImgIdx].SizeX-FMaxCropSize, FImgVolumes[ImgIdx].SizeY-FMaxCropSize);
      end
      else
      begin
        CropSizeX := random(FMaxCropSize + 1);
        CropSizeY := random(FMaxCropSize + 1);
        ImgInputCp.CopyCropping(FImgVolumes[ImgIdx], random(CropSizeX), random(CropSizeY),FImgVolumes[ImgIdx].SizeX-CropSizeX, FImgVolumes[ImgIdx].SizeY-CropSizeY);
        ImgInput.CopyResizing(ImgInputCp, FImgVolumes[ImgIdx].SizeX, FImgVolumes[ImgIdx].SizeY);
      end;

      if (ImgInput.Depth = 3) then
      begin
        if FHasMakeGray and (Random(1000) > 750) then
        begin
          ImgInput.MakeGray(FColorEncoding);
        end;
      end;

      if FHasFlipX and (Random(1000) > 500) then
      begin
        ImgInput.FlipX();
      end;

      if FHasFlipY and (Random(1000) > 500) then
      begin
        ImgInput.FlipY();
      end;
    end
    else begin
      ImgInput.Copy(FImgVolumes[ImgIdx]);
    end;
    ImgInput.Tag := FImgVolumes[ImgIdx].Tag;

    if ImgInput.Tag >= FNumClasses then
    begin
      FErrorProc
      (
        'Invalid image ' + IntToStr(ImgIdx) +
        ' input class: ' + IntToStr(ImgInput.Tag)
      );
      ReadLn;
      Continue;
    end;

    LocalNN.Compute( ImgInput );
    LocalNN.GetOutput( pOutput );

    if FIsSoftmax
      then vOutput.SetClassForSoftMax( ImgInput.Tag )
      else vOutput.SetClass( ImgInput.Tag, +0.9, -0.1);

    LocalErrorSum := LocalErrorSum + vOutput.SumDiff( pOutput );
    OutputValue := pOutput.FData[ ImgInput.Tag ];
    if Not(FIsSoftmax) then
    begin
      OutputValue := Max(OutputValue, 0.001);
      LocalNN.Backpropagate(vOutput);
    end
    else
    begin
      LocalNN.Backpropagate(vOutput);
    end;

    if (OutputValue > 0) then
    begin
      CurrentLoss := -Ln(OutputValue);
    end
    else
    begin
      WriteLn('Error - invalid output value:',OutputValue);
      CurrentLoss := 1;
    end;
    LocalTotalLoss := LocalTotalLoss + CurrentLoss;

    if pOutput.GetClass() = FImgVolumes[ImgIdx].Tag then
    begin
      Inc(LocalHit);
    end
    else
    begin
      Inc(LocalMiss);
    end;
  end; // of for

  if (Index and 1 = 0) and Not(FShouldQuit) then
  begin
    if Index + 1 < FThreadNum then
    begin
      while (FFinishedThread.FData[Index + 1] = 0) and Not(FShouldQuit) do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 1]);
      {$IFDEF FPC}
      FFinishedThread.FData[Index] += FFinishedThread.FData[Index + 1];
      {$ELSE}
      FFinishedThread.FData[Index] := FFinishedThread.FData[Index] +
        FFinishedThread.FData[Index + 1];
      {$ENDIF}
    end;
  end;
  FFinishedThread.FData[Index] := FFinishedThread.FData[Index] + 1;
  if (Index and 3 = 0) and Not(FShouldQuit) then
  begin
    if Index + 2 < FThreadNum then
    begin
      while (FFinishedThread.FData[Index + 2] = 0) and Not(FShouldQuit) do;
      LocalNN.SumDeltasNoChecks(FThreadNN[Index + 2]);
      {$IFDEF FPC}
      FFinishedThread.FData[Index] += FFinishedThread.FData[Index + 2];
      {$ELSE}
      FFinishedThread.FData[Index] := FFinishedThread.FData[Index] +
        FFinishedThread.FData[Index + 2];
      {$ENDIF}
    end;
  end;

  {$IFDEF HASTHREADS}EnterCriticalSection(FCritSec);{$ENDIF}
  FGlobalHit       := FGlobalHit + LocalHit;
  FGlobalMiss      := FGlobalMiss + LocalMiss;
  FGlobalTotalLoss := FGlobalTotalLoss + LocalTotalLoss;
  FGlobalErrorSum  := FGlobalErrorSum + LocalErrorSum;

  FNN.ForwardTime := FNN.ForwardTime + LocalNN.ForwardTime;
  FNN.BackwardTime := FNN.BackwardTime + LocalNN.BackwardTime;
  {$IFDEF Debug}
  if Index and 3 = 0 then FNN.SumDeltas(LocalNN);
  {$ELSE}
  if Index and 3 = 0 then FNN.SumDeltasNoChecks(LocalNN);
  {$ENDIF}
  {$IFDEF HASTHREADS}LeaveCriticalSection(FCritSec);{$ENDIF}
  ImgInputCp.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TNeuralImageFit.TestNNThread(index, threadnum: integer);
var
  BlockSize: integer;
  LocalNN: TNNet;
  ImgInput, ImgInputCp: TNNetVolume;
  pOutput, vOutput, sumOutput: TNNetVolume;
  I, ImgIdx: integer;
  StartPos, FinishPos: integer;
  OutputValue, CurrentLoss: TNeuralFloat;
  LocalHit, LocalMiss: integer;
  LocalTotalLoss, LocalErrorSum: TNeuralFloat;
  PredictedClass: integer;
  TotalDiv: TNeuralFloat;
begin
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  pOutput := TNNetVolume.Create(FNumClasses,1,1);
  vOutput := TNNetVolume.Create(FNumClasses,1,1);
  sumOutput := TNNetVolume.Create(FNumClasses,1,1);

  LocalHit := 0;
  LocalMiss := 0;
  LocalTotalLoss := 0;
  LocalErrorSum := 0;

  BlockSize := (FWorkingVolumes.Count div FThreadNum) {$IFDEF MakeQuick}div 10{$ENDIF};
  StartPos  := BlockSize * index;
  FinishPos := BlockSize * (index + 1) - 1;

  LocalNN := FThreadNN[Index];
  LocalNN.CopyWeights(FAvgWeight);
  LocalNN.EnableDropouts(false);
  for I := StartPos to FinishPos - 1 do
  begin
    if FShouldQuit then Break;
    sumOutput.Fill(0);
    ImgIdx := I;

    if FHasImgCrop then
    begin
      ImgInput.CopyCropping(FWorkingVolumes[ImgIdx], FMaxCropSize div 2, FMaxCropSize div 2, FImgVolumes[ImgIdx].SizeX-FMaxCropSize, FImgVolumes[ImgIdx].SizeY-FMaxCropSize);
    end
    else
    begin
      ImgInput.Copy(FWorkingVolumes[ImgIdx]);
    end;

    ImgInput.Tag := FWorkingVolumes[ImgIdx].Tag;

    LocalNN.Compute( ImgInput );
    LocalNN.GetOutput( pOutput );
    if FMultipleSamplesAtValidation then
    begin
      TotalDiv := 1;
      sumOutput.Add( pOutput );

      if FHasFlipX then
      begin
        ImgInput.FlipX();
        TotalDiv := TotalDiv + 1;
        LocalNN.Compute( ImgInput );
        LocalNN.GetOutput( pOutput );
        sumOutput.Add( pOutput );
      end;

      if ((FMaxCropSize >= 2) and Not(FHasImgCrop)) then
      begin
        ImgInputCp.CopyCropping(ImgInput, FMaxCropSize div 2, FMaxCropSize div 2, ImgInput.SizeX - FMaxCropSize, ImgInput.SizeY - FMaxCropSize);
        ImgInput.CopyResizing(ImgInputCp, ImgInput.SizeX, ImgInput.SizeY);
        LocalNN.Compute( ImgInput );
        LocalNN.GetOutput( pOutput );
        sumOutput.Add( pOutput );
        TotalDiv := TotalDiv + 1;
      end;
      sumOutput.Divi(TotalDiv);

      pOutput.Copy(sumOutput);
    end;

    vOutput.SetClassForSoftMax( ImgInput.Tag );
    LocalErrorSum := LocalErrorSum + vOutput.SumDiff( pOutput );

    OutputValue := pOutput.FData[ ImgInput.Tag ];
    if Not(FIsSoftmax) then OutputValue := OutputValue + 0.5001;

    if (OutputValue > 0) then
    begin
      CurrentLoss := -Ln(OutputValue);
    end
    else
    begin
      WriteLn('Error - invalid output value:',OutputValue);
      CurrentLoss := 1;
    end;
    LocalTotalLoss := LocalTotalLoss + CurrentLoss;

    PredictedClass := pOutput.GetClass();
    if PredictedClass = ImgInput.Tag then
    begin
      Inc(LocalHit);
    end
    else
    begin
      Inc(LocalMiss);
    end;
  end; // of for
  LocalNN.EnableDropouts(true);

  {$IFDEF HASTHREADS}EnterCriticalSection(FCritSec);{$ENDIF}
  FGlobalHit       := FGlobalHit + LocalHit;
  FGlobalMiss      := FGlobalMiss + LocalMiss;
  FGlobalTotalLoss := FGlobalTotalLoss + LocalTotalLoss;
  FGlobalErrorSum  := FGlobalErrorSum + LocalErrorSum;
  {$IFDEF HASTHREADS}LeaveCriticalSection(FCritSec);{$ENDIF}

  sumOutput.Free;
  ImgInputCp.Free;
  ImgInput.Free;
  vOutput.Free;
  pOutput.Free;
end;

procedure TNeuralImageFit.ClassifyImage(pNN: TNNet; pImgInput, pOutput: TNNetVolume);
var
  ImgInput, ImgInputCp: TNNetVolume;
  sumOutput: TNNetVolume;
  TotalDiv: TNeuralFloat;
begin
  ImgInput := TNNetVolume.Create();
  ImgInputCp := TNNetVolume.Create();
  sumOutput := TNNetVolume.Create(FNumClasses,1,1);

  ImgInput.Copy( pImgInput );
  pNN.Compute( ImgInput );
  pNN.GetOutput( pOutput );

  if FMultipleSamplesAtValidation then
  begin
    TotalDiv := 1;
    sumOutput.Add( pOutput );

    if FHasFlipX then
    begin
      ImgInput.FlipX();
      TotalDiv := TotalDiv + 1;
      pNN.Compute( ImgInput );
      pNN.GetOutput( pOutput );
      sumOutput.Add( pOutput );
    end;

    if FMaxCropSize >= 2 then
    begin
      ImgInputCp.CopyCropping(ImgInput, FMaxCropSize div 2, FMaxCropSize div 2, ImgInput.SizeX - FMaxCropSize, ImgInput.SizeY - FMaxCropSize);
      ImgInput.CopyResizing(ImgInputCp, ImgInput.SizeX, ImgInput.SizeY);
      pNN.Compute( ImgInput );
      pNN.GetOutput( pOutput );
      sumOutput.Add( pOutput );
      TotalDiv := TotalDiv + 1;
    end;
    sumOutput.Divi(TotalDiv);

    pOutput.Copy(sumOutput);
  end;

  sumOutput.Free;
  ImgInputCp.Free;
  ImgInput.Free;
end;

end.
