program smallLanguageModelConvFromFile;
(*
Copyright (C) 2023 Joao Paulo Schwarz Schuler

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

{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes,
  neuralnetwork,
  neuralvolume,
  neuralfit,
  neuralthread,
  CustApp,
  Math;

const
  csContextLen = 81;
  csTrainingFileName = 'tinystories.txt';
  csVocabSize  = 128; // Character based vocabulary/dictionary.
  csMinSampleSize = 3; // Minimum of 3 characters.

type

  { TTestFitLoading }

  TTestFitLoading = class(TCustomApplication)
  protected
    FDataset: TStringList;
    FDatasetSize: integer;
    FNN: TNNet;
    NFit: TNeuralDataLoadingFit;
    FSampler: TNNetSamplerBase;
    FMaxPredictCharPos: integer;
    procedure LoadDataset;
    procedure DoRun; override;
  public
    procedure OnAfterEpoch(Sender: TObject);
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

  procedure TTestFitLoading.LoadDataset;
  var
    RowCnt: integer;
  begin
    FDataset.LoadFromFile(csTrainingFileName);
    FDatasetSize := FDataset.Count;
    for RowCnt := FDatasetSize-1 downto 0 do
    begin
      // removes too short strings
      if Length(FDataset[RowCnt])<csMinSampleSize then FDataset.Delete(RowCnt);
    end;
    FDatasetSize := FDataset.Count;
    for RowCnt := FDatasetSize-1 downto 0 do
    begin
      // removes too short strings
      FDataset[RowCnt] := LowerCase(FDataset[RowCnt]) + chr(1);
    end;
    WriteLn('Loaded dataset with ', FDatasetSize, ' rows');
  end;

  procedure TTestFitLoading.DoRun;
  begin
    FDataset := TStringList.Create();
    LoadDataset();
    FNN := TNNet.Create();
    NFit := TNeuralDataLoadingFit.Create();
    FMaxPredictCharPos := csMinSampleSize;
    FSampler := TNNetSamplerTopP.Create(0.4);
    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, csVocabSize),
      TNNetPointwiseConv.Create(32,1),
      TNNetPadXY.Create(1,0),
      TNNetConvolutionReLU.Create(64,3,0,1,1),
      TNNetMaxPool.Create(3),
      TNNetPadXY.Create(1,0),
      TNNetConvolutionReLU.Create(128*3,3,0,1,1),
      TNNetPointwiseConvReLU.Create(1024,0),
      TNNetMaxPoolWithPosition.Create(27,27,0,1,0),
      TNNetPointwiseConvReLU.Create(1024),
      TNNetPointwiseConvReLU.Create(128),
      TNNetFullConnectLinear.Create(csVocabSize),
      TNNetSoftMax.Create()
    ]);
    DebugThreadCount();
    FNN.DebugStructure;

    WriteLn('Computing...');
    NFit.MaxThreadNum := 32;
    NFit.LogEveryBatches := 100;
    NFit.InitialLearningRate := 0.0001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.AvgWeightEpochCount := 1;
    NFit.OnAfterEpoch := @OnAfterEpoch;
    NFit.FitLoading(
      FNN,
      {TrainingVolumesCount=}32000*3,
      {ValidationVolumesCount=}32000*3 div 20,
      {TestVolumesCount=}32000*3 div 20,
      {batchsize=}320,
      {epochs=}500,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );
    FNN.DebugWeights();
    OnAfterEpoch(Self);
    FSampler.Free;
    NFit.Free;
    FNN.Free;
    FDataset.Free;
    Terminate;
  end;

  procedure TTestFitLoading.OnAfterEpoch(Sender: TObject);
  begin
    WriteLn('Testing.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'once', FSampler),'.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'one ', FSampler),'.');
    WriteLn(GenerateStringFromChars(NFit.NN, 'once upon ', FSampler),'.');
    if NFit.TrainingAccuracy < 0.5
      then FMaxPredictCharPos := Max(FMaxPredictCharPos-1, csMinSampleSize)
      else FMaxPredictCharPos := Min(FMaxPredictCharPos+1, csContextLen);
    WriteLn('Max prediction pos is: ', FMaxPredictCharPos);
  end;

  procedure TTestFitLoading.GetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleCutPosition: integer;
    ExpectedTokenChar: char;
    ExpectedTokenInt: integer;
  begin
    // Make sure that expected input and output have the proper sizes.
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    // Get the input sample
    SampleId := Random(FDatasetSize);
    SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
    SampleLen := Min(FMaxPredictCharPos, SampleLen);
    SampleCutPosition := Random(SampleLen-csMinSampleSize)+csMinSampleSize; // -1
    // The expected token is the next character in the string
    ExpectedTokenChar := FDataset[SampleId][SampleCutPosition+1];
    ExpectedTokenInt := Min(Ord(ExpectedTokenChar),pInput.Depth-1);
    // Encode the input and output volumes
    pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
    pOutput.SetClassForSoftMax(ExpectedTokenInt);
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TTestFitLoading.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    SampleId: integer;
    SampleLen: integer;
    SampleCutPosition: integer;
    ExpectedTokenChar: char;
    ExpectedTokenInt: integer;
  begin
    // Make sure that expected input and output have the proper sizes.
    if FNN.GetFirstLayer().Output.Size <> pInput.Size then pInput.ReSize(FNN.GetFirstLayer().Output);
    if FNN.GetLastLayer().Output.Size <> pOutput.Size then pOutput.ReSize(FNN.GetLastLayer().Output);
    // Get the input sample
    SampleId := Idx;
    SampleLen := Min(Length(FDataset[SampleId]), pInput.SizeX);
    SampleCutPosition := (Idx mod (1+SampleLen-csMinSampleSize))+csMinSampleSize-1;
    // The expected token is the next character in the string
    ExpectedTokenChar := FDataset[SampleId][SampleCutPosition+1];
    ExpectedTokenInt := Min(Ord(ExpectedTokenChar),pInput.Depth-1);
    // Encode the input and output volumes
    pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
    pOutput.SetClassForSoftMax(ExpectedTokenInt);
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TTestFitLoading.GetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetValidationPair(Idx, ThreadId, pInput, pOutput);
  end;

var
  Application: TTestFitLoading;
begin
  Application := TTestFitLoading.Create(nil);
  Application.Title:='Nano Covolutional Based NLP Trained from File';
  Application.Run;
  Application.Free;
end.
