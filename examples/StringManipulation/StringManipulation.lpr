program StringManipulation;
(*
HypotenuseFitLoading: learns how to calculate hypotenuse sqrt(X^2 + Y^2).
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
  CustApp,
  Math;

const
  csContextLen = 64;
  csVocabSize  = 128; // Character based vocabulary/dictionary.
  csMinSampleSize = 3; // Minimum of 3 characters.

type

  { TStringManipulation }

  TStringManipulation = class(TCustomApplication)
  protected
    FDataset: array of string;
    FDatasetSize: integer;
    FNN: TNNet;
    procedure LoadDataset;
    procedure DoRun; override;
  public
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

  procedure TStringManipulation.LoadDataset;
  begin
    SetLength(FDataset, 3);
    FDataset[0] := 'happy good morning.'+chr(1);
    FDataset[1] := 'fantastic good evening.'+chr(1);
    FDataset[2] := 'superb good night.'+chr(1);
    FDatasetSize := Length(FDataset);
  end;

  procedure TStringManipulation.DoRun;
  var
    NFit: TNeuralDataLoadingFit;
  begin
    LoadDataset();
    FNN := TNNet.Create();
    NFit := TNeuralDataLoadingFit.Create();

    FNN.AddLayer([
      TNNetInput.Create(csContextLen, 1, csVocabSize),
      TNNetPointwiseConvReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectLinear.Create(csVocabSize),
      TNNetSoftMax.Create()
    ]);
    FNN.DebugStructure;

    WriteLn('Computing...');
    //NFit.MaxThreadNum := 1;
    NFit.InitialLearningRate := 0.001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.LogEveryBatches := 100;
    NFit.EnableClassComparison();
    NFit.EnableDefaultLoss();
    NFit.AvgWeightEpochCount := 1;
    NFit.FitLoading(
      FNN,
      {TrainingVolumesCount=}10000,
      {ValidationVolumesCount=}1000,
      {TestVolumesCount=}1000,
      {batchsize=}32,
      {epochs=}50,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );
    FNN.DebugWeights();
    WriteLn('Testing.');

    WriteLn(GenerateStringFromChars(FNN, 'happy'));
    WriteLn(GenerateStringFromChars(FNN, 'fantastic'));
    WriteLn(GenerateStringFromChars(FNN, 'superb'));

    NFit.Free;
    FNN.Free;
    Write('Press ENTER to exit.');
    ReadLn;
    Terminate;
  end;

  procedure TStringManipulation.GetTrainingPair(Idx: integer; ThreadId: integer;
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
    SampleLen := Length(FDataset[SampleId]);
    SampleCutPosition := Random(SampleLen-csMinSampleSize)+csMinSampleSize; // -1
    // The expected token is the next character in the string
    ExpectedTokenChar := FDataset[SampleId][SampleCutPosition+1];
    ExpectedTokenInt := Min(Ord(ExpectedTokenChar),pInput.Depth-1);
    // Encode the input and output volumes
    pInput.OneHotEncodingReversed(copy(FDataset[SampleId], 1, SampleCutPosition));
    pOutput.SetClassForSoftMax(ExpectedTokenInt);
    pOutput.Tag := ExpectedTokenInt;
  end;

  procedure TStringManipulation.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetTrainingPair(Idx, ThreadId, pInput, pOutput);
  end;

  procedure TStringManipulation.GetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetTrainingPair(Idx, ThreadId, pInput, pOutput);
  end;

var
  Application: TStringManipulation;
begin
  Application := TStringManipulation.Create(nil);
  Application.Title:='Simple String Manipulation';
  Application.Run;
  Application.Free;
end.

