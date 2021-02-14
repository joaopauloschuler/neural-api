program HypotenuseFitLoading;
(*
HypotenuseFitLoading: learns how to calculate hypotenuse sqrt(X^2 + Y^2).
Copyright (C) 2019 Joao Paulo Schwarz Schuler

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
  CustApp;

type

  { TTestFitLoading }

  TTestFitLoading = class(TCustomApplication)
  protected
    procedure DoRun; override;
  public
    procedure GetTrainingPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetValidationPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
    procedure GetTestPair(Idx: integer; ThreadId: integer; pInput, pOutput: TNNetVolume);
  end;

  // Returns TRUE if difference is smaller than 0.1 .
  function LocalFloatCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
  begin
    Result := ( Abs(A.FData[0]-B.FData[0])<0.1 );
  end;

  procedure TTestFitLoading.DoRun;
  var
    NN: TNNet;
    NFit: TNeuralDataLoadingFit;
    TestInput, TestOutput: TNNetVolume;
    Cnt: integer;
    pOutPut: TNNetVolume;
  begin
    NN := TNNet.Create();
    NFit := TNeuralDataLoadingFit.Create();
    TestInput := TNNetVolume.Create(2);
    TestOutput := TNNetVolume.Create(1);

    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(1)
    ]);

    WriteLn('Computing...');
    NFit.InitialLearningRate := 0.00001;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.InferHitFn := @LocalFloatCompare;
    NFit.FitLoading(
      NN,
      {TrainingVolumesCount=}10000,
      {ValidationVolumesCount=}1000,
      {TestVolumesCount=}1000,
      {batchsize=}32,
      {epochs=}50,
      @GetTrainingPair, @GetValidationPair, @GetTestPair
    );
    NN.DebugWeights();

    pOutPut := TNNetVolume.Create({pSizeX=}1, {pSizeY=}1, {pDepth=}1, {FillValue=}1);

    // tests the learning
    for Cnt := 0 to 9 do
    begin
      GetTestPair({Idx=}0, {ThreadId=}0, TestInput, TestOutput);
      NN.Compute(TestInput);
      NN.GetOutput(pOutPut);
      WriteLn
      ( 'Inputs:',
        TestInput.FData[0]:5:2,', ',
        TestInput.FData[1]:5:2,' - ',
        'Output:',
        pOutPut.Raw[0]:5:2,' ',
        ' Desired Output:',
        TestOutput.FData[0]:5:2
      );
    end;
    TestInput.Free;
    TestOutput.Free;
    pOutPut.Free;
    NFit.Free;
    NN.Free;
    Write('Press ENTER to exit.');
    ReadLn;
    Terminate;
  end;

  procedure TTestFitLoading.GetTrainingPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  var
    LocalX, LocalY, Hypotenuse: TNeuralFloat;
  begin
    LocalX := Random(100);
    LocalY := Random(100);
    Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);
    pInput.ReSize(2,1,1);
    pInput.FData[0] := LocalX;
    pInput.FData[1] := LocalY;
    pOutput.ReSize(1,1,1);
    pOutput.FData[0] := Hypotenuse;
  end;

  procedure TTestFitLoading.GetValidationPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetTrainingPair(Idx, ThreadId, pInput, pOutput);
  end;

  procedure TTestFitLoading.GetTestPair(Idx: integer; ThreadId: integer;
    pInput, pOutput: TNNetVolume);
  begin
    GetTrainingPair(Idx, ThreadId, pInput, pOutput);
  end;

var
  Application: TTestFitLoading;
begin
  Application := TTestFitLoading.Create(nil);
  Application.Title:='Hypotenuse Example with FitLoading';
  Application.Run;
  Application.Free;
end.

