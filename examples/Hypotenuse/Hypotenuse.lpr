program Hypotenuse;
(*
Hypotenuse: learns how to calculate hypotenuse sqrt(X^2 + Y^2).
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
  neuralfit;

  function CreateHypotenusePairList(MaxCnt: integer): TNNetVolumePairList;
  var
    Cnt: integer;
    LocalX, LocalY, Hypotenuse: TNeuralFloat;
  begin
    Result := TNNetVolumePairList.Create();
    for Cnt := 1 to MaxCnt do
    begin
      LocalX := Random(100);
      LocalY := Random(100);
      Hypotenuse := sqrt(LocalX*LocalX + LocalY*LocalY);

      Result.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create([LocalX, LocalY]),
          TNNetVolume.Create([Hypotenuse])
        )
      );
    end;
  end;

  // Returns TRUE if difference is smaller than 0.1 .
  function LocalFloatCompare(A, B: TNNetVolume; ThreadId: integer): boolean;
  begin
    Result := ( Abs(A.FData[0]-B.FData[0])<0.1 );
  end;

  procedure RunAlgo();
  var
    NN: TNNet;
    NFit: TNeuralFit;
    TrainingPairs, ValidationPairs, TestPairs: TNNetVolumePairList;
    Cnt: integer;
    pOutPut: TNNetVolume;
  begin
    NN := TNNet.Create();
    NFit := TNeuralFit.Create();
    TrainingPairs := CreateHypotenusePairList(10000);
    ValidationPairs := CreateHypotenusePairList(1000);
    TestPairs := CreateHypotenusePairList(1000);

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
    NFit.Fit(NN, TrainingPairs, ValidationPairs, TestPairs, {batchsize=}32, {epochs=}50);
    NN.DebugWeights();

    pOutPut := TNNetVolume.Create({pSizeX=}1, {pSizeY=}1, {pDepth=}1, {FillValue=}1);

    // tests the learning
    for Cnt := 0 to 9 do
    begin
      NN.Compute(TestPairs[Cnt].I);
      NN.GetOutput(pOutPut);
      WriteLn
      ( 'Inputs:',
        TestPairs[Cnt].I.FData[0]:5:2,', ',
        TestPairs[Cnt].I.FData[1]:5:2,' - ',
        'Output:',
        pOutPut.Raw[0]:5:2,' ',
        ' Desired Output:',
        TestPairs[Cnt].O.FData[0]:5:2
      );
    end;
    pOutPut.Free;
    TestPairs.Free;
    ValidationPairs.Free;
    TrainingPairs.Free;
    NFit.Free;
    NN.Free;
    Write('Press ENTER to exit.');
    ReadLn;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='Hypotenuse Example';
  RunAlgo();
end.
