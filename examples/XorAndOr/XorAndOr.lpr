program XorAndOr;
(*
XorAndOr: learns boolean functions XOR, AND and OR.
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

type TBackInput  = array[0..3] of array[0..1] of TNeuralFloat;
type TBackOutput = array[0..3] of array[0..2] of TNeuralFloat;

const cs_inputs : TBackInput =
  ( // x1,   x2
    ( 0.1,  0.1), // False, False
    ( 0.1,  0.8), // False, True
    ( 0.8,  0.1), // True,  False
    ( 0.8,  0.8)  // True,  True
  );

const cs_outputs : TBackOutput =
  (// XOR, AND,   OR
    ( 0.1, 0.1, 0.1),
    ( 0.8, 0.1, 0.8),
    ( 0.8, 0.1, 0.8),
    ( 0.1, 0.8, 0.8)
  );

  procedure RunAlgo();
  var
    NN: TNNet;
    NFit: TNeuralFit;
    TrainingPairs: TNNetVolumePairList;
    Cnt: integer;
    pOutPut: TNNetVolume;
    vInputs: TBackInput;
    vOutput: TBackOutput;
  begin
    NN := TNNet.Create();
    NFit := TNeuralFit.Create();
    TrainingPairs := TNNetVolumePairList.Create();
    NN.AddLayer( TNNetInput.Create(2) );
    NN.AddLayer( TNNetFullConnect.Create(3) );
    NN.AddLayer( TNNetFullConnectLinear.Create(3) );

    vInputs := cs_inputs;
    vOutput := cs_outputs;
    for Cnt := Low(cs_inputs) to High(cs_inputs) do
    begin
      TrainingPairs.Add(
        TNNetVolumePair.Create(
          TNNetVolume.Create(vInputs[Cnt]),
          TNNetVolume.Create(vOutput[Cnt])
        )
      );
    end;

    WriteLn('Computing...');
    NFit.InitialLearningRate := 0.01;
    NFit.LearningRateDecay := 0;
    NFit.L2Decay := 0;
    NFit.Verbose := false;
    NFit.HideMessages();
    NFit.InferHitFn := @MonopolarCompare;
    NFit.Fit(NN, TrainingPairs, nil, nil, {batchsize=}4, {epochs=}6000);

    pOutPut := TNNetVolume.Create(3,1,1,1);

    // tests the learning
    for Cnt := Low(cs_inputs) to High(cs_inputs) do
    begin
      NN.Compute(vInputs[Cnt]);
      NN.GetOutput(pOutPut);
      WriteLn
      (
        ' Output:',
        pOutPut.Raw[0]:5:2,' ',
        pOutPut.Raw[1]:5:2,' ',
        pOutPut.Raw[2]:5:2,
        ' - Training/Desired Output:',
        vOutput[cnt][0]:5:2,' ',
        vOutput[cnt][1]:5:2,' ' ,
        vOutput[cnt][2]:5:2,' '
      );
    end;

    NN.DebugWeights();
    NN.DebugErrors();
    pOutPut.Free;
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
  Application.Title:='Xor And Or Example';
  RunAlgo();
end.
