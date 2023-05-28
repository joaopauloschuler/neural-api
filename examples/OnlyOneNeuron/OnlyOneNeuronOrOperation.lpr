program OnlyOneNeuronOrOperation;
(*
OnlyOneNeuronOrOperation: learns how to calculate boolean X OR Y.
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
  neuralfit;

type TBackInput  = array[0..3] of array[0..1] of TNeuralFloat;
type TBackOutput = array[0..3] of array[0..0] of TNeuralFloat;

const
  cs_false = 0.1;
  cs_true  = 0.8;
  cs_threshold = (cs_false+cs_true) / 2;

const cs_inputs : TBackInput =
  ( // x1,   x2
    ( cs_false, cs_false),
    ( cs_false, cs_true),
    ( cs_true,  cs_false),
    ( cs_true,  cs_true)
  );

const cs_outputs : TBackOutput =
  ( // OR result
    (cs_false),
    (cs_true),
    (cs_true),
    (cs_true)
  );

  procedure RunAlgo();
  var
    NN: TNNet;
    EpochCnt: integer;
    Cnt: integer;
    pOutPut: TNNetVolume;
    vInputs: TBackInput;
    vOutput: TBackOutput;
  begin
    NN := TNNet.Create();
    NN.AddLayer( TNNetInput.Create(2) );
    NN.AddLayer( TNNetFullConnectLinear.Create(1) );
    NN.SetLearningRate(0.01, 0.9);

    vInputs := cs_inputs;
    vOutput := cs_outputs;
    pOutPut := TNNetVolume.Create(1, 1, 1, 1);

    WriteLn('Value encoding FALSE is: ', cs_false:4:2);
    WriteLn('Value encoding TRUE is: ', cs_true:4:2);
    WriteLn('Threshold is: ', cs_threshold:4:2);
    WriteLn;

    for EpochCnt := 1 to 600 do
    begin
      for Cnt := Low(cs_inputs) to High(cs_inputs) do
      begin
        NN.Compute(vInputs[Cnt]);
        NN.GetOutput(pOutPut);
        NN.Backpropagate(vOutput[Cnt]);
        if EpochCnt mod 100 = 0 then
        WriteLn
        (
          EpochCnt:7,'x',Cnt,
          ' Output:',
          pOutPut.Raw[0]:5:2,' ',
          ' - Training/Desired Output:',
          vOutput[cnt][0]:5:2,' '
        );
      end;
      if EpochCnt mod 100 = 0 then WriteLn();
    end;
    NN.DebugWeights();
    pOutPut.Free;
    NN.Free;
    Write('Press ENTER to exit.');
    ReadLn;
  end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='Only One Neuron - OR Operation';
  RunAlgo();
end.
