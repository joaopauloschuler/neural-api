program OnlyOneNeuron2x3y4;
(*
OnlyOneNeuronOrOperation: this free pascal source code trains a neural network
that contains only one neuron to learn the function f(x,y) = 2x - 3y + 4.
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

type
  // Define the input and output types for training data
  TBackFloatInput  = array[0..1] of TNeuralFloat;  // Input data for 2x - 3y + 4
  TBackFloatOutput = array[0..0] of TNeuralFloat;  // Expected output for 2x - 3y + 4

procedure RunFloatAlgo();
var
  NN: TNNet;
  EpochCnt: integer;
  pOutPut: TNNetVolume;
  vInputs: TBackFloatInput;
  vOutput: TBackFloatOutput;
begin
  NN := TNNet.Create();

  // Create the neural network layers
  NN.AddLayer(TNNetInput.Create(2));                     // Input layer with 2 neurons
  NN.AddLayer(TNNetFullConnectLinear.Create(1));         // Single neuron layer connected to both inputs from the previous layer.

  NN.SetLearningRate(0.0001, 0);                         // Set the learning rate and momentum

  pOutPut := TNNetVolume.Create(1, 1, 1, 1);             // Create a volume to hold the output

  WriteLn;

  for EpochCnt := 1 to 100000 do
  begin
    vInputs[0] := (Random(10000) - 5000)/100;          // Random number in the interval [-50,+50].
    vInputs[1] := (Random(10000) - 5000)/100;          // Random number in the interval [-50,+50].
    vOutput[0] := 2*vInputs[0] - 3*vInputs[1] + 4;     // 2x - 3y + 4
    // Feed forward and backpropagation
    NN.Compute(vInputs);                               // Perform feedforward computation
    NN.GetOutput(pOutPut);                             // Get the output of the network
    NN.Backpropagate(vOutput);                         // Perform backpropagation to adjust weights

    if EpochCnt mod 5000 = 0 then
        WriteLn(
          EpochCnt:7, 'x',
          ' Output:', pOutPut.Raw[0]:5:2,' ',
          ' - Training/Desired Output:', vOutput[0]:5:2,' '
        );
  end;

  NN.DebugWeights();                                    // Display the final weights of the network

  pOutPut.Free;                                         // Free the memory allocated for output
  NN.Free;                                              // Free the memory allocated for the network

  Write('Press ENTER to exit.');
  ReadLn;
end;

var
  // Stops Lazarus errors
  Application: record Title:string; end;

begin
  Application.Title:='Only One Neuron - 2x - 3y + 4';
  RunFloatAlgo();
end.
