program OnlyOneNeuronOrOperation;
(*
OnlyOneNeuronOrOperation: this free pascal source code trains a neural network
that contains only one neuron to learn the OR boolean operation.
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
  TBackInput  = array[0..3] of array[0..1] of TNeuralFloat;  // Input data for OR operation
  TBackOutput = array[0..3] of array[0..0] of TNeuralFloat;  // Expected output for OR operation

const
  cs_false = 0.1;                          // Encoding for "false" value
  cs_true  = 0.8;                          // Encoding for "true" value
  cs_threshold = (cs_false + cs_true) / 2; // Threshold for neuron activation

const
  cs_inputs : TBackInput =
  (
    // Input data for OR operation
    (cs_false, cs_false),
    (cs_false, cs_true),
    (cs_true,  cs_false),
    (cs_true,  cs_true)
  );

const
  cs_outputs : TBackOutput =
  (
    // Expected outputs for OR operation
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

  // Create the neural network layers
  NN.AddLayer(TNNetInput.Create(2));                     // Input layer with 2 neurons
  NN.AddLayer(TNNetFullConnectLinear.Create(1));         // Single neuron layer connected to both inputs from the previous layer.

  NN.SetLearningRate(0.01, 0.9);                         // Set the learning rate and momentum

  vInputs := cs_inputs;                                  // Assign the input data
  vOutput := cs_outputs;                                 // Assign the expected output data
  pOutPut := TNNetVolume.Create(1, 1, 1, 1);             // Create a volume to hold the output

  WriteLn('Value encoding FALSE is: ', cs_false:4:2);    // Display the encoding for "false"
  WriteLn('Value encoding TRUE is: ', cs_true:4:2);      // Display the encoding for "true"
  WriteLn('Threshold is: ', cs_threshold:4:2);           // Display the threshold value
  WriteLn;

  for EpochCnt := 1 to 600 do
  begin
    for Cnt := Low(cs_inputs) to High(cs_inputs) do
    begin
      // Feed forward and backpropagation
      NN.Compute(vInputs[Cnt]);                          // Perform feedforward computation
      NN.GetOutput(pOutPut);                             // Get the output of the network
      NN.Backpropagate(vOutput[Cnt]);                    // Perform backpropagation to adjust weights

      if EpochCnt mod 100 = 0 then
        WriteLn(
          EpochCnt:7, 'x', Cnt,
          ' Output:', pOutPut.Raw[0]:5:2,' ',
          ' - Training/Desired Output:', vOutput[cnt][0]:5:2,' '
        );
    end;

    if EpochCnt mod 100 = 0 then
      WriteLn();
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
  Application.Title:='Only One Neuron - OR Operation';
  RunAlgo();
end.
