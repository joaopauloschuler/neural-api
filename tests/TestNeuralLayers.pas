unit TestNeuralLayers;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, fpcunit, testregistry, neuralnetwork, neuralvolume;

type
  TTestNeuralLayers = class(TTestCase)
  published
    procedure TestFullyConnectedForward;
    procedure TestConvolutionForward;
    procedure TestMaxPoolForward;
    procedure TestNetworkSaveLoad;
    procedure TestSimpleXORLearning;
  end;

implementation

procedure TTestNeuralLayers.TestFullyConnectedForward;
var
  NN: TNNet;
  Input, Output: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(2, 1, 1);
  Output := TNNetVolume.Create(1, 1, 1);
  try
    NN.AddLayer(TNNetInput.Create(2));
    NN.AddLayer(TNNetFullConnectLinear.Create(1));

    Input.Raw[0] := 1.0;
    Input.Raw[1] := 1.0;

    NN.Compute(Input);
    NN.GetOutput(Output);

    // Just verify it runs without error and produces output
    AssertTrue('Output should have size 1', Output.Size = 1);
  finally
    NN.Free;
    Input.Free;
    Output.Free;
  end;
end;

procedure TTestNeuralLayers.TestConvolutionForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetConvolutionReLU.Create(16, 3, 1, 1));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 8', 8, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output depth should be 16', 16, NN.GetLastLayer.Output.Depth);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestMaxPoolForward;
var
  NN: TNNet;
  Input: TNNetVolume;
begin
  NN := TNNet.Create();
  Input := TNNetVolume.Create(8, 8, 3);
  try
    NN.AddLayer(TNNetInput.Create(8, 8, 3));
    NN.AddLayer(TNNetMaxPool.Create(2));

    Input.Fill(1.0);
    NN.Compute(Input);

    AssertEquals('Output SizeX should be 4 after 2x2 pool', 4, NN.GetLastLayer.Output.SizeX);
    AssertEquals('Output SizeY should be 4 after 2x2 pool', 4, NN.GetLastLayer.Output.SizeY);
  finally
    NN.Free;
    Input.Free;
  end;
end;

procedure TTestNeuralLayers.TestNetworkSaveLoad;
var
  NN1, NN2: TNNet;
  Input, Output1, Output2: TNNetVolume;
  TempFile: string;
begin
  NN1 := TNNet.Create();
  NN2 := TNNet.Create();
  Input := TNNetVolume.Create(4, 1, 1);
  Output1 := TNNetVolume.Create(2, 1, 1);
  Output2 := TNNetVolume.Create(2, 1, 1);
  TempFile := GetTempDir() + 'nn_test_' + IntToStr(Random(MaxInt)) + '.nn';
  try
    NN1.AddLayer(TNNetInput.Create(4));
    NN1.AddLayer(TNNetFullConnectReLU.Create(8));
    NN1.AddLayer(TNNetFullConnectLinear.Create(2));

    Input.RandomizeGaussian();
    NN1.Compute(Input);
    NN1.GetOutput(Output1);

    NN1.SaveToFile(TempFile);
    NN2.LoadFromFile(TempFile);

    NN2.Compute(Input);
    NN2.GetOutput(Output2);

    AssertEquals('Loaded network should produce same output', 0.0, Output1.SumDiff(Output2), 0.0001);
  finally
    NN1.Free;
    NN2.Free;
    Input.Free;
    Output1.Free;
    Output2.Free;
    DeleteFile(TempFile);
  end;
end;

procedure TTestNeuralLayers.TestSimpleXORLearning;
var
  NN: TNNet;
  // Quick smoke test - just verify XOR network can be constructed
begin
  NN := TNNet.Create();
  try
    NN.AddLayer([
      TNNetInput.Create(2),
      TNNetFullConnectReLU.Create(4),
      TNNetFullConnectReLU.Create(4),
      TNNetFullConnectLinear.Create(1)
    ]);
    AssertEquals('Network should have 4 layers', 4, NN.CountLayers);
  finally
    NN.Free;
  end;
end;

initialization
  RegisterTest(TTestNeuralLayers);

end.
