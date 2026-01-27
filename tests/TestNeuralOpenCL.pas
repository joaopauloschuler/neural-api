unit TestNeuralOpenCL;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Math, fpcunit, testregistry, neuralnetwork, neuralvolume
  {$IFDEF OpenCL}, neuralopencl, cl{$ENDIF};

type
  TTestNeuralOpenCL = class(TTestCase)
  {$IFDEF OpenCL}
  published
    procedure TestFullConnectBackpropagateOpenCL;
    procedure TestFullConnectBackpropagateOpenCLVsCPU;
    procedure TestFullConnectBackpropagateOpenCLBatchMode;
  {$ENDIF}
  end;

implementation

{$IFDEF OpenCL}

procedure TTestNeuralOpenCL.TestFullConnectBackpropagateOpenCL;
var
  NN: TNNet;
  Input, Target: TNNetVolume;
  Layer: TNNetFullConnectLinear;
  InitialWeightSum, FinalWeightSum: TNeuralFloat;
  ErrorBefore, ErrorAfter: TNeuralFloat;
  EasyCL: TEasyOpenCL;
  platform_id: cl_platform_id;
  device_id: cl_device_id;
  i, Epoch: integer;
begin
  // Initialize OpenCL
  EasyCL := TEasyOpenCL.Create();
  try
    if EasyCL.GetPlatformCount() = 0 then
    begin
      Ignore('No OpenCL platforms available - skipping test');
      Exit;
    end;

    // Set platform
    EasyCL.SetCurrentPlatform(EasyCL.PlatformIds[0]);

    if EasyCL.GetDeviceCount() = 0 then
    begin
      Ignore('No OpenCL devices available - skipping test');
      Exit;
    end;

    // Set device
    EasyCL.SetCurrentDevice(EasyCL.Devices[0]);

    // Get platform and device IDs
    platform_id := EasyCL.CurrentPlatform;
    device_id := EasyCL.CurrentDevice;

    // Create network with proper size for OpenCL (requires 128+ inputs, 512+ neurons)
    NN := TNNet.Create();
    Input := TNNetVolume.Create(128, 1, 1);
    Target := TNNetVolume.Create(512, 1, 1);
    try
      // Build network: 128 inputs -> 512 outputs
      NN.AddLayer(TNNetInput.Create(128));
      Layer := TNNetFullConnectLinear.Create(512);
      NN.AddLayer(Layer);

      // Initialize OpenCL for the network
      NN.EnableOpenCL(platform_id, device_id);

      // Set batch mode to enable OpenCL backprop
      NN.SetBatchUpdate(true);
      NN.SetLearningRate(0.001, 0.0);  // Lower learning rate for stability

      // Set input and target
      for i := 0 to Input.Size - 1 do
        Input.Raw[i] := (i mod 10) * 0.1;
      for i := 0 to Target.Size - 1 do
      begin
        if i mod 2 = 0 then
          Target.Raw[i] := 1.0
        else
          Target.Raw[i] := 0.0;
      end;

      // Get initial weight sum
      InitialWeightSum := 0;
      for i := 0 to Layer.Neurons.Count - 1 do
        InitialWeightSum := InitialWeightSum + Layer.Neurons[i].Weights.GetSum();

      // Forward pass
      NN.Compute(Input);
      ErrorBefore := 0;
      for i := 0 to Target.Size - 1 do
        ErrorBefore := ErrorBefore + Abs(NN.GetLastLayer.Output.Raw[i] - Target.Raw[i]);

      // Train for a few epochs
      for Epoch := 1 to 10 do
      begin
        NN.Compute(Input);
        NN.Backpropagate(Target);
        NN.UpdateWeights();
      end;

      // Get final weight sum
      FinalWeightSum := 0;
      for i := 0 to Layer.Neurons.Count - 1 do
        FinalWeightSum := FinalWeightSum + Layer.Neurons[i].Weights.GetSum();

      // Forward pass again with updated weights
      NN.Compute(Input);
      ErrorAfter := 0;
      for i := 0 to Target.Size - 1 do
        ErrorAfter := ErrorAfter + Abs(NN.GetLastLayer.Output.Raw[i] - Target.Raw[i]);

      // Verify weights changed significantly
      AssertTrue('Weights should change after backprop',
        Abs(FinalWeightSum - InitialWeightSum) > 1.0);

      // Verify error decreased (convergence)
      AssertTrue('Error should decrease after backprop (network converging)',
        ErrorAfter < ErrorBefore);

      NN.DisableOpenCL();
    finally
      NN.Free;
      Input.Free;
      Target.Free;
    end;
  finally
    EasyCL.Free;
  end;
end;

procedure TTestNeuralOpenCL.TestFullConnectBackpropagateOpenCLVsCPU;
var
  NN_OpenCL, NN_CPU: TNNet;
  Input, Target: TNNetVolume;
  EasyCL: TEasyOpenCL;
  platform_id: cl_platform_id;
  device_id: cl_device_id;
  i, Epoch: integer;
  OpenCLWeightSum, CPUWeightSum, WeightDiff: TNeuralFloat;
  OpenCLError, CPUError, InitialError: TNeuralFloat;
begin
  // Initialize OpenCL
  EasyCL := TEasyOpenCL.Create();
  try
    if EasyCL.GetPlatformCount() = 0 then
    begin
      Ignore('No OpenCL platforms available - skipping test');
      Exit;
    end;

    // Set platform
    EasyCL.SetCurrentPlatform(EasyCL.PlatformIds[0]);

    if EasyCL.GetDeviceCount() = 0 then
    begin
      Ignore('No OpenCL devices available - skipping test');
      Exit;
    end;

    // Set device
    EasyCL.SetCurrentDevice(EasyCL.Devices[0]);

    // Get platform and device IDs
    platform_id := EasyCL.CurrentPlatform;
    device_id := EasyCL.CurrentDevice;

    // Create two identical networks with proper size for OpenCL
    NN_OpenCL := TNNet.Create();
    NN_CPU := TNNet.Create();
    Input := TNNetVolume.Create(128, 1, 1);
    Target := TNNetVolume.Create(512, 1, 1);
    try
      // Build identical networks: 128 inputs -> 512 outputs
      NN_OpenCL.AddLayer(TNNetInput.Create(128));
      NN_OpenCL.AddLayer(TNNetFullConnectLinear.Create(512));

      NN_CPU.AddLayer(TNNetInput.Create(128));
      NN_CPU.AddLayer(TNNetFullConnectLinear.Create(512));

      // Synchronize weights from OpenCL to CPU network
      NN_CPU.Layers[1].LoadDataFromString(NN_OpenCL.Layers[1].SaveDataToString());

      // Enable OpenCL for one network
      NN_OpenCL.EnableOpenCL(platform_id, device_id);
      NN_OpenCL.SetBatchUpdate(true);
      NN_OpenCL.SetLearningRate(0.001, 0.0);  // Lower learning rate for stability

      // CPU network with same settings
      NN_CPU.SetBatchUpdate(true);
      NN_CPU.SetLearningRate(0.001, 0.0);

      // Set input and target
      for i := 0 to Input.Size - 1 do
        Input.Raw[i] := (i mod 10) * 0.1;
      for i := 0 to Target.Size - 1 do
      begin
        if i mod 2 = 0 then
          Target.Raw[i] := 1.0
        else
          Target.Raw[i] := 0.0;
      end;

      // Get initial error
      NN_CPU.Compute(Input);
      InitialError := 0;
      for i := 0 to Target.Size - 1 do
        InitialError := InitialError + Abs(NN_CPU.GetLastLayer.Output.Raw[i] - Target.Raw[i]);

      // Train both networks
      for Epoch := 1 to 10 do
      begin
        // OpenCL network
        NN_OpenCL.Compute(Input);
        NN_OpenCL.Backpropagate(Target);
        NN_OpenCL.UpdateWeights();

        // CPU network
        NN_CPU.Compute(Input);
        NN_CPU.Backpropagate(Target);
        NN_CPU.UpdateWeights();
      end;

      // Compare final weights
      OpenCLWeightSum := 0;
      CPUWeightSum := 0;
      for i := 0 to 511 do
      begin
        OpenCLWeightSum := OpenCLWeightSum + TNNetFullConnectLinear(NN_OpenCL.Layers[1]).Neurons[i].Weights.GetSum();
        CPUWeightSum := CPUWeightSum + TNNetFullConnectLinear(NN_CPU.Layers[1]).Neurons[i].Weights.GetSum();
      end;

      WeightDiff := Abs(OpenCLWeightSum - CPUWeightSum);

      // Compare final errors
      NN_OpenCL.Compute(Input);
      NN_CPU.Compute(Input);

      OpenCLError := 0;
      CPUError := 0;
      for i := 0 to Target.Size - 1 do
      begin
        OpenCLError := OpenCLError + Abs(NN_OpenCL.GetLastLayer.Output.Raw[i] - Target.Raw[i]);
        CPUError := CPUError + Abs(NN_CPU.GetLastLayer.Output.Raw[i] - Target.Raw[i]);
      end;

      // Weights should be nearly identical (< 0.01% relative error)
      AssertTrue('OpenCL and CPU weights should be nearly identical',
        (WeightDiff / Abs(OpenCLWeightSum)) < 0.0001);

      // Both should converge
      AssertTrue('OpenCL network should converge',
        OpenCLError < InitialError);
      AssertTrue('CPU network should converge',
        CPUError < InitialError);

      // Final errors should be nearly identical
      AssertTrue('OpenCL and CPU final errors should be nearly identical',
        (Abs(OpenCLError - CPUError) / CPUError) < 0.001);

      NN_OpenCL.DisableOpenCL();
    finally
      NN_OpenCL.Free;
      NN_CPU.Free;
      Input.Free;
      Target.Free;
    end;
  finally
    EasyCL.Free;
  end;
end;

procedure TTestNeuralOpenCL.TestFullConnectBackpropagateOpenCLBatchMode;
var
  NN: TNNet;
  Input, Target: TNNetVolume;
  EasyCL: TEasyOpenCL;
  platform_id: cl_platform_id;
  device_id: cl_device_id;
  InitialError, FinalError: TNeuralFloat;
  InitialWeightSum, FinalWeightSum: TNeuralFloat;
  Epoch, i: integer;
  Layer: TNNetFullConnectLinear;
begin
  // Initialize OpenCL
  EasyCL := TEasyOpenCL.Create();
  try
    if EasyCL.GetPlatformCount() = 0 then
    begin
      Ignore('No OpenCL platforms available - skipping test');
      Exit;
    end;

    // Set platform
    EasyCL.SetCurrentPlatform(EasyCL.PlatformIds[0]);

    if EasyCL.GetDeviceCount() = 0 then
    begin
      Ignore('No OpenCL devices available - skipping test');
      Exit;
    end;

    // Set device
    EasyCL.SetCurrentDevice(EasyCL.Devices[0]);

    // Get platform and device IDs
    platform_id := EasyCL.CurrentPlatform;
    device_id := EasyCL.CurrentDevice;

    NN := TNNet.Create();
    Input := TNNetVolume.Create(128, 1, 1);
    Target := TNNetVolume.Create(512, 1, 1);
    try
      // Build network with proper size for OpenCL
      NN.AddLayer(TNNetInput.Create(128));
      Layer := TNNetFullConnectLinear.Create(512);
      NN.AddLayer(Layer);

      // Enable OpenCL
      NN.EnableOpenCL(platform_id, device_id);
      NN.SetBatchUpdate(true);
      NN.SetLearningRate(0.001, 0.0);  // Lower learning rate for stability

      // Set input and target
      for i := 0 to Input.Size - 1 do
        Input.Raw[i] := (i mod 10) * 0.1;
      for i := 0 to Target.Size - 1 do
      begin
        if i mod 2 = 0 then
          Target.Raw[i] := 1.0
        else
          Target.Raw[i] := 0.0;
      end;

      // Get initial weight sum
      InitialWeightSum := 0;
      for i := 0 to Layer.Neurons.Count - 1 do
        InitialWeightSum := InitialWeightSum + Layer.Neurons[i].Weights.GetSum();

      // Initial error
      NN.Compute(Input);
      InitialError := 0;
      for i := 0 to Target.Size - 1 do
        InitialError := InitialError + Abs(NN.GetLastLayer.Output.Raw[i] - Target.Raw[i]);

      // Train for multiple epochs
      for Epoch := 1 to 20 do
      begin
        NN.Compute(Input);
        NN.Backpropagate(Target);
        NN.UpdateWeights();
      end;

      // Get final weight sum
      FinalWeightSum := 0;
      for i := 0 to Layer.Neurons.Count - 1 do
        FinalWeightSum := FinalWeightSum + Layer.Neurons[i].Weights.GetSum();

      // Final error
      NN.Compute(Input);
      FinalError := 0;
      for i := 0 to Target.Size - 1 do
        FinalError := FinalError + Abs(NN.GetLastLayer.Output.Raw[i] - Target.Raw[i]);

      // Network should learn and reduce error
      AssertTrue('Network should converge with OpenCL backprop',
        FinalError < InitialError);
      AssertTrue('Final error should be significantly reduced',
        FinalError < InitialError * 0.8);

      // Weights should change substantially
      AssertTrue('Weights should change during training',
        Abs(FinalWeightSum - InitialWeightSum) > 1.0);

      NN.DisableOpenCL();
    finally
      NN.Free;
      Input.Free;
      Target.Free;
    end;
  finally
    EasyCL.Free;
  end;
end;

{$ENDIF}

initialization
  {$IFDEF OpenCL}
  RegisterTest(TTestNeuralOpenCL);
  {$ENDIF}

end.
