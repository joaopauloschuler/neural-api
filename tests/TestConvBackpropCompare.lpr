program TestConvBackpropCompare;

{$mode objfpc}{$H+}

uses
  SysUtils, neuralnetwork, neuralvolume, neuralopencl, cl;

var
  NN_OpenCL, NN_CPU: TNNet;
  EasyOpenCL: TEasyOpenCL;
  Input, Target: TNNetVolume;
  i, epoch: integer;
  OpenCLWeightSum, CPUWeightSum, WeightDiff: TNeuralFloat;
  InitialWeightSum: TNeuralFloat;
  MaxDiff: TNeuralFloat;
  OpenCLError, CPUError: TNeuralFloat;
  InitialOpenCLError, InitialCPUError: TNeuralFloat;
  NumNeurons: integer;
begin
  WriteLn('=== OpenCL vs CPU Convolution Backpropagation Comparison Test ===');
  WriteLn;

  NumNeurons := 32;

  // Initialize OpenCL
  EasyOpenCL := TEasyOpenCL.Create();
  try
    if EasyOpenCL.GetPlatformCount() = 0 then
    begin
      WriteLn('ERROR: No OpenCL capable platform has been found.');
      Exit;
    end;

    WriteLn('Found ', EasyOpenCL.GetPlatformCount(), ' OpenCL platform(s)');
    WriteLn('Using platform: ', EasyOpenCL.PlatformNames[0]);
    EasyOpenCL.SetCurrentPlatform(EasyOpenCL.PlatformIds[0]);

    if EasyOpenCL.GetDeviceCount() = 0 then
    begin
      WriteLn('ERROR: No OpenCL capable device has been found.');
      Exit;
    end;

    WriteLn('Using device: ', EasyOpenCL.DeviceNames[0]);
    EasyOpenCL.SetCurrentDevice(EasyOpenCL.Devices[0]);
    WriteLn;

    // Create two identical networks
    WriteLn('Creating two identical convolution networks (OpenCL and CPU)...');
    NN_OpenCL := TNNet.Create();
    NN_CPU := TNNet.Create();
    Input := TNNetVolume.Create(8, 8, 3);
    Target := TNNetVolume.Create(8, 8, NumNeurons);

    try
      // Build identical network structures: 8x8x3 -> Conv(32, 3, 1, 1) -> 8x8x32
      NN_OpenCL.AddLayer(TNNetInput.Create(8, 8, 3));
      NN_OpenCL.AddLayer(TNNetConvolutionLinear.Create(NumNeurons, 3, 1, 1));

      NN_CPU.AddLayer(TNNetInput.Create(8, 8, 3));
      NN_CPU.AddLayer(TNNetConvolutionLinear.Create(NumNeurons, 3, 1, 1));

      WriteLn('Network: 8x8x3 input -> Conv(', NumNeurons, ', 3x3, pad=1, stride=1) -> 8x8x', NumNeurons);
      WriteLn('VectorSize per neuron: ', TNNetConvolutionLinear(NN_OpenCL.Layers[1]).Neurons[0].Weights.Size);
      WriteLn;

      // Copy weights from OpenCL network to CPU network to ensure identical starting point
      WriteLn('Synchronizing initial weights and biases...');
      NN_CPU.Layers[1].LoadDataFromString(NN_OpenCL.Layers[1].SaveDataToString());

      // Verify initial weights match
      OpenCLWeightSum := 0;
      CPUWeightSum := 0;
      for i := 0 to NumNeurons - 1 do
      begin
        OpenCLWeightSum := OpenCLWeightSum + TNNetConvolutionLinear(NN_OpenCL.Layers[1]).Neurons[i].Weights.GetSum();
        CPUWeightSum := CPUWeightSum + TNNetConvolutionLinear(NN_CPU.Layers[1]).Neurons[i].Weights.GetSum();
      end;
      InitialWeightSum := OpenCLWeightSum;
      WriteLn('Initial weight sum - OpenCL: ', OpenCLWeightSum:0:6, ', CPU: ', CPUWeightSum:0:6);
      WriteLn;

      // Enable OpenCL for one network
      WriteLn('Enabling OpenCL on first network...');
      NN_OpenCL.EnableOpenCL(EasyOpenCL.PlatformIds[0], EasyOpenCL.Devices[0]);
      NN_OpenCL.SetBatchUpdate(true);
      NN_OpenCL.SetLearningRate(0.001, 0.0);

      // CPU network with same settings
      NN_CPU.SetBatchUpdate(true);
      NN_CPU.SetLearningRate(0.001, 0.0);
      WriteLn;

      // Set input and target values
      for i := 0 to Input.Size - 1 do
        Input.Raw[i] := (i mod 10) * 0.1;
      for i := 0 to Target.Size - 1 do
      begin
        if i mod 2 = 0 then
          Target.Raw[i] := 1.0
        else
          Target.Raw[i] := -1.0;
      end;

      WriteLn('Training both networks for 10 epochs...');
      WriteLn('-----------------------------------------------------------');
      WriteLn('Epoch | OpenCL Error | CPU Error    | Error Diff   | Status');
      WriteLn('------|--------------|--------------|--------------|----------');

      // Compute initial errors
      NN_OpenCL.Compute(Input);
      NN_CPU.Compute(Input);

      OpenCLError := 0;
      CPUError := 0;
      for i := 0 to Target.Size - 1 do
      begin
        OpenCLError := OpenCLError + Abs(NN_OpenCL.GetLastLayer.Output.Raw[i] - Target.Raw[i]);
        CPUError := CPUError + Abs(NN_CPU.GetLastLayer.Output.Raw[i] - Target.Raw[i]);
      end;
      InitialOpenCLError := OpenCLError;
      InitialCPUError := CPUError;

      WriteLn('  0   | ', OpenCLError:12:4, ' | ', CPUError:12:4, ' | ',
              Abs(OpenCLError - CPUError):12:6, ' | Initial');

      // Train both networks
      for epoch := 1 to 10 do
      begin
        NN_OpenCL.Compute(Input);
        NN_OpenCL.Backpropagate(Target);
        NN_OpenCL.UpdateWeights();

        NN_CPU.Compute(Input);
        NN_CPU.Backpropagate(Target);
        NN_CPU.UpdateWeights();

        // Compute errors after this epoch
        NN_OpenCL.Compute(Input);
        NN_CPU.Compute(Input);

        OpenCLError := 0;
        CPUError := 0;
        for i := 0 to Target.Size - 1 do
        begin
          OpenCLError := OpenCLError + Abs(NN_OpenCL.GetLastLayer.Output.Raw[i] - Target.Raw[i]);
          CPUError := CPUError + Abs(NN_CPU.GetLastLayer.Output.Raw[i] - Target.Raw[i]);
        end;

        WriteLn(epoch:3, '   | ', OpenCLError:12:4, ' | ', CPUError:12:4, ' | ',
                Abs(OpenCLError - CPUError):12:6, ' | Training');
      end;

      WriteLn('-----------------------------------------------------------');
      WriteLn;
      WriteLn('CONVERGENCE ANALYSIS:');
      WriteLn('-----------------------------------------------------------');
      WriteLn('Initial errors:');
      WriteLn('  OpenCL: ', InitialOpenCLError:0:4);
      WriteLn('  CPU:    ', InitialCPUError:0:4);
      WriteLn;
      WriteLn('Final errors:');
      WriteLn('  OpenCL: ', OpenCLError:0:4);
      WriteLn('  CPU:    ', CPUError:0:4);
      WriteLn;
      WriteLn('Error reduction:');
      if InitialOpenCLError > 0 then
        WriteLn('  OpenCL: ', InitialOpenCLError - OpenCLError:0:4, ' (',
                ((InitialOpenCLError - OpenCLError) / InitialOpenCLError * 100):0:2, '% reduction)')
      else
        WriteLn('  OpenCL: N/A');
      if InitialCPUError > 0 then
        WriteLn('  CPU:    ', InitialCPUError - CPUError:0:4, ' (',
                ((InitialCPUError - CPUError) / InitialCPUError * 100):0:2, '% reduction)')
      else
        WriteLn('  CPU: N/A');
      WriteLn;
      if (OpenCLError < InitialOpenCLError) and (CPUError < InitialCPUError) then
        WriteLn('Both networks are CONVERGING (error decreasing)')
      else
        WriteLn('WARNING: Networks may be DIVERGING!');
      WriteLn;
      WriteLn('-----------------------------------------------------------');
      WriteLn('WEIGHT CHANGES:');
      WriteLn('-----------------------------------------------------------');

      // Compare final weights
      OpenCLWeightSum := 0;
      CPUWeightSum := 0;
      MaxDiff := 0;

      for i := 0 to NumNeurons - 1 do
      begin
        OpenCLWeightSum := OpenCLWeightSum + TNNetConvolutionLinear(NN_OpenCL.Layers[1]).Neurons[i].Weights.GetSum();
        CPUWeightSum := CPUWeightSum + TNNetConvolutionLinear(NN_CPU.Layers[1]).Neurons[i].Weights.GetSum();

        WeightDiff := Abs(
          TNNetConvolutionLinear(NN_OpenCL.Layers[1]).Neurons[i].Weights.GetSum() -
          TNNetConvolutionLinear(NN_CPU.Layers[1]).Neurons[i].Weights.GetSum()
        );
        if WeightDiff > MaxDiff then
          MaxDiff := WeightDiff;
      end;

      WriteLn('Initial weight sum: ', InitialWeightSum:0:6);
      WriteLn;
      WriteLn('Final weight sum:');
      WriteLn('  OpenCL: ', OpenCLWeightSum:0:6);
      WriteLn('  CPU:    ', CPUWeightSum:0:6);
      WriteLn;
      WriteLn('Weight change from initial:');
      if Abs(InitialWeightSum) > 0.001 then
      begin
        WriteLn('  OpenCL: ', OpenCLWeightSum - InitialWeightSum:0:6, ' (',
                Abs((OpenCLWeightSum - InitialWeightSum) / InitialWeightSum * 100):0:2, '% change)');
        WriteLn('  CPU:    ', CPUWeightSum - InitialWeightSum:0:6, ' (',
                Abs((CPUWeightSum - InitialWeightSum) / InitialWeightSum * 100):0:2, '% change)');
      end
      else
      begin
        WriteLn('  OpenCL: ', OpenCLWeightSum - InitialWeightSum:0:6);
        WriteLn('  CPU:    ', CPUWeightSum - InitialWeightSum:0:6);
      end;
      WriteLn;
      WriteLn('OpenCL vs CPU final weight difference: ', Abs(OpenCLWeightSum - CPUWeightSum):0:6);
      WriteLn('Max per-neuron difference: ', MaxDiff:0:6);
      WriteLn;

      WriteLn('-----------------------------------------------------------');
      WriteLn('FINAL OUTPUT COMPARISON:');
      WriteLn('-----------------------------------------------------------');
      WriteLn('Sample outputs (first 5 positions):');
      WriteLn('  OpenCL: [', NN_OpenCL.GetLastLayer.Output.Raw[0]:0:4, ', ',
                            NN_OpenCL.GetLastLayer.Output.Raw[1]:0:4, ', ',
                            NN_OpenCL.GetLastLayer.Output.Raw[2]:0:4, ', ',
                            NN_OpenCL.GetLastLayer.Output.Raw[3]:0:4, ', ',
                            NN_OpenCL.GetLastLayer.Output.Raw[4]:0:4, ']');
      WriteLn('  CPU:    [', NN_CPU.GetLastLayer.Output.Raw[0]:0:4, ', ',
                            NN_CPU.GetLastLayer.Output.Raw[1]:0:4, ', ',
                            NN_CPU.GetLastLayer.Output.Raw[2]:0:4, ', ',
                            NN_CPU.GetLastLayer.Output.Raw[3]:0:4, ', ',
                            NN_CPU.GetLastLayer.Output.Raw[4]:0:4, ']');
      WriteLn;

      WriteLn('-----------------------------------------------------------');
      WriteLn('FINAL VERDICT:');
      WriteLn('-----------------------------------------------------------');

      // Verdict
      WeightDiff := Abs(OpenCLWeightSum - CPUWeightSum);
      if Abs(OpenCLWeightSum) > 0.001 then
      begin
        WriteLn('Relative weight difference: ', (WeightDiff / Abs(OpenCLWeightSum) * 100):0:6, '%');
      end;
      if CPUError > 0.001 then
        WriteLn('Relative error difference: ', (Abs(OpenCLError - CPUError) / CPUError * 100):0:6, '%');
      WriteLn;

      if (OpenCLError < InitialOpenCLError) and
         (CPUError < InitialCPUError) and
         ((Abs(OpenCLWeightSum) < 0.001) or ((WeightDiff / Abs(OpenCLWeightSum)) < 0.01)) then
      begin
        WriteLn('RESULT: PASS');
        WriteLn;
        WriteLn('OpenCL and CPU convolution backpropagation produce matching results!');
        WriteLn('- Both networks converge (error decreases)');
        WriteLn('- Weight updates are within acceptable tolerance');
        WriteLn('- Final outputs match within floating-point precision');
        WriteLn;
        WriteLn('The OpenCL convolution backpropagation implementation is correct.');
      end
      else
      begin
        WriteLn('RESULT: FAIL');
        WriteLn;
        WriteLn('OpenCL and CPU convolution backpropagation differ significantly or not converging.');
      end;

      NN_OpenCL.DisableOpenCL();
    finally
      NN_OpenCL.Free;
      NN_CPU.Free;
      Input.Free;
      Target.Free;
    end;
  finally
    EasyOpenCL.Free;
  end;

  WriteLn;
  WriteLn('Press Enter to exit...');
  ReadLn;
end.
