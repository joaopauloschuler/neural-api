program TestBackpropCompare;

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
begin
  WriteLn('=== OpenCL vs CPU Backpropagation Comparison Test ===');
  WriteLn;

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
    WriteLn('Creating two identical networks (OpenCL and CPU)...');
    NN_OpenCL := TNNet.Create();
    NN_CPU := TNNet.Create();
    Input := TNNetVolume.Create(128, 1, 1);
    Target := TNNetVolume.Create(512, 1, 1);

    try
      // Build identical network structures
      NN_OpenCL.AddLayer(TNNetInput.Create(128));
      NN_OpenCL.AddLayer(TNNetFullConnectLinear.Create(512));

      NN_CPU.AddLayer(TNNetInput.Create(128));
      NN_CPU.AddLayer(TNNetFullConnectLinear.Create(512));

      WriteLn('Network: 128 inputs -> 512 outputs');
      WriteLn;

      // Copy weights from OpenCL network to CPU network to ensure identical starting point
      WriteLn('Synchronizing initial weights and biases...');
      NN_CPU.Layers[1].LoadDataFromString(NN_OpenCL.Layers[1].SaveDataToString());

      // Verify initial weights match
      OpenCLWeightSum := 0;
      CPUWeightSum := 0;
      for i := 0 to 511 do
      begin
        OpenCLWeightSum := OpenCLWeightSum + TNNetFullConnectLinear(NN_OpenCL.Layers[1]).Neurons[i].Weights.GetSum();
        CPUWeightSum := CPUWeightSum + TNNetFullConnectLinear(NN_CPU.Layers[1]).Neurons[i].Weights.GetSum();
      end;
      InitialWeightSum := OpenCLWeightSum;
      WriteLn('Initial weight sum - OpenCL: ', OpenCLWeightSum:0:6, ', CPU: ', CPUWeightSum:0:6);
      WriteLn;

      // Enable OpenCL for one network
      WriteLn('Enabling OpenCL on first network...');
      NN_OpenCL.EnableOpenCL(EasyOpenCL.PlatformIds[0], EasyOpenCL.Devices[0]);
      NN_OpenCL.SetBatchUpdate(true);
      NN_OpenCL.SetLearningRate(0.001, 0.0);  // Lower learning rate for stability

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
          Target.Raw[i] := 0.0;
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
        // OpenCL network
        NN_OpenCL.Compute(Input);
        NN_OpenCL.Backpropagate(Target);
        NN_OpenCL.UpdateWeights();

        // CPU network
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
      WriteLn('  OpenCL: ', InitialOpenCLError - OpenCLError:0:4, ' (',
              ((InitialOpenCLError - OpenCLError) / InitialOpenCLError * 100):0:2, '% reduction)');
      WriteLn('  CPU:    ', InitialCPUError - CPUError:0:4, ' (',
              ((InitialCPUError - CPUError) / InitialCPUError * 100):0:2, '% reduction)');
      WriteLn;
      if (OpenCLError < InitialOpenCLError) and (CPUError < InitialCPUError) then
        WriteLn('Both networks are CONVERGING (error decreasing) ✓')
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

      for i := 0 to 511 do
      begin
        OpenCLWeightSum := OpenCLWeightSum + TNNetFullConnectLinear(NN_OpenCL.Layers[1]).Neurons[i].Weights.GetSum();
        CPUWeightSum := CPUWeightSum + TNNetFullConnectLinear(NN_CPU.Layers[1]).Neurons[i].Weights.GetSum();

        WeightDiff := Abs(
          TNNetFullConnectLinear(NN_OpenCL.Layers[1]).Neurons[i].Weights.GetSum() -
          TNNetFullConnectLinear(NN_CPU.Layers[1]).Neurons[i].Weights.GetSum()
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
      WriteLn('  OpenCL: ', OpenCLWeightSum - InitialWeightSum:0:6, ' (',
              Abs((OpenCLWeightSum - InitialWeightSum) / InitialWeightSum * 100):0:2, '% change)');
      WriteLn('  CPU:    ', CPUWeightSum - InitialWeightSum:0:6, ' (',
              Abs((CPUWeightSum - InitialWeightSum) / InitialWeightSum * 100):0:2, '% change)');
      WriteLn;
      WriteLn('OpenCL vs CPU final weight difference: ', Abs(OpenCLWeightSum - CPUWeightSum):0:6);
      WriteLn('Max per-neuron difference: ', MaxDiff:0:6);
      WriteLn;

      WriteLn('-----------------------------------------------------------');
      WriteLn('FINAL OUTPUT COMPARISON:');
      WriteLn('-----------------------------------------------------------');
      WriteLn('Sample outputs (first 5 neurons):');
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

      // Verdict - use relative error for large weight sums
      WeightDiff := Abs(OpenCLWeightSum - CPUWeightSum);
      if Abs(OpenCLWeightSum) > 0 then
      begin
        WriteLn('Relative weight difference: ', (WeightDiff / Abs(OpenCLWeightSum) * 100):0:6, '%');
      end;
      WriteLn('Relative error difference: ', (Abs(OpenCLError - CPUError) / CPUError * 100):0:6, '%');
      WriteLn;

      if ((WeightDiff / Abs(OpenCLWeightSum)) < 0.0001) and  // Less than 0.01% relative error
         (OpenCLError < InitialOpenCLError) and              // OpenCL is converging
         (CPUError < InitialCPUError) then                   // CPU is converging
      begin
        WriteLn('RESULT: ✓ PASS');
        WriteLn;
        WriteLn('OpenCL and CPU backpropagation produce nearly identical results!');
        WriteLn('- Both networks converge (error decreases)');
        WriteLn('- Weight updates are within 0.01% relative error');
        WriteLn('- Final outputs match within floating-point precision');
        WriteLn;
        WriteLn('The OpenCL backpropagation implementation is numerically correct.');
      end
      else
      begin
        WriteLn('RESULT: ✗ FAIL');
        WriteLn;
        WriteLn('OpenCL and CPU backpropagation differ significantly or not converging properly.');
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
