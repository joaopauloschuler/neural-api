program SimpleMNist;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math,
  neuraldatasets, neuralfit;

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
  end;

  procedure TTestCNNAlgo.DoRun;
  var
    NN: TNNet; // Neural network object
    NeuralFit: TNeuralImageFit; // Object for neural network fitting
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList; // Volumes for training, validation, and testing
  begin
    // Check if MNIST files exist
    if not (CheckMNISTFile('train')) or not (CheckMNISTFile('t10k')) then
    begin
      Terminate;
      Exit; // Exit the procedure if MNIST files are not found
    end;

    WriteLn('Creating Neural Network...');
    NN := TNNet.Create(); // Create an instance of the neural network

    // Define the layers of the neural network
    NN.AddLayer([
      TNNetInput.Create(28, 28, 1),
      TNNetConvolutionLinear.Create(32, 5, 2, 1, 1),
      TNNetMaxPool.Create(4),
      TNNetConvolutionReLU.Create(32, 3, 1, 1, 1),
      TNNetConvolutionReLU.Create(32, 3, 1, 1, 1),
      TNNetFullConnectReLU.Create(32),
      TNNetFullConnectReLU.Create(32),
      TNNetDropout.Create(0.2),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);

    // Create MNIST volumes for training, validation, and testing
    CreateMNISTVolumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, 'train', 't10k');

    // Configure the neural network fitting
    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleMNist';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.HasFlipX := False;
    NeuralFit.HasFlipY := False;
    NeuralFit.MaxCropSize := 4;

    // Fit the neural network using the training, validation, and testing volumes
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}20);
    NeuralFit.Free;

    // Clean up resources
    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
    Terminate;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil); // Create an instance of TTestCNNAlgo
  Application.Title := 'MNist Classification Example'; // Set the application title
  Application.Run; // Run the application
  Application.Free; // Free the application instance
end.
