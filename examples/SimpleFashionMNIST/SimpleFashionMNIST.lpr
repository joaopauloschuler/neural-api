program SimpleFashionMNIST;
(*
 Coded by Joao Paulo Schwarz Schuler.
 https://github.com/joaopauloschuler/neural-api
 -----------------------------------------------
 The code shows an example of training and fitting a convolutional
 neural network (CNN) using the Fashion MNIST dataset. It creates a neural
 network with specific layers and configurations, loads the fashion MNIST data,
 and then trains the network using the provided data. The code also sets
 various parameters for training, such as learning rate, decay, and batch size.
*)
{$mode objfpc}{$H+}

uses {$IFDEF UNIX} {$IFDEF UseCThreads}
  cthreads, {$ENDIF} {$ENDIF}
  Classes, SysUtils, CustApp, neuralnetwork, neuralvolume, Math, neuraldatasets, neuralfit;

type
  TTestCNNAlgo = class(TCustomApplication)
  protected
    procedure DoRun; override;
  end;

  // Implementation of the TTestCNNAlgo class
  procedure TTestCNNAlgo.DoRun;
  var
    NN: TNNet;  // Neural network object
    NeuralFit: TNeuralImageFit;  // Object for training and fitting the neural network
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;  // Lists of training, validation, and test image volumes
  begin
    // Checking if the MNIST files exist and loading the data
    if Not(CheckMNISTFile('train', {IsFashion=}true)) or
      Not(CheckMNISTFile('t10k', {IsFashion=}true)) then
    begin
      Terminate;
      exit;
    end;

    WriteLn('Creating Neural Network...');

    // Creating the neural network with specific layers and configurations
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(28, 28, 1),  // Input layer for 28x28 grayscale images
      TNNetConvolutionLinear.Create(64, 5, 2, 1, 1),  // Convolutional layer with linear activation
      TNNetMaxPool.Create(4),  // Max pooling layer
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),  // Convolutional layer with ReLU activation
      TNNetConvolutionReLU.Create(64, 3, 1, 1, 1),  // Convolutional layer with ReLU activation
      TNNetFullConnectReLU.Create(32),  // Fully connected layer with ReLU activation
      TNNetFullConnectReLU.Create(32),  // Fully connected layer with ReLU activation
      TNNetFullConnectLinear.Create(10),  // Fully connected layer with linear activation
      TNNetSoftMax.Create()  // Softmax layer for classification
    ]);

    // Creating the training, validation, and test image volumes from the fashion MNIST files
    CreateMNISTVolumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes,
      'train', 't10k', {Verbose=}true, {IsFashion=}true);

    // Creating and configuring the NeuralFit object for training the neural network
    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleFashionMNIST';
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0.00001;
    NeuralFit.HasFlipX := true;
    NeuralFit.HasFlipY := false;
    NeuralFit.MaxCropSize := 4;

    // Training and fitting the neural network using the provided data
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}128, {epochs=}50);

    NeuralFit.Free;  // Freeing the NeuralFit object

    NN.Free;  // Freeing the neural network object
    ImgTestVolumes.Free;  // Freeing the test data volumes
    ImgValidationVolumes.Free;  // Freeing the validation data volumes
    ImgTrainingVolumes.Free;  // Freeing the training data volumes
    Terminate;
  end;

var
  Application: TTestCNNAlgo;
begin
  Application := TTestCNNAlgo.Create(nil);  // Creating an instance of the TTestCNNAlgo class
  Application.Title:='Simple Fashion MNIST Classification Example';  // Setting the application title
  Application.Run;  // Running the application
  Application.Free;  // Freeing the application instance
end.

