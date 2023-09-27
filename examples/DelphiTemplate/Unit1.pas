unit Unit1;

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes, System.Variants,
  FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics, FMX.Dialogs,
  FMX.Controls.Presentation, FMX.StdCtrls,
  // Neural specifc files.
  neuralnetwork, neuralvolume, neuraldatasets, neuralfit, neuralthread;

// In Delphi, in project options:
// * At compiler, search path (-U), you'll  add the "neural" folder: ..\..\neural\
// * Still at the compiler, set the final output directory (-E) to: ..\..\bin\x86_64-win64\bin\
// * In "generate console application", set it to true.

// In your "uses" section, include:
//  neuralnetwork, neuralvolume, neuraldatasets, neuralfit, neuralthread;

type
  TForm1 = class(TForm)
    Button1: TButton;
    Button2: TButton;
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;

implementation


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

  procedure RunSimpleLearning();
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

    for EpochCnt := 1 to 1200 do
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
            ' Inputs: ', cs_inputs[Cnt][0]:3:1,', ' ,cs_inputs[Cnt][1]:3:1,
            '   Output:', pOutPut.Raw[0]:5:2,' ',
            ' - Training/Desired Output: ', vOutput[cnt][0]:5:2,' '
          );
      end;

      if EpochCnt mod 100 = 0 then
      begin
        WriteLn('');
      end;

    end;

    NN.DebugWeights();                                    // Display the final weights of the network

    pOutPut.Free;                                         // Free the memory allocated for output
    NN.Free;                                              // Free the memory allocated for the network

  end;

  procedure RunNeuralNetwork;
  var
    NN: TNNet;
    NeuralFit: TNeuralImageFit;
    ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes: TNNetVolumeList;
  begin
    if not CheckCIFARFile() then
    begin
      exit;
    end;
    WriteLn('Creating Neural Network...');
    NN := TNNet.Create();
    NN.AddLayer([
      TNNetInput.Create(32, 32, 3),
      TNNetConvolutionLinear.Create({Features=}64, {FeatureSize=}5, {Padding=}2, {Stride=}1, {SuppressBias=}1),
      TNNetMaxPool.Create(4),
      TNNetMovingStdNormalization.Create(),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetConvolutionReLU.Create({Features=}64, {FeatureSize=}3, {Padding=}1, {Stride=}1, {SuppressBias=}1),
      TNNetDropout.Create(0.5),
      TNNetMaxPool.Create(2),
      TNNetFullConnectLinear.Create(10),
      TNNetSoftMax.Create()
    ]);
    NN.DebugStructure();
    CreateCifar10Volumes(ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes);

    NeuralFit := TNeuralImageFit.Create;
    NeuralFit.FileNameBase := 'SimpleImageClassifier-'+IntToStr(GetProcessId());
    NeuralFit.InitialLearningRate := 0.001;
    NeuralFit.LearningRateDecay := 0.01;
    NeuralFit.StaircaseEpochs := 10;
    NeuralFit.Inertia := 0.9;
    NeuralFit.L2Decay := 0;
    NeuralFit.Fit(NN, ImgTrainingVolumes, ImgValidationVolumes, ImgTestVolumes, {NumClasses=}10, {batchsize=}64, {epochs=}50);
    NeuralFit.Free;

    NN.Free;
    ImgTestVolumes.Free;
    ImgValidationVolumes.Free;
    ImgTrainingVolumes.Free;
  end;


{$R *.fmx}

procedure TForm1.Button1Click(Sender: TObject);
begin
  RunNeuralNetwork;
end;

procedure TForm1.Button2Click(Sender: TObject);
begin
  RunSimpleLearning;
end;

end.
