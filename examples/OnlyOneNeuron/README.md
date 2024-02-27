# Only One Neuron - Source Code Examples

To make the learning of neural networks very easy, this folder contains 2 source code examples with neural networks that contain only one neuron:
* Compute the boolean OR operation.
* Compute the linear function 2*x + 3*y + 4.

## Computing the Boolean OR Operation

### The Training Data
The training data consists of all possible input combinations for the OR operation and their corresponding outputs:
```
- False OR False = False
- False OR True  = True
- True  OR False = True
- True  OR True  = True
```
The numeric values for True and False are defined with:
```
const
  cs_false = 0.1;                          // Encoding for "false" value
  cs_true  = 0.8;                          // Encoding for "true" value
  cs_threshold = (cs_false + cs_true) / 2; // Threshold for false/true (neuronal activation)
```

Then, the actual data structures with input and output of the OR operation are given with:
```
type
  // Define the input and output types for training data
  TBackInput  = array[0..3] of array[0..1] of TNeuralFloat;  // Input data for OR operation
  TBackOutput = array[0..3] of array[0..0] of TNeuralFloat;  // Expected output for OR operation

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
```



### Architecture of the Neural Network

The neural network consists of only 2 layers:
* an input layer with two inputs (representing the two inputs of the OR operation)
* and a single output neuron that provides the result. It uses a fully connected architecture with ReLU activation function.

The above neural network is created with:
```
  NN := TNNet.Create();
  // Create the neural network layers
  NN.AddLayer(TNNetInput.Create(2));                     // Input layer with 2 inputs
  NN.AddLayer(TNNetFullConnectLinear.Create(1));         // Single neuron layer connected to both inputs from the previous layer.
```

