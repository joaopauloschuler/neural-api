# Gradient Ascent Example
<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/gradientascent.jpg" height="192">

It's usually very hard to understand neuron by neuron how a neural network dedicated to image classification internally works. 
One technique used to help with the understanding about what individual neurons represent is called Gradient Ascent.

In this technique, an arbitrary neuron is required to activate and then the same backpropagation method used for learning is
applied to an input image producing an image that this neuron expects to see.

To be able to run this example, you'll need to load an already trained neural network file and then select the layer you intend to visualise.

This is the API method used for an arbitrary neuron backpropagation (Gradient Ascent):
```
procedure TNNet.BackpropagateFromLayerAndNeuron(LayerIdx, NeuronIdx: integer; Error: TNeuralFloat);
```
