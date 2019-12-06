# Gradient Ascent Example
<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/gradientascent.jpg" height="192">

It's usually very hard to understand neuron by neuron how a neural network dedicated to image classification internally works. 
One technique used to help with the understanding about what individual neurons represent is called Gradient Ascent.

In this technique, an arbitrary neuron is required to activate and then the same backpropagation method used for learning is
applied to an input image producing an image that this neuron expects to see.

To be able to run this example, you'll need to load an already trained neural network file and then select the layer you intend to visualize.

Deeper convolutional layers tend to produce more complex patterns. Above image was produced from the a first convolutional layer. The following image was produced from a third convolutional layer. Notice that patterns are more complex.

<img src="https://github.com/joaopauloschuler/neural-api/blob/master/docs/gradientascent3layer.jpg" height="192">

This is the API method used for an arbitrary neuron backpropagation (Gradient Ascent):
```
procedure TNNet.BackpropagateFromLayerAndNeuron(LayerIdx, NeuronIdx: integer; Error: TNeuralFloat);
```

You can find more about Gradient Ascent at:
* [Lecture 12: Visualizing and Understanding - CS231n - Stanford](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf)
* [Understanding Neural Networks Through Deep Visualization](http://yosinski.com/deepvis)
