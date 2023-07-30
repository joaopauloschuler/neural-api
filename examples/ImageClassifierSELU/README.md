# Image Classifier with SELU Activation Function

The `ImageClassifierSELU` program demonstrates the usage of a neural network with SELU activation for image classification. It is implemented using the [joaopauloschuler/neural-api](https://github.com/joaopauloschuler/neural-api) library.

The program will create a neural network with the following layers:

1. Input layer with dimensions 32x32x3 (width, height, channels).
2. Convolutional layer with 64 filters, kernel size 5x5, stride 2, and padding 1x1. It uses SELU activation and is initialized with basic patterns.
3. Max Pooling layer with a pool size of 4x4.
4. SELU activation layer.
5. Moving Standard Deviation Normalization layer.
6. Three convolutional layers with 64 filters, kernel size 3x3, stride 1, and padding 1x1. All these layers use SELU activation.
7. One convolutional layers with 64 filters, kernel size 3x3, stride 1, and padding 1x1 without activation function.
8. A Dropout layer with a dropout rate of 0.5.
9. A Max Pooling layer with a pool size of 2x2.
10. Another SELU activation layer.
11. A Fully Connected layer with 10 output neurons (for 10 classes in CIFAR-10).
12. A Softmax activation layer for classification.

The program then proceeds to fit the neural network to the CIFAR-10 dataset using the following hyperparameters:

- Initial Learning Rate: 0.0004
- Learning Rate Decay: 0.03
- Staircase Epochs: 10
- Inertia: 0.9
- L2 Regularization Decay: 0.00001
- Number of Classes: 10
- Batch Size: 64
- Number of Epochs: 50

## Output

During execution, the program will output the progress of the training process, including the loss and accuracy values for each epoch. After training is complete, the neural network will be evaluated on the test dataset, and the final accuracy on the test set will be displayed.

## Acknowledgments

If you encounter any issues or have suggestions for improvements, feel free to create an issue or contribute to the project through pull requests.

Enjoy using the image classifier with SELU activation! Happy coding!
