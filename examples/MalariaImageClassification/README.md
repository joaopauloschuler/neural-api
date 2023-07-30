# Malaria Cell Infection Classification

This repository contains an implementation for classifying malaria-infected cell images using a Convolutional Neural Network (CNN). The goal is to detect whether a cell image is infected with malaria or not.

## Dataset

The dataset used for this project can be obtained from the following sources:

1. Download the dataset from the NIH-LHCBC Data Archive: [Malaria Cell Images](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip).
2. Alternatively, the dataset can also be found on TensorFlow Datasets catalog: [Malaria Dataset](https://www.tensorflow.org/datasets/catalog/malaria).

Please download and extract the dataset before running the code.

## Requirements

To run the code, you will need the following software:

- Free Pascal Compiler (FPC) and Lazarus
- neural-api library, available at [neural-api GitHub repository](https://github.com/joaopauloschuler/neural-api)

## Neural Network Architecture

The Convolutional Neural Network (CNN) architecture used for this classification task is as follows:

1. Input Layer: 64x64x3 (RGB image)
2. Convolution Layer: 64 features, kernel size 5x5, padding 4, stride 1
3. Max Pooling Layer: 2x2
4. Moving Std Normalization Layer
5. Convolution Layer: 64 features, kernel size 3x3, padding 1, stride 1 (ReLU activation)
6. Convolution Layer: 64 features, kernel size 3x3, padding 1, stride 1 (ReLU activation)
7. Max Pooling Layer: 2x2
8. Convolution Layer: 64 features, kernel size 3x3, padding 1, stride 1 (ReLU activation)
9. Convolution Layer: 64 features, kernel size 3x3, padding 1, stride 1 (ReLU activation)
10. Convolution Layer: 64 features, kernel size 3x3, padding 1, stride 2 (ReLU activation)
11. Dropout Layer: 50% dropout rate
12. Max Pooling Layer: 2x2
13. Fully Connected Layer: 2 classes (Output layer)
14. Softmax Activation Layer

## Training and Evaluation

The dataset is divided into three sets: Training, Validation, and Test sets. The code will load a proportion (specified by `ProportionToLoad`) of the dataset into memory for training and evaluation. You can adjust this value based on the available system memory.

The training process includes the following hyperparameters:

- Initial Learning Rate: 0.001
- Learning Rate Decay: 0.01
- Staircase Epochs: 10
- Inertia: 0.9
- L2 Decay: 0.00001
- Batch Size: 64
- Number of Epochs: 50

## Results

After training, the model will be evaluated on the test set, and the accuracy and other relevant metrics will be displayed.

Note: Depending on the dataset size and hardware capabilities, training the model might take a considerable amount of time.

## Acknowledgments

This implementation was coded by Joao Paulo Schwarz Schuler. The neural-api library used in this project can be found at [neural-api GitHub repository](https://github.com/joaopauloschuler/neural-api).

For any issues related to the code or implementation, please refer to the [neural-api repository](https://github.com/joaopauloschuler/neural-api).
