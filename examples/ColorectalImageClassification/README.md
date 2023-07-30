# Colorectal Cancer Image Classification

This source code example contains an implementation of a neural network-based image classifier for the Colorectal Cancer Dataset. The dataset used for this project can be found at the following link:

- [Colorectal Cancer Dataset](https://zenodo.org/record/53169/)
- [Download Link](https://zenodo.org/record/53169/files/Kather_texture_2016_image_tiles_5000.zip?download=1)
- [TensorFlow Dataset Catalog](https://www.tensorflow.org/datasets/catalog/colorectal_histology)

## Overview

The purpose of this source code example is to classify colorectal cancer images into different classes using a Convolutional Neural Network (CNN). The source code provided in this repository is written in Pascal (Free Pascal dialect) and uses the Neural API library for neural network operations.

## Prerequisites

To run this image classifier, you need the following:

- Free Pascal Compiler (FPC).
- Neural API library and its dependencies (can be found at [joaopauloschuler/neural-api](https://github.com/joaopauloschuler/neural-api)).

## Running

The neural network model will be created, and the training process will start. The model will be trained on the Colorectal Cancer Dataset, and the progress will be displayed during the training.

After the training is complete, the model's performance on the validation and test datasets will be evaluated, and the results will be displayed.

Note: If you encounter any issues related to memory constraints, you can adjust the `ProportionToLoad` variable in the source code to a smaller value.
