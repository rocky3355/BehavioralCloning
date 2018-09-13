# Behavioral Cloning
This project aims for teaching a car to drive on a simple track. For this, test data is collected by driving on the track manually, while capturing the images and the steering angles. The thrust will be a constant value. The images and angles are then fed into a Tensorflow model, that will then learn which features in the current view of the car lead to certain steering angles.

## 1. The model
The model consists of the layers shown below. It has mostly been copied from the Udacity lectures. However, two dropout layers have been added in order to reduce overfitting. Mean squared error and Adam optimizer are used for the training process.

|Layer        |Description                                                |
| ----------- | --------------------------------------------------------- |
|Cropping2D   | Input: 160x320x3, Output: 65x320x3                        |
|Lambda       | Normalizes the images from -0.5 to 0.5                    |
|Convolution2D| Filters: 24, Kernel: 5x5, Subsample: 2x2, Activation: Relu|
|Convolution2D| Filters: 36, Kernel: 5x5, Subsample: 2x2, Activation: Relu|
|Convolution2D| Filters: 48, Kernel: 5x5, Subsample: 2x2, Activation: Relu|
|Dropout      | Keeping probability: 0.5                                  |
|Convolution2D| Filters: 64, Kernel: 3x3, Activation: Relu                |
|Convolution2D| Filters: 64, Kernel: 3x3, Activation: Relu                |
|Dropout      | Keeping probability: 0.5                                  |
|Flatten      | Flattens the data to a one-dimensional array              |
|Dense        | Output: 100                                               |
|Dense        | Output: 50                                                |
|Dense        | Output: 10                                                |
|Dense        | Output: 1                                                 |
