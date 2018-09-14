# Behavioral Cloning
This project aims for teaching a car to drive on a simple track. For this, test data is collected by driving on the track manually, while capturing the images and the steering angles. The thrust will be a constant value. The images and angles are then fed into a Tensorflow model, that will then learn which features in the current view of the car lead to certain steering angles.

## 1. The model
The model consists of the layers shown below. It has mostly been copied from the Udacity lectures. However, two dropout layers have been added in order to reduce overfitting. Mean squared error and Adam optimizer are used for the training process. 20% auf the images have been used as validation set. The additional left and right images are also used for training, while adding/substracting a steering angle of 0.1 for them. Furthermore, the center image has been flipped vertically and the according steering angle has been negated, in order to create more test data.

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

## 2. Training strategy
The provided sample images already produced a decent model for keeping the vehicle on the road. Even recovering to the center line when I steered manually to the road edges (in autonomous mode) was working pretty good. However, there were two scenarios where the model tended to fail:

1. Sharp curves that are not marked with yellow lines
2. Not driving in a straight line over the bridge

For the first point, I assume that the fact of yellow lines dominating the track leads to a lack of training data for the sharp curves. For an overall improvement, I recorded the whole track once driving at the center line. I especially created some records for those sharp curves.

For the bridge, center line recovering did not work at all. The reason for this is, that the camera only sees the bridge borders in a straight way, as center line driving is employed for the records. Thus, I created several records solely for recovering on the bridge. In the video, the car still oscillates a little bit from one side to another, but before using the extra records, it just crashed into the borders.

## 3. Conclusion
In the video the car drives near the center line most of the time. The sharp curves and the bridge are still the biggest weak points in my model. But in the passage above, I showed that even with a little extra training, the behaviour can be massively improved already. Below I made some suggestions for further improving the model:

1. Extend the model architecture (i.e. change filter/kernel sizes, add extra layers)
2. Create more training data for the problematic parts of the track
3. Use the second track to collect training data
