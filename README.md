# Dual View Metal Pathlength Regression

This repository contains sources for pathlength regression in X-Ray images. 
The objective of this work is to evaluate if epipolar features are beneficial for model convergence and performance.
Thus, a model is trained which receives two X-Ray images and their correponding projection matrices as input and outputs the metal pathlengths in both views.
These images are acquired at roughly 90 degrees angular distance; e.g. a lateral and anterior-posterior projection.
To make use of the respective other view, an epipolar image translation layer (github.com/maxrohleder/FUME) is integrated into the model.

## Model Architecture

The proposed model architecture is derived from a typical U-Net architecture. The two input images are segmented simultaneously by an identical model. After each Down- and Upblock, the feature maps are translated onto the respective other view. The two similar models are trained with shared weights.

![](https://github.com/maxrohleder/dual-view-pathlength-regression/blob/assets/regression_model.png)

## Data

As input, this model receives two projection images in shape `(976, 976)` and two projection matrices in shape `(3, 4)`. 
One sample is thus a tuple of two numpy arrays: `((2, 976, 976), (2, 3, 4))`
