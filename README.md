# Steering Angle Prediction

Design and Implementation of a Convolutional Neural Network model that predicts steering angle for an autonomous vehicle. 

This project was part of my final year undergrad project.
The research paper that was written can be found [here](https://drive.google.com/file/d/1RKPuq98YTPZa111mJz611l6Urtr8T0sE/view?usp=sharing)

## Project Details
* The idea of this project was to train a CNN on a dataset comprising of around 45K frames taken from the dashboard of a car. Each frame had a steering angle value recorded against it.
* The dataset was further augmented to make it more dynamic. The augmentation technique used was blurring the frames, zooming the frames, altering the brightness of the frames, etc.
* Each frame was then pre-processed before training it. The region of interest was defined by changing the resolution, the color space was changed from RGB to YUV.
* For training, the dataset was divided into two parts namely training set and the validation set with the ratio of 4:1 respectively.
* The CNN was trained for 30 epochs wherein each epoch had a mixture of different data from the training set.
* The model was validated by calculating mean squared error & R2 accuracy.

## CNN notebook
The Google colab notebook can be found [here](https://colab.research.google.com/drive/1QwBx9Pvv8iSyPcG9LaTVBwFklZc_zs8Y?usp=sharing)

## Simulation
The simulation video can be found [here](https://drive.google.com/file/d/1q5Ql3sEdolk7bQ6FUDZt8IZAju5mhAKW/view?usp=sharing)


## Technologies

* Google colab
* Tensorflow, Keras
* Pandas, Numpy, OpenCV

