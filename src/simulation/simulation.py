import cv2
import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
import scipy.misc

#print(tf.__version__)
#print(keras.__version__)

#provide model path here
model_path = r'C:\Projects\Steering Angle Prediction\model\model_final_DV2.h5'
model = load_model(model_path)

#use to pre-process the each frame
def img_preprocess(image):
  height, _, _ = image.shape #this returns height,width,channel
  image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
  image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  
  image = cv2.GaussianBlur(image, (3,3), 0)
  image = cv2.resize(image, (200,66)) # input image size (200,66) Nvidia model
  image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
  return image

#predicts the steering angle for each frame based on the trained model
def compute_steering_angle(frame):
    preprocessed = img_preprocess(frame)
    X = np.asarray([preprocessed])
    steering_angle = model.predict(X)[0]
    #print(X(0))
    return steering_angle

#load the steeing angle image
img = cv2.imread(r'C:\Projects\Steering Angle Prediction\assets\steering.png',0)
rows,cols = img.shape
#xcv2.imshow('img1',img)
#cv2.waitKey()

smoothed_angle = 0
i=0
while(cv2.waitKey(10) != ord('q')):
    #load the data set images one by one 
    full_image = cv2.imread(r'C:\Users\alok\Desktop\dataset2\driving_dataset\\' + str(i) + ".jpg")
    pred_angle = compute_steering_angle(full_image) * (180 / scipy.pi)
    print("Predicted steering angle: " + str(pred_angle) + " degrees")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))

    smoothed_angle += 0.2 * pow(abs((pred_angle - smoothed_angle)), 2.0 / 3.0) * (pred_angle - smoothed_angle) / abs(pred_angle - smoothed_angle)
    smoothed_angle = int(smoothed_angle);
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)

    i +=1

cv2.destroyAllWindows()
