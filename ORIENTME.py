import sys

TRAINING_IMAGES_FOLDER = "C:/Users/user/Desktop/Machine Learning Project/ORIENTME/data/training/images/"
TRAINING_LABELS_PATH = "C:/Users/user/Desktop/Machine Learning Project/ORIENTME/data/training/labels.csv"
TEST_IMAGES_FOLDER = "C:/Users/user/Desktop/Machine Learning Project/ORIENTME/data/images"
SAMPLE_SUBMISSION_FILE_PATH = "C:/Users/user/Desktop/Machine Learning Project/ORIENTME/data/sample_submission.csv"

import os
import tqdm

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt


from PIL import Image


training_labels_df = pd.read_csv(TRAINING_LABELS_PATH)

def pre_process_data_X(image):
    """
    This file takes a loaded image and returns a particular 
    representation of the data point
    
    
    NOTE: This current baseline implements a **very** silly approach
    of representing every image by the mean RGB values for every image.
    
    You are encourage to try to alternate representations of the data,
    or figure out how to learn the best representation from the data ;)
    """
    im_array = np.array(im)
    mean_rgb = im_array.mean(axis=(0, 1))
    return mean_rgb


ALL_DATA = []

for _idx, row in tqdm.tqdm(training_labels_df.iterrows(), total=training_labels_df.shape[0]):
    filepath = os.path.join(
        TRAINING_IMAGES_FOLDER,
        row.filename
    )
    im = Image.open(filepath)
    
    data_X = pre_process_data_X(im)
    data_Y = [row.xRot]
    
    ALL_DATA.append((data_X, data_Y))
    
    
plt.figure(figsize=(20,20))
for i in range(16):
  filename,xRot = training_labels_df.iloc[i]
  filepath = os.path.join(
        TRAINING_IMAGES_FOLDER,
        filename
    )
  im = Image.open(filepath)
  plt.subplot(4,4,i+1)
  plt.axis('off')
  plt.title("xRot: %.3f"%(xRot))
  plt.imshow(im)
  
  
training_set, validation_set= train_test_split(ALL_DATA, test_size=0.2, random_state=42)

X_train, y_train = zip(*training_set)
X_val, y_val = zip(*validation_set)


X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

model = MLPRegressor(hidden_layer_sizes=[10, 10], verbose=True)

model.fit(X_train, y_train)


y_pred = model.predict(X_val)

print('Mean Absolute Error:', mean_absolute_error(y_val, y_pred))  
print('Mean Squared Error:', mean_squared_error(y_val, y_pred))  
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_val, y_pred)))


import glob

TEST_DATA = []
TEST_FILENAMES = []

for _test_image_path in tqdm.tqdm(glob.glob(os.path.join(TEST_IMAGES_FOLDER, "*.jpg"))):
    filename = os.path.basename(_test_image_path)
    im = Image.open(_test_image_path)
    
    data_X = pre_process_data_X(im)
    TEST_DATA.append(data_X)
    TEST_FILENAMES.append(filename)
    

TEST_DATA = np.array(TEST_DATA)

test_predictions = model.predict(TEST_DATA)

test_df = pd.DataFrame(test_predictions, columns=['xRot'])
test_df["filename"] = TEST_FILENAMES

test_df.to_csv('submission.csv', index=False)


np.savetxt("submission.csv", y_pred, delimiter=",")

myfile = pd.read_csv("submission.csv")
myfile.head()