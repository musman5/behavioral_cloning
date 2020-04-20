import csv
import cv2
import numpy as np

lines = []

# Open csv file to read file path and measurements
with open('../data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read image and measurements
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        source_path =source_path.replace('\\','/')
        #print(source_path)    
        tokens = source_path.split('/')
        filename = tokens[-1]                           
        #print(filename)
        current_path = '../data/data/IMG/' + filename
        #print(current_path)
        image = cv2.imread(current_path)
        #print(image)
        images.append(image)
    # Add correction factor for left and right images
    correction = 0.2
    measurement = float(line[3])
    #print(measurement)
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)

# Add augmented data by flipping the image and measurement    
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append (image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

# Build the model, add lambda layer to normalize and crop the upper and lower part of image
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))

# LeNet model commented out below. We will use Navida model as it provides better output

#model.add(Convolution2D(6, (5, 5), activation = 'relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(16, (5, 5), activation = 'relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(24, (5, 5), subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(36, (5, 5), subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(48, (5, 5), subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# use adam optimizer and mean square error for loss reduction
model.compile(loss = 'mse', optimizer = 'adam')
# Train the model, split 20 percent data for validation
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5)

print('done')
# save the model for use in simulator
model.save('model.h5')
