# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ../sample_images/cnn_architecture_used.png "Model Visualization"
[image2]: ../sample_images/center_lane_driving.jpg "Center Lane Training"
[image3]: ../sample_images/starting_at_left_side.jpg "Recovery from left side of the road by starting at left side"
[image4]: ../sample_images/recovering_from_left_side.jpg "Recovery from left side to center of the road"
[image5]: ../sample_images/starting_at_right_side.jpg "Recovery from right side of the road by starting at right side"
[image6]: ../sample_images/recovering_from_right_side.jpg "Recovery from right side to center of the road"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 video file for car driving the lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filters sizes and depths between 24 and 64 (model.py lines 57-77) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 59). 
Code snippet of the model is below:
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
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

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5)

#### 2. Attempts to reduce overfitting in the model

The model contains data augmentation techniques in order to reduce overfitting (model.py lines 38-46). The image is flipped and measurement is also adjusted to have negative value of orignal value.
Moreover the left and right camera images are used to gerneralize the data.
The image is cropped from top and bottom to feed only the related data to the model.
To reduce overfitting dropout is introduced after convolution and before flatten layer.
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 80).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I added both clockwise and counter clockwise track data. Also for recovery i added data where i drove the car from side of the lane to center of the lane especially on curves.
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the well known architecture and tune it to fit our model. 
First i tried convolutional neural network similar to LeNet model but this model did not provide nice and smooth output. The car was driving out of lane randomly and abruptly.
Next i used Nvidia architecture as it has been proven successful in self-driving car tasks. This architecture was recommended in udacity class and it fits best for our driving requirements.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. So i added dropout of 0.25 meaning every fourth image will be dropped from output and will not be used in further or backtrack calculation. Then i adjusted the number of epochs to prevent overfitting as well. I dropped my epoch to just 5. 

Then step by step by putting augumented data to train out model, using image cropping and using large data set i was able to get better output so that car drives in the center of the road. I added error recovery around curves and additional anticlockwise data to train the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track in my LeNet model. In my second model using Nivida architecture to improve the driving behavior in this case, i added correction for left and right camera images. I fine tune the parameters by checking training and validation loss and set epoch to a lower value of 5. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-77) consisted of a convolution neural network with the following layers and layer sizes.
5x5 filter used for first three convolutions and then 3x3 filter is applied on last two convolutions.
To prevent over fitting dropout layer is added with value 0.25. i.e. dropping every fourth image.
Then we flatten the output and we have three fully connected layers with depth of 100, 50 and 10.
Finally we get single out for the steering angle.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][../sample_images/cnn_architecture_used.png]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one and half laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][../sample_images/center_lane_driving.jpg]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to center of the track. These images show what a recovery looks like starting from right side of the track to center of the road and starting from left side of the road recovery to center of the road : (I specially add recovery training data on the curves, as there is higher chance on curves to drag off the road)

![alt text][../sample_images/starting_at_left_side.jpg]
![alt text][../sample_images/recovering_from_left_side.jpg]
![alt text][../sample_images/starting_at_right_side.jpg]
![alt text][../sample_images/recovering_from_right_side.jpg]

Then I repeated this process on track in clockwise direction in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would minimize the vehicle turning on left or right side due to layout of specific track. 

After the collection process, I had 5340 number of data points. 
I read left and right images as well and added correction factor of 0.2. For right camera image we subtracted correction factor and for left camera image we added correction factor which is 0.2
I then preprocessed this data by flipping the image and measurement angle.
I combined all the images including left, right, center and augmented images.
Then i used this combined data in the model where i normalize and also cropped the image from top and bottom.

Total orignal images: 5340
Total images (with center, right , left and augmented images): 32040
20% of validation split: 6408
Total number of images in training set: 25632

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by minimum training and validation loss. Dropout layer is added with value 0.25 to prevent overfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

** I removed the training images from workspace to ~/opt/data/ before submitting the project as i was not able to submit project with training data.
The video file is also included in the project workspace named video.mp4, which is captured at framerate of 60fps. The file is updated based on new code including the dropout layer.