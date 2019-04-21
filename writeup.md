# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/preprocessing.png "preprocessing"
[image4]: ./test_images/image1.jpg "Traffic Sign 1"
[image5]: ./test_images/image2.jpg "Traffic Sign 2"
[image6]: ./test_images/image3.jpg "Traffic Sign 3"
[image7]: ./test_images/image4.jpg "Traffic Sign 4"
[image8]: ./test_images/image5.jpg "Traffic Sign 5"
[image9]: ./test_images/image6.jpg "Traffic Sign 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to add more data to the dataset, because the number of images of different classes in the original dataset are imbalanced. 

To add more data to the the data set, I tested the following techniques:
- image shifting
- image rotation
- gaussian noise

Here is an example of using those technique to generate augmented images:

![alt text][image3]

After testing, I decided to use image rotation to generate artificial image data, and targeting those classes that has less training images. 

I also decided to convert the images to grayscale because it since to reduce overfitting and also in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) it mentions grayscaling helps to increase the accuracy of model.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data for the optimizer to perform better gradient descent based optimization.  


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x1x6  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout               | dropout probabity 0.5                         |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5x6x16  | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Dropout               | dropout probabity 0.5                         |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		| input = 400, output = 200						|
| RELU					|												|
| Dropout               | dropout probabity 0.5                         |
| Fully connected		| input = 200, output = 86						|
| RELU					|												|
| Dropout               | dropout probabity 0.5                         |
| Fully connected		| input = 86, output = 43						|
| Softmax				| normalize to probability distribution							            		|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer and the following hyperparameters:
- EPOCHES = 30
- BATCH_SIZES = 128
- learning rate = 0.001


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992
* validation set accuracy of 0.949
* test set accuracy of 

The iterative approach I used was as follow:

* I chose to use Yann LeCun's LeNet architecture since it has been proven to be useful for image classification.  
* The original model tended to overfit the training set as I noticed that the validation set accuracy was around 0.89 to 0.91 while the training set accuracy was already 0.99.
* To increase accuracy, I added more augmented data to the training dataset. The validation accuracy increased a little and can now sometimes reach 0.93 validation accuracy. 
* To reduce overfitting, I added dropout to the model, and after that the validation set accuracy has significantly improved. With the current model architecture and augmented data, the validation set accuarcy is stably around 0.95.
* I also tuned the size of fully connected layer a little, because the number of classes is different from the original LeNet model.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way at the next section    		| Right of way at the next section      | 
| No vehicles     			                | No vehicles   						|
| 70 km/h				                   	| 50 km/h								|
| Yield	      		                        | Yield					 				|
| Wild animal crossing			            | Wild animal crossing					|
| Go straight or right                      | Go straight or right                  |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. Although the test accuracy is 0.945, the model still fails to recognize "70 km/h" sign.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook.

For the first image, the model is very sure that this is a "Right of Way" sign (probability of 0.999), and the image does contain a "Right of Way". The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9997             	| Right of way 									|
| .00029                | Beware of ice/snow                            |
| .0000007     			| Pedestrians									| 
| .0000003              | Double curve                                  |   
| .0000001				| Children crossing								|

The second image also had simliar result. Model has no trouble to classify it correctly.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9947             	| 15 No vehicles								|
| .0037                 | 12 Priority road                              |
| .0009     			| 13 Yield						    			| 
| .0005                 | 9  No passing                                 |   
| .00007				| 2	Speed limit (30km/h)						|

The third image is the one that model has trouble to classify, as we can see from the softmax that the probability are all pretty low. This may be because of the low resolution of picture from bad down sampling.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .2352             	| Speed limit (50km/h)			    			|
| .2015                 | Speed limit (30km/h)                          |
| .1344     			| Roundabout mandatory			    			| 
| .0877                 | Speed limit (20km/h)                          |   
| .0595				    | Speed limit (70km/h)				    		|

The model performed well on the rest of three images. The top 1 probabilities are all above 96%.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


