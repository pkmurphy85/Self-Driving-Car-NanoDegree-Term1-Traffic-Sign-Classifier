
# **Traffic Sign Recognition**

## Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[allsigns]: ./WriteUpImages/43signs.png
[origTrainBar]: ./WriteUpImages/OrigTrainBar.png
[validBar]: ./WriteUpImages/ValidationBar.png
[testBar]: ./WriteUpImages/TestBar.png
[transformations]: ./WriteUpImages/Transofrmations.png

[augment]: ./WriteUpImages/Augment.png
[augment2]: ./WriteUpImages/Augment2.png


[newImages]: ./WriteUpImages/NewImages.png


### Data Set Summary & Exploration

Below is a statistical summary of the traffic
sign data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Below is an example image from each of the 43 classes in the data set.

![alt text][allsigns]

All of the signs are generally centered in the image and roughly the same size. However, the light intensity varies greatly across different samples.

Below is a bar chart with the number of samples for each class in the training set. As you can see the data set is largley unbalanced. Many classes have more than 1000 examples while others have less than 250 examples. This imbalance may cause the network to be more sensitive to images from over represented classes.

![alt text][origTrainBar]

Below are bar charts with the number of samples for each class in both the validation and test sets. Over represented classes in the training set are also over represented in the validation and test sets.

![alt text][validBar]
![alt text][testBar]

In order to overcome small sample sizes for certain classes I augmented the training set , which I will discuss below.

### Design and Test a Model Architecture

#### 1. Pre-processing

The first step in my model involves a pre-processing stage. I decided to augment the training data with jittered data as described in Pierre Sermanet and Yann LeCun's [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). In this paper pre-processing involved randomly adding transfromed versions of the original traning set to form a jittered dat set. The transforms they employed inclded randomly pertrubing in position ([-2,2] pixels), rotation ([-15,+15] degrees), and scale ([0.9,1.1] ratio). In my pre-processing stage images were randomly pertrubed in position ([-2,2] pixels), rotation ([-15,+15] degrees), brightness ([1.0, 2.5] gamma correction), and gaussian blur (5x5 kernal). Below is an example of each transformation.

![alt text][transformations]

I tested two data augmentation approaches. First, I tested randomly augmenting images from each class until there were at least 1500 samples for each class. This resulted in a more balanced data set. A bar chart with the result of this augmentation is below.

![alt text][augment2]

My second augmentation approach involved addiding 4 augmented versions of each image to the training set. The data set is still unbalanced however, there are now a suitable number of samples for the under represented classes.

![alt text][augment]

In my testing I found adding 4 augmented versions of each image in pre-processing led to more accurate results than  my approach to balance the data set. After augmentation I now had 173,995 training imgages (up from 34,799)

After augmenting the data set each image was converted to grayscale. Sermanet and LeCun achieved very high accuracy (98.97%) primarily using the grayscale component of each image. I focused on using the grayscale version of each image in my model to reduce processing and computation time. 

Finally, I normalized the image data. Pixel data was scaled from [0, 255] to [-1, 1] using the formula: (pixel - 128)/ 128 as suggested in class. This is a good approximation for image normalization.


#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| Relu					| Activation									|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x32	|
| Relu                  | Activation                                    |
| Max pooling	      	| 2x2 stride,  outputs 5x5x32    				|
| Fully connected		| 800 to 400   									|
| Dropout    	      	| Keep probability 35%             				|
| Fully connected		| 400 to 200   									|
| Dropout    	      	| Keep probability 35%             				|
| Fully connected		| 200 to 43   									|



My model uses two convolutional layers which incorporate max pooling and relu activation. The convolutions use a 5x5 filter size. This is the same filter size used in the LeNet architecture which was the starting point for my network testing. After convolution a 2x2 max pooling is performed. Pooling makes the convolution more robust against image noise and decreases the amount of computation necessary. Finally a relu activation function is applied to introduce non-linearities. This is repeated in the second convolution layer. After the convolution layers come three fully conneted layers. The fully connected layers reduce the size from 800 to 400, 400 to 200, and finally 200 to 43. Dropout was pefromed afted the first two fully connected layers to prevent against overfitting, using a keep probability of 35%. The outputs of the network are our logits. The logits are then fed to the function tf.nn.softmax_cross_entropy_with_logits() to convert to a one-hot encoding scheme.



#### 3. Training Parameters

For my model I used a learning rate of 0.0001, a batch size of 128, and 100 epochs. As in the LeNet architecture, the Adam optimizer was used. These numbers were chosen largely by trial and error. Initial network weights were populated usnig a truncated normal distribution with hyper parameters mu = 0 and sigma = 0.1. In addition dropout was employed after the first two fully connected layers with a keep probability of 35%.

#### 4. Approach

I began by using the LeNet implementation as was recommended in class. In my first attempts I used RGB images. After reading Sermanet and LeCun's Traffisc Sign Classifier paper I switched to grayscale images. Without tweaking the LeNet architecture I was unable to get validation accuracy over 90%. Next I started to modify the architecture, specifically the convolutional layers (filter size, pooling size, number of conv layers, etc), but still failed to increase accuracy above 90%.

Next I decided to employ data augmentation. At this point I was still running on my CPU. Given how long pre-processing and training were now taking, I decided to switch over to a GPU. Specifically I started using a g2.2xlarge instance on Amazon Web Services as recommended in calss. My biggest regret with this project is that I did not switch to  AWS sooner, as the processing speed was significantly improved! After adding augmentation I was able to increase accuracy over 93% for the first time. This highlighted for me just how important augmentation can be. Next I started to compare training set accuracy against validation set accuracy. At this stage my training accuracy was much higher than my validation accuracy, a sign that my model was overfitting. I decided to implement dropout to help prevent this. I tried implenting dropout in various sections of the network (after both convolutional and fully connected layers) as was described in Srivastava, Hinton, Krizhevsky, Sutskever, and Salakhutdinov's [dropout paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf). I did not see improvement with dropout after the convolutional layers so I finally settled on dropout after the first two fully connected layers with a keep probabiltity of 35%.

I aslo tried feeding the output of first convolutional layer to the first fully connected layer, as described by Sermanet and LeCun, but did not see improvements with this method.  

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.4%
* test set accuracy of 97.2%

### Test a Model on New Images

#### 1. Visualization of New Images

Here are eight German traffic signs that I found on the web:

1. Speed Limit 30 km/h
2. Road Work
3. Speed Limit 70 km/h
4. Speed Limit 60 km/h
5. General Caution
6. Speed Limit 100 km/h
7. Ahead Only
8. No Entry

![alt text][newImages]

It is difficult to see in these images because they have been resized to 32x32, but many of them are stock images with watermarks over the traffic signs. These watermarks may make it difficult to classify these signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed Limit 30 km/h	| Speed Limit 30 km/h   									|
| Road Work     		| Road Work  										|
| Speed Limit 70 km/h	| Speed Limit 70 km/h											|
| Speed Limit 60 km/h   | Speed Limit 30 km/h 					 				|
| General Caution		| General Caution     							|
| Speed Limit 100 km/h	| Speed Limit 70 km/h	   							|
| Ahead Only			| Ahead Only     							|
| No Entry		        | No Entry      							|

The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. The model was able to achieve an accuracy of 96.x% on the test set. I'm a bit disappointed in the lower accuracy on new images, however the new data set is extremly small, I'm not sure any meaningful conclusions can be drawn form this set. A cursory glance shows the model seems to have difficulty distinguishing different speed limit signs. This makes sense as these signs are extremly similar.

#### 3. Model Certainty - Softmax Probabilities


Below are the top five softmax probabilites for each new image:

First Image: Speed Limit 30 km/h. The model is relatively certain this is a Speed Limit 30 km/h sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 0.88         			| Speed Limit 30 km/h   						|
| 0.12     				| Speed limit (50km/h) 							|
| < 0.01				| Speed limit (80km/h)							|
| < 0.01      			| Keep right					 				|
| < 0.01			    | Wild animals crossing    						|

Second Image: Road Work. The model is 100% certain this is a Road Work sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| Road Work   									|
| 0.00     				| Bumpy road									|
| 0.00 					| Bicycles crossing								|
| 0.00       			| Double curve				 				    |
| 0.00 				    | Road narrows on the right   					|

Third Image: Speed Limit 70 km/h. The model is 100% certain this is a Speed Limit 70 km/h sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| Speed Limit 70 km/h  							|
| 0.00     				| Speed limit (20km/h)							|
| 0.00 					| Speed limit (30km/h)			    			|
| 0.00       			| General caution			 				    |
| 0.00 				    | Keep left   						        	|

Fourth Image: Speed Limit 60 km/h. The model is 100% certain this is a Speed Limit 60 km/h sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| Speed limit (60km/h)  						|
| 0.00     				| Speed limit (80km/h)							|
| 0.00 					| Speed limit (50km/h)							|
| 0.00       			| Stop			 			                 	|
| 0.00 				    | Speed limit (30km/h)   						|

Fifth Image: General Caution. The model is 100% certain this is a General Caution sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| General caution								|
| 0.00     				| Speed limit (20km/h)							|
| 0.00 					| Speed limit (30km/h)							|
| 0.00       			| Speed limit (50km/h)	 			           	|
| 0.00 				    | Speed limit (60km/h)  						|

Sixth Image: Speed Limit 100 km/h. The model is 100% certain this is a Speed Limit 100 km/h sign. However, in this instance the model is incorrect. The actual sign is not even in the top 5 probabilities. The model predicts 4 other speed limit signs as the top 4 choices. This means the model may have trouble distiguishing certain speed limit signs.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| Speed limit (30km/h)							|
| 0.00     				| Speed limit (70km/h)							|
| 0.00 					| Speed limit (50km/h)						    |
| 0.00       			| Speed limit (80km/h)				            |
| 0.00 				    | Priority road							        |

Seventh Image: Ahead Only. The model is 100% certain this is a Ahead Only sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| Ahead only									|
| 0.00     				| Turn left ahead							    |
| 0.00 					| Speed limit (20km/h)							|
| 0.00       			| Speed limit (30km/h)	 				        |
| 0.00 				    | Speed limit (50km/h)							|

Eigth Image:  No Entry. The model is 100% certain this is a No Entry sign.

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00        			| No entry								        |
| 0.00     				| Stop							                |
| 0.00 					| No passing								    |
| 0.00       			| Turn left ahead 				                |
| 0.00 				    | Yield				                    		|

Overall the model classified 7/8 images correctly, for an accuracy of 87.5%. The model had an accuracy of 97.2% on the test data set. I believe the extremely small sample size is responsible for the decrease in accuracy.



```python

```
