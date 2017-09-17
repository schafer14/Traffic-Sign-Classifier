# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---
### Writeup / README

### Data Set Summary & Exploration

The dataset consists of just over 50,000 color images of 32x32 resolution categorized into 43 classes. Most classes had approximately 500 images while some classes had more then 2000 and others had just 180. Getting more images from the underrepresentative classes could be beneficial, alternativly those images could be altered to artificially create more data in under represented classes; however, that approach was not used in this project. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes in training= 43

Here are two images from the training set. There are more examples in the ipython notebook. 

![Sign 1](https://raw.githubusercontent.com/schafer14/Traffic-Sign-Classifier/master/examples/sign1.png)
![Sign 2](https://raw.githubusercontent.com/schafer14/Traffic-Sign-Classifier/master/examples/sign2.png)

### Design and Test a Model Architecture

I did not convert the images to gray scale because color has meaning in traffic signs. Red is a regulatory, yellow is a warning, green is a guide... so using grayscale images is taking a way that information from the network. Instead I normalized each dimension of each pixel to the range (0.1, 0.9). 

My final model was based on the LeNet architecture with one additional convolutional layer and one additional fully connected layer. Along with that I used a drop out of 19% after each max pooling layer. I found that these two additions were incredibly beneficial to my validation accuracy

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Drop Out 19%          |                                               |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Drop Out 19%          |                                               |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x32     |
| RELU					|												|
| Drop Out 19%          |                                               |
| Flatten               |                                               |
| 3x Fully connected    |                                               |

To train the model used the Adam Optimizer, with a learning rate of 0.001 a batch size of 512 and 100 epochs. I tried many different variations of these hyper parameters but did not find them to be particularly helpful. I found that altering the models architecture and drop out rates where much more beneficial. 

To train the model I repeated an interation for each epoch involving splitting up the dataset into batches then training the model on each batch. At the end of each epoch I calculated the accuracy on the validation set without drop out. My model achieved 90% accuracy on the validation set after approximately 10 epochs. After that there was a lot of variation in how fast the accuracy improved. On some runs it would get caught in local minimums and never reach an accuracy above 94% while on other runs it achieved accuracy of 97% in some epochs. The model generally leveled off after about 30 epochs, but I let the maximum number of epochs be much higher just in case it was stuck in a local minimum and got  out after more epochs. I attempted to increase the learning rate but this was uneffective as the model would pass over minimums to frequently. 

The final validation score of the model I submitted was 96.4%.

 

### Test a Model on New Images

The german traffic signs I found on the internet were better quality then the images in the training set. In addition to those five I included two images to try and trick the network. One was a artificially generated image (not taken with a camera) and the other was of a sign that was not in the dataset. 
Of the five legitimate pictures the network categorized each one correctly. The artifically generated image was missclassified. The probabilities form the softmax were above 99% for the five images, while the two other images had lower probabilities below 90%.

Here are two German traffic signs that I found on the web:

![Sign 3](https://raw.githubusercontent.com/schafer14/Traffic-Sign-Classifier/master/examples/sign3.png)
![Sign 4](https://raw.githubusercontent.com/schafer14/Traffic-Sign-Classifier/master/examples/sign4.png)


Here are the results of the prediction:

| Image			        |     Prediction	        |  Probability | Was Fake |
|:---------------------:|:-------------------------:|:------------:|:--------:|
| Right-of-way at the next intersection      		| Right-of-way at the next intersection			    | >99% | 0|
|Ahead only  			|Ahead only				| >99% | 0|
| Stop					| Stop						| >99% | 0|
| Priority road	      		| Priority road				| > 99% | 0|
| Roundabout mandatory		| Roundabout mandatory      		| >99% | 0|
| No passing | Priority road | 87% | 1| 
| No car | Right-of-way at the next intersection | 78% | 1 |

The model correctly classified all real images giving it an accuracy of 100%. I am not concerned about how well it preforms on fake data, but I just found that interesting. 

When I tested on rough 12,000 previously unseen examples I achieved an accuracy of 94%. I ran this test at the very end and retrained the network so some all of
my previous results use the weights trained from a previous run. Most of the data is fairly similar though. 