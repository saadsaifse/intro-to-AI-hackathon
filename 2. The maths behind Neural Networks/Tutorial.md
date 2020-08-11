# Theory behind Image Classification

```
Einav Grinberg, Muhammad Saad Saif, Anna Formaniuk
```

## Overview

```
● What is an image and how do we recognize images
● Different ways to recognize images (before ML and with ML and AI and stuffs; ML vs DL)
● What are Neural Networks
● What are CNNs
● 

```
---

# Will be shortened and simplified

# So now we kind of know what ML and DL are, so what do we do if we want to classify an image? 

## What even is an image?

A digital image is simply a collection of points on the screen. These points are called pixels and the whole screen is composed of such pixels. How many pixels does a screen contain varies in different devices and is defined by pixels per inch or PPI. Resolution of the screen or image defines how these pixels are spread through the screen. In an RGB (Red--Green-Blue) screen, each pixel can have any combination of RGB values that allows a pixel to have millions of colors. The more the pixels per inch, the clearer and sharper the image becomes. When looked at the screen by the naked eye, these pixels form a smooth image that we all see and like taking them for our social media profiles ;)

### How To Recognize Images?

Humans can easily recognize what is inside the image by just looking at the image itself. If they have previously seen the objects inside the image, they must be able to recognize the objects inside the image. But for computers, recognizing an image is way too different. To feel the pain of a computer, consider answering the following question. 

**How would you describe an image to someone or someone who is blind?**

- **Easy:** Using words that you may know
- **Medium:** From geometric primitives (lines, curves, shape, color, etc.)
- **Difficult:** From the raw pixels

To recognize an image, human brain has learnt from the vast number of past experiences of looking at the objects that we can recognize. Humans develop this ability to classify an image and tell whether there is a dog or a cat in the image. Computers only "see" images as the amount of red, blue, and green at each pixel. Everything else we want them to know, we would have had to describe in terms of pixels.

### Image Classification before Machine Learning

Early computer vision models relied on raw pixel data as the input to the model. However, as shown in Figure 2, raw pixel data alone doesn't provide a sufficiently stable representation to encompass the myriad variations of an object as captured in an image. The position of the object, background behind the object, ambient lighting, camera angle, and camera focus all can produce fluctuation in raw pixel data; these differences are significant enough that they cannot be corrected for by taking weighted averages of pixel RGB values.

![cat-variation](images/cat-variation.png)

To model objects more flexibly, classic computer vision models added new features derived from pixel data, such as color histograms, textures, and shapes. The downside of this approach was that feature engineering became a real burden, as there were so many inputs to tweak. For a cat classifier, which colors were most relevant? How flexible should the shape definitions be? Because features needed to be tuned so precisely, building robust models was quite challenging, and accuracy suffered.

### Introduction to image recognition and selected tools

**How would you describe an image to someone or someone who is blind?**

- **Easy:** Using words that you may know
- **Medium:** From geometric primitives (lines, curves, shape, color, etc.)
- **Difficult:** From the raw pixels

Well, just train a Neural Network :)

To recognize an image, human brain has learnt from the vast number of past experiences of looking at the objects that we can recognize. Humans develop this ability to classify an image and tell whether there is a dog or a cat in the image. Computers only "see" images as the amount of red, blue, and green at each pixel. Everything else we want them to know, we would have had to describe in terms of pixels.

Artificial intelligence takes a different approach. Instead of providing instructions, we provide examples. Above, we could show our robot thousands of labeled images of bread and thousands of labeled images of other objects and ask our robot to learn the difference. Our robot could then build its own program to identify new groups of pixels (images) as bread.

Machine and Deep learning both provide ways for computers to classify data such as images. In order to classify images, a model is trained with the sample/training images using Machine or Deep learning based models. 

**What is model anyway?**: In traditional programming your code compiles into a binary that is typically called a program. In machine learning, the item you create from the data and labels is called a model. In short, a model is a trained neural network which is trained using the provided labeled/unlabeled data. For image classification, we already have the labels of training images i.e., it is a supervised learning problem rather than an unsupervised one. We deal only with labeled data and supervised learning here.

### Machine Learning vs. Deep Learning For Image Classification
A valid question that arise here is should one use a Machine Learning based model or a Deep Learning based model? The answer to this question is better explained in [this video](https://www.youtube.com/watch?v=-SgkLEuhfbg).

The basic idea is that, in machine learning, we take many pictures of cats and dogs and come up with an algorithm to extract some features (e.g., edges, corners, etc) from within the images. This is called Feature Selection and there is a whole lots of ways to do that. Once we have extracted the features from the images, we train one of the machine learning models such as Support Vector Machines (SVM), k-nearest neighbor (kNN), and others using the extracted features. Once the model is trained, it knows how to classify dogs and cats and it can take any new (previously unseen) picture to analyze and classify them. The following image illustrated the process using machine learning.

![using-ml](images/using-ml.png)

While in deep learning, which is a sub-discipline of machine learning, training images can be directly def into the model or a network. The model extracts the features from the images on its own to learn to classify the images, as shown in the below image. 

![dl-process](images/dl-process.png)

![using-dl](images/using-dl.png)

Another way to understand the difference is using the below figure which is stating the same difference with different visuals.

![ml-vs-dl-flow](images/ml-vs-dl-flow.png)


With that important difference between machine and deep learning for image classification in mind, the question is which one should be used and under what circumstances.

#### Choosing Between Machine or Deep Learning

The decision comes down to the following two questions

- How big is the dataset?
- How much hardware resources are available? 

Typically, a deep learning based algorithm consumes a lot of hardware resources (GPUs) in order to extract the features from the images and learn from them. This is comparatively done with relatively large datasets than machine learning based models and also takes more time to train the model.

Therefore, if there is a less data and less hardware resources for deep learning, a machine learning based approach is more suitable, otherwise the deep learning based approach can also be considered.

Following image helps in choosing which one to use.

![ml-vs-dl-use](images/ml-vs-dl-stats.png)


## Elements of a Neural Network

### Layers

Neural networks are also called “stacked neural networks”, meaning networks composed of several layers. The layers are made of nodes. On the following image you can see an example of a simple neural network, that consists of the input layer, one hidden layer and the output layer, that has only two possible outcomes. The input layer consists of nodes that get their values directly from the data. Each next layer's output serves as the subsequent layer's input, thus **feeding forward** information. The output represents the combined input of all the nodes. 
If a neural network has more than 1 hidden layer, it is called deep.

!["Simple network" image](./images/mlp.png "source: A Beginner's Guide to Neural Networks and Deep Learning")

### Nodes

!["Node" image](./images/perceptron_node.png "source: A Beginner's Guide to Neural Networks and Deep Learning")

**Weights and inputs**

Similarly to the neurons in a human brain, the nodes are linked together and fire when receive enough stimuli from the other nodes. The nodes both process and store information. As seen on the image, the node takes inputs from the preceding nodes, combines them with a set of coefficients (weights), that either increase or decrease importance of each input, and sums it all up. 

**Activations and outputs**

Then the computed sum is then passed to the activation function, which decides what value in the range from 0 to 1 to store in the node as a result. The closer the value is to 1, the more "activated" it becomes.
The output then becomes the next layer's input or, if it's the output layer, is used to extract a prediction or a decision from the neural network. The most activated node is then the most probable outcome.

---

## Training a neural network

So to be able to classify something, we need the following elements: input data, weights, and an activation function. The first is provided, the last we choose from the available functions and to have the weights we need to train the network. To train it we can provide labels for each item in our training data, or just let the network find some patterns and features automatically, draw connections between them and distinguish various classes by itself. Training it on labeled data can be more performative, as the network will be comparing its results to the results we want it to achieve and adjusting the weights accordingly. This is done through forward propagation and back propagation.

The goal of training the network is to have the labels predicted by the network as close to the real labels as possible. In other words, we must minimize the difference between them, also called the error.
At first all the weights are initialized randomly. Each next step involves an error measurement and a slight update of the weights, as the network slowly learns from its mistakes and is repeated until the least possible error is achieved. A final collection of weights is then called a **model**.

This can be generalized as follows:

Input enters the network. The summary of inputs multiplied by the weights is passed throught he activation function and a set of guesses is made.

```
input * weight = guess
```

Then the guess is compared to the ground-truth about the data (the labels we provide), effectively asking “Did I get this right?”. This answer to this question is provided by what is called a **cost function**

```
ground truth - guess = error
```

The difference between the network’s guess and the ground truth is its error. The network measures that error, and walks the error back over its model, adjusting weights to the extent that they contributed to the error. Some details on this will be explained in the following chapter.

```
error * weight's contribution to error = adjustment
```

### Gradient Descent

To compare the guess with the ground truth and optimize the weights, "Gradient descent" is applied. Gradient basically represents how two or more variables relate to each other: in this case - the relationship between the network’s error and the weights. With the gradient it is possible to see how increasing or decreasing a weight by one step affects the error and then to choose the option that makes it smaller. This is done recursively for all the weights in the model and in the end the essence of learning in deep learning is nothing more than that: adjusting a model’s weights in response to the error it produces, until you can’t reduce the error any more. Going back through the network to adjust the weights is a technique called **backpropagation**.

!["Gradient descent" image](./images/gradient_descent_demystified.png "source: ML Glossary")


## Working of Convolutional Neural Network (CNN) for Image Classification 

A breakthrough in building models for image classification came with the discovery that a convolutional neural network (CNN) could be used to progressively extract higher- and higher-level representations of the image content. Instead of preprocessing the data to derive features like textures and shapes, a CNN takes just the image's raw pixel data as input and "learns" how to extract these features, and ultimately infer what object they constitute.

To start, the CNN receives an input feature map: a three-dimensional matrix where the size of the first two dimensions corresponds to the length and width of the images in pixels. The size of the third dimension is 3 (corresponding to the 3 channels of a color image: red, green, and blue). The CNN comprises a stack of modules, each of which performs three operations.

### 1. Convolution
A convolution extracts tiles of the input feature map, and applies filters to them to compute new features, producing an output feature map, or convolved feature (which may have a different size and depth than the input feature map). Convolutions are defined by two parameters:

- Size of the tiles that are extracted (typically 3x3 or 5x5 pixels).
- The depth of the output feature map, which corresponds to the number of filters that are applied.

During a convolution, the filters (matrices the same size as the tile size) effectively slide over the input feature map's grid horizontally and vertically, one pixel at a time, extracting each corresponding tile (see Figure 3).

![convolution-overview](images/convolution-overview.gif)

Figure 3. A 3x3 convolution of depth 1 performed over a 5x5 input feature map, also of depth 1. There are nine possible 3x3 locations to extract tiles from the 5x5 feature map, so this convolution produces a 3x3 output feature map.

For each filter-tile pair, the CNN performs element-wise multiplication of the filter matrix and the tile matrix, and then sums all the elements of the resulting matrix to get a single value. Each of these resulting values for every filter-tile pair is then output in the convolved feature matrix (see Figures 4a and 4b).

![conv-feature-filter](images/conv-feature-filter.png)

Figure 4a. Left: A 5x5 input feature map (depth 1). Right: a 3x3 convolution (depth 1).

![conv-applied-filter](images/conv-applied-filter.png)
Figure 4b. Left: The 3x3 convolution is performed on the 5x5 input feature map. Right: the resulting convolved feature. Click on a value in the output feature map to see how it was calculated.

During training, the CNN "learns" the optimal values for the filter matrices that enable it to extract meaningful features (textures, edges, shapes) from the input feature map. As the number of filters (output feature map depth) applied to the input increases, so does the number of features the CNN can extract. However, the tradeoff is that filters compose the majority of resources expended by the CNN, so training time also increases as more filters are added. Additionally, each filter added to the network provides less incremental value than the previous one, so engineers aim to construct networks that use the minimum number of filters needed to extract the features necessary for accurate image classification.

### 2. ReLU

Following each convolution operation, the CNN applies a Rectified Linear Unit (ReLU) transformation to the convolved feature, in order to introduce nonlinearity into the model. The ReLU function, , returns x for all values of x > 0, and returns 0 for all values of x ≤ 0.

### 3. Pooling
After ReLU comes a pooling step, in which the CNN downsamples the convolved feature (to save on processing time), reducing the number of dimensions of the feature map, while still preserving the most critical feature information. A common algorithm used for this process is called max pooling.

Max pooling operates in a similar fashion to convolution. We slide over the feature map and extract tiles of a specified size. For each tile, the maximum value is output to a new feature map, and all other values are discarded. Max pooling operations take two parameters:

- Size of the max-pooling filter (typically 2x2 pixels)
- Stride: the distance, in pixels, separating each extracted tile. Unlike with convolution, where filters slide over the feature map pixel by pixel, in max pooling, the stride determines the locations where each tile is extracted. For a 2x2 filter, a stride of 2 specifies that the max pooling operation will extract all nonoverlapping 2x2 tiles from the feature map (see Figure 5).

![maxpool-anim](images/maxpool-animation.gif)
Figure 5. Left: Max pooling performed over a 4x4 feature map with a 2x2 filter and stride of 2. Right: the output of the max pooling operation. Note the resulting feature map is now 2x2, preserving only the maximum values from each tile.

### Fully Connected Layers
At the end of a convolutional neural network are one or more fully connected layers (when two layers are "fully connected," every node in the first layer is connected to every node in the second layer). Their job is to perform classification based on the features extracted by the convolutions. Typically, the final fully connected layer contains a softmax activation function, which outputs a probability value from 0 to 1 for each of the classification labels the model is trying to predict.

Figure 6 illustrates the end-to-end structure of a convolutional neural network.
![conv-full](images/conv-full.png)

Figure 6. The CNN shown here contains two convolution modules (convolution + ReLU + pooling) for feature extraction, and two fully connected layers for classification. Other CNNs may contain larger or smaller numbers of convolutional modules, and greater or fewer fully connected layers. Engineers often experiment to figure out the configuration that produces the best results for their model.



## References

```
● Elements of AI, Helsinki University - https://course.elementsofai.com/
● A Beginner's Guide to Neural Networks and Deep Learning -
https://pathmind.com/wiki/neural-network
● Machine Learning Glossary - https://ml-cheatsheet.readthedocs.io/en/latest/
```