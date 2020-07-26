# Image Recognition with Python

```
Einav Grinberg, Muhammad Saad Saif, Anna Formaniuk
```

---

## Overview

### Objectives

After completing this tutorial, readers will

- Get an understanding of image recognition using Python and related tools
- Learn how to build, train, and utilize an image classifier that can recognize cats and dogs

### Steps

**Total duration:** 50-60 minutes

- Basic introduction to Python - 5 minutes
- [Introduction to image recognition and selected tools](#introduction-to-image-recognition-and-selected-tools) - 5 minutes
- Setting-up the development environment - 5 minutes
- Importing and preparing the data - 5 minutes
- Compiling and training the model - 10 minutes
- Visualizing the training results - 5 minutes
- Using the model to predict classes for new images - 5 minutes
- Tasks - 10-15 minutes

---

## Image Classification before Machine Learning

Early computer vision models relied on raw pixel data as the input to the model. However, as shown in Figure 2, raw pixel data alone doesn't provide a sufficiently stable representation to encompass the myriad variations of an object as captured in an image. The position of the object, background behind the object, ambient lighting, camera angle, and camera focus all can produce fluctuation in raw pixel data; these differences are significant enough that they cannot be corrected for by taking weighted averages of pixel RGB values.

![cat-variation](images/cat-variation.png)

To model objects more flexibly, classic computer vision models added new features derived from pixel data, such as color histograms, textures, and shapes. The downside of this approach was that feature engineering became a real burden, as there were so many inputs to tweak. For a cat classifier, which colors were most relevant? How flexible should the shape definitions be? Because features needed to be tuned so precisely, building robust models was quite challenging, and accuracy suffered.


## Introduction to image recognition and selected tools

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


### TensorFlow for Machine and Deep Learning

We will perform image classification using the following tools:

- TensorFlow - an end-to-end open source platform for machine learning.
- Jupyter Notebook - an open-source application that allows to create and share documents that contain live code, equations, visualizations and narrative text.
- Google Colaboratory - a browser-based environment that allows you to write and execute Python in your browser. Colab notebooks are Jupyter notebooks that are hosted by Colab
- Python 3 - preferred programming language for machine learning with TensorFlow
- Keras - a high-level deep-learning API for configuring neural networks.
- Other libraries - such as Numpy, Pandas, and Matplotlib etc.

We will perform everything in the browser so you do not have to install or perform any kind of setup on your laptop.

For a beginners guide about machine learning using TensorFlow, go through this Hello World tutorial https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/#0

## Setting-up the development environment
Make a video/screen-recording of setting up a mac and windows and run the basic code example