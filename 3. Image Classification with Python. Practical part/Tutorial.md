# Image Classification with Python: Practical part

```
Einav Grinberg, Muhammad Saad Saif, Anna Formaniuk
```

---

## Overview

### Objectives

After completing this tutorial, readers will

- Get familiar with tools used for Image Classification
- Perform a practical ML tutorial to classify cats and dogs images

```
● Familiarity with Image Classification tools - 10 minutes
● Perform practical machine learning to classify cats and dogs images - 25 minutes
● Tasks - 15 minutes 
```
---

## Image Classification Tools

> This tutorial can be performed without installing any of the mentioned tools on your machine. However, it is encouraged to install and experiment with these tools and run the tutorial on your machine.

There are several tool to perform machine and deep learning, but in this section, a brief introduction is provided to the tools that are used in this tutorial.

### Python Programming Language
The language of choice for this tutorial is Python. Mainly because it has the following advantages:

- Open Source: Python is an open source language that means anyone can use and contribute towards the development of the language and its related tools
- Simple to learn: Python is really simple to learn and also contains a lot of libraries that can be readily used to perform difficult tasks with simple function calls 
- Great for data handling: With Python, the handling of data is simple, for example reading CSV files, creating matrices, lists and arrays are quite user friendly

A basic understanding of Python is assumed for this tutorial. To learn more about Python, visit https://www.python.org/about/gettingstarted/. 

### Jupyter Notebook
The Jupyter Notebook is an open-source web application that allows to create and share documents that contain live code, equations, visualizations and narrative text. 
Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more.
JupyterLab is a new interface for the Jupyter Notebooks and is recommended to use. You can install Jupyter Notebook and JupyerLab on your machine to use them with Python or many other languages.

To learn more about Jupyter Notebook, visit https://jupyter.org/index.html

### Google Colaboratory
While Jupyter Notebooks are a great tool to write Python code, it has to be installed. It also has a cloud based Jupyter Hub but there is a better option by Google, called Google Colab.
Google Colaboratory, or "Colab" for short, allows to write and execute Python in the browser, with

- Zero configuration required
- Free access to GPUs
- Easy sharing

Colab notebooks execute code on Google's cloud servers, meaning you can leverage the power of Google hardware, including GPUs and TPUs, regardless of the power of your machine. All you need is a browser. 

It is perfect for students, data scientists or AI researchers. To learn more about Google Colab, visit https://colab.research.google.com/notebooks/intro.ipynb#scrollTo=5fCEDCU_qrC0


### TensorFlow for Machine and Deep Learning

TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.
TensorFlow provides many high-level application programming interfaces (APIs) such as Sequential API and Keras API to easily build and train ML models. 

We will use TensorFlow as our main ML tool to develop a deep learning model that recognizes and classifies cats and dogs images. To learn more about TensorFlow, visit https://www.tensorflow.org/learn

### Keras
Keras is an open-source library to create neural-networks. It is designed to be user-friendly, modular, and extensible. Keras contains numerous implementations of commonly used neural-network building blocks such as layers, objectives, activation functions, optimizers, and a host of tools to make working with image and text data easier to simplify the coding necessary for writing deep neural network code.
Since 2017, Keras is available inside the TensorFlow library which makes it even easier to use. To learn more about Keras, visit https://keras.io/

### Other Libs 

Apart from the main tools and libraries mentioned above, following libraries are also used that provide the core data handling and visualization functionalities using Python.

#### NumPy
The fundamental package for scientific computing with Python. It adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. Visit https://numpy.org/ for more details

#### Pandas
pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool,
built on top of the Python programming language. It offers data structures and operations for manipulating numerical tables and time series (a time-based series of data). Visit https://pandas.pydata.org/ for more details.

#### Matplotlib
Matplotlib is a Python library to visualize results. Static, animated, and interactive visualizations can be created using Matplotlib.
Visit https://matplotlib.org/ for more details.

## Classify Cats & Dogs Images

### Explore the example data

Cats and dogs dataset: https://www.kaggle.com/c/dogs-vs-cats/data

### Complete the tutorial

Tutorial link: https://colab.research.google.com/drive/1ixqbOLgEw8GLQxBK2RjH8L_cCQAG-96u?usp=sharing

## Tasks

### Complete the TensorFlow ML beginners guide
For a beginners guide about machine learning using TensorFlow, go through this Hello World tutorial https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/#0
