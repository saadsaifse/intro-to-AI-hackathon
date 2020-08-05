# The Maths behind a Neural Network

```
Einav Grinberg, Muhammad Saad Saif, Anna Formaniuk
```

## Overview

```
● What are the elements of a Neural Network?
● Training a neural network
●

```
---

## What are the elements of a Neural Network?

### Layers

Neural networks are also called “stacked neural networks”, meaning networks composed of several layers. The layers are made of nodes. On the following image you can see an example of a simple neural network, that consists of the input layer, one hidden layer and the output layer, that has only two possible outcomes. Each layer's output serves as the subsequent layer's input.

!["Simple network" image](./images/mlp.png "source: A Beginner's Guide to Neural Networks and Deep Learning")

### Nodes

!["Node" image](./images/perceptron_node.png "source: A Beginner's Guide to Neural Networks and Deep Learning")

**Weights and inputs**

Similarly to the neurons in a human brain, the nodes are linked together and fire when receive enough stimuli from the other nodes. The nodes both process and store information. As seen on the image, the node takes inputs from the preceding nodes, combines them with a set of coefficients (weights), that either increase or decrease importance of each input, and sums it all up. 

**Activations and outputs**

Then the computed sum is then passed to the activation function, which decides what value in the range from 0 to 1 to store in the node as a result. The closer the value is to 1, the more "activated" it becomes.
The output then becomes the next layer's input or, if it's the output layer, is used to extract a prediction or a decision from the neural network. The most activated node is then the most probable outcome.

## Training a neural network

Before we can use a neural network, it has to be trained on some data