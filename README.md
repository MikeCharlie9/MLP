# MLP

#### This is a collection of Python programs for Multi-Layer Perceptron network


---

##### Introduction

The program can conduct training and test for MNIST which is a data set of handwritten numbers.

Detail information about MNIST can be found in the website: 
[MNIST](https://yann.lecun.com/exdb/mnist/)

The program is written in Python. All the modules called in the program is commonly used, such as "numpy" and so on. No machining learning framework is used in the program to make sure that you can run the program in the simplest configuration.

If you don't have these basic modules, you can install them using the command "pip install". 
For example:
```
pip install numpy
```

Or, you can install anaconda for convenience.


---
##### File list
- mnist_mlp0.py
- mnist_mlp1.py
- mnist_mlp2.py

The number followed the "mnist_mlp" refers to the number of hidden layer in the MLP network. 

For example, you can configure a network with two hidden layers using "mnist_mlp2.py".

The number of nodes in each layer can be configurable if you find the variable named L1, L2 ... But remember, do not modify L1 and the last layer, as they're determined by the data set!  

If you have read the code, you can modify anywhere as you like!


---
##### Run the python

Using the command like :
```
python mnist_mlp1.py
```
Make sure you have installed all needed modules. You should also put the data set files in the right path accoding the the source code.

On a normal PC(in 2019), it takes about 10 minutes to run a network with one hidden layer. Don't worry too much and you can sit down and have a coffee. You will see the middle result in each epoch.
