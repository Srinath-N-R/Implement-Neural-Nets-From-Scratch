#!/usr/bin/env python
# coding: utf-8

# Question 1

# Download the fashion-MNIST dataset and plot 1 sample image for each class as shown in the grid below. Use "from keras.datasets import fashion_mnist" for getting the fashion mnist dataset.

# In[1061]:


import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import math


# In[88]:


((trainX, trainY), (testX, testY))= fashion_mnist.load_data()


# In[1007]:


new_trainY = []
for i in trainY:
    one_hot = [0]*10
    ele = i
    one_hot[i] = 1
    new_trainY.append(one_hot)


# In[73]:


check_list = []
main_list = []
for i in range(0,30):
    if trainY[i] in check_list:
        continue
    else:
        check_list.append(trainY[i])
        main_list.append(i)


# In[74]:


for i,j in enumerate(main_list):
    plt.subplot(2,5,i+1)
    plt.imshow(trainX[j], cmap=plt.get_cmap('gray'))


# Question 2

# Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a 
# probability distribution over the 10 classes.
# 
# Your code should be flexible so that it is easy to change the number of hidden layers and the number of neurons 
# in each hidden layer.

# In[1127]:


input_neurons = int(input("Enter the number of input features: "))
no_of_layers = int(input("Enter the number of hidden layers: "))
neuron_info = [None]*(no_of_layers+1)
activation_info = [None]*(no_of_layers+1)
neuron_info[0] = input_neurons
for i in range(1,no_of_layers+1):
    neuron_info[i] = int(input("Enter the number of neurons in layer {}: ".format(i)))
    activation_info[i] = input("Enter the activation function in layer {}: ".format(i))
neuron_info.append(int(input("Enter the number of neurons in output layer: ")))
activation_info.append(input("Enter the activation function in output layer: "))


# In[1066]:


def sigmoid(m):
    return 1/(1+np.exp(-m))

def sigmoid_derivarive(m):
    return np.multiply(m, (1 - m))

def tanh(m):
    return (np.exp(m) - np.exp(-m))/(np.exp(m) + np.exp(-m))

def tanh_derivarive(m):
    return (1- np.multiply(m, m))

def relu(m):
    return np.maximum(m, 0)

def relu_derivarive(m):
    m[m>0] = 1
    m[m<0] = 0
    return m

def softmax(m):
    return np.exp(m)/np.exp(m).sum()


# In[1087]:


neuron_info


# In[1114]:


def initialize(no_of_layers, neuron_info):
    W = [0]
    b = [0]
    for i in range(1, no_of_layers+2):
        x = neuron_info[i]
        y = neuron_info[i-1]
        W.append(np.multiply(np.random.normal(0, 0.01, size=(x,y)), 1/np.sqrt(2*neuron_info[i-1])))
        b.append(np.random.normal(0, 0.01, size=(x,1)))
    return W, b


# In[1115]:


W, b = initialize(no_of_layers, neuron_info)


# In[1068]:


def forward_propogation(input_features, W, b, no_of_layers, activation_info):
    H = [None]*(no_of_layers+2)
    A = [None]*(no_of_layers+2)
    H[0] = input_features.reshape(input_neurons, 1)/255
    for i in range(1, no_of_layers+2):
        A[i] = np.matmul(W[i],H[i-1]) + b[i]
        if activation_info[i] == 'sigmoid':
            H[i] = sigmoid(A[i])
        elif activation_info[i] == 'tanh':
            H[i] = tanh(A[i])
        elif activation_info[i] == 'relu':
            H[i] = relu(A[i])
        elif activation_info[i] == 'softmax':
            H[i] = softmax(A[i])
    return H


# In[1069]:


def back_propogation(Y, H, W, b, activation_info, output_neurons, no_of_layers):
    Y = Y.reshape(output_neurons, 1)
    dH = [0]*(no_of_layers+2)
    dA = [0]*(no_of_layers+2)
    dW = [0]*(no_of_layers+2)
    db = [0]*(no_of_layers+2)
        
    if activation_info[-1] == 'softmax':
        dH[-1] = -np.multiply(Y, 1/H[-1])
        dA[-1] = dH[-1] - Y 
    elif activation_info[-1] == 'sigmoid':
        dH[-1] = -np.multiply(Y, 1/H[-1])
        dA[-1] = np.multiply(dH[-1], sigmoid_derivarive(H[-1]))
    elif activation_info[-1] == 'tanh':
        dH[-1] = -np.multiply(Y, 1/H[-1])
        dA[-1] = np.multiply(dH[-1], tanh_derivarive(dH[-1]))
    elif activation_info[-1] == 'relu':
        dH[-1] = -np.multiply(2, Y-H)
        dA[-1] = np.multiply(dH[-1], relu_derivarive(dH[-1]))
    
    dW[-1] = np.matmul(dA[-1], H[-2].T)
    db[-1] = dA[-1]
        
    for i in range(no_of_layers):
        dH[no_of_layers-i] = np.matmul(dA[no_of_layers+1-i].T, W[no_of_layers+1-i])
        if activation_info[no_of_layers-i] == 'sigmoid':
            dA[no_of_layers-i] = np.multiply(dH[no_of_layers-i].T, sigmoid_derivarive(H[no_of_layers-i]))
        elif activation_info[no_of_layers-i] == 'tanh':
            dA[no_of_layers-i] = np.multiply(dH[no_of_layers-i].T, tanh_derivarive(H[no_of_layers-i]))
        elif activation_info[no_of_layers-i] == 'relu':
            dA[no_of_layers-i] = np.multiply(dH[no_of_layers-i].T, relu_derivarive(H[no_of_layers-i]))
       
        dW[no_of_layers-i] = np.matmul(dA[no_of_layers-i], H[no_of_layers-i-1].T)
        db[no_of_layers-i] = dA[no_of_layers-i]
    return dW, db


# In[1124]:


W, b = initialize(no_of_layers, neuron_info)
for i in range(10):
    for i in range(len(trainX[:1000])):
        H = forward_propogation(trainX[i], W, b, no_of_layers, activation_info)
        dW, db = back_propogation(np.array(new_trainY[i]), H, W, b, activation_info, neuron_info[-1], no_of_layers)
        for i in range(no_of_layers):
            W[i] = W[i] - dW[i]
            b[i] = b[i] - db[i]


# In[1123]:


W, b = initialize(no_of_layers, neuron_info)
prev_VdW = 0
prev_Vdb = 0
for i in range(10):
    for i in range(len(trainX[:1000])):
        H = forward_propogation(trainX[i], W, b, no_of_layers, activation_info)
        dW, db = back_propogation(np.array(new_trainY[i]), H, W, b, activation_info, neuron_info[-1], no_of_layers)
        VdW = np.multiply(0.9, prev_VdW) + dW
        Vdb = np.multiply(0.9, prev_Vdb) + db            
        prev_VdW = VdW
        prev_Vdb = Vdb
        W = W - VdW
        b = b - Vdb


# In[986]:


W, b = initialize(no_of_layers, neuron_info)
prev_VdW, prev_Vdb = 0, 0
for i in range(5):
    for i in range(len(trainX[:100])):
        H = forward_propogation(trainX[i], W, b, no_of_layers, activation_info)
        VdW = np.multiply(0.9, prev_VdW)
        Vdb = np.multiply(0.9, prev_Vdb)
        W = W - VdW
        b = b - Vdb
        dW, db = back_propogation(np.array(new_trainY[i]), H, W, b, activation_info, neuron_info[-1], no_of_layers)
        VdW = np.multiply(0.9, prev_VdW) + dW
        Vdb = np.multiply(0.9, prev_Vdb) + db 
        prev_VdW = VdW
        prev_Vdb = Vdb
        W = W - VdW
        b = b - Vdb


# In[1022]:


W, b = initialize(no_of_layers, neuron_info)
prev_VdW, prev_Vdb = 0, 0
for i in range(5):
    for i in range(len(trainX[:10000])):
        H = forward_propogation(trainX[i], W, b, no_of_layers, activation_info)
        dW, db = back_propogation(np.array(new_trainY[i]), H, W, b, activation_info, neuron_info[-1], no_of_layers)
        VdW = np.multiply(0.9, prev_VdW) + dW
        Vdb = np.multiply(0.9, prev_Vdb) + db            
        prev_VdW = VdW
        prev_Vdb = Vdb
        dWA = VdW + prev_VdW
        dbA = Vdb + prev_Vdb
    VdW = np.multiply(1/100, dWA)
    Vdb = np.multiply(1/100, dbA)
    W = W - VdW
    b = b - Vdb


# In[1014]:


W, b = initialize(no_of_layers, neuron_info)
prev_VdW = 0
prev_Vdb = 0
for i in range(5):
    for i in range(len(trainX[:100])):
        H = forward_propogation(trainX[i], W, b, no_of_layers, activation_info)
        dW, db = back_propogation(np.array(new_trainY[i]), H, W, b, activation_info, neuron_info[-1], no_of_layers)
        VdW = np.multiply(0.9, prev_VdW) + np.multiply(0.1, np.multiply(dW, dW))
        Vdb = np.multiply(0.9, prev_Vdb) + np.multiply(0.1, np.multiply(db, db))            
        prev_VdW = VdW
        prev_Vdb = Vdb
        for i in range(len(VdW)):
            VdW[i] = np.sqrt(VdW[i] + 0.001)
            Vdb[i] = np.sqrt(Vdb[i] + 0.001)
        W = W - np.multiply(1/VdW, dW)
        b = b - np.multiply(1/Vdb, db)


# In[1006]:


W, b = initialize(no_of_layers, neuron_info)
prev_VdW, prev_Vdb, prev_MdW, prev_Mdb = 0, 0, 0, 0

for k in range(15):
    for i in range(len(trainX[:1000])):
        H = forward_propogation(trainX[i], W, b, no_of_layers, activation_info)
        dW, db = back_propogation(np.array(new_trainY[i]), H, W, b, activation_info, neuron_info[-1], no_of_layers)
        VdW = np.multiply(0.9, prev_VdW) + np.multiply(0.1, np.multiply(dW, dW))
        Vdb = np.multiply(0.8, prev_Vdb) + np.multiply(0.2, np.multiply(db, db))
        VdW = VdW/(1 - math.pow(0.8, k+1))
        Vdb = Vdb/(1 - math.pow(0.8, k+1))
        prev_VdW = VdW
        prev_Vdb = Vdb
        
        for j in range(len(VdW)):
            VdW[j] = np.sqrt(VdW[j] + 0.001)
            Vdb[j] = np.sqrt(Vdb[j] + 0.001)
        
        MdW = np.multiply(0.9, prev_MdW) + np.multiply(0.1, np.multiply(dW, dW))
        Mdb = np.multiply(0.9, prev_Mdb) + np.multiply(0.1, np.multiply(db, db))    
        MdW = MdW/(1 - math.pow(0.8, k+1))
        Mdb = Mdb/(1 - math.pow(0.8, k+1))
        prev_MdW = MdW
        prev_Mdb = Mdb
        
        W = W - np.multiply(1/VdW, MdW)
        b = b - np.multiply(1/Vdb, Mdb)


# In[1054]:


H = forward_propogation(testX[33], W, b, no_of_layers, activation_info)   
np.argmax(H[-1])


# In[1055]:


testY[31]


# In[1128]:


W, b = initialize(no_of_layers, neuron_info)
for i in range(10):
    for i in range(len(trainX[:1000])):
        H = forward_propogation(trainX[i], W, b, no_of_layers, activation_info)
        dW, db = back_propogation(np.array(new_trainY[i]), H, W, b, activation_info, neuron_info[-1], no_of_layers)
        for i in range(no_of_layers):
            W[i] = W[i] - dW[i]
            b[i] = b[i] - db[i]


# In[1129]:


prediction = []
for i in testX:
    H = forward_propogation(i, W, b, no_of_layers, activation_info)
    prediction.append(np.argmax(H[-1])) 


# In[1130]:


confusion_matrix(testY, prediction)


# In[ ]:




