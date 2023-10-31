# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 21:57:28 2022

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
def sigmoid_derivative(s):   
    ds = s * (1 - s)
    return ds
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def relu_derivative(s):   
    s[s<=0] = 0
    s[s>0] = 1
    return s
def relu(x):
    return np.maximum(0,x)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 20, 1, 64, 1
# Create random input and output data
np.random.seed(0)
x = np.arange(0,N,1).reshape(N,D_in)*1.0 #20*1
y = x + np.random.randn(N,D_out)        #20*1
# Randomly initialize weights
w1 = np.random.randn(D_in, H)             #1*64
w2 = np.random.randn(H, D_out)            #64*1
b1 = np.random.randn(H)
b2 = np.random.randn(D_out)

learning_rate = 1e-5
for t in range(20000):
    # Forward pass: compute predicted y
    h = x.dot(w1)+b1
    h_out = relu(h)
    y_pred = h_out.dot(w2)+b2

    # Compute loss
    loss = np.square(y_pred - y).sum()

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_out.T.dot(grad_y_pred)
    grad_b2 = np.ones(N).T.dot(grad_y_pred)
    
    grad_h = grad_y_pred.dot(w2.T)     #[N, H]=[N, 1]*[1, H]
    grad_h = grad_h*relu_derivative(h_out)   #[N, H]=[N, H] . [N, H]]
    grad_w1 = x.T.dot(grad_h)          #[1, H]=[1, N]*{N, H}
    grad_b1 = np.ones(N).T.dot(grad_h)
    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    b1 -= learning_rate * grad_b1
    b2 -= learning_rate * grad_b2
    if (t%1000==0):        
        plt.cla()
        plt.scatter(x,y)
        plt.scatter(x,y_pred)
        plt.plot(x,y_pred,'r-',lw=1, label="plot figure")
        plt.text(0.5, 0, 't=%d:Loss=%.4f' % (t, loss), fontdict={'size': 20, 'color':  'red'})
        plt.show()
