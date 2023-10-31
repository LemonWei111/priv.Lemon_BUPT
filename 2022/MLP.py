# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.special

import matplotlib.pyplot as plt

class neuralNetwork:
    
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    
    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        self.who += self.lr * np.dot((output_errors *final_outputs * (1.0-final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors *hidden_outputs * (1.0-hidden_outputs)), np.transpose(inputs))
        pass
    
    def test(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


train_data_file = open("./mnist_train.csv",'r')
train_data_list = train_data_file.readlines()
train_data_file.close()

test_data_file = open("./mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

net = neuralNetwork(784,100,10,0.2)
epoch = {1,2,3,4,5,6,7}
for e in epoch:
    for record in train_data_list:
        all_values = record.split(',')        
        inputs = (np.asfarray(all_values[1:]) /255.0 * 0.99) + 0.01
        targets = np.zeros(10) + 0.01
        targets[int(all_values[0])] = 0.99
        net.train(inputs, targets)

scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    inputs = (np.asfarray(all_values[1:]) /255.0 *0.99) + 0.01
    outputs = net.test(inputs)
    label = np.argmax(outputs)
    
    correct_label = int(all_values[0])
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        
scorecard_array = np.asarray(scorecard)
print(scorecard_array.sum() / scorecard_array.size)

plt.imshow(inputs.reshape(28,28))
plt.show()
print(outputs)

