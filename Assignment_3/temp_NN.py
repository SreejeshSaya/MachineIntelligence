# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

class NN:
    def __init__(self, layers, alpha, num_iter):
        self.params = dict() #Hyperparameters[weights and biases] and also linear combinations/activations of each layer
        self.grads = dict() #gradients for each layer
        self.layers = layers #a list containing the number of nodes at each layer. First element contains the number of features of the dataset.
        #Initialising the weights, biases and gradients for the weights and biases.
        for i in range(1, len(layers)):
            params[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i-1]) * 0.01
            params[f"b{i}"] = np.random.randn(self.layers[i], 1) * 0.01
            grads[f"W{i}"] = 0
            grads[f"b{i}"] = 0
        self.alpha = alpha #learning rate
        self.num_iter = num_iter #Number of iterations
        #self.loss = list()  Not required
        self.X = None #Input features, to be stored as a numpy matrix.
        self.y = None #Target variable, original values, to be stored as a numpy array.
        self.params["A0"] = X.values 
        
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))



    def relu(self, x):
        return np.maximum(0,x)
    
    
    def der_relu(x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def forward_propagation(self):
        for i in range(1, len(self.layers)):
            #Performing the linear combination of the inputs for one layer and adding the bias
            self.params[f"Z{i}"] = self.params[f"W{i}"].dot(self.params[f"A{i-1}"].T) + self.params[f"b{i}"] 
            #Perform relu activation on every layer except final. For final, perform sigmoid activation.
            if(i!=(len(self.layers)-1)):
                self.params[f"A{i}"] = self.relu(self.params[f"Z{i}"])
            else:
                yhat = self.sigmoid(self.params[f"Z{i}"])
        m = len(self.y)
        #Calculating the loss function value, through cross entropy.
        Error = -(1/m) * (np.sum(np.multiply(np.log(yhat), self.y) + np.multiply((1 - self.y), np.log(1 - yhat))))
        return yhat, Error
    
    def CM(y_test,y_test_obs):
		'''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

		for i in range(len(y_test_obs)):
			if(y_test_obs[i]>0.6):
				y_test_obs[i]=1
			else:
				y_test_obs[i]=0
		
		cm=[[0,0],[0,0]]
		fp=0
		fn=0
		tp=0
		tn=0
		
		for i in range(len(y_test)):
			if(y_test[i]==1 and y_test_obs[i]==1):
				tp=tp+1
			if(y_test[i]==0 and y_test_obs[i]==0):
				tn=tn+1
			if(y_test[i]==1 and y_test_obs[i]==0):
				fp=fp+1
			if(y_test[i]==0 and y_test_obs[i]==1):
				fn=fn+1
		cm[0][0]=tn
		cm[0][1]=fp
		cm[1][0]=fn
		cm[1][1]=tp

		p= tp/(tp+fp)
		r=tp/(tp+fn)
		f1=(2*p*r)/(p+r)
		
		print("Confusion Matrix : ")
		print(cm)
		print("\n")
		print(f"Precision : {p}")
		print(f"Recall : {r}")
		print(f"F1 SCORE : {f1}")