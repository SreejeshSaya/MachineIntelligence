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
        self.params = dict()
        self.grads = dict()
        self.layers = layers
        for i in range(1, len(layers)):
            self.params[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i-1])
            self.params[f"b{i}"] = np.random.randn(self.layers[i], 1)
            self.grads[f"W{i}"] = 0
            self.grads[f"b{i}"] = 0
        self.alpha = alpha
        self.num_iter = num_iter
        self.loss = list()
        self.X = None
        self.y = None
        self.grads = dict()
        
        
        
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))



    def relu(self, x):
        return np.maximum(0,x)
    
    
    def der_relu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    
    
    def forward_propagation(self):
        self.params["A0"] = self.X
        #print(self.X.shape)
        for i in range(1, len(self.layers)):
            #print(self.params[f"A{i-1}"].T.shape)
            self.params[f"Z{i}"] = self.params[f"W{i}"].dot(self.params[f"A{i-1}"]) + self.params[f"b{i}"]
            if(i!=(len(self.layers)-1)):
                self.params[f"A{i}"] = self.relu(self.params[f"Z{i}"])
            else:
                yhat = self.sigmoid(self.params[f"Z{i}"])
        m = len(self.y)
        Error = -(1/m) * (np.sum(np.multiply(np.log(yhat), self.y) + np.multiply((1 - self.y), np.log(1 - yhat))))
        return yhat, Error
                
    
        
    def backward_propagation(self, yhat):
        self.grads[f"dA{len(self.layers)-1}"] = -(np.divide(self.y,yhat) - np.divide((1 - self.y),(1-yhat)))
        m = len(self.y)
        for i in range(len(self.layers)-1, 0, -1):
            if(i==(len(self.layers)-1)):
                sig = self.sigmoid(self.params[f"Z{i}"])
                self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * sig * (1-sig)
            else:
                self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * self.der_relu(self.params[f"Z{i}"])
            self.grads[f"dW{i}"] = (1/m) * np.dot(self.grads[f"dZ{i}"], self.params[f"A{i-1}"].T)
            self.grads[f"db{i}"] = (1/m) * np.sum(self.grads[f"dZ{i}"], axis = 1, keepdims = True)
            self.grads[f"dA{i-1}"] = np.dot(self.params[f"W{i}"].T, self.grads[f"dZ{i}"])
             
            self.params[f"W{i}"] -= (self.alpha * self.grads[f"dW{i}"])
            self.params[f"b{i}"] -= (self.alpha * self.grads[f"db{i}"])
     
        
    ''' X and Y are dataframes '''
    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
		'''
        self.X = X.values.T
        print(self.X.shape)
        self.y = Y.values
        for i in range(self.num_iter):
            print(f"Iteration number : {i+1}.....", end=" ")
            yhat, Error = self.forward_propagation()
            self.backward_propagation(yhat)
            print(f"Error : {Error}")
	
    
    def predict(self,X):
        """
        The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X

        """
        A_prev = X.values.T
        for i in range(1, len(self.layers)):
            z = self.params[f"W{i}"].dot(A_prev) + self.params[f"b{i}"]
            if(i!=(len(self.layers)-1)):
               A_prev  = self.relu(z)
            else:
                yhat = self.sigmoid(z)
        return yhat    
        
	
    def CM(self, y_test, y_test_obs):
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

        acc = tp+tn/tp+fp+fn+tn
        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
		
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Accuracy : {acc}")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
		
	    
layers = [9, 8, 1]
alpha = 0.15
num_iter = 250
model = NN(layers, alpha, num_iter)
df = pd.read_csv('C:/Vishesh/College/Third year engg/PESU-3rd-year/SEM 5/Machine Intelligence - UE18CS303/A3/Final_LBW.csv')
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
y_test = list(Y.values)
model.fit(X, Y)
y_pred = list(model.predict(X)[0])
model.CM(y_test, y_pred)
