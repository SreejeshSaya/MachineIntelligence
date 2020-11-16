```
    def backward_propagation(self, yhat): # [TODO]
        '''
        The sample inputs and yhat are both provided to the backward propagation function. This function involves
        the gradient descent step which reduces the weights and biases of each layer inorder to minimize the error/loss
        function, in the next epoch. 
        
        From chain rule we have:
            dE/dw[i] = dE/dA[i] * dA[i]/dZ[i] * dZ[i]/dW[i]
        
        Variables and theirs corresponding meaning
        E = cross_entropy = -(y)*(log(yhat)) - (1-y)*((log(1-yhat)))
        Z[i] = w[i]*A[i-1] + b[i]

        dA[i] : last_layer    ->  dE/d_yhat  
                other_layers  ->  dA[i-1] = w[i] * dZ[i]

        dZ[i] : dA[i] * dA[i]/dZ[i]
        dw[i] : dZ[i] * dZ[i]/dw[i]
        '''
        #cross_entropy     =  A = -(y)*(log(yhat)) - (1-y)*((log(1-yhat)))
        #der_cross_entropy = dA/d_yhat = dA = -[ (y/yhat) - ((1-y)/(1-yhat)) ]  
        self.grads[f"dA{len(self.layers)-1}"] = -(np.divide(self.y,yhat) - np.divide((1 - self.y),(1-yhat)))
        m = len(self.y) # Size of the vector of actual values.
        for i in range(len(self.layers)-1, 0, -1):
            if(i ==(len(self.layers)-1) ):
                sig = self.sigmoid(self.AandZ[f"Z{i}"]) #der_sigmoid(Z[i]) = sigmoid(Z[i]) * (1-sigmoid(Z[i])
                #dz[i] = dE/dA[i] * der_sigmoid(Z[i])   
                self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * sig * (1-sig)
            else:
                #der_callback calls the respective deravative_activation function set under NN.__init__ 
                #dZ[i] = dE/dA[i] * der_actvn_func(Z[i])
               self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * self.der_callback(self.AandZ[f"Z{i}"])
            #dw[i] = (1/m) * dZ[i] * dZ[i]/dw[i]
            #dw[i] = (1/m) * dZ[i] * dA[i-1]
            self.grads[f"dW{i}"] = (1/m) * np.dot(self.grads[f"dZ{i}"], self.AandZ[f"A{i-1}"].T)
            #db[i] = (1/m) * dZ[i] * dZ[i]/db[i]
            #db[i] = (1/m) * dZ[i] * 1
            self.grads[f"db{i}"] = (1/m) * np.sum(self.grads[f"dZ{i}"], axis = 1, keepdims = True)
            #initialising values for the (i-1)th error/loss function layer
            self.grads[f"dA{i-1}"] = np.dot(self.params[f"W{i}"].T, self.grads[f"dZ{i}"])
            
            #weights and bias updation step 
            self.params[f"W{i}"] -= (self.alpha * self.grads[f"dW{i}"])
            self.params[f"b{i}"] -= (self.alpha * self.grads[f"db{i}"])
```
