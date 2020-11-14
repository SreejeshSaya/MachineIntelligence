'''Importing the modules'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
    #my name is vishesh

class NN:
    verbose = 0 #Used to print the loss function values after each epoch upon request.
    def __init__(self, layers, alpha, num_iter, activation = 'relu'):
        ''' Initialising the required variables for the neural network. '''
        
        self.params = dict() #All the weights and biases of every layer in the form of matrices stored together in a dictionary.
        self.AandZ = dict() #Dictionary containing all the Z's and A's of every layer.
        #Z[i] = W[i].T*A[i-1] + b[i]
        #A[i] = activation(Z[i])
        self.grads = dict() #Stores the gradients of the weights for all the layers in a dictionary.
        self.layers = layers #List of values of the form [9, n, 1] where 
                             #9 is the number of input features
                             #n is the set of values where each value represents the numbe rof neurons in that specific layer.
                             #1 is the output layer neuron.
        if(activation == 'relu'):
            self.activation = activation
            self.der_activation = 'der_relu'
        if(activation == 'tanh'):
            self.activation = activation
            self.der_activation = 'der_tanh'
        for i in range(1, len(layers)):            
            self.params[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(1/self.layers[i-1])
            #Weights are initialised randomly in numpy matrices of the order (n[l], n[l-1]) where 
                                #n[l] is the number of neurons in the current layer
                                #n[l-1] is the number of neurons in the previous layer
            #Xavier initialisation method was also implemented along with random initialisation to avoid exploding/vanishing gradients.
            self.params[f"b{i}"] = np.random.randn(self.layers[i], 1)
            #Biases are initialised randomly in numpy vectors of length n[l].
            self.grads[f"dW{i}"] = 0 
            self.grads[f"db{i}"] = 0
            #Dictionary containing the gradients of the weights and biases(of every layer).
        self.alpha = alpha #Learning rate.
        self.num_iter = num_iter #Number of epochs 
        self.X = None #Matrix of features of the training set
        self.y = None #Vector of target variable values.        
        
        
    def sigmoid(self, x):
        #Given a numpy array x, the function computes the sigmoid function value for every element in x.
        return 1/(1 + np.exp(-x))
    


    def relu(self, x):
        #Given a numpy array x, the function computes the ReLU function value for every element in x.
        return np.maximum(0,x)
    
    
    def tanh(self, x):
        #Given a numpy array x, the function computes the tanh function value for every element in x.
        return np.tanh(x)
    
    
    def der_tanh(self, x):
        #returns the derivative of the tanh function.
        return (1 - np.square(self.tanh(x)))
    
    
    def der_relu(self, x):
        #Returns the derivative of the ReLU function.
        return ((x > 0) * 1)
    
    def callback(self,  x):    
        if(self.activation == 'relu'):
            return self.relu(x)
        if(self.activation == 'tanh'):
            return self.tanh(x)
        
    def der_callback(self, x):
        if(self.der_activation == 'der_relu'):
            return self.der_relu(x)
        if(self.der_activation == 'der_tanh'):
            return self.der_tanh(x)
       
    
    def forward_propagation(self):
        '''
        The sample inputs are provided to the neural network and at each layer the preactivation and activation steps are performed and
        We get the predicted values for all sample inputs. 
        The loss is computed which is used to perform back propagation.

        Returns
        -------
        yhat : Vector of values corresponding to provided inputs X
            
        loss : Numerical value representing the loss function signifying the strength of the neural network.

        '''
        self.AandZ["A0"] = self.X
        #Setting A0 to be the matrix of features X, for simplicity
        #L2 regularisation = (lambda)*[ sigma( w[i]*w[i] )]
        reg_weights = 0 #reg_weights = [sigma( w[i]*w[i] )]
        for i in range(1, len(self.layers)):
            self.AandZ[f"Z{i}"] = self.params[f"W{i}"].dot(self.AandZ[f"A{i-1}"]) + self.params[f"b{i}"]
            #Z[i] = W[i].T*A[i-1] + b[i]
            #W[i] is the weights matrix of layer i
            #A[i-1] is the activations of the previous layer passed as input to the current layer.
            #b[i] is the bias vector of layer i.
            if(i!=(len(self.layers)-1)):
                self.AandZ[f"A{i}"] = self.callback(self.AandZ[f"Z{i}"])
                #A[i] is the activation of Z[i] , where i refers to a particular layer.
                #the callback function returns the respective output of the activation function requested by the user.
            else:
                yhat = self.sigmoid(self.AandZ[f"Z{i}"])
                #yhat is the vector of predicted values for each vector in matrix X.
                #The activation function of the output layer is sigmoid to retrieve a value between 0 and 1 for the entire vector yhat.
            reg_weights += np.sum(np.square(self.params[f"W{i}"]))
        m = len(self.y) #Number of records 
        #L2 Regularization
        lamda = 1
        l2_reg = (lamda/(2*m)) * reg_weights
        try:
        	#cross entropy loss with l2_regularizer
        	loss = -(1/m) * (np.sum(np.multiply(np.log(yhat), self.y) + np.multiply((1 - self.y), np.log(1 - yhat)))) - l2_reg
            #The loss function used here is Cross entropy. 
            #The loss is calculated after predicting the values for the entire training set(Batch processing).
            #L2 regularization has been combined with the loss function to prevent overfitting.
        except ZeroDivisionError:
        	print("ZeroDivisionError encountered while calculating the loss. Run again.")   	
        return yhat, loss
                
            
    def backward_propagation(self, yhat):
        
        
        self.grads[f"dA{len(self.layers)-1}"] = -(np.divide(self.y,yhat) - np.divide((1 - self.y),(1-yhat)))
        m = len(self.y) #Size of the vector of actual values.
        for i in range(len(self.layers)-1, 0, -1):
            if(i ==(len(self.layers)-1) ):
                sig = self.sigmoid(self.AandZ[f"Z{i}"])
                self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * sig * (1-sig)
            else:
               self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * self.der_callback(self.AandZ[f"Z{i}"])
            self.grads[f"dW{i}"] = (1/m) * np.dot(self.grads[f"dZ{i}"], self.AandZ[f"A{i-1}"].T)
            self.grads[f"db{i}"] = (1/m) * np.sum(self.grads[f"dZ{i}"], axis = 1, keepdims = True)
            self.grads[f"dA{i-1}"] = np.dot(self.params[f"W{i}"].T, self.grads[f"dZ{i}"])
             
            self.params[f"W{i}"] -= (self.alpha * self.grads[f"dW{i}"])
            self.params[f"b{i}"] -= (self.alpha * self.grads[f"db{i}"])

     
    ''' X and Y are dataframes '''
    def fit(self,X,Y, verbose = 0):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        
        Batch processing has been implemeted in this neural network 
        i.e. all the records of the inputs are passed to the neural network before performing gradient descent.
		
        Learning-rate decay has also been implemented, which means at the end of every epoch, the learning rate is reduced by a factor,
        which helps the loss function to converge better at the local/glabal minima.
        '''
        NN.verbose = verbose
        #If verbose is set to a non-zero value, then the loss function value is displayed after every epoch, otherwise it isn't displayed.
        self.X = X.values.T
        #Assigning the matrix of features X to self.X.       
        self.y = Y.values
        #Assigning the vector of values y containing the target variable values, which will be later used to compute the loss.
        for i in range(self.num_iter):
            if(verbose):
            	print(f"Iteration number : {i+1}.....", end=" ")
            yhat, loss = self.forward_propagation() #Forward propagation step
            self.backward_propagation(yhat) #Backward propagation step
            if(verbose):
            	print(f"Error : {loss}")
            self.alpha = (0.9995**i) * self.alpha #Learning-rate decay.

	
    def predict(self,X):
        """
        The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X

        """
        A_prev = X.values.T
        #A_prev represents the set of inputs obtaied from the previous layer provided to the current layer's neurons.
        for i in range(1, len(self.layers)):
            z = self.params[f"W{i}"].dot(A_prev) + self.params[f"b{i}"]
            #Preactivation operation performed at each layer
            if(i!=(len(self.layers)-1)):    
                A_prev  = self.callback(z)
                #Activation operation performed at each layer. 
                #Callback function uses the respective activation function as requested by the user.
            else:
                yhat = self.sigmoid(z)
                #Sigmoid function is the dedault activation for the output layer
                #yhat is a vector containing final predicted values corresponding to all the samples in the X matrix.
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

        try:
            p= tp/(tp+fp)
            r=tp/(tp+fn)
            f1=(2*p*r)/(p+r)
            print("Confusion Matrix : ")
            print(cm)
            print("\n")
            print(f"Precision : {p}")
            print(f"Recall : {r}")
            print(f"F1 SCORE : {f1}")
        except ZeroDivisionError:
            print("ZeroDivisionError encountered while computing the confusion matrix. Run again.")
		
        
        
class normalizer:
    def __init__(self):
        self.means = {}
        self.stds = {}
    
    def fit_transform(self, X_train): #X_train is a dataframe.
        self.means = X_train.mean()
        self.stds = X_train.std()
        normalized_df = (X_train - self.means)/self.stds
        X_train = normalized_df
        return X_train
    
    def transform(self, X_test): #X_test is a dataframe.
        normalized_df = (X_test - self.means)/self.stds
        X_test = normalized_df
        return X_test


        
def preprocess(df):
    #computing (median/mean/mode) imputation values
    medianAge = df['Age'].median()
    meanWt = int(df['Weight'].mean())
    medianDP = df.mode()['Delivery phase'][0]
    medianHB = df['HB'].median()
    meanBP = df['BP'].median()
    education = df.mode()['Education'][0]
    residence = df.mode()['Residence'][0]

    # replace all null values with the corresponding column's imputated values (inplace)
    df['Age'].fillna(value=medianAge, inplace=True)
    df['Weight'].fillna(value=meanWt, inplace=True)
    df['Delivery phase'].fillna(value=medianDP, inplace=True)
    df['HB'].fillna(value=medianHB, inplace=True)
    df['BP'].fillna(value=meanBP, inplace=True)
    df['Education'].fillna(value=education, inplace=True)
    df['Residence'].fillna(value=residence, inplace=True)
    
    norm = normalizer()
    target = 'Result'
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train[['Age', 'Weight', 'HB', 'BP']] = norm.fit_transform(X_train[['Age', 'Weight', 'HB', 'BP']])
    X_test[['Age', 'Weight', 'HB', 'BP']] = norm.transform(X_test[['Age', 'Weight', 'HB', 'BP']])
    
    return X_train, X_test, y_train, y_test 


if __name__=='__main__':
    df = pd.read_csv('LBW_Dataset.csv')
    X_train, X_test, y_train, y_test = preprocess(df)
    layers = [9, 2, 3, 5, 1]
    alpha = 0.06
    num_iter = 200
    actvn_func = 'relu'
    model = NN(layers, alpha, num_iter, activation = actvn_func)    
    model.fit(X_train, y_train, verbose = 1)
    
    #Getting the training set accuracy
    y_pred = list(model.predict(X_train)[0])
    y_train_orig = list(y_train.values)  
    model.CM(y_train_orig, y_pred)
    print("++++++++++++++++++++++++++++++++++++++++")
    
    
    #Getting the testing set accuracy
    y_pred = list(model.predict(X_test)[0])
    y_test_orig = list(y_test.values)
    model.CM(y_test_orig, y_pred)