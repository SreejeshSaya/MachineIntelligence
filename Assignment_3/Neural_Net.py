import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
        #np.random.seed(10)
        for i in range(1, len(layers)):
            if(i !=(len(layers)-1) ):
            	#He init
                self.params[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(2/self.layers[i-1])
            else:
            	#Xavier init
                self.params[f"W{i}"] = np.random.randn(self.layers[i], self.layers[i-1]) * np.sqrt(1/self.layers[i-1])
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
    
    
    def tanh(self, x):
        return np.tanh(x)
    
    
    def der_tanh(self, x):
        return (1 - np.square(self.tanh(x)))
    
    
    def der_relu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
       
    
    def forward_propagation(self):
        self.params["A0"] = self.X
        #print(self.X.shape)
        reg_weights = 0
        for i in range(1, len(self.layers)):
            #print(self.params[f"A{i-1}"].T.shape)
            self.params[f"Z{i}"] = self.params[f"W{i}"].dot(self.params[f"A{i-1}"]) + self.params[f"b{i}"]
            if(i!=(len(self.layers)-1)):
                #self.params[f"A{i}"] = self.relu(self.params[f"Z{i}"])
                self.params[f"A{i}"] = self.tanh(self.params[f"Z{i}"])
            else:
                yhat = self.sigmoid(self.params[f"Z{i}"])
            #check this whether to take wi or zi ---> idk ....... formula says sum of squares of weights
            reg_weights += np.sum(np.square(self.params[f"W{i}"]))
        m = len(self.y)
        #L2 Regularization
        l2_reg = (1/(2*m)) * reg_weights
        try:
        	#cross entropy error loss with l2_regularizer
        	Error = -(1/m) * (np.sum(np.multiply(np.log(yhat), self.y) + np.multiply((1 - self.y), np.log(1 - yhat)))) - l2_reg
        except ZeroDivisionError:
        	#print(yhat)
        	Error = 0
        	#Error = l2_reg
        return yhat, Error
                
            
    def backward_propagation(self, yhat):
        self.grads[f"dA{len(self.layers)-1}"] = -(np.divide(self.y,yhat) - np.divide((1 - self.y),(1-yhat)))
        m = len(self.y)
        for i in range(len(self.layers)-1, 0, -1):
            if(i ==(len(self.layers)-1) ):
                sig = self.sigmoid(self.params[f"Z{i}"])
                self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * sig * (1-sig)
            else:
                #self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * self.der_relu(self.params[f"Z{i}"])
                self.grads[f"dZ{i}"] = self.grads[f"dA{i}"] * self.der_tanh(self.params[f"Z{i}"])
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
               #A_prev  = self.relu(z)
                A_prev  = self.tanh(z)
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

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
		
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")


        
def preprocess(in_path, out_path):
    df = pd.read_csv(in_path)

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

    df.to_csv(out_path, index=False)

		
    
if __name__ == "__main__":
    preprocess('LBW_Dataset.csv', 'Final_LBW.csv')
    n1 = normalizer()
    df = pd.read_csv('Final_LBW.csv')
    target = 'Result'
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train[['Age', 'Weight', 'HB', 'BP']] = n1.fit_transform(X_train[['Age', 'Weight', 'HB', 'BP']])
    X_test[['Age', 'Weight', 'HB', 'BP']] = n1.transform(X_test[['Age', 'Weight', 'HB', 'BP']])    
    #print(X_train)
    #print(X_test)
    
    layers = [9, 8, 5, 3, 1]
    alpha = 0.08
    num_iter = 200
    model = NN(layers, alpha, num_iter)
    model.fit(X_train, y_train)
    
    #Getting the training set accuracy
    y_pred = list(model.predict(X_train)[0])
    y_train_orig = list(y_train.values)  
    model.CM(y_train_orig, y_pred)
    
    #Getting the testing set accuracy
    y_pred = list(model.predict(X_test)[0])
    y_test_orig = list(y_test.values)
    model.CM(y_test_orig, y_pred)
        
