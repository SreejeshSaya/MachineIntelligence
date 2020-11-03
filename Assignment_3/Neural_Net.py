import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''

def normalize(df):
	#normalizing (scaling) [Age, Weight, HB, BP] columns
	newdf = df[['Age', 'Weight', 'HB', 'BP']]
	normalized_df = (newdf-newdf.mean())/newdf.std()
	df[['Age', 'Weight', 'HB', 'BP']] = normalized_df[['Age', 'Weight', 'HB', 'BP']]
	return df

def mse(y_true, y_pred):
    """
    Mean Square Error(MSE) error/cost function
    Input:
    ---
    y_true: ground-truth vector
    y_pred: prediction vector
    
    Output:
    ---
    cost: mse loss, <scalar>
    """
    n = y_pred.shape[1]
    cost = (1./(2*n)) * np.sum((y_true - y_pred) ** 2)
    return cost

def sigmoid(z):
    """
    Sigmoid activation function   
    Input:
    ---
    z: pre-activation vector at layer l
        n[l] x batch_size
       	z_l = w_l * a_(l-1) + b_l
        
    Output:
    --- 
    a => pointwise activation on z of layer l
    """
    a =  1 / (1 + np.exp(-z))
    return a
	
class NN:
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

	''' X and Y are dataframes '''
	
	def fit(self,X,Y):
		'''
		Function that trains the neural network by taking x_train and y_train samples as input
		'''
	
	def predict(self,X):

		"""
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
		
		return yhat

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
			


if __name__ == "__main__":
	model = NN
	model.preprocess('LBW_Dataset.csv', 'Final_LBW1.csv')
	df = pd.read_csv('Final_LBW1.csv')
	target = 'Result'
	X = df.drop(target, axis=1)
	y = df[target]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
	X_train = normalize(X_train)
	X_test = normalize(X_test)
	print(X_train)
	print(X_test)



