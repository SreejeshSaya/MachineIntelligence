'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
	entropy = 0
	target =  list(df.iloc[:, -1])
	size = len(target)
	categ = set(target)
	num_of_categ = len(categ) 
	for i in categ:
		p = target.count(i)/size
		entropy = entropy + (-p*np.log2(p))
	return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
	entropy_of_attribute = 0
	size = df.shape[0]
	uniq_attr_cat = set(list(df[attribute]))
	entropies = list()
	for i in uniq_attr_cat:
		df_sub = df.loc[df[attribute] == i]
		sub_size = df_sub.shape[0]
		entropies.append(get_entropy_of_dataset(df_sub))
		entropy_of_attribute = entropy_of_attribute + ((sub_size/size)*entropies[-1])
	return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
	information_gain = 0
	information_gain = get_entropy_of_dataset(df) - get_entropy_of_attribute(df, attribute)
	return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     
def get_selected_attribute(df):
   
	information_gains={}
	selected_column=''

	'''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''


	return (information_gains,selected_column)



'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''