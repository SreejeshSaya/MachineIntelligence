# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split




class Normalizer:
    def __init__(self):
        self.means = {} #dictionary - {key, value} :: {attribute_name, attribute's mean}
        self.stds = {}  #dictionary - {key, value} :: {attribute_name, attribute's standard deviation}
    
    def fit_transform(self, X_train): # X_train is a dataframe.
        self.means = X_train.mean() #calculates mean or average values for every attribute of the X_train dataframe
        self.stds = X_train.std() #calculates standard deviation values for every attribute of the X_train dataframe
        # normalization => z = (X(i) - mean)/std_deviation
        normalized_df = (X_train - self.means)/self.stds #holds the normalised datframe of X_train 
        return normalized_df
    
    def transform(self, X_test): # X_test is a dataframe.
        normalized_df = (X_test - self.means)/self.stds #normlizes X_test based on mean and standard deviation computed on X_train dataframe
        return normalized_df


        
def data_cleaning(df):
    # Compute (median/mean/mode) imputation values
    # For the null values present in the dataset, we have chosen either the mean, median or mode 
    # to fill in depending on the appropriateness of the values for the column
    medianAge = df['Age'].median()
    meanWt = int(df['Weight'].mean())
    modeDP = df.mode()['Delivery phase'][0]
    medianHB = df['HB'].median()
    meanBP = df['BP'].median()
    education = df.mode()['Education'][0]
    residence = df.mode()['Residence'][0]

    # Replace all null values with the corresponding column's imputated values (inplace)
    df['Age'].fillna(value=medianAge, inplace=True)
    df['Weight'].fillna(value=meanWt, inplace=True)
    df['Delivery phase'].fillna(value=modeDP, inplace=True)
    df['HB'].fillna(value=medianHB, inplace=True)
    df['BP'].fillna(value=meanBP, inplace=True)
    df['Education'].fillna(value=education, inplace=True)
    df['Residence'].fillna(value=residence, inplace=True)
    
def normalize(df):
    # Normalizer object to normalize the training and testing sets
    norm = Normalizer()

    # Separate the feature set and the target variable and store in X and y respectively
    target = 'Result'
    X = df.drop(target, axis=1)
    y = df[target]

    # Splitting the dataset into training set and testing set using sklearn library
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    # Copying the df to avoid SettingWithCopy Warning - Pandas
    X_train = X_train.copy()
    X_test = X_test.copy()
    y_train = y_train.copy()
    y_test = y_test.copy()

    # Normalize the numerical values in the dataset
    X_train.loc[:, ['Age', 'Weight', 'HB', 'BP']] = norm.fit_transform(X_train[['Age', 'Weight', 'HB', 'BP']])
    X_test.loc[:, ['Age', 'Weight', 'HB', 'BP']] = norm.transform(X_test[['Age', 'Weight', 'HB', 'BP']])
    return X_train, X_test, y_train, y_test 