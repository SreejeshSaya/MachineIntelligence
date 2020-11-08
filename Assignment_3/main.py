from Neural_Net import *


def settings(layers, alpha, num_iter, actvn_func):
    print("layers   : ",layers)
    print("alpha    : ",alpha)
    print("iters    : ",num_iter)
    print("actvn    : ",actvn_func)
    
    
    
    
if __name__ == "__main__":
    #preprocess('LBW_Dataset.csv', 'Final_LBW.csv')
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
    actvn_func = 'relu'
    model = NN(layers, alpha, num_iter, activation = actvn_func)
    settings(layers, alpha, num_iter, actvn_func)
    model.fit(X_train, y_train, verbose = 0)
    
    #Getting the training set accuracy
    y_pred = list(model.predict(X_train)[0])
    y_train_orig = list(y_train.values)  
    model.CM(y_train_orig, y_pred)
    print("++++++++++++++++++++++++++++++++++++++++")
    #Getting the testing set accuracy
    y_pred = list(model.predict(X_test)[0])
    y_test_orig = list(y_test.values)
    model.CM(y_test_orig, y_pred)
    
    
#--------------------------------------------------------------------------------------------------------------------------    
    print("\n\n====================================")
    actvn_func = 'tanh'
    model = NN(layers, alpha, num_iter, activation = actvn_func)
    settings(layers, alpha, num_iter, actvn_func)
    model.fit(X_train, y_train, verbose = 0)
    
    #Getting the training set accuracy
    y_pred = list(model.predict(X_train)[0])
    y_train_orig = list(y_train.values)  
    model.CM(y_train_orig, y_pred)
    print("++++++++++++++++++++++++++++++++++++++++")
    #Getting the testing set accuracy
    y_pred = list(model.predict(X_test)[0])
    y_test_orig = list(y_test.values)
    model.CM(y_test_orig, y_pred)
 
