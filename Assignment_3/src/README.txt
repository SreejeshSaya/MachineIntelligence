DATA CLEANING
    The missing values in the numerical attributes such as Age, Weight, HB, BP were imputed with the means/medians of those respective attributes, whichever was better suited.
    The missing values in the categorical attributes such as Delivery Phase, Education and Residence were imputed with the modes of the respective attributes.
    The resulting dataset was written into the file named 'processedLBW_Dataset.csv'.




NORMALIZATION
    The cleaned dataset was read from 'processedLBW_Dataset.csv' and split into training and testing sets, with a testing size of 30%, setting a random state to maintain consistency     throughout the training process.
    
    A normalizer class was created to normalize the dataset. This class would take in dataframes consisting of only numerical attributes and normalized all the attributes. The resulting means and variances of all the attributes would now be set to 0 and 1 respectively. This implementation is similar to that of the StandardScaler class in sklearn.preprocessing.
    
    The original dataframe read from the file 'processedLBW_Dataset.csv' was normalized using an object of the above mentioned class and was split into training and testing sets.




HYPERPARAMETER INITIALIZATION
        The following hyperparameters were passed to our neural network.
1. Number of layers and number of neurons in each layer : The input layer consists of 9 neurons since there are 9 features taken in as input from each record of the training set. The output layer consists of 1 neuron since we only need 1 output(Binary classification) for every record provided as input to the neural network. Hence these two values are set accordingly as the first and last elements of the list respectively.


Every element in the list between 9 and 1 corresponds to the number of neurons in that      
        particular layer. The configuration of the layers and neurons are as follows : 
        [9, 7, 7, 15, 9, 15, 5, 1]. This configuration was obtained after a very long 
        process of trial and error, but serves the purpose of providing accuracies beyond 85% 
        for the training and testing sets.


2. Epoch: The epoch was set to 250 [in the range 200-300]. We observed that in our testing, going below 200 would result in underfitting wherein the neural network couldn’t learn the features completely whereas setting it above 300 resulted in overfitting and the model failed miserably to classify the testing set with accuracy of barely above 70%


3. alpha = 0.6
Also known as the learning rate, alpha serves the purpose of controlling the step taken during gradient descent in order to reach the global minima. Our neural network gave the best performance when alpha is set to 0.6 for the above mentioned layer’s configuration.
    


   4. Activation function used : ‘ReLU’
This parameter corresponds to the type of activation function used at every hidden layer. Since our neural network has functionalities to include activations of both ReLU or tanh, we needed to specify one of them. Although we have implementations for both, we decided to pick ReLU as the amount of computation required was lesser and it provided better results.


   5. Verbose : This parameter has been included by us to print the value of the error/loss function after every epoch. If this parameter is set to 1, then the error is displayed at the end of every epoch, else it isn’t displayed. This parameter is passed as a parameter to the fit function of an object of the NN class.


   6. Lamda : This is the regularization parameter, which is used to calculate the regularization term, which is later added to the value of the error/loss function after every epoch, to prevent overfitting. However, we found that implementing it didn’t really help our neural network as we expected it to, hence we set Lamda to 0.




OUT OF THE BOX THINKING 
At the end of each epoch, the learning rate was reduced by a factor of 0.995^(epoch number) to help the loss function converge at the local/global minima (Learning Rate Decay).


To prevent overfitting, an additional information is subtracted from the loss, which the L2 regularization weight. We found that regularisation did not improve the evaluation metrics and hence lamda was set to 0 to not account for it.






NEURAL NETWORK CLASS  
        Initialization of Model [__init__] -
To create the model, we use the following hyperparameters - layers, alpha, epoch, activationFunction, verbose (if set to 1, prints the epoch number and loss). We then initialize the neural network by assigning random weights to every neuron of every layer which is stored in a dictionary. Likewise, we store the bias and the gradients (used in back propagation) in a dictionary.


Activation functions [relu, sigmoid, tanh] -
We have the following activation functions in our class - sigmoid, relu,  tanh. In our testing, we observed that the activation function relu for the hidden layers and sigmoid function for the output layer provided best results. For training the model, we have functions (callback) that return the derivative of the activation function chosen by the programmer.


Training the model [fit] -
To train the model we call the fit method of the model Object which takes in the training feature set and the expected output. Using the forward propagation method we determine the output of the neural network which is then passed to the back propagation method to modify the weights of the neural network to best fit the training dataset.




Testing the model [predict] -
To test/evaluate the model, we call the predict method which takes in the testing feature set and the expected output. We use the forward propagation method to pass the feature set as the input to the model which then returns a value between 0 and 1 indicating the probability of output 


Confusion Matrix [CM] - 
Given the output of the neural network, assign 0 or 1 based on the threshold (0.6) and display the confusion matrix along with metrics such as Precision, Recall, F1 score, Accuracy score, for a given set of ground truth values and predicted values.




TRAINING THE NEURAL NETWORK
        An object of the class NN was created and the hyperparameters mentioned(except) 
        above were passed to the init method. 


The model was fit on the training data. This meant passing the training data to the fit method and the training the neural network for the number of epochs mentioned earlier. In each epoch, the entire training data was passed to the forward propagation function (Batch processing) and the Error/loss obtained was passed to the backward propagation function to perform gradient descent. At the end of each epoch, the learning rate was reduced by a factor of 0.995^(epoch number) to help the loss function converge at the local/global minima (Learning Rate Decay).


In the forward propagation function, the training data was passed as inputs to the neural network. Iterating through the layers of the neural network, the linear combination of the weights of the neurons of each layer with the inputs obtained from the previous layer was taken, which were further added with the bias of the neurons of the same layer. This was further passed into the activation function of our choice(ReLU) and was set to be the output of that layer, which was passed as input to the next layer. Except the output layer, which used Sigmoid as the activation function, the neural network  maintained ReLU as the activation function for every other layer. The outputs obtained at the output layer were the predicted values. These values were later used to calculate the loss, combined with L2 Regularization which is a process of introducing additional information in order to prevent overfitting. The predicted values were passed to the backward propagation to perform gradient descent.


In the Backward propagation function, the predicted values and the original values of the output were used to calculate the derivatives of the loss function with respect to the weights and biases of every neuron of every layer. Chain rule was used to calculate the derivatives, also called gradients. These gradients were later multiplied with the learning rate and subtracted from the original weights/biases respectively to tune the weights and biases. After performing forward propagation and backward propagation for the specified number of epochs, the neural network was trained successfully.






RESULTS
The following configuration of our neural network model provided the best results in training and testing in our model testing. 
        Configuration: 
                Layers = [9, 7, 7, 15, 9, 15, 5, 1]
                alpha         = 0.6
Epoch        = 200
Activation Function: relu (Hidden Layer) sigmoid (Output Layer)
Result:
OVER TRAINING DATASET
Confusion Matrix :
[[17, 2], [4, 44]]


Precision : 0.9565217391304348
Recall : 0.9166666666666666
F1 SCORE : 0.9361702127659574
Accuracy score : 0.9104477611940298
---- ---- ---- ---- ----


OVER TESTING DATASET
Confusion Matrix :
[[1, 1], [2, 25]]


Precision : 0.9615384615384616
Recall : 0.9259259259259259
F1 SCORE : 0.9433962264150944
Accuracy score : 0.896551724137931
---- ---- ---- ---- ----