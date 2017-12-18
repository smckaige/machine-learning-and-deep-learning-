# logistic.py
# Stuart McKaige
# 11/28/2017
# This program implements a logistic regression algorithm 
# The methods in this file is ment to be used in other programs so a test data set 
# will be run if this program is run as the main program
import numpy as np
import matplotlib.pyplot as plt

#function to calculate h (in this case the sigmoid function)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#function to run the logistic regression with optional arguments
def logistic_regression(x_vals, y_vals, Max_Iter= 500, alpha= 0.0005):
    #creating the theta var as all zeros to start
    theta = np.zeros(x_vals.shape[1])
    
    #run for the number of steps 
    for i in range(0,Max_Iter):
        #calulating the new perdictions
        tmp = np.dot(x_vals, theta)
        predictions = sigmoid(tmp)

        #calculating the error
        error= y_vals - predictions
        
        #updating theta
        grad= np.dot(x_vals.T, error)
        theta += alpha * grad       
    return theta

#function to run perception learning
def perceptron_learning(x_vals, y_vals,Max_Iter= 500,alpha=1):
    #creating the theta var as all zeros to start
    theta = np.zeros(len(x_vals[0]))

    #run for the number of steps 
    for step in range(0,Max_Iter):
        for i, x in enumerate(x_vals):
            #if sign(x*theata) is not equal y update theta otherwise do nothing
            if (np.sign(np.dot(x_vals[i], theta))) != y_vals[i]:
                theta += alpha*x_vals[i]*y_vals[i]

    return theta

if __name__ == "__main__":
    #genrating random data
    np.random.seed(12)
    num_observations = 500
    x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
    sim_feat = np.vstack((x1, x2)).astype(np.float32)
    sim_results = np.hstack((np.zeros(num_observations),np.ones(num_observations)))
   
    #running logistic regresiion
    result= logistic_regression(sim_feat, sim_results)
    print(result)

    #running proception learning
    result = perceptron_learning(sim_feat,sim_results)
    print(result)
