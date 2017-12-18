# gradient_descent_alogrithm.py
# Stuart McKaige
# 9/14/2017
# This program implements a 2d gradient decent algorithm.
# An iteration method and a percision method are used to create two methods
# The methods in this file is ment to be used in other programs so a test data set will be run
# if this program is run as the main program

import math

#runn gradent descent using a set number of iterations
#alpha and max iterations are set as optional inputs incase a user wants to set them
def grad_Des_Iter(x_vals, y_vals, alpha=0.05, Max_Iter=200):
    #defining variables
    theta=[0.0,0.0]

    #running through the iterations to calculate the result
    for i in range(0,Max_Iter):
        theta, _ = update_values(theta,x_vals,y_vals,alpha)
           
    return theta

#runn gradent descent using a set amount of percesion
#alpha and percision are set as optional inputs incase a user wants to set them
def grad_Des_Percison(x_vals, y_vals, alpha=0.05, percision=0.0001):
    #defining variables
    theta=[0.0,0.0]

    #running through update untill two subeqent changes are less than pericison
    #update once so the while condition can be used
    theta,last_theta = update_values(theta,x_vals,y_vals,alpha)
    s=0 #this is used so the loop does not run indefently

    #run while the change in each theta is greater than our pression
    while math.fabs(theta[0]-last_theta[0])> percision and math.fabs(theta[0]-last_theta[0])> percision:
        theta,last_theta = update_values(theta,x_vals,y_vals,alpha)
        s+=1

        #if it takes more then 100000 stop running and give current result
        if s>100000:
            print("did not converge after 100000 iterations")
            break

    return theta

#this is the function that updates theta (and also returns the previouse theta
def update_values(theta,x_values,y_values,alpha):
    #set up variables needed for the update
    tmptheta=[0.0,0.0]
    last_theta=[e for e in theta]
    
    #calcuate a tmp var to calcualte the amount theta needs to be updated by
    for i in range(0,len(x_values)):
        tmptheta[0]+=theta[0]+theta[1]*x_values[i]-y_values[i]
        tmptheta[1]+=(theta[0]+theta[1]*x_values[i]-y_values[i])*x_values[i]
        
    #update the theta
    theta[0]=theta[0]-alpha*tmptheta[0]
    theta[1]=theta[1]-alpha*tmptheta[1]

    return theta, last_theta

#if this program is run as the main run this test data set
if __name__ == "__main__":

    #set up test data
    data_x=[1.0,2.0,3.0,4.0]
    data_y=[6.0,5.0,7.0,10.0]

    #run gradiadent descent on the test data
    result = grad_Des_Iter(data_x,data_y)
    print(result)

    result = grad_Des_Percison(data_x,data_y)
    print(result)
