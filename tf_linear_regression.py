# gradient_descent_alogrithm.py
# Stuart McKaige
# 10/6/2017
# This program implements a 2d gradient decent algorithm and normal equation
# using tensorflow
# An iteration method and a percision method are used to create two methods
# The methods in this file is ment to be used in other programs so a test data set will be run
# if this program is run as the main program

import math
import tensorflow as tf
import time

#run gradent descent using a set number of iterations
#alpha and max iterations are set as optional inputs incase a user wants to set them
def tf_grad_Des_Iter(x_vals, y_vals, alpha=0.05, Max_Iter=2):
    #defining variables
    x_mat, y_mat, theta, sess=tf_setup(x_vals,y_vals)
    
    #run the update function for the number of iterations
    for i in range(0,Max_Iter):
        theta, _ = tf_update_values(theta,x_mat,y_mat,alpha,sess)
    #make theta a variable again and close the tensorflow session
    theta=sess.run(theta)
    sess.close()
    
    return theta

#run gradent descent using a set amount of percesion
#alpha and percision are set as optional inputs incase a user wants to set them
def tf_grad_Des_Percison(x_vals, y_vals, alpha=0.05, percision=0.0001):
    #defining variables
    x_mat, y_mat, theta, sess=tf_setup(x_vals,y_vals)
    
    #running through update untill two subeqent changes are less than pericison
    #update once so the while condition can be used
    theta,last_theta = tf_update_values(theta,x_mat,y_mat,alpha,sess)
    #create a tmp var to hold the value of current theta for the while condition
    cur_theta=sess.run(theta)
    s=0 #this is used so the loop does not run indefently
    
    #run while the change in each theta is greater than our pression
    while math.fabs(cur_theta[0]-last_theta[0])> percision and math.fabs(cur_theta[1]-last_theta[1])> percision:
        theta,last_theta = tf_update_values(theta,x_mat,y_mat,alpha,sess)
        cur_theta=sess.run(theta)
        s+=1
        
        #if it takes more then 100000 stop running and give current result
        if s>100000:
           print("did not converge after 100000 iterations")
           break
    
    #make theta a variable again and close the tensorflow session
    theta=sess.run(theta)
    sess.close()
    
    return theta

#this funtion sets up the matrices used in tensorflow
def tf_setup(x_values,y_values):
    #start tensor flow
    tf_session=tf.Session()
    
    #make and shape all of the matrices
    theta=[0.0,0.0]
    theta=tf.reshape(theta,[2,1])
    ones=[1 for e in x_values]
    x_matrix=ones+x_values
    x_matrix=tf.reshape(x_matrix,[2,len(x_values)])
    x_matrix=tf.transpose(x_matrix)
    y_matrix=tf.reshape(y_values,[len(y_values),1])

    return x_matrix, y_matrix, theta, tf_session

#this is the function that updates theta (and 
#also returns the previouse theta
def tf_update_values(theta,x_values,y_values,alpha,sess):
    #set up variables needed for the update
    last_theta=sess.run(theta)
    
    #calcule the new theta value
    tmp=tf.matmul(x_values,theta)
    tmptheta=tf.transpose(tmp-y_values)
    tmptheta=tf.matmul(tmptheta,x_values)
    tmptheta=tf.scalar_mul(alpha,tmptheta)  
    tmptheta=tf.transpose(tmptheta)
    theta=tf.subtract(theta,tmptheta)
    
    return theta, last_theta

#this is the update function for logistic gradiant decent
def tf_update_logis_values(theta,x_values,y_values,alpha,sess):

    #set up variables needed for the update
    last_theta=sess.run(theta)
    theta_trans=tf.transpose(theta)
    x_trans=tf.transpose(x_values)
    h_theta=tf.matmul(theta_trans,x_trans)
    tmptheta=tf.subtract(y_values,h_theta)
    tmptheta=tf.matmul(tmptheta,x_value)
    tmptheta=tf.scalar_mul(alpha,tmptheta)
    theta=tf.add(theta,tmptheta)
    return theta, last_theta



def tf_normal(x_vals,y_vals):
    #defining variables
    x_mat, y_mat, theta, sess=tf_setup(x_vals,y_vals)
    
    #calculate the normal equation
    x_trans=tf.transpose(x_mat)
    result=tf.matmul(x_trans,x_mat)
    result=tf.matrix_inverse(result)
    result=tf.matmul(result,x_trans)
    result=tf.matmul(result,y_mat)
    
    #make theta a variable again and close the tensorflow session
    result=sess.run(result)
    sess.close()
    return result

#if this program is run as the main run this test data set
if __name__ == "__main__":

    #set up test data
    data_x=[3.61, 3.67, 4.0, 3.19, 2.93, 3.0]
    data_y=[0.0, 1.0, 1.0, 1.0, 0.0, 1.0]
    #start_time=time.clock()
    #run gradiadent descent on the test data
    #result = tf_grad_Des_Iter(data_x,data_y)
    #end_time=time.clock()
    #print("the program took %.3f seconds to run" % (end_time-start_time))
    #print("iterative grad descent result")
    #print(result)
    
    #start_time=time.clock()
    #result = tf_grad_Des_Percison(data_x,data_y)
    #end_time=time.clock()
    #print("the program took %.3f seconds to run" % (end_time-start_time))    
    #print("percison grad descent result")
    #print(result)

    #run the normal equation on the sample data
    #start_time=time.clock()
    #result= tf_normal(data_x,data_y)
    #end_time=time.clock()
    #print("the program took %.3f seconds to run" % (end_time-start_time))
    #print("normal equation result")
    #print(result)

    start_time=time.clock()
    #run gradiadent descent logistic on the test data
    result = tf_grad_Des_Iter(data_x,data_y)
    end_time=time.clock()
    print("the program took %.3f seconds to run" % (end_time-start_time))
    print("iterative grad descent result")
    print(result)
