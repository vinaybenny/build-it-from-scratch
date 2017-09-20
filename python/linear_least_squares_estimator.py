# ========================================================================================
# Purpose: Demonstrate the inner workings of a least squares estimator using Tensorflow.
#    This can then be compared against other kinds of estimator. Tensorboard will be used
#    to visualise the components.
# ========================================================================================


import os
import argparse
import tensorflow as tf
import numpy as np

# Define a global variable to store global parameters 
FLAGS = None


def apply_least_squares(x, y, robust_errors = False): 
    
    # Define a least squares named scope for tensorboard visualisation    
    with tf.name_scope('least_squares'):
        
        # Define linear least squares estimator using Moore Penrose pesudoinverse- ((X'X)^-1)X'y 
        inverse = tf.matrix_inverse( tf.matmul( tf.transpose(x), x ) )
        beta_estimate = tf.matmul(tf.matmul( inverse , tf.transpose(x)), y)
        
        # Get the estimated error in y.
        est_err = tf.subtract(y, tf.matmul(x, beta_estimate))        
             
        if robust_errors == True:
            beta_var_cov = white_robust_std_errors(x, est_err, inverse = inverse)
        else:
            beta_var_cov = homoskedastic_std_errors(x, est_err, inverse = inverse)
        
        # Std Errors of regression coefficients      
        beta_stderror = tf.sqrt(tf.matrix_diag_part(beta_var_cov))
        beta_stderror = tf.reshape(beta_stderror, shape = [ tf.shape(beta_stderror)[0] ,1])
        
        # Calculate t-statistic for the beta estimates
        t_stat = tf.truediv(beta_estimate, beta_stderror)
        df = tf.cast( (tf.shape(x)[0] - tf.shape(x)[1]), tf.float32)
        
        # Perform a 2-sided hypothesis test for the beta estimates
        t_dist = tf.contrib.distributions.StudentT(df=df, loc=0.0, scale=1.0)
        probs = tf.multiply(2., tf.subtract(1., t_dist.cdf(t_stat)))
        
    
    # Return the estimated regression coefficients and std errors.
    return tf.concat([beta_estimate, beta_stderror, t_stat, probs], axis=1)

def homoskedastic_std_errors(x, est_error, inverse = None):
    # Get the variance-covariance matrix of the beta estimator.
    # The diagonal elements of this matrix gives us the variance of the estimated beta coefficients
    # For this, get the sum of squared errors between actual and estimated 'y'.
    # Then apply (estimated_error^2)*((X'X)^-1) / (n- k)
    with tf.name_scope('homoskedastic_errors'):
        if inverse == None:
            inverse = tf.matrix_inverse( tf.matmul( tf.transpose(x), x ) )
            
        sq_est_resid = tf.reduce_sum(tf.square(est_error), 
                                             axis = 0) / tf.cast( (tf.shape(x)[0] - tf.shape(x)[1]), tf.float32)
        beta_var_cov = tf.multiply(sq_est_resid, inverse) 
    return beta_var_cov
    

def white_robust_std_errors(x, est_error, inverse = None):
    # In case of heteroskedastic erros, use a robust error estmation
    # var_cov(b_estimate) = (n/n-k)*((X'X)^-1) X' ee' X ((X'X)^-1)
    # Assume no serial correlation in the errors.
    with tf.name_scope('white_robust_std_errors'):
        if inverse == None:
            inverse = tf.matrix_inverse( tf.matmul( tf.transpose(x), x ) )
        
        # Create a diagonal matrix with the variance of error terms and 0 for covariance error terms
        sq_est_resid =  tf.diag(tf.diag_part(tf.matmul(est_error, tf.transpose(est_error) ) ) )
        beta_var_cov = tf.multiply(
                tf.matmul( tf.matmul(inverse, tf.matmul(tf.matmul(tf.transpose(x), sq_est_resid), x) ), inverse),
                 tf.cast( (  tf.shape(x)[0]/ (tf.shape(x)[0] - tf.shape(x)[1]) ), tf.float32))
    return beta_var_cov    
    
    


def run_linear_model(xval, yval, intercept = True):
    
    # If intercept is to be included in the model, add a column of ones at the end of xval.
    if intercept:
        xval = np.hstack(( np.ones((xval.shape[0], 1)), xval ))
       
    # Generate placeholders for x and y    
    x = tf.placeholder(tf.float32, shape = (xval.shape[0], xval.shape[1]), name = 'x_input' )
    y = tf.placeholder(tf.float32, shape = (yval.shape[0], yval.shape[1]), name  = 'y_input' )
    
    # Invoke least squares
    ls_output = apply_least_squares(x,y, robust_errors = False) 
    merged_summary = tf.summary.merge_all()
    
    # Use a tensorflow session to run the code and get the estimates
    with tf.Session() as sess:        
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph = tf.get_default_graph() )        
        res1 = sess.run([ls_output], feed_dict={x: xval, y: yval})
        
    return res1


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--log_dir',
          type=str,
          default=os.path.join('C:\\Users\\vinay.benny\\Documents\\Econometrics\\econometrics-pack\\logs'),
          help='Directory to put the log data.'
      )
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.gfile.MakeDirs(FLAGS.log_dir)
    # Test the estimator
    xval = np.matrix( [ [1, 2, 6], [3, 1, 3],[8, 5, 8], [1, 4, 1], [5, 6, 4], [2, 4, 8] ], dtype = np.float32 )
    yval = np.matrix([ [1], [2], [3], [4], [5], [1.5] ], dtype = np.float32)
    a = run_linear_model(xval, yval)
    
