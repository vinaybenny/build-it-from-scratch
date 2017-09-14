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


def apply_least_squares(x, y): 
    
    # Define a least squares named scope for tensorboard visualisation    
    with tf.name_scope('least_squares'):
        
        # Define linear least squares estimator using Moore Penrose pesudoinverse- ((X'X)^-1)X'y 
        inverse = tf.matrix_inverse( tf.matmul( tf.transpose(x), x ) )
        beta_estimate = tf.matmul(tf.matmul( inverse , tf.transpose(x)), y)
        
        # Get the variance-covariance matrix of the beta estimator.
        # The diagonal elements of this matrix gives us the variance of the estimated beta coefficients
        # For this, get the sum of squared errors between actual and estimated 'y'. 
        # Then apply (estimated_error^2)*((X'X)^-1) / (n- k)
        sum_sq_est_resid = tf.reduce_sum(tf.square(tf.subtract(y, tf.matmul(x, beta_estimate))), 
                                     axis = 0) / tf.cast( (tf.shape(x)[0] -tf.shape(x)[1]), tf.float32)
        beta_var_cov = tf.multiply(sum_sq_est_resid, inverse)
        #tf.summary.scalar('sum_squared_err', sum_sq_resid)        
        #beta_stderror = tf.diag_part(tf.sqrt(tf.multiply(sum_sq_resid, inverse)))
    
    # Return the estimated regression coefficients and the variance-covariance estimators of coefficients.
    return beta_estimate, beta_var_cov

def run_linear_model(xval, yval, intercept = True):
    
    # If intercept is to be included in the model, add a column of ones at the end of xval.
    if intercept:
        xval = np.hstack(( np.ones((xval.shape[0], 1)), xval ))
       
    # Generate placeholders for x and y    
    x = tf.placeholder(tf.float32, shape = (xval.shape[0], xval.shape[1]), name = 'x_input' )
    y = tf.placeholder(tf.float32, shape = (yval.shape[0], yval.shape[1]), name  = 'y_input' )
    
    # Invoke least squares
    beta_estimate, beta_var_cov = apply_least_squares(x,y)              
    merged_summary = tf.summary.merge_all()
    
    # Use a tensorflow session to run the code and get the estimates
    with tf.Session() as sess:        
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph = tf.get_default_graph() )        
        res1, res2 = sess.run([beta_estimate, beta_var_cov], feed_dict={x: xval, y: yval})
        
    return res1, res2


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
    xval = np.matrix( [ [1, 5, 5], [3, 2, 4],[8, 2, 6], [1, 1, 4] ], dtype = np.float32 )
    yval = np.matrix([ [1], [2], [3], [4] ], dtype = np.float32)
    a, b = run_linear_model(xval, yval)
    
