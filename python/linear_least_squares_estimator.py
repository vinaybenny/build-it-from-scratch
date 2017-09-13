# Define a linear least squares estimator using Tensorflow
# Just testing out the tensorflow functionality

import os
import argparse
import tensorflow as tf
import numpy as np

FLAGS = None


def apply_least_squares(x, y): 
    
    with tf.name_scope('least_squares'):
        # Define linear least squares estimator using Moore Penrose pesudoinverse- ((X'X)^-1)X'y 
        inverse = tf.matrix_inverse( tf.matmul( tf.transpose(x), x ) )
        beta_estimate = tf.matmul(tf.matmul( inverse , tf.transpose(x)), y)
        
        # Get the std error of the beta estimator.  (sigma^2)*((X'X)^-1)
        sum_sq_resid = tf.reduce_sum(tf.square(tf.subtract(y, tf.matmul(x, beta_estimate))), axis = 0)
        tf.summary.scalar('sum_squared_err', sum_sq_resid)
        beta_var_cov = tf.multiply(sum_sq_resid, inverse)
        #beta_stderror = tf.diag_part(tf.sqrt(tf.multiply(sum_sq_resid, inverse)))
        
    return beta_estimate, beta_var_cov

def run_linear_model():
    xval = np.matrix( [ [1, 5, 5], [3, 2, 4],[8, 2, 6], [1, 1, 4] ], dtype = np.float32 )
    yval = np.matrix([ [1], [2], [3], [4] ], dtype = np.float32)
    
    # Generate placeholders for x and y    
    x = tf.placeholder(tf.float32, shape = (xval.shape[0], xval.shape[1]), name = 'x_input' )
    y = tf.placeholder(tf.float32, shape = (yval.shape[0], yval.shape[1]), name  = 'y_input' )
    
    beta, cov = apply_least_squares(x,y)              
    merged_summary = tf.summary.merge_all()
        
    with tf.Session() as sess:        
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, graph = tf.get_default_graph() )        
        res1, res2 = sess.run([beta, cov], feed_dict={x: xval, y: yval})
        
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
    a, b = run_linear_model()
    
