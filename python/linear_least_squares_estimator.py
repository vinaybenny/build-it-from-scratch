# Define a linear least squares estimator using Tensorflow
# Just testing out the tensorflow functionality

import tensorflow as tf
import numpy as np


# Define x as covariates and y as dependent variable from input matrix
x = tf.placeholder(tf.float32, shape = (xval.shape[0],xval.shape[1]))
y = tf.placeholder(tf.float32, shape = (xval.shape[0],1))

# Define linear least squares estimator using Moore Penrose pesudoinverse- ((X'X)^-1)X'y 
inverse = tf.matrix_inverse( tf.matmul( tf.transpose(x), x ) )
b_est = tf.matmul(tf.matmul( inverse , tf.transpose(x)), y)

# Get the std error of the beta estimator.  (sigma^2)*((X'X)^-1)
sum_sq_resid = tf.reduce_sum(tf.square(tf.subtract(y, tf.matmul(x, b_est))), axis = 0)
b_stderr = tf.diag_part(tf.sqrt(tf.multiply(sum_sq_resid, inverse)))


# Test the estimator
#sess = tf.Session()
#xval = np.matrix( [ [1, 5, 5], [3, 2, 4],[8, 2, 6], [1, 1, 4] ], dtype = np.float32 )
#yval = np.matrix([ [1], [2], [3], [4] ], dtype = np.float32)
#print(sess.run([b_est, b_stderr], feed_dict={x: xval,y: yval}))
