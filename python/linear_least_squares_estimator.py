# Define a linear least squares estimator using Tensorflow
# Just testing out the tensorflow functionality

import tensorflow as tf
import numpy as np


# Define x as covariates and y as dependent variable
x = tf.placeholder(tf.float32, shape = (4,3))
y = tf.placeholder(tf.float32, shape = (4,1))

# Define linear least squares estimator
b = tf.matmul(tf.matmul( tf.matrix_inverse( tf.matmul( tf.transpose(x), x ) ) , tf.transpose(x)), y)

# Test the estimator
#sess = tf.Session()
#xval = np.matrix('1 3 8; 1 5 2; 2 1 5; 4 6 4')
#yval = np.matrix('1;2;3;4')
#print(sess.run(b, feed_dict={x: xval, y: yval}))
