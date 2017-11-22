import os
import numpy as np
import matplotlib as mtl

os.chdir("C:\\Users\\vinay.benny\\Documents\\Econometrics\\econometrics-pack\\python")

from matplotlib import pyplot
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

def show_image(pixels, width, height):
    
    image = pixels.reshape(width, height)
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.imshow(image, cmap = mtl.cm.Greys)
    pyplot.show()

def sigmoid(z_vector):
    return 1./(1. + np.exp(-1.* z_vector) )

class Neuralnetwork():
    
    def __init__(self, config):        
        # Let  us restrict ourselves to 1 hidden layer
        self.input_layer = config[0]
        self.output_layer = config[2]
        self.hidden_layer = config[1]
        
        # Since we have 3 layers, we will have 2 sets of weights
        # Initialise all these weights to 1 and biases to 0 - for now.
        self.weight_list = [np.random.randn(self.hidden_layer, self.input_layer),
                                 np.random.randn(self.output_layer, self.hidden_layer)]
        self.bias_list = [np.zeros(self.hidden_layer), np.zeros(self.output_layer)]
        self.z_vectors = [ [], [] ]
        self.activation_vectors = [ [], [], [] ]
        return
    
    def feedforward(self, activation_vector):
        # For a given input array, calculate the activations of each successive layer of neurons in the network.
        # For this, multiply the activations of the first layer with the weights, and add the biases of the next layer; then apply
        # the sigmoid function to obtain the activations of the next layer. Repeat this successively until we get to
        # the output layer
        i=0
        self.activation_vectors = [ [], [], [] ]
        self.activation_vectors[i].append(activation_vector)
        for weight, bias in zip(self.weight_list, self.bias_list):
            z_vector = np.dot(weight, np.transpose(activation_vector)) + bias  
            activation_vector = sigmoid(z_vector)
            self.z_vectors[i].append(z_vector)
            self.activation_vectors[i+1].append(activation_vector)            
            i = i+1
            
        return
    
    def cost(self, actual, response):
        return np.square(actual - response)
    
    def dc_da(self, actual, response):
        return -2*(actual - response)
    
    def da_dz(self, activation_vector):
        return np.exp(-1 * self.activation_vectors[-1][0]) / ( np.square(1 + np.exp(-1*self.activation_vectors[-1][0]))  )
    

    
    def backprop(self, output_vector):
        
        delta_w = np.transpose(self.dc_da(output_vector, self.activation_vectors[-1])*self.da_dz(self.activation_vectors[-1]))*self.activation_vectors[-2]
        self.weight_list[-1] = self.weight_list[-1] - delta_w
        
        delta_w = np.transpose(np.sum(self.weight_list[-1], axis = 0)*self.da_dz(self.activation_vectors[-2]))*self.activation_vectors[-3]
        self.weight_list[-2] = self.weight_list[-2] - delta_w
    
        return

    
    
    
    
    
#Test code  
train_x = np.array([[0,0], [0,1], [1,0], [1,1]])  
train_y = np.array([[0, 1],[0,1],[0,1],[1,0]])
tester = Neuralnetwork([train_x.shape[1], 2, train_y.shape[1]])

for i in range(1, 100000):
    tester.feedforward(train_x[0])
    tester.backprop(train_y)
    
tester.activation_vectors










