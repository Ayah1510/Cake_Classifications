import numpy as np
import pvml

# neural network that takes one of the images and gives us one of the classes
cnn = pvml.CNN.load("pvmlnet.npz")
mlp = pvml.MLP.load("cakes2.mlp.npz")

cnn.weights[-1]= mlp.weights[0][None, None, :, :]
cnn.biases[-1]= mlp.biases[0]

cnn.save("cakes2.npz")
