import numpy as np
import scipy.misc
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model

import NeuralNet

"""
This script loads mnist data, then trains a simple autoencoder using first NeuralNet.py (using Numpy as back end), 
then using Keras with TensorFlow as back end as benchmark. 
Sample auto-encoder output is saved to an image file. 
"""


#######################################
#                                     #
#           Load MNIST data           #
#                                     #
#######################################
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((60000, 28 * 28))


#######################################
#                                     #
#         Using NeuralNet.py          #
#                                     #
#######################################

# Define AutoEncoder model for NeuralNet.py
ae_model = [
    {'type': 'input', 'units': 28 * 28},
    {'type': 'lin_act', 'units': 128, 'activation': 'relu'},
    {'type': 'lin_act', 'units': 64, 'activation': 'relu'},
    {'type': 'lin_act', 'units': 32, 'activation': 'relu'},
    {'type': 'lin_act', 'units': 64, 'activation': 'relu'},
    {'type': 'lin_act', 'units': 128, 'activation': 'relu'},
    {'type': 'lin_act', 'units': 28 * 28, 'activation': 'sigmoid'},
    {'type': 'cost', 'cost_function': 'cross-entropy'}
]

# Set parameters
meta_params = {
    'regularization_lambda': 0.0,
    'batch_norm': {'use': True, 'epsilon': 1e-8},
    'print_cost_interval': 1,
    'optim': {'algo': 'adam', 'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999,
              'epsilon': 1e-8, 'mini_batch_size': 128, 'update_counter': 0.0}
}

# Train auto-encoder using NerualNet
nn = NeuralNet.NeuralNet()
nn.load_meta_params(meta_params)
nn.load_data({'x': x_train.T, 'y': x_train.T})
nn.load_model(ae_model)
nn.initialize_parameters()
nn.train(epochs=100)


#######################################
#                                     #
#       Using Keras & Tensorflow      #
#                                     #
#######################################

# Define Auto-Encoder in Keras 
input_img = Input(shape=(28 * 28,))
layer = Dense(128, activation='relu')(input_img)
layer = Dense(64, activation='relu')(layer)
layer = Dense(32, activation='relu')(layer)
layer = Dense(64, activation='relu')(layer)
layer = Dense(128, activation='relu')(layer)
output_img = Dense(784, activation='sigmoid')(layer)

autoencoder = Model(input_img, output_img)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True)


#######################################
#                                     #
#    Save example AE output to img    #
#                                     #
#######################################

x = x_train[0:128, :]
x_hat_nn = nn.predict(x.T)
x_hat_k = autoencoder.predict(x)
nb_images = 10
img = list()
for c in range(nb_images):
    img_in = 255 * x[c, :].T.reshape(28, 28)
    img_nn = 255 * x_hat_nn[:, c].reshape(28, 28)
    img_k = 255 * x_hat_k[c, :].reshape(28, 28)
    img.append(np.vstack([img_in, img_nn, img_k]))
img = np.hstack(img)
scipy.misc.imsave('img\mnist_ae_in_nn_k.jpg', img)

