# Spiro Ganas
# 12/29/17
#
# This loads the model from the checkpoint file and then runs an CT scan through it

# Derived from this code:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
#
# I updated the code to replace the MNIST data with a TensorFlow dataset that fed the model CT Scan images
# stored on AWS S3.





""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dicom
import timeit
from tf_dataset_from_dicoms import get_iterator, dicom_generator_local





####### Change the input dataset here################################################################


# Import CT Scan data
image_side_size = 512
num_input = image_side_size*image_side_size  # 512*512 is the dimensions of a CT Scan image in the Kaggle Data Science Bowl data set.

# This is the number of CT Scans that the dataset will return in each batch
batch_size = 3604








######################################################################################################




# Training Parameters
learning_rate = 0.01  # originally 0.01
num_steps = 101    # This is the number of epochs, originally 30,000
model_path = "./checkpoints/model.ckpt"  # This is where the model checkpoint will be saved

display_step = 1 #originally 10
examples_to_show = 10

# Network Parameters
num_hidden_1 = 512 # 1st layer num features, originally 256
num_hidden_2 = 256 # 2nd layer num features (the latent dim), originally 125



# This line actually gets the batch from the dataset iterator
MyData = get_iterator(dicom_generator_local, batch_size=batch_size, epochs = 10)
X = MyData.get_next()








#X = tf.placeholder("float", [None, num_input])


weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# set up a saver to save my model
saver = tf.train.Saver()



# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    #sess.run(init)

    # Restore the parameters from the prior run
    saver.restore( sess, model_path)






# Testing
    # Encode and decode images from test set and visualize their reconstruction.


    test_image = 'C:\\Users\\Administrator\\Desktop\\ct_scans\\00cba091fa4ad62cc3200a657aeb957e\\b792dbfc27d50dc33e1e0f4b47401a47.dcm'
    pixels = dicom.read_file(test_image).pixel_array

    canvas_orig = pixels
    canvas_recon = sess.run(decoder_op, feed_dict={X: np.resize(pixels, (1, 512*512))})
    canvas_recon = np.resize(canvas_recon[0],(512,512))



    #print("Original Images")
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    #print("Reconstructed Images")
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()



