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
from tf_dataset_from_dicoms import get_iterator, dicom_generator





####### Change the input dataset here################################################################


# Import CT Scan data
image_side_size = 512
num_input = image_side_size*image_side_size  # 512*512 is the dimensions of a CT Scan image in the Kaggle Data Science Bowl data set.

# This is the number of CT Scans that the dataset will return in each batch
batch_size = 50








######################################################################################################




# Training Parameters
learning_rate = 0.01
num_steps = 30000


display_step = 10
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

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

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data

        #print('Currently Running step {}'.format(i))


        # This line actually gets the batch from the dataset iterator
        my_iterator = get_iterator(dicom_generator, batch_size=batch_size)
        batch_x = my_iterator.get_next().eval()
        batch_x = np.reshape(batch_x, (batch_size, num_input))




        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))













    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((image_side_size * n, image_side_size * n))
    canvas_recon = np.empty((image_side_size * n, image_side_size * n))
    for i in range(n):
        # MNIST test set
        #batch_x, _ = mnist.test.next_batch(n)

        # This line actually gets the batch from the dataset iterator
        # THIS SHOULD ACTUALLY BE POINTING TO A TEST DATA SET
        my_iterator = get_iterator(dicom_generator, batch_size=batch_size)
        batch_x = my_iterator.get_next().eval()
        batch_x = np.reshape(batch_x, (batch_size, num_input))


        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * image_side_size:(i + 1) * image_side_size, j * image_side_size:(j + 1) * image_side_size] = \
                batch_x[j].reshape([image_side_size, image_side_size])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * image_side_size:(i + 1) * image_side_size, j * image_side_size:(j + 1) * image_side_size] = \
                g[j].reshape([image_side_size, image_side_size])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()











