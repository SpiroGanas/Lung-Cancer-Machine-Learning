



#This is not yet
#not working
#I need to figure out how to restore the encoder and decoder ops from the saved model.







# Spiro Ganas
# 12/29/17
#
# This loads the model from the checkpoint file and then runs an CT scan through it

# Derived from this code:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
#
# I updated the code to replace the MNIST data with a TensorFlow dataset that fed the model CT Scan images
# stored on AWS S3.


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dicom
from tf_dataset_from_dicoms import get_iterator, dicom_generator_local
#from autoencoder_CNN import encoder, decoder






####### Change the input dataset here################################################################


# Import CT Scan data
image_side_size = 512
num_input = image_side_size*image_side_size  # 512*512 is the dimensions of a CT Scan image in the Kaggle Data Science Bowl data set.

# This is the number of CT Scans that the dataset will return in each batch
batch_size = 250








######################################################################################################




# Training Parameters
learning_rate = 0.01  # originally 0.01
num_steps = 10    # This is the number of epochs, originally 30,000
model_path = "./checkpoints/"  # This is where the model checkpoint will be saved

display_step = 1 #originally 10
examples_to_show = 10

# Network Parameters
num_hidden_1 = 512 # 1st layer num features, originally 256
num_hidden_2 = 256 # 2nd layer num features (the latent dim), originally 125



# This line actually gets the batch from the dataset iterator
MyData = get_iterator(dicom_generator_local, batch_size=batch_size, epochs = 2)
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
    ### Encoder

    conv1 = tf.layers.conv2d(inputs=x, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')

    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')

    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')

    return encoded


# Building the decoder
def decoder(x):

    ### Decoder
    upsample1 = tf.image.resize_images(x, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv4 = tf.layers.conv2d(inputs=upsample1, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    upsample2 = tf.image.resize_images(conv4, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv5 = tf.layers.conv2d(inputs=upsample2, filters=8, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    upsample3 = tf.image.resize_images(conv5, size=(512, 512), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    conv6 = tf.layers.conv2d(inputs=upsample3, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)


    logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3, 3), padding='same', activation=None)

    # Pass logits through sigmoid to get reconstructed image
    decoded = tf.nn.sigmoid(logits)
    return decoded



# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X


# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
# Get cost and define the optimizer
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
#init = tf.global_variables_initializer()















# set up a saver to save my model
saver = tf.train.Saver()



# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    #sess.run(init)

    # Restore the parameters from the prior run
    saver.restore(sess, tf.train.latest_checkpoint(model_path))






# Testing
    # Encode and decode images from test set and visualize their reconstruction.


    test_image = 'C:\\Users\\Administrator\\Desktop\\ct_scans\\00cba091fa4ad62cc3200a657aeb957e\\b792dbfc27d50dc33e1e0f4b47401a47.dcm'
    pixels = dicom.read_file(test_image).pixel_array

    canvas_orig = pixels
    canvas_recon = sess.run(y_pred, feed_dict={X: np.resize(pixels, (1, 512, 512, 1))})
    canvas_recon = np.resize(canvas_recon[0],(512,512))



    #print("Original Images")
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    #print("Reconstructed Images")
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()



