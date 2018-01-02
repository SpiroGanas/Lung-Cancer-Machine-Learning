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
batch_size = 250








######################################################################################################




# Training Parameters
learning_rate = 0.1  # originally 0.01
num_steps = 1_000_000     # This is the number of epochs, originally 30,000
model_path = "C:\\git_repos\\Lung-Cancer-Machine-Learning\\checkpoints\\"  # This is where the model checkpoint will be saved

display_step = 1 #originally 10
examples_to_show = 10

# Network Parameters
num_hidden_1 = 512 # 1st layer num features, originally 256
num_hidden_2 = 256 # 2nd layer num features (the latent dim), originally 125



# This line actually gets the batch from the dataset iterator
#MyData = get_iterator(dicom_generator_local, batch_size=batch_size, epochs = 2)
#X = MyData.get_next()


X = tf.placeholder("float", [1,512,512,1])

#training_image = 'C:\\git_repos\\Lung-Cancer-Machine-Learning\\data\\917ffef820759d2162792dfbcd7a8c35.dcm'
#pixels = dicom.read_file(training_image).pixel_array



import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('C:\\git_repos\\Lung-Cancer-Machine-Learning\\data\\Simple.jpg')
pixels = rgb2gray(img)


pixels = np.reshape(pixels,(1,512,512,1))






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

# This allows me to use the tensorboard
tf.summary.scalar('sigmoid_cross_entropy_with_logits', cost)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()




# set up a saver to save my model
saver = tf.train.Saver(max_to_keep=3)




# Start Training
# Start a new TF session
with tf.Session() as sess:




    # Restore the parameters from the prior run
    try:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        print("Restored the previous run successfully!")
    except ValueError:
        # Initialize the variables (i.e. assign their default value)
        print("Could not restore a prior run.")
        sess.run(tf.global_variables_initializer())



    train_writer = tf.summary.FileWriter('C:\\git_repos\\Lung-Cancer-Machine-Learning\\\\tensorboard_logs', sess.graph)



#    start_time = timeit.default_timer()

    # This deterimes how many epochs to use.
    for i in range(num_steps):

        Mycounter = 0


        _, l, summary = sess.run([optimizer, cost, merged], feed_dict={X: pixels})

#        print('Run Time: {}'.format(timeit.default_timer() - start_time))


        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Epoch %i: Minibatch Loss: %f' % (i+1, l))

         # Save a checkpoint every 50 epochs
        if i % 100 ==0:
            save_path = saver.save(sess, model_path, global_step=i)
            print("Model saved in file: %s" % save_path)











