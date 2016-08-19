"""End-to-end, Variational Autoencoder (VAE) - MNIST
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy
from encoder import FC_Encoder
from decoder import FC_Decoder
from vae import VAE

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
EVAL_FREQUENCY = 500
TRAIN_SIZE = 60000
TEST_SIZE = 10000
PIXEL_DEPTH = 255
NUM_LABELS = 10
SEED = 66478  # Set to None for random seed.

NUM_EPOCHS = 75
BATCH_SIZE = 64
NUM_NODES = 500
NUM_LAYERS = 2
SIZE = 784 
LATENT_SIZE = 20
IMAGE_SIZE = 28

FLAGS = tf.app.flags.FLAGS

def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.Size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [0.0, 1.0].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32) / 255.0
    return data.reshape(num_images, SIZE)

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
    return labels

# Get the data.
train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

# Extract it into numpy arrays.
train_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
test_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)

data = tf.placeholder(tf.float32, shape=(BATCH_SIZE, SIZE))
encoder_network = FC_Encoder(SIZE, LATENT_SIZE, NUM_NODES, NUM_LAYERS)
decoder_network = FC_Decoder(BATCH_SIZE, SIZE, LATENT_SIZE, NUM_NODES, NUM_LAYERS)
vae = VAE(encoder_network, decoder_network)
loss = vae.loss_func(data)
optimizer = tf.train.AdamOptimizer().minimize(loss)

generated_images = vae.random_generate(BATCH_SIZE, LATENT_SIZE, IMAGE_SIZE, 1)
image_summary_op = tf.image_summary("vae_images", generated_images, max_images=BATCH_SIZE) 

# Create a local session to run the training.
start_time = time.time()
with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    tf.get_variable_scope().reuse_variables()
    print('Initialized!')
    # Loop through training steps.
    for epoch in range(NUM_EPOCHS):
        for step in range(int(TRAIN_SIZE / BATCH_SIZE)):
            offset = step * BATCH_SIZE
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            feed_dict = {data: batch_data}

            # Run the graph and fetch some of the nodes.
            _, loss_value = sess.run([optimizer, loss], feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print("Epoch {0}, Time: {1}, Loss: {2}".format(epoch, 1000.0 * elapsed_time / EVAL_FREQUENCY, loss_value))
    writer = tf.train.SummaryWriter("logs/mnist", sess.graph)
    images, summary = sess.run([generated_images, image_summary_op])
    writer.add_summary(summary)
    writer.flush()
