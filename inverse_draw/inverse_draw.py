"""

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python3 inverse_draw.py --data_path=simple-examples/data/

"""
import time
import numpy as np
import tensorflow as tf

from controller import RNN_Controller
from heads import ReadHead
from encoder import LSTM_Encoder
from decoder import NTM_Decoder
from vae import NTM_VAE

import reader

flags = tf.flags
logging = tf.logging
flags.DEFINE_string("data_path", None, "data_path")
FLAGS = flags.FLAGS

# embedding
VOCAB_SIZE = 10000
EMBEDDING_SIZE = 200
NUM_STEPS = 20

# encoder
ENCODER_LSTM_LAYERS = 2
ENCODER_LSTM_SIZE = 200

# memory
MEM_SIZE = 128
MEM_DIM = 50

# attention
NUM_SHIFTS = 3

# controller
DECODER_LAYERS = 2
DECODER_SIZE = 200
WRITE_HEADS = 1
READ_HEADS = 1

# training
BATCH = 1
NUM_EPOCHS = 14
PRINT = 500

input_data = tf.placeholder(tf.int32, [BATCH, NUM_STEPS])
 
with tf.name_scope('ntm_vae_param') as scope:
    controller = RNN_Controller("rnn_ctrl", MEM_DIM, DECODER_SIZE, DECODER_LAYERS, BATCH)

    read_heads = []
    for idx in range(READ_HEADS):
        read_heads.append(ReadHead("read{0}".format(idx), controller, NUM_SHIFTS, MEM_DIM, MEM_SIZE, DECODER_SIZE, BATCH))

    encoder_network = LSTM_Encoder(BATCH, ENCODER_LSTM_SIZE, ENCODER_LSTM_LAYERS, MEM_DIM)
    decoder_network = NTM_Decoder(controller, read_heads, DECODER_SIZE, EMBEDDING_SIZE)
    vae = NTM_VAE(encoder_network, decoder_network, MEM_DIM, MEM_SIZE, BATCH, DECODER_SIZE)

with tf.name_scope('ntm_vae_op') as scope:
    with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, EMBEDDING_SIZE], dtype=tf.float32)
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    loss = vae.loss_func(inputs, NUM_STEPS)
    optimizer = tf.train.AdamOptimizer().minimize(loss)

train_data, valid_data, test_data, vocabulary = reader.ptb_raw_data(FLAGS.data_path)

start_time = time.time()
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    tf.get_variable_scope().reuse_variables()
    print('Initialized!')

    for epoch in range(NUM_EPOCHS):
        for step, x in enumerate(reader.ptb_iterator(train_data, BATCH, NUM_STEPS)):
            _, loss_value = sess.run([optimizer, loss], {input_data: x})
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print("Epoch {0}, Time: {1}, Loss: {2}".format(epoch, 1000.0 * elapsed_time / EVAL_FREQUENCY, loss_value))
    writer = tf.train.SummaryWriter("logs/mnist", sess.graph)
    images, summary = sess.run([generated_images, image_summary_op])
    writer.add_summary(summary)
    writer.flush()
