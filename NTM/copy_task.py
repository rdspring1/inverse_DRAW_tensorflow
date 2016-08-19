import tensorflow as tf
import numpy as np
import numpy.random
import random

from ntm import NTM
from controller import FC_Controller, RNN_Controller
from heads import WriteHead, ReadHead

# memory
MEM_SIZE = 128
MEM_DIM = 20

# attention
NUM_SHIFTS = 3

# controller
WRITE_HEADS = 1
READ_HEADS = 1
LSTM_LAYERS = 1
NUM_NODES = 100

# COPY
BATCH = 1
SIZE = 8
MIN_SEQUENCE_LEN = 1
MAX_SEQUENCE_LEN = 5
INPUT_LEN = MAX_SEQUENCE_LEN + 1
OUTPUT_LEN = MAX_SEQUENCE_LEN

# training
TRAIN_SIZE = 100000
PRINT = 500
SEED = 66478

def copy_generator(num_examples, size, min_seq_len, max_seq_len, SEED=None):
    np.random.seed(SEED)
    np_end_state = np.zeros([1, size])
    np_end_state[:, size-1] = 1

    idx = 0
    while idx < num_examples:
        # Generate Random Sequence
        seq_len = random.randint(min_seq_len, max_seq_len)
        mask = np.random.uniform(0, 1, [seq_len, size]) > 0.5
        seq = mask.astype(float)
        seq[:, size-1:] = 0

        # Input random sequence 
        inputs_ = np.split(seq, seq_len, 0)

        # Sequence End Marker
        inputs_.append(np_end_state)

        # Add Zero placeholders 
        for idx in range(max_seq_len - seq_len):
            inputs_.append(np.zeros([1, size]))

        output_ = np.zeros([max_seq_len, size])
        output_[:seq_len, :] = seq
        yield inputs_, output_ 

zeros = tf.get_variable("zeros", [BATCH, SIZE], tf.float32, tf.constant_initializer(), trainable=False)

def model(l_ntm, ntm_state, memory, inputs, batch_size, num_nodes, size):
    init_ops = []
    init_ops.append(ntm_state.assign(tf.fill([batch_size, num_nodes], 1.0)))
    init_ops.append(memory.assign(tf.fill([MEM_DIM, MEM_SIZE], 1e-6)))

    for idx in range(WRITE_HEADS):
        init_ops.append(l_ntm.write_heads[idx].clear_op)

    for idx in range(READ_HEADS):
        init_ops.append(l_ntm.read_heads[idx].clear_op)

    with tf.control_dependencies(init_ops):
        for idx in range(MAX_SEQUENCE_LEN+1):
            ntm_state, memory = l_ntm.step(inputs[idx], ntm_state, memory)

        states = []
        for idx in range(MAX_SEQUENCE_LEN):
            ntm_state, memory = l_ntm.step(zeros, ntm_state, memory)
            states.append(ntm_state)
        outputs = l_ntm.execute(states)
        return outputs, states, memory

def main(argv=None):  # pylint: disable=unused-argument
    train_data = copy_generator(TRAIN_SIZE, SIZE, MIN_SEQUENCE_LEN, MAX_SEQUENCE_LEN, SEED)

    with tf.name_scope('ntm_param') as scope:
        #controller = FC_Controller("fc_ctrl", READ_HEADS, MEM_DIM, SIZE, NUM_NODES)
        controller = RNN_Controller("rnn_ctrl", MEM_DIM, NUM_NODES, LSTM_LAYERS, BATCH)

        write_heads = []
        for idx in range(WRITE_HEADS):
            write_heads.append(WriteHead("write{0}".format(idx), controller, NUM_SHIFTS, MEM_DIM, MEM_SIZE, NUM_NODES, BATCH))

        read_heads = []
        for idx in range(READ_HEADS):
            read_heads.append(ReadHead("read{0}".format(idx), controller, NUM_SHIFTS, MEM_DIM, MEM_SIZE, NUM_NODES, BATCH))

        l_ntm = NTM("NTM", controller, read_heads, write_heads, NUM_NODES, SIZE)
        ntm_state = tf.get_variable("ntm_state", [BATCH, NUM_NODES], tf.float32, tf.constant_initializer(1.0), trainable=False)
        memory = tf.get_variable("memory", [MEM_DIM, MEM_SIZE], tf.float32, tf.constant_initializer(1e-6), trainable=False)

    with tf.name_scope('ntm_op') as scope:
        inputs = [tf.placeholder(tf.float32, shape=(BATCH, SIZE), name="input_{0}".format(idx)) for idx in range(INPUT_LEN)]
        target = tf.placeholder(tf.float32, shape=(OUTPUT_LEN, SIZE), name="target")

        logits, states, ntm_memory = model(l_ntm, ntm_state, memory, inputs, BATCH, NUM_NODES, SIZE)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, target))
        optimizer = tf.train.AdamOptimizer(1e-4)
        grad_vars = optimizer.compute_gradients(loss)
        new_grad_vars = [(tf.clip_by_value(grad, -10.0, 10.0), var) if grad is not None else (grad, var) for grad, var in grad_vars]
        train_op = optimizer.apply_gradients(new_grad_vars)
        prediction = tf.sigmoid(logits)

    idx = 0
    np.set_printoptions(precision=2, linewidth=200, threshold=5000, suppress=True)
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        writer = tf.train.SummaryWriter("logs/ntm", sess.graph)
        writer.flush()

        for input_, output_ in train_data:
            feed_dict = {key:value for value, key in zip(input_, inputs)}
            feed_dict.update({target:output_})

            _, np_memory, loss_value, output = sess.run([train_op, ntm_memory, loss, prediction], feed_dict=feed_dict)
            print("iteration: {0} Loss: {1}".format(idx, loss_value))
            if idx % PRINT == 0:
                print(np.round(np.transpose(np_memory), decimals=2))
                print(np.round(output, decimals=2))
                print(output_)
            idx += 1
        sys.stdout.flush()

if __name__ == '__main__':
  tf.app.run()
