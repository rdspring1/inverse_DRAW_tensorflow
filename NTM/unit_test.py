import tensorflow as tf
import numpy as np
import util

def cosine_simularity(k, memory):
    return tf.matmul(util.normalize(k), util.normalize(memory))
memsize = 8
memdim = 5

M = [[0.00, 0.80, 0.20, 0.50, 1.00, 1.00, 0.25, 0.20],
     [0.50, 0.20, 0.20, 0.50, 0.20, 0.20, 0.50, 0.20],
     [0.20, 0.25, 0.00, 1.00, 0.50, 0.50, 0.25, 0.85],
     [0.20, 0.00, 0.00, 0.00, 1.00, 1.00, 0.20, 0.50],
     [0.50, 0.75, 0.50, 0.00, 1.00, 0.20, 0.20, 0.50]] 

k = [[1.00,0.20,0.25,0.00,1.00]]

np_mem = np.array(M)
np_k = np.array(k)  
print(np.dot(np_k, np_mem) / (np.linalg.norm(np_k) * np.linalg.norm(np_mem))) 

memory = tf.constant(M)
k_tf = tf.constant(k)
beta_t = tf.constant(25.0)

sess = tf.InteractiveSession()
init_op = tf.initialize_all_variables()
sess.run(init_op)

mem_norm = util.normalize(memory)
mem_k = util.normalize(k_tf)
cs = cosine_simularity(k_tf, memory) 
content_address = tf.nn.softmax(tf.mul(beta_t, cs))

gate = tf.constant(0.75)
w_prev = tf.constant([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
interpolate = tf.mul(gate, content_address) + tf.mul(1.0 - gate, w_prev)

batch_size = 1
shift_t = tf.constant([[0.0, 0.05, 1.0]])
shift_t = tf.expand_dims(shift_t, -1)
shift_t = tf.expand_dims(shift_t, -1)
shift_t = tf.transpose(shift_t, [1, 2, 3, 0])
shift_list = tf.split(3, batch_size, shift_t)

interpolate = tf.expand_dims(interpolate, -1)
interpolate = tf.expand_dims(interpolate, -1)
interpolate_list = tf.split(0, batch_size, interpolate)

conv_list = []
for w, shift in zip(interpolate_list, shift_list):
    conv_shift = tf.nn.conv2d(w, shift, [1,1,1,1], 'SAME')
    conv_list.append(tf.squeeze(conv_shift, [2,3]))
w_tilde = tf.concat(0, conv_list)

gamma_t = tf.constant(25.0)
new_weight = tf.pow(w_tilde + 1e-6, gamma_t)
norm_new_weight = tf.truediv(new_weight, tf.reduce_sum(new_weight))

result = sess.run([norm_new_weight])
for item in result:
    print(np.around(item, decimals=3))
