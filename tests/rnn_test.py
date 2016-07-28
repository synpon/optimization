import sys
sys.path.append('/lstm_vol2/attalus/')
import tensorflow as tf
import numpy as np
import rnn
import rnn_cell

batch_size = 1
num_features = 2
params = 10 # Not normally known during compilation
seq_length = 5
rnn_size = 4

input = []
for i in range(seq_length):
	input.append(tf.zeros([batch_size,params,num_features]))

cell = rnn_cell.BasicRNNCell(rnn_size) 
#cell = rnn_cell.GRUCell(rnn_size)
#cell = rnn_cell.BasicLSTMCell(rnn_size)

initial_state = tf.zeros([batch_size,params,num_features])

output, state = rnn.rnn(cell, input, initial_state=initial_state, sequence_length=seq_length)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#sess.run([output],feed_dict={})