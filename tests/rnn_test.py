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
rnn_size = 2
rnn_types = ['RNN','GRU','LSTM']
rnn_type = rnn_types[0]

input = []
for i in range(seq_length):
	input.append(tf.zeros([batch_size,params,num_features]))

if rnn_type == 'RNN':
	cell = rnn_cell.BasicRNNCell(rnn_size)
elif rnn_type == 'GRU':
	### Only works if rnn_size equals num_features
	cell = rnn_cell.GRUCell(rnn_size)
elif rnn_type == 'LSTM':
	### Only works if rnn_size equals num_features
	cell = rnn_cell.BasicLSTMCell(rnn_size)

if rnn_type == 'LSTM':
	initial_state = tf.zeros([batch_size,params,num_features])
	initial_state = [initial_state,initial_state]
else:
	initial_state = tf.zeros([batch_size,params,num_features])

output, state = rnn.rnn(cell, input, initial_state=initial_state, sequence_length=[seq_length for i in range(batch_size)])

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#sess.run([output],feed_dict={})