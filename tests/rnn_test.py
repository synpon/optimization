import sys
sys.path.append('/lstm_vol2/attalus/')
import tensorflow as tf
import numpy as np
import rnn
import rnn_cell

batch_size = 1
params = 10 # Not normally known during compilation
seq_length = 5
rnn_size = 2
rnn_types = ['RNN','GRU','LSTM']
rnn_type = rnn_types[0]

input = []
for i in range(seq_length):
	input.append(tf.zeros([batch_size,params,rnn_size]))

if rnn_type == 'RNN':
	cell = rnn_cell.BasicRNNCell(rnn_size)
elif rnn_type == 'GRU':
	cell = rnn_cell.GRUCell(rnn_size)
elif rnn_type == 'LSTM':
	cell = rnn_cell.BasicLSTMCell(rnn_size)

if rnn_type == 'LSTM':
	initial_state = tf.zeros([batch_size,params,rnn_size])
	initial_state = [initial_state,initial_state]
else:
	initial_state = tf.zeros([batch_size,params,rnn_size])

output, state = rnn.rnn(cell, input, initial_state=initial_state, sequence_length=[seq_length for i in range(batch_size)])
print output
"""
Output:
[<tf.Tensor 'RNN/cond/Merge:0' shape=(1, 10, 2) dtype=float32>, 
<tf.Tensor 'RNN/cond_1/Merge:0' shape=(1, 10, 2) dtype=float32>, 
<tf.Tensor 'RNN/cond_2/Merge:0' shape=(1, 10, 2) dtype=float32>, 
<tf.Tensor 'RNN/cond_3/Merge:0' shape=(1, 10, 2) dtype=float32>, 
<tf.Tensor 'RNN/cond_4/Merge:0' shape=(1, 10, 2) dtype=float32>]

State:
Tensor("RNN/cond_4/Merge_1:0", shape=(1, 10, 2), dtype=float32)
"""
print state
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#sess.run([output],feed_dict={})