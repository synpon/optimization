from __future__ import division

import numpy as np

#===# Logging constants #===#
summaries_dir = '/tmp/logs'
save_path = 'models/model.ckpt'
summary_freq = 50

#===# Opt net constants #===#
rnn_types = ['rnn','gru']
rnn_type = rnn_types[1]
rnn_size = 8
num_rnn_layers = 2

#===# SNF constants #===#
k = 10 # Number of hyperplanes
m = 30 # Number of dimensions
var_size = 0.2

#===# Training constants #===#
num_iterations = 1000
batch_size = 32 ### use
seq_length = 10
num_SNFs = 100
#num_samples_for_gamma = 100
episode_length = 1000
replay_mem_start_size = 1000
replay_memory_max_size = 1000000

grad_scaling_methods = ['none','full']
grad_scaling_method = grad_scaling_methods[1]

# Random noise is computed each time the point is processed while training the opt net
grad_noise = 0.5 # Determines the size of the standard deviation. The mean is zero.