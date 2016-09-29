from __future__ import division

import numpy as np

#===# Logging constants #===#
summaries_dir = '/tmp/logs'
save_path = 'models/model.ckpt'
summary_freq = 1000

#===# Opt net constants #===#
rnn_types = ['rnn','gru','lstm']
rnn_type = rnn_types[1]
rnn_size = 4
num_rnn_layers = 1

#===# SNF constants #===#
k = 10 # Number of hyperplanes
m = 30 # Number of dimensions
var_size = 0.2

#===# Training constants #===#
batch_size = 32 ### use
seq_length = 3 # 20
num_iterations = 10000
num_SNFs = 1000
replay_mem_start_size = 5000 # DQN: 50000
replay_memory_max_size = 100000 # DQN: 1000000
episode_length = 100
discount_rate = 0.9995
net_sync_freq = 100 # DQN: 10000

grad_scaling_methods = ['none','full']
grad_scaling_method = grad_scaling_methods[0]

# Random noise is added to the loss of the SNF during training of the opt net.
loss_noise = 0#0.75