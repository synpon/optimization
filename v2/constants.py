from __future__ import division

import numpy as np

#===# Logging constants #===#
summaries_dir = '/tmp/logs'
save_path = 'models/model.ckpt'

log_file = 'tmp/a3c_log'
summary_freq = 500

#===# A3C constants #===#
num_threads = 8
local_t_max = 5 # repeat step size
num_steps = 3 # Number of steps to go back for truncated backprop. ### unused
rmsp_alpha = 0.99 # decay parameter for RMSProp
rmsp_epsilon = 0.1 # epsilon parameter for RMSProp
lr_high = 0.001 # upper limit for learning rate for RMSProp
lr_low = 0.001 # lower limit for learning rate for RMSProp
entropy_beta = 1e-2 # entropy regularization constant 0.0001
max_time_steps = 4e7
grad_norm_clip = 40.0 # gradient norm clipping for RMSProp
discount_rate = 0.99

#===# Opt net constants #===#
use_rnn = False # Uses a feed-forward network if false
rnn_types = ['rnn','gru','lstm']
rnn_type = rnn_types[1]
rnn_size = 2
num_rnn_layers = 1

#===# GMM constants #===#
m = 10 # Number of dimensions
var_size = 0.2

grad_scaling_methods = ['none','scalar','full']
grad_scaling_method = grad_scaling_methods[0]
grad_scaling_factor = 0.1
p = 10.0

termination_prob = 0.003 ### check

# Random noise is computed each time the point is processed while training the opt net
grad_noise = 0.5 # Determines the size of the standard deviation. The mean is zero.