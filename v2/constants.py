from __future__ import division

import numpy as np

#===# Logging constants #===#
summaries_dir = '/tmp/logs'
save_path = 'models/model.ckpt'

log_file = 'tmp/a3c_log'
summary_freq = 100

#===# A3C constants #===#
num_threads = 8
local_t_max = 5 # repeat step size
entropy_beta = 0.0001 # entropy regularization constant 0.0001
max_time_steps = 1e7
discount_rate = 0.99

#===# RMSProp constants #===#
rmsp_alpha = 0.9 #0.99# decay parameter for the historical gradient
rmsp_epsilon = 1e-10 #0.1
### Learning rate should be a function of the loss/reward?
lr_high = 0.0001 # upper limit for learning rate
lr_low = 0.0001 # lower limit for learning rate
grad_norm_clip = 40.0
rmsp_momentum = 0.0#0.9

#===# Opt net constants #===#
use_rnn = True # Uses a feed-forward network if false
rnn_types = ['rnn','gru']
rnn_type = rnn_types[1]
rnn_size = 4
num_rnn_layers = 1

#===# SNF constants #===#
k = 10 # Number of hyperplanes
m = 30 # Number of dimensions
var_size = 0.2

grad_scaling_methods = ['none','scalar','full']
grad_scaling_method = grad_scaling_methods[0]
grad_scaling_factor = 0.1
p = 10.0

### Decrease over time
termination_prob = 0.001 # Can be used to control the trade-off between speed and the final loss, as the learning rate does.

# Random noise is computed each time the point is processed while training the opt net
grad_noise = 0.5 # Determines the size of the standard deviation. The mean is zero.