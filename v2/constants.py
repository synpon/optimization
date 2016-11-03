from __future__ import division

import numpy as np

#===# Logging constants #===#
summaries_dir = '/tmp/logs'
save_path = 'models/model.ckpt'

summary_freq = 1000
test_freq = 50000 ### Add periodic testing with MLP

#===# A3C constants #===#
num_threads = 8
local_t_max = 5 # repeat step size
entropy_beta = 0.001 # entropy regularization 0.0001
max_time_steps = 1e9
discount_rate = 0.99
### Add discounting based on the local time step?

#===# RMSProp constants #===#
rmsp_alpha = 0.99 #0.99# decay parameter for the historical gradient
rmsp_epsilon = 1e-10 #0.1
### Learning rate should be a function of the loss/reward?
lr_high = 0.001 # upper limit for learning rate
lr_low = 0.001 # lower limit for learning rate
grad_norm_clip = 5.0
rmsp_momentum = 0.9

#===# Opt net constants #===#
rnn_types = ['rnn','gru']
rnn_type = rnn_types[1]
val_rnn_type = rnn_types[1]
rnn_size = 8
num_rnn_layers = 1
val_rnn_size = 4

#===# SNF constants #===#
k = 10 # Number of hyperplanes
m = 30 # Number of dimensions
var_size = 0.2

grad_scaling_methods = ['none','full']
grad_scaling_method = grad_scaling_methods[0]

### Decrease over time?
termination_prob = 0.01 #0.0001 # Can be used to control the trade-off between speed and the final loss, as the learning rate does.
