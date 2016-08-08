import numpy as np

##### Logging constants #####
summaries_dir = '/tmp/logs'
save_path = 'models/model.ckpt'
log_file = 'tmp/a3c_log'
summary_freq = 100

##### A3C constants #####
num_threads = 8
local_t_max = 5 # repeat step size
num_steps = min(5,local_t_max) # Number of steps to go back for truncated backprop.
rmsp_alpha = 0.99 # decay parameter for RMSProp
rmsp_epsilon = 0.1 # epsilon parameter for RMSProp
initial_alpha_low = 1e-4    # log_uniform low limit for learning rate
initial_alpha_high = 1e-2   # log_uniform high limit for learning rate
initial_alpha_log_rate = 0.4226 # log_uniform interpolate rate for learning rate (around 7 * 10^-4) ### description is incorrect
entropy_beta = 0.0001 # entropy regularization constant 10^-4
max_time_steps = 50000
grad_norm_clip = 40.0 # gradient norm clipping
discount_rate = 0.95

##### Opt net constants #####
use_rnn = False # Uses a feed-forward network if false
rnn_types = ['rnn','gru','lstm']
rnn_type = rnn_types[1]
rnn_size = 4 ### Causes W1 to be the wrong size?
num_rnn_layers = 1

##### GMM constants #####
num_gaussians = 10 # 50 # Number of Gaussians
m = 2 # 10 # Number of dimensions
cov_range = [0,2] # 16 ### Only the upper bound is used
cov_range[1] *= np.sqrt(m)
weight_gaussians = False