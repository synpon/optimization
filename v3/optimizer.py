from __future__ import division

import tensorflow as tf
import numpy as np

import rnn
import rnn_cell
from nn_utils import weight_matrix, bias_vector, fc_layer, fc_layer3, inv_scale_grads
from constants import rnn_size, num_rnn_layers, k, m, rnn_type, grad_scaling_method, \
		episode_length, loss_noise, osc_control
import snf
from nn_utils import tf_print


class Optimizer(object):

	def __init__(self, seq_length,scope_name):
		# Input
		self.points = tf.placeholder(tf.float32, [seq_length,m,1], 'points') # Used to calculate loss only
		self.snf_losses = tf.placeholder(tf.float32, [seq_length], 'snf_losses')
		self.input_grads = tf.placeholder(tf.float32, [seq_length,None,1], 'input_grads')
		self.counters = tf.placeholder(tf.float32, [seq_length], name='counters')
		self.variances = tf.placeholder(tf.float32, [k,1], 'variances')
		self.weights = tf.placeholder(tf.float32, [k,1], 'weights')
		self.hyperplanes = tf.placeholder(tf.float32, [m,m,k], 'hyperplanes') # Points which define the hyperplanes
			
		if rnn_type == 'lstm':
			self.initial_rnn_state = tf.placeholder_with_default(input=tf.zeros([m, 2*num_rnn_layers*rnn_size]), shape=[None, 2*num_rnn_layers*rnn_size])
		else:
			# initial_rnn_state is given during evaluation but not during training
			# each dimension has an independent hidden state, required in order to simulate Adam, RMSProp etc.
			self.initial_rnn_state = tf.placeholder_with_default(input=tf.zeros([m, num_rnn_layers*rnn_size]), shape=[None, num_rnn_layers*rnn_size])
			
		n_dims = tf.shape(self.input_grads)[1]
		
		points = tf.split(0, seq_length, self.points)
		snf_losses = tf.split(0, seq_length, self.snf_losses)
		counters = tf.split(0, seq_length, self.counters)

		# The scope allows these variables to be excluded from being reinitialized during the comparison phase
		with tf.variable_scope("a3c"): ### rename scope
			if rnn_type == 'rnn':
				cell = rnn_cell.BasicRNNCell(rnn_size)
			elif rnn_type == 'gru':
				cell = rnn_cell.GRUCell(rnn_size)
			elif rnn_type == 'lstm':
				cell = rnn_cell.LSTMCell(rnn_size)#, cell_clip=5.0)
				
			self.cell = rnn_cell.MultiRNNCell([cell] * num_rnn_layers)
			
			outputs, rnn_state = rnn.dynamic_rnn(self.cell,
									self.input_grads,
									initial_state = self.initial_rnn_state,
									time_major = True)
			
			# This is not used in training. Only in evaluation when steps are processed one by one.
			self.rnn_state_output = rnn_state
			
			self.total_loss = 0
			outputs = tf.split(0, seq_length, outputs)
			
			snf_losses_output = []
			points_output = []
			grads_output = []
			updates = []
			
			for point,snf_loss,output,counter in zip(points,snf_losses,outputs,counters):
				output = tf.reshape(output,tf.pack([n_dims,rnn_size]))
			
				update = fc_layer(output, num_in=rnn_size, num_out=1, activation_fn=None, bias=False)
				update = tf.reshape(update, tf.pack([n_dims,1]))
				self.update = inv_scale_grads(update) ### Effect of this during comparison (were grads scaled to begin with?)
				updates.append(self.update)
				
				new_point = self.update + tf.squeeze(point, squeeze_dims=[0])
				
				points_output.append(new_point)	
				
				new_snf_loss = snf.calc_snf_loss_tf(new_point, self.hyperplanes, self.variances, self.weights)
				
				# Add loss noise - reduce__mean is only to flatten
				new_snf_loss += tf.reduce_mean(tf.abs(new_snf_loss)*loss_noise*tf.random_uniform([1], minval=-1.0, maxval=1.0))
				snf_losses_output.append(new_snf_loss) ### without noise?
				
				g = snf.calc_grads_tf(new_snf_loss, new_point)
				grads_output.append(g)
				
				# Improvement: 2 - 3 = -1 (small loss)
				loss = new_snf_loss - snf_loss				
				self.total_loss += loss

			self.total_loss /= seq_length
			
			# Asymmetric loss function
			self.total_loss = tf.maximum(self.total_loss,20*self.total_loss)
			
			osc_cost = 0
			for i in range(len(updates)-1):
				u1 = updates[i]
				u2 = updates[i+1]
				u_prime = u1 + u2
				
				u1_norm = tf.sqrt(tf.reduce_sum(tf.square(u1)))
				u2_norm = tf.sqrt(tf.reduce_sum(tf.square(u2)))
				u_prime_norm = tf.sqrt(tf.reduce_sum(tf.square(u_prime)))
				
				osc_cost -= osc_control*(u_prime_norm - u1_norm - u2_norm)
				
			self.total_loss += osc_cost/(seq_length*m)
				
			#===# SNF outputs #===#
			# Used when filling the replay memory during training
			# Indexing is admissible here as these 3 variables only need to be returned when seq_length = 1
			self.snf_losses_output = snf_losses_output[0]
			self.points_output = points_output[0]
			self.grads_output = grads_output[0]
			
			#===# Model training #===#
			opt = tf.train.RMSPropOptimizer(0.01,momentum=0.5)
			vars = [i for i in tf.trainable_variables() if scope_name in i.name] ### could be unreliable in the future
			
			gvs = opt.compute_gradients(self.total_loss, vars)
			self.gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for (grad, var) in gvs]
			#gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]

			self.grads_input = [(tf.placeholder(tf.float32, shape=v.get_shape()), v) for (g,v) in gvs]
			self.train_step = opt.apply_gradients(self.grads_input)

			
	# Update the parameters of another network (eg an MLP)
	def update_params(self, vars, update):
		total = 0
		ret = []

		for i,v in enumerate(vars):
			size = np.prod(list(v.get_shape()))
			size = tf.to_int32(size)
			var_grads = tf.slice(update,begin=[total,0],size=[size,-1])
			var_grads = tf.reshape(var_grads,v.get_shape())
			ret.append(v.assign_add(var_grads))
			size += total		
		return tf.group(*ret)
		
