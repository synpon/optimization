import tensorflow as tf
import numpy as np

batch_size = 2
grad_scaling_methods = ['scalar','full']
grad_scaling_method = grad_scaling_methods[1]
grad_scaling_factor = 0.1
p = 10.0

# Dimension 2 used instead of 1 in the optimizer version

def scale_grads(input):
	if grad_scaling_method == 'scalar':
		input = input*tf.constant(grad_scaling_factor)	
		
	elif grad_scaling_method == 'full':	
		p_ = tf.constant(p)
		grad_threshold = tf.exp(-p_)
	
		# Operations are element-wise
		mask = tf.greater(tf.abs(input),grad_threshold)
		mask = tf.to_float(mask) # Convert from boolean
		inv_mask = 1 - mask
		
		x1_cond1 = tf.log(tf.abs(input))/p_
		x2_cond1 = tf.sign(input)
		x1_cond2 = -tf.ones(tf.shape(input))
		x2_cond2 = tf.exp(p_)*input
		
		x1 = x1_cond1*mask + x1_cond2*inv_mask
		x2 = x2_cond1*mask + x2_cond2*inv_mask
		
		input = tf.concat(1,[x1,x2])
		
	return input
	
	
def inv_scale_grads(input):
	if grad_scaling_method == 'scalar':
		input = input/tf.constant(grad_scaling_factor)	
		
	elif grad_scaling_method == 'full':	
		p_ = tf.constant(p)
	
		# Operations are element-wise
		a,b = tf.split(1,2,input)
		mask = tf.equal(tf.abs(b),1.0)
		mask = tf.to_float(mask) # Convert from boolean
		inv_mask = 1 - mask
		
		x_cond1 = tf.sign(b)*tf.exp(a*p_)
		x_cond2 = b/tf.exp(p_)
		
		x = x_cond1*mask + x_cond2*inv_mask
		
	return x

	
x_ = tf.placeholder(tf.float32, [2,4])
z = scale_grads(x_)

z2 = inv_scale_grads(z)
	
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

x = np.array([	[1.1,	0.5,	-8.6,	4.0],
				[0.9,	0.0003,	-0.12,	0.0000008]], dtype='float32')

y = sess.run(z, feed_dict={x_:x})
print y

y = sess.run(z2, feed_dict={x_:x})
print y