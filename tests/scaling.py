import tensorflow as tf
import numpy as np

batch_size = 2
grad_scaling_methods = ['scalar','full']
grad_scaling_method = grad_scaling_methods[1]
grad_scaling_factor = 0.1
p = 10

##### Test gradient preprocessing (scaling) #####

x_ = tf.placeholder(tf.float32, [2,4])
scale_ = tf.placeholder(tf.float32)
p_ = tf.placeholder(tf.float32)
grad_threshold = tf.exp(-p_)

if grad_scaling_method == 'scalar':
	y_ = x_*scale_
	
elif grad_scaling_method == 'full':
	# Operations are element-wise
	mask = tf.greater(x_,grad_threshold)
	mask = tf.to_float(mask) # Convert from boolean
	inv_mask = 1 - mask
	
	x1_cond1 = tf.log(tf.abs(x_))/p_
	x2_cond1 = tf.sign(x_)
	x1_cond2 = -tf.ones(tf.shape(x_))
	x2_cond2 = tf.exp(p_)*x_
	
	x1 = x1_cond1*mask + x1_cond2*inv_mask
	x2 = x2_cond1*mask + x2_cond2*inv_mask
	
	
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

x = np.array([[1.1,0.5,-8.6,4.0],[0.9,0.0003,-0.12,0.0000008]], dtype='float32')

y = sess.run([x1,x2], feed_dict={x_:x ,scale_:grad_scaling_factor, p_:p})#, scale_:grad_scaling_factor})

print y