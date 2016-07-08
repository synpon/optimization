import tensorflow as tf
import numpy as np

batch_size = 32

# Test this compiles seperately using numpy arrays as input

# Standard version
#x = tf.placeholder(tf.float32, [32, 784])
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
#h = tf.nn.relu(tf.matmul(x,W) + b) # [None, 10]

# Vector features version (2xn)
# 10 is the size of the hidden layer
#x = tf.placeholder(tf.float32, [None, None, 2])
#W = tf.Variable(tf.zeros([2, 10]))
#b = tf.Variable(tf.zeros([10]))
#h = # [None, None, 10]

# Tensor product [nx2,10]
# Can do the tensor product using reshape (into column vectors?) and matmul
#tf.batch_matmul() # matmul over each training example in the batch eg [100,2,5] x [100,5,2] -> [100,2,2]

# Split in to two halves [n,10],[n,10]

# Sum the two halves [n,10]

#init = tf.initialize_all_variables()

#sess = tf.Session()
#sess.run(init)

#x_ = np.ones((batch_size,784))

#z = sess.run([h], feed_dict={x:x_})

##### numpy version #####
# batch_size = 1
# feature length = 3
# [3,2] -> [3,4]
x_ = np.array([[[1,2],[-3,-4],[5,6]],[[11,12],[-13,-14],[15,16]]],dtype='float32') # [2,3,2] # [batch_size, num_parameters, num_features]
W_ = np.array([[1,2,3,4],[5,6,7,8]],dtype='float32') # [2,4]
b_ = np.array([1,2,3,4],dtype='float32') # [4]

x = tf.placeholder(tf.float32, [2,None,2])
W = tf.Variable(tf.zeros([2,4]))
b = tf.Variable(tf.zeros([4]))
#x = tf.reshape(x,[6,2])
W2 = tf.reshape(W,(1,2,4))
W2 = tf.tile(W2,(2,1,1))
print x
print W
z = tf.batch_matmul(x,W2)
z = z + b
h = tf.nn.relu(z)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print sess.run(h,feed_dict={x:x_,W:W_,b:b_})



