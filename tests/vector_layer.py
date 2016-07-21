import tensorflow as tf
import numpy as np

batch_size = 5
features = [2,4,3] # For each layer
params = 10 # Not normally known during compilation

x = tf.placeholder(tf.float32, [batch_size,None,features[0]])

W1 = tf.Variable(tf.zeros([features[0],features[1]]))
W1 = tf.reshape(W1,(-1,features[0],features[1])) # Convert from rank 2 to rank 3
W1 = tf.tile(W1,(batch_size,1,1))
b1 = tf.Variable(tf.zeros([features[1]]))

h = tf.nn.relu(tf.batch_matmul(x,W1) + b1)
print h

W2 = tf.Variable(tf.zeros([features[1],features[2]]))
W2 = tf.reshape(W2,(-1,features[1],features[2])) # Convert from rank 2 to rank 3
W2 = tf.tile(W2,(batch_size,1,1))
b2 = tf.Variable(tf.zeros([features[2]]))

h = tf.nn.relu(tf.batch_matmul(h,W2) + b2)
print h

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

x_ = np.zeros((batch_size,params,features[0])) # [batch_size, num_parameters, num_features]
z = sess.run(h,feed_dict={x:x_})

"""
# batch_size = 2
# number of parameters = 3
# number of input features = 2
# hidden layer size = 4
# [3,2] -> [3,4]

x_ = np.array([[[1,2],[-3,-4],[5,6]],[[11,12],[-13,-14],[15,16]]],dtype='float32') # [2,3,2] # [batch_size, num_parameters, num_features]
W_ = np.array([[1,2,3,4],[5,6,7,8]],dtype='float32') # [2,4]
b_ = np.array([1,2,3,4],dtype='float32') # [4]

x = tf.placeholder(tf.float32, [2,None,2])
W = tf.Variable(tf.zeros([2,4]))
b = tf.Variable(tf.zeros([4]))

W2 = tf.reshape(W,(1,-1,4)) # Convert from rank 2 to rank 3
W2 = tf.tile(W2,(2,1,1)) # 2 is num_batches

h = tf.nn.relu(tf.batch_matmul(x,W2) + b)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
"""
#print h
#print sess.run(h,feed_dict={x:x_,W:W_,b:b_})
