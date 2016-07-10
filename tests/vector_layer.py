import tensorflow as tf
import numpy as np

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

print sess.run(h,feed_dict={x:x_,W:W_,b:b_})
