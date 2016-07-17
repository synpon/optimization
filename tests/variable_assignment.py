import tensorflow as tf
import numpy as np

x = tf.Variable(tf.constant(1.0))
y = tf.constant(2.0)

# Returned value is only needed for monitoring
z = x.assign_add(y)

#vars = [x]
vars = tf.trainable_variables()
outputs = []
for v in vars:
	outputs.append(v.assign_add(y))

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print sess.run(z)

print sess.run(outputs)