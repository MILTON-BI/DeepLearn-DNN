import tensorflow as tf
import numpy as np

x = np.array([-3.1, 1.8, 9.7, -2.5])
pred = tf.nn.softmax(x)
with tf.Session() as sess:
    v = sess.run(pred)
    print(v)