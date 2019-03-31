import tensorflow as tf
import matplotlib.pyplot as plt

# norm = tf.random_normal([100])
norm1 = tf.random_normal([100,4])
with tf.Session() as sess:
    # norm_data = norm.eval(session=sess)
    norm1_data = norm1.eval(session=sess)
# print(norm_data)
print(norm1_data)

# plt.hist(norm_data)
plt.hist(norm1_data)

plt.show()
plt.hist(norm1_data[:,3])
plt.show()
