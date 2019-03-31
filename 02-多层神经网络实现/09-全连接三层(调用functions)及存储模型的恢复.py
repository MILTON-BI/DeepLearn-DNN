import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from functions import *

# 载入数据
mnist = input_data.read_data_sets('MNIST_data_set/', one_hot=True)

# ------------------------------------构建与原模型结构相同的模型----------------------------------------------
# 定义标签占位符
x = tf.placeholder(tf.float32, shape=(None, 784), name="X")
y = tf.placeholder(tf.float32, shape=(None, 10), name="Y")

# 构建隐藏层
H1_NN = 256
H2_NN = 64
H3_NN = 32

# 输入层到第一隐藏层
h1 = fcn_layer(inputs=x, input_dim=784, output_dim=H1_NN, activation=tf.nn.relu)
# 第一隐藏层到第二隐藏层
h2 = fcn_layer(inputs=h1, input_dim=H1_NN, output_dim=H2_NN, activation=tf.nn.relu)
# 第二隐藏层到第三隐藏层
h3 = fcn_layer(inputs=h2, input_dim=H2_NN, output_dim=H3_NN, activation=tf.nn.relu)
# 第三隐藏层到输出
forward = fcn_layer(inputs=h3, input_dim=H3_NN, output_dim=10, activation=None)
pred = tf.nn.softmax(forward)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# -------------------------------------还原模型---------------------------------------------------
# 必须设置读取文件的目录（默认读取最新的一份模型的存盘文件）
ckpt_dir = "./ckpt_dir/"

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(ckpt_dir)  # 获取存盘目录下模型文件的状态：会去找最新的文件

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)  # 从已经保存的模型中读取参数
    print("Restore model from" + ckpt.model_checkpoint_path)

# ------------------------------------还原模型后的应用----------------------------------------------
# 在测试集上评估模型准确率
accu_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("测试准确率：", accu_test)   # 准确率与原模型一样是0.9792

# 用还原的模型进行预测
prediction_result = sess.run(tf.argmax(pred, 1), feed_dict={x: mnist.test.images})
print_predict_errors(labels=mnist.test.labels, prediction=prediction_result)

# 用可视化函数察看结果
plot_images_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result, 149, 25)