import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from functions import *

#--------------------------------步骤一：载入和处理数据--------------------------------------------
mnist = input_data.read_data_sets('MNIST_data_set/', one_hot=True)

#--------------------------------步骤二：建构模型--------------------------------------------
# 建构输入层
# 定义标签占位符
x = tf.placeholder(tf.float32, shape=(None, 784), name="X")
y = tf.placeholder(tf.float32, shape=(None, 10), name="Y")

# 构建神经网络的输入和输出
# 构建隐藏层
h1 = fcn_layer(inputs=x, input_dim=784, output_dim=256, activation=tf.nn.relu)
# 构建输出层
forward = fcn_layer(inputs=h1, input_dim=256, output_dim=10, activation=None)
pred = tf.nn.softmax(forward)

# -------------------------------------第三步：训练模型-------------------------------------------------
# 定义损失函数：结合了softmax的交叉熵损失函数，用于避免log(0)的值为NaN造成的数据不稳定
loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))

# 设置训练参数
train_epochs = 40
batch_size = 50
total_batch = int(mnist.train.num_examples / batch_size)
display_step = 1
learning_rate = 0.01

# 选择优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 记录开始训练时间
from time import time
startTime = time()

# 启动会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 开始训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size) # 读取批次数据
        sess.run(optimizer, feed_dict={x: xs, y: ys})  # 执行批次训练

    loss, acc = sess.run([loss_func, accuracy], feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

    if (epoch+1) % display_step == 0:
        print("训练轮次：", "%02d" % (epoch+1), "  损失值：", "{:.9f}".format(loss), "  准确率：", "{:.4f}".format(acc))

# 显示运行总时间
duration = time() - startTime
print("训练结束！总耗时：", "{:.2f}s".format(duration))

# ------------------------------------第四步：用测试集评估模型------------------------------------------------
# 在测试集上评估模型准确率
accu_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print("测试准确率：", accu_test)

# ------------------------------------第五步：用模型进行预测------------------------------------------------
# 预测结果是one-hot编码格式，需要转换成0-9的数字
prediction_result = sess.run(tf.argmax(pred, 1), feed_dict={x: mnist.test.images})

# 通过输出错误函数察看结果
print_predict_errors(labels=mnist.test.labels, prediction=prediction_result)

# 用可视化函数察看结果
plot_images_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result, 149, 25)
