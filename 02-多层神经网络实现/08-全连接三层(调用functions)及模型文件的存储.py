import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import os
from functions import *

#--------------------------------步骤一：载入和处理数据--------------------------------------------
mnist = input_data.read_data_sets('MNIST_data_set/', one_hot=True)

#--------------------------------步骤二：建构模型--------------------------------------------
# 建构输入层
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


# -------------------------------------第三步：训练模型-------------------------------------------------
# 重新定义损失函数：结合了softmax的交叉熵损失函数，用于避免直接用交叉熵函数时log(0)的值为NaN造成的数据不稳定
loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=y))


# 设置训练参数
train_epochs = 40
batch_size = 50
total_batch = int(mnist.train.num_examples / batch_size)
display_step = 1
learning_rate = 0.005

# 设置文件保存的参数
save_step = 5
# 创建保存文件的目录（系统默认保存最近的5份模型数据文件，多余的会自动删除）
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# 选择优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 调用tf.train.Saver，生成一个文件的存储器节点
saver = tf.train.Saver()

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

    # 按轮次保存文件(每5轮)
    if (epoch+1) % save_step == 0:
        saver.save(sess, os.path.join(ckpt_dir, 'mnist_h256_64_32_model_{:06d}.ckpt'.format(epoch+1)))
        # os.path.join()函数将后面的文件（参数2）存储在前面的目录（参数1）之下
        print("minst_h256_64_32_model_{:06d}.ckpt 存储完毕".format(epoch+1))
# 存储最终的结果
saver.save(sess, os.path.join(ckpt_dir, 'mnist_h256_64_32_model.ckpt'))
print("mnist_h256_64_32_model.ckpt 模型存储完毕！")

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
# 调用函数察看输出错误的数据
print_predict_errors(labels=mnist.test.labels, prediction=prediction_result)

# 用可视化函数察看结果
plot_images_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result, 149, 25)
