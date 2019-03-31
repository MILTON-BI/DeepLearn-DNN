import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

#--------------------------------步骤一：载入和处理数据--------------------------------------------
mnist = input_data.read_data_sets('MNIST_data_set/', one_hot=True)

#--------------------------------步骤二：建构模型--------------------------------------------
# 建构输入层
# 定义标签占位符
x = tf.placeholder(tf.float32, shape=(None, 784), name="X")
y = tf.placeholder(tf.float32, shape=(None, 10), name="Y")

# 构建隐藏层（神经元数量=256）
H1_NN = 256
W1 = tf.Variable(tf.random_normal([784, H1_NN]))
b1 = tf.Variable(tf.zeros([H1_NN]))

Y1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# 构建输出层
W2 = tf.Variable(tf.random_normal([H1_NN, 10]))
b2 = tf.Variable(tf.zeros([10]))

forward = tf.matmul(Y1, W2) + b2
pred = tf.nn.softmax(forward)


# -------------------------------------第三步：训练模型-------------------------------------------------
# 定义损失函数：交叉熵，可能造成log(0)值为NaN造成的数据不稳定
# loss_func = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# 重新定义损失函数：结合了softmax的交叉熵损失函数，用于避免log(0)的值为NaN造成的数据不稳定
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
# 察看预测结果的前10项
# print(prediction_result[:10])

# 找出预测错误的样本
# compare_list = prediction_result == np.argmax(mnist.test.labels, 1)
# print(compare_list)

# error_list = [i for i in range(len(compare_list)) if compare_list[i] == False]
# print(error_list)
# print("预测错误的样本数：", len(error_list))

# 通过输出错误函数察看结果
def print_predict_errors(labels, prediction):  # 参数为标签列表和预测值列表
    count = 0
    compare_list = (prediction == np.argmax(labels, 1))
    error_list = [i for i in range(len(compare_list)) if compare_list[i] == False]
    for x in error_list:
        print("index=" + str(x) +
              " 标签值=", np.argmax(labels[x]),
              "预测值=", prediction[x])
        count += 1
    print("总计：" + str(count))
print_predict_errors(labels=mnist.test.labels, prediction=prediction_result)

# 用可视化函数察看结果
def plot_images_labels_prediction(images, labels, prediction, index, num=10): #index是起始图的索引，从0开始
    fig = plt.gcf() # 获取当前图表，get current figure
    fig.set_size_inches(10, 12)  # 1英寸=2.54厘米
    if num > 25:
        num = 25     # 最多显示25个子图

    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1)   # 获取当前要处理的子图
        ax.imshow(np.reshape(images[index], (28, 28)), cmap="binary")  #显示第index个图像
        title = "label=" + str(np.argmax(labels[index])) # 构建图上要显示的主题

        if len(prediction) > 0:
            title += ", predict=" + str(prediction[index])

        ax.set_title(title, fontsize=10)   # 显示图上的title信息
        ax.set_xticks([])    # 不显示坐标轴
        ax.set_yticks([])
        index += 1

    plt.show()
plot_images_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result, 149, 25)

