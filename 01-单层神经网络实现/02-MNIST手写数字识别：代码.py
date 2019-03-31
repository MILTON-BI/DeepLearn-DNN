import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------第一步：数据读入和处理-------------------------------------------------

mnist = input_data.read_data_sets('MNIST_data_set/', one_hot=True)

# -------------------------------------第二步：模型构建-------------------------------------------------
"""2.1 定义待输入数据的占位符"""
# 特征数据：mnist中的图片28 * 28 = 784px
x = tf.placeholder(tf.float32, shape=(None, 784), name='X')
# 标签数据：标签为0-9十个数字（one-hot数据类别）
y = tf.placeholder(tf.float32, shape=(None, 10), name='Y')

"""2.2 定义模型变量"""
# w初始化为一个符合正态分布的随机数
w = tf.Variable(tf.random_normal(shape=(784, 10), name='W'))
b = tf.Variable(tf.zeros([10]), name='B')

"""2.3 用单个神经元构建模型"""
# 定义前向计算
forward = tf.matmul(x, w) + b
# 用softmax进行分类化
pred = tf.nn.softmax(forward)


# -------------------------------------第三步：训练模型-------------------------------------------------
# 3.1 设置训练参数
train_epochs = 200  # 训练轮数（超参数）
batch_size = 100   # 单次训练的样本数（批次批量大小）
total_batch = int(mnist.train.num_examples / batch_size) # 计算训练一轮有多少批次
display_step = 1   # 显示粒度
learning_rate = 0.01  # 学习率参数（超参数）

# 3.2 定义交叉熵损失函数
loss_func = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# 3.3 选择和定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_func)

# 3.4 定义准确率
# 检查预测类别tf.argmax(pred,1)与实际类别tf.argmax(y, 1)的匹配程度
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# 准确率：将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 3.5 启动会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 3.6 模型训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size) # 读取批次数据
        sess.run(optimizer, feed_dict={x: xs, y: ys}) # 执行此批次的训练

    # total_batch个批次执行完成后，使用验证数据计算误差与准确率；验证集不分批
    loss, acc = sess.run([loss_func, accuracy], feed_dict={x: mnist.validation.images, y: mnist.validation.labels})

    # 输出训练过程中的详细信息
    if (epoch+1) % display_step == 0:
        print("训练轮次：", '%02d' % (epoch+1), "  损失值：", "{:.9f}".format(loss), "  准确率：", "{:.4f}".format(acc))

print("训练结束！！")
# 损失值趋于减小，准确率趋于上升

# ------------------------------------第四步：用测试集评估模型------------------------------------------------
# 在测试集上评估模型准确率
accu_test = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})

print("测试准确率：", accu_test)

# ------------------------------------第五步：用模型进行预测------------------------------------------------
# 预测结果是one-hot编码格式，需要转换成0-9的数字
prediction_result0 = sess.run(tf.argmax(pred,1), feed_dict={x: mnist.test.images})
# 察看预测结果的前10项
print(prediction_result0[:10])

# 可视化预测结果（与标签同时显现）
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

# 调用上面函数显示预测结果图像
plot_images_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result0, 10, 10)
