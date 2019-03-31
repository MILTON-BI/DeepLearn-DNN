import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------第一步：数据读入和处理-------------------------------------------------

mnist = input_data.read_data_sets('MNIST_data_set/', one_hot=True)
# mnist = input_data.read_data_sets('MNIST_data_set/', one_hot=False)

# print("训练集train数据量：", mnist.train.num_examples)
# print("验证集validation数据量：", mnist.validation.num_examples)
# print("测试集test数据量：", mnist.test.num_examples)
# print('train images shape:', mnist.train.images.shape, 'labels shape:', mnist.train.labels.shape)
# # 图像为28px * 28px = 784
# # label为one hot格式，每个标签数据的长度是10（10分类）
# print(len(mnist.train.images[0]))
# print(mnist.train.images[0].shape)
# print(mnist.train.images[0])
#
# print(mnist.train.images[0].reshape(28, 28))
#
# # 显示图片
# def plot_image(image):
#     plt.imshow(image.reshape(28, 28), cmap="binary")
#     plt.show()
# plot_image(mnist.train.images[0])

# 标签数据和one_hot编码
# print(mnist.train.labels[0:20])
# 如果one_hot = true,则结果显示为“[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]”的形式(表示结果7)
# 如果one_hot = false,则结果直接显示为对应的数值(7)


"""
# one hot 编码：
1. 是一种稀疏向量，其中一个元素为1，其他都为0
2. 常用于表示拥有有限个可能值的字符串或标识符
3. 为什么采用one hot编码
（1）将离散特征的取值扩展到了欧式空间，离散特征的某个取值就对应欧式空间的某个点
（2）在机器学习中，特征之间的距离计算或相似度的常用计算方法都是基于欧式空间的
（3）将离散特征使用one-hot编码，会让特征之间距离计算更加合理
4. one hot编码如何取值？
用np.argmax(): 返回最大数的索引，即值为1的数的索引（其他全为0）
5. 其他编码
（1）如果读入数据时，将参数one_hot设为false，本例中的标签值直接显示为具体的数值
（2）表示形式：mnist_no_one_hot.train.labels[]
"""
# 取one-hot编码最大值的下标（即输出取值）
# print(np.argmax(mnist.train.labels[100]))

"""
# 数据集的划分：
1. 为保证训练效果，一般将数据集进行划分
（1）一种方法是分成两个子集：训练集和测试集（用于测试模型训练效果的子集）
    1）在测试集上是否表现良好，是衡量在新数据上是否表现良好的有用指标，前提是：
        测试集要足够大，可产生具有统计意义的结果（避免碰巧表现良好的情况）；且不反复使用相同的测试集来作假
        能代表整个数据集，测试集的特征应该与训练集的特征相同
    2）训练的流程
        使用训练集进行训练—>使用测试集评估模型—>根据评估情况调整模型（比如调整参数）
        —>用训练集训练—>用测试集评估—>根据评估进行调整—>……
        ！！最终选择在测试集上获得最佳效果的模型
    3）可能的问题
        用训练集和测试集来推动模型迭代开发（每次迭代都会用训练集训练，用测试集评估并调整模型各种超参数）。
        这种迭代多次重复执行，可能导致模型不知不觉地拟合了特定测试集的特征，为避免这个问题，需要对数据集进行进一步的划分
        
（2）划分为三个子集，训练集、验证集、测试集 
    1） 过程：训练集训练模型—>验证集评估模型—>根据验证集获得的评估效果调整模型—>迭代训练、评估和调整
              —> …… —>选择在验证集上获得最佳效果的模型—>使用测试集确定模型的效果    
"""

"""
# 数据批量读取：
1. 切片方法：print(mnist.train.labels[M:N])
2. mnist模块中提供了批量读取数据的方法：mnist.train.next_batch(batch_size=要读取的数据个数)
    next_batch（）方法具体实现的时候：（1）如果每次取数据集的一部分，则下一次会取相同数量的另一个部分，直到把数据集中的数据取完；（2）所有数据取完一轮后，会对数据集进行一个shuffle，下轮再取的时候是一个新的顺序
    
"""
# 批量读取数据
# batch_images_xs, batch_labels_ys = mnist.train.next_batch(batch_size=100)
# print(batch_images_xs, batch_images_xs.shape)
# print(batch_labels_ys, batch_labels_ys.shape)



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

"""2.3 定义前向计算
# softmax方法：
    1. 基本思想；在多类别分类问题中，softmax会为每个类别分配一个用小数表示的概率（相加之和=1）
    2. 在手写数字识别案例中，针对某一个特定的图片（模糊的），softmax会为识别为0-9中每个数字的情况各分配一个概率，取概率最高的情况（最有可能的情况）作为分类的结果
    3. 在神经网络中有softmax层，用其判定输出结果（本例中判定输出结果为0-9中的哪个数字）
    4. softmax方程：pi=e**yi/(e**yk求和)
"""

"""
# 交叉熵损失函数：对多元分类问题
    1.交叉熵是信息论中的一个概念，原来是用来估算平均编码长度的
      （1）给定两个概率分布p和q，通过q来表示p的交叉熵为：H(p,q) = -sigema[(p(x)log(q(x))]
      （2）交叉熵是用来刻画两个概率分布之间的距离，p代表正确答案，q代表预测值，交叉熵越小，两个概率分布越接近
    2.举例：
      （1）有一个三分类问题：某个样例的正确答案是(1,0,0),A模型经过softmax回归之后预测结果是(0.5,0.2,0.3),B模型结果softmax回归后的预测结果是(0.7,0.1,0.2)
      （2）根据H(p,q) = -sigema[(p(x)log(q(x))]
            H((1,0,0),(0.5,0.2,0.3))= -log0.5 约等于0.301
            H((1,0,0),(0.7,0.1,0.2))= -log0.7 约等于0.155
      （3）交叉熵越小，表示预测值与答案越接近，因此本例中B模型的预测结果较好
            
"""

forward = tf.matmul(x, w) + b
# 结果分类：将前向计算的结果分类成（0-9）中某类的概率值
# 本案例从预测问题转成分类问题，从线性回归问题到逻辑回归问题
pred = tf.nn.softmax(forward)

# 定义交叉熵损失函数
loss_func = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))