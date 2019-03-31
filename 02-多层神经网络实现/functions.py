import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['fcn_layer', 'print_predict_errors', 'plot_images_labels_prediction']

# 全连接层函数
def fcn_layer(inputs,            # 输入数据   # fcn是全连接网络的缩写
              input_dim,         # 输入神经元数
              output_dim,        # 输出神经元数
              activation=None):       # 激活函数
    W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))  #以截断型正态分布的随机数初始化W
    b = tf.Variable(tf.zeros([output_dim]))   # 以0初始化b

    XWb = tf.matmul(inputs, W) + b

    if activation is None:  # 默认不适用激活函数
        outputs = XWb
    else:                   # 如果指定激活函数，则用其对输出结果进行变换
        outputs = activation(XWb)

    return outputs

# 查看输出错误的函数
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


# 结果可视化函数
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