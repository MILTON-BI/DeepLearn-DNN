import tensorflow as tf
import numpy as np

arr1 = np.array([1,3,2,5,7,0])
arr2 = np.array([[1.0,2,3],[3,2,1],[4,7,2],[8,3,2]])
print("array1=", arr1)
print("array2=\n", arr2)

argmax1 = tf.argmax(arr1)
argmax20 = tf.argmax(arr2, 0)  # 指定第二个参数为0，按第一维(行)的元素取值，即同列的每一行
argmax21 = tf.argmax(arr2, 1)  # 指定第二个参数为1，按第二维(列)的元素取值，即同行的每一列
argmax22 = tf.argmax(arr2, -1) # 指定第二个参数为-1，按最后一维(行)的元素取值

with tf.Session() as sess:
    print(argmax1.eval())   # 输出4
    print(argmax20.eval())  # 输出[3 2 0]
    print(argmax21.eval())  # 输出[2 0 1 0]
    print(argmax22.eval())  # 输出同上，因为只有两维
