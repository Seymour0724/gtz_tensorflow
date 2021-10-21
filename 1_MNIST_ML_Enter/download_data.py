# coding:utf-8
"""
MNIST数据集主要是由一些手写数字的图片和相对应的标签组成，图片一共有十类，分别是0~9，十个阿拉伯数字组成。

其中，MNIST有两类数据：
    1.训练集图像：train_images-idx3-ubyte.gz 和 train_labels-idx1-ubyte.gz
    2.测试集图像：t10k_images-idx3-ubyte.gz 和 t10k_labels-idx1-ubyte.gz

训练图像数目：60000、测试图像数目：10000
"""

# 从tensorflow.examples.tutorials.mnist引入模块
from tensorflow.core.example.tutorials.mnist import input_data
# 从MNIST_data中读取MNIST数据，下面这条语句在数据不存在时候会自行下载
mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True)

# 查看训练数据的大小
print("训练数据的大小为：")
print(mnist.train.images.shape)
print(mnist.train.labels.shape)

# 查看验证数据的大小
print("验证数据的大小为：")
print(mnist.validation.images.shape)
print(mnist.validation.images.shape)
print('\n')

# 查看测试数据的大小
print("验证数据的大小为：")
print(mnist.test.images.shape)
print(mnist.test.images.shape)
print('\n')

#查看一下数据第0张训练图片对应的向量表示
print("mnist数据集第0张图片的向量表示：")
print(mnist.train.images[0,:])
print('\n')