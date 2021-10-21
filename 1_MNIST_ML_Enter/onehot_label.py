"""
图像标签的独热表示：
    所谓独热表示就是：“一位有效编码”。
    我们用N维的向量来表示N个类别，每个类别占据独一位，任何时候读热表示中只有一位是1，其他的都为0。
    例如：[0,0,0,0,0,0,0,1,0,0] = '7'
"""

# codin g : utf-8

import numpy as np
from tensorflow.core.example.tutorials.mnist import input_data

mnist = input_data.read_data_sets('1_MNIST_ML_Enter/MNIST_data' , one_hot = True)

# test one_hot
print(mnist.train.labels[1,:])

#看前二十张图片的label
for i in range(20):
    #得到独热表示
    one_hot_label = mnist.train.labels[i,:]
    #通过np.argmax，是可以获得原始label的
    #因为只有一位是1，其他位都是0
    label= np.argmax(one_hot_label)
    print("mnist_train_%d.jpg label : %d"  %(i,label))