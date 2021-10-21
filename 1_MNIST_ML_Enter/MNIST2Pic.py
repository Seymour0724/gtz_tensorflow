"""
将MNIST数据集保存为图片：
    将MNIST数据集读取出来，并保存为图片文件。
"""

# coding : utf-8
from tensorflow.core.example.tutorials.mnist import input_data
from PIL import Image
import scipy.misc
import os

# 读取MNIST数据集，如果数据不存在会事先下载
mnist = input_data.read_data_sets('1_MNIST_ML_Enter/MNIST_data/' , one_hot = True)

# 把原始图片保存在 "1_MNIST_ML_Enter/MNIST_data/raw" 这个路径下。
# 如果没有这个文件夹，代码会自动创建raw文件夹
save_dir = '1_MNIST_ML_Enter/MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前20张图片
for i in range(20):
    #注意，mnist.train.images[i,:]就是表示第i张图片（从序号0开始）
    image_array = mnist.train.images[i,:]
    #tensorflow中的MNIST图片是一个784维的向量=28x28维的图像
    image_array = image_array.reshape(28,28)
    #保存文件的格式为
    #mnist_train_0.jpg,mnist_train_1.jpg...mnist_train_19.jpg
    filename = save_dir+'mnist_train_%d.jpg' %i
    #将image_array保存为图片
    #先用scipy.misc.toimage转换为图像，再调用save直接保存
    Image.fromarray((image_array*255).astype('uint8'), mode='L').convert('RGB').save(filename)