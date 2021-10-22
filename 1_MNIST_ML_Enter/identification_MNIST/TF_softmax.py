# coding:utf-8

#导入tensorflow模块
import tensorflow as tf

#导入MNIST教学模块
from tensorflow.core.example.tutorials.mnist import input_data
#读取MNIST数据
mnist = input_data.read_data_sets('1_MNIST_ML_Enter/MNIST_data/' , one_hot = True)

#创建一个占位符x，代表待识别的图片
x = tf.placeholder(tf.float32 , [None,784])

#W是softmax模型的参数，将一个784维的输入转化成一个10维的输出
W = tf.Variable(tf.zeros([784,10]))
#b是又一个softmax的参数，一般被叫做偏置项
b = tf.Variable(tf.zeros([10]))

#y表示模型输出
y = tf.nn.softmax(tf.matmul(x , W) + b)

#y_是实际的图像标签，同样以占位符表示
y_ = tf.placeholder(tf.float32 , [None , 10])

"""
上述代码我们得到了两个重要的Tensor，y 与 y_
y是模型的输出，y_是实际的图像标签，注：y_是独热表示
"""
#根据y 与 y_构造交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

#有了损失，就可以用梯度下降法针对模型的参数（W和b）进行优化
"""
TensorFlow 会默认计算所有变量的梯度，并更新他们的值
下面代码中的 0.01 是优化器的学习率（Learning Rate）
"""
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#在优化前，需要创建一个会话Session，并在会话中对变量进行初始化
#这里我们创建一个Session，因为只有在Session中才能运行优化步骤 train_step
sess = tf.InteractiveSession()
#运行之前必须要初始化所有变量，分配内存
tf.global_variables_initializer().run()

"""
有了会话，就可以对变量W、b进行优化了，优化程序的代码如下
"""
#进行1000步梯度下降
for _ in range(1000):
    #在mnist中取100个训练数据
    #batch_xs的形状为(100,784)的图像数据。
    print(_)