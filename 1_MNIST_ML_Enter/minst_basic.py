#MNIST dataset-->hello world

import tensorflow as tf

#载入mnist数据集
mnist = tf.keras.datasets.mnist

#将样本从整数转为浮点数
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0, x_test/255.0

#以Sequential构建模型，将各个层堆叠起来
model = tf.keras.layers.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128,activation = "relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10,activation="softmax")
    ]
)

model.compile(optimizer="adam",
              loss="sparse_categorcial_crossentropy",
              metrics=["accuracy"])

#训练模型并验证
model.fit(x_train,y_train,epochs=5)
model.evaluate(x_test,y_test,verbose=2)              