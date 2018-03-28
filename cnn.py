# -*- coding=utf-8 -*-
import numpy as np
import tensorflow as tf
#下载并载入MNIST手写数字库（55000*28*28）55000张训练图像
from tensorflow .examples.tutorials.mnist import input_data
#one_hot独热码的编码（encoding）形式
#0,1,2到9的10个数字
# 0 表示为0 ： 1000000000
# 1 表示为1 ： 0100000000
# 2 表示为2 ： 0010000000等以此类推
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#输入图片为28*28
#None表示张量（Tensor）的第一个维度可以是任何长度
input_x = tf.placeholder(tf.float32,[None,28*28])

#输出为10个数字
output_y = tf.placeholder(tf.int32,[None,10])

#改变矩阵维度，-1表示根据实际情况而定 1表示图片深度,改变形状之后的输入
images = tf.reshape(input_x,[-1,28,28,1])

#从Test数据集里选取3000个手写数字的图片和对应标签
test_x = mnist.test.images[:3000]#测试图片
test_y = mnist.test.labels[:3000]#测试标签


#构建卷积神经网络
#第一层卷积
conv1 = tf.layers.conv2d(inputs=images,
                         filters=6,
                         kernel_size=[5,5],
                         strides=1,
                         padding='same',
                         activation=tf.nn.relu
                         )#输入的形状是32*32（图片外围加入了两行0），输出的形状是28*28
#第一层池化
pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2,2],
                                strides=2
                                )#形状变为[14*14,32]
#第二层卷积
conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=6,
                         kernel_size=[5,5],
                         strides=1,
                         padding='same',
                         activation=tf.nn.relu)#形状变为[14*14,64]
#第二层池化
pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                pool_size=[2,2],
                                strides=2
                                )#形状变为[7*7,64]

#平坦化
flat = tf.reshape(pool2,[-1,7*7*6])

#经过全连接层
dense = tf.layers.dense(inputs=flat,
                        units=1024,
                        activation=tf.nn.relu)

#dropout 丢弃50%

dropout = tf.layers.dropout(inputs=dense,rate=0.5)

#10个神经元的全连接层，不用激励函数来做非线性化
logits = tf.layers.dense(inputs=dropout,units=10)

#计算误差，再用softmax计算百分比概率

loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y,
                                       logits=logits)

#用一个Adam优化器来最小化误差，学习率设为0.001

#train_op = tf.train.AdamOptimizer(learning_rate=0.001.minimize(loss))
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
#计算与测试值和实际值的匹配程度
#返回（accuracy，update_op），会创建两个局部变量
accuracy = tf.metrics.accuracy(labels=tf.argmax(output_y,axis=1),
                               predictions=tf.argmax(logits,axis=1),)[1]



#创建会话
sess = tf.Session()
#初始化变量,全局和局部

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

sess.run(init)

for x in range(2000):
    batch = mnist.train.next_batch(50) #从train数据集里去下一个50个样本
    train_loss,train_op_ = sess.run([loss,train_op],{input_x:batch[0],output_y:batch[1]})
    if x % 100 == 0:
        test_accuracy = sess.run(accuracy,{input_x:test_x,output_y:test_y})
        print("step=%d,train loss = %.4f,test accuracy = %.2f" % (x,train_loss,test_accuracy))


#测试，打印20个测试值和真实值的对
test_output = sess.run(logits,{input_x:test_x[:20]})
inferenced_y = np.argmax(test_output,1)
print(inferenced_y,'inferenced numbers')#推测的数字
print(np.argmax(test_y[:20],1),'real numbers')#真实的数字










