#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


# 导入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


# In[6]:


#每个批次的大小
batch_size =100
#计算一共有多少个批次
n_batch = mnist.train.num_examples//batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络（无隐藏层）
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy"+str(acc))
            


# In[ ]:




