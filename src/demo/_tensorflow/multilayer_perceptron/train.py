# -*- coding:utf-8 -*-
import tensorflow as tf
from preprocess import MNISTLoader


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output

    def train(self):
        # 定义一些模型超参数：
        num_epochs = 5
        self.batch_size = 50
        learning_rate = 0.001

        # 实例化模型和数据读取类，并实例化一个 tf.keras.optimizer 的优化器
        self.model = MLP()
        data_loader = MNISTLoader()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        num_batches = int(data_loader.num_train_data // self.batch_size * num_epochs)
        for batch_index in range(num_batches):
            X, y = data_loader.get_batch(self.batch_size)
            with tf.GradientTape() as tape:
                y_pred = self.model(X)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                loss = tf.reduce_mean(loss)
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
            grads = tape.gradient(loss, self.model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, self.model.variables))