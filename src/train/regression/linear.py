# -*- coding:utf-8 -*-
from ..base import Base
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils.progress_bar import ProgressBar
import matplotlib.pyplot as plt
import numpy as np


# 训练模型
class Linear(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])
        self.img_path_mse = os.path.join(self.dic_engine['_out'], self.dic_engine['out_img_mse'])
        self.img_path_mae = os.path.join(self.dic_engine['_out'], self.dic_engine['out_img_mae'])

    def load(self):
        # self.train_labels = pd.read_csv(self.train_labels_path, header=0, index_col=0, squeeze=True, encoding='utf-8')
        # self.normed_train_data = pd.read_csv(self.train_path, header=0, index_col=0, encoding='utf-8')
        self.hyperparams = self.dic_engine['hyperparameter']
        with np.load(self.data_path) as train:
            self.train_labels = train['train_labels']
            self.normed_train_data = train['normed_train_data']

    def build_model(self):
        activation = self.hyperparams['activation']

        normed_train_data = self.normed_train_data
        # input_dim = normed_train_data.shape
        # self.logger.info(input_dim)
        model = keras.Sequential([
            layers.Dense(64, activation=activation),
            layers.Dense(64, activation=activation),
            layers.Dense(1)
        ])

        model.compile(loss='mse', metrics=['mae', 'mse'], optimizer=tf.keras.optimizers.RMSprop(0.001))
        self.model = model

    def train(self):
        epochs = self.hyperparams['epochs']
        # self.logger.info(normed_train_data.head())
        # self.logger.info(train_labels.head())
        # self.logger.info(type(self.train_labels))
        # self.logger.info(type(self.normed_train_data))
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.history = self.model.fit(self.normed_train_data, self.train_labels, epochs=epochs,
                                      validation_split=0.2, verbose=0,
                                      callbacks=[early_stop, ProgressBar()])

    # 以图片记录训练过程
    def plot(self):
        history = self.history
        # 解决中文乱码问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('metric - MSE')
        plt.plot(hist['epoch'], hist['mse'], label='训练集')
        plt.plot(hist['epoch'], hist['val_mse'], label='验证集')
        plt.ylim([0, 20])
        plt.legend()
        plt.savefig(self.img_path_mse)

        plt.figure()
        plt.xlabel('epoch')
        plt.ylabel('metric - MAE')
        plt.plot(hist['epoch'], hist['mae'], label='训练集')
        plt.plot(hist['epoch'], hist['val_mae'], label='验证集')
        plt.ylim([0, 10])
        plt.legend()
        plt.savefig(self.img_path_mae)

    # 保存模型
    def dump(self):
        self.model.save(self.model_path)

    def run(self):
        self.init()
        self.load()
        self.build_model()
        self.train()
        self.plot()
        self.dump()
