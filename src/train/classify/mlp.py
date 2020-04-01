# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from utils.plot.accuracy_loss import Ploter


# 训练模型
class Mlp(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        # self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.hyperparams = self.dic_engine['hyperparams']
        self.epoch = self.hyperparams['epoch']
        self.batch_size = self.hyperparams['batch_size']
        self.activation = self.hyperparams['activation']
        self.verbose = self.hyperparams['verbose']

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])
        self.img_path_accuracy = os.path.join(self.dic_engine['_out'], self.dic_engine.get('img_path_accuracy', 'accuracy'))
        self.img_path_loss = os.path.join(self.dic_engine['_out'], self.dic_engine.get('img_path_loss', 'loss'))

    def load(self):
        with np.load(self.train_path) as train:
            self.x_train_n = train['x_train_n']
            self.y_train_onehot = train['y_train_onehot']

    def build_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=1000, input_dim=784, kernel_initializer='normal', activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

        # self.logger(self.model.summary())

    def train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.history = self.model.fit(x=self.x_train_n,
                                      y=self.y_train_onehot,
                                      validation_split=0.2,
                                      epochs=self.epoch,
                                      batch_size=self.batch_size,
                                      verbose=self.verbose)

    # 以图片记录训练过程
    def plot(self):
        ploter = Ploter()
        ploter.plot(self.history, self.img_path_accuracy, self.img_path_loss)

    def dump(self):
        # 保存模型
        model_file = self.model_path
        self.model.save(model_file)

    def run(self):
        self.init()
        self.load()
        self.build_model()
        self.train()
        self.plot()
        self.dump()
