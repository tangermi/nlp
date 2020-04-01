# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
from tensorflow.keras import datasets, layers, models
from utils.plot.accuracy_loss import Ploter


# 训练模型
class Cnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        # self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.hyperparams = self.dic_engine['hyperparams']
        self.epoch = self.hyperparams['epoch']

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])
        self.img_path_accuracy = os.path.join(self.dic_engine['_out'], self.dic_engine['img_path_accuracy'])
        self.img_path_loss = os.path.join(self.dic_engine['_out'], self.dic_engine['img_path_loss'])

    def load(self):
        with np.load(self.train_path) as train:
            self.train_images = train['train_images']
            self.train_labels = train['train_labels']

    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # Add Dense layers on top
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

        # self.logger.info(self.model.summary())

    def train(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(self.train_images, self.train_labels, epochs=self.epoch, validation_split=0.2,)

    # 以图片记录训练过程
    def plot(self):
        ploter = Ploter()
        ploter.plot(self.history, self.img_path_accuracy, self.img_path_loss)

    def dump(self):
        # 保存模型
        self.model.save(self.model_path)

    def run(self):
        self.init()
        self.load()
        self.build_model()
        self.train()
        self.plot()
        self.dump()
