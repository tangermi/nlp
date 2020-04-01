# -*- coding:utf-8 -*-
from ..base import Base
import os
from keras.models import Sequential
from keras import layers
import numpy as np

'''
training a simple net on Chinese Characters classification dataset
we got about 90% accuracy by simply applying a simple CNN net
'''
class AdditionRnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['in_train'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['in_test'])   # 验证集

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])

        self.hyperparams = self.dic_engine['hyperparams']
        self.epochs = self.hyperparams['epochs']
        self.batch_size = self.hyperparams['batch_size']
        self.num_of_layers = self.hyperparams['num_of_layers']
        self.digits = self.hyperparams['digits']
        self.chars = self.hyperparams['chars']

    def load(self):
        with np.load(self.train_path) as train:
            self.x_train = train['x_train']
            self.y_train = train['y_train']

        with np.load(self.test_path) as test:
            self.y_val = test['y_test']
            self.x_val = test['x_test']

    def build_model(self):
        maxlen = self.digits + 1 + self.digits
        self.logger.info("Building model...")
        model = Sequential()
        model.add(layers.LSTM(128, input_shape=(maxlen, len(self.chars))))
        model.add(layers.core.RepeatVector(self.digits + 1))
        # 创建LAYERS层的RNN网络层
        for _ in range(self.num_of_layers):
            model.add(layers.LSTM(128, return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(len(self.chars))))
        model.add(layers.Activation("softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        self.logger.info(model.summary())

        return model

    def train(self):
        self.model = self.build_model()
        self.model.fit(self.x_train,
                       self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(self.x_val, self.y_val))

    def dump(self):
        self.model.save(self.model_path)

    def run(self):
        self.init()
        self.load()
        self.train()
        self.dump()
