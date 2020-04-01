# -*- coding:utf-8 -*-
from ..base import Base
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D


# 训练模型
class Fasttext(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])
        self.max_features_path = os.path.join(self.dic_engine['_in'], self.dic_engine['max_features_file'])

        self.hyperparams = self.dic_engine['hyperparams']

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])


    def load(self):
        with np.load(self.train_path) as train:
            self.x_train = train['x_train']
            self.y_train = train['y_train']

        with np.load(self.test_path) as test:
            self.x_test = test['x_test']
            self.y_test = test['y_test']

        self.max_features = np.load(self.max_features_path)

    def build_model(self):
        embedding_dims = self.hyperparams['embedding_dims']
        maxlen = self.hyperparams['maxlen']

        model = Sequential()
        model.add(Embedding(self.max_features, embedding_dims, input_length=maxlen))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))

        return model

    def train(self):
        batch_size = self.hyperparams['batch_size']
        epochs = self.hyperparams['epochs']

        self.model = self.build_model()
        self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_data=[self.x_test, self.y_test])

    def dump(self):
        # 保存模型
        self.model.save(self.model_path)

    def run(self):
        self.init()
        self.load()
        self.process()
        self.build_model()
        self.train()
        self.dump()
