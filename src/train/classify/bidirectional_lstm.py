
from ..base import Base
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional


# 训练模型
class BidirectionalLstm(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.train_path = os.path.join(self.dic_engine['_in'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_in'], self.dic_engine['test_file'])

        self.hyperparams = self.dic_engine['hyperparams']

        self.model_path = os.path.join(self.dic_engine['_out'], self.dic_engine['model_file'])

    def load(self):
        with np.load(self.train_path) as train:
            self.x_train = train['x_train']
            self.y_train = train['y_train']

        with np.load(self.test_path) as test:
            self.x_test = test['x_test']
            self.y_test = test['y_test']

    def build_model(self):
        max_features = self.hyperparams['max_features']
        maxlen = self.hyperparams['maxlen']
        model = Sequential()
        model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        self.logger.info(model.summary())

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
        self.build_model()
        self.train()
        self.dump()
