# -*- coding:utf-8 -*-
from src.predict.base import Base
import os
import numpy as np
import tensorflow as tf


class Cnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.test_path = self.dic_engine['_in']
        self.model_path = self.dic_engine['_train']

        self.predicted_path = self.dic_engine['_out']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        with np.load(self.test_path) as test:
            self.test_images = test['test_images']
            self.test_labels = test['test_labels']

    def predict(self):
        self.res = self.model.predict_classes(self.test_images)
        # self.logger.info(self.res)
        # self.logger.info(self.test_labels)

    def dump(self):
        predicted_actual = [self.res, self.test_labels]
        np.save(self.predicted_path, arr=predicted_actual)

    def run(self):
        self.init()
        self.load()
        self.predict()
        self.dump()
