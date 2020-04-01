# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image


class Cnn:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def process(self, img):
        self.img = np.array([np.array(Image.open(img).resize((32, 32))).astype('float32') / 255.0])

    def _predict(self, img):
        self.process(img)
        return self.model.predict_classes(self.img)[0]
