# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageOps


class Mlp:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def process(self, img_file):
        self.im = np.array(ImageOps.invert(Image.open(img_file)).convert('L').resize((28, 28)))
        self.im = self.im.reshape(1, 784).astype('float32')

        # normalization
        self.im = self.im / 255

    def _predict(self, img_file):
        self.process(img_file)
        prediction = self.model.predict_classes([self.im])
        return prediction[0]
