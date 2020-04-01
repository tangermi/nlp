# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf


class CaptchaEn:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_file']

        self.charset = dic_config['charset']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def convert2gray(self, img):
        if len(img.shape) <= 2:
            return img

        gray = np.mean(img, -1)
        return gray

    def preprocess(self, img):
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160

        new_img = np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        img = tf.reshape(self.convert2gray(img), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        new_img[0, :] = img
        return new_img

    def vec2text(self, vec):
        text = []
        for i, c in enumerate(vec):
            text.append(self.charset[c])
        return "".join(text)

    def _predict(self, img):
        new_img = self.preprocess(img)
        prediction_value = self.model.predict(new_img)
        return self.vec2text(np.argmax(prediction_value, axis=2)[0])
