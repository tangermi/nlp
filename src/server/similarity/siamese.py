# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as plt


# 运行结果目前有问题，待修改
class Siamese:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def process(self, img_file_1, img_file_2):
        self.im_1 = np.array(Image.open(img_file_1).convert('L').resize((28, 28))).astype('float32') / 255.0
        self.im_2 = np.array(Image.open(img_file_2).convert('L').resize((28, 28))).astype('float32') / 255.0
        pair = [[self.im_1, self.im_2]]
        self.pair = np.array(pair)
        self.pair = self.pair.reshape(self.pair.shape[0], 2, 28, 28, 1)

    def _predict(self, img_file_1, img_file_2):
        self.process(img_file_1, img_file_2)
        # 小于0.5为相同种类， 大于0.5为不同种类
        prediction = self.model.predict([self.pair[:, 0], self.pair[:, 1]])
        return prediction[0]
