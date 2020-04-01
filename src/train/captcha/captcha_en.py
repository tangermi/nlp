# -*- coding:utf-8 -*-
from ..base import Base
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import glob


# 训练模型
class CaptchaEn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.img_dir = self.dic_engine['_in'] + '/'

        self.model_path = self.dic_engine['_out']

        self.hyperparams = self.dic_engine['hyperparams']
        self.CHAR_SET = self.dic_engine['charset']
        self.CHAR_SET_LEN = len(self.CHAR_SET)
        self.MAX_CAPTCHA = self.hyperparams['captcha_length']

    def load(self):
        batch_size = self.hyperparams['batch_size']

        imgs = glob.glob(self.img_dir + r'*.png')
        if len(imgs) != batch_size:
            self.logger.info('batch_size unmatched')
            exit()

        self.text_list = []
        self.img_list = []
        for img in imgs:
            text = os.path.splitext(os.path.basename(img))[0]
            self.text_list.append(text)

            img = Image.open(img)
            img = np.array(img)
            self.img_list.append(img)

    # 灰度化
    def convert2gray(self, img):
        if len(img.shape) <= 2:
            return img

        gray = np.mean(img, -1)
        return gray

    def text2vec(self, text):
        vector = np.zeros([self.MAX_CAPTCHA, self.CHAR_SET_LEN])
        for i, c in enumerate(text):
            idx = self.CHAR_SET.index(c)
            vector[i][idx] = 1.0
        return vector

    def get_batch(self, batch_size=128):
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160

        batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        batch_y = np.zeros([batch_size, self.MAX_CAPTCHA, self.CHAR_SET_LEN])

        for i in range(batch_size):
            text = self.text_list[i]
            image = self.img_list[i]
            image = tf.reshape(self.convert2gray(image), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
            batch_x[i, :] = image
            batch_y[i, :] = self.text2vec(text)

        return batch_x, batch_y

    def build_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(32, (3, 3)))
        model.add(tf.keras.layers.PReLU())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

        model.add(tf.keras.layers.Conv2D(64, (5, 5)))
        model.add(tf.keras.layers.PReLU())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

        model.add(tf.keras.layers.Conv2D(128, (5, 5)))
        model.add(tf.keras.layers.PReLU())
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.MAX_CAPTCHA * self.CHAR_SET_LEN))
        model.add(tf.keras.layers.Reshape([self.MAX_CAPTCHA, self.CHAR_SET_LEN]))

        model.add(tf.keras.layers.Softmax())

        return model

    def train(self):
        epochs = self.hyperparams['epochs']
        batch_size = self.hyperparams['batch_size']

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.logger.info('model loaded')
        except Exception as e:
            print('#######Exception', e)
            self.model = self.build_model()

        self.model.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')

        batch_x, batch_y = self.get_batch(batch_size)
        self.model.fit(batch_x, batch_y, epochs=epochs)

    def dump(self):
        self.model.save(self.model_path)

    def run(self):
        self.init()
        self.load()
        self.train()
        # self.dump()
