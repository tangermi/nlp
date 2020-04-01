# -*- coding:utf-8 -*-
from ..base import Base
from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import os


class Captcha(Base):
    def __init__(self, dic_config={}, dic_engine={}, dic_score={}):
        self.dic_engine = dic_engine
        self.dic_score = dic_score
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)
        self.logger.info(self.dic_score)

    def init(self):
        self.model_path = os.path.join(self.dic_engine['_in'], self.dic_engine['model_file'])

        self.out_file = self.dic_engine['_out']

        number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u',
                    'v', 'w', 'x', 'y', 'z']
        ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U',
                    'V', 'W', 'X', 'Y', 'Z']
        self.CHAR_SET = number + alphabet + ALPHABET
        self.CHAR_SET_LEN = len(self.CHAR_SET)
        text, image = self.gen_captcha_text_and_image(char_set=self.CHAR_SET)
        self.MAX_CAPTCHA = len(text)

    def load(self):
        print(self.model_path)
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

    def random_captcha_text(self, char_set=None, captcha_size=4):
        if char_set is None:
            char_set = number + alphabet + ALPHABET

        captcha_text = []
        for i in range(captcha_size):
            c = random.choice(char_set)
            captcha_text.append(c)
        return captcha_text

    def gen_captcha_text_and_image(self, width=160, height=60, char_set=[]):
        image = ImageCaptcha(width=width, height=height)

        captcha_text = self.random_captcha_text(char_set)
        captcha_text = ''.join(captcha_text)

        captcha = image.generate(captcha_text)

        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image

    def convert2gray(self, img):
        if len(img.shape) > 2:
            gray = np.mean(img, -1)
            return gray
        else:
            return img

    def text2vec(self, text):
        vector = np.zeros([self.MAX_CAPTCHA, self.CHAR_SET_LEN])
        for i, c in enumerate(text):
            idx = self.CHAR_SET.index(c)
            vector[i][idx] = 1.0
        return vector

    def vec2text(self, vec):
        text = []
        for i, c in enumerate(vec):
            text.append(self.CHAR_SET[c])
        return "".join(text)

    def wrap_gen_captcha_text_and_image(self):
        while True:
            text, image = self.gen_captcha_text_and_image(char_set=self.CHAR_SET)
            if image.shape == (60, 160, 3):
                return text, image

    def get_next_batch(self, batch_size=128):
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160

        batch_x = np.zeros([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        batch_y = np.zeros([batch_size, self.MAX_CAPTCHA, self.CHAR_SET_LEN])

        for i in range(batch_size):
            text, image = self.wrap_gen_captcha_text_and_image()
            image = tf.reshape(self.convert2gray(image), (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
            batch_x[i, :] = image
            batch_y[i, :] = self.text2vec(text)

        return batch_x, batch_y

    def evaluate(self):
        success = 0
        count = 100
        self.text = ''
        for _ in range(count):
            data_x, data_y = self.get_next_batch(1)
            prediction_value = self.model.predict(data_x)
            data_y = self.vec2text(np.argmax(data_y, axis=2)[0])
            prediction_value = self.vec2text(np.argmax(prediction_value, axis=2)[0])

            if data_y.upper() == prediction_value.upper():
                self.text += f'y预测={prediction_value} \t y实际={data_y} \t 预测成功。\n'
                # self.logger.info(f'y预测={prediction_value} \t y实际={data_y}预测成功。')
                success += 1
            else:
                self.text += f'y预测={prediction_value} \t y实际={data_y} \t 预测失败。\n'
                # self.logger.info(f'y预测={prediction_value} \t y实际={data_y}预测失败。')
        self.text += f'预测{count}次，成功率={success / count}'
        self.logger.info(f'预测{count}次，成功率={success / count}')

    def dump(self):
        with open(self.out_file, 'w', encoding='utf-8') as f:
            f.seek(0)
            f.write(self.text)
            f.truncate()

    def run(self):
        self.init()
        self.load()
        self.evaluate()
        self.dump()
