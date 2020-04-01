# -*- coding: utf-8 -*-
from PIL import Image
from captcha.image import ImageCaptcha
import random
import glob
import os
from ..base import Base


class CaptchaEn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.nums = self.dic_engine['nums']
        self.charset = self.dic_engine['charset']
        self.captcha_length = self.dic_engine['captcha_length']
        self.width = self.dic_engine['width']
        self.height = self.dic_engine['height']

        self.out_dir = self.dic_engine['_out'] + '/'

    def generate(self, text=None, width=160, height=60):
        """
        get captcha text and np array
        :param width: 160 as default
        :param height: 60 as default
        :param text: source text
        :return: captcha image and array
        """
        image = ImageCaptcha(width=width, height=height)
        captcha = image.generate(text)
        captcha_image = Image.open(captcha)
        return captcha_image

    def get_random_text(self):
        text = ''
        for i in range(self.captcha_length):
            text += random.choice(self.charset)
        return text

    def delete_old_files(self):
        files = glob.glob(self.out_dir + r'*.png')
        for file in files:
            os.remove(file)

    def dump(self):
        for i in range(int(self.nums)):
            text = self.get_random_text()
            captcha = self.generate(text, self.width, self.height)
            img_path = self.out_dir + str(text) + '.png'
            captcha.save(img_path)

    def generate_one(self, out_dir):
        text = self.get_random_text()
        captcha = self.generate(text, self.width, self.height)
        img_path = out_dir + str(text) + '.png'
        captcha.save(img_path)

    def run(self):
        self.init()
        self.delete_old_files()
        self.dump()
