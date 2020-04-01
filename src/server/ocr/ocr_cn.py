# -*- coding:utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np


class OCRCN:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.model_path = dic_config['model_path']
        self.character_path = dic_config['character_path']

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.characters = open(self.character_path, 'r').readlines()

    def preprocess(self, img_f):
        target_size = 64

        ori_img = cv2.imread(img_f)
        img = tf.expand_dims(ori_img[:, :, 0], axis=-1)
        img = tf.image.resize(img, (target_size, target_size))
        img = (img - 128.) / 128.
        img = tf.expand_dims(img, axis=0)
        # self.logger.info(img.shape)
        return img

    def _predict(self, img_f):
        img = self.preprocess(img_f)
        pred = self.model(img).numpy()
        return self.characters[np.argmax(pred[0])]
