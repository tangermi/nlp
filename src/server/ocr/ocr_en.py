# -*- coding:utf-8 -*-
from PIL import Image, ImageOps
import string
import numpy as np
import cv2
import html
import re


class OcrEn:
    def __init__(self, dic_config):
        self.logger = dic_config.get('logger', None)
        self.weight_path = dic_config['weight_path']

        self.hyperparams = dic_config['hyperparams']

        chars = string.printable[:95]
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

    @staticmethod
    def augmentation(imgs,
                     rotation_range=0,
                     scale_range=0,
                     height_shift_range=0,
                     width_shift_range=0,
                     dilate_range=1,
                     erode_range=1):
        """Apply variations to a list of images (rotate, width and height shift, scale, erode, dilate)"""

        imgs = imgs.astype(np.float32)
        print(imgs.shape)
        _, h, w = imgs.shape

        dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
        erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
        height_shift = np.random.uniform(-height_shift_range, height_shift_range)
        rotation = np.random.uniform(-rotation_range, rotation_range)
        scale = np.random.uniform(1 - scale_range, 1)
        width_shift = np.random.uniform(-width_shift_range, width_shift_range)

        trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
        rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

        trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
        rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
        affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

        for i in range(len(imgs)):
            imgs[i] = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
            imgs[i] = cv2.erode(imgs[i], erode_kernel, iterations=1)
            imgs[i] = cv2.dilate(imgs[i], dilate_kernel, iterations=1)

        return imgs

    @staticmethod
    def normalization(imgs):
        """Normalize list of images"""

        imgs = np.asarray(imgs).astype(np.float32)
        _, h, w = imgs.shape

        for i in range(len(imgs)):
            m, s = cv2.meanStdDev(imgs[i])
            imgs[i] = imgs[i] - m[0][0]
            imgs[i] = imgs[i] / s[0][0] if s[0][0] > 0 else imgs[i]

        return np.expand_dims(imgs, axis=-1)

    @staticmethod
    def text_standardize(text):
        """Organize/add spaces around punctuation marks"""

        if text is None:
            return ""

        RE_DASH_FILTER = re.compile(r'[\-\˗\֊\‐\‑\‒\–\—\⁻\₋\−\﹣\－]', re.UNICODE)
        RE_APOSTROPHE_FILTER = re.compile(r'&#39;|[ʼ՚＇‘’‛❛❜ߴߵ`‵´ˊˋ{}{}{}{}{}{}{}{}{}]'.format(
            chr(768), chr(769), chr(832), chr(833), chr(2387),
            chr(5151), chr(5152), chr(65344), chr(8242)), re.UNICODE)
        RE_RESERVED_CHAR_FILTER = re.compile(r'[¶¤«»]', re.UNICODE)
        RE_LEFT_PARENTH_FILTER = re.compile(r'[\(\[\{\⁽\₍\❨\❪\﹙\（]', re.UNICODE)
        RE_RIGHT_PARENTH_FILTER = re.compile(r'[\)\]\}\⁾\₎\❩\❫\﹚\）]', re.UNICODE)
        RE_BASIC_CLEANER = re.compile(r'[^\w\s{}]'.format(re.escape(string.punctuation)), re.UNICODE)

        LEFT_PUNCTUATION_FILTER = """!%&),.:;<=>?@\\]^_`|}~"""
        RIGHT_PUNCTUATION_FILTER = """"(/<=>@[\\^_`{|~"""
        NORMALIZE_WHITESPACE_REGEX = re.compile(r'[^\S\n]+', re.UNICODE)

        text = html.unescape(text).replace("\\n", "").replace("\\t", "")

        text = RE_RESERVED_CHAR_FILTER.sub("", text)
        text = RE_DASH_FILTER.sub("-", text)
        text = RE_APOSTROPHE_FILTER.sub("'", text)
        text = RE_LEFT_PARENTH_FILTER.sub("(", text)
        text = RE_RIGHT_PARENTH_FILTER.sub(")", text)
        text = RE_BASIC_CLEANER.sub("", text)

        text = text.lstrip(LEFT_PUNCTUATION_FILTER)
        text = text.rstrip(RIGHT_PUNCTUATION_FILTER)
        text = text.translate(str.maketrans({c: f" {c} " for c in string.punctuation}))
        text = NORMALIZE_WHITESPACE_REGEX.sub(" ", text.strip())

        return text

    def decode(self, text):
        """Decode vector to text"""
        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)
        decoded = self.text_standardize(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "")

    def process(self, img):
        img = Image.open(img).rotate(-90, expand=True).convert('L').resize((128, 1024))
        img = ImageOps.mirror(img)
        img = np.asarray(img)

        img = np.expand_dims(img, axis=0)
        img = self.augmentation(img,
                                rotation_range=1.5,
                                scale_range=0.05,
                                height_shift_range=0.025,
                                width_shift_range=0.05,
                                erode_range=5,
                                dilate_range=3)
        img = self.normalization(img)

        return img

    def build_model(self):
        from src.utils.network.model import HTRModel
        input_size = (1024, 128, 1)
        arch = self.hyperparams['arch']

        # create and compile HTRModel
        # note: `learning_rate=None` will get architecture default value
        self.model = HTRModel(architecture=arch, input_size=input_size, vocab_size=len(self.chars))
        print(self.weight_path)
        self.model.load_checkpoint(self.weight_path)

    def _predict(self, img):
        self.build_model()

        img = self.process(img)

        # predict() function will return the predicts with the probabilities
        predicts, _ = self.model.predict(img, ctc_decode=True,)

        # decode to string
        predicts = [self.decode(x[0]) for x in predicts]

        return predicts[0]