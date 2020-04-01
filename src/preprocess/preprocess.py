# -*- coding: utf-8 -*-
# 预处理入口
from .base import Base


class Preprocess(Base):
    def __init__(self, dic_config={}):
        self.dic_preprocess = dic_config['preprocess']
        Base.__init__(self, dic_config)

    def get_engine(self, name):
        if 'sogou' == name:
            from .text.sogou import Sogou
            return Sogou(self.dic_config, self.dic_preprocess[name])

        if 'mpg' == name:
            from .text.mpg import Mpg
            return Mpg(self.dic_config, self.dic_preprocess[name])

        if 'fashion_mnist' == name:
            from .image.fashion_mnist import FashionMnist
            return FashionMnist(self.dic_config, self.dic_preprocess[name])

        if 'mnist' == name:
            from .image.mnist import Mnist
            return Mnist(self.dic_config, self.dic_preprocess[name])

        if 'cifar10' == name:
            from .image.cifar10 import Cifar10
            return Cifar10(self.dic_config, self.dic_preprocess[name])

        ##################################
        if 'shakespeare' == name:
            from .text.shakespeare import Shakespeare
            return Shakespeare(self.dic_config, self.dic_preprocess[name])

        if 'babi' == name:
            from .text.babi import Babi
            return Babi(self.dic_config, self.dic_preprocess[name])

        if 'fra' == name:
            from .text.fra import Fra
            return Fra(self.dic_config, self.dic_preprocess[name])

        if 'imdb' == name:
            from .text.imdb import Imdb
            return Imdb(self.dic_config, self.dic_preprocess[name])

        #####################################
        if 'hwdb' == name:
            from .ocr.hwdb import Hwdb
            return Hwdb(self.dic_config, self.dic_preprocess[name])

        #####################################
        if 'coco2tfrecord' == name:
            from .image.coco2tfrecord import Coco2Tfrecord
            return Coco2Tfrecord(self.dic_config, self.dic_preprocess[name])

        # if 'segment2tf' == name:
        #     from . import segment2tf
        #     self.engine = segment2tf.Segment2tf(self.dic_preprocess[name])
        #
        # if 'h5py2bunch' == name:
        #     from . import h5py2bunch
        #     self.engine = h5py2bunch.H5py2bunch(self.dic_preprocess[name])
        #
        # if 'encoder' == name:
        #     from . import coder
        #     self.engine = coder.Coder(self.dic_preprocess[name])
        #
        # if 'csv2dat' == name:
        #     from . import csv2dat
        #     self.engine = csv2dat.Csv2dat(self.dic_preprocess[name])
        #
        # if 'txt2bunch' == name:
        #     from . import txt2bunch
        #     self.engine = txt2bunch.Txt2bunch(self.dic_preprocess[name])
        #
        # if 'segment2bunch' == name:
        #     from . import segment2bunch
        #     self.engine = segment2bunch.Segment2bunch(self.dic_preprocess[name])
        #
        # if 'bunch2csv' == name:
        #     from . import bunch2csv
        #     self.engine = bunch2csv.Bunch2csv(self.dic_preprocess[name])
        #
        # if 'package_dataset' == name:
        #     from . import package_dataset
        #     self.engine = package_dataset.Package_dataset(self.dic_preprocess[name])
        #
        # if 'img_reshape' == name:
        #     from . import img_reshape
        #     self.engine = img_reshape.Img_reshape(self.dic_preprocess[name])
        #
        # if 'img2bunch' == name:
        #     from . import img2bunch
        #     self.engine = img2bunch.Img2bunch(self.dic_preprocess[name])
        #
        # if 'csv2bunch' == name:
        #     from . import csv2bunch
        #     self.engine = csv2bunch.Csv2bunch(self.dic_preprocess[name])
        #
        # if 'matrix2sparse' == name:
        #     from . import matrix2sparse
        #     self.engine = matrix2sparse.Matrix2sparse(self.dic_preprocess[name])
        #
        # if 'txt2tf' == name:
        #     from . import txt2tf
        #     self.file = txt2tf.Txt2tf(self.dic_preprocess[name])
        #
        # if 'data_extract' == name:
        #     from . import data_extract
        #     self.engine = data_extract.Data_extract(self.dic_preprocess[name])
        #
        # if 'series2supervised' == name:
        #     from . import series2supervised
        #     self.engine = series2supervised.Series2supervised(self.dic_preprocess[name])
        #
        # if 'txt2word' == name:
        #     from . import txt2word
        #     self.engine = txt2word.Txt2Word(self.dic_preprocess[name])
        #
        # if 'word2vec' == name:
        #     from . import word2vec
        #     self.engine = word2vec.Word2Vec(self.dic_preprocess[name])

    def run(self):
        self.logger.info('begin preprocess')
        for task_name in self.dic_preprocess.get('task'):
            self.logger.info(task_name)
            engine_name = self.dic_preprocess[task_name].get('engine', task_name)
            engine = self.get_engine(engine_name)
            if engine:
                engine.run()
        self.logger.info('end preprocess')
