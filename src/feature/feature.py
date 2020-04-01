# -*- coding: utf-8 -*-
from .base import Base


class Feature(Base):
    def __init__(self, dic_config={}):
        self.dic_feature = dic_config['feature']
        Base.__init__(self, dic_config)

    def get_engine(self, name, dic_engine=[]):
        if 'multinomial_nb' == name:
            from .text.multinomial_nb import _MultinomialNB
            return _MultinomialNB(self.dic_config, dic_engine)

        if 'xgboost' == name:
            from .text.xgboost import _XGBoost
            return _XGBoost(self.dic_config, dic_engine)

        if 'tg_rnn' == name:
            from .text.tg_rnn import TGRnn
            return TGRnn(self.dic_config, dic_engine)

        if 'memnn' == name:
            from .qa.memnn import Memnn
            return Memnn(self.dic_config, dic_engine)

        if 'seq2seq' == name:
            from .text.seq2seq import Seq2seq
            return Seq2seq(self.dic_config, dic_engine)

        if 'fasttext' == name:
            from .text.fasttext import Fasttext
            return Fasttext(self.dic_config, dic_engine)

        if 'gan' == name:
            from .gan.mnist_gan import Gan
            return Gan(self.dic_config, dic_engine)

####################################################################
        if 'siamese' == name:
            from .similarity.siamese import Siamese
            return Siamese(self.dic_config, dic_engine)

####################################################################
        if 'addition_rnn' == name:
            from .calculation.addition_rnn import AdditionRnn
            return AdditionRnn(self.dic_config, dic_engine)

        # if 'word2id' == name:
        #     from .word2id import Word2ID
        #     self.engine = Word2ID(self.dic_config, dic_engine)
        #
        # if 'word2vec' == name:
        #     from .word2vector import Word2Vector
        #     self.engine = Word2Vector(self.dic_config, dic_engine)

    def run(self):
        self.logger.info('begin feature')
        for task_name in self.dic_feature.get('task'):
            self.logger.info(task_name)
            engine_name = self.dic_feature[task_name].get('engine', task_name)
            engine = self.get_engine(engine_name, self.dic_feature[task_name])
            if engine:
                engine.run()
        self.logger.info('end feature')
