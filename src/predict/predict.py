# -*- coding: utf-8 -*-
from .base import Base


# 预测入口类
class Predict(Base):
    def __init__(self, dic_config={}):
        self.dic_predict = dic_config['predict']
        Base.__init__(self, dic_config)

    def get_engine(self, name, dic_engine={}):
        if 'multinomial_nb' == name:
            from .classify.multinomial_nb import _MultinomialNB
            return _MultinomialNB(self.dic_config, dic_engine)

        if 'xgboost' == name:
            from .classify.xgboost import _XGBoost
            return _XGBoost(self.dic_config, dic_engine)

        if 'mlp' == name:
            from .classify.mlp import Mlp
            return Mlp(self.dic_config, dic_engine)

        if 'cnn' == name:
            from .classify.cnn import Cnn
            return Cnn(self.dic_config, dic_engine)

        if 'resnet' == name:
            from .classify.resnet import Resnet
            return Resnet(self.dic_config, dic_engine)

        if 'nn' == name:
            from .classify.nn import Nn
            return Nn(self.dic_config, dic_engine)

        if 'fasttext' == name:
            from .classify.fasttext import Fasttext
            return Fasttext(self.dic_config, dic_engine)

        ##################################################
        if 'ocr' == name:
            from .ocr.ocr import OCR
            return OCR(self.dic_config, dic_engine)
        #
        # if 'xgb_classifier' == name:
        #     from .xgb_classifier import XgbClassifier
        #     self.engine = XgbClassifier(self.dic_config, dic_engine)
        #
        # if 'captcha_mobilenet' == name:
        #     from .captcha.captcha_mobilenet import CaptchaMobilenet
        #     self.engine = CaptchaMobilenet(self.dic_config, dic_engine)
        #
        # if 'gru' == name:
        #     from .gru import GRU
        #     self.engine = GRU(self.dic_config, dic_engine)
        #
        # if 'logistic' == name:
        #     from .logistic import Logistic
        #     self.engine = Logistic(self.dic_config, dic_engine)
        #
        # if 'xgb_ranker' == name:
        #     from .xgb_ranker import XgbRanker
        #     self.engine = XgbRanker(self.dic_config, dic_engine)
        #
        # if 'bi_lstm' == name:
        #     from .bi_lstm import BiLSTM
        #     self.engine = BiLSTM(self.dic_config, dic_engine)
        #
        # elif model_name == 'sample_bayes':
        #     import bayes
        #     self.model = bayes.Bayes(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'logistic':
        #     import logistic
        #     self.model = logistic.Logistic(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'ai_xgboost':
        #     import ai_xgboost
        #     self.model = ai_xgboost.Ai_xgboost(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'softmax':
        #     import softmax
        #     self.model = softmax.Softmax(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'softmax_keras':
        #     import softmax_keras
        #     self.model = softmax_keras.Softmax_keras(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'cnn_keras':
        #     import cnn_keras
        #     self.model = cnn_keras.Cnn_keras(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'captcha':
        #     import captcha
        #     self.model = captcha.Captcha(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'ai_lstm':
        #     import ai_lstm
        #     self.model = ai_lstm.Ai_lstm(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'ai_face':
        #     import ai_face
        #     self.model = ai_face.Ai_face(self.dic_config[model_name], dic_engine)

    def run(self):
        self.logger.info('begin predict')
        for task_name in self.dic_predict.get('task'):
            self.logger.info(task_name)
            engine_name = self.dic_predict[task_name].get('engine', task_name)
            engine = self.get_engine(engine_name, self.dic_predict[task_name])
            if engine:
                engine.run()
        self.logger.info('end predict')
