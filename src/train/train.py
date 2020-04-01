# -*- coding: utf-8 -*-
from .base import Base


class Trainer(Base):
    def __init__(self, dic_config={}):
        self.dic_train = dic_config['train']
        Base.__init__(self, dic_config)

    def get_engine_classify(self, name, dic_engine={}):
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

        if 'nn' == name:
            from .classify.nn import Nn
            return Nn(self.dic_config, dic_engine)

        if 'bidirectional_lstm' == name:
            from .classify.bidirectional_lstm import BidirectionalLstm
            return BidirectionalLstm(self.dic_config, dic_engine)

        if 'fasttext' == name:
            from .classify.fasttext import Fasttext
            return Fasttext(self.dic_config, dic_engine)

        if 'resnet' == name:
            from .classify.resnet import Resnet
            return Resnet(self.dic_config, dic_engine)

        return None

    def get_engine_regression(self, name, dic_engine={}):
        if 'linear' == name:
            from .regression.linear import Linear
            return Linear(self.dic_config, dic_engine)

        return None

    def get_engine_similarity(self, name, dic_engine={}):
        if 'siamese' == name:
            from .similarity.siamese import Siamese
            return Siamese(self.dic_config, dic_engine)

        return None

    def get_engine_seq2seq(self, name, dic_engine={}):
        if 'tg_rnn' == name:
            from .seq2seq.tg_rnn import TGRnn
            return TGRnn(self.dic_config, dic_engine)

        if 'lstm' == name:
            from .seq2seq.lstm import _LSTM
            return _LSTM(self.dic_config, dic_engine)

        if 'cnn' == name:
            from .seq2seq.cnn import CNN
            return CNN(self.dic_config, dic_engine)

        return None

    def get_engine_qa(self, name, dic_engine={}):
        if 'memnn' == name:
            from .qa.memnn import Memnn
            return Memnn(self.dic_config, dic_engine)

        return None

    def get_engine_gan(self, name, dic_engine={}):
        if 'gan' == name:
            from .gan.gan import Gan
            return Gan(self.dic_config, dic_engine)

        return None

    def get_engine_ocr(self, name, dic_engine={}):
        if 'ocr_en' == name:
            from .ocr.ocr_en import OcrEn
            return OcrEn(self.dic_config, dic_engine)

        if 'ocr_cn' == name:
            from .ocr.ocr_cn import OcrCn
            return OcrCn(self.dic_config, dic_engine)

        return None

    def get_engine_calculation(self, name, dic_engine={}):
        if 'addition_rnn' == name:
            from .calculation.addition_rnn import AdditionRnn
            return AdditionRnn(self.dic_config, dic_engine)

        return None

    def get_engine_captcha(self, name, dic_engine={}):
        if 'captcha_en' == name:
            from .captcha.captcha_en import CaptchaEn
            return CaptchaEn(self.dic_config, dic_engine)

        return None

    def get_engine_object_detection(self, name, dic_engine={}):
        if 'yolo3' == name:
            from .object_detection.yolo3 import Yolo3
            return Yolo3(self.dic_config, dic_engine)

        if 'ssd' == name:
            from .object_detection.ssd import Ssd
            return Ssd(self.dic_config, dic_engine)

        return None

    def get_engine(self, name, dic_engine={}):
        if 'classify' == self.dic_config['name']:
            return self.get_engine_classify(name, dic_engine)

        if 'regression' == self.dic_config['name']:
            return self.get_engine_regression(name, dic_engine)

        if 'seq2seq' == self.dic_config['name']:
            return self.get_engine_seq2seq(name, dic_engine)

        if 'qa' == self.dic_config['name']:
            return self.get_engine_qa(name, dic_engine)

        if 'gan' == self.dic_config['name']:
            return self.get_engine_gan(name, dic_engine)

        if 'ocr' == self.dic_config['name']:
            return self.get_engine_ocr(name, dic_engine)

        if 'calculation' == self.dic_config['name']:
            return self.get_engine_calculation(name, dic_engine)

        if 'captcha' == self.dic_config['name']:
            return self.get_engine_captcha(name, dic_engine)

        if 'object_detection' == self.dic_config['name']:
            return self.get_engine_object_detection(name, dic_engine)

        if 'similarity' == self.dic_config['name']:
            return self.get_engine_similarity(name, dic_engine)

        return None
        # if 'alexnet' == name:
        #     from .base_cnn.alexnet import AlexNet
        #     self.engine = AlexNet(self.dic_config, dic_engine)
        #
        # if 'googlenet' == name:
        #     from .base_cnn.googlenet import Googlenet
        #     self.engine = Googlenet(self.dic_config, dic_engine)
        #
        # if 'mobilenet' == name:
        #     from .base_cnn.mobilenet import MobileNet
        #     self.engine = MobileNet(self.dic_config, dic_engine)
        #
        # if 'mobilenetV2' == name:
        #     from .base_cnn.mobilenetV2 import MobileNetV2
        #     self.engine = MobileNetV2(self.dic_config, dic_engine)
        #
        # if 'resnet34' == name:
        #     from .base_cnn.resnet34 import ResNet34
        #     self.engine = ResNet34(self.dic_config, dic_engine)
        #
        # if 'resnet50' == name:
        #     from .base_cnn.resnet50 import ResNet50
        #     self.engine = ResNet50(self.dic_config, dic_engine)
        #
        # if 'vgg13' == name:
        #     from .base_cnn.vgg13 import VGG13
        #     self.engine = VGG13(self.dic_config, dic_engine)
        #
        # if 'vgg16' == name:
        #     from .base_cnn.vgg16 import VGG16
        #     self.engine = VGG16(self.dic_config, dic_engine)
        #
        # if 'zfnet' == name:
        #     from .base_cnn.zfnet import Zfnet
        #     self.engine = Zfnet(self.dic_config, dic_engine)
        #
        # if 'svm_classifier' == name:
        #     from .svm_classifier import SvmClssifier
        #     self.engine = SvmClssifier(self.dic_config, dic_engine)
        #
        # if 'gru' == name:
        #     from .gru import GRU
        #     self.engine = GRU(self.dic_config, dic_engine)
        #
        # if 'slearn_logistic' == name:
        #     from .sklearn_logistic import Logistic
        #     self.engine = Logistic(self.dic_config, dic_engine)
        #
        # if 'xgb_ranker' == name:
        #     from .xgb_ranker import XgbRanker
        #     self.engine = XgbRanker(self.dic_config, dic_engine)
        #
        # if 'word2vector' == name:
        #     from .word2vector import Word2Vector
        #     self.engine = Word2Vector(self.dic_config, dic_engine)
        #
        # if 'bi_lstm' == name:
        #     from .bi_lstm import BiLSTM
        #     self.engine = BiLSTM(self.dic_config, dic_engine)

        # elif model_name == 'logistic':
        #     import logistic
        #     self.model = logistic.Logistic(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'softmax':
        #     import softmax
        #     self.model = softmax.Softmax(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'softmax_keras':
        #     import softmax_keras
        #     self.model = softmax_keras.Softmax_keras(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'ai_xgboost':
        #     import ai_xgboost
        #     self.model = ai_xgboost.Ai_xgboost(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'cnn_keras':
        #     import cnn_keras
        #     self.model = cnn_keras.Cnn_keras(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'captcha':
        #     import captcha
        #     self.model = captcha.Captcha(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'ai_face':
        #     import ai_face
        #     self.model = ai_face.Ai_face(self.dic_config[model_name], dic_engine)
        #
        # elif model_name == 'ai_lstm':
        #     import ai_lstm
        #     self.model = ai_lstm.Ai_lstm(self.dic_config[model_name], dic_engine)

    def run(self):
        self.logger.info('begin train')
        for task_name in self.dic_train.get('task'):
            self.logger.info(task_name)
            engine_name = self.dic_train[task_name].get('engine', task_name)
            engine = self.get_engine(engine_name, self.dic_train[task_name])
            if engine:
                engine.run()
        self.logger.info('end train')
