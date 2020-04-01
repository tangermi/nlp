# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt


def _test_gan(dic_config={}, logger=None):
    from .gan import Gan

    _model = Gan(dic_config)
    _model.load()

    plt.imshow(_model._predict(8), cmap='gray_r', interpolation='nearest')
    _ = plt.axis('off')
    plt.savefig('/apps/dev/ai_nlp_testing/src/predict_one/gan/8.png')   # gan生成图片的输出路径


def _test(_config, name):
    logger = _config.get('logger', None)

    if 'gan' == name:
        _test_gan(_config['captcha_en'], logger)