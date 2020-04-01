# -*- coding:utf-8 -*-


def _test_ocr_cn(dic_config={}, logger=None):
    from .ocr_cn import OCRCN

    handwritten_1 = dic_config['handwritten_1']
    handwritten_2 = dic_config['handwritten_2']
    handwritten_3 = dic_config['handwritten_3']
    handwritten_4 = dic_config['handwritten_4']
    handwritten_5 = dic_config['handwritten_5']

    _model = OCRCN(dic_config)
    _model.load()
    pred = _model._predict(handwritten_1)
    logger.info(pred)
    pred = _model._predict(handwritten_2)
    logger.info(pred)
    pred = _model._predict(handwritten_3)
    logger.info(pred)
    pred = _model._predict(handwritten_4)
    logger.info(pred)
    pred = _model._predict(handwritten_5)
    logger.info(pred)

def _test_ocr_en(dic_config={}, logger=None):
    from .ocr_en import OcrEn

    handwritten_1 = dic_config['handwritten_1']

    _model = OcrEn(dic_config)
    # _model.load()
    pred = _model._predict(handwritten_1)
    logger.info(pred)


def _test(_config, name):
    logger = _config.get('logger', None)

    if 'ocr_cn' == name:
        _test_ocr_cn(_config['ocr_cn'], logger)

    if 'ocr_en' == name:
        _test_ocr_en(_config['ocr_en'], logger)
