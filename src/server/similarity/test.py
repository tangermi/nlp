# -*- coding:utf-8 -*-


def _test_siamese(dic_config={}, logger=None):
    from .siamese import Siamese
    img_file_1 = dic_config['img_file_1']
    img_file_2 = dic_config['img_file_2']
    # img_file_3 = dic_test['img_file_3']
    _model = Siamese(dic_config)
    _model.load()

    logger.info(_model._predict(img_file_1, img_file_2))


def test_similarity(_config, name):
    logger = _config.get('logger', None)

    if 'siamese' == name:
        _test_siamese(_config['siamese'], logger)
