# -*- coding:utf-8 -*-


def _test_cnn(dic_config={}, logger=None):
    from .cnn import Cnn
    img_file_1 = dic_config['img_file_1']
    img_file_2 = dic_config['img_file_2']
    img_file_3 = dic_config['img_file_3']

    class_map = dic_config['class_map']

    _model = Cnn(dic_config)
    _model.load()

    y = _model._predict(img_file_1)
    logger.info(class_map.get(int(y), 'error'))
    y = _model._predict(img_file_2)
    logger.info(class_map.get(int(y), 'error'))
    y = _model._predict(img_file_3)
    logger.info(class_map.get(int(y), 'error'))


def _test_nn(dic_config={}, logger=None):
    from .nn import Nn
    img_file_1 = dic_config['img_file_1']
    img_file_2 = dic_config['img_file_2']
    img_file_3 = dic_config['img_file_3']

    class_map = dic_config['class_map']

    _model = Nn(dic_config)
    _model.load()

    y = _model._predict(img_file_1)
    logger.info(class_map.get(int(y), 'error'))
    y = _model._predict(img_file_2)
    logger.info(class_map.get(int(y), 'error'))
    y = _model._predict(img_file_3)
    logger.info(class_map.get(int(y), 'error'))


def _test_resnet(dic_config={}, logger=None):
    from .resnet import Resnet
    img_file_1 = dic_config['img_file_1']
    img_file_2 = dic_config['img_file_2']
    img_file_3 = dic_config['img_file_3']

    class_map = dic_config['class_map']

    _model = Resnet(dic_config)
    _model.load()

    y = _model._predict(img_file_1)
    logger.info(class_map.get(int(y), 'error'))
    y = _model._predict(img_file_2)
    logger.info(class_map.get(int(y), 'error'))
    y = _model._predict(img_file_3)
    logger.info(class_map.get(int(y), 'error'))


def _test_mlp(dic_config={}, logger=None):
    from .mlp import Mlp
    img_file_1 = dic_config['img_file_1']
    _model = Mlp(dic_config)
    _model.load()
    logger.info(_model._predict(img_file_1))


def _test_multinomiral_nb(dic_config={}, logger=None):
    from .multinomial_nb import _MultinomialNB
    class_map = dic_config['class_map']
    _model = _MultinomialNB(dic_config)
    _model.load()

    list_text = ['C罗获2018环球足球奖最佳球员 德尚荣膺最佳教练',
                 '英国造航母耗时8年仍未服役 被中国速度远远甩在身后',
                 '研究生考录模式亟待进一步专业化',
                 '如果真想用食物解压,建议可以食用燕麦',
                 '通用及其部分竞争对手目前正在考虑解决库存问题',
                 '归化球员救不了中国足球', ]
    for text in list_text:
        pred = _model._predict(text)[0]
        logger.info(class_map.get(int(pred), 'error'))


def _test_xgboost(dic_config={}, logger=None):
    from .xgboost import _XGBoost
    class_map = dic_config['class_map']
    _model = _XGBoost(dic_config)
    _model.load()

    list_text = ['C罗获2018环球足球奖最佳球员 德尚荣膺最佳教练',
                 '英国造航母向印度发射导弹',
                 '小学生应该接受好的教育',
                 '如果真想用食物解压,建议可以食用燕麦',
                 '由动力驱动，具有4个或4个以上车轮的非轨道承载的车辆，主要用于：载运人员',
                 '归化球员救不了中国足球', ]
    for text in list_text:
        _model.feature([text])
        pred = _model._predict()[0]
        logger.info(class_map.get(int(pred), 'error'))


def _test_bidirectional_lstm(dic_config={}, logger=None):
    from .bidirectional_lstm import BidirectionalLstm
    class_map = dic_config['class_map']
    _model = BidirectionalLstm(dic_config)
    _model.load()

    list_text = ['this is a really good one',
                 'what a horrible movie',
                 'garbage',
                 'its an excellent movie',
                 'i hate this one',
                 'i want to watch it again']
    for text in list_text:
        # _model.feature([text])
        pred = _model._predict(text)[0]
        logger.info(class_map.get(int(pred), 'error'))


def _test_fasttext(dic_config={}, logger=None):
    from .fasttext import Fasttext
    class_map = dic_config['class_map']
    _model = Fasttext(dic_config)
    _model.load()

    list_text = ['this is a really good one',
                 'what a horrible movie',
                 'garbage',
                 'its an excellent movie',
                 'i hate this one',
                 'i want to watch it again']
    for text in list_text:
        # _model.feature([text])
        pred = _model._predict(text)[0]
        logger.info(class_map.get(int(pred), 'error'))


def _test(_config, name):
    logger = _config.get('logger', None)

    if 'multinomial_nb' == name:
        _test_multinomiral_nb(_config['multinomial_nb'], logger)

    if 'xgboost' == name:
        _test_xgboost(_config['xgboost'], logger)

    if 'mlp' == name:
        _test_mlp(_config['mlp'], logger)

    if 'cnn' == name:
        _test_cnn(_config['cnn'], logger)

    if 'nn' == name:
        _test_nn(_config['nn'], logger)

    if 'resnet' == name:
        _test_resnet(_config['resnet'], logger)

    if 'bidirectional_lstm' == name:
        _test_bidirectional_lstm(_config['bidirectional_lstm'], logger)

    if 'fasttext' == name:
        _test_fasttext(_config['fasttext'], logger)
