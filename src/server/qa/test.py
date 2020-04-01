# -*- coding:utf-8 -*-


def _test_memnn(dic_config={}, logger=None):
    from .memnn import Memnn
    _model = Memnn(dic_config)
    _model.load()

    # 输入一个带有逻辑关系的文档，和一个问题，模型会输出答案.
    y = _model.get_answer('Daniel went back to the hallway. Mary moved to the office.', 'where is Daniel')
    logger.info(y)


def _test(_config, name):
    logger = _config.get('logger', None)

    if 'memnn' == name:
        _test_memnn(_config['memnn'], logger)
