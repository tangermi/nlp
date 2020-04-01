# -*- coding:utf-8 -*-


def _test_addition_rnn(dic_config={}, logger=None):
    from .addition_rnn import AdditionRnn

    equation1 = '222+853'
    equation2 = '1+101'
    equation3 = '10+23'

    _model = AdditionRnn(dic_config)
    _model.load()
    pred = _model._predict(equation1)
    logger.info(pred)
    pred = _model._predict(equation2)
    logger.info(pred)
    pred = _model._predict(equation3)
    logger.info(pred)

def _test(_config, name):
    logger = _config.get('logger', None)

    if 'addition_rnn' == name:
        _test_addition_rnn(_config['addition_rnn'], logger)
