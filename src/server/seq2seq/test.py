# -*- coding:utf-8 -*-


def _test_tg_rnn(dic_config={}, logger=None):
    from .tg_rnn import Rnn
    _model = Rnn(dic_config)
    _model.load()

    y = _model.generate_text(u'ROMEO: ')
    logger.info(y)


def _test_lstm(dic_config={}, logger=None):
    from .lstm import Lstm
    _model = Lstm(dic_config)
    _model.load()

    y = _model.translate(u'Hello!')
    logger.info(y)


def _test_cnn(dic_config={}, logger=None):
    from .cnn import Cnn
    _model = Cnn(dic_config)
    _model.load()

    y = _model.translate(u'Hello!')
    logger.info(y)


def _test(_config, name):
    logger = _config.get('logger', None)

    if 'tg_rnn' == name:
        _test_tg_rnn(_config['tg_rnn'], logger)

    if 'seq2seq_lstm' == name:
        _test_lstm(_config['seq2seq_lstm'], logger)

    if 'seq2seq_cnn' == name:
        _test_cnn(_config['seq2seq_cnn'], logger)

