# -*- coding:utf-8 -*-
import numpy as np


def propecess_mpg(dataset):
    if dataset[-1] == 1:
        del dataset[-1]
        dataset += [1, 0, 0]
    elif dataset[-1] == 2:
        del dataset[-1]
        dataset += [0, 1, 0]
    elif dataset[-1] == 3:
        del dataset[-1]
        dataset += [0, 0, 1]
    dataset = np.array([dataset]).reshape((1, -1))
    return dataset


def _test_linear(dic_config={}, logger=None):
    from .linear import Linear
    _model = Linear(dic_config)
    _model.load()
    # 气缸, 排量, 马力, 重量, 加速度, 年份, 产地
    test_data = [
        [6, 250.0, 105.0, 3459., 16.0, 75, 1],
        [6, 250.0, 72.00, 3432., 21.0, 75, 1],
        [4, 97.00, 88.00, 2130., 14.5, 70, 3],
        [8, 400.0, 170.0, 4668., 11.5, 75, 1],
        [8, 350.0, 145.0, 4440., 14.0, 75, 1],
        [4, 104.0, 95.00, 2375., 17.5, 70, 2]
    ]

    for test in test_data:
        test = propecess_mpg(test)
        logger.info(_model._predict(test))


def _test(_config, name):
    logger = _config.get('logger', None)

    if 'linear' == name:
        _test_linear(_config['linear'], logger)
