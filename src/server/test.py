# -*- coding:utf-8 -*-


def _test_all(_config, name):
    _config['logger'].info(_config)
    _config['logger'].info(name)

    if _config['name'] == 'classify':
        from .classify.test import _test
        _test(_config, name)

    if _config['name'] == 'gan':
        from .gan.test import _test
        _test(_config, name)

    if _config['name'] == 'captcha':
        from .captcha.test import _test
        _test(_config, name)

    if _config['name'] == 'ocr':
        from .ocr.test import _test
        _test(_config, name)

    if _config['name'] == 'qa':
        from .qa.test import _test
        _test(_config, name)

    if _config['name'] == 'regression':
        from .regression.test import _test
        _test(_config, name)

    if _config['name'] == 'seq2seq':
        from .seq2seq.test import _test
        _test(_config, name)

    if _config['name'] == 'similarity':
        from .similarity.test import _test
        _test(_config, name)

    if _config['name'] == 'calculation':
        from .calculation.test import _test
        _test(_config, name)

    if _config['name'] == 'object_detection':
        from .object_detection.test import _test
        _test(_config, name)
