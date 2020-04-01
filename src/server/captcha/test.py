# -*- coding:utf-8 -*-

def _test_captcha_en(dic_config={}, logger=None):
    import os
    import glob
    out_dir = dic_config['_out'] + '/'
    # 清空生成目录
    files = glob.glob(out_dir + r'*.png')
    for file in files:
        os.remove(file)

    from src.generate.captcha.captcha_en import CaptchaEn
    generator = CaptchaEn({'logger': logger}, dic_config)
    generator.init()

    n = 3   # 测试数量

    # 生成验证码
    for i in range(n):
        generator.generate_one(out_dir)


    ###############################################
    from .captcha_en import CaptchaEn
    # 加载模型
    _model = CaptchaEn(dic_config)
    _model.load()

    # 验证码识别
    from PIL import Image
    import numpy as np
    files = glob.glob(out_dir + r'*.png')
    for i in range(n):
        img = np.asarray(Image.open(files[i]))
        pred = _model._predict(img)
        logger.info(pred)


def _test(_config, name):
    logger = _config.get('logger', None)

    if 'captcha_en' == name:
        _test_captcha_en(_config['captcha_en'], logger)
