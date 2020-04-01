# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'captcha',
    'task': [
        'captcha_en',
    ],
    'captcha_en': {
        'model_file': '/apps/data/ai_nlp_testing/ocr/train/captcha/captcha.h5',
        'charset': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                    'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],

        # 以下参数生成用
        '_out': '/apps/data/ai_nlp_testing/captcha/predict_one',
        'nums': 512,
        'width': 160,
        'height': 60,
        'captcha_length': 4,
    },
}
