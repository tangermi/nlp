# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'captcha',
    'task': [
        # 'preprocess',
        # 'generate',
        'train',
        # 'predict',
        # 'evaluation'
    ],

    'preprocess': {
    },

    'generate': {
        'task': ['captcha_en'],
        'captcha_en': {
            'nums': 512,
            'charset': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
            'width': 160,
            'height': 60,
            'captcha_length': 4,
        },
    },

    'train': {
        'task': [
            'captcha_en',
        ],
        'depend': 'generate',
        'captcha_en': {
            'depend': 'captcha_en',
            'charset': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
                        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
            'out_file': 'captcha_en.h5',
            'hyperparams': {
                'epochs': 4,
                'batch_size': 512,
                'captcha_length': 4,
            },
        },
    },
}
