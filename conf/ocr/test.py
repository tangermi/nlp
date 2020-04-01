# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'ocr',
    'task': [
        'ocr_en',
        # 'ocr_cn',
    ],
    'ocr_cn': {
        'model_path': '/apps/data/ai_nlp_testing/ocr/train/ocr_cn/ocr_cn.h5',
        'character_path': '/apps/data/ai_nlp_testing/ocr/preprocess/hwdb/characters.txt',
        'handwritten_1': '/apps/data/ai_nlp_testing/ocr/predict_one/ocr_cn/ao.png',
        'handwritten_2': '/apps/data/ai_nlp_testing/ocr/predict_one/ocr_cn/bang.png',
        'handwritten_3': '/apps/data/ai_nlp_testing/ocr/predict_one/ocr_cn/ben.png',
        'handwritten_4': '/apps/data/ai_nlp_testing/ocr/predict_one/ocr_cn/a.png',
        'handwritten_5': '/apps/data/ai_nlp_testing/ocr/predict_one/ocr_cn/feng.png',
    },
    'ocr_en': {
            'weight_path': '/apps/data/ai_nlp_testing/ocr/train/ocr_en/checkpoint_weights.hdf5',
            'handwritten_1': '/apps/data/ai_nlp_testing/ocr/predict_one/ocr_en/1_0_original.png',
            'hyperparams': {
                'arch': 'flor',
            },
        },
}
