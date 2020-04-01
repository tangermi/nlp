# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'ocr',
    'task': [
        # 'preprocess',
        # 'train',
        'predict',
    ],

    'preprocess': {
        'task': ['hwdb'],
        'hwdb': {
            # 手写体识别
            # 训练: http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
            # 测试: http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
            '_in': '/apps/data/ai_nlp_testing/raw/hwdb/HWDB1.1tst_gnt',
            'character_file': 'characters.txt',
            'train_file': 'test.tfrecord'
        },
    },

    'train': {
        'task': [
            'ocr_en',
            # 'ocr_cn',
        ],
        'depend': 'preprocess',
        'ocr_en': {
            'depend': 'ocr_en',
            'in_file': 'bentham.hdf5',
            'source': 'bentham',
            'model_file': 'ocr.h5',
            'weight_file': 'checkpoint_weights_{}.hdf5',
            'train_log': 'train_log.txt',
            'hyperparams': {
                'batch_size': 16,
                'epochs': 4,
                'arch': 'flor',
            },
        },
        'ocr_cn': {
            'depend': 'hwdb',
            'train_file': 'train.tfrecord',
            'val_file': 'test.tfrecord',
            'character_file': 'characters.txt',
            'model_file': 'ocr_cn.h5',
            'checkpoint': 'ocr_cn-{epoch}.ckpt',
            'hyperparams': {
                'target_size': 64,
                'keras': True,
                'epochs': 1,
                'steps_per_epoch': 10,
                'validation_steps': 100,
                'times': 1,
                'batch_size': 512,
            },
        },
    },

    'predict': {
        'task': [
            'ocr_en'  # todo
        ],
        'depend': 'preprocess',
        'ocr_en': {
            'depend': 'ocr_en',
            'in_file': 'bentham.hdf5',
            'model_file': 'checkpoint_weights.hdf5',
            'out_txt': 'predict.txt',
            'hyperparams': {
                'batch_size': 16,
                'arch': 'flor',
                'max_text_length': 128,
            },
        },
    },
}
