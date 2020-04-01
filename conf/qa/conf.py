# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'qa',
    'task': [
        # 'preprocess',
        # 'feature',
        'train',
    ],

    'preprocess': {
        'task': ['babi'],
        'babi': {
            # 数据来源: https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/babi_tasks_1-20_v1-2.tar',
            'train_file': 'train.npy',
            'test_file': 'test.npy',
        },
    },

    'feature': {
        'task': [
            'memnn',
        ],
        'depend': 'preprocess',
        'memnn': {
            'depend': 'babi',
            'train_file': 'train.npy',
            'test_file': 'test.npy',
            'train_feature': 'train_feature.npz',
            'test_feature': 'test_feature.npz',
            'feature_compact': 'feature_compact.npz',
        },
    },

    'train': {
        'task': [
            'memnn'
        ],
        'depend': 'feature',
        'memnn': {
            'depend': 'memnn',
            'in_train': 'train_feature.npz',
            'in_test': 'test_feature.npz',
            'feature_compact': 'feature_compact.npz',
            'model_file': 'memnn.h5',
            'hyperparams': {
                'batch_size': 32,
                'epochs': 120,
            }
        },
    },
}
