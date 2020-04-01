# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'calculation',
    'task': [
        # 'generate',
        # 'feature',
        'train',
        # 'predict',
        # 'evaluation'
    ],

    'generate': {
        'task': ['addition_rnn'],
        'addition_rnn': {
            'out_file': 'addition_rnn.npz',
            'hyperparams': {
                'training_size': 50000,
                'digits': 3,
                'reverse': True
            }
        },
    },

    'feature': {
        'task': ['addition_rnn'],
        'depend': 'generate',
        'addition_rnn': {
            'depend': 'addition_rnn',
            'in_file': 'addition_rnn.npz',
            'train_feature': 'train_feature_class.npz',
            'test_feature': 'test_feature_class.npz',
            'hyperparams': {
                'chars': '0123456789+ ',
                'digits': 3,
            }
        },
    },

    'train': {
        'task': [
            'addition_rnn',
        ],
        'depend': 'feature',
        'addition_rnn': {
            'depend': 'addition_rnn',
            'in_train': 'train_feature_class.npz',
            'in_test': 'test_feature_class.npz',
            'model_file': 'addition_rnn.h5',
            'hyperparams': {
                'batch_size': 128,
                'epochs': 40,
                'digits': 3,
                'num_of_layers': 1,
                'chars': '0123456789+ ',
            },
        },
    },

}
