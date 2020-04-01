# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'regression',
    'task': [
        # 'preprocess',
        # 'train',
        'evaluation'
    ],

    'preprocess': {
        'task': ['mpg'],
        'mpg': {
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/auto-mpg.data',
            'train_file': 'mpg_train.npz',
            'test_file': 'mpg_test.npz',
            'mean_std_file': 'mpg_mean_std.npz',
            'out_img': 'data_relation.png'
        },
    },

    'train': {
        'task': [
            'linear',
        ],
        'depend': 'preprocess',
        'linear': {
            'depend': 'mpg',
            'in_file': 'mpg_train.npz',
            'hyperparameter': {
                'epochs': 1000,
                'activation': 'relu',
                'learning_rate': 0.001,
            },
            'out_img_mse': 'history_mse.png',
            'out_img_mae': 'history_mae.png',
            'model_file': 'linear.h5',
        },
    },

    'evaluation': {
        'task': [
            'mpg',
        ],
        'depend': 'preprocess',
        'mpg': {
            'depend': 'mpg',
            'score': ['mean_square_error'],
            'model_file': 'linear.h5',
            'model_in': '/apps/data/ai_nlp_testing/regression/train/linear',
            'in_file': 'mpg_test.npz',
            'mean_square_error': {
                'out_img': 'evaluate.png',
                'out_file': 'mean_square_error.txt',
            },
        },
    }
}
