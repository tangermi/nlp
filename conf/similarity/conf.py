# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'similarity',
    'task': [
        # 'preprocess',
        # 'feature'
        'train',
        # 'evaluation'
    ],

    'preprocess': {
        'task': ['fashion_mnist'],
        'fashion_mnist': {
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/fashion_mnist.npz',
            'out_img': 'data_relation.png',
            'mean_std_file': 'mean_std.pkl',
            'train_file': 'train.npz',
            'test_file': 'test.npy',
            'train_overview': 'train_overview.png',
        },
    },

    'feature': {
        'task': [
            'siamese',
        ],
        'depend': 'preprocess',
        'siamese': {
            'depend': 'fashion_mnist',
            'in_file': 'train.npz',
            'train_file': 'train_feature.npz',
            'test_file': 'test_feature.npz',
        },
    },

    'train': {
        'task': [
            'siamese',
        ],
        'depend': 'feature',
        'siamese': {
            'depend': 'siamese',
            'train_file': 'train_feature.npz',
            'test_file': 'test_feature.npz',
            'img_path_accuracy': 'history_accuracy.png',
            'img_path_loss': 'history_loss.png',
            'model_file': 'siamese.h5',
        },
    },

    'evaluation': {
        'task': [
            'siamese',
        ],
        'depend': 'preprocess',
        'siamese': {
            'depend': 'siamese',
            'score': ['siamese'],
            'model_file': 'siamese.h5',
            'model_in': '/apps/data/ai_nlp_testing/similarity/train/siamese',
            'test': 'test.npz',
            'siamese': {
                'out_file': 'accuracy.txt',
            },
        },
    }
}
