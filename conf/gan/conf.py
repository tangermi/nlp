# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'gan',
    'task': [
        # 'feature',
        'train',
    ],

    'feature': {
        'task': [
            'gan',
        ],
        'depend': 'preprocess',
        'gan': {
            '_in': '/apps/data/ai_nlp_testing/classify/preprocess/mnist',
            'depend': 'mnist',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'train_feature': 'train_feature.npz',
            'test_feature': 'test_feature.npz',
        },
    },

    'train': {
        'task': [
            'gan'
        ],
        'depend': 'feature',
        'gan': {
            'depend': 'gan',
            'in_train': 'train_feature.npz',
            'in_test': 'test_feature.npz',
            'generator_file': 'params_generator_epoch_{0:03d}000.hdf5',
            'discriminator_file': 'params_discriminator_epoch_{0:03d}000.hdf5',
            'image_file': 'plot_epoch_{0:03d}_generated000.png',
            'hist_file': 'acgan-history000.pkl',
            'hyperparams': {
                'num_classes': 10
            }
        },
    },
}
