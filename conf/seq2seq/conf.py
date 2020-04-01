# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'seq2seq',
    'task': [
        # 'preprocess',
        # 'feature',
        'train',
    ],

    'preprocess': {
        'task': [
            # 'shakespeare',
            'fra'
        ],
        'shakespeare': {
            # 'in_file': 'shakespeare.txt',
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/shakespeare.txt',
            'out_file': 'shakespeare.txt',
        },
        'fra': {
            # 'in_file': 'fra.txt',
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/fra.txt',
            'out_file': 'fra_compact.npz',
        },
    },

    'feature': {
        'task': [
            # 'tg_rnn',
            'seq2seq'
        ],
        'depend': 'preprocess',
        'tg_rnn': {
            'depend': 'shakespeare',
            'in_file': 'shakespeare.txt',
            'out_data_file': 'text_as_int.npy',
            'char2idx_file': 'char2idx.npy',
            'idx2char_file': 'idx2char.npy',
        },
        'seq2seq': {
            'depend': 'fra',
            'in_file': 'fra_compact.npz',
            'out_file': 'feature_compact.npz',
        },
    },

    'train': {
        'task': [
            # 'tg_rnn',
            # 'cnn',
            'lstm'
        ],
        'depend': 'feature',
        'tg_rnn': {
            'depend': 'tg_rnn',
            'feature_file': 'text_as_int.npy',
            'char2idx_file': 'char2idx.npy',
            'idx2char_file': 'idx2char.npy',
            'model_file': 'rnn.m5',
            'checkpoint_file': 'ckpt_{epoch}',
            'embedding_dim': 256,
            'rnn_units': 1024,
        },
        'cnn': {
            'depend': 'fra',
            'in_file': 'feature_compact.npz',
            'hyperparams': {
                'batch_size': 64,
                'epochs': 10,
            },
            'out_file': 'seq2seq_cnn.h5',
        },
        'lstm': {
            'depend': 'seq2seq',
            'in_file': 'feature_compact.npz',
            'hyperparams': {
                'batch_size': 64,
                'epochs': 10,
                'latent_dim': 256,
            },
            'out_file': 'seq2seq_lstm.h5',
        },
    },
}
