# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'seq2seq',
    'task': [
        # 'tg_rnn',
        'seq2seq_lstm',
        # 'seq2seq_cnn'
    ],
    'tg_rnn': {
        'char2idx_path': '/apps/data/ai_nlp_testing/sequence/feature/tg_rnn/char2idx.npy',
        'idx2char_path': '/apps/data/ai_nlp_testing/sequence/feature/tg_rnn/idx2char.npy',
        'weight_path': '/apps/data/ai_nlp_testing/sequence/train/tg_rnn',
        'embedding_dim': 256,
        'rnn_units': 1024,
    },
    'seq2seq_lstm': {
        'model_path': '/apps/data/ai_nlp_testing/sequence/train/seq2seq_lstm/seq2seq_lstm.h5',
        'feature_compact_path': '/apps/data/ai_nlp_testing/sequence/feature/seq2seq/feature_compact.npz',
    },
    'seq2seq_cnn': {
        'model_path': '/apps/data/ai_nlp_testing/sequence/train/seq2seq_cnn/seq2seq_cnn.h5',
        'feature_compact_path': '/apps/data/ai_nlp_testing/sequence/feature/seq2seq/feature_compact.npz',
    },
}
