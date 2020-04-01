# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'text',
    'task': [
        # 'preprocess',
        'feature',
        # 'train',
        # 'predict',
        # 'evaluation'
    ],

    'preprocess': {
        'task': ['sogou'],
        'sogou': {
            # 'in_file': 'sogou',
            '_in': '/apps/data/ai_nlp_testing/raw/training_data_for_classification/sogou_text_classification_corpus_mini',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
        },
    },

    'feature': {
        'task': [
            'multinomial_nb',
            # 'xgboost'
        ],
        'depend': 'preprocess',
        'multinomial_nb': {
            'depend': 'sogou',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'stop_word_file': '/apps/data/ai_nlp_testing/dict/stopwords/stopwords_cn.txt',
            'feature_words': 'feature_words.npy',
            'train_feature': 'train_feature.npy',
            'test_feature': 'test_feature.npy',
            'train_class': 'train_class.npy',
            'test_class': 'test_class.npy',
        },
        'xgboost': {
            'depend': 'sogou',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'train_feature': 'train_feature.npy',
            'test_feature': 'test_feature.npy',
            'train_class': 'train_class.npy',
            'test_class': 'test_class.npy',
            'stop_word_file': '/apps/data/ai_nlp_testing/dict/stopwords/stopwords_cn.txt',
        },
    },

    'train': {
        'task': [
            # 'multinomial_nb',
            'xgboost'
        ],
        'depend': 'feature',
        'multinomial_nb': {
            'depend': 'multinomial_nb',
            'train_feature': 'train_feature.npy',
            'test_feature': 'test_feature.npy',
            'train_class': 'train_class.npy',
            'test_class': 'test_class.npy',
            'model_file': 'multinomial_nb.dat',
        },
        'xgboost': {
            'depend': 'xgboost',
            'train_feature': 'train_feature.npy',
            'test_feature': 'test_feature.npy',
            'train_class': 'train_class.npy',
            'test_class': 'train_class.npy',
            'model_file': 'xgboost.dat',
        },
    },

    'predict': {
        'task': [
            # 'multinomial_nb',
            'xgboost'
        ],
        'depend': 'feature',
        'multinomial_nb': {
            'depend': 'multinomial_nb',
            'model_file': 'multinomial_nb.dat',
            'model_in': '/apps/data/ai_nlp_testing/text/train/multinomial_nb',
            'test_feature': 'test_feature.npy',
            'test_class': 'test_class.npy',
            'predicted_result': 'res.npy'
        },
        'xgboost': {
            'depend': 'xgboost',
            'model_file': 'xgboost.dat',
            'model_in': '/apps/data/ai_nlp_testing/text/train/xgboost',
            'test_feature': 'test_feature.npy',
            'test_class': 'test_class.npy',
            'predicted_result': 'res.npy'
        },
    },

    'evaluation': {
        'task': [
            'multinomial_nb',
            # 'xgboost'
        ],
        'depend': 'predict',
        'multinomial_nb': {
            'depend': 'multinomial_nb',
            'score': ['roc_auc_multiclass'],
            'in_file': 'res.npy',
            'f1': {
                'average': None,
                'out_file': 'f1_score.txt',
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
                'classes': [0, 1, 2, 3, 4],
                'out_file': 'f_score.txt',
            },
            'confuse_matrix': {
                'classes': [0, 1, 2, 3, 4],
                'image_path': 'confuse_matrix.jpg',
            },
            'jaccard': {
                'out_file': 'jaccard.txt',
            },
            'kappa': {
                'out_file': 'kappa.txt',
            },
            'roc_auc_multiclass': {
                'classes': [0, 1, 2, 3, 4],
                'image_path': 'roc_auc.jpg',
            },
        },

        'xgboost': {
            'depend': 'xgboost',
            'score': ['roc_auc_multiclass'],
            'f1': {
                'average': None,
                'out_file': 'f1_score.txt',
                'image_path': 'bayes.jpg',
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
                'classes': [0, 1, 2, 3, 4],
                'out_file': 'f_score.csv',
                'image_path': 'bayes.jpg',
            },
            'confuse_matrix': {
                'classes': [0, 1, 2, 3, 4],
                'image_path': 'confuse_matrix.jpg',
            },
            'jaccard': {
                'out_file': 'jaccard.txt',
            },
            'kappa': {
                'out_file': 'kappa.txt',
            },
            'roc_auc_multiclass': {
                'classes': [0, 1, 2, 3, 4],
                'image_path': 'roc_auc.jpg',
            },
        },
    }
}
