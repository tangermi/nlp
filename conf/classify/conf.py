# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'classify',
    'task': [
        # 'preprocess',
        'feature',
        # 'train',
        # 'predict',
        # 'evaluation'
    ],

    'preprocess': {
        'task': [
            # 'sogou',
            # 'mnist',
            # 'cifar10',
            # 'fashion_mnist',
            # 'imdb'
        ],
        'sogou': {
            '_in': '/apps/data/ai_nlp_testing/raw/training_data_for_classification/sogou_text_classification_corpus_mini',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
        },
        'mnist': {
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/mnist.npz',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            # 'train_overview': 'train_overview.png',
        },
        'cifar10': {
            # 'in_file': 'data_batch_1',  # data_batch_2, data_batch_3, data_batch_4, data_batch_5, test_batch
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/cifar-10-batches-py/data_batch_1',
            '_test_in': '/apps/data/ai_nlp_testing/raw/tensorflow/cifar-10-batches-py/test_batch',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'train_overview': 'train_overview.png',
        },
        'fashion_mnist': {
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/fashion_mnist.npz',
            'out_img': 'data_relation.png',
            'mean_std_file': 'mean_std.pkl',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'train_overview': 'train_overview.png',
        },
        'imdb': {
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/imdb.npz',
            'index_in': '/apps/data/ai_nlp_testing/raw/tensorflow/imdb_word_index.json',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'index_file': 'index.npy'
        },
    },

    'feature': {
        'task': [
            # 'multinomial_nb',
            # 'xgboost',
            'fasttext'
        ],
        'depend': 'preprocess',
        'multinomial_nb': {
            'depend': 'sogou',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'stop_word_file': '/apps/data/ai_nlp_testing/dict/stopwords/stopwords_cn.txt',
            'feature_words': 'feature_words.npy',
            'train_feature': 'train_feature_class.npz',
            'test_feature': 'test_feature_class.npz',
        },
        'xgboost': {
            'depend': 'sogou',
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'vectorizer_path': 'vectorizer.pkl',
            'tfidftransformer_path': 'tfidftransformer.pkl',
            'train_feature': 'train_feature_class.npz',
            'test_feature': 'test_feature_class.npz',
            'stop_word_file': '/apps/data/ai_nlp_testing/dict/stopwords/stopwords_cn.txt',
        },
        'fasttext': {
            'depend': 'imdb',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'train_feature': 'train_feature_class.npz',
            'test_feature': 'test_feature_class.npz',
            'max_features': 'max_features.npy',
            'hyperparams': {
                'ngram_range': 2,
                'max_features': 20000,
                'maxlen': 100,
            },
        }
    },

    'train': {
        'task': [
            # 'multinomial_nb',
            # 'xgboost',
            # 'mlp',
            # 'cnn',
            # 'resnet',
            # 'nn',
            # 'bidirectional_lstm',
            'fasttext'
        ],
        'depend': 'feature',  # preprocess | feature
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
        'mlp': {
            'depend': 'mnist',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'model_file': 'mlp.h5',
            'hyperparams': {
                'epoch': 10,
                'activation': 'relu',
                'batch_size': 200,
                'verbose': 2,
            },
            'img_path_accuracy': 'history_accuracy.png',
            'img_path_loss': 'history_loss.png',
        },
        'cnn': {
            'depend': 'cifar10',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'model_file': 'cnn.h5',
            'hyperparams': {
                'epoch': 10,
            },
            'img_path_accuracy': 'history_accuracy.png',
            'img_path_loss': 'history_loss.png',
        },
        'resnet': {
            'depend': 'cifar10',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'model_file': 'resnet.h5',
            'checkpoint': 'checkpoint',
            'hyperparams': {
                'batch_size': 32,
                'epoch': 10,
                'data_augmentation': True,
                'num_classes': 10,
                'subtract_pixel_mean': True,  # Subtracting pixel mean 提高精确度
            },
            'img_path_accuracy': 'history_accuracy.png',
            'img_path_loss': 'history_loss.png',
        },
        'nn': {
            'depend': 'fashion_mnist',
            'train_file': 'train.npz',
            'weight_file': 'nn.npz',
            'hyperparams': {
                'num_classes': 10,
                'epoch': 10,
                'num_features': 784,
                'learning_rate': 0.001,
                'training_steps': 3000,
                'batch_size': 256,
                'display_step': 100
            },
        },
        'bidirectional_lstm': {
            'depend': 'imdb',
            'train_file': 'train.npz',
            'test_file': 'test.npz',
            'model_file': 'bidirectional_lstm.h5',
            'hyperparams': {
                'batch_size': 32,
                'epochs': 4,
                'max_features': 20000,
                'maxlen': 100
            },
        },
        'fasttext': {
            'depend': 'fasttext',
            'train_file': 'train_feature_class.npz',
            'test_file': 'test_feature_class.npz',
            'max_features_file': 'max_features.npy',
            'model_file': 'fasttext.h5',
            'hyperparams': {
                'epochs': 1,
                'maxlen': 100,
                'embedding_dims': 50,
                'batch_size': 32,
            },
        },
    },

    'predict': {
        'task': [
            # 'multinomial_nb1',
            # 'xgboost',
            # 'mlp',
            # 'cnn',
            # 'resnet',
            # 'nn',
            'fasttext'
        ],
        'depend': 'feature',  # feature | preprocess
        'multinomial_nb1': {
            'engine': 'multinomial_nb',
            'depend': 'multinomial_nb',
            'model_depend': 'multinomial_nb',
            'model_file': 'multinomial_nb.dat',
            'in_file': 'test_feature_class.npz',
            'out_file': 'res.npy'
        },
        'multinomial_nb2': {
            'engine': 'multinomial_nb',
            'depend': 'multinomial_nb',
            'model_file': 'multinomial_nb.dat',
            'in_file': 'test_feature_class.npz',
            'out_file': 'res.npy'
        },
        'xgboost': {
            'depend': 'xgboost',
            'model_file': 'xgboost.dat',
            'in_file': 'test_feature_class.npz',
            'out_file': 'res.npy'
        },
        'mlp': {
            'depend': 'mnist',
            'model_file': 'mlp.h5',
            'in_file': 'test.npz',
            'out_file': 'res.npy'
        },
        'cnn': {
            'depend': 'cifar10',
            'model_file': 'cnn.h5',
            'in_file': 'test.npz',
            'out_file': 'res.npy'
        },
        'resnet': {
            'depend': 'cifar10',
            'model_file': 'resnet.h5',
            'in_file': 'test.npz',
            'out_file': 'res.npy'
        },
        'nn': {
            'depend': 'fashion_mnist',
            'model_file': 'nn.npz',
            'in_file': 'test.npz',
            'out_file': 'res.npy'
        },
        'fasttext': {
            'depend': 'fasttext',
            'model_file': 'fasttext.h5',
            'in_file': 'test_feature_class.npz',
            'out_file': 'res.npy'
        },
    },

    'evaluation': {
        'task': [
            # 'multinomial_nb',
            # 'xgboost'
            # 'mlp',
            # 'cnn',
            # 'resnet',
            # 'nn',
            'fasttext'
        ],
        'depend': 'predict',
        'multinomial_nb': {
            'depend': 'multinomial_nb1',
            'score': ['fs_each_class'],
            'in_file': 'res.npy',
            'classes': [0, 1, 2, 3, 4],
            'f1': {
                'average': 'macro',  # [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
            },
            # 'confuse_matrix': {
            #     'image_path': 'confuse_matrix.jpg',
            # },
            # 'jaccard': {
            #     'out_file': 'jaccard.txt',
            # },
            # 'kappa': {
            #     'out_file': 'kappa.txt',
            # },
            # 'roc_auc_multi': {
            #     'image_path': 'roc_auc.jpg',
            # },
        },

        'xgboost': {
            'depend': 'xgboost',
            'score': ['roc_auc_multiclass'],
            'in_file': 'res.npy',
            'classes': [0, 1, 2, 3, 4],
            'f1': {
                'average': 'macro',  # [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
                'out_file': 'f1_score.txt',
                'image_path': 'bayes.jpg',
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
                'out_file': 'f_score.csv',
                'image_path': 'bayes.jpg',
            },
        },
        'mlp': {
            'depend': 'mlp',
            'score': ['roc_auc_multi'],
            'in_file': 'res.npy',
            'classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'f1': {
                'average': 'macro',  # [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
                'out_file': 'f1_score.txt',
                'image_path': 'bayes.jpg',
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
                'out_file': 'f_score.csv',
                'image_path': 'bayes.jpg',
            },
        },
        'cnn': {
            'depend': 'cnn',
            'score': ['roc_auc_multi'],
            'in_file': 'res.npy',
            'classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'f1': {
                'average': 'macro',  # [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
                'out_file': 'f1_score.txt',
                'image_path': 'bayes.jpg',
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
                'out_file': 'f_score.csv',
                'image_path': 'bayes.jpg',
            },
        },
        'resnet': {
            'depend': 'resnet',
            'score': ['kappa'],  # f1, fs_each_class, confuse_matrix, jaccard, kappa, roc_auc_multi
            'in_file': 'res.npy',
            'classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'f1': {
                'average': 'macro',  # [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
                'out_file': 'f1_score.txt',
                'image_path': 'bayes.jpg',
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
                'out_file': 'f_score.csv',
                'image_path': 'bayes.jpg',
            },
        },
        'nn': {
            'depend': 'nn',
            'score': ['confuse_matrix'],
            'in_file': 'res.npy',
            'classes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            'f1': {
                'average': 'macro',  # [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
                'out_file': 'f1_score.txt',
                'image_path': 'bayes.jpg',
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
                'out_file': 'f_score.csv',
                'image_path': 'bayes.jpg',
            },
        },
        'fasttext': {   # 二分类，使用二分类的roc
            'depend': 'fasttext',
            'score': ['fs_each_class'],   # confuse_matrix, f1, fs_each_class, jaccard, kappa, roc_auc
            'in_file': 'res.npy',
            'classes': [0, 1],
            'f1': {
                'average': 'macro',  # [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
                'out_file': 'f1_score.txt',
                'image_path': 'bayes.jpg',
            },
            'fs_each_class': {
                'average': None,
                'beta': 1.0,
                'out_file': 'f_score.csv',
                'image_path': 'bayes.jpg',
            },
        },
    },
}