# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'classify',
    'task': [
        # 'multinomial_nb',
        # 'xgboost',
        # 'mlp',
        # 'cnn',
        # 'resnet',
        # 'nn',
        'bidirectional_lstm'
    ],
    'multinomial_nb': {
        'model_file': '/apps/data/ai_nlp_testing/classify/train/multinomial_nb/multinomial_nb.dat',
        'feature_words_path': '/apps/data/ai_nlp_testing/text/feature/multinomial_nb/feature_words.npy',
        'class_map': {
            0: '体育',
            1: '健康',
            2: '军事',
            3: '教育',
            4: '汽车',
        }
    },
    'xgboost': {
        'model_file': '/apps/data/ai_nlp_testing/classify/train/xgboost/xgboost.dat',
        'vectorizer_path': '/apps/data/ai_nlp_testing/text/feature/xgboost/vectorizer.pkl',
        'tfidftransformer_path': '/apps/data/ai_nlp_testing/text/feature/xgboost/tfidftransformer.pkl',
        'stopwords_path': '/apps/data/ai_nlp_testing/dict/stopwords/stopwords_cn.txt',
        'class_map': {
            0: '体育',
            1: '健康',
            2: '军事',
            3: '教育',
            4: '汽车',
        }
    },
    'mlp': {
        'model_file': '/apps/data/ai_nlp_testing/classify/train/mlp/mlp.h5',
        'img_file_1': '/apps/data/ai_nlp_testing/classify/raw/hand_written_3.jpg',
        'processed_image': '/apps/data/ai_nlp_testing/image/raw/processed_1.jpg',
    },
    'cnn': {
        'model_file': '/apps/data/ai_nlp_testing/classify/train/cnn/cnn.h5',
        'img_file_1': '/apps/data/ai_nlp_testing/classify/raw/animal_cat.jpg',
        'img_file_2': '/apps/data/ai_nlp_testing/classify/raw/animal_horse.jpg',
        'img_file_3': '/apps/data/ai_nlp_testing/classify/raw/animal_bird.jpg',
        'class_map': {
            0: '飞机',
            1: '机动车',
            2: '鸟',
            3: '猫',
            4: '鹿',
            5: '狗',
            6: '青蛙',
            7: '马',
            8: '船',
            9: '卡车'
        }
    },
    'resnet': {
        'model_file': '/apps/data/ai_nlp_testing/classify/train/resnet/resnet.h5',
        'img_file_1': '/apps/data/ai_nlp_testing/classify/raw/animal_cat.jpg',
        'img_file_2': '/apps/data/ai_nlp_testing/classify/raw/animal_horse.jpg',
        'img_file_3': '/apps/data/ai_nlp_testing/classify/raw/animal_bird.jpg',
        'class_map': {
            0: '飞机',
            1: '机动车',
            2: '鸟',
            3: '猫',
            4: '鹿',
            5: '狗',
            6: '青蛙',
            7: '马',
            8: '船',
            9: '卡车'
        }
    },
    'nn': {
        'weight_path': '/apps/data/ai_nlp_testing/classify/train/nn/nn.npz',
        'img_file_1': '/apps/data/ai_nlp_testing/classify/raw/t_shirt.png',
        'img_file_2': '/apps/data/ai_nlp_testing/classify/raw/sandal.jpg',
        'img_file_3': '/apps/data/ai_nlp_testing/classify/raw/trouser.png',
        'class_map': {
            0: 'T-shirt/top',
            1: 'Trouser',
            2: 'Pullover',
            3: 'Dress',
            4: 'Coat',
            5: 'Sandal',
            6: 'Shirt',
            7: 'Sneaker',
            8: 'Bag',
            9: 'Ankle boot'
        },
    },
    'bidirectional_lstm': {
        'model_file': '/apps/data/ai_nlp_testing/classify/train/bidirectional_lstm/bidirectional_lstm.h5',
        'index_path': '/apps/data/ai_nlp_testing/classify/preprocess/imdb/index.npy',
        'class_map': {
            0: '差评',
            1: '好评',
        },
    },
    'fasttext': {
        'model_file': '/apps/data/ai_nlp_testing/classify/train/fasttext/fasttext.h5',
        'index_path': '/apps/data/ai_nlp_testing/classify/preprocess/imdb/index.npy',
        'class_map': {
            0: '差评',
            1: '好评',
        },
    },
}
