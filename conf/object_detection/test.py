# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'captcha',
    'task': [
        'captcha_en',
    ],
    'captcha_en': {
        'yolo3': {
            'depend': 'coco2tfrecord',
            'out_file': 'yolo3.h5',
            'hyperparams': {
                'train_batch_size': 10,
                'batch_size': 512,
                'mode': 'train',
                'input_shape': 416,
                'max_boxes': 20,
                'num_classes': 80,
                'norm_epsilon': 1e-3,
                'norm_decay': 0.99,
                'anchors': '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326',
                'classes_path': '/apps/data/ai_nlp_testing/raw/tensorflow/coco/coco_classes.txt',
                'pre_train': True,
                'num_anchors': 9,
                'ignore_thresh': .5,
                'learning_rate': 0.001,
                'darknet53_weights_path': '/apps/data/ai_nlp_testing/third_model/coco/darknet53.weights',
                'Epoch': 50,
                'train_num': 118287,
            },
        },
    },
}
