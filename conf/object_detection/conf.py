# -*- coding: utf-8 -*-

CONFIG = {
    'name': 'object_detection',
    'task': [
        # 'preprocess',
        # 'generate',
        'train',
        # 'predict',
        # 'evaluation'
    ],

    'preprocess': {
        'task': ['coco2tfrecord'],
        'coco2tfrecord': {
            # 数据来源
            # 训练: http://images.cocodataset.org/zips/test2017.zip
            # 测试: http://images.cocodataset.org/zips/train2017.zip
            '_in': '/apps/data/ai_nlp_testing/raw/tensorflow/coco',
            'input_shape': 416,
            'max_boxes': 20,
            'tfrecord_num': 12,
            'mode': 'train',
            'data_file': {'train': '/apps/data/ai_nlp_testing/raw/tensorflow/coco/train2017',
                          'val': '/apps/data/ai_nlp_testing/raw/tensorflow/coco/test2017'},
            'annotations_file': {'train': '/apps/data/ai_nlp_testing/raw/tensorflow/coco/annotations/instances_train2017.json',
                                 'val': '/apps/data/ai_nlp_testing/raw/tensorflow/coco/annotations/instances_val2017.json'},
            'anchors': '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326',
            'num_classes': 80,
            'classes_path': '/apps/data/ai_nlp_testing/raw/tensorflow/coco/coco_classes.txt',
        },
    },

    'train': {
        'task': [
            # 'yolo3',
            'ssd'
        ],
        'depend': 'preprocess',
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
        'ssd': {
            'depend': 'coco2tfrecord',
            'data_dir': '/apps/data/ai_nlp_testing/raw/tensorflow/VOCdevkit',
            'data_year': '2012',
            'data_arch': 'ssd300',
            'pretrained_type': 'base',
            'hyperparams': {
                'batch_size': 32,
                'num_batches': 1000,
                'neg_ratio': 3,
                'initial_lr': 1e-3,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'epochs': 120,
                'num_classes': 21,
            },
        },
    },
}
