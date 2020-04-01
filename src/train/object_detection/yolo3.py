# -*- coding:utf-8 -*-
from ..base import Base
import os
import time
import numpy as np
import tensorflow as tf
from .utils import draw_box, load_weights, letterbox_image, voc_ap


# 训练模型
class Yolo3(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_dir = self.dic_engine['_in']
        self.hyperparams = self.dic_engine['hyperparams']
        self.mode = self.hyperparams['mode']
        self.train_batch_size = self.hyperparams['train_batch_size']
        self.input_shape = self.hyperparams['input_shape']
        self.max_boxes = self.hyperparams['max_boxes']
        self.norm_epsilon = self.hyperparams['norm_epsilon']
        self.norm_decay = self.hyperparams['norm_decay']
        self.anchors = self.hyperparams['anchors']
        self.num_classes = self.hyperparams['num_classes']
        self.pre_train = self.hyperparams['pre_train']
        self.num_anchors = self.hyperparams['num_anchors']
        self.classes_path = self.hyperparams['classes_path']
        self.ignore_thresh = self.hyperparams['ignore_thresh']
        self.learning_rate = self.hyperparams['learning_rate']
        self.darknet53_weights_path = self.hyperparams['darknet53_weights_path']
        self.Epoch = self.hyperparams['Epoch']
        self.train_num = self.hyperparams['train_num']
        file_pattern = self.data_dir + "/*" + self.mode + '.tfrecords'
        self.TfrecordFile = tf.io.gfile.glob(file_pattern)

        self.anchors = [float(x) for x in self.anchors.split(',')]
        self.anchors = np.array(self.anchors).reshape(-1, 2)

        self.model_dir = self.dic_engine['_out']

    def Preprocess_true_boxes(self, true_boxes):
        """
        Introduction
        ------------
            对训练数据的ground truth box进行预处理
        Parameters
        ----------
            true_boxes: ground truth box 形状为[boxes, 5], x_min, y_min, x_max, y_max, class_id
        """
        num_layers = len(self.anchors) // 3
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array([self.input_shape, self.input_shape], dtype='int32')
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
        y_true = [
            np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + self.num_classes), dtype='float32')
            for l in range(num_layers)]
        # 这里扩充维度是为了后面应用广播计算每个图中所有box的anchor互相之间的iou
        anchors = np.expand_dims(self.anchors, 0)
        anchors_max = anchors / 2.
        anchors_min = -anchors_max
        # 因为之前对box做了padding, 因此需要去除全0行
        valid_mask = boxes_wh[..., 0] > 0
        wh = boxes_wh[valid_mask]
        # 为了应用广播扩充维度
        wh = np.expand_dims(wh, -2)
        # wh 的shape为[box_num, 1, 2]
        boxes_max = wh / 2.
        boxes_min = -boxes_max

        intersect_min = np.maximum(boxes_min, anchors_min)
        intersect_max = np.minimum(boxes_max, anchors_max)
        intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # 找出和ground truth box的iou最大的anchor box, 然后将对应不同比例的负责该ground turth box 的位置置为ground truth box坐标
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[t, 4].astype('int32')
                    y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[l][j, i, k, 4] = 1.
                    y_true[l][j, i, k, 5 + c] = 1.
        return y_true[0], y_true[1], y_true[2]

    def Preprocess(self, image, bbox):
        """
        Introduction
        ------------
            对图片进行预处理，增强数据集
        Parameters
        ----------
            image: tensorflow解析的图片
            bbox: 图片中对应的box坐标
        """
        image_width, image_high = tf.cast(tf.shape(input=image)[1], tf.float32), tf.cast(tf.shape(input=image)[0], tf.float32)
        input_width = tf.cast(self.input_shape, tf.float32)
        input_high = tf.cast(self.input_shape, tf.float32)
        new_high = image_high * tf.minimum(input_width / image_width, input_high / image_high)
        new_width = image_width * tf.minimum(input_width / image_width, input_high / image_high)
        # 将图片按照固定长宽比进行padding缩放
        dx = (input_width - new_width) / 2
        dy = (input_high - new_high) / 2
        image = tf.image.resize(image, [tf.cast(new_high, tf.int32), tf.cast(new_width, tf.int32)],
                                       method=tf.image.ResizeMethod.BICUBIC)
        new_image = tf.image.pad_to_bounding_box(image, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                                 tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
        image_ones = tf.ones_like(image)
        image_ones_padded = tf.image.pad_to_bounding_box(image_ones, tf.cast(dy, tf.int32), tf.cast(dx, tf.int32),
                                                         tf.cast(input_high, tf.int32), tf.cast(input_width, tf.int32))
        image_color_padded = (1 - image_ones_padded) * 128
        image = image_color_padded + new_image
        # 矫正bbox坐标
        xmin, ymin, xmax, ymax, label = tf.split(value=bbox, num_or_size_splits=5, axis=1)
        xmin = xmin * new_width / image_width + dx
        xmax = xmax * new_width / image_width + dx
        ymin = ymin * new_high / image_high + dy
        ymax = ymax * new_high / image_high + dy
        bbox = tf.concat([xmin, ymin, xmax, ymax, label], 1)
        if self.mode == 'train':
            # 随机左右翻转图片
            def _flip_left_right_boxes(boxes):
                xmin, ymin, xmax, ymax, label = tf.split(value=boxes, num_or_size_splits=5, axis=1)
                flipped_xmin = tf.subtract(input_width, xmax)
                flipped_xmax = tf.subtract(input_width, xmin)
                flipped_boxes = tf.concat([flipped_xmin, ymin, flipped_xmax, ymax, label], 1)
                return flipped_boxes

            flip_left_right = tf.greater(tf.random.uniform([], dtype=tf.float32, minval=0, maxval=1), 0.5)
            image = tf.cond(pred=flip_left_right, true_fn=lambda: tf.image.flip_left_right(image), false_fn=lambda: image)
            bbox = tf.cond(pred=flip_left_right, true_fn=lambda: _flip_left_right_boxes(bbox), false_fn=lambda: bbox)
        # 将图片归一化到0和1之间
        image = image / 255.
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
        bbox = tf.clip_by_value(bbox, clip_value_min=0, clip_value_max=input_width - 1)
        bbox = tf.cond(pred=tf.greater(tf.shape(input=bbox)[0], self.max_boxes), true_fn=lambda: bbox[:self.max_boxes],
                       false_fn=lambda: tf.pad(tensor=bbox, paddings=[[0, self.max_boxes - tf.shape(input=bbox)[0]], [0, 0]],
                                      mode='CONSTANT'))
        return image, bbox

    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord数据
        Parameters
        ----------
            serialized_example: 序列化的每条数据
        """
        features = tf.io.parse_single_example(
            serialized=serialized_example,
            features={
                'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string),
                'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
                'image/object/bbox/label': tf.io.VarLenFeature(dtype=tf.float32)
            }
        )
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, axis=0)
        ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, axis=0)
        xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, axis=0)
        ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, axis=0)
        label = tf.expand_dims(features['image/object/bbox/label'].values, axis=0)
        bbox = tf.concat(axis=0, values=[xmin, ymin, xmax, ymax, label])
        bbox = tf.transpose(a=bbox, perm=[1, 0])
        image, bbox = self.Preprocess(image, bbox)
        bbox_true_13, bbox_true_26, bbox_true_52 = tf.compat.v1.py_func(self.Preprocess_true_boxes, [bbox],
                                                              [tf.float32, tf.float32, tf.float32])
        return image, bbox, bbox_true_13, bbox_true_26, bbox_true_52

    def build_dataset(self, batch_size):
        """
        Introduction
        ------------
            建立数据集dataset
        Parameters
        ----------
            batch_size: batch大小
        Return
        ------
            dataset: 返回tensorflow的dataset
        """
        dataset = tf.data.TFRecordDataset(filenames=self.TfrecordFile)
        dataset = dataset.map(self.parser, num_parallel_calls=10)
        if self.mode == 'train':
            dataset = dataset.repeat().shuffle(9000).batch(batch_size).prefetch(batch_size)
        else:
            dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
        return dataset

    def train(self):
        train_data = self.build_dataset(self.train_batch_size)
        is_training = tf.compat.v1.placeholder(tf.bool, shape=[])
        iterator = tf.compat.v1.data.make_one_shot_iterator(train_data)
        images, bbox, bbox_true_13, bbox_true_26, bbox_true_52 = iterator.get_next()
        images.set_shape([None, self.input_shape, self.input_shape, 3])
        bbox.set_shape([None, self.max_boxes, 5])
        grid_shapes = [self.input_shape // 32, self.input_shape // 16, self.input_shape // 8]
        bbox_true_13.set_shape([None, grid_shapes[0], grid_shapes[0], 3, 5 + self.num_classes])
        bbox_true_26.set_shape([None, grid_shapes[1], grid_shapes[1], 3, 5 + self.num_classes])
        bbox_true_52.set_shape([None, grid_shapes[2], grid_shapes[2], 3, 5 + self.num_classes])
        draw_box(images, bbox)
        model = yolo(self.norm_epsilon, self.norm_decay, self.anchors, self.classes_path, self.pre_train)
        bbox_true = [bbox_true_13, bbox_true_26, bbox_true_52]
        output = model.yolo_inference(images, self.num_anchors / 3, self.num_classes, is_training)
        loss = model.yolo_loss(output, bbox_true, model.anchors, self.num_classes, self.ignore_thresh)
        l2_loss = tf.compat.v1.losses.get_regularization_loss()
        loss += l2_loss
        tf.compat.v1.summary.scalar('loss', loss)
        merged_summary = tf.compat.v1.summary.merge_all()
        global_step = tf.Variable(0, trainable=False)
        lr = tf.compat.v1.train.exponential_decay(self.learning_rate, global_step, decay_steps=2000, decay_rate=0.8)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        # 如果读取预训练权重，则冻结darknet53网络的变量
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.pre_train:
                train_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='yolo')
                train_op = optimizer.minimize(loss=loss, global_step=global_step, var_list=train_var)
            else:
                train_op = optimizer.minimize(loss=loss, global_step=global_step)
        init = tf.compat.v1.global_variables_initializer()
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=False)) as sess:
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('restore model', ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(init)
            if self.pre_train is True:
                load_ops = load_weights(tf.compat.v1.global_variables(scope='darknet53'), self.darknet53_weights_path)
                sess.run(load_ops)
            # summary_writer = tf.summary.FileWriter(config.log_dir, sess.graph)
            loss_value = 0
            for epoch in range(self.Epoch):
                for step in range(int(self.train_num / self.train_batch_size)):
                    start_time = time.time()
                    train_loss, summary, global_step_value, _ = sess.run([loss, merged_summary, global_step, train_op],
                                                                         {is_training: True})
                    loss_value += train_loss
                    duration = time.time() - start_time
                    examples_per_sec = float(duration) / self.train_batch_size
                    format_str = ('Epoch {} step {},  train loss = {} ( {} examples/sec; {} ''sec/batch)')
                    print(format_str.format(epoch, step, loss_value / global_step_value, examples_per_sec, duration))
                    summary_writer.add_summary(
                        summary=tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="train loss", simple_value=train_loss)]),
                        global_step=step)
                    summary_writer.add_summary(summary, step)
                    summary_writer.flush()
                # 每3个epoch保存一次模型
                if epoch % 3 == 0:
                    checkpoint_path = os.path.join(self.model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

    #
    # def dump(self):
    #         # 保存模型
    #         self.model.save(self.model_path)

    def run(self):
        self.init()
        # self.load()
        # self.build_model()
        self.train()
        # self.plot()
        # self.dump()


class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors, classes_path, pre_train):
        """
        Introduction
        ------------
            初始化函数
        Parameters
        ----------
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
            anchors_path: yolo anchor 文件路径
            classes_path: 数据集类别对应文件
            pre_train: 是否使用预训练darknet53模型
        """
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors = anchors
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = anchors
        # self.anchors = self._get_anchors()
        self.classes = self._get_class()

    def _get_class(self):
        """
        Introduction
        ------------
            获取类别名字
        Returns
        -------
            class_names: coco数据集类别对应的名字
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # def _get_anchors(self):
    #     """
    #     Introduction
    #     ------------
    #         获取anchors
    #     """
    #     anchors = [float(x) for x in self.anchors.split(',')]
    #     return np.array(anchors).reshape(-1, 2)

    def _batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        '''
        Introduction
        ------------
            对卷积层提取的feature map使用batch normalization
        Parameters
        ----------
            input_layer: 输入的四维tensor
            name: batchnorm层的名字
            trainging: 是否为训练过程
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            bn_layer: batch normalization处理之后的feature map
        '''
        bn_layer = tf.compat.v1.layers.batch_normalization(inputs=input_layer,
                                                 momentum=norm_decay, epsilon=norm_epsilon, center=True,
                                                 scale=True, training=training, name=name)
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)

    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        """
        Introduction
        ------------
            使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
            经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
            根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
            比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
            因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            strides: 卷积步长
            name: 卷积层名字
            trainging: 是否为训练过程
            use_bias: 是否使用偏置项
            kernel_size: 卷积核大小
        Returns
        -------
            conv: 卷积之后的feature map
        """
        conv = tf.compat.v1.layers.conv2d(
            inputs=inputs, filters=filters_num,
            kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
            padding=('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * (5e-4)),
            use_bias=use_bias, name=name
            )
        return conv

    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True, norm_decay=0.99,
                        norm_epsilon=1e-3):
        """
        Introduction
        ------------
            Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核，使用1x1是为了减少channel的维度
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            trainging: 是否为训练过程
            blocks_num: block的数量
            conv_index: 为了方便加载预训练权重，统一命名序号
            weights_dict: 加载预训练模型的权重
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            inputs: 经过残差网络处理后的结果
        """
        # 在输入feature map的长宽维度进行padding
        inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name="conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index), training=training,
                                                norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size=1, strides=1,
                                       name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def _darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        """
        Introduction
        ------------
            构建yolo3使用的darknet53网络结构
        Parameters
        ----------
            inputs: 模型输入变量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            weights_dict: 预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
            route1: 返回第26层卷积计算结果52x52x256, 供后续使用
            route2: 返回第43层卷积计算结果26x26x512, 供后续使用
            conv_index: 卷积层计数，方便在加载预训练模型时使用
        """
        with tf.compat.v1.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1,
                                      name="conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index),
                                                   training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=64, blocks_num=1,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=128, blocks_num=2,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=256, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route1 = conv
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=512, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=1024, blocks_num=4,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        return route1, route2, conv, conv_index

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.99,
                    norm_epsilon=1e-3):
        """
        Introduction
        ------------
            yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率
        Parameters
        ----------
            inputs: 输入特征
            filters_num: 卷积核数量
            out_filters: 最后输出层的卷积核数量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            route: 返回最后一层卷积的前一层结果
            conv: 返回最后一层卷积的结果
            conv_index: conv层计数
        """
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        """
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
            inputs: 模型的输入变量
            num_anchors: 每个grid cell负责检测的anchor数量
            num_classes: 类别数量
            training: 是否为训练模式
        """
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training=training,
                                                                 norm_decay=self.norm_decay,
                                                                 norm_epsilon=self.norm_epsilon)
        with tf.compat.v1.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, int(num_anchors) * (num_classes + 5),
                                                                conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize(conv2d_60,
                                                          [2 * tf.shape(input=conv2d_60)[1], 2 * tf.shape(input=conv2d_60)[1]],
                                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='upSample_0')
            route0 = tf.concat([unSample_0, conv2d_43], axis=-1, name='route_0')
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, int(num_anchors) * (num_classes + 5),
                                                                conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=128, kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize(conv2d_68,
                                                          [2 * tf.shape(input=conv2d_68)[1], 2 * tf.shape(input=conv2d_68)[1]],
                                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis=-1, name='route_1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, int(num_anchors) * (num_classes + 5), conv_index=conv_index,
                                               training=training, norm_decay=self.norm_decay,
                                               norm_epsilon=self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]

    def yolo_head(self, feats, anchors, num_classes, input_shape, training=True):
        """
        Introduction
        ------------
            根据不同大小的feature map做多尺度的检测，三种feature map大小分别为13x13x255, 26x26x255, 52x52x255
        Parameters
        ----------
            feats: 输入的特征feature map
            anchors: 针对不同大小的feature map的anchor
            num_classes: 类别的数量
            input_shape: 图像的输入大小，一般为416
            trainging: 是否训练，用来控制返回不同的值
        Returns
        -------
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(input=feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)
        # 将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化为占416的比例
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / input_shape[::-1]
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        if training == True:
            return grid, predictions, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def yolo_boxes_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        """
        Introduction
        ------------
            该函数是将box的坐标修正，除去之前按照长宽比缩放填充的部分，最后将box的坐标还原成相对原始图片的
        Parameters
        ----------
            feats: 模型输出feature map
            anchors: 模型anchors
            num_classes: 数据集类别数
            input_shape: 训练输入图片大小
            image_shape: 原始图片的大小
        """
        input_shape = tf.cast(input_shape, tf.float32)
        image_shape = tf.cast(image_shape, tf.float32)
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, num_classes, input_shape,
                                                                         training=False)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        new_shape = tf.round(image_shape * tf.reduce_min(input_tensor=input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.
        box_max = box_yx + box_hw / 2.
        boxes = tf.concat(
            [box_min[..., 0:1],
             box_min[..., 1:2],
             box_max[..., 0:1],
             box_max[..., 1:2]],
            axis=-1
        )
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        boxes = tf.reshape(boxes, [-1, 4])
        boxes_scores = box_confidence * box_class_probs
        boxes_scores = tf.reshape(boxes_scores, [-1, num_classes])
        return boxes, boxes_scores

    def box_iou(self, box1, box2):
        """
        Introduction
        ------------
            计算box tensor之间的iou
        Parameters
        ----------
            box1: shape=[grid_size, grid_size, anchors, xywh]
            box2: shape=[box_num, xywh]
        Returns
        -------
            iou:
        """
        box1 = tf.expand_dims(box1, -2)
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_mins = box1_xy - box1_wh / 2.
        box1_maxs = box1_xy + box1_wh / 2.

        box2 = tf.expand_dims(box2, 0)
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4]
        box2_mins = box2_xy - box2_wh / 2.
        box2_maxs = box2_xy + box2_wh / 2.

        intersect_mins = tf.maximum(box1_mins, box2_mins)
        intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou

    def yolo_loss(self, yolo_output, y_true, anchors, num_classes, ignore_thresh=.5):
        """
        Introduction
        ------------
            yolo模型的损失函数
        Parameters
        ----------
            yolo_output: yolo模型的输出
            y_true: 经过预处理的真实标签，shape为[batch, grid_size, grid_size, 5 + num_classes]
            anchors: yolo模型对应的anchors
            num_classes: 类别数量
            ignore_thresh: 小于该阈值的box我们认为没有物体
        Returns
        -------
            loss: 每个batch的平均损失值
            accuracy
        """
        loss = 0
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = [416.0, 416.0]
        grid_shapes = [tf.cast(tf.shape(input=yolo_output[l])[1:3], tf.float32) for l in range(3)]
        for index in range(3):
            # 只有负责预测ground truth box的grid对应的为1, 才计算相对应的loss
            # object_mask的shape为[batch_size, grid_size, grid_size, 3, 1]
            object_mask = y_true[index][..., 4:5]
            class_probs = y_true[index][..., 5:]
            anchors = np.array(anchors)
            grid, predictions, pred_xy, pred_wh = self.yolo_head(yolo_output[index], anchors[anchor_mask[index]],
                                                                 num_classes, input_shape, training=True)
            # pred_box的shape为[batch, box_num, 4]
            pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
            raw_true_xy = y_true[index][..., :2] * grid_shapes[index][::-1] - grid
            object_mask_bool = tf.cast(object_mask, dtype=tf.bool)
            raw_true_wh = tf.math.log(
                tf.compat.v1.where(tf.equal(y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1], 0),
                         tf.ones_like(y_true[index][..., 2:4]),
                         y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1]))
            # 该系数是用来调整box坐标loss的系数
            box_loss_scale = 2 - y_true[index][..., 2:3] * y_true[index][..., 3:4]
            ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

            def loop_body(internal_index, ignore_mask):
                # true_box的shape为[box_num, 4]
                true_box = tf.boolean_mask(tensor=y_true[index][internal_index, ..., 0:4],
                                           mask=object_mask_bool[internal_index, ..., 0])
                iou = self.box_iou(pred_box[internal_index], true_box)
                # 计算每个true_box对应的预测的iou最大的box
                best_iou = tf.reduce_max(input_tensor=iou, axis=-1)
                ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))
                return internal_index + 1, ignore_mask

            _, ignore_mask = tf.while_loop(
                cond=lambda internal_index, ignore_mask: internal_index < tf.shape(input=yolo_output[0])[0], body=loop_body,
                loop_vars=[0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
            # 计算四个部分的loss
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=raw_true_xy,
                                                                                             logits=predictions[...,
                                                                                                    0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                    logits=predictions[...,
                                                                                           4:5]) + (
                                          1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(
                labels=object_mask, logits=predictions[..., 4:5]) * ignore_mask
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=class_probs,
                                                                               logits=predictions[..., 5:])
            xy_loss = tf.reduce_sum(input_tensor=xy_loss) / tf.cast(tf.shape(input=yolo_output[0])[0], tf.float32)
            wh_loss = tf.reduce_sum(input_tensor=wh_loss) / tf.cast(tf.shape(input=yolo_output[0])[0], tf.float32)
            confidence_loss = tf.reduce_sum(input_tensor=confidence_loss) / tf.cast(tf.shape(input=yolo_output[0])[0], tf.float32)
            class_loss = tf.reduce_sum(input_tensor=class_loss) / tf.cast(tf.shape(input=yolo_output[0])[0], tf.float32)

            loss += xy_loss + wh_loss + confidence_loss + class_loss

        return loss