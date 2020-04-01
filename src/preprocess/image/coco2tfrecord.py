# -*- coding: utf-8 -*-
from ..base import Base
import os
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict


'''

'''
class Coco2Tfrecord(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_dir = self.dic_engine['_in']
        self.input_shape = self.dic_engine['input_shape']
        self.max_boxes = self.dic_engine['max_boxes']
        self.mode = self.dic_engine['mode']
        self.data_file = self.dic_engine['data_file']
        self.annotations_file = self.dic_engine['annotations_file']
        self.anchors = self.dic_engine['anchors']
        self.num_classes = self.dic_engine['num_classes']
        self.out_path = self.dic_engine['_out']
        self.tfrecord_num = self.dic_engine['tfrecord_num']
        self.classes_path = self.dic_engine['classes_path']

    def _get_class(self, classes_path):
        """
        Introduction
        ------------
            获取类别名字
        Returns
        -------
            class_names: coco数据集类别对应的名字
        """
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def read_annotations(self):
        """
        Introduction
        ------------
            读取COCO数据集图片路径和对应的标注
        Parameters
        ----------
            data_file: 文件路径
        """
        image_data = []
        boxes_data = []
        name_box_id = defaultdict(list)
        with open(self.annotations_file[self.mode], encoding='utf-8') as file:
            data = json.load(file)
            annotations = data['annotations']
            for ant in annotations:
                id = ant['image_id']
                name = os.path.join(self.data_file[self.mode], '%012d.jpg' % id)
                cat = ant['category_id']
                if cat >= 1 and cat <= 11:
                    cat = cat - 1
                elif cat >= 13 and cat <= 25:
                    cat = cat - 2
                elif cat >= 27 and cat <= 28:
                    cat = cat - 3
                elif cat >= 31 and cat <= 44:
                    cat = cat - 5
                elif cat >= 46 and cat <= 65:
                    cat = cat - 6
                elif cat == 67:
                    cat = cat - 7
                elif cat == 70:
                    cat = cat - 9
                elif cat >= 72 and cat <= 82:
                    cat = cat - 10
                elif cat >= 84 and cat <= 90:
                    cat = cat - 11
                name_box_id[name].append([ant['bbox'], cat])

            for key in name_box_id.keys():
                boxes = []
                image_data.append(key)
                box_infos = name_box_id[key]
                for info in box_infos:
                    x_min = info[0][0]
                    y_min = info[0][1]
                    x_max = x_min + info[0][2]
                    y_max = y_min + info[0][3]
                    boxes.append(np.array([x_min, y_min, x_max, y_max, info[1]]))
                boxes_data.append(np.array(boxes))

        return image_data, boxes_data

    def convert_to_tfrecord(self, tfrecord_path, num_tfrecords):
        """
        Introduction
        ------------
            将图片和boxes数据存储为tfRecord
        Parameters
        ----------
            tfrecord_path: tfrecord文件存储路径
            num_tfrecords: 分成多少个tfrecord
        """
        image_data, boxes_data = self.read_annotations()
        images_num = int(len(image_data) / num_tfrecords)
        for index_records in range(num_tfrecords):
            output_file = os.path.join(tfrecord_path, str(index_records) + '_' + self.mode + '.tfrecords')
            with tf.io.TFRecordWriter(output_file) as record_writer:
                for index in range(index_records * images_num, (index_records + 1) * images_num):
                    with tf.io.gfile.GFile(image_data[index], 'rb') as file:
                        image = file.read()
                        xmin, xmax, ymin, ymax, label = [], [], [], [], []
                        for box in boxes_data[index]:
                            xmin.append(box[0])
                            ymin.append(box[1])
                            xmax.append(box[2])
                            ymax.append(box[3])
                            label.append(box[4])
                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                                'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
                                'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
                                'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
                                'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
                                'image/object/bbox/label': tf.train.Feature(float_list=tf.train.FloatList(value=label)),
                            }
                        ))
                        record_writer.write(example.SerializeToString())
                        if index % 1000 == 0:
                            print('Processed {} of {} images'.format(index + 1, len(image_data)))

    def read(self):
        self.anchors = [float(x) for x in self.anchors.split(',')]
        self.anchors = np.array(self.anchors).reshape(-1, 2)
        self.class_names = self._get_class(self.classes_path)

    def dump(self):
        file_pattern = self.out_path + "/*" + self.mode + '.tfrecords'
        self.TfrecordFile = tf.io.gfile.glob(file_pattern)
        if len(self.TfrecordFile) == 0:
            self.convert_to_tfrecord(self.out_path, self.tfrecord_num)

    def run(self):
        self.init()
        self.read()
        self.dump()



