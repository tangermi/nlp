# -*- coding: utf-8 -*-
from ..base import Base
import struct
import numpy as np
import tensorflow as tf
import glob
import os


'''
把 HWDB 数据转换为 tfrecord 格式
'''
class Hwdb(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']

        self.character_path = os.path.join(self.dic_engine['_out'], self.dic_engine['character_file'])
        self.train_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])

        # 测试数据
        self.test_data_path = ''
        self.test_path = ''
        if self.dic_engine.get('_test_in'):
            self.test_data_path = self.dic_engine.get('_test_in')
            self.test_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])

    def read(self):
        self.all_hwdb_gnt_files = glob.glob(os.path.join(self.data_path, '*.gnt'))
        self.logger.info('got all {} gnt files.'.format(len(self.all_hwdb_gnt_files)))
        self.logger.info('gathering charset...')
        charset = []
        if os.path.exists(self.character_path):
            self.logger.info('found exist characters.txt...')
            with open(self.character_path, 'r') as f:
                charset = f.readlines()
                charset = [i.strip() for i in charset]
        else:
            if 'trn' in self.data_path:
                for gnt in self.all_hwdb_gnt_files:
                    hwdb = CASIAHWDBGNT(gnt)
                    for img, tagcode in hwdb.get_data_iter():
                        try:
                            label = struct.pack('>H', tagcode).decode('gb2312')
                            label = label.replace('\x00', '')
                            charset.append(label)
                        except Exception as e:
                            continue
                charset = sorted(set(charset))
                with open(self.character_path, 'w') as f:
                    f.writelines('\n'.join(charset))

        self.charset = charset
        self.logger.info('all got {} characters.'.format(len(charset)))
        self.logger.info('{}'.format(charset[:10]))

    def dump(self):
        # tfrecord_f = os.path.basename(os.path.dirname(self.train_path)) + '.tfrecord'
        self.logger.info('tfrecord file saved into: {}'.format(self.train_path))
        i = 0
        with tf.io.TFRecordWriter(self.train_path) as tfrecord_writer:
            for gnt in self.all_hwdb_gnt_files:
                hwdb = CASIAHWDBGNT(gnt)
                for img, tagcode in hwdb.get_data_iter():
                    try:
                        # why do you need resize?
                        w = img.shape[0]
                        h = img.shape[1]
                        # img = cv2.resize(img, (64, 64))
                        label = struct.pack('>H', tagcode).decode('gb2312')
                        label = label.replace('\x00', '')
                        index = self.charset.index(label)
                        # save img, label as example
                        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[w])),
                                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[h])),
                            }))
                        tfrecord_writer.write(example.SerializeToString())
                        if i % 15000 == 0:
                            self.logger.info('solved {} examples. {}: {}'.format(i, label, index))
                        i += 1
                    except Exception as e:
                        self.logger.error(e)
                        continue
        self.logger.info('done.')

    def run(self):
        self.init()
        self.read()
        self.dump()


class CASIAHWDBGNT(object):
    """
    A .gnt file may contains many images and charactors
    """

    def __init__(self, f_p):
        self.f_p = f_p

    def get_data_iter(self):
        header_size = 10
        with open(self.f_p, 'rb') as f:
            while True:
                header = np.fromfile(f, dtype='uint8', count=header_size)
                if not header.size:
                    break
                sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
                tagcode = header[5] + (header[4] << 8)
                width = header[6] + (header[7] << 8)
                height = header[8] + (header[9] << 8)
                if header_size + width * height != sample_size:
                    break
                image = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
                yield image, tagcode
