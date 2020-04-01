# -*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(r"..")

import gflags
from source.config import Config
from utils.logs import Logger

Flags = gflags.FLAGS
gflags.DEFINE_string('config', '', '配置文件')  ## 默认开发环境
gflags.DEFINE_string('log_file', '', 'log文件')
gflags.DEFINE_boolean('debug', False, 'whether debug')  ## 默认是调试环境
Flags(sys.argv)

g_config = {}


def init_log():
    global logger
    logger = Logger(log_path=Flags.log_file, file_handler='file').logger


def init_gpu():
    import tensorflow as tf
    # 获得当前主机上特定运算设备的列表
    # gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    # cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    # logger.info(gpus)
    # logger.info(cpus)

    # 默认情况下TensorFlow会使用其所能够使用的所有GPU。
    # tf.config.experimental.set_visible_devices(devices=gpus[2:4], device_type='GPU')

    # 设置仅在需要时申请显存空间。
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # 设置Tensorflow固定消耗GPU: 0的2GB显存。
    # tf.config.experimental.set_virtual_device_configuration(gpus[0],
    #                     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

    # 单GPU模拟多GPU环境
    # tf.config.experimental.set_virtual_device_configuration(gpus[0],
    #                     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048),
    #                      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


def init():
    init_log()
    command_setting = {}
    command_setting['debug'] = Flags.debug

    # 动态倒入模块
    import importlib
    c = importlib.import_module(Flags.config)
    CONFIG = c.CONFIG

    _config = Config()
    command_setting['logger'] = logger
    _config.init(command_setting, CONFIG)
    dic_config = _config.check()
    g_config.update(dic_config)

    # g_config.update({'GPU': True})
    # if 'GPU' in g_config and g_config['GPU']:
    #     init_gpu()


# 生成数据
def generate():
    from generate.generate import Generate
    Generate(g_config).run()


# 预处理
def preprocess():
    from preprocess.preprocess import Preprocess
    Preprocess(g_config).run()


# 特征处理
def feature():
    from feature.feature import Feature
    Feature(g_config).run()


# 训练
def train():
    from train.train import Trainer
    Trainer(g_config).run()


# 预测
def predict():
    from predict.predict import Predict
    Predict(g_config).run()


# 评估
def evaluation():
    from evaluation.evaluation import Evaluation
    Evaluation(g_config).run()


def run():
    for task_name in g_config.get('task'):
        eval(task_name)()


if __name__ == '__main__':
    init()
    run()
