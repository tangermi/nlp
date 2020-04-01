# -*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append(r"..")

import gflags
from utils.logs import Logger

Flags = gflags.FLAGS
gflags.DEFINE_string('config', '', '配置文件')  ## 默认开发环境
gflags.DEFINE_string('log_file', '', 'log文件')
Flags(sys.argv)


def init():
    global logger
    logger = Logger(log_path=Flags.log_file, file_handler='file').logger

    # 动态倒入模块
    import importlib
    c = importlib.import_module(Flags.config)
    global CONFIG
    CONFIG = c.CONFIG


def run():
    for task_name in CONFIG.get('task', []):
        logger.info(CONFIG[task_name])
        CONFIG['logger'] = logger

        from server.test import _test_all
        _test_all(CONFIG, task_name)


if __name__ == '__main__':
    init()
    run()
