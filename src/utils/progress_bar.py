# -*- coding: utf-8 -*-
from tensorflow import keras
import sys


class ProgressBar(keras.callbacks.Callback):
    def __init__(self, epoch=1000):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs):
        epochs = self.epoch
        # 显示进度条
        self.draw_progress_bar(epoch + 1, epochs)

    def draw_progress_bar(self, cur, total, bar_len=50):
        cur_len = int(cur / total * bar_len)
        sys.stdout.write("\r")
        sys.stdout.write("[{:<{}}] {}/{}".format("=" * cur_len, bar_len, cur, total))
        sys.stdout.flush()
