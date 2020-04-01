# -*- coding:utf-8 -*-
import tensorflow as tf
from preprocess import MNISTLoader


class Evaluator:
    def evaluate(self):
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        data_loader = MNISTLoader()
        num_batches = int(data_loader.num_test_data // self.batch_size)
        for batch_index in range(num_batches):
            start_index, end_index = batch_index * self.batch_size, (batch_index + 1) * self.batch_size
            y_pred = self.model.predict(data_loader.test_data[start_index: end_index])
            sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index],
                                                     y_pred=y_pred)
        print("test accuracy: %f" % sparse_categorical_accuracy.result())
