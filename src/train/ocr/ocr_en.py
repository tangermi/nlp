# -*- coding:utf-8 -*-
from ..base import Base
import os
import string
import datetime


# 训练模型
class OcrEn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        self.source = self.dic_engine['source']
        # self.source_path = os.path.join(self.dic_engine['_in'], self.dic_engine['in_train'])

        self.hyperparams = self.dic_engine['hyperparams']

        self.out_path = self.dic_engine['_out']
        self.checkpoint_path = os.path.join(self.dic_engine['_out'], self.dic_engine['weight_file'])
        self.train_log_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_log'])

    # 读取训练数据
    def load(self):
        from .data.generator import DataGenerator
        max_text_length = 128
        charset_base = string.printable[:95]
        batch_size = self.hyperparams['batch_size']

        self.dtgen = DataGenerator(source=self.data_path,
                              batch_size=batch_size,
                              charset=charset_base,
                              max_text_length=max_text_length)

        self.logger.info(f"Train images: {self.dtgen.size['train']}")
        self.logger.info(f"Validation images: {self.dtgen.size['valid']}")
        self.logger.info(f"Test images: {self.dtgen.size['test']}")


    def build_model(self):
        from src.utils.network.model import HTRModel
        input_size = (1024, 128, 1)
        arch = self.hyperparams['arch']

        # create and compile HTRModel
        # note: `learning_rate=None` will get architecture default value
        self.model = HTRModel(architecture=arch, input_size=input_size, vocab_size=self.dtgen.tokenizer.vocab_size)
        self.model.compile(learning_rate=0.001)

        # save network summary
        # model.summary(output_path, "summary.txt")
        self.logger.info(self.model.summary())

        # get default callbacks and load checkpoint weights file (HDF5) if exists
        # model.load_checkpoint(target=target_path)

        self.callbacks = self.model.get_callbacks(logdir=self.out_path, checkpoint=self.checkpoint_path, verbose=1)

    def train(self):
        # self.dtgen.next_train_batch()
        # exit()
        epochs = self.hyperparams['epochs']

        # to calculate total and average time per epoch
        start_time = datetime.datetime.now()

        h = self.model.fit(x=self.dtgen.next_train_batch(),
                      epochs=epochs,
                      steps_per_epoch=self.dtgen.steps['train'],
                      validation_data=self.dtgen.next_valid_batch(),
                      validation_steps=self.dtgen.steps['valid'],
                      callbacks=self.callbacks,
                      shuffle=True,
                      verbose=1)

        total_time = datetime.datetime.now() - start_time

        loss = h.history['loss']
        val_loss = h.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        time_epoch = (total_time / len(loss))
        total_item = (self.dtgen.size['train'] + self.dtgen.size['valid'])

        self.t_corpus = "\n".join([
            f"Total train images:      {self.dtgen.size['train']}",
            f"Total validation images: {self.dtgen.size['valid']}",
            f"Batch:                   {self.dtgen.batch_size}\n",
            f"Total time:              {total_time}",
            f"Time per epoch:          {time_epoch}",
            f"Time per item:           {time_epoch / total_item}\n",
            f"Total epochs:            {len(loss)}",
            f"Best epoch               {min_val_loss_i + 1}\n",
            f"Training loss:           {loss[min_val_loss_i]:.8f}",
            f"Validation loss:         {min_val_loss:.8f}"
        ])

    def dump(self):
        # save model to file
        self.model.save(self.model_path)

        # save training log
        with open(self.train_log_path, "w") as lg:
            lg.write(t_corpus)
            self.logger.info(self.t_corpus)

    def run(self):
        self.init()
        self.load()
        self.build_model()
        self.train()
        self.dump()
