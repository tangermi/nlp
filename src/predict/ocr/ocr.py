# -*- coding:utf-8 -*-
from src.predict.base import Base
from .data import preproc as pp
from .data.generator import DataGenerator
import datetime
from matplotlib import pyplot as plt
import string
import os


class OCR(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.source_path = self.dic_engine['_in']
        self.weight_path = self.dic_engine['_train']

        self.hyperparams = self.dic_engine['hyperparams']

        self.out_path = self.dic_engine['_out']
        self.predicted_path = os.path.join(self.dic_engine['_out'], self.dic_engine['out_txt'])

    def load(self):
        batch_size = self.hyperparams['batch_size']
        charset_base = string.printable[:95]
        max_text_length = self.hyperparams['max_text_length']

        self.dtgen = DataGenerator(source=self.source_path,
                                   batch_size=batch_size,
                                   charset=charset_base,
                                   max_text_length=max_text_length)

    def build_model(self):
        from .network.model import HTRModel
        input_size = (1024, 128, 1)
        arch = self.hyperparams['arch']

        # create and compile HTRModel
        # note: `learning_rate=None` will get architecture default value
        self.model = HTRModel(architecture=arch, input_size=input_size, vocab_size=self.dtgen.tokenizer.vocab_size)
        print(self.weight_path)
        self.model.load_checkpoint(self.weight_path)

    def predict(self):
        self.build_model()

        start_time = datetime.datetime.now()

        # predict() function will return the predicts with the probabilities
        predicts, _ = self.model.predict(x=self.dtgen.next_test_batch(),
                                    steps=self.dtgen.steps['test'],
                                    ctc_decode=True,
                                    verbose=1)

        # decode to string
        self.predicts = [self.dtgen.tokenizer.decode(x[0]) for x in predicts]

        total_time = datetime.datetime.now() - start_time
        self.logger.info(f'time spent: {total_time}')

        for i, item in enumerate(self.dtgen.dataset['test']['dt'][:10]):
            self.logger.info("=" * 10)
            plt.imshow(pp.adjust_to_see(item), interpolation='nearest')
            plt.savefig(os.path.join(self.out_path, f'img{i}.png'))
            self.logger.info(self.dtgen.dataset['test']['gt'][i])
            self.logger.info(self.predicts[i])

    def dump(self):
        # mount predict corpus file
        with open(self.predicted_path, "w") as lg:
            for pd, gt in zip(self.predicts, self.dtgen.dataset['test']['gt']):
                lg.write(f"TE_L {gt}\nTE_P {pd}\n")

    def run(self):
        self.init()
        self.load()
        self.predict()
        self.dump()
