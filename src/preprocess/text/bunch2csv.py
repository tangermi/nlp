# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from ..base import Base
from utils.tools import read_bunch


"""
input: bunch file
output: csv file
"""
class Bunch2csv(Base):

    """bunch to csv
    """
    def __init__(self, dic_config):
        Base.__init__(self, dic_config)

    def load_data(self):
        bunch = read_bunch(self.dic_config['raw_data'])
        self.x = np.array(bunch.content)
        self.y = np.array(bunch.label).reshape(-1, 10)
        # self.logger.info(self.x.shape)
        # self.logger.info(self.y.shape)

        dataset = np.hstack((self.y, self.x))
        x_list = ['dim' + str(i) for i in range(0, self.x.shape[1])]
        y_list = ['label' + str(i) for i in range(0, self.y.shape[1])]
        self.df = pd.DataFrame(dataset, columns=y_list + x_list)
        self.logger.info(self.df)

    def create(self):
        self.df.to_csv(self.dic_config['csv_path'], index=False)

        self.logger.info('————encode successful————')