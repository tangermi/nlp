# -*- coding: utf-8 -*-


class Base(object):

    def __init__(self, dic_config={}):
        self.logger = dic_config.get('logger', None)
        self.dic_config = dic_config

    def init(self):
        pass

    def load(self):
        pass

    def process(self):
        pass

    def dump(self):
        pass

    def run(self):
        pass
