# -*- coding: utf-8 -*-
from .base import Base


class Generate(Base):
    def __init__(self, dic_config={}):
        self.dic_generate = dic_config['generate']
        Base.__init__(self, dic_config)

    def get_engine(self, name, dic_engine={}):
        if 'captcha_en' == name:
            from .captcha.captcha_en import CaptchaEn
            return CaptchaEn(self.dic_config, dic_engine)

        ###############################################
        if 'addition_rnn' == name:
            from .calculation.addition_rnn import AdditionRnn
            return AdditionRnn(self.dic_config, dic_engine)

    def run(self):
        self.logger.info('begin generate')
        for task_name in self.dic_generate.get('task'):
            self.logger.info(task_name)
            engine_name = self.dic_generate[task_name].get('engine', task_name)
            engine = self.get_engine(engine_name, self.dic_generate[task_name])
            if engine:
                engine.run()
        self.logger.info('end generate')
