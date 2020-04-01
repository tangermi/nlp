# -*- coding: utf-8 -*-
from .base import Base


class Evaluation(Base):
    def __init__(self, dic_config={}):
        self.dic_evaluation = dic_config['evaluation']
        Base.__init__(self, dic_config)

    def get_engine(self, task_name, name):
        if 'confuse_matrix' == name:
            from .classify.confuse_matrix import ConfuseMatrix
            return ConfuseMatrix(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'f1' == name:
            from .classify.f1 import F1
            return F1(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'fs_each_class' == name:
            from .classify.fs_each_class import Fs_each_class
            return Fs_each_class(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'roc_auc' == name:
            from .classify.roc_auc import RocAuc
            return RocAuc(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'roc_auc_multi' == name:
            from .classify.roc_auc_multi import RocAucMulti
            return RocAucMulti(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'jaccard' == name:
            from .classify.jaccard import Jaccard
            return Jaccard(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'kappa' == name:
            from .classify.kappa import Kappa
            return Kappa(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'mean_square_error' == name:
            from .regression.mean_square_error import Linear
            return Linear(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'siamese' == name:
            from .similarity.siamese import Siamese
            return Siamese(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'fasttext' == name:
            from .classify.fasttext import Fasttext
            return Fasttext(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        if 'captcha' == name:
            from .ocr.captcha import Captcha
            return Captcha(self.dic_config, self.dic_evaluation[task_name], self.dic_evaluation[task_name].get(name))

        return None

    def run(self):
        self.logger.info('begin evaluate')
        for task_name in self.dic_evaluation.get('task'):
            self.logger.info(task_name)
            for score_name in self.dic_evaluation[task_name].get('score'):
                engine = self.get_engine(task_name, score_name)
                if engine:
                    engine.run()
        self.logger.info('end evaluate')
