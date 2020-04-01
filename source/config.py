# -*- coding:utf-8 -*-
import os
from conf.base import CONFIG_BASE


class Config:
    dic_config = {}
    name = ''

    def init(self, dic_command, dic_config):
        self.logger = dic_command.get('logger', None)
        self.dic_config.update(CONFIG_BASE)
        self.dic_config.update(dic_config)
        self.dic_config.update(dic_command)

        if 'name' in self.dic_config:
            self.name = self.dic_config['name']

    def set_data_dir(self):
        if 'DATA_DIR' not in self.dic_config:
            return

        dic_data_dir = self.dic_config['DATA_DIR']
        if self.name:
            dic_data_dir['ROOT_DIR'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], self.name)

        dic_data_dir['raw'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['raw'])
        dic_data_dir['generate'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['generate'])
        dic_data_dir['preprocess'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['preprocess'])
        dic_data_dir['feature'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['feature'])
        dic_data_dir['train'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['train'])
        dic_data_dir['predict'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['predict'])
        dic_data_dir['evaluation'] = '%s/%s' % (dic_data_dir['ROOT_DIR'], dic_data_dir['evaluation'])

        if not os.path.exists(dic_data_dir['raw']):
            os.makedirs(dic_data_dir['raw'])

        if not os.path.exists(dic_data_dir['generate']):
            os.makedirs(dic_data_dir['generate'])

        if not os.path.exists(dic_data_dir['preprocess']):
            os.makedirs(dic_data_dir['preprocess'])

        if not os.path.exists(dic_data_dir['feature']):
            os.makedirs(dic_data_dir['feature'])

        if not os.path.exists(dic_data_dir['train']):
            os.makedirs(dic_data_dir['train'])

        if not os.path.exists(dic_data_dir['predict']):
            os.makedirs(dic_data_dir['predict'])

        if not os.path.exists(dic_data_dir['evaluation']):
            os.makedirs(dic_data_dir['evaluation'])

    def generate(self):
        dic_generate = self.dic_config['generate']

        if 'task' not in dic_generate:
            self.logger.warning('no set task')
            return False

        no_task = False
        if 'no_task' in dic_generate:
            no_task = dic_generate['no_task']

        for task_name in dic_generate['task']:
            if task_name not in dic_generate['task']:
                self.logger.error('no set task')
                return False

            if not dic_generate[task_name].get('_in'):
                if 'in' not in dic_generate[task_name]:
                    dic_generate[task_name]['in'] = self.dic_config['DATA_DIR']['raw']
                dic_generate[task_name]['_in'] = dic_generate[task_name]['in']

                if dic_generate[task_name].get('in_file'):
                    dic_generate[task_name]['_in'] = '%s/%s' % (dic_generate[task_name]['in'], dic_generate[task_name].get('in_file'))

            if not dic_generate[task_name].get('_out'):
                if 'out' not in dic_generate[task_name]:
                    if no_task:
                        dic_generate[task_name]['out'] = self.dic_config['DATA_DIR']['generate']
                    else:
                        dic_generate[task_name]['out'] = '%s/%s' % (self.dic_config['DATA_DIR']['generate'], task_name)
                dic_generate[task_name]['_out'] = dic_generate[task_name]['out']

            if not os.path.exists(dic_generate[task_name]['_out']):
                os.makedirs(dic_generate[task_name]['_out'])

            if dic_generate[task_name].get('out_file'):
                dic_generate[task_name]['_out'] = '%s/%s' % (dic_generate[task_name]['out'], dic_generate[task_name].get('out_file'))

        return True

    def preprocess(self):
        dic_preprocess = self.dic_config['preprocess']

        if 'task' not in dic_preprocess:
            self.logger.warning('no set task')
            return False

        no_task = False
        if 'no_task' in dic_preprocess:
            no_task = dic_preprocess['no_task']

        depend = 'raw'
        if 'depend' in dic_preprocess:
            depend = dic_preprocess['depend']

        for task_name in dic_preprocess['task']:
            if task_name not in dic_preprocess['task']:
                self.logger.error('no set task')
                return False

            if not dic_preprocess[task_name].get('_in'):
                if 'in' not in dic_preprocess[task_name]:
                    if dic_preprocess[task_name].gett('depend'):
                        dic_preprocess[task_name]['in'] = '%s/%s' % (self.dic_config['DATA_DIR'][depend], dic_preprocess[task_name]['depend'])
                    else:
                        dic_preprocess[task_name]['in'] = self.dic_config['DATA_DIR'][depend]
                dic_preprocess[task_name]['_in'] = dic_preprocess[task_name]['in']

                if dic_preprocess[task_name].get('in_file'):
                    dic_preprocess[task_name]['_in'] = '%s/%s' % (dic_preprocess[task_name]['in'], dic_preprocess[task_name].get('in_file'))

            if not dic_preprocess[task_name].get('_out'):
                if 'out' not in dic_preprocess[task_name]:
                    if no_task:
                        dic_preprocess[task_name]['out'] = self.dic_config['DATA_DIR']['preprocess']
                    else:
                        dic_preprocess[task_name]['out'] = '%s/%s' % (self.dic_config['DATA_DIR']['preprocess'], task_name)
                dic_preprocess[task_name]['_out'] = dic_preprocess[task_name]['out']

            if not os.path.exists(dic_preprocess[task_name]['_out']):
                os.makedirs(dic_preprocess[task_name]['_out'])

            if dic_preprocess[task_name].get('out_file'):
                dic_preprocess[task_name]['_out'] = '%s/%s' % (dic_preprocess[task_name]['out'], dic_preprocess[task_name].get('out_file'))

        return True

    def feature(self):
        dic_feature = self.dic_config['feature']

        if 'task' not in dic_feature:
            self.logger.warning('no set task')
            return False

        no_task = False
        if 'no_task' in dic_feature:
            no_task = dic_feature['no_task']

        depend = 'raw'
        if 'depend' in dic_feature:
            depend = dic_feature['depend']

        for task_name in dic_feature['task']:
            if task_name not in dic_feature['task']:
                self.logger.error('no set task')
                return False

            if not dic_feature[task_name].get('_in'):
                if 'in' not in dic_feature[task_name]:
                    if dic_feature[task_name].get('depend'):
                        dic_feature[task_name]['in'] = '%s/%s' % (self.dic_config['DATA_DIR'][depend], dic_feature[task_name]['depend'])
                    else:
                        dic_feature[task_name]['in'] = self.dic_config['DATA_DIR'][depend]
                dic_feature[task_name]['_in'] = dic_feature[task_name]['in']

                if dic_feature[task_name].get('in_file'):
                    dic_feature[task_name]['_in'] = '%s/%s' % (dic_feature[task_name]['in'], dic_feature[task_name].get('in_file'))

            if not dic_feature[task_name].get('_out'):
                if 'out' not in dic_feature[task_name]:
                    if no_task:
                        dic_feature[task_name]['out'] = self.dic_config['DATA_DIR']['feature']
                    else:
                        dic_feature[task_name]['out'] = '%s/%s' % (self.dic_config['DATA_DIR']['feature'], task_name)
                dic_feature[task_name]['_out'] = dic_feature[task_name]['out']

            if not os.path.exists(dic_feature[task_name]['_out']):
                os.makedirs(dic_feature[task_name]['_out'])

            if dic_feature[task_name].get('out_file'):
                dic_feature[task_name]['_out'] = '%s/%s' % (dic_feature[task_name]['out'], dic_feature[task_name].get('out_file'))

        return True

    def train(self):
        dic_train = self.dic_config['train']

        if 'task' not in dic_train:
            self.logger.warning('no set task')
            return False

        no_task = False
        if 'no_task' in dic_train:
            no_task = dic_train['no_task']

        depend = 'raw'
        if 'depend' in dic_train:
            depend = dic_train['depend']

        for task_name in dic_train['task']:
            if task_name not in dic_train['task']:
                self.logger.error('no set task')
                return False

            if not dic_train[task_name].get('_in'):
                if 'in' not in dic_train[task_name]:
                    if dic_train[task_name].get('depend'):
                        dic_train[task_name]['in'] = '%s/%s' % (self.dic_config['DATA_DIR'][depend], dic_train[task_name]['depend'])
                    else:
                        dic_train[task_name]['in'] = self.dic_config['DATA_DIR'][depend]
                dic_train[task_name]['_in'] = dic_train[task_name]['in']

                if dic_train[task_name].get('in_file'):
                    dic_train[task_name]['_in'] = '%s/%s' % (dic_train[task_name]['in'], dic_train[task_name].get('in_file'))

            if not dic_train[task_name].get('_out'):
                if 'out' not in dic_train[task_name]:
                    if no_task:
                        dic_train[task_name]['out'] = self.dic_config['DATA_DIR']['train']
                    else:
                        dic_train[task_name]['out'] = '%s/%s' % (self.dic_config['DATA_DIR']['train'], task_name)
                dic_train[task_name]['_out'] = dic_train[task_name]['out']

            if not os.path.exists(dic_train[task_name]['_out']):
                os.makedirs(dic_train[task_name]['_out'])

            if dic_train[task_name].get('out_file'):
                dic_train[task_name]['_out'] = '%s/%s' % (dic_train[task_name]['out'], dic_train[task_name].get('out_file'))

        return True

    def predict(self):
        dic_predict = self.dic_config['predict']
        if 'task' not in dic_predict:
            self.logger.warning('no set task')
            return False

        no_task = False
        if 'no_task' in dic_predict:
            no_task = dic_predict['no_task']

        depend = 'raw'
        if 'depend' in dic_predict:
            depend = dic_predict['depend']

        for task_name in dic_predict['task']:
            if task_name not in dic_predict['task']:
                self.logger.error('no set task')
                return False

            # 模型
            if not dic_predict[task_name].get('_train'):
                if not dic_predict[task_name].get('train'):
                    model_depend = dic_predict[task_name].get('model_depend', task_name)
                    dic_predict[task_name]['train'] = '%s/%s' % (self.dic_config['DATA_DIR']['train'], model_depend)
                dic_predict[task_name]['_train'] = dic_predict[task_name]['train']

                if dic_predict[task_name].get('model_file'):
                    dic_predict[task_name]['_train'] = '%s/%s' % (dic_predict[task_name]['_train'], dic_predict[task_name].get('model_file'))

            # 数据
            if not dic_predict[task_name].get('_in'):
                if 'in' not in dic_predict[task_name]:
                    if dic_predict[task_name].get('depend'):
                        dic_predict[task_name]['in'] = '%s/%s' % (self.dic_config['DATA_DIR'][depend], dic_predict[task_name]['depend'])
                    else:
                        dic_predict[task_name]['in'] = self.dic_config['DATA_DIR'][depend]
                dic_predict[task_name]['_in'] = dic_predict[task_name]['in']

                if dic_predict[task_name].get('in_file'):
                    dic_predict[task_name]['_in'] = '%s/%s' % (dic_predict[task_name]['in'], dic_predict[task_name].get('in_file'))

            if not dic_predict[task_name].get('_out'):
                if 'out' not in dic_predict[task_name]:
                    if no_task:
                        dic_predict[task_name]['out'] = self.dic_config['DATA_DIR']['predict']
                    else:
                        dic_predict[task_name]['out'] = '%s/%s' % (self.dic_config['DATA_DIR']['predict'], task_name)
                dic_predict[task_name]['_out'] = dic_predict[task_name]['out']

            if not os.path.exists(dic_predict[task_name]['_out']):
                os.makedirs(dic_predict[task_name]['_out'])

            if dic_predict[task_name].get('out_file'):
                dic_predict[task_name]['_out'] = '%s/%s' % (dic_predict[task_name]['out'], dic_predict[task_name].get('out_file'))

        return True

    def evaluation(self):
        dic_evaluation = self.dic_config['evaluation']
        if 'task' not in dic_evaluation:
            self.logger.warning('no set task')
            return False

        no_task = False
        if 'no_task' in dic_evaluation:
            no_task = dic_evaluation['no_task']

        depend = 'raw'
        if 'depend' in dic_evaluation:
            depend = dic_evaluation['depend']

        for task_name in dic_evaluation['task']:
            if task_name not in dic_evaluation['task']:
                self.logger.error('no set task')
                return False

            # 输入
            if not dic_evaluation[task_name].get('_in'):
                if 'in' not in dic_evaluation[task_name]:
                    if dic_evaluation[task_name].get('depend'):
                        dic_evaluation[task_name]['in'] = '%s/%s' % (self.dic_config['DATA_DIR'][depend], dic_evaluation[task_name]['depend'])
                    else:
                        dic_evaluation[task_name]['in'] = self.dic_config['DATA_DIR'][depend]
                dic_evaluation[task_name]['_in'] = dic_evaluation[task_name]['in']

                if dic_evaluation[task_name].get('in_file'):
                    dic_evaluation[task_name]['_in'] = '%s/%s' % (dic_evaluation[task_name]['in'], dic_evaluation[task_name].get('in_file'))

            if not dic_evaluation[task_name].get('_out'):
                if 'out' not in dic_evaluation[task_name]:
                    if no_task:
                        dic_evaluation[task_name]['out'] = self.dic_config['DATA_DIR']['evaluation']
                    else:
                        dic_evaluation[task_name]['out'] = '%s/%s' % (self.dic_config['DATA_DIR']['evaluation'], task_name)
                dic_evaluation[task_name]['_out'] = dic_evaluation[task_name]['out']

            if not os.path.exists(dic_evaluation[task_name]['_out']):
                os.makedirs(dic_evaluation[task_name]['_out'])

            if dic_evaluation[task_name].get('out_file'):
                dic_evaluation[task_name]['_out'] = '%s/%s' % (dic_evaluation[task_name]['out'], dic_evaluation[task_name].get('out_file'))

        return True

    def check(self):
        self.set_data_dir()
        if 'generate' in self.dic_config['task']:
            rt = self.generate()
            if not rt:
                self.logger.error('generate config error')
                return {}

        if 'preprocess' in self.dic_config['task']:
            rt = self.preprocess()
            if not rt:
                self.logger.error('preprocess config error')
                return {}

        if 'feature' in self.dic_config['task']:
            rt = self.feature()
            if not rt:
                self.logger.error('feature config error')
                return {}

        if 'train' in self.dic_config['task']:
            rt = self.train()
            if not rt:
                self.logger.error('train config error')
                return {}

        if 'predict' in self.dic_config['task']:
            if 'predict' not in self.dic_config:
                self.logger.error('predict config error')
                return {}
            rt = self.predict()
            if not rt:
                self.logger.error('predict config error')
                return {}

        if 'evaluation' in self.dic_config['task']:
            rt = self.evaluation()
            if not rt:
                self.logger.error('evaluation config error')
                return {}

        for task_name in self.dic_config.get('task'):
            if task_name not in self.dic_config:
                self.logger.error('%s not exist' % task_name)
                return {}

        return self.dic_config
