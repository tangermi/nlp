# -*- coding:utf-8 -*-
import os

from pyltp_segment import PyltpSegment


class CustomisedDict(PyltpSegment):
    def loaddict(self):
        # user_dict = '/apps/data/ai_nlp_testing/raw/customized_dict_for_ltp/chemistry.txt'
        cws_model_path = os.path.join(self.ltp_data_dir, 'cws.model')
        self.segmentor.load_with_lexicon(cws_model_path, self.user_dict)  # 加载模型，第二个参数是您的外部词典文件路径


    def addword(self, word):
        if self.user_dict: #若指定了用户字典，则向字典中添加
            f = open(self.user_dict, 'r', encoding='utf-8')
            if f.read().find(word) == -1:
                with open(self.user_dict, 'a+', encoding='utf-8') as f:
                    f.read()
                    f.write('\n'+word)
        else: #新建字典
            f = open('dict.txt', 'a+', encoding='utf-8')
            f.write('\n' + word)
            self.user_dict = 'dict.txt'
            f.close()
            self.loaddict('dict.txt')


# test
if __name__ == '__main__':
    test_string = '距离除夕越来越近，这几天，铁路部门迎来了返乡客流的最高峰。然而，仍有些人，' \
                  '为买不到返乡的车票发愁。某些热门线路，为什么12306总是显示没票可卖?' \
                  '第三方抢票软件到底能不能顺利抢到票?铁路售票平台有没有更多的渠道向社会开放?' \
                    '亚硝酸盐是一种化学物质，叔丁基锂和苯并芘也是,苛性钠呢'

    customized_dict = CustomisedDict()
    print(customized_dict.cut(test_string))
    customized_dict.addword('叔丁基锂')
    customized_dict.loaddict()
    print(customized_dict.cut(test_string))
