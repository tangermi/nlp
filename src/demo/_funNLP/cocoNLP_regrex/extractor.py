# -*- coding:utf-8 -*-
from cocoNLP.extractor import extractor


class Extractor:
    def __init__(self):
        self.extractor = extractor()

    def get_email(self, text):
        emails = self.extractor.extract_email(text)
        return emails

    def get_phone(self, text, nation='CHN'):
        phones = self.extractor.extract_cellphone(text, nation)
        return phones

    def get_id(self, text):
        ids = self.extractor.extract_ids(text)
        return ids

    def get_locs(self, text):
        phones = self.get_phone(text)
        locs = [self.extractor.extract_cellphone_location(phone, 'CHN') for phone in phones]
        return locs

    def get_locations(self, text):
        locations = self.extractor.extract_locations(text)
        return locations

    def get_time(self, text):
        times = self.extractor.extract_time(text)
        return times

    def get_name(self, text):
        names = self.extractor.extract_name(text)
        print(names)


if __name__ == '__main__':
    text = '急寻特朗普，男孩，于2018年11月27号11时在陕西省安康市汉滨区走失。丢失发型短发，...如有线索，请迅速与警方联系：18100065143，132-6156-2938，baizhantang@sina.com.cn 和yangyangfuture at gmail dot com, 他的身份证号是410105196904010537, 他爸爸的号码是15827469541'

    extractor = Extractor()
    # 提取email
    print(extractor.get_email(text))
    # 提取电话号码
    print(extractor.get_phone(text))
    # 提取身份证号
    print(extractor.get_id(text))
    # 提取电话号码以及运营商/归属地信息
    print(extractor.get_locs(text))
    # 提取地点信息
    print(extractor.get_locations(text))
    # 提取时间
    print(extractor.get_time(text))
    #提取人名
    print(extractor.get_name(text))