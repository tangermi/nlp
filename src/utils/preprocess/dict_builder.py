# -*-coding:utf8-*-
from collections import defaultdict
import os


# 把文档包含的词加入到字典（如果字典里不存在该词）
class WordDictBuilder:
    def __init__(self, ori_path='', filelist=[], tokenlist=[]):
        self.word_dict = defaultdict(int)
        if ori_path != '' and os.path.exists(ori_path):
            with open(ori_path, encoding='utf-8') as ins:
                for line in ins.readlines():
                    try:
                        self.word_dict[line.split('\t')[1]] = int(line.split('\t')[2])
                    except IndexError:
                        print('error lien:', line)
                        continue
        self.filelist = filelist
        self.tokenlist = tokenlist

    def run(self):
        for filepath in self.filelist:
            self._updateDict(filepath)
        self._updateDictByTokenList()
        return self

    def _updateDict(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as ins:
            for line in ins.readlines():
                for word in line.rstrip().split()[0].split():
                    self.word_dict[word] += 1

    def _updateDictByTokenList(self):
        print(self.tokenlist)
        for token in self.tokenlist:
            if isinstance(token, str):
                token = token
            self.word_dict[token] += 1

    def save(self, filepath):
        l = [(value, key) for key, value in list(self.word_dict.items())]
        l = sorted(l, reverse=True)
        result_lines = []
        for idx, (value, key) in enumerate(l):
            result_lines.append('%s\t%s\t%s%s' % (idx, key, value, os.linesep))
        with open(filepath, 'w', newline='', encoding='utf-8') as outs:
            outs.writelines(result_lines)
