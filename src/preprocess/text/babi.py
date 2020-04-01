# -*- coding:utf-8 -*-
from ..base import Base
import os
from keras.preprocessing.text import text_to_word_sequence
from functools import reduce
import tarfile
import numpy as np


'''
# 问答
# 数据结构： 每15行是一个块，每个块包含由3行组成的5个部分(1.文本，2.问题，3.答案。)
# 在后面的部分可以记忆前面部分的文本。
# 例如: 第5个部分的问题可以根据第1个部分的文本回答问题。
'''
class Babi(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.data_path = self.dic_engine['_in']
        self.train_path = os.path.join(self.dic_engine['_out'], self.dic_engine['train_file'])
        self.test_path = os.path.join(self.dic_engine['_out'], self.dic_engine['test_file'])

    @staticmethod
    def tokenize(sent):
        '''Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        '''

        return text_to_word_sequence(sent)

    def parse_stories(self, lines, only_supporting=False):
        '''Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        '''
        data = []
        story = []
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = []
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def get_stories(self, f, only_supporting=False, max_length=None):
        '''Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        '''
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)

        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data
                if not max_length or len(flatten(story)) < max_length]
        return data

    def process(self):
        challenges = {
            # QA1 with 10,000 samples
            'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_'
                                          'single-supporting-fact_{}.txt',   # {}可以被.format填入train或者test
            # QA2 with 10,000 samples
            'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_'
                                        'two-supporting-facts_{}.txt',   # {}可以被.format填入train或者test
        }
        challenge_type = 'single_supporting_fact_10k'
        challenge = challenges[challenge_type]

        self.logger.info('Extracting stories for the challenge:', challenge_type)
        with tarfile.open(self.data_path) as tar:
            self.train_stories = self.get_stories(tar.extractfile(challenge.format('train')))
            self.test_stories = self.get_stories(tar.extractfile(challenge.format('test')))

    def dump(self):
        np.save(self.train_path, self.train_stories)
        np.save(self.test_path, self.test_stories)

    def run(self):
        self.init()
        self.process()
        self.dump()
