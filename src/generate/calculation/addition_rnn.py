# -*- coding: utf-8 -*-
import numpy as np
from ..base import Base


# 生成数字
class AdditionRnn(Base):
    def __init__(self, dic_config={}, dic_engine={}):
        self.dic_engine = dic_engine
        Base.__init__(self, dic_config)
        self.logger.info(dic_engine)

    def init(self):
        self.out_path = self.dic_engine['_out']
        self.hyperparams = self.dic_engine['hyperparams']
        self.training_size = self.hyperparams['training_size']
        self.digits = self.hyperparams['digits']
        self.reverse = self.hyperparams['reverse']

    def generate(self):
        MAXLEN = self.digits + 1 + self.digits
        self.questions = []
        self.expected = []
        seen = set()
        print("Generating data...")
        while len(self.questions) < self.training_size:
            # ——————————————————————————————————————
            # np.random.choice
            # 参数意思分别是从给定的候选集中以概率P随机选择, p没有指定的时候相当于是一致的分布，replacement决定是否放回，size决定选择的个数
            # randint是在[1,4)之间随机产生一个整数，这个整数决定此次产生的字符串的长度，然后依次随机从0-9中选择一个数字，最后拼接起来
            # ——————————————————————————————————————
            f = lambda: int(
                "".join(np.random.choice(list("0123456789")) for i in range(np.random.randint(1, self.digits + 1))))
            a, b = f(), f()
            # 将随机选择的两个整数排序
            key = tuple(sorted((a, b)))
            # 如果此次选择的加法问题已经存在了，则忽略
            if key in seen:
                continue
            seen.add(key)

            q = "{}+{}".format(a, b)                    # 构造问题，组装成a+b的字符串
            query = q + " " * (MAXLEN - len(q))         # 用空格来填充问题成预设的最大长度
            ans = str(a + b)                            # 构造问题的答案
            ans += " " * (self.digits + 1 - len(ans))   # 用空格来填充答案，使其长度为4
            # 如果需要逆序
            if self.reverse:
                query = query[::-1]
            # 将构造的问题和答案加入相应的列表
            self.questions.append(query)
            self.expected.append(ans)
        self.logger.info('Total addition questions:' + str(len(self.questions)))

    def dump(self):
        np.savez(self.out_path, questions=self.questions, answers=self.expected)

    def run(self):
        self.init()
        self.generate()
        self.dump()

