# -*- coding:utf-8 -*-
from HMM import HMM_Model


def get_tags(src):
    tags = []
    if len(src) == 1:
        tags = ['S']
    elif len(src) == 2:
        tags = ['B', 'E']
    else:
        m_num = len(src) - 2
        tags.append('B')
        tags.extend(['M'] * m_num)
        tags.append('E')
    return tags

def cut_sent(src, tags):
    word_list = []
    start = -1
    started = False

    if len(tags) != len(src):
        return None

    if tags[-1] not in {'S', 'E'}:
        if tags[-2] in {'S', 'E'}:
            tags[-1] = 'S'
        else:
            tags[-1] = 'E'

    for i in range(len(tags)):
        if tags[i] == 'S':
            if started:
                started = False
                word_list.append(src[start:i])
            word_list.append(src[i])
        elif tags[i] == 'B':
            if started:
                word_list.append(src[start:i])
            start = i
            started = True
        elif tags[i] == 'E':
            started = False
            word = src[start:i+1]
            word_list.append(word)
        elif tags[i] == 'M':
            continue
    return word_list

class HMMSoyoger(HMM_Model):
    def __init__(self, *args, **kwargs):
        super(HMMSoyoger, self).__init__(*args, **kwargs)
        self.states = STATES
        self.data = None

    # 加载训练数据
    def read_txt(self, filename):
        self.data = open(filename, 'r', encoding='utf-8')

    # 模型训练函数
    def train(self):
        if not self.inited:
            self.setup()

        for line in self.data:
            line = line.strip()
            if not line:
                continue

            # 观测序列
            observes = []
            for i in range(len(line)):
                if line[i] == ' ':
                    continue
                observes.append(line[i])   # 把行内容添加到 observes
            # 状态序列
            words = line.split(" ")

            states = []
            for word in words:
                if word in seg_stop_words:
                    continue
                states.extend(get_tags(word))
            # 开始序列
            if(len(observes) >= len(states)):
                self.do_train(observes, states)
            else:
                pass

    # 模型分词预测
    def lcut(self, sentence):
        # try:
            tags = self.do_predict(sentence)
            return cut_sent(sentence, tags)
        # except:
        #     return sentence


if __name__ == '__main__':
    # 定义HMM中的状态，初始化概率，以及中文停顿词:
    STATES = {'B', 'M', 'E', 'S'}
    # 定义停顿标点
    seg_stop_words = {" ", "，", "。", "“", "”", '“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’", "──", ",", ".",
                      "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", "[", "]",
                      "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n", "\t"}

    soyoger = HMMSoyoger()
    soyoger.read_txt('train_corpus.txt')
    soyoger.train()
    print(soyoger.lcut("中国的人工智能发展进入高潮阶段"))


