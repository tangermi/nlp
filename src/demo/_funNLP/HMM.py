# -*- coding:utf-8 -*-
import pickle   #用作保存模型
import json   #用作保存模型


class HMM_Model:
    def __init__(self):
        self.trans_mat = {}   # 状态转移矩阵 trans_mat[state1][state2]表示训练集中由state1 转移到 state2 的次数.
        self.emit_mat = {}   # 观测矩阵 emit_mat[state][char]表示训练集中单字char被标注为state的次数.
        self.init_vec = {}   # 初始状态分布向量 init_vec[state]表示状态state在训练集中出现的次数.
        self.state_count = {}   # 状态统计向量 state_count[state]表示状态state出现的次数.
        self.states = {}
        self.inited = False

    # 初始化
    def setup(self):
        for state in self.states:
            # build trans_mat
            self.trans_mat[state] = {}
            for target in self.states:
                self.trans_mat[state][target] = 0.0
            self.emit_mat[state] = {}
            self.init_vec[state] = 0
            self.state_count[state] = 0
        self.inited = True

    # 模型保存
    def save(self, filename='hmm.json', code='json'):
        fw = open(filename, 'w', encoding='utf-8')
        data = {
            'trans_mat': self.trans_mat,
            'emit_mat': self.emit_mat,
            'init_vec': self.init_vec,
            'state_count': self.state_count
        }
        if code == 'json':
            txt = json.dumps(data)
            txt = txt.encode('utf-8').decode('unicode-escape')
            fw.write(txt)
        elif code == 'pickle':
            pickle.dump(data, fw)
        fw.close()

    # 模型加载
    def load(self, filename='hmm.json', code='json'):
        fr = open(filename, 'r', encoding='utf-8')
        if code == 'json':
            txt = fr.read()
            model = json.loads(txt)
        elif code == 'pickle':
            model = pickle.load(fr)
        self.trans_mat = model['trans_mat']
        self.emit_mat = model['emit_mat']
        self.init_vec = model['init_vec']
        self.state_count = model['state_count']
        self.inited = True
        fr.close()

    # 模型训练
    def do_train(self, observes, states):
        if not self.inited:
            self.setup()

        for i in range(len(states)):
            if i == 0:
                self.init_vec[states[0]] += 1  # 记录每一行的开头字的状态
                self.state_count[states[0]] += 1
            else:
                self.trans_mat[states[i - 1]][states[i]] += 1   # 记录每一次状态的转移
                self.state_count[states[i]] += 1
                if observes[i] not in self.emit_mat[states[i]]:
                    self.emit_mat[states[i]][observes[i]] = 1   # 记录每个字（非句首）在不同状态的次数
                else:
                    self.emit_mat[states[i]][observes[i]] += 1

    def get_prob(self):
        init_vec = {}
        trans_mat = {}
        emit_mat = {}
        default = max(self.state_count.values())

        for key in self.init_vec:
            if self.state_count[key] != 0:
                init_vec[key] = float(self.init_vec[key]) / self.state_count[key]   # 计算数据集中句首处于各状态的概率
            else:
                init_vec[key] = float(self.init_vec[key]) / default

        for key1 in self.trans_mat:
            trans_mat[key1] = {}
            for key2 in self.trans_mat[key1]:
                if self.state_count[key1] != 0:
                    trans_mat[key1][key2] = float(self.trans_mat[key1][key2]) / self.state_count[key1]  # 计算各个状态下的后一字处于各状态的概率
                else:
                    trans_mat[key1][key2] = float(self.trans_mat[key1][key2]) / default

        for key1 in self.emit_mat:
            emit_mat[key1] = {}
            for key2 in self.emit_mat[key1]:
                if self.state_count[key1] != 0:
                    emit_mat[key1][key2] = float(self.emit_mat[key1][key2]) / self.state_count[key1]   # 计算每个字处于每个状态下的概率
                else:
                    emit_mat[key1][key2] = float(self.emit_mat[key1][key2]) / default
        return init_vec, trans_mat, emit_mat

    # 模型预测
    def do_predict(self, sequence):
        EPS = 0.0001   # 设置默认值，当字典里找不到字对应值时，返回该值
        tab = [{}]
        path = {}
        init_vec, trans_mat, emit_mat = self.get_prob()

        # 初始化
        for state in self.states:
            tab[0][state] = init_vec[state] * emit_mat[state].get(sequence[0], EPS)   # 句首字
            path[state] = [state]

        # 创建动态搜索表
        for t in range(1, len(sequence)):
            tab.append({})
            new_path = {}
            for state1 in self.states:
                items = []
                for state2 in self.states:
                    if tab[t - 1][state2] == 0:
                        continue
                    prob = tab[t - 1][state2] * trans_mat[state2].get(state1, EPS) * emit_mat[state1].get(sequence[t], EPS)
                    items.append((prob, state2))
                best = max(items)
                tab[t][state1] = best[0]
                new_path[state1] = path[best[1]] + [state1]
            path = new_path

        # 搜索最有路径
        prob, state = max([(tab[len(sequence) - 1][state], state) for state in self.states])
        return path[state]

