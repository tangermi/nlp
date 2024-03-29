import numpy as np
import math


# E step
def pmf(i, pro_A, pro_B, por_C):
    pro_1 = pro_A * math.pow(pro_B, data[i]) * math.pow((1 - pro_B), 1 - data[i])
    pro_2 = pro_A * math.pow(pro_C, data[i]) * math.pow((1 - pro_C), 1 - data[i])
    return pro_1 / (pro_1 + pro_2)


# M step
class EM:
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob

    # e_step
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow((1 - self.pro_B), 1 - data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow((1 - self.pro_C), 1 - data[i])
        return pro_1 / (pro_1 + pro_2)

    # m_step
    def fit(self, data):
        count = len(data)
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B, self.pro_C))
        for d in range(count):
            _ = yield
            _pmf = [self.pmf(k) for k in range(count)]
            pro_A = 1 / count * sum(_pmf)
            pro_B = sum([_pmf[k] * data[k] for k in range(count)]) / sum([_pmf[k] for k in range(count)])
            pro_C = sum([(1 - _pmf[k]) * data[k] for k in range(count)]) / sum([(1 - _pmf[k]) for k in range(count)])
            print('{}/{}  pro_a:{:.3f}, pro_b:{:.3f}, pro_c:{:.3f}'.format(d + 1, count, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C


if __name__ == '__main__':
    pro_A, pro_B, por_C = 0.5, 0.5, 0.5
    data = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    em = EM(prob=[0.5, 0.5, 0.5])
    f = em.fit(data)
    next(f)
    # 第一次迭代
    f.send(1)
    # 第二次
    f.send(2)
    # 第二个例子
    em = EM(prob=[0.4, 0.6, 0.7])
    f2 = em.fit(data)
    next(f2)
    f2.send(1)
    f2.send(2)
