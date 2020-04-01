# -*- coding:utf-8 -*-
import numpy as np


# 结构数组
def structured_array():
    # 基于普通数组创建一个record array
    Z = np.array([("Hello", 2.5, 3),
                  ("World", 3.6, 2)])
    R = np.core.records.fromarrays(Z.T,
                                   names='col1, col2, col3',
                                   formats='S8, f8, i8')
    print(R)
    
    # 创建一个结构化数组，用来表达位置和颜色（rgb）
    Z = np.zeros(10, [('position', [('x', float, 1),
                                    ('y', float, 1)]),
                      ('color', [('r', float, 1),
                                 ('g', float, 1),
                                 ('b', float, 1)])])
    print(Z)
    
# 创建一个array对象，包含name这个属性
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

def do_named_array():
    # 创建一个array对象，包含name这个属性
    Z = NamedArray(np.arange(10), "range_10")
    print (Z.name)
    
def run():
    for task in tasks:
        eval(task)()

tasks = ['structured_array', 'do_named_array']

        
if __name__ == '__main__':
    run()
