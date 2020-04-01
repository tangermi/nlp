# -*- coding:utf-8 -*-
import numpy as np


def generate_seeds(model, dataX, int_to_word):
    # 生成种子
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed : ")
    print(''.join([int_to_word[value] for value in pattern]))
    n_generation = 400  # 生成的长度
    print('开始生成，生成长度为', n_generation)
    finall_result = []
    for i in range(n_generation):
        x = np.reshape(pattern, (1, len(pattern)))
        prediction = model.predict(x, verbose=0)[0]
        index = np.argmax(prediction)
        result = int_to_word[index]
        # sys.stdout.write(result)
        finall_result.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return finall_result
