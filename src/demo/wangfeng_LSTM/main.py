# -*- coding:utf-8 -*-
from segment import segment
from one_hot import one_hot
from network import network
from generate_seeds import generate_seeds
from display_result import display_result


if __name__ == '__main__':
    # 此文本用于生成，生成结果会在屏幕上直接输出。
    train_path = '/apps/data/ai_nlp_testing/raw/wangefng_lyrics/new_wangfeng.txt'
    model_path = "/apps/data/ai_nlp_testing/model/wangfeng_lyrics/weights-improvement=26-0.105659.hdf5"

    seg_list = segment(train_path)
    dataX, dataY, n_vocab, seq_length, int_to_word = one_hot(seg_list)
    model = network(model_path, n_vocab, seq_length)
    final_result = generate_seeds(model, dataX, int_to_word)
    display_result(final_result)
