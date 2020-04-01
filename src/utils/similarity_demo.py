# -*- coding: utf-8 -*-
from similarity.compare import Compare


def test():
    doc_path_1 = r'/apps/data/ai_nlp_testing/raw/similarity/doc3_for_sim.txt'
    doc_path_2 = r'/apps/data/ai_nlp_testing/raw/similarity/doc4_for_sim.txt'
    stopword_path = r'/apps/data/ai_nlp_testing/stopwords/stopwords_cn.txt'
    word_dict = r'/apps/data/ai_nlp_testing/raw/similarity/dict'
    # Simhash similarity
    threshold = 10
    compare = Compare(doc_path_1, doc_path_2, stopword_path, word_dict, threshold)
    compare.run(mode='simhash')
    # Cosine similarity
    # threshold = 0.5
    # compare = Compare(doc_path_1, doc_path_2, stopword_path, word_dict, threshold)
    # compare.run(mode='cosine')


if __name__ == '__main__':
    test()
