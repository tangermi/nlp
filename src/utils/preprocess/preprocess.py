# -*-coding:utf8-*-
from segment.segment import Segmentor
from preprocess.dict_builder import WordDictBuilder


class Preprocess:
    # 对文档进行分词处理
    @staticmethod
    def preprocess(doc_path_1, doc_path_2, stopword_path):
        with open(doc_path_1, encoding='utf-8') as ins:
            doc_data_1 = ins.read()
            print('Loaded', doc_path_1)
        with open(doc_path_2, encoding='utf-8') as ins:
            doc_data_2 = ins.read()
            print('Loaded', doc_path_2)

        # 停用词表
        stop_words_file = open(stopword_path, 'r', encoding='utf-8')
        stopwords_list = []
        for line in stop_words_file.readlines():
            stopwords_list.append(line[:-1])

        # Init tokenizer
        jt = Segmentor()

        # Tokenization
        doc_token_1 = jt.cut(doc_data_1, '-j')
        doc_token_2 = jt.cut(doc_data_2, '-j')
        doc_token_1 = [token for token in doc_token_1 if (len(token.strip()) != 0) & (token not in stopwords_list)]
        doc_token_2 = [token for token in doc_token_2 if (len(token.strip()) != 0) & (token not in stopwords_list)]

        return doc_token_1, doc_token_2

    @staticmethod
    def token2word_dict_tf(doc_tokens, worddict_tf_path):
        wdb = WordDictBuilder(worddict_tf_path, tokenlist=doc_tokens)
        wdb.run()
        wdb.save(worddict_tf_path)
        print('Totally', len(wdb.word_dict), 'words')
