# -*- coding: UTF-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdf:
    @staticmethod
    def compute_tf(word_dict, bow):
        tf_dict = {}
        bow_count = len(bow)
        for word, count in word_dict.items():
            tf_dict[word] = count / float(bow_count)
        return tf_dict

    @staticmethod
    def compute_idf(doc_list):
        import math
        idf_dict = {}
        n = len(doc_list)

        idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
        for doc in doc_list:
            for word, val in doc.items():
                if word in idf_dict:
                    if val > 0:
                        idf_dict[word] += 1
                else:
                    if val > 0:
                        idf_dict[word] = 1

        for word, val in idf_dict.items():
            idf_dict[word] = math.log10(n / float(val))

        return idf_dict

    @staticmethod
    def compute_tfidf(tf_bow, idfs):
        tfidf = {}
        for word, val in tf_bow.items():
            tfidf[word] = val * idfs[word]
        return tfidf


def run():
    s1 = 'I love you so much'
    s2 = 'I hate you! shit!'
    s3 = 'I like you, but just like you'
    count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english')
    response = count_vec.fit_transform([s1, s2, s3])  # s must be string
    print(count_vec.get_feature_names())
    print(response.toarray())


if __name__ == '__main__':
    run()
