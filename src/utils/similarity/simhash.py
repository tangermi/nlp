# -*- coding=utf-8 -*-
# Implementation of Charikar simhashes in Python
# See: http://dsrg.mff.cuni.cz/~holub/sw/shash/#a1


# 把文字转化为hash
class SimhashBuilder:
    def __init__(self, word_list=[], hashbits=128):
        self.hashbits = hashbits
        self.hashval_list = [self._string_hash(word) for word in word_list]
        print('Totally: %s words' % (len(self.hashval_list),))

        # with open('word_hash.txt', 'w') as outs:
        #    for word in word_list:
        #        outs.write(word+'\t'+str(self._string_hash(word))+os.linesep)

    def _string_hash(self, word):
        # A variable-length version of Python's builtin hash
        if word == "":
            return 0
        else:
            x = ord(word[0]) << 7
            m = 1000003
            mask = 2 ** self.hashbits - 1
            for c in word:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(word)
            if x == -1:
                x = -2
            return x

    def sim_hash_nonzero(self, feature_vec):
        finger_vec = [0] * self.hashbits
        # Feature_vec is like [(idx,nonzero-value),(idx,nonzero-value)...]
        for idx, feature in feature_vec:
            hashval = self.hashval_list[int(idx)]
            for i in range(self.hashbits):
                bitmask = 1 << i
                if bitmask & hashval != 0:
                    finger_vec[i] += float(feature)
                else:
                    finger_vec[i] -= float(feature)
        # print finger_vec
        fingerprint = 0
        for i in range(self.hashbits):
            if finger_vec[i] >= 0:
                fingerprint += 1 << i

        # 整个文档的fingerprint为最终各个位大于等于0的位的和
        return fingerprint

    def sim_hash(self, feature_vec):
        finger_vec = [0] * self.hashbits
        for idx, feature in enumerate(feature_vec):
            if float(feature) < 1e-6:
                continue
            hashval = self.hashval_list[idx]
            for i in range(self.hashbits):
                bitmask = 1 << i
                if bitmask & hashval != 0:
                    finger_vec[i] += float(feature)
                else:
                    finger_vec[i] -= float(feature)
        # print finger_vec
        fingerprint = 0
        for i in range(self.hashbits):
            if finger_vec[i] >= 0:
                fingerprint += 1 << i
        # 整个文档的fingerprint为最终各个位大于等于0的位的和
        return fingerprint

    def _add_word(self, word):
        self.hashval_list.append(self._string_hash(word))

    def update_words(self, word_list=[]):
        for word in word_list:
            self._add_word(word)
