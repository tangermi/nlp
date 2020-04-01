
from stanfordcorenlp import StanfordCoreNLP
from nltk.tree import Tree


if __name__ == '__main__':

    # 模型包较大，需要耐心等待加载
    nlp = StanfordCoreNLP(r'/apps/data/ai_nlp_testing/model/stanford-corenlp-full-2018-10-05', lang='zh')

    text = '清华大学位于北京市海淀区中关村北大街'
    # 分词
    print(nlp.word_tokenize(text))
    # 词性标注
    print(nlp.pos_tag(text))
    # 命名实体识别
    print(nlp.ner(text))
    # 句法依存分析
    print(nlp.dependency_parse(text))
    # 句法解析
    print(nlp.parse(text))

    #可视化
    tree = Tree.fromstring(nlp.parse(text))
    tree.draw()
    print(tree.height())   # 树的高度
    print(tree.leaves())   # 树的叶子节点
    print(tree.productions)   # 生成与树的非终端节点对应的结果
