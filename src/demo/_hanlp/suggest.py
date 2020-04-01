# -*- coding:utf-8 -*-
from pyhanlp import *


# 注意： 大概运行6次，才能返回一次正确结果。 如果把document放在function内，可以提高成功率。
class Suggester:
    # 文本推荐
    def suggest(self, keyword, n, document):
        Suggester = JClass("com.hankcs.hanlp.suggest.Suggester")
        suggester = Suggester()

        for sentence in document:
            suggester.addSentence(sentence)
        return suggester.suggest(keyword, n)




if __name__ == '__main__':
    # 文本推荐
    suggester = Suggester()

    print('文本推荐' + '-' * 20)
    keywords_for_suggest = ['发言', '危机公共', 'mayun', '徐家汇']
    document_for_suggest = [
        "威廉王子发表演说 呼吁保护野生动物",
        "魅惑天后许佳慧不爱“预谋” 独唱《许某某》",
        "《时代》年度人物最终入围名单出炉 普京马云入选",
        "“黑格比”横扫菲：菲吸取“海燕”经验及早疏散",
        "日本保密法将正式生效 日媒指其损害国民知情权",
        "英报告说空气污染带来“公共健康危机”"
    ]
    for keyword in keywords_for_suggest:
        print(suggester.suggest(keyword, 1, document_for_suggest))


# 以下是pyhanlp提供的demo。它也无法保证每次返回推荐结果，但是有显著更好的成功率
# def demo_suggester():
#
#     Suggester = JClass("com.hankcs.hanlp.suggest.Suggester")
#     suggester = Suggester()
#     title_array = [
#         "威廉王子发表演说 呼吁保护野生动物",
#         "魅惑天后许佳慧不爱“预谋” 独唱《许某某》",
#         "《时代》年度人物最终入围名单出炉 普京马云入选",
#         "“黑格比”横扫菲：菲吸取“海燕”经验及早疏散",
#         "日本保密法将正式生效 日媒指其损害国民知情权",
#         "英报告说空气污染带来“公共健康危机”"
#     ]
#     for title in title_array:
#         suggester.addSentence(title)
#
#     print(suggester.suggest("陈述", 2))      # 语义
#     print(suggester.suggest("危机公关", 1))  # 字符
#     print(suggester.suggest("mayun", 1))   # 拼音
#     print(suggester.suggest("徐家汇", 1)) # 拼音
#
# demo_suggester()