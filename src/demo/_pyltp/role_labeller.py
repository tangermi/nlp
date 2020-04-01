# -*- coding:utf-8 -*-
import os

from pyltp import SementicRoleLabeller
from dependency import DependencyParser


class RoleLabeller(DependencyParser):
    def label(self, text):
        srl_model_path = os.path.join(self.ltp_data_dir, 'pisrl.model')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。
        labeller = SementicRoleLabeller()  # 初始化实例
        labeller.load(srl_model_path)  # 加载模型

        words, postags = self.cut(text, part_of_speech=True)
        arcs = self.parse(text)
        # arcs 使用依存句法分析的结果
        roles = labeller.label(words, postags, arcs)  # 语义角色标注

        # 打印结果
        # for role in roles:
        #     print(role.index, "".join(
        #         ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
        labeller.release()  # 释放模型
        return roles


# test
if __name__ == '__main__':
    test_string = '元芳你怎么看'
    role_labeller = RoleLabeller()
    words, postags = role_labeller.cut(test_string, part_of_speech=True)
    print(words, postags)
    roles = role_labeller.label(test_string)
    for role in roles:
        print(role.index, "".join(
            ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
