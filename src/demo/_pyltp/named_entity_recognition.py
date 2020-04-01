# -*- coding:utf-8 -*-
import os

from pyltp_segment import PyltpSegment


class NER(PyltpSegment):
    def recognise(self, text):
        ner_model_path = os.path.join(self.ltp_data_dir, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`

        from pyltp import NamedEntityRecognizer
        recognizer = NamedEntityRecognizer()  # 初始化实例
        recognizer.load(ner_model_path)  # 加载模型
        words, postags = self.cut(text, part_of_speech=True)

        netags = recognizer.recognize(words, postags)  # 命名实体识别
        recognizer.release()  # 释放模型
        # LTP 提供的命名实体类型为:人名（Nh）、地名（Ns）、机构名（Ni）。
        # LTP 采用 BIESO 标注体系。B 表示实体开始词，I表示实体中间词，E表示实体结束词，S表示单独成实体，O表示不构成命名实体。
        return words, list(netags)



if __name__ == '__main__':
    test_string = ['签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。',
                   '一桶冰水当头倒下，微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克全都不惜湿身入镜，这些硅谷的科技人，飞蛾扑火似地牺牲演出，其实全为了慈善。',
                   '北川景子参演了林诣彬导演的《速度与激情3》',
                   '蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机',
                   '我在上海林原科技有限公司兼职工作，偶尔去地中海影城看电影。',
                   ]

    ner = NER()
    for sentence in test_string:
        print(ner.recognise(sentence))
