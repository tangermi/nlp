#!usr/bin/env python
# -*- coding:utf-8 -*-

import jieba
import jieba.analyse
from lxml import etree

class utils():
    def __init__(self, dicConfig={}, dic_param={}):
        self.dic_tag = dicConfig['JIEBA']['tag']
        
    def get_text(self, content):
        tree = etree.HTML(content)
        node = tree.xpath("//html")
        if len(node) > 0:
            rc = []
            for node in node[0].itertext():
                rc.append(node.strip())
            return ''.join(rc)
        return ''

    def get_tags(self, dic_text):
        dic_tag_id = {}
        for key, val in dic_text.items():
            text, weight = val
            if key == 'content':
                text = self.get_text(text)
            
            seg_list = jieba.cut(text)
            for seg in seg_list:
                if seg in self.dic_tag:
                    tag_id = self.dic_tag[seg]
                    if tag_id not in dic_tag_id:
                        dic_tag_id[tag_id] = None
                    
            # tag_list = jieba.analyse.extract_tags(text, int(len(text) / 20.0))
            # for i in range(0, len(tag_list)):
                # final_tag_list.append((tag_list[i], weight * i))

        return dic_tag_id.keys()
