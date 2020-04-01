#!usr/bin/env python
# -*- coding:utf-8 -*-

from lxml import etree
import logging

from HTMLParser import HTMLParser  
from re import sub

import jieba
import jieba.analyse
import networkx as nx  
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer  

class _DeHTMLParser(HTMLParser):  
    def __init__(self):  
        HTMLParser.__init__(self)  
        self.__text = []  
  
    def handle_data(self, data):  
        text = data.strip()  
        if len(text) > 0:  
            text = sub('[ \t\r\n]+', ' ', text)  
            self.__text.append(text + ' ')  
  
    def handle_starttag(self, tag, attrs):  
        pass
        # if tag == 'p':  
        #     self.__text.append('\n')  
        # elif tag == 'br':  
        #     self.__text.append('\n')  
  
    def handle_startendtag(self, tag, attrs):  
        pass
        # if tag == 'br':  
        #     self.__text.append('\n\n')  
  
    def text(self):  
        return ''.join(self.__text).strip()  


class utils():
    def __init__(self, dicConfig={}, dic_param={}):
        self.dicConfig = dicConfig
        self.dic_jieba = self.dicConfig['JIEBA']

    ####------------------------------------------
    def get_text(self, content=None, page=None):
        if not page:
            try:
                page = etree.HTML(content.decode('utf8'))
            except UnicodeEncodeError as e:
                page = etree.HTML(content)

        node = page.xpath("//html")
        if len(node) > 0:
            rc = []
            for node in node[0].itertext():
                # logging.info(node.strip())
                rc.append(node.strip())
            return '@'.join(rc)
        return ''

    def get_text1(self, content=None, page=None):
        if not page:
            try:
                page = etree.HTML(content.decode('utf8'))
            except UnicodeEncodeError as e:
                page = etree.HTML(content)

        p_text_list = []
        p_node = page.xpath("//p")
        for p in p_node:
            p_text = p.xpath('string(.)').strip()
            if p_text:
                p_text_list.append(p_text)
                # logging.info(p_text)

        return '@'.join(p_text_list)

    def dehtml(self, content):  
        try:  
            parser = _DeHTMLParser()  
            parser.feed(content)  
            parser.close()  
            return parser.text()  
        except:  
            self.logger.info('error')
            return text
    
    ###------
    # def load_stopwords(self, path='/apps/test/moplus/data/jieba/stop_word.dic'):  
    #     """ 
    #     加载停用词 
    #     :param path: 
    #     :return: 
    #     """  
    #     with open(path) as f:  
    #         stopwords = filter(lambda x: x, map(lambda x: x.strip().decode('utf-8'), f.readlines()))  
    #     stopwords.extend([' ', '\t', '\n'])  
    #     return frozenset(stopwords)  

    ###--------
    def cut_words(self, sentence):  
        """ 
        分词 
        :param sentence: 
        :return: 
        """  
        stopwords = load_stopwords()  
        return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))  

    ####-------------------------
    # 自动提取关键词
    def get_tags(self, title, content):
        dic_tag_id = {}

        content_txt = self.get_text(content=content)
        
        # 标题
        seg_list = jieba.cut(title)
        for seg in seg_list:
            if seg in self.dic_jieba['tag']:
                tag_id = self.dic_jieba['tag'][seg]
                if tag_id not in dic_tag_id:
                    dic_tag_id[tag_id] = None
        
        # 正文
        tag_list = jieba.analyse.extract_tags(text, 10)
        for tag in tag_list:
            if seg in self.dic_jieba['tag']:
                tag_id = self.dic_jieba['tag'][seg]
                if tag_id not in dic_tag_id:
                    dic_tag_id[tag_id] = None

        return dic_tag_id.keys()
      
    ###------------
    # 自动提取摘要
    # 分句
    def cut_sentence(self, sentence):  
        if not isinstance(sentence, unicode):  
            sentence = sentence.decode('utf-8')  
        delimiters = frozenset(u'。！？@')  
        begin_ch = frozenset(u'#-/一“')  
        buf = []  
        for ch in sentence: 
            if ch <> u'@': 
                buf.append(ch)
            if delimiters.__contains__(ch): 
                if len(buf) > 6:
                    if begin_ch.__contains__(buf[0]):
                        buf = []
                        continue

                    yield ''.join(buf)  
                    buf = []

        if len(buf) > 6:
            if not begin_ch.__contains__(buf[0]):
                yield ''.join(buf)  
  
    # 利用textrank提取摘要 
    def get_abstract(self, content, size=3):  
        docs = list(self.cut_sentence(content))
        new_docs = []
        for line in docs:
            if line.find('http') >= 0:
                continue

            if line.find('#') == 0:
                continue

            new_docs.append(line)

        if not new_docs:
            return ''

        tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=self.dic_jieba['stop_words'])  
        tfidf_matrix = tfidf_model.fit_transform(new_docs)  
        normalized_matrix = TfidfTransformer().fit_transform(tfidf_matrix)  
        similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)  
        scores = nx.pagerank(similarity)  
        tops = sorted(scores.iteritems(), key=lambda x: x[1], reverse=True)  
        size = min(size, len(new_docs))  
        indices = map(lambda x: x[0], tops)[:size]  
         
        brief = ''
        for i in map(lambda idx: new_docs[idx], indices):  
            brief = brief + i.encode('utf8')

        return brief

    ### 
    def get_abstract_head(self, content, size=3):  
        docs = list(self.cut_sentence(content))
        new_docs = []
        for line in docs:
            if line.find('http') >= 0:
                continue

            if line.find('#') == 0:
                continue
            new_docs.append(line)

        if not new_docs:
            return ''
         
        brief = ''
        for i in range(0, size):  
            brief = brief + new_docs[i].encode('utf8')

        return brief

    ####-------------------------
    # 图文简介
    def get_brief(self, content, brief_size=2):
        if not content:
            return ''

        # try:
        #     page = etree.HTML(content.decode('utf8'))
        # except UnicodeEncodeError as e:
        #     page = etree.HTML(content)

        # 简介
        # content_txt = page.xpath(u"//text()")[0]
        try:
            parser = HTMLParser()
            content = parser.unescape(content.decode('utf8'))
        except UnicodeEncodeError as e:
            pass
        
        content_txt = self.dehtml(content)
        brief = self.get_abstract(content_txt, brief_size)
        # brief = '%s...' % content_txt[0:brief_len]
        # brief = brief.replace(u'。...', u'...')
        return brief

    # 图文图片
    def get_brief_images(self, content, img_num=6):
        if not content:
            return [], 0

        try:
            page = etree.HTML(content.decode('utf8'))
        except UnicodeEncodeError as e:
            page = etree.HTML(content)
        
        img_url_list = []
        imgs = page.xpath(u"//img")
        for img in imgs:
            data_original = img.get('data-original')
            data_original_width = img.get('data-original-width')
            data_original_height = img.get('data-original-height')
            if data_original:
                if data_original_width:
                    if int(data_original_width) < 300 or int(data_original_height) < 300:
                        continue

                img_url_list.append(data_original)
                if len(img_url_list) > img_num:
                    break
                continue

            src = img.get('src')
            src_width = img.get('src-width')
            src_height = img.get('src-height')
            if src:
                if src_width:
                    if int(src_width) < 300 or int(src_height) < 300:
                        continue

                img_url_list.append(src)

        return img_url_list[0:img_num], len(img_url_list)

