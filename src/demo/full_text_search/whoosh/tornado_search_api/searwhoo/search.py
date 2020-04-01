# -*- coding: utf-8 -*-
# @author: NiHao

# to do:
# 2、完成sphinx的相关接口
# 5、整体测试

import os

import pymysql
from jieba.analyse import ChineseAnalyzer
from whoosh.qparser import QueryParser
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, DATETIME
from whoosh.query import Query, Term
from whoosh.writing import AsyncWriter

from searwhoo.log import logger


class WhooshIx(object):
    """ 使用whoosh 建立索引 """
    def __init__(self, ix_dir, ix_name=None):
        self.ix_dir = ix_dir
        self.ix_name = ix_name
        self.analyzer = ChineseAnalyzer()
        self.schema = Schema(id=ID(stored=True, unique=True, sortable=True),
                             name=TEXT(stored=True, analyzer=self.analyzer),
                             address=TEXT(analyzer=self.analyzer),
                             role=ID(sortable=True),
                             intro=TEXT(analyzer=self.analyzer),
                             date=DATETIME(stored=True, sortable=True))

        self.index = self.open_index()
        self.searcher = None
        if self.index:
            self.searcher = self.index.searcher()
        self.db = None

    def create_index(self):
        # 创建索引文件
        if not os.path.exists(self.ix_dir):
            os.mkdir(self.ix_dir)
        self.index = create_in(self.ix_dir, self.schema, indexname=self.ix_name)

    def open_index(self):
        # 打开索引文件
        ix = open_dir(self.ix_dir, indexname=self.ix_name)
        return ix

    def search(self, word, **kwargs):
        # 在索引中搜索，默认按 score 排序
        tag = kwargs.get('tag')
        offset = int(kwargs.get('offset', 10))
        page = int(kwargs.get('page', 1))

        q = QueryParser('intro', self.index.schema).parse(word)
        if tag:
            allow_q = Term('role', tag)
        else:
            allow_q = None

        if self.searcher is None:
            self.searcher = self.index.searcher()
        # 检查searcher 使用的是否是最新版本的index
        if not self.searcher.up_to_date():
            self.searcher = self.searcher.refresh()

        try:
            r = self.searcher.search(q, limit=offset * page,
                    filter=allow_q)
        except:
            self.logger.error('搜索时出错，相关参数：q=%s, limit=%s, filter=%s' % (q, offset * page, allow_q))
        else:
            result = r[(page - 1) * offset:page * offset]
            return self.results_to_ids(result)

    def results_to_ids(self, results):
        # 从搜索结果中提取id字段
        return [r.get('id') for r in results]

    def _get_index_writer(self):
        # 获取index 的writer ，若获取时没有lock ，则使用普通writer 对象：IndexWriter
        # 否则使用 AsyncWriter
        return AsyncWriter(self.index)

    def indexing_doc(self):
        # 建立索引
        if not self.index:
            self.create_index()
        
        if not self.db:
            self.connect_db()

        writer = self._get_index_writer()
        cur = self._read_from_db()
        max_id = 0
        r = cur.fetchone()
        try:
            while r:
                writer.add_document(id=str(r[0]), name=r[1], address=r[2], role=str(r[3]), intro=r[4], date=r[5])
                if int(r[0]) > max_id:
                    max_id = r[0]
                r = cur.fetchone()
        except:
            writer.cancel()
        else:
            writer.commit()
            self._store_indexed_id_to_db(max_id)

    def delete_doc(self, term_or_query, field=None):
        """ 使用字段（unique）或查询结果删除document
        
        Args:
            term_or_query (str or Query object): 为str 时标识唯一字段的短语，为Query 时表示查询
            field (str, optional): Defaults to None. 当term_or_query str 时需要指定 field 值，此值为字段名（unique）
        """ 
        writer = self._get_index_writer()
        if isinstance(term_or_query, Query):
            r = writer.delete_by_query(term_or_query)
        elif isinstance(term_or_query, str) and field is not None:
            r = writer.delete_by_term(field, term_or_query)
        else:
            raise ValueError('value "term_or_query" or "field" error.')
        writer.commit()
        if not r:
            self.logger.debug('Documents will been deleted later.')
        self.logger.debug('%s documents have been deleted.' % r)

    def _validate_doc(self, doc):
        # 将id，role 等字段值转为str
        # 有需要可以添加其他验证或预处理
        for k, v in doc.items():
            if isinstance(v, int):
                doc[k] = str(v)
        return doc
    
    def update_doc(self, doc):
        """ 更新document，index 中不存在此doc 时，则为添加
        
        Args:
            doc (dict): {field: value}, 至少一个字段为 unique。
        """
        doc = self._validate_doc(doc)
        writer = self._get_index_writer()
        try:
            writer.update_document(**doc)
        except:
            writer.cancel()
        else:
            writer.commit()

    def incremental_index(self):
        # 增量索引，适用于新增数据做增量索引。更新的数据，在更新时使用 update_doc 方法。
        if not self.index:
            raise ValueError('索引文件不存在。')

        if not self.db:
            self.connect_db()

        writer = self._get_index_writer()
        max_id = self._get_indexed_id_from_db()
        if not max_id:
            raise ValueError('查询 max_indexed_id 出错。')
        cur = self._read_from_db(max_id=max_id)

        max_id = 0
        count = cur.rowcount
        r = cur.fetchone()
        try:
            while r:
                writer.add_document(id=str(r[0]), name=r[1], address=r[2], role=str(r[3]), intro=r[4], date=r[5])
                if int(r[0]) > max_id:
                    max_id = r[0]
                r = cur.fetchone()
        except:
            writer.cancel()
            return
        else:
            writer.commit()
            self._store_indexed_id_to_db(max_id)
            return count

    def connect_db(self, user='root', password='0000', db='oo', **kw):
        self.db = pymysql.connect(user=user, password=password, db=db, **kw)
    
    def close_db(self):
        if self.db:
            self.db.close()

    def _read_from_db(self, max_id=None):
        # 获取所有数据（max_id=None）/未索引的数据，返回指针对象
        cur = self.db.cursor()
        if max_id is None:
            sql = "select * from fake"
        else:
            sql = "select * from fake where id > %s" % max_id
        r = cur.execute(sql)
        if r:
            return cur

    def _get_indexed_id_from_db(self):
        # 存在一个数据表 table_indexed(id int, table_name varchar(50), max_indexed_id int, index_date datetime)
        # 查询此数据表，获得已建立索引的最大行（最大id）
        cur = self.db.cursor()
        sql = "select max_indexed_id from table_indexed where table_name='fake'"
        r = cur.execute(sql)
        if r:
            return cur.fetchone()[0]

    def _store_indexed_id_to_db(self, max_id):
        # 将已索引的最大id 存入 table_indexed 表中
        cur = self.db.cursor()
        sql = "insert into table_indexed (table_name, max_indexed_id) values ('fake', %s)" % max_id
        try:
            cur.execute(sql)
        except:
            self.logger.error('储存数据出错，sql语句：%s' % sql)
            self.db.rollback()
        else:
            self.db.commit()
