# -*- coding: utf-8 -*-
# @author: NiHao

""" 中文搜索 """
import os

import pymysql
from whoosh.fields import Schema, TEXT, ID, DATETIME
from whoosh.index import create_in, open_dir, exists_in
from whoosh.qparser import QueryParser
from jieba.analyse import ChineseAnalyzer


def read_from_db():
    db = pymysql.connect(user='root', password='0000', db='oo')
    cur = db.cursor()
    sql = "select * from fake"
    r = cur.execute(sql)
    if r:
        return cur


def create_index(path):
    if not os.path.exists(path):
        os.mkdir(path)

    analyzer = ChineseAnalyzer()
    schema = Schema(id=ID(stored=True, unique=True, sortable=True),
                    name=TEXT(stored=True, analyzer=analyzer),
                    address=TEXT(analyzer=analyzer),
                    role=ID(sortable=True),
                    intro=TEXT(analyzer=analyzer),
                    date=DATETIME(stored=True, sortable=True))

    ix = create_in(path, schema)
    writer = ix.writer()

    cur = read_from_db()
    r = cur.fetchone()
    while r:
        writer.add_document(id=str(r[0]), name=r[1], address=r[2], role=str(r[3]), intro=r[4], date=r[5])
        r = cur.fetchone()
    writer.commit()
    return ix


def open_index(path):
    try:
        ix = open_dir(path)
    except:
        return
    else:
        return ix


if __name__ == '__main__':
    # ix = create_index('index6')
    # print(exists_in('index6'))
    # ix = open_index('index6')
    # s = ix.searcher()
    # q = QueryParser('name', ix.schema).parse('路')
    # r = s.search(q)
    print('西欧')