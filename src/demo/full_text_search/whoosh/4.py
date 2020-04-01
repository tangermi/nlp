# -*- coding: utf-8 -*-
# @author: NiHao

""" whoosh with jieba.analyse """

import pymysql
from jieba.analyse import ChineseAnalyzer
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID, DATETIME
from whoosh.qparser import QueryParser


def read_from_db():
    db = pymysql.connect(user='root', password='0000', db='oo')
    cur = db.cursor()
    sql = "select * from fake"
    r = cur.execute(sql)
    if r:
        return cur


def create_index():
    analyzer = ChineseAnalyzer()
    schema = Schema(id=ID(stored=True, unique=True, sortable=True),
                    name=TEXT(stored=True, analyzer=analyzer),
                    address=TEXT(analyzer=analyzer),
                    role=ID(sortable=True),
                    intro=TEXT(analyzer=analyzer),
                    date=DATETIME(stored=True, sortable=True))

    ix = create_in('index', schema, indexname='fake2')
    writer = ix.writer()

    cur = read_from_db()
    r = cur.fetchone()
    while r:
        writer.add_document(id=str(r[0]), name=r[1], address=r[2], role=str(r[3]), intro=r[4], date=r[5])
        r = cur.fetchone()
    writer.commit()
    return ix


def open_index():
    try:
        ix = open_dir('index', indexname='fake2')
    except:
        return
    else:
        return ix


if __name__ == '__main__':
    # ix = create_index()
    ix = open_index()
    s = ix.searcher()
    parser = QueryParser('role', ix.schema)
    q = parser.parse('1')
    q2 = parser.parse('北京')
    r = s.search(q)
    for rr in r:
        print(rr)
    # print([t.decode('utf-8') for t in list(s.lexicon('role'))])
    s.close()
