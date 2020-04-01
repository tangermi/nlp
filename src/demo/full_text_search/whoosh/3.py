# -*- coding: utf-8 -*-
# @author: NiHao

""" how to search """

import pymysql
from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, TEXT, ID, DATETIME


def read_from_db():
    db = pymysql.connect(user='root', password='0000', db='oo')
    cur = db.cursor()
    sql = "select * from fake"
    r = cur.execute(sql)
    if r:
        return cur


def create_index():
    schema = Schema(id=ID(stored=True, unique=True, sortable=True),
                    name=TEXT(stored=True),
                    address=TEXT,
                    role=ID(sortable=True),
                    intro=TEXT,
                    date=DATETIME(stored=True, sortable=True))

    ix = index.create_in('./whoosh/study/index', schema, indexname='fake')
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
        ix = index.open_dir('./index', indexname='fake')
    except:
        return
    else:
        return ix


if __name__ == "__main__":
    ix = open_index()
    # ix = create_index()
    
    with ix.searcher() as s:
        q = QueryParser('intro', ix.schema).parse('国内')
        r = s.search(q)
        print(r)