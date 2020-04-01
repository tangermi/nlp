# -*- coding: utf-8 -*-
# @author: NiHao

""" how to index documents """

from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.index import create_in

schema = Schema(path=ID(unique=True, stored=True), content=TEXT)

# create
ix = create_in('./index', schema, indexname='index_3')
writer = ix.writer()
writer.add_document(path='/a', content='The first document.')
writer.add_document(path='/b', content='The second content.')
writer.commit()


def query(field, str, name):
    q = QueryParser(field, schema).parse(str)
    s = ix.searcher()
    r = s.search(q)
    print(name, r)
    s.close()


query('path', '/a', 'create:/a')


# update
writer = ix.writer()
writer.update_document(path='/a', content='The third document replace first once.')
writer.commit()
query('path', '/a', 'update:/a')
query('content', 'third', 'update:/a')


# delete
writer = ix.writer()
writer.delete_by_term('path', '/a')
writer.commit()
query('path', '/a', 'delete:/a')
query('path', '/b', 'delete:/a search/b')


# clear
from whoosh import writing
writer = ix.writer()
writer.commit(mergetype=writing.CLEAR)
query('path', '/b', 'clear:all')
