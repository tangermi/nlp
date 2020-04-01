# -*- coding: utf-8 -*-
# @author: NiHao

""" quick start """

from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser

schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT)
index = create_in('./quick_start/index', schema, indexname='index_1')

writer = index.writer()

writer.add_document(title='row a', path='/blog/a',
                    content='Whoosh is a library of classes and functions '
                            'for indexing text and then searching the index. ')

writer.add_document(title='row b', path='/blog/b',
                    content='It allows you to develop custom search engines for your content.')

writer.add_document(title='row ab', path='/blog/ab',
                    content='For example, if you were creating blogging software.')

writer.add_document(title='row ac', path='/blog/ac',
                    content='To begin using Whoosh, you need an index object.')
writer.commit()
# with index.searcher() as searcher:
searcher = index.searcher()
query = QueryParser('content', index.schema).parse('whoosh')
results = searcher.search(query)
print(results)
