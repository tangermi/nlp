from whoosh.index import create_in
from whoosh.fields import *
from whoosh.analysis import RegexTokenizer, LowercaseFilter , StopFilter, StemFilter
from json2dict import passages_collection_100000   # a collection of the top 100000 elements in the origin dataset
import os

my_analyzer = analysis.StemmingAnalyzer()   # a pre-packaged analyzer that combines a tokenizer, lower-case filter, optional stop filter, and stem filter
schema = Schema(title=TEXT(stored=True,analyzer=my_analyzer,field_boost=3.0), url=ID(stored=True), content=TEXT(stored=True,analyzer=my_analyzer), id=ID(unique=True))

if not os.path.exists("D:\lizi\indexdir_100000"):
    os.mkdir("D:\lizi\indexdir_100000")
ix = create_in("D:\lizi\indexdir_100000",schema)
writer = ix.writer()

for passage in passages_collection_100000['passages']:
    writer.update_document(title=passage['title'],url=passage['url'],id=passage['id'],content=passage['passage_text'])

writer.commit()

