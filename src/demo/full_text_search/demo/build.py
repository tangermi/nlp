import os.path
import os

from jieba.analyse import ChineseAnalyzer
from whoosh.fields import Schema, TEXT, ID, KEYWORD, STORED
from whoosh.index import create_in


class FullTextBuild:

    def __init__(self, indexpath, indexname=None):
        self.indexpath = indexpath
        self.indexname = indexname
        self.index = None
        self.__build_index()

    def __build_index(self):
        analyzer = ChineseAnalyzer()

        schema = Schema(title=TEXT(stored=True), path=ID(stored=True),
                        content=TEXT(stored=True, analyzer=analyzer), tags=KEYWORD, icon=STORED)

        if not os.path.exists(self.indexpath):
            os.mkdir(self.indexpath)
        self.index = create_in(self.indexpath, schema)

    def add_doc(self, *, title, path, content, tags, icon):
        writer = self.index.writer()
        writer.add_document(
            title=title,
            path=path,
            content=content,
            tags=tags,
            icon=icon
        )
        writer.commit()



if __name__ == '__main__':
    pass