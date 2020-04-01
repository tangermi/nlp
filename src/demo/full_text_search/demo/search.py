from h.qparser import QueryParser

class FullTextSearch:

    def __init__(self):
        self.idx = None
        pass

    def load(self, index):
        self.index = index

    def search(self, keyword):
        searcher = self.index.searcher()
        parser = QueryParser("content", schema=self.index.schema)
        print("result of ", keyword)
        q = parser.parse(keyword)
        results = searcher.search(q)
        return results


if __name__ == '__main__':
    pass