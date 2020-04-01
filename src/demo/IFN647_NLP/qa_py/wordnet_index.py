from whoosh.lang.wordnet import Thesaurus
from whoosh.filedb.filestore import FileStorage
import os

f = open("wordnet/wn_s.pl")
t = Thesaurus.from_file(f)

if not os.path.exists("D:\lizi\index_wordnet"):
    os.mkdir("D:\lizi\index_wordnet")

fs = FileStorage("D:\lizi\index_wordnet")
t.to_storage(fs)

# test
# t_test = Thesaurus.from_storage(fs)
# print(t_test.synonyms("hail"))