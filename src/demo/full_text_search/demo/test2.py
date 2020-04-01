import os


class searchengine:

    def __init__(self, model, updatefield, indexpath=None, indexname=None, formatter=None):
        self.model = model
        self.indexpath = indexpath
        self.indexname = indexname
        self.updatefield = updatefield
        self.indexschema = {}
        self.formatter = BlogFormatter
        # 建立index存放路径
        if self.indexpath is None:
            self.indexpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'engineindex/')
        if self.indexname is None:
            self.indexname = model.__name__
        if formatter is not None:
            self.formatter = formatter
        self.__buildSchema()
        self.__buildindex()


    from django.db.models import *
    from whoosh.fields import *
    from whoosh.index import create_in, exists, exists_in
    from whoosh.filedb.filestore import FileStorage
    from ckeditor_uploader.fields import RichTextUploadingField

    # 为某个model建立schema
    def __buildSchema(self):
        self.indexschema = {}
        modlefields = self.model._meta.get_fields()
        for field in modlefields:
            if type(field) == CharField:
                self.indexschema[field.__str__().split('.')[-1]] = TEXT(stored=True)
            elif type(field) == IntegerField:
                self.indexschema[field.__str__().split('.')[-1]] = NUMERIC(stored=True, numtype=int)
            elif type(field) == FloatField:
                self.indexschema[field.__str__().split('.')[-1]] = NUMERIC(stored=True, numtype=float)
            elif type(field) == DateField or type(field) == DateTimeField:
                self.indexschema[field.__str__().split('.')[-1]] = DATETIME(stored=True)
            elif type(field) == BooleanField:
                self.indexschema[field.__str__().split('.')[-1]] = BOOLEAN(stored=True)
            elif type(field) == AutoField:
                self.indexschema[field.__str__().split('.')[-1]] = STORED()
            elif type(field) == RichTextUploadingField:
                self.indexschema[field.__str__().split('.')[-1]] = TEXT(stored=True)

    def __buildindex(self):
        # schemadict = self.__buildSchema()
        document_dic = {}
        # defaultFolderPath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'engineindex/')
        if self.indexschema is None:
            return False

        if not os.path.exists(self.indexpath):
            os.mkdir(self.indexpath)

        modelSchema = Schema(**self.indexschema)
        if not exists_in(self.indexpath, indexname=self.indexname):
            ix = create_in(self.indexpath, modelSchema, indexname=self.indexname)
            print('index is created')
            writer = ix.writer()
            # 将model对象依次加入index中
            objectlist = self.model.objects.all()
            for obj in objectlist:
                for key in self.indexschema:
                    if hasattr(obj, key):
                        # print(key,getattr(obj,key.split('.')[-1]))
                        document_dic[key] = getattr(obj, key)
                writer.add_document(**document_dic)
                document_dic.clear()
            writer.commit()
            print('all blog has indexed')


    def __addonedoc(self, writer, docId):
        print('docId is %s' % docId)
        obj = self.model.objects.get(id=docId)
        document_dic = {}
        print('enter __addonedoc')
        for key in self.indexschema:
            print('key in __addonedoc is %s' % key)
            print(key)
            if hasattr(obj, key):
                document_dic[key] = getattr(obj, key)
        print(document_dic)
        writer.add_document(**document_dic)


    def search(self, searchfield, searchkeyword, ignoretypo=False):
        storage = FileStorage(self.indexpath)
        ix = storage.open_index(indexname=self.indexname)
        if isinstance(searchfield, str):
            qp = QueryParser(searchfield, schema=self.indexschema, group=OrGroup)
        elif isinstance(searchfield, list):
            qp = MultifieldParser(searchfield, schema=self.indexschema)
        q = qp.parse(searchkeyword)
        resultobjlist = []
        corrected_dict = {}
        with ix.searcher() as searcher:
            corrected = searcher.correct_query(q, searchkeyword)
            if corrected.query != q and ignoretypo == False:
                q = qp.parse(corrected.string)
                corrected_dict = {'corrected': u'您要找的是不是' + corrected.string}
            results = searcher.search(q, limit=None)
            # results.formatter = BlogFormatter()
            results.formatter = self.formatter()
            for result in results:
                obj_dict = {}
                highlightresults = []
                for key in result:
                    obj_dict[key] = result[key]
                if isinstance(searchfield, str):
                    highlightresults.append({searchfield: '<' + result.highlights(searchfield) + '>'})
                elif isinstance(searchfield, list):
                    for _field in searchfield:
                        highlightresults.append({_field: '<' + result.highlights(_field) + '>'})
                obj_dict['highlight'] = highlightresults

                extradata_dic = self.extradata()
                if len(extradata_dic) > 0:
                    obj_dict.update(**extradata_dic)
                resultobjlist.append(obj_dict)
        storage.close()
        return resultobjlist, corrected_dict





