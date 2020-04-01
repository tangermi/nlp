# -*- coding: utf-8 -*-
# @author: NiHao

import datetime

from tornado import ioloop, httpserver, web, escape
from tornado.options import define, options, parse_command_line

from searsphi.search import Sphinx
from searsphi.log import logger
from database import Db


api = [
    "GET     http://host:port                                                               index",
    "GET     http://host:port/search?q=abc&tag=1&date_s=2010-5-9&date_e=2019-9-8&page=1     search",
    "GET     http://host:port/edit                                                          get detail with blank",
    "GET     http://host:port/edit?id=1                                                     get detail with id",
    "POST    http://host:port/edit                                                          insert a new detail",
    "POST    http://host:port/edit?id=1                                                     modify a detail with id",
    "DELETE  http://host:port/edit?id=1                                                     delete a detail with id"
]

define('port', default=8000, type=int)


class IndexHandler(web.RequestHandler):
    def get(self):
        self.write({'code': 0, 'msg': 'hello', 'api': api})


class SearchHandler(web.RequestHandler):
    def get(self):
        # 搜索
        # /search?q=abc&tag=1&date_s=2010-5-9&date_e=2019-9-8&page=1
        results = []
        word = self.get_query_argument('q', default=None)
        tag = self.get_query_argument('tag', default=None)
        date_s = self.get_query_argument('date_s', default=None)
        date_e = self.get_query_argument('date_e', default=None)
        page = int(self.get_query_argument('page', default='1'))
        offset = 10

        if not word:
            self.write({'code': -1, 'msg': 'error', 'reason': 'query argument error'})
            return

        if date_s and date_e:
            try:
                date_s = datetime.datetime.strptime(date_s, '%Y-%m-%d')
                date_e = datetime.datetime.strptime(date_e, '%Y-%m-%d')
            except (ValueError, TypeError):
                self.write({'code': -1, 'msg': 'error', 'reason': 'query argument error'})
                return
        
        ids = self.application.indexer.search(word, tag=tag, page=page, offset=offset)

        if not ids:
            self.write({'code': 0,
                        'msg': 'success',
                        'count': 0,
                        'result': results})
            return

        # 使用id 从数据库查询数据
        r = self.application.db.select(ids, '*', date_s=date_s, date_e=date_e)
        count = len(r)

        for item in r:
            result = {
                'id': item[0],
                'name': item[1],
                'address': item[2],
                'role': item[3],
                'intro': item[4],
                'date': datetime.datetime.strftime(item[5], '%Y-%m-%d'),
                'ts': datetime.datetime.timestamp(item[5])
            }
            results.append(result)
        self.write({'code': 0,
                    'msg': 'success',
                    'page': page,
                    'count': count,
                    'result': results})


class EditHandler(web.RequestHandler):
    """ 编辑 """
    def get(self):
        # '/edit'   edit page
        # '/edit?id=123'    get edit page with detail

        id_ = self.get_query_argument('id', default=None)

        if id_:
            try:
                self.application.db.select([id_], 'fake')

                item = self.application.db.cur.fetchone()

                result = {
                    'id': item[0],
                    'name': item[1],
                    'address': item[2],
                    'role': item[3],
                    'intro': item[4],
                    'date': datetime.datetime.strftime(item[5], '%Y-%m-%d'),
                    'ts': datetime.datetime.timestamp(item[5])
                }
            except Exception as e:
                self.logger.error('error in format data, traceback: %s', e)
                self.write({'code': -1, 'msg': 'error', 'reason': 'not found'})
            else:
                self.write({'code': 0, 'msg': 'success', 'result': result})
        else:
            self.write({'code': 0, 'msg': 'success'})

    def post(self):
        # '/edit'   add data
        # '/edit?id=123'    edit data
        id_ = self.get_query_argument('id', default=None)

        # 获取form 的表单信息（json），加载 json
        try:
            data = escape.json_decode(self.request.body)
        except:
            self.write({'code': -1, 'msg': 'error', 'reason': 'form data type error'})
            return

        if id_:
            r = self.application.db.select([id_], 'fake')
            if not r:
                self.write({'code': -1, 'msg': 'error', 'reason': 'not found'})
                return

            # ? validate data
            # ? validate_data(table_name, data, action)

            try:
                self.application.db.update([id_], 'fake', data)
                self.application.db.connect.commit()
            except:
                self.logger.error("更新数据出错，相关参数：id=%s, table='fake', data=%s" % ([id_], data))
                self.application.db.connect.rollback()
                self.write({'code': -1, 'msg': 'error', 'reason': 'error when save data'})
                return
            else:
                # 更新索引（目前使用了全部的字段），索引可能不需要数据的所有字段
                doc = data.copy()
                doc['id'] = id_
                self.application.indexer.update_doc(doc)
            self.write({'code': 0, 'msg': 'update data success'})
            return
        else:
            # validate data
            # validate_data(table_name, data, action)

            try:
                self.application.db.insert('fake', data)
                self.application.db.connect.commit()
            except:
                self.logger.error("插入数据出错，相关参数：table='fake', data=%s" % (data))
                self.application.db.connect.rollback()
                self.write({'code': -1, 'msg': 'error', 'reason': 'error when save data'})
                return
            else:
                # 返回数据库中的id
                item_id = self.application.db.get_last_insert_id()
                # 添加至索引（目前使用了全部的字段），索引可能不需要数据的所有字段
                doc = data.copy()
                doc['id'] = item_id
                self.application.indexer.update_doc(doc)

            self.write({'code': 0, 'msg': 'insert data success', 'id': item_id})

    def delete(self):
        # '/edit?id=123'
        id_ = self.get_query_argument('id', default=None)
        if not id_:
            self.write({'code': -1, 'msg': 'error', 'reason': 'query argument error'})
            return
        r = self.application.db.select([id_], 'fake')
        if not r:
            self.write({'code': -1, 'msg': 'error', 'reason': 'not found'})
            return

        try:
            self.application.db.delete([id_], 'fake')
            self.application.db.connect.commit()
        except:
            self.logger.error("删除数据出错，相关参数：table='fake', id=%s" % [id_])
            self.application.db.connect.rollback()
            self.write({'code': -1, 'msg': 'error', 'reason': 'error when delete data'})
            return
        else:
            # 从索引中删除
            self.application.indexer.delete_doc(str(id_), field='id')

        self.write({'code': 0, 'msg': 'delete data success'})


class Application(web.Application):
    def __init__(self, db, index_engine):
        self.indexer = index_engine
        self.db = db

        handlers = [
            (r'/', IndexHandler),
            (r'/search', SearchHandler),
            (r'/edit', EditHandler)
        ]

        web.Application.__init__(self, handlers=handlers)


def main():
    parse_command_line()

    index_engine = Sphinx()
    # index_engine.modify_settings(**kw)

    # 启动web服务
    db = Db(user='root', password='0000', db='oo')
    app = Application(db, index_engine)
    server = httpserver.HTTPServer(app)
    server.listen(options.port)
    ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()