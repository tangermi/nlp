# -*- coding: utf-8 -*-
# @author: NiHao

import os
import datetime

import pymysql

from searsphi.log import logger
from sphinxapi import SphinxClient


settings = dict(
    host='lo'
)
HOST = 'localhost'
PORT = 9312

class Sphinx(object):
    """ 使用sphinx 建立索引 """
    def __init__(self, host='localhost', port=9306, **kwargs):
        self.sc = SphinxClient()
        self.connect = pymysql.connect(host=host, port=port, **kwargs)
        self.cur = self.connect.cursor()

    def modify_settings(self, **kw):
        host = kw.get('host', 'localhost')
        port = kw.get('port', 9312)
        self.sc.SetServer(host, port=port)
        # 其他配置

    def search(self, word, index='*', page=1, page_offset=0, page_limit=20, sort=None, filters=None):
        # filter: list(dict(attr=[min_, max_]))，eg. [{'id': [3, 10], '_exclude': False}, {'tag': 'abc'}, {'tag': 'abc', '_exclude': False}]
        # sort: str, startswith '-' : desc, else asc

        # 设置结果的页数等
        offset = page * page_offset
        limit = page_limit
        self.sc.SetLimits(offset, limit)

        # 设置排序
        if isinstance(sort, str):
            if sort.startswith('-'):
                sort_mode = 1   # 'SPH_SORT_ATTR_DESC'
            else:
                sort_mode = 2   # 'SPH_SORT_ATTR_ASC'
        self.sc.SetSortMode(sort_mode, clause=sort)

        # 设置过滤器
        if isinstance(filters, list):
            for filter in filters:
                if len(filter) != 2:
                    raise ValueError('a filter should only has two keys: attr and "_exclude".')
                for k, v in filter.items():
                    exclude = 0
                    if k != '_exclude':
                        exclude = int(v)
                        continue
                    attr = k
                    if isinstance(v, list) and len(v) == 2:
                        self.sc.SetFilterRange(attr, v[0], v[1], exclude=exclude)
                    elif isinstance(v, str):
                        self.sc.SetFilterString(attr, v, exclude=exclude)
                    else:
                        raise ValueError('parse filter error')

        r = self.sc.Query(word, index=index)

        # 重置过滤器
        self.sc.ResetFilters()
        # BuildExcerpts (self, docs, index, words, opts=None)   高亮文档
        # SetMaxQueryTime (maxquerytime))
        # SetSelect(select)
        # SetMatchMode(mode)
        # SetFilterRange(attr, min_, max_, exclude=0)
        # SetFilterString(attr, value, exclude=0)     # exclude=0 : 只包含 value； exclude=1: 不包含value
        # Query(query, index='*', comment="")
        return r

    def results_to_ids(self, results):
        try:
            r = [str(item['id']) for item in results['matches']]
        except:
            raise ValueError('param "results" parse error')
        else:
            return r