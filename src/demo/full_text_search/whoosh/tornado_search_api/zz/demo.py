# -*- coding: utf-8 -*-
# @author: NiHao


import datetime as dt

try:
    d = dt.datetime.strptime(None, '%Y-%m-%d')
except (ValueError, TypeError):
    print('typeerror')


"""
from app.search import WhooshIx as Ws
ws = Ws('index', 'fake')

from app.database import Db
db = Db(user='root', password='0000', db='oo')

import datetime as dt
ds = dt.datetime.strptime('1996-1-1', '%Y-%m-%d')
de = dt.datetime.strptime('2013-11-1', '%Y-%m-%d')
r = db.select([7,6,5,4,3,2,1], 'fake', date_s=ds, date_e=de)

r = db.select([6,4,2,1], 'fake')

for i in r['matches']:
    print(i['attrs']['date'])
"""


