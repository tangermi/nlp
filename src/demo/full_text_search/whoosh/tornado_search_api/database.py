# -*- coding: utf-8 -*-
# @author: NiHao

import datetime

import pymysql

from log import logger


class Db(object):
    def __init__(self, host='localhost', port=3306, **kwargs):
        self.connect = pymysql.connect(host=host, port=port, **kwargs)
        self.cur = self.connect.cursor()

    def select(self, ids, table, column=None, **kwargs):
        """ 查询数据，只以 id 字段作为查询条件，按ids 中id 的顺序排序

        :param ids: list(str or int), ids list
        :param table: str, table name
        :param column: list(str), a list with table column names
        :return: list(result), result rows
        """
        date_s = kwargs.get('date_s')
        date_e = kwargs.get('date_e')
            
        if column is None:
            column = ['*']

        c = ', '.join(['%s' for _ in column])
        ids_string = ', '.join([str(i) for i in ids])

        select = "SELECT {columns} FROM {table} ".format(columns=c, table=table)
        where_1 = " WHERE id IN ({ids}) ".format(ids=ids_string)
        where_2 = ""
        order_by = " ORDER BY FIELD(id, {ids}) ".format(ids=ids_string)

        if date_s is None and date_e is None:
            pass
        elif isinstance(date_s, datetime.datetime) and isinstance(date_e, datetime.datetime):
            where_2 = " AND '{date_s}' <= date AND date <= '{date_e}' ".format(date_s=date_s, date_e=date_e)
        else:
            self.logger.error('param "date_s" and "date_e" must be "datetime.datetime" object.')

        sql = select + where_1 + where_2 + order_by
        sql = sql % tuple(column)
        try:
            r = self.cur.execute(sql)
        except Exception as e:
            self.logger.error('select data error traceback: %s' % e)
            return []
        else:
            return self.cur.fetchall()

    def insert(self, table, data):
        """ 插入数据

        :param table: str, table name
        :param data: dict(column=value), data to insert
        :return: int or None, Number of affected rows
        """

        keys = ', '.join(data.keys())
        values = ', '.join(['%s'] * len(data))
        sql = """INSERT INTO {table}({keys}) VALUES({values})""".\
            format(table=table, keys=keys, values=values)

        try:
            result = self.cur.execute(sql, tuple(data.values()))
            if result:
                self.logger.info('insert %s row successful.', result)
                self.connect.commit()
                return result
        except Exception as e:
            self.logger.error('insert data error traceback: %s' % e)
            self.connect.rollback()
            return

    def get_last_insert_id(self):
        """ 返回最近插入数据的id 每个connection 的值相互独立，不区分table """
        self.cur.execute('SELECT LAST_INSERT_ID()')
        r = self.cur.fetchone()
        if not r or r[0] == 0:
            return
        return r[0]

    def update(self, ids, table, data):
        """ 更新数据，只以 id 字段作为筛选条件

        :param ids: [int or str] , list of row id
        :param table: str, table name
        :param data: dict(column=value), data to update
        :return: int or None, Number of affected rows
        """

        update = ', '.join(['{key}=%s'.format(key=key) for key in data.keys()])
        ids_string = ', '.join([str(i) for i in ids])
        sql = """UPDATE {table} SET {update} WHERE id IN ({ids})""".\
            format(table=table, update=update, ids=ids_string)

        try:
            result = self.cur.execute(sql, tuple(data.values()))
            if result:
                self.logger.info('update %s row successful.', result)
                self.connect.commit()
                return result
        except Exception as e:
            self.logger.error('update data error traceback: %s' % e)
            self.connect.rollback()
            return

    def delete(self, ids, table):
        """ 删除数据，只以 id 字段作为筛选条件

        :param ids: [str or int], list of row id
        :param table: str, table name
        :return: int or None, Number of affected rows
        """
        ids_str = ', '.join([str(i) for i in ids])
        sql = """DELETE FROM {table} WHERE id IN ({ids})""".format(table=table, ids=ids_str)

        try:
            result = self.cur.execute(sql)
            if result:
                self.connect.commit()
            return result
        except Exception as e:
            self.logger.error('delete data error traceback: %s' % e)
            self.connect.rollback()
            return

    def create_table(self, table, column=None, charset='utf8mb4'):
        """ 测试用，修改后需调整测试程序 """
        if column is None:
            column = 'id int primary key auto_increment, name varchar(128), summary text, ' \
                     'detail text, date datetime, author varchar(128), tag varchar(128)'

        sql = """CREATE TABLE {table} ({column}) DEFAULT CHARSET={charset}""".\
            format(table=table, column=column, charset=charset)
        self.cur.execute(sql)
        # print('创建表格成功：%s' % table)
        return True

    def drop_table(self, table):
        """ 测试用，修改后需调整测试程序 """
        sql = "DROP TABLE IF EXISTS %s" % table
        self.cur.execute(sql)
        # print('已删除表格：%s' % table)
        return True
