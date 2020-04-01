# -*- coding: utf-8 -*-
# @author: NiHao

import unittest
import datetime

from database import Db


class DbTestCase(unittest.TestCase):
    def setUp(self):
        self.db = Db(user='root', password='0000', db='oo')

    def test_create_and_drop_table(self):
        # #
        try:
            self.db.cur.execute('drop table test')
        except:
            pass

        self.db.create_table('test')
        r = self.db.cur.execute('desc test')
        self.assertTrue(r > 0)
        self.db.drop_table('test')

        try:
            self.db.cur.execute('desc test')
            self.assertTrue(False)
        except:
            self.assertTrue(True)

    def test_insert(self):
        self.db.drop_table('test')
        self.db.create_table('test')
        r = self.db.insert('test', {'name': 'xiaowang', 'summary': 'abc', 'date': datetime.datetime.now()})
        self.assertEqual(r, 1)

    def test_select(self):
        self.db.drop_table('test')
        self.db.create_table('test')
        self.db.insert('test', {'name': 'xiaowang', 'summary': 'abc', 'date': datetime.datetime.now()})
        self.db.insert('test', {'name': 'xiao', 'summary': 'ab', 'date': datetime.datetime.now()})
        r = self.db.select([1, 2], 'test')
        self.assertEqual(r, 2)
        result = self.db.cur.fetchone()
        self.assertTrue(result[1] == 'xiaowang')

    def test_update(self):
        self.db.drop_table('test')
        self.db.create_table('test')
        self.db.insert('test', {'name': 'xiaowang', 'summary': 'abc', 'date': datetime.datetime.now()})
        self.db.update([1], 'test', {'name': 'xh'})
        self.db.select([1], 'test')
        r = self.db.cur.fetchone()
        self.assertTrue(r[1] == 'xh')

    def test_delete(self):
        self.db.drop_table('test')
        self.db.create_table('test')
        self.db.insert('test', {'name': 'xiaowang', 'summary': 'abc', 'date': datetime.datetime.now()})
        self.db.insert('test', {'name': 'xiao', 'summary': 'ab', 'date': datetime.datetime.now()})
        r = self.db.delete([1], 'test')

        self.assertEqual(r, 1)
        r2 = self.db.select([1, 2], 'test')
        self.assertEqual(r2, 1)

    def tearDown(self):
        self.db.cur.execute('DROP TABLE IF EXISTS test')
        self.db.connect.close()


if __name__ == '__main__':
    unittest.main()