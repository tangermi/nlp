# -*- coding: utf-8 -*-
# @author: NiHao

import pymysql
from faker import Faker


def fake_data(count):
    f = Faker(locale='zh_CN')

    db = pymysql.connect(host='localhost', user='root', password='0000', db='oo')
    cur = db.cursor()

    drop_table = 'DROP TABLE IF EXISTS fake;'
    create_table = """
            CREATE TABLE fake (
                id int(11) primary key auto_increment,
                name varchar(32),
                address varchar(64),
                role tinyint,
                intro text,
                date datetime) 
                ENGINE=InnoDB DEFAULT CHARSET=UTF8MB4;
        """
    cur.execute(drop_table)
    cur.execute(create_table)

    sql = 'INSERT INTO fake(name, address, role, intro, date) values (%s, %s, %s, %s, %s)'

    for _ in range(count):
        try:
            cur.execute(sql, (f.name(),
                              f.country() + ' ' + f.city(),
                              f.random_int(1, 5),
                              f.text(),
                              f.date()))
            db.commit()
        except:
            db.rollback()
    db.close()


if __name__ == '__main__':
    fake_data(10000)