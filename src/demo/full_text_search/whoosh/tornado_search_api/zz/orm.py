# -*- coding: utf-8 -*-
# @author: NiHao


class Column(object):
    def __init__(self, name, column_type, nullable, default, primary_key):
        self.name = name
        self.column_type = column_type
        self.nullable = nullable
        self.default = default
        self.primary_key = primary_key

    def __repr__(self):
        return '<%s: %s %s %s %s %s>' % (
            self.__class__.__name__,
            self.name,
            self.column_type,
            'null' if self.nullable else 'not null',
            'default={}'.format(self.default),
            'PRI' if self.primary_key else '')


class Integer(Column):
    def __init__(self, name, column_type='int(11)', nullable=True, default=None,
                 primary_key=False, auto_increment=False):
        self.auto_increment = auto_increment
        super(Integer, self).__init__(name, column_type, nullable, default, primary_key)


class String(Column):
    def __init__(self, name, column_type='varchar(100)', nullable=True, default=None,
                 primary_key=False):
        super(String, self).__init__(name, column_type, nullable, default, primary_key)


class Text(Column):
    def __init__(self, name, column_type='text', nullable=True, default=None,
                 primary_key=False):
        super(Text, self).__init__(name, column_type, nullable, default, primary_key)


class Boolean(Column):
    def __init__(self, name, column_type='tinyint(1)', nullable=True, default=None,
                 primary_key=False):
        super(Boolean, self).__init__(name, column_type, nullable, default, primary_key)


class Datetime(Column):
    def __init__(self, name, column_type='datetime', nullable=True, default=None,
                 primary_key=False):
        super(Datetime, self).__init__(name, column_type, nullable, default, primary_key)


if __name__ == '__main__':
    c = Column('name', 'varchar(90)', False, 'li', True)
    print(c.__repr__())
