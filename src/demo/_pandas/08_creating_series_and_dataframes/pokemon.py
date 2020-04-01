# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np


def creating_series_and_dataframes():
    dict = {'evolution': ['Ivysaur', 'Charmeleon', 'Wartortle', 'Metapod'],
            'hp': [45, 39, 44, 45],
            'name': ['Bulbasaur', 'Charmander', 'Squirtle', 'Caterpie'],
            'pokedex': ['yes', 'no', 'yes', 'no'],
            'type': ['grass', 'fire', 'water', 'bug']}
    pokemon = pd.DataFrame(dict)

    # 把列的排序修改为  name, type, hp, evolution, pokedex
    pokemon = pokemon[['name', 'type', 'hp', 'evolution', 'pokedex']]

    # 新增一列place
    place = ['park', 'street', 'lake', 'forest']
    pokemon['place'] = place

    # 查看数据的类型
    print(pokemon.dtypes)


if __name__ == '__main__':
    creating_series_and_dataframes()
