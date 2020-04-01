# -*- coding:utf-8 -*-


def display_result(final_result):
    for i in range(len(final_result)):
        if final_result[i] == '一':
            print('')
        elif final_result[i] != '。':
            print(final_result[i], end='')
        else:
            print('。')



