#encoding=utf8

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime

def find_kth_tag(dish_tag_str, k):
    if not isinstance(dish_tag_str, str):
        return 0
    tag_str_list = dish_tag_str.split("|")
    tag_int_list = [int(x) for x in tag_str_list]
    res_bool = False
    for x in tag_int_list:
        if x == k:
            return 1
    return 0

def find_nan_tag(dish_tag):
    if not isinstance(dish_tag, str):
        return 1
    else:
        return 0

def process(dish_url, dish_processed_url):
    df_dish = pd.read_csv(dish_url, sep='\t')

    for k in range(1, 14):
        df_dish['dish_tag_' + str(k)] = df_dish.dish_tag.apply(lambda x : find_kth_tag(x, k))

    ## 该列用于标注dish_tag是否为空, 若值为1则为空，否则不为空
    df_dish['dish_tag_14'] = df_dish.dish_tag.apply(find_nan_tag)

    ## 删除原始的 dish_tag 列
    del df_dish['dish_tag']

    df_dish.to_csv(dish_processed_url, sep='\t', index=False)


if __name__ == '__main__':

    start_time = datetime.now()

    dish_train_url = '../raw_data/dish_train.txt'
    dish_test_url = '../raw_data/dish_test.txt'

    dish_train_processed_url = '../feature/dish_train_processed.txt'
    dish_test_processed_url = '../feature/dish_test_processed.txt'

    if len(sys.argv) != 5:
        print(sys.argv[0] + '\t[dish_train_url]\t[dish_test_url]\t[dish_train_processed_url]\t[dish_test_processed_url]')
    else:
        dish_train_url = sys.argv[1]
        dish_test_url = sys.argv[2]
        dish_train_processed_url = sys.argv[3]
        dish_test_processed_url = sys.argv[4]

    process(dish_train_url, dish_train_processed_url)
    process(dish_test_url, dish_test_processed_url)

    end_time = datetime.now()
    print('- dish_train_test proprocessing time : ', str(end_time - start_time))
