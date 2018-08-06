#encoding=utf8

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime

################################################################################
# 目前暂时不处理 time_available字段
# - time_available:  时间段数据
#                    可能出现多条时间段: '17:00~21:00,11:00~14:00'
#                    可能出现汉字标识跨日情况: '10:00~次日03:00'
# - day_unavailable: 日期列表
#                    可能罗列多天数据: '2018-02-15,2018-02-18'
# - weekday_unavailable: 不可用天数
#                        数据类型: '1,2,3,4', 表征该套餐在某天不可食用
################################################################################

## 计算一天内套餐可用的时间之和（一天按16小时算，>=16小时为1）
def count_time_available(time_available):

    if (time_available != time_available or pd.isnull(time_available)):
        return 1

    time_available = time_available.replace(u'：', ':')
    time_available = time_available.replace('.', ':')

    count = 0
    for i in time_available.split(','):
        start, end = i.split('~')

        start_hour, start_min = start.split(':')
        end_hour, end_min = end.split(':')
        if (len(start_hour) > 2):
            start_hour = 24 + int(start_hour[2:])
        if (len(end_hour) > 2): # e. 次日02:00 or 凌晨02:00
            end_hour = 24 + int(end_hour[2:])
        count += int(end_hour) - int(start_hour)
        count += (int(end_min) - int(start_min)) / 60
    if (count / 16) > 1:
        return 1
    return count / 16


## 计算商家上线两个月之内实际运营天数(即除去day_unavailable的天数)
def get_valid_day_num(day_unavailable, begin_date, weekday_unavailable):

    ## 如果begin_date为空，则返回默认的60日的结果
    if not isinstance(begin_date, str):
        return 60

    ## 如果day_unavailable为空，且weekday_unavailable为空则返回60日的结果
    if not isinstance(day_unavailable, str) and pd.isnull(weekday_unavailable):
        return 60

    ## 如果day_unavailable为空
    if not isinstance(day_unavailable, str):
        unavailable_date_set = []
    else:
        unavailable_date_range = pd.date_range(start=day_unavailable.split(',')[0], end=day_unavailable.split(',')[1])
        unavailable_date_set = set(unavailable_date_range)

    ## 如果weekday_unavailable为空
    if (pd.isnull(weekday_unavailable)):
        unavailable_weekdays = []
    else:
        unavailable_weekdays = weekday_unavailable.split(',')


    valid_date_range = pd.date_range(start=begin_date, periods=60)
    valid_date_set = set(valid_date_range)

    count = 0

    for date_str in valid_date_set:
        if date_str in unavailable_date_set:
            count += 1
        elif str(date_str.weekday() + 1) in unavailable_weekdays:
            count += 1

    return 60 - count
'''
def get_valid_day_num(day_unavailable, begin_date):
    ## 如果begin_date为空，则返回默认的60日的结果
    if not isinstance(begin_date, str):
        return 60

    ## 如果day_unavailable为空，则返回60日的结果
    if not isinstance(day_unavailable, str):
        return 60

    valid_date_range = pd.date_range(start=begin_date, periods=60)
    valid_date_set = set(valid_date_range)

    count = 0
    for date_str in day_unavailable.split(','):
        if datetime.strptime(date_str, '%Y-%m-%d') in valid_date_set:
            count += 1

    return 60 - count
'''

## 计算商家上线之后周几不营业
def find_kth_day(weekday_unavailable, k):
    if not isinstance(weekday_unavailable, str):
        return 1

    weekday_str_list = weekday_unavailable.split(",")
    weekday_int_list = [int(x) for x in weekday_str_list]

    for x in weekday_int_list:
        if x == k:
            return 0
    return 1

## 计算deal在周末是否可用
def weekend_available(weekday_unavailable):
    if pd.isnull(weekday_unavailable):
        return 1
    if "6,7" in weekday_unavailable:
        return 0
    return 1

## 数据处理入口，传入数据url，传出写入文件url
def process(deal_url, deal_processed_url):
    df_deal = pd.read_csv(deal_url, sep='\t')

    ## 处理数据使得begin_date格式正确
    df_deal['new_begin_date'] = df_deal.begin_date.apply(lambda x : x.split(' ')[0])
    del df_deal['begin_date']
    df_deal.rename(columns={'new_begin_date':'begin_date'}, inplace=True)

    ## 处理day_unavailable: 输出两个月的可用天数是多少
    df_deal['available_day_num'] = df_deal.apply(lambda row: get_valid_day_num(row['day_unavailable'], row['begin_date'], row['weekday_unavailable']), axis=1)

    ## weekday_unavailable: 直接解析列成七列标识
    for k in range(1, 8):
        df_deal['available_in_' + str(k)] = df_deal.weekday_unavailable.apply(lambda x : find_kth_day(x, k))

    ## 套餐对比原来价格的折扣
    df_deal['discount_rate'] = df_deal.price / deal_test.market_price

    ## 套餐与原来价格的差价
    df_deal['discount'] = df_deal.market_price - deal_test.price

    ## 一天内套餐可用的时间之和
    df_deal['count_time_available'] = df_deal.time_available.apply(count_time_available)

    ## 套餐周末是否可用
    df_deal['weekend_available'] = df_deal.weekday_unavailable.apply(weekend_available)

    ## 删除time_available, day_unavailable, weekday_unavailable 三个字段
    del df_deal['time_available']
    del df_deal['day_unavailable']
    del df_deal['weekday_unavailable']

    df_deal.to_csv(deal_processed_url, sep='\t', index=False)

if __name__ == '__main__':

    start_time = datetime.now()

    deal_train_url = '../raw_data/deal_train.txt'
    deal_train_processed_url = '../feature/deal_train_processed.txt'
    deal_test_url = '../raw_data/deal_test.txt'
    deal_test_processed_url = '../feature/deal_test_processed.txt'

    if len(sys.argv) != 5:
        print(sys.argv[0] + ' [deal_train_url] [deal_train_processed_url] [deal_test_url] [deal_test_processed_url]')
    else:
        deal_train_url = sys.argv[1]
        deal_train_processed_url = sys.argv[2]
        deal_test_url = sys.argv[3]
        deal_test_processed_url = sys.argv[4]

    print('- now processing the train data ... ')
    process(deal_train_url, deal_train_processed_url)
    print('- now processing the test data ... ')
    process(deal_test_url, deal_test_processed_url)

    end_time = datetime.now()
    print('- deal_train_test proprocessing time : ', str(end_time - start_time))
