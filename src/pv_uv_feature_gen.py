#encoding=utf8

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime

################################################################################
# 判断 partition_date 是否在 begin_date 之前的k天以内
# 例如: begin_date_str = ‘2018-08-02’ partition_date_str = ‘2018-08-01’则
#      结果返回 1
################################################################################
def compute_range(begin_date_str, partition_date_str):
    if (not isinstance(begin_date_str, str)) or (not isinstance(partition_date_str, str)):
        return 999999

    begin_date = datetime.strptime(begin_date_str, '%Y-%m-%d')
    partition_date = datetime.strptime(partition_date_str, '%Y-%m-%d')
    return (begin_date - partition_date).days

################################################################################
# 本函数用于构建pv,uv特征, 读入数据为:
#     - deal_url : 存储[经过处理]的 deal_train 和 deal_test 数据
#                  需要利用到其中的 deal_id 和 begin_date 字段
#     - deal_poi_url : deal_id, poi_id, sales (只有train包含)
#     - deal_poi_pv_url : 用户访问数据
#     - fea_url : 最终生成特征存储地址
#
# 最终构成的特征 Schema 如下(大写字母表示 KEY):
#    [DEAL_ID] [POI_ID] [1d_pv] [1d_uv] [2d_pv] [2d_uv] ... [30d_pv] [30d_uv]
################################################################################

def process(deal_url, deal_poi_url, deal_poi_pv_url, fea_url, day_list):
    ## 只挑选出 deal_id 和 begin_date 两个字段
    df_deal = pd.read_csv(deal_url, sep='\t')
    df_deal = df_deal[['deal_id', 'begin_date']]

    ## 只挑选出 deal_id 和 poi 两个字段, 这两个字段也是我们特征生成的结果数据的 key
    df_deal_poi = pd.read_csv(deal_poi_url, sep='\t')
    df_deal_poi = df_deal_poi[['deal_id', 'poi_id']]

    df_deal_poi_bd = pd.merge(df_deal_poi, df_deal, on=['deal_id'], how='left')

    ## 用户访问数据
    df_deal_poi_pv = pd.read_csv(deal_poi_pv_url, sep='\t')

    df_deal_poi_bd_pv = pd.merge(df_deal_poi_bd, df_deal_poi_pv, on=['deal_id', 'poi_id'], how='left')
    df_deal_poi_bd_pv['delta'] = df_deal_poi_bd_pv.apply(lambda row: compute_range(row['begin_date'], row['partition_date']), axis=1)

    for k in day_list:
        df_deal_poi_bd_pv_k = df_deal_poi_bd_pv[(df_deal_poi_bd_pv['delta'] < k) & (df_deal_poi_bd_pv['delta'] >= 0)]

        del df_deal_poi_bd_pv_k['begin_date']
        del df_deal_poi_bd_pv_k['partition_date']
        del df_deal_poi_bd_pv_k['delta']

        df_deal_poi_bd_pv_k_res = df_deal_poi_bd_pv_k.groupby(['deal_id', 'poi_id']).mean().reset_index()
        df_deal_poi = pd.merge(df_deal_poi, df_deal_poi_bd_pv_k_res, on=['deal_id', 'poi_id'], how='left')
        df_deal_poi.rename(columns={'poi_pv' : str(k)+'_day_mean_pv', 'poi_uv' : str(k)+'_day_mean_uv'}, inplace=True)

    df_deal_poi.to_csv(fea_url, sep='\t', index=False)

if __name__ == '__main__':

    start_time = datetime.now()

    deal_train_url = '../feature/deal_train_processed.txt'
    deal_poi_train_url = '../raw_data/deal_sales_train.txt'
    deal_poi_pv_train_url = '../raw_data/poi_deal_pv_train.txt'
    pv_uv_train_fea_url = '../feature/pv_uv_train_fea.txt'

    deal_test_url = '../feature/deal_test_processed.txt'
    deal_poi_test_url = '../raw_data/deal_poi_test.txt'
    deal_poi_pv_test_url = '../raw_data/poi_deal_pv_test.txt'
    pv_uv_test_fea_url = '../feature/pv_uv_test_fea.txt'

    if len(sys.argv) != 9:
        print(sys.argv[0] + ' [deal_train_url] [deal_poi_train_url] [deal_poi_pv_train_url] [pv_uv_train_fea_url] [deal_test_url] [deal_poi_test_url] [deal_poi_pv_test_url] [pv_uv_test_fea_url]')
    else:
        deal_train_url = sys.arg[1]
        deal_poi_train_url = sys.arg[2]
        deal_poi_pv_train_url = sys.arg[3]
        pv_uv_train_fea_url = sys.arg[4]

        deal_test_url = sys.arg[5]
        deal_poi_test_url = sys.arg[6]
        deal_poi_pv_test_url = sys.arg[7]
        pv_uv_test_fea_url = sys.arg[8]

    day_list = range(1, 31)

    process(deal_train_url, deal_poi_train_url, deal_poi_pv_train_url, pv_uv_train_fea_url, day_list)
    process(deal_test_url, deal_poi_test_url, deal_poi_pv_test_url, pv_uv_test_fea_url, day_list)

    end_time = datetime.now()
    print('- deal_train_test proprocessing time : ', str(end_time - start_time))
