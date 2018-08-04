#encoding=utf8

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime

################################################################################
# 生成套餐信息特征
################################################################################
def process(deal_url, deal_poi_url, deal_fea_url):
    df_deal = pd.read_csv(deal_url, sep='\t')
    del df_deal['begin_date']

    ## 只挑选出 deal_id 和 poi 两个字段, 这两个字段也是我们特征生成的结果数据的 key
    df_deal_poi = pd.read_csv(deal_poi_url, sep='\t')
    df_deal_poi = df_deal_poi[['deal_id', 'poi_id']]

    df_deal_fea = pd.merge(df_deal_poi, df_deal, on=['deal_id'])
    df_deal_fea.to_csv(deal_fea_url, sep='\t', index=False)

if __name__ == '__main__':

    start_time = datetime.now()

    ## input table url
    deal_train_url = '../feature/deal_train_processed.txt'
    deal_test_url = '../feature/deal_test_processed.txt'
    deal_poi_train_url = '../raw_data/deal_sales_train.txt'
    deal_poi_test_url = '../raw_data/deal_poi_test.txt'

    ## output table url
    deal_train_fea_url = '../feature/deal_train_fea.txt'
    deal_test_fea_url = '../feature/deal_test_fea.txt'

    if len(sys.argv) != 7:
        print(sys.argv[0] + ' [deal_train_url] [deal_test_url] [deal_poi_train_url] [deal_poi_test_url] [deal_train_fea_url] [deal_test_fea_url]')
    else:
        deal_train_url = sys.argv[1]
        deal_test_url = sys.argv[2]
        deal_poi_train_url = sys.argv[3]
        deal_poi_test_url = sys.argv[4]

        deal_train_fea_url = sys.argv[5]
        deal_test_fea_url = sys.argv[6]

    ## generate deal train feature
    process(deal_train_url, deal_poi_train_url, deal_train_fea_url)
    ## generate deal test feature
    process(deal_test_url, deal_poi_test_url, deal_test_fea_url)

    end_time = datetime.now()
    print('- poi_info proprocessing time : ', str(end_time - start_time))
