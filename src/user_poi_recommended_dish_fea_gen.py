#encoding=utf8

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime

################################################################################
# 本段代码用于生成 用户推荐菜, 商家推荐菜, 菜品搭配 特征
################################################################################
def process(user_recommended_dish_url, merchant_recommended_dish_url, dish_processed_url, deal_poi_url, user_poi_recommend_dish_fea_url):

    ## 只挑选出 deal_id 和 poi 两个字段, 这两个字段也是我们特征生成的结果数据的 key
    df_deal_poi = pd.read_csv(deal_poi_url, sep='\t')
    df_deal_poi = df_deal_poi[['deal_id', 'poi_id']]

if __name__ == '__main__':
    start_time = datetime.now()

    ## 用户推荐菜 url
    user_recommended_dish_url = '../raw_data/user_recommended_dish.txt'
    ## 商家推荐菜 url
    merchant_recommended_dish_url = '../raw_data/merchant_recommended_dish.txt'
    ## 菜品搭配 url
    dish_train_processed_url = '../feature/dish_train_processed.txt'
    dish_test_processed_url = '../feature/dish_test_processed.txt'

    ## 套餐销量和套餐信息表
    deal_poi_train_url = '../raw_data/deal_sales_train.txt'
    deal_poi_test_url = '../raw_data/deal_poi_test.txt'

    ## 特征结果写入表
    user_poi_recommended_dish_train_fea_url = '../feature/user_poi_rec_dish_train_fea.txt'
    user_poi_recommended_dish_test_fea_url = '../feature/user_poi_rec_dish_test_fea.txt'

    if len(sys.argv) != 9:
        print(sys.argv[0] + ' [user_recommended_dish_url] [merchant_recommended_dish_url] [dish_train_processed_url] [dish_test_processed_url] [deal_poi_train_url] [deal_poi_test_url] [user_poi_recommended_dish_train_fea_url] [user_poi_recommended_dish_test_fea_url]')
    else:
        user_recommended_dish_url = sys.argv[1]
        merchant_recommended_dish_url = sys.argv[2]

        dish_train_processed_url = sys.argv[3]
        dish_test_processed_url = sys.argv[4]

        deal_poi_train_url = sys.argv[5]
        deal_poi_test_url = sys.argv[6]

        user_poi_recommended_dish_train_fea_url = sys.argv[7]
        user_poi_recommended_dish_test_fea_url = sys.argv[8]

    end_time = datetime.now()
    print('- deal_train_test proprocessing time : ', str(end_time - start_time))
