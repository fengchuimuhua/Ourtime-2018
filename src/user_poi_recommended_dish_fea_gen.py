#encoding=utf8

import pandas as pd
import numpy as np
import sys
from datetime import datetime

################################################################################
# 目的：融合 菜品搭配 和 套餐销量（Train）或者套餐与门店关系（Test）和 用户推荐菜 和 商家推荐菜
# 先综合四张表，获取在一个门店下，一个套餐中的一个菜品所对应的信息。再降为去掉menu_name信息，只保留 poi_id 和 deal_id
#
# 输入：各表的url地址，tag为"train"或者"test"
#
# 最终构成的特征为（KEY为大写）:
# POI_ID DEAL_ID dishes_price dish_tag user_recommended merchant_recommended rec_cnt
#
# 数据处理方式（可进一步优化）：
# 1. 去掉了 商家推荐菜 中的price信息，因为已经包含了菜品信息中的菜品price且不包含nan值
# 2. 取同一个 deal_id 中 dishes_price 的和
# 3. 去掉了 dishes_group_name 不知道怎么用
# 4. dish_tag_? 取并集，deal中的dish出现某个tag，deal中对应的tag就是1
# 5. 存在deal_id找不到poi_id的情况，去掉这部分数据
# 6. merchant_recommended 和 user_recommended 求和，推荐菜在deal中所占有的比例
# 7. rec_cnt 求和，推荐菜的推荐的总次数
################################################################################

def process(dish_processed_url, deal_poi_url, user_recommended_dish_url, merchant_recommended_dish_url, user_poi_recommend_dish_fea_url):
    df_dish = pd.read_csv(dish_processed_url, sep='\t')
    df_deal_poi = pd.read_csv(deal_poi_url, sep='\t')

    df_dish_dealPoi = pd.merge(df_dish, df_deal_poi, on="deal_id", how="outer")

    df_user_recommend_dish = pd.read_csv(user_recommended_dish_url, sep='\t')
    df_merchant_recommend_dish = pd.read_csv(merchant_recommended_dish_url, sep='\t')

    df_poi_deal_dish_fea = merge_dish_dealPoi_and_recommended_dish(df_dish_dealPoi, df_user_recommend_dish, "user")
    df_poi_deal_dish_fea = merge_dish_dealPoi_and_recommended_dish(df_poi_deal_dish_fea, df_merchant_recommend_dish, "merchant")

    df_poi_deal_dish_fea.dropna(subset=["poi_id"], inplace=True)

    df_poi_deal_dish_fea.drop(["dishes_group_name", "menu_name", "price"], axis=1, inplace=True)

    tag_ls = list(map(lambda x: 'dish_tag_' + str(x), list(range(1,15)))) + ["user_recommended","merchant_recommended","rec_cnt","dishes_price"]

    keep_ls = ["poi_id", "deal_id"]

    df_user_poi_recommend_dish_fea = df_poi_deal_dish_fea.groupby(keep_ls)[tag_ls].sum()

    for k in range(1, 15):
        tag_str = 'dish_tag_' + str(k)
        df_user_poi_recommend_dish_fea[tag_str] = df_user_poi_recommend_dish_fea[tag_str].apply(dish_tag_setter)

    df_user_poi_recommend_dish_fea.to_csv(user_poi_recommend_dish_fea_url, sep='\t', index=True)


def merge_dish_dealPoi_and_recommended_dish(df_dish_dealPoi, recommended_dish, tag):
    tag_str = tag + "_recommended"
    recommended_dish[tag_str] = 1

    df_dish_dealPoi_recommend = pd.merge(df_dish_dealPoi, recommended_dish, on=["menu_name","poi_id"], how="left")
    df_dish_dealPoi_recommend.loc[df_dish_dealPoi_recommend[tag_str].isnull(), tag_str] = 0

    return df_dish_dealPoi_recommend

def dish_tag_setter(x):
    if x>=1:
        return 1
    return 0

if __name__ == "__main__":
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
    user_poi_recommend_dish_fea_train_url = '../feature/user_poi_rec_dish_train_fea.txt'
    user_poi_recommend_dish_fea_test_url = '../feature/user_poi_rec_dish_test_fea.txt'

    if len(sys.argv) != 9:
        print(sys.argv[0] + ' [user_recommended_dish_url] [merchant_recommended_dish_url] [dish_train_processed_url] [dish_test_processed_url] [deal_poi_train_url] [deal_poi_test_url] [user_poi_recommend_dish_fea_train_url] [user_poi_recommend_dish_fea_test_url]')
    else:
        user_recommended_dish_url = sys.argv[1]
        merchant_recommended_dish_url = sys.argv[2]

        dish_train_processed_url = sys.argv[3]
        dish_test_processed_url = sys.argv[4]

        deal_poi_train_url = sys.argv[5]
        deal_poi_test_url = sys.argv[6]

        user_poi_recommend_dish_fea_train_url = sys.argv[7]
        user_poi_recommend_dish_fea_test_url = sys.argv[8]

    process(dish_train_processed_url, deal_poi_train_url, user_recommended_dish_url, merchant_recommended_dish_url, user_poi_recommend_dish_fea_train_url)
    process(dish_test_processed_url, deal_poi_test_url, user_recommended_dish_url, merchant_recommended_dish_url, user_poi_recommend_dish_fea_test_url)

    end_time = datetime.now()
    print('- deal_train_test proprocessing time : ', str(end_time - start_time))
