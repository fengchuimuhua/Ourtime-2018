#encoding=utf8

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime

################################################################################
# 本文件用于处理../feature/poi_info_processed.txt
# 输出文件为../feature/poi_info_second_processed.txt
# 为后续的poi_info_fea_gen.py提供数据
#
# 目前做的工作有：
# 处理poi_info文件的缺失值，归一化数据和增加商圈相关数据
################################################################################

def rank_mt_score(x):
    if pd.isnull(x) or x < 0.1:
        return -1
    else:
        for i in [50,45,40,35,30]:
            if x >= i:
                return int(i/5) - 5
    return 1

def rank_dp_score(x):
    if pd.isnull(x) or x < 0.1:
        return -1
    else:
        for i in [90,85,80,75,70]:
            if x >= i:
                return int(i/5) - 13
    return 1

def cal_star(x):
    for i in [50,45,40,35,30,25,20,15,10,0]:
        if x >= i:
            return i

## 处理缺失值，分别处理包含mt_score，不包含dp_score的情况和包含dp_score,不包含mt_score的情况
def deal_with_nan_value(poi_info):
    # has_booth和is_dining的缺失值较少，直接按众数处理，并转为int类型
    poi_info.has_booth = poi_info.has_booth.replace(np.nan, poi_info.has_booth.mode()[0]).astype(np.int64)
    poi_info.is_dining = poi_info.is_dining.replace(np.nan, poi_info.is_dining.mode()[0]).astype(np.int64)

    poi_info.dp_score = poi_info.dp_score.replace(0, np.nan)
    poi_info.mt_score = poi_info.mt_score.replace(0, np.nan)
    poi_info.dp_evn_score = poi_info.dp_evn_score.replace(0, np.nan)
    poi_info.dp_taste_score = poi_info.dp_taste_score.replace(0, np.nan)
    poi_info.dp_service_score = poi_info.dp_service_score.replace(0, np.nan)
    poi_info.dp_star = poi_info.dp_star.replace(0, np.nan)

    # 将df_poi_info按dp_score和mt_score是否为空拆分成四部分
    has_dp_no_mt = poi_info[(poi_info.dp_score>0.1)&(poi_info.mt_score!=poi_info.mt_score)]
    has_mt_no_dp = poi_info[(poi_info.mt_score>0.1)&(poi_info.dp_score!=poi_info.dp_score)]
    no_dp_no_mt = poi_info[(poi_info.mt_score!=poi_info.mt_score)&(poi_info.dp_score!=poi_info.dp_score)]
    has_dp_has_mt = poi_info[(poi_info.mt_score==poi_info.mt_score)&(poi_info.dp_score==poi_info.dp_score)]

    has_dp_no_mt.mt_score = (has_dp_no_mt.dp_score + 5) / 2   # mt_score和dp_score的缺失值补全关系：(人为规定)
    has_mt_no_dp.dp_score = has_mt_no_dp.mt_score * 2 - 5     # mt_score * 2 - 5 = dp_score
    has_mt_no_dp.dp_evn_score = has_mt_no_dp.dp_score
    has_mt_no_dp.dp_taste_score = has_mt_no_dp.dp_score
    has_mt_no_dp.dp_service_score = has_mt_no_dp.dp_score
    has_mt_no_dp.dp_star = has_mt_no_dp.mt_score
    has_mt_no_dp.dp_star = has_mt_no_dp.dp_star.apply(cal_star)

    no_dp_no_mt.mt_score = no_dp_no_mt.mt_score.replace([np.nan, 0], poi_info[poi_info.mt_score > 0].mt_score.mean())
    no_dp_no_mt.dp_score = no_dp_no_mt.dp_score.replace([np.nan, 0], poi_info[poi_info.dp_score > 0].dp_score.mean())
    no_dp_no_mt.dp_star = no_dp_no_mt.dp_star.replace([np.nan, 0], poi_info.dp_star.mode()[0])
    no_dp_no_mt.dp_evn_score = no_dp_no_mt.dp_evn_score.replace([np.nan, 0], poi_info[poi_info.dp_evn_score > 0].dp_evn_score.mean())
    no_dp_no_mt.dp_taste_score = no_dp_no_mt.dp_taste_score.replace([np.nan, 0], poi_info[poi_info.dp_taste_score > 0].dp_taste_score.mean())
    no_dp_no_mt.dp_service_score = no_dp_no_mt.dp_service_score.replace([np.nan, 0], poi_info[poi_info.dp_service_score > 0].dp_service_score.mean())

    df_poi_info = pd.concat([has_dp_has_mt,has_dp_no_mt,has_mt_no_dp,no_dp_no_mt])
    df_poi_info.dp_evn_score = df_poi_info.dp_evn_score.replace(np.nan, df_poi_info.dp_evn_score.mean())
    df_poi_info.dp_taste_score = df_poi_info.dp_taste_score.replace(np.nan, df_poi_info.dp_taste_score.mean())
    df_poi_info.dp_service_score = df_poi_info.dp_service_score.replace(np.nan, df_poi_info.dp_service_score.mean())

    return df_poi_info

# 按barea_id做聚合提取一些有关商圈的特征
def extract_business_district_info(df_poi_info):
    # count poi in barea
    p1 = df_poi_info[['barea_id']]
    p1['num_of_poi_in_barea'] = 1
    p1 = p1.groupby(['barea_id']).agg('sum').reset_index()

    # avg price person in barea
    p2 = df_poi_info[['barea_id','price_person']]
    p2 = p2.groupby(['barea_id']).agg('mean').reset_index()
    p2 = p2.rename(columns={'price_person':'avg_price_person_in_barea'})

    # avg mt_score in barea
    p3 = df_poi_info[['barea_id','mt_score']]
    p3 = p3.groupby(['barea_id']).agg('mean').reset_index()
    p3 = p3.rename(columns={'mt_score':'avg_mt_score_in_barea'})

    # avg dp_score in barea
    p4 = df_poi_info[['barea_id','dp_score']]
    p4 = p4.groupby(['barea_id']).agg('mean').reset_index()
    p4 = p4.rename(columns={'dp_score':'avg_dp_score_in_barea'})

    # avg poi_zlf in barea
    p5 = df_poi_info[['barea_id','poi_zlf']]
    p5 = p5.groupby(['barea_id']).agg('mean').reset_index()
    p5 = p5.rename(columns={'poi_zlf':'avg_poi_zlf_in_barea'})

    df_poi_info = pd.merge(df_poi_info, p1, on='barea_id', how='left')
    df_poi_info = pd.merge(df_poi_info, p2, on='barea_id', how='left')
    df_poi_info = pd.merge(df_poi_info, p3, on='barea_id', how='left')
    df_poi_info = pd.merge(df_poi_info, p4, on='barea_id', how='left')
    df_poi_info = pd.merge(df_poi_info, p5, on='barea_id', how='left')

    return df_poi_info

def process(poi_info_url, poi_info_processed_url):
    df_poi_info = pd.read_csv(poi_info_url, sep='\t')

    # dp_score,dp_star,dp_evn_score,dp_taste_score,dp_service_score的缺失值数量一致，说明只要有一个缺失剩下的都缺失
    df_poi_info['is_dp_score_nan'] = df_poi_info.dp_score.apply(lambda x: 1 if pd.isnull(x) or x == 0 else 0)
    df_poi_info['is_mt_score_nan'] = df_poi_info.mt_score.apply(lambda x: 1 if pd.isnull(x) or x == 0 else 0)
    df_poi_info['is_poi_zlf_nan'] = df_poi_info.poi_zlf.apply(lambda x: 1 if pd.isnull(x) or x == 0 else 0)

    # 给dp_score和mt_score按大小划分为5个等级，5为最高，1为最低，-1为缺失
    df_poi_info['dp_score_rank'] = df_poi_info.dp_score.apply(rank_dp_score)
    df_poi_info['mt_score_rank'] = df_poi_info.mt_score.apply(rank_mt_score)

    # 处理缺失值
    df_poi_info = deal_with_nan_value(df_poi_info)

    # normalize
    df_poi_info.mt_score = (df_poi_info.mt_score - df_poi_info.mt_score.min()) / (df_poi_info.mt_score.max() - df_poi_info.mt_score.min())
    df_poi_info.dp_score = (df_poi_info.dp_score - df_poi_info.dp_score.min()) / (df_poi_info.dp_score.max() - df_poi_info.dp_score.min())
    df_poi_info.dp_star = (df_poi_info.dp_star - df_poi_info.dp_star.min()) / (df_poi_info.dp_star.max() - df_poi_info.dp_star.min())
    df_poi_info.dp_evn_score = (df_poi_info.dp_evn_score - df_poi_info.dp_evn_score.min()) / (df_poi_info.dp_evn_score.max() - df_poi_info.dp_evn_score.min())
    df_poi_info.dp_taste_score = (df_poi_info.dp_taste_score - df_poi_info.dp_taste_score.min()) / (df_poi_info.dp_taste_score.max() - df_poi_info.dp_taste_score.min())
    df_poi_info.dp_service_score = (df_poi_info.dp_service_score - df_poi_info.dp_service_score.min()) / (df_poi_info.dp_service_score.max() - df_poi_info.dp_service_score.min())
    #df_poi_info.poi_zlf = df_poi_info.poi_zlf / 200

    # 提取商圈特征
    df_poi_info = extract_business_district_info(df_poi_info)

    #print(df_poi_info.shape)
    #print(df_poi_info.dtypes)
    df_poi_info.to_csv(poi_info_processed_url, sep='\t', index=False)


if __name__ == '__main__':

    start_time = datetime.now()

    poi_info_url = '../feature/poi_info_processed.txt'
    poi_info_processed_url = '../feature/poi_info_second_processed.txt'

    if len(sys.argv) != 3:
        print(sys.argv[0] + '\t[poi_info_url]\t[poi_info_processed_url]')
    else:
        poi_info_url = sys.argv[1]
        poi_info_processed_url = sys.argv[2]

    process(poi_info_url, poi_info_processed_url)

    end_time = datetime.now()
    print('- poi_info proprocessing time : ', str(end_time - start_time))
