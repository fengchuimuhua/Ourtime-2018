#encoding=utf8

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime

################################################################################
# 本函数用于构建 poi info 特征, 读入数据为:
#     - nodup_poi_info_url : 经过处理的 poi_info 表, 并且poi_id不重复的部分
#     - dup_poi_info_url : 经过处理的 poi_info 表, 并且poi_id有重复然后手动删除重复条目的部分
# (上面两个文件有需要手动操作删除数据的部分，这部分数据可以直接找我取)
#     - deal_poi_url : 对于训练集是deal_sales_train，对于预测集是deal_poi_test
#     - poi_info_fea_url : 特征写入url
# 最终构成的特征 Schema 如下(大写字母表示 KEY)，实际上即把 poi_info 转到 deal_id * poi_id维度中去:
#    [DEAL_ID] [POI_ID] [mt_poi_cate2_name] [price_person] ... [is_poi_rank_nan]
################################################################################
def process(poi_info_processed_url, deal_poi_url, poi_info_fea_url):
    df_poi_info = pd.read_csv(poi_info_processed_url, sep='\t')

    ## 只挑选出 deal_id 和 poi 两个字段, 这两个字段也是我们特征生成的结果数据的 key
    df_deal_poi = pd.read_csv(deal_poi_url, sep='\t')
    df_deal_poi = df_deal_poi[['deal_id', 'poi_id']]

    df_poi_info_fea = pd.merge(df_deal_poi, df_poi_info, on=['poi_id'], how='left')
    df_poi_info_fea.to_csv(poi_info_fea_url, sep='\t', index=False)

if __name__ == '__main__':

    start_time = datetime.now()

    ## input table url
    poi_info_processed_url = '../feature/poi_info_second_processed.txt'
    deal_poi_train_url = '../raw_data/deal_sales_train.txt'
    deal_poi_test_url = '../raw_data/deal_poi_test.txt'

    ## output table url
    poi_info_train_fea_url = '../feature/poi_info_train_fea.txt'
    poi_info_test_fea_url = '../feature/poi_info_test_fea.txt'

    if len(sys.argv) != 6:
        print(sys.argv[0] + ' [poi_info_processed_url] [deal_poi_train_url] [deal_poi_test_url] [poi_info_train_fea_url] [poi_info_test_fea_url]')
    else:
        poi_info_processed_url = sys.argv[1]
        deal_poi_train_url = sys.argv[2]
        deal_poi_test_url = sys.argv[3]
        poi_info_train_fea_url = sys.argv[4]
        poi_info_test_fea_url = sys.argv[5]

    ## generate poi_info train feature
    process(poi_info_processed_url, deal_poi_train_url, poi_info_train_fea_url)
    ## generate poi_info test feature
    process(poi_info_processed_url, deal_poi_test_url, poi_info_test_fea_url)

    end_time = datetime.now()
    print('- poi_info proprocessing time : ', str(end_time - start_time))
