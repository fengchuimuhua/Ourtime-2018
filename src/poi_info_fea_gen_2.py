import pandas as pd
import numpy as np

################################################################
#
# 运行完 poi_info_fea_gen.py 之后运行 poi_info_fea_gen_2.py
# 把"mt_score" “dp_score” 缺失其中之一的按照等比转换
# 加入"is_miss_score"字段指示分数都为0的情况 
#
################################################################

def process(poi_info_processed_url, poi_info_processed_2_url):
    df_poi_info = pd.read_csv(poi_info_processed_url, sep='\t')

    df_poi_info.loc[(df_poi_info['mt_score']>0) & (df_poi_info['dp_score']==0),'dp_score'] = df_poi_info.mt_score * 2
    df_poi_info.loc[(df_poi_info['mt_score']==0) & (df_poi_info['dp_score']>0),'mt_score'] = df_poi_info.dp_score / 2

    miss_ls = (df_poi_info["dp_evn_score"]==0) | (df_poi_info["dp_taste_score"]==0) | (df_poi_info["dp_service_score"]==0) | \
    (df_poi_info["dp_service_score"]==0) | (df_poi_info["dp_star"]==0)

    df_poi_info["is_miss_score"] = 0

    df_poi_info.loc[miss_ls, "is_miss_score"] = 1

    df_poi_info.to_csv(poi_info_processed_2_url, sep='\t', index=False)

if __name__=="__main__":
    poi_info_train_processed_url = "../feature/poi_info_train_fea.txt"
    poi_info_train_processed_2_url = "../feature/poi_info_train_fea_2.txt"

    poi_info_test_processed_url = "../feature/poi_info_test_fea.txt"
    poi_info_test_processed_2_url = "../feature/poi_info_test_fea_2.txt"

    process(poi_info_train_processed_url, poi_info_train_processed_2_url)
    process(poi_info_test_processed_url, poi_info_test_processed_2_url)