#encoding=utf8

import sys
import pandas as pd
import numpy as np
import math
from datetime import datetime

def get_poi_rank_value(poi_rank):
    if poi_rank == 'KA':
        return 4
    elif poi_rank == 'A':
        return 3
    elif poi_rank == 'B':
        return 2
    elif poi_rank == 'C':
        return 1
    else:
        return 0

def is_poi_rank_nan(poi_rank):
    if isinstance(poi_rank, str):
        return 0
    else:
        return 1

## 手动处理缺失值(optional)
def fill_in_nan(poi_info):
    poi_info.mt_score = poi_info.mt_score.replace(np.nan, poi_info[poi_info.mt_score > 0].mt_score.mean())
    poi_info.dp_score = poi_info.dp_score.replace(np.nan, poi_info[poi_info.dp_score > 0].dp_score.mean())
    poi_info.dp_star = poi_info.dp_star.replace(np.nan, poi_info.dp_star.mode()[0])
    poi_info.dp_evn_score = poi_info.dp_evn_score.replace(np.nan, poi_info[poi_info.dp_evn_score > 0].dp_evn_score.mean())
    poi_info.dp_taste_score = poi_info.dp_taste_score.replace(np.nan, poi_info[poi_info.dp_taste_score > 0].dp_taste_score.mean())
    poi_info.dp_service_score = poi_info.dp_service_score.replace(np.nan, poi_info[poi_info.dp_service_score > 0].dp_service_score.mean())
    poi_info.poi_zlf = poi_info.poi_zlf.replace(np.nan, poi_info[poi_info.poi_zlf > 0].poi_zlf.mean())
    poi_info.poi_rank = poi_info.poi_rank.replace(np.nan, poi_info.poi_rank.mode()[0])


def process(poi_info_url, poi_info_processed_url):
    df_poi_info = pd.read_csv(poi_info_url, sep='\t')
    df_poi_info['poi_rank_value'] = df_poi_info.poi_rank.apply(get_poi_rank_value)
    df_poi_info['is_poi_rank_nan'] = df_poi_info.poi_rank.apply(is_poi_rank_nan)
    del df_poi_info['poi_rank']
    df_poi_info.to_csv(poi_info_processed_url, sep='\t', index=False)


if __name__ == '__main__':

    start_time = datetime.now()

    poi_info_url = '../raw_data/poi_info.txt'
    poi_info_processed_url = '../feature/poi_info_processed.txt'

    if len(sys.argv) != 3:
        print(sys.argv[0] + '\t[poi_info_url]\t[poi_info_processed_url]')
    else:
        poi_info_url = sys.argv[1]
        poi_info_processed_url = sys.argv[2]

    process(poi_info_url, poi_info_processed_url)

    end_time = datetime.now()
    print('- poi_info proprocessing time : ', str(end_time - start_time))
