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
