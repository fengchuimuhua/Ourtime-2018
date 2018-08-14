import pandas as pd
import numpy as np
import math

def get_feature_str(fea_list):
    res_str = fea_list[0]
    for x in fea_list[1:]:
        res_str += "_"
        res_str += x
    res_str += "_sales"
    return res_str

def df_groupby_feature(df_deal_poi_sales, fea_list):
    res_df = df_deal_poi_sales.groupby(fea_list).sales.mean().reset_index()
    res_df.rename(columns={'sales':get_feature_str(fea_list)}, inplace=True)
    return res_df

################################################################################
# 生成poi_info特征（这里把categorical特征处理成为int类型）
# 生成交叉特征，具体见代码描述
################################################################################
def process(poi_info_new_url, deal_train_processed_url, deal_sales_train_url, deal_test_url, poi_info_train_with_cross_fea_url, poi_info_test_with_cross_fea_url):
    df_poi_info_new = pd.read_csv(poi_info_new_url, sep='\t')

    df_deal_train_processed = pd.read_csv(deal_train_processed_url, sep='\t')
    df_deal_sales_train = pd.read_csv(deal_sales_train_url, sep='\t')

    df_deal_sales_train = pd.merge(df_deal_sales_train, df_deal_train_processed[['deal_id', 'begin_date']], on=['deal_id'], how='left')
    df_deal_sales_train = df_deal_sales_train[df_deal_sales_train['begin_date'] >= '2017-06-01']
    del df_deal_sales_train['begin_date']

    df_deal_poi_sales_train = pd.merge(df_deal_sales_train, df_poi_info_new, on=['poi_id'], how='left')
    df_deal_poi_sales_train['sales'] = np.log(df_deal_poi_sales_train['sales'])

    df_deal_poi_sales_train[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']] = df_deal_poi_sales_train[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']].fillna(value=-999)

    df_deal_poi_sales_train[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']] = df_deal_poi_sales_train[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']].astype(int)

    df_deal_sales_train_whole = pd.read_csv(deal_sales_train_url, sep='\t')
    df_deal_poi_sales_train_whole = pd.merge(df_deal_sales_train_whole, df_poi_info_new, on=['poi_id'], how='left')
    df_deal_poi_sales_train_whole['sales'] = np.log(df_deal_poi_sales_train_whole['sales'])

    df_deal_poi_sales_train_whole[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']] = df_deal_poi_sales_train_whole[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']].fillna(value=-999)

    df_deal_poi_sales_train_whole[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']] = df_deal_poi_sales_train_whole[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']].astype(int)

    df_deal_test = pd.read_csv(deal_test_url, sep='\t')
    df_deal_poi_test = pd.merge(df_deal_test, df_poi_info_new, on=['poi_id'], how='left')

    df_deal_poi_test[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']] = df_deal_poi_test[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']].fillna(value=-999)

    df_deal_poi_test[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']] = df_deal_poi_test[['mt_poi_cate2_name',
                   'has_parking_area',
                   'has_booth',
                   'is_dining',
                   'barea_id',
                   'poi_rank_value',
                   'is_poi_rank_nan',
                   'is_dp_score_nan',
                   'is_mt_score_nan',
                   'is_poi_zlf_nan',
                   'dp_score_rank',
                   'mt_score_rank',
                   'num_of_poi_in_barea']].astype(int)

    ## 交叉特征生成
    cross_fea_list= [['mt_poi_cate2_name'],
                 ['barea_id'],
                 ['poi_rank_value'],
                 ['dp_score_rank'],
                 ['mt_score_rank'],
                 ['has_parking_area'],
                 ['num_of_poi_in_barea'],
                 ['mt_poi_cate2_name','has_parking_area'],
                 ['mt_poi_cate2_name','has_booth'],
                 ['mt_poi_cate2_name','is_dining'],
                 ['mt_poi_cate2_name','barea_id'],
                 ['mt_poi_cate2_name','poi_rank_value'],
                 ['mt_poi_cate2_name','dp_score_rank'],
                 ['mt_poi_cate2_name','mt_score_rank',],
                 ['mt_poi_cate2_name','num_of_poi_in_barea'],
                 ['barea_id','poi_rank_value'],
                 ['barea_id','dp_score_rank'],
                 ['barea_id','mt_score_rank'],
                 ['barea_id','has_parking_area'],
                 ['barea_id','has_booth'],
                 ['barea_id','is_dining'],
                 ['barea_id','num_of_poi_in_barea'],
                 ['poi_rank_value','dp_score_rank'],
                 ['poi_rank_value','mt_score_rank'],
                 ['poi_rank_value','has_parking_area'],
                 ['poi_rank_value','has_booth'],
                 ['poi_rank_value','is_dining'],
                 ['poi_rank_value','num_of_poi_in_barea'],
                 ['dp_score_rank','mt_score_rank'],
                 ['dp_score_rank','has_parking_area'],
                 ['dp_score_rank','has_booth'],
                 ['dp_score_rank','is_dining'],
                 ['dp_score_rank','num_of_poi_in_barea'],
                 ['mt_score_rank','has_parking_area'],
                 ['mt_score_rank','has_booth'],
                 ['mt_score_rank','is_dining'],
                 ['mt_score_rank','num_of_poi_in_barea'],
                 ['has_parking_area','is_dining'],
                 ['has_parking_area','num_of_poi_in_barea'],
                 ['has_booth','num_of_poi_in_barea'],
                 ['mt_poi_cate2_name','barea_id','poi_rank_value'],
                 ['mt_poi_cate2_name','barea_id','dp_score_rank'],
                 ['mt_poi_cate2_name','barea_id','mt_score_rank'],
                 ['mt_poi_cate2_name','barea_id','has_parking_area'],
                 ['mt_poi_cate2_name','barea_id','has_booth'],
                 ['mt_poi_cate2_name','barea_id','is_dining'],
                 ['mt_poi_cate2_name','barea_id','num_of_poi_in_barea']]
                 
    # cross_fea_list= [['mt_poi_cate2_name'],
    #              ['barea_id'],
    #              ['poi_rank_value'],
    #              ['dp_score_rank'],
    #              ['mt_score_rank'],
    #              ['mt_poi_cate2_name','barea_id'],
    #              ['mt_poi_cate2_name','poi_rank_value'],
    #              ['mt_poi_cate2_name','dp_score_rank'],
    #              ['mt_poi_cate2_name','mt_score_rank',],
    #              ['barea_id','poi_rank_value'],
    #              ['barea_id','dp_score_rank'],
    #              ['barea_id','mt_score_rank'],
    #              ['poi_rank_value','dp_score_rank'],
    #              ['poi_rank_value','mt_score_rank'],
    #              ['dp_score_rank','mt_score_rank'],
    #              ['mt_poi_cate2_name','barea_id','poi_rank_value'],
    #              ['mt_poi_cate2_name','barea_id','dp_score_rank'],
    #              ['mt_poi_cate2_name','barea_id','mt_score_rank']]

    for fea_list in cross_fea_list:
        df_tmp = df_groupby_feature(df_deal_poi_sales_train, fea_list)
        df_deal_poi_sales_train_whole = pd.merge(df_deal_poi_sales_train_whole, df_tmp, on=fea_list, how='left')
        df_deal_poi_test = pd.merge(df_deal_poi_test, df_tmp, on=fea_list, how='left')

    del df_deal_poi_sales_train_whole['sales']

    df_deal_poi_sales_train_whole.to_csv(poi_info_train_with_cross_fea_url, sep='\t', index=False)
    df_deal_poi_test.to_csv(poi_info_test_with_cross_fea_url, sep='\t', index=False)

if __name__ == '__main__':

    start_time = datetime.now()

    poi_info_new_url = '../feature/poi_info_second_processed.txt'
    deal_train_processed_url = '../feature/deal_train_processed.txt'
    deal_sales_train_url = '../raw_data/deal_sales_train.txt'
    deal_test_url = '../raw_data/deal_poi_test.txt'

    poi_info_train_with_cross_fea_url = '../feature/poi_info_train_with_cross_fea.txt'
    poi_info_test_with_cross_fea_url = '../feature/poi_info_test_with_cross_fea.txt'

    if len(sys.argv) != 7:
        print(sys.argv[0] + ' [poi_info_new_url] [deal_train_processed_url] [deal_sales_train_url] [deal_test_url] [deal_test_url] [poi_info_train_with_cross_fea_url] [poi_info_test_with_cross_fea_url]')
    else:
        poi_info_new_url = sys.argv[1]
        deal_train_processed_url = sys.argv[2]
        deal_sales_train_url = sys.argv[3]
        deal_test_url = sys.argv[4]

        poi_info_train_with_cross_fea_url = sys.argv[5]
        poi_info_test_with_cross_fea_url = sys.argv[6]

    process(poi_info_new_url, deal_train_processed_url, deal_sales_train_url, deal_test_url, poi_info_train_with_cross_fea_url, poi_info_test_with_cross_fea_url)

    end_time = datetime.now()
    print('- deal_train_test proprocessing time : ', str(end_time - start_time))
