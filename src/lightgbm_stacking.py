import pandas as pd
import numpy as np
import warnings
import time
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

def get_oof(clf, X_train, y_train, X_test):

    ntrain = X_train.shape[0]
    ntest = X_test.shape[0]

    folds = list(KFold(n_splits=5, shuffle=True).split(X_train, y_train))

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))

    for i, (train_index, test_index) in enumerate(folds):
        kf_X_train = X_train.iloc[train_index]
        kf_y_train = y_train.iloc[train_index]
        kf_X_test = X_train.iloc[test_index]

        model = clf.fit(kf_X_train, kf_y_train)

        oof_train[test_index] = model.predict(kf_X_test)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

################################################################################
#
# 生成Stacking必须的若干文件
#
################################################################################
def process(pv_uv_train_fea_url, poi_info_train_with_cross_fea_url, deal_train_fea_url, user_poi_rec_dish_train_fea_url, deal_sales_train_url, pv_uv_test_fea_url, poi_info_test_with_cross_fea_url, deal_test_fea_url, user_poi_rec_dish_test_fea_url, deal_poi_test_url):

    df_train_1 = pd.read_csv(pv_uv_train_fea_url, sep='\t')
    df_train_2 = pd.read_csv(poi_info_train_with_cross_fea_url, sep='\t')
    df_train_3 = pd.read_csv(deal_train_fea_url, sep='\t')
    df_train_4 = pd.read_csv(user_poi_rec_dish_train_fea_url, sep='\t')
    del df_train_4['sales']

    df_train_label = pd.read_csv(deal_sales_train_url, sep='\t')

    df_train_label['sales'] = np.log(df_train_label['sales'])

    df_tmp = pd.merge(df_train_1, df_train_2, on=['deal_id', 'poi_id'])
    df_tmp = pd.merge(df_tmp, df_train_3, on=['deal_id', 'poi_id'])
    df_tmp = pd.merge(df_tmp, df_train_4, on=['deal_id', 'poi_id'])
    df_final = pd.merge(df_tmp, df_train_label, on=['deal_id', 'poi_id'])

    del df_final['deal_id']
    del df_final['poi_id']

    df_test_1 = pd.read_csv(pv_uv_test_fea_url, sep='\t')
    df_test_2 = pd.read_csv(poi_info_test_with_cross_fea_url, sep='\t')
    df_test_3 = pd.read_csv(deal_test_fea_url, sep='\t')
    df_test_4 = pd.read_csv(user_poi_rec_dish_test_fea_url, sep='\t')

    df_test_label = pd.read_csv(deal_poi_test_url, sep='\t')

    df_tmp_te = pd.merge(df_test_1, df_test_2, on=['deal_id', 'poi_id'])
    df_tmp_te = pd.merge(df_tmp_te, df_test_3, on=['deal_id', 'poi_id'])
    df_tmp_te = pd.merge(df_tmp_te, df_test_4, on=['deal_id', 'poi_id'])

    df_final_te = pd.merge(df_tmp_te, df_test_label, on=['deal_id', 'poi_id'])

    del df_final_te['deal_id']
    del df_final_te['poi_id']

    ## 构建 LightGBM 训练需要的数据
    X = df_final.drop(['sales'], axis=1)
    y = df_final.sales

    categorical_features = [c for c, col in enumerate(X.columns) if 'cat' in col]
    train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_features)

    X_train = X
    y_train = y
    X_test = df_final_te

    from lightgbm import LGBMRegressor

    curr_iter = 1

    y_train.to_csv('../stacking_result/oof_train_label.txt', sep='\t', index=True)

    for seed in np.random.randint(0, 10000000, 5):
        print('-- current iteration : ', curr_iter)

        lgb_params = {
            'metric': 'mae',
            'boosting': 'gbdt',
            'learning_rate': 0.15,
            'verbose': 5,
            'num_threads':4,

            'num_leaves': int(round(103.94959609449087)),
            'feature_fraction': 0.8817362739132675,
            'bagging_fraction': 0.9938541288208969,
            'max_depth': int(round(6.997036554415374)),
            'lambda_l1': 4.155258706464617,
            'lambda_l2': 0.10586791933130013,
            'min_split_gain': 0.029005622004794077,
            'min_child_weight': 14.00378847316659,
            'seed' : seed
        }

        lgb_model = LGBMRegressor(**lgb_params, num_boost_round=2000, verbose_eval=5)

        (oof_train, oof_test) = get_oof(lgb_model, X_train, y_train, X_test)

        df_oof_train = pd.DataFrame({'col_'+str(curr_iter) : list(oof_train)})
        df_oof_test = pd.DataFrame({'col_'+str(curr_iter) : list(oof_test)})

        df_oof_train.to_csv('../stacking_result/oof_train_'+str(curr_iter)+'.txt', sep='\t', index=True)
        df_oof_test.to_csv('../stacking_result/oof_test_'+str(curr_iter)+'.txt', sep='\t', index=True)

        curr_iter += 1

if __name__ == '__main__':

    pv_uv_train_fea_url = '../feature/pv_uv_train_fea.txt'
    poi_info_train_with_cross_fea_url = '../feature/poi_info_train_with_cross_fea.txt'
    deal_train_fea_url = '../feature/deal_train_fea.txt'
    user_poi_rec_dish_train_fea_url = '../feature/user_poi_rec_dish_train_fea.txt'
    deal_sales_train_url = '../raw_data/deal_sales_train.txt'

    pv_uv_test_fea_url = '../feature/pv_uv_test_fea.txt'
    poi_info_test_with_cross_fea_url = '../feature/poi_info_test_with_cross_fea.txt'
    deal_test_fea_url = '../feature/deal_test_fea.txt'
    user_poi_rec_dish_test_fea_url = '../feature/user_poi_rec_dish_test_fea.txt'
    deal_poi_test_url = '../raw_data/deal_poi_test.txt'

    start_time = datetime.now()

    if len(sys.argv) != 11:
        print(sys.argv[0] + ' [pv_uv_train_fea_url] [poi_info_train_with_cross_fea_url] [deal_train_fea_url] [user_poi_rec_dish_train_fea_url] [deal_sales_train_url] [pv_uv_test_fea_url] [poi_info_test_with_cross_fea_url] [deal_test_fea_url] [user_poi_rec_dish_test_fea_url] [deal_poi_test_url]')
    else:
        pv_uv_train_fea_url = sys.argv[1]
        poi_info_train_with_cross_fea_url = sys.argv[2]
        deal_train_fea_url = sys.argv[3]
        user_poi_rec_dish_train_fea_url = sys.argv[4]
        deal_sales_train_url = sys.argv[5]

        pv_uv_test_fea_url = sys.argv[6]
        poi_info_test_with_cross_fea_url = sys.argv[7]
        deal_test_fea_url = sys.argv[8]
        user_poi_rec_dish_test_fea_url = sys.argv[9]
        deal_poi_test_url = sys.argv[10]

    process(pv_uv_train_fea_url, poi_info_train_with_cross_fea_url, deal_train_fea_url, user_poi_rec_dish_train_fea_url, deal_sales_train_url, pv_uv_test_fea_url, poi_info_test_with_cross_fea_url, deal_test_fea_url, user_poi_rec_dish_test_fea_url, deal_poi_test_url)

    end_time = datetime.now()
    print('- stacking file generation time : ', str(end_time - start_time))
