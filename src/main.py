import numpy as np
import pandas as pd
import os
import datetime

from core.outer_cv_runner import OuterCVRunner
from core.config import Config

if __name__ == '__main__':

    config = Config()

    # lightGBMによる学習・予測
    run_name = f'lgb_{config.target_dataset}_{config.target_column}_{datetime.datetime.now().strftime("%Y%m%d%H%M")}'

    # モデルディレクトリの作成
    model_dir = f'../model/{run_name}'
    os.makedirs(model_dir, exist_ok=True)

    outer_runner = OuterCVRunner(
            run_name = run_name
        ,model_cls = None
        ,params_dict = {
            'lightgbm': {
                'learning_rate': 0.01,
                'feature_fraction': 0.6,
                'num_leaves': 32,
                'subsample': 0.7,
                'reg_lambda': 1.0,
                'reg_alpha': 1.0,
                'min_data_in_leaf': 50,
            },
            'catboost': {}
        }
        ,cv_seed = config.cv_seed
        ,tuning_seed = config.tuning_seed
        ,model_dir = model_dir
        ,is_tuning = True
        ,train_file_path = f"../data/raw/{config.target_dataset}_{config.target_column}.parquet"
        ,test_file_path = ""
    )
    outer_runner.run_train_cv()

    # outer_runner = OuterCVRunner(
    #      run_name = 'lgb_202502061051_0'
    #     ,model_cls = None
    #     ,params_dict = {'lightgbm': {}, 'catboost': {}}
    #     ,cv_seed = config.cv_seed
    #     ,tuning_seed = config.tuning_seed
    #     ,model_dir = '../model/lgb_202502061051_0'
    #     ,is_tuning = False
    #     ,train_file_path = ""
    #     ,test_file_path = ""
    # )
    # outer_runner.run_predict_cv(
    #     predict_file_path = f"../data/raw/{config.target_dataset}_{config.target_column}.parquet"
    # )