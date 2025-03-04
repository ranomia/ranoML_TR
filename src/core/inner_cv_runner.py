import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import optuna
from typing import Callable, Union
from sklearn.ensemble import VotingRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from core.config import Config
from preprocess.feature_selector import FeatureSelector
from util.util import ShuffledGroupKFold

config = Config()

class InnerCVRunner:
    def __init__(self, model_type: str, tuning_seed: int, model_builder: callable) -> None:
        self.n_repeats = 1
        self.n_splits = 2
        self.model_type = model_type
        self.tuning_seed = tuning_seed
        self.selector = FeatureSelector()
        self.model_builder = model_builder  # OuterCVRunnerのbuild_modelを受け取る

    def objective(self, trial, tr_x: pd.DataFrame, tr_y: pd.Series, va_x: pd.DataFrame, va_y: pd.Series) -> float:
        if self.model_type == 'lightgbm':
            params_range = {
                'learning_rate': trial.suggest_float('lightgbm_learning_rate', 0.001, 0.1, log=True),
                'feature_fraction': trial.suggest_float('lightgbm_feature_fraction', 0.4, 0.9),
                'num_leaves': trial.suggest_int('lightgbm_num_leaves', 16, 128),
                'subsample': trial.suggest_float('lightgbm_subsample', 0.4, 0.9),
                'reg_lambda': trial.suggest_float('lightgbm_reg_lambda', 1e-4, 10, log=True),
                'reg_alpha': trial.suggest_float('lightgbm_reg_alpha', 1e-4, 10, log=True),
                'min_data_in_leaf': trial.suggest_int('lightgbm_min_data_in_leaf', 10, 100),
                # 'device': 'gpu'
            }
            
            # OuterCVRunnerのbuild_modelを使用
            model_pipe = self.model_builder(params_dict={'lightgbm': params_range})

            ### pipelineで完結したいが、eval_setを使う場合は別で適用する必要がありそう。将来的に改善したい。
            # 前処理部分のみを先にfit
            preprocessor = model_pipe.named_steps['preprocessor']
            preprocessor.fit(tr_x, tr_y)

            # 前処理を適用（eval_setでva_xを利用するため）
            tr_x_transformed = preprocessor.transform(tr_x)
            va_x_transformed = preprocessor.transform(va_x)

            model = model_pipe.named_steps['model']
            pruning_callback = optuna.integration.LightGBMPruningCallback(
                trial,
                'rmse',
                valid_name='valid'
            )

            model.fit(
                 tr_x_transformed
                ,tr_y
                ,eval_set=[(tr_x_transformed, tr_y), (va_x_transformed, va_y)]
                ,eval_names=['train', 'valid']
                ,eval_metric='rmse'
                ,callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    pruning_callback
                ]
            )

        elif self.model_type == 'xgboost':
            params_range = {
                'n_estimators': trial.suggest_int('xgboost_n_estimators', 10, 5000),
                'learning_rate': trial.suggest_float('xgboost_learning_rate', 0.01, 1.0, log=True),
                'min_child_weight': trial.suggest_int('xgboost_min_child_weight', 1, 10),
                'max_depth': trial.suggest_int('xgboost_max_depth', 1, 50),
                'max_delta_step': trial.suggest_int('xgboost_max_delta_step', 0, 20),
                'subsample': trial.suggest_float('xgboost_subsample', 0.1, 1.0),
                'colsample_bytree': trial.suggest_float('xgboost_colsample_bytree', 0.1, 1.0),
                'colsample_bylevel': trial.suggest_float('xgboost_colsample_bylevel', 0.1, 1.0),
                'reg_lambda': trial.suggest_float('xgboost_reg_lambda', 1e-9, 100, log=True),
                'reg_alpha': trial.suggest_float('xgboost_reg_alpha', 1e-9, 100, log=True),
                'gamma': trial.suggest_float('xgboost_gamma', 1e-9, 0.5, log=True),
                'scale_pos_weight': trial.suggest_float('xgboost_scale_pos_weight', 1e-6, 500, log=True)
            }

            # OuterCVRunnerのbuild_modelを使用
            model_pipe = self.model_builder(params_dict={'xgboost': params_range})

            # 前処理部分のみを先にfit
            preprocessor = model_pipe.named_steps['preprocessor']
            preprocessor.fit(tr_x, tr_y)

            # 前処理を適用
            tr_x_transformed = preprocessor.transform(tr_x)
            va_x_transformed = preprocessor.transform(va_x)

            model = model_pipe.named_steps['model']

            model.fit(
                tr_x_transformed,
                tr_y,
                eval_set=[(va_x_transformed, va_y)],
                verbose=False
            )

        elif self.model_type == 'catboost':
            params_range = {
                'learning_rate': trial.suggest_float('catboost_learning_rate', 0.001, 0.01, log=True),
                'depth': trial.suggest_int('catboost_depth', 3, 5),
                'iterations': trial.suggest_int('catboost_iterations', 100, 150),
                'l2_leaf_reg': trial.suggest_float('catboost_l2_leaf_reg', 10, 20),
                'random_seed': self.tuning_seed,
                'verbose': 0
                # 'task_type': 'GPU'
            }
            model = CatBoostRegressor(**params_range)
            model.fit(tr_x, tr_y)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        va_y_pred = model_pipe.predict(va_x)
        
        rmse = root_mean_squared_error(va_y, va_y_pred)

        return rmse
    
    def parameter_tuning(self, all_x: pd.DataFrame, all_y: pd.Series, all_group: pd.Series, n_trials: int = 100):
        # FeatureSelectorをfitする
        self.selector.fit(all_x)

        model_types = self.model_type # リストでの複数指定も可能
        best_params_all = {}

        for model_type in model_types:
            score_list = []
            best_params_list = []

            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )

            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.tuning_seed),
                pruner=pruner
            )

            if config.group_column is None:
                kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.tuning_seed)
                for i_fold, (tr_idx, va_idx) in enumerate(kfold.split(all_x, all_y)):
                    tr_x, tr_y = all_x.iloc[tr_idx], all_y.iloc[tr_idx]
                    va_x, va_y = all_x.iloc[va_idx], all_y.iloc[va_idx]

                    # Optunaでのハイパーパラメータチューニング
                    study.optimize(lambda trial: self.objective(trial, tr_x, tr_y, va_x, va_y), n_trials=n_trials)

                    # 各フォールドのスコアとパラメータを記録
                    score_list.append(study.best_trial.value)
                    best_params_list.append(study.best_params)
                
                # 最適なパラメータを保存
                best_index = np.argmin(score_list)
                best_params_all[model_type] = best_params_list[best_index]
            else:
                kfold = ShuffledGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.tuning_seed)
                for i_fold in range(self.n_splits):
                    tr_idx, va_idx = list(kfold.split(all_x, all_y, all_group))[i_fold]
                    tr_x, tr_y = all_x.iloc[tr_idx], all_y.iloc[tr_idx]
                    va_x, va_y = all_x.iloc[va_idx], all_y.iloc[va_idx]

                    study.optimize(lambda trial: self.objective(trial, tr_x, tr_y, va_x, va_y), n_trials=n_trials)
                    
                    score_list.append(study.best_trial.value)
                    best_params_list.append(study.best_params)

                best_index = np.argmin(score_list)
                best_params_all[model_type] = best_params_list[best_index]

        return best_params_all