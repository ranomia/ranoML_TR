import warnings
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

warnings.filterwarnings('ignore', message='.*CatBoostPruningCallback is experimental*')

class InnerCVRunner:
    def __init__(self, model_type: str, tuning_seed: int, model_builder: callable) -> None:
        self.n_repeats = 1
        self.n_splits = 3
        self.model_type = model_type
        self.tuning_seed = tuning_seed
        self.selector = FeatureSelector()
        self.model_builder = model_builder  # OuterCVRunnerのbuild_modelを受け取る

    def objective(self, trial, all_x: pd.DataFrame, all_y: pd.Series, fold_indices: list) -> float:
        rmse_list = []

        for i_fold, (tr_idx, va_idx) in enumerate(fold_indices):
            tr_x, tr_y = all_x.iloc[tr_idx], all_y.iloc[tr_idx]
            va_x, va_y = all_x.iloc[va_idx], all_y.iloc[va_idx]

            if self.model_type == 'lightgbm':
                params_range = {
                    'learning_rate': trial.suggest_float('lightgbm_learning_rate', 0.01, 1.0, log=True),
                    'feature_fraction': trial.suggest_float('lightgbm_feature_fraction', 0.4, 0.9),
                    'num_leaves': trial.suggest_int('lightgbm_num_leaves', 16, 128),
                    'subsample': trial.suggest_float('lightgbm_subsample', 0.4, 0.9),
                    'reg_lambda': trial.suggest_float('lightgbm_reg_lambda', 1e-4, 10, log=True),
                    'reg_alpha': trial.suggest_float('lightgbm_reg_alpha', 1e-4, 10, log=True),
                    'min_data_in_leaf': trial.suggest_int('lightgbm_min_data_in_leaf', 10, 100),
                }
                
                model_pipe = self.model_builder(params_dict={'lightgbm': params_range})
                preprocessor = model_pipe.named_steps['preprocessor']
                preprocessor.fit(tr_x, tr_y)

                tr_x_transformed = preprocessor.transform(tr_x)
                va_x_transformed = preprocessor.transform(va_x)

                model = model_pipe.named_steps['model']
                
                model.fit(
                    tr_x_transformed,
                    tr_y,
                    eval_set=[(tr_x_transformed, tr_y), (va_x_transformed, va_y)],
                    eval_names=['train', 'valid'],
                    eval_metric='rmse',
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False)
                    ]
                )

                # 予測とRMSEを計算し、rmse_listに格納するだけ
                va_y_pred = model_pipe.predict(va_x)
                rmse = root_mean_squared_error(va_y, va_y_pred)
                rmse_list.append(rmse)

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

                model_pipe = self.model_builder(params_dict={'xgboost': params_range})
                preprocessor = model_pipe.named_steps['preprocessor']
                preprocessor.fit(tr_x, tr_y)

                tr_x_transformed = preprocessor.transform(tr_x)
                va_x_transformed = preprocessor.transform(va_x)

                model = model_pipe.named_steps['model']
                model.fit(
                    tr_x_transformed,
                    tr_y,
                    eval_set=[(va_x_transformed, va_y)],
                    verbose=False
                )

                va_y_pred = model_pipe.predict(va_x)
                rmse = root_mean_squared_error(va_y, va_y_pred)
                rmse_list.append(rmse)

            elif self.model_type == 'catboost':
                params_range = {
                    'depth': trial.suggest_int('catboost_depth', 1, 8),
                    'learning_rate': trial.suggest_float('catboost_learning_rate', 0.001, 0.01, log=True),
                    'random_strength': trial.suggest_float('catboost_random_strength', 1e-6, 10, log=True),
                    'bagging_temperature': trial.suggest_float('catboost_bagging_temperature', 0.0, 1.0),
                    'border_count': trial.suggest_int('catboost_border_count', 1, 255),
                    'l2_leaf_reg': trial.suggest_int('catboost_l2_leaf_reg', 2, 30)
                }

                model_pipe = self.model_builder(params_dict={'catboost': params_range})
                preprocessor = model_pipe.named_steps['preprocessor']
                preprocessor.fit(tr_x, tr_y)

                tr_x_transformed = preprocessor.transform(tr_x)
                va_x_transformed = preprocessor.transform(va_x)

                model = model_pipe.named_steps['model']

                model.fit(
                    tr_x_transformed,
                    tr_y,
                    eval_set=[(va_x_transformed, va_y)],
                    early_stopping_rounds=50,
                    verbose=False,
                    use_best_model=True
                )

                va_y_pred = model_pipe.predict(va_x)
                rmse = root_mean_squared_error(va_y, va_y_pred)
                rmse_list.append(rmse)

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        mean_rmse = np.mean(rmse_list)
        return mean_rmse

    def parameter_tuning(self, all_x: pd.DataFrame, all_y: pd.Series, all_group: pd.Series, n_trials: int = 100):
        # FeatureSelectorをfitする
        self.selector.fit(all_x)

        best_params = {}

        if config.group_column is None:
            kfold = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.tuning_seed)
            fold_indices = list(kfold.split(all_x, all_y))
        else:
            kfold = ShuffledGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.tuning_seed)
            fold_indices = list(kfold.split(all_x, all_y, all_group))

        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.tuning_seed)
        )

        study.optimize(lambda trial: self.objective(trial, all_x, all_y, fold_indices), n_trials=n_trials)

        # 最適なパラメータを保存
        best_params[self.model_type] = study.best_params

        return best_params