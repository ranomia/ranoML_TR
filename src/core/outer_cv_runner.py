import numpy as np
import pandas as pd
import os
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import shap
import pickle

from typing import Callable, List, Optional, Tuple, Union

from core.inner_cv_runner import InnerCVRunner
from core.config import Config
from preprocess.data_error_corrector import DataErrorCorrector
from preprocess.custom_target_encoder import CustomTargetEncoder
from preprocess.feature_selector import FeatureSelector
from evaluation.model_evaluator import ModelEvaluator
from util.util import Logger, Util
from util.stratified_shuffled_group_kfold import StratifiedShuffledGroupKFold
from util.index_saver import IndexSaver

logger = Logger()
config = Config()

class OuterCVRunner:
    # build_modelで使用する関数
    @staticmethod
    def _to_string(x):
        return x.astype(str)
    
    @staticmethod
    def _to_float(x):
        return x.astype(float)
    
    @staticmethod
    def _to_int(x):
        return x.astype(int)

    def __init__(self, run_name: str, model_type: str, params_dict: None, cv_seed: int, tuning_seed: int, model_dir: str, is_tuning: bool, train_file_path: str, test_file_path: str):
        """
        コンストラクタ

        :param run_name: ランの名前
        :param model_type: モデルのアーキテクチャ
        :param params: ハイパーパラメータ
        :param n_fold: fold数
        :param dtype_dict: データ型の定義（trainに合わせてtestをロード）
        """
        self.run_name = run_name
        self.model_type = model_type
        self.params_dict = params_dict
        self.n_fold = 5
        self.dtype_dict = {}
        self.cv_seed = cv_seed
        self.tuning_seed = tuning_seed
        self.selector = None
        self.model_dir = model_dir
        self.is_tuning = is_tuning
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path
    
    def train_fold(self, i_fold: Union[int, str], cv_results: dict):
        """
        クロスバリデーションにおける特定のfoldの学習・評価を行う

        他メソッドから呼び出すほか、単体でも確認やパラメータ調整に用いる

        :param i_fold: foldの番号(全量で学習するときは'all'とする)
        :return: (モデルのインスタンス、レコードのインデックス、予測値、評価によるスコア)のタプル
        """
        
        is_validation = i_fold != 'all'
        
        # 学習データの読込
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        # groupありの場合はgroup_columnを特徴量から削除
        if config.group_column is not None:
            train_group = train_x[config.group_column]
            train_x = train_x.drop(config.group_column, axis=1)

        # FeatureSelectorをfitし、保存する
        self.selector = FeatureSelector()
        self.selector.fit(train_x)
        selector_path = os.path.join(self.model_dir, "models/selectors", f"selector_{self.run_name}_fold{i_fold}.pkl")
        self.selector.save(selector_path)

        if is_validation:
            # 学習データ・バリデーションデータをセットする
            tr_idx, va_idx = self.load_index_fold(i_fold)
            
            # インデックスの保存
            index_saver = IndexSaver(self.model_dir, self.run_name)
            index_saver.save_fold_indices(i_fold, tr_idx, va_idx, train_x)

            if config.group_column is None:
                tr_x, tr_y = train_x.iloc[tr_idx], train_y.iloc[tr_idx]
                va_x, va_y = train_x.iloc[va_idx], train_y.iloc[va_idx]
            else:
                tr_x, tr_y, tr_g = train_x.iloc[tr_idx], train_y.iloc[tr_idx], train_group.iloc[tr_idx]
                va_x, va_y, va_g = train_x.iloc[va_idx], train_y.iloc[va_idx], train_group.iloc[va_idx]

            # ハイパーパラメータのチューニングを行う
            if self.is_tuning:
                if config.group_column is None:
                    inner_runner = InnerCVRunner(
                        model_type=self.model_type,
                        tuning_seed=self.tuning_seed,
                        model_builder=self.build_model  # build_modelメソッドを渡す
                    )
                    tuned_params_dict_flatten = inner_runner.parameter_tuning(tr_x, tr_y, None, n_trials=100)
                else:
                    inner_runner = InnerCVRunner(
                        model_type=self.model_type,
                        tuning_seed=self.tuning_seed,
                        model_builder=self.build_model  # build_modelメソッドを渡す
                    )
                    tuned_params_dict_flatten = inner_runner.parameter_tuning(tr_x, tr_y, tr_g, n_trials=100)
                
                tuned_params_dict = {
                    'lightgbm': {},
                    'xgboost': {},
                    'catboost': {}
                }

                for model_type in tuned_params_dict_flatten.keys():
                    for param_name, param_value in tuned_params_dict_flatten[model_type].items():
                        tuned_params_dict[model_type][param_name.replace(model_type+'_', '')] = param_value

                # params_dictの更新
                self.params_dict = self.update_params_dict(params_dict=self.params_dict, tuned_params_dict=tuned_params_dict)
            
            with open(f"{self.model_dir}/models/parameters/params_dict_{self.run_name}_fold{i_fold}.json", "w") as f:
                json.dump(self.params_dict, f)

            model_pipe = self.build_model(params_dict=self.params_dict)

            ### pipelineで完結したいが、eval_setを使う場合は別で適用する必要がありそう。将来的に改善したい。
            # 前処理部分のみを先にfit
            preprocessor = model_pipe.named_steps['preprocessor']
            tr_x_transformed = preprocessor.fit_transform(tr_x, tr_y)
            va_x_transformed = preprocessor.transform(va_x)
            
            # モデルをクローンし、変換済みデータで学習
            model = clone(model_pipe.named_steps['model'])
            if isinstance(model, LGBMRegressor):
                model.fit(
                    tr_x_transformed, tr_y,
                    eval_set=[(tr_x_transformed, tr_y), (va_x_transformed, va_y)],
                    eval_names=['train', 'valid'],
                    eval_metric='rmse',
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            elif isinstance(model, XGBRegressor):
                model.fit(
                    tr_x_transformed, tr_y,
                    eval_set=[(tr_x_transformed, tr_y), (va_x_transformed, va_y)],
                    verbose=False
                )

            # 学習済みの前処理とモデルでパイプラインを再構築
            fitted_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # 学習データ・バリデーションデータへの予測・評価を行う
            tr_y_pred = fitted_pipeline.predict(tr_x)
            va_y_pred = fitted_pipeline.predict(va_x)

            # パイプラインをpickleで保存
            with open(f'{self.model_dir}/models/pipelines/pipeline_{self.run_name}_fold{i_fold}.pkl', 'wb') as f:
                pickle.dump(fitted_pipeline, f)
            
            evaluator = ModelEvaluator(self.model_type, self.model_dir, self.run_name) # dataを適切に渡せば、プロットできるものが増える
            
            # 評価指標の計算
            tr_metrics = evaluator.evaluate_metrics(tr_y, tr_y_pred)
            va_metrics = evaluator.evaluate_metrics(va_y, va_y_pred)
            
            # 学習曲線のプロット
            evaluator.plot_learning_curve(
                model.evals_result_,
                i_fold,
                save_path=f'{self.model_dir}/plots/learning_curves/curve_{self.run_name}_fold{i_fold}.png'
            )

            # 残差プロット
            evaluator.plot_residuals(
                tr_y, tr_y_pred,
                va_y, va_y_pred,
                i_fold,
                save_path=f'{self.model_dir}/plots/residuals/residual_{self.run_name}_fold{i_fold}.png'
            )
            # SHAP値の可視化
            evaluator.plot_shap_values(
                model,
                va_x_transformed,
                preprocessor.get_feature_names_out(),
                i_fold,
                save_path=f'{self.model_dir}/metrics/feature_importance/shap_{self.run_name}_fold{i_fold}.png'
            )

            # 評価指標の算出
            cv_results['tr_rmse'].append(root_mean_squared_error(tr_y, tr_y_pred))
            cv_results['va_rmse'].append(root_mean_squared_error(va_y, va_y_pred))
            cv_results['tr_mae'].append(mean_absolute_error(tr_y, tr_y_pred))
            cv_results['va_mae'].append(mean_absolute_error(va_y, va_y_pred))
            cv_results['tr_rmspe'].append(np.sqrt(np.mean(np.square((tr_y - tr_y_pred) / (tr_y + 1e-10)))))
            cv_results['va_rmspe'].append(np.sqrt(np.mean(np.square((va_y - va_y_pred) / (va_y + 1e-10)))))

            # 実験条件・結果の保存
            cv_results['tr_idx'].append(tr_idx)
            cv_results['va_idx'].append(va_idx)
            cv_results['tr_y'].append(tr_y)
            cv_results['va_y'].append(va_y)
            cv_results['tr_y_pred'].append(tr_y_pred)
            cv_results['va_y_pred'].append(va_y_pred)
            # cv_results['params'].append(best_params)

            # モデル、インデックス、予測値、評価を返す
            return fitted_pipeline, cv_results
        else:
            # 学習データ全てで学習を行う
            model_pipe = self.build_model(i_fold=i_fold, params=self.params)
            model_pipe.train_model(train_x, train_y)

            # モデルを返す
            return model_pipe, None, None, None

    def run_train_cv(self) -> None:
        """
        クロスバリデーションでの学習・評価を行う
        """
        # 学習データの読込
        train_x = self.load_x_train()

        # groupありの場合はgroup_columnを特徴量から削除
        if config.group_column is not None:
            train_group = train_x[config.group_column]
            train_x = train_x.drop(config.group_column, axis=1)

        logger.info(f'{self.run_name} - start training outer cv')

        cv_results = {
            'tr_idx': [],       # 各foldの学習データのインデックス
            'va_idx': [],       # 各foldの検証データのインデックス
            'group': [],             # 分析用のグループ
            'tr_rmse': [],       # 各foldの学習データに対するRMSE
            'va_rmse': [],       # 各foldの検証データに対するRMSE
            'tr_mae': [],        # 各foldの学習データに対するMAE
            'va_mae': [],        # 各foldの検証データに対するMAE
            'tr_rmspe': [],      # 各foldの学習データに対するRMSPE
            'va_rmspe': [],      # 各foldの検証データに対するRMSPE
            'tr_y': [],           # 各foldの学習データに対する予測値
            'va_y': [],           # 各foldの学習データの正解値
            'tr_y_pred': [],           # 各foldの検証データに対する予測値
            'va_y_pred': [],           # 各foldの検証データの正解値
            'params': []            # 各foldのモデルのハイパーパラメータ
        }

        # ディレクトリ構造の作成
        os.makedirs(f"{self.model_dir}/metrics", exist_ok=True)
        os.makedirs(f"{self.model_dir}/metrics/feature_importance/", exist_ok=True)
        os.makedirs(f"{self.model_dir}/plots/learning_curves", exist_ok=True)
        os.makedirs(f"{self.model_dir}/plots/residuals", exist_ok=True)
        os.makedirs(f"{self.model_dir}/models", exist_ok=True)
        os.makedirs(f"{self.model_dir}/models/parameters", exist_ok=True)  # パラメータ保存用
        os.makedirs(f"{self.model_dir}/models/preprocessors", exist_ok=True)  # 前処理パイプライン保存用
        os.makedirs(f"{self.model_dir}/models/selectors", exist_ok=True)  # selector保存用
        os.makedirs(f"{self.model_dir}/models/pipelines", exist_ok=True)  # パイプライン保存用
        os.makedirs(f"{self.model_dir}/models/feature_names", exist_ok=True)  # 特徴量名保存用

        # 各foldで学習を行う
        for i_fold in range(self.n_fold):
            # 学習を行う
            logger.info(f'{self.run_name} fold {i_fold} - start training')
            model, cv_results = self.train_fold(i_fold, cv_results)
            logger.info(f'{self.run_name} fold {i_fold} - end training - rmse score {cv_results["va_rmse"][i_fold]}')

            # 前処理パイプラインの保存
            preprocessor = model.named_steps['preprocessor']
            with open(f'{self.model_dir}/models/preprocessors/preprocessor_{self.run_name}_fold{i_fold}.pkl', 'wb') as f:
                pickle.dump(preprocessor, f)

        logger.info(f'{self.run_name} - end training outer cv - rmse score {np.mean(cv_results["va_rmse"])}')

        # 評価結果の保存
        # RMSEの結果を保存（general.logとresult.logにも出力）
        logger.log_fold_scores('tr_rmse', cv_results['tr_rmse'], self.model_dir)
        logger.log_fold_scores('va_rmse', cv_results['va_rmse'], self.model_dir)
        logger.save_fold_scores_to_model_dir('tr_rmse', cv_results['tr_rmse'], self.model_dir, mode='w')
        logger.save_fold_scores_to_model_dir('va_rmse', cv_results['va_rmse'], self.model_dir, mode='a')
        # MAEの結果を保存（model_dirのみ）
        logger.save_fold_scores_to_model_dir('tr_mae', cv_results['tr_mae'], self.model_dir, mode='a')
        logger.save_fold_scores_to_model_dir('va_mae', cv_results['va_mae'], self.model_dir, mode='a')
        # RMSPEの結果を保存（model_dirのみ）
        logger.save_fold_scores_to_model_dir('tr_rmspe', cv_results['tr_rmspe'], self.model_dir, mode='a')
        logger.save_fold_scores_to_model_dir('va_rmspe', cv_results['va_rmspe'], self.model_dir, mode='a')

    def run_predict_cv(self, predict_file_path: str) -> None:
        """
        クロスバリデーションで学習した各foldのモデルの平均により、新しいデータの予測を行う

        Args:
            predict_file_path (str): 予測対象のファイルパス
        """
        logger.info(f'{self.run_name} - start prediction outer cv')

        # 予測用ディレクトリの作成
        os.makedirs(f"{self.model_dir}/pred", exist_ok=True)
        
        # 予測対象データの読み込み
        predict_data = pd.read_parquet(predict_file_path)
        
        # groupありの場合はgroup_columnを特徴量から削除
        if config.group_column is not None:
            predict_data = predict_data.drop(config.group_column, axis=1)

        preds = []

        # 各foldのモデルで予測を行う
        for i_fold in range(self.n_fold):
            logger.info(f'{self.run_name} - start prediction fold: {i_fold}')
            
            try:
                # パイプラインのロード
                with open(f'{self.model_dir}/models/pipelines/pipeline_{self.run_name}_fold{i_fold}.pkl', 'rb') as f:
                    model_pipe = pickle.load(f)

                pred = model_pipe.predict(predict_data)
                preds.append(pred)
            
                logger.info(f'{self.run_name} - end prediction fold: {i_fold}')
            except Exception as e:
                logger.info(f"Fold {i_fold}の予測中にエラーが発生しました: {str(e)}")
                raise

        # 予測の平均値を計算
        pred_avg = np.mean(preds, axis=0)

        # 予測結果をデータフレームとして保存
        pred_df = pd.DataFrame({
            'predicted_value': pred_avg
        }, index=predict_data.index)
        
        # 予測結果の保存
        pred_df.to_csv(f'{self.model_dir}/pred/predictions_{self.run_name}.csv')

        logger.info(f'{self.run_name} - end prediction outer cv')

    def run_train_all(self) -> None:
        """学習データ全てで学習し、そのモデルを保存する"""
        logger.info(f'{self.run_name} - start training all')

        i_fold = 'all'
        model, _, _, _ = self.train_fold(i_fold)
        model.save_model()

        logger.info(f'{self.run_name} - end training all')

    def run_predict_all(self) -> None:
        """
        学習データ全てで学習したモデルにより、テストデータの予測を行う

        あらかじめrun_train_allを実行しておく必要がある
        """
        logger.info(f'{self.run_name} - start prediction all')

        test_x = self.load_x_test()

        # 学習データすべてで学習したモデルで予測を行う
        i_fold = 'all'
        model = self.build_model(i_fold=i_fold, params={})
        model.load_model()
        pred = model.predict_model(test_x)

        # 予測結果の保存
        Util.dump(pred, f'{self.model_dir}/pred/test.pkl')

        logger.info(f'{self.run_name} - end prediction all')

    def build_model(self, params_dict: dict):
        """
        クロスバリデーションでのfoldを指定して、モデルの作成を行う

        :param params_dict: チューニングされたパラメータ
        :return: モデルのインスタンス
        """
        # ラン名、fold、モデルのクラスからモデルを作成する
        if self.model_type == 'lightgbm':
            model = LGBMRegressor(
                **params_dict['lightgbm']
                ,random_state = self.cv_seed
                ,verbose = -1
                ,n_estimators = 5000
                ,num_threads = 4
            )
        elif self.model_type == 'xgboost':
            model = XGBRegressor(
                **params_dict['xgboost']
                ,eval_metric = 'rmse'
                ,early_stopping_rounds = 50
                ,random_state = self.cv_seed
                ,verbosity = 0
                ,n_jobs = 4
            )
        elif self.model_type == 'catboost':
            model = CatBoostRegressor(
                **params_dict['catboost']
                ,random_seed = self.cv_seed
                ,verbose = False
                ,thread_count = 4
            )

        # カラムの型に応じて異なる変換を適用するColumnTransformer
        numeric_features = self.selector.get_feature_names_out(feature_types=['int64', 'float64'])
        zero_impute_features = []
        mean_impute_features = [col for col in numeric_features if col not in zero_impute_features]

        categorical_features = self.selector.get_feature_names_out(feature_types=['category', 'object'])
        target_encoding_features = []
        non_target_encoding_features = [col for col in categorical_features if col not in target_encoding_features]
        
        boolean_features = self.selector.get_feature_names_out(feature_types=['bool'])

        # ColumnTransformer の出力を pandas の DataFrame に固定するため set_output を使用
        column_transformer = ColumnTransformer(
            transformers=[
                ('num_mean', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean'))
                ]), mean_impute_features),
                ('num_zero', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
                ]), zero_impute_features),
                ('cat_target_encoding', Pipeline([
                    ('to_string', FunctionTransformer(self._to_string, feature_names_out='one-to-one')),
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing'))
                ]), target_encoding_features),
                ('cat_non_target_encoding', Pipeline([
                    ('to_string', FunctionTransformer(self._to_string, feature_names_out='one-to-one')),
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ]), non_target_encoding_features),
                ('bool', Pipeline([
                    ('to_float', FunctionTransformer(self._to_float, feature_names_out='one-to-one')),
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('to_int', FunctionTransformer(self._to_int, feature_names_out='one-to-one'))
                ]), boolean_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")  # ここで出力を DataFrame に固定

        preprocessor = Pipeline([
            ('column_transformer', column_transformer),
            ('data_error_corrector', DataErrorCorrector()),
            ('custom_target_encoder', CustomTargetEncoder(
                    encoding_columns=target_encoding_features
                ,smoothing=10
                ,min_samples_leaf=5
            ))
        ])

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', clone(model))
        ])
        return pipeline
    
    def load_x_train(self) -> pd.DataFrame:
        """
        学習データの特徴量を読み込む

        :return: 学習データの特徴量
        """
        # 学習データのファイルパスから拡張子を取得
        file_extension = self.train_file_path.split('.')[-1].lower()
        
        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            x_train = pd.read_pickle(self.train_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'csv':
            x_train = pd.read_csv(self.train_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'xlsx':
            x_train = pd.read_excel(self.train_file_path).drop(config.target_column, axis=1)
        elif file_extension == 'parquet':
            x_train = pd.read_parquet(self.train_file_path, engine='pyarrow').drop(config.target_column, axis=1)
        elif file_extension == 'jsonl':
            x_train = pd.read_json(self.train_file_path, lines=True).drop(config.target_column, axis=1)
        else:
            raise ValueError(f"Unsupported file format: '{file_extension}'. Supported formats are: pkl, pickle, csv, xlsx, parquet, jsonl.")

        # 除外カラムを削除
        x_train = x_train.drop(config.exclude_columns, axis=1, errors='ignore')

        # 型変換の優先順位: bool -> int -> float -> category
        for column in x_train.columns:
            col_data = x_train[column]
            
            # まずbool型への変換を試みる
            unique_values = col_data.dropna().unique()
            if len(unique_values) <= 2 and set(unique_values).issubset({0, 1, True, False, '0', '1', 'True', 'False', 'true', 'false'}):
                x_train[column] = col_data.map({'True': True, 'true': True, '1': True, 1: True,
                                              'False': False, 'false': False, '0': False, 0: False}).astype(bool)
                continue

            # Noneやnanを含むかチェック
            has_null = col_data.isna().any()
            
            if not has_null:
                # Nullを含まない場合のみint変換を試みる
                try:
                    x_train[column] = pd.to_numeric(col_data, downcast='integer')
                    continue
                except (ValueError, TypeError):
                    pass
            
            # float変換を試みる
            try:
                x_train[column] = pd.to_numeric(col_data, downcast='float')
                continue
            except (ValueError, TypeError):
                pass

            # 上記の変換が全て失敗した場合はcategory型として扱う
            x_train[column] = col_data.astype('category')

        # group_columnをcategory型に変換
        if config.group_column is not None:
            x_train[config.group_column] = x_train[config.group_column].astype(str)

        self.dtype_dict = x_train.dtypes.to_dict()

        # 特殊文字を置き換えるコード例
        x_train.columns = x_train.columns.str.replace('"', '').str.replace("'", "").str.replace('-', '_').str.replace(' ', '').str.replace(',', '_').str.replace('.', '')
        
        return x_train
    
    def load_y_train(self) -> pd.Series:
        """
        学習データの目的変数を読み込む

        :return: 学習データの目的変数
        """
        # 学習データのファイルパスから拡張子を取得
        file_extension = self.train_file_path.split('.')[-1].lower()

        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            y_train = pd.read_pickle(self.train_file_path)[config.target_column]
        elif file_extension == 'csv':
            y_train = pd.read_csv(self.train_file_path)[config.target_column]
        elif file_extension == 'xlsx':
            y_train = pd.read_excel(self.train_file_path)[config.target_column]
        elif file_extension == 'parquet':
            y_train = pd.read_parquet(self.train_file_path, engine='pyarrow')[config.target_column]
        elif file_extension == 'jsonl':
            y_train = pd.read_json(self.train_file_path, lines=True)[config.target_column]
        else:
            # 未対応の形式に対するエラーを発生させる
            raise ValueError(f"Unsupported file format: '{file_extension}'. Supported formats are: pkl, pickle, csv, xlsx, parquet, jsonl.")

        return y_train
    
    def load_x_test(self) -> pd.DataFrame:
        """
        テストデータの特徴量を読み込む

        :return: テストデータの特徴量
        """
        # 学習データのファイルパスから拡張子を取得
        file_extension = self.test_file_path.split('.')[-1].lower()

        # 拡張子に基づいて処理を分岐
        if file_extension in ['pkl', 'pickle']:
            x_test = pd.read_pickle(self.test_file_path)
        elif file_extension == 'csv':
            x_test = pd.read_csv(self.test_file_path)
        elif file_extension == 'xlsx':
            x_test = pd.read_excel(self.test_file_path)
        elif file_extension == 'parquet':
            x_test = pd.read_parquet(self.test_file_path, engine='pyarrow')
        elif file_extension == 'jsonl':
            x_test = pd.read_json(self.test_file_path, lines=True)
        else:
            # 未対応の形式に対するエラーを発生させる
            raise ValueError(f"Unsupported file format: '{file_extension}'. Supported formats are: pkl, pickle, csv, xlsx, parquet, jsonl.")

        # 除外カラムを削除
        x_test = x_test.drop(config.exclude_columns, axis=1, errors='ignore')

        # 訓練データのデータ型に合わせる
        for column, dtype in self.dtype_dict.items():
            if column in x_test.columns:
                x_test[column] = x_test[column].astype(dtype)
        
        # object型のカラムをcategory型に変換
        for col in x_test.select_dtypes(include=['object']).columns:
            x_test[col] = x_test[col].astype('category')

        # group_columnをcategory型に変換
        if config.group_column is not None:
            x_test[config.group_column] = x_test[config.group_column].astype(str)

        # 特殊文字を置き換えるコード例
        x_test.columns = x_test.columns.str.replace('"', '').str.replace("'", "").str.replace('-', '_').str.replace(' ', '').str.replace(',', '_').str.replace('.', '')

        return x_test
    
    def load_index_fold(self, i_fold: int) -> Tuple[np.array, np.array]:
        """
        クロスバリデーションでのfoldを指定して対応するレコードのインデックスを返す

        :param i_fold: foldの番号
        :return: fold_iにおけるtrain, validationのインデックスのタプル
        """
        train_x = self.load_x_train()
        train_y = self.load_y_train()

        if config.group_column is None:
            if config.strata_column is not None:
                # 層化に使用する列が指定されている場合
                strata = train_x[config.strata_column]
            else:
                # 目標変数を4分位数でビン分割して層化に使用
                strata = pd.qcut(train_y, q=4, labels=False)
            
            # tr+tuとvaの分割
            kfold = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=self.cv_seed)
            tr_idx, va_idx = list(kfold.split(train_x, strata))[i_fold]
        else:
            if config.strata_column is not None:
                # 層化に使用する列が指定されている場合
                strata = train_x[config.strata_column]
            else:
                # 目標変数を4分位数でビン分割して層化に使用
                strata = pd.qcut(train_y, q=4, labels=False)

            # tr+tuとvaの分割
            kfold = StratifiedShuffledGroupKFold(n_splits=self.n_fold, shuffle=True, random_state=self.cv_seed)
            tr_idx, va_idx = list(kfold.split(
                X=train_x,
                y=strata,
                groups=train_x[config.group_column].values
            ))[i_fold]

        return tr_idx, va_idx
    
    def update_params_dict(self, params_dict: dict, tuned_params_dict: dict) -> dict:
        """
        InnerCVRunnerでチューニングされたパラメータを上書きするためのプログラム
        元のparams_dictに定義されていない場合は、新規に追加する
        :param params_dict: 元のパラメータ
        :param tuned_params_dict: InnerCVRunnerでチューニングされたパラメータ
        :return: 更新されたパラメータ
        """

        
        for key, value in tuned_params_dict.items():
            # キーが存在しない場合は初期化
            if key not in params_dict:
                params_dict[key] = {}
            # params_dict を tuned_params_dict の内容で更新
            params_dict[key].update(value)
        
        return params_dict
