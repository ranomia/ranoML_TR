import os
import datetime
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold

import torch
from torch.utils.data import Dataset

from sklearn.metrics import cohen_kappa_score

class Util:
    @classmethod
    def dump(cls, value, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(value, path, compress=True)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


class Logger:
    def __init__(self):
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler('../model/general.log')
        file_result_handler = logging.FileHandler('../model/result.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic['name'] = run_name
        dic['score'] = np.mean(scores)
        for i, score in enumerate(scores):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])
    
    def log_fold_scores(self, metric: str, fold_rmse_scores: list, model_dir: str):
        """
        各foldのRMSEスコアとその平均値をログに出力する
        :param metric: 指標の名前
        :param fold_rmse_scores: 各foldのRMSEスコア（リスト形式）
        :param model_dir: モデルディレクトリのパス
        """
        avg_rmse = np.mean(fold_rmse_scores)
        avg_std = np.std(fold_rmse_scores)

        # ログメッセージの作成（実験ディレクトリ情報を追加）
        log_message = f"\nExperiment Directory: {model_dir}\n"
        log_message += f"Fold {metric} Scores:\n" + "\n".join(
            [f"  Fold {i+1}: {score:.4f}" for i, score in enumerate(fold_rmse_scores)]
        )
        log_message += f"\nCV {metric}: {avg_rmse:.4f} +/- {avg_std:.4f}\n"
        log_message += "-" * 80  # 区切り線を追加
        
        # result.logに出力
        self.result(log_message)

    def save_fold_scores_to_model_dir(self, metric: str, fold_rmse_scores: list, model_dir: str, mode: str = 'a'):
        """
        各foldのスコアをモデルディレクトリ内のファイルに保存する
        :param metric: 指標の名前
        :param fold_rmse_scores: 各foldのRMSEスコア（リスト形式）
        :param model_dir: モデルディレクトリのパス
        :param mode: ファイルの書き込みモード ('w'：上書き, 'a'：追記)
        """
        avg_rmse = np.mean(fold_rmse_scores)
        avg_std = np.std(fold_rmse_scores)

        # ログメッセージの作成
        log_message = f"\n{metric} Scores:\n" + "\n".join(
            [f"  Fold {i+1}: {score:.4f}" for i, score in enumerate(fold_rmse_scores)]
        )
        log_message += f"\nCV {metric}: {avg_rmse:.4f} +/- {avg_std:.4f}\n"

        # モデルディレクトリ内にスコアを保存（追記モード）
        with open(f"{model_dir}/metrics/fold_scores.txt", mode) as f:
            # 'w'モードの場合のみ、ディレクトリ情報を先頭に書き込む
            if mode == 'w':
                f.write(f"Experiment Directory: {model_dir}\n")
            f.write(log_message)

class Submission: # 引用元から未修整のため、利用不可
    @classmethod
    def create_submission(cls, run_name):
        submission = pd.read_csv('../data/input/sampleSubmission.csv')
        pred = Util.load(f'../model/{run_name}/pred/test.pkl')
        for i in range(pred.shape[1]):
            submission[f'Class_{i + 1}'] = pred[:, 1]
        submission.to_csv(f'../submission/{run_name}.csv', index=False)

class ShuffledGroupKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.group_kfold = GroupKFold(n_splits=self.n_splits)

    def split(self, X, y=None, groups=None):
        # groups が必要なため、None であってはいけない
        if groups is None:
            raise ValueError("Groups must be provided for ShuffledGroupKFold.")
        
        # グループのユニーク値を取得
        unique_groups = np.unique(groups)
        
        # シャッフルオプションが有効の場合
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(unique_groups)

        # ユニークグループに基づいて split を実行
        group_indices = np.array([np.where(unique_groups == group)[0][0] for group in groups])
        
        # group_kfold の split メソッドを使用
        for train_idx, test_idx in self.group_kfold.split(X, y, groups=group_indices):
            yield train_idx, test_idx

class TabularDatasetTrain(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TabularDatasetTest(Dataset):
    def __init__(self, data_x):
        self.data_x = data_x

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        x = self.data_x[idx]
        return torch.tensor(x, dtype=torch.float32)