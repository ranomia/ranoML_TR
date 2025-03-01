import os
import pandas as pd
import numpy as np
from logging import getLogger

logger = getLogger(__name__)

class IndexSaver:
    """
    クロスバリデーションにおける各foldのインデックスを保存・ログ出力するためのユーティリティクラス
    """
    
    def __init__(self, save_dir: str, run_name: str):
        """
        初期化
        
        Parameters
        ----------
        save_dir : str
            保存先ディレクトリのパス
        run_name : str
            実験名
        """
        self.save_dir = save_dir
        self.run_name = run_name
        
        # インデックス保存用のディレクトリを作成
        self.indices_dir = os.path.join(self.save_dir, 'cv_indices')
        os.makedirs(self.indices_dir, exist_ok=True)
    
    def save_fold_indices(self, i_fold: int, tr_idx: np.ndarray, va_idx: np.ndarray, df: pd.DataFrame) -> None:
        """
        指定されたfoldのインデックスを保存し、ログ出力する
        
        Parameters
        ----------
        i_fold : int
            foldの番号
        tr_idx : np.ndarray
            訓練データのインデックス
        va_idx : np.ndarray
            検証データのインデックス
        df : pd.DataFrame
            元のデータフレーム
        """
        # IDの取得（存在しない場合はインデックスを使用）
        ids = df['id'].values if 'id' in df.columns else np.arange(len(df))
        groups = df['group'].values if 'group' in df.columns else np.array(['不明'] * len(df))
        
        # 訓練データとバリデーションデータのIDを取得
        tr_ids = ids[tr_idx]
        va_ids = ids[va_idx]
        tr_groups = groups[tr_idx]
        va_groups = groups[va_idx]
        
        # ログ出力
        logger.info(f"{self.run_name} - fold {i_fold} - train ids: {tr_ids.tolist()}")
        logger.info(f"{self.run_name} - fold {i_fold} - valid ids: {va_ids.tolist()}")
        logger.info(f"{self.run_name} - fold {i_fold} - train group: {sorted(set(tr_groups.tolist()))}")
        logger.info(f"{self.run_name} - fold {i_fold} - valid group: {sorted(set(va_groups.tolist()))}")
        
        # foldごとのディレクトリを作成
        fold_dir = os.path.join(self.indices_dir, f'fold{i_fold}')
        os.makedirs(fold_dir, exist_ok=True)
        
        # DataFrameとして保存
        tr_df = pd.DataFrame({
            'id': tr_ids,
            'group': tr_groups
        })
        va_df = pd.DataFrame({
            'id': va_ids,
            'group': va_groups
        })
        
        # CSVファイルとして保存（UTF-8 with BOM）
        tr_df.to_csv(
            os.path.join(fold_dir, 'train_indices.csv'),
            index=False,
            encoding='utf-8-sig'  # UTF-8 with BOM
        )
        va_df.to_csv(
            os.path.join(fold_dir, 'valid_indices.csv'),
            index=False,
            encoding='utf-8-sig'  # UTF-8 with BOM
        )