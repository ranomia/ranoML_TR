import numpy as np
import pandas as pd
from typing import List, Union
import pickle

class FeatureSelector:
    def __init__(self):
        self._feature_names = None
        self._feature_types = None

    def fit(self, X: pd.DataFrame) -> 'FeatureSelector':
        """
        データフレームの特徴量名と型を記録します
        
        Args:
            X: 入力データフレーム
        """
        self._feature_names = X.columns.tolist()
        self._feature_types = X.dtypes
        return self
    
    def save(self, filepath: str) -> None:
        """
        特徴量の情報を保存します
        
        Args:
            filepath: 保存先のパス
        """
        if self._feature_names is None or self._feature_types is None:
            raise ValueError("fit メソッドを先に実行してください")
        
        feature_info = {
            'feature_names': self._feature_names,
            'feature_types': self._feature_types.to_dict()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(feature_info, f)
    
    def load(self, filepath: str) -> None:
        """
        保存された特徴量の情報を読み込みます
        
        Args:
            filepath: 読み込むファイルのパス
        """
        with open(filepath, 'rb') as f:
            feature_info = pickle.load(f)
        
        self._feature_names = feature_info['feature_names']
        self._feature_types = pd.Series(feature_info['feature_types'])
    
    def get_feature_names_out(self, feature_types: Union[str, List[str]] = None) -> List[str]:
        """
        指定されたデータ型の特徴量名のリストを返します
        
        Args:
            feature_types: 取得したい特徴量の型（'int64', 'float64', 'category', 'object', 'bool'など）
            
        Returns:
            指定された型の特徴量名のリスト
        """
        if self._feature_names is None:
            raise ValueError("fit メソッドを先に実行してください")
            
        if feature_types is None:
            return self._feature_names
            
        if isinstance(feature_types, str):
            feature_types = [feature_types]
            
        selected_features = []
        for col in self._feature_names:
            if self._feature_types[col].name in feature_types:
                selected_features.append(col)
                
        return selected_features

    def select_dtypes(self, include=None, exclude=None) -> List[str]:
        """
        pandas.DataFrameのselect_dtypesと同様の機能を提供します
        
        Args:
            include: 含めたいデータ型
            exclude: 除外したいデータ型
            
        Returns:
            条件に合致する特徴量名のリスト
        """
        if self._feature_names is None:
            raise ValueError("fit メソッドを先に実行してください")
            
        temp_df = pd.DataFrame(columns=self._feature_names).astype(dict(zip(self._feature_names, self._feature_types)))
        return temp_df.select_dtypes(include=include, exclude=exclude).columns.tolist()