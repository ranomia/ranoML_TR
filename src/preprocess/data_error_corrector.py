import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class DataErrorCorrector(BaseEstimator, TransformerMixin):
    """
    データセットの既知の誤りを修正するTransformer
    
    修正内容：
    """
    
    def __init__(self):
        """
        Args:            
        """
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        """データに対してTransformerをフィットさせる
        
        Args:
            X: pandas DataFrame
            y: 使用しない（scikit-learnの規約に従うため含める）
        
        Returns:
            self: このTransformerのインスタンス
        """
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        return self
        
    def transform(self, X):
        """データを変換する
        
        Args:
            X: pandas DataFrame
            
        Returns:
            pandas DataFrame: 変換後のデータ
        """
        check_is_fitted(self)
        X = X.copy()
        
        ### 修正事項をここに記載
        
        return X
    
    def get_feature_names_out(self, feature_names=None):
        """入力特徴量名をそのまま出力特徴量名として返す
        
        Args:
            feature_names: 入力特徴量名のリスト。Noneの場合は入力データフレームのカラム名を使用
            
        Returns:
            list: 出力特徴量名のリスト
        """
        check_is_fitted(self)
        if feature_names is None:
            feature_names = self.feature_names_in_
        return np.asarray(feature_names, dtype=object) 