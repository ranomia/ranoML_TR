import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder as CE_TargetEncoder

class CustomTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_columns, smoothing=10, min_samples_leaf=5):
        """
        Parameters
        ----------
        encoding_columns : list of str
            エンコーディングする特徴量のカラム名リスト
        smoothing : float, default=10
            スムージングパラメータ。大きいほど事前確率の影響が強くなる
        min_samples_leaf : int, default=5
            各カテゴリの最小サンプル数。これより少ないカテゴリは事前確率の影響が強くなる
        """
        self.encoding_columns = encoding_columns
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoders = {}
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        """
        各カラムに対してTargetEncoderをfitする
        """
        # DataFrameに変換
        X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist()
        
        # 各カラムに対してエンコーダーを作成
        for col in self.encoding_columns:
            if col in X.columns:
                encoder = CE_TargetEncoder(
                    cols=[col],
                    smoothing=self.smoothing,
                    min_samples_leaf=self.min_samples_leaf
                )
                self.encoders[col] = encoder.fit(X[[col]], y)
        
        return self
    
    def transform(self, X):
        """
        fit済みのTargetEncoderを使って変換を行う
        """
        # DataFrameに変換
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_transformed = X.copy()
        
        # 各カラムに対してエンコーディングを適用
        for col, encoder in self.encoders.items():
            if col in X.columns:
                # エンコーディング結果を取得
                encoded_col = encoder.transform(X[[col]])
                # カラム名を変更して追加
                new_col_name = f'fe_{col}_target_enc'
                X_transformed[new_col_name] = encoded_col[col]
                # 元のカラムを削除
                X_transformed = X_transformed.drop(columns=[col])
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """
        変換後の特徴量名を返す
        """
        if input_features is None:
            input_features = self.feature_names_in_
        
        # 出力される特徴量名のリストを作成
        output_features = []
        
        # 入力特徴量を1つずつ確認
        for feature in input_features:
            # エンコーディング対象の列の場合
            if feature in self.encoding_columns:
                # エンコードされた新しい列名を追加
                output_features.append(f'fe_{feature}_target_enc')
            else:
                # エンコーディング対象でない列はそのまま追加
                output_features.append(feature)
        
        return output_features

    # def _more_tags(self):
    #     """
    #     このTransformerがDataFrameの構造（列名やインデックス）を保持することをscikit-learnに伝える
    #     """
    #     return {"preserves_dataframe": True}