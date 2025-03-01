import os
import platform
import matplotlib as mpl
import pandas as pd

class Config:
    def __init__(self):
        self.set_font()

        self._tuning_seed = 75
        self._cv_seed = 31
        self._target_column = 'median_house_value'
        self._group_column = None # group無の場合はNoneを指定
        self._strata_column = ''
        self._exclude_columns = []

    @staticmethod
    def set_font():
        os_name = platform.system()
        if os_name == 'Windows':  # Windows
            mpl.rcParams['font.family'] = 'BIZ UDGothic'
        elif os_name == 'Darwin':  # macOS
            mpl.rcParams['font.family'] = 'Hiragino Sans'
        elif os_name == 'Linux':  # Linux
            mpl.rcParams['font.family'] = 'Noto Sans CJK JP'
        else:
            raise EnvironmentError(f'Unsupported OS: {os_name}')
            
    @property
    def tuning_seed(self):
        return self._tuning_seed
    
    @property
    def cv_seed(self):
        return self._cv_seed

    @property
    def target_column(self):
        return self._target_column
    
    @property
    def group_column(self):
        return self._group_column
    
    @property
    def strata_column(self):
        return self._strata_column
    
    @property
    def exclude_columns(self):
        """モデリングから除外するカラムのリストを返す"""
        return self._exclude_columns