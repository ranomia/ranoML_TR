import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List

from core.config import Config

class ModelEvaluator:
    """モデルの評価と可視化を行うクラス"""
    
    def __init__(self, model_type: str, output_dir: str, run_name: str, data: pd.DataFrame = None):
        """
        Args:
            output_dir (str): 出力ディレクトリのパス
            run_name (str): 実験名
            data (pd.DataFrame): カテゴリ情報を含むデータフレーム
        """
        self.model_type = model_type
        self.output_dir = output_dir
        self.run_name = run_name
        self.data = data
        self.config = Config()  # Configのインスタンス化
        
    def evaluate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """各種評価指標を計算

        Args:
            y_true: 正解値
            y_pred: 予測値

        Returns:
            Dict[str, float]: 評価指標の辞書
        """
        return {
            'rmse': root_mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred)
        }

    def plot_learning_curve(
        self,
        eval_results: Dict[str, Dict[str, List[float]]],
        fold: int,
        save_path: str = None
    ) -> None:
        """学習曲線をプロット

        Args:
            eval_results: 学習履歴
            fold: フォールド番号
            y_limit: y軸の範囲
            save_path: 保存先のパス（デフォルト: None）
        """
        fig = plt.figure(figsize=(10, 6))

        if self.model_type == 'lightgbm':
            plt.plot(eval_results['train']['rmse'], label='Training Loss')
            plt.plot(eval_results['valid']['rmse'], label='Validation Loss')
        elif self.model_type == 'xgboost':
            plt.plot(eval_results['validation_0']['rmse'], label='Training Loss')
            plt.plot(eval_results['validation_1']['rmse'], label='Validation Loss')
        elif self.model_type == 'catboost':
            plt.plot(eval_results['learn']['rmse'], label='Training Loss')
            plt.plot(eval_results['validatation']['rmse'], label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Learning Curve')
        plt.legend()
        plt.ylim(bottom=0)
        
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)

    def plot_scatter_by_category(self, y_true: pd.Series, y_pred: pd.Series, category_col: str , 
                            figsize: tuple = (7.5, 6), save_path: str = None) -> None:
        """
        カテゴリ別に異なるマーカーを使用して実測値と予測値の散布図を描画

        Args:
            y_true (pd.Series): 実測値
            y_pred (pd.Series): 予測値
            category_col (str): カテゴリ列の名前
            figsize (tuple): プロットのサイズ（デフォルト: (10, 6)）
            save_path (str): 保存先のパス（デフォルト: None）
        """
        try:
            # 実行前に開いている全ての図を閉じる
            plt.close('all')
            
            # データフレームの作成
            df_plot = pd.DataFrame({
                '観測値': y_true,
                '予測値': y_pred
            })
            
            # カテゴリ情報の結合
            if category_col in self.data.columns:
                # カテゴリカル型の場合は、一旦文字列に変換してから処理
                category_data = self.data[category_col].copy()
                if pd.api.types.is_categorical_dtype(category_data):
                    category_data = category_data.astype(str)

                df_plot[category_col] = category_data.iloc[y_true.index].values
            else:
                raise ValueError(f"カラム '{category_col}' がデータフレームに存在しません")

            # 欠損値の処理
            df_plot[category_col] = df_plot[category_col].fillna('不明')

            # プロットの作成
            fig = plt.figure(figsize=figsize)
            
            # y=xの線を描画
            min_val = min(df_plot['観測値'].min(), df_plot['予測値'].min())
            max_val = max(df_plot['観測値'].max(), df_plot['予測値'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 
                    color='gray', linestyle='--', zorder=1)

            # カテゴリ別の散布図
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # マーカーのリスト
            categories = df_plot[category_col].unique()
            
            for i, category in enumerate(categories):
                mask = df_plot[category_col] == category
                plt.scatter(df_plot.loc[mask, '観測値'], 
                        df_plot.loc[mask, '予測値'],
                        label=category,
                        marker=markers[i % len(markers)],
                        alpha=0.5,
                        s=30)

            # グラフの設定
            plt.xlabel('実測値')
            plt.ylabel('予測値')
            plt.title(f'{self.config.target_column}の予測性能（カテゴリ別）')
            # 凡例の設定
            plt.legend(title=category_col, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', borderaxespad=0.)
            # グリッド線の追加
            plt.grid(True, linestyle='--', alpha=0.3)
            # アスペクト比を1:1に設定
            plt.axis('equal')
            # 余白の調整
            plt.tight_layout()
            # 保存
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)

        except Exception as e:
            print(f"散布図の作成中にエラーが発生しました: {str(e)}")
            raise

    def plot_scatter_by_continuous(self, y_true: pd.Series, y_pred: pd.Series, color_col: str,
                                 figsize: tuple = (7.5, 6), save_path: str = None) -> None:
        """
        連続値で色分けした実測値と予測値の散布図を描画

        Args:
            y_true (pd.Series): 実測値
            y_pred (pd.Series): 予測値
            color_col (str): 色分けに使用する連続値の列名
            figsize (tuple): プロットのサイズ（デフォルト: (7.5, 6)）
            save_path (str): 保存先のパス（デフォルト: None）
        """
        try:
            # 実行前に開いている全ての図を閉じる
            plt.close('all')
            
            # データフレームの作成
            df_plot = pd.DataFrame({
                '観測値': y_true,
                '予測値': y_pred
            })
            
            # 色分け用の連続値を結合 (インデックスを明示的に合わせる)
            if color_col in self.data.columns:
                color_data = self.data[color_col].copy()
                df_plot[color_col] = color_data.iloc[y_true.index].values
            else:
                raise ValueError(f"カラム '{color_col}' がデータフレームに存在しません")

            # プロットの作成
            fig = plt.figure(figsize=figsize)
            
            # y=xの線を描画
            min_val = min(df_plot['観測値'].min(), df_plot['予測値'].min())
            max_val = max(df_plot['観測値'].max(), df_plot['予測値'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 
                    color='gray', linestyle='--', zorder=1)

            # 散布図の描画（連続値による色分け）
            scatter = plt.scatter(df_plot['観測値'], df_plot['予測値'],
                                c=df_plot[color_col], cmap='viridis',
                                alpha=0.5, s=30, zorder=2)
            # カラーバーの追加
            plt.colorbar(scatter, label=color_col)
            # グラフの設定
            plt.xlabel('観測値')
            plt.ylabel('予測値')
            plt.title(f'Scatter Plot (colored by {color_col})')
            # グリッド線の追加
            plt.grid(True, linestyle=':', alpha=0.6)
            # アスペクト比を1:1に設定
            plt.axis('equal')
            # 保存
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
        except Exception as e:
            print(f"散布図の作成中にエラーが発生しました: {str(e)}")
            raise

    def plot_residuals(
        self,
        train_true: np.ndarray,
        train_pred: np.ndarray,
        valid_true: np.ndarray,
        valid_pred: np.ndarray,
        fold: int,
        save_path: str = None
    ) -> None:
        """残差プロットを作成

        Args:
            train_true: 学習データの正解値
            train_pred: 学習データの予測値
            valid_true: 検証データの正解値
            valid_pred: 検証データの予測値
            fold: フォールド番号
            save_path: 保存先のパス（デフォルト: None）
        """
        tr_res = train_true - train_pred
        va_res = valid_true - valid_pred
        
        res_min = min(tr_res.min(), va_res.min())
        res_max = max(tr_res.max(), va_res.max())
        
        fig = plt.figure(figsize=(14, 6))
        
        # 学習データの残差プロット
        plt.subplot(1, 2, 1)
        self._plot_residual_subplot(train_true, tr_res, res_min, res_max, 'Train')
        
        # 検証データの残差プロット
        plt.subplot(1, 2, 2)
        self._plot_residual_subplot(valid_true, va_res, res_min, res_max, 'Validation')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close(fig)

    def _plot_residual_subplot(
        self,
        x: np.ndarray,
        residuals: np.ndarray,
        res_min: float,
        res_max: float,
        data_type: str
    ) -> None:
        """残差プロットのサブプロット作成

        Args:
            x: x軸の値（正解値）
            residuals: 残差
            res_min: 残差の最小値
            res_max: 残差の最大値
            data_type: データ種別（'Train'または'Validation'）
        """
        plt.scatter(x, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.ylim(res_min*1.2, res_max*1.2)
        plt.xlabel(f'Real Values ({data_type})')
        plt.ylabel(f'Residuals ({data_type})')
        plt.title(f'Residual Plot ({data_type})')

    def plot_shap_values(
        self,
        model,
        valid_data: pd.DataFrame,
        feature_names: List[str],
        fold: int,
        save_path: str = None
    ) -> None:
        """SHAP値の可視化

        Args:
            model: 学習済みモデル
            valid_data: 検証データ
            feature_names: 特徴量名のリスト
            fold: フォールド番号
            save_path: 保存先のパス（デフォルト: None）
        """
        explainer = shap.TreeExplainer(
            model,
            feature_names=feature_names,
            feature_perturbation="interventional"
        )
        shap_values = explainer.shap_values(valid_data)
        
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values=shap_values,
            features=valid_data,
            feature_names=feature_names,
            show=False
        )
        plt.title(f'SHAP Summary Plot for {self.run_name}_{fold}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.close(fig)