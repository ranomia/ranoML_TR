# ranoML_TR
ranoMLのTableRegressor。テーブルデータの回帰問題を解きます。

## 概要
このプロジェクトは、テーブルデータの回帰問題を解くための機械学習フレームワークです。LightGBMを主要なモデルとして使用し、以下の機能を提供します：

- 層化K分割交差検証
- グループ別の層化K分割交差検証
- ハイパーパラメータのチューニング（Optuna）
- モデルの評価と可視化
  - 学習曲線
  - 残差プロット
  - SHAP値による特徴量重要度
  - カテゴリ別の予測性能評価

## フォルダ構成
```
ranoML_TR/
├── data/      # データファイル（.gitignore）
├── jupyter/   # Jupyter notebooks（.gitignore）
├── model/     # 学習済みモデル（.gitignore）
├── src/       # ソースコード
  ├── main.py  # メイン実行ファイル
  ├── core/    # コア機能
  │ ├── config.py           # 設定ファイル
  │ ├── inner_cv_runner.py  # 内部CV実行
  │ └── outer_cv_runner.py  # 外部CV実行
  ├── evaluation/           # 評価機能
  │ └── model_evaluator.py  # モデル評価
  ├── preprocess/           # 前処理
  │ └── feature_selector.py # 特徴量選択
  └── util/                 # ユーティリティ
    ├── index_saver.py      # インデックス保存
    ├── stratified_shuffled_group_kfold.py # 層化グループKFold
    └── util.py             # 汎用ユーティリティ
```

## 主要な機能

### OuterCVRunner
- モデルの学習と評価を行うメインクラス
- クロスバリデーションの実行
- モデルの保存と予測
- 複数のデータフォーマットに対応（Parquet, CSV, Excel, Pickle, JSONL）

### InnerCVRunner
- ハイパーパラメータのチューニングを行うクラス
- Optunaを使用した最適化
- 以下のパラメータを最適化：
  - learning_rate
  - feature_fraction
  - num_leaves
  - subsample
  - reg_lambda
  - reg_alpha
  - min_data_in_leaf

### ModelEvaluator
- 各種評価指標の計算（RMSE, MAE, RMSPE）
- 学習曲線の可視化
- 残差プロットの作成
- SHAP値による特徴量重要度の可視化

### FeatureSelector
- 特徴量の選択と管理
- 特徴量情報の保存と読み込み
- データ型の自動変換（bool, int, float, category）

## 使用方法
```
cd src
python main.py
```

