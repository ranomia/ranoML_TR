import numpy as np

class StratifiedShuffledGroupKFold:
    """
    層化＋グループ別でデータを分割するクロスバリデーション用クラス

    パラメータ:
      n_splits: int
          分割するfold数（デフォルトは5）
      shuffle: bool
          グループをシャッフルするか否か（デフォルトはTrue）
      random_state: int または None
          シャッフルのシード（デフォルトはNone）
    
    層化軸: y
    グループ軸: groups
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups):
        """
        データを分割するジェネレータを返します。

        引数:
          X: array-likeまたはpandas DataFrame
             入力データ（実際の値部分）
          y: array-like
             層化用のラベル
          groups: array-like
             グループ情報

        各foldについて、(train_index, test_index) のタプルをyieldします。
        """
        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)
        
        # 各グループに対して、一意の層化ラベルが存在するか確認し、マッピングを作成
        unique_groups = np.unique(groups)
        group_to_label = {}
        for grp in unique_groups:
            idx = np.where(groups == grp)[0]
            unique_labels = np.unique(y[idx])
            if len(unique_labels) != 1:
                raise ValueError(f"グループ {grp} に対して複数の層化ラベルが存在します: {unique_labels}")
            group_to_label[grp] = unique_labels[0]
        
        # 層化ラベルごとにグループを分ける
        unique_labels = np.unique(list(group_to_label.values()))
        # 各foldのテストインデックスのリストを格納する辞書
        fold_indices = {i: [] for i in range(self.n_splits)}
        
        # ランダムシードを使用してシャッフルするためのRandomStateを用意
        rng = np.random.RandomState(self.random_state)
        
        # 各層（ラベル）ごとに、該当するグループをn_splits個に分割し、foldごとのテストインデックスに追加
        for label in unique_labels:
            groups_label = [grp for grp, lab in group_to_label.items() if lab == label]
            if self.shuffle:
                rng.shuffle(groups_label)
            # np.array_splitを利用することで、なるべく均等にグループ分割を行う
            split_groups = np.array_split(groups_label, self.n_splits)
            for fold, group_subset in enumerate(split_groups):
                for grp in group_subset:
                    fold_indices[fold].extend(np.where(groups == grp)[0].tolist())
        
        # 各foldごとにテストインデックスと、その補集合を訓練インデックスとしてyield
        n_samples = len(X)
        for fold in range(self.n_splits):
            test_idx = np.array(fold_indices[fold])
            train_idx = np.array([i for i in range(n_samples) if i not in test_idx])
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
