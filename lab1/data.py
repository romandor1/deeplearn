"""Завантаження Iris, стратифікований поділ 70/30 та стандартизація."""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris

N_TRAIN_PER_CLASS = 35


def load_split_standardize():
    """Повертає (X_train, y_train, X_test, y_test) у float64.

    Поділ: rng = default_rng(0); для класів 0, 1, 2 по черзі перемішуємо
    індекси класу (rng.shuffle) і беремо перші 35 у train, решту 15 у test.
    Стандартизація: mean та std (ddof=0) лише за train; ті самі статистики
    застосовуються до test.
    """
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for c in (0, 1, 2):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        train_idx.append(idx[:N_TRAIN_PER_CLASS])
        test_idx.append(idx[N_TRAIN_PER_CLASS:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    assert X_train.shape == (105, 4) and X_test.shape == (45, 4)
    assert np.bincount(y_train).tolist() == [35, 35, 35]
    assert np.bincount(y_test).tolist() == [15, 15, 15]
    return X_train, y_train, X_test, y_test
