"""平衡数据的方法 S0–S5。一切抽样只在训练窗口内进行，测试年份始终是真实分布。

每个方法返回 Sample：一个或多个训练子样本（S2、S3 做 bagging），先验校正用的 β，
以及第一个子样本在训练窗口中的行号（论文表 4 的 train_used；SMOTE 的合成样本没有行号，只返回原始行）。"""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

RESAMPLED = {"s2", "s3", "s4", "s5"}  # 改变了训练集构成的方法


@dataclass
class Bag:
    X: np.ndarray
    y: np.ndarray
    weight: np.ndarray | None = None


@dataclass
class Sample:
    bags: list[Bag]
    beta: float
    rows: np.ndarray

    @property
    def n_used(self) -> int:
        return len(self.bags[0].y)


def prior_correct(p, beta):
    """先验校正（Dal Pozzolo et al., 2015）：p = β·pₛ / (β·pₛ − pₛ + 1)。
    β = 保留的负例比例，或正例权重 w 的倒数。单调变换，不改变排序，只把概率水平还原到真实基准率。"""
    return beta * p / (beta * p - p + 1)


def undersample(y, ratio, rng):
    """保留全部正例，无放回随机抽取 min(负例数, ratio × 正例数) 个负例，返回行号。"""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    keep = rng.choice(neg, size=min(len(neg), ratio * len(pos)), replace=False)
    return np.sort(np.concatenate([pos, keep]))


def _subsets(X, y, rows: list[np.ndarray]) -> Sample:
    beta = (y[rows[0]] == 0).sum() / (y == 0).sum()  # 各子样本保留的负例数相同
    return Sample([Bag(X[r], y[r]) for r in rows], beta, rows[0])


def s0(X, y):
    """全数据。"""
    return Sample([Bag(X, y)], 1.0, np.arange(len(y)))


def s1(X, y, w):
    """类权重：正例权重 w、负例 1，β = 1/w。"""
    return Sample([Bag(X, y, np.where(y == 1, float(w), 1.0))], 1 / w, np.arange(len(y)))


def s2(X, y, ratio, rngs):
    """随机欠采样 + bagging。"""
    return _subsets(X, y, [undersample(y, ratio, rng) for rng in rngs])


def s3(X, y, years, ratio, rngs):
    """按年匹配欠采样 + bagging：每年保留全部正例，从同一年的负例中抽 ratio 倍；没有正例的年份不进入训练集。"""
    groups = [g for yr in np.unique(years) if y[g := np.flatnonzero(years == yr)].any()]
    return _subsets(X, y, [np.sort(np.concatenate([g[undersample(y[g], ratio, rng)] for g in groups]))
                           for rng in rngs])


def s4(X, Z, y, years, ratio, C):
    """PSM 式匹配（对照）：Z（插补版）上拟合倾向得分逻辑回归，每个正例在同一年负例中取倾向得分
    （对数几率）最接近的 ratio 个，匹配允许重叠、取并集。确定性，只做一次。"""
    ps = LogisticRegression(C=C, max_iter=2000).fit(Z, y).decision_function(Z)
    keep = [np.flatnonzero(y == 1)]
    for yr in np.unique(years[y == 1]):
        pos = np.flatnonzero((years == yr) & (y == 1))
        neg = np.flatnonzero((years == yr) & (y == 0))
        k = min(ratio, len(neg))
        dist = np.abs(ps[pos, None] - ps[None, neg])
        keep.append(neg[np.argpartition(dist, k - 1, axis=1)[:, :k]].ravel())
    return _subsets(X, y, [np.unique(np.concatenate(keep))])


def s5(Z, y, ratio, k, binary, rng):
    """SMOTE 过采样（对照）：在插补版标准化空间中，合成 m = max(0, ⌈n₀/ratio⌉ − n₁) 个正例，
    新样本 = a + u·(b − a)，b 是 a 的 K 个最近正例邻居之一；binary 列取该列实际出现过的最近取值（0/1）。"""
    pos = Z[y == 1]
    n1, n0 = len(pos), int((y == 0).sum())
    m, K = max(0, int(np.ceil(n0 / ratio)) - n1), min(k, n1 - 1)
    dist = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :K]
    a = rng.integers(0, n1, m)
    b = nn[a, rng.integers(0, K, m)]
    new = pos[a] + rng.random((m, 1)) * (pos[b] - pos[a])
    for j in binary:
        v = np.unique(Z[:, j])
        new[:, j] = v[np.abs(new[:, j, None] - v).argmin(axis=1)]
    bag = Bag(np.vstack([Z, new]), np.concatenate([y, np.ones(m, y.dtype)]))
    return Sample([bag], n1 / (n1 + m), np.arange(len(y)))
