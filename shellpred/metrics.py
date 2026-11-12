"""评价指标：全部指标只在这里实现一次，Top-k 的比例由参数传入。"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score


def within_year_auc(y, s, years):
    """先在每年内转成百分位排名，再合并计算 AUC，与样本外评价的口径一致。
    直接合并计算会奖励"分辨哪一年借壳更多"：训练窗口内只知道年份就能得到约 0.7 的 AUC。"""
    r = pd.Series(np.asarray(s)).groupby(np.asarray(years)).rank(pct=True).to_numpy()
    return roc_auc_score(np.asarray(y), r)


def _top(s, pct):
    k = max(1, int(np.ceil(len(s) * pct)))
    return np.argsort(-s, kind="stable")[:k]


def lift(y, s, pct):
    """前 pct 样本中的正例率 / 总体正例率。1 表示与随机排序无异。"""
    return y[_top(s, pct)].mean() / y.mean() if y.sum() else np.nan


def recall(y, s, pct):
    """前 pct 样本覆盖了多少比例的正例（捕获率）。"""
    return y[_top(s, pct)].sum() / y.sum() if y.sum() else np.nan


def mcfadden_r2(y, p, eps=1e-6):
    """1 - 模型对数似然 / 只用基准率的对数似然。"""
    p, b = np.clip(p, eps, 1 - eps), y.mean()
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll0 = len(y) * (b * np.log(b) + (1 - b) * np.log(1 - b))
    return 1 - ll / ll0


def summarize(y, p, top_k):
    y, p = np.asarray(y, float), np.asarray(p, float)
    both = 0 < y.sum() < len(y)
    out = {
        "N": len(y), "正例": int(y.sum()), "正例率": y.mean(),
        "AUC": roc_auc_score(y, p) if both else np.nan,
        "PR_AUC": average_precision_score(y, p) if both else np.nan,
        "Brier": np.mean((p - y) ** 2),
        "R2": r2_score(y, p),
        "McFadden_R2": mcfadden_r2(y, p) if both else np.nan,
        "RMSE_Y1": np.sqrt(np.mean((1 - p[y == 1]) ** 2)) if y.sum() else np.nan,
    }
    for k in top_k:
        out[f"Lift@{k:.0%}"] = lift(y, p, k)
        out[f"Recall@{k:.0%}"] = recall(y, p, k)
    return out


def within_year_topk(df, pct, col="p"):
    """每年各自取前 pct（与按年/按月分组构建组合一致），再把各年的命中数加总。"""
    hits = picked = 0
    for _, g in df.groupby("year"):
        idx = _top(g[col].to_numpy(), pct)
        hits += g["Y"].to_numpy()[idx].sum()
        picked += len(idx)
    base = df["Y"].mean()
    return {"lift": hits / picked / base, "recall": hits / df["Y"].sum()}


def _weighted_auc(y, s, W):
    """W 的每一行是一组样本权重，返回每组权重下的 AUC（同分算一半），与 roc_auc_score(sample_weight=w) 相同。"""
    order = np.argsort(s, kind="stable")
    s, y, W = s[order], y[order], W[:, order]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])  # 同分组的起点
    pos = np.add.reduceat(W * y, start, axis=1)
    neg = np.add.reduceat(W * (1 - y), start, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + neg / 2)).sum(axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))


def bootstrap_auc(y, scores: dict, n, groups, seed=0, chunk=100):
    """按组（公司）整体重抽样 n 次，对同一组自助样本计算各模型的 AUC，可直接相减得到配对差值的分布。
    每次重抽样表示为各观测被抽中的次数，作为 AUC 的样本权重，等价于把观测复制相应的份数。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    codes = np.unique(np.asarray(groups), return_inverse=True)[1]
    g = codes.max() + 1
    counts = np.stack([np.bincount(rng.integers(0, g, g), minlength=g) for _ in range(n)]).astype(float)
    res = {k: [] for k in scores}
    for i in range(0, n, chunk):
        W = counts[i:i + chunk][:, codes]
        ok = ((W * y).sum(axis=1) > 0) & ((W * (1 - y)).sum(axis=1) > 0)
        for k, s in scores.items():
            res[k].append(_weighted_auc(y, np.asarray(s, float), W[ok]))
    return {k: np.concatenate(v) for k, v in res.items()}


def boot_pvalue(diff):
    """双侧自助法 p 值 = 2 × min(差值 ≤ 0 的比例, 差值 ≥ 0 的比例)。"""
    return min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean()))


def holm(p, alpha=0.05):
    """Holm 逐步校正：p 值从小到大第 i 个与 alpha / (m - i + 1) 比较，一旦不显著，后面全部判为不显著。
    返回调整后的 p 值和是否显著。"""
    p = np.asarray(p, float)
    m, order = len(p), np.argsort(p)
    adj = np.empty(m)
    adj[order] = np.minimum(1, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    return adj, adj <= alpha


def rank_stability(d, pct=0.10):
    """d：Stkcd, year, r（年内百分位排名）。返回相邻年排名的 Spearman 相关（两年都出现的公司）
    和前 pct 名单的年换手率（t+1 年前 pct 中不在 t 年前 pct 的比例），各对相邻年取平均。"""
    top = {y: set(g.loc[g["r"].rank(ascending=False, method="first") <= np.ceil(len(g) * pct), "Stkcd"])
           for y, g in d.groupby("year")}
    wide = d.pivot(index="Stkcd", columns="year", values="r")
    years = sorted(top)
    corr = [wide[[a, b]].dropna().corr(method="spearman").iat[0, 1] for a, b in zip(years, years[1:])]
    turnover = [len(top[b] - top[a]) / len(top[b]) for a, b in zip(years, years[1:])]
    return {"相邻年排名相关": np.mean(corr), f"前{pct:.0%}年换手率": np.mean(turnover), "年份对数": len(corr)}
