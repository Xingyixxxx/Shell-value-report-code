"""组合构建：分组规则（单变量、双重、中性化、复合）和组合月收益。"""
import numpy as np
import pandas as pd


def qgroup(ch, s, n, by=None, mask=None):
    """按年（及 by 分桶）等分为 n 组，1 = 最低；mask 之外的公司不参与排序、不进组合。"""
    # 并列值（如平方误差 XGBoost 截到 0 的预测）用固定种子打破，否则最低组可能凑不出来
    s = pd.Series(np.asarray(s, float) + 1e-9 * np.random.default_rng(0).random(len(ch)), index=ch.index)
    if mask is not None:
        s = s.where(mask)
    keys = [ch["year"]] + ([pd.Series(by, index=ch.index)] if by is not None else [])
    return np.ceil(s.groupby(keys).rank(pct=True) * n).clip(1, n).to_numpy()


def cut_group(ch, s, cuts=(.3, .7), by=None, mask=None):
    """按分位点分组（如 30/40/30），1 = 最低；两值变量也用这个函数。"""
    s = pd.Series(np.asarray(s, float), index=ch.index)
    if mask is not None:
        s = s.where(mask)
    keys = [ch["year"]] + ([pd.Series(by, index=ch.index)] if by is not None else [])
    pct = s.groupby(keys).rank(pct=True)
    return np.where(pct.isna(), np.nan, np.searchsorted(cuts, pct.to_numpy(), side="left") + 1.0)


def topk(ch, s, k):
    """每年信号最高的 k 家为 2 组，最低的 k 家为 1 组，其余不进组合。"""
    s = pd.Series(np.asarray(s, float), index=ch.index)
    g = s.groupby(ch["year"])
    return np.where(g.rank(ascending=False, method="first") <= k, 2.0, np.where(g.rank(method="first") <= k, 1.0, np.nan))


def port(mp, group, weight="ew", ret="ret", include_susp=False):
    """各组月收益：month × 组。group 是按 chars 行号（cid）排列的数组。"""
    d = mp if include_susp else mp[~mp["susp"].astype(bool)]
    x = pd.DataFrame({"m": d["month"].to_numpy(), "g": group[d["cid"].to_numpy()],
                      "r": d[ret].fillna(0).to_numpy() if include_susp else d[ret].to_numpy(),
                      "w": d[f"w_{weight}"].to_numpy()}).dropna()
    x["rw"] = x["r"] * x["w"]
    g = x.groupby(["m", "g"])[["rw", "w"]].sum()
    return (g["rw"] / g["w"]).unstack("g")


def long_short(P):
    return (P.iloc[:, -1] - P.iloc[:, 0]).dropna()


def rank_weighted(mp, s, ret="ret"):
    """不分组：权重正比于（信号 − 当月均值），多空两边各归一为 1。"""
    d = mp[~mp["susp"].astype(bool)]
    x = pd.DataFrame({"m": d["month"].to_numpy(), "s": np.asarray(s, float)[d["cid"].to_numpy()], "r": d[ret].to_numpy()}).dropna()
    x["w"] = x["s"] - x.groupby("m")["s"].transform("mean")
    x["w"] /= x["w"].abs().groupby(x["m"]).transform("sum") / 2
    return (x["w"] * x["r"]).groupby(x["m"]).sum()


def neutral_rank(ch, s, numeric=(), ind=False):
    """逐年把信号对控制变量做横截面回归，残差再排序。控制变量缺失用当年中位数补。"""
    out = pd.Series(np.nan, index=ch.index)
    for _, g in ch.groupby("year"):
        X = [np.ones(len(g))] + [g[c].fillna(g[c].median()).fillna(0).to_numpy(float) for c in numeric]
        if ind:
            X += list(pd.get_dummies(g["ind"], drop_first=True, dtype=float).to_numpy().T)
        X, y = np.column_stack(X), np.asarray(s, float)[g.index]
        ok = ~np.isnan(y)
        out.loc[g.index[ok]] = y[ok] - X[ok] @ np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return out.groupby(ch["year"]).rank(pct=True).to_numpy()


def annual_turnover(ch, group, top):
    """每年 5 月调仓时，top 组里新进入的名字占比（组合内部等权近似），对全部相邻年取平均。"""
    d = ch.assign(g=group)[lambda x: x["g"] == top]
    sets = {y: set(v["Stkcd"]) for y, v in d.groupby("year")}
    ys = sorted(sets)
    return float(np.mean([1 - len(sets[a] & sets[b]) / max(len(sets[b]), 1) for a, b in zip(ys, ys[1:])]))


def stats(r):
    """年化收益、波动、夏普（近似）和最大回撤。"""
    r = r.dropna()
    v = (1 + r).cumprod()
    return {"ann_ret": r.mean() * 12, "ann_vol": r.std() * np.sqrt(12), "sharpe": r.mean() / r.std() * np.sqrt(12),
            "mdd": (v / v.cummax().clip(lower=1) - 1).min(), "win": (r > 0).mean()}
