"""检验统计量：Newey-West、GRS、单调性检验、Romano-Wolf 多重检验调整。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st

NW = {"cov_type": "HAC", "use_t": True}


def nw_fit(y, X=None, lag=4):
    d = y.to_frame("y") if X is None else pd.concat([y.rename("y"), X], axis=1)
    d = d.dropna()
    x = np.ones((len(d), 1)) if X is None else sm.add_constant(d.drop(columns="y"))
    return sm.OLS(d["y"], x).fit(cov_kwds={"maxlags": lag, "use_correction": True}, **NW)


def mean_t(y, lag=4):
    m = nw_fit(y, None, lag)
    return float(m.params.iloc[0]), float(m.tvalues.iloc[0]), float(m.pvalues.iloc[0])


def alpha(y, F, lag=4):
    """时间序列回归的截距。返回 alpha、t、p、R²、因子载荷和 alpha + 残差序列。"""
    m = nw_fit(y, F, lag)
    return {"alpha": float(m.params["const"]), "t": float(m.tvalues["const"]), "p": float(m.pvalues["const"]),
            "se": float(m.bse["const"]), "r2": float(m.rsquared), "betas": m.params.drop("const"),
            "tb": m.tvalues.drop("const"), "e": m.resid + m.params["const"], "n": int(m.nobs)}


def grs(R, F):
    """Gibbons-Ross-Shanken：R 为超额收益的测试资产 T×N，F 为因子 T×K。"""
    d = pd.concat([R, F], axis=1).dropna()
    R_, F_ = d[R.columns].to_numpy(), d[F.columns].to_numpy()
    T, N, K = len(d), R_.shape[1], F_.shape[1]
    X = np.column_stack([np.ones(T), F_])
    b = np.linalg.lstsq(X, R_, rcond=None)[0]
    e = R_ - X @ b
    a, S = b[0], e.T @ e / (T - K - 1)
    mu, Om = F_.mean(0), np.atleast_2d(np.cov(F_, rowvar=False))
    stat = (T - N - K) / N * (a @ np.linalg.solve(S, a)) / (1 + mu @ np.linalg.solve(Om, mu))
    r2 = 1 - e.var(0) / R_.var(0)
    return {"GRS": stat, "p": 1 - st.f.cdf(stat, N, T - N - K), "mean_abs_alpha": np.abs(a).mean(),
            "mean_r2": r2.mean(), "sr2_factors": mu @ np.linalg.solve(Om, mu), "T": T, "N": N}


def stationary_index(T, B, block, rng):
    """平稳自助法（Politis-Romano）的时间下标 B×T。"""
    idx = np.empty((B, T), dtype=np.int32)
    idx[:, 0] = rng.integers(0, T, B)
    new = rng.random((B, T)) < 1 / block
    draw = rng.integers(0, T, (B, T))
    for t in range(1, T):
        idx[:, t] = np.where(new[:, t], draw[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def monotonic(P, B, block, seed):
    """Patton-Timmermann 单调性检验。P：T×N 组合收益，低壳概率到高壳概率。
    原假设：相邻组差值的最小值 ≤ 0；备择：严格单调上升。"""
    D = np.diff(P.dropna().to_numpy(), axis=1)
    obs = D.mean(0).min()
    idx = stationary_index(len(D), B, block, np.random.default_rng(seed))
    Dc = D - D.mean(0)
    return obs, float((Dc[idx].mean(1).min(1) >= obs).mean())


def romano_wolf(E, se, B, block, seed):
    """Romano-Wolf 逐步下降 max-t。E：T×K 的 alpha + 残差序列，se：各策略的 NW 标准误（因子载荷视为已知）。"""
    E = E.to_numpy()
    obs = np.abs(np.nanmean(E, 0) / se)
    Ec = np.nan_to_num(E - np.nanmean(E, 0))
    idx = stationary_index(len(E), B, block, np.random.default_rng(seed))
    tb = np.abs(np.stack([Ec[i].mean(0) for i in idx]) / se)
    order = np.argsort(-obs)
    adj = np.empty(len(obs))
    for j, k in enumerate(order):
        adj[k] = (tb[:, order[j:]].max(1) >= obs[k]).mean()
    adj[order] = np.maximum.accumulate(adj[order])
    return adj


def fama_macbeth(d, y, xs, lag=4, weight=None, fe=None):
    """逐月横截面 OLS。xs 第一项是关注变量；连续控制变量已在外部标准化。fe：按该列做组内去均值（行业固定效应）。"""
    rows = []
    for mth, g in d.groupby("month", sort=True):
        g = g.dropna(subset=[y, *xs])
        if len(g) < len(xs) + 30:
            continue
        Y, X = g[y].to_numpy(float), g[xs].to_numpy(float)
        if fe is not None:
            k = g[fe].to_numpy()
            Y = Y - pd.Series(Y).groupby(k).transform("mean").to_numpy()
            X = X - pd.DataFrame(X).groupby(k).transform("mean").to_numpy()
            A = X
        else:
            A = np.column_stack([np.ones(len(g)), X])
        w = np.sqrt(g[weight].to_numpy(float)) if weight else 1.0
        wa = A * (w[:, None] if weight else 1.0)
        b = np.linalg.lstsq(wa, Y * w, rcond=None)[0]
        res = Y - A @ b
        r2 = 1 - res.var() / Y.var() if Y.var() > 0 else np.nan
        rows.append({"month": mth, **dict(zip(xs, b[-len(xs):])), "r2": r2, "n": len(g)})
    C = pd.DataFrame(rows).set_index("month")
    out = {"r2": C["r2"].mean(), "n": C["n"].mean(), "T": len(C)}
    for x in xs:
        out[x], out[f"t_{x}"], _ = mean_t(C[x], lag)
    return out
