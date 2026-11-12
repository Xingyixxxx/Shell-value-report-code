"""报告第二版（report/报告_v2.md）新增的预测部分结果。不改动 evaluate 的任何输出，只读已有的预测。

1. 平衡方法与主设定的 AUC 显著性：沿用 evaluate 的按公司整体重抽样（同样的次数和种子），
   AUC 对 0.5 的检验、相对同一模型全数据版本的配对差值检验（另给 Holm 调整 p）。
2. 分政策阶段的样本外 AUC 与 Lift：2012–2016（宽松期）、2017–2019（严控期）、2020–2023（放松与注册制）。
3. 逻辑回归 L1 · 全数据 系数均值的显著性：按公司整体重抽样，在每个重抽样面板上用各年选中的 C
   重跑 12 个滚动窗口，取 12 年系数均值的自助分布。

    uv run -m shellpred.revision            # 输出到 results/tables/r_*.csv|md
    uv run -m shellpred.revision --coef-boot 200
"""
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import load_config
from .data import build_dataset, label
from .evaluate import ENS, _ranked, _wide, load, scopes
from .metrics import boot_pvalue, bootstrap_auc, holm, within_year_topk
from .models import Panel, Preprocessor
from .plot import save_table

STAGES = [("2012–2016", 2012, 2016), ("2017–2019", 2017, 2019), ("2020–2023", 2020, 2023)]
L1 = "logit_l1_s0"


def stars(p):
    return "" if pd.isna(p) else "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def auc_tables(R, keys, n_boot):
    runs = list(R.ranked)
    w = _wide(R, runs, keys)
    boot = bootstrap_auc(w["Y"], {n: w[n] for n in runs}, n_boot, groups=w["Stkcd"])
    auc = {n: roc_auc_score(w["Y"], w[n]) for n in runs}
    rows = []
    for n in runs:
        rows.append({"run": n, "模型": R.labels[n], "AUC": auc[n], "p_AUC": boot_pvalue(boot[n] - .5)})
    A = pd.DataFrame(rows)

    # 第一步：同一模型内相对 S0 的配对差值；10 个比较一起做 Holm
    diff = []
    for n in R.names:
        m = R.model(n)
        if R.cfg["models"][m]["step"] != 1 or n == f"{m}_s0":
            continue
        d = boot[n] - boot[f"{m}_s0"]
        diff.append({"run": n, "diff_S0": auc[n] - auc[f"{m}_s0"], "p_diff_S0": boot_pvalue(d)})
    D = pd.DataFrame(diff)
    D["p_diff_S0_holm"] = holm(D["p_diff_S0"])[0]
    # 主设定相对逻辑回归 L2 · 全数据（与 evaluate 的基准相同）
    base = R.cfg["evaluate"]["baseline"]
    B = pd.DataFrame([{"run": n, "diff_L2": auc[n] - auc[base], "p_diff_L2": boot_pvalue(boot[n] - boot[base])}
                      for n in R.plot if n != base])
    B["p_diff_L2_holm"] = holm(B["p_diff_L2"])[0]
    return A.merge(D, on="run", how="left").merge(B, on="run", how="left"), w


def stage_table(R, w, n_boot):
    """各阶段内的年内排序 AUC（自助法 p 值）与 Lift@5%/10%（每年各取前 k%，阶段内加总）。"""
    rows = []
    for lab, a, b in [("全样本", 2012, 2023), *STAGES]:
        ws = w[w["year"].between(a, b)]
        boot = bootstrap_auc(ws["Y"], {n: ws[n] for n in R.plot}, n_boot, groups=ws["Stkcd"])
        for n in R.plot:
            d = _ranked(R.ranked[n])
            d = d[d["year"].between(a, b)]
            l5, l10 = within_year_topk(d, .05), within_year_topk(d, .10)
            rows.append({"run": n, "模型": R.labels[n], "阶段": lab, "N": len(ws), "正例": int(ws["Y"].sum()),
                         "AUC": roc_auc_score(ws["Y"], ws[n]), "p_AUC": boot_pvalue(boot[n] - .5),
                         "Lift@5%": l5["lift"], "Lift@10%": l10["lift"], "Recall@10%": l10["recall"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 逻辑回归 L1 系数的自助法

def _l1_fit(X, y, feats, C, seed):
    prep = Preprocessor().fit(X, feats)
    m = LogisticRegression(C=C, l1_ratio=1, solver="saga", max_iter=5000, random_state=seed)
    m.fit(prep.transform(X), y)
    return dict(zip(prep.columns, m.coef_[0]))


def _l1_path(X, y, t, feats, Cs, window, seed):
    """12 个滚动窗口的系数（宽表：年 × 变量）。"""
    out = {}
    for year, C in Cs.items():
        tr = (t >= year - window) & (t < year)
        out[year] = _l1_fit(X[tr], y[tr], feats, C, seed)
    return pd.DataFrame(out).T


def _one_boot(b, X, y, t, firm, feats, Cs, window, seed):
    rng = np.random.default_rng([seed, b])
    ids = np.unique(firm)
    pick = rng.choice(ids, len(ids), replace=True)
    rows = np.concatenate([np.flatnonzero(firm == f) for f in pick])
    return _l1_path(X[rows], y[rows], t[rows], feats, Cs, window, seed).mean()


def l1_coef_boot(cfg, R, B):
    df, feats = build_dataset(cfg, dropna=False)
    P = Panel.from_frame(df, feats, cfg["rolling"]["window"])
    firm = pd.factorize(df["Stkcd"])[0]
    sel = pd.read_csv(cfg["paths"]["outputs"] / L1 / "selection.csv")
    Cs = dict(zip(sel["year"], sel["C"]))
    seed = cfg["tuning"]["seed"]
    point = _l1_path(P.X, P.y, P.time, feats, Cs, P.window_len, seed)
    # 与训练时保存的系数核对（同一数据、同一 C、同一求解器，应几乎相同）
    meta = R.metas[L1].set_index("year")
    saved = meta[[c for c in meta.columns if c.startswith("coef_")]].rename(columns=lambda c: c[5:])
    gap = float(np.nanmax(np.abs(point.reindex(columns=saved.columns).to_numpy() - saved.reindex(point.index).to_numpy())))
    print(f"  L1 系数重拟合与训练记录的最大差异：{gap:.2e}", flush=True)
    boots = Parallel(n_jobs=-1)(delayed(_one_boot)(b, P.X, P.y, P.time, firm, feats, Cs, P.window_len, seed) for b in range(B))
    Bm = pd.DataFrame(boots)
    mean = point.mean()
    rows = []
    for v in mean.index:
        bv = Bm[v].dropna()
        rows.append({"变量": v, "含义": label(v).split(" ", 1)[-1], "系数均值": mean[v], "自助标准误": bv.std(),
                     "p": boot_pvalue(bv.to_numpy()) if bv.std() > 0 else np.nan, "B": len(bv)})
    T = pd.DataFrame(rows)
    T["显著性"] = T["p"].map(stars)
    return T.reindex(T["系数均值"].abs().sort_values(ascending=False).index)


def run(cfg, coef_boot):
    R = load(cfg)
    n_boot = cfg["evaluate"]["bootstrap"]
    keys = scopes(cfg, R.ranked)["全部样本"]
    A, w = auc_tables(R, keys, n_boot)
    save_table(A, cfg, "r_auc_sig", {"AUC": ".4f", "p_AUC": ".4f"})
    print("  AUC 显著性 → r_auc_sig", flush=True)
    S = stage_table(R, w, n_boot)
    save_table(S, cfg, "r_auc_stage", {"p_AUC": ".4f"})
    print("  分阶段 AUC/Lift → r_auc_stage", flush=True)
    if coef_boot:
        T = l1_coef_boot(cfg, R, coef_boot)
        save_table(T, cfg, "r_l1_coef_sig", {"p": ".4f"})
        print("  L1 系数显著性 → r_l1_coef_sig", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coef-boot", type=int, default=200, help="L1 系数自助法次数，0 = 跳过")
    run(load_config(), ap.parse_args().coef_boot)
