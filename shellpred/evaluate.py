"""模型评价：排序与概率指标、配对检验与多重比较、平衡方法与模型族的比较、排名稳定性、
特征重要性与预测模式，并导出壳概率。全部运行只报告"全部样本"口径；其余口径和主图只用 plot_runs + 组合。"""
import json
from dataclasses import dataclass, field
from functools import reduce

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from .data import build_dataset, label
from .metrics import (boot_pvalue, bootstrap_auc, holm, lift, rank_stability, summarize,
                      within_year_topk)
from .models import LINEAR
from .plot import BG, INK, INK2, MUTED, SERIES, plt, save_fig, save_table
from .sampling import RESAMPLED, prior_correct

KEY = ["Stkcd", "time"]
ENS = "ensemble"  # 等权组合：输出是年内排名而不是概率，只参与排序类指标


@dataclass
class Results:
    cfg: dict
    names: list            # 已有结果的运行，按 config 顺序
    preds: dict
    metas: dict
    sels: dict
    manifests: dict
    labels: dict = field(default_factory=dict)
    tests: dict = field(default_factory=dict)    # 各运行的测试集预测
    ranked: dict = field(default_factory=dict)   # 参与排序类指标的：全部运行 + 组合
    plot: list = field(default_factory=list)     # 主图与逐年表：plot_runs + 组合
    colors: dict = field(default_factory=dict)   # 按模型族分配的颜色
    auc: dict = field(default_factory=dict)      # 全部样本口径的年内排序 AUC
    boot: dict = field(default_factory=dict)     # 全部样本口径的自助法 AUC（同一组重抽样）

    def model(self, run):
        return ENS if run == ENS else self.cfg["runs"][run]["model"]

    def sampler(self, run):
        return self.cfg["runs"][run]["sampler"]


def load(cfg) -> Results:
    out = cfg["paths"]["outputs"]
    names = [r for r in cfg["runs"] if (out / r / "predictions.csv.gz").exists()]
    if missing := [r for r in cfg["runs"] if r not in names]:
        print(f"跳过尚未训练的运行：{missing}")
    preds = dict(zip(names, Parallel(n_jobs=-1, prefer="threads")(
        delayed(pd.read_csv)(out / n / "predictions.csv.gz", dtype={"Stkcd": str}) for n in names)))
    R = Results(cfg, names, preds,
                metas={n: pd.read_csv(out / n / "meta.csv") for n in names},
                sels={n: pd.read_csv(out / n / "selection.csv") for n in names},
                manifests={n: json.loads((out / n / "manifest.json").read_text(encoding="utf-8")) for n in names})
    ev = cfg["evaluate"]
    R.labels = {n: cfg["runs"][n]["label"] for n in names}
    R.tests = {n: p[p["sample"] == "test"].reset_index(drop=True) for n, p in preds.items()}
    R.ranked = dict(R.tests)
    if all(m in R.tests for m in ev["ensemble"]["members"]):
        R.ranked[ENS], R.labels[ENS] = _ensemble(R.tests, ev["ensemble"]["members"]), ev["ensemble"]["label"]
    R.plot = [n for n in [*ev["plot_runs"], ENS] if n in R.ranked]
    R.colors = {m: SERIES[i % len(SERIES)] for i, m in enumerate([*cfg["models"], ENS])}
    return R


def _style(R, run):
    return {"color": R.colors[R.model(run)], "ls": "--" if run == ENS else "-"}


# ---------------------------------------------------------------- 年内排名与评价样本

def _auc(d, col="r"):
    return roc_auc_score(d["Y"], d[col]) if d["Y"].nunique() > 1 else np.nan


def _ranked(d, col="p"):
    """年内百分位排名 r：壳概率用于每期横截面排序，合并多年评估时先在年内排序，
    避免把"哪一年借壳更多"的时间序列差异混进排序能力。"""
    return d.assign(r=d.groupby("year")[col].rank(pct=True))


def _ensemble(tests, members):
    """成员年内百分位排名的等权平均（权重事先固定，不在测试集上调整）。"""
    d = reduce(lambda a, b: a.merge(b, on=KEY),
               [_ranked(tests[m])[KEY + ["r"]].rename(columns={"r": m}) for m in members])
    d = d.merge(tests[members[0]][KEY + ["Y", "year"]], on=KEY)
    return d.assign(p=d[members].mean(axis=1), score=np.nan)[KEY + ["Y", "year", "p", "score"]]


def scopes(cfg, ranked):
    """评价样本。全部样本：所有运行都有预测的观测（主口径）；完整样本 / 含缺失样本：
    按是否有任一特征缺失划分，检验结论是否依赖缺失值的处理。"""
    df, feats = build_dataset(cfg, dropna=False)
    flag = df[KEY].assign(complete=df[feats].notna().all(axis=1).to_numpy())
    common = reduce(lambda a, b: a.merge(b, on=KEY), [te[KEY] for te in ranked.values()]).merge(flag, on=KEY)
    return {"全部样本": common[KEY], "完整样本": common.loc[common["complete"], KEY],
            "含缺失样本": common.loc[~common["complete"], KEY]}


def _wide(R, runs, keys):
    """一个评价样本内各运行的年内排名（在该样本内重新排序），每个运行一列。"""
    cols = [_ranked(R.ranked[n].merge(keys, on=KEY))[KEY + ["r"]].rename(columns={"r": n}) for n in runs]
    first = R.ranked[runs[0]].merge(keys, on=KEY)[KEY + ["Y", "year"]]
    return reduce(lambda a, b: a.merge(b, on=KEY), [first, *cols])


def _ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def _fmt_ci(x):
    lo, hi = _ci(x)
    return f"[{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------- 指标表

def pooled_tables(R, sc):
    """2012-2023 年样本外预测合并评估。全部样本口径包括全部运行 + 组合，其他口径只用 plot 集合。
    每个口径内，所有运行共用同一组按公司重抽样的权重，因此配对差值有效。
    返回 (排序与概率水平表, Top-k 表, 与对照基准的 AUC 差值表)，均带 run 列（保存前去掉）。"""
    ev, base = R.cfg["evaluate"], R.cfg["evaluate"]["baseline"]
    rank_rows, topk_rows, diff_rows = [], [], []
    for scope, keys in sc.items():
        runs = list(R.ranked) if scope == "全部样本" else R.plot
        w = _wide(R, runs, keys)
        boot = bootstrap_auc(w["Y"], {n: w[n] for n in runs}, ev["bootstrap"], groups=w["Stkcd"])
        auc = {n: roc_auc_score(w["Y"], w[n]) for n in runs}
        if scope == "全部样本":
            R.auc, R.boot = auc, boot
        for n in runs:
            d = _ranked(R.ranked[n].merge(keys, on=KEY))
            head = {"run": n, "样本": scope, "模型": R.labels[n], "N": len(d), "正例": int(d["Y"].sum())}
            ap = average_precision_score(d["Y"], d["r"])
            row = {**head, "AUC(年内排序)": auc[n], "95%CI": _fmt_ci(boot[n]),
                   "逐年AUC均值": np.mean([_auc(g) for _, g in d.groupby("year")]),
                   "PR_AUC": ap, "PR_AUC/正例率": ap / d["Y"].mean()}
            if n != ENS:
                s = summarize(d["Y"], d["p"], [])
                row |= {"AUC(原始概率)": s["AUC"], "Brier": s["Brier"], "R2": s["R2"],
                        "McFadden_R2": s["McFadden_R2"], "RMSE_Y1": s["RMSE_Y1"]}
            rank_rows.append(row)
            topk = dict(head)
            for k in ev["top_k"]:
                r = within_year_topk(d, k)
                topk[f"Lift@{k:.0%}"], topk[f"Recall@{k:.0%}"] = r["lift"], r["recall"]
            topk_rows.append(topk)
        if base in runs:
            rows = []
            for n in runs:
                if n == base:
                    continue
                diff = boot[n] - boot[base]
                lo, hi = _ci(diff)
                rows.append({"run": n, "样本": scope, "对比": f"{R.labels[n]} − {R.labels[base]}",
                             "AUC 差值": auc[n] - auc[base], "lo": lo, "hi": hi,
                             "95% 置信区间": f"[{lo:.3f}, {hi:.3f}]", "p 值": boot_pvalue(diff)})
            adj, sig = holm([r["p 值"] for r in rows], ev["alpha"])
            for r, a, s in zip(rows, adj, sig):
                r |= {"Holm 调整 p": a, "Holm 校正后显著": "是" if s else "否"}
            diff_rows += rows
    cols = ["run", "样本", "模型", "N", "正例", "AUC(年内排序)", "95%CI", "逐年AUC均值", "AUC(原始概率)",
            "PR_AUC", "PR_AUC/正例率", "Brier", "R2", "McFadden_R2", "RMSE_Y1"]
    return pd.DataFrame(rank_rows).reindex(columns=cols), pd.DataFrame(topk_rows), pd.DataFrame(diff_rows)


def paper_table(R):
    """论文表 4 的口径：IS = 实际用于训练的样本，IAS = 抽样模型在完整训练窗口上，OS = 预测年。"""
    out = []
    for name in [n for n in R.plot if n != ENS]:
        us = R.sampler(name) in RESAMPLED
        rows = []
        for year, g in R.preds[name].groupby("year"):
            tr, te = g[g["sample"] == "train_all"], g[g["sample"] == "test"]
            fit = g[g["sample"] == "train_used"] if us else tr
            s_fit, s_tr, s_te = (summarize(d["Y"], d["p"], []) for d in (fit, tr, te))
            rows.append({
                "年份": str(year), "R2_IS": s_fit["R2"], "R2_IAS": s_tr["R2"] if us else np.nan, "R2_OS": s_te["R2"],
                "RMSE_IS": s_fit["RMSE_Y1"], "RMSE_OS": s_te["RMSE_Y1"],
                "AUC_IS": s_tr["AUC"], "AUC_OS": s_te["AUC"],
                "TL2_IS": lift(tr["Y"].to_numpy(), tr["p"].to_numpy(), 0.02),
                "TL10_OS": lift(te["Y"].to_numpy(), te["p"].to_numpy(), 0.10),
            })
        t = pd.DataFrame(rows)
        mean = t.drop(columns="年份").mean().to_frame().T.assign(年份="均值")
        out.append(pd.concat([t, mean]).assign(模型=R.labels[name]))
    df = pd.concat(out)
    return df[["模型", *[c for c in df.columns if c != "模型"]]]


def sampler_table(R, pooled, topk):
    """第一步汇总：step = 1 的模型 × 全部平衡方法，差值相对同一模型的 S0。"""
    full = pooled[pooled["样本"] == "全部样本"].set_index("run")
    tk = topk[topk["样本"] == "全部样本"].set_index("run")
    rows = []
    for run in R.names:
        m = R.model(run)
        if R.cfg["models"][m]["step"] != 1:
            continue
        ref, man = f"{m}_s0", R.manifests[run]
        diff = R.boot[run] - R.boot[ref] if ref in R.boot else None
        rows.append({
            "模型": R.cfg["models"][m]["label"], "方法": R.cfg["samplers"][R.sampler(run)]["label"],
            "AUC(年内排序)": R.auc[run], "95%CI": full.at[run, "95%CI"],
            "相对S0差值": R.auc[run] - R.auc[ref] if diff is not None and run != ref else np.nan,
            "差值95%CI": _fmt_ci(diff) if diff is not None and run != ref else "",
            "Lift@5%": tk.at[run, "Lift@5%"], "Lift@10%": tk.at[run, "Lift@10%"],
            "PR_AUC/正例率": full.at[run, "PR_AUC/正例率"], "Brier": full.at[run, "Brier"],
            "McFadden_R2": full.at[run, "McFadden_R2"], "平均训练样本量": R.metas[run]["n_used"].mean(),
            "累计耗时(秒)": man["cand_seconds"] + man["refit_seconds"],
        })
    return pd.DataFrame(rows)


def grid_table(R):
    """模型 × 平衡方法矩阵，格内为年内排序 AUC，未运行的格子留空。"""
    rows = {spec["label"]: {R.cfg["samplers"][s]["label"]: R.auc.get(f"{m}_{s}", np.nan) for s in R.cfg["samplers"]}
            for m, spec in R.cfg["models"].items()}
    return pd.DataFrame(rows).T.rename_axis("模型").reset_index()


def by_year_table(R, keys):
    """逐年 AUC、12 年等权均值，以及各分期合并的年内排序 AUC。
    合并 AUC 相当于按正例数加权，主要由借壳高峰期决定；等权均值与分期结果用来检验这一点。"""
    w = _wide(R, R.plot, keys)
    rows = [{"预测年": str(y), "正例": int(g["Y"].sum()), **{R.labels[n]: _auc(g, n) for n in R.plot}}
            for y, g in w.groupby("year")]
    t = pd.DataFrame(rows)
    extra = [{"预测年": "逐年均值", "正例": int(t["正例"].sum()), **{R.labels[n]: t[R.labels[n]].mean() for n in R.plot}}]
    for a, b in R.cfg["evaluate"]["periods"]:
        g = w[w["year"].between(a, b)]
        extra.append({"预测年": f"{a}–{b} 合并", "正例": int(g["Y"].sum()), **{R.labels[n]: _auc(g, n) for n in R.plot}})
    return pd.concat([t, pd.DataFrame(extra)], ignore_index=True)


def rank_corr_table(R, keys):
    """年内排名的 Spearman 相关：相关越低，两个模型挑出的高风险公司差别越大，组合的潜在收益越大。"""
    c = _wide(R, R.plot, keys)[R.plot].corr(method="spearman")
    c.index = c.columns = [R.labels[n] for n in R.plot]
    return c.reset_index(names="模型")


def stability_table(R, keys):
    """壳排名的年际稳定性：相邻年排名相关越高、头部名单换手越低，后续组合的交易成本越低。"""
    return pd.DataFrame([{"模型": R.labels[n], **rank_stability(_ranked(R.ranked[n].merge(keys, on=KEY)))}
                         for n in R.plot])


def calib_year_table(R, keys):
    """各预测年的实际借壳比例与平均预测概率（%）：检验概率水平能否跨年比较。"""
    runs = [n for n in R.plot if n != ENS]
    t = R.tests[runs[0]].merge(keys, on=KEY).groupby("year").agg(正例=("Y", "sum"), 实际比例=("Y", "mean"))
    t["实际比例"] *= 100
    for n in runs:
        t[f"{R.labels[n]} 平均概率"] = R.tests[n].merge(keys, on=KEY).groupby("year")["p"].mean() * 100
    return t.rename(columns={"实际比例": "实际比例(%)"}).reset_index(names="预测年")


def _param_text(row, params):
    return ", ".join(f"{k}={row[k]}" for k in params)


def selection_tables(R):
    """前推调参每年选中的参数（长表）与各参数取值被选中的年数（汇总）。"""
    long, summary = [], []
    for n in R.names:
        s = R.sels[n]
        params = [c for c in s.columns if c not in ("year", "val_years", "cand_id", "val_auc", "val_auc_2nd")]
        long.append(pd.DataFrame({"运行": R.labels[n], "预测年": s["year"].astype(str), "验证年份": s["val_years"],
                                  "选中参数": [_param_text(r, params) for _, r in s.iterrows()],
                                  "验证AUC": s["val_auc"], "第二名验证AUC": s["val_auc_2nd"]}))
        for p in params:
            for v, k in s[p].astype(str).value_counts().sort_index().items():
                summary.append({"运行": R.labels[n], "参数": p, "取值": v, "选中年数": k})
    return pd.concat(long, ignore_index=True), pd.DataFrame(summary)


def training_table(R):
    """逐运行、逐年的训练样本构成；报告使用此汇总表，无需读取 outputs/。"""
    rows = []
    for n in R.names:
        m = R.metas[n]
        rows.append(pd.DataFrame({"运行": n, "模型": R.labels[n], "预测年": m["year"],
                                  "训练样本": m["n_train"], "训练正例": m["pos_train"],
                                  "处理后样本": m["n_used"], "先验校正系数": m["beta"]}))
    return pd.concat(rows, ignore_index=True)


def runtime_table(R):
    """拟合次数与耗时（各任务单进程耗时的累计，不是墙上时间）。"""
    rows = []
    for n in R.names:
        m = R.manifests[n]
        fits, sec = m["cand_fits"] + m["refit_fits"], m["cand_seconds"] + m["refit_seconds"]
        rows.append({"运行": R.labels[n], "候选数": len(m["candidates"]), "候选拟合次数": m["cand_fits"],
                     "重拟合次数": m["refit_fits"], "累计耗时(秒)": sec, "平均每次拟合(秒)": sec / fits,
                     "收敛警告": m["convergence_warnings"]})
    return pd.DataFrame(rows)


def coef_table(meta, l1=False):
    """标准化系数的 12 年汇总。缺失指示变量只在训练窗口中有缺失的年份出现，占比按出现的年份计算。"""
    cols = [c for c in meta.columns if c.startswith("coef_")]
    c = meta[cols]
    names = [col[len("coef_"):] for col in cols]
    t = pd.DataFrame({"变量": names, "含义": [label(f).split(" ", 1)[-1] for f in names],
                      "系数均值": c.mean().to_numpy(), "系数标准差": c.std().to_numpy(),
                      "为正的年份占比": (c.gt(0).sum() / c.notna().sum()).to_numpy(),
                      "出现年份数": c.notna().sum().to_numpy()})
    if l1:
        t["非零年份数"] = c.ne(0).where(c.notna()).sum().to_numpy().astype(int)
    return t.reindex(t["系数均值"].abs().sort_values(ascending=False).index)


def l1_table(meta):
    """L1 每年选中的 C 与非零系数个数；各变量被选中的年数见系数表的"非零年份数"。"""
    c = meta[[col for col in meta.columns if col.startswith("coef_")]]
    return pd.DataFrame({"预测年": meta["year"].astype(str), "C": meta["C"],
                         "非零系数个数": c.ne(0).where(c.notna()).sum(axis=1).astype(int),
                         "变量总数": c.notna().sum(axis=1)})


# ---------------------------------------------------------------- 图

def _legend_right(ax, **kw):
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), **kw)


def fig_roc_pr(R, keys):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ymax = 0
    for n in R.plot:
        d = _ranked(R.ranked[n].merge(keys, on=KEY))
        fpr, tpr, _ = roc_curve(d["Y"], d["r"])
        a1.plot(fpr, tpr, lw=1.5, label=f"{R.labels[n]}（{_auc(d):.3f}）", **_style(R, n))
        prec, rec, _ = precision_recall_curve(d["Y"], d["r"])
        a2.plot(rec, prec, lw=1.5, label=R.labels[n], **_style(R, n))
        ymax = max(ymax, prec[rec >= 0.05].max())
    base = R.ranked[R.plot[0]].merge(keys, on=KEY)["Y"].mean()
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1)
    a1.set(xlabel="假阳性率", ylabel="真阳性率（召回率）", title="ROC 曲线（括号内为年内排序 AUC）")
    a1.legend(fontsize=8, loc="lower right")
    a2.axhline(base, color=MUTED, lw=1, label=f"随机排序（正例率 {base:.2%}）")
    a2.set(xlabel="召回率", ylabel="精确率", title="PR 曲线", ylim=(0, ymax * 1.1))
    a2.legend(handles=a2.get_lines()[-1:], fontsize=8, loc="upper right")
    fig.suptitle("样本外 2012–2023 年，年内排序后合并", fontsize=10, color=INK2)
    fig.tight_layout()
    save_fig(fig, R.cfg, "fig_roc_pr")


def fig_gains(R, keys):
    """每年按壳概率从高到低筛选前 q 的公司，看能覆盖多少真实借壳公司。"""
    qs = np.linspace(0.01, 0.5, 50)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for n in R.plot:
        d = R.ranked[n].merge(keys, on=KEY)
        r = d.groupby("year")["p"].rank(ascending=False, pct=True, method="first")
        a1.plot(qs * 100, [d.loc[r <= q, "Y"].sum() / d["Y"].sum() * 100 for q in qs], lw=1.5, **_style(R, n))
        a2.plot(qs * 100, [d.loc[r <= q, "Y"].mean() / d["Y"].mean() for q in qs], lw=1.5,
                label=R.labels[n], **_style(R, n))
    a1.plot([0, 50], [0, 50], color=MUTED, lw=1)
    a2.axhline(1, color=MUTED, lw=1)
    for ax in (a1, a2):
        for k in R.cfg["evaluate"]["top_k"]:
            ax.axvline(k * 100, color=MUTED, lw=0.6, ls=":")
    a1.set(xlabel="每年筛选的公司比例 (%)", ylabel="覆盖的借壳公司比例 (%)", title="累积增益曲线")
    a2.set(xlabel="每年筛选的公司比例 (%)", ylabel="提升度 Lift", title="提升度曲线")
    _legend_right(a2)
    fig.tight_layout()
    save_fig(fig, R.cfg, "fig_gains_lift")


def _calib(d, col, bins=10):
    q = pd.qcut(d[col].rank(method="first"), bins, labels=False)
    return d.groupby(q).agg(pred=(col, "mean"), obs=("Y", "mean"))


def _loglog(ax, curves):
    """对数坐标 + 45° 线：概率跨两三个数量级，线性坐标下低分组挤在原点附近。
    实际比例为 0 的分组、平均预测概率低于 0.001% 的分组落在坐标范围之外。"""
    x, y = (np.concatenate([c[k] for c in curves]) * 100 for k in ("pred", "obs"))
    hi = max(x.max(), y.max()) * 1.5
    xlo, ylo = max(x[x > 0].min(), 1e-3) / 1.5, y[y > 0].min() / 1.5
    ax.plot([min(xlo, ylo), hi], [min(xlo, ylo), hi], color=MUTED, lw=1)
    ax.set(xscale="log", yscale="log", xlim=(xlo, hi), ylim=(ylo, hi),
           xlabel="平均预测概率 (%，对数坐标)", ylabel="实际借壳比例 (%，对数坐标)")


def fig_calibration(R, keys):
    """左：主模型先验校正前后（有抽样或加权时）；右：plot_runs 校正后概率的校准曲线（按预测值十等分）。"""
    main = R.cfg["evaluate"]["main"]
    left = main in R.tests and (R.metas[main]["beta"] < 1).any()
    fig, axes = plt.subplots(1, 2 if left else 1, figsize=(12 if left else 6.5, 4.8), squeeze=False)
    if left:
        a1 = axes[0, 0]
        d = R.tests[main].merge(keys, on=KEY)
        m, c = _calib(d, "score"), _calib(d, "p")
        a1.plot(m["pred"] * 100, m["obs"] * 100, marker="o", color=SERIES[1], label="校正前的原始分数")
        a1.plot(c["pred"] * 100, c["obs"] * 100, marker="o", color=SERIES[0], label="先验校正后的概率")
        _loglog(a1, [m, c])
        a1.set_title(f"{R.labels[main]}：先验校正前后")
        a1.legend(fontsize=8, loc="upper left")
    a2, curves = axes[0, -1], []
    for n in [n for n in R.plot if n != ENS]:
        curves.append(c := _calib(R.tests[n].merge(keys, on=KEY), "p"))
        a2.plot(c["pred"] * 100, c["obs"] * 100, marker="o", markersize=4, lw=1.4, label=R.labels[n], **_style(R, n))
    _loglog(a2, curves)
    a2.set_title("校准曲线（按预测值十等分）")
    _legend_right(a2)
    fig.tight_layout()
    save_fig(fig, R.cfg, "fig_calibration")


def fig_auc_by_year(R, by_year):
    t = by_year[by_year["预测年"].str.isdigit()]
    years = t["预测年"].astype(int)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[3, 1.2])
    for n in R.plot:
        a1.plot(years, t[R.labels[n]], marker="o", markersize=4, lw=1.5, label=R.labels[n], **_style(R, n))
    a1.axhline(0.5, color=MUTED, lw=1)
    a1.set(ylabel="样本外 AUC", title="逐年样本外 AUC（全部样本）")
    _legend_right(a1)
    a2.bar(years, t["正例"], color=MUTED, width=0.6)
    a2.set(ylabel="测试年正例数", xlabel="预测年份")
    a2.set_xticks(years)
    fig.tight_layout()
    save_fig(fig, R.cfg, "fig_auc_by_year")


def fig_sampler_forest(R):
    """第一步：各平衡方法相对同一模型全数据（S0）的 AUC 差值，点 + 95% 自助法区间，同一行并排比较模型。"""
    models = [m for m, spec in R.cfg["models"].items() if spec["step"] == 1 and f"{m}_s0" in R.boot]
    samplers = [s for s in R.cfg["samplers"] if s != "s0"]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(samplers) * len(models) + 1.2))
    off = np.linspace(-0.18, 0.18, len(models)) if len(models) > 1 else [0]
    for o, m in zip(off, models):
        for j, s in enumerate(samplers):
            if (run := f"{m}_{s}") not in R.boot:
                continue
            lo, hi = _ci(R.boot[run] - R.boot[f"{m}_s0"])
            ax.plot([lo, hi], [j + o] * 2, color=R.colors[m], lw=2, solid_capstyle="round")
            ax.plot(R.auc[run] - R.auc[f"{m}_s0"], j + o, "o", color=R.colors[m], ms=7, mec=BG, mew=1,
                    label=R.cfg["models"][m]["label"] if j == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(samplers)), [R.cfg["samplers"][s]["label"] for s in samplers])
    ax.set_ylim(len(samplers) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="年内排序 AUC − 同一模型全数据（S0）的 AUC", title="平衡方法的影响（点：差值；线：95% 自助法区间）")
    ax.legend(fontsize=9, loc="lower left")
    save_fig(fig, R.cfg, "fig_sampler_forest")


def fig_grid_heatmap(R, grid):
    t = grid.set_index("模型")
    v = t.to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("seq", ["#eef3fb", SERIES[0]])
    cmap.set_bad(BG)
    lo, hi = np.nanmin(v), np.nanmax(v)
    fig, ax = plt.subplots(figsize=(1.35 * v.shape[1] + 2.5, 0.6 * v.shape[0] + 1.6))
    im = ax.imshow(np.ma.masked_invalid(v), cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isnan(v[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=9)
            else:
                dark = (v[i, j] - lo) / (hi - lo + 1e-12) > 0.6
                ax.text(j, i, f"{v[i, j]:.3f}", ha="center", va="center", fontsize=9, color=BG if dark else INK)
    ax.set_xticks(range(v.shape[1]), t.columns)
    ax.set_yticks(range(v.shape[0]), t.index)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False, colors=INK2)
    ax.tick_params(axis="y", colors=INK2)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="年内排序 AUC")
    ax.set_title("模型 × 平衡方法：合并样本外年内排序 AUC（— 为未运行）", pad=30)
    save_fig(fig, R.cfg, "fig_grid_heatmap")


def fig_forest_all(R, diff):
    """全部运行 + 组合相对对照基准的 AUC 差值，按差值排序；实心点为 Holm 校正后显著。"""
    d = diff[diff["样本"] == "全部样本"].sort_values("AUC 差值").reset_index(drop=True)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(d) + 1.5))
    for i, r in d.iterrows():
        c = R.colors[R.model(r["run"])]
        ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=1.8, solid_capstyle="round")
        ax.plot(r["AUC 差值"], i, "o", ms=6.5, color=c, mfc=c if r["Holm 校正后显著"] == "是" else BG, mew=1.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(d)), [R.labels[n] for n in d["run"]], fontsize=8.5)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.grid(axis="y", visible=False)
    base = R.labels[R.cfg["evaluate"]["baseline"]]
    ax.set(xlabel=f"年内排序 AUC − {base}", title=f"相对 {base} 的配对 AUC 差值（95% 自助法区间）")
    marks = [Line2D([], [], marker="o", ls="", color=INK2, mfc=INK2, label="Holm 校正后显著"),
             Line2D([], [], marker="o", ls="", color=INK2, mfc=BG, mew=1.5, label="不显著")]
    ax.legend(handles=marks, fontsize=8, loc="lower right")
    save_fig(fig, R.cfg, "fig_forest_all")


# ---------------------------------------------------------------- 特征重要性与预测模式（主模型 XGBoost）

def _models(cfg, run, year):
    out = []
    for f in sorted((cfg["paths"]["outputs"] / run / "models").glob(f"{year}_*.json")):
        m = xgb.XGBRegressor()
        m.load_model(f)
        out.append(m)
    return out


def importance(R):
    """增益重要性（论文表 5 口径，归一化为 %，对各年、各子样本取平均），并列主模型同族的全部运行；
    另加主模型的平均 |SHAP|。主模型需使用原始版特征（prep = raw）。"""
    cfg, main = R.cfg, R.cfg["evaluate"]["main"]
    df, feats = build_dataset(cfg, dropna=False)
    years = range(cfg["rolling"]["first_pred_year"], cfg["rolling"]["last_pred_year"] + 1)
    table = pd.DataFrame({"变量": feats})
    for n in [n for n in R.names if R.model(n) == R.model(main)]:
        gains = []
        for year in years:
            for m in _models(cfg, n, year):
                g = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(feats).fillna(0)
                gains.append(g / g.sum())
        table[f"{R.labels[n]} 增益(%)"] = pd.concat(gains, axis=1).mean(axis=1).to_numpy() * 100

    shap, Xs, Ys = [], [], []
    for year in years:
        X = df.loc[df["time"] == year, feats]
        dm = xgb.DMatrix(X)
        shap.append(np.mean([m.get_booster().predict(dm, pred_contribs=True)[:, :-1] for m in _models(cfg, main, year)],
                            axis=0))
        Xs.append(X)
        Ys.append(df.loc[df["time"] == year, "Y"].to_numpy())
    shap, X, Y = np.vstack(shap), pd.concat(Xs), np.concatenate(Ys)
    mabs = np.abs(shap).mean(axis=0)
    table["主模型 平均绝对SHAP(%)"] = mabs / mabs.sum() * 100
    table.insert(1, "含义", [label(f).split(" ", 1)[-1] for f in feats])
    table = table.sort_values(table.columns[-1], ascending=False).reset_index(drop=True)
    table.insert(0, "排序", range(1, len(table) + 1))
    return table, (shap, X, Y, feats)


def fig_importance(R, table):
    main = R.cfg["evaluate"]["main"]
    cols = [f"{R.labels[main]} 增益(%)", "主模型 平均绝对SHAP(%)"]
    n = min(15, len(table))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, col, title in zip(axes, cols, ["增益重要性（论文表 5 口径）", "平均 |SHAP| 贡献"]):
        t = table.nlargest(n, col).iloc[::-1]
        ax.barh([label(v) for v in t["变量"]], t[col], color=SERIES[1], height=0.6)
        ax.set(xlabel="相对重要性 (%)", title=title)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"{R.labels[main]}：前 {n} 个特征", fontsize=12)
    fig.tight_layout()
    save_fig(fig, R.cfg, "fig_importance")


def fig_shap_dependence(R, shap_data):
    shap, X, Y, feats = shap_data
    obj = R.cfg["models"][R.model(R.cfg["evaluate"]["main"])].get("fixed", {}).get("objective")
    scale = "对数几率" if obj == "binary:logistic" else "概率"
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, f in zip(axes.flat, R.cfg["evaluate"]["pattern_features"]):
        j = feats.index(f)
        ax.scatter(X[f], shap[:, j], s=4, color=SERIES[0], alpha=0.25, lw=0, label="未被借壳")
        ax.scatter(X[f][Y == 1], shap[Y == 1, j], s=18, color=SERIES[1], edgecolor=BG, lw=0.8, label="被借壳")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set(xlabel=label(f), ylabel=f"SHAP 值（{scale}）")
    axes.flat[0].legend(fontsize=8, markerscale=2)
    fig.suptitle("SHAP 依赖图：特征取值如何推高或压低壳概率（样本外，2012–2023）", fontsize=12)
    fig.tight_layout()
    save_fig(fig, R.cfg, "fig_shap_dependence")


def fig_patterns(R):
    """论文图 3-6 的预测模式：其他特征保持实际值，只改变目标特征，看平均壳概率怎么变（部分依赖）。"""
    cfg, main = R.cfg, R.cfg["evaluate"]["main"]
    df, feats = build_dataset(cfg, dropna=False)
    years = range(cfg["rolling"]["first_pred_year"], cfg["rolling"]["last_pred_year"] + 1)
    beta = R.metas[main].set_index("year")["beta"]
    test = df[df["time"].between(min(years), max(years))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, f in zip(axes.flat, cfg["evaluate"]["pattern_features"]):
        grid = np.unique(np.nanpercentile(test[f], np.linspace(1, 99, 25)))
        curves = []
        for year in years:
            X = df.loc[df["time"] == year, feats]
            big = pd.concat([X.assign(**{f: v}) for v in grid])  # 一次性预测全部网格点
            s = np.clip(np.mean([m.predict(big) for m in _models(cfg, main, year)], axis=0), 0, 1)
            p = prior_correct(s, beta[year]).reshape(len(grid), len(X)).mean(axis=1)
            curves.append(p * 100)
            ax.plot(grid, p * 100, color=MUTED, lw=0.8, alpha=0.6)
        ax.plot(grid, np.mean(curves, axis=0), color=SERIES[1], lw=2.5, label="各年平均")
        ax.set(xlabel=label(f), ylabel="平均壳概率 (%)")
    axes.flat[0].plot([], [], color=MUTED, lw=0.8, label="单个年份模型")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{R.labels[main]}：重要特征的预测模式（部分依赖）", fontsize=12)
    fig.tight_layout()
    save_fig(fig, cfg, "fig_patterns")


# ---------------------------------------------------------------- 壳概率导出

def export(R):
    """全部运行的样本外概率面板：每个公司-预测年一行，p_<运行名> 各一列，等权组合只有排名 rank_ensemble。
    t 年的预测基于 t-1 年年报，年报 4 月底才披露完，用于月度收益检验时应对应 t 年 5 月至 t+1 年 4 月。"""
    te = R.tests[R.names[0]]
    out = te[KEY + ["Y"]].assign(apply_from=te["time"].astype(str) + "-05", apply_to=(te["time"] + 1).astype(str) + "-04")
    for n in R.names:
        out = out.merge(R.tests[n][KEY + ["p"]].rename(columns={"p": f"p_{n}"}), on=KEY, how="left")
    if ENS in R.ranked:
        out = out.merge(_ranked(R.ranked[ENS])[KEY + ["r"]].rename(columns={"r": "rank_ensemble"}), on=KEY, how="left")
    out = out.rename(columns={"time": "year"}).assign(Stkcd=lambda d: d["Stkcd"].str.zfill(6))
    out.sort_values(["Stkcd", "year"]).to_csv(R.cfg["paths"]["outputs"] / "shell_prob_panel.csv", index=False)


# ---------------------------------------------------------------- 主流程

def run(cfg):
    R = load(cfg)
    if not R.names:
        raise SystemExit("还没有任何训练结果，请先运行 train")
    ev = cfg["evaluate"]
    sc = scopes(cfg, R.ranked)
    full = sc["全部样本"]

    pooled, topk, diff = pooled_tables(R, sc)
    save_table(pooled.drop(columns="run"), cfg, "model_pooled", floatfmt={"PR_AUC": ".4f", "Brier": ".5f", "R2": ".4f"})
    save_table(topk.drop(columns="run"), cfg, "model_topk")
    if len(diff):
        save_table(diff.drop(columns=["run", "lo", "hi"]), cfg, "model_auc_diff")
        fig_forest_all(R, diff)
    save_table(sampler_table(R, pooled, topk), cfg, "model_sampler_compare",
               floatfmt={"Brier": ".5f", "平均训练样本量": ".0f", "累计耗时(秒)": ".1f"})
    fig_sampler_forest(R)
    grid = grid_table(R)
    save_table(grid, cfg, "model_grid")
    fig_grid_heatmap(R, grid)

    if [n for n in R.plot if n != ENS]:
        save_table(paper_table(R), cfg, "model_paper_table")
        save_table(calib_year_table(R, full), cfg, "model_calib_year")
        fig_calibration(R, full)
    by_year = by_year_table(R, full)
    save_table(by_year, cfg, "model_by_year")
    save_table(rank_corr_table(R, full), cfg, "model_rank_corr")
    save_table(stability_table(R, full), cfg, "model_rank_stability")
    fig_roc_pr(R, full)
    fig_gains(R, full)
    fig_auc_by_year(R, by_year)

    long, summary = selection_tables(R)
    save_table(long, cfg, "model_selection", floatfmt=".4f")
    save_table(summary, cfg, "model_selection_summary")
    save_table(training_table(R), cfg, "model_training", floatfmt={"先验校正系数": ".6f"})
    save_table(runtime_table(R), cfg, "model_runtime", floatfmt={"累计耗时(秒)": ".1f", "平均每次拟合(秒)": ".3f"})
    for n in [n for n in R.plot if n != ENS and cfg["models"][R.model(n)]["kind"] in LINEAR]:
        l1 = cfg["models"][R.model(n)]["kind"] == "logit_l1"
        save_table(coef_table(R.metas[n], l1), cfg, f"model_{n}_coef")
        if l1:
            save_table(l1_table(R.metas[n]), cfg, "model_l1_selection", floatfmt="g")

    if ev["main"] in R.tests:
        imp, shap_data = importance(R)
        save_table(imp, cfg, "model_importance", floatfmt=".2f")
        fig_importance(R, imp)
        fig_shap_dependence(R, shap_data)
        fig_patterns(R)
        export(R)
    else:
        print(f"主模型 {ev['main']} 还没有训练结果：跳过特征重要性、预测模式图和壳概率导出")
    print(f"评估完成：表格 → {cfg['paths']['tables']}，图 → {cfg['paths']['figures']}，"
          f"壳概率 → {cfg['paths']['outputs'] / 'shell_prob_panel.csv'}")
