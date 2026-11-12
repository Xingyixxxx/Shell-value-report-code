"""数据处理与描述性统计：输出报告第二、三部分用到的全部表格和图。"""
import numpy as np
import pandas as pd
from scipy import stats

from .data import INDUSTRY_NAMES, VAR_LABELS, build_dataset, load_raw
from .plot import DIVERGING, INK2, MUTED, SERIES, plt, save_fig, save_table

# 论文表 1：A 股 IPO 暂停区间（2003 年以后）
IPO_PAUSES = [("2004-08", "2005-01"), ("2005-05", "2006-06"), ("2008-09", "2009-07"),
              ("2012-10", "2014-01"), ("2015-07", "2015-11")]


def _funnel(cfg, raw, xgb, full, first_pred):
    def row(name, d):
        return {"步骤": name, "观测数": len(d), "公司数": d["Stkcd"].nunique(),
                "正例数": int(d["Y"].sum()), "正例率(%)": d["Y"].mean() * 100,
                "年份": f"{d['time'].min()}–{d['time'].max()}"}
    ev = lambda d: d[d["time"] >= first_pred]  # noqa: E731
    if cfg["data"]["use_industry"]:
        steps = [("Stata 输出 original.csv", raw), ("删除行业缺失 → 全部样本（保留特征缺失，所有模型使用）", xgb)]
    else:
        steps = [("Stata 输出 original.csv = 全部样本（保留特征缺失，所有模型使用）", raw)]
    steps += [("删除任一特征缺失 → 完整样本（对照口径）", full),
              ("全部样本中的预测年（测试集）", ev(xgb)),
              ("完整样本中的预测年（测试集）", ev(full))]
    return pd.DataFrame([row(f"{'①②③④⑤⑥'[i]} {name}", d) for i, (name, d) in enumerate(steps)])


def _annual(xgb, full):
    a = xgb.groupby("time")["Y"].agg(样本数="size", 正例数="sum", 正例率="mean")
    b = full.groupby("time")["Y"].agg(完整样本数="size", 完整样本正例数="sum")
    out = a.join(b).fillna(0).reset_index().rename(columns={"time": "年份"})
    out["正例率"] *= 100
    return out.rename(columns={"正例率": "正例率(%)"}).astype({"完整样本数": int, "完整样本正例数": int})


def _windows(cfg, xgb, full):
    """每个预测年的训练窗口与样本量，以及随机欠采样（S2）、按年匹配欠采样（S3）后的训练样本量。"""
    w, r, k = cfg["rolling"]["window"], cfg["rolling"], cfg["sampling"]["ratio"]
    rows = []
    for t in range(r["first_pred_year"], r["last_pred_year"] + 1):
        row = {"预测年": t, "训练窗口": f"{t - w}–{t - 1}"}
        for tag, d in (("全部样本", xgb), ("完整样本", full)):
            tr, te = d[(d["time"] >= t - w) & (d["time"] < t)], d[d["time"] == t]
            row |= {f"{tag}训练N": len(tr), f"{tag}训练正例": int(tr["Y"].sum()),
                    f"{tag}测试N": len(te), f"{tag}测试正例": int(te["Y"].sum())}
        tr = xgb[(xgb["time"] >= t - w) & (xgb["time"] < t)]
        by_year = tr.groupby("time")["Y"].agg(["sum", "size"])
        n1 = int(by_year["sum"].sum())
        row[f"随机欠采样1:{k}后训练N"] = n1 + min(len(tr) - n1, k * n1)
        row[f"按年匹配1:{k}后训练N"] = int(sum(p + min(n - p, k * p) for p, n in by_year.to_numpy() if p))
        rows.append(row)
    return pd.DataFrame(rows)


def _missing(raw, xgb, feats):
    pos = xgb[xgb["Y"] == 1]
    return pd.DataFrame({
        "变量": feats, "含义": [VAR_LABELS[f] for f in feats],
        # 构造出来的特征（NegEq_1）在 original.csv 中没有对应列，用处理后的取值
        "原始缺失数": [(raw[f] if f in raw else xgb[f]).isna().sum() for f in feats],
        "原始缺失率(%)": [(raw[f] if f in raw else xgb[f]).isna().mean() * 100 for f in feats],
        "处理后缺失数": [xgb[f].isna().sum() for f in feats],
        "处理后缺失率(%)": [xgb[f].isna().mean() * 100 for f in feats],
        "正例中缺失数": [pos[f].isna().sum() for f in feats],
    })


def _desc(full, cols):
    rows = []
    for c in cols:
        s = full[c]
        rows.append({"变量": c, "含义": VAR_LABELS[c], "N": s.size, "均值": s.mean(), "标准差": s.std(),
                     "最小值": s.min(), "P25": s.quantile(.25), "中位数": s.median(),
                     "P75": s.quantile(.75), "最大值": s.max(), "偏度": s.skew()})
    return pd.DataFrame(rows)


def _stars(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def _group(xgb, feats):
    rows = []
    for f in feats:
        a, b = xgb.loc[xgb["Y"] == 1, f].dropna(), xgb.loc[xgb["Y"] == 0, f].dropna()
        t_p = stats.ttest_ind(a, b, equal_var=False).pvalue
        u_p = stats.mannwhitneyu(a, b).pvalue
        rows.append({"变量": f, "含义": VAR_LABELS[f], "Y=1 均值": a.mean(), "Y=0 均值": b.mean(),
                     "均值差": a.mean() - b.mean(), "t 检验 p": t_p, "": _stars(t_p),
                     "Y=1 中位数": a.median(), "Y=0 中位数": b.median(), "秩和检验 p": u_p, " ": _stars(u_p)})
    return pd.DataFrame(rows)


def _industry(xgb):
    g = xgb.groupby("industry").agg(观测数=("Y", "size"), 公司数=("Stkcd", "nunique"), 正例数=("Y", "sum"))
    g["正例率(%)"] = g["正例数"] / g["观测数"] * 100
    g = g.reset_index().rename(columns={"industry": "门类"})
    g.insert(1, "行业名称", g["门类"].map(INDUSTRY_NAMES))
    return g


def _fig_annual(t, cfg, first_pred):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, height_ratios=[3, 2])
    ax1.bar(t["年份"], t["样本数"], color=SERIES[0], width=0.7)
    ax1.set_ylabel("观测数")
    ax1.set_title("各年样本量与被借壳比例")
    ax2.plot(t["年份"], t["正例率(%)"], color=SERIES[0], marker="o", markersize=5)
    ax2.set_ylabel("被借壳比例 (%)")
    peak = t.loc[t["正例率(%)"].idxmax()]
    ax2.annotate(f"{peak['正例率(%)']:.2f}%（{int(peak['年份'])}）", (peak["年份"], peak["正例率(%)"]),
                 xytext=(8, -4), textcoords="offset points", color=INK2, fontsize=9)
    for ax in (ax1, ax2):
        ax.axvspan(first_pred - 0.5, t["年份"].max() + 0.5, color=MUTED, alpha=0.08, lw=0)
    ax1.text(first_pred - 0.3, ax1.get_ylim()[1] * 0.95, "滚动预测期", color=INK2, fontsize=9, va="top")
    ax2.set_xticks(t["年份"][::2])
    save_fig(fig, cfg, "fig_sample_by_year")


def _fig_ipo(cfg):
    ipo = pd.read_excel(cfg["paths"]["ipo"])
    # 补齐没有上会的月份，使通过率曲线在 IPO 暂停期断开而不是直线连接
    ipo = ipo[ipo["month"] <= "2023-12-31"].set_index("month").asfreq("MS").reset_index()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    ax1.bar(ipo["month"], ipo["count"].fillna(0), width=25, color=SERIES[0])
    ax1.set_ylabel("月度上会家数")
    ax1.set_title("IPO 审核情况（灰色区间为 IPO 暂停期）")
    ax2.plot(ipo["month"], ipo["rate"].rolling(3, min_periods=1).mean() * 100, color=SERIES[0], lw=1.5)
    ax2.set_ylabel("通过率 (%，3 个月均值)")
    for ax in (ax1, ax2):
        for s, e in IPO_PAUSES:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e) + pd.offsets.MonthEnd(), color=MUTED, alpha=0.15, lw=0)
    save_fig(fig, cfg, "fig_ipo")


def _fig_corr(full, feats, cfg):
    c = full[feats].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 7))
    from matplotlib.colors import LinearSegmentedColormap
    im = ax.imshow(c, cmap=LinearSegmentedColormap.from_list("div", DIVERGING), vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=60, ha="right")
    ax.set_yticks(range(len(feats)), feats)
    ax.grid(False)
    for i in range(len(feats)):
        for j in range(len(feats)):
            if i != j and abs(c.iat[i, j]) >= 0.3:  # 只标注较强的相关
                ax.text(j, i, f"{c.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman 相关系数")
    ax.set_title("特征间 Spearman 相关系数（仅标注 |ρ| ≥ 0.3）")
    save_fig(fig, cfg, "fig_corr")
    return c


def _fig_box(xgb, cfg):
    cont = ["Size_3", "Op_3", "Sales_3", "reverrate_3", "Insider_3", "Lev_3",
            "Cash_3", "Roe_3", "TOP1_3", "return_3", "IPO_1", "IPOnum_1"]
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for ax, f in zip(axes.flat, cont):
        data = [xgb.loc[xgb["Y"] == y, f].dropna() for y in (0, 1)]
        bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                        medianprops={"color": "#0b0b0b", "lw": 1.2})
        for patch, col in zip(bp["boxes"], SERIES[:2]):
            patch.set(facecolor=col, alpha=0.6, edgecolor=col)
        ax.set_xticks([1, 2], ["未被借壳", "被借壳"])
        ax.set_title(f"{f}\n{VAR_LABELS[f]}", fontsize=9, fontweight="normal")
    fig.suptitle("被借壳与未被借壳公司的特征分布（箱线图，不显示离群点）", fontsize=12)
    fig.tight_layout()
    save_fig(fig, cfg, "fig_feature_box")


def run(cfg):
    feats = cfg["data"]["features"]
    first_pred = cfg["rolling"]["first_pred_year"]
    raw = load_raw(cfg)
    xgb, _ = build_dataset(cfg, dropna=False)
    full, _ = build_dataset(cfg, dropna=True)

    save_table(_funnel(cfg, raw, xgb, full, first_pred), cfg, "data_funnel", floatfmt=".2f")
    annual = _annual(xgb, full)
    save_table(annual, cfg, "data_annual", floatfmt=".2f")
    save_table(_windows(cfg, xgb, full), cfg, "data_windows")
    save_table(_missing(raw, xgb, feats), cfg, "data_missing", floatfmt=".2f")
    save_table(_desc(full, ["Y", *feats]), cfg, "data_desc")
    save_table(_group(xgb, feats), cfg, "data_group_compare")
    save_table(_industry(xgb), cfg, "data_industry", floatfmt=".2f")

    corr = _fig_corr(full, feats, cfg)
    pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().rename("ρ").reset_index()
    pairs = pairs.reindex(pairs["ρ"].abs().sort_values(ascending=False).index).head(10)
    save_table(pairs.rename(columns={"level_0": "变量 1", "level_1": "变量 2"}), cfg, "data_corr_top")

    _fig_annual(annual, cfg, first_pred)
    _fig_ipo(cfg)
    _fig_box(xgb, cfg)
    print(f"描述性统计完成：表格 → {cfg['paths']['tables']}，图 → {cfg['paths']['figures']}")
