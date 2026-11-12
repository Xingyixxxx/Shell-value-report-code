"""补充检验（看过主检验结果之后加的，属探索性）：

1. 分年度、分政策阶段的多空收益与累计曲线（原始十分组 / 规模中性 / 最小 30% 内部）。
2. 月度独立 5×5（壳概率 × 规模）双重分组：规模用“当月市值”还是“上月市值”，
   用来核对组员第 5 章的规模中性化结果（其规模取当月末总市值）。

    uv run -m factor_test extra
"""
import numpy as np
import pandas as pd

from shellpred.plot import INK2, MUTED, SERIES, plt, save_fig, save_table

from . import stats as S
from .analysis import PCT, Ctx
from .panel import build
from .portfolios import cut_group, long_short, port, qgroup

# 三个政策阶段：2016-09 重组新规（严控借壳）；2019-10 重组办法修订（放松）并推进注册制
STAGES = [("宽松期", "2012-05", "2016-08"), ("严控期", "2016-09", "2019-09"), ("放松+注册制", "2019-10", "2023-12")]
SIGNALS = ["xgb_s1", "logit_l1_s0", "ensemble"]


def _form_year(idx):
    return idx.map(lambda m: m.year - (m.month < 5))


def strategies(c, sig):
    """三种构建方式的月度多空序列（等权 / 流通市值加权）。"""
    ch, mp = c.ch, c.mp
    b3, b5 = cut_group(ch, ch["size"]), qgroup(ch, ch["size"], 5)
    inner3, inner5 = qgroup(ch, ch[sig], 5, by=b3), qgroup(ch, ch[sig], 5, by=b5)
    out = {}
    for w in ("ew", "vw"):
        out[("十分组 D10−D1", w)] = long_short(port(mp, qgroup(ch, ch[sig], 10), w))
        # 规模中性：形成时按上月流通市值分 5 组，组内再按壳概率分 5 组，对 5 个规模组等权平均
        ls = [long_short(port(mp, np.where(b5 == k, inner5, np.nan), w)[[1.0, 5.0]]) for k in range(1, 6)]
        out[("规模中性 Q5−Q1", w)] = pd.concat(ls, axis=1).mean(axis=1)
        out[("最小 30% 内 Q5−Q1", w)] = long_short(port(mp, np.where(b3 == 1, inner3, np.nan), w)[[1.0, 5.0]])
    return out


def annual(c):
    rows_y, rows_s, series = [], [], {}
    for sig in SIGNALS:
        for (name, w), ls in strategies(c, sig).items():
            ls = ls.loc[:c.months[-1]]
            series[(sig, name, w)] = ls
            for y, s in ls.groupby(_form_year(ls.index)):
                rows_y.append({"signal": sig, "strategy": name, "weight": w, "year": y, "months": len(s),
                               "mean": s.mean() * PCT, "t": s.mean() / s.std() * np.sqrt(len(s)),
                               "cum": ((1 + s).prod() - 1) * PCT, "win": (s > 0).mean()})
            for lab, a, b in [("全样本", c.months[0], c.months[-1])] + STAGES:
                s = ls.loc[a:b]
                rows_s.append({"signal": sig, "strategy": name, "weight": w, "stage": lab, "T": len(s),
                               **c.row(s, ["FF3", "FF6", "CH3"]), "cum": ((1 + s).prod() - 1) * PCT})
    Y, St = pd.DataFrame(rows_y), pd.DataFrame(rows_s)
    save_table(Y, c.cfg, "x_annual")
    save_table(St, c.cfg, "x_stages")

    # 图 1：累计多空（等权），三种构建方式 × 三个信号，政策阶段加底色
    names = ["十分组 D10−D1", "规模中性 Q5−Q1", "最小 30% 内 Q5−Q1"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, name in zip(axes, names):
        for sig, col in zip(SIGNALS, SERIES):
            v = (1 + series[(sig, name, "ew")]).cumprod()
            ax.plot(v.index.to_timestamp(), v, color=col, label=c.lab[sig], lw=1.8)
        ax.axvspan(pd.Timestamp("2016-09-01"), pd.Timestamp("2019-10-01"), color=MUTED, alpha=.13, lw=0)
        ax.axhline(1, color=INK2, lw=.8)
        ax.set_title(f"{name}（等权）")
    axes[0].set_ylabel("累计净值（多空，期初 = 1）")
    axes[0].legend(loc="upper left")
    axes[1].text(pd.Timestamp("2018-03-15"), axes[1].get_ylim()[1] * .97, "严控期\n2016-09 至 2019-09", ha="center", va="top", color=INK2, fontsize=9)
    save_fig(fig, c.cfg, "fig_x_cum_stages")

    # 图 2：主信号分年度累计多空收益
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, name in zip(axes, names):
        d = Y[(Y.signal == c.main) & (Y.strategy == name)].pivot(index="year", columns="weight", values="cum")
        x = np.arange(len(d))
        ax.bar(x - .2, d["ew"], .38, color=SERIES[0], label="等权")
        ax.bar(x + .2, d["vw"], .38, color=SERIES[1], label="流通市值加权")
        ax.axvspan(4.5, 7.5, color=MUTED, alpha=.13, lw=0)
        ax.axhline(0, color=INK2, lw=.8)
        ax.set_xticks(x, d.index, rotation=45)
        ax.set_title(f"{c.lab[c.main]}：{name}")
    axes[0].set_ylabel("形成年累计多空收益（%）")
    axes[0].legend(loc="upper right")
    save_fig(fig, c.cfg, "fig_x_annual")
    return Y, St


def lookahead(c):
    """月度独立 5×5：SP × 规模。比较规模口径（当月末 / 上月末总市值）与信号对齐（日历年 / 5 月）。"""
    data = c.cfg["paths"]["data"]
    now = pd.read_csv(data / "stock_month.csv", dtype={"Stkcd": str}, usecols=["Stkcd", "Trdmnt", "Msmvttl"])
    now["month"] = pd.PeriodIndex(now["Trdmnt"], freq="M")
    L = c.D.long[~c.D.long["susp"].astype(bool)].merge(now[["Stkcd", "month", "Msmvttl"]], on=["Stkcd", "month"], how="left")
    L = L[(L["month"] >= pd.Period("2012-01", "M")) & (L["month"] <= c.months[-1])].copy()
    L["y_cal"] = L["month"].dt.year
    sigs = c.ch[["Stkcd", "year"] + SIGNALS]
    paper = data / "paper_shell_prob.csv"
    if paper.exists():
        pp = pd.read_csv(paper, dtype={"Stkcd": str})
        sigs = sigs.merge(pp.assign(Stkcd=pp["Stkcd"].str.zfill(6)), on=["Stkcd", "year"], how="outer")
    cols = [x for x in sigs.columns if x not in ("Stkcd", "year")]
    q5 = lambda s: np.ceil(s.rank(pct=True) * 5).clip(1, 5)

    def neutral_ls(d, sig, size):
        d = d.dropna(subset=[sig, size, "ret"])
        g = d.groupby("month")
        d = d.assign(sp=g[sig].transform(q5), sz=g[size].transform(q5))
        cell = d.groupby(["month", "sp", "sz"])["ret"].mean().unstack("sz")
        hi, lo = cell.xs(5.0, level="sp"), cell.xs(1.0, level="sp")
        within = (hi - lo)                       # 每个规模组内的 SP5 − SP1
        unc = d[d.sp == 5].groupby("month")["ret"].mean() - d[d.sp == 1].groupby("month")["ret"].mean()
        szret = d.groupby(["month", "sz"])["ret"].mean().unstack("sz")
        return within.mean(axis=1), within, unc, szret

    rows, cells, keep = [], [], {}
    for align, ycol in {"日历年（1–4 月有前视）": "y_cal", "5 月对齐": "y"}.items():
        d0 = L.merge(sigs.rename(columns={"year": ycol}), on=["Stkcd", ycol], how="inner")
        for sig in cols:
            for sname, size in {"当月末总市值（组员口径）": "Msmvttl", "上月末总市值": "mvt"}.items():
                ls, within, unc, szret = neutral_ls(d0, sig, size)
                m, t, _ = S.mean_t(ls, c.lag)
                mu, tu, _ = S.mean_t(unc, c.lag)
                a = S.alpha(ls.loc[c.months[0]:], c.fm("FF6"), c.lag)
                rows.append({"signal": sig, "align": align, "size": sname, "T": len(ls), "neutral": m * PCT, "t": t,
                             "a_FF6": a["alpha"] * PCT, "t_FF6": a["t"], "uncond": mu * PCT, "t_uncond": tu,
                             "ret_small": szret[1.0].mean() * PCT, "ret_big": szret[5.0].mean() * PCT})
                keep[(sig, align, sname)] = ls
                for k in within.columns:
                    mk, tk, _ = S.mean_t(within[k].dropna(), c.lag)
                    cells.append({"signal": sig, "align": align, "size": sname, "size_group": int(k), "ls": mk * PCT, "t": tk})
    T = pd.DataFrame(rows)
    save_table(T, c.cfg, "x_lookahead")
    save_table(pd.DataFrame(cells), c.cfg, "x_lookahead_cells")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    d = T[T["align"] == "5 月对齐"].pivot(index="signal", columns="size", values="neutral").reindex(cols)
    tt = T[T["align"] == "5 月对齐"].pivot(index="signal", columns="size", values="t").reindex(cols)
    x = np.arange(len(d))
    for i, (col, color) in enumerate(zip(["当月末总市值（组员口径）", "上月末总市值"], [SERIES[7], SERIES[0]])):
        b = axes[0].bar(x + (i - .5) * .38, d[col], .36, color=color, label=col)
        for r, v, t in zip(b, d[col], tt[col]):
            axes[0].text(r.get_x() + r.get_width() / 2, v + (.03 if v >= 0 else -.03), f"{v:.2f}\n(t={t:.1f})", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    axes[0].axhline(0, color=INK2, lw=.8)
    axes[0].set_xticks(x, [c.lab.get(s, "原论文壳概率").replace(" · ", "\n") for s in d.index])
    axes[0].set_ylim(d.min().min() - .45, d.max().max() + 1.1)
    axes[0].set_ylabel("规模中性多空（%/月）")
    axes[0].set_title("同一信号、同一方法，只换规模的取值时点")
    axes[0].legend(loc="upper right")
    for (col, color) in zip(["当月末总市值（组员口径）", "上月末总市值"], [SERIES[7], SERIES[0]]):
        v = (1 + keep[(c.main, "5 月对齐", col)]).cumprod()
        axes[1].plot(v.index.to_timestamp(), v, color=color, label=col)
    axes[1].axhline(1, color=INK2, lw=.8)
    axes[1].set_title(f"{c.lab[c.main]}：规模中性多空累计净值")
    axes[1].legend(loc="upper left")
    save_fig(fig, c.cfg, "fig_x_lookahead")
    return T


def run(cfg):
    c = Ctx(cfg, build(cfg))
    annual(c)
    lookahead(c)
    print("补充检验完成：x_annual、x_stages、x_lookahead、x_lookahead_cells，图 fig_x_*", flush=True)
