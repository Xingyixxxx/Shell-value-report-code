"""报告第二版（report/报告_v2.md）第 5 章新增的检验。全部沿用主检验的口径：
t 年信号用于 t 年 5 月至 t+1 年 4 月；分组用形成时（4 月末）的流通市值，市值加权用上月末流通市值；
Newey-West 滞后 4 期。属于看过主结果之后补充的检验，正文按探索性结果报告。

1. Rank IC：每月壳概率排名与当月收益的 Spearman 相关；全市场、规模组内（5 组平均）、最小 30% 内部。
2. 三种构建方式 × 三个政策阶段：多空均值、FF6 alpha、p 值；分阶段累计净值与 95% 置信区间图。
3. 持有期内各月（形成后第 k 个月，5 月 = 1）：跨形成年的均值与 t 值，按季度合并。
4. 小市值内部（最小 30%）分三个阶段的 alpha；以及“非线性模型相对逻辑回归 L1 多出来的部分”能否预测借壳、能否带来收益。

    uv run -m factor_test revise
"""
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.metrics import roc_auc_score

from shellpred.plot import BG, INK2, MUTED, SERIES, plt, save_fig, save_table

from . import stats as S
from .analysis import PCT, Ctx
from .extra import STAGES
from .panel import build
from .portfolios import cut_group, long_short, neutral_rank, port, qgroup

ALL = [("全样本", "2012-05", "2023-12"), *STAGES]
MAIN3 = ["logit_l1_s0", "xgb_s1", "ensemble"]
SMALL = ["logit_s0", "logit_l1_s0", "lda_s0", "rf_s1", "xgb_s1", "mlp_s1", "ensemble"]
NAMES = ["十分组 D10−D1", "规模中性 Q5−Q1", "最小 30% 内 Q5−Q1"]


def stars(p):
    return "" if pd.isna(p) else "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def mean_p(c, y):
    m, t, p = S.mean_t(y.dropna(), c.lag)
    return m, t, p


def alpha_p(c, y, model="FF6"):
    a = S.alpha(y.dropna(), c.fm(model), c.lag)
    return a["alpha"], a["t"], a["p"]


def strategies(c, sig, size_by=None):
    """三种构建方式的月度多空序列（与 extra.strategies 相同；size_by 可替换形成时的规模变量）。"""
    ch, mp = c.ch, c.mp
    size = ch["size"] if size_by is None else size_by
    b3, b5 = cut_group(ch, size), qgroup(ch, size, 5)
    s = ch[sig] if isinstance(sig, str) else pd.Series(sig, index=ch.index)
    inner3, inner5 = qgroup(ch, s, 5, by=b3), qgroup(ch, s, 5, by=b5)
    out = {}
    for w in ("ew", "vw"):
        out[(NAMES[0], w)] = long_short(port(mp, qgroup(ch, s, 10), w))
        ls = [long_short(port(mp, np.where(b5 == k, inner5, np.nan), w)[[1.0, 5.0]]) for k in range(1, 6)]
        out[(NAMES[1], w)] = pd.concat(ls, axis=1).mean(axis=1)
        out[(NAMES[2], w)] = long_short(port(mp, np.where(b3 == 1, inner3, np.nan), w)[[1.0, 5.0]])
    return {k: v.loc[c.months[0]:c.months[-1]] for k, v in out.items()}


# ---------------------------------------------------------------- Rank IC

def ic_series(c, sig):
    """三种口径的月度 Rank IC 序列。规模组按形成时流通市值分 5 组，组内各算一次 Spearman 再平均。"""
    ch, mp = c.ch, c.mp[~c.mp["susp"].astype(bool)]
    s = ch[sig].to_numpy() if isinstance(sig, str) else np.asarray(sig, float)
    b5, b3 = qgroup(ch, ch["size"], 5), cut_group(ch, ch["size"])
    d = pd.DataFrame({"m": mp["month"].to_numpy(), "s": s[mp["cid"].to_numpy()], "r": mp["ret"].to_numpy(),
                      "b5": b5[mp["cid"].to_numpy()], "b3": b3[mp["cid"].to_numpy()]}).dropna()
    sp = lambda g: g["s"].rank().corr(g["r"].rank()) if len(g) > 10 else np.nan
    raw = d.groupby("m").apply(sp, include_groups=False)
    neu = d.groupby(["m", "b5"]).apply(sp, include_groups=False).groupby(level=0).mean()
    small = d[d["b3"] == 1].groupby("m").apply(sp, include_groups=False)
    return {"全市场": raw, "规模组内": neu, "最小 30% 内": small}


def ic_table(c, sigs):
    rows, keep = [], {}
    for sig in sigs:
        ser = ic_series(c, sig)
        keep[sig] = ser
        for kind, i in ser.items():
            for lab, a, b in ALL:
                x = i.loc[a:b].dropna()
                m, t, p = mean_p(c, x)
                rows.append({"signal": sig, "label": c.lab[sig], "口径": kind, "阶段": lab, "T": len(x), "IC": m, "t": t, "p": p,
                             "ICIR": m / x.std() * np.sqrt(12), "IC>0": (x > 0).mean()})
    return pd.DataFrame(rows), keep


# ---------------------------------------------------------------- 单变量与分阶段

def stage_tables(c, sigs):
    rows, series = [], {}
    for sig in sigs:
        for (name, w), ls in strategies(c, sig).items():
            series[(sig, name, w)] = ls
            for lab, a, b in ALL:
                s = ls.loc[a:b]
                m, tm, pm = mean_p(c, s)
                r = {"signal": sig, "label": c.lab[sig], "构建": name, "权重": w, "阶段": lab, "T": len(s),
                     "mean": m * PCT, "t_mean": tm, "p_mean": pm, "cum": ((1 + s).prod() - 1) * PCT}
                for k in ("FF3", "FF6", "CH3"):
                    al, ta, pa = alpha_p(c, s, k)
                    r |= {f"a_{k}": al * PCT, f"t_{k}": ta, f"p_{k}": pa}
                rows.append(r)
    return pd.DataFrame(rows), series


def fig_stage(c, T, series, sig):
    """上：三个阶段各自从 1 起算的累计多空净值（等权）；下：各阶段月均多空收益与 95% 置信区间（NW）。"""
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=.42, wspace=.6)
    top = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(3)]
    for ax, (lab, a, b) in zip(top, STAGES):
        for name, col in zip(NAMES, SERIES):
            s = series[(sig, name, "ew")].loc[a:b]
            v = (1 + s).cumprod()
            xs = [(s.index[0] - 1).to_timestamp(how="end"), *[p.to_timestamp(how="end") for p in v.index]]
            ys = [1.0, *v.to_numpy()]
            ax.plot(xs, ys, color=col, lw=2, label=name)
            ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.axhline(1, color=INK2, lw=.8)
        ax.set_title(f"{lab}（{a} 至 {b}）")
        ax.tick_params(axis="x", rotation=30)
    top[0].set_ylabel("阶段内累计净值（多空，阶段初 = 1）")
    top[0].legend(loc="upper left", fontsize=8)

    b0 = fig.add_subplot(gs[1, 0:3])
    bot = [b0, fig.add_subplot(gs[1, 3:6], sharey=b0)]
    for ax, w, ttl in zip(bot, ("ew", "vw"), ("等权", "流通市值加权")):
        d = T[(T.signal == sig) & (T["权重"] == w) & (T["阶段"] != "全样本")]
        x = np.arange(len(STAGES))
        for j, (name, col) in enumerate(zip(NAMES, SERIES)):
            r = d[d["构建"] == name].set_index("阶段").reindex([s[0] for s in STAGES])
            se = r["mean"] / r["t_mean"]
            dfree = r["T"] - 1
            half = st.t.ppf(.975, dfree) * se.abs()
            xx = x + (j - 1) * .27
            ax.errorbar(xx, r["mean"], yerr=half, fmt="o", ms=8, color=col, ecolor=col, elinewidth=2, capsize=4, label=name,
                        mec=BG, mew=1.5)
            for xi, m, p in zip(xx, r["mean"], r["p_mean"]):
                ax.annotate(f"{m:.2f}{stars(p)}", (xi, m), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8, color=INK2,
                            bbox=dict(boxstyle="square,pad=0.1", fc=BG, ec="none"))
        ax.axhline(0, color=INK2, lw=.8)
        ax.set_xticks(x, [s[0] for s in STAGES])
        ax.set_title(f"各阶段月均多空收益与 95% 置信区间（{ttl}）")
        ax.set_xlim(-.55, len(STAGES) - .25)
    bot[0].set_ylabel("多空收益（%/月）")
    bot[0].legend(loc="upper right", fontsize=8, ncols=1)
    save_fig(fig, c.cfg, "r_fig_stage")


# ---------------------------------------------------------------- 持有期内各月

def holding_month(c, series, sigs):
    """形成后第 k 个月（5 月 = 1，次年 4 月 = 12）。每个 k 每个形成年一个观测，t 值用形成年之间的标准误。"""
    rows = []
    for (sig, name, w), ls in series.items():
        if sig not in sigs:
            continue
        d = ls.to_frame("r")
        d["fy"] = [m.year - (m.month < 5) for m in d.index]
        d["k"] = [(m.month - 5) % 12 + 1 for m in d.index]
        for k, g in d.groupby("k"):
            x = g["r"]
            t = x.mean() / x.std(ddof=1) * np.sqrt(len(x))
            rows.append({"signal": sig, "构建": name, "权重": w, "k": int(k), "月份": f"{(k + 3) % 12 + 1} 月", "n": len(x),
                         "mean": x.mean() * PCT, "t": t, "p": 2 * st.t.sf(abs(t), len(x) - 1), "win": (x > 0).mean()})
        for lab, ks in {"第 1–3 个月（5–7 月）": (1, 3), "第 4–6 个月（8–10 月）": (4, 6), "第 7–9 个月（11–次年 1 月）": (7, 9),
                        "第 10–12 个月（次年 2–4 月）": (10, 12), "前 6 个月": (1, 6), "后 6 个月": (7, 12)}.items():
            g = d[d["k"].between(*ks)].groupby("fy")["r"].mean()
            t = g.mean() / g.std(ddof=1) * np.sqrt(len(g))
            rows.append({"signal": sig, "构建": name, "权重": w, "k": np.nan, "月份": lab, "n": len(g), "mean": g.mean() * PCT,
                         "t": t, "p": 2 * st.t.sf(abs(t), len(g) - 1), "win": (g > 0).mean()})
    return pd.DataFrame(rows)


def holding_ic(c, ics, sigs):
    rows = []
    for sig in sigs:
        for kind, i in ics[sig].items():
            d = i.dropna().to_frame("ic")
            d["k"] = [(m.month - 5) % 12 + 1 for m in d.index]
            for k, g in d.groupby("k"):
                t = g["ic"].mean() / g["ic"].std(ddof=1) * np.sqrt(len(g))
                rows.append({"signal": sig, "口径": kind, "k": int(k), "IC": g["ic"].mean(), "t": t, "p": 2 * st.t.sf(abs(t), len(g) - 1)})
    return pd.DataFrame(rows)


def fig_holding(c, H, sig):
    d = H[(H.signal == sig) & (H["权重"] == "ew") & H["k"].notna()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    labs = ["5", "6", "7", "8", "9", "10", "11", "12", "1", "2", "3", "4"]
    for ax, name, col in zip(axes, NAMES, SERIES):
        r = d[d["构建"] == name].sort_values("k")
        half = st.t.ppf(.975, r["n"] - 1) * (r["mean"] / r["t"]).abs()
        x = r["k"].to_numpy()
        ax.bar(x, r["mean"], .62, color=col, alpha=.9, zorder=2)
        ax.errorbar(x, r["mean"], yerr=half, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
        for xi, m, h, p in zip(x, r["mean"], half, r["p"]):
            if stars(p):
                ax.text(xi, m + (h + .15 if m >= 0 else -h - .15), stars(p), ha="center", va="bottom" if m >= 0 else "top", fontsize=9, color=INK2)
        ax.axhline(0, color=INK2, lw=.8)
        ax.axvspan(6.5, 12.5, color=MUTED, alpha=.08, lw=0)
        ax.text(9.5, .98, "后 6 个月", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=INK2)
        ax.set_xticks(range(1, 13), labs)
        ax.set_xlabel("持有月份（5 月调仓，至次年 4 月）")
        ax.set_title(f"{name}（等权）")
    axes[0].set_ylabel("多空收益（%/月），误差线为 95% 置信区间")
    save_fig(fig, c.cfg, "r_fig_holding_month")


# ---------------------------------------------------------------- 小市值内部：分阶段、增量信息

def increment(c, pairs):
    """非线性模型的排名对逻辑回归 L1 排名逐年回归取残差，看残差能否预测借壳（AUC）、能否带来小市值内部收益。"""
    rows = []
    ch = c.ch
    for sig, base in pairs:
        res = neutral_rank(ch, ch[sig], [base])
        ok = ~np.isnan(res) & ch["Y"].notna().to_numpy()
        auc = roc_auc_score(ch["Y"].to_numpy()[ok], res[ok])
        auc_by = {lab: roc_auc_score(ch["Y"][ok & ch["year"].between(a, b).to_numpy()], res[ok & ch["year"].between(a, b).to_numpy()])
                  for lab, a, b in [("2012–2016", 2012, 2016), ("2017–2023", 2017, 2023)]}
        rho = pd.Series(res).corr(ch[base], method="spearman")
        ls = strategies(c, res)
        for w in ("ew", "vw"):
            s = ls[(NAMES[2], w)]
            r = {"signal": sig, "base": base, "label": f"{c.lab[sig]} 相对 {c.lab[base]} 的残差", "权重": w,
                 "AUC_全": auc, "AUC_2012–2016": auc_by["2012–2016"], "AUC_2017–2023": auc_by["2017–2023"], "rho_base": rho}
            for lab, a, b in ALL:
                m, tm, pm = mean_p(c, s.loc[a:b])
                al, ta, pa = alpha_p(c, s.loc[a:b])
                r |= {f"mean_{lab}": m * PCT, f"p_mean_{lab}": pm, f"a_{lab}": al * PCT, f"t_a_{lab}": ta, f"p_a_{lab}": pa}
            rows.append(r)
    return pd.DataFrame(rows)


def implied(c, T, sigs, car=None):
    """最小 30% 内部：高、低壳概率组（Q5、Q1）当年实际被借壳的比例。若多空收益全部来自壳价值，
    它的量级约为（借壳率差 × 借壳事件的累计超额收益）/ 12。事件累计超额收益取 event_caar 在完成月（τ = 0）的值。"""
    if car is None:
        ev = pd.read_csv(c.cfg["paths"]["tables"] / "event_caar.csv")
        car = float(ev.loc[ev["tau"] == 0, "caar"].iloc[0]) / PCT
    ch = c.ch
    b3 = cut_group(ch, ch["size"])
    # 形成年 → 阶段（形成年 t 覆盖 t 年 5 月至 t+1 年 4 月，按大部分月份归属）
    fy = {"宽松期": (2012, 2015), "严控期": (2016, 2018), "放松+注册制": (2019, 2023), "全样本": (2012, 2023)}
    rows = []
    for sig in sigs:
        q = qgroup(ch, ch[sig], 5, by=b3)
        d = ch.assign(q=q, b3=b3)
        d = d[(d["b3"] == 1) & d["q"].isin([1.0, 5.0])]
        for lab, (a, b) in fy.items():
            g = d[d["year"].between(a, b)].groupby("q")["Y"].agg(["mean", "sum", "count"])
            dr = g.at[5.0, "mean"] - g.at[1.0, "mean"]
            al = T[(T.signal == sig) & (T["构建"] == NAMES[2]) & (T["权重"] == "ew") & (T["阶段"] == lab)]["a_FF6"].iloc[0]
            rows.append({"signal": sig, "label": c.lab[sig], "阶段": lab, "形成年": f"{a}–{b}",
                         "Q5 借壳数": int(g.at[5.0, "sum"]), "Q5 公司-年": int(g.at[5.0, "count"]), "Q5 借壳率(%)": g.at[5.0, "mean"] * PCT,
                         "Q1 借壳数": int(g.at[1.0, "sum"]), "Q1 借壳率(%)": g.at[1.0, "mean"] * PCT,
                         "事件累计超额收益(%)": car * PCT, "隐含月度壳溢价(%)": dr * car / 12 * PCT, "实际 FF6 α(%)": al})
    return pd.DataFrame(rows)


def run(cfg):
    c = Ctx(cfg, build(cfg))
    print(f"面板：公司-年 {len(c.ch):,}，股票-月 {len(c.mp):,}，{len(c.months)} 个月", flush=True)
    T, series = stage_tables(c, SMALL)
    save_table(T, cfg, "r_stage_all")
    print("  分阶段 → r_stage_all", flush=True)
    I, ics = ic_table(c, SMALL)
    save_table(I, cfg, "r_rank_ic", {"IC": ".4f"})
    print("  Rank IC → r_rank_ic", flush=True)
    H = holding_month(c, series, MAIN3)
    save_table(H, cfg, "r_holding_month")
    save_table(holding_ic(c, ics, MAIN3), cfg, "r_holding_ic", {"IC": ".4f"})
    print("  持有期各月 → r_holding_month, r_holding_ic", flush=True)
    inc = increment(c, [("mlp_s1", "logit_l1_s0"), ("xgb_s1", "logit_l1_s0"), ("rf_s1", "logit_l1_s0"), ("logit_s0", "logit_l1_s0")])
    save_table(inc, cfg, "r_increment")
    print("  增量信息 → r_increment", flush=True)
    save_table(implied(c, T, ["logit_l1_s0", "xgb_s1", "mlp_s1", "ensemble"]), cfg, "r_implied")
    print("  隐含壳溢价 → r_implied", flush=True)
    fig_stage(c, T, series, c.main)
    fig_holding(c, H, c.main)
    print("  图 → r_fig_stage, r_fig_holding_month", flush=True)
