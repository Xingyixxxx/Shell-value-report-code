"""全部检验。每个函数写一张或几张表到 output/tables，图写到 output/figures。
收益、alpha 单位都是 %/月（表头注明），t 值都用 Newey-West。"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from shellpred.plot import INK2, MUTED, SERIES, DIVERGING, plt, save_fig, save_table

from . import stats as S
from .panel import FACTOR_MODELS, build
from .portfolios import (annual_turnover, cut_group, long_short, neutral_rank, port, qgroup, rank_weighted,
                         stats as perf, topk)

PCT = 100
CONTROLS = ["size", "lBM", "EP", "loss", "GP", "IA", "turn", "mom", "rev"]
MON = ["size", "lBM", "EP", "loss", "GP", "IA", "turn", "rev", "mom"]


class Ctx:
    def __init__(self, cfg, D):
        self.cfg, self.D, self.t = cfg, D, cfg["test"]
        self.lag = self.t["nw_lag"]
        self.ch, self.mp, self.F = D.chars, D.mp, D.F
        self.core = [r for r in cfg["signal"]["core"] if r in D.runs]
        self.main = cfg["signal"]["main"]
        self.months = D.F.index
        self.lab = D.labels
        self.neutral = {}

    def fm(self, name):
        return self.F[FACTOR_MODELS[name]]

    def alpha(self, y, name=None):
        return S.alpha(y, self.fm(name or self.t["primary_model"]), self.lag)

    def row(self, y, models=None):
        """均值 + 指定因子模型的 alpha。"""
        m, t, _ = S.mean_t(y, self.lag)
        out = {"mean": m * PCT, "t_mean": t}
        for k in models or self.t["factor_models"]:
            a = self.alpha(y, k)
            out[f"a_{k}"], out[f"t_{k}"] = a["alpha"] * PCT, a["t"]
        return out

    def excess(self, P):
        return P.sub(self.F["rf"], axis=0)

    def neutral_signal(self, sig):
        """规模 + 行业中性后的信号（缓存）。"""
        if sig not in self.neutral:
            self.neutral[sig] = neutral_rank(self.ch, self.ch[sig], ["size"], ind=True)
        return self.neutral[sig]


# ---------------------------------------------------------------- 数据核对

def qa(c):
    ch, F, long = c.ch, c.F, c.D.long
    y = ch.groupby("year").agg(firms=("Stkcd", "size"), post=("post", "sum"), events=("Y", "sum"), loss=("loss", "mean"), st=("st", "mean"))
    n = c.mp[~c.mp["susp"].astype(bool)].groupby("y")["month"].nunique().rename("months")
    obs = c.mp[~c.mp["susp"].astype(bool)].groupby("y").size().rename("stock_months")
    save_table(y.join(n).join(obs).reset_index(), c.cfg, "qa_sample", {"loss": ".3f", "st": ".3f"})

    d = long[~long["susp"].astype(bool) & long["mvf"].notna()]
    mkt = (d["ret"] * d["mvf"]).groupby(d["month"]).sum() / d["mvf"].groupby(d["month"]).sum()
    implied = (mkt - F["MKT"]).reindex(F.index)
    rows = [{"检验": "自算主板流通市值加权收益 − CSMAR MKT 的均值（%/月，应≈无风险利率）", "值": implied.mean() * PCT},
            {"检验": "所用无风险利率的均值（%/月）", "值": F["rf"].mean() * PCT},
            {"检验": "自算市场超额收益与 CSMAR MKT 的相关", "值": (mkt - F["rf"]).corr(F["MKT"])},
            {"检验": "自构 CH3 SMB 与 CSMAR SMB 的相关", "值": F["SMB_CH"].corr(F["SMB"])},
            {"检验": "自构 CH3 VMG 与 CSMAR HML 的相关", "值": F["VMG"].corr(F["HML"])}]
    save_table(pd.DataFrame(rows), c.cfg, "qa_factors")
    s = pd.DataFrame({k: {"mean": F[k].mean() * PCT, **dict(zip(["_", "t"], [0, S.mean_t(F[k], c.lag)[1]])), "std": F[k].std() * PCT}
                      for k in ["MKT", "SMB", "HML", "RMW", "CMA", "UMD", "SMB_CH", "VMG"]}).T.drop(columns="_")
    save_table(s.reset_index(names="factor"), c.cfg, "qa_factor_summary")
    rho = c.ch[c.D.runs].corr(method="spearman")
    save_table(rho.loc[c.core, c.core].reset_index(names="signal"), c.cfg, "signal_rank_corr")


def model_compare(c):
    """保留借壳后样本前后的预测对比：同一批公司-年上的排序能力与排名相关。"""
    from .panel import read_signals
    prev = read_signals(c.cfg["paths"]["prev_outputs"])
    wr = lambda x, col: x.groupby("year")[col].rank(pct=True)
    auc = lambda x, col: roc_auc_score(x["Y"], wr(x, col))
    if not prev:  # 旧预测不在时跳过对比，只留图 fig_auc_alpha 要用的 AUC
        return pd.DataFrame([{"run": r, "label": c.lab[r], "AUC_new_all": auc(d, "p")} for r, d in c.D.tests.items()])
    rows = []
    for r, d in c.D.tests.items():
        if r not in prev:
            continue
        m = d.merge(prev[r], on=["Stkcd", "year"], suffixes=("", "_old"))
        post = d.merge(c.ch[["Stkcd", "year", "post"]], on=["Stkcd", "year"], how="left")
        rk = post.assign(r=post.groupby("year")["p"].rank(pct=True))
        rows.append({"run": r, "label": c.lab[r], "n_common": len(m), "AUC_old": auc(m, "p_old"), "AUC_new_common": auc(m, "p"),
                     "AUC_new_all": auc(d, "p"), "rank_corr": wr(m, "p").corr(wr(m, "p_old"), method="spearman"),
                     "post_mean_rank": rk.loc[rk["post"] == True, "r"].mean()})
    t = pd.DataFrame(rows)
    save_table(t, c.cfg, "model_compare_v2")
    return t


# ---------------------------------------------------------------- 全部信号的主表与多重检验

def grid(c):
    rows, E, se = [], {}, []
    for sig in c.D.runs:
        for w in c.t["headline_weights"]:
            for n in c.t["groups"]:
                P = port(c.mp, qgroup(c.ch, c.ch[sig], n), w)
                ls = long_short(P)
                r = {"signal": sig, "label": c.lab[sig], "weight": w, "n": n, "T": len(ls), **c.row(ls)}
                a = c.alpha(ls)
                E[(sig, w, n)] = a["e"]
                se.append(a["se"])
                r["p_FF6"] = a["p"]
                rows.append(r)
    t = pd.DataFrame(rows)
    Em = pd.DataFrame(E).reindex(c.months)
    t["p_RW"] = S.romano_wolf(Em, np.array(se), c.t["bootstrap"], c.t["block"], c.t["seed"])
    t["p_BH"] = multipletests(t["p_FF6"], method="fdr_bh")[1]
    t["p_Holm"] = multipletests(t["p_FF6"], method="holm")[1]
    t["HLZ_t3"] = t["t_FF6"].abs() > 3
    save_table(t, c.cfg, "grid_all")
    return t


def selection(c, g):
    """事先写定的打分（config [selection]）：六项指标在 全部信号 × {ew, vw} 上的平均百分位。"""
    pri, sel = c.t["primary_model"], c.cfg["selection"]
    rows = []
    d = fm_frame(c)
    for sig in c.D.runs:
        for w in c.t["headline_weights"]:
            grp = qgroup(c.ch, c.ch[sig], 10)
            P = port(c.mp, grp, w)
            ls = long_short(P)
            a, ch3 = c.alpha(ls), c.alpha(ls, "CH3")
            ln = c.alpha(c.excess(P).iloc[:, -1])
            st_ = c.alpha(c.excess(P).iloc[:, 0])
            nt = c.alpha(long_short(port(c.mp, qgroup(c.ch, c.neutral_signal(sig), 10), w)))
            per = [c.alpha(ls.loc[a_:b_]) for a_, b_ in c.t["periods"]]
            fm = S.fama_macbeth(d.assign(sig=c.ch[sig].to_numpy()[d["cid"].to_numpy()]), "ret", ["sig", *MON], c.lag, weight=None if w == "ew" else "mvf")
            _, mr = S.monotonic(c.excess(P), c.t["bootstrap"] // 2, c.t["block"], c.t["seed"])
            rows.append({"signal": sig, "label": c.lab[sig], "weight": w, "a_FF6": a["alpha"] * PCT, "t_FF6": a["t"], "t_CH3": ch3["t"],
                         "t_FF6_neutral": nt["t"], "a_FF6_neutral": nt["alpha"] * PCT, "t_FM": fm["t_sig"], "b_FM": fm["sig"] * PCT,
                         "a_pre": per[0]["alpha"] * PCT, "t_pre": per[0]["t"], "a_post": per[1]["alpha"] * PCT, "t_post": per[1]["t"],
                         "t_FF6_min_period": min(p["t"] for p in per), "a_long": ln["alpha"] * PCT, "t_FF6_long": ln["t"],
                         "a_short": st_["alpha"] * PCT, "t_short": st_["t"], "MR_p": mr, "mean_ret": ls.mean() * PCT, **{k: v for k, v in perf(ls).items() if k in ("sharpe", "mdd")}})
    t = pd.DataFrame(rows)
    t["score"] = t[sel["criteria"]].rank(pct=True).mean(axis=1)
    rw = g[g["n"] == 10][["signal", "weight", "p_RW", "p_BH"]]
    t = t.merge(rw, on=["signal", "weight"])
    t["pass_RW"] = t["p_RW"] < sel["rw_alpha"]
    t = t.sort_values("score", ascending=False)
    save_table(t, c.cfg, "selection")
    # 拆样本选择：前半段选 t 最高的，看后半段；反过来再做一次
    ab = []
    for k, (a_, b_) in zip(("pre", "post"), c.t["periods"]):
        o = "post" if k == "pre" else "pre"
        best = t.sort_values(f"t_{k}", ascending=False).iloc[0]
        ab.append({"选择期": f"{a_}–{b_}", "选中": f"{best['signal']} / {best['weight']}", "选择期 alpha": best[f"a_{k}"], "选择期 t": best[f"t_{k}"],
                   "另一期 alpha": best[f"a_{o}"], "另一期 t": best[f"t_{o}"], "该期 t 与另一期 t 的 Spearman": t[f"t_{k}"].corr(t[f"t_{o}"], method="spearman")})
    save_table(pd.DataFrame(ab), c.cfg, "selection_split")
    return t


# ---------------------------------------------------------------- 组合构建方式

def construction(c):
    rows = []
    ch, mp = c.ch, c.mp
    for sig in c.core:
        s = ch[sig].to_numpy()
        d10 = qgroup(ch, s, 10)
        spec = {f"10 分组 {w}": (long_short(port(mp, d10, w)), d10, 10) for w in c.t["weights"]}
        spec["5 分组 ew"] = (long_short(port(mp, g := qgroup(ch, s, 5), "ew")), g, 5)
        spec["5 分组 vw"] = (long_short(port(mp, g, "vw")), g, 5)
        spec["20 分组 vw（原论文）"] = (long_short(port(mp, g20 := qgroup(ch, s, 20), "vw")), g20, 20)
        spec["30/40/30 ew"] = (long_short(port(mp, g3 := cut_group(ch, s), "ew")), g3, 3)
        spec["30/40/30 vw"] = (long_short(port(mp, g3, "vw")), g3, 3)
        spec["前后 50 只 ew"] = (long_short(port(mp, t50 := topk(ch, s, 50), "ew")), t50, 2)
        spec["前后 50 只 vw"] = (long_short(port(mp, t50, "vw")), t50, 2)
        spec["前后 100 只 ew"] = (long_short(port(mp, t100 := topk(ch, s, 100), "ew")), t100, 2)
        spec["信号加权（不分组）"] = (rank_weighted(mp, s), None, None)
        spec["10 分组 ew 规模调整收益"] = (long_short(port(mp, d10, "ew", "ret_sa")), d10, 10)
        spec["10 分组 vw 规模调整收益"] = (long_short(port(mp, d10, "vw", "ret_sa")), d10, 10)
        # 2×3 因子：流通市值中位数 × 壳概率 30/70，(SH+BH)/2 − (SL+BL)/2
        sz = cut_group(ch, ch["size"], (.5,))
        g6 = (sz - 1) * 3 + cut_group(ch, s, by=sz)
        for w in ("ew", "vw"):
            P6 = port(mp, g6, w)
            spec[f"2×3 因子 {w}"] = ((P6[3.0] + P6[6.0] - P6[1.0] - P6[4.0]) / 2, None, None)
        for name, (ls, g, top) in spec.items():
            ls = ls.loc[c.months[0]:]
            tv = annual_turnover(ch, g, top) if g is not None else np.nan
            bs = tv * 2 * 2 if g is not None else np.nan  # 多空两腿，每腿换入换出
            r = {"signal": sig, "method": name, **c.row(ls, ["FF6", "CH3"]), **perf(ls), "turnover": tv,
                 "cost_ann": bs * c.t["cost_oneway"] * PCT if g is not None else np.nan}
            rows.append(r)
    t = pd.DataFrame(rows)
    t["ann_ret"], t["ann_vol"], t["mdd"], t["win"] = t["ann_ret"] * PCT, t["ann_vol"] * PCT, t["mdd"] * PCT, t["win"] * PCT
    t["net_ann_ret"] = t["ann_ret"] - t["cost_ann"]
    save_table(t, c.cfg, "construction")
    return t


def deciles(c):
    rows, mr, gr, ch_rows = [], [], [], []
    for sig in c.core:
        g = qgroup(c.ch, c.ch[sig], 10)
        for w in ("ew", "vw"):
            P = c.excess(port(c.mp, g, w))
            for k in P.columns:
                a = c.alpha(P[k])
                m, t, _ = S.mean_t(P[k], c.lag)
                rows.append({"signal": sig, "weight": w, "decile": int(k), "ex_ret": m * PCT, "t_ret": t, "a_FF6": a["alpha"] * PCT, "t_FF6": a["t"]})
            d, p = S.monotonic(P, c.t["bootstrap"], c.t["block"], c.t["seed"])
            rho = pd.Series(P.mean().to_numpy()).corr(pd.Series(range(10)), method="spearman")
            mr.append({"signal": sig, "weight": w, "min_step": d * PCT, "MR_p": p, "spearman_decile": rho,
                       **{f"GRS_{k}": S.grs(P, c.fm(k))["GRS"] for k in ("FF6", "CH3")},
                       **{f"GRS_p_{k}": S.grs(P, c.fm(k))["p"] for k in ("FF6", "CH3")}})
        if sig in (c.main, "ensemble", "logit_s0"):
            x = c.ch.assign(d=g).groupby("d").agg(mean_mv=("mvf", lambda v: np.exp(np.log(v).mean()) / 1e6), lBM=("lBM", "mean"), EP=("EP", "mean"),
                                                 loss=("loss", "mean"), GP=("GP", "mean"), turn=("turn", "mean"), mom=("mom", "mean"), st=("st", "mean"),
                                                 soe=("Soe_1", "mean"), post=("post", "mean"), hit=("Y", "mean"), events=("Y", "sum"), n=("Y", "size"))
            ch_rows.append(x.reset_index().assign(signal=sig))
    t = pd.DataFrame(rows)
    save_table(t, c.cfg, "deciles")
    save_table(pd.DataFrame(mr), c.cfg, "deciles_test")
    save_table(pd.concat(ch_rows)[["signal", "d", "n", "events", "hit", "mean_mv", "lBM", "EP", "loss", "GP", "turn", "mom", "st", "soe", "post"]], c.cfg, "deciles_chars",
               {"hit": ".4f", "mean_mv": ".2f"})
    return t


# ---------------------------------------------------------------- 双重分组与联合因子

def double_sorts(c):
    ch, rows = c.ch, []
    dims = {"size": ch["size"], "lBM": ch["lBM"], "EP": ch["EP"], "GP": ch["GP"], "IA": ch["IA"], "turn": ch["turn"], "mom": ch["mom"], "rev": ch["rev"],
            "ST": ch["st"].astype(float), "SOE": ch["Soe_1"], "Post": ch["post"].astype(float)}
    for sig in c.core:
        s = ch[sig].to_numpy()
        for dim, v in dims.items():
            if dim in ("ST", "SOE", "Post"):
                b = (v.to_numpy(float) > 0).astype(float) + 1.0
                b = np.where(v.isna(), np.nan, b)
            else:
                b = cut_group(ch, v)
            for how in ("dep", "ind"):
                inner = qgroup(ch, s, 5, by=b if how == "dep" else None, mask=~np.isnan(b))
                for w in ("ew", "vw"):
                    ser = {}
                    for k in sorted(set(b[~np.isnan(b)])):
                        P = port(c.mp, np.where(b == k, inner, np.nan), w)
                        if len(P.columns) < 2 or {1.0, 5.0} - set(P.columns):
                            continue
                        ser[k] = long_short(P[[1.0, 5.0]])
                    if not ser:
                        continue
                    if len(ser) > 1:
                        ser["avg"] = pd.concat(ser, axis=1).mean(axis=1, skipna=False).dropna()
                        high, low = max(k for k in ser if k != "avg"), min(k for k in ser if k != "avg")
                        ser["high−low"] = (ser[high] - ser[low]).dropna()
                    for k, ls in ser.items():
                        if len(ls) < 24:
                            continue
                        rows.append({"signal": sig, "dim": dim, "how": how, "weight": w, "bucket": k, **c.row(ls, ["FF6", "CH3"])})
    t = pd.DataFrame(rows).astype({"bucket": str})
    save_table(t, c.cfg, "double_sorts")
    # 原论文口径：E/P 30/70 内再分 20 组，流通市值加权
    rows = []
    for sig in c.core:
        b = cut_group(ch, ch["EP"])
        g = qgroup(ch, ch[sig], 20, by=b)
        for k in (1.0, 2.0, 3.0):
            rows.append({"signal": sig, "EP_bucket": {1.0: "L", 2.0: "M", 3.0: "H"}[k], **c.row(long_short(port(c.mp, np.where(b == k, g, np.nan), "vw")[[1.0, 20.0]]), ["FF6", "CH3"])})
    save_table(pd.DataFrame(rows), c.cfg, "double_ep_paper")
    return t


def neutralized(c):
    rows = []
    specs = {"原始": ([], False), "规模中性": (["size"], False), "行业中性": ([], True), "规模+行业中性": (["size"], True),
             "规模+行业+BM+EP+换手+反转+动量": (["size", "lBM", "EP", "loss", "turn", "rev", "mom"], True)}
    for sig in c.core:
        for name, (num, ind) in specs.items():
            s = neutral_rank(c.ch, c.ch[sig], num, ind) if (num or ind) else c.ch[sig].to_numpy()
            for w in ("ew", "vw"):
                rows.append({"signal": sig, "spec": name, "weight": w, **c.row(long_short(port(c.mp, qgroup(c.ch, s, 10), w)), ["FF3", "FF6", "CH3"])})
    t = pd.DataFrame(rows)
    save_table(t, c.cfg, "neutralized")
    return t


def composite(c):
    """联合因子。排名平均（复合信号）、两维交集、多模型一致。检验相对组成部分是否有增量。"""
    ch, rows, inc = c.ch, [], []
    pr = lambda v, sign=1: (sign * pd.Series(v, index=ch.index)).groupby(ch["year"]).rank(pct=True)
    for sig in c.core:
        s = pr(ch[sig])
        partner = {"小市值": pr(ch["size"], -1), "高 EP": pr(ch["EP"]), "低换手": pr(ch["turn"], -1), "短期反转": pr(ch["rev"], -1), "低 BM": pr(ch["lBM"], -1), "高 BM": pr(ch["lBM"])}
        for name, v in partner.items():
            comp = (s + v.fillna(.5)) / 2
            for w in ("ew", "vw"):
                ls = long_short(port(c.mp, qgroup(ch, comp, 10), w))
                base = long_short(port(c.mp, qgroup(ch, s, 10), w))
                other = long_short(port(c.mp, qgroup(ch, v, 10), w))
                span = S.alpha(ls, pd.concat([c.fm("FF6"), base.rename("base"), other.rename("other")], axis=1), c.lag)
                rows.append({"signal": sig, "joint": f"壳概率 + {name}", "kind": "排名平均", "weight": w, **c.row(ls, ["FF6", "CH3"]),
                             "t_other_FF6": c.alpha(other)["t"], "a_span": span["alpha"] * PCT, "t_span": span["t"]})
                # 再看壳概率相对另一特征的增量：壳概率 L-S 对 FF6 + 另一特征 L-S 回归
                inc.append({"signal": sig, "partner": name, "weight": w, **dict(zip(["a_shell", "t_shell"], (lambda a: (a["alpha"] * PCT, a["t"]))(S.alpha(base, pd.concat([c.fm("FF6"), other.rename("other")], axis=1), c.lag))))})
        # 交集：高壳概率 × 小市值 − 低壳概率 × 大市值
        sh, sz = cut_group(ch, s), cut_group(ch, ch["size"])
        g = np.where((sh == 3) & (sz == 1), 2.0, np.where((sh == 1) & (sz == 3), 1.0, np.nan))
        for w in ("ew", "vw"):
            rows.append({"signal": sig, "joint": "高壳×小市值 − 低壳×大市值", "kind": "交集", "weight": w, **c.row(long_short(port(c.mp, g, w)), ["FF6", "CH3"])})
    # 多模型一致：全部成员都在前 20% vs 全部成员都在后 20%
    m = c.cfg["signal"]["ensemble"]
    if all(x in ch for x in m):
        low, high = ch[m].max(axis=1) <= .2, ch[m].min(axis=1) >= .8
        g = np.where(high, 2.0, np.where(low, 1.0, np.nan))
        for w in ("ew", "vw"):
            rows.append({"signal": "members", "joint": "6 模型全部前 20% − 全部后 20%", "kind": "一致", "weight": w, **c.row(long_short(port(c.mp, g, w)), ["FF6", "CH3"])})
    t = pd.DataFrame(rows)
    save_table(t, c.cfg, "composite")
    save_table(pd.DataFrame(inc), c.cfg, "composite_increment")
    return t


# ---------------------------------------------------------------- 横截面回归

def fm_frame(c):
    """逐月横截面回归的面板：控制变量取形成时的特征（年度）与逐月滞后变量；按月缩尾 1/99 并标准化。"""
    d = c.mp[~c.mp["susp"].astype(bool)][["cid", "month", "ret", "mvf", "turn", "rev", "mom"]].copy()
    d["size"] = np.log(d["mvf"])
    d = d.merge(c.ch[["cid", "lBM", "EP", "loss", "GP", "IA", "ind", "post"]], on="cid")
    for x in ["size", "lBM", "EP", "GP", "IA", "turn", "rev", "mom"]:
        g = d.groupby("month")[x]
        v = d[x].clip(g.transform(lambda s: s.quantile(.01)), g.transform(lambda s: s.quantile(.99)))
        d[x] = (v - v.groupby(d["month"]).transform("mean")) / v.groupby(d["month"]).transform("std")
    return d


def cross_section(c):
    d = fm_frame(c)
    specs = {"(1) 单变量": [], "(2) +规模 BM": ["size", "lBM"], "(3) +动量 反转": ["size", "lBM", "mom", "rev"], "(4) +EP 亏损 GP IA": ["size", "lBM", "EP", "loss", "GP", "IA", "mom", "rev"],
             "(5) 全部控制": MON}
    rows, ic = [], []
    for sig in c.core:
        x = d.assign(sig=c.ch[sig].to_numpy()[d["cid"].to_numpy()])
        for name, xs in specs.items():
            r = S.fama_macbeth(x, "ret", ["sig", *xs], c.lag)
            rows.append({"signal": sig, "spec": name, "b": r["sig"] * PCT, "t": r["t_sig"], "r2": r["r2"], "n": r["n"], "T": r["T"], **{f"t_{k}": r.get(f"t_{k}") for k in MON}})
        ex = {"(6) 全部控制 + 行业固定效应": S.fama_macbeth(x, "ret", ["sig", *MON], c.lag, fe="ind"),
              "(7) 全部控制，市值加权 WLS": S.fama_macbeth(x, "ret", ["sig", *MON], c.lag, weight="mvf"),
              "(8) 全部控制，剔除最小 30%": S.fama_macbeth(x[x["size"] > x.groupby("month")["size"].transform(lambda v: v.quantile(.3))], "ret", ["sig", *MON], c.lag),
              "(9) 全部控制，剔除借壳后": S.fama_macbeth(x[~x["post"].astype(bool)], "ret", ["sig", *MON], c.lag)}
        for name, r in ex.items():
            rows.append({"signal": sig, "spec": name, "b": r["sig"] * PCT, "t": r["t_sig"], "r2": r["r2"], "n": r["n"], "T": r["T"], **{f"t_{k}": r.get(f"t_{k}") for k in MON}})
        i = x.dropna(subset=["sig", "ret"]).groupby("month").apply(lambda v: v["sig"].corr(v["ret"], method="spearman"), include_groups=False)
        m, t, _ = S.mean_t(i, c.lag)
        ic.append({"signal": sig, "rank_IC": m, "t": t, "ICIR_ann": m / i.std() * np.sqrt(12), "IC_pos": (i > 0).mean()})
    t = pd.DataFrame(rows)
    save_table(t, c.cfg, "fama_macbeth")
    save_table(pd.DataFrame(ic), c.cfg, "rank_ic", {"rank_IC": ".4f"})
    # 面板固定效应：月份 FE 与 公司 + 月份 FE，公司、月份双重聚类
    rows = []
    for sig in c.core:
        x = d.assign(sig=c.ch[sig].to_numpy()[d["cid"].to_numpy()], firm=c.ch["Stkcd"].to_numpy()[d["cid"].to_numpy()]).dropna(subset=["sig", "ret", *MON])
        for name, fes in {"月份 FE": ["month"], "公司 + 月份 FE": ["firm", "month"]}.items():
            cols = ["ret", "sig", *MON]
            z = x[cols].astype(float).copy()
            for _ in range(15 if len(fes) > 1 else 1):
                for fe in fes:
                    z = z - z.groupby(x[fe]).transform("mean")
            m = sm.OLS(z["ret"], z[["sig", *MON]]).fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([pd.factorize(x["firm"])[0], pd.factorize(x["month"])[0]])})
            rows.append({"signal": sig, "spec": name, "b": m.params["sig"] * PCT, "t": m.tvalues["sig"], "n": int(m.nobs)})
    save_table(pd.DataFrame(rows), c.cfg, "panel_fe", {"n": ",.0f"})
    return t


# ---------------------------------------------------------------- 时变与政策

def regimes(c):
    rows, by_y, reg = [], [], []
    ipo = pd.read_excel(c.cfg["paths"]["ipo"])
    ipo["m"] = pd.PeriodIndex(pd.to_datetime(ipo["month"]), freq="M")
    cnt = ipo.set_index("m")["count"].reindex(pd.period_range("2010-01", c.months[-1], freq="M")).fillna(0)
    tight = -(cnt.shift(1).rolling(6).mean())  # 过去 6 个月上会家数越少，IPO 越紧
    tight = ((tight - tight.mean()) / tight.std()).reindex(c.months)
    pause = pd.Series(0.0, index=c.months)
    for a, b in c.t["ipo_pauses"]:
        pause.loc[a:b] = 1.0
    post = pd.Series(0.0, index=c.months)
    post.loc[c.t["periods"][1][0]:] = 1.0
    for sig in c.core:
        g = qgroup(c.ch, c.ch[sig], 10)
        for w in ("ew", "vw"):
            P = port(c.mp, g, w)
            ls = long_short(P)
            for a, b in c.t["periods"]:
                rows.append({"signal": sig, "weight": w, "period": f"{a}–{b}", "T": len(ls.loc[a:b]), **c.row(ls.loc[a:b], ["FF3", "FF6", "CH3"])})
                lg = c.excess(P).iloc[:, -1].loc[a:b]
                al = c.alpha(lg)
                rows[-1] |= {"a_long": al["alpha"] * PCT, "t_long": al["t"]}
            for name, flag in {"IPO 暂停月": pause == 1, "IPO 非暂停月": pause == 0}.items():
                s = ls[flag.reindex(ls.index).fillna(False)]
                rows.append({"signal": sig, "weight": w, "period": name, "T": len(s), **c.row(s, ["FF3", "FF6", "CH3"])})
            for y, s in ls.groupby(ls.index.map(lambda m: m.year - (m.month < 5))):
                by_y.append({"signal": sig, "weight": w, "year": y, "ret": ((1 + s).prod() - 1) * PCT})
            for name, x in {"IPO 暂停": pause, "重组新规后": post, "IPO 紧度（标准化）": tight}.items():
                a = S.alpha(ls, pd.concat([c.fm("FF6"), x.rename("x")], axis=1), c.lag)
                reg.append({"signal": sig, "weight": w, "state": name, "b": a["betas"]["x"] * PCT, "t": a["tb"]["x"]})
    save_table(pd.DataFrame(rows), c.cfg, "regimes")
    y = pd.DataFrame(by_y)
    save_table(y, c.cfg, "by_year")
    save_table(pd.DataFrame(reg), c.cfg, "regime_regress")
    return pd.DataFrame(rows), y


def decay(c):
    """信号衰减：形成后第 k 个月的 D10 − D1。t 值用 cohort 间标准误（独立性好但样本少）。"""
    long = c.D.long[~c.D.long["susp"].astype(bool)][["Stkcd", "month", "ret", "mvf"]]
    rows = []
    for sig in c.core:
        ch = c.ch.assign(g=qgroup(c.ch, c.ch[sig], 10))[lambda x: x["g"].isin([1, 10])][["Stkcd", "year", "g"]]
        m = ch.merge(long, on="Stkcd")
        m["k"] = (m["month"].dt.year - m["year"]) * 12 + m["month"].dt.month - c.cfg["signal"]["apply_month"] + 1
        m = m[m["k"].between(1, 24)]
        for w, wc in {"ew": None, "vw": "mvf"}.items():
            mm = m.assign(w=1.0 if wc is None else m[wc]).dropna(subset=["w", "ret"])
            mm["rw"] = mm["ret"] * mm["w"]
            a = mm.groupby(["year", "k", "g"])[["rw", "w"]].sum()
            r = (a["rw"] / a["w"]).unstack("g")
            ls = (r[10] - r[1]).rename("ls").reset_index()
            z = ls.groupby("k")["ls"].agg(["mean", "std", "count"])
            rows.extend({"signal": sig, "weight": w, "k": k, "ret": v["mean"] * PCT, "t": v["mean"] / v["std"] * np.sqrt(v["count"]), "cohorts": int(v["count"])} for k, v in z.iterrows())
    t = pd.DataFrame(rows)
    t["cum"] = t.groupby(["signal", "weight"])["ret"].cumsum()
    save_table(t, c.cfg, "decay")
    return t


# ---------------------------------------------------------------- 稳健性

def _realign(c, apply_month, calendar=False):
    """换一种信号与持有期的对齐方式。calendar：旧 Stata 的口径（t 年 1–12 月用 t 年信号，1–4 月有前视）。"""
    h = c.D.long
    y = h["month"].dt.year if calendar else h["month"].dt.year - (h["month"].dt.month < apply_month).astype(int)
    m = h.assign(y=y).merge(c.ch[["Stkcd", "year", "cid"]].rename(columns={"year": "y"}), on=["Stkcd", "y"]).sort_values(["cid", "month"])
    m = m[(m["month"] >= (pd.Period("2012-01", "M") if calendar else pd.Period(f"2012-{apply_month:02d}", "M")))]
    return m.assign(w_ew=1.0, w_vw=m["mvf"])


def robustness(c):
    ch, mp, rows = c.ch, c.mp, []
    micro = ch["size"] <= ch.groupby("year")["size"].transform(lambda s: s.quantile(.3))
    ever = ch["event_year"].notna()
    ret_w = mp["ret"].clip(mp.groupby("month")["ret"].transform(lambda s: s.quantile(.01)), mp.groupby("month")["ret"].transform(lambda s: s.quantile(.99)))
    mp_w = mp.assign(ret=ret_w)
    jul, cal = _realign(c, 7), _realign(c, 5, calendar=True)
    for sig in c.core:
        s = ch[sig].to_numpy()
        g = lambda mask=None: qgroup(ch, s, 10, mask=mask)
        var = {"基准": (mp, g(), False),
               "剔除借壳后公司（形成时已知）": (mp, g(~ch["post"].astype(bool)), False),
               "剔除 ST（形成时已知）": (mp, g(~ch["st"].astype(bool)), False),
               "剔除最小 30%（形成时已知）": (mp, g(~micro), False),
               "剔除形成前月停牌（形成时已知）": (mp, g(ch["traded_last"].astype(bool)), False),
               "剔除当年被借壳公司（事后信息）": (mp, g(ch["Y"] == 0), False),
               "剔除所有曾被借壳公司（事后信息）": (mp, g(~ever), False),
               "停牌月收益记 0、留在组合": (mp, g(), True),
               "月收益按月缩尾 1/99": (mp_w, g(), False),
               "7 月才开始使用信号": (jul, g(), False),
               "旧口径：日历年对齐（1–4 月有前视）": (cal, g(), False)}
        for name, (m, grp, susp) in var.items():
            for w in ("ew", "vw"):
                ls = long_short(port(m, grp, w, include_susp=susp)).loc[lambda x: x.index <= c.months[-1]]
                rows.append({"signal": sig, "variant": name, "weight": w, "T": len(ls), **c.row(ls, ["FF3", "FF6", "CH3"])})
                if name == "基准":
                    a = S.alpha(ls, c.fm("FF6"), 12)
                    rows.append({"signal": sig, "variant": "NW 滞后 12 期", "weight": w, "T": len(ls), "mean": ls.mean() * PCT, "t_mean": S.mean_t(ls, 12)[1], "a_FF6": a["alpha"] * PCT, "t_FF6": a["t"]})
    t = pd.DataFrame(rows)
    save_table(t, c.cfg, "robustness")
    return t


def legacy_bridge(c):
    """从原论文的 Stata 口径一步步走到本文口径，看每一步改变多少：
    同月市值加权（权重含当月涨跌）、日历年对齐（1–4 月用尚未披露的年报）、原论文的壳概率。"""
    long = c.D.long[~c.D.long["susp"].astype(bool)].copy()
    paper = pd.read_csv(c.cfg["paths"]["data"] / "paper_shell_prob.csv", dtype={"Stkcd": str}).rename(columns={"paper_prob": "s"})
    new = lambda r: c.ch[["Stkcd", "year", r]].rename(columns={r: "s"})
    steps = [("原论文壳概率，日历年对齐，同月市值加权（原 Stata 口径）", paper, True, "mvf_now"),
             ("原论文壳概率，日历年对齐，上月市值加权", paper, True, "mvf"),
             ("原论文壳概率，5 月对齐，上月市值加权", paper, False, "mvf")]
    for r in dict.fromkeys(["xgb_sq_s2", c.main, "ensemble"]):
        steps += [(f"{r}，日历年对齐，同月市值加权", new(r), True, "mvf_now"), (f"{r}，5 月对齐，上月市值加权（本文口径）", new(r), False, "mvf")]
    rows = []
    for name, sig, cal, wc in steps:
        d = long.assign(year=long["month"].dt.year if cal else long["y"]).merge(sig, on=["Stkcd", "year"]).dropna(subset=["s", "ret", wc])
        d = d[d["month"] >= (pd.Period("2012-01", "M") if cal else c.months[0])]
        if sig is not paper:  # 原论文的概率保留并列，与 Stata 的 xtile 口径一致
            d["s"] += 1e-9 * np.random.default_rng(0).random(len(d))
        d["g"] = np.ceil(d.groupby("month")["s"].rank(pct=True) * 20).clip(1, 20)
        g = d.assign(rw=d["ret"] * d[wc]).groupby(["month", "g"])[["rw", wc]].sum()
        P = (g["rw"] / g[wc]).unstack("g")
        ls = (P[20.0] - P[1.0]).dropna()
        rows.append({"口径": name, "T": len(ls), "低组收益": P[1.0].mean() * PCT, "高组收益": P[20.0].mean() * PCT, **c.row(ls, ["FF3", "FF6", "CH3"])})
    save_table(pd.DataFrame(rows), c.cfg, "legacy_bridge")


def small_cap(c):
    """小市值内的壳概率溢价。这一块是看到双重分组结果之后才加的探索性检验，因此单独做多重检验调整。"""
    ch, mp, rows, E, se = c.ch, c.mp, [], {}, []
    b3 = cut_group(ch, ch["size"])
    for sig in c.D.runs:
        inner = qgroup(ch, ch[sig], 5, by=b3)
        for w in ("ew", "vw"):
            P = port(mp, np.where(b3 == 1, inner, np.nan), w)
            ls = long_short(P[[1.0, 5.0]])
            a = c.alpha(ls)
            E[(sig, w)], _ = a["e"], se.append(a["se"])
            lg, sh = c.alpha(c.excess(P)[5.0]), c.alpha(c.excess(P)[1.0])
            per = [c.alpha(ls.loc[x:y]) for x, y in c.t["periods"]]
            rows.append({"signal": sig, "label": c.lab[sig], "weight": w, **c.row(ls, ["CAPM", "FF3", "FF6", "CH3"]), "a_long": lg["alpha"] * PCT, "t_long": lg["t"], "a_short": sh["alpha"] * PCT, "t_short": sh["t"],
                         "a_pre": per[0]["alpha"] * PCT, "t_pre": per[0]["t"], "a_post": per[1]["alpha"] * PCT, "t_post": per[1]["t"], **{k: v for k, v in perf(ls).items() if k == "sharpe"}})
    t = pd.DataFrame(rows)
    t["p_RW"] = S.romano_wolf(pd.DataFrame(E).reindex(c.months), np.array(se), c.t["bootstrap"], c.t["block"], c.t["seed"])
    save_table(t, c.cfg, "small_cap")

    ret_w = mp["ret"].clip(mp.groupby("month")["ret"].transform(lambda s: s.quantile(.01)), mp.groupby("month")["ret"].transform(lambda s: s.quantile(.99)))
    mp_w, jul = mp.assign(ret=ret_w), _realign(c, 7)
    rows = []
    for sig in c.core:
        s = ch[sig].to_numpy()
        inner = lambda mask=None, n=5, by=b3: qgroup(ch, s, n, by=by, mask=mask)
        def spec(n=5, mask=None, cuts=(.3, .7)):
            b = cut_group(ch, ch["size"], cuts)
            return np.where(b == 1, qgroup(ch, s, n, by=b, mask=mask), np.nan), n
        var = {"基准（最小 30%，5 组）": (mp, spec()),
               "最小 20%": (mp, spec(cuts=(.2, .7))), "最小 40%": (mp, spec(cuts=(.4, .7))), "最小 50%": (mp, spec(cuts=(.5,))),
               "最小 30%，3 组": (mp, spec(3)), "最小 30%，10 组": (mp, spec(10)),
               "独立分组": (mp, (np.where(b3 == 1, qgroup(ch, s, 5), np.nan), 5)),
               "剔除 ST（形成时已知）": (mp, spec(mask=~ch["st"].astype(bool).to_numpy())),
               "剔除借壳后公司（形成时已知）": (mp, spec(mask=~ch["post"].astype(bool).to_numpy())),
               "剔除形成前月停牌（形成时已知）": (mp, spec(mask=ch["traded_last"].astype(bool).to_numpy())),
               "剔除所有曾被借壳公司（事后信息）": (mp, spec(mask=ch["event_year"].isna().to_numpy())),
               "月收益按月缩尾 1/99": (mp_w, spec()), "7 月才开始使用信号": (jul, spec())}
        for name, (m, (g, n)) in var.items():
            for w in ("ew", "vw"):
                P = port(m, g, w)
                ls = long_short(P[[1.0, float(n)]]).loc[lambda x: x.index <= c.months[-1]]
                rows.append({"signal": sig, "variant": name, "weight": w, "T": len(ls), **c.row(ls, ["FF3", "FF6", "CH3"])})
        P = port(mp, np.where(b3 == 1, inner(), np.nan), "ew")
        ls = long_short(P[[1.0, 5.0]])
        ex = ls[ls.index.map(lambda x: x.year - (x.month < 5)) != 2015]
        rows.append({"signal": sig, "variant": "剔除 2015 形成年", "weight": "ew", "T": len(ex), **c.row(ex, ["FF3", "FF6", "CH3"])})
    save_table(pd.DataFrame(rows), c.cfg, "small_cap_robust")
    return t


# ---------------------------------------------------------------- 借壳前后对比

def post_shell(c):
    ch, long = c.ch, c.D.long
    ev = pd.read_csv(c.cfg["paths"]["data"] / "events.csv", dtype={"Stkcd": str})
    ev["Stkcd"] = ev["Stkcd"].str.zfill(6)
    ev["em"] = pd.PeriodIndex(pd.to_datetime(ev["event_date"], errors="coerce"), freq="M")
    ev = ev.dropna(subset=["em"])
    # 事件研究：相对事件月的规模调整收益。月度收益从 2012-01 开始，只用此后完成的事件；
    # 事件前的窗口对早期事件观测不到，因此不同 τ 的样本构成不完全相同（表中 count 列）
    ev = ev[ev["em"] >= pd.Period("2013-01", "M")]
    m = long[~long["susp"].astype(bool)].merge(ev[["Stkcd", "em"]], on="Stkcd")
    m["tau"] = (m["month"].dt.year - m["em"].dt.year) * 12 + m["month"].dt.month - m["em"].dt.month
    m = m[m["tau"].between(-36, 36)]
    z = m.groupby("tau")["ret_sa"].agg(["mean", "std", "count"])
    z["t"] = z["mean"] / z["std"] * np.sqrt(z["count"])
    z["caar"] = z["mean"].cumsum()
    car = z.reset_index().assign(mean=lambda x: x["mean"] * PCT, caar=lambda x: x["caar"] * PCT)
    save_table(car[["tau", "mean", "t", "count", "caar"]], c.cfg, "event_caar", {"count": ",.0f"})

    # 事件年前后的壳概率排名
    rows = []
    h = ch[ch["event_year"].notna()].assign(tau=lambda x: x["year"] - x["event_year"])
    h = h[h["tau"].between(-4, 6)]
    for sig in c.core:
        rows.append(h.groupby("tau")[sig].agg(["mean", "count"]).reset_index().assign(signal=sig))
    tr = pd.concat(rows)
    save_table(tr, c.cfg, "event_rank", {"count": ",.0f"})

    # 借壳后 / 从未被借壳 / 高壳概率的组合对照
    rows = []
    pg = np.where(ch["post"].astype(bool), 1.0, np.nan)
    d10 = qgroup(ch, ch[c.main], 10)
    size_d = cut_group(ch, ch["size"], (.1, .2, .3, .4, .5, .6, .7, .8, .9))
    never = np.where(ch["event_year"].isna(), 1.0, np.nan)
    for w in ("ew", "vw"):
        for name, g in {"借壳后公司": pg, "从未被借壳公司（事后信息）": never, f"{c.main} 前 10%": np.where(d10 == 10, 1.0, np.nan), f"{c.main} 后 10%": np.where(d10 == 1, 1.0, np.nan)}.items():
            P = c.excess(port(c.mp, g, w)).iloc[:, 0].dropna()
            rows.append({"portfolio": name, "weight": w, "T": len(P), "avg_n": np.nansum(~np.isnan(g)) / ch["year"].nunique(), **c.row(P, ["CAPM", "FF3", "FF6", "CH3"])})
        ls = (port(c.mp, pg, w).iloc[:, 0] - port(c.mp, never, w).iloc[:, 0]).dropna()
        rows.append({"portfolio": "借壳后 − 从未被借壳", "weight": w, "T": len(ls), **c.row(ls, ["CAPM", "FF3", "FF6", "CH3"])})
        # 规模配对：借壳后公司的规模调整收益
        ad = port(c.mp, pg, w, "ret_sa").iloc[:, 0].dropna()
        rows.append({"portfolio": "借壳后公司（规模调整收益）", "weight": w, "T": len(ad), **c.row(ad, ["CAPM", "FF3", "FF6", "CH3"])})
    save_table(pd.DataFrame(rows), c.cfg, "post_shell_portfolios", {"avg_n": ".0f"})
    # 概括：借壳后样本落在壳概率哪里
    rows = []
    for sig in c.core:
        p = ch[ch["post"].astype(bool)]
        rows.append({"signal": sig, "post_mean_rank": p[sig].mean(), "post_share_top10": (p[sig] > ch.loc[p.index, "year"].map(ch.groupby("year")[sig].quantile(.9))).mean(), "post_share_bottom10": (p[sig] < ch.loc[p.index, "year"].map(ch.groupby("year")[sig].quantile(.1))).mean()})
    save_table(pd.DataFrame(rows), c.cfg, "post_shell_rank")
    return car, tr


# ---------------------------------------------------------------- 把壳概率作为因子：模型比较

def factor_models(c):
    """SHELL 因子（2×3，流通市值加权）放进因子模型，能不能降低定价误差？测试资产：规模×BM、规模×EP 各 25 组。"""
    ch, rows, spans = c.ch, [], []
    assets = {}
    for name, v in {"规模×BM 25": "lBM", "规模×EP 25": "EP"}.items():
        a, b = qgroup(ch, ch["size"], 5), qgroup(ch, ch[v], 5)
        assets[name] = c.excess(port(c.mp, (a - 1) * 5 + b, "vw")).reindex(columns=[float(i) for i in range(1, 26)])
    sz = cut_group(ch, ch["size"], (.5,))
    for sig in c.core:
        g6 = (sz - 1) * 3 + cut_group(ch, ch[sig], by=sz)
        for w in ("vw", "ew"):
            P6 = port(c.mp, g6, w)
            sh = ((P6[3.0] + P6[6.0] - P6[1.0] - P6[4.0]) / 2).rename("SHELL")
            spans.append({"signal": sig, "weight": w, **c.row(sh), "sharpe": perf(sh)["sharpe"], **{f"b_{k}": v for k, v in c.alpha(sh)["betas"].items()}, **{f"tb_{k}": v for k, v in c.alpha(sh)["tb"].items()}})
            if w == "ew":
                continue
            for asset, R in assets.items():
                for base in ("FF3", "FF5", "FF6", "CH3"):
                    for add in (False, True):
                        F = pd.concat([c.fm(base), sh], axis=1) if add else c.fm(base)
                        G = S.grs(R, F)
                        rows.append({"signal": sig, "assets": asset, "model": base + ("+SHELL" if add else ""), **G, "mean_abs_alpha": G["mean_abs_alpha"] * PCT})
    save_table(pd.DataFrame(rows), c.cfg, "factor_model_compare", {"p": ".4f", "sr2_factors": ".3f"})
    save_table(pd.DataFrame(spans), c.cfg, "shell_factor")


# ---------------------------------------------------------------- 图

def figs(c, g, sel, dec, cons, y, car, tr, dcy, ds, mc):
    cfg = c.cfg
    # 1. 十分组 alpha（等权 / 流通市值）小多图
    n = len(c.core)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.8), sharey="row", sharex=True)
    for j, sig in enumerate(c.core):
        for i, w in enumerate(("ew", "vw")):
            d = dec[(dec["signal"] == sig) & (dec["weight"] == w)]
            axes[i, j].bar(d["decile"], d["a_FF6"], color=SERIES[0], width=.72)
            axes[i, j].axhline(0, color=MUTED, lw=.8)
            if i == 0:
                axes[i, j].set_title(c.lab[sig].replace(" · ", "\n"), fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(f"FF6 alpha（%/月）\n{'等权' if w == 'ew' else '流通市值加权'}")
            axes[i, j].set_xticks([1, 5, 10])
    fig.supxlabel("壳概率十分组（1 = 最低，10 = 最高）", fontsize=10, color=INK2)
    fig.tight_layout()
    save_fig(fig, cfg, "fig_deciles")

    # 2. 累计多空收益
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, w, tl in zip(axes, ("ew", "vw"), ("等权", "流通市值加权")):
        for i, sig in enumerate(c.core):
            ls = long_short(port(c.mp, qgroup(c.ch, c.ch[sig], 10), w))
            cum = (np.log1p(ls)).cumsum()
            ax.plot(cum.index.to_timestamp(), cum.to_numpy() * 100, color=SERIES[i % len(SERIES)], lw=1.6, ls="--" if sig == "ensemble" else "-", label=c.lab[sig])
        ax.axhline(0, color=MUTED, lw=.8)
        ax.axvline(pd.Timestamp("2016-09-01"), color=MUTED, lw=.8, ls=":")
        ax.set_title(f"D10 − D1，{tl}")
    axes[0].set_ylabel("累计对数收益（%）")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
    axes[0].annotate("2016-09 重组新规", (pd.Timestamp("2016-09-01"), axes[0].get_ylim()[1]), fontsize=8, color=INK2, ha="left", va="top", xytext=(4, 0), textcoords="offset points")
    fig.tight_layout()
    save_fig(fig, cfg, "fig_cum")

    # 3. 全部信号的 t 值热图（十分组）
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("d", DIVERGING)
    ms = c.t["factor_models"]
    for ax, w, tl in zip(axes, ("ew", "vw"), ("等权", "流通市值加权")):
        h = g[(g["weight"] == w) & (g["n"] == 10)].set_index("signal").reindex(c.D.runs)
        cols = ["t_mean"] + [f"t_{k}" for k in ms]
        m = h[cols].to_numpy(float)
        ax.imshow(m, cmap=cmap, vmin=-4, vmax=4, aspect="auto")
        ax.set_xticks(range(len(cols)), ["收益均值"] + ms)
        ax.set_yticks(range(len(h)), [c.lab[r] for r in h.index], fontsize=8)
        ax.grid(False)
        ax.set_title(f"t 值，D10 − D1，{tl}")
        for a in range(m.shape[0]):
            for b in range(m.shape[1]):
                ax.text(b, a, f"{m[a, b]:.1f}", ha="center", va="center", fontsize=7, color="white" if abs(m[a, b]) > 2.8 else INK2)
    fig.tight_layout()
    save_fig(fig, cfg, "fig_heat")

    # 4. 全部信号的 alpha 森林图
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    for ax, w, tl in zip(axes, ("ew", "vw"), ("等权", "流通市值加权")):
        h = g[(g["weight"] == w) & (g["n"] == 10)].set_index("signal").reindex(c.D.runs[::-1])
        pos = np.arange(len(h))
        ax.errorbar(h["a_FF6"], pos, xerr=1.96 * h["a_FF6"] / h["t_FF6"].where(h["t_FF6"] != 0).abs().where(lambda s: s > 0) * np.sign(h["a_FF6"]), fmt="o", color=SERIES[0], ms=4, lw=1, capsize=0)
        ax.axvline(0, color=MUTED, lw=.8)
        ax.set_yticks(pos, [c.lab[r] for r in h.index], fontsize=8)
        ax.set_title(f"FF6 alpha 与 95% 置信区间（%/月），{tl}")
    fig.tight_layout()
    save_fig(fig, cfg, "fig_forest")

    # 5. 分年多空收益（主模型、组合）
    sigs = [c.main, "ensemble"] if "ensemble" in c.core else [c.main]
    fig, axes = plt.subplots(len(sigs), 2, figsize=(11, 3 * len(sigs)), sharex=True, sharey=True, squeeze=False)
    for i, sig in enumerate(sigs):
        for j, (w, tl) in enumerate(zip(("ew", "vw"), ("等权", "流通市值加权"))):
            d = y[(y["signal"] == sig) & (y["weight"] == w)]
            axes[i, j].bar(d["year"], d["ret"], color=np.where(d["ret"] >= 0, SERIES[0], SERIES[1]), width=.72)
            axes[i, j].axhline(0, color=MUTED, lw=.8)
            axes[i, j].set_title(f"{c.lab[sig]}，{tl}", fontsize=10)
            if j == 0:
                axes[i, j].set_ylabel("持有年累计收益（%）")
    fig.supxlabel("形成年（t 年 5 月至 t+1 年 4 月）", fontsize=10, color=INK2)
    fig.tight_layout()
    save_fig(fig, cfg, "fig_by_year")

    # 6. 排序能力与定价能力
    if mc is not None and len(mc):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        fam = {k: i for i, k in enumerate(dict.fromkeys(r.rsplit("_s", 1)[0] for r in mc["run"]))}
        for ax, w, tl in zip(axes, ("ew", "vw"), ("等权", "流通市值加权")):
            h = sel[sel["weight"] == w].merge(mc, left_on="signal", right_on="run")
            for f, i in fam.items():
                d = h[h["run"].str.rsplit("_s", n=1).str[0] == f]
                ax.scatter(d["AUC_new_all"], d["t_FF6"], color=SERIES[i % len(SERIES)], s=42, edgecolor="white", linewidth=1.2, label=f, zorder=3)
            ax.axhline(0, color=MUTED, lw=.8)
            ax.set_xlabel("样本外年内 AUC")
            ax.set_title(f"预测能力与 FF6 alpha t 值，{tl}")
        axes[0].set_ylabel("FF6 alpha t 值")
        axes[1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
        fig.tight_layout()
        save_fig(fig, cfg, "fig_auc_alpha")

    # 8. 规模 / EP / 换手分桶内的壳概率 L-S
    d = ds[(ds["how"] == "dep") & (ds["signal"] == c.main) & (ds["dim"].isin(["size", "EP", "lBM", "turn"])) & (ds["bucket"].isin(["1.0", "2.0", "3.0"]))]
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True, sharey=True)
    for j, dim in enumerate(["size", "EP", "lBM", "turn"]):
        for i, w in enumerate(("ew", "vw")):
            h = d[(d["dim"] == dim) & (d["weight"] == w)]
            axes[i, j].bar(["低", "中", "高"], h["a_FF6"], color=SERIES[0], width=.6)
            for x, (v, t) in enumerate(zip(h["a_FF6"], h["t_FF6"])):
                axes[i, j].text(x, v, f"t={t:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK2)
            axes[i, j].axhline(0, color=MUTED, lw=.8)
            if i == 0:
                axes[i, j].set_title({"size": "流通市值", "EP": "E/P", "lBM": "账面市值比", "turn": "换手率"}[dim])
            if j == 0:
                axes[i, j].set_ylabel(f"Q5 − Q1 FF6 alpha（%/月）\n{'等权' if w == 'ew' else '流通市值加权'}")
    fig.tight_layout()
    save_fig(fig, cfg, "fig_double_sort")

    # 10. 小市值内的壳概率多空
    b3 = cut_group(c.ch, c.ch["size"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, w, tl in zip(axes, ("ew", "vw"), ("等权", "流通市值加权")):
        for i, sig in enumerate(c.core):
            ls = long_short(port(c.mp, np.where(b3 == 1, qgroup(c.ch, c.ch[sig], 5, by=b3), np.nan), w)[[1.0, 5.0]])
            cum = np.log1p(ls).cumsum()
            ax.plot(cum.index.to_timestamp(), cum.to_numpy() * 100, color=SERIES[i % len(SERIES)], lw=1.6, ls="--" if sig == "ensemble" else "-", label=c.lab[sig])
        ax.axhline(0, color=MUTED, lw=.8)
        ax.axvline(pd.Timestamp("2016-09-01"), color=MUTED, lw=.8, ls=":")
        ax.set_title(f"最小 30% 公司内 Q5 − Q1，{tl}")
    axes[0].set_ylabel("累计对数收益（%）")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
    fig.tight_layout()
    save_fig(fig, cfg, "fig_small_cap")

    # 9. 信号衰减
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, w, tl in zip(axes, ("ew", "vw"), ("等权", "流通市值加权")):
        for i, sig in enumerate(c.core):
            h = dcy[(dcy["signal"] == sig) & (dcy["weight"] == w)]
            ax.plot(h["k"], h["cum"], color=SERIES[i % len(SERIES)], lw=1.6, ls="--" if sig == "ensemble" else "-", label=c.lab[sig])
        ax.axvline(12, color=MUTED, lw=.8, ls=":")
        ax.axhline(0, color=MUTED, lw=.8)
        ax.set_title(f"形成后累计 D10 − D1 收益，{tl}")
        ax.set_xlabel("形成后月数")
    axes[0].set_ylabel("累计收益（%）")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
    fig.tight_layout()
    save_fig(fig, cfg, "fig_decay")

    # 7. 事件研究
    if car is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(car["tau"], car["caar"], color=SERIES[0])
    axes[0].axvline(0, color=MUTED, lw=.8, ls=":")
    axes[0].axhline(0, color=MUTED, lw=.8)
    axes[0].set_title("借壳事件前后累计规模调整收益（CAAR）")
    axes[0].set_xlabel("相对借壳完成月份（月）")
    axes[0].set_ylabel("CAAR（%）")
    for i, sig in enumerate(c.core):
        d = tr[tr["signal"] == sig]
        axes[1].plot(d["tau"], d["mean"], color=SERIES[i % len(SERIES)], ls="--" if sig == "ensemble" else "-", lw=1.6, marker="o", ms=4, label=c.lab[sig])
    axes[1].axvline(0, color=MUTED, lw=.8, ls=":")
    axes[1].axhline(.5, color=MUTED, lw=.8)
    axes[1].set_title("壳概率年内排名：事件前后")
    axes[1].set_xlabel("相对借壳年份（年）")
    axes[1].set_ylabel("平均年内百分位排名")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, .5), fontsize=8)
    fig.tight_layout()
    save_fig(fig, cfg, "fig_event")


def run(cfg):
    import time
    t0 = time.perf_counter()
    step = lambda s: print(f"  [{time.perf_counter() - t0:6.0f}s] {s}", flush=True)
    D = build(cfg)
    c = Ctx(cfg, D)
    print(f"信号 {len(D.runs)} 个，公司-年 {len(D.chars):,}，股票-月 {len(D.mp):,}，{len(D.F)} 个月", flush=True)
    qa(c); step("数据核对")
    mc = model_compare(c); step("新旧预测对比")
    g = grid(c); step("全部信号主表")
    sel = selection(c, g); step("选择打分")
    cons = construction(c); step("组合构建方式")
    dec = deciles(c); step("十分组")
    ds = double_sorts(c); step("双重分组")
    neutralized(c); step("中性化")
    composite(c); step("联合因子")
    cross_section(c); step("横截面回归")
    _, y = regimes(c); step("时变")
    dcy = decay(c); step("衰减")
    robustness(c); step("稳健性")
    legacy_bridge(c); step("旧口径桥接")
    small_cap(c); step("小市值内的溢价")
    car, tr = post_shell(c) if D.chars["post"].any() else (None, None); step("借壳前后")  # 旧样本没有借壳后公司
    factor_models(c); step("因子模型比较")
    figs(c, g, sel, dec, cons, y, car, tr, dcy, ds, mc); step("图")
