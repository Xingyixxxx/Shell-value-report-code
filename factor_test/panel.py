"""数据准备：壳概率信号（全部运行 + 等权组合）、股票-月面板、公司-年特征、因子。"""
import tomllib
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def load_config(outputs=None, tag=None) -> dict:
    cfg = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    p = cfg["paths"]
    if outputs:
        p["outputs"] = outputs
    if tag:
        p["tables"], p["figures"] = (f"results/factor_{tag}/{k}" for k in ("tables", "figures"))
    cfg["paths"] = {k: ROOT / v for k, v in p.items()}
    for k in ("data", "tables", "figures"):
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    cfg["tag"] = tag
    return cfg


def run_names():
    """运行名称与标签，顺序与中文标签都取自项目主配置。"""
    from shellpred.config import load_config as main_config
    return {r: info["label"] for r, info in main_config()["runs"].items()}


def read_signals(root) -> dict:
    """各运行的样本外预测（test）。"""
    out = {}
    for run in run_names():
        f = root / run / "predictions.csv.gz"
        if f.exists():
            d = pd.read_csv(f, dtype={"Stkcd": str}, usecols=["Stkcd", "time", "Y", "sample", "p"])
            d = d[d["sample"] == "test"].drop(columns="sample")
            out[run] = d.assign(Stkcd=d["Stkcd"].str.zfill(6), year=d["time"]).drop(columns="time")
    return out


def wide_signals(cfg, tests) -> tuple[pd.DataFrame, list[str]]:
    """年内百分位排名的宽表（概率水平跨年不可比，见主报告）；等权组合 = 成员排名的平均。"""
    key = ["Stkcd", "year"]
    frames = [d.assign(**{r: d.groupby("year")["p"].rank(pct=True)})[key + [r]] for r, d in tests.items()]
    w = reduce(lambda a, b: a.merge(b, on=key, how="outer"), frames)
    runs = list(tests)
    members = cfg["signal"]["ensemble"]
    if all(m in tests for m in members):
        w["ensemble"] = w[members].mean(axis=1)
        runs.append("ensemble")
    y = next(iter(tests.values()))[key + ["Y"]]
    return w.merge(y, on=key, how="left"), runs


def stock_months(cfg) -> pd.DataFrame:
    """主板股票-月长表。滞后变量都在完整月度网格上计算，停牌月不算观测但市值沿用最后一个已知值。"""
    data = cfg["paths"]["data"]
    r = pd.read_csv(data / "stock_month.csv", dtype={"Stkcd": str})
    t = pd.read_csv(data / "turnover.csv", dtype={"Stkcd": str})
    r["month"], t["month"] = pd.PeriodIndex(r["Trdmnt"], freq="M"), pd.PeriodIndex(t["Trdmnt"], freq="M")
    months = pd.period_range(r["month"].min(), r["month"].max(), freq="M")
    piv = lambda d, c: d.pivot_table(index="month", columns="Stkcd", values=c, aggfunc="last").reindex(months)
    ret, mvf, mvt = (piv(r, c) for c in ("Mretwd", "Msmvosd", "Msmvttl"))
    tov = piv(t, "ToverOsM").reindex(columns=ret.columns)
    status = t.pivot(index="month", columns="Stkcd", values="Status").reindex(index=months, columns=ret.columns)
    # 首次与最后一次有收益之间的缺失视为停牌，保留（标记 susp）以便做"停牌月收益记 0"的稳健性
    live = ret.notna().cumsum().gt(0) & ret[::-1].notna().cumsum()[::-1].gt(0)
    susp = live & ret.isna()
    lr = np.log1p(ret.where(ret > -1))
    wide = {"ret": ret.where(live), "susp": susp,
            "mvf": mvf.ffill().shift(1), "mvt": mvt.ffill().shift(1), "mvf_now": mvf,
            "rev": ret.shift(1), "mom": np.expm1(lr.shift(2).rolling(11, min_periods=6).sum()),
            "turn": tov.shift(1).rolling(3, min_periods=1).mean(),
            "st": status.ffill().shift(1).isin(["B", "D", "T"]),
            "traded_last": ret.shift(1).notna()}
    long = pd.concat({k: v.stack(future_stack=True) for k, v in wide.items()}, axis=1)
    long.index.names = ["month", "Stkcd"]
    long = long[long["susp"].astype(bool) | long["ret"].notna()].reset_index()
    # 同月、同流通市值十分组的等权收益作为规模基准
    has = long[~long["susp"].astype(bool) & long["mvf"].notna()]
    dec = has.groupby("month")["mvf"].rank(pct=True).mul(10).apply(np.ceil).clip(1, 10)
    bench = has["ret"].groupby([has["month"], dec]).transform("mean")
    long["ret_sa"] = (has["ret"] - bench).reindex(long.index)
    long["y"] = long["month"].dt.year - (long["month"].dt.month < cfg["signal"]["apply_month"]).astype(int)
    return long


def fundamentals(cfg) -> pd.DataFrame:
    """t 年组合使用 t-1 财年的特征。PE 缺失或非正即亏损（CSMAR 分母非正记 NULL）：EP 记 0、亏损指示记 1。"""
    data = cfg["paths"]["data"]
    rd = lambda n: pd.read_csv(data / n, dtype={"Stkcd": str})
    pb, gp, a = rd("pe_bm.csv"), rd("gp.csv"), rd("assets.csv")
    a = a.sort_values(["Stkcd", "fy"])
    prev = a.assign(fy=a["fy"] + 1).rename(columns={"assets": "a0"})
    a = a.merge(prev, on=["Stkcd", "fy"], how="left").assign(IA=lambda x: x["assets"] / x["a0"] - 1)
    f = pb.merge(gp, on=["Stkcd", "fy"], how="outer").merge(a[["Stkcd", "fy", "IA"]], on=["Stkcd", "fy"], how="outer")
    f["loss"] = (~(f["PE"] > 0)).astype(float)
    f["EP"] = (1 / f["PE"]).where(f["PE"] > 0, 0.0)
    f["lBM"] = np.log(f["BM"].where(f["BM"] > 0))
    return f.assign(y=f["fy"] + 1)[["Stkcd", "y", "EP", "loss", "lBM", "GP", "IA"]]


def factors(cfg, long, fund) -> pd.DataFrame:
    """CSMAR 五因子、动量，加自构 CH3（Liu-Stambaugh-Yuan 2019 的简化版）。
    无风险利率：CSMAR 五因子使用央行三个月定存基准利率，这里按调息表月度化。"""
    data, s = cfg["paths"]["data"], cfg["sample"]
    idx = lambda d: d.assign(month=pd.PeriodIndex(d["Trdmnt"], freq="M")).set_index("month")
    f = idx(pd.read_csv(data / "ff5.csv"))
    F = f[["RiskPremium1", "SMB1", "HML1", "RMW1", "CMA1"]].set_axis(["MKT", "SMB", "HML", "RMW", "CMA"], axis=1)
    m = idx(pd.read_csv(data / "momentum.csv"))
    F["UMD"] = m[m["FormationPeriod"] == 12]["MomRe1"]
    F["REV"] = m[m["FormationPeriod"] == 1]["MomRe1"]

    rates = [("2011-07", .0310), ("2012-06", .0285), ("2012-07", .0260), ("2014-11", .0235), ("2015-03", .0210),
             ("2015-05", .0185), ("2015-06", .0160), ("2015-08", .0135), ("2015-10", .0110)]
    rf = pd.Series({pd.Period(a, "M"): b for a, b in rates}).reindex(pd.period_range("2011-07", F.index.max(), freq="M")).ffill()
    F["rf"] = (rf / 12).reindex(F.index)

    # CH3：剔除上月总市值最小的 30%（壳价值污染），中位数分大小、EP 30/70 分价值，总市值加权，逐月调整
    d = long[~long["susp"].astype(bool) & long["mvt"].notna()].merge(fund[["Stkcd", "y", "EP"]], on=["Stkcd", "y"], how="left")
    d = d[d["mvt"] > d.groupby("month")["mvt"].transform(lambda x: x.quantile(.3))].dropna(subset=["EP"])
    d["big"] = d["mvt"] > d.groupby("month")["mvt"].transform("median")
    pct = d.groupby("month")["EP"].rank(pct=True)
    d["val"] = np.select([pct <= .3, pct > .7], [0, 2], 1)
    d["rw"] = d["ret"] * d["mvt"]
    g = d.groupby(["month", "big", "val"])
    P = (g["rw"].sum() / g["mvt"].sum()).unstack(["big", "val"])
    F["SMB_CH"] = P[False].mean(axis=1) - P[True].mean(axis=1)
    F["VMG"] = (P[(False, 2)] + P[(True, 2)] - P[(False, 0)] - P[(True, 0)]) / 2
    return F.loc[s["start"]:s["end"]]


FACTOR_MODELS = {"CAPM": ["MKT"], "FF3": ["MKT", "SMB", "HML"], "FF4": ["MKT", "SMB", "HML", "UMD"],
                 "FF5": ["MKT", "SMB", "HML", "RMW", "CMA"], "FF6": ["MKT", "SMB", "HML", "RMW", "CMA", "UMD"],
                 "CH3": ["MKT", "SMB_CH", "VMG"]}


@dataclass
class Data:
    chars: pd.DataFrame     # 公司-年（组合形成时），行号 cid 与 mp 对应
    mp: pd.DataFrame        # 组合持有期的股票-月（含停牌月，默认检验时剔除）
    long: pd.DataFrame      # 全部主板股票-月
    F: pd.DataFrame         # 因子
    runs: list              # 全部信号
    labels: dict
    tests: dict             # 各运行的原始预测


def build(cfg) -> Data:
    p, s = cfg["paths"], cfg["sample"]
    tests = read_signals(p["outputs"])
    sig, runs = wide_signals(cfg, tests)
    long = stock_months(cfg)
    fund = fundamentals(cfg)
    F = factors(cfg, long, fund)

    raw = pd.read_csv(p["raw"], dtype={"Stkcd": str}, usecols=["Stkcd", "time", "Y", "Ind", "Soe_1", "ST_1", "Lev_1", "Roe_3"])
    raw = raw.assign(Stkcd=raw["Stkcd"].str.zfill(6))
    first = raw[raw["Y"] == 1].groupby("Stkcd")["time"].min().rename("event_year")
    sig = sig.merge(raw.rename(columns={"time": "year"}).drop(columns="Y"), on=["Stkcd", "year"], how="left").merge(first, on="Stkcd", how="left")
    sig["post"] = sig["year"] > sig["event_year"]           # 已被借壳（形成时已知）
    sig["ind"] = sig["Ind"].str[:2].where(sig["Ind"].str[0] == "C", sig["Ind"].str[0])  # 制造业细到大类

    # 形成时点（t 年 5 月）的市值、换手、动量，及 t-1 财年财务指标；形成时没有市值的公司不进入组合
    apply_m = cfg["signal"]["apply_month"]
    start = long[long["month"].dt.month == apply_m].rename(columns={"y": "year"})
    ch = sig.merge(start[["Stkcd", "year", "mvf", "mvt", "turn", "rev", "mom", "traded_last", "st"]], on=["Stkcd", "year"])
    ch = ch.merge(fund.rename(columns={"y": "year"}), on=["Stkcd", "year"], how="left").dropna(subset=["mvf"])
    ch["size"] = np.log(ch["mvf"])
    ch = ch.reset_index(drop=True)
    ch["cid"] = np.arange(len(ch))

    h = long[(long["month"] >= pd.Period(s["start"], "M")) & (long["month"] <= pd.Period(s["end"], "M"))]
    mp = h.merge(ch[["Stkcd", "year", "cid"]].rename(columns={"year": "y"}), on=["Stkcd", "y"]).sort_values(["cid", "month"])
    mp["bh"] = (1 + mp["ret"].fillna(0)).groupby(mp["cid"]).cumprod().groupby(mp["cid"]).shift(1).fillna(1.0)
    mp["w_ew"], mp["w_vw"], mp["w_vwt"], mp["w_sqrt"], mp["w_bh"] = 1.0, mp["mvf"], mp["mvt"], np.sqrt(mp["mvf"]), mp["bh"]
    labels = run_names() | {"ensemble": "等权组合"}
    return Data(ch, mp.reset_index(drop=True), long, F, runs, labels, tests)
