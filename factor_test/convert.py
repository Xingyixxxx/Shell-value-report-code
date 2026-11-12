"""把检验要用的 CSMAR dta 转成 csv，只保留需要的字段和市场。"""
import pandas as pd


def run(cfg):
    src, out, s = cfg["paths"]["src_download"], cfg["paths"]["data"], cfg["sample"]
    read = lambda rel, **kw: pd.read_stata(src / rel, convert_categoricals=False, **kw)

    r = read("Month_Stock_return/TRD_Mnth.dta", columns=["Stkcd", "Trdmnt", "Mretwd", "Msmvosd", "Msmvttl", "Ndaytrd", "Markettype"])
    r[r["Markettype"].isin(s["markets"])].to_csv(out / "stock_month.csv", index=False)

    t = read("Turnover/LIQ_TOVER_M.dta", columns=["Stkcd", "Trdmnt", "ToverOsM", "Status", "MarketType"])
    t[t["MarketType"].isin(s["markets"])].drop(columns="MarketType").to_csv(out / "turnover.csv", index=False)

    # 财务指标只取年报；账面市值比 F101001A = 资产总计 / A 股市值（CSMAR 定义，分子是总资产）
    annual = lambda d: d[d["Accper"].str.endswith("12-31")].assign(fy=lambda x: x["Accper"].str[:4].astype(int))
    b = annual(read("EP_BM/FI_T10.dta", columns=["Stkcd", "Accper", "F100103C", "F101001A"]))
    b.rename(columns={"F100103C": "PE", "F101001A": "BM"}).drop(columns="Accper").to_csv(out / "pe_bm.csv", index=False)
    g = annual(read("GP/FI_T5.dta", columns=["Stkcd", "Accper", "Typrep", "F053301C"]))
    g[g["Typrep"] == "A"].rename(columns={"F053301C": "GP"})[["Stkcd", "fy", "GP"]].to_csv(out / "gp.csv", index=False)
    a = annual(read("IA/CSR_Finidx.dta", columns=["Stkcd", "Accper", "A100000"]))
    a.rename(columns={"A100000": "assets"})[["Stkcd", "fy", "assets"]].to_csv(out / "assets.csv", index=False)

    # P9706：沪深 A 股（不含创业板、科创板），与主板样本最接近；Portfolios = 1 为 2×3 划分
    f = read("Factor_5/STK_MKT_FIVEFACMONTH.dta")
    f = f[(f["MarkettypeID"] == "P9706") & (f["Portfolios"] == 1)]
    f.drop(columns=["MarkettypeID", "Portfolios"]).rename(columns={"TradingMonth": "Trdmnt"}).to_csv(out / "ff5.csv", index=False)

    # 动量因子取前后 30%、12 个月形成期（Carhart 常用口径），1 个月形成期作短期反转
    m = read("mov/STK_MKT_MOMENTUM.dta")
    m = m[(m["MarkettypeID"] == "P9706") & (m["StockClass"] == 0) & (m["Quantile"] == "30%") & m["FormationPeriod"].isin([1, 12])]
    m[["TradingMonth", "FormationPeriod", "MomRe1", "MomRe2", "MomRe3"]].rename(columns={"TradingMonth": "Trdmnt"}).to_csv(out / "momentum.csv", index=False)

    e = pd.read_stata(cfg["paths"]["src_events"], convert_categoricals=False)
    e = e[e["是否借壳上市"] == "是"].rename(columns={"借壳上市日期": "event_date", "证券简称": "name"})
    e[["Stkcd", "name", "event_date"]].to_csv(out / "events.csv", index=False)
    # 原论文的壳概率（2025 年版本），只用于旧口径的桥接
    old = pd.read_stata(src.parent / "data_processed/Shell_prob.dta", convert_categoricals=False)
    old.rename(columns={"y_pred": "paper_prob"})[["Stkcd", "year", "paper_prob"]].to_csv(out / "paper_shell_prob.csv", index=False)
    for f in sorted(out.glob("*.csv")):
        print(f"  {f.name}: {sum(1 for _ in open(f, encoding='utf-8')) - 1:,} 行")
