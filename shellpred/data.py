import numpy as np
import pandas as pd

# 变量简称（用于表格和图）
VAR_LABELS = {
    "Y": "是否被借壳",
    "IPO_1": "IPO 通过率",
    "IPOnum_1": "IPO 上会家数",
    "Soe_1": "国有企业",
    "ST_1": "ST/*ST",
    "Size_3": "流通市值(对数)",
    "Op_3": "营业利润率",
    "Sales_3": "营业收入增长率",
    "reverrate_3": "流通股比例",
    "Insider_3": "高管持股比例",
    "Lev_3": "资产负债率",
    "Cash_3": "现金持有率",
    "Roe_3": "净资产收益率",
    "TOP1_3": "第一大股东持股(%)",
    "return_3": "年度超额收益率",
    "NegEq_1": "净资产为负",
}

# 证监会 2012 版行业分类门类
INDUSTRY_NAMES = {
    "A": "农、林、牧、渔业", "B": "采矿业", "C": "制造业",
    "D": "电力、热力、燃气及水生产和供应业", "E": "建筑业", "F": "批发和零售业",
    "G": "交通运输、仓储和邮政业", "H": "住宿和餐饮业", "I": "信息传输、软件和信息技术服务业",
    "J": "金融业", "K": "房地产业", "L": "租赁和商务服务业", "M": "科学研究和技术服务业",
    "N": "水利、环境和公共设施管理业", "O": "居民服务、修理和其他服务业", "P": "教育",
    "Q": "卫生和社会工作", "R": "文化、体育和娱乐业", "S": "综合",
}


def label(col: str) -> str:
    if col.startswith("ind_"):
        return f"{col} {INDUSTRY_NAMES[col[4:]]}"
    if col.startswith("miss_"):  # 逻辑回归的缺失指示变量
        return f"{col} {VAR_LABELS.get(col[5:], col[5:])}缺失"
    return f"{col} {VAR_LABELS[col]}" if col in VAR_LABELS else col


def load_raw(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg["paths"]["data"], dtype={"Stkcd": str})


def build_dataset(cfg, dropna: bool) -> tuple[pd.DataFrame, list[str]]:
    """返回建模数据和特征列。use_industry=True 时加行业哑变量；dropna=True 时删除任一特征缺失的样本。"""
    d = cfg["data"]
    feats = d["features"]
    raw = load_raw(cfg)
    df = raw[["Stkcd", "time", "Y", d["industry_col"], *[f for f in feats if f in raw]]].copy()

    # 上一年资产负债率 > 1 即净资产为负。此时净资产收益率无定义（CSMAR 记为缺失），
    # 而资不抵债的公司正是典型的壳，因此单独做成指示变量；Lev_1 缺失时保持缺失。
    if "NegEq_1" in feats:
        df["NegEq_1"] = (raw["Lev_1"] > 1).astype(float).where(raw["Lev_1"].notna())

    # IPO_1/IPOnum_1 由 Stata 的 L. 算子生成，公司在 t-1 年不在面板中时会缺失；
    # 它们是年度市场变量，按年份取唯一值补齐即可。
    for c in d["market_vars"]:
        df[c] = df.groupby("time")[c].transform("max")
    for c in d["zero_as_missing"]:
        df.loc[df[c] == 0, c] = np.nan

    df["industry"] = df.pop(d["industry_col"]).str[0]  # 行业门类，用于描述性统计
    dummies = []
    if d["use_industry"]:
        df = df.dropna(subset=["industry"])
        dm = pd.get_dummies(df["industry"], prefix="ind", dtype=float).drop(columns=f"ind_{d['industry_base']}")
        df, dummies = pd.concat([df, dm], axis=1), list(dm.columns)
    if dropna:
        df = df.dropna(subset=feats)
    return df.reset_index(drop=True), feats + dummies
