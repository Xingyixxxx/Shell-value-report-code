"""报告第二版（报告_v2.md 与 LaTeX 版）用的图表。部分图表取自第一版输出（先运行 uv run -m reports v1）。

    uv run -m shellpred.revision        # 先生成预测部分的新结果（r_auc_sig、r_auc_stage、r_l1_coef_sig）
    uv run -m factor_test revise        # 再生成第 5 章的新结果（r_stage_all、r_rank_ic、r_holding_month 等）
    uv run -m reports v2                # 输出到 results/report/v2/figures、results/report/v2/tables

表格约定：显著性只用星号（*、**、*** 分别为 10%、5%、1%），不再列 t 值和 p 值；每列较好的结果加粗。
附录中直接沿用第一版的表格保留括号内 t 值，便于核对。
"""
import shutil

import numpy as np
import pandas as pd

from shellpred.plot import MUTED, SERIES, plt

from .common import FF, FT, ROOT, RT, arrow, box, rd
V1F, V1T = ROOT / "results/report/v1/figures", ROOT / "results/report/v1/tables"
OUT_F, OUT_T = ROOT / "results/report/v2/figures", ROOT / "results/report/v2/tables"

MAIN = ["logit_s0", "logit_l1_s0", "lda_s0", "rf_s1", "xgb_s1", "mlp_s1", "ensemble"]
STAGE3 = ["宽松期", "严控期", "放松+注册制"]
W = {"ew": "等权", "vw": "流通市值加权"}
SIGLAB = {"logit_s0": "逻辑回归 L2", "logit_l1_s0": "逻辑回归 L1", "lda_s0": "LDA", "rf_s1": "随机森林",
          "xgb_s1": "XGBoost", "mlp_s1": "MLP", "ensemble": "等权组合"}

# 新编号 ← 来源
FIGS = {
    "fig1_1_ipo": V1F / "fig1_2_ipo.png", "fig1_2_sample_by_year": V1F / "fig1_1_sample_by_year.png",
    "fig3_1_rolling_tuning": V1F / "fig2_1_rolling_tuning.png", "fig3_2_imbalance": V1F / "fig2_2_imbalance.png",
    "fig4_1_auc_by_year": V1F / "fig4_3_auc_by_year.png", "fig4_2_shap_dependence": V1F / "fig4_7_shap_dependence.png",
    "fig5_1_stage": FF / "r_fig_stage.png", "fig5_2_holding_month": FF / "r_fig_holding_month.png",
    "figA_1_feature_box": V1F / "fig1_3_feature_box.png", "figA_2_corr": V1F / "figA_1_corr.png",
    "figB_1_grid_heatmap": V1F / "fig3_1_grid_heatmap.png", "figB_2_sampler_forest": V1F / "fig4_1_sampler_forest.png",
    "figB_3_forest_all": V1F / "fig4_2_forest_all.png", "figB_4_gains_lift": V1F / "fig4_4_gains_lift.png",
    "figB_5_calibration": V1F / "fig4_5_calibration.png", "figB_6_importance": V1F / "fig4_6_importance.png",
    "figB_7_roc_pr": V1F / "figA_2_roc_pr.png", "figB_8_patterns": V1F / "figA_3_patterns.png",
    "figC_1_deciles": V1F / "fig5_1_deciles.png", "figC_2_lookahead": V1F / "fig5_2_lookahead.png",
    "figC_3_cum_stages": V1F / "fig5_3_cum_stages.png", "figC_4_annual": V1F / "fig5_4_annual.png",
    "figC_5_small_cap": V1F / "fig5_5_small_cap.png", "figC_6_tstat_heat": V1F / "figA_4_tstat_heat.png",
    "figC_7_cum_all": V1F / "figA_5_cum_all.png", "figC_8_double_sort": V1F / "figA_6_double_sort.png",
    "figC_9_event": V1F / "figA_7_event.png",
}
# 附录中原样沿用的第一版表格
COPY = {"tabA_1_sample": "tab1_1_sample", "tabA_2_windows": "tab2_1_windows", "tabA_3_missing": "tabA_1_missing",
        "tabB_1_all_runs": "tabA_2_all_runs", "tabB_2_auc_diff": "tabA_3_auc_diff", "tabB_3_auc_by_year": "tabA_4_auc_by_year",
        "tabB_4_tuning": "tabA_5_tuning", "tabB_5_calibration": "tabA_6_calibration",
        "tabC_1_lookahead": "tab5_3_lookahead", "tabC_2_lookahead_cells": "tabA_8_lookahead_cells", "tabC_4_annual": "tab5_5_annual",
        "tabC_9_robustness": "tabA_11_robustness", "tabC_10_small_cap_robust": "tabA_12_small_cap_robust",
        "tabC_11_construction": "tabA_13_construction", "tabC_12_fama_macbeth": "tabA_14_fama_macbeth",
        "tabC_13_decile_chars": "tabA_15_decile_chars"}


def stars(p):
    if p is None or pd.isna(p):
        return ""
    return "\\*\\*\\*" if p < .01 else "\\*\\*" if p < .05 else "\\*" if p < .1 else ""


def f(v, d=2, p=None, bold=False):
    if v is None or pd.isna(v):
        return ""
    s = f"{v:.{d}f}" + stars(p)
    return f"**{s}**" if bold else s


def best(values, how="max"):
    """返回最优值所在位置的布尔数组（并列都加粗）；how = max / min / absmax。"""
    v = pd.Series(values, dtype=float)
    if v.notna().sum() == 0:
        return np.zeros(len(v), bool)
    key = v.abs() if how == "absmax" else v
    target = key.max() if how in ("max", "absmax") else key.min()
    return (key.round(6) == round(target, 6)).to_numpy()


def save(df, name, right_from=1):
    cols = list(df.columns)
    align = ["left"] * right_from + ["right"] * (len(cols) - right_from)
    df.to_csv(OUT_T / f"{name}.csv", index=False, encoding="utf-8-sig")
    md = df.astype(str).replace({"nan": ""}).to_markdown(index=False, colalign=align, disable_numparse=True)
    md = md.replace("/*ST", "/\\*ST")  # 避免 markdown 把 *ST 当作强调符号
    (OUT_T / f"{name}.md").write_text(md + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 第 1–3 章

def fig_flow():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13), ax.set_ylim(0, 3.6), ax.axis("off")
    blue, orange, grey = "#dce9f8", "#fbe3d6", "#efeeea"
    steps = [(0.1, "CSMAR 原始数据\n手工整理借壳名单", grey), (2.25, "公司-年度面板\n45,117 行 · 151 起借壳\n15 个滞后特征", grey),
             (4.55, "9 年滚动窗口\n2012–2023 逐年预测", blue), (6.75, "6 类模型 × 6 种平衡方法\n共 25 个运行\n前推调参", blue),
             (9.15, "样本外壳概率\n年内百分位排名", blue), (11.2, "按壳概率分组\n多空收益与 alpha", orange)]
    w = [1.9, 2.05, 1.95, 2.15, 1.8, 1.7]
    for (x, t, c), wi in zip(steps, w):
        box(ax, x, 1.5, wi, 1.3, t, c, fs=9.5)
    for (x, _, _), wi, (x2, _, _) in zip(steps, w, steps[1:]):
        arrow(ax, (x + wi + .02, 2.15), (x2 - .02, 2.15))
    for x0, x1, lab, col in [(0.1, 4.3, "数据", MUTED), (4.55, 10.95, "第一步：预测壳概率", SERIES[0]), (11.2, 12.9, "第二步：因子检验", SERIES[1])]:
        ax.plot([x0, x1], [3.15, 3.15], color=col, lw=2.5, solid_capstyle="round")
        ax.text((x0 + x1) / 2, 3.3, lab, ha="center", va="bottom", fontsize=10.5, color=col)
    box(ax, 6.75, 0.2, 2.15, 0.85, "评价：年内排序 AUC、Lift\nBrier、分政策阶段", "#ffffff", ec="#c3c2b7", fs=9)
    arrow(ax, (7.82, 1.48), (7.82, 1.07))
    box(ax, 10.75, 0.2, 2.15, 0.85, "Rank IC、分政策阶段\n持有月份、小市值内部", "#ffffff", ec="#c3c2b7", fs=9)
    arrow(ax, (12.05, 1.48), (12.05, 1.07))
    fig.savefig(OUT_F / "fig1_3_flow.png", bbox_inches="tight"), plt.close(fig)


def tables_data():
    d = rd(V1T, "tab1_2_desc")
    out = d[["类别", "变量", "含义"]].copy()
    for c in ["均值", "标准差", "P25", "中位数", "P75"]:
        out[c] = [f(v, 3) for v in d[c]]
    out["偏度"] = [f(v, 2) for v in d["偏度"]]
    out["缺失率(%)"] = [f(v, 1) for v in d["缺失率(%)"]]
    save(out, "tab2_1_desc", 3)

    g = rd(V1T, "tab1_3_group_compare")
    out = pd.DataFrame({"变量": g["变量"], "含义": g["含义"],
                        "均值：被借壳": [f(v, 3) for v in g["被借壳均值"]], "均值：未被借壳": [f(v, 3) for v in g["未被借壳均值"]],
                        "均值差": [f(v, 3, p) for v, p in zip(g["均值差"], g["t 检验 p"])],
                        "中位数：被借壳": [f(v, 3) for v in g["被借壳中位数"]], "中位数：未被借壳": [f(v, 3) for v in g["未被借壳中位数"]],
                        "秩和检验": [stars(p) or "—" for p in g["秩和检验 p"]]})
    save(out, "tab2_2_group_compare", 2)

    t = rd(V1T, "tab2_2_imbalance")
    t["正例占比(%)"] = [f(v, 2) for v in t["正例占比(%)"]]
    save(t, "tab3_1_imbalance", 2)


# ---------------------------------------------------------------- 第 4 章

def tables_pred():
    sig = rd(RT, "r_auc_sig").set_index("run")
    pooled = rd(RT, "model_pooled")
    pooled = pooled[pooled["样本"] == "全部样本"].set_index("模型")
    topk = rd(RT, "model_topk")
    topk = topk[topk["样本"] == "全部样本"].set_index("模型")
    samp = rd(RT, "model_sampler_compare")

    # 表 4-1：平衡方法
    rows = []
    for m, pre in [("逻辑回归 L2", "logit"), ("XGBoost", "xgb")]:
        s = samp[samp["模型"] == m].reset_index(drop=True)
        runs = [f"{pre}_s{i}" for i in range(6)]
        auc, dif = [sig.at[r, "AUC"] for r in runs], [sig.at[r, "diff_S0"] for r in runs]
        l5, l10, br = s["Lift@5%"], s["Lift@10%"], s["Brier"] * 1000
        bA, bD, b5, b10, bB = best(auc), best(dif), best(l5), best(l10), best(br, "min")
        for i, r in enumerate(runs):
            rows.append({"模型": m, "平衡方法": f"S{i} {s.at[i, '方法']}", "AUC": f(auc[i], 3, sig.at[r, "p_AUC"], bA[i]),
                         "相对 S0 差值": f(dif[i], 3, sig.at[r, "p_diff_S0"], bD[i]) if i else "—",
                         "Lift@5%": f(l5[i], 2, None, b5[i]), "Lift@10%": f(l10[i], 2, None, b10[i]),
                         "Brier(×10⁻³)": f(br[i], 2, None, bB[i]), "平均训练样本量": f"{s.at[i, '平均训练样本量']:,.0f}"})
    save(pd.DataFrame(rows), "tab4_1_sampler", 2)

    # 表 4-2：六个主设定与等权组合
    lab = [sig.at[r, "模型"] for r in MAIN]
    auc = [sig.at[r, "AUC"] for r in MAIN]
    dif = [sig.at[r, "diff_L2"] for r in MAIN]
    yr = [pooled.at[l, "逐年AUC均值"] for l in lab]
    pr = [pooled.at[l, "PR_AUC/正例率"] for l in lab]
    l5, l10 = [topk.at[l, "Lift@5%"] for l in lab], [topk.at[l, "Lift@10%"] for l in lab]
    br = [pooled.at[l, "Brier"] * 1000 if pd.notna(pooled.at[l, "Brier"]) else np.nan for l in lab]
    B = [best(x) for x in (auc, dif, yr, pr, l5, l10)] + [best(br, "min")]
    rows = []
    for i, r in enumerate(MAIN):
        rows.append({"模型": lab[i], "AUC": f(auc[i], 3, sig.at[r, "p_AUC"], B[0][i]),
                     "相对 L2 差值": f(dif[i], 3, sig.at[r, "p_diff_L2"], B[1][i]) if r != "logit_s0" else "—",
                     "逐年 AUC 均值": f(yr[i], 3, None, B[2][i]), "PR-AUC/正例率": f(pr[i], 2, None, B[3][i]),
                     "Lift@5%": f(l5[i], 2, None, B[4][i]), "Lift@10%": f(l10[i], 2, None, B[5][i]),
                     "Brier(×10⁻³)": f(br[i], 2, None, B[6][i]) if pd.notna(br[i]) else "—"})
    save(pd.DataFrame(rows), "tab4_2_main_models", 1)

    # 表 4-3：分政策阶段
    st = rd(RT, "r_auc_stage")
    stages = ["2012–2016", "2017–2019", "2020–2023", "全样本"]
    npos = st.groupby("阶段")["正例"].first()
    cols = {s: f"{s}（{npos[s]} 例）" if s != "全样本" else f"全样本（{npos[s]} 例）" for s in stages}
    rows = []
    for metric, title in [("AUC", "A. 年内排序 AUC"), ("Lift@10%", "B. Lift@10%"), ("Lift@5%", "C. Lift@5%")]:
        rows.append({"模型": f"**{title}**", **{cols[s]: "" for s in stages}})
        piv = st.pivot(index="run", columns="阶段", values=metric).reindex(MAIN)
        pp = st.pivot(index="run", columns="阶段", values="p_AUC").reindex(MAIN)
        bb = {s: best(piv[s]) for s in stages}
        for i, r in enumerate(MAIN):
            rows.append({"模型": SIGLAB[r] if r != "ensemble" else "等权组合",
                         **{cols[s]: f(piv.at[r, s], 3 if metric == "AUC" else 2, pp.at[r, s] if metric == "AUC" else None, bb[s][i]) for s in stages}})
    save(pd.DataFrame(rows), "tab4_3_stage", 1)

    # 表 4-4：特征贡献
    imp = rd(V1T, "tab4_3_importance")
    cs = rd(RT, "r_l1_coef_sig").set_index("变量")
    top = lambda s, how="max": set((s.abs() if how == "absmax" else s).nlargest(4).index)
    tS, tG, tC = top(imp["XGBoost SHAP 占比(%)"]), top(imp["XGBoost 增益(%)"]), top(imp["逻辑回归 L1 系数均值"], "absmax")
    out = pd.DataFrame({"变量": imp["变量"], "含义": imp["含义"],
                        "XGBoost SHAP 占比(%)": [f(v, 1, None, i in tS) for i, v in imp["XGBoost SHAP 占比(%)"].items()],
                        "XGBoost 增益(%)": [f(v, 1, None, i in tG) for i, v in imp["XGBoost 增益(%)"].items()],
                        "逻辑回归 L1 系数均值": [f(v, 3, cs.at[x, "p"], i in tC) for (i, v), x in zip(imp["逻辑回归 L1 系数均值"].items(), imp["变量"])]})
    save(out, "tab4_4_importance", 2)

    # 附录 B-6：L1 全部系数（含缺失指示变量）
    c = cs.reset_index()
    c = pd.DataFrame({"变量": c["变量"], "含义": c["含义"], "系数均值": [f(v, 3, p) for v, p in zip(c["系数均值"], c["p"])],
                      "自助标准误": [f(v, 3) for v in c["自助标准误"]]})
    save(c, "tabB_6_l1_coef", 2)


# ---------------------------------------------------------------- 第 5 章

def tables_factor():
    T = rd(FT, "r_stage_all")
    I = rd(FT, "r_rank_ic")
    sig3 = ["xgb_s1", "logit_l1_s0", "ensemble"]

    def cell(r, k, pk, d=2):
        v, p = r[k], r[pk]
        return f(v, d, p, bold=(p < .1 and v > 0))

    # 表 5-2：单变量十分组
    rows = []
    for s in sig3:
        ic = I[(I.signal == s) & (I["口径"] == "全市场") & (I["阶段"] == "全样本")].iloc[0]
        for w in ("ew", "vw"):
            r = T[(T.signal == s) & (T["构建"] == "十分组 D10−D1") & (T["权重"] == w) & (T["阶段"] == "全样本")].iloc[0]
            rows.append({"信号": SIGLAB[s], "权重": W[w], "D10−D1": cell(r, "mean", "p_mean"), "FF3 α": cell(r, "a_FF3", "p_FF3"),
                         "FF6 α": cell(r, "a_FF6", "p_FF6"), "CH3 α": cell(r, "a_CH3", "p_CH3"),
                         "Rank IC": cell(ic, "IC", "p", 3) if w == "ew" else ""})
    save(pd.DataFrame(rows), "tab5_2_univariate", 2)

    # 表 5-3 与附录 C-3：分政策阶段
    def stage_tab(s):
        rows = []
        cols = ["全样本", *STAGE3]
        for w in ("ew", "vw"):
            for name in ["十分组 D10−D1", "规模中性 Q5−Q1", "最小 30% 内 Q5−Q1"]:
                d = T[(T.signal == s) & (T["构建"] == name) & (T["权重"] == w)].set_index("阶段")
                for lab, k, pk in [("均值", "mean", "p_mean"), ("FF6 α", "a_FF6", "p_FF6")]:
                    rows.append({"构建方式": name, "权重": W[w], "指标": lab, **{c: cell(d.loc[c], k, pk) for c in cols}})
        for kind, nm in [("全市场", "全市场"), ("规模组内", "规模组内（5 组平均）"), ("最小 30% 内", "最小 30% 内")]:
            d = I[(I.signal == s) & (I["口径"] == kind)].set_index("阶段")
            rows.append({"构建方式": "Rank IC", "权重": "—", "指标": nm, **{c: cell(d.loc[c], "IC", "p", 3) for c in cols}})
        return pd.DataFrame(rows)

    save(stage_tab("xgb_s1"), "tab5_3_stage", 3)
    save(pd.concat([stage_tab(s).assign(信号=SIGLAB[s]) for s in ["logit_l1_s0", "ensemble"]])
         .pipe(lambda x: x[["信号", *[c for c in x.columns if c != "信号"]]]), "tabC_3_stage_other", 4)

    # 表 5-4 与附录 C-6：持有期各月
    H = rd(FT, "r_holding_month")
    order = [f"{m} 月" for m in [5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]]
    groups = ["第 1–3 个月（5–7 月）", "第 4–6 个月（8–10 月）", "第 7–9 个月（11–次年 1 月）", "第 10–12 个月（次年 2–4 月）", "前 6 个月", "后 6 个月"]

    def hold_tab(s, w="ew"):
        rows = []
        d = H[(H.signal == s) & (H["权重"] == w)]
        for g in order + groups:
            r = {"持有月份": g if g in groups else f"第 {order.index(g) + 1} 个月（{g}）"}
            for name in ["十分组 D10−D1", "规模中性 Q5−Q1", "最小 30% 内 Q5−Q1"]:
                x = d[(d["构建"] == name) & (d["月份"] == g)].iloc[0]
                r[name] = cell(x, "mean", "p")
            r["十分组胜率"] = f"{d[(d['构建'] == '十分组 D10−D1') & (d['月份'] == g)].iloc[0]['win']:.0%}"
            rows.append(r)
        return pd.DataFrame(rows)

    save(hold_tab("xgb_s1"), "tab5_4_holding_month", 1)
    save(pd.concat([hold_tab(s).assign(信号=SIGLAB[s]) for s in ["logit_l1_s0", "ensemble"]])
         .pipe(lambda x: x[["信号", *[c for c in x.columns if c != "信号"]]]), "tabC_6_holding_other", 2)

    # 表 5-5 与附录 C-7：小市值内部分阶段
    def small_tab(sigs):
        rows = []
        for s in sigs:
            ic = I[(I.signal == s) & (I["口径"] == "最小 30% 内") & (I["阶段"] == "全样本")].iloc[0]
            for w in ("ew", "vw"):
                d = T[(T.signal == s) & (T["构建"] == "最小 30% 内 Q5−Q1") & (T["权重"] == w)].set_index("阶段")
                rows.append({"信号": SIGLAB[s], "权重": W[w], "Q5−Q1": cell(d.loc["全样本"], "mean", "p_mean"),
                             "FF6 α 全样本": cell(d.loc["全样本"], "a_FF6", "p_FF6"),
                             **{f"FF6 α {c}": cell(d.loc[c], "a_FF6", "p_FF6") for c in STAGE3},
                             "Rank IC": cell(ic, "IC", "p", 3) if w == "ew" else ""})
        return pd.DataFrame(rows)

    save(small_tab(["logit_l1_s0", "xgb_s1", "mlp_s1", "ensemble"]), "tab5_5_small_cap", 2)
    save(small_tab(["logit_s0", "logit_l1_s0", "lda_s0", "rf_s1", "xgb_s1", "mlp_s1", "ensemble"]), "tabC_7_small_cap_all", 2)

    # 表 5-6 与附录 C-8：隐含壳溢价
    P = rd(FT, "r_implied")

    def imp_tab(sigs, with_sig):
        rows = []
        for s in sigs:
            for st in [*STAGE3, "全样本"]:
                x = P[(P.signal == s) & (P["阶段"] == st)].iloc[0]
                r = {"信号": SIGLAB[s]} if with_sig else {}
                r |= {"阶段": st, "形成年": x["形成年"], "Q5 借壳数/公司-年": f"{x['Q5 借壳数']}/{x['Q5 公司-年']}",
                      "Q5 借壳率(%)": f(x["Q5 借壳率(%)"], 2), "Q1 借壳率(%)": f(x["Q1 借壳率(%)"], 2),
                      "隐含壳溢价(%/月)": f(x["隐含月度壳溢价(%)"], 3), "实际 FF6 α(%/月)": f(x["实际 FF6 α(%)"], 2)}
                rows.append(r)
        return pd.DataFrame(rows)

    save(imp_tab(["xgb_s1"], False), "tab5_6_implied", 2)
    save(imp_tab(["logit_l1_s0", "xgb_s1", "mlp_s1", "ensemble"], True), "tabC_8_implied_all", 3)

    # 表 5-7：非线性模型相对逻辑回归 L1 的增量
    inc = rd(FT, "r_increment")
    inc = inc[(inc["权重"] == "ew") & inc.signal.isin(["mlp_s1", "xgb_s1", "rf_s1"])]
    rows = []
    for _, x in inc.iterrows():
        rows.append({"残差信号": f"{SIGLAB[x['signal']]} − 逻辑回归 L1", "预测借壳 AUC 2012–2016": f(x["AUC_2012–2016"], 3),
                     "预测借壳 AUC 2017–2023": f(x["AUC_2017–2023"], 3),
                     **{f"FF6 α {c}": f(x[f"a_{c}"], 2, x[f"p_a_{c}"]) for c in STAGE3}})
    save(pd.DataFrame(rows), "tab5_7_increment", 1)

    # 附录 C-5：全部信号的 Rank IC
    rows = []
    for s in MAIN:
        for kind in ["全市场", "规模组内", "最小 30% 内"]:
            d = I[(I.signal == s) & (I["口径"] == kind)].set_index("阶段")
            rows.append({"信号": SIGLAB[s], "口径": kind, **{c: f(d.at[c, "IC"], 3, d.at[c, "p"]) for c in ["全样本", *STAGE3]},
                         "ICIR（全样本，年化）": f(d.at["全样本", "ICIR"], 2), "IC>0 月份占比": f"{d.at['全样本', 'IC>0']:.0%}"})
    save(pd.DataFrame(rows), "tabC_5_rank_ic", 2)


def main():
    OUT_F.mkdir(parents=True, exist_ok=True), OUT_T.mkdir(parents=True, exist_ok=True)
    for new, src in FIGS.items():
        shutil.copy(src, OUT_F / f"{new}.png")
    fig_flow()
    for new, old in COPY.items():
        for ext in ("md", "csv"):
            shutil.copy(V1T / f"{old}.{ext}", OUT_T / f"{new}.{ext}")
    tables_data()
    tables_pred()
    tables_factor()
    print(f"图 {len(list(OUT_F.glob('*.png')))} 张 → {OUT_F}；表 {len(list(OUT_T.glob('*.md')))} 张 → {OUT_T}")


if __name__ == "__main__":
    main()
