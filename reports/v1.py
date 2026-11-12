"""生成报告用的图表：把 results/ 下的全量结果裁成正文和附录的成品，并画三张示意图。

    uv run -m reports v1

输出到 results/report/v1/figures（png）和 results/report/v1/tables（同名 md + csv）。
"""
import shutil

import numpy as np
import pandas as pd

from shellpred.plot import INK2, MUTED, SERIES, plt

from .common import FF, FT, RF, ROOT, RT, arrow, box, rd
OUT_F, OUT_T = ROOT / "results/report/v1/figures", ROOT / "results/report/v1/tables"

MAIN = ["逻辑回归 L2 · 全数据", "逻辑回归 L1 · 全数据", "LDA · 全数据", "随机森林 · 类权重", "XGBoost · 类权重", "MLP · 类权重", "等权组合"]
SIG3 = {"xgb_s1": "XGBoost · 类权重", "logit_l1_s0": "逻辑回归 L1 · 全数据", "ensemble": "等权组合"}
WNAME = {"ew": "等权", "vw": "流通市值加权"}

# 现成的图：新文件名 ← 原文件
FIGS = {
    "fig1_1_sample_by_year": RF / "fig_sample_by_year.png", "fig1_2_ipo": RF / "fig_ipo.png",
    "fig1_3_feature_box": RF / "fig_feature_box.png",
    "fig3_1_grid_heatmap": RF / "fig_grid_heatmap.png",
    "fig4_1_sampler_forest": RF / "fig_sampler_forest.png", "fig4_2_forest_all": RF / "fig_forest_all.png",
    "fig4_3_auc_by_year": RF / "fig_auc_by_year.png", "fig4_4_gains_lift": RF / "fig_gains_lift.png",
    "fig4_5_calibration": RF / "fig_calibration.png", "fig4_6_importance": RF / "fig_importance.png",
    "fig4_7_shap_dependence": RF / "fig_shap_dependence.png",
    "fig5_1_deciles": FF / "fig_deciles.png", "fig5_2_lookahead": FF / "fig_x_lookahead.png",
    "fig5_3_cum_stages": FF / "fig_x_cum_stages.png", "fig5_4_annual": FF / "fig_x_annual.png",
    "fig5_5_small_cap": FF / "fig_small_cap.png",
    "figA_1_corr": RF / "fig_corr.png", "figA_2_roc_pr": RF / "fig_roc_pr.png", "figA_3_patterns": RF / "fig_patterns.png",
    "figA_4_tstat_heat": FF / "fig_heat.png", "figA_5_cum_all": FF / "fig_cum.png",
    "figA_6_double_sort": FF / "fig_double_sort.png", "figA_7_event": FF / "fig_event.png",
}


def save(df, name, fmt=".3f"):
    df.to_csv(OUT_T / f"{name}.csv", index=False, encoding="utf-8-sig")
    f = [fmt.get(c, ".3f") for c in df.columns] if isinstance(fmt, dict) else fmt
    md = df.astype(object).where(df.notna(), None).to_markdown(index=False, floatfmt=f, missingval="")
    (OUT_T / f"{name}.md").write_text(md + "\n", encoding="utf-8")


def mt(m, t, d=2):
    """均值（t 值）合成一格。"""
    return [f"{a:.{d}f} ({b:.2f})" if pd.notna(a) else "" for a, b in zip(m, t)]


# ---------------------------------------------------------------- 示意图

def fig_flow():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13), ax.set_ylim(0, 3.6), ax.axis("off")
    blue, orange, grey = "#dce9f8", "#fbe3d6", "#efeeea"
    steps = [(0.1, "CSMAR 原始数据\n手工整理借壳名单", grey), (2.25, "公司-年度面板\n45,117 行 · 151 起借壳\n15 个滞后特征", grey),
             (4.55, "9 年滚动窗口\n2012–2023 逐年预测", blue), (6.75, "6 类模型 × 6 种平衡方法\n共 25 个运行\n前推调参", blue),
             (9.15, "样本外壳概率\n年内百分位排名", blue), (11.2, "分组构建组合\n因子模型 alpha", orange)]
    w = [1.9, 2.05, 1.95, 2.15, 1.8, 1.7]
    for (x, t, c), wi in zip(steps, w):
        box(ax, x, 1.5, wi, 1.3, t, c, fs=9.5)
    for (x, _, _), wi, (x2, _, _) in zip(steps, w, steps[1:]):
        arrow(ax, (x + wi + .02, 2.15), (x2 - .02, 2.15))
    for x0, x1, lab, col in [(0.1, 4.3, "数据", MUTED), (4.55, 10.95, "第一步：预测壳概率", SERIES[0]), (11.2, 12.9, "第二步：因子检验", SERIES[1])]:
        ax.plot([x0, x1], [3.15, 3.15], color=col, lw=2.5, solid_capstyle="round")
        ax.text((x0 + x1) / 2, 3.3, lab, ha="center", va="bottom", fontsize=10.5, color=col)
    box(ax, 6.75, 0.2, 2.15, 0.85, "评价：年内排序 AUC、Top-k\n自助法区间、Holm 校正", "#ffffff", ec="#c3c2b7", fs=9)
    arrow(ax, (7.82, 1.48), (7.82, 1.07))
    box(ax, 11.2, 0.2, 1.7, 0.85, "分年度、分政策阶段\n规模中性、口径核对", "#ffffff", ec="#c3c2b7", fs=9)
    arrow(ax, (12.05, 1.48), (12.05, 1.07))
    fig.savefig(OUT_F / "fig1_0_flow.png", bbox_inches="tight"), plt.close(fig)


def fig_rolling():
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    win = rd(RT, "data_windows")
    for i, y in enumerate(range(2012, 2024)):
        a.barh(i, 9, left=y - 9, color=SERIES[0], alpha=.75, height=.62)
        a.barh(i, 1, left=y, color=SERIES[1], height=.62)
        r = win[win["预测年"] == y].iloc[0]
        a.text(y + 1.15, i, f"训练正例 {int(r['全部样本训练正例'])}，测试正例 {int(r['全部样本测试正例'])}", va="center", fontsize=8, color=INK2)
    a.set_yticks(range(12), [f"预测 {y}" for y in range(2012, 2024)])
    a.invert_yaxis(), a.set_xlim(2003, 2028.5)
    a.set_xticks(range(2003, 2025, 3), [str(y) for y in range(2003, 2025, 3)])
    a.grid(axis="y", visible=False)
    a.barh([np.nan], [0], color=SERIES[0], alpha=.75, label="训练窗口（9 年）")
    a.barh([np.nan], [0], color=SERIES[1], label="测试年（1 年）")
    a.legend(loc="lower left", fontsize=9)
    a.set_title("（a）滚动窗口：预测 t 年只用 [t−9, t−1] 年训练")

    # 前推调参：以预测 2015 年为例
    b.set_xlim(2002.5, 2016.6), b.set_ylim(-1.5, 4.9), b.grid(False)
    for s in ("left", "bottom"):
        b.spines[s].set_visible(False)
    b.set_yticks([])
    b.set_xticks(range(2003, 2016), [str(y) for y in range(2003, 2016)], fontsize=8)
    for i, v in enumerate([2012, 2013, 2014]):
        yy = 4 - i
        b.barh(yy, 9, left=v - 9 - .5, color=SERIES[0], alpha=.35, height=.55)
        b.barh(yy, 1, left=v - .5, color=SERIES[2], alpha=.85, height=.55)
        b.text(v, yy, "验证", ha="center", va="center", color="white", fontsize=8)
    b.text(2002.6, 4.55, "① 每个候选参数先各自做滚动预测，得到 2012、2013、2014 年的样本外预测", fontsize=9, color=INK2)
    b.text(2002.6, 1.35, "② 三年预测各自转成年内排名后合并，算每个候选的 AUC，取最高者", fontsize=9, color=INK2)
    b.barh(0.6, 9, left=2006 - .5, color=SERIES[0], alpha=.75, height=.55)
    b.barh(0.6, 1, left=2015 - .5, color=SERIES[1], height=.55)
    b.text(2015, 0.6, "预测", ha="center", va="center", color="white", fontsize=8)
    b.text(2002.6, -0.1, "③ 直接取该候选对 2015 年的滚动预测（训练窗口 2006–2014）", fontsize=9, color=INK2)
    b.text(2002.6, -0.85, "选参只用 2014 年及以前的样本外成绩，不接触 2015 年的标签", fontsize=9, color=SERIES[1])
    b.set_title("（b）前推调参：以预测 2015 年为例")
    fig.savefig(OUT_F / "fig2_1_rolling_tuning.png", bbox_inches="tight"), plt.close(fig)


def fig_imbalance(t):
    """六种平衡方法处理后的训练集构成（2015 年窗口）。"""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    d = t.iloc[::-1]
    ax.barh(d["方法"], d["负例"], color=MUTED, alpha=.55, label="负例")
    ax.barh(d["方法"], d["正例（含加权或合成）"], left=d["负例"], color=SERIES[1], label="正例（含加权或合成）")
    for i, (n, p, r) in enumerate(zip(d["负例"], d["正例（含加权或合成）"], d["负:正"])):
        ax.text(n + p + 150, i, f"负:正 = {r}", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, 21500), ax.set_xlabel("训练集有效行数（预测 2015 年，训练窗口 2006–2014）")
    ax.grid(axis="y", visible=False), ax.legend(loc="center right", fontsize=9, bbox_to_anchor=(1.0, 0.42))
    fig.savefig(OUT_F / "fig2_2_imbalance.png", bbox_inches="tight"), plt.close(fig)


# ---------------------------------------------------------------- 表

def tables_data():
    save(rd(RT, "data_funnel"), "tab1_1_sample", {"正例率(%)": ".2f"})
    d, m = rd(RT, "data_desc"), rd(RT, "data_missing").set_index("变量")
    cat = {"IPO_1": "政策环境", "IPOnum_1": "政策环境", "Soe_1": "企业属性", "ST_1": "经营状况", "Size_3": "企业属性", "Op_3": "经营状况",
           "Sales_3": "经营状况", "reverrate_3": "股权结构", "Insider_3": "股权结构", "Lev_3": "经营状况", "Cash_3": "经营状况",
           "Roe_3": "经营状况", "TOP1_3": "股权结构", "return_3": "市场表现", "NegEq_1": "经营状况", "Y": "被解释变量"}
    d.insert(0, "类别", d["变量"].map(cat))
    d["缺失率(%)"] = d["变量"].map(m["处理后缺失率(%)"])
    save(d.drop(columns=["N", "最小值", "最大值"]), "tab1_2_desc", {"缺失率(%)": ".1f", "偏度": ".2f"})
    g = rd(RT, "data_group_compare")
    g.columns = ["变量", "含义", "被借壳均值", "未被借壳均值", "均值差", "t 检验 p", "显著性", "被借壳中位数", "未被借壳中位数", "秩和检验 p", "显著性(秩和)"]
    save(g, "tab1_3_group_compare")
    save(m.reset_index(), "tabA_1_missing", {"原始缺失率(%)": ".2f", "处理后缺失率(%)": ".2f"})


def tables_design():
    w = rd(RT, "data_windows").iloc[:, :6]
    w.columns = ["预测年", "训练窗口", "训练样本", "训练正例", "测试样本", "测试正例"]
    w.insert(4, "训练负:正", ((w["训练样本"] - w["训练正例"]) / w["训练正例"]).round(0).astype(int))
    save(w, "tab2_1_windows")

    r = w[w["预测年"] == 2015].iloc[0]
    n1, n0 = int(r["训练正例"]), int(r["训练样本"] - r["训练正例"])
    s4 = int(rd(RT, "model_training").query("运行 == 'logit_s4' and 预测年 == 2015")["处理后样本"].iloc[0])
    syn = int(np.ceil(n0 / 20)) - n1
    rows = [("S0 全数据", "不处理", n0, n1), ("S1 类权重（w = 5）", "正例在损失函数中加权，不增删样本", n0, n1 * 5),
            ("S1 类权重（w = 20）", "同上", n0, n1 * 20), ("S2 随机欠采样", "保留全部正例，随机抽 20 倍负例，重复 10 次取平均", n1 * 20, n1),
            ("S3 按年匹配欠采样", "每年内各抽 20 倍负例，训练集每年正负比相同，重复 10 次", n1 * 20, n1),
            ("S4 PSM 式匹配", "每个正例取同年倾向得分最接近的 20 个负例（对照）", s4 - n1, n1),
            ("S5 SMOTE", f"在正例之间插值合成 {syn} 个正例，负例不删（对照）", n0, n1 + syn)]
    t = pd.DataFrame(rows, columns=["方法", "做法", "负例", "正例（含加权或合成）"])
    t["负:正"] = [f"{a / b:.0f}:1" for a, b in zip(t["负例"], t["正例（含加权或合成）"])]
    t["正例占比(%)"] = t["正例（含加权或合成）"] / (t["负例"] + t["正例（含加权或合成）"]) * 100
    save(t, "tab2_2_imbalance", {"正例占比(%)": ".2f"})
    fig_imbalance(t)


def tables_models():
    s = rd(RT, "model_sampler_compare")
    s = s[["模型", "方法", "AUC(年内排序)", "95%CI", "相对S0差值", "差值95%CI", "Lift@10%", "Brier", "平均训练样本量"]]
    save(s, "tab4_1_sampler", {"Brier": ".5f", "平均训练样本量": ".0f", "Lift@10%": ".2f"})

    p = rd(RT, "model_pooled").query("样本 == '全部样本'").set_index("模型")
    k = rd(RT, "model_topk").query("样本 == '全部样本'").set_index("模型")
    by = rd(RT, "model_by_year").set_index("预测年")
    full = p[["AUC(年内排序)", "95%CI", "逐年AUC均值", "PR_AUC/正例率", "Brier"]].join(k[["Lift@5%", "Lift@10%", "Recall@10%"]])
    m = full.loc[MAIN, ["AUC(年内排序)", "95%CI", "逐年AUC均值", "PR_AUC/正例率", "Lift@10%", "Recall@10%"]].copy()
    m.insert(3, "2012–2016 AUC", by.loc["2012–2016 合并", MAIN].astype(float))
    m.insert(4, "2017–2023 AUC", by.loc["2017–2023 合并", MAIN].astype(float))
    save(m.reset_index(), "tab4_2_main_models", {"PR_AUC/正例率": ".2f", "Lift@10%": ".2f"})

    imp = rd(RT, "model_importance")[["变量", "含义", "XGBoost · 类权重 增益(%)", "主模型 平均绝对SHAP(%)"]]
    imp.columns = ["变量", "含义", "XGBoost 增益(%)", "XGBoost SHAP 占比(%)"]
    c = rd(RT, "model_logit_l1_s0_coef").set_index("变量")
    imp["逻辑回归 L1 系数均值"] = imp["变量"].map(c["系数均值"])
    imp["系数为正的年份占比"] = imp["变量"].map(c["为正的年份占比"])
    imp["非零年份数"] = imp["变量"].map(c["非零年份数"])
    save(imp.sort_values("XGBoost SHAP 占比(%)", ascending=False), "tab4_3_importance", {"XGBoost 增益(%)": ".1f", "XGBoost SHAP 占比(%)": ".1f", "系数为正的年份占比": ".2f"})

    # 附录
    a = full.reset_index().sort_values("AUC(年内排序)", ascending=False)
    save(a, "tabA_2_all_runs", {"PR_AUC/正例率": ".2f", "Lift@5%": ".2f", "Lift@10%": ".2f", "Brier": ".5f"})
    d = rd(RT, "model_auc_diff").query("样本 == '全部样本'").drop(columns="样本").sort_values("p 值")
    save(d, "tabA_3_auc_diff")
    save(by.reset_index(), "tabA_4_auc_by_year")
    sel = rd(RT, "model_selection")
    save(sel[sel["运行"].isin(["XGBoost · 类权重", "逻辑回归 L1 · 全数据"])], "tabA_5_tuning", ".4f")
    save(rd(RT, "model_calib_year"), "tabA_6_calibration")
    mi = c.reset_index()
    save(mi[mi["变量"].str.startswith("miss_")], "tabA_7_missing_indicators")


def tables_factor():
    g = rd(FT, "grid_all")
    g = g[(g["n"] == 10) & g["signal"].isin(SIG3)]
    u = pd.DataFrame({"信号": g["label"], "权重": g["weight"].map(WNAME), "均值": g["mean"], "t": g["t_mean"],
                      "FF3 α": g["a_FF3"], "t(FF3)": g["t_FF3"], "FF6 α": g["a_FF6"], "t(FF6)": g["t_FF6"],
                      "CH3 α": g["a_CH3"], "t(CH3)": g["t_CH3"], "Romano-Wolf p": g["p_RW"]})
    save(u, "tab5_2_univariate", ".2f")

    lb = rd(FT, "legacy_bridge")
    a = pd.DataFrame({"检验": "A. 20 组市值加权（原论文做法）", "口径": lb["口径"], "多空": lb["mean"], "t": lb["t_mean"], "FF6 α": lb["a_FF6"], "t(FF6)": lb["t_FF6"]})
    a = a[a["口径"].str.contains("原论文壳概率|xgb_s1")]
    x = rd(FT, "x_lookahead").query("align == '5 月对齐'")
    lab = SIG3 | {"paper_prob": "原论文壳概率"}
    b = pd.DataFrame({"检验": "B. 月度独立 5×5、规模中性、等权（组员做法）", "口径": x["signal"].map(lab) + "，规模取" + x["size"].str.replace("（组员口径）", ""),
                      "多空": x["neutral"], "t": x["t"], "FF6 α": x["a_FF6"], "t(FF6)": x["t_FF6"]})
    save(pd.concat([a, b]), "tab5_3_lookahead", ".2f")
    cells = rd(FT, "x_lookahead_cells").query("align == '5 月对齐' and signal == 'xgb_s1'")
    cw = cells.assign(v=mt(cells["ls"], cells["t"])).pivot(index="size", columns="size_group", values="v")
    cw.columns = [f"规模组 {k}" + ("（最小）" if k == 1 else "（最大）" if k == 5 else "") for k in cw.columns]
    save(cw.reset_index().rename(columns={"size": "规模口径"}), "tabA_8_lookahead_cells")

    s = rd(FT, "x_stages")
    def stage_table(sig):
        d = s[s["signal"] == sig]
        rows = []
        for (st, w), q in d.groupby(["strategy", "weight"], sort=False):
            q = q.set_index("stage")
            for lab, m, t in [("均值", "mean", "t_mean"), ("FF6 α", "a_FF6", "t_FF6")]:
                rows.append({"构建方式": st, "权重": WNAME[w], "指标": lab, **dict(zip(q.index, mt(q[m], q[t])))})
        return pd.DataFrame(rows)
    save(stage_table("xgb_s1"), "tab5_4_stages")
    save(pd.concat([stage_table(k).assign(信号=v) for k, v in SIG3.items() if k != "xgb_s1"]), "tabA_9_stages_other")

    y = rd(FT, "x_annual").query("signal == 'xgb_s1'")
    yw = y.assign(c=y["strategy"] + "，" + y["weight"].map(WNAME)).pivot(index="year", columns="c", values="cum")
    order = [f"{a}，{b}" for a in ["十分组 D10−D1", "规模中性 Q5−Q1", "最小 30% 内 Q5−Q1"] for b in WNAME.values()]
    save(yw[order].reset_index().rename(columns={"year": "形成年"}), "tab5_5_annual", ".1f")
    save(rd(FT, "x_annual").assign(signal=lambda d: d["signal"].map(SIG3), weight=lambda d: d["weight"].map(WNAME)), "tabA_10_annual_all", ".2f")

    sc = rd(FT, "small_cap")
    core = ["xgb_s1", "logit_s0", "logit_l1_s0", "lda_s0", "rf_s1", "mlp_s1", "ensemble"]
    sc = sc[sc["signal"].isin(core)]
    t = pd.DataFrame({"信号": sc["label"], "权重": sc["weight"].map(WNAME), "Q5−Q1": mt(sc["mean"], sc["t_mean"]),
                      "FF3 α": mt(sc["a_FF3"], sc["t_FF3"]), "FF6 α": mt(sc["a_FF6"], sc["t_FF6"]), "CH3 α": mt(sc["a_CH3"], sc["t_CH3"]),
                      "FF6 α，2016-08 前": mt(sc["a_pre"], sc["t_pre"]), "FF6 α，2016-09 后": mt(sc["a_post"], sc["t_post"]), "Romano-Wolf p": sc["p_RW"]})
    save(t, "tab5_6_small_cap", ".2f")

    # 附录：稳健性、组合构建、中性化、Fama-MacBeth、十分组特征
    rb = rd(FT, "robustness").query("signal == 'xgb_s1'")
    save(pd.DataFrame({"口径": rb["variant"], "权重": rb["weight"].map(WNAME), "均值": mt(rb["mean"], rb["t_mean"]), "FF6 α": mt(rb["a_FF6"], rb["t_FF6"]), "CH3 α": mt(rb["a_CH3"], rb["t_CH3"])}), "tabA_11_robustness")
    sr = rd(FT, "small_cap_robust").query("signal == 'xgb_s1'")
    save(pd.DataFrame({"口径": sr["variant"], "权重": sr["weight"].map(WNAME), "均值": mt(sr["mean"], sr["t_mean"]), "FF6 α": mt(sr["a_FF6"], sr["t_FF6"]), "CH3 α": mt(sr["a_CH3"], sr["t_CH3"])}), "tabA_12_small_cap_robust")
    cs = rd(FT, "construction").query("signal == 'xgb_s1'")
    save(pd.DataFrame({"构建方式": cs["method"], "均值": mt(cs["mean"], cs["t_mean"]), "FF6 α": mt(cs["a_FF6"], cs["t_FF6"]), "CH3 α": mt(cs["a_CH3"], cs["t_CH3"]), "夏普": cs["sharpe"], "年换手": cs["turnover"]}), "tabA_13_construction", ".2f")
    fm = rd(FT, "fama_macbeth").query("signal == 'xgb_s1'")
    save(pd.DataFrame({"设定": fm["spec"], "壳概率排名系数(%)": fm["b"], "t": fm["t"], "平均 R²": fm["r2"], "平均截面公司数": fm["n"], "t(规模)": fm["t_size"]}), "tabA_14_fama_macbeth", {"平均截面公司数": ".0f", "t": ".2f", "t(规模)": ".2f"})
    dc = rd(FT, "deciles_chars").query("signal == 'xgb_s1'")
    save(pd.DataFrame({"组": dc["d"].astype(int), "公司-年": dc["n"], "借壳数": dc["events"], "流通市值(十亿元，几何均值)": dc["mean_mv"], "亏损比例": dc["loss"], "ST 比例": dc["st"], "换手(%)": dc["turn"]}), "tabA_15_decile_chars", ".2f")


def main():
    for d in (OUT_F, OUT_T):
        d.mkdir(parents=True, exist_ok=True)
    for name, src in FIGS.items():
        shutil.copy(src, OUT_F / f"{name}.png")
    fig_flow(), fig_rolling()
    tables_data(), tables_design(), tables_models(), tables_factor()
    print(f"图 {len(list(OUT_F.glob('*.png')))} 张 → {OUT_F}\n表 {len(list(OUT_T.glob('*.md')))} 张 → {OUT_T}")


if __name__ == "__main__":
    main()
