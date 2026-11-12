"""绘图和表格输出的公共设置。"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 固定顺序的分类色（色盲安全），按实体（模型族）分配，不随排名变化
SERIES = ["#2a78d6", "#eb6834", "#4a3aa7", "#eda100", "#1baf7a", "#e87ba4", "#008300", "#e34948"]
MUTED, GRID, INK, INK2, BG = "#898781", "#e1e0d9", "#0b0b0b", "#52514e", "#fcfcfb"
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]  # 负 - 中性 - 正

plt.rcParams.update({
    "font.sans-serif": ["Arial Unicode MS", "STHeiti", "PingFang SC", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#c3c2b7", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True, "lines.linewidth": 2, "legend.frameon": False,
    "axes.titlesize": 12, "font.size": 10, "savefig.dpi": 200,
})


def save_fig(fig, cfg, name):
    fig.savefig(cfg["paths"]["figures"] / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def save_table(df, cfg, name, floatfmt=".3f"):
    """同时写 csv（便于再加工）和 markdown（直接贴进报告）。
    floatfmt 可以是 {列名: 格式}，未列出的列用 .3f；缺失值在 markdown 中留空。"""
    df.to_csv(cfg["paths"]["tables"] / f"{name}.csv", index=False, encoding="utf-8-sig")
    fmt = [floatfmt.get(c, ".3f") for c in df.columns] if isinstance(floatfmt, dict) else floatfmt
    md = df.astype(object).where(df.notna(), None).to_markdown(index=False, floatfmt=fmt, missingval="")
    (cfg["paths"]["tables"] / f"{name}.md").write_text(md, encoding="utf-8")
