"""报告共用的输入路径、CSV 读取与示意图绘制。"""
from pathlib import Path

import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from shellpred.plot import INK, INK2

ROOT = Path(__file__).resolve().parents[1]
RT, RF = ROOT / "results/tables", ROOT / "results/figures"
FT, FF = ROOT / "results/factor/tables", ROOT / "results/factor/figures"


def rd(path, name):
    return pd.read_csv(path / f"{name}.csv", encoding="utf-8-sig")


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, linespacing=1.5)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=INK2, lw=1.3, shrinkA=2, shrinkB=2))
