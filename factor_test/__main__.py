"""壳概率因子检验入口。用法（在项目根目录）：

    uv run -m factor_test convert          # dta → csv（factor_test/data/，只需跑一次）
    uv run -m factor_test test             # 全部检验，输出表和图到 results/factor/
    uv run -m factor_test extra            # 补充检验：分年度/分政策阶段、规模口径核对（探索性）
    uv run -m factor_test revise           # 报告第二版新增：Rank IC、分阶段、持有期各月、小市值分阶段
    uv run -m factor_test all              # 依次执行 convert、test、extra
    uv run -m factor_test test --outputs <预测目录> --tag <名称>   # 用另一批预测重跑，结果写到 output/<名称>/
"""
import argparse

from .panel import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["convert", "test", "extra", "revise", "all"])
    ap.add_argument("--outputs", help="预测所在目录，默认取配置")
    ap.add_argument("--tag", help="结果子目录名，用于对照实验")
    args = ap.parse_args()
    cfg = load_config(args.outputs, args.tag)
    if args.step in ("convert", "all"):
        from . import convert
        convert.run(cfg)
    if args.step in ("test", "all"):
        from . import analysis
        analysis.run(cfg)
    if args.step in ("extra", "all"):
        from . import extra
        extra.run(cfg)
    if args.step == "revise":
        from . import revision
        revision.run(cfg)


if __name__ == "__main__":
    main()
