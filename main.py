"""入口。用法：

    uv run main.py describe                     # 数据处理与描述性统计
    uv run main.py train                        # 训练全部运行（模型 × 平衡方法，见 config.toml）
    uv run main.py train logit_s0 xgb_s1        # 只训练指定运行
    uv run main.py train --step 1               # 只训练第一步（step = 1 的模型 × 全部方法）
    uv run main.py evaluate                     # 评价、图表、特征重要性，导出壳概率（缺少的运行跳过）
    uv run main.py all                          # 依次执行以上三步
    加 --quick：候选只取第一组、少量子样本，结果写到 outputs/_quick/，只用于检查流程能否跑通
"""
import argparse

from shellpred import describe, evaluate
from shellpred.config import load_config
from shellpred.models import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["describe", "train", "evaluate", "all"])
    ap.add_argument("runs", nargs="*", help="只训练指定运行，如 logit_s0 xgb_s1")
    ap.add_argument("--step", type=int, choices=[1, 2], dest="train_step", help="只训练第一步或第二步")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.quick)

    if unknown := set(args.runs) - set(cfg["runs"]):
        ap.error(f"未知的运行：{sorted(unknown)}；可选：{list(cfg['runs'])}")
    runs = args.runs or [r for r, info in cfg["runs"].items()
                         if args.train_step is None or info["step"] == args.train_step]

    if args.step in ("describe", "all"):
        describe.run(cfg)
    if args.step in ("train", "all"):
        train(cfg, runs)
    if args.step in ("evaluate", "all"):
        evaluate.run(cfg)


if __name__ == "__main__":
    main()
