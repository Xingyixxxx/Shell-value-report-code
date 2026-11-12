"""从已有的 results/ 表格与图片生成报告图表。"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=["v1", "v2", "all"],
                        help="报告版本；all 按 v1、v2 的顺序生成")
    args = parser.parse_args()

    if args.version in ("v1", "all"):
        from . import v1
        v1.main()
    if args.version in ("v2", "all"):
        from . import v2
        v2.main()


if __name__ == "__main__":
    main()
