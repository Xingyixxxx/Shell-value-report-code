# A 股上市公司被借壳概率的滚动预测与壳因子检验

课程项目（人工智能 B）。两步：

1. **预测**：每年只用当时可得的信息，用多种机器学习模型预测主板上市公司当年被借壳的概率（“壳概率”）。
2. **检验**：把样本外壳概率当作因子，检验它能否解释股票收益。

样本：2003–2023 年沪深主板非金融公司，45,117 个公司-年度、151 起借壳（正例率 0.33%）；样本外预测期 2012–2023 年。

## 目录

```
code/
├── README.md
├── pyproject.toml / uv.lock       # 依赖（Python 3.12，uv 管理）
├── config.toml                    # 预测部分的全部参数：特征、滚动窗口、模型与候选参数、平衡方法、评价口径
├── main.py                        # 预测部分入口：describe / train / evaluate / all
├── reports/                       # 报告图表整理（统一入口：uv run -m reports）
│   ├── __main__.py                #   v1 / v2 / all，all 按版本顺序生成
│   ├── common.py                  #   输入路径、CSV 读取、示意图绘制
│   ├── v1.py                      #   第一版 → results/report/v1/
│   └── v2.py                      #   第二版（最终版）→ results/report/v2/
├── shellpred/                     # 预测部分
│   ├── config.py                  #   读取配置，展开运行列表 <模型>_<平衡方法>
│   ├── data.py                    #   读取面板、补齐市场变量、构造 NegEq_1
│   ├── describe.py                #   描述性统计表和图
│   ├── sampling.py                #   平衡方法 S0–S5、先验校正
│   ├── models.py                  #   预处理、六类模型、前推调参、滚动预测
│   ├── metrics.py                 #   评价指标：AUC、Top-k、自助法、Holm 校正等
│   ├── evaluate.py                #   指标表、图、特征重要性/SHAP、概率面板导出
│   ├── revision.py                #   第二版新增：AUC 显著性、分阶段 AUC/Lift、L1 系数自助法
│   └── plot.py                    #   绘图与表格输出的公共设置
├── factor_test/                   # 定价检验部分
│   ├── config.toml                #   路径、分组、权重、因子模型、子期、事先写定的选择规则
│   ├── convert.py                 #   CSMAR dta → data/*.csv
│   ├── panel.py                   #   信号宽表、股票-月面板、财务特征、因子（含自构 CH3）
│   ├── portfolios.py              #   分组规则、组合收益、中性化
│   ├── stats.py                   #   Newey-West、GRS、单调性检验、Romano-Wolf、Fama-MacBeth
│   ├── analysis.py                #   主检验（全部表和图）
│   ├── extra.py                   #   补充检验：分年度/分政策阶段、规模口径核对
│   ├── revision.py                #   第二版新增：Rank IC、分阶段、持有期各月、隐含壳溢价
│   └── data/                      #   月度数据（不上传）
├── stata/original.do              # CSMAR 原始表 → data/raw/original.csv
├── data/raw/                      # 建模输入（不上传）
├── outputs/                       # 本地训练与预测产物（整个目录不上传，train / evaluate 可重新生成）
│   ├── <运行名>/                  #   调参记录、训练元数据、预测、模型等
│   └── shell_prob_panel.csv       #   全部模型的样本外概率面板
└── results/                       # 表和图
    ├── tables/ figures/           #   预测部分（含调参汇总 model_selection、训练样本汇总 model_training）
    ├── factor/tables|figures/     #   定价检验部分（x_ 开头为补充检验，r_ 开头为第二版新增）
    └── report/v1|v2/              #   报告用的成品图表（v2 即 LaTeX 版所用的图）
```

## 复现

完整复现需要先按下文准备原始数据，再在本目录下运行。由于 `outputs/` 不随仓库上传，
首次运行 `evaluate`、定价检验和新增检验前，需要先执行 `train` 生成本地预测。

```bash
uv sync                           # 安装依赖
uv run main.py describe           # 数据处理与描述性统计
uv run main.py train              # 训练全部 25 个运行（最耗时）；加 --quick 只检查流程
uv run main.py evaluate           # 评价指标、图表、特征重要性，导出概率面板
uv run -m factor_test convert     # CSMAR dta → factor_test/data/（只需一次）
uv run -m factor_test test        # 全部定价检验，约 12 分钟
uv run -m factor_test extra       # 补充检验
uv run -m reports v1              # 报告第一版图表 → results/report/v1/

# 报告第二版（最终版）新增的结果与图表
uv run -m shellpred.revision      # AUC 显著性、分政策阶段 AUC/Lift、逻辑回归 L1 系数的自助法显著性
uv run -m factor_test revise      # Rank IC、分政策阶段、持有期各月、小市值内部分阶段与隐含壳溢价
uv run -m reports v2              # 报告第二版图表 → results/report/v2/
```

只重新整理报告图表时，直接使用仓库中的 `results/` 即可，无需原始数据或 `outputs/`：

```bash
uv sync
uv run -m reports all             # 依次生成 v1、v2 的成品图表，不重新训练或执行统计检验
```

`v2` 会读取 `results/report/v1/` 的部分成品图表。仓库已保留这些文件；若清理过报告目录，
请运行 `all`，或先运行 `v1` 再运行 `v2`。公共逻辑放在 `reports/common.py`，
两版各自保留编号与表格格式。训练样本统计由 `evaluate` 导出到
`results/tables/model_training.csv`，报告整理统一读取结果表。

## 数据

原始数据来自 CSMAR，属授权数据，不随仓库上传。需要的文件见
[data/raw/README.md](data/raw/README.md) 和 [factor_test/data/README.md](factor_test/data/README.md)。
仓库保留了 `results/` 下的结果表、图、调参汇总与训练样本汇总，不拿数据也能核对报告中的数字。
`outputs/` 整个目录由 `.gitignore` 忽略，仅保存在本地。

## 方法概要

| 环节 | 做法 |
|:--|:--|
| 特征 | 15 个，全部为 t−1 年及以前的信息：IPO 政策（2）、经营（7）、股权与属性（5）、前期超额收益（1） |
| 划分 | 9 年滚动窗口，预测 t 年用 [t−9, t−1] 年训练，2012–2023 年逐年预测 |
| 模型 | 逻辑回归 L2/L1、LDA、随机森林、XGBoost、MLP，另加等权组合 |
| 不平衡 | S0 全数据、S1 类权重、S2 随机欠采样、S3 按年匹配欠采样、S4 PSM 式匹配、S5 SMOTE |
| 调参 | 前推调参：预测 t 年时用各候选在 t−3 至 t−1 年的样本外 AUC 选参数 |
| 评价 | 年内排序 AUC、Top-k Lift/Recall、PR-AUC、校准；按公司重抽样的置信区间，Holm 校正 |
| 定价 | 每年 5 月按壳概率年内排名分组，上月市值加权/等权，FF3/FF6/CH3 alpha，Newey-West |
