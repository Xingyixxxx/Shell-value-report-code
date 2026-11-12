本目录放定价检验用的月度数据，不随仓库上传（CSMAR 授权数据）。
由 `uv run -m factor_test convert` 从 CSMAR 的 dta 转出（源路径在 `factor_test/config.toml` 的 `[paths]`）。

`stock_month.csv`（月收益、流通/总市值）、`turnover.csv`（换手、交易状态）、`ff5.csv`、`momentum.csv`、
`pe_bm.csv`、`gp.csv`、`assets.csv`、`events.csv`（借壳事件日）、`paper_shell_prob.csv`（原论文的壳概率，用于复现对比）。
