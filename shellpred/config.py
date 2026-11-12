import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(quick: bool = False) -> dict:
    cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    paths = cfg["paths"]
    if quick:  # 冒烟测试：候选只取第一组、少量子样本，结果写到单独目录，不覆盖正式结果
        q = cfg["quick"]
        for spec in [*cfg["models"].values(), *cfg["samplers"].values()]:
            if "space" in spec:
                spec["space"] = {k: v[:1] for k, v in spec["space"].items()}
        cfg["sampling"]["n_bags"] = q["n_bags"]
        cfg["tuning"]["mlp_seeds"] = q["mlp_seeds"]
        cfg["evaluate"]["bootstrap"] = q["bootstrap"]
        paths.update(outputs=q["outputs"], tables=f"{q['outputs']}/tables", figures=f"{q['outputs']}/figures")
    cfg["paths"] = {k: ROOT / v for k, v in paths.items()}
    for k in ("outputs", "tables", "figures"):
        cfg["paths"][k].mkdir(parents=True, exist_ok=True)
    cfg["runs"] = expand_runs(cfg)
    return cfg


def expand_runs(cfg) -> dict:
    """由各模型的 samplers 展开运行列表：运行名 <模型>_<方法>，按 config 中的顺序。"""
    return {f"{m}_{s}": {"model": m, "sampler": s, "step": spec["step"],
                         "label": f"{spec['label']} · {cfg['samplers'][s]['label']}"}
            for m, spec in cfg["models"].items() for s in spec["samplers"]}
