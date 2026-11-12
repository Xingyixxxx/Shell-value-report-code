"""滚动预测与前推调参。

1. 候选滚动预测：每组候选参数在 first_val_year…last_pred_year 的每一年 v，用 [v-window, v-1] 训练
   （预处理 → 抽样 → 拟合 → bagging 平均 → 先验校正），预测 v 年；
2. 逐年选参数：预测 t 年时，比较各候选在 [t-lookback, t-1] 年的合并年内排序 AUC，选最高者（并列选更简单的）；
3. 最终预测：t 年直接使用选中候选在 t 年的滚动预测；再用相同种子重拟合一次，得到训练集预测并保存
   XGBoost 模型，重拟合的测试预测必须与候选预测一致。
所有运行的全部拟合任务放进同一个进程池，按预计耗时从长到短派发。"""
import json
import shutil
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from xgboost import XGBRegressor

from . import sampling
from .data import build_dataset
from .metrics import within_year_auc

LINEAR = {"logit", "logit_l1"}  # 输出系数的模型
GZIP = {"method": "gzip", "compresslevel": 1}  # 压缩级别 1：文件稍大，写出快约 4 倍


# ---------------------------------------------------------------- 数据

@dataclass
class Panel:
    """建模数据的 numpy 视图。并行时写成临时 npz，每个工作进程只读一次（见 _panel）。"""
    X: np.ndarray
    y: np.ndarray
    time: np.ndarray
    feats: list[str]
    window_len: int

    @classmethod
    def from_frame(cls, df, feats, window):
        return cls(df[feats].to_numpy(float), df["Y"].to_numpy(int), df["time"].to_numpy(int), feats, window)

    def window(self, year):
        """训练窗口 [year-window, year-1] 与预测年 year 的行掩码。"""
        return (self.time >= year - self.window_len) & (self.time < year), self.time == year


@cache
def _panel(path: str) -> Panel:
    d = np.load(path)
    return Panel(d["X"], d["y"], d["time"], list(d["feats"]), int(d["window"]))


class Preprocessor:
    """插补版特征：训练窗口中位数插补 + 缺失指示变量 miss_* + 标准化。
    缺失位置完全相同的特征（同一张原始表的变量，如 Insider_3 与 reverrate_3）只保留第一个指示变量。
    参数在抽样之前的完整训练窗口上估计，再变换训练集和测试集。binary 记下 0/1 列，供 SMOTE 取整。"""

    def fit(self, X, feats):
        self.med = np.nan_to_num(pd.DataFrame(X).median().to_numpy())
        miss = np.isnan(X)
        self.flags = []
        for j in np.flatnonzero(miss.any(axis=0)):
            if not any((miss[:, j] == miss[:, i]).all() for i in self.flags):
                self.flags.append(j)
        Z = self._design(X)
        self.mu, self.sd = Z.mean(axis=0), Z.std(axis=0)
        self.sd[self.sd == 0] = 1
        self.columns = [*feats, *(f"miss_{feats[j]}" for j in self.flags)]
        self.binary = [j for j in range(Z.shape[1]) if np.isin(Z[:, j], (0, 1)).all()]
        return self

    def _design(self, X):
        return np.hstack([np.where(np.isnan(X), self.med, X), np.isnan(X[:, self.flags]).astype(float)])

    def transform(self, X):
        return (self._design(X) - self.mu) / self.sd


# ---------------------------------------------------------------- 模型与候选参数

COMPLEXITY = {  # 前推调参得分并列时选更简单的候选：值越小越简单；S1 再比较正例权重 w
    "logit": lambda p: (p["C"],),
    "logit_l1": lambda p: (p["C"],),
    "lda": lambda p: (p["shrinkage"] != "auto",),
    "rf": lambda p: (-p["min_samples_leaf"], p["max_features"] != "sqrt"),
    "xgb": lambda p: (p["max_depth"], p["learning_rate"] * p["n_estimators"]),
    "mlp": lambda p: (sum(p["hidden"]), -p["alpha"]),
}
COST = {"rf": 5, "mlp": 4, "logit_l1": 3, "xgb": 2, "logit": 1, "lda": 0}  # 派发顺序：预计耗时长的先跑


def complexity(kind, p):
    return (*COMPLEXITY[kind](p), p.get("w", 0))


def candidates(cfg, run) -> list[dict]:
    """模型的 space × 平衡方法的 space（只有 S1 有 w），展开成候选列表，下标即 cand_id。"""
    info = cfg["runs"][run]
    space = {**cfg["models"][info["model"]]["space"], **cfg["samplers"][info["sampler"]].get("space", {})}
    out = []
    for p in ParameterGrid(space):
        if "lr_trees" in p:
            p["learning_rate"], p["n_estimators"] = p.pop("lr_trees")
        if "hidden" in p:
            p["hidden"] = tuple(p["hidden"])
        out.append(p)
    return out


def make_estimator(kind, p, seed, fixed):
    """返回未拟合的模型。树模型和 MLP 单线程：并行在任务层面进行。"""
    match kind:
        case "logit":
            return LogisticRegression(C=p["C"], l1_ratio=0, max_iter=2000)
        case "logit_l1":
            return LogisticRegression(C=p["C"], l1_ratio=1, solver="saga", max_iter=5000, random_state=seed)
        case "lda":
            return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=None if p["shrinkage"] == "none" else "auto")
        case "rf":
            return RandomForestClassifier(min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"],
                                          n_jobs=1, random_state=seed, **fixed)
        case "xgb":
            return XGBRegressor(tree_method="hist", max_depth=p["max_depth"], learning_rate=p["learning_rate"],
                                n_estimators=p["n_estimators"], n_jobs=1, random_state=seed, **fixed)
        case "mlp":
            return MLPClassifier(hidden_layer_sizes=p["hidden"], alpha=p["alpha"], activation="relu", solver="adam",
                                 early_stopping=False, random_state=seed, **fixed)
    raise ValueError(kind)


def _base_score(y, weight):
    """XGBoost 的初始预测设为（加权后的）正例率。自动估计的 base_score 只从 0.5 出发做一步牛顿迭代
    （正例加权 20 倍时约 0.14，而加权正例率约 0.045）；学习率小、树少时模型会停在这个错误的起点附近，
    先验校正后的概率水平随之失真。"""
    w = np.ones(len(y)) if weight is None else weight
    return float((w * y).sum() / w.sum())


def _seed(*keys) -> int:
    return int(np.random.SeedSequence(keys).generate_state(1)[0])


def _predict(m, X):
    return m.predict(X) if isinstance(m, XGBRegressor) else m.predict_proba(X)[:, 1]


def fit_one(panel, cfg, run, params, year, final=False, model_dir=None):
    """用 [year-window, year-1] 训练一个候选，返回 year 年的原始分数（多个子样本 / 种子取平均）和 β。
    final=True 是最终重拟合：另外返回训练窗口的分数、train_used 行号和线性模型系数；
    model_dir 不为空时保存 XGBoost 模型（SHAP 需要）。"""
    info = cfg["runs"][run]
    spec, s = cfg["models"][info["model"]], info["sampler"]
    kind, t, smp = spec["kind"], cfg["tuning"], cfg["sampling"]
    tr, te = panel.window(year)
    X, Xt, y, years = panel.X[tr], panel.X[te], panel.y[tr], panel.time[tr]
    imputed = spec["prep"] == "impute" or s == "s5"
    if imputed or s == "s4":
        prep = Preprocessor().fit(X, panel.feats)
        Z, Zt = prep.transform(X), prep.transform(Xt)
    A, At = (Z, Zt) if imputed else (X, Xt)
    rngs = [np.random.default_rng([t["seed"], year, b]) for b in range(smp["n_bags"])]
    sample = {
        "s0": lambda: sampling.s0(A, y),
        "s1": lambda: sampling.s1(A, y, params["w"]),
        "s2": lambda: sampling.s2(A, y, smp["ratio"], rngs),
        "s3": lambda: sampling.s3(A, y, years, smp["ratio"], rngs),
        "s4": lambda: sampling.s4(A, Z, y, years, smp["ratio"], smp["psm_C"]),
        "s5": lambda: sampling.s5(Z, y, smp["ratio"], smp["smote_k"], prep.binary, rngs[0]),
    }[s]()

    bagged = len(sample.bags) > 1
    fixed = dict(spec.get("fixed", {}))
    if bagged and "bag_n_estimators" in spec:
        fixed["n_estimators"] = spec["bag_n_estimators"]
    n_seeds = t["mlp_seeds"] if kind == "mlp" and not bagged else 1  # bagging 时每个子样本已用不同种子
    mp = {k: v for k, v in params.items() if k != "w"}
    models = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        for b, bag in enumerate(sample.bags):
            for i in range(n_seeds):
                m = make_estimator(kind, mp, _seed(t["seed"], year, b, i), fixed)
                if kind == "xgb":
                    m.set_params(base_score=_base_score(bag.y, bag.weight))
                m.fit(bag.X, bag.y) if bag.weight is None else m.fit(bag.X, bag.y, sample_weight=bag.weight)
                models.append(m)
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    for w in caught:  # 收敛警告只计数，其他警告照常显示
        if not issubclass(w.category, ConvergenceWarning):
            warnings.showwarning(w.message, w.category, w.filename, w.lineno)

    def score(M):  # 平方误差回归的输出可能越界，截到 [0, 1]
        return np.clip(np.mean([_predict(m, M) for m in models], axis=0), 0, 1)

    out = {"test": score(At), "beta": sample.beta, "n_fits": len(models), "n_conv": n_conv}
    if not final:
        return out
    out |= {"train": score(A), "rows": sample.rows, "n_used": sample.n_used, "n_train": len(y),
            "pos_train": int(y.sum())}
    if kind in LINEAR:  # 标准化系数；bagging 时对各子样本取平均
        coef = np.mean([np.r_[m.intercept_, m.coef_[0]] for m in models], axis=0)
        out["coef"] = dict(zip(["intercept", *(f"coef_{c}" for c in prep.columns)], coef))
    if model_dir is not None:
        cols = prep.columns if imputed else panel.feats
        for b, m in enumerate(models):
            m.get_booster().feature_names = cols
            m.save_model(model_dir / f"{year}_{b}.json")
    return out


def _task(tag, path, cfg, run, params, year, final=False, model_dir=None):
    t0 = time.perf_counter()
    out = fit_one(_panel(path), cfg, run, params, year, final, model_dir)
    return tag, out, time.perf_counter() - t0


# ---------------------------------------------------------------- 前推调参

def forward_select(cand, params, kind, cfg) -> pd.DataFrame:
    """预测 t 年时，比较各候选在 [t-lookback, t-1] 年的合并年内排序 AUC，选最高者（并列时选更简单的）。"""
    t, r = cfg["tuning"], cfg["rolling"]
    rows = []
    for year in range(r["first_pred_year"], r["last_pred_year"] + 1):
        v = cand[cand["time"].between(year - t["lookback"], year - 1)]
        auc = {c: within_year_auc(g["Y"], g["score"], g["time"]) for c, g in v.groupby("cand_id")}
        order = sorted(auc, key=lambda c: (-round(auc[c], 10), complexity(kind, params[c])))
        rows.append({"year": year, "val_years": f"{year - t['lookback']}–{year - 1}", "cand_id": order[0],
                     **params[order[0]], "val_auc": auc[order[0]],
                     "val_auc_2nd": auc[order[1]] if len(order) > 1 else np.nan})
    return pd.DataFrame(rows)


def _frame(df, idx, year, sample, score, p):
    d = df.iloc[idx]
    return pd.DataFrame({"Stkcd": d["Stkcd"].to_numpy(), "time": d["time"].to_numpy(), "Y": d["Y"].to_numpy(),
                         "year": year, "sample": sample, "score": score, "p": p})


def train(cfg, runs):
    """训练指定的运行：全部运行的候选滚动预测放进同一个进程池，再逐年选参数、重拟合并写出结果。"""
    t, r = cfg["tuning"], cfg["rolling"]
    df, feats = build_dataset(cfg, dropna=False)
    panel = Panel.from_frame(df, feats, r["window"])
    kinds = {run: cfg["models"][cfg["runs"][run]["model"]]["kind"] for run in runs}
    cands = {run: candidates(cfg, run) for run in runs}
    val_years = range(t["first_val_year"], r["last_pred_year"] + 1)
    pred_years = range(r["first_pred_year"], r["last_pred_year"] + 1)
    jobs = sorted(((run, ci, y) for run in runs for ci in range(len(cands[run])) for y in val_years),
                  key=lambda j: (COST[kinds[j[0]]], j[2]), reverse=True)
    n_jobs = effective_n_jobs(t["n_jobs"])
    print(f"候选滚动预测：{len(runs)} 个运行、{len(jobs)} 个拟合任务，{n_jobs} 个进程并行", flush=True)

    with tempfile.TemporaryDirectory() as tmp, \
            Parallel(n_jobs=n_jobs, return_as="generator_unordered", pre_dispatch="4*n_jobs") as pool:
        path = str(Path(tmp) / "panel.npz")
        np.savez(path, X=panel.X, y=panel.y, time=panel.time, feats=np.array(feats), window=r["window"])

        start, left = time.perf_counter(), Counter(j[0] for j in jobs)
        res = defaultdict(dict)
        for (run, ci, year), out, sec in pool(delayed(_task)((run, ci, year), path, cfg, run, cands[run][ci], year)
                                              for run, ci, year in jobs):
            res[run][ci, year] = out | {"seconds": sec}
            left[run] -= 1
            if not left[run]:
                print(f"  [{run}] {len(cands[run])} 个候选 × {len(val_years)} 年完成"
                      f"（已用 {time.perf_counter() - start:.0f} 秒）", flush=True)

        cand, sel, model_dirs = {}, {}, {}
        for run in runs:
            cand[run] = pd.concat([
                _frame(df, np.flatnonzero(panel.time == y), y, "test", o["test"],
                       sampling.prior_correct(o["test"], o["beta"])).assign(cand_id=ci)
                for (ci, y), o in sorted(res[run].items())])[["cand_id", "Stkcd", "time", "Y", "score", "p"]]
            sel[run] = forward_select(cand[run], cands[run], kinds[run], cfg)
            out_dir = cfg["paths"]["outputs"] / run
            shutil.rmtree(out_dir / "models", ignore_errors=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            if kinds[run] == "xgb":
                model_dirs[run] = out_dir / "models"
                model_dirs[run].mkdir()

        jobs = sorted(((run, year, ci) for run in runs for year, ci in zip(sel[run]["year"], sel[run]["cand_id"])),
                      key=lambda j: (COST[kinds[j[0]]], j[1]), reverse=True)
        print(f"重拟合选中的候选：{len(jobs)} 个任务", flush=True)
        refit = {}
        for (run, year), out, sec in pool(
                delayed(_task)((run, year), path, cfg, run, cands[run][ci], year, True, model_dirs.get(run))
                for run, year, ci in jobs):
            refit[run, year] = out | {"seconds": sec}

    for run in runs:
        _write(cfg, run, df, feats, cands[run], sel[run], cand[run], res[run],
               {y: refit[run, y] for y in pred_years})


def _write(cfg, run, df, feats, cands, sel, cand, res, refit):
    """写出 candidates / selection / predictions / meta / manifest，并核对重拟合与候选预测一致。"""
    info = cfg["runs"][run]
    spec, out_dir = cfg["models"][info["model"]], cfg["paths"]["outputs"] / run
    resampled = info["sampler"] in sampling.RESAMPLED
    preds, metas, mismatch = [], [], []
    for s in sel.to_dict("records"):
        year = s.pop("year")
        o = refit[year]
        if not np.allclose(o["test"], res[s["cand_id"], year]["test"]):
            mismatch.append(year)
        tr, te = (np.flatnonzero(m) for m in ((df["time"] >= year - cfg["rolling"]["window"]) & (df["time"] < year),
                                              df["time"] == year))
        preds += [_frame(df, tr, year, "train_all", o["train"], sampling.prior_correct(o["train"], o["beta"])),
                  _frame(df, te, year, "test", o["test"], sampling.prior_correct(o["test"], o["beta"]))]
        if resampled:  # 实际用于训练的样本（论文的 R²_IS），在该样本的分布下不做校正
            used = o["train"][o["rows"]]
            preds.append(_frame(df, tr[o["rows"]], year, "train_used", used, used))
        metas.append({"year": year, "n_train": o["n_train"], "pos_train": o["pos_train"], "n_used": o["n_used"],
                      "beta": o["beta"], **s, "fit_seconds": o["seconds"], **o.get("coef", {})})

    cand.to_csv(out_dir / "candidates.csv.gz", index=False, compression=GZIP)
    sel.to_csv(out_dir / "selection.csv", index=False)
    pred = pd.concat(preds)
    pred.to_csv(out_dir / "predictions.csv.gz", index=False, compression=GZIP)
    pd.DataFrame(metas).to_csv(out_dir / "meta.csv", index=False)

    manifest = {
        "run": run, "label": info["label"], "model": spec, "sampler": cfg["samplers"][info["sampler"]],
        "sampling": cfg["sampling"], "tuning": cfg["tuning"], "rolling": cfg["rolling"], "features": feats,
        "candidates": cands, "n_rows": len(df), "n_pos": int(df["Y"].sum()),
        "cand_fits": sum(o["n_fits"] for o in res.values()), "cand_seconds": sum(o["seconds"] for o in res.values()),
        "refit_fits": sum(o["n_fits"] for o in refit.values()), "refit_seconds": sum(o["seconds"] for o in refit.values()),
        "convergence_warnings": sum(o["n_conv"] for o in [*res.values(), *refit.values()]),
        "refit_mismatch_years": mismatch, "finished": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if mismatch:
        print(f"  ⚠ [{run}] {mismatch} 年重拟合的测试预测与候选预测不一致：种子或抽样没有固定", flush=True)
    te = pred[pred["sample"] == "test"]
    print(f"  [{run}] 样本外合并年内 AUC = {within_year_auc(te['Y'], te['score'], te['time']):.3f}，"
          f"累计拟合 {manifest['cand_fits'] + manifest['refit_fits']} 次", flush=True)
