# -*- coding: utf-8 -*-
"""测试集评估：模型指标、朴素基线对比、图表与结果文件导出。"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from .metrics import (
    evaluate_predictions,
    inverse_transform_targets,
    persistence_prediction,
    seasonal_naive_prediction,
    skill_score,
)
from .utils import resolve_device, save_json, setup_matplotlib

__all__ = ["run_test", "plot_history"]


def plot_history(history: dict, out_dir: str) -> str:
    plt = setup_matplotlib()
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="训练损失（整轮平均）")
    axes[0].plot(epochs, history["val_loss"], label="验证损失（整轮平均）")
    best_epoch = history.get("best", {}).get("epoch")
    if best_epoch and best_epoch > 0:
        axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1, label="最优轮次")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("损失曲线")
    axes[0].legend()

    axes[1].plot(epochs, history["val_rmse"], color="tab:orange", label="验证集 RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("验证集 RMSE（原始量纲）")
    axes[1].legend()

    path = os.path.join(out_dir, "training_curve.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_results(true_raw: np.ndarray, model_raw: np.ndarray, base_raw: np.ndarray,
                  per_horizon: list, baseline_per_horizon: list, out_dir: str,
                  n_show: int = 480) -> list:
    plt = setup_matplotlib()
    paths = []
    h1_true, h1_pred = true_raw[:, 0], model_raw[:, 0]
    n = min(n_show, len(h1_true))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))
    axes[0].plot(np.arange(n), h1_true[:n], label="真实值", linewidth=1.6)
    axes[0].plot(np.arange(n), h1_pred[:n], label="预测值", linewidth=1.2, alpha=0.9)
    axes[0].plot(np.arange(n), base_raw[:n, 0], label="持续性基线", linewidth=1.0,
                 linestyle="--", alpha=0.8)
    axes[0].set_xlabel("样本序号（1 个点 = 15 分钟）")
    axes[0].set_ylabel("功率")
    axes[0].set_title(f"测试集第 1 步预测对比（前 {n} 条样本）")
    axes[0].legend()

    axes[1].scatter(h1_true, h1_pred, s=6, alpha=0.35, label="预测值")
    lim = [min(h1_true.min(), h1_pred.min()), max(h1_true.max(), h1_pred.max())]
    axes[1].plot(lim, lim, color="gray", linestyle="--", linewidth=1, label="理想对角线")
    axes[1].set_xlabel("真实值")
    axes[1].set_ylabel("预测值")
    axes[1].set_title("散点分布（第 1 步）")
    axes[1].legend()
    fig.tight_layout()
    p1 = os.path.join(out_dir, "prediction_curve.png")
    fig.savefig(p1)
    plt.close(fig)
    paths.append(p1)

    fig, ax = plt.subplots(figsize=(7, 4))
    hs = np.array([m["horizon"] for m in per_horizon])
    model_rmse = [m["rmse"] for m in per_horizon]
    base_rmse = [m["rmse"] for m in baseline_per_horizon]
    ax.bar(hs - 0.18, model_rmse, width=0.36, label="本文模型")
    ax.bar(hs + 0.18, base_rmse, width=0.36, label="持续性基线")
    ax.set_xticks(hs)
    ax.set_xlabel("预测步长（1 步 = 15 分钟）")
    ax.set_ylabel("RMSE")
    ax.set_title("分步长 RMSE：模型与持续性基线对比")
    ax.legend()
    p2 = os.path.join(out_dir, "per_horizon_rmse.png")
    fig.savefig(p2)
    plt.close(fig)
    paths.append(p2)
    return paths


def run_test(model, bundle, out_dir: str, device: str = "auto", n_show: int = 480) -> dict:
    """在测试集上评估，并与持续性 / 日周期朴素基线对比。"""
    device = resolve_device(device)
    model.to(device).eval()

    X_test, y_test = bundle.test
    with torch.no_grad():
        pred_scaled = model(torch.as_tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    true_raw = inverse_transform_targets(bundle.target_scaler, y_test)
    model_raw = inverse_transform_targets(bundle.target_scaler, pred_scaled)

    horizon = true_raw.shape[1]
    period = bundle.meta.get("points_per_day", 96)
    base_persist = persistence_prediction(bundle.test_last_obs, horizon)
    base_seasonal = seasonal_naive_prediction(
        bundle.raw_series, bundle.test_starts, model.look_back, horizon, period
    )

    result = {
        "model": evaluate_predictions(true_raw, model_raw),
        "baseline_persistence": evaluate_predictions(true_raw, base_persist),
        "baseline_seasonal_naive": evaluate_predictions(true_raw, base_seasonal),
    }
    for key in ("baseline_persistence", "baseline_seasonal_naive"):
        result[f"skill_vs_{key.split('_', 1)[1]}"] = skill_score(
            result["model"]["overall"]["rmse"], result[key]["overall"]["rmse"]
        )

    # 结果导出：真实值 / 模型预测 / 基线，按 步长 标注
    rows = []
    for k in range(len(true_raw)):
        for h in range(horizon):
            rows.append({
                "sample": k,
                "horizon": h + 1,
                "真实值": true_raw[k, h],
                "预测值": model_raw[k, h],
                "持续性基线": base_persist[k, h],
                "日周期基线": base_seasonal[k, h],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False, encoding="utf-8-sig")
    df.to_excel(os.path.join(out_dir, "test_predictions.xlsx"), index=False)

    save_json(result, os.path.join(out_dir, "metrics.json"))
    result["plots"] = _plot_results(true_raw, model_raw, base_persist,
                                     result["model"]["per_horizon"],
                                     result["baseline_persistence"]["per_horizon"],
                                     out_dir, n_show)
    return result


def format_report(result: dict) -> str:
    """把指标整理成便于阅读的文本表格。"""
    def row(name, m):
        return (f"{name:<28s} R2={m['r2']:7.4f}  MAE={m['mae']:6.3f}  RMSE={m['rmse']:6.3f}  "
                f"nRMSE={m['nrmse']:6.4f}  非零时段MAPE={m['mape_nonzero']:6.2f}%  "
                f"负值占比={m['negative_ratio'] * 100:5.1f}%")

    lines = ["=" * 108, "整体指标（测试集，原始量纲）", "=" * 108]
    for name, key in [("本文模型", "model"),
                      ("持续性基线", "baseline_persistence"),
                      ("日周期朴素基线", "baseline_seasonal_naive")]:
        lines.append(row(name, result[key]["overall"]))
    lines.append(f"相对持续性基线技能得分 : {result['skill_vs_persistence']:+.4f}"
                 "（>0 表示优于基线）")
    lines.append(f"相对日周期基线技能得分 : {result['skill_vs_seasonal_naive']:+.4f}")
    lines.append("")
    lines.append("=" * 108)
    lines.append("分步长指标（本文模型）")
    lines.append("=" * 108)
    for m in result["model"]["per_horizon"]:
        lines.append(f"第 {m['horizon']} 步  R2={m['r2']:7.4f}  MAE={m['mae']:6.3f}  "
                     f"RMSE={m['rmse']:6.3f}  nRMSE={m['nrmse']:6.4f}  "
                     f"非零时段MAPE={m['mape_nonzero']:6.2f}%")
    return "\n".join(lines)
