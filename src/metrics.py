# -*- coding: utf-8 -*-
"""评价指标与朴素基线。

相对于原始工程的修正
--------------------
1. 分预测步长（horizon）报告指标，不再把 4 个步长混成一行；
2. MAPE 只在真值非零的时段统计（夜间功率为 0 时 MAPE 无定义，原实现用 1e-6 兜底会失真）；
3. 额外给出以极差归一化的 nRMSE，以及超过零值占比的"有效时段"指标；
4. 内置持续性与日周期朴素基线，直接给出技能得分，方便判断模型是否真的有用。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "regression_metrics",
    "evaluate_predictions",
    "persistence_prediction",
    "seasonal_naive_prediction",
    "inverse_transform_targets",
]


def inverse_transform_targets(scaler, values: np.ndarray) -> np.ndarray:
    """把归一化后的预测/真值还原到原始量纲。支持 (N, horizon) 与 (N*horizon, 1)。"""
    flat = np.asarray(values).reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(np.asarray(values).shape)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """单组预测的指标。y_true / y_pred 形状需一致。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0:
        raise ValueError("空输入")

    err = y_pred - y_true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    rng = float(y_true.max() - y_true.min())
    nonzero = np.abs(y_true) > 1e-6

    out = {
        "n": int(y_true.size),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "nrmse": float(np.sqrt(np.mean(err ** 2)) / rng) if rng > 0 else float("nan"),
        "bias": float(np.mean(err)),
        "negative_ratio": float(np.mean(y_pred < 0)),
    }
    if nonzero.any():
        ape = np.abs(err[nonzero] / y_true[nonzero])
        out["mape_nonzero"] = float(np.mean(ape) * 100)
        out["mae_nonzero"] = float(np.mean(np.abs(err[nonzero])))
    else:
        out["mape_nonzero"] = float("nan")
        out["mae_nonzero"] = float("nan")
    return out


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """整体 + 分步长指标。y_true / y_pred: (N, horizon)"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"形状不一致：{y_true.shape} vs {y_pred.shape}")

    result = {"overall": regression_metrics(y_true, y_pred), "per_horizon": []}
    for h in range(y_true.shape[1]):
        m = regression_metrics(y_true[:, h], y_pred[:, h])
        m["horizon"] = h + 1
        result["per_horizon"].append(m)
    return result


def persistence_prediction(test_last_obs: np.ndarray, horizon: int) -> np.ndarray:
    """持续性基线：把窗口内最后一个观测值直接外推 horizon 步。"""
    return np.repeat(np.asarray(test_last_obs)[:, None], horizon, axis=1)


def seasonal_naive_prediction(raw_series: np.ndarray, test_starts: np.ndarray,
                              look_back: int, horizon: int, period: int) -> np.ndarray:
    """日周期朴素基线：取前一日同时刻的值。"""
    out = np.empty((len(test_starts), horizon), dtype=np.float64)
    for k, start in enumerate(test_starts):
        base = start + look_back - period
        if base < 0:
            out[k] = raw_series[start + look_back - 1]
        else:
            out[k] = raw_series[base:base + horizon]
    return out


def skill_score(model_rmse: float, baseline_rmse: float) -> float:
    """相对基线的技能得分 1 - RMSE_model / RMSE_baseline，正值表示优于基线。"""
    if baseline_rmse == 0:
        return float("nan")
    return float(1.0 - model_rmse / baseline_rmse)
