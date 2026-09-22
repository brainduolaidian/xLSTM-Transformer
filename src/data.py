# -*- coding: utf-8 -*-
"""数据读取、切分、归一化与滑窗。

相对于原始工程的修正
--------------------
1. 归一化只在**训练段**上拟合，再变换全量数据，避免测试集信息泄漏；
2. 自动识别并报告"与目标列完全相同"的特征列、特征列之间完全重复的列；
3. 可选加入日内时刻（time-of-day）正余弦特征，这是光伏出力最直接的时间驱动量；
4. 滑窗分别在训练/验证/测试子集内部构造，不跨子集取历史；
5. 同时返回窗口起点与原始序列，便于计算持续性等朴素基线；
6. 支持把 IFS / GFS 等预报文件按时间列自动合并进来（--extra-files），为多源输入留好接口。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

__all__ = ["DataConfig", "DatasetBundle", "load_dataframe", "prepare_data"]


@dataclass
class DataConfig:
    path: str
    sheet: Optional[str] = None
    target_col: Optional[str] = None          # None 表示取最后一列
    feature_cols: Optional[list] = None       # None 表示除目标外的全部数值列
    look_back: int = 96
    horizon: int = 4
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    points_per_day: int = 96                  # 采样点数/天，用于构造日内时刻特征
    add_time_of_day: bool = True
    time_col: Optional[str] = None            # 若数据带时间戳，可给出列名
    extra_files: list = field(default_factory=list)   # 额外数据文件（如 NWP 预报），按时间列左连接
    merge_on: Optional[str] = None            # 合并键，默认取 time_col
    notes: list = field(default_factory=list)


@dataclass
class DatasetBundle:
    train: tuple          # (X, y)
    val: tuple
    test: tuple
    feature_scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    feature_names: list
    raw_series: np.ndarray       # 未归一化的目标序列
    test_starts: np.ndarray      # 测试集每个窗口在原始序列中的起点
    test_last_obs: np.ndarray    # 测试集每个窗口的最后一个历史观测值
    meta: dict


def load_dataframe(cfg: DataConfig) -> pd.DataFrame:
    """读取 Excel，做基础的数值列检查。"""
    if not os.path.exists(cfg.path):
        raise FileNotFoundError(
            f"找不到数据文件：{cfg.path}\n"
            "本仓库不附带原始数据（版权原因），请按 DATA.md 的说明准备数据，"
            "或用 --data 指定你的数据文件路径。"
        )
    df = pd.read_excel(cfg.path, sheet_name=cfg.sheet if cfg.sheet is not None else 0)
    df.columns = [str(c).strip() for c in df.columns]

    if df.columns.duplicated().any():
        dup = list(df.columns[df.columns.duplicated()])
        df = df.loc[:, ~df.columns.duplicated()]
        cfg.notes.append(f"列名重复，已保留首列：{dup}")

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        if cfg.time_col is None and len(non_numeric) == 1:
            cfg.time_col = non_numeric[0]
            cfg.notes.append(f"检测到非数值列 {non_numeric}，作为时间列使用")
        drop_cols = [c for c in non_numeric if c != cfg.time_col]
        if drop_cols:
            cfg.notes.append(f"忽略非数值列：{drop_cols}")
            df = df.drop(columns=drop_cols)

    if len(df) == 0:
        raise ValueError("数据为空")
    if df.isna().any().any():
        n_na = int(df.isna().sum().sum())
        cfg.notes.append(f"存在 {n_na} 个缺失值，已按前向填充处理")
        df = df.ffill().bfill()
    return df


def _merge_extra_files(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """把额外数据（例如 IFS / GFS 数值天气预报）按时间列左连接进主表。

    合并进来的数值列会在后续流程里自动成为输入特征；重名列会加 `_nwp` 后缀。
    注意：这里只负责"按时间戳对齐"，预报的可用性语义（只能用预测时刻之前已发布的预报）
    需要你在准备数据时保证，详见 DATA.md 的"避免预报泄漏"一节。
    """
    key = cfg.merge_on or cfg.time_col
    if not key:
        raise ValueError("使用额外数据文件时必须指定时间列（--time-col）或 --merge-on 作为合并键")
    if key not in df.columns:
        raise ValueError(f"合并键 {key!r} 不在主表中，可选列：{list(df.columns)}")

    left = df.copy()
    left[key] = pd.to_datetime(left[key], errors="coerce")
    if left[key].isna().all():
        raise ValueError(f"主表合并键 {key!r} 无法解析为时间")
    n_rows = len(left)

    for path in cfg.extra_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到额外数据文件：{path}")
        if path.lower().endswith((".xlsx", ".xls")):
            extra = pd.read_excel(path)
        else:
            extra = pd.read_csv(path)
        extra.columns = [str(c).strip() for c in extra.columns]
        if key not in extra.columns:
            raise ValueError(f"{path} 中缺少合并键 {key!r}，可选列：{list(extra.columns)}")

        extra[key] = pd.to_datetime(extra[key], errors="coerce")
        extra = extra.dropna(subset=[key]).drop_duplicates(subset=[key])

        overlap = [c for c in extra.columns if c != key and c in left.columns]
        if overlap:
            cfg.notes.append(f"{path}：列 {overlap} 与主表重名，已加后缀 _nwp")
            extra = extra.rename(columns={c: f"{c}_nwp" for c in overlap})

        before = left.shape[1]
        new_cols = [c for c in extra.columns if c != key]
        left = left.merge(extra, on=key, how="left")
        if len(left) != n_rows:
            raise ValueError(f"{path} 含有重复时间戳，合并后行数发生变化")
        added = left.shape[1] - before
        if added and new_cols:
            match = float(left[new_cols[0]].notna().mean()) * 100
            cfg.notes.append(f"{path}：合并 {added} 列，时间戳匹配率 {match:.1f}%")

    na_after = int(left.isna().sum().sum())
    if na_after:
        cfg.notes.append(f"合并后出现 {na_after} 个缺失值，已按前向/后向填充处理")
        left = left.ffill().bfill()
    return left


def _resolve_columns(df: pd.DataFrame, cfg: DataConfig):
    target = cfg.target_col or df.columns[-1]
    if target not in df.columns:
        raise ValueError(f"目标列 {target!r} 不在数据中，可选：{list(df.columns)}")

    features = list(cfg.feature_cols) if cfg.feature_cols else [
        c for c in df.columns if c != target and c != cfg.time_col
    ]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"特征列不存在：{missing}")

    # 去掉彼此完全重复的特征列
    kept, dropped = [], []
    for c in features:
        if any(np.array_equal(df[c].values, df[k].values) for k in kept):
            dropped.append(c)
        else:
            kept.append(c)
    if dropped:
        cfg.notes.append(f"特征列之间完全重复，已去重：{dropped}")
    features = kept

    # 与目标列完全相同的特征列：等价于把目标自身的历史作为自回归输入，保留但提示
    same_as_target = [c for c in features if np.array_equal(df[c].values, df[target].values)]
    if same_as_target:
        cfg.notes.append(
            f"特征 {same_as_target} 与目标列 {target} 逐点完全相同，"
            "等价于把目标历史作为自回归输入；请确认原始数据是否缺少真正的驱动变量"
        )
    return target, features, same_as_target


def _add_time_features(df: pd.DataFrame, cfg: DataConfig) -> list:
    """加入日内时刻正余弦特征，返回新增列名。"""
    added = []
    if cfg.add_time_of_day:
        ppd = int(cfg.points_per_day)
        if ppd <= 1:
            raise ValueError("points_per_day 必须大于 1")
        idx = np.arange(len(df)) % ppd
        df["tod_sin"] = np.sin(2 * np.pi * idx / ppd)
        df["tod_cos"] = np.cos(2 * np.pi * idx / ppd)
        added += ["tod_sin", "tod_cos"]

    if cfg.time_col and cfg.time_col in df.columns:
        ts = pd.to_datetime(df[cfg.time_col], errors="coerce")
        if ts.notna().any():
            doy = ts.dt.dayofyear.fillna(1).values
            df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
            df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
            added += ["doy_sin", "doy_cos"]
        else:
            cfg.notes.append(f"时间列 {cfg.time_col} 无法解析为日期，跳过年内日期特征")
    return added


def _make_windows(X: np.ndarray, y: np.ndarray, starts: np.ndarray, look_back: int, horizon: int):
    xs, ys, idx = [], [], []
    for i in starts:
        if i + look_back + horizon > len(X):
            continue
        xs.append(X[i:i + look_back])
        ys.append(y[i + look_back:i + look_back + horizon])
        idx.append(i)
    if not xs:
        return np.empty((0, look_back, X.shape[1])), np.empty((0, horizon)), np.empty(0, dtype=int)
    return np.asarray(xs), np.asarray(ys), np.asarray(idx)


def prepare_data(cfg: DataConfig) -> DatasetBundle:
    """完整的数据准备流程。"""
    raw_df = load_dataframe(cfg)
    if cfg.extra_files:
        raw_df = _merge_extra_files(raw_df, cfg)
    target_col, feature_cols, same_as_target = _resolve_columns(raw_df, cfg)

    work = raw_df.copy()
    time_features = _add_time_features(work, cfg)
    feature_cols = list(feature_cols) + time_features
    if not feature_cols:
        raise ValueError("没有任何特征列")

    raw_series = work[target_col].values.astype(np.float64)
    n = len(work)
    train_end = int(n * cfg.train_ratio)
    val_end = train_end + int(n * cfg.val_ratio)
    if val_end >= n - cfg.look_back - cfg.horizon:
        raise ValueError("训练/验证/测试比例过大，剩余样本不足以构造滑窗")

    # 归一化：只用训练段拟合
    feat_scaler = MinMaxScaler(feature_range=(0, 1)).fit(work[feature_cols].values[:train_end])
    targ_scaler = MinMaxScaler(feature_range=(0, 1)).fit(raw_series[:train_end].reshape(-1, 1))
    X_all = feat_scaler.transform(work[feature_cols].values)
    y_all = targ_scaler.transform(raw_series.reshape(-1, 1)).ravel()

    segments = {
        "train": np.arange(0, train_end - cfg.look_back - cfg.horizon + 1, cfg.horizon),
        "val": np.arange(train_end, val_end - cfg.look_back - cfg.horizon + 1, cfg.horizon),
        "test": np.arange(val_end, n - cfg.look_back - cfg.horizon + 1, cfg.horizon),
    }
    windows = {k: _make_windows(X_all, y_all, v, cfg.look_back, cfg.horizon) for k, v in segments.items()}

    for k, (x, y, _) in windows.items():
        if len(x) == 0:
            raise ValueError(f"{k} 子集没有可用样本，请检查切分比例")

    # 测试集每个窗口的"最后一个历史观测值"，用于持续性基线（原始量纲）
    test_starts = windows["test"][2]
    test_last_obs = raw_series[test_starts + cfg.look_back - 1]

    meta = {
        "n_rows": n,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "feature_cols_same_as_target": same_as_target,
        "time_features": time_features,
        "train_end": train_end,
        "val_end": val_end,
        "samples": {k: int(len(v[0])) for k, v in windows.items()},
        "notes": cfg.notes,
        "points_per_day": cfg.points_per_day,
    }
    return DatasetBundle(
        train=windows["train"][:2],
        val=windows["val"][:2],
        test=windows["test"][:2],
        feature_scaler=feat_scaler,
        target_scaler=targ_scaler,
        feature_names=feature_cols,
        raw_series=raw_series,
        test_starts=test_starts,
        test_last_obs=test_last_obs,
        meta=meta,
    )
