# -*- coding: utf-8 -*-
"""数据流程测试：切分、归一化防泄漏、时间特征、列去重、额外数据（NWP 预报）合并。

运行：python tests/test_data_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataConfig, prepare_data  # noqa: E402


def _make_frame(n: int = 1200, points_per_day: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    tod = idx % points_per_day
    power = np.maximum(0.0, np.sin(np.pi * tod / points_per_day)) * 5.0 + rng.normal(0, 0.05, n)
    power = np.clip(power, 0, None)
    temp = np.where(idx < int(n * 0.9), rng.uniform(0, 10, n), rng.uniform(100, 200, n))  # 后段极值
    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({"time": ts, "power": power, "temp": temp, "wind": power.copy()})


def _run_test() -> None:
    # 1. 时间列自动识别 + 时间特征 + 归一化只用训练段
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "synthetic.xlsx")
        _make_frame().to_excel(path, index=False)
        cfg = DataConfig(path=path, target_col="power", look_back=24, horizon=4,
                         train_ratio=0.7, val_ratio=0.1)
        bundle = prepare_data(cfg)

        assert cfg.time_col == "time", f"未识别出时间列：{cfg.time_col}"
        assert "tod_sin" in bundle.feature_names and "doy_sin" in bundle.feature_names, \
            f"缺少时间特征：{bundle.feature_names}"
        assert "time" not in bundle.feature_names, "时间列不应作为数值特征"

        # wind 与 power 逐点相同，应被识别为"与目标相同的特征"
        assert "wind" in bundle.meta["feature_cols_same_as_target"], \
            f"未识别与目标相同的特征：{bundle.meta['feature_cols_same_as_target']}"

        # 2. 归一化防泄漏：temp 的测试段是 100~200，`fit` 只能看训练段
        j = bundle.feature_names.index("temp")
        assert bundle.feature_scaler.data_max_[j] <= 10.5, \
            f"归一化看到了测试段极值，data_max_={bundle.feature_scaler.data_max_[j]}"

        # 3. 切分顺序与不重叠
        n = bundle.meta["n_rows"]
        train_end, val_end = bundle.meta["train_end"], bundle.meta["val_end"]
        assert 0 < train_end < val_end < n
        assert bundle.test_starts.min() >= val_end, "测试窗口侵入了验证段"
        assert bundle.meta["samples"]["train"] > 0 and bundle.meta["samples"]["test"] > 0

        # 4. 滑窗形状与目标对齐
        X_tr, y_tr = bundle.train
        assert X_tr.shape[1] == 24 and X_tr.shape[2] == len(bundle.feature_names)
        assert y_tr.shape[1] == 4
        assert X_tr.min() >= -1e-6 and X_tr.max() <= 1 + 1e-6, "特征归一化超出 [0,1]"

    # 5. 特征与目标完全相同时给出提示（真实数据集走的就是这条路径）
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dup.xlsx")
        df = pd.DataFrame({"Speed": np.arange(300.0), "Direction": np.arange(300.0) * 0.1,
                           "X": np.arange(300.0)})
        df.to_excel(path, index=False)
        cfg = DataConfig(path=path, look_back=16, horizon=4, add_time_of_day=False)
        bundle = prepare_data(cfg)
        assert "Speed" in bundle.meta["feature_cols_same_as_target"]
        assert "Direction" not in bundle.meta["feature_cols_same_as_target"]
        assert any("逐点完全相同" in s for s in bundle.meta["notes"])

    print("  数据流程（切分/防泄漏/时间特征/重复列） : PASS")


def _run_extra_files_test() -> None:
    """额外数据文件（模拟 IFS/GFS 预报）按时间列合并后应自动成为特征。"""
    with tempfile.TemporaryDirectory() as d:
        main_path = os.path.join(d, "power.xlsx")
        nwp_path = os.path.join(d, "nwp_forecast.csv")
        frame = _make_frame(600)
        frame[["time", "power", "temp"]].to_excel(main_path, index=False)

        # 预报只覆盖前 70% 的时间戳（模拟发布频率低于数据频率），且有一列与主表重名
        nwp = pd.DataFrame({
            "time": frame["time"].iloc[:420],
            "ssrd": np.linspace(0, 800, 420),
            "temp": np.linspace(0, 30, 420),          # 与主表重名，应加后缀
        })
        nwp.to_csv(nwp_path, index=False)

        cfg = DataConfig(path=main_path, target_col="power", time_col="time",
                         look_back=24, horizon=4, add_time_of_day=False,
                         extra_files=[nwp_path], merge_on="time")
        bundle = prepare_data(cfg)

        assert "ssrd" in bundle.feature_names, f"预报列未成为特征：{bundle.feature_names}"
        assert "temp_nwp" in bundle.feature_names, f"重名列未加后缀：{bundle.feature_names}"
        assert bundle.meta["n_rows"] == 600, "合并后行数发生变化"
        joined = [s for s in bundle.meta["notes"] if "匹配率" in s]
        assert joined, f"没有输出匹配率提示：{bundle.meta['notes']}"
        assert bundle.train[0].shape[2] == len(bundle.feature_names)

        # 主表没有可用时间列时，应给出明确的报错
        no_time_path = os.path.join(d, "no_time.xlsx")
        frame[["power", "temp"]].to_excel(no_time_path, index=False)
        cfg_bad = DataConfig(path=no_time_path, target_col="power", look_back=24, horizon=4,
                             add_time_of_day=False, extra_files=[nwp_path])
        try:
            prepare_data(cfg_bad)
            raise AssertionError("缺少合并键时本应报错")
        except ValueError as e:
            assert "--time-col" in str(e) or "--merge-on" in str(e), f"报错信息不够明确：{e}"

    print("  额外数据文件合并（NWP 预报接口） : PASS")


if __name__ == "__main__":
    _run_test()
    _run_extra_files_test()
