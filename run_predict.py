# -*- coding: utf-8 -*-
"""独立推理脚本：加载已训练的权重，对测试集或最新窗口做预测。

用法
----
python run_predict.py                                  # 默认读 results/main
python run_predict.py --run-dir results/no_xlstm       # 指定某次实验
python run_predict.py --run-dir results/main --last-window   # 只用最后一段历史做一次预测

说明：模型结构和数据配置直接取自该次实验保存的 config.json，
因此不需要手动重复训练时的命令行参数。
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.data import DataConfig, prepare_data
from src.metrics import inverse_transform_targets
from src.model import XLSTMTransformer
from src.utils import resolve_device

HERE = os.path.dirname(os.path.abspath(__file__))


def build_from_config(cfg_path: str):
    with open(cfg_path, encoding="utf-8") as f:
        c = json.load(f)
    data_cfg = DataConfig(
        path=c["data"],
        sheet=c["sheet"],
        target_col=c["target_col"],
        look_back=c["look_back"],
        horizon=c["horizon"],
        train_ratio=c["train_ratio"],
        val_ratio=c["val_ratio"],
        points_per_day=c["points_per_day"],
        add_time_of_day=not c["no_time_of_day"],
        time_col=c["time_col"],
        extra_files=c.get("extra_files", []),
        merge_on=c.get("merge_on"),
    )
    bundle = prepare_data(data_cfg)
    model = XLSTMTransformer(
        num_features=len(bundle.feature_names),
        look_back=c["look_back"],
        horizon=c["horizon"],
        embed_dim=c["embed_dim"],
        dense_dim=c["dense_dim"],
        num_heads=c["num_heads"],
        num_blocks=c["num_blocks"],
        dropout=c["dropout"],
        use_xlstm=not c["no_xlstm"],
        use_decoder=not c["no_decoder"],
        xlstm_layers=c["xlstm_layers"],
        xlstm_heads=c["xlstm_heads"],
        xlstm_conv_axis=c["xlstm_conv_axis"],
        pos_encoding=c["pos_encoding"],
        clamp_min=None if c["no_clamp"] else 0.0,
        clamp_in_train=c["clamp_in_train"],
        head_bias=float(np.mean(bundle.train[1])),
    )
    return model, bundle, c


def main() -> int:
    p = argparse.ArgumentParser(description="加载权重做推理")
    p.add_argument("--run-dir", default=os.path.join(HERE, "results", "main"))
    p.add_argument("--ckpt", default=None, help="权重路径，默认 <run-dir>/best_model.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--last-window", action="store_true",
                   help="只对数据末尾的一个窗口做预测（模拟真实上线场景）")
    p.add_argument("--out", default=None, help="输出 csv 路径")
    args = p.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"找不到 {cfg_path}，请先运行 run_train.py")
    ckpt_path = args.ckpt or os.path.join(args.run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"找不到权重 {ckpt_path}")

    model, bundle, c = build_from_config(cfg_path)
    device = resolve_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"已加载权重（第 {ckpt.get('epoch')} 轮，验证 RMSE={ckpt.get('val_rmse'):.4f}）")

    if args.last_window:
        # 取测试集最后一个窗口，归一化参数与训练时完全一致
        X = bundle.test[0][-1:]
        with torch.no_grad():
            pred = model(torch.as_tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
        pred_raw = inverse_transform_targets(bundle.target_scaler, pred)[0]
        print("末尾窗口的预测（原始量纲）：", np.round(pred_raw, 4).tolist())
        out = args.out or os.path.join(args.run_dir, "last_window_prediction.csv")
        pd.DataFrame({"horizon": np.arange(1, len(pred_raw) + 1), "prediction": pred_raw}).to_csv(
            out, index=False, encoding="utf-8-sig")
        print(f"已写入 {out}")
        return 0

    X_test, y_test = bundle.test
    with torch.no_grad():
        pred = model(torch.as_tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    pred_raw = inverse_transform_targets(bundle.target_scaler, pred)
    true_raw = inverse_transform_targets(bundle.target_scaler, y_test)
    out = args.out or os.path.join(args.run_dir, "test_predictions_inference.csv")
    rows = [{"sample": k, "horizon": h + 1, "真实值": true_raw[k, h], "预测值": pred_raw[k, h]}
            for k in range(len(pred_raw)) for h in range(pred_raw.shape[1])]
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"测试集 {len(pred_raw)} 个窗口已预测完成，结果写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
