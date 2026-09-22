# -*- coding: utf-8 -*-
"""训练 + 测试入口。

示例
----
python run_train.py                                  # 默认配置，结果写入 results/
python run_train.py --epochs 60 --tag run60          # 换实验标签
python run_train.py --no-xlstm --tag ablation_wo_xlstm   # 消融：去掉 xLSTM
python run_train.py --no-decoder --tag ablation_no_decoder
python run_train.py --extra-files gfs.csv ifs.csv --time-col time   # 引入 NWP 预报
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from src.data import DataConfig, prepare_data
from src.evaluate import format_report, plot_history, run_test
from src.model import XLSTMTransformer
from src.trainer import TrainConfig, train_model
from src.utils import ensure_dir, save_json

HERE = os.path.dirname(os.path.abspath(__file__))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="xLSTM-Transformer 光伏出力多步预测")
    p.add_argument("--data", default=os.path.join(HERE, "data", "dataset.xlsx"), help="数据文件路径")
    p.add_argument("--sheet", default=None, help="工作表名，默认第一张")
    p.add_argument("--target-col", default=None, help="目标列名，默认最后一列")
    p.add_argument("--time-col", default=None, help="时间戳列名（可选）")
    p.add_argument("--extra-files", nargs="*", default=[],
                   help="额外数据文件（如 IFS/GFS 数值天气预报，可多个），按时间列左连接主表")
    p.add_argument("--merge-on", default=None, help="额外数据的合并键，默认与 --time-col 相同")
    p.add_argument("--no-time-of-day", action="store_true", help="不加日内时刻正余弦特征")
    p.add_argument("--points-per-day", type=int, default=96, help="每天采样点数，默认 96（15 分钟）")

    p.add_argument("--look-back", type=int, default=96, help="历史窗口长度")
    p.add_argument("--horizon", type=int, default=4, help="预测步数")
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.1)

    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--dense-dim", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--xlstm-layers", default="m", help="xLSTM 层序列，如 'm'、'msm'、's'")
    p.add_argument("--xlstm-heads", type=int, default=None, help="xLSTM 头数，默认与注意力一致")
    p.add_argument("--xlstm-conv-axis", default="time", choices=["time", "feature"])
    p.add_argument("--pos-encoding", default="learned", choices=["learned", "sinusoidal"])
    p.add_argument("--no-xlstm", action="store_true", help="消融：编码器内不使用 xLSTM")
    p.add_argument("--no-decoder", action="store_true", help="消融：去掉解码器分支")
    p.add_argument("--no-clamp", action="store_true", help="推理时也不对输出做非负约束")
    p.add_argument("--clamp-in-train", action="store_true",
                   help="训练时也截断到非负（不推荐：会让输出卡在 0 上）")

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--tag", default="xlstm_transformer", help="本次实验标签，决定子目录名")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = ensure_dir(os.path.join(args.out_dir, args.tag))

    data_cfg = DataConfig(
        path=args.data,
        sheet=args.sheet,
        target_col=args.target_col,
        look_back=args.look_back,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        points_per_day=args.points_per_day,
        add_time_of_day=not args.no_time_of_day,
        time_col=args.time_col,
        extra_files=args.extra_files,
        merge_on=args.merge_on,
    )
    bundle = prepare_data(data_cfg)
    print("数据概况：")
    for note in bundle.meta["notes"]:
        print("  -", note)
    print(f"  目标列={bundle.meta['target_col']}  特征列={bundle.meta['feature_cols']}")
    print(f"  样本数={bundle.meta['samples']}  总行数={bundle.meta['n_rows']}")

    model = XLSTMTransformer(
        num_features=len(bundle.feature_names),
        look_back=args.look_back,
        horizon=args.horizon,
        embed_dim=args.embed_dim,
        dense_dim=args.dense_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        use_xlstm=not args.no_xlstm,
        use_decoder=not args.no_decoder,
        xlstm_layers=args.xlstm_layers,
        xlstm_heads=args.xlstm_heads,
        xlstm_conv_axis=args.xlstm_conv_axis,
        pos_encoding=args.pos_encoding,
        clamp_min=None if args.no_clamp else 0.0,
        clamp_in_train=args.clamp_in_train,
        head_bias=float(np.mean(bundle.train[1])),   # 输出偏置初始化到目标均值
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{n_params:,}")

    save_json({k: v for k, v in vars(args).items()}, os.path.join(out_dir, "config.json"))
    save_json(bundle.meta, os.path.join(out_dir, "data_meta.json"))

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        verbose=not args.quiet,
    )
    t0 = time.time()
    history = train_model(model, bundle, train_cfg, out_dir)
    elapsed = time.time() - t0
    history["elapsed_sec"] = elapsed
    history["n_params"] = n_params
    save_json(history, os.path.join(out_dir, "history.json"))
    plot_history(history, out_dir)
    print(f"训练耗时 {elapsed / 60:.2f} 分钟，最优轮次 {history['best']['epoch']}"
          f"（验证 RMSE={history['best']['rmse']:.4f}）")

    ckpt = torch.load(history["best"]["path"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    result = run_test(model, bundle, out_dir, device=args.device)
    print()
    print(format_report(result))
    save_json(result, os.path.join(out_dir, "metrics.json"))
    print(f"\n结果目录：{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
