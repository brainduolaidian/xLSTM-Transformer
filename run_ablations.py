# -*- coding: utf-8 -*-
"""消融实验：批量跑若干配置，并把结果汇总成一张对比表。

用法
----
    python run_ablations.py                     # 默认 3 组
    python run_ablations.py --epochs 30 --patience 8
    python run_ablations.py --force             # 即使已有结果也重跑

已有结果的配置会被直接复用（除非 --force），避免重复训练。
汇总表写入 results/ablation_summary.md 与 results/ablation_summary.csv。
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from run_train import main as train_main

HERE = os.path.dirname(os.path.abspath(__file__))

# (标签, 说明, 额外参数)
#
# 说明：默认只跑 mLSTM 相关的配置。若把 xlstm_layers 设成含 's' 的形式（例如 "msm"），
# sLSTM 因为存在隐状态之间的门控混合而无法并行，只能逐步递归，在 CPU 上会慢一个数量级，
# 需要时可自行执行：python run_train.py --xlstm-layers msm --tag xlstm_msm
ABLATIONS = [
    ("main", "完整模型（xLSTM 融进编码器 + 解码器）", []),
    ("no_xlstm", "去掉编码器内的 xLSTM", ["--no-xlstm"]),
    ("no_decoder", "去掉解码器分支", ["--no-decoder"]),
]

FIELDS = ["标签", "说明", "R2", "MAE", "RMSE", "nRMSE", "非零时段MAPE(%)",
          "负值占比(%)", "相对持续性技能得分"]


def load_result(out_root: str, tag: str):
    path = os.path.join(out_root, tag, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    model = r["model"]["overall"]
    return {
        "R2": round(model["r2"], 4),
        "MAE": round(model["mae"], 4),
        "RMSE": round(model["rmse"], 4),
        "nRMSE": round(model["nrmse"], 4),
        "非零时段MAPE(%)": round(model["mape_nonzero"], 2),
        "负值占比(%)": round(model["negative_ratio"] * 100, 2),
        "相对持续性技能得分": round(r["skill_vs_persistence"], 4),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="批量消融实验")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=os.path.join(HERE, "results"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", default=None, help="只跑指定标签，逗号分隔")
    args = p.parse_args()

    wanted = set(args.only.split(",")) if args.only else None
    rows = []
    for tag, desc, extra in ABLATIONS:
        if wanted and tag not in wanted:
            continue
        cached = None if args.force else load_result(args.out_dir, tag)
        if cached is None:
            print(f"\n===== 训练配置 {tag}：{desc} =====")
            argv = ["--epochs", str(args.epochs), "--patience", str(args.patience),
                    "--seed", str(args.seed), "--tag", tag, "--out-dir", args.out_dir] + extra
            train_main(argv)
            cached = load_result(args.out_dir, tag)
        if cached is None:
            print(f"跳过 {tag}：没有结果文件")
            continue
        order = ["标签", "说明"] + FIELDS[2:]
        row = {**cached, "标签": tag, "说明": desc}
        rows.append({k: row[k] for k in order})

    if not rows:
        return 1

    md_path = os.path.join(args.out_dir, "ablation_summary.md")
    csv_path = os.path.join(args.out_dir, "ablation_summary.csv")
    header = "| " + " | ".join(FIELDS) + " |"
    sep = "|" + "|".join(["---"] * len(FIELDS)) + "|"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(header + "\n" + sep + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in FIELDS) + " |\n")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("\n消融汇总：")
    for r in rows:
        print("  " + " | ".join(f"{k}={r[k]}" for k in FIELDS))
    print(f"\n已写入 {md_path} 与 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
