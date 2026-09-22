# -*- coding: utf-8 -*-
"""训练流程：随机种子、整轮平均损失、梯度裁剪、学习率调度、早停与最优权重保存。

相对于原始工程的修正
--------------------
1. 每个 epoch 记录的是**整轮平均**损失，而不是最后一个 batch 的损失（原实现的曲线不可读）；
2. 以验证集 RMSE（还原到原始量纲）选择最优模型并 torch.save，而不是直接用最后一轮权重；
3. 加入梯度裁剪（xLSTM 含指数门，长序列下有溢出风险）与学习率衰减、早停；
4. 固定随机种子，产出可复现的历史记录。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import inverse_transform_targets, regression_metrics
from .utils import resolve_device, set_seed

__all__ = ["TrainConfig", "train_model"]


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 10
    min_delta: float = 1e-6
    device: str = "auto"
    seed: int = 42
    verbose: bool = True


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train_model(model, bundle, cfg: TrainConfig, out_dir: str) -> dict:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model.to(device)

    train_loader = _loader(*bundle.train, cfg.batch_size, shuffle=True)
    val_loader = _loader(*bundle.val, cfg.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, cfg.patience // 3), min_lr=1e-5
    )

    history = {"train_loss": [], "val_loss": [], "val_rmse": [], "lr": []}
    best = {"rmse": float("inf"), "epoch": -1, "path": None}
    best_path = os.path.join(out_dir, "best_model.pt")
    wait = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n_batch = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += loss.item()
            n_batch += 1
        train_loss = total / max(n_batch, 1)

        val_pred, val_true = _predict(model, val_loader, device)
        val_loss = float(np.mean((val_pred - val_true) ** 2))
        val_pred_raw = inverse_transform_targets(bundle.target_scaler, val_pred)
        val_true_raw = inverse_transform_targets(bundle.target_scaler, val_true)
        val_rmse = regression_metrics(val_true_raw, val_pred_raw)["rmse"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        if cfg.verbose:
            print(f"[epoch {epoch:3d}/{cfg.epochs}] train_loss={train_loss:.6f} "
                  f"val_loss={val_loss:.6f} val_rmse={val_rmse:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_rmse < best["rmse"] - cfg.min_delta:
            best = {"rmse": val_rmse, "epoch": epoch, "path": best_path}
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_rmse": val_rmse},
                       best_path)
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                if cfg.verbose:
                    print(f"验证集 {cfg.patience} 轮无改善，提前停止于第 {epoch} 轮")
                break

    history["best"] = best
    history["device"] = str(device)
    return history
