# -*- coding: utf-8 -*-
"""xLSTM 与模型结构的回归测试。

运行方式
--------
    python tests/test_xlstm_equivalence.py        # 不依赖 pytest
    pytest tests/ -v                              # 若已安装 pytest

覆盖内容
--------
1. mLSTM 的并行实现与论文式(1)-(6)的逐步递归实现在数值上一致；
2. 多头维度没有被拍平（状态形状为 (B, H, d_h, d_h)）；
3. 前向形状正确、梯度可回传且无 NaN；
4. 关键回归：同一批输入下，单条样本的预测不随同批其他样本变化
   （原工程 nn.MultiheadAttention 未设 batch_first，导致跨样本信息串扰）；
5. 注意力矩阵落在时间维上，形状为 (B, T, T)。
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import XLSTMTransformer  # noqa: E402
from src.xlstm import mLSTM, sLSTM, xLSTM  # noqa: E402

torch.manual_seed(0)


def test_mlstm_parallel_matches_recurrent():
    """并行形式与递归形式必须一致（这是"矩阵记忆 + 稳定化门控"写对与否的判据）。"""
    for conv_axis in ("time", "feature"):
        block = mLSTM(input_size=8, head_size=4, num_heads=2, conv_axis=conv_axis).eval()
        x = torch.randn(3, 11, 8)
        with torch.no_grad():
            fast = block(x)
            slow, state = block.forward_recurrent(x)
        diff = (fast - slow).abs().max().item()
        assert diff < 1e-5, f"conv_axis={conv_axis} 并行与递归结果不一致，最大差 {diff}"
        m, C, n, _ = state
        assert C.shape == (3, 2, 4, 4), f"矩阵记忆形状错误：{C.shape}"
        assert n.shape == (3, 2, 4), f"归一化状态形状错误：{n.shape}"
        assert m.shape == (3, 2), f"稳定化状态形状错误：{m.shape}"
        assert torch.isfinite(C).all() and torch.isfinite(n).all()
    print("  并行/递归一致性 : PASS")


def test_mlstm_shapes_gradients_and_stability():
    block = mLSTM(input_size=6, head_size=3, num_heads=2)
    x = torch.randn(4, 9, 6, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    out.pow(2).mean().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} 没有梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度出现 NaN/Inf"

    # 门控饱和测试：输入放大到 1e3 也不应产生 NaN
    big = block(torch.randn(2, 9, 6) * 1000.0)
    assert torch.isfinite(big).all(), "大输入下出现 NaN/Inf"
    print("  形状/梯度/数值稳定 : PASS")


def test_slstm_runs():
    block = sLSTM(input_size=6, head_size=3, num_heads=2)
    x = torch.randn(2, 8, 6)
    out = block(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.sum().backward()
    print("  sLSTM 前向/反向   : PASS")


def test_xlstm_stack_residual():
    layer = xLSTM(input_size=8, head_size=4, num_heads=2, layers="msm")
    x = torch.randn(2, 6, 8)
    out, _ = layer(x)
    assert out.shape == x.shape
    print("  xLSTM 层堆叠      : PASS")


def test_no_cross_sample_leakage():
    """回归测试：同一输入样本的预测不应受同批其他样本影响。"""
    model = XLSTMTransformer(num_features=5, look_back=16, horizon=4, embed_dim=16,
                             dense_dim=32, num_heads=4, num_blocks=2, dropout=0.0).eval()
    x1 = torch.randn(4, 16, 5)
    x2 = x1.clone()
    x2[1:] = torch.randn(3, 16, 5)          # 只替换后 3 条，第 0 条保持不变
    with torch.no_grad():
        o1 = model(x1)[0]
        o2 = model(x2)[0]
    diff = (o1 - o2).abs().max().item()
    assert diff == 0.0, f"第 0 条样本的预测随同批样本变化，存在跨样本串扰（差值 {diff}）"
    print("  无跨样本串扰      : PASS")


def test_attention_is_over_time():
    model = XLSTMTransformer(num_features=5, look_back=20, horizon=2, embed_dim=16,
                             dense_dim=32, num_heads=4, num_blocks=1, dropout=0.0).eval()
    x = torch.randn(3, 20, 5)
    maps = model.attention_maps(x)
    assert maps, "未取到注意力权重"
    w = maps[0]
    assert w.shape[0] == 3 and w.shape[-2:] == (20, 20), \
        f"注意力形状应为 (batch=3, T=20, T=20)，实际 {tuple(w.shape)}"
    print("  注意力在时间维    : PASS")


def main() -> int:
    print("开始运行测试...")
    tests = [
        test_mlstm_parallel_matches_recurrent,
        test_mlstm_shapes_gradients_and_stability,
        test_slstm_runs,
        test_xlstm_stack_residual,
        test_no_cross_sample_leakage,
        test_attention_is_over_time,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  {t.__name__} : FAIL -> {e}")
    print(f"\n共 {len(tests)} 项，失败 {failed} 项")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
