# -*- coding: utf-8 -*-
"""xLSTM（sLSTM / mLSTM）模块。

公式依据
--------
[1] Beck, M. et al. "xLSTM: Extended Long Short-Term Memory." arXiv:2405.04517
[2] Beck, M. et al. "Tiled Flash Linear Attention: More Efficient Linear RNN and
    xLSTM Kernels." arXiv:2503.14376, Sec. 2.1 式 (1)-(6)

mLSTM 的递归式（式 1-6）
------------------------
    m_t = max( log sigma(f~_t) + m_{t-1}, i~_t )                      (1)
    C_t = f_t C_{t-1} + i_t k_t v_t^T                                 (2)
    n_t = f_t n_{t-1} + i_t k_t                                       (3)
    h~_t = C_t^T (q_t / sqrt(d_qk)) / max( |n_t^T (q_t / sqrt(d_qk))|,
                                           exp(-m_t) )                 (4)
    h_t = o_t * NORM(h~_t)                                            (5)
    f_t = exp( log sigma(f~_t) + m_{t-1} - m_t )                      (6)

本实现把 (1)-(6) 展开成并行形式，一次算出整条序列，不需要在 Python 里逐步循环：

    记 ell_t = sum_{u<=t} log sigma(f~_u)，则遗忘门的连乘为 exp(ell_t - ell_s)，
    于是  C_t = exp(-m_t) * sum_{s<=t} w_{t,s} k_s v_s^T，
          n_t = exp(-m_t) * sum_{s<=t} w_{t,s} k_s，
    其中  w_{t,s} = exp(ell_t - ell_s + i~_s)，而 m_t 恰好等于 log w_{t,·} 在 s<=t 上的最大值，
    与式 (1) 的递归最大值严格等价（可展开验证）。
    exp(-m_t) 在 h~_t 的分子分母中相互抵消，最终只需 max(., exp(-m_t)) 这一个除零保护，
    所以并行写法与递归写法在数学上完全一致（见 tests/test_xlstm_equivalence.py）。

关于"因果卷积"
--------------
块内输入先做一次因果 1D 卷积做局部混合，核长 4。本实现默认沿时间轴做深度可分离卷积
（不会看到未来时刻）；把 conv_axis 设为 "feature" 可以复现"卷积作用在特征维"的旧行为，
方便对照实验。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CausalConv1d", "mLSTM", "sLSTM", "xLSTM"]


class CausalConv1d(nn.Module):
    """因果 1D 卷积。

    axis="time"    沿时间轴做深度可分离卷积（默认，不会泄漏未来信息）。
    axis="feature" 沿特征维卷积（复现旧实现的行为，仅用于对照实验）。
    """

    def __init__(self, channels: int, kernel_size: int = 4, axis: str = "time"):
        super().__init__()
        if axis not in ("time", "feature"):
            raise ValueError("axis 只能是 'time' 或 'feature'")
        self.channels = channels
        self.kernel_size = kernel_size
        self.axis = axis
        self.pad = kernel_size - 1
        if axis == "time":
            self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels)
        else:
            self.conv = nn.Conv1d(1, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.axis == "time":
            y = x.transpose(1, 2)                     # (B, C, T)
            y = F.pad(y, (self.pad, 0))               # 只在左侧补齐，保证因果
            return self.conv(y).transpose(1, 2)       # (B, T, C)
        # feature 轴：逐时间步在特征维上卷积（复现旧实现的行为）
        B, T, C = x.shape
        y = x.reshape(B * T, 1, C)
        y = F.pad(y, (self.pad, 0))
        return self.conv(y).reshape(B, T, C)


class mLSTM(nn.Module):
    """矩阵记忆的 mLSTM 块（并行实现）。

    参数
    ----
    input_size   输入特征维度
    head_size    每个头的 q/k/v 维度
    num_heads    头数；门 i/f/o 是"每个头一个标量"，与论文一致
    proj_factor  输入升维比例（默认 2）
    conv_kernel  因果卷积核长
    conv_axis    "time"（默认）或 "feature"
    dropout      块输出 dropout
    input_gate_bias  输入门偏置初值（论文建议取较大负值以稳定训练）
    """

    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        proj_factor: float = 2.0,
        conv_kernel: int = 4,
        conv_axis: str = "time",
        dropout: float = 0.0,
        input_gate_bias: float = -2.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert proj_factor > 0
        self.input_size = input_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.hidden_size = head_size * num_heads
        self.proj_factor = proj_factor
        self.dropout_p = dropout
        inner = int(input_size * proj_factor)

        self.norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, inner)          # 门控 / 卷积 / skip 用
        self.up_proj_right = nn.Linear(input_size, self.hidden_size)  # 输出门控用
        self.conv = CausalConv1d(inner, conv_kernel, axis=conv_axis)
        self.skip = nn.Linear(inner, self.hidden_size)

        self.Wq = nn.Linear(inner, self.hidden_size)
        self.Wk = nn.Linear(inner, self.hidden_size)
        self.Wv = nn.Linear(inner, self.hidden_size)
        # 门是每个头一个标量（论文：exponential gating with scalar gates per head）
        self.Wi = nn.Linear(inner, num_heads)
        self.Wf = nn.Linear(inner, num_heads)
        self.Wo = nn.Linear(inner, num_heads)

        self.group_norm = nn.GroupNorm(num_heads, self.hidden_size, eps=norm_eps)
        self.down_proj = nn.Linear(self.hidden_size, input_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.constant_(self.Wi.bias, input_gate_bias)          # 论文建议的初始化
        nn.init.zeros_(self.Wo.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, hd = self.num_heads, self.head_size

        x_norm = self.norm(x)
        x_left = F.silu(self.conv(self.up_proj_left(x_norm)))      # (B,T,inner)
        x_right = self.up_proj_right(x_norm)                       # (B,T,hidden)

        q = self.Wq(x_left).view(B, T, H, hd)
        k = self.Wk(x_left).view(B, T, H, hd) * hd ** -0.5         # 1/sqrt(d_qk) 折进 k
        v = self.Wv(x_left).view(B, T, H, hd)

        i_tilde = self.Wi(x_left)                                  # (B,T,H) 逐头标量
        f_tilde = self.Wf(x_left)
        o = torch.sigmoid(self.Wo(x_left))

        # 式(1) 的展开：m_t 等于 log_w[t, s] 在 s<=t 上的最大值
        log_f = F.logsigmoid(f_tilde)                              # <= 0
        ell = torch.cumsum(log_f, dim=1)                           # (B,T,H)
        # log_w[b, t, s, h] = ell[b,t,h] - ell[b,s,h] + i~[b,s,h]
        log_w = ell[:, :, None, :] - ell[:, None, :, :] + i_tilde[:, None, :, :]
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        log_w = log_w.masked_fill(~causal[None, :, :, None], float("-inf"))
        m = log_w.amax(dim=2, keepdim=True)                        # (B,T,1,H)
        w = torch.exp(log_w - m)                                   # 稳定化权重，<=1，不会溢出

        # 式(2)(3)(4)：一次算出所有时间步
        s = torch.einsum("bthd,bshd->btsh", q, k)                  # k_s · q_t
        num = torch.einsum("btsh,btsh,bshd->bthd", w, s, v)        # C_t^T q_t 的等比例量
        den = torch.einsum("btsh,btsh->bth", w, s).abs()           # |n_t^T q_t| 的等比例量

        floor = torch.exp(-m.squeeze(2))                           # exp(-m_t)
        h = o.unsqueeze(-1) * num / torch.clamp(den, min=floor).unsqueeze(-1)
        h = h.reshape(B, T, self.hidden_size)

        # 式(5)：按头的归一化
        h = self.group_norm(h.reshape(B * T, self.hidden_size)).view(B, T, self.hidden_size)

        out = h + self.skip(x_left)
        out = out * F.silu(x_right)
        out = self.down_proj(out)
        return self.dropout(out) + x

    @torch.no_grad()
    def forward_recurrent(self, x: torch.Tensor):
        """逐步递归实现，只用于与并行实现做数值等价性校验（速度慢）。

        返回 (输出, 末状态)；末状态为 (m, C, n, h) 以便检查。
        """
        B, T, _ = x.shape
        H, hd = self.num_heads, self.head_size
        x_norm = self.norm(x)
        x_left = F.silu(self.conv(self.up_proj_left(x_norm)))
        x_right = self.up_proj_right(x_norm)

        q = self.Wq(x_left).view(B, T, H, hd)
        k = self.Wk(x_left).view(B, T, H, hd) * hd ** -0.5
        v = self.Wv(x_left).view(B, T, H, hd)
        i_tilde = self.Wi(x_left)
        f_tilde = self.Wf(x_left)
        o = torch.sigmoid(self.Wo(x_left))

        m_prev = torch.full((B, H), float("-inf"), device=x.device, dtype=x.dtype)
        C = torch.zeros(B, H, hd, hd, device=x.device, dtype=x.dtype)   # 矩阵记忆
        n = torch.zeros(B, H, hd, device=x.device, dtype=x.dtype)       # 归一化状态
        outs = []
        for t in range(T):
            m_t = torch.maximum(F.logsigmoid(f_tilde[:, t]) + m_prev, i_tilde[:, t])
            i_t = torch.exp(i_tilde[:, t] - m_t)
            f_t = torch.exp(F.logsigmoid(f_tilde[:, t]) + m_prev - m_t)
            C = f_t[..., None, None] * C + i_t[..., None, None] * (
                k[:, t][..., :, None] * v[:, t][..., None, :]
            )
            n = f_t[..., None] * n + i_t[..., None] * k[:, t]
            num = torch.einsum("bhij,bhi->bhj", C, q[:, t])             # C_t^T q_t
            den = torch.einsum("bhi,bhi->bh", n, q[:, t]).abs()         # |n_t^T q_t|
            h = o[:, t].unsqueeze(-1) * num / torch.clamp(
                den, min=torch.exp(-m_t)
            ).unsqueeze(-1)
            outs.append(h)
            m_prev = m_t
        h_seq = torch.stack(outs, dim=1).reshape(B, T, self.hidden_size)
        h_seq = self.group_norm(
            h_seq.reshape(B * T, self.hidden_size)
        ).view(B, T, self.hidden_size)
        out = h_seq + self.skip(x_left)
        out = out * F.silu(x_right)
        out = self.down_proj(out)
        out = self.dropout(out) + x
        return out, (m_prev, C, n, h_seq)


class sLSTM(nn.Module):
    """标量记忆的 sLSTM 块。

    这里的"标量记忆"指细胞状态是向量（逐元素更新，c_t = f_t*c_{t-1} + i_t*z_t），
    与 mLSTM 的矩阵记忆相对；门与状态都在 d 维上逐元素计算。
    门之间通过隐状态 h_{t-1} 相互混合（R 矩阵），因此无法像 mLSTM 那样并行，
    只能逐步递归，速度明显更慢。默认配置只用 mLSTM；需要 sLSTM 时显式指定 layers。
    """

    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        proj_factor: float = 4.0 / 3.0,
        conv_kernel: int = 4,
        conv_axis: str = "time",
        dropout: float = 0.0,
        input_gate_bias: float = -2.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert proj_factor > 0
        self.input_size = input_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.hidden_size = head_size * num_heads
        self.proj_factor = proj_factor
        inner = int(input_size * proj_factor)

        self.norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, inner)
        self.up_proj_right = nn.Linear(input_size, self.hidden_size)
        self.conv = CausalConv1d(inner, conv_kernel, axis=conv_axis)
        self.skip = nn.Linear(inner, self.hidden_size)

        self.Wz = nn.Linear(inner, self.hidden_size)
        self.Rz = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wi = nn.Linear(inner, self.hidden_size)
        self.Wf = nn.Linear(inner, self.hidden_size)
        self.Wo = nn.Linear(inner, self.hidden_size)
        self.Ri = nn.Linear(self.hidden_size, self.hidden_size)
        self.Rf = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ro = nn.Linear(self.hidden_size, self.hidden_size)

        self.group_norm = nn.GroupNorm(num_heads, self.hidden_size, eps=norm_eps)
        self.down_proj = nn.Linear(self.hidden_size, input_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.constant_(self.Wi.bias, input_gate_bias)
        nn.init.zeros_(self.Wo.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H = self.num_heads

        x_norm = self.norm(x)
        x_left = F.silu(self.conv(self.up_proj_left(x_norm)))
        x_right = self.up_proj_right(x_norm)

        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)  # 标量（逐元素）记忆
        n = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)  # 归一化状态
        m = torch.full((B, self.hidden_size), float("-inf"), device=x.device, dtype=x.dtype)

        outs = []
        for t in range(T):
            z = torch.tanh(self.Wz(x_left[:, t]) + self.Rz(h))
            i_tilde = self.Wi(x_left[:, t]) + self.Ri(h)
            f_tilde = self.Wf(x_left[:, t]) + self.Rf(h)
            o = torch.sigmoid(self.Wo(x_left[:, t]) + self.Ro(h))

            m_new = torch.maximum(F.logsigmoid(f_tilde) + m, i_tilde)
            i_t = torch.exp(i_tilde - m_new)
            f_t = torch.exp(F.logsigmoid(f_tilde) + m - m_new)
            c = f_t * c + i_t * z
            n = f_t * n + i_t
            h = o * c / torch.clamp(n.abs(), min=torch.exp(-m_new))
            outs.append(h)
            m = m_new

        h_seq = torch.stack(outs, dim=1)
        h_seq = self.group_norm(
            h_seq.reshape(B * T, self.hidden_size)
        ).view(B, T, self.hidden_size)

        out = h_seq + self.skip(x_left)
        out = out * F.silu(x_right)
        out = self.down_proj(out)
        return self.dropout(out) + x


class xLSTM(nn.Module):
    """按 layers 字符串堆叠 xLSTM 层。

    layers 例如 "m"（只有 mLSTM，默认）、"msm"（mLSTM/sLSTM/mLSTM 交替）、"s"。
    与旧实现保持相同的调用形式：forward(x) -> (x, state)。
    """

    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        layers: str = "m",
        proj_factor_slstm: float = 4.0 / 3.0,
        proj_factor_mlstm: float = 2.0,
        conv_kernel: int = 4,
        conv_axis: str = "time",
        dropout: float = 0.0,
        input_gate_bias: float = -2.0,
    ):
        super().__init__()
        if not layers or any(c not in "sm" for c in layers):
            raise ValueError("layers 只能由 's' 和 'm' 组成，例如 'm'、'msm'")
        self.input_size = input_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.hidden_size = head_size * num_heads
        self.layers = layers
        self.num_layers = len(layers)

        blocks = []
        for layer_type in layers:
            if layer_type == "s":
                blocks.append(sLSTM(input_size, head_size, num_heads, proj_factor_slstm,
                                    conv_kernel, conv_axis, dropout, input_gate_bias))
            else:
                blocks.append(mLSTM(input_size, head_size, num_heads, proj_factor_mlstm,
                                    conv_kernel, conv_axis, dropout, input_gate_bias))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, state=None):
        for block in self.blocks:
            x = block(x)
        return x, state
