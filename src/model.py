# -*- coding: utf-8 -*-
"""XLSTM-Transformer 模型。

结构（把 xLSTM 层融进 encoder 块内，这是工程原本的改进点）：

    输入 (B, look_back, F)
      -> 线性投影 + 位置编码
      -> N 个编码器块，每块 = [多头自注意力] -> [xLSTM] -> [前馈网络]
      -> M 个解码器块，每块 = [因果自注意力] -> [对编码器输出的交叉注意力] -> [前馈网络]
      -> 展平 + 线性输出头 -> (B, horizon)

相对于原始工程的修正
--------------------
1. 多头注意力显式使用 batch_first=True，注意力在时间维上计算；
   （PyTorch 的 nn.MultiheadAttention 默认 batch_first=False，会把 batch 当序列）
2. 补上位置编码。自注意力本身对时间步是置换不变的，缺了它编码器看不到先后顺序；
3. 解码器自注意力加因果掩码，避免在窗内看到未来时刻；
4. 解码器第二段前馈不再复用同一组全连接层；
5. 输出加物理约束（光伏出力非负），clamp_min 可配置；
6. 引入 use_xlstm / use_decoder 开关，便于做消融实验。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .xlstm import xLSTM

__all__ = ["SinusoidalPositionalEncoding", "EncoderBlock", "DecoderBlock", "XLSTMTransformer"]


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码；kind="learned" 时改为可学习的位置向量。"""

    def __init__(self, embed_dim: int, max_len: int, kind: str = "learned", dropout: float = 0.0):
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.pos = nn.Parameter(torch.zeros(1, max_len, embed_dim))
            nn.init.normal_(self.pos, std=0.02)
        elif kind == "sinusoidal":
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
            self.register_buffer("pos", pe.unsqueeze(0), persistent=False)
        else:
            raise ValueError("kind 只能是 'learned' 或 'sinusoidal'")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pos[:, : x.size(1), :])


class EncoderBlock(nn.Module):
    """编码器块：自注意力 ->（可选）xLSTM -> 前馈网络，全部走 pre-norm 残差。"""

    def __init__(
        self,
        embed_dim: int,
        dense_dim: int,
        num_heads: int,
        dropout: float,
        look_back: int,
        use_xlstm: bool = True,
        xlstm_layers: str = "m",
        xlstm_heads: int | None = None,
        xlstm_conv_axis: str = "time",
        xlstm_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)

        self.use_xlstm = use_xlstm
        if use_xlstm:
            heads = xlstm_heads or num_heads
            head_size = embed_dim // heads
            self.xlstm = xLSTM(
                input_size=embed_dim,
                head_size=head_size,
                num_heads=heads,
                layers=xlstm_layers,
                conv_axis=xlstm_conv_axis,
                dropout=xlstm_dropout,
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.drop2 = nn.Dropout(dropout)
        else:
            self.xlstm = None
            self.norm2 = None
            self.drop2 = None

        self.ff1 = nn.Linear(embed_dim, dense_dim)
        self.ff2 = nn.Linear(dense_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, need_weights: bool = False):
        attn, weights = self.self_attn(x, x, x, need_weights=need_weights)
        x = self.norm1(x + self.drop1(attn))

        if self.xlstm is not None:
            recurrent, _ = self.xlstm(x)
            x = self.norm2(x + self.drop2(recurrent))

        h = self.ff2(self.drop3(F.gelu(self.ff1(x))))
        x = self.norm3(x + self.drop3(h))
        return x, weights


class DecoderBlock(nn.Module):
    """解码器块：因果自注意力 -> 交叉注意力 -> 独立的前馈网络。"""

    def __init__(self, embed_dim: int, dense_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.ff1 = nn.Linear(embed_dim, dense_dim)
        self.ff2 = nn.Linear(dense_dim, embed_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        causal = torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device).tril()
        attn, _ = self.self_attn(x, x, x, attn_mask=~causal, need_weights=False)
        x = self.norm1(x + self.drop1(attn))

        cross, _ = self.cross_attn(x, memory, memory, need_weights=False)
        x = self.norm2(x + self.drop2(cross))

        h = self.ff2(self.drop3(F.gelu(self.ff1(x))))
        x = self.norm3(x + self.drop3(h))
        return x


class XLSTMTransformer(nn.Module):
    """多步出力预测模型。

    num_features   输入特征维度
    look_back      历史窗口长度
    horizon        预测步数
    embed_dim      嵌入维度
    dense_dim      前馈隐层维度
    num_heads      注意力头数
    num_blocks     编码器 / 解码器块数
    dropout        失活率
    use_xlstm      是否在编码器内融合 xLSTM（关掉即可做消融）
    use_decoder    是否使用解码器分支
    clamp_min      输出下界（光伏出力非负，默认 0）
    """

    def __init__(
        self,
        num_features: int,
        look_back: int,
        horizon: int,
        embed_dim: int = 32,
        dense_dim: int = 64,
        num_heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
        use_xlstm: bool = True,
        use_decoder: bool = True,
        xlstm_layers: str = "m",
        xlstm_heads: int | None = None,
        xlstm_conv_axis: str = "time",
        pos_encoding: str = "learned",
        clamp_min: float | None = 0.0,
        clamp_in_train: bool = False,
        head_bias: float = 0.0,
    ):
        super().__init__()
        self.look_back = look_back
        self.horizon = horizon
        # 非负约束只在推理/评估时生效：训练时若也截断，输出一旦落到负半轴
        # 梯度就会变成 0，模型会被永久压在 0 上（这正是"预测恒为 0"的成因）。
        self.clamp_min = clamp_min
        self.clamp_in_train = clamp_in_train

        self.embedding = nn.Linear(num_features, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(embed_dim, look_back, pos_encoding, dropout)

        self.encoders = nn.ModuleList(
            [
                EncoderBlock(embed_dim, dense_dim, num_heads, dropout, look_back,
                             use_xlstm, xlstm_layers, xlstm_heads, xlstm_conv_axis, 0.0)
                for _ in range(num_blocks)
            ]
        )

        self.use_decoder = use_decoder
        if use_decoder:
            self.decoders = nn.ModuleList(
                [DecoderBlock(embed_dim, dense_dim, num_heads, dropout) for _ in range(num_blocks)]
            )
            self.norm_out = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim * look_back, dense_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, horizon),
        )
        # 输出偏置初始化到训练目标均值附近，避免一开始就朝 0 收敛
        nn.init.constant_(self.head[-1].bias, head_bias)

    def forward(self, x: torch.Tensor):
        """x: (B, look_back, num_features) -> (B, horizon)"""
        h = self.pos_encoding(self.embedding(x))
        for encoder in self.encoders:
            h, _ = encoder(h)

        if self.use_decoder:
            # 解码器输入与编码器相同（不使用 teacher forcing 的位移），
            # 自注意力已加因果掩码，等价于在编码结果上再做一次因果细化。
            d = h
            for decoder in self.decoders:
                d = decoder(d, h)
            h = self.norm_out(d)

        flat = h.reshape(h.size(0), -1)
        out = self.head(flat)
        if self.clamp_min is not None and (self.clamp_in_train or not self.training):
            out = torch.clamp(out, min=self.clamp_min)
        return out

    @torch.no_grad()
    def attention_maps(self, x: torch.Tensor):
        """返回第一层编码器自注意力的权重，便于检查注意力是否落在时间轴上。"""
        h = self.pos_encoding(self.embedding(x))
        maps = []
        for i, encoder in enumerate(self.encoders):
            h, w = encoder(h, need_weights=True)
            if w is not None:
                maps.append(w.detach().cpu())
            if i == 0:
                break
        return maps
