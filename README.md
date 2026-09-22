# xLSTM-Transformer 光伏出力多步预测

把 **xLSTM**（指数门控 + 矩阵记忆的线性循环网络）融进 Transformer 编码器，做光伏出力的多步短期预测：
用过去 1 天（96 个 15 分钟采样点）的历史，预测未来 1 小时（4 个点）。

模型内部实现严格对齐论文公式，数据流程无信息泄漏，训练与评测可复现，并内置持续性与日周期朴素基线，
直接回答"这个模型到底有没有用"。

---

## 1. 快速开始

```bash
pip install -r requirements.txt

# 训练 + 测试（结果写入 results/main/）
python run_train.py --tag main

# 只用已有权重做推理（自动读取该次实验保存的配置，无需重复传参）
python run_predict.py --run-dir results/main
python run_predict.py --run-dir results/main --last-window   # 只对末尾窗口预测一次

# 消融实验（已有结果自动复用）
python run_ablations.py

# 测试：并行 mLSTM 与论文递归式数值等价、无跨样本串扰、注意力落在时间维、数据流程防泄漏
python tests/test_xlstm_equivalence.py
python tests/test_data_pipeline.py
```

默认配置：`look_back=96`、`horizon=4`、`embed_dim=32`、`dense_dim=64`、`num_heads=4`、
`num_blocks=2`、`dropout=0.1`、`epochs=50`、`batch_size=64`、`lr=1e-3`。
全部超参数见 `python run_train.py --help`。

---

## 2. 数据

**本仓库不附带原始数据**（论文与实验所用的数据集涉及第三方版权）。
完整的数据要求、接入方式与防泄漏注意事项见 [DATA.md](DATA.md)，这里只给要点。

代码期望的数据形态：

| 要求 | 说明 |
|---|---|
| 格式 | Excel 或 csv，每行一个时间点、按时间升序、等间隔采样（默认 15 分钟、96 点/天） |
| 目标 | 一列待预测的功率值，用 `--target-col` 指定，不指定则取最后一列 |
| 特征 | 除目标列外的所有数值列自动作为输入特征 |
| 时间列 | 可选，给时间戳列名后会追加年内日期正余弦特征 |
| 规模 | 论文实验用的那份数据为 19583 行 × 3 列、15 分钟、约 204 天，目标列夜间为 0、正午达峰 |

把数据放到 `data/dataset.xlsx`（该目录已被 `.gitignore` 忽略），然后：

```bash
python run_train.py                             # 目标列取最后一列
python run_train.py --data D:/mydata.xlsx --target-col power --time-col time
```

两点提醒：

1. 论文实验数据里有一列与目标列**逐点完全相同**（程序会在日志里自动提示）。
   这等价于把目标自身的历史作为自回归输入，本身合法，但说明数据中缺少真正的
   气象驱动变量（辐照度、温度、湿度、云量）。要论证"多变量预测"或提升精度需要补上这些列。
2. 程序默认自动加入**日内时刻正余弦特征**（由行号对 `points_per_day` 取模得到，假设采样均匀），
   可用 `--no-time-of-day` 关闭。

### 想引入更多信息（IFS / GFS 数值天气预报等）

接口已经预留好，两种方式都不需要改代码：

```bash
# 方式一：把预报列拼进主表，数值列自动成为特征
python run_train.py --data power_with_nwp.xlsx --target-col power --time-col time

# 方式二：预报单独放文件，按时间列自动左连接（可给多个文件）
python run_train.py --data power.xlsx --target-col power --time-col time \
                    --extra-files gfs_forecast.csv ifs_forecast.xlsx --merge-on time
```

合并进来的列会自动成为特征，重名列加 `_nwp` 后缀，日志里会打印**时间戳匹配率**。
需要自定义对齐或派生逻辑时改 `src/data.py::_merge_extra_files` 与 `_add_time_features`；
**注意只能用预测时刻之前已发布的预报**，避免把未来的观测当成预报特征，详见 [DATA.md](DATA.md) 第 4 节。

---

## 3. 目录结构

```
xLSTM-Transformer/
├── run_train.py                     # 训练 + 测试入口（argparse）
├── run_predict.py                   # 独立推理入口（加载权重，复用实验配置）
├── run_ablations.py                 # 消融实验批跑与汇总
├── requirements.txt
├── DATA.md                          # 数据要求、NWP 预报接入方式、防泄漏说明
├── data/                            # 放你自己的数据（表格文件已被 .gitignore 忽略）
├── src/
│   ├── xlstm.py                     # sLSTM / mLSTM / xLSTM，按论文公式实现
│   ├── model.py                     # 编码器内融合 xLSTM 的 Encoder-Decoder 模型
│   ├── data.py                      # 读取、去重、切分、归一化、滑窗
│   ├── trainer.py                   # 训练循环：种子、梯度裁剪、调度、早停、保存最优
│   ├── evaluate.py                  # 测试集评估、基线对比、结果导出与绘图
│   ├── metrics.py                   # 指标与朴素基线
│   └── utils.py                     # 随机种子、绘图风格
├── tests/
│   ├── test_xlstm_equivalence.py    # 数值等价性与结构回归测试
│   └── test_data_pipeline.py        # 数据流程与防泄漏测试
└── results/<tag>/                   # 每次实验的配置、日志、指标、图表
```

---

## 4. 模型与关键实现

### 4.1 整体结构

```
输入 (B, 96, F)
  → 线性投影 + 位置编码
  → N × 编码器块 [ 多头自注意力 → xLSTM → 前馈网络 ]     ← xLSTM 融合点
  → M × 解码器块 [ 因果自注意力 → 交叉注意力 → 前馈网络 ]
  → 展平 + 输出头 → (B, 4)，推理时截断到非负
```

### 4.2 mLSTM 的并行实现（核心）

论文的递归形式：

```
m_t   = max( log σ(f̃_t) + m_{t-1}, ĩ_t )                      (1)
C_t   = f_t C_{t-1} + i_t k_t v_tᵀ                            (2)   ← 矩阵记忆
n_t   = f_t n_{t-1} + i_t k_t                                 (3)
h̃_t   = C_tᵀ (q_t/√d) / max( |n_tᵀ (q_t/√d)|, exp(-m_t) )      (4)
h_t   = o_t ⊙ NORM(h̃_t)                                      (5)
f_t   = exp( log σ(f̃_t) + m_{t-1} - m_t )                      (6)
```

逐步循环在 Python 里很慢：按 4 个编码器块 × 3 层 × 96 个时间步算，单次前向要做 1152 次递归。
本项目把式 (1)-(6) 展开成并行形式：
记 `ℓ_t = Σ_{u≤t} log σ(f̃_u)`，则遗忘门连乘为 `exp(ℓ_t − ℓ_s)`，于是

```
C_t = exp(−m_t) · Σ_{s≤t} w_{t,s} k_s v_sᵀ ,   w_{t,s} = exp(ℓ_t − ℓ_s + ĩ_s)
```

而 `m_t` 恰好等于 `log w_{t,·}` 在 `s ≤ t` 上的**行最大值**（与式 (1) 的递归最大值严格等价，
`exp(−m_t)` 在式 (4) 的分子分母中相互抵消，只留下 `max(·, exp(−m_t))` 这一处除零保护）。
这样一次矩阵运算就能算出整条序列，数值上也不会溢出。

`tests/test_xlstm_equivalence.py` 用论文式 (1)-(6) 的逐步递归实现与并行实现逐点对比，
最大误差需小于 `1e-5`，保证两种写法数学等价。

### 4.3 实现要点：容易踩的坑与本文的处理

下表列出这类模型在实现时最容易出偏差的几处，以及本仓库的处理方式。级别是严重度标注：
P0 会让结果不可信，P1 影响可复现性或指标口径，P2 属工程规范。

| 级别 | 容易踩的坑 | 本仓库的处理 |
|---|---|---|
| P0 | `nn.MultiheadAttention` 的 `batch_first` 默认是 False，若按 (batch, seq, feat) 直接传入，就会把 batch 当序列、时间步当样本，注意力落在**样本之间** | 显式 `batch_first=True`；并加回归测试：同一输入样本的预测不随同批其他样本变化 |
| P0 | 无位置编码，自注意力对时间步置换不变，编码器看不到先后顺序 | 加入可学习（或正弦）位置编码 |
| P0 | 把多头维度拍平、用逐元素乘 `i*(v*k)` 代替外积，矩阵记忆会退化成逐元素运算；归一化分母写成 `max(abs(n_tᵀ @ q), 1)` 会跨样本耦合 | 重写为 `(B, H, d, d)` 矩阵记忆 + 外积更新 + 按 head 归一化，状态形状与论文一致 |
| P0 | 归一化在**全量数据**上 `fit`，测试集极值参与缩放，属信息泄漏 | 只用训练段 `fit`，再 `transform` 全量 |
| P1 | 训练/验证损失记录的是**最后一个 batch** 的值，曲线不可读 | 记录整轮平均损失 |
| P1 | 无随机种子、无 `torch.save`（用最后一轮权重）、无早停、无梯度裁剪 | 全部补齐，按验证集 RMSE 选最优并保存 `best_model.pt` |
| P1 | MAPE 用 `1e-6` 兜底分母，夜间真值≈0 时失真；4 个步长混成一个指标 | 非零时段 MAPE + 分步长指标 + nRMSE |
| P1 | 输出层不加约束时会出现负值预测，而光伏出力不可能为负 | 推理时 `clamp(min=0)`；训练时不截断（否则输出会被永久压在 0 上，梯度消失） |
| P1 | 解码器连续两段复用同一组 `dense1/dense2`，第二段没有独立参数；自注意力无因果掩码 | 解码器各段独立参数，自注意力加因果掩码 |
| P2 | 单文件脚本、无 `__main__` 保护、相对路径依赖工作目录、大段注释死代码、重复 import、类里直接用全局变量 | 拆成 `src/` 模块 + CLI 入口，绝对路径，删除死代码 |
| P2 | 无基线对比 | 内置持续性与日周期朴素基线，输出技能得分 |
| P2 | 没有独立推理脚本，只能重训才能出结果 | 新增 `run_predict.py`，读取实验 `config.json` 复用模型结构与数据配置，支持整测试集推理与单窗口预测 |
| P2 | 无消融实验 | 新增 `run_ablations.py`，一键批跑并汇总成表 |

---

## 5. 实验结果

### 5.1 主实验

配置：`look_back=96`、`horizon=4`、`embed_dim=32`、`dense_dim=64`、`num_heads=4`、`num_blocks=2`、
`dropout=0.1`，参数量 **270,652**。训练上限 50 轮，验证集 12 轮无改善后提前停止于第 37 轮，
最优权重取第 25 轮（验证 RMSE = 0.8387）。

| 方案 | R² | MAE | RMSE | nRMSE | 非零时段 MAPE | 负值占比 |
|---|---|---|---|---|---|---|
| **本文模型** | **0.9066** | **0.618** | **1.173** | **0.0858** | **30.63%** | **0.0%** |
| 持续性基线（沿用上一时刻） | 0.8579 | 0.756 | 1.447 | 0.1059 | 46.68% | 0.0% |
| 日周期朴素基线（沿用前一日同时刻） | 0.6322 | 1.166 | 2.328 | 0.1703 | 70.04% | 0.0% |

相对持续性基线的技能得分 **+0.189**，相对日周期基线 **+0.496**；模型在每一个预测步长上都优于持续性基线，
负值预测占比为 0（推理时对输出做了非负截断）。

分步长指标：

| 步长 | R² | MAE | RMSE | nRMSE | 非零时段 MAPE |
|---|---|---|---|---|---|
| 第 1 步（15 分钟） | 0.9358 | 0.523 | 0.975 | 0.0718 | 24.29% |
| 第 2 步（30 分钟） | 0.9085 | 0.585 | 1.177 | 0.0861 | 29.18% |
| 第 3 步（45 分钟） | 0.8942 | 0.649 | 1.235 | 0.0974 | 30.77% |
| 第 4 步（1 小时） | 0.8871 | 0.714 | 1.284 | 0.0975 | 38.05% |

误差随步长单调上升，符合多步预测的预期。

> 说明：原始数据不随仓库分发，因此**结果图表与逐样本预测值也没有提交**
> （见 `.gitignore`）。把数据按 [DATA.md](DATA.md) 准备好后运行 `python run_train.py`，
> 会在 `results/<tag>/` 下重新生成：
> `training_curve.png`（整轮平均损失与验证 RMSE）、
> `prediction_curve.png`（测试集前 480 条曲线对比与散点）、
> `per_horizon_rmse.png`（与持续性基线的分步长 RMSE 对比）、
> `test_predictions.csv` 与 `metrics.json`（逐样本预测与全部指标）。
> 仓库里保留了 `config.json` 与 `metrics.json`，用于核对实验配置与上述数值。

### 5.2 消融实验

训练协议与主实验完全一致（50 轮上限、`patience=12`、种子 42），每次只改动一个因素。
汇总由 `python run_ablations.py` 自动生成，同时写入 `results/ablation_summary.md` 与 `.csv`。

| 配置 | R² | MAE | RMSE | nRMSE | 非零时段 MAPE | 相对持续性技能得分 |
|---|---|---|---|---|---|---|
| **完整模型** | **0.9066** | **0.618** | **1.173** | **0.0858** | **30.63%** | **+0.189** |
| 去掉编码器内的 xLSTM | 0.8998 | 0.698 | 1.215 | 0.0889 | 30.81% | +0.160 |
| 去掉解码器分支 | 0.8714 | 0.828 | 1.376 | 0.1007 | 34.66% | +0.049 |

几点结论：

1. **xLSTM 分支在这份数据上只有小幅增益**（R² 0.9066 → 0.8998，RMSE 上升 3.6%）。
   原因不难解释：数据里只有约一个有效输入信号，自相关已经解释了大部分方差，
   循环结构能补的信息有限。若要把 xLSTM 的价值做出来，先补上辐照度、温度等驱动变量，
   再重跑这组消融。
2. **解码器分支的贡献更大**（去掉后 R² 下降 0.035、RMSE 上升 17%），
   说明编码器输出之后的因果自注意力 + 交叉注意力这一层在这个任务上是有效的。
3. 三个配置都优于持续性基线，说明改进方向本身站得住。
4. 含 sLSTM 的配置（`--xlstm-layers msm`）因为门控混合无法并行、只能逐步递归，
   在 CPU 上会慢一个数量级，默认不跑；需要时单独执行即可。

消融脚本：`python run_ablations.py`（已有结果会自动复用，不会重复训练）。

---

## 6. 复现环境

| 项目 | 版本 |
|---|---|
| 操作系统 | Windows |
| Python | 3.12.4 |
| torch | 2.7.0+cpu |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.8 |
| openpyxl | 3.1.5 |

随机种子固定为 42，每次实验的完整配置与日志保存在 `results/<tag>/config.json`、
`results/<tag>/history.json`、`results/<tag>/metrics.json`。

---

## 7. 参考文献

- Beck, M. et al. *xLSTM: Extended Long Short-Term Memory.* arXiv:2405.04517
- Beck, M. et al. *Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels.* arXiv:2503.14376（mLSTM 递归式与稳定化门控）
- Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017

---

## 8. 说明

- 本仓库的 xLSTM 是自包含实现，仅依赖 PyTorch，便于阅读与消融实验；若追求训练速度，
  可换成官方 `xlstm` 包或 CUDA 内核。
- **本仓库不包含任何原始数据**。论文实验所用的数据涉及第三方版权，不能公开分发；
  请按 [DATA.md](DATA.md) 准备自己的数据。同理，结果图表与逐样本预测值也未提交。
  若你希望把预报信息（IFS / GFS 等）接进来，`--extra-files` 已经留好了接口。
- 仓库暂未附带开源协议，如需正式公开建议补一个（例如 MIT）。
