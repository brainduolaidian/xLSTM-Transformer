# -*- coding: utf-8 -*-
"""xLSTM-Transformer 光伏出力预测。

子模块按需导入，避免 `import src` 就拉起 torch 等重依赖：

    from src.data import DataConfig, prepare_data
    from src.model import XLSTMTransformer
    from src.xlstm import mLSTM, sLSTM, xLSTM
    from src.trainer import TrainConfig, train_model
    from src.evaluate import run_test, format_report
    from src.metrics import evaluate_predictions, regression_metrics
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
