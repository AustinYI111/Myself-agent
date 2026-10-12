# agents/mean_reversion_agent.py

import pandas as pd
import numpy as np


class MeanReversionAgent:
    """
    均值回归策略 Agent（基于 Bollinger Bands）

    逻辑：
    - price < lower_band -> buy
    - price > upper_band -> sell
    - else hold

    confidence：
    - 使用 Z-score 来度量“偏离程度”，更稳健：
      z = (price - mean) / std
      price < lower 时 z < -num_std
      price > upper 时 z > +num_std
    [1](https://blog.csdn.net/gitblog_07246/article/details/148988537)[2](https://ask.csdn.net/questions/8583327)
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, conf_scale: float = 3.0):
        """
        window: 布林带窗口
        num_std: 布林带倍数
        conf_scale: 置信度缩放因子（越小越“激进”，默认 3.0 比较温和）
        """
        self.window = window
        self.num_std = num_std
        self.conf_scale = conf_scale

    def generate_signal(self, data: pd.DataFrame) -> dict:
        """
        输入：
            data: 包含 'close' 列的 DataFrame（可以有 DateIndex，也可以普通 index）
        输出：
            {
                "signal": "buy" | "sell" | "hold",
                "confidence": float (0~1),
                "meta": dict
            }
        """
        # 1) 数据检查
        if "close" not in data.columns or len(data) < self.window:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "meta": {"reason": "insufficient_data", "window": self.window}
            }

        close = pd.to_numeric(data["close"], errors="coerce")
        if close.isna().iloc[-1]:
            return {
                "signal": "hold",
                "confidence": 0.0,
                "meta": {"reason": "close_nan"}
            }

        # 2) 计算布林带
        rolling_mean = close.rolling(self.window).mean()
        rolling_std = close.rolling(self.window).std()

        mean = rolling_mean.iloc[-1]
        std = rolling_std.iloc[-1]

        # std 兜底（避免除 0/NaN）
        if std is None or np.isnan(std) or std == 0 or mean is None or np.isnan(mean):
            return {
                "signal": "hold",
                "confidence": 0.0,
                "meta": {"reason": "std_or_mean_invalid", "mean": mean, "std": std}
            }

        upper = mean + self.num_std * std
        lower = mean - self.num_std * std

        price = close.iloc[-1]
        z = (price - mean) / std  # Z-score（偏离均值的标准差倍数）[1](https://blog.csdn.net/gitblog_07246/article/details/148988537)[2](https://ask.csdn.net/questions/8583327)

        signal = "hold"
        confidence = 0.0

        # 3) 生成信号
        if price < lower:
            signal = "buy"
            # z < -num_std，越小越极端 -> 置信度越高
            confidence = (-z - self.num_std) / self.conf_scale

        elif price > upper:
            signal = "sell"
            # z > +num_std，越大越极端 -> 置信度越高
            confidence = (z - self.num_std) / self.conf_scale

        # 4) 裁剪到 0~1
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "meta": {
                "window": self.window,
                "num_std": self.num_std,
                "conf_scale": self.conf_scale,
                "price": round(float(price), 4),
                "mean": round(float(mean), 4),
                "std": round(float(std), 4),
                "zscore": round(float(z), 4),
                "upper_band": round(float(upper), 4),
                "lower_band": round(float(lower), 4),
            }
        }