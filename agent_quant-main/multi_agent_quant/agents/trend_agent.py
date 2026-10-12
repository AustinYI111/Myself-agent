# agents/trend_agent.py
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np


@dataclass
class TrendAgent:
    short_window: int = 5
    long_window: int = 20
    min_conf_when_signal: float = 0.20   # ✅ 关键：最低可交易置信度
    scale: float = 15.0                 # ✅ 关键：把均线差距放大成 0~1

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        if data is None or len(data) < self.long_window:
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "insufficient_data"}}

        close = data["close"].astype(float)
        short_ma = close.rolling(self.short_window).mean()
        long_ma = close.rolling(self.long_window).mean()

        curr_short = float(short_ma.iloc[-1])
        curr_long = float(long_ma.iloc[-1])
        prev_short = float(short_ma.iloc[-2])
        prev_long = float(long_ma.iloc[-2])

        if np.isnan(curr_short) or np.isnan(curr_long) or curr_long == 0:
            return {"signal": "hold", "confidence": 0.0, "meta": {"reason": "nan_ma"}}

        # ✅ 趋势方向：不只在交叉那天输出，而是趋势成立期间持续输出
        if curr_short > curr_long:
            sig = "buy"
        elif curr_short < curr_long:
            sig = "sell"
        else:
            sig = "hold"

        # ✅ 置信度：用均线“相对差距”并放大到可交易量级
        spread = abs(curr_short - curr_long) / abs(curr_long)
        conf = min(1.0, spread * self.scale)

        if sig != "hold":
            conf = max(self.min_conf_when_signal, conf)
        else:
            conf = 0.05 * conf  # hold 时压低一点

        return {
            "signal": sig,
            "confidence": float(conf),
            "meta": {
                "short_window": self.short_window,
                "long_window": self.long_window,
                "prev_short_ma": prev_short,
                "prev_long_ma": prev_long,
                "curr_short_ma": curr_short,
                "curr_long_ma": curr_long,
                "spread": float(spread),
            },
        }
