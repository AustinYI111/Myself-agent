# agents/ml_agent.py
import numpy as np
import pandas as pd


class MLAgent:
    """
    辅助决策 ML Agent（不发起交易）
    --------------------------------
    设计目标：
    - 不直接输出 buy/sell（避免 ML-only 乱交易）
    - 只输出“确认/否决/微调”信息给 Coordinator 使用

    输出：
    {
      "signal": "hold",
      "confidence": aux_strength (0~1),
      "meta": {
          "aux": True,
          "p_up": 0~1,
          "aux_suggest": "buy"|"sell"|"hold",
          "aux_strength": 0~1,
          "reason": ...
      }
    }
    """

    def __init__(self, lookback: int = 10, min_train_size: int = 60, prob_threshold: float = 0.55):
        self.lookback = int(lookback)
        self.min_train_size = int(min_train_size)
        self.prob_threshold = float(prob_threshold)

    def generate_signal(self, data: pd.DataFrame) -> dict:
        # 基础检查
        if data is None or "close" not in data.columns:
            return {"signal": "hold", "confidence": 0.0, "meta": {"aux": True, "reason": "missing_close"}}

        close = pd.to_numeric(data["close"], errors="coerce").dropna()
        if len(close) < max(5, self.min_train_size):
            # ✅ 明确告诉你“为什么不输出”
            return {"signal": "hold", "confidence": 0.0, "meta": {"aux": True, "reason": "insufficient_data"}}

        # ---- 轻量特征（保证可运行） ----
        lb = min(self.lookback, len(close) - 1)

        # 动量：近 lookback 收益
        ret_lb = (close.iloc[-1] / close.iloc[-1 - lb]) - 1.0 if lb > 0 else 0.0

        # 波动：近 lookback 的收益率 std
        rets = close.pct_change().dropna()
        if len(rets) >= lb and lb > 2:
            vol = float(rets.iloc[-lb:].std())
        elif len(rets) > 2:
            vol = float(rets.std())
        else:
            vol = 0.0

        # 均线偏离：优先用 ma_20，否则自己算
        if "ma_20" in data.columns and not pd.isna(data["ma_20"].iloc[-1]):
            ma = float(data["ma_20"].iloc[-1])
        else:
            ma = float(close.iloc[-min(20, len(close)):].mean())
        dist = (float(close.iloc[-1]) / ma - 1.0) if ma != 0 else 0.0

        # ---- 概率打分（sigmoid）----
        # 趋势更强、均线偏离更正 -> p_up 上升；波动越大 -> 更保守
        x = 3.0 * ret_lb + 2.0 * dist - 5.0 * vol
        p_up = 1.0 / (1.0 + np.exp(-x))

        # ---- 输出“辅助建议” ----
        aux_suggest = "hold"
        if p_up >= self.prob_threshold:
            aux_suggest = "buy"
        elif p_up <= (1.0 - self.prob_threshold):
            aux_suggest = "sell"

        # 强度：离 0.5 越远越强
        aux_strength = float(min(1.0, abs(p_up - 0.5) * 2.0))

        return {
            "signal": "hold",  # ✅ 永远 hold：不直接下单
            "confidence": round(aux_strength, 4),  # 仅作为“强度”
            "meta": {
                "aux": True,
                "p_up": round(float(p_up), 4),
                "aux_suggest": aux_suggest,
                "aux_strength": round(aux_strength, 4),
                "features": {
                    "ret_lb": round(float(ret_lb), 4),
                    "dist_ma": round(float(dist), 4),
                    "vol": round(float(vol), 4),
                },
            }
        }