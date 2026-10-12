import pandas as pd
from agents.trend_agent import TrendAgent

def test_trend_agent_basic():
    data = pd.DataFrame({
        "close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    })

    agent = TrendAgent(short_window=3, long_window=5)
    result = agent.generate_signal(data)

    print("TrendAgent output:", result)

    assert result["signal"] in ["buy", "sell", "hold"]
    assert 0 <= result["confidence"] <= 1