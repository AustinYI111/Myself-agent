import pandas as pd
from agents.mean_reversion_agent import MeanReversionAgent


def test_mean_reversion_agent_basic():
    data = pd.DataFrame({
        "close": [
            100, 101, 99, 100, 98, 97, 96, 95, 94, 93,
            92, 91, 90, 89, 88, 87, 86, 85, 84, 83
        ]
    })

    agent = MeanReversionAgent(window=5, num_std=2)
    result = agent.generate_signal(data)

    print("MeanReversionAgent output:", result)

    assert result["signal"] in ["buy", "sell", "hold"]
    assert 0 <= result["confidence"] <= 1
