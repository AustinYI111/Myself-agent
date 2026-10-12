import pandas as pd

from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent


def test_debug_agents_outputs():
    close = []
    close += [12.0] * 5
    close += [11.8, 11.6, 11.4, 11.2, 11.0, 10.8, 10.6, 10.4, 10.2, 10.0]
    close += [10.2, 10.4, 10.7, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4, 13.8, 14.2]
    close += [14.0, 13.7, 13.3, 12.9, 12.4, 12.0, 11.6, 11.2, 10.9, 10.6, 10.3, 10.0]

    df = pd.DataFrame({"close": close})
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")

    trend = TrendAgent(short_window=3, long_window=8)
    mr = MeanReversionAgent(window=8)

    print("\n===== Debug Trend / MeanRev outputs =====")
    for i in range(len(df)):
        w = df.iloc[: i + 1]
        out_t = trend.generate_signal(w)
        out_m = mr.generate_signal(w)
        print(i, df.index[i].date(), w["close"].iloc[-1], "trend=", out_t, "mr=", out_m)

    # 这里只是 debug，不做强断言
    assert True
