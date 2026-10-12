# tests/test_backtest_engine.py
import pandas as pd

from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.coordinator_agent import CoordinatorAgent
from backtest.backtest_engine import BacktestEngine


def test_backtest_engine_basic():
    close = []
    close += [12.0] * 5
    close += [11.8, 11.6, 11.4, 11.2, 11.0, 10.8, 10.6, 10.4, 10.2, 10.0]
    close += [10.2, 10.4, 10.7, 11.0, 11.4, 11.8, 12.2, 12.6, 13.0, 13.4, 13.8, 14.2]
    close += [14.0, 13.7, 13.3, 12.9, 12.4, 12.0, 11.6, 11.2, 10.9, 10.6, 10.3, 10.0]

    df = pd.DataFrame({"close": close})
    df.index = pd.date_range("2024-01-01", periods=len(df), freq="D")

    agents = {
        "trend": TrendAgent(short_window=3, long_window=8),
        "mean_reversion": MeanReversionAgent(window=8),
    }

    coordinator = CoordinatorAgent(
        agent_weights={"trend": 0.55, "mean_reversion": 0.45},
        min_edge=0.0,
        min_score_to_trade=0.0,
        regime_boost=0.50,
        min_conf_when_trade=0.20,
    )

    engine = BacktestEngine(
        init_cash=100000.0,
        fee_rate=0.0003,
        slippage_bps=5.0,
        max_position_pct=0.98,
        min_order_value=100.0,
        allow_fractional=True,   # 测试阶段先开，避免整手=0股
        lot_size=100,
    )

    metrics, trades = engine.run_fusion(
        df=df,
        agents=agents,
        coordinator=coordinator,
        strategy_name="Fusion(Trend+MeanRev)",
        verbose=False,
    )

    print("\n===== Fusion(Trend+MeanRev) =====")
    print("Metrics:", metrics)
    print("Trades:", len(trades))
    if trades:
        print("First trade:", trades[0])
        print("Last trade:", trades[-1])

    assert len(trades) > 0
