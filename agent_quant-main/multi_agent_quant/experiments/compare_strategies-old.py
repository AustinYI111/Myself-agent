# experiments/compare_strategies.py
import pandas as pd  # pyright: ignore[reportMissingImports]

from agents.data_agent import DataAgent
from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.coordinator_agent import CoordinatorAgent
from backtest.backtest_engine import BacktestEngine


# ========== 1) 统一配置 ==========
CONFIG = dict(
    symbol="688256",          # 寒武纪（科创板）
    start_date="20240101",
    end_date="20241231",

    # BacktestEngine 参数（名字必须和 BacktestEngine.__init__ 一致）
    init_cash=100000,
    fee_rate=0.0003,
    slippage_bps=5.0,
    max_position_pct=0.98,
    min_order_value=1000.0,
    lot_size=100,
    allow_fractional=False,
)


def _to_dt(s: str) -> pd.Timestamp:
    # "20240101" -> Timestamp("2024-01-01")
    return pd.to_datetime(s, format="%Y%m%d")


# ========== 2) 单次运行封装 ==========
def run_trend_only(engine: BacktestEngine, df: pd.DataFrame, trend: TrendAgent):
    return engine.run_single_agent(
        df=df,
        agent=trend,
        strategy_name="Trend-only",
        verbose=False,
    )


def run_meanrev_only(engine: BacktestEngine, df: pd.DataFrame, meanrev: MeanReversionAgent):
    return engine.run_single_agent(
        df=df,
        agent=meanrev,
        strategy_name="MeanRev-only",
        verbose=False,
    )


def run_fusion(engine: BacktestEngine, df: pd.DataFrame, trend, meanrev, coordinator):
    agents = {
        "trend": trend,
        "mean_reversion": meanrev,
    }
    return engine.run_fusion(
        df=df,
        agents=agents,
        coordinator=coordinator,
        strategy_name="Fusion(Trend+MeanRev)",
        verbose=False,
    )


# ========== 3) 主流程 ==========
def main():
    # ---- 数据 ----
    data_agent = DataAgent(
        symbol=CONFIG["symbol"],
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
    )
    df = data_agent.get_feature_data()

    # ✅ 强制排序 + 强制截断（避免 DataAgent 缓存/接口返回超出区间）
    df = df.sort_index()
    start_dt = _to_dt(CONFIG["start_date"])
    end_dt = _to_dt(CONFIG["end_date"])
    df = df.loc[start_dt:end_dt]

    # ✅ 打印验证：你必须看到这里是 2024-01-01 ~ 2024-12-31 才算对
    print(f"Symbol={CONFIG['symbol']} DF range: {df.index.min()} -> {df.index.max()} rows={len(df)}")
    if len(df) == 0:
        raise ValueError("DataFrame is empty after date slicing. Check DataAgent / symbol / date range.")

    # ---- 回测引擎 ----
    engine = BacktestEngine(
        init_cash=CONFIG["init_cash"],
        fee_rate=CONFIG["fee_rate"],
        slippage_bps=CONFIG["slippage_bps"],
        max_position_pct=CONFIG["max_position_pct"],
        min_order_value=CONFIG["min_order_value"],
        lot_size=CONFIG["lot_size"],
        allow_fractional=CONFIG["allow_fractional"],
    )

    # ---- Agents ----
    trend = TrendAgent(short_window=5, long_window=20)
    meanrev = MeanReversionAgent(window=20, num_std=2)
    coordinator = CoordinatorAgent(agent_weights={"trend": 0.5, "mean_reversion": 0.5})

    # ---- 运行对比 ----
    rows = []

    runs = [
        ("Trend-only", lambda: run_trend_only(engine, df, trend)),
        ("MeanRev-only", lambda: run_meanrev_only(engine, df, meanrev)),
        ("Fusion(Trend+MeanRev)", lambda: run_fusion(engine, df, trend, meanrev, coordinator)),
    ]

    for name, fn in runs:
        metrics, trades = fn()

        row = {
            "strategy": name,
            "total_return": metrics.get("total_return"),
            "annual_return": metrics.get("annual_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe": metrics.get("sharpe"),
            "num_trades": metrics.get("num_trades"),
        }
        rows.append(row)

        print(f"\n===== {name} =====")
        print("Metrics:", row)
        print("Trades:", len(trades))
        if trades:
            print("First trade:", trades[0])
            print("Last trade:", trades[-1])

    table = pd.DataFrame(rows)
    print("\n=== Comparison Table ===")
    print(table)


if __name__ == "__main__":
    main()
