from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from agents.data_agent import DataAgent
from agents.trend_agent import TrendAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.ml_agent import MLAgent
from agents.coordinator_agent import CoordinatorAgent
from backtest.backtest_engine import BacktestEngine


def _slice_by_yyyymmdd(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    start_dt = pd.to_datetime(start_date, format="%Y%m%d")
    end_dt = pd.to_datetime(end_date, format="%Y%m%d")
    return df.sort_index().loc[start_dt:end_dt]


def print_summary(name: str, metrics: Dict[str, Any], trades: List[Any]) -> None:
    print(f"\n===== {name} =====")
    print("Metrics:", {k: metrics[k] for k in metrics if k not in ("equity_curve", "dates")})
    print("Trades:", len(trades))
    if trades:
        print("First trade:", trades[0])
        print("Last trade:", trades[-1])


def to_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["strategy", "total_return", "annual_return", "max_drawdown", "sharpe", "num_trades"]
    return {k: metrics.get(k) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="Compare strategies with ML-veto fusion + DataAgent.")

    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--verbose", action="store_true")

    # DataAgent
    parser.add_argument("--symbol", type=str, default="600519")
    parser.add_argument("--start_date", type=str, default="20200101")
    parser.add_argument("--end_date", type=str, default="20241231")
    parser.add_argument("--adjust", type=str, default="qfq")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--force_refresh", action="store_true")

    # Engine
    parser.add_argument("--init_cash", type=float, default=100000.0)
    parser.add_argument("--fee_rate", type=float, default=0.0003)
    parser.add_argument("--slippage_bps", type=float, default=5.0)
    parser.add_argument("--max_position_pct", type=float, default=0.98)
    parser.add_argument("--min_order_value", type=float, default=1000.0)
    parser.add_argument("--lot_size", type=int, default=100)

    # Trend
    parser.add_argument("--trend_short", type=int, default=5)
    parser.add_argument("--trend_long", type=int, default=20)

    # MeanRev
    parser.add_argument("--mr_window", type=int, default=20)
    parser.add_argument("--mr_num_std", type=float, default=1.2)
    parser.add_argument("--mr_conf_scale", type=float, default=1.0)

    # ML
    parser.add_argument("--ml_lookback", type=int, default=10)
    parser.add_argument("--ml_min_train_size", type=int, default=60)
    parser.add_argument("--ml_prob_threshold", type=float, default=0.55)

    # Coordinator weights
    parser.add_argument("--w_trend", type=float, default=0.60)
    parser.add_argument("--w_mr", type=float, default=0.40)
    parser.add_argument("--w_ml", type=float, default=0.25)

    # Coordinator thresholds
    parser.add_argument("--min_edge", type=float, default=0.01)
    parser.add_argument("--min_score", type=float, default=0.01)
    parser.add_argument("--regime_boost", type=float, default=0.25)
    parser.add_argument("--min_conf_when_trade", type=float, default=0.05)

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ===== Data =====
    da = DataAgent(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        adjust=args.adjust,
        data_dir=args.data_dir,
    )
    df = da.get_feature_data(
        use_cache=(not args.no_cache),
        force_refresh=args.force_refresh,
        add_indicators=True,
    )
    df = _slice_by_yyyymmdd(df, args.start_date, args.end_date)
    print(f"Symbol={args.symbol} DF range: {df.index.min()} -> {df.index.max()} rows={len(df)}")

    # ===== Agents =====
    trend_agent = TrendAgent(args.trend_short, args.trend_long)
    mr_agent = MeanReversionAgent(args.mr_window, args.mr_num_std, args.mr_conf_scale)
    ml_agent = MLAgent(args.ml_lookback, args.ml_min_train_size, args.ml_prob_threshold)

    engine = BacktestEngine(
        init_cash=args.init_cash,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        max_position_pct=args.max_position_pct,
        min_order_value=args.min_order_value,
        lot_size=args.lot_size,
    )

    # ===== 1) Trend-only =====
    m1, t1 = engine.run_single_agent(df, trend_agent, "Trend-only", args.verbose)

    # ===== 2) MeanRev-only =====
    m2, t2 = engine.run_single_agent(df, mr_agent, "MeanRev-only", args.verbose)

    # ===== 3) Fusion(no-ML) —— ✅ veto 关闭 =====
    coordinator_no_ml = CoordinatorAgent(
        agent_weights={"trend": args.w_trend, "mean_reversion": args.w_mr},
        ml_veto_enabled=False,   # ✅ 关键改动
        min_edge=args.min_edge,
        min_score_to_trade=args.min_score,
        regime_boost=args.regime_boost,         
        min_conf_when_trade=args.min_conf_when_trade,
    )
    fusion_agents_no_ml = {"trend": trend_agent, "mean_reversion": mr_agent}
    m3, t3 = engine.run_fusion(df, fusion_agents_no_ml, coordinator_no_ml, "Fusion(no-ML)", args.verbose)

    # ===== 4) Fusion(+ML-veto) =====
    coordinator_with_ml_veto = CoordinatorAgent(
        agent_weights={"trend": args.w_trend, "mean_reversion": args.w_mr, "ml": args.w_ml},
        ml_veto_enabled=True,
        min_edge=args.min_edge,
        min_score_to_trade=args.min_score,
        regime_boost=args.regime_boost,
        min_conf_when_trade=args.min_conf_when_trade,
    )
    fusion_agents_with_ml = {"trend": trend_agent, "mean_reversion": mr_agent, "ml": ml_agent}
    m4, t4 = engine.run_fusion(
        df, fusion_agents_with_ml, coordinator_with_ml_veto, "Fusion(+ML-veto)", args.verbose
    )

    # ===== Print =====
    print_summary("Trend-only", m1, t1)
    print_summary("MeanRev-only", m2, t2)
    print_summary("Fusion(no-ML)", m3, t3)
    print_summary("Fusion(+ML-veto)", m4, t4)

    # ===== Table =====
    table = pd.DataFrame([to_row(m1), to_row(m2), to_row(m3), to_row(m4)])
    print("\n=== Comparison Table ===")
    print(table)

    table_path = outdir / "comparison_table.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {table_path}")


if __name__ == "__main__":
    main()