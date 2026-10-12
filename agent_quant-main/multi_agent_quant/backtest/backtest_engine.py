# backtest/backtest_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable, Tuple

import math
import numpy as np
import pandas as pd


@dataclass
class Trade:
    dt: pd.Timestamp
    action: str          # "buy" or "sell"
    price: float
    size: float          # shares
    fee: float
    slippage_cost: float
    cash_after: float
    position_after: float
    meta: Optional[Dict[str, Any]] = None


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def detect_market_state(row: pd.Series, dist_th: float = 0.02, vol_th: float = 0.03) -> str:
    """
    轻量级市场状态识别：
    - high_vol: 波动率过高(优先级最高)
    - trend: 价格明显偏离 MA20
    - range: 价格围绕 MA20
    需要 row 至少包含: close, ma_20 (可选: vol_20)
    """
    close = _safe_float(row.get("close", None), default=np.nan)
    ma20 = _safe_float(row.get("ma_20", None), default=close)

    if np.isnan(close) or np.isnan(ma20) or ma20 == 0:
        return "range"

    dist = abs(close - ma20) / abs(ma20)

    vol20 = row.get("vol_20", None)
    if vol20 is not None and not (isinstance(vol20, float) and math.isnan(vol20)):
        vol20 = _safe_float(vol20, default=0.0)
        if vol20 >= vol_th:
            return "high_vol"

    if dist >= dist_th:
        return "trend"
    return "range"


class BacktestEngine:
    """
    轻量级回测引擎（只做多）：
    - 每个 bar 调用 signal_fn(row, ctx)-> {signal, confidence, meta}
    - 下单规则：
        buy：开仓/加仓（✅分批调仓，按 confidence 目标仓位补齐）
        sell：平仓（一次全平）
    - 支持：佣金、滑点、最大仓位限制、最小下单金额
    - 输出：交易列表 + 指标（total_return/annual_return/max_drawdown/sharpe/num_trades）
    - 返回 metrics 中包含：equity_curve, dates（便于画图）

    ✅ 支持 Fusion 场景的在线反馈学习（next day return）：
      - run_fusion 的 _signal_fn 把 agent_outputs 塞进 ctx["_agent_outputs"]，把 coordinator 塞进 ctx["coord"]
      - _run_with_index_context 每根 bar（除最后一根）用 next_ret 调 coordinator.update_performance(...)
      - 成交后：若 coordinator 支持 on_trade_executed()，自动调用
    """

    def __init__(
        self,
        init_cash: float = 100000.0,
        fee_rate: float = 0.0003,         # 3bp
        slippage_bps: float = 5.0,        # 5 bps
        max_position_pct: float = 0.98,   # 最多用掉 98% 现金
        min_order_value: float = 1000.0,  # 最小下单金额
        lot_size: int = 100,              # A股常见一手 100 股（演示用）
        allow_fractional: bool = False,   # True 则不强制 lot_size
        price_col: str = "close",
    ) -> None:
        self.init_cash = float(init_cash)
        self.fee_rate = float(fee_rate)
        self.slippage_bps = float(slippage_bps)
        self.max_position_pct = float(max_position_pct)
        self.min_order_value = float(min_order_value)
        self.lot_size = int(lot_size)
        self.allow_fractional = bool(allow_fractional)
        self.price_col = price_col

    # --------------------------
    # Public APIs
    # --------------------------
    def run(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.Series, Dict[str, Any]], Dict[str, Any]],
        strategy_name: str = "Strategy",
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], List[Trade]]:
        df = self._prepare_df(df)
        return self._run_with_index_context(df, signal_fn, strategy_name=strategy_name, verbose=verbose)

    def run_single_agent(
        self,
        df: pd.DataFrame,
        agent,
        strategy_name: str,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], List[Trade]]:
        """
        agent: 需要有 generate_signal(df_slice)->dict
        每个 bar 传当前到 i 的窗口（需要 df 有足够历史）
        """
        df = self._prepare_df(df)

        def _signal_fn(row: pd.Series, ctx: Dict[str, Any]) -> Dict[str, Any]:
            i = ctx.get("_i", 0)
            window_df = df.iloc[: i + 1]  # ✅累积历史，ML 才能训练
            return agent.generate_signal(window_df)

        return self._run_with_index_context(df, _signal_fn, strategy_name=strategy_name, verbose=verbose)

    def run_fusion(
        self,
        df: pd.DataFrame,
        agents: Dict[str, Any],
        coordinator,
        strategy_name: str,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], List[Trade]]:
        df = self._prepare_df(df)

        def _signal_fn(row: pd.Series, ctx: Dict[str, Any]) -> Dict[str, Any]:
            ctx["coord"] = coordinator

            i = ctx.get("_i", 0)
            window_df = df.iloc[: i + 1]  # ✅累积历史，ML 才能训练

            agent_outputs: Dict[str, Dict[str, Any]] = {}
            for name, ag in agents.items():
                agent_outputs[name] = ag.generate_signal(window_df)

            if verbose:
                dt_str = str(row.name.date()) if hasattr(row.name, "date") else str(row.name)[:10]
                for name, out in agent_outputs.items():
                    ag_sig = out.get("signal", "hold")
                    ag_conf = float(out.get("confidence", 0.0))
                    print(f"  [{dt_str}] AGENT {name:20s}: signal={ag_sig:<6} conf={ag_conf:.4f}")

            ctx["_agent_outputs"] = agent_outputs

            market_state = ctx.get("market_state", "range")
            try:
                fused = coordinator.aggregate(agent_outputs, market_state=market_state)
            except TypeError:
                fused = coordinator.aggregate(agent_outputs)

            if verbose:
                dt_str = str(row.name.date()) if hasattr(row.name, "date") else str(row.name)[:10]
                fused_meta = fused.get("meta", {}) or {}
                score = fused_meta.get("score", {})
                edge = fused_meta.get("edge", 0.0)
                veto = fused_meta.get("ml_veto", {})
                fused_sig = fused.get("signal", "hold")
                fused_conf = float(fused.get("confidence", 0.0))
                print(
                    f"  [{dt_str}] COORDINATOR: sig={fused_sig:<6} conf={fused_conf:.4f} "
                    f"score_buy={score.get('buy', 0.0):.4f} score_sell={score.get('sell', 0.0):.4f} "
                    f"edge={edge:.4f} veto={veto.get('applied', False)} market={market_state}"
                )

            fused = dict(fused or {})
            fused.setdefault("meta", {})
            fused["meta"] = dict(fused["meta"])
            fused["meta"]["market_state"] = market_state
            fused["meta"]["agent_outputs"] = agent_outputs
            return fused

        return self._run_with_index_context(
            df,
            _signal_fn,
            strategy_name=strategy_name,
            verbose=verbose,
            coordinator=coordinator,
        )

    # --------------------------
    # Internals
    # --------------------------
    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ 关键改动：即便输入只有 close，也自动补齐一套“通用特征”
        这样 MLAgent 在 demo 数据上也不至于一直 insufficient_data/feature_missing
        """
        if df is None or len(df) == 0:
            raise ValueError("df is empty")

        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                raise ValueError("df needs DatetimeIndex or a 'date' column")

        if self.price_col not in df.columns:
            raise ValueError(f"df must contain '{self.price_col}' column")

        # 保证 close 数值
        df[self.price_col] = pd.to_numeric(df[self.price_col], errors="coerce")

        # === 基础收益率 ===
        if "return" not in df.columns:
            df["return"] = df[self.price_col].pct_change()

        # === 均线 ===
        if "ma_5" not in df.columns:
            df["ma_5"] = df[self.price_col].rolling(5).mean()
        if "ma_10" not in df.columns:
            df["ma_10"] = df[self.price_col].rolling(10).mean()
        if "ma_20" not in df.columns:
            df["ma_20"] = df[self.price_col].rolling(20).mean()

        # === 波动率 ===
        if "vol_20" not in df.columns:
            df["vol_20"] = df["return"].rolling(20).std()

        # === RSI(14) ===
        if "rsi" not in df.columns:
            delta = df[self.price_col].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

        # === Bollinger(20,2)（给 ML/分析用，不影响 MeanRev Agent 自己计算）===
        if "boll_upper" not in df.columns or "boll_lower" not in df.columns:
            m = df[self.price_col].rolling(20).mean()
            s = df[self.price_col].rolling(20).std()
            df["boll_upper"] = m + 2 * s
            df["boll_lower"] = m - 2 * s

        return df

    def _run_with_index_context(
        self,
        df: pd.DataFrame,
        signal_fn: Callable[[pd.Series, Dict[str, Any]], Dict[str, Any]],
        strategy_name: str,
        verbose: bool,
        coordinator=None,
    ) -> Tuple[Dict[str, Any], List[Trade]]:
        cash = self.init_cash
        position = 0.0
        trades: List[Trade] = []
        equity_curve: List[float] = []
        dates: List[pd.Timestamp] = []

        if verbose:
            print(
                f"[{strategy_name}] Data range: {df.index[0].date()} ~ {df.index[-1].date()} "
                f"| {len(df)} rows"
            )
            null_counts = df.isnull().sum()
            missing = null_counts[null_counts > 0]
            if len(missing) > 0:
                print(f"[{strategy_name}] WARNING: missing values detected — {missing.to_dict()}")

        ctx: Dict[str, Any] = {
            "cash": cash,
            "position": position,
            "strategy": strategy_name,
        }
        if coordinator is not None:
            ctx["_coordinator"] = coordinator

        prev_price_for_feedback: Optional[float] = None
        prev_agent_outputs_for_feedback: Optional[Dict[str, Dict[str, Any]]] = None
        bar_count = 0

        for i, (dt, row) in enumerate(df.iterrows()):
            ctx["_i"] = i

            price = _safe_float(row[self.price_col], default=np.nan)
            if np.isnan(price) or price <= 0:
                equity_curve.append(cash + position * 0.0)
                dates.append(dt)
                continue

            market_state = detect_market_state(row)
            ctx["market_state"] = market_state

            out = signal_fn(row, ctx) or {}
            signal = str(out.get("signal", "hold")).lower()
            confidence = float(out.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            meta = out.get("meta", {}) or {}
            meta = dict(meta)
            meta["market_state"] = market_state

            agent_outputs_meta = meta.get("agent_outputs", None)
            if isinstance(agent_outputs_meta, dict):
                prev_agent_outputs_for_feedback = agent_outputs_meta

            tr: Optional[Trade] = None

            if verbose and signal != "hold":
                print(f"[{dt.date()}] SIGNAL {signal.upper():<5} conf={confidence:.4f} market={market_state}")

            if signal == "buy":
                cash, position, tr, skip_reason = self._execute_buy(dt, price, cash, position, confidence, meta)
                if tr:
                    trades.append(tr)
                    if verbose:
                        print(f"[{dt.date()}] BUY  price={price:.4f} size={tr.size} cash={cash:.2f} pos={position}")
                elif verbose:
                    print(f"[{dt.date()}] BUY SKIPPED: {skip_reason}")
            elif signal == "sell":
                cash, position, tr = self._execute_sell(dt, price, cash, position, confidence, meta)
                if tr:
                    trades.append(tr)
                    if verbose:
                        print(f"[{dt.date()}] SELL price={price:.4f} size={tr.size} cash={cash:.2f} pos={position}")
                elif verbose:
                    print(f"[{dt.date()}] SELL SKIPPED: no position to sell")

            if tr is not None and coordinator is not None and hasattr(coordinator, "on_trade_executed"):
                try:
                    coordinator.on_trade_executed()
                except Exception:
                    pass

            ctx["cash"] = cash
            ctx["position"] = position

            # ====== 动态权重：用“下一天收益”给 coordinator 做在线更新 ======
            coordinator2 = ctx.get("coord", None) or coordinator
            agent_outputs2 = ctx.get("_agent_outputs", None)

            if (
                coordinator2 is not None
                and agent_outputs2 is not None
                and hasattr(coordinator2, "update_performance")
                and i < (len(df) - 1)
            ):
                price_next = _safe_float(df.iloc[i + 1][self.price_col], default=np.nan)
                if (not np.isnan(price_next)) and price > 0:
                    next_ret = (price_next / price) - 1.0
                    try:
                        # ✅统一用 next_ret（你的 Coordinator 定义是 next_ret）
                        coordinator2.update_performance(agent_outputs=agent_outputs2, next_ret=float(next_ret))
                    except TypeError:
                        # 兼容旧签名
                        try:
                            coordinator2.update_performance(agent_outputs2, float(next_ret))
                        except Exception:
                            pass
                    except Exception:
                        pass

            equity = cash + position * price
            equity_curve.append(equity)
            dates.append(dt)
            bar_count += 1

            if verbose and bar_count % 50 == 0:
                print(f"[{dt.date()}] EQUITY SNAPSHOT (bar {bar_count}): equity={equity:.2f} cash={cash:.2f} pos={position:.4f}")

            # 旧反馈逻辑（保留兼容）
            if ctx.get("_agent_outputs", None) is None:
                if (
                    coordinator is not None
                    and hasattr(coordinator, "update_performance")
                    and prev_price_for_feedback is not None
                    and prev_agent_outputs_for_feedback is not None
                ):
                    realized_ret = price / max(prev_price_for_feedback, 1e-12) - 1.0
                    try:
                        coordinator.update_performance(
                            agent_outputs=prev_agent_outputs_for_feedback,
                            next_ret=float(realized_ret),
                        )
                    except Exception:
                        pass

            prev_price_for_feedback = price

        metrics = self._compute_metrics(dates, equity_curve, trades)
        metrics["strategy"] = strategy_name
        metrics["equity_curve"] = equity_curve
        metrics["dates"] = dates

        if verbose:
            print(
                f"[{strategy_name}] DONE: trades={metrics['num_trades']} "
                f"total_return={metrics['total_return']:.4f} "
                f"annual_return={metrics['annual_return']:.4f} "
                f"max_drawdown={metrics['max_drawdown']:.4f} "
                f"sharpe={metrics['sharpe']:.4f}"
            )

        return metrics, trades

    def _execute_buy(
        self,
        dt: pd.Timestamp,
        price: float,
        cash: float,
        position: float,
        confidence: float,
        meta: Dict[str, Any],
    ) -> Tuple[float, float, Optional[Trade], Optional[str]]:
        target_value = self.init_cash * self.max_position_pct * confidence
        current_value = position * price

        if current_value >= target_value:
            return cash, position, None, f"already at target (curr={current_value:.2f} >= target={target_value:.2f})"

        need_value = target_value - current_value
        if need_value < self.min_order_value:
            return cash, position, None, f"need_value={need_value:.2f} < min_order_value={self.min_order_value}"

        exec_price = price * (1.0 + self.slippage_bps / 10000.0)
        shares = need_value / exec_price

        if not self.allow_fractional:
            shares = (shares // self.lot_size) * self.lot_size

        if shares <= 0:
            return cash, position, None, "shares rounded to 0 (lot_size constraint)"

        gross = shares * exec_price
        fee = gross * self.fee_rate
        slippage_cost = shares * (exec_price - price)
        total_cost = gross + fee

        if total_cost > cash:
            shares = cash / (exec_price * (1.0 + self.fee_rate))
            if not self.allow_fractional:
                shares = (shares // self.lot_size) * self.lot_size
            if shares <= 0:
                return cash, position, None, f"insufficient cash (cash={cash:.2f} need={total_cost:.2f})"

            gross = shares * exec_price
            fee = gross * self.fee_rate
            slippage_cost = shares * (exec_price - price)
            total_cost = gross + fee

        cash_after = cash - total_cost
        position_after = position + shares

        tr = Trade(
            dt=dt,
            action="buy",
            price=float(exec_price),
            size=float(shares),
            fee=float(fee),
            slippage_cost=float(slippage_cost),
            cash_after=float(cash_after),
            position_after=float(position_after),
            meta=meta,
        )
        return cash_after, position_after, tr, None

    def _execute_sell(
        self,
        dt: pd.Timestamp,
        price: float,
        cash: float,
        position: float,
        confidence: float,
        meta: Dict[str, Any],
    ) -> Tuple[float, float, Optional[Trade]]:
        if position <= 0:
            return cash, position, None

        shares = position
        exec_price = price * (1.0 - self.slippage_bps / 10000.0)
        gross = shares * exec_price
        fee = gross * self.fee_rate
        slippage_cost = shares * (price - exec_price)

        cash_after = cash + gross - fee
        position_after = 0.0

        tr = Trade(
            dt=dt,
            action="sell",
            price=float(exec_price),
            size=float(shares),
            fee=float(fee),
            slippage_cost=float(slippage_cost),
            cash_after=float(cash_after),
            position_after=float(position_after),
            meta=meta,
        )
        return cash_after, position_after, tr

    def _compute_metrics(
        self,
        dates: List[pd.Timestamp],
        equity_curve: List[float],
        trades: List[Trade],
    ) -> Dict[str, Any]:
        eq = np.array(equity_curve, dtype=float)
        if len(eq) == 0:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "num_trades": 0,
            }

        total_return = eq[-1] / eq[0] - 1.0 if eq[0] != 0 else 0.0

        n = len(eq)
        annual_return = (1.0 + total_return) ** (252.0 / max(1.0, n - 1)) - 1.0

        peak = np.maximum.accumulate(eq)
        dd = (eq / peak) - 1.0
        max_drawdown = float(dd.min()) if len(dd) else 0.0

        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        if len(rets) < 2 or np.std(rets) == 0:
            sharpe = 0.0
        else:
            sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252.0))

        return {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "max_drawdown": float(max_drawdown),
            "sharpe": float(sharpe),
            "num_trades": int(len(trades)),
        }