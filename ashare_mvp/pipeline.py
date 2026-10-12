from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from .data import validate, month_ends
from .factors import default_library, technical_panel, snapshot
from .model import Config, training_slice, fit_ranker


def forward_labels(data, dates, horizons=(3, 6, 12)):
    """Calendar-month holding periods, next-session open entry; no shortening."""
    opens = data.prices.pivot(index="date", columns="symbol", values="total_return_open")
    rows = []
    for date in dates:
        next_pos = data.calendar.searchsorted(date, side="right")
        if next_pos >= len(data.calendar):
            continue
        entry = data.calendar[next_pos]
        for horizon in horizons:
            exit_pos = data.calendar.searchsorted(entry + pd.DateOffset(months=horizon))
            end = data.calendar[exit_pos] if exit_pos < len(data.calendar) else pd.NaT
            for symbol in opens.columns:
                price = opens.at[entry, symbol]
                ret = (opens.at[end, symbol] / price - 1) if pd.notna(end) and price > 0 else np.nan
                rows.append(dict(date=date, symbol=symbol, horizon=horizon,
                                 entry_date=entry, label_end=end, target=ret))
    return pd.DataFrame(rows, columns=["date", "symbol", "horizon", "entry_date", "label_end", "target"])


def build_samples(data, config, dates=None, library=None):
    library = library or default_library()
    dates = month_ends(data.calendar) if dates is None else pd.DatetimeIndex(dates)
    panel = technical_panel(data)
    frames = []
    for date in dates:
        snap = snapshot(data, panel, date)
        if snap.empty:
            continue
        factors = library.compute(snap, config.oneil)
        factors["date"] = date
        factors["available_date"] = snap.available_date
        frames.append(factors.rename_axis("symbol").reset_index())
    if not frames:
        raise ValueError("no eligible factor snapshots; need >=252 sessions and PIT fundamentals")
    samples = pd.concat(frames, ignore_index=True)
    features = [n for n, factor in library.factors.items() if config.oneil or factor.family != "oneil"]
    labels = forward_labels(data, dates)
    training_labels = labels[labels.horizon == config.train_horizon].drop(columns="horizon")
    return samples.merge(training_labels, on=["date", "symbol"], how="left", validate="one_to_one"), labels, features


def rolling_rank(samples, features, calendar, config, model_dir=None):
    selections, audit, importance = [], [], []
    for date, current in samples.groupby("date", sort=True):
        train, cutoff = training_slice(samples, date, calendar, config)
        groups = train.date.nunique()
        reason = ""
        if groups < config.min_train_groups:
            reason = "insufficient_mature_training_groups"
        elif len(current) < max(config.top_k, config.min_group_size):
            reason = "insufficient_eligible_universe"
        audit.append(dict(date=date, cutoff=cutoff, train_rows=len(train), train_groups=groups,
                          train_start=train.date.min(), train_end=train.date.max(),
                          max_label_end=train.label_end.max(), candidates=len(current),
                          status=reason or "trained"))
        if reason:
            continue
        if not (train.label_end < cutoff).all() or not (train.available_date <= train.date).all():
            raise AssertionError("point-in-time training invariant broken")
        model = fit_ranker(train, features, config)
        if model_dir is not None:
            path = Path(model_dir)
            path.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(path / f"ranker_{date:%Y%m%d}.txt"))
        ranked = current[["date", "symbol", "available_date"]].copy()
        ranked["score"] = model.predict(current[features])
        ranked = ranked.sort_values(["score", "symbol"], ascending=[False, True]).head(config.top_k)
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["target_weight"] = 1.0 / config.top_k
        selections.append(ranked)
        importance.extend(dict(date=date, factor=f, gain=float(v))
                          for f, v in zip(features, model.feature_importances_))
    picks = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(
        columns=["date", "symbol", "available_date", "score", "rank", "target_weight"])
    return picks, pd.DataFrame(audit), pd.DataFrame(importance, columns=["date", "factor", "gain"])


def cohort_performance(picks, labels, samples, data, config):
    """Overlapping buy-and-hold cohorts, NOT a compounded NAV series."""
    rows = []
    buy_rate = (config.commission_bps + config.slippage_bps) / 10000
    sell_rate = buy_rate + config.sell_tax_bps / 10000
    for date, selected in picks.groupby("date", sort=True):
        eligible = samples.loc[samples.date == date, "symbol"]
        for h in [3, 6, 12]:
            all_labels = labels[(labels.date == date) & (labels.horizon == h)].set_index("symbol")
            s = all_labels.reindex(selected.symbol)
            bench = all_labels.reindex(eligible)
            complete = len(s) == config.top_k and s.target.notna().all()
            benchmark_complete = len(bench) > 0 and bench.target.notna().all()
            gross = float(s.target.mean()) if complete else np.nan
            benchmark = float(bench.target.mean()) if benchmark_complete else np.nan
            executable = False
            if complete:
                entry, end = s.entry_date.iloc[0], s.label_end.iloc[0]
                start_flags = data.prices[data.prices.date == entry].set_index("symbol").reindex(selected.symbol)
                end_flags = data.prices[data.prices.date == end].set_index("symbol").reindex(selected.symbol)
                executable = bool(start_flags.can_buy.all() and end_flags.can_sell.all())
            rows.append(dict(date=date, horizon_months=h, selected=len(s), observed=int(s.target.notna().sum()),
                             complete=complete, executable_at_endpoints=executable,
                             gross_return=gross, indicative_net_return=(1 + gross) * (1 - sell_rate) / (1 + buy_rate) - 1,
                             universe_return=benchmark, excess_return=gross - benchmark))
    return pd.DataFrame(rows, columns=["date", "horizon_months", "selected", "observed", "complete",
                                       "executable_at_endpoints", "gross_return", "indicative_net_return",
                                       "universe_return", "excess_return"])


def simulate_monthly(data, picks, config):
    """Fractional adjusted shares: economic research NAV, with blocked orders.

    Uses contemporaneous execution flags only; blocked sales remain held,
    blocked buys stay in cash. No substitution using future returns.
    """
    if picks.empty:
        return pd.DataFrame(), pd.DataFrame()
    orders = {}
    for date, selected in picks.groupby("date"):
        idx = data.calendar.searchsorted(date, side="right")
        if idx < len(data.calendar):
            orders[data.calendar[idx]] = selected.set_index("symbol").target_weight.to_dict()
    if not orders:
        return pd.DataFrame(), pd.DataFrame()
    cash, holdings = 1.0, {}
    nav, trades = [], []
    buy_rate = (config.commission_bps + config.slippage_bps) / 10000
    sell_rate = buy_rate + config.sell_tax_bps / 10000
    for date, daily in data.prices.groupby("date", sort=True):
        if date < min(orders):
            continue
        bars = daily.set_index("symbol")
        prices = bars.total_return_open
        equity = cash + sum(q * prices[s] for s, q in holdings.items())
        if date in orders:
            targets = orders[date]
            for symbol in sorted(holdings):
                price = prices[symbol]
                value = holdings[symbol] * price
                sale = max(0.0, value - targets.get(symbol, 0.0) * equity)
                if sale > 1e-12:
                    if bars.at[symbol, "can_sell"] and price > 0:
                        holdings[symbol] -= sale / price
                        cash += sale * (1 - sell_rate)
                        trades.append(dict(date=date, symbol=symbol, side="sell", notional=sale,
                                           cost=sale * sell_rate, status="filled"))
                    else:
                        trades.append(dict(date=date, symbol=symbol, side="sell", notional=sale,
                                           cost=0.0, status="blocked"))
            buys = {}
            for symbol, weight in targets.items():
                price = prices[symbol]
                amount = max(0.0, weight * equity - holdings.get(symbol, 0.0) * price)
                if amount > 1e-12:
                    if bars.at[symbol, "can_buy"] and price > 0:
                        buys[symbol] = amount
                    else:
                        trades.append(dict(date=date, symbol=symbol, side="buy", notional=amount,
                                           cost=0.0, status="blocked"))
            scale = min(1.0, cash / (sum(buys.values()) * (1 + buy_rate))) if buys else 0.0
            for symbol, amount in buys.items():
                amount *= scale
                holdings[symbol] = holdings.get(symbol, 0.0) + amount / prices[symbol]
                cash -= amount * (1 + buy_rate)
                trades.append(dict(date=date, symbol=symbol, side="buy", notional=amount,
                                   cost=amount * buy_rate, status="filled"))
        if cash < -1e-9:
            raise AssertionError("negative cash")
        cash = max(cash, 0.0)  # remove floating-point subtraction residue
        equity = cash + sum(q * prices[s] for s, q in holdings.items())
        nav.append(dict(date=date, nav=equity, cash=cash, positions=sum(q > 1e-12 for q in holdings.values())))
    return pd.DataFrame(nav), pd.DataFrame(trades)


def nav_metrics(nav):
    if nav.empty:
        return {"sessions": 0}
    if not np.isfinite(nav.nav).all() or (nav.nav < 0).any():
        raise ValueError("NAV must be finite and nonnegative")
    values = np.r_[1.0, nav.nav.to_numpy()]
    # Returns after bankruptcy have no defined denominator; preserve total loss
    # and drawdown, but do not fabricate a Sharpe ratio or emit JSON NaN.
    returns = np.divide(values[1:], values[:-1], out=np.full(len(values) - 1, np.nan), where=values[:-1] > 0) - 1
    years = max((nav.date.iloc[-1] - nav.date.iloc[0]).days / 365.25, 1 / 365.25)
    std = returns.std(ddof=1) if len(returns) > 1 else 0
    return dict(sessions=len(nav), total_return=float(values[-1] - 1),
                annual_return=float(values[-1] ** (1 / years) - 1),
                max_drawdown=float((values / np.maximum.accumulate(values) - 1).min()),
                sharpe=float(returns.mean() / std * np.sqrt(252)) if np.isfinite(returns).all() and std > 0 else None)


def _finite_mean(series):
    value = series.mean()
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def run(data, out, config=Config(), asof=None):
    data = validate(data)
    path = Path(out)
    if path.exists() and any(path.iterdir()):
        raise ValueError("output directory is not empty; choose a new run directory")
    asof = pd.Timestamp(asof) if asof is not None else data.calendar[-1]
    if asof not in data.calendar:
        raise ValueError("asof must be an observed exchange session")
    # The exchange calendar may be known in advance, but outcomes must not be.
    completed_months = month_ends(data.calendar)
    completed_months = completed_months[completed_months <= asof]
    data = type(data)(data.prices[data.prices.date <= asof].copy(),
                      data.fundamentals[data.fundamentals.available_date <= asof].copy(),
                      data.calendar[data.calendar <= asof], data.metadata)
    dates = completed_months.union(pd.DatetimeIndex([asof]))
    path.mkdir(parents=True, exist_ok=True)
    samples, labels, features = build_samples(data, config, dates)
    picks, audit, importance = rolling_rank(samples, features, data.calendar, config, path / "models")
    # The arbitrary asof score is published separately; rebalance only month ends.
    monthly = picks[picks.date.isin(completed_months)]
    cohort = cohort_performance(monthly, labels, samples, data, config)
    nav, trades = simulate_monthly(data, monthly, config)
    for name, frame in [("monthly_top20", monthly), ("latest_top20", picks[picks.date == asof]),
                        ("training_audit", audit), ("cohort_performance", cohort),
                        ("daily_nav", nav), ("orders", trades), ("factor_importance", importance)]:
        frame.to_csv(path / f"{name}.csv", index=False)
    summary = {"data": data.metadata, "config": asdict(config), "asof": str(asof.date()),
               "features": features, "monthly_predictions": int(monthly.date.nunique()),
               "latest_selection_count": int((picks.date == asof).sum()), "nav": nav_metrics(nav),
               "cohorts": {str(h): {"complete_cohorts": int(g.complete.sum()),
                                     "mean_gross_return": _finite_mean(g.gross_return),
                                     "mean_excess_return": _finite_mean(g.excess_return)}
                           for h, g in cohort.groupby("horizon_months")},
               "limitations": ["Fractional adjusted shares; no lot rounding/minimum commission/T+1 order engine",
                               "Constant cost assumptions; supply historical dated fees for production",
                               "Overlapping cohorts are not independent and must not be compounded",
                               "Universe and availability provenance are provider assertions requiring external audit"]}
    hashes = {}
    for name, frame in [("prices", data.prices), ("fundamentals", data.fundamentals)]:
        hashes[name] = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    hashes["calendar"] = hashlib.sha256(str(list(data.calendar)).encode()).hexdigest()
    summary["input_hashes"] = hashes
    (path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
