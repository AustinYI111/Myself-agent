"""Seeded synthetic data: infrastructure validation, never investment evidence."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .data import Dataset, validate


def make_demo(seed=42, stocks=40, periods=1512):
    rng = np.random.default_rng(seed)
    calendar = pd.bdate_range("2018-01-02", periods=periods)  # mock calendar, NOT SSE sessions
    prices, fundamentals = [], []
    market = rng.normal(0.0001, 0.009, periods)
    for i in range(stocks):
        symbol = f"SYN{i:04d}"
        quality = rng.normal()
        returns = market + rng.normal(0.00012 * quality, 0.012, periods)
        close = (20 + i) * np.exp(np.cumsum(returns))
        opens = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.001, periods))
        prices.append(pd.DataFrame(dict(date=calendar, symbol=symbol, close=close,
                                       total_return_open=opens, volume=rng.lognormal(13, 0.3, periods),
                                       market_cap=close * 1e8, is_st=False, eligible=True,
                                       can_buy=True, can_sell=True, listing_date=pd.Timestamp("2010-01-01"))))
        for report in pd.date_range("2017-09-30", calendar[-1], freq="QE"):
            fundamentals.append(dict(symbol=symbol, report_date=report, available_date=report + pd.Timedelta(days=55),
                                     roe=0.12 + 0.02 * quality + rng.normal(0, 0.01),
                                     earnings_ttm=1e8 * (1 + 0.2 * quality), book_equity=8e8,
                                     revenue_growth=0.1 + quality * 0.03,
                                     earnings_growth=0.12 + quality * 0.04 + rng.normal(0, 0.03),
                                     debt_ratio=np.clip(0.4 - quality * 0.05, 0, 1)))
    return validate(Dataset(pd.concat(prices, ignore_index=True), pd.DataFrame(fundamentals), calendar,
                            dict(source="seeded synthetic generator", synthetic=True,
                                 price_basis="total_return_consistent", universe_policy="fixed synthetic universe",
                                 availability_policy="mock report end +55 calendar days; mock weekday calendar")))


def save(data, directory):
    path = Path(directory)
    if path.exists() and any(path.iterdir()):
        raise ValueError("demo data directory must be empty")
    path.mkdir(parents=True, exist_ok=True)
    data.prices.to_csv(path / "prices.csv", index=False)
    data.fundamentals.to_csv(path / "fundamentals.csv", index=False)
    pd.DataFrame({"date": data.calendar}).to_csv(path / "calendar.csv", index=False)
    (path / "metadata.json").write_text(json.dumps(data.metadata, indent=2), encoding="utf-8")
