from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import fundamentals_asof


@dataclass(frozen=True)
class Factor:
    name: str
    family: str
    compute: Callable


class FactorLibrary:
    """Extend by registering a function of the historical snapshot only."""
    def __init__(self):
        self.factors = {}

    def register(self, name, family, compute):
        if name in self.factors:
            raise ValueError(f"duplicate factor {name}")
        self.factors[name] = Factor(name, family, compute)
        return self

    def compute(self, snapshot, oneil=False):
        result = pd.DataFrame(index=snapshot.index)
        for name, factor in self.factors.items():
            if factor.family == "oneil" and not oneil:
                continue
            raw = factor.compute(snapshot).replace([np.inf, -np.inf], np.nan)
            # Same-date cross-sectional transforms need no future fitting.
            lo, hi = raw.quantile([0.01, 0.99])
            raw = raw.clip(lo, hi)
            std = raw.std(ddof=0)
            result[name] = (raw - raw.mean()) / std if std > 0 else raw * 0
        return result


def default_library():
    lib = FactorLibrary()
    for name in ["roe", "revenue_growth", "earnings_growth"]:
        lib.register(name, "fundamental", lambda x, n=name: x[n])
    lib.register("low_debt", "fundamental", lambda x: -x.debt_ratio)
    lib.register("earnings_yield", "valuation", lambda x: x.earnings_ttm / x.market_cap)
    lib.register("book_yield", "valuation", lambda x: x.book_equity / x.market_cap)
    for name in ["momentum_63", "momentum_126", "momentum_252_skip21"]:
        lib.register(name, "momentum", lambda x, n=name: x[n])
    lib.register("low_volatility", "risk", lambda x: -x.volatility_63)
    lib.register("near_high_252", "oneil", lambda x: x.close / x.high_252)
    lib.register("volume_expansion", "oneil", lambda x: x.volume / x.volume_ma50)
    lib.register("growth_acceleration", "oneil", lambda x: x.earnings_growth - x.revenue_growth)
    return lib


def technical_panel(data):
    p = data.prices.sort_values(["symbol", "date"]).copy()
    g = p.groupby("symbol", sort=False)
    for days in [63, 126]:
        p[f"momentum_{days}"] = g.close.pct_change(days, fill_method=None)
    p["momentum_252_skip21"] = g.close.shift(21) / g.close.shift(252) - 1
    p["volatility_63"] = g.close.transform(lambda x: x.pct_change(fill_method=None).rolling(63).std())
    p["high_252"] = g.close.transform(lambda x: x.rolling(252).max())
    p["volume_ma50"] = g.volume.transform(lambda x: x.rolling(50).mean())
    return p


def snapshot(data, panel, date, min_listing_days=365, max_fundamental_age=550):
    p = panel[panel.date == date].copy()
    p = p[p.eligible & ~p.is_st & (p.volume > 0) & (p.market_cap > 0)
          & ((date - p.listing_date).dt.days >= min_listing_days)]
    p = p.merge(fundamentals_asof(data, date), on="symbol", how="inner", validate="one_to_one")
    p = p[(date - p.report_date).dt.days <= max_fundamental_age]
    required = ["roe", "earnings_ttm", "book_equity", "revenue_growth", "earnings_growth", "debt_ratio",
                "momentum_63", "momentum_126", "momentum_252_skip21", "volatility_63"]
    return p.dropna(subset=required).set_index("symbol").sort_index()
