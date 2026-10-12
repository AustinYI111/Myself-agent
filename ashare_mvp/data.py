from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    prices: pd.DataFrame
    fundamentals: pd.DataFrame
    calendar: pd.DatetimeIndex
    metadata: dict


def _required(frame, names, table):
    absent = set(names) - set(frame.columns)
    if absent:
        raise ValueError(f"{table}: missing columns {sorted(absent)}")


def validate(data: Dataset) -> Dataset:
    """Fail closed. Dates are Shanghai session dates, not UTC timestamps."""
    p, f = data.prices.copy(), data.fundamentals.copy()
    _required(p, ["date", "symbol", "close", "total_return_open", "volume",
                  "market_cap", "is_st", "eligible", "can_buy", "can_sell", "listing_date"], "prices")
    _required(f, ["symbol", "report_date", "available_date", "roe", "earnings_ttm",
                  "book_equity", "revenue_growth", "earnings_growth", "debt_ratio"], "fundamentals")
    if p.empty:
        raise ValueError("prices must not be empty")
    for frame, cols in [(p, ["date", "listing_date"]), (f, ["report_date", "available_date"])]:
        for col in cols:
            frame[col] = pd.to_datetime(frame[col], errors="raise")
            if frame[col].dt.tz is not None:
                raise ValueError(f"{col}: require timezone-free Shanghai session dates")
            if frame[col].isna().any() or (frame[col] != frame[col].dt.normalize()).any():
                raise ValueError(f"{col}: require non-null, normalized session dates")
        if frame.symbol.isna().any():
            raise ValueError("symbol is null")
        frame["symbol"] = frame.symbol.astype(str)
        if frame.symbol.str.strip().eq("").any():
            raise ValueError("symbol is blank")
    if p.duplicated(["symbol", "date"]).any():
        raise ValueError("duplicate price key")
    if f.duplicated(["symbol", "report_date", "available_date"]).any():
        raise ValueError("ambiguous fundamental revision key")
    if (f.available_date <= f.report_date).any():
        raise ValueError("fundamentals must become available after report period end")
    if (p.date < p.listing_date).any():
        raise ValueError("price before listing")
    for col in ["close", "total_return_open", "volume", "market_cap"]:
        p[col] = pd.to_numeric(p[col], errors="raise")
        if not np.isfinite(p[col]).all() or (p[col] < 0).any():
            raise ValueError(f"invalid {col}")
    if ((p.close <= 0) & p.eligible.astype(bool)).any():
        raise ValueError("eligible close must be positive")
    for col in ["is_st", "eligible", "can_buy", "can_sell"]:
        if not p[col].isin([0, 1, False, True]).all():
            raise ValueError(f"{col}: require 0/1")
        p[col] = p[col].astype(bool)
    for col in ["roe", "earnings_ttm", "book_equity", "revenue_growth", "earnings_growth", "debt_ratio"]:
        f[col] = pd.to_numeric(f[col], errors="raise")
        if np.isinf(f[col]).any():
            raise ValueError(f"infinite {col}")
    cal = pd.DatetimeIndex(pd.to_datetime(data.calendar))
    if cal.tz is not None or len(cal) < 2 or cal.hasnans or cal.has_duplicates or not cal.is_monotonic_increasing or not cal.equals(cal.normalize()):
        raise ValueError("calendar must be ordered unique session dates")
    if not p.date.isin(cal).all():
        raise ValueError("price outside supplied exchange calendar")
    # Dense valuations are mandatory: suspensions/delistings cannot disappear.
    for symbol, rows in p.groupby("symbol"):
        expected = cal[cal >= max(rows.listing_date.min(), cal.min())]
        if not pd.DatetimeIndex(rows.date.sort_values()).equals(expected):
            raise ValueError(f"{symbol}: incomplete calendar coverage; include suspended/terminal valuations")
        if rows.listing_date.nunique() != 1:
            raise ValueError(f"{symbol}: listing_date changed")
    for key in ["source", "synthetic", "price_basis", "universe_policy", "availability_policy"]:
        if key not in data.metadata:
            raise ValueError(f"metadata missing {key}")
    if not isinstance(data.metadata["synthetic"], bool):
        raise ValueError("synthetic must be JSON boolean")
    if data.metadata["price_basis"] != "total_return_consistent":
        raise ValueError("prices must use a consistently adjusted total-return basis")
    if not set(f.symbol).issubset(set(p.symbol)):
        raise ValueError("fundamental symbol not in price universe")
    return Dataset(p.sort_values(["date", "symbol"]).reset_index(drop=True),
                   f.sort_values(["symbol", "report_date", "available_date"]).reset_index(drop=True), cal, data.metadata)


def load(directory):
    path = Path(directory)
    return validate(Dataset(
        pd.read_csv(path / "prices.csv", dtype={"symbol": str}),
        pd.read_csv(path / "fundamentals.csv", dtype={"symbol": str}),
        pd.DatetimeIndex(pd.read_csv(path / "calendar.csv")["date"]),
        json.loads((path / "metadata.json").read_text(encoding="utf-8")),
    ))


def fundamentals_asof(data, date):
    # Select newest known period, then its newest known revision. A correction
    # to an older period must never overwrite a newer period.
    known = data.fundamentals[data.fundamentals.available_date <= date]
    return known.sort_values(["symbol", "report_date", "available_date"]).groupby("symbol").tail(1)


def month_ends(calendar):
    series = pd.Series(calendar, index=calendar)
    # Drop an incomplete final calendar month (caller may score latest separately).
    ends = series.groupby(calendar.to_period("M")).max()
    return pd.DatetimeIndex(ends.iloc[:-1])
